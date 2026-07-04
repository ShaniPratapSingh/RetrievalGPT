import re
import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
from src.core.observability import Logger, telemetry

logger = Logger("retriever")

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


def clean_tokenize(text: str) -> List[str]:
    """Helper to tokenize and lowercase text for BM25."""
    return re.findall(r'\w+', text.lower())


class HybridRetriever:
    def __init__(self, storage_manager, embedding_fn, local_reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.storage = storage_manager
        self.get_embedding = embedding_fn
        self.local_reranker_name = local_reranker_name
        self.bm25 = None
        self.bm25_chunks = []
        self.reranker = None
        self.reranker_loaded = False
        
        # Configurable RRF parameter (default to standard 60)
        self.rrf_k = int(os.getenv("RAG_RRF_K", 60))
        # Configurable fusion weights: dense weight vs sparse weight
        self.dense_weight = float(os.getenv("RAG_DENSE_WEIGHT", 1.0))
        self.sparse_weight = float(os.getenv("RAG_SPARSE_WEIGHT", 1.0))
                
    def rebuild_sparse_index(self):
        """Fit BM25 on all current chunks from storage."""
        telemetry.start_span("rebuild_sparse_index")
        chunks = self.storage.get_all_chunks()
        if not chunks:
            self.bm25 = None
            self.bm25_chunks = []
            telemetry.end_span("rebuild_sparse_index")
            return
            
        corpus = [clean_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.bm25_chunks = chunks
        telemetry.end_span("rebuild_sparse_index")
        logger.info("Sparse BM25 index rebuilt", docs_count=len(chunks))

    def retrieve_sparse(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using BM25."""
        if not self.bm25 or not self.bm25_chunks:
            return []
            
        tokenized_query = clean_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            if score > 0:
                results.append({
                    "chunk": self.bm25_chunks[idx],
                    "score": float(score)
                })
        return results

    def retrieve_dense(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using dense vector embeddings."""
        query_emb = self.get_embedding(query)
        
        if self.storage.db_enabled:
            return self.storage.query_similarity(query_emb, top_k=top_k)
            
        chunks = self.storage.get_all_chunks()
        if not chunks:
            return []
            
        q_vec = np.array(query_emb)
        scored = []
        for c in chunks:
            if not c.get("embedding"):
                continue
            c_vec = np.array(c["embedding"])
            dot = np.dot(q_vec, c_vec)
            nq = np.linalg.norm(q_vec)
            nc = np.linalg.norm(c_vec)
            sim = dot / (nq * nc) if (nq > 0 and nc > 0) else 0.0
            scored.append({
                "chunk": c,
                "score": float(sim)
            })
            
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def reciprocal_rank_fusion(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], rrf_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """Perform Reciprocal Rank Fusion (RRF) with configurable weights and smoothing parameter."""
        rrf_k = rrf_k or self.rrf_k
        rrf_scores = {}
        
        # Compute weighted scores for dense results
        for rank, item in enumerate(dense_results):
            chunk = item["chunk"]
            cid = f"{chunk['doc_id']}_{chunk['id']}"
            rrf_scores[cid] = (chunk, self.dense_weight * (1.0 / (rrf_k + rank + 1)))
            
        # Compute weighted scores for sparse results
        for rank, item in enumerate(sparse_results):
            chunk = item["chunk"]
            cid = f"{chunk['doc_id']}_{chunk['id']}"
            score = self.sparse_weight * (1.0 / (rrf_k + rank + 1))
            if cid in rrf_scores:
                rrf_scores[cid] = (chunk, rrf_scores[cid][1] + score)
            else:
                rrf_scores[cid] = (chunk, score)
                
        # Sort by RRF score descending
        sorted_rrf = sorted(rrf_scores.values(), key=lambda x: x[1], reverse=True)
        return sorted_rrf

    def rerank_results(self, query: str, rrf_results: List[Tuple[Dict[str, Any], float]], top_n: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Re-rank RRF outputs using Cross-Encoder or default vector scoring."""
        if not rrf_results:
            return []
            
        reranking_enabled = os.getenv("RAG_RERANKING_ENABLED", "false").lower() == "true"
        if not reranking_enabled:
            return rrf_results[:top_n]
            
        if HAS_CROSS_ENCODER and not self.reranker_loaded:
            try:
                logger.info("Lazy loading Cross-Encoder model...", name=self.local_reranker_name)
                self.reranker = CrossEncoder(self.local_reranker_name)
                logger.info("Cross-Encoder loaded successfully")
            except Exception as e:
                logger.warn("Failed to load local Cross-Encoder re-ranker. Using fallback scoring.", error=str(e))
                self.reranker = None
            self.reranker_loaded = True
            
        if self.reranker:
            telemetry.start_span("cross_encoder_rerank")
            pairs = [[query, item[0]["text"]] for item in rrf_results]
            try:
                scores = self.reranker.predict(pairs)
                reranked = []
                for i in range(len(rrf_results)):
                    chunk = rrf_results[i][0]
                    score = float(scores[i])
                    # Attach rerank score directly
                    chunk["rerank_score"] = score
                    reranked.append((chunk, score))
                reranked.sort(key=lambda x: x[1], reverse=True)
                telemetry.end_span("cross_encoder_rerank")
                return reranked[:top_n]
            except Exception as e:
                logger.error("Failed to run local Cross-Encoder re-ranker", error=str(e))
                telemetry.end_span("cross_encoder_rerank")
                
        return rrf_results[:top_n]

    def compress_context(self, items: List[Tuple[Dict[str, Any], float]], similarity_threshold: float = 0.85) -> List[Tuple[Dict[str, Any], float]]:
        """Context compression: Remove duplicate chunks or extremely similar text snippets."""
        compressed = []
        seen_texts = []
        
        for chunk, score in items:
            text_cleaned = re.sub(r'\s+', '', chunk["text"].lower())
            
            # Additional low-information context filter: skip if text contains too few alphabetic chars
            alpha_chars = len(re.sub(r'[^a-z]', '', text_cleaned))
            if alpha_chars < 10:
                logger.info("Context compression dropped low-information chunk", id=chunk.get("id"))
                continue
                
            is_duplicate = False
            for prev_text in seen_texts:
                if prev_text in text_cleaned or text_cleaned in prev_text:
                    is_duplicate = True
                    break
            if not is_duplicate:
                compressed.append((chunk, score))
                seen_texts.append(text_cleaned)
                
        return compressed

    def retrieve_hybrid(self, query: str, top_k: int = 4, web_fallback_threshold: float = 0.4) -> Tuple[List[Tuple[Dict[str, Any], float]], bool]:
        """
        Execute the full hybrid pipeline:
        1. Dense Retrieval
        2. Sparse Retrieval
        3. RRF Fusion
        4. Cross-Encoder Re-ranking
        5. Context Compression
        Returns: (List[Tuple[chunk, score]], needs_web_fallback)
        """
        telemetry.start_span("hybrid_retrieve")
        
        dense_hits = self.retrieve_dense(query, top_k=top_k * 2)
        sparse_hits = self.retrieve_sparse(query, top_k=top_k * 2)
        
        rrf_hits = self.reciprocal_rank_fusion(dense_hits, sparse_hits)
        reranked_hits = self.rerank_results(query, rrf_hits, top_n=top_k)
        final_hits = self.compress_context(reranked_hits)
        
        telemetry.end_span("hybrid_retrieve")
        
        # Calculate retrieval confidence score and stamp it onto matched chunks
        for rank, (chunk, score) in enumerate(final_hits):
            # Calculate dynamic confidence (normalized between 0.0 and 1.0)
            if self.reranker:
                # Map cross-encoder scores (typically in range [-10, 10]) to sigmoid range
                confidence = float(1.0 / (1.0 + np.exp(-score)))
            else:
                # Normalized RRF score confidence
                confidence = float(min(1.0, score * 10))
            chunk["retrieval_confidence"] = round(confidence, 2)
            
        needs_web_fallback = False
        if not final_hits:
            needs_web_fallback = True
        else:
            top_score = final_hits[0][1]
            if self.reranker:
                if top_score < -2.0:
                    needs_web_fallback = True
            else:
                if top_score < 0.01:
                    needs_web_fallback = True
                    
        return final_hits, needs_web_fallback
