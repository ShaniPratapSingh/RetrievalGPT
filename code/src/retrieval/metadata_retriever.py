import os
from typing import List, Dict, Any, Tuple
from src.core.observability import Logger, telemetry

logger = Logger("metadata_retriever")

class MetadataRetriever:
    def __init__(self, storage_manager, get_embedding_fn, hybrid_retriever):
        self.storage = storage_manager
        self.get_embedding = get_embedding_fn
        self.hybrid = hybrid_retriever

    def get_dynamic_top_k(self, intent: str) -> int:
        """Determines the number of chunks to retrieve based on the user intent classification."""
        intent_lower = intent.lower()
        if intent_lower in ["factual question", "definition"]:
            return 4
        elif intent_lower in ["important quote", "important line", "key concepts"]:
            return 8
        elif intent_lower in ["comparison"]:
            return 10
        elif intent_lower in ["reasoning", "analytical question"]:
            return 16
        return 5

    def apply_filters(self, chunks: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter chunks based on metadata keys: document, page, chapter, section."""
        if not filters:
            return chunks
            
        filtered = []
        for c in chunks:
            match = True
            for k, val in filters.items():
                if val is None or val == "All" or val == "":
                    continue
                # Normalize values
                chunk_val = c.get(k)
                if k == "document":
                    chunk_val = c.get("source") # Map document filter to source field
                elif k == "page":
                    chunk_val = c.get("page")
                    
                if chunk_val is not None:
                    # Compare as strings case-insensitive
                    if str(chunk_val).strip().lower() != str(val).strip().lower():
                        match = False
                        break
                else:
                    match = False
                    break
            if match:
                filtered.append(c)
        return filtered

    def retrieve(self, query: str, intent: str = "factual question", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Runs the hybrid dense-sparse retrieval, applies dynamic top-k limits, and filters metadata."""
        top_k = self.get_dynamic_top_k(intent)
        
        # 1. Fetch dense and sparse scores
        dense_hits = self.hybrid.retrieve_dense(query, top_k=top_k * 2)
        sparse_hits = self.hybrid.retrieve_sparse(query, top_k=top_k * 2)
        
        # 2. Run RRF Fusion
        rrf_hits = self.hybrid.reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        # Map back to dict list representation
        merged_chunks = []
        for chunk, score in rrf_hits:
            c = chunk.copy()
            c["rrf_score"] = score
            merged_chunks.append(c)
            
        # 3. Apply active filters
        filtered_chunks = self.apply_filters(merged_chunks, filters)
        
        # 4. Limit to top_k
        return filtered_chunks[:top_k]
