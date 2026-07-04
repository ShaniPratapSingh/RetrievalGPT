import os
from typing import List, Dict, Any
from src.core.observability import Logger, telemetry

logger = Logger("reranker")

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


class Reranker:
    def __init__(self, local_reranker_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.local_reranker_name = local_reranker_name
        self.reranker = None
        self.reranker_loaded = False

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """Re-rank chunks using local Cross-Encoder if enabled, falling back to original sorting."""
        if not chunks:
            return []
            
        reranking_enabled = os.getenv("RAG_RERANKING_ENABLED", "false").lower() == "true"
        if not reranking_enabled:
            return chunks[:top_n]
            
        if HAS_CROSS_ENCODER and not self.reranker_loaded:
            try:
                logger.info("Lazy loading Cross-Encoder model...", name=self.local_reranker_name)
                self.reranker = CrossEncoder(self.local_reranker_name)
                logger.info("Cross-Encoder loaded successfully")
            except Exception as e:
                logger.warn("Failed to load local Cross-Encoder re-ranker. Using default similarity scoring.", error=str(e))
                self.reranker = None
            self.reranker_loaded = True
            
        if self.reranker:
            telemetry.start_span("cross_encoder_rerank")
            # Pack input pairs: [query, chunk_text]
            pairs = [[query, c["text"]] for c in chunks]
            try:
                scores = self.reranker.predict(pairs)
                # Zip scores and sort
                scored_chunks = []
                for idx, score in enumerate(scores):
                    c = chunks[idx].copy()
                    c["rerank_score"] = float(score)
                    scored_chunks.append(c)
                # Sort by rerank score descending
                scored_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                telemetry.end_span("cross_encoder_rerank")
                return scored_chunks[:top_n]
            except Exception as e:
                logger.error("Failed to run local Cross-Encoder re-ranker", error=str(e))
                telemetry.end_span("cross_encoder_rerank")
                
        return chunks[:top_n]
