import os
import hashlib
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from src.core.observability import Logger, telemetry

logger = Logger("metadata_retriever")

class MetadataRetriever:
    def __init__(self, storage_manager, get_embedding_fn, hybrid_retriever):
        self.storage = storage_manager
        self.get_embedding = get_embedding_fn
        self.hybrid = hybrid_retriever
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.retrieval_cache = {} # Simple in-memory retrieval cache

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
        """
        Filters chunks dynamically.
        Supported keys:
        - document (or source)
        - document_id
        - chapter
        - page (specific page number)
        - page_range (tuple of page_start, page_end)
        - tags (list of tags, match if overlaps)
        - upload_date
        """
        if not filters:
            return chunks
            
        filtered = []
        for c in chunks:
            match = True
            for k, val in filters.items():
                if val is None or val == "All" or val == "":
                    continue
                
                if k in ["document", "source"]:
                    chunk_val = c.get("source") or c.get("filename")
                    if str(chunk_val).strip().lower() != str(val).strip().lower():
                        match = False
                        break
                elif k == "document_id":
                    chunk_val = c.get("document_id") or c.get("doc_id")
                    if str(chunk_val).strip() != str(val).strip():
                        match = False
                        break
                elif k == "chapter":
                    chunk_val = c.get("chapter")
                    if str(chunk_val).strip().lower() != str(val).strip().lower():
                        match = False
                        break
                elif k == "page":
                    chunk_val = c.get("page")
                    if str(chunk_val).strip() != str(val).strip():
                        match = False
                        break
                elif k == "page_range":
                    # Expected tuple: (start_page, end_page)
                    chunk_val = c.get("page")
                    if chunk_val is not None:
                        try:
                            page_num = int(chunk_val)
                            start_page, end_page = int(val[0]), int(val[1])
                            if not (start_page <= page_num <= end_page):
                                match = False
                                break
                        except (ValueError, TypeError, IndexError):
                            match = False
                            break
                    else:
                        match = False
                        break
                elif k == "tags":
                    chunk_tags = c.get("tags", [])
                    if isinstance(val, list):
                        if not any(t in chunk_tags for t in val):
                            match = False
                            break
                    elif val not in chunk_tags:
                        match = False
                        break
                elif k == "upload_date":
                    chunk_date = c.get("created_at") or c.get("upload_date")
                    if chunk_date and str(val) not in str(chunk_date):
                        match = False
                        break
            if match:
                filtered.append(c)
        return filtered

    def retrieve(self, query: str, intent: str = "factual question", filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Runs parallel dense-sparse retrieval, applies metadata filters, and handles caching."""
        # Check Cache
        cache_key = hashlib.md5(f"q:{query}_i:{intent}_f:{str(filters)}".encode("utf-8")).hexdigest()
        if cache_key in self.retrieval_cache:
            logger.info("Serving retrieval results from cache", query=query)
            return self.retrieval_cache[cache_key]

        top_k = self.get_dynamic_top_k(intent)
        
        # 1. Parallel dense & sparse fetches
        future_dense = self.executor.submit(self.hybrid.retrieve_dense, query, top_k * 2)
        future_sparse = self.executor.submit(self.hybrid.retrieve_sparse, query, top_k * 2)
        
        dense_hits = future_dense.result()
        sparse_hits = future_sparse.result()
        
        # 2. RRF Fusion
        rrf_hits = self.hybrid.reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        merged_chunks = []
        for chunk, score in rrf_hits:
            c = chunk.copy()
            c["rrf_score"] = score
            merged_chunks.append(c)
            
        # 3. Apply active filters
        filtered_chunks = self.apply_filters(merged_chunks, filters)
        
        # Limit to top_k
        results = filtered_chunks[:top_k]
        
        # Save to Cache
        self.retrieval_cache[cache_key] = results
        return results
