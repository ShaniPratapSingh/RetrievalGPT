from src.orchestrator.context import AgentContext
from src.retrieval.metadata_retriever import MetadataRetriever

class RetrievalAgent:
    def __init__(self, hybrid_retriever, storage_manager, embedding_fn):
        self.hybrid = hybrid_retriever
        self.metadata_retriever = MetadataRetriever(storage_manager, embedding_fn, hybrid_retriever)

    def run(self, context: AgentContext) -> AgentContext:
        top_k = context.plan.get("top_k", 5)
        strategy = context.plan.get("retrieval_strategy", "hybrid")
        
        dense_hits = []
        sparse_hits = []
        
        # Parallel dense & sparse fetches are optimized within metadata_retriever
        if strategy in ["dense", "hybrid"]:
            dense_hits = self.hybrid.retrieve_dense(context.rewritten_query, top_k=top_k * 2)
        if strategy in ["sparse", "hybrid"]:
            sparse_hits = self.hybrid.retrieve_sparse(context.rewritten_query, top_k=top_k * 2)
            
        # reciprocal rank fusion
        rrf_hits = self.hybrid.reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        # Format list to dict array
        merged_chunks = []
        for chunk, score in rrf_hits:
            c = chunk.copy()
            c["rrf_score"] = score
            merged_chunks.append(c)
            
        # Apply filters & reranking
        filtered = self.metadata_retriever.apply_filters(merged_chunks, context.filters)
        reranked = self.hybrid.rerank_results(context.rewritten_query, [(c, c.get("rrf_score", 1.0)) for c in filtered], top_n=context.plan.get("reranking_depth", 10))
        
        # Context Compression Jaccard filter
        compressed = self.hybrid.compress_context(reranked)
        
        # Mapping to context matches format
        context.retrieved_chunks = compressed[:top_k]
        
        context.add_log(
            "RetrievalAgent",
            f"Context hybrid retrieve complete. Found {len(context.retrieved_chunks)} chunks.",
            {"chunk_ids": [c[0].get("id") for c in context.retrieved_chunks]}
        )
        return context
