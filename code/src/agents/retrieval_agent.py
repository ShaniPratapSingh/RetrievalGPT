import re
from src.orchestrator.context import AgentContext
from src.retrieval.metadata_retriever import MetadataRetriever
from src.tools.tool_registry import ToolRegistry
from src.connectors.github_connector import GitHubConnector
from src.connectors.sql_connector import SQLConnector

class RetrievalAgent:
    def __init__(self, hybrid_retriever, storage_manager, embedding_fn):
        self.hybrid = hybrid_retriever
        self.metadata_retriever = MetadataRetriever(storage_manager, embedding_fn, hybrid_retriever)
        
        # Instantiate Tool Registry and Connectors
        self.tool_registry = ToolRegistry()
        self.gh_connector = GitHubConnector()
        self.sql_connector = SQLConnector()
        
        # Authenticate with safe simulated credentials
        self.gh_connector.authenticate({"token": "mock_token", "repo": "mock/repo"})
        self.sql_connector.authenticate({"db_path": ":memory:"})
        
        # Register Tools
        self.tool_registry.register_tool(
            name="github",
            description="Searches GitHub code repositories",
            func=self.gh_connector.search,
            schema={"query": "str"}
        )
        self.tool_registry.register_tool(
            name="sql",
            description="Queries SQL database tables schemas",
            func=self.sql_connector.search,
            schema={"query": "str"}
        )

    def run(self, context: AgentContext) -> AgentContext:
        top_k = context.plan.get("top_k", 5)
        strategy = context.plan.get("retrieval_strategy", "hybrid")
        
        # Parse chapter references dynamically
        query_lower = context.rewritten_query.lower()
        if "chapter" in query_lower:
            match = re.search(r'chapter\s*(\d+|one|two|three|four|five|six|seven|eight|nine|ten)', query_lower)
            if match:
                ch_val = match.group(1).strip()
                context.filters["chapter"] = f"chapter {ch_val}"
                
        # Parse page references dynamically
        if "page" in query_lower:
            page_match = re.search(r'page\s*(\d+)', query_lower)
            if page_match:
                p_val = int(page_match.group(1).strip())
                context.filters["page"] = p_val

        dense_hits = []
        sparse_hits = []
        
        if strategy in ["dense", "hybrid"]:
            dense_hits = self.hybrid.retrieve_dense(context.rewritten_query, top_k=top_k * 2)
        if strategy in ["sparse", "hybrid"]:
            sparse_hits = self.hybrid.retrieve_sparse(context.rewritten_query, top_k=top_k * 2)
            
        # RRF Fusion
        rrf_hits = self.hybrid.reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        merged_chunks = []
        for chunk, score in rrf_hits:
            c = chunk.copy()
            c["rrf_score"] = score
            merged_chunks.append(c)
            
        # Execute Knowledge Connector Tools if planner demands them
        connector_hits = []
        tool_choice = context.plan.get("required_tool", "none")
        if tool_choice in ["github", "sql"]:
            context.add_log("RetrievalAgent", f"Invoking tool call connector: {tool_choice}")
            try:
                results = self.tool_registry.execute_tool(tool_choice, {"query": context.rewritten_query})
                # Map scores to fused structure
                for r in results:
                    r["rrf_score"] = 0.95
                    r["rerank_score"] = 0.95
                    connector_hits.append(r)
            except Exception as e:
                context.add_log("RetrievalAgent", f"Tool invocation error: {str(e)}")

        # Merge Evidence chunks
        fused_chunks = connector_hits + merged_chunks
        
        # Apply filters & reranking
        filtered = self.metadata_retriever.apply_filters(fused_chunks, context.filters)
        reranked = self.hybrid.rerank_results(context.rewritten_query, [(c, c.get("rrf_score", 1.0)) for c in filtered], top_n=context.plan.get("reranking_depth", 10))
        
        # Context Compression
        compressed = self.hybrid.compress_context(reranked)
        
        context.retrieved_chunks = compressed[:top_k]
        
        context.add_log(
            "RetrievalAgent",
            f"Context retrieve complete. Found {len(context.retrieved_chunks)} total chunks after tool execution.",
            {"chunk_ids": [c[0].get("id") for c in context.retrieved_chunks]}
        )
        return context
