from src.orchestrator.context import AgentContext
from src.core.web_search import WebSearchClient

class WebSearchAgent:
    def __init__(self):
        self.web_search = WebSearchClient()

    def run(self, context: AgentContext) -> AgentContext:
        context.add_log("WebSearchAgent", "Triggering Web Search Fallback query...")
        web_hits = self.web_search.search(context.rewritten_query)
        
        # Map hits to retriever context chunks structure
        for hit in web_hits:
            context.retrieved_chunks.append((hit, 0.90))
            
        context.web_search_fallback = True
        context.add_log(
            "WebSearchAgent",
            f"Web search execution complete. Added {len(web_hits)} web chunks.",
            {"web_hits_count": len(web_hits)}
        )
        return context
