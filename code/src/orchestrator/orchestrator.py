import time
from src.orchestrator.context import AgentContext
from src.orchestrator.registry import AgentRegistry
from src.core.observability import Logger

logger = Logger("orchestrator")

class AgentOrchestrator:
    def __init__(self, registry: AgentRegistry):
        self.registry = registry

    def execute(self, context: AgentContext) -> AgentContext:
        """Coordinates transition flows between agents sequentially with structured error fallbacks."""
        context.add_log("Orchestrator", "Initializing orchestrator workflow pipeline execution.")
        
        try:
            # 1. Query Understanding
            query_agent = self.registry.get("query_agent")
            context = query_agent.run(context)
            
            # 2. Planning
            planner_agent = self.registry.get("planner_agent")
            context = planner_agent.run(context)
            
            # 3. Retrieval
            retrieval_agent = self.registry.get("retrieval_agent")
            context = retrieval_agent.run(context)
            
            # 4. Verification (Groundedness Check)
            verification_agent = self.registry.get("verification_agent")
            context = verification_agent.run(context)
            
            # 5. Web Search Fallback check
            # Trigger ONLY if no uploaded documents exist, OR query explicitly asks for web search
            ret_agent = self.registry.get("retrieval_agent")
            has_uploaded_docs = False
            if hasattr(ret_agent, "metadata_retriever"):
                has_uploaded_docs = len(ret_agent.metadata_retriever.storage.get_all_chunks()) > 0
                
            user_wants_web = any(w in context.query.lower() for w in ["google", "web search", "internet", "online", "tavily", "serper"])
            
            if (not has_uploaded_docs or user_wants_web) and (not context.retrieved_chunks or context.plan.get("web_search_necessity", False)):
                try:
                    web_agent = self.registry.get("web_agent")
                    context = web_agent.run(context)
                except KeyError:
                    context.add_log("Orchestrator", "Web Search Agent is not registered, bypassing search fallback.")
                
            # 6. Citation Extraction
            citation_agent = self.registry.get("citation_agent")
            context = citation_agent.run(context)
            
            # 7. Final Response Generation
            response_agent = self.registry.get("response_agent")
            context = response_agent.run(context)
            
        except Exception as e:
            logger.error("Orchestrator failed during step transition execution", error=str(e))
            context.add_log("Orchestrator", f"Pipeline execution errored: {str(e)}")
            context.final_answer = "An unexpected error occurred during multi-agent orchestration."
            
        context.add_log("Orchestrator", "Completed orchestrator pipeline execution flow.")
        return context
