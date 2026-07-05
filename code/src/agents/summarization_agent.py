from src.orchestrator.context import AgentContext
from src.core.services.summarization_service import SummarizationService

class SummarizationAgent:
    def __init__(self, call_llm_fn):
        self.summarizer = SummarizationService(call_llm_fn)

    def run(self, context: AgentContext) -> AgentContext:
        # If intent is summarization, we can trigger summaries
        context.add_log("SummarizationAgent", "SummarizationAgent parsed request.")
        return context
