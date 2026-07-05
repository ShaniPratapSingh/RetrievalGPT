from src.orchestrator.context import AgentContext

class ComparisonAgent:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn

    def run(self, context: AgentContext) -> AgentContext:
        context.add_log("ComparisonAgent", "ComparisonAgent registered request.")
        return context
