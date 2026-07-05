from src.orchestrator.context import AgentContext
from src.agents.query_planner import QueryPlanner

class PlanningAgent:
    def __init__(self):
        self.planner = QueryPlanner()

    def run(self, context: AgentContext) -> AgentContext:
        plan = self.planner.create_plan(context.intent, context.rewritten_query, context.filters)
        context.plan = plan
        
        context.add_log(
            "PlanningAgent",
            "Structured execution retrieval plan generated.",
            {"plan": plan}
        )
        return context
