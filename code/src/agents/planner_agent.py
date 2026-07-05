from src.orchestrator.context import AgentContext
from src.agents.query_planner import QueryPlanner

class PlanningAgent:
    def __init__(self):
        self.planner = QueryPlanner()

    def run(self, context: AgentContext) -> AgentContext:
        plan = self.planner.create_plan(context.intent, context.rewritten_query, context.filters)
        
        # Phase 6: Dynamic Tool Selection
        query_lower = context.query.lower()
        if "github" in query_lower or "repo" in query_lower or "master branch" in query_lower:
            plan["required_tool"] = "github"
        elif "database" in query_lower or "sql" in query_lower or "table schema" in query_lower:
            plan["required_tool"] = "sql"
        elif "google" in query_lower or "web search" in query_lower:
            plan["required_tool"] = "web_search"
        else:
            plan["required_tool"] = "none"
            
        context.plan = plan
        context.add_log(
            "PlanningAgent",
            "Execution plan generated with tool selection targets.",
            {"plan": plan}
        )
        return context
