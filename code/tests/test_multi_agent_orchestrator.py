import unittest
import os
import sys

# Add source directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.orchestrator.context import AgentContext
from src.orchestrator.registry import AgentRegistry
from src.orchestrator.orchestrator import AgentOrchestrator

class TestMultiAgentOrchestrator(unittest.TestCase):
    def setUp(self):
        self.registry = AgentRegistry()
        self.orchestrator = AgentOrchestrator(self.registry)
        
        # Setup dummy mock LLM responder
        def mock_llm(prompt, system_prompt=None):
            return "Mock Provider", '<answer>{"intent": "factual_question", "query": "rewritten query"}</answer>'
        self.mock_llm = mock_llm

    def test_agent_registry(self):
        # Register a mock agent instance
        dummy_agent = object()
        self.registry.register("test_agent", dummy_agent)
        
        self.assertEqual(self.registry.get("test_agent"), dummy_agent)
        # Test case-insensitivity mapping
        self.assertEqual(self.registry.get("TEST_AGENT"), dummy_agent)
        
        with self.assertRaises(KeyError):
            self.registry.get("non_existent_agent")

    def test_agent_context_logs(self):
        ctx = AgentContext("Search query")
        ctx.add_log("DummyAgent", "Running logic check.", {"step": 1})
        
        self.assertEqual(len(ctx.logs), 1)
        self.assertEqual(ctx.logs[0]["agent"], "DummyAgent")
        self.assertEqual(ctx.logs[0]["data"]["step"], 1)

    def test_orchestrator_execution_flow(self):
        # Register simplified mock agents
        class MockAgent:
            def __init__(self, name):
                self.name = name
            def run(self, ctx: AgentContext) -> AgentContext:
                ctx.add_log(self.name, "Running mock agent task.")
                if self.name == "ResponseAgent":
                    ctx.final_answer = "Mock orchestrated response."
                return ctx

        self.registry.register("query_agent", MockAgent("QueryAgent"))
        self.registry.register("planner_agent", MockAgent("PlannerAgent"))
        self.registry.register("retrieval_agent", MockAgent("RetrievalAgent"))
        self.registry.register("verification_agent", MockAgent("VerificationAgent"))
        self.registry.register("citation_agent", MockAgent("CitationAgent"))
        self.registry.register("response_agent", MockAgent("ResponseAgent"))
        
        ctx = AgentContext("Verify prompt info")
        ctx = self.orchestrator.execute(ctx)
        
        # Verify execution succeeded and final answer was set
        self.assertEqual(ctx.final_answer, "Mock orchestrated response.")
        # Verify sequence was logged properly
        agents_logged = [log["agent"] for log in ctx.logs]
        self.assertIn("QueryAgent", agents_logged)
        self.assertIn("PlannerAgent", agents_logged)
        self.assertIn("ResponseAgent", agents_logged)

if __name__ == "__main__":
    unittest.main()
