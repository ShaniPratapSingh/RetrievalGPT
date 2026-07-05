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

from src.tools.tool_registry import ToolRegistry
from src.connectors.github_connector import GitHubConnector
from src.connectors.sql_connector import SQLConnector
from src.agents.planner_agent import PlanningAgent
from src.orchestrator.context import AgentContext

class TestKnowledgeConnectors(unittest.TestCase):
    def setUp(self):
        self.registry = ToolRegistry()
        self.gh_connector = GitHubConnector()
        self.sql_connector = SQLConnector()
        
        self.gh_connector.authenticate({"token": "test", "repo": "mock/repo"})
        self.sql_connector.authenticate({"db_path": ":memory:"})

    def test_tool_registration_and_execution(self):
        # Register GitHub tool
        self.registry.register_tool(
            name="github_search",
            description="Search repository",
            func=self.gh_connector.search,
            schema={"query": "str"}
        )
        
        results = self.registry.execute_tool("github_search", {"query": "auth"})
        self.assertEqual(len(results), 1)
        self.assertIn("github:mock/repo", results[0]["source"])

    def test_tool_permissions_denial(self):
        self.registry.register_tool(
            name="admin_tool",
            description="Admin Only",
            func=lambda: "done",
            schema={},
            permissions="admin"
        )
        
        # Calling with default user role should raise PermissionError
        with self.assertRaises(PermissionError):
            self.registry.execute_tool("admin_tool", {}, user_role="user")
            
        # Calling with admin role should succeed
        res = self.registry.execute_tool("admin_tool", {}, user_role="admin")
        self.assertEqual(res, "done")

    def test_planner_tool_selection_logic(self):
        planner = PlanningAgent()
        
        # Test GitHub query triggers github tool selection
        ctx_gh = AgentContext("Search github repository for main changes")
        ctx_gh = planner.run(ctx_gh)
        self.assertEqual(ctx_gh.plan["required_tool"], "github")
        
        # Test database query triggers sql tool selection
        ctx_sql = AgentContext("Fetch SQL schema table details")
        ctx_sql = planner.run(ctx_sql)
        self.assertEqual(ctx_sql.plan["required_tool"], "sql")

if __name__ == "__main__":
    unittest.main()
