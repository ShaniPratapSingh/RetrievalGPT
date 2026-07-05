from typing import Dict, Any, List, Callable
from src.core.observability import Logger

logger = Logger("tool_registry")

class ToolRegistry:
    def __init__(self):
        self._tools = {}
        self._audit_logs = []

    def register_tool(self, name: str, description: str, func: Callable, schema: Dict[str, Any], permissions: str = "user"):
        """Register a tool schema and execution function with user access roles."""
        self._tools[name.lower().strip()] = {
            "name": name,
            "description": description,
            "func": func,
            "schema": schema,
            "permissions": permissions
        }
        logger.info("Registered enterprise execution tool", name=name)

    def execute_tool(self, name: str, args: Dict[str, Any], user_role: str = "user") -> Any:
        """Invokes a tool with permission auditing checks and logger captures."""
        key = name.lower().strip()
        if key not in self._tools:
            raise KeyError(f"Tool '{name}' not found.")
            
        tool = self._tools[key]
        
        # Security permission verification
        if tool["permissions"] == "admin" and user_role != "admin":
            logger.warn("Access denied for tool execution", name=name, user=user_role)
            raise PermissionError(f"Insufficient permissions to run tool '{name}'.")
            
        # Logging audits
        self._audit_logs.append({
            "tool": name,
            "args": args,
            "user_role": user_role
        })
        
        try:
            return tool["func"](**args)
        except Exception as e:
            logger.error("Tool execution errored", name=name, error=str(e))
            raise e

    def get_all_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "schema": t["schema"],
                "permissions": t["permissions"]
            }
            for t in self._tools.values()
        ]
