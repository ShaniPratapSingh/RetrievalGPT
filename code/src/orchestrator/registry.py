class AgentRegistry:
    def __init__(self):
        self._agents = {}

    def register(self, name: str, agent_instance):
        """Registers a specialized agent instance."""
        self._agents[name.lower().strip()] = agent_instance

    def get(self, name: str):
        """Retrieves registered agent instance."""
        key = name.lower().strip()
        if key not in self._agents:
            raise KeyError(f"Agent '{name}' is not registered.")
        return self._agents[key]
