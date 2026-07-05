import time
from typing import List, Dict, Any, Optional

class AgentContext:
    def __init__(self, query: str, history: Optional[List[Any]] = None, filters: Optional[Dict[str, Any]] = None):
        self.query = query
        self.history = history or []
        self.filters = filters or {}
        
        # Intermediate state variables
        self.intent: str = "factual_question"
        self.rewritten_query: str = query
        self.language: str = "english"
        self.complexity: str = "medium"
        
        # Planner details
        self.plan: Dict[str, Any] = {}
        
        # Retrieval results
        self.retrieved_chunks: List[Any] = []
        self.web_search_fallback: bool = False
        
        # Final outputs
        self.final_answer: str = ""
        self.citations: List[Dict[str, Any]] = []
        self.confidence_metrics: Dict[str, Any] = {
            "retrieval_confidence": 0.0,
            "evidence_coverage": 0.0,
            "answer_confidence": 0.0
        }
        
        # Traces & Logging
        self.logs: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_log(self, agent_name: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Append a trace log step to the context execution list."""
        self.logs.append({
            "timestamp": time.time() - self.start_time,
            "agent": agent_name,
            "message": message,
            "data": data or {}
        })
