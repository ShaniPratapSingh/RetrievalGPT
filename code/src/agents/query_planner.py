from typing import Dict, Any, List
from src.core.observability import Logger

logger = Logger("agents_query_planner")

class QueryPlanner:
    def __init__(self):
        pass

    def create_plan(self, intent: str, query: str, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Creates a query plan mapping the intent to optimal retrieval parameters."""
        filters = filters or {}
        
        # Default plan parameters
        plan = {
            "retrieval_strategy": "hybrid",
            "top_k": 5,
            "reranking_depth": 10,
            "metadata_filters": filters,
            "web_search_necessity": False,
            "context_size": 2000
        }
        
        # Normalize legacy space intents to underscore
        intent_norm = intent.lower().replace(" ", "_")
        
        if intent_norm in ["factual_question", "factual", "definition"]:
            plan["top_k"] = 4
            plan["reranking_depth"] = 8
            plan["context_size"] = 1500
        elif intent_norm == "summarization":
            plan["top_k"] = 16
            plan["reranking_depth"] = 20
            plan["context_size"] = 8000
        elif intent_norm == "chapter_summary":
            plan["top_k"] = 12
            plan["reranking_depth"] = 16
            plan["context_size"] = 6000
        elif intent_norm in ["quote_extraction", "important_quote", "important_line"]:
            plan["top_k"] = 8
            plan["reranking_depth"] = 12
            plan["context_size"] = 2500
        elif intent_norm == "comparison":
            plan["top_k"] = 10
            plan["reranking_depth"] = 15
            plan["context_size"] = 5000
        elif intent_norm == "reasoning":
            plan["top_k"] = 16
            plan["reranking_depth"] = 24
            plan["context_size"] = 8000
        elif intent_norm == "search":
            plan["web_search_necessity"] = True
            
        logger.info("Structured query plan created successfully", intent=intent, plan=plan)
        return plan
