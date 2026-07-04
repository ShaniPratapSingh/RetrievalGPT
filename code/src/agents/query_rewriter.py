import re
import json
from typing import Dict, Any, List
from src.core.observability import Logger

logger = Logger("agents_query_rewriter")

class QueryRewriter:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn

    def rewrite(self, query: str, history_str: str = "") -> str:
        """Rewrites conversational queries to resolve pronouns and implicit references."""
        system_prompt = (
            "You are an expert search query rewriter. Your task is to expand the user query, "
            "resolving pronouns and ambiguity into specific search terms. "
            "Respond strictly in JSON format within <answer> </answer> tags."
        )
        
        prompt = f"""Rewrite the following user query to be fully self-contained and descriptive for index retrieval.

Conversation History:
{history_str}

Current Query: "{query}"

Output JSON format:
<answer>
{{
    "query": "rewritten search terms"
}}
</answer>
"""
        try:
            provider, response_text = self.call_llm(prompt, system_prompt)
            if "Demo Fallback" in provider:
                return query
                
            ans_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
            if ans_match:
                result = json.loads(ans_match.group(1).strip())
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON block found")
            return result.get("query", query)
        except Exception as e:
            logger.warn("Query rewriting failed, falling back to original query", error=str(e))
            return query
