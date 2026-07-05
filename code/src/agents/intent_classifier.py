import re
import json
from typing import Dict, Any, List
from src.core.observability import Logger

logger = Logger("agents_intent_classifier")

class QueryIntentClassifier:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn
        # Accept both legacy format and new underscores format for backward compatibility
        self.valid_intents = [
            "factual_question", "factual question",
            "summarization",
            "chapter_summary", "chapter summary",
            "quote_extraction", "important quote", "important line",
            "definition",
            "explanation", "comparison", "reasoning", "analytical", "follow_up",
            "document_overview", "document overview",
            "search", "out_of_scope"
        ]

    def classify_locally(self, query: str) -> str:
        """Fallback rule-based classification."""
        query_lower = query.lower()
        if any(w in query_lower for w in ["quote", "line", "sentence", "memorable", "extract"]):
            return "important quote"
        if any(w in query_lower for w in ["summarize", "summary", "overview", "what is this document about"]):
            if "chapter" in query_lower:
                return "chapter summary"
            if "overview" in query_lower or "about" in query_lower:
                return "document overview"
            return "summarization"
        if any(w in query_lower for w in ["compare", "difference", "vs", "versus"]):
            return "comparison"
        if any(w in query_lower for w in ["define", "definition"]):
            return "definition"
        if any(w in query_lower for w in ["what is", "meaning of"]):
            return "factual question"
        if any(w in query_lower for w in ["why", "how", "reason"]):
            return "reasoning"
        if any(w in query_lower for w in ["google", "web", "search"]):
            return "search"
        return "factual question"

    def classify(self, query: str, history_str: str = "") -> Dict[str, Any]:
        """Classifies the query intent using LLM and local fallback rules."""
        system_prompt = (
            "You are an expert intent classifier. Categorize user queries based on the target intent. "
            "Respond strictly in JSON format within <answer> </answer> tags."
        )
        prompt = f"""Classify the user's search query intent.
        
Recent conversation history:
{history_str}

User's current query: "{query}"

Available intents:
{self.valid_intents}

Output JSON format:
<answer>
{{
    "intent": "one of the available intents",
    "summary_mode": "short" | "detailed" | "bullet" (only if intent is summarization or chapter summary),
    "explanation": "concise explanation of intent selection"
}}
</answer>
"""
        try:
            provider, response_text = self.call_llm(prompt, system_prompt)
            if provider in ["Test Fallback (Mock)", "Demo Fallback (Mock)"]:
                intent = self.classify_locally(query)
                return {"intent": intent, "summary_mode": "short", "explanation": "Fallback local classification"}
                
            ans_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
            if ans_match:
                result = json.loads(ans_match.group(1).strip())
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON block found")
                    
            intent = result.get("intent", "").strip()
            if intent not in self.valid_intents:
                intent = self.classify_locally(query)
                
            summary_mode = result.get("summary_mode", "short")
            return {
                "intent": intent,
                "summary_mode": summary_mode,
                "explanation": result.get("explanation", "LLM classified")
            }
        except Exception as e:
            logger.warn("Query classification failed, running local rules", error=str(e))
            intent = self.classify_locally(query)
            return {"intent": intent, "summary_mode": "short", "explanation": "Fallback local classification"}
