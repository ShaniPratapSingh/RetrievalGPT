import json
import re
from typing import Dict, Any, Tuple
from src.core.observability import Logger, telemetry

logger = Logger("agent")

class QueryAnalysisAgent:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn

    def analyze_query_local(self, query: str) -> Dict[str, Any]:
        """Simple rule-based regex fallback analyzer for when LLM is unavailable."""
        query_lower = query.lower()
        classification = "factual"
        strategy = "hybrid"
        chunk_count = 4
        web_fallback_needed = False
        summary_mode = "short"
        
        # Detect mode
        if "detailed" in query_lower:
            summary_mode = "detailed"
        elif any(w in query_lower for w in ["bullet", "points", "ideas"]):
            summary_mode = "bullet"
        elif any(w in query_lower for w in ["chapter", "section", "parts"]):
            summary_mode = "chapter-wise"
            
        if any(w in query_lower for w in ["compare", "difference", "versus", "vs"]):
            classification = "comparison"
            chunk_count = 6
        elif any(w in query_lower for w in ["summarize", "summary", "overview", "brief", "concise summary", "key points", "main ideas", "explain this document", "what is this document about"]):
            classification = "summarization"
            chunk_count = 8
            strategy = "dense"
        elif any(w in query_lower for w in ["why", "how", "reason", "explain"]):
            classification = "reasoning"
            chunk_count = 5
        elif any(w in query_lower for w in ["today", "recent", "latest", "weather", "news", "current"]):
            classification = "factual"
            web_fallback_needed = True

        return {
            "classification": classification,
            "strategy": strategy,
            "chunk_count": chunk_count,
            "search_depth": chunk_count * 2,
            "web_fallback_needed": web_fallback_needed,
            "rewritten_query": query,
            "summary_mode": summary_mode
        }

    def analyze_query(self, query: str, history_str: str = "") -> Dict[str, Any]:
        """Classify query using the LLM and return dynamic parameters."""
        telemetry.start_span("agent_query_analysis")
        
        system_prompt = (
            "You are an expert Query Analysis Agent. Classify user queries and output a structured plan. "
            "Respond in JSON format within <answer> </answer> tags."
        )
        
        prompt = f"""Analyze the user's search query and formulate the optimal retrieval plan.
        
Recent conversation history (for coreference resolution, e.g., resolving 'he', 'they', 'this'):
{history_str}

User's current query: "{query}"

Output your final decision as a JSON object inside <answer> </answer> tags.
Example output format:
<answer>
{{
    "classification": "factual" | "summarization" | "comparison" | "reasoning" | "conversational" | "multi-hop" | "analytical" | "out-of-domain",
    "strategy": "dense" | "sparse" | "hybrid",
    "chunk_count": 4,
    "search_depth": 8,
    "web_fallback_needed": false | true,
    "rewritten_query": "Fully expanded search query resolving any shorthand or references",
    "summary_mode": "short" | "detailed" | "bullet" | "chapter-wise"
}}
</answer>
"""
        try:
            provider, response_text = self.call_llm(prompt, system_prompt)
            telemetry.record_provider(provider)
            
            if "Demo Fallback" in provider:
                # LLM is mock, fall back to rule-based analysis
                res = self.analyze_query_local(query)
                telemetry.end_span("agent_query_analysis")
                return res
                
            # Extract JSON from tags
            ans_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
            if ans_match:
                plan_json = json.loads(ans_match.group(1).strip())
            else:
                # Direct JSON search fallback
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    plan_json = json.loads(json_match.group(0))
                else:
                    raise ValueError("No valid JSON found in query analysis response")
                    
            # Set default values if keys are missing
            plan_json["classification"] = plan_json.get("classification", "factual")
            plan_json["strategy"] = plan_json.get("strategy", "hybrid")
            plan_json["chunk_count"] = int(plan_json.get("chunk_count", 4))
            plan_json["search_depth"] = int(plan_json.get("search_depth", 8))
            plan_json["web_fallback_needed"] = bool(plan_json.get("web_fallback_needed", False))
            plan_json["rewritten_query"] = plan_json.get("rewritten_query", query)
            plan_json["summary_mode"] = plan_json.get("summary_mode", "short")
            
            logger.info("Query analyzed by agent", provider=provider, classification=plan_json["classification"], strategy=plan_json["strategy"])
            telemetry.end_span("agent_query_analysis")
            return plan_json
            
        except Exception as e:
            logger.warn("Agentic query analysis failed. Falling back to local rule-based plan.", error=str(e))
            res = self.analyze_query_local(query)
            telemetry.end_span("agent_query_analysis")
            return res
