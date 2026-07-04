from typing import List, Dict, Any, Tuple
from src.core.observability import Logger

logger = Logger("agents_response_generator")

class ResponseGenerator:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn

    def estimate_confidence(self, hits: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        """Calculates evidence coverage, retrieval confidence scores, and answer confidence."""
        if not hits:
            return {
                "retrieval_confidence": 0.0,
                "evidence_coverage": 0.0,
                "answer_confidence": 0.0
            }
            
        # Retrieval confidence is the average confidence across matched hits
        retrieval_conf = sum(h[0].get("retrieval_confidence", 0.5) for h in hits) / len(hits)
        
        # Evidence coverage maps to total distinct sources
        distinct_sources = len({h[0].get("source") for h in hits if h[0].get("source")})
        evidence_coverage = min(1.0, distinct_sources * 0.25)
        
        # Combined dynamic answer confidence
        ans_conf = (retrieval_conf * 0.7) + (evidence_coverage * 0.3)
        
        return {
            "retrieval_confidence": round(retrieval_conf, 2),
            "evidence_coverage": round(evidence_coverage, 2),
            "answer_confidence": round(ans_conf, 2)
        }

    def generate(self, query: str, hits: List[Tuple[Dict[str, Any], float]], intent: str = "factual_question") -> Tuple[str, Dict[str, Any]]:
        """Generates answer using LLM, verifying refusal rules when confidence or sources are low."""
        scores = self.estimate_confidence(hits)
        
        # Low confidence refusal criteria
        if not hits or scores["retrieval_confidence"] < 0.25:
            logger.warn("Low retrieval confidence, executing refusal rules", confidence=scores["retrieval_confidence"])
            return "I could not find enough evidence in the uploaded documents.", scores
            
        # Format the contexts for LLM prompt
        context_str = ""
        for i, (chunk, _) in enumerate(hits):
            src = chunk.get("source", "unknown")
            page = chunk.get("page", "?")
            context_str += f"[Source {i+1} - {src} Page {page}]:\n{chunk['text']}\n\n"
            
        system_prompt = (
            "You are a production-grade enterprise assistant. "
            "Formulate answers based strictly on the provided context evidence. "
            "Cite sources cleanly using [Source N] tags."
        )
        
        prompt = f"""Synthesize a clear and comprehensive response for the query: "{query}"

Intent Mode: {intent}

Context Evidence:
{context_str}
"""
        try:
            _, response_text = self.call_llm(prompt, system_prompt)
            return response_text, scores
        except Exception as e:
            logger.error("Answer generation failed, returning error response", error=str(e))
            return "Error during response synthesis.", scores
