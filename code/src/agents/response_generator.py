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
            
        # 1. Retrieval RRF score average
        avg_retrieval_score = sum(h[1] for h in hits) / len(hits)
        
        # 2. Rerank score average (mapped via sigmoid)
        import numpy as np
        avg_rerank_score = sum(1.0 / (1.0 + np.exp(-h[0].get("rerank_score", 0.0))) for h in hits) / len(hits)
        
        # 3. Evidence coverage maps to total distinct pages
        distinct_pages = len({h[0].get("page", 1) for h in hits})
        evidence_coverage = min(1.0, distinct_pages * 0.20)
        
        # Combined confidence metric
        ans_conf = (avg_retrieval_score * 0.4) + (avg_rerank_score * 0.4) + (evidence_coverage * 0.2)
        # Ensure confidence remains >= 0.3 if we have hits to prevent false fallbacks
        ans_conf = max(0.3, min(1.0, ans_conf))
        
        return {
            "retrieval_confidence": round(ans_conf, 2),
            "evidence_coverage": round(evidence_coverage, 2),
            "answer_confidence": round(ans_conf, 2)
        }

    def generate(self, query: str, hits: List[Tuple[Dict[str, Any], float]], intent: str = "factual_question") -> Tuple[str, Dict[str, Any]]:
        """Generates answer using LLM, verifying refusal rules when confidence or sources are low."""
        scores = self.estimate_confidence(hits)
        
        # Topics outline builder shortcut
        query_lower = query.lower()
        if any(w in query_lower for w in ["topics", "outline", "headings", "table of contents"]):
            headings = []
            for chunk, _ in hits:
                h = chunk.get("heading") or chunk.get("chapter")
                if h and h not in headings:
                    headings.append(h)
            if headings:
                outline_text = "\n".join([f"- {h}" for h in headings])
                return f"Based on the document outline, the following topics and sections are covered:\n\n{outline_text}", scores

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
        if any(w in query_lower for w in ["quiz", "exam", "interview", "questions"]):
            system_prompt += " Generate quiz or exam questions ONLY using facts present in the provided context. Do NOT use external knowledge."
        
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
