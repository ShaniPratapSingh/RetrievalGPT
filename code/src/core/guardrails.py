import re
import json
from typing import List, Dict, Any, Tuple
from src.core.observability import Logger

logger = Logger("guardrails")

class GuardrailsManager:
    def __init__(self, call_llm_fn=None):
        self.call_llm = call_llm_fn

    def sanitize_input(self, user_query: str) -> str:
        """Strip dangerous HTML/JS tags and check for prompt injection patterns."""
        # Simple HTML sanitization
        clean = re.sub(r'<[^>]*>', '', user_query)
        return clean.strip()

    def detect_prompt_injection(self, user_query: str) -> Tuple[bool, str]:
        """Scans the user query for classic prompt injection keywords/phrases."""
        dangerous_patterns = [
            r"ignore\s+(?:all\s+)?prior\s+instructions",
            r"ignore\s+previous",
            r"system\s+prompt",
            r"jailbreak",
            r"you\s+must\s+now\s+act\s+as",
            r"bypass\s+restrictions",
            r"do\s+anything\s+now"
        ]
        
        query_lower = user_query.lower()
        for pattern in dangerous_patterns:
            if re.search(pattern, query_lower):
                logger.warn("Potential prompt injection attempt blocked", query=user_query)
                return True, "Jailbreak/injection attempt detected and blocked."
                
        return False, ""

    def evaluate_groundedness(self, answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[float, bool]:
        """
        Verify if the generated answer is grounded in the retrieved context chunks.
        Returns: (groundedness_score, is_hallucinating)
        """
        if not context_chunks:
            return 0.0, True
            
        if not self.call_llm:
            # Local fallback rule check:
            # Simple check if some nouns/main words in the answer exist in the context text
            answer_words = set(re.findall(r'\w+', answer.lower()))
            # Remove small stopwords
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "of", "to", "for", "with", "by", "that", "this", "it", "on"}
            meaningful_words = answer_words - stopwords
            
            context_text = " ".join([c["text"].lower() for c in context_chunks])
            if not meaningful_words:
                return 1.0, False
                
            matches = sum(1 for w in meaningful_words if w in context_text)
            score = matches / len(meaningful_words)
            return round(score, 2), score < 0.3
            
        context_str = ""
        for idx, c in enumerate(context_chunks):
            context_str += f"[Doc {idx+1}]: {c['text']}\n\n"
            
        system_prompt = (
            "You are a strict Groundedness Evaluator. Your job is to verify if the Assistant Answer is strictly "
            "supported by the provided Source Documents. Output your response in JSON format within <answer> </answer> tags."
        )
        
        prompt = f"""Source Documents:
{context_str}

Assistant Answer:
{answer}

Evaluate if there are any hallucinated, fabricated, or unsupported statements in the Assistant Answer.
Output your evaluation as JSON inside <answer> </answer> tags:
<answer>
{{
    "score": 0.0 to 1.0 (1.0 means fully grounded, 0.0 means completely hallucinated),
    "hallucinated_claims": ["list of claims that are unsupported by the Source Documents"]
}}
</answer>
"""
        try:
            provider, response_text = self.call_llm(prompt, system_prompt)
            ans_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
            if ans_match:
                result = json.loads(ans_match.group(1).strip())
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON payload found")
                    
            score = float(result.get("score", 1.0))
            is_hallucinating = score < 0.6
            logger.info("Groundedness evaluated", score=score, is_hallucinating=is_hallucinating)
            return score, is_hallucinating
        except Exception as e:
            logger.warn("LLM-based groundedness check failed. Using lexical fallback.", error=str(e))
            # Lexical fallback check
            answer_words = set(re.findall(r'\w+', answer.lower()))
            stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "of", "to", "for", "with", "by", "that", "this", "it", "on"}
            meaningful_words = answer_words - stopwords
            context_text = " ".join([c["text"].lower() for c in context_chunks])
            if not meaningful_words:
                return 1.0, False
            matches = sum(1 for w in meaningful_words if w in context_text)
            score = matches / len(meaningful_words)
            return round(score, 2), score < 0.35
