import re
from typing import List, Dict, Any
from src.core.observability import Logger

logger = Logger("chunk_filter")

class ChunkFilter:
    def __init__(self, min_quality_score: float = 0.35):
        self.min_quality_score = min_quality_score

    def get_quality_score(self, text: str) -> float:
        """Evaluate text and compute a quality score between 0.0 and 1.0."""
        text_lower = text.lower().strip()
        if not text_lower:
            return 0.0
            
        score = 0.5 # Start at base score
        
        # 1. Negative factors (down-weights / filters)
        
        # Bibliography / References
        if any(w in text_lower for w in ["bibliography", "references", "cited by", "literature cited"]):
            score -= 0.3
            
        # Table of contents / index
        if any(w in text_lower for w in ["table of contents", "contents", "index page", "subject index"]):
            score -= 0.3
            
        # Copyright / Publisher metadata
        if any(w in text_lower for w in ["copyright", "all rights reserved", "isbn", "published by"]):
            score -= 0.2
            
        # Figure captions
        if re.search(r"fig(?:ure)?\s*\d+", text_lower):
            score -= 0.1
            
        # Math equations density
        # If there are many operators/symbols (+, =, \alpha, etc.) relative to clean words
        words = re.findall(r'\w+', text_lower)
        symbols = re.findall(r'[=+/*\\{}#_<>]', text_lower)
        if len(words) > 0:
            sym_ratio = len(symbols) / len(words)
            if sym_ratio > 0.15:
                # Highly mathematical or code-heavy chunk, down-weight
                score -= 0.25
                
        # Non-alphabetic lines count (e.g. page fragments or lists of page numbers)
        lines = text.split("\n")
        noisy_lines = 0
        for line in lines:
            line_clean = line.strip()
            if line_clean and not re.search(r"[a-zA-Z]{3,}", line_clean):
                noisy_lines += 1
        if len(lines) > 0 and (noisy_lines / len(lines)) > 0.4:
            score -= 0.2
            
        # 2. Positive factors (priority concepts)
        
        # Definitions & Explanations
        definition_indicators = [
            r"is defined as", r"refers to", r"represents the", r"means that",
            r"because", r"consequently", r"for example", r"such as", r"specifically"
        ]
        for pattern in definition_indicators:
            if re.search(pattern, text_lower):
                score += 0.08
                
        # Chapter intros & Conclusions
        intro_concl_indicators = [
            r"in this chapter", r"we introduce", r"to summarize", r"in conclusion",
            r"finally", r"overall", r"key concepts discussed", r"the main objective"
        ]
        for pattern in intro_concl_indicators:
            if re.search(pattern, text_lower):
                score += 0.1
                
        # Ensure score boundaries
        return max(0.0, min(1.0, round(score, 2)))

    def filter_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score each chunk and filter out chunks falling below the min quality threshold."""
        filtered = []
        for chunk in chunks:
            text = chunk.get("text", "")
            q_score = self.get_quality_score(text)
            
            # Attach score for metadata observability
            chunk["quality_score"] = q_score
            
            if q_score >= self.min_quality_score:
                filtered.append(chunk)
            else:
                logger.info("Filtered out noisy/low-quality chunk", chunk_idx=chunk.get("id"), score=q_score, source=chunk.get("source"))
                
        return filtered
