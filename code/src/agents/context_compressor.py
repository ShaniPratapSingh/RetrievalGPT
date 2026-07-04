import re
from typing import List, Dict, Any, Tuple
from src.core.observability import Logger

logger = Logger("agents_context_compressor")

class ContextCompressor:
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def compress(self, items: List[Tuple[Dict[str, Any], float]]) -> List[Tuple[Dict[str, Any], float]]:
        """Remove duplicate and low-information chunks from the matched contexts."""
        compressed = []
        seen_texts = []
        
        for chunk, score in items:
            text_cleaned = re.sub(r'\s+', '', chunk["text"].lower())
            
            # Drop low-information chunks
            alpha_chars = len(re.sub(r'[^a-z]', '', text_cleaned))
            if alpha_chars < 10:
                logger.info("Context compressor dropped low-information noise chunk", id=chunk.get("id"))
                continue
                
            is_duplicate = False
            for prev_text in seen_texts:
                if prev_text in text_cleaned or text_cleaned in prev_text:
                    is_duplicate = True
                    break
            if not is_duplicate:
                compressed.append((chunk, score))
                seen_texts.append(text_cleaned)
                
        return compressed
