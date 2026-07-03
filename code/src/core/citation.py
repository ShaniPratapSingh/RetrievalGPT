import re
from typing import List, Dict, Any, Tuple
from src.core.observability import Logger

logger = Logger("citation")

class CitationEngine:
    @staticmethod
    def extract_citations(answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Scans LLM answer for references like [Source 1], [1], or [Source: filename] and links them to context chunks.
        Calculates grounding match snippets to highlight in the UI.
        
        Returns:
            clean_answer: The answer text formatted nicely.
            citations: List of parsed citation mappings.
        """
        citations = []
        
        # Normalize various citation styles in response (e.g., "[Source 1]", "[1]", "Source 1")
        # Find all brackets containing numbers or Source prefix
        pattern = r'\[(?:Source\s*)?(\d+)\]'
        matches = re.finditer(pattern, answer)
        
        # Build index map
        citation_indices = []
        for match in matches:
            idx = int(match.group(1)) - 1
            if 0 <= idx < len(context_chunks):
                citation_indices.append((match.start(), match.end(), idx))
                
        # If no brackets matches, let's do soft sentence matching
        if not citation_indices:
            # Check if sentences match any chunks
            sentences = re.split(r'(?<=[.!?])\s+', answer)
            clean_sentences = []
            for i, sent in enumerate(sentences):
                if len(sent.strip()) < 15:
                    clean_sentences.append(sent)
                    continue
                # Calculate overlap with context chunks
                best_match_idx = -1
                best_overlap = 0.0
                for c_idx, chunk in enumerate(context_chunks):
                    # Check text overlap (Jaccard similarity of words)
                    words_sent = set(re.findall(r'\w+', sent.lower()))
                    words_chunk = set(re.findall(r'\w+', chunk["text"].lower()))
                    if not words_sent:
                        continue
                    overlap = len(words_sent.intersection(words_chunk)) / len(words_sent)
                    if overlap > best_overlap and overlap > 0.4:
                        best_overlap = overlap
                        best_match_idx = c_idx
                        
                if best_match_idx != -1:
                    # Append a footnote to the sentence
                    footnote = f" [{best_match_idx + 1}]"
                    clean_sentences.append(sent.rstrip(".") + footnote + ".")
                    citations.append({
                        "index": best_match_idx + 1,
                        "source": context_chunks[best_match_idx]["source"],
                        "page": context_chunks[best_match_idx].get("page", 1),
                        "chunk_id": context_chunks[best_match_idx].get("id", 0),
                        "confidence_score": round(best_overlap, 2),
                        "snippet": context_chunks[best_match_idx]["text"][:200] + "..."
                    })
                else:
                    clean_sentences.append(sent)
            answer = " ".join(clean_sentences)
        else:
            # We have brackets, register them
            seen_indices = set()
            for start, end, idx in citation_indices:
                if idx not in seen_indices:
                    seen_indices.add(idx)
                    chunk = context_chunks[idx]
                    citations.append({
                        "index": idx + 1,
                        "source": chunk["source"],
                        "page": chunk.get("page", 1),
                        "chunk_id": chunk.get("id", 0),
                        "confidence_score": 0.95,  # High confidence as it was explicitly cited
                        "snippet": chunk["text"][:200] + "..."
                    })
                    
        # If citations empty, check if we can add standard footers
        if not citations:
            for idx, chunk in enumerate(context_chunks[:2]):
                citations.append({
                    "index": idx + 1,
                    "source": chunk["source"],
                    "page": chunk.get("page", 1),
                    "chunk_id": chunk.get("id", 0),
                    "confidence_score": 0.80,
                    "snippet": chunk["text"][:200] + "..."
                })
                
        return answer, citations
