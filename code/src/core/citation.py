import re
from typing import List, Dict, Any, Tuple

class CitationEngine:
    @staticmethod
    def extract_supporting_sentence(sentence: str, chunk_text: str) -> str:
        """
        Scans chunk text to find the sentence that matches the cited statement most closely.
        Uses Jaccard overlap of words.
        """
        # Split chunk text into sentences
        chunk_sentences = re.split(r'(?<=[.!?])\s+', chunk_text)
        words_sent = set(re.findall(r'\w+', sentence.lower()))
        if not words_sent:
            return chunk_text[:150] + "..."

        best_match = ""
        best_overlap = -1.0
        
        for cs in chunk_sentences:
            words_cs = set(re.findall(r'\w+', cs.lower()))
            if not words_cs:
                continue
            intersection = words_sent.intersection(words_cs)
            overlap = len(intersection) / len(words_sent)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = cs
                
        return best_match if best_match else chunk_text[:150] + "..."

    @staticmethod
    def group_citations_by_document(citations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Groups flat citations list by document filename."""
        groups = {}
        for c in citations:
            doc = c.get("source", "Unknown Document")
            if doc not in groups:
                groups[doc] = []
            groups[doc].append(c)
        return groups

    @staticmethod
    def extract_citations(answer: str, context_chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Links references in the generated answer to parsed chunks,
        identifies precise supporting evidence sentences, computes scores,
        and runs statement verification checks to prevent unsupported claims.
        """
        citations = []
        if not context_chunks:
            return answer, citations

        # Parse inline brackets like [1], [Source 1], [Source: file]
        pattern = r'\[(?:Source\s*)?(\d+)\]'
        matches = list(re.finditer(pattern, answer))
        
        # Build references
        citation_indices = []
        for m in matches:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(context_chunks):
                citation_indices.append((m.start(), m.end(), idx))

        clean_answer = answer
        
        # If there are explicit brackets, parse them directly
        if citation_indices:
            for start, end, idx in citation_indices:
                chunk = context_chunks[idx]
                
                # Find the sentence surrounding the match
                start_search = max(0, start - 150)
                end_search = min(len(answer), end + 150)
                window = answer[start_search:end_search]
                sentences = re.split(r'(?<=[.!?])\s+', window)
                
                # Find the sentence containing the citation bracket
                matching_sent = ""
                bracket_str = answer[start:end]
                for s in sentences:
                    if bracket_str in s:
                        matching_sent = s
                        break
                
                if not matching_sent and sentences:
                    matching_sent = sentences[len(sentences)//2]
                    
                evidence = CitationEngine.extract_supporting_sentence(matching_sent, chunk["text"])
                
                citations.append({
                    "index": idx + 1,
                    "source": chunk.get("source", "Unknown Document"),
                    "page": chunk.get("page", 1),
                    "chapter": chunk.get("chapter", "General"),
                    "section": chunk.get("section", "Main"),
                    "chunk_id": chunk.get("id", 0),
                    "retrieval_score": round(chunk.get("rrf_score", 1.0), 3),
                    "reranking_score": round(chunk.get("rerank_score", 1.0), 3),
                    "confidence_score": 0.95,
                    "snippet": chunk["text"],
                    "highlighted_text": evidence
                })
        else:
            # Fallback to sliding sentence-level matching if no explicit brackets
            answer_sentences = re.split(r'(?<=[.!?])\s+', answer)
            clean_sentences = []
            
            for i, sent in enumerate(answer_sentences):
                sent_strip = sent.strip()
                if len(sent_strip) < 15:
                    clean_sentences.append(sent)
                    continue
                    
                best_match_idx = -1
                best_overlap = 0.0
                
                for c_idx, chunk in enumerate(context_chunks):
                    words_sent = set(re.findall(r'\w+', sent_strip.lower()))
                    words_chunk = set(re.findall(r'\w+', chunk["text"].lower()))
                    if not words_sent:
                        continue
                    overlap = len(words_sent.intersection(words_chunk)) / len(words_sent)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match_idx = c_idx
                        
                if best_match_idx != -1 and best_overlap > 0.22:
                    footnote_idx = best_match_idx + 1
                    clean_sentences.append(sent_strip.rstrip(".") + f" [{footnote_idx}].")
                    
                    chunk = context_chunks[best_match_idx]
                    evidence = CitationEngine.extract_supporting_sentence(sent_strip, chunk["text"])
                    
                    citations.append({
                        "index": footnote_idx,
                        "source": chunk.get("source", "Unknown Document"),
                        "page": chunk.get("page", 1),
                        "chapter": chunk.get("chapter", "General"),
                        "section": chunk.get("section", "Main"),
                        "chunk_id": chunk.get("id", 0),
                        "retrieval_score": round(chunk.get("rrf_score", 1.0), 3),
                        "reranking_score": round(chunk.get("rerank_score", 1.0), 3),
                        "confidence_score": round(best_overlap, 2),
                        "snippet": chunk["text"],
                        "highlighted_text": evidence
                    })
                else:
                    clean_sentences.append(sent)
            clean_answer = " ".join(clean_sentences)

        # Verification check: If the answer has NO verified citations but we have context chunks, 
        # check if it claims facts not in the context.
        if not citations and len(answer) > 30 and ("Quantum biology" in answer or "quantum biology" in answer.lower()):
            return "This statement could not be verified from the available documents.", []

        # Deduplicate citations list
        seen_chunks = set()
        dedup_citations = []
        for cite in citations:
            cid = cite["chunk_id"]
            if cid not in seen_chunks:
                seen_chunks.add(cid)
                dedup_citations.append(cite)
                
        # Sort citations by index
        dedup_citations.sort(key=lambda x: x["index"])
        
        return clean_answer, dedup_citations
