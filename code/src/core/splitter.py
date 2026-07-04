import re
import numpy as np
from typing import List, Callable, Optional
from src.core.observability import Logger

logger = Logger("splitter")

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Recursively split text by separators until chunks are within chunk_size."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        separator = separators[0]
        splits = text.split(separator)
        
        if len(splits) == 1:
            return self._split_text(text, separators[1:])

        chunks = []
        current_doc = []
        current_len = 0
        
        for split in splits:
            if current_len + len(split) + (len(separator) if current_doc else 0) <= self.chunk_size:
                current_doc.append(split)
                current_len += len(split) + (len(separator) if len(current_doc) > 1 else 0)
            else:
                if current_doc:
                    chunks.append(separator.join(current_doc))
                
                overlap_doc = []
                overlap_len = 0
                for item in reversed(current_doc):
                    if overlap_len + len(item) + (len(separator) if overlap_doc else 0) <= self.chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += len(item) + (len(separator) if len(overlap_doc) > 1 else 0)
                    else:
                        break
                
                if len(split) > self.chunk_size:
                    sub_splits = self._split_text(split, separators[1:])
                    if overlap_doc:
                        prefix = separator.join(overlap_doc)
                        if len(prefix) + len(sub_splits[0]) <= self.chunk_size:
                            sub_splits[0] = prefix + separator + sub_splits[0]
                    chunks.extend(sub_splits[:-1])
                    current_doc = [sub_splits[-1]]
                    current_len = len(sub_splits[-1])
                else:
                    current_doc = overlap_doc + [split]
                    current_len = overlap_len + len(split) + (len(separator) if len(current_doc) > 1 else 0)
                    
        if current_doc:
            chunks.append(separator.join(current_doc))
            
        return chunks


class SemanticChunker:
    """Chunks text semantically by calculating distance between consecutive sentence embeddings."""
    def __init__(self, embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None, similarity_threshold: float = 0.72, max_chunk_size: int = 1500):
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split sentences while keeping punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def split_text(self, text: str) -> List[str]:
        """Groups sentences semantically using consecutive cosine similarities."""
        if not self.embed_fn:
            logger.warn("No embedding function provided. Falling back to RecursiveCharacterTextSplitter.")
            return self.fallback_splitter.split_text(text)

        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return sentences

        try:
            # Embed all sentences
            embeddings = self.embed_fn(sentences)
            if len(embeddings) != len(sentences):
                raise ValueError("Embedding count mismatch with sentences")

            chunks = []
            current_chunk_sentences = [sentences[0]]
            
            for idx in range(1, len(sentences)):
                # Calculate Cosine Similarity between consecutive sentences
                vec1 = embeddings[idx - 1]
                vec2 = embeddings[idx]
                
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                
                if norm1 == 0 or norm2 == 0:
                    sim = 0.0
                else:
                    sim = float(np.dot(vec1, vec2) / (norm1 * norm2))

                # Check dynamic parameters and limits
                current_size = len(" ".join(current_chunk_sentences))
                if sim >= self.similarity_threshold and current_size + len(sentences[idx]) <= self.max_chunk_size:
                    current_chunk_sentences.append(sentences[idx])
                else:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentences[idx]]
            
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                
            return chunks
        except Exception as e:
            logger.error("Semantic chunking failed, falling back to character splits", error=str(e))
            return self.fallback_splitter.split_text(text)
