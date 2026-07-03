from typing import List

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Recursively split text by separators until chunks are within chunk_size."""
        return self._split_text(text, self.separators)

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        # If text is already smaller than chunk_size, return it
        if len(text) <= self.chunk_size:
            return [text]

        # Find the first separator that splits the text
        if not separators:
            # No separators left, force chunk split
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

        separator = separators[0]
        splits = text.split(separator)
        
        # If separator didn't do any splits, try next separator
        if len(splits) == 1:
            return self._split_text(text, separators[1:])

        # Merge splits into chunks of size <= chunk_size with overlap
        chunks = []
        current_doc = []
        current_len = 0
        
        for split in splits:
            if current_len + len(split) + (len(separator) if current_doc else 0) <= self.chunk_size:
                current_doc.append(split)
                current_len += len(split) + (len(separator) if len(current_doc) > 1 else 0)
            else:
                # Store the current chunk if it has content
                if current_doc:
                    chunks.append(separator.join(current_doc))
                
                # Setup overlap for next chunk
                # Find how many elements from current_doc we can include as overlap
                overlap_doc = []
                overlap_len = 0
                for item in reversed(current_doc):
                    if overlap_len + len(item) + (len(separator) if overlap_doc else 0) <= self.chunk_overlap:
                        overlap_doc.insert(0, item)
                        overlap_len += len(item) + (len(separator) if len(overlap_doc) > 1 else 0)
                    else:
                        break
                
                # Check if split itself is larger than chunk_size, if so split recursively
                if len(split) > self.chunk_size:
                    sub_splits = self._split_text(split, separators[1:])
                    # If we had overlap, prepend it to first sub-split if possible
                    if overlap_doc:
                        prefix = separator.join(overlap_doc)
                        if len(prefix) + len(sub_splits[0]) <= self.chunk_size:
                            sub_splits[0] = prefix + separator + sub_splits[0]
                    chunks.extend(sub_splits[:-1])
                    # Setup current state for the last sub_split
                    current_doc = [sub_splits[-1]]
                    current_len = len(sub_splits[-1])
                else:
                    current_doc = overlap_doc + [split]
                    current_len = overlap_len + len(split) + (len(separator) if len(current_doc) > 1 else 0)
                    
        if current_doc:
            chunks.append(separator.join(current_doc))
            
        return chunks
