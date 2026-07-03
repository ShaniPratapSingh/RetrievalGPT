import os
import chromadb
from typing import Dict, List, Any, Optional
from src.core.observability import Logger

logger = Logger("storage")

class StorageManager:
    def __init__(self, db_dir: str = "chroma_db"):
        self.db_dir = os.path.abspath(db_dir)
        os.makedirs(self.db_dir, exist_ok=True)
        
        logger.info("Initializing ChromaDB", path=self.db_dir)
        try:
            self.client = chromadb.PersistentClient(path=self.db_dir)
            self.collection = self.client.get_or_create_collection("retrieval_gpt_chunks")
            self.db_enabled = True
            logger.info("ChromaDB persistent client loaded successfully")
        except Exception as e:
            logger.error("ChromaDB initialization failed, falling back to in-memory dictionary", error=str(e))
            self.db_enabled = False
            self.in_memory_docs = []
            self.in_memory_chunks = []

    def clear_database(self):
        """Clears all records from storage."""
        if self.db_enabled:
            try:
                self.client.delete_collection("retrieval_gpt_chunks")
                self.collection = self.client.create_collection("retrieval_gpt_chunks")
                logger.info("ChromaDB collection cleared")
            except Exception as e:
                logger.error("Failed to clear ChromaDB collection", error=str(e))
        else:
            self.in_memory_docs = []
            self.in_memory_chunks = []
            logger.info("In-memory storage cleared")

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Adds a list of pre-generated chunk dicts (containing text, embedding, metadata) to ChromaDB."""
        if not chunks:
            return
            
        if self.db_enabled:
            ids = [f"chk_{c['id']}" for c in chunks]
            documents = [c["text"] for c in chunks]
            embeddings = [c["embedding"] for c in chunks]
            
            # Extract metadatas, converting nested objects or non-supported types to strings
            metadatas = []
            for c in chunks:
                meta = {
                    "doc_id": c.get("doc_id", 0),
                    "source": c.get("source", "unknown"),
                    "chunk_idx": c.get("id", 0),
                    "page": c.get("page", 1)
                }
                metadatas.append(meta)
                
            try:
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )
                logger.info("Saved chunks to ChromaDB", count=len(chunks))
            except Exception as e:
                logger.error("Failed to save chunks to ChromaDB", error=str(e))
        else:
            # InMemory fallback
            for c in chunks:
                self.in_memory_chunks.append(c)
            logger.info("Saved chunks to in-memory store", count=len(chunks))

    def get_all_chunks(self) -> List[Dict[str, Any]]:
        """Returns all chunks stored in database/in-memory."""
        if self.db_enabled:
            try:
                results = self.collection.get(include=["documents", "metadatas", "embeddings"])
                chunks = []
                if results and "ids" in results:
                    for i in range(len(results["ids"])):
                        meta = results["metadatas"][i]
                        # Reconstruct the dict format expected by RAGEngine
                        chunks.append({
                            "id": meta.get("chunk_idx", i),
                            "doc_id": meta.get("doc_id", 0),
                            "source": meta.get("source", "unknown"),
                            "text": results["documents"][i],
                            "embedding": results["embeddings"][i] if results.get("embeddings") is not None else None,
                            "page": meta.get("page", 1)
                        })
                return chunks
            except Exception as e:
                logger.error("Failed to query all chunks from ChromaDB", error=str(e))
                return []
        else:
            return self.in_memory_chunks

    def query_similarity(self, query_emb: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """Query Chroma for the top K closest chunks by cosine/l2 similarity."""
        if self.db_enabled:
            try:
                results = self.collection.query(
                    query_embeddings=[query_emb],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                
                formatted = []
                if results and "ids" in results and results["ids"][0]:
                    for idx in range(len(results["ids"][0])):
                        meta = results["metadatas"][0][idx]
                        dist = results["distances"][0][idx]
                        # Convert distance to cosine similarity (Chroma returns squared L2 distance or cosine distance depending on config, default is L2)
                        # We convert distance to similarity score
                        similarity = 1.0 / (1.0 + dist)
                        
                        formatted.append({
                            "chunk": {
                                "id": meta.get("chunk_idx", 0),
                                "doc_id": meta.get("doc_id", 0),
                                "source": meta.get("source", "unknown"),
                                "text": results["documents"][0][idx],
                                "page": meta.get("page", 1)
                            },
                            "score": float(similarity)
                        })
                return formatted
            except Exception as e:
                logger.error("Querying similarity from ChromaDB failed", error=str(e))
                return []
        else:
            return []
