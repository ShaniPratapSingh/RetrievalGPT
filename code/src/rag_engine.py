import os
import re
import json
import time
from typing import Tuple, List, Dict, Any
import numpy as np
from dotenv import load_dotenv
import requests


from src.core.storage import StorageManager
from src.core.retriever import HybridRetriever
from src.core.agent import QueryAnalysisAgent
from src.core.memory import ConversationalMemory
from src.core.citation import CitationEngine
from src.core.web_search import WebSearchClient
from src.core.guardrails import GuardrailsManager
from src.core.cache import RAGCache
from src.core.multimodal import MultiDocumentParser
from src.core.services.summarization_service import SummarizationService
from src.retrieval.query_router import QueryRouter
from src.core.observability import telemetry, Logger

logger = Logger("rag_engine")
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class RAGEngine:
    def __init__(self, local_model_name="all-MiniLM-L6-v2"):
        """
        Initialize the enterprise-grade RAG Engine.
        Uses local open-source sentence-transformers embeddings.
        """
        self.local_model_name = local_model_name
        self.documents = []  # List of dict: {id, source, text}
        self.chunks = []     # List of dict: {id, doc_id, source, text, embedding}
        
        # Load local embedding model
        self.local_model = None
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.local_model = SentenceTransformer(self.local_model_name)
            except Exception as e:
                logger.error("Failed to load local embedding model", name=self.local_model_name, error=str(e))
                self.local_model = None

        # Instantiate core subsystems
        self.storage = StorageManager()
        self.retriever = HybridRetriever(self.storage, self.get_embedding)
        self.agent = QueryAnalysisAgent(self._call_free_llm)
        self.guardrails = GuardrailsManager(self._call_free_llm)
        self.web_search = WebSearchClient()
        self.cache = RAGCache()
        self.summarizer = SummarizationService(self._call_free_llm)
        self.router = QueryRouter(self._call_free_llm, self.storage, self.get_embedding, self.retriever)
        
        # Load any existing database records into local lists for legacy compatibility
        self.sync_local_lists()
        self.retriever.rebuild_sparse_index()

    def sync_local_lists(self):
        """Sync core local list structures with ChromaDB for backward compatibility."""
        all_ch = self.storage.get_all_chunks()
        self.chunks = all_ch
        
        # Deduplicate docs from chunks
        seen_doc_ids = set()
        self.documents = []
        for ch in all_ch:
            doc_id = ch["doc_id"]
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                self.documents.append({
                    "id": doc_id,
                    "source": ch["source"],
                    "text": ""  # Lazy loaded/aggregated
                })

    def get_embedding(self, text: str) -> list:
        """Generate open-source embedding for a given text locally (with caching)."""
        text = text.replace("\n", " ").strip()
        
        # Check Cache first
        cached = self.cache.get_embedding(text)
        if cached:
            return cached
            
        emb = None
        # Try Local SentenceTransformers Embeddings
        if self.local_model:
            try:
                emb = self.local_model.encode(text)
                emb_list = emb.tolist() if hasattr(emb, "tolist") else list(emb)
                self.cache.set_embedding(text, emb_list)
                return emb_list
            except Exception as e:
                logger.error("Local embedding generation failed", error=str(e))
                
        # Fallback Mock Embeddings (simple hash-based unit vector to prevent crash)
        np.random.seed(hash(text) % (2**32))
        mock_vec = np.random.randn(384)
        mock_vec /= np.linalg.norm(mock_vec)
        emb_list = mock_vec.tolist()
        self.cache.set_embedding(text, emb_list)
        return emb_list

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Batch generate embeddings using local model or fallback."""
        if self.local_model:
            try:
                embs = self.local_model.encode(texts)
                return embs
            except Exception as e:
                logger.error("Batch embedding generation failed, falling back to sequential", error=str(e))
        import numpy as np
        return np.array([self.get_embedding(t) for t in texts])

    def load_document(self, file_path: str) -> int:
        """Load document, perform duplicate checking, and extract metadata."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        doc_hash = MultiDocumentParser.get_file_hash(file_path)
        
        # Duplicate detection check
        for doc in self.documents:
            if doc.get("hash") == doc_hash:
                logger.info("Duplicate document detected, bypassing ingestion", file=file_name)
                return doc["id"]
                
        ext = file_path.lower()
        content = ""
        # Route to appropriate parser
        if ext.endswith(".pdf"):
            pages = MultiDocumentParser.parse_pdf(file_path)
            content = "\n".join([p[0] for p in pages])
        elif ext.endswith(".docx"):
            pages = MultiDocumentParser.parse_docx(file_path)
            content = "\n".join([p[0] for p in pages])
        elif ext.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
            content = MultiDocumentParser.parse_image_ocr(file_path)
        elif ext.endswith((".txt", ".md", ".json", ".html")):
            pages = MultiDocumentParser.parse_text_or_markdown(file_path)
            content = "\n".join([p[0] for p in pages])
        else:
            raise ValueError(f"Unsupported file format for: {file_name}")
            
        doc_id = len(self.documents)
        meta = MultiDocumentParser.extract_metadata(file_path, content)
        
        self.documents.append({
            "id": doc_id,
            "source": file_name,
            "text": content,
            "hash": doc_hash,
            "metadata": meta
        })
        return doc_id

    def chunk_text(self, text: str, chunk_size=400, chunk_overlap=80) -> list:
        """Split text into manageable chunks with overlap."""
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        
        chunks = []
        i = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
            i += (chunk_size - chunk_overlap)
            if chunk_size <= chunk_overlap:
                break
        return chunks

    def index_document(self, doc_id: int, chunk_size=400, chunk_overlap=80) -> int:
        """Index a loaded document using SemanticChunker, score quality, and save to storage."""
        from src.core.splitter import SemanticChunker
        doc = self.documents[doc_id]
        
        # Initialize Semantic Chunker with RAGEngine batch embedding function
        chunker = SemanticChunker(embed_fn=self.get_embeddings_batch)
        raw_chunks = chunker.split_text(doc["text"])
        
        indexed_chunks = []
        for idx, text in enumerate(raw_chunks):
            embedding = self.get_embedding(text)
            # Fetch pre-indexed quality score
            quality_score = self.router.filter.get_quality_score(text)
            
            chunk_info = {
                "id": len(self.chunks) + idx,
                "doc_id": doc_id,
                "source": doc["source"],
                "text": text,
                "embedding": embedding,
                "page": (idx // 3) + 1,  # Page estimation
                "quality_score": quality_score
            }
            indexed_chunks.append(chunk_info)
            
        # Add to storage manager
        self.storage.add_chunks(indexed_chunks)
        self.sync_local_lists()
        
        # Rebuild BM25 index
        self.retriever.rebuild_sparse_index()
        return len(indexed_chunks)

    def retrieve(self, query: str, top_k=3) -> list:
        """Retrieve relevant chunks for a query using hybrid retrieval."""
        final_hits, needs_web = self.retriever.retrieve_hybrid(query, top_k=top_k)
        
        # Format results matching legacy [(chunk_dict, score)] format
        legacy_format = [(item[0], item[1]) for item in final_hits]
        return legacy_format

    def _call_free_llm(self, prompt: str, system_prompt=None, ollama_url=None, ollama_model=None) -> Tuple[str, str]:
        """
        Executes a prompt using the Free-Only LLM fallback chain.
        Supports caching.
        """
        # Check cache
        cache_key = f"system: {system_prompt or ''}\nprompt: {prompt}"
        cached_val = self.cache.get_completion(cache_key)
        if cached_val:
            return "Cached (Local)", cached_val

        # Load local keys / overrides
        ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1")
        google_key = os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        # 1. Try Ollama (Local)
        try:
            url = f"{ollama_url.rstrip('/')}/api/chat"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": ollama_model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2}
            }
            response = requests.post(url, json=payload, timeout=5)
            if response.status_code == 200:
                result = response.json()
                output = result["message"]["content"]
                self.cache.set_completion(cache_key, output)
                return "Ollama (Local)", output
        except Exception:
            pass

        # 2. Try Google Gemini Free Tier
        if google_key and google_key != "your_google_gemini_api_key_here":
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={google_key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2}
                }
                if system_prompt:
                    payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
                
                headers = {"Content-Type": "application/json"}
                response = requests.post(url, headers=headers, json=payload, timeout=8)
                if response.status_code == 200:
                    result = response.json()
                    output = result["candidates"][0]["content"]["parts"][0]["text"]
                    self.cache.set_completion(cache_key, output)
                    return "Gemini Free Tier (Cloud)", output
            except Exception:
                pass

        # 3. Try Groq Free Tier
        if groq_key and groq_key != "your_groq_api_key_here":
            try:
                url = "https://api.groq.com/openai/v1/chat/completions"
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                payload = {
                    "model": "llama-3.3-70b-versatile",
                    "messages": messages,
                    "temperature": 0.2
                }
                headers = {
                    "Authorization": f"Bearer {groq_key}",
                    "Content-Type": "application/json"
                }
                response = requests.post(url, headers=headers, json=payload, timeout=8)
                if response.status_code == 200:
                    result = response.json()
                    output = result["choices"][0]["message"]["content"]
                    self.cache.set_completion(cache_key, output)
                    return "Groq Free Tier (Cloud)", output
            except Exception:
                pass

        # 4. Try Hugging Face Free Inference API
        if hf_token and hf_token != "your_huggingface_token_here":
            try:
                url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"
                formatted_input = (f"<|system|>\n{system_prompt}\n" if system_prompt else "") + f"<|user|>\n{prompt}\n<|assistant|>\n"
                payload = {
                    "inputs": formatted_input,
                    "parameters": {"max_new_tokens": 512, "temperature": 0.2}
                }
                headers = {
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": "application/json"
                }
                response = requests.post(url, headers=headers, json=payload, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        output = result[0].get("generated_text", "")
                        if output.startswith(formatted_input):
                            output = output[len(formatted_input):].strip()
                        self.cache.set_completion(cache_key, output)
                        return "Hugging Face Hub (Cloud)", output
            except Exception:
                pass

        # 5. Mock Fallback
        mock_response = (
            "**[DEMO MODE: Offline & No API keys configured]**\n\n"
            "I parsed your query and simulated the RAG pipeline processing locally on your system."
        )
        return "Demo Fallback (Mock)", mock_response

    def rewrite_query(self, query: str, ollama_url=None, ollama_model=None) -> Tuple[str, str]:
        """Rewrite query to Boolean terms using the free LLM chain."""
        system_prompt = "You are a query rewriting expert. Your task is to create query terms for user query to find literature."
        prompt_content = f"""Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags.
<think>
[reasoning]
</think>
<answer>
{{
    "query": "...."
}}
</answer>

User Query: {query}
"""
        provider, response_text = self._call_free_llm(
            prompt=prompt_content,
            system_prompt=system_prompt,
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
        
        if "Demo Fallback" in provider:
            return "Demo fallback: query rewriting bypassed.", query
            
        thought, rewritten_query = self._extract_thought_and_query(response_text)
        thought = f"[Reasoned via {provider}]\n{thought}"
        return thought, rewritten_query

    def _extract_thought_and_query(self, response_text: str) -> Tuple[str, str]:
        thought = ""
        query = ""
        
        # Extract thought
        thought_match = re.search(r'<think>(.*?)(</think>|$)', response_text, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            parts = response_text.split("<answer>")
            if len(parts) > 1:
                thought = parts[0].strip()
                
        # Extract query
        answer_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            try:
                answer_json = json.loads(answer_content)
                query = answer_json.get("query", "")
            except:
                q_match = re.search(r'"query"\s*:\s*"(.*?)"', answer_content)
                if q_match:
                    query = q_match.group(1)
                else:
                    query = answer_content
        else:
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    answer_json = json.loads(json_match.group(0))
                    query = answer_json.get("query", "")
            except:
                pass
                
        if not query:
            query = response_text
            
        return thought or "Thinking complete.", query

    def generate_answer(self, query: str, context_chunks: list, chat_history=None, ollama_url=None, ollama_model=None) -> str:
        """Generate final response based on retrieved chunks and query, with citation formatting."""
        # Sanitize query
        query = self.guardrails.sanitize_input(query)
        is_inj, msg = self.guardrails.detect_prompt_injection(query)
        if is_inj:
            return msg

        # If context is low confidence, execute Web Search Fallback
        if not context_chunks:
            logger.info("Local context empty. Triggering Web Search Fallback...")
            web_hits = self.web_search.search(query)
            # Map web hits to context chunks structure
            context_chunks = [(hit, 1.0) for hit in web_hits]

        context_str = ""
        for idx, (chunk, score) in enumerate(context_chunks):
            context_str += f"[Source {idx+1}]: {chunk['source']}\nContent: {chunk['text']}\n\n"
            
        chat_history_str = ""
        if chat_history:
            for speaker, text in chat_history[-6:]:
                chat_history_str += f"{speaker}: {text}\n"

        system_prompt = (
            "You are a Staff AI Research Assistant. Answer the user's question accurately using ONLY "
            "the provided Source Documents. Write fluent, highly professional, human-readable answers. "
            "DO NOT under any circumstances mention RAG implementation details, 'retrieved chunks', "
            "'snippets', or 'the files provided'. If the source documents do not contain the answer, "
            "state that explicitly. Format citations clearly at the end of the text like:\n"
            "Source: [document_name]\n"
            "Page: [page_number]"
        )

        prompt = f"""Source Documents:
{context_str}

Chat History:
{chat_history_str}

User Question: {query}
Answer:"""

        provider, response_text = self._call_free_llm(
            prompt=prompt,
            system_prompt=system_prompt,
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
        
        if "Demo Fallback" in provider:
            mock_ans = (
                f"**[DEMO MODE: Offline & No API keys configured]**\n\n"
                f"The system processed your request locally. The matched source document segments indicate:\n\n"
            )
            for idx, (chunk, score) in enumerate(context_chunks):
                mock_ans += f"- **{chunk['source']}** (Page {chunk.get('page', 1)}):\n  \"{chunk['text'][:160]}...\"\n\n"
            return mock_ans
            
        # Run Hallucination groundedness check
        chunks_only = [item[0] for item in context_chunks]
        score, is_hallucinating = self.guardrails.evaluate_groundedness(response_text, chunks_only)
        if is_hallucinating:
            return "I cannot confidently answer this based on the available information as it seems to contain unsupported claims."

        # Run Citation grounding engine
        clean_answer, citations = CitationEngine.extract_citations(response_text, chunks_only)
        
        # Save output in global telemetry
        telemetry.record_provider(provider)
        
        # Append references to bottom of output for rendering in Streamlit
        footer = f"\n\n*— Generated via {provider}*"
        return f"{clean_answer}{footer}"

    def summarize_active_document(self, mode: str = "short", doc_name: str = None) -> dict:
        """Summarize the active document or a document specified by doc_name."""
        if not self.documents:
            return {
                "summary": "No documents uploaded. Please upload a document to summarize.",
                "summary_type": mode,
                "document_name": "None",
                "pages_processed": 0,
                "confidence": 0.0
            }
            
        # If doc_name is provided, search for it
        target_doc = None
        if doc_name:
            for doc in self.documents:
                if doc["source"].lower() == doc_name.lower():
                    target_doc = doc
                    break
                    
        # Fallback to the last uploaded document
        if not target_doc:
            target_doc = self.documents[-1]
            
        return self.summarizer.summarize_document(target_doc["text"], target_doc["source"], mode)

