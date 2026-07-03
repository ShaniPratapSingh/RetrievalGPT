import os
import re
import json
import time
import numpy as np
from pypdf import PdfReader
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Sentence Transformers for local embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class RAGEngine:
    def __init__(self, local_model_name="all-MiniLM-L6-v2"):
        """
        Initialize the RAG Engine.
        Always uses local open-source sentence-transformers embeddings.
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
                print(f"Failed to load local embedding model {self.local_model_name}: {e}")
                self.local_model = None

    def get_embedding(self, text):
        """Generate open-source embedding for a given text locally."""
        text = text.replace("\n", " ").strip()
        
        # Try Local SentenceTransformers Embeddings
        if self.local_model:
            try:
                emb = self.local_model.encode(text)
                return emb.tolist() if hasattr(emb, "tolist") else list(emb)
            except Exception as e:
                print(f"Local embedding generation failed: {e}")
                
        # Fallback Mock Embeddings (simple hash-based unit vector to prevent crash)
        np.random.seed(hash(text) % (2**32))
        mock_vec = np.random.randn(384)
        mock_vec /= np.linalg.norm(mock_vec)
        return mock_vec.tolist()

    def load_document(self, file_path):
        """Load text or PDF documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        content = ""
        
        if file_path.lower().endswith(".pdf"):
            try:
                reader = PdfReader(file_path)
                for page_num, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    content += text + "\n"
            except Exception as e:
                raise ValueError(f"Failed to parse PDF {file_name}: {e}")
        elif file_path.lower().endswith((".txt", ".md", ".json")):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception as e:
                raise ValueError(f"Failed to read file {file_name}: {e}")
        else:
            raise ValueError(f"Unsupported file format for: {file_name}")
            
        doc_id = len(self.documents)
        self.documents.append({
            "id": doc_id,
            "source": file_name,
            "text": content
        })
        return doc_id

    def chunk_text(self, text, chunk_size=400, chunk_overlap=80):
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

    def index_document(self, doc_id, chunk_size=400, chunk_overlap=80):
        """Index a loaded document into chunks and generate embeddings."""
        doc = self.documents[doc_id]
        raw_chunks = self.chunk_text(doc["text"], chunk_size, chunk_overlap)
        
        indexed_chunks = []
        for idx, text in enumerate(raw_chunks):
            embedding = self.get_embedding(text)
            chunk_info = {
                "id": len(self.chunks),
                "doc_id": doc_id,
                "source": doc["source"],
                "text": text,
                "embedding": embedding
            }
            self.chunks.append(chunk_info)
            indexed_chunks.append(chunk_info)
            
        return len(indexed_chunks)

    def retrieve(self, query, top_k=3):
        """Retrieve relevant chunks for a query using cosine similarity."""
        if not self.chunks:
            return []
            
        query_emb = np.array(self.get_embedding(query))
        
        scores = []
        for chunk in self.chunks:
            chunk_emb = np.array(chunk["embedding"])
            # Cosine similarity
            dot_product = np.dot(query_emb, chunk_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_c = np.linalg.norm(chunk_emb)
            similarity = dot_product / (norm_q * norm_c) if (norm_q > 0 and norm_c > 0) else 0.0
            scores.append((chunk, similarity))
            
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _call_free_llm(self, prompt, system_prompt=None, ollama_url=None, ollama_model=None):
        """
        Executes a prompt using the Free-Only LLM fallback chain.
        1. Ollama (local)
        2. Gemini Free Tier (cloud)
        3. Groq Free Tier (cloud)
        4. Hugging Face Inference API (cloud)
        5. Mock Fallback
        
        Returns: (resolved_provider, response_text)
        """
        # Load local keys / overrides
        ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "llama3.1")
        google_key = os.getenv("GOOGLE_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

        # 1. Try Ollama (Local)
        try:
            # Query the chat endpoint
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
                return "Ollama (Local)", result["message"]["content"]
        except Exception as e:
            # Ollama not running or timeout, proceed to next
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
                    text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return "Gemini Free Tier (Cloud)", text
            except Exception as e:
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
                    text = result["choices"][0]["message"]["content"]
                    return "Groq Free Tier (Cloud)", text
            except Exception as e:
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
                    # Some HF responses are list of dict
                    if isinstance(result, list) and len(result) > 0:
                        text = result[0].get("generated_text", "")
                        # Remove prompt if model returned the whole transcript
                        if text.startswith(formatted_input):
                            text = text[len(formatted_input):].strip()
                        return "Hugging Face Hub (Cloud)", text
            except Exception as e:
                pass

        # 5. Mock/Demo Fallback mode
        mock_response = (
            "**[DEMO MODE: Offline & No API keys configured]**\n\n"
            "I parsed your query and successfully simulated the RAG pipeline processing locally on your system."
        )
        return "Demo Fallback (Mock)", mock_response

    def rewrite_query(self, query, ollama_url=None, ollama_model=None):
        """
        Rewrite query to Boolean terms using the free LLM chain.
        """
        system_prompt = "You are a query rewriting expert. Your task is to create query terms for user query to find relevant literature in a massive corpus."
        
        prompt_content = f"""Show your work in <think> </think> tags. Your final response must be in JSON format within <answer> </answer> tags. For example,
<think>
[thinking process]
</think>
<answer>
{{
    "query": "...."
}} 
</answer>. 
Note: The query should use Boolean operators (AND, OR) and parentheses for grouping terms appropriately. You don't need to rewrite the query when the query is already good.

Here's the user query:
{query}
Assistant: Let me rewrite the query with reasoning. 
<think>
"""
        provider, response_text = self._call_free_llm(
            prompt=prompt_content,
            system_prompt=system_prompt,
            ollama_url=ollama_url,
            ollama_model=ollama_model
        )
        
        if "Demo Fallback" in provider:
            # Return raw query in mock mode
            return "Demo fallback: query rewriting bypassed.", query
            
        thought, rewritten_query = self._extract_thought_and_query(response_text)
        # Append provider name to thought for transparency
        thought = f"[Reasoned via {provider}]\n{thought}"
        return thought, rewritten_query

    def _extract_thought_and_query(self, response_text):
        """Helper to parse <think> and <answer> blocks from generator output."""
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

    def generate_answer(self, query, context_chunks, chat_history=None, ollama_url=None, ollama_model=None):
        """Generate final response based on retrieved chunks and query using free chain."""
        context_str = ""
        for idx, (chunk, score) in enumerate(context_chunks):
            context_str += f"[Source {idx+1}]: {chunk['source']}\nContent: {chunk['text']}\n\n"
            
        chat_history_str = ""
        if chat_history:
            for speaker, text in chat_history[-6:]:
                chat_history_str += f"{speaker}: {text}\n"

        system_prompt = "You are RetrievalGPT, an intelligent RAG assistant. Answer the user's question accurately using ONLY the provided Source Documents. If the documents do not contain the answer, tell the user that the information is not present in the uploaded files."

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
        
        # If we got a successful answer, return it with provider footnote
        if "Demo Fallback" in provider:
            # Custom beautiful mock answer
            mock_ans = (
                f"**[DEMO MODE: Offline & No API keys configured]**\n\n"
                f"Based on your query *'{query}'*, I parsed the matching document snippets:\n\n"
            )
            for idx, (chunk, score) in enumerate(context_chunks):
                mock_ans += f"- **From {chunk['source']}** (relevance: {score:.2f}):\n  \"{chunk['text'][:160]}...\"\n\n"
            mock_ans += "Please configure **Ollama** locally or enter a **Gemini/Groq API Key** in the sidebar to get actual generated answers."
            return mock_ans
            
        return f"{response_text}\n\n*— Generated via {provider}*"
