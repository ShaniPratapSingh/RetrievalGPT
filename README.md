# 🚀 RetrievalGPT - Enterprise Agentic Hybrid RAG Platform

RetrievalGPT is a production-grade, enterprise-level Retrieval-Augmented Generation (RAG) platform built with agentic query understanding, hybrid search capabilities, persistent storage, and rigorous guardrails.

---

## 🏗️ Architecture & Features

RetrievalGPT features a modular, performant, and secure pipeline:

```text
User Query ──> [Guardrails: Sanitization & Jailbreak Block]
                    │
                    ▼
          [Agentic Query Analyzer] ──> Classifies query (factual, summarization, comparative, etc.)
                    │
                    ▼
          [Hybrid Retriever] 
          ├── Dense Retrieval: Persistent ChromaDB + local embeddings (all-MiniLM-L6-v2)
          └── Sparse Retrieval: BM25 (rank-bm25)
                    │
                    ▼
          [Reciprocal Rank Fusion - RRF] ──> Merges dense & sparse results
                    │
                    ▼
          [Cross-Encoder Re-ranking] ──> MS-MARCO model (optional, lazy loaded)
                    │
                    ▼
          [Context Compression] ──> Deduplicates overlapping snippets
                    │
                    ▼
          [Groundedness Guardrails] ──> Run LLM self-consistency check for hallucinations
                    │
                    ▼
     ┌──────────────┴──────────────┐
     ▼                             ▼
[High Confidence]           [Low Confidence / Out-of-Domain]
     │                             │
[Source Citation Engine]    [Web Search Fallback]
     │                       ├── Tavily API
     │                       └── Serper API
     ▼                             ▼
[Clean Response + Cited Highlights] <─── [Merged context answer]
```

* **Advanced Hybrid Retrieval**: Combines semantic vector similarity (ChromaDB) with lexical relevance (BM25) fused via Reciprocal Rank Fusion (RRF) and re-ranked with Cross-Encoder models.
* **Agentic Query Classification**: Automatically analyzes query intent (factual, comparison, etc.) to customize retrieval depth, chunk constraints, and fallback strategies.
* **Citation Grounding Engine**: Sentence-level claims are mapped to source documents, page numbers, and chunk IDs to generate clickable, highlighted evidence.
* **Hallucination Detection Guardrails**: Evaluates answer groundedness against context before returning, blocking fabricated claims.
* **Redis Caching & Persistence**: Optional local/remote Redis integration to persist conversational history and cache embeddings and LLM completions.
* **Multimodal OCR Ingestion**: Supports PDF, DOCX, Markdown, HTML, TXT, and OCR text extraction from image formats (`.png`, `.jpg`, `.jpeg`).

---

## ⚙️ Environment Configuration

Copy the example environment file and add your credentials:
```bash
cp .env.example .env
```

The application supports the following `.env` settings:
```ini
# Core LLM Credentials (any of these free/local fallbacks can be configured)
GOOGLE_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
OLLAMA_API_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

# Web Search Fallback (Optional)
TAVILY_API_KEY=your_tavily_api_key
SERPER_API_KEY=your_serper_api_key

# Redis Configuration (Optional, for persistent memory/cache)
REDIS_HOST=localhost
REDIS_PORT=6379

# Cross-Encoder Re-ranking (Optional, set to true to enable)
RAG_RERANKING_ENABLED=false
```

---

## 🚀 Execution & Deployment

### 1. Local Run
Install dependencies and launch the platform:
```bash
pip install -r code/requirements.txt
python run.py
```
Access the premium interface in your browser at `http://localhost:8501`.

### 2. Run with Docker Compose
Spin up the RAG application and a persistent Redis cache instance automatically:
```bash
docker-compose up --build
```

### 3. Kubernetes Deployment
Apply the deployment manifest:
```yaml
kubectl apply -f k8s-deployment.yaml
```

---

## 🧪 Testing Strategy
To execute the comprehensive test suite (verifying storage sync, hybrid retrieval, coreference resolution, citations, and input sanitization):
```bash
python -m unittest discover -s code/tests
```
