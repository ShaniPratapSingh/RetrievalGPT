import os

class Settings:
    def __init__(self):
        # General Application Config
        self.APP_NAME = os.getenv("APP_NAME", "RetrievalGPT")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Model Providers API Keys
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
        self.HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        
        # Local LLM config
        self.OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
        
        # Web Search Providers
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
        
        # Memory & Redis cache
        self.REDIS_HOST = os.getenv("REDIS_HOST", "")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        
        # Hybrid Search Parameters
        self.RAG_RRF_K = int(os.getenv("RAG_RRF_K", "60"))
        self.RAG_DENSE_WEIGHT = float(os.getenv("RAG_DENSE_WEIGHT", "1.0"))
        self.RAG_SPARSE_WEIGHT = float(os.getenv("RAG_SPARSE_WEIGHT", "1.0"))
        self.RAG_RERANKING_ENABLED = os.getenv("RAG_RERANKING_ENABLED", "false").lower() == "true"
        
        # AWS Keys
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-2")

settings = Settings()
