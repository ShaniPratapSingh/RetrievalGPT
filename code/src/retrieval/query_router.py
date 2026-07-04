from typing import Dict, Any, List
from src.retrieval.intent_classifier import QueryIntentClassifier
from src.retrieval.quote_extractor import QuoteExtractor
from src.retrieval.summarizer import Summarizer
from src.retrieval.metadata_retriever import MetadataRetriever
from src.retrieval.chunk_filter import ChunkFilter
from src.retrieval.reranker import Reranker
from src.core.observability import Logger

logger = Logger("query_router")

class QueryRouter:
    def __init__(self, call_llm_fn, storage_manager, get_embedding_fn, hybrid_retriever):
        self.classifier = QueryIntentClassifier(call_llm_fn)
        self.filter = ChunkFilter()
        self.reranker = Reranker()
        self.summarizer = Summarizer(call_llm_fn)
        self.quote_extractor = QuoteExtractor(call_llm_fn)
        self.metadata_retriever = MetadataRetriever(storage_manager, get_embedding_fn, hybrid_retriever)

    def route_query(self, query: str, history_str: str = "", filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Classify intent, dynamically filter noise, and route to specialized generation systems."""
        # 1. Classify Intent
        classification = self.classifier.classify(query, history_str)
        intent = classification["intent"]
        mode = classification["summary_mode"]
        
        logger.info("Routing query based on intent", query=query, intent=intent)
        
        # 2. Route Intent
        if intent in ["summarization", "document overview", "chapter summary"]:
            return {
                "route": "summarization",
                "intent": intent,
                "summary_mode": mode,
                "data": None
            }
            
        elif intent in ["important quote", "important line"]:
            # Retrieve segments to feed to quote extractor
            candidate_chunks = self.metadata_retriever.retrieve(query, intent, filters)
            # Extracted clean quote card details
            quote_card = self.quote_extractor.extract_quote(query, candidate_chunks)
            return {
                "route": "quote_extraction",
                "intent": intent,
                "data": quote_card
            }
            
        else:
            # Standard QA routing
            raw_chunks = self.metadata_retriever.retrieve(query, intent, filters)
            
            # Apply quality filtering
            clean_chunks = self.filter.filter_chunks(raw_chunks)
            if not clean_chunks:
                clean_chunks = raw_chunks[:4] # Fallback to top-k if all filtered
                
            # Run Rerank
            final_chunks = self.reranker.rerank(query, clean_chunks)
            
            return {
                "route": "standard_qa",
                "intent": intent,
                "data": final_chunks
            }
