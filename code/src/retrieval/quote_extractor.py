import re
import json
from typing import List, Dict, Any
from src.retrieval.chunk_filter import ChunkFilter
from src.core.observability import Logger, telemetry

logger = Logger("quote_extractor")

class QuoteExtractor:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn
        self.filter = ChunkFilter(min_quality_score=0.45) # Higher threshold for quotes

    def extract_quote(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Filters context chunks and extracts the most impactful quote and conceptual explanation."""
        telemetry.start_span("quote_extraction")
        
        # 1. Filter out noise (bibliographies, math, copyrights, etc.)
        conceptual_chunks = self.filter.filter_chunks(context_chunks)
        if not conceptual_chunks:
            # If all were filtered, fallback to the top context chunks directly
            conceptual_chunks = context_chunks[:3]
            
        if not conceptual_chunks:
            telemetry.end_span("quote_extraction")
            return {
                "quote": "No quotes found.",
                "explanation": "No documents are uploaded to extract quotes from.",
                "source_document": "N/A",
                "page": "N/A",
                "confidence": 0.0,
                "found": False
            }

        # 2. Combine the text for analysis
        combined_sections = ""
        for idx, chunk in enumerate(conceptual_chunks[:4]): # Limit to top 4 conceptual chunks
            combined_sections += f"[Section {idx+1}] Source: {chunk['source']} | Page: {chunk.get('page', 'unknown')}\nText: {chunk['text']}\n\n"

        system_prompt = (
            "You are an expert literary scholar and quote extractor. Your task is to identify the most impactful "
            "sentence or passage from the provided text segments. Respond strictly in JSON format within <answer> </answer> tags."
        )
        
        prompt = f"""Review the following source segments of a document:
{combined_sections}

User Query: "{query}"

Identify the single most important, memorable, or impactful sentence/passage matching the query.
If no exact, direct memorable quote exists, set "found" to false, and summarize the central idea.

Output format:
<answer>
{{
    "quote": "The exact quote verbatim from the text",
    "explanation": "Why this quote is key and its conceptual context",
    "source_document": "Name of the source file",
    "page": "Page number or section header",
    "confidence": 0.0 to 1.0,
    "found": true | false
}}
</answer>
"""
        try:
            provider, response_text = self.call_llm(prompt, system_prompt)
            telemetry.record_provider(provider)
            
            if "Demo Fallback" in provider:
                # Local mock fallback
                top_c = conceptual_chunks[0]
                text = top_c["text"]
                # Extract first full sentence as mock quote
                sentences = re.split(r'(?<=[.!?])\s+', text)
                quote = sentences[0] if sentences else text[:100]
                telemetry.end_span("quote_extraction")
                return {
                    "quote": quote,
                    "explanation": "Simulated quote explanation extracted locally during offline fallback.",
                    "source_document": top_c["source"],
                    "page": str(top_c.get("page", 1)),
                    "confidence": 0.85,
                    "found": True
                }
                
            ans_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
            if ans_match:
                result = json.loads(ans_match.group(1).strip())
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("No JSON block found")
                    
            telemetry.end_span("quote_extraction")
            
            # Ensure safe outputs
            return {
                "quote": result.get("quote", "No verbatim quote identified."),
                "explanation": result.get("explanation", "The document discusses matching concepts."),
                "source_document": result.get("source_document", conceptual_chunks[0]["source"]),
                "page": str(result.get("page", conceptual_chunks[0].get("page", 1))),
                "confidence": float(result.get("confidence", 0.90)),
                "found": bool(result.get("found", True))
            }
        except Exception as e:
            logger.error("Failed to extract quote from document", error=str(e))
            top_c = conceptual_chunks[0]
            telemetry.end_span("quote_extraction")
            return {
                "quote": f"This book does not contain one definitive quote matching your query.",
                "explanation": f"Failed parsing quotes due to evaluation failure. The central concept centers on: '{top_c['text'][:120]}...'",
                "source_document": top_c["source"],
                "page": str(top_c.get("page", 1)),
                "confidence": 0.50,
                "found": False
            }
