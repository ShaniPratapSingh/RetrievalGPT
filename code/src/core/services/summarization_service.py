import re
import os
import math
from typing import List, Dict, Any, Tuple
from src.core.splitter import RecursiveCharacterTextSplitter
from src.core.observability import Logger, telemetry

logger = Logger("summarization_service")

class SummarizationService:
    def __init__(self, call_llm_fn):
        self.call_llm = call_llm_fn
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)

    def _is_junk_chunk(self, text: str) -> bool:
        """Filter out bibliography, references, index, tables of contents, and lists of page numbers."""
        text_lower = text.lower().strip()
        
        # 1. Direct heading match check
        junk_headings = [
            r"^\s*bibliography\s*$",
            r"^\s*references\s*$",
            r"^\s*table of contents\s*$",
            r"^\s*index\s*$",
            r"^\s*page\s+\d+\s*$"
        ]
        for pattern in junk_headings:
            if re.search(pattern, text_lower, re.MULTILINE):
                logger.info("Filtered chunk due to junk heading match")
                return True
                
        # 2. Check density of references/citations
        # If the chunk has many citation brackets e.g. [1], [2], or lines matching citation styles
        citations = re.findall(r'\[\d+\]', text_lower)
        if len(citations) > 8 and len(text_lower) < 1500:
            logger.info("Filtered chunk due to high citation density")
            return True
            
        # 3. Check if chunk looks like a bibliography list:
        # e.g., lists of authors with dates like (1998) or (2020)
        dates_in_brackets = re.findall(r'\((?:19|20)\d{2}\)', text_lower)
        if len(dates_in_brackets) > 5 and len(text_lower) < 1500:
            logger.info("Filtered chunk due to bibliography/reference patterns")
            return True
            
        return False

    def summarize_chunks(self, chunks: List[str], mode: str = "short") -> List[str]:
        """Map phase: Summarize each individual chunk using target mode instructions."""
        summaries = []
        
        system_prompt = (
            "You are an expert research assistant. Generate a concise summary of the provided text chunk. "
            "Focus only on core concepts, exclude references/bibliography, and highlight key ideas."
        )
        
        for idx, chunk in enumerate(chunks):
            # Formulate mode prompt instructions
            prompt = f"""Generate a summary of this section of a document.
Mode instructions:
- Use clear, professional language.
- Exclude citation references or list formatting.
- Summarize key points.

Document Section:
{chunk}

Summary:"""
            try:
                provider, summary_text = self.call_llm(prompt, system_prompt)
                telemetry.record_provider(provider)
                if summary_text and provider not in ["Test Fallback (Mock)", "Demo Fallback (Mock)"]:
                    summaries.append(summary_text.strip())
                else:
                    # Fallback simulation summary if LLM offline/test
                    summaries.append(f"[Section {idx+1} Summary]: {chunk[:150]}...")
            except Exception as e:
                logger.error("Failed to summarize chunk", idx=idx, error=str(e))
                summaries.append(f"[Section {idx+1} Fallback]: {chunk[:150]}...")
                
        return summaries

    def generate_final_summary(self, chunk_summaries: List[str], mode: str = "short") -> str:
        """Reduce phase: Combine chunk summaries into a cohesive document summary."""
        combined_text = "\n\n".join(chunk_summaries)
        
        system_prompt = (
            "You are a Staff Research Analyst. Write a coherent, unified final summary of the entire document based "
            "strictly on the provided section summaries. Ignore bibliography and formatting lists."
        )
        
        mode_rules = {
            "short": "Write a high-level concise summary (1-2 cohesive paragraphs).",
            "detailed": "Write a comprehensive detailed summary covering all key aspects in 4-6 paragraphs.",
            "bullet": "Summarize key ideas in structured, clear bullet points.",
            "chapter-wise": "Organize the summary section-by-section or chapter-wise with headings."
        }
        
        rules = mode_rules.get(mode.lower(), mode_rules["short"])
        
        prompt = f"""Source section summaries of the document:
{combined_text}

Rules:
- {rules}
- Do not make up any facts outside the provided content.
- Ignore table of contents, bibliography, page listings.
- Maintain logical continuity.

Final Unified Summary:"""
        
        try:
            provider, final_text = self.call_llm(prompt, system_prompt)
            telemetry.record_provider(provider)
            
            if provider in ["Test Fallback (Mock)", "Demo Fallback (Mock)"]:
                # Local mock summary generation
                mock_sum = f"### [SUMMARY WORKSPACE: {mode.upper()} MODE]\n\n"
                if mode == "bullet":
                    mock_sum += "\n".join([f"- Simulated bullet summary from section: {c[:80]}..." for c in chunk_summaries[:5]])
                else:
                    mock_sum += f"Simulated cohesive summary merging {len(chunk_summaries)} sections: " + " ".join([c[:120] for c in chunk_summaries[:3]])
                return mock_sum
                
            return final_text.strip()
        except Exception as e:
            logger.error("Failed to generate final summary", error=str(e))
            return "Unable to compile final summary due to model query error."

    def summarize_document(self, doc_text: str, doc_name: str, mode: str = "short") -> Dict[str, Any]:
        """
        Bypasses standard retrieval. Split document into chunks, filter junk chunks,
        summarize hierarchically using Map-Reduce, and return structured output.
        """
        telemetry.start_span("summarize_document")
        
        # 1. Chunk document
        raw_chunks = self.splitter.split_text(doc_text)
        
        # 2. Filter bibliographies / references
        filtered_chunks = [c for c in raw_chunks if not self._is_junk_chunk(c)]
        if not filtered_chunks:
            # Fallback if everything got filtered
            filtered_chunks = raw_chunks[:5]
            
        logger.info("Filtered chunks count", original=len(raw_chunks), remaining=len(filtered_chunks))
        
        # Page processing calculation (estimate 1500 chars/page)
        pages_processed = max(1, math.ceil(len(doc_text) / 1500))
        
        # 3. Recursive Hierarchical Map-Reduce
        current_tier = filtered_chunks
        tier_level = 0
        
        # Keep reducing if intermediate text is too large
        while len(current_tier) > 1 and (tier_level == 0 or len("\n\n".join(current_tier)) > 8000):
            logger.info("Running summarization tier", level=tier_level, chunk_count=len(current_tier))
            
            # Batch chunks if there are too many (e.g. batch size of 5 chunks per LLM call)
            batch_size = 5
            next_tier = []
            
            for i in range(0, len(current_tier), batch_size):
                batch = current_tier[i:i+batch_size]
                if tier_level == 0:
                    # Map phase: Summarize raw chunks
                    summarized_batch = self.summarize_chunks(batch, mode)
                else:
                    # Intermediate reduce phase: Condense intermediate summaries
                    summarized_batch = [self.generate_final_summary(batch, mode)]
                next_tier.extend(summarized_batch)
                
            current_tier = next_tier
            tier_level += 1
            if len(current_tier) <= 1:
                break
                
        # 4. Final Reduce
        final_summary = self.generate_final_summary(current_tier, mode)
        
        # 5. Estimate confidence (base 0.95, slightly down if a lot of reference pages were skipped)
        skipped_ratio = 1.0 - (len(filtered_chunks) / max(1, len(raw_chunks)))
        confidence = round(0.95 - (skipped_ratio * 0.1), 2)
        
        telemetry.end_span("summarize_document")
        
        return {
            "summary": final_summary,
            "summary_type": mode,
            "document_name": doc_name,
            "pages_processed": pages_processed,
            "confidence": confidence
        }
