import os
import re
import csv
import json
import hashlib
import math
from datetime import datetime
from typing import List, Dict, Any, Tuple, Callable
from src.core.multimodal import MultiDocumentParser
from src.core.splitter import SemanticChunker
from src.retrieval.chunk_filter import ChunkFilter
from src.core.observability import Logger, telemetry

logger = Logger("ingestion_service")

class DocumentIntelligencePipeline:
    def __init__(self):
        self.filter = ChunkFilter()

    def get_chunk_hash(self, text: str) -> str:
        """Returns the MD5 hash of the normalized chunk text."""
        normalized = " ".join(text.lower().split())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def parse_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Parses document layout and metadata across PDF, DOCX, TXT, MD, HTML, CSV."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        
        pages = [] # List of dict: {text, page, chapter, heading, section}
        
        if ext == ".csv":
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    headers = next(reader, [])
                    row_idx = 1
                    for row in reader:
                        # Format row as descriptive key-value text block
                        items = []
                        for h, val in zip(headers, row):
                            items.append(f"{h}: {val}")
                        row_text = f"Row {row_idx}: " + ", ".join(items)
                        pages.append({
                            "text": row_text,
                            "page": row_idx,
                            "chapter": "Data Rows",
                            "heading": f"Row {row_idx}",
                            "section": "Main Table"
                        })
                        row_idx += 1
            except Exception as e:
                logger.error("Failed to parse CSV document", path=file_path, error=str(e))
                
        elif ext == ".pdf":
            raw_pages = MultiDocumentParser.parse_pdf(file_path)
            
            # Scanned PDF detection: if average page length is under 50 characters, route to OCR
            total_chars = sum(len(p[0].strip()) for p in raw_pages)
            avg_chars = total_chars / max(1, len(raw_pages))
            
            if avg_chars < 50:
                logger.info("Scanned PDF detected by character density metrics. Routing pages to OCR loop.", path=file_path)
                # In a real environment, pdf2image would render images.
                # Here we fallback/simulate OCR pages matching the page counts.
                for idx in range(len(raw_pages)):
                    page_num = idx + 1
                    simulated_ocr_text = MultiDocumentParser.parse_image_ocr(f"{file_name}_page_{page_num}.png")
                    pages.append({
                        "text": simulated_ocr_text,
                        "page": page_num,
                        "chapter": "OCR Scan Output",
                        "heading": f"Page {page_num} Scan",
                        "section": "Scanned Content"
                    })
            else:
                for text, page_num in raw_pages:
                    # Detect structural indicators
                    chapter = self._detect_structure(text, r"(?i)chapter\s+\d+") or "Introduction"
                    heading = self._detect_structure(text, r"^[A-Z\s]{4,}\n") or f"Heading Page {page_num}"
                    section = self._detect_structure(text, r"^#+\s+(.*?)$") or "General Context"
                    
                    pages.append({
                        "text": text,
                        "page": page_num,
                        "chapter": chapter,
                        "heading": heading,
                        "section": section
                    })
                    
        elif ext == ".docx":
            raw_pages = MultiDocumentParser.parse_docx(file_path)
            for text, page_num in raw_pages:
                pages.append({
                    "text": text,
                    "page": page_num,
                    "chapter": "Document Context",
                    "heading": f"Section {page_num}",
                    "section": "Main Content"
                })
                
        elif ext in [".txt", ".md", ".json", ".html"]:
            raw_pages = MultiDocumentParser.parse_text_or_markdown(file_path)
            for text, page_num in raw_pages:
                pages.append({
                    "text": text,
                    "page": page_num,
                    "chapter": "Overview",
                    "heading": f"Part {page_num}",
                    "section": "Content Flow"
                })
        else:
            raise ValueError(f"Unsupported document format: {ext}")
            
        return pages

    def _detect_structure(self, text: str, regex_pattern: str) -> str:
        """Extract matching headers or structural keywords from text."""
        match = re.search(regex_pattern, text, re.MULTILINE)
        if match:
            return match.group(0).strip()
        return ""

    def chunk_document_semantically(self, parsed_pages: List[Dict[str, Any]], doc_id: int, filename: str, embed_fn: Callable) -> List[Dict[str, Any]]:
        """Groups sentences semantically and attaches structural and quality score metadata."""
        chunker = SemanticChunker(embed_fn=embed_fn)
        chunks = []
        
        chunk_idx = 0
        for page_data in parsed_pages:
            text = page_data["text"]
            page_num = page_data["page"]
            chapter = page_data["chapter"]
            heading = page_data["heading"]
            section = page_data["section"]
            
            # Split page text semantically
            page_chunks = chunker.split_text(text)
            
            for item_text in page_chunks:
                clean_text = item_text.strip()
                if not clean_text:
                    continue
                    
                quality_score = self.filter.get_quality_score(clean_text)
                text_hash = self.get_chunk_hash(clean_text)
                
                chunks.append({
                    "id": chunk_idx,
                    "document_id": doc_id,
                    "filename": filename,
                    "source": filename,
                    "text": clean_text,
                    "page": page_num,
                    "chapter": chapter,
                    "heading": heading,
                    "section": section,
                    "chunk_id": f"doc_{doc_id}_chunk_{chunk_idx}",
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "source_type": os.path.splitext(filename)[1].replace(".", "").upper(),
                    "quality_score": quality_score,
                    "text_hash": text_hash
                })
                chunk_idx += 1
                
        return chunks

    def generate_doc_summary_metadata(self, doc_text: str, filename: str, call_llm_fn: Callable) -> Dict[str, Any]:
        """Generate high-level metadata (summary, keywords, topics) using the LLM."""
        system_prompt = (
            "You are an expert document indexer. Analyze the document text and extract the key summary, "
            "topics, and keywords. Respond strictly in JSON format within <answer> </answer> tags."
        )
        
        prompt = f"""Analyze this document:
Filename: {filename}
Sample Text: {doc_text[:4000]}

Generate:
- A concise summary (2-3 sentences)
- Key topics discussed (List of 3-5 strings)
- Index keywords (List of 5-8 words)

Output JSON format:
<answer>
{{
    "summary": "Document overview text here",
    "topics": ["topic1", "topic2"],
    "keywords": ["keyword1", "keyword2"]
}}
</answer>
"""
        try:
            provider, response_text = call_llm_fn(prompt, system_prompt)
            
            if provider in ["Test Fallback (Mock)", "Demo Fallback (Mock)"]:
                return {
                    "summary": f"A technical document describing details of {filename}.",
                    "topics": ["Data Processing", "Technical Reference"],
                    "keywords": ["system", "data", "metadata"]
                }
                
            ans_match = re.search(r'<answer>(.*?)(</answer>|$)', response_text, re.DOTALL)
            if ans_match:
                result = json.loads(ans_match.group(1).strip())
            else:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(0))
                else:
                    raise ValueError("JSON block not found")
                    
            return {
                "summary": result.get("summary", "Document content details."),
                "topics": result.get("topics", ["Knowledge Platform"]),
                "keywords": result.get("keywords", ["rag"])
            }
        except Exception as e:
            logger.warn("Document summary metadata extraction failed. Using fallback defaults.", error=str(e))
            return {
                "summary": f"Knowledge base document file {filename}.",
                "topics": ["Information Ingestion"],
                "keywords": ["document"]
            }

    def deduplicate_chunks(self, chunks: List[Dict[str, Any]], existing_hashes: set) -> List[Dict[str, Any]]:
        """Filters out chunks that already exist in the database (avoid duplicates)."""
        filtered = []
        for c in chunks:
            h = c["text_hash"]
            if h not in existing_hashes:
                filtered.append(c)
                existing_hashes.add(h)
            else:
                logger.info("Index duplicate chunk prevented", hash=h, doc=c["filename"])
        return filtered
