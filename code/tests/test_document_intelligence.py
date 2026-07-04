import unittest
import os
import sys
import tempfile

# Add source directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.core.services.ingestion_service import DocumentIntelligencePipeline

class TestDocumentIntelligencePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = DocumentIntelligencePipeline()
        
        # Setup mock LLM for summarization metadata
        def mock_llm(prompt, system_prompt=None):
            return "Mock provider", '<answer>{"summary": "Mock summary", "topics": ["Data"], "keywords": ["system"]}</answer>'
        self.mock_llm = mock_llm

    def test_csv_parsing(self):
        # Create a mock CSV file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("Name,Age,Role\nAlice,30,Engineer\nBob,25,Analyst\n")
            temp_path = f.name
            
        try:
            pages = self.pipeline.parse_document(temp_path)
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0]["page"], 1)
            self.assertEqual(pages[0]["heading"], "Row 1")
            self.assertIn("Name: Alice", pages[0]["text"])
            self.assertIn("Age: 30", pages[0]["text"])
            self.assertIn("Role: Engineer", pages[0]["text"])
        finally:
            os.unlink(temp_path)

    def test_scanned_pdf_ocr_routing(self):
        # Create a mock scanned PDF file (short text pages)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            # Simple page simulation
            f.write(b"%PDF-1.4 mock pdf structure")
            temp_path = f.name
            
        try:
            # We mock MultiDocumentParser.parse_pdf to return very little text (simulating scanned file)
            from src.core.multimodal import MultiDocumentParser
            original_parse = MultiDocumentParser.parse_pdf
            MultiDocumentParser.parse_pdf = staticmethod(lambda path: [("OCR", 1), ("OCR", 2)])
            
            pages = self.pipeline.parse_document(temp_path)
            
            # Restore
            MultiDocumentParser.parse_pdf = original_parse
            
            self.assertEqual(len(pages), 2)
            self.assertEqual(pages[0]["chapter"], "OCR Scan Output")
            self.assertIn("simulated metadata", pages[0]["text"].lower())
        finally:
            os.unlink(temp_path)

    def test_duplicate_chunk_deduplication(self):
        chunks = [
            {"text_hash": "hash_a", "filename": "doc1.pdf", "text": "Unique text a"},
            {"text_hash": "hash_b", "filename": "doc1.pdf", "text": "Unique text b"},
            {"text_hash": "hash_a", "filename": "doc1.pdf", "text": "Unique text a"} # Duplicate
        ]
        
        existing_hashes = {"hash_existing"}
        filtered = self.pipeline.deduplicate_chunks(chunks, existing_hashes)
        
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["text_hash"], "hash_a")
        self.assertEqual(filtered[1]["text_hash"], "hash_b")
        # existing_hashes should now contain hash_a and hash_b
        self.assertIn("hash_a", existing_hashes)
        self.assertIn("hash_b", existing_hashes)

    def test_document_summary_metadata(self):
        doc_text = "This is extensive document content talking about data indexing and retrieval pipelines."
        meta = self.pipeline.generate_doc_summary_metadata(doc_text, "ref.pdf", self.mock_llm)
        
        self.assertEqual(meta["summary"], "Mock summary")
        self.assertEqual(meta["topics"], ["Data"])
        self.assertEqual(meta["keywords"], ["system"])

if __name__ == "__main__":
    unittest.main()
