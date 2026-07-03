import unittest
import os
import sys

# Add source directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.core.services.summarization_service import SummarizationService
from src.core.splitter import RecursiveCharacterTextSplitter

class TestSummarizationService(unittest.TestCase):
    def setUp(self):
        # Setup a mock LLM responder that simulates summary outputs
        def mock_llm(prompt, system_prompt=None):
            if "Summarize this section" in prompt or "Generate a summary" in prompt:
                return "Mock provider", "Summarized section overview."
            elif "Source section summaries" in prompt or "Final Unified Summary" in prompt:
                return "Mock provider", "Cohesive final unified document summary."
            return "Mock provider", "Standard mock response."
            
        self.service = SummarizationService(call_llm_fn=mock_llm)

    def test_recursive_character_splitter(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
        text = (
            "Paragraph one is long and descriptive. It explains the core concepts of RAG pipelines.\n\n"
            "Paragraph two describes metadata highlights and persistent vector storage configurations."
        )
        chunks = splitter.split_text(text)
        self.assertTrue(len(chunks) >= 2)
        # Ensure no chunk exceeds chunk_size
        for c in chunks:
            self.assertTrue(len(c) <= 100)

    def test_small_pdf_summarization(self):
        doc_text = "This is a brief 2-page document describing neural networks and convolution layers."
        res = self.service.summarize_document(doc_text, "neural_net.pdf", mode="short")
        
        self.assertEqual(res["document_name"], "neural_net.pdf")
        self.assertEqual(res["summary_type"], "short")
        self.assertEqual(res["pages_processed"], 1)
        self.assertTrue(res["confidence"] > 0.90)
        self.assertIn("Cohesive final unified document summary", res["summary"])

    def test_large_book_summarization(self):
        # Simulate a 700+ page book by repeating paragraphs (720 pages, ~1,080,000 characters)
        single_page_text = "This is a section of an extensive manual covering advanced distributed networks. " * 20 # ~ 1500 chars
        large_text = single_page_text * 720
        
        res = self.service.summarize_document(large_text, "distributed_systems_bible.pdf", mode="detailed")
        self.assertEqual(res["document_name"], "distributed_systems_bible.pdf")
        self.assertTrue(res["pages_processed"] >= 720)
        self.assertIn("Cohesive", res["summary"])

    def test_research_paper_with_bibliography_filtering(self):
        doc_text = (
            "Introduction: Section about AI advancements.\n\n"
            "Bibliography\n"
            "[1] Harrison Chase. LangChain overview, 2023.\n"
            "[2] Andrew Ng. Machine Learning lessons, 2021.\n"
            "[3] Vaswani et al. Attention is all you need, 2017."
        )
        # Verify the bibliography chunk is recognized as junk
        is_junk = self.service._is_junk_chunk("Bibliography\n[1] Harrison Chase. LangChain overview, 2023.\n[2] Andrew Ng.")
        self.assertTrue(is_junk)
        
        res = self.service.summarize_document(doc_text, "paper.pdf")
        # Ensure it maps/reduces successfully after filtering out junk
        self.assertEqual(res["document_name"], "paper.pdf")

    def test_scanned_pdf(self):
        # Simulated scanned OCR text with bounding indicators
        ocr_text = (
            "[OCR SCAN: page 1]\n"
            "The image shows a diagram representing model weights and loss curves.\n"
            "[OCR SCAN: page 2]\n"
            "Performance metrics: Accuracy reaches 98.4% after 15 epochs."
        )
        res = self.service.summarize_document(ocr_text, "scanned_invoice.png", mode="bullet")
        self.assertEqual(res["document_name"], "scanned_invoice.png")
        self.assertEqual(res["summary_type"], "bullet")

if __name__ == "__main__":
    unittest.main()
