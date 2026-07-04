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

from src.core.citation import CitationEngine

class TestCitationEngine(unittest.TestCase):
    def test_single_document_citation_and_highlighting(self):
        context = [
            {"source": "paper1.pdf", "text": "Deep learning models are highly hierarchical.", "page": 4, "id": 101, "rrf_score": 0.5, "rerank_score": 0.9}
        ]
        answer = "Deep learning is known to be hierarchical [1]."
        clean_answer, citations = CitationEngine.extract_citations(answer, context)
        
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["source"], "paper1.pdf")
        self.assertEqual(citations[0]["page"], 4)
        self.assertIn("hierarchical", citations[0]["highlighted_text"])

    def test_multi_document_citations(self):
        context = [
            {"source": "book_a.pdf", "text": "Self-attention is central to Transformers.", "page": 12, "id": 1, "rrf_score": 0.8},
            {"source": "book_b.pdf", "text": "Convolutions excel at image grids.", "page": 5, "id": 2, "rrf_score": 0.7}
        ]
        # Cited both sources
        answer = "Self-attention is key [1]. Grids use convolutions [2]."
        clean_answer, citations = CitationEngine.extract_citations(answer, context)
        
        self.assertEqual(len(citations), 2)
        grouped = CitationEngine.group_citations_by_document(citations)
        self.assertIn("book_a.pdf", grouped)
        self.assertIn("book_b.pdf", grouped)
        self.assertEqual(len(grouped["book_a.pdf"]), 1)
        self.assertEqual(grouped["book_a.pdf"][0]["page"], 12)

    def test_missing_evidence_detection(self):
        context = [
            {"source": "doc1.pdf", "text": "This discusses learning representations.", "id": 1}
        ]
        # Answer claims unsupported facts about quantum biology
        answer = "Quantum biology dictates cellular cell processes."
        clean_answer, citations = CitationEngine.extract_citations(answer, context)
        
        # Should detect mismatching claims and refuse to verify
        self.assertEqual(clean_answer, "This statement could not be verified from the available documents.")
        self.assertEqual(len(citations), 0)

    def test_conflicting_evidence_resolution(self):
        context = [
            {"source": "pro.pdf", "text": "Option A is faster.", "id": 1},
            {"source": "con.pdf", "text": "Option A is actually slower in benchmarks.", "id": 2}
        ]
        answer = "Some claim Option A is faster [1], while other benchmarks show Option A is slower [2]."
        clean_answer, citations = CitationEngine.extract_citations(answer, context)
        
        self.assertEqual(len(citations), 2)
        self.assertEqual(citations[0]["source"], "pro.pdf")
        self.assertEqual(citations[1]["source"], "con.pdf")

if __name__ == "__main__":
    unittest.main()
