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

from src.retrieval.intent_classifier import QueryIntentClassifier
from src.retrieval.chunk_filter import ChunkFilter
from src.retrieval.quote_extractor import QuoteExtractor
from src.retrieval.metadata_retriever import MetadataRetriever

class TestAgenticRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        # Mock LLM responder
        def mock_llm(prompt, system_prompt=None):
            if "Classify" in prompt or "intent classifier" in system_prompt:
                # Mock classification result JSON
                return "Mock provider", '<answer>{"intent": "important quote", "summary_mode": "short", "explanation": "Mock explanation"}</answer>'
            elif "Identify the single most important" in prompt:
                # Mock quote extraction result JSON
                return "Mock provider", '<answer>{"quote": "Deep learning is hierarchical.", "explanation": "Explaining hierarchies.", "source_document": "dl.pdf", "page": "45", "confidence": 0.98, "found": true}</answer>'
            return "Mock provider", "Standard mock response."
            
        self.mock_llm = mock_llm
        self.classifier = QueryIntentClassifier(call_llm_fn=mock_llm)
        self.chunk_filter = ChunkFilter()
        self.quote_extractor = QuoteExtractor(call_llm_fn=mock_llm)

    def test_local_intent_classifier(self):
        # Rule-based locally classified intents check
        intent1 = self.classifier.classify_locally("What is the most memorable quote in chapter 2?")
        self.assertEqual(intent1, "important quote")
        
        intent2 = self.classifier.classify_locally("Give me a brief summary of the book")
        self.assertEqual(intent2, "summarization")
        
        intent3 = self.classifier.classify_locally("Define backpropagation algorithm")
        self.assertEqual(intent3, "definition")

    def test_llm_intent_classifier(self):
        res = self.classifier.classify("Extract the best quote from chapter 4")
        self.assertEqual(res["intent"], "important quote")

    def test_chunk_filter_and_scoring(self):
        # Test bibliography filter
        text_bib = "Bibliography\n[1] Bengio et al. Representation learning, 2013."
        score_bib = self.chunk_filter.get_quality_score(text_bib)
        self.assertTrue(score_bib < 0.35)
        
        # Test math filters
        text_math = "y = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2 + \alpha_i \beta_j \gamma_k"
        score_math = self.chunk_filter.get_quality_score(text_math)
        self.assertTrue(score_math < 0.35)
        
        # Test priority definition block
        text_concept = "Representation learning is defined as a set of techniques that allows a system to automatically discover the representations needed."
        score_concept = self.chunk_filter.get_quality_score(text_concept)
        self.assertTrue(score_concept > 0.50)

    def test_quote_extractor(self):
        chunks = [
            {"text": "Deep learning is hierarchical. This allows complex abstractions to be learned.", "source": "dl.pdf", "page": 45}
        ]
        res = self.quote_extractor.extract_quote("Find quote about hierarchies", chunks)
        self.assertTrue(res["found"])
        self.assertEqual(res["quote"], "Deep learning is hierarchical.")
        self.assertEqual(res["page"], "45")

    def test_metadata_filters(self):
        # Fake storage and hybrid retrieves
        class FakeHybrid:
            def retrieve_dense(self, q, top_k): return []
            def retrieve_sparse(self, q, top_k): return []
            def reciprocal_rank_fusion(self, d, s): return []
            
        retriever = MetadataRetriever(None, None, FakeHybrid())
        
        chunks = [
            {"text": "A", "source": "book1.pdf", "page": 5, "chapter": "Intro"},
            {"text": "B", "source": "book2.pdf", "page": 12, "chapter": "Math"},
            {"text": "C", "source": "book1.pdf", "page": 5, "chapter": "Deep"}
        ]
        
        # Filter by document name
        res_doc = retriever.apply_filters(chunks, {"document": "book1.pdf"})
        self.assertEqual(len(res_doc), 2)
        self.assertEqual(res_doc[0]["source"], "book1.pdf")
        
        # Filter by page
        res_page = retriever.apply_filters(chunks, {"page": 5})
        self.assertEqual(len(res_page), 2)
        
        # Filter by chapter
        res_chapter = retriever.apply_filters(chunks, {"chapter": "Intro"})
        self.assertEqual(len(res_chapter), 1)

    def test_metadata_filters_page_range_and_dates(self):
        class FakeHybrid:
            def retrieve_dense(self, q, top_k): return []
            def retrieve_sparse(self, q, top_k): return []
            def reciprocal_rank_fusion(self, d, s): return []
            
        retriever = MetadataRetriever(None, None, FakeHybrid())
        chunks = [
            {"text": "A", "source": "book1.pdf", "page": 5, "created_at": "2026-07-04"},
            {"text": "B", "source": "book2.pdf", "page": 12, "created_at": "2026-07-04"},
            {"text": "C", "source": "book1.pdf", "page": 18, "created_at": "2026-07-05"}
        ]
        
        # Filter by page range (start, end)
        res_range = retriever.apply_filters(chunks, {"page_range": (5, 15)})
        self.assertEqual(len(res_range), 2)
        
        # Filter by date
        res_date = retriever.apply_filters(chunks, {"upload_date": "2026-07-05"})
        self.assertEqual(len(res_date), 1)
        self.assertEqual(res_date[0]["text"], "C")

    def test_rrf_weights_configuration(self):
        from src.core.retriever import HybridRetriever
        retriever = HybridRetriever(None, None)
        
        # Set weights and verify RRF scores shift
        retriever.dense_weight = 2.0
        retriever.sparse_weight = 0.5
        
        dense = [{"chunk": {"id": 1, "doc_id": 0}, "score": 0.9}]
        sparse = [{"chunk": {"id": 2, "doc_id": 0}, "score": 0.8}]
        
        results = retriever.reciprocal_rank_fusion(dense, sparse, rrf_k=60)
        # ID 1 should have score: 2.0 * (1/61) ~ 0.0327
        # ID 2 should have score: 0.5 * (1/61) ~ 0.0081
        self.assertEqual(results[0][0]["id"], 1)
        self.assertTrue(results[0][1] > results[1][1] * 2)

    def test_context_compression_low_info(self):
        from src.core.retriever import HybridRetriever
        retriever = HybridRetriever(None, None)
        
        items = [
            ({"text": "Valid long conceptual sentence explaining information retrieval.", "id": 1}, 0.9),
            ({"text": "Short", "id": 2}, 0.8) # Low info, dropped
        ]
        compressed = retriever.compress_context(items)
        self.assertEqual(len(compressed), 1)
        self.assertEqual(compressed[0][0]["id"], 1)

if __name__ == "__main__":
    unittest.main()
