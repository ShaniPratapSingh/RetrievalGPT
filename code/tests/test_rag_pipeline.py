import unittest
import os
import tempfile
import sys

# Add source directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
src_dir = os.path.join(parent_dir, "src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.rag_engine import RAGEngine

class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        # Initialize RAG Engine (defaults to local embeddings)
        self.engine = RAGEngine()
        self.engine.storage.clear_database()
        self.engine.sync_local_lists()
        self.engine.retriever.rebuild_sparse_index()
        
        # Create a temporary file with mock document contents
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
        self.test_content = (
            "RetrievalGPT is an advanced RAG application that transforms information retrieval. "
            "It integrates query rewriting reasoning processes into the retrieval workflow. "
            "This enables it to convert search requests into structured Boolean query terms, "
            "greatly enhancing matching performance on local indices."
        )
        self.temp_file.write(self.test_content.encode("utf-8"))
        self.temp_file.close()

    def tearDown(self):
        # Clean up the temporary file
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

    def test_load_and_chunk_document(self):
        # Test document loading
        doc_id = self.engine.load_document(self.temp_file.name)
        self.assertEqual(len(self.engine.documents), 1)
        self.assertEqual(self.engine.documents[doc_id]["source"], os.path.basename(self.temp_file.name))
        
        # Test chunking
        chunks = self.engine.chunk_text(self.test_content, chunk_size=20, chunk_overlap=5)
        self.assertTrue(len(chunks) > 0)
        self.assertIn("RetrievalGPT", chunks[0])

    def test_indexing_and_retrieval(self):
        # Index document
        doc_id = self.engine.load_document(self.temp_file.name)
        num_chunks = self.engine.index_document(doc_id, chunk_size=15, chunk_overlap=5)
        self.assertTrue(num_chunks > 0)
        self.assertEqual(len(self.engine.chunks), num_chunks)
        
        # Test retrieval
        results = self.engine.retrieve("Boolean query terms", top_k=2)
        self.assertTrue(len(results) > 0)
        
        # Verify document score and retrieval relevance
        chunk, score = results[0]
        self.assertTrue(score > 0.0)
        self.assertIn("source", chunk)
        self.assertEqual(chunk["source"], os.path.basename(self.temp_file.name))

    def test_query_rewriting_extraction(self):
        # Test tag extraction helper
        raw_response = (
            "<think>\nNeed to find info about boolean queries.\n</think>\n"
            "<answer>\n{\n  \"query\": \"boolean AND queries\"\n}\n</answer>"
        )
        thought, query = self.engine._extract_thought_and_query(raw_response)
        self.assertEqual(thought, "Need to find info about boolean queries.")
        self.assertEqual(query, "boolean AND queries")

        # Test partial/missing tags
        broken_response = "Here is my reasoning: find search terms.\n{\n  \"query\": \"search\"\n}"
        thought, query = self.engine._extract_thought_and_query(broken_response)
        self.assertEqual(query, "search")

    def test_generate_answer_fallback(self):
        # Retrieve chunks first
        doc_id = self.engine.load_document(self.temp_file.name)
        self.engine.index_document(doc_id)
        retrieved = self.engine.retrieve("RetrievalGPT", top_k=1)
        
        # Call answer generation in fallback mode (since we have mock/unauthorized API keys)
        answer = self.engine.generate_answer("What is RetrievalGPT?", retrieved)
        self.assertTrue("Demo Mode" in answer or len(answer) > 0)

if __name__ == "__main__":
    unittest.main()
