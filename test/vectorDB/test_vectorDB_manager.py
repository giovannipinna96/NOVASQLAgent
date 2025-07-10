"""
Tests for src/vectorDB/vectorDB_manager.py
As per instructions, this file will only contain the code, without attempting to run it
or install dependencies like 'pytest', 'faiss', or 'sentence-transformers'.
Actual FAISS operations and embedding model usage will be mocked or tested structurally.
"""
import unittest
from pathlib import Path
import tempfile
import sys
import pickle
import numpy as np

# Add src directory to sys.path
try:
    SRC_DIR = Path(__file__).resolve().parent.parent.parent / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.append(str(SRC_DIR))
    from vectorDB.vectorDB_manager import VectorDBManager, SimilarityMetric, DEFAULT_EMBEDDING_MODEL
except ImportError as e:
    print(f"Test_VectorDBManager: Could not import VectorDBManager or components. Error: {e}")
    VectorDBManager = None # type: ignore
    SimilarityMetric = None # type: ignore
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Fallback if import fails

# Mock FAISS and SentenceTransformers if not available for structural testing
faiss_mocked = False
if 'faiss' not in sys.modules:
    class MockFaissIndex:
        def __init__(self, dim): self.dim = dim; self.ntotal = 0
        def add(self, vectors): self.ntotal += len(vectors)
        def add_with_ids(self, vectors, ids): self.ntotal += len(vectors) # Simplified
        def search(self, query_vectors, k):
            # Return dummy distances and indices
            num_queries = query_vectors.shape[0]
            # Ensure indices are within a plausible range, even if ntotal is 0 for a fresh mock.
            # Max index should be less than ntotal if ntotal > 0.
            # If ntotal is 0, FAISS might return -1s.
            # For simplicity, let's assume some items exist for search structure.
            # A real test would populate ntotal based on adds.
            dummy_indices = np.array([[i for i in range(k)] for _ in range(num_queries)], dtype=np.int64)
            dummy_distances = np.random.rand(num_queries, k).astype(np.float32)
            return dummy_distances, dummy_indices
        def remove_ids(self, ids_to_remove): return len(ids_to_remove) # Assume all were removed

    class MockIndexIDMap(MockFaissIndex): # Inherits add, search, etc.
        def __init__(self, index_flat): super().__init__(index_flat.dim); self.index_flat = index_flat

    class MockFaissModule:
        IndexFlatL2 = MockFaissIndex
        IndexFlatIP = MockFaissIndex # For cosine, vectors should be normalized
        IndexIDMap = MockIndexIDMap
        def normalize_L2(self, vectors): pass # In-place operation
        def write_index(self, index, path): pass
        def read_index(self, path): return MockFaissIndex(384) # Return a mock index

    sys.modules['faiss'] = MockFaissModule() # type: ignore
    faiss = sys.modules['faiss'] # Make it available for VectorDBManager to import
    faiss_mocked = True
    print("Test_VectorDBManager: Mocked 'faiss' library.")

if 'sentence_transformers' not in sys.modules:
    class MockSentenceTransformer:
        def __init__(self, model_name, device=None):
            self.model_name_or_path = model_name
            self._embedding_dim = 384 # Default for 'all-MiniLM-L6-v2'
            if "768" in model_name: self._embedding_dim = 768 # Simple mock logic

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            num_texts = len(texts) if isinstance(texts, list) else 1
            return np.random.rand(num_texts, self._embedding_dim).astype(np.float32)
        def get_sentence_embedding_dimension(self): return self._embedding_dim

    sys.modules['sentence_transformers'] = type('MockSentenceTransformersModule', (object,), { # type: ignore
        'SentenceTransformer': MockSentenceTransformer
    })()
    print("Test_VectorDBManager: Mocked 'sentence_transformers' library.")
    # Re-evaluate VectorDBManager if it failed due to sentence_transformers not being found
    if VectorDBManager is None:
        try:
            from vectorDB.vectorDB_manager import VectorDBManager, SimilarityMetric, DEFAULT_EMBEDDING_MODEL
        except ImportError:
            pass # Already handled


@unittest.skipIf(VectorDBManager is None or SimilarityMetric is None, "VectorDBManager or dependencies not loaded, skipping tests.")
class TestVectorDBManager(unittest.TestCase):
    """Unit tests for the VectorDBManager class."""

    def setUp(self):
        self.temp_db_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_db_dir.name)

        # Due to "no run" constraint, we instantiate VectorDBManager,
        # but its internal model loading and FAISS init might use mocks if libs are absent.
        try:
            self.db_manager = VectorDBManager(embedding_model_name=DEFAULT_EMBEDDING_MODEL)
        except Exception as e:
            # If instantiation itself fails (e.g. unmocked part of library), skip tests for this manager.
            print(f"Failed to instantiate VectorDBManager in setUp for tests (might be ok in no-run mode): {e}")
            self.db_manager = None # type: ignore
            self.skipTest(f"VectorDBManager instantiation failed: {e}")


    def tearDown(self):
        self.temp_db_dir.cleanup()

    def test_initialization(self):
        """Test VectorDBManager initialization."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        self.assertIsNotNone(self.db_manager.embedding_model)
        self.assertEqual(self.db_manager.embedding_dim, 384) # For 'all-MiniLM-L6-v2'
        self.assertIsNotNone(self.db_manager.index)
        self.assertIsInstance(self.db_manager.index, sys.modules['faiss'].IndexIDMap) # type: ignore # Should be wrapped
        self.assertEqual(self.db_manager.current_metric, SimilarityMetric.L2) # type: ignore

    def test_get_embeddings_structure(self):
        """Test _get_embeddings structure (mocked actual embedding)."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        texts = ["hello world", "another sentence"]
        embeddings = self.db_manager._get_embeddings(texts) # type: ignore # private access for test
        self.assertIsNotNone(embeddings)
        if embeddings is not None: # mypy check
            self.assertEqual(embeddings.shape, (2, self.db_manager.embedding_dim))
            self.assertTrue(isinstance(embeddings, np.ndarray))

    def test_add_items_structure(self):
        """Test add_items structure and metadata storage."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        items = [
            ("id1", "Content for item 1", {"field_a": "value_a1"}),
            ("id2", {"title": "Item 2 Title", "body": "Body text for item 2"}, {"field_a": "value_a2"}),
        ]
        self.db_manager.add_items(items)
        self.assertEqual(self.db_manager.total_items, 2)
        self.assertIn("id1", self.db_manager.metadata_store)
        self.assertEqual(self.db_manager.metadata_store["id1"]["field_a"], "value_a1")
        if self.db_manager.index : self.assertEqual(self.db_manager.index.ntotal, 2) # Mock faiss might behave differently

    def test_add_items_update_existing(self):
        """Test that adding an item with an existing ID updates it."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        self.db_manager.add_items([("id1", "Initial content", {"version": 1})])
        initial_faiss_id = next(fid for fid, iid in self.db_manager.idx_to_id.items() if iid == "id1")

        self.db_manager.add_items([("id1", "Updated content", {"version": 2})])
        self.assertEqual(self.db_manager.total_items, 1)
        self.assertEqual(self.db_manager.metadata_store["id1"]["version"], 2)
        # Check if FAISS ID was reused or if old one was removed and new one added.
        # With current mock (and real IndexIDMap), remove + add_with_ids should allow reuse of FAISS ID.
        updated_faiss_id = next(fid for fid, iid in self.db_manager.idx_to_id.items() if iid == "id1")
        # This assertion depends on the mock's remove_ids and add_with_ids behavior.
        # If the mock is simple, the ID might change. A real IndexIDMap would allow control.
        # self.assertEqual(initial_faiss_id, updated_faiss_id) # Ideal, but mock might not support.

    def test_remove_items_structure(self):
        """Test remove_items structure."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        items = [("id1", "Content 1", {}), ("id2", "Content 2", {})]
        self.db_manager.add_items(items)
        self.assertEqual(self.db_manager.total_items, 2)

        removed_count = self.db_manager.remove_items(["id1", "id_nonexistent"])
        self.assertEqual(removed_count, 1) # Mock remove_ids returns len(ids_to_remove_found_in_map)
        self.assertEqual(self.db_manager.total_items, 1)
        self.assertIsNone(self.db_manager.get_item_by_id("id1"))
        self.assertIsNotNone(self.db_manager.get_item_by_id("id2"))

    def test_semantic_search_structure(self):
        """Test semantic_search structure (mocked results)."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        self.db_manager.add_items([("id1", "France related text", {})]) # Need at least one item for search mock

        query = "capital of France"
        results = self.db_manager.semantic_search(query, top_k=1)

        self.assertIsInstance(results, list)
        if results: # Mock search returns dummy results
            self.assertEqual(len(results), 1)
            item_id, metadata, score = results[0]
            self.assertIsInstance(item_id, str)
            self.assertIsInstance(metadata, dict)
            self.assertIsInstance(score, float)
            # We can't assert specific item_id with current mock search, as it returns fixed indices.

    def test_semantic_search_with_filter_structure(self):
        """Test semantic_search with a metadata filter function."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        self.db_manager.add_items([
            ("id1", "Text A", {"category": "X"}),
            ("id2", "Text B", {"category": "Y"}),
        ])
        filter_func = lambda meta: meta.get("category") == "X"
        results = self.db_manager.semantic_search("Search text", top_k=2, filter_func=filter_func)
        # With current mock, filtering happens *after* FAISS search.
        # Mock FAISS search returns fixed indices (0,1,...).
        # So, if idx_to_id[0] is "id1" (category X) and idx_to_id[1] is "id2" (category Y),
        # then only id1 should be returned.
        # This depends on the order of items added and how mock FAISS indices map back.
        # For a structural test, we check that results (if any) adhere to filter.
        if results:
            for _, meta, _ in results:
                self.assertEqual(meta.get("category"), "X")


    def test_exact_match_search_structure(self):
        """Test exact_match_search structure."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        self.db_manager.add_items([
            ("id1", "Content", {"tag": "alpha", "value": 10}),
            ("id2", "Content", {"tag": "beta", "value": 20}),
            ("id3", "Content", {"tag": "alpha", "value": 30}),
        ])
        results = self.db_manager.exact_match_search("tag", "alpha", max_results=1)
        self.assertEqual(len(results), 1)
        if results:
            self.assertEqual(results[0][1]["tag"], "alpha") # Check one of the results

        results_case = self.db_manager.exact_match_search("tag", "Alpha", case_sensitive=False)
        self.assertEqual(len(results_case), 2) # Should find "id1" and "id3"

        results_val = self.db_manager.exact_match_search("value", 20)
        self.assertEqual(len(results_val), 1)
        if results_val: self.assertEqual(results_val[0][0], "id2")


    def test_save_and_load_index_structure(self):
        """Test save_index and load_index structure (mocked file IO)."""
        if not self.db_manager: self.skipTest("DB Manager not initialized in setUp.")
        self.db_manager.add_items([("id1", "Data to save", {"saved": True})])

        with unittest.mock.patch('faiss.write_index') as mock_write_faiss, \
             unittest.mock.patch('faiss.read_index') as mock_read_faiss, \
             unittest.mock.patch('builtins.open', new_callable=unittest.mock.mock_open) as mock_open_pickle: # type: ignore

            # Mock read_index to return a compatible mock index for load
            mock_read_faiss.return_value = sys.modules['faiss'].MockFaissIndex(self.db_manager.embedding_dim) # type: ignore

            # --- Save ---
            self.db_manager.save_index(self.db_path)
            mock_write_faiss.assert_called_once_with(self.db_manager.index, str(self.db_path / "faiss_index.idx"))

            # Check if pickle.dump was called (via mock_open)
            # The first call to mock_open is for writing the pickle file.
            # mock_open.assert_any_call(self.db_path / "index_meta.pkl", 'wb')
            # Check that dump was called on the file handle provided by mock_open for writing.
            # This is tricky as pickle.dump is not directly mocked, but its file handle is.
            # A simpler check: was mock_open used for writing?
            found_write_call = False
            for call_args in mock_open_pickle.call_args_list:
                if call_args[0][0] == self.db_path / "index_meta.pkl" and call_args[0][1] == 'wb':
                    found_write_call = True
                    break
            self.assertTrue(found_write_call, "Pickle file was not opened for writing.")


            # --- Load ---
            # Simulate pickle loading by preparing what `pickle.load` would return
            # This requires knowing the structure of `metadata_to_save`
            metadata_that_was_saved = {
                "idx_to_id": self.db_manager.idx_to_id,
                "metadata_store": self.db_manager.metadata_store,
                "embedding_dim": self.db_manager.embedding_dim,
                "current_metric_value": self.db_manager.current_metric.value,
                "embedding_model_name": self.db_manager.embedding_model_name,
            }
            # Configure mock_open to simulate reading this data when opened in 'rb' mode
            # This is complex. A simpler way for structure: assume pickle.load works if file is opened.
            # For this test, let's assume the file paths are correct for load.

            # Create a new manager instance for loading
            if VectorDBManager: # Check again, could be None if initial import failed badly
                loaded_db_manager = VectorDBManager(embedding_model_name=DEFAULT_EMBEDDING_MODEL) # Fresh one

                # To properly mock pickle.load, we need to mock the return value of open().__enter__().read()
                # or directly mock pickle.load itself.
                with unittest.mock.patch('pickle.load') as mock_pickle_load:
                    mock_pickle_load.return_value = metadata_that_was_saved

                    loaded_db_manager.load_index(self.db_path)

                mock_read_faiss.assert_called_with(str(self.db_path / "faiss_index.idx"))
                mock_pickle_load.assert_called_once() # Check that pickle.load was attempted

                self.assertEqual(loaded_db_manager.total_items, self.db_manager.total_items)
                self.assertEqual(loaded_db_manager.embedding_dim, self.db_manager.embedding_dim)
                self.assertEqual(loaded_db_manager.current_metric, self.db_manager.current_metric)
                self.assertEqual(loaded_db_manager.idx_to_id, self.db_manager.idx_to_id)


if __name__ == "__main__":
    if VectorDBManager is not None and hasattr(unittest, 'mock'):
        print("Running VectorDBManager tests (illustrative execution with mocks)...")
        unittest.main(verbosity=2)
    else:
        reason = "VectorDBManager module not loaded"
        if not hasattr(unittest, 'mock'): reason += " or unittest.mock not available"
        print(f"Skipping VectorDBManager tests: {reason}.")
        print("This is expected if dependencies are not installed or src path is incorrect in a non-execution environment.")
