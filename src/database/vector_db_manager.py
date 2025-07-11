"""
This module is designed to create and manage an open-source vector database using a dedicated Python class
that encapsulates all required functionality for storage, retrieval, and similarity-based search.

OBJECTIVES AND FUNCTIONALITY:

1. VECTOR DATABASE BACKEND:
    - This file is responsible for initializing and interacting with an open-source vector database.
    - The specific vector database implementation (e.g., FAISS, Chroma, Qdrant, Weaviate, etc.) can be selected freely,
      as long as it is efficient, well-maintained, and open-source.
    - The implementation must be wrapped inside a dedicated management class for full modularity and abstraction.

2. CORE CLASS REQUIREMENTS:
    - A single class must handle all interactions with the vector database, including:
        • Adding new data points (vectors + metadata)
        • Removing existing data points based on unique identifiers or filters
        • Saving and loading the index/database to/from disk

3. SEMANTIC SIMILARITY SEARCH:
    - The class must include functionality to perform semantic search based on vector similarity.
    - A default embedding model can be used to convert textual data into vector representations.
    - The system must support **two similarity metrics**:
        • Cosine Similarity
        • L2 (Euclidean Distance)
    - A configurable switch must allow the developer to select which metric to use during a query.

4. COLUMN-WISE SEARCH LOGIC:
    - The database must support semantic similarity search across **multiple specific fields (columns)**.
    - For a given query, the system should compute similarity independently for each field and return the top-matching
      value from each field.
    - This enables targeted semantic extraction from structured documents or datasets.
    (Note: This is an advanced feature. Initial implementation might focus on single-vector search per document,
     with metadata allowing field identification. True multi-field vector search often requires specific index design.)

5. NON-VECTOR SEARCH CAPABILITIES:
    - Include simple keyword-based or exact-match search capabilities for querying specific fields in the database
      without relying on embeddings or similarity measures.
    - These methods must support filters, field-based lookups, and logical query composition (e.g., AND/OR conditions).

6. PERSISTENCE AND FILESYSTEM INTEGRATION:
    - The system must include methods to save the current state of the vector index/database to disk.
    - It must also support loading an existing index on startup, including appropriate file/folder structure handling.

7. ADDITIONAL RECOMMENDATIONS:
    - All methods should include robust error handling, type annotations, and logging support.
    - Where possible, use modern Python features (e.g., dataclasses, context managers, pathlib).
    - Provide informative docstrings for all methods and maintain a clean object-oriented architecture.
    - Consider exposing an interface or configuration object to allow users to easily plug in different embedding models
      or switch between vector backends.
    - Ensure the class supports batch insertion and batch querying for performance scalability.

8. DEVELOPMENT STANDARDS:
    - Follow Python best practices in every part of this implementation:
        • PEP8-compliant code formatting
        • Type hints and input validation
        • Unit-testable modular methods
        • Well-documented and maintainable codebase

This module is a core component of a semantic intelligence system and must serve as a high-performance layer
for managing and querying contextual embeddings within a flexible, customizable, and production-grade architecture.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterator
from enum import Enum
import numpy as np

# FAISS and SentenceTransformers will be imported conditionally for type hinting and structure,
# but actual execution will not happen in this environment.
faiss = None
SentenceTransformer = None

try:
    import faiss
except ImportError:
    logging.warning("FAISS library not found. VectorDBManager will not be fully functional.")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers library not found. VectorDBManager will not be fully functional.")


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

class SimilarityMetric(Enum):
    """Enum for supported similarity metrics."""
    COSINE = "cosine"
    L2 = "l2"

class VectorDBManager:
    """
    Manages a vector database using FAISS for storage, retrieval, and similarity search.
    Focus is on code structure; execution is not guaranteed due to potential missing dependencies.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        embedding_device: Optional[str] = None,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model: Optional[SentenceTransformer] = None
        self.embedding_dim: int = 384 # Default for all-MiniLM-L6-v2, will be updated if model loads

        if SentenceTransformer is not None:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name, device=embedding_device)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension() # type: ignore
                logger.info(f"SentenceTransformer model '{embedding_model_name}' loaded. Dim: {self.embedding_dim}.")
            except Exception as e:
                logger.error(f"Could not load SentenceTransformer model '{embedding_model_name}': {e}. Using default embedding_dim={self.embedding_dim}.")
        else:
            logger.warning(f"sentence-transformers library not available. Using default embedding_dim={self.embedding_dim}.")


        self.index: Optional[faiss.Index] = None if faiss else None
        self.idx_to_id: Dict[int, str] = {}
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        self.current_metric: SimilarityMetric = SimilarityMetric.L2

        if faiss:
            self._initialize_faiss_index(self.current_metric)
        else:
            logger.warning("FAISS library not available. FAISS index not initialized.")

    def _initialize_faiss_index(self, metric: SimilarityMetric) -> None:
        """Initializes or re-initializes the FAISS index."""
        if not faiss:
            logger.error("FAISS not available, cannot initialize index.")
            return

        logger.info(f"Initializing FAISS index with metric: {metric.value} and dimension {self.embedding_dim}")
        if metric == SimilarityMetric.L2:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
        elif metric == SimilarityMetric.COSINE:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Should not happen with Enum
            raise ValueError(f"Unsupported similarity metric: {metric}")

        self.current_metric = metric
        # Wrap with IndexIDMap immediately for custom ID support
        if self.index:
            self.index = faiss.IndexIDMap(self.index)
            logger.info("FAISS index wrapped with IndexIDMap.")


    def _get_embeddings(self, texts: Union[str, List[str]], normalize: bool = False) -> Optional[np.ndarray]:
        """Generates embeddings for given texts. Returns None if model not available."""
        if not self.embedding_model:
            logger.error("Embedding model not loaded. Cannot generate embeddings.")
            return None
        if not texts:
            return np.array([])

        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            if normalize and faiss:
                faiss.normalize_L2(embeddings)
            return embeddings # type: ignore
        except Exception as e:
            logger.error(f"Error during embedding generation: {e}")
            return None

    def add_items(self, items: List[Tuple[str, Union[str, Dict[str, str]], Dict[str, Any]]]) -> None:
        """Adds new items to the database. Updates if ID exists."""
        if not self.index or not faiss:
            logger.error("FAISS index not initialized or FAISS library not available. Cannot add items.")
            return
        if not self.embedding_model:
            logger.error("Embedding model not available. Cannot process items for embedding.")
            return

        ids_to_remove_from_faiss: List[int] = []
        processed_item_data: List[Tuple[int, np.ndarray, str, Dict[str, Any]]] = [] # faiss_id, vector, item_id, metadata

        for item_id, content_or_fields, metadata in items:
            # Check for update
            existing_faiss_id = next((fid for fid, iid in self.idx_to_id.items() if iid == item_id), None)

            target_faiss_id: int
            if existing_faiss_id is not None:
                logger.debug(f"Item ID '{item_id}' exists with FAISS ID {existing_faiss_id}. Preparing for update.")
                ids_to_remove_from_faiss.append(existing_faiss_id)
                target_faiss_id = existing_faiss_id
            else:
                # Generate a new unique int64 FAISS ID. Using hash is simple but can collide.
                # A robust system might use a counter or UUIDs mapped to integers.
                # For this non-runnable version, hash is illustrative.
                target_faiss_id = hash(item_id) % (2**63 - 1)
                while target_faiss_id in self.idx_to_id: # Extremely basic collision avoidance
                    target_faiss_id = (target_faiss_id + 1) % (2**63 - 1)

            # Process content for embedding
            item_vector_list: List[np.ndarray] = []
            if isinstance(content_or_fields, str):
                emb = self._get_embeddings(content_or_fields, normalize=(self.current_metric == SimilarityMetric.COSINE))
                if emb is not None and emb.ndim > 1 : item_vector_list.append(emb[0])
                elif emb is not None and emb.ndim == 1: item_vector_list.append(emb)
            elif isinstance(content_or_fields, dict):
                field_texts = list(content_or_fields.values())
                if field_texts:
                    field_embeddings = self._get_embeddings(field_texts, normalize=(self.current_metric == SimilarityMetric.COSINE))
                    if field_embeddings is not None and len(field_embeddings) > 0:
                        item_vector_list.append(np.mean(field_embeddings, axis=0))

            if not item_vector_list:
                logger.warning(f"Could not generate vector for item ID '{item_id}'. Skipping.")
                continue

            item_vector = item_vector_list[0]
            processed_item_data.append((target_faiss_id, item_vector.astype(np.float32), item_id, metadata))

        # Perform removals if any updates
        if ids_to_remove_from_faiss and isinstance(self.index, faiss.IndexIDMap):
            removed_count = self.index.remove_ids(np.array(ids_to_remove_from_faiss, dtype=np.int64))
            logger.info(f"FAISS reported {removed_count} actual removals for update.")
            for fid_removed in ids_to_remove_from_faiss: # Assume all attempts were meant to clear space
                if fid_removed in self.idx_to_id:
                    old_item_id = self.idx_to_id.pop(fid_removed)
                    # No need to remove from metadata_store here, it will be overwritten or added fresh.

        # Add new/updated items
        if processed_item_data:
            final_vectors = np.array([data[1] for data in processed_item_data], dtype=np.float32)
            final_faiss_ids = np.array([data[0] for data in processed_item_data], dtype=np.int64)

            if final_vectors.size > 0:
                self.index.add_with_ids(final_vectors, final_faiss_ids) # type: ignore
                logger.info(f"Added/Updated {len(processed_item_data)} items in FAISS index.")

                for faiss_id, _, item_id, metadata in processed_item_data:
                    self.idx_to_id[faiss_id] = item_id
                    self.metadata_store[item_id] = metadata
        else:
            logger.info("No valid items to add/update.")


    def remove_items(self, item_ids: List[str]) -> int:
        """Removes items by their unique string IDs."""
        if not self.index or not isinstance(self.index, faiss.IndexIDMap) or not faiss:
            logger.warning("Index not suitable for ID-based removal or FAISS not available.")
            return 0

        faiss_ids_to_remove = [fid for fid, iid in self.idx_to_id.items() if iid in item_ids]
        if not faiss_ids_to_remove:
            return 0

        num_removed = self.index.remove_ids(np.array(faiss_ids_to_remove, dtype=np.int64)) # type: ignore

        # Update local stores
        if num_removed > 0 : # Check if FAISS reported any actual removals
            items_effectively_removed_str : List[str] = []
            for fid_removed in faiss_ids_to_remove: # Iterate through what we tried to remove
                if fid_removed in self.idx_to_id: # If it was in our map
                    item_str_id = self.idx_to_id.pop(fid_removed) # Remove from map
                    self.metadata_store.pop(item_str_id, None) # Remove from metadata
                    items_effectively_removed_str.append(item_str_id)
            logger.info(f"Successfully removed {len(items_effectively_removed_str)} items from internal stores. FAISS reported {num_removed} direct removals.")
            return len(items_effectively_removed_str)
        return 0


    def semantic_search(
        self, query_text: str, top_k: int = 5,
        metric: Optional[SimilarityMetric] = None,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        """Performs semantic similarity search."""
        if not self.index or self.index.ntotal == 0 or not faiss:
            logger.info("Index empty, not initialized, or FAISS not available. Cannot search.")
            return []
        if not self.embedding_model:
            logger.error("Embedding model not available. Cannot perform search.")
            return []

        query_metric = metric if metric else self.current_metric

        query_embedding_arr = self._get_embeddings(query_text, normalize=(query_metric == SimilarityMetric.COSINE))
        if query_embedding_arr is None or query_embedding_arr.size == 0:
            logger.error("Could not generate query embedding.")
            return []

        query_embedding_np = np.array([query_embedding_arr[0] if query_embedding_arr.ndim > 1 else query_embedding_arr], dtype=np.float32)

        num_to_search = top_k * 5 if filter_func else top_k
        distances, faiss_indices = self.index.search(query_embedding_np, k=max(1, num_to_search))

        results: List[Tuple[str, Dict[str, Any], float]] = []
        if faiss_indices.size == 0: return results

        for i in range(faiss_indices[0].shape[0]):
            faiss_idx = faiss_indices[0][i]
            if faiss_idx == -1: continue

            item_id = self.idx_to_id.get(faiss_idx)
            if item_id is None: continue

            metadata = self.metadata_store.get(item_id, {})
            score = float(distances[0][i])

            if query_metric == SimilarityMetric.L2 and score >= 0: score = np.sqrt(score)

            if filter_func and not filter_func(metadata): continue
            results.append((item_id, metadata, score))
            if len(results) >= top_k: break

        results.sort(key=lambda x: x[2], reverse=(query_metric == SimilarityMetric.COSINE))
        return results[:top_k]

    def exact_match_search(
        self, field_name: str, query_value: Any,
        case_sensitive: bool = False, max_results: Optional[int] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Performs exact match search on a specific metadata field."""
        results: List[Tuple[str, Dict[str, Any]]] = []
        for item_id, metadata in self.metadata_store.items():
            field_val = metadata.get(field_name)
            if field_val is None: continue

            match = False
            if isinstance(query_value, str) and isinstance(field_val, str):
                match = (query_value == field_val) if case_sensitive else (query_value.lower() == field_val.lower())
            else:
                match = (query_value == field_val)

            if match:
                results.append((item_id, metadata))
                if max_results is not None and len(results) >= max_results: break
        return results

    def save_index(self, dir_path: Union[str, Path]) -> None:
        """Saves the FAISS index and metadata to disk."""
        if not faiss:
            logger.error("FAISS library not available. Cannot save index.")
            return

        save_dir = Path(dir_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        index_file = save_dir / "faiss_index.idx"
        meta_file = save_dir / "index_meta.pkl"

        if self.index:
            faiss.write_index(self.index, str(index_file))
            logger.info(f"FAISS index saved to {index_file}")
        else: logger.warning("No FAISS index to save.")

        metadata_to_save = {
            "idx_to_id": self.idx_to_id,
            "metadata_store": self.metadata_store,
            "embedding_dim": self.embedding_dim,
            "current_metric_value": self.current_metric.value,
            "embedding_model_name": self.embedding_model_name,
        }
        with open(meta_file, 'wb') as f: pickle.dump(metadata_to_save, f)
        logger.info(f"Index metadata saved to {meta_file}")

    def load_index(self, dir_path: Union[str, Path], embedding_device: Optional[str] = None) -> None:
        """Loads the FAISS index and metadata from disk."""
        if not faiss:
            logger.error("FAISS library not available. Cannot load index.")
            return

        load_dir = Path(dir_path)
        index_file = load_dir / "faiss_index.idx"
        meta_file = load_dir / "index_meta.pkl"

        if not index_file.exists() or not meta_file.exists():
            raise FileNotFoundError(f"Index files not found in {load_dir}.")

        with open(meta_file, 'rb') as f: loaded_meta = pickle.load(f)

        self.embedding_model_name = loaded_meta.get("embedding_model_name", DEFAULT_EMBEDDING_MODEL)
        expected_dim = loaded_meta.get("embedding_dim")

        if SentenceTransformer:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name, device=embedding_device)
                self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension() # type: ignore
                logger.info(f"Reloaded SentenceTransformer model '{self.embedding_model_name}'.")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{self.embedding_model_name}': {e}. Using dim from loaded meta if possible.")
                if expected_dim: self.embedding_dim = expected_dim
        else:
            logger.warning(f"sentence-transformers not available. Using embedding_dim from saved metadata: {expected_dim}")
            if expected_dim: self.embedding_dim = expected_dim


        if expected_dim and self.embedding_dim != expected_dim:
             raise RuntimeError(f"Embedding dimension mismatch! Current/Re-loaded model dim: {self.embedding_dim}, Saved index dim: {expected_dim}")

        self.index = faiss.read_index(str(index_file))
        self.idx_to_id = loaded_meta["idx_to_id"]
        self.metadata_store = loaded_meta["metadata_store"]
        self.current_metric = SimilarityMetric(loaded_meta.get("current_metric_value", SimilarityMetric.L2.value))

        logger.info(f"Index and metadata loaded. Metric: {self.current_metric.value}, Items: {self.index.ntotal if self.index else 'N/A'}")

    def get_item_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        return self.metadata_store.get(item_id)

    @property
    def total_items(self) -> int:
        return len(self.metadata_store)


if __name__ == "__main__":
    # This block is for illustrative purposes. Actual execution is not intended in the current environment.
    logger.info("VectorDBManager: Illustrative __main__ block.")
    logger.info("Dependencies (faiss, sentence-transformers) would be needed for this to run.")

    # INDEX_SAVE_DIR = Path("./temp_vector_db_data_illustrative")
    # INDEX_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # logger.info(f"Illustrative save directory: {INDEX_SAVE_DIR}")

    # try:
    #     logger.info("--- Illustrative: Initializing VectorDBManager ---")
    #     # db_manager = VectorDBManager() # This would attempt to load models
    #     # logger.info(f"Illustrative DB Manager: Dim={db_manager.embedding_dim}, Model={db_manager.embedding_model_name}")

    #     # items_to_add = [
    #     #     ("doc1", "Paris is the capital of France.", {"category": "geography"}),
    #     # ]
    #     # if db_manager.embedding_model and db_manager.index: # Check if deps loaded
    #     #     db_manager.add_items(items_to_add)
    #     #     logger.info(f"Illustrative: Total items after adding: {db_manager.total_items}")
    #     #     results_l2 = db_manager.semantic_search("France capital", top_k=1)
    #     #     logger.info(f"Illustrative: Search results: {results_l2}")
    #     #     db_manager.save_index(INDEX_SAVE_DIR)
    #     #     db_manager_loaded = VectorDBManager()
    #     #     db_manager_loaded.load_index(INDEX_SAVE_DIR)
    #     #     logger.info(f"Illustrative: Total items in loaded DB: {db_manager_loaded.total_items}")
    # except Exception as e:
    #     logger.error(f"Illustrative error: {e}")
    # finally:
        # import shutil
        # if INDEX_SAVE_DIR.exists(): shutil.rmtree(INDEX_SAVE_DIR)
        # logger.info("Illustrative: Cleaned up temp directory.")

    logger.info("VectorDBManager illustrative __main__ finished.")
