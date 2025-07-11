# examples/vectorDB_vectorDB_manager_example.py

import sys
import os
import json # For metadata

# Adjust the path to include the src directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import the module
try:
    from src.vectorDB.vectorDB_manager import VectorDBManager
    # May also need an embedding model, let's assume it's passed or configured
    # For this example, we can create a dummy embedding function/class if needed by VectorDBManager
    # from src.model.LLMmodel import LLMmodel # If it provides embeddings
    NEEDS_EMBEDDING_MODEL = True # Assume true, adjust if VectorDBManager handles it internally
except ImportError:
    print("Warning: Could not import VectorDBManager from src.vectorDB.vectorDB_manager.")
    print("Using a dummy VectorDBManager class and a dummy embedding function for demonstration.")
    NEEDS_EMBEDDING_MODEL = True # Still true for the dummy

    class VectorDBManager:
        """
        Dummy VectorDBManager class for demonstration.
        Simulates interactions with a vector database.
        """
        def __init__(self, db_type="in_memory_dummy", config=None, embedding_function=None):
            self.db_type = db_type
            self.config = config if config else {}
            self.collections = {} # Store dummy collections: collection_name -> {"vectors": [], "metadata": [], "texts": []}
            self.embedding_function = embedding_function

            if not self.embedding_function and NEEDS_EMBEDDING_MODEL:
                print("Warning: No embedding function provided to DummyVectorDBManager. Search will be very basic.")
                # Fallback to a very simple "embedding" if none given for the dummy
                self.embedding_function = lambda texts: [[len(text) * 0.01, len(text.split()) * 0.05] for text in texts]


            print(f"Dummy VectorDBManager initialized. Type: {self.db_type}, Config: {self.config}")
            if self.embedding_function:
                print("Embedding function provided/initialized.")

        def create_collection(self, collection_name, vector_size=None, distance_metric="cosine"):
            """
            Dummy method to create a new collection/index.
            vector_size might be inferred from embedding_function or required.
            """
            if collection_name in self.collections:
                print(f"Error: Collection '{collection_name}' already exists (dummy).")
                return False

            # Simulate inferring vector_size from a test embedding if not provided
            if not vector_size and self.embedding_function:
                try:
                    sample_embedding = self.embedding_function(["test"])[0]
                    vector_size = len(sample_embedding)
                except:
                    print("Could not infer vector size from embedding function. Please provide it.")
                    return False
            elif not vector_size:
                print("Error: vector_size is required if it cannot be inferred (dummy).")
                return False


            self.collections[collection_name] = {
                "vectors": [], "metadata": [], "texts": [],
                "vector_size": vector_size, "distance_metric": distance_metric
            }
            print(f"\nCollection '{collection_name}' created (dummy) with vector_size={vector_size}, metric='{distance_metric}'.")
            return True

        def delete_collection(self, collection_name):
            if collection_name not in self.collections:
                print(f"Error: Collection '{collection_name}' not found for deletion (dummy).")
                return False
            del self.collections[collection_name]
            print(f"\nCollection '{collection_name}' deleted (dummy).")
            return True

        def add_documents(self, collection_name, documents, metadatas=None, ids=None):
            """
            Dummy method to add documents (texts) to a collection.
            'documents' is a list of strings.
            'metadatas' (optional) is a list of dictionaries.
            'ids' (optional) is a list of unique string/int identifiers.
            """
            if collection_name not in self.collections:
                print(f"Error: Collection '{collection_name}' does not exist for adding documents (dummy).")
                return False
            if not self.embedding_function:
                print("Error: No embedding function available to process documents (dummy).")
                return False

            print(f"\nAdding {len(documents)} documents to collection '{collection_name}' (dummy)...")

            try:
                doc_embeddings = self.embedding_function(documents)
            except Exception as e:
                print(f"Error during dummy embedding generation: {e}")
                return 0

            collection = self.collections[collection_name]
            added_count = 0
            for i, text in enumerate(documents):
                vector = doc_embeddings[i]
                # Basic check for vector dimension compatibility
                if len(vector) != collection["vector_size"]:
                    print(f"Error: Embedding for document '{text[:20]}...' has dimension {len(vector)}, expected {collection['vector_size']}. Skipping.")
                    continue

                collection["vectors"].append(vector)
                collection["texts"].append(text)
                collection["metadata"].append(metadatas[i] if metadatas and i < len(metadatas) else {"source": f"doc_{len(collection['texts'])}"})
                # Real DBs would handle ID generation or use provided IDs
                added_count += 1

            print(f"{added_count} documents (with embeddings) added to '{collection_name}' (dummy).")
            return added_count

        def search_similar(self, collection_name, query_text=None, query_vector=None, top_k=3, filter_criteria=None):
            """
            Dummy method to search for similar documents.
            Can search by query_text (needs embedding_function) or by query_vector.
            """
            if collection_name not in self.collections:
                print(f"Error: Collection '{collection_name}' does not exist for search (dummy).")
                return []

            collection_data = self.collections[collection_name]
            if not collection_data["vectors"]:
                print(f"Collection '{collection_name}' is empty. No search results (dummy).")
                return []

            target_vector = None
            if query_vector:
                target_vector = query_vector
                print(f"\nSearching in '{collection_name}' for top {top_k} similar to provided vector (dummy)...")
            elif query_text and self.embedding_function:
                print(f"\nSearching in '{collection_name}' for top {top_k} similar to query: '{query_text}' (dummy)...")
                try:
                    target_vector = self.embedding_function([query_text])[0]
                except Exception as e:
                    print(f"Error generating query embedding: {e}")
                    return []
            else:
                print("Error: Must provide query_text (with embedding_function) or query_vector for search (dummy).")
                return []

            if len(target_vector) != collection_data["vector_size"]:
                print(f"Error: Query vector dim ({len(target_vector)}) != collection dim ({collection_data['vector_size']}).")
                return []

            # Dummy similarity search: just return the first few items or random items,
            # as actual vector math is complex for a simple dummy.
            # A slightly better dummy would calculate a mock distance.

            results = []
            for i, vec in enumerate(collection_data["vectors"]):
                # Simplified dummy distance (e.g., sum of absolute differences, not actual cosine)
                # This is NOT a real distance metric, just for making the dummy respond.
                dummy_distance = sum(abs(tv - v) for tv, v in zip(target_vector, vec))

                # Apply dummy filter if provided
                passes_filter = True
                if filter_criteria:
                    doc_meta = collection_data["metadata"][i]
                    for key, value in filter_criteria.items():
                        if doc_meta.get(key) != value:
                            passes_filter = False
                            break
                if not passes_filter:
                    continue

                results.append({
                    "id": f"dummy_id_{i}", # Real DBs provide actual IDs
                    "text": collection_data["texts"][i],
                    "metadata": collection_data["metadata"][i],
                    "score": 1.0 / (1.0 + dummy_distance) # Higher score for smaller "distance"
                })

            # Sort by dummy score (descending) and take top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            final_results = results[:top_k]

            print(f"Found {len(final_results)} similar documents (dummy):")
            for res in final_results:
                print(f"  Text: '{res['text'][:50]}...', Score: {res['score']:.4f}, Meta: {res['metadata']}")
            return final_results

        def list_collections(self):
            print("\nAvailable collections (dummy):")
            if not self.collections:
                print("  No collections found.")
                return []
            for name in self.collections.keys():
                print(f"  - {name} (Size: {self.collections[name]['vector_size']}, Metric: {self.collections[name]['distance_metric']})")
            return list(self.collections.keys())

# Dummy Embedding Function (if not importing a real one)
def dummy_embedding_function(texts: list[str]) -> list[list[float]]:
    """ A very basic dummy embedding function. """
    print(f"Generating dummy embeddings for {len(texts)} texts...")
    embeddings = []
    for text in texts:
        # Create a fixed-size embedding based on character codes and length
        # This is NOT a meaningful embedding, just for structure.
        # Let's say our dummy vector size is 5.
        emb = [0.0] * 5
        if text:
            emb[0] = len(text) / 100.0 # Normalized length
            emb[1] = sum(ord(c) for c in text[:3]) / 1000.0 if len(text) >=3 else 0.0
            emb[2] = sum(ord(c) for c in text[-3:]) / 1000.0 if len(text) >=3 else 0.0
            emb[3] = text.count(' ') / 10.0 # Normalized space count
            emb[4] = (ord(text[0]) % 10) / 10.0 if text else 0.0
        embeddings.append([round(x, 4) for x in emb]) # Round for cleaner printing
    return embeddings


def main():
    print("--- VectorDBManager Module Example ---")

    # Configuration for the VectorDB
    # This would depend on the specific vector DB being used (Chroma, FAISS, Pinecone, etc.)
    vdb_config = {
        "path": "./my_vector_db_storage", # For local DBs like Chroma/FAISS
        "api_key": "YOUR_PINECONE_OR_OTHER_API_KEY", # For cloud DBs
        "environment": "us-west1-gcp" # For cloud DBs
    }

    # Embedding model/function setup
    # In a real app, you'd use a sentence transformer, OpenAI embeddings, etc.
    # For this example, we use the dummy_embedding_function defined above.
    embed_func = dummy_embedding_function
    # If using LLMmodel for embeddings:
    # llm_for_embeddings = LLMmodel(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embed_func = llm_for_embeddings.get_text_embedding # Assuming this method exists and returns list of floats

    try:
        vdb_manager = VectorDBManager(
            db_type="chroma_simulated", # Or "faiss_simulated", "pinecone_simulated"
            config=vdb_config,
            embedding_function=embed_func
        )
    except NameError: # Fallback for dummy
        vdb_manager = VectorDBManager(embedding_function=embed_func) # Use default dummy params

    # Example 1: Create a new collection
    print("\n[Example 1: Create Collection]")
    collection_name = "nova_sql_documents"
    # Vector size might be inferred by the dummy if embedding_function is good.
    # For our basic dummy_embedding_function, it produces size 5 embeddings.
    vdb_manager.create_collection(collection_name=collection_name, vector_size=5, distance_metric="euclidean_dummy")

    # Example 2: List collections
    print("\n[Example 2: List Collections]")
    vdb_manager.list_collections()

    # Example 3: Add documents to the collection
    print("\n[Example 3: Add Documents]")
    docs_to_add = [
        "NovaSQLAgent is a tool for converting natural language to SQL queries.",
        "It uses large language models and vector databases for context retrieval.",
        "The core components include a translator, a sandbox, and a memory module.",
        "Database schema information is crucial for accurate SQL generation.",
        "Users can interact with NovaSQLAgent via a command-line interface or API."
    ]
    doc_metadatas = [
        {"source": "introduction.md", "type": "general_info"},
        {"source": "architecture.md", "type": "technical_detail"},
        {"source": "components.md", "type": "technical_detail"},
        {"source": "usage_guide.md", "type": "user_facing"},
        {"source": "interaction_modes.md", "type": "user_facing"}
    ]
    vdb_manager.add_documents(collection_name, docs_to_add, metadatas=doc_metadatas)

    # Example 4: Search for similar documents using query text
    print("\n[Example 4: Search by Query Text]")
    query = "How does NovaSQLAgent generate SQL?"
    search_results_text = vdb_manager.search_similar(collection_name, query_text=query, top_k=2)
    # for result in search_results_text:
    #     print(f"  Found: '{result['text'][:60]}...' (Score: {result['score']:.3f}) Meta: {result['metadata']}")

    # Example 5: Search with a filter
    print("\n[Example 5: Search with Filter]")
    query_filter = "Find components of the agent"
    filter_meta = {"type": "technical_detail"}
    search_results_filtered = vdb_manager.search_similar(
        collection_name,
        query_text=query_filter,
        top_k=2,
        filter_criteria=filter_meta
    )

    # Example 6: Search using a pre-computed query vector
    # (Requires getting an embedding for a query first)
    print("\n[Example 6: Search by Query Vector]")
    if embed_func:
        vector_query_text = "information about user interaction"
        # Get the embedding for this text using the same function the DB uses
        query_emb_vector = embed_func([vector_query_text])[0]

        search_results_vector = vdb_manager.search_similar(
            collection_name,
            query_vector=query_emb_vector,
            top_k=1
        )
    else:
        print("Skipping search by vector as no embedding function is available for the query.")


    # Example 7: Delete the collection
    print("\n[Example 7: Delete Collection]")
    vdb_manager.delete_collection(collection_name)
    vdb_manager.list_collections() # Verify deletion


    print("\n--- VectorDBManager Module Example Complete ---")

if __name__ == "__main__":
    main()
