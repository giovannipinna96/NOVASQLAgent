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
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)

SimilarityMetric = Literal["cosine", "l2"]


@dataclass
class VectorRecord:
    id: str
    fields: Dict[str, Any]
    vector: np.ndarray


class VectorDBManager:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_path: Optional[Union[str, Path]] = None,
        fields: Optional[List[str]] = None,
        metric: SimilarityMetric = "cosine"
    ):
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.db_path = Path(db_path) if db_path else None
        self.fields = fields or []
        self.metric = metric
        self.vectorstore: Optional[FAISS] = None

        if self.db_path and (self.db_path / "index.faiss").exists():
            self.load()

    def add(self, records: List[Dict[str, Any]]) -> None:
        docs = []
        for rec in records:
            meta = rec.get("fields", {}).copy()
            meta["id"] = rec["id"]
            text = " ".join(str(meta.get(f, "")) for f in self.fields)
            docs.append(Document(page_content=text, metadata=meta))
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        else:
            self.vectorstore.add_documents(docs)

    def remove(self, ids: List[str]) -> None:
        if not self.vectorstore:
            return
        keep_docs = [doc for doc in self.vectorstore.docstore._dict.values()
                     if doc.metadata.get("id") not in ids]
        self.vectorstore = FAISS.from_documents(keep_docs, self.embeddings)

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        if not self.vectorstore:
            raise RuntimeError("No vectorstore to save.")
        path = Path(path) if path else self.db_path
        if not path:
            raise ValueError("No path provided.")
        path.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(path))

    def load(self, path: Optional[Union[str, Path]] = None) -> None:
        path = Path(path) if path else self.db_path
        if not path:
            raise ValueError("No path provided.")
        self.vectorstore = FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)

    def search(
        self,
        query: str,
        top_k: int = 5,
        fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            return []

        results = []
        if fields:
            for field in fields:
                docs = [(doc, str(doc.metadata.get(field, "")))
                        for doc in self.vectorstore.docstore._dict.values()]
                qvec = np.array(self.embeddings.embed_query(query))
                vecs = np.vstack([self.embeddings.embed_query(text) for _, text in docs])

                if self.metric == "cosine":
                    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
                    qvec = qvec / np.linalg.norm(qvec)
                    sims = vecs @ qvec
                else:
                    sims = -np.linalg.norm(vecs - qvec, axis=1)

                top_idx = np.argsort(sims)[::-1][:top_k]
                for idx in top_idx:
                    doc, _ = docs[idx]
                    results.append({
                        "id": doc.metadata.get("id"),
                        "field": field,
                        "score": float(sims[idx]),
                        "meta": doc.metadata
                    })
        else:
            docs_and_scores = self.vectorstore.similarity_search_with_score(query, k=top_k)
            for doc, score in docs_and_scores:
                results.append({
                    "id": doc.metadata.get("id"),
                    "score": float(score),
                    "meta": doc.metadata
                })
        return results

    def keyword_search(
        self,
        query: str,
        fields: Optional[List[str]] = None,
        exact: bool = False,
        logic: Literal["or", "and"] = "or"
    ) -> List[Dict[str, Any]]:
        if not self.vectorstore:
            return []
        results = []
        for doc in self.vectorstore.docstore._dict.values():
            meta = doc.metadata
            matches = []
            for field in (fields or self.fields):
                value = str(meta.get(field, ""))
                match = query == value if exact else query.lower() in value.lower()
                matches.append(match)
            if (logic == "or" and any(matches)) or (logic == "and" and all(matches)):
                results.append({"id": meta.get("id"), "meta": meta})
        return results



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VectorDBManager CLI usage example.")
    parser.add_argument('--db_path', type=str, default="./vector_db", help="Path to save/load the vector DB.")
    parser.add_argument('--test_all', action='store_true', help="Test all VectorDBManager functionality.")
    args = parser.parse_args()

    db = VectorDBManager(fields=["title", "content"], db_path=args.db_path)

    if args.test_all:
        print("--- Testing all VectorDBManager functionality ---")
        records = [
            {"id": "1", "fields": {"title": "Python AI", "content": "Build a vector database with FAISS."}},
            {"id": "2", "fields": {"title": "Machine Learning", "content": "Similarity search and embeddings."}},
            {"id": "3", "fields": {"title": "Vector Search", "content": "Efficient semantic retrieval."}},
            {"id": "4", "fields": {"title": "Remove Me", "content": "This record will be removed."}},
        ]
        db.add(records)
        db.save()
        db.load()
        
        print("\nBefore removal:")
        for r in db.search(query="vector", top_k=10):
            print(r)

        print("\nSemantic Search (Global):")
        for r in db.search(query="semantic search", top_k=2):
            print(r)

        print("\nColumn-wise Search:")
        for r in db.search(query="vector", top_k=1, fields=["title", "content"]):
            print(r)

        print("\nKeyword Search (partial match):")
        for r in db.keyword_search(query="Python", fields=["title"]):
            print(r)

        print("\nKeyword Search (exact AND):")
        for r in db.keyword_search(query="Build a vector database with FAISS.", fields=["content"], exact=True, logic="and"):
            print(r)

        db.remove(["4"])
        db.save()
        db.load()

        print("\nAfter removal:")
        for r in db.search(query="vector", top_k=10):
            print(r)

        print("--- All tests completed ---")
