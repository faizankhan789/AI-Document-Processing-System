"""
Semantic Search Module
Implements document retrieval using FAISS and SentenceTransformers
"""

import os
import numpy as np
import faiss
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Enable offline mode if models are cached
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize semantic search with a SentenceTransformer model."""
        print(f"Loading search model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.documents: Dict[str, str] = {}
        self.doc_names: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_documents(self, documents: Dict[str, str]) -> None:
        """
        Index documents for semantic search.

        Args:
            documents: Dictionary mapping filename to document text
        """
        self.documents = documents
        self.doc_names = list(documents.keys())

        print(f"Indexing {len(documents)} documents...")

        # Get document texts
        texts = list(documents.values())

        # Compute embeddings
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # Create FAISS index (Inner Product for cosine similarity with normalized vectors)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype(np.float32))

        print(f"Indexed {self.index.ntotal} documents")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        """
        Search for documents semantically similar to the query.

        Args:
            query: Search query string
            top_k: Number of results to return

        Returns:
            List of tuples (filename, similarity_score, text_snippet)
        """
        if self.index is None:
            raise ValueError("No documents indexed. Call index_documents first.")

        # Encode query
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # Search
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding.astype(np.float32), k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.doc_names):
                doc_name = self.doc_names[idx]
                text = self.documents[doc_name]
                snippet = text[:200] + "..." if len(text) > 200 else text
                results.append((doc_name, float(score), snippet))

        return results

    def save_index(self, path: str) -> None:
        """Save the FAISS index to disk."""
        if self.index is not None:
            faiss.write_index(self.index, path)
            print(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load a FAISS index from disk."""
        self.index = faiss.read_index(path)
        print(f"Index loaded from {path}")


def demo_search(search_engine: SemanticSearch) -> None:
    """Run interactive search demo."""
    print("\n" + "="*60)
    print("SEMANTIC SEARCH DEMO")
    print("="*60)
    print("Type your search query (or 'quit' to exit)")
    print("-"*60)

    example_queries = [
        "Find all documents mentioning payments due in January",
        "Documents about electricity usage",
        "Professional with software engineering experience",
        "Invoice from a company"
    ]

    print("\nExample queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    print()

    while True:
        query = input("\nSearch query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue

        results = search_engine.search(query, top_k=5)

        print(f"\nResults for: '{query}'")
        print("-"*40)
        for i, (doc_name, score, snippet) in enumerate(results, 1):
            print(f"\n{i}. {doc_name} (score: {score:.3f})")
            print(f"   {snippet}")
