"""Local vector index implementation for Pensieve Phase 4 RAG Grounding.

Provides a fast, lightweight local FAISS index (IndexFlatIP) for exact cosine similarity
retrieval over L2-normalized embeddings, with a clean NumPy fallback.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


class VectorIndex:
    """Interface and reference implementation for local vector indexing and retrieval."""

    def __init__(self, dimension: int = 384, use_faiss: bool = True) -> None:
        """Initialize vector index.

        Args:
            dimension: Dimensionality of embeddings (default: 384 for all-MiniLM-L6-v2).
            use_faiss: Whether to use FAISS if available.
        """
        self.dimension = dimension
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        self.doc_ids: List[str] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._faiss_index: Optional[Any] = None

        if self.use_faiss:
            # IndexFlatIP computes exact Inner Product (equivalent to Cosine Similarity for normalized vectors)
            self._faiss_index = faiss.IndexFlatIP(dimension)

    def is_faiss_active(self) -> bool:
        """Return True if FAISS backend is actively powering the index."""
        return self.use_faiss and (self._faiss_index is not None)

    def add_documents(self, doc_ids: List[str], embeddings: np.ndarray) -> None:
        """Add document IDs and their normalized embeddings to the index.

        Args:
            doc_ids: List of unique document identifiers.
            embeddings: 2D numpy array of shape (N, dimension).
        """
        if len(doc_ids) == 0:
            return

        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of doc_ids ({len(doc_ids)}) and embeddings rows ({embeddings.shape[0]})."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
            )

        # Ensure float32 contiguous array
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Update ID list
        self.doc_ids.extend(doc_ids)

        if self.use_faiss and self._faiss_index is not None:
            self._faiss_index.add(vectors)
        else:
            if self._embeddings_matrix is None:
                self._embeddings_matrix = vectors
            else:
                self._embeddings_matrix = np.vstack([self._embeddings_matrix, vectors])

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Search the index for the top-k most similar documents.

        Args:
            query_embedding: 1D or 2D normalized query vector.
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of (doc_id, similarity_score) tuples sorted in descending order of similarity.
        """
        if len(self.doc_ids) == 0:
            return []

        effective_k = min(top_k, len(self.doc_ids))
        if effective_k <= 0:
            return []

        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension {query.shape[1]} does not match index dimension {self.dimension}."
            )

        if self.use_faiss and self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query, effective_k)
            results: List[Tuple[str, float]] = []
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(self.doc_ids):
                    results.append((self.doc_ids[idx], float(score)))
            return results
        else:
            # Exact NumPy cosine similarity fallback
            scores = np.dot(self._embeddings_matrix, query[0])
            top_indices = np.argsort(scores)[::-1][:effective_k]
            return [(self.doc_ids[idx], float(scores[idx])) for idx in top_indices]

    def size(self) -> int:
        """Return number of documents indexed."""
        return len(self.doc_ids)

    def clear(self) -> None:
        """Reset the index."""
        self.doc_ids = []
        self._embeddings_matrix = None
        if self.use_faiss and self._faiss_index is not None:
            self._faiss_index.reset()
