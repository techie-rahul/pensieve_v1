"""Semantic embedding module for Pensieve Phase 4 RAG Grounding.

Uses sentence-transformers/all-MiniLM-L6-v2 (reusing the Phase 2 embedding approach)
to generate 384-dimensional L2-normalized dense representations for both concept
documents and search queries.
"""

from typing import List, Optional, Union
import numpy as np

from ml.rag.documents import ConceptDocument
from ml.theme.embedding import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION, TextEmbedder, get_embedder


class ConceptEmbedder:
    """Embeds concept documents and retrieval queries using all-MiniLM-L6-v2."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        embedder: Optional[TextEmbedder] = None,
    ) -> None:
        """Initialize concept embedder.

        Args:
            model_name_or_path: Hugging Face model checkpoint name.
            device: 'cuda', 'cpu', or None for auto-detection.
            embedder: Optional pre-instantiated TextEmbedder to share model weights.
        """
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = TextEmbedder(
                model_name_or_path=model_name_or_path,
                device=device,
                normalize_embeddings=True,
            )
        self.embedding_dim = self.embedder.embedding_dim

    def embed_document(self, doc: ConceptDocument) -> np.ndarray:
        """Embed a single ConceptDocument into an L2-normalized 1D vector.

        Args:
            doc: ConceptDocument instance.

        Returns:
            1D numpy array of shape (384,) with L2 norm == 1.0.
        """
        text = doc.to_embedding_text()
        return self.embedder.embed_text(text)

    def embed_documents(
        self,
        docs: List[ConceptDocument],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Embed a collection of ConceptDocuments into an L2-normalized 2D matrix.

        Args:
            docs: List of ConceptDocument instances.
            batch_size: Batch size for model inference.
            show_progress_bar: Whether to display progress bar.

        Returns:
            2D numpy array of shape (N, 384) with unit-norm rows.
        """
        texts = [doc.to_embedding_text() for doc in docs]
        return self.embedder.embed_texts(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a retrieval query into an L2-normalized 1D vector.

        Args:
            query: Query string.

        Returns:
            1D numpy array of shape (384,) with L2 norm == 1.0.
        """
        return self.embedder.embed_text(query)


# Global singleton instance for efficient reuse across retrieval calls
_DEFAULT_CONCEPT_EMBEDDER: Optional[ConceptEmbedder] = None


def get_concept_embedder() -> ConceptEmbedder:
    """Get or create singleton ConceptEmbedder instance."""
    global _DEFAULT_CONCEPT_EMBEDDER
    if _DEFAULT_CONCEPT_EMBEDDER is None:
        _DEFAULT_CONCEPT_EMBEDDER = ConceptEmbedder(embedder=get_embedder())
    return _DEFAULT_CONCEPT_EMBEDDER
