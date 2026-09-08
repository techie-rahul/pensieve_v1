"""Semantic embedding engine for Pensieve using Sentence-BERT.

Generates 384-dimensional L2-normalized dense vectors for arbitrary text.
"""

from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class TextEmbedder:
    """Wrapper for Sentence-BERT text embedding with L2 normalization."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize SentenceTransformer model.

        Args:
            model_name_or_path: Model checkpoint name or local path.
            device: 'cuda', 'cpu', or None (auto-detected).
            normalize_embeddings: Whether to apply L2 normalization to output embeddings.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name_or_path
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name_or_path, device=self.device)
        if hasattr(self.model, "get_embedding_dimension"):
            self.embedding_dim = self.model.get_embedding_dimension()
        else:
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate a single normalized 1D embedding vector for input text.

        Args:
            text: Input string.

        Returns:
            1D numpy array of shape (384,) with L2 norm == 1.0.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            # Fallback for empty text: zero vector
            return np.zeros(self.embedding_dim, dtype=np.float32)

        emb = self.model.encode(
            cleaned,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 64,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Generate normalized 2D embedding matrix for a list of texts.

        Args:
            texts: List of input strings.
            batch_size: Batch size for encoding.
            show_progress_bar: Whether to display tqdm progress bar.

        Returns:
            2D numpy array of shape (N, 384) with unit norm rows.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Replace empty/whitespace strings with space so sentence-transformers encodes without error
        cleaned = [t.strip() if (t and t.strip()) else " " for t in texts]

        embeddings = self.model.encode(
            cleaned,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
        return embeddings.astype(np.float32)


# Global singleton instance for easy reuse
_DEFAULT_EMBEDDER: Optional[TextEmbedder] = None


def get_embedder() -> TextEmbedder:
    """Get or create singleton TextEmbedder instance."""
    global _DEFAULT_EMBEDDER
    if _DEFAULT_EMBEDDER is None:
        _DEFAULT_EMBEDDER = TextEmbedder()
    return _DEFAULT_EMBEDDER
