"""Pensieve ML - Theme Discovery Module.

Phase 2 implementation for unsupervised semantic theme extraction.
Combines Sentence-BERT embeddings with HDBSCAN density clustering.
"""

from ml.theme.embedding import TextEmbedder, get_embedder
from ml.theme.clustering import ThemeClusterer

__all__ = [
    "TextEmbedder",
    "get_embedder",
    "ThemeClusterer",
]
