"""Pensieve ML — Phase 4 RAG / Knowledge Grounding Retrieval Layer.

Provides structured concept documents, semantic embedding via all-MiniLM-L6-v2,
local FAISS vector indexing, deterministic query construction, and threshold-gated retrieval.

Strict non-diagnostic ethical framing; zero LLMs.
"""

from ml.rag.documents import (
    ConceptDocument,
    KnowledgeBase,
    DEVELOPMENT_CONCEPTS_DATA,
)
from ml.rag.embeddings import (
    ConceptEmbedder,
    get_concept_embedder,
)
from ml.rag.index import (
    VectorIndex,
)
from ml.rag.retriever import (
    ConceptRetriever,
    DeterministicQueryBuilder,
    PatternSignal,
    THEME_CLUSTER_NAMES,
)

__all__ = [
    "ConceptDocument",
    "KnowledgeBase",
    "DEVELOPMENT_CONCEPTS_DATA",
    "ConceptEmbedder",
    "get_concept_embedder",
    "VectorIndex",
    "ConceptRetriever",
    "DeterministicQueryBuilder",
    "PatternSignal",
    "THEME_CLUSTER_NAMES",
]
