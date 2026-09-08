"""HDBSCAN unsupervised clustering module for Pensieve theme discovery.

Finds dense semantic clusters in Sentence-BERT embedding space without forcing
a fixed cluster count. Preserves outlier/noise labels (-1).
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Robust HDBSCAN import (prefer scikit-learn built-in, fallback to standalone hdbscan package)
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "Neither scikit-learn>=1.3.0 (with HDBSCAN) nor hdbscan is installed. "
            "Please run: pip install scikit-learn>=1.3.0"
        ) from exc

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Simple English stop words for human-readable keyword extraction (no LLM)
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're",
    "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves", "just", "like", "get", "got", "also"
}


class ThemeClusterer:
    """Unsupervised density-based theme clustering engine using UMAP + HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int = 25,
        min_samples: Optional[int] = 5,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        copy: bool = True,
        use_umap: bool = True,
        umap_components: int = 5,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.0,
        umap_metric: str = "cosine",
        random_state: int = 42,
    ) -> None:
        """Initialize UMAP + HDBSCAN clustering parameters.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN.
            min_samples: Core neighborhood threshold for HDBSCAN.
            metric: Distance metric for HDBSCAN ('euclidean').
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'.
            copy: Whether to copy input data.
            use_umap: Whether to apply UMAP dimensionality reduction prior to clustering.
            umap_components: Reduced manifold dimension (default: 5).
            umap_neighbors: Local neighborhood size for UMAP (default: 15).
            umap_min_dist: Minimum distance between points in UMAP space (default: 0.0).
            umap_metric: Distance metric for UMAP (default: 'cosine').
            random_state: Random seed for deterministic UMAP projection.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples is not None else min_cluster_size
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.copy = copy

        self.use_umap = use_umap and UMAP_AVAILABLE
        self.umap_components = umap_components
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.umap_metric = umap_metric
        self.random_state = random_state

        self.clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method,
            copy=self.copy,
        )

        self.umap_model: Optional[Any] = None
        self.reduced_embeddings_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.probabilities_: Optional[np.ndarray] = None
        self.embeddings_: Optional[np.ndarray] = None
        self.texts_: Optional[List[str]] = None
        self.cluster_centers_: Dict[int, np.ndarray] = {}

    def fit(self, embeddings: np.ndarray, texts: Optional[List[str]] = None) -> "ThemeClusterer":
        """Fit UMAP + HDBSCAN on L2-normalized embedding matrix.

        Args:
            embeddings: 2D array of shape (N, D).
            texts: Optional list of raw text strings corresponding to embeddings.

        Returns:
            self
        """
        self.embeddings_ = embeddings
        self.texts_ = texts

        cluster_input = embeddings

        if self.use_umap and embeddings.shape[0] > self.umap_neighbors:
            self.umap_model = umap.UMAP(
                n_neighbors=min(self.umap_neighbors, embeddings.shape[0] - 1),
                n_components=self.umap_components,
                min_dist=self.umap_min_dist,
                metric=self.umap_metric,
                random_state=self.random_state,
            )
            self.reduced_embeddings_ = self.umap_model.fit_transform(embeddings)
            cluster_input = self.reduced_embeddings_
        else:
            self.reduced_embeddings_ = None

        self.clusterer.fit(cluster_input)
        self.labels_ = self.clusterer.labels_
        if hasattr(self.clusterer, "probabilities_"):
            self.probabilities_ = self.clusterer.probabilities_
        else:
            self.probabilities_ = np.ones_like(self.labels_, dtype=np.float32)

        self._compute_centroids()
        return self

    def fit_predict(self, embeddings: np.ndarray, texts: Optional[List[str]] = None) -> np.ndarray:
        """Fit HDBSCAN and return assigned cluster labels."""
        self.fit(embeddings, texts=texts)
        assert self.labels_ is not None
        return self.labels_

    def _compute_centroids(self) -> None:
        """Compute normalized mean centroid for each discovered non-noise cluster."""
        self.cluster_centers_ = {}
        if self.labels_ is None or self.embeddings_ is None:
            return

        unique_clusters = set(self.labels_) - {-1}
        for cid in unique_clusters:
            mask = self.labels_ == cid
            cluster_embs = self.embeddings_[mask]
            mean_vec = cluster_embs.mean(axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 0:
                mean_vec = mean_vec / norm
            self.cluster_centers_[int(cid)] = mean_vec

    def get_representative_indices(self, cluster_id: int, top_n: int = 3) -> List[int]:
        """Find indices of the most central samples in a cluster based on centroid cosine similarity.

        Args:
            cluster_id: Cluster ID (>= 0).
            top_n: Number of representative indices to return.

        Returns:
            List of row indices in self.embeddings_.
        """
        if self.labels_ is None or self.embeddings_ is None or cluster_id not in self.cluster_centers_:
            return []

        cluster_indices = np.where(self.labels_ == cluster_id)[0]
        if len(cluster_indices) == 0:
            return []

        centroid = self.cluster_centers_[cluster_id]
        member_embs = self.embeddings_[cluster_indices]
        # Cosine similarity between normalized member embeddings and normalized centroid
        sims = np.dot(member_embs, centroid)
        sorted_order = np.argsort(sims)[::-1]

        top_indices = cluster_indices[sorted_order[:top_n]]
        return [int(idx) for idx in top_indices]

    def _extract_theme_keywords(self, texts: List[str], top_k: int = 4) -> List[str]:
        """Extract dominant non-stopword tokens from texts as simple theme descriptor."""
        words: List[str] = []
        for text in texts:
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
            for t in tokens:
                if t not in STOP_WORDS:
                    words.append(t)
        counts = Counter(words)
        return [word for word, _ in counts.most_common(top_k)]

    def get_cluster_summary(self, top_n_examples: int = 3) -> Dict[str, Any]:
        """Generate comprehensive structural summary of all discovered clusters."""
        if self.labels_ is None:
            raise ValueError("ThemeClusterer must be fitted before calling get_cluster_summary().")

        total_samples = len(self.labels_)
        unique_labels = np.unique(self.labels_)
        num_clusters = int(len(set(unique_labels) - {-1}))
        noise_count = int(np.sum(self.labels_ == -1))
        noise_percentage = round((noise_count / total_samples) * 100, 2) if total_samples > 0 else 0.0

        clusters_info = {}
        for cid in sorted(set(unique_labels) - {-1}):
            cid_int = int(cid)
            size = int(np.sum(self.labels_ == cid))
            rep_indices = self.get_representative_indices(cid_int, top_n=top_n_examples)
            rep_texts = [self.texts_[i] for i in rep_indices] if self.texts_ else []

            all_cluster_indices = np.where(self.labels_ == cid)[0]
            all_cluster_texts = [self.texts_[i] for i in all_cluster_indices] if self.texts_ else []
            keywords = self._extract_theme_keywords(all_cluster_texts)

            clusters_info[str(cid_int)] = {
                "cluster_id": cid_int,
                "size": size,
                "percentage": round((size / total_samples) * 100, 2),
                "representative_indices": rep_indices,
                "representative_examples": rep_texts,
                "descriptive_keywords": keywords,
                "simple_theme_descriptor": " / ".join(keywords) if keywords else f"Cluster {cid_int}",
            }

        return {
            "total_samples": total_samples,
            "num_clusters": num_clusters,
            "noise_samples": noise_count,
            "noise_percentage": noise_percentage,
            "clusters": clusters_info,
        }
