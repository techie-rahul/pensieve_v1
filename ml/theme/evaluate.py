"""Unsupervised clustering evaluation pipeline for Pensieve theme discovery.

Evaluates Sentence-BERT + HDBSCAN on the GoEmotions text corpus (strictly treating
text as an unlabelled text collection, NOT using emotion annotations as themes).

Computes clustering diagnostics:
- Number of discovered clusters
- Number & percentage of noise/outlier samples
- Cluster size distribution
- Silhouette Score (clustered samples only)
- Davies-Bouldin Index (clustered samples only)
- Calinski-Harabasz Score (clustered samples only)
- Representative central entries per cluster
- Human-readable theme keywords (without LLM)

Saves results directly to ml/theme_evaluation_results.json.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from datasets import load_dataset
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.theme.clustering import ThemeClusterer
from ml.theme.embedding import TextEmbedder

DEFAULT_OUTPUT_PATH = "ml/theme_evaluation_results.json"


def compute_clustering_diagnostics(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute intrinsic clustering diagnostics on clustered (non-noise) samples.

    Note: These metrics measure geometric compactness and separation in embedding
    space, NOT semantic correctness or psychological truth.
    """
    total_samples = len(labels)
    clustered_mask = labels != -1
    num_clustered = int(np.sum(clustered_mask))
    num_noise = int(np.sum(~clustered_mask))
    noise_percentage = round((num_noise / total_samples) * 100, 2) if total_samples > 0 else 0.0

    unique_clusters = sorted(set(labels) - {-1})
    num_clusters = len(unique_clusters)

    # Compute cluster size distribution
    cluster_sizes = {int(c): int(np.sum(labels == c)) for c in unique_clusters}

    diagnostics: Dict[str, Any] = {
        "total_samples": total_samples,
        "num_clusters": num_clusters,
        "clustered_samples": num_clustered,
        "noise_samples": num_noise,
        "noise_percentage": noise_percentage,
        "cluster_size_distribution": cluster_sizes,
        "metrics_disclaimer": (
            "These metrics evaluate geometric compactness and separation in 384-d "
            "embedding space. They do not measure semantic or psychological ground truth."
        ),
    }

    # Silhouette, Davies-Bouldin, and Calinski-Harabasz require at least 2 distinct clusters
    if num_clusters >= 2 and num_clustered >= 2:
        clustered_embs = embeddings[clustered_mask]
        clustered_labels = labels[clustered_mask]

        # Use euclidean metric on L2-normalized embeddings
        sil = float(silhouette_score(clustered_embs, clustered_labels, metric="euclidean"))
        db = float(davies_bouldin_score(clustered_embs, clustered_labels))
        ch = float(calinski_harabasz_score(clustered_embs, clustered_labels))

        diagnostics["silhouette_score_clustered_only"] = round(sil, 4)
        diagnostics["davies_bouldin_index_clustered_only"] = round(db, 4)
        diagnostics["calinski_harabasz_score_clustered_only"] = round(ch, 2)
    else:
        diagnostics["silhouette_score_clustered_only"] = None
        diagnostics["davies_bouldin_index_clustered_only"] = None
        diagnostics["calinski_harabasz_score_clustered_only"] = None

    return diagnostics


def run_theme_evaluation(
    output_path: str = DEFAULT_OUTPUT_PATH,
    min_cluster_size: int = 15,
    min_samples: Optional[int] = 5,
    metric: str = "euclidean",
    cluster_selection_method: str = "eom",
    split: str = "test",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full unsupervised theme discovery evaluation pipeline on GoEmotions texts."""
    print("=" * 75)
    print("PENSIEVE ML - THEME DISCOVERY EVALUATION PIPELINE (SENTENCE-BERT + HDBSCAN)")
    print("=" * 75)
    print(f"Timestamp UTC: {datetime.now(timezone.utc).isoformat()}")

    # 1. Load dataset texts
    print(f"\n[1/4] Loading GoEmotions '{split}' split text corpus...")
    print("NOTE: GoEmotions emotion labels are strictly ignored; only raw text is processed.")
    ds = load_dataset("go_emotions")
    texts = ds[split]["text"]
    if max_samples and max_samples < len(texts):
        texts = texts[:max_samples]
    print(f"Loaded {len(texts)} texts for theme discovery.")

    # 2. Generate Sentence-BERT embeddings
    print("\n[2/4] Generating L2-normalized Sentence-BERT embeddings (all-MiniLM-L6-v2)...")
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(texts, batch_size=64, show_progress_bar=True)
    print(f"Generated embedding matrix: shape {embeddings.shape}, dtype {embeddings.dtype}")

    # 3. Fit HDBSCAN clusterer
    print(f"\n[3/4] Fitting HDBSCAN clusterer (min_cluster_size={min_cluster_size}, min_samples={min_samples}, metric={metric})...")
    clusterer = ThemeClusterer(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    labels = clusterer.fit_predict(embeddings, texts=texts)

    # 4. Compute clustering diagnostics
    print("\n[4/4] Computing clustering diagnostics and representative themes...")
    diagnostics = compute_clustering_diagnostics(embeddings, labels)
    summary = clusterer.get_cluster_summary(top_n_examples=3)

    print("\n" + "-" * 75)
    print("CLUSTERING DIAGNOSTIC REPORT")
    print("-" * 75)
    print(f"Total samples:                 {diagnostics['total_samples']}")
    print(f"Discovered clusters:           {diagnostics['num_clusters']}")
    print(f"Clustered samples:             {diagnostics['clustered_samples']} ({100 - diagnostics['noise_percentage']:.2f}%)")
    print(f"Noise / outlier samples (-1):  {diagnostics['noise_samples']} ({diagnostics['noise_percentage']}%)")
    print(f"Silhouette Score (clustered):  {diagnostics.get('silhouette_score_clustered_only')}")
    print(f"Davies-Bouldin Index:          {diagnostics.get('davies_bouldin_index_clustered_only')}")
    print(f"Calinski-Harabasz Score:       {diagnostics.get('calinski_harabasz_score_clustered_only')}")
    print("-" * 75)

    print("\nTop Discovered Clusters (by size):")
    sorted_clusters = sorted(summary["clusters"].values(), key=lambda x: x["size"], reverse=True)
    for c in sorted_clusters[:8]:
        print(f"\nCluster {c['cluster_id']} (size: {c['size']}, {c['percentage']}%):")
        print(f"  Keywords / Theme: {c['simple_theme_descriptor']}")
        print(f"  Top Representative Entry: \"{c['representative_examples'][0]}\"")

    results_payload = {
        "metadata": {
            "pipeline": "Sentence-BERT + HDBSCAN Unsupervised Theme Discovery",
            "model_name": embedder.model_name,
            "embedding_dimension": embedder.embedding_dim,
            "normalized_embeddings": embedder.normalize_embeddings,
            "dataset": "go_emotions (raw text only)",
            "split": split,
            "num_samples": len(texts),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
        "hdbscan_configuration": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "metric": metric,
            "cluster_selection_method": cluster_selection_method,
        },
        "diagnostics": diagnostics,
        "discovered_clusters": summary["clusters"],
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSaved theme evaluation results to: {output_path}")
    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sentence-BERT + HDBSCAN theme discovery")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Output JSON path")
    parser.add_argument("--min-cluster-size", type=int, default=15, help="HDBSCAN min_cluster_size")
    parser.add_argument("--min-samples", type=int, default=5, help="HDBSCAN min_samples")
    parser.add_argument("--split", type=str, default="test", help="GoEmotions split to use")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample cap")
    args = parser.parse_args()

    run_theme_evaluation(
        output_path=args.output,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        split=args.split,
        max_samples=args.max_samples,
    )
