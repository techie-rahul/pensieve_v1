"""Benchmark evaluation script for Pensieve Phase 4 RAG Grounding.

Evaluates retrieval quality against a manually curated benchmark of realistic journaling
pattern queries and out-of-domain rejection controls.

IMPORTANT ETHICAL AND SCIENTIFIC NOTICE:
This evaluation measures algorithmic retrieval ranking and similarity threshold gating
against a small, manually curated development benchmark. It does NOT constitute a clinical
validation study, psychiatric assessment, or psychological efficacy trial.

Rejection of out-of-domain queries demonstrates that the retriever reliably abstains
from forcing irrelevant concepts onto unrelated text.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.rag.documents import KnowledgeBase
from ml.rag.retriever import ConceptRetriever, PatternSignal


@dataclass
class BenchmarkItem:
    """A single evaluation query scenario with expected relevant concept IDs."""

    id: str
    scenario_description: str
    query: str
    expected_concept_ids: List[str]
    is_out_of_domain: bool = False


# Curated benchmark representing realistic journaling reflection scenarios.
# Explicitly uses non-diagnostic targets (e.g. stress appraisal and locus of control rather than burnout diagnosis).
RETRIEVAL_BENCHMARK: List[BenchmarkItem] = [
    BenchmarkItem(
        id="bench-01",
        scenario_description="Work-related exhaustion and deadline frustration with perceived insurmountable demands",
        query="work-related exhaustion and deadline frustration with perceived insurmountable demands and depleted resources",
        expected_concept_ids=["stress_appraisal_framework", "stoic_dichotomy_of_control", "decatastrophizing"],
    ),
    BenchmarkItem(
        id="bench-02",
        scenario_description="Differentiating between what is and is not within personal control",
        query="contemplating what is inside versus outside my direct control amidst organizational uncertainty",
        expected_concept_ids=["stoic_dichotomy_of_control", "locus_of_control"],
    ),
    BenchmarkItem(
        id="bench-03",
        scenario_description="Harsh self-criticism and isolation after a perceived mistake",
        query="berating myself harshly after making a minor mistake at work and feeling isolated in failure",
        expected_concept_ids=["self_compassion", "cognitive_reframing"],
    ),
    BenchmarkItem(
        id="bench-04",
        scenario_description="Catastrophic anticipation and spiraling what-if scenarios",
        query="anticipatory dread and spiraling catastrophic thoughts about worst-case scenario outcomes",
        expected_concept_ids=["decatastrophizing", "cognitive_reframing"],
    ),
    BenchmarkItem(
        id="bench-05",
        scenario_description="Repetitive looping thoughts and unconstructive brooding over past regrets",
        query="looping thoughts and repetitive unconstructive brooding over past conversations without finding solutions",
        expected_concept_ids=["rumination_interruption", "cognitive_reframing"],
    ),
    BenchmarkItem(
        id="bench-06",
        scenario_description="Vague emotional distress and difficulty articulating feelings",
        query="feeling vague distress and having difficulty distinguishing whether I am feeling irritation fatigue or sadness",
        expected_concept_ids=["emotional_granularity"],
    ),
    BenchmarkItem(
        id="bench-07",
        scenario_description="Inertia, loss of routine, and waiting for motivation before acting",
        query="struggling with weekend inertia passivity and waiting for motivation to return before engaging in activities",
        expected_concept_ids=["behavioral_activation"],
    ),
    BenchmarkItem(
        id="bench-08",
        scenario_description="Fear of incompetence when learning challenging new technical skills",
        query="frustration when learning difficult skills and fear of looking incompetent or revealing fixed limitations",
        expected_concept_ids=["growth_mindset", "cognitive_reframing"],
    ),
    BenchmarkItem(
        id="bench-09",
        scenario_description="Pausing to prolong and deeply appreciate a peaceful morning routine",
        query="slowing down to savor a peaceful morning walk and feeling genuine appreciation for supportive colleagues",
        expected_concept_ids=["savoring", "gratitude_orientation"],
    ),
    BenchmarkItem(
        id="bench-10",
        scenario_description="Navigating waves of grief and loss while balancing daily responsibilities",
        query="navigating emotional waves of grief and sadness while gradually re-engaging with daily work routines",
        expected_concept_ids=["dual_process_coping", "self_compassion"],
    ),
    # Out-of-Domain Negative Controls (testing threshold rejection & no_relevant_concepts)
    BenchmarkItem(
        id="bench-ood-01",
        scenario_description="Mechanical vehicle repair instructions (out of domain)",
        query="how to adjust carburetor idle mixture screws on a motorcycle engine",
        expected_concept_ids=[],
        is_out_of_domain=True,
    ),
    BenchmarkItem(
        id="bench-ood-02",
        scenario_description="Theoretical high-energy particle physics (out of domain)",
        query="quantum chromodynamics gluons quarks particle collision scattering matrix",
        expected_concept_ids=[],
        is_out_of_domain=True,
    ),
]


def evaluate_retrieval(
    retriever: ConceptRetriever,
    benchmark: Optional[List[BenchmarkItem]] = None,
    k_values: List[int] = [1, 3, 5],
    similarity_threshold: float = 0.25,
) -> Dict[str, Any]:
    """Execute evaluation over the benchmark and compute actual retrieval metrics.

    Args:
        retriever: Configured ConceptRetriever instance.
        benchmark: List of BenchmarkItem objects (defaults to RETRIEVAL_BENCHMARK).
        k_values: List of K values for Top-K metrics.
        similarity_threshold: Similarity cutoff for concept relevance.

    Returns:
        Dictionary of computed benchmark metrics, per-query breakdowns, and execution metadata.
    """
    if benchmark is None:
        benchmark = RETRIEVAL_BENCHMARK

    in_domain_items = [b for b in benchmark if not b.is_out_of_domain]
    out_of_domain_items = [b for b in benchmark if b.is_out_of_domain]

    per_query_results: List[Dict[str, Any]] = []

    # Aggregators for in-domain queries
    recall_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
    precision_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
    hit_at_k: Dict[int, List[float]] = {k: [] for k in k_values}
    reciprocal_ranks: List[float] = []

    # 1. Evaluate In-Domain Queries
    for item in in_domain_items:
        # Retrieve with max K
        max_k = max(k_values)
        res = retriever.retrieve(
            item.query,
            top_k=max_k,
            similarity_threshold=similarity_threshold,
        )

        retrieved_ids = [r["concept_id"] for r in res["results"]]
        expected_set = set(item.expected_concept_ids)

        # Reciprocal Rank calculation (rank of first relevant item, 1-indexed)
        rr = 0.0
        for rank, cid in enumerate(retrieved_ids, start=1):
            if cid in expected_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        query_metrics: Dict[str, Any] = {
            "query_id": item.id,
            "scenario": item.scenario_description,
            "query": item.query,
            "expected_concept_ids": item.expected_concept_ids,
            "retrieved_concept_ids": retrieved_ids,
            "retrieved_scores": [r["similarity_score"] for r in res["results"]],
            "reciprocal_rank": round(rr, 4),
        }

        # K-specific metrics
        for k in k_values:
            top_k_ids = retrieved_ids[:k]
            relevant_retrieved = set(top_k_ids).intersection(expected_set)

            # Hit@K: 1 if any relevant item in top K, else 0
            hit = 1.0 if len(relevant_retrieved) > 0 else 0.0
            hit_at_k[k].append(hit)

            # Recall@K: fraction of expected relevant items retrieved in top K
            rec = len(relevant_retrieved) / len(expected_set) if expected_set else 0.0
            recall_at_k[k].append(rec)

            # Precision@K: fraction of top K retrieved items that are relevant
            prec = len(relevant_retrieved) / k if k > 0 else 0.0
            precision_at_k[k].append(prec)

            query_metrics[f"recall@{k}"] = round(rec, 4)
            query_metrics[f"precision@{k}"] = round(prec, 4)
            query_metrics[f"hit@{k}"] = hit

        per_query_results.append(query_metrics)

    # 2. Evaluate Out-of-Domain Negative Controls
    ood_results: List[Dict[str, Any]] = []
    rejections = 0

    for ood_item in out_of_domain_items:
        res = retriever.retrieve(
            ood_item.query,
            top_k=3,
            similarity_threshold=similarity_threshold,
        )
        is_rejected = (res["status"] == "no_relevant_concepts") or (len(res["results"]) == 0)
        if is_rejected:
            rejections += 1

        ood_results.append({
            "query_id": ood_item.id,
            "scenario": ood_item.scenario_description,
            "query": ood_item.query,
            "status": res["status"],
            "retrieved_count": len(res["results"]),
            "correctly_rejected": is_rejected,
        })

    ood_rejection_rate = rejections / len(out_of_domain_items) if out_of_domain_items else 1.0

    # Summary Metrics
    summary_metrics = {
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "ood_rejection_rate": round(ood_rejection_rate, 4),
        "total_in_domain_queries": len(in_domain_items),
        "total_ood_queries": len(out_of_domain_items),
        "similarity_threshold_tested": similarity_threshold,
    }

    for k in k_values:
        summary_metrics[f"mean_hit@{k}"] = round(sum(hit_at_k[k]) / len(hit_at_k[k]), 4)
        summary_metrics[f"mean_recall@{k}"] = round(sum(recall_at_k[k]) / len(recall_at_k[k]), 4)
        summary_metrics[f"mean_precision@{k}"] = round(sum(precision_at_k[k]) / len(precision_at_k[k]), 4)

    return {
        "benchmark_metadata": {
            "name": "Pensieve Phase 4 Development Retrieval Benchmark",
            "knowledge_base_size": len(retriever.knowledge_base),
            "knowledge_base_type": "DEVELOPMENT / TEST dataset (20 concepts)",
            "embedding_model": retriever.embedder.embedder.model_name,
            "vector_index_type": "FAISS IndexFlatIP (exact cosine similarity)" if retriever.index.is_faiss_active() else "NumPy Cosine Similarity",
            "scientific_disclaimer": (
                "Algorithmic retrieval evaluation on a manually curated development set. "
                "Not a clinical trial, psychiatric validation, or medical effectiveness study."
            ),
        },
        "summary_metrics": summary_metrics,
        "in_domain_results": per_query_results,
        "out_of_domain_results": ood_results,
    }


def run_and_save_benchmark(output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run full benchmark and persist results to JSON."""
    if output_path is None:
        output_path = PROJECT_ROOT / "ml" / "rag_evaluation_results.json"

    print("=" * 70)
    print("RUNNING PENSIEVE PHASE 4 RETRIEVAL BENCHMARK")
    print("=" * 70)

    kb = KnowledgeBase.load_development_dataset()
    retriever = ConceptRetriever(kb, default_threshold=0.25)

    print(f"Knowledge Base Size: {len(kb)} development concepts")
    print(f"Index Backend: {'FAISS' if retriever.index.is_faiss_active() else 'NumPy'}")
    print(f"Embedding Model: {retriever.embedder.embedder.model_name}")
    print(f"Similarity Threshold: {retriever.default_threshold}")
    print("-" * 70)

    results = evaluate_retrieval(retriever, similarity_threshold=0.25)
    summary = results["summary_metrics"]

    print("\nRETRIEVAL BENCHMARK SUMMARY RESULTS:")
    print(f"  Mean Reciprocal Rank (MRR) : {summary['mrr']:.4f}")
    print(f"  Hit@1                      : {summary['mean_hit@1']:.4f}")
    print(f"  Hit@3                      : {summary['mean_hit@3']:.4f}")
    print(f"  Hit@5                      : {summary['mean_hit@5']:.4f}")
    print(f"  Recall@1                   : {summary['mean_recall@1']:.4f}")
    print(f"  Recall@3                   : {summary['mean_recall@3']:.4f}")
    print(f"  Recall@5                   : {summary['mean_recall@5']:.4f}")
    print(f"  Precision@1                : {summary['mean_precision@1']:.4f}")
    print(f"  Precision@3                : {summary['mean_precision@3']:.4f}")
    print(f"  Precision@5                : {summary['mean_precision@5']:.4f}")
    print(f"  Out-of-Domain Rejection    : {summary['ood_rejection_rate'] * 100:.1f}%")
    print("-" * 70)

    # Save to JSON file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark results successfully saved to: {output_path}")
    print("=" * 70)
    return results


if __name__ == "__main__":
    run_and_save_benchmark()
