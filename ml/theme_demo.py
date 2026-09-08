"""Qualitative Demonstration Script for Pensieve Theme Discovery.

IMPORTANT NOTICE:
The entries below are simulated qualitative journal reflections representing realistic
user topics (work, health, relationships, finances, etc.). They are NOT benchmark
evaluation data and are NOT from GoEmotions.
Clusters discovered are strictly unsupervised semantic text groupings, NOT psychological
or psychiatric profiles.
"""

import sys
from pathlib import Path
from typing import Dict, List

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.theme.clustering import ThemeClusterer
from ml.theme.embedding import TextEmbedder

QUALITATIVE_JOURNAL_ENTRIES = [
    # Academics & Learning
    {
        "id": "entry-01",
        "topic": "Academics",
        "text": "Spent five hours in the library preparing for the multivariate calculus midterm. The practice problems are finally starting to click.",
    },
    {
        "id": "entry-02",
        "topic": "Academics",
        "text": "Submitted my literature review chapter today. My thesis advisor gave encouraging feedback, though I need to revise the methodology section.",
    },
    # Work & Career
    {
        "id": "entry-03",
        "topic": "Work",
        "text": "Tough sprint deadline at work. Our engineering team pushed the backend patch just in time before the staging freeze tonight.",
    },
    {
        "id": "entry-04",
        "topic": "Work",
        "text": "Long product roadmap meeting with leadership today. We agreed on Q3 priorities, but managing stakeholder expectations will be challenging.",
    },
    # Health & Fitness
    {
        "id": "entry-05",
        "topic": "Health & Fitness",
        "text": "Woke up at 6 AM and ran a 5k around the lake. Felt energized all morning, and my resting heart rate has noticeably improved.",
    },
    {
        "id": "entry-06",
        "topic": "Health & Fitness",
        "text": "Hit a new personal record on deadlifts at the gym. Focusing on form and proper post-workout nutrition is paying off.",
    },
    # Relationships & Social
    {
        "id": "entry-07",
        "topic": "Relationships",
        "text": "Had a long, honest conversation with Alex over dinner. It was uncomfortable at first, but we resolved the misunderstanding between us.",
    },
    {
        "id": "entry-08",
        "topic": "Relationships",
        "text": "Caught up with my college best friends on a group video call. Even after months apart, we laughed until our stomachs hurt.",
    },
    # Family
    {
        "id": "entry-09",
        "topic": "Family",
        "text": "Visited my parents for Sunday lunch. Mom made her special lasagna, and we spent the afternoon looking through old childhood photo albums.",
    },
    # Financial Concerns
    {
        "id": "entry-10",
        "topic": "Financial Concerns",
        "text": "Reviewed my monthly bank statements and categorized expenses. Need to cut back on takeout coffee and cancel unused digital subscriptions.",
    },
    {
        "id": "entry-11",
        "topic": "Financial Concerns",
        "text": "Feeling anxious about the rising rent and inflation this year. Building a 6-month emergency fund needs to be my top financial priority.",
    },
    # Personal Goals & Future Planning
    {
        "id": "entry-12",
        "topic": "Personal Goals",
        "text": "Committed to reading 20 pages every night before sleep. It's helping me reduce screen time and expand my thinking.",
    },
    {
        "id": "entry-13",
        "topic": "Future Planning",
        "text": "Contemplating moving to a quieter city near the coast next year. Made a pros and cons list regarding cost of living and remote work flexibility.",
    },
    # Hobbies & Leisure
    {
        "id": "entry-14",
        "topic": "Hobbies",
        "text": "Spent a quiet rainy Saturday afternoon practicing watercolor landscape painting. The rhythm of the brush strokes was deeply therapeutic.",
    },
]


def run_demo() -> None:
    """Run Sentence-BERT + HDBSCAN theme discovery demo on qualitative entries."""
    print("=" * 80)
    print("PENSIEVE THEME DISCOVERY - QUALITATIVE JOURNAL DEMONSTRATION")
    print("=" * 80)
    print("NOTE: Demonstrating unsupervised semantic clustering on 14 simulated entries.")
    print("Clusters represent textual similarity, NOT psychological profiles.\n")

    texts = [e["text"] for e in QUALITATIVE_JOURNAL_ENTRIES]

    # 1. Embed texts
    print("Generating Sentence-BERT embeddings (all-MiniLM-L6-v2, 384-dim, L2-normalized)...")
    embedder = TextEmbedder()
    embeddings = embedder.embed_texts(texts, show_progress_bar=False)

    # 2. Fit HDBSCAN (min_cluster_size=2 to allow micro-themes in small sample)
    print("Clustering with HDBSCAN (min_cluster_size=2, min_samples=1, metric='euclidean')...\n")
    clusterer = ThemeClusterer(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings, texts=texts)

    # 3. Print per-entry assignments
    print("-" * 80)
    print(f"{'ID':<10} | {'Cluster':<10} | {'Status':<12} | {'Domain Topic':<18} | Text Preview")
    print("-" * 80)
    for entry, label in zip(QUALITATIVE_JOURNAL_ENTRIES, labels):
        cid = int(label)
        status = "OUTLIER" if cid == -1 else f"Cluster {cid}"
        cluster_str = str(cid) if cid >= 0 else "-1"
        preview = entry["text"][:55] + "..." if len(entry["text"]) > 55 else entry["text"]
        print(f"{entry['id']:<10} | {cluster_str:<10} | {status:<12} | {entry['topic']:<18} | \"{preview}\"")

    # 4. Print cluster summary with representative central entries
    summary = clusterer.get_cluster_summary(top_n_examples=2)
    print("\n" + "=" * 80)
    print(f"DISCOVERED CLUSTERS SUMMARY ({summary['num_clusters']} clusters, {summary['noise_samples']} outliers):")
    print("=" * 80)

    for cid, cinfo in summary["clusters"].items():
        print(f"\n[Cluster {cid}] (Size: {cinfo['size']} entries | Keywords: {cinfo['simple_theme_descriptor']})")
        print("  Representative / Central Examples:")
        for idx, ex in enumerate(cinfo["representative_examples"], start=1):
            print(f"    {idx}. \"{ex}\"")

    if summary["noise_samples"] > 0:
        noise_indices = [i for i, l in enumerate(labels) if l == -1]
        print(f"\n[Outliers / Unclustered Points] ({len(noise_indices)} entries):")
        for i in noise_indices:
            print(f"  - [{QUALITATIVE_JOURNAL_ENTRIES[i]['id']}] \"{QUALITATIVE_JOURNAL_ENTRIES[i]['text']}\"")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_demo()
