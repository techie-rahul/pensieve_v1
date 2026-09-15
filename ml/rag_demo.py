"""Qualitative Demonstration Script for Pensieve Phase 4 RAG Grounding.

Demonstrates deterministic query construction and semantic concept retrieval over 8
realistic synthetic pattern scenarios (representing Phase 1–3 inputs) and an out-of-domain
negative control scenario.

IMPORTANT ETHICAL AND NON-DIAGNOSTIC NOTICE:
- Phase 4 contains NO LLM and performs NO text or reflection generation.
- Retrieval relevance reflects conceptual similarity to descriptive reflective frameworks.
- It does NOT diagnose psychological conditions, evaluate mental health disorders, or infer
  clinical burnout.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.rag.documents import KnowledgeBase
from ml.rag.retriever import ConceptRetriever, PatternSignal


# 8 realistic synthetic pattern scenarios representing diverse journaling contexts
QUALITATIVE_SCENARIOS: List[Dict[str, Any]] = [
    {
        "title": "Scenario 1: Work-Related Exhaustion and Frustration",
        "description": "User exhibits sustained high annoyance and nervousness over work demands, with recurring engineering theme.",
        "signals": PatternSignal(
            emotions={"annoyance": 0.74, "nervousness": 0.62, "joy": 0.05},
            themes=[{"cluster_id": 0, "name": "Work & Engineering", "frequency": 0.55}],
            linguistic_patterns={"negation_ratio": 0.08, "question_count": 2, "first_person_pronoun_ratio": 0.18},
            longitudinal_patterns=[
                "Work-related theme recurring across 3 consecutive windows",
                "Annoyance increased significantly (+0.25) across recent entries",
                "Perceived imbalance between task demands and available recovery time",
            ],
        ),
    },
    {
        "title": "Scenario 2: Focus on Controllable Agency Amidst External Uncertainty",
        "description": "User writes about frustration regarding organizational reshuffling outside their control.",
        "signals": PatternSignal(
            emotions={"annoyance": 0.68, "nervousness": 0.45},
            themes=[{"cluster_id": 0, "name": "Work & Engineering"}],
            linguistic_patterns={"question_frequency": 0.14},
            longitudinal_patterns=[
                "Repeated focus on executive decisions and external corporate restructuring",
                "Questioning personal agency and what can realistically be influenced",
            ],
        ),
    },
    {
        "title": "Scenario 3: Harsh Self-Criticism and Rumination After a Mistake",
        "description": "User experiences intense self-judgment after a minor error during a team presentation.",
        "signals": PatternSignal(
            emotions={"remorse": 0.70, "sadness": 0.52, "nervousness": 0.48},
            themes=[{"cluster_id": 0, "name": "Work & Engineering"}],
            linguistic_patterns={"negation_ratio": 0.09, "first_person_pronoun_ratio": 0.24},
            longitudinal_patterns=[
                "Looping playback of a past meeting error",
                "Harsh self-critical language and feelings of personal inadequacy",
            ],
        ),
    },
    {
        "title": "Scenario 4: Anticipatory Dread and Spiraling What-If Scenarios",
        "description": "User is paralyzed by anxious anticipation ahead of an upcoming major launch.",
        "signals": PatternSignal(
            emotions={"fear": 0.76, "nervousness": 0.81},
            themes=[{"cluster_id": 0, "name": "Work & Engineering"}],
            linguistic_patterns={"question_count": 3},
            longitudinal_patterns=[
                "Escalating nervousness preceding product launch date",
                "Fixation on worst-case failure outcomes and catastrophic what-if thoughts",
            ],
        ),
    },
    {
        "title": "Scenario 5: Adaptation and Coping with Personal Loss",
        "description": "User navigates grief and emotional waves while balancing daily responsibilities.",
        "signals": PatternSignal(
            emotions={"grief": 0.65, "sadness": 0.72, "relief": 0.18},
            themes=[{"cluster_id": 1, "name": "Relationships & Social Connection"}],
            linguistic_patterns={"first_person_pronoun_ratio": 0.16},
            longitudinal_patterns=[
                "Oscillating between painful memories of bereavement and returning to daily tasks",
                "Navigating sadness alongside moments of routine functioning",
            ],
        ),
    },
    {
        "title": "Scenario 6: Routine Inertia and Behavioral Disengagement",
        "description": "User reports spending weekends passively waiting for motivation before exercising or writing.",
        "signals": PatternSignal(
            emotions={"neutral": 0.68, "sadness": 0.32},
            themes=[{"cluster_id": 2, "name": "Personal Habits & Goals"}],
            linguistic_patterns={"negation_ratio": 0.06},
            longitudinal_patterns=[
                "Diminishing frequency of weekend active routines",
                "Waiting for motivation to arrive before engaging in valued personal habits",
            ],
        ),
    },
    {
        "title": "Scenario 7: Savoring Joy and Social Gratitude Consolidation",
        "description": "User celebrates a milestone with close friends and reflects on positive shared experiences.",
        "signals": PatternSignal(
            emotions={"joy": 0.89, "gratitude": 0.85, "admiration": 0.60},
            themes=[{"cluster_id": 1, "name": "Relationships & Social Connection"}],
            linguistic_patterns={"first_person_pronoun_ratio": 0.12},
            longitudinal_patterns=[
                "Celebrated team milestone and shared dinner with supportive friends",
                "Lingering warmth and appreciation for personal relationships",
            ],
        ),
    },
    {
        "title": "Scenario 8: Out-of-Domain Query (Rejection & Abstention Control)",
        "description": "An off-topic, non-reflective mechanical instruction query tested against similarity threshold.",
        "signals": PatternSignal(
            raw_query="how to adjust carburetor idle mixture screws and replace ignition coils on a four stroke engine",
        ),
    },
]


def run_qualitative_demo() -> None:
    """Execute the qualitative demonstration across all scenarios."""
    print("=" * 80)
    print("PENSIEVE PHASE 4: RAG KNOWLEDGE GROUNDING DEMONSTRATION")
    print("=" * 80)
    print("Notice: Phase 4 operates without an LLM. It performs deterministic query")
    print("construction and semantic vector retrieval. No diagnostic claims are made.")
    print("=" * 80)

    kb = KnowledgeBase.load_development_dataset()
    retriever = ConceptRetriever(kb, default_top_k=3, default_threshold=0.25)

    print(f"\n[Knowledge Base]: {len(kb)} development concepts indexed using FAISS.")
    print(f"[Embedding Model]: sentence-transformers/all-MiniLM-L6-v2 (normalized 384-d)")
    print(f"[Relevance Threshold]: {retriever.default_threshold}\n")

    for i, scenario in enumerate(QUALITATIVE_SCENARIOS, start=1):
        print("=" * 80)
        print(f"SCENARIO {i}: {scenario['title']}")
        print(f"Context: {scenario['description']}")
        print("-" * 80)

        # 1. Inspect structured input signals
        signals = scenario["signals"]
        if signals.raw_query:
            print(f"[Input Mode]: Direct Text Query")
            print(f"  Raw Query: \"{signals.raw_query}\"")
        else:
            print(f"[Structured Phase 1–3 Input Signals]:")
            if signals.emotions:
                top_emo = sorted(signals.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  * Phase 1 Emotions: {top_emo}")
            if signals.themes:
                print(f"  * Phase 2 Themes: {signals.themes}")
            if signals.linguistic_patterns:
                print(f"  * Phase 3 Linguistic: {signals.linguistic_patterns}")
            if signals.longitudinal_patterns:
                print(f"  * Phase 3 Longitudinal: {signals.longitudinal_patterns}")

        # 2. Retrieve concepts
        result = retriever.retrieve(signals, top_k=3, similarity_threshold=0.25)

        # 3. Deterministic query output
        print(f"\n[Deterministic Query Built (No LLM)]:")
        print(f"  Query: \"{result['query']}\"")
        if "query_audit" in result and result["query_audit"]:
            print(f"  Audit Trail: {result['query_audit']}")

        # 4. Display retrieval results or rejection
        print(f"\n[Retrieval Status]: {result['status'].upper()}")
        if result["status"] == "no_relevant_concepts":
            print(f"  Message: {result.get('message', 'No concepts exceeded the relevance threshold.')}")
            print(f"  Threshold Applied: {result['threshold_applied']}")
        else:
            print(f"[Retrieved Top Concepts (Top-{len(result['results'])} above threshold {result['threshold_applied']})]:\n")
            for rank, r in enumerate(result["results"], start=1):
                print(f"  {rank}. {r['name']} ({r['category']})")
                print(f"     Similarity Score : {r['similarity_score']:.4f}")
                print(f"     Definition       : {r['definition']}")
                print(f"     Explanation      : {r['explanation']}")
                print(f"     Source           : {r['source']}")
                print(f"     Safety Cautions  : {r['cautions']}")
                print()

        print(f"[Disclaimer]: {result['disclaimer']}")
        print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE: All 8 scenarios evaluated successfully.")
    print("=" * 80)


if __name__ == "__main__":
    run_qualitative_demo()
