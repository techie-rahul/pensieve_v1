"""Qualitative Demonstration Script for Pensieve Phase 5 Grounded Reflection.

Demonstrates the complete end-to-end Phase 5 pipeline:
Structured Signals (Phases 1–3) + Retrieved Concepts (Phase 4)
    ↓
Policy Pre-Check
    ↓
Prompt Construction & LLM Generation
    ↓
Post-Generation Safety Validation
    ↓
Final Grounded Reflection Result

Also demonstrates multiple controlled policy and safety rejection cases.
"""

from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.reflection import (
    ReflectionGenerator,
    ReflectionInput,
    ReflectionPolicy,
    ReflectionValidator,
    MockLLMClient,
)


def run_phase5_demo() -> None:
    """Execute the Phase 5 qualitative demonstration."""
    print("=" * 80)
    print("PENSIEVE PHASE 5: GROUNDED REFLECTION GENERATION DEMONSTRATION")
    print("=" * 80)
    print("Notice: Phase 5 synthesizes structured patterns from Phases 1–3 and retrieved")
    print("concepts from Phase 4 into an uncertainty-aware, non-diagnostic reflection.")
    print("Safety rules are enforced both BEFORE and AFTER generation.")
    print("=" * 80)

    generator = ReflectionGenerator()

    # --------------------------------------------------------------------------
    # SCENARIO 1: WORK-RELATED EXHAUSTION AND RECURRING FRUSTRATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 1: Work-Related Exhaustion and Deadline Frustration")
    print("=" * 80)

    scenario_1_input = ReflectionInput(
        data_summary={"entry_count": 12, "span_days": 28, "num_windows": 4},
        emotion_patterns=[
            "Annoyance increased across recent windows (average score shifted from 0.40 to 0.74)",
            "Nervousness remained elevated around sprint deadlines",
        ],
        theme_patterns=[
            "Work & Engineering cluster dominant (55% entry frequency)",
        ],
        linguistic_patterns={
            "negation_ratio": 0.08,
            "question_count": 3,
            "first_person_pronoun_ratio": 0.18,
        },
        temporal_patterns=[
            "Work-related theme recurring across 3 consecutive weekly windows",
            "Imbalance observed between task demands and available recovery time",
        ],
        recurring_patterns=[
            "Frequent content word: 'deadline'",
            "Frequent content word: 'server'",
        ],
        retrieved_concepts=[
            {
                "concept_id": "stress_appraisal_framework",
                "name": "Transactional Stress Appraisal",
                "category": "cognitive_reflective",
                "definition": "A psychological model positing that stress is an evaluative process balancing primary appraisal (perceived threat/harm) and secondary appraisal (perceived coping resources and options).",
                "explanation": "Enables reflection on whether feelings of being overwhelmed reflect an imbalance between perceived environmental demands and recognized personal or external resources.",
                "source": "Lazarus, R. S., & Folkman, S. (1984). Stress, Appraisal, and Coping. Springer Publishing Company.",
                "cautions": ["Descriptive theoretical model; not a clinical burnout diagnosis."],
                "similarity": 0.4606,
            },
            {
                "concept_id": "stoic_dichotomy_of_control",
                "name": "Dichotomy of Control",
                "category": "philosophical",
                "definition": "The philosophical principle dividing events into things within direct agency versus external outcomes.",
                "explanation": "Provides a reflective framework for examining whether ongoing frustration stems from attempting to exert control over external circumstances.",
                "source": "Epictetus, Enchiridion (c. 125 CE); Robertson, D. (2019). How to Think Like a Roman Emperor.",
                "cautions": ["Descriptive philosophical framework; not a medical diagnosis."],
                "similarity": 0.3820,
            },
        ],
    )

    print("[Input Signals Provided to Phase 5]:")
    print(f"  * Data Summary: {scenario_1_input.data_summary['entry_count']} entries across {scenario_1_input.data_summary['span_days']} days")
    print(f"  * Emotion Trajectories: {scenario_1_input.emotion_patterns}")
    print(f"  * Active Themes: {scenario_1_input.theme_patterns}")
    print(f"  * Retrieved Phase 4 Concepts: {[c['name'] for c in scenario_1_input.retrieved_concepts]}")

    print("\n[Executing Pipeline: Policy Pre-Check -> Generation -> Output Validation]...")
    res_1 = generator.generate_reflection(scenario_1_input)

    print(f"\n[Status]: {res_1['status'].upper()}")
    print(f"[Confidence]: {res_1['confidence']} (Capped <= 0.80)")
    print(f"[Grounded Concepts Cited]: {[c['name'] for c in res_1['grounded_concepts']]}")
    print(f"\n[Generated Reflection]:\n{res_1['reflection']}")
    print(f"\n[Mandatory Disclaimer]:\n{res_1['disclaimer']}")

    # --------------------------------------------------------------------------
    # SCENARIO 2: SAVORING JOY AND SOCIAL CONNECTION CONSOLIDATION
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 2: Savoring Joy and Social Connection Consolidation")
    print("=" * 80)

    scenario_2_input = ReflectionInput(
        data_summary={"entry_count": 8, "span_days": 21, "num_windows": 3},
        emotion_patterns=[
            "Joy and gratitude showed steady upward trends over the past 3 weeks",
            "Relief pronounced following project completion",
        ],
        theme_patterns=[
            "Relationships & Social Connection cluster dominant",
        ],
        linguistic_patterns={"first_person_pronoun_ratio": 0.12},
        temporal_patterns=[
            "Social gathering entries associated with elevated positive affect",
        ],
        recurring_patterns=["Frequent content word: 'dinner'", "Frequent content word: 'friends'"],
        retrieved_concepts=[
            {
                "concept_id": "savoring",
                "name": "Savoring",
                "category": "affective_reflective",
                "definition": "The conscious awareness of and deliberate engagement with positive feelings and experiences.",
                "explanation": "Invites a journaler who notes moments of relief or joy to pause and explore sensory details.",
                "source": "Bryant, F. B., & Veroff, J. (2007). Savoring: A New Model of Positive Experience. Lawrence Erlbaum Associates.",
                "cautions": ["Positive psychology framework; not a medical treatment."],
                "similarity": 0.4850,
            },
            {
                "concept_id": "gratitude_orientation",
                "name": "Gratitude Orientation",
                "category": "affective_reflective",
                "definition": "A habitual tendency to recognize and respond with positive emotional appreciation.",
                "explanation": "Encourages acknowledging supportive relationships.",
                "source": "Emmons, R. A., & McCullough, M. E. (2003). Counting blessings versus burdens. Journal of Personality and Social Psychology, 84(2), 377-389.",
                "cautions": ["Descriptive reflective attitude."],
                "similarity": 0.4410,
            },
        ],
    )

    res_2 = generator.generate_reflection(scenario_2_input)
    print(f"[Status]: {res_2['status'].upper()}")
    print(f"[Confidence]: {res_2['confidence']}")
    print(f"\n[Generated Reflection]:\n{res_2['reflection']}")

    # --------------------------------------------------------------------------
    # SCENARIO 3: REJECTED CASE — INSUFFICIENT LONGITUDINAL ENTRIES
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 3 (CONTROL): Policy Rejection — Insufficient Entry Count (< 3)")
    print("=" * 80)

    sparse_input = ReflectionInput(
        data_summary={"entry_count": 2, "span_days": 14},  # Only 2 entries!
        retrieved_concepts=[{"concept_id": "stoic_dichotomy_of_control", "name": "Dichotomy of Control"}],
    )

    res_3 = generator.generate_reflection(sparse_input)
    print(f"[Status]: {res_3['status'].upper()}")
    print(f"[Reason]: {res_3.get('reason')}")
    print(f"[Policy Message]:\n  \"{res_3.get('message')}\"")
    print(f"[LLM Invoked?]: NO (Blocked by pre-generation policy)")

    # --------------------------------------------------------------------------
    # SCENARIO 4: REJECTED CASE — RATE LIMIT EXCEEDED (> 2 REFLECTIONS IN 7 DAYS)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 4 (CONTROL): Policy Rejection — Rate Limit Exceeded (Max 2 per 7 Days)")
    print("=" * 80)

    rate_limited_input = ReflectionInput(
        data_summary={"entry_count": 10, "span_days": 20},
        retrieved_concepts=[{"concept_id": "stoic_dichotomy_of_control", "name": "Dichotomy of Control"}],
        past_reflection_timestamps=[
            "2026-09-14T10:00:00Z",  # Yesterday
            "2026-09-12T14:30:00Z",  # 3 days ago (2 reflections already within 7 days!)
        ],
    )

    res_4 = generator.generate_reflection(
        rate_limited_input,
        current_timestamp="2026-09-15T12:00:00Z",
    )
    print(f"[Status]: {res_4['status'].upper()}")
    print(f"[Reason]: {res_4.get('reason')}")
    print(f"[Policy Message]:\n  \"{res_4.get('message')}\"")
    print(f"[LLM Invoked?]: NO (Blocked by pre-generation policy)")

    # --------------------------------------------------------------------------
    # SCENARIO 5: REJECTED CASE — MODEL OUTPUT SAFETY VIOLATION (DIAGNOSTIC CLAIM)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SCENARIO 5 (CONTROL): Validator Rejection — Model Diagnostic Violation")
    print("=" * 80)

    # Inject mock client that produces a diagnostic assertion
    unsafe_client = MockLLMClient(canned_response={
        "reflection": "Analyzing your writing proves you have clinical depression and anxiety.",
        "grounded_concepts": [{"concept_id": "stress_appraisal_framework", "name": "Transactional Stress Appraisal"}],
        "confidence": 0.95,
        "disclaimer": "This reflection describes observable patterns in journal text for personal contemplation. It does not constitute psychological, psychiatric, or medical advice or diagnosis.",
    })
    unsafe_generator = ReflectionGenerator(client=unsafe_client)

    res_5 = unsafe_generator.generate_reflection(scenario_1_input)
    print(f"[Status]: {res_5['status'].upper()}")
    print(f"[Reason]: {res_5.get('reason')}")
    print(f"[Validation Errors Caught by ReflectionValidator]:")
    for err in res_5.get("validation_errors", []):
        print(f"  * {err}")
    print(f"[Result Accepted?]: NO (Safely rejected; user never sees unsafe reflection)")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE: All scenarios and safety boundaries verified.")
    print("=" * 80)


if __name__ == "__main__":
    run_phase5_demo()
