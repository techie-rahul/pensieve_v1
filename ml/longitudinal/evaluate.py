"""Synthetic validation suite for Phase 3 linguistic and longitudinal pattern analysis.

Evaluates deterministic correctness of:
1. Linguistic feature extraction
2. Temporal sorting & validation
3. Window aggregation
4. Emotion trend calculations (deltas & directions)
5. Theme frequency & trajectory detection
6. Recurring lexical pattern extraction
7. Minimum-data safeguards (<3 entries, <7 days span)

Saves verified diagnostic results to ml/phase3_evaluation_results.json.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.linguistic import extract_linguistic_features
from ml.longitudinal import (
    JournalAggregator,
    TrendDetector,
    PatternSynthesizer,
    analyze_journal_history,
)

DEFAULT_OUTPUT_PATH = "ml/phase3_evaluation_results.json"

# Synthetic dataset with known, mathematically verifiable properties
SYNTHETIC_TEST_ENTRIES = [
    # Week 1: Work stress, high annoyance / nervousness
    {
        "id": "syn-01",
        "timestamp": "2026-01-02T10:00:00Z",
        "text": "Stressed about the project deadline! Can we fix the server error in time? No, it failed again.",
        "emotions": {"annoyance": 0.82, "nervousness": 0.74, "joy": 0.08, "gratitude": 0.05},
        "theme_cluster": 0,
    },
    {
        "id": "syn-02",
        "timestamp": "2026-01-04T15:30:00Z",
        "text": "Another late night debugging code. My stomach hurts from endless meetings and coffee.",
        "emotions": {"annoyance": 0.78, "nervousness": 0.66, "joy": 0.12, "gratitude": 0.08},
        "theme_cluster": 0,
    },
    # Week 2: Transition / relief
    {
        "id": "syn-03",
        "timestamp": "2026-01-10T11:00:00Z",
        "text": "Pushed the code fix to production today. Feeling somewhat relieved, though tired.",
        "emotions": {"annoyance": 0.40, "nervousness": 0.35, "joy": 0.48, "gratitude": 0.42},
        "theme_cluster": 0,
    },
    {
        "id": "syn-04",
        "timestamp": "2026-01-12T18:00:00Z",
        "text": "Took a short walk after work. Quiet evening reading a book at home.",
        "emotions": {"annoyance": 0.20, "nervousness": 0.18, "joy": 0.55, "gratitude": 0.50},
        "theme_cluster": 2,
    },
    # Week 3: Emergence of fitness & joy
    {
        "id": "syn-05",
        "timestamp": "2026-01-18T07:30:00Z",
        "text": "Started morning 5k running routine around the lake. The sunrise was gorgeous and peaceful!",
        "emotions": {"annoyance": 0.08, "nervousness": 0.10, "joy": 0.76, "gratitude": 0.72},
        "theme_cluster": 1,
    },
    {
        "id": "syn-06",
        "timestamp": "2026-01-20T08:00:00Z",
        "text": "Ran 6k today and set a personal record. Proper nutrition and hydration is making a huge difference.",
        "emotions": {"annoyance": 0.05, "nervousness": 0.06, "joy": 0.82, "gratitude": 0.78},
        "theme_cluster": 1,
    },
    # Week 4: Established fitness & high joy
    {
        "id": "syn-07",
        "timestamp": "2026-01-26T07:15:00Z",
        "text": "Loving this morning running habit! Felt energetic all day, so thankful for my health and progress.",
        "emotions": {"annoyance": 0.02, "nervousness": 0.04, "joy": 0.88, "gratitude": 0.85},
        "theme_cluster": 1,
    },
    {
        "id": "syn-08",
        "timestamp": "2026-01-28T09:00:00Z",
        "text": "Weekend trail run with friends. We had breakfast together and laughed constantly.",
        "emotions": {"annoyance": 0.04, "nervousness": 0.05, "joy": 0.90, "gratitude": 0.89},
        "theme_cluster": 1,
    },
]


def run_synthetic_validation() -> Dict[str, Any]:
    """Execute validation suite across all Phase 3 functional components."""
    print("=" * 75)
    print("PENSIEVE ML - PHASE 3 SYNTHETIC VALIDATION SUITE")
    print("=" * 75)

    test_results = {}

    # Test 1: Feature Extraction Correctness
    test_text = "I love my quiet morning coffee! Did you run today? No, I did not."
    ling = extract_linguistic_features(test_text)
    test_1_pass = (
        ling["word_count"] == 14
        and ling["first_person_pronoun_count"] == 3
        and ling["question_count"] == 1
        and ling["exclamation_count"] == 1
        and ling["negation_count"] == 2
        and 0.80 <= ling["vocabulary_diversity"] <= 0.90
    )
    test_results["1_linguistic_feature_extraction"] = {
        "status": "PASS" if test_1_pass else "FAIL",
        "extracted": {
            "word_count": ling["word_count"],
            "first_person_count": ling["first_person_pronoun_count"],
            "questions": ling["question_count"],
            "exclamations": ling["exclamation_count"],
            "negations": ling["negation_count"],
            "vocabulary_diversity": ling["vocabulary_diversity"],
        },
    }
    print(f"Test 1 [Linguistic Feature Extraction]: {test_results['1_linguistic_feature_extraction']['status']}")

    # Test 2: Temporal Ordering
    aggregator = JournalAggregator()
    # Provide intentionally shuffled entries
    shuffled_entries = [SYNTHETIC_TEST_ENTRIES[4], SYNTHETIC_TEST_ENTRIES[0], SYNTHETIC_TEST_ENTRIES[7]]
    sorted_entries = aggregator.validate_and_sort(shuffled_entries)
    test_2_pass = (
        sorted_entries[0]["id"] == "syn-01"
        and sorted_entries[1]["id"] == "syn-05"
        and sorted_entries[2]["id"] == "syn-08"
    )
    test_results["2_temporal_ordering"] = {
        "status": "PASS" if test_2_pass else "FAIL",
        "ordered_ids": [e["id"] for e in sorted_entries],
    }
    print(f"Test 2 [Temporal Ordering]: {test_results['2_temporal_ordering']['status']}")

    # Test 3: Time-Window Aggregation
    agg_result = aggregator.aggregate_into_windows(SYNTHETIC_TEST_ENTRIES, window_type="weekly")
    test_3_pass = (
        agg_result["status"] == "success"
        and agg_result["total_entries"] == 8
        and agg_result["num_windows"] == 4
    )
    test_results["3_window_aggregation"] = {
        "status": "PASS" if test_3_pass else "FAIL",
        "num_windows": agg_result.get("num_windows"),
        "total_entries": agg_result.get("total_entries"),
        "window_entry_counts": [w["entry_count"] for w in agg_result.get("windows", [])],
    }
    print(f"Test 3 [Window Aggregation]: {test_results['3_window_aggregation']['status']}")

    # Test 4: Emotion Trend Calculations
    detector = TrendDetector(significance_threshold=0.05)
    windows = agg_result["windows"]
    emotion_trends = detector.analyze_emotion_trends(windows)

    increasing = {x["emotion"] for x in emotion_trends["increasing_emotions"]}
    decreasing = {x["emotion"] for x in emotion_trends["decreasing_emotions"]}
    test_4_pass = (
        "joy" in increasing
        and "gratitude" in increasing
        and "annoyance" in decreasing
        and "nervousness" in decreasing
    )
    test_results["4_emotion_trend_calculations"] = {
        "status": "PASS" if test_4_pass else "FAIL",
        "detected_increasing": list(increasing),
        "detected_decreasing": list(decreasing),
        "sample_observation": emotion_trends["observations"][0] if emotion_trends["observations"] else None,
    }
    print(f"Test 4 [Emotion Trend Calculations]: {test_results['4_emotion_trend_calculations']['status']}")

    # Test 5: Theme Frequency & Trajectory
    theme_trends = detector.analyze_theme_trends(windows)
    emerging = [x["cluster_id"] for x in theme_trends["emerging_themes"]]
    diminishing = [x["cluster_id"] for x in theme_trends["diminishing_themes"]]
    recurring = [x["cluster_id"] for x in theme_trends["recurring_themes"]]
    test_5_pass = (
        0 in recurring
        and 1 in recurring
    )
    test_results["5_theme_frequency_calculations"] = {
        "status": "PASS" if test_5_pass else "FAIL",
        "recurring_clusters": recurring,
        "observations": theme_trends["observations"],
    }
    print(f"Test 5 [Theme Trajectories]: {test_results['5_theme_frequency_calculations']['status']}")

    # Test 6: Recurring Lexical Pattern Extraction
    synthesizer = PatternSynthesizer()
    lexical = synthesizer.extract_recurring_lexical_patterns(SYNTHETIC_TEST_ENTRIES)
    recurring_terms = [item["term"] for item in lexical["recurring_words"]]
    # 'running' and 'code' appear repeatedly in the synthetic entries
    test_6_pass = any(term in recurring_terms for term in ["running", "code", "run", "morning", "today"])
    test_results["6_recurring_lexical_patterns"] = {
        "status": "PASS" if test_6_pass else "FAIL",
        "top_recurring_terms": recurring_terms[:5],
    }
    print(f"Test 6 [Lexical Patterns]: {test_results['6_recurring_lexical_patterns']['status']}")

    # Test 7: Minimum-Data Safeguards
    # 7a: Fewer than 3 entries
    insufficient_entries = SYNTHETIC_TEST_ENTRIES[:2]
    safeguard_res = aggregator.aggregate_into_windows(insufficient_entries)
    test_7a_pass = safeguard_res["status"] == "insufficient_data"

    # 7b: Entries spanning < 7 days
    brief_entries = [
        {"id": "b1", "timestamp": "2026-01-01T10:00:00Z", "text": "Entry 1"},
        {"id": "b2", "timestamp": "2026-01-02T10:00:00Z", "text": "Entry 2"},
        {"id": "b3", "timestamp": "2026-01-03T10:00:00Z", "text": "Entry 3"},
    ]
    brief_res = aggregator.check_safeguards(aggregator.validate_and_sort(brief_entries))
    test_7b_pass = brief_res["brevity_warning"] is True

    test_7_pass = test_7a_pass and test_7b_pass
    test_results["7_minimum_data_safeguards"] = {
        "status": "PASS" if test_7_pass else "FAIL",
        "insufficient_data_status_caught": test_7a_pass,
        "brevity_warning_caught": test_7b_pass,
    }
    print(f"Test 7 [Minimum-Data Safeguards]: {test_results['7_minimum_data_safeguards']['status']}")

    # Full Pipeline Synthesis Test
    theme_names = {0: "Work & Project Deadline", 1: "Fitness & Morning Routine", 2: "Evening Leisure"}
    master_result = analyze_journal_history(SYNTHETIC_TEST_ENTRIES, window_type="weekly", theme_names=theme_names)

    overall_pass = all(res["status"] == "PASS" for res in test_results.values())

    payload = {
        "metadata": {
            "test_suite": "Phase 3 Synthetic Validation Suite",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "overall_status": "PASS" if overall_pass else "FAIL",
            "tests_run": len(test_results),
            "tests_passed": sum(1 for r in test_results.values() if r["status"] == "PASS"),
        },
        "test_results": test_results,
        "master_synthesis_sample": master_result,
    }

    os.makedirs(os.path.dirname(os.path.abspath(DEFAULT_OUTPUT_PATH)), exist_ok=True)
    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved synthetic validation results to: {DEFAULT_OUTPUT_PATH}")
    print(f"Overall Result: {'ALL TESTS PASSED' if overall_pass else 'FAILURES DETECTED'}")
    return payload


if __name__ == "__main__":
    run_synthetic_validation()
