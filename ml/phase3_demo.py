"""Qualitative Demonstration Script for Pensieve Phase 3.

Linguistic and Longitudinal Pattern Analysis on Realistic Dated Journal Entries.

IMPORTANT NOTICE:
The entries below are simulated qualitative journal reflections.
All analyses are strictly descriptive linguistic and statistical co-occurrence summaries.
They do NOT constitute psychological assessment, personality profiling, or psychiatric diagnoses.
"""

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.longitudinal import (
    JournalAggregator,
    TrendDetector,
    PatternSynthesizer,
    analyze_journal_history,
)

# 12 realistic dated journal entries spanning 4 weeks (January 2026)
QUALITATIVE_JOURNAL_HISTORY = [
    # Week 1: Work project stress & sprint pressure
    {
        "id": "entry-01",
        "timestamp": "2026-01-02T09:30:00Z",
        "text": "Stressed about the quarterly product launch deadline. My chest is tight and our staging server crashed again this morning!",
        "emotions": {"annoyance": 0.72, "nervousness": 0.81, "joy": 0.05, "gratitude": 0.08},
        "theme_cluster": 0,  # Work & Engineering
    },
    {
        "id": "entry-02",
        "timestamp": "2026-01-04T22:15:00Z",
        "text": "Another late night debugging pull requests. Feeling completely exhausted from constant context switching.",
        "emotions": {"annoyance": 0.68, "nervousness": 0.65, "joy": 0.09, "gratitude": 0.11},
        "theme_cluster": 0,
    },
    {
        "id": "entry-03",
        "timestamp": "2026-01-06T18:00:00Z",
        "text": "Spent three hours in back-to-back status meetings. Why does every minor design update require a committee discussion?",
        "emotions": {"annoyance": 0.75, "nervousness": 0.40, "joy": 0.06, "gratitude": 0.07},
        "theme_cluster": 0,
    },
    # Week 2: Transition, relief & social reconnection
    {
        "id": "entry-04",
        "timestamp": "2026-01-10T14:30:00Z",
        "text": "Finally shipped the backend patch! Team celebrated with coffee and pastries. Feeling a palpable wave of relief today.",
        "emotions": {"annoyance": 0.22, "nervousness": 0.18, "joy": 0.64, "gratitude": 0.58},
        "theme_cluster": 0,
    },
    {
        "id": "entry-05",
        "timestamp": "2026-01-12T19:45:00Z",
        "text": "Met Sarah and Marcus for dinner at our favorite noodle shop. We talked for hours and laughed so hard my cheeks hurt.",
        "emotions": {"annoyance": 0.04, "nervousness": 0.08, "joy": 0.82, "gratitude": 0.76},
        "theme_cluster": 1,  # Relationships & Social
    },
    {
        "id": "entry-06",
        "timestamp": "2026-01-14T21:00:00Z",
        "text": "Quiet evening at home catching up on personal reading. Need to make time for hobbies and self-care more consistently.",
        "emotions": {"annoyance": 0.08, "nervousness": 0.12, "joy": 0.51, "gratitude": 0.62},
        "theme_cluster": 2,  # Personal Habits & Goals
    },
    # Week 3: Health, fitness & morning routine emergence
    {
        "id": "entry-07",
        "timestamp": "2026-01-17T07:15:00Z",
        "text": "Woke up at 6 AM and went for a brisk 5k run around the neighborhood lake. The crisp morning air was wonderfully invigorating.",
        "emotions": {"annoyance": 0.03, "nervousness": 0.05, "joy": 0.78, "gratitude": 0.72},
        "theme_cluster": 3,  # Health & Fitness
    },
    {
        "id": "entry-08",
        "timestamp": "2026-01-19T08:00:00Z",
        "text": "Second morning run this week! Paced myself well and prepared a nutritious smoothie afterwards. Feeling focused and present.",
        "emotions": {"annoyance": 0.02, "nervousness": 0.04, "joy": 0.81, "gratitude": 0.79},
        "theme_cluster": 3,
    },
    {
        "id": "entry-09",
        "timestamp": "2026-01-21T18:30:00Z",
        "text": "Productive workday without unnecessary fire drills. Left the office on time and enjoyed a calm walk through the park.",
        "emotions": {"annoyance": 0.10, "nervousness": 0.09, "joy": 0.68, "gratitude": 0.65},
        "theme_cluster": 2,
    },
    # Week 4: Habit consolidation, gratitude & vitality
    {
        "id": "entry-10",
        "timestamp": "2026-01-24T08:30:00Z",
        "text": "Saturday morning 8k trail run with my running group. The sunrise over the ridge was stunning. So grateful for my health!",
        "emotions": {"annoyance": 0.02, "nervousness": 0.03, "joy": 0.91, "gratitude": 0.89},
        "theme_cluster": 3,
    },
    {
        "id": "entry-11",
        "timestamp": "2026-01-26T20:00:00Z",
        "text": "Finished reading my third book of the month. Establishing this evening reading routine has drastically improved my sleep quality.",
        "emotions": {"annoyance": 0.01, "nervousness": 0.04, "joy": 0.74, "gratitude": 0.82},
        "theme_cluster": 2,
    },
    {
        "id": "entry-12",
        "timestamp": "2026-01-28T21:30:00Z",
        "text": "Reflecting on this month: moving from chaotic project panic to steady morning running has completely transformed my daily energy.",
        "emotions": {"annoyance": 0.05, "nervousness": 0.06, "joy": 0.87, "gratitude": 0.92},
        "theme_cluster": 3,
    },
]

THEME_NAMES = {
    0: "Work & Engineering Sprint",
    1: "Relationships & Social Life",
    2: "Personal Habits & Goals",
    3: "Health, Fitness & Running",
}


def run_phase3_demo() -> None:
    """Execute end-to-end Phase 3 demonstration."""
    print("=" * 80)
    print("PENSIEVE ML - PHASE 3 LINGUISTIC & LONGITUDINAL PATTERN DEMO")
    print("=" * 80)
    print("NOTE: Demonstrating observable longitudinal patterns across 12 dated journal entries.")
    print("All statements reflect descriptive text co-occurrences, NOT clinical assessments.\n")

    # 1. Full Master Analysis
    analysis = analyze_journal_history(
        entries=QUALITATIVE_JOURNAL_HISTORY,
        window_type="weekly",
        theme_names=THEME_NAMES,
    )

    print(f"Dataset Overview:")
    print(f"  - Total entries:   {analysis['entry_count']}")
    print(f"  - Timespan:        {analysis['time_span_days']} days (~4 weeks)")
    print(f"  - Windows created: {analysis['num_windows']} weekly windows\n")

    # 2. Emotion Trends
    print("-" * 80)
    print("EMOTION TRAJECTORIES (Continuous Phase 1 Probabilities Across Windows):")
    print("-" * 80)
    em_trends = analysis["emotion_trends"]
    print("Dominant emotions per window:")
    for wd in em_trends["window_dominants"]:
        print(f"  - {wd['label']}: {wd['top_emotions']}")

    print("\nDescriptive Emotion Trend Observations:")
    for obs in em_trends["observations"]:
        print(f"  -> {obs}")

    # 3. Theme Trends
    print("\n" + "-" * 80)
    print("THEME DYNAMICS (Tracking Phase 2 Clusters Over Time):")
    print("-" * 80)
    theme_trends = analysis["theme_trends"]
    for obs in theme_trends["observations"]:
        print(f"  -> {obs}")

    # 4. Linguistic Style Shifts
    print("\n" + "-" * 80)
    print("LINGUISTIC STYLE & STRUCTURE (spaCy Feature Analysis):")
    print("-" * 80)
    ling_trends = analysis["linguistic_trends"]
    for wm in ling_trends["window_metrics"]:
        print(f"  - {wm['label']}: Avg {wm['avg_words_per_entry']} words/entry | "
              f"Vocab Diversity: {wm['avg_vocabulary_diversity']} | "
              f"1st-Person Ratio: {wm['avg_first_person_ratio']:.1%} | "
              f"Questions: {wm['avg_questions_per_entry']}/entry")

    for obs in ling_trends["observations"]:
        print(f"  -> Style observation: {obs}")

    # 5. Recurring Lexical Content
    print("\n" + "-" * 80)
    print("RECURRING CONTENT WORDS & PHRASES (Filtered for Meaningful Vocabulary):")
    print("-" * 80)
    lexical = analysis["recurring_lexical_patterns"]
    print("Most frequent content terms:")
    for item in lexical["recurring_words"][:6]:
        print(f"  - '{item['term']}': appeared in {item['document_frequency']} entries ({item['entry_percentage']}%)")

    print("\nRecurring phrases:")
    for item in lexical["recurring_phrases"][:4]:
        print(f"  - \"{item['phrase']}\" (in {item['document_frequency']} entries)")

    # 6. Theme-Emotion Associations
    print("\n" + "-" * 80)
    print("THEME-EMOTION ASSOCIATIONS (Descriptive Co-occurrences, NOT Causation):")
    print("-" * 80)
    associations = analysis["theme_emotion_associations"]["associations"]
    for assoc in associations:
        print(f"  - [{assoc['theme_name']}] ({assoc['entry_count']} entries):")
        print(f"    {assoc['summary']}")

    print(f"\n  Note: {analysis['theme_emotion_associations']['disclaimer']}")

    # 7. Minimum-Data Safeguard Demonstration
    print("\n" + "=" * 80)
    print("DEMONSTRATION: MINIMUM-DATA SAFEGUARDS (< 3 ENTRIES)")
    print("=" * 80)
    sparse_history = QUALITATIVE_JOURNAL_HISTORY[:2]  # Only 2 entries
    sparse_analysis = analyze_journal_history(sparse_history)
    print(f"Input: {len(sparse_history)} entries provided.")
    print(f"Safeguard Status:  {sparse_analysis['status']}")
    print(f"Safeguard Message: \"{sparse_analysis['message']}\"")
    print("=" * 80)


if __name__ == "__main__":
    run_phase3_demo()
