"""Qualitative Demonstration Script for Pensieve Emotion Detection.

IMPORTANT NOTICE:
The examples below are strictly qualitative demonstration entries representing
realistic user journaling text. They are NOT evaluation benchmarks and are not
part of the GoEmotions test dataset.
"""

import json
import sys
from pathlib import Path

# Ensure workspace root is in sys.path when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.emotion.inference import EmotionClassifier

# 8 diverse, realistic journal entries covering various affective states
QUALITATIVE_JOURNAL_SAMPLES = [
    {
        "id": "demo-01",
        "category": "Gratitude & Serenity",
        "entry": (
            "Woke up early, drank a warm cup of coffee while watching the sunrise. "
            "I feel so deeply at peace and thankful for this quiet morning."
        ),
    },
    {
        "id": "demo-02",
        "category": "Anxiety & Anticipation",
        "entry": (
            "I can't stop thinking about tomorrow's presentation. My hands are shaking, "
            "my stomach hurts, and I feel like I'm going to mess up completely."
        ),
    },
    {
        "id": "demo-03",
        "category": "Workplace Frustration",
        "entry": (
            "Another meeting that could have been an email. Spent two hours listening "
            "to circular arguments and nothing got decided. So frustrated and exhausted."
        ),
    },
    {
        "id": "demo-04",
        "category": "Achievement & Celebration",
        "entry": (
            "After three months of grueling debugging and late nights, we finally shipped "
            "the milestone! The whole team is thrilled, and I couldn't be prouder of our work."
        ),
    },
    {
        "id": "demo-05",
        "category": "Bittersweet Grief & Love",
        "entry": (
            "Walking through the park today, I suddenly remembered my grandmother's laugh. "
            "It made me smile warmly, but my chest still aches with missing her so much."
        ),
    },
    {
        "id": "demo-06",
        "category": "Introspective Epiphany",
        "entry": (
            "I realized today that I don't actually want that corporate promotion. It took talking "
            "to Sarah to see that what I really value is creative freedom, not managing meetings."
        ),
    },
    {
        "id": "demo-07",
        "category": "Neutral Daily Log",
        "entry": (
            "Woke up at 7:30 AM, picked up groceries, did laundry, and prepared lunch for the week. "
            "Read 25 pages of my book before bed."
        ),
    },
    {
        "id": "demo-08",
        "category": "Social Uncertainty & Rumination",
        "entry": (
            "I don't know what I did wrong, but he hasn't answered my messages since yesterday. "
            "Did I say something insensitive? I keep re-reading what I wrote and agonizing."
        ),
    },
]


def run_demo() -> None:
    """Run emotion inference on qualitative journal entries and display formatted results."""
    print("=" * 80)
    print("PENSIEVE EMOTION DETECTION - QUALITATIVE JOURNAL DEMONSTRATION")
    print("=" * 80)
    print("NOTE: The entries below are simulated journal entries for qualitative assessment,")
    print("NOT formal benchmark evaluation data.\n")

    classifier = EmotionClassifier(default_threshold=0.30)

    for sample in QUALITATIVE_JOURNAL_SAMPLES:
        result = classifier.predict(sample["entry"])

        print(f"[{sample['id']}] Category: {sample['category']}")
        print(f"Text: \"{sample['entry']}\"")
        print("Detected Emotions (threshold >= 0.30):")

        for pred in result["predictions"]:
            bar = "#" * int(pred["score"] * 20)
            print(f"  • {pred['emotion']:<15} (score: {pred['score']:.4f}, sentiment: {pred['sentiment']:<9}) {bar}")

        top_3 = sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True)[:3]
        top_3_str = ", ".join([f"{k}: {v:.3f}" for k, v in top_3])
        print(f"Top-3 raw scores: {top_3_str}")
        print("-" * 80)


if __name__ == "__main__":
    run_demo()

