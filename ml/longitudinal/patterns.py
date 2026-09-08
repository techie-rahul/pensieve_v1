"""Recurring lexical pattern extraction and theme-emotion association analysis.

Identifies repeated content terms and descriptive correlations between topics and emotions.
Explicitly labels all findings as associations, NOT causal relationships.
"""

from collections import Counter
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ml.linguistic.analyzer import LinguisticAnalyzer, get_linguistic_analyzer
from ml.longitudinal.aggregation import JournalAggregator
from ml.longitudinal.trends import TrendDetector
from ml.theme.clustering import STOP_WORDS


class PatternSynthesizer:
    """Synthesizes recurring vocabulary and theme-emotion associations across journal entries."""

    def __init__(self, top_lexical_k: int = 8) -> None:
        """Initialize pattern synthesizer settings."""
        self.top_lexical_k = top_lexical_k

    def extract_recurring_lexical_patterns(self, entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify frequently repeated meaningful words and phrases across separate entries.

        Args:
            entries: List of journal entry dicts containing 'text'.

        Returns:
            Dictionary of top recurring terms with overall count and document frequency.
        """
        if not entries:
            return {"recurring_words": [], "recurring_phrases": []}

        num_entries = len(entries)
        doc_word_counts = Counter()
        total_word_counts = Counter()
        doc_bigram_counts = Counter()

        for entry in entries:
            text = entry.get("text", "").lower()
            tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text)
            content_tokens = [t for t in tokens if t not in STOP_WORDS]

            # Track unique tokens per entry
            for token in set(content_tokens):
                doc_word_counts[token] += 1
            for token in content_tokens:
                total_word_counts[token] += 1

            # Simple bigrams
            if len(content_tokens) >= 2:
                bigrams = [
                    f"{content_tokens[i]} {content_tokens[i+1]}"
                    for i in range(len(content_tokens) - 1)
                ]
                for bg in set(bigrams):
                    doc_bigram_counts[bg] += 1

        # Keep terms appearing in at least 2 separate entries (or >= 2 times if few entries)
        min_doc_freq = 2 if num_entries >= 4 else 1

        recurring_words = []
        for word, doc_freq in doc_word_counts.most_common(self.top_lexical_k * 2):
            if doc_freq >= min_doc_freq and total_word_counts[word] >= 2:
                recurring_words.append({
                    "term": word,
                    "document_frequency": doc_freq,
                    "total_count": total_word_counts[word],
                    "entry_percentage": round((doc_freq / num_entries) * 100, 1),
                })
            if len(recurring_words) >= self.top_lexical_k:
                break

        recurring_phrases = []
        for phrase, doc_freq in doc_bigram_counts.most_common(self.top_lexical_k):
            if doc_freq >= min_doc_freq:
                recurring_phrases.append({
                    "phrase": phrase,
                    "document_frequency": doc_freq,
                })

        return {
            "recurring_words": recurring_words,
            "recurring_phrases": recurring_phrases,
            "interpretation_note": (
                "These content words and phrases appeared repeatedly across multiple entries, "
                "highlighting central recurring subjects in your reflections."
            ),
        }

    def compute_theme_emotion_associations(
        self,
        entries: List[Dict[str, Any]],
        theme_names: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """Calculate statistical associations between discovered themes and emotion probabilities.

        Args:
            entries: List of journal entries with 'theme_cluster' and 'emotions'.
            theme_names: Optional mapping from cluster ID to human-readable descriptor.

        Returns:
            Dictionary mapping themes to their most salient co-occurring emotions.
        """
        # Filter entries that possess both theme_cluster and emotions
        valid_entries = [
            e for e in entries
            if "theme_cluster" in e and "emotions" in e and isinstance(e["emotions"], dict)
        ]

        if not valid_entries:
            return {
                "status": "no_data",
                "message": "Theme-emotion associations require entries with both theme_cluster and emotions.",
                "associations": [],
            }

        # Group emotion scores by cluster
        cluster_emotions: Dict[int, Dict[str, List[float]]] = {}
        cluster_counts = Counter()

        for entry in valid_entries:
            cid = int(entry["theme_cluster"])
            if cid == -1:  # Skip unclustered noise points
                continue
            cluster_counts[cid] += 1
            if cid not in cluster_emotions:
                cluster_emotions[cid] = {}

            for emotion, score in entry["emotions"].items():
                if emotion not in cluster_emotions[cid]:
                    cluster_emotions[cid][emotion] = []
                cluster_emotions[cid][emotion].append(float(score))

        # Baseline average of each emotion across all entries
        global_emotion_means: Dict[str, float] = {}
        all_emotions = {k for e in valid_entries for k in e["emotions"].keys()}
        for em in all_emotions:
            all_scores = [e["emotions"].get(em, 0.0) for e in valid_entries]
            global_emotion_means[em] = float(np.mean(all_scores)) if all_scores else 0.0

        associations = []
        for cid, em_dict in sorted(cluster_emotions.items()):
            theme_label = (
                theme_names.get(cid, f"Theme Cluster {cid}")
                if theme_names
                else f"Theme Cluster {cid}"
            )

            # Compute average score per emotion for this cluster
            cluster_means = {
                emotion: float(np.mean(scores))
                for emotion, scores in em_dict.items()
            }

            # Identify top-3 emotions with highest mean probability in this cluster
            top_emotions = sorted(cluster_means.items(), key=lambda x: x[1], reverse=True)[:3]
            top_emotions_info = [
                {
                    "emotion": em,
                    "mean_probability": round(score, 3),
                    "baseline_mean": round(global_emotion_means.get(em, 0.0), 3),
                    "relative_elevation": round(score - global_emotion_means.get(em, 0.0), 3),
                }
                for em, score in top_emotions if score > 0.05
            ]

            salient_emotions_str = ", ".join(
                [f"{item['emotion']} (avg {item['mean_probability']:.2f})" for item in top_emotions_info]
            )

            associations.append({
                "cluster_id": cid,
                "theme_name": theme_label,
                "entry_count": cluster_counts[cid],
                "associated_emotions": top_emotions_info,
                "summary": (
                    f"Reflections categorized under '{theme_label}' were most frequently associated with "
                    f"{salient_emotions_str}."
                    if salient_emotions_str
                    else f"Reflections under '{theme_label}' did not exhibit strong dominant emotions."
                ),
            })

        return {
            "status": "success",
            "associations": associations,
            "disclaimer": (
                "IMPORTANT: These associations reflect descriptive co-occurrence patterns in your writing. "
                "They do NOT establish causal relationships between topics and feelings."
            ),
        }


def analyze_journal_history(
    entries: List[Dict[str, Any]],
    window_type: str = "weekly",
    custom_days: Optional[int] = None,
    theme_names: Optional[Dict[int, str]] = None,
    enrich_linguistics: bool = True,
) -> Dict[str, Any]:
    """Top-level master function running the full Phase 3 pattern analysis pipeline.

    Args:
        entries: List of journal entry dicts ({id, text, timestamp, [emotions], [theme_cluster]}).
        window_type: 'weekly' or 'monthly' or 'custom'.
        custom_days: Number of days if window_type == 'custom'.
        theme_names: Optional dict mapping theme cluster IDs to human-readable names.
        enrich_linguistics: Whether to automatically compute spaCy linguistic features if absent.

    Returns:
        Structured dictionary containing linguistic, emotion, theme, and longitudinal insights.
    """
    aggregator = JournalAggregator()
    sorted_entries = aggregator.validate_and_sort(entries)
    safeguards = aggregator.check_safeguards(sorted_entries)

    if not safeguards["sufficient_data"]:
        return {
            "status": "insufficient_data",
            "message": safeguards["reason"],
            "entry_count": len(sorted_entries),
            "safeguards": safeguards,
            "linguistic_summary": None,
            "longitudinal_trends": None,
            "lexical_patterns": None,
            "theme_emotion_associations": None,
        }

    # 1. Linguistic feature extraction (enrich entries if not present)
    if enrich_linguistics:
        analyzer = get_linguistic_analyzer()
        for e in sorted_entries:
            if "linguistic_features" not in e:
                e["linguistic_features"] = analyzer.analyze_text(e.get("text", ""))

    # 2. Window Aggregation
    agg_result = aggregator.aggregate_into_windows(
        sorted_entries, window_type=window_type, custom_days=custom_days
    )
    windows = agg_result["windows"]

    # 3. Longitudinal Trends
    trend_detector = TrendDetector()
    emotion_trends = trend_detector.analyze_emotion_trends(windows)
    theme_trends = trend_detector.analyze_theme_trends(windows)
    linguistic_trends = trend_detector.analyze_linguistic_trends(windows)

    # 4. Lexical and Association Patterns
    synthesizer = PatternSynthesizer()
    lexical_patterns = synthesizer.extract_recurring_lexical_patterns(sorted_entries)
    theme_emotion_associations = synthesizer.compute_theme_emotion_associations(
        sorted_entries, theme_names=theme_names
    )

    return {
        "status": "success",
        "entry_count": len(sorted_entries),
        "time_span_days": safeguards["timespan_days"],
        "history_window_type": window_type,
        "num_windows": len(windows),
        "safeguards": safeguards,
        "linguistic_trends": linguistic_trends,
        "emotion_trends": emotion_trends,
        "theme_trends": theme_trends,
        "recurring_lexical_patterns": lexical_patterns,
        "theme_emotion_associations": theme_emotion_associations,
    }
