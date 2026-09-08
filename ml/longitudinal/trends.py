"""Longitudinal trend detection for emotions, themes, and linguistic style.

Calculates temporal shifts across aggregated time windows using uncertainty-aware,
non-diagnostic language.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class TrendDetector:
    """Computes descriptive longitudinal trends across chronological journal windows."""

    def __init__(self, significance_threshold: float = 0.05) -> None:
        """Initialize detector with minimum delta threshold for noting trend direction."""
        self.significance_threshold = significance_threshold

    def analyze_emotion_trends(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track continuous emotion probability trajectories over time windows.

        Args:
            windows: List of window dictionaries containing 'entries'.

        Returns:
            Dictionary containing window-level averages, overall deltas, and descriptive observations.
        """
        if not windows:
            return {"status": "no_windows", "observations": []}

        # Collect all emotion keys present across entries
        all_emotions = set()
        for w in windows:
            for entry in w["entries"]:
                if "emotions" in entry and isinstance(entry["emotions"], dict):
                    all_emotions.update(entry["emotions"].keys())

        if not all_emotions:
            return {
                "status": "no_emotion_data",
                "message": "Entries do not contain Phase 1 emotion probability distributions.",
                "observations": [],
            }

        # Calculate average probability for each emotion per window
        window_emotion_means: List[Dict[str, float]] = []
        for w in windows:
            counts = len(w["entries"])
            means: Dict[str, float] = {}
            for emotion in all_emotions:
                scores = [
                    entry["emotions"].get(emotion, 0.0)
                    for entry in w["entries"]
                    if "emotions" in entry and isinstance(entry["emotions"], dict)
                ]
                means[emotion] = round(float(np.mean(scores)), 4) if scores else 0.0
            window_emotion_means.append(means)

        # Calculate change between first and last window (or average trajectory)
        first_means = window_emotion_means[0]
        last_means = window_emotion_means[-1]

        increasing_emotions = []
        decreasing_emotions = []
        stable_emotions = []
        observations = []

        for emotion in sorted(all_emotions):
            first_val = first_means.get(emotion, 0.0)
            last_val = last_means.get(emotion, 0.0)
            delta = round(last_val - first_val, 4)

            # Only report on emotions that have at least some noticeable presence (> 0.05 in any window)
            max_presence = max(wm.get(emotion, 0.0) for wm in window_emotion_means)
            if max_presence < 0.05:
                continue

            summary_item = {
                "emotion": emotion,
                "first_window_avg": first_val,
                "last_window_avg": last_val,
                "delta": delta,
            }

            if delta >= self.significance_threshold:
                increasing_emotions.append(summary_item)
                observations.append(
                    f"'{emotion}' appears to show an upward trend over time "
                    f"(average score moved from {first_val:.2f} to {last_val:.2f})."
                )
            elif delta <= -self.significance_threshold:
                decreasing_emotions.append(summary_item)
                observations.append(
                    f"'{emotion}' was more pronounced during earlier entries and "
                    f"appeared less frequently later (average score shifted from {first_val:.2f} to {last_val:.2f})."
                )
            else:
                stable_emotions.append(summary_item)

        # Identify dominant emotions in each window
        window_dominants = []
        for idx, wm in enumerate(window_emotion_means):
            sorted_top = sorted(wm.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{e} ({s:.2f})" for e, s in sorted_top if s > 0.05])
            window_dominants.append({
                "window_index": idx,
                "label": windows[idx]["label"],
                "top_emotions": top_str if top_str else "low overall intensity",
            })

        return {
            "status": "success",
            "increasing_emotions": increasing_emotions,
            "decreasing_emotions": decreasing_emotions,
            "stable_emotions": stable_emotions,
            "window_dominants": window_dominants,
            "observations": observations if observations else ["Emotion intensity appeared relatively stable."],
        }

    def analyze_theme_trends(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track thematic cluster occurrences over chronological windows.

        Args:
            windows: List of window dictionaries containing 'entries'.

        Returns:
            Dictionary containing cluster presence, emerging/diminishing clusters, and observations.
        """
        if not windows:
            return {"status": "no_windows", "observations": []}

        # Check if theme cluster annotations exist
        has_themes = any(
            "theme_cluster" in e
            for w in windows for e in w["entries"]
        )
        if not has_themes:
            return {
                "status": "no_theme_data",
                "message": "Entries do not contain Phase 2 theme cluster IDs.",
                "observations": [],
            }

        num_windows = len(windows)
        window_cluster_counts: List[Dict[int, int]] = []
        all_clusters = set()

        for w in windows:
            counts: Dict[int, int] = {}
            for e in w["entries"]:
                cid = e.get("theme_cluster", -1)
                counts[cid] = counts.get(cid, 0) + 1
                all_clusters.add(cid)
            window_cluster_counts.append(counts)

        # Classify cluster trajectories
        recurring = []
        emerging = []
        diminishing = []
        observations = []

        # Remove noise (-1) from primary thematic patterns
        content_clusters = [c for c in all_clusters if c != -1]

        for cid in sorted(content_clusters):
            presence_indices = [
                idx for idx, counts in enumerate(window_cluster_counts)
                if counts.get(cid, 0) > 0
            ]
            total_occurrences = sum(counts.get(cid, 0) for counts in window_cluster_counts)

            info = {
                "cluster_id": cid,
                "total_entries": total_occurrences,
                "present_in_windows": presence_indices,
            }

            if len(presence_indices) > 1:
                recurring.append(info)
                observations.append(
                    f"Theme Cluster {cid} reappeared across multiple periods ({len(presence_indices)}/{num_windows} windows), "
                    "suggesting an ongoing area of reflection."
                )
            elif presence_indices and presence_indices[0] == num_windows - 1 and num_windows > 1:
                emerging.append(info)
                observations.append(
                    f"Theme Cluster {cid} emerged in the most recent time window, representing a newer subject in your journal."
                )
            elif presence_indices and presence_indices[0] == 0 and num_windows > 1:
                diminishing.append(info)
                observations.append(
                    f"Theme Cluster {cid} was discussed in early entries but did not appear in subsequent windows."
                )

        return {
            "status": "success",
            "recurring_themes": recurring,
            "emerging_themes": emerging,
            "diminishing_themes": diminishing,
            "observations": observations if observations else ["Thematic subjects remained varied with few repeating clusters."],
        }

    def analyze_linguistic_trends(self, windows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect stylistic and structural changes in writing habits across time windows.

        Args:
            windows: List of window dictionaries containing 'entries' with 'linguistic_features'.

        Returns:
            Dictionary containing comparative metrics across windows and descriptive commentary.
        """
        if not windows:
            return {"status": "no_windows", "observations": []}

        has_ling = any(
            "linguistic_features" in e
            for w in windows for e in w["entries"]
        )
        if not has_ling:
            return {
                "status": "no_linguistic_data",
                "message": "Entries do not contain spaCy linguistic features.",
                "observations": [],
            }

        window_metrics: List[Dict[str, Any]] = []
        for w in windows:
            feats = [
                e["linguistic_features"]
                for e in w["entries"]
                if "linguistic_features" in e and isinstance(e["linguistic_features"], dict)
            ]
            if not feats:
                continue

            avg_words = round(float(np.mean([f.get("word_count", 0) for f in feats])), 1)
            avg_diversity = round(float(np.mean([f.get("vocabulary_diversity", 0.0) for f in feats])), 3)
            avg_first_person = round(float(np.mean([f.get("first_person_pronoun_ratio", 0.0) for f in feats])), 3)
            avg_questions = round(float(np.mean([f.get("question_count", 0) for f in feats])), 2)
            avg_exclamations = round(float(np.mean([f.get("exclamation_count", 0) for f in feats])), 2)

            window_metrics.append({
                "window_index": w["window_index"],
                "label": w["label"],
                "avg_words_per_entry": avg_words,
                "avg_vocabulary_diversity": avg_diversity,
                "avg_first_person_ratio": avg_first_person,
                "avg_questions_per_entry": avg_questions,
                "avg_exclamations_per_entry": avg_exclamations,
            })

        observations = []
        if len(window_metrics) >= 2:
            first = window_metrics[0]
            last = window_metrics[-1]

            # Length shift
            word_diff = last["avg_words_per_entry"] - first["avg_words_per_entry"]
            if abs(word_diff) >= 15:
                direction = "longer" if word_diff > 0 else "more concise"
                observations.append(
                    f"Journal entries became noticeably {direction} over time "
                    f"(averaging {first['avg_words_per_entry']} words early on vs. {last['avg_words_per_entry']} words recently)."
                )

            # First-person shift
            fp_diff = last["avg_first_person_ratio"] - first["avg_first_person_ratio"]
            if abs(fp_diff) >= 0.03:
                direction = "more focused on direct self-reference" if fp_diff > 0 else "less centered on direct first-person pronouns"
                observations.append(
                    f"Reflections showed a shift toward being {direction} (first-person ratio changed by {fp_diff:+.1%})."
                )

            # Inquisitiveness / questions
            q_diff = last["avg_questions_per_entry"] - first["avg_questions_per_entry"]
            if q_diff >= 0.5:
                observations.append(
                    f"Recent entries feature more questioning and inquiry (averaging {last['avg_questions_per_entry']} questions/entry vs {first['avg_questions_per_entry']})."
                )

        return {
            "status": "success",
            "window_metrics": window_metrics,
            "observations": observations if observations else ["Writing length and stylistic features remained consistent over time."],
        }
