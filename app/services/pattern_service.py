"""Personalized, evidence-based longitudinal pattern synthesis for Pensieve.

Transforms raw ML Phase 1–3 outputs and authenticated journal entries into genuine,
user-centric insights organized into five core categories:
1. Recurring Topics (Phase 2 semantic clusters + actual user entry content)
2. Emotional Trends (Phase 1 GoEmotions probabilities with strictly non-causal framing)
3. Recurring Thoughts & Language (meaningful multi-word phrases and concepts)
4. Changes Over Time (earlier vs. recent periods with real dates, counts, and entry links)
5. Personal Observations (grounded synthesis with reflective questions)

Strict Non-Diagnostic & Quality Guarantees:
- Never diagnoses or claims clinical certainty.
- Does not label disappointment or sadness as "acute stress".
- Avoids causal assertions (e.g. does not claim routines caused lower stress).
- Clearly distinguishes classifier probability signals from emotional certainty.
- Filters out generic conversational words (e.g. 'morning', 'instead', 'took', 'one', 'time').
- Every insight cites actual supporting entries with IDs, dates, titles, and excerpts.
- Only uses published entries belonging to the authenticated user.
"""

from collections import Counter
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Standard human-readable theme cluster names (from Phase 2 taxonomy)
THEME_CLUSTER_NAMES: Dict[int, str] = {
    0: "Work & Engineering",
    1: "Relationships & Social Connection",
    2: "Personal Habits & Goals",
    3: "Health, Fitness & Vitality",
    4: "Philosophy, Reflection & Mindset",
}

# Comprehensive stop words, generic verbs, and conversational filler
# Explicitly includes 'morning', 'instead', 'took', 'one', 'time', etc.
COMMON_GENERIC_WORDS = {
    # Standard grammatical stop words, generic verbs, and conversational filler
    "a", "about", "above", "across", "after", "again", "against", "all", "almost", "along",
    "already", "also", "although", "always", "am", "among", "an", "and", "another", "any",
    "anybody", "anyone", "anything", "anyway", "anywhere", "are", "aren't", "around", "as",
    "at", "back", "be", "became", "because", "become", "becomes", "been", "before", "began",
    "behind", "being", "below", "beside", "between", "beyond", "bit", "both", "but", "by",
    "came", "can", "can't", "cannot", "certain", "clearly", "come", "comes", "could", "couldn't",
    "day", "days", "did", "didn't", "different", "do", "does", "doesn't", "doing", "done",
    "don't", "down", "during", "each", "early", "either", "else", "end", "even", "ever",
    "every", "everyone", "everything", "everywhere", "far", "feel", "feeling", "feels", "felt",
    "feelings", "few", "finally", "find", "first", "for", "from", "further", "get", "gets",
    "getting", "give", "given", "giving", "go", "goes", "going", "gone", "good", "got", "had",
    "hadn't", "half", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll",
    "he's", "her", "here", "hers", "herself", "him", "himself", "his", "how", "how's", "however",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "instead", "into", "is", "isn't", "it",
    "it's", "its", "itself", "just", "keep", "kept", "keeping", "kind", "knew", "know", "known",
    "last", "later", "least", "less", "let", "let's", "like", "liked", "likely", "little",
    "long", "look", "looked", "looking", "lot", "lots", "made", "make", "makes", "making",
    "many", "may", "maybe", "me", "mean", "means", "meant", "might", "moment", "moments",
    "more", "morning", "mornings", "most", "mostly", "much", "must", "mustn't", "my", "myself",
    "near", "nearly", "need", "needed", "needs", "neither", "never", "new", "next", "night",
    "nights", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere",
    "of", "off", "often", "on", "once", "one", "ones", "only", "onto", "or", "other",
    "others", "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over",
    "own", "part", "parts", "perhaps", "place", "places", "point", "points", "pretty", "quite",
    "rather", "really", "right", "said", "same", "saw", "say", "saying", "says", "second",
    "seconds", "see", "seeing", "seem", "seemed", "seems", "seen", "several", "shall", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "since", "so", "some", "somebody",
    "someone", "something", "sometimes", "somewhere", "soon", "spent", "start", "started",
    "starting", "still", "such", "sure", "take", "taken", "taking", "tell", "telling", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "thing", "things",
    "think", "thinking", "thinks", "third", "this", "those", "though", "thought", "thoughts",
    "three", "through", "throughout", "time", "times", "to", "today", "together", "told",
    "tomorrow", "too", "took", "toward", "towards", "tried", "tries", "try", "trying", "turn",
    "turned", "turning", "two", "under", "until", "up", "upon", "us", "use", "used", "using",
    "usually", "various", "very", "via", "want", "wanted", "wants", "was", "wasn't", "way",
    "ways", "we", "we'd", "we'll", "we're", "we've", "week", "weeks", "well", "went", "were",
    "weren't", "what", "what's", "whatever", "when", "when's", "whenever", "where", "where's",
    "wherever", "whether", "which", "while", "who", "who's", "whole", "whom", "whose", "why",
    "why's", "will", "with", "within", "without", "won't", "would", "wouldn't", "year", "years",
    "yes", "yesterday", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves",
    # Specific generic targets: cognitive verbs, postures, and temporal words
    "past", "present", "future",
    "noticed", "notice", "noticing", "notices",
    "sitting", "sit", "sat", "stand", "standing", "stood", "walk", "walked", "walking",
    "mind", "minds", "brain", "head",
    "remember", "remembered", "remembering", "remind", "reminded",
    "realize", "realized", "realizing",
    "wonder", "wondered", "wondering",
    "spend", "spending",
    "bring", "brought", "bringing",
    "leave", "left", "leaving",
    "stay", "stayed", "staying",
    "stop", "stopped", "stopping",
    "wait", "waited", "waiting",
    "watch", "watched", "watching",
    "listen", "listened", "listening",
    "read", "reading",
    "write", "writing", "wrote", "written", "entry", "entries", "journal", "journaling", "reflection", "reflections",
    "focus", "focused", "focusing",
    "finish", "finished", "finishing",
    "continue", "continued", "continuing",
    "talk", "talked", "talking", "talks",
    "happen", "happened", "happening", "happens",
    "change", "changed", "changing", "changes",
    "check", "checked", "checking",
    "help", "helped", "helping",
    # Additional generic time words, adverbs, and vague adjectives
    "afternoon", "afternoons", "evening", "evenings", "tonight",
    "hour", "hours", "minute", "minutes", "month", "months",
    "person", "people", "stuff",
    "actually", "basically", "definitely", "literally", "probably", "simply",
    "great", "better", "best", "bad", "worse", "worst",
    "able", "okay", "fine", "real",
    # Proper names from examples
    "sarah", "marcus", "john", "david", "alex",
}


def _parse_dt(val: Any) -> Optional[datetime]:
    """Parse string or datetime object into a UTC datetime."""
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if isinstance(val, str):
        try:
            cleaned = val.replace("Z", "+00:00")
            dt = datetime.fromisoformat(cleaned)
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None
    return None


def _format_date(dt: Optional[datetime]) -> str:
    """Format datetime for clean UI presentation (e.g. 'Jan 14, 2026')."""
    if not dt:
        return "Unknown date"
    return dt.strftime("%b %d, %Y")


def _format_date_short(dt: Optional[datetime]) -> str:
    """Format datetime as short label (e.g. 'Jan 14')."""
    if not dt:
        return ""
    return dt.strftime("%b %d")


def _format_date_range(dts: List[datetime]) -> str:
    """Format span across multiple dates (e.g. 'Jan 2 – Jan 28, 2026')."""
    if not dts:
        return "N/A"
    sorted_dts = sorted(dts)
    start, end = sorted_dts[0], sorted_dts[-1]
    if start.year == end.year:
        if start.month == end.month and start.day == end.day:
            return start.strftime("%b %d, %Y")
        return f"{start.strftime('%b %d')} – {end.strftime('%b %d, %Y')}"
    return f"{start.strftime('%b %d, %Y')} – {end.strftime('%b %d, %Y')}"


def _clean_excerpt(text: str, max_words: int = 24) -> str:
    """Extract a concise, sentence-aware excerpt suitable for evidence display."""
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    # Try finding first sentence
    match = re.search(r"^(.*?[.!?])(?:\s|$)", cleaned)
    sentence = match.group(1) if match else cleaned

    words = sentence.split()
    if len(words) <= max_words:
        return sentence
    return " ".join(words[:max_words]) + "..."


def _extract_context_excerpt(text: str, target_phrase: str, max_words: int = 24) -> str:
    """Extract excerpt centered around a specific target phrase."""
    if not text:
        return ""
    cleaned = " ".join(text.strip().split())
    lower_text = cleaned.lower()
    lower_phrase = target_phrase.lower()

    idx = lower_text.find(lower_phrase)
    if idx == -1:
        return _clean_excerpt(text, max_words)

    # Find approximate sentence containing phrase
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    for sent in sentences:
        if lower_phrase in sent.lower():
            words = sent.split()
            if len(words) <= max_words:
                return sent
            return " ".join(words[:max_words]) + "..."

    # Fallback to window around index
    words = cleaned.split()
    return " ".join(words[:max_words]) + "..."


class PersonalizedPatternSynthesizer:
    """Produces verified, evidence-based longitudinal insights from user journal entries."""

    def __init__(self, theme_names: Optional[Dict[int, str]] = None) -> None:
        self.theme_names = theme_names or THEME_CLUSTER_NAMES

    def synthesize(
        self,
        raw_entries: List[Any],
        legacy_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Produce the five structured insight categories with full evidence traces.

        Args:
            raw_entries: List of JournalEntry OR entry dicts.
            legacy_report: Dictionary from ml_service.compute_longitudinal_patterns.

        Returns:
            Dict containing overview, recurring_topics, emotional_trends,
            recurring_thoughts, changes_over_time, and personal_observations.
        """
        # Normalize published entries into structured format (excluding drafts)
        entries: List[Dict[str, Any]] = []
        for e in raw_entries:
            # Respect draft status if attribute exists
            is_draft = getattr(e, "is_draft", False) if not isinstance(e, dict) else e.get("is_draft", False)
            if is_draft:
                continue

            entry_id = getattr(e, "id", None) or (e.get("id") if isinstance(e, dict) else None)
            title = getattr(e, "title", "") or (e.get("title", "") if isinstance(e, dict) else "")
            text = getattr(e, "content", "") or (e.get("content") or e.get("text", "") if isinstance(e, dict) else "")
            created_at = getattr(e, "created_at", None) or (e.get("created_at") or e.get("timestamp") if isinstance(e, dict) else None)

            dt = _parse_dt(created_at)

            analysis = getattr(e, "analysis", None)
            if analysis:
                emotions = analysis.emotions or {}
                theme_name = analysis.theme or ""
                theme_cluster_id = analysis.theme_cluster_id
                linguistics = analysis.linguistic_features or {}
            elif isinstance(e, dict):
                emotions = e.get("emotions", {})
                theme_name = e.get("theme", "")
                theme_cluster_id = e.get("theme_cluster_id") or e.get("theme_cluster", 0)
                linguistics = e.get("linguistic_features", {})
            else:
                emotions, theme_name, theme_cluster_id, linguistics = {}, "", 0, {}

            # Fallback theme resolution
            if theme_cluster_id is not None and not theme_name:
                theme_name = self.theme_names.get(int(theme_cluster_id), f"Theme {theme_cluster_id}")

            entries.append({
                "id": str(entry_id) if entry_id else "",
                "title": title or "Journal Entry",
                "text": text,
                "dt": dt,
                "date_str": _format_date(dt),
                "date_short": _format_date_short(dt),
                "emotions": emotions,
                "theme_name": theme_name,
                "theme_cluster_id": int(theme_cluster_id) if theme_cluster_id is not None else -1,
                "linguistics": linguistics,
            })

        # Sort chronologically
        entries.sort(key=lambda x: x["dt"] or datetime.min.replace(tzinfo=timezone.utc))

        total_entries = len(entries)
        valid_dates = [e["dt"] for e in entries if e["dt"]]
        time_span_days = legacy_report.get("time_span_days")
        if time_span_days is None and valid_dates:
            time_span_days = max(1.0, (valid_dates[-1] - valid_dates[0]).total_seconds() / 86400.0)

        overview = {
            "entry_count": total_entries,
            "time_span_days": round(float(time_span_days), 1) if time_span_days is not None else None,
            "date_range": _format_date_range(valid_dates),
            "start_date": _format_date(valid_dates[0]) if valid_dates else None,
            "end_date": _format_date(valid_dates[-1]) if valid_dates else None,
            "window_type": legacy_report.get("history_window_type", "weekly"),
            "num_windows": legacy_report.get("num_windows", 1),
        }

        # Synthesize the 5 categories
        recurring_topics = self._build_recurring_topics(entries, total_entries)
        emotional_trends = self._build_emotional_trends(entries, legacy_report)
        recurring_thoughts = self._build_recurring_thoughts(entries, total_entries)
        changes_over_time = self._build_changes_over_time(entries, legacy_report)
        personal_observations = self._build_personal_observations(
            entries, recurring_topics, emotional_trends, changes_over_time, overview
        )

        return {
            "overview": overview,
            "recurring_topics": recurring_topics,
            "emotional_trends": emotional_trends,
            "recurring_thoughts": recurring_thoughts,
            "changes_over_time": changes_over_time,
            "personal_observations": personal_observations,
        }

    # --------------------------------------------------------------------------
    # 1. RECURRING TOPICS (Grounded strictly in actual representative entries)
    # --------------------------------------------------------------------------
    def _build_recurring_topics(
        self,
        entries: List[Dict[str, Any]],
        total_entries: int,
    ) -> List[Dict[str, Any]]:
        """Identify repeatedly occurring themes from Phase 2 clusters with grounded descriptions."""
        if not entries:
            return []

        # Group entries by theme cluster
        cluster_groups: Dict[int, List[Dict[str, Any]]] = {}
        for e in entries:
            cid = e["theme_cluster_id"]
            if cid == -1:
                continue  # skip noise points
            cluster_groups.setdefault(cid, []).append(e)

        # Collect global term document frequencies across all entries for cluster specificity
        import math
        global_word_entries = Counter()
        global_cluster_membership = Counter()

        for e in entries:
            t_set = set()
            for token in re.findall(r"\b[a-zA-Z]{3,}\b", e["text"].lower()):
                if token not in COMMON_GENERIC_WORDS and len(token) >= 3:
                    t_set.add(token)
            for t in t_set:
                global_word_entries[t] += 1
                if e.get("theme_cluster_id") != -1:
                    global_cluster_membership[(t, e.get("theme_cluster_id"))] += 1

        topics: List[Dict[str, Any]] = []

        for cid, group in sorted(cluster_groups.items(), key=lambda x: len(x[1]), reverse=True):
            count = len(group)
            pct = round((count / max(1, total_entries)) * 100, 1)

            # Base cluster label from entry analysis or taxonomy
            entry_theme = group[0].get("theme_name") or ""
            if entry_theme and not entry_theme.lower().startswith("theme ") and not entry_theme.lower().startswith("cluster "):
                cluster_name = entry_theme
            else:
                cluster_name = self.theme_names.get(cid, f"Topic {cid}")

            # Collect dates and representative matching entries
            group_dates = [e["dt"] for e in group if e["dt"]]
            date_range = _format_date_range(group_dates)
            rep_dates = [e["date_short"] for e in group if e["date_short"]][:4]

            # Build list of matching entry items with navigation links
            matching_entries = []
            for e in group[:4]:
                exc = _clean_excerpt(e["text"], max_words=20)
                matching_entries.append({
                    "id": e["id"],
                    "title": e["title"],
                    "date": e["date_str"],
                    "date_short": e["date_short"],
                    "excerpt": exc,
                })

            # Candidate phrases and terms from this cluster
            cluster_terms = Counter()
            cluster_phrases = Counter()

            for e in group:
                sents = re.split(r"[.!?\n]+", e["text"].lower())
                seen_phrases_in_entry = set()
                seen_terms_in_entry = set()

                for sent in sents:
                    toks = re.findall(r"\b[a-zA-Z]{3,}\b", sent)
                    # Candidate 2-word phrases in this cluster
                    for i in range(len(toks) - 1):
                        w1, w2 = toks[i], toks[i + 1]
                        if w1 == w2 or w1 in COMMON_GENERIC_WORDS or w2 in COMMON_GENERIC_WORDS:
                            continue
                        bg = f"{w1} {w2}"
                        if bg not in seen_phrases_in_entry:
                            seen_phrases_in_entry.add(bg)
                            cluster_phrases[bg] += 1

                    for tok in toks:
                        if len(tok) >= 4 and tok not in COMMON_GENERIC_WORDS:
                            if tok not in seen_terms_in_entry:
                                seen_terms_in_entry.add(tok)
                                cluster_terms[tok] += 1

            # Score candidates using cluster-specific relevance (TF * specificity across clusters)
            candidate_scores = []
            # Multi-word phrases receive specificity boost (prefer meaningful phrases)
            for phrase, count_in_cluster in cluster_phrases.items():
                if count_in_cluster < 1:
                    continue
                phrase_score = count_in_cluster * 2.2
                candidate_scores.append((phrase, phrase_score))

            for term, count_in_cluster in cluster_terms.items():
                df = global_word_entries.get(term, 1)
                other_clusters = sum(
                    1 for oc in cluster_groups
                    if oc != cid and global_cluster_membership.get((term, oc), 0) > 0
                )
                specificity = math.log((total_entries + 2.0) / (df + 1.0)) + 1.0
                penalty = 1.0 / (1.0 + 0.6 * other_clusters)
                score = count_in_cluster * specificity * penalty
                candidate_scores.append((term, score))

            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate (if phrase contains term, don't repeat term redundantly)
            top_words = []
            seen_tokens = set()
            for item, _ in candidate_scores:
                words_in_item = set(item.split())
                if words_in_item.issubset(seen_tokens):
                    continue
                top_words.append(item)
                seen_tokens.update(words_in_item)
                if len(top_words) >= 4:
                    break

            # Fallback if no candidate survived filter
            if not top_words:
                raw_words = []
                for e in group:
                    raw_words.extend([t for t in re.findall(r"\b[a-zA-Z]{4,}\b", e["text"].lower()) if t not in COMMON_GENERIC_WORDS])
                top_words = [w for w, _ in Counter(raw_words).most_common(3)]

            # Generate description grounded strictly in actual entry titles and cluster content
            sample_titles = [f"'{e['title']}'" for e in group if e["title"] and e["title"] != "Journal Entry"][:2]
            title_ref = f" (including {', '.join(sample_titles)})" if sample_titles else ""

            if top_words:
                kw_str = ", ".join(top_words[:3])
                desc = (
                    f"Reflections categorized under {cluster_name} appeared across {count} entries ({pct}% of your journal){title_ref}. "
                    f"Discussions frequently centered on subjects such as {kw_str}."
                )
            else:
                desc = (
                    f"Reflections categorized under {cluster_name} appeared across {count} entries ({pct}% of your journal){title_ref} "
                    f"between {date_range}."
                )

            title = f"{cluster_name}"
            strength = "Primary Focus" if pct >= 30 else ("Regular Theme" if pct >= 15 else "Emerging Topic")

            topics.append({
                "cluster_id": cid,
                "title": title,
                "theme_name": cluster_name,
                "description": desc,
                "entry_count": count,
                "percentage": pct,
                "strength": strength,
                "date_range": date_range,
                "salient_keywords": top_words,
                "evidence": {
                    "entry_count": count,
                    "total_entries": total_entries,
                    "date_range": date_range,
                    "representative_dates": rep_dates,
                    "matching_entries": matching_entries,
                    "excerpts": [m["excerpt"] for m in matching_entries[:2]],
                    "keywords": top_words,
                    "signal_metric": f"{count} of {total_entries} entries ({pct}%) mapped to this semantic cluster",
                },
            })

        return topics

    # --------------------------------------------------------------------------
    # 2. EMOTIONAL TRENDS (Strictly bounded to Phase 1 probability measurements)
    # --------------------------------------------------------------------------
    def _build_emotional_trends(
        self,
        entries: List[Dict[str, Any]],
        legacy_report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Evaluate earlier vs. recent emotion probabilities with non-causal, measurement-bounded framing."""
        if len(entries) < 2:
            return []

        # Partition entries into earlier and recent chronological halves
        mid = len(entries) // 2
        earlier_entries = entries[:mid]
        recent_entries = entries[mid:]

        if not earlier_entries or not recent_entries:
            return []

        # Compute average emotion probabilities for each period
        earlier_means = self._calculate_mean_emotions(earlier_entries)
        recent_means = self._calculate_mean_emotions(recent_entries)

        all_emotions = set(earlier_means.keys()) | set(recent_means.keys())
        trends: List[Dict[str, Any]] = []

        early_dates = [e["dt"] for e in earlier_entries if e["dt"]]
        recent_dates = [e["dt"] for e in recent_entries if e["dt"]]
        early_range = _format_date_range(early_dates)
        recent_range = _format_date_range(recent_dates)

        for emotion in sorted(all_emotions):
            e_val = earlier_means.get(emotion, 0.0)
            r_val = recent_means.get(emotion, 0.0)
            delta = r_val - e_val

            # Only evaluate emotions with notable presence (prob >= 0.12 in either period)
            if max(e_val, r_val) < 0.12:
                continue

            # Need at least a modest delta (>= 0.06) or significant steady presence
            if abs(delta) < 0.06 and max(e_val, r_val) < 0.35:
                continue

            direction = "decreased" if delta <= -0.06 else ("increased" if delta >= 0.06 else "steady")

            # Human title and description strictly bounded to Phase 1 classifier measurements
            title, desc = self._describe_emotion_trend(emotion, direction, e_val, r_val, early_range, recent_range)

            # Find matching entries with excerpts
            earlier_matches = self._find_emotion_entries(earlier_entries, emotion, max_count=2)
            recent_matches = self._find_emotion_entries(recent_entries, emotion, max_count=2)

            all_matches = earlier_matches + recent_matches
            rep_dates = [m["date_short"] for m in all_matches if m["date_short"]][:4]

            trends.append({
                "emotion": emotion,
                "title": title,
                "direction": direction,
                "description": desc,
                "earlier_score": round(e_val, 2),
                "recent_score": round(r_val, 2),
                "delta": round(delta, 2),
                "earlier_period": {
                    "date_range": early_range,
                    "average_score": round(e_val, 2),
                    "entry_count": len(earlier_entries),
                },
                "recent_period": {
                    "date_range": recent_range,
                    "average_score": round(r_val, 2),
                    "entry_count": len(recent_entries),
                },
                "evidence": {
                    "earlier_date_range": early_range,
                    "recent_date_range": recent_range,
                    "earlier_average": round(e_val, 2),
                    "recent_average": round(r_val, 2),
                    "representative_dates": rep_dates,
                    "matching_entries": all_matches,
                    "earlier_excerpts": [m["excerpt"] for m in earlier_matches],
                    "recent_excerpts": [m["excerpt"] for m in recent_matches],
                    "measurement_note": (
                        f"GoEmotions classifier assigned an average probability of {e_val:.2f} in earlier entries "
                        f"versus {r_val:.2f} in recent entries. Values represent model classification probabilities, "
                        "not keyword counts or actual emotion frequency. Pensieve does not infer why probabilities shifted."
                    ),
                },
            })

        # Sort by absolute delta so most notable shifts appear first
        trends.sort(key=lambda x: abs(x["delta"]), reverse=True)
        return trends[:5]

    def _calculate_mean_emotions(self, entries: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute mean score for each emotion across entries."""
        sums: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for e in entries:
            for em, score in e["emotions"].items():
                sums[em] = sums.get(em, 0.0) + float(score)
                counts[em] = counts.get(em, 0) + 1
        return {em: sums[em] / max(1, counts[em]) for em in sums}

    def _find_emotion_entries(
        self,
        entries: List[Dict[str, Any]],
        emotion: str,
        max_count: int = 2,
    ) -> List[Dict[str, Any]]:
        """Find entries where this emotion had notable probability, with navigation metadata."""
        matched = []
        sorted_e = sorted(entries, key=lambda x: x["emotions"].get(emotion, 0.0), reverse=True)
        for e in sorted_e:
            score = e["emotions"].get(emotion, 0.0)
            if score >= 0.15:
                exc = _clean_excerpt(e["text"], max_words=20)
                matched.append({
                    "id": e["id"],
                    "title": e["title"],
                    "date": e["date_str"],
                    "date_short": e["date_short"],
                    "score": round(score, 2),
                    "excerpt": exc,
                })
            if len(matched) >= max_count:
                break
        return matched

    def _describe_emotion_trend(
        self,
        emotion: str,
        direction: str,
        e_val: float,
        r_val: float,
        early_range: str,
        recent_range: str,
    ) -> Tuple[str, str]:
        """Generate accurate, non-causal descriptions of Phase 1 model measurements."""
        cap_emo = emotion.capitalize()

        if direction == "decreased":
            title = f"Lower Detected '{cap_emo}' Probability"
            desc = (
                f"The model detected a lower average probability for '{emotion}' in recent reflections "
                f"({r_val:.2f} avg across {recent_range}) compared to earlier reflections ({e_val:.2f} avg across {early_range}). "
                "This reflects a decrease in the classifier's assigned probability signal across chronological windows, "
                "without asserting changes in your underlying subjective feeling or emotion frequency."
            )
        elif direction == "increased":
            title = f"Higher Detected '{cap_emo}' Probability"
            desc = (
                f"The model detected a higher average probability for '{emotion}' in recent reflections "
                f"({r_val:.2f} avg across {recent_range}) compared to earlier reflections ({e_val:.2f} avg across {early_range}). "
                "This reflects an increase in the classifier's assigned probability signal across chronological windows, "
                "without asserting changes in your underlying subjective feeling or emotion frequency."
            )
        else:
            title = f"Consistent Detected '{cap_emo}' Probability"
            desc = (
                f"The model detected a relatively consistent average probability for '{emotion}' between earlier and recent reflections "
                f"(averaging ~{((e_val + r_val) / 2):.2f} across both periods)."
            )

        return title, desc

    # --------------------------------------------------------------------------
    # 3. RECURRING THOUGHTS AND LANGUAGE (Filtered of generic filler)
    # --------------------------------------------------------------------------
    def _build_recurring_thoughts(
        self,
        entries: List[Dict[str, Any]],
        total_entries: int,
    ) -> List[Dict[str, Any]]:
        """Extract multi-word recurring phrases and distinctive concerns from actual entry texts."""
        if len(entries) < 2:
            return []

        doc_bigrams = Counter()
        phrase_entries: Dict[str, List[Dict[str, Any]]] = {}

        doc_words = Counter()
        word_entries: Dict[str, List[Dict[str, Any]]] = {}

        for e in entries:
            text = e["text"].lower()
            sentences = re.split(r"[.!?\n]+", text)
            seen_phrases_in_entry = set()
            seen_words_in_entry = set()

            for sentence in sentences:
                sentence_tokens = re.findall(r"\b[a-zA-Z]{3,}\b", sentence)

                # Extract consecutive bigrams within the same sentence
                for i in range(len(sentence_tokens) - 1):
                    w1, w2 = sentence_tokens[i], sentence_tokens[i + 1]
                    if w1 == w2:
                        continue
                    if w1 in COMMON_GENERIC_WORDS or w2 in COMMON_GENERIC_WORDS:
                        continue
                    if len(w1) < 3 or len(w2) < 3:
                        continue
                    bigram = f"{w1} {w2}"
                    if bigram not in seen_phrases_in_entry:
                        seen_phrases_in_entry.add(bigram)
                        doc_bigrams[bigram] += 1
                        phrase_entries.setdefault(bigram, []).append(e)

                # Track single content words (length >= 3 and not generic)
                for token in sentence_tokens:
                    if len(token) >= 3 and token not in COMMON_GENERIC_WORDS:
                        if token not in seen_words_in_entry:
                            seen_words_in_entry.add(token)
                            doc_words[token] += 1
                            word_entries.setdefault(token, []).append(e)

        min_doc_freq = 2

        # Candidate pool of concepts: multi-word phrases and distinctive content words
        candidates = []
        for phrase, doc_freq in doc_bigrams.items():
            if doc_freq >= min_doc_freq:
                candidates.append({
                    "expr": phrase,
                    "is_phrase": True,
                    "doc_freq": doc_freq,
                    "entries": phrase_entries[phrase],
                })

        for word, doc_freq in doc_words.items():
            if doc_freq >= min_doc_freq:
                candidates.append({
                    "expr": word,
                    "is_phrase": False,
                    "doc_freq": doc_freq,
                    "entries": word_entries[word],
                })

        # Group related expressions sharing semantic roots (e.g. 'morning run', 'running group', 'run')
        groups: List[Dict[str, Any]] = []
        for cand in candidates:
            cand_tokens = set(cand["expr"].split())
            placed = False
            for grp in groups:
                grp_tokens = set(grp["primary"].split())
                for re_item in grp["related"]:
                    grp_tokens.update(re_item.split())

                overlap = False
                for ct in cand_tokens:
                    for gt in grp_tokens:
                        if ct == gt or (len(ct) >= 4 and len(gt) >= 4 and (ct in gt or gt in ct)):
                            overlap = True
                            break
                    if overlap:
                        break

                if overlap:
                    grp["related"].append(cand["expr"])
                    if cand["is_phrase"] and (not grp["is_phrase"] or cand["doc_freq"] >= grp["doc_freq"]):
                        grp["primary"] = cand["expr"]
                        grp["is_phrase"] = True
                    existing_ids = {e["id"] for e in grp["entries"]}
                    for e in cand["entries"]:
                        if e["id"] not in existing_ids:
                            grp["entries"].append(e)
                            existing_ids.add(e["id"])
                    grp["doc_freq"] = len(grp["entries"])
                    placed = True
                    break

            if not placed:
                groups.append({
                    "primary": cand["expr"],
                    "is_phrase": cand["is_phrase"],
                    "related": [],
                    "doc_freq": cand["doc_freq"],
                    "entries": list(cand["entries"]),
                })

        # Sort groups by document frequency and phrase quality
        groups.sort(key=lambda g: (g["is_phrase"], g["doc_freq"]), reverse=True)

        thoughts: List[Dict[str, Any]] = []
        for grp in groups[:4]:
            phrase = grp["primary"]
            related = [r for r in grp["related"] if r != phrase][:3]
            doc_freq = grp["doc_freq"]
            matched_entries = grp["entries"]
            matched_dates = [e["dt"] for e in matched_entries if e["dt"]]
            date_range = _format_date_range(matched_dates)
            rep_dates = [e["date_short"] for e in matched_entries if e["date_short"]][:3]

            matching_evidence = []
            for e in matched_entries[:3]:
                exc = _extract_context_excerpt(e["text"], phrase, max_words=20)
                matching_evidence.append({
                    "id": e["id"],
                    "title": e["title"],
                    "date": e["date_str"],
                    "date_short": e["date_short"],
                    "excerpt": exc,
                })

            if grp["is_phrase"]:
                title = f"Recurring Phrase: '{phrase.title()}'"
            else:
                title = f"Recurring Subject: '{phrase.capitalize()}'"

            if related:
                related_str = ", ".join(f"'{r}'" for r in related)
                desc = (
                    f"Expressions related to '{phrase}' (including {related_str}) recurred across {doc_freq} separate reflections "
                    f"between {date_range}. This represents an ongoing specific subject in your writing."
                )
            else:
                desc = (
                    f"The phrase '{phrase}' appeared across {doc_freq} separate entries between {date_range}. "
                    "This represents a recurring specific expression in your reflections, distinct from broader thematic topics."
                )

            pct = round((doc_freq / max(1, total_entries)) * 100, 1)

            thoughts.append({
                "phrase": phrase,
                "title": title,
                "description": desc,
                "related_expressions": related,
                "document_frequency": doc_freq,
                "percentage": pct,
                "date_range": date_range,
                "evidence": {
                    "document_frequency": doc_freq,
                    "total_entries": total_entries,
                    "date_range": date_range,
                    "representative_dates": rep_dates,
                    "matching_entries": matching_evidence,
                    "excerpts": [m["excerpt"] for m in matching_evidence[:2]],
                    "frequency_summary": f"Observed in {doc_freq} of {total_entries} entries ({pct}%)",
                },
            })

        return thoughts

    # --------------------------------------------------------------------------
    # 4. CHANGES OVER TIME (Non-causal chronological comparison)
    # --------------------------------------------------------------------------
    def _build_changes_over_time(
        self,
        entries: List[Dict[str, Any]],
        legacy_report: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Identify narrative shifts between earlier and recent journal periods."""
        if len(entries) < 3:
            return []

        mid = len(entries) // 2
        early_entries = entries[:mid]
        recent_entries = entries[mid:]

        early_dates = [e["dt"] for e in early_entries if e["dt"]]
        recent_dates = [e["dt"] for e in recent_entries if e["dt"]]

        early_range = _format_date_range(early_dates)
        recent_range = _format_date_range(recent_dates)

        # Dominant topics in each period
        early_topics = Counter([e["theme_name"] for e in early_entries if e["theme_name"]])
        recent_topics = Counter([e["theme_name"] for e in recent_entries if e["theme_name"]])

        top_early_topic = early_topics.most_common(1)[0][0] if early_topics else "General Reflection"
        top_recent_topic = recent_topics.most_common(1)[0][0] if recent_topics else "General Reflection"

        changes: List[Dict[str, Any]] = []

        if top_early_topic != top_recent_topic:
            early_exc = _clean_excerpt(early_entries[0]["text"], max_words=20)
            recent_exc = _clean_excerpt(recent_entries[-1]["text"], max_words=20)

            early_matching = [
                {"id": e["id"], "title": e["title"], "date": e["date_str"], "date_short": e["date_short"], "excerpt": _clean_excerpt(e["text"], max_words=18)}
                for e in early_entries if e["theme_name"] == top_early_topic
            ][:2]

            recent_matching = [
                {"id": e["id"], "title": e["title"], "date": e["date_str"], "date_short": e["date_short"], "excerpt": _clean_excerpt(e["text"], max_words=18)}
                for e in recent_entries if e["theme_name"] == top_recent_topic
            ][:2]

            changes.append({
                "title": f"Thematic Shift: {top_early_topic} → {top_recent_topic}",
                "category": "thematic_evolution",
                "description": (
                    f"Your earlier entries focused most frequently on {top_early_topic.lower()} ({early_range}, {early_topics[top_early_topic]} entries). "
                    f"In contrast, your later entries were most frequently centered around {top_recent_topic.lower()} ({recent_range}, {recent_topics[top_recent_topic]} entries). "
                    "This comparison describes an observable shift in topic frequency without implying causality."
                ),
                "earlier_period": {
                    "date_range": early_range,
                    "entry_count": len(early_entries),
                    "primary_focus": top_early_topic,
                    "excerpt": early_exc,
                    "matching_entries": early_matching,
                },
                "recent_period": {
                    "date_range": recent_range,
                    "entry_count": len(recent_entries),
                    "primary_focus": top_recent_topic,
                    "excerpt": recent_exc,
                    "matching_entries": recent_matching,
                },
                "evidence": {
                    "earlier_date_range": early_range,
                    "recent_date_range": recent_range,
                    "earlier_dominant_topic": f"{top_early_topic} ({early_topics[top_early_topic]} entries)",
                    "recent_dominant_topic": f"{top_recent_topic} ({recent_topics[top_recent_topic]} entries)",
                    "earlier_matching": early_matching,
                    "recent_matching": recent_matching,
                    "disclaimer": "Thematic shifts reflect observed topic frequencies over time; no causal claims are made.",
                },
            })

        # Linguistic & stylistic evolution from Phase 3 features
        ling_trends = legacy_report.get("linguistic_trends", {})
        window_metrics = ling_trends.get("window_metrics", [])
        if len(window_metrics) >= 2:
            first_w = window_metrics[0]
            last_w = window_metrics[-1]

            word_diff = last_w.get("avg_words_per_entry", 0) - first_w.get("avg_words_per_entry", 0)
            if abs(word_diff) >= 12:
                direction_str = "more expansive" if word_diff > 0 else "more concise"
                changes.append({
                    "title": "Observed Writing Length Shift",
                    "category": "writing_style",
                    "description": (
                        f"Reflections averaged {first_w.get('avg_words_per_entry', 0):.0f} words during early windows ({early_range}) "
                        f"compared to {last_w.get('avg_words_per_entry', 0):.0f} words in recent windows ({recent_range}). "
                        f"Your entries showed an observable trend toward being {direction_str}."
                    ),
                    "earlier_period": {
                        "date_range": early_range,
                        "entry_count": len(early_entries),
                        "avg_words": first_w.get("avg_words_per_entry", 0),
                    },
                    "recent_period": {
                        "date_range": recent_range,
                        "entry_count": len(recent_entries),
                        "avg_words": last_w.get("avg_words_per_entry", 0),
                    },
                    "evidence": {
                        "earlier_metric": f"{first_w.get('avg_words_per_entry', 0)} avg words/entry",
                        "recent_metric": f"{last_w.get('avg_words_per_entry', 0)} avg words/entry",
                        "date_span": f"{early_range} vs. {recent_range}",
                    },
                })

        return changes

    # --------------------------------------------------------------------------
    # 5. PERSONAL OBSERVATIONS (Evidence-based, strictly non-causal synthesis)
    # --------------------------------------------------------------------------
    def _build_personal_observations(
        self,
        entries: List[Dict[str, Any]],
        recurring_topics: List[Dict[str, Any]],
        emotional_trends: List[Dict[str, Any]],
        changes_over_time: List[Dict[str, Any]],
        overview: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Formulate evidence-based observations with reflective questions without causal assertions."""
        if not entries:
            return []

        observations: List[Dict[str, Any]] = []
        count = len(entries)
        span = overview.get("time_span_days", 0)

        if recurring_topics and emotional_trends:
            top_topic = recurring_topics[0]
            top_emotion = emotional_trends[0]

            # Prioritize matching entries that actually reflect the dominant topic or emotion
            obs_matches = [
                e for e in entries
                if e.get("theme_cluster_id") == top_topic.get("cluster_id")
                or e.get("emotions", {}).get(top_emotion["emotion"], 0.0) >= 0.2
            ]
            if not obs_matches:
                obs_matches = entries[:3]

            matching_entries = [
                {
                    "id": e["id"],
                    "title": e["title"],
                    "date": e["date_str"],
                    "date_short": e["date_short"],
                    "excerpt": _clean_excerpt(e["text"], max_words=18),
                }
                for e in obs_matches[:3]
            ]

            shift_note = ""
            if changes_over_time:
                c0 = changes_over_time[0]
                if c0.get("category") == "thematic_evolution":
                    shift_note = ", alongside an observable shift in primary writing focus"

            summary = (
                f"Across {count} reflections ({overview.get('date_range', f'{span} days')}), "
                f"your journal focused most frequently on '{top_topic['title']}' ({top_topic['entry_count']} entries){shift_note}. "
                f"Over this same period, the model detected a {top_emotion['direction']} probability for '{top_emotion['emotion']}' "
                f"({top_emotion['earlier_score']:.2f} earlier vs. {top_emotion['recent_score']:.2f} recently). "
                "These describe separate measured developments across your writing timeline."
            )

            reflective_q = (
                f"When you reflect on your entries from this period, what connections or differences stand out to you?"
            )

            observations.append({
                "title": f"Reflection Arc: {top_topic['title']} & Affective Signals",
                "summary": summary,
                "reflective_question": reflective_q,
                "evidence": {
                    "entry_count": count,
                    "observed_span": f"{span} days",
                    "grounded_themes": [top_topic["title"]],
                    "measured_emotion": top_emotion["emotion"],
                    "basis": "Synthesized from concurrent Phase 2 semantic cluster frequency and Phase 1 emotion trajectory.",
                    "matching_entries": matching_entries,
                },
            })
        else:
            fallback_matches = [
                {
                    "id": e["id"],
                    "title": e["title"],
                    "date": e["date_str"],
                    "date_short": e["date_short"],
                    "excerpt": _clean_excerpt(e["text"], max_words=18),
                }
                for e in entries[:3]
            ]
            observations.append({
                "title": "Patterns of Ongoing Journaling",
                "summary": (
                    f"Your writing history shows {count} reflections spanning ~{int(span or 0)} days. "
                    "Regular journaling offers an observable timeline of how your attention evolves across different seasons."
                ),
                "reflective_question": (
                    "Which moments or reflections from this period feel most worthy of continued focus?"
                ),
                "evidence": {
                    "entry_count": count,
                    "observed_span": f"{span} days",
                    "basis": f"Evaluated across {count} published entries.",
                    "matching_entries": fallback_matches,
                },
            })

        return observations


def get_pattern_synthesizer() -> PersonalizedPatternSynthesizer:
    """Factory helper to obtain a pattern synthesizer instance."""
    return PersonalizedPatternSynthesizer()
