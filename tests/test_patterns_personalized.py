"""Tests for personalized, evidence-based longitudinal pattern synthesis."""

from datetime import datetime, timedelta, timezone
from app.services.pattern_service import PersonalizedPatternSynthesizer, get_pattern_synthesizer


def _generate_synthetic_journal_history():
    """Generate 12 realistic dated synthetic journal entries with ML metadata."""
    base_time = datetime(2026, 1, 2, 9, 30, tzinfo=timezone.utc)
    return [
        # Week 1: Work pressure and sprint stress
        {
            "id": "1",
            "title": "Deadline Stress",
            "content": "Stressed about the quarterly product launch deadline. My chest is tight and our staging server crashed again this morning!",
            "created_at": base_time,
            "emotions": {"annoyance": 0.74, "nervousness": 0.81, "joy": 0.05, "gratitude": 0.08},
            "theme_cluster_id": 0,
            "theme": "Work & Engineering",
        },
        {
            "id": "2",
            "title": "Late Night Debugging",
            "content": "Another late night debugging pull requests. Feeling completely exhausted from constant context switching.",
            "created_at": base_time + timedelta(days=2),
            "emotions": {"annoyance": 0.68, "nervousness": 0.65, "joy": 0.09, "gratitude": 0.11},
            "theme_cluster_id": 0,
            "theme": "Work & Engineering",
        },
        {
            "id": "3",
            "title": "Back to Back Meetings",
            "content": "Spent three hours in back-to-back status meetings. Why does every minor design update require a committee discussion?",
            "created_at": base_time + timedelta(days=4),
            "emotions": {"annoyance": 0.75, "nervousness": 0.40, "joy": 0.06, "gratitude": 0.07},
            "theme_cluster_id": 0,
            "theme": "Work & Engineering",
        },
        # Week 2: Relief, celebration and social connection
        {
            "id": "4",
            "title": "Patch Shipped",
            "content": "Finally shipped the backend patch! Team celebrated with coffee and pastries. Feeling a palpable wave of relief today.",
            "created_at": base_time + timedelta(days=8),
            "emotions": {"annoyance": 0.22, "nervousness": 0.18, "joy": 0.64, "gratitude": 0.58},
            "theme_cluster_id": 0,
            "theme": "Work & Engineering",
        },
        {
            "id": "5",
            "title": "Dinner with Friends",
            "content": "Met Sarah and Marcus for dinner at our favorite noodle shop. We talked for hours and laughed so hard my cheeks hurt.",
            "created_at": base_time + timedelta(days=10),
            "emotions": {"annoyance": 0.04, "nervousness": 0.08, "joy": 0.82, "gratitude": 0.76},
            "theme_cluster_id": 1,
            "theme": "Relationships & Social Connection",
        },
        {
            "id": "6",
            "title": "Quiet Evening",
            "content": "Quiet evening at home catching up on personal reading. Need to make time for hobbies and self-care more consistently.",
            "created_at": base_time + timedelta(days=12),
            "emotions": {"annoyance": 0.08, "nervousness": 0.12, "joy": 0.51, "gratitude": 0.62},
            "theme_cluster_id": 2,
            "theme": "Personal Habits & Goals",
        },
        # Week 3: Health, fitness & morning routine
        {
            "id": "7",
            "title": "First Morning Run",
            "content": "Woke up at 6 AM and went for a brisk morning run around the neighborhood lake. The crisp morning air was wonderfully invigorating.",
            "created_at": base_time + timedelta(days=15),
            "emotions": {"annoyance": 0.03, "nervousness": 0.05, "joy": 0.78, "gratitude": 0.72},
            "theme_cluster_id": 3,
            "theme": "Health, Fitness & Vitality",
        },
        {
            "id": "8",
            "title": "Second Morning Run",
            "content": "Second morning run this week! Paced myself well and prepared a nutritious smoothie afterwards. Feeling focused and present.",
            "created_at": base_time + timedelta(days=17),
            "emotions": {"annoyance": 0.02, "nervousness": 0.04, "joy": 0.81, "gratitude": 0.79},
            "theme_cluster_id": 3,
            "theme": "Health, Fitness & Vitality",
        },
        {
            "id": "9",
            "title": "Productive and Calm",
            "content": "Productive workday without unnecessary fire drills. Left the office on time and enjoyed a calm walk through the park.",
            "created_at": base_time + timedelta(days=19),
            "emotions": {"annoyance": 0.10, "nervousness": 0.09, "joy": 0.68, "gratitude": 0.65},
            "theme_cluster_id": 2,
            "theme": "Personal Habits & Goals",
        },
        # Week 4: Vitality, gratitude and habit consolidation
        {
            "id": "10",
            "title": "Trail Run Saturday",
            "content": "Saturday morning run with my running group. The sunrise over the ridge was stunning. So grateful for my health!",
            "created_at": base_time + timedelta(days=22),
            "emotions": {"annoyance": 0.02, "nervousness": 0.03, "joy": 0.91, "gratitude": 0.89},
            "theme_cluster_id": 3,
            "theme": "Health, Fitness & Vitality",
        },
        {
            "id": "11",
            "title": "Evening Reading",
            "content": "Finished reading my third book of the month. Establishing this evening reading routine has drastically improved my sleep quality.",
            "created_at": base_time + timedelta(days=24),
            "emotions": {"annoyance": 0.01, "nervousness": 0.04, "joy": 0.74, "gratitude": 0.82},
            "theme_cluster_id": 2,
            "theme": "Personal Habits & Goals",
        },
        {
            "id": "12",
            "title": "Month Reflection",
            "content": "Reflecting on this month: moving from chaotic project panic to steady morning run routines has completely transformed my daily energy.",
            "created_at": base_time + timedelta(days=26),
            "emotions": {"annoyance": 0.05, "nervousness": 0.06, "joy": 0.87, "gratitude": 0.92},
            "theme_cluster_id": 3,
            "theme": "Health, Fitness & Vitality",
        },
    ]


def test_personalized_pattern_synthesis_categories():
    """Verify that all 5 user-centric categories and overview are synthesized."""
    synthesizer = get_pattern_synthesizer()
    entries = _generate_synthetic_journal_history()
    legacy_report = {
        "status": "success",
        "entry_count": 12,
        "time_span_days": 26.0,
        "history_window_type": "weekly",
        "num_windows": 4,
    }

    result = synthesizer.synthesize(entries, legacy_report)

    # 1. Overview
    assert "overview" in result
    overview = result["overview"]
    assert overview["entry_count"] == 12
    assert overview["time_span_days"] == 26.0
    assert "date_range" in overview

    # 2. Recurring Topics
    assert "recurring_topics" in result
    topics = result["recurring_topics"]
    assert len(topics) >= 2
    # Ensure no internal cluster IDs appear as titles
    for t in topics:
        assert "Cluster 0" not in t["title"]
        assert "Cluster 1" not in t["title"]
        assert t["entry_count"] > 0
        assert "evidence" in t
        assert len(t["evidence"]["excerpts"]) > 0

    # 3. Emotional Trends
    assert "emotional_trends" in result
    emotions = result["emotional_trends"]
    assert len(emotions) >= 1
    # Annoyance should show a downward trend
    annoyance_trend = next((e for e in emotions if e["emotion"] == "annoyance"), None)
    assert annoyance_trend is not None
    assert annoyance_trend["direction"] == "decreased"
    assert annoyance_trend["delta"] < 0
    assert "evidence" in annoyance_trend
    assert len(annoyance_trend["evidence"]["earlier_excerpts"]) > 0

    # 4. Recurring Thoughts
    assert "recurring_thoughts" in result
    thoughts = result["recurring_thoughts"]
    # Check that 'morning run' was identified as a multi-entry recurring thought
    morning_run_thought = next((t for t in thoughts if "run" in t["phrase"]), None)
    assert morning_run_thought is not None
    assert morning_run_thought["document_frequency"] >= 2
    assert "evidence" in morning_run_thought

    # 5. Changes Over Time
    assert "changes_over_time" in result
    changes = result["changes_over_time"]
    assert len(changes) >= 1
    first_change = changes[0]
    assert "earlier_period" in first_change
    assert "recent_period" in first_change

    # 6. Personal Observations
    assert "personal_observations" in result
    observations = result["personal_observations"]
    assert len(observations) >= 1
    for obs in observations:
        assert "summary" in obs
        assert "reflective_question" in obs
        assert "evidence" in obs


def test_evidence_correctness():
    """Verify that every excerpt and date in evidence matches the actual input entries."""
    synthesizer = get_pattern_synthesizer()
    entries = _generate_synthetic_journal_history()
    result = synthesizer.synthesize(entries, {"status": "success", "time_span_days": 26.0})

    # Check recurring topics excerpts
    for topic in result["recurring_topics"]:
        for exc in topic["evidence"]["excerpts"]:
            # Cleaned excerpt must be a substring of one of the actual entries
            assert any(exc[:15] in e["content"] for e in entries), f"Excerpt '{exc}' not found in entries"

    # Check emotion trend excerpts
    for trend in result["emotional_trends"]:
        for exc in trend["evidence"]["earlier_excerpts"]:
            assert any(exc[:15] in e["content"] for e in entries)


def test_noisy_clusters_ignored():
    """Verify that noise cluster (-1) is ignored and never presented as a topic."""
    synthesizer = get_pattern_synthesizer()
    entries = [
        {
            "id": "1",
            "title": "Noise 1",
            "content": "Unrelated single reflection note.",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "emotions": {"neutral": 0.9},
            "theme_cluster_id": -1,  # Noise cluster
            "theme": "Noise",
        },
        {
            "id": "2",
            "title": "Noise 2",
            "content": "Another isolated piece of text.",
            "created_at": datetime(2026, 1, 5, tzinfo=timezone.utc),
            "emotions": {"neutral": 0.8},
            "theme_cluster_id": -1,
            "theme": "Noise",
        },
    ]

    result = synthesizer.synthesize(entries, {"status": "success"})
    # Cluster -1 must NOT produce a recurring topic
    assert len(result["recurring_topics"]) == 0


def test_user_isolation_guarantee():
    """Verify that synthesis only contains entries provided to it (multi-tenant boundary)."""
    synthesizer = get_pattern_synthesizer()
    user_1_entries = [
        {
            "id": "u1-1",
            "title": "User 1 Secret",
            "content": "This is user 1 confidential medical journal entry.",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "emotions": {"sadness": 0.6},
            "theme_cluster_id": 0,
        },
    ]
    user_2_entries = [
        {
            "id": "u2-1",
            "title": "User 2 Entry",
            "content": "User 2 writing about hiking in the mountains.",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "emotions": {"joy": 0.7},
            "theme_cluster_id": 3,
        },
    ]

    result_2 = synthesizer.synthesize(user_2_entries, {"status": "success"})

    # Check that user 1's secret text never appears anywhere in user 2's patterns
    result_2_str = str(result_2)
    assert "user 1 confidential" not in result_2_str
    assert "u1-1" not in result_2_str


def test_non_diagnostic_language():
    """Verify observations never use clinical diagnostic terms."""
    synthesizer = get_pattern_synthesizer()
    entries = _generate_synthetic_journal_history()
    result = synthesizer.synthesize(entries, {"status": "success", "time_span_days": 26.0})

    forbidden_terms = [
        "clinical depression",
        "diagnose",
        "disorder",
        "pathology",
        "patient",
        "prescribe",
        "treatment",
        "symptom of",
    ]

    full_output = str(result).lower()
    for term in forbidden_terms:
        assert term not in full_output, f"Forbidden diagnostic term '{term}' found in output!"


def test_generic_term_filtering():
    """Verify that generic words ('past', 'noticed', 'sitting', 'mind', 'morning', 'instead', etc.) are never extracted as concepts."""
    synthesizer = get_pattern_synthesizer()
    entries = [
        {
            "id": "g1",
            "title": "Entry One",
            "content": "In the past I noticed I was sitting quietly with a racing mind instead of working. Time passed quickly today.",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "emotions": {"joy": 0.5},
            "theme_cluster_id": 0,
        },
        {
            "id": "g2",
            "title": "Entry Two",
            "content": "Another day I noticed past patterns while sitting at my desk. My mind was wandering yesterday instead of focusing.",
            "created_at": datetime(2026, 1, 3, tzinfo=timezone.utc),
            "emotions": {"joy": 0.6},
            "theme_cluster_id": 0,
        },
        {
            "id": "g3",
            "title": "Entry Three",
            "content": "Took my time sitting with peace of mind. Noticed how past worries faded.",
            "created_at": datetime(2026, 1, 5, tzinfo=timezone.utc),
            "emotions": {"joy": 0.7},
            "theme_cluster_id": 0,
        },
    ]

    result = synthesizer.synthesize(entries, {"status": "success", "time_span_days": 4.0})

    generic_terms = {
        "past", "noticed", "sitting", "mind", "morning", "instead", "took", "one",
        "time", "today", "yesterday", "day", "days"
    }

    # Check recurring thoughts
    for thought in result.get("recurring_thoughts", []):
        phrase = thought["phrase"].lower()
        # Phrase must not be just a single generic word
        assert phrase not in generic_terms, f"Generic word '{phrase}' extracted as recurring thought!"
        # Nor can any word in a phrase be a generic word
        for w in phrase.split():
            assert w not in generic_terms, f"Generic word '{w}' found inside phrase '{phrase}'!"

    # Check salient keywords in topics
    for topic in result.get("recurring_topics", []):
        keywords = topic.get("salient_keywords", [])
        for kw in keywords:
            assert kw.lower() not in generic_terms, f"Generic word '{kw}' extracted as topic keyword!"


def test_semantic_grouping_and_distinctive_cluster_keywords():
    """Verify semantic grouping of related expressions and cluster-specific keyword relevance."""
    synthesizer = get_pattern_synthesizer()
    entries = _generate_synthetic_journal_history()
    result = synthesizer.synthesize(entries, {"status": "success", "time_span_days": 26.0})

    # 1. Semantic grouping in recurring thoughts
    thoughts = result["recurring_thoughts"]
    run_thought = next((t for t in thoughts if "run" in t["phrase"]), None)
    assert run_thought is not None
    # Check that related expressions list exists and groups terms
    assert "related_expressions" in run_thought
    assert run_thought["document_frequency"] >= 2

    # 2. Distinctive cluster keywords (multi-word phrases preferred where available)
    topics = result["recurring_topics"]
    work_topic = next((t for t in topics if t["cluster_id"] == 0), None)
    assert work_topic is not None
    assert len(work_topic["salient_keywords"]) > 0
    # Cluster keywords should NOT be generic words
    for kw in work_topic["salient_keywords"]:
        assert kw not in {"past", "noticed", "sitting", "mind", "time", "one", "day"}

    # 3. Emotional trends describe model probability without keyword counts or causal reasons
    for trend in result["emotional_trends"]:
        assert "probability" in trend["title"].lower() or "signal" in trend["title"].lower()
        desc = trend["description"].lower()
        assert "keyword count" not in desc
        assert "because" not in desc
        assert "caused" not in desc
        assert "probability" in desc


def test_emotional_interpretation_boundaries():
    """Verify emotions only claim what Phase 1 measures and never conflate disappointment with stress."""
    synthesizer = get_pattern_synthesizer()
    entries = [
        {
            "id": "e1",
            "title": "Early Entry",
            "content": "Felt a lot of disappointment with our project results.",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "emotions": {"disappointment": 0.82, "sadness": 0.40},
            "theme_cluster_id": 0,
        },
        {
            "id": "e2",
            "title": "Middle Entry",
            "content": "Still some disappointment around the team outcome.",
            "created_at": datetime(2026, 1, 5, tzinfo=timezone.utc),
            "emotions": {"disappointment": 0.70, "sadness": 0.30},
            "theme_cluster_id": 0,
        },
        {
            "id": "e3",
            "title": "Recent Entry 1",
            "content": "Moving on to new work with acceptance and focus.",
            "created_at": datetime(2026, 1, 10, tzinfo=timezone.utc),
            "emotions": {"disappointment": 0.10, "sadness": 0.05},
            "theme_cluster_id": 0,
        },
        {
            "id": "e4",
            "title": "Recent Entry 2",
            "content": "Good momentum on the next initiative.",
            "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc),
            "emotions": {"disappointment": 0.08, "sadness": 0.05},
            "theme_cluster_id": 0,
        },
    ]

    result = synthesizer.synthesize(entries, {"status": "success", "time_span_days": 14.0})

    disappointment_trend = next((t for t in result["emotional_trends"] if t["emotion"] == "disappointment"), None)
    assert disappointment_trend is not None
    assert disappointment_trend["direction"] == "decreased"

    # Must NOT claim acute stress or causal reduction in stress
    desc = disappointment_trend["description"].lower()
    assert "acute stress" not in desc
    assert "stress" not in desc
    assert "deadlines became less stressful" not in desc

    # Must include measurement note clarifying probability vs frequency
    measurement_note = disappointment_trend["evidence"]["measurement_note"]
    assert "GoEmotions" in measurement_note or "probabilities" in measurement_note
    assert "probability" in measurement_note.lower()


def test_supporting_evidence_navigation_metadata():
    """Verify that all evidence sections provide matching entries with valid navigation metadata."""
    synthesizer = get_pattern_synthesizer()
    entries = _generate_synthetic_journal_history()
    result = synthesizer.synthesize(entries, {"status": "success", "time_span_days": 26.0})

    # Check recurring topics evidence
    for topic in result["recurring_topics"]:
        evidence = topic["evidence"]
        assert "matching_entries" in evidence
        assert len(evidence["matching_entries"]) > 0
        for item in evidence["matching_entries"]:
            assert item["id"]
            assert item["title"]
            assert item["date"]
            assert item["excerpt"]

    # Check emotional trends evidence
    for trend in result["emotional_trends"]:
        evidence = trend["evidence"]
        assert "matching_entries" in evidence
        assert len(evidence["matching_entries"]) > 0
        for item in evidence["matching_entries"]:
            assert item["id"]
            assert item["title"]
            assert item["date"]
            assert item["excerpt"]

    # Check recurring thoughts evidence
    for thought in result["recurring_thoughts"]:
        evidence = thought["evidence"]
        assert "matching_entries" in evidence
        assert len(evidence["matching_entries"]) > 0
        for item in evidence["matching_entries"]:
            assert item["id"]
            assert item["title"]
            assert item["date"]
            assert item["excerpt"]

    # Check changes over time evidence
    for change in result["changes_over_time"]:
        evidence = change["evidence"]
        if "earlier_matching" in evidence:
            for item in evidence["earlier_matching"]:
                assert item["id"]
                assert item["title"]
                assert item["excerpt"]

    # Check personal observations evidence
    for obs in result["personal_observations"]:
        evidence = obs["evidence"]
        assert "matching_entries" in evidence
        assert len(evidence["matching_entries"]) > 0
        for item in evidence["matching_entries"]:
            assert item["id"]
            assert item["title"]
            assert item["excerpt"]


def test_draft_entries_excluded():
    """Verify draft entries are strictly excluded from counts, percentages, and evidence."""
    synthesizer = get_pattern_synthesizer()
    entries = [
        {
            "id": "pub-1",
            "title": "Published Entry",
            "content": "This is a published reflection on personal habits.",
            "created_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
            "emotions": {"joy": 0.8},
            "theme_cluster_id": 2,
            "is_draft": False,
        },
        {
            "id": "draft-1",
            "title": "Draft Entry Secret",
            "content": "Secret draft text that should never be in patterns.",
            "created_at": datetime(2026, 1, 2, tzinfo=timezone.utc),
            "emotions": {"annoyance": 0.99},
            "theme_cluster_id": 0,
            "is_draft": True,
        },
    ]

    result = synthesizer.synthesize(entries, {"status": "success"})
    # Entry count must be 1, not 2
    assert result["overview"]["entry_count"] == 1
    # Draft content must not appear in any evidence
    result_str = str(result)
    assert "Secret draft text" not in result_str
    assert "draft-1" not in result_str


def test_insufficient_data_safeguard():
    """Verify empty or single entries produce no fabricated patterns."""
    synthesizer = get_pattern_synthesizer()
    result_empty = synthesizer.synthesize([], {"status": "insufficient_data"})
    assert result_empty["overview"]["entry_count"] == 0
    assert len(result_empty["recurring_topics"]) == 0
    assert len(result_empty["emotional_trends"]) == 0
    assert len(result_empty["recurring_thoughts"]) == 0
    assert len(result_empty["changes_over_time"]) == 0
    assert len(result_empty["personal_observations"]) == 0
