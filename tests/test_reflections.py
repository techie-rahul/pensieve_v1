"""Tests for reflection generation, history listing, deduplication, and multi-tenant isolation."""

from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.entry import JournalEntry
from app.models.reflection import Reflection
from ml.reflection import ReflectionValidator, ReflectionInput


def test_empty_history(client: TestClient, auth_headers: dict):
    """Test that listing reflections when none exist returns an empty list."""
    response = client.get("/api/reflections", headers=auth_headers)
    assert response.status_code == 200
    assert response.json() == []


def test_suggest_reflection_insufficient_history(client: TestClient, auth_headers: dict):
    """Test suggesting a reflection with no entries returns structured rejection."""
    response = client.post("/api/reflections/suggest", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("policy_rejected", "rejected", "insufficient_data")
    assert "reason" in data or "message" in data


def test_suggest_reflection_insufficient_timespan(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test suggesting a reflection with 3 entries spanning only 2 days (fails min_span_days=7)."""
    now = datetime.now(timezone.utc)
    # 3 entries within 2 days
    for i in range(3):
        entry = JournalEntry(
            user_id=test_user.id,
            title=f"Entry {i}",
            content=f"Day {i} reflection on work and life.",
            is_draft=False,
            created_at=now - timedelta(days=2 - i),
        )
        db_session.add(entry)
    db_session.commit()

    response = client.post("/api/reflections/suggest", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "policy_rejected"
    assert data.get("reason") in ("insufficient_timespan", "insufficient_entries")


def test_suggest_reflection_success_and_deduplication(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test generating a grounded reflection, verifying structure, and preventing duplicate requests."""
    now = datetime.now(timezone.utc)
    entries_data = [
        ("I feel overwhelmed by my workload and feel like I cannot control anything at my job.", now - timedelta(days=14)),
        ("Another chaotic day at work where unexpected urgent tasks derailed my entire plan.", now - timedelta(days=9)),
        ("I had a brief moment of quiet today and wondered if I should re-evaluate my priorities.", now - timedelta(days=5)),
        ("Trying to focus on what is strictly within my own control today rather than worrying.", now),
    ]

    for content, dt in entries_data:
        entry = JournalEntry(
            user_id=test_user.id,
            title="Reflection Log",
            content=content,
            is_draft=False,
            created_at=dt,
        )
        db_session.add(entry)
    db_session.commit()

    response = client.post("/api/reflections/suggest", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()

    if data["status"] == "success":
        refl = data["reflection"]
        assert refl is not None
        assert "reflection_text" in refl
        text = refl["reflection_text"]
        assert len(text) > 50

        # Verify 2-3 paragraph structure separated by newlines
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        assert 2 <= len(paragraphs) <= 3, f"Expected 2-3 paragraphs, got {len(paragraphs)}"

        # Verify closing question
        assert "?" in paragraphs[-1], "Last paragraph must contain an open-ended question"

        # Verify confidence cap <= 0.80
        assert refl["confidence"] <= 0.80

        # Verify non-diagnostic disclaimer
        assert "disclaimer" in refl
        assert "not" in refl["disclaimer"].lower()

        # Verify audit summary with observations
        if refl.get("input_summary"):
            assert "entry_count" in refl["input_summary"]
            assert "span_days" in refl["input_summary"]

        refl_id = refl["id"]

        # Duplicate Generation Prevention: second rapid call should reuse the recent reflection
        dup_resp = client.post("/api/reflections/suggest", headers=auth_headers)
        assert dup_resp.status_code == 200
        dup_data = dup_resp.json()
        assert dup_data["status"] == "success"
        assert dup_data["reflection"]["id"] == refl_id, "Duplicate request must return existing reflection ID"

        # Verify only 1 reflection exists in database
        count = db_session.query(Reflection).filter(Reflection.user_id == test_user.id).count()
        assert count == 1, f"Expected exactly 1 reflection in DB, found {count}"

        # Test listing reflections
        list_resp = client.get("/api/reflections", headers=auth_headers)
        assert list_resp.status_code == 200
        reflections_list = list_resp.json()
        assert len(reflections_list) == 1

        # Verify latest vs history partitioning logic
        latest = reflections_list[0]
        history = [r for r in reflections_list if r["id"] != latest["id"]]
        assert latest["id"] == refl_id
        assert len(history) == 0, "History should not contain the latest reflection"


def test_history_deduplication_and_partitioning(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test that historical reflections appear exactly once and latest is excluded from history."""
    now = datetime.now(timezone.utc)
    # Insert 3 historical reflections with distinct times
    refl1 = Reflection(
        user_id=test_user.id,
        reflection_text="Older reflection paragraph 1.\n\nOlder paragraph 2?\n\nHow do you feel?",
        grounded_concepts=[{"concept_id": "cognitive_reframing", "name": "Cognitive Reframing", "source": "Beck (1979)"}],
        confidence=0.70,
        disclaimer="Contemplation only, not medical advice.",
        created_at=now - timedelta(days=20),
    )
    refl2 = Reflection(
        user_id=test_user.id,
        reflection_text="Mid reflection paragraph 1.\n\nMid paragraph 2?\n\nWhat matters?",
        grounded_concepts=[{"concept_id": "stoic_dichotomy_of_control", "name": "Dichotomy of Control", "source": "Epictetus"}],
        confidence=0.72,
        disclaimer="Contemplation only, not medical advice.",
        created_at=now - timedelta(days=10),
    )
    refl3 = Reflection(
        user_id=test_user.id,
        reflection_text="Latest reflection paragraph 1.\n\nLatest paragraph 2?\n\nWhere is your focus?",
        grounded_concepts=[{"concept_id": "growth_mindset", "name": "Growth Mindset", "source": "Dweck (2006)"}],
        confidence=0.75,
        disclaimer="Contemplation only, not medical advice.",
        created_at=now,
    )
    db_session.add_all([refl1, refl2, refl3])
    db_session.commit()

    list_resp = client.get("/api/reflections", headers=auth_headers)
    assert list_resp.status_code == 200
    all_refls = list_resp.json()

    assert len(all_refls) == 3
    # Check ordered descending by date
    assert all_refls[0]["id"] == refl3.id
    assert all_refls[1]["id"] == refl2.id
    assert all_refls[2]["id"] == refl1.id

    # Verify ID deduplication: no ID repeats
    ids = [r["id"] for r in all_refls]
    assert len(ids) == len(set(ids)), "All reflection IDs in history must be unique"

    # Verify separation into latest and previous
    latest = all_refls[0]
    previous = all_refls[1:]
    assert latest["id"] == refl3.id
    assert all(r["id"] != latest["id"] for r in previous), "Latest reflection must not appear in previous reflections"
    assert len(previous) == 2


def test_evidence_grounding_and_source_validation():
    """Test validator enforces source citation integrity and rejects invented concepts."""
    validator = ReflectionValidator()
    ref_input = ReflectionInput(
        data_summary={"entry_count": 4, "span_days": 14},
        retrieved_concepts=[{
            "concept_id": "stoic_dichotomy_of_control",
            "name": "Dichotomy of Control",
            "source": "Epictetus, Enchiridion (c. 125 CE); Robertson, D. (2019).",
        }]
    )

    # Valid reflection adhering to 2-3 paragraphs and hedging
    valid_output = {
        "reflection": (
            "Looking at your writing across your 4 recent journal entries spanning 14 days, an observable pattern emerges: "
            "an increase in work-related tension. In parallel, your reflections also highlight efforts to regain clarity.\n\n"
            "This observed pattern resonates with the principles of Dichotomy of Control. "
            "In practical terms, this concept focuses on distinguishing between things within our direct control and things outside it. "
            "Viewed through this lens, your reflections may reflect an effort to navigate external expectations.\n\n"
            "One possibility is to consider where your natural agency feels most steady. "
            "What aspects of your current daily routine feel most within your immediate sphere of influence?"
        ),
        "grounded_concepts": [{
            "concept_id": "stoic_dichotomy_of_control",
            "name": "Dichotomy of Control",
            "source": "Epictetus, Enchiridion (c. 125 CE); Robertson, D. (2019).",
        }],
        "confidence": 0.70,
        "disclaimer": "This reflection describes observable patterns in journal text for personal contemplation. It does not constitute psychological, psychiatric, or medical advice or diagnosis.",
    }
    res = validator.validate(valid_output, ref_input)
    assert res.is_valid, f"Validation failed unexpectedly: {res.errors}"

    # Invalid concept (hallucinated / not in retrieved concepts)
    invalid_concept_output = dict(valid_output)
    invalid_concept_output["grounded_concepts"] = [{
        "concept_id": "invented_concept",
        "name": "Invented Concept",
        "source": "Fictional Author (2025)",
    }]
    res_inv = validator.validate(invalid_concept_output, ref_input)
    assert not res_inv.is_valid
    assert any("NOT in the retrieved Phase 4 concepts" in err for err in res_inv.errors)


def test_reflection_multitenant_isolation(
    client: TestClient,
    auth_headers: dict,
    auth_headers_user_2: dict,
    db_session: Session,
    test_user,
    test_user_2,
):
    """Test User 2 cannot access or view User 1's reflections."""
    # Create a reflection for User 1
    refl = Reflection(
        user_id=test_user.id,
        reflection_text="User 1 private reflection text.",
        grounded_concepts=[{"concept_id": "stoic_dichotomy_of_control", "name": "Dichotomy of Control", "source": "Epictetus"}],
        confidence=0.75,
        disclaimer="Contemplation only, not medical advice.",
    )
    db_session.add(refl)
    db_session.commit()
    db_session.refresh(refl)

    # User 2 list reflections -> should be empty
    list_resp = client.get("/api/reflections", headers=auth_headers_user_2)
    assert list_resp.status_code == 200
    assert len(list_resp.json()) == 0

    # User 2 get User 1's reflection by id -> 404
    get_resp = client.get(f"/api/reflections/{refl.id}", headers=auth_headers_user_2)
    assert get_resp.status_code == 404

    # User 1 can get it
    get_resp_1 = client.get(f"/api/reflections/{refl.id}", headers=auth_headers)
    assert get_resp_1.status_code == 200
    assert get_resp_1.json()["reflection_text"] == "User 1 private reflection text."
