"""Tests for single-entry ML analysis (Phases 1-3) and longitudinal patterns."""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.entry import JournalEntry


def test_analyze_entry_success(client: TestClient, auth_headers: dict):
    """Test running Phase 1-3 ML analysis on a single journal entry."""
    # Create an entry
    create_resp = client.post(
        "/api/entries",
        json={
            "title": "Work and Stress",
            "content": "I feel completely overwhelmed by the constant deadlines and lack of control over my schedule.",
            "mood": "stressed",
        },
        headers=auth_headers,
    )
    assert create_resp.status_code == 201
    entry_id = create_resp.json()["id"]

    # Analyze the entry
    analyze_resp = client.post(f"/api/analyze/{entry_id}", headers=auth_headers)
    assert analyze_resp.status_code == 200
    data = analyze_resp.json()

    assert data["entry_id"] == entry_id
    assert "emotions" in data and isinstance(data["emotions"], dict)
    assert len(data["emotions"]) > 0
    assert "top_emotion" in data
    assert "theme" in data
    assert "linguistic_features" in data
    assert "first_person_pronoun_count" in data["linguistic_features"]

    # Verify fetching entry via GET now has nested analysis
    entry_resp = client.get(f"/api/entries/{entry_id}", headers=auth_headers)
    assert entry_resp.status_code == 200
    entry_data = entry_resp.json()
    assert entry_data["analysis"] is not None
    assert entry_data["analysis"]["id"] == data["id"]


def test_analyze_entry_empty_content(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test analyzing an entry with empty text returns 400."""
    # Insert entry with whitespace/empty content directly or via draft
    autosave_resp = client.post(
        "/api/entries/autosave",
        json={"title": "Empty Entry", "content": "   "},
        headers=auth_headers,
    )
    entry_id = autosave_resp.json()["id"]

    response = client.post(f"/api/analyze/{entry_id}", headers=auth_headers)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_analyze_entry_multitenant_isolation(
    client: TestClient,
    auth_headers: dict,
    auth_headers_user_2: dict,
):
    """Test User 2 cannot trigger analysis on User 1's entry."""
    create_resp = client.post(
        "/api/entries",
        json={"title": "Private Entry", "content": "This is user 1's private journal."},
        headers=auth_headers,
    )
    entry_id = create_resp.json()["id"]

    response = client.post(f"/api/analyze/{entry_id}", headers=auth_headers_user_2)
    assert response.status_code == 404


def test_patterns_empty_history(client: TestClient, auth_headers: dict):
    """Test /api/patterns with no entries returns insufficient_data status."""
    response = client.get("/api/patterns", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "insufficient_data"
    assert data["entry_count"] == 0
    assert "message" in data


def test_patterns_with_entries(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test /api/patterns with entries across multiple days."""
    now = datetime.now(timezone.utc)

    # Seed 4 entries separated by several days
    entries_data = [
        ("I had a very calm and restful morning reflecting on my personal goals.", now - timedelta(days=12)),
        ("Work was busy today and I felt a bit stressed about the upcoming review.", now - timedelta(days=8)),
        ("Had a great conversation with a colleague and felt encouraged about our project.", now - timedelta(days=4)),
        ("Finished the project milestone and felt very satisfied with the outcome.", now),
    ]

    for content, dt in entries_data:
        entry = JournalEntry(
            user_id=test_user.id,
            title="Log",
            content=content,
            is_draft=False,
            created_at=dt,
        )
        db_session.add(entry)
    db_session.commit()

    response = client.get("/api/patterns", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["entry_count"] >= 4
    assert data["status"] in ("success", "ok", "sparse_history", "insufficient_data")
    assert "safeguards" in data


def test_patterns_with_fractional_time_span_days(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test /api/patterns correctly handles and returns non-integer fractional time_span_days."""
    now = datetime.now(timezone.utc)

    # Seed entries separated by fractional days (hours/minutes offset)
    entries_data = [
        ("Early morning reflection on priorities.", now - timedelta(days=14, hours=10)),
        ("Midday check-in on project status.", now - timedelta(days=9, hours=4)),
        ("Evening thoughts after a long walk.", now - timedelta(days=4, hours=12)),
        ("Current reflection on our architecture and progress.", now),
    ]

    for content, dt in entries_data:
        entry = JournalEntry(
            user_id=test_user.id,
            title="Fractional Span Entry",
            content=content,
            is_draft=False,
            created_at=dt,
        )
        db_session.add(entry)
    db_session.commit()

    response = client.get("/api/patterns", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["entry_count"] >= 4
    assert isinstance(data["time_span_days"], (int, float))
    assert data["time_span_days"] > 14.0


def test_patterns_dynamically_includes_new_published_entry(
    client: TestClient,
    auth_headers: dict,
    db_session: Session,
    test_user,
):
    """Regression test: verify that adding and analyzing a new published entry increments

    the entry count reported by GET /api/patterns dynamically without stale caching.
    """
    now = datetime.now(timezone.utc)

    # 1. Seed 4 baseline published entries across 20 days
    base_entries = [
        ("Early baseline observation of morning routines.", now - timedelta(days=20)),
        ("Midweek reflection on work challenges and focus.", now - timedelta(days=15)),
        ("Weekend notes on reading and quiet moments.", now - timedelta(days=8)),
        ("Followup entry on team progress and calm reflection.", now - timedelta(days=2)),
    ]
    for content, dt in base_entries:
        entry = JournalEntry(
            user_id=test_user.id,
            title="Baseline Entry",
            content=content,
            is_draft=False,
            created_at=dt,
        )
        db_session.add(entry)
    db_session.commit()

    # 2. Verify baseline pattern count is 4
    resp1 = client.get("/api/patterns", headers=auth_headers)
    assert resp1.status_code == 200
    data1 = resp1.json()
    assert data1["status"] == "success"
    assert data1["entry_count"] == 4

    # 3. Simulate user workflow: Autosave creates a draft entry
    autosave_resp = client.post(
        "/api/entries/autosave",
        json={
            "title": "A New Milestone",
            "content": "Today we completed the major refactoring milestone and things are feeling solid.",
        },
        headers=auth_headers,
    )
    assert autosave_resp.status_code == 200
    new_entry_id = autosave_resp.json()["id"]

    # 4. Before analysis/publish, draft must NOT be included in patterns
    resp_draft = client.get("/api/patterns", headers=auth_headers)
    assert resp_draft.status_code == 200
    assert resp_draft.json()["entry_count"] == 4

    # 5. User saves as published and analyzes entry
    put_resp = client.put(
        f"/api/entries/{new_entry_id}",
        json={
            "title": "A New Milestone",
            "content": "Today we completed the major refactoring milestone and things are feeling solid.",
            "is_draft": False,
        },
        headers=auth_headers,
    )
    assert put_resp.status_code == 200
    assert put_resp.json()["is_draft"] is False

    analyze_resp = client.post(f"/api/analyze/{new_entry_id}", headers=auth_headers)
    assert analyze_resp.status_code == 200

    # 6. Subsequent autosave must NOT revert is_draft to True on an already-published entry
    autosave_again = client.post(
        "/api/entries/autosave",
        json={
            "id": new_entry_id,
            "title": "A New Milestone",
            "content": "Today we completed the major refactoring milestone and things are feeling solid.",
        },
        headers=auth_headers,
    )
    assert autosave_again.status_code == 200
    assert autosave_again.json()["is_draft"] is False

    # 7. GET /api/patterns must now reflect 5 entries
    resp2 = client.get("/api/patterns", headers=auth_headers)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "success"
    assert data2["entry_count"] == 5, f"Expected 5 entries in patterns, got {data2['entry_count']}"


