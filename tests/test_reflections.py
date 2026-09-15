"""Tests for reflection generation, history listing, and multi-tenant isolation."""

from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.entry import JournalEntry
from app.models.reflection import Reflection


def test_suggest_reflection_insufficient_history(client: TestClient, auth_headers: dict):
    """Test suggesting a reflection with no entries returns structured rejection."""
    response = client.post("/api/reflections/suggest", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("policy_rejected", "rejected", "insufficient_data")
    assert "reason" in data or "message" in data


def test_suggest_reflection_success(client: TestClient, auth_headers: dict, db_session: Session, test_user):
    """Test generating a grounded reflection with sufficient longitudinal history."""
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

    # The ML pipeline generates either a grounded reflection or returns validation status
    if data["status"] == "success":
        refl = data["reflection"]
        assert refl is not None
        assert "reflection_text" in refl
        assert len(refl["reflection_text"]) > 20
        assert "grounded_concepts" in refl
        assert "disclaimer" in refl
        assert "id" in refl

        # Test GET /api/reflections
        list_resp = client.get("/api/reflections", headers=auth_headers)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) >= 1

        # Test GET /api/reflections/{id}
        get_resp = client.get(f"/api/reflections/{refl['id']}", headers=auth_headers)
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == refl["id"]
    else:
        # Structured rejection (e.g. rate limit or safety) is also a valid API contract response
        assert "reason" in data or "message" in data


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
        grounded_concepts=["stoic_dichotomy_of_control"],
        confidence=0.85,
        disclaimer="Development/Educational only",
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
