"""Tests for Pensieve journal entry CRUD, draft autosave, and multi-tenant isolation."""

import pytest
from fastapi.testclient import TestClient

from app.models.user import User


def test_create_entry(client: TestClient, auth_headers: dict):
    """Test creating a normal journal entry."""
    payload = {
        "title": "First Reflection",
        "content": "Today was a productive day reflecting on systems design.",
        "tags": ["work", "design"],
        "mood": "focused",
        "is_draft": False,
    }
    response = client.post("/api/entries", json=payload, headers=auth_headers)
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == payload["title"]
    assert data["content"] == payload["content"]
    assert data["tags"] == payload["tags"]
    assert data["mood"] == "focused"
    assert data["is_draft"] is False
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data


def test_list_entries_and_filtering(client: TestClient, auth_headers: dict):
    """Test listing entries and filtering by draft status."""
    # Create one published entry and one draft
    client.post(
        "/api/entries",
        json={"title": "Published Entry", "content": "Content 1", "is_draft": False},
        headers=auth_headers,
    )
    client.post(
        "/api/entries",
        json={"title": "Draft Entry", "content": "Draft Content", "is_draft": True},
        headers=auth_headers,
    )

    # List all entries
    all_resp = client.get("/api/entries", headers=auth_headers)
    assert all_resp.status_code == 200
    assert len(all_resp.json()) == 2

    # Filter published only
    published_resp = client.get("/api/entries?is_draft=false", headers=auth_headers)
    assert published_resp.status_code == 200
    pub_items = published_resp.json()
    assert len(pub_items) == 1
    assert pub_items[0]["title"] == "Published Entry"

    # Filter draft only
    drafts_resp = client.get("/api/entries?is_draft=true", headers=auth_headers)
    assert drafts_resp.status_code == 200
    draft_items = drafts_resp.json()
    assert len(draft_items) == 1
    assert draft_items[0]["title"] == "Draft Entry"


def test_get_entry_by_id(client: TestClient, auth_headers: dict):
    """Test fetching a single entry by ID."""
    create_resp = client.post(
        "/api/entries",
        json={"title": "Specific Entry", "content": "Specific entry body."},
        headers=auth_headers,
    )
    entry_id = create_resp.json()["id"]

    get_resp = client.get(f"/api/entries/{entry_id}", headers=auth_headers)
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == entry_id
    assert get_resp.json()["title"] == "Specific Entry"


def test_update_entry(client: TestClient, auth_headers: dict):
    """Test updating an existing entry."""
    create_resp = client.post(
        "/api/entries",
        json={"title": "Initial Title", "content": "Initial content."},
        headers=auth_headers,
    )
    entry_id = create_resp.json()["id"]

    update_payload = {
        "title": "Updated Title",
        "content": "Updated content with deeper thoughts.",
        "tags": ["updated"],
        "mood": "calm",
        "is_draft": False,
    }
    update_resp = client.put(f"/api/entries/{entry_id}", json=update_payload, headers=auth_headers)
    assert update_resp.status_code == 200
    updated_data = update_resp.json()
    assert updated_data["title"] == "Updated Title"
    assert updated_data["content"] == "Updated content with deeper thoughts."
    assert updated_data["tags"] == ["updated"]
    assert updated_data["mood"] == "calm"


def test_delete_entry(client: TestClient, auth_headers: dict):
    """Test deleting an entry."""
    create_resp = client.post(
        "/api/entries",
        json={"title": "To Delete", "content": "Will be deleted."},
        headers=auth_headers,
    )
    entry_id = create_resp.json()["id"]

    delete_resp = client.delete(f"/api/entries/{entry_id}", headers=auth_headers)
    assert delete_resp.status_code == 204

    # Verify subsequent GET returns 404
    get_resp = client.get(f"/api/entries/{entry_id}", headers=auth_headers)
    assert get_resp.status_code == 404


def test_autosave_new_draft(client: TestClient, auth_headers: dict):
    """Test autosave creating a brand new draft."""
    autosave_payload = {
        "title": "Autosaved Draft",
        "content": "Incomplete thought being typed...",
    }
    response = client.post("/api/entries/autosave", json=autosave_payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["title"] == "Autosaved Draft"
    assert data["content"] == "Incomplete thought being typed..."
    assert data["is_draft"] is True
    assert "id" in data


def test_autosave_existing_draft(client: TestClient, auth_headers: dict):
    """Test autosave updating an existing draft without creating duplicate records."""
    # First save
    save_1 = client.post(
        "/api/entries/autosave",
        json={"title": "Draft 1", "content": "Initial typing"},
        headers=auth_headers,
    )
    entry_id = save_1.json()["id"]

    # Second save with entry_id
    save_2 = client.post(
        "/api/entries/autosave",
        json={"entry_id": entry_id, "title": "Draft 1 Updated", "content": "Initial typing continues..."},
        headers=auth_headers,
    )
    assert save_2.status_code == 200
    assert save_2.json()["id"] == entry_id
    assert save_2.json()["title"] == "Draft 1 Updated"
    assert save_2.json()["content"] == "Initial typing continues..."

    # Ensure only 1 entry exists
    list_resp = client.get("/api/entries", headers=auth_headers)
    assert len(list_resp.json()) == 1


def test_multitenant_isolation(
    client: TestClient,
    auth_headers: dict,
    auth_headers_user_2: dict,
):
    """Test that User 2 cannot access, update, or delete User 1's entries."""
    # User 1 creates an entry
    create_resp = client.post(
        "/api/entries",
        json={"title": "User 1 Secret", "content": "Private journal content."},
        headers=auth_headers,
    )
    entry_id = create_resp.json()["id"]

    # User 2 tries to GET User 1's entry -> 404
    get_resp = client.get(f"/api/entries/{entry_id}", headers=auth_headers_user_2)
    assert get_resp.status_code == 404

    # User 2 tries to PUT User 1's entry -> 404
    put_resp = client.put(
        f"/api/entries/{entry_id}",
        json={"title": "Hacked Title", "content": "Malicious edit"},
        headers=auth_headers_user_2,
    )
    assert put_resp.status_code == 404

    # User 2 tries to autosave over User 1's entry -> 404
    autosave_resp = client.post(
        "/api/entries/autosave",
        json={"entry_id": entry_id, "title": "Hijacked Draft", "content": "Hijack attempt"},
        headers=auth_headers_user_2,
    )
    assert autosave_resp.status_code == 404

    # User 2 lists entries -> should see 0 entries
    list_resp = client.get("/api/entries", headers=auth_headers_user_2)
    assert len(list_resp.json()) == 0

    # User 2 tries to DELETE User 1's entry -> 404
    del_resp = client.delete(f"/api/entries/{entry_id}", headers=auth_headers_user_2)
    assert del_resp.status_code == 404

    # User 1 should still be able to retrieve their entry
    verify_resp = client.get(f"/api/entries/{entry_id}", headers=auth_headers)
    assert verify_resp.status_code == 200
    assert verify_resp.json()["title"] == "User 1 Secret"
