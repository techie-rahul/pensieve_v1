"""Tests for Knowledge Base concepts endpoints."""

import pytest
from fastapi.testclient import TestClient


def test_list_concepts_returns_20_development_concepts(client: TestClient):
    """Test listing all concepts from the 20-concept development knowledge base."""
    response = client.get("/api/concepts")
    assert response.status_code == 200
    concepts = response.json()
    assert len(concepts) == 20
    for c in concepts:
        assert "id" in c
        assert "name" in c
        assert "category" in c
        assert "definition" in c


def test_list_concepts_filter_by_category(client: TestClient):
    """Test filtering concepts by category."""
    response = client.get("/api/concepts?category=philosophical")
    assert response.status_code == 200
    concepts = response.json()
    assert len(concepts) > 0
    for c in concepts:
        assert c["category"].lower() == "philosophical"


def test_list_concepts_search(client: TestClient):
    """Test searching concepts by keyword."""
    response = client.get("/api/concepts?search=reframing")
    assert response.status_code == 200
    concepts = response.json()
    assert len(concepts) >= 1
    found_ids = [c["id"] for c in concepts]
    assert any("cognitive_reframing" in cid for cid in found_ids)


def test_get_concept_detail_success(client: TestClient):
    """Test fetching a single concept with full detail and citation."""
    # First get list to pick a valid id
    list_resp = client.get("/api/concepts")
    first_id = list_resp.json()[0]["id"]

    response = client.get(f"/api/concepts/{first_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == first_id
    assert "name" in data
    assert "category" in data
    assert "definition" in data
    assert "source" in data
    assert "explanation" in data
    assert "cautions" in data
    assert "related_patterns" in data


def test_get_concept_not_found(client: TestClient):
    """Test fetching a non-existent concept returns 404."""
    response = client.get("/api/concepts/non_existent_concept_12345")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()
