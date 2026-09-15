"""Tests for Pensieve authentication endpoints."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.user import User


def test_register_success(client: TestClient):
    """Test successful user registration."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "newuser@example.com",
            "password": "SecurePassword123!",
            "name": "New User",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "newuser@example.com"
    assert data["name"] == "New User"
    assert "id" in data
    assert "created_at" in data
    assert "hashed_password" not in data
    assert "password" not in data


def test_register_duplicate_email(client: TestClient, test_user: User):
    """Test registering with an already existing email returns 400."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": test_user.email,
            "password": "AnotherPassword123!",
            "name": "Duplicate User",
        },
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


def test_register_invalid_email(client: TestClient):
    """Test registering with an invalid email returns 422."""
    response = client.post(
        "/api/auth/register",
        json={
            "email": "not-an-email",
            "password": "ValidPassword123!",
            "name": "Invalid Email",
        },
    )
    assert response.status_code == 422


def test_login_success(client: TestClient, test_user: User):
    """Test successful login returns access token."""
    response = client.post(
        "/api/auth/login",
        json={
            "email": test_user.email,
            "password": "Password123!",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert len(data["access_token"]) > 20


def test_login_wrong_password(client: TestClient, test_user: User):
    """Test login with incorrect password returns 401."""
    response = client.post(
        "/api/auth/login",
        json={
            "email": test_user.email,
            "password": "WrongPassword999!",
        },
    )
    assert response.status_code == 401
    assert "invalid email or password" in response.json()["detail"].lower()


def test_login_nonexistent_email(client: TestClient):
    """Test login with an unknown email returns 401."""
    response = client.post(
        "/api/auth/login",
        json={
            "email": "doesnotexist@example.com",
            "password": "Password123!",
        },
    )
    assert response.status_code == 401
    assert "invalid email or password" in response.json()["detail"].lower()


def test_get_me_success(client: TestClient, auth_headers: dict, test_user: User):
    """Test GET /api/auth/me with valid bearer token returns user profile."""
    response = client.get("/api/auth/me", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == test_user.id
    assert data["email"] == test_user.email
    assert data["name"] == test_user.name


def test_get_me_unauthorized(client: TestClient):
    """Test GET /api/auth/me without authorization token returns 401."""
    response = client.get("/api/auth/me")
    assert response.status_code == 401


def test_get_me_invalid_token(client: TestClient):
    """Test GET /api/auth/me with an invalid token returns 401."""
    response = client.get("/api/auth/me", headers={"Authorization": "Bearer invalid_garbage_token"})
    assert response.status_code == 401
