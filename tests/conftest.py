"""Pytest test fixtures and configuration for Pensieve backend tests."""

import os
from typing import Dict, Generator
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.auth.security import create_access_token, hash_password
from app.database import Base, get_db
from app.main import app
from app.models.user import User

# In-memory SQLite test database with StaticPool to share connection across threads
TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Provide an isolated database session per test with clean tables."""
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """TestClient with database dependency overridden to test in-memory SQLite."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture(scope="function")
def test_user(db_session: Session) -> User:
    """Create and return a standard test user."""
    user = User(
        email="testuser@example.com",
        hashed_password=hash_password("Password123!"),
        name="Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def test_user_2(db_session: Session) -> User:
    """Create and return a second test user for ownership/isolation testing."""
    user = User(
        email="otheruser@example.com",
        hashed_password=hash_password("OtherPassword123!"),
        name="Other User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def auth_headers(test_user: User) -> Dict[str, str]:
    """Return HTTP authorization headers for primary test user."""
    token = create_access_token(subject=str(test_user.id), extra_claims={"email": test_user.email})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="function")
def auth_headers_user_2(test_user_2: User) -> Dict[str, str]:
    """Return HTTP authorization headers for secondary test user."""
    token = create_access_token(subject=str(test_user_2.id), extra_claims={"email": test_user_2.email})
    return {"Authorization": f"Bearer {token}"}
