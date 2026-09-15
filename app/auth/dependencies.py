"""FastAPI authentication dependencies."""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.auth.security import decode_access_token
from app.database import get_db
from app.models.user import User

# HTTPBearer scheme handles 'Authorization: Bearer <token>'
http_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that extracts and validates JWT token, returning authenticated User.

    Raises:
        HTTPException(401): If token is missing, invalid, expired, or user not found.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not auth or not auth.credentials:
        raise credentials_exception

    token_payload = decode_access_token(auth.credentials)
    if not token_payload or not token_payload.sub:
        raise credentials_exception

    # User lookup by ID or Email
    user = None
    if token_payload.sub.isdigit():
        user = db.query(User).filter(User.id == int(token_payload.sub)).first()
    if not user:
        user = db.query(User).filter(User.email == token_payload.sub).first()

    if not user:
        raise credentials_exception

    return user
