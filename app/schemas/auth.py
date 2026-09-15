"""Pydantic schemas for authentication and user accounts."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRegister(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    password: str = Field(..., min_length=6, description="Plain text password (minimum 6 characters)")
    name: Optional[str] = Field(None, max_length=255)


class UserLogin(BaseModel):
    """Schema for user login credentials."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for returning user profile."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    name: Optional[str] = None
    created_at: datetime


class Token(BaseModel):
    """JWT bearer token response schema."""

    access_token: str
    token_type: str = "bearer"


class TokenPayload(BaseModel):
    """Payload decoded from valid JWT token."""

    sub: str
    exp: Optional[int] = None
