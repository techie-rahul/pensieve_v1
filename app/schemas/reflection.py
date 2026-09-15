"""Pydantic schemas for reflections."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict


class ReflectionResponse(BaseModel):
    """Schema for returning a stored reflection."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    reflection_text: str
    grounded_concepts: List[Any]
    confidence: float
    disclaimer: str
    input_summary: Optional[Dict[str, Any]] = None
    created_at: datetime


class ReflectionSuggestResponse(BaseModel):
    """Schema for reflection suggestion generation."""

    status: str  # 'success', 'policy_rejected', 'validation_failed', 'service_unavailable'
    reflection: Optional[ReflectionResponse] = None
    reason: Optional[str] = None
    message: Optional[str] = None
    details: Optional[Any] = None
