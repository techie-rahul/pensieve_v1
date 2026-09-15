"""Pydantic schema for longitudinal patterns endpoint."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class PatternsResponse(BaseModel):
    """Schema for longitudinal patterns across user's chronological entries."""

    status: str
    entry_count: int
    time_span_days: Optional[int] = None
    history_window_type: Optional[str] = None
    num_windows: Optional[int] = None
    safeguards: Optional[Dict[str, Any]] = None
    linguistic_trends: Optional[Dict[str, Any]] = None
    emotion_trends: Optional[Dict[str, Any]] = None
    theme_trends: Optional[Dict[str, Any]] = None
    recurring_lexical_patterns: Optional[Dict[str, Any]] = None
    theme_emotion_associations: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
