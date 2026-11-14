"""Pydantic schema for longitudinal patterns endpoint."""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel


class PatternsResponse(BaseModel):
    """Schema for longitudinal patterns across user's chronological entries.

    Maintains full backwards compatibility with Phase 1–5 legacy fields while providing
    the 5 evidence-based personalized insight categories:
    1. Recurring Topics
    2. Emotional Trends
    3. Recurring Thoughts & Language
    4. Changes Over Time
    5. Personal Observations
    """

    status: str
    entry_count: int
    time_span_days: Optional[Union[float, int]] = None
    history_window_type: Optional[str] = None
    num_windows: Optional[int] = None
    safeguards: Optional[Dict[str, Any]] = None

    # Legacy pipeline outputs (preserved for backwards-compatibility)
    linguistic_trends: Optional[Dict[str, Any]] = None
    emotion_trends: Optional[Dict[str, Any]] = None
    theme_trends: Optional[Dict[str, Any]] = None
    recurring_lexical_patterns: Optional[Dict[str, Any]] = None
    theme_emotion_associations: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

    # Personalized, evidence-based pattern categories
    overview: Optional[Dict[str, Any]] = None
    recurring_topics: Optional[List[Dict[str, Any]]] = None
    emotional_trends: Optional[List[Dict[str, Any]]] = None
    recurring_thoughts: Optional[List[Dict[str, Any]]] = None
    changes_over_time: Optional[List[Dict[str, Any]]] = None
    personal_observations: Optional[List[Dict[str, Any]]] = None
