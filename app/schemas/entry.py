"""Pydantic schemas for journal entries and entry analyses."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class EntryCreate(BaseModel):
    """Schema for creating a new journal entry."""

    title: Optional[str] = Field("", max_length=255)
    content: str = Field(..., min_length=1, description="Raw journal entry text")
    mood: Optional[str] = Field(None, max_length=64)
    tags: Optional[List[str]] = Field(default_factory=list)
    is_draft: bool = Field(False, description="Whether this entry is a draft")


class EntryUpdate(BaseModel):
    """Schema for updating an existing journal entry."""

    title: Optional[str] = Field(None, max_length=255)
    content: Optional[str] = Field(None, min_length=1)
    mood: Optional[str] = Field(None, max_length=64)
    tags: Optional[List[str]] = None
    is_draft: Optional[bool] = None


class EntryAutosave(BaseModel):
    """Schema for autosaving entry drafts."""

    id: Optional[int] = Field(None, description="Existing entry ID if updating, or null to create new draft")
    entry_id: Optional[int] = Field(None, description="Alias for id")
    title: Optional[str] = Field("", max_length=255)
    content: str = Field(..., description="Draft text content")
    mood: Optional[str] = Field(None, max_length=64)
    tags: Optional[List[str]] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Schema for returning entry-level ML analysis."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    entry_id: int
    emotions: Any
    top_emotion: Optional[Any] = None
    theme: Optional[str] = None
    theme_cluster_id: Optional[int] = None
    linguistic_features: Optional[Any] = None
    created_at: datetime


class EntryResponse(BaseModel):
    """Schema for returning a full journal entry."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    title: str
    content: str
    mood: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_draft: bool
    created_at: datetime
    updated_at: datetime
    analysis: Optional[AnalysisResponse] = None
