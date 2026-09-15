"""Pydantic schemas for knowledge base concepts."""

from typing import List
from pydantic import BaseModel, ConfigDict


class ConceptSummaryResponse(BaseModel):
    """Brief concept schema for catalog listing."""

    id: str
    name: str
    category: str
    definition: str


class ConceptDetailResponse(BaseModel):
    """Full concept detail schema with sources and safety cautions."""

    id: str
    name: str
    category: str
    definition: str
    explanation: str
    related_patterns: List[str]
    cautions: List[str]
    source: str
