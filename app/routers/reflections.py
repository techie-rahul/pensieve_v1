"""Reflections API endpoints."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.models.entry import JournalEntry
from app.models.reflection import Reflection
from app.models.user import User
from app.schemas.reflection import (
    ReflectionResponse,
    ReflectionSuggestResponse,
)
from app.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/reflections", tags=["Reflections"])


@router.post(
    "/suggest",
    response_model=ReflectionSuggestResponse,
    summary="Generate a grounded psychological/philosophical reflection",
)
def suggest_reflection(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service),
) -> ReflectionSuggestResponse:
    """Generate a reflective response from longitudinal signals and retrieved concepts."""
    # 1. Fetch user's non-draft entries
    entries = (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == current_user.id, JournalEntry.is_draft == False)
        .order_by(JournalEntry.created_at.asc())
        .all()
    )

    # 2. Fetch timestamps of user's past reflections for rate limiting
    past_reflections = (
        db.query(Reflection.created_at)
        .filter(Reflection.user_id == current_user.id)
        .order_by(Reflection.created_at.desc())
        .all()
    )
    past_timestamps = [r[0].isoformat() for r in past_reflections if r[0]]

    # 3. Execute generation pipeline
    gen_result = ml_service.generate_reflection(
        entries=entries,
        past_reflection_timestamps=past_timestamps,
    )

    if gen_result["status"] == "success":
        # Persist reflection
        reflection = Reflection(
            user_id=current_user.id,
            reflection_text=gen_result["reflection"],
            grounded_concepts=gen_result["grounded_concepts"],
            confidence=gen_result["confidence"],
            disclaimer=gen_result["disclaimer"],
            input_summary=gen_result.get("audit"),
        )
        db.add(reflection)
        db.commit()
        db.refresh(reflection)

        return ReflectionSuggestResponse(
            status="success",
            reflection=reflection,
        )

    # Return structured rejection or failure
    return ReflectionSuggestResponse(
        status=gen_result["status"],
        reason=gen_result.get("reason"),
        message=gen_result.get("message"),
        details=gen_result.get("details") or gen_result.get("validation_errors"),
    )


@router.get(
    "",
    response_model=List[ReflectionResponse],
    summary="List all reflections for the authenticated user",
)
def list_reflections(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[ReflectionResponse]:
    """Retrieve all reflections generated for the current user, ordered by date descending."""
    reflections = (
        db.query(Reflection)
        .filter(Reflection.user_id == current_user.id)
        .order_by(Reflection.created_at.desc())
        .all()
    )
    return reflections


@router.get(
    "/{reflection_id}",
    response_model=ReflectionResponse,
    summary="Get single reflection by ID",
)
def get_reflection(
    reflection_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> ReflectionResponse:
    """Retrieve a single reflection by ID, enforcing user ownership."""
    reflection = (
        db.query(Reflection)
        .filter(Reflection.id == reflection_id, Reflection.user_id == current_user.id)
        .first()
    )
    if not reflection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Reflection not found.",
        )
    return reflection
