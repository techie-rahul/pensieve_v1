"""Journal entries API endpoints with strict ownership validation."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session, joinedload

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.models.entry import JournalEntry
from app.models.user import User
from app.schemas.entry import (
    EntryAutosave,
    EntryCreate,
    EntryResponse,
    EntryUpdate,
)

router = APIRouter(prefix="/entries", tags=["Journal Entries"])


@router.get(
    "",
    response_model=List[EntryResponse],
    summary="List journal entries for current user",
)
def list_entries(
    is_draft: Optional[bool] = Query(None, description="Filter by draft status"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[EntryResponse]:
    """Retrieve all journal entries owned by the authenticated user, ordered by creation date descending."""
    query = (
        db.query(JournalEntry)
        .options(joinedload(JournalEntry.analysis))
        .filter(JournalEntry.user_id == current_user.id)
    )

    if is_draft is not None:
        query = query.filter(JournalEntry.is_draft == is_draft)

    entries = query.order_by(JournalEntry.created_at.desc()).offset(offset).limit(limit).all()
    return entries


@router.post(
    "",
    response_model=EntryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new journal entry",
)
def create_entry(
    entry_in: EntryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EntryResponse:
    """Create a new journal entry for the authenticated user."""
    entry = JournalEntry(
        user_id=current_user.id,
        title=entry_in.title.strip() if entry_in.title else "",
        content=entry_in.content.strip(),
        mood=entry_in.mood.strip() if entry_in.mood else None,
        tags=entry_in.tags or [],
        is_draft=entry_in.is_draft,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get(
    "/{entry_id}",
    response_model=EntryResponse,
    summary="Get single journal entry by ID",
)
def get_entry(
    entry_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EntryResponse:
    """Get an existing journal entry. Enforces user ownership."""
    entry = (
        db.query(JournalEntry)
        .options(joinedload(JournalEntry.analysis))
        .filter(JournalEntry.id == entry_id, JournalEntry.user_id == current_user.id)
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found.",
        )
    return entry


@router.put(
    "/{entry_id}",
    response_model=EntryResponse,
    summary="Update a journal entry",
)
def update_entry(
    entry_id: int,
    entry_in: EntryUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EntryResponse:
    """Update an existing journal entry owned by the authenticated user."""
    entry = (
        db.query(JournalEntry)
        .filter(JournalEntry.id == entry_id, JournalEntry.user_id == current_user.id)
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found.",
        )

    if entry_in.title is not None:
        entry.title = entry_in.title.strip()
    if entry_in.content is not None:
        entry.content = entry_in.content.strip()
    if entry_in.mood is not None:
        entry.mood = entry_in.mood.strip() if entry_in.mood else None
    if entry_in.tags is not None:
        entry.tags = entry_in.tags
    if entry_in.is_draft is not None:
        entry.is_draft = entry_in.is_draft

    db.commit()
    db.refresh(entry)
    return entry


@router.delete(
    "/{entry_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a journal entry",
)
def delete_entry(
    entry_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Permanently delete a journal entry and its associated analysis."""
    entry = (
        db.query(JournalEntry)
        .filter(JournalEntry.id == entry_id, JournalEntry.user_id == current_user.id)
        .first()
    )
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Journal entry not found.",
        )

    db.delete(entry)
    db.commit()
    return None


@router.post(
    "/autosave",
    response_model=EntryResponse,
    summary="Autosave draft journal entry",
)
def autosave_entry(
    draft_in: EntryAutosave,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> EntryResponse:
    """Autosave a journal draft. Updates if entry ID provided; creates new draft if not."""
    target_id = draft_in.id or draft_in.entry_id
    if target_id is not None:
        entry = (
            db.query(JournalEntry)
            .filter(JournalEntry.id == target_id, JournalEntry.user_id == current_user.id)
            .first()
        )
        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Journal entry not found to autosave.",
            )
        entry.title = draft_in.title.strip() if draft_in.title else ""
        entry.content = draft_in.content
        entry.mood = draft_in.mood
        entry.tags = draft_in.tags or []
        entry.is_draft = True
    else:
        entry = JournalEntry(
            user_id=current_user.id,
            title=draft_in.title.strip() if draft_in.title else "",
            content=draft_in.content,
            mood=draft_in.mood,
            tags=draft_in.tags or [],
            is_draft=True,
        )
        db.add(entry)

    db.commit()
    db.refresh(entry)
    return entry
