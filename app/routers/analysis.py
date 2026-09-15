"""Analysis and longitudinal patterns API endpoints."""

import logging
from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.models.entry import EntryAnalysis, JournalEntry
from app.models.user import User
from app.schemas.entry import AnalysisResponse
from app.schemas.patterns import PatternsResponse
from app.services.ml_service import MLService, get_ml_service

logger = logging.getLogger("pensieve.analysis")

router = APIRouter(tags=["ML Analysis & Patterns"])


@router.post(
    "/analyze/{entry_id}",
    response_model=AnalysisResponse,
    summary="Run Phase 1–3 ML analysis on a single journal entry",
)
def analyze_entry(
    entry_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service),
) -> AnalysisResponse:
    """Execute RoBERTa emotion detection, Sentence-BERT theme alignment, and spaCy linguistic analysis."""
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

    if not entry.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot analyze empty journal entry.",
        )

    # Execute ML pipeline Phases 1-3
    analysis_data = ml_service.analyze_single_entry(entry.content)

    # Persist or update EntryAnalysis record
    analysis = db.query(EntryAnalysis).filter(EntryAnalysis.entry_id == entry.id).first()
    if not analysis:
        analysis = EntryAnalysis(
            entry_id=entry.id,
            user_id=current_user.id,
            emotions=analysis_data["emotions"],
            top_emotion=analysis_data["top_emotion"],
            theme=analysis_data["theme"],
            theme_cluster_id=analysis_data["theme_cluster_id"],
            linguistic_features=analysis_data["linguistic_features"],
        )
        db.add(analysis)
    else:
        analysis.emotions = analysis_data["emotions"]
        analysis.top_emotion = analysis_data["top_emotion"]
        analysis.theme = analysis_data["theme"]
        analysis.theme_cluster_id = analysis_data["theme_cluster_id"]
        analysis.linguistic_features = analysis_data["linguistic_features"]

    entry.is_draft = False
    db.commit()
    db.refresh(analysis)
    return analysis


@router.get(
    "/patterns",
    response_model=PatternsResponse,
    summary="Compute longitudinal patterns across user's chronological entries",
)
def get_patterns(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service),
) -> PatternsResponse:
    """Aggregate chronological journal history to detect emotion shifts, theme trajectories, and recurring lexical patterns."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    # Retrieve non-draft entries for current user in chronological order
    entries = (
        db.query(JournalEntry)
        .filter(JournalEntry.user_id == current_user.id, JournalEntry.is_draft == False)
        .order_by(JournalEntry.created_at.asc())
        .all()
    )

    if not entries:
        return PatternsResponse(
            status="insufficient_data",
            entry_count=0,
            message="No published journal entries found. At least 3 entries spanning 7+ days are required for longitudinal analysis.",
        )

    entry_ids = [e.id for e in entries]
    earliest = entries[0].created_at if entries else None
    latest = entries[-1].created_at if entries else None
    logger.info(
        f"compute_longitudinal_patterns called: user_id={current_user.id}, "
        f"count={len(entries)}, entry_ids={entry_ids}, earliest={earliest}, latest={latest}"
    )

    report = ml_service.compute_longitudinal_patterns(entries)
    time_span_days = report.get("time_span_days")
    if time_span_days is None and report.get("safeguards"):
        time_span_days = report.get("safeguards", {}).get("timespan_days")

    return PatternsResponse(
        status=report.get("status", "unknown"),
        entry_count=report.get("entry_count", len(entries)),
        time_span_days=time_span_days,
        history_window_type=report.get("history_window_type"),
        num_windows=report.get("num_windows"),
        safeguards=report.get("safeguards"),
        linguistic_trends=report.get("linguistic_trends"),
        emotion_trends=report.get("emotion_trends"),
        theme_trends=report.get("theme_trends"),
        recurring_lexical_patterns=report.get("recurring_lexical_patterns"),
        theme_emotion_associations=report.get("theme_emotion_associations"),
        message=report.get("message"),
    )
