"""JournalEntry and EntryAnalysis SQLAlchemy database models."""

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class JournalEntry(Base):
    """User journal entry entity."""

    __tablename__ = "journal_entries"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    title = Column(String(255), default="", nullable=False)
    content = Column(Text, nullable=False)
    mood = Column(String(64), nullable=True)
    tags = Column(JSON, default=list, nullable=False)
    is_draft = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="entries")
    analysis = relationship("EntryAnalysis", back_populates="entry", uselist=False, cascade="all, delete-orphan")


class EntryAnalysis(Base):
    """ML analysis result entity computed for an individual journal entry."""

    __tablename__ = "entry_analyses"

    id = Column(Integer, primary_key=True, index=True)
    entry_id = Column(Integer, ForeignKey("journal_entries.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    emotions = Column(JSON, nullable=False)  # List of predicted emotions or all_scores dictionary
    top_emotion = Column(JSON, nullable=True)  # Top predicted emotion dict
    theme = Column(String(128), nullable=True)  # Name of closest semantic theme
    theme_cluster_id = Column(Integer, nullable=True)  # ID of theme cluster (0-4)
    linguistic_features = Column(JSON, nullable=True)  # 12 spaCy linguistic indicators

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    entry = relationship("JournalEntry", back_populates="analysis")
    user = relationship("User")
