"""Reflection SQLAlchemy database model."""

from datetime import datetime, timezone
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import relationship

from app.database import Base


class Reflection(Base):
    """Grounded psychological/philosophical reflection generated from longitudinal analysis."""

    __tablename__ = "reflections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    reflection_text = Column(Text, nullable=False)
    grounded_concepts = Column(JSON, nullable=False)  # List of {concept_id, name, source}
    confidence = Column(Float, nullable=False)  # Capped <= 0.80
    disclaimer = Column(Text, nullable=False)
    input_summary = Column(JSON, nullable=True)  # Context summary (entry_count, timespan)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="reflections")
