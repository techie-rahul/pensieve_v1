"""Pensieve ML - Emotion Detection Module.

Phase 1 implementation for privacy-first journaling system.
"""

from ml.emotion.inference import EmotionClassifier, predict_emotions
from ml.emotion.labels import EMOTION_LABELS, ID2LABEL, LABEL2ID

__all__ = [
    "EmotionClassifier",
    "predict_emotions",
    "EMOTION_LABELS",
    "ID2LABEL",
    "LABEL2ID",
]
