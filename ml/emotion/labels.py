"""GoEmotions taxonomy, label mappings, and category groupings.

The GoEmotions dataset contains 58k Reddit comments annotated with 27 fine-grained
emotions plus 'neutral' (28 total classes).
"""

from typing import Dict, List, Set
import numpy as np

# Official 28 GoEmotions labels ordered exactly as in the dataset & model config
EMOTION_LABELS: List[str] = [
    "admiration",      # 0
    "amusement",       # 1
    "anger",           # 2
    "annoyance",       # 3
    "approval",        # 4
    "caring",          # 5
    "confusion",       # 6
    "curiosity",       # 7
    "desire",          # 8
    "disappointment",  # 9
    "disapproval",     # 10
    "disgust",         # 11
    "embarrassment",   # 12
    "excitement",      # 13
    "fear",            # 14
    "gratitude",       # 15
    "grief",           # 16
    "joy",             # 17
    "love",            # 18
    "nervousness",     # 19
    "optimism",        # 20
    "pride",           # 21
    "realization",     # 22
    "relief",          # 23
    "remorse",         # 24
    "sadness",         # 25
    "surprise",        # 26
    "neutral",         # 27
]

NUM_LABELS: int = len(EMOTION_LABELS)

# Index and label bidirectional mappings
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(EMOTION_LABELS)}
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(EMOTION_LABELS)}

# High-level sentiment groupings based on GoEmotions paper (Demszky et al., ACL 2020)
POSITIVE_EMOTIONS: Set[str] = {
    "admiration",
    "amusement",
    "approval",
    "caring",
    "desire",
    "excitement",
    "gratitude",
    "joy",
    "love",
    "optimism",
    "pride",
    "relief",
}

NEGATIVE_EMOTIONS: Set[str] = {
    "anger",
    "annoyance",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "fear",
    "grief",
    "nervousness",
    "remorse",
    "sadness",
}

AMBIGUOUS_OR_COGNITIVE_EMOTIONS: Set[str] = {
    "confusion",
    "curiosity",
    "realization",
    "surprise",
}

NEUTRAL_EMOTIONS: Set[str] = {
    "neutral",
}


def labels_to_multihot(label_ids: List[int], num_classes: int = NUM_LABELS) -> np.ndarray:
    """Convert a list of integer class IDs into a binary multi-hot vector.

    Args:
        label_ids: List of integer class indices.
        num_classes: Total number of classes (default 28).

    Returns:
        1D numpy array of shape (num_classes,) with 1s at active positions.
    """
    vec = np.zeros(num_classes, dtype=np.float32)
    for idx in label_ids:
        if 0 <= idx < num_classes:
            vec[idx] = 1.0
    return vec


def multihot_to_labels(vec: np.ndarray) -> List[str]:
    """Convert a multi-hot binary vector back to a list of emotion names.

    Args:
        vec: 1D binary array or list of floats/ints.

    Returns:
        List of emotion name strings where vec[i] == 1.
    """
    return [ID2LABEL[i] for i, val in enumerate(vec) if val > 0.5]


def get_sentiment_group(emotion: str) -> str:
    """Return the high-level sentiment group ('positive', 'negative', 'ambiguous', 'neutral') for an emotion."""
    if emotion in POSITIVE_EMOTIONS:
        return "positive"
    if emotion in NEGATIVE_EMOTIONS:
        return "negative"
    if emotion in AMBIGUOUS_OR_COGNITIVE_EMOTIONS:
        return "ambiguous"
    return "neutral"
