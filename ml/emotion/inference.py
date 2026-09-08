"""Inference interface for emotion detection on arbitrary journal text.

Designed for modular integration into the Pensieve journaling backend.
Uses pretrained RoBERTa fine-tuned on GoEmotions.
"""

from typing import Any, Dict, List, Optional, Union
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ml.emotion.labels import EMOTION_LABELS, ID2LABEL, get_sentiment_group

DEFAULT_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
DEFAULT_THRESHOLD = 0.30  # Calibrated decision threshold on validation set


class EmotionClassifier:
    """Emotion classification engine using fine-tuned RoBERTa on GoEmotions."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        default_threshold: float = DEFAULT_THRESHOLD,
        max_length: int = 128,
    ) -> None:
        """Initialize tokenizer, model, and inference settings.

        Args:
            model_name_or_path: Hugging Face model checkpoint or local path.
            device: 'cuda', 'cpu', or None (auto-detects).
            default_threshold: Default probability threshold for multi-label assignment.
            max_length: Maximum tokenization sequence length.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.default_threshold = default_threshold
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        # Verify output dimension matches GoEmotions taxonomy (28 classes)
        self.num_labels = len(EMOTION_LABELS)
        assert self.model.config.num_labels == self.num_labels, (
            f"Model output dimension ({self.model.config.num_labels}) "
            f"does not match expected GoEmotions classes ({self.num_labels})"
        )

    def predict(
        self,
        text: str,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Predict emotion probabilities for a single text entry.

        Args:
            text: Raw input string (e.g. journal entry).
            threshold: Probability threshold for positive prediction (defaults to self.default_threshold).
            top_k: Optional cap on the number of returned predictions.

        Returns:
            Dictionary containing:
                - text: Original input string
                - predictions: List of dicts with emotion name, score, and sentiment category
                - top_emotion: Highest probability emotion regardless of threshold
                - all_scores: Dict mapping all 28 emotions to predicted probabilities
        """
        thresh = threshold if threshold is not None else self.default_threshold

        if not text or not text.strip():
            return {
                "text": text,
                "predictions": [],
                "top_emotion": {"emotion": "neutral", "score": 1.0, "sentiment": "neutral"},
                "all_scores": {label: (1.0 if label == "neutral" else 0.0) for label in EMOTION_LABELS},
            }

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply sigmoid because GoEmotions is a multi-label classification task
            probabilities = torch.sigmoid(outputs.logits)[0].cpu().numpy()

        all_scores = {
            ID2LABEL[i]: round(float(prob), 4)
            for i, prob in enumerate(probabilities)
        }

        # Sort all emotions by descending probability
        sorted_indices = probabilities.argsort()[::-1]
        top_idx = int(sorted_indices[0])
        top_emotion_name = ID2LABEL[top_idx]
        top_emotion = {
            "emotion": top_emotion_name,
            "score": round(float(probabilities[top_idx]), 4),
            "sentiment": get_sentiment_group(top_emotion_name),
        }

        # Filter predictions exceeding threshold
        predictions = []
        for idx in sorted_indices:
            score = float(probabilities[idx])
            if score >= thresh:
                label_name = ID2LABEL[int(idx)]
                predictions.append({
                    "emotion": label_name,
                    "score": round(score, 4),
                    "sentiment": get_sentiment_group(label_name),
                })
            else:
                break

        # Fallback: if no label cleared the threshold, retain top-1 prediction
        if not predictions:
            predictions.append(top_emotion)

        if top_k is not None and top_k > 0:
            predictions = predictions[:top_k]

        return {
            "text": text,
            "predictions": predictions,
            "top_emotion": top_emotion,
            "all_scores": all_scores,
        }

    def predict_batch(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Run batched inference over a list of texts.

        Args:
            texts: List of input strings.
            threshold: Probability threshold for positive prediction.
            batch_size: Number of samples per batch.

        Returns:
            List of prediction result dictionaries.
        """
        thresh = threshold if threshold is not None else self.default_threshold
        results: List[Dict[str, Any]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.sigmoid(outputs.logits).cpu().numpy()

            for text, probs in zip(batch_texts, probabilities):
                all_scores = {
                    ID2LABEL[idx]: round(float(p), 4)
                    for idx, p in enumerate(probs)
                }
                sorted_indices = probs.argsort()[::-1]
                top_idx = int(sorted_indices[0])
                top_emotion_name = ID2LABEL[top_idx]
                top_emotion = {
                    "emotion": top_emotion_name,
                    "score": round(float(probs[top_idx]), 4),
                    "sentiment": get_sentiment_group(top_emotion_name),
                }

                predictions = []
                for idx in sorted_indices:
                    score = float(probs[idx])
                    if score >= thresh:
                        label_name = ID2LABEL[int(idx)]
                        predictions.append({
                            "emotion": label_name,
                            "score": round(score, 4),
                            "sentiment": get_sentiment_group(label_name),
                        })
                    else:
                        break

                if not predictions:
                    predictions.append(top_emotion)

                results.append({
                    "text": text,
                    "predictions": predictions,
                    "top_emotion": top_emotion,
                    "all_scores": all_scores,
                })

        return results


# Module-level cached classifier instance for simple one-off calls
_DEFAULT_CLASSIFIER: Optional[EmotionClassifier] = None


def get_classifier() -> EmotionClassifier:
    """Get or create singleton EmotionClassifier instance."""
    global _DEFAULT_CLASSIFIER
    if _DEFAULT_CLASSIFIER is None:
        _DEFAULT_CLASSIFIER = EmotionClassifier()
    return _DEFAULT_CLASSIFIER


def predict_emotions(
    text: str,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Top-level convenience function for single-text emotion inference."""
    return get_classifier().predict(text=text, threshold=threshold, top_k=top_k)
