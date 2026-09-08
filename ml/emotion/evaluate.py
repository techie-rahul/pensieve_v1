"""Evaluation pipeline for GoEmotions emotion classification.

Computes reproducible multi-label evaluation metrics:
- Macro & Micro F1
- Macro & Micro Precision
- Macro & Micro Recall
- Hamming Loss
- Per-class metrics

Methodology:
- Evaluates candidate probability thresholds strictly on the VALIDATION split.
- Applies the optimal threshold (and standard 0.50 baseline) to the official TEST split.
- Zero test leakage.
- Saves results directly to JSON.
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter
import numpy as np

# Ensure workspace root is in sys.path when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from ml.emotion.labels import (
    EMOTION_LABELS,
    ID2LABEL,
    NUM_LABELS,
    labels_to_multihot,
)

DEFAULT_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
DEFAULT_OUTPUT_PATH = "ml/evaluation_results.json"


def inspect_dataset_splits(dataset: Any) -> Dict[str, Any]:
    """Inspect and summarize split sizes, label frequencies, and multi-label distribution."""
    inspection: Dict[str, Any] = {}

    for split_name in ["train", "validation", "test"]:
        split_data = dataset[split_name]
        total = len(split_data)
        counts_per_sample = [len(x["labels"]) for x in split_data]
        distribution = Counter(counts_per_sample)

        class_freq = Counter()
        for x in split_data:
            for l in x["labels"]:
                class_freq[EMOTION_LABELS[l]] += 1

        inspection[split_name] = {
            "total_samples": total,
            "single_label_samples": distribution.get(1, 0),
            "multi_label_samples": sum(v for k, v in distribution.items() if k > 1),
            "zero_label_samples": distribution.get(0, 0),
            "labels_per_sample_distribution": {str(k): v for k, v in sorted(distribution.items())},
            "top_5_frequent_emotions": class_freq.most_common(5),
            "least_5_frequent_emotions": class_freq.most_common()[-5:],
        }

    return inspection


def predict_probabilities(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: List[str],
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 128,
) -> np.ndarray:
    """Run model inference over texts and return sigmoid probability array."""
    model.eval()
    all_probs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Binary Cross Entropy with Logits -> Sigmoid probabilities
            probs = torch.sigmoid(outputs.logits).cpu().numpy()
            all_probs.append(probs)

    return np.vstack(all_probs)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute multi-label classification metrics.

    Args:
        y_true: Binary ground-truth matrix of shape (N, num_classes).
        y_pred: Binary prediction matrix of shape (N, num_classes).

    Returns:
        Dictionary of aggregate and per-class metrics.
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    macro_precision = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    micro_precision = float(precision_score(y_true, y_pred, average="micro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    micro_recall = float(recall_score(y_true, y_pred, average="micro", zero_division=0))
    h_loss = float(hamming_loss(y_true, y_pred))

    # Per-class F1, Precision, and Recall
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class_metrics = {}
    for idx, name in enumerate(EMOTION_LABELS):
        per_class_metrics[name] = {
            "f1": round(float(per_class_f1[idx]), 4),
            "precision": round(float(per_class_precision[idx]), 4),
            "recall": round(float(per_class_recall[idx]), 4),
            "support": int(y_true[:, idx].sum()),
        }

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "micro_precision": round(micro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "micro_recall": round(micro_recall, 4),
        "hamming_loss": round(h_loss, 5),
        "per_class": per_class_metrics,
    }


def find_optimal_threshold(
    y_true_val: np.ndarray,
    val_probs: np.ndarray,
    candidate_thresholds: List[float],
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """Search for optimal global probability threshold maximizing Macro F1 on validation split."""
    tuning_history: Dict[float, Dict[str, float]] = {}
    best_threshold = 0.50
    best_macro_f1 = -1.0

    for thresh in candidate_thresholds:
        y_pred = (val_probs >= thresh).astype(int)
        macro_f1 = float(f1_score(y_true_val, y_pred, average="macro", zero_division=0))
        micro_f1 = float(f1_score(y_true_val, y_pred, average="micro", zero_division=0))
        h_loss = float(hamming_loss(y_true_val, y_pred))

        tuning_history[round(thresh, 2)] = {
            "macro_f1": round(macro_f1, 4),
            "micro_f1": round(micro_f1, 4),
            "hamming_loss": round(h_loss, 5),
        }

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = thresh

    return best_threshold, tuning_history


def run_evaluation(
    model_name: str = DEFAULT_MODEL_NAME,
    output_path: str = DEFAULT_OUTPUT_PATH,
    batch_size: int = 64,
) -> Dict[str, Any]:
    """Run full evaluation: inspect dataset, tune threshold on val, evaluate on test, save results."""
    print("=" * 70)
    print("PENSIEVE ML - GOEMOTIONS EVALUATION PIPELINE")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model & tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    print("Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions")
    inspection_stats = inspect_dataset_splits(dataset)

    print("\n--- DATASET INSPECTION SUMMARY ---")
    for split_name, stats in inspection_stats.items():
        print(
            f"Split [{split_name}]: {stats['total_samples']} samples | "
            f"{stats['single_label_samples']} single-label, "
            f"{stats['multi_label_samples']} multi-label ({stats['multi_label_samples'] / stats['total_samples']:.1%})"
        )

    # Prepare ground-truth matrices
    print("\nEncoding validation and test ground-truth labels...")
    y_true_val = np.array([labels_to_multihot(x["labels"]) for x in dataset["validation"]])
    y_true_test = np.array([labels_to_multihot(x["labels"]) for x in dataset["test"]])

    # Generate probabilities on validation set
    print(f"\nEvaluating validation split ({len(dataset['validation'])} samples) for threshold calibration...")
    val_texts = dataset["validation"]["text"]
    val_probs = predict_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=val_texts,
        device=device,
        batch_size=batch_size,
    )

    # Search for optimal threshold strictly on validation split
    candidate_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    best_thresh, tuning_history = find_optimal_threshold(
        y_true_val=y_true_val,
        val_probs=val_probs,
        candidate_thresholds=candidate_thresholds,
    )

    print(f"\nValidation threshold search results:")
    for t, m in tuning_history.items():
        mark = " <-- OPTIMAL (Validation Macro F1)" if abs(t - best_thresh) < 1e-4 else ""
        print(f"  Threshold {t:.2f}: Macro F1 = {m['macro_f1']:.4f}, Micro F1 = {m['micro_f1']:.4f}, Hamming Loss = {m['hamming_loss']:.5f}{mark}")

    print(f"\nSelected Optimal Threshold from Validation: {best_thresh:.2f}")

    # Generate probabilities on official test set
    print(f"\nEvaluating official test split ({len(dataset['test'])} samples)...")
    test_texts = dataset["test"]["text"]
    test_probs = predict_probabilities(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        device=device,
        batch_size=batch_size,
    )

    # Evaluate on test set with selected optimal threshold
    y_pred_test_calibrated = (test_probs >= best_thresh).astype(int)
    test_metrics_calibrated = compute_multilabel_metrics(y_true_test, y_pred_test_calibrated)

    # Also evaluate on test set with default standard threshold 0.50 for baseline comparison
    y_pred_test_standard = (test_probs >= 0.50).astype(int)
    test_metrics_standard = compute_multilabel_metrics(y_true_test, y_pred_test_standard)

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 70)
    print(f"Calibrated Threshold (t = {best_thresh:.2f}):")
    print(f"  Macro F1:         {test_metrics_calibrated['macro_f1']:.4f}")
    print(f"  Micro F1:         {test_metrics_calibrated['micro_f1']:.4f}")
    print(f"  Macro Precision:  {test_metrics_calibrated['macro_precision']:.4f}")
    print(f"  Micro Precision:  {test_metrics_calibrated['micro_precision']:.4f}")
    print(f"  Macro Recall:     {test_metrics_calibrated['macro_recall']:.4f}")
    print(f"  Micro Recall:     {test_metrics_calibrated['micro_recall']:.4f}")
    print(f"  Hamming Loss:     {test_metrics_calibrated['hamming_loss']:.5f}")

    print(f"\nStandard Threshold Baseline (t = 0.50):")
    print(f"  Macro F1:         {test_metrics_standard['macro_f1']:.4f}")
    print(f"  Micro F1:         {test_metrics_standard['micro_f1']:.4f}")
    print(f"  Macro Precision:  {test_metrics_standard['macro_precision']:.4f}")
    print(f"  Micro Precision:  {test_metrics_standard['micro_precision']:.4f}")
    print(f"  Macro Recall:     {test_metrics_standard['macro_recall']:.4f}")
    print(f"  Micro Recall:     {test_metrics_standard['micro_recall']:.4f}")
    print(f"  Hamming Loss:     {test_metrics_standard['hamming_loss']:.5f}")

    results_payload = {
        "metadata": {
            "model_name": model_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "num_test_samples": len(dataset["test"]),
            "num_val_samples": len(dataset["validation"]),
            "num_classes": NUM_LABELS,
        },
        "dataset_inspection": inspection_stats,
        "validation_threshold_calibration": {
            "selected_optimal_threshold": best_thresh,
            "tuning_history": tuning_history,
        },
        "test_results_calibrated_threshold": {
            "threshold": best_thresh,
            **test_metrics_calibrated,
        },
        "test_results_standard_threshold_0_5": {
            "threshold": 0.50,
            **test_metrics_standard,
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nSaved evaluation results to: {output_path}")
    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RoBERTa on GoEmotions")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Model checkpoint")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH, help="Path to output JSON")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")
    args = parser.parse_args()

    run_evaluation(
        model_name=args.model,
        output_path=args.output,
        batch_size=args.batch_size,
    )
