# Pensieve ML — Phase 1: Emotion Detection Pipeline

Pensieve is a privacy-first AI journaling system. This module implements **Phase 1: Emotion Detection**, providing linguistic emotion classification over user reflections using a fine-tuned RoBERTa model evaluated on the Google GoEmotions benchmark.

---

## 1. Project Structure

```
ml/
├── data/                      # Dataset caching and documentation
│   └── README.md
├── emotion/                   # Core emotion detection package
│   ├── __init__.py            # Public package exports
│   ├── inference.py           # Clean inference interface for arbitrary journal entries
│   ├── evaluate.py            # Reproducible multi-label evaluation pipeline
│   └── labels.py              # GoEmotions taxonomy, mappings, and multi-hot encoders
├── notebooks/                 # Exploratory and interactive walkthroughs
│   └── 01_emotion_detection_demo.ipynb
├── demo.py                    # Qualitative demonstration on 8 realistic journal entries
├── evaluation_results.json    # Exact evaluation results on the official GoEmotions test split
├── requirements.txt           # Minimal pinned dependencies
└── README.md                  # Comprehensive documentation and technical report
```

---

## 2. Dependencies & Installation

### Requirements
- Python 3.10+
- PyTorch >= 2.0.0
- Hugging Face Transformers >= 4.30.0
- Hugging Face Datasets >= 2.14.0
- scikit-learn >= 1.2.0
- NumPy >= 1.24.0
- tqdm >= 4.65.0

### Installation
From the project root:

```bash
pip install -r ml/requirements.txt
```

---

## 3. Dataset Inspection: GoEmotions

Before building the evaluation pipeline, the official `go_emotions` dataset splits were systematically inspected:

### Split Sample Counts
| Split | Samples | Single-Label Samples | Multi-Label Samples | % Multi-Label | Max Labels / Sample |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | 43,410 | 36,308 | 7,102 | 16.36% | 5 |
| **Validation** | 5,426 | 4,548 | 878 | 16.18% | 4 |
| **Test** | 5,427 | 4,590 | 837 | 15.42% | 4 |
| **Total** | **54,263** | **45,446** | **8,817** | **16.25%** | **5** |

### Label Representation & Encoding
- **Taxonomy**: 28 total classes (27 fine-grained emotions + 1 `neutral` class).
- **Format in Dataset**: List of integer class indices `0` through `27` per sample.
- **Evaluation Encoding**: Binary multi-hot vectors of shape `(N, 28)` where $y_{i, c} \in \{0, 1\}$.

### Class Distribution & Imbalance
The dataset exhibits extreme long-tail class imbalance:
- **Dominant classes**: `neutral` (~32.9% of test set), `admiration` (~9.3%), `gratitude` (~6.5%), `approval` (~6.5%), `annoyance` (~5.9%).
- **Minority / Sparse classes**: `grief` (only 6 test instances, 77 train instances), `relief` (11 test instances), `pride` (16 test instances), `nervousness` (23 test instances), `embarrassment` (37 test instances).

---

## 4. Model Architecture

- **Checkpoint**: [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- **Base Architecture**: `roberta-base` (12 layers, 768 hidden dimensions, 12 attention heads, 125M parameters)
- **Classification Head**: Multi-label sequence classification head with 28 outputs trained with Binary Cross-Entropy with Logits loss (`BCEWithLogitsLoss`).
- **Inference Activation**: Sigmoid activation applied independently to each output logit:
  $$p_c = \sigma(z_c) = \frac{1}{1 + e^{-z_c}}$$

---

## 5. Evaluation Methodology

### Why Multi-Label Metrics?
In multi-label classification, standard single-label accuracy (e.g. `argmax` top-1 matching) is inappropriate because samples frequently exhibit multiple co-occurring emotions (e.g., `joy` + `gratitude`, or `confusion` + `sadness`). We evaluate using:
- **Macro F1**: Unweighted mean of F1 across all 28 classes (sensitive to rare classes).
- **Micro F1**: Globally pooled true positives, false positives, and false negatives.
- **Precision & Recall**: Both Macro and Micro variants.
- **Hamming Loss**: Fraction of labels that are incorrectly predicted across all samples and classes.

### Threshold Calibration (Strict Validation Tuning)
Because rare classes output lower average logits, a standard fixed threshold of $0.50$ produces high precision but severely penalizes recall for minority classes. 

To determine the optimal probability threshold without test-set leakage:
1. We evaluated candidate thresholds $t \in [0.10, 0.60]$ strictly on the **Validation Split** (5,426 samples).
2. The threshold maximizing validation Macro F1 was identified as **$t = 0.15$** (Validation Macro F1: 0.5041, Micro F1: 0.5832).
3. For balanced precision-recall in production inference, $t \in [0.25, 0.30]$ achieved peak validation Micro F1 (0.6089).
4. The calibrated threshold $t = 0.15$ was then locked and applied to the **official Test Split** (5,427 samples).
5. For rigorous comparison, the standard baseline threshold $t = 0.50$ was also evaluated on the test set.

---

## 6. Official Test Set Evaluation Results

Calculated directly from the official GoEmotions test set ($N = 5,427$ samples) and stored in `ml/evaluation_results.json`:

| Metric | Calibrated Threshold ($t = 0.15$) | Standard Baseline ($t = 0.50$) |
| :--- | :--- | :--- |
| **Macro F1** | **0.4925** | 0.4503 |
| **Micro F1** | 0.5775 | **0.5855** |
| **Macro Precision** | 0.4419 | **0.5752** |
| **Micro Precision** | 0.4724 | **0.6846** |
| **Macro Recall** | **0.5998** | 0.3962 |
| **Micro Recall** | **0.7428** | 0.5115 |
| **Hamming Loss** | 0.04526 | **0.03016** |

### Per-Class F1 Highlights (Test Set)
- **High-Performing Emotions ($F_1 > 0.70$)**:
  - `amusement`: F1 = 0.8124 (P = 0.7135, R = 0.9432, support = 264)
  - `gratitude`: F1 = 0.8656 (P = 0.8415, R = 0.8920, support = 352)
  - `love`: F1 = 0.7601 (P = 0.7302, R = 0.7931, support = 232)
  - `admiration`: F1 = 0.6836 (P = 0.5835, R = 0.8254, support = 504)
  - `neutral`: F1 = 0.6769 (P = 0.5480, R = 0.8853, support = 1,787)
- **Challenging / Low-Support Emotions ($F_1 < 0.40$)**:
  - `annoyance`: F1 = 0.3486 (diffuse linguistic cues, frequently confused with anger/disapproval)
  - `caring`: F1 = 0.3977 (high overlap with love/admiration)
  - `confusion`: F1 = 0.3766 (high recall 0.7582, lower precision 0.2505)
  - `grief`: F1 = 0.4000 (only 6 test instances)
  - `pride`: F1 = 0.3529 (only 16 test instances)

---

## 7. How to Run Inference

### Python API

```python
from ml.emotion import EmotionClassifier, predict_emotions

# Option 1: Module-level helper
result = predict_emotions("I'm so grateful for this quiet morning with warm coffee.")
print("Top emotion:", result["top_emotion"])
print("Predicted emotions:", result["predictions"])

# Option 2: Dedicated classifier instance
classifier = EmotionClassifier(default_threshold=0.30)
entry = (
    "I have a presentation tomorrow and my stomach is in knots. "
    "What if I forget what to say?"
)
output = classifier.predict(entry)

# Output structure:
# {
#   "text": entry,
#   "predictions": [
#       {"emotion": "nervousness", "score": 0.482, "sentiment": "negative"},
#       {"emotion": "fear", "score": 0.315, "sentiment": "negative"}
#   ],
#   "top_emotion": {"emotion": "nervousness", "score": 0.482, "sentiment": "negative"},
#   "all_scores": {"admiration": 0.001, ..., "nervousness": 0.482, ...}
# }
```

### Batched Inference

```python
entries = [
    "Shipped our v1 release today! Celebrating with the team.",
    "Another meeting with zero outcomes. So exhausted and annoyed.",
]
batch_results = classifier.predict_batch(entries, batch_size=32)
```

### CLI Qualitative Demo
Run the standalone qualitative demo on 8 realistic journal entries:

```bash
python ml/demo.py
```

---

## 8. How to Run Evaluation

To reproduce the full evaluation run against the official GoEmotions validation and test splits:

```bash
python ml/emotion/evaluate.py --batch-size 64 --output ml/evaluation_results.json
```

This will:
1. Download/load GoEmotions official splits via Hugging Face.
2. Inspect split statistics and class distributions.
3. Compute sigmoid probabilities on the validation set.
4. Sweep candidate decision thresholds strictly on the validation set.
5. Compute Macro F1, Micro F1, Precision, Recall, and Hamming Loss on the official test set.
6. Export the full results payload to `ml/evaluation_results.json`.

---

## 9. Important Limitations & Ethical Considerations

> [!WARNING]
> ### 1. Strict Non-Diagnostic Classification
> All predictions generated by this model are **strictly linguistic emotion classifications** reflecting textual sentiment and lexical tone.
> - They do **NOT** constitute psychological, psychiatric, or clinical diagnoses.
> - They must **NOT** be used to assess mental health disorders, depression, suicide risk, or psychological wellness.
> - Pensieve displays these classifications solely to assist users in reflective self-discovery.

> [!IMPORTANT]
> ### 2. Reddit Domain Bias vs. Personal Journaling
> - The GoEmotions dataset was annotated from publicly posted Reddit comments from 2005 to 2019.
> - Reddit comments tend to be short, public-facing, sarcastic, argumentative, or reactive.
> - Personal journal entries, by contrast, are introspective, private, long-form, vulnerable, and nuanced.
> - Certain emotions (e.g. `grief`, `pride`, `remorse`) occur rarely on Reddit, leading to lower model sensitivity when expressed in private writing.

> [!NOTE]
> ### 3. Extreme Class Imbalance
> - Minority categories like `grief` (6 test samples) and `relief` (11 test samples) suffer from small sample variance.
> - In downstream Pensieve phases, aggregating emotions into higher-level affective clusters (positive, negative, ambiguous) or combining with semantic embeddings can mitigate single-class sparsity.
