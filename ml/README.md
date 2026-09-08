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

---

## 10. Phase 2: Theme Discovery (Sentence-BERT + UMAP + HDBSCAN)

Phase 2 implements an unsupervised theme discovery pipeline that translates user reflections into dense semantic embeddings, reduces high-dimensional distance concentration via UMAP manifold learning, and dynamically groups recurring topics using density-based HDBSCAN clustering.

### 10.1 Final Pipeline Architecture
$$\text{Journal Text} \longrightarrow \text{Sentence-BERT (all-MiniLM-L6-v2)} \longrightarrow \text{384-d Vector} \longrightarrow \text{UMAP (5-d)} \longrightarrow \text{HDBSCAN} \longrightarrow \text{Theme Cluster ID / Outlier (-1)}$$

### 10.2 Final Configuration
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, L2 normalized).
- **UMAP Dimensionality Reduction**:
  - `n_components`: `5`
  - `n_neighbors`: `15`
  - `min_dist`: `0.0`
  - `metric`: `'cosine'`
  - `random_state`: `42`
- **HDBSCAN Clustering**:
  - `min_cluster_size`: `25` (corpus) / `2` (small journal sets)
  - `min_samples`: `5` (corpus) / `1` (small journal sets)
  - `metric`: `'euclidean'`
  - `cluster_selection_method`: `'eom'` (Excess of Mass)
- **Outlier Preservation**: Unclustered/sparse entries strictly retain the `-1` noise label and are never forced into clusters.
- **Centroid & Representative Identification**: For each discovered cluster, a normalized mean centroid $\mu_c$ is calculated in 384-d semantic space. Member entries are ranked by descending cosine similarity $\cos(x_i, \mu_c)$ to retrieve central representative entries.

### 10.3 Quantitative Diagnostics on GoEmotions Corpus
Evaluated on the official GoEmotions test split ($N = 5,427$ texts, strictly ignoring emotion labels):

| Metric | Final Architecture (UMAP + HDBSCAN) | Raw Baseline (384-d + HDBSCAN) |
| :--- | :---: | :---: |
| **Discovered Clusters** | **38** | 9 |
| **Clustered Samples** | **3,635 (66.98%)** | 768 (14.15%) |
| **Noise Samples (-1)** | **1,792 (33.02%)** | 4,659 (85.85%) |
| **Silhouette Score (5-d UMAP space)** | **0.3501** | N/A |
| **Davies-Bouldin Index (5-d UMAP space)** | **0.8285** | N/A |
| **Calinski-Harabasz Score (5-d UMAP space)**| **738.56** | N/A |
| **Silhouette Score (projected 384-d)** | 0.0186 | 0.1052 |
| **Davies-Bouldin (projected 384-d)** | 4.0827 | 2.8062 |
| **Calinski-Harabasz (projected 384-d)** | 15.51 | 17.12 |

*Diagnostic metrics evaluate mathematical compactness and cluster separation in embedding space, not semantic correctness.*

### 10.4 Running Theme Discovery

#### Python API
```python
from ml.theme import TextEmbedder, ThemeClusterer

embedder = TextEmbedder()
clusterer = ThemeClusterer(
    min_cluster_size=2,
    min_samples=1,
    use_umap=True,
    umap_components=5,
    umap_neighbors=15,
)

texts = [
    "Spent hours studying calculus in the library for the exam.",
    "Submitted my thesis chapter; advisor gave encouraging feedback.",
    "Sprint deadline at work; pushed the backend deployment patch.",
    "Ran a 5k around the lake this morning and hit a new personal best.",
]

embeddings = embedder.embed_texts(texts)
labels = clusterer.fit_predict(embeddings, texts=texts)
summary = clusterer.get_cluster_summary(top_n_examples=1)
```

#### Run Qualitative Demo
```bash
python ml/theme_demo.py
```

#### Run Quantitative Evaluation
```bash
python ml/theme/evaluate.py --output ml/theme_evaluation_results.json
```

---

## 11. Phase 3: Linguistic & Longitudinal Pattern Analysis

Phase 3 introduces a lightweight, descriptive analytical layer that tracks observable linguistic, affective, and thematic trajectories across a user's chronological journal history.

> [!IMPORTANT]
> ### Ethical Notice & Non-Diagnostic Purpose
> - **Observational Only**: Phase 3 detects statistical shifts and stylistic tendencies in writing over time. It does **not** diagnose psychological conditions, evaluate mental disorders, predict mental health, or make causal claims.
> - **Uncertainty-Aware Language**: All insights use non-definitive phrasing (*"appears more frequently"*, *"shows a gradual upward trend"*, *"may indicate a recurring pattern"*).
> - **Descriptive Associations**: Relationships between themes and emotions represent co-occurrences in text, **not** psychological cause-and-effect.

### 11.1 Pipeline Architecture
$$\text{Dated Journal Entries} \longrightarrow \text{spaCy Linguistic Analysis} \longrightarrow \text{Chronological Aggregation} \longrightarrow \begin{cases} \text{Emotion Trends (Phase 1 Probabilities)} \\ \text{Theme Trends (Phase 2 Cluster IDs)} \\ \text{Linguistic Style Trajectories} \\ \text{Recurring Lexical Patterns} \\ \text{Theme-Emotion Associations} \end{cases}$$

### 11.2 Interpretable Linguistic Features
Extracted per entry using spaCy (`en_core_web_sm`):
1. **Token Count**: Total tokens including punctuation.
2. **Word Count**: Content and function words (excluding punctuation/whitespace).
3. **Sentence Count**: Sentence boundary segmentation.
4. **Average Sentence Length**: Words per sentence.
5. **Vocabulary Diversity**: Type-Token Ratio (unique lemmas / total words).
6. **Pronoun Usage**: Total pronoun count and pronoun-to-word ratio.
7. **First-Person Pronoun Frequency**: Frequency of singular/plural first-person pronouns (`I`, `me`, `my`, `mine`, `myself`, `we`, `us`, `our`, `ours`, `ourselves`).
8. **Question Count**: Inquisitive sentences ending in `?`.
9. **Exclamation Count**: Exclamatory sentences containing `!`.
10. **Negation Count**: Negation tokens (`no`, `not`, `n't`, `never`, `nobody`, `nowhere`, etc.).
11. **Adjective & Adverb Frequency**: Modifiers and descriptive words.
12. **Verb Frequency**: Action and auxiliary verbs.
13. **POS Distribution**: Full coarse-grained Part-of-Speech breakdown.

### 11.3 Longitudinal Analysis Modules

#### 1. Time-Window Aggregation (`ml/longitudinal/aggregation.py`)
- Sorts entries chronologically by ISO-8601 timestamps.
- Aggregates entries into configurable windows (`weekly`, `monthly`, or custom days).
- **Minimum-Data Safeguards**:
  - Requires $\ge 3$ entries; returns `{"status": "insufficient_data"}` if fewer entries are provided.
  - Flags histories spanning $< 7$ days with a brevity warning to avoid premature trend claims.

#### 2. Emotion Trajectories (`ml/longitudinal/trends.py`)
- Preserves continuous Phase 1 probabilities rather than binarizing into single labels.
- Calculates mean probability per emotion across windows and flags meaningful shifts ($\Delta \ge 0.05$).

#### 3. Theme Dynamics (`ml/longitudinal/trends.py`)
- Tracks Phase 2 cluster frequencies over time, identifying recurring, emerging, and diminishing topics.

#### 4. Recurring Lexical Patterns (`ml/longitudinal/patterns.py`)
- Extracts content words and bigrams appearing across multiple entries, filtering English stopwords.

#### 5. Theme-Emotion Associations (`ml/longitudinal/patterns.py`)
- Computes conditional emotion distributions per theme cluster to identify descriptive co-occurrences.

### 11.4 Running Phase 3

#### Python API
```python
from ml.longitudinal import analyze_journal_history

entries = [
    {
        "id": "e1",
        "timestamp": "2026-01-02T10:00:00Z",
        "text": "Stressed about project deadline and late night debugging.",
        "emotions": {"annoyance": 0.72, "nervousness": 0.81, "joy": 0.05},
        "theme_cluster": 0,
    },
    {
        "id": "e2",
        "timestamp": "2026-01-10T12:00:00Z",
        "text": "Shipped the code patch! Feeling relieved and happy.",
        "emotions": {"annoyance": 0.15, "nervousness": 0.10, "joy": 0.75},
        "theme_cluster": 0,
    },
    {
        "id": "e3",
        "timestamp": "2026-01-18T08:00:00Z",
        "text": "Morning 5k run around the lake. Energetic and grateful.",
        "emotions": {"annoyance": 0.02, "nervousness": 0.04, "joy": 0.85},
        "theme_cluster": 1,
    },
]

report = analyze_journal_history(entries, window_type="weekly")
```

#### Run Qualitative Demo
```bash
python ml/phase3_demo.py
```

#### Run Synthetic Validation Suite
```bash
python ml/longitudinal/evaluate.py
```



