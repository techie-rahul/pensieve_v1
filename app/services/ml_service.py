"""ML pipeline integration service.

Orchestrates real inference across Phases 1–5:
- Phase 1: RoBERTa emotion detection (SamLowe/roberta-base-go_emotions)
- Phase 2: Sentence-BERT semantic theme alignment
- Phase 3: spaCy linguistic analysis & longitudinal pattern synthesis
- Phase 4: Local FAISS concept retrieval (20-concept DEVELOPMENT / TEST dataset)
- Phase 5: Grounded reflection generation with pre-policy checks and post-validation
"""

from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# ML Phase imports
from ml.emotion.inference import get_classifier
from ml.theme.embedding import get_embedder
from ml.linguistic.analyzer import get_linguistic_analyzer
from ml.longitudinal import analyze_journal_history
from ml.rag import KnowledgeBase, ConceptRetriever, PatternSignal, THEME_CLUSTER_NAMES
from ml.reflection import ReflectionGenerator, ReflectionInput

logger = logging.getLogger("pensieve.ml_service")

# Canonical descriptive prototypes for the 5 Phase 2 theme clusters
THEME_PROTOTYPES: Dict[int, str] = {
    0: "work, engineering, code, software, projects, tasks, meetings, deadlines, career, workplace",
    1: "relationships, friends, family, partner, social connection, conversations, dinner, bonding",
    2: "personal habits, goals, reading, routine, hobbies, discipline, productivity, self-improvement",
    3: "health, fitness, exercise, running, workout, walking, vitality, nutrition, physical energy",
    4: "philosophy, reflection, mindset, meditation, existential, contemplating, wisdom, inner thoughts",
}


class MLService:
    """Singleton service providing thread-safe, cached access to the ML pipeline."""

    _instance: Optional["MLService"] = None

    def __init__(self) -> None:
        """Initialize and cache models lazily."""
        self._classifier = None
        self._embedder = None
        self._linguistic_analyzer = None
        self._knowledge_base = None
        self._retriever = None
        self._reflection_generator = None
        self._theme_prototype_embeddings: Optional[Dict[int, np.ndarray]] = None

    @classmethod
    def get_instance(cls) -> "MLService":
        """Get or create singleton MLService instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def classifier(self):
        """Phase 1: RoBERTa Emotion Classifier."""
        if self._classifier is None:
            self._classifier = get_classifier()
        return self._classifier

    @property
    def embedder(self):
        """Phase 2: Sentence-BERT Text Embedder."""
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    @property
    def linguistic_analyzer(self):
        """Phase 3: spaCy Linguistic Analyzer."""
        if self._linguistic_analyzer is None:
            self._linguistic_analyzer = get_linguistic_analyzer()
        return self._linguistic_analyzer

    @property
    def knowledge_base(self):
        """Phase 4: Curated Development Knowledge Base (20 concepts)."""
        if self._knowledge_base is None:
            self._knowledge_base = KnowledgeBase.load_development_dataset()
        return self._knowledge_base

    @property
    def retriever(self):
        """Phase 4: FAISS Concept Retriever."""
        if self._retriever is None:
            self._retriever = ConceptRetriever(self.knowledge_base, default_top_k=3, default_threshold=0.25)
        return self._retriever

    @property
    def reflection_generator(self):
        """Phase 5: Grounded Reflection Generator."""
        if self._reflection_generator is None:
            self._reflection_generator = ReflectionGenerator()
        return self._reflection_generator

    def _get_theme_prototype_embeddings(self) -> Dict[int, np.ndarray]:
        """Pre-compute embeddings for canonical theme prototypes for fast single-entry alignment."""
        if self._theme_prototype_embeddings is None:
            embeddings = {}
            for cluster_id, desc in THEME_PROTOTYPES.items():
                embeddings[cluster_id] = self.embedder.embed_text(desc)
            self._theme_prototype_embeddings = embeddings
        return self._theme_prototype_embeddings

    def assign_theme(self, text: str) -> Tuple[int, str]:
        """Align entry text to closest semantic theme cluster using Sentence-BERT cosine similarity.

        Returns:
            Tuple of (cluster_id, theme_name).
        """
        text_emb = self.embedder.embed_text(text)
        prototypes = self._get_theme_prototype_embeddings()

        best_cluster = 0
        best_sim = -1.0

        for cluster_id, proto_emb in prototypes.items():
            sim = float(np.dot(text_emb, proto_emb))
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster_id

        theme_name = THEME_CLUSTER_NAMES.get(best_cluster, f"Theme {best_cluster}")
        return best_cluster, theme_name

    def analyze_single_entry(self, text: str) -> Dict[str, Any]:
        """Run Phase 1 (Emotions) + Phase 2 (Theme) + Phase 3 (Linguistics) on a single entry.

        Args:
            text: Raw journal entry content.

        Returns:
            Dictionary containing emotions, theme cluster, and linguistic features.
        """
        cleaned = text.strip() if text else ""
        if not cleaned:
            return {
                "emotions": {},
                "top_emotion": None,
                "theme": "Unassigned",
                "theme_cluster_id": None,
                "linguistic_features": {},
            }

        # 1. Phase 1: Emotion classification
        emotion_pred = self.classifier.predict(cleaned, top_k=5)
        all_scores = emotion_pred.get("all_scores", {})
        top_emotion = emotion_pred.get("top_emotion", None)

        # 2. Phase 2: Theme alignment
        cluster_id, theme_name = self.assign_theme(cleaned)

        # 3. Phase 3: Linguistic features
        linguistic_feats = self.linguistic_analyzer.analyze_text(cleaned)

        return {
            "emotions": all_scores,
            "top_emotion": top_emotion,
            "theme": theme_name,
            "theme_cluster_id": cluster_id,
            "linguistic_features": linguistic_feats,
        }

    def compute_longitudinal_patterns(self, entries: List[Any]) -> Dict[str, Any]:
        """Run Phase 3 longitudinal pattern synthesis across user's chronological journal history.

        Args:
            entries: List of JournalEntry OR dicts containing text, timestamp, emotions, etc.

        Returns:
            Structured longitudinal analysis dictionary.
        """
        formatted_entries = []
        for e in entries:
            # Support both SQLAlchemy model and dict
            entry_id = getattr(e, "id", None) or e.get("id")
            text = getattr(e, "content", None) or e.get("text", "") or e.get("content", "")
            created_at = getattr(e, "created_at", None) or e.get("timestamp") or e.get("created_at")

            ts_str = created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at)

            # Extract existing analysis if present
            analysis = getattr(e, "analysis", None)
            if analysis:
                emotions = analysis.emotions
                theme_cluster = analysis.theme_cluster_id
                ling = analysis.linguistic_features
            else:
                emotions = e.get("emotions", {}) if isinstance(e, dict) else {}
                theme_cluster = e.get("theme_cluster", 0) if isinstance(e, dict) else 0
                ling = e.get("linguistic_features") if isinstance(e, dict) else None

            formatted_entries.append({
                "id": str(entry_id),
                "timestamp": ts_str,
                "text": text,
                "emotions": emotions,
                "theme_cluster": theme_cluster if theme_cluster is not None else 0,
                "linguistic_features": ling,
            })

        # Run Phase 3 analyze_journal_history
        report = analyze_journal_history(
            formatted_entries,
            window_type="weekly",
            theme_names=THEME_CLUSTER_NAMES,
            enrich_linguistics=True,
        )
        return report

    def generate_reflection(
        self,
        entries: List[Any],
        past_reflection_timestamps: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute full Phase 4 retrieval + Phase 5 reflection generation pipeline.

        Args:
            entries: List of user's journal entries.
            past_reflection_timestamps: Timestamps of previous reflections for rate limiting.

        Returns:
            Structured generation dictionary from Phase 5 generator.
        """
        # 1. Run Phase 3 pattern analysis
        patterns = self.compute_longitudinal_patterns(entries)
        if patterns.get("status") == "insufficient_data":
            return {
                "status": "policy_rejected",
                "reason": "insufficient_entries",
                "message": patterns.get("message", "Insufficient journal entries for longitudinal reflection."),
                "details": patterns.get("safeguards", {}),
                "reflection": None,
                "grounded_concepts": [],
                "confidence": 0.0,
                "disclaimer": None,
            }

        # 2. Extract salient signals for Phase 4 Concept Retrieval
        emotions_summary: Dict[str, float] = {}
        emotion_trends = patterns.get("emotion_trends", {})
        if isinstance(emotion_trends, dict):
            for item in emotion_trends.get("shifted_emotions", []):
                emo = item.get("emotion")
                end_score = item.get("end_score", 0.0)
                if emo:
                    emotions_summary[emo] = float(end_score)

        theme_items = []
        theme_trends = patterns.get("theme_trends", {})
        if isinstance(theme_trends, dict):
            for t in theme_trends.get("dominant_themes", []):
                theme_items.append(t)

        longitudinal_elements = []
        for obs in emotion_trends.get("observations", []):
            longitudinal_elements.append(obs)
        for obs in theme_trends.get("observations", []):
            longitudinal_elements.append(obs)

        pattern_signal = PatternSignal(
            emotions=emotions_summary,
            themes=theme_items,
            longitudinal_patterns=longitudinal_elements[:4],
        )

        # 3. Phase 4 Concept Retrieval
        rag_result = self.retriever.retrieve(pattern_signal, top_k=3, similarity_threshold=0.25)

        # 4. Construct Phase 5 ReflectionInput
        reflection_input = ReflectionInput.from_pipeline_outputs(
            history_analysis=patterns,
            rag_result=rag_result,
            past_reflection_timestamps=past_reflection_timestamps or [],
        )

        # 5. Phase 5 Reflection Generation with policy & validator
        result = self.reflection_generator.generate_reflection(reflection_input)
        return result

    def list_concepts(
        self,
        category: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List concepts from the 20-concept development dataset with optional filtering."""
        docs = self.knowledge_base.get_all_documents()
        results = []
        for doc in docs:
            if category and doc.category.lower() != category.lower():
                continue
            if search:
                query = search.lower()
                matches_search = (
                    query in doc.name.lower()
                    or query in doc.definition.lower()
                    or any(query in p.lower() for p in doc.related_patterns)
                )
                if not matches_search:
                    continue
            results.append(doc.to_dict())
        return results

    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Fetch single concept document by ID."""
        doc = self.knowledge_base.get_document(concept_id)
        return doc.to_dict() if doc is not None else None


def get_ml_service() -> MLService:
    """FastAPI dependency for accessing singleton MLService."""
    return MLService.get_instance()
