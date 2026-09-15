"""Retrieval layer and deterministic query builder for Pensieve Phase 4 RAG Grounding.

Transforms structured outputs from Phases 1–3 into deterministic retrieval queries and
executes top-k semantic retrieval against the local vector index with similarity thresholding.

IMPORTANT NOTICE:
Phase 4 contains NO LLM, performs NO text generation, and makes NO medical or psychiatric
diagnoses. Retrieval relevance reflects conceptual similarity for reflective journaling,
not psychological certainty about the user.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ml.rag.documents import ConceptDocument, KnowledgeBase
from ml.rag.embeddings import ConceptEmbedder, get_concept_embedder
from ml.rag.index import VectorIndex


# Standard human-readable labels for theme clusters (from Phase 2 taxonomy)
THEME_CLUSTER_NAMES: Dict[int, str] = {
    0: "Work & Engineering",
    1: "Relationships & Social Connection",
    2: "Personal Habits & Goals",
    3: "Health, Fitness & Vitality",
    4: "Philosophy, Reflection & Mindset",
}


@dataclass
class PatternSignal:
    """Structured container for signals aggregated from ML Phases 1–3."""

    emotions: Dict[str, float] = field(default_factory=dict)
    themes: List[Union[Dict[str, Any], str, int]] = field(default_factory=list)
    linguistic_patterns: Dict[str, Any] = field(default_factory=dict)
    longitudinal_patterns: List[str] = field(default_factory=list)
    raw_query: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternSignal":
        """Instantiate PatternSignal from dictionary representation."""
        return cls(
            emotions=dict(data.get("emotions", {})),
            themes=list(data.get("themes", [])),
            linguistic_patterns=dict(data.get("linguistic_patterns", {})),
            longitudinal_patterns=list(data.get("longitudinal_patterns", [])),
            raw_query=data.get("raw_query"),
        )


class DeterministicQueryBuilder:
    """Constructs deterministic, interpretable retrieval queries from pattern signals.

    Operates strictly via rule-based formatting and ranking of observed signals.
    Does NOT use an LLM or prompt generation.
    """

    def __init__(
        self,
        emotion_salience_threshold: float = 0.20,
        max_emotions: int = 3,
    ) -> None:
        """Initialize query builder settings.

        Args:
            emotion_salience_threshold: Minimum probability for an emotion to be considered salient.
            max_emotions: Maximum number of prominent emotions to include in query.
        """
        self.emotion_salience_threshold = emotion_salience_threshold
        self.max_emotions = max_emotions

    def build_query(self, signals: PatternSignal) -> Tuple[str, Dict[str, Any]]:
        """Transform structured pattern signals into a descriptive text query.

        Args:
            signals: PatternSignal instance containing emotions, themes, linguistics, and trends.

        Returns:
            Tuple of (retrieval_query_string, audit_trail_dict).
        """
        if signals.raw_query and signals.raw_query.strip():
            return signals.raw_query.strip(), {
                "source": "raw_query",
                "direct_text": signals.raw_query.strip(),
            }

        clauses: List[str] = []
        audit: Dict[str, Any] = {
            "salient_emotions": [],
            "identified_themes": [],
            "linguistic_markers": [],
            "longitudinal_elements": [],
        }

        # 1. Longitudinal trends take priority (narrative trajectory over time)
        if signals.longitudinal_patterns:
            for item in signals.longitudinal_patterns:
                cleaned_item = str(item).strip()
                if cleaned_item:
                    clauses.append(cleaned_item)
                    audit["longitudinal_elements"].append(cleaned_item)

        # 2. Themes from Phase 2
        for theme_item in signals.themes:
            theme_name = ""
            if isinstance(theme_item, dict):
                cluster_id = theme_item.get("cluster_id")
                theme_name = theme_item.get("name", "")
                if not theme_name and cluster_id is not None:
                    theme_name = THEME_CLUSTER_NAMES.get(int(cluster_id), f"Theme {cluster_id}")
            elif isinstance(theme_item, int):
                theme_name = THEME_CLUSTER_NAMES.get(theme_item, f"Theme {theme_item}")
            elif isinstance(theme_item, str):
                theme_name = theme_item

            if theme_name:
                theme_clause = f"Focus on {theme_name.lower()}"
                if theme_clause not in clauses:
                    clauses.append(theme_clause)
                    audit["identified_themes"].append(theme_name)

        # 3. Salient emotions from Phase 1
        sorted_emotions = sorted(
            signals.emotions.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        salient_emotions = [
            (emo, score)
            for emo, score in sorted_emotions
            if score >= self.emotion_salience_threshold and emo != "neutral"
        ][: self.max_emotions]

        if salient_emotions:
            emotion_names = [emo for emo, _ in salient_emotions]
            emotion_clause = f"Experiencing {', '.join(emotion_names)}"
            clauses.append(emotion_clause)
            audit["salient_emotions"] = [
                {"emotion": emo, "score": round(score, 3)}
                for emo, score in salient_emotions
            ]

        # 4. Linguistic style markers from Phase 3
        ling = signals.linguistic_patterns
        if ling:
            # Check for high negation (indicating struggle or conflict)
            negation_ratio = ling.get("negation_ratio", 0.0)
            if negation_ratio > 0.05:
                clauses.append("repeated difficulties and obstacles")
                audit["linguistic_markers"].append("elevated_negation")

            # Check for high question frequency (indicating uncertainty or seeking answers)
            question_count = ling.get("question_count", 0)
            question_freq = ling.get("question_frequency", 0.0)
            if question_count >= 2 or question_freq > 0.10:
                clauses.append("questioning circumstances and searching for clarity")
                audit["linguistic_markers"].append("elevated_questions")

            # Check for high first-person pronoun ratio (deep self-focus or introspection)
            fp_ratio = ling.get("first_person_pronoun_ratio", 0.0) or ling.get("first_person_usage", 0.0)
            if fp_ratio > 0.15:
                clauses.append("intense personal introspection")
                audit["linguistic_markers"].append("high_first_person_focus")

        # Fallback if no specific signals were provided
        if not clauses:
            query = "Reflective personal journaling and emotional self-inquiry."
            audit["fallback"] = True
        else:
            query = ". ".join(clauses) + "."

        return query, audit


class ConceptRetriever:
    """Semantic concept retriever with top-k matching and similarity threshold gating."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        embedder: Optional[ConceptEmbedder] = None,
        index: Optional[VectorIndex] = None,
        default_top_k: int = 3,
        default_threshold: float = 0.25,
    ) -> None:
        """Initialize retriever and populate vector index.

        Args:
            knowledge_base: KnowledgeBase containing concept documents.
            embedder: ConceptEmbedder instance.
            index: VectorIndex instance.
            default_top_k: Default number of concepts to retrieve.
            default_threshold: Minimum cosine similarity required to return a concept.
        """
        self.knowledge_base = knowledge_base
        self.embedder = embedder or get_concept_embedder()
        self.index = index or VectorIndex(dimension=self.embedder.embedding_dim)
        self.default_top_k = default_top_k
        self.default_threshold = default_threshold
        self.query_builder = DeterministicQueryBuilder()

        # Build index if empty
        if self.index.size() == 0 and len(self.knowledge_base) > 0:
            self._build_index()

    def _build_index(self) -> None:
        """Index all documents from the knowledge base."""
        docs = self.knowledge_base.get_all_documents()
        doc_ids = [doc.id for doc in docs]
        embeddings = self.embedder.embed_documents(docs)
        self.index.add_documents(doc_ids, embeddings)

    def retrieve(
        self,
        signals_or_query: Union[PatternSignal, Dict[str, Any], str],
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Retrieve relevant reflective concepts for incoming pattern signals or raw text query.

        Args:
            signals_or_query: PatternSignal, signal dictionary, or raw string query.
            top_k: Number of results to return (defaults to self.default_top_k).
            similarity_threshold: Minimum cosine similarity (defaults to self.default_threshold).

        Returns:
            Structured dictionary with retrieval results, similarity scores, sources, and cautions.
        """
        k = top_k if top_k is not None else self.default_top_k
        threshold = similarity_threshold if similarity_threshold is not None else self.default_threshold

        # Parse input into PatternSignal
        if isinstance(signals_or_query, str):
            signals = PatternSignal(raw_query=signals_or_query)
        elif isinstance(signals_or_query, dict):
            signals = PatternSignal.from_dict(signals_or_query)
        elif isinstance(signals_or_query, PatternSignal):
            signals = signals_or_query
        else:
            raise TypeError(f"Unsupported input type for retriever: {type(signals_or_query)}")

        # 1. Deterministic query construction (No LLM)
        query_text, query_audit = self.query_builder.build_query(signals)

        # 2. Query embedding
        query_embedding = self.embedder.embed_query(query_text)

        # 3. Vector similarity search
        raw_matches = self.index.search(query_embedding, top_k=k)

        # 4. Filter by similarity threshold
        filtered_results: List[Dict[str, Any]] = []
        for doc_id, score in raw_matches:
            if score >= threshold:
                doc = self.knowledge_base.get_document(doc_id)
                if doc is not None:
                    filtered_results.append({
                        "concept_id": doc.id,
                        "name": doc.name,
                        "category": doc.category,
                        "definition": doc.definition,
                        "explanation": doc.explanation,
                        "similarity_score": round(float(score), 4),
                        "source": doc.source,
                        "cautions": doc.cautions,
                    })

        # 5. Build structured response
        if not filtered_results:
            return {
                "status": "no_relevant_concepts",
                "query": query_text,
                "query_audit": query_audit,
                "threshold_applied": threshold,
                "results": [],
                "message": (
                    f"No concept in the knowledge base met the minimum similarity threshold ({threshold}). "
                    "Pensieve does not force irrelevant concepts into reflection."
                ),
                "disclaimer": (
                    "Retrieval relevance is a descriptive semantic signal. It is NOT a medical, "
                    "psychiatric, or psychological diagnosis."
                ),
            }

        return {
            "status": "success",
            "query": query_text,
            "query_audit": query_audit,
            "threshold_applied": threshold,
            "results": filtered_results,
            "disclaimer": (
                "Retrieval relevance is a descriptive semantic signal. It is NOT a medical, "
                "psychiatric, or psychological diagnosis."
            ),
        }
