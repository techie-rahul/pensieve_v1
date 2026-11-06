"""Reflection generation layer and provider abstraction for Pensieve Phase 5.

Provides a clean, backend-ready interface that orchestrates:
1. Input synthesis and validation
2. Pre-generation policy enforcement
3. Provider-isolated LLM generation
4. Confidence calibration and capping (max 0.80)
5. Post-generation safety validation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
import re
from typing import Any, Dict, List, Optional, Union

from ml.reflection.policy import ReflectionPolicy, PolicyCheckResult
from ml.reflection.prompts import build_reflection_prompt


DEFAULT_REFLECTION_MODEL = "gpt-4o-mini"
MAX_CONFIDENCE_CAP = 0.80


@dataclass
class ReflectionInput:
    """Structured container bridging Phases 1–4 signals to Phase 5 generation."""

    data_summary: Dict[str, Any] = field(default_factory=dict)
    emotion_patterns: List[Any] = field(default_factory=list)
    theme_patterns: List[Any] = field(default_factory=list)
    linguistic_patterns: Any = field(default_factory=dict)
    temporal_patterns: List[Any] = field(default_factory=list)
    recurring_patterns: List[Any] = field(default_factory=list)
    retrieved_concepts: List[Dict[str, Any]] = field(default_factory=list)
    past_reflection_timestamps: List[str] = field(default_factory=list)
    evidence_excerpts: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert input to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionInput":
        """Instantiate ReflectionInput from raw dictionary."""
        return cls(
            data_summary=dict(data.get("data_summary", {})),
            emotion_patterns=list(data.get("emotion_patterns", [])),
            theme_patterns=list(data.get("theme_patterns", [])),
            linguistic_patterns=data.get("linguistic_patterns", {}),
            temporal_patterns=list(data.get("temporal_patterns", [])),
            recurring_patterns=list(data.get("recurring_patterns", [])),
            retrieved_concepts=list(data.get("retrieved_concepts", [])),
            past_reflection_timestamps=list(data.get("past_reflection_timestamps", [])),
            evidence_excerpts=data.get("evidence_excerpts"),
        )

    @classmethod
    def from_pipeline_outputs(
        cls,
        history_analysis: Dict[str, Any],
        rag_result: Dict[str, Any],
        past_reflection_timestamps: Optional[List[str]] = None,
        evidence_excerpts: Optional[List[str]] = None,
    ) -> "ReflectionInput":
        """Factory method directly bridging Phase 3 analysis and Phase 4 retrieval.

        Args:
            history_analysis: Output from Phase 3 analyze_journal_history.
            rag_result: Output from Phase 4 ConceptRetriever.retrieve.
            past_reflection_timestamps: Timestamps of past generated reflections.
            evidence_excerpts: Optional limited text excerpts.

        Returns:
            Validated ReflectionInput ready for generation.
        """
        data_summary = {
            "entry_count": history_analysis.get("entry_count", 0),
            "span_days": history_analysis.get("time_span_days", 0),
            "num_windows": history_analysis.get("num_windows", 0),
        }

        # Format emotion trends from Phase 3
        emotion_trends = history_analysis.get("emotion_trends", {})
        emotion_obs = emotion_trends.get("observations", []) if isinstance(emotion_trends, dict) else []

        # Format theme trends from Phase 3
        theme_trends = history_analysis.get("theme_trends", {})
        theme_obs = theme_trends.get("observations", []) if isinstance(theme_trends, dict) else []

        # Format linguistic patterns
        ling_trends = history_analysis.get("linguistic_trends", {})
        ling_obs = ling_trends.get("observations", []) if isinstance(ling_trends, dict) else []

        # Recurring lexical patterns
        lexical = history_analysis.get("recurring_lexical_patterns", {})
        recurring_words = [f"Frequent term: {w['term']}" for w in lexical.get("recurring_words", [])] if isinstance(lexical, dict) else []

        # Retrieved concepts from Phase 4
        retrieved_concepts = rag_result.get("results", [])

        return cls(
            data_summary=data_summary,
            emotion_patterns=emotion_obs,
            theme_patterns=theme_obs,
            linguistic_patterns=ling_obs,
            temporal_patterns=emotion_obs + theme_obs,
            recurring_patterns=recurring_words,
            retrieved_concepts=retrieved_concepts,
            past_reflection_timestamps=past_reflection_timestamps or [],
            evidence_excerpts=evidence_excerpts,
        )


# ==============================================================================
# PROVIDER ISOLATION LAYER
# ==============================================================================

class LLMClient(ABC):
    """Abstract interface isolating LLM provider details from Pensieve logic."""

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Execute model generation and return raw text string (expected JSON)."""
        pass


class OpenAIClient(LLMClient):
    """OpenAI API provider implementation reading credentials strictly from environment."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_REFLECTION_MODEL,
        temperature: float = 0.3,
    ) -> None:
        """Initialize client.

        Args:
            api_key: Optional explicit key; if None, reads from REFLECTION_API_KEY or OPENAI_API_KEY.
            model: Model name checkpoint.
            temperature: Sampling temperature (conservative for structured reflection).
        """
        self.api_key = api_key or os.environ.get("REFLECTION_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Missing LLM API key. Set the 'REFLECTION_API_KEY' or 'OPENAI_API_KEY' environment variable. "
                "Pensieve strictly prohibits hardcoded credentials."
            )
        self.model = model
        self.temperature = temperature
        
        # Lazy import openai to ensure isolation
        import openai
        self._client = openai.OpenAI(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI API with JSON output mode."""
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response returned from language model service.")
        return content


class MockLLMClient(LLMClient):
    """Deterministic mock provider for offline unit tests, evaluation benchmarks, and CI.

    Produces strictly grounded, compliant 2-3 paragraph responses meeting Reflections 2.0 criteria.
    """

    def __init__(self, canned_response: Optional[Dict[str, Any]] = None) -> None:
        """Initialize mock client with optional fixed response."""
        self.canned_response = canned_response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return deterministic JSON response."""
        if self.canned_response is not None:
            return json.dumps(self.canned_response)

        # 1. Dynamically extract all retrieved concepts from user_prompt
        grounded_concepts: List[Dict[str, str]] = []
        concept_details: List[Dict[str, str]] = []
        concept_blocks = user_prompt.split("Concept ID:")
        for block in concept_blocks[1:]:
            cid_match = re.search(r"^\s*([^\n]+)", block)
            name_match = re.search(r"Name:\s*([^\n(]+)", block)
            src_match = re.search(r"Source Citation:\s*([^\n]+)", block)
            lens_match = re.search(r"Reflective Lens:\s*([^\n]+)", block)
            defn_match = re.search(r"Definition:\s*([^\n]+)", block)
            if cid_match and name_match and src_match:
                cid = cid_match.group(1).strip()
                name = name_match.group(1).strip()
                src = src_match.group(1).strip()
                lens = lens_match.group(1).strip() if lens_match else ""
                defn = defn_match.group(1).strip() if defn_match else ""
                grounded_concepts.append({
                    "concept_id": cid,
                    "name": name,
                    "source": src,
                })
                concept_details.append({
                    "concept_id": cid,
                    "name": name,
                    "source": src,
                    "lens": lens,
                    "definition": defn,
                })

        # Fallback if no concept pattern matched in prompt
        if not grounded_concepts:
            grounded_concepts.append({
                "concept_id": "cognitive_reframing",
                "name": "Cognitive Reframing",
                "source": "Beck, A. T. (1979). Cognitive Therapy of Depression; Clark, D. A. (2014). Cognitive Restructuring.",
            })
            concept_details.append({
                "concept_id": "cognitive_reframing",
                "name": "Cognitive Reframing",
                "source": "Beck, A. T. (1979). Cognitive Therapy of Depression; Clark, D. A. (2014). Cognitive Restructuring.",
                "lens": "Helps a journaler reflect on rigid interpretations of distressing events, asking whether another reasonable interpretation exists that is grounded in evidence.",
                "definition": "The practice of identifying automatic interpretive biases and consciously considering alternate, balanced perspectives on an event or situation.",
            })

        # 2. Extract observations and metadata from prompt
        obs_list: List[str] = []
        for section in ["### 5. LONGITUDINAL TRAJECTORIES", "### 2. OBSERVED EMOTION PATTERNS", "### 3. ACTIVE THEMATIC AREAS"]:
            if section in user_prompt:
                sec_part = user_prompt.split(section)[1].split("###")[0]
                lines = [l.strip().lstrip("- ").strip() for l in sec_part.splitlines() if l.strip().startswith("-")]
                obs_list.extend(lines)

        span_match = re.search(r"Timespan:\s*(\d+)\s*days", user_prompt)
        entries_match = re.search(r"Total Entries:\s*(\d+)", user_prompt)
        span_str = f"spanning {span_match.group(1)} days" if span_match else "over this period"
        entry_str = f"across your {entries_match.group(1)} recent journal entries" if entries_match else "across your entries"

        primary_concept = concept_details[0]
        concept_name = primary_concept["name"]
        raw_lens = primary_concept.get("lens") or primary_concept.get("definition") or "examining our responses to daily events"
        # Make explanation accessible
        accessible_lens = raw_lens.rstrip(".")
        if accessible_lens and accessible_lens[0].isupper():
            accessible_lens = accessible_lens[0].lower() + accessible_lens[1:]

        # 3. Construct Paragraph 1: Specific, verified observation from longitudinal data
        if obs_list:
            first_obs = obs_list[0].rstrip(".")
            p1 = (
                f"Looking at your writing {entry_str} {span_str}, an observable pattern emerges: {first_obs}."
            )
            if len(obs_list) > 1 and len(obs_list[1]) > 5:
                second_obs = obs_list[1].rstrip(".")
                second_lower = second_obs[0].lower() + second_obs[1:] if len(second_obs) > 1 else second_obs
                p1 += f" In parallel, your reflections also highlight {second_lower}."
        else:
            p1 = (
                f"Across your writing {entry_str} {span_str}, an observable trajectory shows shifting patterns in your emotional tone and recurring focal themes. "
                f"These shifts provide a tangible timeline of how your attention and energy have evolved."
            )

        # 4. Construct Paragraph 2: Accessible conceptual connection
        p2 = (
            f"This observed pattern resonates with the principles of {concept_name}. "
            f"In practical terms, this concept focuses on {accessible_lens}. "
            f"Viewed through this lens, the fluctuations in your entries may reflect an ongoing effort to balance external responsibilities with your personal capacity."
        )

        # 5. Construct Paragraph 3: Thoughtful contemplation ending with exactly ONE open question
        p3 = (
            f"One meaningful possibility to explore is what these moments reveal about what felt most grounding versus draining for you. "
            f"What aspects of your current daily routine feel most within your immediate sphere of influence?"
        )

        reflection_body = f"{p1}\n\n{p2}\n\n{p3}"

        return json.dumps({
            "reflection": reflection_body,
            "grounded_concepts": grounded_concepts,
            "confidence": 0.70,
            "disclaimer": (
                "This reflection describes observable patterns in journal text for personal contemplation. "
                "It does not constitute psychological, psychiatric, or medical advice or diagnosis."
            ),
        })


# ==============================================================================
# MAIN REFLECTION GENERATOR
# ==============================================================================

class ReflectionGenerator:
    """End-to-end grounded reflection orchestrator with pre-policy and post-validation."""

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        policy: Optional[ReflectionPolicy] = None,
        validator: Optional[Any] = None,
    ) -> None:
        """Initialize reflection generator.

        Args:
            client: LLMClient instance (defaults to OpenAIClient or MockLLMClient if no key).
            policy: ReflectionPolicy instance (defaults to standard policy).
            validator: ReflectionValidator instance (defaults to standard validator).
        """
        self.policy = policy or ReflectionPolicy()

        # Import validator lazily to avoid circular imports
        if validator is None:
            from ml.reflection.validator import ReflectionValidator
            self.validator = ReflectionValidator()
        else:
            self.validator = validator

        # Set up client
        if client is not None:
            self.client = client
        else:
            # Check if environment key is present; otherwise default to MockLLMClient
            env_key = os.environ.get("REFLECTION_API_KEY") or os.environ.get("OPENAI_API_KEY")
            if env_key:
                self.client = OpenAIClient(api_key=env_key)
            else:
                self.client = MockLLMClient()

    def generate_reflection(
        self,
        input_data: Union[ReflectionInput, Dict[str, Any]],
        current_timestamp: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """Execute the full reflection generation lifecycle.

        Args:
            input_data: ReflectionInput or equivalent dictionary.
            current_timestamp: Optional reference timestamp for rate limiting.

        Returns:
            Structured dictionary with generation result, status, and metadata.
        """
        # 1. Parse input contract
        if isinstance(input_data, dict):
            reflection_input = ReflectionInput.from_dict(input_data)
        elif isinstance(input_data, ReflectionInput):
            reflection_input = input_data
        else:
            raise TypeError(f"Expected ReflectionInput or dict, got {type(input_data)}")

        # 2. Pre-generation Policy Check
        entry_count = reflection_input.data_summary.get("entry_count", 0)
        span_days = reflection_input.data_summary.get("span_days", 0)
        policy_result = self.policy.evaluate(
            entry_count=entry_count,
            span_days=span_days,
            retrieved_concepts=reflection_input.retrieved_concepts,
            past_reflection_timestamps=reflection_input.past_reflection_timestamps,
            current_timestamp=current_timestamp,
        )

        if not policy_result.allowed:
            return {
                "status": "policy_rejected",
                "reason": policy_result.reason,
                "message": policy_result.message,
                "details": policy_result.details,
                "reflection": None,
                "grounded_concepts": [],
                "confidence": 0.0,
                "disclaimer": None,
            }

        # 3. Build prompts
        prompts = build_reflection_prompt(reflection_input)

        # 4. Invoke LLM client
        try:
            raw_response = self.client.generate(
                system_prompt=prompts["system"],
                user_prompt=prompts["user"],
            )
        except Exception as e:
            return {
                "status": "service_unavailable",
                "reason": "generation_failed",
                "message": f"Language model service unavailable: {str(e)}",
                "reflection": None,
                "grounded_concepts": [],
                "confidence": 0.0,
                "disclaimer": None,
            }

        # 5. Parse JSON output
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "status": "invalid_model_output",
                "reason": "json_decode_error",
                "message": "Model response was not valid JSON.",
                "raw_output": raw_response[:200],
                "reflection": None,
                "grounded_concepts": [],
                "confidence": 0.0,
                "disclaimer": None,
            }

        # 6. Post-generation Confidence Capping (Max 0.80)
        raw_confidence = parsed.get("confidence", 0.5)
        try:
            numeric_conf = float(raw_confidence)
            capped_confidence = min(numeric_conf, MAX_CONFIDENCE_CAP)
        except (ValueError, TypeError):
            capped_confidence = 0.50
        parsed["confidence"] = round(capped_confidence, 2)

        # 7. Post-generation Validation
        val_result = self.validator.validate(parsed, reflection_input)
        if not val_result.is_valid:
            return {
                "status": "validation_failed",
                "reason": "safety_or_grounding_violation",
                "message": "Generated reflection failed post-generation safety validation.",
                "validation_errors": val_result.errors,
                "reflection": None,
                "grounded_concepts": [],
                "confidence": 0.0,
                "disclaimer": None,
            }

        # Extract top 1-3 actual Phase 3 observations for audit / input_summary
        top_observations: List[str] = []
        if reflection_input.temporal_patterns:
            top_observations.extend([str(p) for p in reflection_input.temporal_patterns[:3]])
        elif reflection_input.emotion_patterns:
            top_observations.extend([str(p) for p in reflection_input.emotion_patterns[:2]])
            if reflection_input.theme_patterns:
                top_observations.append(str(reflection_input.theme_patterns[0]))
        elif reflection_input.theme_patterns:
            top_observations.extend([str(p) for p in reflection_input.theme_patterns[:3]])

        # 8. Success: Return fully verified reflection with rich audit metadata
        return {
            "status": "success",
            "reflection": parsed.get("reflection", "").strip(),
            "grounded_concepts": parsed.get("grounded_concepts", []),
            "confidence": parsed["confidence"],
            "disclaimer": parsed.get("disclaimer", "").strip(),
            "audit": {
                "policy_passed": True,
                "validated": True,
                "entry_count": entry_count,
                "span_days": span_days,
                "observations": top_observations[:3],
                "concept_ids_used": [c.get("concept_id") for c in parsed.get("grounded_concepts", [])],
            },
        }
