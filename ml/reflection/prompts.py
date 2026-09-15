"""Prompt construction and templating for Pensieve Phase 5 Grounded Reflection.

Defines the system instructions, ethical guidelines, uncertainty-aware framing,
and user prompt construction for structured language model generation.
"""

import json
from typing import Any, Dict, List


SYSTEM_INSTRUCTION = """You are Pensieve's reflective journaling assistant. Your purpose is to synthesize observable longitudinal writing patterns into a thoughtful, concise, non-diagnostic reflection grounded in established psychological and philosophical concepts.

CRITICAL SAFETY & ETHICAL RULES:
1. OBSERVABLE PATTERNS ONLY: Base your reflection exclusively on the provided statistical patterns (emotions, themes, linguistic indicators, and temporal trajectories). Do not assume unstated facts.
2. GROUNDING BOUNDARY: Use ONLY the concepts explicitly retrieved and provided in the prompt. You are strictly forbidden from introducing outside psychological theories, diagnostic classifications (e.g. DSM/ICD categories), attachment styles, trauma frameworks, or neurodevelopmental labels (e.g. ADHD) unless provided in the retrieved concepts.
3. PRESERVE UNCERTAINTY & HEDGING: You must never state conclusions with certainty. Always use descriptive, tentative hedging language (e.g., "may suggest", "could reflect", "resembles", "one possibility is", "might indicate", "appears to").
4. ZERO DIAGNOSIS: You are not a clinician, psychologist, or psychiatrist. You must NEVER diagnose the user or make clinical claims. Never say: "You have depression", "You have anxiety", "You are clinically burned out", "You suffer from", or "This proves...".
5. ZERO MEDICAL ADVICE: Never recommend medication, dosages, clinical therapies, psychiatric treatment, or medical interventions.
6. AVOID DIRECTIVE ADVICE: Focus on gentle perspective-taking and open-ended reflective questions rather than prescriptive life instructions.
7. CITATION INTEGRITY: When referencing a grounded concept, accurately cite the concept name and source as provided. Never invent citations, authors, or research findings.
8. CONFIDENCE: Provide a realistic, modest confidence estimate reflecting the observational nature of textual pattern analysis (maximum 0.80).
9. MANDATORY DISCLAIMER: Every reflection must conclude with an explicit non-diagnostic disclaimer.

REQUIRED OUTPUT FORMAT:
You must respond with a strictly valid JSON object adhering to this schema:
{
    "reflection": "A 2 to 4 paragraph thoughtful, grounded reflection synthesizing observed trajectories with the retrieved concept(s), ending with an open question.",
    "grounded_concepts": [
        {
            "concept_id": "id_matching_retrieved_concept",
            "name": "Name of concept",
            "source": "Exact citation from retrieved concept"
        }
    ],
    "confidence": 0.65,
    "disclaimer": "This reflection describes observable patterns in journal text for personal contemplation. It does not constitute psychological, psychiatric, or medical advice or diagnosis."
}
"""


def format_patterns_for_prompt(
    data_summary: Dict[str, Any],
    emotion_patterns: List[Any],
    theme_patterns: List[Any],
    linguistic_patterns: Any,
    temporal_patterns: List[Any],
    recurring_patterns: List[Any],
    retrieved_concepts: List[Dict[str, Any]],
    evidence_excerpts: List[str] = None,
) -> str:
    """Format structured Phase 1–4 signals into an interpretable text payload for the user prompt.

    Args:
        data_summary: Entry count and time span.
        emotion_patterns: Salient emotions and trajectories.
        theme_patterns: Active thematic clusters.
        linguistic_patterns: Style and structure indicators.
        temporal_patterns: Chronological trend observations.
        recurring_patterns: Lexical or thematic recurrence.
        retrieved_concepts: Concepts retrieved by Phase 4.
        evidence_excerpts: Optional limited excerpts (if any).

    Returns:
        Formatted prompt string.
    """
    sections: List[str] = []

    # 1. Summary
    entries = data_summary.get("entry_count", 0)
    span = data_summary.get("span_days", 0)
    sections.append(f"### 1. JOURNAL HISTORY SUMMARY\n- Total Entries: {entries}\n- Timespan: {span} days")

    # 2. Emotion Trajectories (Phase 1 & 3)
    if emotion_patterns:
        emo_str = "\n".join(f"- {p}" for p in emotion_patterns)
        sections.append(f"### 2. OBSERVED EMOTION PATTERNS\n{emo_str}")

    # 3. Themes (Phase 2 & 3)
    if theme_patterns:
        theme_str = "\n".join(f"- {p}" for p in theme_patterns)
        sections.append(f"### 3. ACTIVE THEMATIC AREAS\n{theme_str}")

    # 4. Linguistic Characteristics (Phase 3)
    if linguistic_patterns:
        if isinstance(linguistic_patterns, dict):
            ling_str = "\n".join(f"- {k}: {v}" for k, v in linguistic_patterns.items())
        else:
            ling_str = "\n".join(f"- {p}" for p in linguistic_patterns)
        sections.append(f"### 4. LINGUISTIC & STYLISTIC INDICATORS\n{ling_str}")

    # 5. Temporal & Recurring Patterns (Phase 3)
    combined_trends = []
    if temporal_patterns:
        combined_trends.extend(temporal_patterns)
    if recurring_patterns:
        combined_trends.extend(recurring_patterns)
    if combined_trends:
        trend_str = "\n".join(f"- {t}" for t in combined_trends)
        sections.append(f"### 5. LONGITUDINAL TRAJECTORIES\n{trend_str}")

    # 6. Retrieved Grounding Concepts (Phase 4) - Strict boundary
    concept_lines = []
    for c in retrieved_concepts:
        cid = c.get("concept_id") or c.get("id", "")
        name = c.get("name", "")
        cat = c.get("category", "")
        defn = c.get("definition", "")
        expl = c.get("explanation", "")
        src = c.get("source", "")
        caut = c.get("cautions", [])
        caut_str = "; ".join(caut) if isinstance(caut, list) else str(caut)
        concept_lines.append(
            f"Concept ID: {cid}\n"
            f"Name: {name} ({cat})\n"
            f"Definition: {defn}\n"
            f"Reflective Lens: {expl}\n"
            f"Source Citation: {src}\n"
            f"Ethical Cautions: {caut_str}\n"
        )
    sections.append("### 6. RETRIEVED GROUNDING CONCEPTS (Strict Boundary)\n" + "\n---\n".join(concept_lines))

    # 7. Optional limited excerpts
    if evidence_excerpts:
        excerpts_str = "\n".join(f"\"{ex}\"" for ex in evidence_excerpts)
        sections.append(f"### 7. SELECTED EVIDENCE EXCERPTS (Privacy Filtered)\n{excerpts_str}")

    sections.append(
        "Please generate a grounded, uncertainty-aware reflection in valid JSON format matching the schema."
    )
    return "\n\n".join(sections)


def build_reflection_prompt(reflection_input: Any) -> Dict[str, str]:
    """Build the complete system and user prompt pair from a ReflectionInput object.

    Args:
        reflection_input: Instance of ReflectionInput.

    Returns:
        Dictionary with 'system' and 'user' prompt strings.
    """
    user_prompt = format_patterns_for_prompt(
        data_summary=reflection_input.data_summary,
        emotion_patterns=reflection_input.emotion_patterns,
        theme_patterns=reflection_input.theme_patterns,
        linguistic_patterns=reflection_input.linguistic_patterns,
        temporal_patterns=reflection_input.temporal_patterns,
        recurring_patterns=reflection_input.recurring_patterns,
        retrieved_concepts=reflection_input.retrieved_concepts,
        evidence_excerpts=getattr(reflection_input, "evidence_excerpts", None),
    )
    return {
        "system": SYSTEM_INSTRUCTION.strip(),
        "user": user_prompt.strip(),
    }
