"""Post-generation safety and grounding validation for Pensieve Phase 5.

Verifies that model outputs strictly adhere to ethical constraints, non-diagnostic boundaries,
grounding boundaries, hedging requirements, and disclaimer standards.
"""

from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Set

from ml.reflection.policy import PolicyCheckResult


# Prohibited diagnostic assertions (regex patterns)
DIAGNOSTIC_PATTERNS = [
    r"\byou have (clinical |major )?(depression|anxiety|bipolar|adhd|ptsd|ocd)\b",
    r"\byou are (clinically |chronically )?(depressed|anxious|burned out)\b",
    r"\byou suffer from\b",
    r"\byou are diagnosed with\b",
    r"\bthis (proves|confirms) you have\b",
    r"\bclinical diagnosis\b",
    r"\byour pathology\b",
    r"\bdiagnostic conclusion\b",
]

# Prohibited medical advice and medication patterns
MEDICAL_ADVICE_PATTERNS = [
    r"\b(prescribe|prescription|medication|dosage|dose|mg daily)\b",
    r"\b(ssri|antidepressant|benzodiazepine|adderall|sertraline|fluoxetine)\b",
    r"\byou should start therapy with a psychiatrist\b",
    r"\bseek immediate medication\b",
]

# Common unauthorized external theories/frameworks that must not be hallucinated
UNAUTHORIZED_THEORY_KEYWORDS = [
    "attachment theory",
    "anxious attachment",
    "avoidant attachment",
    "borderline personality",
    "narcissistic personality",
    "bipolar disorder",
    "schizophrenia",
    "adhd",
    "generalized anxiety disorder",
    "major depressive disorder",
    "trauma bonding",
    "complex ptsd",
]

# Required uncertainty/hedging markers
HEDGING_MARKERS = [
    "may",
    "could",
    "might",
    "suggest",
    "suggests",
    "reflect",
    "reflects",
    "resemble",
    "resembles",
    "possibility",
    "perhaps",
    "appear",
    "appears",
    "perspective",
    "tendency",
    "potential",
]


@dataclass
class ValidationResult:
    """Outcome of post-generation safety validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class ReflectionValidator:
    """Validates generated reflections against safety, grounding, and ethical rules."""

    def __init__(self, min_word_count: int = 25, max_word_count: int = 600) -> None:
        """Initialize validator settings.

        Args:
            min_word_count: Minimum words in reflection text.
            max_word_count: Maximum words in reflection text.
        """
        self.min_word_count = min_word_count
        self.max_word_count = max_word_count

    def validate(
        self,
        output: Dict[str, Any],
        reflection_input: Any,
    ) -> ValidationResult:
        """Validate model output against input boundaries and safety constraints.

        Args:
            output: Parsed model response dictionary.
            reflection_input: The ReflectionInput instance provided to the generator.

        Returns:
            ValidationResult indicating validity and any error messages.
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Schema check
        required_keys = ["reflection", "grounded_concepts", "confidence", "disclaimer"]
        for key in required_keys:
            if key not in output:
                errors.append(f"Missing required schema field: '{key}'.")

        if errors:
            return ValidationResult(is_valid=False, errors=errors)

        reflection_text = str(output.get("reflection", "")).strip()
        grounded_concepts = output.get("grounded_concepts", [])
        confidence = output.get("confidence", 0.0)
        disclaimer = str(output.get("disclaimer", "")).strip()

        # 2. Length check
        words = reflection_text.split()
        if len(words) < self.min_word_count:
            errors.append(
                f"Reflection too short ({len(words)} words; minimum {self.min_word_count})."
            )
        elif len(words) > self.max_word_count:
            errors.append(
                f"Reflection too long ({len(words)} words; maximum {self.max_word_count})."
            )

        # 3. Confidence cap check (<= 0.80)
        try:
            conf_val = float(confidence)
            if conf_val > 0.80:
                errors.append(f"Confidence score {conf_val} exceeds maximum allowed cap of 0.80.")
        except (ValueError, TypeError):
            errors.append("Invalid non-numeric confidence value.")

        # 4. Mandatory Disclaimer check
        if not disclaimer:
            errors.append("Missing mandatory non-diagnostic disclaimer.")
        else:
            lower_disc = disclaimer.lower()
            if "not" not in lower_disc or ("diagnosis" not in lower_disc and "medical" not in lower_disc and "psychological" not in lower_disc):
                errors.append("Disclaimer does not contain required non-diagnostic disclaimer phrasing.")

        # 5. Grounding Boundary & Concept Integrity
        # Extract valid concept IDs from input
        valid_input_concepts: Dict[str, Dict[str, Any]] = {}
        for c in getattr(reflection_input, "retrieved_concepts", []):
            cid = c.get("concept_id") or c.get("id")
            if cid:
                valid_input_concepts[cid] = c

        if not isinstance(grounded_concepts, list):
            errors.append("'grounded_concepts' must be a list.")
        else:
            for item in grounded_concepts:
                if not isinstance(item, dict):
                    errors.append("Items in 'grounded_concepts' must be dictionaries.")
                    continue
                cid = item.get("concept_id")
                if not cid or cid not in valid_input_concepts:
                    errors.append(
                        f"Grounded concept ID '{cid}' was NOT in the retrieved Phase 4 concepts. "
                        f"Valid concepts: {list(valid_input_concepts.keys())}."
                    )
                else:
                    # Check source citation integrity if provided
                    input_source = valid_input_concepts[cid].get("source", "").lower()
                    cited_source = item.get("source", "").lower()
                    # If cited source does not share words with input source, flag error
                    if cited_source and input_source:
                        cited_tokens = set(re.findall(r"\w{4,}", cited_source))
                        input_tokens = set(re.findall(r"\w{4,}", input_source))
                        if not cited_tokens.intersection(input_tokens):
                            errors.append(
                                f"Cited source for '{cid}' does not match supplied Phase 4 source."
                            )

        # 6. Unauthorized outside theories check
        reflection_lower = reflection_text.lower()
        # Find which unauthorized keywords are already part of supplied concepts (if any)
        allowed_framework_names = set()
        for c in valid_input_concepts.values():
            allowed_framework_names.add(c.get("name", "").lower())
            allowed_framework_names.add(c.get("id", "").lower())

        for keyword in UNAUTHORIZED_THEORY_KEYWORDS:
            if keyword in reflection_lower:
                # Check if it was part of an allowed concept name
                if not any(keyword in allowed_name for allowed_name in allowed_framework_names):
                    errors.append(
                        f"Reflection introduced unretrieved external psychological framework: '{keyword}'."
                    )

        # 7. Non-diagnostic phrasing check
        for pattern in DIAGNOSTIC_PATTERNS:
            match = re.search(pattern, reflection_lower, re.IGNORECASE)
            if match:
                errors.append(f"Prohibited diagnostic language detected: '{match.group(0)}'.")

        # 8. Medical advice check
        for pattern in MEDICAL_ADVICE_PATTERNS:
            match = re.search(pattern, reflection_lower, re.IGNORECASE)
            if match:
                errors.append(f"Prohibited medical advice detected: '{match.group(0)}'.")

        # 9. Hedging & Uncertainty requirement
        has_hedging = any(re.search(rf"\b{marker}\b", reflection_lower) for marker in HEDGING_MARKERS)
        if not has_hedging:
            errors.append(
                "Reflection lacks required uncertainty-aware hedging language (e.g. 'may', 'could', 'suggest', 'resemble')."
            )

        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
        )
