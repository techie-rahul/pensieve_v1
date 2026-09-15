"""Pensieve ML — Phase 5: Grounded Reflection Generation.

Provides privacy-first, grounded reflection synthesis from structured Phase 1–4 outputs.
Features a two-stage safety architecture:
1. Pre-generation Policy Engine (minimum entries, time-span, rate limits, grounding checks)
2. Post-generation Safety Validator (concept grounding integrity, non-diagnostic checks, hedging)
"""

from ml.reflection.policy import (
    ReflectionPolicy,
    PolicyCheckResult,
)
from ml.reflection.prompts import (
    SYSTEM_INSTRUCTION,
    build_reflection_prompt,
    format_patterns_for_prompt,
)
from ml.reflection.generator import (
    ReflectionInput,
    ReflectionGenerator,
    LLMClient,
    OpenAIClient,
    MockLLMClient,
    DEFAULT_REFLECTION_MODEL,
    MAX_CONFIDENCE_CAP,
)
from ml.reflection.validator import (
    ReflectionValidator,
    ValidationResult,
)

__all__ = [
    "ReflectionPolicy",
    "PolicyCheckResult",
    "SYSTEM_INSTRUCTION",
    "build_reflection_prompt",
    "format_patterns_for_prompt",
    "ReflectionInput",
    "ReflectionGenerator",
    "LLMClient",
    "OpenAIClient",
    "MockLLMClient",
    "DEFAULT_REFLECTION_MODEL",
    "MAX_CONFIDENCE_CAP",
    "ReflectionValidator",
    "ValidationResult",
]
