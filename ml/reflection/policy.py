"""Safety and Policy Engine for Pensieve Phase 5 Grounded Reflection Generation.

The policy engine evaluates whether a reflection request satisfies all strict privacy,
minimum-data, rate-limiting, and grounding prerequisites BEFORE any LLM generation begins.

IMPORTANT ETHICAL AND SAFETY PRINCIPLES:
1. Do not infer longitudinal trends from fewer than 3 entries or shorter than 7 days.
2. Limit reflections to a maximum of 2 in any rolling 7-day window to prevent AI-dependency.
3. Require explicit Phase 4 concept grounding before allowing reflection generation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union


@dataclass
class PolicyCheckResult:
    """Outcome of pre-generation safety and eligibility evaluation."""

    allowed: bool
    reason: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "message": self.message,
            "details": self.details,
        }


def _parse_iso_timestamp(ts: Union[str, datetime]) -> datetime:
    """Parse ISO timestamp string or return timezone-aware datetime."""
    if isinstance(ts, datetime):
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)
    # Handle string timestamps, including 'Z' notation
    clean_ts = ts.replace("Z", "+00:00")
    dt = datetime.fromisoformat(clean_ts)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


class ReflectionPolicy:
    """Pre-generation policy engine enforcing Pensieve's longitudinal and safety constraints."""

    def __init__(
        self,
        min_entries: int = 3,
        min_span_days: int = 7,
        max_reflections_per_7_days: int = 2,
    ) -> None:
        """Initialize policy parameters.

        Args:
            min_entries: Minimum number of journal entries required (default: 3).
            min_span_days: Minimum time span in days required for longitudinal reflection (default: 7).
            max_reflections_per_7_days: Maximum reflections allowed in a rolling 7-day window (default: 2).
        """
        self.min_entries = min_entries
        self.min_span_days = min_span_days
        self.max_reflections_per_7_days = max_reflections_per_7_days

    def evaluate(
        self,
        entry_count: int,
        span_days: int,
        retrieved_concepts: List[Dict[str, Any]],
        past_reflection_timestamps: Optional[List[str]] = None,
        current_timestamp: Optional[Union[str, datetime]] = None,
    ) -> PolicyCheckResult:
        """Evaluate pre-generation safety policies.

        Args:
            entry_count: Number of journal entries in the analysis window.
            span_days: Number of days spanned by the entries.
            retrieved_concepts: List of concept documents retrieved by Phase 4.
            past_reflection_timestamps: List of previous reflection generation timestamps.
            current_timestamp: Timestamp of the requested reflection (defaults to now).

        Returns:
            PolicyCheckResult indicating whether generation is allowed or why it was rejected.
        """
        # 1. Minimum entries safeguard
        if entry_count < self.min_entries:
            return PolicyCheckResult(
                allowed=False,
                reason="insufficient_entries",
                message=(
                    f"At least {self.min_entries} journal entries are required to generate a longitudinal "
                    f"reflection (received {entry_count}). Pensieve does not construct longitudinal summaries "
                    "from isolated observations."
                ),
                details={"entry_count": entry_count, "min_required": self.min_entries},
            )

        # 2. Minimum timespan safeguard
        if span_days < self.min_span_days:
            return PolicyCheckResult(
                allowed=False,
                reason="insufficient_timespan",
                message=(
                    f"At least {self.min_span_days} days of journal history are required to observe meaningful "
                    f"trajectories (history spans {span_days} days). Shorter periods do not represent sustained patterns."
                ),
                details={"span_days": span_days, "min_required": self.min_span_days},
            )

        # 3. Grounding requirement
        if not retrieved_concepts:
            return PolicyCheckResult(
                allowed=False,
                reason="insufficient_grounding",
                message=(
                    "No relevant psychological or philosophical concepts were retrieved by Phase 4. "
                    "Pensieve requires grounded concept anchors to prevent unconstrained generation."
                ),
                details={"retrieved_count": 0},
            )

        # 4. Reflection rate limiting (Max 2 per rolling 7 days)
        if past_reflection_timestamps:
            now_dt = _parse_iso_timestamp(current_timestamp) if current_timestamp else datetime.now(timezone.utc)
            rolling_window_seconds = 7 * 86400  # 7 days

            recent_reflections = 0
            for ts in past_reflection_timestamps:
                try:
                    past_dt = _parse_iso_timestamp(ts)
                    diff_seconds = (now_dt - past_dt).total_seconds()
                    if 0 <= diff_seconds <= rolling_window_seconds:
                        recent_reflections += 1
                except (ValueError, TypeError):
                    continue

            if recent_reflections >= self.max_reflections_per_7_days:
                return PolicyCheckResult(
                    allowed=False,
                    reason="rate_limit_exceeded",
                    message=(
                        f"Reflection rate limit reached ({recent_reflections} reflections generated in the "
                        f"past 7 days; maximum allowed is {self.max_reflections_per_7_days}). Pensieve encourages "
                        "personal integration and space between AI reflection summaries."
                    ),
                    details={
                        "recent_reflections": recent_reflections,
                        "limit": self.max_reflections_per_7_days,
                        "window_days": 7,
                    },
                )

        return PolicyCheckResult(
            allowed=True,
            reason=None,
            message="All pre-generation safety and eligibility checks passed.",
            details={
                "entry_count": entry_count,
                "span_days": span_days,
                "concept_count": len(retrieved_concepts),
            },
        )
