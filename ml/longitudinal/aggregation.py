"""Chronological sorting, validation, and time-window aggregation for journal entries.

Enforces minimum data safeguards:
- Requires >= 3 entries for longitudinal analysis.
- Flags histories spanning < 7 days as brief/preliminary.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_timestamp(ts: Union[str, datetime]) -> datetime:
    """Parse various timestamp representations into a timezone-aware UTC datetime."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            return ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc)

    # String parsing
    ts_str = str(ts).strip()
    # Support ISO formats with or without Z, or simple YYYY-MM-DD
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str.replace("Z", "+0000"), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue

    # Fallback to fromisoformat
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception as exc:
        raise ValueError(f"Unable to parse timestamp string '{ts_str}' into datetime") from exc


class JournalAggregator:
    """Manages chronological validation, grouping, and window aggregation of journal entries."""

    def __init__(
        self,
        min_entries_required: int = 3,
        min_span_days: int = 7,
    ) -> None:
        """Initialize aggregator configuration and safeguards."""
        self.min_entries_required = min_entries_required
        self.min_span_days = min_span_days

    def validate_and_sort(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate presence of required fields and sort chronologically ascending.

        Args:
            entries: List of journal entry dicts containing at least 'id', 'text', 'timestamp'.

        Returns:
            Sorted list of entries with standardized 'parsed_timestamp' attribute.
        """
        validated: List[Dict[str, Any]] = []

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if "text" not in entry or "timestamp" not in entry:
                continue

            parsed_dt = parse_timestamp(entry["timestamp"])
            entry_copy = dict(entry)
            entry_copy["parsed_timestamp"] = parsed_dt
            validated.append(entry_copy)

        # Sort strictly ascending by parsed datetime
        validated.sort(key=lambda x: x["parsed_timestamp"])
        return validated

    def check_safeguards(self, sorted_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify whether data meets minimum requirements for meaningful longitudinal analysis."""
        num_entries = len(sorted_entries)
        if num_entries < self.min_entries_required:
            return {
                "sufficient_data": False,
                "reason": (
                    f"Insufficient data: longitudinal analysis requires at least {self.min_entries_required} "
                    f"dated entries, but only {num_entries} was provided."
                ),
                "timespan_days": 0.0,
                "brevity_warning": False,
            }

        start_time = sorted_entries[0]["parsed_timestamp"]
        end_time = sorted_entries[-1]["parsed_timestamp"]
        timespan_days = round((end_time - start_time).total_seconds() / 86400.0, 2)

        brevity_warning = timespan_days < self.min_span_days
        warning_msg = (
            f"History spans {timespan_days:.1f} days (< {self.min_span_days} days). "
            "Temporal patterns represent a short window and should be interpreted cautiously."
            if brevity_warning
            else None
        )

        return {
            "sufficient_data": True,
            "timespan_days": timespan_days,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "brevity_warning": brevity_warning,
            "warning_message": warning_msg,
        }

    def aggregate_into_windows(
        self,
        entries: List[Dict[str, Any]],
        window_type: str = "weekly",
        custom_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Aggregate sorted entries into chronological time windows.

        Args:
            entries: List of journal entries.
            window_type: 'weekly' (7 days), 'monthly' (30 days), or 'custom'.
            custom_days: Window duration in days if window_type == 'custom'.

        Returns:
            Dictionary containing aggregation status, time span, and list of window objects.
        """
        sorted_entries = self.validate_and_sort(entries)
        safeguards = self.check_safeguards(sorted_entries)

        if not safeguards["sufficient_data"]:
            return {
                "status": "insufficient_data",
                "message": safeguards["reason"],
                "total_entries": len(sorted_entries),
                "windows": [],
                "safeguards": safeguards,
            }

        if window_type == "weekly":
            window_delta = timedelta(days=7)
        elif window_type == "monthly":
            window_delta = timedelta(days=30)
        elif window_type == "custom" and custom_days and custom_days > 0:
            window_delta = timedelta(days=custom_days)
        else:
            window_delta = timedelta(days=7)

        start_time = sorted_entries[0]["parsed_timestamp"]
        end_time = sorted_entries[-1]["parsed_timestamp"]

        windows: List[Dict[str, Any]] = []
        current_window_start = start_time
        window_idx = 0

        while current_window_start <= end_time:
            current_window_end = current_window_start + window_delta
            # Gather entries in [current_window_start, current_window_end)
            # For the last boundary, include equality if at or past end_time
            window_entries = [
                e for e in sorted_entries
                if current_window_start <= e["parsed_timestamp"] < current_window_end
                or (e["parsed_timestamp"] == end_time and current_window_end >= end_time)
            ]

            if window_entries:
                # Deduplicate entries that might touch exact border
                seen_ids = set()
                unique_entries = []
                for e in window_entries:
                    if e["id"] not in seen_ids:
                        seen_ids.add(e["id"])
                        unique_entries.append(e)

                windows.append({
                    "window_index": window_idx,
                    "label": f"Window {window_idx + 1} ({current_window_start.strftime('%Y-%m-%d')} to {current_window_end.strftime('%Y-%m-%d')})",
                    "start_date": current_window_start.isoformat(),
                    "end_date": current_window_end.isoformat(),
                    "entry_count": len(unique_entries),
                    "entry_ids": [e["id"] for e in unique_entries],
                    "entries": unique_entries,
                })
                window_idx += 1

            current_window_start = current_window_end

        return {
            "status": "success",
            "window_type": window_type,
            "window_duration_days": window_delta.days,
            "total_entries": len(sorted_entries),
            "num_windows": len(windows),
            "safeguards": safeguards,
            "windows": windows,
        }
