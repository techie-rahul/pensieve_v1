"""Pensieve ML - Longitudinal Analysis Module.

Phase 3 implementation for temporal aggregation, trend detection, and pattern synthesis.
"""

from ml.longitudinal.aggregation import JournalAggregator, parse_timestamp
from ml.longitudinal.trends import TrendDetector
from ml.longitudinal.patterns import PatternSynthesizer, analyze_journal_history

__all__ = [
    "JournalAggregator",
    "parse_timestamp",
    "TrendDetector",
    "PatternSynthesizer",
    "analyze_journal_history",
]
