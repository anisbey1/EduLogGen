"""Marginal and session-structure metrics (SAD §29E.1)."""

from __future__ import annotations

from collections import Counter
from typing import Any, ClassVar

from eduloggen.analysis import session_durations, session_sequences
from eduloggen.models import Dataset
from eduloggen.validation.base import BaseMetric, Category, Direction, ValidationContext
from eduloggen.validation.distances import (
    jensen_shannon,
    ks_statistic,
    total_variation,
    wasserstein1,
)

__all__ = [
    "ActivityJSD",
    "EventTypeTVD",
    "SessionDurationW1",
    "SessionLengthKS",
]

Result = tuple[float | None, dict[str, Any]]


def _empty(reason: str) -> Result:
    return None, {"reason": reason}


class EventTypeTVD(BaseMetric):
    """Total variation distance between event-type frequencies."""

    name: ClassVar[str] = "event_type_tvd"
    category: ClassVar[Category] = "marginal"
    direction: ClassVar[Direction] = "lower_better"
    description = "Share of event-type probability mass that must move (0 = same)"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        p = Counter(e.event_type for e in real.events)
        q = Counter(e.event_type for e in synthetic.events)
        if not p or not q:
            return _empty("a dataset has no events")
        return total_variation(p, q), {
            "real_types": len(p),
            "synthetic_types": len(q),
            "unseen_in_synthetic": sorted(p.keys() - q.keys()),
        }


class ActivityJSD(BaseMetric):
    """Jensen-Shannon divergence between activity frequencies."""

    name: ClassVar[str] = "activity_jsd"
    category: ClassVar[Category] = "marginal"
    direction: ClassVar[Direction] = "lower_better"
    description = "Jensen-Shannon divergence (bits) of activity_id frequencies"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        p = Counter(e.activity_id for e in real.events)
        q = Counter(e.activity_id for e in synthetic.events)
        if not p or not q:
            return _empty("a dataset has no events")
        return jensen_shannon(p, q), {
            "real_activities": len(p),
            "synthetic_activities": len(q),
        }


class SessionLengthKS(BaseMetric):
    """KS statistic between session-length distributions."""

    name: ClassVar[str] = "session_length_ks"
    category: ClassVar[Category] = "session_structure"
    direction: ClassVar[Direction] = "lower_better"
    description = "Largest gap between session-length CDFs"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        a = [float(len(s)) for s in session_sequences(real)]
        b = [float(len(s)) for s in session_sequences(synthetic)]
        if not a or not b:
            return _empty("a dataset has no sessions")
        return ks_statistic(a, b), {
            "real_mean": sum(a) / len(a),
            "synthetic_mean": sum(b) / len(b),
        }


class SessionDurationW1(BaseMetric):
    """Wasserstein-1 distance between session durations, in seconds."""

    name: ClassVar[str] = "session_duration_w1"
    category: ClassVar[Category] = "session_structure"
    direction: ClassVar[Direction] = "lower_better"
    description = "Earth mover's distance between session durations (seconds)"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> Result:
        a, b = session_durations(real), session_durations(synthetic)
        if not a or not b:
            return _empty("a dataset has no sessions")
        real_mean = sum(a) / len(a)
        distance = wasserstein1(a, b)
        return distance, {
            "unit": "seconds",
            "real_mean_s": real_mean,
            "synthetic_mean_s": sum(b) / len(b),
            "relative_to_real_mean": distance / real_mean if real_mean else None,
        }
