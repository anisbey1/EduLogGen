"""Temporal metrics (SAD §29E.1)."""

from __future__ import annotations

from typing import Any, ClassVar

from eduloggen.analysis import describe, interevent_times
from eduloggen.models import Dataset
from eduloggen.validation.base import BaseMetric, Category, Direction, ValidationContext
from eduloggen.validation.distances import ks_statistic

__all__ = ["InterEventTimeKS"]


class InterEventTimeKS(BaseMetric):
    """KS statistic between within-session inter-event times."""

    name: ClassVar[str] = "interevent_time_ks"
    category: ClassVar[Category] = "temporal"
    direction: ClassVar[Direction] = "lower_better"
    description = "Largest gap between inter-event time CDFs"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> tuple[float | None, dict[str, Any]]:
        a, b = interevent_times(real), interevent_times(synthetic)
        if not a or not b:
            return None, {"reason": "a dataset has no multi-event sessions"}
        return ks_statistic(a, b), {
            "real_median_s": describe(a).median,
            "synthetic_median_s": describe(b).median,
        }
