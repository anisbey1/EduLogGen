"""Metric contract and result types (PRD §12.3, §12.5, SAD §14.6, §29A.5)."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar, Literal

from eduloggen.core import ValidationError
from eduloggen.models import Dataset

__all__ = [
    "BaseMetric",
    "Category",
    "Direction",
    "MetricResult",
    "Status",
    "ValidationContext",
]

Direction = Literal["lower_better", "higher_better"]
Category = Literal[
    "marginal", "session_structure", "temporal", "sequential", "navigation",
    "privacy", "utility",
]  # fmt: skip
Status = Literal["pass", "fail", "warn", "skip", "info"]
"""``info``: computed but no threshold set; ``skip``: could not be computed."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationContext:
    """Settings shared by all metrics of one validation run.

    Attributes:
        seed: Seed for any subsampling a metric performs.
        params: Metric name to metric-specific parameters.
    """

    seed: int = 0
    params: Mapping[str, Mapping[str, Any]] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    def params_for(self, metric: str) -> Mapping[str, Any]:
        """Parameters for one metric (empty if none)."""
        return self.params.get(metric, MappingProxyType({}))


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricResult:
    """Outcome of one metric.

    Attributes:
        name: Stable metric id (e.g. ``"event_type_tvd"``).
        category: Metric category.
        direction: Whether lower or higher values are better.
        value: Metric value, or ``None`` if it could not be computed.
        threshold: Pass/fail threshold, if set.
        status: Evaluation status.
        details: JSON-compatible extra information.
    """

    name: str
    category: Category
    direction: Direction
    value: float | None
    threshold: float | None = None
    status: Status = "info"
    details: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    def evaluate(self, threshold: float | None) -> MetricResult:
        """Return a copy with ``threshold`` applied to set the status.

        Lower-better metrics pass when ``value <= threshold``; higher-better
        metrics pass when ``value >= threshold``. Uncomputed metrics stay
        ``skip``; metrics without a threshold are ``info``.
        """
        if self.value is None:
            return replace(self, threshold=threshold, status="skip")
        if threshold is None:
            return replace(self, threshold=None, status="info")
        if self.direction == "lower_better":
            passed = self.value <= threshold
        else:
            passed = self.value >= threshold
        return replace(self, threshold=threshold, status="pass" if passed else "fail")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "name": self.name,
            "category": self.category,
            "direction": self.direction,
            "value": self.value,
            "threshold": self.threshold,
            "status": self.status,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetricResult:
        """Restore a result serialized with :meth:`to_dict`."""
        return cls(
            name=data["name"],
            category=data["category"],
            direction=data["direction"],
            value=data["value"],
            threshold=data.get("threshold"),
            status=data.get("status", "info"),
            details=MappingProxyType(dict(data.get("details") or {})),
        )


class BaseMetric(ABC):
    """A real-versus-synthetic comparison.

    Subclasses set the class attributes and implement :meth:`_compute`,
    returning the value (``None`` if not computable) and details.
    """

    name: ClassVar[str]
    category: ClassVar[Category]
    direction: ClassVar[Direction]
    description: ClassVar[str] = ""

    def compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> MetricResult:
        """Compare two sessionized datasets.

        Raises:
            ValidationError: If the metric returns a non-finite value.
        """
        value, details = self._compute(real, synthetic, context)
        if value is not None and not math.isfinite(value):
            raise ValidationError(
                f"metric {self.name} produced a non-finite value",
                code="validation_invalid_value",
                context={"metric": self.name},
            )
        return MetricResult(
            name=self.name,
            category=self.category,
            direction=self.direction,
            value=value,
            details=MappingProxyType(dict(details)),
        )

    @abstractmethod
    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> tuple[float | None, dict[str, Any]]:
        """Return the metric value and JSON-compatible details."""
