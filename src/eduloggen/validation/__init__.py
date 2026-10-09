"""Real-versus-synthetic validation (PRD §12, SAD §14).

Typical use::

    from eduloggen.validation import validate

    report = validate(real, synthetic, thresholds={"event_type_tvd": 0.1})
    print(report.to_markdown())
    report.passed

Built-in metrics cover marginal, session-structure, temporal, sequential,
navigation, and privacy categories. Custom metrics subclass
:class:`BaseMetric` and are added with :func:`register_metric`.
"""

from __future__ import annotations

from eduloggen.validation.base import (
    BaseMetric,
    Category,
    Direction,
    MetricResult,
    Status,
    ValidationContext,
)
from eduloggen.validation.detailed import DetailedComparison, compare_detailed
from eduloggen.validation.distances import (
    jensen_shannon,
    ks_statistic,
    normalized_edit_distance,
    total_variation,
    wasserstein1,
)
from eduloggen.validation.metrics import BUILTIN_METRICS
from eduloggen.validation.registry import (
    DEFAULT_METRICS,
    available_metrics,
    get_metric,
    register_metric,
    unregister_metric,
)
from eduloggen.validation.report import ReportStatus, ValidationReport
from eduloggen.validation.service import validate

__all__ = [
    "BUILTIN_METRICS",
    "DEFAULT_METRICS",
    "BaseMetric",
    "Category",
    "DetailedComparison",
    "Direction",
    "MetricResult",
    "ReportStatus",
    "Status",
    "ValidationContext",
    "ValidationReport",
    "available_metrics",
    "compare_detailed",
    "get_metric",
    "jensen_shannon",
    "ks_statistic",
    "normalized_edit_distance",
    "register_metric",
    "total_variation",
    "unregister_metric",
    "validate",
    "wasserstein1",
]
