"""Score an anomaly detector against ground-truth annotations (Level 2, M1).

Predictions are either a set of flagged ids or a mapping of id to score.
Every id at the chosen level that has no anomaly annotation counts as
normal; ids missing from score predictions get the lowest score.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from eduloggen.core import ValidationError
from eduloggen.models import AnnotationLevel, Annotations, Dataset

__all__ = ["DetectionReport", "average_precision", "evaluate_detection", "roc_auc"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DetectionReport:
    """Detection quality at one level.

    Attributes:
        level: ``session`` or ``event`` (or ``learner``).
        n_items: Items at that level in the dataset.
        n_anomalous: Items with at least one anomaly annotation.
        tp, fp, fn, tn: Confusion counts at the decision threshold.
        precision, recall, f1, false_positive_rate: ``None`` when undefined.
        roc_auc, average_precision: Only for score predictions.
        recall_by_type: Anomaly type to share of its items flagged.
        recall_by_category: ``invalid_workflow`` / ``unusual_valid`` recall.
        threshold: Score threshold used (``None`` for flag predictions).
    """

    level: str
    n_items: int
    n_anomalous: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    false_positive_rate: float | None
    roc_auc: float | None = None
    average_precision: float | None = None
    recall_by_type: Mapping[str, float] = field(default_factory=dict, hash=False)
    recall_by_category: Mapping[str, float] = field(default_factory=dict, hash=False)
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "level": self.level,
            "n_items": self.n_items,
            "n_anomalous": self.n_anomalous,
            "confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "false_positive_rate": self.false_positive_rate,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "recall_by_type": dict(self.recall_by_type),
            "recall_by_category": dict(self.recall_by_category),
            "threshold": self.threshold,
        }

    def to_markdown(self) -> str:
        """Human-readable summary."""

        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:.3f}"

        lines = [
            f"# Detection evaluation ({self.level} level)",
            "",
            f"{self.n_anomalous} of {self.n_items} {self.level}s are annotated "
            "as anomalous.",
            "",
            "| Metric | Value |",
            "| ------ | ----- |",
            f"| Precision | {fmt(self.precision)} |",
            f"| Recall | {fmt(self.recall)} |",
            f"| F1 | {fmt(self.f1)} |",
            f"| False positive rate | {fmt(self.false_positive_rate)} |",
            f"| ROC-AUC | {fmt(self.roc_auc)} |",
            f"| Average precision | {fmt(self.average_precision)} |",
            "",
            f"Confusion: TP {self.tp}, FP {self.fp}, FN {self.fn}, TN {self.tn}"
            + ("" if self.threshold is None else f" (threshold {self.threshold:g})"),
            "",
            "| Anomaly type | Recall |",
            "| ------------ | ------ |",
        ]
        lines += [f"| {k} | {v:.3f} |" for k, v in sorted(self.recall_by_type.items())]
        lines += ["", "| Category | Recall |", "| -------- | ------ |"]
        lines += [
            f"| {k} | {v:.3f} |" for k, v in sorted(self.recall_by_category.items())
        ]
        return "\n".join(lines) + "\n"


def evaluate_detection(
    dataset: Dataset,
    annotations: Annotations,
    predictions: Iterable[str] | Mapping[str, float],
    *,
    level: AnnotationLevel = "session",
    threshold: float = 0.5,
) -> DetectionReport:
    """Compare a detector's output with the ground truth.

    Args:
        dataset: The dataset the detector saw (defines all items at ``level``).
        annotations: Ground truth for ``dataset``.
        predictions: Flagged ids, or id to anomaly score (higher means more
            anomalous). Booleans count as scores 1/0.
        level: Which items are being classified.
        threshold: Scores at or above it count as flagged.

    Returns:
        The detection report.

    Raises:
        ValidationError: If a prediction names an id that does not exist at
            ``level``, or a score is not a finite number.
    """
    universe = _universe(dataset, level)
    truth = annotations.anomalous_ids(level)
    scored = isinstance(predictions, Mapping)
    if scored:
        scores = {}
        for item, value in predictions.items():  # type: ignore[union-attr]
            if isinstance(value, bool):
                value = float(value)
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ValidationError(
                    "prediction scores must be finite numbers",
                    code="evaluation_invalid_score",
                    context={"id": item},
                )
            scores[item] = float(value)
        flagged = {i for i, s in scores.items() if s >= threshold}
        named = set(scores)
    else:
        flagged = set(predictions)
        named = flagged
    unknown = sorted(named - universe)
    if unknown:
        raise ValidationError(
            f"{len(unknown)} predicted ids do not exist at {level} level",
            code="evaluation_unknown_id",
            context={"level": level, "examples": unknown[:5]},
        )

    tp = len(flagged & truth)
    fp = len(flagged - truth)
    fn = len(truth - flagged)
    tn = len(universe) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else (0.0 if precision is not None and recall is not None else None)
    )
    by_type = {
        kind: len(ids & flagged) / len(ids)
        for kind in sorted({a.type for a in annotations.anomalies(level)})
        if (ids := annotations.anomalous_ids(level, type=kind))
    }
    by_category = {
        category: len(ids & flagged) / len(ids)
        for category in ("invalid_workflow", "unusual_valid")
        if (ids := annotations.anomalous_ids(level, category=category))
    }
    auc = ap = None
    if scored:
        ranked = [
            (scores.get(item, -math.inf), item in truth) for item in sorted(universe)
        ]
        auc = roc_auc(ranked)
        ap = average_precision(ranked)
    return DetectionReport(
        level=level,
        n_items=len(universe),
        n_anomalous=len(truth),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fp / (fp + tn) if fp + tn else None,
        roc_auc=auc,
        average_precision=ap,
        recall_by_type=MappingProxyType(by_type),
        recall_by_category=MappingProxyType(by_category),
        threshold=threshold if scored else None,
    )


def roc_auc(ranked: list[tuple[float, bool]]) -> float | None:
    """Area under the ROC curve via the rank-sum statistic (ties count half).

    Returns ``None`` when only one class is present.
    """
    positives = sum(1 for _, label in ranked if label)
    negatives = len(ranked) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(ranked, key=lambda pair: pair[0])
    rank_sum = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        mean_rank = (i + 1 + j) / 2
        rank_sum += mean_rank * sum(1 for _, label in ordered[i:j] if label)
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(ranked: list[tuple[float, bool]]) -> float | None:
    """Average precision over score thresholds (tied scores handled as a group).

    Returns ``None`` when there are no positives.
    """
    positives = sum(1 for _, label in ranked if label)
    if not positives:
        return None
    ordered = sorted(ranked, key=lambda pair: -pair[0])
    total = hits = seen = 0
    ap = 0.0
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        group_hits = sum(1 for _, label in ordered[i:j] if label)
        seen += j - i
        hits += group_hits
        if group_hits:
            ap += (group_hits / positives) * (hits / seen)
        total += group_hits
        i = j
    return ap


def _universe(dataset: Dataset, level: AnnotationLevel) -> set[str]:
    if level == "session":
        return {s.session_id for s in dataset.sessions or ()}
    if level == "event":
        return {e.event_id for e in dataset.events}
    return set(dataset.learner_ids)
