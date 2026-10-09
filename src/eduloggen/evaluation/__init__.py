"""Score methods against Level 2 ground truth (design §11).

M1 provides anomaly-detection scoring; clustering and outcome-prediction
scoring follow in later milestones.
"""

from __future__ import annotations

from eduloggen.evaluation.detection import (
    DetectionReport,
    average_precision,
    evaluate_detection,
    roc_auc,
)

__all__ = ["DetectionReport", "average_precision", "evaluate_detection", "roc_auc"]
