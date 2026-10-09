"""Score methods against Level 2 ground truth (design §11).

Anomaly-detection scoring (M1) and profile-recovery scoring (M3);
outcome-prediction scoring follows in M4.
"""

from __future__ import annotations

from eduloggen.evaluation.clustering import (
    ClusteringReport,
    adjusted_rand_index,
    evaluate_clustering,
    normalized_mutual_information,
    purity,
)
from eduloggen.evaluation.detection import (
    DetectionReport,
    average_precision,
    evaluate_detection,
    roc_auc,
)

__all__ = [
    "ClusteringReport",
    "DetectionReport",
    "adjusted_rand_index",
    "average_precision",
    "evaluate_clustering",
    "evaluate_detection",
    "normalized_mutual_information",
    "purity",
    "roc_auc",
]
