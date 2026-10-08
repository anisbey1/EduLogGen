"""Built-in validation metrics (SAD §29E)."""

from __future__ import annotations

from eduloggen.validation.metrics.marginal import (
    ActivityJSD,
    EventTypeTVD,
    SessionDurationW1,
    SessionLengthKS,
)
from eduloggen.validation.metrics.privacy import (
    ExactSessionDuplicateRate,
    NearestNeighbourDistance,
    RareNgramReplayRate,
)
from eduloggen.validation.metrics.sequential import (
    BigramTVD,
    TopNPathOverlap,
    TransitionJSD,
)
from eduloggen.validation.metrics.temporal import InterEventTimeKS

__all__ = [
    "BUILTIN_METRICS",
    "ActivityJSD",
    "BigramTVD",
    "EventTypeTVD",
    "ExactSessionDuplicateRate",
    "InterEventTimeKS",
    "NearestNeighbourDistance",
    "RareNgramReplayRate",
    "SessionDurationW1",
    "SessionLengthKS",
    "TopNPathOverlap",
    "TransitionJSD",
]

BUILTIN_METRICS = (
    EventTypeTVD,
    ActivityJSD,
    SessionLengthKS,
    SessionDurationW1,
    InterEventTimeKS,
    BigramTVD,
    TransitionJSD,
    TopNPathOverlap,
    ExactSessionDuplicateRate,
    RareNgramReplayRate,
    NearestNeighbourDistance,
)
"""All built-in metrics, in report order."""
