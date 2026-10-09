"""Sessionization and behavioural analysis of interaction logs (SAD §11).

Typical use::

    from eduloggen.analysis import analyze, sessionize

    sessionized = sessionize(dataset, strategy="idle_timeout", idle_timeout_s=1800)
    result = analyze(sessionized, ngram_order=3)
    print(result.to_markdown())

Analysis never samples; it produces the statistics that generators fit and
that validation and visualization compare.
"""

from __future__ import annotations

from eduloggen.analysis.activity import (
    ActivityProfile,
    activity_profiles,
    activity_table_markdown,
)
from eduloggen.analysis.features import corpus_features, session_feature_rows
from eduloggen.analysis.graphs import NavigationGraph, navigation_graph
from eduloggen.analysis.profiles import learner_profiles
from eduloggen.analysis.result import AnalysisResult
from eduloggen.analysis.sequences import ngram_counts, rare_ngrams, session_sequences
from eduloggen.analysis.service import analyze
from eduloggen.analysis.sessionize import (
    DEFAULT_IDLE_TIMEOUT_S,
    SessionStrategy,
    sessionize,
)
from eduloggen.analysis.stats import Summary, describe, entropy, quantile
from eduloggen.analysis.strata import (
    GROUP_KEYS,
    GroupSummary,
    StratifiedAnalysis,
    analyze_by,
    session_groups,
    split_by,
)
from eduloggen.analysis.temporal import (
    DeadlineEffect,
    TemporalProfile,
    deadline_effects,
    temporal_profile,
)
from eduloggen.analysis.timing import (
    interevent_times,
    session_durations,
    sojourn_times,
)
from eduloggen.analysis.transitions import START, TransitionCounts, count_transitions

__all__ = [
    "DEFAULT_IDLE_TIMEOUT_S",
    "GROUP_KEYS",
    "START",
    "ActivityProfile",
    "AnalysisResult",
    "DeadlineEffect",
    "GroupSummary",
    "NavigationGraph",
    "SessionStrategy",
    "StratifiedAnalysis",
    "Summary",
    "TemporalProfile",
    "TransitionCounts",
    "activity_profiles",
    "activity_table_markdown",
    "analyze",
    "analyze_by",
    "corpus_features",
    "count_transitions",
    "deadline_effects",
    "describe",
    "entropy",
    "interevent_times",
    "learner_profiles",
    "navigation_graph",
    "ngram_counts",
    "quantile",
    "rare_ngrams",
    "session_durations",
    "session_feature_rows",
    "session_groups",
    "session_sequences",
    "sessionize",
    "sojourn_times",
    "split_by",
    "temporal_profile",
]
