"""Tests for transitions, n-grams, timing, stats, features, and graphs."""

from __future__ import annotations

import math

import pytest

from eduloggen.analysis import (
    START,
    TransitionCounts,
    count_transitions,
    describe,
    entropy,
    interevent_times,
    learner_profiles,
    navigation_graph,
    ngram_counts,
    quantile,
    rare_ngrams,
    session_durations,
    session_feature_rows,
    session_sequences,
    sessionize,
    sojourn_times,
)
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset

SEQS = [("a", "b", "a"), ("a", "c")]

# --------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------


def test_first_order_counts() -> None:
    counts = count_transitions(SEQS)
    assert counts.order == 1
    assert counts.vocabulary == {"a", "b", "c"}
    assert dict(counts.start_counts()) == {"a": 2}
    assert dict(counts.counts[("a",)]) == {"b": 1, "c": 1}
    assert dict(counts.counts[("b",)]) == {"a": 1}
    assert ("c",) not in counts.counts
    assert counts.n_transitions() == 5
    assert counts.density() == pytest.approx(3 / 9)


def test_second_order_counts_use_padding() -> None:
    counts = count_transitions(SEQS, order=2)
    assert counts.start_context == (START, START)
    assert dict(counts.counts[(START, "a")]) == {"b": 1, "c": 1}
    assert dict(counts.counts[("a", "b")]) == {"a": 1}
    assert counts.density() == pytest.approx(1 / 27)


def test_probabilities() -> None:
    counts = count_transitions(SEQS)
    plain = counts.probabilities()
    assert plain[("a",)] == {"b": 0.5, "c": 0.5}
    smoothed = counts.probabilities(smoothing_alpha=1.0)
    assert smoothed[("a",)] == pytest.approx({"a": 0.2, "b": 0.4, "c": 0.4})
    for row in smoothed.values():
        assert math.fsum(row.values()) == pytest.approx(1.0)
    with pytest.raises(AnalysisError):
        counts.probabilities(smoothing_alpha=-0.1)


def test_transition_round_trip_and_immutability() -> None:
    counts = count_transitions(SEQS, order=2)
    restored = TransitionCounts.from_dict(counts.to_dict())
    assert restored.to_dict() == counts.to_dict()
    assert restored.counts == counts.counts
    with pytest.raises(TypeError):
        counts.counts[("x", "y")] = {}  # type: ignore[index]


@pytest.mark.parametrize(
    "data",
    [
        {},
        {"order": 1, "vocabulary": [], "rows": [{"context": ["a"]}]},
        {"order": 2, "vocabulary": ["a"], "rows": [{"context": ["a"], "next": {}}]},
    ],
)
def test_transition_from_dict_rejects_malformed(data: dict[str, object]) -> None:
    with pytest.raises(AnalysisError):
        TransitionCounts.from_dict(data)


def test_transition_errors() -> None:
    with pytest.raises(AnalysisError):
        count_transitions(SEQS, order=0)
    with pytest.raises(AnalysisError) as info:
        count_transitions([("a", START)])
    assert info.value.code == "analysis_reserved_token"
    empty = count_transitions([])
    assert empty.density() == 0.0
    assert dict(empty.start_counts()) == {}


# --------------------------------------------------------------------------
# Sequences and stats
# --------------------------------------------------------------------------


def test_ngrams_do_not_cross_sequences() -> None:
    bigrams = ngram_counts(SEQS, 2)
    assert bigrams == {("a", "b"): 1, ("b", "a"): 1, ("a", "c"): 1}
    assert ("a", "a") not in bigrams
    assert ngram_counts(SEQS, 4) == {}
    assert rare_ngrams(ngram_counts(SEQS, 1)) == {("b",), ("c",)}
    with pytest.raises(AnalysisError):
        ngram_counts(SEQS, 0)


def test_describe_and_quantile() -> None:
    summary = describe([4, 1, 3, 2])
    assert summary.count == 4
    assert summary.mean == 2.5
    assert summary.median == 2.5
    assert summary.p25 == 1.75
    assert summary.p75 == 3.25
    assert summary.std == pytest.approx(math.sqrt(1.25))
    assert describe([]).mean is None
    assert describe([7]).p75 == 7.0
    with pytest.raises(ValueError):
        quantile([], 0.5)
    with pytest.raises(ValueError):
        quantile([1.0], 1.5)


def test_entropy() -> None:
    assert entropy({"a": 1, "b": 1}) == pytest.approx(1.0)
    assert entropy([4]) == 0.0
    assert entropy({}) == 0.0
    assert entropy({"a": 0, "b": 2}) == 0.0


# --------------------------------------------------------------------------
# Dataset-level statistics
# --------------------------------------------------------------------------


@pytest.fixture
def sessionized(raw: Dataset) -> Dataset:
    return sessionize(raw)


def test_requires_sessionized(raw: Dataset) -> None:
    for function in (
        session_sequences,
        interevent_times,
        sojourn_times,
        session_durations,
        session_feature_rows,
        learner_profiles,
    ):
        with pytest.raises(AnalysisError) as info:
            function(raw)
        assert info.value.code == "analysis_not_sessionized"


def test_session_sequences_ordered_by_id(sessionized: Dataset) -> None:
    assert session_sequences(sessionized) == (
        ("view", "attempt", "submit"),
        ("view", "attempt"),
        ("view", "video_play"),
    )


def test_timing(sessionized: Dataset) -> None:
    assert sorted(interevent_times(sessionized)) == [60.0, 120.0, 300.0, 300.0]
    assert sojourn_times(sessionized) == {
        "attempt": [300.0],
        "view": [300.0, 120.0, 60.0],
    }
    assert sorted(session_durations(sessionized)) == [60.0, 120.0, 600.0]


def test_session_feature_rows(sessionized: Dataset) -> None:
    rows = {row["session_id"]: row for row in session_feature_rows(sessionized)}
    first = rows["A:s1"]
    assert first["session_length"] == 3
    assert first["session_duration_s"] == 600.0
    assert first["n_unique_activities"] == 1
    assert first["event_type_entropy"] == pytest.approx(math.log2(3))
    assert first["success_rate"] == pytest.approx(2 / 3)
    assert rows["A:s2"]["success_rate"] is None
    assert rows["b-x"]["learner_id"] == "B"


def test_learner_profiles(sessionized: Dataset) -> None:
    profiles = learner_profiles(sessionized)
    assert profiles == [
        {
            "learner_id": "A",
            "learner_n_sessions": 2,
            "n_events": 5,
            "mean_session_length": 2.5,
            "mean_session_duration_s": 360.0,
            "n_courses": 1,
        },
        {
            "learner_id": "B",
            "learner_n_sessions": 1,
            "n_events": 2,
            "mean_session_length": 2.0,
            "mean_session_duration_s": 60.0,
            "n_courses": 1,
        },
    ]


def test_profiles_handle_learner_without_sessions() -> None:
    from .conftest import event

    dataset = Dataset(dataset_id="d", events=(event("1", "A", 0, "view", "q"),))
    dataset = sessionize(dataset)
    dataset = Dataset(
        dataset_id="d",
        events=(*dataset.events, event("2", "Z", 0, "view", "q")),
        sessions=dataset.sessions,
    )
    profile = learner_profiles(dataset)[1]
    assert profile["learner_id"] == "Z"
    assert profile["learner_n_sessions"] == 0
    assert profile["mean_session_length"] == 0.0


def test_navigation_graph() -> None:
    graph = navigation_graph([("a", "a", "b"), ("b", "a")])
    assert dict(graph.nodes) == {"a": 3, "b": 2}
    assert dict(graph.edges) == {("a", "a"): 1, ("a", "b"): 1, ("b", "a"): 1}
    assert graph.top_edges(2) == [(("a", "a"), 1), (("a", "b"), 1)]
    assert graph.self_loop_share() == pytest.approx(1 / 3)
    assert navigation_graph([]).self_loop_share() == 0.0
    assert graph.to_dict()["edges"][0] == {"from": "a", "to": "a", "count": 1}
