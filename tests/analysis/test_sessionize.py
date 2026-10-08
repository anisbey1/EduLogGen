"""Tests for sessionization strategies."""

from __future__ import annotations

import pytest

from eduloggen.analysis import sessionize
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset, GenerationMetadata, SyntheticDataset

from .conftest import event


def _summary(dataset: Dataset) -> dict[str, tuple[str, ...]]:
    return {s.session_id: s.event_sequence for s in dataset.sessions or ()}


def test_composite_keeps_explicit_ids_and_splits_others(raw: Dataset) -> None:
    result = sessionize(raw)
    assert _summary(result) == {
        "A:s1": ("view", "attempt", "submit"),
        "A:s2": ("view", "attempt"),
        "b-x": ("view", "video_play"),
    }
    assert all(e.session_id is not None for e in result.events)
    assert result.n_events == raw.n_events
    assert raw.sessions is None


def test_idle_timeout_ignores_source_ids(raw: Dataset) -> None:
    result = sessionize(raw, strategy="idle_timeout")
    assert set(_summary(result)) == {"A:s1", "A:s2", "B:s1"}


def test_timeout_boundary_is_exclusive(raw: Dataset) -> None:
    exact = sessionize(raw, strategy="idle_timeout", idle_timeout_s=70 * 60)
    assert _summary(exact)["A:s1"] == ("view", "attempt", "submit", "view", "attempt")
    split = sessionize(raw, strategy="idle_timeout", idle_timeout_s=70 * 60 - 1)
    assert "A:s2" in _summary(split)


def test_short_timeout_splits_every_event(raw: Dataset) -> None:
    result = sessionize(raw, strategy="idle_timeout", idle_timeout_s=30)
    assert result.n_sessions == raw.n_events


def test_tokenization(raw: Dataset) -> None:
    result = sessionize(raw, tokenization="activity_id")
    assert _summary(result)["A:s2"] == ("video", "quiz")


def test_explicit_strategy() -> None:
    dataset = Dataset(
        dataset_id="d",
        events=(
            event("1", "A", 0, "view", "q", session="s"),
            event("2", "A", 500, "submit", "q", session="s"),
        ),
    )
    result = sessionize(dataset, strategy="explicit", idle_timeout_s=1)
    assert _summary(result) == {"s": ("view", "submit")}


def test_explicit_requires_ids(raw: Dataset) -> None:
    with pytest.raises(AnalysisError) as info:
        sessionize(raw, strategy="explicit")
    assert info.value.code == "analysis_missing_session_id"


def test_explicit_session_spanning_learners() -> None:
    dataset = Dataset(
        dataset_id="d",
        events=(
            event("1", "A", 0, "view", "q", session="shared"),
            event("2", "B", 1, "view", "q", session="shared"),
        ),
    )
    with pytest.raises(AnalysisError) as info:
        sessionize(dataset, strategy="explicit")
    assert info.value.code == "analysis_mixed_session"


def test_generated_id_collision() -> None:
    dataset = Dataset(
        dataset_id="d",
        events=(
            event("1", "A", 0, "view", "q"),
            event("2", "B", 0, "view", "q", session="A:s1"),
        ),
    )
    with pytest.raises(AnalysisError) as info:
        sessionize(dataset)
    assert info.value.code == "analysis_session_id_collision"


@pytest.mark.parametrize(
    "kwargs",
    [{"idle_timeout_s": 0}, {"idle_timeout_s": -5}, {"strategy": "weekly"}],
)
def test_invalid_parameters(raw: Dataset, kwargs: dict[str, object]) -> None:
    with pytest.raises(AnalysisError) as info:
        sessionize(raw, **kwargs)  # type: ignore[arg-type]
    assert info.value.code == "analysis_invalid_parameter"


def test_empty_dataset() -> None:
    result = sessionize(Dataset(dataset_id="e", events=()))
    assert result.sessions == ()


def test_keeps_synthetic_type(raw: Dataset) -> None:
    synthetic = SyntheticDataset(
        dataset_id="s",
        events=raw.events,
        generation=GenerationMetadata(
            generator_id="g", model_fingerprint="sha256:x", seed=0, n_sessions=1
        ),
    )
    assert isinstance(sessionize(synthetic), SyntheticDataset)


def test_resessionizing_replaces_sessions(raw: Dataset) -> None:
    once = sessionize(raw, strategy="idle_timeout", idle_timeout_s=30)
    twice = sessionize(once, strategy="idle_timeout")
    assert twice.n_sessions == 3
