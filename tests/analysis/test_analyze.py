"""Tests for the analyze() façade and AnalysisResult exports."""

from __future__ import annotations

import json

import pytest

from eduloggen.analysis import analyze, sessionize
from eduloggen.core import AnalysisError
from eduloggen.models import Dataset


@pytest.fixture
def sessionized(raw: Dataset) -> Dataset:
    return sessionize(raw)


def test_analyze(sessionized: Dataset) -> None:
    result = analyze(sessionized, ngram_order=3)
    assert result.dataset_id == "toy"
    assert result.dataset_fingerprint == sessionized.fingerprint()
    assert dict(result.event_type_counts) == {
        "view": 3,
        "attempt": 2,
        "submit": 1,
        "video_play": 1,
    }
    assert dict(result.token_counts) == dict(result.event_type_counts)
    assert dict(result.transitions.start_counts()) == {"view": 3}
    assert set(result.ngram_counts) == {2, 3}
    assert result.ngram_counts[3] == {("view", "attempt", "submit"): 1}
    assert result.n_rare_ngrams[2] == 2
    assert result.session_length.count == 3
    assert result.session_length.max == 3.0
    assert result.session_duration_s.max == 600.0
    assert result.interevent_s.count == 4
    assert result.sojourn_s["view"].count == 3
    assert result.feature("corpus_vocab_size").value == 4
    assert result.feature("n_learners").value == 2
    assert result.graph.nodes["view"] == 3
    with pytest.raises(KeyError):
        result.feature("missing")


def test_unigram_order(sessionized: Dataset) -> None:
    result = analyze(sessionized, ngram_order=1)
    assert dict(result.ngram_counts) == {}


def test_to_dict_is_json(sessionized: Dataset) -> None:
    data = json.loads(json.dumps(analyze(sessionized).to_dict()))
    assert data["ngram_order"] == 2
    assert data["transitions"]["rows"][0]["context"] == ["<start>"]
    assert data["ngrams"]["2"][0] == {"ngram": ["attempt", "submit"], "count": 1}
    assert data["session_length"]["median"] == 2.0
    assert {f["name"] for f in data["corpus_features"]} >= {"corpus_transition_density"}


def test_to_markdown(sessionized: Dataset) -> None:
    text = analyze(sessionized).to_markdown()
    assert text.startswith("# Analysis of `toy`")
    assert "| corpus_vocab_size | 4 |" in text
    assert "| Session duration (s) | 3 |" in text
    assert "| view | attempt | 2 |" in text
    assert "| view | 3 |" in text


def test_markdown_handles_empty_dataset() -> None:
    empty = sessionize(Dataset(dataset_id="e", events=()))
    text = analyze(empty).to_markdown()
    assert "| Inter-event time (s) | 0 | - |" in text


def test_analyze_errors(raw: Dataset, sessionized: Dataset) -> None:
    with pytest.raises(AnalysisError):
        analyze(raw)
    with pytest.raises(AnalysisError):
        analyze(sessionized, ngram_order=0)
