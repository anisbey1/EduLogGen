"""Tests for profile-recovery metrics."""

from __future__ import annotations

import json

import pytest

from eduloggen.core import ConfigError
from eduloggen.evaluation import (
    adjusted_rand_index,
    evaluate_clustering,
    normalized_mutual_information,
    purity,
)
from eduloggen.models import Annotation, Annotations

TRUE = list("aaabbbccc")
PRED = list("xxyyyzzzz")


def test_metrics_hand_checked() -> None:
    # pairs together in both = 5, expected = 9 * 10 / 36 = 2.5, max = 9.5
    assert adjusted_rand_index(TRUE, PRED) == pytest.approx(2.5 / 7)
    assert purity(TRUE, PRED) == pytest.approx(7 / 9)
    assert 0 < normalized_mutual_information(TRUE, PRED) < 1


def test_metrics_are_label_invariant() -> None:
    renamed = [{"a": "q", "b": "r", "c": "s"}[t] for t in TRUE]
    assert adjusted_rand_index(TRUE, renamed) == 1.0
    assert normalized_mutual_information(TRUE, renamed) == pytest.approx(1.0)
    assert purity(TRUE, renamed) == 1.0


def test_degenerate_labelings() -> None:
    one = ["a"] * 4
    assert adjusted_rand_index(one, one) == 1.0
    assert normalized_mutual_information(one, one) == 1.0
    assert adjusted_rand_index(["a"], ["x"]) == 1.0
    assert normalized_mutual_information(TRUE, ["x"] * 9) == 0.0
    assert adjusted_rand_index(TRUE, ["x"] * 9) == 0.0


def _annotations() -> Annotations:
    rows = [
        Annotation(
            level="learner", id=f"l{i}", annotation="profile", type="profile", value=t
        )
        for i, t in enumerate(TRUE)
    ]
    rows.append(
        Annotation(
            level="session",
            id="s1",
            annotation="anomaly",
            type="repetition",
            category="unusual_valid",
        )
    )
    return Annotations(tuple(rows))


def test_evaluate_clustering() -> None:
    predictions = {f"l{i}": p for i, p in enumerate(PRED) if i != 0}
    predictions["stranger"] = "x"
    report = evaluate_clustering(_annotations(), predictions)
    assert report.n_learners == 8
    assert report.missing == 1
    assert report.contingency["c"] == {"z": 3}
    assert json.loads(json.dumps(report.to_dict()))["n_learners"] == 8
    assert "| ARI | NMI | Purity |" in report.to_markdown()


def test_evaluate_clustering_errors() -> None:
    with pytest.raises(ConfigError, match="no learner profile"):
        evaluate_clustering(Annotations(()), {"l0": "x"})
    with pytest.raises(ConfigError, match="no predicted"):
        evaluate_clustering(_annotations(), {"nobody": "x"})
