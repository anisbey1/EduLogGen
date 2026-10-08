"""Tests for distance functions."""

from __future__ import annotations

import math

import pytest

from eduloggen.validation import (
    jensen_shannon,
    ks_statistic,
    normalized_edit_distance,
    total_variation,
    wasserstein1,
)


def test_total_variation() -> None:
    assert total_variation({"a": 1}, {"a": 5}) == 0.0
    assert total_variation({"a": 1}, {"b": 1}) == 1.0
    assert total_variation({"a": 1, "b": 1}, {"a": 3, "b": 1}) == pytest.approx(0.25)


def test_jensen_shannon() -> None:
    assert jensen_shannon({"a": 2, "b": 2}, {"a": 1, "b": 1}) == pytest.approx(0.0)
    assert jensen_shannon({"a": 1}, {"b": 1}) == pytest.approx(1.0)
    value = jensen_shannon({"a": 1, "b": 1}, {"a": 1})
    assert 0 < value < 1
    expected = 0.5 * (0.5 * math.log2(0.5 / 0.75) + 0.5 * 1) + 0.5 * math.log2(1 / 0.75)
    assert value == pytest.approx(expected)


@pytest.mark.parametrize("function", [total_variation, jensen_shannon])
def test_count_distances_reject_empty(function: object) -> None:
    with pytest.raises(ValueError):
        function({}, {"a": 1})  # type: ignore[operator]


def test_ks_statistic() -> None:
    assert ks_statistic([1, 2, 3], [3, 2, 1]) == 0.0
    assert ks_statistic([1, 2], [3, 4]) == 1.0
    assert ks_statistic([1, 1, 2, 2], [1, 2, 2, 2]) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        ks_statistic([], [1])


def test_wasserstein() -> None:
    assert wasserstein1([0.0], [0.0]) == 0.0
    assert wasserstein1([0.0, 1.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert wasserstein1([0.0], [10.0]) == pytest.approx(10.0)
    assert wasserstein1([0, 0, 10], [0, 10, 10]) == pytest.approx(10 / 3)
    with pytest.raises(ValueError):
        wasserstein1([1.0], [])


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((), (), 0.0),
        (("a",), (), 1.0),
        (("a", "b", "c"), ("a", "b", "c"), 0.0),
        (("a", "b", "c"), ("a", "b", "d"), 1 / 3),
        (("a", "b"), ("a", "b", "c", "d"), 0.5),
        (("x", "y"), ("y", "x"), 1.0),
    ],
)
def test_normalized_edit_distance(
    a: tuple[str, ...], b: tuple[str, ...], expected: float
) -> None:
    assert normalized_edit_distance(a, b) == pytest.approx(expected)
    assert normalized_edit_distance(b, a) == pytest.approx(expected)
