"""Tests for fitted distributions and samplers."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

import pytest

from eduloggen.core import FitError, GenerationError
from eduloggen.generators import (
    Categorical,
    DurationSampler,
    LengthSampler,
    fit_duration,
    fit_length,
)
from eduloggen.generators.distributions import sketch


def test_categorical_proportions_and_determinism() -> None:
    sampler = Categorical(["a", "b"], [1, 3])
    rng = random.Random(5)
    counts = Counter(sampler.sample(rng) for _ in range(4000))
    assert counts["b"] / 4000 == pytest.approx(0.75, abs=0.03)
    first = [sampler.sample(random.Random(9)) for _ in range(3)]
    second = [sampler.sample(random.Random(9)) for _ in range(3)]
    assert first == second


def test_categorical_zero_weight_never_drawn() -> None:
    sampler = Categorical.from_counts({"x": 0, "y": 2})
    rng = random.Random(0)
    assert {sampler.sample(rng) for _ in range(200)} == {"y"}


@pytest.mark.parametrize(
    ("values", "weights"), [([], []), (["a"], [1, 2]), (["a"], [-1]), (["a"], [0])]
)
def test_categorical_rejects_bad_input(values: list[str], weights: list[float]) -> None:
    with pytest.raises(GenerationError):
        Categorical(values, weights)


def test_sketch() -> None:
    assert sketch([3, 1, 2]) == [1, 2, 3]
    points = sketch(range(10_001), size=11)
    assert points == [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


# --------------------------------------------------------------------------
# Lengths
# --------------------------------------------------------------------------


def test_fit_length_models() -> None:
    assert fit_length([2, 2, 3], "empirical") == {
        "model": "empirical",
        "counts": {"2": 2, "3": 1},
    }
    assert fit_length([1, 3], "poisson") == {"model": "poisson", "mean": 2.0}
    assert fit_length([1], "fixed", 4) == {"model": "fixed", "length": 4}
    with pytest.raises(FitError):
        fit_length([], "empirical")
    with pytest.raises(FitError):
        fit_length([1], "fixed")


@pytest.mark.parametrize("mean", [1.0, 4.0, 80.0])
def test_poisson_lengths(mean: float) -> None:
    sampler = LengthSampler({"model": "poisson", "mean": mean})
    rng = random.Random(0)
    draws = [sampler.sample(rng) for _ in range(3000)]
    assert min(draws) >= 1
    assert sum(draws) / len(draws) == pytest.approx(mean, rel=0.05)


def test_empirical_and_fixed_lengths() -> None:
    rng = random.Random(0)
    empirical = LengthSampler(fit_length([2, 5, 5], "empirical"))
    assert {empirical.sample(rng) for _ in range(100)} == {2, 5}
    assert LengthSampler({"model": "fixed", "length": 7}).sample(rng) == 7


@pytest.mark.parametrize(
    "params",
    [{}, {"model": "zipf"}, {"model": "fixed"}, {"model": "empirical", "counts": []}],
)
def test_length_sampler_rejects_malformed(params: dict[str, Any]) -> None:
    with pytest.raises(GenerationError):
        LengthSampler(params)


# --------------------------------------------------------------------------
# Durations
# --------------------------------------------------------------------------

SAMPLE = [1.0, 2.0, 2.0, 4.0, 11.0]


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("constant", {"family": "constant", "seconds": 2.0}),
        ("empirical", {"family": "empirical", "points": [1.0, 2.0, 2.0, 4.0, 11.0]}),
        ("exponential", {"family": "exponential", "mean": 4.0}),
    ],
)
def test_fit_duration_simple(family: Any, expected: dict[str, Any]) -> None:
    assert fit_duration(SAMPLE, family) == expected


def test_fit_duration_parametric() -> None:
    lognormal = fit_duration(SAMPLE, "lognormal")
    assert lognormal["family"] == "lognormal"
    assert lognormal["sigma"] > 0
    gamma = fit_duration(SAMPLE, "gamma")
    assert gamma["shape"] * gamma["scale"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("values", "family", "expected"),
    [
        ([], "empirical", {"family": "constant", "seconds": 0.0}),
        ([-1.0], "gamma", {"family": "constant", "seconds": 0.0}),
        ([0.0, 0.0], "lognormal", {"family": "constant", "seconds": 0.0}),
        ([0.0, 0.0], "exponential", {"family": "constant", "seconds": 0.0}),
        ([3.0, 3.0], "gamma", {"family": "constant", "seconds": 3.0}),
        ([1.0, 2.0, 3.0, 4.0], "constant", {"family": "constant", "seconds": 2.5}),
    ],
)
def test_fit_duration_fallbacks(
    values: list[float], family: Any, expected: dict[str, Any]
) -> None:
    assert fit_duration(values, family) == expected


@pytest.mark.parametrize(
    "family", ["constant", "empirical", "exponential", "lognormal", "gamma"]
)
def test_duration_samplers(family: Any) -> None:
    sampler = DurationSampler(fit_duration(SAMPLE * 20, family))
    rng = random.Random(0)
    draws = [sampler.sample(rng) for _ in range(2000)]
    assert min(draws) >= 0
    if family not in ("constant", "lognormal"):
        assert sum(draws) / len(draws) == pytest.approx(4.0, rel=0.15)


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"family": "weibull"},
        {"family": "empirical", "points": []},
        {"family": "gamma", "shape": "x", "scale": 1},
    ],
)
def test_duration_sampler_rejects_malformed(params: dict[str, Any]) -> None:
    with pytest.raises(GenerationError):
        DurationSampler(params)
