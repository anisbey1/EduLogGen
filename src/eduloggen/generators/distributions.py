"""Fitted distributions used by generators, stored as plain JSON parameters.

Every model is fitted into a small ``dict`` (so artifacts stay inspectable)
and turned back into a sampler with ``*.from_params``. Samplers draw only from
the :class:`random.Random` they are given, so output is fully seeded.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from itertools import accumulate
from typing import Any, Final, Generic, Literal, TypeVar

from eduloggen.core import FitError, GenerationError

__all__ = [
    "Categorical",
    "DurationFamily",
    "DurationSampler",
    "LengthModel",
    "LengthSampler",
    "fit_duration",
    "fit_length",
    "sketch",
]

T = TypeVar("T")

SKETCH_SIZE: Final = 1000
"""Maximum number of points kept for empirical distributions."""

LengthModel = Literal["empirical", "poisson", "fixed"]
DurationFamily = Literal["empirical", "lognormal", "gamma", "exponential", "constant"]


class Categorical(Generic[T]):
    """Draw values with probability proportional to non-negative weights."""

    def __init__(self, values: Sequence[T], weights: Sequence[float]) -> None:
        """Build the sampler.

        Raises:
            GenerationError: If there are no values, lengths differ, or the
                weights are negative or sum to zero.
        """
        if not values or len(values) != len(weights):
            raise GenerationError(
                "categorical needs one weight per value",
                code="generation_invalid_model",
            )
        if any(w < 0 for w in weights) or not math.fsum(weights) > 0:
            raise GenerationError(
                "categorical weights must be non-negative with a positive sum",
                code="generation_invalid_model",
            )
        self.values = list(values)
        self._cumulative = list(accumulate(weights))

    @classmethod
    def from_counts(cls, counts: Mapping[T, float]) -> Categorical[T]:
        """Build from a value-to-weight mapping, in sorted key order."""
        items = sorted(counts.items(), key=lambda item: repr(item[0]))
        return cls([value for value, _ in items], [weight for _, weight in items])

    def sample(self, rng: random.Random) -> T:
        """Draw one value."""
        point = rng.random() * self._cumulative[-1]
        index = min(bisect_right(self._cumulative, point), len(self.values) - 1)
        return self.values[index]


def sketch(values: Iterable[float], size: int = SKETCH_SIZE) -> list[float]:
    """Sorted sample of at most ``size`` evenly spaced order statistics.

    Keeps the shape of a large empirical distribution in bounded space and is
    deterministic (no random subsampling).
    """
    data = sorted(values)
    if len(data) <= size:
        return data
    step = (len(data) - 1) / (size - 1)
    return [data[round(i * step)] for i in range(size)]


# ---------------------------------------------------------------------------
# Session length
# ---------------------------------------------------------------------------


def fit_length(
    lengths: Sequence[int], model: LengthModel, fixed_length: int | None = None
) -> dict[str, Any]:
    """Fit a session length model.

    Args:
        lengths: Observed session lengths (all at least 1).
        model: ``empirical`` resamples observed lengths, ``poisson`` draws
            ``1 + Poisson(mean - 1)``, ``fixed`` always uses ``fixed_length``.
        fixed_length: Length for the ``fixed`` model.

    Raises:
        FitError: If ``lengths`` is empty or ``fixed_length`` is missing.
    """
    if not lengths:
        raise FitError("no sessions to fit a length model", code="fit_empty_corpus")
    if model == "fixed":
        if fixed_length is None:
            raise FitError(
                "length_model 'fixed' needs fixed_length",
                code="fit_invalid_hyperparameter",
            )
        return {"model": "fixed", "length": fixed_length}
    if model == "poisson":
        return {"model": "poisson", "mean": math.fsum(lengths) / len(lengths)}
    counts = Counter(lengths)
    return {
        "model": "empirical",
        "counts": {str(k): v for k, v in sorted(counts.items())},
    }


class LengthSampler:
    """Draw session lengths from fitted parameters."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Restore the sampler.

        Raises:
            GenerationError: If the parameters are malformed.
        """
        try:
            self.model: str = params["model"]
            if self.model == "fixed":
                self._fixed = int(params["length"])
            elif self.model == "poisson":
                self._mean = float(params["mean"])
            elif self.model == "empirical":
                self._categorical = Categorical.from_counts(
                    {int(k): float(v) for k, v in params["counts"].items()}
                )
            else:
                raise KeyError(self.model)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GenerationError(
                "malformed length model", code="generation_invalid_model"
            ) from exc

    def sample(self, rng: random.Random) -> int:
        """Draw a length of at least 1."""
        if self.model == "fixed":
            return self._fixed
        if self.model == "poisson":
            return 1 + _poisson(rng, max(self._mean - 1.0, 0.0))
        return self._categorical.sample(rng)


def _poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    threshold = math.exp(-lam)
    count, product = 0, rng.random()
    while product > threshold:
        count += 1
        product *= rng.random()
    return count


# ---------------------------------------------------------------------------
# Durations (inter-event and sojourn times)
# ---------------------------------------------------------------------------


def fit_duration(values: Sequence[float], family: DurationFamily) -> dict[str, Any]:
    """Fit a non-negative duration distribution in seconds.

    Falls back to ``constant`` when the sample cannot support the family
    (e.g. no positive values for ``lognormal`` or zero variance for
    ``gamma``). An empty sample gives a constant of 0.
    """
    data = [float(v) for v in values if v >= 0]
    if not data:
        return {"family": "constant", "seconds": 0.0}
    mean = math.fsum(data) / len(data)
    if family == "constant":
        return {"family": "constant", "seconds": _median(data)}
    if family == "empirical":
        return {"family": "empirical", "points": sketch(data)}
    if family == "exponential":
        if mean <= 0:
            return {"family": "constant", "seconds": 0.0}
        return {"family": "exponential", "mean": mean}
    if family == "lognormal":
        logs = [math.log(v) for v in data if v > 0]
        if not logs:
            return {"family": "constant", "seconds": 0.0}
        mu = math.fsum(logs) / len(logs)
        sigma = math.sqrt(math.fsum((x - mu) ** 2 for x in logs) / len(logs))
        return {"family": "lognormal", "mu": mu, "sigma": sigma}
    variance = math.fsum((v - mean) ** 2 for v in data) / len(data)
    if mean <= 0 or variance <= 0:
        return {"family": "constant", "seconds": mean}
    return {
        "family": "gamma",
        "shape": mean * mean / variance,
        "scale": variance / mean,
    }


class DurationSampler:
    """Draw durations from fitted parameters."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        """Restore the sampler.

        Raises:
            GenerationError: If the parameters are malformed.
        """
        try:
            self.family: str = params["family"]
            if self.family == "constant":
                self._a = float(params["seconds"])
            elif self.family == "empirical":
                self._points = [float(p) for p in params["points"]]
                if not self._points:
                    raise ValueError("empty")
            elif self.family == "exponential":
                self._a = float(params["mean"])
            elif self.family == "lognormal":
                self._a, self._b = float(params["mu"]), float(params["sigma"])
            elif self.family == "gamma":
                self._a, self._b = float(params["shape"]), float(params["scale"])
            else:
                raise KeyError(self.family)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GenerationError(
                "malformed duration model", code="generation_invalid_model"
            ) from exc

    def sample(self, rng: random.Random) -> float:
        """Draw a non-negative duration in seconds."""
        if self.family == "constant":
            return self._a
        if self.family == "empirical":
            return self._points[rng.randrange(len(self._points))]
        if self.family == "exponential":
            return rng.expovariate(1.0 / self._a)
        if self.family == "lognormal":
            return rng.lognormvariate(self._a, self._b)
        return rng.gammavariate(self._a, self._b)


def _median(data: Sequence[float]) -> float:
    ordered = sorted(data)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2
