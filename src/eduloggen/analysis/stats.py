"""Descriptive statistics without third-party dependencies."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Summary", "describe", "entropy", "quantile"]


@dataclass(frozen=True, slots=True)
class Summary:
    """Distribution summary of a numeric sample.

    All statistics are ``None`` for an empty sample. ``std`` is the
    population standard deviation.
    """

    count: int
    mean: float | None
    std: float | None
    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        return asdict(self)


def quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted, non-empty sample.

    Matches NumPy's default (``method="linear"``).

    Raises:
        ValueError: If the sample is empty or ``q`` is outside ``[0, 1]``.
    """
    if not sorted_values:
        raise ValueError("quantile of an empty sample")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be within [0, 1]")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def describe(values: Iterable[float]) -> Summary:
    """Summarize a numeric sample."""
    data = sorted(float(value) for value in values)
    if not data:
        return Summary(0, None, None, None, None, None, None, None)
    mean = math.fsum(data) / len(data)
    variance = math.fsum((value - mean) ** 2 for value in data) / len(data)
    return Summary(
        count=len(data),
        mean=mean,
        std=math.sqrt(variance),
        min=data[0],
        p25=quantile(data, 0.25),
        median=quantile(data, 0.5),
        p75=quantile(data, 0.75),
        max=data[-1],
    )


def entropy(counts: Mapping[Any, int] | Iterable[int]) -> float:
    """Shannon entropy in bits of a count distribution (0 if empty)."""
    values = list(counts.values() if isinstance(counts, Mapping) else counts)
    total = sum(values)
    if total == 0:
        return 0.0
    return -math.fsum(
        (count / total) * math.log2(count / total) for count in values if count
    )
