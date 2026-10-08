"""Distribution and sequence distances used by metrics.

All functions are dependency-free and deterministic. Count-based distances
take mappings of category to count and normalize them internally.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from itertools import pairwise
from typing import TypeVar

K = TypeVar("K", bound=Hashable)

__all__ = [
    "jensen_shannon",
    "ks_statistic",
    "normalized_edit_distance",
    "total_variation",
    "wasserstein1",
]


def _normalize(counts: Mapping[K, float]) -> dict[K, float]:
    total = math.fsum(counts.values())
    if total <= 0:
        raise ValueError("counts must have a positive total")
    return {key: value / total for key, value in counts.items()}


def total_variation(p: Mapping[K, float], q: Mapping[K, float]) -> float:
    """Total variation distance in ``[0, 1]`` between two count distributions.

    Raises:
        ValueError: If either distribution is empty.
    """
    pn, qn = _normalize(p), _normalize(q)
    return 0.5 * math.fsum(abs(pn.get(k, 0.0) - qn.get(k, 0.0)) for k in pn.keys() | qn)


def jensen_shannon(p: Mapping[K, float], q: Mapping[K, float]) -> float:
    """Jensen-Shannon divergence in bits, in ``[0, 1]``.

    Raises:
        ValueError: If either distribution is empty.
    """
    pn, qn = _normalize(p), _normalize(q)
    total = 0.0
    for key in pn.keys() | qn:
        a, b = pn.get(key, 0.0), qn.get(key, 0.0)
        m = (a + b) / 2
        if a > 0:
            total += 0.5 * a * math.log2(a / m)
        if b > 0:
            total += 0.5 * b * math.log2(b / m)
    return min(max(total, 0.0), 1.0)


def ks_statistic(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic (max CDF gap) in ``[0, 1]``.

    Raises:
        ValueError: If either sample is empty.
    """
    if not a or not b:
        raise ValueError("samples must be non-empty")
    xs, ys = sorted(a), sorted(b)
    i = j = 0
    gap = 0.0
    while i < len(xs) and j < len(ys):
        value = min(xs[i], ys[j])
        while i < len(xs) and xs[i] == value:
            i += 1
        while j < len(ys) and ys[j] == value:
            j += 1
        gap = max(gap, abs(i / len(xs) - j / len(ys)))
    return gap


def wasserstein1(a: Sequence[float], b: Sequence[float]) -> float:
    """First Wasserstein (earth mover's) distance between two samples.

    Equals the area between the empirical CDFs, in the samples' units.

    Raises:
        ValueError: If either sample is empty.
    """
    if not a or not b:
        raise ValueError("samples must be non-empty")
    xs, ys = sorted(a), sorted(b)
    points = sorted(set(xs) | set(ys))
    total = 0.0
    i = j = 0
    for left, right in pairwise(points):
        while i < len(xs) and xs[i] <= left:
            i += 1
        while j < len(ys) and ys[j] <= left:
            j += 1
        total += abs(i / len(xs) - j / len(ys)) * (right - left)
    return total


def normalized_edit_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """Levenshtein distance divided by the longer length, in ``[0, 1]``.

    Two empty sequences are at distance 0.
    """
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1] / len(a)
