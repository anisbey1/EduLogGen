"""Helpers shared by the optional neural generators (``gru``, ``transformer``).

Nothing here imports PyTorch at module level: :func:`_torch` imports it on
demand and explains how to install it when it is missing.
"""

from __future__ import annotations

import base64
import math
import random
from array import array
from bisect import bisect_left
from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from eduloggen.core import FitError, GenerationError
from eduloggen.generators.distributions import sketch
from eduloggen.models import Dataset

__all__: list[str] = []


def _torch(error: type[FitError] | type[GenerationError]) -> Any:
    try:
        import torch
    except ImportError:
        raise error(
            "the gru generator requires PyTorch; install it with: "
            'pip install "eduloggen[neural]"',
            code="missing_dependency",
            context={"package": "torch", "extra": "neural"},
        ) from None
    return torch


def _session_gaps(dataset: Dataset) -> list[list[float]]:
    by_session: dict[str, list[Any]] = {}
    for event in dataset.events:
        if event.session_id is not None:
            by_session.setdefault(event.session_id, []).append(event)
    out = []
    # same order as session_sequences(): by session id
    for session in sorted(dataset.sessions or (), key=lambda s: s.session_id):
        events = sorted(
            by_session.get(session.session_id, ()), key=lambda e: e.timestamp
        )
        out.append(
            [(b.timestamp - a.timestamp).total_seconds() for a, b in pairwise(events)]
        )
    return out


def _bins(gaps: list[float], n_bins: int) -> dict[str, Any]:
    """Bins of log gaps: one per distinct value if few, else quantile bins.

    ``edges`` are the inner bin boundaries in log seconds (a gap falls in
    bin ``i`` when ``edges[i-1] < log(gap) <= edges[i]``); ``values`` holds,
    per bin, a sketch of the training gaps in seconds that fell in it.
    """
    if not gaps:
        return {"edges": [], "values": [[1.0]]}
    logs = sorted(math.log(g) for g in gaps)
    distinct = sorted(set(logs))
    if len(distinct) <= n_bins:
        edges = [(a + b) / 2 for a, b in pairwise(distinct)]
    else:
        quantiles = [
            logs[round(i * (len(logs) - 1) / n_bins)] for i in range(1, n_bins)
        ]
        edges = sorted(set(quantiles))
    members: list[list[float]] = [[] for _ in range(len(edges) + 1)]
    for gap in gaps:
        members[bisect_left(edges, math.log(gap))].append(gap)
    values = [
        sketch(m, 64) or [math.exp(e)]
        for m, e in zip(members, [*edges, edges[-1] if edges else 0.0], strict=True)
    ]
    return {"edges": edges, "values": values}


def _encode(tensor: Any) -> dict[str, Any]:
    data = array("f", tensor.detach().reshape(-1).tolist()).tobytes()
    return {
        "shape": list(tensor.shape),
        "float32": base64.b64encode(data).decode("ascii"),
    }


def _decode(torch: Any, item: Mapping[str, Any]) -> Any:
    values = array("f")
    values.frombytes(base64.b64decode(item["float32"]))
    return torch.tensor(values.tolist(), dtype=torch.float32).reshape(item["shape"])


def _draw(rng: random.Random, weights: list[float]) -> int:
    threshold = rng.random() * sum(weights)
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if threshold < acc:
            return i
    return len(weights) - 1


def _session_learners(dataset: Dataset) -> list[str]:
    """Learner of each session, in the order of ``session_sequences()``."""
    return [
        s.learner_id for s in sorted(dataset.sessions or (), key=lambda s: s.session_id)
    ]


def _learner_validation(
    learners: list[str], fraction: float, rng: random.Random
) -> set[int]:
    """Indices of the sessions whose learners form the validation group.

    Learners, not sessions, are split, so no learner is in both groups. At
    least one learner is held out when ``fraction > 0``; with fewer than two
    learners nothing is held out (the caller then trains without early
    stopping).
    """
    distinct = sorted(set(learners))
    if fraction <= 0 or len(distinct) < 2:
        return set()
    rng.shuffle(distinct)
    held = set(distinct[: max(1, int(len(distinct) * fraction))])
    return {i for i, learner in enumerate(learners) if learner in held}
