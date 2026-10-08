"""Identifier remapping and metadata stripping (PRD §19, SAD §29J).

These are engineering controls that reduce obvious re-identification paths.
They do not make data anonymous: rare behavioural pathways can still identify
people, which is why validation also reports privacy indicators.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from dataclasses import replace
from typing import Literal, TypeVar

from eduloggen.core import ConfigError
from eduloggen.models import Dataset
from eduloggen.utils import make_rng

__all__ = ["IdStrategy", "apply_id_strategy", "remap_ids", "strip_metadata"]

logger = logging.getLogger(__name__)

IdStrategy = Literal["remap", "preserve"]
"""``remap`` assigns fresh ids; ``preserve`` keeps them (local use only)."""

D = TypeVar("D", bound=Dataset)


def remap_ids(dataset: D, seed: int | None) -> D:
    """Replace learner, session, and event ids with fresh synthetic ids.

    Learners become ``L1``, ``L2``, … and sessions ``S1``, ``S2``, … in a
    seeded random order, so new ids reveal nothing about the original ids or
    their sort order. Events become ``E1``, ``E2``, … in canonical
    ``(learner, time)`` order of the new ids. References between events and
    sessions are kept consistent; all other fields are unchanged.

    Args:
        dataset: Dataset to remap; a :class:`SyntheticDataset` stays one.
        seed: Seed for the id permutation; ``None`` for a random one.

    Returns:
        A new dataset of the same type.
    """
    rng = make_rng(seed, "privacy", "remap_ids")
    learners = _shuffled_ids(dataset.learner_ids, "L", rng)
    session_ids = {e.session_id for e in dataset.events if e.session_id is not None}
    session_ids.update(s.session_id for s in dataset.sessions or ())
    sessions = _shuffled_ids(session_ids, "S", rng)

    renamed = [
        replace(
            event,
            learner_id=learners[event.learner_id],
            session_id=None if event.session_id is None else sessions[event.session_id],
        )
        for event in dataset.events
    ]
    renamed.sort(key=lambda event: event.sort_key)
    width = len(str(len(renamed)))
    events = tuple(
        replace(event, event_id=f"E{i:0{width}d}")
        for i, event in enumerate(renamed, start=1)
    )
    new_sessions = (
        None
        if dataset.sessions is None
        else tuple(
            sorted(
                (
                    replace(
                        session,
                        session_id=sessions[session.session_id],
                        learner_id=learners[session.learner_id],
                    )
                    for session in dataset.sessions
                ),
                key=lambda session: session.session_id,
            )
        )
    )
    return replace(dataset, events=events, sessions=new_sessions)


def strip_metadata(dataset: D, keys: Iterable[str]) -> D:
    """Remove the given keys from every event's metadata.

    Args:
        dataset: Dataset to clean.
        keys: Metadata keys to drop, e.g. ``["ip", "email"]``.

    Returns:
        A new dataset of the same type; unchanged if no event has the keys.
    """
    drop = frozenset(keys)
    if not drop or not any(drop & e.metadata.keys() for e in dataset.events):
        return dataset
    events = tuple(
        replace(
            event,
            metadata={k: v for k, v in event.metadata.items() if k not in drop},
        )
        for event in dataset.events
    )
    return replace(dataset, events=events)


def apply_id_strategy(dataset: D, strategy: IdStrategy, seed: int | None) -> D:
    """Apply a configured id strategy.

    Raises:
        ConfigError: If ``strategy`` is unknown.
    """
    if strategy == "remap":
        return remap_ids(dataset, seed)
    if strategy == "preserve":
        logger.warning(
            "id_strategy=preserve keeps source ids; do not share this output"
        )
        return dataset
    raise ConfigError(
        "unknown id strategy",
        code="config_invalid_value",
        context={"key": "generation.id_strategy", "allowed": ["remap", "preserve"]},
    )


def _shuffled_ids(
    originals: Iterable[str], prefix: str, rng: random.Random
) -> dict[str, str]:
    ordered = sorted(originals)
    rng.shuffle(ordered)
    width = len(str(len(ordered)))
    return {old: f"{prefix}{i:0{width}d}" for i, old in enumerate(ordered, start=1)}
