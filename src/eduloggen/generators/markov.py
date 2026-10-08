"""Order-k Markov chain generator (PRD §11.3, SAD §29D.2).

Fit:
    Count transitions for every order from 1 to ``k`` (start-padded, see
    :mod:`eduloggen.analysis.transitions`), plus overall token frequencies.

Generate:
    Draw a length, then each token from the longest context seen in
    training, backing off to shorter contexts and finally to token
    frequencies when a context was never observed (e.g. after a token that
    only ever ended sessions). Events are spaced by the median training
    inter-event gap; use ``semi_markov`` for realistic timing.

Hyperparameters:
    ``order`` (int, default 1), ``smoothing_alpha`` (float >= 0, default 0:
    only observed transitions), ``on_insufficient_data`` (``fail`` or
    ``backoff``, default ``fail``), ``length_model`` (``empirical``,
    ``poisson``, ``fixed``), ``fixed_length``.
"""

from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from eduloggen.analysis import (
    START,
    TransitionCounts,
    count_transitions,
    interevent_times,
)
from eduloggen.core import FitError, GenerationError
from eduloggen.generators.base import BaseGenerator, SequenceSampler, _choice, _number
from eduloggen.generators.distributions import (
    Categorical,
    DurationSampler,
    fit_duration,
)
from eduloggen.models import Dataset, GeneratorModel

__all__ = ["MarkovGenerator"]

logger = logging.getLogger(__name__)

Context = tuple[str, ...]


class MarkovGenerator(BaseGenerator):
    """Sessions as order-k Markov chains over tokens."""

    name: ClassVar[str] = "markov"
    tags: ClassVar[frozenset[str]] = frozenset({"probabilistic", "sequence"})
    defaults: ClassVar[Mapping[str, Any]] = {
        "order": 1,
        "smoothing_alpha": 0.0,
        "on_insufficient_data": "fail",
    }

    def _validate_family(self, hyperparameters: dict[str, Any]) -> None:
        _number(hyperparameters, "order", minimum=1, integer=True)
        _number(hyperparameters, "smoothing_alpha", minimum=0.0)
        _choice(hyperparameters, "on_insufficient_data", ("fail", "backoff"))

    def _fit_family(
        self,
        dataset: Dataset,
        sequences: tuple[tuple[str, ...], ...],
        hyperparameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        order = _supported_order(sequences, hyperparameters)
        return {
            "order": order,
            "transitions": {
                str(k): count_transitions(sequences, order=k).to_dict()
                for k in range(1, order + 1)
            },
            "unigram": dict(sorted(Counter(t for s in sequences for t in s).items())),
            "timing": self._fit_timing(dataset, hyperparameters),
        }

    def _fit_timing(
        self, dataset: Dataset, hyperparameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {"pooled": fit_duration(interevent_times(dataset), "constant")}

    def _describe_family(self, model: GeneratorModel) -> dict[str, Any]:
        transitions = model.parameters["transitions"]
        top = transitions[str(model.parameters["order"])]
        return {
            "order": model.parameters["order"],
            "n_contexts": len(top["rows"]),
            "timing": model.parameters["timing"]["pooled"]["family"],
        }

    def _sequence_sampler(self, model: GeneratorModel) -> SequenceSampler:
        tables, unigram, order = _transition_tables(model)
        gap = self._gap_sampler(model)

        def sample(rng: random.Random, length: int) -> tuple[list[str], list[float]]:
            history: list[str] = [START] * order
            tokens: list[str] = []
            for _ in range(length):
                token = _next_token(rng, history, tables, unigram, order)
                tokens.append(token)
                history.append(token)
            gaps = [gap(rng, token) for token in tokens[:-1]]
            return tokens, gaps

        return sample

    def _gap_sampler(
        self, model: GeneratorModel
    ) -> Callable[[random.Random, str], float]:
        pooled = DurationSampler(model.parameters["timing"]["pooled"])
        return lambda rng, _token: pooled.sample(rng)


def _supported_order(
    sequences: tuple[tuple[str, ...], ...], hyperparameters: Mapping[str, Any]
) -> int:
    requested: int = hyperparameters["order"]
    longest = max(len(s) for s in sequences)
    if longest > requested:
        return requested
    supported = max(longest - 1, 1)
    if hyperparameters["on_insufficient_data"] == "fail":
        raise FitError(
            f"order {requested} needs sessions longer than {requested} events; "
            f"the longest has {longest}. Lower 'order' or set "
            "on_insufficient_data: backoff",
            code="fit_insufficient_data",
            context={"order": requested, "longest_session": longest},
        )
    logger.warning(
        "reducing Markov order from %d to %d: sessions are too short",
        requested,
        supported,
    )
    return supported


def _transition_tables(
    model: GeneratorModel,
) -> tuple[dict[int, dict[Context, Categorical[str]]], Categorical[str], int]:
    try:
        params = model.parameters
        order = int(params["order"])
        alpha = float(model.hyperparameters["smoothing_alpha"])
        tables: dict[int, dict[Context, Categorical[str]]] = {}
        for k in range(1, order + 1):
            counts = TransitionCounts.from_dict(params["transitions"][str(k)])
            tables[k] = {
                context: Categorical.from_counts(row)
                for context, row in counts.probabilities(alpha).items()
            }
        unigram = Categorical.from_counts(params["unigram"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise GenerationError(
            "model parameters are malformed", code="generation_invalid_model"
        ) from exc
    return tables, unigram, order


def _next_token(
    rng: random.Random,
    history: list[str],
    tables: dict[int, dict[Context, Categorical[str]]],
    unigram: Categorical[str],
    order: int,
) -> str:
    for k in range(order, 0, -1):
        row = tables[k].get(tuple(history[-k:]))
        if row is not None:
            return row.sample(rng)
    return unigram.sample(rng)
