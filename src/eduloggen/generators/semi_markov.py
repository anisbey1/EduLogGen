"""Semi-Markov generator: Markov transitions plus per-token timing (SAD §29D.3).

After each event, the time until the next one is drawn from the *sojourn*
distribution of the current token, so learners linger on videos longer than
on navigation clicks if the training data says so. Tokens with fewer than
``min_samples`` observed sojourns use the pooled distribution of all gaps.
Timestamps are strictly increasing within each session.

Hyperparameters (in addition to ``markov``'s):
    ``timing_family`` (``empirical`` resampling, ``lognormal``, ``gamma``,
    ``exponential``; default ``empirical``), ``min_samples`` (int >= 1,
    default 5).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Mapping
from typing import Any, ClassVar

from eduloggen.analysis import interevent_times, sojourn_times
from eduloggen.generators.base import _choice, _number
from eduloggen.generators.distributions import DurationSampler, fit_duration
from eduloggen.generators.markov import MarkovGenerator
from eduloggen.models import Dataset, GeneratorModel

__all__ = ["SemiMarkovGenerator"]


class SemiMarkovGenerator(MarkovGenerator):
    """Markov token chains with token-dependent event timing."""

    name: ClassVar[str] = "semi_markov"
    tags: ClassVar[frozenset[str]] = frozenset(
        {"probabilistic", "sequence", "supports_timing", "supports_event_weights"}
    )
    defaults: ClassVar[Mapping[str, Any]] = {
        **MarkovGenerator.defaults,
        "timing_family": "empirical",
        "min_samples": 5,
    }

    def _validate_family(self, hyperparameters: dict[str, Any]) -> None:
        super()._validate_family(hyperparameters)
        _choice(
            hyperparameters,
            "timing_family",
            ("empirical", "lognormal", "gamma", "exponential"),
        )
        _number(hyperparameters, "min_samples", minimum=1, integer=True)

    def _fit_timing(
        self, dataset: Dataset, hyperparameters: Mapping[str, Any]
    ) -> dict[str, Any]:
        family = hyperparameters["timing_family"]
        minimum = hyperparameters["min_samples"]
        return {
            "pooled": fit_duration(interevent_times(dataset), family),
            "per_token": {
                token: fit_duration(values, family)
                for token, values in sojourn_times(dataset).items()
                if len(values) >= minimum
            },
        }

    def _describe_family(self, model: GeneratorModel) -> dict[str, Any]:
        timing = model.parameters["timing"]
        return {
            **super()._describe_family(model),
            "tokens_with_own_timing": len(timing["per_token"]),
        }

    def _gap_sampler(
        self, model: GeneratorModel
    ) -> Callable[[random.Random, str], float]:
        timing = model.parameters["timing"]
        pooled = DurationSampler(timing["pooled"])
        per_token = {
            token: DurationSampler(params)
            for token, params in timing.get("per_token", {}).items()
        }

        def gap(rng: random.Random, token: str) -> float:
            return per_token.get(token, pooled).sample(rng)

        return gap
