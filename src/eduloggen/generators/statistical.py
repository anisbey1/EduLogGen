"""Statistical baseline: independent tokens (SAD §12.4, "Statistical").

Tokens are drawn independently from their overall frequencies, ignoring
order. It reproduces event-type marginals but no sequential structure, which
makes it a useful lower bound in benchmarks: a sequence model should beat it
on transition metrics.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Mapping
from typing import Any, ClassVar

from eduloggen.analysis import interevent_times
from eduloggen.core import GenerationError
from eduloggen.generators.base import BaseGenerator, SequenceSampler
from eduloggen.generators.distributions import (
    Categorical,
    DurationSampler,
    fit_duration,
)
from eduloggen.models import Dataset, GeneratorModel

__all__ = ["IndependentGenerator"]


class IndependentGenerator(BaseGenerator):
    """Sessions of independently drawn tokens."""

    name: ClassVar[str] = "independent"
    tags: ClassVar[frozenset[str]] = frozenset({"statistical", "baseline"})

    def _fit_family(
        self,
        dataset: Dataset,
        sequences: tuple[tuple[str, ...], ...],
        hyperparameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "unigram": dict(sorted(Counter(t for s in sequences for t in s).items())),
            "timing": {"pooled": fit_duration(interevent_times(dataset), "constant")},
        }

    def _sequence_sampler(self, model: GeneratorModel) -> SequenceSampler:
        try:
            unigram = Categorical.from_counts(model.parameters["unigram"])
            gap = DurationSampler(model.parameters["timing"]["pooled"])
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise GenerationError(
                "model parameters are malformed", code="generation_invalid_model"
            ) from exc

        def sample(rng: random.Random, length: int) -> tuple[list[str], list[float]]:
            tokens = [unigram.sample(rng) for _ in range(length)]
            return tokens, [gap.sample(rng) for _ in tokens[:-1]]

        return sample
