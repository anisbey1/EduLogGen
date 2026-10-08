"""The generator contract and shared fit/generate machinery (SAD §12.1-12.3).

Subclasses only model token sequences and within-session timing:

- :meth:`BaseGenerator._fit_family` returns family-specific parameters.
- :meth:`BaseGenerator._sequence_sampler` turns them into a function that,
  given a length, draws tokens and the gaps (seconds) between them.

Everything else is shared: hyperparameter validation, session lengths, how
many sessions each synthetic learner has, gaps between a learner's sessions,
session start times, companion fields and courses, id remapping, and saving.
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar, Final

from eduloggen.analysis import session_sequences
from eduloggen.core import ConfigError, FitError, GenerationError, PathLike
from eduloggen.generators.artifact import load_model, save_model
from eduloggen.generators.distributions import (
    Categorical,
    DurationSampler,
    LengthSampler,
    fit_length,
    sketch,
)
from eduloggen.generators.tokenization import (
    companion_counts,
    companion_field,
    detect_tokenization,
)
from eduloggen.models import (
    Dataset,
    DatasetMetadata,
    GenerationMetadata,
    GeneratorModel,
    LogRecord,
    Session,
    SyntheticDataset,
    TokenField,
)
from eduloggen.privacy import IdStrategy, apply_id_strategy
from eduloggen.utils import derive_seed, fingerprint, make_rng

__all__ = ["BaseGenerator", "SequenceSampler"]

logger = logging.getLogger(__name__)

SequenceSampler = Callable[[random.Random, int], tuple[list[str], list[float]]]
"""Draws ``length`` tokens and ``length - 1`` non-negative gaps in seconds."""

MIN_GAP_S: Final = 0.001
"""Smallest gap between events, keeping timestamps strictly increasing."""

DEFAULT_SESSION_GAP_S: Final = 86_400.0
"""Gap between sessions when training learners never had two sessions."""

_BASE_DEFAULTS: Final[dict[str, Any]] = {
    "length_model": "empirical",
    "fixed_length": None,
}


class BaseGenerator(ABC):
    """Fit a model of learner sessions and sample synthetic datasets.

    Class attributes:
        name: Registry name.
        version: Implementation version, stored in artifacts.
        tags: Capability tags (e.g. ``"supports_timing"``).
        defaults: Family hyperparameters and their defaults.
    """

    name: ClassVar[str]
    version: ClassVar[str] = "1.0"
    tags: ClassVar[frozenset[str]] = frozenset()
    defaults: ClassVar[Mapping[str, Any]] = {}

    # -- hyperparameters ---------------------------------------------------

    def validate_hyperparameters(
        self, hyperparameters: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Merge with defaults and validate.

        Raises:
            ConfigError: If a key is unknown or a value is invalid.
        """
        merged = {**_BASE_DEFAULTS, **self.defaults}
        unknown = sorted(set(hyperparameters or {}) - set(merged))
        if unknown:
            raise ConfigError(
                f"unknown hyperparameters for {self.name}: {', '.join(unknown)}",
                code="config_unknown_key",
                context={
                    "generator": self.name,
                    "keys": unknown,
                    "allowed": sorted(merged),
                },
            )
        merged.update(hyperparameters or {})
        _choice(merged, "length_model", ("empirical", "poisson", "fixed"))
        fixed = merged["fixed_length"]
        if fixed is not None and (_not_int(fixed) or fixed < 1):
            raise _bad("fixed_length", "must be a positive integer")
        if merged["length_model"] == "fixed" and fixed is None:
            raise _bad("fixed_length", "is required when length_model is 'fixed'")
        self._validate_family(merged)
        return merged

    def _validate_family(  # noqa: B027 - optional hook
        self, hyperparameters: dict[str, Any]
    ) -> None:
        """Validate family hyperparameters in place (override as needed)."""

    # -- fit ----------------------------------------------------------------

    def fit(
        self, dataset: Dataset, hyperparameters: Mapping[str, Any] | None = None
    ) -> GeneratorModel:
        """Fit the generator on a sessionized dataset.

        Args:
            dataset: Training data; must be sessionized and non-empty.
            hyperparameters: Overrides of :attr:`defaults`.

        Returns:
            The fitted, JSON-serializable model.

        Raises:
            ConfigError: If hyperparameters are invalid.
            FitError: If the dataset is not sessionized, empty, or does not
                support the requested model.
        """
        hp = self.validate_hyperparameters(hyperparameters)
        if dataset.sessions is None:
            raise FitError(
                "dataset must be sessionized before fitting", code="fit_not_sessionized"
            )
        if not dataset.sessions:
            raise FitError("dataset has no sessions", code="fit_empty_corpus")
        sequences = session_sequences(dataset)
        tokenization = detect_tokenization(dataset)
        family = self._fit_family(dataset, sequences, hp)
        parameters = {
            "population": _fit_population(dataset, tokenization),
            "length": fit_length(
                [len(s) for s in sequences], hp["length_model"], hp["fixed_length"]
            ),
            **family,
        }
        training = fingerprint(
            {
                "dataset": dataset.fingerprint(),
                "generator": self.name,
                "version": self.version,
                "hyperparameters": hp,
            }
        )
        logger.info(
            "fitted %s on %d sessions (%d tokens)",
            self.name,
            len(sequences),
            len({t for s in sequences for t in s}),
        )
        return GeneratorModel(
            generator_id=self.name,
            generator_version=self.version,
            hyperparameters=hp,
            vocabulary=tuple(sorted({t for s in sequences for t in s})),
            parameters=parameters,
            training_fingerprint=training,
        )

    @abstractmethod
    def _fit_family(
        self,
        dataset: Dataset,
        sequences: tuple[tuple[str, ...], ...],
        hyperparameters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return family-specific parameters (JSON-compatible)."""

    @abstractmethod
    def _sequence_sampler(self, model: GeneratorModel) -> SequenceSampler:
        """Build the token/gap sampler for a fitted model."""

    # -- generate -----------------------------------------------------------

    def generate(
        self,
        model: GeneratorModel,
        n_sessions: int,
        seed: int,
        *,
        id_strategy: IdStrategy = "remap",
        start_time: datetime | None = None,
        dataset_id: str | None = None,
    ) -> SyntheticDataset:
        """Sample a synthetic, sessionized dataset.

        Args:
            model: A model fitted by this generator.
            n_sessions: Number of sessions to generate.
            seed: Seed; the same model, seed, and arguments give the same
                output.
            id_strategy: ``remap`` shuffles synthetic ids; ``preserve``
                keeps them in generation order.
            start_time: Earliest possible session start; defaults to the
                earliest start seen in training.
            dataset_id: Name of the output; defaults to ``<generator>-synthetic``.

        Returns:
            The synthetic dataset with sessions attached.

        Raises:
            GenerationError: If the model belongs to another generator, is
                malformed, or an argument is invalid.
        """
        self._check_generate_args(model, n_sessions, seed, start_time)
        rng = make_rng(seed, "generate", self.name)
        sampler = self._sequence_sampler(model)
        events, sessions = _assemble(model, n_sessions, rng, sampler, start_time)
        synthetic = SyntheticDataset(
            dataset_id=dataset_id or f"{self.name}-synthetic",
            events=tuple(events),
            sessions=tuple(sessions),
            metadata=DatasetMetadata(
                source_description=f"synthetic sample from {self.name}",
                privacy_notes=(
                    "Synthetic data with generated identifiers. Synthetic data "
                    "can still resemble real sessions; check privacy indicators "
                    "before sharing."
                ),
                seeds={"generate": seed},
            ),
            generation=GenerationMetadata(
                generator_id=self.name,
                model_fingerprint=model.fingerprint(),
                seed=seed,
                n_sessions=n_sessions,
                id_strategy=id_strategy,
            ),
        )
        return apply_id_strategy(
            synthetic, id_strategy, derive_seed(seed, "generate", "ids")
        )

    sample = generate
    """Alias of :meth:`generate` (PRD naming)."""

    def _check_generate_args(
        self,
        model: GeneratorModel,
        n_sessions: int,
        seed: int,
        start_time: datetime | None,
    ) -> None:
        self._check_owner(model)
        if _not_int(n_sessions) or n_sessions < 1:
            raise GenerationError(
                "n_sessions must be a positive integer",
                code="generation_invalid_argument",
            )
        if _not_int(seed) or seed < 0:
            raise GenerationError(
                "seed must be a non-negative integer",
                code="generation_invalid_argument",
            )
        if start_time is not None and start_time.utcoffset() is None:
            raise GenerationError(
                "start_time must be timezone-aware",
                code="generation_invalid_argument",
            )

    # -- inspection and persistence ----------------------------------------

    def _check_owner(self, model: GeneratorModel) -> None:
        if model.generator_id != self.name:
            raise GenerationError(
                f"model was fitted by {model.generator_id!r}, not {self.name!r}",
                code="generation_model_mismatch",
            )

    def describe(self, model: GeneratorModel) -> dict[str, Any]:
        """Summarize a fitted model for logs and papers.

        Raises:
            GenerationError: If the model belongs to another generator.
        """
        self._check_owner(model)
        population = model.parameters["population"]
        return {
            "generator": model.generator_id,
            "generator_version": model.generator_version,
            "hyperparameters": dict(model.hyperparameters),
            "vocabulary_size": len(model.vocabulary),
            "tokenization": population["tokenization"],
            "training_sessions": population["n_sessions"],
            "training_learners": population["n_learners"],
            "length_model": model.parameters["length"]["model"],
            "training_fingerprint": model.training_fingerprint,
            "model_fingerprint": model.fingerprint(),
            **self._describe_family(model),
        }

    def _describe_family(self, model: GeneratorModel) -> dict[str, Any]:
        """Family-specific description entries (override as needed)."""
        return {}

    def save(
        self, model: GeneratorModel, path: PathLike, *, force: bool = False
    ) -> Path:
        """Write a model artifact directory (see :mod:`.artifact`).

        Raises:
            GenerationError: If the model belongs to another generator.
            ExportError: If the target is unsafe or exists without ``force``.
        """
        return save_model(model, path, description=self.describe(model), force=force)

    def load(self, path: PathLike) -> GeneratorModel:
        """Read a model artifact written by this generator.

        Raises:
            GenerationError: If the artifact belongs to another generator.
        """
        model = load_model(path)
        self._check_owner(model)
        return model


# ---------------------------------------------------------------------------
# Population model shared by all generators
# ---------------------------------------------------------------------------


def _fit_population(dataset: Dataset, tokenization: TokenField) -> dict[str, Any]:
    sessions = sorted(
        dataset.sessions or (), key=lambda s: (s.learner_id, s.start_time)
    )
    per_learner: defaultdict[str, list[Session]] = defaultdict(list)
    for session in sessions:
        per_learner[session.learner_id].append(session)
    gaps = [
        (later.start_time - earlier.end_time).total_seconds()
        for owned in per_learner.values()
        for earlier, later in pairwise(owned)
    ]
    starts = [session.start_time for session in sessions]
    courses = Counter(session.course_id or "" for session in sessions)
    return {
        "tokenization": tokenization,
        "companion_field": companion_field(tokenization),
        "companions": companion_counts(dataset, tokenization),
        "courses": dict(sorted(courses.items())),
        "sessions_per_learner": {
            str(k): v
            for k, v in sorted(Counter(len(o) for o in per_learner.values()).items())
        },
        "session_gaps_s": sketch(max(gap, 0.0) for gap in gaps),
        "time_origin": min(starts).isoformat(),
        "time_span_s": (max(starts) - min(starts)).total_seconds(),
        "n_sessions": len(sessions),
        "n_learners": len(per_learner),
    }


def _assemble(
    model: GeneratorModel,
    n_sessions: int,
    rng: random.Random,
    sampler: SequenceSampler,
    start_time: datetime | None,
) -> tuple[list[LogRecord], list[Session]]:
    params = model.parameters
    try:
        population = params["population"]
        tokenization: TokenField = population["tokenization"]
        other: TokenField = population["companion_field"]
        companions = {
            token: Categorical.from_counts(counts)
            for token, counts in population["companions"].items()
        }
        courses = Categorical.from_counts(population["courses"])
        per_learner = Categorical.from_counts(
            {int(k): v for k, v in population["sessions_per_learner"].items()}
        )
        gap_points = population["session_gaps_s"]
        session_gaps = DurationSampler(
            {"family": "empirical", "points": gap_points}
            if gap_points
            else {"family": "constant", "seconds": DEFAULT_SESSION_GAP_S}
        )
        origin = start_time or datetime.fromisoformat(population["time_origin"])
        span = float(population["time_span_s"])
        lengths = LengthSampler(params["length"])
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise GenerationError(
            "model parameters are malformed", code="generation_invalid_model"
        ) from exc

    origin = origin.astimezone(UTC)
    events: list[LogRecord] = []
    sessions: list[Session] = []
    learner = 0
    while len(sessions) < n_sessions:
        learner += 1
        learner_id = f"L{learner}"
        course = courses.sample(rng) or None
        clock = origin + timedelta(seconds=rng.uniform(0.0, span))
        count = min(per_learner.sample(rng), n_sessions - len(sessions))
        for _ in range(count):
            session_id = f"S{len(sessions) + 1}"
            tokens, gaps = sampler(rng, lengths.sample(rng))
            members: list[LogRecord] = []
            for index, token in enumerate(tokens):
                if index:
                    clock += timedelta(seconds=max(gaps[index - 1], MIN_GAP_S))
                companion = companions.get(token)
                if companion is None:
                    raise GenerationError(
                        "sampled a token the model has no companion data for",
                        code="generation_invalid_model",
                    )
                fields = {tokenization: token, other: companion.sample(rng)}
                members.append(
                    LogRecord(
                        event_id=f"E{len(events) + len(members) + 1}",
                        learner_id=learner_id,
                        timestamp=clock,
                        session_id=session_id,
                        course_id=course,
                        **fields,
                    )
                )
            events.extend(members)
            sessions.append(
                Session.from_events(session_id, members, token_field=tokenization)
            )
            clock += timedelta(seconds=max(session_gaps.sample(rng), MIN_GAP_S))
    return events, sessions


# ---------------------------------------------------------------------------
# Hyperparameter helpers for subclasses
# ---------------------------------------------------------------------------


def _not_int(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int)


def _bad(key: str, reason: str) -> ConfigError:
    return ConfigError(
        f"generator.{key} {reason}",
        code="config_invalid_value",
        context={"key": f"generator.{key}"},
    )


def _choice(hp: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> None:
    if hp[key] not in allowed:
        raise _bad(key, f"must be one of {', '.join(allowed)}")


def _number(
    hp: Mapping[str, Any], key: str, *, minimum: float, integer: bool = False
) -> None:
    value = hp[key]
    valid_type = (
        not _not_int(value)
        if integer
        else (not isinstance(value, bool) and isinstance(value, int | float))
    )
    if not valid_type or value < minimum:
        kind = "an integer" if integer else "a number"
        raise _bad(key, f"must be {kind} >= {minimum:g}")
