"""Datasets: immutable snapshots of events and sessions (SAD §9.2, §9.9).

A :class:`Dataset` validates its events and sessions on construction and
never changes afterwards (ADR-008). Transformations such as attaching sessions
return new snapshots. :class:`SyntheticDataset` has the same shape plus the
provenance of the generator run that produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from eduloggen.__version__ import __version__
from eduloggen.core import SCHEMA_VERSION, SchemaError
from eduloggen.models import _checks as chk
from eduloggen.models.records import LogRecord, Participant, Session
from eduloggen.models.schema_validate import (
    check_schema_version,
    validate_events,
    validate_sessions,
)

__all__ = [
    "Dataset",
    "DatasetMetadata",
    "GenerationMetadata",
    "SyntheticDataset",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class DatasetMetadata:
    """Provenance of a dataset.

    Attributes:
        source_description: Human description of where the data came from.
        privacy_notes: Notes on de-identification and sharing constraints.
        eduloggen_version: Framework version that built the dataset.
        created_at: UTC creation time.
        seeds: Named seeds used for derived artifacts.
        config_fingerprint: Fingerprint of the configuration used, if any.
    """

    source_description: str = ""
    privacy_notes: str = ""
    eduloggen_version: str = __version__
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    seeds: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )
    config_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Validate fields and freeze ``seeds``.

        Raises:
            SchemaError: If any field is invalid.
        """
        name = "DatasetMetadata"
        if not isinstance(self.source_description, str):
            raise chk.invalid(name, "source_description", "must be a string")
        if not isinstance(self.privacy_notes, str):
            raise chk.invalid(name, "privacy_notes", "must be a string")
        chk.require_str(name, "eduloggen_version", self.eduloggen_version)
        chk.optional_str(name, "config_fingerprint", self.config_fingerprint)
        seeds = chk.freeze_mapping(name, "seeds", self.seeds)
        for seed in seeds.values():
            chk.non_negative_int(name, "seeds", seed)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(
            self, "created_at", chk.require_utc(name, "created_at", self.created_at)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        return {
            "source_description": self.source_description,
            "privacy_notes": self.privacy_notes,
            "eduloggen_version": self.eduloggen_version,
            "created_at": self.created_at.isoformat(),
            "seeds": dict(self.seeds),
            "config_fingerprint": self.config_fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DatasetMetadata:
        """Restore metadata serialized with :meth:`to_dict`.

        Raises:
            SchemaError: If keys are missing or unknown, or values are invalid.
        """
        fields = frozenset(cls.__dataclass_fields__)
        chk.check_keys("DatasetMetadata", data, ("created_at",), fields)
        values = dict(data)
        values["created_at"] = chk.parse_utc(
            "DatasetMetadata", "created_at", values["created_at"]
        )
        return cls(**values)


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationMetadata:
    """Provenance of a synthetic dataset (SAD §9.9).

    Attributes:
        generator_id: Registered generator name (e.g. ``"markov"``).
        model_fingerprint: Fingerprint of the fitted model used.
        seed: Seed used for sampling.
        n_sessions: Number of sessions requested.
        id_strategy: How learner and event ids were assigned.
    """

    generator_id: str
    model_fingerprint: str
    seed: int
    n_sessions: int
    id_strategy: str = "remap"

    def __post_init__(self) -> None:
        """Validate fields.

        Raises:
            SchemaError: If any field is invalid.
        """
        name = "GenerationMetadata"
        chk.require_str(name, "generator_id", self.generator_id)
        chk.require_str(name, "model_fingerprint", self.model_fingerprint)
        chk.require_str(name, "id_strategy", self.id_strategy)
        chk.non_negative_int(name, "seed", self.seed)
        if chk.non_negative_int(name, "n_sessions", self.n_sessions) == 0:
            raise chk.invalid(name, "n_sessions", "must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        return {
            "generator_id": self.generator_id,
            "model_fingerprint": self.model_fingerprint,
            "seed": self.seed,
            "n_sessions": self.n_sessions,
            "id_strategy": self.id_strategy,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Dataset:
    """An immutable, validated collection of events and optional sessions.

    Attributes:
        dataset_id: Name of the dataset.
        events: Event records, in source order.
        sessions: Session records, or ``None`` before sessionization.
        metadata: Provenance information.
        schema_version: Canonical schema version of the contents.

    Raises:
        SchemaError: On construction, if any record is invalid, identifiers
            repeat, sessions disagree with events, or the schema version is
            incompatible.
    """

    dataset_id: str
    events: tuple[LogRecord, ...]
    sessions: tuple[Session, ...] | None = None
    metadata: DatasetMetadata = field(default_factory=DatasetMetadata)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the snapshot and convert collections to tuples."""
        chk.require_str("Dataset", "dataset_id", self.dataset_id)
        chk.require_str("Dataset", "schema_version", self.schema_version)
        check_schema_version(self.schema_version)
        if not isinstance(self.metadata, DatasetMetadata):
            raise chk.invalid("Dataset", "metadata", "must be DatasetMetadata")
        events = _as_tuple("events", self.events)
        validate_events(events)
        object.__setattr__(self, "events", events)
        if self.sessions is not None:
            sessions = _as_tuple("sessions", self.sessions)
            validate_sessions(sessions, events)
            object.__setattr__(self, "sessions", sessions)

    @property
    def n_events(self) -> int:
        """Number of events."""
        return len(self.events)

    @property
    def n_sessions(self) -> int:
        """Number of sessions; zero before sessionization."""
        return len(self.sessions) if self.sessions is not None else 0

    @property
    def is_sessionized(self) -> bool:
        """Whether sessions have been attached."""
        return self.sessions is not None

    @property
    def learner_ids(self) -> frozenset[str]:
        """Distinct learner ids across events and sessions."""
        ids = {event.learner_id for event in self.events}
        ids.update(session.learner_id for session in self.sessions or ())
        return frozenset(ids)

    def sorted_events(self) -> tuple[LogRecord, ...]:
        """Events in canonical ``(learner_id, timestamp, event_id)`` order."""
        return tuple(sorted(self.events, key=lambda event: event.sort_key))

    def participants(self) -> tuple[Participant, ...]:
        """One participant per learner, sorted by ``learner_id``."""
        courses: dict[str, set[str]] = {lid: set() for lid in self.learner_ids}
        records: list[LogRecord | Session] = [*self.events, *(self.sessions or ())]
        for record in records:
            if record.course_id is not None:
                courses[record.learner_id].add(record.course_id)
        return tuple(
            Participant(learner_id=learner_id, course_ids=frozenset(ids))
            for learner_id, ids in sorted(courses.items())
        )

    def with_sessions(self, sessions: Iterable[Session]) -> Dataset:
        """Return a copy with sessions attached (or replaced).

        Args:
            sessions: Sessions consistent with this dataset's events.

        Returns:
            New validated snapshot of the same type.
        """
        return replace(self, sessions=tuple(sessions))

    def fingerprint(self) -> str:
        """Content hash of the schema version, events, and sessions.

        The hash ignores ``dataset_id``, metadata, and record order, so two
        datasets with the same records share a fingerprint.

        Returns:
            ``"sha256:"`` followed by the hex digest.

        Raises:
            SchemaError: If event metadata is not JSON-serializable.
        """
        payload = {
            "schema_version": self.schema_version,
            "events": [event.to_dict() for event in self.sorted_events()],
            "sessions": (
                None
                if self.sessions is None
                else [
                    s.to_dict()
                    for s in sorted(self.sessions, key=lambda s: s.session_id)
                ]
            ),
        }
        try:
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
        except (TypeError, ValueError) as exc:
            raise chk.invalid(
                "LogRecord", "metadata", "must be JSON-serializable"
            ) from exc
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_manifest(self) -> dict[str, Any]:
        """Describe the dataset for ``manifest.json`` (SAD §29A.3).

        Returns:
            JSON-compatible mapping with id, schema version, counts,
            fingerprint, and metadata. Contains no event rows.
        """
        return {
            "dataset_id": self.dataset_id,
            "schema_version": self.schema_version,
            "synthetic": False,
            "counts": {
                "events": self.n_events,
                "sessions": self.n_sessions,
                "learners": len(self.learner_ids),
            },
            "fingerprint": self.fingerprint(),
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class SyntheticDataset(Dataset):
    """A dataset produced by a generator, carrying generation provenance.

    Attributes:
        generation: Generator, model, seed, and id strategy used.
    """

    generation: GenerationMetadata

    def __post_init__(self) -> None:
        """Validate the snapshot and the generation metadata."""
        Dataset.__post_init__(self)
        if not isinstance(self.generation, GenerationMetadata):
            raise chk.invalid(
                "SyntheticDataset", "generation", "must be GenerationMetadata"
            )

    def to_manifest(self) -> dict[str, Any]:
        """Describe the dataset, including generation provenance."""
        manifest = Dataset.to_manifest(self)
        manifest["synthetic"] = True
        manifest["generation"] = self.generation.to_dict()
        return manifest


def _as_tuple(field_name: str, items: object) -> tuple[Any, ...]:
    if isinstance(items, str | bytes | Mapping) or not isinstance(items, Iterable):
        raise SchemaError(
            f"Dataset.{field_name} must be a sequence of records",
            code=chk.INVALID_VALUE,
            context={"record": "Dataset", "field": field_name},
        )
    return tuple(items)
