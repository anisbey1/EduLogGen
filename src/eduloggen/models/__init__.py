"""Canonical domain model for EduLogGen (PRD §9, SAD §9).

``models`` defines the interoperability contract shared by every stage:
immutable :class:`LogRecord`, :class:`Session`, and :class:`Participant`
records, :class:`Dataset` snapshots, the controlled event vocabulary, and
cross-record schema checks. It depends only on :mod:`eduloggen.core`.
"""

from __future__ import annotations

from eduloggen.models.dataset import (
    Dataset,
    DatasetMetadata,
    GenerationMetadata,
    SyntheticDataset,
)
from eduloggen.models.features import Feature, FeatureScope
from eduloggen.models.generator_model import GeneratorModel
from eduloggen.models.records import LogRecord, Participant, Session, TokenField
from eduloggen.models.schema_validate import (
    check_schema_version,
    validate_events,
    validate_sessions,
)
from eduloggen.models.vocab import EventVocabulary, UnknownEventPolicy

Event = LogRecord
"""Alias: *event* and *log record* are synonyms at the domain layer (SAD §9.5)."""

__all__ = [
    "Dataset",
    "DatasetMetadata",
    "Event",
    "EventVocabulary",
    "Feature",
    "FeatureScope",
    "GenerationMetadata",
    "GeneratorModel",
    "LogRecord",
    "Participant",
    "Session",
    "SyntheticDataset",
    "TokenField",
    "UnknownEventPolicy",
    "check_schema_version",
    "validate_events",
    "validate_sessions",
]
