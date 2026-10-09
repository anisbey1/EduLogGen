"""Ground-truth annotations, kept apart from event records (Level 2, ADR-027).

An :class:`Annotations` table says which learners, sessions, or events carry
an injected anomaly (and, in later milestones, which profile or outcome they
have). Anything without an anomaly row is normal. Storing ground truth
separately means an algorithm under test only ever sees the events.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, get_args

from eduloggen.core import SchemaError
from eduloggen.models import _checks as chk
from eduloggen.models.dataset import Dataset

__all__ = [
    "ANNOTATION_COLUMNS",
    "Annotation",
    "AnnotationKind",
    "AnnotationLevel",
    "Annotations",
    "AnomalyCategory",
]

AnnotationLevel = Literal["learner", "session", "event"]
AnnotationKind = Literal["anomaly", "profile", "outcome"]
AnomalyCategory = Literal["invalid_workflow", "unusual_valid"]
"""``invalid_workflow`` breaks the event model; ``unusual_valid`` is rare but valid."""

ANNOTATION_COLUMNS: Final = (
    "level",
    "id",
    "annotation",
    "type",
    "category",
    "value",
    "parameters",
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Annotation:
    """One ground-truth fact about a learner, session, or event.

    Attributes:
        level: What ``id`` refers to.
        id: Learner, session, or event id.
        annotation: ``anomaly``, ``profile``, or ``outcome``.
        type: Anomaly type, profile name, or outcome target.
        category: Anomaly category; ``None`` for other annotations.
        value: ``1`` for anomalies; profile name or outcome value otherwise.
        parameters: JSON-compatible generation or injection parameters.
    """

    level: AnnotationLevel
    id: str
    annotation: AnnotationKind
    type: str
    category: AnomalyCategory | None = None
    value: str | int | float | bool = 1
    parameters: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    def __post_init__(self) -> None:
        """Validate fields.

        Raises:
            SchemaError: If a field is invalid or the category does not fit
                the annotation kind.
        """
        name = "Annotation"
        if self.level not in get_args(AnnotationLevel):
            raise chk.invalid(name, "level", "must be learner, session, or event")
        if self.annotation not in get_args(AnnotationKind):
            raise chk.invalid(
                name, "annotation", "must be anomaly, profile, or outcome"
            )
        chk.require_str(name, "id", self.id)
        chk.require_str(name, "type", self.type)
        if self.annotation == "anomaly":
            if self.category not in get_args(AnomalyCategory):
                raise chk.invalid(
                    name, "category", "must be invalid_workflow or unusual_valid"
                )
        elif self.category is not None:
            raise chk.invalid(name, "category", "is only allowed for anomalies")
        if isinstance(self.value, str) and not self.value:
            raise chk.invalid(name, "value", "must not be empty")
        frozen = chk.freeze_mapping(name, "parameters", self.parameters)
        try:
            json.dumps(dict(frozen), allow_nan=False)
        except (TypeError, ValueError):
            raise chk.invalid(name, "parameters", "must be JSON-serializable") from None
        object.__setattr__(self, "parameters", frozen)

    def to_row(self) -> dict[str, str]:
        """Serialize to a flat row of strings (CSV-friendly)."""
        return {
            "level": self.level,
            "id": self.id,
            "annotation": self.annotation,
            "type": self.type,
            "category": self.category or "",
            "value": json.dumps(self.value),
            "parameters": json.dumps(dict(self.parameters), sort_keys=True),
        }

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> Annotation:
        """Restore an annotation written with :meth:`to_row`.

        Raises:
            SchemaError: If columns are missing or values are invalid.
        """
        chk.check_keys(
            "Annotation", row, ANNOTATION_COLUMNS[:4], frozenset(ANNOTATION_COLUMNS)
        )
        try:
            value = json.loads(row.get("value") or "1")
            parameters = json.loads(row.get("parameters") or "{}")
        except (TypeError, json.JSONDecodeError):
            raise chk.invalid("Annotation", "value", "is not valid JSON") from None
        return cls(
            level=row["level"],
            id=row["id"],
            annotation=row["annotation"],
            type=row["type"],
            category=row.get("category") or None,
            value=value,
            parameters=parameters,
        )


@dataclass(frozen=True, slots=True)
class Annotations:
    """An immutable table of :class:`Annotation` rows.

    Attributes:
        rows: Annotations in insertion order.
    """

    rows: tuple[Annotation, ...] = ()

    def __post_init__(self) -> None:
        """Freeze rows and reject duplicates of the same fact."""
        rows = tuple(self.rows)
        seen: set[tuple[str, str, str, str]] = set()
        for row in rows:
            if not isinstance(row, Annotation):
                raise chk.invalid(
                    "Annotations", "rows", "must contain Annotation items"
                )
            key = (row.level, row.id, row.annotation, row.type)
            if key in seen:
                raise SchemaError(
                    "duplicate annotation",
                    code=chk.DUPLICATE_ID,
                    context={"level": row.level, "id": row.id, "type": row.type},
                )
            seen.add(key)
        object.__setattr__(self, "rows", rows)

    def __len__(self) -> int:
        """Number of rows."""
        return len(self.rows)

    def __iter__(self) -> Iterator[Annotation]:
        """Iterate over rows."""
        return iter(self.rows)

    def __add__(self, other: Annotations) -> Annotations:
        """Concatenate two tables."""
        return Annotations((*self.rows, *other.rows))

    def anomalies(
        self,
        level: AnnotationLevel | None = None,
        *,
        type: str | None = None,
        category: AnomalyCategory | None = None,
    ) -> tuple[Annotation, ...]:
        """Anomaly rows, optionally filtered."""
        return tuple(
            row
            for row in self.rows
            if row.annotation == "anomaly"
            and (level is None or row.level == level)
            and (type is None or row.type == type)
            and (category is None or row.category == category)
        )

    def anomalous_ids(
        self,
        level: AnnotationLevel,
        *,
        type: str | None = None,
        category: AnomalyCategory | None = None,
    ) -> frozenset[str]:
        """Ids at ``level`` that carry at least one matching anomaly."""
        return frozenset(
            r.id for r in self.anomalies(level, type=type, category=category)
        )

    def remap(
        self, mapping: Mapping[AnnotationLevel, Mapping[str, str]]
    ) -> Annotations:
        """Translate ids through per-level old-to-new mappings.

        Raises:
            SchemaError: If an id has no entry in its level's mapping.
        """
        rows = []
        for row in self.rows:
            table = mapping.get(row.level, {})
            if row.id not in table:
                raise SchemaError(
                    "annotation refers to an id missing from the mapping",
                    code=chk.INCONSISTENT,
                    context={"level": row.level},
                )
            rows.append(
                Annotation(
                    level=row.level,
                    id=table[row.id],
                    annotation=row.annotation,
                    type=row.type,
                    category=row.category,
                    value=row.value,
                    parameters=row.parameters,
                )
            )
        return Annotations(tuple(rows))

    def check_against(self, dataset: Dataset) -> None:
        """Ensure every annotated id exists in ``dataset``.

        Raises:
            SchemaError: If an id is unknown at its level.
        """
        known: dict[str, frozenset[str]] = {
            "learner": dataset.learner_ids,
            "session": frozenset(s.session_id for s in dataset.sessions or ()),
            "event": frozenset(e.event_id for e in dataset.events),
        }
        for row in self.rows:
            if row.id not in known[row.level]:
                raise SchemaError(
                    f"annotation refers to an unknown {row.level}",
                    code=chk.INCONSISTENT,
                    context={"level": row.level, "id": row.id},
                )

    def to_rows(self) -> list[dict[str, str]]:
        """All rows as CSV-friendly dicts, sorted for stable output."""
        return [
            row.to_row()
            for row in sorted(
                self.rows, key=lambda r: (r.level, r.id, r.annotation, r.type)
            )
        ]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> Annotations:
        """Restore a table from :meth:`to_rows` output."""
        return cls(tuple(Annotation.from_row(row) for row in rows))

    def fingerprint(self) -> str:
        """Order-independent content hash, ``"sha256:<hex>"``."""
        encoded = json.dumps(self.to_rows(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def summary(self) -> dict[str, Any]:
        """Counts of anomalies per level, type, and category."""
        counts: dict[str, dict[str, int]] = {}
        for row in self.anomalies():
            per_level = counts.setdefault(row.level, {})
            per_level[row.type] = per_level.get(row.type, 0) + 1
        categories: dict[str, int] = {}
        for row in self.anomalies("session"):
            categories[row.category or ""] = categories.get(row.category or "", 0) + 1
        return {
            "anomalies": counts,
            "session_categories": categories,
            "rows": len(self),
        }
