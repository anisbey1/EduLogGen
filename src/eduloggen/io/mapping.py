"""Map source columns onto the canonical event schema (FR-I.2, SAD §29A.2).

A :class:`FieldMapping` names, for each canonical field, the source column it
comes from plus optional transforms: timestamp parsing with a timezone policy,
a default for missing values, and salted hashing of identifiers. Values are
cast to the canonical field's type automatically. Source columns that are not
mapped are dropped unless listed under ``metadata``.

Example mapping (as a dict, e.g. loaded from YAML)::

    {
        "fields": {
            "learner_id": {"source": "user", "hash": {"salt": "s3cret"}},
            "timestamp": {"source": "time", "format": "%d/%m/%Y %H:%M",
                          "timezone": "Europe/Paris"},
            "activity_id": "item",
            "event_type": {"source": "action", "default": "other"},
            "score": "grade",
        },
        "metadata": ["device"],
        "drop": ["email"],
    }

``event_id`` may be left unmapped; ids are then generated from row numbers.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import tzinfo
from types import MappingProxyType
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from eduloggen.core import EVENT_REQUIRED_FIELDS, ConfigError
from eduloggen.io import coerce

__all__ = ["FieldMapping", "FieldSpec", "MappedRow", "RowIssue"]

_STRING_FIELDS: Final = frozenset(
    {"event_id", "learner_id", "activity_id", "event_type", "session_id", "course_id"}
)
_CASTS: Final[dict[str, Callable[[object], object]]] = {
    **dict.fromkeys(_STRING_FIELDS, coerce.to_str),
    "score": coerce.to_float,
    "success": coerce.to_bool,
    "duration_ms": coerce.to_int,
}
MAPPABLE_FIELDS: Final = frozenset({*_CASTS, "timestamp"})
_SPEC_KEYS: Final = frozenset({"source", "default", "format", "timezone", "hash"})
_MAPPING_KEYS: Final = frozenset({"fields", "metadata", "drop", "timezone"})


@dataclass(frozen=True, slots=True)
class RowIssue:
    """A problem with one source row. Holds no source values.

    Attributes:
        row: 1-based row number in the source.
        field: Canonical field concerned.
        code: Reason code (e.g. ``"missing_required"``, ``"naive_timestamp"``).
    """

    row: int
    field: str
    code: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        return {"row": self.row, "field": self.field, "code": self.code}


@dataclass(frozen=True, slots=True)
class MappedRow:
    """Outcome of mapping one source row.

    Attributes:
        values: Canonical field values ready for :class:`LogRecord`, or
            ``None`` if the row had issues.
        issues: Problems found; empty when ``values`` is set.
        assumed_timezone: Whether a naive timestamp was given a timezone.
    """

    values: dict[str, Any] | None
    issues: tuple[RowIssue, ...] = ()
    assumed_timezone: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class FieldSpec:
    """How to fill one canonical field.

    Attributes:
        source: Source column name.
        default: Value used when the source value is missing.
        format: Timestamp format: ``"iso"``, ``"epoch_s"``, ``"epoch_ms"``, or
            a ``strptime`` pattern. Only valid for ``timestamp``.
        timezone: IANA zone assumed for naive timestamps; overrides the
            mapping-wide timezone. Only valid for ``timestamp``.
        hash_salt: If set, string values are replaced by a salted
            HMAC-SHA256 pseudonym. Only valid for identifier fields.
    """

    source: str
    default: Any = None
    format: str = coerce.ISO
    timezone: str | None = None
    hash_salt: str | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class FieldMapping:
    """Source-to-canonical mapping for event rows.

    Attributes:
        fields: Canonical field name to :class:`FieldSpec`.
        metadata: Source columns copied into the event ``metadata`` bag.
        drop: Source columns deliberately ignored (silences warnings).
        timezone: IANA zone assumed for naive timestamps; ``None`` rejects
            naive timestamps.
    """

    fields: Mapping[str, FieldSpec]
    metadata: tuple[str, ...] = ()
    drop: tuple[str, ...] = ()
    timezone: str | None = None
    _zones: Mapping[str, tzinfo | None] = field(
        init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        """Validate the mapping.

        Raises:
            ConfigError: If a field is unknown, a required field is unmapped,
                a transform is not allowed for its field, a timezone is
                unknown, or a source column is used for two purposes.
        """
        unknown = sorted(set(self.fields) - MAPPABLE_FIELDS)
        if unknown:
            raise ConfigError(
                f"mapping names unknown canonical fields: {', '.join(unknown)}",
                code="mapping_unknown_field",
                context={"fields": unknown, "allowed": sorted(MAPPABLE_FIELDS)},
            )
        missing = [
            name
            for name in EVENT_REQUIRED_FIELDS
            if name != "event_id" and name not in self.fields
        ]
        if missing:
            raise ConfigError(
                f"required canonical fields are unmapped: {', '.join(missing)}",
                code="mapping_missing_required",
                context={"fields": missing},
            )
        for name, spec in self.fields.items():
            _check_spec(name, spec)
        overlap = sorted(
            (set(self.metadata) | set(self.drop)) & set(self.sources().values())
            | set(self.metadata) & set(self.drop)
        )
        if overlap:
            raise ConfigError(
                "source columns are used more than once",
                code="mapping_conflict",
                context={"columns": overlap},
            )
        zones = {
            name: _zone(spec.timezone or self.timezone)
            for name, spec in self.fields.items()
            if name == "timestamp"
        }
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        object.__setattr__(self, "metadata", tuple(self.metadata))
        object.__setattr__(self, "drop", tuple(self.drop))
        object.__setattr__(self, "_zones", MappingProxyType(zones))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FieldMapping:
        """Build a mapping from plain data (e.g. parsed YAML or JSON).

        Each entry under ``fields`` is either a source column name or a mapping
        with keys ``source``, ``default``, ``format``, ``timezone``, and
        ``hash`` (``{"salt": ...}``).

        Raises:
            ConfigError: If the structure or any value is invalid.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("mapping must be a mapping", code="mapping_invalid")
        _reject_unknown("mapping", data, _MAPPING_KEYS)
        raw_fields = data.get("fields")
        if not isinstance(raw_fields, Mapping) or not raw_fields:
            raise ConfigError(
                "mapping.fields must be a non-empty mapping", code="mapping_invalid"
            )
        specs = {
            str(name): _parse_spec(str(name), raw) for name, raw in raw_fields.items()
        }
        return cls(
            fields=specs,
            metadata=_str_list("metadata", data.get("metadata", ())),
            drop=_str_list("drop", data.get("drop", ())),
            timezone=_optional_text("timezone", data.get("timezone")),
        )

    @classmethod
    def identity(cls) -> FieldMapping:
        """Mapping for sources already using canonical column names."""
        return cls(fields={name: FieldSpec(source=name) for name in MAPPABLE_FIELDS})

    def sources(self) -> dict[str, str]:
        """Canonical field name to source column name."""
        return {name: spec.source for name, spec in self.fields.items()}

    def known_columns(self) -> frozenset[str]:
        """Source columns this mapping uses, keeps, or deliberately drops."""
        return frozenset(self.sources().values()) | set(self.metadata) | set(self.drop)

    def missing_columns(self, columns: Iterable[str]) -> list[str]:
        """Required source columns absent from ``columns``.

        Columns for fields with a default are not required.
        """
        present = set(columns)
        return sorted(
            spec.source
            for name, spec in self.fields.items()
            if name in EVENT_REQUIRED_FIELDS
            and spec.default is None
            and spec.source not in present
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for provenance (``mapping.used``); salts are redacted."""
        fields: dict[str, Any] = {}
        for name, spec in self.fields.items():
            entry: dict[str, Any] = {"source": spec.source}
            if spec.default is not None:
                entry["default"] = spec.default
            if spec.format != coerce.ISO:
                entry["format"] = spec.format
            if spec.timezone is not None:
                entry["timezone"] = spec.timezone
            if spec.hash_salt is not None:
                entry["hash"] = {"salt": "<redacted>"}
            fields[name] = entry
        return {
            "fields": fields,
            "metadata": list(self.metadata),
            "drop": list(self.drop),
            "timezone": self.timezone,
        }

    def apply(self, row: Mapping[str, Any], row_number: int) -> MappedRow:
        """Map one source row to canonical values.

        Args:
            row: Source column name to raw value.
            row_number: 1-based position of the row in the source.

        Returns:
            The mapped values, or the issues that prevented mapping.
        """
        values: dict[str, Any] = {}
        issues: list[RowIssue] = []
        assumed = False
        for name, spec in self.fields.items():
            raw = row.get(spec.source)
            if coerce.is_missing(raw):
                raw = spec.default
            if coerce.is_missing(raw):
                if name in EVENT_REQUIRED_FIELDS:
                    issues.append(RowIssue(row_number, name, "missing_required"))
                continue
            try:
                if name == "timestamp":
                    values[name], assumed = coerce.to_datetime(
                        raw, fmt=spec.format, assume_tz=self._zones[name]
                    )
                else:
                    value = _CASTS[name](raw)
                    if spec.hash_salt is not None:
                        value = pseudonymize(str(value), spec.hash_salt)
                    values[name] = value
            except coerce.CoercionError as exc:
                issues.append(RowIssue(row_number, name, exc.reason))
        if "event_id" not in self.fields:
            values["event_id"] = f"row-{row_number}"
        if self.metadata:
            values["metadata"] = {
                column: row[column]
                for column in self.metadata
                if not coerce.is_missing(row.get(column))
            }
        if issues:
            return MappedRow(values=None, issues=tuple(issues))
        return MappedRow(values=values, assumed_timezone=assumed)


def pseudonymize(value: str, salt: str) -> str:
    """Return a stable salted pseudonym for an identifier.

    The same value and salt always give the same pseudonym; without the salt
    the original cannot be recovered by dictionary lookup.
    """
    digest = hmac.new(salt.encode("utf-8"), value.encode("utf-8"), hashlib.sha256)
    return "h_" + digest.hexdigest()[:32]


def _check_spec(name: str, spec: FieldSpec) -> None:
    if not isinstance(spec, FieldSpec):
        raise ConfigError(
            f"mapping for {name!r} must be a FieldSpec", code="mapping_invalid"
        )
    if not isinstance(spec.source, str) or not spec.source:
        raise ConfigError(
            f"mapping for {name!r} needs a source column", code="mapping_invalid"
        )
    if name != "timestamp" and (spec.format != coerce.ISO or spec.timezone):
        raise ConfigError(
            f"format/timezone are only valid for timestamp, not {name!r}",
            code="mapping_invalid",
            context={"field": name},
        )
    if spec.hash_salt is not None:
        if name not in _STRING_FIELDS:
            raise ConfigError(
                f"hash is only valid for identifier fields, not {name!r}",
                code="mapping_invalid",
                context={"field": name},
            )
        if not spec.hash_salt:
            raise ConfigError(
                f"hash for {name!r} needs a non-empty salt",
                code="mapping_invalid",
                context={"field": name},
            )


def _parse_spec(name: str, raw: object) -> FieldSpec:
    if isinstance(raw, str):
        return FieldSpec(source=raw)
    if not isinstance(raw, Mapping):
        raise ConfigError(
            f"mapping for {name!r} must be a column name or a mapping",
            code="mapping_invalid",
            context={"field": name},
        )
    _reject_unknown(f"fields.{name}", raw, _SPEC_KEYS)
    salt: str | None = None
    if "hash" in raw:
        hash_cfg = raw["hash"]
        if not isinstance(hash_cfg, Mapping) or set(hash_cfg) != {"salt"}:
            raise ConfigError(
                f"fields.{name}.hash must be {{'salt': ...}}",
                code="mapping_invalid",
                context={"field": name},
            )
        salt = _optional_text(f"fields.{name}.hash.salt", hash_cfg["salt"]) or ""
    return FieldSpec(
        source=_optional_text(f"fields.{name}.source", raw.get("source")) or "",
        default=raw.get("default"),
        format=_optional_text(f"fields.{name}.format", raw.get("format")) or coerce.ISO,
        timezone=_optional_text(f"fields.{name}.timezone", raw.get("timezone")),
        hash_salt=salt,
    )


def _reject_unknown(
    where: str, data: Mapping[str, Any], allowed: frozenset[str]
) -> None:
    unknown = sorted(str(key) for key in set(data) - allowed)
    if unknown:
        raise ConfigError(
            f"{where} has unknown keys: {', '.join(unknown)}",
            code="mapping_unknown_key",
            context={"where": where, "keys": unknown, "allowed": sorted(allowed)},
        )


def _optional_text(where: str, value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{where} must be a string", code="mapping_invalid")
    return value


def _str_list(where: str, value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ConfigError(f"{where} must be a list of strings", code="mapping_invalid")
    if not all(isinstance(item, str) and item for item in value):
        raise ConfigError(f"{where} must be a list of strings", code="mapping_invalid")
    return tuple(value)


def _zone(name: str | None) -> tzinfo | None:
    if name is None:
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        raise ConfigError(
            "unknown IANA timezone",
            code="mapping_invalid",
            context={"timezone": name},
        ) from None
