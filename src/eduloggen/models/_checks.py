"""Field-level checks shared by domain model constructors.

Every check raises :class:`~eduloggen.core.SchemaError` with a stable code and
a context naming the record type and field. Values are never copied into the
error context, so learner data cannot leak through exception messages.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from eduloggen.core import SchemaError

MISSING_FIELD = "schema_missing_field"
INVALID_VALUE = "schema_invalid_value"
UNKNOWN_FIELD = "schema_unknown_field"
DUPLICATE_ID = "schema_duplicate_id"
INCONSISTENT = "schema_inconsistent"
UNSUPPORTED_VERSION = "schema_unsupported_version"


def invalid(record: str, field: str, reason: str) -> SchemaError:
    """Build an error for a field holding an invalid value."""
    return SchemaError(
        f"{record}.{field} {reason}",
        code=INVALID_VALUE,
        context={"record": record, "field": field},
    )


def inconsistent(record: str, field: str, reason: str) -> SchemaError:
    """Build an error for a field that contradicts related data."""
    return SchemaError(
        f"{record}.{field} {reason}",
        code=INCONSISTENT,
        context={"record": record, "field": field},
    )


def require_str(record: str, field: str, value: object) -> str:
    """Return ``value`` if it is a non-empty string."""
    if not isinstance(value, str) or not value:
        raise invalid(record, field, "must be a non-empty string")
    return value


def optional_str(record: str, field: str, value: object) -> str | None:
    """Return ``value`` if it is ``None`` or a non-empty string."""
    return None if value is None else require_str(record, field, value)


def require_utc(record: str, field: str, value: object) -> datetime:
    """Return ``value`` converted to UTC; naive datetimes are rejected."""
    if not isinstance(value, datetime):
        raise invalid(record, field, "must be a datetime")
    if value.utcoffset() is None:
        raise invalid(record, field, "must be timezone-aware")
    return value.astimezone(UTC)


def parse_utc(record: str, field: str, value: object) -> datetime:
    """Parse an ISO 8601 string or datetime into a UTC datetime."""
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except ValueError:
            raise invalid(record, field, "is not an ISO 8601 timestamp") from None
    return require_utc(record, field, value)


def optional_float(record: str, field: str, value: object) -> float | None:
    """Return ``value`` as a finite float, or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise invalid(record, field, "must be a number")
    if not math.isfinite(value):
        raise invalid(record, field, "must be finite")
    return float(value)


def non_negative_int(record: str, field: str, value: object) -> int:
    """Return ``value`` if it is an integer greater than or equal to zero."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise invalid(record, field, "must be a non-negative integer")
    return value


def optional_bool(record: str, field: str, value: object) -> bool | None:
    """Return ``value`` if it is ``None`` or a boolean."""
    if value is not None and not isinstance(value, bool):
        raise invalid(record, field, "must be a boolean")
    return value


def freeze_mapping(record: str, field: str, value: object) -> Mapping[str, Any]:
    """Return a read-only shallow copy of a string-keyed mapping."""
    if not isinstance(value, Mapping):
        raise invalid(record, field, "must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise invalid(record, field, "must have string keys")
    return MappingProxyType(dict(value))


def check_keys(
    record: str,
    data: Mapping[str, Any],
    required: tuple[str, ...],
    allowed: frozenset[str],
) -> None:
    """Ensure ``data`` has every required key and no unknown keys."""
    for key in required:
        if key not in data or data[key] is None:
            raise SchemaError(
                f"{record} is missing required field {key!r}",
                code=MISSING_FIELD,
                context={"record": record, "field": key},
            )
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SchemaError(
            f"{record} has unknown fields: {', '.join(unknown)}",
            code=UNKNOWN_FIELD,
            context={"record": record, "fields": unknown},
        )
