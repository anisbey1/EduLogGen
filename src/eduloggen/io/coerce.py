"""Convert raw source values into canonical Python types.

Readers yield whatever the source format holds: strings for CSV, JSON types
for JSON Lines, typed values for Parquet. These functions turn any of those
into the types the canonical model expects. They raise :class:`CoercionError`
with a short reason code and never echo the offending value.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, tzinfo
from typing import Final

__all__ = [
    "CoercionError",
    "is_missing",
    "to_bool",
    "to_datetime",
    "to_float",
    "to_int",
    "to_str",
]

_TRUE: Final = frozenset({"true", "t", "yes", "y", "1"})
_FALSE: Final = frozenset({"false", "f", "no", "n", "0"})

ISO: Final = "iso"
EPOCH_S: Final = "epoch_s"
EPOCH_MS: Final = "epoch_ms"


class CoercionError(ValueError):
    """A value could not be converted.

    Attributes:
        reason: Short machine-readable reason (e.g. ``"not_a_number"``).
    """

    def __init__(self, reason: str) -> None:
        """Initialize with a reason code."""
        super().__init__(reason)
        self.reason = reason


def is_missing(value: object) -> bool:
    """Return whether a raw value means "no value".

    ``None``, empty or whitespace-only strings, and NaN floats are missing.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return isinstance(value, float) and math.isnan(value)


def to_str(value: object) -> str:
    """Convert to a non-empty string; integral floats lose their ``.0``."""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise CoercionError("empty_string")
        return text
    if isinstance(value, bool):
        raise CoercionError("not_a_string")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, int | float):
        return str(value)
    raise CoercionError("not_a_string")


def to_int(value: object) -> int:
    """Convert to an integer; integral floats and numeric strings accepted."""
    if isinstance(value, bool):
        raise CoercionError("not_an_integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            pass
    number = to_float(value)
    if not number.is_integer():
        raise CoercionError("not_an_integer")
    return int(number)


def to_float(value: object) -> float:
    """Convert to a finite float."""
    if isinstance(value, bool):
        raise CoercionError("not_a_number")
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            raise CoercionError("not_a_number") from None
    else:
        raise CoercionError("not_a_number")
    if not math.isfinite(number):
        raise CoercionError("not_finite")
    return number


def to_bool(value: object) -> bool:
    """Convert to a boolean from bools, 0/1, or common true/false words."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise CoercionError("not_a_boolean")


def to_datetime(
    value: object,
    *,
    fmt: str = ISO,
    assume_tz: tzinfo | None = None,
) -> tuple[datetime, bool]:
    """Convert to a timezone-aware UTC datetime.

    Args:
        value: A datetime, a string, or a number (for epoch formats).
        fmt: ``"iso"`` (ISO 8601), ``"epoch_s"``, ``"epoch_ms"``, or a
            :meth:`datetime.strptime` format string.
        assume_tz: Zone applied to naive values. ``None`` rejects them.

    Returns:
        The UTC datetime and whether ``assume_tz`` was applied.

    Raises:
        CoercionError: If parsing fails or a naive value has no assumed zone.
    """
    if fmt in (EPOCH_S, EPOCH_MS):
        seconds = to_float(value)
        if fmt == EPOCH_MS:
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, UTC), False
        except (OverflowError, OSError, ValueError):
            raise CoercionError("timestamp_out_of_range") from None

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            if fmt == ISO:
                parsed = datetime.fromisoformat(value.strip())
            else:
                parsed = datetime.strptime(value.strip(), fmt)
        except ValueError:
            raise CoercionError("bad_timestamp_format") from None
    else:
        raise CoercionError("not_a_timestamp")

    if parsed.utcoffset() is not None:
        return parsed.astimezone(UTC), False
    if assume_tz is None:
        raise CoercionError("naive_timestamp")
    return parsed.replace(tzinfo=assume_tz).astimezone(UTC), True
