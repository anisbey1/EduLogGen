"""Tests for raw value coercion."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from eduloggen.io import coerce
from eduloggen.io.coerce import CoercionError


@pytest.mark.parametrize("value", [None, "", "   ", math.nan])
def test_is_missing(value: object) -> None:
    assert coerce.is_missing(value)


@pytest.mark.parametrize("value", [0, "0", False, "x", 0.0])
def test_is_not_missing(value: object) -> None:
    assert not coerce.is_missing(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" L1 ", "L1"), (42, "42"), (42.0, "42"), (4.5, "4.5")],
)
def test_to_str(value: object, expected: str) -> None:
    assert coerce.to_str(value) == expected


@pytest.mark.parametrize("value", ["  ", True, [1], None])
def test_to_str_rejects(value: object) -> None:
    with pytest.raises(CoercionError):
        coerce.to_str(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3),
        ("12", 12),
        (" 7 ", 7),
        ("3.0", 3),
        (4.0, 4),
        ("12345678901234567890", 12345678901234567890),
    ],
)
def test_to_int(value: object, expected: int) -> None:
    assert coerce.to_int(value) == expected


@pytest.mark.parametrize("value", [True, "3.5", 2.5, "abc", None])
def test_to_int_rejects(value: object) -> None:
    with pytest.raises(CoercionError) as info:
        coerce.to_int(value)
    assert info.value.reason in {"not_an_integer", "not_a_number"}


@pytest.mark.parametrize(("value", "expected"), [(1, 1.0), ("0.5", 0.5), (2.5, 2.5)])
def test_to_float(value: object, expected: float) -> None:
    assert coerce.to_float(value) == expected


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (False, "not_a_number"),
        ("x", "not_a_number"),
        ("inf", "not_finite"),
        ([], "not_a_number"),
    ],
)
def test_to_float_rejects(value: object, reason: str) -> None:
    with pytest.raises(CoercionError) as info:
        coerce.to_float(value)
    assert info.value.reason == reason


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (0, False),
        (1.0, True),
        ("Yes", True),
        (" f ", False),
        ("0", False),
    ],
)
def test_to_bool(value: object, expected: bool) -> None:
    assert coerce.to_bool(value) is expected


@pytest.mark.parametrize("value", [2, "maybe", None])
def test_to_bool_rejects(value: object) -> None:
    with pytest.raises(CoercionError):
        coerce.to_bool(value)


T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "fmt"),
    [
        ("2026-03-01T09:00:00Z", "iso"),
        ("2026-03-01T11:00:00+02:00", "iso"),
        (T0, "iso"),
        (T0.timestamp(), "epoch_s"),
        (str(int(T0.timestamp())), "epoch_s"),
        (T0.timestamp() * 1000, "epoch_ms"),
        ("01/03/2026 09:00 +0000", "%d/%m/%Y %H:%M %z"),
    ],
)
def test_to_datetime(value: object, fmt: str) -> None:
    parsed, assumed = coerce.to_datetime(value, fmt=fmt)
    assert parsed == T0
    assert parsed.tzinfo is UTC
    assert not assumed


def test_to_datetime_assumes_zone_for_naive() -> None:
    parsed, assumed = coerce.to_datetime(
        "2026-03-01 10:00:00", assume_tz=ZoneInfo("Europe/Paris")
    )
    assert parsed == T0
    assert assumed


def test_to_datetime_keeps_explicit_offset_over_assumed_zone() -> None:
    value = datetime(2026, 3, 1, 9, tzinfo=timezone(timedelta(0)))
    parsed, assumed = coerce.to_datetime(value, assume_tz=ZoneInfo("Asia/Tokyo"))
    assert parsed == T0
    assert not assumed


@pytest.mark.parametrize(
    ("value", "fmt", "reason"),
    [
        ("2026-03-01 09:00:00", "iso", "naive_timestamp"),
        ("yesterday", "iso", "bad_timestamp_format"),
        ("2026-03-01", "%d/%m/%Y", "bad_timestamp_format"),
        (12, "iso", "not_a_timestamp"),
        ("abc", "epoch_s", "not_a_number"),
        (1e20, "epoch_s", "timestamp_out_of_range"),
    ],
)
def test_to_datetime_rejects(value: object, fmt: str, reason: str) -> None:
    with pytest.raises(CoercionError) as info:
        coerce.to_datetime(value, fmt=fmt)
    assert info.value.reason == reason
