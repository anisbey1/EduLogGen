"""Tests for :class:`eduloggen.core.RunContext`."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

import eduloggen
from eduloggen.core import ConfigError, RunContext, current_platform


def test_create_populates_fields() -> None:
    before = datetime.now(UTC)
    ctx = RunContext.create(seed=42)
    after = datetime.now(UTC)

    assert ctx.seed == 42
    assert len(ctx.run_id) == 32
    assert before <= ctx.started_at <= after
    assert ctx.eduloggen_version == eduloggen.__version__
    assert set(ctx.platform) == {"python", "implementation", "os", "machine"}


def test_create_without_seed_and_unique_ids() -> None:
    first, second = RunContext.create(), RunContext.create()
    assert first.seed is None
    assert first.run_id != second.run_id


def test_is_immutable() -> None:
    ctx = RunContext.create(seed=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.seed = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        ctx.platform["os"] = "x"  # type: ignore[index]


def test_round_trip_through_json() -> None:
    ctx = RunContext.create(seed=7)
    restored = RunContext.from_dict(json.loads(json.dumps(ctx.to_dict())))
    assert restored == ctx


@pytest.mark.parametrize("seed", [-1, 1.5, True, "3"])
def test_rejects_invalid_seed(seed: Any) -> None:
    with pytest.raises(ConfigError) as info:
        RunContext(run_id="r", seed=seed, started_at=datetime.now(UTC))
    assert "seed" in info.value.context


def test_rejects_empty_run_id() -> None:
    with pytest.raises(ConfigError):
        RunContext(run_id="", seed=None, started_at=datetime.now(UTC))


@pytest.mark.parametrize(
    "started_at",
    [
        datetime(2026, 1, 1),
        datetime(2026, 1, 1, tzinfo=timezone(timedelta(hours=2))),
    ],
)
def test_rejects_non_utc_start(started_at: datetime) -> None:
    with pytest.raises(ConfigError):
        RunContext(run_id="r", seed=None, started_at=started_at)


def test_accepts_utc_equivalent_timezone() -> None:
    started = datetime(2026, 1, 1, tzinfo=timezone(timedelta(0)))
    assert RunContext(run_id="r", seed=0, started_at=started).seed == 0


def test_from_dict_missing_key() -> None:
    data = RunContext.create().to_dict()
    del data["started_at"]
    with pytest.raises(ConfigError) as info:
        RunContext.from_dict(data)
    assert info.value.context == {"key": "started_at"}


def test_from_dict_bad_timestamp() -> None:
    data = RunContext.create().to_dict()
    data["started_at"] = "not-a-date"
    with pytest.raises(ConfigError):
        RunContext.from_dict(data)


def test_current_platform_is_read_only() -> None:
    info = current_platform()
    assert info["python"]
    with pytest.raises(TypeError):
        info["python"] = "x"  # type: ignore[index]
