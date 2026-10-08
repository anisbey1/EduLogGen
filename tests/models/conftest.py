"""Synthetic fixtures for domain model tests. No real learner data."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from eduloggen.models import LogRecord

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)

MakeEvent = Callable[..., LogRecord]


@pytest.fixture
def make_event() -> MakeEvent:
    """Factory for valid events; keyword arguments override defaults."""

    def factory(event_id: str = "e1", minutes: int = 0, **overrides: Any) -> LogRecord:
        values: dict[str, Any] = {
            "event_id": event_id,
            "learner_id": "L1",
            "timestamp": T0 + timedelta(minutes=minutes),
            "activity_id": "quiz-1",
            "event_type": "view",
        }
        values.update(overrides)
        return LogRecord(**values)

    return factory
