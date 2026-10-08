"""Hand-checkable synthetic fixture for analysis tests.

Learner ``A`` (no source session ids)::

    t=0  view   quiz      success=True
    t=5  attempt quiz     success=False
    t=10 submit quiz      success=True
    --- 70 minute gap ---
    t=80 view   video
    t=82 attempt quiz

Learner ``B`` (explicit session ``b-x``)::

    t=0 view       intro
    t=1 video_play video
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eduloggen.models import Dataset, LogRecord

T0 = datetime(2026, 3, 1, 9, 0, tzinfo=UTC)


def event(
    event_id: str,
    learner: str,
    minute: int,
    event_type: str,
    activity: str,
    session: str | None = None,
    success: bool | None = None,
) -> LogRecord:
    return LogRecord(
        event_id=event_id,
        learner_id=learner,
        timestamp=T0 + timedelta(minutes=minute),
        activity_id=activity,
        event_type=event_type,
        session_id=session,
        success=success,
        course_id="c1",
    )


@pytest.fixture
def raw() -> Dataset:
    return Dataset(
        dataset_id="toy",
        events=(
            event("a3", "A", 10, "submit", "quiz", success=True),
            event("a1", "A", 0, "view", "quiz", success=True),
            event("a2", "A", 5, "attempt", "quiz", success=False),
            event("a4", "A", 80, "view", "video"),
            event("a5", "A", 82, "attempt", "quiz"),
            event("b1", "B", 0, "view", "intro", session="b-x"),
            event("b2", "B", 1, "video_play", "video", session="b-x"),
        ),
    )
