"""A seeded, fully synthetic demo corpus (ADR-025, FR-B.4).

The simulator models an introductory course with four modules. In each
session a learner opens the course home page, works through the current
module's reading, video, and quiz (with repeated attempts until success or
giving up), sometimes visits the forum, and may move on to the next module.
Three learner profiles differ in diligence, quiz skill, and engagement:

- ``steady``: watches videos fully, usually passes within two attempts;
- ``skimmer``: short reads, often skips videos, fewer sessions;
- ``struggler``: many quiz attempts, more forum visits.

Time on each activity is lognormal with an activity-specific median, so
semi-Markov models have real timing structure to learn. No real data is
involved; identifiers are obviously synthetic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

from eduloggen.core import ConfigError
from eduloggen.models import Dataset, DatasetMetadata, LogRecord
from eduloggen.utils import make_rng

__all__ = ["DEMO_COURSE", "demo_dataset"]

DEMO_COURSE: Final = "intro-stats"
_START: Final = datetime(2026, 2, 2, tzinfo=UTC)
_MODULES: Final = 4

# Median seconds spent on each activity kind, and lognormal spread.
_MEDIAN_S: Final = {
    "home": 15.0,
    "reading": 240.0,
    "video": 480.0,
    "attempt": 90.0,
    "submit": 20.0,
    "forum_read": 120.0,
    "forum_post": 300.0,
}
_SIGMA: Final = 0.5


@dataclass(frozen=True, slots=True)
class _Profile:
    name: str
    weight: float
    sessions: tuple[int, int]
    watch_video: float
    pass_rate: float
    max_attempts: int
    forum: float
    advance: float
    pace: float


_PROFILES: Final = (
    _Profile("steady", 0.5, (4, 8), 0.9, 0.7, 4, 0.2, 0.35, 1.0),
    _Profile("skimmer", 0.3, (2, 5), 0.3, 0.5, 2, 0.05, 0.5, 0.5),
    _Profile("struggler", 0.2, (5, 10), 0.8, 0.3, 6, 0.5, 0.2, 1.4),
)


class _Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.events: list[LogRecord] = []

    def emit(
        self,
        learner: str,
        clock: datetime,
        kind: str,
        activity: str,
        event_type: str,
        device: str,
        pace: float,
        *,
        score: float | None = None,
        success: bool | None = None,
    ) -> datetime:
        seconds = self.rng.lognormvariate(math.log(_MEDIAN_S[kind] * pace), _SIGMA)
        self.events.append(
            LogRecord(
                event_id=f"ev{len(self.events) + 1:06d}",
                learner_id=learner,
                timestamp=clock,
                activity_id=activity,
                event_type=event_type,
                course_id=DEMO_COURSE,
                score=score,
                success=success,
                duration_ms=round(seconds * 1000),
                metadata={"device": device},
            )
        )
        return clock + timedelta(seconds=seconds)


def demo_dataset(n_learners: int = 60, *, seed: int = 0) -> Dataset:
    """Simulate the demo course.

    Args:
        n_learners: Number of synthetic learners.
        seed: Seed; the same arguments always give the same dataset.

    Returns:
        An unsessionized dataset (sessions are separated by at least a day,
        so the default 30 minute idle timeout recovers them).

    Raises:
        ConfigError: If ``n_learners`` is not positive or ``seed`` is negative.
    """
    if (
        isinstance(n_learners, bool)
        or not isinstance(n_learners, int)
        or n_learners < 1
    ):
        raise ConfigError(
            "n_learners must be a positive integer", code="config_invalid_value"
        )
    rng = make_rng(seed, "demo", n_learners)
    builder = _Builder(rng)
    weights = [profile.weight for profile in _PROFILES]
    for index in range(1, n_learners + 1):
        profile = rng.choices(_PROFILES, weights)[0]
        learner = f"learner-{index:03d}"
        device = rng.choice(["desktop", "desktop", "mobile"])
        day = _START + timedelta(days=rng.randint(0, 6))
        module = 1
        for _ in range(rng.randint(*profile.sessions)):
            if module > _MODULES:
                break
            clock = day + timedelta(hours=rng.uniform(8, 21))
            module = _session(builder, profile, learner, device, clock, module)
            day += timedelta(days=rng.randint(1, 4))
    return Dataset(
        dataset_id="demo",
        events=tuple(builder.events),
        metadata=DatasetMetadata(
            source_description=(
                f"EduLogGen demo course simulator ({n_learners} learners, seed {seed})"
            ),
            privacy_notes="Fully synthetic; contains no real learner data.",
            seeds={"demo": seed},
        ),
    )


def _session(
    builder: _Builder,
    profile: _Profile,
    learner: str,
    device: str,
    clock: datetime,
    module: int,
) -> int:
    rng = builder.rng
    pace = profile.pace

    def emit(
        kind: str,
        activity: str,
        event_type: str,
        score: float | None = None,
        success: bool | None = None,
    ) -> None:
        nonlocal clock
        clock = builder.emit(
            learner,
            clock,
            kind,
            activity,
            event_type,
            device,
            pace,
            score=score,
            success=success,
        )

    emit("home", "course-home", "navigate")
    while module <= _MODULES:
        prefix = f"m{module}"
        emit("reading", f"{prefix}-reading", "view")
        if rng.random() < profile.watch_video:
            emit("video", f"{prefix}-video", "video_play")
        passed = False
        for attempt in range(1, profile.max_attempts + 1):
            passed = rng.random() < profile.pass_rate + 0.1 * (attempt - 1)
            emit("attempt", f"{prefix}-quiz", "attempt", success=passed)
            if passed:
                break
        score = round(rng.uniform(0.7, 1.0) if passed else rng.uniform(0.2, 0.6), 2)
        emit("submit", f"{prefix}-quiz", "submit", score=score, success=passed)
        if rng.random() < profile.forum or not passed:
            emit("forum_read", "forum", "view")
            if rng.random() < 0.4:
                emit("forum_post", "forum", "forum_post")
        if passed:
            module += 1
        if not passed or rng.random() > profile.advance:
            break
    return module
