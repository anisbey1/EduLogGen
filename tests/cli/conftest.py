"""Fixtures: a small synthetic LMS export and its mapping."""

from __future__ import annotations

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

EMAIL = "student{}@uni.example"
MAPPING = """\
fields:
  event_id: log_id
  learner_id:
    source: user_id
    hash: {salt: test-salt}
  timestamp:
    source: time
    format: "%Y-%m-%d %H:%M:%S"
  activity_id: resource
  event_type: action
  score: grade
metadata: [device, ip]
drop: [user_email]
timezone: UTC
"""


@pytest.fixture
def export(tmp_path: Path) -> Path:
    """Write ``lms.csv`` and ``mapping.yaml``; return the directory."""
    rng = random.Random(1)
    following = {
        "view": ["view", "attempt", "video_play"],
        "attempt": ["attempt", "submit"],
        "submit": ["view"],
        "video_play": ["view", "attempt"],
    }
    with (tmp_path / "lms.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "log_id",
                "user_id",
                "time",
                "resource",
                "action",
                "grade",
                "device",
                "ip",
                "user_email",
            ]
        )
        row = 0
        for user in range(12):
            clock = datetime(2026, 3, 2, 8) + timedelta(hours=rng.random() * 24)
            for _ in range(rng.randint(1, 3)):
                action = "view"
                for _ in range(rng.randint(2, 6)):
                    row += 1
                    writer.writerow(
                        [
                            row,
                            EMAIL.format(user),
                            clock.strftime("%Y-%m-%d %H:%M:%S"),
                            f"res-{action}",
                            action,
                            "0.5" if action == "submit" else "",
                            rng.choice(["mobile", "desktop"]),
                            "10.0.0.1",
                            EMAIL.format(user),
                        ]
                    )
                    clock += timedelta(seconds=rng.expovariate(1 / 60))
                    action = rng.choice(following[action])
                clock += timedelta(hours=5)
    (tmp_path / "mapping.yaml").write_text(MAPPING)
    return tmp_path
