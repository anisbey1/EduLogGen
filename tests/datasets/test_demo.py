"""Tests for the synthetic demo corpus."""

from __future__ import annotations

import pytest

from eduloggen.analysis import sessionize
from eduloggen.core import ConfigError
from eduloggen.datasets import DEMO_COURSE, demo_dataset


def test_demo_is_deterministic_and_sessionizable() -> None:
    dataset = demo_dataset(20, seed=1)
    assert dataset.fingerprint() == demo_dataset(20, seed=1).fingerprint()
    assert dataset.fingerprint() != demo_dataset(20, seed=2).fingerprint()
    assert len(dataset.learner_ids) == 20
    assert {e.course_id for e in dataset.events} == {DEMO_COURSE}
    sessions = sessionize(dataset).sessions or ()
    assert all(s.event_sequence[0] == "navigate" for s in sessions)


@pytest.mark.parametrize("n", [0, -1, True])
def test_demo_rejects_bad_size(n: int) -> None:
    with pytest.raises(ConfigError):
        demo_dataset(n)
