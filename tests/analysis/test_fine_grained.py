"""Tests for per-activity, temporal, and stratified analysis.

Uses the hand-checkable fixture from ``conftest.py`` (2026-03-01 is a Sunday):
learner A has sessions ``view attempt submit`` (09:00-09:10 UTC) and
``view attempt`` (10:20-10:22, after a 70 minute gap); learner B has
``view video_play`` (09:00-09:01, explicit session ``b-x``).
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from eduloggen.analysis import (
    activity_profiles,
    activity_table_markdown,
    analyze_by,
    deadline_effects,
    session_groups,
    sessionize,
    split_by,
    temporal_profile,
)
from eduloggen.core import AnalysisError
from eduloggen.datasets import demo_dataset
from eduloggen.models import Dataset


@pytest.fixture
def toy(raw: Dataset) -> Dataset:
    return sessionize(raw)


# --------------------------------------------------------------------------
# Per-activity detail
# --------------------------------------------------------------------------


def test_activity_profiles(toy: Dataset) -> None:
    profiles = {p.token: p for p in activity_profiles(toy)}
    assert [p.token for p in activity_profiles(toy)][:2] == ["view", "attempt"]
    view = profiles["view"]
    assert view.count == 3
    assert view.share == pytest.approx(3 / 7)
    assert view.session_share == 1.0
    assert view.start_share == 1.0
    assert view.end_share == 0.0
    assert dict(view.successors) == pytest.approx(
        {"attempt": 2 / 3, "video_play": 1 / 3}
    )
    assert view.dwell_s.median == 120.0
    assert view.success_rate == 1.0  # the first view event is flagged successful
    assert dict(view.companions) == pytest.approx(
        {"quiz": 1 / 3, "video": 1 / 3, "intro": 1 / 3}
    )

    attempt = profiles["attempt"]
    assert dict(attempt.predecessors) == {"view": 1.0}
    assert dict(attempt.successors) == pytest.approx({"submit": 0.5, "(end)": 0.5})
    assert attempt.success_rate == 0.0
    assert attempt.learner_share == 0.5

    submit = profiles["submit"]
    assert submit.success_rate == 1.0
    assert submit.end_share == pytest.approx(1 / 3)
    assert submit.dwell_s.count == 0
    assert profiles["video_play"].mean_score is None


def test_activity_repeat_share() -> None:
    from ..generators.conftest import build

    data = build([("a", "a", "a", "b")])
    a = next(p for p in activity_profiles(data) if p.token == "a")
    assert a.repeat_share == pytest.approx(2 / 3)


def test_activity_table_and_serialization(toy: Dataset) -> None:
    profiles = activity_profiles(toy)
    table = activity_table_markdown(profiles)
    assert table.splitlines()[0].startswith("| Token | Share |")
    row = (
        "| view | 42.9% | 100% | 100% | 0% | 0% | 120 | 100% "
        "| attempt 67%, video_play 33% |"
    )
    assert row in table
    assert json.loads(json.dumps([p.to_dict() for p in profiles]))[0]["token"] == "view"
    assert activity_profiles(sessionize(Dataset(dataset_id="e", events=()))) == ()


# --------------------------------------------------------------------------
# Temporal
# --------------------------------------------------------------------------


def test_temporal_profile_utc(toy: Dataset) -> None:
    t = temporal_profile(toy)
    assert t.sessions_by_hour[9] == 2
    assert t.sessions_by_hour[10] == 1
    assert t.peak_hour == 9
    assert t.sessions_by_weekday[6] == 3
    assert t.heatmap[6][9] == 2
    assert sum(t.events_by_hour) == 7
    assert t.weekly == ({"week": 1, "sessions": 3, "events": 7, "learners": 2},)
    assert t.daily_sessions == {"2026-03-01": 3}
    assert t.between_sessions_s.median == pytest.approx(70 * 60)
    assert (t.first_day, t.last_day) == ("2026-03-01", "2026-03-01")


def test_temporal_profile_timezone(toy: Dataset) -> None:
    tokyo = temporal_profile(toy, timezone="Asia/Tokyo")
    assert tokyo.sessions_by_hour[18] == 2
    assert tokyo.peak_hour == 18
    with pytest.raises(AnalysisError):
        temporal_profile(toy, timezone="Mars/Base")


def test_temporal_markdown_and_empty(toy: Dataset) -> None:
    text = temporal_profile(toy).to_markdown()
    assert "## Activity over time (UTC)" in text
    assert "| 1 | 3 | 7 | 2 |" in text
    empty = temporal_profile(sessionize(Dataset(dataset_id="e", events=())))
    assert empty.peak_hour is None
    assert empty.weekly == ()
    assert json.loads(json.dumps(empty.to_dict()))["first_day"] is None


def test_deadline_effects() -> None:
    data = sessionize(demo_dataset(60, seed=1))
    effects = deadline_effects(data, ["2026-02-10", date(2026, 2, 3)], days_before=2)
    assert [e.deadline for e in effects] == ["2026-02-10", "2026-02-03"]
    assert all(e.ratio is None or e.ratio > 0 for e in effects)
    assert effects[0].to_dict()["days_before"] == 2
    with pytest.raises(AnalysisError):
        deadline_effects(data, ["next friday"])
    with pytest.raises(AnalysisError):
        deadline_effects(data, ["2026-02-10"], days_before=-1)
    assert (
        deadline_effects(sessionize(Dataset(dataset_id="e", events=())), ["2026-01-01"])
        == ()
    )


# --------------------------------------------------------------------------
# Stratified
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("by", "expected"),
    [
        ("course", {"c1": 3}),
        ("week", {"week 01": 3}),
        ("weekday", {"7-Sun": 3}),
        ("hour", {"09": 2, "10": 1}),
    ],
)
def test_session_groups(toy: Dataset, by: str, expected: dict[str, int]) -> None:
    groups = session_groups(toy, by)
    assert {k: len(v) for k, v in groups.items()} == expected


def test_learner_and_metadata_groups(toy: Dataset) -> None:
    groups = session_groups(toy, "learner_group", groups={"A": "cohort-1"})
    assert {k: len(v) for k, v in groups.items()} == {"(unassigned)": 1, "cohort-1": 2}
    by_device = session_groups(toy, "metadata:device")
    assert by_device == {"(none)": ["A:s1", "b-x", "A:s2"]}  # sessions in time order


def test_metadata_groups_on_demo() -> None:
    data = sessionize(demo_dataset(30, seed=2))
    groups = session_groups(data, "metadata:device")
    assert set(groups) <= {"desktop", "mobile"}
    assert sum(len(v) for v in groups.values()) == data.n_sessions


@pytest.mark.parametrize("by", ["country", "metadata:", "learner_group"])
def test_group_errors(toy: Dataset, by: str) -> None:
    with pytest.raises(AnalysisError):
        session_groups(toy, by)


def test_split_by(toy: Dataset) -> None:
    parts = split_by(toy, "hour")
    assert set(parts) == {"09", "10"}
    assert parts["09"].n_sessions == 2
    assert parts["09"].n_events == 5
    assert parts["10"].dataset_id == "toy[hour=10]"
    assert sum(p.n_events for p in parts.values()) == toy.n_events


def test_analyze_by(toy: Dataset) -> None:
    result = analyze_by(toy, "hour", min_sessions=2)
    assert result.overall.n_sessions == 3
    assert [g.group for g in result.groups] == ["09"]
    assert dict(result.omitted) == {"10": 1}
    nine = result.group("09")
    assert nine.n_learners == 2
    assert nine.mix_tvd > 0
    assert "view -> attempt" in nine.top_bigrams
    with pytest.raises(KeyError):
        result.group("23")
    text = result.to_markdown()
    assert text.startswith("## Analysis by hour")
    assert "| all | 3 | 2 | 7 |" in text
    assert "Omitted (too few sessions): 10 (1)" in text
    assert json.loads(json.dumps(result.to_dict()))["by"] == "hour"
    with pytest.raises(AnalysisError):
        analyze_by(toy, "hour", min_sessions=0)


def test_analyze_by_week_on_demo() -> None:
    data = sessionize(demo_dataset(60, seed=1))
    result = analyze_by(data, "week", timezone="Europe/Paris")
    assert result.groups[0].group == "week 01"
    assert sum(g.n_sessions for g in result.groups) == data.n_sessions
