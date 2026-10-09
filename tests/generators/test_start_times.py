"""Models learn when sessions start (generator version 1.1)."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from eduloggen.analysis import sessionize, temporal_profile
from eduloggen.core import ConfigError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import SessionCalendar, get_generator
from eduloggen.models import Dataset, GeneratorModel


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(60, seed=4))


def _without_start_times(model: GeneratorModel) -> GeneratorModel:
    data = model.to_dict()
    population = dict(data["parameters"]["population"])
    population.pop("start_hours_utc")
    population.pop("start_days")
    data["parameters"] = {**data["parameters"], "population": population}
    return GeneratorModel.from_dict(data)


def test_fit_records_start_times(real: Dataset) -> None:
    model = get_generator("markov").fit(real)
    population = model.parameters["population"]
    assert model.generator_version == "1.1"
    assert sum(population["start_hours_utc"]) == real.n_sessions
    assert sum(population["start_days"]) == real.n_sessions
    profile = temporal_profile(real)
    assert population["start_hours_utc"] == list(profile.sessions_by_hour)


@pytest.mark.parametrize("name", ["markov", "semi_markov", "independent"])
def test_sessions_start_only_in_learned_hours_and_days(
    real: Dataset, name: str
) -> None:
    generator = get_generator(name)
    model = generator.fit(real)
    synthetic = generator.generate(model, 300, seed=3)
    hours = model.parameters["population"]["start_hours_utc"]
    days = model.parameters["population"]["start_days"]
    first = min(s.start_time for s in real.sessions or ()).date()
    for session in synthetic.sessions or ():
        assert hours[session.start_time.hour] > 0
        offset = (session.start_time.date() - first).days
        assert 0 <= offset < len(days) and days[offset] > 0
    real_profile, synth_profile = temporal_profile(real), temporal_profile(synthetic)
    tvd = 0.5 * sum(
        abs(a / real.n_sessions - b / synthetic.n_sessions)
        for a, b in zip(
            real_profile.sessions_by_hour, synth_profile.sessions_by_hour, strict=True
        )
    )
    assert tvd < 0.2


def test_models_without_start_times_keep_old_behaviour(real: Dataset) -> None:
    generator = get_generator("markov")
    old = _without_start_times(generator.fit(real))
    hours = generator.fit(real).parameters["population"]["start_hours_utc"]
    synthetic = generator.generate(old, 300, seed=3)
    assert any(hours[s.start_time.hour] == 0 for s in synthetic.sessions or ())
    again = generator.generate(old, 300, seed=3)
    assert synthetic.fingerprint() == again.fingerprint()


def test_start_time_moves_the_learned_pattern(real: Dataset) -> None:
    generator = get_generator("semi_markov")
    model = generator.fit(real)
    start = datetime(2030, 5, 6, 12, 0, tzinfo=UTC)
    synthetic = generator.generate(model, 200, seed=1, start_time=start)
    starts = [s.start_time for s in synthetic.sessions or ()]
    assert min(starts) >= start
    assert max(starts).date() < date(2030, 5, 6).replace(
        day=6 + len(model.parameters["population"]["start_days"])
    )


def test_explicit_calendar_takes_precedence(real: Dataset) -> None:
    generator = get_generator("markov")
    calendar = SessionCalendar.from_dict(
        {"start": "2031-01-05", "days": 3, "hours": [0] * 3 + [1] + [0] * 20}
    )
    synthetic = generator.generate(generator.fit(real), 50, seed=0, calendar=calendar)
    assert {s.start_time.hour for s in synthetic.sessions or ()} <= {3, 4}
    assert min(s.start_time for s in synthetic.sessions or ()).date() >= date(
        2031, 1, 5
    )


def test_calendar_daily_weights() -> None:
    calendar = SessionCalendar.from_dict({"start": "2026-01-01", "daily": [0, 2, 0]})
    assert calendar.days == 3
    assert calendar.day_weight(date(2026, 1, 2)) == 2.0
    assert calendar.day_weight(date(2026, 1, 1)) == 0.0
    assert SessionCalendar.from_dict(calendar.to_dict()) == calendar
    for bad in (
        {"daily": "1,2"},
        {"daily": [1, 2], "days": 3},
        {"daily": [0, 0]},
        {"daily": [-1, 1]},
    ):
        with pytest.raises(ConfigError):
            SessionCalendar.from_dict({"start": "2026-01-01", **bad})
