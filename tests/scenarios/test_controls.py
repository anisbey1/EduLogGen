"""Tests for controls, the session calendar, the manipulation check, and experiments."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import pytest

from eduloggen.analysis import interevent_times, session_sequences, sessionize
from eduloggen.cli import main
from eduloggen.core import ConfigError, GenerationError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import (
    IndependentGenerator,
    MarkovGenerator,
    SessionCalendar,
    get_generator,
)
from eduloggen.models import Dataset, GeneratorModel
from eduloggen.scenarios import (
    Controls,
    ExperimentSettings,
    apply_controls,
    manipulation_check,
    run_experiment,
)
from eduloggen.utils import make_rng

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/configs/experiment.yaml"


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(60, seed=1))


@pytest.fixture(scope="module")
def model(real: Dataset) -> GeneratorModel:
    return get_generator("semi_markov").fit(real)


def _tokens(dataset: Dataset) -> Counter[str]:
    return Counter(t for s in session_sequences(dataset) for t in s)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------


def test_controls_parse_and_serialize() -> None:
    controls = Controls.from_dict(
        {
            "event_weights": {"view": 2},
            "session_length": {"fixed": 5},
            "dwell_scale": None,
        }
    )
    assert controls.to_dict() == {
        "event_weights": {"view": 2.0},
        "session_length": {"fixed": 5},
    }
    assert not controls.is_empty
    assert Controls.from_dict(None).is_empty
    assert Controls.from_dict(controls.to_dict()) == controls


@pytest.mark.parametrize(
    "data",
    [
        {"event_weight": {"view": 2}},
        {"event_weights": {"view": -1}},
        {"event_weights": {"view": "high"}},
        {"event_weights": ["view"]},
        {"dwell_scale": {"view": 0}},
        {"session_length": {"scale": 0}},
        {"session_length": {"fixed": 1.5}},
        {"session_length": {"double": 2}},
        {"session_length": {"scale": 2, "fixed": 3}},
        {"sessions_per_learner": {"fixed": 0}},
        {"sessions_per_learner": {"mean": True}},
        "view",
    ],
)
def test_controls_validation(data: Any) -> None:
    with pytest.raises(ConfigError):
        Controls.from_dict(data)


def test_apply_controls_returns_new_model(model: GeneratorModel) -> None:
    controlled = apply_controls(model, Controls(event_weights={"view": 2.0}))
    assert controlled.fingerprint() != model.fingerprint()
    assert controlled.parameters["token_weights"] == {"view": 2.0}
    assert controlled.parameters["controls"] == [{"event_weights": {"view": 2.0}}]
    assert "token_weights" not in model.parameters
    assert apply_controls(model, Controls()) is model
    twice = apply_controls(controlled, Controls(event_weights={"view": 1.5}))
    assert twice.parameters["token_weights"]["view"] == pytest.approx(3.0)
    assert len(twice.parameters["controls"]) == 2


def test_apply_controls_errors(model: GeneratorModel) -> None:
    with pytest.raises(ConfigError) as info:
        apply_controls(model, Controls(event_weights={"teleport": 2.0}))
    assert info.value.context["tokens"] == ["teleport"]
    with pytest.raises(ConfigError):
        apply_controls(model, Controls(dwell_scale={"teleport": 2.0}))
    with pytest.raises(ConfigError):
        apply_controls(
            model, Controls(event_weights=dict.fromkeys(model.vocabulary, 0.0))
        )
    assert apply_controls(model, Controls(dwell_scale={"*": 2.0})).parameters[
        "timing_scale"
    ] == {"*": 2.0}


@pytest.mark.parametrize(
    ("length", "spec", "expected"),
    [
        (
            {"model": "empirical", "counts": {"2": 3, "4": 1}},
            {"scale": 2},
            {"model": "empirical", "counts": {"4": 3.0, "8": 1.0}},
        ),
        (
            {"model": "empirical", "counts": {"1": 1, "2": 1}},
            {"scale": 0.1},
            {"model": "empirical", "counts": {"1": 2.0}},
        ),
        (
            {"model": "poisson", "mean": 4.0},
            {"scale": 1.5},
            {"model": "poisson", "mean": 6.0},
        ),
        (
            {"model": "fixed", "length": 4},
            {"scale": 1.5},
            {"model": "fixed", "length": 6},
        ),
        (
            {"model": "empirical", "counts": {"3": 1}},
            {"fixed": 7},
            {"model": "fixed", "length": 7},
        ),
    ],
)
def test_session_length_transforms(
    model: GeneratorModel,
    length: dict[str, Any],
    spec: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    data = model.to_dict()
    data["parameters"] = {**data["parameters"], "length": length}
    custom = GeneratorModel.from_dict(data)
    assert (
        apply_controls(custom, Controls(session_length=spec)).parameters["length"]
        == expected
    )


def test_sessions_per_learner_transforms(model: GeneratorModel) -> None:
    fixed = apply_controls(model, Controls(sessions_per_learner={"fixed": 3}))
    assert fixed.parameters["population"]["sessions_per_learner"] == {"3": 1}
    mean = apply_controls(model, Controls(sessions_per_learner={"mean": 6}))
    counts = {
        int(k): v
        for k, v in mean.parameters["population"]["sessions_per_learner"].items()
    }
    assert sum(k * v for k, v in counts.items()) / sum(
        counts.values()
    ) == pytest.approx(6, rel=0.25)


@pytest.mark.parametrize("generator", ["markov", "semi_markov", "independent"])
def test_zero_weight_removes_token(real: Dataset, generator: str) -> None:
    gen = get_generator(generator)
    controlled = apply_controls(
        gen.fit(real), Controls(event_weights={"video_play": 0})
    )
    synthetic = gen.generate(controlled, 200, seed=0)
    assert "video_play" not in _tokens(synthetic)


def test_weights_need_generator_support(real: Dataset) -> None:
    class Plain(MarkovGenerator):
        name: ClassVar[str] = "markov"
        tags: ClassVar[frozenset[str]] = frozenset({"probabilistic"})

    controlled = apply_controls(
        MarkovGenerator().fit(real), Controls(event_weights={"view": 2})
    )
    with pytest.raises(GenerationError) as info:
        Plain().generate(controlled, 5, seed=0)
    assert info.value.code == "generation_unsupported_control"


def test_dwell_scale_is_exact_for_markov(real: Dataset) -> None:
    generator = MarkovGenerator()
    base_model = generator.fit(real)
    base = set(interevent_times(generator.generate(base_model, 50, seed=0)))
    scaled = set(
        interevent_times(
            generator.generate(
                apply_controls(base_model, Controls(dwell_scale={"*": 3.0})), 50, seed=0
            )
        )
    )
    assert len(base) == len(scaled) == 1
    assert scaled.pop() == pytest.approx(base.pop() * 3, rel=1e-6)


def test_independent_weights(real: Dataset) -> None:
    generator = IndependentGenerator()
    fitted = generator.fit(real)
    before = _tokens(generator.generate(fitted, 300, seed=1))
    after = _tokens(
        generator.generate(
            apply_controls(fitted, Controls(event_weights={"forum_post": 5})),
            300,
            seed=1,
        )
    )
    assert after["forum_post"] / sum(after.values()) > before["forum_post"] / sum(
        before.values()
    )


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

HOURS = [0.0] * 8 + [1.0] * 12 + [0.0] * 4


def _calendar(**overrides: Any) -> SessionCalendar:
    data: dict[str, Any] = {
        "start": "2026-09-07",
        "weeks": 4,
        "timezone": "Europe/Paris",
        "hours": HOURS,
        "weekdays": [1, 1, 1, 1, 1, 0, 0],
        "deadlines": [{"date": "2026-09-18", "surge": 4, "days_before": 1}],
    }
    data.update(overrides)
    return SessionCalendar.from_dict(data)


def test_calendar_basics() -> None:
    calendar = _calendar()
    assert calendar.days == 28
    assert calendar.end == date(2026, 10, 4)
    assert calendar.day_weight(date(2026, 9, 18)) == 4.0
    assert calendar.day_weight(date(2026, 9, 17)) == 4.0
    assert calendar.day_weight(date(2026, 9, 16)) == 1.0
    assert calendar.day_weight(date(2026, 9, 19)) == 0.0
    assert calendar.day_weight(date(2026, 12, 1)) == 0.0
    assert SessionCalendar.from_dict(calendar.to_dict()) == calendar
    rng = make_rng(0, "test")
    for _ in range(300):
        local = calendar.local_time(calendar.sample(rng))
        assert 8 <= local.hour < 20
        assert local.weekday() < 5
        assert calendar.start <= local.date() <= calendar.end


def test_calendar_days_and_defaults() -> None:
    calendar = SessionCalendar.from_dict({"start": date(2026, 1, 1), "days": 3})
    assert calendar.timezone == "UTC"
    assert calendar.hours == (1.0,) * 24
    assert SessionCalendar.from_dict(
        {"start": datetime(2026, 1, 1, tzinfo=UTC), "days": 1}
    ).start == date(2026, 1, 1)


@pytest.mark.parametrize(
    "overrides",
    [
        {"weeks": 0},
        {"weeks": "4"},
        {"days": 3},
        {"start": "September"},
        {"timezone": "Mars/Base"},
        {"hours": [1] * 23},
        {"hours": [-1] + [1] * 23},
        {"hours": [0] * 24},
        {"weekdays": [0] * 7},
        {"deadlines": [{"date": "2026-09-18"}]},
        {"deadlines": [{"date": "2026-09-18", "surge": 2, "days_before": -1}]},
        {"deadlines": [{"date": "2026-09-18", "surge": 2, "when": 1}]},
        {"term": "autumn"},
    ],
)
def test_calendar_validation(overrides: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        _calendar(**overrides)
    with pytest.raises(ConfigError):
        SessionCalendar.from_dict("monday")  # type: ignore[arg-type]


def test_generation_with_calendar(model: GeneratorModel) -> None:
    calendar = _calendar()
    synthetic = get_generator("semi_markov").generate(
        model, 300, seed=2, calendar=calendar
    )
    by_learner: defaultdict[str, list[Any]] = defaultdict(list)
    for session in synthetic.sessions or ():
        local = calendar.local_time(session.start_time)
        assert 8 <= local.hour < 20
        assert local.weekday() < 5
        by_learner[session.learner_id].append((session.start_time, session.end_time))
    for spans in by_learner.values():
        spans.sort()
        for (_, end), (start, _) in pairwise(spans):
            assert end < start


# --------------------------------------------------------------------------
# Manipulation check
# --------------------------------------------------------------------------


def test_check_passes_for_real_effects(model: GeneratorModel) -> None:
    generator = get_generator("semi_markov")
    controls = Controls(
        event_weights={"forum_post": 3.0, "video_play": 0},
        dwell_scale={"attempt": 2.0},
        session_length={"scale": 1.3},
        sessions_per_learner={"mean": 5},
    )
    calendar = _calendar()
    base = generator.generate(model, 600, seed=3, calendar=calendar)
    ctl = generator.generate(
        apply_controls(model, controls), 600, seed=3, calendar=calendar
    )
    check = manipulation_check(base, ctl, controls=controls, calendar=calendar)
    statuses = {(i.control, i.target): i.status for i in check.items}
    assert check.passed, check.to_markdown()
    assert statuses[("event_weights", "video_play")] == "ok"
    assert statuses[("calendar.hours", "zero-weight hours")] == "ok"
    assert ("calendar.deadlines", "2026-09-18") in statuses
    assert json.loads(json.dumps(check.to_dict()))["passed"] is True
    assert "| event_weights | forum_post |" in check.to_markdown()


def test_check_detects_missing_effects(model: GeneratorModel) -> None:
    generator = get_generator("semi_markov")
    base = generator.generate(model, 300, seed=4)
    claimed = Controls(
        event_weights={"forum_post": 3.0, "video_play": 0},
        dwell_scale={"attempt": 3.0},
        session_length={"fixed": 4},
        sessions_per_learner={"fixed": 9},
    )
    check = manipulation_check(base, base, controls=claimed, calendar=_calendar())
    assert not check.passed
    failed = {(i.control, i.target) for i in check.items if i.status == "failed"}
    assert {
        ("event_weights", "forum_post"),
        ("event_weights", "video_play"),
        ("dwell_scale", "attempt"),
        ("session_length", "fixed"),
        ("sessions_per_learner", "fixed"),
        ("calendar", "period"),
    } <= failed
    assert "FAILED" in check.to_markdown()


def test_check_not_measurable_and_anomalies(model: GeneratorModel) -> None:
    base = get_generator("semi_markov").generate(model, 50, seed=5)
    data = model.to_dict()
    check = manipulation_check(
        base,
        base,
        controls=Controls(event_weights={"view": 1.0}, dwell_scale={"never_seen": 2.0}),
        anomaly_report={
            "anomalies": {"inactivity": {"rate": 0.1, "requested": 5, "injected": 4}}
        },
    )
    statuses = {(i.control, i.target): i.status for i in check.items}
    assert statuses == {
        ("dwell_scale", "never_seen"): "not_measurable",
        ("anomalies", "inactivity"): "failed",
    }
    assert data


# --------------------------------------------------------------------------
# Experiments and CLI
# --------------------------------------------------------------------------


def test_experiment_settings() -> None:
    settings = ExperimentSettings.from_file(EXAMPLE)
    assert not settings.controls.is_empty
    assert settings.calendar is not None and settings.calendar.days == 84
    assert [a.type for a in settings.anomalies] == [
        "inactivity",
        "unexpected_transition",
    ]
    bad: list[dict[str, Any]] = [{}, {"profiles": {}}, {"anomalies": "inactivity"}]
    for data in bad:
        with pytest.raises(ConfigError):
            ExperimentSettings.from_dict(data)


def test_run_experiment(model: GeneratorModel) -> None:
    settings = ExperimentSettings.from_file(EXAMPLE)
    result = run_experiment(
        get_generator("semi_markov"), model, settings, n_sessions=400, seed=6
    )
    assert result.annotations is not None
    assert result.model.fingerprint() != model.fingerprint()
    assert result.check.passed, result.check.to_markdown()
    anomaly_items = [i for i in result.check.items if i.control == "anomalies"]
    assert len(anomaly_items) == 2
    calendar_only = run_experiment(
        get_generator("semi_markov"),
        model,
        ExperimentSettings.from_dict({"calendar": settings.calendar.to_dict()}),  # type: ignore[union-attr]
        n_sessions=50,
        seed=1,
    )
    assert calendar_only.annotations is None
    assert calendar_only.model is model


def test_cli_generate_experiment(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for command in (
        "demo --output {d}/demo --learners 40",
        "fit --input {d}/demo --generator semi_markov --output {d}/m",
    ):
        assert main([*command.format(d=tmp_path).split(), "--quiet"]) == 0
    command = (
        f"generate --model {tmp_path}/m --n-sessions 300 --seed 2 "
        f"--experiment {EXAMPLE} --output {tmp_path}/out"
    )
    assert main([*command.split(), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "# Manipulation check: PASSED" in out
    for name in (
        "manipulation_check.json",
        "manipulation_check.md",
        "annotations.csv",
        "run_manifest.json",
    ):
        assert (tmp_path / "out" / name).exists()
    manifest = json.loads((tmp_path / "out" / "run_manifest.json").read_text())
    assert manifest["inputs"]["controlled_model"].startswith("sha256:")

    both = (
        f"generate --model {tmp_path}/m --seed 2 --experiment {EXAMPLE} "
        f"--anomalies {EXAMPLE} --output {tmp_path}/x"
    )
    assert main([*both.split(), "--quiet"]) == 2
    no_seed = (
        f"generate --model {tmp_path}/m --experiment {EXAMPLE} --output {tmp_path}/y"
    )
    assert main([*no_seed.split(), "--quiet"]) == 2
    assert "generation_missing_seed" in capsys.readouterr().err


def test_next_allowed_moves_minimally() -> None:
    calendar = _calendar()  # open 08-20 Paris time, Monday-Friday
    paris = calendar.local_time
    open_moment = datetime(2026, 9, 8, 10, 15, tzinfo=UTC)  # Tue 12:15 Paris
    assert calendar.next_allowed(open_moment) == open_moment
    evening = datetime(2026, 9, 8, 19, 30, tzinfo=UTC)  # Tue 21:30 Paris, closed
    moved = paris(calendar.next_allowed(evening))
    assert (moved.date(), moved.hour, moved.minute) == (date(2026, 9, 9), 8, 0)
    friday_night = datetime(2026, 9, 11, 19, 0, tzinfo=UTC)  # Fri 21:00, weekend next
    moved = paris(calendar.next_allowed(friday_night))
    assert (moved.date().weekday(), moved.hour) == (0, 8)
    after_period = datetime(2026, 12, 1, tzinfo=UTC)
    assert calendar.next_allowed(after_period) == after_period


def test_calendar_rules_hold_for_busy_learners(model: GeneratorModel) -> None:
    """Many sessions per learner force moves; closed hours must stay closed."""
    calendar = _calendar(weeks=1)
    busy = apply_controls(model, Controls(sessions_per_learner={"fixed": 12}))
    synthetic = get_generator("semi_markov").generate(
        busy, 240, seed=8, calendar=calendar
    )
    for session in synthetic.sessions or ():
        local = calendar.local_time(session.start_time)
        assert 8 <= local.hour < 20 and local.weekday() < 5


def test_example_experiment_tokens_exist_in_demo(model: GeneratorModel) -> None:
    settings = ExperimentSettings.from_file(EXAMPLE)
    apply_controls(model, settings.controls)
