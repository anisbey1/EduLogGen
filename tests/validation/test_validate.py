"""Tests for metrics, validate(), reports, and the metric registry."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest

from eduloggen.analysis import sessionize
from eduloggen.core import ConfigError, PluginError, ValidationError
from eduloggen.generators import get_generator
from eduloggen.models import Dataset
from eduloggen.validation import (
    BUILTIN_METRICS,
    DEFAULT_METRICS,
    BaseMetric,
    MetricResult,
    ValidationContext,
    ValidationReport,
    available_metrics,
    get_metric,
    register_metric,
    unregister_metric,
    validate,
)

from ..generators.conftest import build


@pytest.fixture
def real() -> Dataset:
    return build(
        [
            ("view", "attempt", "submit"),
            ("view", "attempt", "attempt", "submit"),
            ("view", "video_play", "view", "attempt"),
            ("navigate", "view"),
            ("view",),
        ],
        gap_after={"video_play": 200.0},
    )


@pytest.fixture
def disjoint() -> Dataset:
    return build([("x", "y", "z"), ("z", "y", "x", "x")])


# --------------------------------------------------------------------------
# Metric values
# --------------------------------------------------------------------------


def test_identical_datasets(real: Dataset) -> None:
    report = validate(real, real)
    values = {m.name: m.value for m in report.metrics}
    for name in (
        "event_type_tvd",
        "activity_jsd",
        "session_length_ks",
        "session_duration_w1",
        "interevent_time_ks",
        "bigram_tvd",
        "transition_jsd",
    ):
        assert values[name] == pytest.approx(0.0, abs=1e-12), name
    assert values["topn_path_overlap"] == 1.0
    assert values["exact_session_dup_rate"] == 1.0
    assert values["rare_ngram_replay_rate"] == 1.0
    assert values["nn_distance_p05"] == 0.0
    assert report.metric("nn_distance_p05").details["share_exact"] == 1.0


def test_disjoint_datasets(real: Dataset, disjoint: Dataset) -> None:
    report = validate(real, disjoint)
    values = {m.name: m.value for m in report.metrics}
    assert values["event_type_tvd"] == 1.0
    assert values["activity_jsd"] == pytest.approx(1.0)
    assert values["bigram_tvd"] == 1.0
    assert values["transition_jsd"] == pytest.approx(1.0)
    assert values["topn_path_overlap"] == 0.0
    assert values["exact_session_dup_rate"] == 0.0
    assert values["rare_ngram_replay_rate"] == 0.0
    assert values["nn_distance_p05"] == pytest.approx(1.0)
    details = report.metric("transition_jsd").details
    assert details["states_missing_in_synthetic"] == details["real_states"]
    assert report.metric("event_type_tvd").details["unseen_in_synthetic"] == [
        "attempt",
        "navigate",
        "submit",
        "video_play",
        "view",
    ]


def test_skips_when_not_computable(real: Dataset) -> None:
    singles = build([("view",), ("attempt",)])
    report = validate(real, singles)
    statuses = {m.name: m.status for m in report.metrics}
    for name in ("interevent_time_ks", "bigram_tvd", "exact_session_dup_rate"):
        assert statuses[name] == "skip"
        assert report.metric(name).details["reason"]
    assert validate(singles, real).metric("rare_ngram_replay_rate").status == "skip"
    assert validate(singles, real).metric("transition_jsd").status == "skip"
    assert validate(singles, real).metric("topn_path_overlap").status == "skip"


def test_empty_datasets_skip() -> None:
    empty = sessionize(Dataset(dataset_id="e", events=()))
    report = validate(empty, empty)
    assert {m.status for m in report.metrics} == {"skip"}
    assert report.status == "pass"


def test_metric_params(real: Dataset) -> None:
    report = validate(
        real,
        real,
        metrics=[
            "topn_path_overlap",
            "exact_session_dup_rate",
            "rare_ngram_replay_rate",
        ],
        params={
            "topn_path_overlap": {"n": 2, "path_length": 2},
            "exact_session_dup_rate": {"min_length": 1},
            "rare_ngram_replay_rate": {"n": 2},
        },
    )
    assert report.metric("topn_path_overlap").details["compared"] == 2
    assert (
        report.metric("exact_session_dup_rate").details["synthetic_sessions_checked"]
        == 5
    )
    assert report.metric("rare_ngram_replay_rate").details["n"] == 2


def test_nn_distance_subsampling_is_seeded(real: Dataset, disjoint: Dataset) -> None:
    params = {"nn_distance_p05": {"max_real": 2, "max_synthetic": 1}}
    first = validate(real, disjoint, metrics=["nn_distance_p05"], params=params, seed=1)
    second = validate(
        real, disjoint, metrics=["nn_distance_p05"], params=params, seed=1
    )
    assert (
        first.metric("nn_distance_p05").value == second.metric("nn_distance_p05").value
    )
    details = first.metric("nn_distance_p05").details
    assert details["real_sessions_compared"] == 2
    assert details["synthetic_sessions_checked"] == 1


def test_generators_rank_sensibly() -> None:
    """Cyclic sessions: a Markov chain should match bigrams far better."""
    rng = random.Random(0)
    tokens = ["view", "attempt", "submit", "hint"]
    training = build(
        [[tokens[(i + j) % 4] for j in range(rng.randint(3, 8))] for i in range(80)],
        learners=20,
    )
    scores: dict[str, float] = {}
    for name in ("markov", "independent"):
        generator = get_generator(name)
        synthetic = generator.generate(generator.fit(training), 200, seed=0)
        value = validate(training, synthetic).metric("bigram_tvd").value
        assert value is not None
        scores[name] = value
    assert scores["markov"] < scores["independent"]


# --------------------------------------------------------------------------
# Thresholds and report
# --------------------------------------------------------------------------


def test_thresholds(real: Dataset, disjoint: Dataset) -> None:
    report = validate(
        real,
        disjoint,
        metrics=["event_type_tvd", "topn_path_overlap", "session_length_ks"],
        thresholds={"event_type_tvd": 0.2, "topn_path_overlap": 0.0},
    )
    assert report.metric("event_type_tvd").status == "fail"
    assert report.metric("topn_path_overlap").status == "pass"
    assert report.metric("session_length_ks").status == "info"
    assert report.status == "fail"
    assert not report.passed

    lenient = validate(real, real, thresholds={"event_type_tvd": 0.0})
    assert lenient.passed


def test_metric_result_evaluate() -> None:
    result = MetricResult(
        name="m", category="privacy", direction="higher_better", value=0.3
    )
    assert result.evaluate(0.2).status == "pass"
    assert result.evaluate(0.5).status == "fail"
    assert result.evaluate(None).status == "info"
    missing = MetricResult(
        name="m", category="privacy", direction="lower_better", value=None
    )
    assert missing.evaluate(0.1).status == "skip"
    assert missing.evaluate(0.1).threshold == 0.1


def test_report_contents(real: Dataset) -> None:
    generator = get_generator("markov")
    synthetic = generator.generate(generator.fit(real), 10, seed=5)
    report = validate(real, synthetic, seed=3, real_split="holdout", run_id="run-1")
    assert report.run_id == "run-1"
    assert report.real["fingerprint"] == real.fingerprint()
    assert report.synthetic["generation"]["generator_id"] == "markov"
    assert dict(report.protocol) == {
        "real_split": "holdout",
        "synthetic_sessions": 10,
        "seed": 3,
        "ids_remapped": True,
    }
    assert [m.name for m in report.metrics] == list(DEFAULT_METRICS)
    assert report.duration_s >= 0
    with pytest.raises(KeyError):
        report.metric("missing")

    plain = validate(real, real)
    assert plain.protocol["ids_remapped"] is None
    assert "generation" not in plain.synthetic


def test_report_serialization(real: Dataset, disjoint: Dataset) -> None:
    report = validate(real, disjoint, thresholds={"event_type_tvd": 0.5})
    data = json.loads(json.dumps(report.to_dict()))
    assert data["status"] == "fail"
    assert data["report_version"] == "1.0"
    assert "do not prove" in data["notes"][0]
    restored = ValidationReport.from_dict(data | {"future_field": 1})
    assert restored.to_dict() == report.to_dict()


def test_report_markdown(real: Dataset) -> None:
    singles = build([("view",)])
    text = validate(real, singles, thresholds={"event_type_tvd": 0.9}).to_markdown()
    assert text.startswith("# Validation report: **PASS**")
    assert "| event_type_tvd | marginal |" in text
    assert "| 0.9 | lower | PASS |" in text
    assert "Skipped:" in text
    assert "- bigram_tvd: a dataset has no multi-event sessions" in text


# --------------------------------------------------------------------------
# Argument errors
# --------------------------------------------------------------------------


def test_requires_sessionized(real: Dataset) -> None:
    raw = Dataset(dataset_id="raw", events=real.events)
    with pytest.raises(ValidationError) as info:
        validate(raw, real)
    assert info.value.context["dataset"] == "real"
    with pytest.raises(ValidationError):
        validate(real, raw)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_rejects_bad_seed(real: Dataset, seed: Any) -> None:
    with pytest.raises(ValidationError):
        validate(real, real, seed=seed)


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"metrics": ["nope"]}, "metric_unknown"),
        ({"metrics": ["bigram_tvd", "bigram_tvd"]}, "metric_duplicate"),
        (
            {"metrics": ["bigram_tvd"], "thresholds": {"event_type_tvd": 0.1}},
            "metric_not_selected",
        ),
        (
            {"metrics": ["bigram_tvd"], "params": {"nn_distance_p05": {}}},
            "metric_not_selected",
        ),
    ],
)
def test_config_errors(real: Dataset, kwargs: dict[str, Any], code: str) -> None:
    with pytest.raises(ConfigError) as info:
        validate(real, real, **kwargs)
    assert info.value.code == code


# --------------------------------------------------------------------------
# Registry and custom metrics
# --------------------------------------------------------------------------


class SessionCountRatio(BaseMetric):
    name: ClassVar[str] = "session_count_ratio"
    category: ClassVar[Any] = "utility"
    direction: ClassVar[Any] = "higher_better"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> tuple[float | None, dict[str, Any]]:
        return synthetic.n_sessions / real.n_sessions, {"real": real.n_sessions}


class Broken(SessionCountRatio):
    name: ClassVar[str] = "broken"

    def _compute(
        self, real: Dataset, synthetic: Dataset, context: ValidationContext
    ) -> tuple[float | None, dict[str, Any]]:
        return math.inf, {}


@pytest.fixture
def custom() -> Iterator[None]:
    register_metric(SessionCountRatio())
    register_metric(Broken())
    yield
    unregister_metric("session_count_ratio")
    unregister_metric("broken")


def test_builtins_registered() -> None:
    assert set(DEFAULT_METRICS) <= set(available_metrics())
    assert len(BUILTIN_METRICS) == 11
    assert get_metric("bigram_tvd").description


def test_custom_metric(real: Dataset, custom: None) -> None:
    report = validate(
        real,
        real,
        metrics=["session_count_ratio"],
        thresholds={"session_count_ratio": 1.0},
    )
    assert report.metric("session_count_ratio").value == 1.0
    assert report.passed
    with pytest.raises(ValidationError) as info:
        validate(real, real, metrics=["broken"])
    assert info.value.code == "validation_invalid_value"


def test_registry_errors(custom: None) -> None:
    with pytest.raises(PluginError) as info:
        register_metric(SessionCountRatio())
    assert info.value.code == "plugin_duplicate"
    register_metric(SessionCountRatio(), replace=True)
    with pytest.raises(PluginError):
        register_metric("bigram_tvd")  # type: ignore[arg-type]

    class Nameless(SessionCountRatio):
        name: ClassVar[str] = "has space"

    with pytest.raises(PluginError) as info:
        register_metric(Nameless())
    assert info.value.code == "plugin_invalid_name"
    with pytest.raises(ConfigError):
        get_metric("nope")
