"""Tests for benchmark protocols, runner, reports, and the demo/benchmark CLI."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

import eduloggen as elg
from eduloggen.analysis import sessionize
from eduloggen.benchmark import (
    PROTOCOLS,
    BenchmarkReport,
    get_protocol,
    run_benchmark,
    split_by_learner,
    summarize,
)
from eduloggen.cli import main
from eduloggen.config import AppConfig
from eduloggen.core import BenchmarkError, ConfigError, FitError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import (
    IndependentGenerator,
    register_generator,
    unregister_generator,
)
from eduloggen.models import Dataset


@pytest.fixture(scope="module")
def demo() -> Dataset:
    return sessionize(demo_dataset(30, seed=2))


# --------------------------------------------------------------------------
# Protocol and split
# --------------------------------------------------------------------------


def test_protocol_registry() -> None:
    spec = get_protocol("session_fidelity_v1")
    assert spec is PROTOCOLS["session_fidelity_v1"]
    assert spec.repeats == 3
    assert spec.to_dict()["metrics"] == list(spec.metrics)
    with pytest.raises(ConfigError) as info:
        get_protocol("v0")
    assert info.value.code == "benchmark_unknown_protocol"


def test_split_by_learner(demo: Dataset) -> None:
    train, holdout = split_by_learner(demo, 0.3, seed=0)
    assert not train.learner_ids & holdout.learner_ids
    assert train.learner_ids | holdout.learner_ids == demo.learner_ids
    assert len(holdout.learner_ids) == 9
    assert train.n_events + holdout.n_events == demo.n_events
    assert train.n_sessions + holdout.n_sessions == demo.n_sessions
    again = split_by_learner(demo, 0.3, seed=0)[1]
    assert again.learner_ids == holdout.learner_ids
    assert split_by_learner(demo, 0.3, seed=1)[1].learner_ids != holdout.learner_ids


def test_split_keeps_one_learner_on_each_side(demo: Dataset) -> None:
    train, holdout = split_by_learner(demo, 0.99, seed=0)
    assert len(train.learner_ids) == 1
    train, holdout = split_by_learner(demo, 0.001, seed=0)
    assert len(holdout.learner_ids) == 1


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.5])
def test_split_rejects_bad_fraction(demo: Dataset, fraction: float) -> None:
    with pytest.raises(BenchmarkError):
        split_by_learner(demo, fraction, seed=0)


def test_split_errors(demo: Dataset) -> None:
    with pytest.raises(BenchmarkError) as info:
        split_by_learner(Dataset(dataset_id="d", events=demo.events), 0.3, 0)
    assert info.value.code == "benchmark_not_sessionized"
    one = sessionize(Dataset(dataset_id="d", events=demo.events[:1]))
    with pytest.raises(BenchmarkError) as info:
        split_by_learner(one, 0.3, 0)
    assert info.value.code == "benchmark_insufficient_data"


def test_summarize() -> None:
    summary = summarize("lower_better", [1.0, None, 3.0])
    assert summary.values == (1.0, 3.0)
    assert summary.mean == 2.0
    assert summary.std == pytest.approx(2**0.5)
    assert (summary.min, summary.max) == (1.0, 3.0)
    assert summarize("lower_better", [5.0]).std == 0.0
    assert summarize("lower_better", [None]).mean is None


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def test_run_benchmark(demo: Dataset) -> None:
    report = run_benchmark(
        demo, ["markov", "semi_markov", "independent"], seed=4, repeats=2
    )
    assert [r.name for r in report.results] == ["markov", "semi_markov", "independent"]
    assert report.protocol["repeats"] == 2
    assert report.protocol["n_sessions"] == report.dataset["holdout"]["n_sessions"]
    assert report.dataset["fingerprint"] == demo.fingerprint()
    markov = report.result("markov")
    assert len(markov.seeds) == 2
    assert markov.error is None
    assert markov.fit_s is not None and markov.generate_s is not None
    assert len(markov.metrics["bigram_tvd"].values) == 2
    assert set(report.reference) == set(PROTOCOLS["session_fidelity_v1"].metrics)
    assert report.best("bigram_tvd") in {"markov", "semi_markov"}
    assert report.best("interevent_time_ks") == "semi_markov"
    with pytest.raises(KeyError):
        report.result("gan")


def test_run_benchmark_is_reproducible(demo: Dataset) -> None:
    first = run_benchmark(demo, ["markov"], seed=3, repeats=1)
    second = run_benchmark(demo, ["markov"], seed=3, repeats=1)
    assert first.result("markov").metrics == second.result("markov").metrics
    assert first.result("markov").seeds == second.result("markov").seeds


def test_hyperparameters_and_sample_size(demo: Dataset) -> None:
    report = run_benchmark(
        demo,
        ["markov"],
        repeats=1,
        n_sessions=5,
        hyperparameters={"markov": {"smoothing_alpha": 0.5}},
    )
    assert report.protocol["n_sessions"] == 5
    assert report.result("markov").hyperparameters == {"smoothing_alpha": 0.5}


class Failing(IndependentGenerator):
    name: ClassVar[str] = "failing"

    def _fit_family(self, *args: Any) -> dict[str, Any]:
        raise FitError("cannot fit", code="fit_test")


@pytest.fixture
def failing() -> Iterator[None]:
    register_generator("failing", Failing)
    yield
    unregister_generator("failing")


def test_failed_generator_is_recorded(demo: Dataset, failing: None) -> None:
    report = run_benchmark(demo, ["failing", "independent"], repeats=1)
    result = report.result("failing")
    assert result.error is not None and "fit_test" in result.error
    assert result.metrics == {}
    assert report.result("independent").error is None
    assert "| failing | - | - | [fit_test] cannot fit |" in report.to_markdown()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"generators": []}, BenchmarkError),
        ({"generators": ["markov", "markov"]}, BenchmarkError),
        ({"generators": ["gan"]}, ConfigError),
        ({"repeats": 0}, BenchmarkError),
        ({"n_sessions": 0}, BenchmarkError),
        ({"seed": -1}, BenchmarkError),
        ({"protocol": "v9"}, ConfigError),
    ],
)
def test_runner_argument_errors(
    demo: Dataset, kwargs: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        run_benchmark(demo, **kwargs)


def test_report_serialization(demo: Dataset) -> None:
    report = run_benchmark(demo, ["markov", "independent"], repeats=1)
    data = json.loads(json.dumps(report.to_dict()))
    assert "general ranking" in data["notes"][0]
    restored = BenchmarkReport.from_dict(data)
    assert restored.to_dict() == report.to_dict()
    text = report.to_markdown()
    assert text.startswith("# Benchmark `session_fidelity_v1`")
    assert "| Metric | Better | Reference | markov | independent |" in text
    assert "**" in text


def test_best_without_values(demo: Dataset) -> None:
    report = run_benchmark(demo, ["markov"], repeats=1)
    assert report.best("unknown_metric") is None


# --------------------------------------------------------------------------
# API and CLI
# --------------------------------------------------------------------------


def test_api_run_benchmark_uses_config() -> None:
    config = AppConfig.from_dict(
        {
            "benchmark": {"generators": ["markov", "independent"], "repeats": 1},
            "generator": {
                "name": "markov",
                "order": 2,
                "on_insufficient_data": "backoff",
            },
            "generation": {"seed": 7},
        }
    )
    report = elg.run_benchmark(elg.demo_dataset(20), config=config)
    assert [r.name for r in report.results] == ["markov", "independent"]
    assert report.seed == 7
    assert report.result("markov").hyperparameters["order"] == 2
    assert report.result("independent").hyperparameters == {}


def test_cli_demo_and_benchmark(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    demo_dir, out = tmp_path / "demo", tmp_path / "bench"
    assert main(["demo", "--output", str(demo_dir), "--learners", "20", "--quiet"]) == 0
    assert "20 synthetic learners" in capsys.readouterr().out
    assert (demo_dir / "run_manifest.json").exists()

    code = main(
        [
            "benchmark",
            "--input",
            str(demo_dir),
            "--generators",
            "markov,independent",
            "--seeds",
            "1",
            "--n-sessions",
            "10",
            "--output",
            str(out),
            "--quiet",
        ]
    )
    stdout = capsys.readouterr().out
    assert code == 0
    assert "| Metric | Better | Reference | markov | independent |" in stdout
    data = json.loads((out / "benchmark.json").read_text())
    assert data["protocol"]["n_sessions"] == 10
    assert data["protocol"]["repeats"] == 1
    assert (out / "benchmark.md").exists()


def test_cli_benchmark_exit_1_on_generator_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failing: None
) -> None:
    main(["demo", "--output", str(tmp_path / "d"), "--learners", "10", "--quiet"])
    command = f"benchmark --input {tmp_path / 'd'} --generators failing --seeds 1"
    code = main([*command.split(), "--quiet"])
    capsys.readouterr()
    assert code == 1
