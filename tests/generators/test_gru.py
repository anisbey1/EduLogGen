"""Tests for the optional GRU generator (skipped without PyTorch)."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import pytest

from eduloggen.analysis import session_sequences, sessionize
from eduloggen.core import ConfigError, FitError, GenerationError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import BUILTIN_GENERATORS, get_generator
from eduloggen.models import Dataset

from .conftest import build

SMALL = {"embedding_dim": 8, "hidden_dim": 8, "epochs": 3, "batch_size": 16}


@pytest.fixture(scope="module")
def data() -> Dataset:
    return sessionize(demo_dataset(40, seed=1))


def test_registered_without_torch_import() -> None:
    assert "gru" in BUILTIN_GENERATORS
    assert "neural" in get_generator("gru").tags


def test_fit_generate_save_load(data: Dataset, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    gru = get_generator("gru")
    model = gru.fit(data, SMALL)
    params = model.parameters["gru"]
    assert params["epochs_trained"] <= SMALL["epochs"]
    assert params["n_parameters"] > 0
    assert set(params["vocabulary"]) == set(model.vocabulary)
    assert gru.fit(data, SMALL).fingerprint() == model.fingerprint()  # deterministic

    a = gru.generate(model, 30, seed=3)
    b = gru.generate(model, 30, seed=3)
    assert a.events == b.events
    assert a.n_sessions == 30
    assert {t for s in session_sequences(a) for t in s} <= set(model.vocabulary)
    for session in a.sessions or ():
        assert session.end_time >= session.start_time

    target = gru.save(model, tmp_path / "model")
    assert gru.load(target).fingerprint() == model.fingerprint()
    assert gru.describe(model)["n_parameters"] == params["n_parameters"]


def test_single_event_sessions_and_validation_split() -> None:
    pytest.importorskip("torch")
    short = build([("a",), ("b",), ("a", "b")] * 4)
    model = get_generator("gru").fit(short, {**SMALL, "validation_fraction": 0.0})
    synthetic = get_generator("gru").generate(model, 10, seed=0)
    assert synthetic.n_sessions == 10


@pytest.mark.parametrize(
    "change",
    [
        {"hidden_dim": 0},
        {"epochs": 1.5},
        {"validation_fraction": 1.0},
        {"learning_rate": -1},
        {"seed": -1},
        {"timing": "gamma"},
        {"n_bins": 0},
    ],
)
def test_hyperparameter_validation(change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        get_generator("gru").validate_hyperparameters({**SMALL, **change})


def test_malformed_model(data: Dataset) -> None:
    pytest.importorskip("torch")
    gru = get_generator("gru")
    model = gru.fit(data, SMALL)
    broken = type(model)(
        **{
            **{f: getattr(model, f) for f in model.__dataclass_fields__},
            "parameters": {**model.parameters, "gru": {"vocabulary": ["x"]}},
        }
    )
    with pytest.raises(GenerationError, match="malformed"):
        gru.generate(broken, 5, seed=0)


def test_missing_torch(data: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(FitError, match="eduloggen\\[neural\\]") as info:
        get_generator("gru").fit(data, SMALL)
    assert info.value.code == "missing_dependency"


def test_mixture_timing_head(data: Dataset) -> None:
    pytest.importorskip("torch")
    model = get_generator("gru").fit(data, {**SMALL, "timing": "mixture"})
    assert model.parameters["gru"]["timing_bins"] is None
    synthetic = get_generator("gru").generate(model, 10, seed=1)
    assert synthetic.n_sessions == 10


def test_binned_timing_reproduces_discrete_gaps() -> None:
    pytest.importorskip("torch")
    from eduloggen.analysis import interevent_times

    discrete = build(
        [("a", "b", "a"), ("b", "a")] * 8, gap_after={"a": 600.0, "b": 1200.0}
    )
    gru = get_generator("gru")
    model = gru.fit(discrete, {**SMALL, "validation_fraction": 0.0})
    bins = model.parameters["gru"]["timing_bins"]
    assert [sorted(set(v)) for v in bins["values"]] == [[600.0], [1200.0]]
    gaps = set(interevent_times(gru.generate(model, 20, seed=2)))
    assert gaps <= {600.0, 1200.0}


def test_bins_helper() -> None:
    from eduloggen.generators.gru import _bins

    assert _bins([], 4) == {"edges": [], "values": [[1.0]]}
    many = _bins([float(i) for i in range(1, 101)], 4)
    assert len(many["values"]) == 4
    assert sum(len(v) for v in many["values"]) == 100
