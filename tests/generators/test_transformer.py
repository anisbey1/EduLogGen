"""Tests for the optional Transformer generator (skipped without PyTorch)."""

from __future__ import annotations

import builtins
import logging
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from eduloggen.analysis import interevent_times, session_sequences, sessionize
from eduloggen.core import ConfigError, FitError, GenerationError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import BUILTIN_GENERATORS, get_generator
from eduloggen.generators._neural import _bins, _learner_validation
from eduloggen.generators.transformer import (
    BOS,
    NO_GAP,
    PAD,
    _bin_ids,
    _windows,
)
from eduloggen.models import Dataset

from .conftest import build

SMALL = {
    "d_model": 16,
    "num_heads": 2,
    "num_layers": 1,
    "ff_dim": 32,
    "epochs": 3,
    "batch_size": 16,
    "dropout": 0.0,
}
WINDOW_HP = {"max_seq_len": 4, "window_overlap": 0.5}


@pytest.fixture(scope="module")
def data() -> Dataset:
    return sessionize(demo_dataset(40, seed=1))


# --------------------------------------------------------------------------
# configuration and registration (no PyTorch needed)
# --------------------------------------------------------------------------


def test_registered() -> None:
    assert "transformer" in BUILTIN_GENERATORS
    assert {"neural", "supports_timing"} <= get_generator("transformer").tags
    assert "supports_event_weights" not in get_generator("transformer").tags


@pytest.mark.parametrize(
    "change",
    [
        {"d_model": 30, "num_heads": 4},
        {"num_heads": 0},
        {"dropout": 1.0},
        {"window_overlap": 1.0},
        {"validation_fraction": 1.0},
        {"max_seq_len": 1},
        {"top_k": -1},
        {"temperature": 0},
        {"device": "tpu"},
        {"epochs": 2.5},
        {"unknown": 1},
    ],
)
def test_configuration_validation(change: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        get_generator("transformer").validate_hyperparameters({**SMALL, **change})


def test_defaults() -> None:
    hp = get_generator("transformer").validate_hyperparameters()
    assert (hp["d_model"], hp["num_heads"], hp["num_layers"]) == (128, 4, 2)
    assert (hp["max_seq_len"], hp["epochs"], hp["patience"]) == (256, 100, 10)
    assert hp["top_k"] == 0 and hp["temperature"] == 1.0 and hp["device"] == "cpu"


def test_missing_torch(data: Dataset, monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    with pytest.raises(FitError, match="eduloggen\\[neural\\]") as info:
        get_generator("transformer").fit(data, SMALL)
    assert info.value.code == "missing_dependency"


# --------------------------------------------------------------------------
# sequence preparation (no PyTorch needed)
# --------------------------------------------------------------------------


def test_target_alignment_short_session() -> None:
    ((ev_in, gap_in, ev_out, gap_out),) = _windows([10, 11, 12], [0, 1], WINDOW_HP)
    assert ev_in == [BOS, 10, 11]  # BOS, then e1, e2
    assert ev_out == [10, 11, 12]  # each position predicts the next event
    assert gap_in == [NO_GAP, NO_GAP, 0 + 2]  # gap leading to e2 enters at e2
    assert gap_out == [-1, 0, 1]  # position p predicts the gap after e_p


def test_single_event_session() -> None:
    assert _windows([10], [], WINDOW_HP) == [([BOS], [NO_GAP], [10], [-1])]


@pytest.mark.parametrize("n", [2, 4, 5, 9, 10, 37])
@pytest.mark.parametrize("overlap", [0.0, 0.25, 0.5, 0.75])
def test_windows_score_every_target_once(n: int, overlap: float) -> None:
    events = list(range(100, 100 + n))
    bins = list(range(n - 1))
    windows = _windows(events, bins, {"max_seq_len": 4, "window_overlap": overlap})
    assert [t for _, _, out, _ in windows for t in out if t >= 0] == events
    assert [t for _, _, _, out in windows for t in out if t >= 0] == bins
    assert all(len(w[0]) <= 4 and len(set(map(len, w))) == 1 for w in windows)
    if n > 4:  # later windows keep the overlap as (masked) context
        context = int(4 * overlap)
        assert all(out[:context] == [-1] * context for _, _, out, _ in windows[1:])


def test_bin_ids_match_bins() -> None:
    gaps = [1.0, 2.0, 2.0, 5.0, 60.0, 3600.0]
    bins = _bins(gaps, 32)
    ids = _bin_ids(gaps, bins["edges"])
    for gap, b in zip(gaps, ids, strict=True):
        assert gap in bins["values"][b]


def test_learner_validation_split() -> None:
    import random

    learners = ["a", "a", "b", "c", "c", "c", "d"]
    held = _learner_validation(learners, 0.25, random.Random(0))
    held_learners = {learners[i] for i in held}
    assert len(held_learners) == 1
    assert all((learners[i] in held_learners) == (i in held) for i in range(7))
    assert _learner_validation(["a", "a"], 0.5, random.Random(0)) == set()
    assert _learner_validation(learners, 0.0, random.Random(0)) == set()


# --------------------------------------------------------------------------
# network (PyTorch)
# --------------------------------------------------------------------------


def _net(
    n_events: int = 6, n_bins: int = 3, **hp: Any
) -> tuple[Any, Any, dict[str, Any]]:
    torch = pytest.importorskip("torch")
    from eduloggen.generators.transformer import _network

    params = get_generator("transformer").validate_hyperparameters({**SMALL, **hp})
    torch.manual_seed(0)
    net = _network(torch, n_events, n_bins, params)
    net.eval()
    return torch, net, params


def test_head_shapes() -> None:
    torch, net, _ = _net(n_events=7, n_bins=5)
    from eduloggen.generators.transformer import _forward

    events = torch.tensor([[BOS, 2, 3, 4]])
    gaps = torch.tensor([[NO_GAP, NO_GAP, 2, 3]])
    hidden = _forward(torch, net, events, gaps, None)
    assert tuple(net["event_head"](hidden).shape) == (1, 4, 7)
    assert tuple(net["time_head"](hidden).shape) == (1, 4, 5)


def test_no_causal_leakage() -> None:
    """Changing future inputs never changes earlier hidden states."""
    torch, net, _ = _net()
    from eduloggen.generators.transformer import _forward

    events = torch.tensor([[BOS, 2, 3, 4, 5, 2]])
    gaps = torch.tensor([[NO_GAP, NO_GAP, 2, 3, 4, 2]])
    base = _forward(torch, net, events, gaps, None)
    for cut in range(1, 6):
        changed_e, changed_g = events.clone(), gaps.clone()
        changed_e[0, cut:] = 3
        changed_g[0, cut:] = 4
        other = _forward(torch, net, changed_e, changed_g, None)
        assert torch.allclose(base[0, :cut], other[0, :cut], atol=1e-6)
        assert not torch.allclose(base[0, cut:], other[0, cut:])


def test_padding_does_not_change_real_positions() -> None:
    torch, net, _ = _net()
    from eduloggen.generators.transformer import _forward

    events = torch.tensor([[BOS, 2, 3, PAD, PAD], [BOS, 2, 3, 4, 5]])
    gaps = torch.tensor([[NO_GAP, NO_GAP, 2, PAD, PAD], [NO_GAP, NO_GAP, 2, 3, 2]])
    padded = _forward(torch, net, events, gaps, events == PAD)
    alone = _forward(torch, net, events[:1, :3], gaps[:1, :3], None)
    assert torch.allclose(padded[0, :3], alone[0], atol=1e-6)


def test_kv_cache_matches_full_forward() -> None:
    torch, net, _ = _net()
    from eduloggen.generators.transformer import _forward, _step

    events = [BOS, 2, 3, 4, 2, 5]
    gaps = [NO_GAP, NO_GAP, 2, 3, 4, 2]
    full = _forward(torch, net, torch.tensor([events]), torch.tensor([gaps]), None)
    cache = None
    for n in range(1, len(events) + 1):
        hidden, cache = _step(torch, net, events[:n], gaps[:n], cache)
        assert torch.allclose(hidden, full[0, n - 1], atol=1e-5)


def test_loss_ignores_masked_targets() -> None:
    torch, net, params = _net()
    from eduloggen.generators.transformer import _loss

    device = torch.device("cpu")
    one = [([BOS, 2, 3], [NO_GAP, NO_GAP, 2], [2, 3, 4], [-1, 0, 1])]
    padded = [*one, ([BOS], [NO_GAP], [-1], [-1])]  # a window with no targets
    with torch.no_grad():
        a = float(_loss(torch, net, one, params, device))
        b = float(_loss(torch, net, padded, params, device))
    assert math.isfinite(a) and a == pytest.approx(b, rel=1e-6)


def test_sampling_weights() -> None:
    torch = pytest.importorskip("torch")
    from eduloggen.generators.transformer import _weights

    logits = torch.tensor([1.0, 3.0, 2.0])
    assert sum(_weights(torch, logits, 1.0, 0)) == pytest.approx(1.0)
    assert _weights(torch, logits, 1.0, 1) == [0.0, 1.0, 0.0]
    sharp = _weights(torch, logits, 0.1, 0)
    flat = _weights(torch, logits, 10.0, 0)
    assert max(sharp) > max(flat)
    assert sorted(_weights(torch, logits, 1.0, 2))[0] == 0.0


# --------------------------------------------------------------------------
# fit and generate (PyTorch)
# --------------------------------------------------------------------------


def test_fit_generate_save_load(data: Dataset, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    gen = get_generator("transformer")
    model = gen.fit(data, SMALL)
    params = model.parameters["transformer"]
    training = params["training"]
    assert len(training["history"]) == training["epochs_trained"] <= 3
    assert all(
        math.isfinite(h["train"]) and math.isfinite(h["validation"])
        for h in training["history"]
    )
    assert training["validation_learners"] >= 1
    assert training["train_learners"] + training["validation_learners"] == len(
        data.learner_ids
    )
    assert gen.fit(data, SMALL).fingerprint() == model.fingerprint()  # deterministic

    a = gen.generate(model, 25, seed=3)
    assert a.events == gen.generate(model, 25, seed=3).events
    assert a.events != gen.generate(model, 25, seed=4).events
    assert {t for s in session_sequences(a) for t in s} <= set(model.vocabulary)
    for session in a.sessions or ():
        assert session.end_time >= session.start_time
    stamps: dict[str | None, list[Any]] = {}
    for event in a.events:
        stamps.setdefault(event.session_id, []).append(event.timestamp)
    assert all(v == sorted(v) for v in stamps.values())
    assert len({e.event_id for e in a.events}) == len(a.events)

    target = gen.save(model, tmp_path / "model")
    loaded = gen.load(target)
    assert loaded.fingerprint() == model.fingerprint()
    assert gen.generate(loaded, 25, seed=3).events == a.events
    assert gen.describe(model)["n_parameters"] == params["n_parameters"]


def test_overfits_a_deterministic_corpus() -> None:
    pytest.importorskip("torch")
    corpus = build([("a", "b", "c", "a", "b", "c")] * 30, learners=6)
    gen = get_generator("transformer")
    model = gen.fit(
        corpus,
        {
            **SMALL,
            "epochs": 80,
            "patience": 80,
            "learning_rate": 0.01,
            "validation_fraction": 0.0,
        },
    )
    history = model.parameters["transformer"]["training"]["history"]
    assert history[-1]["train"] < history[0]["train"] / 3
    synthetic = gen.generate(model, 20, seed=0)
    pairs = [(x, y) for s in session_sequences(synthetic) for x, y in pairwise(s)]
    allowed = {("a", "b"), ("b", "c"), ("c", "a")}
    assert sum(p in allowed for p in pairs) / len(pairs) > 0.9


def test_long_sessions_use_windows_and_rolling_context() -> None:
    pytest.importorskip("torch")
    long = build([tuple("abcab" * 30)] * 6, learners=3)  # 150 events each
    gen = get_generator("transformer")
    model = gen.fit(long, {**SMALL, "max_seq_len": 16, "epochs": 2})
    training = model.parameters["transformer"]["training"]
    assert training["train_windows"] > training["train_sessions"]
    assert training["context_length"] == 16 and training["longest_session"] == 150
    synthetic = gen.generate(model, 3, seed=1)
    assert {len(s) for s in session_sequences(synthetic)} == {150}


def test_discrete_gaps_reproduced() -> None:
    pytest.importorskip("torch")
    discrete = build(
        [("a", "b", "a"), ("b", "a")] * 8, gap_after={"a": 600.0, "b": 1200.0}
    )
    gen = get_generator("transformer")
    model = gen.fit(discrete, {**SMALL, "validation_fraction": 0.0})
    assert set(interevent_times(gen.generate(model, 20, seed=2))) <= {600.0, 1200.0}


def test_single_event_sessions_and_no_validation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pytest.importorskip("torch")
    short = build([("a",), ("b",), ("a", "b")] * 4, learners=1)
    with caplog.at_level(logging.WARNING, logger="eduloggen"):
        model = get_generator("transformer").fit(short, SMALL)
    assert "without early stopping" in caplog.text
    assert model.parameters["transformer"]["training"]["validation_learners"] == 0
    assert get_generator("transformer").generate(model, 10, seed=0).n_sessions == 10


def test_temperature_and_top_k(data: Dataset) -> None:
    pytest.importorskip("torch")
    gen = get_generator("transformer")
    greedy = gen.fit(data, {**SMALL, "top_k": 1})
    synthetic = gen.generate(greedy, 10, seed=0)
    assert synthetic.n_sessions == 10


def test_non_finite_loss_is_an_error(
    data: Dataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from eduloggen.generators import transformer

    monkeypatch.setattr(
        transformer,
        "_loss",
        lambda *a, **k: torch.tensor(float("nan"), requires_grad=True),
    )
    with pytest.raises(FitError, match="diverged") as info:
        get_generator("transformer").fit(data, SMALL)
    assert info.value.code == "fit_diverged"


def test_unavailable_device(data: Dataset) -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("CUDA is available here")
    with pytest.raises(FitError, match="not available"):
        get_generator("transformer").fit(data, {**SMALL, "device": "cuda"})


def test_malformed_model(data: Dataset) -> None:
    pytest.importorskip("torch")
    gen = get_generator("transformer")
    model = gen.fit(data, SMALL)
    broken = type(model)(
        **{
            **{f: getattr(model, f) for f in model.__dataclass_fields__},
            "parameters": {**model.parameters, "transformer": {"vocabulary": ["x"]}},
        }
    )
    with pytest.raises(GenerationError, match="malformed"):
        gen.generate(broken, 5, seed=0)


def test_validation_and_benchmark_accept_output(data: Dataset) -> None:
    pytest.importorskip("torch")
    import eduloggen as elg
    from eduloggen.benchmark import run_benchmark

    model = get_generator("transformer").fit(data, SMALL)
    synthetic = get_generator("transformer").generate(model, 30, seed=0)
    assert elg.validate(data, synthetic).metrics
    report = run_benchmark(
        data, ["transformer"], seed=0, repeats=1, hyperparameters={"transformer": SMALL}
    )
    assert report.results[0].error is None


def test_cli_fit_and_generate(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from eduloggen.cli import main

    d = tmp_path
    sets = [
        a
        for kv in ("d_model=16", "num_heads=2", "num_layers=1", "ff_dim=32", "epochs=2")
        for a in ("--set", kv)
    ]
    assert (
        main(["demo", "--output", str(d / "demo"), "--learners", "20", "--quiet"]) == 0
    )
    assert (
        main(
            [
                "fit",
                "--input",
                str(d / "demo"),
                "--generator",
                "transformer",
                *sets,
                "--output",
                str(d / "m"),
                "--quiet",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "generate",
                "--model",
                str(d / "m"),
                "--n-sessions",
                "15",
                "--seed",
                "1",
                "--output",
                str(d / "s"),
                "--quiet",
            ]
        )
        == 0
    )
    assert (d / "s").is_dir()
