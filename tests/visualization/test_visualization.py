"""Tests for figures, export, and the plot command."""

from __future__ import annotations

import builtins
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("matplotlib")

from eduloggen.analysis import sessionize
from eduloggen.benchmark import run_benchmark
from eduloggen.cli import main
from eduloggen.core import ConfigError, ExportError, IoError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import get_generator
from eduloggen.models import Dataset
from eduloggen.validation import validate
from eduloggen.visualization import (
    DATASET_PLOTS,
    plot_benchmark,
    plot_datasets,
    plot_event_frequencies,
    plot_interevent_times,
    plot_sankey,
    plot_session_lengths,
    plot_timeline,
    plot_transition_graph,
    plot_transition_heatmap,
    plot_validation_report,
    require_matplotlib,
    sankey_flows,
    save_figure,
    transition_matrix,
)
from eduloggen.visualization.timeline import MAX_SESSIONS


@pytest.fixture(scope="module")
def real() -> Dataset:
    return sessionize(demo_dataset(25, seed=3))


@pytest.fixture(scope="module")
def synthetic(real: Dataset) -> Dataset:
    generator = get_generator("semi_markov")
    return generator.generate(generator.fit(real), 40, seed=1)


def _texts(figure: Any) -> str:
    return " ".join(
        t.get_text() for t in figure.findobj(lambda o: hasattr(o, "get_text"))
    )


def test_every_dataset_plot_renders(real: Dataset, synthetic: Dataset) -> None:
    for name, function in DATASET_PLOTS.items():
        for other in (None, synthetic):
            figure = function(real, other)
            assert figure.axes, name
            require_matplotlib().close(figure)


def test_labels_never_show_learner_ids(real: Dataset, synthetic: Dataset) -> None:
    for function in DATASET_PLOTS.values():
        figure = function(real, synthetic)
        text = _texts(figure)
        require_matplotlib().close(figure)
        assert "learner-" not in text
        assert ":s" not in text


def test_event_frequencies(real: Dataset, synthetic: Dataset) -> None:
    figure = plot_event_frequencies(real, synthetic, top_n=3)
    ax = figure.axes[0]
    assert len(ax.get_xticklabels()) == 3
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == ["real", "synthetic"]
    heights = [bar.get_height() for bar in ax.patches[:3]]  # type: ignore[attr-defined]
    assert heights == sorted(heights, reverse=True)
    single = plot_event_frequencies(real)
    assert single.axes[0].get_legend() is None


def test_session_length_bins_are_integers(real: Dataset) -> None:
    ax = plot_session_lengths(real).axes[0]
    edges = [patch.get_x() for patch in ax.patches]  # type: ignore[attr-defined]
    assert edges[0] == pytest.approx(0.5)
    assert all(b - a == pytest.approx(1.0) for a, b in pairwise(edges))


def test_transition_matrix(real: Dataset) -> None:
    matrix = transition_matrix(real, ["navigate", "view"])
    assert matrix[0] == [0.0, 1.0]
    assert sum(matrix[1]) in (0.0, pytest.approx(1.0))
    figure = plot_transition_heatmap(real, top_n=4)
    assert len(figure.axes[0].get_xticklabels()) == 4


def test_timeline_samples_and_caps(real: Dataset) -> None:
    ax = plot_timeline(real, n_sessions=5, seed=1).axes[0]
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == [f"session {i}" for i in range(1, 6)]
    capped = plot_timeline(real, n_sessions=1000).axes[0]
    assert len(capped.get_yticklabels()) == min(MAX_SESSIONS, real.n_sessions)


def test_empty_data_does_not_crash() -> None:
    empty = sessionize(Dataset(dataset_id="e", events=()))
    for function in (plot_session_lengths, plot_interevent_times, plot_timeline):
        require_matplotlib().close(function(empty))


def test_validation_and_benchmark_charts(real: Dataset, synthetic: Dataset) -> None:
    report = validate(real, synthetic, thresholds={"event_type_tvd": 0.01})
    text = _texts(plot_validation_report(report))
    assert "Validation: FAIL" in text
    assert "(off scale)" in text
    assert "fail" in text
    bench = run_benchmark(real, ["markov", "independent"], repeats=2)
    figure = plot_benchmark(bench)
    assert "real-data reference" in figure.get_suptitle()
    visible = [ax for ax in figure.axes if ax.get_visible()]
    assert len(visible) == len(bench.protocol["metrics"])


def test_save_figure(tmp_path: Path, real: Dataset) -> None:
    paths = save_figure(
        plot_session_lengths(real),
        tmp_path / "out" / "lengths.png",
        formats=["png", "svg", "pdf"],
    )
    assert [p.suffix for p in paths] == [".png", ".svg", ".pdf"]
    assert all(p.stat().st_size > 0 for p in paths)
    assert (
        save_figure(plot_session_lengths(real), tmp_path / "x.svg")[0].suffix == ".svg"
    )


def test_svg_output_is_reproducible(tmp_path: Path, real: Dataset) -> None:
    a = save_figure(plot_session_lengths(real), tmp_path / "a.svg")[0].read_bytes()
    b = save_figure(plot_session_lengths(real), tmp_path / "b.svg")[0].read_bytes()
    assert a == b


@pytest.mark.parametrize(("name", "formats"), [("x.gif", None), ("x", ["jpg"])])
def test_save_figure_rejects_formats(
    tmp_path: Path, real: Dataset, name: str, formats: list[str] | None
) -> None:
    with pytest.raises(ExportError):
        save_figure(plot_session_lengths(real), tmp_path / name, formats=formats)


def test_plot_datasets(tmp_path: Path, real: Dataset, synthetic: Dataset) -> None:
    written = plot_datasets(
        real, synthetic, tmp_path, plots=["timeline", "transitions"], formats=["svg"]
    )
    assert set(written) == {"timeline", "transitions"}
    assert (tmp_path / "transitions.svg").exists()
    with pytest.raises(ConfigError) as info:
        plot_datasets(real, None, tmp_path, plots=["pie"])
    assert info.value.code == "plot_unknown"


def test_missing_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(IoError) as info:
        require_matplotlib()
    assert "eduloggen[viz]" in info.value.message


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def test_cli_plot(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corpus, figures = tmp_path / "demo", tmp_path / "figures"
    assert main(["demo", "--output", str(corpus), "--learners", "15", "--quiet"]) == 0
    bench = tmp_path / "bench"
    assert (
        main(
            [
                "benchmark",
                "--input",
                str(corpus),
                "--generators",
                "markov",
                "--seeds",
                "1",
                "--output",
                str(bench),
                "--quiet",
            ]
        )
        == 0
    )
    capsys.readouterr()
    code = main(
        [
            "plot",
            "--real",
            str(corpus),
            "--plots",
            "session_lengths,transitions",
            "--benchmark",
            str(bench / "benchmark.json"),
            "--format",
            "png,svg",
            "--output",
            str(figures),
            "--quiet",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert {p.name for p in figures.iterdir()} == {
        "session_lengths.png",
        "session_lengths.svg",
        "transitions.png",
        "transitions.svg",
        "benchmark.png",
        "benchmark.svg",
    }
    assert out.count("wrote ") == 6


def test_cli_plot_validation_report(
    tmp_path: Path,
    real: Dataset,
    synthetic: Dataset,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(validate(real, synthetic).to_dict()))
    assert (
        main(
            [
                "plot",
                "--validation",
                str(path),
                "--output",
                str(tmp_path / "f"),
                "--quiet",
            ]
        )
        == 0
    )
    assert (tmp_path / "f" / "validation.png").exists()


def test_cli_plot_requires_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["plot", "--output", str(tmp_path), "--quiet"]) == 2
    assert "cli_missing_argument" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Sankey and transition graph
# --------------------------------------------------------------------------


def test_sankey_flows_counts_paths_and_ends() -> None:
    from eduloggen.visualization.sankey import END, OTHER

    layout = sankey_flows(
        [("a", "b", "c"), ("a", "b"), ("a", "c", "d"), ("x",)], steps=3, top_n=2
    )
    assert layout.nodes[0] == {"a": 3, "x": 1}
    assert layout.nodes[1] == {"b": 2, "c": 1, END: 1}
    assert layout.nodes[2] == {"c": 1, "d": 1, END: 1}
    assert layout.flows[0] == {("a", "b"): 2, ("a", "c"): 1, ("x", END): 1}
    assert layout.flows[1] == {("b", "c"): 1, ("b", END): 1, ("c", "d"): 1}

    grouped = sankey_flows([("a",), ("b",), ("c",)], steps=1, top_n=1)
    assert grouped.nodes[0] == {"a": 1, OTHER: 2}
    assert grouped.flows == ()


def test_sankey_flow_conservation(real: Dataset) -> None:
    from eduloggen.analysis import session_sequences

    layout = sankey_flows(session_sequences(real), steps=5)
    assert sum(layout.nodes[0].values()) == real.n_sessions
    for step, flows in enumerate(layout.flows):
        out: dict[str, int] = {}
        into: dict[str, int] = {}
        for (src, dst), count in flows.items():
            out[src] = out.get(src, 0) + count
            into[dst] = into.get(dst, 0) + count
        assert out == {k: v for k, v in layout.nodes[step].items() if k != "(end)"}
        assert into == layout.nodes[step + 1]


def test_plot_sankey(real: Dataset, synthetic: Dataset) -> None:
    figure = plot_sankey(real, synthetic, steps=3, top_n=4)
    titles = [ax.get_title() for ax in figure.axes]
    assert titles == [
        "Session pathways, first 3 steps (real)",
        "Session pathways, first 3 steps (synthetic)",
    ]
    assert "step 3" in _texts(figure)
    empty = sessionize(Dataset(dataset_id="e", events=()))
    assert "no sessions" in _texts(plot_sankey(empty))


def test_plot_sankey_draws_end_and_other_nodes() -> None:
    from ..generators.conftest import build

    data = build([("a", "b"), ("a",), ("a", "b", "c"), ("d", "e"), ("f", "e")])
    text = _texts(plot_sankey(data, steps=3, top_n=1))
    assert "(end) (" in text
    assert "other (" in text


def test_plot_transition_graph(real: Dataset, synthetic: Dataset) -> None:
    figure = plot_transition_graph(real, synthetic, top_n=4, max_edges=5)
    assert [ax.get_title() for ax in figure.axes] == [
        "Transition graph (real)",
        "Transition graph (synthetic)",
    ]
    from matplotlib.patches import Circle, FancyArrowPatch

    ax = figure.axes[0]
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    nodes = [p for p in ax.patches if isinstance(p, Circle) and p.get_fill()]
    assert len(nodes) == 4
    assert 0 < len(arrows) <= 5
    single = plot_transition_graph(real)
    assert len(single.axes) == 1
