"""Tests for plugin registration, entry-point discovery, and the CLI."""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path
from typing import Any

import pytest

from eduloggen.analysis import sessionize
from eduloggen.benchmark import get_protocol, run_benchmark
from eduloggen.cli import main
from eduloggen.core import PluginError
from eduloggen.datasets import demo_dataset
from eduloggen.generators import available_generators, get_generator
from eduloggen.io import available_readers, get_reader, ingest
from eduloggen.plugins import (
    KINDS,
    PluginInfo,
    discover_plugins,
    list_plugins,
    register_plugin,
    unregister_plugin,
)
from eduloggen.validation import available_metrics, validate
from eduloggen.visualization import DATASET_PLOTS

from . import sample_plugin

MODULE = "tests.plugins.sample_plugin"
VALID: dict[str, tuple[str, str]] = {
    "generator": ("echo", "EchoGenerator"),
    "metric": ("session_ratio", "SessionRatio"),
    "reader": ("pipe", "PipeReader"),
    "visualizer": ("event_count", "event_count_plot"),
    "benchmark_suite": ("quick_check", "quick_check"),
}


def ep(kind: str, name: str, attr: str) -> EntryPoint:
    return EntryPoint(
        name=name, value=f"{MODULE}:{attr}", group=f"eduloggen.plugins.{kind}"
    )


@pytest.fixture(autouse=True)
def clean() -> Iterator[None]:
    yield
    for kind, (name, _) in VALID.items():
        unregister_plugin(kind, name)  # type: ignore[arg-type]
    for name in ("not_a_generator", "something_else", "crash", "bad_protocol", "eggs"):
        for kind in KINDS:
            unregister_plugin(kind, name)


# --------------------------------------------------------------------------
# Explicit registration
# --------------------------------------------------------------------------


def test_register_every_kind(tmp_path: Path) -> None:
    for kind, (name, attr) in VALID.items():
        info = register_plugin(kind, name, getattr(sample_plugin, attr))  # type: ignore[arg-type]
        assert info == PluginInfo(
            kind=kind,  # type: ignore[arg-type]
            name=name,
            source="api",
            tags=("baseline", "third_party") if kind == "generator" else (),
        )
    assert "echo" in available_generators()
    assert "session_ratio" in available_metrics()
    assert "pipe" in available_readers()
    assert "event_count" in DATASET_PLOTS
    assert get_protocol("quick_check").repeats == 1

    source = tmp_path / "log.pipe"
    source.write_text(
        "id=1|user=u|time=2026-03-01T09:00:00Z|item=a|action=view\n"
        "id=2|user=u|time=2026-03-01T09:01:00Z|item=b|action=submit\n"
    )
    from eduloggen.io import FieldMapping

    mapping = FieldMapping.from_dict(
        {
            "fields": {
                "event_id": "id",
                "learner_id": "user",
                "timestamp": "time",
                "activity_id": "item",
                "event_type": "action",
            }
        }
    )
    result = ingest(source, mapping)
    assert result.dataset.n_events == 2
    assert isinstance(get_reader("auto", source), sample_plugin.PipeReader)


def test_registered_plugins_work_in_the_pipeline() -> None:
    for kind, (name, attr) in VALID.items():
        register_plugin(kind, name, getattr(sample_plugin, attr))  # type: ignore[arg-type]
    real = sessionize(demo_dataset(15))
    generator = get_generator("echo")
    synthetic = generator.generate(generator.fit(real), 10, seed=0)
    report = validate(real, synthetic, metrics=["session_ratio"])
    assert report.metric("session_ratio").value == pytest.approx(10 / real.n_sessions)
    bench = run_benchmark(real, ["echo"], protocol="quick_check")
    assert bench.protocol["metrics"] == ["event_type_tvd", "bigram_tvd"]
    pytest.importorskip("matplotlib")
    assert DATASET_PLOTS["event_count"](real, synthetic).axes


def test_builtins_cannot_be_replaced() -> None:
    for kind, name in (
        ("generator", "markov"),
        ("metric", "bigram_tvd"),
        ("reader", "csv"),
        ("visualizer", "timeline"),
        ("benchmark_suite", "session_fidelity_v1"),
    ):
        with pytest.raises(PluginError) as info:
            register_plugin(kind, name, object(), replace=True)  # type: ignore[arg-type]
        assert info.value.code == "plugin_duplicate"
        unregister_plugin(kind, name)  # type: ignore[arg-type]
    assert "markov" in available_generators()


def test_duplicates_need_replace() -> None:
    register_plugin("generator", "echo", sample_plugin.EchoGenerator)
    with pytest.raises(PluginError) as info:
        register_plugin("generator", "echo", sample_plugin.EchoGenerator)
    assert info.value.code == "plugin_duplicate"
    register_plugin("generator", "echo", sample_plugin.EchoGenerator, replace=True)


@pytest.mark.parametrize(
    ("kind", "name", "obj", "code"),
    [
        ("gizmo", "x", object(), "plugin_unknown_kind"),
        ("generator", "bad-name", sample_plugin.EchoGenerator, "plugin_invalid_name"),
        (
            "generator",
            "not_a_generator",
            sample_plugin.NotAGenerator,
            "plugin_contract_violation",
        ),
        ("generator", "eggs", "not callable", "plugin_contract_violation"),
        ("generator", "eggs", lambda: object(), "plugin_contract_violation"),
        ("generator", "eggs", sample_plugin.Misnamed, "plugin_contract_violation"),
        ("metric", "eggs", sample_plugin.NotAGenerator, "plugin_contract_violation"),
        ("metric", "eggs", 42, "plugin_contract_violation"),
        ("metric", "eggs", sample_plugin.SessionRatio, "plugin_contract_violation"),
        ("reader", "eggs", sample_plugin.NotAGenerator, "plugin_contract_violation"),
        ("reader", "eggs", 3, "plugin_contract_violation"),
        (
            "visualizer",
            "eggs",
            sample_plugin.EchoGenerator,
            "plugin_contract_violation",
        ),
        ("benchmark_suite", "eggs", 3, "plugin_contract_violation"),
        (
            "benchmark_suite",
            "eggs",
            sample_plugin.quick_check,
            "plugin_contract_violation",
        ),
        (
            "benchmark_suite",
            "bad_protocol",
            sample_plugin.bad_protocol,
            "plugin_contract_violation",
        ),
    ],
)
def test_contract_violations(kind: Any, name: str, obj: Any, code: str) -> None:
    with pytest.raises(PluginError) as info:
        register_plugin(kind, name, obj)
    assert info.value.code == code


def test_metric_direction_is_checked() -> None:
    class Sideways(sample_plugin.SessionRatio):
        name = "eggs"
        direction = "sideways"

    with pytest.raises(PluginError):
        register_plugin("metric", "eggs", Sideways)


def test_reader_factory_contract_checked_on_use() -> None:
    register_plugin("reader", "eggs", lambda: object())
    with pytest.raises(PluginError) as info:
        get_reader("eggs")
    assert info.value.code == "plugin_contract_violation"
    register_plugin("reader", "crash", sample_plugin.crashing_factory)
    with pytest.raises(PluginError) as info:
        get_reader("crash")
    assert info.value.code == "plugin_factory_failed"


def test_suffix_conflicts() -> None:
    from eduloggen.io.base import register_suffixes

    with pytest.raises(PluginError):
        register_suffixes("eggs", [".csv"])
    with pytest.raises(PluginError):
        register_suffixes("eggs", ["pipe"])
    register_plugin("reader", "pipe", sample_plugin.PipeReader)
    with pytest.raises(PluginError):
        register_suffixes("eggs", [".pipe"])


# --------------------------------------------------------------------------
# Entry-point discovery
# --------------------------------------------------------------------------


def test_discover_from_entry_points() -> None:
    points = [ep(kind, name, attr) for kind, (name, attr) in VALID.items()]
    result = discover_plugins(entry_points_override=reversed(points))
    assert [(i.kind, i.name) for i in result.loaded] == sorted(
        (kind, name) for kind, (name, _) in VALID.items()
    )
    assert all(i.source == "entry_point" for i in result.loaded)
    assert result.errors == ()
    again = discover_plugins(entry_points_override=points)
    assert again.loaded == ()


def test_discovery_reports_broken_plugins(caplog: pytest.LogCaptureFixture) -> None:
    points = [
        ep("generator", "echo", "EchoGenerator"),
        ep("generator", "not_a_generator", "NotAGenerator"),
        ep("generator", "missing", "DoesNotExist"),
        ep("generator", "markov", "EchoGenerator"),
        EntryPoint(
            name="x", value=f"{MODULE}:EchoGenerator", group="eduloggen.plugins.gizmo"
        ),
    ]
    with caplog.at_level(logging.WARNING, logger="eduloggen.plugins"):
        result = discover_plugins(entry_points_override=points)
    assert [i.name for i in result.loaded] == ["echo"]
    labels = [label for label, _ in result.errors]
    assert labels == [
        "eduloggen.plugins.generator:markov",
        "eduloggen.plugins.generator:missing",
        "eduloggen.plugins.generator:not_a_generator",
        "eduloggen.plugins.gizmo:x",
    ]
    assert (
        "plugin_load_failed"
        in dict(result.errors)["eduloggen.plugins.generator:missing"]
    )
    assert "skipping plugin" in caplog.text
    with pytest.raises(PluginError):
        discover_plugins(strict=True, entry_points_override=points[1:2])


def test_list_plugins() -> None:
    register_plugin("generator", "echo", sample_plugin.EchoGenerator)
    generators = list_plugins("generator")
    names = [i.name for i in generators]
    assert names == sorted(names)
    markov = next(i for i in generators if i.name == "markov")
    assert markov.source == "builtin"
    assert "probabilistic" in markov.tags
    assert next(i for i in generators if i.name == "echo").source == "api"
    assert {i.kind for i in list_plugins()} == set(KINDS)
    assert list_plugins("generator")[0].to_dict()["kind"] == "generator"


# --------------------------------------------------------------------------
# Installed package (real importlib.metadata discovery) and CLI
# --------------------------------------------------------------------------


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Fake-install a distribution 'eduloggen-echo' declaring entry points."""
    site = tmp_path / "site"
    package = site / "eduloggen_echo"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        f"from {MODULE} import EchoGenerator, NotAGenerator, SessionRatio\n"
    )
    dist = site / "eduloggen_echo-1.2.3.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: eduloggen-echo\nVersion: 1.2.3\n"
    )
    (dist / "entry_points.txt").write_text(
        "[eduloggen.plugins.generator]\n"
        "echo = eduloggen_echo:EchoGenerator\n"
        "not_a_generator = eduloggen_echo:NotAGenerator\n"
        "[eduloggen.plugins.metric]\n"
        "session_ratio = eduloggen_echo:SessionRatio\n"
    )
    monkeypatch.syspath_prepend(str(site))
    yield site
    sys.modules.pop("eduloggen_echo", None)


def test_installed_distribution_is_discovered(installed: Path) -> None:
    result = discover_plugins()
    echo = next(i for i in result.loaded if i.name == "echo")
    assert echo.distribution == "eduloggen-echo"
    assert echo.version == "1.2.3"
    assert echo.target == "eduloggen_echo:EchoGenerator"
    assert [label for label, _ in result.errors] == [
        "eduloggen.plugins.generator:not_a_generator"
    ]


def test_cli_with_installed_plugin(
    installed: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["plugins", "--quiet"])
    out = capsys.readouterr().out
    assert code == 1
    assert "  echo  [baseline, third_party; from eduloggen-echo 1.2.3]" in out
    assert (
        "session_ratio  [utility, higher is better; from eduloggen-echo 1.2.3]" in out
    )
    assert "errors:\n  eduloggen.plugins.generator:not_a_generator" in out

    demo, model = tmp_path / "demo", tmp_path / "model"
    assert main(["demo", "--output", str(demo), "--learners", "10", "--quiet"]) == 0
    assert main(["fit", "--input", str(demo), "--generator", "echo",
                 "--output", str(model), "--quiet"]) == 0  # fmt: skip
    import json

    manifest = json.loads((model / "run_manifest.json").read_text())
    external = {p["name"]: p for p in manifest["plugins"]["external"]}
    assert external["echo"]["version"] == "1.2.3"
