"""End-to-end tests of the ``eduloggen`` command line."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from eduloggen import __version__
from eduloggen.cli import build_parser, main

from .conftest import EMAIL


def sh(capsys: pytest.CaptureFixture[str], command: str, **paths: Path) -> int:
    """Run a command line written with ``{name}`` path placeholders."""
    values = {name: shlex.quote(str(path)) for name, path in paths.items()}
    return run(capsys, *shlex.split(command.format(**values)))[0]


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = (
        main([*argv, "--quiet"])
        if argv and not argv[0].startswith("-")
        else main(list(argv))
    )
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def pipeline(export: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Run ingest -> sessionize -> fit -> generate; return the working dir."""
    d = export
    commands = [
        "ingest --input {d}/lms.csv --mapping {d}/mapping.yaml --output {d}/corpus",
        "sessionize --input {d}/corpus --output {d}/sessions --idle-timeout 1800",
        "fit --input {d}/sessions --generator semi_markov --set order=2 "
        "--set on_insufficient_data=backoff --output {d}/model",
        "generate --model {d}/model --n-sessions 30 --seed 3 --output {d}/synthetic",
    ]
    for command in commands:
        assert sh(capsys, command, d=d) == 0, command
    return d


def test_pipeline_outputs(pipeline: Path) -> None:
    corpus = pipeline / "corpus"
    assert {p.name for p in corpus.iterdir()} == {
        "events.csv",
        "manifest.json",
        "mapping.used.json",
        "quality_report.json",
        "run_manifest.json",
    }
    text = (corpus / "events.csv").read_text()
    assert EMAIL.format(0) not in text
    assert "test-salt" not in (corpus / "mapping.used.json").read_text()

    manifest = json.loads((pipeline / "model" / "run_manifest.json").read_text())
    assert manifest["argv"][0] == "fit"
    assert manifest["exit_status"] == 0
    assert manifest["inputs"]["corpus"].startswith("sha256:")
    assert manifest["outputs"]["model"].endswith("model")
    assert manifest["config_fingerprint"].startswith("sha256:")
    assert "semi_markov" in manifest["plugins"]["generators"]

    synthetic = json.loads((pipeline / "synthetic" / "manifest.json").read_text())
    assert synthetic["synthetic"] is True
    assert synthetic["generation"]["seed"] == 3
    assert synthetic["counts"]["sessions"] == 30


def test_validate_exit_codes(
    pipeline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real, synthetic = str(pipeline / "sessions"), str(pipeline / "synthetic")
    code, out, _ = run(
        capsys,
        *f"validate --real {real} --synthetic {synthetic}".split(),
        *["--threshold", "event_type_tvd=0.5", "--output", str(pipeline / "report")],
    )
    assert code == 0
    assert "# Validation report: **PASS**" in out
    report = json.loads((pipeline / "report" / "report.json").read_text())
    assert report["status"] == "pass"
    assert (pipeline / "report" / "run_manifest.json").exists()

    code, out, _ = run(
        capsys,
        *f"validate --real {real} --synthetic {synthetic}".split(),
        *["--metrics", "event_type_tvd,bigram_tvd", "--threshold", "event_type_tvd=0"],
    )
    assert code == 1
    assert "FAIL" in out
    assert "activity_jsd" not in out


def test_validate_thresholds_file(
    pipeline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (pipeline / "limits.yaml").write_text("event_type_tvd: 0.0\n")
    command = (
        "validate --real {d}/sessions --synthetic {d}/synthetic "
        "--thresholds {d}/limits.yaml --threshold event_type_tvd=1"
    )
    assert sh(capsys, command, d=pipeline) == 0


def test_analyze(pipeline: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "analyze", "--input", str(pipeline / "corpus"))
    assert code == 0
    assert out.startswith("# Analysis of `lms`")
    command = "analyze --input {d}/sessions --ngram-order 3 --output {d}/analysis"
    assert sh(capsys, command, d=pipeline) == 0
    data = json.loads((pipeline / "analysis" / "analysis.json").read_text())
    assert data["ngram_order"] == 3


def test_overwrite_requires_force(
    pipeline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = (
        "fit",
        "--input",
        str(pipeline / "sessions"),
        "--output",
        str(pipeline / "model"),
    )
    code, _, err = run(capsys, *args)
    assert code == 2
    assert "export_exists" in err
    assert str(pipeline / "model") in err
    assert run(capsys, *args, "--force")[0] == 0


def test_global_options_before_command(
    pipeline: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command = (
        "--seed 9 --quiet generate --model {d}/model --n-sessions 2 --output {d}/s9"
    )
    assert sh(capsys, command, d=pipeline) == 0
    manifest = json.loads((pipeline / "s9" / "manifest.json").read_text())
    assert manifest["generation"]["seed"] == 9


def test_config_file_paths(export: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_dir = export / "configs"
    config_dir.mkdir()
    (config_dir / "exp.yaml").write_text(
        "io:\n  input: ../lms.csv\n  mapping: ../mapping.yaml\n"
        "  output: ../from-config\n  output_format: jsonl\n"
    )
    code, out, _ = run(capsys, "ingest", "--config", str(config_dir / "exp.yaml"))
    assert code == 0
    assert (export / "from-config" / "events.jsonl").exists()
    assert "kept" in out


def test_ingest_quality_failure(
    export: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (export / "bad.csv").write_text(
        "log_id,user_id,time,resource,action\n1,,2026-01-01 00:00:00,r,view\n"
    )
    command = "ingest --input {d}/bad.csv --mapping {d}/mapping.yaml --output {d}/never"
    code, out, _ = run(capsys, *shlex.split(command.format(d=export)))
    assert code == 1
    assert "error: no row produced a valid event" in out
    assert not (export / "never").exists()


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["ingest"], "cli_missing_argument"),
        (["fit", "--input", "nowhere", "--output", "m"], "io_not_found"),
        (["info", "--config", "missing.yaml"], "config_not_found"),
    ],
)
def test_errors_exit_2(
    argv: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    code, _, err = run(capsys, *argv)
    assert code == 2
    assert message in err


def test_bad_key_values(pipeline: Path, capsys: pytest.CaptureFixture[str]) -> None:
    sessions = str(pipeline / "sessions")
    code, _, err = run(
        capsys, "fit", "--input", sessions, "--output", "x", "--set", "order"
    )
    assert code == 2
    assert "KEY=VALUE" in err
    command = f"validate --real {sessions} --synthetic {sessions}"
    code, _, err = run(capsys, *command.split(), "--threshold", "event_type_tvd=low")
    assert code == 2
    assert "must be a number" in err


def test_usage_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["frobnicate"]) == 2
    assert main(["fit"]) == 2
    assert main(["--help"]) == 0
    assert "examples:" in capsys.readouterr().out
    assert main(["fit", "--help"]) == 0
    assert "example:" in capsys.readouterr().out


def test_version_and_no_command(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out
    assert main([]) == 0
    assert "<command>" in capsys.readouterr().out
    assert build_parser().prog == "eduloggen"


def test_info_and_plugins(pipeline: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, "info", "--input", str(pipeline / "corpus"))
    assert code == 0
    assert f"eduloggen {__version__}" in out
    assert "sessions: not sessionized" in out
    code, out, _ = run(capsys, "info", "--input", str(pipeline / "synthetic"))
    assert "(SyntheticDataset)" in out
    code, out, _ = run(capsys, "plugins")
    assert "semi_markov  [probabilistic, sequence, supports_timing]" in out
    assert "nn_distance_p05  [privacy, higher is better]" in out
    assert "benchmark_suites:" in out
    assert "  session_fidelity_v1" in out


def test_logging_options(pipeline: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main(["plugins", "--json-logs", "-vv", "--run-id", "run-42"])
    capsys.readouterr()
    logging.getLogger("eduloggen.test").info("hello")
    line = capsys.readouterr().err.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["run_id"] == "run-42"
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"

    main(["plugins", "-v"])
    capsys.readouterr()
    logging.getLogger("eduloggen.test").info("plain")
    assert "INFO eduloggen.test: plain" in capsys.readouterr().err

    main(["plugins", "--quiet"])
    capsys.readouterr()
    logging.getLogger("eduloggen.test").warning("hidden")
    assert capsys.readouterr().err == ""


def test_python_dash_m() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    result = subprocess.run(
        [sys.executable, "-m", "eduloggen", "--version"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(src)},
        check=False,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout
