"""Command handlers. Each maps parsed arguments onto the API façade.

Handlers return a process exit code and record inputs and outputs on the
:class:`Run` so :mod:`eduloggen.cli.main` can write ``run_manifest.json``.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import yaml

from eduloggen import api
from eduloggen.__version__ import __version__
from eduloggen.config import AppConfig, config_fingerprint, load_file
from eduloggen.core import ConfigError, RunContext
from eduloggen.generators import available_generators, get_generator, load_model
from eduloggen.io import write_corpus
from eduloggen.models import Dataset
from eduloggen.utils.fs import atomic_directory, write_json
from eduloggen.validation import available_metrics, get_metric

__all__ = ["COMMANDS", "Run"]

logger = logging.getLogger("eduloggen.cli")

EXIT_OK: Final = 0
EXIT_FAILED: Final = 1


@dataclass
class Run:
    """Bookkeeping for one CLI invocation."""

    context: RunContext
    config: AppConfig
    argv: list[str]
    force: bool = False
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    manifest_dir: Path | None = None

    def manifest(self, exit_status: int) -> dict[str, Any]:
        """The ``run_manifest.json`` payload (SAD §29P)."""
        return {
            **self.context.to_dict(),
            "argv": self.argv,
            "config_fingerprint": config_fingerprint(self.config),
            "plugins": {"generators": available_generators()},
            "inputs": self.inputs,
            "outputs": self.outputs,
            "exit_status": exit_status,
        }


Handler = Callable[[argparse.Namespace, Run], int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path(cli_value: str | None, config_value: str | None, run: Run, what: str) -> Path:
    """CLI paths are relative to the working directory, config paths to the file."""
    if cli_value:
        return Path(cli_value).expanduser().absolute()
    if config_value:
        return run.config.resolve_path(config_value)
    raise ConfigError(
        f"missing {what}: pass it as an option or set it in the config",
        code="cli_missing_argument",
        context={"argument": what},
    )


def _load(path: Path, run: Run, label: str) -> Dataset:
    dataset = api.load_dataset(path)
    run.inputs[label] = dataset.fingerprint()
    if dataset.sessions is None:
        logger.info("%s is not sessionized; sessionizing with config settings", label)
        dataset = api.sessionize(dataset, config=run.config)
    return dataset


def _key_values(pairs: Iterable[str], option: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = pair.partition("=")
        if not sep or not key:
            raise ConfigError(
                f"{option} expects KEY=VALUE, got {pair!r}",
                code="cli_invalid_argument",
            )
        result[key.strip()] = yaml.safe_load(raw)
    return result


def _write_bundle(
    directory: Path, files: Mapping[str, str | Mapping[str, Any]], marker: str, run: Run
) -> None:
    with atomic_directory(directory, marker=marker, force=run.force) as staging:
        for name, content in files.items():
            if isinstance(content, str):
                (staging / name).write_text(content, encoding="utf-8")
            else:
                write_json(staging / name, content)
    run.outputs["directory"] = str(directory)
    run.manifest_dir = directory


def _print(text: str) -> None:
    sys.stdout.write(text if text.endswith("\n") else text + "\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, run: Run) -> int:
    """Read a source log, check quality, and write a corpus."""
    io = run.config.io
    source = _path(args.input, io.input, run, "--input")
    mapping = _path(args.mapping, io.mapping, run, "--mapping")
    output = _path(args.output, io.output, run, "--output")
    result = api.ingest(
        source, mapping, config=run.config, format=args.format, strict=args.strict
    )
    report = result.report
    run.inputs["source"] = str(source)
    _print(
        f"read {report.rows_read} rows, kept {report.rows_kept}, "
        f"dropped {report.rows_dropped} ({report.status})"
    )
    for warning in report.warnings:
        _print(f"warning: {warning}")
    if report.status == "failed":
        for error in report.errors:
            _print(f"error: {error}")
        return EXIT_FAILED
    target = write_corpus(
        result.dataset,
        output,
        format=args.output_format or io.output_format,
        quality_report=report,
        mapping=result.mapping,
        force=run.force,
    )
    run.outputs["corpus"] = str(target)
    run.manifest_dir = target
    _print(f"wrote corpus {target}")
    return EXIT_OK


def cmd_sessionize(args: argparse.Namespace, run: Run) -> int:
    """Group a corpus's events into sessions."""
    source = _path(args.input, None, run, "--input")
    output = _path(args.output, None, run, "--output")
    dataset = api.load_dataset(source)
    run.inputs["corpus"] = dataset.fingerprint()
    sessionized = api.sessionize(dataset, config=run.config)
    target = write_corpus(
        sessionized, output, format=run.config.io.output_format, force=run.force
    )
    run.outputs["corpus"] = str(target)
    run.manifest_dir = target
    _print(f"built {sessionized.n_sessions} sessions; wrote corpus {target}")
    return EXIT_OK


def cmd_analyze(args: argparse.Namespace, run: Run) -> int:
    """Compute statistics; print or write them."""
    dataset = _load(_path(args.input, None, run, "--input"), run, "corpus")
    result = api.analyze(dataset, config=run.config)
    markdown = result.to_markdown()
    if args.output:
        directory = _path(args.output, None, run, "--output")
        _write_bundle(
            directory,
            {"analysis.json": result.to_dict(), "analysis.md": markdown},
            "analysis.json",
            run,
        )
        _print(f"wrote analysis to {directory}")
    else:
        _print(markdown)
    return EXIT_OK


def cmd_fit(args: argparse.Namespace, run: Run) -> int:
    """Fit a generator and save the model."""
    dataset = _load(_path(args.input, None, run, "--input"), run, "corpus")
    output = _path(args.output, None, run, "--output")
    model = api.fit_generator(
        args.generator,
        dataset,
        config=run.config,
        hyperparameters=_key_values(args.set or (), "--set"),
    )
    generator = get_generator(model.generator_id)
    target = generator.save(model, output, force=run.force)
    run.outputs["model"] = str(target)
    run.manifest_dir = target
    info = generator.describe(model)
    _print(
        f"fitted {info['generator']} on {info['training_sessions']} sessions "
        f"({info['vocabulary_size']} tokens); wrote model {target}"
    )
    return EXIT_OK


def cmd_generate(args: argparse.Namespace, run: Run) -> int:
    """Sample a synthetic corpus from a saved model."""
    model_path = _path(args.model, None, run, "--model")
    output = _path(args.output, None, run, "--output")
    model = load_model(model_path)
    run.inputs["model"] = model.fingerprint()
    synthetic = api.generate(
        model,
        config=run.config,
        n_sessions=args.n_sessions,
        id_strategy=args.id_strategy,
    )
    target = write_corpus(
        synthetic,
        output,
        format=args.format or run.config.io.output_format,
        force=run.force,
    )
    run.outputs["corpus"] = str(target)
    run.manifest_dir = target
    _print(
        f"generated {synthetic.n_sessions} sessions ({synthetic.n_events} events) "
        f"with seed {synthetic.generation.seed}; wrote corpus {target}"
    )
    return EXIT_OK


def cmd_validate(args: argparse.Namespace, run: Run) -> int:
    """Compare corpora; exit 1 if any threshold fails."""
    real = _load(_path(args.real, None, run, "--real"), run, "real")
    synthetic = _load(_path(args.synthetic, None, run, "--synthetic"), run, "synthetic")
    thresholds: dict[str, Any] = {}
    if args.thresholds:
        thresholds.update(load_file(_path(args.thresholds, None, run, "--thresholds")))
    thresholds.update(_key_values(args.threshold or (), "--threshold"))
    for name, value in thresholds.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(
                f"threshold for {name} must be a number",
                code="cli_invalid_argument",
            )
    metrics = (
        [m.strip() for m in args.metrics.split(",") if m.strip()]
        if args.metrics
        else None
    )
    report = api.validate(
        real,
        synthetic,
        config=run.config,
        metrics=metrics,
        thresholds={k: float(v) for k, v in thresholds.items()},
        real_split=args.real_split,
    )
    markdown = report.to_markdown()
    _print(markdown)
    if args.output:
        _write_bundle(
            _path(args.output, None, run, "--output"),
            {"report.json": report.to_dict(), "report.md": markdown},
            "report.json",
            run,
        )
    return EXIT_OK if report.passed else EXIT_FAILED


def cmd_benchmark(args: argparse.Namespace, run: Run) -> int:
    """Compare generators under a protocol."""
    dataset = _load(_path(args.input, None, run, "--input"), run, "corpus")
    generators = (
        [g.strip() for g in args.generators.split(",") if g.strip()]
        if args.generators
        else None
    )
    report = api.run_benchmark(
        dataset,
        config=run.config,
        generators=generators,
        protocol=args.protocol,
        repeats=args.seeds,
        n_sessions=args.n_sessions,
    )
    markdown = report.to_markdown()
    _print(markdown)
    if args.output:
        _write_bundle(
            _path(args.output, None, run, "--output"),
            {"benchmark.json": report.to_dict(), "benchmark.md": markdown},
            "benchmark.json",
            run,
        )
    return EXIT_FAILED if any(r.error for r in report.results) else EXIT_OK


def cmd_demo(args: argparse.Namespace, run: Run) -> int:
    """Write the synthetic demo course corpus."""
    output = _path(args.output, None, run, "--output")
    dataset = api.demo_dataset(args.learners, seed=run.config.generation.seed or 0)
    target = write_corpus(
        dataset, output, format=run.config.io.output_format, force=run.force
    )
    run.outputs["corpus"] = str(target)
    run.manifest_dir = target
    _print(
        f"wrote demo corpus {target}: {dataset.n_events} events from "
        f"{len(dataset.learner_ids)} synthetic learners"
    )
    return EXIT_OK


def cmd_info(args: argparse.Namespace, run: Run) -> int:
    """Show version, environment, plugins, and optionally a corpus summary."""
    lines = [
        f"eduloggen {__version__}",
        f"python {platform.python_version()} "
        f"({platform.system()} {platform.machine()})",
        f"generators: {', '.join(available_generators())}",
        f"metrics: {', '.join(available_metrics())}",
    ]
    if args.input:
        dataset = api.load_dataset(_path(args.input, None, run, "--input"))
        lines += [
            f"corpus: {dataset.dataset_id} ({type(dataset).__name__})",
            f"  events: {dataset.n_events}",
            "  sessions: "
            + (
                str(dataset.n_sessions) if dataset.is_sessionized else "not sessionized"
            ),
            f"  learners: {len(dataset.learner_ids)}",
            f"  fingerprint: {dataset.fingerprint()}",
        ]
    _print("\n".join(lines))
    return EXIT_OK


def cmd_plugins(args: argparse.Namespace, run: Run) -> int:
    """List registered generators and metrics."""
    lines = ["generators:"]
    for name in available_generators():
        tags = ", ".join(sorted(get_generator(name).tags))
        lines.append(f"  {name}" + (f"  [{tags}]" if tags else ""))
    lines.append("metrics:")
    for name in available_metrics():
        metric = get_metric(name)
        better = "lower" if metric.direction == "lower_better" else "higher"
        lines.append(f"  {name}  ({metric.category}, {better} is better)")
    _print("\n".join(lines))
    return EXIT_OK


COMMANDS: Final[dict[str, Handler]] = {
    "ingest": cmd_ingest,
    "sessionize": cmd_sessionize,
    "analyze": cmd_analyze,
    "fit": cmd_fit,
    "generate": cmd_generate,
    "validate": cmd_validate,
    "benchmark": cmd_benchmark,
    "demo": cmd_demo,
    "info": cmd_info,
    "plugins": cmd_plugins,
}
