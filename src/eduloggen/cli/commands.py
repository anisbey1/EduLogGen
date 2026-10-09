"""Command handlers. Each maps parsed arguments onto the API façade.

Handlers return a process exit code and record inputs and outputs on the
:class:`Run` so :mod:`eduloggen.cli.main` can write ``run_manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
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
from eduloggen.analysis import activity_table_markdown, deadline_effects
from eduloggen.config import AppConfig, config_fingerprint, load_file
from eduloggen.core import ConfigError, RunContext
from eduloggen.generators import available_generators, get_generator, load_model
from eduloggen.io import read_annotations, write_corpus
from eduloggen.models import Dataset
from eduloggen.plugins import KINDS, list_plugins
from eduloggen.scenarios import ExperimentSettings, load_anomaly_specs, run_experiment
from eduloggen.utils import derive_seed
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
    plugin_errors: tuple[tuple[str, str], ...] = ()

    def manifest(self, exit_status: int) -> dict[str, Any]:
        """The ``run_manifest.json`` payload (SAD §29P)."""
        return {
            **self.context.to_dict(),
            "argv": self.argv,
            "config_fingerprint": config_fingerprint(self.config),
            "plugins": {
                "generators": available_generators(),
                "external": [
                    info.to_dict()
                    for info in list_plugins()
                    if info.source != "builtin"
                ],
            },
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
    files: dict[str, str | Mapping[str, Any]] = {
        "analysis.json": result.to_dict(),
        "analysis.md": markdown,
    }
    groups = (
        _read_groups(_path(args.groups, None, run, "--groups")) if args.groups else None
    )
    if args.detail:
        profiles = api.activity_profiles(dataset)
        temporal = api.temporal_profile(dataset, timezone=args.timezone)
        activities_md = "## Activities\n\n" + activity_table_markdown(profiles)
        temporal_md = temporal.to_markdown()
        temporal_data: dict[str, Any] = temporal.to_dict()
        if args.deadlines:
            effects = deadline_effects(
                dataset,
                [d.strip() for d in args.deadlines.split(",") if d.strip()],
                timezone=args.timezone,
            )
            temporal_data["deadlines"] = [e.to_dict() for e in effects]
            temporal_md += "\n| Deadline | Sessions/day before | Other days | Ratio |\n"
            temporal_md += "| --- | --- | --- | --- |\n" + "".join(
                f"| {e.deadline} | {e.window_per_day:.1f} | {e.other_per_day:.1f} | "
                f"{'-' if e.ratio is None else f'{e.ratio:.2f}'} |\n"
                for e in effects
            )
        files |= {
            "activities.json": {"activities": [p.to_dict() for p in profiles]},
            "activities.md": activities_md,
            "temporal.json": temporal_data,
            "temporal.md": temporal_md,
        }
        markdown += "\n" + activities_md + "\n" + temporal_md
    if args.by:
        strata = api.analyze_by(dataset, args.by, timezone=args.timezone, groups=groups)
        files |= {"strata.json": strata.to_dict(), "strata.md": strata.to_markdown()}
        markdown += "\n" + strata.to_markdown()
    if args.output:
        directory = _path(args.output, None, run, "--output")
        _write_bundle(directory, files, "analysis.json", run)
        _print(f"wrote {', '.join(sorted(files))} to {directory}")
    else:
        _print(markdown)
    return EXIT_OK


def _read_groups(path: Path) -> dict[str, str]:
    """Read a ``learner_id,group`` CSV."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not {"learner_id", "group"} <= set(reader.fieldnames or ()):
                raise ConfigError(
                    "groups file needs 'learner_id' and 'group' columns",
                    code="cli_invalid_argument",
                )
            return {row["learner_id"]: row["group"] for row in reader}
    except FileNotFoundError:
        raise ConfigError(
            "groups file not found",
            code="cli_missing_argument",
            context={"path": str(path)},
        ) from None


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
    """Sample a synthetic corpus from a saved model (optionally an experiment)."""
    model_path = _path(args.model, None, run, "--model")
    output = _path(args.output, None, run, "--output")
    model = load_model(model_path)
    run.inputs["model"] = model.fingerprint()
    if args.experiment and args.anomalies:
        raise ConfigError(
            "use --experiment (with an anomalies section) or --anomalies, not both",
            code="cli_invalid_argument",
        )
    if args.experiment:
        return _generate_experiment(args, run, model, output)
    synthetic = api.generate(
        model,
        config=run.config,
        n_sessions=args.n_sessions,
        id_strategy=args.id_strategy,
    )
    annotations = None
    dataset: Dataset = synthetic
    if args.anomalies:
        specs = load_anomaly_specs(_path(args.anomalies, None, run, "--anomalies"))
        injected = api.inject_anomalies(
            synthetic,
            specs,
            seed=derive_seed(synthetic.generation.seed, "anomalies") or 0,
        )
        dataset, annotations = injected.dataset, injected.annotations
        _print_injection(injected.report)
    target = write_corpus(
        dataset,
        output,
        format=args.format or run.config.io.output_format,
        annotations=annotations,
        force=run.force,
    )
    run.outputs["corpus"] = str(target)
    run.manifest_dir = target
    _print(
        f"generated {dataset.n_sessions} sessions ({dataset.n_events} events) "
        f"with seed {synthetic.generation.seed}; wrote corpus {target}"
    )
    return EXIT_OK


def _print_injection(report: Mapping[str, Any]) -> None:
    for name, info in report["anomalies"].items():
        _print(
            f"injected {name} ({info['category']}): {info['injected']} of "
            f"{info['requested']} sessions"
        )


def _generate_experiment(
    args: argparse.Namespace, run: Run, model: Any, output: Path
) -> int:
    settings = ExperimentSettings.from_file(
        _path(args.experiment, None, run, "--experiment")
    )
    seed = run.config.generation.seed
    if seed is None:
        raise ConfigError(
            "generation needs a seed: pass --seed or set generation.seed",
            code="generation_missing_seed",
        )
    result = run_experiment(
        get_generator(model.generator_id),
        model,
        settings,
        n_sessions=args.n_sessions or run.config.generation.n_sessions,
        seed=seed,
        id_strategy=args.id_strategy or run.config.generation.id_strategy,
    )
    target = write_corpus(
        result.dataset,
        output,
        format=args.format or run.config.io.output_format,
        annotations=result.annotations,
        force=run.force,
    )
    markdown = result.check.to_markdown()
    write_json(target / "manipulation_check.json", result.check.to_dict())
    (target / "manipulation_check.md").write_text(markdown, encoding="utf-8")
    run.inputs["controlled_model"] = result.model.fingerprint()
    run.outputs["corpus"] = str(target)
    run.manifest_dir = target
    _print(markdown)
    _print(
        f"generated {result.dataset.n_sessions} sessions "
        f"({result.dataset.n_events} events) with seed {seed}; wrote corpus {target}"
    )
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace, run: Run) -> int:
    """Score detector predictions against a corpus's ground truth."""
    corpus = _path(args.corpus, None, run, "--corpus")
    dataset = api.load_dataset(corpus)
    annotations = read_annotations(corpus)
    if annotations is None:
        raise ConfigError(
            "the corpus has no annotations; generate it with --anomalies",
            code="cli_missing_annotations",
            context={"path": str(corpus)},
        )
    run.inputs["corpus"] = dataset.fingerprint()
    predictions = _read_predictions(_path(args.predictions, None, run, "--predictions"))
    report = api.evaluate_detection(
        dataset, annotations, predictions, level=args.level, threshold=args.threshold
    )
    markdown = report.to_markdown()
    _print(markdown)
    if args.output:
        _write_bundle(
            _path(args.output, None, run, "--output"),
            {"evaluation.json": report.to_dict(), "evaluation.md": markdown},
            "evaluation.json",
            run,
        )
    return EXIT_OK


def _read_predictions(path: Path) -> set[str] | dict[str, float]:
    """Read ``id`` plus optional ``score`` or ``flag`` columns from CSV."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            if "id" not in columns:
                raise ConfigError(
                    "predictions need an 'id' column (plus optional 'score' or 'flag')",
                    code="cli_invalid_argument",
                )
            rows = list(reader)
    except FileNotFoundError:
        raise ConfigError(
            "predictions file not found",
            code="cli_missing_argument",
            context={"path": str(path)},
        ) from None
    try:
        if "score" in columns:
            return {row["id"]: float(row["score"]) for row in rows}
        if "flag" in columns:
            truthy = {"1", "true", "yes", "y", "t"}
            return {row["id"] for row in rows if row["flag"].strip().lower() in truthy}
    except (TypeError, ValueError):
        raise ConfigError(
            "prediction scores must be numbers", code="cli_invalid_argument"
        ) from None
    return {row["id"] for row in rows}


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
    files: dict[str, str | Mapping[str, Any]] = {
        "report.json": report.to_dict(),
        "report.md": markdown,
    }
    if args.detailed:
        groups = (
            _read_groups(_path(args.groups, None, run, "--groups"))
            if args.groups
            else None
        )
        detailed = api.compare_detailed(
            real, synthetic, by=args.by, timezone=args.timezone, groups=groups
        )
        files |= {
            "detailed.json": detailed.to_dict(),
            "detailed.md": detailed.to_markdown(),
        }
        markdown += "\n" + detailed.to_markdown()
    elif args.by:
        raise ConfigError("--by needs --detailed", code="cli_invalid_argument")
    _print(markdown)
    if args.output:
        _write_bundle(
            _path(args.output, None, run, "--output"), files, "report.json", run
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


def cmd_plot(args: argparse.Namespace, run: Run) -> int:
    """Render figures for corpora and/or saved reports."""
    from eduloggen.benchmark import BenchmarkReport
    from eduloggen.validation import ValidationReport
    from eduloggen.visualization import (
        plot_benchmark,
        plot_datasets,
        plot_validation_report,
        save_figure,
    )

    if not (args.real or args.validation or args.benchmark):
        raise ConfigError(
            "plot needs --real, --validation, or --benchmark",
            code="cli_missing_argument",
            context={"argument": "--real"},
        )
    section = run.config.visualization
    output = _path(args.output, section.output_dir, run, "--output")
    formats = (
        [f.strip() for f in args.format.split(",") if f.strip()]
        if args.format
        else list(section.formats)
    )
    written: list[Path] = []
    if args.real:
        real = _load(_path(args.real, None, run, "--real"), run, "real")
        synthetic = (
            _load(_path(args.synthetic, None, run, "--synthetic"), run, "synthetic")
            if args.synthetic
            else None
        )
        plots = [p.strip() for p in args.plots.split(",")] if args.plots else None
        for paths in plot_datasets(
            real, synthetic, output, plots=plots, formats=formats, dpi=section.dpi
        ).values():
            written.extend(paths)
    reports: list[tuple[str, Any]] = []
    if args.validation:
        data = load_file(_path(args.validation, None, run, "--validation"))
        reports.append(
            ("validation", plot_validation_report(ValidationReport.from_dict(data)))
        )
    if args.benchmark:
        data = load_file(_path(args.benchmark, None, run, "--benchmark"))
        reports.append(("benchmark", plot_benchmark(BenchmarkReport.from_dict(data))))
    for name, figure in reports:
        written.extend(
            save_figure(figure, output / name, formats=formats, dpi=section.dpi)
        )
    run.outputs["figures"] = str(output)
    _print("\n".join(f"wrote {path}" for path in written))
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
    """List generators, metrics, readers, plots, and benchmark suites."""
    lines: list[str] = []
    for kind in KINDS:
        lines.append(f"{kind}s:")
        for info in list_plugins(kind):
            detail = []
            if info.tags:
                detail.append(", ".join(info.tags))
            if kind == "metric":
                metric = get_metric(info.name)
                better = "lower" if metric.direction == "lower_better" else "higher"
                detail.append(f"{metric.category}, {better} is better")
            if info.source != "builtin":
                origin = info.distribution or info.source
                detail.append(
                    f"from {origin}" + (f" {info.version}" if info.version else "")
                )
            lines.append(
                f"  {info.name}" + (f"  [{'; '.join(detail)}]" if detail else "")
            )
    if run.plugin_errors:
        lines.append("errors:")
        lines += [f"  {label}: {message}" for label, message in run.plugin_errors]
    _print("\n".join(lines))
    return EXIT_FAILED if run.plugin_errors else EXIT_OK


COMMANDS: Final[dict[str, Handler]] = {
    "ingest": cmd_ingest,
    "sessionize": cmd_sessionize,
    "analyze": cmd_analyze,
    "fit": cmd_fit,
    "generate": cmd_generate,
    "evaluate": cmd_evaluate,
    "validate": cmd_validate,
    "benchmark": cmd_benchmark,
    "demo": cmd_demo,
    "plot": cmd_plot,
    "info": cmd_info,
    "plugins": cmd_plugins,
}
