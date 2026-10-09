"""Argument parsing, dispatch, and exit codes (PRD §14, SAD §19, §29F).

Exit codes: ``0`` success or validation pass, ``1`` validation thresholds
failed or ingest quality failed, ``2`` usage or runtime error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Any, Final

from eduloggen.__version__ import __version__
from eduloggen.cli.commands import COMMANDS, Run
from eduloggen.cli.logs import configure_logging, level_for
from eduloggen.config import resolve_config
from eduloggen.core import EduLogGenError, RunContext
from eduloggen.plugins import discover_plugins
from eduloggen.utils.fs import write_json

__all__ = ["build_parser", "main"]

EXIT_ERROR: Final = 2

logger = logging.getLogger("eduloggen.cli")

_EPILOG = """\
examples:
  eduloggen ingest --input events.csv --mapping mapping.yaml --output corpus/
  eduloggen fit --input corpus/ --generator semi_markov --set order=2 --output model/
  eduloggen generate --model model/ --n-sessions 1000 --seed 42 --output synthetic/
  eduloggen validate --real corpus/ --synthetic synthetic/ \
      --threshold event_type_tvd=0.1
  eduloggen --config experiment.yaml fit --input corpus/ --output model/
  eduloggen demo --output demo/ && eduloggen benchmark --input demo/

exit codes: 0 success, 1 validation or quality failure, 2 usage or runtime error
"""


def _common() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    group = common.add_argument_group("global options")
    group.add_argument(
        "--config",
        default=argparse.SUPPRESS,
        help="YAML/TOML/JSON configuration file",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="global seed (generation.seed)",
    )
    group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=argparse.SUPPRESS,
        help="more logging (-v info, -vv debug)",
    )
    group.add_argument(
        "--quiet", action="store_true", default=argparse.SUPPRESS, help="errors only"
    )
    group.add_argument(
        "--json-logs",
        action="store_true",
        default=argparse.SUPPRESS,
        help="structured JSON logs on stderr",
    )
    group.add_argument(
        "--force",
        action="store_true",
        default=argparse.SUPPRESS,
        help="replace existing outputs",
    )
    group.add_argument(
        "--run-id",
        default=argparse.SUPPRESS,
        help="run identifier for logs and manifests",
    )
    return common


def build_parser() -> argparse.ArgumentParser:
    """Build the ``eduloggen`` argument parser."""
    common = _common()
    parser = argparse.ArgumentParser(
        prog="eduloggen",
        description=(
            "EduLogGen: analyze, generate, validate, and benchmark "
            "synthetic educational interaction logs."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help_text: str, example: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            help=help_text,
            description=help_text,
            epilog=f"example:\n  {example}",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            parents=[common],
        )

    p = add(
        "ingest",
        "read a source log into a validated corpus",
        "eduloggen ingest --input events.csv --mapping mapping.yaml --output corpus/",
    )
    p.add_argument("--input", help="source file (default: io.input)")
    p.add_argument("--mapping", help="field mapping file (default: io.mapping)")
    p.add_argument("--output", help="corpus directory (default: io.output)")
    p.add_argument(
        "--format", help="auto (default), csv, tsv, jsonl, parquet, or a plugin reader"
    )
    p.add_argument("--output-format", choices=["csv", "tsv", "jsonl", "parquet"])
    p.add_argument(
        "--strict", action="store_true", default=None, help="stop at the first bad row"
    )

    p = add(
        "sessionize",
        "group a corpus's events into sessions",
        "eduloggen sessionize --input corpus/ --strategy idle_timeout --output out/",
    )
    p.add_argument("--input", required=True, help="corpus directory")
    p.add_argument("--output", required=True, help="output corpus directory")
    p.add_argument("--strategy", choices=["explicit", "idle_timeout", "composite"])
    p.add_argument("--idle-timeout", type=float, help="inactivity gap in seconds")
    p.add_argument("--tokenization", choices=["event_type", "activity_id"])

    p = add(
        "analyze",
        "compute descriptive and transition statistics",
        "eduloggen analyze --input corpus/ --output analysis/",
    )
    p.add_argument("--input", required=True, help="corpus directory")
    p.add_argument("--output", help="write analysis.json and analysis.md here")
    p.add_argument("--ngram-order", type=int, help="longest n-gram to count")
    p.add_argument(
        "--detail",
        action="store_true",
        help="add per-activity and temporal reports (activities.*, temporal.*)",
    )
    p.add_argument(
        "--by",
        help="also break statistics down by course, week, weekday, hour, "
        "learner_group (needs --groups), or metadata:<key> (strata.*)",
    )
    p.add_argument("--groups", help="CSV with learner_id,group for --by learner_group")
    p.add_argument(
        "--timezone",
        default="UTC",
        help="IANA zone for hours and weekdays (default UTC)",
    )
    p.add_argument(
        "--deadlines",
        help="comma-separated dates for the deadline effect (with --detail)",
    )

    p = add(
        "fit",
        "fit a generator and save the model",
        "eduloggen fit --input corpus/ --generator markov --set order=2 --output m/",
    )
    p.add_argument("--input", required=True, help="corpus directory")
    p.add_argument("--output", required=True, help="model directory")
    p.add_argument("--generator", help="registered generator (default: generator.name)")
    p.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="hyperparameter override"
    )

    p = add(
        "generate",
        "sample a synthetic corpus from a model",
        "eduloggen generate --model model/ --n-sessions 500 --seed 7 --output out/",
    )
    p.add_argument("--model", required=True, help="model directory")
    p.add_argument("--output", required=True, help="synthetic corpus directory")
    p.add_argument("--n-sessions", type=int, help="number of sessions")
    p.add_argument("--id-strategy", choices=["remap", "preserve"])
    p.add_argument("--format", choices=["csv", "tsv", "jsonl", "parquet"])
    p.add_argument(
        "--anomalies",
        help="YAML/TOML/JSON with an 'anomalies' list to inject; writes "
        "annotations.csv (ids are always remapped)",
    )
    p.add_argument(
        "--experiment",
        help="YAML/TOML/JSON with optional 'controls', 'calendar', and 'anomalies' "
        "sections; writes manipulation_check.json/.md",
    )

    p = add(
        "evaluate",
        "score anomaly-detector predictions against a corpus's annotations",
        "eduloggen evaluate --corpus synthetic/ --predictions preds.csv",
    )
    p.add_argument("--corpus", required=True, help="corpus with annotations.csv")
    p.add_argument(
        "--predictions",
        required=True,
        help="CSV with an 'id' column and optional 'score' or 'flag' column",
    )
    p.add_argument(
        "--level", choices=["session", "event", "learner"], default="session"
    )
    p.add_argument(
        "--threshold", type=float, default=0.5, help="score cut-off (default 0.5)"
    )
    p.add_argument("--output", help="write evaluation.json and evaluation.md here")

    p = add(
        "validate",
        "compare a real and a synthetic corpus",
        "eduloggen validate --real corpus/ --synthetic synthetic/ --output report/",
    )
    p.add_argument("--real", required=True, help="real corpus directory")
    p.add_argument("--synthetic", required=True, help="synthetic corpus directory")
    p.add_argument("--metrics", help="comma-separated metric names")
    p.add_argument("--thresholds", help="YAML/TOML/JSON file of metric thresholds")
    p.add_argument(
        "--threshold", action="append", metavar="METRIC=VALUE", help="one threshold"
    )
    p.add_argument("--real-split", default="full", help="label of the real reference")
    p.add_argument(
        "--detailed",
        action="store_true",
        help="add a ranked report of where synthetic differs (detailed.*)",
    )
    p.add_argument("--by", help="with --detailed: compare per group (see analyze --by)")
    p.add_argument("--groups", help="CSV with learner_id,group for --by learner_group")
    p.add_argument("--timezone", default="UTC", help="IANA zone for hours and weekdays")
    p.add_argument("--output", help="write report.json and report.md here")

    p = add(
        "benchmark",
        "compare generators under a benchmark protocol",
        "eduloggen benchmark --input corpus/ --generators markov,semi_markov",
    )
    p.add_argument("--input", required=True, help="corpus directory")
    p.add_argument(
        "--generators", help="comma-separated (default: benchmark.generators)"
    )
    p.add_argument("--protocol", help="protocol name (default: benchmark.protocol)")
    p.add_argument("--seeds", type=int, help="repeats per generator")
    p.add_argument("--n-sessions", type=int, help="sessions per sample")
    p.add_argument("--output", help="write benchmark.json and benchmark.md here")

    p = add(
        "demo",
        "write a fully synthetic demo course corpus",
        "eduloggen demo --output demo/ --learners 100",
    )
    p.add_argument("--output", required=True, help="corpus directory")
    p.add_argument("--learners", type=int, default=60, help="number of learners")

    p = add(
        "plot",
        "render figures (requires the viz extra)",
        "eduloggen plot --real corpus/ --synthetic synthetic/ --output figures/",
    )
    p.add_argument("--real", help="real corpus directory")
    p.add_argument("--synthetic", help="synthetic corpus to overlay")
    p.add_argument(
        "--plots",
        help="comma-separated: event_frequencies, activity_frequencies, "
        "session_lengths, session_durations, interevent_times, transitions, "
        "transition_graph, sankey, activity_heatmap, timeline (default: all)",
    )
    p.add_argument("--validation", help="report.json from validate")
    p.add_argument("--benchmark", help="benchmark.json from benchmark")
    p.add_argument("--format", help="comma-separated png, svg, pdf")
    p.add_argument(
        "--output", help="figure directory (default: visualization.output_dir)"
    )

    p = add("info", "show version, environment, and plugins", "eduloggen info")
    p.add_argument("--input", help="also summarize this corpus")

    add(
        "plugins",
        "list built-in and installed extensions (exit 1 if a plugin failed)",
        "eduloggen plugins",
    )
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generation.seed": getattr(args, "seed", None),
        "sessionization.strategy": getattr(args, "strategy", None),
        "sessionization.idle_timeout_s": getattr(args, "idle_timeout", None),
        "sessionization.tokenization": getattr(args, "tokenization", None),
        "analysis.ngram_order": getattr(args, "ngram_order", None),
        "generation.n_sessions": (
            None if args.command == "benchmark" else getattr(args, "n_sessions", None)
        ),
        "generation.id_strategy": getattr(args, "id_strategy", None),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the EduLogGen command-line interface.

    Args:
        argv: Arguments; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(arguments)
    except SystemExit as exc:
        return int(exc.code or 0)
    if args.command is None:
        parser.print_help()
        return 0

    try:
        config = resolve_config(
            getattr(args, "config", None), overrides=_overrides(args)
        )
        run_id = getattr(args, "run_id", None)
        context = (
            RunContext(
                run_id=run_id,
                seed=config.generation.seed,
                started_at=datetime.now(UTC),
            )
            if run_id
            else RunContext.create(seed=config.generation.seed)
        )
        configure_logging(
            level_for(
                getattr(args, "verbose", 0),
                getattr(args, "quiet", False),
                config.logging.level,
            ),
            json_logs=getattr(args, "json_logs", False) or config.logging.json,
            run_id=context.run_id,
        )
        discovery = discover_plugins()
        run = Run(
            context=context,
            config=config,
            argv=arguments,
            force=getattr(args, "force", False),
            plugin_errors=discovery.errors,
        )
        code = COMMANDS[args.command](args, run)
    except EduLogGenError as exc:
        details = ", ".join(f"{k}={v}" for k, v in exc.context.items())
        sys.stderr.write(f"error: {exc}" + (f" ({details})" if details else "") + "\n")
        return EXIT_ERROR

    if run.manifest_dir is not None:
        try:
            write_json(run.manifest_dir / "run_manifest.json", run.manifest(code))
        except EduLogGenError as exc:  # pragma: no cover - disk failure after success
            logger.warning("could not write run manifest: %s", exc)
    return code
