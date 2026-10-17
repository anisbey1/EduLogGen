"""v1.6.0 tuning: equal-budget hyperparameter search for the GRU and the Transformer.

    python studies/tune_v16.py --study ednet --model transformer --stage search --device cuda
    python studies/tune_v16.py --study ednet --model transformer --stage confirm --device cuda
    python studies/tune_v16.py --study ednet --model transformer --stage final --device cuda
    python studies/tune_v16.py --report            # Markdown summary of all result files

Protocol (fixed before any run):

* The benchmark's 70/30 learner split is reused; the 30% holdout is used only
  in the ``final`` stage. Selection happens inside the 70% training
  partition, which is split again by learner into inner-training (80%) and
  inner-validation (20%) learners. Each model's own early-stopping split lies
  inside the inner-training learners.
* Both models get the same budget: a grid of 12 configurations
  (3 sizes × 2 learning rates × 2 values of a third setting), 100 epochs at
  most, patience 10, and the GRU's learner-level early-stopping split.
* Selection score, lower is better: the mean of event-type TVD, bigram TVD,
  and inter-event time KS between one inner-validation-size sample and the
  inner-validation learners (session lengths, learners, and start times come
  from machinery shared by all generators, so they are not scored). On OULAD
  the score is averaged over the seven modules; one configuration is chosen
  per model and dataset. Generation seeds depend only on the training
  seed, so all configurations are scored on the same random streams.
* ``search``: every configuration with training seed 0. ``confirm``: the
  three best configurations with training seeds 0, 1, 2 (seed 0 reused); the
  best mean score wins. ``final``: the winner and the untuned defaults
  (``gru_matched`` and ``transformer`` of the v1.6.0 study), each with
  training seeds 0, 1, 2, fitted on the whole training partition and
  compared with the holdout on the fast fidelity metrics (three generation
  samples each, metric definitions unchanged).

Results go to ``<results>/<study>-<model>.json`` after every run, and
finished runs are skipped when the command is repeated, so an interrupted
session (e.g. Colab) resumes where it stopped. Every run records its device;
GPU training is best-effort deterministic only, so GPU and CPU numbers are
reported separately.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import study  # noqa: E402
from worker import load  # noqa: E402

HERE = Path(__file__).resolve().parent
OULAD_DATA = HERE / "oulad" / "data"
EDNET_DATA = HERE / "ednet" / "data" / "ednet-kt4-2000.csv"
SPLIT_SEED = 0  # the v1.6.0 study's benchmark split
INNER_SEED = 1
INNER_HOLDOUT = 0.2
SEEDS = (0, 1, 2)
TOP = 3
SCORE = ("event_type_tvd", "bigram_tvd", "interevent_time_ks")
FIXED = {
    "gru": {"validation_split": "learner", "epochs": 100, "patience": 10},
    "transformer": {"epochs": 100, "patience": 10},
}
SIZES = {
    "gru": [
        {"embedding_dim": 32, "hidden_dim": 64},
        {"embedding_dim": 64, "hidden_dim": 128},
        {"embedding_dim": 64, "hidden_dim": 256},
    ],
    "transformer": [
        {"d_model": 64, "num_heads": 4, "num_layers": 2, "ff_dim": 128},
        {"d_model": 128, "num_heads": 4, "num_layers": 2, "ff_dim": 256},
        {"d_model": 128, "num_heads": 4, "num_layers": 4, "ff_dim": 256},
    ],
}
LEARNING_RATES = {"gru": (0.001, 0.003), "transformer": (0.0003, 0.001)}


def third(model: str, name: str) -> tuple[str, tuple[Any, ...]]:
    """The third grid setting: context length where it can matter."""
    if model == "gru":
        return "batch_size", (32, 128)
    if name == "ednet":
        return "max_seq_len", (32, 128)
    return "dropout", (0.1, 0.3)  # OULAD sessions have at most seven events


def grid(model: str, name: str) -> dict[str, dict[str, Any]]:
    """The 12 configurations, keyed by a stable id."""
    key, values = third(model, name)
    out = {}
    for (i, size), lr, value in product(
        enumerate(SIZES[model]), LEARNING_RATES[model], values
    ):
        out[f"s{i}-lr{lr:g}-{key}{value:g}"] = {
            **FIXED[model],
            **size,
            "learning_rate": lr,
            key: value,
        }
    return out


def defaults(model: str) -> dict[str, Any]:
    label = "gru_matched" if model == "gru" else "transformer"
    params = dict(study.GENERATORS[label][1])
    params.pop("threads", None)
    return params


def datasets(name: str) -> list[tuple[str, Any]]:
    """(key, sessionized dataset) per OULAD module, or the EdNet sample."""
    if name == "ednet":
        return [("ednet-kt4-2000", load("ednet", EDNET_DATA, None))]
    return [
        (p.stem, load("oulad", OULAD_DATA, p.stem))
        for p in sorted(OULAD_DATA.glob("*.csv"))
    ]


def device_name(device: str) -> str:
    import torch

    if device in ("cuda", "auto") and torch.cuda.is_available():
        return f"cuda: {torch.cuda.get_device_name(0)}"
    if device in ("mps", "auto") and torch.backends.mps.is_available():
        return "mps"
    return f"cpu: {platform.processor() or platform.machine()}"


def run(
    generator: str,
    params: dict[str, Any],
    train: Any,
    target: Any,
    label: str,
    repeats: int,
) -> dict[str, Any]:
    """Fit on ``train`` and score samples against ``target``."""
    result = study.fast_fidelity(
        train, target, generator, params, SPLIT_SEED, label, repeats
    )
    result["score"] = mean(result["metrics"][m]["mean"] for m in SCORE)
    return result


def summary(runs: dict[str, Any], stage: str) -> dict[str, dict[str, Any]]:
    """Mean score per configuration over datasets and seeds of one stage."""
    by_config: dict[str, list[float]] = {}
    for r in runs.get(stage, {}).values():
        by_config.setdefault(r["config"], []).append(r["score"])
    return {
        c: {"score": mean(v), "sd": stdev(v) if len(v) > 1 else 0.0, "n": len(v)}
        for c, v in sorted(by_config.items(), key=lambda kv: (mean(kv[1]), kv[0]))
    }


def plan(
    stage: str, model: str, name: str, store: dict[str, Any], n_data: int
) -> list[tuple[str, int]]:
    """(config id, training seed) pairs of a stage, checking earlier stages."""
    configs = grid(model, name)
    if stage == "search":
        return [(c, 0) for c in configs]
    expected = {"search": len(configs) * n_data, "confirm": TOP * len(SEEDS) * n_data}
    previous = "search" if stage == "confirm" else "confirm"
    if len(store.get(previous, {})) < expected[previous]:
        raise SystemExit(f"run the {previous} stage to completion first")
    ranked = list(summary(store, previous))
    if stage == "confirm":
        return [(c, s) for c in ranked[:TOP] for s in SEEDS]
    return [(c, s) for c in (ranked[0], "default") for s in SEEDS]


def tune(args: argparse.Namespace) -> None:
    path = args.results / f"{args.study}-{args.model}.json"
    store: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    store.setdefault(
        "protocol",
        __doc__.split("Protocol (fixed before any run):")[1]
        .split("Results go")[0]
        .strip(),
    )
    store.setdefault("grid", grid(args.model, args.study))
    store.setdefault("default", defaults(args.model))
    data = datasets(args.study)
    todo = plan(args.stage, args.model, args.study, store, len(data))
    runs = store.setdefault(args.stage, {})
    device = device_name(args.device)
    for key, dataset in data:
        train, holdout = study.split(dataset, SPLIT_SEED)
        if args.stage == "final":
            fit_on, target, repeats = train, holdout, 3
        else:
            fit_on, target = study.split_by_learner(train, INNER_HOLDOUT, INNER_SEED)
            repeats = 1
        for config, seed in todo:
            run_key = f"{config}|{key}|seed{seed}"
            if run_key in runs:
                continue
            base = store["default"] if config == "default" else store["grid"][config]
            params = {
                **base,
                "seed": seed,
                "device": args.device,
                "threads": args.threads,
            }
            started = time.perf_counter()
            result = run(
                args.model, params, fit_on, target, f"tune-seed{seed}", repeats
            )
            runs[run_key] = {
                "config": config,
                "dataset": key,
                "seed": seed,
                "params": params,
                "device": device,
                **result,
            }
            args.results.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(store, indent=2, default=str))
            print(
                f"{args.stage} {run_key}: score {result['score']:.4f}, "
                f"fit {result['fit_s']:.0f}s, total {time.perf_counter() - started:.0f}s",
                flush=True,
            )
    print(f"{args.stage} done: {path}")


def report(results: Path) -> str:
    lines = ["# v1.6.0 tuning results", ""]
    for path in sorted(results.glob("*-*.json")):
        store = json.loads(path.read_text())
        lines += [f"## {path.stem}", ""]
        for stage in ("search", "confirm"):
            if not store.get(stage):
                continue
            lines += [
                f"### {stage.capitalize()} (inner validation; score ↓)",
                "",
                "| Configuration | Score | SD | Runs |",
                "| --- | --- | --- | --- |",
            ]
            lines += [
                f"| `{c}` | {s['score']:.4f} | {s['sd']:.4f} | {s['n']} |"
                for c, s in summary(store, stage).items()
            ]
            lines.append("")
        final = store.get("final", {})
        if final:
            lines += [
                "### Final (holdout; mean ± SD over training seeds and datasets)",
                "",
                "| Configuration | "
                + " | ".join(study.FAST_METRICS)
                + " | Fit (s) | Device |",
                "| --- |" + " --- |" * (len(study.FAST_METRICS) + 2),
            ]
            for config in dict.fromkeys(r["config"] for r in final.values()):
                rs = [r for r in final.values() if r["config"] == config]
                cells = []
                for m in study.FAST_METRICS:
                    v = [r["metrics"][m]["mean"] for r in rs if m in r["metrics"]]
                    cells.append(
                        f"{mean(v):.3f} ± {stdev(v):.3f}" if len(v) > 1 else "–"
                    )
                devices = sorted({r["device"] for r in rs})
                lines.append(
                    f"| `{config}` | "
                    + " | ".join(cells)
                    + f" | {mean(r['fit_s'] for r in rs):.0f} | {', '.join(devices)} |"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Run one tuning stage, or print the report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--study", choices=("oulad", "ednet"))
    parser.add_argument("--model", choices=("gru", "transformer"))
    parser.add_argument("--stage", choices=("search", "confirm", "final"))
    parser.add_argument(
        "--device", default="cpu", choices=("cpu", "cuda", "mps", "auto")
    )
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--results", type=Path, default=HERE / "tuning")
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        print(report(args.results))
        return
    if not (args.study and args.model and args.stage):
        parser.error("--study, --model, and --stage are required (or --report)")
    tune(args)


if __name__ == "__main__":
    main()
