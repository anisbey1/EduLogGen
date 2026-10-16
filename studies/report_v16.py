"""Markdown report of the v1.6.0 studies (six-generator comparison, costs,
Transformer sensitivity). Missing results are shown as "pending", so the
same script produces the result template before the runs finish.

    python studies/report_v16.py --output studies/report/v16.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

HERE = Path(__file__).resolve().parent
LABELS = {
    "independent": "Independent",
    "markov": "Markov-1",
    "markov2": "Markov-2",
    "semi_markov": "Semi-Markov",
    "gru": "GRU (v1.5.0 defaults)",
    "gru_matched": "GRU (matched)",
    "transformer": "Transformer",
}
METRICS = {
    "event_type_tvd": "Event-type TVD ↓",
    "activity_jsd": "Activity JSD ↓",
    "bigram_tvd": "Bigram TVD ↓",
    "transition_jsd": "Transition JSD ↓",
    "session_length_ks": "Session-length KS ↓",
    "interevent_time_ks": "Inter-event time KS ↓",
    "topn_path_overlap": "Top-10 path overlap ↑",
    "exact_session_dup_rate": "Exact duplicates vs holdout (diagnostic)",
    "nn_distance_p05": "NN edit distance, 5th pct. ↑",
}
PENDING = "pending"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt(values: list[float], sd: bool = True) -> str:
    if not values:
        return PENDING
    if len(values) == 1 or not sd:
        return f"{mean(values):.3f}"
    return f"{mean(values):.3f} ± {stdev(values):.3f}"


def _metric(entry: dict[str, Any], label: str, metric: str) -> dict[str, Any] | None:
    gen = entry.get("fidelity", {}).get("generators", {}).get(label)
    return None if gen is None else gen.get("metrics", {}).get(metric)


def comparison(title: str, results: dict[str, Any], over: str) -> list[str]:
    entries = list(results.get("presentations", {}).values())
    lines = [
        f"### {title}",
        "",
        f"Mean ± SD over {over}. Real vs real: training vs holdout learners.",
        "",
        "| Metric | Real vs real | " + " | ".join(LABELS.values()) + " |",
        "| --- |" + " --- |" * (len(LABELS) + 1),
    ]
    for metric, name in METRICS.items():
        refs = [e["fidelity"]["reference"].get(metric) for e in entries]
        refs = [r for r in refs if r is not None]
        cells = []
        for label in LABELS:
            if len(entries) > 1:
                values = [
                    m["mean"]
                    for e in entries
                    if (m := _metric(e, label, metric)) and m["mean"] is not None
                ]
                cells.append(_fmt(values) if len(values) == len(entries) else PENDING)
            else:
                m = _metric(entries[0], label, metric) if entries else None
                cells.append(
                    _fmt([v for v in m["values"] if v is not None]) if m else PENDING
                )
        lines.append(f"| {name} | {_fmt(refs, sd=False)} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def per_module(results: dict[str, Any]) -> list[str]:
    lines = [
        "### OULAD per module (mean ± SD over three seeds)",
        "",
        "| Module | Metric | Real vs real | " + " | ".join(LABELS.values()) + " |",
        "| --- | --- |" + " --- |" * (len(LABELS) + 1),
    ]
    for name, e in results.get("presentations", {}).items():
        for metric in (
            "bigram_tvd",
            "interevent_time_ks",
            "event_type_tvd",
            "session_length_ks",
        ):
            ref = e["fidelity"]["reference"].get(metric)
            cells = []
            for label in LABELS:
                m = _metric(e, label, metric)
                cells.append(
                    _fmt([v for v in m["values"] if v is not None]) if m else PENDING
                )
            lines.append(
                f"| {name} | {METRICS[metric]} | {ref:.3f} | "
                + " | ".join(cells)
                + " |"
            )
    return [*lines, ""]


def per_seed(results: dict[str, Any]) -> list[str]:
    lines = [
        "### EdNet per seed",
        "",
        "| Generator | Metric | Seed 1 | Seed 2 | Seed 3 |",
        "| --- | --- | --- | --- | --- |",
    ]
    entry = next(iter(results.get("presentations", {}).values()), {})
    for label, title in LABELS.items():
        for metric in ("bigram_tvd", "interevent_time_ks"):
            m = _metric(entry, label, metric)
            vals = [f"{v:.4f}" for v in m["values"]] if m else [PENDING] * 3
            lines.append(f"| {title} | {METRICS[metric]} | " + " | ".join(vals) + " |")
    return [*lines, ""]


def memorisation(title: str, results: dict[str, Any]) -> list[str]:
    lines = [
        f"### Memorisation against training learners: {title}",
        "",
        "Ranges over modules where several. Holdout = unseen real learners (baseline).",
        "",
        "| Group | Once-seen session matches | Any-session matches | Trajectory matches | Feature-identical | Median distance |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    entries = list(results.get("presentations", {}).values())
    for label in (
        "holdout",
        "markov2",
        "semi_markov",
        "gru",
        "gru_matched",
        "transformer",
    ):
        rows = [
            e["memorisation"][label]
            for e in entries
            if label in e.get("memorisation", {})
        ]
        if len(rows) < len(entries) or not rows:
            lines.append(
                f"| {LABELS.get(label, 'Holdout')} | {PENDING} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |"
            )
            continue

        def rng(key: str, sub: str | None = None, pct: bool = True) -> str:
            vals = [(r[sub][key] if sub else r[key]) or 0.0 for r in rows]
            lo, hi = min(vals), max(vals)
            f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.2f}")
            return f(lo) if f(lo) == f(hi) else f"{f(lo)}–{f(hi)}"

        lines.append(
            f"| {LABELS.get(label, 'Holdout')} | {rng('match_unique_train', 'session')} | {rng('match_any_train', 'session')} "
            f"| {rng('trajectory_match_train')} | {rng('share_dcr_zero')} | {rng('dcr_median', pct=False)} |"
        )
    return [*lines, ""]


def costs(title: str, results: dict[str, Any]) -> list[str]:
    lines = [
        f"### Cost: {title}",
        "",
        "One fit on the training learners and one holdout-size sample, each generator in its own process "
        "(Intel Core i9-9880H; neural training with 8 threads, generation on 1 thread). Ranges over modules where several.",
        "",
        "| Generator | Fit (s) | Generation (ms per event) | Peak memory (MB) | Parameters | Model JSON (MB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    entries = list(results.get("presentations", {}).values())
    for label, title_ in LABELS.items():
        rows = [e["costs"][label] for e in entries if label in e.get("costs", {})]
        if len(rows) < len(entries) or not rows:
            lines.append(
                f"| {title_} | {PENDING} | {PENDING} | {PENDING} | {PENDING} | {PENDING} |"
            )
            continue

        def rng(values: list[float], digits: int) -> str:
            lo, hi = min(values), max(values)
            return (
                f"{lo:.{digits}f}"
                if f"{lo:.{digits}f}" == f"{hi:.{digits}f}"
                else f"{lo:.{digits}f}–{hi:.{digits}f}"
            )

        params = [r["n_parameters"] for r in rows if r.get("n_parameters")]
        lines.append(
            f"| {title_} | {rng([r['fit_s'] for r in rows], 1)} "
            f"| {rng([1000 * r['generate_s'] / max(r['generated_events'], 1) for r in rows], 3)} "
            f"| {rng([r['peak_rss_mb'] for r in rows], 0)} "
            f"| {f'{max(params):,}' if params else '–'} "
            f"| {rng([r['model_json_bytes'] / 2**20 for r in rows], 2)} |"
        )
    return [*lines, ""]


def context(sens: dict[str, Any]) -> list[str]:
    lines = [
        "### Transformer context length (EdNet)",
        "",
        "Fast fidelity metrics (no privacy diagnostics), mean ± SD over three seeds.",
        "",
        "| max_seq_len | Bigram TVD ↓ | Transition JSD ↓ | Inter-event time KS ↓ | Session-length KS ↓ | Fit (s) | Generation (s per sample) | Peak memory (MB) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for length in ("32", "64", "128", "256"):
        r = sens.get("context", {}).get(length)
        if not r:
            lines.append(f"| {length} | " + " | ".join([PENDING] * 7) + " |")
            continue
        m = r["metrics"]
        lines.append(
            f"| {length} | {_fmt(m['bigram_tvd']['values'])} | {_fmt(m['transition_jsd']['values'])} "
            f"| {_fmt(m['interevent_time_ks']['values'])} | {_fmt(m['session_length_ks']['values'])} "
            f"| {r['fit_s']:.0f} | {r['generate_s']:.0f} | {r['peak_rss_mb']:.0f} |"
        )
    return [*lines, ""]


def fraction(sens: dict[str, Any]) -> list[str]:
    lines = [
        "### Training-data size (nested subsets of the training learners)",
        "",
        "Bigram TVD ↓ / inter-event time KS ↓; OULAD: mean over the seven modules; EdNet: mean over three seeds.",
        "",
        "| Data | Generator | 10% | 25% | 50% | 100% |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    blocks = sens.get("fraction", {})
    groups = {
        "OULAD": [k for k in blocks if k.endswith("2014J")],
        "EdNet": [k for k in blocks if k.startswith("ednet")],
    }
    for data, keys in groups.items():
        for label in ("markov", "semi_markov", "gru_matched", "transformer"):
            cells = []
            for f in ("0.1", "0.25", "0.5", "1.0"):
                runs = [blocks[k].get(f"{label}@{f}") for k in keys]
                if not keys or any(r is None for r in runs):
                    cells.append(PENDING)
                    continue
                b = mean(r["metrics"]["bigram_tvd"]["mean"] for r in runs)
                t = mean(r["metrics"]["interevent_time_ks"]["mean"] for r in runs)
                cells.append(f"{b:.3f} / {t:.3f}")
            lines.append(f"| {data} | {LABELS[label]} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def main() -> None:
    """Write the v1.6.0 study report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=HERE / "report" / "v16.md")
    args = parser.parse_args()
    oulad = _load(HERE / "oulad" / "results" / "results.json")
    ednet = _load(HERE / "ednet" / "results" / "results.json")
    sens = _load(HERE / "sensitivity" / "results.json")
    lines = [
        "# EduLogGen v1.6.0 study results",
        "",
        "Generated by `studies/report_v16.py` from the saved results; `pending` marks results not yet computed.",
        "Defaults are untuned; no hyperparameter search was run for any generator.",
        "",
        "## Six-generator comparison (session_fidelity_v1)",
        "",
        *comparison("OULAD, seven 2014J modules", oulad, "the seven modules"),
        *comparison("EdNet-KT4, 2,000 students", ednet, "three generation seeds"),
        *per_module(oulad),
        *per_seed(ednet),
        "## Memorisation",
        "",
        *memorisation("OULAD", oulad),
        *memorisation("EdNet", ednet),
        "## Cost",
        "",
        *costs("OULAD", oulad),
        *costs("EdNet", ednet),
        "## Transformer sensitivity",
        "",
        *context(sens),
        *fraction(sens),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
