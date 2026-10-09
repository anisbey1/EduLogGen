"""Turn ``results.json`` from ``run.py`` into the paper's tables and figure.

    python studies/oulad/report.py --results studies/oulad/out/results.json \
        --output paper/softwarex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

GENERATORS = {
    "independent": "Independent",
    "markov": "Markov (1)",
    "markov2": "Markov (2)",
    "semi_markov": "Semi-Markov",
}
METRICS = {
    "event_type_tvd": ("Event-type TVD", "lower"),
    "bigram_tvd": ("Bigram TVD", "lower"),
    "transition_jsd": ("Transition JSD", "lower"),
    "session_length_ks": ("Session length KS", "lower"),
    "interevent_time_ks": ("Inter-event time KS", "lower"),
    "topn_path_overlap": ("Top-10 path overlap", "higher"),
    "exact_session_dup_rate": ("Exact duplicate sessions", "context"),
}


def _value(entry: dict[str, Any], generator: str, metric: str) -> float | None:
    summary = entry["fidelity"]["generators"][generator]["metrics"].get(metric)
    return None if summary is None else summary["mean"]


def fidelity_table(results: dict[str, Any]) -> tuple[str, str]:
    presentations = results["presentations"]
    rows = []
    for metric, (label, better) in METRICS.items():
        ref = [p["fidelity"]["reference"].get(metric) for p in presentations.values()]
        ref_mean = mean(v for v in ref if v is not None)
        values = {
            g: mean(
                v
                for p in presentations.values()
                if (v := _value(p, g, metric)) is not None
            )
            for g in GENERATORS
        }
        if better == "lower":
            best = min(values, key=lambda g: values[g])
        elif better == "higher":
            best = max(values, key=lambda g: values[g])
        else:
            best = None
        rows.append((label, better, ref_mean, values, best))

    md = ["| Metric | Real vs real | " + " | ".join(GENERATORS.values()) + " |"]
    md.append("| --- | --- |" + " --- |" * len(GENERATORS))
    tex = [
        r"\begin{tabular}{l r " + "r " * len(GENERATORS) + "}",
        r"\toprule",
        "Metric & Real vs real & " + " & ".join(GENERATORS.values()) + r" \\",
        r"\midrule",
    ]
    for label, better, ref_mean, values, best in rows:
        arrow = {"lower": r"$\downarrow$", "higher": r"$\uparrow$"}.get(better, "")
        cells_md = [
            f"**{values[g]:.3f}**" if g == best else f"{values[g]:.3f}"
            for g in GENERATORS
        ]
        cells_tex = [
            rf"\textbf{{{values[g]:.3f}}}" if g == best else f"{values[g]:.3f}"
            for g in GENERATORS
        ]
        md.append(f"| {label} | {ref_mean:.3f} | " + " | ".join(cells_md) + " |")
        tex.append(
            f"{label} {arrow} & {ref_mean:.3f} & " + " & ".join(cells_tex) + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def per_presentation_table(results: dict[str, Any]) -> tuple[str, str]:
    md = [
        "| Presentation | Learners | Learner-days | Sessions | Profiles (k) | "
        "Recovery ARI | Outcome NMI (holdout) | Detector ROC-AUC | Detector AP | Run time (s) |",
        "| --- |" + " --- |" * 9,
    ]
    tex = [
        r"\begin{tabular}{l r r r r r r r r}",
        r"\toprule",
        r"Presentation & Learners & Learner-days & Sessions & $k$ & ARI & NMI$_{\mathrm{outcome}}$ & ROC-AUC & AP \\",
        r"\midrule",
    ]
    for name, p in results["presentations"].items():
        prof, det = p["profiles"], p["detection"]["detection"]
        total = p["load_s"] + p["fidelity_s"] + p["profiles_s"] + p["detection_s"]
        cells = [
            name,
            f"{p['n_learners']:,}",
            f"{p['n_events']:,}",
            f"{p['n_sessions']:,}",
            str(prof["k"]),
            f"{prof['recovery']['ari']:.2f}",
            f"{prof['holdout_outcome_nmi']:.3f}",
            f"{det['roc_auc']:.3f}",
            f"{det['average_precision']:.3f}",
        ]
        md.append("| " + " | ".join(cells) + f" | {total:.0f} |")
        tex.append(" & ".join(c.replace(",", "{,}") for c in cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(md) + "\n", "\n".join(tex) + "\n"


def figure(results: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    presentations = list(results["presentations"])
    panels = [("bigram_tvd", "Bigram TVD (lower is better)"),
              ("interevent_time_ks", "Inter-event time KS (lower is better)")]  # fmt: skip
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
    width = 0.8 / (len(GENERATORS) + 1)
    for ax, (metric, title) in zip(axes, panels, strict=True):
        xs = range(len(presentations))
        series = [("reference", "Real vs real")] + list(GENERATORS.items())
        for i, (key, label) in enumerate(series):
            values = [
                (
                    results["presentations"][p]["fidelity"]["reference"].get(metric)
                    if key == "reference"
                    else _value(results["presentations"][p], key, metric)
                )
                for p in presentations
            ]
            ax.bar(
                [x + (i - len(series) / 2 + 0.5) * width for x in xs],
                [v or 0 for v in values],
                width,
                label=label,
                color="0.25" if key == "reference" else None,
            )
        ax.set_xticks(list(xs))
        ax.set_xticklabels([p.split("-")[0] for p in presentations])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("OULAD module (2014J)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".png"), dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    fid_md, fid_tex = fidelity_table(results)
    per_md, per_tex = per_presentation_table(results)
    (args.output / "table_fidelity.tex").write_text(fid_tex)
    (args.output / "table_presentations.tex").write_text(per_tex)
    (args.output / "results.md").write_text(
        "# OULAD study results\n\n## Fidelity (mean over presentations)\n\n"
        + fid_md
        + "\n## Per presentation\n\n"
        + per_md
    )
    figure(results, args.output / "fig_fidelity.pdf")
    print((args.output / "results.md").read_text())


if __name__ == "__main__":
    main()
