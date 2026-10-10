"""Tables and figures for the paper from the OULAD and EdNet study results.

    python studies/report.py --oulad studies/oulad/results/results.json \
        --ednet studies/ednet/results/results.json --output studies/report

Writes ``table_fidelity.tex``, ``table_diagnostics.tex`` (main text),
``supp_tables.tex`` (supplementary material), ``fig_fidelity.pdf``, and a
Markdown summary ``results.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

SYNTHETIC = ("markov2", "semi_markov", "gru")
GENERATORS = {
    "independent": "Independent",
    "markov": "Markov-1",
    "markov2": "Markov-2",
    "semi_markov": "Semi-Markov",
    "gru": "GRU",
}
METRICS = {
    "event_type_tvd": ("Event-type TVD", r"$\downarrow$"),
    "bigram_tvd": ("Bigram TVD", r"$\downarrow$"),
    "transition_jsd": ("Transition JSD", r"$\downarrow$"),
    "session_length_ks": ("Session-length KS", r"$\downarrow$"),
    "interevent_time_ks": ("Inter-event time KS", r"$\downarrow$"),
    "topn_path_overlap": ("Top-10 path overlap", r"$\uparrow$"),
}
# Okabe-Ito colour-blind-safe palette, plus hatching for greyscale print.
COLOURS = ["#555555", "#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#D55E00"]
HATCHES = ["", "//", "..", "xx", "\\\\", "--"]


def _sd(values: list[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def _gen(entry: dict[str, Any], g: str, m: str) -> float | None:
    s = entry["fidelity"]["generators"][g]["metrics"].get(m)
    return None if s is None else s["mean"]


def _cell(values: list[float]) -> str:
    return f"{mean(values):.3f} $\\pm$ {_sd(values):.3f}"


def fidelity_rows(results: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """Per metric: label, arrow, and cells (reference then generators).

    Several presentations: mean +- SD over presentations of the per-seed
    means. One dataset: mean +- SD over the three seeds.
    """
    entries = list(results["presentations"].values())
    rows = []
    for metric, (label, arrow) in METRICS.items():
        if len(entries) > 1:
            ref = [e["fidelity"]["reference"][metric] for e in entries]
            cells = [f"{mean(ref):.3f}"]
            for g in GENERATORS:
                cells.append(
                    _cell([v for e in entries if (v := _gen(e, g, metric)) is not None])
                )
        else:
            e = entries[0]
            cells = [f"{e['fidelity']['reference'][metric]:.3f}"]
            for g in GENERATORS:
                values = e["fidelity"]["generators"][g]["metrics"][metric]["values"]
                cells.append(_cell(values))
        rows.append((label, arrow, cells))
    return rows


def fidelity_table(datasets: dict[str, dict[str, Any]]) -> str:
    """Main-text Table 1."""
    tex = [
        r"\begin{tabular}{l r " + "r " * len(GENERATORS) + "}",
        r"\toprule",
        "Metric & Real vs real & " + " & ".join(GENERATORS.values()) + r" \\",
    ]
    for title, results in datasets.items():
        tex += [
            r"\midrule",
            rf"\multicolumn{{{len(GENERATORS) + 2}}}{{l}}{{\textit{{{title}}}}} \\",
        ]
        for label, arrow, cells in fidelity_rows(results):
            tex.append(f"{label} {arrow} & " + " & ".join(cells) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n"


def _n(value: int) -> str:
    return f"{value:,}".replace(",", "{,}")


def _clean(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def diagnostics(results: dict[str, Any]) -> dict[str, Any]:
    """Summary numbers per dataset (ranges over presentations where several)."""
    es = list(results["presentations"].values())

    def rng(values: list[float]) -> tuple[float, float]:
        return (min(values), max(values))

    stab = [e["profiles"]["stability"] for e in es]
    mem = [e["memorisation"] for e in es]
    det = [e["detection"] for e in es]
    return {
        "seed_ari": rng([median(_clean(s["seed_ari"])) for s in stab]),
        "subsample_ari": rng([median(_clean(s["subsample_ari"])) for s in stab]),
        "replication_ari": rng(_clean([s["holdout_replication_ari"] for s in stab])),
        "recovery_ari": rng(
            _clean([(e["profiles"]["recovery"] or {}).get("ari") for e in es])
        ),
        "unique_holdout": rng(
            [m["holdout"]["session"]["match_unique_train"] for m in mem]
        ),
        "unique_synthetic": rng(
            [m[g]["session"]["match_unique_train"] for m in mem for g in SYNTHETIC]
        ),
        "trajectory_holdout": rng(
            [m["holdout"]["trajectory_match_train"] or 0 for m in mem]
        ),
        "trajectory_synthetic": rng(
            [m[g]["trajectory_match_train"] or 0 for m in mem for g in SYNTHETIC]
        ),
        "dcr_zero_holdout": rng([m["holdout"]["share_dcr_zero"] for m in mem]),
        "dcr_zero_synthetic": rng(
            [m[g]["share_dcr_zero"] for m in mem for g in SYNTHETIC]
        ),
        "dcr_median_holdout": rng([m["holdout"]["dcr_median"] for m in mem]),
        "dcr_median_synthetic": rng(
            [m[g]["dcr_median"] for m in mem for g in SYNTHETIC]
        ),
        "auc_real": rng([d["real_background"]["roc_auc"] for d in det]),
        "auc_synthetic": rng([d["synthetic_background"]["roc_auc"] for d in det]),
        "ap_real": rng([d["real_background"]["average_precision"] for d in det]),
        "artifact_auc": rng([d["artifact_auc"] for d in det]),
    }


def _r(pair: tuple[float, float], digits: int = 2, pct: bool = False) -> str:
    lo, hi = (100 * pair[0], 100 * pair[1]) if pct else pair
    unit = r"\%" if pct else ""
    if f"{lo:.{digits}f}" == f"{hi:.{digits}f}":
        return f"{lo:.{digits}f}{unit}"
    return f"{lo:.{digits}f}--{hi:.{digits}f}{unit}"


def diagnostics_table(datasets: dict[str, dict[str, Any]]) -> str:
    """Main-text Table 2."""
    d = {k: diagnostics(v) for k, v in datasets.items()}
    rows = [
        ("Profile stability, seeds (ARI)", "seed_ari", 2, False),
        ("Profile stability, 80\\% subsamples (ARI)", "subsample_ari", 2, False),
        ("Profile replication on holdout (ARI)", "replication_ari", 2, False),
        ("Profile recovery, synthetic mixture (ARI)", "recovery_ari", 2, False),
        (
            "Sessions matching a once-seen training session: holdout",
            "unique_holdout",
            1,
            True,
        ),
        ("\\quad synthetic (Markov-2, semi-Markov, GRU)", "unique_synthetic", 1, True),
        (
            "Learners repeating a training learner's trajectory: holdout",
            "trajectory_holdout",
            1,
            True,
        ),
        ("\\quad synthetic", "trajectory_synthetic", 1, True),
        (
            "Learners with a feature-identical training learner: holdout",
            "dcr_zero_holdout",
            1,
            True,
        ),
        ("\\quad synthetic", "dcr_zero_synthetic", 1, True),
        (
            "Median distance to nearest training learner: holdout",
            "dcr_median_holdout",
            2,
            False,
        ),
        ("\\quad synthetic", "dcr_median_synthetic", 2, False),
        ("Detector ROC-AUC: real background", "auc_real", 2, False),
        ("\\quad synthetic background", "auc_synthetic", 2, False),
        ("Detector: synthetic vs real normal sessions (AUC)", "artifact_auc", 2, False),
    ]
    tex = [
        r"\begin{tabular}{l " + "r " * len(d) + "}",
        r"\toprule",
        "Diagnostic & " + " & ".join(d) + r" \\",
        r"\midrule",
    ]
    for label, key, digits, pct in rows:
        tex.append(
            label
            + " & "
            + " & ".join(_r(x[key], digits, pct) for x in d.values())
            + r" \\"
        )
    tex += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(tex) + "\n"


def supplementary(datasets: dict[str, dict[str, Any]]) -> str:
    """Per-presentation tables for the supplementary material."""
    out = []
    for title, results in datasets.items():
        for name, e in results["presentations"].items():
            rows = []
            for metric, (label, arrow) in METRICS.items():
                ref = e["fidelity"]["reference"][metric]
                cells = [f"{ref:.3f}"]
                for g in GENERATORS:
                    s = e["fidelity"]["generators"][g]["metrics"][metric]
                    cells.append(f"{s['mean']:.3f} $\\pm$ {s['std'] or 0:.3f}")
                rows.append(f"{label} {arrow} & " + " & ".join(cells) + r" \\")
            prof = e["profiles"]
            det = e["detection"]
            mem = e["memorisation"]
            per_type = "; ".join(
                f"{k.replace('_', ' ')}: ROC-AUC {v['roc_auc']:.2f}, AP {v['average_precision']:.2f}"
                for k, v in det["real_background"]["per_type"].items()
            )
            out += [
                rf"\subsection*{{{title}: {name}}}",
                f"{_n(e['n_learners'])} learners, {_n(e['n_events'])} events, "
                f"{_n(e['n_sessions'])} sessions; wall time "
                f"{sum(e[k] for k in e if k.endswith('_s')):.0f}~s.",
                "",
                r"\begin{tabular}{l r " + "r " * len(GENERATORS) + "}",
                r"\toprule",
                "Metric & Real vs real & " + " & ".join(GENERATORS.values()) + r" \\",
                r"\midrule",
                *rows,
                r"\bottomrule",
                r"\end{tabular}",
                "",
                rf"Profiles ($k={prof['k']}$): seed ARI "
                + ", ".join(f"{v:.2f}" for v in _clean(prof["stability"]["seed_ari"]))
                + "; subsample ARI "
                + ", ".join(
                    f"{v:.2f}" for v in _clean(prof["stability"]["subsample_ari"])
                )
                + f"; holdout replication ARI {prof['stability']['holdout_replication_ari']:.2f}"
                + (
                    f"; recovery ARI {prof['recovery']['ari']:.2f}"
                    if prof.get("recovery")
                    else ""
                )
                + (
                    f"; NMI with final result (holdout) {prof['holdout_outcome_nmi']:.3f}."
                    if "holdout_outcome_nmi" in prof
                    else "."
                ),
                "",
                "Memorisation (sessions with $\\geq$3 events; matches against training learners): "
                + "; ".join(
                    f"{lab}: any {m['session']['match_any_train']:.1%}, once-seen {m['session']['match_unique_train']:.1%}, "
                    f"trajectories {m['trajectory_match_train'] or 0:.1%}, feature-identical {m['share_dcr_zero']:.1%}, median distance {m['dcr_median']:.2f}"
                    for lab, m in (
                        ("holdout", mem["holdout"]),
                        ("Markov-2", mem["markov2"]),
                        ("semi-Markov", mem["semi_markov"]),
                        ("GRU", mem["gru"]),
                    )
                ).replace("%", r"\%")
                + ".",
                "",
                f"Detection (real background): {per_type}; synthetic background ROC-AUC "
                f"{det['synthetic_background']['roc_auc']:.2f}; synthetic-vs-real AUC {det['artifact_auc']:.2f}.",
                "",
            ]
    return "\n".join(out) + "\n"


def figure(datasets: dict[str, dict[str, Any]], path: Path) -> None:
    """Fig. 2: bigram TVD and inter-event KS per presentation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = [
        (title, n, e)
        for title, r in datasets.items()
        for n, e in r["presentations"].items()
    ]
    labels = [
        n.split("-")[0] if t.startswith("OULAD") else "EdNet" for t, n, _ in groups
    ]
    panels = [
        ("bigram_tvd", "Bigram TVD"),
        ("interevent_time_ks", "Inter-event time KS"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
    series = [("reference", "Real vs real"), *GENERATORS.items()]
    width = 0.8 / len(series)
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for i, (key, label) in enumerate(series):
            values = [
                (
                    e["fidelity"]["reference"][metric]
                    if key == "reference"
                    else _gen(e, key, metric)
                )
                for _, _, e in groups
            ]
            ax.bar(
                [x + (i - len(series) / 2 + 0.5) * width for x in range(len(groups))],
                [v or 0 for v in values],
                width,
                label=label,
                color=COLOURS[i],
                hatch=HATCHES[i],
                edgecolor="white" if i == 0 else "black",
                linewidth=0.4,
            )
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(title, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        if len(groups) > 1 and labels[-1] == "EdNet":
            ax.axvline(len(groups) - 1.5, color="0.6", linewidth=0.8, linestyle=":")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, names, loc="upper center", ncol=len(series), fontsize=8, frameon=False
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=150)


def main() -> None:
    """Write every table and figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--oulad", type=Path, required=True)
    parser.add_argument("--ednet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    datasets = {"OULAD (7 modules)": json.loads(args.oulad.read_text())}
    if args.ednet:
        datasets["EdNet-KT4"] = json.loads(args.ednet.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "table_fidelity.tex").write_text(fidelity_table(datasets))
    (args.output / "table_diagnostics.tex").write_text(diagnostics_table(datasets))
    (args.output / "supp_tables.tex").write_text(supplementary(datasets))
    figure(datasets, args.output / "fig_fidelity.pdf")
    summary = {k: diagnostics(v) for k, v in datasets.items()}
    (args.output / "results.md").write_text(
        "# Study summary\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
