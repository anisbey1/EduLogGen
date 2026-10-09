"""Reproducible OULAD study for the EduLogGen software paper.

Run ``prepare.py`` first, then:

    python studies/oulad/run.py --data studies/oulad/data \
        --oulad OULAD --output studies/oulad/out

For each module presentation this script:

1. ingests and sessionizes the events (explicit learner-week sessions);
2. runs the ``session_fidelity_v1`` benchmark (70/30 learner split, three
   seeds) for the independent baseline, first- and second-order Markov, and
   semi-Markov generators, with the real-versus-real reference;
3. learns behavioural profiles on training learners only, assigns holdout
   learners, relates profiles to final results (descriptively), and measures
   profile recovery on a synthetic profile mixture;
4. injects labelled sequence anomalies into real holdout sessions and scores a
   simple transition-likelihood detector fitted on training learners.

Results go to ``<output>/results.json`` and Markdown/LaTeX tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from collections import Counter
from pathlib import Path
from typing import Any

import eduloggen as elg
from eduloggen.analysis import session_sequences, sessionize
from eduloggen.benchmark import run_benchmark, split_by_learner
from eduloggen.core import ConfigError
from eduloggen.evaluation import (
    evaluate_clustering,
    evaluate_detection,
    normalized_mutual_information,
)
from eduloggen.models import Dataset
from eduloggen.scenarios import (
    AnomalySpec,
    fit_profiles,
    generate_profiles,
    inject_anomalies,
)

REPORTED = [
    "event_type_tvd",
    "activity_jsd",
    "session_length_ks",
    "interevent_time_ks",
    "bigram_tvd",
    "transition_jsd",
    "topn_path_overlap",
    "exact_session_dup_rate",
    "nn_distance_p05",
]
SEED = 0
N_PROFILES = 4


def load(path: Path, mapping: Path) -> Dataset:
    result = elg.ingest(path, mapping)
    return sessionize(result.dataset, strategy="explicit")


def final_results(oulad: Path, presentation: str) -> dict[str, str]:
    module, code = presentation.split("-")
    with (oulad / "studentInfo.csv").open(newline="") as handle:
        return {
            row["id_student"]: row["final_result"]
            for row in csv.DictReader(handle)
            if row["code_module"] == module and row["code_presentation"] == code
        }


def run_fidelity(data: Dataset) -> dict[str, Any]:
    names = {
        "independent": ("independent", {}),
        "markov": ("markov", {"order": 1}),
        "markov2": ("markov", {"order": 2}),
        "semi_markov": ("semi_markov", {}),
    }
    out: dict[str, Any] = {"generators": {}}
    for label, (generator, params) in names.items():
        report = run_benchmark(
            data, [generator], seed=SEED, hyperparameters={generator: params}
        )
        result = report.results[0]
        out["generators"][label] = {
            "fit_s": result.fit_s,
            "generate_s": result.generate_s,
            "error": result.error,
            "metrics": {
                m: {"mean": s.mean, "std": s.std}
                for m, s in result.metrics.items()
                if m in REPORTED
            },
        }
        out["reference"] = {m: v for m, v in report.reference.items() if m in REPORTED}
        out["split"] = dict(report.dataset["train"]), dict(report.dataset["holdout"])
        out["protocol"] = dict(report.protocol)
    return out


def run_profiles(data: Dataset, outcomes: dict[str, str]) -> dict[str, Any]:
    train, holdout = split_by_learner(data, 0.3, SEED)
    started = time.perf_counter()
    # k = 4, lowered only if a profile would have fewer than 5 learners
    for k in range(N_PROFILES, 1, -1):
        try:
            profiles = fit_profiles(train, n_profiles=k, seed=SEED)
            break
        except ConfigError:
            continue
    fit_s = time.perf_counter() - started
    assigned = profiles.assign(holdout)
    train_assigned = profiles.assign(train)

    def crosstab(mapping: dict[str, str]) -> dict[str, dict[str, int]]:
        table: dict[str, Counter[str]] = {}
        for learner, profile in mapping.items():
            table.setdefault(profile, Counter())[outcomes.get(learner, "?")] += 1
        return {k: dict(sorted(v.items())) for k, v in sorted(table.items())}

    def outcome_nmi(mapping: dict[str, str]) -> float:
        ids = [i for i in sorted(mapping) if i in outcomes]
        return normalized_mutual_information(
            [outcomes[i] for i in ids], [mapping[i] for i in ids]
        )

    # Recovery: a synthetic population mixing the learned profiles, then
    # cluster it again and compare with the known profile of each learner.
    mixture = generate_profiles(profiles, train.n_sessions, seed=SEED)
    recovered = fit_profiles(mixture.dataset, n_profiles=k, seed=SEED)
    recovery = evaluate_clustering(
        mixture.annotations, recovered.assign(mixture.dataset)
    )
    return {
        "fit_s": fit_s,
        "k": k,
        "profiles": [
            {
                "name": p.name,
                "share": p.share,
                "n_learners": p.n_learners,
                "mean_sessions": p.description["mean_sessions"],
                "mean_session_length": p.description["mean_session_length"],
                "distinguishing": p.description["distinguishing"],
            }
            for p in profiles.profiles
        ],
        "train_outcomes": crosstab(train_assigned),
        "holdout_outcomes": crosstab(assigned),
        "train_outcome_nmi": outcome_nmi(train_assigned),
        "holdout_outcome_nmi": outcome_nmi(assigned),
        "recovery": recovery.to_dict(),
    }


def transition_scores(train: Dataset, target: Dataset) -> dict[str, float]:
    """Mean negative log-likelihood per transition under a smoothed bigram model."""
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    vocab: set[str] = set()
    for seq in session_sequences(train):
        path = ("<s>", *seq)
        vocab.update(seq)
        for a, b in zip(path, path[1:], strict=False):
            counts[a, b] += 1
            totals[a] += 1
    v = len(vocab) + 1
    scores = {}
    for session in target.sessions or ():
        path = ("<s>", *session.event_sequence)
        nll = [
            -math.log((counts[a, b] + 1) / (totals[a] + v))
            for a, b in zip(path, path[1:], strict=False)
        ]
        scores[session.session_id] = sum(nll) / len(nll)
    return scores


def run_detection(data: Dataset) -> dict[str, Any]:
    train, holdout = split_by_learner(data, 0.3, SEED)
    specs = [
        AnomalySpec(type="unexpected_transition", rate=0.05),
        AnomalySpec(
            type="repetition",
            rate=0.05,
            params={"pattern": "ping_pong", "length": 4, "gap_s": 86_400.0},
        ),
    ]
    injected = inject_anomalies(holdout, specs, seed=SEED, reference=train)
    target = sessionize(injected.dataset, strategy="explicit")
    scores = transition_scores(train, target)
    report = evaluate_detection(target, injected.annotations, scores, level="session")
    return {
        "injection": json.loads(json.dumps(injected.report, default=str)),
        "detection": report.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--oulad", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mapping", type=Path, default=Path(__file__).with_name("mapping.yaml")
    )
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "environment": {
            "eduloggen": elg.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "os": platform.system(),
        },
        "seed": SEED,
        "presentations": {},
    }
    for path in sorted(args.data.glob("*.csv")):
        presentation = path.stem
        if args.only and presentation not in args.only:
            continue
        print(f"== {presentation}", flush=True)
        started = time.perf_counter()
        data = load(path, args.mapping)
        load_s = time.perf_counter() - started
        entry: dict[str, Any] = {
            "n_events": data.n_events,
            "n_sessions": data.n_sessions,
            "n_learners": len(data.learner_ids),
            "load_s": load_s,
        }
        for name, step in (
            ("fidelity", lambda: run_fidelity(data)),
            (
                "profiles",
                lambda: run_profiles(data, final_results(args.oulad, presentation)),
            ),
            ("detection", lambda: run_detection(data)),
        ):
            t = time.perf_counter()
            entry[name] = step()
            entry[f"{name}_s"] = time.perf_counter() - t
            print(f"   {name}: {entry[f'{name}_s']:.1f}s", flush=True)
        results["presentations"][presentation] = entry
        (args.output / "results.json").write_text(
            json.dumps(results, indent=2, default=str)
        )
    print(f"wrote {args.output / 'results.json'}")


if __name__ == "__main__":
    main()
