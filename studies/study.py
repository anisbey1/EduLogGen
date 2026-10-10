"""Analyses shared by the EduLogGen paper studies (OULAD, EdNet).

Every function takes a sessionized dataset and a seed and returns
JSON-compatible results. The 70/30 learner split of the benchmark protocol is
reused everywhere, so all analyses fit on the same training learners.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping
from itertools import pairwise
from typing import Any

from eduloggen.analysis import session_sequences, sessionize
from eduloggen.benchmark import run_benchmark, split_by_learner
from eduloggen.core import ConfigError
from eduloggen.evaluation import (
    adjusted_rand_index,
    average_precision,
    evaluate_clustering,
    evaluate_detection,
    normalized_mutual_information,
    roc_auc,
)
from eduloggen.generators import get_generator
from eduloggen.models import Dataset
from eduloggen.scenarios import (
    AnomalySpec,
    fit_profiles,
    generate_profiles,
    inject_anomalies,
)
from eduloggen.scenarios.clustering import Standardizer, learner_features

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
GENERATORS = {
    "independent": ("independent", {}),
    "markov": ("markov", {"order": 1}),
    "markov2": ("markov", {"order": 2}),
    "semi_markov": ("semi_markov", {}),
    "gru": ("gru", {}),
}
MEMORISATION = ("markov2", "semi_markov", "gru")
HOLDOUT = 0.3
N_PROFILES = 4
N_STABILITY = 10
MIN_LENGTH = 3


def split(data: Dataset, seed: int) -> tuple[Dataset, Dataset]:
    """The benchmark protocol's learner split."""
    return split_by_learner(data, HOLDOUT, seed)


def timed(step: Callable[[], dict[str, Any]]) -> tuple[dict[str, Any], float]:
    """Run ``step`` and return its result and wall time in seconds."""
    started = time.perf_counter()
    result = step()
    return result, time.perf_counter() - started


# ---------------------------------------------------------------------------
# 1. Fidelity: session_fidelity_v1 per generator
# ---------------------------------------------------------------------------


def fidelity(
    data: Dataset, seed: int, labels: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Benchmark every generator; metrics as mean and SD over three seeds.

    The reference compares the training learners (as "synthetic") with the
    holdout learners once per dataset; each generator sample has the size of
    the holdout and is compared with the holdout.
    """
    out: dict[str, Any] = {"generators": {}}
    for label, (generator, params) in GENERATORS.items():
        if labels is not None and label not in labels:
            continue
        report = run_benchmark(
            data, [generator], seed=seed, hyperparameters={generator: params}
        )
        result = report.results[0]
        out["generators"][label] = {
            "fit_s": result.fit_s,
            "generate_s": result.generate_s,
            "error": result.error,
            "metrics": {
                m: {"mean": s.mean, "std": s.std, "values": list(s.values)}
                for m, s in result.metrics.items()
                if m in REPORTED
            },
        }
        out["reference"] = {m: v for m, v in report.reference.items() if m in REPORTED}
        out["split"] = dict(report.dataset["train"]), dict(report.dataset["holdout"])
        out["protocol"] = dict(report.protocol)
    return out


# ---------------------------------------------------------------------------
# 2. Profiles: real-data stability, outcome association, recovery
# ---------------------------------------------------------------------------


def _keep(data: Dataset, learners: set[str], name: str) -> Dataset:
    return Dataset(
        dataset_id=f"{data.dataset_id}[{name}]",
        events=tuple(e for e in data.events if e.learner_id in learners),
        sessions=tuple(s for s in data.sessions or () if s.learner_id in learners),
        metadata=data.metadata,
    )


def _ari(a: Mapping[str, str], b: Mapping[str, str]) -> float:
    ids = sorted(set(a) & set(b))
    return adjusted_rand_index([a[i] for i in ids], [b[i] for i in ids])


def _fit_k(data: Dataset, k: int, seed: int) -> Any:
    try:
        return fit_profiles(data, n_profiles=k, seed=seed)
    except ConfigError:
        return None


def profiles(
    data: Dataset, seed: int, outcomes: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Automatic profiles on training learners, with three kinds of evidence.

    - stability on real data: agreement (ARI) of training-learner assignments
      across clustering seeds and across 80% learner subsamples, and
      replication on holdout learners (profiles fitted on the holdout alone
      versus holdout learners assigned to the training profiles);
    - association with final results, if ``outcomes`` are given (descriptive);
    - recovery: cluster a synthetic mixture of the profiles again and compare
      with each synthetic learner's known profile.
    """
    train, holdout = split(data, seed)
    k = N_PROFILES
    base = None
    while k > 1 and (base := _fit_k(train, k, seed)) is None:
        k -= 1
    assert base is not None
    train_assigned = base.assign(train)
    holdout_assigned = base.assign(holdout)

    seed_ari = []
    for s in range(1, N_STABILITY):
        other = _fit_k(train, k, seed + s)
        seed_ari.append(
            None if other is None else _ari(train_assigned, other.assign(train))
        )

    rng = random.Random(seed)
    learners = sorted(train.learner_ids)
    subsample_ari = []
    for b in range(N_STABILITY):
        chosen = set(rng.sample(learners, int(0.8 * len(learners))))
        other = _fit_k(_keep(train, chosen, f"sub{b}"), k, seed)
        subsample_ari.append(
            None if other is None else _ari(train_assigned, other.assign(train))
        )

    own = _fit_k(holdout, k, seed)
    replication_ari = (
        None if own is None else _ari(holdout_assigned, own.assign(holdout))
    )

    mixture = generate_profiles(base, train.n_sessions, seed=seed)
    recovered = _fit_k(mixture.dataset, k, seed)
    recovery = (
        None
        if recovered is None
        else evaluate_clustering(
            mixture.annotations, recovered.assign(mixture.dataset)
        ).to_dict()
    )

    result: dict[str, Any] = {
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
            for p in base.profiles
        ],
        "stability": {
            "seed_ari": seed_ari,
            "subsample_ari": subsample_ari,
            "holdout_replication_ari": replication_ari,
        },
        "recovery": recovery,
    }
    if outcomes is not None:

        def crosstab(mapping: Mapping[str, str]) -> dict[str, dict[str, int]]:
            table: dict[str, Counter[str]] = {}
            for learner, profile in mapping.items():
                table.setdefault(profile, Counter())[outcomes.get(learner, "?")] += 1
            return {p: dict(sorted(c.items())) for p, c in sorted(table.items())}

        def nmi(mapping: Mapping[str, str]) -> float:
            ids = [i for i in sorted(mapping) if i in outcomes]
            return normalized_mutual_information(
                [outcomes[i] for i in ids], [mapping[i] for i in ids]
            )

        result |= {
            "train_outcomes": crosstab(train_assigned),
            "holdout_outcomes": crosstab(holdout_assigned),
            "train_outcome_nmi": nmi(train_assigned),
            "holdout_outcome_nmi": nmi(holdout_assigned),
        }
    return result


# ---------------------------------------------------------------------------
# 3. Memorisation of training data (session and learner level)
# ---------------------------------------------------------------------------


def _trajectories(data: Dataset) -> dict[str, tuple[tuple[str, ...], ...]]:
    sessions = sorted(data.sessions or (), key=lambda s: (s.learner_id, s.start_time))
    out: dict[str, list[tuple[str, ...]]] = {}
    for s in sessions:
        out.setdefault(s.learner_id, []).append(tuple(s.event_sequence))
    return {k: tuple(v) for k, v in out.items()}


def _session_exposure(
    candidates: list[tuple[str, ...]], train_counts: Counter[tuple[str, ...]]
) -> dict[str, Any]:
    long = [s for s in candidates if len(s) >= MIN_LENGTH]
    if not long:
        return {"sessions": 0}
    unique = {s for s, c in train_counts.items() if c == 1 and len(s) >= MIN_LENGTH}
    return {
        "sessions": len(long),
        "match_any_train": sum(s in train_counts for s in long) / len(long),
        "match_unique_train": sum(s in unique for s in long) / len(long),
    }


def _nearest(row: list[float], rows: list[list[float]]) -> float:
    return math.sqrt(
        min(sum((a - b) ** 2 for a, b in zip(row, r, strict=True)) for r in rows)
    )


def memorisation(
    data: Dataset, seed: int, generators: tuple[str, ...]
) -> dict[str, Any]:
    """Compare synthetic data with the training learners it was fitted on.

    Baseline: holdout learners, who were never seen by the generator. If a
    generator memorised training data, its output would sit closer to the
    training learners than unseen real learners do.

    - session level (sessions with >= 3 events): share identical to any
      training session, and to a session that occurs only once in training;
    - learner level: share of learners (>= 2 sessions) whose whole ordered
      trajectory of sessions equals a training learner's, and the distance to
      the closest training learner in standardised behavioural features.
    """
    train, holdout = split(data, seed)
    train_counts = Counter(session_sequences(train))
    train_traj = set(_trajectories(train).values())
    names, train_vectors = learner_features(train)
    tokens = [n.removeprefix("share:") for n in names if n.startswith("share:")]
    scaler = Standardizer.fit(list(train_vectors.values()), clip=None)
    train_rows = [scaler.transform(v) for v in train_vectors.values()]

    def describe(target: Dataset) -> dict[str, Any]:
        traj = [t for t in _trajectories(target).values() if len(t) >= 2]
        _, vectors = learner_features(target, tokens)
        dcr = sorted(
            _nearest(scaler.transform(v), train_rows) for v in vectors.values()
        )
        return {
            "session": _session_exposure(list(session_sequences(target)), train_counts),
            "learners_with_2plus_sessions": len(traj),
            "trajectory_match_train": (
                (sum(t in train_traj for t in traj) / len(traj)) if traj else None
            ),
            "dcr": dcr,
        }

    holdout_desc = describe(holdout)
    p05 = holdout_desc["dcr"][max(0, int(0.05 * len(holdout_desc["dcr"])) - 1)]
    out: dict[str, Any] = {"holdout": _summarise_dcr(holdout_desc, p05)}
    for label in generators:
        generator, params = GENERATORS[label]
        family = get_generator(generator)
        model = family.fit(train, params)
        synthetic = family.generate(model, holdout.n_sessions, seed)
        out[label] = _summarise_dcr(describe(synthetic), p05)
    return out


def _summarise_dcr(desc: dict[str, Any], holdout_p05: float) -> dict[str, Any]:
    dcr = desc.pop("dcr")
    n = len(dcr)
    desc["dcr_median"] = dcr[n // 2] if n else None
    desc["dcr_p05"] = dcr[max(0, int(0.05 * n) - 1)] if n else None
    desc["share_below_holdout_p05"] = (
        (sum(d < holdout_p05 for d in dcr) / n) if n else None
    )
    desc["share_dcr_zero"] = (sum(d < 1e-9 for d in dcr) / n) if n else None
    desc["learners"] = n
    return desc


# ---------------------------------------------------------------------------
# 4. Detection against injected ground truth
# ---------------------------------------------------------------------------


def transition_scores(train: Dataset, target: Dataset) -> dict[str, float]:
    """Mean negative log-likelihood per transition (add-one smoothed bigrams)."""
    counts: Counter[tuple[str, str]] = Counter()
    totals: Counter[str] = Counter()
    vocab: set[str] = set()
    for seq in session_sequences(train):
        vocab.update(seq)
        for a, b in pairwise(("<s>", *seq)):
            counts[a, b] += 1
            totals[a] += 1
    v = len(vocab) + 1
    scores = {}
    for session in target.sessions or ():
        path = ("<s>", *session.event_sequence)
        nll = [
            -math.log((counts[a, b] + 1) / (totals[a] + v)) for a, b in pairwise(path)
        ]
        scores[session.session_id] = sum(nll) / len(nll)
    return scores


def _specs(repetition_gap_s: float) -> list[AnomalySpec]:
    return [
        AnomalySpec(type="unexpected_transition", rate=0.05),
        AnomalySpec(
            type="repetition",
            rate=0.05,
            params={"pattern": "ping_pong", "length": 4, "gap_s": repetition_gap_s},
        ),
    ]


def _score_injection(
    train: Dataset, background: Dataset, seed: int, repetition_gap_s: float
) -> dict[str, Any]:
    injected = inject_anomalies(
        background, _specs(repetition_gap_s), seed=seed, reference=train
    )
    target = sessionize(injected.dataset, strategy="explicit")
    scores = transition_scores(train, target)
    report = evaluate_detection(target, injected.annotations, scores, level="session")
    by_type: dict[str, set[str]] = {}
    for row in injected.annotations.rows:
        if row.level == "session":
            by_type.setdefault(str(row.type), set()).add(row.id)
    anomalous = set().union(*by_type.values()) if by_type else set()
    per_type = {}
    for kind, ids in sorted(by_type.items()):
        ranked = [
            (s, i in ids) for i, s in scores.items() if i in ids or i not in anomalous
        ]
        per_type[kind] = {
            "n": len(ids),
            "roc_auc": roc_auc(ranked),
            "average_precision": average_precision(ranked),
        }
    return {
        "sessions": len(scores),
        "anomalous": len(anomalous),
        "roc_auc": report.roc_auc,
        "average_precision": report.average_precision,
        "per_type": per_type,
        "injection": {
            k: {kk: vv for kk, vv in v.items() if kk != "parameters"}
            | {"parameters": dict(v.get("parameters", {}))}
            for k, v in injected.report["anomalies"].items()
        },
    }


def detection(data: Dataset, seed: int, repetition_gap_s: float) -> dict[str, Any]:
    """Inject anomalies into real holdout sessions and into synthetic sessions.

    Each anomaly type goes into a disjoint 5% of sessions (one anomaly per
    session at most), labels are per session, and the detector is fitted on
    training learners only. ``artifact_auc`` checks whether the detector
    tells normal synthetic sessions from normal holdout sessions (0.5 means
    it cannot, so it is not keying on generator artefacts).
    """
    train, holdout = split(data, seed)
    family = get_generator("semi_markov")
    synthetic = family.generate(family.fit(train), holdout.n_sessions, seed)
    clean_real = transition_scores(train, holdout)
    clean_synthetic = transition_scores(train, synthetic)
    ranked = [(s, False) for s in clean_real.values()] + [
        (s, True) for s in clean_synthetic.values()
    ]
    return {
        "real_background": _score_injection(train, holdout, seed, repetition_gap_s),
        "synthetic_background": _score_injection(
            train, synthetic, seed, repetition_gap_s
        ),
        "artifact_auc": roc_auc(ranked),
    }
