"""Behavioural profiles and profile mixtures (Level 2, M3, design §4).

Three ways to get profiles:

- ``auto`` — cluster real learners (seeded k-means++ on standardised
  behavioural features). Fitted on the data given, which should be the
  training split; :meth:`ProfileSet.assign` maps other learners (e.g. a
  holdout) to the nearest learned profile without changing it.
- ``provided`` — a ``learner_id -> profile`` mapping supplied by you.
- ``manual`` — profiles defined by their behaviour (start and transition
  probabilities, session lengths, time spent, sessions per learner) without
  any real data. Proportions alone are not enough.

Each profile gets its own generator model; :func:`generate_profiles` samples
a mixture and records every synthetic learner's profile in the annotations.

Profile names are descriptive labels for behavioural clusters. An
unsupervised cluster is **not** a validated psychological or pedagogical
type, and should not be reported as one.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal

from eduloggen.analysis import START, session_sequences
from eduloggen.core import ConfigError, FitError, IngestionError, PathLike
from eduloggen.generators import (
    SemiMarkovGenerator,
    SessionCalendar,
    get_generator,
    load_model,
    save_model,
)
from eduloggen.models import (
    Annotation,
    Annotations,
    Dataset,
    DatasetMetadata,
    GenerationMetadata,
    GeneratorModel,
    LogRecord,
    Session,
    SyntheticDataset,
)
from eduloggen.privacy import remap_ids_with_mapping
from eduloggen.scenarios.clustering import (
    Standardizer,
    feature_means,
    kmeans,
    learner_features,
    nearest,
)
from eduloggen.scenarios.controls import Controls, apply_controls
from eduloggen.utils import derive_seed, fingerprint, make_rng
from eduloggen.utils.fs import atomic_directory, write_json

__all__ = [
    "Profile",
    "ProfileMode",
    "ProfileResult",
    "ProfileSet",
    "define_profiles",
    "fit_profiles",
    "generate_profiles",
    "load_profile_assignments",
    "load_profiles",
]

ProfileMode = Literal["auto", "provided", "manual"]
NOTE: Final = (
    "Profiles are descriptive behavioural clusters or researcher-defined "
    "behaviours, not validated psychological or pedagogical types."
)
_MANUAL_KEYS: Final = frozenset(
    (
        "share",
        "start",
        "transitions",
        "session_length",
        "sessions_per_learner",
        "dwell_s",
        "dwell_spread",
        "activities",
        "course",
    )
)
_SCALE: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class Profile:
    """One profile.

    Attributes:
        name: Descriptive label.
        model: Generator model of this profile's behaviour.
        share: Default share of synthetic learners.
        n_learners: Training learners in the profile (0 for manual).
        description: JSON-compatible summary (feature means, what stands out).
    """

    name: str
    model: GeneratorModel
    share: float
    n_learners: int
    description: Mapping[str, Any] = field(default_factory=dict, hash=False)


@dataclass(frozen=True, slots=True)
class ProfileSet:
    """Profiles of one population, with how they were obtained.

    Attributes:
        mode: ``auto``, ``provided``, or ``manual``.
        generator_id: Generator family used for every profile.
        profiles: Profiles in a stable order.
        clustering: For ``auto``: feature names, tokens, standardiser, and
            centroids, so new learners can be assigned.
    """

    mode: ProfileMode
    generator_id: str
    profiles: tuple[Profile, ...]
    clustering: Mapping[str, Any] | None = field(default=None, hash=False)

    @property
    def names(self) -> list[str]:
        """Profile names in order."""
        return [p.name for p in self.profiles]

    def profile(self, name: str) -> Profile:
        """Look up a profile.

        Raises:
            KeyError: If there is no such profile.
        """
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise KeyError(name)

    def assign(self, dataset: Dataset) -> dict[str, str]:
        """Map each learner of ``dataset`` to the nearest learned profile.

        Only for ``auto`` profiles. Uses the training standardiser and
        centroids, so the assignment never changes the profiles.

        Raises:
            ConfigError: If the profiles were not learned by clustering.
        """
        if self.mode != "auto" or self.clustering is None:
            raise ConfigError(
                "only auto profiles can assign new learners",
                code="config_invalid_value",
            )
        scaler = Standardizer(
            tuple(self.clustering["means"]), tuple(self.clustering["stds"])
        )
        centroids = self.clustering["centroids"]
        _, vectors = learner_features(dataset, self.clustering["tokens"])
        return {
            learner: self.profiles[nearest(scaler.transform(row), centroids)].name
            for learner, row in sorted(vectors.items())
        }

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible summary (models are referenced by fingerprint)."""
        return {
            "mode": self.mode,
            "generator_id": self.generator_id,
            "profiles": [
                {
                    "name": p.name,
                    "share": p.share,
                    "n_learners": p.n_learners,
                    "description": dict(p.description),
                    "model_fingerprint": p.model.fingerprint(),
                }
                for p in self.profiles
            ],
            "clustering": None if self.clustering is None else dict(self.clustering),
            "note": NOTE,
        }

    def fingerprint(self) -> str:
        """Content hash of profiles, shares, models, and clustering."""
        return fingerprint(self.to_dict())

    def to_markdown(self) -> str:
        """Table of profiles and what distinguishes them."""
        lines = [
            f"# Profiles ({self.mode}, {self.generator_id})",
            "",
            "| Profile | Share | Training learners | Sessions/learner "
            "| Session length | Stands out |",
            "| ------- | ----- | ----------------- | ---------------- "
            "| -------------- | ---------- |",
        ]
        for p in self.profiles:
            d = p.description
            stands = ", ".join(
                f"{s['feature']} ({s['z']:+.1f} sd)"
                for s in d.get("distinguishing", [])
            )
            sessions = _fmt(d.get("mean_sessions"))
            length = _fmt(d.get("mean_session_length"))
            lines.append(
                f"| {p.name} | {p.share:.0%} | {p.n_learners} | {sessions} "
                f"| {length} | {stands or '-'} |"
            )
        lines += ["", f"_{NOTE}_"]
        return "\n".join(lines) + "\n"

    def save(self, path: PathLike, *, force: bool = False) -> Path:
        """Write the profile set as a directory (profiles.json + one model per profile).

        Raises:
            ExportError: If the target is unsafe or exists without ``force``.
        """
        target = Path(path).expanduser().absolute()
        with atomic_directory(target, marker="profiles.json", force=force) as staging:
            for index, profile in enumerate(self.profiles):
                save_model(profile.model, staging / "models" / f"{index:02d}")
            write_json(
                staging / "profiles.json",
                {**self.to_dict(), "fingerprint": self.fingerprint()},
            )
            (staging / "PROFILES.md").write_text(self.to_markdown(), encoding="utf-8")
        return target


def load_profiles(path: PathLike) -> ProfileSet:
    """Read a profile set written by :meth:`ProfileSet.save`.

    Raises:
        IngestionError: If files are missing or the content does not match
            the recorded fingerprint.
    """
    root = Path(path).expanduser().resolve()
    try:
        data = json.loads((root / "profiles.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise IngestionError(
            "profiles.json not found", code="io_not_found", context={"path": str(root)}
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise IngestionError(
            "profiles.json is not readable JSON",
            code="io_corpus_corrupt",
            context={"path": str(root)},
        ) from exc
    profiles = tuple(
        Profile(
            name=item["name"],
            model=load_model(root / "models" / f"{index:02d}"),
            share=float(item["share"]),
            n_learners=int(item["n_learners"]),
            description=item["description"],
        )
        for index, item in enumerate(data["profiles"])
    )
    loaded = ProfileSet(
        data["mode"], data["generator_id"], profiles, data.get("clustering")
    )
    if loaded.fingerprint() != data.get("fingerprint"):
        raise IngestionError(
            "profile set does not match its fingerprint",
            code="io_corpus_integrity",
            context={"path": str(root)},
        )
    return loaded


def load_profile_assignments(path: PathLike) -> dict[str, str]:
    """Read a ``learner_id,profile`` CSV for ``provided`` profiles.

    Raises:
        IngestionError: If the file is missing, lacks the two columns, or
            assigns a learner twice.
    """
    source = Path(path).expanduser()
    try:
        with source.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"learner_id", "profile"} <= set(reader.fieldnames or ()):
                raise IngestionError(
                    "assignment CSV needs columns learner_id and profile",
                    code="io_schema_mismatch",
                    context={"path": str(source)},
                )
            mapping: dict[str, str] = {}
            for row in reader:
                learner, profile = row["learner_id"], row["profile"]
                if not learner or not profile:
                    continue
                if mapping.setdefault(learner, profile) != profile:
                    raise IngestionError(
                        f"learner {learner!r} is assigned to two profiles",
                        code="io_schema_mismatch",
                        context={"path": str(source)},
                    )
    except FileNotFoundError:
        raise IngestionError(
            "assignment CSV not found",
            code="io_not_found",
            context={"path": str(source)},
        ) from None
    return mapping


# ---------------------------------------------------------------------------
# auto / provided
# ---------------------------------------------------------------------------


def fit_profiles(
    dataset: Dataset,
    *,
    mode: Literal["auto", "provided"] = "auto",
    n_profiles: int = 3,
    assignments: Mapping[str, str] | None = None,
    generator: str = "semi_markov",
    hyperparameters: Mapping[str, Any] | None = None,
    names: Mapping[str, str] | None = None,
    min_learners: int = 5,
    seed: int = 0,
) -> ProfileSet:
    """Learn profiles from data and fit one generator model per profile.

    Args:
        dataset: Sessionized training data (use the training split only).
        mode: ``auto`` clusters learners; ``provided`` uses ``assignments``.
        n_profiles: Number of clusters for ``auto``.
        assignments: Learner id to profile name for ``provided``; learners
            without an entry are left out.
        generator: Generator family fitted per profile.
        hyperparameters: Generator hyperparameters.
        names: Renames, e.g. ``{"profile_1": "steady"}``.
        min_learners: Smallest allowed profile (training learners).
        seed: Seed for clustering.

    Raises:
        ConfigError: If arguments are invalid or a profile is too small.
        FitError: If fitting a profile's model fails.
    """
    session_sequences(dataset)
    if (
        isinstance(min_learners, bool)
        or not isinstance(min_learners, int)
        or min_learners < 1
    ):
        raise ConfigError(
            "min_learners must be a positive integer", code="config_invalid_value"
        )
    feature_names, vectors = learner_features(dataset)
    if not vectors:
        raise FitError("dataset has no learners with sessions", code="fit_empty_corpus")
    clustering: dict[str, Any] | None = None
    if mode == "auto":
        learners = sorted(vectors)
        scaler = Standardizer.fit([vectors[i] for i in learners])
        scaled = [scaler.transform(vectors[i]) for i in learners]
        result = kmeans(scaled, n_profiles, rng=make_rng(seed, "profiles", "kmeans"))
        clusters: dict[int, list[str]] = {}
        for learner, label in zip(learners, result.labels, strict=True):
            clusters.setdefault(label, []).append(learner)
        order = sorted(clusters, key=lambda c: (-len(clusters[c]), result.centroids[c]))
        groups = {f"profile_{rank + 1}": clusters[c] for rank, c in enumerate(order)}
        clustering = {
            "method": "kmeans++",
            "features": feature_names,
            "tokens": [
                n.removeprefix("share:")
                for n in feature_names
                if n.startswith("share:")
            ],
            "means": list(scaler.means),
            "stds": list(scaler.stds),
            "centroids": [list(result.centroids[c]) for c in order],
            "inertia": result.inertia,
            "seed": seed,
        }
    elif mode == "provided":
        if not assignments:
            raise ConfigError(
                "mode 'provided' needs assignments (learner_id -> profile)",
                code="config_invalid_value",
            )
        groups = {}
        for learner in sorted(vectors):
            if learner in assignments:
                groups.setdefault(str(assignments[learner]), []).append(learner)
        groups = dict(sorted(groups.items()))
        if not groups:
            raise ConfigError(
                "no learner in the data has a provided profile",
                code="config_invalid_value",
            )
    else:
        raise ConfigError(
            f"unknown profile mode {mode!r}",
            code="config_invalid_value",
            context={"allowed": ["auto", "provided", "manual"]},
        )

    renamed = _rename(groups, names)
    small = {name: len(ids) for name, ids in renamed.items() if len(ids) < min_learners}
    if small:
        raise ConfigError(
            f"profiles with fewer than {min_learners} learners: "
            + ", ".join(f"{k} ({v})" for k, v in small.items())
            + "; lower n_profiles, merge profiles, or lower min_learners",
            code="config_invalid_value",
            context={"profiles": small},
        )
    total = sum(len(ids) for ids in renamed.values())
    overall = feature_means(vectors, sorted(vectors))
    scaler_all = Standardizer.fit([vectors[i] for i in sorted(vectors)])
    family = get_generator(generator)
    profiles = []
    for name, ids in renamed.items():
        subset = _subset(dataset, set(ids), name)
        try:
            model = family.fit(subset, hyperparameters)
        except FitError as exc:
            raise FitError(
                f"profile {name!r}: {exc.message}",
                code=exc.code,
                context={"profile": name},
            ) from exc
        profiles.append(
            Profile(
                name=name,
                model=model,
                share=len(ids) / total,
                n_learners=len(ids),
                description=_describe(
                    feature_names, feature_means(vectors, ids), overall, scaler_all
                ),
            )
        )
    return ProfileSet(mode, generator, tuple(profiles), clustering)


def _rename(
    groups: dict[str, list[str]], names: Mapping[str, str] | None
) -> dict[str, list[str]]:
    if not names:
        return groups
    unknown = sorted(set(names) - set(groups))
    if unknown:
        raise ConfigError(
            f"names refer to unknown profiles: {', '.join(unknown)}",
            code="config_invalid_value",
        )
    renamed = {names.get(k, k): v for k, v in groups.items()}
    if len(renamed) != len(groups) or not all(
        isinstance(k, str) and k for k in renamed
    ):
        raise ConfigError(
            "profile names must be unique, non-empty strings",
            code="config_invalid_value",
        )
    return renamed


def _subset(dataset: Dataset, learners: set[str], name: str) -> Dataset:
    return Dataset(
        dataset_id=f"{dataset.dataset_id}[profile={name}]",
        events=tuple(
            e
            for e in dataset.events
            if e.learner_id in learners and e.session_id is not None
        ),
        sessions=tuple(s for s in dataset.sessions or () if s.learner_id in learners),
        metadata=dataset.metadata,
    )


def _describe(
    names: list[str], means: list[float], overall: list[float], scaler: Standardizer
) -> dict[str, Any]:
    z = [(m - o) / s for m, o, s in zip(means, overall, scaler.stds, strict=True)]
    ranked = sorted(range(len(names)), key=lambda i: -abs(z[i]))[:3]
    return {
        "mean_sessions": means[0],
        "mean_session_length": means[1],
        "mean_duration_min": math.expm1(means[2]) / 60,
        "success_rate": means[3],
        "token_shares": {
            n.removeprefix("share:"): v
            for n, v in zip(names, means, strict=True)
            if n.startswith("share:")
        },
        "distinguishing": [
            {"feature": names[i], "z": z[i]} for i in ranked if abs(z[i]) >= 0.25
        ],
    }


# ---------------------------------------------------------------------------
# manual
# ---------------------------------------------------------------------------


def define_profiles(definitions: Mapping[str, Mapping[str, Any]]) -> ProfileSet:
    """Build profiles from behavioural definitions, without real data.

    Each definition needs ``start`` (token -> probability) and
    ``transitions`` (token -> token -> probability); optional keys:
    ``share`` (default equal), ``session_length`` (``{mean}``,
    ``{fixed}``, or ``{counts}``; default mean 6), ``sessions_per_learner``
    (same forms; default mean 3), ``dwell_s`` (token or ``"*"`` -> median
    seconds; default 60), ``dwell_spread`` (lognormal sigma, default 0.5),
    ``activities`` (token -> list of activity ids), ``course``.
    Probabilities are normalised per row.

    Raises:
        ConfigError: If a definition lacks behaviour or a value is invalid.
    """
    if not isinstance(definitions, Mapping) or not definitions:
        raise ConfigError(
            "manual profiles need at least one definition", code="config_invalid_value"
        )
    profiles = []
    shares = {}
    for name, spec in definitions.items():
        if not isinstance(name, str) or not name:
            raise ConfigError(
                "profile names must be non-empty strings", code="config_invalid_value"
            )
        if not isinstance(spec, Mapping):
            raise _bad(name, "must be a mapping")
        unknown = sorted(set(spec) - _MANUAL_KEYS)
        if unknown:
            raise ConfigError(
                f"profile {name}: unknown keys {', '.join(unknown)}",
                code="config_unknown_key",
                context={"allowed": sorted(_MANUAL_KEYS)},
            )
        if "start" not in spec or "transitions" not in spec:
            raise _bad(
                name,
                "needs behavioural parameters (start and transitions), "
                "not only a share",
            )
        share = spec.get("share", 1.0)
        if isinstance(share, bool) or not isinstance(share, int | float) or share < 0:
            raise _bad(name, "share must be a non-negative number")
        shares[name] = float(share)
        profiles.append((name, _compile(name, spec)))
    total = sum(shares.values())
    if total <= 0:
        raise ConfigError(
            "profile shares must not all be zero", code="config_invalid_value"
        )
    return ProfileSet(
        "manual",
        "semi_markov",
        tuple(
            Profile(
                name=name,
                model=model,
                share=shares[name] / total,
                n_learners=0,
                description={"defined": True},
            )
            for name, model in profiles
        ),
    )


def _compile(name: str, spec: Mapping[str, Any]) -> GeneratorModel:
    start = _probabilities(name, "start", spec["start"])
    transitions_raw = spec["transitions"]
    if not isinstance(transitions_raw, Mapping) or not transitions_raw:
        raise _bad(name, "transitions must map tokens to next-token probabilities")
    rows = {
        token: _probabilities(name, f"transitions.{token}", row)
        for token, row in transitions_raw.items()
    }
    vocabulary = sorted(
        set(start) | set(rows) | {t for row in rows.values() for t in row}
    )
    counts = [{"context": [START], "next": _counts(start)}]
    counts += [
        {"context": [token], "next": _counts(row)}
        for token, row in sorted(rows.items())
    ]
    unigram: Counter[str] = Counter()
    for row in counts:
        unigram.update(row["next"])
    for token in vocabulary:
        unigram.setdefault(token, 1)

    dwell = spec.get("dwell_s", {"*": 60.0})
    spread = spec.get("dwell_spread", 0.5)
    if not isinstance(dwell, Mapping) or not all(_positive(v) for v in dwell.values()):
        raise _bad(name, "dwell_s must map tokens (or '*') to positive seconds")
    if not _positive(spread):
        raise _bad(name, "dwell_spread must be positive")
    unknown_dwell = sorted(set(dwell) - set(vocabulary) - {"*"})
    if unknown_dwell:
        raise _bad(name, f"dwell_s names unknown tokens: {', '.join(unknown_dwell)}")
    pooled = float(dwell.get("*", 60.0))

    activities = spec.get("activities", {})
    if not isinstance(activities, Mapping):
        raise _bad(name, "activities must map tokens to lists of activity ids")
    companions = {}
    for token in vocabulary:
        values = activities.get(token, [token])
        if (
            isinstance(values, str)
            or not values
            or not all(isinstance(v, str) and v for v in values)
        ):
            raise _bad(name, f"activities.{token} must be a non-empty list of ids")
        companions[token] = dict.fromkeys(values, 1)
    hyperparameters = SemiMarkovGenerator().validate_hyperparameters(
        {"timing_family": "lognormal"}
    )
    parameters = {
        "population": {
            "tokenization": "event_type",
            "companion_field": "activity_id",
            "companions": companions,
            "courses": {str(spec.get("course") or ""): 1},
            "sessions_per_learner": _per_learner(
                name, spec.get("sessions_per_learner", {"mean": 3})
            ),
            "session_gaps_s": [],
            "time_origin": "2026-01-05T08:00:00+00:00",
            "time_span_s": 28 * 86_400.0,
            "n_sessions": 0,
            "n_learners": 0,
        },
        "length": _length(name, spec.get("session_length", {"mean": 6})),
        "order": 1,
        "transitions": {"1": {"order": 1, "vocabulary": vocabulary, "rows": counts}},
        "unigram": dict(sorted(unigram.items())),
        "timing": {
            "pooled": {
                "family": "lognormal",
                "mu": math.log(pooled),
                "sigma": float(spread),
            },
            "per_token": {
                token: {
                    "family": "lognormal",
                    "mu": math.log(float(seconds)),
                    "sigma": float(spread),
                }
                for token, seconds in sorted(dwell.items())
                if token != "*"
            },
        },
        "manual_definition": json.loads(json.dumps(spec, default=str)),
    }
    return GeneratorModel(
        generator_id="semi_markov",
        generator_version=SemiMarkovGenerator.version,
        hyperparameters=hyperparameters,
        vocabulary=tuple(vocabulary),
        parameters=parameters,
        training_fingerprint=fingerprint(
            {"manual_profile": name, "definition": parameters["manual_definition"]}
        ),
    )


def _probabilities(name: str, key: str, row: Any) -> dict[str, float]:
    if (
        not isinstance(row, Mapping)
        or not row
        or not all(isinstance(t, str) and t for t in row)
    ):
        raise _bad(name, f"{key} must map tokens to probabilities")
    if (
        not all(_number(v) and v >= 0 for v in row.values())
        or not sum(row.values()) > 0
    ):
        raise _bad(
            name, f"{key} probabilities must be non-negative with a positive sum"
        )
    if START in row:
        raise _bad(name, f"token {START!r} is reserved")
    total = float(sum(row.values()))
    return {t: v / total for t, v in row.items() if v > 0}


def _counts(row: Mapping[str, float]) -> dict[str, int]:
    return {t: max(1, round(p * _SCALE)) for t, p in sorted(row.items())}


def _length(name: str, spec: Any) -> dict[str, Any]:
    mode, value = _single(name, "session_length", spec)
    if mode == "fixed":
        return {"model": "fixed", "length": value}
    if mode == "mean":
        return {"model": "poisson", "mean": float(value)}
    return {
        "model": "empirical",
        "counts": {str(k): float(v) for k, v in value.items()},
    }


def _per_learner(name: str, spec: Any) -> dict[str, float]:
    mode, value = _single(name, "sessions_per_learner", spec)
    if mode == "fixed":
        return {str(value): 1.0}
    if mode == "mean":
        low = max(1, math.floor(value))
        high = low + 1
        frac = max(0.0, min(1.0, float(value) - low))
        return {str(low): 1.0 - frac, str(high): frac} if frac else {str(low): 1.0}
    return {str(k): float(v) for k, v in value.items()}


def _single(name: str, key: str, spec: Any) -> tuple[str, Any]:
    if (
        not isinstance(spec, Mapping)
        or len(spec) != 1
        or next(iter(spec)) not in ("mean", "fixed", "counts")
    ):
        raise _bad(
            name, f"{key} must be {{mean: x}}, {{fixed: n}}, or {{counts: {{n: w}}}}"
        )
    mode, value = next(iter(spec.items()))
    if mode == "fixed" and (
        isinstance(value, bool) or not isinstance(value, int) or value < 1
    ):
        raise _bad(name, f"{key}.fixed must be a positive integer")
    if mode == "mean" and (not _number(value) or value < 1):
        raise _bad(name, f"{key}.mean must be at least 1")
    if mode == "counts":
        if not isinstance(value, Mapping) or not value:
            raise _bad(name, f"{key}.counts must map sizes to weights")
        for k, v in value.items():
            if not str(k).isdigit() or int(k) < 1 or not _number(v) or v < 0:
                raise _bad(
                    name,
                    f"{key}.counts must map positive sizes to non-negative weights",
                )
    return mode, value


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileResult:
    """Output of :func:`generate_profiles`.

    Attributes:
        dataset: The mixed synthetic dataset.
        annotations: One ``profile`` row per synthetic learner.
        allocation: Per profile: requested learner share, sessions, learners.
    """

    dataset: SyntheticDataset
    annotations: Annotations
    allocation: Mapping[str, Mapping[str, Any]] = field(hash=False)


def generate_profiles(
    profiles: ProfileSet,
    n_sessions: int,
    seed: int,
    *,
    mixture: Mapping[str, float] | None = None,
    controls: Mapping[str, Controls] | None = None,
    calendar: SessionCalendar | None = None,
    id_strategy: Literal["remap", "preserve"] = "remap",
) -> ProfileResult:
    """Sample a population mixing the profiles.

    Args:
        profiles: The profile set.
        n_sessions: Total sessions to generate.
        seed: Seed; each profile derives its own.
        mixture: Profile name to share of synthetic **learners** (default:
            the profile shares). Normalised; unnamed profiles get 0.
        controls: Per-profile controls (see :class:`Controls`).
        calendar: Calendar applied to every profile.
        id_strategy: ``remap`` shuffles all ids after mixing.

    Raises:
        ConfigError: If the mixture or controls name unknown profiles, or
            profiles use different tokenizations.
        GenerationError: If sampling fails.
    """
    if id_strategy not in ("remap", "preserve"):
        raise ConfigError(
            "unknown id strategy",
            code="config_invalid_value",
            context={"allowed": ["remap", "preserve"]},
        )
    names = profiles.names
    shares = (
        dict(mixture)
        if mixture is not None
        else {p.name: p.share for p in profiles.profiles}
    )
    unknown = sorted(set(shares) | set(controls or {}))
    unknown = [n for n in unknown if n not in names]
    if unknown:
        raise ConfigError(
            f"unknown profiles: {', '.join(unknown)}",
            code="config_invalid_value",
            context={"profiles": names},
        )
    if (
        any(not _number(v) or v < 0 for v in shares.values())
        or sum(shares.values()) <= 0
    ):
        raise ConfigError(
            "mixture shares must be non-negative with a positive sum",
            code="config_invalid_value",
        )
    total_share = sum(shares.values())
    learner_share = {n: shares.get(n, 0.0) / total_share for n in names}
    tokenizations = {
        p.model.parameters["population"]["tokenization"] for p in profiles.profiles
    }
    if len(tokenizations) > 1:
        raise ConfigError(
            "profiles use different tokenizations", code="config_invalid_value"
        )

    models = {
        p.name: apply_controls(p.model, (controls or {}).get(p.name, Controls()))
        for p in profiles.profiles
    }
    per_learner = {n: _mean_sessions(models[n]) for n in names}
    weights = {n: learner_share[n] * per_learner[n] for n in names}
    allocation = _largest_remainder(weights, n_sessions)

    events: list[LogRecord] = []
    sessions: list[Session] = []
    rows: list[Annotation] = []
    summary: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        count = allocation[name]
        summary[name] = {
            "learner_share": learner_share[name],
            "sessions": count,
            "learners": 0,
        }
        if not count:
            continue
        model = models[name]
        with _quiet_preserve():
            part = get_generator(model.generator_id).generate(
                model,
                count,
                derive_seed(seed, "profiles", name) or 0,
                id_strategy="preserve",
                calendar=calendar,
            )
        prefix = f"P{index + 1}-"
        learners = sorted(part.learner_ids)
        summary[name]["learners"] = len(learners)
        events += [
            replace(
                e,
                event_id=prefix + e.event_id,
                learner_id=prefix + e.learner_id,
                session_id=prefix + (e.session_id or ""),
            )
            for e in part.events
        ]
        sessions += [
            replace(
                s, session_id=prefix + s.session_id, learner_id=prefix + s.learner_id
            )
            for s in part.sessions or ()
        ]
        rows += [
            Annotation(
                level="learner",
                id=prefix + learner,
                annotation="profile",
                type="profile",
                value=name,
                parameters={"mode": profiles.mode},
            )
            for learner in learners
        ]
    mixed = SyntheticDataset(
        dataset_id="profiles-synthetic",
        events=tuple(events),
        sessions=tuple(sessions),
        metadata=DatasetMetadata(
            source_description=(
                f"synthetic mixture of {len(names)} {profiles.mode} profiles"
            ),
            privacy_notes=(
                "Synthetic data with generated identifiers; "
                "check privacy indicators before sharing."
            ),
            seeds={"generate": seed},
        ),
        generation=GenerationMetadata(
            generator_id=f"profiles:{profiles.generator_id}",
            model_fingerprint=profiles.fingerprint(),
            seed=seed,
            n_sessions=n_sessions,
            id_strategy=id_strategy,
        ),
    )
    annotations = Annotations(tuple(rows))
    if id_strategy == "preserve":
        logging.getLogger(__name__).warning(
            "id_strategy=preserve keeps profile-prefixed ids; do not share this output"
        )
    elif id_strategy == "remap":
        mixed, mapping = remap_ids_with_mapping(
            mixed, derive_seed(seed, "profiles", "ids")
        )
        annotations = annotations.remap({"learner": mapping.learners})
    return ProfileResult(dataset=mixed, annotations=annotations, allocation=summary)


@contextmanager
def _quiet_preserve() -> Iterator[None]:
    # Parts are generated with preserved ids and remapped (or warned about)
    # once after mixing, so the per-part privacy warning would be noise.
    policy = logging.getLogger("eduloggen.privacy.policies")
    level = policy.level
    policy.setLevel(logging.ERROR)
    try:
        yield
    finally:
        policy.setLevel(level)


def _mean_sessions(model: GeneratorModel) -> float:
    counts = {
        int(k): float(v)
        for k, v in model.parameters["population"]["sessions_per_learner"].items()
    }
    total = sum(counts.values())
    return sum(k * v for k, v in counts.items()) / total if total else 1.0


def _largest_remainder(weights: Mapping[str, float], total: int) -> dict[str, int]:
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ConfigError(
            "n_sessions must be a positive integer", code="config_invalid_value"
        )
    weight_sum = sum(weights.values()) or 1.0
    raw = {n: total * w / weight_sum for n, w in weights.items()}
    base = {n: math.floor(v) for n, v in raw.items()}
    rest = total - sum(base.values())
    for name in sorted(raw, key=lambda n: (-(raw[n] - base[n]), n))[:rest]:
        base[name] += 1
    return base


def _number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _positive(value: Any) -> bool:
    return _number(value) and value > 0


def _fmt(value: Any) -> str:
    return "-" if value is None else f"{value:.3g}"


def _bad(name: str, reason: str) -> ConfigError:
    return ConfigError(
        f"profile {name}: {reason}",
        code="config_invalid_value",
        context={"profile": name},
    )
