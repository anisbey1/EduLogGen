"""Behavioural controls applied to a fitted model (Level 2, design §5).

Controls never modify the original model: :func:`apply_controls` returns a
new :class:`~eduloggen.models.GeneratorModel` whose parameters record the
controls, so its fingerprint differs and provenance stays exact.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

from eduloggen.core import ConfigError
from eduloggen.models import GeneratorModel

__all__ = ["Controls", "apply_controls"]

_KEYS: Final = frozenset(
    {"event_weights", "dwell_scale", "session_length", "sessions_per_learner"}
)


def _empty() -> Mapping[str, Any]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class Controls:
    """Requested changes to a model's behaviour.

    Attributes:
        event_weights: Token to multiplier on the probability of moving into
            it (renormalised per row); ``0`` removes the token.
        dwell_scale: Token (or ``"*"`` for all) to multiplier on the time
            spent before the next event.
        session_length: ``{"scale": x}`` or ``{"fixed": n}``.
        sessions_per_learner: ``{"mean": m}`` or ``{"fixed": n}``.
    """

    event_weights: Mapping[str, float] = field(default_factory=_empty, hash=False)
    dwell_scale: Mapping[str, float] = field(default_factory=_empty, hash=False)
    session_length: Mapping[str, float] = field(default_factory=_empty, hash=False)
    sessions_per_learner: Mapping[str, float] = field(
        default_factory=_empty, hash=False
    )

    def __post_init__(self) -> None:
        """Validate values.

        Raises:
            ConfigError: If a value is out of range or a mode is unknown.
        """
        for key, minimum in (("event_weights", 0.0), ("dwell_scale", None)):
            values = getattr(self, key)
            if not isinstance(values, Mapping):
                raise _bad(key, "must be a mapping of token to number")
            for token, value in values.items():
                if not isinstance(token, str) or not token:
                    raise _bad(key, "keys must be tokens")
                if not _number(value) or (value < 0 if minimum == 0.0 else value <= 0):
                    raise _bad(
                        f"{key}.{token}",
                        (
                            "must be a non-negative number"
                            if minimum == 0.0
                            else "must be positive"
                        ),
                    )
            object.__setattr__(
                self, key, MappingProxyType({k: float(v) for k, v in values.items()})
            )
        for key, modes in (
            ("session_length", ("scale", "fixed")),
            ("sessions_per_learner", ("mean", "fixed")),
        ):
            spec = getattr(self, key)
            if not isinstance(spec, Mapping):
                raise _bad(key, f"must be {{{modes[0]}: x}} or {{{modes[1]}: n}}")
            if spec:
                if len(spec) != 1 or next(iter(spec)) not in modes:
                    raise _bad(key, f"must be {{{modes[0]}: x}} or {{{modes[1]}: n}}")
                mode, value = next(iter(spec.items()))
                if mode == "fixed" and (
                    isinstance(value, bool) or not isinstance(value, int) or value < 1
                ):
                    raise _bad(f"{key}.fixed", "must be a positive integer")
                if mode != "fixed" and (not _number(value) or value <= 0):
                    raise _bad(f"{key}.{mode}", "must be a positive number")
            object.__setattr__(self, key, MappingProxyType(dict(spec)))

    @property
    def is_empty(self) -> bool:
        """Whether no control is set."""
        return not (
            self.event_weights
            or self.dwell_scale
            or self.session_length
            or self.sessions_per_learner
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize for :meth:`from_dict`, omitting empty parts."""
        data = {
            "event_weights": dict(self.event_weights),
            "dwell_scale": dict(self.dwell_scale),
            "session_length": dict(self.session_length),
            "sessions_per_learner": dict(self.sessions_per_learner),
        }
        return {key: value for key, value in data.items() if value}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> Controls:
        """Build from a ``controls:`` section.

        Raises:
            ConfigError: If keys are unknown or values invalid.
        """
        if data is None:
            return cls()
        if not isinstance(data, Mapping):
            raise _bad("controls", "must be a mapping")
        unknown = sorted(set(data) - _KEYS)
        if unknown:
            raise ConfigError(
                f"unknown controls: {', '.join(unknown)}",
                code="config_unknown_key",
                context={"keys": unknown, "allowed": sorted(_KEYS)},
            )
        return cls(**{key: value or {} for key, value in data.items()})


def apply_controls(model: GeneratorModel, controls: Controls) -> GeneratorModel:
    """Return a copy of ``model`` with ``controls`` applied.

    Raises:
        ConfigError: If a control names a token the model does not know, or
            the weights would remove every token.
    """
    if controls.is_empty:
        return model
    vocabulary = set(model.vocabulary)
    for key in ("event_weights", "dwell_scale"):
        unknown = sorted(
            set(getattr(controls, key))
            - vocabulary
            - ({"*"} if key == "dwell_scale" else set())
        )
        if unknown:
            raise ConfigError(
                f"{key} names tokens the model does not know: {', '.join(unknown)}",
                code="config_invalid_value",
                context={
                    "key": f"controls.{key}",
                    "tokens": unknown,
                    "vocabulary": sorted(vocabulary),
                },
            )
    params: dict[str, Any] = json.loads(json.dumps(model.parameters))
    if controls.event_weights:
        weights = dict(params.get("token_weights", {}))
        for token, value in controls.event_weights.items():
            weights[token] = weights.get(token, 1.0) * value
        if all(weights.get(token, 1.0) == 0 for token in vocabulary):
            raise ConfigError(
                "event_weights remove every token",
                code="config_invalid_value",
                context={"key": "controls.event_weights"},
            )
        params["token_weights"] = weights
    if controls.dwell_scale:
        scale = dict(params.get("timing_scale", {}))
        for token, value in controls.dwell_scale.items():
            scale[token] = scale.get(token, 1.0) * value
        params["timing_scale"] = scale
    if controls.session_length:
        params["length"] = _scale_length(params["length"], controls.session_length)
    if controls.sessions_per_learner:
        population = params["population"]
        population["sessions_per_learner"] = _scale_counts(
            population["sessions_per_learner"], controls.sessions_per_learner
        )
    params["controls"] = [*params.get("controls", []), controls.to_dict()]
    return GeneratorModel(
        generator_id=model.generator_id,
        generator_version=model.generator_version,
        hyperparameters=model.hyperparameters,
        vocabulary=model.vocabulary,
        parameters=params,
        training_fingerprint=model.training_fingerprint,
        eduloggen_version=model.eduloggen_version,
        artifact_version=model.artifact_version,
        fitted_at=model.fitted_at,
    )


def _scale_length(
    length: Mapping[str, Any], spec: Mapping[str, float]
) -> dict[str, Any]:
    mode, value = next(iter(spec.items()))
    if mode == "fixed":
        return {"model": "fixed", "length": int(value)}
    if length["model"] == "fixed":
        return {"model": "fixed", "length": max(1, round(length["length"] * value))}
    if length["model"] == "poisson":
        return {"model": "poisson", "mean": max(1.0, float(length["mean"]) * value)}
    return {
        "model": "empirical",
        "counts": _rescale({int(k): v for k, v in length["counts"].items()}, value),
    }


def _scale_counts(
    counts: Mapping[str, Any], spec: Mapping[str, float]
) -> dict[str, Any]:
    mode, value = next(iter(spec.items()))
    if mode == "fixed":
        return {str(int(value)): 1}
    current = {int(k): float(v) for k, v in counts.items()}
    total = sum(current.values())
    mean = sum(k * v for k, v in current.items()) / total if total else 1.0
    return _rescale(current, value / mean)


def _rescale(counts: Mapping[int, float], factor: float) -> dict[str, float]:
    rescaled: dict[int, float] = {}
    for size, weight in counts.items():
        new = max(1, round(size * factor))
        rescaled[new] = rescaled.get(new, 0.0) + weight
    return {str(k): v for k, v in sorted(rescaled.items())}


def _number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _bad(key: str, reason: str) -> ConfigError:
    return ConfigError(
        f"controls.{key} {reason}",
        code="config_invalid_value",
        context={"key": f"controls.{key}"},
    )
