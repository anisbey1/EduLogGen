"""Fitted generator state (PRD §9.4, SAD §9.13).

A :class:`GeneratorModel` holds everything a generator needs to sample:
hyperparameters, vocabulary, and fitted parameters, all as JSON-compatible
data so artifacts stay inspectable (ADR-009). It holds no raw training rows.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from eduloggen.__version__ import __version__
from eduloggen.core import ARTIFACT_VERSION, SchemaError
from eduloggen.models import _checks as chk

__all__ = ["GeneratorModel"]


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorModel:
    """A fitted generator.

    Attributes:
        generator_id: Registered generator name (e.g. ``"markov"``).
        generator_version: Version of the generator implementation.
        hyperparameters: Validated hyperparameters used for fitting.
        vocabulary: Sorted session tokens known to the model.
        parameters: Family-specific fitted parameters (JSON-compatible).
        training_fingerprint: Hash of training data and configuration.
        eduloggen_version: Framework version that fitted the model.
        artifact_version: Artifact layout version.
        fitted_at: UTC time of fitting.
    """

    generator_id: str
    generator_version: str
    hyperparameters: Mapping[str, Any] = field(hash=False)
    vocabulary: tuple[str, ...]
    parameters: Mapping[str, Any] = field(hash=False)
    training_fingerprint: str
    eduloggen_version: str = __version__
    artifact_version: str = ARTIFACT_VERSION
    fitted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate identifiers and check that payloads are JSON-compatible.

        Raises:
            SchemaError: If a field is invalid or a payload is not JSON data.
        """
        name = "GeneratorModel"
        for attr in (
            "generator_id",
            "generator_version",
            "training_fingerprint",
            "eduloggen_version",
            "artifact_version",
        ):
            chk.require_str(name, attr, getattr(self, attr))
        object.__setattr__(
            self, "fitted_at", chk.require_utc(name, "fitted_at", self.fitted_at)
        )
        vocabulary = tuple(self.vocabulary)
        for token in vocabulary:
            chk.require_str(name, "vocabulary", token)
        object.__setattr__(self, "vocabulary", vocabulary)
        for attr in ("hyperparameters", "parameters"):
            value = getattr(self, attr)
            if not isinstance(value, Mapping):
                raise chk.invalid(name, attr, "must be a mapping")
            try:
                frozen = json.loads(_encode(value))
            except (TypeError, ValueError):
                raise chk.invalid(name, attr, "must be JSON-serializable") from None
            object.__setattr__(self, attr, frozen)

    def fingerprint(self) -> str:
        """Hash of the model content, excluding ``fitted_at``.

        Two fits of the same data with the same configuration and versions
        share a fingerprint.
        """
        payload = self.to_dict()
        del payload["fitted_at"]
        return "sha256:" + hashlib.sha256(_encode(payload).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "generator_id": self.generator_id,
            "generator_version": self.generator_version,
            "hyperparameters": json.loads(_encode(self.hyperparameters)),
            "vocabulary": list(self.vocabulary),
            "parameters": json.loads(_encode(self.parameters)),
            "training_fingerprint": self.training_fingerprint,
            "eduloggen_version": self.eduloggen_version,
            "artifact_version": self.artifact_version,
            "fitted_at": self.fitted_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GeneratorModel:
        """Restore a model serialized with :meth:`to_dict`.

        Raises:
            SchemaError: If keys are missing or unknown, or values are invalid.
        """
        fields = frozenset(cls.__dataclass_fields__)
        chk.check_keys("GeneratorModel", data, tuple(sorted(fields)), fields)
        values = dict(data)
        values["fitted_at"] = chk.parse_utc(
            "GeneratorModel", "fitted_at", values["fitted_at"]
        )
        if not isinstance(values["vocabulary"], list | tuple):
            raise chk.invalid("GeneratorModel", "vocabulary", "must be a list")
        return cls(**values)

    def check_artifact_version(self) -> None:
        """Ensure this release can read the artifact layout.

        Raises:
            SchemaError: If the major artifact version differs.
        """
        major = self.artifact_version.partition(".")[0]
        if major != ARTIFACT_VERSION.partition(".")[0]:
            raise SchemaError(
                f"artifact_version {self.artifact_version} is incompatible "
                f"with {ARTIFACT_VERSION}",
                code=chk.UNSUPPORTED_VERSION,
                context={"artifact_version": self.artifact_version},
            )


def _encode(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
