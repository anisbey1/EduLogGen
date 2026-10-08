"""Derived measurements produced by analysis extractors (SAD §9.7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from eduloggen.core import JSONValue
from eduloggen.models import _checks as chk

__all__ = ["Feature", "FeatureScope"]


class FeatureScope(StrEnum):
    """Level of the domain a feature describes."""

    EVENT = "event"
    SESSION = "session"
    LEARNER = "learner"
    CORPUS = "corpus"


@dataclass(frozen=True, slots=True, kw_only=True)
class Feature:
    """A named, typed measurement over events, sessions, learners, or a corpus.

    Attributes:
        name: Stable feature identifier (e.g. ``"session_length"``).
        scope: Level the feature describes.
        value: JSON-serializable value.
        dtype: Logical type of ``value`` (e.g. ``"int"``, ``"float"``).
        params: Read-only extractor parameters used to compute the value.
    """

    name: str
    scope: FeatureScope
    value: JSONValue
    dtype: str
    params: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), hash=False
    )

    def __post_init__(self) -> None:
        """Validate fields and normalize the scope.

        Raises:
            SchemaError: If a field is invalid or the scope is unknown.
        """
        chk.require_str("Feature", "name", self.name)
        chk.require_str("Feature", "dtype", self.dtype)
        try:
            scope = FeatureScope(self.scope)
        except ValueError:
            raise chk.invalid("Feature", "scope", "is not a known scope") from None
        object.__setattr__(self, "scope", scope)
        object.__setattr__(
            self, "params", chk.freeze_mapping("Feature", "params", self.params)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible mapping."""
        return {
            "name": self.name,
            "scope": self.scope.value,
            "value": self.value,
            "dtype": self.dtype,
            "params": dict(self.params),
        }
