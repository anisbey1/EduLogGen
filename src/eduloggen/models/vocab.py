"""Controlled ``event_type`` vocabulary (PRD §9.6).

v1.0 ships a default vocabulary that configuration may extend. Event types
outside the vocabulary are handled by an :class:`UnknownEventPolicy`; the
default is to preserve them as opaque tokens.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import StrEnum

from eduloggen.core import DEFAULT_EVENT_TYPES, OTHER_EVENT_TYPE, SchemaError
from eduloggen.models import _checks as chk

__all__ = ["EventVocabulary", "UnknownEventPolicy"]


class UnknownEventPolicy(StrEnum):
    """How to treat event types outside the vocabulary."""

    PRESERVE = "preserve"
    """Keep the original token unchanged (default)."""

    MAP_TO_OTHER = "map_to_other"
    """Replace the token with ``"other"``."""

    REJECT = "reject"
    """Raise :class:`~eduloggen.core.SchemaError`."""


@dataclass(frozen=True, slots=True)
class EventVocabulary:
    """A set of known event types plus a policy for unknown ones.

    Attributes:
        types: Known event type tokens; always includes ``"other"``.
        unknown_policy: Treatment of tokens outside ``types``.
    """

    types: frozenset[str] = DEFAULT_EVENT_TYPES
    unknown_policy: UnknownEventPolicy = UnknownEventPolicy.PRESERVE

    def __post_init__(self) -> None:
        """Validate tokens and policy, and ensure ``"other"`` is present.

        Raises:
            SchemaError: If a token is not a non-empty string.
            ValueError: If the policy is not a valid policy name.
        """
        types = frozenset(self.types)
        for token in types:
            chk.require_str("EventVocabulary", "types", token)
        object.__setattr__(self, "types", types | {OTHER_EVENT_TYPE})
        object.__setattr__(
            self, "unknown_policy", UnknownEventPolicy(self.unknown_policy)
        )

    def __contains__(self, event_type: object) -> bool:
        """Return whether ``event_type`` is a known token."""
        return event_type in self.types

    def extend(self, types: Iterable[str]) -> EventVocabulary:
        """Return a vocabulary with additional known types.

        Args:
            types: Tokens to add.

        Returns:
            New vocabulary with the same policy.
        """
        return replace(self, types=self.types | frozenset(types))

    def normalize(self, event_type: str) -> str:
        """Apply the unknown-type policy to a token.

        Args:
            event_type: Raw event type token.

        Returns:
            The token itself if known or preserved, otherwise ``"other"``.

        Raises:
            SchemaError: If the token is unknown and the policy is ``REJECT``.
        """
        if event_type in self.types:
            return event_type
        if self.unknown_policy is UnknownEventPolicy.MAP_TO_OTHER:
            return OTHER_EVENT_TYPE
        if self.unknown_policy is UnknownEventPolicy.REJECT:
            raise SchemaError(
                "event_type is not in the controlled vocabulary",
                code=chk.INVALID_VALUE,
                context={"record": "LogRecord", "field": "event_type"},
            )
        return event_type
