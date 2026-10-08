"""Shared type aliases used across EduLogGen packages."""

from __future__ import annotations

import os
from typing import TypeAlias

__all__ = ["JSONScalar", "JSONValue", "PathLike", "Seed"]

PathLike: TypeAlias = str | os.PathLike[str]
"""A filesystem path given as a string or path-like object."""

Seed: TypeAlias = int | None
"""A random seed; ``None`` means non-deterministic."""

JSONScalar: TypeAlias = str | int | float | bool | None
"""A JSON scalar value."""

JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
"""Any JSON-serializable value."""
