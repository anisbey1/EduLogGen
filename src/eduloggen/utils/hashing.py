"""Canonical JSON encoding and content fingerprints."""

from __future__ import annotations

import hashlib
import json
from typing import Any

__all__ = ["canonical_json", "fingerprint"]


def canonical_json(value: Any) -> str:
    """Encode ``value`` as compact JSON with sorted keys.

    Raises:
        TypeError: If ``value`` contains non-JSON types.
        ValueError: If ``value`` contains NaN or infinity.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    """Return ``"sha256:<hex>"`` of the canonical JSON encoding of ``value``."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return "sha256:" + digest
