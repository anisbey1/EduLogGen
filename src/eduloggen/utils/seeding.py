"""Deterministic seed derivation (SAD §29O, ADR-012).

Every random component derives its own seed from the global seed plus a
label path, e.g. ``derive_seed(42, "generate", "markov", 0)``. Derived seeds
are independent of call order and of how many other components draw random
numbers, so adding a new stage never changes the output of an existing one.

The derivation is fixed for ``SEED_SCHEME_VERSION``; changing it requires a
new scheme version.
"""

from __future__ import annotations

import hashlib
import random

from eduloggen.core import SEED_SCHEME_VERSION, ConfigError

__all__ = ["derive_seed", "make_rng"]

_MASK_63 = (1 << 63) - 1


def derive_seed(seed: int | None, *labels: str | int) -> int | None:
    """Derive a component seed from a global seed and a label path.

    Args:
        seed: Global seed, or ``None`` for non-deterministic runs.
        *labels: Path identifying the component, e.g. ``"fit", "markov"``.

    Returns:
        A non-negative 63-bit seed, or ``None`` if ``seed`` is ``None``.

    Raises:
        ConfigError: If ``seed`` is negative or not an integer, or a label is
            not a string or integer.
    """
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigError(
            "seed must be a non-negative integer", code="config_invalid_value"
        )
    parts = [f"v{SEED_SCHEME_VERSION}", str(seed)]
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, str | int):
            raise ConfigError(
                "seed labels must be strings or integers",
                code="config_invalid_value",
            )
        parts.append(f"{type(label).__name__}:{label}")
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _MASK_63


def make_rng(seed: int | None, *labels: str | int) -> random.Random:
    """Return a :class:`random.Random` seeded from a derived seed.

    With ``seed=None`` the generator is seeded from OS entropy.
    """
    return random.Random(derive_seed(seed, *labels))
