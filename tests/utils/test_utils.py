"""Tests for seed derivation and hashing helpers."""

from __future__ import annotations

import math
from typing import Any

import pytest

from eduloggen.core import ConfigError
from eduloggen.utils import canonical_json, derive_seed, fingerprint, make_rng


def test_derive_seed_is_stable_and_label_sensitive() -> None:
    seed = derive_seed(42, "fit", "markov")
    assert seed == derive_seed(42, "fit", "markov")
    assert seed is not None and 0 <= seed < 2**63
    assert seed != derive_seed(42, "fit", "semi_markov")
    assert seed != derive_seed(43, "fit", "markov")
    assert derive_seed(42, "a", "b") != derive_seed(42, "ab")
    assert derive_seed(42, 1) != derive_seed(42, "1")
    assert derive_seed(42) != 42


def test_derive_seed_is_pinned() -> None:
    """Changing this value breaks reproducibility; bump SEED_SCHEME_VERSION."""
    assert derive_seed(42, "generate", 0) == 663994126890940873


def test_derive_seed_none() -> None:
    assert derive_seed(None, "fit") is None


@pytest.mark.parametrize("seed", [-1, True, 1.0, "1"])
def test_derive_seed_rejects_bad_seed(seed: Any) -> None:
    with pytest.raises(ConfigError):
        derive_seed(seed)


@pytest.mark.parametrize("label", [True, 1.5, None])
def test_derive_seed_rejects_bad_label(label: Any) -> None:
    with pytest.raises(ConfigError):
        derive_seed(1, label)


def test_make_rng() -> None:
    a = make_rng(7, "x")
    b = make_rng(7, "x")
    assert [a.random() for _ in range(3)] == [b.random() for _ in range(3)]
    assert make_rng(7, "y").random() != make_rng(7, "x").random()
    assert 0 <= make_rng(None).random() < 1


def test_canonical_json_and_fingerprint() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'
    assert fingerprint({"b": 1, "a": 2}) == fingerprint({"a": 2, "b": 1})
    assert fingerprint({"a": 1}).startswith("sha256:")
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})
    with pytest.raises(TypeError):
        canonical_json({"x": object()})
