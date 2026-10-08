"""Tests for the event vocabulary and features."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import eduloggen.models
from eduloggen.core import DEFAULT_EVENT_TYPES, SchemaError
from eduloggen.models import EventVocabulary, Feature, FeatureScope, UnknownEventPolicy

MODELS_DIR = Path(eduloggen.models.__file__).parent


def test_default_vocabulary_preserves_unknown() -> None:
    vocab = EventVocabulary()
    assert vocab.types == DEFAULT_EVENT_TYPES
    assert vocab.unknown_policy is UnknownEventPolicy.PRESERVE
    assert "view" in vocab
    assert vocab.normalize("view") == "view"
    assert vocab.normalize("hint_request") == "hint_request"


def test_map_to_other_policy() -> None:
    vocab = EventVocabulary(unknown_policy=UnknownEventPolicy.MAP_TO_OTHER)
    assert vocab.normalize("hint_request") == "other"
    assert vocab.normalize("submit") == "submit"


def test_reject_policy_from_string() -> None:
    vocab = EventVocabulary(unknown_policy="reject")  # type: ignore[arg-type]
    assert vocab.unknown_policy is UnknownEventPolicy.REJECT
    with pytest.raises(SchemaError) as info:
        vocab.normalize("hint_request")
    assert "hint_request" not in str(info.value)


def test_custom_vocabulary_always_has_other() -> None:
    vocab = EventVocabulary(types=frozenset({"click"}))
    assert vocab.types == {"click", "other"}


def test_extend_keeps_policy() -> None:
    vocab = EventVocabulary(unknown_policy=UnknownEventPolicy.REJECT).extend(["hint"])
    assert "hint" in vocab
    assert vocab.unknown_policy is UnknownEventPolicy.REJECT


def test_vocabulary_rejects_bad_input() -> None:
    with pytest.raises(SchemaError):
        EventVocabulary(types=frozenset({""}))
    with pytest.raises(ValueError):
        EventVocabulary(unknown_policy="drop")  # type: ignore[arg-type]


def test_feature() -> None:
    feature = Feature(
        name="session_length",
        scope="session",  # type: ignore[arg-type]
        value=12,
        dtype="int",
        params={"unit": "events"},
    )
    assert feature.scope is FeatureScope.SESSION
    assert json.loads(json.dumps(feature.to_dict())) == {
        "name": "session_length",
        "scope": "session",
        "value": 12,
        "dtype": "int",
        "params": {"unit": "events"},
    }
    with pytest.raises(TypeError):
        feature.params["unit"] = "s"  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "field"),
    [({"name": ""}, "name"), ({"dtype": ""}, "dtype"), ({"scope": "team"}, "scope")],
)
def test_feature_rejects_invalid(overrides: dict[str, object], field: str) -> None:
    values: dict[str, object] = {
        "name": "n",
        "scope": FeatureScope.CORPUS,
        "value": 1.0,
        "dtype": "float",
    }
    values.update(overrides)
    with pytest.raises(SchemaError) as info:
        Feature(**values)  # type: ignore[arg-type]
    assert info.value.context["field"] == field


def test_models_imports_only_core() -> None:
    """``models`` may depend only on ``core`` (SAD §8.2)."""
    allowed = ("eduloggen.core", "eduloggen.models", "eduloggen.__version__")
    for path in MODELS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            elif isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            else:
                continue
            for name in names:
                if name.startswith("eduloggen"):
                    assert name.startswith(allowed), f"{path.name} imports {name}"
