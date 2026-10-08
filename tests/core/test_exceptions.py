"""Tests for the EduLogGen exception hierarchy."""

from __future__ import annotations

import pytest

from eduloggen.core import exceptions as exc


@pytest.mark.parametrize(
    ("error_cls", "parent", "code"),
    [
        (exc.ConfigError, exc.EduLogGenError, "config_error"),
        (exc.SchemaError, exc.EduLogGenError, "schema_error"),
        (exc.IoError, exc.EduLogGenError, "io_error"),
        (exc.IngestionError, exc.IoError, "ingestion_error"),
        (exc.ExportError, exc.IoError, "export_error"),
        (exc.AnalysisError, exc.EduLogGenError, "analysis_error"),
        (exc.FitError, exc.EduLogGenError, "fit_error"),
        (exc.GenerationError, exc.EduLogGenError, "generation_error"),
        (exc.ValidationError, exc.EduLogGenError, "validation_error"),
        (exc.BenchmarkError, exc.EduLogGenError, "benchmark_error"),
        (exc.PluginError, exc.EduLogGenError, "plugin_error"),
        (exc.InternalError, exc.EduLogGenError, "internal_error"),
    ],
)
def test_hierarchy_and_default_codes(
    error_cls: type[exc.EduLogGenError],
    parent: type[exc.EduLogGenError],
    code: str,
) -> None:
    assert issubclass(error_cls, parent)
    error = error_cls("boom")
    assert error.code == code
    assert error.message == "boom"
    assert isinstance(error, Exception)


def test_codes_are_unique() -> None:
    codes = [
        cls.default_code
        for cls in vars(exc).values()
        if isinstance(cls, type) and issubclass(cls, exc.EduLogGenError)
    ]
    assert len(codes) == len(set(codes))


def test_all_exports_are_errors() -> None:
    for name in exc.__all__:
        assert issubclass(getattr(exc, name), exc.EduLogGenError)


def test_code_override_and_str() -> None:
    error = exc.SchemaError("missing field", code="schema_missing_field")
    assert error.code == "schema_missing_field"
    assert str(error) == "[schema_missing_field] missing field"


def test_context_is_copied_and_read_only() -> None:
    source = {"field": "timestamp"}
    error = exc.SchemaError("bad", context=source)
    source["field"] = "changed"
    assert error.context["field"] == "timestamp"
    with pytest.raises(TypeError):
        error.context["field"] = "x"  # type: ignore[index]


def test_context_defaults_to_empty() -> None:
    assert dict(exc.FitError("x").context) == {}


def test_to_dict_and_repr() -> None:
    error = exc.ConfigError("unknown key", context={"key": "foo"})
    assert error.to_dict() == {
        "error": "ConfigError",
        "code": "config_error",
        "message": "unknown key",
        "context": {"key": "foo"},
    }
    assert repr(error) == (
        "ConfigError('unknown key', code='config_error', context={'key': 'foo'})"
    )


def test_catchable_via_root() -> None:
    with pytest.raises(exc.EduLogGenError) as info:
        raise exc.ExportError("disk full")
    assert info.value.code == "export_error"
