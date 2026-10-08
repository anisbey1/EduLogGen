"""Exception hierarchy for EduLogGen.

Every error raised by EduLogGen derives from :class:`EduLogGenError`. Each
exception carries a human-readable message, a stable machine-readable ``code``
for CLI and automation handling, and an optional ``context`` mapping with
non-sensitive details such as paths or field names.

Context values must never contain raw learner data or other PII payloads.

The hierarchy follows the architecture document (SAD §18.1, ADR-016)::

    EduLogGenError
    ├── ConfigError
    ├── SchemaError
    ├── IoError
    │   ├── IngestionError
    │   └── ExportError
    ├── AnalysisError
    ├── FitError
    ├── GenerationError
    ├── ValidationError
    ├── BenchmarkError
    ├── PluginError
    └── InternalError
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar

__all__ = [
    "AnalysisError",
    "BenchmarkError",
    "ConfigError",
    "EduLogGenError",
    "ExportError",
    "FitError",
    "GenerationError",
    "IngestionError",
    "InternalError",
    "IoError",
    "PluginError",
    "SchemaError",
    "ValidationError",
]


class EduLogGenError(Exception):
    """Root of all EduLogGen exceptions.

    Attributes:
        message: Human-readable description of the failure.
        code: Stable machine-readable error code (e.g. ``"config_error"``).
        context: Read-only mapping of non-sensitive diagnostic details.
    """

    default_code: ClassVar[str] = "eduloggen_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the error.

        Args:
            message: Human-readable description of the failure.
            code: Optional machine-readable code overriding the class default.
            context: Optional non-sensitive diagnostic details.
        """
        super().__init__(message)
        self.message = message
        self.code = code if code is not None else self.default_code
        self.context: Mapping[str, Any] = MappingProxyType(dict(context or {}))

    def __str__(self) -> str:
        """Return the message prefixed with the machine code."""
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return a debug representation including code and context."""
        return (
            f"{type(self).__name__}({self.message!r}, code={self.code!r}, "
            f"context={dict(self.context)!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for structured logs and CLI JSON output.

        Returns:
            Mapping with ``error``, ``code``, ``message``, and ``context`` keys.
        """
        return {
            "error": type(self).__name__,
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }


class ConfigError(EduLogGenError):
    """Invalid configuration: bad YAML, wrong types, or unknown keys."""

    default_code = "config_error"


class SchemaError(EduLogGenError):
    """Data does not match the canonical schema (missing fields, bad dtypes)."""

    default_code = "schema_error"


class IoError(EduLogGenError):
    """Reading or writing data failed (missing files, permissions, format)."""

    default_code = "io_error"


class IngestionError(IoError):
    """Reading source logs into the canonical model failed."""

    default_code = "ingestion_error"


class ExportError(IoError):
    """Writing datasets, artifacts, or reports failed."""

    default_code = "export_error"


class AnalysisError(EduLogGenError):
    """Sessionization or behavioral analysis failed."""

    default_code = "analysis_error"


class FitError(EduLogGenError):
    """Fitting a generator failed (insufficient data, order too high)."""

    default_code = "fit_error"


class GenerationError(EduLogGenError):
    """Sampling synthetic data failed (bad seed or n, incompatible model)."""

    default_code = "generation_error"


class ValidationError(EduLogGenError):
    """Validation API was misused.

    Metric threshold failures are reported as statuses in the validation
    report, not raised as this exception (SAD §18.3).
    """

    default_code = "validation_error"


class BenchmarkError(EduLogGenError):
    """Running a benchmark protocol failed."""

    default_code = "benchmark_error"


class PluginError(EduLogGenError):
    """Plugin discovery, registration, or contract check failed."""

    default_code = "plugin_error"


class InternalError(EduLogGenError):
    """An internal invariant was violated; indicates a bug in EduLogGen."""

    default_code = "internal_error"
