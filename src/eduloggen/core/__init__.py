"""Foundation layer shared by every EduLogGen package.

``core`` holds the exception hierarchy, framework-wide constants, the
:class:`RunContext`, and shared type aliases. It depends only on the standard
library and must never import other EduLogGen subpackages (SAD §7.2.1).
"""

from __future__ import annotations

from eduloggen.core.constants import (
    ARTIFACT_VERSION,
    DEFAULT_EVENT_TYPES,
    EVENT_REQUIRED_FIELDS,
    OTHER_EVENT_TYPE,
    REPORT_VERSION,
    SCHEMA_VERSION,
    SEED_SCHEME_VERSION,
    SESSION_REQUIRED_FIELDS,
)
from eduloggen.core.context import RunContext, current_platform
from eduloggen.core.exceptions import (
    AnalysisError,
    BenchmarkError,
    ConfigError,
    EduLogGenError,
    ExportError,
    FitError,
    GenerationError,
    IngestionError,
    InternalError,
    IoError,
    PluginError,
    SchemaError,
    ValidationError,
)
from eduloggen.core.typing import JSONScalar, JSONValue, PathLike, Seed

__all__ = [
    "ARTIFACT_VERSION",
    "DEFAULT_EVENT_TYPES",
    "EVENT_REQUIRED_FIELDS",
    "OTHER_EVENT_TYPE",
    "REPORT_VERSION",
    "SCHEMA_VERSION",
    "SEED_SCHEME_VERSION",
    "SESSION_REQUIRED_FIELDS",
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
    "JSONScalar",
    "JSONValue",
    "PathLike",
    "PluginError",
    "RunContext",
    "SchemaError",
    "Seed",
    "ValidationError",
    "current_platform",
]
