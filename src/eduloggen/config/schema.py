"""Typed, immutable run configuration (PRD §16, SAD §16).

:class:`AppConfig` holds one frozen dataclass per section. Plain data (parsed
YAML, TOML, or JSON) is converted with :meth:`AppConfig.from_dict`, which
checks every value against the field's type hint and rejects unknown keys
unless ``allow_unknown`` is set.

The ``generator`` section is open: any key other than ``name`` is a
hyperparameter passed to the generator, which validates it.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Literal, Union, get_args, get_origin, get_type_hints

from eduloggen.core import ConfigError
from eduloggen.models import TokenField

__all__ = [
    "AnalysisConfig",
    "AppConfig",
    "BenchmarkConfig",
    "GenerationConfig",
    "GeneratorConfig",
    "IoConfig",
    "LoggingConfig",
    "PrivacyConfig",
    "ProjectConfig",
    "SessionizationConfig",
    "ValidationConfig",
    "VisualizationConfig",
]

logger = logging.getLogger(__name__)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
TableFormat = Literal["csv", "tsv", "jsonl", "parquet"]
DuplicatePolicy = Literal["keep_first", "error"]
UnknownEventPolicy = Literal["preserve", "map_to_other", "reject"]
SessionStrategy = Literal["explicit", "idle_timeout", "composite"]
IdStrategy = Literal["remap", "preserve"]
FigureFormat = Literal["png", "svg", "pdf"]


def _empty() -> Mapping[str, Any]:
    return types.MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class ProjectConfig:
    """Run identity and output location.

    Attributes:
        name: Human-readable experiment name.
        output_dir: Directory for run outputs, relative to the config file.
    """

    name: str = "eduloggen"
    output_dir: str = "runs"


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggingConfig:
    """Logging behaviour of CLI runs.

    Attributes:
        level: Minimum level emitted.
        json: Emit structured JSON log lines.
    """

    level: LogLevel = "INFO"
    json: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class IoConfig:
    """Source data, mapping, and corpus output.

    Attributes:
        input: Source log file, relative to the config file.
        format: Reader name (built-in ``csv``, ``tsv``, ``jsonl``,
            ``parquet``, or a plugin), or ``"auto"`` to use the extension.
        mapping: Field mapping file (YAML/TOML/JSON).
        output: Corpus directory to write.
        output_format: Table format for written corpora.
        strict: Abort ingest on the first invalid row.
        on_duplicate: Handling of repeated ``event_id`` values.
        event_types: Event types added to the default vocabulary.
        unknown_event_policy: Handling of types outside the vocabulary.
    """

    input: str | None = None
    format: str = "auto"
    mapping: str | None = None
    output: str | None = None
    output_format: TableFormat = "csv"
    strict: bool = False
    on_duplicate: DuplicatePolicy = "keep_first"
    event_types: tuple[str, ...] = ()
    unknown_event_policy: UnknownEventPolicy = "preserve"


@dataclass(frozen=True, slots=True, kw_only=True)
class SessionizationConfig:
    """How events are grouped into sessions.

    Attributes:
        strategy: ``explicit`` uses source session ids, ``idle_timeout``
            splits on inactivity, ``composite`` uses ids when present and
            falls back to the timeout.
        idle_timeout_s: Inactivity gap that starts a new session, in seconds.
        tokenization: Event attribute used as the session sequence token.
    """

    strategy: SessionStrategy = "composite"
    idle_timeout_s: float = 1800.0
    tokenization: TokenField = "event_type"

    def __post_init__(self) -> None:
        """Check value ranges."""
        if self.idle_timeout_s <= 0:
            raise _invalid("sessionization.idle_timeout_s", "must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalysisConfig:
    """Descriptive analysis options.

    Attributes:
        ngram_order: Longest n-gram counted in sequence statistics.
    """

    ngram_order: int = 2

    def __post_init__(self) -> None:
        """Check value ranges."""
        if self.ngram_order < 1:
            raise _invalid("analysis.ngram_order", "must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class GeneratorConfig:
    """Generator choice and hyperparameters.

    Attributes:
        name: Registered generator name (e.g. ``"markov"``).
        params: Generator hyperparameters; validated by the generator.
    """

    name: str = "markov"
    params: Mapping[str, Any] = field(default_factory=_empty, hash=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationConfig:
    """Sampling options.

    Attributes:
        n_sessions: Number of synthetic sessions to generate.
        seed: Global seed; ``None`` means non-deterministic.
        id_strategy: ``remap`` assigns fresh synthetic ids; ``preserve``
            keeps source ids (local experiments only).
    """

    n_sessions: int = 100
    seed: int | None = None
    id_strategy: IdStrategy = "remap"

    def __post_init__(self) -> None:
        """Check value ranges."""
        if self.n_sessions < 1:
            raise _invalid("generation.n_sessions", "must be positive")
        if self.seed is not None and self.seed < 0:
            raise _invalid("generation.seed", "must be non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationConfig:
    """Real-versus-synthetic comparison options.

    Attributes:
        metrics: Metric names to compute; empty means the default set.
        thresholds: Metric name to pass/fail threshold.
    """

    metrics: tuple[str, ...] = ()
    thresholds: Mapping[str, float] = field(default_factory=_empty, hash=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualizationConfig:
    """Figure output options.

    Attributes:
        formats: Image formats to save.
        dpi: Resolution of raster images.
        output_dir: Figure directory; defaults to the run output directory.
    """

    formats: tuple[FigureFormat, ...] = ("png",)
    dpi: int = 150
    output_dir: str | None = None

    def __post_init__(self) -> None:
        """Check value ranges."""
        if not self.formats:
            raise _invalid("visualization.formats", "must not be empty")
        if self.dpi < 1:
            raise _invalid("visualization.dpi", "must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class BenchmarkConfig:
    """Benchmark protocol options.

    Attributes:
        protocol: Versioned protocol name.
        generators: Generators to compare.
        repeats: Seeds per generator.
    """

    protocol: str = "session_fidelity_v1"
    generators: tuple[str, ...] = ("markov", "semi_markov")
    repeats: int = 3

    def __post_init__(self) -> None:
        """Check value ranges."""
        if not self.generators:
            raise _invalid("benchmark.generators", "must not be empty")
        if self.repeats < 1:
            raise _invalid("benchmark.repeats", "must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class PrivacyConfig:
    """Privacy policies applied across the pipeline.

    Attributes:
        strip_metadata_keys: Event metadata keys removed before export.
    """

    strip_metadata_keys: tuple[str, ...] = ()


_SECTIONS: Final[dict[str, type[Any]]] = {
    "project": ProjectConfig,
    "logging": LoggingConfig,
    "io": IoConfig,
    "sessionization": SessionizationConfig,
    "analysis": AnalysisConfig,
    "generator": GeneratorConfig,
    "generation": GenerationConfig,
    "validation": ValidationConfig,
    "visualization": VisualizationConfig,
    "benchmark": BenchmarkConfig,
    "privacy": PrivacyConfig,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class AppConfig:
    """Complete, validated configuration of a run.

    Attributes:
        base_dir: Directory relative paths are resolved against (the config
            file's directory). Not part of the fingerprint or ``to_dict``.
    """

    project: ProjectConfig = field(default_factory=ProjectConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    io: IoConfig = field(default_factory=IoConfig)
    sessionization: SessionizationConfig = field(default_factory=SessionizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    base_dir: Path | None = field(default=None, compare=False)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        allow_unknown: bool = False,
        base_dir: Path | None = None,
    ) -> AppConfig:
        """Validate plain data into a configuration.

        Args:
            data: Section name to section mapping. Missing sections and keys
                take their defaults.
            allow_unknown: Ignore unknown keys (with a warning) instead of
                failing.
            base_dir: Directory for resolving relative paths.

        Returns:
            The validated configuration.

        Raises:
            ConfigError: If a key is unknown (and not allowed) or a value has
                the wrong type or is out of range.
        """
        if not isinstance(data, Mapping):
            raise ConfigError("configuration must be a mapping", code="config_invalid")
        sections: dict[str, Any] = {}
        for name, value in data.items():
            if name not in _SECTIONS:
                _unknown(str(name), allow_unknown)
                continue
            if value is None:
                value = {}
            if not isinstance(value, Mapping):
                raise _invalid(str(name), "must be a mapping")
            sections[name] = _build_section(name, value, allow_unknown)
        return cls(**sections, base_dir=base_dir)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to plain data accepted by :meth:`from_dict`."""
        result: dict[str, Any] = {}
        for name in _SECTIONS:
            section = getattr(self, name)
            values = {f.name: _plain(getattr(section, f.name)) for f in fields(section)}
            if name == "generator":
                values = {"name": values["name"], **values["params"]}
            result[name] = values
        return result

    def resolve_path(self, path: str) -> Path:
        """Resolve a configured path against :attr:`base_dir`.

        Absolute paths are returned unchanged; relative ones are joined to
        ``base_dir`` (or the working directory when there is none).
        """
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.base_dir or Path.cwd()) / candidate


# ---------------------------------------------------------------------------
# Conversion from plain data
# ---------------------------------------------------------------------------


def _build_section(name: str, data: Mapping[str, Any], allow_unknown: bool) -> Any:
    cls = _SECTIONS[name]
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    values: dict[str, Any] = {}
    params: dict[str, Any] = {}
    for key, raw in data.items():
        where = f"{name}.{key}"
        if cls is GeneratorConfig and key not in known:
            params[str(key)] = raw
        elif key not in known or (cls is GeneratorConfig and key == "params"):
            _unknown(where, allow_unknown)
        else:
            values[key] = _convert(raw, hints[key], where)
    if cls is GeneratorConfig:
        values["params"] = types.MappingProxyType(params)
    return cls(**values)


def _convert(value: Any, hint: Any, where: str) -> Any:
    origin = get_origin(hint)
    args = get_args(hint)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (inner,) = [arg for arg in args if arg is not type(None)]
        return _convert(value, inner, where)
    if origin is Literal:
        if value not in args:
            choices = ", ".join(repr(arg) for arg in args)
            raise _invalid(where, f"must be one of {choices}")
        return value
    if origin is tuple:
        if isinstance(value, str) or not isinstance(value, list | tuple):
            raise _invalid(where, "must be a list")
        return tuple(
            _convert(item, args[0], f"{where}[{i}]") for i, item in enumerate(value)
        )
    if origin is Mapping:
        if not isinstance(value, Mapping):
            raise _invalid(where, "must be a mapping")
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise _invalid(where, "must have string keys")
            converted[key] = _convert(item, args[1], f"{where}.{key}")
        return types.MappingProxyType(converted)
    if hint is Any:
        return value
    if hint is bool:
        if not isinstance(value, bool):
            raise _invalid(where, "must be true or false")
        return value
    if hint is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _invalid(where, "must be an integer")
        return value
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise _invalid(where, "must be a number")
        return float(value)
    if hint is str:
        if not isinstance(value, str) or not value:
            raise _invalid(where, "must be a non-empty string")
        return value
    raise ConfigError(  # pragma: no cover - guards future schema additions
        f"unsupported config type for {where}", code="config_internal"
    )


def _plain(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    return value


def _unknown(where: str, allow_unknown: bool) -> None:
    if allow_unknown:
        logger.warning("ignoring unknown config key %s", where)
        return
    raise ConfigError(
        f"unknown config key {where}",
        code="config_unknown_key",
        context={"key": where},
    )


def _invalid(where: str, reason: str) -> ConfigError:
    return ConfigError(
        f"{where} {reason}", code="config_invalid_value", context={"key": where}
    )
