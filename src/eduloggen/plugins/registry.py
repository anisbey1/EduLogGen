"""Plugin registration and entry-point discovery (PRD §10, SAD §13).

Third-party packages declare entry points in the group
``eduloggen.plugins.<kind>``::

    [project.entry-points."eduloggen.plugins.generator"]
    my_gen = "my_package.generator:MyGenerator"

The entry point name becomes the plugin name. Discovery is explicit
(:func:`discover_plugins`; the CLI calls it at startup), deterministic
(sorted by kind, then name), and contract-checked before anything is
registered. Plugins never replace built-ins; a clash is reported as an error.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Final, Literal, get_args

from eduloggen.benchmark import (
    BUILTIN_PROTOCOLS,
    PROTOCOLS,
    BenchmarkProtocol,
    register_protocol,
    unregister_protocol,
)
from eduloggen.core import PluginError
from eduloggen.generators import (
    BUILTIN_GENERATORS,
    BaseGenerator,
    available_generators,
    register_generator,
    unregister_generator,
)
from eduloggen.io import (
    BUILTIN_READERS,
    BaseReader,
    available_readers,
    register_reader,
    unregister_reader,
)
from eduloggen.validation import (
    DEFAULT_METRICS,
    BaseMetric,
    available_metrics,
    register_metric,
    unregister_metric,
)
from eduloggen.visualization import (
    BUILTIN_PLOTS,
    DATASET_PLOTS,
    register_plot,
    unregister_plot,
)

__all__ = [
    "ENTRY_POINT_PREFIX",
    "DiscoveryResult",
    "PluginInfo",
    "PluginKind",
    "discover_plugins",
    "list_plugins",
    "register_plugin",
    "unregister_plugin",
]

logger = logging.getLogger(__name__)

PluginKind = Literal["generator", "metric", "reader", "visualizer", "benchmark_suite"]
KINDS: Final[tuple[PluginKind, ...]] = get_args(PluginKind)
ENTRY_POINT_PREFIX: Final = "eduloggen.plugins."


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """Metadata of one registered extension.

    Attributes:
        kind: Extension point.
        name: Registered name.
        source: ``builtin``, ``entry_point``, or ``api``.
        distribution: Installed package providing it (entry points only).
        version: That package's version.
        target: ``module:attribute`` the plugin was loaded from.
        tags: Capability tags (generators).
    """

    kind: PluginKind
    name: str
    source: Literal["builtin", "entry_point", "api"]
    distribution: str | None = None
    version: str | None = None
    target: str | None = None
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible data."""
        return {
            "kind": self.kind,
            "name": self.name,
            "source": self.source,
            "distribution": self.distribution,
            "version": self.version,
            "target": self.target,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """Outcome of :func:`discover_plugins`.

    Attributes:
        loaded: Plugins registered by this call.
        errors: ``(entry point, message)`` for plugins that were rejected.
    """

    loaded: tuple[PluginInfo, ...] = ()
    errors: tuple[tuple[str, str], ...] = field(default=())


_EXTERNAL: dict[tuple[str, str], PluginInfo] = {}
_LOADED_ENTRY_POINTS: set[tuple[str, str, str]] = set()


def register_plugin(
    kind: PluginKind,
    name: str,
    obj: Any,
    *,
    replace: bool = False,
    _info: PluginInfo | None = None,
) -> PluginInfo:
    """Register an extension after checking its contract.

    Args:
        kind: Extension point.
        name: Name used in configs and the CLI.
        obj: ``generator``: a :class:`BaseGenerator` subclass or factory;
            ``metric``: a :class:`BaseMetric` subclass or instance whose
            ``name`` matches; ``reader``: a :class:`BaseReader` subclass or
            factory (its ``suffixes`` attribute, if any, enables
            auto-detection); ``visualizer``: a function
            ``(real, synthetic_or_None) -> Figure``; ``benchmark_suite``: a
            :class:`BenchmarkProtocol` whose ``name`` matches.
        replace: Allow overriding a previous non-built-in registration.

    Returns:
        The plugin's metadata.

    Raises:
        PluginError: If the kind is unknown, the name clashes with a built-in
            or (without ``replace``) another plugin, or the object breaks
            the contract.
    """
    if kind not in KINDS:
        raise PluginError(
            f"unknown plugin kind {kind!r}",
            code="plugin_unknown_kind",
            context={"kind": kind, "available": list(KINDS)},
        )
    if not isinstance(name, str) or not name.isidentifier():
        raise PluginError(
            "plugin names must be identifiers",
            code="plugin_invalid_name",
            context={"name": repr(name)},
        )
    if name in _builtin_names(kind):
        raise PluginError(
            f"{kind} {name!r} is built in and cannot be replaced by a plugin",
            code="plugin_duplicate",
            context={"kind": kind, "name": name},
        )
    tags = _REGISTRARS[kind](name, obj, replace)
    info = _info or PluginInfo(kind=kind, name=name, source="api")
    info = PluginInfo(
        kind=info.kind,
        name=info.name,
        source=info.source,
        distribution=info.distribution,
        version=info.version,
        target=info.target,
        tags=tags,
    )
    _EXTERNAL[(kind, name)] = info
    logger.debug("registered %s plugin %s from %s", kind, name, info.source)
    return info


def unregister_plugin(kind: PluginKind, name: str) -> None:
    """Remove a non-built-in extension (no-op if absent or built in).

    The entry point it came from may be discovered again afterwards.
    """
    if name in _builtin_names(kind):
        return
    _UNREGISTRARS[kind](name)
    info = _EXTERNAL.pop((kind, name), None)
    if info is not None and info.target is not None:
        group = ENTRY_POINT_PREFIX + kind
        _LOADED_ENTRY_POINTS.discard((group, name, info.target))


def discover_plugins(
    *,
    strict: bool = False,
    entry_points_override: Iterable[EntryPoint] | None = None,
) -> DiscoveryResult:
    """Load and register plugins declared through entry points.

    Each entry point is processed at most once per process, so calling this
    repeatedly is cheap. Broken plugins are skipped and reported.

    Args:
        strict: Raise on the first broken plugin instead of skipping it.
        entry_points_override: Entry points to use instead of the installed
            ones (for tests and embedding).

    Returns:
        Plugins loaded by this call and errors encountered.

    Raises:
        PluginError: In strict mode, if a plugin fails to load or register.
    """
    candidates = (
        list(entry_points_override)
        if entry_points_override is not None
        else [
            ep for kind in KINDS for ep in entry_points(group=ENTRY_POINT_PREFIX + kind)
        ]
    )
    loaded: list[PluginInfo] = []
    errors: list[tuple[str, str]] = []
    for ep in sorted(candidates, key=lambda e: (e.group, e.name, e.value)):
        key = (ep.group, ep.name, ep.value)
        if key in _LOADED_ENTRY_POINTS:
            continue
        label = f"{ep.group}:{ep.name}"
        try:
            kind = _kind_of(ep.group)
            obj = ep.load()
            dist = ep.dist
            info = PluginInfo(
                kind=kind,
                name=ep.name,
                source="entry_point",
                distribution=dist.name if dist is not None else None,
                version=dist.version if dist is not None else None,
                target=ep.value,
            )
            loaded.append(register_plugin(kind, ep.name, obj, _info=info))
            _LOADED_ENTRY_POINTS.add(key)
        except Exception as exc:  # third-party code: report, do not crash
            error = (
                exc
                if isinstance(exc, PluginError)
                else PluginError(
                    f"could not load plugin: {type(exc).__name__}: {exc}",
                    code="plugin_load_failed",
                    context={"entry_point": label},
                )
            )
            if strict:
                raise error from exc
            logger.warning("skipping plugin %s: %s", label, error)
            errors.append((label, str(error)))
    return DiscoveryResult(loaded=tuple(loaded), errors=tuple(errors))


def list_plugins(kind: PluginKind | None = None) -> list[PluginInfo]:
    """All registered extensions (built-in and external), sorted by kind and name."""
    from eduloggen.generators import get_generator

    infos: list[PluginInfo] = []
    for current in KINDS if kind is None else (kind,):
        for name in _current_names(current):
            info = _EXTERNAL.get((current, name))
            if info is None:
                tags = (
                    tuple(sorted(get_generator(name).tags))
                    if current == "generator"
                    else ()
                )
                info = PluginInfo(kind=current, name=name, source="builtin", tags=tags)
            infos.append(info)
    return infos


# ---------------------------------------------------------------------------
# Per-kind contract checks
# ---------------------------------------------------------------------------


def _register_generator(name: str, obj: Any, replace: bool) -> tuple[str, ...]:
    if isinstance(obj, type) and not issubclass(obj, BaseGenerator):
        raise _contract("generator", name, "must subclass BaseGenerator")
    if not callable(obj):
        raise _contract("generator", name, "must be a BaseGenerator class or factory")
    instance = obj()
    if not isinstance(instance, BaseGenerator):
        raise _contract("generator", name, "factory must return a BaseGenerator")
    if instance.name != name:
        raise _contract("generator", name, f"its 'name' attribute is {instance.name!r}")
    register_generator(name, obj, replace=replace)
    return tuple(sorted(instance.tags))


def _register_metric(name: str, obj: Any, replace: bool) -> tuple[str, ...]:
    if isinstance(obj, type):
        if not issubclass(obj, BaseMetric):
            raise _contract("metric", name, "must subclass BaseMetric")
        obj = obj()
    if not isinstance(obj, BaseMetric):
        raise _contract("metric", name, "must be a BaseMetric subclass or instance")
    if obj.name != name:
        raise _contract("metric", name, f"its 'name' attribute is {obj.name!r}")
    if obj.direction not in ("lower_better", "higher_better"):
        raise _contract(
            "metric", name, "direction must be lower_better or higher_better"
        )
    register_metric(obj, replace=replace)
    return ()


def _register_reader(name: str, obj: Any, replace: bool) -> tuple[str, ...]:
    if isinstance(obj, type) and not issubclass(obj, BaseReader):
        raise _contract("reader", name, "must subclass BaseReader")
    if not callable(obj):
        raise _contract("reader", name, "must be a BaseReader class or factory")
    suffixes = tuple(getattr(obj, "suffixes", ()) or ())
    register_reader(name, obj, suffixes=suffixes, replace=replace)
    return ()


def _register_visualizer(name: str, obj: Any, replace: bool) -> tuple[str, ...]:
    if not callable(obj) or isinstance(obj, type):
        raise _contract("visualizer", name, "must be a plotting function")
    register_plot(name, obj, replace=replace)
    return ()


def _register_suite(name: str, obj: Any, replace: bool) -> tuple[str, ...]:
    if callable(obj) and not isinstance(obj, BenchmarkProtocol):
        obj = obj()
    if not isinstance(obj, BenchmarkProtocol):
        raise _contract("benchmark_suite", name, "must be a BenchmarkProtocol")
    if obj.name != name:
        raise _contract("benchmark_suite", name, f"its 'name' is {obj.name!r}")
    register_protocol(obj, replace=replace)
    return ()


_REGISTRARS: Final[dict[str, Callable[[str, Any, bool], tuple[str, ...]]]] = {
    "generator": _register_generator,
    "metric": _register_metric,
    "reader": _register_reader,
    "visualizer": _register_visualizer,
    "benchmark_suite": _register_suite,
}

_UNREGISTRARS: Final[dict[str, Callable[[str], None]]] = {
    "generator": unregister_generator,
    "metric": unregister_metric,
    "reader": unregister_reader,
    "visualizer": unregister_plot,
    "benchmark_suite": unregister_protocol,
}

_BUILTINS: Final[dict[str, frozenset[str]]] = {
    "generator": BUILTIN_GENERATORS,
    "metric": frozenset(DEFAULT_METRICS),
    "reader": BUILTIN_READERS,
    "visualizer": BUILTIN_PLOTS,
    "benchmark_suite": BUILTIN_PROTOCOLS,
}


def _builtin_names(kind: str) -> frozenset[str]:
    return _BUILTINS[kind]


def _current_names(kind: str) -> list[str]:
    if kind == "generator":
        return available_generators()
    if kind == "metric":
        return available_metrics()
    if kind == "reader":
        return available_readers()
    if kind == "visualizer":
        return sorted(DATASET_PLOTS)
    return sorted(PROTOCOLS)


def _kind_of(group: str) -> PluginKind:
    kind = group.removeprefix(ENTRY_POINT_PREFIX)
    if not group.startswith(ENTRY_POINT_PREFIX) or kind not in KINDS:
        raise PluginError(
            f"unsupported entry point group {group!r}",
            code="plugin_unknown_kind",
            context={
                "group": group,
                "available": [ENTRY_POINT_PREFIX + k for k in KINDS],
            },
        )
    return kind


def _contract(kind: str, name: str, reason: str) -> PluginError:
    return PluginError(
        f"{kind} plugin {name!r} {reason}",
        code="plugin_contract_violation",
        context={"kind": kind, "name": name},
    )
