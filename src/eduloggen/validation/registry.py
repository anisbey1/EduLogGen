"""Metric registry (PRD §12.5, FR-V.8)."""

from __future__ import annotations

from eduloggen.core import ConfigError, PluginError
from eduloggen.validation.base import BaseMetric
from eduloggen.validation.metrics import BUILTIN_METRICS

__all__ = [
    "DEFAULT_METRICS",
    "available_metrics",
    "get_metric",
    "register_metric",
    "unregister_metric",
]

_REGISTRY: dict[str, BaseMetric] = {}

DEFAULT_METRICS: tuple[str, ...] = tuple(metric.name for metric in BUILTIN_METRICS)
"""Metrics computed when none are requested."""


def register_metric(metric: BaseMetric, *, replace: bool = False) -> None:
    """Make a metric instance available by its ``name``.

    Raises:
        PluginError: If the object is not a metric, its name is invalid, or
            the name is taken (without ``replace``).
    """
    if not isinstance(metric, BaseMetric):
        raise PluginError(
            "metrics must be BaseMetric instances", code="plugin_contract_violation"
        )
    name = getattr(metric, "name", None)
    if not isinstance(name, str) or not name.isidentifier():
        raise PluginError(
            "metric names must be identifiers",
            code="plugin_invalid_name",
            context={"name": repr(name)},
        )
    if name in _REGISTRY and not replace:
        raise PluginError(
            f"metric {name!r} is already registered",
            code="plugin_duplicate",
            context={"name": name},
        )
    _REGISTRY[name] = metric


def unregister_metric(name: str) -> None:
    """Remove a registration (no-op if absent)."""
    _REGISTRY.pop(name, None)


def available_metrics() -> list[str]:
    """Registered metric names, sorted."""
    return sorted(_REGISTRY)


def get_metric(name: str) -> BaseMetric:
    """Look up a registered metric.

    Raises:
        ConfigError: If no metric has that name.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ConfigError(
            f"unknown metric {name!r}",
            code="metric_unknown",
            context={"name": name, "available": available_metrics()},
        ) from None


for _builtin in BUILTIN_METRICS:
    register_metric(_builtin())
