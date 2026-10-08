"""Plugin discovery and registration (PRD §10, SAD §13).

Extensions come from three sources, in this order: built-ins, installed
packages declaring ``eduloggen.plugins.<kind>`` entry points
(:func:`discover_plugins`), and explicit :func:`register_plugin` calls.
Kinds: ``generator``, ``metric``, ``reader``, ``visualizer``,
``benchmark_suite``. See ``docs/plugins.md`` for a plugin author guide.
"""

from __future__ import annotations

from eduloggen.plugins.registry import (
    ENTRY_POINT_PREFIX,
    KINDS,
    DiscoveryResult,
    PluginInfo,
    PluginKind,
    discover_plugins,
    list_plugins,
    register_plugin,
    unregister_plugin,
)

__all__ = [
    "ENTRY_POINT_PREFIX",
    "KINDS",
    "DiscoveryResult",
    "PluginInfo",
    "PluginKind",
    "discover_plugins",
    "list_plugins",
    "register_plugin",
    "unregister_plugin",
]
