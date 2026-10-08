"""Generator registry (SAD §12.6, FR-G.8).

Built-ins are registered at import. Third parties call
:func:`register_generator` (directly or from a plugin entry point) with a
zero-argument factory returning a :class:`BaseGenerator`.
"""

from __future__ import annotations

from collections.abc import Callable

from eduloggen.core import ConfigError, PluginError
from eduloggen.generators.base import BaseGenerator
from eduloggen.generators.markov import MarkovGenerator
from eduloggen.generators.semi_markov import SemiMarkovGenerator
from eduloggen.generators.statistical import IndependentGenerator

__all__ = [
    "GeneratorFactory",
    "available_generators",
    "get_generator",
    "register_generator",
    "unregister_generator",
]

GeneratorFactory = Callable[[], BaseGenerator]

_REGISTRY: dict[str, GeneratorFactory] = {}


def register_generator(
    name: str, factory: GeneratorFactory, *, replace: bool = False
) -> None:
    """Make a generator available under ``name``.

    Args:
        name: Registry name used in configs and the CLI.
        factory: Zero-argument callable returning a generator instance.
        replace: Allow overriding an existing registration.

    Raises:
        PluginError: If the name is invalid or taken (without ``replace``),
            or the factory is not callable.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise PluginError(
            "generator names must be identifiers (letters, digits, underscores)",
            code="plugin_invalid_name",
            context={"name": repr(name)},
        )
    if not callable(factory):
        raise PluginError("factory must be callable", code="plugin_invalid_factory")
    if name in _REGISTRY and not replace:
        raise PluginError(
            f"generator {name!r} is already registered",
            code="plugin_duplicate",
            context={"name": name},
        )
    _REGISTRY[name] = factory


def unregister_generator(name: str) -> None:
    """Remove a registration (no-op if absent)."""
    _REGISTRY.pop(name, None)


def available_generators() -> list[str]:
    """Registered generator names, sorted."""
    return sorted(_REGISTRY)


def get_generator(name: str) -> BaseGenerator:
    """Instantiate a registered generator.

    Raises:
        ConfigError: If no generator has that name.
        PluginError: If the factory fails or returns a non-generator.
    """
    factory = _REGISTRY.get(name)
    if factory is None:
        raise ConfigError(
            f"unknown generator {name!r}",
            code="generator_unknown",
            context={"name": name, "available": available_generators()},
        )
    try:
        generator = factory()
    except Exception as exc:
        raise PluginError(
            f"factory for generator {name!r} failed: {exc}",
            code="plugin_factory_failed",
            context={"name": name},
        ) from exc
    if not isinstance(generator, BaseGenerator):
        raise PluginError(
            f"factory for {name!r} did not return a BaseGenerator",
            code="plugin_contract_violation",
            context={"name": name},
        )
    return generator


for _builtin in (MarkovGenerator, SemiMarkovGenerator, IndependentGenerator):
    register_generator(_builtin.name, _builtin)
