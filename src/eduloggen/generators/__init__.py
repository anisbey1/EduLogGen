"""Synthetic session generators (PRD §11, SAD §12).

Typical use::

    from eduloggen.generators import get_generator

    generator = get_generator("semi_markov")
    model = generator.fit(sessionized, {"order": 2, "smoothing_alpha": 0.1})
    generator.save(model, "models/semi_markov")
    synthetic = generator.generate(model, n_sessions=1000, seed=42)

Built-ins: ``markov`` (order-k chains), ``semi_markov`` (adds per-token
timing), and ``independent`` (i.i.d. baseline). Register others with
:func:`register_generator`.
"""

from __future__ import annotations

from eduloggen.generators.artifact import describe_markdown, load_model, save_model
from eduloggen.generators.base import BaseGenerator, SequenceSampler
from eduloggen.generators.calendar import Deadline, SessionCalendar
from eduloggen.generators.distributions import (
    Categorical,
    DurationSampler,
    LengthSampler,
    fit_duration,
    fit_length,
)
from eduloggen.generators.markov import MarkovGenerator
from eduloggen.generators.registry import (
    BUILTIN_GENERATORS,
    GeneratorFactory,
    available_generators,
    get_generator,
    register_generator,
    unregister_generator,
)
from eduloggen.generators.semi_markov import SemiMarkovGenerator
from eduloggen.generators.statistical import IndependentGenerator

__all__ = [
    "BUILTIN_GENERATORS",
    "BaseGenerator",
    "Categorical",
    "Deadline",
    "DurationSampler",
    "GeneratorFactory",
    "IndependentGenerator",
    "LengthSampler",
    "MarkovGenerator",
    "SemiMarkovGenerator",
    "SequenceSampler",
    "SessionCalendar",
    "available_generators",
    "describe_markdown",
    "fit_duration",
    "fit_length",
    "get_generator",
    "load_model",
    "register_generator",
    "save_model",
    "unregister_generator",
]
