"""Level 2 experimental generation (design: ``docs/04_EXPERIMENTAL_GENERATOR.md``).

Available so far: labelled anomaly injection (M1); controls, calendars, and
the manipulation check (M2); behavioural profiles and mixtures (M3). Anomaly injection::

    from eduloggen.scenarios import inject_anomalies

    result = inject_anomalies(
        synthetic,
        [{"type": "event_frequency", "rate": 0.05, "count": 25, "window_s": 180},
         {"type": "unexpected_transition", "rate": 0.02}],
        seed=1,
    )
    result.dataset       # events only, ids remapped
    result.annotations   # ground truth, kept separate

Anomalies describe data patterns, not intentions; they are never evidence of
misconduct.
"""

from __future__ import annotations

from eduloggen.scenarios.anomalies import (
    ANOMALIES,
    BUILTIN_ANOMALIES,
    AbnormalTiming,
    BaseAnomaly,
    EventFrequency,
    Inactivity,
    Injection,
    InjectionContext,
    Repetition,
    UnexpectedTransition,
    available_anomalies,
    get_anomaly,
    register_anomaly,
)
from eduloggen.scenarios.check import CheckItem, ManipulationCheck, manipulation_check
from eduloggen.scenarios.controls import Controls, apply_controls
from eduloggen.scenarios.experiment import (
    ExperimentResult,
    ExperimentSettings,
    run_experiment,
)
from eduloggen.scenarios.inject import (
    AnomalySpec,
    InjectionResult,
    inject_anomalies,
    load_anomaly_specs,
)
from eduloggen.scenarios.profiles import (
    Profile,
    ProfileMode,
    ProfileResult,
    ProfileSet,
    define_profiles,
    fit_profiles,
    generate_profiles,
    load_profile_assignments,
    load_profiles,
)

__all__ = [
    "ANOMALIES",
    "BUILTIN_ANOMALIES",
    "AbnormalTiming",
    "AnomalySpec",
    "BaseAnomaly",
    "CheckItem",
    "Controls",
    "EventFrequency",
    "ExperimentResult",
    "ExperimentSettings",
    "Inactivity",
    "Injection",
    "InjectionContext",
    "InjectionResult",
    "ManipulationCheck",
    "Profile",
    "ProfileMode",
    "ProfileResult",
    "ProfileSet",
    "Repetition",
    "UnexpectedTransition",
    "apply_controls",
    "available_anomalies",
    "define_profiles",
    "fit_profiles",
    "generate_profiles",
    "get_anomaly",
    "inject_anomalies",
    "load_anomaly_specs",
    "load_profile_assignments",
    "load_profiles",
    "manipulation_check",
    "register_anomaly",
    "run_experiment",
]
