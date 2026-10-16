# Use cases

What researchers can do with EduLogGen, with a runnable example for each.
Every example uses the built-in synthetic demo course; replace
`elg.demo_dataset(...)` with your own data, read through a mapping:

```python
data = elg.sessionize(elg.ingest("events.csv", "mapping.yaml").dataset)
```

| Goal | Main functions | Command line |
| ---- | -------------- | ------------ |
| [Share data you cannot release](#1-share-data-you-cannot-release) | `fit_generator`, `generate`, `validate` | `fit`, `generate`, `validate` |
| [Compare generators fairly](#2-compare-generators-fairly) | `run_benchmark` | `benchmark` |
| [Test anomaly detectors](#3-test-anomaly-detectors-against-known-truth) | `inject_anomalies`, `evaluate_detection` | `generate --anomalies`, `evaluate` |
| [Test clustering methods](#4-test-clustering-methods-against-known-profiles) | `fit_profiles`, `generate_profiles`, `evaluate_clustering` | |
| [Run controlled simulations](#5-run-controlled-what-if-simulations) | `Controls`, `SessionCalendar`, `run_experiment` | `generate --experiment` |
| [Simulate learners without data](#6-simulate-learners-without-real-data) | `define_profiles`, `generate_profiles` | |
| [Analyse real logs](#7-analyse-real-logs) | `activity_profiles`, `temporal_profile`, `analyze_by` | `analyze --detail --by` |
| [Teaching](#8-teaching) | any of the above | `demo` |

## 1. Share data you cannot release

Fit a generator on the private logs (training learners), generate a
synthetic corpus, and publish it with its validation report: how close it is
to held-out real learners, and what the privacy diagnostics say.

```python
import eduloggen as elg
from eduloggen.benchmark import split_by_learner

real = elg.sessionize(elg.demo_dataset(200, seed=1))   # your data: elg.ingest(...)
train, holdout = split_by_learner(real, 0.3, seed=0)

model = elg.fit_generator("semi_markov", train)
synthetic = elg.generate(model, n_sessions=holdout.n_sessions, seed=7)

report = elg.validate(holdout, synthetic)
print(report.to_markdown())        # fidelity metrics + privacy diagnostics
elg.save_dataset(synthetic, "synthetic_corpus")   # the corpus you would share
```

The privacy diagnostics estimate memorisation risk; they do not make the
output anonymous. Your data protection rules still apply before release.
See [Privacy](privacy.md).

## 2. Compare generators fairly

The versioned protocol `session_fidelity_v1` fits each generator on 70% of
the learners and compares three seeded samples with the other 30%, next to a
real-versus-real reference (how much two samples of real learners differ).
Your own generator can join through a [plugin](plugins.md).

```python
import eduloggen as elg
from eduloggen.benchmark import run_benchmark

real = elg.sessionize(elg.demo_dataset(200, seed=1))
report = run_benchmark(
    real, ["independent", "markov", "semi_markov"], seed=0,
    hyperparameters={"markov": {"order": 2}},
)
for result in report.results:
    bigram = result.metrics["bigram_tvd"]
    print(f"{result.name:12s} bigram TVD {bigram.mean:.3f} ± {bigram.std:.3f}")
print("real vs real:", round(report.reference["bigram_tvd"], 3))
```

## 3. Test anomaly detectors against known truth

Inject labelled anomalies (event bursts, abnormal timing, unexpected
transitions, repetition, inactivity) into a fraction of sessions, run your
detector, and score it against the annotations. Each anomaly is labelled a
workflow violation or unusual but valid behaviour; anomalies are data
patterns, never evidence of misconduct.

```python
import eduloggen as elg
from eduloggen.scenarios import AnomalySpec

real = elg.sessionize(elg.demo_dataset(200, seed=1))
model = elg.fit_generator("semi_markov", real)
background = elg.generate(model, n_sessions=1000, seed=3)

injected = elg.inject_anomalies(
    background,
    [
        AnomalySpec(type="repetition", rate=0.05),
        AnomalySpec(type="inactivity", rate=0.05, params={"gap_s": 2400}),
    ],
    seed=0,
    reference=real,
)

# Replace this with your detector: session id -> anomaly score.
def my_detector(dataset):
    return {s.session_id: s.duration_s for s in dataset.sessions}

scores = my_detector(injected.dataset)
report = elg.evaluate_detection(injected.dataset, injected.annotations, scores)
print(report.to_markdown())
```

Injecting into **real** sessions (`inject_anomalies(real_holdout, ...)`)
gives the most realistic background. In our EdNet study, a detector scored
higher on synthetic backgrounds (ROC-AUC 0.90) than on real ones (0.77), so
use synthetic backgrounds to *compare* detectors rather than to estimate
their absolute accuracy.

## 4. Test clustering methods against known profiles

Learn behavioural profiles from real learners (or define them yourself, see
below), generate a population whose every learner's profile is known, run
your clustering method, and score it.

```python
import eduloggen as elg

real = elg.sessionize(elg.demo_dataset(200, seed=1))
profiles = elg.fit_profiles(real, n_profiles=3, seed=0)
print(profiles.to_markdown())          # what distinguishes each profile

mix = elg.generate_profiles(profiles, 3000, seed=1,
                            mixture={"profile_1": 0.5, "profile_2": 0.3, "profile_3": 0.2})

# Replace with your clustering method: learner id -> cluster label.
found = elg.fit_profiles(mix.dataset, n_profiles=3, seed=0)
predictions = found.assign(mix.dataset)

print(elg.evaluate_clustering(mix.annotations, predictions).to_markdown())
```

Profiles are descriptive behavioural clusters, not validated learner types.

## 5. Run controlled "what if" simulations

Change behaviour on purpose (re-weight events, scale time spent, fix session
lengths, apply a course calendar with deadline surges) and let the
manipulation check confirm that each change actually took effect, compared
with an uncontrolled baseline.

```python
from datetime import date

import eduloggen as elg
from eduloggen.generators import SessionCalendar, get_generator
from eduloggen.generators.calendar import Deadline
from eduloggen.scenarios import Controls, ExperimentSettings, run_experiment

real = elg.sessionize(elg.demo_dataset(200, seed=1))
generator = get_generator("semi_markov")
model = generator.fit(real)

settings = ExperimentSettings(
    controls=Controls(
        event_weights={"forum_post": 3.0},   # three times more forum posts
        dwell_scale={"video_play": 1.5},     # 50% longer on videos
    ),
    calendar=SessionCalendar(
        start=date(2026, 2, 2), days=56, timezone="Europe/Paris",
        deadlines=(Deadline(date=date(2026, 3, 13), surge=3.0, days_before=3),),
    ),
    anomalies=(),
)
result = run_experiment(generator, model, settings, n_sessions=2000, seed=7)
print(result.check.to_markdown())   # did each manipulation take effect?
```

The same settings can come from a YAML file:
`eduloggen generate --model model/ --experiment experiment.yaml ...`
(see `examples/configs/experiment.yaml`).

## 6. Simulate learners without real data

Define profiles by their behaviour (how sessions start, what follows what,
how long sessions are, how much time is spent) and generate a population,
for example to prototype a dashboard or pilot an analysis before any data
exist.

```python
import eduloggen as elg

profiles = elg.define_profiles({
    "steady": {
        "share": 0.6,
        "start": {"view": 1},
        "transitions": {"view": {"attempt": 0.7, "view": 0.3},
                        "attempt": {"submit": 0.8, "view": 0.2},
                        "submit": {"view": 1}},
        "session_length": {"mean": 6},
        "sessions_per_learner": {"mean": 8},
        "dwell_s": {"*": 30, "view": 180},
    },
    "crammer": {
        "share": 0.4,
        "start": {"attempt": 1},
        "transitions": {"attempt": {"attempt": 0.6, "submit": 0.4},
                        "submit": {"attempt": 1}},
        "session_length": {"fixed": 15},
        "sessions_per_learner": {"fixed": 2},
        "dwell_s": {"*": 10},
    },
})
population = elg.generate_profiles(profiles, 1000, seed=0)
print(population.dataset.n_events, "events from",
      len(population.dataset.learner_ids), "learners")
```

## 7. Analyse real logs

EduLogGen is also an analysis toolkit: per-activity profiles, timing and
deadline effects, and statistics per course, week, cohort, or any metadata
field.

```python
import eduloggen as elg

data = elg.sessionize(elg.demo_dataset(200, seed=1))   # your data: elg.ingest(...)

for activity in elg.activity_profiles(data)[:3]:
    print(activity.token, f"{activity.share:.0%} of events,",
          f"median dwell {activity.dwell_s.median:.0f}s")

timing = elg.temporal_profile(data, timezone="Europe/Paris")
print("busiest hour:", timing.peak_hour)

by_week = elg.analyze_by(data, "week", timezone="Europe/Paris")
print(by_week.to_markdown())
```

Command line: `eduloggen analyze --input corpus/ --detail --by week --timezone Europe/Paris`.

## 8. Teaching

Give students realistic logs without personal data:
`eduloggen demo --output course/ --learners 300` writes a ready-made
synthetic course, and examples 1 and 6 produce synthetic versions of a real
course or of invented learner types.

## Current limits

- Sessions are generated independently, so long-range trajectories (a
  learner's progression through a course) are not realistic.
- Learning outcomes (grades, pass/fail) are not simulated yet.
- Privacy diagnostics are empirical indicators, not anonymisation.
- See the [case studies](case-study-oulad.md) for how the generators perform
  on real data.
