# EduLogGen Level 2: Experimental Generator (design note)

| | |
| --- | --- |
| Status | **Draft for review** — no code yet |
| Builds on | Vision, PRD, SAD (`docs/01`–`03`) |
| Target release | 0.3.0 |

## 1. Context: three levels

| Level | Capability | Research value | Status |
| ----- | ---------- | -------------- | ------ |
| 1 — Log Generator | Learn from real logs, generate synthetic sessions | Data availability | Done (0.2) |
| 2 — Experimental Generator | Control behavioural profiles, event frequencies, temporal patterns, and anomalies | Algorithm evaluation | **This note** |
| 3 — Educational Simulator | Simulate populations and interactions over time | Simulation-based research | Later research track |

Level 1 answers *"can I get data like mine?"*. Level 2 answers *"does my
algorithm find what I know is there?"*. That requires two things Level 1
lacks:

1. **Control** — the researcher decides what the data contains (how many
   struggling learners, more forum use, a deadline surge, 3% bot-like
   sessions).
2. **Ground truth** — every controlled property is labelled, so detectors,
   clustering, and prediction methods can be scored against the truth.

## 2. Goals and non-goals

**Goals**

- G1. Generate datasets whose behavioural composition is set by a
  declarative, versioned **scenario file**.
- G2. Ship **labels** (learner profile, injected anomalies) with every
  generated corpus.
- G3. Provide **evaluation helpers** that score a method's output against
  those labels.
- G4. Provide a **manipulation check**: a report proving each control had its
  intended effect.
- G5. Stay reproducible: scenario + seed → identical dataset and labels.

**Non-goals (Level 2)**

- Modelling learning, knowledge, or motivation over time (Level 3).
- Interactions between learners (Level 3).
- Outcomes such as grades or course completion as labels (possible later).
- Causal claims: controls change the generator, not real learners.

## 3. Concepts

| Concept | Meaning |
| ------- | ------- |
| **Base model** | A fitted Level 1 generator (`markov`, `semi_markov`, plugin). |
| **Profile** | A named behavioural type with its own base model (e.g. `steady`, `struggler`). |
| **Control** | A transformation of a profile's model: event weights, timing, session lengths. |
| **Calendar** | When sessions happen: hour-of-day, weekday, date range, deadline surges. |
| **Anomaly** | A labelled perturbation injected after sampling (bursts, rapid guessing, …). |
| **Scenario** | Profiles + mixture + controls + calendar + anomalies, in one YAML file. |
| **Labels** | Ground truth written next to the events. |

Pipeline:

```text
real sessions ──► profiles (cluster or given) ──► one base model per profile
                                                        │ controls
scenario.yaml ──────────────────────────────────────────┤
                                                        ▼
                     sample per profile mix + calendar ──► inject anomalies
                                                        ▼
                         synthetic corpus + labels.csv + manipulation check
```

## 4. Profiles

Two ways to define them:

1. **Learned (default).** Cluster real learners on features already computed
   by `analysis` (sessions per learner, mean session length and duration,
   event-type shares, success rate). Standardised features, seeded k-means
   (pure Python), `k` from the scenario. Each profile is named
   `profile_1..k` and described in a summary table so researchers can rename
   them (`profile_2: struggler`).
2. **Given.** A `learner_id,profile` CSV from the researcher (e.g. from a
   prior study), applied to the real data before fitting.

One base model is fitted per profile with the same generator family and
hyperparameters. A profile with fewer than `min_learners` (default 5)
learners is an error with guidance (merge profiles or lower `k`), never
silently dropped.

The **mixture** sets the share of synthetic learners per profile; default is
the real proportions.

## 5. Controls

Applied to a copy of a profile's fitted model; the original is unchanged.

| Control | Effect | Example |
| ------- | ------ | ------- |
| `event_weights` | Multiply the probability of moving *into* a token, then renormalise every row. `0` removes the token. | `forum_post: 1.5` |
| `dwell_scale` | Multiply time spent on a token (or `"*"` for all). | `video_play: 0.5` |
| `session_length` | Scale or fix session lengths. | `{scale: 1.2}` |
| `sessions_per_learner` | Override the distribution. | `{mean: 6}` |

Rows emptied by a `0` weight fall back to shorter contexts, as in normal
backoff. Controls can be set globally or per profile.

## 6. Calendar

Replaces the current "uniform start within the training span".

```yaml
calendar:
  start: 2026-09-01
  weeks: 12
  hours: [0,0,0,0,0,0,1,2,4,6,6,5,4,5,6,6,5,5,6,7,7,5,3,1]   # relative weights
  weekdays: [1, 1, 1, 1, 0.9, 0.5, 0.6]                     # Mon..Sun
  deadlines:
    - {date: 2026-10-15, surge: 3.0, days_before: 3}
```

Session start times are drawn from this calendar; within-session timing still
comes from the model. A learner's sessions stay in time order and never
overlap.

## 7. Anomalies

Injected after sampling; each instance gets a label. Rates are shares of
sessions (or learners for learner-level anomalies). Initial catalogue:

| Type | Level | What it looks like | Typical research use |
| ---- | ----- | ------------------ | -------------------- |
| `burst` | session | Many events within seconds | Bot / scripted-client detection |
| `rapid_guessing` | session | Repeated `attempt` with dwell below a threshold | Disengagement / gaming the system |
| `loop` | session | The same activity repeated many times | UI problems, confusion |
| `copied_session` | session | Near-copy of another learner's session, time-shifted | Account sharing, plagiarism |
| `off_hours` | session | Session at an unusual hour | Contract cheating, time-zone issues |
| `dropout` | learner | Activity stops after a share of the course | Early-warning / dropout prediction |

Anomaly injectors are a new **plugin kind** (`eduloggen.plugins.anomaly`), so
researchers can add domain-specific ones.

## 8. Labels

Written as `labels.csv` in the corpus directory and declared in its manifest:

| Column | Meaning |
| ------ | ------- |
| `level` | `learner`, `session`, or `event` |
| `id` | Learner, session, or event id (after id remapping) |
| `label` | `profile` or an anomaly type |
| `value` | Profile name, or `1` for an anomaly |
| `details` | JSON: parameters (e.g. `{"rate_s": 0.8, "n_events": 12}`) |

Labels get their own fingerprint in the manifest, so tampering is detected
like the event data. Id remapping must apply the same mapping to labels; the
privacy module will return its mapping for this.

## 9. Manipulation check

Every generation also writes `manipulation_check.json` / `.md`: for each
control, the intended change and the measured one, e.g.

| Control | Intended | Measured | OK |
| ------- | -------- | -------- | -- |
| `forum_post` weight ×1.5 | share 4.0% → ~6.0% | 5.9% | ✓ |
| deadline surge ×3, 3 days | 3× sessions | 2.8× | ✓ |
| `rapid_guessing` 5% of sessions | 5% | 5.0% (exact) | ✓ |

This is what a methods section needs to show the synthetic conditions were
as stated.

## 10. Evaluation helpers

Score a method's output against `labels.csv`:

- `evaluate_detection(labels, predictions)` — precision, recall, F1 per
  anomaly type and overall; ROC-AUC and PR-AUC when predictions are scores.
- `evaluate_clustering(labels, assignments)` — adjusted Rand index,
  normalised mutual information, purity, against the true profiles.
- CLI: `eduloggen evaluate --labels out/ --predictions preds.csv`.

Pure Python, no new dependencies.

## 11. Scenario file

```yaml
scenario: struggling_cohort_v1        # name, recorded in outputs
generator: {name: semi_markov, order: 2, smoothing_alpha: 0.1}

profiles:
  source: cluster                     # or: file (learner_id,profile CSV)
  k: 3
  names: {profile_1: steady, profile_2: skimmer, profile_3: struggler}
  mixture: {steady: 0.5, skimmer: 0.2, struggler: 0.3}

controls:
  struggler:
    event_weights: {attempt: 1.4, forum_post: 1.5}
    dwell_scale: {video_play: 1.3}

calendar:
  start: 2026-09-01
  weeks: 12
  deadlines: [{date: 2026-10-15, surge: 3.0, days_before: 3}]

anomalies:
  - {type: rapid_guessing, rate: 0.05, max_dwell_s: 3}
  - {type: copied_session, rate: 0.01}
  - {type: dropout, rate: 0.10, after: 0.4}

generation: {n_learners: 500, seed: 42}
```

## 12. Interfaces

```python
import eduloggen as elg

scenario = elg.load_scenario("struggling_cohort_v1.yaml")
model = elg.fit_scenario(real_sessions, scenario)      # profiles + base models
result = elg.generate_scenario(model, seed=42)         # dataset + labels + check
result.labels.for_level("session")
elg.evaluate_detection(result.labels, my_detector_output)
```

```bash
eduloggen scenario fit --input sessions/ --scenario s.yaml --output smodel/
eduloggen scenario generate --model smodel/ --seed 42 --output out/
eduloggen evaluate --labels out/ --predictions preds.csv
```

A scenario model is a directory of per-profile model artifacts plus the
resolved scenario, so it can be shared without real data. Without real data,
`fit_scenario` can start from `eduloggen demo`.

## 13. Architecture

- New package `eduloggen.scenarios` (scenario schema, profiles, controls,
  calendar, anomalies, labels, manipulation check), depending on `core`,
  `config`, `models`, `analysis`, `generators`, `privacy`, `utils`.
- New package `eduloggen.evaluation` (detection and clustering scores),
  depending only on `core` and `models`.
- `generators`: a hook to sample session start times from a calendar; a
  model-transform API for controls.
- `io.corpus`: optional `labels.csv` with its own fingerprint.
- `privacy.remap_ids`: return the id mapping.
- Plugins: new kind `anomaly`.
- ADR-026 (scenarios are layered on generators, never inside them) and
  ADR-027 (labels are first-class corpus artifacts).

## 14. Milestones

| # | Deliverable | Depends on |
| - | ----------- | ---------- |
| M1 | Labels in corpora + id-mapping + 3 anomalies (`burst`, `rapid_guessing`, `dropout`) + `evaluate_detection` | — |
| M2 | Controls (`event_weights`, `dwell_scale`, `session_length`) + calendar + manipulation check | M1 |
| M3 | Profiles (cluster / file) + mixture + `evaluate_clustering` | M1 |
| M4 | Scenario file, CLI, remaining anomalies, anomaly plugins, tutorial | M2, M3 |

Each milestone is usable on its own: after M1, researchers can already test
anomaly detectors on any Level 1 model.

## 15. Open questions for review

1. **Profiles:** clustering by default plus a CSV option — is that right, or
   should profiles be hand-specified from the start?
2. **Anomaly catalogue:** which anomalies matter most for your research? Any
   missing (e.g. answer copying across learners, time-zone shifts)?
3. **Outcomes:** should Level 2 already attach outcome labels (pass/fail,
   final score) driven by profile, for prediction research, or leave that to
   Level 3?
4. **CLI shape:** separate `scenario fit` / `scenario generate`, or one
   `scenario` command that does both?
5. **Release:** ship M1 as 0.3.0 quickly, or wait for the full Level 2?
