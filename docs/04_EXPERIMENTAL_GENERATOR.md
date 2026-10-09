# EduLogGen Level 2: Experimental Generator (design)

| | |
| --- | --- |
| Status | **Accepted** — decisions recorded in §15; implementation in milestones |
| Builds on | Vision, PRD, SAD (`docs/01`–`03`), release 1.0.0 |
| Releases | M1 → 1.1.0, then one minor release per milestone |

## 1. Context: three levels

| Level | Capability | Research value | Status |
| ----- | ---------- | -------------- | ------ |
| 1 — Log Generator | Learn from real logs, generate synthetic sessions | Data availability | Done (1.0.0) |
| 2 — Experimental Generator | Control behavioural profiles, event frequencies, temporal patterns, anomalies, and outcomes | Algorithm evaluation | **This design** |
| 3 — Educational Simulator | Simulate populations and interactions over time | Simulation-based research | Later research track |

Level 1 answers *"can I get data like mine?"*. Level 2 answers *"does my
algorithm find what I know is there?"*. That needs **control** (the
researcher decides what the data contains) and **ground truth** (everything
controlled is recorded, so methods can be scored).

The key architectural property: Level 2 supports both **data-driven**
generation (learned from real logs) and **fully controlled** generation
without any real student data.

## 2. Goals and non-goals

**Goals**

- G1. Declarative, versioned **scenarios**: profiles, controls, calendar,
  anomalies, outcomes.
- G2. **Ground truth in a separate annotation table**, never mixed into the
  event records.
- G3. **Evaluation helpers** scoring detection, clustering, and prediction
  methods against the ground truth.
- G4. A **manipulation check** proving each control had its intended effect.
- G5. Reproducibility: explicit seeds, versioned artifacts, training-only
  fitting; scenario + seed → identical data and annotations.
- G6. Compatibility with the existing APIs, CLI, corpora, and model artifacts.

**Non-goals (Level 2)**

- Modelling learning or motivation as latent states over time (Level 3).
- Interactions between learners (Level 3).
- Causal claims: controls and simulated outcomes describe the generator,
  not real learners.

## 3. Concepts

| Concept | Meaning |
| ------- | ------- |
| **Base model** | A fitted Level 1 generator (`markov`, `semi_markov`, plugin). |
| **Profile** | A named behavioural type with its own behavioural definition. |
| **Control** | A transformation of a profile's behaviour (event weights, timing, lengths). |
| **Calendar** | When sessions happen: hour-of-day, weekday, date range, deadline surges. |
| **Anomaly** | A perturbation injected after sampling, recorded in the annotations. |
| **Outcome** | A learner-level target (pass/fail, score, band), observational or simulated. |
| **Annotations** | The ground-truth table stored next to the events. |
| **Scenario** | All of the above in one YAML file. |

## 4. Profiles

Three modes:

| Mode | Description | Research application |
| ---- | ----------- | -------------------- |
| `auto` | Discover profiles by clustering real learners (k-means, seeded) | Educational data mining |
| `provided` | Import a `learner_id,profile` CSV | Reproducing existing research |
| `manual` | Define profiles and their behaviour without real data | Controlled simulation experiments |

Rules:

- `auto` is the default **only when training data are given**. Without data,
  profiles must be `manual`.
- A `manual` profile needs a **behavioural definition**, not just a
  proportion: transition probabilities (or a start distribution plus
  next-event probabilities), event frequencies, session lengths, and dwell
  times. Proportions alone are rejected.
- Clustering is fitted on the **training split only**; holdout learners are
  assigned to the nearest learned profile and never influence it.
- `auto` profiles are named `profile_1..k` with a description table of their
  features. Renaming is allowed, but documentation must say that an
  unsupervised cluster is **not a validated psychological type**.
- One base model is fitted per profile with the same generator family. A
  profile with fewer than `min_learners` (default 5) is an error with
  guidance, never silently dropped.
- The **mixture** sets the share of synthetic learners per profile; the
  default is the observed proportions.

## 5. Controls

Applied to a copy of a profile's behaviour; the original model is unchanged.

| Control | Effect | Example |
| ------- | ------ | ------- |
| `event_weights` | Multiply the probability of moving *into* a token, then renormalise every row; `0` removes it | `forum_post: 1.5` |
| `dwell_scale` | Multiply time spent on a token (or `"*"`) | `video_play: 0.5` |
| `session_length` | Scale or fix session lengths | `{scale: 1.2}` |
| `sessions_per_learner` | Override the distribution | `{mean: 6}` |

## 6. Calendar

Replaces "uniform start within the training span": hour-of-day and weekday
weights, a date range, and deadline surges. A learner's sessions stay in
order and do not overlap (overlap is the `concurrent_activity` anomaly).

## 7. Anomalies

Injected after sampling (or into any sessionized dataset). Each anomaly has a
**category**:

- **`invalid_workflow`** — violates the application's event model, e.g.
  `submit` before `start`. Detectors of malformed records should find these.
- **`unusual_valid`** — structurally valid but behaviourally rare, e.g. 25
  AI-assistant requests in 3 minutes.

Benchmarks report results per category, because many methods detect invalid
records rather than meaningful behavioural deviations.

| Anomaly | Priority | Category | Example | Milestone |
| ------- | -------- | -------- | ------- | --------- |
| `event_frequency` | P0 | unusual_valid | 25 AI requests in 3 minutes | M1 |
| `abnormal_timing` | P0 | unusual_valid | A complex assessment completed in 10 s | M1 |
| `unexpected_transition` | P0 | invalid_workflow | `submit` → `start` | M1 |
| `repetition` | P0 | unusual_valid | Switching back and forth between two questions | M1 |
| `inactivity` | P0 | unusual_valid | 40 minutes without activity inside a session | M1 |
| `session_interruption` | P1 | unusual_valid | Session ends abruptly mid-task | later |
| `concurrent_activity` | P1 | invalid_workflow | Overlapping sessions of one learner | later |
| `rare_sequence` | P1 | unusual_valid | Valid but rare navigation path | later |

`unexpected_transition` uses transitions never observed in the reference
data, or a researcher-supplied list of forbidden transitions.

Anomalies describe **data patterns, not intentions**. EduLogGen never labels
anything as cheating or misconduct, and documentation must not equate the two.

Anomaly injectors become a plugin kind (`eduloggen.plugins.anomaly`).

## 8. Annotations (ground truth)

A separate table, `annotations.csv`, stored in the corpus directory, declared
in the manifest, and protected by its own fingerprint:

| Column | Meaning |
| ------ | ------- |
| `level` | `learner`, `session`, or `event` |
| `id` | Learner, session, or event id |
| `annotation` | `anomaly`, `profile`, or `outcome` |
| `type` | Anomaly type, profile name, or outcome target |
| `category` | Anomaly category (empty otherwise) |
| `value` | `1` for anomalies; profile name or outcome value |
| `parameters` | JSON injection or generation parameters |

Rows exist only for annotated items; anything without an anomaly row is
normal. After injection all ids are remapped again, so ids, ordering, and
id formats never reveal which records were injected.

## 9. Outcomes

Included in Level 2, at learner level, in two **distinct** modes:

| Mode | Meaning |
| ---- | ------- |
| `observational` | Learn the relationship between interaction features and real outcomes when labelled training data exist; generated outcomes follow that learned relationship. |
| `simulated` | Generate outcomes from an explicitly stated mechanism (e.g. a logistic model for pass/fail). |

Targets: binary (`pass`/`fail`), continuous (`final_score`, bounded, e.g. a
beta or clipped linear model), categorical (`low`/`medium`/`high`), and
optional intermediate outcomes (question score, quiz completion, task
success).

Scientific rules:

- Simulated outcomes are labelled as such in annotations and reports; they
  are **not evidence** that the mechanism describes real students.
- Every outcome has a **prediction time** (e.g. end of week 4). Prediction
  benchmarks expose only events before it: no **temporal leakage**, and the
  outcome is never derivable from a feature built from it (**target
  leakage**).
- All assumptions (mechanism, coefficients, noise) are written to the
  scenario model and the run manifest.

## 10. Manipulation check

Each generation writes `manipulation_check.json` / `.md` listing every
control and anomaly with its intended and measured effect (e.g. `forum_post`
share 4.0% → target ~6.0%, measured 5.9%).

## 11. Evaluation helpers (`eduloggen.evaluation`)

- `evaluate_detection(dataset, annotations, predictions)` — precision,
  recall, F1 overall; recall per anomaly type and per category; ROC-AUC and
  average precision when predictions are scores.
- `evaluate_clustering(annotations, assignments)` — adjusted Rand index,
  normalised mutual information, purity (M3).
- `evaluate_prediction(annotations, predictions)` — accuracy, F1, AUC,
  RMSE/MAE depending on the outcome type (M4).
- CLI: `eduloggen evaluate --corpus out/ --predictions preds.csv`.

Pure Python, no new dependencies.

## 12. Interfaces

```bash
eduloggen scenario fit --input real/ --config scenario.yaml --output scenario_model/
eduloggen scenario generate --model scenario_model/ --n-sessions 5000 --seed 42 --output synthetic/
eduloggen scenario validate --model scenario_model/ --synthetic synthetic/   # fidelity + manipulation check
eduloggen scenario inspect --model scenario_model/                           # profiles, controls, assumptions
eduloggen evaluate --corpus synthetic/ --predictions preds.csv
```

A combined `scenario run` may come later as a thin wrapper over fit and
generate. A scenario model is a versioned directory of per-profile model
artifacts plus the resolved scenario, shareable without real data.

Before scenarios exist (M1), anomalies are available directly:
`eduloggen generate --anomalies anomalies.yaml` and
`eduloggen.inject_anomalies(dataset, specs, seed=...)`.

## 13. Architecture

- `eduloggen.models.annotations` — the `Annotations` table (models layer, so
  `io` can read and write it).
- `eduloggen.scenarios` — anomalies (M1); profiles, controls, calendar,
  outcomes, scenario files, manipulation check (later). Depends on `core`,
  `config`, `models`, `analysis`, `generators`, `privacy`, `utils`.
- `eduloggen.evaluation` — scoring; depends on `core`, `models`.
- `privacy.remap_ids` gains a variant that returns the id mapping so
  annotations can follow remapping.
- `io.corpus` — optional `annotations.csv` with its own fingerprint.
- ADR-026: scenarios are layered on generators, never inside them.
- ADR-027: ground truth lives in a separate annotation artifact.

## 14. Milestones

| # | Deliverable | Release |
| - | ----------- | ------- |
| M1 | Annotations in corpora; five P0 anomalies with categories; `inject_anomalies`; `generate --anomalies`; `evaluate_detection` and `eduloggen evaluate` | 1.1.0 |
| M2 | Controls + calendar + manipulation check | 1.2.0 |
| M3 | Profiles (`auto`, `provided`, `manual`) + mixture + `evaluate_clustering` | 1.3.0 |
| M4 | Outcomes (observational, simulated) + leakage-safe prediction splits + `evaluate_prediction` | 1.4.0 |
| M5 | Scenario files and `scenario fit/generate/validate/inspect`; P1 anomalies; anomaly plugins; tutorial | 1.5.0 |

Each milestone ships with tests and documentation and is useful on its own.

## 15. Decisions

1. **Profiles:** three modes (`auto`, `provided`, `manual`); `auto` is the
   default only with training data; manual profiles require behavioural
   parameters; clustering on training data only; cluster names are
   descriptive, not psychological types.
2. **Anomalies:** P0 — event frequency, abnormal timing, unexpected
   transitions, repetition, long inactivity; P1 — session interruption,
   concurrent activity, rare sequences. Annotations are separate from events;
   `invalid_workflow` and `unusual_valid` are distinguished; anomalies are
   never equated with misconduct.
3. **Outcomes:** part of Level 2; binary, continuous, categorical, and
   intermediate targets; observational and simulated modes; temporal and
   target leakage prevented; assumptions documented.
4. **CLI:** separate `scenario fit`, `generate`, `validate`, `inspect`;
   `scenario run` possibly later.
5. **Release:** ship M1 as soon as its acceptance criteria are met. The
   repository is at 1.0.0, so under SemVer M1 is **1.1.0** (additive).
