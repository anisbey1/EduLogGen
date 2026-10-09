# Getting started

This guide installs EduLogGen and walks through the full workflow: ingest a
log, build sessions, fit a generator, sample synthetic data, and validate it.

## Install

```bash
git clone https://github.com/anisbey1/EduLogGen.git
cd EduLogGen
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev,docs]"
```

Check the installation:

```bash
eduloggen info
pytest
```

## Try it on demo data first

No data at hand? EduLogGen can simulate a small course (fully synthetic):

```bash
eduloggen demo --output demo/
eduloggen benchmark --input demo/
```

The benchmark prints a comparison of all generators. Continue with your own
data below.

## 1. Describe your log with a field mapping

EduLogGen reads any tabular log (CSV, TSV, JSON Lines, Parquet) once you say
which column fills each canonical field. Required fields are `learner_id`,
`timestamp`, `activity_id`, and `event_type`; `event_id` is generated from
row numbers when absent.

```yaml
# mapping.yaml
fields:
  event_id: log_id
  learner_id:
    source: user_id
    hash: {salt: change-me}      # replace ids with salted pseudonyms
  timestamp:
    source: time
    format: "%Y-%m-%d %H:%M:%S"  # or iso, epoch_s, epoch_ms
  activity_id: resource
  event_type:
    source: action
    default: other
  score: grade
metadata: [device]               # extra columns to keep
drop: [user_email]               # columns ignored on purpose
timezone: Europe/Paris           # zone for timestamps without one
```

A fuller example is in `examples/configs/demo_mapping.yaml`.

## 2. Run the pipeline

```bash
eduloggen ingest --input events.csv --mapping mapping.yaml --output corpus/
eduloggen sessionize --input corpus/ --strategy idle_timeout --output sessions/
eduloggen analyze --input sessions/ --output analysis/
eduloggen fit --input sessions/ --generator semi_markov --set order=2 --output model/
eduloggen generate --model model/ --n-sessions 1000 --seed 42 --output synthetic/
eduloggen validate --real sessions/ --synthetic synthetic/ --output report/
```

Each output directory also holds a `run_manifest.json` recording the command,
configuration fingerprint, and input fingerprints. Outputs are never
overwritten unless you pass `--force`.

## 3. Use a configuration file

Put settings in YAML, TOML, or JSON and pass `--config`; see
`examples/configs/minimal.yaml`. Relative paths are resolved against the
config file. Precedence, highest first: command-line flags, `EDULOGGEN_*`
environment variables (`EDULOGGEN_SEED`, `EDULOGGEN_LOG_LEVEL`,
`EDULOGGEN_OUTPUT_DIR`, `EDULOGGEN_CONFIG`), the config file, built-in
defaults.

## 4. Or work in Python

```python
import eduloggen as elg

cfg = elg.load_config("experiment.yaml")
real = elg.sessionize(elg.ingest("events.csv", "mapping.yaml", config=cfg).dataset, config=cfg)
print(elg.analyze(real).to_markdown())

model = elg.fit_generator("semi_markov", real, config=cfg)
synthetic = elg.generate(model, n_sessions=1000, seed=42)
report = elg.validate(real, synthetic, thresholds={"event_type_tvd": 0.1})
print(report.status, report.metric("bigram_tvd").value)
```

## Look closer: fine-grained analysis

```bash
eduloggen analyze --input sessions/ --detail --timezone Europe/Paris \
    --deadlines 2026-10-16,2026-11-27 --by week --output analysis/
```

- `--detail` adds **per-activity** profiles (share, reach, where sessions
  start and end, repeats, median time spent, success rate, what usually
  comes next) and **temporal** analysis in your timezone (hours, weekdays,
  a weekday x hour table, weekly trends, gaps between a learner's sessions,
  and the activity ratio before each deadline).
- `--by` breaks every statistic down by `course`, `week`, `weekday`, `hour`,
  `learner_group` (with `--groups cohorts.csv`, columns `learner_id,group`),
  or `metadata:<key>` such as `metadata:device`. Each group's event mix is
  compared with the overall mix.

To see **where** synthetic data differs from real data, not just how much:

```bash
eduloggen validate --real sessions/ --synthetic synthetic/ \
    --detailed --by weekday --timezone Europe/Paris --output report/
```

`detailed.md` ranks the largest gaps per activity (share and time spent),
per transition (including transitions the generator invents or never
produces), per session length, per hour and weekday, and per group.
`eduloggen plot --plots activity_heatmap` draws the weekday x hour pattern
of session starts side by side.

## Generators

| Name | Models | Use when |
| ---- | ------ | -------- |
| `markov` | Order-k token transitions; constant gap between events | Sequence structure matters, timing does not |
| `semi_markov` | Transitions plus time spent on each token | Realistic timing matters |
| `independent` | Tokens drawn independently | Baseline for benchmarks |

Key hyperparameters: `order`, `smoothing_alpha`, `length_model`
(`empirical`, `poisson`, `fixed`), and for `semi_markov` `timing_family`
(`empirical`, `lognormal`, `gamma`, `exponential`).

## Plot the results

Install the `viz` extra (`pip install 'eduloggen[viz]'`), then:

```bash
eduloggen plot --real sessions/ --synthetic synthetic/ --output figures/
eduloggen plot --validation report/report.json --benchmark benchmark/benchmark.json \
    --format png,svg --output figures/
```

Dataset plots: `event_frequencies`, `activity_frequencies`, `session_lengths`,
`session_durations`, `interevent_times`, `transitions` (heatmaps on a shared
scale), `transition_graph` (node-link diagram: node area is token frequency,
arrow width the share of transitions), `sankey` (session pathways over the
first steps, with `(end)` for sessions that stop), and `timeline` (a capped
random sample of sessions, labelled without ids). In Python, every function in `eduloggen.visualization` returns a
matplotlib figure.

## Compare generators

```bash
eduloggen benchmark --input sessions/ --generators markov,semi_markov,independent \
    --seeds 3 --output benchmark/
```

Protocol `session_fidelity_v1` holds out 30% of learners, fits each generator
on the rest, generates as many sessions as the holdout has with three seeds,
and scores every sample against the holdout. The **Reference** column scores
the real training data against the holdout: it shows what a perfect generator
could reach on this dataset, so compare generators to it rather than to zero.
Results describe your dataset only; they are not a general ranking.

## Test an anomaly detector (Level 2)

Generate data with known, labelled anomalies, run your detector on the
events, and score it against the ground truth:

```bash
eduloggen generate --model model/ --n-sessions 1000 --seed 7 \
    --anomalies examples/configs/anomalies.yaml --output synthetic/
# run your detector on synthetic/events.csv and write preds.csv with
# columns: id (session id) and score (higher = more anomalous)
eduloggen evaluate --corpus synthetic/ --predictions preds.csv
```

Five anomaly types are available. `unexpected_transition` is an
**invalid workflow** (it breaks the event model); `event_frequency`,
`abnormal_timing`, `repetition`, and `inactivity` are **unusual but valid**.
The report gives precision, recall, F1, ROC-AUC, and average precision, plus
recall per type and per category, since many methods find malformed records
but miss rare valid behaviour. Ground truth lives in `annotations.csv`,
separate from the events, and all ids are remapped after injection so they
cannot give the answer away. Anomalies describe data patterns, never
intentions; do not read them as evidence of misconduct.

In Python: `elg.inject_anomalies(dataset, specs, seed=...)` and
`elg.evaluate_detection(dataset, annotations, predictions)`.

## Run a controlled experiment (Level 2)

Change behaviour on purpose and prove the change took effect:

```bash
eduloggen generate --model model/ --n-sessions 1000 --seed 7 \
    --experiment examples/configs/experiment.yaml --output synthetic/
```

The experiment file has three optional sections:

- `controls` — `event_weights` (e.g. `forum_post: 2.0`; `0` removes a
  token), `dwell_scale` (time spent before the next event, per token or
  `"*"`), `session_length` (`{scale: 1.2}` or `{fixed: 8}`),
  `sessions_per_learner` (`{mean: 5}` or `{fixed: 3}`);
- `calendar` — period (`start`, `weeks`), `timezone`, 24 `hours` weights,
  7 `weekdays` weights, and `deadlines` with a `surge` over `days_before`;
- `anomalies` — as above.

The output corpus contains `manipulation_check.md`: each control is compared
with an **uncontrolled baseline** drawn with the same model, seed, and size.
Exact rules (removed tokens, closed hours, the period) must hold strictly;
distributional checks (hour and weekday mix, deadline surges) use
sample-size-aware tolerances, become more sensitive with more sessions, and
fail by chance roughly 1% of the time with small samples. Controls change the
generator, not real learners: report them as simulated conditions.

In Python: `apply_controls(model, Controls(...))`, `SessionCalendar`, and
`run_experiment(...)` in `eduloggen.scenarios`; `elg.generate(...,
calendar=...)`.

## Simulate behavioural profiles (Level 2)

Generate a population that mixes distinct kinds of learners, and keep the
ground truth of who is which:

```python
import eduloggen as elg
from eduloggen.benchmark import split_by_learner
from eduloggen.scenarios import load_profile_assignments

train, holdout = split_by_learner(dataset, 0.3, seed=0)

# auto: cluster the training learners (seeded k-means++ on behaviour)
profiles = elg.fit_profiles(train, n_profiles=3, seed=0)
print(profiles.to_markdown())          # what distinguishes each profile
holdout_profiles = profiles.assign(holdout)  # nearest learned profile

# provided: your own learner_id,profile CSV
provided = elg.fit_profiles(
    train, mode="provided", assignments=load_profile_assignments("groups.csv")
)

# manual: no real data, behaviour defined explicitly
manual = elg.define_profiles({
    "steady": {
        "share": 2,
        "start": {"view": 1},
        "transitions": {"view": {"attempt": 0.8, "view": 0.2},
                        "attempt": {"submit": 1}, "submit": {"view": 1}},
        "session_length": {"mean": 5},
        "sessions_per_learner": {"mean": 4},
        "dwell_s": {"*": 30, "view": 120},
    },
    "crammer": {
        "start": {"attempt": 1},
        "transitions": {"attempt": {"attempt": 0.7, "submit": 0.3},
                        "submit": {"attempt": 1}},
        "session_length": {"fixed": 10},
        "sessions_per_learner": {"fixed": 1},
    },
})

result = elg.generate_profiles(profiles, 1000, seed=7, mixture={"profile_1": 0.5,
                                "profile_2": 0.3, "profile_3": 0.2})
result.dataset        # synthetic events and sessions
result.annotations    # one "profile" row per synthetic learner
profiles.save("profiles/")  # profiles.json, PROFILES.md, one model each
```

Rules worth knowing:

- Fit `auto` profiles on the **training split only**; `assign` maps other
  learners to the nearest profile without changing it.
- A profile with fewer than `min_learners` (default 5) training learners is
  an error, never silently dropped.
- `mixture` sets the share of synthetic **learners** per profile (default:
  the observed shares); per-profile `controls` and a shared `calendar` work
  as in experiments.
- Manual profiles need behaviour (`start` and `transitions`); a share alone
  is rejected.
- Profile names are descriptive. An unsupervised cluster is **not** a
  validated psychological or pedagogical type; do not report it as one.

To score a clustering method against the ground truth, use
`elg.evaluate_clustering(result.annotations, predictions)` (adjusted Rand
index, normalised mutual information, purity). A command-line interface for
profiles arrives with the `scenario` commands (M5).

## Before sharing outputs

Read [Privacy and responsible use](privacy.md). Synthetic data can still
resemble real sessions; check the privacy indicators in every validation
report.
