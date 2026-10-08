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

## Before sharing outputs

Read [Privacy and responsible use](privacy.md). Synthetic data can still
resemble real sessions; check the privacy indicators in every validation
report.
