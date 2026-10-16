# Transformer generator

`transformer` is an optional, time-aware, autoregressive Transformer that
learns event sequences and the time between events within a session. It is
new in 1.6.0 and requires PyTorch:

```bash
pip install "eduloggen[neural]"
```

Without PyTorch the rest of EduLogGen works unchanged; selecting the
Transformer then fails with an error that gives this installation command.

## Tutorial

Fit on the built-in synthetic demo course, generate, and validate on
held-out learners (about 20 seconds on a laptop CPU with these small
settings):

```python
import eduloggen as elg
from eduloggen.benchmark import split_by_learner

data = elg.sessionize(elg.demo_dataset(200, seed=1))
train, holdout = split_by_learner(data, 0.3, seed=0)

model = elg.fit_generator(
    "transformer",
    train,
    hyperparameters={"d_model": 64, "num_heads": 4, "epochs": 30, "seed": 42},
)
synthetic = elg.generate(model, n_sessions=holdout.n_sessions, seed=42)
print(elg.validate(holdout, synthetic).to_markdown())
```

The same on the command line:

```bash
eduloggen demo --output demo/ --learners 200
eduloggen fit --input demo/ --generator transformer \
    --set d_model=64 --set num_heads=4 --set epochs=30 --output transformer_model/
eduloggen generate --model transformer_model/ --n-sessions 500 --seed 42 \
    --output transformer_synthetic/
eduloggen validate --real demo/ --synthetic transformer_synthetic/ --output report/
```

To compare it with the other generators on held-out learners:

```python
from eduloggen.benchmark import run_benchmark

report = run_benchmark(
    data, ["semi_markov", "gru", "transformer"], seed=0,
    hyperparameters={"transformer": {"d_model": 64, "epochs": 30}},
)
```

## How it works

### What the model predicts

A session is a sequence of events `e1 … eT`. The gap `g_t` is the time from
`e_t` to `e_(t+1)`, the convention every EduLogGen generator uses. At
position `p` the model receives:

| Input | Position 0 | Position 1 | Position `p ≥ 2` |
| ----- | ---------- | ---------- | ---------------- |
| Event | `BOS` | `e1` | `e_p` |
| Gap | "no gap" | "no gap" | bin of `g_(p-1)` (the gap that led to `e_p`) |
| Position | learned embedding of the position in the context window | | |

The three embeddings are added. `BOS` never implies an invented gap. From
the hidden state at `p`, two heads predict:

- the **next event** `e_(p+1)` (softmax over the fitted event types);
- for `p ≥ 1`, the **gap after the current event** `g_p` (softmax over gap
  bins).

The joint distribution is factorised as
`P(e_(p+1), g_p | history) = P(e_(p+1) | history) · P(g_p | history)`: the
time head does not see the event it precedes. This is the same convention as
the `gru` generator, so the two neural generators are directly comparable.
The training loss is `L_event + time_loss_weight × L_time`, both mean
cross-entropies over real (non-padded) targets.

### Time

Gap bins are fitted on the training learners' gaps only: one bin per
distinct value when there are at most `n_bins` of them (for example whole
days in OULAD), otherwise `n_bins` quantile bins of the log gap
(millisecond clickstreams such as EdNet). A sampled bin becomes a gap by
drawing one of the training gaps stored for it, so units are preserved and
discrete values (such as exactly one day) are reproduced exactly. Zero gaps
count as 1 ms, as everywhere in EduLogGen. Gaps between sessions and
session start times are not modelled by the Transformer; they come from the
shared machinery.

### Network

A decoder-only stack of `num_layers` pre-norm blocks: causal multi-head
self-attention, residual connections, layer normalisation, a position-wise
feed-forward network (GELU), and dropout (training only). A causal mask
stops any position from seeing later ones, and padded positions are masked
out of attention and of both losses. The test suite checks that changing
future inputs never changes earlier predictions.

### Training

1. The event vocabulary comes from the data being fitted; gap bins from the
   inner training learners.
2. The fitted learners are split **by learner** into inner training and
   early-stopping validation groups (`validation_fraction`, default 10%), so
   no learner is in both. With fewer than two learners there is no
   validation group: training runs for `epochs` epochs and a warning says so.
   In the benchmark, all of this happens inside the 70% training partition;
   holdout learners are never seen.
3. Sessions longer than `max_seq_len` are split into overlapping windows
   (`window_overlap`, default half a window). Every target is scored exactly
   once, in the first window that contains it after the overlap, so no
   training target is discarded; later windows keep the overlap as context.
4. AdamW with gradient clipping; after each epoch the validation loss decides
   early stopping (`patience`) and the best epoch's weights are kept. A
   non-finite loss stops training with an error.

The model records, per epoch, the training loss (running mean over the
epoch's batches) and the validation loss, plus the best epoch, the numbers
of learners, sessions, and windows, the context length, the device, and the
PyTorch version (`model.parameters["transformer"]["training"]`). Training
time is not stored in the model, so that identical fits share a
fingerprint; the benchmark and `describe()`-based reports measure it.

### Generation

Session lengths, sessions per learner, learners, and start times come from
the same models as for every generator (there is no end-of-session token),
so benchmark comparisons are like for like. For each session the model
starts from `BOS` and draws one event and one gap per step with EduLogGen's
own random number generator (`temperature`, optional `top_k`), on one CPU
thread with a key/value cache. Beyond `max_seq_len` events the context rolls
over the last `max_seq_len` positions while the whole session is kept.
Events always come from the fitted vocabulary.

## Hyperparameters

All are set through `hyperparameters={...}` in Python or `--set key=value`
on the command line. Defaults are reasonable starting points, not tuned
values.

| Name | Default | Meaning |
| ---- | ------- | ------- |
| `d_model` | 128 | Width of embeddings and hidden states (divisible by `num_heads`) |
| `num_heads` | 4 | Attention heads |
| `num_layers` | 2 | Transformer blocks |
| `ff_dim` | 256 | Feed-forward width |
| `dropout` | 0.1 | Dropout during training |
| `learning_rate` | 0.001 | AdamW learning rate |
| `weight_decay` | 0.01 | AdamW weight decay |
| `batch_size` | 64 | Windows per batch |
| `epochs` | 100 | Maximum epochs |
| `patience` | 10 | Epochs without validation improvement before stopping |
| `max_seq_len` | 256 | Context window (events) |
| `window_overlap` | 0.5 | Overlap of training windows, as a share of the window |
| `time_loss_weight` | 1.0 | Weight `λ` of the time loss |
| `n_bins` | 32 | Maximum number of gap bins |
| `validation_fraction` | 0.1 | Share of learners used for early stopping |
| `grad_clip` | 1.0 | Gradient norm limit |
| `temperature` | 1.0 | Sampling temperature (events and gaps) |
| `top_k` | 0 | Sample from the `k` most likely values only (0 = off) |
| `device` | `cpu` | `cpu`, `cuda`, `mps`, or `auto` (training only) |
| `seed` | 0 | Training seed |
| `threads` | 1 | CPU threads for training |

Plus the shared `length_model` and `fixed_length`.

## Reproducibility and devices

- Training is seeded. On CPU, with deterministic algorithms enabled, a fit
  with the same data, hyperparameters (including `threads`), PyTorch
  version, and hardware gives the same model and fingerprint. Different
  thread counts, PyTorch versions, or hardware may give slightly different
  numbers; exact equivalence across them is not guaranteed.
- Generation always runs on CPU and is reproducible for a given model and
  seed.
- `device="cuda"`/`"mps"` (or `"auto"`) trains on a GPU when one is
  available, and fails with a clear error otherwise. GPU training is not
  covered by the test suite and is not guaranteed to be deterministic.

## Saving and loading

Transformer models are ordinary EduLogGen model directories: JSON only, with
the weights stored as base64-encoded float32 arrays, nothing pickled, and a
fingerprint checked on load. Models of the default size take about 1.6 MB.

```python
gen = elg.get_generator("transformer")
gen.save(model, "transformer_model/")
same = gen.load("transformer_model/")
```

## Cost

Measured on a laptop CPU (Intel Core i9-9880H, 8 training threads):

| Data | Training | Generation |
| ---- | -------- | ---------- |
| Demo course, tutorial settings | about 20 s | about 1 s per 100 sessions |
| OULAD module (short weekly sessions) | about a minute | about 1 ms per event |
| EdNet sample (10,580 sessions, median 49 events) | about 90 s per epoch | about 2.3 ms per event |

Generation is sequential, one event at a time, so it dominates the cost of
large benchmarks. The [case studies](case-study-oulad.md) compare all
generators on held-out learners.

## Limitations

- The Transformer models **one session at a time**. It captures longer
  within-session dependencies than Markov models, but not dependencies
  between the sessions of the same learner; it is not a hierarchical learner
  model.
- Event-weight controls (`Controls(event_weights=...)`) are not supported;
  time-scaling and session-length controls and calendars are.
- The default hyperparameters are untuned starting points. A more flexible
  model is not automatically more faithful: compare generators with the
  benchmark before choosing one.
