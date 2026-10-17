# EduLogGen v1.6.0: Transformer generator, technical report

## Summary

v1.6.0 adds `transformer`, an optional time-aware autoregressive Transformer
generator (`pip install "eduloggen[neural]"`), integrated through the
existing generator contract with no change to v1.5.0 behaviour. Measured as
an untuned baseline on OULAD and EdNet, it reproduces inter-event timing but
not event sequences better than the Markov generators, at a much higher
cost. Full results: `studies/RESULTS_v16.md`.

## Implementation

| Component | Where | Notes |
| --------- | ----- | ----- |
| Generator | `src/eduloggen/generators/transformer.py` | Subclass of `BaseGenerator`; only `_fit_family` and `_sequence_sampler` are model-specific |
| Shared neural helpers | `src/eduloggen/generators/_neural.py` | Moved from `gru.py` unchanged: optional PyTorch import, gap bins, gap extraction, base64 weights, sampling; plus a learner-level validation split |
| Registration | `generators/registry.py` | Built-in name `transformer`; CLI and API work through the existing `--generator`/`--set` and `fit_generator(..., hyperparameters=)` |
| GRU | `generators/gru.py` | Opt-in `validation_split="learner"`; stored only when set, so default models keep v1.5.0 fingerprints |
| Tests | `tests/generators/test_transformer.py` (59), `test_gru.py` (+1) | Run in the CI PyTorch job, which now enforces the coverage gate |
| Docs | `docs/transformer.md`, generator tables, changelog | Tutorial run as written |
| Studies | `studies/run_v16.py`, `worker.py`, `report_v16.py`, `study.py` | Resumable; one process per generator task |

### Model

- **Inputs per position:** an event embedding (`BOS` at position 0), a
  gap-bin embedding of the gap that led to the current event (a reserved
  "no gap" id at positions 0 and 1), and a learned position embedding,
  added.
- **Outputs at position `p`:** the next event (softmax over the vocabulary)
  and, for `p ≥ 1`, the gap after the current event (softmax over gap
  bins). This is the GRU's timing convention, factorised as
  `P(e | h)·P(g | h)`: the time head does not see the sampled next event.
- **Gap bins:** fitted on inner-training learners only. There is one bin per
  distinct value when values are few (OULAD whole days), otherwise quantile
  bins; a bin is turned into a gap by drawing a stored training gap, so
  discrete values are reproduced exactly.
- **Network:** pre-norm causal blocks with explicit causal and padding
  masks, GELU feed-forward, dropout, and a ModuleDict network. The ModuleDict
  means no PyTorch class is subclassed, so type checks pass without PyTorch.
- **Loss:** `L_event + λ·L_time`, cross-entropies over real targets only.
- **Training:**
  - learner-level early-stopping split inside the fitted data;
  - sessions longer than `max_seq_len` are split into overlapping windows,
    each target scored exactly once;
  - AdamW, gradient clipping, best-epoch checkpoint, and an error on
    non-finite loss.
- **Generation:**
  - shared length, learner and start-time models (no EOS), so benchmarks
    compare like with like;
  - per-session sampling with a key/value cache, and a rolling context
    beyond `max_seq_len`;
  - temperature and top-k; one CPU thread.
- **Artifacts:** JSON with base64 float32 weights (about 1.6 MB at default
  size). The training history and counts are stored. Training time is not
  stored, so identical fits share a fingerprint.

### Verified properties

- **Causal masking:** changing future inputs never changes earlier hidden
  states (test).
- **Padding:** padding never changes real positions (test).
- **Key/value cache:** cached decoding equals the full forward pass (test).
- **Windowing:** windows score every target exactly once, for several
  lengths and overlaps (test).
- **Overfitting check:** the model learns a deterministic `a→b→c` corpus
  (more than 90% correct transitions).
- **Determinism:** fitting is deterministic (identical fingerprints) on CPU
  for a fixed thread count, and generation is reproducible. Save/reload
  generates identical output.
- **v1.5.0 compatibility:**
  - artifacts saved by the v1.5.0 release (independent, Markov, semi-Markov,
    GRU) load and generate identical output;
  - default GRU fingerprints and outputs are unchanged after the refactor.
- **Validation and benchmark:** both accept Transformer output; event-weight
  controls are refused with `generation_unsupported_control`, like the GRU.
- **Tests:** 1,086 pass with 98.5% coverage. Without PyTorch, 1,063 pass and
  23 neural tests are skipped.

## Experiments

**Protocol:**
- `session_fidelity_v1`, unchanged: learner-level 70/30 split, training-only
  fitting, three seeds, holdout-size samples, and the real-vs-real
  reference.
- **OULAD:** 7 modules of the 2014J presentation, daily activity in weekly
  sessions.
- **EdNet-KT4:** 2,000 students, 30-minute idle sessions.
- Hyperparameters are defaults; nothing was tuned for any generator.
- "GRU (matched)" uses the Transformer's validation and stopping rules:
  learner split, 100 epochs, patience 10.
- Neural training used 8 threads, recorded in each model.
- Sensitivity runs use the fidelity metrics without the slow privacy
  diagnostics; metric definitions are unchanged.
- Total compute: 18.5 h on an Intel Core i9-9880H laptop.

**Main results** (mean; SD and per-module/per-seed values in the results
file):

| | Bigram TVD ↓ OULAD / EdNet | Inter-event KS ↓ OULAD / EdNet | EdNet fit | EdNet generation |
| --- | --- | --- | --- | --- |
| Real vs real | 0.038 / 0.052 | 0.010 / 0.019 | | |
| Markov-1 | 0.074 / 0.061 | 0.281 / 0.507 | 10 s | 0.04 ms/event |
| Semi-Markov | 0.073 / 0.063 | 0.013 / 0.028 | 10 s | 0.04 ms/event |
| GRU (v1.5.0 defaults) | 0.098 / 0.094 | 0.017 / 0.032 | 6 min | 0.31 ms/event |
| GRU (matched) | 0.095 / 0.079 | 0.014 / 0.028 | 17 min | 0.23 ms/event |
| Transformer | 0.096 / 0.108 | 0.019 / 0.043 | 84 min | 2.4 ms/event |

**Findings:**

1. **Sequences:** with untuned defaults, neither neural generator matches
   the Markov models on event sequences. On EdNet the Transformer is the
   least faithful neural generator, including on the event mix (TVD 0.097
   against 0.039 between real samples).
2. **GRU training rules:** matching the GRU's validation and stopping rules
   to the Transformer's improved its EdNet sequences (0.094 → 0.079). The
   v1.5.0 defaults stop early.
3. **Context length** (EdNet; 32/64/128/256 events) is non-monotonic: the
   bigram TVD is 0.069, 0.115, 0.128 and 0.104. The shortest context did
   best; longer contexts did not help these untuned models and cost up to
   4× more training time.
4. **Training-data size** (nested 10/25/50/100% of the training learners):
   Markov and semi-Markov improve steadily with more data (EdNet bigram
   0.148 → 0.061). The Transformer is non-monotonic (EdNet 0.170, 0.083,
   0.147, 0.104), which points to run-to-run training variability that a
   single training seed per configuration cannot separate from a data-size
   effect.
5. **Memorisation:** neural samples sit closer to the training data than
   Markov samples on EdNet (6.6–7.6% vs 1.8–2.9% of sessions match a
   training session). All generators stay well below unseen real learners
   (20%), and no synthetic learner repeats a training learner's trajectory.

These are measurements of one untuned configuration per model. They are not
evidence about Transformers or recurrent networks in general.

## Limitations and open issues

- **No tuning:** hyperparameter tuning on inner-validation learners, with
  equal budgets for the GRU and the Transformer, and several training seeds
  per configuration, would be needed to separate architecture from
  optimisation effects.
- **One session at a time:** the Transformer models sessions independently,
  so there are no dependencies between a learner's sessions. It is not a
  hierarchical learner model.
- **Batching:** cross-session batch generation would change the random
  stream shared by all generators, so it was not implemented. Generation is
  per session with a key/value cache, and is the main cost on long
  sessions.
- **GPU:** the `cuda`/`mps` training paths are implemented but untested
  (CPU-only machine and CI). GPU training is not guaranteed to be
  deterministic.
- **Determinism scope:** exact numerical equivalence across thread counts,
  PyTorch versions and hardware is not guaranteed.

## Release status

- **Branch:** `feature/v1.6.0-transformer`, version 1.6.0, changelog dated
  2026-10-17. Not merged, tagged, pushed, published to PyPI, or archived on
  Zenodo.
- **v1.5.0:** the tag and its archive are untouched.
