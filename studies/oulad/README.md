# OULAD study

Reproduces the evaluation in the EduLogGen software paper on the Open
University Learning Analytics Dataset (OULAD; Kuzilek, Hlosta & Zdrahal,
*Scientific Data* 4, 170171, 2017; CC BY 4.0).

## Data

Download OULAD (for example from the UCI Machine Learning Repository,
dataset 349) and unpack the seven CSV files into a folder, e.g. `OULAD/`.
The data are not part of this repository.

## Representation

OULAD records clicks per student, resource and **day**, with no time of day,
and the order of rows within a day has no meaning. `prepare.py` therefore
builds a representation where everything is observed:

- **event** = one active learner-day; event type = activity type with the
  most clicks that day; activity = most-clicked resource;
- **session** = one learner's week of the module (`floor(day / 7)`), so a
  session is the ordered sequence of active days and gaps are whole days;
- timestamps = presentation start (1 February for B, 1 October for J) plus the
  day offset at 12:00 UTC. **Hour of day and weekday are not observed**; the
  study does not analyse them.

## Steps

```bash
python studies/oulad/prepare.py --oulad OULAD --output studies/oulad/data \
    --presentations AAA-2014J BBB-2014J CCC-2014J DDD-2014J EEE-2014J FFF-2014J GGG-2014J
python studies/oulad/run.py --data studies/oulad/data --oulad OULAD --output studies/oulad/out
python studies/oulad/report.py --results studies/oulad/out/results.json --output paper/softwarex
```

`run.py` performs, per presentation, with seed 0:

1. **Fidelity**: the `session_fidelity_v1` benchmark (fit on 70% of learners,
   compare three seeded samples with the 30% holdout) for the independent
   baseline, first- and second-order Markov, and semi-Markov generators, next
   to the real-versus-real reference (training vs holdout learners).
2. **Profiles**: `auto` profiles (k = 4) on training learners only; holdout
   learners assigned to the nearest profile; profiles cross-tabulated with
   final results (descriptive only); recovery of a synthetic profile mixture
   (ARI, NMI, purity).
3. **Detection**: 5% unexpected transitions and 5% repetitions injected into
   real holdout sessions; a smoothed transition-likelihood detector fitted on
   training learners; ROC-AUC and average precision.

Profiles are behavioural clusters, not validated learner types. Anomalies
are data patterns, not misconduct.
