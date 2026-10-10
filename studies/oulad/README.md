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
python studies/report.py --oulad studies/oulad/out/results.json \
    --ednet studies/ednet/out/results.json --output paper/softwarex
```

`run.py` applies the analyses in `studies/study.py` to each presentation
(seed 0, 70/30 learner split):

1. **Fidelity**: the `session_fidelity_v1` benchmark for the independent,
   first- and second-order Markov, semi-Markov, and GRU generators (the GRU
   needs `pip install "eduloggen[neural]"`), next to the real-versus-real
   reference.
2. **Profiles**: `auto` profiles (k = 4) on training learners; stability
   across seeds, 80% subsamples, and on holdout learners; relation to final
   results (descriptive); recovery of a synthetic profile mixture.
3. **Memorisation**: synthetic data and unseen holdout learners compared with
   the training learners, per session and per learner.
4. **Detection**: unexpected transitions and repetitions injected into real
   and synthetic sessions; a transition-likelihood detector fitted on
   training learners; an artefact check (normal synthetic vs normal real).

Aggregate results (no learner data) are kept in `results/results.json`.

Profiles are behavioural clusters, not validated learner types. Anomalies
are data patterns, not misconduct.
