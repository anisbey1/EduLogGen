# Case study: OULAD

EduLogGen applied to the seven modules of the 2014J presentation of the
[Open University Learning Analytics Dataset](https://doi.org/10.1038/sdata.2017.171)
(OULAD; CC BY 4.0): 10,143 learners and 623,704 active learner-days. All numbers
are reproduced by the scripts in
[`studies/oulad/`](https://github.com/anisbey1/EduLogGen/tree/main/studies/oulad)
with seed 0.

**Representation.** OULAD records clicks per learner, resource, and *day*,
without time of day. An event is an active learner-day (its most-clicked
activity type), and a session is a learner's week of the module, so sequences
and gaps between events (whole days) are observed rather than invented. Hour
of day and weekday are not analysed.

**What it shows.**

- Every generator matches the event mix and session lengths within the
  variation between two samples of real learners ("real vs real").
- Markov generators cut the bigram error of the order-blind baseline by about
  two thirds; only the semi-Markov generator reproduces the gaps between
  active days.
- Exact duplicate sessions must be read against the reference: 77% of real
  holdout sessions also match a training session, because weekly patterns of
  at most seven days repeat naturally.
- Automatic profiles, learned on training learners only, separate
  low-engagement learners (0–11% pass in six of seven modules) from the most
  successful profile (72–93%). They are descriptive clusters, not learner
  types; withdrawn learners stop producing activity, so persistence explains
  part of the association.
- A simple transition-likelihood detector reaches ROC-AUC 0.88–0.92 on
  injected anomalies, a known-truth baseline for better detectors.
- The whole study runs in about ten minutes on a laptop.

## Fidelity (mean over presentations)

| Metric | Real vs real | Independent | Markov (1) | Markov (2) | Semi-Markov |
| --- | --- | --- | --- | --- | --- |
| Event-type TVD | 0.022 | **0.021** | 0.028 | 0.031 | 0.027 |
| Bigram TVD | 0.038 | 0.220 | 0.074 | 0.079 | **0.073** |
| Transition JSD | 0.002 | 0.049 | **0.003** | 0.003 | 0.003 |
| Session length KS | 0.020 | 0.023 | **0.020** | 0.020 | 0.021 |
| Inter-event time KS | 0.010 | 0.281 | 0.281 | 0.281 | **0.013** |
| Top-10 path overlap | 0.900 | 0.700 | 0.833 | 0.814 | **0.838** |
| Exact duplicate sessions | 0.770 | 0.657 | 0.726 | 0.747 | 0.726 |

## Per presentation

| Presentation | Learners | Learner-days | Sessions | Profiles (k) | Recovery ARI | Outcome NMI (holdout) | Detector ROC-AUC | Detector AP | Run time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AAA-2014J | 357 | 30,703 | 9,914 | 4 | 0.91 | 0.116 | 0.907 | 0.547 | 26 |
| BBB-2014J | 1,921 | 83,111 | 36,982 | 4 | 0.86 | 0.141 | 0.917 | 0.535 | 95 |
| CCC-2014J | 2,302 | 131,279 | 47,579 | 4 | 0.78 | 0.103 | 0.896 | 0.570 | 140 |
| DDD-2014J | 1,647 | 109,161 | 38,264 | 4 | 0.84 | 0.145 | 0.881 | 0.508 | 104 |
| EEE-2014J | 1,097 | 76,751 | 26,198 | 4 | 0.88 | 0.302 | 0.901 | 0.514 | 70 |
| FFF-2014J | 2,121 | 167,164 | 50,431 | 4 | 0.92 | 0.199 | 0.886 | 0.532 | 136 |
| GGG-2014J | 698 | 25,535 | 12,467 | 4 | 0.85 | 0.198 | 0.898 | 0.471 | 25 |
