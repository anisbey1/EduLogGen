# Case studies: OULAD and EdNet

EduLogGen applied to two public datasets with the same analyses
([`studies/`](https://github.com/anisbey1/EduLogGen/tree/main/studies), seed
0, 70/30 learner split). Only aggregate results are kept in the repository.

| Dataset | Granularity | Size |
| ------- | ----------- | ---- |
| [OULAD](https://doi.org/10.1038/sdata.2017.171), 7 modules of 2014J (CC BY 4.0) | **Aggregated**: one event per active learner-day (most-clicked activity), one session per learner-week | 10,143 learners, 623,704 learner-days |
| [EdNet-KT4](https://doi.org/10.1007/978-3-030-52240-7_13), random 2,000 students (CC BY-NC 4.0) | **Fine-grained**: every app action with a millisecond timestamp; 30-minute idle-timeout sessions | 967,968 events, 10,580 sessions (median 49 events, up to 1,575) |

OULAD has no time of day, so hour and weekday are not analysed there.

## Fidelity on held-out learners

"Real vs real" compares the training learners with the holdout learners: how
much two samples of real learners differ. Values: mean over OULAD modules /
EdNet seeds.

| Metric | Dataset | Real vs real | Independent | Markov-1 | Markov-2 | Semi-Markov | GRU |
| ------ | ------- | ------------ | ----------- | -------- | -------- | ----------- | --- |
| Bigram TVD ↓ | OULAD | 0.038 | 0.220 | 0.074 | 0.079 | 0.073 | 0.098 |
| | EdNet | 0.052 | 0.655 | 0.061 | 0.059 | 0.063 | 0.094 |
| Inter-event time KS ↓ | OULAD | 0.010 | 0.281 | 0.281 | 0.281 | 0.013 | 0.017 |
| | EdNet | 0.019 | 0.507 | 0.507 | 0.507 | 0.028 | 0.032 |
| Session-length KS ↓ | OULAD | 0.020 | 0.023 | 0.020 | 0.020 | 0.021 | 0.022 |
| | EdNet | 0.084 | 0.079 | 0.085 | 0.085 | 0.086 | 0.089 |

- Markov models cut the order-blind baseline's bigram error by two thirds
  (OULAD) to about 90% (EdNet), approaching but not reaching the
  real-vs-real reference.
- Only the semi-Markov generator and the optional GRU network reproduce
  inter-event timing. The untuned GRU is less faithful to event sequences
  than the Markov models: flexibility alone does not buy fidelity.

## Memorisation, profiles, and detection

| Diagnostic | OULAD | EdNet |
| ---------- | ----- | ----- |
| Sessions matching a once-seen training session: holdout / synthetic | 5.3–8.5% / 5.1–8.9% | 2.8% / 0.2–1.4% |
| Learners repeating a training learner's trajectory: holdout / synthetic | 0–3.1% / 0–1.4% | 0% / 0% |
| Median distance to nearest training learner: holdout / synthetic | 0.64–0.90 / 0.62–0.94 | 0.42 / 1.07–1.63 |
| Profile stability across seeds / 80% subsamples (ARI) | 0.93–1.00 / 0.78–0.95 | 0.99 / 0.98 |
| Profile replication on holdout learners (ARI) | 0.36–0.84 | 0.93 |
| Detector ROC-AUC: real / synthetic background | 0.88–0.92 / 0.88–0.92 | 0.77 / 0.90 |
| Detector: normal synthetic vs normal real (AUC) | 0.48–0.52 | 0.49 |

- **Memorisation** is measured against the training learners, with unseen
  holdout learners as the baseline. Synthetic data are no closer to the
  training learners than unseen real learners are. On EdNet they are
  further away, which also shows a learner-level realism gap: sessions are
  generated independently of each other.
- **Profiles** are stable on EdNet. On OULAD they are stable across seeds
  but only moderately reproducible on new learners. On OULAD, the
  low-engagement profile passes 0–11% of the time in six of seven modules,
  against 72–93% for the most successful profile. This is descriptive only:
  withdrawn learners stop producing activity.
- **Detection:** anomalies injected into synthetic sessions were easier to
  detect than in real ones on EdNet. Use synthetic backgrounds to compare
  detectors, not to estimate their absolute accuracy.

These diagnostics estimate risk; they do not make synthetic data anonymous.
Profiles are descriptive clusters, not learner types; anomalies are data
patterns, not misconduct.

## Cost

On a laptop (Intel Core i9-9880H, one process), OULAD takes 20–108 s per
module for the four-generator benchmark. EdNet takes 59 minutes, almost all
of it in the nearest-neighbour edit distances between long sessions.
