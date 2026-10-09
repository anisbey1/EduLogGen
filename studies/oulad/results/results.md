# OULAD study results

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
