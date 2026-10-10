# EdNet study (fine-grained clickstreams)

Uses EdNet-KT4 (Choi et al., AIED 2020; CC BY-NC 4.0, research use): every
action of students of the Santa TOEIC app with a millisecond timestamp.
Download KT4 from <https://github.com/riiid/ednet> (about 1.2 GB) and keep
the zip outside version control.

```bash
python studies/ednet/prepare.py --kt4 EdNet-KT4.zip --output studies/ednet/data \
    --students 2000 --seed 0
python studies/ednet/run.py --data studies/ednet/data/ednet-kt4-2000.csv \
    --output studies/ednet/out
```

Event type = action plus item kind (e.g. `respond:question`,
`play_audio:bundle`); sessions use a 30-minute idle timeout. The analyses are
those of `studies/study.py` (see `studies/oulad/README.md`); EdNet has no
final results, so profiles are not related to outcomes. Aggregate results
(no learner data) are kept in `results/results.json`.
