# Response to the review

We thank the reviewer for a careful and constructive review. Each comment is
answered below, with the changes made.

## Major comments

**1. OULAD evaluates aggregated activity, not ordinary event-level logs.**
We agree. Section 3.1 now calls OULAD a benchmark of *aggregated* weekly
activity. It states the consequences of dominant-activity aggregation:
sequences have at most seven events, within-day order and secondary
activities are lost, and gaps are whole days. We added a fine-grained
dataset, EdNet-KT4 (CC BY-NC 4.0). It has millisecond-timestamped app
actions (entering items, answering, erasing choices, submitting, playing and
pausing media), sessionised with a 30-minute idle timeout. Our random sample
of 2,000 students gives 967,968 events in 10,580 sessions: median 49 events,
29% with 100 or more, at most 1,575. All analyses were run on both datasets with the
same code (`studies/study.py`).

**2. Fidelity claims stronger than the numbers.**
We adopted the suggested wording (Section 3.2). It no longer claims that
generators match "within the real-versus-real range" or that they are
equivalent to real data. The bigram error of the Markov models is described
as about twice the reference on OULAD and 1.1–1.2 times on EdNet. Table 1 now reports mean ± SD: over modules
for OULAD, over seeds for EdNet. The supplementary material gives every
module with seed-level SDs. Table 1's caption and S1 define the reference:
the training learners, treated as a sample, are compared once with the same
holdout that each generator sample is compared with.

**3. Circularity of profile evaluation.**
Section 3.4 now separates the three claims the reviewer listed:
- *Reconstruction (synthetic):* ARI 0.78–0.92 on OULAD, 0.75 on EdNet. Explicitly described as
  showing only that the generator preserves its own profiles.
- *Stability on real data (new):* on OULAD, ARI across clustering seeds
  0.93–1.00 and across 80% learner subsamples 0.78–0.95; replication on
  independent holdout learners was only 0.36–0.84, which we report as showing
  that profile boundaries are not uniquely determined there. On EdNet the
  profiles were very stable (0.99, 0.98, and 0.93).
- *Relation to outcomes:* kept, with the persistence caveat. It is described
  as neither predictive nor causal evidence.

We do not claim downstream usefulness of profile-based synthetic data for
outcome prediction. Simulated outcomes are planned work (Section 5).

**4. Privacy: diagnostics vs guarantees.**
Answering the reviewer's questions led us to correct the manuscript. The
benchmark's duplication diagnostic compares synthetic data with *holdout*
learners, so it does not measure memorisation of training data. We added a
memorisation analysis against the *training* learners, with unseen holdout
learners as the baseline (Section 3.3, Table 2, S1). It answers the
reviewer's questions as follows:
- *Granularity:* stated explicitly for each diagnostic. Session level means
  token sequences of at least three events; learner level means the whole
  ordered sequence of sessions, and distance to the nearest training learner
  in standardised behavioural features.
- *Rare patterns:* synthetic sessions reproduce sessions seen only once in
  training about as often as unseen real learners do (differences of at
  most 2.3 percentage points, in both directions).
- *Rare multi-week trajectories:* no more synthetic than holdout learners
  repeat a training learner's trajectory. The generators sample sessions
  independently, so they cannot reproduce rare trajectories by design.
- *Exposure of unusual learners:* no more synthetic than holdout learners
  have a feature-identical training learner, and median distances are
  similar or larger.
- *EdNet:* 1.8–2.9% of synthetic sessions match a training session, against
  20% of holdout sessions. Synthetic learners are further from the training
  learners than real unseen learners are (median distance 1.26–1.63 against
  0.42). We report this both as low memorisation and as a learner-level
  realism gap.

The text states that these diagnostics do not make the output anonymous or
safe for unrestricted sharing. A membership-inference study is left for
future work.

**5. Anomaly-detection protocol.**
Section 3.5 and S1 now specify:
- the two anomaly types go into disjoint random 5% samples, at most one per
  session;
- how many events each inserts: two for an unexpected transition, four for a
  repetition;
- that the "invalid workflow" label comes from construction: a pair never
  observed among training learners;
- that labels are per session, and that insertions lengthen sessions, while
  the detector uses transitions only;
- that the detector is fitted on training learners and evaluated on modified
  *real* holdout sessions.

The text states that this is an injection experiment, not detector
performance on natural anomalies. We added the suggested artefact check: the detector
cannot separate normal synthetic from normal real sessions (AUC 0.48–0.52 on
OULAD, 0.49 on EdNet). We also injected the same anomalies into synthetic
sessions. On OULAD this gave the same ROC-AUC as real sessions, but on EdNet
a higher one (0.90 against 0.77). The paper now warns that synthetic
backgrounds can overstate absolute detector performance and are better
suited to comparing detectors. Per-type results are in the supplement:
unexpected transitions were easier to detect than repetitions on OULAD and
harder on EdNet.

## Additional change

To address a likely question about neural baselines, we added an optional
`gru` generator, `pip install "eduloggen[neural]"`. It is a recurrent
network over the session history that predicts the next event type and the
time spent after the current event. It was evaluated under the same protocol
on both datasets:
- *Timing:* close to the semi-Markov generator (inter-event KS 0.017 on
  OULAD, 0.032 on EdNet).
- *Sequences:* untuned, it was less faithful than the Markov models (bigram
  TVD 0.098 and 0.094).
- *Memorisation:* on EdNet it came closest to the training data among the
  generators (7.1% of sessions match a training session), but still well
  below unseen real learners (20%).

We report this as showing that model flexibility alone does not guarantee
fidelity, and that the framework makes such comparisons reproducible.

## SoftwareX-specific points

**A. Word limit.** About 2,780 words (abstract, body, captions), plus about
100 words of code listings. This is within both the 3,000- and 4,000-word
figures. Details were moved to the supplementary material. There are two
figures.

**B. Impact.** Section 4 now separates *demonstrated* impact (the analyses
shown on public data, reproducible from the released scripts), *potential*
impact, and *availability*. It states that no external adoption is claimed
yet.

**C. Versioned artifact.** The paper cites v1.5.0, which contains the study
scripts and aggregate results, and its version-specific Zenodo DOI
(10.5281/zenodo.23281595, C3), in addition to the concept DOI. We checked that the archived release matches
the reported functionality. A fresh `pip install eduloggen` in a new
environment runs both code listings unmodified and reproduces the snippet's
ARI exactly. Re-running the full OULAD study on the released package
reproduced all reported values (largest relative difference 4×10⁻¹⁴, from
floating-point summation order).

## Detailed comments

- **Title:** shortened to "EduLogGen: A Python package for generating and
  validating synthetic educational interaction logs".
- **Abstract:** OULAD described as daily activity aggregated into weekly
  sequences; EdNet added; the fidelity claim qualified; the detection caveat
  added.
- **§1:** sharper comparison with tabular synthesisers, process-mining log
  simulators, and the lack of held-out validation and disclosure
  measurement.
- **§2.2:**
  - *Semi-Markov semantics:* the gap depends on the current event type only
    (not the next); empirical by default; within sessions only; zero gaps
    become 1 ms; between-session gaps and start times are modelled
    separately.
  - *Privacy granularity:* given for each diagnostic.
  - *Calendars:* described as a controlled capability, not validated
    calendar fidelity (also repeated in §5).
- **§3.6 Cost:** states what the timing includes (ingestion,
  sessionisation, analyses) and excludes (raw-data conversion, report
  generation). On EdNet, the benchmark took 59 minutes, almost all of it in
  nearest-neighbour edit distances; this is now listed as a limitation.
- **Fig. 1:** redrawn at print size so labels stay at least about 7 pt.
- **Fig. 2:**
  - colour-blind-safe Okabe–Ito palette with hatching;
  - stacked panels;
  - the caption explains the abbreviations;
  - EdNet added.
- **Table 1:** the exact-duplicate row was removed from the fidelity table.
  It is a diagnostic, now discussed in §3.3 with its definition.
- **Table 2:** new; ARI defined in the caption. The former per-module table
  moved to the supplement.
- **Title page:** ORCID added.
- **Code metadata:** C1 and C2 point to the tagged release; C3 is the
  version DOI.
