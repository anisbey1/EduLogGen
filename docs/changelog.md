# Changelog

Release history for EduLogGen follows
[Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/).

The canonical changelog lives in the repository root as
[`CHANGELOG.md`](https://github.com/anisbey1/EduLogGen/blob/main/CHANGELOG.md).

## [1.5.0] - 2026-10-10

An optional GRU neural generator (`eduloggen[neural]`), and reproducible
case studies on OULAD (aggregated weekly activity) and EdNet (fine-grained
clickstreams) for the SoftwareX article.

## [1.4.2] - 2026-10-09

Author and citation metadata (Anis Bey, ORCID 0000-0001-9410-0851).

## [1.4.1] - 2026-10-09

Ready for public release: PyPI publishing workflow, citation metadata, an
OULAD case study, and more robust automatic profiles on real data (rare
activities pooled, capped z-scores, back-off for very short sessions).

## [1.4.0] - 2026-10-09

Level 2, milestone M3: behavioural profiles learned from data (`auto`),
imported (`provided`), or defined without data (`manual`); profile mixtures
with ground-truth annotations; `evaluate_clustering`. Generators now also
learn when sessions start, so synthetic sessions follow real hours and days.

## [1.3.0] - 2026-10-09

Fine-grained analysis: per-activity profiles, temporal analysis with
deadline effects, stratified analysis by course, week, cohort, or device, and
a ranked comparison of where synthetic data differs from real data.

## [1.2.0] - 2026-10-09

Level 2, milestone M2: behavioural controls, course calendars with deadline
surges, and manipulation checks (`eduloggen generate --experiment`).

## [1.1.0] - 2026-10-09

Level 2, milestone M1: five labelled anomaly types, ground-truth
annotations stored separately from events, and detector scoring
(`eduloggen evaluate`).

## [1.0.0] - 2026-10-09

First stable release, completing Level 1 (the log generator). See the
repository `CHANGELOG.md` for the full list of additions.

## [0.1.0] - 2026-07-29

### Added

- Initial production-quality package skeleton (`src` layout)
- Package metadata and build configuration via `pyproject.toml`
- Tooling: Ruff, Black, MyPy, PyTest, pre-commit, MkDocs
- GitHub Actions continuous integration workflow
- Documented subpackage namespaces for ingestion, analysis, generators,
  validation, visualization, and CLI
- MIT license and Semantic Versioning policy
