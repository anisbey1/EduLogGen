# EduLogGen examples

Small, self-contained examples that use synthetic toy data only—never real
learner identifiers or institutional exports.

## Configuration

- [`configs/minimal.yaml`](configs/minimal.yaml) — a complete experiment
  configuration covering every section (SAD §29M.1).
- [`configs/demo_mapping.yaml`](configs/demo_mapping.yaml) — a field mapping
  for a generic LMS export, including salted ID pseudonymization and a
  timezone for naive timestamps.

Both files are loaded by the test suite, so they stay valid as the schema
evolves.
