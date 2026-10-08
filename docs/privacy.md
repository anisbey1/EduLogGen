# Privacy and responsible use

EduLogGen helps teams work with educational interaction data while reducing
exposure of real learners. It provides **engineering controls and risk
indicators, not legal guarantees**. Nothing here is legal advice.

## What EduLogGen does

- **Runs locally.** EduLogGen never sends data anywhere. There is no
  telemetry and no hosted service.
- **Pseudonymizes on ingest.** Field mappings can replace identifiers with
  salted HMAC-SHA256 pseudonyms (`hash: {salt: ...}`). Keep the salt secret
  and out of version control; anyone with the salt can test guesses.
- **Drops what you do not map.** Source columns that are not mapped or listed
  under `metadata` never enter the corpus. `privacy.strip_metadata_keys`
  removes kept metadata keys as well.
- **Keeps values out of reports and errors.** Quality reports and error
  messages contain counts, field names, row numbers, and reason codes, never
  source values.
- **Remaps synthetic identifiers.** Generated learners, sessions, and events
  get fresh, shuffled IDs by default (`generation.id_strategy: remap`).
- **Measures memorisation.** Validation reports include privacy indicators:
  - `exact_session_dup_rate`: share of synthetic sessions (3+ events) that
    copy a real session exactly;
  - `rare_ngram_replay_rate`: share of real n-grams seen only once that the
    synthetic data reproduces;
  - `nn_distance_p05`: how close the closest 5% of synthetic sessions are to
    a real one (0 means copies).

## What EduLogGen does not do

- It does **not** make data anonymous. Synthetic sessions can resemble real
  ones, especially when the training set is small or behaviour is
  repetitive. Markov models with high order or no smoothing reproduce
  training paths more often.
- It does **not** provide differential privacy guarantees (roadmap).
- It does **not** certify compliance with GDPR, FERPA, or institutional
  policies.

## Before sharing synthetic data

1. Run `eduloggen validate` and read the privacy indicators, not only the
   fidelity metrics.
2. If duplicate or replay rates are high, lower the Markov order, add
   smoothing (`smoothing_alpha`), train on more data, or remove rare
   activities.
3. Check that metadata and activity IDs do not themselves identify people
   (for example free-text titles or personal pages).
4. Follow your institution's data governance and ethics review process.

## Repository rules

The repository, its tests, examples, and CI contain **synthetic data only**.
Never commit real learner data, even in issues or pull requests.
