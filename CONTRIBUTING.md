# Contributing

Use a new branch, install `.[test]`, write a failing behavioral regression, and run `pytest`, `ruff check src tests experiments`, and `ruff format --check src tests experiments` before proposing changes. Native tests require separately installed pinned libraries via `.[native]` and are visibly skipped otherwise.

Keep the core driver-agnostic. Preserve direct U/P/R comparisons, identical one-step calls, non-emitting inspection, semantic output identity, unknown/incomplete statuses and safe evidence-only verification. Add adversarial tests to both production and independent readers when the evidence schema changes.

Never edit a frozen protocol to improve measured outcomes. Introduce a new version, justify it, and retain the old results. Controlled fixtures must remain labelled intentional mutations. A full research claim needs historical affected/fixed reproduction and human scientific-domain review.

No unsolicited upstream issues or automated maintainer outreach is part of this project. Respect each upstream contribution policy, including REBOUND's restriction on AI-generated issues/PRs.
