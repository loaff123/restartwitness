# Frozen controlled evaluation: measured results

Protocol SHA-256: `ca53adf18c73b01ceb7c445df3f87d5f68351c9a1e538faf0d8fb0e88356e539`.

All256 planned experiment cases completed and were independently re-adjudicated from raw arrays. No evaluation errors, unrun cases or unmet declared acceptance expectations occurred. These are original hand-selected fixtures; no real-world defect-rate, novelty or general-superiority claim follows.

## Acceptance

All61 expectations were met:39 clean model/schedule combinations,12 applicable deliberate faults,7 controls, and3 native-profile runs (REBOUND checkpoint, OpenMM checkpoint, and the intentionally weaker OpenMM State-only workflow). Unsupported/timeout/nondeterministic/observer/malformed-output controls remain their explicitly limited statuses; “61 expectations met” does not mean61 numerically equivalent simulations.

## Comparisons

| Method | Deliberate faults detected | Completed steps | Process starts | Recorded group wall time |
|---|---:|---:|---:|---:|
| Fixed full-trajectory cuts |11/12|19,200|360|247.70s|
| Seeded-random full-trajectory cuts |10/12|19,200|360|265.78s|
| Boundary-directed full-trajectory cuts |12/12|19,200|360|269.98s|
| Conventional midpoint/final-only |5/12|4,800|90|17.85s|

The first three each use1,280 steps per fixture including U1/U2/O/P/R controls. Conventional has a smaller budget and is excluded from equal-budget claims. Wall times are coordinator measurements, not an independently attested timing result or a speed comparison. All three fair methods had zero flagged clean fixtures in this corpus.

Fixed cuts missed the controlled skipped-boundary-output fixture. Random cuts missed the duplicate-boundary and skipped-boundary fixtures. The explicit boundary subset was `[7],[8],[9],[32]`; it was frozen before outcomes and does not imply discovery of undeclared application events.

## Same-cut history witness

For `acceptance/fault-restored_callback`, the cut was7. The final ordinary P/R snapshots agree bit-for-bit, while full observation history first differs at logical step8. The earlier output means retain the difference. This derived check uses the already frozen raw case, isolating the limited view of a final-only comparison at the same cut.

## Reduction demonstration

A separate controlled cache-omission witness shrank from `[7,8,9]` to `[9]` in10 complete evaluator calls. Each accepted candidate reproduced the same restore-path/x signature twice. Removing the remaining cut passed twice; the final candidate was reconfirmed twice. The result is `one_minimal`, not a claim of globally optimal cut placement. All10 trial bundles and the starting bundle were independently checked.

The267 raw case bundles (256 evaluation +10 reduction trials +1 reduction input) are retained in the complete download. They contain over560,000 small numeric/metadata files. The compressed archive is small, but allow roughly3GB of free disk to extract everything. Use the small per-case downloads for ordinary inspection.

## Reproduction and source association

The numerical evaluation used local commit `71ca6c071e14fd2309270752544a8fb6e0d0b071`. The reduction used the reader-hardened source at `9d914aeb403e4c4ebd30b5dfa11a730b72bb80de`. Both commits are retained in the downloadable Git bundle; the evaluation source and its original wheel are also exported directly. Later release fixes improve malformed-file handling and short-horizon examples; they do not alter the frozen fixture equations, seeds, tolerances, schedules, or comparator.

Exact replay deliberately requires matching recorded source/environment identities. Offline verification does not require the original environment. In a different environment, run the extracted study as a new experiment instead of calling it an exact replay. See `reproduction.md`.

Historical affected/fixed reproduction, scientific-domain review and independent adoption remain absent. Research and impact gates remain closed.
