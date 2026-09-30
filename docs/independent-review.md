# Independent engineering review: developer preview

Review date: 2026-09-30. Scope: the local Python/CPU implementation and its evidence semantics. This is an engineering review by a separate implementation worker, **not an independent scientific/domain review**, an upstream defect claim, or approval for a research-positioned release.

## Independent reader

Run from the repository using the environment with NumPy installed:

```sh
python -I tests/independent_check.py path/to/case
```

Exit zero means the raw evidence supports its stored adjudication; a correctly recorded failure case can therefore verify successfully. A nonzero exit prints `evidence_integrity_error`. `--ignore-cached` returns the raw re-adjudication for diagnosis without accepting the cached labels. It still enforces integrity, contract, cadence, and checkpoint-ledger checks.

`tests/independent_check.py` imports no production comparator, contracts, evidence loader, driver, or runner. Its implementation has a separate loader, explicit contract validation, scalar numeric comparison, exact rational tolerance arithmetic, output-key validation, and adjudication. Only NumPy and the standard library are used. The standalone test executes the script with Python's `-I` isolated mode, and a source-import test guards the absence of production imports. This is an independent implementation of a shared specified contract, not an independent origin of that contract.

The reader:

- Checks the complete artifact inventory, SHA-256 hashes and sizes, duplicate JSON keys, relative paths, symlink ancestors, and file-count/byte budgets
- Parses NPY headers before allocation; forbids pickle/object/complex arrays, oversized dimensions and payloads, unsupported headers, and truncated or trailing payloads
- Caches unique loaded arrays but also bounds expanded logical references, so repeated references cannot silently multiply work without limit
- Reconstructs complete observation cadence and phase order from the frozen study and schedule, including initial/final and repeated cuts; incomplete arms must have valid observed prefixes
- Recomputes U1–U2, U1–O, U–P, P–R and U–R, including final-ordinary observation neutrality, rather than assuming tolerance agreement is transitive
- Recomputes first witnesses, contract metadata, exact float bit patterns, full-width integer values, per-array first indices and mismatch counts, nonfinite errors, and symmetric tolerance bounds
- Checks ordered semantic output keys for duplicate, missing, unexpected and reordered rows; row numbers never replace semantic keys
- Checks frozen study/driver/termination metadata against each retained worker request and checks complete-arm aggregate arrays against the original per-segment raw arrays
- Checks native-checkpoint acknowledgment artifacts against the retained opaque files and requires scheduled save coverage in complete P/R arms. It never opens native checkpoint contents through an application library
- Checks cached overall/pair status, findings, and every stored first witness, including nested trajectory/output/table first witnesses when present

The reader intentionally does not execute a driver or native checkpoint, reproduce a scientific result, authenticate an author, independently attest a historical process, or certify full hardware/software identity. Hashes alone do not prevent an author coherently rewriting all evidence and metadata. Bounded scalar re-adjudication prioritizes auditability over high-throughput analysis.

## Engineering findings and resolutions

1. **Ordinary observation occurrence mismatch:** the worker originally emitted occurrence `-1` while the comparator rejected negative occurrences. Reported before integration; worker now uses `0` and edge-cut/runner integrations verify.
2. **Common observation deletion:** hashing and pairwise comparison alone accepted deleting the same observation from all arms. Independent cadence reconstruction now rejects this, as does the coordinator's verified-reader path.
3. **Coherently rehashed raw changes:** the checker rejects a stale pass even if an altered numeric file's manifest hash is updated. It also rejects stale fault labels and invented witness locations/values while retaining the independently computed result for diagnosis.
4. **Checkpoint acknowledgment coverage:** verifying only artifacts that happen to be listed accepted removing an entire checkpoint acknowledgment. A real counter run reproduced this bypass in the coordinator reader; the independent checker requires the complete cut list, cut/step consistency, nonempty artifact lists and exact artifact sets. The coordinator added the same scheduled acknowledgment invariant and segment-count/status/process-token checks.
5. **Source changes during a run:** a temporary counter driver appended a harmless comment to its own source after save. The original coordinator returned `equivalent_under_contract` with a now-stale source hash. This was reproduced with fresh workers. The coordinator now samples identity before every worker, after each arm, and before sealing; drift receives an explicit integrity failure. Concurrent source edits must not be represented by one unqualified source identity.
6. **Verification complexity:** reconstructing sparse expected observations by scanning every step and using list membership is quadratic for dense large studies. The reader now uses the union of declared observation/cut steps; the coordinator was updated to the same asymptotic approach.
7. **Frozen configuration provenance:** changing study.json config and coherently updating the manifest originally left worker requests describing different inputs. The independent checker now compares frozen study keys, cut schedules and arm/segment boundaries to every retained request, and checks driver/termination metadata consistency. The coordinator also checks frozen study/schedule digests and request/metadata consistency.
8. **Duplicated raw evidence:** substituting P observation references into R.json and recomputing cached pass labels originally contradicted untouched R segment arrays without rejection. The independent checker and coordinator now check byte-exact complete-arm aggregate versus original per-segment raw observations and final outputs. Incomplete/killed segments remain unknown.
9. **Tolerance at rounded boundaries:** a late adversarial check found that extended precision alone accepted `1` versus `-minimum-subnormal` at absolute tolerance `1`: the true binary-float difference is slightly greater than the bound. The independent scalar Fraction checker classifies this case as a difference. The production comparator now uses an exact rational fallback near rounded boundaries, in addition to overflow handling. Its expanded numerical suite passed 77 tests, and the post-fix full suite passed 354 tests.
10. **Top-level first witness order:** pair witnesses follow recorded observation order. The overall first witness is selected in fixed comparison order, which is not necessarily the earliest logical time across all comparisons. Reports must make that distinction rather than imply a cross-pair chronological minimum.

## Process, provenance, and checkpoint review

The coordinator launches each segment with a new `sys.executable -m restartwitness.worker` process. R restores from its checkpoint path and reconstructed configuration; no simulation state is passed through forked interpreter memory. Unit/integration checks exercise distinct process tokens and segment counts, repeated cuts, initial/final cuts, and acknowledged abrupt exit. Process identifiers/tokens retained in evidence provide consistency information; they are not cryptographic process attestation.

Native checkpoint files remain opaque during verification. Workers reject symlinked checkpoint artifacts and require nonempty persisted data. The coordinator checks the completion acknowledgment before moving to another R segment. Normal termination and `exit_after_save` are separate declared modes. A successful acknowledgment or restart does not establish fsync behavior, crash consistency, or power-loss durability.

The source identity includes top-level core Python source hashes, the selected driver's source hash, built-in adapter sibling Python source hashes, selected package versions, Python version, OS and architecture. It does **not** exhaustively identify custom-driver imported helper modules, dependency binary build hashes, CPU instruction details, all environment variables, thread scheduler behavior, or transitive system libraries. Passing an exact profile remains local empirical evidence, not a portability guarantee. Replay requires explicit trust in executable driver code. Trusted drivers can access the local system; worker separation is not a security sandbox.

## Observation and attribution review

The coordinator advances one logical step at a time in all arms. U/P/R diagnostic calls are phase-aligned; O deliberately removes intermediate observations to test observation neutrality. Diagnostic snapshots are non-emitting by adapter contract. The harness cannot prove purity or discover unobserved internal state, and two baseline repetitions cannot establish stochastic determinism.

The recorded key's step is the harness counter. The supplied examples separately observe application step/time in numeric fields; a third-party adapter must explicitly expose the application clock/state it wants checked. No physical-time or adaptive-event coverage is inferred from integer cut placement. The adapter contract's narrower field selection must be visible to users.

Baseline and observation-control failures prevent causal attribution to save or restore. U–P and P–R localize an intervention associated with a discrepancy, not a hidden-variable root cause. Direct U–R catches nontransitive tolerances. Output-table schema failures remain failures even if all arms make the same mistake.

## Budgets and reduction review

The coordinator has per-worker and per-case wall limits, logs simulated advance calls/process starts, and explicitly counts unknown work from killed/error segments. These are measured/declared limits, not speed guarantees. The coordinator now crosschecks cached process/step/unknown-work counts against the segment ledger. The standalone checker independently checks equivalence and witnesses; it does not itself independently derive those counts or measure wall time. Its success must not be advertised as an independent timing audit. File-size limits for final evidence do not make arbitrary trusted drivers resource-sandboxed while they write checkpoints or run. A timed-out worker may have completed unknown scientific work before termination; that work must not be presented as a known zero-cost trial.

The reducer requires two clean trials of the same failure signature before accepting a candidate, records every evaluation, treats exceptions/timeouts/mixed repeats as unknown, and performs final single-removal verification plus reconfirmation. It does not assume monotone failure. `one_minimal` is deletion-local and must not be described as globally minimal. The evaluator/caller owns the frozen configuration, clean run directories, actual worker execution, and any total wall budget; the reducer alone cannot enforce those external obligations.

## Validation record and release boundary

The checker tests were written and observed failing before implementation, then rerun after each change. The adversarial suite covers stale labels, coherent rehashing, malformed/unsafe evidence, declared cadence, repeated phase order, exact/tolerance boundaries, signed zero, large unsigned integers, nonfinite values, output identity, worker error classifications, independent invocation, and real runner evidence. The independent-reader suite reached 70 passing tests. An initial whole-project run during concurrent implementation produced 311 passes and 21 failures: the report module had not been created for 15 report/CLI cases, two cases detected concurrent source changes, and four newly introduced coordinator-ledger regressions exercised a previously loaded implementation. A later stable-source run recorded 350 passes in 127.21 seconds with no failures, inspected directly in .superpowers/full-final.log. After tolerance-boundary and metrics-ledger hardening, the final full suite recorded **354 passes in 168.51 seconds, zero failures**, inspected directly in .superpowers/full-release.log. This count includes the 70 independent-reader tests. These are observed local Linux results, not unexecuted macOS/Windows or scientific-domain validation claims.

This review does not establish historical reproduction, equal-budget scientific superiority, independent external adoption, or a scientific-domain review. Those research/release gates remain separate. An honest developer preview may document verified behavior and the remaining limits without claiming those later outcomes.

## Release-owner follow-up

After this independent review, the release owner added regression-backed rejection of nonregular files, normalization of malformed case objects, and fixes for short-horizon example contracts/default cuts. Final coordinated verification passed363 tests in146.24 seconds. The frozen256-case experiment remains associated with71ca6c0; exact source is retained. These later changes were not represented as a new independent scientific review.
