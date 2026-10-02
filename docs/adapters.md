# A trusted adapter in six operations

Implement an importable Python module. See `src/restartwitness/adapters` for complete original examples.

1. `create(config, run_dir)` returns new application state from declared inputs
2. `advance_one(state)` completes exactly one logical integration step, including its callbacks and real application output
3. `observe(state)` returns numeric NumPy arrays without changing state or writing output; return an application `step` and physical `time` where applicable
4. `save(state, checkpoint_dir)` invokes the application's existing persistence path and returns only after its nonempty artifacts are complete
5. `restore(config, checkpoint_dir, run_dir)` reconstructs a state in a fresh interpreter solely from configuration and checkpoint; never import parent state or replace native serialization with a test-specific shortcut
6. `collect_outputs(run_dir)` returns tables with ordered native semantic `keys`, numeric `fields`, and `units`. Do not invent row identities during collection

`UNITS` maps snapshot fields to literal unit labels. No conversion, resampling, deduplication, interpolation, missing-field imputation, or automatic tolerance adjustment occurs.

Each arm has a separate output directory. R reuses its own directory across segments to preserve append behavior. Checkpoints have new segment/cut directories. U1/U2/O never save; P saves but does not restore; R does both. Default termination is normal interpreter exit; abrupt exit75 is permitted only after completion acknowledgment.

The runner's observation key `[logical_step, phase, occurrence]` is separate from your observed application counters. Ordinary occurrence is0; save/restore occurrence is the zero-based cut index, so `[8,8]` has distinct diagnostic keys without duplicate ordinary observations.

Your study specifies an exact schema for every numeric field and each expected table/key sequence. Arrays may be bool, integer or real floats up to64bits. Nonfinite values are invalid by default. Trust and provenance limitations are in limitations.md.

Observation fields and output-table columns must be plain `numpy.ndarray` values.
Masked arrays (including arrays with no masked elements), other ndarray subclasses,
lists, and Python or NumPy scalars are rejected at ingestion with a recorded worker
error. Return a zero-dimensional plain ndarray for a scalar field. Accepted
observations are defensively copied without changing shape, dtype, or values;
noncontiguous and read-only arrays are supported. Rejection does not establish
equivalence under the contract.

Exit codes: CLI0 = equivalent;1 = an attributed/detected difference;2 = invalid, unsupported, incomplete or error. Invalid findings take precedence over a coexisting difference for the CLI exit code. Report generation returns0 for any valid rendered evidence, including a failure report; never infer scientific success from report generation alone.
