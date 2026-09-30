# Continuation contracts and comparison semantics

RestartWitness compares only declared observations and output tables. Agreement
under a contract does not certify hidden state, crash durability, arbitrary
hardware, or a library's general reproducibility. The contract is fixed before
the experiment and is included in its evidence.

## Python and JSON interfaces

```python
from restartwitness.contracts import Contract, FieldContract, RecordContract

contract = Contract(
    fields=(
        FieldContract("position", "float64", (3,), unit="m"),
        FieldContract("app_step", "int64", ()),
    ),
    outputs={
        "samples": RecordContract(
            name="samples",
            expected_keys=("step:0", "step:2", "step:4"),
            fields=(FieldContract("position", "float64", (3,), unit="m"),),
        ),
    },
)
encoded = contract.to_dict()
assert Contract.from_dict(encoded) == contract
```

`FieldContract`, `RecordContract`, and `Contract` are frozen dataclasses. Field
collections, shapes, and expected record keys must be tuples at construction.
`Contract.outputs` copies its input mapping into a read-only mapping. Output map
keys must equal the associated `RecordContract.name`.

The serialized root has exactly `schema_version`, `fields`, and `outputs`;
`schema_version` is integer `1`. A serialized field has exactly `name`, `dtype`,
`shape`, `unit`, `mode`, `atol`, and `rtol`. A serialized record contract has
exactly `name`, `expected_keys`, and `fields`. JSON arrays become tuples on
explicit deserialization. Unknown or missing properties, duplicate field names,
duplicate expected record keys, strings in numeric parameters, boolean shape
dimensions/tolerances, and nonfinite/negative tolerances are rejected with
`ValueError`. Empty observation contracts are rejected. Empty expected output
tables and key-only record contracts are allowed.

Supported observations use ordinary NumPy ndarrays containing boolean, signed
integer, unsigned integer, or real floating values of up to 64 bits. Object,
structured, complex, datetime, extended-precision, masked, and other ndarray
subclass observations are unsupported. A scalar has shape `()`; empty dimensions
are allowed. Dtype aliases are interpreted by NumPy, and actual dtype identity
must match the declared NumPy dtype, including nonnative byte order. The
comparator does not cast an input to make it fit a contract.

## Observations and their ordering

```python
snapshot = {
    "key": [4, "ordinary", 0],
    "fields": {
        "position": numpy.array([1.0, 2.0, 3.0], dtype="float64"),
        "app_step": numpy.array(4, dtype="int64"),
    },
    "units": {"position": "m", "app_step": ""},
}
```

The key is `[application_logical_step, phase, occurrence]`. Both integers are
nonnegative Python integers, never booleans. Ordinary observations use occurrence
`0`; diagnostic observations use the schedule's zero-based cut index so repeated
cuts at one step remain distinct. Valid phases are `ordinary`, `before_save`,
`after_save`, and `after_restore`. A runner's optional `harness_step` metadata is
not substituted for the application's reported step.

Trajectories are matched by position and exact key equality, in recorded order.
They are never lexicographically sorted by phase, re-aligned, interpolated, or
deduplicated. Duplicate keys, missing observations, and empty trajectories are
invalid. The first witness follows observed order, then the declared field order
for numeric differences, then C-order array indexing. All later observations are
still checked, so an earlier difference cannot conceal later invalid evidence.
The adapter's `observe` operation must return a fresh/non-mutating observation.
The comparator reads but does not modify input arrays.

## Numeric and unit rules

- Field names, shapes, dtype identity, and unit labels must exactly fit the
  contract. Unit maps must contain exactly the declared field names; an
  undeclared unit is the explicit empty string. There is no unit conversion or
  NumPy broadcasting
- Integer and boolean observations always compare exactly, including when a
  field's mode is `tolerance`; they are never rounded through floating point
- `mode="exact"` compares finite floating-point bit patterns. Positive and
  negative zero differ. Witnesses include canonical hexadecimal bit patterns,
  independent of the array's byte order
- `mode="tolerance"` uses the symmetric rule
  `abs(a-b) <= atol + rtol * max(abs(a), abs(b))`. Tolerance parameters in exact
  mode have no effect. Extended-precision intermediates reduce overflow and
  underflow, with exact-rational fallback for overflowing intermediate results
- NaN and either infinity are invalid, including when both sides contain the
  same nonfinite value. Diagnostics encode these as strings rather than illegal
  JSON numbers

The absolute difference is calculated without integer overflow. The relative
diagnostic is `abs(a-b)/max(abs(a),abs(b))`, with zero for two zero magnitudes.
An absolute difference beyond the finite JSON float range is `null` with
`absolute_difference_overflow: true`; the normalized relative value remains
available. Bit differences can have zero absolute difference, as with signed zero.

## Append-only tables

```python
batch = {
    "keys": ["step:0", "step:2", "step:4"],
    "fields": {"position": numpy.zeros((3, 3), dtype="float64")},
    "units": {"position": "m"},
}
```

Every key is a nonempty semantic string supplied by the adapter, identifying the
scientific event. Synthetic consecutive row numbers can hide lost or duplicated
events and should not replace native identities. `expected_keys` declares the
complete ordered sequence before running. Duplicates, missing keys, unexpected
keys, and reordering are invalid even when both compared arms contain the same
bad keys. Multiple identity problems are retained together.

A record field's declared shape describes **one record**. For a table with `n`
keys, its array must have shape `(n,) + field.shape`. Thus scalar columns have
shape `(n,)`, and the example's 3-vector column has shape `(3, 3)`. A numeric
table witness includes the semantic `record_key` and its full array index.

## Comparator results

The public functions are `compare_snapshots(left, right, contract)`,
`compare_trajectories(left, right, contract)`,
`compare_records(left, right, record_contract)`, and
`adjudicate_arms(arms, contract)`. The first three return dictionaries with:

- `status`: `equivalent`, `difference`, or `invalid`
- `first_witness`: the first diagnostic, or `null` on agreement
- `diagnostics`: all structural problems and one first numeric witness per
  mismatching field per observation, including its `mismatch_count`

Numeric witnesses include the field, first index, both values, field contract,
absolute/relative differences, and observation key or output record key. A
trajectory adds `observation_index`. `invalid` takes status precedence over
numeric differences, while preserving the earliest observed witness. Comparison
with an unvalidated object in place of a contract returns invalid.

## Five-arm adjudication

Each required arm is a mapping with `status`, `observations`, and `outputs`:

- `U1`, `U2`: independent uninterrupted baselines
- `O`: uninterrupted observation-neutrality control with a final ordinary
  observation and no intermediate observations
- `P`: save and continue in the same process
- `R`: save and restore in fresh processes

Successful workers declare `status="complete"`. Their label alone never
establishes success; all observations and output contracts are compared. The
result retains `comparisons` for `U1-U2`, `U1-O`, `U-P`, `P-R`, and `U-R`. Each
contrast includes its trajectory and output checks. `U1-O` compares the **last
ordinary** U1 observation to O's final ordinary observation, plus output tables;
it does not compare sparse O observations to U1's full trajectory or choose a
later save/restore diagnostic snapshot.

Baseline or observation-control disagreement/invalidity produces
`baseline_inconclusive` or `observation_inconclusive`, suppressing intervention
attribution while preserving every direct contrast. With valid controls,
differences yield `save_path_difference` and/or `restore_path_difference`.
Output failures also yield `output_contract_violation`. All applicable findings
are retained; these labels localize an intervention and do not diagnose hidden
state or prove a root cause.

The U–R contrast is mandatory because tolerance agreement is nontransitive. For
example, U=0, P=0.75, and R=1.5 under absolute tolerance 1 pass U–P and P–R but fail
U–R. That case yields `restart_difference` without inventing save/restore
attribution. Only fully valid agreeing evidence yields
`equivalent_under_contract`.

Missing arms yield `incomplete`; malformed direct-path observations yield
`invalid`. Worker `driver_error`, `checkpoint_error`, `timeout`, `unsupported`,
and `evidence_integrity_error` statuses remain failures. No missing, empty,
skipped, unrun, or failed arm is equivalent. The first finding is the primary
`status`; inspect the complete `findings`, `comparisons`, and `diagnostics` for
combined outcomes.

## Rounded diagnostics

Numeric witness values and displayed absolute/relative differences are ordinary JSON floats and may be rounded. Near a tolerance boundary, production comparison uses an exact rational fallback, including lost subnormal addends. The independent reader uses exact rational arithmetic for every tolerance check. Do not re-grade a boundary witness from rounded display text alone; use the raw arrays and declared contract.
