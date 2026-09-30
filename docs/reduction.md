# Schedule generation and witness reduction

RestartWitness changes **only the restart-cut sequence** during reduction.
The evaluator must keep the original horizon, seed, inputs, termination mode,
ordinary observations, fields, tolerances, and clean-worker policy unchanged.
The reducer receives none of that configuration and cannot shorten the horizon
to make a later divergence disappear. It passes an immutable tuple of logical
step numbers to the evaluator on every call.

## Frozen schedule recipe

`boundary_schedules(total_steps, boundaries, limit=128)` returns ordered tuples:

1. Increasing unique singles at 0, 1, N−1, N, and each declared event's valid
   predecessor, event step, and successor
2. One distributed control at 1, floor(N/2), and N−1
3. One predecessor/event/successor control around the earliest declared event
4. One repeated cut at that earliest event

Beginning/end controls are clipped for N < 2. Ordinary multiple-cut controls
are sorted, deduplicated, and omitted if fewer than two distinct cuts remain.
The explicit repeated-cut control keeps both positions. Duplicate schedule
tuples are omitted. Event order and duplicate event declarations do not affect
the recipe. Empty event declarations still get the distributed control.

For N=64 and events 8 and 32, the complete result is:

```text
(0,), (1,), (7,), (8,), (9,), (31,), (32,), (33,), (63,), (64,),
(1, 32, 63), (7, 8, 9), (8, 8)
```

N=0 is supported: a single `(0,)` is generated, plus `(0, 0)` when step zero
is a declared event. All input steps, limits, and counts must be nonnegative
Python integers; booleans and fractional values are rejected. An event outside
0…N is an error, even when the limit would truncate it away.

The limit selects an exact prefix, including a possible empty prefix. This
plain-list API does not attach coverage metadata: **the caller must disclose
truncation** and must not describe a truncated prefix as full boundary coverage.
For the canonical fixture the untruncated count is 13. More generally, a caller
can regenerate with a sufficient limit (at most `3 * distinct_events + 7`) to
obtain the actual count. The recipe does not infer solver-internal events or
create a neighborhood multi-cut control for every declared event.

## Baselines

`baseline_schedules(method, total_steps, count, seed=271828)` emits unique
single-cut schedules, capped at N+1:

- `fixed`: for at least two schedules, use integer-floor equally spaced
  positions from 0 through N; one requested schedule uses floor(N/2)
- `random`: sample without replacement using a local `random.Random(seed)`,
  preserving sample order and leaving the application's global RNG untouched

Zero schedules return an empty list. A recorded seed and runtime identify the
random recipe. Equal schedule counts do **not** establish equal computational
budgets; the study must measure simulated steps and process starts separately.

## Evaluator protocol

```python
def evaluator(cuts: tuple[int, ...]) -> dict:
    # Run a new clean full-horizon study using the fixed experiment context.
    return {"status": "difference", "signature": "restore_path_difference:position"}

result = reduce_schedule((1, 8, 9), evaluator,
                         "restore_path_difference:position", budget=64)
```

The external adjudicator defines the signature from failure class and observed
field. The reducer treats that string as an opaque exact identity. The only
conclusive evaluator statuses are:

- `difference` with a nonempty string signature: preserves the target only
  when the signature matches exactly
- `equivalent` with `signature=None`: conclusive absence of the target

A stable different `difference` signature is conclusive **non-preservation**
of the target; it never replaces the witness. Driver errors, timeouts,
unsupported runs, malformed results, and every other status are unknown.
Exceptions are recorded with type and message as `evaluator_error`. Keyboard
interrupts and process-exit exceptions are not swallowed. The evaluator must
enforce its own worker and wall-clock timeouts; this synchronous reducer cannot
interrupt a blocked evaluator.

Every tested schedule gets two calls, including the input and each proposed
deletion. A candidate is accepted only after both calls return the requested
signature. Conclusive non-preservation likewise requires two agreeing outcomes:
both equivalent, or both the same different signature. Mixed outcomes are
nonrepeatable and unknown. Each evaluator call must launch a fresh experiment;
the reducer cannot verify that an arbitrary user callback actually does so.

JSON-compatible extra evaluator fields are retained in full. Returned outcome
objects are snapshotted, so reusing or mutating an evaluator's dictionary cannot
rewrite earlier trial evidence. Non-JSON results are recorded as errors rather
than silently serialized with arbitrary object representations.

## Search and claims

After reproducing the original twice, deterministic ddmin-style chunk deletion
tries complements from left to right. It increases the number of chunks when
none can be removed. There is no monotonicity assumption. After that search,
each individual **position** is removed and tested, including both positions
when cuts repeat. Any accepted removal restarts this final sweep. The retained
candidate is confirmed again after the final sweep, except an originally empty
schedule needs only its initial confirmations.

`one_minimal` means that the retained candidate reproduced and every individual
remaining cut removal was conclusively non-preserving. It is a local deletion
claim, **never** a globally shortest schedule claim. An empty reproduced
schedule is vacuously one-minimal and indicates the selected failure does not
require any restart cuts.

The implementation is deliberately conservative: any unknown outcome anywhere
in the search, not only during final checks, prevents `one_minimal`. Such runs
return `smallest_observed`, as do interrupted verification and budget exhaustion.
The last twice-confirmed candidate is retained; an unrelated error cannot
replace it. If the original does not reproduce, it is returned unchanged with
`target_reproduced=false`, rather than being presented as a confirmed witness.

## Budgets and result evidence

The budget counts **evaluator calls**, not schedules. Original confirmation,
search, final single removals, and final confirmation all count. The default is
64 calls. An odd remaining budget can record one trial without permitting
acceptance. A required check that cannot finish sets `budget_exhausted=true`;
finishing all required checks exactly at the budget is not an incomplete run.

The JSON-serializable result contains:

- `original`, `candidate`: lists of cuts, preserving duplicate positions
- `status`: `one_minimal` or `smallest_observed`
- `failure_signature`: the unchanged requested class/field identity
- `target_reproduced`: whether the retained candidate has a pair of matching
  confirmations and has not subsequently failed its final confirmation
- `evaluations`, `budget`, `budget_exhausted`, `unknown_observed`
- `trials`: every call in order, with evaluation number, schedule, phase,
  repeat number within the pair, and the complete JSON outcome

Trial phases are `initial`, `ddmin`, `single_removal`, and
`final_confirmation`. A lack of budget for a final confirmation does not erase
an earlier recorded reproduction; it still prevents a minimality claim.
If final confirmation is actually performed and fails, the earlier smallest
observed candidate remains in the report but `target_reproduced=false` exposes
the failed recheck. Raw trials provide the distinction.
