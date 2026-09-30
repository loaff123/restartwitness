# RestartWitness

**Where does a scientific restart stop agreeing?**

Run the same local experiment without saving, saving and continuing, and saving then restoring in a fresh Python interpreter. Inspect all three direct comparisons, distinguish inconclusive controls from attributable differences, and reduce a failing checkpoint schedule without shortening the experiment.

**Developer preview.** Controlled synthetic faults and two native checkpoint integrations are engineering evidence. They are not new scientific defects, proof of novelty, or academic impact. Historical affected/fixed reproduction and independent scientific-domain review remain open gates.

## Quick start

Python3.11+ and NumPy2.x;3.11–3.13 are CI targets. Actual platform coverage is in [compatibility](docs/compatibility.md). From a source checkout:

```sh
python -m pip install .
restartwitness example --model integrator --fault integrator_cache --steps 16 --out study.json
restartwitness run study.json --cuts '[7,8,9]' --out evidence
# Exit 1 means a detected difference; 0 means equivalent under this contract.
restartwitness verify evidence
restartwitness report evidence --out report.html
restartwitness replay evidence --out replayed --trust-driver
restartwitness reduce evidence --out reduced --trust-driver --trials 32
```

Open `report.html` locally. It needs no server, network, account, or telemetry. The report exposes the contract, first observed witness in each contrast, checkpoint markers, controls, records, and work counts.

Only run drivers you trust. Importing a driver executes arbitrary local Python. Fresh processes are an experimental control, **not a security sandbox**. `verify` and `report` do not import drivers or restore native checkpoint files. Replay requires explicit trust and matching recorded source/environment identities.

## What the experiment means

| Arm | Action |
|---|---|
| U1, U2 | Independent no-save baselines, identical one-step integration calls |
| O | No-save control with only a final observation |
| P | Save at each cut, continue in the same process |
| R | Save at each cut, acknowledge completion, terminate, restore in a fresh interpreter |

All ordinary observation points and save-phase diagnostic calls align across U/P/R. The O control detects observable effects of intermediate inspection. Baseline disagreement or observer interference blocks restart attribution. Neither control proves probabilistic determinism or universal observer purity.

Compare U–P, P–R **and U–R directly**. Tolerance agreement is nontransitive. A difference identifies an associated intervention, not the hidden cause. Final-step and repeated-at-the-same-step cuts are valid and retain before-save, after-save, and after-restore inspections.

- Exact float comparisons preserve signed-zero distinctions; integer/bool fields stay exact
- Tolerances use `abs(a-b) <= atol + rtol*max(abs(a),abs(b))`
- Dtype, shape, unit labels and ordered native record keys are part of the contract
- NaN/Inf are invalid evidence, not equivalent observations
- Missing, duplicate, reordered or unexpected output keys are never silently repaired
- Only declared fields and observation points are covered

## Native examples

Install optional dependencies separately: `python -m pip install '.[native]'`.

- REBOUND 4.4.11: original two-body leapfrog system using native Simulationarchive
- OpenMM 8.4.0.post2: original seeded Langevin harmonic particle using native checkpoints on Reference only
- OpenMM State-only example: intentionally weaker documented workflow, **not an OpenMM bug**

```sh
restartwitness example --model openmm --out openmm-study.json
restartwitness run openmm-study.json --cuts '[8,32]' --out openmm-evidence
```

OpenMM exact replay is an empirical same-environment workflow contract; its API does not promise universal bitwise replay. Unavailable or mismatched pinned native packages are `unsupported`. See [feasibility](docs/feasibility.md), [dependencies](docs/dependencies.md), and [limitations](docs/limitations.md).

## Python API

```python
from restartwitness.examples import example_study
from restartwitness.runner import run_case, read_verified_case
from restartwitness.schedules import boundary_schedules

study = example_study("stochastic", fault="missing_cached_stochastic")
result = run_case(study, [7, 8, 9], "new-evidence-directory")
verified = read_verified_case("new-evidence-directory")
print(verified["adjudication"]["findings"])
print(boundary_schedules(64, [8, 32]))
```

A trusted adapter supplies `create`, `advance_one`, `observe`, `save`, `restore`, and `collect_outputs`, plus explicit units. Study JSON contains immutable fields, cadence, output-key expectations, inputs, timeout and termination mode. See [adapter guide](docs/adapters.md), [contracts](docs/contracts.md), and [reduction](docs/reduction.md).

## Evidence you can challenge

Every case retains bounded JSON and numeric NPY arrays, native checkpoint bytes, worker requests, process tokens, source hashes, checkpoint acknowledgments, logs and metrics. Pickle/object arrays and unsafe paths are rejected. Offline verification rechecks hashes, study/request identities, cadence, checkpoint artifacts, and cached labels against raw observations.

A separate reader imports none of the production comparator, contracts, runner, or evidence code:

```sh
python -I tests/independent_check.py evidence
```

The independent checker uses its own bounded NPY parser and exact rational tolerance arithmetic. Hashes detect changes; they do not authenticate an adversarial author who forges every part of a bundle. Evidence-only reports never execute an uploaded driver. Read the [independent engineering review](docs/independent-review.md).

The optional full evidence archive contains over560,000 files; allow about3GB of disk to extract it. Normal installation and the offline demo do not extract that archive. Small individual cases are available from the preview.

## Frozen experiments

`protocols/v1.json` freezes three original models, twelve intentional omissions, seven controls, native profiles, seeds, schedules, contracts and metrics. Its SHA-256 is recorded beside it. Fixture construction tests are development checks; the confirmatory results are separate, run from a committed source snapshot.

```sh
python -m pip install '.[test,native]'
pytest
python experiments/evaluate.py --group acceptance --out results/acceptance
python experiments/evaluate.py --group fixed --out results/fixed
python experiments/evaluate.py --group random --out results/random
python experiments/evaluate.py --group boundary --out results/boundary
python experiments/evaluate.py --group conventional --out results/conventional
```

Each group has an explicit 600-second budget and reports every planned case, including unrun ones. Fixed/random/boundary each allocate 1,280 simulated steps per fixture including all controls. Process starts and wall time are recorded separately. The conventional midpoint/final-only row has a smaller workload and is excluded from equal-budget claims. No confidence intervals or population defect-rate estimates are attached to this hand-selected deterministic corpus. See [fixtures](docs/fixtures.md), [measured results](docs/experiment-results.md), and [reproduction](docs/reproduction.md).

The [interactive preview](https://restartwitness.lyczz.chatgpt.site) includes verified reports and a complete source/evidence download. The [public GitHub repository](https://github.com/loaff123/restartwitness) contains the source. See the [exact-commit CI results](https://github.com/loaff123/restartwitness/actions) for platform verification. The complete download records its verified commit and CI run in release-manifest.json.

## Scope and related work

Restart testing is established practice. AMReX and PyRETIS already support restart comparisons; PyRETIS includes whole-history and record-count diagnostics. DMTCP/SCR provide checkpoint mechanisms and ReFrame provides regression orchestration. RestartWitness is a focused integration of controls, event-cut schedules, reduction and replayable evidence, not a claim to have invented checkpointing or trajectory comparison. See [research context](docs/research-context.md).

No GPU/MPI, crash-at-mid-write, power-loss, cross-hardware certification, stochastic distribution testing, automatic repair, hosted code execution or unsolicited upstream reporting. REBOUND disallows AI-generated issues/PRs; none are submitted here.

BSD-3-Clause for original code. Optional scientific libraries retain their own licenses and are not bundled.
