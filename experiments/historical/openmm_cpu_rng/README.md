# Historical OpenMM CPU checkpoint case

An opt-in validation of the resolved [OpenMM issue #4716](https://github.com/openmm/openmm/issues/4716), fixed by [PR #4740](https://github.com/openmm/openmm/pull/4740) in November 2024. Native checkpoints on affected OpenMM 8.2.0 omit CPU RNG state; upstream explains and fixes this. This experiment uses an unchanged RestartWitness engine and genuine native checkpoint bytes.

**Result and qualification:** a separately documented, source-informed absolute 1e-5 profile distinguishes 8.2.0 from 8.3.0 on this bounded CPU fixture. The initial exact and 1e-12 profiles also reject fixed CPU continuation, so those contracts did **not** cleanly distinguish the versions. All negatives and the amendment remain visible. These are local experiment records, not an external preregistration. Seeds 43 and 44 were subsequently tested after the amendment was written.

## Scope and source pins

| Role | Source revision |
|---|---|
| Original evaluated engine | `7ba891733412dd6ad03010c0bde5c9122f9c032b` |
| Affected official PyPI OpenMM 8.2.0 | `53770948682c40bd460b39830d4e0f0fd3a4b868` |
| Fixed official PyPI OpenMM 8.3.0 | `1ce5d91d9dedfdc273066fafa1a618bf05c25b85` |
| Upstream fix | `f67ae730a13197bda8a75b9a99556f354bff5e1f` |
| Upstream fix parent | `9fe1bae6efa18a55994522a8aac4f24338a2894e` |

The adapter checks package version and embedded Git revision. Each restore uses the same package/platform as its checkpoint. These release wheels contain other changes: this is not a runtime comparison of the adjacent fix and parent, and does not isolate a single patch causally.

The original small Python fixture takes physical parameters from upstream [`testLangevin()`](https://github.com/openmm/openmm/blob/f67ae730a13197bda8a75b9a99556f354bff5e1f/tests/TestCheckpoints.h). Differences are explicit: a deterministic partial lattice replaces SFMT random positions, velocities and Langevin seed are explicit, `LangevinMiddleIntegrator` matches the issue, and restoration crosses interpreter boundaries. Ten particles, a 3-nm periodic box, 300 K and 0.001-ps steps keep it inexpensive. Details are in [protocol.json](protocol.json).

`createCheckpoint()`/`loadCheckpoint()` are the only persistence path. `getState()` is observation only. No defect is injected. CPU has one thread and deterministic forces; Reference is a negative control. This does not change the project's default native versions, examples, engine or comparator.

## Results

Every case executes U1/U2 independent no-save baselines, O final-only observation, P save-and-continue, and R fresh-process restoration. All 36 cases have exact U1-U2, U1-O and U-P agreement at their declared observations. The original full evaluation completed 19,800 integration steps in 224 processes; summed case wall time was 97.32 seconds, excluding installation and verification.

At cut 100, the visible state initially restores exactly. Affected CPU diverges at step 101. Maximum P-R differences in the original observations:

| Profile | Position, nm | Velocity, nm/ps |
|---|---:|---:|
| 8.2.0 CPU, seed 42 | 0.00491594 | 0.697121 |
| 8.3.0 CPU, seed 42 | 1.54e-12 | 2.38e-10 |
| 8.2.0 CPU, seed 43 | 0.00255135 | 0.569734 |
| 8.3.0 CPU, seed 43 | 2.14e-12 | 3.56e-10 |
| 8.2.0 CPU, seed 44 | 0.00540565 | 0.985144 |
| 8.3.0 CPU, seed 44 | 0 | 0 |

Both Reference versions agree exactly throughout the tested schedules/seeds. Final-only restoration at step 110 passes even on affected CPU: a visible round trip alone misses this latent continuation defect.

Upstream's existing test uses TOL=1e-5 scaled by max(1, expected value or vector norm). The amendment uses a distinct absolute componentwise 1e-5 bound in each native unit, no looser than those assertions. It is source-informed, not the identical upstream comparison or a domain-general scientific error budget. The cause of the tiny fixed-version differences is unestablished here. Integer step counts remain exact.

[summary.json](summary.json) preserves all 36 outcomes, including exact and strict-tolerance negatives. [amendments.json](amendments.json) records the initial 8.1.1/NumPy compatibility failure and the subsequent tolerance decision. Independent scientific-domain review, practical superiority and adoption remain open; no new project, current defect, novelty or academic impact is claimed.

## Public evidence and independent checking

The four files in [evidence/](evidence/) are **newly identified observation projections**, not redacted sealed runner bundles. They retain the exact binary64 values as hexadecimal strings at selected keys: initial, step 99, step 100 and its diagnostic phases, step 101, and final step 110. Only numeric observations and public scientific configuration are exported. No worker requests, process tokens, native binaries, checkpoint bytes, absolute executor paths, upstream API dumps or issue/comment text are published.

Original sealed evidence was not changed. Each projection records its source manifest hash and has a fresh SHA-256 in the excerpt manifest. Hashes are integrity checks, not third-party authentication. The small independent standard-library checker validates included field shapes, finite canonical hex values, keys, controls and three comparison profiles using exact rational tolerance arithmetic. It **cannot verify omitted observations or original process provenance**.

From a source checkout, without OpenMM or NumPy:

```sh
python experiments/historical/openmm_cpu_rng/verify_excerpt.py
```

The original full 36-case evaluation also passed the project's independent full-bundle checker. That broader result is reported in the summary, not purportedly proved by the smaller public projection.

## Reproduce fresh full cases

Use Python 3.12 and two isolated environments. From the repository root:

```sh
python3.12 -m venv .venv-historical-affected
.venv-historical-affected/bin/pip install -r experiments/historical/openmm_cpu_rng/requirements-affected.txt
.venv-historical-affected/bin/pip install --no-deps -e .
export OPENMM_CPU_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv-historical-affected/bin/python -m experiments.historical.openmm_cpu_rng.test_adapter
.venv-historical-affected/bin/python -m experiments.historical.openmm_cpu_rng.run --out historical-affected
```

Repeat in a separate environment with `requirements-fixed.txt` and a distinct output directory. The default opt-in run produces the two representative CPU cases: seed-42 exact and seed-43 amended. `--suite primary` runs 14 cases per version; `--suite amended` runs four. Output directories must be new. Workers are bounded to 30 seconds, cases to 120, and each suite to 600 seconds. Simulation uses one CPU thread; no GPU or external dataset is required.

Verify each new full case without loading its native checkpoint:

```sh
python -I tests/independent_check.py historical-affected/affected-cpu-42-exact-100
```

Generate a separately identified public projection without changing its source:

```sh
python -m experiments.historical.openmm_cpu_rng.export_excerpt historical-affected/affected-cpu-42-exact-100 --out new-excerpts --name affected-cpu-100
```

Export the matching fixed exact and affected/fixed seed-43 cases to make the four-file set. Native simulation values may vary on another build or hardware; no universal bitwise claim is made. Only run trusted adapters and checkpoint files.

## Independent upstream oracle

`upstream-TestCheckpoints.h` is byte-for-byte upstream at the fix commit, Git blob `cc5d493d43c9105cb184d5aebb323fb10ea6d577`, with its MIT notice. The wrapper runs only `testLangevin()` and selects CPU/Reference. The test's SFMT positions, default seed, same-context restore, 100+10+10 steps and scaled assertions stay unchanged. It failed affected CPU and passed fixed CPU and both Reference versions, independently corroborating the version contrast.

For the tested Linux wheel, from the repository root:

```sh
prefix="$PWD/.venv-historical-affected/lib/python3.12/site-packages/OpenMM.libs"
case_dir=experiments/historical/openmm_cpu_rng
mkdir -p historical-oracle-build
g++ -std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0 -O2 "$case_dir/upstream_oracle.cpp" -I"$prefix/include" -L"$prefix/lib" -Wl,-rpath,"$prefix/lib" -lOpenMM -o historical-oracle-build/affected
OPENMM_PLUGIN_DIR="$prefix/lib/plugins" TEST_PLATFORM=CPU OPENMM_CPU_THREADS=1 historical-oracle-build/affected
```

The affected CPU test is expected to exit 1. Repeat against the fixed environment with a distinct executable name, expecting exit 0. Both versions expect exit 0 with `TEST_PLATFORM=Reference`. The wrapper is source only; no executable is distributed. ABI flags and wheel paths may differ on another distribution.

## License

Original scripts, documentation and generated data are BSD-3-Clause; see [LICENSE.md](LICENSE.md). The copied test retains its exact MIT header. [licenses/SOURCE.json](licenses/SOURCE.json) pins the clean-UTF-8 current upstream license overview. The historical overview itself contained a replacement character, so it was not silently repaired or presented as an exact historical copy.
