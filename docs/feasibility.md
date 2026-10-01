# Native adapter feasibility, 2026-09-30

Status: Gate A passed for a controlled developer-preview build. This is not a research validation or upstream defect report.

## Frozen profiles for subsequent experiments

- Python 3.12.14, NumPy 2.4.3, Linux x86-64 (glibc 2.41), REBOUND 4.4.11, OpenMM 8.4.0.post2
- REBOUND: original two-body system, leapfrog, dt=0.001, native `save_to_file` / `Simulation` archive restoration, no callbacks. Identical `step()` segmentation. Fields: application steps_done, time, particle positions and velocities. Bitwise exact same-environment profile
- OpenMM: original one-particle harmonic force, LangevinMiddleIntegrator seed 271828, 300 K, 1/ps friction, dt=0.001 ps, Reference platform only. Native `Context.createCheckpoint` / `loadCheckpoint`. Fields: Context step count, time, positions and velocities. Bitwise exact is an empirical workflow contract, not an OpenMM guarantee
- Scalar units are explicitly declared in later study specifications; no conversion in the comparator

## Executed exploratory feasibility

Three tests passed in 3.15 seconds on the above Linux environment. Each native profile compared two independent uninterrupted baselines, dense versus final-only observation, and a checkpoint at logical step 8 restored in a separate interpreter to a total of 64 steps. Numeric dtype, shape and bytes matched. Save and restore PIDs differed. No application checkpoint source was changed; native checkpoint artifacts were used.

A separate, deliberately weaker OpenMM State-only workflow keeps the save/continue trajectory but diverges after fresh-process restore. This is a documented workflow limitation: State omits internal RNG state. It is not an OpenMM defect, nor a new discovery. It establishes a useful save/restore intervention contrast.

## Decision and boundary

The tiny native tests alone do not justify another framework. The conditional build proceeds to test whether strict three-arm adjudication, event-directed schedules, reproducible reduction, and independently rechecked offline evidence provide a reusable tool beyond them. Failure to deliver those features means retaining the examples rather than claiming a substantial separate project.

A bounded [historical OpenMM release comparison](../experiments/historical/openmm_cpu_rng/README.md) is available, with the failed exact/strict profiles and separate source-informed tolerance profile disclosed. It is not adjacent fix/parent causal isolation. Independent scientific-domain review has NOT been completed. Research-positioned release Gate B is closed. Impact Gate C is closed. Only Linux has actually been run at this stage.

## Primary sources and licensing

- [REBOUND native archive](https://rebound.hanno-rein.de/simulationarchive/)
- [REBOUND restart/synchronization requirements](https://rebound.hanno-rein.de/ipython_examples/SimulationarchiveRestart/)
- [OpenMM Context checkpoint semantics](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.Context.html)
- [OpenMM Simulation State/checkpoint distinction](https://docs.openmm.org/latest/api-python/generated/openmm.app.simulation.Simulation.html)
- [OpenMM licensing](https://docs.openmm.org/latest/userguide/library/01_introduction.html)

REBOUND is separately installed GPL-3.0-or-later software. No library binaries or source are bundled in RestartWitness. OpenMM licensing varies by component; this profile only selects Reference. No upstream issue or pull request will be sent, including REBOUND's disallowed AI-generated submissions.
