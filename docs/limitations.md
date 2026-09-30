# Limits and honest interpretation

This is a developer preview. No historical affected-versus-fixed scientific defect has been reproduced and no independent scientific-domain review or adoption has been established. Engineering review is not domain endorsement.

- A pass is observational equivalence under selected fields/cadence/units/tolerances, not certification of hidden state, scientific validity, reproducibility across hardware, or checkpoint durability
- Two baselines and one sparse observation control are smoke controls, not statistical proofs of determinism or observer noninterference
- Physical time and application step are returned by the adapter. Phase keys are the harness's logical counter; the observed application step remains separately compared
- Direct U/P, P/R and U/R contrasts localize interventions, not causes. The overall first witness follows comparison order; each contrast has its own earliest observed witness
- Default per-worker timeout is 60 seconds; per-case/group budget is 600 seconds; schedules cap at128 cuts; reduction caps at64 trials. Actual incomplete coverage is retained. No promised tutorial/suite runtime until measured
- Drivers are trusted executable code with the user's filesystem/network privileges. Process isolation is not a sandbox. Timeouts terminate harness-owned process groups on POSIX; Windows only guarantees terminating the worker itself. Driver-created grandchildren require responsible driver cleanup on Windows
- Checkpoint saves must finish before returning. No interrupted writes, power cuts or storage durability are tested. `exit_after_save` is an explicitly labelled abrupt process exit after acknowledgment
- Evidence size limits protect readers, not against every resource abuse by a trusted driver during computation or file writing. Use OS/container quotas for hostile or unbounded workloads
- Recorded source identity covers core Python modules, the selected driver, and built-in adapter siblings. Custom transitive imports, compiler flags, BLAS build details, system libraries, all CPU properties, and external files are not comprehensively captured. Supply a hermetic environment and extend provenance for your own research
- Source/config consistency checks detect ordinary drift. Hashes are not signatures; an author can forge a mutually consistent fabricated bundle
- Same-version OpenMM Reference bitwise replay is empirical. State-only persistence omits some RNG internal state by design; a mismatch is not an upstream bug
- Schedule reduction is deletion-only, confirms a class/field twice and checks single removals. It is not globally minimal. Unknown, timeout, nonrepeatable or exhausted trials block a one-minimal claim
- Synthetic omissions measure a bounded tool behavior; they cannot establish real-world defect prevalence, novelty, research importance, stars, citations or impact
- No upstream project endorsement, unsolicited issue, maintainer contact or publication submission is implied
