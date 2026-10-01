# Established work and closed research gates

Restart testing is not a new relation. A useful preview must go beyond short per-library pytest examples through controlled comparisons, schedule generation/reduction and independently inspectable evidence.

- [AMReX regression testing](https://github.com/AMReX-Codes/regression_testing) already includes restart testing
- [PyRETIS testing API](https://www.pyretis.org/dev/api/pyretis.testing.html) includes full histories, sample-count differences and numerical diagnostics
- [DMTCP](https://dmtcp.github.io/quick-start.html) checkpoints processes; [SCR](https://scr.readthedocs.io/en/latest/users/integration.html) addresses scalable checkpoint storage
- [ReFrame](https://reframe-hpc.readthedocs.io/en/stable/index.html) provides portable regression orchestration; RestartWitness's JUnit/CLI can be used inside such tools
- [Property Testing for Ocean Models. Can We Specify It?](https://arxiv.org/abs/2510.13692) discusses restart testing in ocean-model validation

[Oceananigans issue4857](https://github.com/CliMA/Oceananigans.jl/issues/4857) and [fix4892](https://github.com/CliMA/Oceananigans.jl/pull/4892) motivate attention to simulation-level averaging/scheduler/timestep state. The fix was merged in January2026. This Python preview has NOT reproduced either revision; the historical example is motivation only. A Julia adapter and affected/fixed experiment would be separately scoped.

A separately documented [OpenMM CPU RNG case](../experiments/historical/openmm_cpu_rng/README.md) now provides narrow historical release-version evidence; its exact and strict-tolerance contracts do not distinguish the fixed version, while a subsequent source-informed tolerance profile does. This does not reproduce Oceananigans or establish scientific novelty.

Gate A allows this controlled engineering preview after two real native checkpoint adapters and a useful save/restore distinction. Gate B remains closed: one bounded historical release comparison does not establish broader historical coverage, equal-budget practical superiority or independent scientific-domain acceptance. Those requirements and full negative reporting still constrain a research-positioned release. Gate C remains closed until actual independent use, accepted substantive contributions or citations exist.

Repository publication, many tests and twelve synthetic detections do not establish importance or impact. No claim that this exact combination is universally novel is made. No automatic upstream issue, PR, outreach or journal submission is included.
