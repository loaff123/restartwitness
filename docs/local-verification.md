# Local engineering verification

2026-09-30 developer-preview source, before retained confirmatory evaluation.

- Coordinated full suite after all hardening: **363 passed in146.24 seconds** on Linux/Python3.12.14, NumPy2.4.3, REBOUND4.4.11 and OpenMM8.4.0.post2
- Ruff0.14.0 check and format checks pass; wheel and source distribution build
- Separate clean-wheel acceptance outside the source checkout, with NumPy2.5.3: clean integrator equivalent (exit0); integrator-cache mutation detected (exit1); verify, HTML report, trusted replay and standalone independent reader succeed; missing optional OpenMM explicitly unsupported (exit2)
- Twelve applicable controlled faults and seven controls pass their expected classifications; this is bounded fixture behavior, not real-world defect evidence
- Important review regressions cover source drift, lost subnormal tolerance-boundary terms, missing/altered checkpoint acknowledgments, study/request disagreement, inconsistent duplicate raw observations and stale work counts

The363-test runtime exceeds the aspirational one-minute smoke target; no sub-minute full-suite claim is made. The repository's cross-platform workflow is configured, but actual remote results must be verified at the published commit before claiming Linux/macOS/Windows CI success. Local independent engineering review is not scientific-domain review.

Recorded wall time is a coordinator measurement. Counts are crosschecked against segment records; no process attestation or independent timing oracle is claimed. Retained confirmatory group results will supply their exact source commit and protocol hash.

A concurrent packaging run caused four non-fault unit workers to hit their original3-second test limit. The same focused tests passed unchanged after load eased. Non-timeout unit fixtures now allow15 seconds; explicit timeout tests and the production/frozen60-second worker limits are unchanged. The final363-test suite above passed after that correction.
