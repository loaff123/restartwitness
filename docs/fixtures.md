# Original controlled fixtures

All formulas and seeds were declared in protocols/v1.json before confirmatory evaluation. Development tests verify reachability; later raw evaluation is retained separately. These are deliberate mutations, never new upstream scientific defects.

## Models

- Integrator: velocity Verlet with cached acceleration `a=-x+0.1*sin(phase*0.2)`; phase increments; dt starts0.01 and scales by0.8 after logical steps8 and32
- Stochastic: `v=0.8*v+0.05*z`, `x+=0.01*v`; paired Box-Muller Gaussian values driven by Python Random seed271828, including a cached second variate
- Events: `v=0.05*sin(pre_step*0.2)` and `x+=0.01*v`; callback adds0.2 at8; external reset sets x=1 at32; means emit/reset at8,32,64. Keys contain actual persistent output ordinal and event step

The event callback omission has a transient trajectory difference that disappears after the reset, while output history preserves the earlier difference. This exposes final-only checks' limited view.

## Twelve mutations

1. Missing RNG state restores a deterministic seed0 generator instead of serialized state
2. Seed reset restarts the original seeded generator
3. Missing cached stochastic variate discards the paired Box-Muller second draw; odd cuts are applicable
4. Integrator cache omission restores acceleration cache to0
5. Evolving timestep omission restores the initial dt
6. Forcing phase omission resets its counter
7. Average sum omission resets accumulated sum
8. Average count omission resets count
9. Output ID omission resets native event ordinal
10. Duplicate boundary record appends the already-emitted real event row after restore at8/32/64
11. Skipped boundary record suppresses the next real event after restore at7/31/63
12. Restored callback omission drops the event8 kick

Applicability schedules are explicit in the protocol; no fault is expected to diverge for every cut. In particular, an empty schedule does not exercise restoration and late cuts cannot recover a missed past callback.

## Controls

Clean variants, harmless hidden-state change, nondeterministic baseline, mutating observer, explicit unsupported profile, timeout, malformed output identity and save-side effect are separate labels. Additional unit tests exercise failed/empty checkpoints, nonfinite arrays and interrupted workers. Two native checkpoint profiles are useful integrations; their State-only misuse contrast is a documented limitation, not a scientific fault discovery.

## Fairness

Fixed/random/boundary groups each use four schedules and exactly5×64 steps per completed case (U1/U2/O/P/R), totalling1,280 steps per fixture. Random seed is frozen. The boundary subset `[7],[8],[9],[32]` is explicit; full13-schedule clean coverage is measured separately. The conventional midpoint/final-only experiment ignores output history for its conventional score; its full raw controls remain in evidence. It has a smaller work budget and is not used for a speed advantage claim.

Each group independently receives the declared600-second budget. Overall experiment duration spans several groups and may exceed ten minutes. Timeouts/errors/not-run cases remain rows; completed-step counts never substitute for unknown work in terminated workers.
