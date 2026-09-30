# Reproduce and inspect the preview

Start with the HTML reports; they are offline-capable and execute no Python. The small per-case archives contain the original raw arrays and native checkpoint artifacts. Evidence-only `verify` and `report` do not import a driver or restore a checkpoint.

The complete download includes:
- Current source, wheel and source distribution
- `source-history.bundle`, preserving all recorded local source commits
- Exact evaluation source export and original wheel under `reproduce/`
- All267 raw case bundles in `all-evidence.tar.xz`
- The static interactive demo and frozen result summaries
- SHA-256 checksums and the original licensing/dependency notes

The raw archive contains over560,000 small files. Allow approximately3GB of free disk before extracting all of it; archive size alone understates filesystem allocation. Archive extraction is outside the bounded evidence reader. Extract trusted archives into a new directory with appropriate resource limits.

For exact experimental source:

```sh
git clone source-history.bundle experimental-source
cd experimental-source
git checkout 71ca6c071e14fd2309270752544a8fb6e0d0b071
```

The recorded environment was Linux x86-64, Python3.12.14 and NumPy2.4.3. Native profiles additionally used REBOUND4.4.11 or OpenMM8.4.0.post2; native libraries are separately installed and not bundled. `requirements-lock.txt` records the complete local development environment. Replay checks identities and refuses drift rather than assuming different hardware/software is interchangeable.

For an installed matching source/environment:

```sh
restartwitness verify path/to/case
python -I tests/independent_check.py path/to/case
restartwitness replay path/to/case --out new-replay --trust-driver
```

If your environment differs, verification and HTML reporting remain useful. To explore behavior there, explicitly run `case/study.json` with the recorded cut list as a new experiment; its new identity is recorded. That is not an exact replay of the published environment.

The reduction source is local commit `9d914aeb403e4c4ebd30b5dfa11a730b72bb80de`; its scientific inputs match the declared model while the reader includes additional safety checks. Trial records retain every candidate, repeated outcome and manifest hash.

Hashes establish internal integrity, not authenticity of a dishonest author. Python drivers and native replay code must be trusted. No MPI/GPU/crash-durability or cross-platform numerical certificate is provided.
