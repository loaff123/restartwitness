# Offline evidence reports

RestartWitness renders a static HTML evidence notebook and a JUnit XML result from a **bundle directory**, not from a result dictionary or cached status label.

```python
from pathlib import Path
from restartwitness.report import render_report, render_junit

bundle = Path('example-case')
Path('report.html').write_text(render_report(bundle), encoding='utf-8')
Path('junit.xml').write_text(render_junit(bundle), encoding='utf-8')
```

Store generated reports **outside** the sealed bundle. The CLI rejects output paths inside the input bundle before writing, because added files would change its verified file set. The same rule applies to replay and reduction destinations. Open `report.html` directly in a browser; no server, account or connection is needed.

## Verification before presentation

Both entry points call `read_verified_case` every time. Artifact hashes, observation cadence, checkpoint records and cached adjudication are checked, and comparison outcomes are recomputed from raw numeric arrays. A dictionary is rejected. Missing, changed or inconsistent evidence raises an exception; the renderer never replaces a verification failure with a passing report. Rendering does not import a driver or load native checkpoints.

Hashes establish internal consistency and detect modifications. They are not signatures: a malicious author can forge a new internally consistent bundle. Neither a verified bundle nor a green result certifies scientific validity, all hidden state, determinism, storage durability or a complete environment.

## Reading the HTML notebook

- The summary retains every adjudicated finding. A case with invalid controls is prominently inconclusive, even when raw direct contrasts also differ.
- U-P compares uninterrupted execution to saving and continuing. P-R compares saving and continuing to a fresh-process restore. U-R is also evaluated directly because tolerance-based agreement is not transitive.
- Each contrast exposes its own first witness, including the recorded logical step, diagnostic phase, zero-based occurrence, field, array index, exact scalar values and numeric diagnostics when applicable. IEEE bit patterns distinguish exact-mode floating values such as signed zero.
- The top-level witness is the **first witness in comparison order**: U1-U2, U1-O, U-P, P-R, U-R, with trajectory diagnostics before output diagnostics. It is not a global earliest-time claim across those comparisons. A phase mismatch or invalid observation is reported honestly rather than presented as a numeric difference.
- The small SVG plot shows ordinary observations of scalar `x`, or the first flattened component of `position`. U, P and R use both distinct colors and line patterns. Checkpoint markers retain repeated cuts. The plot contains at most 600 samples per arm; downsampling is disclosed. It is illustrative, never the basis of the verdict. Exact before-save, after-save and after-restore witnesses remain visible even if ordinary lines overlap.
- U1-U2 and U1-O controls, all arm execution states and recorded worker errors are visible. Metrics distinguish simulated work, process starts, elapsed wall time and segments whose work is unknown.
- Output tables show ordered semantic record identities and counts for all arms. Lists longer than 32 entries are explicitly abbreviated. Nothing is silently sorted or deduplicated.
- Declared units, dtype, shape, comparison mode and absolute/relative tolerances are listed for observation fields and output columns. The core performs no unit conversion.
- Evidence and source SHA-256 identities remain inspectable. Local filesystem locations are omitted from configuration/error summaries. Complete inputs and raw arrays remain in the original bundle; they are not embedded in the HTML.

The report uses responsive layouts, keyboard-visible focus, a skip link, semantic headings and table headers, and an accessible SVG title/description. It has no JavaScript, telemetry, remote assets or external fonts. Its content security policy prohibits network resource loading. User-supplied text is HTML-escaped; XML uses ElementTree escaping and sanitizes invalid XML 1.0 characters.

Free-form input and error text is displayed in bounded summaries. Recognizable absolute local paths in that text are redacted as a convenience, **not** as a general de-identification guarantee. Review evidence and reports before sharing them; scientific field names and exact witnesses are deliberately retained.

## JUnit mapping

One testcase represents the whole experiment. This prevents several successful-looking contrasts from obscuring an inconclusive or failed overall case.

| Outcome | JUnit element |
| --- | --- |
| `equivalent_under_contract`, with all required arms complete | No failure/error/skipped child |
| Save-path, restore-path, direct restart or output-contract difference | `failure` |
| Baseline/observation inconclusive, or unsupported profile | `skipped` |
| Driver/checkpoint error, timeout, invalid/incomplete evidence or contract | `error` |
| Unrecognized future classification | `error` |
| Bundle integrity verification fails | Raise an exception; no XML result |

Execution/invalidity findings take priority over skipped findings; skipped findings take priority over intervention differences. A valid save-path difference alongside malformed or nonfinite restore evidence is an `error`, with both findings retained. A worker marked `unsupported` alone remains `skipped`, including when other raw contrasts differ. All findings and all five contrasts remain in `system-out`, including each available witness and recorded error. Properties include the manifest digest, driver, schedule and primary status. XML preserves a distinction between “not established” and “different.”

## Replaying

After inspecting and trusting the local driver and checkpoint files:

```sh
restartwitness replay ./CASE --output ./replayed-case --trust-driver
```

Replace `CASE` with the bundle location. Replay executes local Python and native checkpoint code; it is not a safe operation on an untrusted downloaded bundle. Recorded source and environment identity must still match. The output destination must be new.

Reports are labelled **developer preview**. Controlled faults are original fixtures; native-package comparisons are observations under the declared same-environment workflow contract, not claims of a new upstream bug, endorsement, academic novelty or scientific certification.
