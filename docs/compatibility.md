# Compatibility and publication status

| Profile | Actually executed |
|---|---|
| Linux x86-64, Python3.12.14, NumPy2.4.3 | Full local suite, frozen studies, reduction, packaging |
| Separate clean wheel with NumPy2.5.3 | Basic clean/fault/verify/report/replay and unsupported-optional checks |
| REBOUND4.4.11 on Linux | Native fresh-process archive profile |
| OpenMM8.4.0.post2 Reference on Linux | Native seeded checkpoint profile |
| macOS / Windows, Python3.11–3.13 | CI matrix executes core tests; consult the exact commit run below |

Python3.11+ is the package requirement;3.11–3.13 are CI targets. Coverage is tied to the exact tested commit. Native coverage is explicitly Linux-only at this stage. Missing/mismatched optional packages are unsupported outcomes.

The [public repository](https://github.com/loaff123/restartwitness) includes all project source. Consult [Actions](https://github.com/loaff123/restartwitness/actions) for the exact commit's results; the complete Site/Library download also records its verified CI run and commit in release-manifest.json.

The first cross-platform run exposed two portability defects: eight-byte NumPy longdouble diagnostics were not JSON serializable on macOS/Windows, and Windows Git checkout converted the frozen protocol to CRLF. The fixes explicitly convert supported floating diagnostics and preserve LF checkout bytes using .gitattributes. Regression tests cover both behaviors. Protocol content and its frozen SHA-256 were not changed. Native scientific checkpoints remain Linux-only coverage.
