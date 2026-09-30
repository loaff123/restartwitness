# Safety boundary

This developer tool runs trusted local Python drivers. It is not a sandbox. Never execute/replay an untrusted study, module or native checkpoint received from a stranger. The explicit `--trust-driver` flag is a human decision, not a security mechanism.

`verify` and `report` read bounded JSON and numeric NPY files without importing drivers. They reject unsafe paths, object/pickle arrays, hash mismatches and internally inconsistent claims. Hashes are integrity checks, not proof of an author's honesty. Decompression of externally obtained archives is outside the verifier; inspect/extract safely with size limits first.

Do not include credentials, private inputs, environment dumps, production outputs or sensitive exception messages in evidence you plan to publish. Public examples contain only original synthetic data and original model definitions. Keep real scientific data local unless authorized to share it.

Run under appropriate filesystem/network/CPU/memory quotas when resource containment matters. Report security concerns privately through the repository's GitHub security advisory flow if enabled; do not upload sensitive evidence to a public issue.
