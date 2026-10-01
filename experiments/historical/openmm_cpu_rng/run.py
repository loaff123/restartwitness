"""Explicitly opt-in execution; uses the existing unmodified engine."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

from restartwitness.contracts import Contract, FieldContract
from restartwitness.runner import read_verified_case, run_case

COMMITS = {
    "8.2.0": "53770948682c40bd460b39830d4e0f0fd3a4b868",
    "8.3.0": "1ce5d91d9dedfdc273066fafa1a618bf05c25b85",
}


def make_study(version, platform, seed, comparison):
    if version not in COMMITS or platform not in ("CPU", "Reference"):
        raise ValueError("unsupported historical version/platform")
    if seed not in (42, 43, 44) or comparison not in ("exact", "source-informed"):
        raise ValueError("unsupported historical seed/comparison")
    fields = []
    for name, dtype, shape, unit in [
        ("step", "<i8", (), "step"),
        ("time", "<f8", (), "ps"),
        ("position", "<f8", (10, 3), "nm"),
        ("velocity", "<f8", (10, 3), "nm/ps"),
        ("box", "<f8", (3, 3), "nm"),
    ]:
        tolerant = comparison == "source-informed" and dtype == "<f8"
        fields.append(
            FieldContract(
                name,
                dtype,
                shape,
                unit,
                "tolerance" if tolerant else "exact",
                1e-5 if tolerant else 0.0,
                0.0,
            )
        )
    return {
        "driver": "experiments.historical.openmm_cpu_rng.historical_openmm_native",
        "config": {
            "package_version": version,
            "git_revision": COMMITS[version],
            "platform": platform,
            "seed": seed,
        },
        "total_steps": 110,
        "observations": list(range(111)),
        "contract": Contract(tuple(fields), {}).to_dict(),
        "worker_timeout": 30.0,
        "study_budget": 120.0,
        "termination": "normal",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["representative", "primary", "amended"],
        default="representative",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    version = importlib.metadata.version("openmm")
    if version not in COMMITS:
        parser.error("use the isolated, pinned 8.2.0 or 8.3.0 environment")
    args.out.mkdir(parents=True, exist_ok=False)
    profile = "affected" if version == "8.2.0" else "fixed"
    if args.suite == "primary":
        cases = [
            (p, 42, "exact", cuts)
            for p in ("CPU", "Reference")
            for cuts in ([], [99], [100], [101], [99, 100, 101], [100, 100], [110])
        ]
    elif args.suite == "amended":
        cases = [
            (p, seed, "source-informed", [100])
            for p in ("CPU", "Reference")
            for seed in (43, 44)
        ]
    else:
        cases = [("CPU", 42, "exact", [100]), ("CPU", 43, "source-informed", [100])]
    rows = []
    started = time.monotonic()
    for platform, seed, comparison, cuts in cases:
        remaining = 600.0 - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("600-second suite budget exhausted")
        cut_label = "-".join(map(str, cuts)) or "none"
        name = f"{profile}-{platform.lower()}-{seed}-{comparison}-{cut_label}"
        study = make_study(version, platform, seed, comparison)
        study["study_budget"] = min(120.0, remaining)
        result = run_case(study, cuts, args.out / name)
        verified = read_verified_case(args.out / name)
        row = {
            "case": name,
            "status": verified["adjudication"]["status"],
            "metrics": result["metrics"],
        }
        rows.append(row)
        (args.out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
