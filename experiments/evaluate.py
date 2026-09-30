#!/usr/bin/env python3
"""Frozen controlled evaluation; groups have separate explicit ten-minute budgets.

Every completed case is independently re-adjudicated. No online tuning, skipped
result deletion, or inference of real-world scientific defect prevalence.
"""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from restartwitness.runner import run_case  # noqa: E402
from restartwitness.evidence import read_json, write_json, digest  # noqa: E402
from restartwitness.compare import compare_snapshots  # noqa: E402
from restartwitness.contracts import Contract  # noqa: E402
from tests.independent_check import check_case  # noqa: E402


def load_protocol(path):
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text().split()[0]
    if digest(path) != expected:
        raise ValueError("frozen protocol hash mismatch")
    return read_json(path)


def build_jobs(protocol, group):
    rows = []

    def add(case, cuts, label, expected=None):
        rows.append(
            {
                "id": label,
                "fixture": case["id"],
                "study": copy.deepcopy(case["study"]),
                "schedule": list(cuts),
                "expected_finding": expected,
                "group": group,
            }
        )

    if group == "acceptance":
        for case in protocol["clean"]:
            for i, cuts in enumerate(protocol["schedule_recipe"]):
                add(
                    case,
                    cuts,
                    f"clean-{case['id']}-{i:02}",
                    "equivalent_under_contract",
                )
        for case in protocol["faults"]:
            add(
                case,
                case["applicability"][0],
                f"fault-{case['id']}",
                case["expected_finding"],
            )
        for case in protocol["controls"]:
            add(
                case,
                case["schedule"],
                f"control-{case['id']}",
                case["expected_finding"],
            )
        for case in protocol["native"]:
            add(
                case, case["schedule"], f"native-{case['id']}", case["expected_finding"]
            )
    elif group in protocol["evaluation"]["equal_budget_methods"]:
        for case in protocol["clean"] + protocol["faults"]:
            for i, cuts in enumerate(
                protocol["evaluation"]["equal_budget_methods"][group]
            ):
                add(case, cuts, f"{group}-{case['id']}-{i:02}")
    elif group == "conventional":
        for case in protocol["clean"] + protocol["faults"]:
            add(
                case,
                protocol["evaluation"]["conventional"]["schedule"],
                f"conventional-{case['id']}",
            )
            rows[-1]["study"]["observations"] = [protocol["steps"]]
    else:
        raise ValueError("unknown frozen group")
    return rows


def evaluate(protocol_path, group, out):
    p = load_protocol(protocol_path)
    jobs = build_jobs(p, group)
    out = Path(out)
    if out.exists():
        raise ValueError("evaluation destination must be new")
    # A recorded commit associates source and frozen protocol before outcomes.
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise ValueError("commit source changes before confirmatory evaluation")
    out.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + p["evaluation"]["study_budget_seconds"]
    rows = []
    for job in jobs:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            rows.append(
                {
                    **{k: v for k, v in job.items() if k != "study"},
                    "status": "not_run",
                    "reason": "group_budget_exhausted",
                }
            )
            continue
        study = copy.deepcopy(job["study"])
        study["study_budget"] = min(study["study_budget"], remaining)
        row = {k: v for k, v in job.items() if k != "study"}
        try:
            result = run_case(study, job["schedule"], out / job["id"])
            independent = check_case(out / job["id"])
            row.update(
                status=result["adjudication"]["status"],
                findings=result["adjudication"]["findings"],
                first_witness=result["adjudication"]["first_witness"],
                metrics=result["metrics"],
                manifest_sha256=result["manifest_sha256"],
                independent="verified",
                contrasts={
                    k: v["status"] for k, v in independent["comparisons"].items()
                },
            )
            if group == "conventional":

                def final(arm):
                    return next(
                        s
                        for s in reversed(result["arms"][arm]["observations"])
                        if s["key"][1] == "ordinary"
                    )

                control_ok = all(
                    result["adjudication"]["comparisons"][name]["status"]
                    == "equivalent"
                    for name in ("U1-U2", "U1-O")
                )
                cmp = compare_snapshots(
                    final("U1"), final("R"), Contract.from_dict(study["contract"])
                )
                row["conventional_final_only"] = {
                    "status": cmp["status"] if control_ok else "inconclusive",
                    "first_witness": cmp["first_witness"],
                }
            if job["expected_finding"] is not None:
                row["expected_met"] = (
                    job["expected_finding"] in result["adjudication"]["findings"]
                )
        except Exception as exc:
            row.update(
                status="evaluation_error",
                error=f"{type(exc).__name__}: {exc}",
                independent="not_verified",
            )
        rows.append(row)
        print(
            json.dumps(
                {
                    "case": job["id"],
                    "status": row["status"],
                    "completed": len(rows),
                    "planned": len(jobs),
                }
            ),
            flush=True,
        )
        write_json(out / "progress.json", {"rows": rows})
    summary = {
        "schema_version": 1,
        "protocol_sha256": digest(protocol_path),
        "source_commit": commit,
        "source_dirty": False,
        "group": group,
        "budget_seconds": p["evaluation"]["study_budget_seconds"],
        "budget_scope": "this frozen group",
        "wall_seconds": time.monotonic() - started,
        "planned": len(jobs),
        "rows": rows,
        "limitations": p["limits"],
        "gates": {
            "engineering": "evaluate per acceptance row",
            "research": "closed: no historical reproduction or domain review",
            "impact": "closed: no adoption evidence",
        },
    }
    write_json(out / "results.json", summary)
    # Individual cases are already sealed; the group index binds their manifest hashes.
    write_json(
        out / "index-integrity.json",
        {
            "results_sha256": digest(out / "results.json"),
            "protocol_sha256": digest(protocol_path),
            "case_manifests": {
                r["id"]: r["manifest_sha256"] for r in rows if "manifest_sha256" in r
            },
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocols/v1.json")
    parser.add_argument(
        "--group",
        required=True,
        choices=["acceptance", "fixed", "random", "boundary", "conventional"],
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    r = evaluate(args.protocol, args.group, args.out)
    failed = any(
        row["status"] in ("not_run", "evaluation_error")
        or row.get("expected_met") is False
        for row in r["rows"]
    )
    print(
        json.dumps(
            {
                "group": args.group,
                "planned": r["planned"],
                "wall_seconds": r["wall_seconds"],
                "acceptance_errors": failed,
            }
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
