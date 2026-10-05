"""Explicit trusted-code execution and offline verified reporting."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
from .evidence import read_json, write_json, IntegrityError
from .examples import example_study
from .runner import (
    run_case,
    read_verified_case,
    replay,
    _identity,
    _require_external_output,
    validate_study,
)
from .reduce import reduce_schedule

DIFFERENCES = {
    "save_path_difference",
    "restore_path_difference",
    "restart_difference",
    "output_contract_violation",
}


def signature(adjudication):
    if any(
        adjudication["comparisons"].get(k, {}).get("status") != "equivalent"
        for k in ("U1-U2", "U1-O")
    ):
        return None
    for name, pair in [
        ("save_path_difference", "U-P"),
        ("restore_path_difference", "P-R"),
        ("restart_difference", "U-R"),
        ("output_contract_violation", "P-R"),
    ]:
        if name in adjudication["findings"]:
            witness = adjudication["comparisons"][pair].get("first_witness") or {}
            return (
                name
                + ":"
                + str(witness.get("table", ""))
                + ":"
                + str(witness.get("field", witness.get("kind", "unknown")))
            )
    return None


def _summary(result):
    return {
        "status": result["adjudication"]["status"],
        "findings": result["adjudication"]["findings"],
        "first_witness": result["adjudication"]["first_witness"],
        "schedule": result["schedule"],
        "metrics": result["metrics"],
        "manifest_sha256": result["manifest_sha256"],
    }


def _exit(result):
    if "invalid" in result["adjudication"]["findings"]:
        return 2
    s = result["adjudication"]["status"]
    return 0 if s == "equivalent_under_contract" else (1 if s in DIFFERENCES else 2)


def _output(path, text):
    p = Path(path)
    if p.exists() or p.is_symlink() or any(x.is_symlink() for x in p.parents):
        raise ValueError("output must be a new non-symlink file")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="restartwitness",
        description="Auditable trusted-local scientific restart experiments. Developer preview.",
    )
    parser.add_argument("--version", action="version", version="RestartWitness 0.1.0a1")
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("example", help="write a fixed example study JSON")
    ex.add_argument(
        "--model",
        default="integrator",
        choices=[
            "integrator",
            "stochastic",
            "events",
            "rebound",
            "openmm",
            "openmm-state-misuse",
        ],
    )
    ex.add_argument("--fault", default="clean")
    ex.add_argument("--steps", type=int, default=64)
    ex.add_argument("--out", "--output", required=True)
    run = sub.add_parser("run", help="execute a trusted local study driver")
    run.add_argument("study")
    run.add_argument(
        "--cuts",
        default=None,
        help="JSON list, including repeated cuts; default is the midpoint",
    )
    run.add_argument("--out", "--output", required=True)
    verify = sub.add_parser(
        "verify",
        help="verify and recompute adjudication from raw evidence, without running drivers",
    )
    verify.add_argument("bundle")
    rp = sub.add_parser(
        "replay",
        help="rerun a verified bundle using exactly matching local code/environment",
    )
    rp.add_argument("bundle")
    rp.add_argument("--out", "--output", required=True)
    rp.add_argument("--trust-driver", action="store_true")
    report = sub.add_parser("report", help="render verified offline HTML or JUnit")
    report.add_argument("bundle")
    report.add_argument("--out", "--output", required=True)
    report.add_argument("--format", choices=["html", "junit"], default="html")
    reduce = sub.add_parser(
        "reduce", help="delete cuts while preserving a repeated class/field witness"
    )
    reduce.add_argument("bundle")
    reduce.add_argument("--out", "--output", required=True)
    reduce.add_argument("--trust-driver", action="store_true")
    reduce.add_argument("--trials", type=int, default=64)
    reduce.add_argument("--seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "example":
            study = example_study(args.model, args.fault, args.steps)
            validate_study(study)
            _output(args.out, json.dumps(study, indent=2, allow_nan=False) + "\n")
            print(json.dumps({"study": str(args.out)}))
            return 0
        if args.command == "run":
            study = validate_study(read_json(args.study))
            cuts = (
                json.loads(args.cuts)
                if args.cuts is not None
                else [study["total_steps"] // 2]
            )
            result = run_case(study, cuts, args.out)
        elif args.command == "verify":
            result = read_verified_case(args.bundle)
        elif args.command == "replay":
            result = replay(args.bundle, args.out, args.trust_driver)
        elif args.command == "report":
            from .report import render_report, render_junit

            _require_external_output(args.bundle, args.out)
            content = (
                render_report(args.bundle)
                if args.format == "html"
                else render_junit(args.bundle)
            )
            _output(args.out, content)
            print(json.dumps({"report": str(args.out), "format": args.format}))
            return 0
        else:
            if not args.trust_driver:
                raise ValueError(
                    "reduction executes local code; --trust-driver is required"
                )
            original = read_verified_case(args.bundle)
            if _identity(original["study"]["driver"]) != original["identity"]:
                raise IntegrityError("source/environment identity changed")
            target = signature(original["adjudication"])
            if target is None:
                raise ValueError(
                    "no attributable repeatable class/field witness to reduce"
                )
            if not 0 <= args.trials <= 64 or not 0 < args.seconds <= 600:
                raise ValueError("reduction limits: at most64 trials and600 seconds")
            _require_external_output(args.bundle, args.out)
            out = Path(args.out)
            if out.exists() or any(x.is_symlink() for x in (out, *out.parents)):
                raise ValueError("reduction output must be new")
            out.mkdir(parents=True)
            deadline = time.monotonic() + args.seconds
            count = 0

            def evaluator(cuts):
                nonlocal count
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"status": "timeout", "signature": None}
                trial = dict(original["study"])
                trial["study_budget"] = min(trial["study_budget"], remaining)
                count += 1
                case = run_case(trial, cuts, out / f"trial-{count:03}")
                sign = signature(case["adjudication"])
                return {
                    "status": "difference"
                    if sign
                    else (
                        "equivalent"
                        if case["adjudication"]["status"] == "equivalent_under_contract"
                        else case["adjudication"]["status"]
                    ),
                    "signature": sign,
                    "bundle": f"trial-{count:03}",
                    "manifest_sha256": case["manifest_sha256"],
                }

            reduction = reduce_schedule(
                original["schedule"], evaluator, target, args.trials
            )
            reduction["original_manifest_sha256"] = original["manifest_sha256"]
            write_json(out / "reduction.json", reduction)
            print(json.dumps(reduction, allow_nan=False))
            return 0 if reduction["target_reproduced"] else 2
        print(json.dumps(_summary(result), allow_nan=False))
        return _exit(result)
    except (ValueError, OSError, ImportError) as exc:
        print(
            json.dumps(
                {
                    "status": "evidence_integrity_error"
                    if isinstance(exc, IntegrityError)
                    else "invalid_request",
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
