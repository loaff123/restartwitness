"""Bounded five-arm coordinator with true fresh interpreter restoration."""

from __future__ import annotations
import importlib.metadata
import importlib.util
import os
import json
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import numpy as np
from .evidence import (
    read_data,
    write_data,
    write_json,
    read_json,
    seal_bundle,
    digest,
    verify_bundle,
    IntegrityError,
    safe_path,
)
from .contracts import Contract
from .compare import adjudicate_arms

DEFAULT_WORKER_TIMEOUT = 60.0
DEFAULT_STUDY_BUDGET = 600.0


def validate_study(study):
    allowed = {
        "driver",
        "config",
        "total_steps",
        "observations",
        "contract",
        "worker_timeout",
        "study_budget",
        "termination",
    }
    if not isinstance(study, dict) or set(study) - allowed:
        raise ValueError("unknown study fields")
    s = dict(study)
    for name in ("driver", "config", "total_steps", "observations", "contract"):
        if name not in s:
            raise ValueError(f"missing study {name}")
    if not isinstance(s["driver"], str) or any(
        not p.isidentifier() for p in s["driver"].split(".")
    ):
        raise ValueError("invalid driver module")
    if not isinstance(s["config"], dict):
        raise ValueError("config must be an object")
    n = s["total_steps"]
    if type(n) is not int or n < 0 or n > 1000000:
        raise ValueError("total_steps must be bounded nonnegative integer")
    obs = s["observations"]
    if (
        not isinstance(obs, list)
        or not obs
        or any(type(k) is not int or k < 0 or k > n for k in obs)
        or obs != sorted(set(obs))
        or n not in obs
    ):
        raise ValueError(
            "observations must be sorted, unique, in bounds and include final step"
        )
    Contract.from_dict(s["contract"])
    for name, default in [
        ("worker_timeout", DEFAULT_WORKER_TIMEOUT),
        ("study_budget", DEFAULT_STUDY_BUDGET),
    ]:
        s.setdefault(name, default)
        if (
            isinstance(s[name], bool)
            or not isinstance(s[name], (float, int))
            or not 0 < s[name] <= 3600
        ):
            raise ValueError(f"invalid {name}")
    s.setdefault("termination", "normal")
    if s["termination"] not in ("normal", "exit_after_save"):
        raise ValueError("unsupported termination mode")
    return s


def validate_schedule(schedule, n):
    if (
        not isinstance(schedule, (list, tuple))
        or len(schedule) > 128
        or any(type(k) is not int or k < 0 or k > n for k in schedule)
        or list(schedule) != sorted(schedule)
    ):
        raise ValueError(
            "cuts must be a sorted bounded list; repeated cuts are permitted"
        )
    return list(schedule)


def _new_root(path):
    path = Path(path).absolute()
    if path.exists() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("output must be a new non-symlink directory")
    path.mkdir(parents=True)
    return path


def _identity(driver):
    spec = importlib.util.find_spec(driver)
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        raise ValueError("driver source not found")
    sources = {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    versions = {}
    packages = ["numpy"]
    if driver.endswith("rebound_native"):
        packages.append("rebound")
    if driver.endswith(("openmm_native", "openmm_state_misuse")):
        packages.append("openmm")
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    adapters = (
        {p.name: digest(p) for p in sorted(Path(spec.origin).parent.glob("*.py"))}
        if driver.startswith("restartwitness.adapters.")
        else {}
    )
    return {
        "driver": driver,
        "driver_sha256": digest(spec.origin),
        "core_sources": sources,
        "adapter_sources": adapters,
        "python": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": versions,
    }


def _stop(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(process.pid, signal.SIGKILL)
    else:
        process.kill()
    process.wait()


def _launch(request, request_path, timeout):
    write_json(request_path, request)
    log = request_path.with_suffix(".log")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(x) for x in sys.path if x)
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    started = time.monotonic()
    reason = None
    with log.open("wb") as out:
        p = subprocess.Popen(
            [sys.executable, "-m", "restartwitness.worker", str(request_path)],
            stdout=out,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=(os.name == "posix"),
        )
        while p.poll() is None:
            if time.monotonic() - started > timeout:
                reason = "timeout"
                _stop(p)
                break
            if log.stat().st_size > 4 * 1024 * 1024:
                reason = "driver_error"
                _stop(p)
                break
            time.sleep(0.01)
    elapsed = time.monotonic() - started
    result_path = Path(request["segment_dir"]) / "result.json"
    if reason:
        return {
            "status": reason,
            "error": {
                "phase": "worker",
                "message": "worker timeout"
                if reason == "timeout"
                else "log limit exceeded",
            },
            "observations": [],
            "outputs": {},
            "advance_calls": None,
            "process_token": None,
            "exit_code": p.returncode,
            "wall_seconds": elapsed,
        }
    try:
        result = read_data(result_path)
    except (OSError, ValueError) as exc:
        return {
            "status": "driver_error",
            "error": {"phase": "worker", "message": str(exc)[:2000]},
            "observations": [],
            "outputs": {},
            "advance_calls": None,
            "process_token": None,
            "exit_code": p.returncode,
            "wall_seconds": elapsed,
        }
    result.update(exit_code=p.returncode, wall_seconds=elapsed)
    if result["status"] == "checkpoint":
        try:
            ack = read_json(Path(request["segment_dir"]) / "checkpoint-complete.json")
            expected = 75 if request["termination"] == "exit_after_save" else 0
            if (
                p.returncode != expected
                or ack.get("process_token") != result["process_token"]
                or ack.get("checkpoint_complete") is not True
            ):
                raise ValueError("checkpoint completion acknowledgment mismatch")
        except (OSError, ValueError) as exc:
            result.update(
                status="checkpoint_error",
                error={"phase": "acknowledgment", "message": str(exc)},
            )
    elif result["status"] == "complete" and p.returncode != 0:
        result.update(status="driver_error")
    return result


def run_case(study, schedule, output_dir):
    """Execute trusted driver code. Never pass an untrusted downloaded study."""
    s = validate_study(study)
    cuts = validate_schedule(schedule, s["total_steps"])
    identity = _identity(s["driver"])
    root = _new_root(output_dir)
    start = time.monotonic()
    deadline = start + s["study_budget"]
    arms = {}
    write_json(root / "study.json", s)
    write_json(root / "schedule.json", cuts)
    for name in ("U1", "U2", "O", "P", "R"):
        arm = {"status": "complete", "observations": [], "outputs": {}, "segments": []}
        arm_root = root / name
        arm_root.mkdir()
        next_cut = 0
        k = 0
        restore = None
        index = 0
        while True:
            if _identity(s["driver"]) != identity:
                arm.update(
                    status="evidence_integrity_error",
                    error={
                        "phase": "source_identity",
                        "message": "source or environment changed during case",
                    },
                )
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                arm.update(
                    status="timeout",
                    error={"phase": "study_budget", "message": "case budget exhausted"},
                )
                break
            request = {
                **s,
                "arm": name,
                "cuts": cuts,
                "next_cut": next_cut,
                "start_step": k,
                "restore": restore,
                "run_dir": str(arm_root / "output"),
                "segment_dir": str(arm_root / f"segment-{index:03}"),
            }
            result = _launch(
                request,
                arm_root / f"request-{index:03}.json",
                min(s["worker_timeout"], remaining),
            )
            arm["segments"].append(
                {
                    key: value
                    for key, value in result.items()
                    if key not in ("observations", "outputs")
                }
            )
            arm["observations"].extend(result["observations"])
            if result["status"] == "checkpoint":
                restore = result["checkpoint_path"]
                next_cut = result["next_cut"]
                k = result["end_step"]
                index += 1
                continue
            arm["status"] = result["status"]
            arm["outputs"] = result["outputs"]
            if "error" in result:
                arm["error"] = result["error"]
            break
        if _identity(s["driver"]) != identity:
            arm.update(
                status="evidence_integrity_error",
                error={
                    "phase": "source_identity",
                    "message": "source or environment changed during case",
                },
            )
        arms[name] = arm
        write_data(root / f"{name}.json", arm)
    final_identity = _identity(s["driver"])
    if final_identity != identity:
        if arms["R"]["status"] != "evidence_integrity_error":
            raise IntegrityError("source changed while sealing case")
    contract = Contract.from_dict(s["contract"])
    adjudication = adjudicate_arms(arms, contract)
    result = {
        "study": s,
        "schedule": cuts,
        "identity": identity,
        "arms": arms,
        "adjudication": adjudication,
        "metrics": {
            "wall_seconds": time.monotonic() - start,
            "process_starts": sum(len(a["segments"]) for a in arms.values()),
            "simulated_steps": sum(
                x["advance_calls"] or 0 for a in arms.values() for x in a["segments"]
            ),
            "unknown_work_segments": sum(
                x["advance_calls"] is None for a in arms.values() for x in a["segments"]
            ),
        },
    }
    write_json(root / "adjudication.json", adjudication)
    write_json(root / "metrics.json", result["metrics"])
    result["manifest_sha256"] = seal_bundle(
        root,
        {
            "kind": "restartwitness-case",
            "identity": identity,
            "source_status": "source_hashes_recorded",
            "termination": s["termination"],
            "study_sha256": digest(root / "study.json"),
            "schedule_sha256": digest(root / "schedule.json"),
        },
    )
    return result


def _same_raw(left, right):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _same_raw(left[k], right[k]) for k in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same_raw(a, b) for a, b in zip(left, right)
        )
    return left == right


def _read_verified_case(path, check_cached=True):
    root = Path(path)
    integrity = verify_bundle(root)
    if integrity["metadata"].get("kind") != "restartwitness-case":
        raise IntegrityError("not a RestartWitness case")
    s = validate_study(read_json(root / "study.json"))
    schedule = validate_schedule(read_json(root / "schedule.json"), s["total_steps"])
    metadata = integrity["metadata"]
    if metadata.get("study_sha256") != digest(root / "study.json") or metadata.get(
        "schedule_sha256"
    ) != digest(root / "schedule.json"):
        raise IntegrityError("frozen study/schedule identity mismatch")
    if (
        metadata.get("identity", {}).get("driver") != s["driver"]
        or metadata.get("termination") != s["termination"]
    ):
        raise IntegrityError("study metadata identity mismatch")
    arms = {
        name: read_data(root / f"{name}.json") for name in ("U1", "U2", "O", "P", "R")
    }
    expected = []
    at = {}
    for index, cut in enumerate(schedule):
        at.setdefault(cut, []).append(index)
    ordinary = set(s["observations"])
    for step in sorted(ordinary | set(schedule)):
        if step in ordinary:
            expected.append([step, "ordinary", 0])
        for index in at.get(step, []):
            expected.extend(
                [
                    [step, phase, index]
                    for phase in ("before_save", "after_save", "after_restore")
                ]
            )
    for name, arm in arms.items():
        segments = arm.get("segments", [])
        if not isinstance(segments, list):
            raise IntegrityError("invalid segment records")
        if arm.get("status") == "complete":
            raw_observations = []
            raw_outputs = {}
            for segment_index, segment in enumerate(segments):
                raw = read_data(
                    root / name / f"segment-{segment_index:03}" / "result.json"
                )
                if raw.get("status") != segment.get("status") or raw.get(
                    "process_token"
                ) != segment.get("process_token"):
                    raise IntegrityError("segment raw terminal identity mismatch")
                if raw.get("advance_calls") != segment.get("advance_calls"):
                    raise IntegrityError("segment completed-work count mismatch")
                raw_observations.extend(raw.get("observations", []))
                raw_outputs = raw.get("outputs", {})
            if not _same_raw(
                raw_observations, arm.get("observations")
            ) or not _same_raw(raw_outputs, arm.get("outputs")):
                raise IntegrityError(
                    "aggregate observations/outputs contradict segment raw data"
                )
            if len(segments) != (len(schedule) + 1 if name == "R" else 1):
                raise IntegrityError("segment count mismatch")
            tokens = [x.get("process_token") for x in segments]
            if any(not isinstance(t, str) or not t for t in tokens) or len(
                set(tokens)
            ) != len(tokens):
                raise IntegrityError("process identity missing/reused")
            acknowledgments = [
                c for segment in segments for c in segment.get("checkpoints", [])
            ]
            expected_cuts = list(enumerate(schedule)) if name in ("P", "R") else []
            if [
                (c.get("cut"), c.get("step")) for c in acknowledgments
            ] != expected_cuts:
                raise IntegrityError("checkpoint acknowledgment sequence mismatch")
            for i, segment in enumerate(segments):
                expected_status = (
                    "checkpoint" if name == "R" and i < len(schedule) else "complete"
                )
                if segment.get("status") != expected_status:
                    raise IntegrityError("segment terminal status mismatch")
                if (
                    expected_status == "checkpoint"
                    and segment.get("checkpoint_complete") is not True
                ):
                    raise IntegrityError("checkpoint completion missing")
                expected_exit = (
                    75
                    if expected_status == "checkpoint"
                    and s["termination"] == "exit_after_save"
                    else 0
                )
                if segment.get("exit_code") != expected_exit:
                    raise IntegrityError("segment termination mismatch")
            if sum(x.get("advance_calls", -1) for x in segments) != s["total_steps"]:
                raise IntegrityError("completed step count mismatch")
        for segment_index, segment in enumerate(segments):
            request = read_json(root / name / f"request-{segment_index:03}.json")
            declared = {key: request.get(key) for key in s}
            if json.dumps(declared, sort_keys=True, allow_nan=False) != json.dumps(
                s, sort_keys=True, allow_nan=False
            ):
                raise IntegrityError("worker request differs from frozen study")
            if request.get("arm") != name or request.get("cuts") != schedule:
                raise IntegrityError("worker arm/schedule request mismatch")
            if segment_index > (len(schedule) if name == "R" else 0):
                raise IntegrityError("worker request segment boundary mismatch")
            expected_start = (
                schedule[segment_index - 1] if name == "R" and segment_index else 0
            )
            expected_cut = segment_index if name == "R" else 0
            if (
                type(request.get("start_step")) is not int
                or request["start_step"] != expected_start
                or type(request.get("next_cut")) is not int
                or request["next_cut"] != expected_cut
            ):
                raise IntegrityError("worker request segment boundary mismatch")
            for checkpoint in segment.get("checkpoints", []):
                base = (
                    root
                    / name
                    / f"segment-{segment_index:03}"
                    / f"checkpoint-{checkpoint['cut']:03}"
                )
                if not checkpoint.get("artifacts"):
                    raise IntegrityError("checkpoint artifacts missing")
                for artifact in checkpoint["artifacts"]:
                    f = safe_path(base, artifact["path"])
                    if (
                        not f.is_file()
                        or f.stat().st_size != artifact["bytes"]
                        or digest(f) != artifact["sha256"]
                    ):
                        raise IntegrityError(
                            "checkpoint acknowledgment artifact mismatch"
                        )
        if arm.get("status") == "complete":
            keys = [x.get("key") for x in arm.get("observations", [])]
            if keys != (
                [[s["total_steps"], "ordinary", 0]] if name == "O" else expected
            ):
                raise IntegrityError(
                    f"{name} observations do not match study cadence/cuts"
                )
        else:
            keys = [x.get("key") for x in arm.get("observations", [])]
            declared = [[s["total_steps"], "ordinary", 0]] if name == "O" else expected
            if keys != declared[: len(keys)]:
                raise IntegrityError("partial observations are not a declared prefix")
    metrics = read_json(root / "metrics.json")
    segments = [segment for arm in arms.values() for segment in arm["segments"]]
    expected_metrics = {
        "process_starts": len(segments),
        "simulated_steps": sum(segment["advance_calls"] or 0 for segment in segments),
        "unknown_work_segments": sum(
            segment["advance_calls"] is None for segment in segments
        ),
    }
    if any(
        type(metrics.get(k)) is not int or metrics[k] != v
        for k, v in expected_metrics.items()
    ):
        raise IntegrityError("reported work counts contradict segment ledger")
    result = adjudicate_arms(arms, Contract.from_dict(s["contract"]))
    if check_cached and result != read_json(root / "adjudication.json"):
        raise IntegrityError("cached adjudication does not match raw observations")
    return {
        "study": s,
        "schedule": schedule,
        "identity": integrity["metadata"]["identity"],
        "arms": arms,
        "adjudication": result,
        "metrics": metrics,
        "manifest_sha256": integrity["manifest_sha256"],
    }


def read_verified_case(path, check_cached=True):
    """Read verified evidence; normalize malformed schemas to integrity errors."""
    try:
        return _read_verified_case(path, check_cached)
    except IntegrityError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise IntegrityError(f"malformed case evidence: {exc}") from exc


def _require_external_output(bundle, output):
    """Derived output must not change the sealed input's file inventory."""
    try:
        source = Path(bundle).resolve()
        destination = Path(output).resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("input or output path could not be resolved") from exc
    if destination.is_relative_to(source):
        raise ValueError("output must be outside the input evidence bundle")


def replay(path, output_dir, trust_driver=False):
    if not trust_driver:
        raise ValueError(
            "replay executes local code; explicitly trust the driver first"
        )
    old = read_verified_case(path)
    if _identity(old["study"]["driver"]) != old["identity"]:
        raise IntegrityError("source/environment identity changed; replay refused")
    _require_external_output(path, output_dir)
    return run_case(old["study"], old["schedule"], output_dir)
