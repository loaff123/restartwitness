"""One exec-style worker segment. Checkpoint completion precedes termination."""

from __future__ import annotations
import os
from pathlib import Path
import sys
import traceback
import uuid
import numpy as np
from .driver import load_driver, UnsupportedProfile
from .evidence import read_json, write_data, write_json, digest, IntegrityError


def _require_array_fields(fields, location):
    for name, value in fields.items():
        if type(value) is not np.ndarray:
            raise TypeError(
                f"{location} field {name!r} must be a plain numpy.ndarray; "
                f"got {type(value).__name__}"
            )


def run_segment(request):
    root = Path(request["segment_dir"])
    root.mkdir(parents=True, exist_ok=False)
    output = Path(request["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "complete",
        "observations": [],
        "outputs": {},
        "pid": os.getpid(),
        "process_token": uuid.uuid4().hex,
        "advance_calls": 0,
        "checkpoint_complete": False,
        "start_step": request["start_step"],
        "next_cut": request["next_cut"],
    }
    phase = "create"
    try:
        d = load_driver(request["driver"])
        units = getattr(d, "UNITS", {})
        phase = "restore" if request.get("restore") else "create"
        s = (
            d.restore(request["config"], Path(request["restore"]), output)
            if request.get("restore")
            else d.create(request["config"], output)
        )
        k = request["start_step"]
        cut_index = request["next_cut"]
        cuts = request["cuts"]

        def observe(label, index=0):
            nonlocal phase
            phase = "observe"
            fields = d.observe(s)
            if not isinstance(fields, dict) or not fields:
                raise ValueError("empty observation")
            # Copy only supported arrays: coercion would discard masks/subclass semantics.
            _require_array_fields(fields, "observation")
            fields = {
                name: np.array(value, copy=True) for name, value in fields.items()
            }
            result["observations"].append(
                {
                    "key": [k, label, index],
                    "fields": fields,
                    "units": {name: units.get(name, "") for name in fields},
                }
            )

        if request.get("restore"):
            observe("after_restore", cut_index - 1)
        elif request["arm"] != "O" and k in request["observations"]:
            observe("ordinary")
        if request["arm"] == "O" and k == request["total_steps"]:
            observe("ordinary")
        while True:
            if request["arm"] != "O" and cut_index < len(cuts) and cuts[cut_index] == k:
                observe("before_save", cut_index)
                checkpoint = root / f"checkpoint-{cut_index:03}"
                if request["arm"] in ("P", "R"):
                    phase = "save"
                    checkpoint.mkdir()
                    d.save(s, checkpoint)
                    artifacts = []
                    for p in sorted(checkpoint.rglob("*")):
                        if p.is_symlink():
                            raise IntegrityError("checkpoint symlink not allowed")
                        if p.is_file():
                            artifacts.append(
                                {
                                    "path": p.relative_to(checkpoint).as_posix(),
                                    "sha256": digest(p),
                                    "bytes": p.stat().st_size,
                                }
                            )
                    if not artifacts or not any(x["bytes"] for x in artifacts):
                        raise IntegrityError("checkpoint returned without artifacts")
                    result.setdefault("checkpoints", []).append(
                        {"cut": cut_index, "step": k, "artifacts": artifacts}
                    )
                observe("after_save", cut_index)
                cut_index += 1
                if request["arm"] == "R":
                    result.update(
                        status="checkpoint",
                        checkpoint_complete=True,
                        checkpoint_path=str(checkpoint),
                        next_cut=cut_index,
                        end_step=k,
                    )
                    write_data(root / "result.json", result)
                    write_json(
                        root / "checkpoint-complete.json",
                        {
                            "process_token": result["process_token"],
                            "checkpoint_complete": True,
                            "cut": cut_index - 1,
                        },
                    )
                    if request["termination"] == "exit_after_save":
                        os._exit(75)
                    return result
                observe("after_restore", cut_index - 1)
                continue
            if k == request["total_steps"]:
                break
            phase = "advance"
            d.advance_one(s)
            result["advance_calls"] += 1
            k += 1
            if (request["arm"] == "O" and k == request["total_steps"]) or (
                request["arm"] != "O" and k in request["observations"]
            ):
                observe("ordinary")
        phase = "outputs"
        outputs = d.collect_outputs(output)
        if isinstance(outputs, dict):
            for name, table in outputs.items():
                if isinstance(table, dict) and isinstance(table.get("fields"), dict):
                    _require_array_fields(table["fields"], f"output table {name!r}")
        result["outputs"] = outputs
        result.update(end_step=k, next_cut=cut_index)
    except Exception as exc:
        result.update(
            status="unsupported"
            if isinstance(exc, UnsupportedProfile)
            else (
                "checkpoint_error" if phase in ("save", "restore") else "driver_error"
            ),
            error={
                "phase": phase,
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            },
            end_step=locals().get("k", request["start_step"]),
        )
        traceback.print_exc()
    write_data(root / "result.json", result)
    return result


def main():
    request = read_json(sys.argv[1])
    result = run_segment(request)
    return 0 if result["status"] in ("complete", "checkpoint") else 1


if __name__ == "__main__":
    sys.exit(main())
