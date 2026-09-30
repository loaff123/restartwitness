"""Original controlled fixtures. These omissions are intentional, not upstream bugs."""

from __future__ import annotations
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np

UNITS = {"step": "step", "time": "s", "x": "m", "v": "m/s"}
MODELS = {"integrator", "stochastic", "events"}
FAULTS = {
    "missing_rng",
    "seed_reset",
    "missing_cached_stochastic",
    "integrator_cache",
    "evolving_timestep",
    "forcing_phase",
    "average_sum",
    "average_count",
    "output_id",
    "duplicate_record",
    "skipped_record",
    "restored_callback",
    "save_side_effect",
    "nondeterministic",
    "mutating_observer",
    "harmless_hidden",
    "malformed_outputs",
    "timeout",
    "failed_save",
    "failed_restore",
    "empty_checkpoint",
}


def create(config, run_dir):
    from restartwitness.driver import UnsupportedProfile

    if config.get("fault") == "unsupported":
        raise UnsupportedProfile("controlled unsupported profile")
    model = config.get("model", "integrator")
    fault = config.get("fault", "clean")
    seed = config.get("seed", 271828)
    if model not in MODELS or fault not in FAULTS | {"clean"} or type(seed) is not int:
        raise ValueError("unsupported fixture configuration")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    x = (
        1.0
        if fault != "nondeterministic"
        else int.from_bytes(os.urandom(4), "little") / 2**32
    )
    return {
        "config": dict(config),
        "model": model,
        "fault": fault,
        "seed": seed,
        "root": str(root),
        "step": 0,
        "time": 0.0,
        "x": x,
        "v": 0.0,
        "dt": 0.01,
        "cache": -x,
        "phase": 0,
        "sum": 0.0,
        "count": 0,
        "output_id": 0,
        "rng": random.Random(seed),
        "normal_cache": None,
        "callback": True,
        "skip_next": False,
        "hidden": 0,
    }


def _normal(s):
    if s["normal_cache"] is not None:
        z = s["normal_cache"]
        s["normal_cache"] = None
        return z
    u = max(s["rng"].random(), 2**-53)
    v = s["rng"].random()
    r = math.sqrt(-2 * math.log(u))
    theta = 2 * math.pi * v
    s["normal_cache"] = r * math.sin(theta)
    return r * math.cos(theta)


def _emit(s):
    row = {
        "key": f"average:{s['output_id']}@{s['step']}",
        "step": s["step"],
        "mean": s["sum"] / max(1, s["count"]),
    }
    if s["fault"] == "malformed_outputs":
        row["key"] = None
    if not s["skip_next"]:
        with (Path(s["root"]) / "events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, allow_nan=False) + "\n")
    s["skip_next"] = False
    s["output_id"] += 1
    s["sum"] = 0.0
    s["count"] = 0


def advance_one(s):
    if s["fault"] == "timeout":
        time.sleep(2.0)
    dt = s["dt"]
    if s["model"] == "integrator":
        s["v"] += 0.5 * dt * s["cache"]
        s["x"] += dt * s["v"]
        s["cache"] = -s["x"] + 0.1 * math.sin(s["phase"] * 0.2)
        s["v"] += 0.5 * dt * s["cache"]
        s["phase"] += 1
    elif s["model"] == "stochastic":
        s["v"] = 0.8 * s["v"] + 0.05 * _normal(s)
        s["x"] += dt * s["v"]
    else:
        s["v"] = 0.05 * math.sin(s["step"] * 0.2)
        s["x"] += s["v"] * dt
    s["time"] += dt
    s["step"] += 1
    if s["model"] == "integrator" and s["step"] in (8, 32):
        s["dt"] *= 0.8
    if s["model"] == "events":
        if s["callback"] and s["step"] == 8:
            s["x"] += 0.2
        if s["step"] == 32:
            s["x"] = 1.0
        s["sum"] += s["x"]
        s["count"] += 1
        if s["step"] in (8, 32, 64):
            _emit(s)


def observe(s):
    if s["fault"] == "mutating_observer":
        s["x"] += 0.001
    return {
        "step": np.array(s["step"], dtype=np.int64),
        "time": np.array(s["time"]),
        "x": np.array(s["x"]),
        "v": np.array(s["v"]),
    }


def save(s, checkpoint_dir):
    if s["fault"] == "failed_save":
        raise RuntimeError("controlled save failure")
    if s["fault"] == "empty_checkpoint":
        return
    data = {k: v for k, v in s.items() if k not in ("rng", "root")}
    data["rng_state"] = s["rng"].getstate()
    (Path(checkpoint_dir) / "state.json").write_text(
        json.dumps(data, allow_nan=False), encoding="utf-8"
    )
    if s["fault"] == "save_side_effect":
        s["x"] += 0.125


def _tuples(v):
    return tuple(_tuples(x) for x in v) if isinstance(v, list) else v


def restore(config, checkpoint_dir, run_dir):
    if config.get("fault") == "failed_restore":
        raise RuntimeError("controlled restore failure")
    s = json.loads((Path(checkpoint_dir) / "state.json").read_text(encoding="utf-8"))
    s["root"] = str(run_dir)
    rng_state = s.pop("rng_state")
    s["rng"] = random.Random(s["seed"])
    s["rng"].setstate(_tuples(rng_state))
    f = s["fault"]
    if f == "missing_rng":
        s["rng"] = random.Random(0)
    elif f == "seed_reset":
        s["rng"] = random.Random(s["seed"])
    elif f == "missing_cached_stochastic":
        s["normal_cache"] = None
    elif f == "integrator_cache":
        s["cache"] = 0.0
    elif f == "evolving_timestep":
        s["dt"] = 0.01
    elif f == "forcing_phase":
        s["phase"] = 0
    elif f == "average_sum":
        s["sum"] = 0.0
    elif f == "average_count":
        s["count"] = 0
    elif f == "output_id":
        s["output_id"] = 0
    elif f == "restored_callback":
        s["callback"] = False
    elif f == "skipped_record" and s["step"] in (7, 31, 63):
        s["skip_next"] = True
    elif f == "duplicate_record" and s["step"] in (8, 32, 64):
        output = Path(run_dir) / "events.jsonl"
        rows = output.read_text(encoding="utf-8").splitlines()
        with output.open("a", encoding="utf-8") as out:
            out.write(rows[-1] + "\n")
    elif f == "harmless_hidden":
        s["hidden"] += 1
    return s


def collect_outputs(run_dir):
    path = Path(run_dir) / "events.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]
    return {
        "events": {
            "keys": [r["key"] for r in rows],
            "fields": {
                "step": np.array([r["step"] for r in rows], dtype=np.int64),
                "mean": np.array([r["mean"] for r in rows], dtype=np.float64),
            },
            "units": {"step": "step", "mean": "m"},
        }
    }
