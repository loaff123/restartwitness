"""Trusted test driver with genuinely persisted application step and output keys."""

import json
import time
from pathlib import Path
import numpy as np

UNITS = {"step": "", "x": ""}


def create(config, run_dir):
    return {"step": 0, "x": 0.0, "config": config, "root": str(run_dir)}


def advance_one(s):
    if s["config"].get("timeout"):
        time.sleep(5)
    s["step"] += 1
    s["x"] += 1.0


def observe(s):
    if s["config"].get("mutate"):
        s["x"] += 0.1
    return {"step": np.array(s["step"], dtype=np.int64), "x": np.array(s["x"])}


def save(s, p):
    if s["config"].get("failed_save"):
        raise RuntimeError("intentional save failure")
    if s["config"].get("empty"):
        return
    (Path(p) / "state.json").write_text(json.dumps(s))
    if s["config"].get("save_side_effect"):
        s["x"] += 0.5


def restore(c, p, run_dir):
    if c.get("failed_restore"):
        raise RuntimeError("intentional restore failure")
    s = json.loads((Path(p) / "state.json").read_text())
    s["root"] = str(run_dir)
    if c.get("restore_fault"):
        s["x"] += 0.5
    return s


def collect_outputs(p):
    return {}
