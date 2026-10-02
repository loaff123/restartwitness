"""Synthetic adapter for testing unsupported values at the ingestion boundary."""

import json
import numpy as np
from restartwitness.adapters import synthetic

UNITS = synthetic.UNITS
advance_one = synthetic.advance_one
save = synthetic.save


class ArraySubclass(np.ndarray):
    pass


def unsupported(value, kind):
    if kind == "masked":
        return np.ma.array(value, mask=True)
    if kind == "unmasked":
        return np.ma.array(value, mask=False)
    if kind == "subclass":
        return value.view(ArraySubclass)
    if kind == "list":
        return [value.item()] if value.ndim == 0 else value.tolist()
    if kind == "scalar":
        return float(value.flat[0])
    if kind == "numpy_scalar":
        return value.flat[0]
    return value


def _configure(state):
    from pathlib import Path

    (Path(state["root"]) / "array-config.json").write_text(
        json.dumps(state["config"]), encoding="utf-8"
    )
    return state


def create(config, run_dir):
    return _configure(synthetic.create(config, run_dir))


def restore(config, checkpoint_dir, run_dir):
    state = synthetic.restore(config, checkpoint_dir, run_dir)
    state["restored"] = True
    return _configure(state)


def observe(state):
    fields = synthetic.observe(state)
    config = state["config"]
    stage = config.get("array_stage", "initial")
    active = (
        stage == "initial"
        or (stage == "prefix" and state["step"] >= 1)
        or (stage == "after_restore" and state.get("restored", False))
    )
    if config.get("array_target") == "observation" and active:
        fields["x"] = unsupported(fields["x"], config["array_kind"])
    return fields


def collect_outputs(run_dir):
    outputs = synthetic.collect_outputs(run_dir)
    config = json.loads((run_dir / "array-config.json").read_text(encoding="utf-8"))
    if config.get("array_target") == "output":
        fields = outputs["events"]["fields"]
        fields["mean"] = unsupported(fields["mean"], config["array_kind"])
    return outputs
