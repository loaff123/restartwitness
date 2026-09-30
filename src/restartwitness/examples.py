"""Predeclared original examples. User adapters supply their own fixed contracts."""

from .contracts import Contract, FieldContract, RecordContract


def example_study(model="integrator", fault="clean", total_steps=64, seed=271828):
    if model in ("integrator", "stochastic", "events"):
        fields = (
            FieldContract("step", "<i8", (), "step"),
            FieldContract("time", "<f8", (), "s"),
            FieldContract("x", "<f8", (), "m"),
            FieldContract("v", "<f8", (), "m/s"),
        )
        events = tuple(k for k in (8, 32, 64) if k <= total_steps)
        outputs = (
            {
                "events": RecordContract(
                    "events",
                    tuple(f"average:{i}@{k}" for i, k in enumerate(events)),
                    (
                        FieldContract("step", "<i8", (), "step"),
                        FieldContract("mean", "<f8", (), "m"),
                    ),
                )
            }
            if model == "events" and events
            else {}
        )
        driver = "restartwitness.adapters.synthetic"
        config = {"model": model, "fault": fault, "seed": seed}
    elif model in ("rebound", "openmm", "openmm-state-misuse"):
        is_rebound = model == "rebound"
        n = 2 if is_rebound else 1
        units = (
            ("rebound time", "rebound length", "rebound velocity")
            if is_rebound
            else ("ps", "nm", "nm/ps")
        )
        fields = (
            FieldContract("step", "<i8", (), "step"),
            FieldContract("time", "<f8", (), units[0]),
            FieldContract("position", "<f8", (n, 3), units[1]),
            FieldContract("velocity", "<f8", (n, 3), units[2]),
        )
        driver = (
            "restartwitness.adapters."
            + {
                "rebound": "rebound_native",
                "openmm": "openmm_native",
                "openmm-state-misuse": "openmm_state_misuse",
            }[model]
        )
        outputs = {}
        config = {}
    else:
        raise ValueError("unknown example model")
    return {
        "driver": driver,
        "config": config,
        "total_steps": total_steps,
        "observations": list(range(total_steps + 1)),
        "contract": Contract(fields, outputs).to_dict(),
        "worker_timeout": 60.0,
        "study_budget": 600.0,
        "termination": "normal",
    }
