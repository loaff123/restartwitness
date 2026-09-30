"""REBOUND 4.4.11 leapfrog orbital system using its native Simulationarchive."""

from pathlib import Path
import numpy as np

UNITS = {
    "step": "step",
    "time": "rebound time",
    "position": "rebound length",
    "velocity": "rebound velocity",
}


def create(config, run_dir):
    from restartwitness.driver import require_version

    require_version("rebound", "4.4.11")
    import rebound

    s = rebound.Simulation()
    s.integrator = "leapfrog"
    s.dt = 0.001
    s.add(m=1.0)
    s.add(m=0.001, x=1.0, vy=1.0)
    s.move_to_com()
    return s


def advance_one(state):
    state.step()


def observe(state):
    return {
        "step": np.array(state.steps_done, dtype=np.int64),
        "time": np.array(state.t),
        "position": np.array([[p.x, p.y, p.z] for p in state.particles]),
        "velocity": np.array([[p.vx, p.vy, p.vz] for p in state.particles]),
    }


def save(state, checkpoint_dir):
    state.save_to_file(str(Path(checkpoint_dir) / "native.bin"))


def restore(config, checkpoint_dir, run_dir):
    import rebound

    return rebound.Simulation(str(Path(checkpoint_dir) / "native.bin"))


def collect_outputs(run_dir):
    return {}
