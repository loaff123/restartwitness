"""Original bounded fixture for historical OpenMM #4716, not an injected bug.

Parameters and 100+10-step scale derive from the upstream PR #4740
testLangevin regression. See protocol.json and amendments.json for changes.
Only native Context checkpoints restore application state.
"""

from pathlib import Path
import numpy as np

UNITS = {
    "step": "step",
    "time": "ps",
    "position": "nm",
    "velocity": "nm/ps",
    "box": "nm",
}


def create(config, run_dir):
    from restartwitness.driver import require_version, UnsupportedProfile

    require_version("openmm", config["package_version"])
    import openmm as mm

    if mm.version.git_revision != config["git_revision"]:
        raise UnsupportedProfile("OpenMM embedded git revision mismatch")
    system = mm.System()
    force = mm.NonbondedForce()
    force.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(1.0)
    for i in range(10):
        system.addParticle(1.0)
        force.addParticle(0.1 if i % 2 == 0 else -0.1, 0.2, 0.1)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(3, 0, 0), mm.Vec3(0, 3, 0), mm.Vec3(0, 0, 3)
    )
    integrator = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.001)
    integrator.setRandomNumberSeed(config["seed"])
    platform = mm.Platform.getPlatformByName(config["platform"])
    properties = (
        {"Threads": "1", "DeterministicForces": "true"}
        if config["platform"] == "CPU"
        else {}
    )
    context = mm.Context(system, integrator, platform, properties)
    context.setPositions(
        [
            [0.3 + 0.8 * (i % 3), 0.3 + 0.8 * ((i // 3) % 3), 0.3 + 0.8 * (i // 9)]
            for i in range(10)
        ]
    )
    context.setVelocities([[0.0, 0.0, 0.0]] * 10)
    return context, integrator


def advance_one(state):
    state[1].step(1)


def observe(state):
    from openmm import unit

    context = state[0]
    snapshot = context.getState(getPositions=True, getVelocities=True)
    return {
        "step": np.array(context.getStepCount(), dtype=np.int64),
        "time": np.array(
            snapshot.getTime().value_in_unit(unit.picosecond), dtype=np.float64
        ),
        "position": np.array(
            snapshot.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        ),
        "velocity": np.array(
            snapshot.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            dtype=np.float64,
        ),
        "box": np.array(
            snapshot.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        ),
    }


def save(state, checkpoint_dir):
    (Path(checkpoint_dir) / "native.chk").write_bytes(state[0].createCheckpoint())


def restore(config, checkpoint_dir, run_dir):
    state = create(config, run_dir)
    state[0].loadCheckpoint((Path(checkpoint_dir) / "native.chk").read_bytes())
    return state


def collect_outputs(run_dir):
    return {}
