"""Seeded harmonic particle on OpenMM Reference; exact replay is empirical."""

from pathlib import Path
import numpy as np

UNITS = {"step": "step", "time": "ps", "position": "nm", "velocity": "nm/ps"}


def create(config, run_dir):
    from restartwitness.driver import require_version

    require_version("openmm", "8.4.0.post2")
    import openmm as mm

    system = mm.System()
    system.addParticle(12.0)
    force = mm.CustomExternalForce("0.5*k*(x*x+y*y+z*z)")
    force.addGlobalParameter("k", 25.0)
    force.addParticle(0, [])
    system.addForce(force)
    integrator = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.001)
    integrator.setRandomNumberSeed(271828)
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName("Reference"))
    context.setPositions([[0.1, 0.2, 0.3]])
    context.setVelocities([[0.01, 0.02, -0.03]])
    return (context, integrator)


def advance_one(state):
    state[1].step(1)


def observe(state):
    from openmm import unit

    c = state[0]
    s = c.getState(getPositions=True, getVelocities=True)
    return {
        "step": np.array(c.getStepCount(), dtype=np.int64),
        "time": np.array(s.getTime().value_in_unit(unit.picosecond)),
        "position": np.array(
            s.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        ),
        "velocity": np.array(
            s.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            )
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
