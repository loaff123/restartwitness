"""Documented misuse example: State does not preserve all RNG internal state.

This intentionally weaker workflow is NOT an OpenMM checkpoint bug.
"""

from pathlib import Path
from .openmm_native import (
    create as create,
    advance_one as advance_one,
    observe as observe,
    collect_outputs as collect_outputs,
    UNITS as UNITS,
)


def save(state, checkpoint_dir):
    from openmm import XmlSerializer

    s = state[0].getState(getPositions=True, getVelocities=True, getParameters=True)
    (Path(checkpoint_dir) / "state.xml").write_text(
        XmlSerializer.serialize(s), encoding="utf-8"
    )


def restore(config, checkpoint_dir, run_dir):
    from openmm import XmlSerializer

    state = create(config, run_dir)
    state[0].setState(
        XmlSerializer.deserialize(
            (Path(checkpoint_dir) / "state.xml").read_text(encoding="utf-8")
        )
    )
    return state
