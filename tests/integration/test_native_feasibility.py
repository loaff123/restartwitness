"""Independent-process feasibility probes, before any harness exists."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

PROBE = r"""
import importlib, json, sys
from pathlib import Path
import numpy as np
name, root, mode, dense = sys.argv[1:]
d = importlib.import_module('restartwitness.adapters.' + name)
p = Path(root); p.mkdir(parents=True, exist_ok=True)
s = d.restore({}, p/'checkpoint', p) if mode == 'restore' else d.create({}, p)
start = 8 if mode == 'restore' else 0
stop = 8 if mode == 'save' else 64
for i in range(start, stop):
    d.advance_one(s)
    if mode == 'continue_save' and i == 7:
        (p/'checkpoint').mkdir(); d.save(s, p/'checkpoint')
    if dense == 'yes': d.observe(s)
if mode == 'save':
    (p/'checkpoint').mkdir()
    d.save(s, p/'checkpoint')
np.savez(p/(mode+'.npz'), **d.observe(s))
print(json.dumps({'pid': __import__('os').getpid()}))
"""


def probe(name, root, mode, dense="yes"):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    p = subprocess.run(
        [sys.executable, "-c", PROBE, name, str(root), mode, dense],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    with np.load(root / (mode + ".npz"), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    return json.loads(p.stdout)["pid"], data


@pytest.mark.native
@pytest.mark.parametrize(
    "name,dependency", [("rebound_native", "rebound"), ("openmm_native", "openmm")]
)
def test_native_fresh_interpreter_roundtrip(name, dependency, tmp_path):
    if importlib.util.find_spec(dependency) is None:
        pytest.skip(f"NOT RUN: optional {dependency} is not installed")
    assert (
        Path(__file__).parents[2] / "src/restartwitness/adapters" / f"{name}.py"
    ).exists(), "native adapter missing"
    p1, u1 = probe(name, tmp_path / "u1", "continue")
    p2, u2 = probe(name, tmp_path / "u2", "continue")
    _, sparse = probe(name, tmp_path / "sparse", "continue", "no")
    ps, _ = probe(name, tmp_path / "r", "save")
    pr, restarted = probe(name, tmp_path / "r", "restore")
    assert ps != pr
    for key in u1:
        for comparison in (u2, sparse, restarted):
            assert u1[key].dtype == comparison[key].dtype
            assert u1[key].shape == comparison[key].shape
            assert u1[key].tobytes() == comparison[key].tobytes(), key
    assert int(u1["step"]) == 64
    assert float(u1["time"]) > 0


@pytest.mark.native
def test_state_only_misuse_distinguishes_save_from_restore(tmp_path):
    if importlib.util.find_spec("openmm") is None:
        pytest.skip("NOT RUN: optional OpenMM is not installed")
    assert (
        Path(__file__).parents[2] / "src/restartwitness/adapters/openmm_state_misuse.py"
    ).exists()
    _, u = probe("openmm_native", tmp_path / "u", "continue")
    _, p = probe("openmm_state_misuse", tmp_path / "p", "continue_save")
    probe("openmm_state_misuse", tmp_path / "r", "save")
    _, r = probe("openmm_state_misuse", tmp_path / "r", "restore")
    assert u["position"].tobytes() == p["position"].tobytes()
    assert u["position"].tobytes() != r["position"].tobytes()
