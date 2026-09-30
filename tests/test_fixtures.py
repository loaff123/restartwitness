import importlib.util
import numpy as np
import pytest


def adapter():
    assert importlib.util.find_spec("restartwitness.adapters.synthetic"), (
        "synthetic adapter missing"
    )
    from restartwitness.adapters import synthetic

    return synthetic


@pytest.mark.parametrize("model", ["integrator", "stochastic", "events"])
def test_clean_fixture_native_state_roundtrip_and_output_keys(model, tmp_path):
    d = adapter()
    c = {"model": model, "seed": 271828}
    root = tmp_path / "out"
    root.mkdir()
    s = d.create(c, root)
    for _ in range(7):
        d.advance_one(s)
    before = d.observe(s)
    assert all(np.array_equal(v, d.observe(s)[k]) for k, v in before.items())
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    d.save(s, checkpoint)
    r = d.restore(c, checkpoint, root)
    for _ in range(9):
        d.advance_one(s)
        d.advance_one(r)
    assert all(
        v.tobytes() == d.observe(r)[k].tobytes() for k, v in d.observe(s).items()
    )


@pytest.mark.parametrize(
    "fault,model,cut",
    [
        ("missing_rng", "stochastic", 7),
        ("seed_reset", "stochastic", 7),
        ("missing_cached_stochastic", "stochastic", 7),
        ("integrator_cache", "integrator", 7),
        ("evolving_timestep", "integrator", 9),
        ("forcing_phase", "integrator", 9),
        ("average_sum", "events", 7),
        ("average_count", "events", 7),
        ("output_id", "events", 9),
        ("duplicate_record", "events", 8),
        ("skipped_record", "events", 7),
        ("restored_callback", "events", 7),
    ],
)
def test_each_controlled_fault_has_real_reachable_state(fault, model, cut, tmp_path):
    d = adapter()
    c = {"model": model, "fault": fault, "seed": 271828}
    for name in ["u", "r"]:
        (tmp_path / name).mkdir()
    u = d.create(c, tmp_path / "u")
    r = d.create(c, tmp_path / "r")
    for _ in range(cut):
        d.advance_one(u)
        d.advance_one(r)
    cp = tmp_path / "cp"
    cp.mkdir()
    d.save(r, cp)
    r = d.restore(c, cp, tmp_path / "r")
    changed = False
    for _ in range(cut, 64):
        d.advance_one(u)
        d.advance_one(r)
        changed |= any(
            v.tobytes() != d.observe(r)[k].tobytes() for k, v in d.observe(u).items()
        )
    a, b = d.collect_outputs(tmp_path / "u"), d.collect_outputs(tmp_path / "r")
    if a:
        changed |= a["events"]["keys"] != b["events"]["keys"]
        changed |= (
            a["events"]["fields"]["mean"].tobytes()
            != b["events"]["fields"]["mean"].tobytes()
        )
    assert changed, fault


def test_negative_fixture_contract_controls(tmp_path):
    d = adapter()
    from restartwitness.driver import UnsupportedProfile

    with pytest.raises(UnsupportedProfile):
        d.create({"fault": "unsupported"}, tmp_path / "unsupported")
    s = d.create(
        {"model": "events", "fault": "malformed_outputs"}, tmp_path / "malformed"
    )
    for _ in range(8):
        d.advance_one(s)
    assert d.collect_outputs(tmp_path / "malformed")["events"]["keys"][0] is None
    assert d.collect_outputs(tmp_path / "empty") == {}


def test_callback_fault_is_transient_but_history_preserves_it(tmp_path):
    d = adapter()
    c = {"model": "events", "fault": "restored_callback"}
    for name in ("u", "r"):
        (tmp_path / name).mkdir()
    u = d.create(c, tmp_path / "u")
    r = d.create(c, tmp_path / "r")
    for _ in range(7):
        d.advance_one(u)
        d.advance_one(r)
    cp = tmp_path / "cp"
    cp.mkdir()
    d.save(r, cp)
    r = d.restore(c, cp, tmp_path / "r")
    d.advance_one(u)
    d.advance_one(r)
    assert d.observe(u)["x"] != d.observe(r)["x"]
    for _ in range(8, 64):
        d.advance_one(u)
        d.advance_one(r)
    assert d.observe(u)["x"].tobytes() == d.observe(r)["x"].tobytes()
    assert (
        d.collect_outputs(tmp_path / "u")["events"]["fields"]["mean"].tobytes()
        != d.collect_outputs(tmp_path / "r")["events"]["fields"]["mean"].tobytes()
    )
