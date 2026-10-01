"""Offline checks for the opt-in historical experiment and public data."""

import copy
import json
from pathlib import Path

import pytest

from experiments.historical.openmm_cpu_rng import verify_excerpt
from experiments.historical.openmm_cpu_rng.run import make_study

ROOT = Path(__file__).parents[1] / "experiments/historical/openmm_cpu_rng"


def test_four_public_excerpts_are_hash_verified():
    reports = verify_excerpt.verify_directory(ROOT / "evidence")
    assert len(reports) == 4
    by_name = {r["case"]: r for r in reports}
    for name, report in by_name.items():
        assert report["controls_exact"]
        assert report["immediate_restore_exact"]
        expected = name.startswith("affected") or name.startswith("heldout-affected")
        assert report["absolute_1e-5_difference"] == expected
    assert by_name["fixed-cpu-100"]["exact_difference"]
    assert by_name["fixed-cpu-100"]["strict_1e-12_difference"]


def test_tampered_excerpt_rejected(tmp_path):
    for p in (ROOT / "evidence").iterdir():
        if p.is_file():
            (tmp_path / p.name).write_bytes(p.read_bytes())
    target = tmp_path / "affected-cpu-100.json"
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        verify_excerpt.verify_directory(tmp_path)


def test_nonfinite_and_duplicate_observation_rejected():
    source = json.loads((ROOT / "evidence/affected-cpu-100.json").read_text())
    invalid = copy.deepcopy(source)
    invalid["arms"]["R"][-1]["values"]["time"] = "inf"
    with pytest.raises(ValueError):
        verify_excerpt.check_excerpt(invalid)
    invalid = copy.deepcopy(source)
    invalid["arms"]["R"][-1]["key"] = invalid["arms"]["R"][-2]["key"]
    with pytest.raises(ValueError, match="keys"):
        verify_excerpt.check_excerpt(invalid)


def test_studies_pin_profiles_without_default_native_changes():
    exact = make_study("8.2.0", "CPU", 42, "exact")
    relaxed = make_study("8.3.0", "CPU", 43, "source-informed")
    assert exact["total_steps"] == relaxed["total_steps"] == 110
    assert exact["driver"].endswith("historical_openmm_native")
    assert exact["config"]["git_revision"] == "53770948682c40bd460b39830d4e0f0fd3a4b868"
    assert (
        relaxed["config"]["git_revision"] == "1ce5d91d9dedfdc273066fafa1a618bf05c25b85"
    )
    assert all(f["mode"] == "exact" for f in exact["contract"]["fields"])
    assert all(
        f["atol"] == 1e-5 for f in relaxed["contract"]["fields"] if f["dtype"] == "<f8"
    )
    with pytest.raises(ValueError):
        make_study("8.4.0", "CPU", 42, "exact")


def test_export_rejects_source_overlap_before_writing(tmp_path):
    from experiments.historical.openmm_cpu_rng.export_excerpt import export_case

    source = tmp_path / "sealed"
    source.mkdir()
    for destination in (source, source / "nested", tmp_path):
        with pytest.raises(ValueError, match="overlap"):
            export_case(source, destination, "case")
        assert list(source.iterdir()) == []


def test_export_rejects_symlink_destination_before_writing(tmp_path):
    from experiments.historical.openmm_cpu_rng.export_excerpt import export_case

    source = tmp_path / "sealed"
    source.mkdir()
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="symlink"):
        export_case(source, link, "case")
    assert list(target.iterdir()) == []


def test_export_reserves_manifest_name(tmp_path):
    from experiments.historical.openmm_cpu_rng.export_excerpt import export_case

    with pytest.raises(ValueError, match="reserved"):
        export_case(tmp_path / "source", tmp_path / "out", "manifest")
    assert not (tmp_path / "out").exists()
