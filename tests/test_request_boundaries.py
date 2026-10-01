"""Segment provenance must agree with the declared synthetic experiment."""

import shutil

import pytest

from restartwitness.evidence import (
    IntegrityError,
    digest,
    read_json,
    verify_bundle,
    write_json,
)
from restartwitness.runner import read_verified_case, run_case
from tests.independent_check import CheckError, check_case
from tests.test_independent_check import reseal
from tests.test_runner import study


@pytest.fixture(scope="module")
def boundary_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("request-boundaries")
    cases = {}
    for label, cuts in [("single", [2]), ("multi", [0, 2, 2, 4]), ("empty", [])]:
        path = root / label
        run_case(study(), cuts, path)
        cases[label] = path
    return cases


@pytest.mark.parametrize("label", ["single", "multi", "empty"])
def test_valid_request_boundaries_cover_repeated_and_terminal_cuts(
    boundary_cases, label
):
    root = boundary_cases[label]
    assert (
        read_verified_case(root)["adjudication"]["status"]
        == "equivalent_under_contract"
    )
    assert check_case(root)["status"] == "equivalent_under_contract"
    cuts = read_json(root / "schedule.json")
    for i in range(len(cuts) + 1):
        request = read_json(root / "R" / f"request-{i:03}.json")
        assert request["start_step"] == (cuts[i - 1] if i else 0)
        assert request["next_cut"] == i


@pytest.mark.parametrize(
    "reader,error", [(read_verified_case, IntegrityError), (check_case, CheckError)]
)
@pytest.mark.parametrize(
    "label,arm,index,field,value",
    [
        ("single", "R", 1, "start_step", 0),
        ("single", "R", 1, "next_cut", 0),
        ("multi", "R", 2, "next_cut", 1),
        ("multi", "R", 3, "start_step", 3),
        ("multi", "R", 4, "start_step", 0),
        ("multi", "R", 4, "next_cut", 3),
        ("single", "P", 0, "start_step", 2),
        ("single", "U2", 0, "next_cut", 1),
        ("empty", "R", 0, "next_cut", 1),
        ("empty", "R", 0, "start_step", False),
        ("single", "R", 1, "start_step", 2.0),
        ("single", "R", 1, "next_cut", True),
        ("single", "R", 1, "next_cut", None),
        ("single", "R", 1, "start_step", -1),
    ],
)
def test_resealed_request_boundary_mismatch_rejected_without_numeric_changes(
    boundary_cases, tmp_path, reader, error, label, arm, index, field, value
):
    root = tmp_path / "case"
    shutil.copytree(boundary_cases[label], root)
    arrays_before = {p.relative_to(root): digest(p) for p in root.rglob("*.npy")}
    request_path = root / arm / f"request-{index:03}.json"
    request = read_json(request_path)
    request[field] = value
    write_json(request_path, request)
    reseal(root)
    # Hash integrity and all numeric observations still hold; only provenance changed.
    verify_bundle(root)
    assert {
        p.relative_to(root): digest(p) for p in root.rglob("*.npy")
    } == arrays_before
    with pytest.raises(error, match="segment boundary"):
        reader(root)


@pytest.mark.parametrize(
    "reader,error", [(read_verified_case, IntegrityError), (check_case, CheckError)]
)
@pytest.mark.parametrize("arm", ["P", "R"])
def test_partial_arm_cannot_declare_a_segment_past_its_schedule(
    boundary_cases, tmp_path, reader, error, arm
):
    root = tmp_path / "case"
    shutil.copytree(boundary_cases["single"], root)
    data = read_json(root / f"{arm}.json")
    index = len(data["segments"])
    request = read_json(root / arm / "request-000.json")
    write_json(root / arm / f"request-{index:03}.json", request)
    data["status"] = "unsupported"
    data["segments"].append({"status": "unsupported", "advance_calls": 0})
    write_json(root / f"{arm}.json", data)
    metrics = read_json(root / "metrics.json")
    metrics["process_starts"] += 1
    write_json(root / "metrics.json", metrics)
    reseal(root)
    with pytest.raises(error, match="segment boundary"):
        reader(root, check_cached=False)
