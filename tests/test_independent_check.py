"""Adversarial checks of a reader that does not import production adjudication."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec("tests.independent_check"), (
        "independent re-adjudicator is required"
    )
    return __import__("tests.independent_check", fromlist=["check_case"])


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf8")


def reseal(root):
    previous = root / "manifest.json"
    metadata = (
        json.loads(previous.read_text())["metadata"]
        if previous.exists()
        else {
            "kind": "restartwitness-case",
            "identity": {
                "driver": "example.local",
                "driver_sha256": "0" * 64,
                "core_sources": {},
            },
            "termination": "normal",
        }
    )
    manifest = {"schema_version": 1, "metadata": metadata, "artifacts": []}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            b = p.read_bytes()
            manifest["artifacts"].append(
                {
                    "path": p.relative_to(root).as_posix(),
                    "bytes": len(b),
                    "sha256": hashlib.sha256(b).hexdigest(),
                }
            )
    dump(root / "manifest.json", manifest)


def field(dtype="<f8", mode="exact", atol=0.0, rtol=0.0):
    return {
        "name": "x",
        "dtype": dtype,
        "shape": [],
        "unit": "m",
        "mode": mode,
        "atol": atol,
        "rtol": rtol,
    }


def fixture(
    root, values=None, mode="exact", atol=0.0, rtol=0.0, dtype="<f8", output_keys=None
):
    """Hand-built evidence, avoiding production encoders and contracts."""
    root.mkdir()
    values = values or {}
    f = field(dtype, mode, atol, rtol)
    contract = {"schema_version": 1, "fields": [f], "outputs": {}}
    if output_keys is not None:
        contract["outputs"]["samples"] = {
            "name": "samples",
            "expected_keys": ["step:0", "step:1"],
            "fields": [f],
        }
    dump(
        root / "study.json",
        {
            "driver": "example.local",
            "config": {},
            "total_steps": 1,
            "observations": [0, 1],
            "contract": contract,
            "termination": "normal",
        },
    )
    dump(root / "schedule.json", [])
    for name in ("U1", "U2", "O", "P", "R"):
        observations = []
        for step in [1] if name == "O" else [0, 1]:
            path = root / f"{name}-{step}.npy"
            value = values.get(name, 0.0)
            if isinstance(value, (list, tuple)):
                value = value[step]
            np.save(path, np.array(value, dtype=dtype), allow_pickle=False)
            observations.append(
                {
                    "key": [step, "ordinary", 0],
                    "fields": {"x": {"$array": path.name}},
                    "units": {"x": "m"},
                }
            )
        outputs = {}
        if output_keys is not None:
            keys = output_keys.get(name, ["step:0", "step:1"])
            p = root / f"{name}-output.npy"
            np.save(p, np.arange(len(keys), dtype=dtype), allow_pickle=False)
            outputs["samples"] = {
                "keys": keys,
                "fields": {"x": {"$array": p.name}},
                "units": {"x": "m"},
            }
        dump(
            root / f"{name}.json",
            {
                "status": "complete",
                "observations": observations,
                "outputs": outputs,
                "segments": [],
            },
        )
    cached = {
        "status": "equivalent_under_contract",
        "findings": ["equivalent_under_contract"],
        "first_witness": None,
        "comparisons": {
            name: {"status": "equivalent", "first_witness": None}
            for name in ("U1-U2", "U1-O", "U-P", "P-R", "U-R")
        },
    }
    dump(root / "adjudication.json", cached)
    reseal(root)
    return root


def test_independent_reader_accepts_clean_bundle(tmp_path):
    result = api().check_case(fixture(tmp_path / "case"))
    assert result["status"] == "equivalent_under_contract"
    assert set(result["comparisons"]) == {"U1-U2", "U1-O", "U-P", "P-R", "U-R"}
    assert all(p["status"] == "equivalent" for p in result["comparisons"].values())


def test_stale_cached_pass_rejected_after_raw_array_and_hash_coherently_changed(
    tmp_path,
):
    root = fixture(tmp_path / "case")
    np.save(root / "R-0.npy", np.array(9.0), allow_pickle=False)
    reseal(root)  # A hash-only checker would approve this bundle.
    with pytest.raises(api().CheckError, match="cached"):
        api().check_case(root)
    result = api().check_case(root, check_cached=False)
    assert result["status"] == "restore_path_difference"
    assert result["comparisons"]["P-R"]["first_witness"]["key"] == [0, "ordinary", 0]
    assert result["comparisons"]["P-R"]["first_witness"]["right"] == 9.0


def test_cached_fault_label_cannot_override_clean_raw_arrays(tmp_path):
    root = fixture(tmp_path / "case")
    cached = json.loads((root / "adjudication.json").read_text())
    cached["status"] = "restore_path_difference"
    dump(root / "adjudication.json", cached)
    reseal(root)
    with pytest.raises(api().CheckError, match="cached"):
        api().check_case(root)


@pytest.mark.parametrize(
    "name,status",
    [
        ("U2", "baseline_inconclusive"),
        ("O", "observation_inconclusive"),
        ("P", "save_path_difference"),
        ("R", "restore_path_difference"),
    ],
)
def test_negative_controls_and_attribution_recomputed(name, status, tmp_path):
    result = api().check_case(
        fixture(tmp_path / "case", {name: 2.0}), check_cached=False
    )
    assert result["status"] == status
    if name in ("U2", "O"):
        assert "restore_path_difference" not in result["findings"]


def test_direct_u_r_prevents_tolerance_transitivity_error(tmp_path):
    result = api().check_case(
        fixture(tmp_path / "case", {"P": 0.75, "R": 1.5}, mode="tolerance", atol=1.0),
        check_cached=False,
    )
    assert result["comparisons"]["U-P"]["status"] == "equivalent"
    assert result["comparisons"]["P-R"]["status"] == "equivalent"
    assert result["comparisons"]["U-R"]["status"] == "difference"
    assert result["status"] == "restart_difference"


@pytest.mark.parametrize(
    "dtype,left,right,mode,atol,rtol,status",
    [
        ("<f8", 0.0, -0.0, "exact", 0.0, 0.0, "difference"),
        ("<f8", 0.0, -0.0, "tolerance", 0.0, 0.0, "equivalent"),
        ("<u8", 2**64 - 1, 2**64 - 2, "tolerance", 1e30, 1e30, "difference"),
        ("<f8", -1e308, 1e308, "tolerance", 0.0, 1.9, "difference"),
        ("<f8", -1e308, 1e308, "tolerance", 0.0, 2.0, "equivalent"),
        ("<f8", float("nan"), float("nan"), "exact", 0.0, 0.0, "invalid"),
    ],
)
def test_numeric_rules_independently_recomputed(
    tmp_path, dtype, left, right, mode, atol, rtol, status
):
    root = fixture(
        tmp_path / "case",
        {"U1": left, "U2": left, "O": left, "P": left, "R": right},
        mode,
        atol,
        rtol,
        dtype,
    )
    result = api().check_case(root, check_cached=False)
    assert result["comparisons"]["P-R"]["status"] == status
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "keys,kind",
    [
        (["step:0", "step:0", "step:1"], "duplicate_record_keys"),
        (["step:0"], "missing_record_keys"),
        (["step:0", "step:1", "step:2"], "unexpected_record_keys"),
        (["step:1", "step:0"], "reordered_record_keys"),
    ],
)
def test_output_key_contract_checked_even_when_every_arm_has_same_fault(
    tmp_path, keys, kind
):
    root = fixture(
        tmp_path / "case",
        output_keys={name: keys for name in ("U1", "U2", "O", "P", "R")},
    )
    result = api().check_case(root, check_cached=False)
    assert "output_contract_violation" in result["findings"]
    assert kind in [d["kind"] for d in result["comparisons"]["U-P"]["diagnostics"]]


def test_first_intermediate_difference_not_hidden_by_equal_final_state(tmp_path):
    result = api().check_case(
        fixture(tmp_path / "case", {"R": [4.0, 0.0]}), check_cached=False
    )
    assert result["comparisons"]["U-R"]["first_witness"]["key"] == [0, "ordinary", 0]


@pytest.mark.parametrize(
    "mutation",
    [
        "bytes",
        "missing",
        "extra",
        "duplicate_path",
        "escape",
        "symlink",
        "truncated",
        "object",
        "huge",
        "trailing",
        "duplicate_json",
    ],
)
def test_integrity_limits_fail_closed(tmp_path, mutation):
    root = fixture(tmp_path / "case")
    if mutation == "bytes":
        (root / "R-0.npy").write_bytes(b"corrupt")
    elif mutation == "missing":
        (root / "R-0.npy").unlink()
    elif mutation == "extra":
        (root / "extra").write_text("unlisted")
    elif mutation == "duplicate_path":
        m = json.loads((root / "manifest.json").read_text())
        m["artifacts"].append(m["artifacts"][0])
        dump(root / "manifest.json", m)
    elif mutation == "escape":
        a = json.loads((root / "R.json").read_text())
        a["observations"][0]["fields"]["x"]["$array"] = "../escape.npy"
        dump(root / "R.json", a)
        reseal(root)
    elif mutation == "symlink":
        (root / "R-0.npy").unlink()
        (root / "R-0.npy").symlink_to(root / "U1-0.npy")
        reseal(root)
    elif mutation == "truncated":
        (root / "manifest.json").write_text("{")
    elif mutation == "object":
        np.save(root / "R-0.npy", np.array({"x": 1}, dtype=object))
        reseal(root)
    elif mutation == "huge":
        with (root / "R-0.npy").open("wb") as out:
            np.lib.format.write_array_header_1_0(
                out, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)}
            )
        reseal(root)
    elif mutation == "trailing":
        with (root / "R-0.npy").open("ab") as out:
            out.write(b"trailing")
        reseal(root)
    elif mutation == "duplicate_json":
        (root / "R.json").write_text('{"status":"complete","status":"complete"}')
        reseal(root)
    with pytest.raises(api().CheckError):
        api().check_case(root)


def test_standalone_cli_works_without_production_package_on_path(tmp_path):
    root = fixture(tmp_path / "case")
    script = Path(__file__).with_name("independent_check.py")
    api()
    result = subprocess.run(
        [sys.executable, "-I", str(script), str(root)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["status"] == "equivalent_under_contract"


def test_common_deleted_observation_cannot_become_equivalent_by_rehashing(tmp_path):
    root = fixture(tmp_path / "case")
    for name in ("U1", "U2", "P", "R"):
        arm = json.loads((root / f"{name}.json").read_text())
        arm["observations"] = arm["observations"][1:]
        dump(root / f"{name}.json", arm)
    reseal(root)
    with pytest.raises(api().CheckError, match="observation|cadence"):
        api().check_case(root)


@pytest.mark.parametrize(
    "mutation",
    ["phase_order", "missing_phase", "wrong_cut", "stale_occurrence", "bad_schedule"],
)
def test_declared_cut_sequence_and_phases_cannot_be_rewritten_consistently(
    tmp_path, mutation
):
    root = fixture(tmp_path / "case")
    dump(root / "schedule.json", [0, 0])
    for name in ("U1", "U2", "P", "R"):
        arm = json.loads((root / f"{name}.json").read_text())
        ordinary = arm["observations"][0]
        phases = [
            dict(ordinary, key=[0, phase, cut])
            for cut in (0, 1)
            for phase in ("before_save", "after_save", "after_restore")
        ]
        arm["observations"] = [ordinary] + phases + arm["observations"][1:]
        if mutation == "phase_order":
            arm["observations"][1:3] = arm["observations"][1:3][::-1]
        elif mutation == "missing_phase":
            del arm["observations"][2]
        elif mutation == "wrong_cut":
            arm["observations"][1]["key"][0] = 1
        elif mutation == "stale_occurrence":
            arm["observations"][4]["key"][2] = 0
        dump(root / f"{name}.json", arm)
    if mutation == "bad_schedule":
        dump(root / "schedule.json", [True])
    reseal(root)
    with pytest.raises(api().CheckError, match="observation|cadence|schedule"):
        api().check_case(root)


@pytest.mark.parametrize(
    "field,value",
    [
        ("key", [1, "ordinary", 0]),
        ("field", "invented"),
        ("index", [1]),
        ("right", 300.0),
        ("left_bits", "0xff"),
        ("absolute_difference", 0.0),
    ],
)
def test_cached_first_witness_must_match_actual_numeric_evidence(
    tmp_path, field, value
):
    root = fixture(tmp_path / "case", {"R": [4.0, 0.0]})
    cached = api().check_case(root, check_cached=False)
    # A fully consistent cached fault initially verifies; alter only a witness.
    dump(root / "adjudication.json", cached)
    reseal(root)
    assert api().check_case(root)["status"] == "restore_path_difference"
    cached["comparisons"]["P-R"]["first_witness"][field] = value
    dump(root / "adjudication.json", cached)
    reseal(root)
    with pytest.raises(api().CheckError, match="cached"):
        api().check_case(root)


def test_output_numeric_witness_recomputed_by_semantic_key(tmp_path):
    root = fixture(tmp_path / "case", output_keys={})
    np.save(root / "R-output.npy", np.array([0.0, 9.0]), allow_pickle=False)
    reseal(root)
    result = api().check_case(root, check_cached=False)
    witness = result["comparisons"]["P-R"]["first_witness"]
    assert witness["record_key"] == "step:1"
    assert witness["index"] == [1]
    assert witness["right"] == 9.0
    assert "restore_path_difference" in result["findings"]
    assert "output_contract_violation" in result["findings"]


@pytest.mark.parametrize(
    "status", ["timeout", "driver_error", "checkpoint_error", "unsupported", "not_run"]
)
def test_recorded_failed_arm_never_counts_as_complete(status, tmp_path):
    root = fixture(tmp_path / "case")
    arm = json.loads((root / "R.json").read_text())
    arm["status"] = status
    dump(root / "R.json", arm)
    reseal(root)
    result = api().check_case(root, check_cached=False)
    assert result["status"] == (status if status != "not_run" else "incomplete")
    assert "equivalent_under_contract" not in result["findings"]


@pytest.mark.parametrize(
    "config,cuts",
    [
        ({}, []),
        ({}, [0, 2, 2, 4]),
        ({"restore_fault": True}, [2]),
        ({"save_side_effect": True}, [4]),
        ({"mutate": True}, [2]),
        ({"failed_save": True}, [2]),
    ],
)
def test_independent_checker_matches_actual_runner_evidence(tmp_path, config, cuts):
    from restartwitness.runner import run_case
    from tests.test_runner import study

    root = tmp_path / "case"
    run = run_case(study(**config), cuts, root)
    checked = api().check_case(root)
    assert checked["status"] == run["adjudication"]["status"]
    assert checked["findings"] == run["adjudication"]["findings"]
    for name, pair in checked["comparisons"].items():
        assert pair["status"] == run["adjudication"]["comparisons"][name]["status"]
        assert (
            pair["first_witness"]
            == run["adjudication"]["comparisons"][name]["first_witness"]
        )


def test_logical_reference_budget_limits_repeated_shared_arrays(tmp_path, monkeypatch):
    root = fixture(tmp_path / "case")
    # Nine references to the same eight-byte scalar still imply repeated work.
    for name in ("U1", "U2", "O", "P", "R"):
        arm = json.loads((root / f"{name}.json").read_text())
        for snapshot in arm["observations"]:
            snapshot["fields"]["x"] = {"$array": "U1-0.npy"}
        dump(root / f"{name}.json", arm)
    reseal(root)
    monkeypatch.setattr(api(), "MAX_DECODE_BYTES", 64, raising=False)
    with pytest.raises(api().CheckError, match="decoded|reference"):
        api().check_case(root)


@pytest.mark.parametrize("mutation", ["dtype", "shape", "unit", "field", "unit_names"])
def test_observed_schema_independently_validated(tmp_path, mutation):
    root = fixture(tmp_path / "case")
    arm = json.loads((root / "R.json").read_text())
    if mutation == "dtype":
        np.save(root / "R-0.npy", np.array(0, dtype="<f4"), allow_pickle=False)
    elif mutation == "shape":
        np.save(root / "R-0.npy", np.array([0.0]), allow_pickle=False)
    elif mutation == "unit":
        arm["observations"][0]["units"]["x"] = "cm"
    elif mutation == "field":
        arm["observations"][0]["fields"]["other"] = arm["observations"][0]["fields"][
            "x"
        ]
    else:
        arm["observations"][0]["units"] = {}
    dump(root / "R.json", arm)
    reseal(root)
    result = api().check_case(root, check_cached=False)
    assert result["comparisons"]["P-R"]["status"] == "invalid"
    assert result["status"] != "equivalent_under_contract"


def test_checker_has_no_production_import_dependency():
    import ast

    source = Path(__file__).with_name("independent_check.py").read_text()
    parsed = ast.parse(source)
    imported = [n.module for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)]
    imported += [
        a.name for n in ast.walk(parsed) if isinstance(n, ast.Import) for a in n.names
    ]
    assert not any(
        name and (name.startswith("restartwitness") or name.startswith("tests.test"))
        for name in imported
    )


@pytest.mark.parametrize(
    "mutation", ["changed_bytes", "missing_ack", "wrong_step", "escaping_artifact"]
)
def test_checkpoint_acknowledgments_remain_tied_to_native_files_and_cuts(
    tmp_path, mutation
):
    from restartwitness.runner import run_case
    from tests.test_runner import study

    root = tmp_path / "case"
    run_case(study(), [2], root)
    arm = json.loads((root / "R.json").read_text())
    checkpoint = arm["segments"][0]["checkpoints"][0]
    if mutation == "changed_bytes":
        native = (
            root
            / "R"
            / "segment-000"
            / "checkpoint-000"
            / checkpoint["artifacts"][0]["path"]
        )
        native.write_bytes(native.read_bytes() + b"\n")
    elif mutation == "missing_ack":
        arm["segments"][0]["checkpoints"] = []
    elif mutation == "wrong_step":
        checkpoint["step"] = 1
    else:
        checkpoint["artifacts"][0]["path"] = "../outside"
    dump(root / "R.json", arm)
    reseal(root)
    with pytest.raises(api().CheckError, match="checkpoint|path"):
        api().check_case(root)


@pytest.mark.parametrize("mutation", ["config", "identity_driver", "termination"])
def test_frozen_study_and_identity_must_match_recorded_worker_requests(
    tmp_path, mutation
):
    from restartwitness.runner import run_case
    from tests.test_runner import study

    root = tmp_path / "case"
    run_case(study(), [2], root)
    s = json.loads((root / "study.json").read_text())
    if mutation == "config":
        s["config"] = {"restore_fault": True}
        dump(root / "study.json", s)
    elif mutation == "identity_driver":
        m = json.loads((root / "manifest.json").read_text())
        m["metadata"]["identity"]["driver"] = "other.driver"
        dump(root / "manifest.json", m)
    else:
        s["termination"] = "exit_after_save"
        dump(root / "study.json", s)
    reseal(root)
    with pytest.raises(api().CheckError, match="study|identity|request|termination"):
        api().check_case(root)


def test_root_raw_and_labels_cannot_contradict_retained_worker_raw_arrays(tmp_path):
    from restartwitness.runner import run_case
    from tests.test_runner import study

    root = tmp_path / "case"
    run = run_case(study(restore_fault=True), [2], root)
    assert run["adjudication"]["status"] == "restore_path_difference"
    p = json.loads((root / "P.json").read_text())
    r = json.loads((root / "R.json").read_text())
    r["observations"] = p["observations"]
    dump(root / "R.json", r)
    # Both root raw values AND cached labels have now been coherently forged.
    cached = {
        "status": "equivalent_under_contract",
        "findings": ["equivalent_under_contract"],
        "first_witness": None,
        "comparisons": {
            name: {"status": "equivalent", "first_witness": None}
            for name in ("U1-U2", "U1-O", "U-P", "P-R", "U-R")
        },
    }
    dump(root / "adjudication.json", cached)
    reseal(root)
    with pytest.raises(api().CheckError, match="segment|aggregate|worker raw"):
        api().check_case(root)


@pytest.mark.parametrize("invalid_evidence", ["nonfinite", "unit"])
def test_independent_findings_retain_invalidity_with_valid_save_difference(
    tmp_path, invalid_evidence
):
    root = fixture(tmp_path / "case", {"P": 1.0})
    if invalid_evidence == "nonfinite":
        np.save(root / "R-0.npy", np.array(np.nan), allow_pickle=False)
    else:
        arm = json.loads((root / "R.json").read_text())
        arm["observations"][0]["units"]["x"] = "cm"
        dump(root / "R.json", arm)
    reseal(root)
    result = api().check_case(root, check_cached=False)
    # Expected findings come from the declared contract, not production output.
    assert result["findings"] == ["save_path_difference", "invalid"]
    assert result["comparisons"]["U-P"]["status"] == "difference"
    assert result["comparisons"]["P-R"]["status"] == "invalid"
    assert result["comparisons"]["U-R"]["status"] == "invalid"
    assert result["first_witness"]["comparison"] == "U-P"


def test_independent_unsupported_with_valid_difference_is_not_invalid_evidence(
    tmp_path,
):
    root = fixture(tmp_path / "case", {"P": 1.0})
    arm = json.loads((root / "R.json").read_text())
    arm["status"] = "unsupported"
    dump(root / "R.json", arm)
    reseal(root)
    result = api().check_case(root, check_cached=False)
    assert result["findings"] == ["unsupported"]
    assert result["comparisons"]["U-P"]["status"] == "difference"
