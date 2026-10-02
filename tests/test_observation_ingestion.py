"""Unsupported adapter values must never become equivalent after serialization."""

import json
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from restartwitness.evidence import IntegrityError, read_data, write_data
from restartwitness.examples import example_study
from restartwitness.report import render_junit, render_report
from restartwitness.runner import read_verified_case, run_case
from restartwitness.worker import run_segment
from tests.independent_check import check_case
from tests.test_cli import cli

KINDS = ["masked", "unmasked", "subclass", "list", "scalar", "numpy_scalar"]


def study(target, kind, stage="initial"):
    result = example_study("events" if target == "output" else "integrator", "clean", 8)
    result["driver"] = "tests.helpers.observation_types"
    result["config"].update(array_target=target, array_kind=kind, array_stage=stage)
    return result


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("array-types")
    cases = {}
    for stage in ("initial", "prefix", "after_restore"):
        for kind in KINDS:
            key = ("observation", kind, stage)
            path = root / "-".join(key)
            cases[key] = (path, run_case(study(*key), [4], path))
    for kind in KINDS + ["plain"]:
        key = ("output", kind, "initial")
        path = root / "-".join(key)
        cases[key] = (path, run_case(study(*key), [4], path))
    key = ("observation", "plain", "initial")
    path = root / "-".join(key)
    cases[key] = (path, run_case(study(*key), [4], path))
    return cases


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("stage", ["initial", "prefix", "after_restore"])
def test_unsupported_observations_fail_before_coercion(cases, kind, stage):
    root, result = cases[("observation", kind, stage)]
    affected = ["R"] if stage == "after_restore" else result["arms"]
    for name in affected:
        arm = result["arms"][name]
        assert arm["status"] == "driver_error"
        error = arm["segments"][-1]["error"]
        assert error["phase"] == "observe"
        assert "x" in error["message"]
        assert "ndarray" in error["message"]
    r = result["arms"]["R"]
    if stage == "initial":
        assert r["observations"] == []
    elif stage == "prefix":
        assert [x["key"] for x in r["observations"]] == [[0, "ordinary", 0]]
    else:
        assert r["segments"][0]["status"] == "checkpoint"
        assert r["observations"][-1]["key"] == [4, "after_save", 0]
        assert len({x["process_token"] for x in r["segments"]}) == 2
    assert_failed_bundle(root, result)


@pytest.mark.parametrize("kind", KINDS)
def test_unsupported_output_columns_leave_readable_failed_evidence(cases, kind):
    root, result = cases[("output", kind, "initial")]
    for arm in result["arms"].values():
        assert arm["status"] == "driver_error"
        error = arm["segments"][-1]["error"]
        assert error["phase"] == "outputs"
        assert "events" in error["message"] and "mean" in error["message"]
        assert "ndarray" in error["message"]
        assert arm["observations"]
        assert arm["outputs"] == {}
    assert_failed_bundle(root, result)


def assert_failed_bundle(root, result):
    findings = result["adjudication"]["findings"]
    assert "driver_error" in findings
    assert "equivalent_under_contract" not in findings
    assert read_verified_case(root)["adjudication"]["findings"] == findings
    assert check_case(root)["findings"] == findings
    suite = ET.fromstring(render_junit(root))
    assert suite.attrib["errors"] == "1"
    assert suite.find("testcase/error") is not None
    assert 'class="verdict error"' in render_report(root)
    completed = cli("verify", root)
    assert completed.returncode == 2, completed.stderr


@pytest.mark.parametrize("target", ["observation", "output"])
def test_plain_arrays_still_pass_all_five_arms(cases, target):
    root, result = cases[(target, "plain", "initial")]
    assert all(x["status"] == "complete" for x in result["arms"].values())
    assert result["adjudication"]["findings"] == ["equivalent_under_contract"]
    assert read_verified_case(root)["adjudication"]["findings"] == [
        "equivalent_under_contract"
    ]
    assert check_case(root)["findings"] == ["equivalent_under_contract"]
    assert cli("verify", root).returncode == 0


@pytest.mark.parametrize(
    "target,kind", [("observation", "masked"), ("output", "subclass")]
)
def test_cli_run_rejects_original_false_success_cases(tmp_path, target, kind):
    config = tmp_path / "study.json"
    config.write_text(json.dumps(study(target, kind)), encoding="utf-8")
    root = tmp_path / "case"
    completed = cli("run", config, "--cuts", "[4]", "--out", root)
    assert completed.returncode == 2, completed.stderr
    assert "driver_error" in json.loads(completed.stdout)["findings"]
    assert (
        check_case(root)["findings"]
        == read_verified_case(root)["adjudication"]["findings"]
    )


@pytest.mark.parametrize("kind", ["masked", "unmasked", "subclass"])
def test_numeric_writer_rejects_array_subclasses_before_creating_array(tmp_path, kind):
    from tests.helpers.observation_types import unsupported

    value = unsupported(np.array([-0.0, 1.0]), kind)
    with pytest.raises(IntegrityError, match="ndarray"):
        write_data(tmp_path / "data.json", {"x": value})
    assert not list(tmp_path.glob("*.npy"))
    assert not (tmp_path / "data.json").exists()


def supported_array(kind):
    if kind == "scalar":
        return np.array(-0.0)
    if kind == "empty":
        return np.empty((0, 2), dtype=np.int64)
    if kind == "strided":
        return np.array([-0.0, 99.0, 2.0, 99.0])[::2]
    if kind == "readonly":
        value = np.array([-0.0, 2.0])
        value.flags.writeable = False
        return value
    if kind == "nonnative":
        return np.array([-0.0, 2.0], dtype=np.dtype("f8").newbyteorder("S"))
    return np.asfortranarray([[-0.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize(
    "kind", ["scalar", "empty", "strided", "readonly", "nonnative", "fortran"]
)
def test_plain_observation_copy_preserves_shape_dtype_bytes_and_isolation(
    tmp_path, monkeypatch, kind
):
    value = supported_array(kind)
    original = value.tobytes()
    driver = SimpleNamespace(
        UNITS={"x": ""},
        create=lambda config, output: value,
        observe=lambda state: {"x": state},
        advance_one=lambda state: None,
        collect_outputs=lambda output: {},
    )
    monkeypatch.setattr("restartwitness.worker.load_driver", lambda name: driver)
    request = {
        "driver": "test",
        "config": {},
        "segment_dir": str(tmp_path / "segment"),
        "run_dir": str(tmp_path / "run"),
        "arm": "U1",
        "start_step": 0,
        "next_cut": 0,
        "cuts": [],
        "total_steps": 1,
        "observations": [0, 1],
    }
    result = run_segment(request)
    assert result["status"] == "complete"
    for snapshot in result["observations"]:
        copied = snapshot["fields"]["x"]
        assert type(copied) is np.ndarray
        assert copied is not value and not np.shares_memory(copied, value)
        assert copied.shape == value.shape and copied.dtype.str == value.dtype.str
        assert copied.tobytes() == original
    if value.flags.writeable:
        value[...] = 7
    assert result["observations"][0]["fields"]["x"].tobytes() == original
    stored = read_data(tmp_path / "segment/result.json")
    assert stored["observations"][0]["fields"]["x"].tobytes() == original
