"""Offline report behavior is checked through real, reverified bundles."""

import importlib.util


import copy
from html.parser import HTMLParser
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from restartwitness.compare import adjudicate_arms
from restartwitness.contracts import Contract
from restartwitness.evidence import (
    IntegrityError,
    digest,
    read_data,
    read_json,
    seal_bundle,
    write_data,
    write_json,
)
from restartwitness.runner import read_verified_case, run_case
from tests.test_runner import study


def report_api():
    assert importlib.util.find_spec("restartwitness.report"), (
        "offline report module is missing"
    )
    from restartwitness.report import render_junit, render_report

    return render_report, render_junit


def test_report_api_exists():
    html_report, junit_report = report_api()
    assert callable(html_report) and callable(junit_report)


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("report-evidence")
    cases = {}
    for name, config in [
        ("clean", {}),
        ("restore", {"restore_fault": True}),
        ("observer", {"mutate": True}),
    ]:
        path = root / name
        run_case(study(**config), [2, 2], path)
        cases[name] = path
    return cases


def write_case(root, case, source):
    """Create fresh numeric artifacts; cached results are derived from the arrays."""
    import shutil

    root.mkdir()
    for name, arm in case["arms"].items():
        shutil.copytree(source / name, root / name)
        for request_path in (root / name).glob("request-*.json"):
            request = read_json(request_path)
            request.update(case["study"])
            write_json(request_path, request)
        offset = 0
        for index, segment in enumerate(arm["segments"]):
            result_path = root / name / f"segment-{index:03}" / "result.json"
            raw = read_data(result_path)
            count = len(raw["observations"])
            raw["observations"] = arm["observations"][offset : offset + count]
            offset += count
            if index == len(arm["segments"]) - 1:
                raw["outputs"] = arm["outputs"]
                raw["status"] = arm["status"]
                segment["status"] = arm["status"]
                if "error" in arm:
                    raw["error"] = segment["error"] = arm["error"]
            result_path.unlink()
            for array_path in result_path.parent.glob("result-*.npy"):
                array_path.unlink()
            write_data(result_path, raw)
        assert offset == len(arm["observations"])
    write_json(root / "study.json", case["study"])
    write_json(root / "schedule.json", case["schedule"])
    for name, arm in case["arms"].items():
        write_data(root / f"{name}.json", arm)
    case["adjudication"] = adjudicate_arms(
        case["arms"], Contract.from_dict(case["study"]["contract"])
    )
    write_json(root / "adjudication.json", case["adjudication"])
    write_json(root / "metrics.json", case["metrics"])
    seal_bundle(
        root,
        {
            "kind": "restartwitness-case",
            "identity": case["identity"],
            "source_status": "development",
            "termination": case["study"]["termination"],
            "study_sha256": digest(root / "study.json"),
            "schedule_sha256": digest(root / "schedule.json"),
        },
    )
    return root


class PageInspection(HTMLParser):
    def __init__(self, page):
        super().__init__()
        self.tags = []
        self.links = []
        self.ids = []
        self.text = []
        self.feed(page)
        self.close()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.tags.append((tag, attrs))
        if "id" in attrs:
            self.ids.append(attrs["id"])
        for attribute in ("href", "src"):
            if attribute in attrs:
                self.links.append(attrs[attribute])

    def handle_data(self, data):
        self.text.append(data)


def test_html_is_offline_accessible_and_shows_reverified_evidence(cases):
    render, _ = report_api()
    case = read_verified_case(cases["clean"])
    html = render(cases["clean"])
    page = PageInspection(html)
    content = " ".join(page.text)
    assert "<!doctype html>" in html.lower()
    assert "RestartWitness" in content and "Developer preview" in content
    assert "equivalent_under_contract" in content
    assert "U-P" in content and "P-R" in content and "U-R" in content
    assert "U1-U2" in content and "U1-O" in content
    assert "No upstream bug claim" in content
    assert case["manifest_sha256"] in content
    assert case["identity"]["driver_sha256"] in content
    assert "2, 2" in content
    assert "simulated steps" in content.lower() and "process starts" in content.lower()
    assert "unknown work segments" in content.lower()
    assert "--trust-driver" in content
    assert str(cases["clean"]) not in html
    assert "<script" not in html.lower()
    assert all(link.startswith("#") for link in page.links)
    assert all(not key.startswith("on") for _, attrs in page.tags for key in attrs)
    assert len(page.ids) == len(set(page.ids))
    assert any(tag == "html" and attrs.get("lang") == "en" for tag, attrs in page.tags)
    assert any(
        tag == "meta" and attrs.get("name") == "viewport" for tag, attrs in page.tags
    )
    assert any(
        tag == "svg" and attrs.get("role") == "img" and attrs.get("aria-labelledby")
        for tag, attrs in page.tags
    )
    series = [
        attrs for tag, attrs in page.tags if tag == "polyline" and attrs.get("data-arm")
    ]
    assert {p["data-arm"] for p in series} == {"U", "P", "R"}
    assert len({p.get("stroke-dasharray", "solid") for p in series}) == 3
    assert all(p.get("points") for p in series)
    assert len(html) < 100_000


def test_difference_report_shows_exact_first_witness(cases):
    render, _ = report_api()
    result = read_verified_case(cases["restore"])
    witness = result["adjudication"]["comparisons"]["P-R"]["first_witness"]
    html = render(cases["restore"])
    text = " ".join(PageInspection(html).text)
    assert "restore_path_difference" in text
    assert "after_restore" in text
    assert "Field" in text and witness["field"] in text
    assert "Index" in text and "[]" in text
    assert str(witness["left"]) in text and str(witness["right"]) in text
    assert witness["left_bits"] in text and witness["right_bits"] in text
    assert "Absolute difference" in text and "Relative difference" in text
    assert "not a root-cause proof" in text


def test_junit_equivalent_failure_and_inconclusive(cases):
    _, render = report_api()
    expected = {"clean": None, "restore": "failure", "observer": "skipped"}
    for name, outcome in expected.items():
        suite = ET.fromstring(render(cases[name]))
        assert suite.tag == "testsuite" and suite.attrib["tests"] == "1"
        assert suite.attrib["failures"] == str(int(outcome == "failure"))
        assert suite.attrib["errors"] == "0"
        assert suite.attrib["skipped"] == str(int(outcome == "skipped"))
        case = suite.find("testcase")
        assert case is not None
        assert all(
            case.find(tag) is None
            for tag in {"failure", "error", "skipped"} - {outcome}
        )
        if outcome:
            assert case.find(outcome) is not None
        assert "U-P" in case.findtext("system-out")


@pytest.mark.parametrize(
    "status,tag",
    [
        ("timeout", "error"),
        ("driver_error", "error"),
        ("checkpoint_error", "error"),
        ("unsupported", "skipped"),
        ("incomplete", "error"),
        ("evidence_integrity_error", "error"),
    ],
)
def test_junit_nonpassing_worker_outcomes(cases, tmp_path, status, tag):
    _, render = report_api()
    case = copy.deepcopy(read_verified_case(cases["clean"]))
    case["arms"]["R"].update(
        status=status, error={"phase": "restore", "message": 'failure <&> "quoted"\x00'}
    )
    root = write_case(tmp_path / status, case, cases["clean"])
    suite = ET.fromstring(render(root))
    assert suite.find(f"testcase/{tag}") is not None
    assert 'failure <&> "quoted"' in ET.tostring(suite, encoding="unicode").replace(
        "&lt;", "<"
    ).replace("&gt;", ">").replace("&amp;", "&")


def test_html_and_xml_escape_untrusted_field_names_config_and_errors(cases, tmp_path):
    render, junit = report_api()
    case = copy.deepcopy(read_verified_case(cases["restore"]))
    hostile = '</script><img src=x onerror="alert(1)"> & scientific-field'
    case["study"]["config"]["description"] = '<script>alert("config")</script>'
    case["study"]["config"]["local_path"] = "/home/private-person/sensitive-file"
    field = next(f for f in case["study"]["contract"]["fields"] if f["name"] == "x")
    field["name"] = hostile
    for arm in case["arms"].values():
        for obs in arm["observations"]:
            obs["fields"][hostile] = obs["fields"].pop("x")
            obs["units"][hostile] = obs["units"].pop("x")
    root = write_case(tmp_path / "hostile", case, cases["restore"])
    html = render(root)
    page = PageInspection(html)
    assert all(tag not in ("script", "img") for tag, _ in page.tags)
    assert "&lt;script&gt;" in html and "&lt;img" in html
    assert hostile in "".join(page.text)
    assert "/home/private-person" not in html
    xml = ET.fromstring(junit(root))
    assert hostile in xml.findtext("testcase/failure")


def test_record_identities_units_and_tolerances_are_visible(cases, tmp_path):
    render, junit = report_api()
    case = copy.deepcopy(read_verified_case(cases["clean"]))
    record_field = {
        "name": "energy",
        "dtype": "<f8",
        "shape": [],
        "unit": "J",
        "mode": "tolerance",
        "atol": 0.01,
        "rtol": 0.001,
    }
    case["study"]["contract"]["outputs"]["energy_log"] = {
        "name": "energy_log",
        "expected_keys": ["sample:0", "sample:1"],
        "fields": [record_field],
    }
    for name, arm in case["arms"].items():
        arm["outputs"]["energy_log"] = {
            "keys": ["sample:0", "sample:1"],
            "fields": {"energy": np.array([1.0, 2.0])},
            "units": {"energy": "J"},
        }
    case["arms"]["R"]["outputs"]["energy_log"]["keys"] = ["sample:0", "sample:0"]
    root = write_case(tmp_path / "records", case, cases["clean"])
    html = render(root)
    content = " ".join(PageInspection(html).text)
    for item in [
        "energy_log",
        "sample:0",
        "sample:1",
        "duplicate_record_keys",
        "output_contract_violation",
        "J",
        "0.01",
        "0.001",
    ]:
        assert item in content
    assert ET.fromstring(junit(root)).find("testcase/failure") is not None


@pytest.mark.parametrize("renderer_index", [0, 1])
def test_renderer_rejects_tampered_evidence_and_unverified_dict(
    cases, tmp_path, renderer_index
):
    import shutil

    render = report_api()[renderer_index]
    with pytest.raises(TypeError):
        render(read_verified_case(cases["clean"]))
    root = tmp_path / "tampered"
    shutil.copytree(cases["clean"], root)
    (root / "adjudication.json").write_text('{"status":"equivalent_under_contract"}')
    with pytest.raises(IntegrityError):
        render(root)
    # Rehashing a stale adjudication still cannot replace raw-array adjudication.
    manifest = read_json(root / "manifest.json")
    for item in manifest["artifacts"]:
        path = root / item["path"]
        item.update(bytes=path.stat().st_size, sha256=digest(path))
    write_json(root / "manifest.json", manifest)
    with pytest.raises(IntegrityError):
        render(root)


def test_plot_can_render_first_position_component(cases, tmp_path):
    render, _ = report_api()
    case = copy.deepcopy(read_verified_case(cases["clean"]))
    field = next(f for f in case["study"]["contract"]["fields"] if f["name"] == "x")
    field.update(name="position", shape=[1, 1], unit="nm")
    for arm in case["arms"].values():
        for obs in arm["observations"]:
            obs["fields"]["position"] = obs["fields"].pop("x").reshape(1, 1)
            obs["units"].pop("x")
            obs["units"]["position"] = "nm"
    root = write_case(tmp_path / "position", case, cases["clean"])
    html = render(root)
    page = PageInspection(html)
    assert "position[0, …]" in "".join(page.text)
    assert (
        len(
            [1 for tag, attrs in page.tags if tag == "polyline" and "data-arm" in attrs]
        )
        == 3
    )


def test_final_step_save_difference_is_not_concealed_by_ordinary_plot(tmp_path):
    render, junit = report_api()
    root = tmp_path / "final-cut"
    run_case(study(save_side_effect=True), [4], root)
    html = render(root)
    content = " ".join(PageInspection(html).text)
    assert "save_path_difference" in content
    assert '[4, "after_save", 0]' in content
    assert "ordinary observations only" in content
    assert ET.fromstring(junit(root)).find("testcase/failure") is not None


def test_real_budget_timeout_remains_error_and_displays_partial_work(tmp_path):
    render, junit = report_api()
    root = tmp_path / "timeout"
    config = study(timeout=True)
    config.update(worker_timeout=0.1, study_budget=0.2)
    run_case(config, [2], root)
    assert ET.fromstring(junit(root)).find("testcase/error") is not None
    html = render(root)
    content = " ".join(PageInspection(html).text)
    assert "timeout" in content
    assert "Unknown work segments" in content
    assert "cannot establish equivalence" in content


def test_nondeterministic_baseline_is_skipped_even_with_restart_differences(
    cases, tmp_path
):
    _, junit = report_api()
    case = copy.deepcopy(read_verified_case(cases["restore"]))
    for observation in case["arms"]["U2"]["observations"]:
        observation["fields"]["x"] += 1.0
    root = write_case(tmp_path / "baseline-inconclusive", case, cases["restore"])
    suite = ET.fromstring(junit(root))
    assert suite.find("testcase/skipped") is not None
    assert "baseline_inconclusive" in suite.findtext("testcase/skipped")
    assert suite.attrib["failures"] == "0"
