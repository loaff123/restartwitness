import importlib.util
from pathlib import Path
import pytest


def api():
    assert importlib.util.find_spec("restartwitness.runner"), "runner missing"
    from restartwitness.runner import run_case

    return run_case


def study(**config):
    return {
        "driver": "tests.helpers.counter",
        "config": config,
        "total_steps": 4,
        "observations": [0, 1, 2, 3, 4],
        "contract": {
            "schema_version": 1,
            "fields": [
                {
                    "name": "step",
                    "dtype": "<i8",
                    "shape": [],
                    "unit": "",
                    "mode": "exact",
                    "atol": 0.0,
                    "rtol": 0.0,
                },
                {
                    "name": "x",
                    "dtype": "<f8",
                    "shape": [],
                    "unit": "",
                    "mode": "exact",
                    "atol": 0.0,
                    "rtol": 0.0,
                },
            ],
            "outputs": {},
        },
        "worker_timeout": 15,
    }


@pytest.mark.parametrize("cuts", [[], [0], [4], [2, 2], [0, 2, 4]])
def test_all_edge_cuts_and_fresh_processes(cuts, tmp_path):
    result = api()(study(), cuts, tmp_path / "case")
    assert result["adjudication"]["status"] == "equivalent_under_contract"
    r = result["arms"]["R"]
    assert len(r["segments"]) == len(cuts) + 1
    assert len({s["process_token"] for s in r["segments"]}) == len(cuts) + 1
    assert sum(s["advance_calls"] for s in r["segments"]) == 4
    for arm in ("U1", "U2", "P", "R"):
        assert sum(s["advance_calls"] for s in result["arms"][arm]["segments"]) == 4
    assert len(r["observations"]) == 5 + 3 * len(cuts)


@pytest.mark.parametrize(
    "config,status",
    [
        ({"mutate": True}, "observation_inconclusive"),
        ({"restore_fault": True}, "restore_path_difference"),
        ({"save_side_effect": True}, "save_path_difference"),
        ({"failed_save": True}, "checkpoint_error"),
        ({"failed_restore": True}, "checkpoint_error"),
        ({"empty": True}, "checkpoint_error"),
        ({"timeout": True}, "timeout"),
    ],
)
def test_failure_classifications(config, status, tmp_path):
    s = study(**config)
    s["worker_timeout"] = 0.5 if config.get("timeout") else 15
    result = api()(s, [4] if config.get("save_side_effect") else [2], tmp_path / "case")
    assert (
        status in result["adjudication"]["findings"]
        or status == result["adjudication"]["status"]
    )


def test_invalid_cuts_and_existing_or_symlink_destination(tmp_path):
    run = api()
    for cuts in ([3, 2], [-1], [5], [True]):
        with pytest.raises(ValueError):
            run(study(), cuts, tmp_path / "bad")
    (tmp_path / "existing").mkdir()
    with pytest.raises(ValueError):
        run(study(), [], tmp_path / "existing")
    (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        run(study(), [], tmp_path / "link" / "case")


def test_acknowledged_abrupt_mode_and_phase_order(tmp_path):
    s = study()
    s["termination"] = "exit_after_save"
    result = api()(s, [2, 2], tmp_path / "case")
    assert result["adjudication"]["status"] == "equivalent_under_contract"
    assert [x["exit_code"] for x in result["arms"]["R"]["segments"]] == [75, 75, 0]
    assert all(x["checkpoint_complete"] for x in result["arms"]["R"]["segments"][:-1])


def test_verified_case_rejects_coherently_omitted_observation(tmp_path):
    run = api()
    root = tmp_path / "case"
    run(study(), [2], root)
    from restartwitness.evidence import read_json, write_json, digest, IntegrityError
    from restartwitness.runner import read_verified_case

    for name in ("U1", "U2", "P", "R"):
        p = root / f"{name}.json"
        data = read_json(p)
        data["observations"] = [
            x for x in data["observations"] if x["key"] != [1, "ordinary", 0]
        ]
        write_json(p, data)
    manifest = read_json(root / "manifest.json")
    for artifact in manifest["artifacts"]:
        p = root / artifact["path"]
        artifact.update(bytes=p.stat().st_size, sha256=digest(p))
    write_json(root / "manifest.json", manifest)
    with pytest.raises(IntegrityError):
        read_verified_case(root)


def test_checkpoint_ack_hash_is_crosschecked_against_saved_artifact(tmp_path):
    run = api()
    root = tmp_path / "case"
    run(study(), [2], root)
    from restartwitness.evidence import read_json, write_json, digest, IntegrityError
    from restartwitness.runner import read_verified_case

    p = root / "R/segment-000/checkpoint-000/state.json"
    p.write_text(p.read_text() + " ")
    m = read_json(root / "manifest.json")
    for a in m["artifacts"]:
        f = root / a["path"]
        a.update(bytes=f.stat().st_size, sha256=digest(f))
    write_json(root / "manifest.json", m)
    with pytest.raises(IntegrityError):
        read_verified_case(root)


def test_source_changes_during_experiment_fail_closed(tmp_path, monkeypatch):
    run = api()
    helper = Path(__file__).parent / "helpers/counter.py"
    source = (
        helper.read_text()
        + "\n_original_save=save\ndef save(s,p):\n    _original_save(s,p)\n    q=Path(__file__);q.write_text(q.read_text()+'\\n# source changed\\n')\n"
    )
    (tmp_path / "changing_driver.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    s = study()
    s["driver"] = "changing_driver"
    result = run(s, [2], tmp_path / "case")
    assert "evidence_integrity_error" in result["adjudication"]["findings"]


@pytest.mark.parametrize(
    "mutation", ["missing_ack", "wrong_step", "missing_segment", "duplicate_process"]
)
def test_verified_case_requires_process_and_checkpoint_protocol(tmp_path, mutation):
    run = api()
    root = tmp_path / "case"
    run(study(), [2], root)
    from restartwitness.evidence import read_json, write_json, digest, IntegrityError
    from restartwitness.runner import read_verified_case

    p = root / "R.json"
    data = read_json(p)
    if mutation == "missing_ack":
        data["segments"][0]["checkpoints"] = []
    elif mutation == "wrong_step":
        data["segments"][0]["checkpoints"][0]["step"] = 3
    elif mutation == "missing_segment":
        data["segments"].pop()
    else:
        data["segments"][1]["process_token"] = data["segments"][0]["process_token"]
    write_json(p, data)
    m = read_json(root / "manifest.json")
    for a in m["artifacts"]:
        f = root / a["path"]
        a.update(bytes=f.stat().st_size, sha256=digest(f))
    write_json(root / "manifest.json", m)
    with pytest.raises(IntegrityError):
        read_verified_case(root)


@pytest.mark.parametrize("target", ["study", "request", "metadata"])
def test_verified_case_checks_config_and_request_identity(tmp_path, target):
    run = api()
    root = tmp_path / "case"
    run(study(), [2], root)
    from restartwitness.evidence import read_json, write_json, digest, IntegrityError
    from restartwitness.runner import read_verified_case

    m = read_json(root / "manifest.json")
    if target == "metadata":
        m["metadata"]["identity"]["driver"] = "different.driver"
    else:
        p = root / ("study.json" if target == "study" else "R/request-000.json")
        data = read_json(p)
        data["config"] = {"restore_fault": True}
        write_json(p, data)
    for a in m["artifacts"]:
        f = root / a["path"]
        a.update(bytes=f.stat().st_size, sha256=digest(f))
    write_json(root / "manifest.json", m)
    with pytest.raises(IntegrityError):
        read_verified_case(root)


def test_aggregate_observations_cannot_contradict_segment_raw_data(tmp_path):
    run = api()
    root = tmp_path / "case"
    run(study(restore_fault=True), [2], root)
    from restartwitness.evidence import (
        read_json,
        read_data,
        write_json,
        digest,
        IntegrityError,
    )
    from restartwitness.runner import read_verified_case
    from restartwitness.compare import adjudicate_arms
    from restartwitness.contracts import Contract

    r = read_json(root / "R.json")
    r["observations"] = read_json(root / "P.json")["observations"]
    write_json(root / "R.json", r)
    arms = {n: read_data(root / f"{n}.json") for n in ("U1", "U2", "O", "P", "R")}
    write_json(
        root / "adjudication.json",
        adjudicate_arms(arms, Contract.from_dict(study()["contract"])),
    )
    m = read_json(root / "manifest.json")
    for a in m["artifacts"]:
        f = root / a["path"]
        a.update(bytes=f.stat().st_size, sha256=digest(f))
    write_json(root / "manifest.json", m)
    with pytest.raises(IntegrityError):
        read_verified_case(root)


def test_reported_work_counts_recomputed_from_segment_ledger(tmp_path):
    run = api()
    root = tmp_path / "case"
    run(study(), [2], root)
    from restartwitness.evidence import read_json, write_json, digest, IntegrityError
    from restartwitness.runner import read_verified_case

    p = root / "metrics.json"
    data = read_json(p)
    data["simulated_steps"] = 999
    write_json(p, data)
    m = read_json(root / "manifest.json")
    for a in m["artifacts"]:
        f = root / a["path"]
        a.update(bytes=f.stat().st_size, sha256=digest(f))
    write_json(root / "manifest.json", m)
    with pytest.raises(IntegrityError):
        read_verified_case(root)


@pytest.mark.parametrize("bad_value", [None, 42, []])
def test_malformed_arm_schema_is_an_integrity_error(tmp_path, bad_value):
    run = api()
    root = tmp_path / "case"
    run(study(), [2], root)
    from restartwitness.evidence import read_json, write_json, digest, IntegrityError
    from restartwitness.runner import read_verified_case

    write_json(root / "R.json", bad_value)
    m = read_json(root / "manifest.json")
    for a in m["artifacts"]:
        f = root / a["path"]
        a.update(bytes=f.stat().st_size, sha256=digest(f))
    write_json(root / "manifest.json", m)
    with pytest.raises(IntegrityError):
        read_verified_case(root)
