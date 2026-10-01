import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def cli(*args):
    assert importlib.util.find_spec("restartwitness.cli"), "CLI missing"
    return subprocess.run(
        [sys.executable, "-m", "restartwitness", *map(str, args)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[1] / "src")},
        timeout=60,
    )


def test_example_run_verify_and_trusted_replay(tmp_path):
    config = tmp_path / "study.json"
    r = cli("example", "--model", "integrator", "--steps", "4", "--out", config)
    assert r.returncode == 0, r.stderr
    root = tmp_path / "case"
    r = cli("run", config, "--cuts", "[2,2]", "--out", root)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "equivalent_under_contract"
    r = cli("verify", root)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "equivalent_under_contract"
    r = cli("replay", root, "--out", tmp_path / "replayed")
    assert r.returncode == 2
    r = cli("replay", root, "--out", tmp_path / "replayed", "--trust-driver")
    assert r.returncode == 0, r.stderr


def test_example_refuses_overwrite_and_invalid_cut_json(tmp_path):
    config = tmp_path / "study.json"
    config.write_text("old")
    assert cli("example", "--out", config).returncode == 2
    assert config.read_text() == "old"
    assert (
        cli("run", config, "--cuts", "[true]", "--out", tmp_path / "bad").returncode
        == 2
    )


def test_difference_exit_status_and_report(tmp_path):
    config = tmp_path / "study.json"
    assert (
        cli(
            "example", "--fault", "integrator_cache", "--steps", "4", "--out", config
        ).returncode
        == 0
    )
    root = tmp_path / "case"
    r = cli("run", config, "--cuts", "[2]", "--out", root)
    assert r.returncode == 1, r.stderr
    assert json.loads(r.stdout)["status"] == "restore_path_difference"
    r = cli("report", root, "--out", tmp_path / "report.html")
    assert r.returncode == 0, r.stderr
    assert "RestartWitness" in (tmp_path / "report.html").read_text()


def test_default_cut_is_midpoint_of_requested_horizon(tmp_path):
    config = tmp_path / "study.json"
    assert cli("example", "--steps", "4", "--out", config).returncode == 0
    result = cli("run", config, "--out", tmp_path / "case")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["schedule"] == [2]


def test_mixed_valid_difference_and_invalid_evidence_exit_is_error():
    from restartwitness.cli import _exit
    from restartwitness.compare import adjudicate_arms
    from tests.test_compare import arms, contract

    result = adjudicate_arms(arms({"P": 1.0, "R": float("nan")}), contract())
    assert result["findings"] == ["save_path_difference", "invalid"]
    assert _exit({"adjudication": result}) == 2
