"""Read/replay commands must leave their sealed input evidence untouched."""

import json
from pathlib import Path
import shutil

import pytest

from restartwitness.cli import main
from restartwitness.evidence import digest
from restartwitness.examples import example_study
from restartwitness.runner import read_verified_case, replay, run_case
from tests.independent_check import check_case


@pytest.fixture(scope="module")
def input_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("preservation-inputs")
    cases = {}
    for fault in ("clean", "integrator_cache"):
        path = root / fault
        run_case(example_study(fault=fault, total_steps=2), [1], path)
        cases[fault] = path
    return cases


def inventory(root):
    return {
        p.relative_to(root).as_posix(): digest(p) if p.is_file() else None
        for p in root.rglob("*")
    }


def command_args(command, root, output):
    args = [command, str(root), "--out", str(output)]
    if command in ("replay", "reduce"):
        args += ["--trust-driver"]
    if command == "reduce":
        args += ["--trials", "2"]
    return args


@pytest.mark.parametrize("command", ["report", "replay", "reduce"])
@pytest.mark.parametrize("path_form", ["nested", "parent_components", "relative"])
def test_cli_rejects_output_inside_input_without_changes(
    input_cases, tmp_path, capsys, monkeypatch, command, path_form
):
    root = tmp_path / "case"
    shutil.copytree(input_cases["integrator_cache"], root)
    before = inventory(root)
    source = root
    output = root / "new" / "output"
    if path_form == "parent_components":
        source = root / ".." / "case"
        output = root / "missing" / ".." / "output"
    elif path_form == "relative":
        monkeypatch.chdir(tmp_path)
        source = Path("case")
        output = Path("case/new/output")
    assert main(command_args(command, source, output)) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "invalid_request"
    assert "outside" in error["error"] and "bundle" in error["error"]
    assert inventory(root) == before
    assert (
        read_verified_case(root)["adjudication"]["status"] == "restore_path_difference"
    )
    assert check_case(root)["status"] == "restore_path_difference"


@pytest.mark.parametrize("form", ["nested", "same_root", "parent_components"])
def test_python_replay_rejects_input_descendant_without_changes(
    input_cases, tmp_path, form
):
    root = tmp_path / "case"
    shutil.copytree(input_cases["clean"], root)
    before = inventory(root)
    output = {
        "nested": root / "new" / "replay",
        "same_root": root,
        "parent_components": root / "missing" / ".." / "replay",
    }[form]
    with pytest.raises(ValueError, match="outside.*bundle"):
        replay(root, output, trust_driver=True)
    assert inventory(root) == before
    assert (
        read_verified_case(root)["adjudication"]["status"]
        == "equivalent_under_contract"
    )
    assert check_case(root)["status"] == "equivalent_under_contract"


@pytest.mark.parametrize("format", ["html", "junit"])
def test_report_formats_reject_input_descendant(input_cases, tmp_path, format):
    root = tmp_path / "case"
    shutil.copytree(input_cases["clean"], root)
    before = inventory(root)
    assert (
        main(
            ["report", str(root), "--format", format, "--out", str(root / "new-report")]
        )
        == 2
    )
    assert inventory(root) == before
    assert check_case(root)["status"] == "equivalent_under_contract"


@pytest.mark.parametrize("command", ["report", "replay", "reduce"])
def test_sibling_output_preserves_input_bundle(input_cases, tmp_path, command):
    root = tmp_path / "case"
    shutil.copytree(input_cases["integrator_cache"], root)
    before = inventory(root)
    output = tmp_path / "case-results"
    assert main(command_args(command, root, output)) == (
        1 if command == "replay" else 0
    )
    assert output.exists()
    assert inventory(root) == before
    assert (
        read_verified_case(root)["adjudication"]["status"] == "restore_path_difference"
    )
    assert check_case(root)["status"] == "restore_path_difference"
    if command == "replay":
        assert check_case(output)["status"] == "restore_path_difference"
    elif command == "reduce":
        assert json.loads((output / "reduction.json").read_text())["target_reproduced"]


@pytest.mark.parametrize("command", ["report", "replay", "reduce"])
def test_output_alias_does_not_change_input(input_cases, tmp_path, command):
    root = tmp_path / "case"
    shutil.copytree(input_cases["integrator_cache"], root)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    before = inventory(root)
    assert main(command_args(command, root, alias / "new-output")) == 2
    assert inventory(root) == before
    assert check_case(root)["status"] == "restore_path_difference"


@pytest.mark.parametrize("command", ["report", "replay", "reduce", "replay-api"])
def test_output_symlink_loop_is_an_invalid_request(
    input_cases, tmp_path, capsys, command
):
    root = tmp_path / "case"
    shutil.copytree(input_cases["integrator_cache"], root)
    loop = tmp_path / "loop"
    try:
        loop.symlink_to(loop, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")
    before = inventory(root)
    if command == "replay-api":
        with pytest.raises(ValueError):
            replay(root, loop / "new-output", trust_driver=True)
    else:
        assert main(command_args(command, root, loop / "new-output")) == 2
        assert json.loads(capsys.readouterr().err)["status"] == "invalid_request"
    assert loop.is_symlink()
    assert inventory(root) == before
    assert check_case(root)["status"] == "restore_path_difference"
