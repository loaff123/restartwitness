"""Small full-harness controlled corpus acceptance; protocol stays immutable."""

import json
from pathlib import Path
import pytest
from restartwitness.runner import run_case, read_verified_case

PROTOCOL = json.loads((Path(__file__).parents[1] / "protocols/v1.json").read_text())


@pytest.mark.parametrize("case", PROTOCOL["faults"], ids=lambda x: x["id"])
def test_all_twelve_frozen_applicable_faults(case, tmp_path):
    result = run_case(case["study"], case["applicability"][0], tmp_path / "case")
    assert case["expected_finding"] in result["adjudication"]["findings"]
    assert (
        read_verified_case(tmp_path / "case")["adjudication"] == result["adjudication"]
    )


@pytest.mark.parametrize("case", PROTOCOL["controls"], ids=lambda x: x["id"])
def test_negative_and_inconclusive_controls(case, tmp_path):
    result = run_case(case["study"], case["schedule"], tmp_path / "case")
    assert case["expected_finding"] in result["adjudication"]["findings"]
