import importlib.util
import json
from pathlib import Path
import pytest


def examples():
    assert importlib.util.find_spec("restartwitness.examples"), (
        "example study builder missing"
    )
    from restartwitness.examples import example_study

    return example_study


@pytest.mark.parametrize(
    "model",
    ["integrator", "stochastic", "events", "rebound", "openmm", "openmm-state-misuse"],
)
def test_examples_are_complete_fixed_contracts(model):
    from restartwitness.runner import validate_study

    s = examples()(model)
    assert s["observations"] == list(range(65))
    assert validate_study(s)["total_steps"] == 64
    assert s["contract"]["fields"][0]["name"] == "step"


def test_frozen_protocol_covers_faults_controls_and_exact_recipes():
    p = Path(__file__).parents[1] / "protocols/v1.json"
    assert p.exists(), "frozen protocol absent"
    spec = json.loads(p.read_text())
    assert len(spec["faults"]) == 12
    assert len({x["id"] for x in spec["faults"]}) == 12
    assert spec["schedule_recipe"] == [
        [0],
        [1],
        [7],
        [8],
        [9],
        [31],
        [32],
        [33],
        [63],
        [64],
        [1, 32, 63],
        [7, 8, 9],
        [8, 8],
    ]
    for f in spec["faults"]:
        assert f["applicability"] and f["expected_finding"]
        assert f["study"]["config"]["fault"] == f["id"]
    assert set(spec["evaluation"]["equal_budget_methods"]) == {
        "fixed",
        "random",
        "boundary",
    }


def test_event_example_before_first_output_expects_no_table(tmp_path):
    from restartwitness.runner import run_case

    s = examples()("events", total_steps=4)
    assert s["contract"]["outputs"] == {}
    assert (
        run_case(s, [2], tmp_path / "case")["adjudication"]["status"]
        == "equivalent_under_contract"
    )
