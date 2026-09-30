"""Contract schemas must reject ambiguity rather than coerce evidence."""

import importlib
import json
from dataclasses import FrozenInstanceError

import pytest


def api():
    assert importlib.util.find_spec("restartwitness.contracts") is not None, (
        "contract implementation is required"
    )
    from restartwitness.contracts import Contract, FieldContract, RecordContract

    return Contract, FieldContract, RecordContract


def example():
    Contract, FieldContract, RecordContract = api()
    field = FieldContract("position", "float64", (2,), unit="m")
    record = RecordContract("samples", ("step:0", "step:2"), (field,))
    return Contract((field,), {"samples": record})


def test_roundtrip_contract_is_strict_json_and_immutable():
    Contract, _, _ = api()
    contract = example()
    encoded = json.loads(json.dumps(contract.to_dict(), allow_nan=False))
    assert encoded["schema_version"] == 1
    assert encoded["fields"][0]["shape"] == [2]
    assert encoded["outputs"]["samples"]["expected_keys"] == ["step:0", "step:2"]
    assert Contract.from_dict(encoded) == contract
    with pytest.raises(FrozenInstanceError):
        contract.fields = ()
    with pytest.raises(TypeError):
        contract.outputs["new"] = contract.outputs["samples"]
    with pytest.raises(FrozenInstanceError):
        contract.fields[0].atol = 1


def test_contract_copies_input_output_mapping():
    Contract, FieldContract, RecordContract = api()
    field = FieldContract("x", "float64", ())
    outputs = {"rows": RecordContract("rows", (), ())}
    contract = Contract((field,), outputs)
    outputs.clear()
    assert "rows" in contract.outputs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"name": 1},
        {"dtype": "object"},
        {"dtype": "str"},
        {"dtype": "complex128"},
        {"dtype": "datetime64[ns]"},
        {"dtype": "not-a-dtype"},
        {"dtype": 64},
        {"shape": [1]},
        {"shape": (-1,)},
        {"shape": (True,)},
        {"shape": (1.0,)},
        {"unit": None},
        {"mode": "relative"},
        {"atol": -1},
        {"rtol": float("inf")},
        {"atol": float("nan")},
        {"atol": "0"},
        {"rtol": True},
    ],
)
def test_rejects_invalid_field_contract(kwargs):
    _, FieldContract, _ = api()
    args = dict(name="x", dtype="float64", shape=())
    args.update(kwargs)
    with pytest.raises(ValueError):
        FieldContract(**args)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(extra=True),
        lambda d: d.update(schema_version=True),
        lambda d: d.update(schema_version=2),
        lambda d: d.pop("fields"),
        lambda d: d["fields"][0].update(extra=1),
        lambda d: d["fields"][0].pop("unit"),
        lambda d: d["fields"][0].update(shape="2"),
        lambda d: d["outputs"]["samples"].update(extra=1),
        lambda d: d["outputs"]["samples"].update(expected_keys=[1, 2]),
        lambda d: d["outputs"]["samples"].update(name="other"),
    ],
)
def test_from_dict_does_not_accept_unknown_missing_or_coerced_fields(mutation):
    Contract, _, _ = api()
    data = example().to_dict()
    mutation(data)
    with pytest.raises(ValueError):
        Contract.from_dict(data)


def test_duplicate_field_names_and_record_keys_rejected():
    Contract, FieldContract, RecordContract = api()
    field = FieldContract("x", "int64", ())
    with pytest.raises(ValueError):
        Contract((field, field))
    with pytest.raises(ValueError):
        RecordContract("rows", ("step:1", "step:1"), (field,))
    with pytest.raises(ValueError):
        RecordContract("rows", ("step:1",), (field, field))


def test_empty_or_mutable_field_collection_rejected():
    Contract, FieldContract, _ = api()
    field = FieldContract("x", "int64", ())
    with pytest.raises(ValueError):
        Contract(())
    with pytest.raises(ValueError):
        Contract([field])


def test_output_table_names_must_match_contract_names():
    Contract, FieldContract, RecordContract = api()
    field = FieldContract("x", "float64", ())
    with pytest.raises(ValueError):
        Contract((field,), {"a": RecordContract("b", (), ())})
