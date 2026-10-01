"""Scientific continuation comparisons, including deliberately adverse inputs."""

import copy
import importlib
import json

import numpy as np
import pytest

from restartwitness.contracts import Contract, FieldContract, RecordContract


def api():
    assert importlib.util.find_spec("restartwitness.compare") is not None, (
        "comparator implementation is required"
    )
    return importlib.import_module("restartwitness.compare")


def contract(dtype="float64", shape=(), mode="exact", atol=0.0, rtol=0.0, outputs=None):
    return Contract(
        (FieldContract("x", dtype, shape, "m", mode, atol, rtol),), outputs or {}
    )


def snapshot(
    value=0.0, step=0, phase="ordinary", occurrence=0, dtype="float64", unit="m"
):
    return {
        "key": [step, phase, occurrence],
        "fields": {"x": np.asarray(value, dtype=dtype)},
        "units": {"x": unit},
    }


def records(keys=("step:0", "step:2"), values=(1.0, 2.0), dtype="float64"):
    return {
        "keys": list(keys),
        "fields": {"x": np.asarray(values, dtype=dtype)},
        "units": {"x": "m"},
    }


def record_contract(keys=("step:0", "step:2"), shape=()):
    return RecordContract("samples", keys, (FieldContract("x", "float64", shape, "m"),))


def arms(values=None):
    values = values or {}
    return {
        name: {
            "status": "complete",
            "observations": [snapshot(values.get(name, 0.0), 0)],
            "outputs": {},
        }
        for name in ("U1", "U2", "O", "P", "R")
    }


@pytest.mark.parametrize(
    "dtype,value",
    [
        ("bool", True),
        ("int64", 2**60 + 1),
        ("uint64", 2**64 - 1),
        ("float32", 1.5),
        ("float64", 1.5),
    ],
)
def test_exact_numeric_identity(dtype, value):
    result = api().compare_snapshots(
        snapshot(value, dtype=dtype), snapshot(value, dtype=dtype), contract(dtype)
    )
    assert result["status"] == "equivalent"
    assert result["first_witness"] is None
    assert result["diagnostics"] == []


@pytest.mark.parametrize(
    "dtype,left,right",
    [
        ("bool", True, False),
        ("int64", 2**60 + 1, 2**60 + 2),
        ("uint64", 2**64 - 1, 2**64 - 2),
    ],
)
def test_integers_and_booleans_never_use_float_tolerance(dtype, left, right):
    result = api().compare_snapshots(
        snapshot(left, dtype=dtype),
        snapshot(right, dtype=dtype),
        contract(dtype, mode="tolerance", atol=1e30),
    )
    assert result["status"] == "difference"
    assert result["first_witness"]["left"] == left
    assert result["first_witness"]["right"] == right
    assert result["first_witness"]["absolute_difference"] == 1


def test_exact_signed_zero_differs_but_tolerance_signed_zero_matches():
    left, right = snapshot(0.0), snapshot(-0.0)
    result = api().compare_snapshots(left, right, contract())
    assert result["status"] == "difference"
    assert result["first_witness"]["absolute_difference"] == 0
    assert result["first_witness"]["left_bits"] != result["first_witness"]["right_bits"]
    assert (
        api().compare_snapshots(left, right, contract(mode="tolerance"))["status"]
        == "equivalent"
    )


@pytest.mark.parametrize(
    "left,right,rtol,atol,status",
    [
        (9.0, 10.0, 0.1, 0.0, "equivalent"),
        (9.0, 10.0, 0.099, 0.0, "difference"),
        (0.0, 1.0, 0.0, 1.0, "equivalent"),
        (0.0, 1.0, 0.0, 0.999, "difference"),
        (-1e308, 1e308, 1.9, 0.0, "difference"),
        (-1e308, 1e308, 2.0, 0.0, "equivalent"),
    ],
)
def test_tolerance_bound_is_symmetric_and_avoids_overflow(
    left, right, rtol, atol, status
):
    c = contract(mode="tolerance", rtol=rtol, atol=atol)
    forward = api().compare_snapshots(snapshot(left), snapshot(right), c)
    reverse = api().compare_snapshots(snapshot(right), snapshot(left), c)
    assert forward["status"] == reverse["status"] == status
    json.dumps(forward, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_values_are_invalid_even_when_both_sides_identical(value):
    result = api().compare_snapshots(snapshot(value), snapshot(value), contract())
    assert result["status"] == "invalid"
    assert result["first_witness"]["kind"] == "nonfinite"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "mutation,kind",
    [
        (
            lambda s: s["fields"].update(x=np.array(0, dtype="float32")),
            "dtype_mismatch",
        ),
        (lambda s: s["fields"].update(x=np.array([0.0])), "shape_mismatch"),
        (lambda s: s["units"].update(x="cm"), "unit_mismatch"),
        (lambda s: s["fields"].update(y=np.array(0.0)), "field_names_mismatch"),
        (lambda s: s["fields"].pop("x"), "field_names_mismatch"),
        (lambda s: s["units"].clear(), "unit_names_mismatch"),
        (lambda s: s["fields"].update(x=0.0), "not_numeric_array"),
        (lambda s: s["fields"].update(x=np.array(0, dtype=object)), "dtype_mismatch"),
    ],
)
def test_schema_mismatches_are_invalid_without_broadcasting_or_coercion(mutation, kind):
    right = snapshot()
    mutation(right)
    result = api().compare_snapshots(snapshot(), right, contract())
    assert result["status"] == "invalid"
    assert kind in [d["kind"] for d in result["diagnostics"]]


def test_first_array_index_and_values_are_reported_in_contract_order():
    result = api().compare_snapshots(
        snapshot([[1, 2], [3, 4]]), snapshot([[1, 9], [8, 4]]), contract(shape=(2, 2))
    )
    witness = result["first_witness"]
    assert witness["field"] == "x"
    assert witness["index"] == [0, 1]
    assert witness["left"] == 2.0
    assert witness["right"] == 9.0
    assert witness["absolute_difference"] == 7.0
    assert witness["relative_difference"] == pytest.approx(7 / 9)
    assert witness["mismatch_count"] == 2
    assert witness["key"] == [0, "ordinary", 0]
    assert witness["contract"] == contract(shape=(2, 2)).fields[0].to_dict()


def test_noncontiguous_exact_arrays_are_compared_logically():
    value = np.arange(12, dtype="float64").reshape(3, 4).T
    left, right = snapshot(value), snapshot(value.copy())
    assert (
        api().compare_snapshots(left, right, contract(shape=(4, 3)))["status"]
        == "equivalent"
    )


def test_snapshot_key_is_application_reported_and_not_substituted_by_harness():
    left, right = snapshot(step=4), snapshot(step=3)
    left["harness_step"] = right["harness_step"] = 4
    result = api().compare_snapshots(left, right, contract())
    assert result["status"] == "difference"
    assert result["first_witness"]["kind"] == "observation_key_mismatch"


@pytest.mark.parametrize(
    "key",
    [
        [True, "ordinary", 0],
        [0, "unknown", 0],
        [0, "ordinary", -1],
        [0, "ordinary"],
        "0",
    ],
)
def test_bad_snapshot_keys_are_invalid(key):
    right = snapshot()
    right["key"] = key
    assert api().compare_snapshots(snapshot(), right, contract())["status"] == "invalid"


def test_trajectory_preserves_first_intermediate_difference_even_when_final_matches():
    left = [snapshot(0, 0), snapshot(1, 1), snapshot(0, 2)]
    right = [snapshot(0, 0), snapshot(2, 1), snapshot(0, 2)]
    result = api().compare_trajectories(left, right, contract())
    assert result["status"] == "difference"
    assert result["first_witness"]["key"] == [1, "ordinary", 0]
    assert result["first_witness"]["observation_index"] == 1


def test_phase_sequence_is_insertion_order_not_lexicographic_order():
    sequence = [
        snapshot(0, 2),
        snapshot(0, 2, "before_save"),
        snapshot(0, 2, "after_save"),
        snapshot(0, 2, "after_restore"),
        snapshot(0, 2, "before_save", 1),
        snapshot(0, 2, "after_save", 1),
        snapshot(0, 2, "after_restore", 1),
    ]
    changed = copy.deepcopy(sequence)
    changed[1]["fields"]["x"][...] = 1
    changed[2]["fields"]["x"][...] = 2
    result = api().compare_trajectories(sequence, changed, contract())
    assert result["first_witness"]["key"] == [2, "before_save", 0]
    assert (
        api().compare_trajectories(sequence, sequence, contract())["status"]
        == "equivalent"
    )


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        ([snapshot()], []),
        ([snapshot(), snapshot()], [snapshot(), snapshot()]),
    ],
)
def test_empty_missing_or_duplicate_trajectory_observations_are_invalid(left, right):
    assert api().compare_trajectories(left, right, contract())["status"] == "invalid"


def test_late_nonfinite_is_not_hidden_by_earlier_difference():
    result = api().compare_trajectories(
        [snapshot(0, 0), snapshot(0, 1)],
        [snapshot(1, 0), snapshot(float("nan"), 1)],
        contract(),
    )
    assert result["status"] == "invalid"
    assert result["first_witness"]["key"] == [0, "ordinary", 0]
    assert any(d["kind"] == "nonfinite" for d in result["diagnostics"])


def test_record_shape_prepends_row_count_and_compares_vector_columns():
    batch = records(values=[[1, 2], [3, 4]])
    assert (
        api().compare_records(batch, batch, record_contract(shape=(2,)))["status"]
        == "equivalent"
    )


@pytest.mark.parametrize(
    "keys,values,kind",
    [
        (("step:0", "step:0", "step:2"), (1, 1, 2), "duplicate_record_keys"),
        (("step:0",), (1,), "missing_record_keys"),
        (("step:0", "step:1", "step:2"), (1, 9, 2), "unexpected_record_keys"),
        (("step:2", "step:0"), (2, 1), "reordered_record_keys"),
    ],
)
def test_record_identity_failures_even_when_both_arms_have_same_bad_keys(
    keys, values, kind
):
    bad = records(keys=keys, values=values)
    result = api().compare_records(bad, bad, record_contract())
    assert result["status"] == "invalid"
    assert kind in [d["kind"] for d in result["diagnostics"]]


def test_record_witness_names_semantic_row_not_synthetic_counter():
    result = api().compare_records(records(), records(values=(1, 9)), record_contract())
    assert result["status"] == "difference"
    assert result["first_witness"]["record_key"] == "step:2"
    assert result["first_witness"]["index"] == [1]


def test_zero_expected_records_valid_with_explicit_empty_columns():
    batch = records(keys=(), values=())
    assert (
        api().compare_records(batch, batch, record_contract(keys=()))["status"]
        == "equivalent"
    )


def test_all_five_complete_matching_arms_are_equivalent():
    result = api().adjudicate_arms(arms(), contract())
    assert result["status"] == "equivalent_under_contract"
    assert result["findings"] == ["equivalent_under_contract"]
    assert set(result["comparisons"]) == {"U1-U2", "U1-O", "U-P", "P-R", "U-R"}
    assert all(c["status"] == "equivalent" for c in result["comparisons"].values())


def test_sparse_observer_neutrality_compares_final_ordinary_not_diagnostic_phase():
    case = arms()
    for name in ("U1", "U2", "P", "R"):
        case[name]["observations"] = [
            snapshot(0, 0),
            snapshot(1, 1),
            snapshot(2, 1, "after_save"),
        ]
    case["O"]["observations"] = [snapshot(1, 1)]
    assert (
        api().adjudicate_arms(case, contract())["status"] == "equivalent_under_contract"
    )


@pytest.mark.parametrize(
    "name,status", [("U2", "baseline_inconclusive"), ("O", "observation_inconclusive")]
)
def test_control_difference_blocks_restart_attribution_but_preserves_direct_pairs(
    name, status
):
    case = arms({name: 1.0, "R": 2.0})
    result = api().adjudicate_arms(case, contract())
    assert result["status"] == status
    assert "restore_path_difference" not in result["findings"]
    assert result["comparisons"]["P-R"]["status"] == "difference"
    assert result["comparisons"]["U-R"]["status"] == "difference"


def test_save_and_restore_findings_are_both_retained():
    result = api().adjudicate_arms(arms({"P": 1.0, "R": 2.0}), contract())
    assert "save_path_difference" in result["findings"]
    assert "restore_path_difference" in result["findings"]


def test_tolerance_nontransitivity_never_becomes_success():
    result = api().adjudicate_arms(
        arms({"P": 0.75, "R": 1.5}), contract(mode="tolerance", atol=1.0)
    )
    assert result["comparisons"]["U-P"]["status"] == "equivalent"
    assert result["comparisons"]["P-R"]["status"] == "equivalent"
    assert result["comparisons"]["U-R"]["status"] == "difference"
    assert result["status"] == "restart_difference"
    assert "equivalent_under_contract" not in result["findings"]


@pytest.mark.parametrize("name", ["U1", "U2", "O", "P", "R"])
def test_missing_arms_never_succeed(name):
    case = arms()
    del case[name]
    assert api().adjudicate_arms(case, contract())["status"] == "incomplete"


@pytest.mark.parametrize(
    "status", ["driver_error", "timeout", "checkpoint_error", "unsupported"]
)
def test_failed_driver_status_is_not_masked_by_matching_observations(status):
    case = arms()
    case["R"]["status"] = status
    result = api().adjudicate_arms(case, contract())
    assert result["status"] == status
    assert "equivalent_under_contract" not in result["findings"]


def test_output_violations_are_findings_even_if_all_arms_repeat_same_bad_table():
    case = arms()
    for arm in case.values():
        arm["outputs"] = {"samples": records(keys=("step:0",), values=(1,))}
    result = api().adjudicate_arms(
        case, contract(outputs={"samples": record_contract()})
    )
    assert "output_contract_violation" in result["findings"]
    assert result["status"] != "equivalent_under_contract"


def test_restore_output_numeric_difference_retains_output_and_path_findings():
    case = arms()
    for arm in case.values():
        arm["outputs"] = {"samples": records()}
    case["R"]["outputs"]["samples"] = records(values=(1, 9))
    result = api().adjudicate_arms(
        case, contract(outputs={"samples": record_contract()})
    )
    assert "output_contract_violation" in result["findings"]
    assert "restore_path_difference" in result["findings"]
    assert result["first_witness"]["record_key"] == "step:2"


def test_undeclared_output_tables_and_missing_output_tables_fail_closed():
    case = arms()
    case["R"]["outputs"] = {"unexpected": records()}
    result = api().adjudicate_arms(case, contract())
    assert "output_contract_violation" in result["findings"]
    case["R"]["outputs"] = {}
    result = api().adjudicate_arms(
        case, contract(outputs={"samples": record_contract()})
    )
    assert "output_contract_violation" in result["findings"]


def test_comparator_rejects_an_unvalidated_contract_as_json_result():
    result = api().compare_snapshots(snapshot(), snapshot(), {})
    assert result["status"] == "invalid"
    assert result["first_witness"]["kind"] == "invalid_contract"


def test_integer_relative_diagnostic_does_not_round_distinct_uint64_to_zero():
    result = api().compare_snapshots(
        snapshot(2**64 - 1, dtype="uint64"),
        snapshot(2**64 - 2, dtype="uint64"),
        contract("uint64"),
    )
    assert result["first_witness"]["relative_difference"] == 1 / (2**64 - 1)


def test_masked_arrays_cannot_hide_a_difference_or_nonfinite_observation():
    left, right = snapshot(), snapshot()
    left["fields"]["x"] = np.ma.array(float("nan"), mask=True)
    right["fields"]["x"] = np.ma.array(3.0, mask=True)
    result = api().compare_snapshots(left, right, contract())
    assert result["status"] == "invalid"
    assert result["first_witness"]["kind"] == "not_numeric_array"
    json.dumps(result, allow_nan=False)


def test_unsupported_extended_precision_nonfinite_diagnostic_stays_json_safe():
    right = snapshot()
    right["fields"]["x"] = np.array(float("inf"), dtype=np.longdouble)
    result = api().compare_snapshots(snapshot(), right, contract())
    assert result["status"] == "invalid"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1.25, 1.25),
        (float("inf"), "Infinity"),
        (-float("inf"), "-Infinity"),
        (float("nan"), "NaN"),
    ],
)
def test_longdouble_diagnostics_when_storage_is_eight_bytes(value, expected):
    # ARM macOS/Windows longdouble has 8-byte storage, yet item() still
    # returns np.longdouble. Emulate only that dtype width on other platforms.
    class EightByteLongdouble(np.longdouble):
        @property
        def dtype(self):
            return np.dtype("float64")

    scalar = EightByteLongdouble(value)
    assert isinstance(scalar.item(), np.longdouble)
    result = api()._value(scalar)
    assert json.loads(json.dumps(result, allow_nan=False)) == expected


def test_records_report_all_identity_problems_in_one_pass():
    result = api().compare_records(
        records(keys=("step:0", "step:0", "wrong"), values=(1, 2, 3)),
        records(),
        record_contract(),
    )
    kinds = {d["kind"] for d in result["diagnostics"]}
    assert {
        "duplicate_record_keys",
        "missing_record_keys",
        "unexpected_record_keys",
    } <= kinds


def test_big_endian_float_arrays_have_canonical_bit_witnesses():
    result = api().compare_snapshots(
        snapshot(0.0, dtype=">f8"), snapshot(-0.0, dtype=">f8"), contract(">f8")
    )
    assert result["first_witness"]["left_bits"] == "0x0000000000000000"
    assert result["first_witness"]["right_bits"] == "0x8000000000000000"


def test_first_field_order_follows_contract_rather_than_input_mapping_order():
    c = Contract((FieldContract("b", "float64", ()), FieldContract("a", "float64", ())))
    left = {
        "key": [1, "ordinary", 0],
        "fields": {"a": np.array(1.0), "b": np.array(1.0)},
        "units": {"a": "", "b": ""},
    }
    right = copy.deepcopy(left)
    right["fields"] = {"a": np.array(2.0), "b": np.array(2.0)}
    assert api().compare_snapshots(left, right, c)["first_witness"]["field"] == "b"


def test_same_length_reordered_observations_are_not_aligned_away():
    left = [snapshot(0, 0), snapshot(1, 1)]
    result = api().compare_trajectories(left, list(reversed(left)), contract())
    assert result["status"] == "difference"
    assert result["first_witness"]["kind"] == "observation_key_mismatch"


def test_comparison_does_not_mutate_raw_input_arrays():
    left, right = snapshot([-0.0, 2.0]), snapshot([0.0, 3.0])
    before = (left["fields"]["x"].tobytes(), right["fields"]["x"].tobytes())
    api().compare_snapshots(left, right, contract(shape=(2,)))
    assert before == (left["fields"]["x"].tobytes(), right["fields"]["x"].tobytes())


def test_tolerance_boundary_uses_exact_fallback_for_lost_subnormal():
    tiny = float(np.nextafter(0.0, 1.0))
    result = api().compare_snapshots(
        snapshot(1.0), snapshot(-tiny), contract(mode="tolerance", atol=1.0)
    )
    assert result["status"] == "difference"


def test_tolerance_matches_independent_rational_bound_on_adversarial_floats():
    from fractions import Fraction

    values = [
        0.0,
        float(np.nextafter(0.0, 1.0)),
        1e-300,
        1e-20,
        1.0,
        float(np.nextafter(1.0, 2.0)),
        1e300,
    ]
    for a in values:
        for b in values:
            for signed in (b, -b):
                for atol, rtol in ((a, 0.0), (0.0, 1.0), (1.0, 1e-20)):
                    expected = abs(Fraction(a) - Fraction(signed)) <= Fraction(
                        atol
                    ) + Fraction(rtol) * max(abs(Fraction(a)), abs(Fraction(signed)))
                    got = api().compare_snapshots(
                        snapshot(a),
                        snapshot(signed),
                        contract(mode="tolerance", atol=atol, rtol=rtol),
                    )
                    assert (got["status"] == "equivalent") == expected, (
                        a,
                        signed,
                        atol,
                        rtol,
                    )


def test_generated_tolerance_cases_match_exact_rational_oracle():
    from fractions import Fraction
    from hypothesis import given, settings, strategies as st

    finite = st.floats(width=64, allow_nan=False, allow_infinity=False)
    nonnegative = st.floats(
        min_value=0.0, width=64, allow_nan=False, allow_infinity=False
    )

    @settings(max_examples=150, deadline=None, derandomize=True)
    @given(finite, finite, nonnegative, nonnegative)
    def check(a, b, atol, rtol):
        expected = abs(Fraction(a) - Fraction(b)) <= Fraction(atol) + Fraction(
            rtol
        ) * max(abs(Fraction(a)), abs(Fraction(b)))
        got = api().compare_snapshots(
            snapshot(a), snapshot(b), contract(mode="tolerance", atol=atol, rtol=rtol)
        )
        assert (got["status"] == "equivalent") == expected

    check()


@pytest.mark.parametrize("invalid_evidence", ["nonfinite", "unit"])
def test_valid_save_difference_cannot_hide_invalid_restore_evidence(invalid_evidence):
    case = arms({"P": 1.0})
    if invalid_evidence == "nonfinite":
        case["R"]["observations"][0]["fields"]["x"][...] = np.nan
    else:
        case["R"]["observations"][0]["units"]["x"] = "cm"
    result = api().adjudicate_arms(case, contract())
    assert result["findings"] == ["save_path_difference", "invalid"]
    assert result["comparisons"]["U-P"]["status"] == "difference"
    assert result["comparisons"]["P-R"]["status"] == "invalid"
    assert result["comparisons"]["U-R"]["status"] == "invalid"
    assert result["first_witness"]["comparison"] == "U-P"


def test_unsupported_worker_with_valid_difference_is_not_invalid_evidence():
    case = arms({"P": 1.0})
    case["R"]["status"] = "unsupported"
    result = api().adjudicate_arms(case, contract())
    assert result["findings"] == ["unsupported"]
    assert result["comparisons"]["U-P"]["status"] == "difference"
    assert any(d["kind"] == "arm_not_complete" for d in result["diagnostics"])


def test_unsupported_does_not_hide_malformed_partial_observations():
    case = arms({"P": 1.0, "R": np.nan})
    case["R"]["status"] = "unsupported"
    result = api().adjudicate_arms(case, contract())
    assert result["findings"] == ["unsupported", "invalid"]


def test_complete_empty_evidence_is_invalid_despite_an_unsupported_peer():
    case = arms()
    case["U1"]["status"] = "unsupported"
    case["O"]["observations"] = []
    result = api().adjudicate_arms(case, contract())
    assert "invalid" in result["findings"]


def test_unsupported_observer_with_present_malformed_key_remains_invalid():
    from tests.independent_check import adjudicate

    case = arms()
    case["O"]["status"] = "unsupported"
    case["O"]["observations"][0]["key"] = [4, "unknown_phase", 0]
    for result in (
        api().adjudicate_arms(case, contract()),
        adjudicate(case, contract().to_dict()),
    ):
        assert "invalid" in result["findings"]


def test_unsupported_observer_with_nonlist_evidence_fails_closed():
    from tests.independent_check import adjudicate

    case = arms()
    case["O"]["status"] = "unsupported"
    case["O"]["observations"] = np.array([1.0, 2.0])
    for result in (
        api().adjudicate_arms(case, contract()),
        adjudicate(case, contract().to_dict()),
    ):
        assert "invalid" in result["findings"]
