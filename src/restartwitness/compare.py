"""Independently adjudicate numeric evidence without trusting cached labels.

All public comparisons return JSON-compatible dictionaries. Invalid evidence is
never equal, even when both arms contain the same malformed value or record key.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from fractions import Fraction
import math
from typing import Any

import numpy as np

from .contracts import Contract, FieldContract, RecordContract

_PHASES = {"ordinary", "before_save", "after_save", "after_restore"}
_ARM_NAMES = ("U1", "U2", "O", "P", "R")


def _issue(kind: str, severity: str = "invalid", **details: Any) -> dict:
    return {"kind": kind, "severity": severity, **details}


def _result(diagnostics: list[dict]) -> dict:
    status = (
        "invalid"
        if any(d["severity"] == "invalid" for d in diagnostics)
        else ("difference" if diagnostics else "equivalent")
    )
    return {
        "status": status,
        "first_witness": diagnostics[0] if diagnostics else None,
        "diagnostics": diagnostics,
    }


def _value(value: Any) -> Any:
    """Serialize numeric witnesses without nonstandard JSON NaN/Infinity."""
    if isinstance(value, np.generic):
        if isinstance(value, np.floating) and value.dtype.itemsize > 8:
            if not np.isfinite(value):
                return (
                    "NaN"
                    if np.isnan(value)
                    else ("Infinity" if value > 0 else "-Infinity")
                )
            return str(value)
        # longdouble.item() can remain a NumPy scalar even when its storage
        # is eight bytes (for example ARM macOS and Windows).
        value = float(value) if isinstance(value, np.floating) else value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return (
            "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
        )
    return value


def _key(snapshot: Any) -> list | None:
    if not isinstance(snapshot, Mapping):
        return None
    key = snapshot.get("key")
    if not isinstance(key, (list, tuple)) or len(key) != 3:
        return None
    step, phase, occurrence = key
    if (
        type(step) is not int
        or step < 0
        or not isinstance(phase, str)
        or phase not in _PHASES
        or type(occurrence) is not int
        or occurrence < 0
    ):
        return None
    return list(key)


def _names(mapping: Mapping) -> list[str]:
    return sorted(
        name if isinstance(name, str) else f"<{type(name).__name__}>"
        for name in mapping
    )


def _validate_fields(
    batch: Any,
    fields: tuple[FieldContract, ...],
    side: str,
    prefix: tuple[int, ...] = (),
) -> list[dict]:
    if not isinstance(batch, Mapping):
        return [_issue("invalid_observation", side=side)]
    arrays, units = batch.get("fields"), batch.get("units")
    if not isinstance(arrays, Mapping) or not isinstance(units, Mapping):
        return [
            _issue(
                "invalid_observation",
                side=side,
                detail="fields and units must be mappings",
            )
        ]
    diagnostics = []
    expected = {f.name for f in fields}
    if set(arrays) != expected:
        diagnostics.append(
            _issue(
                "field_names_mismatch",
                side=side,
                expected=sorted(expected),
                actual=_names(arrays),
            )
        )
    if set(units) != expected:
        diagnostics.append(
            _issue(
                "unit_names_mismatch",
                side=side,
                expected=sorted(expected),
                actual=_names(units),
            )
        )
    for field in fields:
        if field.name not in arrays:
            continue
        array = arrays[field.name]
        context = {"side": side, "field": field.name, "contract": field.to_dict()}
        if type(array) is not np.ndarray:
            diagnostics.append(_issue("not_numeric_array", **context))
            continue
        if array.dtype != np.dtype(field.dtype):
            diagnostics.append(
                _issue(
                    "dtype_mismatch",
                    **context,
                    expected=field.dtype,
                    actual=str(array.dtype),
                )
            )
        shape = prefix + field.shape
        if array.shape != shape:
            diagnostics.append(
                _issue(
                    "shape_mismatch",
                    **context,
                    expected=list(shape),
                    actual=list(array.shape),
                )
            )
        if units.get(field.name) != field.unit or not isinstance(
            units.get(field.name), str
        ):
            diagnostics.append(
                _issue(
                    "unit_mismatch",
                    **context,
                    expected=field.unit,
                    actual=units.get(field.name)
                    if isinstance(units.get(field.name), str)
                    else None,
                )
            )
        if array.dtype.kind in "biuf":
            nonfinite = ~np.isfinite(array)
            if np.any(nonfinite):
                flat = int(np.flatnonzero(nonfinite)[0])
                index = np.unravel_index(flat, array.shape)
                diagnostics.append(
                    _issue(
                        "nonfinite",
                        **context,
                        index=[int(n) for n in index],
                        value=_value(array[index]),
                        nonfinite_count=int(np.count_nonzero(nonfinite)),
                    )
                )
    return diagnostics


def _float_bits(array: np.ndarray) -> np.ndarray:
    unsigned = np.dtype(f"{array.dtype.byteorder}u{array.dtype.itemsize}")
    return np.ascontiguousarray(array).view(unsigned).reshape(array.shape)


def _tolerance_mismatches(
    left: np.ndarray, right: np.ndarray, field: FieldContract
) -> np.ndarray:
    # Extended precision avoids common overflow/underflow in float64 bounds.
    # On platforms where longdouble is float64, exact rational fallback handles
    # overflowing differences or bounds. It never rounds integer observations.
    a, b = left.astype(np.longdouble), right.astype(np.longdouble)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        difference = np.abs(a - b)
        bound = np.longdouble(field.atol) + np.longdouble(field.rtol) * np.maximum(
            np.abs(a), np.abs(b)
        )
        mismatch = difference > bound
    # Near a rounded boundary, even extended precision may lose a subnormal
    # addend. Resolve those comparisons using exact values of the binary floats.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rounding_guard = (
            16 * np.finfo(np.longdouble).eps * (np.abs(a) + np.abs(b) + np.abs(bound))
        )
        near_boundary = np.abs(difference - bound) <= rounding_guard
    uncertain = (~np.isfinite(difference) | ~np.isfinite(bound) | near_boundary) & (
        left != right
    )
    if np.any(uncertain):
        mismatch = np.array(mismatch, copy=True)
        for flat in np.flatnonzero(uncertain):
            index = np.unravel_index(int(flat), left.shape)
            x, y = Fraction(float(left[index])), Fraction(float(right[index]))
            limit = Fraction(field.atol) + Fraction(field.rtol) * max(abs(x), abs(y))
            mismatch[index] = abs(x - y) > limit
    return mismatch


def _numeric_diagnostics(
    left: Mapping, right: Mapping, fields: tuple[FieldContract, ...]
) -> list[dict]:
    diagnostics = []
    for field in fields:
        a, b = left["fields"][field.name], right["fields"][field.name]
        if a.dtype.kind == "f" and field.mode == "exact":
            different = _float_bits(a) != _float_bits(b)
        elif a.dtype.kind == "f":
            different = _tolerance_mismatches(a, b, field)
        else:
            different = a != b
        if not np.any(different):
            continue
        flat = int(np.flatnonzero(different)[0])
        index = np.unravel_index(flat, a.shape)
        x, y = _value(a[index]), _value(b[index])
        absolute = abs(x - y)
        scale = max(abs(x), abs(y))
        # Scaling first keeps a finite relative diagnostic for opposite extremes.
        if isinstance(absolute, float) and not math.isfinite(absolute):
            relative = abs(x / scale - y / scale)
        else:
            relative = absolute / scale if scale else 0.0
        witness = _issue(
            "numeric_difference",
            "difference",
            field=field.name,
            index=[int(n) for n in index],
            left=x,
            right=y,
            absolute_difference=absolute
            if not isinstance(absolute, float) or math.isfinite(absolute)
            else None,
            relative_difference=relative,
            mismatch_count=int(np.count_nonzero(different)),
            contract=field.to_dict(),
        )
        if a.dtype.kind == "f" and field.mode == "exact":
            width = a.dtype.itemsize * 2
            witness["left_bits"] = f"0x{int(_float_bits(a)[index]):0{width}x}"
            witness["right_bits"] = f"0x{int(_float_bits(b)[index]):0{width}x}"
        if isinstance(absolute, float) and not math.isfinite(absolute):
            witness["absolute_difference_overflow"] = True
        diagnostics.append(witness)
    return diagnostics


def compare_snapshots(left: Any, right: Any, contract: Contract) -> dict:
    """Compare one matched observation, using the application's reported key."""
    if not isinstance(contract, Contract):
        return _result([_issue("invalid_contract")])
    diagnostics = []
    left_key, right_key = _key(left), _key(right)
    for side, key in (("left", left_key), ("right", right_key)):
        if key is None:
            diagnostics.append(_issue("invalid_observation_key", side=side))
    if left_key is not None and right_key is not None and left_key != right_key:
        diagnostics.append(
            _issue(
                "observation_key_mismatch", "difference", left=left_key, right=right_key
            )
        )
    validation = _validate_fields(left, contract.fields, "left") + _validate_fields(
        right, contract.fields, "right"
    )
    diagnostics.extend(validation)
    if not validation:
        diagnostics.extend(_numeric_diagnostics(left, right, contract.fields))
    for diagnostic in diagnostics:
        if left_key is not None:
            diagnostic["key"] = left_key
        if right_key is not None and right_key != left_key:
            diagnostic["right_key"] = right_key
    return _result(diagnostics)


def compare_trajectories(left: Any, right: Any, contract: Contract) -> dict:
    """Compare observations in recorded order; never sort phase names."""
    if not isinstance(contract, Contract):
        return _result([_issue("invalid_contract")])
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return _result([_issue("invalid_trajectory")])
    diagnostics = []
    for side, observations in (("left", left), ("right", right)):
        if not observations:
            diagnostics.append(_issue("empty_trajectory", side=side))
    seen = {"left": set(), "right": set()}
    for position in range(max(len(left), len(right))):
        for side, observations in (("left", left), ("right", right)):
            if position >= len(observations):
                continue
            key = _key(observations[position])
            if key is not None:
                identity = tuple(key)
                if identity in seen[side]:
                    diagnostics.append(
                        _issue(
                            "duplicate_observation_key",
                            side=side,
                            key=key,
                            observation_index=position,
                        )
                    )
                seen[side].add(identity)
        if position >= len(left) or position >= len(right):
            diagnostics.append(
                _issue(
                    "missing_observation",
                    side="left" if position >= len(left) else "right",
                    observation_index=position,
                )
            )
            observations = right if position >= len(left) else left
            # Still validate unmatched evidence rather than hiding a later NaN.
            checked = compare_snapshots(
                observations[position], observations[position], contract
            )
        else:
            checked = compare_snapshots(left[position], right[position], contract)
        diagnostics.extend(
            {**d, "observation_index": position} for d in checked["diagnostics"]
        )
    return _result(diagnostics)


def _validate_keys(batch: Any, contract: RecordContract, side: str) -> list[dict]:
    if not isinstance(batch, Mapping):
        return [_issue("invalid_record_batch", side=side)]
    keys = batch.get("keys")
    if not isinstance(keys, (list, tuple)) or any(
        not isinstance(k, str) or not k for k in keys
    ):
        return [_issue("invalid_record_keys", side=side)]
    diagnostics = []
    counts = Counter(keys)
    duplicate = list(dict.fromkeys(key for key in keys if counts[key] > 1))
    missing = [key for key in contract.expected_keys if key not in counts]
    expected_set = set(contract.expected_keys)
    unexpected = list(dict.fromkeys(key for key in keys if key not in expected_set))
    for kind, values in (
        ("duplicate_record_keys", duplicate),
        ("missing_record_keys", missing),
        ("unexpected_record_keys", unexpected),
    ):
        if values:
            diagnostics.append(_issue(kind, side=side, keys=values))
    if not diagnostics and tuple(keys) != contract.expected_keys:
        diagnostics.append(
            _issue(
                "reordered_record_keys",
                side=side,
                expected=list(contract.expected_keys),
                actual=list(keys),
            )
        )
    return diagnostics


def compare_records(left: Any, right: Any, contract: RecordContract) -> dict:
    """Validate expected semantic identities before comparing table columns."""
    if not isinstance(contract, RecordContract):
        return _result([_issue("invalid_contract")])
    diagnostics = _validate_keys(left, contract, "left") + _validate_keys(
        right, contract, "right"
    )
    for side, batch in (("left", left), ("right", right)):
        keys = batch.get("keys") if isinstance(batch, Mapping) else None
        if isinstance(keys, (list, tuple)):
            diagnostics.extend(
                _validate_fields(batch, contract.fields, side, (len(keys),))
            )
        else:
            diagnostics.extend(_validate_fields(batch, contract.fields, side))
    if not diagnostics:
        diagnostics = _numeric_diagnostics(left, right, contract.fields)
        for diagnostic in diagnostics:
            diagnostic["record_key"] = contract.expected_keys[diagnostic["index"][0]]
    for diagnostic in diagnostics:
        diagnostic["table"] = contract.name
    return _result(diagnostics)


def _compare_outputs(left: Any, right: Any, contract: Contract) -> dict:
    diagnostics = []
    expected = set(contract.outputs)
    for side, outputs in (("left", left), ("right", right)):
        if not isinstance(outputs, Mapping):
            diagnostics.append(_issue("invalid_outputs", side=side))
        elif set(outputs) != expected:
            diagnostics.append(
                _issue(
                    "output_table_names_mismatch",
                    side=side,
                    expected=sorted(expected),
                    actual=_names(outputs),
                )
            )
    tables = {}
    for name, record in contract.outputs.items():
        left_table = left.get(name) if isinstance(left, Mapping) else None
        right_table = right.get(name) if isinstance(right, Mapping) else None
        tables[name] = compare_records(left_table, right_table, record)
        diagnostics.extend(tables[name]["diagnostics"])
    return {**_result(diagnostics), "tables": tables}


def _final_ordinary(arm: Mapping) -> list:
    observations = arm.get("observations")
    if isinstance(observations, (list, tuple)):
        for snapshot in reversed(observations):
            key = _key(snapshot)
            if key is not None and key[1] == "ordinary":
                return [snapshot]
    return []


def _compare_arms(
    left: Any, right: Any, contract: Contract, final_only: bool = False
) -> dict:
    diagnostics = []
    for side, arm in (("left", left), ("right", right)):
        if not isinstance(arm, Mapping):
            diagnostics.append(_issue("missing_arm", side=side))
        elif arm.get("status") != "complete":
            diagnostics.append(
                _issue(
                    "arm_not_complete",
                    side=side,
                    arm_status=arm.get("status")
                    if isinstance(arm.get("status"), str)
                    else None,
                )
            )
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return {
            **_result(diagnostics),
            "trajectory": _result(diagnostics.copy()),
            "outputs": _result([]),
        }
    trajectory = compare_trajectories(
        _final_ordinary(left) if final_only else left.get("observations"),
        _final_ordinary(right) if final_only else right.get("observations"),
        contract,
    )
    outputs = _compare_outputs(left.get("outputs"), right.get("outputs"), contract)
    diagnostics.extend(trajectory["diagnostics"])
    diagnostics.extend(outputs["diagnostics"])
    return {**_result(diagnostics), "trajectory": trajectory, "outputs": outputs}


def _invalid_evidence(diagnostic: dict, arm: Any) -> bool:
    """Separate malformed data from evidence a non-complete worker never produced."""
    kind = diagnostic["kind"]
    if diagnostic["severity"] != "invalid" or kind in (
        "missing_arm",
        "arm_not_complete",
    ):
        return False
    if isinstance(arm, Mapping) and arm.get("status") != "complete":
        if kind == "missing_observation":
            return False
        observations = arm.get("observations")
        if (
            kind == "empty_trajectory"
            and isinstance(observations, (list, tuple))
            and not observations
        ):
            return False
        outputs = arm.get("outputs")
        if isinstance(outputs, Mapping):
            if kind == "output_table_names_mismatch":
                return bool(set(outputs) - set(diagnostic["expected"]))
            if kind in ("invalid_record_batch", "invalid_observation") and (
                "table" in diagnostic and diagnostic["table"] not in outputs
            ):
                return False
    return True


def adjudicate_arms(arms: Any, contract: Contract) -> dict:
    """Retain all direct contrasts, gating intervention attribution on controls.

    U1/U2 are repeated uninterrupted baselines; O observes only at the end;
    P saves and continues; R saves and restores in fresh processes.
    """
    if not isinstance(contract, Contract):
        return {
            "status": "invalid_contract",
            "findings": ["invalid_contract"],
            "comparisons": {},
            "first_witness": _issue("invalid_contract"),
            "diagnostics": [],
        }
    if not isinstance(arms, Mapping):
        arms = {}
    pairs = {
        "U1-U2": ("U1", "U2"),
        "U1-O": ("U1", "O"),
        "U-P": ("U1", "P"),
        "P-R": ("P", "R"),
        "U-R": ("U1", "R"),
    }
    comparisons = {
        label: _compare_arms(
            arms.get(left_name), arms.get(right_name), contract, label == "U1-O"
        )
        for label, (left_name, right_name) in pairs.items()
    }
    findings = []
    missing = [name for name in _ARM_NAMES if not isinstance(arms.get(name), Mapping)]
    if missing:
        findings.append("incomplete")
    else:
        for name in _ARM_NAMES:
            status = arms[name].get("status")
            if status != "complete":
                failure = (
                    status
                    if status
                    in (
                        "driver_error",
                        "checkpoint_error",
                        "timeout",
                        "unsupported",
                        "evidence_integrity_error",
                    )
                    else "incomplete"
                )
                if failure not in findings:
                    findings.append(failure)
    # Malformed controls cannot establish intervention attribution. Their raw
    # contrasts remain available even when a worker failure has higher priority.
    if comparisons["U1-U2"]["status"] != "equivalent":
        findings.append("baseline_inconclusive")
    if comparisons["U1-O"]["status"] != "equivalent":
        findings.append("observation_inconclusive")
    controls_valid = all(
        comparisons[p]["status"] == "equivalent" for p in ("U1-U2", "U1-O")
    )
    if controls_valid and not findings:
        for pair, finding in (
            ("U-P", "save_path_difference"),
            ("P-R", "restore_path_difference"),
        ):
            if comparisons[pair]["status"] == "difference":
                findings.append(finding)
        if comparisons["U-R"]["status"] == "difference" and not findings:
            findings.append("restart_difference")
    output_bad = any(
        c["outputs"]["status"] != "equivalent" for c in comparisons.values()
    )
    if output_bad:
        findings.append("output_contract_violation")
    # Evidence invalidity is independent of valid differences or control findings.
    # Worker availability already has its own finding (including unsupported).
    if any(
        _invalid_evidence(
            d, arms.get(pairs[label][0 if d.get("side") == "left" else 1])
        )
        for label, comparison in comparisons.items()
        for d in comparison["diagnostics"]
    ):
        findings.append("invalid")
    if not findings:
        findings.append("equivalent_under_contract")
    diagnostics = [
        {**diagnostic, "comparison": label}
        for label, comparison in comparisons.items()
        for diagnostic in comparison["diagnostics"]
    ]
    return {
        "status": findings[0],
        "findings": findings,
        "comparisons": comparisons,
        "first_witness": diagnostics[0] if diagnostics else None,
        "diagnostics": diagnostics,
    }
