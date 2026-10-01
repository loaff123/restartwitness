#!/usr/bin/env python3
"""Independent, bounded re-adjudication of a local RestartWitness case.

Uses only the standard library and NumPy. It deliberately imports none of the
production reader, contracts, comparator, driver or runner. Numeric tolerance
arithmetic uses exact rational values of the recorded binary floats. Run:

    python -I tests/independent_check.py CASE_DIRECTORY

Hash verification is integrity checking, not authentication of the case author.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np

MAX_JSON = 4 * 1024 * 1024
MAX_ARRAY = 32 * 1024 * 1024
MAX_BUNDLE = 128 * 1024 * 1024
MAX_FILES = 20000
MAX_DECODE_BYTES = 128 * 1024 * 1024
ARMS = ("U1", "U2", "O", "P", "R")
PAIRS = {
    "U1-U2": ("U1", "U2"),
    "U1-O": ("U1", "O"),
    "U-P": ("U1", "P"),
    "P-R": ("P", "R"),
    "U-R": ("U1", "R"),
}


class CheckError(ValueError):
    """Evidence cannot support the stored claim."""


def require(condition, message):
    if not condition:
        raise CheckError(message)


def _json(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            require(k not in result, "duplicate JSON key")
            result[k] = v
        return result

    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_JSON,
        "missing or oversized JSON",
    )
    try:
        return json.loads(
            path.read_text(encoding="utf8"),
            object_pairs_hook=unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                CheckError("nonfinite JSON")
            ),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise CheckError(f"invalid JSON: {exc}") from exc


def _path(root, name):
    require(
        isinstance(name, str) and name and "\\" not in name and ":" not in name,
        "unsafe path",
    )
    relative = PurePosixPath(name)
    require(
        not relative.is_absolute()
        and str(relative) == name
        and all(part not in (".", "..") for part in relative.parts),
        "unsafe path",
    )
    target = root.joinpath(*relative.parts)
    require(not any(p.is_symlink() for p in (target, *target.parents)), "symlink path")
    require(target.resolve().is_relative_to(root.resolve()), "path escape")
    return target


def _hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(root):
    require(
        root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
        "unsafe case directory",
    )
    entries, total = {}, 0
    for p in root.rglob("*"):
        require(not p.is_symlink(), "symlink in bundle")
        if p.is_dir():
            continue
        require(p.is_file(), "nonregular bundle entry")
        if p == root / "manifest.json":
            continue
        total += p.stat().st_size
        require(
            total <= MAX_BUNDLE and len(entries) < MAX_FILES, "bundle size/file limit"
        )
        entries[p.relative_to(root).as_posix()] = p
    return entries


def _verify(root):
    entries = _inventory(root)  # Bound I/O before hashing attacker-controlled files.
    m = _json(root / "manifest.json")
    require(
        isinstance(m, dict)
        and set(m) == {"schema_version", "metadata", "artifacts"}
        and type(m["schema_version"]) is int
        and m["schema_version"] == 1,
        "unsupported manifest",
    )
    require(
        isinstance(m["metadata"], dict)
        and m["metadata"].get("kind") == "restartwitness-case",
        "not a case manifest",
    )
    require(
        isinstance(m["artifacts"], list) and 0 < len(m["artifacts"]) <= MAX_FILES,
        "invalid artifact list",
    )
    seen = set()
    for item in m["artifacts"]:
        require(
            isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"},
            "invalid artifact metadata",
        )
        p = _path(root, item["path"])
        require(item["path"] not in seen, "duplicate manifest path")
        seen.add(item["path"])
        require(
            item["path"] in entries
            and type(item["bytes"]) is int
            and 0 <= item["bytes"] <= MAX_BUNDLE,
            "missing artifact or invalid size",
        )
        require(
            isinstance(item["sha256"], str)
            and re.fullmatch("[0-9a-f]{64}", item["sha256"]),
            "invalid artifact digest",
        )
        require(
            p.stat().st_size == item["bytes"] and _hash(p) == item["sha256"],
            "artifact hash/size mismatch",
        )
    require(seen == set(entries), "manifest file set mismatch")
    return m


def _array(path):
    require(
        path.is_file() and path.stat().st_size <= MAX_ARRAY + 65536,
        "missing/oversized NPY",
    )
    try:
        with path.open("rb") as f:
            require(f.read(6) == b"\x93NUMPY", "invalid NPY magic")
            version = f.read(2)
            require(version in (b"\x01\x00", b"\x02\x00"), "unsupported NPY version")
            length_size = 2 if version[0] == 1 else 4
            length_bytes = f.read(length_size)
            require(len(length_bytes) == length_size, "truncated NPY header")
            length = int.from_bytes(length_bytes, "little")
            require(length <= 65536, "oversized NPY header")
            raw = f.read(length)
            require(len(raw) == length, "truncated NPY header")
            header = ast.literal_eval(raw.decode("latin1"))
            require(
                isinstance(header, dict)
                and set(header) == {"descr", "shape", "fortran_order"},
                "invalid NPY header",
            )
            shape, dtype = header["shape"], np.dtype(header["descr"])
            require(
                type(header["fortran_order"]) is bool
                and isinstance(shape, tuple)
                and len(shape) <= 8
                and all(type(n) is int and n >= 0 for n in shape),
                "invalid NPY dimensions",
            )
            require(
                dtype.kind in "biuf" and not dtype.hasobject and dtype.itemsize <= 8,
                "unsafe NPY dtype",
            )
            size = math.prod(shape) * dtype.itemsize
            require(
                size <= MAX_ARRAY and f.tell() + size == path.stat().st_size,
                "oversized/truncated/trailing NPY",
            )
        return np.load(path, allow_pickle=False)
    except (ValueError, SyntaxError, EOFError, TypeError, UnicodeError) as exc:
        raise CheckError(f"invalid NPY: {exc}") from exc


def _decode(root, value, cache, budget, depth=0):
    require(depth <= 40, "excessive JSON nesting")
    if isinstance(value, dict):
        if "$array" in value:
            require(set(value) == {"$array"}, "malformed array reference")
            p = _path(root, value["$array"])
            if p not in cache:
                cache[p] = _array(p)
            budget[0] += cache[p].nbytes
            require(
                budget[0] <= MAX_DECODE_BYTES, "decoded array reference budget exceeded"
            )
            return cache[p]
        return {k: _decode(root, v, cache, budget, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(root, v, cache, budget, depth + 1) for v in value]
    return value


def _contract(c):
    require(
        isinstance(c, dict)
        and set(c) == {"schema_version", "fields", "outputs"}
        and type(c["schema_version"]) is int
        and c["schema_version"] == 1,
        "invalid contract",
    )

    def fields(fs, nonempty=False):
        require(isinstance(fs, list) and (fs or not nonempty), "invalid field list")
        names = []
        for f in fs:
            require(
                isinstance(f, dict)
                and set(f)
                == {"name", "dtype", "shape", "unit", "mode", "atol", "rtol"},
                "invalid field contract",
            )
            require(
                isinstance(f["name"], str) and f["name"] and f["name"] not in names,
                "duplicate/invalid field name",
            )
            names.append(f["name"])
            require(isinstance(f["dtype"], str), "invalid dtype contract")
            dtype = np.dtype(f["dtype"])
            require(
                dtype.kind in "biuf" and dtype.itemsize <= 8, "invalid dtype contract"
            )
            require(
                isinstance(f["shape"], list)
                and len(f["shape"]) <= 8
                and all(type(n) is int and n >= 0 for n in f["shape"]),
                "invalid shape contract",
            )
            require(
                math.prod(f["shape"]) * dtype.itemsize <= MAX_ARRAY,
                "oversized shape contract",
            )
            require(
                isinstance(f["unit"], str) and f["mode"] in ("exact", "tolerance"),
                "invalid unit/mode contract",
            )
            for t in ("atol", "rtol"):
                require(
                    type(f[t]) in (int, float) and math.isfinite(f[t]) and f[t] >= 0,
                    "invalid tolerance",
                )

    fields(c["fields"], True)
    require(isinstance(c["outputs"], dict), "invalid output contracts")
    for name, table in c["outputs"].items():
        require(
            isinstance(name, str)
            and name
            and isinstance(table, dict)
            and set(table) == {"name", "expected_keys", "fields"}
            and table["name"] == name,
            "invalid output contract",
        )
        keys = table["expected_keys"]
        require(
            isinstance(keys, list)
            and all(isinstance(k, str) and k for k in keys)
            and len(keys) == len(set(keys)),
            "invalid expected output keys",
        )
        fields(table["fields"])
    return c


def _issue(kind, severity="invalid", **data):
    return {"kind": kind, "severity": severity, **data}


def _result(items):
    return {
        "status": "invalid"
        if any(d["severity"] == "invalid" for d in items)
        else "difference"
        if items
        else "equivalent",
        "first_witness": items[0] if items else None,
        "diagnostics": items,
    }


def _number(x):
    x = x.item() if isinstance(x, np.generic) else x
    if isinstance(x, float) and not math.isfinite(x):
        return "NaN" if math.isnan(x) else "Infinity" if x > 0 else "-Infinity"
    return x


def _valid_key(s):
    k = s.get("key") if isinstance(s, dict) else None
    return (
        isinstance(k, list)
        and len(k) == 3
        and type(k[0]) is int
        and k[0] >= 0
        and k[1] in ("ordinary", "before_save", "after_save", "after_restore")
        and type(k[2]) is int
        and k[2] >= 0
    )


def _schema(batch, fs, side, prefix=()):
    if (
        not isinstance(batch, dict)
        or not isinstance(batch.get("fields"), dict)
        or not isinstance(batch.get("units"), dict)
    ):
        return [_issue("invalid_observation", side=side)]
    issues = []
    names = {f["name"] for f in fs}
    for item, kind in (
        ("fields", "field_names_mismatch"),
        ("units", "unit_names_mismatch"),
    ):
        if set(batch[item]) != names:
            issues.append(
                _issue(
                    kind, side=side, expected=sorted(names), actual=sorted(batch[item])
                )
            )
    for f in fs:
        if f["name"] not in batch["fields"]:
            continue
        a = batch["fields"][f["name"]]
        context = {"side": side, "field": f["name"], "contract": f}
        if not isinstance(a, np.ndarray):
            issues.append(_issue("not_numeric_array", **context))
            continue
        if a.dtype != np.dtype(f["dtype"]):
            issues.append(
                _issue(
                    "dtype_mismatch",
                    expected=f["dtype"],
                    actual=str(a.dtype),
                    **context,
                )
            )
        expected = prefix + tuple(f["shape"])
        if a.shape != expected:
            issues.append(
                _issue(
                    "shape_mismatch",
                    expected=list(expected),
                    actual=list(a.shape),
                    **context,
                )
            )
        unit = batch["units"].get(f["name"])
        if unit != f["unit"] or not isinstance(unit, str):
            issues.append(
                _issue(
                    "unit_mismatch",
                    expected=f["unit"],
                    actual=unit if isinstance(unit, str) else None,
                    **context,
                )
            )
        if a.dtype.kind in "biuf":
            indices = [i for i in np.ndindex(a.shape) if not np.isfinite(a[i])]
            if indices:
                issues.append(
                    _issue(
                        "nonfinite",
                        index=list(indices[0]),
                        value=_number(a[indices[0]]),
                        nonfinite_count=len(indices),
                        **context,
                    )
                )
    return issues


def _bits(a, index):
    raw = np.asarray(a[index], dtype=a.dtype).tobytes()
    order = (
        "little"
        if a.dtype.byteorder == "<"
        or (a.dtype.byteorder in ("=", "|") and sys.byteorder == "little")
        else "big"
    )
    return f"0x{int.from_bytes(raw, order):0{2 * a.dtype.itemsize}x}"


def _numeric(a, b, fs):
    issues = []
    for f in fs:
        left, right = a["fields"][f["name"]], b["fields"][f["name"]]
        mismatches = []
        for index in np.ndindex(left.shape):
            x, y = left[index].item(), right[index].item()
            if left.dtype.kind == "f" and f["mode"] == "exact":
                mismatch = _bits(left, index) != _bits(right, index)
            elif left.dtype.kind == "f":
                xq, yq = Fraction(x), Fraction(y)
                mismatch = abs(xq - yq) > Fraction(f["atol"]) + Fraction(
                    f["rtol"]
                ) * max(abs(xq), abs(yq))
            else:
                mismatch = x != y
            if mismatch:
                mismatches.append(index)
        if not mismatches:
            continue
        i = mismatches[0]
        x, y = left[i].item(), right[i].item()
        absolute, scale = abs(x - y), max(abs(x), abs(y))
        overflow = isinstance(absolute, float) and not math.isfinite(absolute)
        issue = _issue(
            "numeric_difference",
            "difference",
            field=f["name"],
            index=list(i),
            left=x,
            right=y,
            absolute_difference=None if overflow else absolute,
            relative_difference=float(abs(Fraction(x) - Fraction(y)) / Fraction(scale))
            if scale
            else 0.0,
            mismatch_count=len(mismatches),
            contract=f,
        )
        if overflow:
            issue["absolute_difference_overflow"] = True
        if left.dtype.kind == "f" and f["mode"] == "exact":
            issue.update(left_bits=_bits(left, i), right_bits=_bits(right, i))
        issues.append(issue)
    return issues


def _snapshot(left, right, fs):
    issues = []
    for side, snap in (("left", left), ("right", right)):
        if not _valid_key(snap):
            issues.append(_issue("invalid_observation_key", side=side))
    if _valid_key(left) and _valid_key(right) and left["key"] != right["key"]:
        issues.append(
            _issue(
                "observation_key_mismatch",
                "difference",
                left=left["key"],
                right=right["key"],
            )
        )
    validation = _schema(left, fs, "left") + _schema(right, fs, "right")
    issues += validation
    if not validation:
        issues += _numeric(left, right, fs)
    for d in issues:
        if _valid_key(left):
            d["key"] = left["key"]
        if _valid_key(right) and (not _valid_key(left) or right["key"] != left["key"]):
            d["right_key"] = right["key"]
    return issues


def _trajectory(left, right, fs):
    if not isinstance(left, list) or not isinstance(right, list):
        return _result([_issue("invalid_trajectory")])
    issues, seen = [], {"left": set(), "right": set()}
    for side, trajectory in (("left", left), ("right", right)):
        if not trajectory:
            issues.append(_issue("empty_trajectory", side=side))
    for i in range(max(len(left), len(right))):
        for side, trajectory in (("left", left), ("right", right)):
            if i < len(trajectory) and _valid_key(trajectory[i]):
                k = tuple(trajectory[i]["key"])
                if k in seen[side]:
                    issues.append(
                        _issue(
                            "duplicate_observation_key",
                            side=side,
                            key=list(k),
                            observation_index=i,
                        )
                    )
                seen[side].add(k)
        if i >= min(len(left), len(right)):
            issues.append(
                _issue(
                    "missing_observation",
                    side="left" if i >= len(left) else "right",
                    observation_index=i,
                )
            )
            snap = right[i] if i >= len(left) else left[i]
            checked = _snapshot(snap, snap, fs)
        else:
            checked = _snapshot(left[i], right[i], fs)
        issues.extend(dict(d, observation_index=i) for d in checked)
    return _result(issues)


def _record_keys(batch, expected, side):
    if not isinstance(batch, dict):
        return [_issue("invalid_record_batch", side=side)]
    keys = batch.get("keys")
    if not isinstance(keys, list) or any(not isinstance(k, str) or not k for k in keys):
        return [_issue("invalid_record_keys", side=side)]
    counts, issues = Counter(keys), []
    for kind, bad in (
        (
            "duplicate_record_keys",
            list(dict.fromkeys(k for k in keys if counts[k] > 1)),
        ),
        ("missing_record_keys", [k for k in expected if k not in counts]),
        (
            "unexpected_record_keys",
            list(dict.fromkeys(k for k in keys if k not in set(expected))),
        ),
    ):
        if bad:
            issues.append(_issue(kind, side=side, keys=bad))
    if not issues and keys != expected:
        issues.append(
            _issue("reordered_record_keys", side=side, expected=expected, actual=keys)
        )
    return issues


def _outputs(left, right, contracts):
    issues, tables = [], {}
    for side, outputs in (("left", left), ("right", right)):
        if not isinstance(outputs, dict):
            issues.append(_issue("invalid_outputs", side=side))
        elif set(outputs) != set(contracts):
            issues.append(
                _issue(
                    "output_table_names_mismatch",
                    side=side,
                    expected=sorted(contracts),
                    actual=sorted(outputs),
                )
            )
    for name, c in contracts.items():
        a, b = (
            (left.get(name) if isinstance(left, dict) else None),
            (right.get(name) if isinstance(right, dict) else None),
        )
        ds = _record_keys(a, c["expected_keys"], "left") + _record_keys(
            b, c["expected_keys"], "right"
        )
        for side, batch in (("left", a), ("right", b)):
            keys = batch.get("keys") if isinstance(batch, dict) else None
            ds += _schema(
                batch, c["fields"], side, (len(keys),) if isinstance(keys, list) else ()
            )
        if not ds:
            ds = _numeric(a, b, c["fields"])
            for d in ds:
                d["record_key"] = c["expected_keys"][d["index"][0]]
        ds = [dict(d, table=name) for d in ds]
        tables[name] = _result(ds)
        issues += ds
    return dict(_result(issues), tables=tables)


def _pair(left, right, c, final=False):
    issues = []
    for side, arm in (("left", left), ("right", right)):
        if not isinstance(arm, dict):
            issues.append(_issue("missing_arm", side=side))
        elif arm.get("status") != "complete":
            issues.append(
                _issue("arm_not_complete", side=side, arm_status=arm.get("status"))
            )
    if not isinstance(left, dict) or not isinstance(right, dict):
        return dict(_result(issues), trajectory=_result(issues), outputs=_result([]))

    def observations(arm):
        obs = arm.get("observations")
        if final:
            ordinary = (
                [s for s in obs if _valid_key(s) and s["key"][1] == "ordinary"]
                if isinstance(obs, list)
                else []
            )
            return ordinary[-1:]
        return obs

    trajectory = _trajectory(observations(left), observations(right), c["fields"])
    outputs = _outputs(left.get("outputs"), right.get("outputs"), c["outputs"])
    return dict(
        _result(issues + trajectory["diagnostics"] + outputs["diagnostics"]),
        trajectory=trajectory,
        outputs=outputs,
    )


def adjudicate(arms, contract):
    """Recompute every pair and first witness from decoded evidence."""
    comparisons = {
        name: _pair(arms.get(a), arms.get(b), contract, name == "U1-O")
        for name, (a, b) in PAIRS.items()
    }
    findings = []
    if any(not isinstance(arms.get(n), dict) for n in ARMS):
        findings.append("incomplete")
    else:
        for n in ARMS:
            status = arms[n].get("status")
            if status != "complete":
                problem = (
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
                if problem not in findings:
                    findings.append(problem)
    for pair, problem in (
        ("U1-U2", "baseline_inconclusive"),
        ("U1-O", "observation_inconclusive"),
    ):
        if comparisons[pair]["status"] != "equivalent":
            findings.append(problem)
    if not findings:
        for pair, problem in (
            ("U-P", "save_path_difference"),
            ("P-R", "restore_path_difference"),
        ):
            if comparisons[pair]["status"] == "difference":
                findings.append(problem)
        if not findings and comparisons["U-R"]["status"] == "difference":
            findings.append("restart_difference")
    if any(p["outputs"]["status"] != "equivalent" for p in comparisons.values()):
        findings.append("output_contract_violation")
    evidence_issues = []
    for label, pair in comparisons.items():
        for issue in pair["diagnostics"]:
            kind = issue["kind"]
            if issue["severity"] != "invalid" or kind in (
                "missing_arm",
                "arm_not_complete",
            ):
                continue
            side = 0 if issue.get("side") == "left" else 1
            arm = arms.get(PAIRS[label][side])
            if isinstance(arm, dict) and arm.get("status") != "complete":
                if kind == "missing_observation":
                    continue
                if (
                    kind == "empty_trajectory"
                    and isinstance(arm.get("observations"), list)
                    and len(arm["observations"]) == 0
                ):
                    continue
                outputs = arm.get("outputs")
                if isinstance(outputs, dict):
                    if kind == "output_table_names_mismatch" and set(outputs).issubset(
                        issue["expected"]
                    ):
                        continue
                    if (
                        kind in ("invalid_record_batch", "invalid_observation")
                        and "table" in issue
                        and issue["table"] not in outputs
                    ):
                        continue
            evidence_issues.append(issue)
    if evidence_issues:
        findings.append("invalid")
    if not findings:
        findings.append("equivalent_under_contract")
    all_issues = [
        dict(d, comparison=name)
        for name, p in comparisons.items()
        for d in p["diagnostics"]
    ]
    return {
        "status": findings[0],
        "findings": findings,
        "comparisons": comparisons,
        "first_witness": all_issues[0] if all_issues else None,
        "diagnostics": all_issues,
    }


def _cached(actual, cached, where="adjudication"):
    require(isinstance(cached, dict), f"cached {where} malformed")
    for field in ("status", "first_witness"):
        require(
            field in cached and cached[field] == actual[field],
            f"cached {where}.{field} disagrees with raw evidence",
        )
    if "comparisons" in actual:
        require(
            cached.get("findings") == actual["findings"],
            "cached findings disagree with raw evidence",
        )
        require(
            isinstance(cached.get("comparisons"), dict)
            and set(cached["comparisons"]) == set(actual["comparisons"]),
            "cached comparisons missing",
        )
        for name, pair in actual["comparisons"].items():
            _cached(pair, cached["comparisons"][name], name)
    for section in ("trajectory", "outputs"):
        if section in cached:
            _cached(actual[section], cached[section], f"{where}.{section}")
    if "tables" in cached:
        require(
            set(cached["tables"]) == set(actual.get("tables", {})),
            "cached table set mismatch",
        )
        for name in cached["tables"]:
            _cached(
                actual["tables"][name], cached["tables"][name], f"{where}.table.{name}"
            )


def _cadence(study, cuts, arms):
    n, steps = study.get("total_steps"), study.get("observations")
    require(type(n) is int and 0 <= n <= 1000000, "invalid study horizon")
    require(
        isinstance(steps, list)
        and steps
        and all(type(s) is int and 0 <= s <= n for s in steps)
        and steps == sorted(set(steps))
        and n in steps,
        "invalid observation cadence",
    )
    require(
        isinstance(cuts, list)
        and len(cuts) <= 128
        and all(type(k) is int and 0 <= k <= n for k in cuts)
        and cuts == sorted(cuts),
        "invalid schedule",
    )
    expected, ordinary_steps, cut_indices = [], set(steps), {}
    for index, cut in enumerate(cuts):
        cut_indices.setdefault(cut, []).append(index)
    for step in sorted(ordinary_steps | set(cuts)):
        if step in ordinary_steps:
            expected.append([step, "ordinary", 0])
        for index in cut_indices.get(step, []):
            expected.extend(
                [step, phase, index]
                for phase in ("before_save", "after_save", "after_restore")
            )
    for name, arm in arms.items():
        require(
            isinstance(arm, dict) and isinstance(arm.get("observations"), list),
            "invalid arm observations",
        )
        keys = [
            s.get("key") if isinstance(s, dict) else None for s in arm["observations"]
        ]
        declared = [[n, "ordinary", 0]] if name == "O" else expected
        if arm.get("status") == "complete":
            require(
                keys == declared,
                f"{name} observation cadence/phase sequence disagrees with study/schedule",
            )
        else:
            require(
                keys == declared[: len(keys)],
                f"{name} partial observation cadence is not a declared prefix",
            )


def _checkpoint_ledger(root, cuts, arms):
    for name, arm in arms.items():
        segments = arm.get("segments", [])
        require(isinstance(segments, list), "invalid checkpoint segment ledger")
        found = []
        for segment_index, segment in enumerate(segments):
            require(isinstance(segment, dict), "invalid checkpoint segment")
            checkpoints = segment.get("checkpoints", [])
            require(
                isinstance(checkpoints, list), "invalid checkpoint acknowledgment list"
            )
            for checkpoint in checkpoints:
                require(
                    isinstance(checkpoint, dict), "invalid checkpoint acknowledgment"
                )
                cut, step = checkpoint.get("cut"), checkpoint.get("step")
                require(
                    type(cut) is int
                    and 0 <= cut < len(cuts)
                    and type(step) is int
                    and step == cuts[cut],
                    "checkpoint cut/step mismatch",
                )
                found.append(cut)
                artifacts = checkpoint.get("artifacts")
                require(
                    isinstance(artifacts, list) and artifacts,
                    "checkpoint artifacts missing",
                )
                base = _path(
                    root, f"{name}/segment-{segment_index:03}/checkpoint-{cut:03}"
                )
                require(base.is_dir(), "checkpoint directory missing")
                acknowledged, total = set(), 0
                for a in artifacts:
                    require(
                        isinstance(a, dict) and set(a) == {"path", "bytes", "sha256"},
                        "invalid checkpoint artifact",
                    )
                    path = _path(base, a["path"])
                    require(
                        a["path"] not in acknowledged, "duplicate checkpoint artifact"
                    )
                    acknowledged.add(a["path"])
                    require(
                        type(a["bytes"]) is int
                        and a["bytes"] >= 0
                        and path.is_file()
                        and path.stat().st_size == a["bytes"],
                        "checkpoint artifact size mismatch",
                    )
                    require(
                        isinstance(a["sha256"], str) and _hash(path) == a["sha256"],
                        "checkpoint acknowledgment hash mismatch",
                    )
                    total += a["bytes"]
                actual = {
                    p.relative_to(base).as_posix()
                    for p in base.rglob("*")
                    if p.is_file()
                }
                require(
                    total > 0 and actual == acknowledged,
                    "checkpoint acknowledged file set mismatch",
                )
                if name == "R" and segment.get("status") == "checkpoint":
                    ack = _json(
                        _path(
                            root,
                            f"{name}/segment-{segment_index:03}/checkpoint-complete.json",
                        )
                    )
                    require(
                        isinstance(ack, dict)
                        and ack.get("checkpoint_complete") is True
                        and ack.get("cut") == cut
                        and ack.get("process_token") == segment.get("process_token"),
                        "checkpoint completion acknowledgment mismatch",
                    )
        required = list(range(len(cuts))) if name in ("P", "R") else []
        if arm.get("status") == "complete":
            require(
                found == required,
                f"{name} checkpoint acknowledgments do not cover scheduled cuts",
            )
        else:
            require(
                found == required[: len(found)],
                f"{name} checkpoint acknowledgment sequence is not a prefix",
            )


def _provenance(root, manifest, study, cuts, arms):
    metadata = manifest["metadata"]
    identity = metadata.get("identity")
    require(
        isinstance(identity, dict) and identity.get("driver") == study.get("driver"),
        "source identity driver disagrees with study",
    )
    require(
        metadata.get("termination") == study.get("termination", "normal"),
        "termination metadata disagrees with study",
    )
    for name, arm in arms.items():
        for i, segment in enumerate(arm.get("segments", [])):
            request = _json(_path(root, f"{name}/request-{i:03}.json"))
            require(
                isinstance(request, dict)
                and all(request.get(k) == value for k, value in study.items()),
                "worker request disagrees with frozen study",
            )
            require(
                request.get("arm") == name and request.get("cuts") == cuts,
                "worker request arm/schedule mismatch",
            )
            require(
                i <= (len(cuts) if name == "R" else 0),
                "worker request segment boundary mismatch",
            )
            expected_start = cuts[i - 1] if name == "R" and i else 0
            require(
                type(request.get("start_step")) is int
                and request["start_step"] == expected_start
                and type(request.get("next_cut")) is int
                and request["next_cut"] == (i if name == "R" else 0),
                "worker request segment boundary mismatch",
            )


def _same_raw(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and a.tobytes(order="C") == b.tobytes(order="C")
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict)
            and isinstance(b, dict)
            and set(a) == set(b)
            and all(_same_raw(a[k], b[k]) for k in a)
        )
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(_same_raw(x, y) for x, y in zip(a, b))
        )
    return type(a) is type(b) and a == b


def _segment_raw(root, arms, cache, budget):
    for name, arm in arms.items():
        segments = arm.get("segments", [])
        # A killed segment may have incomplete artifacts the coordinator did
        # not accept. Its outcome remains unknown; never promote it to complete.
        if arm.get("status") != "complete" or not segments:
            continue
        combined, last = [], None
        for i, segment in enumerate(segments):
            base = _path(root, f"{name}/segment-{i:03}")
            raw = _decode(base, _json(base / "result.json"), cache, budget)
            require(
                isinstance(raw, dict) and isinstance(raw.get("observations"), list),
                "invalid worker raw segment",
            )
            require(
                raw.get("status") == segment.get("status")
                and raw.get("process_token") == segment.get("process_token"),
                "worker raw segment identity/status mismatch",
            )
            combined.extend(raw["observations"])
            last = raw
        require(
            _same_raw(combined, arm.get("observations")),
            f"{name} aggregate observations disagree with worker raw segment arrays",
        )
        require(
            _same_raw(last.get("outputs"), arm.get("outputs")),
            f"{name} aggregate outputs disagree with worker raw segment arrays",
        )


def check_case(path, check_cached=True):
    """Fail closed on corruption; otherwise return independent adjudication."""
    root = Path(path).absolute()
    try:
        manifest = _verify(root)
        study = _json(root / "study.json")
        require(isinstance(study, dict), "invalid study")
        c = _contract(study.get("contract"))
        cache, budget = {}, [0]
        arms = {
            name: _decode(root, _json(root / f"{name}.json"), cache, budget)
            for name in ARMS
        }
        cuts = _json(root / "schedule.json")
        _cadence(study, cuts, arms)
        _checkpoint_ledger(root, cuts, arms)
        _provenance(root, manifest, study, cuts, arms)
        _segment_raw(root, arms, cache, budget)
        result = adjudicate(arms, c)
        if check_cached:
            _cached(result, _json(root / "adjudication.json"))
        return result
    except CheckError:
        raise
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise CheckError(f"malformed evidence: {exc}") from exc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case")
    parser.add_argument(
        "--ignore-cached",
        action="store_true",
        help="report the raw re-adjudication without accepting cached labels",
    )
    args = parser.parse_args()
    try:
        result = check_case(args.case, not args.ignore_cached)
    except CheckError as exc:
        print(
            json.dumps({"status": "evidence_integrity_error", "error": str(exc)}),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
