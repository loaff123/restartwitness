"""Check the public projection with standard-library arithmetic only.

These excerpts are newly hashed observation projections, not sealed runner
bundles. This checker cannot establish omitted observations or process history.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

KEYS = [
    [0, "ordinary", 0],
    [99, "ordinary", 0],
    [100, "ordinary", 0],
    [100, "before_save", 0],
    [100, "after_save", 0],
    [100, "after_restore", 0],
    [101, "ordinary", 0],
    [110, "ordinary", 0],
]
SHAPES = {
    "step": [],
    "time": [],
    "position": [10, 3],
    "velocity": [10, 3],
    "box": [3, 3],
}
UNITS = {
    "step": "step",
    "time": "ps",
    "position": "nm",
    "velocity": "nm/ps",
    "box": "nm",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decode(value, shape, integer=False):
    if shape:
        require(isinstance(value, list) and len(value) == shape[0], "invalid shape")
        return [item for child in value for item in decode(child, shape[1:], integer)]
    if integer:
        require(type(value) is int, "integer required")
        return [value]
    require(isinstance(value, str) and len(value) < 40, "hex float required")
    try:
        number = float.fromhex(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("invalid hex float") from exc
    require(
        math.isfinite(number) and number.hex() == value,
        "nonfinite or noncanonical float",
    )
    return [number]


def compare(left, right, mode):
    for field, shape in SHAPES.items():
        a = decode(left[field], shape, field == "step")
        b = decode(right[field], shape, field == "step")
        if mode == "exact":
            if left[field] != right[field]:
                return True
        else:
            for x, y in zip(a, b):
                if field == "step":
                    if x != y:
                        return True
                    continue
                x, y = Fraction.from_float(x), Fraction.from_float(y)
                tolerance = (
                    Fraction.from_float(1e-5)
                    if mode == "absolute"
                    else Fraction.from_float(1e-12) * (1 + max(abs(x), abs(y)))
                )
                if abs(x - y) > tolerance:
                    return True
    return False


def check_excerpt(data):
    require(
        data.get("format") == "restartwitness-observation-excerpt-v1", "wrong format"
    )
    require(data.get("shapes") == SHAPES and data.get("units") == UNITS, "wrong schema")
    require(
        re.fullmatch(r"[0-9a-f]{64}", data.get("source_manifest_sha256", ""))
        is not None,
        "invalid source digest",
    )
    arms = data["arms"]
    require(set(arms) == {"U1", "U2", "O", "P", "R"}, "wrong arms")
    for name, records in arms.items():
        expected = KEYS[-1:] if name == "O" else KEYS
        require(
            isinstance(records, list) and [r["key"] for r in records] == expected,
            "wrong observation keys",
        )
        for record in records:
            require(set(record["values"]) == set(SHAPES), "wrong fields")
            for field, shape in SHAPES.items():
                decode(record["values"][field], shape, field == "step")

    def differs(a, b, mode):
        left = arms[a][-1:] if b == "O" else arms[a]
        return any(
            compare(x["values"], y["values"], mode) for x, y in zip(left, arms[b])
        )

    controls = not any(
        differs(a, b, "exact") for a, b in [("U1", "U2"), ("U1", "O"), ("U1", "P")]
    )
    immediate = all(
        not compare(arms["P"][i]["values"], arms["R"][i]["values"], "exact")
        for i in range(6)
    )
    return {
        "case": data["case"],
        "controls_exact": controls,
        "immediate_restore_exact": immediate,
        "exact_difference": differs("P", "R", "exact"),
        "strict_1e-12_difference": differs("P", "R", "strict"),
        "absolute_1e-5_difference": differs("P", "R", "absolute"),
        "u_r_absolute_1e-5_difference": differs("U1", "R", "absolute"),
        "scope": "selected observation projection only",
    }


def read_json(path):
    require(
        not path.is_symlink() and path.stat().st_size <= 8_000_000,
        "unsafe or oversized file",
    )

    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out

    return json.loads(path.read_text(), object_pairs_hook=pairs)


def verify_directory(path):
    path = Path(path)
    manifest = read_json(path / "manifest.json")
    require(
        manifest.get("format") == "restartwitness-excerpt-manifest-v1", "wrong manifest"
    )
    files = manifest["files"]
    require(isinstance(files, dict) and len(files) == 4, "four excerpts required")
    require(
        {p.name for p in path.iterdir()} == set(files) | {"manifest.json"},
        "unexpected evidence files",
    )
    reports = []
    for name, expected in sorted(files.items()):
        require(re.fullmatch(r"[a-z0-9-]+\.json", name) is not None, "unsafe filename")
        file = path / name
        data = read_json(file)
        require(
            hashlib.sha256(file.read_bytes()).hexdigest() == expected,
            "excerpt hash mismatch",
        )
        reports.append(check_excerpt(data))
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path, default=Path(__file__).with_name("evidence")
    )
    args = parser.parse_args()
    print(json.dumps(verify_directory(args.directory), indent=2))


if __name__ == "__main__":
    main()
