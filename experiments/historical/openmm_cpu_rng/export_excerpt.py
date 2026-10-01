"""Create a separately identified, path-free projection of verified local data.

Never rewrites the source bundle. Only whitelisted numeric observations and
public scientific configuration are exported; process/request/checkpoint files
are deliberately excluded. The result is not a complete runner evidence bundle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from restartwitness.runner import read_verified_case
from .verify_excerpt import KEYS, SHAPES, UNITS, check_excerpt


def encode(value):
    if isinstance(value, np.ndarray):
        return encode(value.tolist())
    if isinstance(value, list):
        return [encode(item) for item in value]
    if isinstance(value, float):
        return value.hex()
    if type(value) is int:
        return value
    raise ValueError("unsupported observation scalar")


def export_case(case_path, destination, name):
    if name == "manifest":
        raise ValueError("manifest is a reserved case name")
    if re.fullmatch(r"[a-z0-9-]+", name) is None:
        raise ValueError("case name must be a plain slug")
    case_path = Path(case_path).resolve()
    destination = Path(destination).absolute()
    if any(path.is_symlink() for path in (destination, *destination.parents)):
        raise ValueError("symlink destination is not supported")
    resolved_destination = destination.resolve()
    if (
        resolved_destination == case_path
        or case_path in resolved_destination.parents
        or resolved_destination in case_path.parents
    ):
        raise ValueError("source and destination must not overlap")
    target = destination / f"{name}.json"
    if target.exists() or target.is_symlink():
        raise ValueError("do not overwrite an existing excerpt")
    existing = sorted(destination.glob("*.json")) if destination.exists() else []
    for file in existing:
        if file.is_symlink():
            raise ValueError("symlink output file is not supported")
        if file.name != "manifest.json":
            check_excerpt(json.loads(file.read_text()))
    source = read_verified_case(case_path)
    if source["schedule"] != [100] or source["study"]["total_steps"] != 110:
        raise ValueError("projection supports the cut100/110-step profile only")
    config = source["study"]["config"]
    arms = {}
    for arm, content in source["arms"].items():
        arms[arm] = [
            {
                "key": row["key"],
                "values": {key: encode(row["fields"][key]) for key in SHAPES},
            }
            for row in content["observations"]
            if row["key"] in KEYS
        ]
    excerpt = {
        "format": "restartwitness-observation-excerpt-v1",
        "case": name,
        "source_manifest_sha256": source["manifest_sha256"],
        "derivation": "Selected observations encoded with float.hex; no interpolation, rounding, redaction or mutation of the sealed source bundle. This separate projection cannot verify omitted records or process provenance.",
        "configuration": {
            key: config[key]
            for key in ["package_version", "git_revision", "platform", "seed"]
        },
        "shapes": SHAPES,
        "units": UNITS,
        "arms": arms,
    }
    check_excerpt(excerpt)
    destination.mkdir(parents=True, exist_ok=True)
    with target.open("x") as output:
        output.write(json.dumps(excerpt, indent=2, allow_nan=False) + "\n")
    files = sorted(p for p in destination.glob("*.json") if p.name != "manifest.json")
    for file in files:
        check_excerpt(json.loads(file.read_text()))
    manifest = {
        "format": "restartwitness-excerpt-manifest-v1",
        "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    print(export_case(args.case, args.out, args.name).name)


if __name__ == "__main__":
    main()
