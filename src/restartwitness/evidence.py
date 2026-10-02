"""Bounded, pickle-free numeric evidence and integrity verification.

Hashes detect accidental modification, not a malicious author's forgery. Verification
never imports drivers or opens native checkpoints. Replay executes trusted code only.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import numpy as np

MAX_JSON = 4 * 1024 * 1024
MAX_ARRAY = 32 * 1024 * 1024
MAX_BUNDLE = 128 * 1024 * 1024
MAX_FILES = 20000


class IntegrityError(ValueError):
    """Malformed, unsafe or inconsistent evidence."""


def _pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise IntegrityError(f"duplicate JSON key: {k}")
        d[k] = v
    return d


def read_json(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise IntegrityError("unsafe or oversized JSON")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(
                IntegrityError("nonfinite JSON")
            ),
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise IntegrityError(f"invalid JSON: {exc}") from exc


def write_json(path, data):
    path = Path(path)
    raw = json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if len(raw.encode()) > MAX_JSON:
        raise IntegrityError("oversized JSON")
    path.write_text(raw, encoding="utf-8")


def safe_path(root, relative):
    root = Path(root).absolute()
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or ":" in relative
    ):
        raise IntegrityError("invalid relative path")
    p = PurePosixPath(relative)
    if p.is_absolute() or any(v in ("..", ".") for v in p.parts) or str(p) != relative:
        raise IntegrityError("unsafe evidence path")
    target = root.joinpath(*p.parts)
    if any(x.is_symlink() for x in (root, *root.parents)) or any(
        x.is_symlink()
        for x in (target, *target.parents)
        if x != root.parent and (x == root or root in x.parents)
    ):
        raise IntegrityError("symlink in evidence path")
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise IntegrityError("path escapes evidence root") from exc
    return target


def _numeric(a):
    if type(a) is not np.ndarray:
        raise IntegrityError("only plain numpy.ndarray values allowed")
    if a.dtype.kind not in "biuf" or a.dtype.hasobject:
        raise IntegrityError("only bool/integer/real numeric arrays allowed")
    if a.nbytes > MAX_ARRAY or a.ndim > 8:
        raise IntegrityError("oversized numeric array")
    return a


def write_data(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise IntegrityError("refusing to overwrite evidence")
    counter = 0

    def encode(v):
        nonlocal counter
        if isinstance(v, np.ndarray):
            a = _numeric(v)
            name = f"{path.stem}-{counter:05}.npy"
            counter += 1
            target = safe_path(path.parent, name)
            if target.exists():
                raise IntegrityError("array destination exists")
            with target.open("xb") as out:
                np.save(out, a, allow_pickle=False)
            return {"$array": name}
        if isinstance(v, dict):
            if "$array" in v:
                raise IntegrityError("reserved array key")
            if not all(isinstance(k, str) for k in v):
                raise IntegrityError("JSON keys must be strings")
            return {k: encode(x) for k, x in v.items()}
        if isinstance(v, (tuple, list)):
            return [encode(x) for x in v]
        if isinstance(v, np.generic):
            return v.item()
        return v

    write_json(path, encode(data))


def _load_array(path):
    if not path.is_file() or path.is_symlink():
        raise IntegrityError("nonregular array file")
    if path.stat().st_size > MAX_ARRAY + 65536:
        raise IntegrityError("oversized array file")
    try:
        with path.open("rb") as f:
            version = np.lib.format.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
            else:
                raise IntegrityError("unsupported npy version")
            if len(shape) > 8 or any(type(n) is not int or n < 0 for n in shape):
                raise IntegrityError("invalid array shape")
            size = math.prod(shape) * dtype.itemsize
            if dtype.kind not in "biuf" or dtype.hasobject or size > MAX_ARRAY:
                raise IntegrityError("unsafe array dtype/size")
            if path.stat().st_size != f.tell() + size:
                raise IntegrityError("truncated or trailing array bytes")
        return _numeric(np.load(path, allow_pickle=False))
    except (ValueError, EOFError, OSError) as exc:
        raise IntegrityError(f"invalid numeric file: {exc}") from exc


def read_data(path):
    path = Path(path)
    cache = {}

    def decode(v, depth=0):
        if depth > 40:
            raise IntegrityError("excessive evidence nesting")
        if isinstance(v, dict):
            if "$array" in v:
                if set(v) != {"$array"}:
                    raise IntegrityError("invalid array reference")
                target = safe_path(path.parent, v["$array"])
                if target not in cache:
                    cache[target] = _load_array(target)
                    cache[target].flags.writeable = False
                return cache[target]
            return {k: decode(x, depth + 1) for k, x in v.items()}
        if isinstance(v, list):
            return [decode(x, depth + 1) for x in v]
        return v

    return decode(read_json(path))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _files(root):
    root = Path(root)
    files = []
    size = 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise IntegrityError("bundle contains symlink")
        if p.is_dir():
            continue
        if not p.is_file():
            raise IntegrityError("nonregular artifact")
        if p.name == "manifest.json" and p.parent == root:
            continue
        size += p.stat().st_size
        if size > MAX_BUNDLE or len(files) >= MAX_FILES:
            raise IntegrityError("bundle limit exceeded")
        files.append(p)
    return files


def seal_bundle(root, metadata):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise IntegrityError("bundle already sealed")
    artifacts = [
        {
            "path": p.relative_to(root).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": digest(p),
        }
        for p in _files(root)
    ]
    write_json(
        root / "manifest.json",
        {"schema_version": 1, "metadata": metadata, "artifacts": artifacts},
    )
    return digest(root / "manifest.json")


def verify_bundle(root):
    root = Path(root)
    actual = {p.relative_to(root).as_posix() for p in _files(root)}
    m = read_json(root / "manifest.json")
    if (
        not isinstance(m, dict)
        or set(m) != {"schema_version", "metadata", "artifacts"}
        or type(m["schema_version"]) is not int
        or m["schema_version"] != 1
    ):
        raise IntegrityError("unsupported manifest")
    if (
        not isinstance(m["metadata"], dict)
        or not isinstance(m["artifacts"], list)
        or not m["artifacts"]
    ):
        raise IntegrityError("missing manifest data")
    expected = set()
    for a in m["artifacts"]:
        if not isinstance(a, dict) or set(a) != {"path", "bytes", "sha256"}:
            raise IntegrityError("invalid artifact metadata")
        p = safe_path(root, a["path"])
        if a["path"] in expected:
            raise IntegrityError("duplicate manifest path")
        expected.add(a["path"])
        if (
            not p.is_file()
            or type(a["bytes"]) is not int
            or p.stat().st_size != a["bytes"]
            or digest(p) != a["sha256"]
        ):
            raise IntegrityError(f"artifact hash/size mismatch: {a['path']}")
    if actual != expected:
        raise IntegrityError("manifest file set mismatch")
    return {
        "status": "verified",
        "files": len(expected),
        "manifest_sha256": digest(root / "manifest.json"),
        "metadata": m["metadata"],
    }
