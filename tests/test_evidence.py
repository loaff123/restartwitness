import json
from pathlib import Path
import numpy as np
import pytest


def api():
    import importlib.util

    assert importlib.util.find_spec("restartwitness.evidence"), (
        "evidence module missing"
    )
    from restartwitness import evidence

    return evidence


def test_numeric_roundtrip_and_digest(tmp_path):
    e = api()
    data = {
        "observations": [
            {"fields": {"x": np.array([1.0, 2.0]), "i": np.array(8, dtype=np.int64)}}
        ]
    }
    e.write_data(tmp_path / "arm.json", data)
    got = e.read_data(tmp_path / "arm.json")
    assert (
        got["observations"][0]["fields"]["x"].tobytes()
        == data["observations"][0]["fields"]["x"].tobytes()
    )
    e.seal_bundle(tmp_path, {"schema_version": 1, "kind": "test"})
    assert e.verify_bundle(tmp_path)["status"] == "verified"
    f = next(tmp_path.glob("*.npy"))
    f.write_bytes(f.read_bytes()[:-1])
    with pytest.raises(e.IntegrityError):
        e.verify_bundle(tmp_path)


def test_reject_duplicate_keys_path_escape_and_object_dtype(tmp_path):
    e = api()
    f = tmp_path / "bad.json"
    f.write_text('{"x":1,"x":2}')
    with pytest.raises(e.IntegrityError):
        e.read_json(f)
    with pytest.raises(e.IntegrityError):
        e.safe_path(tmp_path, "../escape")
    with pytest.raises(e.IntegrityError):
        e.write_data(tmp_path / "obj.json", {"x": np.array([{}], dtype=object)})


def test_extra_file_and_symlink_are_not_verified(tmp_path):
    e = api()
    e.write_data(tmp_path / "data.json", {"x": np.array([1])})
    e.seal_bundle(tmp_path, {})
    (tmp_path / "extra.txt").write_text("new")
    with pytest.raises(e.IntegrityError):
        e.verify_bundle(tmp_path)
    (tmp_path / "extra.txt").unlink()
    (tmp_path / "link").symlink_to(tmp_path / "data.json")
    with pytest.raises(e.IntegrityError):
        e.verify_bundle(tmp_path)


def test_declared_huge_npy_rejected_before_allocation(tmp_path):
    e = api()
    f = tmp_path / "big.npy"
    with f.open("wb") as out:
        np.lib.format.write_array_header_1_0(
            out, {"descr": "<f8", "fortran_order": False, "shape": (10**12,)}
        )
    (tmp_path / "data.json").write_text(json.dumps({"x": {"$array": "big.npy"}}))
    with pytest.raises(e.IntegrityError):
        e.read_data(tmp_path / "data.json")


def test_repeated_array_references_share_bounded_read_storage(tmp_path):
    e = api()
    np.save(tmp_path / "x.npy", np.ones(8))
    (tmp_path / "data.json").write_text(json.dumps([{"$array": "x.npy"}] * 100))
    got = e.read_data(tmp_path / "data.json")
    assert got[0] is got[-1]


def test_symlink_ancestor_rejected(tmp_path):
    e = api()
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "inside").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(e.IntegrityError):
        e.safe_path(tmp_path / "alias" / "inside", "file")


def test_bound_inventory_precedes_hashing_untrusted_artifacts(tmp_path, monkeypatch):
    e = api()
    p = tmp_path / "large.bin"
    with p.open("wb") as f:
        f.truncate(e.MAX_BUNDLE + 1)
    e.write_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "metadata": {},
            "artifacts": [
                {"path": "large.bin", "bytes": e.MAX_BUNDLE + 1, "sha256": "0" * 64}
            ],
        },
    )

    def unexpected(*a):
        raise AssertionError("hashed oversized untrusted data")

    monkeypatch.setattr(e, "digest", unexpected)
    with pytest.raises(e.IntegrityError):
        e.verify_bundle(tmp_path)


@pytest.mark.parametrize("loader", ["read_json", "_load_array", "_files"])
def test_nonregular_json_and_array_are_rejected_before_read(
    tmp_path, monkeypatch, loader
):
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO is a POSIX-only input type")
    e = api()
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    original_read = Path.read_text

    def guarded_read(path, *args, **kwargs):
        if path == fifo:
            raise AssertionError("attempted to read a FIFO")
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == fifo:
            raise AssertionError("attempted to open a FIFO")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(e.IntegrityError):
        getattr(e, loader)(tmp_path if loader == "_files" else fifo)


def test_nonportable_colon_path_is_rejected(tmp_path):
    e = api()
    with pytest.raises(e.IntegrityError):
        e.safe_path(tmp_path, "file:stream")
