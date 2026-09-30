from pathlib import Path
import importlib.util
import pytest


def jobs(group):
    assert importlib.util.find_spec("experiments.evaluate"), "evaluation runner absent"
    from experiments.evaluate import load_protocol, build_jobs

    p = load_protocol(Path(__file__).parents[1] / "protocols/v1.json")
    return build_jobs(p, group)


@pytest.mark.parametrize(
    "group,count",
    [
        ("acceptance", 61),
        ("fixed", 60),
        ("random", 60),
        ("boundary", 60),
        ("conventional", 15),
    ],
)
def test_frozen_job_accounting(group, count):
    rows = jobs(group)
    assert len(rows) == count
    assert len({x["id"] for x in rows}) == count
    if group in ("fixed", "random", "boundary"):
        assert sum(5 * x["study"]["total_steps"] for x in rows) == 19200
        assert all(x["study"]["observations"] == list(range(65)) for x in rows)


def test_protocol_hash_is_required(tmp_path):
    jobs("fixed")
    from experiments.evaluate import load_protocol

    p = tmp_path / "v1.json"
    p.write_text("{}")
    p.with_suffix(".sha256").write_text("0" * 64 + " v1.json\n")
    with pytest.raises(ValueError):
        load_protocol(p)


def test_frozen_protocol_survives_autocrlf_checkout(tmp_path):
    """Git checkout must preserve frozen bytes even with Windows defaults."""
    import hashlib
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        pytest.skip("Git checkout regression requires Git")
    root = Path(__file__).parents[1]
    protocol = root / "protocols/v1.json"
    target = tmp_path / "protocols/v1.json"
    target.parent.mkdir()
    target.write_bytes(protocol.read_bytes())
    attributes = root / ".gitattributes"
    if attributes.exists():
        shutil.copyfile(attributes, tmp_path / ".gitattributes")

    def run(*args):
        subprocess.run([git, *args], cwd=tmp_path, check=True, capture_output=True)

    run("init", "-q")
    run("config", "core.autocrlf", "true")
    run("add", ".")
    target.unlink()
    run("checkout-index", "--all", "--force")
    expected = protocol.with_suffix(".sha256").read_text().split()[0]
    assert hashlib.sha256(target.read_bytes()).hexdigest() == expected
