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
