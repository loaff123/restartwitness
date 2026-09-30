"""Deterministic schedule recipes retain boundary and repeated-cut controls."""

from importlib import import_module

import pytest


def boundary_schedules(*args, **kwargs):
    return import_module("restartwitness.schedules").boundary_schedules(*args, **kwargs)


def baseline_schedules(*args, **kwargs):
    return import_module("restartwitness.schedules").baseline_schedules(*args, **kwargs)


CANONICAL = [
    (0,),
    (1,),
    (7,),
    (8,),
    (9,),
    (31,),
    (32,),
    (33,),
    (63,),
    (64,),
    (1, 32, 63),
    (7, 8, 9),
    (8, 8),
]


def test_frozen_64_step_recipe_is_exact():
    assert boundary_schedules(64, [8, 32]) == CANONICAL


def test_boundary_order_and_duplicates_do_not_change_recipe():
    assert boundary_schedules(64, [32, 8, 32]) == CANONICAL


@pytest.mark.parametrize("limit", [0, 1, 10, 12, 13, 128])
def test_limit_is_an_exact_prefix_not_an_unreported_reordering(limit):
    assert boundary_schedules(64, [8, 32], limit) == CANONICAL[:limit]


@pytest.mark.parametrize(
    "total,boundaries",
    [(0, []), (0, [0]), (1, [0, 1]), (2, [0, 1, 2]), (3, []), (9, [0, 9])],
)
def test_small_horizons_and_endpoint_events_remain_valid(total, boundaries):
    schedules = boundary_schedules(total, boundaries)
    assert len(schedules) == len(set(schedules))
    assert (0,) in schedules and (total,) in schedules
    for cuts in schedules:
        assert cuts and tuple(sorted(cuts)) == cuts
        assert all(type(cut) is int and 0 <= cut <= total for cut in cuts)
    for boundary in boundaries:
        for cut in (boundary - 1, boundary, boundary + 1):
            if 0 <= cut <= total:
                assert (cut,) in schedules


def test_no_events_still_has_beginning_end_and_distributed_control():
    assert boundary_schedules(10, []) == [(0,), (1,), (9,), (10,), (1, 5, 9)]


@pytest.mark.parametrize(
    "total,boundaries,limit",
    [
        (-1, [], 2),
        (True, [], 2),
        (1.5, [], 2),
        (4, [-1], 2),
        (4, [5], 2),
        (4, [2.0], 2),
        (4, [True], 2),
        (4, [], -1),
        (4, [], True),
        (4, [], 1.5),
    ],
)
def test_invalid_bounds_are_rejected_even_when_generation_would_truncate(
    total, boundaries, limit
):
    with pytest.raises((TypeError, ValueError)):
        boundary_schedules(total, boundaries, limit)


def test_zero_limit_does_not_hide_an_invalid_event():
    with pytest.raises(ValueError):
        boundary_schedules(4, [5], 0)


def test_fixed_baseline_is_evenly_distributed_and_caps_unique_cuts():
    assert baseline_schedules("fixed", 8, 5) == [(0,), (2,), (4,), (6,), (8,)]
    assert baseline_schedules("fixed", 8, 1) == [(4,)]
    assert baseline_schedules("fixed", 2, 8) == [(0,), (1,), (2,)]
    assert baseline_schedules("fixed", 0, 0) == []


def test_random_baseline_is_seed_reproducible_and_does_not_use_global_rng():
    import random

    state = random.getstate()
    first = baseline_schedules("random", 64, 12, seed=271828)
    assert first == baseline_schedules("random", 64, 12, seed=271828)
    assert first != baseline_schedules("random", 64, 12, seed=9)
    assert len(first) == len(set(first)) == 12
    assert all(len(cuts) == 1 and 0 <= cuts[0] <= 64 for cuts in first)
    assert random.getstate() == state


@pytest.mark.parametrize(
    "method,total,count,seed",
    [
        ("unknown", 8, 3, 1),
        ("fixed", -1, 3, 1),
        ("random", 8, -1, 1),
        ("fixed", 8, True, 1),
        ("random", 8, 3, True),
    ],
)
def test_invalid_baseline_requests_are_rejected(method, total, count, seed):
    with pytest.raises((TypeError, ValueError)):
        baseline_schedules(method, total, count, seed)
