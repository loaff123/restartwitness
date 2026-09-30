"""Frozen, deterministic restart schedules; no inferred scientific events."""

from __future__ import annotations

from collections.abc import Iterable
import random


def _nonnegative_integer(value: int, name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer, not a boolean or fractional value")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def boundary_schedules(
    total_steps: int, boundaries: Iterable[int], limit: int = 128
) -> list[tuple[int, ...]]:
    """Return a prefix of the boundary recipe, preserving its frozen order.

    Singles cover start/end controls and valid before/on/after event cuts.
    They are followed by a distributed control, the earliest event's local
    neighborhood, and a repeated cut at that event. Ordinary controls with
    fewer than two distinct cuts are omitted; the repeated-cut control is
    always retained when an event exists. Caller must disclose truncation.
    """
    _nonnegative_integer(total_steps, "total_steps")
    _nonnegative_integer(limit, "limit")
    events = list(boundaries)
    for event in events:
        _nonnegative_integer(event, "boundary")
        if event > total_steps:
            raise ValueError("boundary must not exceed total_steps")
    events = sorted(set(events))
    singles = {0, min(1, total_steps), max(total_steps - 1, 0), total_steps}
    for event in events:
        singles.update(
            cut for cut in (event - 1, event, event + 1) if 0 <= cut <= total_steps
        )
    schedules = [(cut,) for cut in sorted(singles)]

    def add_control(cuts: Iterable[int], *, repeated: bool = False) -> None:
        control = tuple(cuts) if repeated else tuple(sorted(set(cuts)))
        if len(control) > 1 and control not in schedules:
            schedules.append(control)

    add_control((min(1, total_steps), total_steps // 2, max(total_steps - 1, 0)))
    if events:
        event = events[0]
        add_control(
            cut for cut in (event - 1, event, event + 1) if 0 <= cut <= total_steps
        )
        add_control((event, event), repeated=True)
    return schedules[:limit]


def baseline_schedules(
    method: str, total_steps: int, count: int, seed: int = 271828
) -> list[tuple[int, ...]]:
    """Return at most ``count`` unique single-cut baseline schedules.

    ``fixed`` uses integer-floor evenly spaced cuts including both endpoints
    (or the midpoint for one schedule). ``random`` samples without replacement
    using an isolated Python RNG. Requests cap at ``total_steps + 1``; these
    recipe counts alone do not establish equal simulated-work budgets.
    """
    _nonnegative_integer(total_steps, "total_steps")
    _nonnegative_integer(count, "count")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if method not in ("fixed", "random"):
        raise ValueError("method must be 'fixed' or 'random'")
    count = min(count, total_steps + 1)
    if count == 0:
        return []
    if method == "random":
        cuts = random.Random(seed).sample(range(total_steps + 1), count)
    elif count == 1:
        cuts = [total_steps // 2]
    else:
        cuts = [index * total_steps // (count - 1) for index in range(count)]
    return [(cut,) for cut in cuts]
