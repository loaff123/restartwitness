"""Reduction only claims a repeatable, locally deletion-minimal witness."""

from collections import Counter
from importlib import import_module
import json

import pytest


SIGNATURE = "restore_path_difference:position"


def reduce_schedule(*args, **kwargs):
    return import_module("restartwitness.reduce").reduce_schedule(*args, **kwargs)


def result_for(preserves):
    return {
        "status": "difference" if preserves else "equivalent",
        "signature": SIGNATURE if preserves else None,
    }


def test_reduces_to_repeatable_failure_and_checks_every_final_single_removal():
    calls = []

    def evaluator(cuts):
        calls.append(cuts)
        return result_for(8 in cuts)

    result = reduce_schedule([1, 8, 9, 32], evaluator, SIGNATURE)
    assert result["original"] == [1, 8, 9, 32]
    assert result["candidate"] == [8]
    assert result["status"] == "one_minimal"
    assert result["evaluations"] == len(calls) == len(result["trials"])
    assert calls.count((8,)) >= 2
    assert calls.count(()) >= 2
    assert all(isinstance(cuts, tuple) for cuts in calls)
    json.dumps(result, allow_nan=False)


def test_nonmonotonic_predicate_is_not_assumed_upward_closed_or_globally_minimal():
    failures = {(1, 2, 3, 4), (3, 4), (2,)}
    result = reduce_schedule(
        (1, 2, 3, 4), lambda cuts: result_for(cuts in failures), SIGNATURE
    )
    candidate = tuple(result["candidate"])
    assert candidate in failures
    assert result["status"] == "one_minimal"
    assert all(
        candidate[:i] + candidate[i + 1 :] not in failures
        for i in range(len(candidate))
    )


def test_repeated_cuts_are_distinct_deletion_positions_and_not_deduplicated():
    result = reduce_schedule(
        [8, 8, 9], lambda cuts: result_for(cuts.count(8) >= 2), SIGNATURE
    )
    assert result["candidate"] == [8, 8]
    assert result["status"] == "one_minimal"
    final_removals = [
        trial for trial in result["trials"] if trial["phase"] == "single_removal"
    ]
    assert len([trial for trial in final_removals if trial["schedule"] == [8]]) == 4


def test_fixed_experiment_context_is_only_captured_by_evaluator_never_rewritten():
    config = {
        "total_steps": 64,
        "seed": 271828,
        "observation_steps": [0, 8, 64],
        "contract": {"atol": 0.01},
        "termination": "normal",
    }
    before = json.dumps(config, sort_keys=True)
    original = [0, 8, 64]

    def evaluator(cuts):
        assert json.dumps(config, sort_keys=True) == before
        assert all(cut in original for cut in cuts)
        return result_for(8 in cuts)

    result = reduce_schedule(original, evaluator, SIGNATURE)
    assert original == [0, 8, 64]
    assert result["candidate"] == [8]
    assert json.dumps(config, sort_keys=True) == before


@pytest.mark.parametrize(
    "status", ["timeout", "driver_error", "unsupported", "checkpoint_error"]
)
def test_unknown_single_removals_prevent_minimality_and_never_replace_failure(status):
    def evaluator(cuts):
        return (
            result_for(True)
            if cuts
            else {"status": status, "signature": None, "detail": "retained"}
        )

    result = reduce_schedule([8], evaluator, SIGNATURE)
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"
    assert any(
        trial["outcome"].get("detail") == "retained" for trial in result["trials"]
    )


def test_different_failure_signature_cannot_be_accepted():
    def evaluator(cuts):
        return (
            result_for(True)
            if cuts == (1, 8)
            else {"status": "difference", "signature": "save_path_difference:energy"}
        )

    result = reduce_schedule((1, 8), evaluator, SIGNATURE)
    assert result["candidate"] == [1, 8]
    assert result["status"] == "one_minimal"


def test_two_clean_trials_are_required_before_any_candidate_is_accepted():
    counts = Counter()

    def evaluator(cuts):
        counts[cuts] += 1
        return result_for(cuts == (1, 8) or counts[cuts] % 2 == 1)

    result = reduce_schedule((1, 8), evaluator, SIGNATURE)
    assert result["candidate"] == [1, 8]
    assert result["status"] == "smallest_observed"


def test_mixed_absent_outcomes_are_nonrepeatable_and_block_minimality():
    count = 0

    def evaluator(cuts):
        nonlocal count
        if cuts:
            return result_for(True)
        count += 1
        return (
            result_for(False)
            if count % 2
            else {"status": "difference", "signature": "other:field"}
        )

    result = reduce_schedule([8], evaluator, SIGNATURE)
    assert result["status"] == "smallest_observed"


@pytest.mark.parametrize("budget", [0, 1, 2, 3])
def test_budget_is_a_strict_call_limit_and_cannot_prove_minimality(budget):
    calls = []

    def evaluator(cuts):
        calls.append(cuts)
        return result_for(bool(cuts))

    result = reduce_schedule([8], evaluator, SIGNATURE, budget=budget)
    assert result["evaluations"] == len(calls) <= budget
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"


def test_budget_ending_after_acceptance_retains_smallest_confirmed_candidate():
    result = reduce_schedule(
        [1, 8], lambda cuts: result_for(8 in cuts), SIGNATURE, budget=4
    )
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"
    assert result["evaluations"] == 4


def test_budget_blocks_final_recheck_without_erasing_prior_reproduction():
    result = reduce_schedule(
        [8], lambda cuts: result_for(bool(cuts)), SIGNATURE, budget=4
    )
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"
    assert result["budget_exhausted"] is True
    assert result["target_reproduced"] is True


def test_unconfirmed_original_is_reported_without_searching_or_claiming_failure():
    result = reduce_schedule([8], lambda cuts: result_for(False), SIGNATURE)
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"
    assert result["target_reproduced"] is False
    assert result["evaluations"] == 2


def test_empty_schedule_can_be_vacuously_one_minimal_only_after_reproduction():
    result = reduce_schedule([], lambda cuts: result_for(True), SIGNATURE)
    assert result["candidate"] == []
    assert result["status"] == "one_minimal"
    assert result["evaluations"] >= 2


def test_evaluator_exceptions_are_retained_as_unknown_trials():
    def evaluator(cuts):
        if not cuts:
            raise TimeoutError("worker missed deadline")
        return result_for(True)

    result = reduce_schedule([8], evaluator, SIGNATURE)
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"
    assert any(
        trial["outcome"].get("exception_type") == "TimeoutError"
        for trial in result["trials"]
    )
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "bad_outcome",
    [
        None,
        {},
        {"status": "difference", "signature": None},
        {"status": "equivalent", "signature": SIGNATURE},
    ],
)
def test_malformed_evaluator_results_are_unknown(bad_outcome):
    result = reduce_schedule(
        [8], lambda cuts: result_for(True) if cuts else bad_outcome, SIGNATURE
    )
    assert result["candidate"] == [8]
    assert result["status"] == "smallest_observed"


@pytest.mark.parametrize(
    "cuts,budget,signature",
    [
        ([8, 1], 64, SIGNATURE),
        ([-1], 64, SIGNATURE),
        ([1.0], 64, SIGNATURE),
        ([True], 64, SIGNATURE),
        ([1], -1, SIGNATURE),
        ([1], True, SIGNATURE),
        ([1], 64, ""),
        ([1], 64, None),
    ],
)
def test_invalid_requests_fail_before_calling_evaluator(cuts, budget, signature):
    def evaluator(cuts):
        pytest.fail("invalid input reached evaluator")

    with pytest.raises((ValueError, TypeError)):
        reduce_schedule(cuts, evaluator, signature, budget)
