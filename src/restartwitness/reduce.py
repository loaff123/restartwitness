"""Deletion-only reduction with repeatability and explicit minimality limits."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from typing import Any


def reduce_schedule(
    schedule: Iterable[int],
    evaluator: Callable[[tuple[int, ...]], dict[str, Any]],
    failure_signature: str,
    budget: int = 64,
) -> dict[str, Any]:
    """Reduce cuts, never the workload or comparison contract.

    The evaluator must execute a fresh, clean full-horizon experiment on each
    call. It returns JSON-compatible evidence with ``status`` and ``signature``.
    Only repeatable ``difference`` results with the requested signature can
    replace the candidate. Repeatable ``equivalent`` or a different, stable
    difference signature are conclusive non-preservation. All other outcomes,
    including exceptions and nonrepeatability, are unknown.

    ``budget`` counts evaluator calls, not candidate schedules. The caller owns
    wall-clock deadlines and all fixed experiment context. The returned trial
    log includes every call; unknowns anywhere conservatively prohibit a
    ``one_minimal`` claim. An unconfirmed input is returned unchanged with
    ``target_reproduced=False``, never labelled a reproduced witness.
    """
    original = tuple(schedule)
    if any(type(cut) is not int for cut in original):
        raise TypeError("cuts must be integers, not booleans or fractional values")
    if any(cut < 0 for cut in original) or tuple(sorted(original)) != original:
        raise ValueError("cuts must be nonnegative and nondecreasing")
    if type(budget) is not int:
        raise TypeError("budget must be an integer")
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if not isinstance(failure_signature, str) or not failure_signature.strip():
        raise ValueError("failure_signature must be a nonempty class/field signature")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")

    current = original
    trials: list[dict[str, Any]] = []
    unknown_observed = False
    budget_exhausted = False
    target_reproduced = False

    def finish(status: str = "smallest_observed") -> dict[str, Any]:
        return {
            "original": list(original),
            "candidate": list(current),
            "status": status,
            "failure_signature": failure_signature,
            "target_reproduced": target_reproduced,
            "trials": trials,
            "evaluations": len(trials),
            "budget": budget,
            "budget_exhausted": budget_exhausted,
            "unknown_observed": unknown_observed,
        }

    def token(outcome: Any) -> tuple[str, ...]:
        if not isinstance(outcome, dict):
            return ("unknown",)
        status, signature = outcome.get("status"), outcome.get("signature")
        if status == "difference" and isinstance(signature, str) and signature:
            return (
                ("target", signature)
                if signature == failure_signature
                else ("absent", status, signature)
            )
        if status == "equivalent" and "signature" in outcome and signature is None:
            return ("absent", status)
        return ("unknown",)

    def evaluate(cuts: tuple[int, ...], phase: str) -> str:
        nonlocal unknown_observed, budget_exhausted
        outcomes = []
        for repeat in (1, 2):
            if len(trials) >= budget:
                budget_exhausted = True
                return "budget"
            try:
                raw = evaluator(cuts)
                # Snapshot mutable return values and reject non-JSON evidence.
                try:
                    outcome = json.loads(json.dumps(raw, allow_nan=False))
                except (TypeError, ValueError, OverflowError):
                    outcome = {
                        "status": "evaluator_error",
                        "signature": None,
                        "error": "Evaluator returned non-JSON evidence",
                        "result_type": type(raw).__name__,
                    }
            except Exception as exc:
                outcome = {
                    "status": "evaluator_error",
                    "signature": None,
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                }
            outcome_token = token(outcome)
            if outcome_token[0] == "unknown":
                unknown_observed = True
            outcomes.append(outcome_token)
            trials.append(
                {
                    "evaluation": len(trials) + 1,
                    "schedule": list(cuts),
                    "phase": phase,
                    "repeat": repeat,
                    "outcome": outcome,
                }
            )
        if outcomes[0] != outcomes[1] or outcomes[0][0] == "unknown":
            unknown_observed = True
            return "unknown"
        return outcomes[0][0]

    if evaluate(current, "initial") != "target":
        return finish()
    target_reproduced = True

    # Deterministic ddmin-style chunk deletion; no monotonicity assumptions.
    granularity = 2
    while len(current) >= 2:
        accepted = False
        for part in range(granularity):
            start = part * len(current) // granularity
            end = (part + 1) * len(current) // granularity
            proposal = current[:start] + current[end:]
            verdict = evaluate(proposal, "ddmin")
            if verdict == "budget":
                return finish()
            if verdict == "target":
                current = proposal
                granularity = min(len(current), max(2, granularity - 1))
                accepted = True
                break
        if not accepted:
            if granularity == len(current):
                break
            granularity = min(len(current), 2 * granularity)

    # Retest every individual position, including equivalent duplicate-cut
    # positions. A newly accepted removal starts an entirely fresh final pass.
    while True:
        accepted = False
        for position in range(len(current)):
            proposal = current[:position] + current[position + 1 :]
            verdict = evaluate(proposal, "single_removal")
            if verdict == "budget":
                return finish()
            if verdict == "target":
                current = proposal
                accepted = True
                break
        if not accepted:
            break

    # The input's two confirmations suffice when it was already empty; every
    # other candidate is independently reconfirmed after the final sweep.
    if original or current:
        verdict = evaluate(current, "final_confirmation")
        if verdict != "target":
            if verdict != "budget":
                target_reproduced = False
            return finish()
    return finish("smallest_observed" if unknown_observed else "one_minimal")
