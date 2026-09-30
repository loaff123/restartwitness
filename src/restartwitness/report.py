"""Offline, path-only views of independently re-adjudicated case evidence.

Rendering never imports a driver, executes a checkpoint, fetches remote assets or
trusts a supplied verdict. Integrity errors propagate to the caller. Hashes are
integrity checks, not signatures or scientific certification.
"""

from __future__ import annotations

from collections import Counter
import html
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

from .runner import read_verified_case

__all__ = ["render_report", "render_junit"]

_PAIRS = ("U1-U2", "U1-O", "U-P", "P-R", "U-R")
_ARMS = ("U1", "U2", "O", "P", "R")
_PAIR_LABELS = {
    "U1-U2": ("Repeated baseline", "Two independent uninterrupted runs"),
    "U1-O": ("Observation control", "Final state and outputs with fewer observations"),
    "U-P": ("Save-path contrast", "Uninterrupted versus save and continue"),
    "P-R": ("Restore-path contrast", "Save and continue versus fresh-process restore"),
    "U-R": ("Direct restart contrast", "Uninterrupted versus fresh-process restore"),
}
_ERRORS = {
    "driver_error",
    "checkpoint_error",
    "timeout",
    "evidence_integrity_error",
    "incomplete",
    "invalid",
    "invalid_contract",
}
_SKIPS = {"baseline_inconclusive", "observation_inconclusive", "unsupported"}
_FAILURES = {
    "save_path_difference",
    "restore_path_difference",
    "restart_difference",
    "output_contract_violation",
}

_CSS = """
:root{color-scheme:light;--navy:#142b3a;--ink:#172e3c;--muted:#526471;--paper:#f6f4ed;
--line:#d7dedb;--green:#17614d;--amber:#805519;--red:#993e36;--mono:ui-monospace,SFMono-Regular,Consolas,monospace}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:inherit}a:focus-visible,summary:focus-visible{outline:3px solid #b5740a;outline-offset:5px}
.skip{position:absolute;top:-5em;left:1em;background:white;padding:.5em 1em;z-index:9}.skip:focus{top:.5em}
.topbar{background:var(--navy);color:#f7f5ed;border-bottom:1px solid #425465}.topbar>div{max-width:1320px;margin:auto;padding:20px 36px;display:flex;gap:24px;justify-content:space-between;align-items:center}
.brand{font-weight:750;letter-spacing:-.02em}.brand-mark{display:inline-block;color:#b9dfc9;margin-right:10px;font:800 20px var(--mono)}nav{display:flex;gap:24px;font-size:13px}nav a{text-decoration:none}nav a:hover{text-decoration:underline}
main{max-width:1320px;margin:auto;padding:0 36px 60px}.eyebrow{text-transform:uppercase;font:700 11px/1.5 var(--mono);letter-spacing:.14em;color:var(--green);margin:0 0 16px}
.hero{display:grid;grid-template-columns:1.25fr 1fr;gap:70px;padding:60px 0 40px;align-items:center}h1{font-size:clamp(40px,5vw,66px);line-height:1.05;letter-spacing:-.06em;margin:0 0 22px}h1 span{color:#66837b}h2{font-size:27px;letter-spacing:-.035em;line-height:1.25;margin:0 0 12px}h3{font-size:18px;line-height:1.3;margin:0 0 8px}p{margin:0 0 14px}.intro{font-size:18px;max-width:620px;color:var(--muted)}.small,.hint{font-size:13px;color:var(--muted)}.tag{display:inline-block;padding:5px 10px;border:1px solid #c4cfc8;border-radius:4px;font:600 11px var(--mono);margin-top:8px}
.verdict{border:1px solid var(--line);background:#fffefa;padding:28px;border-top:4px solid var(--green);box-shadow:0 8px 30px #142b3a08}.verdict.error,.verdict.failure{border-top-color:var(--red)}.verdict.skipped{border-top-color:var(--amber)}.verdict .status{font:700 19px/1.5 var(--mono);overflow-wrap:anywhere;margin:12px 0}.verdict ul{margin:12px 0 0;padding-left:20px}.badges{display:flex;gap:8px;flex-wrap:wrap}.badge{font:600 11px/1.5 var(--mono);padding:4px 7px;border-radius:4px;background:#e7f0e9;color:var(--green);overflow-wrap:anywhere}.badge.bad{background:#fbece7;color:var(--red)}.badge.warn{background:#faf0db;color:var(--amber)}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);border:1px solid var(--line);margin-bottom:40px;background:#eeefe8}.metric{padding:20px 24px;border-right:1px solid var(--line)}.metric:last-child{border:0}.metric strong{display:block;font:600 28px/1.3 var(--mono)}.metric span{font-size:12px;color:var(--muted)}
section{scroll-margin-top:24px;margin:42px 0}.section-head{display:flex;align-items:start;justify-content:space-between;gap:30px;margin-bottom:20px}.section-head p{max-width:570px}.section-no{color:#698079;font:13px var(--mono);margin-right:9px}.grid-three{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.grid-two{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}.card{border:1px solid var(--line);padding:24px;background:#fffefa;min-width:0}.pair{font:600 13px var(--mono);color:var(--green);margin-bottom:14px}.card .substatus{font:600 12px var(--mono);margin:14px 0}.card .witness-small{font-size:13px;overflow-wrap:anywhere;margin-top:16px;padding-top:16px;border-top:1px solid var(--line)}
.witness{padding:28px;background:#fffefa;border:1px solid var(--line);border-left:4px solid #bb7356}.witness.good{border-left-color:var(--green)}dl.facts{display:grid;grid-template-columns:minmax(130px,1fr) 3fr;gap:8px 20px;margin:14px 0 0}dt{font-size:12px;font-weight:650;color:var(--muted)}dd{margin:0;overflow-wrap:anywhere;font:13px/1.7 var(--mono)}
.plot-panel{background:var(--navy);padding:28px;color:#f6f3e9;border-radius:4px}.plot-panel .hint{color:#bdc9cc}.plot-panel svg{display:block;width:100%;height:auto;min-height:220px;margin:8px 0}.plot-panel text{font-family:var(--mono);font-size:12px;fill:#c8d2d3}.legend{display:flex;gap:24px;flex-wrap:wrap;font:12px var(--mono);margin-top:12px}.legend span::before{content:"";display:inline-block;width:30px;margin:0 9px 3px 0;border-top:3px solid #bae6cc}.legend .p::before{border-color:#e6ba82;border-top-style:dashed}.legend .r::before{border-color:#9cbde6;border-top-style:dotted}
.schedule{font:13px/1.7 var(--mono);word-break:break-word}.table-wrap{overflow-x:auto;border:1px solid var(--line);background:#fffefa}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}th{font-size:11px;letter-spacing:.04em;text-transform:uppercase;background:#ecefe8;color:#465b64}th,td{padding:13px 16px;border-bottom:1px solid #e2e5de;vertical-align:top}tbody tr:last-child td{border-bottom:0}td{overflow-wrap:anywhere}td code{white-space:normal}code{font-family:var(--mono);font-size:.88em;overflow-wrap:anywhere}pre{font:12px/1.7 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere;margin:0;max-width:100%}.terminal{background:#142b3a;color:#dcebdc;border-radius:4px;padding:18px 20px;margin:16px 0;overflow-wrap:anywhere}details{margin-top:16px}summary{cursor:pointer;font-size:13px;color:var(--green);font-weight:650;padding:8px 0}.details-body{margin-top:12px}.error-text{color:var(--red)}.limitations{padding:26px;background:#e9eee6;border:1px solid #cbd6ca}.limitations li{margin:6px 0}.limitations ul{padding-left:20px;margin-bottom:0}.hash{font:11px/1.7 var(--mono);word-break:break-all}.footer{border-top:1px solid var(--line);padding-top:22px;color:var(--muted);font-size:12px;display:flex;justify-content:space-between;gap:24px}.record-list{font:12px/1.7 var(--mono)}
@media(max-width:850px){.hero{grid-template-columns:1fr;gap:24px;padding-top:36px}.grid-three{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.metric:nth-child(2){border-right:0}.metric:nth-child(-n+2){border-bottom:1px solid var(--line)}nav{gap:14px}.section-head{display:block}.section-head p{margin-top:10px}}
@media(max-width:540px){main{padding:0 18px 35px}.topbar>div{padding:16px 18px;display:block}nav{margin-top:12px}.grid-two{grid-template-columns:1fr}.metric{padding:16px}.metric strong{font-size:24px}.card,.witness,.plot-panel,.verdict{padding:20px}dl.facts{grid-template-columns:1fr;gap:3px}dd{margin-bottom:7px}.footer{display:block}h1{font-size:42px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white;font-size:11px}.topbar nav,.skip{display:none}main{padding:0}.hero{padding-top:20px;gap:20px}h1{font-size:34px}.card,.witness,.verdict,.plot-panel{break-inside:avoid}.plot-panel{print-color-adjust:exact}details{display:block}section{margin:24px 0}.metrics{margin-bottom:20px}}
"""


def _load(bundle_path: str | os.PathLike) -> dict:
    if not isinstance(bundle_path, (str, os.PathLike)):
        raise TypeError(
            "report rendering requires an evidence bundle path, not a supplied verdict"
        )
    return read_verified_case(bundle_path)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _e(value: Any) -> str:
    return html.escape(_text(value), quote=True)


def _redact_paths(value: str) -> str:
    # Only used for free-form configuration/error text. Scientific field names,
    # unit labels and exact witnesses retain their spelling.
    value = re.sub(
        r'(?<![\w<])(?:[A-Za-z]:[\\/]|\\\\)[^\s<>"\']+', "[local path omitted]", value
    )
    return re.sub(
        r'(?<![\w<])/(?:[^\s<>"\'/]+/)*[^\s<>"\'/]+', "[local path omitted]", value
    )


def _summary(value: Any, depth: int = 0) -> Any:
    """Bound display material without serializing numeric observation arrays."""
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if depth > 6:
        return "[nested content omitted]"
    if isinstance(value, dict):
        items = list(value.items())
        result = {
            _redact_paths(str(k))[:300]: _summary(v, depth + 1) for k, v in items[:32]
        }
        if len(items) > 32:
            result["…"] = f"{len(items) - 32} more entries"
        return result
    if isinstance(value, (list, tuple)):
        result = [_summary(x, depth + 1) for x in value[:32]]
        if len(value) > 32:
            result.append(f"… {len(value) - 32} more entries")
        return result
    if isinstance(value, str):
        safe = _redact_paths(value)
        return safe[:2000] + ("… [truncated]" if len(safe) > 2000 else "")
    return value


def _pretty(value: Any) -> str:
    return json.dumps(_summary(value), indent=2, ensure_ascii=False, allow_nan=False)


def _outcome(case: dict) -> str:
    findings = set(case["adjudication"]["findings"])
    if findings & _ERRORS:
        return "error"
    if findings & _SKIPS:
        return "skipped"
    if findings & _FAILURES:
        return "failure"
    if findings == {"equivalent_under_contract"} and all(
        case["arms"][n]["status"] == "complete" for n in _ARMS
    ):
        return "pass"
    return "error"  # Unknown future classifications fail closed.


def _badge(status: str) -> str:
    tone = (
        ""
        if status in ("equivalent", "complete", "equivalent_under_contract")
        else (" warn" if status in _SKIPS else " bad")
    )
    return f'<span class="badge{tone}">{_e(status)}</span>'


def _facts(rows: list[tuple[str, Any]]) -> str:
    return (
        '<dl class="facts">'
        + "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for k, v in rows)
        + "</dl>"
    )


def _witness_rows(witness: dict | None) -> list[tuple[str, Any]]:
    if not witness:
        return []
    names = [
        ("comparison", "Comparison"),
        ("kind", "Diagnostic"),
        ("key", "Step / phase / occurrence"),
        ("right_key", "Right observation key"),
        ("field", "Field"),
        ("index", "Index"),
        ("left", "Left value"),
        ("right", "Right value"),
        ("left_bits", "Left IEEE bits"),
        ("right_bits", "Right IEEE bits"),
        ("absolute_difference", "Absolute difference"),
        ("relative_difference", "Relative difference"),
        ("absolute_difference_overflow", "Absolute difference overflow"),
        ("table", "Output table"),
        ("record_key", "Record identity"),
        ("side", "Affected side"),
        ("keys", "Affected record identities"),
        ("expected", "Expected"),
        ("actual", "Observed"),
        ("arm_status", "Worker status"),
        ("nonfinite_count", "Nonfinite count"),
        ("value", "Value"),
    ]
    return [(label, witness[key]) for key, label in names if key in witness]


def _witness_html(witness: dict | None) -> str:
    if not witness:
        return '<p class="hint">No observed difference under the declared contract.</p>'
    return _facts(_witness_rows(witness))


def _comparison_card(label: str, result: dict) -> str:
    title, description = _PAIR_LABELS[label]
    witness = result.get("first_witness")
    short = '<p class="hint">No observed difference</p>'
    if witness:
        short = '<div class="witness-small">' + _witness_html(witness) + "</div>"
    return (
        f'<article class="card"><div class="pair">{_e(label)}</div><h3>{title}</h3>'
        f'<p class="small">{description}</p>{_badge(result["status"])}'
        f'<p class="substatus">Trajectory: {_e(result["trajectory"]["status"])}<br>'
        f"Output records: {_e(result['outputs']['status'])}</p>{short}</article>"
    )


def _trajectory(case: dict) -> str:
    """A bounded scalar illustration; adjudication always uses all raw arrays."""
    contract = case["study"]["contract"]
    field = next(
        (f for f in contract["fields"] if f["name"] == "x" and not f["shape"]), None
    )
    if field is None:
        field = next(
            (
                f
                for f in contract["fields"]
                if f["name"] == "position" and math.prod(f["shape"]) > 0
            ),
            None,
        )
    if field is None:
        return '<p class="hint">No scalar x or nonempty position field is declared for the trajectory illustration. Exact evidence remains available above.</p>'
    series = {}
    total_points = 0
    for label, name in [("U", "U1"), ("P", "P"), ("R", "R")]:
        points = []
        for observation in case["arms"][name].get("observations", []):
            # Plot ordinary samples only: before/after-save distinctions are
            # shown as exact phase witnesses, never visually interpolated away.
            if observation.get("key", [None, None])[1] != "ordinary":
                continue
            array = observation.get("fields", {}).get(field["name"])
            if (
                not isinstance(array, np.ndarray)
                or not array.size
                or array.dtype.kind not in "biuf"
            ):
                continue
            value = float(array.flat[0])
            if math.isfinite(value):
                points.append((observation["key"][0], value))
        total_points += len(points)
        if len(points) > 600:
            indices = np.linspace(0, len(points) - 1, 600, dtype=int)
            points = [points[int(i)] for i in indices]
        series[label] = points
    all_points = [p for points in series.values() for p in points]
    if not all_points:
        return '<p class="hint">No finite ordinary samples are available for a trajectory illustration.</p>'
    ymin = min(p[1] for p in all_points)
    ymax = max(p[1] for p in all_points)
    # Normalize before subtraction to keep huge finite magnitudes drawable.
    scale = max(abs(ymin), abs(ymax), 1e-300)
    low, high = ymin / scale, ymax / scale
    if low == high:
        low -= 0.1
        high += 0.1
    span = high - low
    n = max(1, case["study"]["total_steps"])

    def x(step):
        return 80 + 870 * (step / n)

    def y(value):
        return 230 - 170 * ((value / scale - low) / span)

    unit = field["unit"] or "unitless"
    name = field["name"] + ("[0, …]" if field["shape"] else "")
    plot = [
        '<svg viewBox="0 0 1000 290" role="img" aria-labelledby="trajectory-title trajectory-desc">',
        f'<title id="trajectory-title">U, P and R ordinary trajectory: {_e(name)}</title>',
        '<desc id="trajectory-desc">Numeric ordinary observations versus logical step. U is a solid green line, P a dashed amber line, R a dotted blue line. Vertical markers are checkpoint cuts. Exact phase differences are listed separately.</desc>',
    ]
    for fraction in (0, 0.5, 1):
        yp = 230 - 170 * fraction
        value = (low + fraction * span) * scale
        plot.append(
            f'<line x1="80" x2="950" y1="{yp:.2f}" y2="{yp:.2f}" stroke="#405361" stroke-width="1"/>'
        )
        plot.append(
            f'<text x="68" y="{yp + 4:.2f}" text-anchor="end">{value:.3g}</text>'
        )
    for cut, count in Counter(case["schedule"]).items():
        xp = x(cut)
        plot.append(
            f'<line x1="{xp:.2f}" x2="{xp:.2f}" y1="42" y2="238" stroke="#afaa95" stroke-dasharray="3 6" opacity=".7"/>'
        )
        plot.append(
            f'<text x="{xp:.2f}" y="28" text-anchor="middle">{cut}'
            + (f" ×{count}" if count > 1 else "")
            + "</text>"
        )
    for label, color, dash in [
        ("U", "#bae6cc", None),
        ("P", "#e6ba82", "9 5"),
        ("R", "#9cbde6", "2 5"),
    ]:
        points = series[label]
        if not points:
            continue
        points_text = " ".join(
            f"{x(step):.2f},{y(value):.2f}" for step, value in points
        )
        dashed = f' stroke-dasharray="{dash}"' if dash else ""
        plot.append(
            f'<polyline data-arm="{label}" points="{points_text}" fill="none" stroke="{color}" stroke-width="2.7"{dashed} stroke-linecap="round"/>'
        )
        if len(points) == 1:
            plot.append(
                f'<circle cx="{x(points[0][0]):.2f}" cy="{y(points[0][1]):.2f}" r="4" fill="{color}"/>'
            )
    for step in sorted(
        {0, case["study"]["total_steps"] // 2, case["study"]["total_steps"]}
    ):
        plot.append(
            f'<text x="{x(step):.2f}" y="260" text-anchor="middle">{step}</text>'
        )
    plot.append('<text x="515" y="286" text-anchor="middle">Logical step</text></svg>')
    shown = sum(len(points) for points in series.values())
    sampled = (
        f" Display sampled to {shown} of {total_points} points."
        if shown < total_points
        else ""
    )
    return (
        f'<p class="hint">{_e(name)} · {_e(unit)} · ordinary observations only.{sampled}</p>'
        + "".join(plot)
        + '<div class="legend"><span>U · uninterrupted</span><span class="p">P · save + continue</span><span class="r">R · fresh-process restart</span></div>'
        + '<p class="hint" style="margin-top:16px">Overlapping lines indicate visual agreement only. Phase witnesses and full-array adjudication determine the result.</p>'
    )


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f'<th scope="col">{_e(h)}</th>' for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{_e(v)}</td>" for v in row) + "</tr>" for row in rows
        )
        + "</tbody></table></div>"
    )


def _records(case: dict) -> str:
    outputs = case["study"]["contract"]["outputs"]
    if not outputs:
        return '<p class="hint">No output tables are declared. This case makes no append-only record-identity claim.</p>'
    parts = []
    for name, contract in outputs.items():
        parts.append(
            f'<article class="card"><h3>{_e(name)}</h3><p class="small">Ordered semantic identities; no deduplication, interpolation or sorting.</p>'
        )
        rows = [
            [
                "Expected",
                len(contract["expected_keys"]),
                _summary(contract["expected_keys"]),
            ]
        ]
        for arm_name in _ARMS:
            batch = case["arms"][arm_name].get("outputs", {}).get(name, {})
            keys = batch.get("keys", []) if isinstance(batch, dict) else []
            rows.append(
                [
                    arm_name,
                    len(keys) if isinstance(keys, (list, tuple)) else "invalid",
                    _summary(keys),
                ]
            )
        parts.append(
            _table(["Source", "Record count", "Record identities (first 32)"], rows)
            + "</article>"
        )
    return "".join(parts)


def _contracts(case: dict) -> str:
    contract = case["study"]["contract"]
    groups = [("Observation", contract["fields"])] + [
        (f"Output: {name}", record["fields"])
        for name, record in contract["outputs"].items()
    ]
    rows = [
        [
            scope,
            f["name"],
            f["dtype"],
            f["shape"],
            f["unit"] or "(unitless)",
            f["mode"],
            f["atol"],
            f["rtol"],
        ]
        for scope, fields in groups
        for f in fields
    ]
    return _table(
        ["Scope", "Field", "Dtype", "Shape", "Unit", "Mode", "atol", "rtol"], rows
    )


def _arm_errors(case: dict) -> list[tuple[str, Any]]:
    errors = []
    for name in _ARMS:
        arm = case["arms"][name]
        if arm.get("error"):
            errors.append((name, arm["error"]))
        for index, segment in enumerate(arm.get("segments", [])):
            if segment.get("error") and segment["error"] != arm.get("error"):
                errors.append((f"{name} / segment {index}", segment["error"]))
    return errors


def render_report(bundle_path: str | os.PathLike) -> str:
    """Return one standalone HTML page after raw-evidence re-adjudication.

    Raises IntegrityError (or the underlying read error) on unsafe/inconsistent
    evidence. Supply a directory path; precomputed dictionaries are not accepted.
    """
    case = _load(bundle_path)
    adjudication, study = case["adjudication"], case["study"]
    outcome, status = _outcome(case), adjudication["status"]
    messages = {
        "pass": "All declared comparisons agree. This result covers the recorded fields, cadence and environment only.",
        "failure": "A declared continuation or output contract differs. Inspect the direct contrasts and exact witness below.",
        "skipped": "The controls or supported profile do not establish restart equivalence. This case is inconclusive.",
        "error": "Execution or evidence is incomplete or invalid. This case cannot establish equivalence.",
    }
    comparison = adjudication["comparisons"]
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'\">",
        '<meta name="color-scheme" content="light"><title>RestartWitness · continuation evidence</title>',
        "<style>" + _CSS + "</style></head><body>",
        '<a class="skip" href="#main">Skip to evidence</a>',
        '<header class="topbar"><div><div class="brand"><span class="brand-mark" aria-hidden="true">↳</span>RestartWitness</div>',
        '<nav aria-label="Report sections"><a href="#contrasts">Contrasts</a><a href="#trajectory">Trajectory</a><a href="#evidence">Evidence</a><a href="#replay">Replay</a></nav></div></header>',
        '<main id="main"><div class="hero"><div><p class="eyebrow">Continuation contract / evidence notebook</p>',
        '<h1>Pause. Restore.<br><span>Compare.</span></h1><p class="intro">One experiment, deliberately interrupted. See whether saving and restarting preserves the scientific state you chose to observe.</p>',
        '<span class="tag">Developer preview</span></div>',
        f'<div class="verdict {outcome}"><p class="eyebrow">Re-adjudicated from raw evidence</p><div class="status">{_e(status)}</div>',
        f'<p>{messages[outcome]}</p><div class="badges">'
        + "".join(_badge(f) for f in adjudication["findings"])
        + "</div>",
        '<p class="small" style="margin-top:16px">No upstream bug claim. A controlled example or empirical profile is not scientific certification.</p></div></div>',
    ]
    metrics = case["metrics"]
    wall = metrics.get("wall_seconds")
    wall_label = (
        f"{wall:.3f} s"
        if isinstance(wall, (int, float)) and math.isfinite(wall)
        else "unknown"
    )
    parts.append(
        '<div class="metrics">'
        + "".join(
            f'<div class="metric"><strong>{_e(value)}</strong><span>{label}</span></div>'
            for label, value in [
                (
                    "Simulated steps · all arms",
                    metrics.get("simulated_steps", "unknown"),
                ),
                ("Process starts", metrics.get("process_starts", "unknown")),
                ("Measured wall time", wall_label),
                (
                    "Unknown work segments",
                    metrics.get("unknown_work_segments", "unknown"),
                ),
            ]
        )
        + "</div>"
    )
    parts.append(
        '<section id="contrasts"><div class="section-head"><h2><span class="section-no">01</span>Three direct contrasts</h2><p class="small">Save-path and restore-path localization is not a root-cause proof. U-R is evaluated directly because tolerance-based agreement is not transitive.</p></div><div class="grid-three">'
    )
    parts.extend(
        _comparison_card(label, comparison[label]) for label in ("U-P", "P-R", "U-R")
    )
    parts.append(
        '</div></section><section id="witness"><h2><span class="section-no">02</span>First witness in comparison order</h2>'
    )
    witness = adjudication.get("first_witness")
    parts.append(
        f'<div class="witness{" good" if not witness else ""}">'
        + _witness_html(witness)
        + "</div>"
    )
    parts.append(
        '<p class="hint">First means comparator traversal order: baseline, observation control, U-P, P-R, U-R; trajectory observations precede output records. It is not a global earliest-time claim across arms.</p></section>'
    )
    parts.append(
        '<section id="trajectory"><div class="section-head"><h2><span class="section-no">03</span>The restart schedule</h2>'
    )
    parts.append(
        f'<p class="schedule">Cuts: {_e(", ".join(map(str, case["schedule"])) or "none")}<br>Total logical steps: {study["total_steps"]} · termination: {_e(study["termination"])}</p></div>'
    )
    parts.append(
        '<div class="plot-panel">'
        + _trajectory(case)
        + '</div><p class="hint">Cuts occur after the ordinary observation. Diagnostic phases are before_save, after_save and after_restore; repeated cuts retain their zero-based schedule occurrence. The illustration does not substitute for those exact comparisons.</p></section>'
    )
    parts.append(
        '<section id="controls"><h2><span class="section-no">04</span>Controls and execution</h2><p class="small">Two matching baselines do not establish probabilistic determinism. The observation control compares final ordinary state and outputs.</p><div class="grid-two">'
    )
    parts.extend(
        _comparison_card(label, comparison[label]) for label in ("U1-U2", "U1-O")
    )
    parts.append("</div>")
    arm_rows = []
    for name in _ARMS:
        arm = case["arms"][name]
        segments = arm.get("segments", [])
        calls = [s.get("advance_calls") for s in segments]
        arm_rows.append(
            [
                name,
                arm["status"],
                len(segments),
                len(arm.get("observations", [])),
                sum(x for x in calls if isinstance(x, int)),
                sum(x is None for x in calls),
            ]
        )
    parts.append(
        '<div style="margin-top:16px">'
        + _table(
            [
                "Arm",
                "Status",
                "Processes",
                "Observations",
                "Known advance calls",
                "Unknown segments",
            ],
            arm_rows,
        )
        + "</div>"
    )
    for label, error in _arm_errors(case):
        parts.append(
            f'<div class="card error-text" style="margin-top:12px"><h3>{_e(label)} error</h3><pre>{_e(_pretty(error))}</pre></div>'
        )
    parts.append(
        '</section><section id="records"><h2><span class="section-no">05</span>Output record identities</h2>'
        + _records(case)
        + "</section>"
    )
    parts.append(
        '<section id="contract"><h2><span class="section-no">06</span>The declared contract</h2><p class="small">Field names, dtype, shape, units and record identities are exact. Integer and boolean values are exact. Floating exact mode compares bit patterns; tolerance mode uses abs(a−b) ≤ atol + rtol × max(abs(a), abs(b)). No unit conversion is performed. Nonfinite values are invalid evidence.</p>'
        + _contracts(case)
    )
    parts.append(
        '<details><summary>Configuration and observation policy</summary><div class="details-body"><pre>'
        + _e(
            _pretty(
                {
                    "config": study["config"],
                    "observations": study["observations"],
                    "worker_timeout": study["worker_timeout"],
                    "study_budget": study["study_budget"],
                }
            )
        )
        + '</pre><p class="hint">Long lists are summarized; local paths are omitted. Complete inputs remain in the evidence bundle.</p></div></details></section>'
    )
    identity = case["identity"]
    parts.append(
        '<section id="evidence"><h2><span class="section-no">07</span>Evidence and source identity</h2><div class="card">'
    )
    parts.append(
        _facts(
            [
                ("Manifest SHA-256", case["manifest_sha256"]),
                ("Driver", identity.get("driver", "unknown")),
                ("Driver SHA-256", identity.get("driver_sha256", "unknown")),
                ("Python", identity.get("python", "unknown")),
                (
                    "Platform / machine",
                    f"{identity.get('platform', 'unknown')} / {identity.get('machine', 'unknown')}",
                ),
                ("Package versions", identity.get("packages", {})),
            ]
        )
    )
    parts.append(
        '<details><summary>Core source digests</summary><div class="details-body">'
        + _table(
            ["Source file", "SHA-256"],
            [
                [_redact_paths(name), sha]
                for name, sha in identity.get("core_sources", {}).items()
            ],
        )
        + "</div></details>"
    )
    parts.append(
        '<p class="hint" style="margin-top:18px">Every artifact hash was checked and every verdict recomputed from numeric arrays. Hashes detect changes; they do not authenticate the author or certify a complete environment. Report generation does not execute the driver.</p></div></section>'
    )
    parts.append(
        '<section id="replay"><h2><span class="section-no">08</span>Inspect, then replay locally</h2><p>Keep this page beside the original bundle. After reviewing the trusted local driver, replace CASE with the bundle directory:</p>'
    )
    parts.append(
        '<div class="terminal"><code>restartwitness replay ./CASE --output ./replayed-case --trust-driver</code></div><p class="small">Replay launches local Python code and native checkpoint loaders. Only use trusted drivers and checkpoints. Source and environment identity must match; a new output directory is required.</p></section>'
    )
    parts.append(
        '<section class="limitations" aria-labelledby="limits-title"><h2 id="limits-title">What this evidence does not establish</h2><ul><li>Unobserved state, undeclared event boundaries and unmeasured intervals may differ.</li><li>A tolerance pass does not establish exact continuation or scientific validity.</li><li>Clean process restarts do not test interrupted checkpoint writes, power-loss durability, MPI or GPU behavior.</li><li>Results apply to this recorded same-environment profile. Native-package examples do not imply endorsement or a new upstream defect.</li><li>Controlled fixture sensitivity is not a real-world defect rate. Historical motivations and release gates require separate evidence.</li></ul></section>'
    )
    parts.append(
        '<footer class="footer"><span>RestartWitness · developer preview · local evidence</span><span>Standalone HTML · no scripts, external fonts, telemetry or runtime network</span></footer></main></body></html>'
    )
    return "\n".join(parts)


def _xml_text(value: Any) -> str:
    text = _text(value)
    # ElementTree escapes XML markup but does not reject invalid XML 1.0 chars.
    return "".join(
        c
        if c in "\t\n\r"
        or "\x20" <= c <= "\ud7ff"
        or "\ue000" <= c <= "\ufffd"
        or "\U00010000" <= c <= "\U0010ffff"
        else "\ufffd"
        for c in text
    )


def _junit_details(case: dict) -> str:
    lines = [
        "Developer preview; no upstream bug claim.",
        "Status: " + case["adjudication"]["status"],
        "Findings: " + ", ".join(case["adjudication"]["findings"]),
    ]
    for label in _PAIRS:
        result = case["adjudication"]["comparisons"][label]
        lines.append(
            f"{label}: {result['status']}; trajectory={result['trajectory']['status']}; outputs={result['outputs']['status']}"
        )
        for key, value in _witness_rows(result.get("first_witness")):
            lines.append(f"  {key}: {_text(value)}")
    for name, error in _arm_errors(case):
        safe = _summary(error)
        if isinstance(safe, dict):
            lines.append(
                f"{name} error [{safe.get('phase', 'unknown phase')}]: {safe.get('message', _text(safe))}"
            )
        else:
            lines.append(f"{name} error: {_text(safe)}")
    return "\n".join(lines)


def render_junit(bundle_path: str | os.PathLike) -> str:
    """Return JUnit XML with one testcase for the complete experiment.

    Differences are failures; baseline/observation inconclusiveness and unsupported
    profiles are skipped; execution/contract/integrity errors are errors. An
    unreadable or tampered bundle raises instead of producing a passing XML file.
    """
    case = _load(bundle_path)
    outcome = _outcome(case)
    wall = case["metrics"].get("wall_seconds", 0)
    duration = (
        f"{wall:.6f}"
        if isinstance(wall, (int, float)) and math.isfinite(wall) and wall >= 0
        else "0"
    )
    suite = ET.Element(
        "testsuite",
        name="RestartWitness",
        tests="1",
        failures=str(int(outcome == "failure")),
        errors=str(int(outcome == "error")),
        skipped=str(int(outcome == "skipped")),
        time=duration,
    )
    properties = ET.SubElement(suite, "properties")
    for name, value in [
        ("manifest_sha256", case["manifest_sha256"]),
        ("status", case["adjudication"]["status"]),
        ("driver", case["identity"].get("driver", "unknown")),
        ("schedule", case["schedule"]),
        (
            "profile",
            "developer preview; declared same-environment continuation contract",
        ),
    ]:
        ET.SubElement(properties, "property", name=name, value=_xml_text(value))
    testcase = ET.SubElement(
        suite,
        "testcase",
        classname="restartwitness.continuation",
        name=_xml_text(case["identity"].get("driver", "unknown")),
        time=duration,
    )
    details = _xml_text(_junit_details(case))
    if outcome != "pass":
        node = ET.SubElement(
            testcase,
            outcome,
            type=_xml_text(case["adjudication"]["status"]),
            message=_xml_text(", ".join(case["adjudication"]["findings"])),
        )
        node.text = details
    ET.SubElement(testcase, "system-out").text = details
    ET.indent(suite, space="  ")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        + ET.tostring(suite, encoding="unicode")
        + "\n"
    )
