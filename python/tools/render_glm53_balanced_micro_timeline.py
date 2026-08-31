#!/usr/bin/env python3
"""Render a production GLM-5.3 target-cycle micro-timeline as SVG."""

from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from validate_glm52_exl3_serving_qualification import (
    blended,
    deployment,
    evidence_identity,
    read_jsonl,
)
from validate_glm53_agentic_release_evidence import signed_serving
from validate_glm53_exl3_serving_qualification import (
    GLM53_MODEL_ID,
    MODES,
    QualificationError,
    require_eight_type_blended,
    verify_cycle_curve,
)
from validate_glm53_profile_release_evidence import _benchmark_metadata


SCHEMA = "glmrt-glm53-balanced-micro-timeline-v1"
BG = "#07111f"
PANEL = "#0d1b2d"
GRID = "#203a55"
TEXT = "#e9f3ff"
MUTED = "#91a8bf"
CYAN = "#46d7e8"
BLUE = "#5d8cff"
VIOLET = "#a978ff"
GREEN = "#56d69c"
AMBER = "#ffc35a"
ORANGE = "#ff8f5a"
RED = "#ff6b79"
M_COLORS = (MUTED, CYAN, GREEN, BLUE, VIOLET, AMBER, ORANGE, RED)


class MicroTimelineError(RuntimeError):
    """The production cycle evidence is incomplete or mismatched."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity(deployed: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_revision": deployed["model_revision"],
        "profile": deployed["profile"],
        "power_limit_w": deployed["power_limit_w"],
        "engine_identity": deployed["engine_identity"],
        "sparkinfer_revision": deployed["sparkinfer_revision"],
        "coordinator_slot": deployed["fingerprints"]["coordinator_slot"],
        "expert_slot": deployed["fingerprints"]["expert_slot"],
        "expert_runtime": deployed["fingerprints"]["expert_runtime"],
        "speculation_settings": deployed["speculation_settings"],
    }


def selected_code_cycles(record: dict[str, Any]) -> list[dict[str, Any]]:
    physical = record.get("target_cycle_physical_m")
    elapsed = record.get("target_cycle_ms")
    draft_lengths = record.get("draft_lengths")
    accepted = record.get("accepted_draft_lengths")
    if not all(isinstance(value, list) for value in (physical, elapsed, draft_lengths, accepted)):
        raise MicroTimelineError("code replay has no aligned target-cycle arrays")
    if len(physical) != len(elapsed) or len(draft_lengths) != len(accepted):
        raise MicroTimelineError("code replay target-cycle arrays differ in length")
    cycles = []
    verify_index = 0
    for index, (raw_m, raw_ms) in enumerate(zip(physical, elapsed, strict=True)):
        if isinstance(raw_m, bool) or not isinstance(raw_m, int) or not 1 <= raw_m <= 8:
            raise MicroTimelineError("code replay has an invalid physical M")
        if isinstance(raw_ms, bool) or not isinstance(raw_ms, (int, float)):
            raise MicroTimelineError("code replay has a nonnumeric cycle time")
        elapsed_ms = float(raw_ms)
        if not math.isfinite(elapsed_ms) or elapsed_ms <= 0.0:
            raise MicroTimelineError("code replay has a nonpositive cycle time")
        if raw_m == 1:
            committed = 1
        else:
            if verify_index >= len(draft_lengths):
                raise MicroTimelineError("code replay is missing a verifier cycle")
            drafts = draft_lengths[verify_index]
            accepted_count = accepted[verify_index]
            if (
                isinstance(drafts, bool)
                or not isinstance(drafts, int)
                or drafts + 1 != raw_m
                or isinstance(accepted_count, bool)
                or not isinstance(accepted_count, int)
                or not 0 <= accepted_count <= drafts
            ):
                raise MicroTimelineError("code replay verifier width/acceptance differs")
            committed = accepted_count + 1
            verify_index += 1
        cycles.append(
            {
                "cycle": index + 1,
                "physical_m": raw_m,
                "elapsed_ms": elapsed_ms,
                "committed_tokens": committed,
            }
        )
    completion_tokens = record.get("completion_tokens")
    emitted = record.get("emitted_tokens_from_verify")
    decode_ms = record.get("decode_ms")
    if (
        verify_index != len(draft_lengths)
        or isinstance(completion_tokens, bool)
        or not isinstance(completion_tokens, int)
        or completion_tokens < 1
        or isinstance(emitted, bool)
        or not isinstance(emitted, int)
        or emitted < 0
        or emitted
        > sum(
            cycle["committed_tokens"]
            for cycle in cycles
            if cycle["physical_m"] > 1
        )
        or not math.isclose(
            sum(cycle["elapsed_ms"] for cycle in cycles),
            float(decode_ms),
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        )
    ):
        raise MicroTimelineError("code replay cycles do not reconcile to decode")

    # The acceptance counters describe the verifier result before the API
    # applies max-token and stop-token truncation. A terminal cycle can
    # therefore report N accepted drafts while exposing fewer than N+1 tokens
    # to the client (an empty EOS is the common case). The complete target
    # timeline still contains that cycle and its elapsed time. Trim its nominal
    # commits, starting at the end, so the visualization reconciles to the
    # exact post-TTFT token count used by the benchmark rather than inventing
    # output that the API did not expose.
    remaining_trim = (
        sum(cycle["committed_tokens"] for cycle in cycles)
        - (completion_tokens - 1)
    )
    if remaining_trim < 0:
        raise MicroTimelineError("code replay cycles do not reconcile to decode")
    for cycle in reversed(cycles):
        if remaining_trim == 0:
            break
        removed = min(cycle["committed_tokens"], remaining_trim)
        cycle["committed_tokens"] -= removed
        remaining_trim -= removed
    if remaining_trim != 0:
        raise MicroTimelineError("code replay cycles do not reconcile to decode")
    return cycles


def validate_inputs(
    serving_path: Path, deployment_path: Path, blended_path: Path
) -> dict[str, Any]:
    try:
        serving_file, serving = signed_serving(serving_path)
    except Exception as error:
        raise MicroTimelineError("serving qualification is invalid") from error
    selected = serving.get("results", {}).get("default_speculation")
    runtime = serving.get("runtime")
    if selected not in MODES or not isinstance(runtime, dict):
        raise MicroTimelineError("serving qualification has no measured default")
    try:
        deployed = deployment(
            deployment_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=selected,
        )
        decoded = blended(
            blended_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
        )
        require_eight_type_blended(decoded)
        curve = verify_cycle_curve(
            blended_path,
            expected_fixed_drafts=(
                deployed["speculation_settings"]["fixed_drafts"]
                if selected == "dflash2"
                else None
            ),
        )
        run = _benchmark_metadata(
            blended_path,
            kind="blended",
            profile="balanced",
            launch_started_ns=deployed["launch_started_ns"],
        )
    except (QualificationError, RuntimeError) as error:
        raise MicroTimelineError("selected production decode evidence is invalid") from error
    expected_runtime = {
        "model_revision": runtime.get("model_revision"),
        "profile": "balanced",
        "power_limit_w": runtime.get("power_limit_w"),
        "engine_identity": runtime.get("engine_identity"),
        "sparkinfer_revision": runtime.get("sparkinfer_revision"),
        "coordinator_slot": runtime.get("coordinator_slot_fingerprint"),
        "expert_slot": runtime.get("expert_slot_fingerprint"),
        "expert_runtime": runtime.get("expert_runtime_fingerprints", {}).get(selected),
        "speculation_settings": runtime.get("speculation_settings", {}).get(selected),
    }
    if (
        _runtime_identity(deployed) != expected_runtime
        or deployed["identity"] != serving["evidence"][f"{selected}_deployment"]
        or decoded["identity"] != serving["evidence"][f"{selected}_blended"]
        or serving["results"]["modes"][selected]["verify_cycle_by_physical_m"]
        != curve
    ):
        raise MicroTimelineError("decode evidence differs from the selected runtime")
    resolved, records = read_jsonl(blended_path)
    matches = [
        record
        for record in records
        if "aggregate" not in record
        and record.get("case") == "code"
        and record.get("repeat") == 1
    ]
    if len(matches) != 1:
        raise MicroTimelineError("selected decode has no unique first code replay")
    cycles = selected_code_cycles(matches[0])
    return {
        "serving": serving,
        "deployment": deployed,
        "blended": decoded,
        "curve": curve,
        "run": run,
        "record": matches[0],
        "cycles": cycles,
        "evidence": {
            "serving": evidence_identity(serving_file, serving["schema"]),
            "deployment": deployed["identity"],
            "blended": evidence_identity(resolved, "glmrt-mtp-acceptance-jsonl-v4"),
        },
    }


def render_svg(data: dict[str, Any]) -> str:
    serving = data["serving"]
    selected = serving["results"]["default_speculation"]
    mode = serving["results"]["modes"][selected]
    record = data["record"]
    cycles = data["cycles"]
    curve = data["curve"]
    total_committed = sum(cycle["committed_tokens"] for cycle in cycles)
    mean_commit = total_committed / len(cycles)
    width, height = 1760, 700

    def text(x: float, y: float, value: str, cls: str = "body", anchor: str = "start") -> str:
        return (
            f'<text x="{x:.2f}" y="{y:.2f}" class="{cls}" '
            f'text-anchor="{anchor}">{escape(value)}</text>'
        )

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GLMRT GLM-5.3 balanced production micro-timeline</title>',
        (
            '<desc id="desc">Measured target-cycle timing by physical M and '
            "the production target-cycle topology.</desc>"
        ),
        '<defs><style>',
        f'.title{{font:700 31px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        f'.subtitle{{font:400 14px Inter,Segoe UI,sans-serif;fill:{MUTED}}}',
        f'.section{{font:700 18px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        f'.body{{font:500 12px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        f'.small{{font:400 11px Inter,Segoe UI,sans-serif;fill:{MUTED}}}',
        f'.metric{{font:700 24px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        '</style></defs>',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        text(52, 51, "GLMRT GLM-5.3 · PRODUCTION MICRO-TIMELINE", "title"),
        text(
            52,
            78,
            (
                f"balanced · {selected} · code replay 1 · revision "
                f"{serving['runtime']['model_revision'][:12]} · no synchronized instrumentation"
            ),
            "subtitle",
        ),
    ]
    metrics = (
        (52, f"{mode['weighted_decode_tps']:.2f} tok/s", "8-TYPE WEIGHTED", GREEN),
        (452, f"{record['decode_tps']:.2f} tok/s", "CODE REPLAY", CYAN),
        (852, f"{mode['accepted_draft_rate'] * 100:.1f}%", "DRAFT ACCEPTANCE", VIOLET),
        (1252, f"{mean_commit:.2f}", "TOKENS / TARGET CYCLE", AMBER),
    )
    for x, value, label, color in metrics:
        items.extend(
            [
                (
                    f'<rect x="{x}" y="102" width="360" height="86" '
                    f'rx="14" fill="{PANEL}" stroke="{color}"/>'
                ),
                text(x + 20, 138, value, "metric"),
                text(x + 20, 169, label, "small"),
            ]
        )

    items.extend(
        [
            (
                f'<rect x="52" y="220" width="1050" height="405" '
                f'rx="17" fill="{PANEL}" stroke="{GREEN}"/>'
            ),
            text(80, 258, "A · MEASURED TARGET-CYCLE CURVE", "section"),
            text(
                80,
                283,
                "All five eight-type replays; medians use the exact decode clock.",
                "subtitle",
            ),
        ]
    )
    plot_left, plot_top, plot_width, plot_height = 130.0, 330.0, 900.0, 210.0
    max_curve = max(float(row["median_ms"]) for row in curve.values()) * 1.15
    ordered = sorted((int(key), row) for key, row in curve.items())
    points = []
    for index, (physical_m, row) in enumerate(ordered):
        px = plot_left + index * (plot_width / max(len(ordered) - 1, 1))
        value = float(row["median_ms"])
        py = plot_top + plot_height - value / max_curve * plot_height
        points.append((px, py))
        items.append(
            f'<path d="M{px:.2f},{plot_top} V{plot_top + plot_height}" '
            f'stroke="{GRID}" stroke-dasharray="3 5"/>'
        )
        items.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="7" fill="{GREEN}" '
            f'stroke="{TEXT}" stroke-width="2"/>'
        )
        items.append(text(px, py - 16, f"{value:.1f} ms", "small", "middle"))
        items.append(text(px, plot_top + plot_height + 25, f"M{physical_m}", "body", "middle"))
        items.append(
            text(
                px,
                plot_top + plot_height + 43,
                f"n={row['samples']}",
                "small",
                "middle",
            )
        )
    items.append(
        (
            '<path d="M'
            + " L".join(f"{px:.2f},{py:.2f}" for px, py in points)
            + f'" fill="none" stroke="{GREEN}" stroke-width="3"/>'
        )
    )

    items.extend(
        [
            (
                f'<rect x="1132" y="220" width="576" height="405" '
                f'rx="17" fill="{PANEL}" stroke="{CYAN}"/>'
            ),
            text(1160, 258, "B · ONE TARGET CYCLE", "section"),
            text(1160, 283, "Collapsed topology · schematic, not time-scaled", "subtitle"),
        ]
    )
    boxes = (
        (1165, 325, "M target rows", "current token + proposals", BLUE),
        (1425, 325, "L0–2 dense", "local coordinator", GREEN),
        (1165, 425, "L3–77 sparse", "four-Spark TP4 per layer", VIOLET),
        (1425, 425, "terminal", "score → accept → commit", AMBER),
    )
    for x, y, title, subtitle, color in boxes:
        items.append(
            f'<rect x="{x}" y="{y}" width="235" height="72" rx="10" '
            f'fill="{BG}" stroke="{color}"/>'
        )
        items.append(text(x + 16, y + 29, title, "body"))
        items.append(text(x + 16, y + 52, subtitle, "small"))
    items.extend(
        [
            f'<path d="M1400,361 H1425" stroke="{CYAN}" stroke-width="2"/>',
            f'<path d="M1542,397 V425" stroke="{CYAN}" stroke-width="2"/>',
            f'<path d="M1425,461 H1400" stroke="{CYAN}" stroke-width="2"/>',
            f'<path d="M1282,425 V397" stroke="{CYAN}" stroke-width="2"/>',
            text(1160, 545, "75 serial sparse dependency edges", "body"),
            text(1160, 571, "physical M changes compute/transport width", "small"),
            text(1160, 597, "committed tokens depend on acceptance", "small"),
            text(
                52,
                672,
                (
                    "Production request clocks only; no synchronized stage "
                    "attribution or reused legacy data."
                ),
                "subtitle",
            ),
            '</svg>',
            '',
        ]
    )
    return "\n".join(items)


def atomic_bytes(path: Path, payload: bytes) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise MicroTimelineError(f"refusing to overwrite output: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def render(
    *,
    serving_path: Path,
    deployment_path: Path,
    blended_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    resolved_output = output_path.expanduser().resolve()
    resolved_report = report_path.expanduser().resolve()
    if resolved_output == resolved_report:
        raise MicroTimelineError("SVG and report outputs must be distinct")
    if resolved_output.exists() or resolved_report.exists():
        raise MicroTimelineError("refusing to overwrite timeline output")
    data = validate_inputs(serving_path, deployment_path, blended_path)
    svg = render_svg(data).encode()
    record = data["record"]
    body = {
        "schema": SCHEMA,
        "status": "rendered",
        "model_id": GLM53_MODEL_ID,
        "model_revision": data["deployment"]["model_revision"],
        "profile": "balanced",
        "speculation": data["deployment"]["speculation"],
        "run_id": data["run"]["run_id"],
        "selected_request": {
            "case": record["case"],
            "repeat": record["repeat"],
            "prompt_sha256": record["prompt_sha256"],
            "decode_ms": record["decode_ms"],
            "decode_tps": record["decode_tps"],
            "target_cycles": len(data["cycles"]),
        },
        "svg": {
            "path": os.fspath(resolved_output),
            "bytes": len(svg),
            "sha256": hashlib.sha256(svg).hexdigest(),
        },
        "evidence": data["evidence"],
    }
    report = body | {
        "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()
    }
    atomic_bytes(resolved_output, svg)
    atomic_bytes(
        resolved_report,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--blended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = render(
        serving_path=args.serving,
        deployment_path=args.deployment,
        blended_path=args.blended,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
