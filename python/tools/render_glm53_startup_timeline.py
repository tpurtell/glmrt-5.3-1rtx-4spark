#!/usr/bin/env python3
"""Render measured GLM-5.3 cold and warm startup timelines as one SVG."""

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

from analyze_glm53_full_startup import SCHEMA as STARTUP_SCHEMA, canonical_json
from validate_glm53_agentic_release_evidence import signed_serving
from validate_glm53_exl3_serving_qualification import GLM53_MODEL_ID, MODES


SCHEMA = "glmrt-glm53-startup-timeline-v1"
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
COLORS = (BLUE, VIOLET, CYAN, AMBER, GREEN, ORANGE)


class TimelineError(RuntimeError):
    """Startup timeline inputs are not exact signed final measurements."""


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def regular_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise TimelineError(f"{label} is a symbolic link")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise TimelineError(f"{label} is not one regular file")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TimelineError(f"{label} is not valid JSON") from error
    body = (
        {key: value for key, value in report.items() if key != "report_sha256"}
        if isinstance(report, dict)
        else None
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != STARTUP_SCHEMA
        or report.get("status") != "accepted"
        or not isinstance(body, dict)
        or report.get("report_sha256")
        != hashlib.sha256(canonical_json(body)).hexdigest()
    ):
        raise TimelineError(f"{label} is not signed accepted startup evidence")
    evidence = report.get("evidence")
    if not isinstance(evidence, dict) or set(evidence) != {
        "deployment",
        "expert_startup",
        "launcher_log",
        "coordinator_log",
    }:
        raise TimelineError(f"{label} has incomplete source evidence")
    for source_label, source in evidence.items():
        if not isinstance(source, dict):
            raise TimelineError(f"{label}/{source_label} identity is malformed")
        source_path = Path(str(source.get("path", ""))).expanduser()
        if source_path.is_symlink():
            raise TimelineError(f"{label}/{source_label} is now a symbolic link")
        try:
            source_path = source_path.resolve(strict=True)
        except FileNotFoundError as error:
            raise TimelineError(
                f"{label}/{source_label} source evidence is missing"
            ) from error
        if (
            not source_path.is_file()
            or source.get("bytes") != source_path.stat().st_size
            or source.get("sha256") != hash_file(source_path)
        ):
            raise TimelineError(f"{label}/{source_label} source evidence changed")
    return resolved, report


def identity(path: Path, schema: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "path": os.fspath(path),
        "bytes": path.stat().st_size,
        "sha256": hash_file(path),
    }


def validate_inputs(
    serving_path: Path, cold_path: Path, warm_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        serving_file, serving = signed_serving(serving_path)
    except Exception as error:
        raise TimelineError("serving qualification is invalid") from error
    cold_file, cold = regular_json(cold_path, "cold startup")
    warm_file, warm = regular_json(warm_path, "warm startup")
    selected = serving.get("results", {}).get("default_speculation")
    runtime = serving.get("runtime")
    if selected not in MODES or not isinstance(runtime, dict):
        raise TimelineError("serving qualification has no measured default")
    expected = {
        "model_id": GLM53_MODEL_ID,
        "model_revision": runtime.get("model_revision"),
        "profile": "balanced",
        "speculation": selected,
        "speculation_settings": runtime.get("speculation_settings", {}).get(selected),
        "power_limit_w": runtime.get("power_limit_w"),
        "engine_identity": runtime.get("engine_identity"),
        "sparkinfer_revision": runtime.get("sparkinfer_revision"),
        "expert_runtime_fingerprint": runtime.get(
            "expert_runtime_fingerprints", {}
        ).get(selected),
    }
    for label, report, state in (
        ("cold", cold, "cold"),
        ("warm", warm, "warm"),
    ):
        actual = {key: report.get(key) for key in expected}
        alignment = report.get("alignment")
        if (
            actual != expected
            or report.get("launch_state") != state
            or not isinstance(alignment, dict)
            or alignment.get("experts_resident_at_start") is (state == "cold")
            or not isinstance(alignment.get("launcher_wall_ms"), (int, float))
            or not math.isfinite(float(alignment["launcher_wall_ms"]))
            or float(alignment["launcher_wall_ms"]) <= 0.0
        ):
            raise TimelineError(f"{label} startup differs from the selected runtime")
    return serving, cold, warm, {
        "serving": identity(serving_file, serving["schema"]),
        "cold": identity(cold_file, STARTUP_SCHEMA),
        "warm": identity(warm_file, STARTUP_SCHEMA),
    }


def phase_value(report: dict[str, Any], group: str, stage: str) -> float:
    rows = report["phases"][group]
    matches = [float(row["elapsed_ms"]) for row in rows if row["stage"] == stage]
    if len(matches) != 1:
        raise TimelineError(f"startup report has no unique {group}/{stage} phase")
    return matches[0]


def coordinator_segments(report: dict[str, Any]) -> list[tuple[str, float, str]]:
    real_rows = report["phases"]["real_full"]
    before_wait = {
        "validation",
        "catalog-kv-config",
        "targets-tokenizer",
        "kv-snapshot-config",
        "prewarm-prompts",
        "coordinator-resident-preload",
        "dspark-preload",
    }
    assembly = {
        "dispatch-worker",
        "executor-assembly",
        "python-capture-barrier",
        "request-worker-spawn",
        "request-worker-inline",
    }
    prewarm = {
        "prewarm-paired-lm-head-initial",
        "prewarm-main",
        "prewarm-batched-dspark",
        "prewarm-audit-seal",
        "complete",
    }

    def total(stages: set[str]) -> float:
        return sum(float(row["elapsed_ms"]) for row in real_rows if row["stage"] in stages)

    return [
        (
            "shell + cache identity",
            float(report["alignment"]["coordinator_shell_ms"]),
            BLUE,
        ),
        ("weights / draft preload", total(before_wait), VIOLET),
        (
            "Spark readiness",
            phase_value(report, "real_full", "sparse-target-connect"),
            AMBER,
        ),
        (
            "expert warmup",
            phase_value(report, "real_full", "expert-warmup"),
            ORANGE,
        ),
        ("executor assembly", total(assembly), CYAN),
        ("targeted graph prewarm", total(prewarm), GREEN),
        ("API bind", phase_value(report, "coordinator_daemon", "api-bind"), BLUE),
    ]


def launcher_segments(report: dict[str, Any]) -> list[tuple[str, float, str]]:
    return [
        (str(row["stage"]), float(row["elapsed_ms"]), COLORS[index % len(COLORS)])
        for index, row in enumerate(report["phases"]["launcher"])
    ]


def render_svg(cold: dict[str, Any], warm: dict[str, Any]) -> str:
    width, height = 1600, 1050
    left, chart_right = 190.0, 1530.0
    maximum_ms = max(
        float(cold["alignment"]["launcher_wall_ms"]),
        float(warm["alignment"]["launcher_wall_ms"]),
    )
    axis_ms = math.ceil(maximum_ms / 5_000.0) * 5_000.0
    px_per_ms = (chart_right - left) / axis_ms

    def rect(x: float, y: float, w: float, h: float, color: str, opacity: float = 0.8) -> str:
        return (
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w, 1.5):.2f}" '
            f'height="{h:.2f}" rx="6" fill="{color}" fill-opacity="{opacity}" '
            f'stroke="{color}"/>'
        )

    items = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">GLMRT GLM-5.3 cold and warm startup timelines</title>',
        (
            '<desc id="desc">Measured aligned startup phases for a full '
            "four-Spark expert reload and a fingerprint-matched retained-expert "
            "restart.</desc>"
        ),
        '<defs><style>',
        f'.title{{font:700 31px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        f'.subtitle{{font:400 14px Inter,Segoe UI,sans-serif;fill:{MUTED}}}',
        f'.section{{font:700 18px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        f'.body{{font:500 12px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        f'.small{{font:400 11px Inter,Segoe UI,sans-serif;fill:{MUTED}}}',
        f'.metric{{font:700 25px Inter,Segoe UI,sans-serif;fill:{TEXT}}}',
        '</style></defs>',
        f'<rect width="{width}" height="{height}" fill="{BG}"/>',
        f'<text id="title" x="52" y="52" class="title">GLMRT GLM-5.3 STARTUP</text>',
        (
            f'<text x="52" y="78" class="subtitle">{escape(cold["profile"])} · '
            f'{escape(cold["speculation"])} · {cold["power_limit_w"]} W · '
            f'revision {escape(cold["model_revision"][:12])}</text>'
        ),
    ]
    cold_wall = float(cold["alignment"]["launcher_wall_ms"])
    warm_wall = float(warm["alignment"]["launcher_wall_ms"])
    metrics = (
        (52, f"{cold_wall / 1000:.2f} s", "COLD · FULL EXPERT RELOAD", GREEN),
        (430, f"{warm_wall / 1000:.2f} s", "WARM · RETAINED EXPERTS", CYAN),
        (808, f"{cold_wall / warm_wall:.2f}×", "COLD / WARM WALL RATIO", VIOLET),
        (
            1186,
            f"{cold['alignment']['spark_ready_ms'] / 1000:.2f} s",
            "COLD · ALL SPARKS READY",
            AMBER,
        ),
    )
    for x, value, label, color in metrics:
        items.extend(
            [
                (
                    f'<rect x="{x}" y="104" width="340" height="86" '
                    f'rx="14" fill="{PANEL}" stroke="{color}"/>'
                ),
                f'<text x="{x + 20}" y="139" class="metric" fill="{color}">{escape(value)}</text>',
                f'<text x="{x + 20}" y="169" class="small">{escape(label)}</text>',
            ]
        )

    for tick in range(0, 6):
        value = axis_ms * tick / 5
        x = left + value * px_per_ms
        items.append(
            f'<path d="M{x:.2f},230 V970" stroke="{GRID}" stroke-width="1" opacity=".55"/>'
        )
        items.append(
            (
                f'<text x="{x:.2f}" y="1010" class="small" '
                f'text-anchor="middle">{value / 1000:.1f} s</text>'
            )
        )

    panels = (("COLD", cold, 255.0), ("WARM", warm, 620.0))
    for panel_label, report, y in panels:
        items.append(
            (
                f'<rect x="52" y="{y - 35}" width="1490" height="320" '
                f'rx="17" fill="{PANEL}" stroke="{GRID}"/>'
            )
        )
        items.append(
            f'<text x="78" y="{y}" class="section">{panel_label} START</text>'
        )
        rows = (
            ("launcher", launcher_segments(report), 0.0),
            (
                "coordinator",
                coordinator_segments(report),
                float(report["alignment"]["coordinator_dispatch_offset_ms"]),
            ),
        )
        for row_index, (row_label, segments, offset_ms) in enumerate(rows):
            row_y = y + 42 + row_index * 82
            items.append(
                (
                    f'<text x="{left - 18}" y="{row_y + 27}" class="body" '
                    f'text-anchor="end">{escape(row_label)}</text>'
                )
            )
            x = left + offset_ms * px_per_ms
            for label, value, color in segments:
                segment_width = value * px_per_ms
                items.append(rect(x, row_y, segment_width, 46, color))
                if segment_width >= 76:
                    items.append(
                        (
                            f'<text x="{x + segment_width / 2:.2f}" '
                            f'y="{row_y + 20}" class="small" '
                            f'text-anchor="middle">{escape(label)}</text>'
                        )
                    )
                    items.append(
                        (
                            f'<text x="{x + segment_width / 2:.2f}" '
                            f'y="{row_y + 37}" class="small" '
                            f'text-anchor="middle">{value / 1000:.2f}s</text>'
                        )
                    )
                x += segment_width
        spark_y = y + 206
        items.append(
            (
                f'<text x="{left - 18}" y="{spark_y + 27}" class="body" '
                'text-anchor="end">four Sparks</text>'
            )
        )
        if report["alignment"]["experts_resident_at_start"]:
            items.append(rect(left, spark_y, 170, 46, GREEN))
            items.append(
                (
                    f'<text x="{left + 85}" y="{spark_y + 29}" class="body" '
                    'text-anchor="middle">resident at t=0</text>'
                )
            )
        else:
            spark_start = float(report["alignment"]["spark_dispatch_offset_ms"])
            spark_end = float(report["alignment"]["spark_ready_ms"])
            x = left + spark_start * px_per_ms
            items.append(rect(x, spark_y, (spark_end - spark_start) * px_per_ms, 46, VIOLET))
            items.append(
                (
                    f'<text x="{x + (spark_end - spark_start) * px_per_ms / 2:.2f}" '
                    f'y="{spark_y + 29}" class="body" text-anchor="middle">'
                    "parallel slab reload → all ready</text>"
                )
            )
        wall_x = left + float(report["alignment"]["launcher_wall_ms"]) * px_per_ms
        items.append(
            (
                f'<path d="M{wall_x:.2f},{y + 25} V{y + 260}" '
                f'stroke="{TEXT}" stroke-width="2" stroke-dasharray="5 5"/>'
            )
        )
        items.append(
            (
                f'<text x="{wall_x - 7:.2f}" y="{y + 20}" class="small" '
                'text-anchor="end">API ready</text>'
            )
        )

    items.extend(
        [
            (
                '<text x="52" y="1034" class="subtitle">All bars are measured '
                "phase clocks. Coordinator rows are aligned at launcher dispatch; "
                "four Spark reloads run in parallel.</text>"
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
        raise TimelineError(f"refusing to overwrite output: {destination}")
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
    cold_path: Path,
    warm_path: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    resolved_output = output_path.expanduser().resolve()
    resolved_report = report_path.expanduser().resolve()
    if resolved_output == resolved_report:
        raise TimelineError("SVG and report outputs must be distinct")
    if resolved_output.exists() or resolved_report.exists():
        raise TimelineError("refusing to overwrite timeline output")
    serving, cold, warm, sources = validate_inputs(
        serving_path, cold_path, warm_path
    )
    svg = render_svg(cold, warm).encode()
    body = {
        "schema": SCHEMA,
        "status": "rendered",
        "model_id": GLM53_MODEL_ID,
        "model_revision": serving["runtime"]["model_revision"],
        "default_speculation": serving["results"]["default_speculation"],
        "cold_wall_ms": cold["alignment"]["launcher_wall_ms"],
        "warm_wall_ms": warm["alignment"]["launcher_wall_ms"],
        "cold_to_warm_ratio": (
            cold["alignment"]["launcher_wall_ms"]
            / warm["alignment"]["launcher_wall_ms"]
        ),
        "svg": {
            "path": os.fspath(resolved_output),
            "bytes": len(svg),
            "sha256": hashlib.sha256(svg).hexdigest(),
        },
        "sources": sources,
    }
    report = body | {
        "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()
    }
    atomic_bytes(output_path, svg)
    atomic_bytes(
        report_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--cold", type=Path, required=True)
    parser.add_argument("--warm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = render(
        serving_path=args.serving,
        cold_path=args.cold,
        warm_path=args.warm,
        output_path=args.output,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
