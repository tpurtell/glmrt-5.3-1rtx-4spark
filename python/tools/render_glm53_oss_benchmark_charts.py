#!/usr/bin/env python3
"""Render GLM-5.3 release prefill/decode SVGs from signed OSS evidence."""

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

from bench_release_decode_matrix import DEFAULT_CONTEXTS, WORKLOADS
from bench_release_prefill_matrix import DEFAULT_BASE_CONTEXTS, DEFAULT_SUFFIX_ROWS
from render_glm53_oss_benchmark_markdown import (
    MODE_LABELS,
    context_label,
    dflash2_prefill_value,
    finite,
    indexed_cells,
    validate_evidence,
)
from validate_glm52_exl3_serving_qualification import evidence_identity
from validate_glm53_exl3_serving_qualification import MODE_DFLASH2
from validate_glm53_oss_release_evidence import canonical_json


SCHEMA = "glmrt-glm53-oss-benchmark-charts-v1"
COLORS = ("#58a6ff", "#3fb950", "#d29922", "#f778ba", "#a371f7")


class ChartRenderError(RuntimeError):
    """The signed OSS evidence cannot produce complete benchmark charts."""


def nice_axis(maximum: float, ticks: int = 5) -> tuple[float, float]:
    value = finite(maximum, "chart maximum", minimum=0.000001)
    if ticks < 2:
        raise ChartRenderError("chart axis requires at least two ticks")
    raw = value / ticks
    magnitude = 10.0 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    factor = 1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0
    if normalized > 5.0:
        factor = 10.0
    step = factor * magnitude
    return math.ceil(value / step) * step, step


def number(value: float) -> str:
    if value >= 1_000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def line_chart(
    *,
    title: str,
    subtitle: str,
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    x_axis: str,
    y_axis: str = "tokens / second",
) -> bytes:
    if (
        len(x_labels) < 2
        or not series
        or len(series) > len(COLORS)
        or any(len(values) != len(x_labels) for _label, values in series)
    ):
        raise ChartRenderError("chart series has an invalid shape")
    values = [
        finite(value, "chart throughput", minimum=0.000001)
        for _label, row in series
        for value in row
    ]
    y_top, y_step = nice_axis(max(values))
    width, height = 1_200, 680
    left, right, top, bottom = 110.0, 1_100.0, 135.0, 525.0
    plot_width = right - left
    plot_height = bottom - top
    x_positions = [
        left + index * plot_width / (len(x_labels) - 1)
        for index in range(len(x_labels))
    ]

    def y_position(value: float) -> float:
        return bottom - value / y_top * plot_height

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
            'aria-labelledby="title desc">'
        ),
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(subtitle)}</desc>',
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '<stop offset="0" stop-color="#0d1117"/>',
        '<stop offset="1" stop-color="#111b2a"/>',
        "</linearGradient>",
        '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">',
        '<feGaussianBlur stdDeviation="2.5" result="blur"/>',
        '<feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>',
        "</filter>",
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="18" fill="url(#bg)"/>',
        '<rect x="28" y="25" width="144" height="27" rx="13.5" fill="#1f6feb"/>',
        (
            '<text x="100" y="44" text-anchor="middle" fill="#ffffff" '
            'font-family="system-ui,sans-serif" font-size="12" font-weight="700" '
            'letter-spacing="1.4">GLM-5.3 · K4</text>'
        ),
        (
            f'<text x="28" y="87" fill="#f0f6fc" font-family="system-ui,sans-serif" '
            f'font-size="25" font-weight="700">{escape(title)}</text>'
        ),
        (
            f'<text x="28" y="112" fill="#8b949e" font-family="system-ui,sans-serif" '
            f'font-size="13">{escape(subtitle)}</text>'
        ),
    ]
    tick_count = int(round(y_top / y_step))
    for tick in range(tick_count + 1):
        value = tick * y_step
        y = y_position(value)
        lines.extend(
            [
                (
                    f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" '
                    f'y2="{y:.1f}" stroke="#30363d" stroke-width="1"/>'
                ),
                (
                    f'<text x="{left - 16:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                    'fill="#8b949e" font-family="system-ui,sans-serif" '
                    f'font-size="12">{number(value)}</text>'
                ),
            ]
        )
    for x, label in zip(x_positions, x_labels, strict=True):
        lines.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" '
                    f'y2="{bottom:.1f}" stroke="#21262d" stroke-width="1"/>'
                ),
                (
                    f'<text x="{x:.1f}" y="{bottom + 25:.1f}" text-anchor="middle" '
                    'fill="#b1bac4" font-family="system-ui,sans-serif" '
                    f'font-size="12">{escape(label)}</text>'
                ),
            ]
        )
    for (label, row), color in zip(series, COLORS, strict=False):
        points = " ".join(
            f"{x:.1f},{y_position(value):.1f}"
            for x, value in zip(x_positions, row, strict=True)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round" '
            'filter="url(#glow)"/>'
        )
        for x, value in zip(x_positions, row, strict=True):
            y = y_position(value)
            lines.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" '
                'stroke="#0d1117" stroke-width="2"/>'
            )
    legend_width = 930.0 / len(series)
    legend_start = (width - legend_width * len(series)) / 2
    for index, ((label, _row), color) in enumerate(
        zip(series, COLORS, strict=False)
    ):
        x = legend_start + index * legend_width
        lines.extend(
            [
                (
                    f'<line x1="{x:.1f}" y1="604" x2="{x + 28:.1f}" y2="604" '
                    f'stroke="{color}" stroke-width="4" stroke-linecap="round"/>'
                ),
                (
                    f'<text x="{x + 38:.1f}" y="608" fill="#e6edf3" '
                    'font-family="system-ui,sans-serif" font-size="12">'
                    f'{escape(label)}</text>'
                ),
            ]
        )
    lines.extend(
        [
            (
                f'<text x="{(left + right) / 2:.1f}" y="{bottom + 62:.1f}" '
                'text-anchor="middle" fill="#8b949e" '
                'font-family="system-ui,sans-serif" '
                f'font-size="12">{escape(x_axis)}</text>'
            ),
            (
                f'<text x="25" y="{(top + bottom) / 2:.1f}" text-anchor="middle" '
                'fill="#8b949e" font-family="system-ui,sans-serif" font-size="12" '
                f'transform="rotate(-90 25 {(top + bottom) / 2:.1f})">'
                f'{escape(y_axis)}</text>'
            ),
            (
                '<text x="1170" y="650" text-anchor="end" fill="#484f58" '
                'font-family="ui-monospace,monospace" font-size="10">'
                'source-validated release evidence</text>'
            ),
            "</svg>",
        ]
    )
    return ("\n".join(lines) + "\n").encode()


def chart_inputs(report: dict[str, Any]) -> tuple[bytes, bytes]:
    selected = report["default_speculation"]
    mode_label = MODE_LABELS[selected]
    if selected == MODE_DFLASH2:
        settings = report["runtime"]["speculation_settings"][selected]
        width = settings["proposal_drafts"]
        if settings.get("draft_policy") != "adaptive" or settings.get("fixed_drafts") is not None:
            raise ChartRenderError("DFlash2 chart source is not adaptive")
        mode_label += f" adaptive K1-K{width}"
    serving = report["results"]["serving"]
    prefill = indexed_cells(
        serving["prefill"]["cells"],
        keys=("base_context_tokens", "suffix_tokens"),
        label="prefill",
    )
    expected_prefill = {
        (base, suffix)
        for base in DEFAULT_BASE_CONTEXTS
        for suffix in DEFAULT_SUFFIX_ROWS
    }
    if set(prefill) != expected_prefill:
        raise ChartRenderError("prefill chart requires the complete 5x6 matrix")
    context = indexed_cells(
        report["results"]["context_decode"]["cells"],
        keys=("context_bucket_tokens", "workload"),
        label="context decode",
    )
    expected_context = {
        (base, workload) for base in DEFAULT_CONTEXTS for workload in WORKLOADS
    }
    if set(context) != expected_context:
        raise ChartRenderError("decode chart requires the complete 5x3 matrix")

    prefill_svg = line_chart(
        title=f"EXL3 K4 · {mode_label} prefill",
        subtitle="Median new-suffix throughput over retained KV-cache contexts",
        x_labels=[context_label(value) for value in DEFAULT_SUFFIX_ROWS],
        series=[
            (
                f"{context_label(base)} cached",
                [
                    dflash2_prefill_value(prefill[(base, suffix)])
                    for suffix in DEFAULT_SUFFIX_ROWS
                ],
            )
            for base in DEFAULT_BASE_CONTEXTS
        ],
        x_axis="new suffix tokens",
    )
    workload_labels = {
        "code": "Python code",
        "writing": "Creative writing",
        "math": "Math",
    }
    decode_svg = line_chart(
        title=f"EXL3 K4 · {mode_label} decode",
        subtitle="Pooled deterministic decode throughput across retained context",
        x_labels=[context_label(value) for value in DEFAULT_CONTEXTS],
        series=[
            (
                workload_labels[workload],
                [
                    finite(
                        context[(base, workload)]["decode_tps"],
                        "context decode throughput",
                        minimum=0.000001,
                    )
                    for base in DEFAULT_CONTEXTS
                ],
            )
            for workload in WORKLOADS
        ],
        x_axis="retained context tokens",
    )
    return prefill_svg, decode_svg


def regular_outputs(paths: tuple[Path, ...]) -> tuple[Path, ...]:
    resolved = tuple(path.expanduser().resolve() for path in paths)
    if len(set(resolved)) != len(resolved):
        raise ChartRenderError("chart output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in resolved):
        raise ChartRenderError("refusing to overwrite chart output")
    return resolved


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render(
    *,
    evidence_path: Path,
    prefill_output: Path,
    decode_output: Path,
    report_output: Path,
) -> dict[str, Any]:
    prefill_path, decode_path, rendered_report_path = regular_outputs(
        (prefill_output, decode_output, report_output)
    )
    evidence_file, evidence = validate_evidence(evidence_path)
    prefill_svg, decode_svg = chart_inputs(evidence)
    body = {
        "schema": SCHEMA,
        "status": "rendered",
        "model_id": evidence["model_id"],
        "model_revision": evidence["model_revision"],
        "default_speculation": evidence["default_speculation"],
        "source": evidence_identity(evidence_file, evidence["schema"]),
        "charts": {
            "prefill": {
                "path": os.fspath(prefill_path),
                "bytes": len(prefill_svg),
                "sha256": hashlib.sha256(prefill_svg).hexdigest(),
            },
            "decode": {
                "path": os.fspath(decode_path),
                "bytes": len(decode_svg),
                "sha256": hashlib.sha256(decode_svg).hexdigest(),
            },
        },
    }
    report = body | {
        "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()
    }
    atomic_bytes(prefill_path, prefill_svg)
    atomic_bytes(decode_path, decode_svg)
    atomic_bytes(
        rendered_report_path,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--prefill-output", type=Path, required=True)
    parser.add_argument("--decode-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = render(
        evidence_path=args.evidence,
        prefill_output=args.prefill_output,
        decode_output=args.decode_output,
        report_output=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
