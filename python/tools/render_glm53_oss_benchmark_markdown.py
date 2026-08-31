#!/usr/bin/env python3
"""Render the complete GLM-5.3 OSS benchmark section from signed evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any

from bench_release_decode_matrix import DEFAULT_CONTEXTS, WORKLOADS
from bench_release_prefill_matrix import (
    DEFAULT_BASE_CONTEXTS,
    DEFAULT_SUFFIX_ROWS,
)
from validate_glm53_exl3_serving_qualification import (
    GLM53_MODEL_ID,
    MODE_DFLASH2,
    MODE_NATIVE_MTP,
    REQUIRED_CONCURRENCIES,
    REQUIRED_NEEDLE_CONTEXTS,
)
from validate_glm52_exl3_serving_qualification import evidence_identity
from validate_glm53_oss_release_evidence import (
    SCHEMA as OSS_SCHEMA,
    canonical_json,
    revalidate_identities,
    signed_report,
)
from validate_glm53_profile_release_evidence import PROFILES


class BenchmarkRenderError(RuntimeError):
    """The accepted OSS evidence cannot produce a complete benchmark section."""


MODE_LABELS = {
    MODE_NATIVE_MTP: "Native MTP",
    MODE_DFLASH2: "DFlash2",
}
CHART_SCHEMA = "glmrt-glm53-oss-benchmark-charts-v1"


def finite(value: Any, label: str, *, minimum: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise BenchmarkRenderError(f"{label} is not numeric") from error
    if not minimum <= number < float("inf"):
        raise BenchmarkRenderError(f"{label} is outside its valid range")
    return number


def integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BenchmarkRenderError(f"{label} is not a valid integer")
    return value


def tps(value: Any) -> str:
    return f"{finite(value, 'throughput', minimum=0.000001):.2f} tok/s"


def percent(value: Any) -> str:
    rate = finite(value, "acceptance")
    if rate > 1.0:
        raise BenchmarkRenderError("acceptance exceeds one")
    return f"{rate * 100.0:.1f}%"


def numeric_tps(value: Any, label: str) -> str:
    return f"{finite(value, label, minimum=0.000001):.2f}"


def context_label(value: int) -> str:
    if value == 0:
        return "0"
    if value % 1_024 != 0:
        raise BenchmarkRenderError("context size is not an integral Ki-token value")
    return f"{value // 1_024}K"


def dflash2_prefill_value(cell: dict[str, Any]) -> float:
    return finite(cell.get("dflash2_tps"), "DFlash2 prefill throughput")


def table(headers: list[str], rows: list[list[str]], aligns: str | None = None) -> str:
    if not rows or any(len(row) != len(headers) for row in rows):
        raise BenchmarkRenderError("Markdown table has an invalid shape")
    if aligns is None:
        aligns = "l" + "r" * (len(headers) - 1)
    if len(aligns) != len(headers) or any(char not in "lrc" for char in aligns):
        raise BenchmarkRenderError("Markdown table alignment is invalid")
    separator = {
        "l": "---",
        "r": "---:",
        "c": ":---:",
    }
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(separator[char] for char in aligns) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def indexed_cells(
    cells: Any,
    *,
    keys: tuple[str, ...],
    label: str,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    if not isinstance(cells, list):
        raise BenchmarkRenderError(f"{label} cells are missing")
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise BenchmarkRenderError(f"{label} cell is not an object")
        key = tuple(cell.get(field) for field in keys)
        if key in result:
            raise BenchmarkRenderError(f"{label} has duplicate cell {key}")
        result[key] = cell
    return result


def validate_evidence(path: Path) -> tuple[Path, dict[str, Any]]:
    try:
        resolved, report = signed_report(
            path, schema=OSS_SCHEMA, statuses={"accepted"}
        )
        revalidate_identities(report.get("evidence"), checked={})
    except RuntimeError as error:
        raise BenchmarkRenderError("OSS release evidence is invalid") from error
    if (
        report.get("model_id") != GLM53_MODEL_ID
        or report.get("default_speculation") not in MODE_LABELS
        or report.get("runtime", {}).get("profile") != "balanced"
    ):
        raise BenchmarkRenderError("OSS evidence does not select GLM-5.3 balanced")
    return resolved, report


def validate_charts(
    path: Path,
    *,
    evidence_file: Path,
    evidence: dict[str, Any],
) -> dict[str, Path]:
    try:
        _resolved, report = signed_report(
            path, schema=CHART_SCHEMA, statuses={"rendered"}
        )
        revalidate_identities(report, checked={})
    except RuntimeError as error:
        raise BenchmarkRenderError("benchmark chart evidence is invalid") from error
    expected_source = evidence_identity(evidence_file, evidence["schema"])
    if (
        report.get("model_id") != evidence["model_id"]
        or report.get("model_revision") != evidence["model_revision"]
        or report.get("default_speculation") != evidence["default_speculation"]
        or report.get("source") != expected_source
    ):
        raise BenchmarkRenderError("benchmark charts use different OSS evidence")
    charts = report.get("charts")
    if not isinstance(charts, dict) or set(charts) != {"prefill", "decode"}:
        raise BenchmarkRenderError("benchmark chart set is incomplete")
    return {
        name: Path(charts[name]["path"]).expanduser().resolve(strict=True)
        for name in ("prefill", "decode")
    }


def render_markdown(
    report: dict[str, Any],
    *,
    output_parent: Path,
    charts: dict[str, Path] | None = None,
) -> str:
    try:
        selected = report["default_speculation"]
        results = report["results"]
        serving = results["serving"]
        modes = serving["modes"]
        prefill = serving["prefill"]["cells"]
        semantic = serving["semantic_decode"]["cells"]
        context_decode = results["context_decode"]["cells"]
        agentic = results["agentic"]
        profile_results = results["profiles"]["results"]
        startup = results["startup"]
        micro = results["micro_timeline"]
    except (KeyError, TypeError) as error:
        raise BenchmarkRenderError("OSS evidence results are incomplete") from error
    if set(modes) != set(MODE_LABELS):
        raise BenchmarkRenderError(
            "serving results do not contain both speculation modes"
        )

    prefill_by_key = indexed_cells(
        prefill,
        keys=("base_context_tokens", "suffix_tokens"),
        label="prefill",
    )
    expected_prefill = {
        (base, suffix)
        for base in DEFAULT_BASE_CONTEXTS
        for suffix in DEFAULT_SUFFIX_ROWS
    }
    if set(prefill_by_key) != expected_prefill:
        raise BenchmarkRenderError("prefill evidence is not the complete 5x6 matrix")
    context_by_key = indexed_cells(
        context_decode,
        keys=("context_bucket_tokens", "workload"),
        label="context decode",
    )
    expected_context = {
        (context, workload) for context in DEFAULT_CONTEXTS for workload in WORKLOADS
    }
    if set(context_by_key) != expected_context:
        raise BenchmarkRenderError(
            "context decode evidence is not the complete 5x3 matrix"
        )

    revision = str(report["model_revision"])
    settings = report["runtime"]["speculation_settings"].get(selected, {})
    selected_detail = MODE_LABELS[selected]
    if selected == MODE_DFLASH2:
        width = integer(
            settings.get("proposal_drafts"),
            "selected DFlash2 proposal width",
            minimum=1,
        )
        if settings.get("draft_policy") != "adaptive" or settings.get("fixed_drafts") is not None:
            raise BenchmarkRenderError("selected DFlash2 policy is not adaptive")
        if any(
            not isinstance(settings.get(field), str) or not settings[field]
            for field in ("checkpoint_model_id", "checkpoint_revision", "topk_backend")
        ):
            raise BenchmarkRenderError("selected DFlash2 checkpoint identity is incomplete")
        selected_detail += f" adaptive K1-K{width}"

    lines = [
        "# GLM-5.3 EXL3 K4 release benchmarks",
        "",
        (
            f"Model `{GLM53_MODEL_ID}` at revision `{revision}`. The measured "
            f"balanced-profile default is **{selected_detail}**. All figures below "
            "come from one recursively source-validated OSS evidence report."
        ),
        "",
        (
            "Hardware: one RTX PRO 6000 Blackwell coordinator at "
            f"{integer(report['runtime']['power_limit_w'], 'power limit', minimum=1)} W "
            "and four resident DGX Spark expert workers. DFlash2 uses "
            f"`{settings.get('checkpoint_model_id')}` at revision "
            f"`{settings.get('checkpoint_revision')}` with "
            f"`{settings.get('topk_backend')}` top-k."
        ),
        "",
        "## High-level performance",
        "",
    ]

    high_rows = []
    for mode in MODE_LABELS:
        mode_result = modes[mode]
        high_rows.append(
            [
                MODE_LABELS[mode] + (" (default)" if mode == selected else ""),
                tps(mode_result["weighted_decode_tps"]),
                tps(mode_result["agentic_code_decode_tps"]),
                tps(mode_result["repeat_decode_tps"]),
                percent(mode_result["accepted_draft_rate"]),
            ]
        )
    lines.extend(
        [
            table(
                [
                    "Speculation",
                    "Weighted decode",
                    "Python code",
                    "Orchid repeat",
                    "Acceptance",
                ],
                high_rows,
            ),
            "",
            (
                "Weighted decode pools five replays of eight semantic workload types. "
                "Orchid is a low-entropy throughput probe; it is not a "
                "counting-quality test."
            ),
            "",
        ]
    )
    policy = serving.get("dflash2_adaptive")
    if not isinstance(policy, dict):
        raise BenchmarkRenderError("DFlash2 adaptive-policy evidence is incomplete")
    adaptive_weighted = finite(
        modes[MODE_DFLASH2]["weighted_decode_tps"],
        "adaptive weighted decode",
        minimum=0.000001,
    )
    weighted_ratio = finite(
        policy.get("weighted_decode_ratio_vs_k5"),
        "adaptive weighted ratio",
        minimum=0.000001,
    )
    adaptive_geomean = finite(
        policy.get("concurrency_geomean_tps"),
        "adaptive concurrency geomean",
        minimum=0.000001,
    )
    fixed_geomean = finite(
        policy.get("k5_concurrency_geomean_tps"),
        "fixed-K5 concurrency geomean",
        minimum=0.000001,
    )
    adaptive_score = finite(
        policy.get("response_performance_score"),
        "adaptive response score",
        minimum=0.000001,
    )
    fixed_score = finite(
        policy.get("k5_response_performance_score"),
        "fixed-K5 response score",
        minimum=0.000001,
    )
    reference_width = integer(
        policy.get("reference_width"), "DFlash2 reference width", minimum=1
    )
    lines.extend(
        [
            "### DFlash2 draft-policy selection",
            "",
            table(
                [
                    "Policy",
                    "Weighted decode",
                    "C1/C2/C4 geometric mean",
                    "Response-performance score",
                ],
                [
                    [
                        f"Adaptive K1-K{width} (default)",
                        tps(adaptive_weighted),
                        tps(adaptive_geomean),
                        f"{adaptive_score:.3f}",
                    ],
                    [
                        f"Fixed K{reference_width}",
                        tps(adaptive_weighted / weighted_ratio),
                        tps(fixed_geomean),
                        f"{fixed_score:.3f}",
                    ],
                ],
            ),
            "",
            (
                f"Adaptive improves weighted decode by "
                f"**{(weighted_ratio - 1.0) * 100:.2f}%** over fixed "
                f"K{reference_width}."
            ),
            "",
            "## Eight-type decode and acceptance",
            "",
        ]
    )
    semantic_rows = []
    if not isinstance(semantic, list) or len(semantic) != 8:
        raise BenchmarkRenderError("eight-type semantic decode evidence is incomplete")
    for cell in semantic:
        semantic_rows.append(
            [
                str(cell["category"]),
                str(cell["case"]),
                tps(cell["native_mtp_decode_tps"]),
                percent(cell["native_mtp_accepted_draft_rate"]),
                tps(cell["dflash2_decode_tps"]),
                percent(cell["dflash2_accepted_draft_rate"]),
            ]
        )
    lines.extend(
        [
            table(
                [
                    "Type",
                    "Case",
                    "Native MTP",
                    "MTP acceptance",
                    "DFlash2",
                    "DFlash2 acceptance",
                ],
                semantic_rows,
                "llrrrr",
            ),
            "",
            "## Cache-aware prefill",
            "",
            (
                "Each cell is median new-suffix throughput after retaining the row's "
                "base context in KV cache."
            ),
            "",
        ]
    )
    if charts is not None:
        lines.extend(
            [
                (
                    "![GLM-5.3 EXL3 K4 balanced prefill throughput]"
                    f"({os.path.relpath(charts['prefill'], output_parent)})"
                ),
                "",
            ]
        )
    rows = [
        [context_label(base)]
        + [
            f"{dflash2_prefill_value(prefill_by_key[(base, suffix)]):.0f}"
            for suffix in DEFAULT_SUFFIX_ROWS
        ]
        for base in DEFAULT_BASE_CONTEXTS
    ]
    lines.extend(
        [
            f"### {MODE_LABELS[MODE_DFLASH2]}",
            "",
            table(
                ["Cached context"]
                + [f"+{context_label(suffix)}" for suffix in DEFAULT_SUFFIX_ROWS],
                rows,
            ),
            "",
        ]
    )

    lines.extend(
        [
            f"## Decode across retained context — {selected_detail}",
            "",
            (
                "Each cell pools two deterministic responses. Nonzero base contexts "
                "are primed once and reused by all three workloads."
            ),
            "",
            table(
                ["Context", "Python code", "Creative writing", "Math"],
                [
                    [context_label(context)]
                    + [
                        numeric_tps(
                            context_by_key[(context, workload)]["decode_tps"],
                            "context decode",
                        )
                        for workload in WORKLOADS
                    ]
                    for context in DEFAULT_CONTEXTS
                ],
            ),
            "",
        ]
    )
    if charts is not None:
        lines.extend(
            [
                (
                    "![GLM-5.3 EXL3 K4 balanced decode throughput]"
                    f"({os.path.relpath(charts['decode'], output_parent)})"
                ),
                "",
            ]
        )
    lines.extend(["## Decode concurrency", ""])
    concurrency_rows = []
    cells = modes[MODE_DFLASH2].get("decode_concurrency")
    if not isinstance(cells, dict) or set(cells) != {
        str(value) for value in REQUIRED_CONCURRENCIES
    }:
        raise BenchmarkRenderError("DFlash2 concurrency evidence is incomplete")
    base = finite(
        cells["1"]["median_aggregate_decode_tps"],
        "C1 decode",
        minimum=0.000001,
    )
    for concurrency in REQUIRED_CONCURRENCIES:
        value = finite(
            cells[str(concurrency)]["median_aggregate_decode_tps"],
            f"DFlash2 C{concurrency} decode",
            minimum=0.000001,
        )
        concurrency_rows.append(
            [
                str(concurrency),
                tps(value),
                f"{value / base:.2f}x",
            ]
        )
    lines.extend(
        [
            table(
                ["Concurrent requests", "Median aggregate", "Scaling"],
                concurrency_rows,
                "rrr",
            ),
            "",
            f"## Agentic evaluation — {selected_detail}",
            "",
        ]
    )
    tool = agentic.get("tool_eval")
    if (
        not isinstance(tool, dict)
        or len(tool.get("runs", [])) != 3
        or len(tool.get("seeds", [])) != 3
    ):
        raise BenchmarkRenderError(
            "publication tool-evaluation evidence is incomplete"
        )
    tool_max = integer(tool.get("maximum_points"), "tool maximum", minimum=1)
    lines.extend(
        [
            table(
                ["Seed", "Points", "Displayed score"],
                [
                    [
                        str(seed),
                        f"{integer(run['points'], 'tool points')}/{tool_max}",
                        str(run["score"]),
                    ]
                    for seed, run in zip(tool["seeds"], tool["runs"], strict=True)
                ],
            ),
            "",
            (
                f"Median: **{finite(tool['median_points'], 'median tool points'):g}/"
                f"{tool_max}**."
            ),
            "",
            "### Pi coding-agent task",
            "",
        ]
    )
    pi = agentic.get("pi")
    if not isinstance(pi, dict) or set(pi) != {"off", "high"}:
        raise BenchmarkRenderError("Pi agent evidence is incomplete")
    pi_rows = []
    for thinking in ("off", "high"):
        run = pi[thinking]
        usage = run["usage"]
        artifact = run["artifact"]
        artifact_kib = (
            integer(artifact["bytes"], "Pi artifact bytes", minimum=1) / 1_024
        )
        pi_rows.append(
            [
                thinking.title(),
                f"{finite(run['wall_seconds'], 'Pi wall time'):.2f} s",
                str(integer(run["turns"], "Pi turns", minimum=1)),
                str(integer(run["tool_calls"], "Pi tool calls", minimum=1)),
                str(integer(run["tool_errors"], "Pi tool errors")),
                f"{integer(usage['fresh_input'], 'Pi fresh input'):,}",
                f"{integer(usage['cache_read'], 'Pi cache read'):,}",
                f"{integer(usage['output'], 'Pi output'):,}",
                f"{integer(usage['reasoning'], 'Pi reasoning'):,}",
                f"{integer(usage['total'], 'Pi total'):,}",
                f"{artifact_kib:.1f} KB",
            ]
        )
    lines.extend(
        [
            table(
                [
                    "Reasoning",
                    "Wall time",
                    "Turns",
                    "Tool calls",
                    "Tool errors",
                    "Fresh input",
                    "Cache read",
                    "Output",
                    "Reasoning tokens",
                    "Total",
                    "File",
                ],
                pi_rows,
            ),
            "",
            "## Long-context needle recall",
            "",
        ]
    )
    needle_rows = []
    measurements = modes[MODE_DFLASH2]["long_context_needle"]["measurements"]
    by_context: dict[int, list[float]] = {}
    for measurement in measurements:
        context = integer(measurement.get("context_tokens"), "needle context")
        by_context.setdefault(context, []).append(
            finite(measurement.get("wall_seconds"), "needle wall", minimum=0.000001)
        )
    if set(by_context) != set(REQUIRED_NEEDLE_CONTEXTS) or any(
        len(values) != 3 for values in by_context.values()
    ):
        raise BenchmarkRenderError("DFlash2 needle evidence is incomplete")
    for context in REQUIRED_NEEDLE_CONTEXTS:
        needle_rows.append(
            [
                context_label(context),
                "3/3",
                f"{max(by_context[context]):.2f} s",
            ]
        )
    lines.extend(
        [
            table(
                ["Context", "Exact recalls", "Slowest request"],
                needle_rows,
                "rrr",
            ),
            "",
            "## Performance by serving profile",
            "",
        ]
    )
    profile_rows = []
    if set(profile_results) != set(PROFILES):
        raise BenchmarkRenderError("profile evidence is incomplete")
    for profile in PROFILES:
        cell = profile_results[profile][MODE_DFLASH2]
        profile_rows.append(
            [
                profile.title(),
                tps(cell["weighted_decode_tps"]),
                tps(cell["verify_tokens_per_second"]),
                percent(cell["accepted_draft_rate"]),
                tps(cell["cached_2k_plus_8k_prefill_tps"]),
            ]
        )
    lines.extend(
        [
            table(
                [
                    "Profile",
                    "Weighted decode",
                    "Verify throughput",
                    "Acceptance",
                    "Cached 2K + fresh 8K",
                ],
                profile_rows,
                "lrrrr",
            ),
            "",
            "## Startup and production timing",
            "",
        ]
    )
    for key in ("cold_wall_ms", "warm_wall_ms", "cold_to_warm_ratio"):
        finite(startup.get(key), f"startup {key}", minimum=0.000001)
    selected_request = micro.get("selected_request")
    if not isinstance(selected_request, dict):
        raise BenchmarkRenderError("production micro-timeline selection is missing")
    target_cycles = integer(
        selected_request["target_cycles"], "target cycles", minimum=1
    )
    startup_svg = Path(startup["svg"]["path"]).expanduser().resolve(strict=True)
    micro_svg = Path(micro["svg"]["path"]).expanduser().resolve(strict=True)
    startup_link = os.path.relpath(startup_svg, output_parent)
    micro_link = os.path.relpath(micro_svg, output_parent)
    lines.extend(
        [
            table(
                ["Launch state", "Ready wall time"],
                [
                    [
                        "Cold — reload four expert slabs",
                        f"{startup['cold_wall_ms']:.2f} ms",
                    ],
                    [
                        "Warm — retain matched experts",
                        f"{startup['warm_wall_ms']:.2f} ms",
                    ],
                ],
            ),
            "",
            f"[Startup timeline]({startup_link})",
            "",
            (
                f"The production micro-timeline selects `{selected_request['case']}` "
                f"replay {selected_request['repeat']}: "
                f"{tps(selected_request['decode_tps'])} over "
                f"{finite(selected_request['decode_ms'], 'micro decode time'):.2f} ms "
                f"and {target_cycles} target cycles."
            ),
            "",
            f"[Production micro-timeline]({micro_link})",
            "",
            (
                f"Evidence: `{report['report_sha256']}` · engine "
                f"`{report['runtime']['engine_identity']}` · SparkInfer "
                f"`{report['runtime']['sparkinfer_revision']}` · "
                f"{report['runtime']['power_limit_w']} W coordinator cap."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def atomic_text(path: Path, content: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise BenchmarkRenderError(f"refusing to overwrite output: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--charts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resolved, report = validate_evidence(args.evidence)
    output = args.output.expanduser().resolve()
    charts = (
        validate_charts(args.charts, evidence_file=resolved, evidence=report)
        if args.charts is not None
        else None
    )
    markdown = render_markdown(
        report, output_parent=output.parent, charts=charts
    )
    atomic_text(output, markdown)
    print(output)


if __name__ == "__main__":
    main()
