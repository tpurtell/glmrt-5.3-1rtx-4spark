#!/usr/bin/env python3
"""Build a route-qualified dSpark verification-cost profile from server logs.

The startup sweep provides complete C/M coverage.  Routed corpus observations
are standardized to a declared semantic route distribution with independently
replayed route-cost sensitivity.  The resulting arithmetic mean estimates the
causal E[T | C,M] available before target routing and can render the Rust table
consumed by the serving scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


STARTUP_RE = re.compile(
    r"real_full_dspark_sps_profile requests=(?P<requests>\d+) "
    r"target_rows=(?P<rows>\d+) latency_ms=(?P<ms>[0-9.]+) "
    r"samples=(?P<samples>\d+) source=startup-opt-in"
)
CORPUS_RE = re.compile(
    r"real_full_dspark_runtime_cost requests=(?P<requests>\d+) "
    r"context_work_bucket=(?P<work>\d+) max_context_bucket=(?P<maximum>\d+) "
    r"target_rows=(?P<rows>\d+) observed_ms=(?P<ms>[0-9.]+) "
    r"predicted_ms_before=(?P<predicted>[0-9.]+) exact_samples=(?P<samples>\d+)"
    r"(?P<route>.*)$"
)
ROUTE_FIELD_RE = re.compile(
    r"route_wire_batches=(?P<wire_batches>\d+) "
    r"route_assignments=(?P<assignments>\d+) "
    r"route_unique_experts=(?P<unique>\d+) "
    r"route_critical_unique_experts=(?P<critical_unique>\d+) "
    r"route_reused_assignments=(?P<reused>\d+) "
    r"route_max_expert_load=(?P<maximum>\d+) "
    r"route_load_square_sum=(?P<squares>\d+)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a complete startup dSpark C/M sweep with route-qualified "
            "short-context corpus observations."
        )
    )
    parser.add_argument("--startup-log", type=Path, required=True, action="append")
    parser.add_argument(
        "--corpus-log",
        required=True,
        action="append",
        metavar="[C=]PATH",
        help=(
            "runtime trace; prefix with a request count such as 4=trace.log "
            "to reject shrinking-tail observations from other concurrencies"
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-rust", type=Path)
    parser.add_argument(
        "--rust-constant-prefix",
        default="GLM52_REDHAT_DSPARK_COST_PROFILE",
        help=(
            "uppercase Rust constant prefix for --output-rust; the default "
            "preserves the original GLM-5.2 dSpark artifact"
        ),
    )
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--dspark-model", required=True)
    parser.add_argument("--dspark-revision", required=True)
    parser.add_argument("--sparkinfer-revision", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--power-limit-watts", type=int, required=True)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-drafts", type=int, default=7)
    parser.add_argument("--minimum-corpus-samples", type=int, default=5)
    parser.add_argument(
        "--route-reference",
        type=Path,
        help=(
            "counterfactual C/M route distribution from "
            "build_dspark_route_reference.py; required to promote corpus cells"
        ),
    )
    parser.add_argument(
        "--route-replay-plan",
        type=Path,
        help="exact-route replay plan paired with --route-replay-result",
    )
    parser.add_argument(
        "--route-replay-result",
        type=Path,
        help="75-layer replay result used to fit route-cost sensitivity",
    )
    parser.add_argument(
        "--startup-samples",
        type=int,
        help="accept only startup-sweep cells with this per-cell sample count",
    )
    parser.add_argument(
        "--baseline-profile",
        type=Path,
        help="previous qualified profile used for performance-gated fallback",
    )
    parser.add_argument(
        "--retain-baseline-concurrency",
        type=int,
        action="append",
        default=[],
        help=(
            "retain this concurrency's previous qualified curve after the "
            "route-aware candidate failed or lacked a performance gate"
        ),
    )
    return parser.parse_args()


def read_lines(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        with path.open(encoding="utf-8", errors="replace") as stream:
            yield from stream


def parse_startup(
    paths: Iterable[Path], required_samples: int | None = None
) -> dict[tuple[int, int], dict[str, Any]]:
    cells: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for line in read_lines(paths):
        match = STARTUP_RE.search(line)
        if match is None:
            continue
        if required_samples is not None and int(match["samples"]) != required_samples:
            continue
        key = (int(match["requests"]), int(match["rows"]))
        cells[key].append((float(match["ms"]), int(match["samples"])))
    return {
        key: {
            "latency_ms": statistics.median(value for value, _ in samples),
            "sweeps": len(samples),
            "samples_per_sweep": [sample_count for _, sample_count in samples],
        }
        for key, samples in cells.items()
    }


def percentile(values: list[float], fraction: float) -> float:
    if len(values) == 1:
        return values[0]
    position = fraction * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def corpus_log_specs(values: Iterable[str]) -> list[tuple[Path, int | None]]:
    specs = []
    for value in values:
        prefix, separator, remainder = value.partition("=")
        if separator and prefix.isdigit():
            request_count = int(prefix)
            if request_count < 1 or not remainder:
                raise SystemExit(f"invalid concurrency-scoped corpus log: {value}")
            specs.append((Path(remainder), request_count))
        else:
            specs.append((Path(value), None))
    return specs


def parse_corpus(
    specs: Iterable[tuple[Path, int | None]],
) -> dict[tuple[int, int], dict[str, Any]]:
    cells: dict[tuple[int, int], list[dict[str, float | int] | None]] = defaultdict(list)
    ignored_context_samples = 0
    for path, required_request_count in specs:
        for line in read_lines([path]):
            match = CORPUS_RE.search(line)
            if match is None:
                continue
            request_count = int(match["requests"])
            if (
                required_request_count is not None
                and request_count != required_request_count
            ):
                continue
            if int(match["work"]) != 0 or int(match["maximum"]) != 0:
                ignored_context_samples += 1
                continue
            key = (request_count, int(match["rows"]))
            route_match = ROUTE_FIELD_RE.search(match["route"])
            cells[key].append(
                {
                    "latency_ms": float(match["ms"]),
                    "wire_batches": int(route_match["wire_batches"]),
                    "route_assignments": int(route_match["assignments"]),
                    "unique_experts": int(route_match["unique"]),
                    "critical_unique_experts": int(route_match["critical_unique"]),
                    "reused_assignments": int(route_match["reused"]),
                    "max_expert_load": int(route_match["maximum"]),
                    "load_square_sum": int(route_match["squares"]),
                }
                if route_match is not None
                else {"latency_ms": float(match["ms"])}
            )
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for key, samples in cells.items():
        values = sorted(float(sample["latency_ms"]) for sample in samples)
        route_samples = [sample for sample in samples if "wire_batches" in sample]
        result[key] = {
            "samples": len(values),
            "median_ms": statistics.median(values),
            "minimum_ms": values[0],
            "p25_ms": percentile(values, 0.25),
            "p75_ms": percentile(values, 0.75),
            "maximum_ms": values[-1],
            "route_profiled_samples": len(route_samples),
            "_observations": samples,
        }
        if route_samples:
            result[key]["route_shape"] = {
                name: {
                    "minimum": min(float(sample[name]) for sample in route_samples),
                    "median": statistics.median(
                        float(sample[name]) for sample in route_samples
                    ),
                    "maximum": max(float(sample[name]) for sample in route_samples),
                }
                for name in (
                    "wire_batches",
                    "route_assignments",
                    "unique_experts",
                    "critical_unique_experts",
                    "reused_assignments",
                    "max_expert_load",
                    "load_square_sum",
                )
            }
    result[(0, 0)] = {"ignored_nonzero_context_samples": ignored_context_samples}
    return result


def load_route_reference(
    path: Path | None, max_concurrency: int, max_drafts: int
) -> dict[tuple[int, int], dict[str, Any]]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "glmrt-dspark-route-reference-v1":
        raise SystemExit(f"unsupported route-reference schema in {path}")
    if (
        int(value["max_concurrency"]) != max_concurrency
        or int(value["max_drafts"]) != max_drafts
    ):
        raise SystemExit("route reference does not cover the requested C/M surface")
    cells = {}
    for raw in value["cells"].values():
        key = (int(raw["requests"]), int(raw["target_rows"]))
        cells[key] = raw
    return cells


def route_replay_sensitivity(
    plan_path: Path | None, result_path: Path | None
) -> dict[str, Any] | None:
    if plan_path is None and result_path is None:
        return None
    if plan_path is None or result_path is None:
        raise SystemExit(
            "--route-replay-plan and --route-replay-result must be supplied together"
        )
    chain_unique: dict[str, tuple[int, int]] = {}
    with plan_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record.get("record") != "chain":
                continue
            critical_unique = sum(
                len({expert for row in layer["routes"] for expert in row})
                for layer in record["layers"]
            )
            chain_unique[str(record["chain_id"])] = (
                int(record["physical_m"]),
                critical_unique,
            )
    rows = []
    with result_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if (
                record.get("record") != "measurement"
                or record.get("path") != "coordinator"
            ):
                continue
            chain_id = str(record["chain_id"])
            if chain_id not in chain_unique:
                continue
            physical_m, unique = chain_unique[chain_id]
            rows.append((physical_m, float(unique), float(record["dispatch_ms"])))
    if len(rows) < 32:
        raise SystemExit("route replay has fewer than 32 joined coordinator measurements")
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for physical_m, unique, latency_ms in rows:
        grouped[physical_m].append((unique, latency_ms))
    numerator = 0.0
    denominator = 0.0
    total_latency_variance = 0.0
    residual_variance = 0.0
    slopes = {}
    for physical_m, values in grouped.items():
        if len(values) < 2:
            continue
        mean_unique = statistics.mean(unique for unique, _ in values)
        mean_latency = statistics.mean(latency for _, latency in values)
        local_num = sum(
            (unique - mean_unique) * (latency - mean_latency)
            for unique, latency in values
        )
        local_den = sum((unique - mean_unique) ** 2 for unique, _ in values)
        if local_den > 0.0:
            slopes[str(physical_m)] = local_num / local_den
            numerator += local_num
            denominator += local_den
    if denominator == 0.0:
        raise SystemExit("route replay contains no within-M route variation")
    slope = numerator / denominator
    for physical_m, values in grouped.items():
        mean_unique = statistics.mean(unique for unique, _ in values)
        mean_latency = statistics.mean(latency for _, latency in values)
        for unique, latency in values:
            total_latency_variance += (latency - mean_latency) ** 2
            residual_variance += (
                latency - mean_latency - slope * (unique - mean_unique)
            ) ** 2
    return {
        "plan": str(plan_path),
        "plan_sha256": file_sha256(plan_path),
        "result": str(result_path),
        "result_sha256": file_sha256(result_path),
        "measurements": len(rows),
        "physical_ms": sorted(grouped),
        "latency_ms_per_critical_unique_expert": slope,
        "within_m_r_squared": (
            1.0 - residual_variance / total_latency_variance
            if total_latency_variance > 0.0
            else 0.0
        ),
        "slopes_by_m": slopes,
    }


def route_standardized_latency(
    corpus_cell: dict[str, Any],
    reference_cell: dict[str, Any],
    route_sensitivity: dict[str, Any],
) -> dict[str, Any]:
    reference_unique = float(
        reference_cell["route_shape"]["critical_unique_experts"]["mean"]
    )
    slope = float(route_sensitivity["latency_ms_per_critical_unique_expert"])
    observations = [
        observation
        for observation in corpus_cell["_observations"]
        if "critical_unique_experts" in observation
    ]
    adjusted = [
        float(observation["latency_ms"])
        + slope
        * (
            reference_unique
            - float(observation["critical_unique_experts"])
        )
        for observation in observations
    ]
    # Remove only gross non-route stalls after standardization.  Route tails
    # themselves are part of the expectation and must not be median-filtered.
    retained = list(adjusted)
    if len(adjusted) >= 9:
        center = statistics.median(adjusted)
        deviations = [abs(value - center) for value in adjusted]
        mad = statistics.median(deviations)
        if mad > 0.0:
            retained = [
                value for value in adjusted if abs(value - center) <= 6.0 * mad
            ]
            if len(retained) < max(5, len(adjusted) * 3 // 4):
                retained = list(adjusted)
    standardized_mean = statistics.mean(retained)
    standardized_stddev = statistics.stdev(retained) if len(retained) > 1 else 0.0
    standardized_standard_error = standardized_stddev / math.sqrt(len(retained))
    return {
        "reference_critical_unique_experts": reference_unique,
        "latency_ms_per_critical_unique_expert": slope,
        "raw_mean_ms": statistics.mean(
            float(observation["latency_ms"]) for observation in observations
        ),
        "standardized_mean_ms": standardized_mean,
        "standardized_median_ms": statistics.median(retained),
        "standardized_minimum_ms": min(retained),
        "standardized_maximum_ms": max(retained),
        "standardized_sample_stddev_ms": standardized_stddev,
        "standardized_standard_error_ms": standardized_standard_error,
        "standardized_mean_95pct_ci_ms": [
            standardized_mean - 1.96 * standardized_standard_error,
            standardized_mean + 1.96 * standardized_standard_error,
        ],
        "route_profiled_samples": len(observations),
        "retained_samples": len(retained),
    }


def validate_complete_startup(
    startup: dict[tuple[int, int], dict[str, Any]],
    max_concurrency: int,
    max_drafts: int,
) -> None:
    missing = []
    for requests in range(1, max_concurrency + 1):
        for rows in range(requests, requests * (max_drafts + 1) + 1):
            if (requests, rows) not in startup:
                missing.append(f"C{requests}/M{rows}")
    if missing:
        raise SystemExit(
            "startup profile is incomplete; missing " + ", ".join(missing)
        )


def build_profile(args: argparse.Namespace) -> dict[str, Any]:
    startup = parse_startup(args.startup_log, args.startup_samples)
    validate_complete_startup(startup, args.max_concurrency, args.max_drafts)
    corpus_specs = corpus_log_specs(args.corpus_log)
    corpus = parse_corpus(corpus_specs)
    ignored_context_samples = corpus.pop((0, 0))["ignored_nonzero_context_samples"]
    route_reference = load_route_reference(
        args.route_reference, args.max_concurrency, args.max_drafts
    )
    route_sensitivity = route_replay_sensitivity(
        args.route_replay_plan, args.route_replay_result
    )
    curves: dict[str, list[dict[str, Any]]] = {}
    corpus_cells = 0
    corpus_samples = 0
    for requests in range(1, args.max_concurrency + 1):
        rows_out = []
        previous_latency_ms = 0.0
        previous_reference_unique: float | None = None
        for rows in range(requests, requests * (args.max_drafts + 1) + 1):
            key = (requests, rows)
            startup_cell = startup[key]
            corpus_cell = corpus.get(key)
            use_corpus = (
                corpus_cell is not None
                and corpus_cell["samples"] >= args.minimum_corpus_samples
                and corpus_cell["route_profiled_samples"]
                >= args.minimum_corpus_samples
                and key in route_reference
                and route_sensitivity is not None
            )
            if use_corpus:
                standardization = route_standardized_latency(
                    corpus_cell, route_reference[key], route_sensitivity
                )
                latency_ms = standardization["standardized_mean_ms"]
                source = "route-standardized-corpus-mean"
                corpus_cells += 1
                corpus_samples += standardization["retained_samples"]
            else:
                latency_ms = startup_cell["latency_ms"]
                source = "startup-sweep-fallback"
                standardization = None
            unconstrained_latency_ms = latency_ms
            reference_unique = (
                float(
                    route_reference[key]["route_shape"][
                        "critical_unique_experts"
                    ]["mean"]
                )
                if key in route_reference
                else None
            )
            route_work_increment_ms = 0.0
            if (
                reference_unique is not None
                and previous_reference_unique is not None
                and route_sensitivity is not None
            ):
                route_work_increment_ms = max(
                    0.0, reference_unique - previous_reference_unique
                ) * float(
                    route_sensitivity[
                        "latency_ms_per_critical_unique_expert"
                    ]
                )
            route_work_floor_ms = previous_latency_ms + route_work_increment_ms
            latency_ms = max(latency_ms, route_work_floor_ms)
            previous_latency_ms = latency_ms
            previous_reference_unique = reference_unique
            corpus_evidence = None
            if corpus_cell is not None:
                corpus_evidence = {
                    name: value
                    for name, value in corpus_cell.items()
                    if name != "_observations"
                }
            rows_out.append(
                {
                    "target_rows": rows,
                    "latency_ms": round(latency_ms, 6),
                    "unconstrained_latency_ms": round(
                        unconstrained_latency_ms, 6
                    ),
                    "monotonic_adjustment_ms": round(
                        latency_ms - unconstrained_latency_ms, 6
                    ),
                    "route_work_increment_ms": round(
                        route_work_increment_ms, 6
                    ),
                    "route_work_floor_ms": round(route_work_floor_ms, 6),
                    "source": source,
                    "startup": startup_cell,
                    "corpus": corpus_evidence,
                    "route_standardization": standardization,
                    "route_reference": route_reference.get(key),
                }
            )
        curves[str(requests)] = rows_out
    retained_concurrencies = sorted(set(args.retain_baseline_concurrency))
    baseline_profile = None
    if retained_concurrencies:
        if args.baseline_profile is None:
            raise SystemExit(
                "--retain-baseline-concurrency requires --baseline-profile"
            )
        baseline_profile = json.loads(
            args.baseline_profile.read_text(encoding="utf-8")
        )
        baseline_identity = baseline_profile["identity"]
        expected_identity = {
            "target_model": args.target_model,
            "target_revision": args.target_revision,
            "dspark_model": args.dspark_model,
            "dspark_revision": args.dspark_revision,
            "sparkinfer_revision": args.sparkinfer_revision,
            "topology": args.topology,
            "power_limit_watts": args.power_limit_watts,
            "max_concurrency": args.max_concurrency,
            "max_drafts": args.max_drafts,
        }
        for name, expected in expected_identity.items():
            if baseline_identity.get(name) != expected:
                raise SystemExit(
                    f"baseline profile identity mismatch for {name}: "
                    f"{baseline_identity.get(name)!r} != {expected!r}"
                )
        for requests in retained_concurrencies:
            if not 1 <= requests <= args.max_concurrency:
                raise SystemExit(
                    f"retained baseline concurrency {requests} is out of range"
                )
            baseline_cells = baseline_profile["curves"][str(requests)]
            if len(baseline_cells) != len(curves[str(requests)]):
                raise SystemExit(
                    f"baseline C{requests} curve has incompatible coverage"
                )
            for candidate_cell, baseline_cell in zip(
                curves[str(requests)], baseline_cells, strict=True
            ):
                if candidate_cell["target_rows"] != baseline_cell["target_rows"]:
                    raise SystemExit(
                        f"baseline C{requests} target-row coverage is incompatible"
                    )
                candidate_cell["candidate_latency_ms"] = candidate_cell[
                    "latency_ms"
                ]
                candidate_cell["candidate_source"] = candidate_cell["source"]
                candidate_cell["latency_ms"] = baseline_cell["latency_ms"]
                candidate_cell["source"] = "performance-gated-baseline"
                candidate_cell["performance_gate"] = {
                    "baseline_profile_id": baseline_profile["profile_id"],
                    "baseline_source_sha256": baseline_profile.get(
                        "source_sha256"
                    ),
                    "baseline_cell_source": baseline_cell["source"],
                }
    adopted_corpus_cells = sum(
        cell["source"] == "route-standardized-corpus-mean"
        for cells in curves.values()
        for cell in cells
    )
    return {
        "schema": "glmrt-dspark-cost-profile-v1",
        "profile_id": args.profile_id,
        "identity": {
            "target_model": args.target_model,
            "target_revision": args.target_revision,
            "dspark_model": args.dspark_model,
            "dspark_revision": args.dspark_revision,
            "sparkinfer_revision": args.sparkinfer_revision,
            "engine_commit": args.engine_commit,
            "topology": args.topology,
            "power_limit_watts": args.power_limit_watts,
            "max_concurrency": args.max_concurrency,
            "max_drafts": args.max_drafts,
        },
        "qualification": {
            "minimum_corpus_samples_per_cell": args.minimum_corpus_samples,
            "startup_samples_per_cell_filter": args.startup_samples,
            "corpus_qualified_cells": corpus_cells,
            "route_qualified_cells_adopted": adopted_corpus_cells,
            "corpus_samples_used": corpus_samples,
            "ignored_nonzero_context_samples": ignored_context_samples,
            "startup_logs": [str(path) for path in args.startup_log],
            "startup_log_sha256": {
                str(path): file_sha256(path) for path in args.startup_log
            },
            "corpus_logs": [
                {
                    "path": str(path),
                    "request_count": request_count,
                    "sha256": file_sha256(path),
                }
                for path, request_count in corpus_specs
            ],
            "route_reference": str(args.route_reference)
            if args.route_reference is not None
            else None,
            "route_reference_source_sha256": (
                json.loads(args.route_reference.read_text(encoding="utf-8"))[
                    "source_sha256"
                ]
                if args.route_reference is not None
                else None
            ),
            "route_sensitivity": route_sensitivity,
            "baseline_profile": str(args.baseline_profile)
            if args.baseline_profile is not None
            else None,
            "baseline_profile_id": baseline_profile["profile_id"]
            if baseline_profile is not None
            else None,
            "retained_baseline_concurrencies": retained_concurrencies,
        },
        "curves": curves,
    }


def render_rust(
    profile: dict[str, Any], source_sha256: str, constant_prefix: str
) -> str:
    identity = profile["identity"]
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", constant_prefix) is None:
        raise SystemExit(
            "--rust-constant-prefix must be an uppercase Rust identifier"
        )
    lines = [
        "// @generated by python/tools/calibrate_dspark_cost_profile.py; do not edit.",
        f'pub(super) const {constant_prefix}_ID: &str = {json.dumps(profile["profile_id"])};',
        f'pub(super) const {constant_prefix}_SOURCE_SHA256: &str = "{source_sha256}";',
        f'pub(super) const {constant_prefix}_TARGET_MODEL: &str = {json.dumps(identity["target_model"])};',
        f'pub(super) const {constant_prefix}_TARGET_REVISION: &str = {json.dumps(identity["target_revision"])};',
        f'pub(super) const {constant_prefix}_DSPARK_MODEL: &str = {json.dumps(identity["dspark_model"])};',
        f'pub(super) const {constant_prefix}_DSPARK_REVISION: &str = {json.dumps(identity["dspark_revision"])};',
        f'pub(super) const {constant_prefix}_SPARKINFER_REVISION: &str = {json.dumps(identity["sparkinfer_revision"])};',
        f'pub(super) const {constant_prefix}_TOPOLOGY: &str = {json.dumps(identity["topology"])};',
        f"pub(super) const {constant_prefix}_POWER_LIMIT_WATTS: usize = {identity['power_limit_watts']};",
        f"pub(super) const {constant_prefix}_MAX_CONCURRENCY: usize = {identity['max_concurrency']};",
        f"pub(super) const {constant_prefix}_MAX_DRAFTS: usize = {identity['max_drafts']};",
        f"pub(super) static {constant_prefix}_MS: [&[(usize, f64)]; {constant_prefix}_MAX_CONCURRENCY] = [",
    ]
    for requests in range(1, identity["max_concurrency"] + 1):
        lines.append("    &[")
        for cell in profile["curves"][str(requests)]:
            lines.append(
                f"        ({cell['target_rows']}, {cell['latency_ms']:.6}), // {cell['source']}"
            )
        lines.append("    ],")
    lines.extend(["];"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.max_concurrency < 1 or args.max_drafts < 1:
        raise SystemExit("--max-concurrency and --max-drafts must be positive")
    if args.minimum_corpus_samples < 1:
        raise SystemExit("--minimum-corpus-samples must be positive")
    profile = build_profile(args)
    canonical = json.dumps(profile, ensure_ascii=False, sort_keys=True).encode()
    source_sha256 = hashlib.sha256(canonical).hexdigest()
    profile["source_sha256"] = source_sha256
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_rust is not None:
        args.output_rust.parent.mkdir(parents=True, exist_ok=True)
        args.output_rust.write_text(
            render_rust(profile, source_sha256, args.rust_constant_prefix),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "profile_id": profile["profile_id"],
                "source_sha256": source_sha256,
                **profile["qualification"],
                "output_json": str(args.output_json),
                "output_rust": str(args.output_rust) if args.output_rust else None,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
