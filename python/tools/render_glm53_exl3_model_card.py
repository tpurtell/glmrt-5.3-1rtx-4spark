#!/usr/bin/env python3
"""Render the accepted GLM-5.3 EXL3 K4 model card from signed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from validate_glm52_exl3_artifact import (
    GLM53_MODEL_ID,
    GLM53_VALIDATION_SCHEMA,
    ArtifactValidationError,
)
from validate_glm52_exl3_quant_evidence import GLM53_SCHEMA as GLM53_QUANT_SCHEMA
from validate_glm53_exl3_serving_qualification import (
    DEFAULT_SELECTION_POLICY,
    DFLASH2_CONFIG_SHA256,
    DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP,
    DFLASH2_TOPK_SERVICE_SELECTION_POLICY,
    DFLASH2_WEIGHT_LFS_SHA256,
    DFLASH2_SERVING_KV_CAPACITY_TOKENS,
    DFLASH2_SERVING_KV_ELEMENT_BYTES,
    DFLASH2_SERVING_KV_PAGE_SIZE,
    DFLASH2_SERVING_KV_STORAGE,
    MODES,
    REQUIRED_CONCURRENCIES,
    REQUIRED_GATES,
    REQUIRED_NEEDLE_CONTEXTS,
    REQUIRED_NEEDLE_DEPTHS,
    REQUIRED_SEMANTIC_CASE_IDS,
    REQUIRED_SEMANTIC_REPEATS,
    SCHEMA as SERVING_SCHEMA,
    QualificationError,
    revalidate_dflash2_adaptive_evidence,
    revalidate_dflash2_fusion_evidence,
    revalidate_dflash2_topk_evidence,
    revalidate_native_evidence,
)

MARKER = "GLMRT_PUBLICATION_RESULTS_PENDING"
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelCardError(RuntimeError):
    """Qualification evidence cannot produce an accepted GLM-5.3 model card."""


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


def signed_report(path: Path, schema: str) -> tuple[Path, dict[str, Any]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ModelCardError(f"evidence is a symbolic link: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ModelCardError(f"evidence is not one regular file: {resolved}")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCardError(f"evidence is not valid JSON: {resolved}") from error
    digest = report.get("report_sha256") if isinstance(report, dict) else None
    body = (
        {key: value for key, value in report.items() if key != "report_sha256"}
        if isinstance(report, dict)
        else None
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != schema
        or report.get("status") != "accepted"
        or not isinstance(digest, str)
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise ModelCardError(f"evidence is not an accepted signed {schema} report")
    return resolved, report


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelCardError(f"{label} is not a nonnegative integer")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelCardError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ModelCardError(f"{label} is not finite and nonnegative")
    return result


def gib(value: int) -> str:
    return f"{value / (1024**3):,.2f} GiB"


def verify_cycle_curve(value: Any, label: str) -> dict[int, dict[str, float | int]]:
    if not isinstance(value, dict) or not value:
        raise ModelCardError(f"{label} has no physical-M verify-cycle curve")
    result: dict[int, dict[str, float | int]] = {}
    for raw_physical_m, raw_row in value.items():
        try:
            physical_m = int(raw_physical_m)
        except (TypeError, ValueError) as error:
            raise ModelCardError(f"{label} has an invalid physical M") from error
        if str(physical_m) != str(raw_physical_m) or not 1 <= physical_m <= 8:
            raise ModelCardError(f"{label} has an invalid physical M")
        if (
            physical_m in result
            or not isinstance(raw_row, dict)
            or set(raw_row)
            != {
                "samples",
                "total_ms",
                "mean_ms",
                "median_ms",
                "min_ms",
                "max_ms",
            }
        ):
            raise ModelCardError(f"{label} has a malformed physical-M timing row")
        samples = integer(raw_row.get("samples"), f"{label} M{physical_m} samples")
        total_ms = number(raw_row.get("total_ms"), f"{label} M{physical_m} total")
        mean_ms = number(raw_row.get("mean_ms"), f"{label} M{physical_m} mean")
        median_ms = number(raw_row.get("median_ms"), f"{label} M{physical_m} median")
        minimum_ms = number(raw_row.get("min_ms"), f"{label} M{physical_m} minimum")
        maximum_ms = number(raw_row.get("max_ms"), f"{label} M{physical_m} maximum")
        if (
            samples < 1
            or min(total_ms, mean_ms, median_ms, minimum_ms, maximum_ms) <= 0.0
            or not math.isclose(total_ms, samples * mean_ms, rel_tol=1e-9, abs_tol=1e-6)
            or not minimum_ms <= median_ms <= maximum_ms
            or not minimum_ms <= mean_ms <= maximum_ms
        ):
            raise ModelCardError(f"{label} has inconsistent physical-M timings")
        result[physical_m] = {
            "samples": samples,
            "total_ms": total_ms,
            "mean_ms": mean_ms,
            "median_ms": median_ms,
            "min_ms": minimum_ms,
            "max_ms": maximum_ms,
        }
    return result


def render_section(
    *,
    artifact: dict[str, Any],
    quant: dict[str, Any],
    serving: dict[str, Any],
    hub_revision: str | None,
) -> str:
    coverage = quant.get("coverage")
    integrity = quant.get("integrity")
    metrics = quant.get("metrics")
    global_metrics = metrics.get("global") if isinstance(metrics, dict) else None
    runtime = serving.get("runtime")
    results = serving.get("results")
    if (
        not isinstance(coverage, dict)
        or coverage.get("projection_count") != 57_600
        or coverage.get("complete_expert_count") != 75 * 256
        or not isinstance(integrity, dict)
        or integrity.get("tensor_payload_hashes_verified") is not True
        or not isinstance(global_metrics, dict)
        or not isinstance(runtime, dict)
        or not isinstance(results, dict)
    ):
        raise ModelCardError("qualification reports do not prove complete K4 coverage")
    if (
        artifact.get("retained_native_bytes_verified") is not True
        or artifact.get("artifact_manifest_file_hashes_verified") is not True
        or artifact.get("projection_checkpoint_bytes_verified") is not True
        or integer(artifact.get("quantized_modules"), "quantized modules") != 57_600
        or integer(artifact.get("exl3_tensors"), "EXL3 tensors") != 230_400
    ):
        raise ModelCardError("artifact report does not prove final tensor replacement")
    source_metadata = artifact.get("source_metadata")
    if (
        not isinstance(source_metadata, list)
        or len(source_metadata) != 3
        or {
            record.get("name") for record in source_metadata if isinstance(record, dict)
        }
        != {"tokenizer.json", "tokenizer_config.json", "generation_config.json"}
        or any(
            not isinstance(record, dict)
            or set(record) != {"name", "bytes", "sha256"}
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
            for record in source_metadata
        )
    ):
        raise ModelCardError("artifact report does not prove exact source metadata")
    mode_results = results.get("modes")
    comparisons = results.get("comparisons")
    prefill = results.get("prefill")
    semantic_decode = results.get("semantic_decode")
    native_kernel = results.get("native_kernel")
    dflash2_preflight = results.get("dflash2_preflight")
    dflash2_topk_tuning = results.get("dflash2_topk_tuning")
    dflash2_fusion_tuning = results.get("dflash2_fusion_tuning")
    adaptive = results.get("dflash2_adaptive")
    if (
        not isinstance(mode_results, dict)
        or set(mode_results) != set(MODES)
        or not isinstance(comparisons, dict)
        or not isinstance(prefill, dict)
        or not isinstance(semantic_decode, dict)
        or not isinstance(native_kernel, dict)
        or native_kernel.get("trellis_bits") != 4
        or native_kernel.get("expert_slot_fingerprint")
        != serving.get("runtime", {}).get("expert_slot_fingerprint")
        or not isinstance(dflash2_preflight, dict)
        or not isinstance(dflash2_topk_tuning, dict)
        or not isinstance(dflash2_fusion_tuning, dict)
        or set(dflash2_fusion_tuning) != {"selector", "body"}
        or not isinstance(adaptive, dict)
    ):
        raise ModelCardError("serving report has incomplete native-MTP/DFlash2 results")
    default = results.get("default_speculation")
    if default not in MODES or runtime.get("default_speculation") != default:
        raise ModelCardError("serving report has no valid measured default")
    speculation_settings = runtime.get("speculation_settings")
    dflash2_settings = (
        speculation_settings.get("dflash2")
        if isinstance(speculation_settings, dict)
        else None
    )
    if (
        not isinstance(dflash2_settings, dict)
        or dflash2_settings.get("checkpoint_model_id") != "incoai/GLM-5.3-DFlash2"
        or dflash2_settings.get("checkpoint_revision")
        != "425aa615ce320caac34400208b30808c8f14f76c"
        or dflash2_settings.get("draft_policy") != "adaptive"
        or dflash2_settings.get("fixed_drafts") is not None
        or isinstance(dflash2_settings.get("proposal_drafts"), bool)
        or not isinstance(dflash2_settings.get("proposal_drafts"), int)
        or not 1 <= dflash2_settings["proposal_drafts"] <= 7
        or dflash2_settings.get("topk_backend")
        not in {"torch", "flashinfer", "flashinfer-dsa"}
    ):
        raise ModelCardError("serving report has no exact adaptive DFlash2 contract")
    dflash2_proposal_drafts = dflash2_settings["proposal_drafts"]
    dflash2_topk_backend = dflash2_settings["topk_backend"]
    dflash2_topk_micro_backend = dflash2_topk_tuning.get("micro_selected_backend")
    dflash2_topk_service = dflash2_topk_tuning.get("full_service_gate")
    if (
        dflash2_preflight.get("checkpoint_repo_id")
        != dflash2_settings["checkpoint_model_id"]
        or dflash2_preflight.get("checkpoint_revision")
        != dflash2_settings["checkpoint_revision"]
        or dflash2_preflight.get("checkpoint_config_sha256") != DFLASH2_CONFIG_SHA256
        or dflash2_preflight.get("checkpoint_weight_lfs_sha256")
        != DFLASH2_WEIGHT_LFS_SHA256
        or dflash2_preflight.get("kv_storage") != DFLASH2_SERVING_KV_STORAGE
        or dflash2_preflight.get("kv_element_bytes") != DFLASH2_SERVING_KV_ELEMENT_BYTES
        or dflash2_preflight.get("page_size") != DFLASH2_SERVING_KV_PAGE_SIZE
        or dflash2_preflight.get("kv_capacity_tokens")
        != DFLASH2_SERVING_KV_CAPACITY_TOKENS
        or dflash2_preflight.get("proposal_tokens_per_request")
        != dflash2_proposal_drafts
        or dflash2_preflight.get("topk_backend") != dflash2_topk_backend
        or dflash2_topk_tuning.get("selected_backend") != dflash2_topk_backend
        or dflash2_topk_micro_backend not in {"torch", "flashinfer", "flashinfer-dsa"}
        or dflash2_topk_tuning.get("fastest_valid_backend")
        not in {"torch", "flashinfer", "flashinfer-dsa"}
        or dflash2_topk_backend not in dflash2_topk_tuning.get("valid_backends", [])
        or not isinstance(dflash2_topk_service, dict)
        or dflash2_topk_service.get("selection_policy")
        != DFLASH2_TOPK_SERVICE_SELECTION_POLICY
        or dflash2_topk_service.get("minimum_non_torch_speedup")
        != DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP
        or dflash2_topk_service.get("selected_backend") != dflash2_topk_backend
        or dflash2_topk_service.get("candidate_backend") != dflash2_topk_micro_backend
        or not isinstance(dflash2_topk_service.get("candidate_quality_passed"), bool)
        or not isinstance(dflash2_topk_service.get("candidate_quality_failures"), list)
        or not isinstance(dflash2_topk_service.get("candidate_speedup_vs_torch"), dict)
        or not isinstance(dflash2_topk_service.get("weighted_decode_tps"), dict)
        or not isinstance(dflash2_topk_service.get("median_repeat_decode_tps"), dict)
        or integer(dflash2_topk_service.get("requests"), "top-k service requests")
        != len(REQUIRED_SEMANTIC_CASE_IDS) * REQUIRED_SEMANTIC_REPEATS
        or not 0
        <= integer(
            dflash2_topk_service.get("response_hash_mismatches"),
            "top-k response hash mismatches",
        )
        <= dflash2_topk_service["requests"]
    ):
        raise ModelCardError(
            "serving report does not prove the exact production DFlash2 KV geometry"
        )
    topk_service_speedups = dflash2_topk_service["candidate_speedup_vs_torch"]
    topk_candidate_eligible = (
        dflash2_topk_service["candidate_quality_passed"]
        and min(
            number(
                topk_service_speedups.get("weighted_decode"),
                "top-k weighted service speedup",
            ),
            number(
                topk_service_speedups.get("median_repeat_decode"),
                "top-k median service speedup",
            ),
        )
        >= DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP
    )
    expected_topk_backend = (
        dflash2_topk_micro_backend if topk_candidate_eligible else "torch"
    )
    if dflash2_topk_backend != expected_topk_backend:
        raise ModelCardError("serving report has an invalid top-k service decision")
    cost_profile = adaptive.get("cost_profile")
    if (
        not isinstance(cost_profile, dict)
        or adaptive.get("reference_width") != 5
        or integer(cost_profile.get("route_qualified_cells"), "route-qualified cells")
        < 1
        or integer(cost_profile.get("corpus_samples"), "cost-profile corpus samples")
        < 1
    ):
        raise ModelCardError("serving report has no qualified adaptive cost profile")
    adaptive_response_score = number(
        adaptive.get("response_performance_score"), "adaptive response score"
    )
    k5_response_score = number(
        adaptive.get("k5_response_performance_score"), "K5 response score"
    )
    adaptive_concurrency_score = number(
        adaptive.get("concurrency_geomean_tps"), "adaptive concurrency score"
    )
    k5_concurrency_score = number(
        adaptive.get("k5_concurrency_geomean_tps"), "K5 concurrency score"
    )
    adaptive_weighted_ratio = number(
        adaptive.get("weighted_decode_ratio_vs_k5"), "adaptive/K5 weighted ratio"
    )
    if adaptive_response_score < k5_response_score or (
        adaptive_concurrency_score < k5_concurrency_score
    ):
        raise ModelCardError("adaptive DFlash2 does not beat its fixed-K5 reference")

    mode_lines = [
        "| Mode | Code decode | Weighted decode | Orchid repeat | Draft acceptance | Expert handoff |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    mode_curves: dict[str, dict[int, dict[str, float | int]]] = {}
    mode_candidates: dict[str, tuple[float, float, float, bool]] = {}
    for mode in MODES:
        result = mode_results[mode]
        if not isinstance(result, dict):
            raise ModelCardError(f"{mode} result is malformed")
        mode_curves[mode] = verify_cycle_curve(
            result.get("verify_cycle_by_physical_m"), mode
        )
        marker = " (default)" if mode == default else ""
        code_decode_tps = number(
            result.get("agentic_code_decode_tps"), f"{mode} code decode"
        )
        weighted_decode_tps = number(
            result.get("weighted_decode_tps"), f"{mode} weighted decode"
        )
        mode_candidates[mode] = (
            math.sqrt(code_decode_tps * weighted_decode_tps),
            weighted_decode_tps,
            code_decode_tps,
            mode == "dflash2",
        )
        mode_lines.append(
            "| {mode}{marker} | {code:.3f} tok/s | {decode:.3f} tok/s | "
            "{repeat:.3f} tok/s | {acceptance:.2%} | {startup:,.1f} ms |".format(
                mode=mode,
                marker=marker,
                code=code_decode_tps,
                decode=weighted_decode_tps,
                repeat=number(result.get("repeat_decode_tps"), f"{mode} repeat"),
                acceptance=number(
                    result.get("accepted_draft_rate"), f"{mode} acceptance"
                ),
                startup=number(
                    result.get("maximum_service_handoff_total_ms"), f"{mode} startup"
                ),
            )
        )
    measured_default = max(mode_candidates, key=mode_candidates.__getitem__)
    if measured_default != default:
        raise ModelCardError(
            "measured default differs from its agentic selection policy"
        )
    adaptive_weighted_tps = number(
        mode_results["dflash2"].get("weighted_decode_tps"),
        "adaptive weighted decode",
    )
    k5_weighted_tps = adaptive_weighted_tps / adaptive_weighted_ratio
    adaptive_lines = [
        "| DFlash2 policy | Response score | C1/C2/C4 geomean | Weighted decode |",
        "|---|---:|---:|---:|",
        (
            f"| Adaptive K1-K{dflash2_proposal_drafts} (selected) | "
            f"{adaptive_response_score:.3f} | {adaptive_concurrency_score:.3f} tok/s | "
            f"{adaptive_weighted_tps:.3f} tok/s |"
        ),
        (
            f"| Fixed K5 reference | {k5_response_score:.3f} | "
            f"{k5_concurrency_score:.3f} tok/s | {k5_weighted_tps:.3f} tok/s |"
        ),
    ]
    if dflash2_proposal_drafts + 1 not in mode_curves["dflash2"] or any(
        physical_m > dflash2_proposal_drafts + 1
        for physical_m in mode_curves["dflash2"]
    ):
        raise ModelCardError(
            "final DFlash2 result did not measure its configured full physical M"
        )
    verify_cycle_lines = [
        "| Physical M | Native MTP median | Native samples | DFlash2 median | DFlash2 samples | DFlash2/native |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for physical_m in sorted(set(mode_curves["mtp"]) | set(mode_curves["dflash2"])):
        native_row = mode_curves["mtp"].get(physical_m)
        dflash2_row = mode_curves["dflash2"].get(physical_m)
        native_median = (
            f"{native_row['median_ms']:.3f} ms" if native_row is not None else "—"
        )
        native_samples = f"{native_row['samples']:,}" if native_row is not None else "—"
        dflash2_median = (
            f"{dflash2_row['median_ms']:.3f} ms" if dflash2_row is not None else "—"
        )
        dflash2_samples = (
            f"{dflash2_row['samples']:,}" if dflash2_row is not None else "—"
        )
        cycle_ratio = (
            f"{dflash2_row['median_ms'] / native_row['median_ms']:.3f}x"
            if native_row is not None and dflash2_row is not None
            else "—"
        )
        verify_cycle_lines.append(
            f"| {physical_m} | {native_median} | {native_samples} | "
            f"{dflash2_median} | {dflash2_samples} | {cycle_ratio} |"
        )
    semantic_cells = semantic_decode.get("cells")
    if (
        semantic_decode.get("case_ids") != list(REQUIRED_SEMANTIC_CASE_IDS)
        or semantic_decode.get("repeats") != REQUIRED_SEMANTIC_REPEATS
        or not isinstance(semantic_cells, list)
        or len(semantic_cells) != len(REQUIRED_SEMANTIC_CASE_IDS)
    ):
        raise ModelCardError("serving report has no complete eight-type decode table")
    semantic_lines = [
        "| Content type | Native MTP tok/s | DFlash2 tok/s | Ratio | Native acceptance | DFlash2 acceptance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for expected_case, cell in zip(
        REQUIRED_SEMANTIC_CASE_IDS, semantic_cells, strict=True
    ):
        if not isinstance(cell, dict) or cell.get("case") != expected_case:
            raise ModelCardError("serving report has a malformed eight-type decode row")
        semantic_lines.append(
            "| {category} | {native:.3f} | {dflash:.3f} | {ratio:.3f}x | "
            "{native_accept:.2%} | {dflash_accept:.2%} |".format(
                category=cell.get("category"),
                native=number(
                    cell.get("native_mtp_decode_tps"),
                    f"native {expected_case} decode",
                ),
                dflash=number(
                    cell.get("dflash2_decode_tps"),
                    f"DFlash2 {expected_case} decode",
                ),
                ratio=number(
                    cell.get("dflash2_to_native_decode_ratio"),
                    f"{expected_case} decode ratio",
                ),
                native_accept=number(
                    cell.get("native_mtp_accepted_draft_rate"),
                    f"native {expected_case} acceptance",
                ),
                dflash_accept=number(
                    cell.get("dflash2_accepted_draft_rate"),
                    f"DFlash2 {expected_case} acceptance",
                ),
            )
        )
    concurrency_lines = [
        "| Concurrency | DFlash2 aggregate tok/s |",
        "|---:|---:|",
    ]
    dflash_concurrency = mode_results["dflash2"].get("decode_concurrency")
    if not isinstance(dflash_concurrency, dict):
        raise ModelCardError("serving report has no DFlash2 concurrency results")
    for concurrency in REQUIRED_CONCURRENCIES:
        dflash_cell = dflash_concurrency.get(str(concurrency))
        if not isinstance(dflash_cell, dict):
            raise ModelCardError(f"serving report has no C{concurrency} result")
        concurrency_lines.append(
            "| {concurrency} | {dflash:.3f} |".format(
                concurrency=concurrency,
                dflash=number(
                    dflash_cell.get("mean_aggregate_decode_tps"),
                    f"DFlash2 C{concurrency} decode",
                ),
            )
        )

    needle = mode_results["dflash2"].get("long_context_needle")
    measurements = needle.get("measurements") if isinstance(needle, dict) else None
    if not isinstance(measurements, list):
        raise ModelCardError("DFlash2 serving report has no needle measurements")
    needle_by_context: dict[int, list[dict[str, Any]]] = {}
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise ModelCardError("DFlash2 serving report has malformed needle results")
        context = integer(measurement.get("context_tokens"), "needle context")
        needle_by_context.setdefault(context, []).append(measurement)
    if set(needle_by_context) != set(REQUIRED_NEEDLE_CONTEXTS) or any(
        {measurement.get("depth") for measurement in rows}
        != set(REQUIRED_NEEDLE_DEPTHS)
        for rows in needle_by_context.values()
    ):
        raise ModelCardError("DFlash2 serving report has an incomplete needle grid")
    needle_lines = [
        "| Context | Needle depths | DFlash2 max wall | Recall |",
        "|---:|---:|---:|---:|",
    ]
    for context in REQUIRED_NEEDLE_CONTEXTS:
        dflash_wall = max(
            number(measurement.get("wall_seconds"), "DFlash2 needle wall")
            for measurement in needle_by_context[context]
        )
        needle_lines.append(
            f"| {context:,} | 10%, 50%, 90% | {dflash_wall:.1f} s | 3/3 exact |"
        )
    cells = prefill.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ModelCardError("serving report has no prefill cells")
    prefill_lines = [
        "| Context | Prefill rows | DFlash2 tok/s |",
        "|---:|---:|---:|",
    ]
    for cell in cells:
        if not isinstance(cell, dict):
            raise ModelCardError("serving report contains a malformed prefill cell")
        prefill_lines.append(
            "| {context:,} | {rows:,} | {dflash:,.1f} |".format(
                context=integer(cell.get("base_context_tokens"), "prefill context"),
                rows=integer(cell.get("suffix_tokens"), "prefill rows"),
                dflash=number(cell.get("dflash2_tps"), "DFlash2 prefill TPS"),
            )
        )
    hub_line = (
        f"- Verified Hub revision: `{hub_revision}`"
        if hub_revision is not None
        else "- Hub revision: assigned and verified immediately after initial upload"
    )
    return "\n".join(
        [
            "The exact artifact passed GLMRT's content-bound GLM-5.3 K4 qualification gate.",
            f"The measured default for agentic coding is `{default}`. The `{DEFAULT_SELECTION_POLICY}` policy compares the geometric mean of code and eight-type weighted decode, then weighted decode, code decode, and finally prefers DFlash2 on an exact tie.",
            "",
            "### Structural and quantizer evidence",
            "",
            f"- Routed projections quantized: {integer(artifact.get('quantized_modules'), 'quantized modules'):,}",
            f"- EXL3 tensors: {integer(artifact.get('exl3_tensors'), 'EXL3 tensors'):,}",
            f"- EXL3 tensor payload: {gib(integer(artifact.get('exl3_tensor_bytes'), 'EXL3 bytes'))}",
            f"- TP4 resident payload per Spark: {gib(integer(artifact.get('tp4_resident_bytes_per_spark'), 'resident bytes'))}",
            "- Aggregate Hessian-weighted relative projection error: "
            f"{number(global_metrics.get('aggregate_hessian_weighted_relative_error'), 'aggregate quantization error'):.8g}",
            "- Complete standalone `quantize_config.json.tensor_storage`: validated",
            "- Four-field embedded EXL3 discovery declaration: exact agreement validated; calibration and storage metadata remain standalone",
            "- Source `config.json`, tokenizer metadata, and generation metadata: exact values validated",
            "",
            "### Native MTP and DFlash2",
            "",
            f"DFlash2 was qualified with adaptive K1-K{dflash2_proposal_drafts} proposals using `incoai/GLM-5.3-DFlash2@{dflash2_settings['checkpoint_revision']}` (config SHA-256 `{DFLASH2_CONFIG_SHA256}`, weight LFS SHA-256 `{DFLASH2_WEIGHT_LFS_SHA256}`), the `{dflash2_topk_backend}` candidate selector, BF16 draft KV, {DFLASH2_SERVING_KV_PAGE_SIZE}-token pages, and a {DFLASH2_SERVING_KV_CAPACITY_TOKENS:,}-token cache envelope.",
            "The matched C1/C2/C4, K1-K7 top-k sweep selected "
            f"`{dflash2_topk_micro_backend}` as the service candidate; its fastest valid backend was "
            f"`{dflash2_topk_tuning['fastest_valid_backend']}` at "
            f"{number(dflash2_topk_tuning.get('fastest_valid_speedup_vs_torch'), 'DFlash2 top-k speedup'):.3f}x aggregate versus Torch. The matched fixed-K5 service gate selected "
            f"`{dflash2_topk_backend}`: Torch measured "
            f"{number(dflash2_topk_service['weighted_decode_tps'].get('torch'), 'Torch top-k service TPS'):.3f} tok/s versus "
            f"{number(dflash2_topk_service['weighted_decode_tps'].get(dflash2_topk_micro_backend), 'candidate top-k service TPS'):.3f} tok/s for `{dflash2_topk_micro_backend}`, with "
            f"{integer(dflash2_topk_service.get('response_hash_mismatches'), 'top-k response mismatches')}/{integer(dflash2_topk_service.get('requests'), 'top-k service requests')} response hashes changed. BF16 cutoff ties can admit different token IDs, so the full-service performance and quality gate is authoritative.",
            "The fused candidate selector and fused dynamic-convolution/residual/RMS epilogue each passed exact C1/C2/C4, K1-K7 gates with the measured serving warp choices.",
            "",
            "### DFlash2 adaptive policy",
            "",
            "The embedded route-cost surface was calibrated from {samples:,} corpus samples and adopted {cells:,} route-qualified cells. Selection is performance-only; tool-call quality is qualified independently.".format(
                samples=integer(
                    cost_profile.get("corpus_samples"), "cost-profile corpus samples"
                ),
                cells=integer(
                    cost_profile.get("route_qualified_cells"), "route-qualified cells"
                ),
            ),
            "",
            *adaptive_lines,
            "",
            *mode_lines,
            "",
            "### Verify-cycle cost by physical M",
            "",
            *verify_cycle_lines,
            "",
            "These are C1 post-TTFT target-cycle timings from the matched eight-type decode replay. Their per-request sums are required to equal the same `decode_ms` denominator used for TPS. Physical M is the number of target rows evaluated together: one committed-token row plus the proposed draft rows.",
            "",
            "DFlash2/native ratios: weighted decode {decode:.3f}x, repeat {repeat:.3f}x, acceptance {acceptance:.3f}x.".format(
                decode=number(
                    comparisons.get("dflash2_to_native_weighted_decode_ratio"),
                    "weighted decode ratio",
                ),
                repeat=number(
                    comparisons.get("dflash2_to_native_repeat_ratio"),
                    "repeat ratio",
                ),
                acceptance=number(
                    comparisons.get("dflash2_to_native_acceptance_ratio"),
                    "acceptance ratio",
                ),
            ),
            "",
            "### Seven-type decode mix",
            "",
            *semantic_lines,
            "",
            "### Decode concurrency",
            "",
            *concurrency_lines,
            "",
            "### Prefill",
            "",
            *prefill_lines,
            "",
            "### Long-context needle recall",
            "",
            *needle_lines,
            "",
            "Every needle request completed within the 600-second ceiling with exact recall, complete attention, numeric progression, and zero runtime graph captures.",
            "",
            "### Reproducibility identity",
            "",
            f"- GLMRT WIP engine: `{runtime.get('engine_identity')}`",
            f"- SparkInfer revision: `{runtime.get('sparkinfer_revision')}`",
            f"- Coordinator slot SHA-256: `{runtime.get('coordinator_slot_fingerprint')}`",
            f"- Spark slot SHA-256: `{runtime.get('expert_slot_fingerprint')}`",
            "- Native EXL3 parity: TP ranks {ranks}; calibrated layer {layer}; {rows} row shapes".format(
                ranks=", ".join(
                    str(value) for value in native_kernel.get("tp_ranks", [])
                ),
                layer=integer(native_kernel.get("layer_id"), "native layer"),
                rows=len(native_kernel.get("required_rows", [])),
            ),
            f"- Qualification evidence SHA-256: `{serving.get('report_sha256')}`",
            hub_line,
        ]
    )


def render(
    *,
    template_path: Path,
    artifact_validation_path: Path,
    quant_evidence_path: Path,
    serving_qualification_path: Path,
    hub_revision: str | None,
) -> str:
    template_file = template_path.expanduser()
    if template_file.is_symlink():
        raise ModelCardError("model-card template is a symbolic link")
    template_file = template_file.resolve(strict=True)
    template = template_file.read_text(encoding="utf-8")
    if template.count(MARKER) != 1:
        raise ModelCardError(
            "model-card template must contain exactly one pending marker"
        )
    artifact_path, artifact = signed_report(
        artifact_validation_path, GLM53_VALIDATION_SCHEMA
    )
    quant_path, quant = signed_report(quant_evidence_path, GLM53_QUANT_SCHEMA)
    _serving_path, serving = signed_report(serving_qualification_path, SERVING_SCHEMA)
    runtime = serving.get("runtime")
    gates = serving.get("gates")
    if (
        artifact.get("model_id") != GLM53_MODEL_ID
        or serving.get("model_id") != GLM53_MODEL_ID
        or serving.get("artifact_validation", {}).get("sha256")
        != hash_file(artifact_path)
        or serving.get("quant_evidence", {}).get("sha256") != hash_file(quant_path)
        or serving.get("artifact_manifest_sha256")
        != artifact.get("artifact_manifest_sha256")
        or serving.get("plan_sha256") != artifact.get("plan_sha256")
        or quant.get("plan", {}).get("plan_sha256") != artifact.get("plan_sha256")
        or not isinstance(runtime, dict)
        or runtime.get("profile") != "balanced"
        or runtime.get("speculation") not in MODES
        or runtime.get("default_speculation") != runtime.get("speculation")
        or runtime.get("qualified_speculation") != list(MODES)
        or REVISION_RE.fullmatch(str(runtime.get("sparkinfer_revision", ""))) is None
        or SHA256_RE.fullmatch(str(runtime.get("coordinator_slot_fingerprint", "")))
        is None
        or SHA256_RE.fullmatch(str(runtime.get("expert_slot_fingerprint", ""))) is None
        or not isinstance(gates, dict)
        or set(gates) != REQUIRED_GATES
        or any(value is not True for value in gates.values())
        or serving.get("failed_gates") != []
    ):
        raise ModelCardError(
            "serving report is not bound to the K4 structural evidence"
        )
    try:
        revalidate_native_evidence(
            serving,
            expected_sparkinfer_revision=runtime["sparkinfer_revision"],
            expected_checkpoint_root=Path(
                serving["results"]["native_kernel"]["weight_source_root"]
            )
            .expanduser()
            .resolve(),
            expected_expert_slot_fingerprint=runtime["expert_slot_fingerprint"],
        )
        revalidate_dflash2_fusion_evidence(serving)
        revalidate_dflash2_topk_evidence(serving)
        revalidate_dflash2_adaptive_evidence(serving)
    except (QualificationError, ArtifactValidationError, KeyError, TypeError) as error:
        raise ModelCardError(
            "serving report has unverifiable K4 native or adaptive DFlash2 evidence"
        ) from error
    if hub_revision is not None and REVISION_RE.fullmatch(hub_revision) is None:
        raise ModelCardError("Hub revision must be lowercase 40-64 digit hex")
    return template.replace(
        MARKER,
        render_section(
            artifact=artifact,
            quant=quant,
            serving=serving,
            hub_revision=hub_revision,
        ),
    )


def atomic_text(path: Path, value: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--artifact-validation", type=Path, required=True)
    parser.add_argument("--quant-evidence", type=Path, required=True)
    parser.add_argument("--serving-qualification", type=Path, required=True)
    parser.add_argument("--hub-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render(
        template_path=args.template,
        artifact_validation_path=args.artifact_validation,
        quant_evidence_path=args.quant_evidence,
        serving_qualification_path=args.serving_qualification,
        hub_revision=args.hub_revision,
    )
    atomic_text(args.output, rendered)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
