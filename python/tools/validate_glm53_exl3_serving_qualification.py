#!/usr/bin/env python3
"""Qualify native-MTP and DFlash2 serving for one exact GLM-5.3 EXL3 K4 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any

from _b12x_exl3_k4_profile import (
    EXL3_K4_REQUIRED_LIVE_ROWS,
    exl3_k4_capacity_rows,
    exl3_k4_route_block_rows,
)
from bench_real_full_mtp_acceptance import (
    CASES as SEMANTIC_DECODE_CASES,
    QUALITY_CONTRACT_VERSION as SEMANTIC_QUALITY_CONTRACT_VERSION,
    REQUEST_BINDING_VERSION as SEMANTIC_REQUEST_BINDING_VERSION,
    WEIGHTED_CASE_IDS as REQUIRED_SEMANTIC_CASE_IDS,
    completion_payload as semantic_completion_payload,
    structured_edit_response_format,
)
from stage_glm52_exl3_hf_snapshot import _quant_evidence, _validation_evidence
from validate_glm52_exl3_artifact import (
    GLM53_ARTIFACT_SCHEMA,
    GLM53_MODEL_ID,
    ArtifactValidationError,
    _artifact_contract,
    _json_object,
)
from validate_glm52_exl3_serving_qualification import (
    NATIVE_VALIDATION_SCHEMA,
    TOOL_EVAL_VERSION,
    QualificationError,
    SHA256_RE,
    blended,
    deployment,
    evidence_identity,
    finite_nonnegative,
    finite_positive,
    integer,
    native_validations,
    paired_equal,
    prefill,
    read_jsonl,
    repeat_decode,
    require_close,
    startup,
    tool_eval,
)

SCHEMA = "glmrt-glm5-exl3-serving-qualification-v1"
MODE_NATIVE_MTP = "mtp"
MODE_DFLASH2 = "dflash2"
MODES = (MODE_NATIVE_MTP, MODE_DFLASH2)
REQUIRED_GATES = frozenset(
    {
        "blended_decode",
        "blended_acceptance",
        "decode_concurrency",
        "repeat_decode",
        "prefill_every_cell",
        "tool_eval_points",
        "expert_resident_preload",
        "expert_startup",
        "long_context_needle",
        "native_kernel_parity",
        "dflash2_preflight",
        "dflash2_topk_tuning",
        "dflash2_topk_service_gate",
        "dflash2_selector_fusion",
        "dflash2_body_fusion",
        "dflash2_adaptive_cost_profile",
        "dflash2_adaptive_beats_k5",
        "verify_cycle_measurements",
    }
)
DFLASH2_REQUIRED_WIDTHS = tuple(range(1, 8))
DFLASH2_REFERENCE_WIDTH = 5
DFLASH2_COST_PROFILE_SCHEMA = "glmrt-dspark-cost-profile-v1"
GLM53_STARTUP_SCHEMA = "glmrt-glm53-expert-startup-v1"
CONCURRENCY_SCHEMA = "glmrt-decode-concurrency-summary-v1"
REQUIRED_CONCURRENCIES = (1, 2, 4)
CONCURRENCY_FIXTURE = "code"
NEEDLE_META_SCHEMA = "glmrt-long-context-needle-meta-v1"
NEEDLE_MEASUREMENT_SCHEMA = "glmrt-long-context-needle-measurement-v1"
NEEDLE_SUMMARY_SCHEMA = "glmrt-long-context-needle-summary-v1"
REQUIRED_NEEDLE_CONTEXTS = (8_192, 32_768, 131_072, 262_144, 393_216)
REQUIRED_NEEDLE_DEPTHS = (0.1, 0.5, 0.9)
NEEDLE_MAX_REQUEST_SECONDS = 600.0
REQUIRED_PREFILL_BASE_CONTEXTS = (0, 32_768, 65_536, 131_072, 262_144)
REQUIRED_PREFILL_SUFFIX_ROWS = (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)
REQUIRED_PREFILL_REPEATS = (1, 2)
K4_REQUIRED_NATIVE_ROWS = frozenset(EXL3_K4_REQUIRED_LIVE_ROWS)
DFLASH2_PREFLIGHT_SCHEMA = "glmrt-dflash2-preflight-v1"
DFLASH2_TOPK_TUNING_SCHEMA = "glmrt-dflash2-topk-tuning-v1"
DFLASH2_REPO_ID = "incoai/GLM-5.3-DFlash2"
DFLASH2_REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
DFLASH2_CONFIG_SHA256 = (
    "f59e1da17d41d24a1aba588aecee1607788adb34a03805f2c883add8ca954e9b"
)
DFLASH2_WEIGHT_LFS_SHA256 = (
    "3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161"
)
DFLASH2_TENSOR_COUNT = 96
DFLASH2_PAYLOAD_BYTES = 4_918_848_512
DFLASH2_SERVING_KV_STORAGE = "bf16"
DFLASH2_SERVING_KV_ELEMENT_BYTES = 2
DFLASH2_SERVING_KV_PAGE_SIZE = 64
DFLASH2_SERVING_KV_CAPACITY_TOKENS = 2_176
DFLASH2_SERVING_MAX_PAGES_PER_REQUEST = 34
DFLASH2_SERVING_PHYSICAL_PAGES = 136
DFLASH2_TOPK_BACKENDS = ("torch", "flashinfer", "flashinfer-dsa")
DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP = 1.01
DFLASH2_TOPK_SELECTION_POLICY = (
    "lowest_valid_case_aggregate_median_with_1pct_non_torch_gate"
)
DFLASH2_TOPK_SERVICE_SELECTION_POLICY = (
    "quality_then_min_wall_and_median_repeat_1pct_gate_else_torch"
)
DEFAULT_SELECTION_POLICY = (
    "code_and_eight_type_decode_geomean_then_weighted_then_code_then_"
    "dflash2_literal_tie"
)
DFLASH2_WIDTH_SELECTION_POLICY = (
    "code_and_eight_type_decode_geomean_then_weighted_then_code_then_" "narrower_width"
)
TOOLS_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = TOOLS_ROOT.parent / "reference" / "glmrt_reference"
DFLASH2_SELECTOR_TUNING_SCHEMA = "glmrt-dflash2-selector-tuning-v1"
DFLASH2_BODY_FUSION_TUNING_SCHEMA = "glmrt-dflash2-body-fusion-tuning-v1"
DFLASH2_FUSION_MIN_SPEEDUP = 1.01
sys.path.insert(0, os.fspath(REFERENCE_ROOT))
from dflash_tuning_profile import (  # noqa: E402
    dflash2_body_num_warps,
    dflash2_selector_num_warps,
)

REQUIRED_SEMANTIC_REPEATS = 5
VERIFY_CYCLE_FIELDS = frozenset(
    {"samples", "total_ms", "mean_ms", "median_ms", "min_ms", "max_ms"}
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def source_sha256(path: Path) -> str:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise QualificationError(f"source binding is a symbolic link: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise QualificationError(f"source binding is not one regular file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ratio(numerator: float, denominator: float, label: str) -> float:
    if (
        not math.isfinite(numerator)
        or numerator < 0.0
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        raise QualificationError(f"{label} cannot form a finite performance ratio")
    return numerator / denominator


def verify_cycle_curve(
    path: Path, *, expected_fixed_drafts: int | None = None
) -> dict[str, dict[str, float | int]]:
    """Recompute and validate C1 post-TTFT target-cycle cost by physical M."""

    resolved, records = read_jsonl(path)
    aggregates = [
        record.get("aggregate") for record in records if "aggregate" in record
    ]
    if len(aggregates) != 1 or not isinstance(aggregates[0], dict):
        raise QualificationError(f"{resolved} must contain one blended aggregate")
    aggregate = aggregates[0]
    measurements = [record for record in records if "aggregate" not in record]
    grouped: dict[int, list[float]] = {}
    total_target_cycles = 0
    full_width_verify_seen = False
    for record_index, record in enumerate(measurements):
        drafts = record.get("draft_lengths")
        accepted = record.get("accepted_draft_lengths")
        cycle_ms = record.get("verify_cycle_ms")
        target_physical_m = record.get("target_cycle_physical_m")
        target_cycle_ms = record.get("target_cycle_ms")
        cycles = integer(
            record.get("verify_cycles"),
            f"{resolved}: blended record {record_index} verify cycles",
            minimum=0,
        )
        if expected_fixed_drafts is not None and cycles == 0:
            raise QualificationError(
                f"{resolved} fixed-width DFlash2 record has no verify cycles"
            )
        if (
            not isinstance(drafts, list)
            or not isinstance(accepted, list)
            or not isinstance(cycle_ms, list)
            or len(drafts) != cycles
            or len(accepted) != cycles
            or len(cycle_ms) != cycles
        ):
            raise QualificationError(
                f"{resolved} has unaligned per-cycle verify measurements"
            )
        for cycle_index, (draft, accepted_count, elapsed_ms) in enumerate(
            zip(drafts, accepted, cycle_ms, strict=True)
        ):
            draft_count = integer(
                draft,
                f"{resolved}: record {record_index} cycle {cycle_index} draft length",
                minimum=1,
            )
            if draft_count > max(DFLASH2_REQUIRED_WIDTHS):
                raise QualificationError(f"{resolved} selected physical M above 8")
            accepted_value = integer(
                accepted_count,
                f"{resolved}: record {record_index} cycle {cycle_index} accepted drafts",
            )
            if accepted_value > draft_count:
                raise QualificationError(
                    f"{resolved} accepted more than one cycle's drafts"
                )
            if (
                expected_fixed_drafts is not None
                and draft_count > expected_fixed_drafts
            ):
                raise QualificationError(
                    f"{resolved} exceeds fixed draft width {expected_fixed_drafts}"
                )
            full_width_verify_seen |= draft_count == expected_fixed_drafts
            finite_positive(
                elapsed_ms,
                f"{resolved}: record {record_index} cycle {cycle_index} milliseconds",
            )
        if (
            not isinstance(target_physical_m, list)
            or not isinstance(target_cycle_ms, list)
            or len(target_physical_m) != len(target_cycle_ms)
        ):
            raise QualificationError(
                f"{resolved} has unaligned post-TTFT target-cycle measurements"
            )
        total_target_cycles += len(target_cycle_ms)
        validated_target_cycle_ms: list[float] = []
        for cycle_index, (physical_m_value, elapsed_ms) in enumerate(
            zip(target_physical_m, target_cycle_ms, strict=True)
        ):
            physical_m = integer(
                physical_m_value,
                f"{resolved}: record {record_index} target cycle {cycle_index} physical M",
                minimum=1,
            )
            if physical_m > max(DFLASH2_REQUIRED_WIDTHS) + 1:
                raise QualificationError(f"{resolved} selected physical M above 8")
            if (
                expected_fixed_drafts is not None
                and physical_m > expected_fixed_drafts + 1
            ):
                raise QualificationError(
                    f"{resolved} exceeds fixed physical M {expected_fixed_drafts + 1}"
                )
            validated_ms = finite_positive(
                elapsed_ms,
                f"{resolved}: record {record_index} target cycle {cycle_index} milliseconds",
            )
            validated_target_cycle_ms.append(validated_ms)
            grouped.setdefault(physical_m, []).append(validated_ms)
        record_decode_ms = finite_nonnegative(
            record.get("decode_ms"),
            f"{resolved}: record {record_index} decode milliseconds",
        )
        if not math.isclose(
            record_decode_ms,
            sum(validated_target_cycle_ms),
            rel_tol=1.0e-9,
            abs_tol=1.0e-6,
        ):
            raise QualificationError(
                f"{resolved}: record {record_index} target-cycle/decode timing differs"
            )
    if not grouped or total_target_cycles == 0:
        raise QualificationError(
            f"{resolved} has no post-TTFT target-cycle measurements"
        )
    if expected_fixed_drafts is not None and (
        not full_width_verify_seen or expected_fixed_drafts + 1 not in grouped
    ):
        raise QualificationError(
            f"{resolved} never measures configured full physical M "
            f"{expected_fixed_drafts + 1}"
        )

    expected: dict[str, dict[str, float | int]] = {
        str(physical_m): {
            "samples": len(values),
            "total_ms": sum(values),
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "min_ms": min(values),
            "max_ms": max(values),
        }
        for physical_m, values in sorted(grouped.items())
    }
    raw_curve = aggregate.get("target_cycle_ms_by_physical_m")
    if not isinstance(raw_curve, dict) or set(raw_curve) != set(expected):
        raise QualificationError(
            f"{resolved} has an incomplete physical-M timing curve"
        )
    raw_histogram = aggregate.get("target_cycle_physical_m_histogram")
    if not isinstance(raw_histogram, dict):
        raise QualificationError(f"{resolved} has no physical-M histogram")
    for physical_m, expected_row in expected.items():
        raw_row = raw_curve.get(physical_m)
        if not isinstance(raw_row, dict) or set(raw_row) != VERIFY_CYCLE_FIELDS:
            raise QualificationError(
                f"{resolved} has a malformed physical-M timing row"
            )
        if (
            integer(
                raw_histogram.get(physical_m),
                f"{resolved}: physical M {physical_m} histogram",
                minimum=1,
            )
            != expected_row["samples"]
        ):
            raise QualificationError(
                f"{resolved} physical-M histogram and cycle timings differ"
            )
        if (
            integer(
                raw_row.get("samples"),
                f"{resolved}: physical M {physical_m} timing samples",
                minimum=1,
            )
            != expected_row["samples"]
        ):
            raise QualificationError(f"{resolved} physical-M sample count differs")
        for field in VERIFY_CYCLE_FIELDS - {"samples"}:
            require_close(
                raw_row.get(field),
                expected_row[field],
                f"{resolved}: physical M {physical_m} {field}",
            )
    return expected


def require_eight_type_blended(evidence: dict[str, Any]) -> None:
    """Require the exact maintained eight-type, five-replay decode corpus."""

    nonce_seed = evidence.get("prompt_contract", {}).get("nonce_seed")
    tokenizer_sha256 = evidence.get("prompt_contract", {}).get("tokenizer_sha256")
    expected_contract = {
        "suite": "weighted",
        "cases": [
            {
                "id": case_id,
                "category": SEMANTIC_DECODE_CASES[case_id].category,
                "prompt": SEMANTIC_DECODE_CASES[case_id].prompt,
                "max_tokens": SEMANTIC_DECODE_CASES[case_id].max_tokens,
                "weight": SEMANTIC_DECODE_CASES[case_id].weight,
                "response_format": (
                    structured_edit_response_format()
                    if SEMANTIC_DECODE_CASES[case_id].json_schema
                    else None
                ),
            }
            for case_id in REQUIRED_SEMANTIC_CASE_IDS
        ],
        "repeats": REQUIRED_SEMANTIC_REPEATS,
        "nonce_seed": nonce_seed,
        "nonce_policy": "token-zero",
        "tokenizer_sha256": tokenizer_sha256,
        "temperature": 0,
        "enable_thinking": False,
        "quality_contract_version": SEMANTIC_QUALITY_CONTRACT_VERSION,
        "request_binding_version": SEMANTIC_REQUEST_BINDING_VERSION,
    }
    case_results = evidence.get("case_results")
    prompts = evidence.get("prompts")
    if (
        isinstance(nonce_seed, bool)
        or not isinstance(nonce_seed, int)
        or not isinstance(tokenizer_sha256, str)
        or not SHA256_RE.fullmatch(tokenizer_sha256)
        or evidence.get("prompt_contract") != expected_contract
        or evidence.get("cases")
        != len(REQUIRED_SEMANTIC_CASE_IDS) * REQUIRED_SEMANTIC_REPEATS
        or not isinstance(case_results, list)
        or [row.get("case") if isinstance(row, dict) else None for row in case_results]
        != list(REQUIRED_SEMANTIC_CASE_IDS)
        or any(
            row.get("samples") != REQUIRED_SEMANTIC_REPEATS
            for row in case_results
            if isinstance(row, dict)
        )
        or not isinstance(prompts, list)
        or len(prompts) != len(REQUIRED_SEMANTIC_CASE_IDS) * REQUIRED_SEMANTIC_REPEATS
    ):
        raise QualificationError(
            "blended decode must use the exact nonce-bound eight-type weighted corpus "
            "with five complete replays"
        )
    markers = set()
    first_token_ids = set()
    for request_index, prompt in enumerate(prompts):
        if not isinstance(prompt, dict):
            raise QualificationError("blended decode has a malformed request binding")
        case_index = request_index % len(REQUIRED_SEMANTIC_CASE_IDS)
        repeat_index = request_index // len(REQUIRED_SEMANTIC_CASE_IDS)
        case_id = REQUIRED_SEMANTIC_CASE_IDS[case_index]
        nonce = prompt.get("nonce")
        marker = nonce.get("marker") if isinstance(nonce, dict) else None
        first_token_id = (
            nonce.get("first_content_token_id") if isinstance(nonce, dict) else None
        )
        full_prompt = prompt.get("prompt")
        expected_prompt = (
            f"{marker} request nonce {nonce_seed}-{request_index}.\n"
            "Treat the preceding request nonce as irrelevant.\n"
            f"{SEMANTIC_DECODE_CASES[case_id].prompt}"
        )
        request_body = semantic_completion_payload(
            GLM53_MODEL_ID,
            SEMANTIC_DECODE_CASES[case_id],
            prompt_prefix=(
                f"{marker} request nonce {nonce_seed}-{request_index}.\n"
                "Treat the preceding request nonce as irrelevant.\n"
            ),
        )
        if (
            not isinstance(marker, str)
            or len(marker) != 1
            or isinstance(first_token_id, bool)
            or not isinstance(first_token_id, int)
            or first_token_id < 0
            or prompt.get("case") != case_id
            or prompt.get("repeat") != repeat_index + 1
            or full_prompt != expected_prompt
            or prompt.get("prompt_sha256")
            != hashlib.sha256(expected_prompt.encode()).hexdigest()
            or prompt.get("request_sha256") != hashlib.sha256(request_body).hexdigest()
            or marker in markers
            or first_token_id in first_token_ids
        ):
            raise QualificationError(
                "blended decode has an invalid token-zero nonce or request binding"
            )
        markers.add(marker)
        first_token_ids.add(first_token_id)


def decode_concurrency(
    paths: list[Path], *, expected_model: str = GLM53_MODEL_ID
) -> dict[str, Any]:
    """Validate a prompt-bound C1/C2/C4 decode curve for one serving mode."""

    if len(paths) != len(REQUIRED_CONCURRENCIES):
        raise QualificationError(
            "decode concurrency requires exactly C1, C2, and C4 evidence"
        )
    cells: dict[int, dict[str, Any]] = {}
    identities = []
    for path in paths:
        resolved, records = read_jsonl(path)
        aggregate_records = [
            record.get("aggregate") for record in records if "aggregate" in record
        ]
        batches = [record for record in records if "aggregate" not in record]
        if len(aggregate_records) != 1 or not isinstance(aggregate_records[0], dict):
            raise QualificationError(
                f"{resolved} must contain one concurrency aggregate"
            )
        aggregate = aggregate_records[0]
        concurrency = integer(
            aggregate.get("concurrency"), f"{resolved}: concurrency", minimum=1
        )
        warmups = integer(aggregate.get("warmups"), f"{resolved}: warmups", minimum=2)
        repeats = integer(aggregate.get("repeats"), f"{resolved}: repeats", minimum=3)
        nonce_seed = integer(aggregate.get("nonce_seed"), f"{resolved}: nonce_seed")
        tokenizer_sha256 = aggregate.get("tokenizer_sha256")
        contract = aggregate.get("request_contract")
        contract_sha256 = aggregate.get("request_contract_sha256")
        if (
            aggregate.get("schema") != CONCURRENCY_SCHEMA
            or aggregate.get("model") != expected_model
            or aggregate.get("fixture") != CONCURRENCY_FIXTURE
            or aggregate.get("cache_state") != "token-zero-nonce"
            or not isinstance(tokenizer_sha256, str)
            or not SHA256_RE.fullmatch(tokenizer_sha256)
            or not isinstance(contract, dict)
            or not isinstance(contract_sha256, str)
            or not SHA256_RE.fullmatch(contract_sha256)
            or hashlib.sha256(canonical_json(contract)).hexdigest() != contract_sha256
        ):
            raise QualificationError(f"{resolved} has an invalid concurrency contract")
        if (
            contract.get("model") != expected_model
            or contract.get("fixture") != CONCURRENCY_FIXTURE
            or contract.get("concurrency") != concurrency
            or contract.get("warmups") != warmups
            or contract.get("repeats") != repeats
            or contract.get("cache_state") != "token-zero-nonce"
            or contract.get("nonce_seed") != nonce_seed
            or contract.get("tokenizer_sha256") != tokenizer_sha256
            or not isinstance(contract.get("prompt"), str)
            or not contract["prompt"]
            or contract.get("max_tokens") != 320
            or contract.get("enable_thinking") is not False
            or len(batches) != warmups + repeats
        ):
            raise QualificationError(
                f"{resolved} concurrency schedule differs from its contract"
            )
        request_schedule = contract.get("request_sha256")
        if (
            not isinstance(request_schedule, list)
            or len(request_schedule) != len(batches)
            or any(
                not isinstance(row, list)
                or len(row) != concurrency
                or any(
                    not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)
                    for digest in row
                )
                for row in request_schedule
            )
        ):
            raise QualificationError(
                f"{resolved} has an invalid request digest schedule"
            )

        samples = []
        response_samples = []
        for batch_index, (batch, expected_requests) in enumerate(
            zip(batches, request_schedule, strict=True)
        ):
            expected_kind = "warmup" if batch_index < warmups else "repeat"
            expected_index = (
                batch_index + 1
                if expected_kind == "warmup"
                else batch_index - warmups + 1
            )
            lanes = batch.get("lanes")
            if (
                batch.get(expected_kind) != expected_index
                or ("repeat" if expected_kind == "warmup" else "warmup") in batch
                or batch.get("fixture") != CONCURRENCY_FIXTURE
                or batch.get("concurrency") != concurrency
                or batch.get("cache_state") != "token-zero-nonce"
                or batch.get("nonce_seed") != nonce_seed
                or batch.get("all_correct") is not True
                or batch.get("all_zero_runtime_captures") is not True
                or not isinstance(lanes, list)
                or len(lanes) != concurrency
            ):
                raise QualificationError(f"{resolved} has an invalid concurrency batch")
            if sorted(
                lane.get("lane") for lane in lanes if isinstance(lane, dict)
            ) != list(range(concurrency)):
                raise QualificationError(f"{resolved} has invalid concurrency lane IDs")
            lanes = sorted(lanes, key=lambda lane: lane["lane"])
            timed_tokens = 0
            first_tokens = []
            decode_ends = []
            response_ends = []
            for lane, expected_request in zip(lanes, expected_requests, strict=True):
                completion_tokens = integer(
                    lane.get("completion_tokens"),
                    f"{resolved}: completion_tokens",
                    minimum=2,
                )
                request_sha256 = lane.get("request_sha256")
                nonce = lane.get("prompt_nonce")
                if (
                    request_sha256 != expected_request
                    or lane.get("correct") is not True
                    or lane.get("runtime_captures") != 0
                    or not isinstance(nonce, dict)
                    or not isinstance(nonce.get("marker"), str)
                    or len(nonce["marker"]) != 1
                    or integer(
                        nonce.get("first_content_token_id"),
                        f"{resolved}: nonce token",
                    )
                    < 0
                ):
                    raise QualificationError(
                        f"{resolved} has unbound or invalid lane evidence"
                    )
                first_token = finite_nonnegative(
                    lane.get("first_token_ms"), f"{resolved}: first_token_ms"
                )
                decode_end = finite_positive(
                    lane.get("decode_end_ms"), f"{resolved}: decode_end_ms"
                )
                response_end = finite_positive(
                    lane.get("response_end_ms"), f"{resolved}: response_end_ms"
                )
                if decode_end <= first_token or response_end < decode_end:
                    raise QualificationError(f"{resolved} has impossible lane timing")
                timed_tokens += completion_tokens - 1
                first_tokens.append(first_token)
                decode_ends.append(decode_end)
                response_ends.append(response_end)
            decode_window_ms = max(decode_ends) - min(first_tokens)
            response_window_ms = max(response_ends) - min(first_tokens)
            require_close(
                batch.get("timed_tokens"),
                float(timed_tokens),
                f"{resolved}: timed_tokens",
            )
            require_close(
                batch.get("decode_window_ms"),
                decode_window_ms,
                f"{resolved}: decode_window_ms",
            )
            require_close(
                batch.get("aggregate_decode_tps"),
                timed_tokens * 1_000.0 / decode_window_ms,
                f"{resolved}: aggregate_decode_tps",
            )
            require_close(
                batch.get("response_window_ms"),
                response_window_ms,
                f"{resolved}: response_window_ms",
            )
            require_close(
                batch.get("aggregate_response_window_tps"),
                timed_tokens * 1_000.0 / response_window_ms,
                f"{resolved}: aggregate_response_window_tps",
            )
            if expected_kind == "repeat":
                samples.append(timed_tokens * 1_000.0 / decode_window_ms)
                response_samples.append(timed_tokens * 1_000.0 / response_window_ms)

        summary_fields = {
            "mean_aggregate_decode_tps": statistics.mean(samples),
            "median_aggregate_decode_tps": statistics.median(samples),
            "min_aggregate_decode_tps": min(samples),
            "max_aggregate_decode_tps": max(samples),
            "stdev_aggregate_decode_tps": (
                statistics.stdev(samples) if len(samples) > 1 else 0.0
            ),
            "mean_aggregate_response_window_tps": statistics.mean(response_samples),
            "median_aggregate_response_window_tps": statistics.median(response_samples),
            "min_aggregate_response_window_tps": min(response_samples),
            "max_aggregate_response_window_tps": max(response_samples),
            "stdev_aggregate_response_window_tps": (
                statistics.stdev(response_samples) if len(response_samples) > 1 else 0.0
            ),
        }
        for field, expected in summary_fields.items():
            require_close(aggregate.get(field), expected, f"{resolved}: {field}")
        if any(
            aggregate.get(field) is not True
            for field in (
                "all_correct",
                "all_zero_runtime_captures",
                "all_warmups_correct",
                "all_warmups_zero_runtime_captures",
            )
        ):
            raise QualificationError(f"{resolved} failed concurrency correctness gates")
        if concurrency in cells:
            raise QualificationError(f"duplicate C{concurrency} concurrency evidence")
        cells[concurrency] = {
            "mean_aggregate_decode_tps": summary_fields["mean_aggregate_decode_tps"],
            "median_aggregate_decode_tps": summary_fields[
                "median_aggregate_decode_tps"
            ],
            "mean_response_window_tps": summary_fields[
                "mean_aggregate_response_window_tps"
            ],
            "request_contract_sha256": contract_sha256,
            "request_contract": contract,
        }
        identities.append(evidence_identity(resolved, CONCURRENCY_SCHEMA))
    if tuple(sorted(cells)) != REQUIRED_CONCURRENCIES:
        raise QualificationError("decode concurrency must cover exactly C1, C2, and C4")
    return {"identities": identities, "fixture": CONCURRENCY_FIXTURE, "cells": cells}


def long_context_needle(
    path: Path, *, expected_model: str = GLM53_MODEL_ID
) -> dict[str, Any]:
    resolved, records = read_jsonl(path)
    meta = records[0]
    summary = records[-1]
    measurements = records[1:-1]
    contract = meta.get("request_contract")
    contract_sha256 = meta.get("request_contract_sha256")
    if (
        meta.get("schema") != NEEDLE_META_SCHEMA
        or meta.get("model") != expected_model
        or not isinstance(contract, dict)
        or not isinstance(contract_sha256, str)
        or not SHA256_RE.fullmatch(contract_sha256)
        or hashlib.sha256(canonical_json(contract)).hexdigest() != contract_sha256
        or contract.get("model") != expected_model
        or meta.get("tokenizer_sha256") != contract.get("tokenizer_sha256")
        or meta.get("filler_sha256") != contract.get("filler_sha256")
        or not isinstance(meta.get("tokenizer_sha256"), str)
        or not SHA256_RE.fullmatch(meta["tokenizer_sha256"])
        or not isinstance(meta.get("filler_sha256"), str)
        or not SHA256_RE.fullmatch(meta["filler_sha256"])
    ):
        raise QualificationError(f"{resolved} has an invalid needle contract")
    contexts = contract.get("contexts")
    depths = contract.get("depths")
    prompt_contracts = contract.get("prompts")
    if (
        contexts != list(REQUIRED_NEEDLE_CONTEXTS)
        or depths != list(REQUIRED_NEEDLE_DEPTHS)
        or contract.get("maximum_request_seconds") != NEEDLE_MAX_REQUEST_SECONDS
        or contract.get("max_context_tokens") != 400_000
        or contract.get("max_output_tokens") != 32
        or not isinstance(contract.get("session_id"), str)
        or not contract["session_id"]
        or not isinstance(prompt_contracts, list)
        or len(prompt_contracts)
        != len(REQUIRED_NEEDLE_CONTEXTS) * len(REQUIRED_NEEDLE_DEPTHS)
        or len(measurements) != len(prompt_contracts)
    ):
        raise QualificationError(f"{resolved} does not cover the required needle grid")
    expected_schedule = [
        (context, depth)
        for context in REQUIRED_NEEDLE_CONTEXTS
        for depth in REQUIRED_NEEDLE_DEPTHS
    ]
    walls = []
    results = []
    for index, (prompt, measurement, (context, depth)) in enumerate(
        zip(prompt_contracts, measurements, expected_schedule, strict=True)
    ):
        if not isinstance(prompt, dict) or not isinstance(measurement, dict):
            raise QualificationError(f"{resolved} has a malformed needle row")
        actual_context = integer(
            prompt.get("actual_context_tokens"),
            f"{resolved}: actual needle context",
            minimum=1,
        )
        needle_key = prompt.get("needle_key")
        request_sha256 = prompt.get("request_sha256")
        messages_sha256 = prompt.get("messages_sha256")
        prompt_sha256 = hashlib.sha256(canonical_json(prompt)).hexdigest()
        content = measurement.get("content")
        if (
            prompt.get("target_context_tokens") != context
            or prompt.get("needle_depth") != depth
            or prompt.get("target_tolerance_tokens") != 8
            or abs(actual_context - context) > 8
            or not isinstance(needle_key, str)
            or not needle_key.startswith("N53-")
            or not isinstance(request_sha256, str)
            or not SHA256_RE.fullmatch(request_sha256)
            or not isinstance(messages_sha256, str)
            or not SHA256_RE.fullmatch(messages_sha256)
            or integer(
                prompt.get("filler_tokens_before_needle"),
                f"{resolved}: filler before needle",
                minimum=1,
            )
            < 1
            or integer(
                prompt.get("filler_tokens_after_needle"),
                f"{resolved}: filler after needle",
                minimum=1,
            )
            < 1
            or measurement.get("schema") != NEEDLE_MEASUREMENT_SCHEMA
            or measurement.get("target_context_tokens") != context
            or measurement.get("prompt_tokens") != actual_context
            or measurement.get("needle_depth") != depth
            or measurement.get("needle_key") != needle_key
            or measurement.get("request_sha256") != request_sha256
            or measurement.get("prompt_contract_sha256") != prompt_sha256
            or measurement.get("exact_recall") is not True
            or measurement.get("within_request_time_ceiling") is not True
            or measurement.get("numeric_progression_passed") is not True
            or measurement.get("attention_complete") is not True
            or measurement.get("runtime_captures") != 0
            or not isinstance(content, str)
            or content.strip() != needle_key
            or hashlib.sha256(content.encode()).hexdigest()
            != measurement.get("content_sha256")
        ):
            raise QualificationError(f"{resolved} failed needle row {index}")
        wall = finite_positive(
            measurement.get("wall_seconds"), f"{resolved}: needle wall_seconds"
        )
        if wall > NEEDLE_MAX_REQUEST_SECONDS:
            raise QualificationError(
                f"{resolved} exceeded the ten-minute needle ceiling"
            )
        finite_positive(measurement.get("prefill_ms"), f"{resolved}: needle prefill_ms")
        finite_positive(
            measurement.get("time_to_first_token_ms"),
            f"{resolved}: needle time_to_first_token_ms",
        )
        integer(
            measurement.get("output_tokens"),
            f"{resolved}: needle output_tokens",
            minimum=1,
        )
        walls.append(wall)
        results.append(
            {
                "context_tokens": context,
                "depth": depth,
                "wall_seconds": wall,
                "prefill_ms": float(measurement["prefill_ms"]),
                "time_to_first_token_ms": float(measurement["time_to_first_token_ms"]),
            }
        )
    expected_max = max(walls)
    expected_median = statistics.median(walls)
    if (
        summary.get("schema") != NEEDLE_SUMMARY_SCHEMA
        or summary.get("status") != "accepted"
        or summary.get("model") != expected_model
        or summary.get("request_contract_sha256") != contract_sha256
        or summary.get("measurements") != len(measurements)
        or summary.get("contexts") != list(REQUIRED_NEEDLE_CONTEXTS)
        or summary.get("depths") != list(REQUIRED_NEEDLE_DEPTHS)
        or summary.get("maximum_request_seconds") != NEEDLE_MAX_REQUEST_SECONDS
        or any(
            summary.get(field) is not True
            for field in (
                "all_exact_recall",
                "all_within_request_time_ceiling",
                "all_numeric_progression_passed",
                "all_attention_complete",
                "all_zero_runtime_captures",
            )
        )
    ):
        raise QualificationError(f"{resolved} failed its needle summary gates")
    require_close(
        summary.get("maximum_measured_wall_seconds"),
        expected_max,
        f"{resolved}: maximum needle wall",
    )
    require_close(
        summary.get("median_measured_wall_seconds"),
        expected_median,
        f"{resolved}: median needle wall",
    )
    return {
        "identity": evidence_identity(resolved, NEEDLE_SUMMARY_SCHEMA),
        "contract": contract,
        "request_contract_sha256": contract_sha256,
        "maximum_wall_seconds": expected_max,
        "median_wall_seconds": expected_median,
        "measurements": results,
    }


def require_release_prefill_grid(evidence: dict[str, Any]) -> None:
    required_cells = {
        (base, suffix)
        for base in REQUIRED_PREFILL_BASE_CONTEXTS
        for suffix in REQUIRED_PREFILL_SUFFIX_ROWS
    }
    cells = evidence.get("cells")
    prompts = evidence.get("prompts")
    if (
        not isinstance(cells, dict)
        or set(cells) != required_cells
        or not isinstance(prompts, list)
    ):
        raise QualificationError(
            "prefill evidence does not cover the required 0/32K/64K/128K/256K "
            "by 1K/2K/4K/8K/16K/32K grid"
        )
    schedule: dict[tuple[int, int], list[int]] = {}
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise QualificationError("prefill evidence has a malformed repeat schedule")
        key = (prompt.get("base_context_tokens"), prompt.get("suffix_tokens"))
        schedule.setdefault(key, []).append(prompt.get("repeat"))
    if set(schedule) != required_cells or any(
        tuple(repeats) != REQUIRED_PREFILL_REPEATS for repeats in schedule.values()
    ):
        raise QualificationError(
            "prefill evidence must contain repeats 1 and 2 for every cell"
        )


def dflash2_preflight(path: Path) -> dict[str, Any]:
    resolved = path.expanduser()
    if resolved.is_symlink():
        raise QualificationError("DFlash2 preflight evidence is a symbolic link")
    resolved = resolved.resolve(strict=True)
    report = _json_object(resolved)
    resident = report.get("resident_preload")
    aliases = report.get("target_alias_preload")
    plans = report.get("concurrency_plans")
    graphs = report.get("static_graphs")
    if (
        report.get("schema") != DFLASH2_PREFLIGHT_SCHEMA
        or report.get("status") != "accepted"
        or report.get("checkpoint_repo_id") != DFLASH2_REPO_ID
        or report.get("checkpoint_revision") != DFLASH2_REVISION
        or report.get("checkpoint_config_sha256") != DFLASH2_CONFIG_SHA256
        or report.get("checkpoint_weight_lfs_sha256") != DFLASH2_WEIGHT_LFS_SHA256
        or report.get("target_repo_id") != "zai-org/GLM-5.3"
        or report.get("tensor_count") != DFLASH2_TENSOR_COUNT
        or report.get("payload_bytes") != DFLASH2_PAYLOAD_BYTES
        or report.get("kv_storage") != DFLASH2_SERVING_KV_STORAGE
        or report.get("kv_element_bytes") != DFLASH2_SERVING_KV_ELEMENT_BYTES
        or report.get("page_size") != DFLASH2_SERVING_KV_PAGE_SIZE
        or report.get("kv_capacity_tokens") != DFLASH2_SERVING_KV_CAPACITY_TOKENS
        or not isinstance(resident, dict)
        or not isinstance(aliases, dict)
        or not isinstance(plans, list)
        or not isinstance(graphs, list)
    ):
        raise QualificationError(
            "DFlash2 preflight did not validate the pinned checkpoint"
        )
    if (
        resident.get("source_tensors") != DFLASH2_TENSOR_COUNT
        or resident.get("loaded_source_tensors") != DFLASH2_TENSOR_COUNT
        or resident.get("selected_bytes") != DFLASH2_PAYLOAD_BYTES
        or resident.get("loaded_bytes") != DFLASH2_PAYLOAD_BYTES
        or aliases.get("selected_tensors") != 2
        or aliases.get("loaded_tensors") != 2
        or aliases.get("selected_bytes") != aliases.get("loaded_bytes")
    ):
        raise QualificationError(
            "DFlash2 preflight did not preload every resident tensor"
        )
    if [plan.get("active_requests") for plan in plans if isinstance(plan, dict)] != [
        1,
        2,
        4,
    ]:
        raise QualificationError(
            "DFlash2 preflight does not cover C1/C2/C4 buffer plans"
        )
    if [
        graph.get("active_requests") for graph in graphs if isinstance(graph, dict)
    ] != [1, 2, 4]:
        raise QualificationError(
            "DFlash2 preflight does not cover C1/C2/C4 static graphs"
        )
    if any(
        not isinstance(record, dict)
        or record.get("total_physical_pages") != DFLASH2_SERVING_PHYSICAL_PAGES
        or record.get("max_pages_per_request") != DFLASH2_SERVING_MAX_PAGES_PER_REQUEST
        for record in [*plans, *graphs]
    ):
        raise QualificationError(
            "DFlash2 preflight does not use the production four-slot shared KV pool"
        )
    proposal_widths = {
        graph.get("proposal_tokens_per_request")
        for graph in graphs
        if isinstance(graph, dict)
    }
    if len(proposal_widths) != 1 or not all(
        isinstance(width, int) and not isinstance(width, bool) and 1 <= width <= 7
        for width in proposal_widths
    ):
        raise QualificationError(
            "DFlash2 preflight has inconsistent internal proposal widths"
        )
    proposal_width = proposal_widths.pop()
    if (
        report.get("proposal_tokens_per_request") != proposal_width
        or report.get("query_rows_per_request") != proposal_width + 1
        or report.get("topk_backend") not in {"torch", "flashinfer", "flashinfer-dsa"}
    ):
        raise QualificationError(
            "DFlash2 preflight top-level width does not match its graphs"
        )
    if any(
        not isinstance(plan, dict)
        or plan.get("proposal_tokens_per_request") != proposal_width
        or plan.get("query_rows_per_request") != proposal_width + 1
        for plan in plans
    ):
        raise QualificationError(
            "DFlash2 preflight buffer plans do not match its internal width"
        )
    for graph in graphs:
        active_requests = (
            graph.get("active_requests") if isinstance(graph, dict) else None
        )
        expected_packed_rows = {
            1: [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024],
            2: [2, 4, 8, 16],
            4: [4, 8, 16, 32],
        }.get(active_requests)
        base_update_validation = graph.get("base_update_graph_validation", {})
        update_validation = graph.get("packed_update_graph_validation", [])
        if (
            not isinstance(graph, dict)
            or graph.get("accepted_rows_per_request") != 1
            or graph.get("eager_replay_exact") is not True
            or graph.get("dynamic_anchor_changes_output") is not True
            or graph.get("restored_replay_exact") is not True
            or graph.get("target_embedding_alias") is not True
            or graph.get("target_lm_head_alias") is not True
            or graph.get("body_output_is_head_source") is not True
            or graph.get("shared_update_body_kv") is not True
            or graph.get("candidate_topk_sorted") is not True
            or graph.get("candidate_score_accumulation") != "bf16-edge-plus-unary-bf16"
            or graph.get("hot_replay_python_calls") != 0
            or graph.get("proposal_tokens_per_request") != proposal_width
            or graph.get("query_rows_per_request") != proposal_width + 1
            or graph.get("sliding_window_tokens") != 2048
            or graph.get("packed_update_graph_rows") != expected_packed_rows
            or not isinstance(base_update_validation, dict)
            or base_update_validation.get("rows") != active_requests
            or not isinstance(update_validation, list)
            or [
                item.get("rows") for item in update_validation if isinstance(item, dict)
            ]
            != expected_packed_rows
        ):
            raise QualificationError("DFlash2 preflight static graph contract failed")
        for item in [base_update_validation, *update_validation]:
            if (
                not isinstance(item, dict)
                or item.get("eager_replay_exact") is not True
                or item.get("dynamic_positions_change_keys") is not True
                or item.get("restored_replay_exact") is not True
                or not isinstance(item.get("dynamic_key_changed_bytes"), int)
                or isinstance(item.get("dynamic_key_changed_bytes"), bool)
                or item["dynamic_key_changed_bytes"] <= 0
                or not isinstance(
                    item.get("reference_key_bf16_steps_at_max_abs"), int
                )
                or isinstance(item.get("reference_key_bf16_steps_at_max_abs"), bool)
                or item["reference_key_bf16_steps_at_max_abs"] < 0
                or item["reference_key_bf16_steps_at_max_abs"] > 4
                or not isinstance(
                    item.get("reference_value_bf16_steps_at_max_abs"), int
                )
                or isinstance(item.get("reference_value_bf16_steps_at_max_abs"), bool)
                or item["reference_value_bf16_steps_at_max_abs"] < 0
                or item["reference_value_bf16_steps_at_max_abs"] > 1
            ):
                raise QualificationError(
                    "DFlash2 packed update graph failed eager/reference validation"
                )
            for metric, maximum in (
                ("reference_fused_hidden_max_abs", 0.125),
                ("reference_fused_hidden_relative_l2", 0.01),
                ("reference_key_relative_l2", 0.01),
                ("reference_value_relative_l2", 0.01),
            ):
                value = item.get(metric)
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    or float(value) > maximum
                ):
                    raise QualificationError(
                        "DFlash2 packed update graph failed eager/reference validation"
                    )
            key_max_abs = item.get("reference_key_max_abs")
            if (
                not isinstance(key_max_abs, (int, float))
                or isinstance(key_max_abs, bool)
                or not math.isfinite(float(key_max_abs))
                or float(key_max_abs) < 0.0
            ):
                raise QualificationError(
                    "DFlash2 packed update graph failed eager/reference validation"
                )
        for timing_name in (
            "gpu_ms_per_update_replay",
            "gpu_ms_per_suffix_replay",
            "gpu_ms_per_full_cycle",
            "host_ms_per_full_cycle",
        ):
            timing = graph.get(timing_name)
            if not isinstance(timing, dict) or any(
                not isinstance(timing.get(field), (int, float))
                or isinstance(timing.get(field), bool)
                or not math.isfinite(float(timing[field]))
                or float(timing[field]) <= 0.0
                for field in ("min", "median", "p90", "max")
            ):
                raise QualificationError(f"DFlash2 preflight has invalid {timing_name}")
    return {
        "identity": evidence_identity(resolved, DFLASH2_PREFLIGHT_SCHEMA),
        "checkpoint_repo_id": report["checkpoint_repo_id"],
        "checkpoint_revision": report["checkpoint_revision"],
        "checkpoint_config_sha256": report["checkpoint_config_sha256"],
        "checkpoint_weight_lfs_sha256": report["checkpoint_weight_lfs_sha256"],
        "kv_storage": report["kv_storage"],
        "kv_element_bytes": report["kv_element_bytes"],
        "page_size": report["page_size"],
        "kv_capacity_tokens": report["kv_capacity_tokens"],
        "proposal_tokens_per_request": proposal_width,
        "topk_backend": report["topk_backend"],
        "resident_preload": resident,
        "target_alias_preload": aliases,
        "static_graphs": graphs,
    }


def dflash2_topk_tuning(path: Path) -> dict[str, Any]:
    """Recompute the backend choice from the complete signed top-k sweep."""

    resolved = path.expanduser()
    if resolved.is_symlink():
        raise QualificationError("DFlash2 top-k tuning evidence is a symbolic link")
    resolved = resolved.resolve(strict=True)
    report = _json_object(resolved)
    report_sha256 = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        report.get("schema") != DFLASH2_TOPK_TUNING_SCHEMA
        or report.get("status") != "measured"
        or report.get("repo_id") != DFLASH2_REPO_ID
        or report.get("revision") != DFLASH2_REVISION
        or report.get("concurrency") != list(REQUIRED_CONCURRENCIES)
        or report.get("widths") != list(DFLASH2_REQUIRED_WIDTHS)
        or report.get("minimum_non_torch_speedup") != DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP
        or report.get("selection_policy") != DFLASH2_TOPK_SELECTION_POLICY
        or report.get("full_service_acceptance_required") is not True
        or report.get("script_sha256")
        != source_sha256(TOOLS_ROOT / "tune_dflash2_topk.py")
        or report.get("runtime_head_sha256")
        != source_sha256(REFERENCE_ROOT / "dflash_head_capture.py")
        or isinstance(report.get("captured_launches"), bool)
        or not isinstance(report.get("captured_launches"), int)
        or report["captured_launches"] < 8
        or not isinstance(report_sha256, str)
        or SHA256_RE.fullmatch(report_sha256) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != report_sha256
    ):
        raise QualificationError("DFlash2 top-k tuning report contract is invalid")

    valid_backends = report.get("valid_backends")
    unsupported_backends = report.get("unsupported_backends", {})
    aggregate = report.get("aggregate_median_ms")
    results = report.get("results")
    if (
        not isinstance(valid_backends, list)
        or not valid_backends
        or len(valid_backends) != len(set(valid_backends))
        or any(backend not in DFLASH2_TOPK_BACKENDS for backend in valid_backends)
        or "torch" not in valid_backends
        or not isinstance(unsupported_backends, dict)
        or any(
            backend == "torch"
            or backend not in DFLASH2_TOPK_BACKENDS
            or not isinstance(reason, str)
            or not reason
            for backend, reason in unsupported_backends.items()
        )
        or not isinstance(aggregate, dict)
        or set(aggregate) != set(valid_backends)
        or not isinstance(results, list)
        or len(results) != len(REQUIRED_CONCURRENCIES) * len(DFLASH2_REQUIRED_WIDTHS)
    ):
        raise QualificationError("DFlash2 top-k tuning report is incomplete")

    cases: set[tuple[int, int]] = set()
    recomputed_aggregate = {backend: 0.0 for backend in valid_backends}
    recomputed_valid = {backend: True for backend in DFLASH2_TOPK_BACKENDS}
    for result in results:
        if not isinstance(result, dict):
            raise QualificationError("DFlash2 top-k tuning has a malformed case")
        concurrency = result.get("active_requests")
        width = result.get("proposal_tokens")
        if (
            concurrency not in REQUIRED_CONCURRENCIES
            or width not in DFLASH2_REQUIRED_WIDTHS
            or result.get("rows") != concurrency * width
        ):
            raise QualificationError("DFlash2 top-k tuning has an invalid row case")
        case = (concurrency, width)
        if case in cases:
            raise QualificationError("DFlash2 top-k tuning repeats a row case")
        cases.add(case)
        initial = result.get("initial_valid")
        changed = result.get("changed_input_valid")
        initial_index_exact = result.get("initial_index_exact")
        changed_index_exact = result.get("changed_input_index_exact")
        timings = result.get("timing_ms")
        speedups = result.get("speedup_vs_torch")
        case_unsupported = result.get("unsupported_backends", {})
        active_backends = set(initial) if isinstance(initial, dict) else set()
        if (
            not isinstance(initial, dict)
            or not isinstance(changed, dict)
            or not isinstance(initial_index_exact, dict)
            or not isinstance(changed_index_exact, dict)
            or result.get("tie_policy")
            != "equal_topk_values_valid_unique_ids_boundary_ties_allowed"
            or not isinstance(timings, dict)
            or not isinstance(speedups, dict)
            or not isinstance(case_unsupported, dict)
            or "torch" not in active_backends
            or not active_backends <= set(DFLASH2_TOPK_BACKENDS)
            or set(changed) != active_backends
            or set(initial_index_exact) != active_backends
            or set(changed_index_exact) != active_backends
            or set(timings) != active_backends
            or set(speedups) != active_backends
            or any(
                backend == "torch"
                or backend not in DFLASH2_TOPK_BACKENDS
                or not isinstance(reason, str)
                or not reason
                for backend, reason in case_unsupported.items()
            )
            or set(DFLASH2_TOPK_BACKENDS) - active_backends != set(case_unsupported)
            or any(
                not isinstance(initial[backend], bool)
                or not isinstance(changed[backend], bool)
                or not isinstance(initial_index_exact[backend], bool)
                or not isinstance(changed_index_exact[backend], bool)
                for backend in active_backends
            )
        ):
            raise QualificationError("DFlash2 top-k tuning parity evidence is invalid")
        for backend in DFLASH2_TOPK_BACKENDS:
            recomputed_valid[backend] &= (
                backend in active_backends
                and initial.get(backend, False)
                and changed.get(backend, False)
            )
        torch_median: float | None = None
        medians: dict[str, float] = {}
        for backend in active_backends:
            summary = timings[backend]
            if not isinstance(summary, dict) or set(summary) != {
                "minimum",
                "median",
                "p90",
                "maximum",
            }:
                raise QualificationError("DFlash2 top-k timing summary is malformed")
            values = [summary[key] for key in ("minimum", "median", "p90", "maximum")]
            if (
                any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in values
                )
                or not values[0] <= values[1] <= values[2] <= values[3]
            ):
                raise QualificationError("DFlash2 top-k timing summary is invalid")
            medians[backend] = float(summary["median"])
        torch_median = medians["torch"]
        for backend in active_backends:
            expected_speedup = torch_median / medians[backend]
            require_close(
                speedups[backend],
                expected_speedup,
                f"DFlash2 top-k {case} {backend} speedup",
            )
            if backend in valid_backends:
                recomputed_aggregate[backend] += medians[backend]

    if unsupported_backends != results[-1].get("unsupported_backends", {}):
        raise QualificationError("DFlash2 top-k unsupported backend summary is invalid")

    if cases != {
        (concurrency, width)
        for concurrency in REQUIRED_CONCURRENCIES
        for width in DFLASH2_REQUIRED_WIDTHS
    }:
        raise QualificationError("DFlash2 top-k tuning does not cover C1/C2/C4 K1-K7")
    if valid_backends != [
        backend for backend in DFLASH2_TOPK_BACKENDS if recomputed_valid[backend]
    ]:
        raise QualificationError("DFlash2 top-k valid backend summary is invalid")
    for backend, expected in recomputed_aggregate.items():
        require_close(
            aggregate[backend],
            expected,
            f"DFlash2 top-k {backend} aggregate",
        )
    fastest = min(valid_backends, key=lambda backend: recomputed_aggregate[backend])
    speedup = recomputed_aggregate["torch"] / recomputed_aggregate[fastest]
    selected = (
        fastest
        if fastest == "torch" or speedup >= DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP
        else "torch"
    )
    if (
        report.get("fastest_valid_backend") != fastest
        or report.get("selected_backend") != selected
    ):
        raise QualificationError("DFlash2 top-k tuning backend selection is invalid")
    require_close(
        report.get("fastest_valid_speedup_vs_torch"),
        speedup,
        "DFlash2 top-k aggregate speedup",
    )
    return {
        "identity": evidence_identity(resolved, DFLASH2_TOPK_TUNING_SCHEMA),
        "selected_backend": selected,
        "fastest_valid_backend": fastest,
        "fastest_valid_speedup_vs_torch": speedup,
        "aggregate_median_ms": recomputed_aggregate,
        "valid_backends": valid_backends,
    }


def dflash2_fusion_tuning(path: Path, *, kind: str) -> dict[str, Any]:
    if kind == "selector":
        schema = DFLASH2_SELECTOR_TUNING_SCHEMA
        script = TOOLS_ROOT / "tune_dflash2_selector.py"
        runtime = REFERENCE_ROOT / "dflash_head_capture.py"
        runtime_hash_field = "runtime_selector_sha256"
        candidate_dtypes: tuple[str | None, ...] = ("int64", "int32")
    elif kind == "body":
        schema = DFLASH2_BODY_FUSION_TUNING_SCHEMA
        script = TOOLS_ROOT / "tune_dflash2_body_fusion.py"
        runtime = REFERENCE_ROOT / "dspark_body_capture.py"
        runtime_hash_field = "runtime_body_sha256"
        candidate_dtypes = (None,)
    else:
        raise AssertionError(f"unknown DFlash2 fusion kind: {kind}")

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise QualificationError(f"DFlash2 {kind} tuning evidence is a symbolic link")
    resolved = expanded.resolve(strict=True)
    report = _json_object(resolved)
    report_sha256 = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        report.get("schema") != schema
        or report.get("status") != "accepted"
        or report.get("repo_id") != DFLASH2_REPO_ID
        or report.get("revision") != DFLASH2_REVISION
        or report.get("minimum_fused_speedup") != DFLASH2_FUSION_MIN_SPEEDUP
        or report.get("script_sha256") != source_sha256(script)
        or report.get(runtime_hash_field) != source_sha256(runtime)
        or report.get("runtime_profile_sha256")
        != source_sha256(REFERENCE_ROOT / "dflash_tuning_profile.py")
        or report.get("fused_wins_all_cases") is not True
        or report.get("runtime_matches_winners") is not True
        or isinstance(report.get("captured_launches"), bool)
        or not isinstance(report.get("captured_launches"), int)
        or report["captured_launches"] < 8
        or not isinstance(report_sha256, str)
        or SHA256_RE.fullmatch(report_sha256) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != report_sha256
    ):
        raise QualificationError(f"DFlash2 {kind} fusion tuning contract is invalid")

    results = report.get("results")
    winners = report.get("winning_fused_warps")
    runtime_warps = report.get("runtime_fused_warps")
    expected_cases = {
        (dtype, concurrency, width, warps)
        for dtype in candidate_dtypes
        for concurrency in REQUIRED_CONCURRENCIES
        for width in DFLASH2_REQUIRED_WIDTHS
        for warps in (4, 8)
    }
    if (
        not isinstance(results, list)
        or len(results) != len(expected_cases)
        or not isinstance(winners, dict)
        or not isinstance(runtime_warps, dict)
    ):
        raise QualificationError(f"DFlash2 {kind} fusion tuning is incomplete")
    measured: dict[tuple[str | None, int, int, int], dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict):
            raise QualificationError(f"DFlash2 {kind} fusion has a malformed case")
        dtype = result.get("candidate_dtype") if kind == "selector" else None
        case = (
            dtype,
            result.get("active_requests"),
            result.get("proposal_tokens"),
            result.get("fused_warps"),
        )
        if case not in expected_cases or case in measured:
            raise QualificationError(f"DFlash2 {kind} fusion has an invalid row case")
        split = result.get("split_gpu_ms")
        fused = result.get("fused_gpu_ms")
        for label, summary in (("split", split), ("fused", fused)):
            if not isinstance(summary, dict) or set(summary) != {
                "minimum",
                "median",
                "p90",
                "maximum",
            }:
                raise QualificationError(
                    f"DFlash2 {kind} {label} timing summary is malformed"
                )
            values = [summary[key] for key in ("minimum", "median", "p90", "maximum")]
            if (
                any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    or float(value) <= 0.0
                    for value in values
                )
                or not values[0] <= values[1] <= values[2] <= values[3]
            ):
                raise QualificationError(
                    f"DFlash2 {kind} {label} timing summary is invalid"
                )
        speedup = float(split["median"]) / float(fused["median"])
        require_close(
            result.get("fused_speedup"),
            speedup,
            f"DFlash2 {kind} {case} fused speedup",
        )
        if (
            result.get("winner")
            != ("fused" if fused["median"] < split["median"] else "split")
            or result.get("performance_gate_passed")
            is not (speedup >= DFLASH2_FUSION_MIN_SPEEDUP)
            or (kind == "selector" and result.get("reference_exact") is not True)
            or (
                kind == "body"
                and (
                    result.get("residual_exact") is not True
                    or result.get("normalized_exact") is not True
                )
            )
        ):
            raise QualificationError(f"DFlash2 {kind} fusion gate is invalid")
        measured[case] = result

    if set(measured) != expected_cases:
        raise QualificationError(f"DFlash2 {kind} fusion does not cover C1/C2/C4 K1-K7")
    recomputed_winners: dict[str, int] = {}
    for dtype in candidate_dtypes:
        for concurrency in REQUIRED_CONCURRENCIES:
            for width in DFLASH2_REQUIRED_WIDTHS:
                selected = min(
                    (measured[(dtype, concurrency, width, warps)] for warps in (4, 8)),
                    key=lambda result: result["fused_gpu_ms"]["median"],
                )
                key = (
                    f"{dtype}-c{concurrency}-k{width}"
                    if dtype is not None
                    else f"c{concurrency}-k{width}"
                )
                recomputed_winners[key] = selected["fused_warps"]
                if selected["performance_gate_passed"] is not True:
                    raise QualificationError(
                        f"DFlash2 {kind} fusion does not win every production case"
                    )
    current_runtime_warps = {
        key: (
            dflash2_selector_num_warps(concurrency, width, dtype)
            if dtype is not None
            else dflash2_body_num_warps(concurrency, width)
        )
        for dtype in candidate_dtypes
        for concurrency in REQUIRED_CONCURRENCIES
        for width in DFLASH2_REQUIRED_WIDTHS
        for key in (
            (
                f"{dtype}-c{concurrency}-k{width}"
                if dtype is not None
                else f"c{concurrency}-k{width}"
            ),
        )
    }
    if (
        winners != recomputed_winners
        or runtime_warps != recomputed_winners
        or current_runtime_warps != recomputed_winners
    ):
        raise QualificationError(
            f"DFlash2 {kind} serving warp choices differ from measured winners"
        )
    if kind == "body":
        real_validation = report.get("real_weight_validation")
        expected_real_cases = {
            (f"layer-{layer}-{side}", concurrency, width)
            for layer in range(6)
            for side in ("attention", "mlp")
            for concurrency in REQUIRED_CONCURRENCIES
            for width in DFLASH2_REQUIRED_WIDTHS
        }
        actual_real_cases: set[tuple[str, int, int]] = set()
        if not isinstance(real_validation, list) or len(real_validation) != len(
            expected_real_cases
        ):
            raise QualificationError(
                "DFlash2 body fusion lacks complete real-weight validation"
            )
        for validation in real_validation:
            if not isinstance(validation, dict):
                raise QualificationError(
                    "DFlash2 body fusion has malformed real-weight validation"
                )
            real_case = (
                validation.get("weight_case"),
                validation.get("active_requests"),
                validation.get("proposal_tokens"),
            )
            if (
                real_case not in expected_real_cases
                or real_case in actual_real_cases
                or validation.get("query_rows_per_request") != real_case[2] + 1
                or validation.get("total_rows") != real_case[1] * (real_case[2] + 1)
                or validation.get("fused_warps")
                != recomputed_winners[f"c{real_case[1]}-k{real_case[2]}"]
                or validation.get("residual_exact") is not True
                or validation.get("normalized_exact") is not True
            ):
                raise QualificationError(
                    "DFlash2 body fusion real-weight validation is invalid"
                )
            actual_real_cases.add(real_case)
        if actual_real_cases != expected_real_cases:
            raise QualificationError(
                "DFlash2 body fusion does not validate every real-weight branch"
            )
    return {
        "identity": evidence_identity(resolved, schema),
        "kind": kind,
        "winning_fused_warps": recomputed_winners,
        "captured_launches": report["captured_launches"],
    }


def _agentic_code_decode_tps(evidence: dict[str, Any], label: str) -> float:
    case_results = evidence.get("blended", {}).get("case_results")
    code_rows = (
        [
            row
            for row in case_results
            if isinstance(row, dict) and row.get("case") == "code"
        ]
        if isinstance(case_results, list)
        else []
    )
    if len(code_rows) != 1:
        raise QualificationError(f"{label} has no unique code decode result")
    value = code_rows[0].get("decode_tps")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0.0
    ):
        raise QualificationError(f"{label} code decode TPS is invalid")
    return float(value)


def _agentic_concurrency_geomean_tps(evidence: dict[str, Any], label: str) -> float:
    cells = evidence.get("concurrency", {}).get("cells")
    if not isinstance(cells, dict) or set(cells) != set(REQUIRED_CONCURRENCIES):
        raise QualificationError(f"{label} has no complete C1/C2/C4 code curve")
    values = []
    for concurrency in REQUIRED_CONCURRENCIES:
        cell = cells.get(concurrency)
        value = (
            cell.get("median_aggregate_decode_tps") if isinstance(cell, dict) else None
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0.0
        ):
            raise QualificationError(
                f"{label} C{concurrency} median aggregate decode TPS is invalid"
            )
        values.append(float(value))
    return math.exp(statistics.fmean(math.log(value) for value in values))


def select_default_mode(modes: dict[str, dict[str, Any]]) -> str:
    """Choose the default from response throughput, independently of tool quality."""

    if set(modes) != set(MODES):
        raise QualificationError("default selection requires native MTP and DFlash2")

    def key(mode: str) -> tuple[float, float, float, int]:
        evidence = modes[mode]
        weighted_decode_tps = evidence["blended"]["wall_decode_tps"]
        if not math.isfinite(weighted_decode_tps) or weighted_decode_tps <= 0.0:
            raise QualificationError(f"{mode} weighted decode TPS is invalid")
        code_decode_tps = _agentic_code_decode_tps(evidence, mode)
        # Prefer DFlash2 only for a literal tie so selection is deterministic.
        return (
            math.sqrt(code_decode_tps * weighted_decode_tps),
            weighted_decode_tps,
            code_decode_tps,
            int(mode == MODE_DFLASH2),
        )

    return max(MODES, key=key)


def select_dflash2_width(trials: dict[int, dict[str, Any]]) -> int:
    if set(trials) != set(DFLASH2_REQUIRED_WIDTHS):
        raise QualificationError("DFlash2 tuning requires fixed widths 1 through 7")

    def key(width: int) -> tuple[float, float, float, int]:
        trial = trials[width]
        weighted_decode_tps = trial["blended"]["wall_decode_tps"]
        if not math.isfinite(weighted_decode_tps) or weighted_decode_tps <= 0.0:
            raise QualificationError(f"DFlash2 width {width} decode TPS is invalid")
        code_decode_tps = _agentic_code_decode_tps(trial, f"DFlash2 width {width}")
        # Exact ties prefer the narrower target-verification wave.
        return (
            math.sqrt(code_decode_tps * weighted_decode_tps),
            weighted_decode_tps,
            code_decode_tps,
            -width,
        )

    return max(DFLASH2_REQUIRED_WIDTHS, key=key)


def require_distinct_launch_instances(
    label: str, deployments: list[dict[str, Any]]
) -> list[int]:
    launch_ids = [deployment["launch_started_ns"] for deployment in deployments]
    if len(set(launch_ids)) != len(launch_ids):
        raise QualificationError(f"{label} reused a service launch")
    return launch_ids


def dflash2_cost_profile(path: Path, *, deployed: dict[str, Any]) -> dict[str, Any]:
    """Validate the generated adaptive cost surface used by the deployment."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise QualificationError("DFlash2 cost profile is a symbolic link")
    resolved = expanded.resolve(strict=True)
    report = _json_object(resolved)
    source_sha256 = report.get("source_sha256")
    body = {key: value for key, value in report.items() if key != "source_sha256"}
    identity = report.get("identity")
    qualification = report.get("qualification")
    curves = report.get("curves")
    settings = deployed["speculation_settings"]
    if (
        report.get("schema") != DFLASH2_COST_PROFILE_SCHEMA
        or not isinstance(source_sha256, str)
        or SHA256_RE.fullmatch(source_sha256) is None
        or hashlib.sha256(
            json.dumps(body, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
        != source_sha256
        or not isinstance(identity, dict)
        or identity.get("target_model") != GLM53_MODEL_ID
        or identity.get("target_revision") != deployed["model_revision"]
        or identity.get("dspark_model") != settings.get("checkpoint_model_id")
        or identity.get("dspark_revision") != settings.get("checkpoint_revision")
        or identity.get("sparkinfer_revision") != deployed["sparkinfer_revision"]
        or identity.get("power_limit_watts") != deployed["power_limit_w"]
        or identity.get("max_concurrency") != max(REQUIRED_CONCURRENCIES)
        or identity.get("max_drafts") != max(DFLASH2_REQUIRED_WIDTHS)
        or settings.get("draft_policy") != "adaptive"
        or settings.get("fixed_drafts") is not None
        or settings.get("proposal_drafts") != max(DFLASH2_REQUIRED_WIDTHS)
        or not isinstance(qualification, dict)
        or integer(
            qualification.get("route_qualified_cells_adopted"),
            "DFlash2 route-qualified cells",
            minimum=1,
        )
        < 1
        or integer(
            qualification.get("corpus_samples_used"),
            "DFlash2 cost-profile corpus samples",
            minimum=1,
        )
        < 1
        or not isinstance(curves, dict)
        or set(curves) != {"1", "2", "3", "4"}
    ):
        raise QualificationError("DFlash2 adaptive cost-profile contract is invalid")
    for concurrency in range(1, max(REQUIRED_CONCURRENCIES) + 1):
        cells = curves[str(concurrency)]
        expected_rows = list(
            range(concurrency, concurrency * (max(DFLASH2_REQUIRED_WIDTHS) + 1) + 1)
        )
        if (
            not isinstance(cells, list)
            or [cell.get("target_rows") for cell in cells] != expected_rows
            or any(
                finite_positive(
                    cell.get("latency_ms"),
                    f"DFlash2 C{concurrency} cost-profile latency",
                )
                <= 0.0
                for cell in cells
                if isinstance(cell, dict)
            )
            or any(not isinstance(cell, dict) for cell in cells)
        ):
            raise QualificationError(
                f"DFlash2 adaptive cost profile has an invalid C{concurrency} curve"
            )
    generated = (
        Path(__file__).resolve().parents[2]
        / "rust/crates/glmrt-daemon/src/commands/real_full/dflash2_cost_profile.rs"
    ).read_text(encoding="utf-8")
    if source_sha256 not in generated or str(report.get("profile_id")) not in generated:
        raise QualificationError(
            "runtime DFlash2 cost profile differs from JSON evidence"
        )
    return {
        "identity": evidence_identity(resolved, DFLASH2_COST_PROFILE_SCHEMA),
        "profile_id": report["profile_id"],
        "source_sha256": source_sha256,
        "route_qualified_cells": qualification["route_qualified_cells_adopted"],
        "corpus_samples": qualification["corpus_samples_used"],
    }


def dflash2_k5_reference(
    *,
    deployment_path: Path,
    blended_path: Path,
    concurrency_paths: list[Path],
) -> dict[str, Any]:
    """Validate the fixed-K5 response-throughput reference arm."""

    deployed = deployment(
        deployment_path,
        candidate=True,
        expected_model=GLM53_MODEL_ID,
        expected_speculation=MODE_DFLASH2,
    )
    if (
        deployed["speculation_settings"].get("draft_policy") != "fixed"
        or deployed["speculation_settings"].get("proposal_drafts")
        != max(DFLASH2_REQUIRED_WIDTHS)
        or deployed["speculation_settings"].get("fixed_drafts")
        != DFLASH2_REFERENCE_WIDTH
    ):
        raise QualificationError("DFlash2 reference deployment is not fixed K5")
    decoded = blended(
        blended_path,
        candidate=True,
        expected_model=GLM53_MODEL_ID,
    )
    require_eight_type_blended(decoded)
    decoded["verify_cycle_by_physical_m"] = verify_cycle_curve(
        blended_path, expected_fixed_drafts=DFLASH2_REFERENCE_WIDTH
    )
    concurrent = decode_concurrency(
        concurrency_paths,
        expected_model=GLM53_MODEL_ID,
    )
    return {
        "deployment": deployed,
        "blended": decoded,
        "concurrency": concurrent,
    }


def dflash2_topk_service_gate(
    *,
    torch_deployment_path: Path,
    torch_blended_path: Path,
    candidate_deployment_path: Path,
    candidate_blended_path: Path,
    topk_tuning: dict[str, Any],
) -> dict[str, Any]:
    """Choose the production top-k backend from a matched fixed-K5 service A/B."""

    arms: dict[str, dict[str, Any]] = {}
    for arm, deployment_path, blended_path in (
        ("torch", torch_deployment_path, torch_blended_path),
        ("candidate", candidate_deployment_path, candidate_blended_path),
    ):
        deployed = deployment(
            deployment_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=MODE_DFLASH2,
        )
        settings = deployed["speculation_settings"]
        if (
            settings.get("draft_policy") != "fixed"
            or settings.get("fixed_drafts") != DFLASH2_REFERENCE_WIDTH
            or settings.get("proposal_drafts") != max(DFLASH2_REQUIRED_WIDTHS)
        ):
            raise QualificationError(f"DFlash2 top-k {arm} service arm is not fixed K5")
        decoded = blended(
            blended_path,
            candidate=False,
            expected_model=GLM53_MODEL_ID,
        )
        require_eight_type_blended(decoded)
        _, raw_records = read_jsonl(blended_path)
        content_sha256 = [
            record.get("content_sha256")
            for record in raw_records
            if "aggregate" not in record
        ]
        if len(content_sha256) != decoded["cases"] or any(
            not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None
            for digest in content_sha256
        ):
            raise QualificationError(
                f"DFlash2 top-k {arm} service arm has invalid response hashes"
            )
        arms[arm] = {
            "deployment": deployed,
            "blended": decoded,
            "content_sha256": content_sha256,
        }

    torch = arms["torch"]
    candidate = arms["candidate"]
    torch_backend = torch["deployment"]["speculation_settings"]["topk_backend"]
    candidate_backend = candidate["deployment"]["speculation_settings"]["topk_backend"]
    if torch_backend != "torch":
        raise QualificationError("DFlash2 top-k service reference is not Torch")
    if (
        candidate_backend != topk_tuning["selected_backend"]
        or candidate_backend == "torch"
    ):
        raise QualificationError(
            "DFlash2 top-k service candidate differs from the micro-selected backend"
        )
    for label in (
        "slot",
        "profile",
        "power_limit_w",
        "engine_identity",
        "sparkinfer_revision",
        "model_revision",
    ):
        paired_equal(
            f"DFlash2 top-k service A/B deployment {label}",
            torch["deployment"][label],
            candidate["deployment"][label],
        )
    for label in ("coordinator_slot", "expert_slot"):
        paired_equal(
            f"DFlash2 top-k service A/B {label} fingerprint",
            torch["deployment"]["fingerprints"][label],
            candidate["deployment"]["fingerprints"][label],
        )
    for label in (
        "checkpoint_model_id",
        "checkpoint_revision",
        "draft_policy",
        "fixed_drafts",
        "proposal_drafts",
    ):
        paired_equal(
            f"DFlash2 top-k service A/B {label}",
            torch["deployment"]["speculation_settings"][label],
            candidate["deployment"]["speculation_settings"][label],
        )
    if (
        torch["deployment"]["launch_started_ns"]
        == candidate["deployment"]["launch_started_ns"]
    ):
        raise QualificationError("DFlash2 top-k service arms reuse one launch")
    for label in ("contract", "prompt_contract", "prompts"):
        paired_equal(
            f"DFlash2 top-k service A/B blended {label}",
            torch["blended"][label],
            candidate["blended"][label],
        )
    if torch["blended"]["all_quality_contracts_passed"] is not True:
        raise QualificationError("DFlash2 Torch top-k service reference failed quality")

    wall_speedup = ratio(
        candidate["blended"]["wall_decode_tps"],
        torch["blended"]["wall_decode_tps"],
        "DFlash2 candidate/Torch top-k weighted decode",
    )
    median_speedup = ratio(
        candidate["blended"]["median_repeat_wall_decode_tps"],
        torch["blended"]["median_repeat_wall_decode_tps"],
        "DFlash2 candidate/Torch top-k median repeat decode",
    )
    candidate_quality = candidate["blended"]["all_quality_contracts_passed"]
    selected_backend = (
        candidate_backend
        if candidate_quality
        and min(wall_speedup, median_speedup) >= DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP
        else "torch"
    )
    return {
        "selection_policy": DFLASH2_TOPK_SERVICE_SELECTION_POLICY,
        "minimum_non_torch_speedup": DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP,
        "selected_backend": selected_backend,
        "candidate_backend": candidate_backend,
        "candidate_quality_passed": candidate_quality,
        "candidate_quality_failures": candidate["blended"]["quality_contract_failures"],
        "candidate_speedup_vs_torch": {
            "weighted_decode": wall_speedup,
            "median_repeat_decode": median_speedup,
        },
        "weighted_decode_tps": {
            "torch": torch["blended"]["wall_decode_tps"],
            candidate_backend: candidate["blended"]["wall_decode_tps"],
        },
        "median_repeat_decode_tps": {
            "torch": torch["blended"]["median_repeat_wall_decode_tps"],
            candidate_backend: candidate["blended"]["median_repeat_wall_decode_tps"],
        },
        "accepted_draft_rate": {
            "torch": torch["blended"]["accepted_draft_rate"],
            candidate_backend: candidate["blended"]["accepted_draft_rate"],
        },
        "response_hash_mismatches": sum(
            torch_hash != candidate_hash
            for torch_hash, candidate_hash in zip(
                torch["content_sha256"],
                candidate["content_sha256"],
                strict=True,
            )
        ),
        "requests": torch["blended"]["cases"],
        "evidence": {
            "torch_deployment": torch["deployment"]["identity"],
            "torch_blended": torch["blended"]["identity"],
            "candidate_deployment": candidate["deployment"]["identity"],
            "candidate_blended": candidate["blended"]["identity"],
        },
    }


def dflash2_width_sweep(
    trials: list[tuple[int, Path, Path, Path]],
    *,
    expected_tool_eval_version: str,
) -> dict[str, Any]:
    if len(trials) != len(DFLASH2_REQUIRED_WIDTHS):
        raise QualificationError("DFlash2 tuning requires exactly seven width trials")
    parsed: dict[int, dict[str, Any]] = {}
    for width, deployment_path, blended_path, tool_eval_path in trials:
        if width in parsed or width not in DFLASH2_REQUIRED_WIDTHS:
            raise QualificationError(
                "DFlash2 tuning widths must be unique values in 1..7"
            )
        deployed = deployment(
            deployment_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=MODE_DFLASH2,
        )
        if deployed["speculation_settings"]["fixed_drafts"] != width:
            raise QualificationError(
                f"DFlash2 width {width} deployment reports another fixed width"
            )
        blended_evidence = blended(
            blended_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
        )
        require_eight_type_blended(blended_evidence)
        blended_evidence["verify_cycle_by_physical_m"] = verify_cycle_curve(
            blended_path, expected_fixed_drafts=width
        )
        parsed[width] = {
            "deployment": deployed,
            "blended": blended_evidence,
            "tool_eval": tool_eval(
                tool_eval_path,
                candidate=True,
                expected_version=expected_tool_eval_version,
                expected_model=GLM53_MODEL_ID,
            ),
        }
    if set(parsed) != set(DFLASH2_REQUIRED_WIDTHS):
        raise QualificationError(
            "DFlash2 tuning does not cover fixed widths 1 through 7"
        )
    require_distinct_launch_instances(
        "DFlash2 width sweep",
        [parsed[width]["deployment"] for width in DFLASH2_REQUIRED_WIDTHS],
    )

    reference = parsed[DFLASH2_REQUIRED_WIDTHS[0]]
    for width in DFLASH2_REQUIRED_WIDTHS[1:]:
        trial = parsed[width]
        for label in (
            "slot",
            "profile",
            "power_limit_w",
            "engine_identity",
            "sparkinfer_revision",
            "model_revision",
        ):
            paired_equal(
                f"DFlash2 width sweep deployment {label}",
                reference["deployment"][label],
                trial["deployment"][label],
            )
        paired_equal(
            "DFlash2 width sweep top-k backend",
            reference["deployment"]["speculation_settings"]["topk_backend"],
            trial["deployment"]["speculation_settings"]["topk_backend"],
        )
        for label in ("coordinator_slot", "expert_slot"):
            paired_equal(
                f"DFlash2 width sweep {label} fingerprint",
                reference["deployment"]["fingerprints"][label],
                trial["deployment"]["fingerprints"][label],
            )
        paired_equal(
            "DFlash2 width sweep blended prompt contract",
            reference["blended"]["contract"],
            trial["blended"]["contract"],
        )
        paired_equal(
            "DFlash2 width sweep blended prompt body",
            reference["blended"]["prompt_contract"],
            trial["blended"]["prompt_contract"],
        )
        paired_equal(
            "DFlash2 width sweep blended prompts",
            reference["blended"]["prompts"],
            trial["blended"]["prompts"],
        )
        _pair_tool_evaluations(reference["tool_eval"], trial["tool_eval"])

    winner = select_dflash2_width(parsed)
    return {
        "winner": winner,
        "selection": DFLASH2_WIDTH_SELECTION_POLICY,
        "runtime": {
            key: reference["deployment"][key]
            for key in (
                "slot",
                "profile",
                "power_limit_w",
                "engine_identity",
                "sparkinfer_revision",
                "model_revision",
            )
        },
        "slot_fingerprints": {
            key: reference["deployment"]["fingerprints"][key]
            for key in ("coordinator_slot", "expert_slot")
        },
        "trials": [
            {
                "width": width,
                "launch_started_ns": parsed[width]["deployment"]["launch_started_ns"],
                "tool_points": parsed[width]["tool_eval"]["total_points"],
                "tool_maximum_points": parsed[width]["tool_eval"]["maximum_points"],
                "tool_score": parsed[width]["tool_eval"]["final_score"],
                "code_decode_tps": _agentic_code_decode_tps(
                    parsed[width], f"DFlash2 width {width}"
                ),
                "weighted_decode_tps": parsed[width]["blended"]["wall_decode_tps"],
                "accepted_draft_rate": parsed[width]["blended"]["accepted_draft_rate"],
                "verify_cycle_by_physical_m": parsed[width]["blended"][
                    "verify_cycle_by_physical_m"
                ],
                "evidence": {
                    "deployment": parsed[width]["deployment"]["identity"],
                    "blended": parsed[width]["blended"]["identity"],
                    "tool_eval": parsed[width]["tool_eval"]["identity"],
                },
            }
            for width in DFLASH2_REQUIRED_WIDTHS
        ],
        "parsed": parsed,
    }


def _read_mode(
    *,
    mode: str,
    blended_path: Path,
    repeat_path: Path,
    prefill_path: Path | None,
    tool_eval_path: Path | None,
    startup_path: Path,
    deployment_path: Path,
    concurrency_paths: list[Path] | None,
    needle_path: Path | None,
    expected_tool_eval_version: str,
) -> dict[str, Any]:
    if mode not in MODES:
        raise QualificationError(f"unsupported GLM-5.3 speculation mode: {mode}")
    deployed = deployment(
        deployment_path,
        candidate=True,
        expected_model=GLM53_MODEL_ID,
        expected_speculation=mode,
    )
    started = startup(
        startup_path,
        candidate=True,
        expected_model=GLM53_MODEL_ID,
        expected_weight_format="exl3",
        expected_preload_modes={"direct-resident", "cooperative-coalesced"},
        expected_include_mtp=mode == MODE_NATIVE_MTP,
        expected_schema=GLM53_STARTUP_SCHEMA,
    )
    if (
        started["expert_runtime_fingerprint"]
        != deployed["fingerprints"]["expert_runtime"]
    ):
        raise QualificationError(f"{mode} startup/runtime fingerprint differs")
    blended_evidence = blended(
        blended_path,
        candidate=True,
        expected_model=GLM53_MODEL_ID,
    )
    require_eight_type_blended(blended_evidence)
    blended_evidence["verify_cycle_by_physical_m"] = verify_cycle_curve(
        blended_path,
        expected_fixed_drafts=(
            deployed["speculation_settings"]["fixed_drafts"]
            if mode == MODE_DFLASH2
            else None
        ),
    )
    result = {
        "deployment": deployed,
        "startup": started,
        "blended": blended_evidence,
        "repeat": repeat_decode(
            repeat_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
        ),
    }
    extended = (prefill_path, tool_eval_path, concurrency_paths, needle_path)
    if mode == MODE_DFLASH2:
        if any(value is None for value in extended):
            raise QualificationError("DFlash2 requires complete release evidence")
        assert prefill_path is not None
        assert tool_eval_path is not None
        assert concurrency_paths is not None
        assert needle_path is not None
        result.update(
            {
                "prefill": prefill(
                    prefill_path,
                    candidate=True,
                    expected_model=GLM53_MODEL_ID,
                ),
                "tool_eval": tool_eval(
                    tool_eval_path,
                    candidate=True,
                    expected_version=expected_tool_eval_version,
                    expected_model=GLM53_MODEL_ID,
                ),
                "concurrency": decode_concurrency(
                    concurrency_paths,
                    expected_model=GLM53_MODEL_ID,
                ),
                "needle": long_context_needle(
                    needle_path,
                    expected_model=GLM53_MODEL_ID,
                ),
            }
        )
    elif any(value is not None for value in extended):
        raise QualificationError("native MTP accepts only paired decode evidence")
    return result


def _pair_modes(native: dict[str, Any], dflash2: dict[str, Any]) -> None:
    for label in (
        "slot",
        "profile",
        "power_limit_w",
        "engine_identity",
        "sparkinfer_revision",
        "model_revision",
    ):
        paired_equal(
            f"native-MTP/DFlash2 deployment {label}",
            native["deployment"][label],
            dflash2["deployment"][label],
        )
    for label in ("coordinator_slot", "expert_slot"):
        paired_equal(
            f"native-MTP/DFlash2 deployment {label} fingerprint",
            native["deployment"]["fingerprints"][label],
            dflash2["deployment"]["fingerprints"][label],
        )
    paired_equal(
        "blended prompt contract",
        native["blended"]["contract"],
        dflash2["blended"]["contract"],
    )
    paired_equal(
        "blended prompt contract body",
        native["blended"]["prompt_contract"],
        dflash2["blended"]["prompt_contract"],
    )
    paired_equal(
        "blended prompt sequence",
        native["blended"]["prompts"],
        dflash2["blended"]["prompts"],
    )
    paired_equal(
        "repeat prompt contract",
        native["repeat"]["contract"],
        dflash2["repeat"]["contract"],
    )
    paired_equal(
        "repeat tokenizer",
        native["repeat"]["tokenizer_sha256"],
        dflash2["repeat"]["tokenizer_sha256"],
    )
    paired_equal(
        "repeat prompt sequence",
        native["repeat"]["prompts"],
        dflash2["repeat"]["prompts"],
    )


def _pair_tool_evaluations(native: dict[str, Any], dflash2: dict[str, Any]) -> None:
    """Require the same test opportunity without requiring the same outcome."""

    paired_equal(
        "tool-evaluation configuration",
        native["config"],
        dflash2["config"],
    )
    paired_equal(
        "tool-evaluation scenario sequence",
        native["scenario_ids"],
        dflash2["scenario_ids"],
    )
    paired_equal(
        "tool-evaluation maximum points",
        native["maximum_points"],
        dflash2["maximum_points"],
    )


def _mode_result(mode: dict[str, Any]) -> dict[str, Any]:
    result = {
        "speculation": mode["deployment"]["speculation"],
        "speculation_settings": mode["deployment"]["speculation_settings"],
        "weighted_decode_tps": mode["blended"]["wall_decode_tps"],
        "agentic_code_decode_tps": _agentic_code_decode_tps(
            mode, mode["deployment"]["speculation"]
        ),
        "weighted_median_repeat_tps": mode["blended"]["median_repeat_wall_decode_tps"],
        "accepted_draft_rate": mode["blended"]["accepted_draft_rate"],
        "verify_cycle_by_physical_m": mode["blended"]["verify_cycle_by_physical_m"],
        "semantic_cases": len(mode["blended"]["case_results"]),
        "semantic_generations": mode["blended"]["cases"],
        "semantic_case_results": mode["blended"]["case_results"],
        "all_semantic_contracts_passed": mode["blended"][
            "all_quality_contracts_passed"
        ],
        "repeat_decode_tps": mode["repeat"]["aggregate_decode_tps"],
        "repeat_all_exact": mode["repeat"]["all_exact_repetition_count"],
        "maximum_resident_preload_ms": mode["startup"]["maximum_resident_preload_ms"],
        "maximum_service_handoff_total_ms": mode["startup"][
            "maximum_service_handoff_total_ms"
        ],
        "include_mtp": mode["startup"]["include_mtp"],
    }
    if "tool_eval" in mode:
        result.update(
            {
                "tool_points": mode["tool_eval"]["total_points"],
                "tool_maximum_points": mode["tool_eval"]["maximum_points"],
                "tool_score": mode["tool_eval"]["final_score"],
                "agentic_c1_c2_c4_geomean_tps": _agentic_concurrency_geomean_tps(
                    mode, mode["deployment"]["speculation"]
                ),
                "decode_concurrency": {
                    str(concurrency): {
                        "mean_aggregate_decode_tps": cell["mean_aggregate_decode_tps"],
                        "median_aggregate_decode_tps": cell[
                            "median_aggregate_decode_tps"
                        ],
                        "mean_response_window_tps": cell["mean_response_window_tps"],
                    }
                    for concurrency, cell in mode["concurrency"]["cells"].items()
                },
                "long_context_needle": {
                    "maximum_wall_seconds": mode["needle"]["maximum_wall_seconds"],
                    "median_wall_seconds": mode["needle"]["median_wall_seconds"],
                    "measurements": mode["needle"]["measurements"],
                },
            }
        )
    return result


def qualify(
    *,
    artifact_path: Path,
    native_model_snapshot_path: Path,
    artifact_validation_path: Path,
    quant_evidence_path: Path,
    native_blended_path: Path,
    dflash2_blended_path: Path,
    native_repeat_path: Path,
    dflash2_repeat_path: Path,
    dflash2_prefill_path: Path,
    dflash2_tool_eval_path: Path,
    native_startup_path: Path,
    dflash2_startup_path: Path,
    native_deployment_path: Path,
    dflash2_deployment_path: Path,
    dflash2_concurrency_paths: list[Path],
    dflash2_needle_path: Path,
    dflash2_preflight_path: Path,
    dflash2_topk_tuning_path: Path,
    dflash2_selector_tuning_path: Path,
    dflash2_body_fusion_tuning_path: Path,
    dflash2_cost_profile_path: Path,
    dflash2_k5_deployment_path: Path,
    dflash2_k5_blended_path: Path,
    dflash2_k5_concurrency_paths: list[Path],
    dflash2_topk_candidate_deployment_path: Path,
    dflash2_topk_candidate_blended_path: Path,
    native_validation_paths: list[Path],
    expected_default: str = "auto",
    expected_tool_eval_version: str = TOOL_EVAL_VERSION,
) -> dict[str, Any]:
    artifact = artifact_path.expanduser().resolve(strict=True)
    native_model_snapshot = native_model_snapshot_path.expanduser().resolve(strict=True)
    plan = _json_object(artifact / "glmrt-gptqmodel-plan.json")
    contract = _artifact_contract(plan)
    if (
        contract.model_id != GLM53_MODEL_ID
        or contract.artifact_schema != GLM53_ARTIFACT_SCHEMA
        or contract.exl3_bits != 4
    ):
        raise QualificationError(
            "serving qualification requires the GLM-5.3 EXL3 K4 contract"
        )
    artifact_manifest = _json_object(artifact / "glmrt-gptqmodel-artifact.json")
    if artifact_manifest.get("schema") != contract.artifact_schema:
        raise QualificationError(
            "qualification artifact is not a completed GLM-5.3 K4 export"
        )
    native_snapshot_manifest = _json_object(
        native_model_snapshot / "glmrt-gptqmodel-artifact.json"
    )
    if native_snapshot_manifest.get("manifest_sha256") != artifact_manifest.get(
        "manifest_sha256"
    ):
        raise QualificationError(
            "native validation snapshot differs from the accepted K4 artifact"
        )
    validation_identity, validation = _validation_evidence(
        artifact_validation_path,
        artifact=artifact,
        artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
        contract=contract,
    )
    quant_identity, quant = _quant_evidence(
        quant_evidence_path,
        plan_sha256=validation["plan_sha256"],
        contract=contract,
    )
    if (
        validation["projection_checkpoint"]["checkpoint_inventory_sha256"]
        != quant["integrity"]["checkpoint_inventory_sha256"]
    ):
        raise QualificationError(
            "artifact and quant evidence bind different projection inventories"
        )

    modes = {
        MODE_NATIVE_MTP: _read_mode(
            mode=MODE_NATIVE_MTP,
            blended_path=native_blended_path,
            repeat_path=native_repeat_path,
            prefill_path=None,
            tool_eval_path=None,
            startup_path=native_startup_path,
            deployment_path=native_deployment_path,
            concurrency_paths=None,
            needle_path=None,
            expected_tool_eval_version=expected_tool_eval_version,
        ),
        MODE_DFLASH2: _read_mode(
            mode=MODE_DFLASH2,
            blended_path=dflash2_blended_path,
            repeat_path=dflash2_repeat_path,
            prefill_path=dflash2_prefill_path,
            tool_eval_path=dflash2_tool_eval_path,
            startup_path=dflash2_startup_path,
            deployment_path=dflash2_deployment_path,
            concurrency_paths=dflash2_concurrency_paths,
            needle_path=dflash2_needle_path,
            expected_tool_eval_version=expected_tool_eval_version,
        ),
    }
    native = modes[MODE_NATIVE_MTP]
    dflash2 = modes[MODE_DFLASH2]
    _pair_modes(native, dflash2)
    require_release_prefill_grid(dflash2["prefill"])
    dflash_preflight = dflash2_preflight(dflash2_preflight_path)
    topk_tuning = dflash2_topk_tuning(dflash2_topk_tuning_path)
    selector_tuning = dflash2_fusion_tuning(
        dflash2_selector_tuning_path, kind="selector"
    )
    body_fusion_tuning = dflash2_fusion_tuning(
        dflash2_body_fusion_tuning_path, kind="body"
    )
    cost_profile = dflash2_cost_profile(
        dflash2_cost_profile_path,
        deployed=dflash2["deployment"],
    )
    k5_reference = dflash2_k5_reference(
        deployment_path=dflash2_k5_deployment_path,
        blended_path=dflash2_k5_blended_path,
        concurrency_paths=dflash2_k5_concurrency_paths,
    )
    topk_service_gate = dflash2_topk_service_gate(
        torch_deployment_path=dflash2_k5_deployment_path,
        torch_blended_path=dflash2_k5_blended_path,
        candidate_deployment_path=dflash2_topk_candidate_deployment_path,
        candidate_blended_path=dflash2_topk_candidate_blended_path,
        topk_tuning=topk_tuning,
    )
    release_launch_ids = require_distinct_launch_instances(
        "GLM-5.3 qualification",
        [
            native["deployment"],
            dflash2["deployment"],
            k5_reference["deployment"],
            deployment(
                dflash2_topk_candidate_deployment_path,
                candidate=True,
                expected_model=GLM53_MODEL_ID,
                expected_speculation=MODE_DFLASH2,
            ),
        ],
    )
    selected_topk_backend = dflash2["deployment"]["speculation_settings"][
        "topk_backend"
    ]
    if (
        dflash_preflight["proposal_tokens_per_request"]
        != dflash2["deployment"]["speculation_settings"]["proposal_drafts"]
    ):
        raise QualificationError(
            "DFlash2 preflight graph geometry does not use the adaptive proposal width"
        )
    if dflash_preflight["topk_backend"] != selected_topk_backend:
        raise QualificationError(
            "DFlash2 preflight graph does not use the final top-k backend"
        )
    if topk_service_gate["selected_backend"] != selected_topk_backend:
        raise QualificationError(
            "final DFlash2 deployment does not use the service-qualified top-k backend"
        )
    for label in (
        "slot",
        "profile",
        "power_limit_w",
        "engine_identity",
        "sparkinfer_revision",
        "model_revision",
    ):
        paired_equal(
            f"DFlash2 K5/final deployment {label}",
            k5_reference["deployment"][label],
            dflash2["deployment"][label],
        )
    for label in ("coordinator_slot", "expert_slot"):
        paired_equal(
            f"DFlash2 K5/final {label} fingerprint",
            k5_reference["deployment"]["fingerprints"][label],
            dflash2["deployment"]["fingerprints"][label],
        )
    for label in (
        "checkpoint_model_id",
        "checkpoint_revision",
        "proposal_drafts",
        "topk_backend",
    ):
        paired_equal(
            f"DFlash2 K5/final {label}",
            k5_reference["deployment"]["speculation_settings"][label],
            dflash2["deployment"]["speculation_settings"][label],
        )
    paired_equal(
        "DFlash2 K5/final blended prompt contract",
        k5_reference["blended"]["contract"],
        dflash2["blended"]["contract"],
    )
    paired_equal(
        "DFlash2 K5/final blended prompt body",
        k5_reference["blended"]["prompt_contract"],
        dflash2["blended"]["prompt_contract"],
    )
    paired_equal(
        "DFlash2 K5/final blended prompts",
        k5_reference["blended"]["prompts"],
        dflash2["blended"]["prompts"],
    )
    paired_equal(
        "DFlash2 K5/final concurrency fixture",
        k5_reference["concurrency"]["fixture"],
        dflash2["concurrency"]["fixture"],
    )
    for concurrency in REQUIRED_CONCURRENCIES:
        paired_equal(
            f"DFlash2 K5/final C{concurrency} request contract",
            k5_reference["concurrency"]["cells"][concurrency]["request_contract"],
            dflash2["concurrency"]["cells"][concurrency]["request_contract"],
        )
    native_kernel = native_validations(
        native_validation_paths,
        expected_sparkinfer_revision=native["deployment"]["sparkinfer_revision"],
        expected_checkpoint_root=native_model_snapshot,
        expected_expert_slot_fingerprint=native["deployment"]["fingerprints"][
            "expert_slot"
        ],
        expected_trellis_bits=4,
        expected_required_rows=K4_REQUIRED_NATIVE_ROWS,
        capacity_rows_for_live_rows=exl3_k4_capacity_rows,
        route_block_rows_for_capacity=exl3_k4_route_block_rows,
        expected_artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
    )

    prefill_cells = [
        {
            "base_context_tokens": key[0],
            "suffix_tokens": key[1],
            "dflash2_tps": dflash2["prefill"]["cells"][key],
        }
        for key in sorted(dflash2["prefill"]["cells"])
    ]
    semantic_decode_cells = []
    for native_case, dflash2_case in zip(
        native["blended"]["case_results"],
        dflash2["blended"]["case_results"],
        strict=True,
    ):
        paired_equal("semantic decode case", native_case["case"], dflash2_case["case"])
        paired_equal(
            "semantic decode category",
            native_case["category"],
            dflash2_case["category"],
        )
        semantic_decode_cells.append(
            {
                "case": native_case["case"],
                "category": native_case["category"],
                "samples": native_case["samples"],
                "native_mtp_decode_tps": native_case["decode_tps"],
                "dflash2_decode_tps": dflash2_case["decode_tps"],
                "dflash2_to_native_decode_ratio": ratio(
                    dflash2_case["decode_tps"],
                    native_case["decode_tps"],
                    f"semantic decode case {native_case['case']}",
                ),
                "native_mtp_accepted_draft_rate": native_case["accepted_draft_rate"],
                "dflash2_accepted_draft_rate": dflash2_case["accepted_draft_rate"],
            }
        )
    selected_default = select_default_mode(modes)
    if expected_default != "auto" and expected_default not in MODES:
        raise QualificationError("expected default must be auto, mtp, or dflash2")
    default_matches = expected_default == "auto" or expected_default == selected_default
    adaptive_response_score = math.sqrt(
        _agentic_code_decode_tps(dflash2, "DFlash2 adaptive")
        * dflash2["blended"]["wall_decode_tps"]
    )
    k5_response_score = math.sqrt(
        _agentic_code_decode_tps(k5_reference, "DFlash2 K5")
        * k5_reference["blended"]["wall_decode_tps"]
    )
    adaptive_concurrency_score = _agentic_concurrency_geomean_tps(
        dflash2, "DFlash2 adaptive"
    )
    k5_concurrency_score = _agentic_concurrency_geomean_tps(k5_reference, "DFlash2 K5")

    gates = {
        "blended_decode": default_matches,
        "blended_acceptance": all(
            mode["blended"]["accepted_draft_rate"] > 0.0 for mode in modes.values()
        ),
        "decode_concurrency": all(
            cell["mean_aggregate_decode_tps"] > 0.0
            for cell in dflash2["concurrency"]["cells"].values()
        ),
        "long_context_needle": dflash2["needle"]["maximum_wall_seconds"]
        <= NEEDLE_MAX_REQUEST_SECONDS,
        "repeat_decode": all(
            mode["repeat"]["aggregate_decode_tps"] > 0.0 for mode in modes.values()
        ),
        "prefill_every_cell": all(cell["dflash2_tps"] > 0.0 for cell in prefill_cells),
        "tool_eval_points": dflash2["tool_eval"]["maximum_points"] > 0,
        "expert_resident_preload": all(
            mode["startup"]["maximum_resident_preload_ms"] > 0.0
            for mode in modes.values()
        ),
        "expert_startup": all(
            mode["startup"]["maximum_service_handoff_total_ms"] > 0.0
            for mode in modes.values()
        ),
        "native_kernel_parity": True,
        "dflash2_preflight": True,
        "dflash2_topk_tuning": True,
        "dflash2_topk_service_gate": topk_service_gate["selected_backend"]
        == selected_topk_backend,
        "dflash2_selector_fusion": True,
        "dflash2_body_fusion": True,
        "dflash2_adaptive_cost_profile": True,
        "dflash2_adaptive_beats_k5": adaptive_response_score >= k5_response_score
        and adaptive_concurrency_score >= k5_concurrency_score,
        "verify_cycle_measurements": all(
            bool(mode["blended"]["verify_cycle_by_physical_m"])
            for mode in modes.values()
        )
        and str(DFLASH2_REFERENCE_WIDTH + 1)
        in k5_reference["blended"]["verify_cycle_by_physical_m"],
    }
    if set(gates) != REQUIRED_GATES:
        raise AssertionError("GLM-5.3 serving qualification gate contract drifted")
    failed_gates = sorted(name for name, passed in gates.items() if not passed)
    deployment_identity = native["deployment"]
    body = {
        "schema": SCHEMA,
        "status": "accepted" if not failed_gates else "rejected",
        "model_id": GLM53_MODEL_ID,
        "artifact": os.fspath(artifact),
        "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
        "plan_sha256": validation["plan_sha256"],
        "artifact_validation": validation_identity,
        "quant_evidence": quant_identity,
        "runtime": {
            "engine_identity": deployment_identity["engine_identity"],
            "coordinator_slot_fingerprint": deployment_identity["fingerprints"][
                "coordinator_slot"
            ],
            "expert_slot_fingerprint": deployment_identity["fingerprints"][
                "expert_slot"
            ],
            "sparkinfer_revision": deployment_identity["sparkinfer_revision"],
            "profile": deployment_identity["profile"],
            "power_limit_w": deployment_identity["power_limit_w"],
            "speculation": selected_default,
            "default_speculation": selected_default,
            "qualified_speculation": list(MODES),
            "speculation_settings": {
                mode: modes[mode]["deployment"]["speculation_settings"]
                for mode in MODES
            },
            "model_revision": deployment_identity["model_revision"],
            "launch_started_ns": {
                MODE_NATIVE_MTP: native["deployment"]["launch_started_ns"],
                MODE_DFLASH2: dflash2["deployment"]["launch_started_ns"],
                "dflash2_k5_reference": release_launch_ids[2],
                "dflash2_topk_candidate": release_launch_ids[3],
            },
            "expert_runtime_fingerprints": {
                mode: modes[mode]["deployment"]["fingerprints"]["expert_runtime"]
                for mode in MODES
            },
        },
        "thresholds": {
            "default_selection": DEFAULT_SELECTION_POLICY,
            "dflash2_reference_width": DFLASH2_REFERENCE_WIDTH,
            "dflash2_topk_service_selection": DFLASH2_TOPK_SERVICE_SELECTION_POLICY,
            "tool_eval_version": expected_tool_eval_version,
            "expected_default": expected_default,
        },
        "gates": gates,
        "failed_gates": failed_gates,
        "evidence": {
            **{
                f"{mode}_{kind}": modes[mode][kind]["identity"]
                for mode in MODES
                for kind in (
                    "blended",
                    "repeat",
                    "startup",
                    "deployment",
                )
            },
            "dflash2_prefill": dflash2["prefill"]["identity"],
            "dflash2_tool_eval": dflash2["tool_eval"]["identity"],
            "dflash2_concurrency": dflash2["concurrency"]["identities"],
            "dflash2_needle": dflash2["needle"]["identity"],
            "candidate_native_validations": native_kernel["identities"],
            "dflash2_preflight": dflash_preflight["identity"],
            "dflash2_topk_tuning": topk_tuning["identity"],
            "dflash2_selector_tuning": selector_tuning["identity"],
            "dflash2_body_fusion_tuning": body_fusion_tuning["identity"],
            "dflash2_cost_profile": cost_profile["identity"],
            "dflash2_k5_deployment": k5_reference["deployment"]["identity"],
            "dflash2_k5_blended": k5_reference["blended"]["identity"],
            "dflash2_k5_concurrency": k5_reference["concurrency"]["identities"],
            "dflash2_topk_candidate_deployment": topk_service_gate["evidence"][
                "candidate_deployment"
            ],
            "dflash2_topk_candidate_blended": topk_service_gate["evidence"][
                "candidate_blended"
            ],
        },
        "results": {
            "default_speculation": selected_default,
            "modes": {mode: _mode_result(modes[mode]) for mode in MODES},
            "comparisons": {
                "dflash2_to_native_weighted_decode_ratio": ratio(
                    dflash2["blended"]["wall_decode_tps"],
                    native["blended"]["wall_decode_tps"],
                    "weighted decode",
                ),
                "dflash2_to_native_acceptance_ratio": ratio(
                    dflash2["blended"]["accepted_draft_rate"],
                    native["blended"]["accepted_draft_rate"],
                    "draft acceptance",
                ),
                "dflash2_to_native_repeat_ratio": ratio(
                    dflash2["repeat"]["aggregate_decode_tps"],
                    native["repeat"]["aggregate_decode_tps"],
                    "repeat decode",
                ),
            },
            "prefill": {
                "profile": dflash2["prefill"]["profile"],
                "prompt_contract_sha256": dflash2["prefill"]["contract"],
                "corpus_sha256": dflash2["prefill"]["corpus_sha256"],
                "cells": prefill_cells,
            },
            "semantic_decode": {
                "case_ids": list(REQUIRED_SEMANTIC_CASE_IDS),
                "repeats": REQUIRED_SEMANTIC_REPEATS,
                "cells": semantic_decode_cells,
            },
            "native_kernel": {
                "weight_source_root": os.fspath(native_model_snapshot),
                "expert_slot_fingerprint": native_kernel["expert_slot_fingerprint"],
                "trellis_bits": native_kernel["trellis_bits"],
                "tp_ranks": native_kernel["tp_ranks"],
                "layer_id": native_kernel["layer_id"],
                "checkpoint_inventory_sha256": native_kernel[
                    "checkpoint_inventory_sha256"
                ],
                "native_library": native_kernel["native_library"],
                "required_rows": native_kernel["required_rows"],
            },
            "dflash2_preflight": {
                "checkpoint_repo_id": dflash_preflight["checkpoint_repo_id"],
                "checkpoint_revision": dflash_preflight["checkpoint_revision"],
                "checkpoint_config_sha256": dflash_preflight[
                    "checkpoint_config_sha256"
                ],
                "checkpoint_weight_lfs_sha256": dflash_preflight[
                    "checkpoint_weight_lfs_sha256"
                ],
                "kv_storage": dflash_preflight["kv_storage"],
                "kv_element_bytes": dflash_preflight["kv_element_bytes"],
                "page_size": dflash_preflight["page_size"],
                "kv_capacity_tokens": dflash_preflight["kv_capacity_tokens"],
                "proposal_tokens_per_request": dflash_preflight[
                    "proposal_tokens_per_request"
                ],
                "topk_backend": dflash_preflight["topk_backend"],
                "resident_preload": dflash_preflight["resident_preload"],
                "target_alias_preload": dflash_preflight["target_alias_preload"],
                "static_graphs": dflash_preflight["static_graphs"],
            },
            "dflash2_topk_tuning": {
                "selected_backend": topk_service_gate["selected_backend"],
                "micro_selected_backend": topk_tuning["selected_backend"],
                "fastest_valid_backend": topk_tuning["fastest_valid_backend"],
                "fastest_valid_speedup_vs_torch": topk_tuning[
                    "fastest_valid_speedup_vs_torch"
                ],
                "aggregate_median_ms": topk_tuning["aggregate_median_ms"],
                "valid_backends": topk_tuning["valid_backends"],
                "full_service_gate": {
                    key: value
                    for key, value in topk_service_gate.items()
                    if key != "evidence"
                },
            },
            "dflash2_fusion_tuning": {
                "selector": {
                    "winning_fused_warps": selector_tuning["winning_fused_warps"],
                    "captured_launches": selector_tuning["captured_launches"],
                },
                "body": {
                    "winning_fused_warps": body_fusion_tuning["winning_fused_warps"],
                    "captured_launches": body_fusion_tuning["captured_launches"],
                },
            },
            "dflash2_adaptive": {
                "cost_profile": {
                    "profile_id": cost_profile["profile_id"],
                    "source_sha256": cost_profile["source_sha256"],
                    "route_qualified_cells": cost_profile["route_qualified_cells"],
                    "corpus_samples": cost_profile["corpus_samples"],
                },
                "reference_width": DFLASH2_REFERENCE_WIDTH,
                "response_performance_score": adaptive_response_score,
                "k5_response_performance_score": k5_response_score,
                "concurrency_geomean_tps": adaptive_concurrency_score,
                "k5_concurrency_geomean_tps": k5_concurrency_score,
                "weighted_decode_ratio_vs_k5": ratio(
                    dflash2["blended"]["wall_decode_tps"],
                    k5_reference["blended"]["wall_decode_tps"],
                    "DFlash2 adaptive/K5 weighted decode",
                ),
            },
        },
    }
    return {**body, "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def revalidate_native_evidence(
    report: dict[str, Any],
    *,
    expected_sparkinfer_revision: str,
    expected_checkpoint_root: Path,
    expected_expert_slot_fingerprint: str,
) -> dict[str, Any]:
    evidence = report.get("evidence")
    results = report.get("results")
    identities = (
        evidence.get("candidate_native_validations")
        if isinstance(evidence, dict)
        else None
    )
    embedded = results.get("native_kernel") if isinstance(results, dict) else None
    if (
        not isinstance(identities, list)
        or len(identities) != 4
        or any(
            not isinstance(identity, dict)
            or identity.get("schema") != NATIVE_VALIDATION_SCHEMA
            or not isinstance(identity.get("path"), str)
            for identity in identities
        )
        or not isinstance(embedded, dict)
    ):
        raise QualificationError("GLM-5.3 qualification has incomplete native evidence")
    native = native_validations(
        [Path(identity["path"]) for identity in identities],
        expected_sparkinfer_revision=expected_sparkinfer_revision,
        expected_checkpoint_root=expected_checkpoint_root,
        expected_expert_slot_fingerprint=expected_expert_slot_fingerprint,
        expected_trellis_bits=4,
        expected_required_rows=K4_REQUIRED_NATIVE_ROWS,
        capacity_rows_for_live_rows=exl3_k4_capacity_rows,
        route_block_rows_for_capacity=exl3_k4_route_block_rows,
        expected_artifact_manifest_sha256=report.get("artifact_manifest_sha256"),
    )
    expected = {
        "weight_source_root": os.fspath(expected_checkpoint_root.resolve()),
        "expert_slot_fingerprint": native["expert_slot_fingerprint"],
        "trellis_bits": native["trellis_bits"],
        "tp_ranks": native["tp_ranks"],
        "layer_id": native["layer_id"],
        "checkpoint_inventory_sha256": native["checkpoint_inventory_sha256"],
        "native_library": native["native_library"],
        "required_rows": native["required_rows"],
    }
    if identities != native["identities"] or embedded != expected:
        raise QualificationError(
            "GLM-5.3 native evidence summary differs from its files"
        )
    return expected


def revalidate_dflash2_width_evidence(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("evidence")
    results = report.get("results")
    thresholds = report.get("thresholds")
    identities = (
        evidence.get("dflash2_width_trials") if isinstance(evidence, dict) else None
    )
    embedded = results.get("dflash2_width_sweep") if isinstance(results, dict) else None
    tool_version = (
        thresholds.get("tool_eval_version") if isinstance(thresholds, dict) else None
    )
    if (
        not isinstance(identities, dict)
        or set(identities) != {str(width) for width in DFLASH2_REQUIRED_WIDTHS}
        or not isinstance(embedded, dict)
        or not isinstance(tool_version, str)
        or not tool_version
    ):
        raise QualificationError("GLM-5.3 qualification has incomplete width evidence")
    trials: list[tuple[int, Path, Path, Path]] = []
    for width in DFLASH2_REQUIRED_WIDTHS:
        trial = identities[str(width)]
        if not isinstance(trial, dict) or set(trial) != {
            "deployment",
            "blended",
            "tool_eval",
        }:
            raise QualificationError(
                "GLM-5.3 qualification has malformed width evidence"
            )
        paths = []
        for kind in ("deployment", "blended", "tool_eval"):
            identity = trial[kind]
            if not isinstance(identity, dict) or not isinstance(
                identity.get("path"), str
            ):
                raise QualificationError(
                    "GLM-5.3 qualification has malformed width evidence identity"
                )
            paths.append(Path(identity["path"]))
        trials.append((width, *paths))
    sweep = dflash2_width_sweep(
        trials,
        expected_tool_eval_version=tool_version,
    )
    expected_embedded = {
        "winner": sweep["winner"],
        "selection": sweep["selection"],
        "trials": sweep["trials"],
    }
    expected_identities = {
        str(trial["width"]): trial["evidence"] for trial in sweep["trials"]
    }
    if embedded != expected_embedded or identities != expected_identities:
        raise QualificationError(
            "GLM-5.3 DFlash2 width evidence summary differs from its files"
        )
    return expected_embedded


def revalidate_dflash2_adaptive_evidence(report: dict[str, Any]) -> dict[str, Any]:
    """Reopen the adaptive cost surface and its matched fixed-K5 reference."""

    evidence = report.get("evidence")
    results = report.get("results")
    if not isinstance(evidence, dict) or not isinstance(results, dict):
        raise QualificationError(
            "GLM-5.3 qualification has incomplete adaptive DFlash2 evidence"
        )
    required = {
        "dflash2_deployment",
        "dflash2_cost_profile",
        "dflash2_k5_deployment",
        "dflash2_k5_blended",
        "dflash2_k5_concurrency",
    }
    if not required <= set(evidence):
        raise QualificationError(
            "GLM-5.3 qualification has incomplete adaptive DFlash2 identities"
        )
    identities = {name: evidence[name] for name in required}
    if any(
        not isinstance(identity, dict) or not isinstance(identity.get("path"), str)
        for name, identity in identities.items()
        if name != "dflash2_k5_concurrency"
    ):
        raise QualificationError(
            "GLM-5.3 qualification has malformed adaptive DFlash2 identities"
        )
    concurrency_identities = identities["dflash2_k5_concurrency"]
    if not isinstance(concurrency_identities, list) or any(
        not isinstance(identity, dict) or not isinstance(identity.get("path"), str)
        for identity in concurrency_identities
    ):
        raise QualificationError(
            "GLM-5.3 qualification has malformed fixed-K5 concurrency identities"
        )

    deployed = deployment(
        Path(identities["dflash2_deployment"]["path"]),
        candidate=True,
        expected_model=GLM53_MODEL_ID,
        expected_speculation=MODE_DFLASH2,
    )
    cost_profile = dflash2_cost_profile(
        Path(identities["dflash2_cost_profile"]["path"]), deployed=deployed
    )
    reference = dflash2_k5_reference(
        deployment_path=Path(identities["dflash2_k5_deployment"]["path"]),
        blended_path=Path(identities["dflash2_k5_blended"]["path"]),
        concurrency_paths=[
            Path(identity["path"]) for identity in concurrency_identities
        ],
    )
    embedded = results.get("dflash2_adaptive")
    mode = results.get("modes", {}).get(MODE_DFLASH2)
    if not isinstance(embedded, dict) or not isinstance(mode, dict):
        raise QualificationError(
            "GLM-5.3 qualification has no adaptive DFlash2 summary"
        )
    adaptive_response_score = math.sqrt(
        finite_positive(
            mode.get("agentic_code_decode_tps"), "adaptive DFlash2 code decode"
        )
        * finite_positive(
            mode.get("weighted_decode_tps"), "adaptive DFlash2 weighted decode"
        )
    )
    adaptive_concurrency_score = finite_positive(
        mode.get("agentic_c1_c2_c4_geomean_tps"),
        "adaptive DFlash2 concurrency geomean",
    )
    k5_response_score = math.sqrt(
        _agentic_code_decode_tps(reference, "DFlash2 K5")
        * reference["blended"]["wall_decode_tps"]
    )
    k5_concurrency_score = _agentic_concurrency_geomean_tps(reference, "DFlash2 K5")
    expected = {
        "cost_profile": {
            "profile_id": cost_profile["profile_id"],
            "source_sha256": cost_profile["source_sha256"],
            "route_qualified_cells": cost_profile["route_qualified_cells"],
            "corpus_samples": cost_profile["corpus_samples"],
        },
        "reference_width": DFLASH2_REFERENCE_WIDTH,
        "response_performance_score": adaptive_response_score,
        "k5_response_performance_score": k5_response_score,
        "concurrency_geomean_tps": adaptive_concurrency_score,
        "k5_concurrency_geomean_tps": k5_concurrency_score,
        "weighted_decode_ratio_vs_k5": ratio(
            finite_positive(
                mode.get("weighted_decode_tps"),
                "adaptive DFlash2 weighted decode",
            ),
            reference["blended"]["wall_decode_tps"],
            "adaptive DFlash2/K5 weighted decode",
        ),
    }
    expected_identities = {
        "dflash2_cost_profile": cost_profile["identity"],
        "dflash2_k5_deployment": reference["deployment"]["identity"],
        "dflash2_k5_blended": reference["blended"]["identity"],
        "dflash2_k5_concurrency": reference["concurrency"]["identities"],
    }
    if embedded != expected or any(
        evidence[name] != identity for name, identity in expected_identities.items()
    ):
        raise QualificationError(
            "GLM-5.3 adaptive DFlash2 summary differs from its files"
        )
    return expected


def revalidate_dflash2_topk_evidence(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("evidence")
    results = report.get("results")
    identity = (
        evidence.get("dflash2_topk_tuning") if isinstance(evidence, dict) else None
    )
    embedded = results.get("dflash2_topk_tuning") if isinstance(results, dict) else None
    if (
        not isinstance(identity, dict)
        or not isinstance(identity.get("path"), str)
        or not isinstance(embedded, dict)
    ):
        raise QualificationError("GLM-5.3 qualification has incomplete top-k evidence")
    measured = dflash2_topk_tuning(Path(identity["path"]))
    required_service_identities = {
        key: evidence.get(key)
        for key in (
            "dflash2_k5_deployment",
            "dflash2_k5_blended",
            "dflash2_topk_candidate_deployment",
            "dflash2_topk_candidate_blended",
        )
    }
    if any(
        not isinstance(value, dict) or not isinstance(value.get("path"), str)
        for value in required_service_identities.values()
    ):
        raise QualificationError(
            "GLM-5.3 qualification has incomplete top-k service evidence"
        )
    service = dflash2_topk_service_gate(
        torch_deployment_path=Path(
            required_service_identities["dflash2_k5_deployment"]["path"]
        ),
        torch_blended_path=Path(
            required_service_identities["dflash2_k5_blended"]["path"]
        ),
        candidate_deployment_path=Path(
            required_service_identities["dflash2_topk_candidate_deployment"]["path"]
        ),
        candidate_blended_path=Path(
            required_service_identities["dflash2_topk_candidate_blended"]["path"]
        ),
        topk_tuning=measured,
    )
    expected = {
        "selected_backend": service["selected_backend"],
        "micro_selected_backend": measured["selected_backend"],
        "fastest_valid_backend": measured["fastest_valid_backend"],
        "fastest_valid_speedup_vs_torch": measured["fastest_valid_speedup_vs_torch"],
        "aggregate_median_ms": measured["aggregate_median_ms"],
        "valid_backends": measured["valid_backends"],
        "full_service_gate": {
            key: value for key, value in service.items() if key != "evidence"
        },
    }
    expected_service_identities = {
        "dflash2_k5_deployment": service["evidence"]["torch_deployment"],
        "dflash2_k5_blended": service["evidence"]["torch_blended"],
        "dflash2_topk_candidate_deployment": service["evidence"][
            "candidate_deployment"
        ],
        "dflash2_topk_candidate_blended": service["evidence"]["candidate_blended"],
    }
    if (
        identity != measured["identity"]
        or embedded != expected
        or required_service_identities != expected_service_identities
    ):
        raise QualificationError(
            "GLM-5.3 DFlash2 top-k evidence summary differs from its file"
        )
    return expected


def revalidate_dflash2_fusion_evidence(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report.get("evidence")
    results = report.get("results")
    embedded = (
        results.get("dflash2_fusion_tuning") if isinstance(results, dict) else None
    )
    if not isinstance(evidence, dict) or not isinstance(embedded, dict):
        raise QualificationError("GLM-5.3 qualification has incomplete fusion evidence")
    expected: dict[str, Any] = {}
    for kind, evidence_key in (
        ("selector", "dflash2_selector_tuning"),
        ("body", "dflash2_body_fusion_tuning"),
    ):
        identity = evidence.get(evidence_key)
        if not isinstance(identity, dict) or not isinstance(identity.get("path"), str):
            raise QualificationError(
                "GLM-5.3 qualification has malformed fusion evidence"
            )
        measured = dflash2_fusion_tuning(Path(identity["path"]), kind=kind)
        if identity != measured["identity"]:
            raise QualificationError(
                "GLM-5.3 DFlash2 fusion evidence identity differs from its file"
            )
        expected[kind] = {
            "winning_fused_warps": measured["winning_fused_warps"],
            "captured_launches": measured["captured_launches"],
        }
    if embedded != expected:
        raise QualificationError(
            "GLM-5.3 DFlash2 fusion evidence summary differs from its files"
        )
    return expected


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--native-model-snapshot", type=Path, required=True)
    parser.add_argument("--artifact-validation", type=Path, required=True)
    parser.add_argument("--quant-evidence", type=Path, required=True)
    for mode in ("native", "dflash2"):
        parser.add_argument(f"--{mode}-blended", type=Path, required=True)
        parser.add_argument(f"--{mode}-repeat", type=Path, required=True)
        parser.add_argument(f"--{mode}-startup", type=Path, required=True)
        parser.add_argument(f"--{mode}-deployment", type=Path, required=True)
    parser.add_argument("--dflash2-prefill", type=Path, required=True)
    parser.add_argument("--dflash2-tool-eval", type=Path, required=True)
    for mode in ("dflash2",):
        parser.add_argument(
            f"--{mode}-concurrency",
            type=Path,
            action="append",
            required=True,
            help="one code-fixture concurrency report; repeat for C1, C2, and C4",
        )
        parser.add_argument(f"--{mode}-needle", type=Path, required=True)
    parser.add_argument(
        "--native-validation",
        type=Path,
        action="append",
        required=True,
        help="one K4 native-parity report; repeat exactly four times for TP ranks 0..3",
    )
    parser.add_argument("--dflash2-preflight", type=Path, required=True)
    parser.add_argument("--dflash2-topk-tuning", type=Path, required=True)
    parser.add_argument("--dflash2-selector-tuning", type=Path, required=True)
    parser.add_argument("--dflash2-body-fusion-tuning", type=Path, required=True)
    parser.add_argument("--dflash2-cost-profile", type=Path, required=True)
    parser.add_argument("--dflash2-k5-deployment", type=Path, required=True)
    parser.add_argument("--dflash2-k5-blended", type=Path, required=True)
    parser.add_argument("--dflash2-topk-candidate-deployment", type=Path, required=True)
    parser.add_argument("--dflash2-topk-candidate-blended", type=Path, required=True)
    parser.add_argument(
        "--dflash2-k5-concurrency",
        type=Path,
        action="append",
        required=True,
        help="one fixed-K5 code concurrency report; repeat for C1, C2, and C4",
    )
    parser.add_argument("--expected-default", choices=("auto", *MODES), default="auto")
    parser.add_argument("--tool-eval-version", default=TOOL_EVAL_VERSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = qualify(
        artifact_path=args.artifact,
        native_model_snapshot_path=args.native_model_snapshot,
        artifact_validation_path=args.artifact_validation,
        quant_evidence_path=args.quant_evidence,
        native_blended_path=args.native_blended,
        dflash2_blended_path=args.dflash2_blended,
        native_repeat_path=args.native_repeat,
        dflash2_repeat_path=args.dflash2_repeat,
        dflash2_prefill_path=args.dflash2_prefill,
        dflash2_tool_eval_path=args.dflash2_tool_eval,
        native_startup_path=args.native_startup,
        dflash2_startup_path=args.dflash2_startup,
        native_deployment_path=args.native_deployment,
        dflash2_deployment_path=args.dflash2_deployment,
        dflash2_concurrency_paths=args.dflash2_concurrency,
        dflash2_needle_path=args.dflash2_needle,
        dflash2_preflight_path=args.dflash2_preflight,
        dflash2_topk_tuning_path=args.dflash2_topk_tuning,
        dflash2_selector_tuning_path=args.dflash2_selector_tuning,
        dflash2_body_fusion_tuning_path=args.dflash2_body_fusion_tuning,
        dflash2_cost_profile_path=args.dflash2_cost_profile,
        dflash2_k5_deployment_path=args.dflash2_k5_deployment,
        dflash2_k5_blended_path=args.dflash2_k5_blended,
        dflash2_k5_concurrency_paths=args.dflash2_k5_concurrency,
        dflash2_topk_candidate_deployment_path=args.dflash2_topk_candidate_deployment,
        dflash2_topk_candidate_blended_path=args.dflash2_topk_candidate_blended,
        native_validation_paths=args.native_validation,
        expected_default=args.expected_default,
        expected_tool_eval_version=args.tool_eval_version,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
