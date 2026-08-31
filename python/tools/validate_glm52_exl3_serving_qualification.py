#!/usr/bin/env python3
"""Pair and gate final NVFP4-versus-EXL3 serving qualification evidence.

The quantizer and artifact validators prove that the checkpoint is internally
consistent.  This tool supplies the deliberately separate end-to-end gate: it
requires byte-identified, prompt-matched decode, repetition, prefill, and tool
evaluation runs and binds their results to the exact accepted artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
from typing import Any, Callable

from _b12x_exl3_k3_profile import (
    exl3_k3_capacity_rows,
    exl3_k3_route_block_rows,
)

from stage_glm52_exl3_hf_snapshot import (
    MODEL_ID,
    _quant_evidence,
    _validation_evidence,
)
from validate_glm52_exl3_artifact import (
    ARTIFACT_SCHEMA,
    ArtifactValidationError,
    _json_object,
)


SCHEMA = "glmrt-glm52-exl3-serving-qualification-v1"
STARTUP_SCHEMA = "glmrt-glm52-expert-startup-v2"
DEPLOYMENT_SCHEMA = "glmrt-wip-deployment-evidence-v2"
DFLASH2_MODEL_ID = "incoai/GLM-5.3-DFlash2"
DFLASH2_REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
NATIVE_VALIDATION_SCHEMA = "glmrt-b12x-exl3-native-validation-v1"
NATIVE_VALIDATOR_SOURCE = Path(__file__).resolve().with_name(
    "validate_b12x_exl3_native.py"
)
TOOL_EVAL_VERSION = "2.3.2.dev3+g5df1e9e0c"
QUALITY_CONTRACT_VERSION = "glmrt-semantic-decode-contract-v2"
WEIGHTED_QUALITY_CONTRACT_VERSION = "glmrt-semantic-decode-contract-v3"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
BASELINE_MODELS = {
    "lukealonso/GLM-5.2-NVFP4",
    "lukealonso/GLM-5.2-NVFP4-full",
    "nvidia/GLM-5.2-NVFP4",
    "nvidia/GLM-5.2-NVFP4-full",
}
REQUIRED_GATES = frozenset(
    {
        "blended_decode",
        "blended_acceptance",
        "repeat_decode",
        "prefill_every_cell",
        "tool_eval_points",
        "expert_resident_preload",
        "expert_startup",
        "native_kernel_parity",
    }
)
REQUIRED_NATIVE_ROWS = frozenset({1, 3, 9, 10, 129, 257, 513, 1025, 2049, 2064})


class QualificationError(RuntimeError):
    """An evidence file is malformed, stale, or not a matched comparison."""


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


def evidence_identity(path: Path, schema: str) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise QualificationError(f"evidence is not one regular file: {resolved}")
    return {
        "path": os.fspath(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": hash_file(resolved),
        "schema": schema,
    }


def read_jsonl(path: Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved = path.expanduser().resolve(strict=True)
    if resolved.is_symlink() or not resolved.is_file():
        raise QualificationError(f"JSONL evidence is not one regular file: {resolved}")
    records: list[dict[str, Any]] = []
    with resolved.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise QualificationError(
                    f"{resolved} contains a blank JSONL line at {line_number}"
                )
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise QualificationError(
                    f"{resolved} contains invalid JSON at line {line_number}"
                ) from error
            if not isinstance(record, dict):
                raise QualificationError(
                    f"{resolved} line {line_number} is not a JSON object"
                )
            records.append(record)
    if not records:
        raise QualificationError(f"JSONL evidence is empty: {resolved}")
    return resolved, records


def finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{field} is not numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0.0:
        raise QualificationError(f"{field} is not finite and positive")
    return normalized


def finite_nonnegative(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{field} is not numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise QualificationError(f"{field} is not finite and nonnegative")
    return normalized


def integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualificationError(f"{field} is not an integer >= {minimum}")
    return value


def require_close(actual: Any, expected: float, field: str) -> float:
    normalized = finite_nonnegative(actual, field)
    if not math.isclose(normalized, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise QualificationError(
            f"{field} differs from measurements: {normalized} != {expected}"
        )
    return normalized


def sha256_canonical(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def require_model(
    model: Any,
    *,
    candidate: bool,
    field: str,
    expected_model: str | None = None,
) -> str:
    if not isinstance(model, str):
        raise QualificationError(f"{field} has no model identity")
    if expected_model is not None:
        if model != expected_model:
            raise QualificationError(
                f"{field} model is {model!r}, not {expected_model!r}"
            )
        return model
    if candidate and model != MODEL_ID:
        raise QualificationError(f"{field} candidate model is {model!r}, not {MODEL_ID}")
    if not candidate and model not in BASELINE_MODELS:
        raise QualificationError(f"{field} is not a supported NVFP4 baseline: {model!r}")
    return model


def blended(
    path: Path, *, candidate: bool, expected_model: str | None = None
) -> dict[str, Any]:
    resolved, records = read_jsonl(path)
    aggregates = [record.get("aggregate") for record in records if "aggregate" in record]
    if len(aggregates) != 1 or not isinstance(aggregates[0], dict):
        raise QualificationError(f"{resolved} must contain one blended aggregate")
    aggregate = aggregates[0]
    aggregate_schema = aggregate.get("schema")
    if aggregate_schema not in {
        "glmrt-mtp-acceptance-aggregate-v3",
        "glmrt-mtp-acceptance-aggregate-v4",
    }:
        raise QualificationError(f"{resolved} does not use the prompt-bound blended schema")
    weighted_contract = aggregate_schema == "glmrt-mtp-acceptance-aggregate-v4"
    quality_contract_version = (
        WEIGHTED_QUALITY_CONTRACT_VERSION
        if weighted_contract
        else QUALITY_CONTRACT_VERSION
    )
    require_model(
        aggregate.get("model"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    measurements = [record for record in records if "aggregate" not in record]
    if len(measurements) != aggregate.get("cases") or not measurements:
        raise QualificationError(f"{resolved} blended measurement count differs")
    selected_case_ids = aggregate.get("selected_case_ids")
    cases_per_repeat = integer(
        aggregate.get("cases_per_repeat"),
        f"{resolved}: blended cases_per_repeat",
        minimum=1,
    )
    corpus_repeats = integer(
        aggregate.get("corpus_repeats"),
        f"{resolved}: blended corpus_repeats",
        minimum=1,
    )
    if (
        not isinstance(selected_case_ids, list)
        or len(selected_case_ids) != cases_per_repeat
        or any(not isinstance(case_id, str) or not case_id for case_id in selected_case_ids)
        or len(set(selected_case_ids)) != len(selected_case_ids)
        or len(measurements) != cases_per_repeat * corpus_repeats
    ):
        raise QualificationError(f"{resolved} has an invalid blended case schedule")
    raw_case_weights = aggregate.get("case_weights")
    if weighted_contract:
        if (
            not isinstance(raw_case_weights, dict)
            or set(raw_case_weights) != set(selected_case_ids)
            or any(
                isinstance(raw_case_weights.get(case_id), bool)
                or not isinstance(raw_case_weights.get(case_id), (int, float))
                or not math.isfinite(float(raw_case_weights[case_id]))
                or float(raw_case_weights[case_id]) <= 0.0
                for case_id in selected_case_ids
            )
        ):
            raise QualificationError(f"{resolved} has invalid blended case weights")
        case_weights = {
            case_id: float(raw_case_weights[case_id]) for case_id in selected_case_ids
        }
    else:
        if raw_case_weights is not None:
            raise QualificationError(f"{resolved} legacy blended evidence has case weights")
        case_weights = {case_id: 1.0 for case_id in selected_case_ids}
    prompt_contract = aggregate.get("prompt_contract")
    contract = aggregate.get("prompt_contract_sha256")
    if (
        not isinstance(prompt_contract, dict)
        or not isinstance(contract, str)
        or not SHA256_RE.fullmatch(contract)
        or sha256_canonical(prompt_contract) != contract
        or prompt_contract.get("repeats") != corpus_repeats
        or prompt_contract.get("nonce_seed") != aggregate.get("nonce_seed")
        or prompt_contract.get("quality_contract_version") != quality_contract_version
        or [
            case.get("id") if isinstance(case, dict) else None
            for case in prompt_contract.get("cases", [])
        ]
        != selected_case_ids
        or (
            weighted_contract
            and any(
                not isinstance(case, dict)
                or case.get("weight") != case_weights[case_id]
                or "response_format" not in case
                for case_id, case in zip(
                    selected_case_ids,
                    prompt_contract.get("cases", []),
                    strict=True,
                )
            )
        )
    ):
        raise QualificationError(f"{resolved} has an invalid blended prompt contract")
    prompts: list[dict[str, Any]] = []
    total_timed_tokens = 0.0
    total_decode_ms = 0.0
    total_drafts = 0
    total_accepted = 0
    repeat_wall_tps: list[float] = []
    quality_contract_failures: list[dict[str, Any]] = []
    for index, record in enumerate(measurements):
        expected_case = selected_case_ids[index % cases_per_repeat]
        expected_repeat = index // cases_per_repeat + 1
        prompt_sha256 = record.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not SHA256_RE.fullmatch(prompt_sha256):
            raise QualificationError(f"{resolved} has an unbound blended prompt")
        if record.get("case") != expected_case or record.get("repeat") != expected_repeat:
            raise QualificationError(f"{resolved} blended case schedule differs")
        if integer(
            record.get("runtime_captures"),
            f"{resolved}: blended runtime captures",
        ) != 0:
            raise QualificationError(f"{resolved} captured a runtime graph")
        quality_passed = record.get("quality_contract_passed")
        quality_issues = record.get("quality_contract_issues")
        if (
            record.get("quality_contract_version") != quality_contract_version
            or not isinstance(quality_passed, bool)
            or not isinstance(quality_issues, list)
            or any(not isinstance(issue, str) or not issue for issue in quality_issues)
            or quality_passed != (not quality_issues)
        ):
            raise QualificationError(f"{resolved} has malformed semantic output evidence")
        if not quality_passed:
            failure = {
                "case": record.get("case"),
                "repeat": record.get("repeat"),
                "issues": quality_issues,
            }
            quality_contract_failures.append(failure)
            if candidate:
                raise QualificationError(f"{resolved} failed a semantic output contract")
        completion_tokens = integer(
            record.get("completion_tokens"),
            f"{resolved}: blended completion_tokens",
            minimum=2,
        )
        integer(
            record.get("content_chars"),
            f"{resolved}: blended content_chars",
            minimum=1,
        )
        decode_ms = finite_positive(
            record.get("decode_ms"), f"{resolved}: blended decode_ms"
        )
        drafts = integer(
            record.get("draft_tokens"), f"{resolved}: blended draft_tokens"
        )
        accepted = integer(
            record.get("accepted_draft_tokens"),
            f"{resolved}: blended accepted_draft_tokens",
        )
        if accepted > drafts:
            raise QualificationError(f"{resolved} accepted more drafts than it issued")
        require_close(
            record.get("accepted_draft_rate"),
            accepted / drafts if drafts else 0.0,
            f"{resolved}: blended per-case acceptance",
        )
        if completion_tokens < 1:
            raise QualificationError(f"{resolved} has an empty blended generation")
        weight = case_weights[expected_case]
        total_timed_tokens += weight * (completion_tokens - 1)
        total_decode_ms += weight * decode_ms
        total_drafts += drafts
        total_accepted += accepted
        prompts.append(
            {
                "case": record.get("case"),
                "repeat": record.get("repeat"),
                "prompt_sha256": prompt_sha256,
                "prompt": record.get("prompt"),
                "request_sha256": record.get("request_sha256"),
                "nonce": record.get("nonce"),
            }
        )
        if (index + 1) % cases_per_repeat == 0:
            repeat_records = measurements[index + 1 - cases_per_repeat : index + 1]
            repeat_tokens = sum(
                case_weights[str(item.get("case"))]
                * (
                    integer(
                        item.get("completion_tokens"),
                        f"{resolved}: repeat completion_tokens",
                        minimum=2,
                    )
                    - 1
                )
                for item in repeat_records
            )
            repeat_ms = sum(
                case_weights[str(item.get("case"))]
                * finite_positive(
                    item.get("decode_ms"), f"{resolved}: repeat decode_ms"
                )
                for item in repeat_records
            )
            repeat_wall_tps.append(repeat_tokens * 1_000.0 / repeat_ms)
    if (
        aggregate.get("quality_contract_version") != quality_contract_version
        or aggregate.get("all_quality_contracts_passed")
        != (not quality_contract_failures)
        or aggregate.get("quality_contract_failures") != quality_contract_failures
        or aggregate.get("all_zero_runtime_captures") is not True
    ):
        raise QualificationError(f"{resolved} aggregate failed semantic output contracts")
    raw_repeat_summaries = aggregate.get("repeat_summaries")
    if not isinstance(raw_repeat_summaries, list) or len(raw_repeat_summaries) != corpus_repeats:
        raise QualificationError(f"{resolved} has invalid blended repeat summaries")
    for repeat_index, (raw, expected_tps) in enumerate(
        zip(raw_repeat_summaries, repeat_wall_tps, strict=True), 1
    ):
        if not isinstance(raw, dict) or raw.get("repeat") != repeat_index:
            raise QualificationError(f"{resolved} has invalid blended repeat summaries")
        require_close(
            raw.get("wall_decode_tps"),
            expected_tps,
            f"{resolved}: repeat {repeat_index} wall decode TPS",
        )
    expected_wall_tps = total_timed_tokens * 1_000.0 / total_decode_ms
    wall_decode_tps = require_close(
        aggregate.get("wall_decode_tps"),
        expected_wall_tps,
        f"{resolved}: wall decode TPS",
    )
    median_repeat_wall_decode_tps = require_close(
        aggregate.get("median_repeat_wall_decode_tps"),
        statistics.median(repeat_wall_tps),
        f"{resolved}: median repeat wall decode TPS",
    )
    accepted_draft_rate = require_close(
        aggregate.get("accepted_draft_rate"),
        total_accepted / total_drafts if total_drafts else 0.0,
        f"{resolved}: aggregate acceptance",
    )
    if total_drafts <= 0 or not 0.0 < accepted_draft_rate <= 1.0:
        raise QualificationError(f"{resolved} has no usable draft acceptance evidence")
    case_results = []
    contract_cases = prompt_contract["cases"]
    for case_index, case_id in enumerate(selected_case_ids):
        case_records = measurements[case_index::cases_per_repeat]
        contract_case = contract_cases[case_index]
        category = contract_case.get("category") if isinstance(contract_case, dict) else None
        if (
            len(case_records) != corpus_repeats
            or not isinstance(category, str)
            or not category
            or any(
                record.get("category", category) != category
                for record in case_records
            )
        ):
            raise QualificationError(f"{resolved} has inconsistent per-case evidence")
        case_tokens = sum(record["completion_tokens"] - 1 for record in case_records)
        case_decode_ms = sum(float(record["decode_ms"]) for record in case_records)
        case_drafts = sum(record["draft_tokens"] for record in case_records)
        case_accepted = sum(record["accepted_draft_tokens"] for record in case_records)
        case_results.append(
            {
                "case": case_id,
                "category": category,
                "samples": len(case_records),
                "timed_tokens": case_tokens,
                "decode_ms": case_decode_ms,
                "decode_tps": case_tokens * 1_000.0 / case_decode_ms,
                "draft_tokens": case_drafts,
                "accepted_draft_tokens": case_accepted,
                "accepted_draft_rate": (
                    case_accepted / case_drafts if case_drafts else 0.0
                ),
            }
        )
    return {
        "identity": evidence_identity(
            resolved,
            "glmrt-mtp-acceptance-jsonl-v4"
            if weighted_contract
            else "glmrt-mtp-acceptance-jsonl-v3",
        ),
        "model": aggregate["model"],
        "contract": contract,
        "prompt_contract": prompt_contract,
        "case_weights": case_weights,
        "prompts": prompts,
        "wall_decode_tps": wall_decode_tps,
        "median_repeat_wall_decode_tps": median_repeat_wall_decode_tps,
        "accepted_draft_rate": accepted_draft_rate,
        "cases": len(measurements),
        "case_results": case_results,
        "all_quality_contracts_passed": not quality_contract_failures,
        "quality_contract_failures": quality_contract_failures,
    }


def repeat_decode(
    path: Path, *, candidate: bool, expected_model: str | None = None
) -> dict[str, Any]:
    resolved, records = read_jsonl(path)
    meta = [record for record in records if record.get("record") == "meta"]
    summaries = [record for record in records if record.get("record") == "summary"]
    measurements = [
        record for record in records if record.get("record") == "measurement"
    ]
    if len(meta) != 1 or len(summaries) != 1 or not measurements:
        raise QualificationError(f"{resolved} has an invalid repeat-decode record set")
    metadata = meta[0]
    summary = summaries[0]
    if metadata.get("schema") != "glmrt-repeat-decode-v2":
        raise QualificationError(f"{resolved} does not use the prompt-bound repeat schema")
    require_model(
        metadata.get("model"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    contract = metadata.get("prompt_contract_sha256")
    tokenizer = metadata.get("tokenizer_sha256")
    warmups = integer(
        metadata.get("warmups"), f"{resolved}: repeat warmups"
    )
    repeats = integer(
        metadata.get("repeats"), f"{resolved}: repeat repeats", minimum=1
    )
    requested_repetitions = integer(
        metadata.get("requested_repetitions"),
        f"{resolved}: requested repetitions",
        minimum=1,
    )
    requested_max_tokens = integer(
        metadata.get("requested_max_tokens"),
        f"{resolved}: requested max tokens",
        minimum=1,
    )
    word = metadata.get("word")
    nonce_seed = metadata.get("nonce_seed")
    if (
        not all(
            isinstance(value, str) and SHA256_RE.fullmatch(value)
            for value in (contract, tokenizer)
        )
        or not isinstance(word, str)
        or not word.strip()
        or isinstance(nonce_seed, bool)
        or not isinstance(nonce_seed, int)
    ):
        raise QualificationError(f"{resolved} has no repeat prompt/tokenizer identity")
    expected_contract = {
        "word": word,
        "requested_repetitions": requested_repetitions,
        "requested_max_tokens": requested_max_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "nonce_seed": nonce_seed,
        "temperature": 0,
        "enable_thinking": False,
        "tokenizer_sha256": tokenizer,
    }
    if sha256_canonical(expected_contract) != contract:
        raise QualificationError(f"{resolved} repeat prompt contract is inconsistent")
    if len(measurements) != warmups + repeats:
        raise QualificationError(f"{resolved} repeat measurement count differs")
    prompts: list[dict[str, Any]] = []
    timed_measurements: list[dict[str, Any]] = []
    for sample, record in enumerate(measurements):
        prompt_sha256 = record.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not SHA256_RE.fullmatch(prompt_sha256):
            raise QualificationError(f"{resolved} has an unbound repeat prompt")
        expected_timed = sample >= warmups
        if record.get("sample") != sample or record.get("timed") is not expected_timed:
            raise QualificationError(f"{resolved} repeat sample schedule differs")
        if (
            record.get("word") != word
            or record.get("requested_repetitions") != requested_repetitions
            or record.get("requested_max_tokens") != requested_max_tokens
        ):
            raise QualificationError(f"{resolved} repeat request contract differs")
        if integer(
            record.get("runtime_captures"),
            f"{resolved}: repeat runtime captures",
        ) != 0:
            raise QualificationError(f"{resolved} captured a runtime graph")
        integer(
            record.get("completion_tokens"),
            f"{resolved}: repeat completion tokens",
            minimum=2,
        )
        finite_positive(record.get("decode_ms"), f"{resolved}: repeat decode ms")
        observed_occurrences = integer(
            record.get("observed_word_occurrences"),
            f"{resolved}: observed word repetitions",
            minimum=1,
        )
        exact_repetition = observed_occurrences == requested_repetitions
        if record.get("exact_repetition_count") is not exact_repetition:
            raise QualificationError(
                f"{resolved} has an inconsistent exact-repetition flag"
            )
        # This is the low-entropy decode benchmark commonly published for a
        # "repeat word N times" prompt, not a counting-quality evaluation.
        # GLM-5.2 itself is not exact on this prompt (matched NVFP4 runs also
        # overshoot), so retain exactness diagnostically while requiring the
        # response to remain a bounded repetition workload.
        repetition_ratio = observed_occurrences / requested_repetitions
        if not 0.8 <= repetition_ratio <= 1.25:
            raise QualificationError(
                f"{resolved} did not produce a bounded word-repetition workload"
            )
        if expected_timed:
            timed_measurements.append(record)
        prompts.append(
            {
                "sample": record.get("sample"),
                "timed": record.get("timed"),
                "prompt_sha256": prompt_sha256,
            }
        )
    total_tokens = sum(
        integer(
            record.get("completion_tokens"),
            f"{resolved}: timed repeat completion tokens",
            minimum=2,
        )
        - 1
        for record in timed_measurements
    )
    total_decode_ms = sum(
        finite_positive(
            record.get("decode_ms"), f"{resolved}: timed repeat decode ms"
        )
        for record in timed_measurements
    )
    aggregate_decode_tps = require_close(
        summary.get("aggregate_decode_tps"),
        total_tokens * 1_000.0 / total_decode_ms,
        f"{resolved}: repeat decode TPS",
    )
    if (
        summary.get("timed_samples") != repeats
        or summary.get("requested_completion_tokens") != requested_max_tokens
        or summary.get("actual_completion_tokens")
        != [record["completion_tokens"] for record in timed_measurements]
        or summary.get("observed_word_occurrences")
        != [record["observed_word_occurrences"] for record in timed_measurements]
        or summary.get("all_zero_runtime_captures") is not True
        or summary.get("all_exact_repetition_count")
        is not all(
            bool(record["exact_repetition_count"])
            for record in timed_measurements
        )
    ):
        raise QualificationError(f"{resolved} repeat summary differs from measurements")
    return {
        "identity": evidence_identity(resolved, "glmrt-repeat-decode-jsonl-v2"),
        "model": metadata["model"],
        "contract": contract,
        "tokenizer_sha256": tokenizer,
        "prompts": prompts,
        "aggregate_decode_tps": aggregate_decode_tps,
        "all_exact_repetition_count": bool(
            summary.get("all_exact_repetition_count")
        ),
        "observed_word_occurrences": [
            record["observed_word_occurrences"] for record in timed_measurements
        ],
        "timed_samples": repeats,
    }


def prefill(
    path: Path, *, candidate: bool, expected_model: str | None = None
) -> dict[str, Any]:
    resolved, records = read_jsonl(path)
    summaries = [
        record
        for record in records
        if record.get("schema") == "glmrt-release-prefill-summary-v3"
    ]
    measurements = [
        record
        for record in records
        if record.get("schema") == "glmrt-release-prefill-v2"
    ]
    if len(summaries) != 1 or len(measurements) + 1 != len(records):
        raise QualificationError(f"{resolved} has an invalid prefill record set")
    summary = summaries[0]
    require_model(
        summary.get("model"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    contract = summary.get("prompt_contract_sha256")
    corpus = summary.get("corpus_sha256")
    tokenizer = summary.get("tokenizer_sha256")
    profile = summary.get("profile")
    run_id = summary.get("run_id")
    if not all(
        isinstance(value, str) and SHA256_RE.fullmatch(value)
        for value in (contract, corpus, tokenizer)
    ) or not all(isinstance(value, str) and value for value in (profile, run_id)):
        raise QualificationError(f"{resolved} has incomplete prefill identities")
    prompts: list[dict[str, Any]] = []
    for record in measurements:
        prompt_sha256 = record.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not SHA256_RE.fullmatch(prompt_sha256):
            raise QualificationError(f"{resolved} has an unbound prefill prompt")
        base_context_tokens = integer(
            record.get("base_context_tokens"),
            f"{resolved}: prefill base context",
        )
        suffix_tokens = integer(
            record.get("suffix_tokens"), f"{resolved}: prefill suffix", minimum=1
        )
        repeat = integer(
            record.get("repeat"), f"{resolved}: prefill repeat", minimum=1
        )
        prompt_tokens = integer(
            record.get("prompt_tokens"),
            f"{resolved}: prefill prompt tokens",
            minimum=1,
        )
        cached_prompt_tokens = integer(
            record.get("cached_prompt_tokens"),
            f"{resolved}: cached prefill prefix",
        )
        if (
            record.get("model") != summary["model"]
            or record.get("profile") != profile
            or record.get("run_id") != run_id
            or record.get("corpus_sha256") != corpus
            or record.get("tokenizer_sha256") != tokenizer
            or integer(
                record.get("runtime_captures"),
                f"{resolved}: prefill runtime captures",
            )
            != 0
            or record.get("numeric_progression_passed") is not True
            or record.get("attention_complete") is not True
            or record.get("prefill_rows") != suffix_tokens
            or prompt_tokens - cached_prompt_tokens - 1 != suffix_tokens
            or (
                base_context_tokens > 0
                and cached_prompt_tokens < base_context_tokens
            )
        ):
            raise QualificationError(f"{resolved} failed prefill runtime correctness")
        finite_positive(record.get("prefill_ms"), f"{resolved}: prefill ms")
        finite_positive(record.get("prefill_tps"), f"{resolved}: prefill TPS")
        prompts.append(
            {
                "base_context_tokens": base_context_tokens,
                "suffix_tokens": suffix_tokens,
                "repeat": repeat,
                "prompt_sha256": prompt_sha256,
            }
        )
    if sha256_canonical(prompts) != contract or len(set(
        (prompt["base_context_tokens"], prompt["suffix_tokens"], prompt["repeat"])
        for prompt in prompts
    )) != len(prompts):
        raise QualificationError(f"{resolved} prefill prompt contract is inconsistent")
    cells: dict[tuple[int, int], float] = {}
    for raw in summary.get("cells") or []:
        if not isinstance(raw, dict):
            raise QualificationError(f"{resolved} has an invalid prefill cell")
        key = (raw.get("base_context_tokens"), raw.get("suffix_tokens"))
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in key):
            raise QualificationError(f"{resolved} has an invalid prefill cell key")
        if key in cells:
            raise QualificationError(f"{resolved} duplicates a prefill cell")
        cell_measurements = [
            record
            for record in measurements
            if (
                record.get("base_context_tokens"),
                record.get("suffix_tokens"),
            )
            == key
        ]
        if not cell_measurements:
            raise QualificationError(f"{resolved} prefill cell {key} has no measurements")
        expected_tps = [float(record["prefill_tps"]) for record in cell_measurements]
        expected_ms = [float(record["prefill_ms"]) for record in cell_measurements]
        if raw.get("samples") != len(cell_measurements):
            raise QualificationError(f"{resolved} prefill cell {key} sample count differs")
        require_close(
            raw.get("median_prefill_ms"),
            statistics.median(expected_ms),
            f"{resolved}: prefill cell {key} median ms",
        )
        median_tps = require_close(
            raw.get("median_prefill_tps"),
            statistics.median(expected_tps),
            f"{resolved}: prefill cell {key} median TPS",
        )
        require_close(
            raw.get("min_prefill_tps"),
            min(expected_tps),
            f"{resolved}: prefill cell {key} minimum TPS",
        )
        require_close(
            raw.get("max_prefill_tps"),
            max(expected_tps),
            f"{resolved}: prefill cell {key} maximum TPS",
        )
        cells[key] = median_tps
    if not cells:
        raise QualificationError(f"{resolved} has no prefill cells")
    measured_cells = {
        (record["base_context_tokens"], record["suffix_tokens"])
        for record in measurements
    }
    if set(cells) != measured_cells:
        raise QualificationError(f"{resolved} prefill cell coverage differs")
    return {
        "identity": evidence_identity(resolved, "glmrt-release-prefill-jsonl-v3"),
        "model": summary["model"],
        "profile": profile,
        "run_id": run_id,
        "contract": contract,
        "corpus_sha256": corpus,
        "tokenizer_sha256": tokenizer,
        "prompts": prompts,
        "cells": cells,
    }


def tool_eval(
    path: Path,
    *,
    candidate: bool,
    expected_version: str,
    expected_model: str | None = None,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    report = _json_object(resolved)
    config = report.get("config")
    scores = report.get("scores")
    metadata = report.get("metadata")
    if (
        report.get("schema_version") != "1"
        or report.get("status") != "completed"
        or report.get("tool_eval_bench_version") != expected_version
        or not isinstance(config, dict)
        or not isinstance(scores, dict)
        or not isinstance(metadata, dict)
        or metadata.get("tool_version") != expected_version
    ):
        raise QualificationError(f"{resolved} is not a completed matched tool evaluation")
    require_model(
        config.get("model"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    require_model(
        metadata.get("model"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    scenario_results = scores.get("scenario_results")
    if not isinstance(scenario_results, list) or not scenario_results:
        raise QualificationError(f"{resolved} has no tool-evaluation scenarios")
    scenarios: list[tuple[str, int, str]] = []
    expected_points_by_status = {"pass": 2, "partial": 1, "fail": 0}
    for result in scenario_results:
        if not isinstance(result, dict):
            raise QualificationError(f"{resolved} has an invalid tool scenario")
        scenario_id = result.get("scenario_id")
        points = result.get("points")
        status = result.get("status")
        if (
            not isinstance(scenario_id, str)
            or isinstance(points, bool)
            or not isinstance(points, int)
            or status not in expected_points_by_status
            or points != expected_points_by_status[status]
        ):
            raise QualificationError(f"{resolved} has an incomplete tool scenario")
        scenarios.append((scenario_id, points, status))
    total_points = scores.get("total_points")
    maximum_points = scores.get("max_points")
    if (
        isinstance(total_points, bool)
        or not isinstance(total_points, int)
        or isinstance(maximum_points, bool)
        or not isinstance(maximum_points, int)
        or maximum_points <= 0
        or not 0 <= total_points <= maximum_points
        or len({scenario[0] for scenario in scenarios}) != len(scenarios)
        or config.get("scenario_count") != len(scenarios)
        or config.get("scenario_ids") != [scenario[0] for scenario in scenarios]
        or maximum_points != len(scenarios) * 2
        or total_points != sum(scenario[1] for scenario in scenarios)
        or report.get("final_score")
        != round((total_points / maximum_points) * 100)
        or scores.get("excluded_scenarios", []) != []
        or scores.get("completion_rate", 100.0) != 100.0
        or config.get("error_rate") != 0.0
        or config.get("temperature") != 0.0
    ):
        raise QualificationError(f"{resolved} has invalid tool-evaluation totals")
    return {
        "identity": evidence_identity(resolved, "tool-eval-bench-json-v1"),
        "model": config["model"],
        "version": expected_version,
        "config": config,
        "metadata": metadata,
        "scenario_ids": [scenario[0] for scenario in scenarios],
        "scenarios": scenarios,
        "total_points": total_points,
        "maximum_points": maximum_points,
        "final_score": report.get("final_score"),
        "completion_rate": scores.get("completion_rate"),
    }


def startup(
    path: Path,
    *,
    candidate: bool,
    expected_model: str | None = None,
    expected_weight_format: str | None = None,
    expected_preload_modes: set[str] | None = None,
    expected_include_mtp: bool = False,
    expected_schema: str = STARTUP_SCHEMA,
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    report = _json_object(resolved)
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    hosts = report.get("hosts")
    summary = report.get("summary")
    expected_format = expected_weight_format or ("exl3" if candidate else "nvfp4")
    accepted_preload_modes = expected_preload_modes or (
        {"direct-resident", "cooperative-coalesced"}
        if candidate
        else {"nvfp4-production"}
    )
    expert_runtime_fingerprint = report.get("expert_runtime_fingerprint")
    if (
        report.get("schema") != expected_schema
        or report.get("status") != "accepted"
        or report.get("weight_format") != expected_format
        or report.get("preload_mode") not in accepted_preload_modes
        or SHA256_RE.fullmatch(str(expert_runtime_fingerprint or "")) is None
        or report.get("cache_state") not in {"cold", "warm"}
        or report.get("include_mtp") is not expected_include_mtp
        or not isinstance(digest, str)
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
        or not isinstance(hosts, list)
        or len(hosts) != 4
        or not isinstance(summary, dict)
    ):
        raise QualificationError(f"{resolved} is not accepted expert-startup evidence")
    require_model(
        report.get("model"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    host_names = [host.get("host") if isinstance(host, dict) else None for host in hosts]
    if any(not isinstance(host, str) or not host for host in host_names) or len(
        set(host_names)
    ) != 4:
        raise QualificationError(f"{resolved} does not identify four Spark hosts")
    resident_ms = finite_positive(
        summary.get("maximum_resident_preload_ms"),
        f"{resolved}: maximum resident preload",
    )
    total_ms = finite_positive(
        summary.get("maximum_service_handoff_total_ms"),
        f"{resolved}: maximum expert startup",
    )
    return {
        "identity": evidence_identity(resolved, expected_schema),
        "model": report["model"],
        "preload_mode": report["preload_mode"],
        "expert_runtime_fingerprint": expert_runtime_fingerprint,
        "cache_state": report.get("cache_state"),
        "include_mtp": report.get("include_mtp"),
        "hosts": host_names,
        "maximum_resident_preload_ms": resident_ms,
        "maximum_service_handoff_total_ms": total_ms,
    }


def deployment(
    path: Path,
    *,
    candidate: bool,
    expected_model: str | None = None,
    expected_speculation: str = "dspark",
) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    report = _json_object(resolved)
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    fingerprints = report.get("fingerprints")
    inputs = report.get("inputs")
    speculation_settings = report.get("speculation_settings", {})
    required_fingerprints = {
        "coordinator_slot",
        "expert_slot",
        "expert_runtime",
        "deployment",
    }
    valid_inputs = (
        isinstance(inputs, dict)
        and set(inputs) == {"resolved_profile", "configuration"}
        and all(
            isinstance(record, dict)
            and isinstance(record.get("bytes"), int)
            and not isinstance(record.get("bytes"), bool)
            and record["bytes"] > 0
            and SHA256_RE.fullmatch(str(record.get("sha256", ""))) is not None
            for record in inputs.values()
        )
    )
    if (
        report.get("schema") != DEPLOYMENT_SCHEMA
        or report.get("status") != "ready"
        or not isinstance(digest, str)
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
        or REVISION_RE.fullmatch(str(report.get("model_revision", ""))) is None
        or not isinstance(report.get("slot"), str)
        or not report["slot"]
        or report.get("profile") not in {"balanced", "long", "accuracy"}
        or report.get("speculation") != expected_speculation
        or isinstance(report.get("launch_started_ns"), bool)
        or not isinstance(report.get("launch_started_ns"), int)
        or report["launch_started_ns"] <= 0
        or isinstance(report.get("power_limit_w"), bool)
        or not isinstance(report.get("power_limit_w"), int)
        or report["power_limit_w"] <= 0
        or not isinstance(report.get("engine_identity"), str)
        or REVISION_RE.fullmatch(str(report.get("sparkinfer_revision", ""))) is None
        or not isinstance(fingerprints, dict)
        or set(fingerprints) != required_fingerprints
        or any(
            SHA256_RE.fullmatch(str(fingerprints.get(name, ""))) is None
            for name in required_fingerprints
        )
        or report["engine_identity"]
        != (
            f"wip-{report['slot']}-{fingerprints['coordinator_slot'][:12]}-"
            f"{fingerprints['expert_slot'][:12]}"
        )
        or not valid_inputs
        or not isinstance(speculation_settings, dict)
    ):
        raise QualificationError(f"{resolved} is not valid WIP deployment evidence")
    if expected_speculation == "dflash2":
        legacy_keys = {
            "checkpoint_model_id",
            "checkpoint_revision",
            "fixed_drafts",
            "topk_backend",
        }
        policy_keys = legacy_keys | {"draft_policy", "proposal_drafts"}
        settings_keys = set(speculation_settings)
        fixed_drafts = speculation_settings.get("fixed_drafts")
        proposal_drafts = speculation_settings.get("proposal_drafts")
        draft_policy = speculation_settings.get("draft_policy")
        valid_fixed_drafts = (
            not isinstance(fixed_drafts, bool) and isinstance(fixed_drafts, int)
        )
        valid_policy = settings_keys == policy_keys and (
            (
                draft_policy == "adaptive"
                and fixed_drafts is None
                and not isinstance(proposal_drafts, bool)
                and isinstance(proposal_drafts, int)
                and 1 <= proposal_drafts <= 7
            )
            or (
                draft_policy == "fixed"
                and valid_fixed_drafts
                and not isinstance(proposal_drafts, bool)
                and isinstance(proposal_drafts, int)
                and 1 <= fixed_drafts <= proposal_drafts <= 7
            )
        )
        valid_legacy = (
            settings_keys == legacy_keys
            and valid_fixed_drafts
            and 0 <= fixed_drafts <= 7
        )
        if (
            not (valid_legacy or valid_policy)
            or speculation_settings.get("checkpoint_model_id") != DFLASH2_MODEL_ID
            or speculation_settings.get("checkpoint_revision") != DFLASH2_REVISION
            or speculation_settings.get("topk_backend")
            not in {"torch", "flashinfer", "flashinfer-dsa"}
        ):
            raise QualificationError(
                f"{resolved} has invalid DFlash2 deployment settings"
            )
    elif speculation_settings:
        raise QualificationError(
            f"{resolved} has settings for inactive DFlash2 speculation"
        )
    model = require_model(
        report.get("model_id"),
        candidate=candidate,
        field=os.fspath(resolved),
        expected_model=expected_model,
    )
    return {
        "identity": evidence_identity(resolved, DEPLOYMENT_SCHEMA),
        "model": model,
        "model_revision": report["model_revision"],
        "slot": report["slot"],
        "profile": report["profile"],
        "speculation": report["speculation"],
        "speculation_settings": speculation_settings,
        "launch_started_ns": report["launch_started_ns"],
        "power_limit_w": report["power_limit_w"],
        "engine_identity": report["engine_identity"],
        "sparkinfer_revision": report["sparkinfer_revision"],
        "fingerprints": fingerprints,
    }


def native_validations(
    paths: list[Path],
    *,
    expected_sparkinfer_revision: str,
    expected_checkpoint_root: Path,
    expected_expert_slot_fingerprint: str,
    expected_trellis_bits: int = 3,
    expected_required_rows: frozenset[int] = REQUIRED_NATIVE_ROWS,
    capacity_rows_for_live_rows: Callable[[int], int] = exl3_k3_capacity_rows,
    route_block_rows_for_capacity: Callable[[int], int] = exl3_k3_route_block_rows,
    expected_artifact_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_expert_slot_fingerprint) is None:
        raise QualificationError("native EXL3 expert-slot fingerprint is invalid")
    if expected_trellis_bits not in {3, 4}:
        raise QualificationError("native EXL3 trellis bits must be 3 or 4")
    if (
        expected_artifact_manifest_sha256 is not None
        and SHA256_RE.fullmatch(expected_artifact_manifest_sha256) is None
    ):
        raise QualificationError("native EXL3 artifact-manifest digest is invalid")
    artifact_backed = expected_artifact_manifest_sha256 is not None
    expected_weight_kind = (
        "finalized-exl3-artifact"
        if artifact_backed
        else "calibrated-projection-checkpoints"
    )
    if len(paths) != 4:
        raise QualificationError("native EXL3 qualification requires exactly four TP-rank reports")
    reports: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for path in paths:
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise QualificationError(f"native EXL3 evidence is a symbolic link: {expanded}")
        try:
            resolved = expanded.resolve(strict=True)
            report = _json_object(resolved)
        except (FileNotFoundError, OSError, ArtifactValidationError) as error:
            raise QualificationError(
                f"native EXL3 evidence cannot be read: {expanded}"
            ) from error
        digest = report.get("report_sha256")
        body = {key: value for key, value in report.items() if key != "report_sha256"}
        weight_source = report.get("weight_source")
        native_library = report.get("native_library")
        device = report.get("device")
        cases = report.get("cases")
        if (
            report.get("schema") != NATIVE_VALIDATION_SCHEMA
            or report.get("status") != "accepted"
            or report.get("script_sha256") != hash_file(NATIVE_VALIDATOR_SOURCE)
            or report.get("expert_slot_fingerprint")
            != expected_expert_slot_fingerprint
            or report.get("trellis_bits") != expected_trellis_bits
            or not isinstance(digest, str)
            or not SHA256_RE.fullmatch(digest)
            or hashlib.sha256(canonical_json(body)).hexdigest() != digest
            or report.get("sparkinfer_revision") != expected_sparkinfer_revision
            or not isinstance(weight_source, dict)
            or weight_source.get("kind") != expected_weight_kind
            or not isinstance(native_library, dict)
            or not isinstance(device, dict)
            or device.get("compute_capability") not in {"12.0", "12.1"}
            or not isinstance(cases, list)
        ):
            raise QualificationError(f"{resolved} is not accepted native EXL3 evidence")
        if Path(str(weight_source.get("root", ""))).expanduser().resolve() != expected_checkpoint_root:
            raise QualificationError(f"{resolved} uses another real-weight root")
        common_source_invalid = (
            weight_source.get("tp_world_size") != 4
            or not isinstance(weight_source.get("tp_rank"), int)
            or isinstance(weight_source.get("tp_rank"), bool)
            or not 0 <= weight_source["tp_rank"] < 4
            or not isinstance(weight_source.get("layer_id"), int)
            or not 3 <= weight_source["layer_id"] <= 77
            or weight_source.get("projection_count") != 768
        )
        if artifact_backed:
            authenticated_files = weight_source.get("authenticated_files")
            artifact_source_invalid = (
                weight_source.get("artifact_manifest_sha256")
                != expected_artifact_manifest_sha256
                or SHA256_RE.fullmatch(str(weight_source.get("plan_sha256", "")))
                is None
                or weight_source.get("tensor_count") != 3_072
                or not isinstance(authenticated_files, list)
                or len(authenticated_files) < 3
                or any(
                    not isinstance(record, dict)
                    or not isinstance(record.get("name"), str)
                    or integer(
                        record.get("bytes"),
                        f"{resolved}: authenticated artifact bytes",
                        minimum=1,
                    )
                    <= 0
                    or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
                    for record in authenticated_files
                )
            )
            if common_source_invalid or artifact_source_invalid:
                raise QualificationError(
                    f"{resolved} has invalid finalized-artifact coverage"
                )
        elif (
            common_source_invalid
            or integer(
                weight_source.get("tensor_bytes"),
                f"{resolved}: native checkpoint tensor bytes",
                minimum=1,
            )
            <= 0
            or SHA256_RE.fullmatch(str(weight_source.get("inventory_sha256", "")))
            is None
        ):
            raise QualificationError(
                f"{resolved} has invalid calibrated checkpoint coverage"
            )
        if (
            not isinstance(native_library.get("path"), str)
            or integer(
                native_library.get("bytes"),
                f"{resolved}: native library bytes",
                minimum=1,
            )
            <= 0
            or SHA256_RE.fullmatch(str(native_library.get("sha256", ""))) is None
        ):
            raise QualificationError(f"{resolved} has no native library identity")
        observed_rows: set[int] = set()
        for case in cases:
            if not isinstance(case, dict):
                raise QualificationError(f"{resolved} has an invalid native case")
            rows = integer(case.get("rows"), f"{resolved}: native rows", minimum=1)
            if rows in observed_rows:
                raise QualificationError(f"{resolved} duplicates native rows {rows}")
            observed_rows.add(rows)
            expected_capacity = capacity_rows_for_live_rows(rows)
            expected_route_block = route_block_rows_for_capacity(expected_capacity)
            packed_route_count = integer(
                case.get("packed_route_count"),
                f"{resolved}: native packed route count",
                minimum=rows * 8,
            )
            if (
                case.get("capacity_rows") != expected_capacity
                or case.get("route_block_rows") != expected_route_block
                or packed_route_count > 32_640
                or not isinstance(case.get("fc1_tile"), list)
                or len(case["fc1_tile"]) != 2
                or not isinstance(case.get("fc2_tile"), list)
                or len(case["fc2_tile"]) != 2
                or integer(
                    case.get("blocks_per_sm"),
                    f"{resolved}: native blocks per SM",
                    minimum=1,
                )
                <= 0
                or integer(
                    case.get("registers_per_thread"),
                    f"{resolved}: native registers per thread",
                    minimum=1,
                )
                <= 0
                or integer(
                    case.get("local_memory_bytes"),
                    f"{resolved}: native local memory",
                )
                != 0
                or finite_nonnegative(
                    case.get("relative_l2"), f"{resolved}: native relative L2"
                )
                > 2.0e-2
                or finite_nonnegative(
                    case.get("cosine"), f"{resolved}: native cosine"
                )
                < 0.999
            ):
                raise QualificationError(f"{resolved} failed native EXL3 parity at rows {rows}")
            finite_nonnegative(
                case.get("max_abs"), f"{resolved}: native maximum absolute error"
            )
        if not expected_required_rows.issubset(observed_rows):
            missing = sorted(expected_required_rows - observed_rows)
            raise QualificationError(f"{resolved} misses required native row cases {missing}")
        reports.append(report)
        identities.append(evidence_identity(resolved, NATIVE_VALIDATION_SCHEMA))

    ranks = [report["weight_source"]["tp_rank"] for report in reports]
    if sorted(ranks) != [0, 1, 2, 3]:
        raise QualificationError(f"native EXL3 reports do not cover TP ranks 0..3: {ranks}")
    shared_source_fields = (
        ("root", "layer_id", "projection_count", "tensor_count", "artifact_manifest_sha256", "plan_sha256")
        if artifact_backed
        else ("root", "layer_id", "projection_count", "tensor_bytes", "inventory_sha256")
    )
    for field in shared_source_fields:
        values = {str(report["weight_source"][field]) for report in reports}
        if len(values) != 1:
            raise QualificationError(f"native EXL3 reports disagree on weight source {field}")
    library_identities = {
        (report["native_library"]["bytes"], report["native_library"]["sha256"])
        for report in reports
    }
    if len(library_identities) != 1:
        raise QualificationError("native EXL3 reports used different native libraries")
    return {
        "identities": identities,
        "expert_slot_fingerprint": expected_expert_slot_fingerprint,
        "trellis_bits": expected_trellis_bits,
        "tp_ranks": sorted(ranks),
        "layer_id": reports[0]["weight_source"]["layer_id"],
        "checkpoint_inventory_sha256": reports[0]["weight_source"][
            "artifact_manifest_sha256" if artifact_backed else "inventory_sha256"
        ],
        "native_library": reports[0]["native_library"],
        "required_rows": sorted(expected_required_rows),
    }


def revalidate_native_evidence(
    report: dict[str, Any],
    *,
    expected_sparkinfer_revision: str,
    expected_checkpoint_root: Path,
    expected_expert_slot_fingerprint: str,
    expected_trellis_bits: int = 3,
    expected_required_rows: frozenset[int] = REQUIRED_NATIVE_ROWS,
    capacity_rows_for_live_rows: Callable[[int], int] = exl3_k3_capacity_rows,
    route_block_rows_for_capacity: Callable[[int], int] = exl3_k3_route_block_rows,
) -> dict[str, Any]:
    """Re-read all four native reports and match their embedded summary."""

    evidence = report.get("evidence")
    results = report.get("results")
    raw_identities = (
        evidence.get("candidate_native_validations")
        if isinstance(evidence, dict)
        else None
    )
    embedded = results.get("native_kernel") if isinstance(results, dict) else None
    if (
        not isinstance(raw_identities, list)
        or len(raw_identities) != 4
        or any(
            not isinstance(identity, dict)
            or identity.get("schema") != NATIVE_VALIDATION_SCHEMA
            or not isinstance(identity.get("path"), str)
            or not identity["path"]
            for identity in raw_identities
        )
        or not isinstance(embedded, dict)
    ):
        raise QualificationError(
            "serving qualification has incomplete native EXL3 evidence"
        )
    native = native_validations(
        [Path(identity["path"]) for identity in raw_identities],
        expected_sparkinfer_revision=expected_sparkinfer_revision,
        expected_checkpoint_root=expected_checkpoint_root,
        expected_expert_slot_fingerprint=expected_expert_slot_fingerprint,
        expected_trellis_bits=expected_trellis_bits,
        expected_required_rows=expected_required_rows,
        capacity_rows_for_live_rows=capacity_rows_for_live_rows,
        route_block_rows_for_capacity=route_block_rows_for_capacity,
    )
    expected_summary = {
        "expert_slot_fingerprint": native["expert_slot_fingerprint"],
        "trellis_bits": native["trellis_bits"],
        "tp_ranks": native["tp_ranks"],
        "layer_id": native["layer_id"],
        "checkpoint_inventory_sha256": native["checkpoint_inventory_sha256"],
        "native_library": native["native_library"],
        "required_rows": native["required_rows"],
    }
    if raw_identities != native["identities"] or embedded != expected_summary:
        raise QualificationError(
            "serving qualification native EXL3 summary differs from its evidence"
        )
    return expected_summary


def paired_equal(label: str, baseline: Any, candidate: Any) -> None:
    if baseline != candidate:
        raise QualificationError(f"baseline and candidate {label} differ")


def ratio(candidate: float, baseline: float) -> float:
    if (
        not math.isfinite(candidate)
        or candidate < 0.0
        or not math.isfinite(baseline)
        or baseline <= 0.0
    ):
        raise QualificationError(
            "performance ratio requires a finite non-negative candidate and "
            "a finite positive baseline"
        )
    return candidate / baseline


def qualify(
    *,
    artifact_path: Path,
    artifact_validation_path: Path,
    quant_evidence_path: Path,
    baseline_blended_path: Path,
    candidate_blended_path: Path,
    baseline_repeat_path: Path,
    candidate_repeat_path: Path,
    baseline_prefill_path: Path,
    candidate_prefill_path: Path,
    baseline_tool_eval_path: Path,
    candidate_tool_eval_path: Path,
    baseline_startup_path: Path,
    candidate_startup_path: Path,
    baseline_deployment_path: Path,
    candidate_deployment_path: Path,
    candidate_native_validation_paths: list[Path],
    minimum_decode_ratio: float,
    minimum_acceptance_ratio: float,
    minimum_repeat_ratio: float,
    minimum_prefill_ratio: float,
    minimum_tool_eval_points_ratio: float,
    maximum_resident_preload_ratio: float,
    maximum_expert_startup_ratio: float,
    expected_tool_eval_version: str = TOOL_EVAL_VERSION,
) -> dict[str, Any]:
    artifact = artifact_path.expanduser().resolve(strict=True)
    artifact_manifest = _json_object(artifact / "glmrt-gptqmodel-artifact.json")
    if artifact_manifest.get("schema") != ARTIFACT_SCHEMA:
        raise QualificationError("qualification artifact is not a completed GLMRT export")
    validation_identity, validation = _validation_evidence(
        artifact_validation_path,
        artifact=artifact,
        artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
    )
    quant_identity, quant = _quant_evidence(
        quant_evidence_path,
        plan_sha256=validation["plan_sha256"],
    )
    if (
        validation["projection_checkpoint"]["checkpoint_inventory_sha256"]
        != quant["integrity"]["checkpoint_inventory_sha256"]
    ):
        raise QualificationError(
            "artifact and quant evidence bind different projection inventories"
        )

    baseline_deployment = deployment(baseline_deployment_path, candidate=False)
    candidate_deployment = deployment(candidate_deployment_path, candidate=True)
    for label in ("coordinator_slot", "expert_slot"):
        paired_equal(
            f"deployment {label} fingerprint",
            baseline_deployment["fingerprints"][label],
            candidate_deployment["fingerprints"][label],
        )
    for label in (
        "slot",
        "profile",
        "speculation",
        "power_limit_w",
        "engine_identity",
        "sparkinfer_revision",
    ):
        paired_equal(
            f"deployment {label}",
            baseline_deployment[label],
            candidate_deployment[label],
        )
    profile = baseline_deployment["profile"]
    power_limit_w = baseline_deployment["power_limit_w"]
    native = native_validations(
        candidate_native_validation_paths,
        expected_sparkinfer_revision=candidate_deployment["sparkinfer_revision"],
        expected_checkpoint_root=Path(
            validation["projection_checkpoint"]["root"]
        ).expanduser().resolve(),
        expected_expert_slot_fingerprint=candidate_deployment["fingerprints"][
            "expert_slot"
        ],
    )

    baseline_blended = blended(baseline_blended_path, candidate=False)
    candidate_blended = blended(candidate_blended_path, candidate=True)
    paired_equal("blended prompt contract", baseline_blended["contract"], candidate_blended["contract"])
    paired_equal("blended prompt sequence", baseline_blended["prompts"], candidate_blended["prompts"])

    baseline_repeat = repeat_decode(baseline_repeat_path, candidate=False)
    candidate_repeat = repeat_decode(candidate_repeat_path, candidate=True)
    paired_equal("repeat prompt contract", baseline_repeat["contract"], candidate_repeat["contract"])
    paired_equal("repeat tokenizer", baseline_repeat["tokenizer_sha256"], candidate_repeat["tokenizer_sha256"])
    paired_equal("repeat prompt sequence", baseline_repeat["prompts"], candidate_repeat["prompts"])

    baseline_prefill = prefill(baseline_prefill_path, candidate=False)
    candidate_prefill = prefill(candidate_prefill_path, candidate=True)
    for label in ("profile", "run_id", "contract", "corpus_sha256", "tokenizer_sha256", "prompts"):
        paired_equal(f"prefill {label}", baseline_prefill[label], candidate_prefill[label])
    if baseline_prefill["profile"] != profile:
        raise QualificationError("prefill evidence was collected under another profile")
    paired_equal("prefill cell set", set(baseline_prefill["cells"]), set(candidate_prefill["cells"]))

    baseline_tools = tool_eval(
        baseline_tool_eval_path, candidate=False, expected_version=expected_tool_eval_version
    )
    candidate_tools = tool_eval(
        candidate_tool_eval_path, candidate=True, expected_version=expected_tool_eval_version
    )
    tool_config_fields = (
        "temperature",
        "timeout_seconds",
        "max_turns",
        "seed",
        "reference_date",
        "scenario_count",
        "scenario_ids",
        "concurrency",
        "error_rate",
        "alpha",
        "extra_params",
        "weight_by_difficulty",
    )
    for field in tool_config_fields:
        paired_equal(
            f"tool-evaluation config.{field}",
            baseline_tools["config"].get(field),
            candidate_tools["config"].get(field),
        )
    paired_equal("tool-evaluation scenario sequence", baseline_tools["scenario_ids"], candidate_tools["scenario_ids"])
    paired_equal("tool-evaluation maximum points", baseline_tools["maximum_points"], candidate_tools["maximum_points"])

    baseline_startup = startup(baseline_startup_path, candidate=False)
    candidate_startup = startup(candidate_startup_path, candidate=True)
    for label in ("cache_state", "include_mtp", "hosts"):
        paired_equal(
            f"expert-startup {label}", baseline_startup[label], candidate_startup[label]
        )
    paired_equal(
        "baseline startup/runtime fingerprint",
        baseline_deployment["fingerprints"]["expert_runtime"],
        baseline_startup["expert_runtime_fingerprint"],
    )
    paired_equal(
        "candidate startup/runtime fingerprint",
        candidate_deployment["fingerprints"]["expert_runtime"],
        candidate_startup["expert_runtime_fingerprint"],
    )

    decode_ratio = ratio(
        candidate_blended["wall_decode_tps"], baseline_blended["wall_decode_tps"]
    )
    acceptance_ratio = ratio(
        finite_positive(candidate_blended["accepted_draft_rate"], "candidate acceptance"),
        finite_positive(baseline_blended["accepted_draft_rate"], "baseline acceptance"),
    )
    repetition_ratio = ratio(
        candidate_repeat["aggregate_decode_tps"], baseline_repeat["aggregate_decode_tps"]
    )
    prefill_cells: list[dict[str, Any]] = []
    for key in sorted(baseline_prefill["cells"]):
        baseline_tps = baseline_prefill["cells"][key]
        candidate_tps = candidate_prefill["cells"][key]
        prefill_cells.append(
            {
                "base_context_tokens": key[0],
                "suffix_tokens": key[1],
                "baseline_tps": baseline_tps,
                "candidate_tps": candidate_tps,
                "ratio": ratio(candidate_tps, baseline_tps),
            }
        )
    minimum_observed_prefill_ratio = min(cell["ratio"] for cell in prefill_cells)
    tool_points_ratio = ratio(
        float(candidate_tools["total_points"]), float(baseline_tools["total_points"])
    )
    resident_preload_ratio = ratio(
        candidate_startup["maximum_resident_preload_ms"],
        baseline_startup["maximum_resident_preload_ms"],
    )
    expert_startup_ratio = ratio(
        candidate_startup["maximum_service_handoff_total_ms"],
        baseline_startup["maximum_service_handoff_total_ms"],
    )

    thresholds = {
        "minimum_blended_decode_ratio": minimum_decode_ratio,
        "minimum_blended_acceptance_ratio": minimum_acceptance_ratio,
        "minimum_repeat_decode_ratio": minimum_repeat_ratio,
        "minimum_per_cell_prefill_ratio": minimum_prefill_ratio,
        "minimum_tool_eval_points_ratio": minimum_tool_eval_points_ratio,
        "maximum_expert_resident_preload_ratio": maximum_resident_preload_ratio,
        "maximum_expert_startup_ratio": maximum_expert_startup_ratio,
    }
    gates = {
        "blended_decode": decode_ratio >= minimum_decode_ratio,
        "blended_acceptance": acceptance_ratio >= minimum_acceptance_ratio,
        "repeat_decode": repetition_ratio >= minimum_repeat_ratio,
        "prefill_every_cell": minimum_observed_prefill_ratio >= minimum_prefill_ratio,
        "tool_eval_points": tool_points_ratio >= minimum_tool_eval_points_ratio,
        "expert_resident_preload": resident_preload_ratio
        <= maximum_resident_preload_ratio,
        "expert_startup": expert_startup_ratio <= maximum_expert_startup_ratio,
        "native_kernel_parity": True,
    }
    if set(gates) != REQUIRED_GATES:
        raise AssertionError("serving qualification gate contract drifted")
    failed_gates = sorted(label for label, passed in gates.items() if not passed)
    body = {
        "schema": SCHEMA,
        "status": "accepted" if not failed_gates else "rejected",
        "model_id": MODEL_ID,
        "artifact": os.fspath(artifact),
        "artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
        "plan_sha256": validation["plan_sha256"],
        "artifact_validation": validation_identity,
        "quant_evidence": quant_identity,
        "runtime": {
            "engine_identity": baseline_deployment["engine_identity"],
            "coordinator_slot_fingerprint": baseline_deployment["fingerprints"][
                "coordinator_slot"
            ],
            "expert_slot_fingerprint": baseline_deployment["fingerprints"][
                "expert_slot"
            ],
            "sparkinfer_revision": baseline_deployment["sparkinfer_revision"],
            "profile": profile,
            "power_limit_w": power_limit_w,
            "speculation": baseline_deployment["speculation"],
            "baseline_model_revision": baseline_deployment["model_revision"],
            "candidate_model_revision": candidate_deployment["model_revision"],
            "baseline_expert_runtime_fingerprint": baseline_deployment[
                "fingerprints"
            ]["expert_runtime"],
            "candidate_expert_runtime_fingerprint": candidate_deployment[
                "fingerprints"
            ]["expert_runtime"],
            "baseline_deployment_fingerprint": baseline_deployment["fingerprints"][
                "deployment"
            ],
            "candidate_deployment_fingerprint": candidate_deployment["fingerprints"][
                "deployment"
            ],
        },
        "thresholds": thresholds,
        "gates": gates,
        "failed_gates": failed_gates,
        "evidence": {
            "baseline_blended": baseline_blended["identity"],
            "candidate_blended": candidate_blended["identity"],
            "baseline_repeat": baseline_repeat["identity"],
            "candidate_repeat": candidate_repeat["identity"],
            "baseline_prefill": baseline_prefill["identity"],
            "candidate_prefill": candidate_prefill["identity"],
            "baseline_tool_eval": baseline_tools["identity"],
            "candidate_tool_eval": candidate_tools["identity"],
            "baseline_startup": baseline_startup["identity"],
            "candidate_startup": candidate_startup["identity"],
            "baseline_deployment": baseline_deployment["identity"],
            "candidate_deployment": candidate_deployment["identity"],
            "candidate_native_validations": native["identities"],
        },
        "results": {
            "blended": {
                "baseline_model": baseline_blended["model"],
                "candidate_model": candidate_blended["model"],
                "prompt_contract_sha256": baseline_blended["contract"],
                "cases": baseline_blended["cases"],
                "baseline_wall_decode_tps": baseline_blended["wall_decode_tps"],
                "candidate_wall_decode_tps": candidate_blended["wall_decode_tps"],
                "decode_ratio": decode_ratio,
                "baseline_accepted_draft_rate": baseline_blended["accepted_draft_rate"],
                "candidate_accepted_draft_rate": candidate_blended["accepted_draft_rate"],
                "acceptance_ratio": acceptance_ratio,
                "baseline_all_quality_contracts_passed": baseline_blended[
                    "all_quality_contracts_passed"
                ],
                "candidate_all_quality_contracts_passed": candidate_blended[
                    "all_quality_contracts_passed"
                ],
                "baseline_quality_contract_failures": baseline_blended[
                    "quality_contract_failures"
                ],
                "candidate_quality_contract_failures": candidate_blended[
                    "quality_contract_failures"
                ],
            },
            "repeat": {
                "prompt_contract_sha256": baseline_repeat["contract"],
                "baseline_decode_tps": baseline_repeat["aggregate_decode_tps"],
                "candidate_decode_tps": candidate_repeat["aggregate_decode_tps"],
                "decode_ratio": repetition_ratio,
                "baseline_all_exact": baseline_repeat["all_exact_repetition_count"],
                "candidate_all_exact": candidate_repeat["all_exact_repetition_count"],
            },
            "prefill": {
                "profile": profile,
                "prompt_contract_sha256": baseline_prefill["contract"],
                "corpus_sha256": baseline_prefill["corpus_sha256"],
                "minimum_cell_ratio": minimum_observed_prefill_ratio,
                "cells": prefill_cells,
            },
            "tool_eval": {
                "version": expected_tool_eval_version,
                "scenarios": len(baseline_tools["scenario_ids"]),
                "baseline_points": baseline_tools["total_points"],
                "candidate_points": candidate_tools["total_points"],
                "maximum_points": baseline_tools["maximum_points"],
                "points_ratio": tool_points_ratio,
                "baseline_score": baseline_tools["final_score"],
                "candidate_score": candidate_tools["final_score"],
            },
            "expert_startup": {
                "cache_state": baseline_startup["cache_state"],
                "include_mtp": baseline_startup["include_mtp"],
                "hosts": baseline_startup["hosts"],
                "baseline_preload_mode": baseline_startup["preload_mode"],
                "candidate_preload_mode": candidate_startup["preload_mode"],
                "baseline_expert_runtime_fingerprint": baseline_startup[
                    "expert_runtime_fingerprint"
                ],
                "candidate_expert_runtime_fingerprint": candidate_startup[
                    "expert_runtime_fingerprint"
                ],
                "baseline_maximum_resident_preload_ms": baseline_startup[
                    "maximum_resident_preload_ms"
                ],
                "candidate_maximum_resident_preload_ms": candidate_startup[
                    "maximum_resident_preload_ms"
                ],
                "resident_preload_ratio": resident_preload_ratio,
                "baseline_maximum_service_handoff_total_ms": baseline_startup[
                    "maximum_service_handoff_total_ms"
                ],
                "candidate_maximum_service_handoff_total_ms": candidate_startup[
                    "maximum_service_handoff_total_ms"
                ],
                "startup_ratio": expert_startup_ratio,
            },
            "native_kernel": {
                "expert_slot_fingerprint": native["expert_slot_fingerprint"],
                "trellis_bits": native["trellis_bits"],
                "tp_ranks": native["tp_ranks"],
                "layer_id": native["layer_id"],
                "checkpoint_inventory_sha256": native[
                    "checkpoint_inventory_sha256"
                ],
                "native_library": native["native_library"],
                "required_rows": native["required_rows"],
            },
        },
    }
    return {**body, "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


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
    parser.add_argument("--artifact-validation", type=Path, required=True)
    parser.add_argument("--quant-evidence", type=Path, required=True)
    parser.add_argument("--baseline-blended", type=Path, required=True)
    parser.add_argument("--candidate-blended", type=Path, required=True)
    parser.add_argument("--baseline-repeat", type=Path, required=True)
    parser.add_argument("--candidate-repeat", type=Path, required=True)
    parser.add_argument("--baseline-prefill", type=Path, required=True)
    parser.add_argument("--candidate-prefill", type=Path, required=True)
    parser.add_argument("--baseline-tool-eval", type=Path, required=True)
    parser.add_argument("--candidate-tool-eval", type=Path, required=True)
    parser.add_argument("--baseline-startup", type=Path, required=True)
    parser.add_argument("--candidate-startup", type=Path, required=True)
    parser.add_argument("--baseline-deployment", type=Path, required=True)
    parser.add_argument("--candidate-deployment", type=Path, required=True)
    parser.add_argument(
        "--candidate-native-validation",
        type=Path,
        action="append",
        required=True,
        help="one calibrated native-parity JSON report; repeat exactly four times for TP ranks 0..3",
    )
    parser.add_argument("--minimum-decode-ratio", type=float, default=1.0)
    parser.add_argument("--minimum-acceptance-ratio", type=float, default=0.95)
    parser.add_argument("--minimum-repeat-ratio", type=float, default=1.0)
    parser.add_argument("--minimum-prefill-ratio", type=float, default=0.95)
    parser.add_argument("--minimum-tool-eval-points-ratio", type=float, default=0.98)
    parser.add_argument("--maximum-resident-preload-ratio", type=float, default=1.0)
    parser.add_argument("--maximum-expert-startup-ratio", type=float, default=1.0)
    parser.add_argument("--tool-eval-version", default=TOOL_EVAL_VERSION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    thresholds = (
        args.minimum_decode_ratio,
        args.minimum_acceptance_ratio,
        args.minimum_repeat_ratio,
        args.minimum_prefill_ratio,
        args.minimum_tool_eval_points_ratio,
        args.maximum_resident_preload_ratio,
        args.maximum_expert_startup_ratio,
    )
    if (
        any(not math.isfinite(value) or value <= 0.0 for value in thresholds)
    ):
        parser.error("thresholds must be positive")
    report = qualify(
        artifact_path=args.artifact,
        artifact_validation_path=args.artifact_validation,
        quant_evidence_path=args.quant_evidence,
        baseline_blended_path=args.baseline_blended,
        candidate_blended_path=args.candidate_blended,
        baseline_repeat_path=args.baseline_repeat,
        candidate_repeat_path=args.candidate_repeat,
        baseline_prefill_path=args.baseline_prefill,
        candidate_prefill_path=args.candidate_prefill,
        baseline_tool_eval_path=args.baseline_tool_eval,
        candidate_tool_eval_path=args.candidate_tool_eval,
        baseline_startup_path=args.baseline_startup,
        candidate_startup_path=args.candidate_startup,
        baseline_deployment_path=args.baseline_deployment,
        candidate_deployment_path=args.candidate_deployment,
        candidate_native_validation_paths=args.candidate_native_validation,
        minimum_decode_ratio=args.minimum_decode_ratio,
        minimum_acceptance_ratio=args.minimum_acceptance_ratio,
        minimum_repeat_ratio=args.minimum_repeat_ratio,
        minimum_prefill_ratio=args.minimum_prefill_ratio,
        minimum_tool_eval_points_ratio=args.minimum_tool_eval_points_ratio,
        maximum_resident_preload_ratio=args.maximum_resident_preload_ratio,
        maximum_expert_startup_ratio=args.maximum_expert_startup_ratio,
        expected_tool_eval_version=args.tool_eval_version,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted" else 2


if __name__ == "__main__":
    raise SystemExit(main())
