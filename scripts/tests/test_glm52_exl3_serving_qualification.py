from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
import stage_glm52_exl3_hf_snapshot as STAGE  # noqa: E402
import validate_glm52_exl3_artifact as ARTIFACT  # noqa: E402

TOOL_PATH = TOOLS / "validate_glm52_exl3_serving_qualification.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_serving_qualification", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
NATIVE_VALIDATOR_SHA256 = hashlib.sha256(
    (TOOLS / "validate_b12x_exl3_native.py").read_bytes()
).hexdigest()


BASELINE = "lukealonso/GLM-5.2-NVFP4"
CANDIDATE = TOOL.MODEL_ID
PLAN_SHA256 = "b" * 64
MANIFEST_SHA256 = "a" * 64
PROMPT_SHA256 = "c" * 64
CONTRACT_SHA256 = "d" * 64
TOKENIZER_SHA256 = "e" * 64
CORPUS_SHA256 = "f" * 64


def bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(STAGE._canonical_json(value)).hexdigest()}


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def structural_evidence(root: Path) -> tuple[Path, Path, Path]:
    artifact = root / "artifact"
    artifact.mkdir()
    (artifact / "glmrt-gptqmodel-plan.json").write_text(
        json.dumps(
            {
                "schema": "glmrt-glm52-gptqmodel-plan-v2",
                "recipe": "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1",
                "source": {"release": "glm-5.2", "format": "bf16"},
            }
        ),
        encoding="utf-8",
    )
    (artifact / "glmrt-gptqmodel-artifact.json").write_text(
        json.dumps(
            {
                "schema": ARTIFACT.ARTIFACT_SCHEMA,
                "manifest_sha256": MANIFEST_SHA256,
            }
        ),
        encoding="utf-8",
    )
    validation = root / "artifact-validation.json"
    validation_body = {
        "schema": STAGE.VALIDATION_SCHEMA,
        "status": "accepted",
        "model_id": CANDIDATE,
        "artifact": str(artifact.resolve()),
        "artifact_manifest_sha256": MANIFEST_SHA256,
        "plan_sha256": PLAN_SHA256,
        "retained_native_bytes_verified": True,
        "artifact_manifest_file_hashes_verified": True,
        "projection_checkpoint_bytes_verified": True,
        "projection_checkpoint": {
            "root": str(root / "projection-checkpoints"),
            "projection_count": STAGE.EXPECTED_PROJECTIONS,
            "tensor_count": STAGE.EXPECTED_PROJECTIONS * 4,
            "tensor_bytes": 272_734_848_000,
            "checkpoint_inventory_sha256": "e" * 64,
        },
        "tokenizer_evidence": {
            "mode": "plan-bound",
            "tokenizer_files": [
                {"name": "tokenizer.json", "bytes": 1, "sha256": "1" * 64},
                {
                    "name": "tokenizer_config.json",
                    "bytes": 1,
                    "sha256": "2" * 64,
                },
            ],
        },
    }
    validation.write_bytes(
        STAGE._canonical_json(bound(validation_body, "report_sha256")) + b"\n"
    )
    quant = root / "quant-evidence.json"
    quant_body = {
        "schema": STAGE.QUANT_EVIDENCE_SCHEMA,
        "status": "accepted",
        "quality_scope": "projection-quantizer-evidence-not-end-to-end-model-quality",
        "plan": {"plan_sha256": PLAN_SHA256},
        "coverage": {
            "expected_projection_count": STAGE.EXPECTED_PROJECTIONS,
            "projection_count": STAGE.EXPECTED_PROJECTIONS,
            "expected_expert_count": 75 * 256,
            "observed_expert_count": 75 * 256,
            "complete_expert_count": 75 * 256,
            "recovered_expert_count": 0,
            "layers": list(range(3, 78)),
        },
        "integrity": {
            "tensor_payload_hashes_verified": True,
            "journal_record_count": STAGE.EXPECTED_PROJECTIONS,
            "checkpoint_inventory_sha256": "e" * 64,
        },
        "metrics": {
            "global": {"aggregate_hessian_weighted_relative_error": 0.003}
        },
    }
    quant.write_bytes(STAGE._canonical_json(bound(quant_body, "report_sha256")) + b"\n")
    return artifact, validation, quant


def blended(root: Path, name: str, model: str, tps: float, acceptance: float) -> Path:
    prompt_contract = {
        "suite": "weighted",
        "cases": [
            {
                "id": "code",
                "category": "code",
                "prompt": "fixture prompt",
                "max_tokens": 100,
            }
        ],
        "repeats": 1,
        "nonce_seed": 7,
        "temperature": 0,
        "enable_thinking": False,
        "quality_contract_version": "glmrt-semantic-decode-contract-v2",
    }
    drafts = 100
    accepted = round(drafts * acceptance)
    assert accepted / drafts == acceptance
    decode_ms = 99_000.0 / tps
    return write_jsonl(
        root / name,
        [
            {
                "case": "code",
                "repeat": 1,
                "prompt_sha256": PROMPT_SHA256,
                "runtime_captures": 0,
                "completion_tokens": 100,
                "decode_ms": decode_ms,
                "content_chars": 200,
                "draft_tokens": drafts,
                "accepted_draft_tokens": accepted,
                "accepted_draft_rate": acceptance,
                "quality_contract_version": "glmrt-semantic-decode-contract-v2",
                "quality_contract_passed": True,
                "quality_contract_issues": [],
            },
            {
                "aggregate": {
                    "schema": "glmrt-mtp-acceptance-aggregate-v3",
                    "model": model,
                    "nonce_seed": 7,
                    "prompt_contract": prompt_contract,
                    "cases": 1,
                    "cases_per_repeat": 1,
                    "corpus_repeats": 1,
                    "selected_case_ids": ["code"],
                    "prompt_contract_sha256": hashlib.sha256(
                        TOOL.canonical_json(prompt_contract)
                    ).hexdigest(),
                    "repeat_summaries": [
                        {"repeat": 1, "wall_decode_tps": tps}
                    ],
                    "wall_decode_tps": tps,
                    "median_repeat_wall_decode_tps": tps,
                    "accepted_draft_rate": acceptance,
                    "all_zero_runtime_captures": True,
                    "quality_contract_version": "glmrt-semantic-decode-contract-v2",
                    "all_quality_contracts_passed": True,
                    "quality_contract_failures": [],
                }
            },
        ],
    )


def repeat(root: Path, name: str, model: str, tps: float) -> Path:
    contract_body = {
        "word": "orchid",
        "requested_repetitions": 100,
        "requested_max_tokens": 1500,
        "warmups": 0,
        "repeats": 1,
        "nonce_seed": 9,
        "temperature": 0,
        "enable_thinking": False,
        "tokenizer_sha256": TOKENIZER_SHA256,
    }
    decode_ms = 99_000.0 / tps
    return write_jsonl(
        root / name,
        [
            {
                "record": "meta",
                "schema": "glmrt-repeat-decode-v2",
                "model": model,
                "word": "orchid",
                "requested_repetitions": 100,
                "requested_max_tokens": 1500,
                "warmups": 0,
                "repeats": 1,
                "nonce_seed": 9,
                "prompt_contract_sha256": hashlib.sha256(
                    TOOL.canonical_json(contract_body)
                ).hexdigest(),
                "tokenizer_sha256": TOKENIZER_SHA256,
            },
            {
                "record": "measurement",
                "sample": 0,
                "timed": True,
                "prompt_sha256": PROMPT_SHA256,
                "runtime_captures": 0,
                "completion_tokens": 100,
                "decode_ms": decode_ms,
                "word": "orchid",
                "requested_repetitions": 100,
                "observed_word_occurrences": 100,
                "exact_repetition_count": True,
                "requested_max_tokens": 1500,
            },
            {
                "record": "summary",
                "aggregate_decode_tps": tps,
                "all_zero_runtime_captures": True,
                "requested_completion_tokens": 1500,
                "actual_completion_tokens": [100],
                "observed_word_occurrences": [100],
                "all_exact_repetition_count": True,
                "timed_samples": 1,
            },
        ],
    )


def prefill(root: Path, name: str, model: str, tps: float) -> Path:
    prompts = [
        {
            "base_context_tokens": 0,
            "suffix_tokens": 1024,
            "repeat": 1,
            "prompt_sha256": PROMPT_SHA256,
        }
    ]
    prefill_ms = 1_024_000.0 / tps
    return write_jsonl(
        root / name,
        [
            {
                "schema": "glmrt-release-prefill-v2",
                "run_id": "paired-v1",
                "profile": "balanced",
                "model": model,
                "base_context_tokens": 0,
                "suffix_tokens": 1024,
                "prompt_tokens": 1028,
                "cached_prompt_tokens": 3,
                "prefill_rows": 1024,
                "repeat": 1,
                "prompt_sha256": PROMPT_SHA256,
                "runtime_captures": 0,
                "numeric_progression_passed": True,
                "attention_complete": True,
                "prefill_ms": prefill_ms,
                "prefill_tps": tps,
                "corpus_sha256": CORPUS_SHA256,
                "tokenizer_sha256": TOKENIZER_SHA256,
            },
            {
                "schema": "glmrt-release-prefill-summary-v3",
                "model": model,
                "profile": "balanced",
                "run_id": "paired-v1",
                "prompt_contract_sha256": hashlib.sha256(
                    TOOL.canonical_json(prompts)
                ).hexdigest(),
                "corpus_sha256": CORPUS_SHA256,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "cells": [
                    {
                        "base_context_tokens": 0,
                        "suffix_tokens": 1024,
                        "samples": 1,
                        "median_prefill_ms": prefill_ms,
                        "median_prefill_tps": tps,
                        "min_prefill_tps": tps,
                        "max_prefill_tps": tps,
                    }
                ],
            },
        ],
    )


def tool_eval(root: Path, name: str, model: str, points: int) -> Path:
    scenario_ids = ["TC-01", "TC-02"]
    remaining = points
    scenario_points = []
    for _scenario_id in scenario_ids:
        earned = min(2, max(0, remaining))
        remaining -= earned
        scenario_points.append(earned)
    assert remaining == 0
    status_by_points = {0: "fail", 1: "partial", 2: "pass"}
    config = {
        "model": model,
        "temperature": 0.0,
        "timeout_seconds": 120.0,
        "max_turns": 8,
        "seed": None,
        "reference_date": None,
        "scenario_count": 2,
        "scenario_ids": scenario_ids,
        "concurrency": 1,
        "error_rate": 0.0,
        "alpha": 0.7,
        "extra_params": None,
        "weight_by_difficulty": False,
    }
    report = {
        "schema_version": "1",
        "tool_eval_bench_version": TOOL.TOOL_EVAL_VERSION,
        "status": "completed",
        "final_score": round(points / 4 * 100),
        "config": config,
        "metadata": {
            "model": model,
            "tool_version": TOOL.TOOL_EVAL_VERSION,
        },
        "scores": {
            "total_points": points,
            "max_points": 4,
            "completion_rate": 100.0,
            "scenario_results": [
                {
                    "scenario_id": scenario_id,
                    "points": earned,
                    "status": status_by_points[earned],
                }
                for scenario_id, earned in zip(
                    scenario_ids, scenario_points, strict=True
                )
            ],
        },
    }
    path = root / name
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def startup(
    root: Path,
    name: str,
    model: str,
    weight_format: str,
    resident_ms: float,
    total_ms: float,
    expert_runtime_fingerprint: str,
) -> Path:
    body = {
        "schema": TOOL.STARTUP_SCHEMA,
        "status": "accepted",
        "model": model,
        "expert_runtime_fingerprint": expert_runtime_fingerprint,
        "weight_format": weight_format,
        "preload_mode": (
            "direct-resident" if weight_format == "exl3" else "nvfp4-production"
        ),
        "cache_state": "cold",
        "include_mtp": False,
        "hosts": [
            {"host": host} for host in ("ostrich", "dodo", "emu", "kiwi")
        ],
        "summary": {
            "maximum_resident_preload_ms": resident_ms,
            "maximum_service_handoff_total_ms": total_ms,
        },
    }
    path = root / name
    path.write_bytes(STAGE._canonical_json(bound(body, "report_sha256")) + b"\n")
    return path


def deployment(root: Path, name: str, model: str, *, candidate: bool) -> Path:
    coordinator_slot = "1" * 64
    expert_slot = "2" * 64
    body = {
        "schema": TOOL.DEPLOYMENT_SCHEMA,
        "status": "ready",
        "model_id": model,
        "model_revision": ("3" if candidate else "4") * 64,
        "slot": "exl3-qualified",
        "profile": "balanced",
        "speculation": "dspark",
        "launch_started_ns": 1_725_000_000_000_000_001 if candidate else 1_725_000_000_000_000_002,
        "power_limit_w": 400,
        "engine_identity": (
            f"wip-exl3-qualified-{coordinator_slot[:12]}-{expert_slot[:12]}"
        ),
        "sparkinfer_revision": "5" * 40,
        "fingerprints": {
            "coordinator_slot": coordinator_slot,
            "expert_slot": expert_slot,
            "expert_runtime": ("6" if candidate else "7") * 64,
            "deployment": ("8" if candidate else "9") * 64,
        },
        "inputs": {
            "resolved_profile": {"bytes": 100, "sha256": "a" * 64},
            "configuration": {"bytes": 200, "sha256": "b" * 64},
        },
    }
    path = root / name
    path.write_bytes(STAGE._canonical_json(bound(body, "report_sha256")) + b"\n")
    return path


def test_dflash2_deployment_accepts_adaptive_policy_contract(tmp_path: Path) -> None:
    path = deployment(tmp_path, "dflash2-adaptive.json", CANDIDATE, candidate=True)
    report = json.loads(path.read_text())
    report.pop("report_sha256")
    report["speculation"] = "dflash2"
    report["speculation_settings"] = {
        "checkpoint_model_id": TOOL.DFLASH2_MODEL_ID,
        "checkpoint_revision": TOOL.DFLASH2_REVISION,
        "draft_policy": "adaptive",
        "fixed_drafts": None,
        "proposal_drafts": 7,
        "topk_backend": "torch",
    }
    path.write_bytes(STAGE._canonical_json(bound(report, "report_sha256")) + b"\n")

    parsed = TOOL.deployment(
        path,
        candidate=True,
        expected_model=CANDIDATE,
        expected_speculation="dflash2",
    )

    assert parsed["speculation_settings"]["draft_policy"] == "adaptive"
    assert parsed["speculation_settings"]["proposal_drafts"] == 7


def test_dflash2_deployment_rejects_incoherent_policy_contract(tmp_path: Path) -> None:
    path = deployment(tmp_path, "dflash2-incoherent.json", CANDIDATE, candidate=True)
    report = json.loads(path.read_text())
    report.pop("report_sha256")
    report["speculation"] = "dflash2"
    report["speculation_settings"] = {
        "checkpoint_model_id": TOOL.DFLASH2_MODEL_ID,
        "checkpoint_revision": TOOL.DFLASH2_REVISION,
        "draft_policy": "adaptive",
        "fixed_drafts": 5,
        "proposal_drafts": 7,
        "topk_backend": "torch",
    }
    path.write_bytes(STAGE._canonical_json(bound(report, "report_sha256")) + b"\n")

    with pytest.raises(TOOL.QualificationError, match="invalid DFlash2"):
        TOOL.deployment(
            path,
            candidate=True,
            expected_model=CANDIDATE,
            expected_speculation="dflash2",
        )


def native_validations(root: Path) -> list[Path]:
    library = root / "libglmrt_native.so"
    library.write_bytes(b"test-native-library")
    library_identity = {
        "path": str(library.resolve()),
        "bytes": library.stat().st_size,
        "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    paths: list[Path] = []
    for tp_rank in range(4):
        body = {
            "schema": TOOL.NATIVE_VALIDATION_SCHEMA,
            "status": "accepted",
            "script_sha256": NATIVE_VALIDATOR_SHA256,
            "expert_slot_fingerprint": "2" * 64,
            "trellis_bits": 3,
            "sparkinfer_revision": "5" * 40,
            "native_library": library_identity,
            "device": {
                "name": "NVIDIA GB10",
                "compute_capability": "12.1",
            },
            "weight_source": {
                "kind": "calibrated-projection-checkpoints",
                "root": str((root / "projection-checkpoints").resolve()),
                "layer_id": 3,
                "tp_rank": tp_rank,
                "tp_world_size": 4,
                "projection_count": 768,
                "tensor_bytes": 3_636_464_640,
                "inventory_sha256": "2" * 64,
            },
            "cases": [
                {
                    "rows": rows,
                    "capacity_rows": (
                        rows
                        if rows in (9, 257)
                        else 2064
                        if rows > 2048
                        else 1 << (rows - 1).bit_length()
                    ),
                    "route_block_rows": TOOL.exl3_k3_route_block_rows(
                        TOOL.exl3_k3_capacity_rows(rows)
                    ),
                    "packed_route_count": rows * 8,
                    "fc1_tile": [64, 256],
                    "fc2_tile": [64, 256],
                    "blocks_per_sm": 1,
                    "registers_per_thread": 200,
                    "local_memory_bytes": 0,
                    "source_scale": 1.0,
                    "relative_l2": 0.0,
                    "cosine": 1.0,
                    "max_abs": 0.0,
                }
                for rows in sorted(TOOL.REQUIRED_NATIVE_ROWS)
            ],
        }
        path = root / f"native-tp{tp_rank}.json"
        path.write_bytes(STAGE._canonical_json(bound(body, "report_sha256")) + b"\n")
        paths.append(path)
    return paths


def test_native_evidence_rejects_the_wrong_trellis_bitrate(tmp_path: Path) -> None:
    checkpoint_root = tmp_path / "projection-checkpoints"
    checkpoint_root.mkdir()
    paths = native_validations(tmp_path)
    report = json.loads(paths[0].read_text(encoding="utf-8"))
    report.pop("report_sha256")
    report["trellis_bits"] = 4
    paths[0].write_bytes(
        STAGE._canonical_json(bound(report, "report_sha256")) + b"\n"
    )

    with pytest.raises(TOOL.QualificationError, match="accepted native EXL3"):
        TOOL.native_validations(
            paths,
            expected_sparkinfer_revision="5" * 40,
            expected_checkpoint_root=checkpoint_root.resolve(),
            expected_expert_slot_fingerprint="2" * 64,
        )

    report.pop("report_sha256", None)
    report["trellis_bits"] = 3
    report["script_sha256"] = "0" * 64
    paths[0].write_bytes(
        STAGE._canonical_json(bound(report, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.QualificationError, match="accepted native EXL3"):
        TOOL.native_validations(
            paths,
            expected_sparkinfer_revision="5" * 40,
            expected_checkpoint_root=checkpoint_root.resolve(),
            expected_expert_slot_fingerprint="2" * 64,
        )

    report.pop("report_sha256", None)
    report["script_sha256"] = NATIVE_VALIDATOR_SHA256
    report["expert_slot_fingerprint"] = "0" * 64
    paths[0].write_bytes(
        STAGE._canonical_json(bound(report, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.QualificationError, match="accepted native EXL3"):
        TOOL.native_validations(
            paths,
            expected_sparkinfer_revision="5" * 40,
            expected_checkpoint_root=checkpoint_root.resolve(),
            expected_expert_slot_fingerprint="2" * 64,
        )


def inputs(root: Path, *, candidate_decode_tps: float = 48.0) -> dict:
    artifact, validation, quant = structural_evidence(root)
    return {
        "artifact_path": artifact,
        "artifact_validation_path": validation,
        "quant_evidence_path": quant,
        "baseline_blended_path": blended(root, "baseline-blended.jsonl", BASELINE, 32.0, 0.70),
        "candidate_blended_path": blended(root, "candidate-blended.jsonl", CANDIDATE, candidate_decode_tps, 0.69),
        "baseline_repeat_path": repeat(root, "baseline-repeat.jsonl", BASELINE, 60.0),
        "candidate_repeat_path": repeat(root, "candidate-repeat.jsonl", CANDIDATE, 80.0),
        "baseline_prefill_path": prefill(root, "baseline-prefill.jsonl", BASELINE, 1000.0),
        "candidate_prefill_path": prefill(root, "candidate-prefill.jsonl", CANDIDATE, 1050.0),
        "baseline_tool_eval_path": tool_eval(root, "baseline-tools.json", BASELINE, 4),
        "candidate_tool_eval_path": tool_eval(root, "candidate-tools.json", CANDIDATE, 4),
        "baseline_startup_path": startup(
            root,
            "baseline-startup.json",
            BASELINE,
            "nvfp4",
            20_000.0,
            25_000.0,
            "7" * 64,
        ),
        "candidate_startup_path": startup(
            root,
            "candidate-startup.json",
            CANDIDATE,
            "exl3",
            10_000.0,
            15_000.0,
            "6" * 64,
        ),
        "baseline_deployment_path": deployment(
            root, "baseline-deployment.json", BASELINE, candidate=False
        ),
        "candidate_deployment_path": deployment(
            root, "candidate-deployment.json", CANDIDATE, candidate=True
        ),
        "candidate_native_validation_paths": native_validations(root),
        "minimum_decode_ratio": 1.0,
        "minimum_acceptance_ratio": 0.95,
        "minimum_repeat_ratio": 1.0,
        "minimum_prefill_ratio": 0.95,
        "minimum_tool_eval_points_ratio": 0.98,
        "maximum_resident_preload_ratio": 1.0,
        "maximum_expert_startup_ratio": 1.0,
    }


def test_accepts_exactly_paired_serving_evidence(tmp_path: Path) -> None:
    report = TOOL.qualify(**inputs(tmp_path))

    assert report["status"] == "accepted"
    assert report["failed_gates"] == []
    assert report["results"]["blended"]["decode_ratio"] == 1.5
    assert report["gates"]["native_kernel_parity"] is True
    assert report["results"]["native_kernel"]["tp_ranks"] == [0, 1, 2, 3]
    assert report["results"]["native_kernel"]["expert_slot_fingerprint"] == "2" * 64
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == hashlib.sha256(TOOL.canonical_json(body)).hexdigest()


def test_rejects_unpaired_prompt_sequence(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    candidate = arguments["candidate_repeat_path"]
    records = [json.loads(line) for line in candidate.read_text().splitlines()]
    records[1]["prompt_sha256"] = "0" * 64
    write_jsonl(candidate, records)

    with pytest.raises(TOOL.QualificationError, match="prompt sequence"):
        TOOL.qualify(**arguments)


def test_rejects_blended_aggregate_that_differs_from_measurements(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    candidate = arguments["candidate_blended_path"]
    records = [json.loads(line) for line in candidate.read_text().splitlines()]
    records[-1]["aggregate"]["wall_decode_tps"] = 999.0
    write_jsonl(candidate, records)

    with pytest.raises(TOOL.QualificationError, match="differs from measurements"):
        TOOL.qualify(**arguments)


def test_rejects_a_failed_semantic_output_contract(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    candidate = arguments["candidate_blended_path"]
    records = [json.loads(line) for line in candidate.read_text().splitlines()]
    records[0]["quality_contract_passed"] = False
    records[0]["quality_contract_issues"] = ["Python code does not parse"]
    records[1]["aggregate"]["all_quality_contracts_passed"] = False
    records[1]["aggregate"]["quality_contract_failures"] = [
        {"case": "code", "repeat": 1, "issues": ["Python code does not parse"]}
    ]
    write_jsonl(candidate, records)

    with pytest.raises(TOOL.QualificationError, match="semantic output contract"):
        TOOL.qualify(**arguments)


def test_records_baseline_semantic_failures_without_blocking_candidate(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    baseline = arguments["baseline_blended_path"]
    records = [json.loads(line) for line in baseline.read_text().splitlines()]
    records[0]["quality_contract_passed"] = False
    records[0]["quality_contract_issues"] = ["baseline formatting miss"]
    records[1]["aggregate"]["all_quality_contracts_passed"] = False
    records[1]["aggregate"]["quality_contract_failures"] = [
        {"case": "code", "repeat": 1, "issues": ["baseline formatting miss"]}
    ]
    write_jsonl(baseline, records)

    report = TOOL.qualify(**arguments)
    blended = report["results"]["blended"]
    assert blended["baseline_all_quality_contracts_passed"] is False
    assert blended["candidate_all_quality_contracts_passed"] is True
    assert blended["baseline_quality_contract_failures"] == [
        {"case": "code", "repeat": 1, "issues": ["baseline formatting miss"]}
    ]


def test_records_performance_rejection_without_accepting_it(tmp_path: Path) -> None:
    report = TOOL.qualify(**inputs(tmp_path, candidate_decode_tps=31.0))

    assert report["status"] == "rejected"
    assert report["failed_gates"] == ["blended_decode"]


def test_accepts_bounded_inexact_repeat_and_preserves_diagnostic(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    candidate = arguments["candidate_repeat_path"]
    records = [json.loads(line) for line in candidate.read_text().splitlines()]
    records[1]["exact_repetition_count"] = False
    records[1]["observed_word_occurrences"] = 99
    records[2]["all_exact_repetition_count"] = False
    records[2]["observed_word_occurrences"] = [99]
    write_jsonl(candidate, records)

    report = TOOL.qualify(**arguments)
    assert report["results"]["repeat"]["candidate_all_exact"] is False


def test_rejects_output_that_is_not_a_bounded_repeat_workload(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    candidate = arguments["candidate_repeat_path"]
    records = [json.loads(line) for line in candidate.read_text().splitlines()]
    records[1]["exact_repetition_count"] = False
    records[1]["observed_word_occurrences"] = 79
    records[2]["all_exact_repetition_count"] = False
    records[2]["observed_word_occurrences"] = [79]
    write_jsonl(candidate, records)

    with pytest.raises(TOOL.QualificationError, match="bounded word-repetition"):
        TOOL.qualify(**arguments)


def test_rejects_prefill_summary_that_differs_from_measurements(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    candidate = arguments["candidate_prefill_path"]
    records = [json.loads(line) for line in candidate.read_text().splitlines()]
    records[-1]["cells"][0]["median_prefill_tps"] = 999.0
    write_jsonl(candidate, records)

    with pytest.raises(TOOL.QualificationError, match="differs from measurements"):
        TOOL.qualify(**arguments)


def test_rejects_prefill_that_did_not_reuse_its_claimed_base_context(
    tmp_path: Path,
) -> None:
    path = prefill(tmp_path, "cached-prefill.jsonl", CANDIDATE, 1_000.0)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    measurement = records[0]
    measurement["base_context_tokens"] = 32_768
    measurement["cached_prompt_tokens"] = 16_384
    measurement["prompt_tokens"] = 17_409
    records[-1]["cells"][0]["base_context_tokens"] = 32_768
    prompts = [
        {
            "base_context_tokens": 32_768,
            "suffix_tokens": 1_024,
            "repeat": 1,
            "prompt_sha256": PROMPT_SHA256,
        }
    ]
    records[-1]["prompt_contract_sha256"] = hashlib.sha256(
        TOOL.canonical_json(prompts)
    ).hexdigest()
    write_jsonl(path, records)

    with pytest.raises(TOOL.QualificationError, match="runtime correctness"):
        TOOL.prefill(path, candidate=True)


def test_rejects_zero_point_baseline_instead_of_dividing_by_zero(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    arguments["baseline_tool_eval_path"] = tool_eval(
        tmp_path, "baseline-tools-zero.json", BASELINE, 0
    )

    with pytest.raises(TOOL.QualificationError, match="positive baseline"):
        TOOL.qualify(**arguments)


def test_rejects_tool_eval_aggregate_that_differs_from_scenarios(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_tool_eval_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["scores"]["total_points"] = 3
    report["final_score"] = 75
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="invalid tool-evaluation totals"):
        TOOL.qualify(**arguments)


def test_rejects_non_greedy_tool_eval(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_tool_eval_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["config"]["temperature"] = 0.7
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="invalid tool-evaluation totals"):
        TOOL.qualify(**arguments)


def test_rejects_candidate_built_from_different_expert_slot(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_deployment_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["fingerprints"]["expert_slot"] = "0" * 64
    report["engine_identity"] = (
        f"wip-{report['slot']}-{report['fingerprints']['coordinator_slot'][:12]}-"
        f"{report['fingerprints']['expert_slot'][:12]}"
    )
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="expert_slot fingerprint"):
        TOOL.qualify(**arguments)


def test_rejects_candidate_startup_from_legacy_preload_mode(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_startup_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["preload_mode"] = "cooperative"
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="accepted expert-startup"):
        TOOL.qualify(**arguments)


def test_accepts_candidate_startup_from_cooperative_coalesced_mode(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_startup_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["preload_mode"] = "cooperative-coalesced"
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    result = TOOL.qualify(**arguments)

    assert result["results"]["expert_startup"]["candidate_preload_mode"] == (
        "cooperative-coalesced"
    )


def test_rejects_startup_from_another_deployed_expert_runtime(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_startup_path"]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["expert_runtime_fingerprint"] = "0" * 64
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="startup/runtime fingerprint"):
        TOOL.qualify(**arguments)


def test_rejects_native_validation_without_the_combined_suffix_bucket(
    tmp_path: Path,
) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_native_validation_paths"][0]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["cases"] = [case for case in report["cases"] if case["rows"] != 2_064]
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="misses required native row cases"):
        TOOL.qualify(**arguments)


def test_rejects_duplicate_native_tp_rank(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    path = arguments["candidate_native_validation_paths"][3]
    report = json.loads(path.read_text(encoding="utf-8"))
    report["weight_source"]["tp_rank"] = 2
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.QualificationError, match="do not cover TP ranks"):
        TOOL.qualify(**arguments)
