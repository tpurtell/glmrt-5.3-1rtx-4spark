from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "render_glm52_exl3_model_card.py"
SPEC = importlib.util.spec_from_file_location("_glm52_exl3_model_card", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
NATIVE_VALIDATOR_SHA256 = hashlib.sha256(
    (ROOT / "python" / "tools" / "validate_b12x_exl3_native.py").read_bytes()
).hexdigest()


def signed(path: Path, body: dict) -> Path:
    report = {
        **body,
        "report_sha256": hashlib.sha256(TOOL.canonical_json(body)).hexdigest(),
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def native_validations(root: Path) -> tuple[list[Path], dict]:
    library = root / "libglmrt_native.so"
    library.write_bytes(b"test-native-library")
    library_identity = {
        "path": str(library.resolve()),
        "bytes": library.stat().st_size,
        "sha256": TOOL.hash_file(library),
    }
    rows = [1, 3, 9, 10, 129, 257, 513, 1025, 2049, 2064]
    paths: list[Path] = []
    for tp_rank in range(4):
        path = signed(
            root / f"native-tp{tp_rank}.json",
            {
                "schema": "glmrt-b12x-exl3-native-validation-v1",
                "status": "accepted",
                "script_sha256": NATIVE_VALIDATOR_SHA256,
                "expert_slot_fingerprint": "2" * 64,
                "trellis_bits": 3,
                "sparkinfer_revision": "3" * 40,
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
                    "inventory_sha256": "f" * 64,
                },
                "cases": [
                    {
                        "rows": row,
                        "capacity_rows": (
                            row
                            if row in (9, 257)
                            else 2064
                            if row > 2048
                            else 1 << (row - 1).bit_length()
                        ),
                        "route_block_rows": (
                            8
                            if row <= 128
                            else 16
                            if row <= 257
                            else 32
                            if row <= 512
                            else 48
                            if row <= 1024
                            else 64
                        ),
                        "packed_route_count": row * 8,
                        "fc1_tile": [64, 256],
                        "fc2_tile": [64, 256],
                        "blocks_per_sm": 1,
                        "registers_per_thread": 200,
                        "local_memory_bytes": 0,
                        "relative_l2": 0.0,
                        "cosine": 1.0,
                        "max_abs": 0.0,
                    }
                    for row in rows
                ],
            },
        )
        paths.append(path)
    summary = {
        "expert_slot_fingerprint": "2" * 64,
        "trellis_bits": 3,
        "tp_ranks": [0, 1, 2, 3],
        "layer_id": 3,
        "checkpoint_inventory_sha256": "f" * 64,
        "native_library": library_identity,
        "required_rows": rows,
    }
    return paths, summary


def evidence(root: Path) -> tuple[Path, Path, Path, Path]:
    template = root / "template.md"
    template.write_text(
        "---\nlicense: mit\n---\n\n# Candidate\n\n## Qualification\n\n"
        f"{TOOL.MARKER}\n",
        encoding="utf-8",
    )
    plan = "a" * 64
    manifest = "b" * 64
    artifact = signed(
        root / "artifact.json",
        {
            "schema": TOOL.ARTIFACT_SCHEMA,
            "status": "accepted",
            "model_id": TOOL.MODEL_ID,
            "plan_sha256": plan,
            "artifact_manifest_sha256": manifest,
            "quantized_modules": 57_600,
            "exl3_tensors": 230_400,
            "retained_native_tensors": 1_234,
            "exl3_tensor_bytes": 272_734_848_000,
            "tp4_resident_bytes_per_spark": 68_714_572_800,
            "retained_native_bytes_verified": True,
            "artifact_manifest_file_hashes_verified": True,
            "projection_checkpoint_bytes_verified": True,
            "projection_checkpoint": {
                "root": str((root / "projection-checkpoints").resolve()),
                "projection_count": 57_600,
                "tensor_count": 230_400,
                "tensor_bytes": 272_734_848_000,
                "checkpoint_inventory_sha256": "e" * 64,
            },
        },
    )
    quant = signed(
        root / "quant.json",
        {
            "schema": TOOL.QUANT_SCHEMA,
            "status": "accepted",
            "plan": {"plan_sha256": plan},
            "coverage": {
                "projection_count": 57_600,
                "complete_expert_count": 75 * 256,
            },
            "integrity": {
                "tensor_payload_hashes_verified": True,
                "checkpoint_inventory_sha256": "e" * 64,
            },
            "metrics": {
                "global": {"aggregate_hessian_weighted_relative_error": 0.003}
            },
        },
    )
    native_paths, native_summary = native_validations(root)
    serving = signed(
        root / "serving.json",
        {
            "schema": TOOL.SERVING_SCHEMA,
            "status": "accepted",
            "model_id": TOOL.MODEL_ID,
            "artifact_manifest_sha256": manifest,
            "plan_sha256": plan,
            "artifact_validation": {"sha256": TOOL.hash_file(artifact)},
            "quant_evidence": {"sha256": TOOL.hash_file(quant)},
            "runtime": {
                "engine_identity": "wip-exl3-qualified-111111111111-222222222222",
                "sparkinfer_revision": "3" * 40,
                "coordinator_slot_fingerprint": "1" * 64,
                "expert_slot_fingerprint": "2" * 64,
                "profile": "balanced",
                "power_limit_w": 400,
                "speculation": "dspark",
            },
            "thresholds": {
                "minimum_blended_acceptance_ratio": 0.94,
                "minimum_per_cell_prefill_ratio": 0.79,
            },
            "gates": {name: True for name in sorted(TOOL.REQUIRED_GATES)},
            "failed_gates": [],
            "evidence": {
                "candidate_native_validations": [
                    {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": TOOL.hash_file(path),
                        "schema": "glmrt-b12x-exl3-native-validation-v1",
                    }
                    for path in native_paths
                ]
            },
            "results": {
                "blended": {
                    "baseline_wall_decode_tps": 32.0,
                    "candidate_wall_decode_tps": 48.0,
                    "decode_ratio": 1.5,
                    "baseline_accepted_draft_rate": 0.70,
                    "candidate_accepted_draft_rate": 0.69,
                    "acceptance_ratio": 0.69 / 0.70,
                    "cases": 35,
                    "candidate_all_quality_contracts_passed": True,
                },
                "repeat": {
                    "baseline_decode_tps": 60.0,
                    "candidate_decode_tps": 80.0,
                    "decode_ratio": 4 / 3,
                },
                "prefill": {
                    "minimum_cell_ratio": 1.05,
                    "cells": [
                        {
                            "base_context_tokens": 2048,
                            "suffix_tokens": 1024,
                            "baseline_tps": 1000.0,
                            "candidate_tps": 1050.0,
                            "ratio": 1.05,
                        }
                    ],
                },
                "tool_eval": {
                    "baseline_points": 88,
                    "candidate_points": 89,
                    "maximum_points": 100,
                    "points_ratio": 89 / 88,
                },
                "expert_startup": {
                    "baseline_maximum_resident_preload_ms": 22000.0,
                    "candidate_maximum_resident_preload_ms": 15000.0,
                    "resident_preload_ratio": 15 / 22,
                    "baseline_maximum_service_handoff_total_ms": 23000.0,
                    "candidate_maximum_service_handoff_total_ms": 16000.0,
                    "startup_ratio": 16 / 23,
                },
                "native_kernel": native_summary,
            },
        },
    )
    return template, artifact, quant, serving


def test_renders_exact_signed_qualification_without_pending_marker(tmp_path: Path) -> None:
    template, artifact, quant, serving = evidence(tmp_path)
    rendered = TOOL.render(
        template_path=template,
        artifact_validation_path=artifact,
        quant_evidence_path=quant,
        serving_qualification_path=serving,
        hub_revision=None,
    )

    serving_digest = json.loads(serving.read_text(encoding="utf-8"))["report_sha256"]
    assert TOOL.MARKER not in rendered
    assert "48.000 tok/s" in rendered
    assert "57,600" in rendered
    assert "Native EXL3 parity: TP ranks 0, 1, 2, 3" in rendered
    assert "rows 1, 3, 9, 10, 129, 257, 513, 1025, 2049, 2064" in rendered
    assert f"Qualification evidence SHA-256: `{serving_digest}`" in rendered
    assert "explicit decode-optimized tradeoff" in rendered
    assert "Candidate semantic contracts: 35/35 passed" in rendered


def test_rejects_missing_native_evidence_file(tmp_path: Path) -> None:
    template, artifact, quant, serving = evidence(tmp_path)
    report = json.loads(serving.read_text(encoding="utf-8"))
    Path(report["evidence"]["candidate_native_validations"][0]["path"]).unlink()

    with pytest.raises(TOOL.ModelCardError, match="verifiable native EXL3"):
        TOOL.render(
            template_path=template,
            artifact_validation_path=artifact,
            quant_evidence_path=quant,
            serving_qualification_path=serving,
            hub_revision=None,
        )


def test_rejects_serving_report_bound_to_another_quant_report(tmp_path: Path) -> None:
    template, artifact, quant, serving = evidence(tmp_path)
    report = json.loads(serving.read_text(encoding="utf-8"))
    report["quant_evidence"]["sha256"] = "0" * 64
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    serving.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.ModelCardError, match="not bound"):
        TOOL.render(
            template_path=template,
            artifact_validation_path=artifact,
            quant_evidence_path=quant,
            serving_qualification_path=serving,
            hub_revision=None,
        )


def test_rejects_incomplete_gates_or_a_nonbalanced_report(tmp_path: Path) -> None:
    template, artifact, quant, serving = evidence(tmp_path)
    report = json.loads(serving.read_text(encoding="utf-8"))
    report["gates"] = {"blended_decode": True}
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    serving.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.ModelCardError, match="not bound"):
        TOOL.render(
            template_path=template,
            artifact_validation_path=artifact,
            quant_evidence_path=quant,
            serving_qualification_path=serving,
            hub_revision=None,
        )

    template, artifact, quant, serving = evidence(tmp_path)
    report = json.loads(serving.read_text(encoding="utf-8"))
    report["runtime"]["profile"] = "long"
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    serving.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(TOOL.ModelCardError, match="not bound"):
        TOOL.render(
            template_path=template,
            artifact_validation_path=artifact,
            quant_evidence_path=quant,
            serving_qualification_path=serving,
            hub_revision=None,
        )
