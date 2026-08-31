from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "validate_glm52_exl3_quant_evidence.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_evidence_validator", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def _bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(TOOL.canonical_json(value)).hexdigest()}


def _metrics(*, apply_out_scales: bool = False) -> dict:
    return {
        "schema": TOOL.METRICS_SCHEMA,
        "schema_version": 1,
        "quantizer_path": "hessian_ldlq",
        "reported_metric_kind": "hessian_weighted_relative_error",
        "reported_metric_value": 0.1,
        "hessian_domain": "regularized_exl3_search_space",
        "hessian_sample_count": 1024,
        "hessian_regularization_sigma": 0.025,
        "hessian_numerical_contract": "signed-block-hadamard-congruence-fp64-v1",
        "hessian_transform_compute_dtype": "torch.float64",
        "hessian_storage_dtype": "torch.float32",
        "hessian_regularization_placement": "before-fp64-congruence",
        "hessian_regularization_diagonal_addend": 0.001,
        "hessian_symmetry_restoration": "mean-with-transpose-fp64",
        "hessian_symmetry_correction_max_abs": 0.0,
        "hessian_weighted_error_numerator": 2.0,
        "hessian_weighted_reference_denominator": 20.0,
        "hessian_weighted_relative_error": 0.1,
        "hessian_metric_status": "ok",
        "selected_global_scale": 0.95,
        "scale_search_mse": 0.02,
        "apply_out_scales": apply_out_scales,
        "reconstruction": {
            "domain": "regularized_exl3_search_space",
            "shape": [16, 16],
            "element_count": 256,
            "error_sum_sq": 4.0,
            "reference_sum_sq": 40.0,
            "mse": 4.0 / 256,
            "nmse": 0.1,
            "relative_frobenius": math.sqrt(0.1),
            "mean_abs_error": 0.01,
            "max_abs_error": 0.2,
            "reference_finite": True,
            "error_finite": True,
            "tile_shape": [16, 16],
            "tile_count": 1,
            "tile_sse_sum": 4.0,
            "tile_sse_max": 4.0,
            "tile_sse_percentiles": {
                "p50": 4.0,
                "p90": 4.0,
                "p99": 4.0,
                "p99_9": 4.0,
            },
            "worst_tiles": [{"row": 0, "column": 0, "sse": 4.0}],
        },
    }


def _fixture(tmp_path: Path, *, bits: int = 3) -> tuple[Path, Path, Path]:
    checkpoint_root = tmp_path / "projections"
    run_state = tmp_path / "run-state"
    run_state.mkdir()
    family = {
        "bits": bits,
        "codebook": "mcg",
        "quantizer_numerics": {"sigma_reg": 0.025},
    }
    plan_body = {
        "schema": (
            "glmrt-glm5-gptqmodel-plan-v3"
            if bits == 4
            else "glmrt-glm52-gptqmodel-plan-v2"
        ),
        "projection_checkpoint": {
            "contract": "ds4rt.exl3-projection-checkpoint-v1",
            "root": str(checkpoint_root),
        },
        "run_state_dir": str(run_state),
        "ledger_provenance": {"family_join": family},
    }
    if bits == 4:
        plan_body.update(
            {
                "recipe": "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1",
                "source": {
                    "release": "glm-5.3",
                    "format": "fp8-e4m3-block128x128-dynamic",
                    "geometry": {
                        "first_target_layer": 3,
                        "last_target_layer": 77,
                        "n_routed_experts": 256,
                        "mtp_layer_index": 78,
                    },
                },
                "exl3": {"bits": 4, "codebook": "mcg"},
            }
        )
    plan = _bound(plan_body, "plan_sha256")
    plan_path = run_state / "glmrt-gptqmodel-plan.json"
    plan_path.write_bytes(TOOL.canonical_json(plan) + b"\n")

    module = "model.layers.3.mlp.experts.0.gate_proj"
    route = {
        "block_namespace": "base",
        "logical_layer": 3,
        "expert": 0,
    }
    request = _bound(
        {
            "schema": TOOL.CHECKPOINT_SCHEMA,
            "schema_version": TOOL.CHECKPOINT_SCHEMA_VERSION,
            "module": module,
            "processor_layer_index": 3,
            "sample_count": 1024,
            "input_weight": {
                "shape": [16, 16],
                "dtype": "torch.float32",
                "numel": 256,
                "bytes": 1024,
                "sha256": "a" * 64,
            },
            "quantizer_contract": {
                "bits": bits,
                "codebook": "mcg",
                "apply_out_scales": None,
            },
            "family_join": family,
            "route_evidence": route,
        },
        "request_sha256",
    )
    metrics = _metrics(apply_out_scales=False)
    tensor_specs = {
        "trellis": {
            "shape": [1, 1, 16 * bits],
            "dtype": "torch.int16",
            "numel": 16 * bits,
            "bytes": 32 * bits,
            "sha256": "b" * 64,
        },
        "suh": {
            "shape": [16],
            "dtype": "torch.float16",
            "numel": 16,
            "bytes": 32,
            "sha256": "c" * 64,
        },
        "svh": {
            "shape": [16],
            "dtype": "torch.float16",
            "numel": 16,
            "bytes": 32,
            "sha256": "d" * 64,
        },
        "mcg": {
            "shape": [],
            "dtype": "torch.int32",
            "numel": 1,
            "bytes": 4,
            "sha256": TOOL.MCG_SHA256,
        },
    }
    ledger = {
        "schema": "ds4rt.exl3-error-ledger",
        "schema_version": 1,
        "record_kind": "projection",
        "module": module,
        "logical_layer": 3,
        "processor_layer_index": 3,
        "expert": 0,
        "projection": "w1",
        "bits": bits,
        "codebook": "mcg",
        "sample_count": 1024,
        "encoded_bytes": 32 * bits + 68,
        "route_evidence": route,
        "quantizer_metrics": metrics,
        "provenance": {"family_join": family},
    }
    result = {
        "proxy_error": 0.1,
        "quantizer_metrics": metrics,
        "ledger_record": ledger,
    }
    digest = request["request_sha256"]
    directory = checkpoint_root / digest[:2] / digest[2:4]
    directory.mkdir(parents=True)
    tensor_path = directory / f"{digest}.safetensors"
    tensor_path.write_bytes(b"packed-test-payload")
    manifest = _bound(
        {
            "schema": TOOL.CHECKPOINT_SCHEMA,
            "schema_version": TOOL.CHECKPOINT_SCHEMA_VERSION,
            "request": request,
            "request_sha256": digest,
            "tensor_file": tensor_path.name,
            "tensor_sha256": TOOL.sha256_file(tensor_path),
            "tensors": tensor_specs,
            "result": result,
        },
        "manifest_sha256",
    )
    (directory / f"{digest}.json").write_bytes(TOOL.canonical_json(manifest) + b"\n")
    journal = run_state / ".glmrt-exl3-error-journal.jsonl"
    journal.write_bytes(TOOL.canonical_json(_bound(ledger, "record_sha256")) + b"\n")
    return plan_path, checkpoint_root, journal


def _add_unjournaled_checkpoint(checkpoint_root: Path) -> None:
    source_manifest_path = next(checkpoint_root.rglob("*.json"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_tensor = source_manifest_path.with_suffix(".safetensors")
    manifest = {
        key: value
        for key, value in source_manifest.items()
        if key != "manifest_sha256"
    }
    request = dict(manifest["request"])
    request.pop("request_sha256")
    request["module"] = "model.layers.3.mlp.experts.1.gate_proj"
    request["route_evidence"] = dict(request["route_evidence"], expert=1)
    request = _bound(request, "request_sha256")
    ledger = dict(manifest["result"]["ledger_record"])
    ledger["module"] = request["module"]
    ledger["expert"] = 1
    ledger["route_evidence"] = request["route_evidence"]
    result = dict(manifest["result"], ledger_record=ledger)
    digest = request["request_sha256"]
    directory = checkpoint_root / digest[:2] / digest[2:4]
    directory.mkdir(parents=True)
    tensor_path = directory / f"{digest}.safetensors"
    tensor_path.write_bytes(source_tensor.read_bytes())
    manifest.update(
        {
            "request": request,
            "request_sha256": digest,
            "tensor_file": tensor_path.name,
            "tensor_sha256": TOOL.sha256_file(tensor_path),
            "result": result,
        }
    )
    manifest = _bound(manifest, "manifest_sha256")
    (directory / f"{digest}.json").write_bytes(TOOL.canonical_json(manifest) + b"\n")


def test_partial_validation_authenticates_autoscale_false_checkpoint(
    tmp_path: Path,
) -> None:
    plan, checkpoints, journal = _fixture(tmp_path)

    report = TOOL.validate_evidence(
        plan_path=plan,
        checkpoint_root=checkpoints,
        journal_path=journal,
        require_complete=False,
        verify_tensor_hashes=True,
    )

    assert report["status"] == "partial-accepted"
    assert report["coverage"]["projection_count"] == 1
    assert report["coverage"]["observed_expert_count"] == 1
    assert report["integrity"]["tensor_payload_hashes_verified"] is True
    assert report["metrics"]["global"][
        "aggregate_hessian_weighted_relative_error"
    ] == pytest.approx(0.1)
    assert report["metrics"]["by_route_evidence"]["natural-route"][
        "projection_count"
    ] == 1


def test_glm53_k4_evidence_uses_64_lane_trellis_and_new_schema(
    tmp_path: Path,
) -> None:
    plan, checkpoints, journal = _fixture(tmp_path, bits=4)

    report = TOOL.validate_evidence(
        plan_path=plan,
        checkpoint_root=checkpoints,
        journal_path=journal,
        require_complete=False,
        verify_tensor_hashes=True,
    )

    assert report["schema"] == TOOL.GLM53_SCHEMA
    assert report["scope"] == "glm-5.3-base-routed-experts-layers-3-through-77"
    assert report["coverage"]["expected_projection_count"] == 57_600
    assert report["integrity"]["logical_encoded_tensor_bytes"] == 196


def test_live_snapshot_authenticates_only_the_durable_journal_frontier(
    tmp_path: Path,
) -> None:
    plan, checkpoints, journal = _fixture(tmp_path, bits=4)
    _add_unjournaled_checkpoint(checkpoints)

    report = TOOL.validate_evidence(
        plan_path=plan,
        checkpoint_root=checkpoints,
        journal_path=journal,
        require_complete=False,
        verify_tensor_hashes=True,
        live_journal_snapshot=True,
    )

    assert report["status"] == "partial-live-snapshot-accepted"
    assert report["coverage"]["projection_count"] == 1
    assert report["integrity"]["journal_record_count"] == 1
    assert report["integrity"]["checkpoint_pairs_seen"] == 2
    assert report["integrity"]["post_snapshot_checkpoint_count"] == 1


def test_live_snapshot_cannot_be_used_as_complete_release_evidence(
    tmp_path: Path,
) -> None:
    plan, checkpoints, journal = _fixture(tmp_path, bits=4)

    with pytest.raises(TOOL.EvidenceValidationError, match="cannot prove complete"):
        TOOL.validate_evidence(
            plan_path=plan,
            checkpoint_root=checkpoints,
            journal_path=journal,
            require_complete=True,
            verify_tensor_hashes=True,
            live_journal_snapshot=True,
        )


def test_validation_rejects_quantizer_arithmetic_drift(tmp_path: Path) -> None:
    plan, checkpoints, journal = _fixture(tmp_path)
    manifest_path = next(checkpoints.rglob("*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result"]["quantizer_metrics"]["hessian_weighted_relative_error"] = 0.2
    manifest["manifest_sha256"] = hashlib.sha256(
        TOOL.canonical_json(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        )
    ).hexdigest()
    manifest_path.write_bytes(TOOL.canonical_json(manifest) + b"\n")

    with pytest.raises(TOOL.EvidenceValidationError):
        TOOL.validate_evidence(
            plan_path=plan,
            checkpoint_root=checkpoints,
            journal_path=journal,
            require_complete=False,
            verify_tensor_hashes=True,
        )


def test_final_validation_rejects_partial_coverage(tmp_path: Path) -> None:
    plan, checkpoints, journal = _fixture(tmp_path)

    with pytest.raises(TOOL.EvidenceValidationError, match="coverage"):
        TOOL.validate_evidence(
            plan_path=plan,
            checkpoint_root=checkpoints,
            journal_path=journal,
            require_complete=True,
            verify_tensor_hashes=True,
        )
