#!/usr/bin/env python3
"""Validate and summarize GLM-5 EXL3 projection quality evidence.

This tool is independent of GPTQModel.  It authenticates the immutable plan,
projection manifests, packed payloads, and append-only error journal; requires
complete routed-expert coverage by default; checks the arithmetic and
shape invariants in every quantizer report; and writes a content-bound summary.
It deliberately does not set a model-quality threshold: held-out end-to-end
qualification remains a separate release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO_ROOT / "quantization"))
from glm52_execution_upgrade import (  # noqa: E402
    EXECUTION_UPGRADE_FILENAME,
    ExecutionUpgradeError,
    read_execution_upgrade_chain,
)


SCHEMA = "glmrt-glm52-exl3-quant-evidence-validation-v2"
GLM53_SCHEMA = "glmrt-glm5-exl3-quant-evidence-validation-v1"
PLAN_SCHEMAS = {
    "glmrt-glm52-gptqmodel-plan-v1",
    "glmrt-glm52-gptqmodel-plan-v2",
}
CHECKPOINT_SCHEMA = "ds4rt.exl3-projection-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
METRICS_SCHEMA = "gptqmodel.exl3-trellis-error"
FIRST_LAYER = 3
LAST_LAYER = 77
EXPERTS = 256
PROJECTIONS = {
    "gate_proj": "w1",
    "down_proj": "w2",
    "up_proj": "w3",
}
EXPECTED_PROJECTIONS = (LAST_LAYER - FIRST_LAYER + 1) * EXPERTS * 3
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
MODULE_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\.mlp\.experts\."
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\Z"
)
MCG_SHA256 = "ade4fb124dda0f3537386cdd4a3cdcea3a223d386e506a4be89394bb33ee13fe"


class EvidenceValidationError(RuntimeError):
    """The projection evidence cannot be accepted."""


@dataclass(frozen=True)
class QuantEvidenceContract:
    report_schema: str
    release: str
    recipe: str
    source_format: str
    bits: int
    first_layer: int = FIRST_LAYER
    last_layer: int = LAST_LAYER
    experts: int = EXPERTS

    @property
    def expected_projections(self) -> int:
        return (self.last_layer - self.first_layer + 1) * self.experts * len(
            PROJECTIONS
        )

    @property
    def expected_experts(self) -> int:
        return (self.last_layer - self.first_layer + 1) * self.experts

    @property
    def scope(self) -> str:
        return (
            f"{self.release}-base-routed-experts-layers-"
            f"{self.first_layer}-through-{self.last_layer}"
        )


def quant_evidence_contract(plan: dict[str, Any]) -> QuantEvidenceContract:
    schema = plan.get("schema")
    recipe = plan.get("recipe")
    source = plan.get("source")
    if schema in PLAN_SCHEMAS and recipe in {
        None,
        "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1",
    }:
        return QuantEvidenceContract(
            report_schema=SCHEMA,
            release="glm-5.2",
            recipe="glm52_exl3_trellis_3bpw_calibrated_natural_route_v1",
            source_format="bf16",
            bits=3,
        )
    if (
        schema == "glmrt-glm5-gptqmodel-plan-v3"
        and recipe == "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1"
        and isinstance(source, dict)
        and source.get("release") == "glm-5.3"
        and source.get("format") == "fp8-e4m3-block128x128-dynamic"
    ):
        geometry = source.get("geometry")
        if not isinstance(geometry, dict) or any(
            geometry.get(field) != expected
            for field, expected in {
                "first_target_layer": FIRST_LAYER,
                "last_target_layer": LAST_LAYER,
                "n_routed_experts": EXPERTS,
                "mtp_layer_index": 78,
            }.items()
        ):
            raise EvidenceValidationError(
                "GLM-5.3 quantization plan has the wrong routed-expert geometry"
            )
        return QuantEvidenceContract(
            report_schema=GLM53_SCHEMA,
            release="glm-5.3",
            recipe=recipe,
            source_format="fp8-e4m3-block128x128-dynamic",
            bits=4,
        )
    raise EvidenceValidationError("unsupported GLM-5 EXL3 quantization plan")


def canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as error:
        raise EvidenceValidationError("evidence contains a non-finite value") from error


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceValidationError(f"not a regular JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"JSON value is not an object: {path}")
    return value


def bound_body(record: dict[str, Any], digest_field: str, label: str) -> dict[str, Any]:
    digest = record.get(digest_field)
    body = {key: value for key, value in record.items() if key != digest_field}
    if not isinstance(digest, str) or sha256_bytes(canonical_json(body)) != digest:
        raise EvidenceValidationError(f"{label} content digest is invalid")
    return body


def finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise EvidenceValidationError(f"{label} is outside its valid range")
    return result


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise EvidenceValidationError(f"{label} is not a positive integer")
    return value


def close(
    actual: float,
    expected: float,
    label: str,
    *,
    rel_tol: float = 1.0e-10,
    abs_tol: float = 1.0e-14,
) -> None:
    if not math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=abs_tol):
        raise EvidenceValidationError(
            f"{label} is inconsistent: {actual!r} != {expected!r}"
        )


def validate_plan(path: Path) -> dict[str, Any]:
    plan = json_object(path)
    bound_body(plan, "plan_sha256", "quantization plan")
    contract = quant_evidence_contract(plan)
    checkpoint = plan.get("projection_checkpoint")
    provenance = plan.get("ledger_provenance")
    family = provenance.get("family_join") if isinstance(provenance, dict) else None
    numerics = family.get("quantizer_numerics") if isinstance(family, dict) else None
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("contract") != "ds4rt.exl3-projection-checkpoint-v1"
        or not isinstance(checkpoint.get("root"), str)
        or not isinstance(plan.get("run_state_dir"), str)
        or not isinstance(family, dict)
        or family.get("bits") != contract.bits
        or family.get("codebook") != "mcg"
        or not isinstance(numerics, dict)
    ):
        raise EvidenceValidationError("quantization plan has the wrong EXL3 contract")
    exl3 = plan.get("exl3")
    if plan.get("schema") == "glmrt-glm5-gptqmodel-plan-v3" and (
        not isinstance(exl3, dict)
        or exl3.get("bits") != contract.bits
        or exl3.get("codebook") != "mcg"
    ):
        raise EvidenceValidationError("quantization plan EXL3 declaration is invalid")
    finite_number(numerics.get("sigma_reg"), "plan sigma_reg", minimum=1.0e-300)
    return plan


def read_journal(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        raise EvidenceValidationError(f"error journal is unavailable: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open("rb") as source:
        for line_number, line in enumerate(source, 1):
            if not line.endswith(b"\n"):
                raise EvidenceValidationError("error journal ends with a partial record")
            try:
                value = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise EvidenceValidationError(
                    f"error journal line {line_number} is invalid"
                ) from error
            if not isinstance(value, dict):
                raise EvidenceValidationError(
                    f"error journal line {line_number} is not an object"
                )
            body = bound_body(value, "record_sha256", f"error journal line {line_number}")
            digest = value["record_sha256"]
            if body.get("record_kind") != "projection" or digest in records:
                raise EvidenceValidationError("error journal contains an invalid record")
            records[digest] = body
    return records


def checkpoint_paths(root: Path) -> list[tuple[str, Path, Path]]:
    if not root.is_dir() or root.is_symlink():
        raise EvidenceValidationError("projection checkpoint root is not a directory")
    manifests: dict[str, Path] = {}
    tensors: dict[str, Path] = {}
    for first in root.iterdir():
        if not first.is_dir() or first.is_symlink() or not re.fullmatch(r"[0-9a-f]{2}", first.name):
            raise EvidenceValidationError("checkpoint root contains an unsafe entry")
        for second in first.iterdir():
            if (
                not second.is_dir()
                or second.is_symlink()
                or not re.fullmatch(r"[0-9a-f]{2}", second.name)
            ):
                raise EvidenceValidationError("checkpoint root contains an unsafe prefix")
            for path in second.iterdir():
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or path.suffix not in {".json", ".safetensors"}
                ):
                    raise EvidenceValidationError("checkpoint store contains an unsafe file")
                digest = path.name.removesuffix(path.suffix)
                if HEX64.fullmatch(digest) is None or digest[:4] != first.name + second.name:
                    raise EvidenceValidationError("checkpoint is under the wrong prefix")
                target = manifests if path.suffix == ".json" else tensors
                if digest in target:
                    raise EvidenceValidationError("checkpoint store contains a duplicate")
                target[digest] = path
    if set(manifests) != set(tensors):
        raise EvidenceValidationError("checkpoint store contains an incomplete pair")
    return [(digest, manifests[digest], tensors[digest]) for digest in sorted(manifests)]


def expected_tensor_specs(
    input_shape: list[int], bits: int = 3
) -> dict[str, tuple[list[int], str, int]]:
    if (
        len(input_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in input_shape)
        or any(value <= 0 or value % 16 for value in input_shape)
        or bits not in {3, 4}
    ):
        raise EvidenceValidationError("projection input weight shape is invalid")
    rows, columns = input_shape
    return {
        "trellis": (
            [rows // 16, columns // 16, 16 * bits],
            "torch.int16",
            rows * columns * bits // 8,
        ),
        "suh": ([rows], "torch.float16", rows * 2),
        "svh": ([columns], "torch.float16", columns * 2),
        "mcg": ([], "torch.int32", 4),
    }


def validate_tensor_specs(
    specs: Any, input_shape: list[int], encoded_bytes: Any, *, bits: int = 3
) -> int:
    expected = expected_tensor_specs(input_shape, bits)
    if not isinstance(specs, dict) or set(specs) != set(expected):
        raise EvidenceValidationError("checkpoint tensor set is invalid")
    total = 0
    for name, (shape, dtype, byte_count) in expected.items():
        spec = specs[name]
        numel = math.prod(shape)
        if (
            not isinstance(spec, dict)
            or spec.get("shape") != shape
            or spec.get("dtype") != dtype
            or spec.get("numel") != numel
            or spec.get("bytes") != byte_count
            or HEX64.fullmatch(str(spec.get("sha256", ""))) is None
            or (name == "mcg" and spec.get("sha256") != MCG_SHA256)
        ):
            raise EvidenceValidationError(f"checkpoint tensor {name} is invalid")
        total += byte_count
    if encoded_bytes != total:
        raise EvidenceValidationError("checkpoint encoded byte count is invalid")
    return total


def validate_metrics(
    metrics: Any,
    *,
    sample_count: int,
    sigma_reg: float,
    input_shape: list[int],
) -> dict[str, float]:
    if not isinstance(metrics, dict):
        raise EvidenceValidationError("quantizer metrics are absent")
    if (
        metrics.get("schema") != METRICS_SCHEMA
        or metrics.get("schema_version") != 1
        or metrics.get("quantizer_path") != "hessian_ldlq"
        or metrics.get("reported_metric_kind") != "hessian_weighted_relative_error"
        or metrics.get("hessian_metric_status") != "ok"
        or metrics.get("hessian_sample_count") != sample_count
        or metrics.get("hessian_regularization_sigma") != sigma_reg
        or metrics.get("hessian_numerical_contract")
        != "signed-block-hadamard-congruence-fp64-v1"
        or metrics.get("hessian_transform_compute_dtype") != "torch.float64"
        or metrics.get("hessian_storage_dtype") != "torch.float32"
        or metrics.get("hessian_regularization_placement")
        != "before-fp64-congruence"
        or metrics.get("hessian_symmetry_restoration") != "mean-with-transpose-fp64"
        or not isinstance(metrics.get("apply_out_scales"), bool)
    ):
        raise EvidenceValidationError("quantizer numerical contract is invalid")

    numerator = finite_number(
        metrics.get("hessian_weighted_error_numerator"),
        "Hessian error numerator",
        minimum=0.0,
    )
    denominator = finite_number(
        metrics.get("hessian_weighted_reference_denominator"),
        "Hessian reference denominator",
        minimum=1.0e-300,
    )
    relative = finite_number(
        metrics.get("hessian_weighted_relative_error"),
        "Hessian relative error",
        minimum=0.0,
    )
    close(relative, numerator / denominator, "Hessian relative error")
    close(
        finite_number(metrics.get("reported_metric_value"), "reported metric", minimum=0.0),
        relative,
        "reported metric",
    )
    finite_number(
        metrics.get("hessian_regularization_diagonal_addend"),
        "Hessian regularization",
        minimum=1.0e-300,
    )
    finite_number(
        metrics.get("hessian_symmetry_correction_max_abs"),
        "Hessian symmetry correction",
        minimum=0.0,
    )
    scale = finite_number(metrics.get("selected_global_scale"), "global scale", minimum=1.0e-300)
    finite_number(metrics.get("scale_search_mse"), "scale-search MSE", minimum=0.0)

    reconstruction = metrics.get("reconstruction")
    element_count = math.prod(input_shape)
    if (
        not isinstance(reconstruction, dict)
        or reconstruction.get("domain") != "regularized_exl3_search_space"
        or reconstruction.get("shape") != input_shape
        or reconstruction.get("element_count") != element_count
        or reconstruction.get("reference_finite") is not True
        or reconstruction.get("error_finite") is not True
        or reconstruction.get("tile_shape") != [16, 16]
        or reconstruction.get("tile_count") != element_count // 256
    ):
        raise EvidenceValidationError("reconstruction evidence is invalid")
    error_sum = finite_number(reconstruction.get("error_sum_sq"), "reconstruction SSE", minimum=0.0)
    reference_sum = finite_number(
        reconstruction.get("reference_sum_sq"), "reconstruction reference SSE", minimum=1.0e-300
    )
    mse = finite_number(reconstruction.get("mse"), "reconstruction MSE", minimum=0.0)
    nmse = finite_number(reconstruction.get("nmse"), "reconstruction NMSE", minimum=0.0)
    frobenius = finite_number(
        reconstruction.get("relative_frobenius"), "relative Frobenius error", minimum=0.0
    )
    close(mse, error_sum / element_count, "reconstruction MSE")
    close(nmse, error_sum / reference_sum, "reconstruction NMSE")
    close(frobenius, math.sqrt(nmse), "relative Frobenius error")
    close(
        finite_number(reconstruction.get("tile_sse_sum"), "tile SSE sum", minimum=0.0),
        error_sum,
        "tile SSE sum",
        # These are independent FP32 reductions over millions of elements.
        rel_tol=1.0e-6,
        abs_tol=1.0e-5,
    )
    finite_number(reconstruction.get("mean_abs_error"), "mean absolute error", minimum=0.0)
    finite_number(reconstruction.get("max_abs_error"), "maximum absolute error", minimum=0.0)
    finite_number(reconstruction.get("tile_sse_max"), "maximum tile SSE", minimum=0.0)
    percentiles = reconstruction.get("tile_sse_percentiles")
    if not isinstance(percentiles, dict) or set(percentiles) != {"p50", "p90", "p99", "p99_9"}:
        raise EvidenceValidationError("tile SSE percentiles are invalid")
    ordered = [
        finite_number(percentiles[name], f"tile SSE {name}", minimum=0.0)
        for name in ("p50", "p90", "p99", "p99_9")
    ]
    if ordered != sorted(ordered) or ordered[-1] > float(reconstruction["tile_sse_max"]):
        raise EvidenceValidationError("tile SSE percentiles are inconsistent")
    worst = reconstruction.get("worst_tiles")
    if not isinstance(worst, list) or not worst:
        raise EvidenceValidationError("worst-tile evidence is absent")
    for tile in worst:
        if (
            not isinstance(tile, dict)
            or isinstance(tile.get("row"), bool)
            or not isinstance(tile.get("row"), int)
            or tile["row"] < 0
            or isinstance(tile.get("column"), bool)
            or not isinstance(tile.get("column"), int)
            or tile["column"] < 0
        ):
            raise EvidenceValidationError("worst-tile evidence is invalid")
        finite_number(tile.get("sse"), "worst-tile SSE", minimum=0.0)
    return {
        "hessian_numerator": numerator,
        "hessian_denominator": denominator,
        "hessian_relative_error": relative,
        "reconstruction_nmse": nmse,
        "relative_frobenius": frobenius,
        "selected_global_scale": scale,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    materialized = list(values)
    if not materialized:
        raise EvidenceValidationError("cannot summarize an empty metric")
    return {
        "count": len(materialized),
        "minimum": min(materialized),
        "p50": percentile(materialized, 0.50),
        "p90": percentile(materialized, 0.90),
        "p99": percentile(materialized, 0.99),
        "p99_9": percentile(materialized, 0.999),
        "maximum": max(materialized),
        "mean": math.fsum(materialized) / len(materialized),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    def one(group: list[dict[str, Any]]) -> dict[str, Any]:
        numerator = math.fsum(item["hessian_numerator"] for item in group)
        denominator = math.fsum(item["hessian_denominator"] for item in group)
        return {
            "projection_count": len(group),
            "aggregate_hessian_weighted_relative_error": numerator / denominator,
            "hessian_weighted_relative_error": distribution(
                item["hessian_relative_error"] for item in group
            ),
            "reconstruction_nmse": distribution(item["reconstruction_nmse"] for item in group),
            "relative_frobenius": distribution(item["relative_frobenius"] for item in group),
            "selected_global_scale": distribution(item["selected_global_scale"] for item in group),
        }

    by_projection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_route_evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_projection[record["projection"]].append(record)
        by_layer[record["layer"]].append(record)
        by_route_evidence[record["route_evidence"]].append(record)
    return {
        "percentile_method": "linear-r7",
        "global": one(records),
        "by_projection": {name: one(by_projection[name]) for name in sorted(by_projection)},
        "by_route_evidence": {
            name: one(by_route_evidence[name]) for name in sorted(by_route_evidence)
        },
        "by_layer": [
            {"layer": layer, **one(by_layer[layer])} for layer in sorted(by_layer)
        ],
    }


def validate_evidence(
    *,
    plan_path: Path,
    checkpoint_root: Path,
    journal_path: Path,
    require_complete: bool,
    verify_tensor_hashes: bool,
    live_journal_snapshot: bool = False,
) -> dict[str, Any]:
    if require_complete and live_journal_snapshot:
        raise EvidenceValidationError(
            "a live journal snapshot cannot prove complete release evidence"
        )
    plan_path = plan_path.expanduser().resolve(strict=True)
    checkpoint_root = checkpoint_root.expanduser().resolve(strict=True)
    journal_path = journal_path.expanduser().resolve(strict=True)
    plan = validate_plan(plan_path)
    contract = quant_evidence_contract(plan)
    planned_root = Path(plan["projection_checkpoint"]["root"]).expanduser().resolve()
    planned_journal = (
        Path(plan["run_state_dir"]).expanduser().resolve()
        / ".glmrt-exl3-error-journal.jsonl"
    )
    if checkpoint_root != planned_root or journal_path != planned_journal:
        raise EvidenceValidationError("evidence paths differ from the immutable plan")
    family = plan["ledger_provenance"]["family_join"]
    sigma_reg = float(family["quantizer_numerics"]["sigma_reg"])
    run_state = Path(plan["run_state_dir"]).expanduser().resolve()
    upgrade_chain: tuple[dict[str, Any], ...] = ()
    if (run_state / EXECUTION_UPGRADE_FILENAME).exists():
        try:
            upgrade_chain = read_execution_upgrade_chain(
                run_state,
                parent_plan_sha256=plan["plan_sha256"],
            )
        except ExecutionUpgradeError as error:
            raise EvidenceValidationError(str(error)) from error
    upgrades_by_digest = {
        record["upgrade_sha256"]: record for record in upgrade_chain
    }
    projection_execution_counts: dict[str, int] = defaultdict(int)
    journal = read_journal(journal_path)
    pairs = checkpoint_paths(checkpoint_root)
    if not pairs:
        raise EvidenceValidationError("projection checkpoint store is empty")

    modules: set[str] = set()
    checkpoint_modules: set[str] = set()
    ledger_digests: set[str] = set()
    experts: dict[tuple[int, int], tuple[Any, Any, int]] = {}
    records: list[dict[str, Any]] = []
    inventory = hashlib.sha256()
    packed_bytes = 0
    encoded_bytes = 0
    post_snapshot_checkpoint_count = 0
    for request_digest, manifest_path, tensor_path in pairs:
        manifest = json_object(manifest_path)
        bound_body(manifest, "manifest_sha256", f"checkpoint {request_digest}")
        request = manifest.get("request")
        result = manifest.get("result")
        if not isinstance(request, dict) or not isinstance(result, dict):
            raise EvidenceValidationError("checkpoint request or result is absent")
        request_body = {key: value for key, value in request.items() if key != "request_sha256"}
        tensor_digest = manifest.get("tensor_sha256")
        if (
            manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or manifest.get("request_sha256") != request_digest
            or request.get("request_sha256") != request_digest
            or sha256_bytes(canonical_json(request_body)) != request_digest
            or manifest.get("tensor_file") != tensor_path.name
            or HEX64.fullmatch(str(tensor_digest or "")) is None
        ):
            raise EvidenceValidationError("checkpoint manifest contract is invalid")
        if verify_tensor_hashes and sha256_file(tensor_path) != tensor_digest:
            raise EvidenceValidationError(f"packed checkpoint failed hashing: {tensor_path}")

        module = request.get("module")
        match = MODULE_RE.fullmatch(module) if isinstance(module, str) else None
        if match is None or module in checkpoint_modules:
            raise EvidenceValidationError("checkpoint module identity is invalid")
        checkpoint_modules.add(module)
        layer = int(match.group("layer"))
        expert = int(match.group("expert"))
        projection_name = match.group("projection")
        projection = PROJECTIONS[projection_name]
        if not contract.first_layer <= layer <= contract.last_layer or not (
            0 <= expert < contract.experts
        ):
            raise EvidenceValidationError("checkpoint module is outside the GLM-5 scope")
        route = request.get("route_evidence")
        recovery = request.get("zero_route_recovery")
        sample_count = positive_int(request.get("sample_count"), "projection sample count")
        input_weight = request.get("input_weight")
        quantizer_contract = request.get("quantizer_contract")
        if (
            request.get("schema") != CHECKPOINT_SCHEMA
            or request.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or request.get("processor_layer_index") != layer
            or request.get("family_join") != family
            or not isinstance(input_weight, dict)
            or input_weight.get("dtype") != "torch.float32"
            or not isinstance(input_weight.get("shape"), list)
            or not isinstance(quantizer_contract, dict)
            or quantizer_contract.get("bits") != contract.bits
            or quantizer_contract.get("codebook") != "mcg"
            or not isinstance(route, dict)
            or route.get("logical_layer") != layer
            or route.get("expert") != expert
            or route.get("block_namespace") != "base"
        ):
            raise EvidenceValidationError(f"projection request is invalid: {module}")
        previous_expert = experts.setdefault(
            (layer, expert), (route, recovery, sample_count)
        )
        if previous_expert != (route, recovery, sample_count):
            raise EvidenceValidationError("expert route evidence differs by projection")

        ledger = result.get("ledger_record")
        metrics = result.get("quantizer_metrics")
        if not isinstance(ledger, dict):
            raise EvidenceValidationError("checkpoint has no error-ledger record")
        ledger_digest = sha256_bytes(canonical_json(ledger))
        if live_journal_snapshot and ledger_digest not in journal:
            # Projection checkpoints become durable immediately before their
            # corresponding append-only journal commit.  When inspecting a
            # running quantizer, authenticate the exact journal frontier read
            # above and report (but do not claim) any newer checkpoint pairs.
            post_snapshot_checkpoint_count += 1
            continue
        if ledger_digest in ledger_digests or journal.get(ledger_digest) != ledger:
            raise EvidenceValidationError("checkpoint ledger record is absent or duplicated")
        ledger_digests.add(ledger_digest)
        modules.add(module)
        if (
            ledger.get("schema") != "ds4rt.exl3-error-ledger"
            or ledger.get("schema_version") != 1
            or ledger.get("record_kind") != "projection"
            or ledger.get("module") != module
            or ledger.get("logical_layer") != layer
            or ledger.get("processor_layer_index") != layer
            or ledger.get("expert") != expert
            or ledger.get("projection") != projection
            or ledger.get("bits") != contract.bits
            or ledger.get("codebook") != "mcg"
            or ledger.get("sample_count") != sample_count
            or ledger.get("route_evidence") != route
            or ledger.get("zero_route_recovery") != recovery
            or ledger.get("quantizer_metrics") != metrics
            or result.get("proxy_error") != metrics.get("hessian_weighted_relative_error")
        ):
            raise EvidenceValidationError(f"projection ledger is invalid: {module}")
        provenance = ledger.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("family_join") != family:
            raise EvidenceValidationError("projection ledger family differs from the plan")
        provenance_run = provenance.get("run")
        execution_reference = (
            provenance_run.get("execution_upgrade")
            if isinstance(provenance_run, dict)
            else None
        )
        if execution_reference is None:
            projection_execution_counts["parent-plan"] += 1
        else:
            digest = (
                execution_reference.get("upgrade_sha256")
                if isinstance(execution_reference, dict)
                else None
            )
            upgrade = upgrades_by_digest.get(digest)
            if (
                not isinstance(upgrade, dict)
                or execution_reference.get("schema") != upgrade.get("schema")
                or execution_reference.get("parent_plan_sha256")
                != upgrade.get("parent_plan_sha256")
                or execution_reference.get("upgraded_execution")
                != upgrade.get("upgraded_execution")
                or execution_reference.get("resume_state")
                != upgrade.get("resume_state")
            ):
                raise EvidenceValidationError(
                    "projection ledger execution upgrade is invalid"
                )
            projection_execution_counts[digest] += 1
        logical_bytes = validate_tensor_specs(
            manifest.get("tensors"),
            input_weight["shape"],
            ledger.get("encoded_bytes"),
            bits=contract.bits,
        )
        values = validate_metrics(
            metrics,
            sample_count=sample_count,
            sigma_reg=sigma_reg,
            input_shape=input_weight["shape"],
        )
        stat_bytes = tensor_path.stat().st_size
        packed_bytes += stat_bytes
        encoded_bytes += logical_bytes
        inventory.update(
            canonical_json(
                {
                    "request_sha256": request_digest,
                    "manifest_sha256": manifest["manifest_sha256"],
                    "tensor_sha256": tensor_digest,
                    "tensor_file_bytes": stat_bytes,
                    "record_sha256": ledger_digest,
                }
            )
            + b"\n"
        )
        records.append(
            {
                "layer": layer,
                "expert": expert,
                "projection": projection,
                "route_evidence": (
                    "zero-route-recovery" if recovery is not None else "natural-route"
                ),
                **values,
            }
        )

    if set(journal) != ledger_digests:
        raise EvidenceValidationError("error journal and checkpoint inventory differ")
    expected_modules = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(contract.first_layer, contract.last_layer + 1)
        for expert in range(contract.experts)
        for projection in PROJECTIONS
    }
    if not modules <= expected_modules or (require_complete and modules != expected_modules):
        missing = len(expected_modules - modules)
        raise EvidenceValidationError(
            f"projection coverage is incomplete or invalid: missing={missing}"
        )
    expert_projection_counts: dict[tuple[int, int], int] = defaultdict(int)
    for record in records:
        expert_projection_counts[(record["layer"], record["expert"])] += 1
    complete_experts = sum(value == 3 for value in expert_projection_counts.values())
    recovered_experts = sum(recovery is not None for _route, recovery, _count in experts.values())
    report = {
        "schema": contract.report_schema,
        "status": (
            "accepted"
            if require_complete
            else (
                "partial-live-snapshot-accepted"
                if live_journal_snapshot
                else "partial-accepted"
            )
        ),
        "scope": contract.scope,
        "quality_scope": "projection-quantizer-evidence-not-end-to-end-model-quality",
        "plan": {
            "path": os.fspath(plan_path),
            "plan_sha256": plan["plan_sha256"],
        },
        "inputs": {
            "projection_checkpoint_root": os.fspath(checkpoint_root),
            "error_journal": os.fspath(journal_path),
        },
        "coverage": {
            "expected_projection_count": contract.expected_projections,
            "projection_count": len(records),
            "expected_expert_count": contract.expected_experts,
            "observed_expert_count": len(experts),
            "complete_expert_count": complete_experts,
            "recovered_expert_count": recovered_experts,
            "layers": sorted({record["layer"] for record in records}),
        },
        "integrity": {
            "tensor_payload_hashes_verified": verify_tensor_hashes,
            "checkpoint_inventory_sha256": inventory.hexdigest(),
            "packed_tensor_file_bytes": packed_bytes,
            "logical_encoded_tensor_bytes": encoded_bytes,
            "journal_record_count": len(journal),
            "live_journal_snapshot": live_journal_snapshot,
            "checkpoint_pairs_seen": len(pairs),
            "post_snapshot_checkpoint_count": post_snapshot_checkpoint_count,
        },
        "execution_upgrade": (
            {
                "active_upgrade_sha256": upgrade_chain[0]["upgrade_sha256"],
                "chain": [record["upgrade_sha256"] for record in upgrade_chain],
                "projection_records": dict(
                    sorted(projection_execution_counts.items())
                ),
            }
            if upgrade_chain
            else None
        ),
        "metrics": summarize(records),
    }
    report["report_sha256"] = sha256_bytes(canonical_json(report))
    return report


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(canonical_json(value) + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--projection-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--error-journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="accept a valid projection subset for an in-progress diagnostic",
    )
    parser.add_argument(
        "--skip-tensor-hashes",
        action="store_true",
        help="skip packed payload hashing (not acceptable for final release)",
    )
    parser.add_argument(
        "--live-journal-snapshot",
        action="store_true",
        help=(
            "validate the exact journal frontier of a running quantizer and "
            "report newer durable checkpoints; requires --allow-incomplete"
        ),
    )
    args = parser.parse_args()
    if args.live_journal_snapshot and not args.allow_incomplete:
        parser.error("--live-journal-snapshot requires --allow-incomplete")
    report = validate_evidence(
        plan_path=args.plan,
        checkpoint_root=args.projection_checkpoint_dir,
        journal_path=args.error_journal,
        require_complete=not args.allow_incomplete,
        verify_tensor_hashes=not args.skip_tensor_hashes,
        live_journal_snapshot=args.live_journal_snapshot,
    )
    atomic_json(args.output, report)
    print(
        f"Validated {report['coverage']['projection_count']} EXL3 projections; "
        f"aggregate Hessian-weighted relative error "
        f"{report['metrics']['global']['aggregate_hessian_weighted_relative_error']:.8g}; "
        f"report {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
