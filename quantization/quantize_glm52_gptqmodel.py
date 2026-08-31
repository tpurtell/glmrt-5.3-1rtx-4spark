#!/usr/bin/env python3
"""Quantize supported GLM-5 base routed experts to calibrated EXL3."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any

from glm52_layer_boundary_store import (
    BOUNDARY_CONTRACT,
    BOUNDARY_SCHEMA,
    BOUNDARY_SCHEMA_VERSION,
    Glm52LayerBoundaryController,
    Glm52LayerBoundaryStore,
    LayerBoundaryStop,
    PAYLOAD_HASH_ALGORITHM as BOUNDARY_PAYLOAD_HASH_ALGORITHM,
    sha256_file,
)
from glm52_execution_upgrade import (
    EXECUTION_UPGRADE_FILENAME,
    EXECUTION_UPGRADE_HISTORY_DIRNAME,
    EXECUTION_UPGRADE_SCHEMA,
    ExecutionUpgradeError,
    read_execution_upgrade,
)
from preflight import report_identity_sha256

LEGACY_PLAN_SCHEMA = "glmrt-glm52-gptqmodel-plan-v1"
GLM52_PLAN_SCHEMA = "glmrt-glm52-gptqmodel-plan-v2"
PLAN_SCHEMA = "glmrt-glm5-gptqmodel-plan-v3"
SUPPORTED_PLAN_SCHEMAS = frozenset(
    (LEGACY_PLAN_SCHEMA, GLM52_PLAN_SCHEMA, PLAN_SCHEMA)
)
LEGACY_RUN_SCHEMA = "glmrt-glm52-gptqmodel-run-v1"
RUN_SCHEMA = "glmrt-glm5-gptqmodel-run-v2"
LEGACY_ARTIFACT_MANIFEST_SCHEMA = "glmrt-glm52-gptqmodel-artifact-v1"
ARTIFACT_MANIFEST_SCHEMA = "glmrt-glm5-gptqmodel-artifact-v2"
CALIBRATION_MANIFEST_SCHEMA = "ds4rt-flash-exl3-calibration-corpus-v3"
ROUTE_QUALIFICATION_INLINE = "inline-full-corpus"
GLM52_NATURAL_ROUTE_RECIPE = (
    "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1"
)
GLM53_NATURAL_ROUTE_RECIPE = (
    "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1"
)
# Compatibility alias used by the existing K3 evidence and tests.
NATURAL_ROUTE_RECIPE = GLM52_NATURAL_ROUTE_RECIPE
ROUTE_EVIDENCE_CONTRACT = "ds4rt.exl3-natural-route"
ZERO_ROUTE_RECOVERY_CONTRACT = "ds4rt.exl3-zero-route-recovery"
ZERO_ROUTE_RECOVERY_TRIGGER = "natural-route-count-below-1024"
ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE = "same-fixed-calibration-selection"
ZERO_ROUTE_RECOVERY_CAPTURE_METHOD = (
    "direct-expert-router-ranks-9-16-then-identity-residual"
)
ZERO_ROUTE_RECOVERY_SELECTION_POLICY = (
    "rank-ascending-then-fixed-replay-order-v1"
)
ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN = 9
ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX = 16
ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT = 1024
ZERO_ROUTE_RECOVERY_SELECTION_CAP = 1024
ZERO_ROUTE_RECOVERY_IDENTITY_POLICY = (
    "normalized-2i-residual-to-effective-count-1024-v2"
)
PROJECTION_CHECKPOINT_CONTRACT = "ds4rt.exl3-projection-checkpoint-v1"
PROJECTION_CHECKPOINT_SEED_CONTRACT = (
    "glmrt.exl3-projection-checkpoint-seed-v1"
)
PROJECTION_FAMILY_ORCHESTRATION_FIELDS = frozenset(
    ("image_digest", "preflight_sha256", "quantization_toolchain")
)
PROJECTION_SEED_GPTQMODEL_ORCHESTRATION_TRANSITIONS = frozenset(
    {
        (
            "343290cddb72329a4bb3d1ee603ef579a3c488bf",
            "bedfe6dafbca9eb974736260c2cb1ee307f7ed84ff104a522f55cc662b7867b6",
            "fde23e4e3165843a9dfa74d1a8463e0375b43d42",
            "2fac9376f978a33e105c6f3382d0ad75535708a787e00ecf61029d47ba3bca46",
        )
    }
)
BASE_EXPERT_PATTERN = (
    r"^model\.layers\.(?:[3-9]|[1-6][0-9]|7[0-7])\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
EXL3_SEED = 787
EXL3_SIGMA_REG = 0.025
EXL3_MCG_MULTIPLIER = 0xCBAC1FED
EXL3_HESSIAN_CAPTURE_CONTRACT = "raw-xtx-sum-fp32-v1"
EXL3_HESSIAN_NUMERICAL_CONTRACT = "signed-block-hadamard-congruence-fp64-v1"
EXL3_HESSIAN_SYMMETRY_CONTRACT = "mean-with-transpose-fp64"
REVISION_RE = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
HF_BLOB_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
TOKENIZER_SOURCE_FILES = ("tokenizer.json", "tokenizer_config.json")
EXACT_SOURCE_METADATA_FILES = (*TOKENIZER_SOURCE_FILES, "generation_config.json")
PREFLIGHT_ROLE_CONTRACTS = {
    "coordinator": ("linux/amd64", "120"),
    "expert": ("linux/arm64", "121"),
}
PLAN_FILENAME = "glmrt-gptqmodel-plan.json"
RUN_FILENAME = "glmrt-gptqmodel-run.json"
ARTIFACT_MANIFEST_FILENAME = "glmrt-gptqmodel-artifact.json"
ERROR_JOURNAL_FILENAME = ".glmrt-exl3-error-journal.jsonl"
DIRECT_STATE_PREFLIGHT_FILENAME = "lazy-direct-state-preflight.json"
STORAGE_PREFLIGHT_FILENAME = "storage-preflight.json"
PROJECTION_CHECKPOINT_DIRNAME = "projection-checkpoints"
ACTIVE_LAYER_SOURCE_DIRNAME = "active-layer-source"
EXPORT_STAGE_DIRNAME = "export-stage"
LAYER_BOUNDARY_DIRNAME = "layer-boundary"
CAPTURE_FRONTIER_DIRNAME = "layer-capture-frontier"
CAPTURE_BATCH_SPOOL_DIRNAME = "capture-batch-journal"
POST_QUANT_REPLAY_DIRNAME = "post-quant-replay"
EXLLAMAV3_JIT_DIRNAME = "jit/exllamav3"
JIT_CACHE_DIRNAME = "jit"
HOST_RSS_LIMIT_BYTES = 150 * 1024**3
CUDA_ALLOCATION_LIMIT_BYTES = 82 * 1024**3
MEMORY_TELEMETRY_INTERVAL_BATCHES = 64
CAPTURE_FRONTIER_CONTRACT = "ds4rt.exl3-capture-frontier-v1"
ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT = (
    "gptqmodel.exl3-router-candidate-capture-v3"
)
BOUNDARY_DIRECTORY_RE = re.compile(
    r"layer-(?P<layer>[0-9]{6})-(?P<digest>[0-9a-f]{16})\Z"
)
FORWARD_REPLICA_POLICY = "serialized-deepcopy-v1"
FORWARD_REPLICA_ENV = "GPTQMODEL_USE_TORCH_REPLICATE"
STORAGE_CONTRACT = "glmrt-glm5-exl3-storage-v2"
ARTIFACT_EXPORT_OVERHEAD_BYTES = 4 * 1024**3
PROJECTION_CHECKPOINT_OVERHEAD_BYTES = 2 * 1024**3
RUN_STATE_PEAK_BYTES = 128 * 1024**3
OFFLOAD_PEAK_BYTES = 12 * 1024**3
FILESYSTEM_FREE_FLOOR_BYTES = 32 * 1024**3


class LaunchError(RuntimeError):
    """The requested production quantization run is not reproducible."""


@contextmanager
def exllamav3_jit_cache_scope(root: Path):
    """Persist the content-fingerprinted EXL3 extension across containers."""

    variable = "GPTQMODEL_EXLLAMAV3_BUILD_ROOT"
    expected = os.fspath(root)
    previous = os.environ.get(variable)
    if previous not in {None, expected}:
        raise LaunchError(f"{variable} conflicts with the immutable run state")
    os.environ[variable] = expected
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def quantization_variant(source: dict[str, Any], bits: int) -> dict[str, str]:
    """Select only the two reviewed GLM source/EXL3 production pairings."""

    release = source.get("release")
    source_format = source.get("format")
    if (release, source_format, bits) == ("glm-5.2", "bf16", 3):
        return {
            "recipe": GLM52_NATURAL_ROUTE_RECIPE,
            "operator_contract": "glmrt-glm52-base-routed-exl3-k3-v1",
        }
    if (release, source_format, bits) == (
        "glm-5.3",
        "fp8-e4m3-block128x128-dynamic",
        4,
    ):
        return {
            "recipe": GLM53_NATURAL_ROUTE_RECIPE,
            "operator_contract": "glmrt-glm53-base-routed-exl3-k4-v1",
        }
    raise LaunchError(
        "supported production pairings are GLM-5.2 BF16 -> EXL3 K3 and "
        "GLM-5.3 block-FP8 -> EXL3 K4"
    )


def natural_route_recipe(bits: int) -> str:
    """Return a recipe by bitrate for compatibility with older callers."""

    if bits == 3:
        return GLM52_NATURAL_ROUTE_RECIPE
    if bits == 4:
        return GLM53_NATURAL_ROUTE_RECIPE
    raise LaunchError("supported GLM production recipes require EXL3 K3 or K4")


def artifact_contract_schemas(plan: dict[str, Any]) -> tuple[str, str]:
    if plan.get("schema") in {LEGACY_PLAN_SCHEMA, GLM52_PLAN_SCHEMA}:
        return LEGACY_RUN_SCHEMA, LEGACY_ARTIFACT_MANIFEST_SCHEMA
    return RUN_SCHEMA, ARTIFACT_MANIFEST_SCHEMA


@contextmanager
def capture_frontier_scope(root: Path):
    """Keep one exact recovery frontier active through target and dSpark."""

    variable = "GPTQMODEL_EXL3_CAPTURE_FRONTIER"
    expected = os.fspath(root)
    previous = os.environ.get(variable)
    if previous not in {None, expected}:
        raise LaunchError(f"{variable} conflicts with the immutable run state")
    os.environ[variable] = expected
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


@contextmanager
def capture_batch_spool_scope(root: Path):
    """Bind additive Hessian recovery records to the immutable run state."""

    variable = "GPTQMODEL_EXL3_CAPTURE_BATCH_SPOOL"
    expected = os.fspath(root)
    previous = os.environ.get(variable)
    if previous not in {None, expected}:
        raise LaunchError(f"{variable} conflicts with the immutable run state")
    os.environ[variable] = expected
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


@contextmanager
def memory_safety_scope(contract: dict[str, Any]):
    """Expose immutable fail-closed capture limits to GPTQModel."""

    values = {
        "GPTQMODEL_EXL3_HOST_RSS_LIMIT_BYTES": contract["host_rss_limit_bytes"],
        "GPTQMODEL_EXL3_CUDA_ALLOCATION_LIMIT_BYTES": contract[
            "cuda_allocation_limit_bytes"
        ],
        "GPTQMODEL_EXL3_MEMORY_TELEMETRY_INTERVAL_BATCHES": contract[
            "telemetry_interval_batches"
        ],
    }
    previous = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        expected = str(value)
        if previous[name] not in {None, expected}:
            raise LaunchError(f"{name} conflicts with the immutable run state")
        os.environ[name] = expected
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def forward_replica_scope(contract: dict[str, Any]):
    """Bind replay cloning before GPTQModel imports its looper helpers."""

    if contract != {
        "policy": FORWARD_REPLICA_POLICY,
        "torch_parallel_replicate": False,
    }:
        raise LaunchError("forward replay clone policy is not supported")
    previous = os.environ.get(FORWARD_REPLICA_ENV)
    if previous not in {None, "0"}:
        raise LaunchError(
            f"{FORWARD_REPLICA_ENV} conflicts with the immutable run state"
        )
    os.environ[FORWARD_REPLICA_ENV] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(FORWARD_REPLICA_ENV, None)
        else:
            os.environ[FORWARD_REPLICA_ENV] = previous


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LaunchError(f"expected a JSON object in {path}")
    return value


def shard_identity(snapshot: Path, name: str) -> dict[str, Any]:
    path = snapshot / name
    if path.is_symlink():
        link = Path(os.readlink(path))
        if link.is_absolute() or link.parts[:3] != ("..", "..", "blobs"):
            raise LaunchError(
                f"source snapshot shard is not a canonical Hugging Face blob: {path}"
            )
        if len(link.parts) != 4 or SHA256_RE.fullmatch(link.parts[3]) is None:
            raise LaunchError(
                f"source snapshot shard has no SHA-256 blob identity: {path}"
            )
        blob_root = (snapshot.parent.parent / "blobs").resolve(strict=True)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LaunchError(
                f"source snapshot has a broken shard link: {path}"
            ) from exc
        expected_blob = blob_root / link.parts[3]
        if (
            resolved != expected_blob
            or expected_blob.is_symlink()
            or not expected_blob.is_file()
        ):
            raise LaunchError(
                f"source snapshot shard escapes its Hugging Face blob store: {path}"
            )
        return {
            "name": name,
            "bytes": resolved.stat().st_size,
            "hf_blob_sha256": link.parts[3],
        }
    if not path.is_file():
        raise LaunchError(f"source snapshot is missing regular shard {path}")
    return {"name": name, "bytes": path.stat().st_size}


def source_metadata_identity(snapshot: Path, name: str) -> dict[str, Any]:
    """Bind one tokenizer input, including its canonical HF blob identity."""

    if Path(name).name != name:
        raise LaunchError(f"source metadata has an unsafe name: {name}")
    path = snapshot / name
    resolved = path
    blob_id: str | None = None
    if path.is_symlink():
        link = Path(os.readlink(path))
        if (
            link.is_absolute()
            or len(link.parts) != 4
            or link.parts[:3] != ("..", "..", "blobs")
            or HF_BLOB_RE.fullmatch(link.parts[3]) is None
        ):
            raise LaunchError(
                f"source metadata is not a canonical Hugging Face blob: {path}"
            )
        blob_root = (snapshot.parent.parent / "blobs").resolve(strict=True)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LaunchError(f"source metadata has a broken link: {path}") from exc
        expected_blob = blob_root / link.parts[3]
        if (
            resolved != expected_blob
            or expected_blob.is_symlink()
            or not expected_blob.is_file()
        ):
            raise LaunchError(
                f"source metadata escapes its Hugging Face blob store: {path}"
            )
        blob_id = link.parts[3]
    elif not path.is_file():
        raise LaunchError(f"source snapshot is missing metadata file {path}")

    digest = sha256_file(resolved)
    if blob_id is not None and len(blob_id) == 64 and blob_id != digest:
        raise LaunchError(f"source metadata SHA-256 blob has changed: {path}")
    identity = {
        "name": name,
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }
    if blob_id is not None:
        identity["hf_blob_id"] = blob_id
    return identity


def source_variant(config: dict[str, Any]) -> dict[str, Any]:
    """Identify the reviewed full-model GLM source format."""

    quantization = config.get("quantization_config")
    if quantization is None:
        return {
            "release": "glm-5.2",
            "format": "bf16",
            "quantization_config_sha256": None,
        }
    if not isinstance(quantization, dict):
        raise LaunchError("source quantization_config is not an object")
    required = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    if any(quantization.get(key) != value for key, value in required.items()):
        raise LaunchError("source uses an unsupported quantization format")
    return {
        "release": "glm-5.3",
        "format": "fp8-e4m3-block128x128-dynamic",
        "quantization_config_sha256": hashlib.sha256(
            canonical_json(quantization)
        ).hexdigest(),
    }


def snapshot_identity(snapshot: Path) -> dict[str, Any]:
    """Bind an immutable supported GLM-5 source and its tensor inventory."""

    snapshot = snapshot.expanduser().resolve(strict=True)
    if not snapshot.is_dir() or not REVISION_RE.fullmatch(snapshot.name):
        raise LaunchError(
            "--snapshot must be an immutable 40- or 64-hex Hugging Face snapshot"
        )
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config = read_json_object(config_path)
    index = read_json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LaunchError("source model index has no weight map")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(name, str) or Path(name).name != name for name in shards):
        raise LaunchError("source model index has unsafe shard names")
    shard_records = [shard_identity(snapshot, name) for name in shards]
    tokenizer_files = [
        source_metadata_identity(snapshot, name)
        for name in TOKENIZER_SOURCE_FILES
    ]
    geometry = {
        "num_hidden_layers": config.get("num_hidden_layers"),
        "first_k_dense_replace": config.get("first_k_dense_replace"),
        "n_routed_experts": config.get("n_routed_experts"),
        "num_experts_per_tok": config.get("num_experts_per_tok"),
        "hidden_size": config.get("hidden_size"),
        "moe_intermediate_size": config.get("moe_intermediate_size"),
        "num_nextn_predict_layers": config.get("num_nextn_predict_layers"),
        "mtp_layer_index": config.get("num_hidden_layers"),
        "first_target_layer": config.get("first_k_dense_replace"),
        "last_target_layer": (
            config.get("num_hidden_layers") - 1
            if isinstance(config.get("num_hidden_layers"), int)
            else None
        ),
        "activation_rank": 3,
    }
    expected_geometry = {
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "num_nextn_predict_layers": 1,
        "mtp_layer_index": 78,
        "first_target_layer": 3,
        "last_target_layer": 77,
        "activation_rank": 3,
    }
    if (
        geometry != expected_geometry
        or config.get("model_type") != "glm_moe_dsa"
        or config.get("architectures") != ["GlmMoeDsaForCausalLM"]
    ):
        raise LaunchError(
            f"source snapshot is not supported GLM-5 geometry: {geometry}"
        )
    variant = source_variant(config)
    namespace_audit = glm52_namespace_audit(
        config,
        weight_map,
        source_format=variant["format"],
    )
    return {
        "path": os.fspath(snapshot),
        "revision": snapshot.name,
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "shards": shard_records,
        "tokenizer_files": tokenizer_files,
        "total_shard_bytes": sum(record["bytes"] for record in shard_records),
        **variant,
        "geometry": geometry,
        "namespace_audit": namespace_audit,
    }


def storage_contract(source: dict[str, Any], bits: int = 3) -> dict[str, Any]:
    """Return deterministic peak-space targets for a supported production run."""

    geometry = source.get("geometry")
    source_bytes = source.get("total_shard_bytes")
    if (
        not isinstance(geometry, dict)
        or isinstance(source_bytes, bool)
        or not isinstance(source_bytes, int)
        or source_bytes <= 0
    ):
        raise LaunchError("source identity has no storage geometry")
    values = tuple(
        geometry.get(name)
        for name in (
            "first_target_layer",
            "last_target_layer",
            "n_routed_experts",
            "hidden_size",
            "moe_intermediate_size",
        )
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise LaunchError("source identity has unsupported storage geometry")
    first_layer, last_layer, experts, hidden, intermediate = values
    layers = last_layer - first_layer + 1
    if (
        layers != 75
        or experts != 256
        or hidden != 6144
        or intermediate != 2048
    ):
        raise LaunchError("source identity has unsupported storage geometry")
    source_format = source.get("format", "bf16")
    if source_format == "bf16":
        native_projection_bytes = hidden * intermediate * 2
        native_projection_scale_bytes = 0
    elif source_format == "fp8-e4m3-block128x128-dynamic":
        native_projection_bytes = hidden * intermediate
        native_projection_scale_bytes = (
            ((hidden + 127) // 128)
            * ((intermediate + 127) // 128)
            * 4
        )
    else:
        raise LaunchError("source identity has an unsupported native format")
    projection_count = layers * experts * 3
    native_replaced_bytes = projection_count * (
        native_projection_bytes + native_projection_scale_bytes
    )
    if bits not in (3, 4):
        raise LaunchError("storage contract requires EXL3 K3 or K4")
    trellis_bytes = hidden * intermediate * bits // 8
    rotation_bytes = (hidden + intermediate) * 2
    exl3_projection_bytes = trellis_bytes + rotation_bytes + 4
    exl3_payload_bytes = projection_count * exl3_projection_bytes
    artifact_payload_estimate = (
        source_bytes - native_replaced_bytes + exl3_payload_bytes
    )
    if artifact_payload_estimate <= 0:
        raise LaunchError("computed EXL3 artifact payload is invalid")
    contract = {
        "contract": (
            "glmrt-glm52-exl3-storage-v1"
            if "format" not in source and bits == 3
            else STORAGE_CONTRACT
        ),
        "source_shard_bytes": source_bytes,
        "native_replaced_payload_bytes": native_replaced_bytes,
        "exl3_projection_payload_bytes": exl3_payload_bytes,
        "artifact_payload_estimate_bytes": artifact_payload_estimate,
        "artifact_export_overhead_bytes": ARTIFACT_EXPORT_OVERHEAD_BYTES,
        "projection_checkpoint_overhead_bytes": (
            PROJECTION_CHECKPOINT_OVERHEAD_BYTES
        ),
        "run_state_peak_bytes": RUN_STATE_PEAK_BYTES,
        "offload_peak_bytes": OFFLOAD_PEAK_BYTES,
        "filesystem_free_floor_bytes": FILESYSTEM_FREE_FLOOR_BYTES,
        "retention": {
            "layer_boundary": "latest-complete-layer",
            "capture_frontier": "current-layer-only",
            "active_source": "current-layer-only",
            "projection_checkpoints": "all-completed-projections",
            "export": "one-atomic-final-stage",
        },
    }
    if "format" in source or bits != 3:
        contract.update(
            {
                "source_format": source_format,
                "trellis_bits": bits,
                "native_replaced_scale_payload_bytes": (
                    projection_count * native_projection_scale_bytes
                ),
            }
        )
    return contract


def glm52_namespace_audit(
    config: dict[str, Any],
    weight_map: dict[str, Any],
    *,
    source_format: str = "bf16",
) -> dict[str, Any]:
    """Prove the base and checkpoint-only native-MTP tensor inventory."""

    layer_count = int(config["num_hidden_layers"])
    first_sparse = int(config["first_k_dense_replace"])
    mtp_layer = layer_count
    expert_count = int(config["n_routed_experts"])
    names = set(weight_map)
    projections = ("gate_proj", "up_proj", "down_proj")

    expected_base_experts = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in range(first_sparse, layer_count)
        for expert in range(expert_count)
        for projection in projections
    }
    expected_mtp_experts = {
        f"model.layers.{mtp_layer}.mlp.experts.{expert}.{projection}.weight"
        for expert in range(expert_count)
        for projection in projections
    }
    actual_experts = {
        name
        for name in names
        if re.fullmatch(
            r"model\.layers\.\d+\.mlp\.experts\.\d+\."
            r"(?:gate_proj|up_proj|down_proj)\.weight",
            name,
        )
    }
    expected_all_experts = expected_base_experts | expected_mtp_experts
    if actual_experts != expected_all_experts:
        raise LaunchError(
            "source routed-expert inventory differs: "
            f"missing={sorted(expected_all_experts - actual_experts)[:4]} "
            f"unexpected={sorted(actual_experts - expected_all_experts)[:4]}"
        )

    actual_expert_scales = {
        name
        for name in names
        if re.fullmatch(
            r"model\.layers\.\d+\.mlp\.experts\.\d+\."
            r"(?:gate_proj|up_proj|down_proj)\.weight_scale_inv",
            name,
        )
    }
    expected_expert_scales = (
        {f"{name[:-len('.weight')]}.weight_scale_inv" for name in expected_all_experts}
        if source_format == "fp8-e4m3-block128x128-dynamic"
        else set()
    )
    if actual_expert_scales != expected_expert_scales:
        raise LaunchError(
            "source routed-expert scale inventory differs: "
            f"missing={sorted(expected_expert_scales - actual_expert_scales)[:4]} "
            f"unexpected={sorted(actual_expert_scales - expected_expert_scales)[:4]}"
        )

    sparse_layers = range(first_sparse, mtp_layer + 1)
    expected_shared = {
        f"model.layers.{layer}.mlp.shared_experts.{projection}.weight"
        for layer in sparse_layers
        for projection in projections
    }
    if not expected_shared <= names:
        raise LaunchError(
            "source shared-expert inventory is incomplete: "
            f"{sorted(expected_shared - names)[:4]}"
        )

    expected_router = {
        f"model.layers.{layer}.mlp.gate.{suffix}"
        for layer in sparse_layers
        for suffix in ("weight", "e_score_correction_bias")
    }
    if not expected_router <= names:
        raise LaunchError(
            "source learned-router inventory is incomplete: "
            f"{sorted(expected_router - names)[:4]}"
        )

    dense_experts = {
        name
        for name in actual_experts
        if int(name.split(".")[2]) < first_sparse
    }
    if dense_experts:
        raise LaunchError("source dense prefix unexpectedly contains routed experts")

    return {
        "contract": "glmrt.glm5-native-namespace-audit-v2",
        "source_format": source_format,
        "base_layers": layer_count,
        "dense_prefix_layers": first_sparse,
        "base_routed_layers": layer_count - first_sparse,
        "mtp_layers": 1,
        "mtp_layer_index": mtp_layer,
        "routed_experts_per_block": expert_count,
        "base_routed_projection_tensors": len(expected_base_experts),
        "mtp_routed_projection_tensors": len(expected_mtp_experts),
        "base_routed_projection_scale_tensors": len(
            expected_expert_scales & {
                f"{name[:-len('.weight')]}.weight_scale_inv"
                for name in expected_base_experts
            }
        ),
        "mtp_routed_projection_scale_tensors": len(
            expected_expert_scales & {
                f"{name[:-len('.weight')]}.weight_scale_inv"
                for name in expected_mtp_experts
            }
        ),
        "shared_expert_tensors": len(expected_shared),
        "learned_router_layers": list(sparse_layers),
        "quantized_scope": "base-routed-expert-gate-up-down-only",
        "preserved_scope": "model.layers.78-native-mtp-overlay",
    }

def calibration_stream(path: Path) -> tuple[list[str], dict[str, Any]]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise LaunchError("--calibration-jsonl must be one regular file")
    texts: list[str] = []
    identifiers: list[str] = []
    text_field: str | None = None
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise LaunchError(
                        f"calibration JSONL line {line_number} is not an object"
                    )
                present_text_fields = [
                    field for field in ("text", "prompt") if field in value
                ]
                if len(present_text_fields) != 1:
                    raise LaunchError(
                        f"calibration JSONL line {line_number} must contain exactly "
                        "one of `text` or `prompt`"
                    )
                row_text_field = present_text_fields[0]
                if text_field is None:
                    text_field = row_text_field
                elif row_text_field != text_field:
                    raise LaunchError(
                        "calibration JSONL mixes `text` and `prompt` row schemas"
                    )
                text = value[row_text_field]
                identifier = value.get("id", f"line-{line_number:08d}")
                if not isinstance(text, str) or not text.strip():
                    raise LaunchError(
                        f"calibration JSONL line {line_number} has no non-empty "
                        f"`{row_text_field}`"
                    )
                if not isinstance(identifier, str) or not identifier:
                    raise LaunchError(
                        f"calibration JSONL line {line_number} has an invalid id"
                    )
                texts.append(text)
                identifiers.append(identifier)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LaunchError(f"cannot read calibration JSONL {path}: {exc}") from exc
    if not texts or len(set(identifiers)) != len(identifiers):
        raise LaunchError("calibration JSONL is empty or contains duplicate ids")
    normalized = canonical_json(
        [
            {"id": identifier, "text": text}
            for identifier, text in zip(identifiers, texts, strict=True)
        ]
    )
    return texts, {
        "path": os.fspath(path),
        "file_sha256": sha256_file(path),
        "normalized_stream_sha256": hashlib.sha256(normalized).hexdigest(),
        "text_field": text_field,
        "examples": len(texts),
        "utf8_bytes": sum(len(text.encode()) for text in texts),
    }


def calibration_evidence(
    manifest_path: Path,
    route_screen_path: Path | None,
    *,
    corpus: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    """Bind the frozen corpus; natural routing is measured on the full replay."""

    if route_screen_path is not None:
        raise LaunchError(
            "GLM-5 uses inline full-corpus routing evidence, not a screen report"
        )
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    manifest = read_json_object(manifest_path)
    splits = manifest.get("splits")
    calibration = splits.get("calibration") if isinstance(splits, dict) else None
    screening = splits.get("screening") if isinstance(splits, dict) else None
    heldout = splits.get("heldout") if isinstance(splits, dict) else None
    builder = manifest.get("builder")
    training_data = manifest.get("training_data_snapshot")
    if (
        manifest.get("schema") != CALIBRATION_MANIFEST_SCHEMA
        or not isinstance(calibration, dict)
        or not isinstance(screening, dict)
        or not isinstance(heldout, dict)
        or not isinstance(builder, dict)
        or not isinstance(training_data, dict)
        or manifest.get("screening_calibration_identity_subset") is not True
        or manifest.get("source_group_overlap") != []
        or calibration.get("sha256") != corpus["file_sha256"]
        or calibration.get("summary", {}).get("records") != corpus["examples"]
        or not isinstance(calibration.get("records"), list)
        or len(calibration["records"]) != corpus["examples"]
        or screening.get("derived_from") != "calibration"
        or REVISION_RE.fullmatch(str(builder.get("revision", ""))) is None
        or REVISION_RE.fullmatch(str(training_data.get("revision", ""))) is None
        or SHA256_RE.fullmatch(str(builder.get("sha256", ""))) is None
        or SHA256_RE.fullmatch(str(manifest.get("tokenizer_sha256", ""))) is None
    ):
        raise LaunchError("calibration manifest does not bind a reproducible corpus")
    tokenizer_files = source.get("tokenizer_files")
    source_tokenizer_sha256 = next(
        (
            record.get("sha256")
            for record in tokenizer_files
            if isinstance(record, dict) and record.get("name") == "tokenizer.json"
        ),
        None,
    ) if isinstance(tokenizer_files, list) else None
    if manifest.get("tokenizer_sha256") != source_tokenizer_sha256:
        raise LaunchError(
            "calibration tokenizer differs from the immutable model tokenizer"
        )
    manifest_corpus_path = (
        manifest_path.parent / str(calibration.get("file", ""))
    ).resolve()
    if manifest_corpus_path != Path(corpus["path"]):
        raise LaunchError("calibration manifest selects a different JSONL stream")

    calibration_ids = [record.get("id") for record in calibration["records"]]
    prompt_hashes = [
        record.get("prompt_sha256") for record in calibration["records"]
    ]
    token_hashes = [
        record.get("token_ids_sha256") for record in calibration["records"]
    ]
    if (
        len(set(calibration_ids)) != len(calibration_ids)
        or any(not isinstance(value, str) or not value for value in calibration_ids)
        or any(SHA256_RE.fullmatch(str(value)) is None for value in prompt_hashes)
        or any(SHA256_RE.fullmatch(str(value)) is None for value in token_hashes)
    ):
        raise LaunchError("calibration manifest has invalid prompt/token identities")

    return {
        "manifest": {
            "path": os.fspath(manifest_path),
            "sha256": sha256_file(manifest_path),
            "schema": manifest["schema"],
            "builder_revision": builder["revision"],
            "builder_sha256": builder["sha256"],
            "training_data_revision": training_data["revision"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "calibration_prompt_tokens": calibration["summary"].get(
                "prompt_tokens"
            ),
            "calibration_examples": corpus["examples"],
            "calibration_token_identity_sha256": hashlib.sha256(
                canonical_json(token_hashes)
            ).hexdigest(),
        },
        "route_qualification": {
            "mode": ROUTE_QUALIFICATION_INLINE,
            "status": "bound-to-full-corpus-capture",
            "scope": "base-routed-layers-3-through-77",
            "natural_route_contract": ROUTE_EVIDENCE_CONTRACT,
            "recovery_contract": ZERO_ROUTE_RECOVERY_CONTRACT,
            "recovery_trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
            "candidate_ranks": [
                ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
                ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
            ],
            "target_effective_rows": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
            "failure_policy": "fail-only-on-router-or-evidence-invariant",
        },
    }


def quantization_toolchain_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    names = [
        "preflight.py",
        "quantize_glm52_gptqmodel.py",
        "glm52_layer_boundary_store.py",
        "glm52_execution_upgrade.py",
    ]
    return {
        "files": {name: sha256_file(root / name) for name in names}
    }


def projection_seed_family_numerics(
    family_join: dict[str, Any],
    *,
    seeded_family_join: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return projection numerics after explicit orchestration migrations."""

    numerics = {
        key: value
        for key, value in family_join.items()
        if key not in PROJECTION_FAMILY_ORCHESTRATION_FIELDS
    }
    if seeded_family_join is None:
        return numerics
    seeded_lock = seeded_family_join.get("gptqmodel")
    current_lock = family_join.get("gptqmodel")
    if isinstance(seeded_lock, dict) and isinstance(current_lock, dict):
        transition = (
            seeded_lock.get("revision"),
            seeded_lock.get("source_tree_sha256"),
            current_lock.get("revision"),
            current_lock.get("source_tree_sha256"),
        )
        if transition in PROJECTION_SEED_GPTQMODEL_ORCHESTRATION_TRANSITIONS:
            numerics.pop("gptqmodel", None)
    return numerics

def preflight_identity(
    path: Path,
    expected_revision: str,
    *,
    role: str = "coordinator",
    expected_gpu_count: int | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    report = read_json_object(path)
    gptqmodel = report.get("gptqmodel")
    image_digest = report.get("image_digest")
    gpus = report.get("gpus")
    expected_platform_arch = PREFLIGHT_ROLE_CONTRACTS.get(role)
    if expected_gpu_count is None:
        expected_gpu_count = 2 if role == "coordinator" else 1
    if (
        isinstance(expected_gpu_count, bool)
        or not isinstance(expected_gpu_count, int)
        or expected_gpu_count not in ({1, 2} if role == "coordinator" else {1})
    ):
        raise LaunchError(f"{role} preflight GPU count is invalid")
    if (
        expected_platform_arch is None
        or report.get("status") != "qualified"
        or report.get("role") != role
        or (report.get("target_platform"), report.get("cuda_arch"))
        != expected_platform_arch
        or not isinstance(gptqmodel, dict)
        or gptqmodel.get("revision") != expected_revision
        or not isinstance(image_digest, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest) is None
        or report.get("python", {}).get("gil_enabled") is not False
        or not isinstance(gpus, list)
        or len(gpus) != expected_gpu_count
        or any(not isinstance(gpu, dict) for gpu in gpus)
        or [gpu.get("index") for gpu in gpus] != list(range(expected_gpu_count))
        or any(
            not isinstance(gpu.get("uuid"), str) or not gpu["uuid"]
            for gpu in gpus
        )
        or len({gpu["uuid"] for gpu in gpus}) != expected_gpu_count
    ):
        raise LaunchError(f"{role} preflight does not match the production run")
    return {
        "path": os.fspath(path),
        "sha256": report_identity_sha256(report),
        "image_digest": image_digest,
        "gptqmodel": gptqmodel,
        "python": report["python"],
        "torch": report.get("torch"),
        "gpus": gpus,
    }


def build_plan(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    """Create an immutable local dual-RTX GLM-5 production plan."""

    if getattr(args, "remote_worker", None):
        raise LaunchError("GLM-5 quantization currently supports local RTX GPUs only")
    lock_path = args.gptqmodel_lock.expanduser().resolve(strict=True)
    lock = read_json_object(lock_path)
    if (
        lock.get("schema") != 1
        or lock.get("repository") != "https://github.com/tpurtell/GPTQModel.git"
        or REVISION_RE.fullmatch(str(lock.get("revision", ""))) is None
        or SHA256_RE.fullmatch(str(lock.get("source_tree_sha256", ""))) is None
    ):
        raise LaunchError("GPTQModel source lock is invalid")

    source = snapshot_identity(args.snapshot)
    texts, corpus = calibration_stream(args.calibration_jsonl)
    evidence = calibration_evidence(
        args.calibration_manifest,
        getattr(args, "route_screen_report", None),
        corpus=corpus,
        source=source,
    )
    toolchain = quantization_toolchain_identity()
    bits = int(getattr(args, "bits", 3))
    variant = quantization_variant(source, bits)
    recipe = variant["recipe"]
    coordinator_gpu_count = int(getattr(args, "coordinator_gpu_count", 2))
    preflight = preflight_identity(
        args.preflight_report,
        str(lock["revision"]),
        expected_gpu_count=coordinator_gpu_count,
    )
    if (
        preflight["gptqmodel"].get("revision") != lock["revision"]
        or preflight["gptqmodel"].get("source_tree_sha256")
        != lock["source_tree_sha256"]
    ):
        raise LaunchError("preflight GPTQModel identity differs from the source lock")

    output = args.output.expanduser().resolve()
    offload = args.offload_dir.expanduser().resolve()
    raw_run_state = getattr(args, "run_state_dir", None)
    run_state = (
        raw_run_state.expanduser().resolve()
        if raw_run_state is not None
        else output.with_name(f".{output.name}.glmrt-run")
    )
    raw_projection_root = getattr(args, "projection_checkpoint_dir", None)
    projection_root = (
        raw_projection_root.expanduser().resolve()
        if raw_projection_root is not None
        else run_state / PROJECTION_CHECKPOINT_DIRNAME
    )
    raw_projection_seed = getattr(args, "projection_checkpoint_seed_dir", None)
    projection_seed = (
        projection_checkpoint_seed_identity(raw_projection_seed)
        if raw_projection_seed is not None
        else None
    )
    raw_active_source = getattr(args, "active_layer_source_dir", None)
    active_source = (
        raw_active_source.expanduser().resolve()
        if raw_active_source is not None
        else run_state / ACTIVE_LAYER_SOURCE_DIRNAME
    )

    independent_paths = (output, run_state, offload)
    if len(set(independent_paths)) != len(independent_paths) or any(
        left.is_relative_to(right) or right.is_relative_to(left)
        for index, left in enumerate(independent_paths)
        for right in independent_paths[index + 1 :]
    ):
        raise LaunchError(
            "output, run-state, and offload paths must be distinct and non-nested"
        )
    subordinate_paths = (
        (projection_root, run_state / PROJECTION_CHECKPOINT_DIRNAME),
        (active_source, run_state / ACTIVE_LAYER_SOURCE_DIRNAME),
    )
    for path, canonical_child in subordinate_paths:
        if path in independent_paths or any(
            path.is_relative_to(other) or other.is_relative_to(path)
            for other in (output, offload)
        ):
            raise LaunchError("active and checkpoint stores overlap another run path")
        if (path.is_relative_to(run_state) or run_state.is_relative_to(path)) and (
            path != canonical_child
        ):
            raise LaunchError(
                "a run-state child store must use its canonical directory name"
            )
    if (
        projection_root == active_source
        or projection_root.is_relative_to(active_source)
        or active_source.is_relative_to(projection_root)
    ):
        raise LaunchError("projection checkpoints and active source staging overlap")
    if projection_seed is not None:
        seed_root = Path(projection_seed["root"])
        if any(
            seed_root == path
            or seed_root.is_relative_to(path)
            or path.is_relative_to(seed_root)
            for path in (*independent_paths, projection_root, active_source)
        ):
            raise LaunchError(
                "projection-checkpoint seed overlaps an active run path"
            )

    recovery_recipe = {
        "trigger": ZERO_ROUTE_RECOVERY_TRIGGER,
        "sample_source": ZERO_ROUTE_RECOVERY_SAMPLE_SOURCE,
        "capture_method": ZERO_ROUTE_RECOVERY_CAPTURE_METHOD,
        "selection_policy": ZERO_ROUTE_RECOVERY_SELECTION_POLICY,
        "candidate_rank_min": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MIN,
        "candidate_rank_max": ZERO_ROUTE_RECOVERY_CANDIDATE_RANK_MAX,
        "selection_cap": ZERO_ROUTE_RECOVERY_SELECTION_CAP,
        "target_sample_count": ZERO_ROUTE_RECOVERY_TARGET_SAMPLE_COUNT,
        "identity_calibration_policy": ZERO_ROUTE_RECOVERY_IDENTITY_POLICY,
    }
    family_join = {
        "recipe": recipe,
        "source": source,
        "corpus": corpus,
        "calibration_evidence": evidence,
        "quantization_toolchain": toolchain,
        "gptqmodel": lock,
        "preflight_sha256": preflight["sha256"],
        "image_digest": preflight["image_digest"],
        "quantizer_seed": EXL3_SEED,
        "quantizer_numerics": {
            "sigma_reg": EXL3_SIGMA_REG,
            "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
            "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        },
        "bits": bits,
        "codebook": "mcg",
        "module_include": BASE_EXPERT_PATTERN,
        "operator_contract": variant["operator_contract"],
        "route_evidence_contract": ROUTE_EVIDENCE_CONTRACT,
        "zero_route_recovery_contract": ZERO_ROUTE_RECOVERY_CONTRACT,
        "zero_route_recovery_recipe": recovery_recipe,
        "forward_replay": {
            "policy": FORWARD_REPLICA_POLICY,
            "torch_parallel_replicate": False,
        },
    }
    if projection_seed is not None:
        seeded_family_join = projection_seed["family_join"]
        current_numerics = projection_seed_family_numerics(
            family_join,
            seeded_family_join=seeded_family_join,
        )
        seeded_numerics = projection_seed_family_numerics(seeded_family_join)
        if "gptqmodel" not in current_numerics:
            seeded_numerics.pop("gptqmodel", None)
        if current_numerics != seeded_numerics:
            raise LaunchError(
                "projection-checkpoint seed has different quantization numerics"
            )
        # ``family_join`` is the numerical compatibility identity used as part
        # of every projection checkpoint key.  Projection requests predate the
        # layer-boundary v3 orchestration fix, so preserve the seed's exact
        # family for both the seeded layer and newly generated projections.
        # This does not claim that the old GPTQModel revision executed the new
        # work: ``provenance.run.coordinator`` and the enclosing plan bind the
        # actual source tree, image, hardware, and preflight independently.
        family_join = seeded_family_join
    projection_checkpoint = {
        "contract": PROJECTION_CHECKPOINT_CONTRACT,
        "root": os.fspath(projection_root),
    }
    storage = storage_contract(source, bits)
    geometry = source["geometry"]
    layer_boundary = {
        "contract": BOUNDARY_CONTRACT,
        "root": os.fspath(run_state / LAYER_BOUNDARY_DIRNAME),
        "retention": "latest-complete-layer",
        "dtype": "bfloat16",
        "activation_rank": geometry["activation_rank"],
        "first_target_layer": geometry["first_target_layer"],
        "last_target_layer": geometry["last_target_layer"],
    }
    provenance = {
        "family_join": family_join,
        "run": {
            "coordinator": preflight,
            "output": os.fspath(output),
            "run_state": os.fspath(run_state),
            "offload": os.fspath(offload),
            "active_layer_source": os.fspath(active_source),
            "target_batch_size": args.batch_size,
            "projection_checkpoint": projection_checkpoint,
            "projection_checkpoint_seed": projection_seed,
            "layer_boundary": layer_boundary,
            "storage": storage,
        },
    }
    plan = {
        "schema": PLAN_SCHEMA,
        "recipe": recipe,
        "source": source,
        "corpus": corpus,
        "calibration_evidence": evidence,
        "quantization_toolchain": toolchain,
        "preflight": preflight,
        "output": os.fspath(output),
        "run_state_dir": os.fspath(run_state),
        "projection_checkpoint_dir": os.fspath(projection_root),
        "active_layer_source_dir": os.fspath(active_source),
        "offload_dir": os.fspath(offload),
        "target_batch_size": args.batch_size,
        "projection_checkpoint": projection_checkpoint,
        "projection_checkpoint_seed": projection_seed,
        "layer_boundary": layer_boundary,
        "storage": storage,
        "memory_safety": {
            "host_rss_limit_bytes": HOST_RSS_LIMIT_BYTES,
            "cuda_allocation_limit_bytes": CUDA_ALLOCATION_LIMIT_BYTES,
            "telemetry_interval_batches": MEMORY_TELEMETRY_INTERVAL_BATCHES,
            "spill_policy": "fail-closed-no-cpu-or-managed-memory",
        },
        "forward_replay": {
            "policy": FORWARD_REPLICA_POLICY,
            "torch_parallel_replicate": False,
        },
        "remote_workers": None,
        "exl3": {
            "bits": bits,
            "codebook": "mcg",
            "seed": EXL3_SEED,
            "module_include": [BASE_EXPERT_PATTERN],
            "fallback": None,
            "out_scales": "auto",
            "sigma_reg": EXL3_SIGMA_REG,
            "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
            "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
            "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
            "zero_route_recovery": {
                "contract": ZERO_ROUTE_RECOVERY_CONTRACT,
                **recovery_recipe,
                "scope": "learned-top8-base-routers-layers-3-through-77",
            },
        },
        "ledger_provenance": provenance,
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json(plan)).hexdigest()
    return plan, texts

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _bound_record(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    if digest_field in value:
        raise LaunchError(f"record already contains reserved field {digest_field}")
    clean = dict(value)
    clean[digest_field] = hashlib.sha256(canonical_json(value)).hexdigest()
    return clean


def _validate_bound_record(
    value: dict[str, Any],
    *,
    digest_field: str,
    label: str,
) -> None:
    digest = value.get(digest_field)
    body = {key: item for key, item in value.items() if key != digest_field}
    if (
        not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise LaunchError(f"{label} digest is invalid")


def _read_execution_upgrade(
    root: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    try:
        return read_execution_upgrade(
            root,
            parent_plan_sha256=str(plan.get("plan_sha256", "")),
        )
    except ExecutionUpgradeError as error:
        raise LaunchError(str(error)) from error


def _repair_execution_upgrade_archive_staging(
    root: Path,
    plan: dict[str, Any],
) -> None:
    """Remove only the harmless duplicate left by an interrupted chain swap.

    Chaining deliberately archives the current active record before atomically
    replacing it.  If the process stops between those two operations, the
    archive contains an exact copy of the still-active record and therefore is
    not ancestry yet.  That is the sole state repaired here; every differing,
    malformed, or otherwise unlinked history record continues to fail closed.
    """

    active_path = root / EXECUTION_UPGRADE_FILENAME
    history = root / EXECUTION_UPGRADE_HISTORY_DIRNAME
    if not active_path.exists() or not history.exists():
        return
    if (
        not active_path.is_file()
        or active_path.is_symlink()
        or not history.is_dir()
        or history.is_symlink()
    ):
        raise LaunchError("execution-upgrade staging paths are unsafe")
    active = read_json_object(active_path)
    if (
        active.get("schema") != EXECUTION_UPGRADE_SCHEMA
        or active.get("parent_plan_sha256") != plan.get("plan_sha256")
    ):
        raise LaunchError("active execution-upgrade staging record is invalid")
    _validate_bound_record(
        active,
        digest_field="upgrade_sha256",
        label="active execution upgrade",
    )
    duplicate = history / f"{active['upgrade_sha256']}.json"
    if not duplicate.exists():
        return
    if (
        not duplicate.is_file()
        or duplicate.is_symlink()
        or read_json_object(duplicate) != active
    ):
        raise LaunchError("execution-upgrade staging archive differs from active")
    duplicate.unlink()
    _fsync_directory(history)


def _journal_resume_identity(path: Path) -> dict[str, Any]:
    """Bind the exact error-ledger frontier present at a code transition."""

    if not path.is_file() or path.is_symlink():
        raise LaunchError("execution upgrade requires a regular error journal")
    digest = hashlib.sha256()
    records = 0
    total_bytes = 0
    last_byte = b""
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
            records += block.count(b"\n")
            total_bytes += len(block)
            last_byte = block[-1:]
    if total_bytes and last_byte != b"\n":
        raise LaunchError("execution-upgrade journal ends with a partial record")
    return {
        "bytes": total_bytes,
        "records": records,
        "sha256": digest.hexdigest(),
    }


def _latest_boundary_resume_identity(
    run_state: Path,
    plan: dict[str, Any],
) -> dict[str, Any] | None:
    """Authenticate the one rolling GLM boundary retained across new code."""

    root = run_state / LAYER_BOUNDARY_DIRNAME
    if not root.exists():
        return None
    _regular_directory(root, "layer-boundary directory")
    entries = list(root.iterdir())
    if not entries:
        return None
    if len(entries) != 1:
        raise LaunchError("execution upgrade requires one retained layer boundary")
    directory = entries[0]
    match = BOUNDARY_DIRECTORY_RE.fullmatch(directory.name)
    if match is None or not directory.is_dir() or directory.is_symlink():
        raise LaunchError("execution upgrade found an unsafe layer boundary")
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise LaunchError("execution upgrade boundary manifest is unavailable")
    manifest = read_json_object(manifest_path)
    digest = manifest.get("manifest_sha256")
    body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("schema") != BOUNDARY_SCHEMA
        or manifest.get("schema_version") != BOUNDARY_SCHEMA_VERSION
        or manifest.get("payload_hash_algorithm")
        != BOUNDARY_PAYLOAD_HASH_ALGORITHM
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
        or match.group("digest") != digest[:16]
        or int(match.group("layer")) != manifest.get("layer_index")
    ):
        raise LaunchError("execution-upgrade layer boundary failed validation")
    return {
        "directory": directory.name,
        "layer_index": manifest["layer_index"],
        "layer_name": manifest.get("layer_name"),
        "manifest_sha256": digest,
        "activation_batches": manifest.get("activation_batches"),
        "activation_bytes": manifest.get("activation_bytes"),
        "replay_state_bytes": manifest.get("replay_state_bytes"),
        "completed_projection_entries": len(
            manifest.get("completed_projection_entries", ())
        ),
    }


def _valid_projection_checkpoint_seed(value: Any) -> bool:
    if value is None:
        return True
    if (
        not isinstance(value, dict)
        or value.get("contract") != PROJECTION_CHECKPOINT_SEED_CONTRACT
        or not isinstance(value.get("root"), str)
        or not Path(value["root"]).is_absolute()
        or not isinstance(value.get("files"), list)
        or not value["files"]
        or not isinstance(value.get("family_join"), dict)
        or not value["family_join"]
        or isinstance(value.get("checkpoint_count"), bool)
        or not isinstance(value.get("checkpoint_count"), int)
        or value["checkpoint_count"] <= 0
        or isinstance(value.get("total_bytes"), bool)
        or not isinstance(value.get("total_bytes"), int)
        or value["total_bytes"] <= 0
        or SHA256_RE.fullmatch(str(value.get("inventory_sha256", ""))) is None
    ):
        return False
    paths: list[str] = []
    pairs: dict[str, set[str]] = {}
    total_bytes = 0
    for record in value["files"]:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "bytes",
            "sha256",
        }:
            return False
        relative = record["path"]
        size = record["bytes"]
        digest = record["sha256"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            return False
        parts = Path(relative).parts
        path = Path(relative)
        stem = path.stem
        if (
            len(parts) != 3
            or re.fullmatch(r"[0-9a-f]{2}", parts[0]) is None
            or re.fullmatch(r"[0-9a-f]{2}", parts[1]) is None
            or SHA256_RE.fullmatch(stem) is None
            or not stem.startswith(parts[0] + parts[1])
            or path.suffix not in {".json", ".safetensors"}
        ):
            return False
        paths.append(relative)
        total_bytes += size
        pairs.setdefault(stem, set()).add(path.suffix)
    return (
        paths == sorted(set(paths))
        and total_bytes == value["total_bytes"]
        and all(suffixes == {".json", ".safetensors"} for suffixes in pairs.values())
        and len(pairs) == value["checkpoint_count"]
        and hashlib.sha256(canonical_json(value["files"])).hexdigest()
        == value["inventory_sha256"]
    )


def _validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") not in SUPPORTED_PLAN_SCHEMAS:
        raise LaunchError("plan does not use a supported GLM-5 GPTQModel schema")
    digest = plan.get("plan_sha256")
    body = {key: value for key, value in plan.items() if key != "plan_sha256"}
    exl3 = plan.get("exl3")
    checkpoint = plan.get("projection_checkpoint")
    checkpoint_seed = plan.get("projection_checkpoint_seed")
    boundary = plan.get("layer_boundary")
    source = plan.get("source")
    storage = plan.get("storage")
    tokenizer_files = (
        source.get("tokenizer_files") if isinstance(source, dict) else None
    )
    bits = exl3.get("bits") if isinstance(exl3, dict) else None
    if plan.get("schema") == PLAN_SCHEMA and isinstance(source, dict):
        variant = quantization_variant(source, bits)
        variant_valid = plan.get("recipe") == variant["recipe"]
    else:
        variant_valid = (
            bits == 3 and plan.get("recipe") == GLM52_NATURAL_ROUTE_RECIPE
        )
    schema_contract_valid = (
        plan.get("schema") == LEGACY_PLAN_SCHEMA
        and storage is None
        and tokenizer_files is None
    ) or (
        plan.get("schema") == GLM52_PLAN_SCHEMA
        and isinstance(source, dict)
        and isinstance(tokenizer_files, list)
        and all(
            isinstance(record, dict)
            and isinstance(record.get("bytes"), int)
            and record["bytes"] > 0
            and SHA256_RE.fullmatch(str(record.get("sha256", ""))) is not None
            for record in tokenizer_files
        )
        and [record["name"] for record in tokenizer_files]
        == list(TOKENIZER_SOURCE_FILES)
        and storage == storage_contract(source)
    ) or (
        plan.get("schema") == PLAN_SCHEMA
        and isinstance(source, dict)
        and source.get("release") in {"glm-5.2", "glm-5.3"}
        and isinstance(tokenizer_files, list)
        and all(
            isinstance(record, dict)
            and isinstance(record.get("bytes"), int)
            and record["bytes"] > 0
            and SHA256_RE.fullmatch(str(record.get("sha256", ""))) is not None
            for record in tokenizer_files
        )
        and [record["name"] for record in tokenizer_files]
        == list(TOKENIZER_SOURCE_FILES)
        and storage == storage_contract(source, bits)
    )
    path_fields = (
        "output",
        "run_state_dir",
        "projection_checkpoint_dir",
        "active_layer_source_dir",
        "offload_dir",
    )
    if (
        not isinstance(digest, str)
        or SHA256_RE.fullmatch(digest) is None
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
        or not schema_contract_valid
        or not variant_valid
        or not isinstance(exl3, dict)
        or bits not in (3, 4)
        or exl3.get("codebook") != "mcg"
        or exl3.get("module_include") != [BASE_EXPERT_PATTERN]
        or plan.get("remote_workers") is not None
        or any(
            not isinstance(plan.get(field), str)
            or not plan[field]
            or not Path(plan[field]).is_absolute()
            for field in path_fields
        )
        or not isinstance(checkpoint, dict)
        or checkpoint.get("contract") != PROJECTION_CHECKPOINT_CONTRACT
        or checkpoint.get("root") != plan["projection_checkpoint_dir"]
        or not _valid_projection_checkpoint_seed(checkpoint_seed)
        or not isinstance(boundary, dict)
        or boundary.get("contract") != BOUNDARY_CONTRACT
        or boundary.get("root")
        != os.fspath(Path(plan["run_state_dir"]) / LAYER_BOUNDARY_DIRNAME)
        or boundary.get("retention") != "latest-complete-layer"
        or boundary.get("dtype") != "bfloat16"
        or boundary.get("activation_rank") != 3
        or boundary.get("first_target_layer") != 3
        or boundary.get("last_target_layer") != 77
        or plan.get("memory_safety")
        != {
            "host_rss_limit_bytes": HOST_RSS_LIMIT_BYTES,
            "cuda_allocation_limit_bytes": CUDA_ALLOCATION_LIMIT_BYTES,
            "telemetry_interval_batches": MEMORY_TELEMETRY_INTERVAL_BATCHES,
            "spill_policy": "fail-closed-no-cpu-or-managed-memory",
        }
        or plan.get("forward_replay")
        != {
            "policy": FORWARD_REPLICA_POLICY,
            "torch_parallel_replicate": False,
        }
        or plan.get("ledger_provenance", {})
        .get("family_join", {})
        .get("forward_replay")
        != plan.get("forward_replay")
        or plan.get("ledger_provenance", {})
        .get("run", {})
        .get("projection_checkpoint_seed")
        != checkpoint_seed
        or plan.get("ledger_provenance", {})
        .get("run", {})
        .get("coordinator")
        != plan.get("preflight")
        or plan.get("ledger_provenance", {}).get("run", {}).get("storage")
        != storage
    ):
        raise LaunchError("plan contract is invalid or inconsistent")


def _regular_directory(path: Path, label: str) -> None:
    if not path.is_dir() or path.is_symlink():
        raise LaunchError(f"{label} is not a regular directory: {path}")


def _empty_or_missing_directory(path: Path, label: str) -> None:
    if path.is_symlink():
        raise LaunchError(f"{label} is a symbolic link: {path}")
    if not path.exists():
        return
    _regular_directory(path, label)
    if any(path.iterdir()):
        raise LaunchError(f"{label} is not empty: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _filesystem_anchor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise LaunchError(f"no existing filesystem anchor for {path}")
        candidate = parent
    if candidate.is_symlink():
        raise LaunchError(f"filesystem anchor is a symbolic link: {candidate}")
    return candidate.resolve(strict=True)


def _tree_file_bytes(root: Path, *, exclude: tuple[Path, ...] = ()) -> int:
    if not root.exists():
        return 0
    if root.is_symlink() or not root.is_dir():
        raise LaunchError(f"storage root is not a regular directory: {root}")
    excluded = tuple(path.resolve() for path in exclude if path.exists())
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LaunchError(f"storage root contains a symbolic link: {path}")
        resolved = path.resolve()
        if any(
            resolved == blocked or resolved.is_relative_to(blocked)
            for blocked in excluded
        ):
            continue
        if path.is_file():
            total += path.stat().st_size
        elif not path.is_dir():
            raise LaunchError(f"storage root contains an unsupported entry: {path}")
    return total


def validate_storage_capacity(plan: dict[str, Any]) -> dict[str, Any]:
    """Fail before model loading if bounded durable stores cannot finish."""

    _validate_plan(plan)
    contract = plan.get("storage") or storage_contract(
        plan["source"], plan["exl3"]["bits"]
    )
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    projection = Path(plan["projection_checkpoint_dir"])
    offload = Path(plan["offload_dir"])
    output_anchor = _filesystem_anchor(output.parent)
    run_anchor = _filesystem_anchor(run_state)
    if output_anchor.stat().st_dev != run_anchor.stat().st_dev:
        raise LaunchError(
            "atomic artifact publication requires output and run-state on one filesystem"
        )

    projection_excluded = (
        (projection,)
        if projection.exists() and projection.is_relative_to(run_state)
        else ()
    )
    targets = (
        (
            "atomic-artifact-export",
            output.parent,
            contract["artifact_payload_estimate_bytes"]
            + contract["artifact_export_overhead_bytes"],
            0,
        ),
        (
            "bounded-run-state",
            run_state,
            contract["run_state_peak_bytes"],
            _tree_file_bytes(run_state, exclude=projection_excluded),
        ),
        (
            "projection-checkpoints",
            projection,
            contract["exl3_projection_payload_bytes"]
            + contract["projection_checkpoint_overhead_bytes"],
            _tree_file_bytes(projection),
        ),
        (
            "bounded-offload",
            offload,
            contract["offload_peak_bytes"],
            _tree_file_bytes(offload),
        ),
    )
    filesystems: dict[int, dict[str, Any]] = {}
    for role, path, target_bytes, existing_bytes in targets:
        anchor = _filesystem_anchor(path)
        device = anchor.stat().st_dev
        record = filesystems.setdefault(
            device,
            {
                "device": device,
                "anchor": os.fspath(anchor),
                "roles": [],
                "required_future_bytes": contract["filesystem_free_floor_bytes"],
            },
        )
        required = max(0, target_bytes - existing_bytes)
        record["roles"].append(
            {
                "role": role,
                "path": os.fspath(path),
                "target_bytes": target_bytes,
                "existing_bytes": existing_bytes,
                "required_future_bytes": required,
            }
        )
        record["required_future_bytes"] += required

    for record in filesystems.values():
        stats = os.statvfs(record["anchor"])
        available = stats.f_bavail * stats.f_frsize
        record["available_bytes"] = available
        record["projected_free_bytes"] = available - record["required_future_bytes"]
        if available < record["required_future_bytes"]:
            raise LaunchError(
                "insufficient storage for GLM-5 EXL3 completion on "
                f"{record['anchor']}: available={available} "
                f"required={record['required_future_bytes']}"
            )
    ordered = sorted(filesystems.values(), key=lambda record: record["device"])
    return {
        "schema": "glmrt-glm5-exl3-storage-preflight-v2",
        "status": "accepted",
        "plan_sha256": plan["plan_sha256"],
        "contract": contract,
        "filesystems": ordered,
    }


def _initialize_empty_error_journal(run_state: Path) -> None:
    path = run_state / ERROR_JOURNAL_FILENAME
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
        )
    except FileExistsError as error:
        raise LaunchError(f"run-state error journal already exists: {path}") from error
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(run_state)


def _validate_run_state_entries(run_state: Path) -> None:
    allowed = {
        PLAN_FILENAME,
        EXECUTION_UPGRADE_FILENAME,
        ERROR_JOURNAL_FILENAME,
        DIRECT_STATE_PREFLIGHT_FILENAME,
        STORAGE_PREFLIGHT_FILENAME,
        PROJECTION_CHECKPOINT_DIRNAME,
        ACTIVE_LAYER_SOURCE_DIRNAME,
        LAYER_BOUNDARY_DIRNAME,
        CAPTURE_FRONTIER_DIRNAME,
        CAPTURE_BATCH_SPOOL_DIRNAME,
        POST_QUANT_REPLAY_DIRNAME,
        JIT_CACHE_DIRNAME,
        EXECUTION_UPGRADE_HISTORY_DIRNAME,
        EXPORT_STAGE_DIRNAME,
    }
    unexpected = sorted(
        path.name for path in run_state.iterdir() if path.name not in allowed
    )
    if unexpected:
        raise LaunchError(
            "run-state directory contains unexpected entries: "
            + ", ".join(unexpected)
        )
    for name in (
        PLAN_FILENAME,
        EXECUTION_UPGRADE_FILENAME,
        ERROR_JOURNAL_FILENAME,
        DIRECT_STATE_PREFLIGHT_FILENAME,
        STORAGE_PREFLIGHT_FILENAME,
    ):
        path = run_state / name
        if path.exists() and (not path.is_file() or path.is_symlink()):
            raise LaunchError(f"run-state entry is not a regular file: {path}")
    for name in allowed - {
        PLAN_FILENAME,
        EXECUTION_UPGRADE_FILENAME,
        ERROR_JOURNAL_FILENAME,
        DIRECT_STATE_PREFLIGHT_FILENAME,
        STORAGE_PREFLIGHT_FILENAME,
    }:
        path = run_state / name
        if path.exists():
            _regular_directory(path, "run-state entry")


def _artifact_file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise LaunchError(f"artifact entry is not a regular file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise LaunchError(f"artifact file changed while hashing: {path}")
    return {"bytes": after.st_size, "sha256": digest.hexdigest()}


def projection_checkpoint_seed_identity(root: Path) -> dict[str, Any]:
    """Bind an intentionally reused projection store to exact file content."""

    resolved = root.expanduser().resolve(strict=True)
    _regular_directory(resolved, "projection-checkpoint seed directory")
    records: list[dict[str, Any]] = []
    family_join: dict[str, Any] | None = None
    checkpoint_count = 0
    for path in sorted(resolved.rglob("*")):
        if path.is_symlink():
            raise LaunchError(
                f"projection-checkpoint seed contains a symbolic link: {path}"
            )
        if path.is_dir():
            continue
        relative = path.relative_to(resolved).as_posix()
        parts = Path(relative).parts
        stem = path.stem
        if (
            len(parts) != 3
            or re.fullmatch(r"[0-9a-f]{2}", parts[0]) is None
            or re.fullmatch(r"[0-9a-f]{2}", parts[1]) is None
            or SHA256_RE.fullmatch(stem) is None
            or not stem.startswith(parts[0] + parts[1])
            or path.suffix not in {".json", ".safetensors"}
        ):
            raise LaunchError(
                f"projection-checkpoint seed contains an invalid entry: {relative}"
            )
        if path.suffix == ".json":
            manifest = read_json_object(path)
            request = manifest.get("request")
            result = manifest.get("result")
            ledger_record = (
                result.get("ledger_record") if isinstance(result, dict) else None
            )
            provenance = (
                ledger_record.get("provenance")
                if isinstance(ledger_record, dict)
                else None
            )
            request_family = (
                request.get("family_join") if isinstance(request, dict) else None
            )
            ledger_family = (
                provenance.get("family_join")
                if isinstance(provenance, dict)
                else None
            )
            request_body = (
                {
                    key: value
                    for key, value in request.items()
                    if key != "request_sha256"
                }
                if isinstance(request, dict)
                else None
            )
            manifest_body = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            if (
                not isinstance(request_family, dict)
                or not request_family
                or ledger_family != request_family
                or request.get("request_sha256") != stem
                or manifest.get("request_sha256") != stem
                or not isinstance(request_body, dict)
                or hashlib.sha256(canonical_json(request_body)).hexdigest() != stem
                or manifest.get("tensor_file") != f"{stem}.safetensors"
                or manifest.get("manifest_sha256")
                != hashlib.sha256(canonical_json(manifest_body)).hexdigest()
            ):
                raise LaunchError(
                    f"projection-checkpoint seed manifest is inconsistent: {relative}"
                )
            if family_join is None:
                family_join = request_family
            elif request_family != family_join:
                raise LaunchError(
                    "projection-checkpoint seed mixes projection families"
                )
            checkpoint_count += 1
        records.append({"path": relative, **_artifact_file_identity(path)})
    if not records:
        raise LaunchError("projection-checkpoint seed directory is empty")
    inventory = {
        "contract": PROJECTION_CHECKPOINT_SEED_CONTRACT,
        "root": os.fspath(resolved),
        "files": records,
        "family_join": family_join,
        "checkpoint_count": checkpoint_count,
        "total_bytes": sum(record["bytes"] for record in records),
    }
    inventory["inventory_sha256"] = hashlib.sha256(
        canonical_json(records)
    ).hexdigest()
    if not _valid_projection_checkpoint_seed(inventory):
        raise LaunchError(
            "projection-checkpoint seed does not contain complete checkpoint pairs"
        )
    return inventory


def _seed_projection_checkpoints(
    seed: dict[str, Any], projection_root: Path, error_journal_path: Path
) -> None:
    """Atomically clone a verified immutable seed into a fresh run store."""

    current = projection_checkpoint_seed_identity(Path(seed["root"]))
    if current != seed:
        raise LaunchError("projection-checkpoint seed content has changed")
    stage = projection_root.with_name(
        f".{projection_root.name}.seed-{os.getpid()}"
    )
    _empty_or_missing_directory(stage, "projection-checkpoint seed stage")
    if stage.exists():
        stage.rmdir()
    stage.mkdir(parents=True)
    try:
        for record in seed["files"]:
            source = Path(seed["root"]) / record["path"]
            target = stage / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target, follow_symlinks=False)
            except OSError as error:
                if error.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, target, follow_symlinks=False)
        staged = projection_checkpoint_seed_identity(stage)
        staged["root"] = seed["root"]
        if staged != seed:
            raise LaunchError("seeded projection checkpoints failed verification")
        os.replace(stage, projection_root)
        _fsync_directory(projection_root.parent)
        _seed_projection_error_journal(seed, error_journal_path)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _seed_projection_error_journal(
    seed: dict[str, Any], error_journal_path: Path
) -> None:
    """Recreate journal membership for verified, intentionally reused records."""

    if (
        not error_journal_path.is_file()
        or error_journal_path.is_symlink()
        or error_journal_path.stat().st_size != 0
    ):
        raise LaunchError(
            "fresh projection-seed error journal is unavailable or nonempty"
        )
    records: dict[str, dict[str, Any]] = {}
    for file_record in seed["files"]:
        if not file_record["path"].endswith(".json"):
            continue
        manifest = read_json_object(Path(seed["root"]) / file_record["path"])
        result = manifest.get("result")
        ledger_record = (
            result.get("ledger_record") if isinstance(result, dict) else None
        )
        provenance = (
            ledger_record.get("provenance")
            if isinstance(ledger_record, dict)
            else None
        )
        if (
            not isinstance(ledger_record, dict)
            or ledger_record.get("record_kind") != "projection"
            or not isinstance(provenance, dict)
            or provenance.get("family_join") != seed["family_join"]
        ):
            raise LaunchError("projection seed contains invalid ledger evidence")
        bound = _bound_record(ledger_record, "record_sha256")
        digest = bound["record_sha256"]
        if digest in records:
            raise LaunchError("projection seed contains duplicate ledger evidence")
        records[digest] = bound
    if len(records) != seed["checkpoint_count"]:
        raise LaunchError("projection seed ledger count is incomplete")
    payload = b"".join(
        canonical_json(record) + b"\n"
        for record in sorted(
            records.values(),
            key=lambda record: (
                record.get("logical_layer", -1),
                record.get("expert", -1),
                record.get("projection", ""),
                record.get("module", ""),
            ),
        )
    )
    temporary = error_journal_path.with_name(
        f".{error_journal_path.name}.seed-{os.getpid()}"
    )
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o644,
        )
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, error_journal_path)
        _fsync_directory(error_journal_path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _artifact_paths(root: Path) -> list[Path]:
    _regular_directory(root, "artifact root")
    paths: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise LaunchError(f"artifact contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise LaunchError(f"artifact contains an unsupported entry: {path}")
        paths.append(path)
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def write_artifact_manifest(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    excluded = {ARTIFACT_MANIFEST_FILENAME, RUN_FILENAME}
    records = {}
    for path in _artifact_paths(root):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records[relative] = _artifact_file_identity(path)
    if PLAN_FILENAME not in records or not any(
        name.endswith(".safetensors") for name in records
    ):
        raise LaunchError("export does not contain its plan and safetensor payload")
    _, artifact_schema = artifact_contract_schemas(plan)
    body = {
        "schema": artifact_schema,
        "plan_sha256": plan["plan_sha256"],
        "files": records,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records.values()),
    }
    manifest = _bound_record(body, "manifest_sha256")
    atomic_json(root / ARTIFACT_MANIFEST_FILENAME, manifest)
    return manifest


def validate_published_artifact(
    root: Path,
    plan: dict[str, Any],
    *,
    verify_file_hashes: bool,
) -> dict[str, Any]:
    _validate_plan(plan)
    _regular_directory(root, "published artifact")
    if read_json_object(root / PLAN_FILENAME) != plan:
        raise LaunchError("published artifact plan differs from the requested run")
    manifest = read_json_object(root / ARTIFACT_MANIFEST_FILENAME)
    run = read_json_object(root / RUN_FILENAME)
    artifact_upgrade = (
        _read_execution_upgrade(root, plan)
        if (root / EXECUTION_UPGRADE_FILENAME).exists()
        else None
    )
    execution_upgrade_sha256 = (
        artifact_upgrade["upgrade_sha256"]
        if artifact_upgrade is not None
        else None
    )
    _validate_bound_record(
        manifest,
        digest_field="manifest_sha256",
        label="artifact manifest",
    )
    _validate_bound_record(run, digest_field="run_sha256", label="run manifest")
    records = manifest.get("files")
    expected_run_schema, expected_artifact_schema = artifact_contract_schemas(plan)
    geometry = plan["source"]["geometry"]
    expected_layers = list(
        range(geometry["first_target_layer"], geometry["last_target_layer"] + 1)
    )
    if (
        manifest.get("schema") != expected_artifact_schema
        or manifest.get("plan_sha256") != plan["plan_sha256"]
        or not isinstance(records, dict)
        or manifest.get("file_count") != len(records)
        or manifest.get("total_bytes")
        != sum(
            record.get("bytes", -1)
            for record in records.values()
            if isinstance(record, dict)
        )
        or run.get("schema") != expected_run_schema
        or run.get("status") != "complete"
        or run.get("plan_sha256") != plan["plan_sha256"]
        or run.get("artifact_manifest_sha256") != manifest.get("manifest_sha256")
        or run.get("execution_upgrade_sha256") != execution_upgrade_sha256
        or run.get("quantized_base_layers") != expected_layers
        or run.get("preserved_mtp_layer") != geometry["mtp_layer_index"]
    ):
        raise LaunchError("published artifact manifests are inconsistent")
    expected_paths = set(records) | {ARTIFACT_MANIFEST_FILENAME, RUN_FILENAME}
    actual_paths = {
        path.relative_to(root).as_posix() for path in _artifact_paths(root)
    }
    if actual_paths != expected_paths:
        raise LaunchError("published artifact file set differs from its manifest")
    for relative, record in records.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(record, dict)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise LaunchError("published artifact contains an invalid file record")
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_size != record["bytes"]
        ):
            raise LaunchError(f"published artifact file differs in size: {relative}")
        if verify_file_hashes and _artifact_file_identity(path) != record:
            raise LaunchError(f"published artifact file failed hashing: {relative}")
    return run


def _fsync_artifact_tree(root: Path) -> None:
    """Make an already-validated export durable before its atomic rename."""

    paths = _artifact_paths(root)
    directories = [root]
    directories.extend(path for path in root.rglob("*") if path.is_dir())
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(
        directories,
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    ):
        _regular_directory(directory, "artifact directory")
        _fsync_directory(directory)


def _publish_validated_export_stage(export_stage: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise LaunchError(f"output appeared before artifact publication: {output}")
    _fsync_artifact_tree(export_stage)
    os.replace(export_stage, output)
    _fsync_directory(output.parent)


def _recover_export_stage(
    export_stage: Path,
    output: Path,
    plan: dict[str, Any],
) -> bool:
    """Recover one owned final-export stage after an interrupted process."""

    if not export_stage.exists() and not export_stage.is_symlink():
        return False
    _regular_directory(export_stage, "export stage")
    _artifact_paths(export_stage)
    run_marker = export_stage / RUN_FILENAME
    if run_marker.exists() or run_marker.is_symlink():
        for name in (PLAN_FILENAME, ARTIFACT_MANIFEST_FILENAME, RUN_FILENAME):
            marker = export_stage / name
            if not marker.is_file() or marker.is_symlink():
                raise LaunchError(
                    "committed export stage is missing a regular publication "
                    f"marker: {marker}"
                )
        validate_published_artifact(
            export_stage,
            plan,
            verify_file_hashes=True,
        )
        _publish_validated_export_stage(export_stage, output)
        return True

    # RUN_FILENAME is the final, atomically written commit marker. Without it,
    # this is unpublished scratch. Durable projection checkpoints and rolling
    # state are outside this directory, so rebuilding it repeats no trellis
    # search.
    shutil.rmtree(export_stage)
    _fsync_directory(export_stage.parent)
    return False


def prepare_run(plan: dict[str, Any], *, resume: bool) -> bool:
    """Prepare exact unpublished state; return True for a complete artifact."""

    _validate_plan(plan)
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    projection_root = Path(plan["projection_checkpoint_dir"])
    active_source = Path(plan["active_layer_source_dir"])
    offload = Path(plan["offload_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() or output.is_symlink():
        if not resume:
            raise LaunchError(f"output already exists: {output}")
        validate_published_artifact(output, plan, verify_file_hashes=True)
        return True

    if resume:
        _regular_directory(run_state, "run-state directory")
        _validate_run_state_entries(run_state)
        if read_json_object(run_state / PLAN_FILENAME) != plan:
            raise LaunchError("resume plan differs from the immutable saved plan")
        if (run_state / EXECUTION_UPGRADE_FILENAME).exists():
            _read_execution_upgrade(run_state, plan)
        journal = run_state / ERROR_JOURNAL_FILENAME
        if not journal.is_file() or journal.is_symlink():
            raise LaunchError("resume error journal is unavailable")
        for path, label in (
            (projection_root, "projection-checkpoint directory"),
            (active_source, "active-source directory"),
            (offload, "offload directory"),
        ):
            _regular_directory(path, label)
        return _recover_export_stage(
            run_state / EXPORT_STAGE_DIRNAME,
            output,
            plan,
        )

    for path, label in (
        (run_state, "run-state directory"),
        (projection_root, "projection-checkpoint directory"),
        (active_source, "active-source directory"),
        (offload, "offload directory"),
    ):
        _empty_or_missing_directory(path, label)
    run_state.mkdir(parents=True, exist_ok=True)
    active_source.mkdir(parents=True, exist_ok=True)
    offload.mkdir(parents=True, exist_ok=True)
    atomic_json(run_state / PLAN_FILENAME, plan)
    _initialize_empty_error_journal(run_state)
    seed = plan.get("projection_checkpoint_seed")
    if seed is None:
        projection_root.mkdir(parents=True, exist_ok=True)
    else:
        projection_root.parent.mkdir(parents=True, exist_ok=True)
        _seed_projection_checkpoints(
            seed, projection_root, run_state / ERROR_JOURNAL_FILENAME
        )
    return False


def _stable_execution_identity(
    preflight: dict[str, Any],
    toolchain: dict[str, Any],
) -> dict[str, Any]:
    return {
        "image_digest": preflight.get("image_digest"),
        "preflight_sha256": preflight.get("sha256"),
        "gptqmodel": preflight.get("gptqmodel"),
        "python": preflight.get("python"),
        "torch": preflight.get("torch"),
        "gpus": preflight.get("gpus"),
        "quantization_toolchain": toolchain,
    }


def execution_upgrade_source_matches_parent(
    plan: dict[str, Any],
    current_source: dict[str, Any],
) -> bool:
    """Allow only newly bound metadata fields on a legacy parent source."""

    parent_source = plan.get("source")
    if not isinstance(parent_source, dict) or not isinstance(current_source, dict):
        return False
    if plan.get("schema") == PLAN_SCHEMA:
        return current_source == parent_source
    if plan.get("schema") == GLM52_PLAN_SCHEMA:
        allowed_additions = {
            "release",
            "format",
            "quantization_config_sha256",
        }
        return all(
            current_source.get(key) == value
            for key, value in parent_source.items()
        ) and set(current_source).issubset(set(parent_source) | allowed_additions)
    return all(
        current_source.get(key) == value for key, value in parent_source.items()
    )


def build_execution_upgrade(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Bind corrected execution code while preserving the parent GLM plan."""

    if getattr(args, "remote_worker", None):
        raise LaunchError("GLM-5 execution upgrades cannot add remote workers")
    output = args.output.expanduser().resolve()
    raw_run_state = getattr(args, "run_state_dir", None)
    run_state = (
        raw_run_state.expanduser().resolve()
        if raw_run_state is not None
        else output.with_name(f".{output.name}.glmrt-run")
    )
    _regular_directory(run_state, "run-state directory")
    plan = read_json_object(run_state / PLAN_FILENAME)
    _validate_plan(plan)
    _validate_run_state_entries(run_state)

    source = snapshot_identity(args.snapshot)
    parent_source = plan.get("source")
    if not isinstance(parent_source, dict):
        raise LaunchError("parent plan has no source identity")
    source_matches_parent = execution_upgrade_source_matches_parent(plan, source)
    texts, corpus = calibration_stream(args.calibration_jsonl)
    evidence = calibration_evidence(
        args.calibration_manifest,
        getattr(args, "route_screen_report", None),
        corpus=corpus,
        source=parent_source,
    )
    raw_projection = getattr(args, "projection_checkpoint_dir", None)
    raw_active_source = getattr(args, "active_layer_source_dir", None)
    expected_paths = {
        "output": os.fspath(output),
        "run_state_dir": os.fspath(run_state),
        "projection_checkpoint_dir": os.fspath(
            raw_projection.expanduser().resolve()
            if raw_projection is not None
            else run_state / PROJECTION_CHECKPOINT_DIRNAME
        ),
        "active_layer_source_dir": os.fspath(
            raw_active_source.expanduser().resolve()
            if raw_active_source is not None
            else run_state / ACTIVE_LAYER_SOURCE_DIRNAME
        ),
        "offload_dir": os.fspath(args.offload_dir.expanduser().resolve()),
    }
    parent_gpu_count = len(plan.get("preflight", {}).get("gpus", ()))
    if (
        any(plan.get(key) != value for key, value in expected_paths.items())
        or not source_matches_parent
        or plan.get("corpus") != corpus
        or plan.get("calibration_evidence") != evidence
        or plan.get("exl3", {}).get("bits") != getattr(args, "bits", 3)
        or plan.get("target_batch_size") != args.batch_size
        or parent_gpu_count != getattr(args, "coordinator_gpu_count", 2)
        or plan.get("remote_workers") is not None
    ):
        raise LaunchError("execution upgrade inputs differ from the parent plan")
    raw_seed = getattr(args, "projection_checkpoint_seed_dir", None)
    parent_seed = plan.get("projection_checkpoint_seed")
    if raw_seed is not None and (
        not isinstance(parent_seed, dict)
        or os.fspath(raw_seed.expanduser().resolve()) != parent_seed.get("root")
    ):
        raise LaunchError(
            "execution upgrade projection seed differs from the parent plan"
        )

    lock_path = args.gptqmodel_lock.expanduser().resolve(strict=True)
    current_lock = read_json_object(lock_path)
    if (
        current_lock.get("schema") != 1
        or current_lock.get("repository")
        != "https://github.com/tpurtell/GPTQModel.git"
        or REVISION_RE.fullmatch(str(current_lock.get("revision", ""))) is None
        or SHA256_RE.fullmatch(
            str(current_lock.get("source_tree_sha256", ""))
        )
        is None
    ):
        raise LaunchError("GPTQModel source lock is invalid")
    current_preflight = preflight_identity(
        args.preflight_report,
        str(current_lock["revision"]),
        expected_gpu_count=parent_gpu_count,
    )
    if (
        current_preflight["gptqmodel"].get("source_tree_sha256")
        != current_lock["source_tree_sha256"]
    ):
        raise LaunchError("upgrade preflight GPTQModel identity differs from its lock")

    parent_preflight = plan.get("preflight")
    parent_toolchain = plan.get("quantization_toolchain")
    if not isinstance(parent_preflight, dict) or not isinstance(
        parent_toolchain, dict
    ):
        raise LaunchError("parent plan lacks execution provenance")
    parent_gpu_ids = [
        (gpu.get("index"), gpu.get("uuid"))
        for gpu in parent_preflight.get("gpus", ())
        if isinstance(gpu, dict)
    ]
    current_gpu_ids = [
        (gpu.get("index"), gpu.get("uuid"))
        for gpu in current_preflight.get("gpus", ())
        if isinstance(gpu, dict)
    ]
    if (
        current_preflight.get("python") != parent_preflight.get("python")
        or current_preflight.get("torch") != parent_preflight.get("torch")
        or current_gpu_ids != parent_gpu_ids
    ):
        raise LaunchError(
            "execution upgrade changes the parent runtime or GPU identities"
        )
    current_toolchain = quantization_toolchain_identity()
    parent_execution = _stable_execution_identity(
        parent_preflight,
        parent_toolchain,
    )
    upgraded_execution = _stable_execution_identity(
        current_preflight,
        current_toolchain,
    )
    if upgraded_execution == parent_execution:
        raise LaunchError("execution upgrade does not change the execution code")

    upgrade_body = {
        "schema": EXECUTION_UPGRADE_SCHEMA,
        "parent_plan_sha256": plan["plan_sha256"],
        "parent_execution": parent_execution,
        "upgraded_execution": upgraded_execution,
        "change_contract": {
            "purpose": "rolling-layer-boundary-resume",
            "layer_boundary": BOUNDARY_CONTRACT,
            "capture_frontier": CAPTURE_FRONTIER_CONTRACT,
            "capture_batch_payload": (
                ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT
            ),
            "router_ranking": (
                "explicit-score-adapter-plus-live-topk-verification-v1"
            ),
            "projection_restore": "packed-checkpoint-direct-v1",
            "quantization_algorithm": "declared-seeded-true-sequential-v1",
            "forward_replay": "unchanged-parent-plan",
            "quantization_numerics": "unchanged-parent-family-join",
            "hardware": "unchanged-parent-gpu-identities",
            "remote_workers": "none",
        },
    }
    boundary_identity = _latest_boundary_resume_identity(run_state, plan)
    if boundary_identity is not None:
        upgrade_body["resume_state"] = {
            "contract": "latest-boundary-plus-journal-v1",
            "layer_boundary": boundary_identity,
            "error_journal": _journal_resume_identity(
                run_state / ERROR_JOURNAL_FILENAME
            ),
        }
        upgrade_body["change_contract"]["resume_state"] = (
            "content-bound-immutable-frontier"
        )

    upgrade_path = run_state / EXECUTION_UPGRADE_FILENAME
    if upgrade_path.exists():
        _repair_execution_upgrade_archive_staging(run_state, plan)
        saved_upgrade = _read_execution_upgrade(run_state, plan)
        if {
            key: value
            for key, value in saved_upgrade.items()
            if key
            not in {
                "upgrade_sha256",
                "previous_upgrade_sha256",
                "previous_failed_upgrade_sha256",
            }
        } == upgrade_body:
            return plan, texts, saved_upgrade
        if boundary_identity is not None:
            upgrade_body["previous_upgrade_sha256"] = saved_upgrade[
                "upgrade_sha256"
            ]
        else:
            upgrade_body["previous_failed_upgrade_sha256"] = saved_upgrade[
                "upgrade_sha256"
            ]
        upgrade = _bound_record(upgrade_body, "upgrade_sha256")
        history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
        history.mkdir(exist_ok=True)
        _regular_directory(history, "execution-upgrade history")
        archived = history / f"{saved_upgrade['upgrade_sha256']}.json"
        if archived.exists():
            if read_json_object(archived) != saved_upgrade:
                raise LaunchError("execution-upgrade history collision")
        else:
            atomic_json(archived, saved_upgrade)
        atomic_json(upgrade_path, upgrade)
    else:
        upgrade = _bound_record(upgrade_body, "upgrade_sha256")
        atomic_json(upgrade_path, upgrade)
    return plan, texts, upgrade


def runtime_ledger_provenance(plan: dict[str, Any]) -> dict[str, Any]:
    """Describe the actual executor without changing projection-family keys."""

    provenance = json.loads(json.dumps(plan["ledger_provenance"]))
    run_state = Path(plan["run_state_dir"])
    upgrade_path = run_state / EXECUTION_UPGRADE_FILENAME
    if upgrade_path.exists():
        upgrade = _read_execution_upgrade(run_state, plan)
        provenance["run"]["execution_upgrade"] = {
            "schema": upgrade["schema"],
            "upgrade_sha256": upgrade["upgrade_sha256"],
            "parent_plan_sha256": upgrade["parent_plan_sha256"],
            "upgraded_execution": upgrade["upgraded_execution"],
            "resume_state": upgrade.get("resume_state"),
        }
    return provenance


def preflight_lazy_nonpersistent_buffers(
    model: Any,
    *,
    device: Any,
    plan_sha256: str,
) -> dict[str, Any]:
    """Exercise constructor-only GLM buffers before expensive trellis work."""

    import torch

    target_device = torch.device(device)
    if target_device.type == "meta":
        raise LaunchError("lazy direct-state preflight cannot target the META device")
    candidates: list[tuple[str, Any, list[str]]] = []
    for module_name, module in model.model.named_modules():
        direct_buffers = dict(module.named_buffers(recurse=False))
        nonpersistent = set(
            getattr(module, "_non_persistent_buffers_set", set())
        )
        meta_names = sorted(
            name
            for name, buffer in direct_buffers.items()
            if name in nonpersistent and getattr(buffer, "is_meta", False)
        )
        if meta_names:
            candidates.append((module_name, module, meta_names))
    if not candidates:
        raise LaunchError(
            "lazy direct-state preflight found no constructor-only META buffers"
        )

    records: list[dict[str, Any]] = []
    for module_name, module, expected_names in candidates:
        model.shell_direct_meta_materialize(
            target_submodule=module,
            device=target_device,
        )
        direct_buffers = dict(module.named_buffers(recurse=False))
        remaining = sorted(
            name
            for name in expected_names
            if name not in direct_buffers
            or getattr(direct_buffers[name], "is_meta", False)
        )
        if remaining:
            raise LaunchError(
                "lazy direct-state preflight left META constructor buffers under "
                f"`{module_name}`: {remaining}"
            )
        nonpersistent = set(
            getattr(module, "_non_persistent_buffers_set", set())
        )
        for buffer_name in expected_names:
            if buffer_name not in nonpersistent:
                raise LaunchError(
                    "lazy direct-state preflight changed persistence for "
                    f"`{module_name}.{buffer_name}`"
                )
            buffer = direct_buffers[buffer_name]
            if buffer.device != target_device:
                raise LaunchError(
                    "lazy direct-state preflight restored a buffer on the wrong "
                    f"device: `{module_name}.{buffer_name}` is on {buffer.device}, "
                    f"expected {target_device}"
                )
            payload = (
                buffer.detach()
                .to(device="cpu")
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            records.append(
                {
                    "module": module_name,
                    "buffer": buffer_name,
                    "shape": list(buffer.shape),
                    "dtype": str(buffer.dtype),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    report = {
        "schema": "glmrt-glm52-lazy-direct-state-preflight-v1",
        "plan_sha256": plan_sha256,
        "device": str(target_device),
        "owner_count": len(candidates),
        "buffer_count": len(records),
        "owners": [
            {"module": name, "buffers": names}
            for name, _, names in candidates
        ],
        "buffers_sha256": hashlib.sha256(canonical_json(records)).hexdigest(),
    }
    print(
        "Lazy direct-state preflight restored "
        f"{report['buffer_count']} constructor-only buffers across "
        f"{report['owner_count']} owners on {target_device}.",
        flush=True,
    )
    return report


_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def _export_safetensors_inventory(
    root: Path,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Read and fully cross-check the exported index and shard headers."""

    index = read_json_object(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise LaunchError("exported safetensors index has no weight_map")
    names_by_file: dict[str, set[str]] = {}
    for name, shard in weight_map.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(shard, str)
            or Path(shard).name != shard
            or not shard.endswith(".safetensors")
        ):
            raise LaunchError("exported safetensors index contains an unsafe entry")
        names_by_file.setdefault(shard, set()).add(name)

    inventory: dict[str, tuple[str, tuple[int, ...]]] = {}
    for shard, expected_names in sorted(names_by_file.items()):
        path = root / shard
        if not path.is_file() or path.is_symlink():
            raise LaunchError(f"exported safetensors shard is not a regular file: {shard}")
        size = path.stat().st_size
        try:
            with path.open("rb") as source:
                prefix = source.read(8)
                if len(prefix) != 8:
                    raise LaunchError(f"exported safetensors shard has no header: {shard}")
                header_bytes = struct.unpack("<Q", prefix)[0]
                if header_bytes <= 0 or header_bytes > min(size - 8, 1 << 30):
                    raise LaunchError(
                        f"exported safetensors shard has an invalid header length: {shard}"
                    )
                header = json.loads(source.read(header_bytes))
        except LaunchError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as error:
            raise LaunchError(
                f"cannot inspect exported safetensors shard {shard}: {error}"
            ) from error
        if not isinstance(header, dict):
            raise LaunchError(f"exported safetensors header is not an object: {shard}")
        metadata = header.pop("__metadata__", None)
        if metadata is not None and not isinstance(metadata, dict):
            raise LaunchError(f"exported safetensors metadata is malformed: {shard}")
        if set(header) != expected_names:
            raise LaunchError(
                f"exported safetensors index/header names differ for {shard}"
            )
        ranges: list[tuple[int, int, str]] = []
        for name, entry in header.items():
            if not isinstance(entry, dict):
                raise LaunchError(f"exported tensor metadata is malformed: {name}")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if (
                dtype not in _SAFETENSORS_DTYPE_BYTES
                or not isinstance(shape, list)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in shape
                )
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in offsets
                )
                or offsets[0] < 0
                or offsets[1] < offsets[0]
                or offsets[1] - offsets[0]
                != math.prod(shape) * _SAFETENSORS_DTYPE_BYTES[dtype]
            ):
                raise LaunchError(f"exported tensor metadata is invalid: {name}")
            inventory[name] = (dtype, tuple(shape))
            ranges.append((offsets[0], offsets[1], name))
        cursor = 0
        for start, end, name in sorted(ranges):
            if start != cursor:
                raise LaunchError(
                    f"exported safetensors payload is not contiguous before {name}"
                )
            cursor = end
        if 8 + header_bytes + cursor != size:
            raise LaunchError(
                f"exported safetensors payload length differs for {shard}"
            )
    if set(inventory) != set(weight_map):
        raise LaunchError("exported safetensors inventory differs from its index")
    return inventory


def _expected_exl3_modules(plan: dict[str, Any]) -> dict[str, tuple[int, int]]:
    geometry = plan["source"]["geometry"]
    hidden = geometry["hidden_size"]
    intermediate = geometry["moe_intermediate_size"]
    experts = geometry["n_routed_experts"]
    modules: dict[str, tuple[int, int]] = {}
    for layer in range(
        geometry["first_target_layer"],
        geometry["last_target_layer"] + 1,
    ):
        for expert in range(experts):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            modules[f"{prefix}.gate_proj"] = (hidden, intermediate)
            modules[f"{prefix}.up_proj"] = (hidden, intermediate)
            modules[f"{prefix}.down_proj"] = (intermediate, hidden)
    return modules


def _expected_exl3_tensor_metadata(
    module: str,
    input_features: int,
    output_features: int,
    bits: int,
) -> dict[str, tuple[str, tuple[int, ...], str]]:
    if (
        input_features <= 0
        or output_features <= 0
        or input_features % 16
        or output_features % 16
    ):
        raise LaunchError(f"EXL3 module geometry is not tile aligned: {module}")
    return {
        f"{module}.trellis": (
            "I16",
            (input_features // 16, output_features // 16, 16 * bits),
            "int16",
        ),
        f"{module}.suh": ("F16", (input_features,), "float16"),
        f"{module}.svh": ("F16", (output_features,), "float16"),
        f"{module}.mcg": ("I32", (), "int32"),
    }


def compact_exl3_declaration(external: dict[str, Any]) -> dict[str, Any]:
    """Keep only EXL3 discovery fields in the model's main config."""

    return {
        field: external.get(field)
        for field in ("quant_method", "format", "checkpoint_format", "bits")
    }


def validate_export_quantization_contract(root: Path, plan: dict[str, Any]) -> None:
    """Fail closed before publication if model or EXL3 metadata is incomplete."""

    config = read_json_object(root / "config.json")
    external = read_json_object(root / "quantize_config.json")
    embedded = config.get("quantization_config")
    if not isinstance(embedded, dict):
        raise LaunchError("exported config.json has no quantization_config")
    source_config = read_json_object(Path(plan["source"]["path"]) / "config.json")
    # Quantization may update only its own declaration. It must not silently
    # turn GLM-5.3 into a different architecture/configuration.
    unexpected_config_fields = set(config) - (
        set(source_config) | {"quantization_config"}
    )
    if unexpected_config_fields:
        raise LaunchError(
            "exported model config added fields absent from the source: "
            f"{sorted(unexpected_config_fields)}"
        )
    for field, expected in source_config.items():
        if field == "quantization_config":
            continue
        if config.get(field) != expected:
            raise LaunchError(
                f"exported model config changed source field {field}: "
                f"{config.get(field)!r} != {expected!r}"
            )
    source_root = Path(plan["source"]["path"])
    for name in EXACT_SOURCE_METADATA_FILES:
        source_path = (source_root / name).resolve(strict=True)
        exported_path = root / name
        if _artifact_file_identity(source_path) != _artifact_file_identity(exported_path):
            raise LaunchError(f"exported source metadata differs: {name}")

    exl3 = plan["exl3"]
    numeric_bits = float(exl3["bits"])
    bits = int(numeric_bits)
    if numeric_bits != bits or bits <= 0:
        raise LaunchError(f"EXL3 export requires an integral bitrate, got {numeric_bits}")
    required = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": numeric_bits,
        "codebook": exl3["codebook"],
        "out_scales": exl3["out_scales"],
        "group_size": -1,
        "desc_act": False,
        "module_include": exl3["module_include"],
    }
    for field, expected in required.items():
        if external.get(field) != expected:
            raise LaunchError(
                f"exported external EXL3 field {field} differs from the plan"
            )
    compact = dict(external)
    storage = compact.pop("tensor_storage", None)
    if compact_exl3_declaration(external) != embedded:
        raise LaunchError(
            "config.json quantization_config is not the exact minimal EXL3 declaration"
        )

    modules = _expected_exl3_modules(plan)
    if not isinstance(storage, dict) or set(storage) != set(modules):
        actual = set(storage) if isinstance(storage, dict) else set()
        raise LaunchError(
            "exported tensor_storage module inventory differs: "
            f"expected={len(modules)} actual={len(actual)} "
            f"missing={sorted(set(modules) - actual)[:8]} "
            f"unexpected={sorted(actual - set(modules))[:8]}"
        )

    expected_tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for module, (input_features, output_features) in modules.items():
        expected = _expected_exl3_tensor_metadata(
            module,
            input_features,
            output_features,
            bits,
        )
        entry = storage[module]
        if (
            not isinstance(entry, dict)
            or set(entry)
            != {
                "stored_tensors",
                "quant_format",
                "bits_per_weight",
                "mcg_multiplier",
            }
            or entry.get("quant_format") != "exl3"
            or entry.get("bits_per_weight") != bits
            or entry.get("mcg_multiplier") != EXL3_MCG_MULTIPLIER
        ):
            raise LaunchError(f"exported tensor_storage entry is invalid: {module}")
        stored = entry.get("stored_tensors")
        if not isinstance(stored, dict) or set(stored) != set(expected):
            raise LaunchError(f"exported stored tensor set is incomplete: {module}")
        for name, (dtype, shape, torch_dtype) in expected.items():
            if stored[name] != {
                "shape": list(shape),
                "torch_dtype": torch_dtype,
            }:
                raise LaunchError(f"exported tensor_storage metadata is invalid: {name}")
            expected_tensors[name] = (dtype, shape)

    source_index = read_json_object(
        Path(plan["source"]["path"]) / "model.safetensors.index.json"
    )
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not source_weight_map:
        raise LaunchError("source model index has no weight_map during export validation")
    replaced = {f"{module}.weight" for module in modules}
    if plan["source"].get("format") == "fp8-e4m3-block128x128-dynamic":
        replaced.update(f"{module}.weight_scale_inv" for module in modules)
    if not replaced.issubset(source_weight_map):
        raise LaunchError("source model is missing tensors selected for EXL3 replacement")
    expected_inventory = (set(source_weight_map) - replaced) | set(expected_tensors)
    inventory = _export_safetensors_inventory(root)
    if set(inventory) != expected_inventory:
        raise LaunchError(
            "exported tensor namespace differs from source replacement contract: "
            f"missing={sorted(expected_inventory - set(inventory))[:8]} "
            f"unexpected={sorted(set(inventory) - expected_inventory)[:8]}"
        )
    for name, expected in expected_tensors.items():
        if inventory[name] != expected:
            raise LaunchError(
                f"exported EXL3 tensor metadata differs for {name}: "
                f"{inventory[name]!r} != {expected!r}"
            )


def normalize_export_model_config(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    """Replace Transformers serialization drift with exact source metadata."""

    source_config = read_json_object(Path(plan["source"]["path"]) / "config.json")
    exported_config = read_json_object(root / "config.json")
    external = read_json_object(root / "quantize_config.json")
    embedded = exported_config.get("quantization_config")
    compact_external = dict(external)
    compact_external.pop("tensor_storage", None)
    minimal_external = compact_exl3_declaration(external)
    # GPTQModel serializers have emitted both shapes over time: either the
    # metadata-rich declaration or the complete standalone object (including
    # the large tensor_storage map) duplicated into config.json. Both are safe
    # inputs only when they agree exactly with quantize_config.json. Always
    # normalize to the four-field discovery declaration while leaving the
    # standalone file byte-for-byte intact.
    if not isinstance(embedded, dict) or embedded not in (
        minimal_external,
        compact_external,
        external,
    ):
        raise LaunchError(
            "cannot normalize an export whose embedded and standalone EXL3 configurations disagree"
        )
    normalized = dict(source_config)
    normalized["quantization_config"] = minimal_external
    atomic_json(root / "config.json", normalized)
    source_root = Path(plan["source"]["path"])
    for name in EXACT_SOURCE_METADATA_FILES:
        source = (source_root / name).resolve(strict=True)
        destination = root / name
        if not source.is_file() or not destination.is_file() or destination.is_symlink():
            raise LaunchError(f"cannot normalize exported source metadata: {name}")
        shutil.copyfile(source, destination)
    return normalized


def publish_export(plan: dict[str, Any]) -> None:
    _validate_plan(plan)
    output = Path(plan["output"])
    run_state = Path(plan["run_state_dir"])
    export_stage = run_state / EXPORT_STAGE_DIRNAME
    if output.exists() or output.is_symlink():
        raise LaunchError(f"output appeared before artifact publication: {output}")
    atomic_json(export_stage / PLAN_FILENAME, plan)
    execution_upgrade = run_state / EXECUTION_UPGRADE_FILENAME
    upgrade = None
    if execution_upgrade.exists():
        upgrade = _read_execution_upgrade(run_state, plan)
        atomic_json(export_stage / EXECUTION_UPGRADE_FILENAME, upgrade)
        history = run_state / EXECUTION_UPGRADE_HISTORY_DIRNAME
        if history.exists():
            export_history = export_stage / EXECUTION_UPGRADE_HISTORY_DIRNAME
            export_history.mkdir()
            for archived in sorted(history.iterdir()):
                atomic_json(
                    export_history / archived.name,
                    read_json_object(archived),
                )
    normalize_export_model_config(export_stage, plan)
    validate_export_quantization_contract(export_stage, plan)
    manifest = write_artifact_manifest(export_stage, plan)
    geometry = plan["source"]["geometry"]
    run_schema, _ = artifact_contract_schemas(plan)
    run = _bound_record(
        {
            "schema": run_schema,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "execution_upgrade_sha256": (
                upgrade["upgrade_sha256"] if upgrade is not None else None
            ),
            "quantized_base_layers": list(
                range(
                    geometry["first_target_layer"],
                    geometry["last_target_layer"] + 1,
                )
            ),
            "preserved_mtp_layer": geometry["mtp_layer_index"],
        },
        "run_sha256",
    )
    atomic_json(export_stage / RUN_FILENAME, run)
    validate_published_artifact(
        export_stage,
        plan,
        verify_file_hashes=False,
    )
    _publish_validated_export_stage(export_stage, output)


def execute(
    plan: dict[str, Any],
    texts: list[str],
    *,
    resume: bool = False,
    stop_after_layer: int | None = None,
) -> None:
    jit_root = Path(plan["run_state_dir"]) / EXLLAMAV3_JIT_DIRNAME
    with forward_replica_scope(plan.get("forward_replay")), exllamav3_jit_cache_scope(
        jit_root
    ):
        _execute_bound(
            plan,
            texts,
            resume=resume,
            stop_after_layer=stop_after_layer,
        )


def _execute_bound(
    plan: dict[str, Any],
    texts: list[str],
    *,
    resume: bool = False,
    stop_after_layer: int | None = None,
) -> None:
    output = Path(plan["output"])
    if resume and (output.exists() or output.is_symlink()):
        if prepare_run(plan, resume=True):
            return
    if prepare_run(plan, resume=resume):
        return

    run_state = Path(plan["run_state_dir"])
    # The run-state emptiness/identity check above must happen before the
    # persistent compiler cache creates its canonical child directory.
    jit_root = run_state / EXLLAMAV3_JIT_DIRNAME
    jit_root.mkdir(parents=True, exist_ok=True)
    storage_report = validate_storage_capacity(plan)
    atomic_json(run_state / STORAGE_PREFLIGHT_FILENAME, storage_report)

    import torch
    from gptqmodel import GPTQModel
    from gptqmodel.models.definitions.glm_moe_dsa import GlmMoeDsaQModel
    from gptqmodel.quantization import AutoModuleDecoderConfig, EXL3Config

    export_stage = run_state / EXPORT_STAGE_DIRNAME
    offload = Path(plan["offload_dir"])
    os.environ["GPTQMODEL_EXL3_ERROR_JOURNAL"] = os.fspath(
        run_state / ERROR_JOURNAL_FILENAME
    )
    coordinator_devices = [
        f"cuda:{gpu['index']}" for gpu in plan["preflight"]["gpus"]
    ]
    if len(coordinator_devices) != 2:
        raise LaunchError("production quantization requires exactly two RTX GPUs")
    primary_device = coordinator_devices[0]
    ledger_provenance = runtime_ledger_provenance(plan)
    qcfg = EXL3Config(
        bits=float(plan["exl3"]["bits"]),
        codebook="mcg",
        out_scales="auto",
        module_include=[BASE_EXPERT_PATTERN],
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        fallback=None,
        offload_to_disk=True,
        offload_to_disk_path=os.fspath(offload),
        device=primary_device,
        calibration_data_device="cpu",
        dense_vram_strategy_devices=[primary_device],
        moe_vram_strategy="balanced",
        moe_vram_strategy_devices=coordinator_devices,
        meta={"ds4rt_error_ledger": ledger_provenance},
    )
    model = GPTQModel.load(
        plan["source"]["path"],
        quantize_config=qcfg,
        trust_remote_code=False,
    )
    if not isinstance(model, GlmMoeDsaQModel):
        raise LaunchError(f"unexpected GPTQModel definition: {type(model).__name__}")
    turtle = getattr(model, "turtle_model", None)
    configure_active_source = getattr(
        turtle, "configure_active_source_staging", None
    )
    if not callable(configure_active_source):
        raise LaunchError("GLM-5 lazy source cannot stage active layers")
    configure_active_source(
        plan["active_layer_source_dir"],
        provenance={
            "plan_sha256": plan["plan_sha256"],
            "source_revision": plan["source"]["revision"],
            "source_index_sha256": plan["source"]["index_sha256"],
        },
    )
    direct_state_report = preflight_lazy_nonpersistent_buffers(
        model,
        device=primary_device,
        plan_sha256=plan["plan_sha256"],
    )
    atomic_json(
        run_state / DIRECT_STATE_PREFLIGHT_FILENAME,
        direct_state_report,
    )

    geometry = plan["source"]["geometry"]
    boundary_contract = plan["layer_boundary"]
    boundary_store = Glm52LayerBoundaryStore(
        boundary_contract["root"],
        plan_sha256=plan["plan_sha256"],
        family_join=plan["ledger_provenance"]["family_join"],
        projection_checkpoint_root=plan["projection_checkpoint"]["root"],
        error_journal_path=run_state / ERROR_JOURNAL_FILENAME,
        hidden_size=geometry["hidden_size"],
        activation_rank=geometry["activation_rank"],
        routed_experts=geometry["n_routed_experts"],
        first_target_layer=geometry["first_target_layer"],
        last_target_layer=geometry["last_target_layer"],
    )
    boundary_controller = Glm52LayerBoundaryController(
        boundary_store,
        defer_publication_materialization=True,
        stop_after_layer=stop_after_layer,
    )
    model.quantization_layer_boundary_checkpoint = boundary_controller
    configure_base_replay = getattr(model, "configure_base_replay_store", None)
    if not callable(configure_base_replay):
        raise LaunchError("GLM-5 model cannot checkpoint base replay batches")
    configure_base_replay(
        run_state / POST_QUANT_REPLAY_DIRNAME,
        provenance={
            "plan_sha256": plan["plan_sha256"],
            "family_join": plan["ledger_provenance"]["family_join"],
            "output_dtype": "torch.bfloat16",
        },
    )

    with capture_frontier_scope(
        run_state / CAPTURE_FRONTIER_DIRNAME
    ), capture_batch_spool_scope(
        run_state / CAPTURE_BATCH_SPOOL_DIRNAME
    ), memory_safety_scope(plan["memory_safety"]):
        try:
            model.quantize(
                texts,
                batch_size=plan["target_batch_size"],
                calibration_sort=None,
            )
        except LayerBoundaryStop as stop:
            if stop_after_layer is None or stop.layer_index != stop_after_layer:
                raise LaunchError(
                    "quantization stopped at an unexpected layer boundary"
                ) from stop
            print(
                json.dumps(
                    {
                        "event": "quantization-stopped-after-durable-layer",
                        "layer": stop.layer_index,
                        "plan_sha256": plan["plan_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

    boundary_controller.materialize_deferred_prefix(
        model=model,
        force=True,
    )
    if export_stage.exists():
        raise LaunchError("export stage unexpectedly exists before publication")
    export_stage.mkdir()
    model.save(os.fspath(export_stage), max_shard_size="8GB")
    publish_export(plan)

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--calibration-jsonl", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--run-state-dir",
        type=Path,
        help="bounded durable NVMe run-state root",
    )
    parser.add_argument(
        "--projection-checkpoint-dir",
        type=Path,
        help="durable packed-projection root; scratch storage is recommended",
    )
    parser.add_argument(
        "--projection-checkpoint-seed-dir",
        type=Path,
        help=(
            "content-bind and clone a completed projection store into a fresh run"
        ),
    )
    parser.add_argument(
        "--active-layer-source-dir",
        type=Path,
        help="rolling NVMe staging root for only the currently active source layer",
    )
    parser.add_argument("--offload-dir", type=Path, required=True)
    parser.add_argument(
        "--gptqmodel-lock",
        type=Path,
        default=root / "third_party" / "gptqmodel.lock.json",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=(3, 4),
        default=3,
        help=(
            "EXL3 bitrate: K3 for GLM-5.2 BF16 or K4 for GLM-5.3 block-FP8"
        ),
    )
    parser.add_argument(
        "--coordinator-gpu-count",
        type=int,
        choices=(2,),
        default=2,
        help="production topology is exactly two visible RTX GPUs",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stop-after-layer",
        type=int,
        help=(
            "qualification stop after this routed decoder layer commits its "
            "rolling boundary"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only the exact content-bound unfinished run",
    )
    parser.add_argument(
        "--execution-upgrade",
        action="store_true",
        help=(
            "resume the immutable parent plan under separately content-bound "
            "checkpoint-only execution code"
        ),
    )
    args = parser.parse_args()
    args.route_screen_report = None
    args.remote_worker = None
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.stop_after_layer is not None and not 3 <= args.stop_after_layer <= 77:
        parser.error("--stop-after-layer must be in the routed range 3..77")
    if args.execution_upgrade and not args.resume:
        parser.error("--execution-upgrade requires --resume")
    return args


def main() -> int:
    args = parse_args()
    if args.execution_upgrade:
        plan, texts, upgrade = build_execution_upgrade(args)
        print(json.dumps(upgrade, indent=2, sort_keys=True), flush=True)
    else:
        plan, texts = build_plan(args)
    print(json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if not args.plan_only:
        execute(
            plan,
            texts,
            resume=args.resume,
            stop_after_layer=args.stop_after_layer,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LaunchError as exc:
        print(f"quantize-glm5-gptqmodel: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
