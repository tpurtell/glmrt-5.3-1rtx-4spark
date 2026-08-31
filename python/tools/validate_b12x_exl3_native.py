#!/usr/bin/env python3
"""Numerically validate glmrt's production-geometry EXL3 C ABI.

This test deliberately crosses the native boundary used by the Spark daemon:

* construct checkpoint-native K3/K4 MCG Trellis weights at GLM TP4 geometry;
* ask SparkInfer to prepare and pack the routes;
* quantize BF16 activations with glmrt's NVFP4 wire-payload kernel;
* execute the exported glmrt EXL3 kernel using Rust-equivalent device buffers;
* compare the result with SparkInfer on the exactly dequantized BF16 input.

Run this inside the Spark development image after building libglmrt_native.so.
The SparkInfer source tree must be importable (normally through PYTHONPATH).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
import heapq
import hashlib
import json
import math
import os
import re
import statistics
from pathlib import Path

# A disk-cache hit has no live kernel resource-introspection surface. Disable
# both caches before importing SparkInfer so every validation report contains
# the actual register and spill evidence for the pinned source.
os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

import _pinned_sparkinfer  # noqa: E402,F401
import torch  # noqa: E402

from _b12x_exl3_k3_profile import (  # noqa: E402
    EXL3_K3_AOT_REGIMES,
    exl3_k3_capacity_rows,
    exl3_k3_grid_x,
    exl3_k3_route_block_rows,
    exl3_k3_tile_config,
)
from _b12x_exl3_k4_profile import (  # noqa: E402
    EXL3_K4_AOT_REGIMES,
    exl3_k4_capacity_rows,
    exl3_k4_grid_x,
    exl3_k4_route_block_rows,
    exl3_k4_tile_config,
)
from b12x.moe import fused_moe  # noqa: E402
from b12x.moe._shared.kernels.w4a16.host import select_route_block_size_m  # noqa: E402
from b12x.moe._shared.kernels.w4a16.kernel import (  # noqa: E402
    _w4a16_fused_persistent_grid_x,
    _w4a16_num_regs,
)


EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 512
TOP_K = 8
BITS = 3
TRELLIS_WORDS = 16 * BITS
MAX_PACKED_ROUTE_SLOTS = 32_640
MAX_ROUTE_BLOCKS = 760
SCRATCH_ELEMENTS = 3_145_728
LOCK_ELEMENTS = 1_026
NVFP4_ROW_STRIDE = HIDDEN // 2 + HIDDEN // 16
EXPERT_TP_WORLD_SIZE = 4
EXPERT_MODULE_RE = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)
REPORT_SCHEMA = "glmrt-b12x-exl3-native-validation-v1"
ROUTE_PROFILE_SCHEMA = "glmrt-glm52-exl3-route-profile-v1"
GLM5_ROUTE_PROFILE_SCHEMA = "glmrt-glm5-exl3-route-profile-v1"
CHECKPOINT_SCHEMA = "ds4rt.exl3-projection-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ARTIFACT_MANIFEST_FILENAME = "glmrt-gptqmodel-artifact.json"
ARTIFACT_MANIFEST_SCHEMA = "glmrt-glm5-gptqmodel-artifact-v2"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"symbolic links are not accepted: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"not one regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _load_route_profile_sample(
    path: Path,
    sample_index: int,
    rows: int,
    trellis_bits: int | None = None,
) -> tuple[list[int], dict[str, object]]:
    if path.expanduser().is_symlink():
        raise ValueError("--route-profile must not be a symbolic link")
    resolved = path.expanduser().resolve(strict=True)
    try:
        profile = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid route profile: {error}") from error
    if not isinstance(profile, dict):
        raise ValueError("route profile must be a JSON object")
    report_sha256 = profile.get("report_sha256")
    body = {key: value for key, value in profile.items() if key != "report_sha256"}
    expected_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    if report_sha256 != expected_sha256:
        raise ValueError("route profile report_sha256 does not match its content")
    schema = profile.get("schema")
    geometry = profile.get("geometry")
    if (
        schema not in {ROUTE_PROFILE_SCHEMA, GLM5_ROUTE_PROFILE_SCHEMA}
        or profile.get("status") != "accepted"
        or (
            schema == ROUTE_PROFILE_SCHEMA
            and trellis_bits not in {None, 3}
        )
        or (
            schema == GLM5_ROUTE_PROFILE_SCHEMA
            and (
                not isinstance(geometry, dict)
                or geometry.get("trellis_bits") not in {3, 4}
                or trellis_bits is None
                or geometry["trellis_bits"] != trellis_bits
            )
        )
    ):
        raise ValueError("route profile is not accepted for this GLM-5 EXL3 bitrate")
    samples = profile.get("samples")
    if not isinstance(samples, list) or not 0 <= sample_index < len(samples):
        raise ValueError(
            f"route profile sample {sample_index} is outside 0..{len(samples) - 1 if isinstance(samples, list) else -1}"
        )
    sample = samples[sample_index]
    if not isinstance(sample, dict) or sample.get("rows") != rows:
        raise ValueError(
            f"route profile sample {sample_index} does not have requested rows={rows}"
        )
    sparse_counts = sample.get("expert_route_counts")
    if not isinstance(sparse_counts, list):
        raise ValueError("route profile sample has no expert_route_counts")
    dense_counts = [0] * EXPERTS
    seen: set[int] = set()
    for pair in sparse_counts:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or isinstance(pair[0], bool)
            or not isinstance(pair[0], int)
            or isinstance(pair[1], bool)
            or not isinstance(pair[1], int)
        ):
            raise ValueError("route profile expert counts are malformed")
        expert_id, count = pair
        if not 0 <= expert_id < EXPERTS or count <= 0 or expert_id in seen:
            raise ValueError("route profile expert counts are out of range or duplicated")
        seen.add(expert_id)
        dense_counts[expert_id] = count
    identity = _file_identity(resolved)
    return dense_counts, {
        "kind": "content-bound-route-profile-sample",
        "profile": identity,
        "profile_report_sha256": report_sha256,
        "capture_id": profile.get("capture_id"),
        "sample_index": sample_index,
        "sample": sample,
    }


class DeviceBuffer(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("bytes", ctypes.c_size_t),
        ("device_id", ctypes.c_int),
        ("flags", ctypes.c_uint64),
    ]


class Exl3Buffers(ctypes.Structure):
    _fields_ = [
        ("input_bf16", DeviceBuffer),
        ("rotation_a_gate", DeviceBuffer),
        ("rotation_a_up", DeviceBuffer),
        ("w13_trellis", DeviceBuffer),
        ("w2_trellis", DeviceBuffer),
        ("unit_global_scale", DeviceBuffer),
        ("fc1_output", DeviceBuffer),
        ("activated", DeviceBuffer),
        ("fc2_output", DeviceBuffer),
        ("output_f32", DeviceBuffer),
        ("packed_route_indices", DeviceBuffer),
        ("block_expert_ids", DeviceBuffer),
        ("packed_route_count", DeviceBuffer),
        ("topk_ids", DeviceBuffer),
        ("topk_weights", DeviceBuffer),
        ("fc1_scratch", DeviceBuffer),
        ("fc2_scratch", DeviceBuffer),
        ("locks", DeviceBuffer),
        ("intermediate_rotations", DeviceBuffer),
        ("gate_suh", DeviceBuffer),
        ("up_suh", DeviceBuffer),
        ("down_svh", DeviceBuffer),
    ]


def _capacity(rows: int) -> int:
    return exl3_k3_capacity_rows(rows) if BITS == 3 else exl3_k4_capacity_rows(rows)


def _aot_regimes() -> tuple[int, ...]:
    return EXL3_K3_AOT_REGIMES if BITS == 3 else EXL3_K4_AOT_REGIMES


def _grid_x(capacity_rows: int) -> int:
    return (
        exl3_k3_grid_x(capacity_rows)
        if BITS == 3
        else exl3_k4_grid_x(capacity_rows)
    )


def _route_block_rows(capacity_rows: int) -> int:
    return (
        exl3_k3_route_block_rows(capacity_rows)
        if BITS == 3
        else exl3_k4_route_block_rows(capacity_rows)
    )


def _tile_config(capacity_rows: int) -> tuple[int, int, int, int]:
    return (
        exl3_k3_tile_config(capacity_rows)
        if BITS == 3
        else exl3_k4_tile_config(capacity_rows)
    )


def _device_buffer(tensor: torch.Tensor) -> DeviceBuffer:
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("native buffers must be contiguous CUDA tensors")
    return DeviceBuffer(
        ctypes.c_void_p(tensor.data_ptr()),
        tensor.numel() * tensor.element_size(),
        tensor.device.index or 0,
        0,
    )


def _last_error(library: ctypes.CDLL) -> str:
    message = ctypes.create_string_buffer(2048)
    status = library.glmrt_last_error(message, len(message))
    if status != 0:
        return f"glmrt_last_error failed with status {status}"
    return message.value.decode("utf-8", errors="replace")


def _check_status(library: ctypes.CDLL, status: int, operation: str) -> None:
    if status != 0:
        raise RuntimeError(f"{operation} returned status {status}: {_last_error(library)}")


def _load_native(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    library.glmrt_last_error.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    library.glmrt_last_error.restype = ctypes.c_int
    library.glmrt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async.argtypes = [
        DeviceBuffer,
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    library.glmrt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async.restype = ctypes.c_int
    launch = getattr(
        library, f"glmrt_cuda_b12x_spark_exl3_k{BITS}_topk8_nvfp4_async"
    )
    launch.argtypes = [
        ctypes.POINTER(Exl3Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    launch.restype = ctypes.c_int
    library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_async = launch
    launch_bf16 = getattr(
        library, f"glmrt_cuda_b12x_spark_exl3_k{BITS}_topk8_nvfp4_bf16_async"
    )
    launch_bf16.argtypes = [
        ctypes.POINTER(Exl3Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    launch_bf16.restype = ctypes.c_int
    library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_bf16_async = launch_bf16
    candidate = getattr(
        library,
        f"glmrt_cuda_b12x_spark_exl3_k{BITS}_topk8_nvfp4_capacity_candidate_async",
    )
    candidate.argtypes = [
        ctypes.POINTER(Exl3Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    candidate.restype = ctypes.c_int
    library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_capacity_candidate_async = candidate
    grid_candidate = getattr(
        library,
        f"glmrt_cuda_b12x_spark_exl3_k{BITS}_topk8_nvfp4_capacity_grid_candidate_async",
    )
    grid_candidate.argtypes = [
        ctypes.POINTER(Exl3Buffers),
        DeviceBuffer,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    grid_candidate.restype = ctypes.c_int
    library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_capacity_grid_candidate_async = (
        grid_candidate
    )
    return library


def _make_weight_tensors(device: torch.device) -> tuple[torch.Tensor, ...]:
    w13 = torch.empty(
        (2, EXPERTS, HIDDEN // 16, INTERMEDIATE // 16, TRELLIS_WORDS),
        dtype=torch.int16,
        device=device,
    ).random_(-32768, 32767)
    w2 = torch.empty(
        (EXPERTS, INTERMEDIATE // 16, HIDDEN // 16, TRELLIS_WORDS),
        dtype=torch.int16,
        device=device,
    ).random_(-32768, 32767)

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (0.875 + 0.25 * torch.rand(shape, device=device)).to(torch.float16)

    return (
        w13,
        w2,
        scales((EXPERTS, HIDDEN)).contiguous(),
        scales((EXPERTS, HIDDEN)).contiguous(),
        scales((EXPERTS, 3 * INTERMEDIATE)).contiguous(),
        scales((EXPERTS, HIDDEN)).contiguous(),
    )


def _checkpoint_projection_files(
    root: Path,
    layer_id: int,
) -> tuple[dict[tuple[int, str], Path], dict[str, object]]:
    projections: dict[tuple[int, str], Path] = {}
    inventory: list[dict[str, object]] = []
    for manifest_path in root.rglob("*.json"):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"checkpoint manifest is not one regular file: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"checkpoint manifest is not an object: {manifest_path}")
        request = manifest.get("request")
        module = request.get("module") if isinstance(request, dict) else None
        match = EXPERT_MODULE_RE.fullmatch(module or "")
        if match is None or int(match.group("layer")) != layer_id:
            continue
        manifest_digest = manifest.get("manifest_sha256")
        manifest_body = {
            key: value for key, value in manifest.items() if key != "manifest_sha256"
        }
        request_digest = request.get("request_sha256")
        request_body = {
            key: value for key, value in request.items() if key != "request_sha256"
        }
        if (
            manifest.get("schema") != CHECKPOINT_SCHEMA
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or not isinstance(manifest_digest, str)
            or not SHA256_RE.fullmatch(manifest_digest)
            or hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
            != manifest_digest
            or not isinstance(request_digest, str)
            or not SHA256_RE.fullmatch(request_digest)
            or hashlib.sha256(_canonical_json(request_body)).hexdigest() != request_digest
        ):
            raise ValueError(f"checkpoint manifest digest is invalid: {manifest_path}")
        key = (int(match.group("expert")), match.group("projection"))
        tensor_file = manifest.get("tensor_file")
        if not isinstance(tensor_file, str) or Path(tensor_file).name != tensor_file:
            raise ValueError(f"invalid tensor_file in {manifest_path}")
        tensor_path = manifest_path.with_name(tensor_file)
        if tensor_path.is_symlink() or not tensor_path.is_file():
            raise ValueError(f"projection tensor is not one regular file: {tensor_path}")
        tensor_digest = manifest.get("tensor_sha256")
        if (
            not isinstance(tensor_digest, str)
            or not SHA256_RE.fullmatch(tensor_digest)
            or _sha256_file(tensor_path) != tensor_digest
        ):
            raise ValueError(f"checkpoint tensor digest is invalid: {tensor_path}")
        if key in projections:
            raise ValueError(
                f"duplicate layer {layer_id} expert {key[0]} {key[1]} checkpoint"
            )
        projections[key] = tensor_path
        inventory.append(
            {
                "module": module,
                "manifest": str(manifest_path.relative_to(root)),
                "manifest_sha256": manifest_digest,
                "tensor": str(tensor_path.relative_to(root)),
                "tensor_bytes": tensor_path.stat().st_size,
                "tensor_sha256": tensor_digest,
            }
        )

    expected = {
        (expert_id, projection)
        for expert_id in range(EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }
    if projections.keys() != expected:
        missing = sorted(expected - projections.keys())[:8]
        unexpected = sorted(projections.keys() - expected)[:8]
        raise ValueError(
            f"layer {layer_id} projection checkpoint set is incomplete: "
            f"expected={len(expected)} found={len(projections)} "
            f"missing={missing} unexpected={unexpected}"
        )
    inventory.sort(key=lambda record: str(record["module"]))
    inventory_bytes = sum(int(record["tensor_bytes"]) for record in inventory)
    return projections, {
        "projection_count": len(inventory),
        "tensor_bytes": inventory_bytes,
        "inventory_sha256": hashlib.sha256(_canonical_json(inventory)).hexdigest(),
    }


def _load_checkpoint_weight_tensors(
    root: Path,
    layer_id: int,
    tp_rank: int,
    device: torch.device,
    trellis_bits: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Assemble the exact resident TP4 layout from calibrated checkpoints."""

    if not 0 <= tp_rank < EXPERT_TP_WORLD_SIZE:
        raise ValueError(
            f"TP rank must be in 0..{EXPERT_TP_WORLD_SIZE - 1}, got {tp_rank}"
        )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "loading calibrated projection checkpoints requires safetensors"
        ) from exc

    projection_files, checkpoint_identity = _checkpoint_projection_files(root, layer_id)
    trellis_words = 16 * trellis_bits
    shard_start = tp_rank * INTERMEDIATE
    shard_end = shard_start + INTERMEDIATE
    gate_trellis: list[torch.Tensor] = []
    up_trellis: list[torch.Tensor] = []
    down_trellis: list[torch.Tensor] = []
    gate_suh: list[torch.Tensor] = []
    up_suh: list[torch.Tensor] = []
    intermediate_rotations: list[torch.Tensor] = []
    down_svh: list[torch.Tensor] = []

    expected_trellis_shapes = {
        "gate_proj": (
            HIDDEN // 16,
            INTERMEDIATE * EXPERT_TP_WORLD_SIZE // 16,
            trellis_words,
        ),
        "up_proj": (
            HIDDEN // 16,
            INTERMEDIATE * EXPERT_TP_WORLD_SIZE // 16,
            trellis_words,
        ),
        "down_proj": (
            INTERMEDIATE * EXPERT_TP_WORLD_SIZE // 16,
            HIDDEN // 16,
            trellis_words,
        ),
    }
    for expert_id in range(EXPERTS):
        loaded: dict[str, dict[str, torch.Tensor]] = {}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            tensors = load_file(
                str(projection_files[(expert_id, projection)]), device="cpu"
            )
            if set(tensors) != {"trellis", "suh", "svh", "mcg"}:
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} {projection} has tensor "
                    f"set {sorted(tensors)}"
                )
            if (
                tensors["trellis"].dtype != torch.int16
                or tuple(tensors["trellis"].shape)
                != expected_trellis_shapes[projection]
                or tensors["suh"].dtype != torch.float16
                or tensors["svh"].dtype != torch.float16
                or tensors["mcg"].dtype != torch.int32
                or tensors["mcg"].shape != torch.Size([])
                or int(tensors["mcg"].item()) & 0xFFFFFFFF != 0xCBAC1FED
            ):
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} {projection} violates "
                    f"the calibrated K{trellis_bits}/MCG tensor contract"
                )
            loaded[projection] = tensors

        gate = loaded["gate_proj"]
        up = loaded["up_proj"]
        down = loaded["down_proj"]
        if (
            tuple(gate["suh"].shape) != (HIDDEN,)
            or tuple(up["suh"].shape) != (HIDDEN,)
            or tuple(gate["svh"].shape) != (INTERMEDIATE * EXPERT_TP_WORLD_SIZE,)
            or tuple(up["svh"].shape) != (INTERMEDIATE * EXPERT_TP_WORLD_SIZE,)
            or tuple(down["suh"].shape) != (INTERMEDIATE * EXPERT_TP_WORLD_SIZE,)
            or tuple(down["svh"].shape) != (HIDDEN,)
        ):
            raise ValueError(
                f"layer {layer_id} expert {expert_id} has invalid EXL3 rotations"
            )

        gate_trellis.append(
            gate["trellis"][:, shard_start // 16 : shard_end // 16, :].contiguous()
        )
        up_trellis.append(
            up["trellis"][:, shard_start // 16 : shard_end // 16, :].contiguous()
        )
        down_trellis.append(
            down["trellis"][shard_start // 16 : shard_end // 16, :, :].contiguous()
        )
        gate_suh.append(gate["suh"])
        up_suh.append(up["suh"])
        intermediate_rotations.append(
            torch.cat(
                (
                    gate["svh"][shard_start:shard_end],
                    up["svh"][shard_start:shard_end],
                    down["suh"][shard_start:shard_end],
                )
            )
        )
        down_svh.append(down["svh"])

    cpu_tensors = (
        torch.stack((torch.stack(gate_trellis), torch.stack(up_trellis))),
        torch.stack(down_trellis),
        torch.stack(gate_suh),
        torch.stack(up_suh),
        torch.stack(intermediate_rotations),
        torch.stack(down_svh),
    )
    return (
        tuple(tensor.contiguous().to(device) for tensor in cpu_tensors),
        checkpoint_identity,
    )


def _load_artifact_weight_tensors(
    root: Path,
    layer_id: int,
    tp_rank: int,
    device: torch.device,
    trellis_bits: int,
) -> tuple[tuple[torch.Tensor, ...], dict[str, object]]:
    """Assemble one resident TP4 rank directly from a finalized HF artifact."""

    if not 0 <= tp_rank < EXPERT_TP_WORLD_SIZE:
        raise ValueError(
            f"TP rank must be in 0..{EXPERT_TP_WORLD_SIZE - 1}, got {tp_rank}"
        )
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("loading a finalized model requires safetensors") from exc

    def json_object(path: Path) -> dict[str, object]:
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid finalized-model metadata: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"finalized-model metadata is not an object: {path}")
        return value

    manifest_path = root / ARTIFACT_MANIFEST_FILENAME
    index_path = root / "model.safetensors.index.json"
    quantization_path = root / "quantize_config.json"
    manifest = json_object(manifest_path)
    manifest_digest = manifest.get("manifest_sha256")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    if (
        manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA
        or not isinstance(manifest_digest, str)
        or SHA256_RE.fullmatch(manifest_digest) is None
        or hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
        != manifest_digest
    ):
        raise ValueError("finalized-model artifact manifest is invalid")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("finalized-model artifact manifest has no file inventory")

    index = json_object(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("finalized-model tensor index has no weight_map")
    quantization = json_object(quantization_path)
    tensor_storage = quantization.get("tensor_storage")
    if (
        quantization.get("method") != "exl3"
        or quantization.get("bits") != trellis_bits
        or not isinstance(tensor_storage, dict)
    ):
        raise ValueError(
            f"finalized model does not satisfy the EXL3 K{trellis_bits} contract"
        )

    module_names = [
        f"model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}"
        for expert_id in range(EXPERTS)
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    tensor_names: list[str] = []
    for module in module_names:
        storage = tensor_storage.get(module)
        expected_names = {
            f"{module}.trellis",
            f"{module}.suh",
            f"{module}.svh",
            f"{module}.mcg",
        }
        if (
            not isinstance(storage, dict)
            or storage.get("quant_format") != "exl3"
            or storage.get("bits_per_weight") != trellis_bits
            or set(storage.get("stored_tensors", ())) != expected_names
        ):
            raise ValueError(f"invalid EXL3 tensor-storage record: {module}")
        tensor_names.extend(sorted(expected_names))

    selected_shards: set[str] = set()
    for tensor_name in tensor_names:
        shard = weight_map.get(tensor_name)
        if not isinstance(shard, str) or not shard.endswith(".safetensors"):
            raise ValueError(f"missing finalized tensor index entry: {tensor_name}")
        selected_shards.add(shard)

    selected_files = {
        "model.safetensors.index.json",
        "quantize_config.json",
        *selected_shards,
    }
    file_identities: list[dict[str, object]] = []
    for name in sorted(selected_files):
        record = files.get(name)
        path = root / name
        if (
            not isinstance(record, dict)
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
            or not path.is_file()
            or path.stat().st_size != record["bytes"]
            or _sha256_file(path) != record["sha256"]
        ):
            raise ValueError(f"finalized-model file failed authentication: {name}")
        file_identities.append({"name": name, **record})

    trellis_words = 16 * trellis_bits
    shard_start = tp_rank * INTERMEDIATE
    shard_end = shard_start + INTERMEDIATE
    gate_trellis: list[torch.Tensor] = []
    up_trellis: list[torch.Tensor] = []
    down_trellis: list[torch.Tensor] = []
    gate_suh: list[torch.Tensor] = []
    up_suh: list[torch.Tensor] = []
    intermediate_rotations: list[torch.Tensor] = []
    down_svh: list[torch.Tensor] = []
    expected_trellis_shapes = {
        "gate_proj": (
            HIDDEN // 16,
            INTERMEDIATE * EXPERT_TP_WORLD_SIZE // 16,
            trellis_words,
        ),
        "up_proj": (
            HIDDEN // 16,
            INTERMEDIATE * EXPERT_TP_WORLD_SIZE // 16,
            trellis_words,
        ),
        "down_proj": (
            INTERMEDIATE * EXPERT_TP_WORLD_SIZE // 16,
            HIDDEN // 16,
            trellis_words,
        ),
    }

    with ExitStack() as stack:
        readers = {
            name: stack.enter_context(
                safe_open(root / name, framework="pt", device="cpu")
            )
            for name in selected_shards
        }

        def load_tensor(name: str) -> torch.Tensor:
            shard = weight_map[name]
            assert isinstance(shard, str)
            return readers[shard].get_tensor(name)

        for expert_id in range(EXPERTS):
            loaded: dict[str, dict[str, torch.Tensor]] = {}
            for projection in ("gate_proj", "up_proj", "down_proj"):
                module = (
                    f"model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}"
                )
                tensors = {
                    suffix: load_tensor(f"{module}.{suffix}")
                    for suffix in ("trellis", "suh", "svh", "mcg")
                }
                if (
                    tensors["trellis"].dtype != torch.int16
                    or tuple(tensors["trellis"].shape)
                    != expected_trellis_shapes[projection]
                    or tensors["suh"].dtype != torch.float16
                    or tensors["svh"].dtype != torch.float16
                    or tensors["mcg"].dtype != torch.int32
                    or tensors["mcg"].shape != torch.Size([])
                    or int(tensors["mcg"].item()) & 0xFFFFFFFF != 0xCBAC1FED
                ):
                    raise ValueError(
                        f"layer {layer_id} expert {expert_id} {projection} violates "
                        f"the finalized K{trellis_bits}/MCG tensor contract"
                    )
                loaded[projection] = tensors

            gate = loaded["gate_proj"]
            up = loaded["up_proj"]
            down = loaded["down_proj"]
            if (
                tuple(gate["suh"].shape) != (HIDDEN,)
                or tuple(up["suh"].shape) != (HIDDEN,)
                or tuple(gate["svh"].shape)
                != (INTERMEDIATE * EXPERT_TP_WORLD_SIZE,)
                or tuple(up["svh"].shape)
                != (INTERMEDIATE * EXPERT_TP_WORLD_SIZE,)
                or tuple(down["suh"].shape)
                != (INTERMEDIATE * EXPERT_TP_WORLD_SIZE,)
                or tuple(down["svh"].shape) != (HIDDEN,)
            ):
                raise ValueError(
                    f"layer {layer_id} expert {expert_id} has invalid EXL3 rotations"
                )
            gate_trellis.append(
                gate["trellis"][
                    :, shard_start // 16 : shard_end // 16, :
                ].contiguous()
            )
            up_trellis.append(
                up["trellis"][:, shard_start // 16 : shard_end // 16, :].contiguous()
            )
            down_trellis.append(
                down["trellis"][shard_start // 16 : shard_end // 16, :, :].contiguous()
            )
            gate_suh.append(gate["suh"])
            up_suh.append(up["suh"])
            intermediate_rotations.append(
                torch.cat(
                    (
                        gate["svh"][shard_start:shard_end],
                        up["svh"][shard_start:shard_end],
                        down["suh"][shard_start:shard_end],
                    )
                )
            )
            down_svh.append(down["svh"])

    cpu_tensors = (
        torch.stack((torch.stack(gate_trellis), torch.stack(up_trellis))),
        torch.stack(down_trellis),
        torch.stack(gate_suh),
        torch.stack(up_suh),
        torch.stack(intermediate_rotations),
        torch.stack(down_svh),
    )
    identity = {
        "artifact_manifest_sha256": manifest_digest,
        "plan_sha256": manifest.get("plan_sha256"),
        "authenticated_files": file_identities,
        "projection_count": len(module_names),
        "tensor_count": len(tensor_names),
    }
    return tuple(tensor.contiguous().to(device) for tensor in cpu_tensors), identity


def _prepare_weights(
    tensors: tuple[torch.Tensor, ...],
    tile_config: tuple[int, int, int, int],
) -> fused_moe.ExpertWeights:
    w13, w2, gate_suh, up_suh, intermediate_rotations, down_svh = tensors

    weight_plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        w13_layout="w13",
        trellis_bits=BITS,
        trellis_codebook="mcg",
        trellis_tile_config=tile_config,
    )
    return fused_moe.prepare_weights(
        plan=weight_plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        trellis_mcg=0xCBAC1FED,
    )


def _copy_prefix(destination: torch.Tensor, source: torch.Tensor) -> None:
    flat_destination = destination.view(-1)
    flat_source = source.view(-1)
    if flat_source.numel() > flat_destination.numel():
        raise ValueError(
            f"source with {flat_source.numel()} elements exceeds native capacity "
            f"{flat_destination.numel()}"
        )
    flat_destination[: flat_source.numel()].copy_(flat_source)


def _route_ids_from_counts(
    counts: list[int], rows: int, top_k: int
) -> list[list[int]]:
    """Realize observed expert degrees without repeating an expert in a row."""
    row_heap = [(0, row_id) for row_id in range(rows)]
    heapq.heapify(row_heap)
    assignments = [[] for _ in range(rows)]
    for expert_id, count in sorted(
        enumerate(counts), key=lambda item: (-item[1], item[0])
    ):
        selected = [heapq.heappop(row_heap) for _ in range(count)]
        for degree, row_id in selected:
            assignments[row_id].append(expert_id)
            heapq.heappush(row_heap, (degree + 1, row_id))
    if any(len(assignment) != top_k for assignment in assignments):
        raise ValueError("expert route counts do not realize a balanced top-k plan")
    return assignments


def _validate_route_counts(counts: object, rows: int) -> list[int]:
    if (
        not isinstance(counts, list)
        or len(counts) != EXPERTS
        or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in counts
        )
        or any(count > rows for count in counts)
        or sum(counts) != rows * TOP_K
    ):
        raise ValueError(
            "expert route counts must contain 256 non-negative integers, "
            "each no larger than rows and summing to rows * 8"
        )
    _route_ids_from_counts(counts, rows, TOP_K)
    return counts


def _run_case(
    library: ctypes.CDLL,
    experts: fused_moe.ExpertWeights,
    rows: int,
    device: torch.device,
    source_scale: float,
    benchmark_iterations: int,
    benchmark_rounds: int,
    benchmark_warmup: int,
    compare_capacity_rows: int | None,
    compare_grid_x: int | None,
    force_grid_x: int | None,
    expert_route_counts: list[int] | None,
    bf16_output: bool,
) -> dict[str, object]:
    capacity = _capacity(rows)
    if capacity > 2064:
        raise ValueError(f"rows={rows} exceeds the largest exported EXL3 regime")
    candidate_capacity = compare_capacity_rows or capacity
    allocation_capacity = max(capacity, candidate_capacity)
    block_size = select_route_block_size_m(capacity, TOP_K, EXPERTS)
    if block_size != _route_block_rows(capacity):
        raise ValueError(
            f"SparkInfer EXL3 M={capacity} route-block ABI no longer matches "
            "the exported glmrt profile"
        )
    candidate_block_size = select_route_block_size_m(
        candidate_capacity, TOP_K, EXPERTS
    )
    if candidate_block_size != block_size:
        raise ValueError(
            "capacity comparison crosses a route-block ABI boundary: "
            f"M={capacity} uses {block_size}, candidate M={candidate_capacity} "
            f"uses {candidate_block_size}"
        )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=capacity,
            num_topk=TOP_K,
            route_num_experts=EXPERTS,
            device=device,
            weight_plan=experts.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=block_size,
        )
    )
    launch_token_count = rows if capacity <= 32 else capacity
    fused_launches = dict(plan._prewarmed_fused_launches)
    fused_launch = fused_launches[launch_token_count]
    automatic_grid_x = _w4a16_fused_persistent_grid_x(
        fused=fused_launch,
        m=capacity,
        topk=TOP_K,
        intermediate_size=INTERMEDIATE,
        activation="silu",
        direct_topk_routes=False,
        sms=48,
    )
    production_grid_x = _grid_x(capacity)
    if production_grid_x > automatic_grid_x:
        raise ValueError(
            f"profile grid {production_grid_x} exceeds compiled M={capacity} "
            f"cooperative limit {automatic_grid_x}"
        )
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=device)
    # Arbitrary Trellis code streams can have much smaller effective scales
    # than trained weights.  Keep enough input energy for a two-linear MoE
    # result that cannot pass the comparison vacuously as an all-zero tensor.
    source = (torch.randn((rows, HIDDEN), device=device) * source_scale).to(
        torch.bfloat16
    )
    if expert_route_counts is None:
        topk_ids = torch.stack(
            [torch.randperm(EXPERTS, device=device)[:TOP_K] for _ in range(rows)]
        ).to(torch.int32)
    else:
        topk_ids = torch.tensor(
            _route_ids_from_counts(expert_route_counts, rows, TOP_K),
            dtype=torch.int32,
            device=device,
        )
    topk_weights = torch.rand((rows, TOP_K), dtype=torch.float32, device=device)
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)

    # A first SparkInfer run both supplies the oracle specialization and creates
    # the exact expert-packed route metadata consumed by the AOT C entry point.
    binding = plan.bind(
        scratch=scratch,
        a=source,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    binding.run()
    torch.cuda.synchronize(device)

    packed_routes = torch.full(
        (MAX_PACKED_ROUTE_SLOTS,), -1, dtype=torch.int32, device=device
    )
    block_experts = torch.full(
        (MAX_ROUTE_BLOCKS,), -1, dtype=torch.int32, device=device
    )
    packed_count = torch.zeros((1,), dtype=torch.int32, device=device)
    _copy_prefix(packed_routes, binding.packed_route_indices)
    _copy_prefix(block_experts, binding.block_expert_ids)
    _copy_prefix(packed_count, binding.packed_route_count)

    native_topk_ids = torch.full(
        (allocation_capacity, TOP_K), -1, dtype=torch.int32, device=device
    )
    native_topk_weights = torch.zeros(
        (allocation_capacity, TOP_K), dtype=torch.float32, device=device
    )
    native_topk_ids[:rows].copy_(topk_ids)
    native_topk_weights[:rows].copy_(topk_weights)

    prepared = experts.representation_for("w4a16")
    input_bf16 = torch.empty(
        (allocation_capacity, HIDDEN), dtype=torch.bfloat16, device=device
    )
    rotation_gate = torch.empty(
        (allocation_capacity * TOP_K, HIDDEN), dtype=torch.float16, device=device
    )
    rotation_up = torch.empty_like(rotation_gate)
    fc1 = torch.empty(
        (allocation_capacity * TOP_K, 2 * INTERMEDIATE),
        dtype=torch.float16,
        device=device,
    )
    activated = torch.empty(
        (allocation_capacity * TOP_K, INTERMEDIATE),
        dtype=torch.float16,
        device=device,
    )
    fc2 = torch.empty(
        (allocation_capacity * TOP_K, HIDDEN), dtype=torch.float16, device=device
    )
    native_output = torch.full(
        (allocation_capacity, HIDDEN), math.nan, dtype=torch.float32, device=device
    )
    fc1_scratch = torch.empty((SCRATCH_ELEMENTS,), dtype=torch.float32, device=device)
    fc2_scratch = torch.empty_like(fc1_scratch)
    locks = torch.empty((LOCK_ELEMENTS,), dtype=torch.int32, device=device)
    payload = torch.empty((rows, NVFP4_ROW_STRIDE), dtype=torch.uint8, device=device)
    unit_global_scale = torch.ones((EXPERTS,), dtype=torch.float32, device=device)

    buffers = Exl3Buffers(
        _device_buffer(input_bf16),
        _device_buffer(rotation_gate),
        _device_buffer(rotation_up),
        _device_buffer(prepared.w13),
        _device_buffer(prepared.w2),
        _device_buffer(unit_global_scale),
        _device_buffer(fc1),
        _device_buffer(activated),
        _device_buffer(fc2),
        _device_buffer(native_output),
        _device_buffer(packed_routes),
        _device_buffer(block_experts),
        _device_buffer(packed_count),
        _device_buffer(native_topk_ids),
        _device_buffer(native_topk_weights),
        _device_buffer(fc1_scratch),
        _device_buffer(fc2_scratch),
        _device_buffer(locks),
        _device_buffer(prepared.intermediate_rotations),
        _device_buffer(prepared.gate_suh),
        _device_buffer(prepared.up_suh),
        _device_buffer(prepared.down_svh),
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    _check_status(
        library,
        library.glmrt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async(
            _device_buffer(source),
            _device_buffer(payload),
            rows,
            HIDDEN,
            stream,
        ),
        "quantizing the BF16 input to an NVFP4 row payload",
    )
    def launch_native() -> None:
        if force_grid_x is not None:
            _check_status(
                library,
                library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_capacity_grid_candidate_async(
                    ctypes.byref(buffers),
                    _device_buffer(payload),
                    NVFP4_ROW_STRIDE,
                    rows,
                    capacity,
                    force_grid_x,
                    stream,
                ),
                f"launching the native EXL3 K{BITS} MoE at grid {force_grid_x}",
            )
            return
        launch = (
            library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_bf16_async
            if bf16_output
            else library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_async
        )
        _check_status(
            library,
            launch(
                ctypes.byref(buffers),
                _device_buffer(payload),
                NVFP4_ROW_STRIDE,
                rows,
                stream,
            ),
            f"launching the native EXL3 K{BITS} MoE",
        )

    def launch_capacity_candidate() -> None:
        if compare_grid_x is None:
            assert compare_capacity_rows is not None
            status = library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_capacity_candidate_async(
                ctypes.byref(buffers),
                _device_buffer(payload),
                NVFP4_ROW_STRIDE,
                rows,
                candidate_capacity,
                stream,
            )
        else:
            grid_candidate = (
                library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_capacity_grid_candidate_async
            )
            status = grid_candidate(
                ctypes.byref(buffers),
                _device_buffer(payload),
                NVFP4_ROW_STRIDE,
                rows,
                candidate_capacity,
                compare_grid_x,
                stream,
            )
        _check_status(
            library,
            status,
            f"launching the native EXL3 K{BITS} capacity/grid candidate",
        )

    # The fused native launch owns the NVFP4 decoder that materializes
    # input_bf16.  Preserve the decoder output only after that launch has run;
    # cloning it immediately after the payload quantizer would copy
    # uninitialized allocator memory.  The BF16-output epilogue reuses
    # input_bf16 for its result, so run the FP32-output form first in that case.
    fp32_epilogue_reference = None
    if bf16_output:
        _check_status(
            library,
            library.glmrt_cuda_b12x_spark_exl3_topk8_nvfp4_async(
                ctypes.byref(buffers),
                _device_buffer(payload),
                NVFP4_ROW_STRIDE,
                rows,
                stream,
            ),
            f"launching the native EXL3 K{BITS} FP32 reference epilogue",
        )
        torch.cuda.synchronize(device)
        reference_input = input_bf16[:rows].clone()
        fp32_epilogue_reference = native_output[:rows].clone()
    else:
        launch_native()
        torch.cuda.synchronize(device)
        reference_input = input_bf16[:rows].clone()

    launch_native()
    torch.cuda.synchronize(device)
    samples_ms: list[float] = []
    comparison_samples_ms: list[float] = []
    if benchmark_iterations > 0:
        for _ in range(benchmark_warmup):
            launch_native()
            if compare_capacity_rows is not None or compare_grid_x is not None:
                launch_capacity_candidate()
        torch.cuda.synchronize(device)

        def measure(launch) -> float:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(benchmark_iterations):
                launch()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / benchmark_iterations

        for round_index in range(benchmark_rounds):
            has_comparison = (
                compare_capacity_rows is not None or compare_grid_x is not None
            )
            if has_comparison and round_index % 2:
                comparison_samples_ms.append(measure(launch_capacity_candidate))
                samples_ms.append(measure(launch_native))
            else:
                samples_ms.append(measure(launch_native))
                if compare_capacity_rows is not None or compare_grid_x is not None:
                    comparison_samples_ms.append(measure(launch_capacity_candidate))
    launch_native()
    torch.cuda.synchronize(device)
    actual = (
        input_bf16[:rows].float().clone()
        if bf16_output
        else native_output[:rows].clone()
    )

    # Compare to SparkInfer using the exact BF16 values produced by glmrt's
    # NVFP4 decoder, so the check isolates the native binding and execution.
    reference_binding = None
    if bf16_output:
        assert fp32_epilogue_reference is not None
        reference = fp32_epilogue_reference
    else:
        reference_binding = plan.bind(
            scratch=scratch,
            a=reference_input,
            experts=experts,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
        )
        reference = reference_binding.run().clone()
        torch.cuda.synchronize(device)

    actual_finite = torch.isfinite(actual)
    reference_finite = torch.isfinite(reference)
    if not actual_finite.all() or not reference_finite.all():
        native_fc2_nonfinite = int((~torch.isfinite(fc2[: rows * TOP_K])).sum())
        if reference_binding is None:
            reference_fc2_nonfinite = 0
        else:
            reference_fc2 = reference_binding.intermediate_cache13.view(-1)[
                : capacity * TOP_K * HIDDEN
            ].view(capacity * TOP_K, HIDDEN)
            reference_fc2_nonfinite = int(
                (~torch.isfinite(reference_fc2[: rows * TOP_K])).sum()
            )
        raise AssertionError(
            f"non-finite output in EXL3 case rows={rows}: "
            f"native={int((~actual_finite).sum())}, "
            f"oracle={int((~reference_finite).sum())}, "
            f"native_fc2={native_fc2_nonfinite}, "
            f"oracle_fc2={reference_fc2_nonfinite}, "
            f"packed_count={int(packed_count.item())}, "
            f"dequant_nonzero={int(torch.count_nonzero(input_bf16[:rows]))}, "
            f"w13_scale={tuple(prepared.w13_scale.shape)}/{prepared.w13_scale.dtype}, "
            f"w2_scale={tuple(prepared.w2_scale.shape)}/{prepared.w2_scale.dtype}, "
            f"w13_global={tuple(prepared.w13_global_scale.shape)}/{prepared.w13_global_scale.dtype}, "
            f"w2_global={tuple(prepared.w2_global_scale.shape)}/{prepared.w2_global_scale.dtype}, "
            f"w13_global_head={prepared.w13_global_scale[:4].tolist()}, "
            f"w2_global_head={prepared.w2_global_scale[:4].tolist()}, "
            f"w13_scale_bytes={prepared.w13_scale.tolist()}, "
            f"w2_scale_bytes={prepared.w2_scale.tolist()}"
            f", capacity={capacity}, production_grid={production_grid_x}, "
            f"automatic_grid={automatic_grid_x}, forced_grid={force_grid_x}"
        )
    reference_norm = reference.norm()
    if float(reference_norm) <= 1.0e-9:
        raise AssertionError(f"vacuous all-zero EXL3 oracle at rows={rows}")
    difference = actual - reference
    relative_l2 = float(difference.norm() / reference_norm)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), reference.flatten(), dim=0
        )
    )
    maximum_absolute = float(difference.abs().max())
    if relative_l2 > 2.0e-2 or cosine < 0.999:
        raise AssertionError(
            f"EXL3 native mismatch at rows={rows}: relative_l2={relative_l2:.6g}, "
            f"cosine={cosine:.9f}, max_abs={maximum_absolute:.6g}"
        )
    registers_per_thread = getattr(fused_launch, "registers_per_thread", None)
    local_memory_bytes = getattr(fused_launch, "local_memory_bytes", None)
    if registers_per_thread is None:
        # Newer SparkInfer compile results keep only launch-relevant fields.
        # Reconstruct the same pinned SM121 resource value used by its
        # occupancy planner; every catalogued specialization is spill-free.
        cta_m_blocks = (int(fused_launch.moe_block_size) + 15) // 16
        registers_per_thread = max(
            _w4a16_num_regs(
                cta_threads=fused_launch.cta_threads,
                cta_m_blocks=cta_m_blocks,
                cta_n_blocks=tile_n // 16,
                cta_k_blocks=tile_k // 16,
                uses_m_block_8=fused_launch.moe_block_size == 8,
                weight_layout=fused_launch.weight_layout,
            )
            for tile_n, tile_k in (
                (fused_launch.fc1_tile_n, fused_launch.fc1_tile_k),
                (fused_launch.fc2_tile_n, fused_launch.fc2_tile_k),
            )
        )
        local_memory_bytes = 0
    result: dict[str, object] = {
        "rows": rows,
        "capacity_rows": capacity,
        "grid_x": production_grid_x,
        "automatic_grid_x": automatic_grid_x,
        "route_block_rows": block_size,
        "packed_route_count": int(packed_count.item()),
        "fc1_tile": [fused_launch.fc1_tile_k, fused_launch.fc1_tile_n],
        "fc2_tile": [fused_launch.fc2_tile_k, fused_launch.fc2_tile_n],
        "blocks_per_sm": fused_launch.blocks_per_sm,
        "registers_per_thread": registers_per_thread,
        "local_memory_bytes": local_memory_bytes,
        "source_scale": source_scale,
        "expert_route_counts": expert_route_counts,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_abs": maximum_absolute,
        "output_dtype": "bf16" if bf16_output else "fp32",
        "reference": "native-fp32-epilogue" if bf16_output else "sparkinfer",
    }
    if samples_ms:
        result.update(
            {
                "benchmark_iterations": benchmark_iterations,
                "benchmark_rounds": benchmark_rounds,
                "median_ms": statistics.median(samples_ms),
                "minimum_ms": min(samples_ms),
                "samples_ms": samples_ms,
            }
        )
    if comparison_samples_ms:
        production_median = statistics.median(samples_ms)
        comparison_median = statistics.median(comparison_samples_ms)
        result["candidate_comparison"] = {
            "capacity_rows": candidate_capacity,
            "grid_x": compare_grid_x,
            "route_block_rows": candidate_block_size,
            "median_ms": comparison_median,
            "minimum_ms": min(comparison_samples_ms),
            "samples_ms": comparison_samples_ms,
            "delta_percent": 100.0 * (comparison_median / production_median - 1.0),
        }
    return result


def _atomic_json(path: Path, value: dict[str, object]) -> None:
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


def main() -> None:
    global BITS, TRELLIS_WORDS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument(
        "--trellis-bits",
        type=int,
        choices=(3, 4),
        default=3,
        help="checkpoint and AOT Trellis bitrate (default: K3)",
    )
    parser.add_argument(
        "--rows",
        default="1,3,129",
        help="comma-separated live row counts (each is bucketed to an AOT capacity)",
    )
    weight_source_group = parser.add_mutually_exclusive_group()
    weight_source_group.add_argument(
        "--projection-checkpoint-dir",
        type=Path,
        help=(
            "optional calibrated projection store; when set, assemble real "
            "checkpoint tensors instead of deterministic synthetic tensors"
        ),
    )
    weight_source_group.add_argument(
        "--model-snapshot",
        type=Path,
        help=(
            "optional finalized EXL3 Hugging Face artifact; assemble the exact "
            "resident TP4 rank directly from its indexed safetensors"
        ),
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=3,
        help="calibrated decoder layer to load from the selected real weight source",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=0,
        help="TP4 intermediate rank to assemble from calibrated checkpoints",
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--synthetic-source-scale",
        type=float,
        help="override the deterministic synthetic activation scale",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=0,
        help="when positive, time this many complete native launches per round",
    )
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    parser.add_argument("--benchmark-warmup", type=int, default=20)
    parser.add_argument(
        "--bf16-output",
        action="store_true",
        help=(
            "validate the FP32-accumulating, BF16-storing full-rotation "
            "epilogue used by row-sharded prefill"
        ),
    )
    parser.add_argument(
        "--compare-capacity-rows",
        type=int,
        help=(
            "benchmark this alternate exported AOT capacity against production "
            "dispatch in the same process"
        ),
    )
    parser.add_argument(
        "--compare-grid-x",
        type=int,
        help=(
            "benchmark this positive persistent-grid size at the production or "
            "--compare-capacity-rows bucket; requires one --rows case"
        ),
    )
    parser.add_argument(
        "--force-grid-x",
        type=int,
        help="validate using this positive grid candidate instead of production dispatch",
    )
    route_source = parser.add_mutually_exclusive_group()
    route_source.add_argument(
        "--expert-route-counts-json",
        type=Path,
        help=(
            "dense 256-entry route-count vector from a live route profile; "
            "requires exactly one --rows case"
        ),
    )
    route_source.add_argument(
        "--route-profile",
        type=Path,
        help="accepted route-profile report containing the selected live fixture",
    )
    parser.add_argument(
        "--route-profile-sample",
        type=int,
        help="zero-based sample index to replay from --route-profile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional content-bound JSON evidence destination",
    )
    parser.add_argument(
        "--expert-slot-fingerprint",
        help=(
            "lowercase SHA-256 fingerprint of the expert WIP slot containing "
            "--native-library; required for checkpoint-backed output evidence"
        ),
    )
    args = parser.parse_args()
    BITS = args.trellis_bits
    TRELLIS_WORDS = 16 * BITS
    rows = [int(value) for value in args.rows.split(",") if value.strip()]
    if not rows or any(value <= 0 for value in rows):
        raise SystemExit("--rows must contain positive integers")
    if (
        args.output is not None
        and (
            args.projection_checkpoint_dir is not None
            or args.model_snapshot is not None
        )
        and (
            not isinstance(args.expert_slot_fingerprint, str)
            or SHA256_RE.fullmatch(args.expert_slot_fingerprint) is None
        )
    ):
        raise SystemExit(
            "real-weight --output evidence requires "
            "--expert-slot-fingerprint"
        )
    if args.synthetic_source_scale is not None and (
        not math.isfinite(args.synthetic_source_scale)
        or args.synthetic_source_scale <= 0.0
    ):
        raise SystemExit("--synthetic-source-scale must be positive and finite")
    if (
        args.benchmark_iterations < 0
        or args.benchmark_rounds < 1
        or args.benchmark_warmup < 0
    ):
        raise SystemExit(
            "benchmark iterations/warmup must be nonnegative and rounds positive"
        )
    if args.compare_capacity_rows is not None and (
        args.benchmark_iterations <= 0
        or len(rows) != 1
        or args.compare_capacity_rows < rows[0]
        or args.compare_capacity_rows not in _aot_regimes()
    ):
        raise SystemExit(
            "--compare-capacity-rows requires one row case, positive benchmark "
            "iterations, and an exported AOT capacity no smaller than live rows"
        )
    if args.compare_grid_x is not None and (
        args.benchmark_iterations <= 0
        or len(rows) != 1
        or args.compare_grid_x <= 0
    ):
        raise SystemExit(
            "--compare-grid-x requires one row case, positive benchmark "
            "iterations, and a positive grid size"
        )
    if args.force_grid_x is not None and (
        len(rows) != 1 or args.force_grid_x <= 0 or args.bf16_output
    ):
        raise SystemExit(
            "--force-grid-x requires one row case, a positive grid, and FP32 output"
        )
    if args.bf16_output and (
        args.compare_capacity_rows is not None or args.compare_grid_x is not None
    ):
        raise SystemExit("--bf16-output cannot be combined with capacity/grid comparison")
    expert_route_counts = None
    route_fixture = None
    if (args.route_profile is None) != (args.route_profile_sample is None):
        raise SystemExit("--route-profile and --route-profile-sample are required together")
    if args.route_profile_sample is not None and args.route_profile_sample < 0:
        raise SystemExit("--route-profile-sample must be non-negative")
    if args.expert_route_counts_json is not None:
        if len(rows) != 1:
            raise SystemExit("--expert-route-counts-json requires exactly one row case")
        route_path = args.expert_route_counts_json.expanduser()
        if route_path.is_symlink():
            raise SystemExit("--expert-route-counts-json must not be a symbolic link")
        route_path = route_path.resolve(strict=True)
        try:
            expert_route_counts = json.loads(route_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid expert route-count JSON: {error}") from error
        try:
            expert_route_counts = _validate_route_counts(
                expert_route_counts, rows[0]
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        route_fixture = _file_identity(route_path)
    elif args.route_profile is not None:
        if len(rows) != 1:
            raise SystemExit("--route-profile requires exactly one row case")
        try:
            expert_route_counts, route_fixture = _load_route_profile_sample(
                args.route_profile,
                args.route_profile_sample,
                rows[0],
                BITS,
            )
            expert_route_counts = _validate_route_counts(
                expert_route_counts, rows[0]
            )
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
    if not torch.cuda.is_available():
        raise SystemExit("an SM120/SM121 CUDA device is required")

    device = torch.device("cuda", torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(device)
    if properties.major != 12 or properties.minor not in (0, 1):
        raise SystemExit(
            f"an SM120/SM121 CUDA device is required, got sm_{properties.major}{properties.minor}"
        )
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.native_library.expanduser().is_symlink():
        raise SystemExit("--native-library must not be a symbolic link")
    native_library = args.native_library.expanduser().resolve(strict=True)
    library = _load_native(native_library)
    if args.projection_checkpoint_dir is None and args.model_snapshot is None:
        weight_tensors = _make_weight_tensors(device)
        # Arbitrary random MCG streams include extreme decoded values that are
        # unlike calibrated model weights. Keep a bounded non-vacuous stimulus
        # for both K3 and K4; _run_case independently rejects an all-zero oracle.
        source_scale = (
            args.synthetic_source_scale
            if args.synthetic_source_scale is not None
            else 0.002
        )
        weight_source: dict[str, object] = {
            "kind": "deterministic-synthetic",
            "seed": args.seed,
        }
    elif args.projection_checkpoint_dir is not None:
        if args.projection_checkpoint_dir.expanduser().is_symlink():
            raise SystemExit("--projection-checkpoint-dir must not be a symbolic link")
        checkpoint_root = args.projection_checkpoint_dir.expanduser().resolve(
            strict=True
        )
        if not checkpoint_root.is_dir():
            raise SystemExit(
                f"projection checkpoint directory does not exist: {checkpoint_root}"
            )
        weight_tensors, checkpoint_identity = _load_checkpoint_weight_tensors(
            checkpoint_root,
            args.layer_id,
            args.tp_rank,
            device,
            BITS,
        )
        source_scale = 1.0
        weight_source = {
            "kind": "calibrated-projection-checkpoints",
            "root": str(checkpoint_root),
            "layer_id": args.layer_id,
            "tp_rank": args.tp_rank,
            "tp_world_size": EXPERT_TP_WORLD_SIZE,
            **checkpoint_identity,
        }
    else:
        model_root = args.model_snapshot.expanduser().resolve(strict=True)
        if not model_root.is_dir():
            raise SystemExit(f"model snapshot does not exist: {model_root}")
        try:
            weight_tensors, artifact_identity = _load_artifact_weight_tensors(
                model_root,
                args.layer_id,
                args.tp_rank,
                device,
                BITS,
            )
        except (OSError, ValueError) as error:
            raise SystemExit(str(error)) from error
        source_scale = 1.0
        weight_source = {
            "kind": "finalized-exl3-artifact",
            "root": str(model_root),
            "layer_id": args.layer_id,
            "tp_rank": args.tp_rank,
            "tp_world_size": EXPERT_TP_WORLD_SIZE,
            **artifact_identity,
        }
    tile_configs = {_tile_config(_capacity(value)) for value in rows}
    experts_by_tile = {
        tile_config: _prepare_weights(weight_tensors, tile_config)
        for tile_config in tile_configs
    }
    results = [
        _run_case(
            library,
            experts_by_tile[_tile_config(_capacity(value))],
            value,
            device,
            source_scale,
            args.benchmark_iterations,
            args.benchmark_rounds,
            args.benchmark_warmup,
            args.compare_capacity_rows,
            args.compare_grid_x,
            args.force_grid_x,
            expert_route_counts,
            args.bf16_output,
        )
        for value in rows
    ]
    body: dict[str, object] = {
        "schema": REPORT_SCHEMA,
        "status": "accepted",
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "expert_slot_fingerprint": args.expert_slot_fingerprint,
        "trellis_bits": BITS,
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "native_library": _file_identity(native_library),
        "device": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "weight_source": weight_source,
        "route_fixture": route_fixture,
        "cases": results,
    }
    report = {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }
    if args.output is not None:
        _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
