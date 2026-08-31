#!/usr/bin/env python3
"""Performance-gate single-residency W8A16 DFlash2 body projections.

This is a post-quantization measurement tool, not a serving-format converter.
It tunes layer 0 over every live DFlash2 row shape, then validates the selected
tile for the same shape against layers 1--5.  A projection class is promotable
only when W8 wins every row/layer case and passes a conservative numerical
sanity check.  Full held-out draft acceptance remains the authoritative gate.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
import triton

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, os.fspath(TOOLS))
from tune_w8a16_projection import check_status, metrics  # noqa: E402
from tune_w8a16_triton_prefill import w8a16_group256_gemm  # noqa: E402

REPO_ID = "incoai/GLM-5.3-DFlash2"
REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
WEIGHT_BYTES = 4_918_859_112
GROUP_SIZE = 256
MIN_W8_SPEEDUP = 1.01
MAX_RELATIVE_L2 = 0.10
MIN_COSINE = 0.99
LIVE_ROWS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32)
SMALL_ROW_CONFIGS = (
    (16, 64, 64, 4, 3),
    (16, 128, 64, 4, 3),
    (16, 256, 64, 8, 3),
    (16, 64, 128, 4, 3),
    (32, 64, 64, 4, 3),
    (32, 128, 64, 8, 3),
    (32, 256, 64, 8, 3),
    (32, 64, 128, 8, 3),
    (32, 128, 128, 8, 3),
)


@dataclass(frozen=True)
class Projection:
    name: str
    source_suffixes: tuple[str, ...]
    output_dim: int
    input_dim: int


PROJECTIONS = {
    projection.name: projection
    for projection in (
        Projection(
            "qkv",
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ),
            10_240,
            6_144,
        ),
        Projection("attention-output", ("self_attn.o_proj.weight",), 6_144, 8_192),
        Projection(
            "gate-up",
            ("mlp.gate_proj.weight", "mlp.up_proj.weight"),
            24_576,
            6_144,
        ),
        Projection("down", ("mlp.down_proj.weight",), 6_144, 12_288),
        Projection(
            "attention-conv",
            ("attention_conv.kernel_projection.weight",),
            1_536,
            6_144,
        ),
        Projection(
            "mlp-conv",
            ("mlp_conv.kernel_projection.weight",),
            1_536,
            6_144,
        ),
    )
}
BF16_PROJECTION_BYTES_PER_PASS = 6 * sum(
    projection.output_dim * projection.input_dim * 2
    for projection in PROJECTIONS.values()
)
W8_PROJECTION_BYTES_PER_PASS = 6 * sum(
    projection.output_dim * projection.input_dim
    + projection.output_dim * (projection.input_dim // GROUP_SIZE) * 4
    for projection in PROJECTIONS.values()
)
if (
    BF16_PROJECTION_BYTES_PER_PASS != 4_303_355_904
    or W8_PROJECTION_BYTES_PER_PASS != 2_185_297_920
):
    raise RuntimeError("internal DFlash2 projection byte geometry changed")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_ints(
    value: str,
    *,
    label: str,
    allowed: set[int],
) -> tuple[int, ...]:
    if value == "dflash" and label == "rows":
        return LIVE_ROWS
    try:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if (
        not values
        or len(values) != len(set(values))
        or any(item not in allowed for item in values)
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must contain unique values from {sorted(allowed)}"
        )
    return values


def _parse_config(value: str) -> tuple[int, int, int, int, int]:
    try:
        config = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "config must be BLOCK_M,BLOCK_N,BLOCK_K,warps,stages"
        ) from error
    if len(config) != 5 or any(item <= 0 for item in config):
        raise argparse.ArgumentTypeError("config must contain five positive integers")
    return config  # type: ignore[return-value]


def _config_label(config: tuple[int, int, int, int, int]) -> str:
    block_m, block_n, block_k, warps, stages = config
    return f"m{block_m}-n{block_n}-k{block_k}-w{warps}-s{stages}"


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "maximum": ordered[-1],
    }


def _configure_quantizer(path: Path):
    native = ctypes.CDLL(os.fspath(path.resolve(strict=True)))
    quantize = native.glmrt_cuda_quantize_bf16_w8a16_group256_async
    quantize.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    quantize.restype = ctypes.c_int
    return quantize


def _load_projection(
    weight_path: Path,
    projection: Projection,
    layer: int,
    device: torch.device,
) -> torch.Tensor:
    names = tuple(f"layers.{layer}.{suffix}" for suffix in projection.source_suffixes)
    with safe_open(weight_path, framework="pt", device=str(device)) as weights:
        pieces = [weights.get_tensor(name) for name in names]
    weight = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    expected = (projection.output_dim, projection.input_dim)
    if tuple(weight.shape) != expected or weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"DFlash2 {projection.name} layer {layer} changed: "
            f"{tuple(weight.shape)} {weight.dtype} != {expected} BF16"
        )
    return weight.contiguous()


def _quantize_weight(
    weight: torch.Tensor,
    quantize: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    output_dim, input_dim = weight.shape
    packed = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(
        (output_dim, input_dim // GROUP_SIZE),
        dtype=torch.float32,
        device=weight.device,
    )
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    check_status(
        quantize(
            weight.data_ptr(),
            packed.data_ptr(),
            scales.data_ptr(),
            input_dim,
            output_dim,
            0,
            stream,
        ),
        "DFlash2 row-major W8A16 quantization",
    )
    torch.cuda.synchronize()
    return packed, scales


def _capture(launch: Callable[[], None]) -> torch.cuda.CUDAGraph:
    current = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current)
    with torch.cuda.stream(warmup_stream):
        launch()
    current.wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    torch.cuda.synchronize()
    return graph


def _measure_cold_pair(
    bf16_graph: torch.cuda.CUDAGraph,
    w8_graph: torch.cuda.CUDAGraph,
    flush: torch.Tensor,
    *,
    warmup: int,
    rounds: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        bf16_graph.replay()
        w8_graph.replay()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"bf16": [], "w8a16": []}
    for round_index in range(rounds):
        order = (
            (("bf16", bf16_graph), ("w8a16", w8_graph))
            if round_index % 2 == 0
            else (("w8a16", w8_graph), ("bf16", bf16_graph))
        )
        for label, graph in order:
            # The flush is ordered before the start event on the same stream,
            # so its time is excluded while the measured projection sees the
            # cold-weight condition of a real six-layer draft pass.
            flush.add_(1)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[label].append(start.elapsed_time(end))
    return _summary(samples["bf16"]), _summary(samples["w8a16"])


def _case(
    *,
    projection: Projection,
    layer: int,
    rows: int,
    config: tuple[int, int, int, int, int],
    weight: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    flush: torch.Tensor,
    seed: int,
    warmup: int,
    rounds: int,
) -> dict[str, Any]:
    generator = torch.Generator(device=weight.device)
    generator.manual_seed(seed + layer * 10_000 + rows * 100 + projection.output_dim)
    activation = torch.randn(
        (rows, projection.input_dim),
        dtype=torch.bfloat16,
        device=weight.device,
        generator=generator,
    )
    bf16_output = torch.empty(
        (rows, projection.output_dim), dtype=torch.bfloat16, device=weight.device
    )
    w8_output = torch.empty_like(bf16_output)
    block_m, block_n, block_k, warps, stages = config
    if projection.output_dim % block_n or projection.input_dim % block_k:
        raise RuntimeError(
            f"tile {_config_label(config)} does not divide {projection.name}"
        )
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(projection.output_dim, block_n),)

    def launch_bf16() -> None:
        torch.mm(activation, weight.T, out=bf16_output)

    def launch_w8() -> None:
        w8a16_group256_gemm[grid](
            activation,
            packed,
            scales,
            w8_output,
            M=rows,
            N=projection.output_dim,
            K=projection.input_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=8,
            DEQUANT_BF16=False,
            POST_SCALE_GROUP=False,
            ROW_MAJOR_WEIGHT=True,
            num_warps=warps,
            num_stages=stages,
        )

    bf16_graph = _capture(launch_bf16)
    w8_graph = _capture(launch_w8)
    bf16_graph.replay()
    w8_graph.replay()
    torch.cuda.synchronize()
    quality = metrics(w8_output, bf16_output)
    finite = all(
        float(value) == float(value) and abs(float(value)) != float("inf")
        for value in quality.values()
    )
    numerical_sanity = (
        finite
        and quality["relative_l2"] <= MAX_RELATIVE_L2
        and quality["cosine"] >= MIN_COSINE
    )
    bf16_timing, w8_timing = _measure_cold_pair(
        bf16_graph,
        w8_graph,
        flush,
        warmup=warmup,
        rounds=rounds,
    )
    speedup = bf16_timing["median"] / w8_timing["median"]
    return {
        "projection": projection.name,
        "layer": layer,
        "rows": rows,
        "config": list(config),
        "config_label": _config_label(config),
        "bf16_ms": bf16_timing,
        "w8a16_ms": w8_timing,
        "w8a16_speedup": speedup,
        "quality": quality,
        "numerical_sanity": numerical_sanity,
        "performance_gate_passed": speedup >= MIN_W8_SPEEDUP,
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument(
        "--projection",
        action="append",
        choices=tuple(PROJECTIONS),
        help="projection class to test; omit to test all classes",
    )
    parser.add_argument("--layers", default="0,1,2,3,4,5")
    parser.add_argument("--rows", default="dflash")
    parser.add_argument(
        "--config",
        action="append",
        type=_parse_config,
        help="candidate BLOCK_M,BLOCK_N,BLOCK_K,warps,stages; repeat as needed",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--flush-mib", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        layers = _parse_ints(args.layers, label="layers", allowed=set(range(6)))
        rows = _parse_ints(args.rows, label="rows", allowed=set(LIVE_ROWS))
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if 0 not in layers:
        parser.error("--layers must include tuning layer 0")
    if args.warmup < 1 or args.rounds < 3:
        parser.error("warmup must be positive and rounds must be at least three")
    if args.flush_mib < 128:
        parser.error("flush-mib must be at least 128 for the coordinator L2 gate")
    if args.output.exists() or args.output.is_symlink():
        parser.error(f"refusing to overwrite output: {args.output}")
    configs = tuple(args.config or SMALL_ROW_CONFIGS)
    selected_projections = tuple(args.projection or PROJECTIONS)

    snapshot = (
        args.snapshot.expanduser().resolve(strict=True)
        if args.snapshot is not None
        else Path(snapshot_download(REPO_ID, revision=REVISION, local_files_only=True))
    )
    if snapshot.name != REVISION:
        raise RuntimeError(
            f"DFlash2 snapshot must resolve to pinned revision {REVISION}, got {snapshot}"
        )
    config_path = snapshot / "config.json"
    weight_path = snapshot / "model.safetensors"
    if weight_path.stat().st_size != WEIGHT_BYTES:
        raise RuntimeError(
            f"DFlash2 weight size changed: {weight_path.stat().st_size} != {WEIGHT_BYTES}"
        )
    quantize = _configure_quantizer(args.native_library)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    flush = torch.zeros(
        args.flush_mib * 1024 * 1024,
        dtype=torch.uint8,
        device=device,
    )

    tuning: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    selected_tiles: dict[str, dict[str, list[int]]] = {}
    projection_status: dict[str, dict[str, Any]] = {}
    for projection_name in selected_projections:
        projection = PROJECTIONS[projection_name]
        weight = _load_projection(weight_path, projection, 0, device)
        packed, scales = _quantize_weight(weight, quantize)
        winners: dict[int, tuple[int, int, int, int, int]] = {}
        for row_count in rows:
            candidates = [
                _case(
                    projection=projection,
                    layer=0,
                    rows=row_count,
                    config=candidate,
                    weight=weight,
                    packed=packed,
                    scales=scales,
                    flush=flush,
                    seed=args.seed,
                    warmup=args.warmup,
                    rounds=args.rounds,
                )
                for candidate in configs
            ]
            tuning.extend(candidates)
            winner = min(candidates, key=lambda item: item["w8a16_ms"]["median"])
            winners[row_count] = tuple(winner["config"])  # type: ignore[assignment]
            validation.append(winner)
        del weight, packed, scales
        torch.cuda.empty_cache()

        for layer in layers:
            if layer == 0:
                continue
            weight = _load_projection(weight_path, projection, layer, device)
            packed, scales = _quantize_weight(weight, quantize)
            for row_count in rows:
                validation.append(
                    _case(
                        projection=projection,
                        layer=layer,
                        rows=row_count,
                        config=winners[row_count],
                        weight=weight,
                        packed=packed,
                        scales=scales,
                        flush=flush,
                        seed=args.seed,
                        warmup=args.warmup,
                        rounds=args.rounds,
                    )
                )
            del weight, packed, scales
            torch.cuda.empty_cache()

        selected_tiles[projection_name] = {
            str(row_count): list(winners[row_count]) for row_count in rows
        }
        cases = [case for case in validation if case["projection"] == projection_name]
        projection_status[projection_name] = {
            "cases": len(cases),
            "numerical_sanity_all": all(case["numerical_sanity"] for case in cases),
            "performance_gate_all": all(
                case["performance_gate_passed"] for case in cases
            ),
            "minimum_speedup": min(case["w8a16_speedup"] for case in cases),
            "promotable_to_full_service_gate": all(
                case["numerical_sanity"] and case["performance_gate_passed"]
                for case in cases
            ),
        }

    promotable = sorted(
        name
        for name, status in projection_status.items()
        if status["promotable_to_full_service_gate"]
    )
    body = {
        "schema": "glmrt-dflash2-w8a16-body-tuning-v1",
        "status": "measured",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot": os.fspath(snapshot),
        "config_sha256": _hash_file(config_path),
        "weight_sha256": _hash_file(weight_path),
        "native_library": {
            "path": os.fspath(args.native_library.resolve(strict=True)),
            "sha256": _hash_file(args.native_library.resolve(strict=True)),
        },
        "script_sha256": _hash_file(Path(__file__).resolve()),
        "kernel_source_sha256": _hash_file(TOOLS / "tune_w8a16_triton_prefill.py"),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "layers": list(layers),
        "rows": list(rows),
        "seed": args.seed,
        "warmup": args.warmup,
        "rounds": args.rounds,
        "flush_mib": args.flush_mib,
        "minimum_w8_speedup": MIN_W8_SPEEDUP,
        "maximum_relative_l2": MAX_RELATIVE_L2,
        "minimum_cosine": MIN_COSINE,
        "bf16_projection_bytes_per_pass": BF16_PROJECTION_BYTES_PER_PASS,
        "w8a16_projection_bytes_per_pass": W8_PROJECTION_BYTES_PER_PASS,
        "projection_bytes_avoided_per_pass": (
            BF16_PROJECTION_BYTES_PER_PASS - W8_PROJECTION_BYTES_PER_PASS
        ),
        "single_residency_required": True,
        "full_service_acceptance_required": True,
        "selected_tiles": selected_tiles,
        "projection_status": projection_status,
        "promotable_projections": promotable,
        "tuning_cases": tuning,
        "validation_cases": validation,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    report = {**body, "report_sha256": hashlib.sha256(canonical).hexdigest()}
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
