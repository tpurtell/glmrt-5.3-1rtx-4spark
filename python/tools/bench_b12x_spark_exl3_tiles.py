#!/usr/bin/env python3
"""Tune SparkInfer K3/K4 EXL3 tiles at GLM-5's exact TP4 geometry."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

import _pinned_sparkinfer  # noqa: E402,F401
import torch  # noqa: E402

from _b12x_exl3_k3_profile import (  # noqa: E402
    EXL3_K3_AOT_REGIMES,
    K128_N128_FC1,
    K64_N128,
    K64_N256,
    exl3_k3_capacity_rows,
    exl3_k3_route_block_rows,
)
from _b12x_exl3_k4_profile import (  # noqa: E402
    EXL3_K4_AOT_REGIMES,
    EXL3_K4_REQUIRED_LIVE_ROWS,
    exl3_k4_capacity_rows,
    exl3_k4_route_block_rows,
)

from b12x.moe import fused_moe  # noqa: E402
from b12x.moe._shared.kernels.w4a16.host import (  # noqa: E402
    select_route_block_size_m,
)
from b12x.moe._shared.kernels.w4a16.kernel import _w4a16_num_regs  # noqa: E402
from validate_b12x_exl3_native import (  # noqa: E402
    _canonical_json,
    _load_artifact_weight_tensors,
    _load_checkpoint_weight_tensors,
    _load_route_profile_sample,
    _route_ids_from_counts,
    _validate_route_counts,
)


EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 512
TOP_K = 8
TILE_CANDIDATES = (
    K64_N256,
    K128_N128_FC1,
    (64, 256, 128, 128),
    (128, 128, 128, 128),
    K64_N128,
    (128, 64, 128, 64),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--trellis-bits",
        type=int,
        choices=(3, 4),
        default=3,
        help="checkpoint-native Trellis bitrate (default: retained GLM-5.2 K3)",
    )
    parser.add_argument(
        "--rows",
        default="1",
        help=(
            "comma-separated candidate M regimes, 'all-aot', or the K4-only "
            "'required-native' live-row qualification surface"
        ),
    )
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument(
        "--route-block-rows",
        type=int,
        choices=(8, 16, 32, 48, 64),
        help=(
            "override SparkInfer's automatic same-expert route block for an "
            "isolated scheduling sweep"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--route-profile",
        type=Path,
        help="accepted live GLM route profile; requires one exact --rows value",
    )
    parser.add_argument(
        "--route-profile-sample",
        type=int,
        help="zero-based fixture index from --route-profile",
    )
    weight_source_group = parser.add_mutually_exclusive_group()
    weight_source_group.add_argument(
        "--projection-checkpoint-dir",
        type=Path,
        help="authenticated projection store used to assemble one calibrated layer",
    )
    weight_source_group.add_argument(
        "--model-snapshot",
        type=Path,
        help=(
            "authenticated finalized EXL3 artifact used to assemble one "
            "calibrated layer directly from indexed safetensors"
        ),
    )
    parser.add_argument(
        "--layer-id",
        type=int,
        default=3,
        help="calibrated routed layer to load (default: 3)",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=0,
        help="TP4 intermediate rank to assemble (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="optional content-bound JSON result; refuses to overwrite",
    )
    return parser.parse_args()


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    *,
    iterations: int,
    rounds: int,
) -> list[float]:
    samples: list[float] = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return samples


def _capacity_rows(rows: int, trellis_bits: int) -> int:
    return (
        exl3_k3_capacity_rows(rows)
        if trellis_bits == 3
        else exl3_k4_capacity_rows(rows)
    )


def _profile_route_block_rows(capacity_rows: int, trellis_bits: int) -> int:
    return (
        exl3_k3_route_block_rows(capacity_rows)
        if trellis_bits == 3
        else exl3_k4_route_block_rows(capacity_rows)
    )


def _make_source_tensors(
    device: torch.device,
    *,
    trellis_bits: int,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    trellis_words = 16 * trellis_bits
    w13 = torch.randint(
        -32768,
        32767,
        (
            2,
            EXPERTS,
            HIDDEN // 16,
            INTERMEDIATE // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    w2 = torch.randint(
        -32768,
        32767,
        (
            EXPERTS,
            INTERMEDIATE // 16,
            HIDDEN // 16,
            trellis_words,
        ),
        dtype=torch.int16,
        device=device,
        generator=generator,
    )
    gate_suh = torch.ones(
        (EXPERTS, HIDDEN), dtype=torch.float16, device=device
    )
    up_suh = torch.ones(
        (EXPERTS, HIDDEN), dtype=torch.float16, device=device
    )
    intermediate_rotations = torch.ones(
        (EXPERTS, 3 * INTERMEDIATE), dtype=torch.float16, device=device
    )
    down_svh = torch.ones(
        (EXPERTS, HIDDEN), dtype=torch.float16, device=device
    )
    return w13, w2, gate_suh, up_suh, intermediate_rotations, down_svh


def _prepare_weights(
    source_tensors: tuple[torch.Tensor, ...],
    *,
    tile_config: tuple[int, int, int, int],
    trellis_bits: int,
) -> fused_moe.ExpertWeights:
    w13, w2, gate_suh, up_suh, intermediate_rotations, down_svh = source_tensors
    plan = fused_moe.plan_weights(
        quant_modes="w4a16",
        source_format="b12x_trellis",
        activation="silu",
        params_dtype=torch.bfloat16,
        num_experts=EXPERTS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        w13_layout="w13",
        trellis_bits=trellis_bits,
        trellis_codebook="mcg",
        trellis_tile_config=tile_config,
    )
    return fused_moe.prepare_weights(
        plan=plan,
        params_dtype=torch.bfloat16,
        w1_fp4=w13,
        w2_fp4=w2,
        gate_suh=gate_suh,
        up_suh=up_suh,
        intermediate_rotations=intermediate_rotations,
        down_svh=down_svh,
        trellis_mcg=0xCBAC1FED,
    )


def _benchmark_case(
    *,
    experts: fused_moe.ExpertWeights,
    rows: int,
    device: torch.device,
    iterations: int,
    rounds: int,
    warmup: int,
    reference: torch.Tensor | None,
    expert_route_counts: list[int] | None,
    route_block_rows: int | None,
    seed: int,
    trellis_bits: int,
) -> tuple[dict[str, object], torch.Tensor]:
    capacity_rows = _capacity_rows(rows, trellis_bits)
    block_size = (
        select_route_block_size_m(capacity_rows, TOP_K, EXPERTS)
        if route_block_rows is None
        else int(route_block_rows)
    )
    if route_block_rows is None and block_size != _profile_route_block_rows(
        capacity_rows, trellis_bits
    ):
        raise RuntimeError(
            f"SparkInfer route block for K{trellis_bits} capacity "
            f"M={capacity_rows} differs from the exported profile"
        )
    plan = fused_moe.plan(
        fused_moe.Caps(
            max_tokens=capacity_rows,
            num_topk=TOP_K,
            route_num_experts=EXPERTS,
            device=device,
            weight_plan=experts.plan,
            quant_mode="w4a16",
            w4a16_block_size_m=block_size,
        )
    )
    scratch_spec = plan.scratch_specs()[0]
    scratch = torch.empty(scratch_spec.shape, dtype=scratch_spec.dtype, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + rows)
    source = (
        torch.randn((rows, HIDDEN), device=device, generator=generator) * 0.002
    ).to(torch.bfloat16)
    if expert_route_counts is None:
        row_ids = torch.arange(rows, device=device, dtype=torch.int32).view(-1, 1)
        route_offsets = torch.arange(TOP_K, device=device, dtype=torch.int32).view(1, -1)
        topk_ids = (row_ids * 17 + route_offsets * 29) % EXPERTS
    else:
        topk_ids = torch.tensor(
            _route_ids_from_counts(expert_route_counts, rows, TOP_K),
            dtype=torch.int32,
            device=device,
        )
    topk_weights = torch.rand(
        (rows, TOP_K), dtype=torch.float32, device=device, generator=generator
    )
    topk_weights /= topk_weights.sum(dim=1, keepdim=True)
    binding = plan.bind(
        scratch=scratch,
        a=source,
        experts=experts,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
    )
    eager = binding.run().clone()
    torch.cuda.synchronize(device)
    if not torch.isfinite(eager).all() or float(eager.norm()) <= 1.0e-9:
        raise RuntimeError(f"non-finite or zero EXL3 output at rows={rows}")

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = binding.run()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    if not torch.equal(captured, eager):
        raise RuntimeError(f"CUDA graph differs from eager EXL3 output at rows={rows}")

    if reference is None:
        reference = eager.clone()
    difference = eager - reference
    relative_l2 = float(difference.norm() / reference.norm().clamp_min(1.0e-9))
    cosine = float(
        torch.nn.functional.cosine_similarity(
            eager.flatten(), reference.flatten(), dim=0
        )
    )
    samples = _time_graph(graph, iterations=iterations, rounds=rounds)
    launch_token_count = rows if capacity_rows <= 32 else capacity_rows
    launch = dict(plan._prewarmed_fused_launches)[launch_token_count]
    registers_per_thread = getattr(launch, "registers_per_thread", None)
    local_memory_bytes = getattr(launch, "local_memory_bytes", None)
    if registers_per_thread is None:
        # Current SparkInfer retains only launch-relevant resource metadata.
        # Reconstruct the same SM121 register estimate used by its occupancy
        # planner so reports remain comparable with earlier tuning sweeps.
        cta_m_blocks = (int(launch.moe_block_size) + 15) // 16
        registers_per_thread = max(
            _w4a16_num_regs(
                cta_threads=launch.cta_threads,
                cta_m_blocks=cta_m_blocks,
                cta_n_blocks=tile_n // 16,
                cta_k_blocks=tile_k // 16,
                uses_m_block_8=launch.moe_block_size == 8,
                weight_layout=launch.weight_layout,
            )
            for tile_n, tile_k in (
                (launch.fc1_tile_n, launch.fc1_tile_k),
                (launch.fc2_tile_n, launch.fc2_tile_k),
            )
        )
        local_memory_bytes = 0
    result: dict[str, object] = {
        "rows": rows,
        "capacity_rows": capacity_rows,
        "route_block_rows": block_size,
        "median_ms": statistics.median(samples),
        "minimum_ms": min(samples),
        "samples_ms": samples,
        "fc1_tile": [launch.fc1_tile_k, launch.fc1_tile_n],
        "fc2_tile": [launch.fc2_tile_k, launch.fc2_tile_n],
        "blocks_per_sm": launch.blocks_per_sm,
        "registers_per_thread": registers_per_thread,
        "local_memory_bytes": local_memory_bytes,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }
    del captured, graph, binding, scratch, plan, eager
    return result, reference


def main() -> None:
    args = _parse_args()
    row_spec = args.rows.strip().lower()
    if row_spec == "all-aot":
        rows = (
            EXL3_K3_AOT_REGIMES
            if args.trellis_bits == 3
            else EXL3_K4_AOT_REGIMES
        )
    elif row_spec == "required-native":
        if args.trellis_bits != 4:
            raise SystemExit("--rows required-native is available only with --trellis-bits 4")
        rows = EXL3_K4_REQUIRED_LIVE_ROWS
    else:
        try:
            rows = tuple(int(value) for value in args.rows.split(",") if value.strip())
        except ValueError as error:
            raise SystemExit(
                "--rows must be comma-separated integers, all-aot, or required-native"
            ) from error
    if not rows or any(value < 1 or value > 2064 for value in rows):
        raise SystemExit("--rows must contain exact M values in 1..2064")
    if args.iterations < 1 or args.rounds < 1 or args.warmup < 1:
        raise SystemExit("iterations, rounds, and warmup must be positive")
    if (args.route_profile is None) != (args.route_profile_sample is None):
        raise SystemExit("--route-profile and --route-profile-sample are required together")
    if args.route_profile is not None and len(rows) != 1:
        raise SystemExit("--route-profile requires exactly one --rows value")
    if args.route_profile_sample is not None and args.route_profile_sample < 0:
        raise SystemExit("--route-profile-sample must be non-negative")
    if not 3 <= args.layer_id <= 77:
        raise SystemExit("--layer-id must be in the routed range 3..77")
    if not 0 <= args.tp_rank < 4:
        raise SystemExit("--tp-rank must be in 0..3")
    if args.output is not None and args.output.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("--device must select CUDA")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    major, minor = torch.cuda.get_device_capability(device)
    if major != 12 or minor not in (0, 1):
        raise SystemExit(f"benchmark requires SM120/SM121, got SM{major}{minor}")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    expert_route_counts = None
    route_fixture = None
    if args.route_profile is not None:
        expert_route_counts, route_fixture = _load_route_profile_sample(
            args.route_profile,
            args.route_profile_sample,
            rows[0],
            args.trellis_bits,
        )
        expert_route_counts = _validate_route_counts(expert_route_counts, rows[0])

    if args.projection_checkpoint_dir is None and args.model_snapshot is None:
        source_tensors = _make_source_tensors(
            device,
            trellis_bits=args.trellis_bits,
            seed=args.seed,
        )
        weight_source: dict[str, object] = {
            "kind": "deterministic-synthetic",
            "seed": args.seed,
        }
    elif args.projection_checkpoint_dir is not None:
        checkpoint_root = args.projection_checkpoint_dir.expanduser()
        if checkpoint_root.is_symlink():
            raise SystemExit("--projection-checkpoint-dir must not be a symbolic link")
        checkpoint_root = checkpoint_root.resolve(strict=True)
        if not checkpoint_root.is_dir():
            raise SystemExit("--projection-checkpoint-dir must be a directory")
        source_tensors, checkpoint_identity = _load_checkpoint_weight_tensors(
            checkpoint_root,
            args.layer_id,
            args.tp_rank,
            device,
            args.trellis_bits,
        )
        weight_source = {
            "kind": "authenticated-calibrated-checkpoints",
            "root": str(checkpoint_root),
            "layer_id": args.layer_id,
            "tp_rank": args.tp_rank,
            **checkpoint_identity,
        }
    else:
        model_root = args.model_snapshot.expanduser().resolve(strict=True)
        if not model_root.is_dir():
            raise SystemExit("--model-snapshot must be a directory")
        source_tensors, artifact_identity = _load_artifact_weight_tensors(
            model_root,
            args.layer_id,
            args.tp_rank,
            device,
            args.trellis_bits,
        )
        weight_source = {
            "kind": "authenticated-finalized-exl3-artifact",
            "root": str(model_root),
            "layer_id": args.layer_id,
            "tp_rank": args.tp_rank,
            **artifact_identity,
        }
    references: dict[int, torch.Tensor] = {}
    results: list[dict[str, object]] = []
    for tile_config in TILE_CANDIDATES:
        experts = _prepare_weights(
            source_tensors,
            tile_config=tile_config,
            trellis_bits=args.trellis_bits,
        )
        for row_count in rows:
            try:
                result, reference = _benchmark_case(
                    experts=experts,
                    rows=row_count,
                    device=device,
                    iterations=args.iterations,
                    rounds=args.rounds,
                    warmup=args.warmup,
                    reference=references.get(row_count),
                    expert_route_counts=expert_route_counts,
                    route_block_rows=args.route_block_rows,
                    seed=args.seed,
                    trellis_bits=args.trellis_bits,
                )
            except ValueError as exc:
                if "force_tile_config" not in str(exc) or "does not fit" not in str(exc):
                    raise
                results.append(
                    {
                        "rows": row_count,
                        "tile_config": list(tile_config),
                        "skipped": str(exc),
                    }
                )
                continue
            references.setdefault(row_count, reference)
            result["tile_config"] = list(tile_config)
            results.append(result)
        del experts
        gc.collect()
        torch.cuda.empty_cache()

    winners = {
        str(row_count): min(
            (
                result
                for result in results
                if result["rows"] == row_count and "median_ms" in result
            ),
            key=lambda result: float(result["median_ms"]),
        )["tile_config"]
        for row_count in rows
    }
    body = {
        "schema": "glmrt-b12x-exl3-tile-sweep-v1",
        "status": "complete",
        "sparkinfer_revision": _pinned_sparkinfer.REVISION,
        "sparkinfer_version": _pinned_sparkinfer.VERSION,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(device),
        "geometry": {
            "experts": EXPERTS,
            "hidden": HIDDEN,
            "intermediate_tp4": INTERMEDIATE,
            "top_k": TOP_K,
            "trellis_bits": args.trellis_bits,
        },
        "seed": args.seed,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "route_block_rows_override": args.route_block_rows,
        "route_fixture": route_fixture,
        "weight_source": weight_source,
        "results": results,
        "winners": winners,
    }
    report = {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
