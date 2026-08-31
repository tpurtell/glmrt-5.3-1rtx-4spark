#!/usr/bin/env python3
from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import os
from pathlib import Path

from _b12x_exl3_k3_profile import (
    EXL3_K3_AOT_REGIMES,
    exl3_k3_grid_x,
    exl3_k3_route_block_rows,
    exl3_k3_tile_config,
)
from _b12x_exl3_k4_profile import (
    EXL3_K4_AOT_REGIMES,
    exl3_k4_grid_x,
    exl3_k4_route_block_rows,
    exl3_k4_tile_config,
)


PREFILL_REGIMES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
DECODE_GRID_X = 32
TOP1_M1_GRID_X = 32
TOP1_MULTIROW_GRID_X = 48


def prepare_export(output_dir: Path):
    # A disk-cache hit is executable-only and has no IR for export_to_c().
    os.environ["B12X_COMPILE_DISK_CACHE"] = "0"
    os.environ["B12X_COMPILE_MEMORY_CACHE"] = "0"

    import cuda.bindings.driver as cuda
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.empty(1, dtype=torch.uint8, device=device)
    return cuda, torch, device


def export_kernels(output_dir: Path, target_sms: int) -> None:
    cuda, torch, device = prepare_export(output_dir)
    from b12x.moe._shared.kernels.w4a16.host import (
        max_packed_route_slots,
        select_route_block_size_m,
    )
    from b12x.moe._shared.kernels.w4a16.kernel import (
        W4A16FusedMoeKernel,
        _w4a16_fused_persistent_grid_x,
        compile_w4a16_fused_moe,
        compile_w4a16_topk_sum,
    )

    w4a16_launch = W4A16FusedMoeKernel.__call__
    w4a16_launch.__annotations__["stream"] = cuda.CUstream
    w4a16_wrapped = getattr(w4a16_launch, "__wrapped__", None)
    if w4a16_wrapped is not None:
        w4a16_wrapped.__annotations__["stream"] = cuda.CUstream

    properties = torch.cuda.get_device_properties(device)
    physical_sms = int(properties.multi_processor_count)
    if target_sms <= 0 or target_sms > physical_sms:
        raise ValueError(
            f"target_sms must be in 1..{physical_sms} on this export host, got {target_sms}"
        )
    sms = target_sms
    max_shared_mem = int(properties.shared_memory_per_block_optin)
    config_lines = ["#pragma once"]
    metadata_lines = [f"w4a16_target_sms={sms}"]

    # compile_w4a16_fused_moe() uses its `sms` argument for planning, but the
    # underlying W4A16 kernel independently rereads the export GPU and bakes
    # that physical SM count into cooperative-barrier offsets.  Present the
    # 48-SM GB10 target while exporting W4A16 so its lock ABI remains valid
    # even when this script is run on the 188-SM coordinator.
    physical_get_device_properties = torch.cuda.get_device_properties

    class TargetDeviceProperties:
        def __init__(self, base: object) -> None:
            self._base = base

        @property
        def multi_processor_count(self) -> int:
            return target_sms

        def __getattr__(self, name: str) -> object:
            return getattr(self._base, name)

    def target_get_device_properties(device_arg: object = None) -> object:
        return TargetDeviceProperties(physical_get_device_properties(device_arg))

    torch.cuda.get_device_properties = target_get_device_properties

    def export_w4a16(
        *,
        rows: int,
        top_k: int,
        label: str,
        direct_topk: bool | None = None,
        block_size: int | None = None,
        tc_decode_fused_sum: bool | None = None,
    ) -> None:
        block_size_overridden = block_size is not None
        if block_size is None:
            block_size = select_route_block_size_m(rows, top_k, 256)
        if direct_topk is None:
            direct_topk = rows <= (8 if top_k == 1 else 4)
        if tc_decode_fused_sum is None:
            tc_decode_fused_sum = direct_topk and top_k == 1
        if top_k == 8 and not direct_topk and not block_size_overridden:
            # This is part of the native route-metadata ABI, not just a kernel
            # tuning choice. Keep it aligned with route.rs and the benchmark.
            block_size = 32 if rows <= 2048 else 48
        packed_route_slots = max_packed_route_slots(rows * top_k, block_size, 256)
        max_m_blocks = (
            rows * top_k
            if direct_topk
            else (packed_route_slots + block_size - 1) // block_size
        )
        fused = compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=6144,
            intermediate_size=512,
            num_experts=256,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=max_m_blocks,
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="packed",
            scale_format="e4m3_k16",
            direct_topk_routes=direct_topk,
            # Keep route outputs separate. The native wrapper reduces them in
            # fixed route order, avoiding atomic top-k accumulation.
            tc_decode_fused_sum=tc_decode_fused_sum,
        )
        export_name = f"moe_tp4_w4a16_{label}"
        fused.compiled.export_to_c(
            str(output_dir),
            export_name,
            f"glmrt_b12x_{export_name}",
        )
        persistent_grid = _w4a16_fused_persistent_grid_x(
            fused=fused,
            m=rows,
            topk=top_k,
            intermediate_size=512,
            activation="silu",
            direct_topk_routes=direct_topk,
            sms=sms,
        )
        if label.endswith("decode_m1"):
            persistent_grid = DECODE_GRID_X
        elif label == "prefill_m2048_topk8":
            persistent_grid = 80
        elif label == "prefill_m2064_topk8":
            # The 96-block auto choice falls off a scheduling cliff on GB10.
            # A complete native-call sweep (including input dequantization and
            # top-k reduction) puts 92 blocks within ~1% of the M=2048 kernel
            # while retaining the dedicated 2049..2064 tail bucket.
            persistent_grid = 92
        elif top_k == 1:
            persistent_grid = (
                TOP1_M1_GRID_X if rows == 1 else TOP1_MULTIROW_GRID_X
            )
        macro = label.upper()
        config_lines.extend(
            [
                f"#define GLMRT_B12X_W4A16_{macro}_GRID_X {persistent_grid}",
                f"#define GLMRT_B12X_W4A16_{macro}_BLOCK_SIZE {block_size}",
                f"#define GLMRT_B12X_W4A16_{macro}_PACKED_ROUTE_SLOTS {packed_route_slots}",
                f"#define GLMRT_B12X_W4A16_{macro}_MAX_M_BLOCKS {max_m_blocks}",
            ]
        )
        metadata_lines.append(
            f"{label}=grid:{persistent_grid},block:{block_size},"
            f"route_slots:{packed_route_slots},max_m_blocks:{max_m_blocks},"
            f"layout:packed,direct_topk:{int(direct_topk)},"
            f"tc_decode_fused_sum:{int(tc_decode_fused_sum)}"
        )

    def export_exl3(*, rows: int, trellis_bits: int) -> None:
        """Export checkpoint-native EXL3 with exact full rotations.

        EXL3 keeps FP16 rotation scratch even though the layer input on the
        Spark wire is BF16.  The raw BF16 input, the two routed FP16 A
        scratch planes, and the projection rotation tables are distinct
        arguments in the generated ABI.  Keep route packing enabled here;
        direct-route candidates are performance-gated separately so the
        initial production path matches SparkInfer's planned Trellis path.
        """

        top_k = 8
        block_size = select_route_block_size_m(rows, top_k, 256)
        if trellis_bits == 3:
            expected_block_size = exl3_k3_route_block_rows(rows)
            tile_config = exl3_k3_tile_config(rows)
            persistent_grid = exl3_k3_grid_x(rows)
        elif trellis_bits == 4:
            expected_block_size = exl3_k4_route_block_rows(rows)
            tile_config = exl3_k4_tile_config(rows)
            persistent_grid = exl3_k4_grid_x(rows)
        else:
            raise ValueError(f"unsupported EXL3 trellis bitrate {trellis_bits}")
        if block_size != expected_block_size:
            raise ValueError(
                f"SparkInfer EXL3 K{trellis_bits} M={rows} route ABI changed: "
                f"profile={expected_block_size}, source={block_size}"
            )
        packed_route_slots = max_packed_route_slots(
            rows * top_k,
            block_size,
            256,
        )
        max_m_blocks = (packed_route_slots + block_size - 1) // block_size
        fused = compile_w4a16_fused_moe(
            size_m=rows,
            hidden_size=6144,
            intermediate_size=512,
            num_experts=256,
            top_k=top_k,
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=block_size,
            max_m_blocks=max_m_blocks,
            element_dtype="fp16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout="trellis_t256",
            scale_format="e4m3_k32",
            w13_layout="trellis_t256_proj",
            trellis_bits=trellis_bits,
            trellis_codebook="mcg",
            direct_topk_routes=False,
            tc_decode_fused_sum=False,
            force_tile_config=tile_config,
            intermediate_rotation=True,
            full_rotation=True,
            rotation_input_dtype="bf16",
        )
        label = f"exl3_k{trellis_bits}_m{rows}_topk8"
        export_name = f"moe_tp4_{label}"
        fused.compiled.export_to_c(
            str(output_dir),
            export_name,
            f"glmrt_b12x_{export_name}",
        )
        automatic_grid = _w4a16_fused_persistent_grid_x(
            fused=fused,
            m=rows,
            topk=top_k,
            intermediate_size=512,
            activation="silu",
            direct_topk_routes=False,
            sms=sms,
        )
        if persistent_grid <= 0 or persistent_grid > automatic_grid:
            raise ValueError(
                f"EXL3 K{trellis_bits} M={rows} grid {persistent_grid} is outside the safe "
                f"cooperative range 1..{automatic_grid} for its selected tiles"
            )
        macro = label.upper()
        config_lines.extend(
            [
                f"#define GLMRT_B12X_{macro}_GRID_X {persistent_grid}",
                f"#define GLMRT_B12X_{macro}_MAX_GRID_X {automatic_grid}",
                f"#define GLMRT_B12X_{macro}_BLOCK_SIZE {block_size}",
                f"#define GLMRT_B12X_{macro}_PACKED_ROUTE_SLOTS {packed_route_slots}",
                f"#define GLMRT_B12X_{macro}_MAX_M_BLOCKS {max_m_blocks}",
                f"#define GLMRT_B12X_{macro}_FC1_TILE_K {int(fused.fc1_tile_k)}",
                f"#define GLMRT_B12X_{macro}_FC1_TILE_N {int(fused.fc1_tile_n)}",
                f"#define GLMRT_B12X_{macro}_FC2_TILE_K {int(fused.fc2_tile_k)}",
                f"#define GLMRT_B12X_{macro}_FC2_TILE_N {int(fused.fc2_tile_n)}",
            ]
        )
        metadata_lines.append(
            f"{label}=grid:{persistent_grid},auto_grid:{automatic_grid},"
            f"block:{block_size},"
            f"route_slots:{packed_route_slots},max_m_blocks:{max_m_blocks},"
            f"tiles:{int(fused.fc1_tile_k)}x{int(fused.fc1_tile_n)}+"
            f"{int(fused.fc2_tile_k)}x{int(fused.fc2_tile_n)},"
            "layout:trellis_t256,w13_layout:trellis_t256_proj,"
            f"bits:{trellis_bits},codebook:mcg,full_rotation:1,rotation_input:bf16,"
            "direct_topk:0"
        )

    export_w4a16(rows=1, top_k=8, label="decode_m1")
    export_w4a16(
        rows=1,
        top_k=8,
        label="decode_m1_fused_sum",
        direct_topk=True,
        tc_decode_fused_sum=True,
    )
    for rows in PREFILL_REGIMES[1:]:
        export_w4a16(rows=rows, top_k=8, label=f"prefill_m{rows}_topk8")
    export_w4a16(rows=512, top_k=8, label="prefill_m512_topk8")
    export_w4a16(rows=1024, top_k=8, label="prefill_m1024_topk8")
    export_w4a16(rows=2048, top_k=8, label="prefill_m2048_topk8")
    export_w4a16(rows=2064, top_k=8, label="prefill_m2064_topk8")
    for rows in PREFILL_REGIMES:
        export_w4a16(rows=rows, top_k=1, label=f"top1_m{rows}")
    for rows in EXL3_K3_AOT_REGIMES:
        export_exl3(rows=rows, trellis_bits=3)
    for rows in EXL3_K4_AOT_REGIMES:
        export_exl3(rows=rows, trellis_bits=4)
    exl3_topk_sum = compile_w4a16_topk_sum(
        m=1,
        topk=8,
        hidden_size=6144,
        element_dtype="fp16",
        full_rotation=True,
        num_experts=256,
        route_num_experts=0,
        route_ids_dtype=torch.int32,
        use_expert_map=False,
        broadcast_svh=False,
    )
    exl3_topk_sum.compiled.export_to_c(
        str(output_dir),
        "moe_tp4_exl3_k3_topk_sum",
        "glmrt_b12x_moe_tp4_exl3_k3_topk_sum",
    )
    metadata_lines.append(
        "exl3_k3_topk_sum=topk:8,hidden:6144,element:fp16,"
        "output:fp32,full_rotation:1,route_ids:int32"
    )
    exl3_topk_sum_bf16 = compile_w4a16_topk_sum(
        m=1,
        topk=8,
        hidden_size=6144,
        element_dtype="fp16",
        full_rotation=True,
        full_rotation_output_dtype="bf16",
        num_experts=256,
        route_num_experts=0,
        route_ids_dtype=torch.int32,
        use_expert_map=False,
        broadcast_svh=False,
    )
    exl3_topk_sum_bf16.compiled.export_to_c(
        str(output_dir),
        "moe_tp4_exl3_k3_topk_sum_bf16",
        "glmrt_b12x_moe_tp4_exl3_k3_topk_sum_bf16",
    )
    metadata_lines.append(
        "exl3_k3_topk_sum_bf16=topk:8,hidden:6144,element:fp16,"
        "output:bf16,full_rotation:1,route_ids:int32"
    )

    (output_dir / "b12x_spark_moe_aot_config.h").write_text(
        "\n".join(config_lines) + "\n",
        encoding="ascii",
    )
    (output_dir / "b12x_spark_moe_aot.meta").write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="ascii",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the SparkInfer MoE AOT kernels."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-sms", type=int, default=48)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    export_kernels(output_dir, args.target_sms)


if __name__ == "__main__":
    main()
