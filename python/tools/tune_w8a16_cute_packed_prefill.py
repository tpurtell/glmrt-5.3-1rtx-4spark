#!/usr/bin/env python3
"""Benchmark SparkInfer-style register-dequantized CuTe W8A16 prefill GEMM."""

from __future__ import annotations

import _pinned_sparkinfer  # noqa: F401

import argparse
import ctypes
import json
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.cute.runtime import from_dlpack

from b12x._lib.intrinsics import (
    bfloat2_mul,
    broadcast_f32_to_bfloat2,
    cp_async4_shared_global,
    cp_async4_shared_global_pred,
    get_ptr_as_int64,
    ld_shared_f32,
    ld_shared_i32_relaxed,
    ld_shared_v4_f32,
    ld_shared_v4_u32,
    st_shared_v4_f32,
    st_shared_v4_u32,
)
from b12x.moe._shared.kernels.w4a16.kernel import W4A16GemmKernel

from tune_w8a16_projection import (
    CATALOG_PATH,
    PROJECTION_TENSORS,
    bench,
    check_status,
    load_bf16_weight,
    metrics,
)


GROUP_SIZE = 256
PACKED_TILE_K = 16
PACKED_TILE_N = 64


@dsl_user_op
def dequant_s8x8_to_bf16_fragments(
    q_lo: Uint32,
    q_hi: Uint32,
    scale_0: Float32,
    scale_1: Float32,
    *,
    loc=None,
    ip=None,
):
    """Convert the eight signed bytes for one lane into four BF16x2 operands."""
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [
            Uint32(q_lo).ir_value(loc=loc, ip=ip),
            Uint32(q_hi).ir_value(loc=loc, ip=ip),
            Float32(scale_0).ir_value(loc=loc, ip=ip),
            Float32(scale_1).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .s32 i0, i1, i2, i3, i4, i5, i6, i7;
            .reg .f32 f0, f1, f2, f3, f4, f5, f6, f7;

            prmt.b32 i0, $4, 0, 0x8880;
            prmt.b32 i1, $4, 0, 0x9991;
            prmt.b32 i2, $4, 0, 0xaaa2;
            prmt.b32 i3, $4, 0, 0xbbb3;
            prmt.b32 i4, $5, 0, 0x8880;
            prmt.b32 i5, $5, 0, 0x9991;
            prmt.b32 i6, $5, 0, 0xaaa2;
            prmt.b32 i7, $5, 0, 0xbbb3;

            cvt.rn.f32.s32 f0, i0; cvt.rn.f32.s32 f1, i1;
            cvt.rn.f32.s32 f2, i2; cvt.rn.f32.s32 f3, i3;
            cvt.rn.f32.s32 f4, i4; cvt.rn.f32.s32 f5, i5;
            cvt.rn.f32.s32 f6, i6; cvt.rn.f32.s32 f7, i7;

            mul.f32 f0, f0, $6; mul.f32 f1, f1, $6;
            mul.f32 f4, f4, $6; mul.f32 f5, f5, $6;
            mul.f32 f2, f2, $7; mul.f32 f3, f3, $7;
            mul.f32 f6, f6, $7; mul.f32 f7, f7, $7;

            cvt.rn.satfinite.bf16x2.f32 $0, f4, f0;
            cvt.rn.satfinite.bf16x2.f32 $1, f5, f1;
            cvt.rn.satfinite.bf16x2.f32 $2, f6, f2;
            cvt.rn.satfinite.bf16x2.f32 $3, f7, f3;
        }
        """,
        "=r,=r,=r,=r,r,r,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


@dsl_user_op
def convert_s8x8_to_bf16_fragments(
    q_lo: Uint32,
    q_hi: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Convert eight signed bytes to four unscaled BF16x2 operands."""
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [
            Uint32(q_lo).ir_value(loc=loc, ip=ip),
            Uint32(q_hi).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .s32 i0, i1, i2, i3, i4, i5, i6, i7;
            .reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;
            prmt.b32 i0, $4, 0, 0x8880;
            prmt.b32 i1, $4, 0, 0x9991;
            prmt.b32 i2, $4, 0, 0xaaa2;
            prmt.b32 i3, $4, 0, 0xbbb3;
            prmt.b32 i4, $5, 0, 0x8880;
            prmt.b32 i5, $5, 0, 0x9991;
            prmt.b32 i6, $5, 0, 0xaaa2;
            prmt.b32 i7, $5, 0, 0xbbb3;
            cvt.rn.bf16.s32 b0, i0; cvt.rn.bf16.s32 b1, i1;
            cvt.rn.bf16.s32 b2, i2; cvt.rn.bf16.s32 b3, i3;
            cvt.rn.bf16.s32 b4, i4; cvt.rn.bf16.s32 b5, i5;
            cvt.rn.bf16.s32 b6, i6; cvt.rn.bf16.s32 b7, i7;
            mov.b32 $0, {b0, b4};
            mov.b32 $1, {b1, b5};
            mov.b32 $2, {b2, b6};
            mov.b32 $3, {b3, b7};
        }
        """,
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(4)
    )


class W8A16PackedGemmKernel(W4A16GemmKernel):
    """Dense W8 specialization reusing SparkInfer's persistent warp-MMA skeleton."""

    def __init__(
        self,
        *,
        size_m: int,
        size_n: int,
        size_k: int,
        block_m: int,
        tile_n: int,
        tile_k: int,
        stages: int,
        bf16_scale_mul: bool,
        post_scale_groups: bool = False,
    ):
        super().__init__(
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            num_experts=1,
            top_k=1,
            mul_topk_weights=False,
            tile_n=tile_n,
            tile_k=tile_k,
            moe_block_size=block_m,
            max_m_blocks=(size_m + block_m - 1) // block_m,
            element_dtype="bf16",
            scale_format="e4m3_k16",
        )
        if block_m not in (16, 32, 48, 64):
            raise ValueError("packed W8 prototype block_m must be 16, 32, 48, or 64")
        if (tile_n, tile_k) not in ((256, 64), (128, 64), (64, 128)):
            raise ValueError(
                "packed W8 prototype supports tiles 256x64, 128x64, and 64x128"
            )
        if stages not in (2, 3, 4):
            raise ValueError("packed W8 prototype supports two, three, or four stages")
        self.stages = stages
        self.bf16_scale_mul = bf16_scale_mul
        self.post_scale_groups = post_scale_groups
        if size_k % GROUP_SIZE != 0:
            raise ValueError("packed W8 prototype requires K divisible by 256")

        # Keep the logical W4 fragment geometry (two MMA bundles per warp),
        # but double physical B storage because each eight-value fragment is
        # two uint32 words instead of one.
        self.logical_b_sh_stage = self.b_sh_stage
        self.b_sh_stage *= 2
        self.b_copy_iters = self.b_sh_stage // self.cta_threads
        self.s_sh_stage = self.tile_n // 4  # one FP32 scale per output channel

        sh_red_size = (2 * self.cta_n_blocks + 1) * 16 * self.cta_m_blocks
        sh_b_size = self.stages * self.b_sh_stage
        sh_size_min = min(sh_red_size, sh_b_size)
        sh_size_max = max(sh_red_size, sh_b_size)
        sh_bias_size = self.cta_n_blocks * 16 // 8
        sh_b_red_bias_size = max(sh_size_max, sh_size_min + sh_bias_size)
        self.sh_b_off = self.sh_valid_count_off
        self.sh_red_off = self.sh_valid_count_off
        self.sh_s_off = self.sh_valid_count_off + sh_b_red_bias_size
        self.sh_a_off = self.sh_s_off + self.stages * self.s_sh_stage
        self.shared_int4 = self.sh_a_off + self.stages * self.a_sh_stage
        self.shared_words = self.shared_int4 * 4
        self.blocks_per_sm = 2 if self.shared_words * 4 <= 50_688 else 1

    @cute.jit
    def __call__(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        active_m: Int32,
        grid_x: Int32,
        stream: cuda.CUstream,
    ):
        # SparkInfer's W4A16 base kernel now accepts a second activation
        # pointer for rotated/full-route variants. This dense W8 path has one
        # activation allocation, so retain its stable AOT ABI and bind that
        # allocation to both internal inputs.
        self.kernel(
            a_bf16_flat,
            a_bf16_flat,
            b_i32_flat,
            c_bf16_flat,
            scales_f32_flat,
            global_scale,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            topk_weights_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            # The upstream W4 kernel ABI now carries a trellis execution LUT.
            # Packed W8 never dereferences it, so reuse an aligned resident
            # tensor without widening this stable exported AOT interface.
            scales_f32_flat,
            active_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=[self.cta_threads, 1, 1],
            min_blocks_per_mp=self.blocks_per_sm,
            stream=stream,
        )

    @cute.jit
    def _mma_accumulate_large_m(
        self,
        acc: cute.Tensor,
        a_regs: cute.Tensor,
        mb: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
        b_frag: cute.Tensor,
    ):
        # SparkInfer's W4A16 accumulator was split into scalar fragments to
        # control LLVM register promotion. The packed-W8 kernels deliberately
        # retain their original tensor accumulator, so keep their matching MMA
        # adapter local instead of inheriting W4A16's representation.
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[mb, jj, 0, 0],
            acc[mb, jj, 0, 1],
            acc[mb, jj, 0, 2],
            acc[mb, jj, 0, 3],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[0, 0],
            b_frag[0, 1],
        )
        acc[mb, jj, 0, 0] = d0
        acc[mb, jj, 0, 1] = d1
        acc[mb, jj, 0, 2] = d2
        acc[mb, jj, 0, 3] = d3
        d0, d1, d2, d3 = self._mma_m16n8k16_f32(
            acc[mb, jj, 1, 0],
            acc[mb, jj, 1, 1],
            acc[mb, jj, 1, 2],
            acc[mb, jj, 1, 3],
            a_regs[mb, 0],
            a_regs[mb, 1],
            a_regs[mb, 2],
            a_regs[mb, 3],
            b_frag[1, 0],
            b_frag[1, 1],
        )
        acc[mb, jj, 1, 0] = d0
        acc[mb, jj, 1, 1] = d1
        acc[mb, jj, 1, 2] = d2
        acc[mb, jj, 1, 3] = d3

    @cute.jit
    def _finish_tile(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        if cutlass.const_expr(uses_m_block_8):
            self._fold_cta_partials_m8(acc, smem_base, tid)
        else:
            self._fold_cta_partials_large_m(acc, smem_base, tid)

        if reduce_slice_count > Int32(1):
            self._wait_for_reduction_turn(
                locks_i32_flat, lock_slot, reduce_slice_idx, tid
            )
            self._combine_splitk_accumulators(
                acc,
                c_tmp_f32_flat,
                block_valid_rows,
                lock_slot,
                reduce_slice_idx,
                reduce_slice_count,
                tid,
                uses_m_block_8,
            )
            self._publish_reduction_turn(
                locks_i32_flat,
                lock_slot,
                reduce_slice_idx == reduce_slice_count - Int32(1),
                tid,
            )

        if reduce_slice_idx == reduce_slice_count - Int32(1):
            if cutlass.const_expr(uses_m_block_8):
                self._store_tile_m8(
                    acc,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )
            else:
                self._store_tile_large_m(
                    acc,
                    c_bf16_flat,
                    smem_base,
                    tid,
                    output_n_tile,
                    block_valid_rows,
                    global_scale_f32,
                )

    @cute.jit
    def _fold_cta_partials_large_m(
        self, acc: cute.Tensor, smem_base: Int32, tid: Int32
    ):
        red_off = self.cta_threads // self.b_sh_stride_threads // 2
        if cutlass.const_expr(red_off >= 1):
            red_idx, red_sh_stride, red_sh_delta, red_sh_rd = self._reduction_offsets(
                tid
            )

            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                if cutlass.const_expr(red_off == 2):
                    if Int32(2) <= red_idx and red_idx < Int32(4):
                        for flat_j in cutlass.range_constexpr(8):
                            jj = flat_j // 2
                            half = flat_j % 2
                            red_sh_wr = red_sh_delta * Int32(flat_j) + (
                                red_sh_rd - red_sh_stride * Int32(2)
                            )
                            st_shared_v4_f32(
                                self._int4_addr(
                                    smem_base, Int32(self.sh_red_off) + red_sh_wr
                                ),
                                acc[mb, jj, half, 0],
                                acc[mb, jj, half, 1],
                                acc[mb, jj, half, 2],
                                acc[mb, jj, half, 3],
                            )
                    cute.arch.sync_threads()

                if Int32(1) <= red_idx and red_idx < Int32(2):
                    for flat_j in cutlass.range_constexpr(8):
                        jj = flat_j // 2
                        half = flat_j % 2
                        red_sh_wr = red_sh_delta * Int32(flat_j) + (
                            red_sh_rd - red_sh_stride
                        )
                        if cutlass.const_expr(red_off > 1):
                            rd_addr = self._int4_addr(
                                smem_base,
                                Int32(self.sh_red_off)
                                + red_sh_delta * Int32(flat_j)
                                + red_sh_rd,
                            )
                            wr_addr = self._int4_addr(
                                smem_base,
                                Int32(self.sh_red_off) + red_sh_wr,
                            )
                            r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                            w0, w1, w2, w3 = ld_shared_v4_f32(wr_addr)
                            acc[mb, jj, half, 0] = acc[mb, jj, half, 0] + r0 + w0
                            acc[mb, jj, half, 1] = acc[mb, jj, half, 1] + r1 + w1
                            acc[mb, jj, half, 2] = acc[mb, jj, half, 2] + r2 + w2
                            acc[mb, jj, half, 3] = acc[mb, jj, half, 3] + r3 + w3
                        st_shared_v4_f32(
                            self._int4_addr(
                                smem_base, Int32(self.sh_red_off) + red_sh_wr
                            ),
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        )
                cute.arch.sync_threads()

                if red_idx == Int32(0):
                    for flat_j in cutlass.range_constexpr(8):
                        jj = flat_j // 2
                        half = flat_j % 2
                        rd_addr = self._int4_addr(
                            smem_base,
                            Int32(self.sh_red_off)
                            + red_sh_delta * Int32(flat_j)
                            + red_sh_rd,
                        )
                        r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                        acc[mb, jj, half, 0] = acc[mb, jj, half, 0] + r0
                        acc[mb, jj, half, 1] = acc[mb, jj, half, 1] + r1
                        acc[mb, jj, half, 2] = acc[mb, jj, half, 2] + r2
                        acc[mb, jj, half, 3] = acc[mb, jj, half, 3] + r3
                cute.arch.sync_threads()

    @cute.jit
    def _combine_splitk_accumulators(
        self,
        acc: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        block_valid_rows: Int32,
        lock_slot: Int32,
        reduce_slice_idx: Int32,
        reduce_slice_count: Int32,
        tid: Int32,
        uses_m_block_8: cutlass.Constexpr[bool],
    ):
        active_threads = Int32(32 * self.tb_n_warps)
        c_size_int4 = Int32((self.cta_m_blocks * 16 * self.cta_n_blocks * 16) // 4)
        c_cur_offset = lock_slot * c_size_int4
        if cutlass.const_expr(uses_m_block_8):
            if tid < active_threads:
                for jj in cutlass.range_constexpr(4):
                    k = jj * 2
                    acc[jj, 0], acc[jj, 1], acc[jj, 2], acc[jj, 3] = (
                        self._merge_splitk_slot(
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            Int32(k),
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                            acc[jj, 0],
                            acc[jj, 1],
                            acc[jj, 2],
                            acc[jj, 3],
                        )
                    )
        else:
            lane_row = (tid & Int32(31)) // Int32(4)
            if tid < active_threads:
                for k in cutlass.range_constexpr(self.cta_m_blocks * 8):
                    mb = k // 8
                    flat_j = k % 8
                    jj = flat_j // 2
                    half = flat_j % 2
                    row_valid = Int32(mb * 16) + lane_row < block_valid_rows
                    if row_valid:
                        (
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        ) = self._merge_splitk_slot(
                            c_tmp_f32_flat,
                            c_cur_offset,
                            active_threads,
                            Int32(k),
                            tid,
                            reduce_slice_idx,
                            reduce_slice_count,
                            acc[mb, jj, half, 0],
                            acc[mb, jj, half, 1],
                            acc[mb, jj, half, 2],
                            acc[mb, jj, half, 3],
                        )

    @cute.jit
    def _store_tile_large_m(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: cutlass.Float32,
    ):
        if cutlass.const_expr(self.has_n_tile_tail):
            (
                c_gl_stride,
                c_gl_stride_covered,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor_tail(tid, output_n_tile)
        else:
            (
                c_gl_stride,
                c_sh_stride,
                c_gl_wr_delta,
                c_sh_rd_delta,
                c_gl_wr,
                c_sh_rd,
            ) = self._output_store_cursor(tid, output_n_tile)
            c_gl_stride_covered = c_gl_stride
        c_sh_wr = (
            Int32(4) * c_sh_stride * ((tid & Int32(31)) // Int32(4))
            + (tid & Int32(31)) % Int32(4)
            + Int32(32) * (tid // Int32(32))
        )

        if tid // Int32(32) < Int32(self.tb_n_warps):
            write_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(not self.mul_topk_weights):
                write_scale = global_scale_f32
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                for jj in cutlass.range_constexpr(4):
                    wr = c_sh_wr + Int32(8 * jj)
                    self._write_bf16x2_shared(
                        smem_base,
                        wr,
                        acc[mb, jj, 0, 0],
                        acc[mb, jj, 0, 1],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8),
                        acc[mb, jj, 0, 2],
                        acc[mb, jj, 0, 3],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + Int32(4),
                        acc[mb, jj, 1, 0],
                        acc[mb, jj, 1, 1],
                        write_scale,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(4),
                        acc[mb, jj, 1, 2],
                        acc[mb, jj, 1, 3],
                        write_scale,
                    )
                c_sh_wr += Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        cute.arch.sync_threads()

        store_iters = (
            16 * self.cta_m_blocks
            + self.cta_threads // (2 * self.cta_n_blocks)
            - 1
        ) // (self.cta_threads // (2 * self.cta_n_blocks))
        if cutlass.const_expr(self.has_n_tile_tail):
            self._drain_output_smem_tail(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_stride_covered,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )
        else:
            self._drain_output_smem(
                c_bf16_flat,
                smem_base,
                c_gl_stride,
                c_gl_wr,
                c_gl_wr_delta,
                c_sh_rd,
                c_sh_rd_delta,
                block_valid_rows,
                store_iters,
            )

    @cute.jit
    def _prefetch_initial_tiles(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        for pipe in cutlass.range_constexpr(self.stages - 1):
            if Int32(pipe) < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    b_i32_flat,
                    scales_f32_flat,
                    smem_base,
                    tid,
                    Int32(pipe),
                    reduce_k_tile + Int32(pipe),
                    block_valid_rows,
                    a_gl_stride,
                    b_gl_stride,
                    s_gl_stride,
                    scales_expert_off,
                    b_gl_rd_base,
                    a_gl_rd_row,
                    a_gl_rd_col0,
                    a_sh_wr,
                    a_rows_per_iter,
                    output_n_tile,
                    expert_idx,
                )
            else:
                cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_lookahead_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        fetch_tile = tile_idx + Int32(self.stages - 1)
        if fetch_tile < k_tiles:
            self._stage_k_tile_async(
                a_bf16_flat,
                b_i32_flat,
                scales_f32_flat,
                smem_base,
                tid,
                Int32((pipe + self.stages - 1) % self.stages),
                reduce_k_tile + fetch_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
                expert_idx,
            )
        else:
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(self.stages - 2)
        cute.arch.sync_threads()

    @cute.jit
    def _prefetch_pipeline_step(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        # W8 has one activation view and its own staging implementation. Keep
        # this override so changes to W4A16's optional alternate-activation
        # pipeline do not leak into the stable packed-W8 kernel.
        if cutlass.const_expr(kk == self.b_sh_wr_iters - 2):
            self._prefetch_lookahead_tile(
                a_bf16_flat,
                b_i32_flat,
                scales_f32_flat,
                smem_base,
                tid,
                pipe,
                tile_idx,
                k_tiles,
                reduce_k_tile,
                block_valid_rows,
                a_gl_stride,
                b_gl_stride,
                s_gl_stride,
                scales_expert_off,
                b_gl_rd_base,
                a_gl_rd_row,
                a_gl_rd_col0,
                a_sh_wr,
                a_rows_per_iter,
                output_n_tile,
                expert_idx,
            )

    @cute.jit
    def _stage_k_tile_async(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        for iteration in cutlass.range_constexpr(self.a_sh_wr_iters):
            row = a_rows_per_iter * Int32(iteration) + a_gl_rd_row
            route_index = Int32(0)
            if row < Int32(self.moe_block_size):
                route_index = self._read_route_index(smem_base, row)
            a_int4 = (
                route_index * a_gl_stride
                + tile_idx * Int32(self.a_gl_rd_delta_o)
                + a_gl_rd_col0
            )
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(
                    Int32(iteration * self.a_sh_wr_delta) + a_sh_wr
                ),
            )
            cp_async4_shared_global_pred(
                a_dst,
                get_ptr_as_int64(a_bf16_flat, a_int4 * Int32(8)),
                (row < block_valid_rows).to(Int32),
            )

        vectors_per_k16 = Int32(self.tile_n)
        global_vectors_per_k16 = Int32(self.size_n)
        for iteration in cutlass.range_constexpr(self.b_copy_iters):
            local_vector = Int32(iteration * self.cta_threads) + tid
            local_k16 = local_vector // vectors_per_k16
            vector_in_k16 = local_vector - local_k16 * vectors_per_k16
            global_k16 = (
                tile_idx * Int32(self.tile_k // PACKED_TILE_K) + local_k16
            )
            global_vector = (
                global_k16 * global_vectors_per_k16
                + output_n_tile * Int32(self.tile_n)
                + vector_in_k16
            )
            b_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + local_vector,
            )
            cp_async4_shared_global(
                b_dst,
                get_ptr_as_int64(b_i32_flat, global_vector * Int32(4)),
            )

        if tid < Int32(self.s_sh_stage):
            group = tile_idx * Int32(self.tile_k) // Int32(GROUP_SIZE)
            scale_float = (
                Int64(expert_idx) * Int64((self.size_k // GROUP_SIZE) * self.size_n)
                + Int64(group) * Int64(self.size_n)
                + Int64(output_n_tile * self.tile_n)
                + Int64(tid * 4)
            )
            s_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_s_off) + pipe * Int32(self.s_sh_stage) + tid,
            )
            cp_async4_shared_global(
                s_dst,
                get_ptr_as_int64(scales_f32_flat, scale_float),
            )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _read_route_index(self, smem_base: Int32, row: Int32) -> Int32:
        # Keep the route load isolated so CuTe does not infer the packed weight
        # tensor's scalar type for this shared-memory access.
        return ld_shared_i32_relaxed(
            smem_base + Int32(self.sh_rd_route_off * 16) + row * Int32(4)
        )

    @cute.jit
    def _load_w8_bundle(
        self,
        q_regs: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
    ):
        # Multiplying the logical W4 fragment offset by two reaches the pair of
        # W8 words emitted for the same lane/warp fragment.
        physical_vector = Int32(2) * (
            Int32(self.b_sh_stride) * kk + b_sh_rd
        )
        b_addr = self._int4_addr(
            smem_base,
            Int32(self.sh_b_off)
            + pipe * Int32(self.b_sh_stage)
            + physical_vector,
        )
        q0, q1, q2, q3 = ld_shared_v4_u32(b_addr)
        q4, q5, q6, q7 = ld_shared_v4_u32(b_addr + Int32(16))
        q_regs[0, 0], q_regs[1, 0] = q0, q1
        q_regs[0, 1], q_regs[1, 1] = q2, q3
        q_regs[0, 2], q_regs[1, 2] = q4, q5
        q_regs[0, 3], q_regs[1, 3] = q6, q7

    @cute.jit
    def _load_w8_scales(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        warp_n = (tid // Int32(32)) % Int32(self.tb_n_warps)
        tc_col = lane // Int32(4)
        n64_base = warp_n * Int32(PACKED_TILE_N)
        scale_base = (
            smem_base
            + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
        )
        col0 = n64_base + Int32(jj * 16) + tc_col
        return (
            ld_shared_f32(scale_base + col0 * Int32(4)),
            ld_shared_f32(scale_base + (col0 + Int32(8)) * Int32(4)),
        )

    @cute.jit
    def _load_w8_output_scales(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
    ):
        lane = tid & Int32(31)
        warp_n = (tid // Int32(32)) % Int32(self.tb_n_warps)
        n64_base = warp_n * Int32(PACKED_TILE_N)
        scale_base = (
            smem_base
            + Int32(self.sh_s_off * 16)
            + pipe * Int32(self.s_sh_stage * 16)
        )
        col0 = n64_base + Int32(jj * 16) + (lane & Int32(3)) * Int32(2)
        return (
            ld_shared_f32(scale_base + col0 * Int32(4)),
            ld_shared_f32(scale_base + (col0 + Int32(1)) * Int32(4)),
            ld_shared_f32(scale_base + (col0 + Int32(8)) * Int32(4)),
            ld_shared_f32(scale_base + (col0 + Int32(9)) * Int32(4)),
        )

    @cute.jit
    def _clear_w8_bundle(self, q_regs: cute.Tensor):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                q_regs[row, col] = Uint32(0)

    @cute.jit
    def _copy_w8_bundle(
        self,
        q_dst: cute.Tensor,
        q_src: cute.Tensor,
    ):
        for row in cutlass.range_constexpr(2):
            for col in cutlass.range_constexpr(4):
                q_dst[row, col] = q_src[row, col]

    @cute.jit
    def _load_next_w8_bundle(
        self,
        q_next: cute.Tensor,
        a_next: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
    ):
        self._clear_w8_bundle(q_next)
        self._clear_a_register_bundle(a_next, False)
        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_w8_bundle(
                    q_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_a_register_bundle(
                    a_next, smem_base, a_sh_rd, Int32(pipe), Int32(kk + 1), False
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                next_pipe = Int32((pipe + 1) % self.stages)
                self._load_w8_bundle(
                    q_next,
                    smem_base,
                    tid,
                    b_sh_rd,
                    next_pipe,
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_next, smem_base, a_sh_rd, next_pipe, Int32(0), False
                )

    @cute.jit
    def _run_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        acc: cute.Tensor,
        q_cur: cute.Tensor,
        q_next: cute.Tensor,
        a_cur: cute.Tensor,
        a_next: cute.Tensor,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        if cutlass.const_expr(self.post_scale_groups):
            group_acc = cute.make_rmem_tensor(
                (self.cta_m_blocks, 4, 2, 4), Float32
            )
            group_acc.fill(0.0)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(self.stages):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_w8_bundle(
                            q_next,
                            a_next,
                            smem_base,
                            tid,
                            b_sh_rd,
                            a_sh_rd,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                        )
                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            b_i32_flat,
                            scales_f32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            reduce_k_tile,
                            block_valid_rows,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                            expert_idx,
                        )
                        for jj in cutlass.range_constexpr(4):
                            scale_0, scale_1 = self._load_w8_scales(
                                smem_base, tid, pipe, jj
                            )
                            if cutlass.const_expr(self.post_scale_groups):
                                b00, b01, b10, b11 = (
                                    convert_s8x8_to_bf16_fragments(
                                        q_cur[0, jj], q_cur[1, jj]
                                    )
                                )
                            elif cutlass.const_expr(self.bf16_scale_mul):
                                b00, b01, b10, b11 = (
                                    convert_s8x8_to_bf16_fragments(
                                        q_cur[0, jj], q_cur[1, jj]
                                    )
                                )
                                scale_bf16_0 = broadcast_f32_to_bfloat2(scale_0)
                                scale_bf16_1 = broadcast_f32_to_bfloat2(scale_1)
                                b00 = bfloat2_mul(b00, scale_bf16_0)
                                b01 = bfloat2_mul(b01, scale_bf16_0)
                                b10 = bfloat2_mul(b10, scale_bf16_1)
                                b11 = bfloat2_mul(b11, scale_bf16_1)
                            else:
                                b00, b01, b10, b11 = (
                                    dequant_s8x8_to_bf16_fragments(
                                        q_cur[0, jj],
                                        q_cur[1, jj],
                                        scale_0,
                                        scale_1,
                                    )
                                )
                            frag[0, 0], frag[0, 1] = b00, b01
                            frag[1, 0], frag[1, 1] = b10, b11
                            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                                if cutlass.const_expr(self.post_scale_groups):
                                    self._mma_accumulate_large_m(
                                        group_acc, a_cur, mb, jj, frag
                                    )
                                else:
                                    self._mma_accumulate_large_m(
                                        acc, a_cur, mb, jj, frag
                                    )
                            if cutlass.const_expr(
                                self.post_scale_groups
                                and kk + 1 == self.b_sh_wr_iters
                            ):
                                next_tile = reduce_k_tile + tile_idx + Int32(1)
                                group_tiles = Int32(GROUP_SIZE // self.tile_k)
                                if next_tile % group_tiles == Int32(
                                    0
                                ) or tile_idx + Int32(1) == k_tiles:
                                    out_scale_00, out_scale_01, out_scale_10, out_scale_11 = (
                                        self._load_w8_output_scales(
                                            smem_base, tid, pipe, jj
                                        )
                                    )
                                    for mb in cutlass.range_constexpr(
                                        self.cta_m_blocks
                                    ):
                                        for half in cutlass.range_constexpr(2):
                                            scale_even = (
                                                out_scale_00
                                                if half == 0
                                                else out_scale_10
                                            )
                                            scale_odd = (
                                                out_scale_01
                                                if half == 0
                                                else out_scale_11
                                            )
                                            for value in cutlass.range_constexpr(4):
                                                output_scale = (
                                                    scale_even
                                                    if value % 2 == 0
                                                    else scale_odd
                                                )
                                                acc[mb, jj, half, value] = (
                                                    acc[mb, jj, half, value]
                                                    + group_acc[mb, jj, half, value]
                                                    * output_scale
                                                )
                                                group_acc[mb, jj, half, value] = 0.0
                        self._copy_w8_bundle(q_cur, q_next)
                        self._copy_a_register_bundle(a_cur, a_next, False)
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_w8_bundle(
                    q_cur,
                    smem_base,
                    tid,
                    b_sh_rd,
                    Int32(0),
                    Int32(0),
                )
                self._load_a_register_bundle(
                    a_cur, smem_base, a_sh_rd, Int32(0), Int32(0), False
                )

    @cute.jit
    def _run_tile_large_m(
        self,
        a_bf16_flat: cute.Tensor,
        _a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        _trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        expert_idx: Int32,
        output_n_tile: Int32,
        reduce_k_tile: Int32,
        reduce_tile_count: Int32,
        reduce_slice_count: Int32,
        reduce_slice_idx: Int32,
        lock_slot: Int32,
        active_size_m: Int32,
        _dynamic_pair_override: cutlass.Constexpr[int],
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            b_sh_rd,
            _s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            expert_idx,
            output_n_tile,
            active_size_m,
        )
        a_sh_rd = self._a_shared_read_offset(tid, 16)
        acc = cute.make_rmem_tensor((self.cta_m_blocks, 4, 2, 4), Float32)
        acc.fill(0.0)
        k_tiles = reduce_tile_count
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_f32_flat,
            smem_base,
            tid,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
        )

        q_cur = cute.make_rmem_tensor((2, 4), Uint32)
        q_next = cute.make_rmem_tensor((2, 4), Uint32)
        self._load_w8_bundle(
            q_cur, smem_base, tid, b_sh_rd, Int32(0), Int32(0)
        )
        a_cur = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        a_next = cute.make_rmem_tensor((self.cta_m_blocks, 4), Uint32)
        self._load_a_register_bundle(
            a_cur, smem_base, a_sh_rd, Int32(0), Int32(0), False
        )
        self._run_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_f32_flat,
            smem_base,
            tid,
            acc,
            q_cur,
            q_next,
            a_cur,
            a_next,
            b_sh_rd,
            a_sh_rd,
            k_tiles,
            reduce_k_tile,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            expert_idx,
        )
        self._finish_tile(
            acc,
            c_bf16_flat,
            c_tmp_f32_flat,
            locks_i32_flat,
            smem_base,
            tid,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
            reduce_slice_count,
            reduce_slice_idx,
            lock_slot,
            False,
        )


class W8A16PackedTwoStageGemmKernel(W8A16PackedGemmKernel):
    """Register-buffered two-stage mainloop for the block-M=64 regime.

    Both K16 bundles of the current K64 tile are loaded into registers before
    its shared stage is released.  cp.async can then refill the alternate
    stage while MMA consumes those registers.  This keeps shared memory below
    the two-CTA/SM limit without serializing the global load ahead of MMA.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.stages != 2:
            raise ValueError("two-stage register pipeline requires --stages 2")
        if self.b_sh_wr_iters != 2:
            raise ValueError("two-stage register pipeline requires two K16 bundles")
        if self.post_scale_groups:
            raise ValueError("two-stage register pipeline does not support group postscale")

    @cute.jit
    def _accumulate_register_bundle(
        self,
        smem_base: Int32,
        tid: Int32,
        pipe: Int32,
        q_regs: cute.Tensor,
        a_regs: cute.Tensor,
        acc: cute.Tensor,
    ):
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        for jj in cutlass.range_constexpr(4):
            scale_0, scale_1 = self._load_w8_scales(
                smem_base, tid, pipe, jj
            )
            if cutlass.const_expr(self.bf16_scale_mul):
                b00, b01, b10, b11 = convert_s8x8_to_bf16_fragments(
                    q_regs[0, jj], q_regs[1, jj]
                )
                scale_bf16_0 = broadcast_f32_to_bfloat2(scale_0)
                scale_bf16_1 = broadcast_f32_to_bfloat2(scale_1)
                b00 = bfloat2_mul(b00, scale_bf16_0)
                b01 = bfloat2_mul(b01, scale_bf16_0)
                b10 = bfloat2_mul(b10, scale_bf16_1)
                b11 = bfloat2_mul(b11, scale_bf16_1)
            else:
                b00, b01, b10, b11 = dequant_s8x8_to_bf16_fragments(
                    q_regs[0, jj],
                    q_regs[1, jj],
                    scale_0,
                    scale_1,
                )
            frag[0, 0], frag[0, 1] = b00, b01
            frag[1, 0], frag[1, 1] = b10, b11
            for mb in cutlass.range_constexpr(self.cta_m_blocks):
                self._mma_accumulate_large_m(acc, a_regs, mb, jj, frag)

    @cute.jit
    def _run_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        acc: cute.Tensor,
        q_cur: cute.Tensor,
        q_next: cute.Tensor,
        a_cur: cute.Tensor,
        a_next: cute.Tensor,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        reduce_k_tile: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
        expert_idx: Int32,
    ):
        tile_idx = Int32(0)
        pipe = Int32(0)
        while tile_idx < k_tiles:
            next_tile = tile_idx + Int32(1)
            next_pipe = pipe ^ Int32(1)
            if next_tile < k_tiles:
                self._stage_k_tile_async(
                    a_bf16_flat,
                    b_i32_flat,
                    scales_f32_flat,
                    smem_base,
                    tid,
                    next_pipe,
                    reduce_k_tile + next_tile,
                    block_valid_rows,
                    a_gl_stride,
                    b_gl_stride,
                    s_gl_stride,
                    scales_expert_off,
                    b_gl_rd_base,
                    a_gl_rd_row,
                    a_gl_rd_col0,
                    a_sh_wr,
                    a_rows_per_iter,
                    output_n_tile,
                    expert_idx,
                )
            else:
                cute.arch.cp_async_commit_group()

            # The lookahead writes the alternate shared stage, so one A/B
            # register bundle can be reused for both K16 steps.  Avoiding a
            # second block-M64 A bundle is essential: it otherwise drives the
            # generated kernel to 255 registers and a local stack spill.
            for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                self._load_w8_bundle(
                    q_cur, smem_base, tid, b_sh_rd, pipe, Int32(kk)
                )
                self._load_a_register_bundle(
                    a_cur, smem_base, a_sh_rd, pipe, Int32(kk), False
                )
                self._accumulate_register_bundle(
                    smem_base, tid, pipe, q_cur, a_cur, acc
                )
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            tile_idx = next_tile
            pipe = next_pipe


class W8A16GroupedMWarpKernel(W8A16PackedGemmKernel):
    """Two 32-row MMA groups share one packed W8 tile in a 256-thread CTA."""

    LOCAL_THREADS = 128
    LOCAL_M_BLOCKS = 2
    GROUPS = 2
    TOTAL_BLOCK_M = 64

    def __init__(
        self,
        *,
        size_m: int,
        size_n: int,
        size_k: int,
        stages: int,
        bf16_scale_mul: bool,
        post_scale_groups: bool = False,
    ):
        super().__init__(
            size_m=size_m,
            size_n=size_n,
            size_k=size_k,
            block_m=self.TOTAL_BLOCK_M,
            tile_n=128,
            tile_k=64,
            stages=stages,
            bf16_scale_mul=bf16_scale_mul,
            post_scale_groups=post_scale_groups,
        )
        self.cta_threads = self.LOCAL_THREADS * self.GROUPS
        self.a_sh_wr_delta = self.a_sh_stride * (
            self.cta_threads // self.a_gl_rd_delta_o
        )
        self.a_sh_wr_iters = (
            self.a_sh_stage + self.a_sh_wr_delta - 1
        ) // self.a_sh_wr_delta
        self.b_copy_iters = self.b_sh_stage // self.cta_threads
        self.blocks_per_sm = 1

    @cute.jit
    def _run_persistent_gemm(
        self,
        a_bf16_flat: cute.Tensor,
        _a_alt_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        block_expert_ids: cute.Tensor,
        packed_route_count: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        c_tmp_f32_flat: cute.Tensor,
        locks_i32_flat: cute.Tensor,
        _trellis_lut_addr: Int64,
        smem_base: Int32,
        tid: Int32,
        cta: Int32,
        grid_x: Int32,
        active_size_m: Int32,
    ):
        n_tiles = Int32(self.n_tiles)
        m_blocks = (
            active_size_m + Int32(self.TOTAL_BLOCK_M - 1)
        ) // Int32(self.TOTAL_BLOCK_M)
        work_tile = cta
        total_tiles = m_blocks * n_tiles
        while work_tile < total_tiles:
            route_block_idx = work_tile // n_tiles
            output_n_tile = work_tile - route_block_idx * n_tiles
            self._run_grouped_tile(
                a_bf16_flat,
                b_i32_flat,
                c_bf16_flat,
                scales_f32_flat,
                global_scale,
                packed_route_indices,
                topk_weights_flat,
                smem_base,
                tid,
                route_block_idx,
                output_n_tile,
                active_size_m,
            )
            work_tile += grid_x

    @cute.jit
    def _group_a_shared_read_offset(self, local_tid: Int32) -> Int32:
        lane = local_tid & Int32(31)
        local_warp = local_tid // Int32(32)
        offset = Int32(self.a_sh_stride) * (lane % Int32(16)) + lane // Int32(16)
        offset += (
            Int32(2)
            * (local_warp // Int32(self.tb_n_warps))
            * Int32(self.b_sh_wr_iters)
        )
        return offset

    @cute.jit
    def _group_b_shared_read_offset(self, local_tid: Int32) -> Int32:
        if cutlass.const_expr(self.LOCAL_THREADS <= self.b_sh_stride):
            return local_tid
        return Int32(self.b_sh_stride) * (
            local_tid // Int32(self.b_sh_stride)
        ) + (local_tid % Int32(self.b_sh_stride)) + (
            local_tid // Int32(self.b_sh_stride)
        ) * Int32(self.b_sh_stride * (self.b_sh_wr_iters - 1))

    @cute.jit
    def _load_group_a_bundle(
        self,
        regs: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        pipe: Int32,
        kk: Int32,
        group: Int32,
    ):
        for local_mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
            global_mb = group * Int32(self.LOCAL_M_BLOCKS) + Int32(local_mb)
            a0, a1, a2, a3 = self._load_a_registers_large_m(
                smem_base, a_sh_rd, pipe, kk, global_mb
            )
            regs[local_mb, 0] = a0
            regs[local_mb, 1] = a1
            regs[local_mb, 2] = a2
            regs[local_mb, 3] = a3

    @cute.jit
    def _clear_group_a_bundle(self, regs: cute.Tensor):
        for mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
            for reg in cutlass.range_constexpr(4):
                regs[mb, reg] = Uint32(0)

    @cute.jit
    def _copy_group_a_bundle(self, dst: cute.Tensor, src: cute.Tensor):
        for mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
            for reg in cutlass.range_constexpr(4):
                dst[mb, reg] = src[mb, reg]

    @cute.jit
    def _load_next_group_bundle(
        self,
        q_next: cute.Tensor,
        a_next: cute.Tensor,
        smem_base: Int32,
        local_tid: Int32,
        group: Int32,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
    ):
        self._clear_w8_bundle(q_next)
        self._clear_group_a_bundle(a_next)
        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_w8_bundle(
                    q_next,
                    smem_base,
                    local_tid,
                    b_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                )
                self._load_group_a_bundle(
                    a_next,
                    smem_base,
                    a_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    group,
                )
        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                next_pipe = Int32((pipe + 1) % self.stages)
                self._load_w8_bundle(
                    q_next,
                    smem_base,
                    local_tid,
                    b_sh_rd,
                    next_pipe,
                    Int32(0),
                )
                self._load_group_a_bundle(
                    a_next,
                    smem_base,
                    a_sh_rd,
                    next_pipe,
                    Int32(0),
                    group,
                )

    @cute.jit
    def _dequant_fragment(
        self,
        frag: cute.Tensor,
        q_cur: cute.Tensor,
        smem_base: Int32,
        local_tid: Int32,
        pipe: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
    ):
        scale_0, scale_1 = self._load_w8_scales(
            smem_base, local_tid, pipe, jj
        )
        if cutlass.const_expr(self.bf16_scale_mul):
            b00, b01, b10, b11 = convert_s8x8_to_bf16_fragments(
                q_cur[0, jj], q_cur[1, jj]
            )
            scale_bf16_0 = broadcast_f32_to_bfloat2(scale_0)
            scale_bf16_1 = broadcast_f32_to_bfloat2(scale_1)
            b00 = bfloat2_mul(b00, scale_bf16_0)
            b01 = bfloat2_mul(b01, scale_bf16_0)
            b10 = bfloat2_mul(b10, scale_bf16_1)
            b11 = bfloat2_mul(b11, scale_bf16_1)
        else:
            b00, b01, b10, b11 = dequant_s8x8_to_bf16_fragments(
                q_cur[0, jj], q_cur[1, jj], scale_0, scale_1
            )
        frag[0, 0], frag[0, 1] = b00, b01
        frag[1, 0], frag[1, 1] = b10, b11

    @cute.jit
    def _run_grouped_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        local_tid: Int32,
        group: Int32,
        acc: cute.Tensor,
        q_cur: cute.Tensor,
        q_next: cute.Tensor,
        a_cur: cute.Tensor,
        a_next: cute.Tensor,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(self.stages):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_group_bundle(
                            q_next,
                            a_next,
                            smem_base,
                            local_tid,
                            group,
                            b_sh_rd,
                            a_sh_rd,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                        )
                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            b_i32_flat,
                            scales_f32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            Int32(0),
                            block_valid_rows,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                            Int32(0),
                        )
                        for jj in cutlass.range_constexpr(4):
                            self._dequant_fragment(
                                frag, q_cur, smem_base, local_tid, pipe, jj
                            )
                            for mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
                                self._mma_accumulate_large_m(
                                    acc, a_cur, mb, jj, frag
                                )
                        self._copy_w8_bundle(q_cur, q_next)
                        self._copy_group_a_bundle(a_cur, a_next)
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_w8_bundle(
                    q_cur,
                    smem_base,
                    local_tid,
                    b_sh_rd,
                    Int32(0),
                    Int32(0),
                )
                self._load_group_a_bundle(
                    a_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    group,
                )

    @cute.jit
    def _fold_group_partials(
        self,
        acc: cute.Tensor,
        smem_base: Int32,
        local_tid: Int32,
        group: Int32,
    ):
        red_idx = local_tid // Int32(self.b_sh_stride_threads)
        red_sh_stride = Int32(self.b_sh_stride_threads * 4 * 2)
        red_sh_delta = Int32(self.b_sh_stride_threads)
        red_sh_rd = red_sh_stride * red_idx + (
            local_tid % Int32(self.b_sh_stride_threads)
        )
        for global_mb in cutlass.range_constexpr(
            self.LOCAL_M_BLOCKS * self.GROUPS
        ):
            owns = group == Int32(global_mb // self.LOCAL_M_BLOCKS)
            local_mb = global_mb % self.LOCAL_M_BLOCKS
            if owns and red_idx == Int32(1):
                for flat_j in cutlass.range_constexpr(8):
                    jj = flat_j // 2
                    half = flat_j % 2
                    red_sh_wr = red_sh_delta * Int32(flat_j) + (
                        red_sh_rd - red_sh_stride
                    )
                    st_shared_v4_f32(
                        self._int4_addr(
                            smem_base, Int32(self.sh_red_off) + red_sh_wr
                        ),
                        acc[local_mb, jj, half, 0],
                        acc[local_mb, jj, half, 1],
                        acc[local_mb, jj, half, 2],
                        acc[local_mb, jj, half, 3],
                    )
            cute.arch.sync_threads()
            if owns and red_idx == Int32(0):
                for flat_j in cutlass.range_constexpr(8):
                    jj = flat_j // 2
                    half = flat_j % 2
                    rd_addr = self._int4_addr(
                        smem_base,
                        Int32(self.sh_red_off)
                        + red_sh_delta * Int32(flat_j)
                        + red_sh_rd,
                    )
                    r0, r1, r2, r3 = ld_shared_v4_f32(rd_addr)
                    acc[local_mb, jj, half, 0] += r0
                    acc[local_mb, jj, half, 1] += r1
                    acc[local_mb, jj, half, 2] += r2
                    acc[local_mb, jj, half, 3] += r3
            cute.arch.sync_threads()

    @cute.jit
    def _store_grouped_tile(
        self,
        acc: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        local_tid: Int32,
        group: Int32,
        output_n_tile: Int32,
        block_valid_rows: Int32,
        global_scale_f32: Float32,
    ):
        (
            c_gl_stride,
            c_sh_stride,
            c_gl_wr_delta,
            c_sh_rd_delta,
            c_gl_wr,
            c_sh_rd,
        ) = self._output_store_cursor(tid, output_n_tile)
        c_sh_wr = (
            Int32(4)
            * c_sh_stride
            * ((local_tid & Int32(31)) // Int32(4))
            + (local_tid & Int32(31)) % Int32(4)
            + Int32(32) * (local_tid // Int32(32))
        )
        mb_stride = Int32(16 * (4 * (2 * self.cta_n_blocks + 1)))
        if group < Int32(self.GROUPS) and local_tid // Int32(32) < Int32(
            self.tb_n_warps
        ):
            for local_mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
                global_mb = group * Int32(self.LOCAL_M_BLOCKS) + Int32(local_mb)
                mb_write = c_sh_wr + global_mb * mb_stride
                for jj in cutlass.range_constexpr(4):
                    wr = mb_write + Int32(8 * jj)
                    self._write_bf16x2_shared(
                        smem_base,
                        wr,
                        acc[local_mb, jj, 0, 0],
                        acc[local_mb, jj, 0, 1],
                        global_scale_f32,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8),
                        acc[local_mb, jj, 0, 2],
                        acc[local_mb, jj, 0, 3],
                        global_scale_f32,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + Int32(4),
                        acc[local_mb, jj, 1, 0],
                        acc[local_mb, jj, 1, 1],
                        global_scale_f32,
                    )
                    self._write_bf16x2_shared(
                        smem_base,
                        wr + (Int32(4) * c_sh_stride) * Int32(8) + Int32(4),
                        acc[local_mb, jj, 1, 2],
                        acc[local_mb, jj, 1, 3],
                        global_scale_f32,
                    )
        cute.arch.sync_threads()
        store_iters = (
            self.TOTAL_BLOCK_M
            + self.cta_threads // (2 * self.cta_n_blocks)
            - 1
        ) // (self.cta_threads // (2 * self.cta_n_blocks))
        self._drain_output_smem(
            c_bf16_flat,
            smem_base,
            c_gl_stride,
            c_gl_wr,
            c_gl_wr_delta,
            c_sh_rd,
            c_sh_rd_delta,
            block_valid_rows,
            store_iters,
        )

    @cute.jit
    def _run_grouped_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        active_size_m: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            _unused_b_sh_rd,
            _unused_s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            Int32(0),
            output_n_tile,
            active_size_m,
        )
        local_tid = tid % Int32(self.LOCAL_THREADS)
        group = tid // Int32(self.LOCAL_THREADS)
        a_sh_rd = self._group_a_shared_read_offset(local_tid)
        b_sh_rd = self._group_b_shared_read_offset(local_tid)
        k_tiles = Int32(self.k_tiles)
        self._prefetch_initial_tiles(
            a_bf16_flat,
            b_i32_flat,
            scales_f32_flat,
            smem_base,
            tid,
            k_tiles,
            Int32(0),
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
            Int32(0),
        )
        acc = cute.make_rmem_tensor((self.LOCAL_M_BLOCKS, 4, 2, 4), Float32)
        acc.fill(0.0)
        q_cur = cute.make_rmem_tensor((2, 4), Uint32)
        q_next = cute.make_rmem_tensor((2, 4), Uint32)
        a_cur = cute.make_rmem_tensor((self.LOCAL_M_BLOCKS, 4), Uint32)
        a_next = cute.make_rmem_tensor((self.LOCAL_M_BLOCKS, 4), Uint32)
        self._load_w8_bundle(
            q_cur, smem_base, local_tid, b_sh_rd, Int32(0), Int32(0)
        )
        self._load_group_a_bundle(
            a_cur, smem_base, a_sh_rd, Int32(0), Int32(0), group
        )
        self._run_grouped_mma_pipeline(
            a_bf16_flat,
            b_i32_flat,
            scales_f32_flat,
            smem_base,
            tid,
            local_tid,
            group,
            acc,
            q_cur,
            q_next,
            a_cur,
            a_next,
            b_sh_rd,
            a_sh_rd,
            k_tiles,
            block_valid_rows,
            a_gl_stride,
            b_gl_stride,
            s_gl_stride,
            scales_expert_off,
            b_gl_rd_base,
            a_gl_rd_row,
            a_gl_rd_col0,
            a_sh_wr,
            a_rows_per_iter,
            output_n_tile,
        )
        self._fold_group_partials(acc, smem_base, local_tid, group)
        self._store_grouped_tile(
            acc,
            c_bf16_flat,
            smem_base,
            tid,
            local_tid,
            group,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
        )


class W8A16GroupedMSharedFragmentKernel(W8A16GroupedMWarpKernel):
    """Dequantize B once, then share exact MMA fragments across M groups."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sh_frag_off = self.shared_int4
        self.frag_stage_int4 = (
            self.b_sh_wr_iters * self.LOCAL_THREADS * 4
        )
        self.shared_int4 += self.frag_stage_int4
        self.shared_words = self.shared_int4 * 4
        self.blocks_per_sm = 1

    @cute.jit
    def _load_next_group_a_only(
        self,
        a_next: cute.Tensor,
        smem_base: Int32,
        a_sh_rd: Int32,
        group: Int32,
        pipe: cutlass.Constexpr[int],
        kk: cutlass.Constexpr[int],
        tile_idx: Int32,
        k_tiles: Int32,
    ):
        self._clear_group_a_bundle(a_next)
        if cutlass.const_expr(kk + 1 < self.b_sh_wr_iters):
            if tile_idx < k_tiles:
                self._load_group_a_bundle(
                    a_next,
                    smem_base,
                    a_sh_rd,
                    Int32(pipe),
                    Int32(kk + 1),
                    group,
                )


        else:
            next_tile = tile_idx + Int32(1)
            if next_tile < k_tiles:
                self._load_group_a_bundle(
                    a_next,
                    smem_base,
                    a_sh_rd,
                    Int32((pipe + 1) % self.stages),
                    Int32(0),
                    group,
                )

    @cute.jit
    def _fragment_cache_addr(
        self,
        smem_base: Int32,
        local_tid: Int32,
        kk: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
    ) -> Int32:
        index = (
            Int32(kk * self.LOCAL_THREADS * 4)
            + local_tid * Int32(4)
            + Int32(jj)
        )
        return self._int4_addr(
            smem_base, Int32(self.sh_frag_off) + index
        )

    @cute.jit
    def _run_grouped_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        local_tid: Int32,
        group: Int32,
        acc: cute.Tensor,
        q_cur: cute.Tensor,
        q_next: cute.Tensor,
        a_cur: cute.Tensor,
        a_next: cute.Tensor,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(self.stages):
                if tile_idx < k_tiles:
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_group_a_only(
                            a_next,
                            smem_base,
                            a_sh_rd,
                            group,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                        )
                        self._prefetch_pipeline_step(
                            a_bf16_flat,
                            b_i32_flat,
                            scales_f32_flat,
                            smem_base,
                            tid,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                            Int32(0),
                            block_valid_rows,
                            a_gl_stride,
                            b_gl_stride,
                            s_gl_stride,
                            scales_expert_off,
                            b_gl_rd_base,
                            a_gl_rd_row,
                            a_gl_rd_col0,
                            a_sh_wr,
                            a_rows_per_iter,
                            output_n_tile,
                            Int32(0),
                        )
                        if group == Int32(0):
                            self._load_w8_bundle(
                                q_cur,
                                smem_base,
                                local_tid,
                                b_sh_rd,
                                Int32(pipe),
                                Int32(kk),
                            )
                            for jj in cutlass.range_constexpr(4):
                                self._dequant_fragment(
                                    frag,
                                    q_cur,
                                    smem_base,
                                    local_tid,
                                    pipe,
                                    jj,
                                )
                                st_shared_v4_u32(
                                    self._fragment_cache_addr(
                                        smem_base, local_tid, kk, jj
                                    ),
                                    frag[0, 0],
                                    frag[0, 1],
                                    frag[1, 0],
                                    frag[1, 1],
                                )
                        cute.arch.sync_threads()
                        for jj in cutlass.range_constexpr(4):
                            b00, b01, b10, b11 = ld_shared_v4_u32(
                                self._fragment_cache_addr(
                                    smem_base, local_tid, kk, jj
                                )
                            )
                            frag[0, 0], frag[0, 1] = b00, b01
                            frag[1, 0], frag[1, 1] = b10, b11
                            for mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
                                self._mma_accumulate_large_m(
                                    acc, a_cur, mb, jj, frag
                                )
                        cute.arch.sync_threads()
                        self._copy_group_a_bundle(a_cur, a_next)
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_group_a_bundle(
                    a_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    group,
                )


class W8A16GroupedMNamedTileFragmentKernel(
    W8A16GroupedMSharedFragmentKernel
):
    """Fan one dequantized B tile out to both M groups with one handoff.

    The first shared-fragment experiment synchronized the whole CTA before and
    after every 16-wide MMA step.  This variant converts the complete 64-wide
    K tile, performs one hardware named-barrier handoff, and relies on the
    existing cp.async stage boundary to keep the following overwrite behind
    both consumers.  It is intentionally an intermediate diagnostic: a final
    kernel should overlap conversion of the next tile with current-tile MMA.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fragment_ready_barrier = pipeline.NamedBarrier(
            barrier_id=11,
            num_threads=self.cta_threads,
        )

    @cute.jit
    def _run_grouped_mma_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        local_tid: Int32,
        group: Int32,
        acc: cute.Tensor,
        q_cur: cute.Tensor,
        q_next: cute.Tensor,
        a_cur: cute.Tensor,
        a_next: cute.Tensor,
        b_sh_rd: Int32,
        a_sh_rd: Int32,
        k_tiles: Int32,
        block_valid_rows: Int32,
        a_gl_stride: Int32,
        b_gl_stride: Int32,
        s_gl_stride: Int32,
        scales_expert_off: Int32,
        b_gl_rd_base: Int32,
        a_gl_rd_row: Int32,
        a_gl_rd_col0: Int32,
        a_sh_wr: Int32,
        a_rows_per_iter: Int32,
        output_n_tile: Int32,
    ):
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            for pipe in cutlass.range_constexpr(self.stages):
                if tile_idx < k_tiles:
                    # The inherited lookahead waits for the next cp.async stage
                    # and synchronizes the CTA.  That synchronization also
                    # proves both groups have stopped reading the fragment
                    # cache before group 0 overwrites it for this tile.
                    self._prefetch_pipeline_step(
                        a_bf16_flat,
                        b_i32_flat,
                        scales_f32_flat,
                        smem_base,
                        tid,
                        pipe,
                        0,
                        tile_idx,
                        k_tiles,
                        Int32(0),
                        block_valid_rows,
                        a_gl_stride,
                        b_gl_stride,
                        s_gl_stride,
                        scales_expert_off,
                        b_gl_rd_base,
                        a_gl_rd_row,
                        a_gl_rd_col0,
                        a_sh_wr,
                        a_rows_per_iter,
                        output_n_tile,
                        Int32(0),
                    )
                    if group == Int32(0):
                        for kk in cutlass.range_constexpr(
                            self.b_sh_wr_iters
                        ):
                            self._load_w8_bundle(
                                q_cur,
                                smem_base,
                                local_tid,
                                b_sh_rd,
                                Int32(pipe),
                                Int32(kk),
                            )
                            for jj in cutlass.range_constexpr(4):
                                self._dequant_fragment(
                                    frag,
                                    q_cur,
                                    smem_base,
                                    local_tid,
                                    pipe,
                                    jj,
                                )
                                st_shared_v4_u32(
                                    self._fragment_cache_addr(
                                        smem_base, local_tid, kk, jj
                                    ),
                                    frag[0, 0],
                                    frag[0, 1],
                                    frag[1, 0],
                                    frag[1, 1],
                                )
                    self.fragment_ready_barrier.arrive_and_wait()
                    for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                        self._load_next_group_a_only(
                            a_next,
                            smem_base,
                            a_sh_rd,
                            group,
                            pipe,
                            kk,
                            tile_idx,
                            k_tiles,
                        )
                        for jj in cutlass.range_constexpr(4):
                            b00, b01, b10, b11 = ld_shared_v4_u32(
                                self._fragment_cache_addr(
                                    smem_base, local_tid, kk, jj
                                )
                            )
                            frag[0, 0], frag[0, 1] = b00, b01
                            frag[1, 0], frag[1, 1] = b10, b11
                            for mb in cutlass.range_constexpr(
                                self.LOCAL_M_BLOCKS
                            ):
                                self._mma_accumulate_large_m(
                                    acc, a_cur, mb, jj, frag
                                )
                        self._copy_group_a_bundle(a_cur, a_next)
                    tile_idx += Int32(1)
            cute.arch.sync_threads()
            if tile_idx < k_tiles:
                self._load_group_a_bundle(
                    a_cur,
                    smem_base,
                    a_sh_rd,
                    Int32(0),
                    Int32(0),
                    group,
                )


class W8A16WarpSpecializedFragmentKernel(
    W8A16GroupedMSharedFragmentKernel
):
    """Two M=32 consumers share a double-buffered dequant producer.

    One 128-thread warp group owns packed-W8/scales/activation staging and
    fragment conversion.  Two 128-thread warp groups consume the resulting
    BF16 MMA fragments for disjoint M=32 slabs.  Ready/free named barriers let
    the producer build tile T+1 while both consumers execute tile T.
    """

    PRODUCER_GROUP = 2
    PRODUCER_THREADS = 128
    PIPE_STAGES = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cta_threads = self.LOCAL_THREADS * (self.GROUPS + 1)
        # The parent reserves one complete fragment tile.  Add the second
        # parity immediately after it and use only two of the inherited A/B
        # stages in this warp-specialized schedule.
        self.shared_int4 += self.frag_stage_int4
        self.shared_words = self.shared_int4 * 4
        self.blocks_per_sm = 1
        self.producer_stage_barrier = pipeline.NamedBarrier(
            barrier_id=11, num_threads=self.PRODUCER_THREADS
        )
        self.ready_barrier_0 = pipeline.NamedBarrier(
            barrier_id=12, num_threads=self.cta_threads
        )
        self.ready_barrier_1 = pipeline.NamedBarrier(
            barrier_id=13, num_threads=self.cta_threads
        )
        self.free_barrier_0 = pipeline.NamedBarrier(
            barrier_id=14, num_threads=self.cta_threads
        )
        self.free_barrier_1 = pipeline.NamedBarrier(
            barrier_id=15, num_threads=self.cta_threads
        )

    @cute.jit
    def _fragment_cache_parity_addr(
        self,
        smem_base: Int32,
        parity: Int32,
        local_tid: Int32,
        kk: cutlass.Constexpr[int],
        jj: cutlass.Constexpr[int],
    ) -> Int32:
        index = (
            parity * Int32(self.frag_stage_int4)
            + Int32(kk * self.LOCAL_THREADS * 4)
            + local_tid * Int32(4)
            + Int32(jj)
        )
        return self._int4_addr(smem_base, Int32(self.sh_frag_off) + index)

    @cute.jit
    def _producer_stage_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        local_tid: Int32,
        pipe: Int32,
        tile_idx: Int32,
        block_valid_rows: Int32,
        output_n_tile: Int32,
    ):
        a_gl_stride = Int32(self.size_k // 8)
        a_cols_int4 = Int32(self.tile_k // 8)
        producer_rows = Int32(self.PRODUCER_THREADS // (self.tile_k // 8))
        for iteration in cutlass.range_constexpr(
            self.TOTAL_BLOCK_M
            // (self.PRODUCER_THREADS // (self.tile_k // 8))
        ):
            row = producer_rows * Int32(iteration) + local_tid // a_cols_int4
            col = local_tid - (local_tid // a_cols_int4) * a_cols_int4
            route_index = self._read_route_index(smem_base, row)
            a_src = (
                route_index * a_gl_stride
                + tile_idx * a_cols_int4
                + col
            )
            a_local = Int32(iteration * self.PRODUCER_THREADS) + local_tid
            a_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_a_off)
                + pipe * Int32(self.a_sh_stage)
                + self._activation_smem_permuted_offset(a_local),
            )
            cp_async4_shared_global_pred(
                a_dst,
                get_ptr_as_int64(a_bf16_flat, a_src * Int32(8)),
                (row < block_valid_rows).to(Int32),
            )

        vectors_per_k16 = Int32(self.tile_n)
        global_vectors_per_k16 = Int32(self.size_n)
        for iteration in cutlass.range_constexpr(
            self.b_sh_stage // self.PRODUCER_THREADS
        ):
            local_vector = Int32(iteration * self.PRODUCER_THREADS) + local_tid
            local_k16 = local_vector // vectors_per_k16
            vector_in_k16 = local_vector - local_k16 * vectors_per_k16
            global_k16 = (
                tile_idx * Int32(self.tile_k // PACKED_TILE_K) + local_k16
            )
            global_vector = (
                global_k16 * global_vectors_per_k16
                + output_n_tile * Int32(self.tile_n)
                + vector_in_k16
            )
            b_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_b_off)
                + pipe * Int32(self.b_sh_stage)
                + local_vector,
            )
            cp_async4_shared_global(
                b_dst,
                get_ptr_as_int64(b_i32_flat, global_vector * Int32(4)),
            )

        if local_tid < Int32(self.s_sh_stage):
            scale_group = tile_idx * Int32(self.tile_k) // Int32(GROUP_SIZE)
            scale_float = (
                Int64(scale_group) * Int64(self.size_n)
                + Int64(output_n_tile * self.tile_n)
                + Int64(local_tid * 4)
            )
            s_dst = self._int4_addr(
                smem_base,
                Int32(self.sh_s_off)
                + pipe * Int32(self.s_sh_stage)
                + local_tid,
            )
            cp_async4_shared_global(
                s_dst,
                get_ptr_as_int64(scales_f32_flat, scale_float),
            )
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _producer_dequant_tile(
        self,
        smem_base: Int32,
        local_tid: Int32,
        pipe: Int32,
        parity: Int32,
        b_sh_rd: Int32,
    ):
        q_regs = cute.make_rmem_tensor((2, 4), Uint32)
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
            self._load_w8_bundle(
                q_regs,
                smem_base,
                local_tid,
                b_sh_rd,
                pipe,
                Int32(kk),
            )
            for jj in cutlass.range_constexpr(4):
                self._dequant_fragment(
                    frag, q_regs, smem_base, local_tid, pipe, jj
                )
                st_shared_v4_u32(
                    self._fragment_cache_parity_addr(
                        smem_base, parity, local_tid, kk, jj
                    ),
                    frag[0, 0],
                    frag[0, 1],
                    frag[1, 0],
                    frag[1, 1],
                )

    @cute.jit
    def _producer_pipeline(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        smem_base: Int32,
        local_tid: Int32,
        k_tiles: Int32,
        block_valid_rows: Int32,
        output_n_tile: Int32,
    ):
        b_sh_rd = self._group_b_shared_read_offset(local_tid)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            parity = tile_idx & Int32(1)
            if tile_idx >= Int32(self.PIPE_STAGES):
                if parity == Int32(0):
                    self.free_barrier_0.wait_unaligned()
                else:
                    self.free_barrier_1.wait_unaligned()
            self._producer_stage_tile(
                a_bf16_flat,
                b_i32_flat,
                scales_f32_flat,
                smem_base,
                local_tid,
                parity,
                tile_idx,
                block_valid_rows,
                output_n_tile,
            )
            cute.arch.cp_async_wait_group(0)
            # cp.async completion is thread-local.  Each producer thread's
            # MMA fragment reads vectors staged by neighboring producer
            # threads, so the warp group must rendezvous before dequant.
            self.producer_stage_barrier.wait_unaligned()
            self._producer_dequant_tile(
                smem_base, local_tid, parity, parity, b_sh_rd
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            if parity == Int32(0):
                self.ready_barrier_0.arrive_unaligned()
            else:
                self.ready_barrier_1.arrive_unaligned()
            tile_idx += Int32(1)
        # Complete the final two free-barrier generations before a persistent
        # CTA reuses the same barrier IDs for another output tile.
        if k_tiles >= Int32(1):
            final_parity = (k_tiles - Int32(1)) & Int32(1)
            if final_parity == Int32(0):
                self.free_barrier_0.wait_unaligned()
            else:
                self.free_barrier_1.wait_unaligned()
        if k_tiles >= Int32(2):
            previous_parity = (k_tiles - Int32(2)) & Int32(1)
            if previous_parity == Int32(0):
                self.free_barrier_0.wait_unaligned()
            else:
                self.free_barrier_1.wait_unaligned()

    @cute.jit
    def _consumer_pipeline(
        self,
        smem_base: Int32,
        local_tid: Int32,
        group: Int32,
        acc: cute.Tensor,
        k_tiles: Int32,
    ):
        frag = cute.make_rmem_tensor((2, 2), Uint32)
        a_regs = cute.make_rmem_tensor((self.LOCAL_M_BLOCKS, 4), Uint32)
        a_sh_rd = self._group_a_shared_read_offset(local_tid)
        tile_idx = Int32(0)
        while tile_idx < k_tiles:
            parity = tile_idx & Int32(1)
            if parity == Int32(0):
                self.ready_barrier_0.wait_unaligned()
            else:
                self.ready_barrier_1.wait_unaligned()
            for kk in cutlass.range_constexpr(self.b_sh_wr_iters):
                self._load_group_a_bundle(
                    a_regs,
                    smem_base,
                    a_sh_rd,
                    parity,
                    Int32(kk),
                    group,
                )
                for jj in cutlass.range_constexpr(4):
                    b00, b01, b10, b11 = ld_shared_v4_u32(
                        self._fragment_cache_parity_addr(
                            smem_base, parity, local_tid, kk, jj
                        )
                    )
                    frag[0, 0], frag[0, 1] = b00, b01
                    frag[1, 0], frag[1, 1] = b10, b11
                    for mb in cutlass.range_constexpr(self.LOCAL_M_BLOCKS):
                        self._mma_accumulate_large_m(
                            acc, a_regs, mb, jj, frag
                        )
            if parity == Int32(0):
                self.free_barrier_0.arrive_unaligned()
            else:
                self.free_barrier_1.arrive_unaligned()
            tile_idx += Int32(1)

    @cute.jit
    def _run_grouped_tile(
        self,
        a_bf16_flat: cute.Tensor,
        b_i32_flat: cute.Tensor,
        c_bf16_flat: cute.Tensor,
        scales_f32_flat: cute.Tensor,
        global_scale: cute.Tensor,
        packed_route_indices: cute.Tensor,
        topk_weights_flat: cute.Tensor,
        smem_base: Int32,
        tid: Int32,
        route_block_idx: Int32,
        output_n_tile: Int32,
        active_size_m: Int32,
    ):
        (
            global_scale_f32,
            block_valid_rows,
            _a_gl_stride,
            _b_gl_stride,
            _s_gl_stride,
            _scales_expert_off,
            _b_gl_rd_base,
            _a_gl_rd_row,
            _a_gl_rd_col0,
            _a_sh_wr,
            _a_rows_per_iter,
            _b_sh_rd,
            _s_sh_rd,
        ) = self._tile_common_prologue(
            global_scale,
            packed_route_indices,
            topk_weights_flat,
            smem_base,
            tid,
            route_block_idx,
            Int32(0),
            output_n_tile,
            active_size_m,
        )
        local_tid = tid % Int32(self.LOCAL_THREADS)
        group = tid // Int32(self.LOCAL_THREADS)
        k_tiles = Int32(self.k_tiles)
        acc = cute.make_rmem_tensor(
            (self.LOCAL_M_BLOCKS, 4, 2, 4), Float32
        )
        acc.fill(0.0)
        if group == Int32(self.PRODUCER_GROUP):
            self._producer_pipeline(
                a_bf16_flat,
                b_i32_flat,
                scales_f32_flat,
                smem_base,
                local_tid,
                k_tiles,
                block_valid_rows,
                output_n_tile,
            )
        else:
            self._consumer_pipeline(
                smem_base, local_tid, group, acc, k_tiles
            )
        self._fold_group_partials(acc, smem_base, local_tid, group)
        self._store_grouped_tile(
            acc,
            c_bf16_flat,
            smem_base,
            tid,
            local_tid,
            group,
            output_n_tile,
            block_valid_rows,
            global_scale_f32,
        )


def repack_w8_for_mma(weight_k: torch.Tensor) -> torch.Tensor:
    """Reorder K-major signed W8 into SparkInfer's lane/fragment tile order."""
    size_k, size_n = weight_k.shape
    if size_k % PACKED_TILE_K or size_n % PACKED_TILE_N:
        raise ValueError("packed W8 requires K/N multiples of 16/64")
    k_tiles = size_k // PACKED_TILE_K
    n_tiles = size_n // PACKED_TILE_N
    tiles = weight_k.view(k_tiles, PACKED_TILE_K, n_tiles, PACKED_TILE_N).permute(
        0, 2, 1, 3
    )
    device = weight_k.device
    out_pos = torch.arange(128, device=device, dtype=torch.long)
    lane = out_pos // 4
    warp_n = out_pos % 4
    tc_col = lane // 4
    tc_row = (lane % 4) * 2
    offsets = torch.tensor([0, 8, 0, 8, 1, 9, 1, 9], device=device)
    columns = torch.stack(
        (
            warp_n * 16 + tc_col,
            warp_n * 16 + tc_col,
            warp_n * 16 + tc_col + 8,
            warp_n * 16 + tc_col + 8,
            warp_n * 16 + tc_col,
            warp_n * 16 + tc_col,
            warp_n * 16 + tc_col + 8,
            warp_n * 16 + tc_col + 8,
        ),
        dim=1,
    )
    elements = tc_row[:, None] + offsets[None, :]
    gathered = tiles[:, :, elements, columns]
    packed = torch.empty(
        (k_tiles, n_tiles, 128, 2), device=device, dtype=torch.int32
    )
    for half in range(2):
        word = torch.zeros(
            (k_tiles, n_tiles, 128), device=device, dtype=torch.int32
        )
        for byte in range(4):
            value = gathered[..., half * 4 + byte].to(torch.int32) & 0xFF
            word |= value << (byte * 8)
        packed[..., half] = word
    return packed.contiguous().view(torch.int32)


def configure_native(path: Path):
    native = ctypes.CDLL(str(path.resolve()))
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
    quantize_packed = native.glmrt_cuda_quantize_bf16_w8a16_group256_packed_async
    quantize_packed.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    quantize_packed.restype = ctypes.c_int
    simt = native.glmrt_cuda_linear_w8a16_group256_m1_simt_async
    simt.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    simt.restype = ctypes.c_int
    packed_m1 = native.glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async
    packed_m1.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    packed_m1.restype = ctypes.c_int
    w8a8 = native.glmrt_cuda_linear_w8a8_group256_wmma_async
    w8a8.argtypes = (
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
    )
    w8a8.restype = ctypes.c_int
    return quantize, quantize_packed, simt, packed_m1, w8a8


def as_cute(tensor: torch.Tensor):
    return from_dlpack(tensor, assumed_align=16)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--native-library",
        type=Path,
        default=Path("native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"),
    )
    parser.add_argument(
        "--tensor", choices=tuple(PROJECTION_TENSORS), default="o"
    )
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--tile-n", type=int, default=128)
    parser.add_argument("--tile-k", type=int, default=64)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--bf16-scale-mul", action="store_true")
    parser.add_argument(
        "--post-scale-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply group scales to FP32 partial sums (default: enabled)",
    )
    parser.add_argument("--grouped-m-warps", action="store_true")
    parser.add_argument("--shared-fragment-cache", action="store_true")
    parser.add_argument("--named-tile-fragment-cache", action="store_true")
    parser.add_argument("--warp-specialized-fragments", action="store_true")
    parser.add_argument("--two-stage-register-pipeline", action="store_true")
    parser.add_argument("--grid", type=int, default=188)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument(
        "--weight-copies",
        type=int,
        default=1,
        help="rotate this many packed, SIMT, and BF16 weights to model cold layers",
    )
    parser.add_argument("--compare-simt", action="store_true")
    parser.add_argument("--compare-packed-m1", action="store_true")
    parser.add_argument("--compare-w8a8-wmma", action="store_true")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.weight_copies < 1:
        raise ValueError("--weight-copies must be positive")
    if args.post_scale_groups and args.bf16_scale_mul:
        raise ValueError("--post-scale-groups and --bf16-scale-mul are mutually exclusive")

    with CATALOG_PATH.open() as handle:
        catalog = json.load(handle)
    name = PROJECTION_TENSORS[args.tensor]
    weight = load_bf16_weight(catalog, name)
    size_n, size_k = weight.shape
    quantize, quantize_packed, simt, packed_m1, w8a8 = configure_native(
        args.native_library
    )
    weight_k = torch.empty((size_k, size_n), device="cuda", dtype=torch.int8)
    scales = torch.empty(
        (size_k // GROUP_SIZE, size_n), device="cuda", dtype=torch.float32
    )
    stream_ptr = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    check_status(
        quantize(
            weight.data_ptr(),
            weight_k.data_ptr(),
            scales.data_ptr(),
            size_k,
            size_n,
            1,
            stream_ptr,
        ),
        "K-major W8 quantization",
    )
    torch.cuda.synchronize()
    packed_reference = repack_w8_for_mma(weight_k)
    packed_bytes = torch.empty(size_k * size_n, device="cuda", dtype=torch.int8)
    packed_scales = torch.empty_like(scales)
    check_status(
        quantize_packed(
            weight.data_ptr(),
            packed_bytes.data_ptr(),
            packed_scales.data_ptr(),
            size_k,
            size_n,
            stream_ptr,
        ),
        "direct packed W8 quantization",
    )
    torch.cuda.synchronize()
    reference_bytes = packed_reference.view(torch.int8).reshape(-1)
    if not torch.equal(packed_bytes, reference_bytes):
        mismatch = torch.nonzero(packed_bytes != reference_bytes, as_tuple=False).flatten()
        first = int(mismatch[0])
        raise RuntimeError(
            "direct packed W8 quantizer does not match reference repack: "
            f"mismatches={mismatch.numel()} first={first} "
            f"native={int(packed_bytes[first])} reference={int(reference_bytes[first])}"
        )
    if not torch.equal(packed_scales, scales):
        raise RuntimeError("direct packed W8 quantizer scales do not match K-major scales")
    packed = packed_bytes.view(torch.int32)
    m1_packed = packed if args.compare_packed_m1 else None
    del weight_k, packed_reference, packed_scales
    packed_weights = [packed]
    packed_weights.extend(packed.clone() for _ in range(args.weight_copies - 1))
    m1_packed_weights = None
    m1_packed_output = None
    if m1_packed is not None:
        if args.rows != 1:
            raise ValueError("--compare-packed-m1 requires --rows 1")
        m1_packed_weights = [m1_packed]
        m1_packed_weights.extend(
            m1_packed.clone() for _ in range(args.weight_copies - 1)
        )
        m1_packed_output = torch.empty(
            (1, size_n), device="cuda", dtype=torch.bfloat16
        )
    bf16_weights = [weight]
    bf16_weights.extend(weight.clone() for _ in range(args.weight_copies - 1))
    simt_weight = None
    simt_weights = None
    simt_scales = None
    simt_output = None
    if args.compare_simt or args.compare_w8a8_wmma:
        simt_weight = torch.empty((size_n, size_k), device="cuda", dtype=torch.int8)
        simt_scales = torch.empty(
            (size_n, size_k // GROUP_SIZE), device="cuda", dtype=torch.float32
        )
        simt_output = torch.empty(
            (args.rows, size_n), device="cuda", dtype=torch.bfloat16
        )
        check_status(
            quantize(
                weight.data_ptr(),
                simt_weight.data_ptr(),
                simt_scales.data_ptr(),
                size_k,
                size_n,
                0,
                stream_ptr,
            ),
            "row-major W8 quantization",
        )
        simt_weights = [simt_weight]
        simt_weights.extend(
            simt_weight.clone() for _ in range(args.weight_copies - 1)
        )

    def launch_simt_projection(simt_weight_tensor: torch.Tensor) -> None:
        activation_ptr = activations[0].data_ptr()
        output_ptr = simt_output.data_ptr()
        for row in range(args.rows):
            check_status(
                simt(
                    activation_ptr + row * size_k * torch.bfloat16.itemsize,
                    simt_weight_tensor.data_ptr(),
                    simt_scales.data_ptr(),
                    output_ptr + row * size_n * torch.bfloat16.itemsize,
                    size_k,
                    size_n,
                    3,
                    stream_ptr,
                ),
                "row-major W8 SIMT projection",
            )

    def launch_m1_packed_projection(
        m1_packed_weight: torch.Tensor,
    ) -> None:
        check_status(
            packed_m1(
                activations[0].data_ptr(),
                m1_packed_weight.data_ptr(),
                scales.data_ptr(),
                m1_packed_output.data_ptr(),
                size_k,
                size_n,
                stream_ptr,
            ),
            "shared-layout packed W8 M=1 projection",
        )

    generator = torch.Generator(device="cuda")
    generator.manual_seed(20260721 + args.rows)
    activations = [
        torch.randn(
            (args.rows, size_k),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        for _ in range(args.lanes)
    ]
    activation_w8 = None
    activation_scales = None
    w8a8_output = None
    if args.compare_w8a8_wmma:
        activation_w8 = torch.empty(
            (args.rows, size_k), device="cuda", dtype=torch.int8
        )
        activation_scales = torch.empty(
            (args.rows, size_k // GROUP_SIZE),
            device="cuda",
            dtype=torch.float32,
        )
        w8a8_output = torch.empty(
            (args.rows, size_n), device="cuda", dtype=torch.bfloat16
        )

    def quantize_activations() -> None:
        check_status(
            quantize(
                activations[0].data_ptr(),
                activation_w8.data_ptr(),
                activation_scales.data_ptr(),
                size_k,
                args.rows,
                0,
                stream_ptr,
            ),
            "row-major activation W8 quantization",
        )

    def launch_w8a8_projection(weight_tensor: torch.Tensor) -> None:
        check_status(
            w8a8(
                activation_w8.data_ptr(),
                activation_scales.data_ptr(),
                weight_tensor.data_ptr(),
                simt_scales.data_ptr(),
                w8a8_output.data_ptr(),
                args.rows,
                size_k,
                size_n,
                stream_ptr,
            ),
            "W8A8 WMMA projection",
        )
    outputs = [
        torch.empty((args.rows, size_n), device="cuda", dtype=torch.bfloat16)
        for _ in range(args.lanes)
    ]
    route_slots = ((args.rows + args.block_m - 1) // args.block_m) * args.block_m
    routes = torch.arange(route_slots, device="cuda", dtype=torch.int32)
    block_experts = torch.zeros(
        route_slots // args.block_m, device="cuda", dtype=torch.int32
    )
    route_count = torch.tensor([route_slots], device="cuda", dtype=torch.int32)
    topk = torch.ones(route_slots, device="cuda", dtype=torch.float32)
    global_scale = torch.ones(1, device="cuda", dtype=torch.float32)
    scratch_elements = max(
        size_n * route_slots,
        4 * 256 * args.block_m * 256,
    )
    scratches = [
        torch.empty(scratch_elements, device="cuda", dtype=torch.float32)
        for _ in range(args.lanes)
    ]
    locks_per_lane = [
        torch.zeros(4 * 256, device="cuda", dtype=torch.int32)
        for _ in range(args.lanes)
    ]
    lane_streams = [torch.cuda.Stream() for _ in range(args.lanes)]

    def make_cute_args(
        lane: int,
        stream: torch.cuda.Stream,
        packed_weight: torch.Tensor = packed,
    ):
        return (
            as_cute(activations[lane].flatten()),
            as_cute(packed_weight.flatten()),
            as_cute(outputs[lane].flatten()),
            as_cute(scales.flatten()),
            as_cute(global_scale),
            as_cute(routes),
            as_cute(block_experts),
            as_cute(route_count),
            as_cute(topk),
            as_cute(scratches[lane]),
            as_cute(locks_per_lane[lane]),
            Int32(args.rows),
            Int32(args.grid),
            cuda.CUstream(stream.cuda_stream),
        )

    cute_args = make_cute_args(0, torch.cuda.current_stream())
    if args.grouped_m_warps:
        if args.block_m != 64 or (args.tile_n, args.tile_k) != (128, 64):
            raise ValueError(
                "--grouped-m-warps requires --block-m 64 --tile-n 128 --tile-k 64"
            )
        if args.warp_specialized_fragments:
            grouped_kernel = W8A16WarpSpecializedFragmentKernel
        elif args.named_tile_fragment_cache:
            grouped_kernel = W8A16GroupedMNamedTileFragmentKernel
        elif args.shared_fragment_cache:
            grouped_kernel = W8A16GroupedMSharedFragmentKernel
        else:
            grouped_kernel = W8A16GroupedMWarpKernel
        kernel = grouped_kernel(
            size_m=args.rows,
            size_n=size_n,
            size_k=size_k,
            stages=args.stages,
            bf16_scale_mul=args.bf16_scale_mul,
            post_scale_groups=args.post_scale_groups,
        )
    else:
        if (
            args.shared_fragment_cache
            or args.named_tile_fragment_cache
            or args.warp_specialized_fragments
        ):
            raise ValueError(
                "fragment-cache variants require --grouped-m-warps"
            )
        packed_kernel = (
            W8A16PackedTwoStageGemmKernel
            if args.two_stage_register_pipeline
            else W8A16PackedGemmKernel
        )
        kernel = packed_kernel(
            size_m=args.rows,
            size_n=size_n,
            size_k=size_k,
            block_m=args.block_m,
            tile_n=args.tile_n,
            tile_k=args.tile_k,
            stages=args.stages,
            bf16_scale_mul=args.bf16_scale_mul,
            post_scale_groups=args.post_scale_groups,
        )
    print(
        "compile "
        f"tensor={args.tensor} rows={args.rows} shape={size_n}x{size_k} "
        f"block_m={args.block_m} tile={args.tile_n}x{args.tile_k} stages={args.stages} "
        f"bf16_scale_mul={args.bf16_scale_mul} "
        f"post_scale_groups={args.post_scale_groups} "
        f"grouped_m_warps={args.grouped_m_warps} "
        f"shared_fragment_cache={args.shared_fragment_cache} "
        f"named_tile_fragment_cache={args.named_tile_fragment_cache} "
        f"warp_specialized_fragments={args.warp_specialized_fragments} "
        f"two_stage_register_pipeline={args.two_stage_register_pipeline} "
        f"grid={args.grid} weight_copies={args.weight_copies} "
        f"shared_bytes={kernel.shared_words * 4}"
    )
    compiled = cute.compile(kernel, *cute_args)
    if args.export_dir is not None:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        name = (
            f"w8a16_{args.tensor.replace('-', '_')}_m{args.rows}_"
            f"bm{args.block_m}_n{args.tile_n}_k{args.tile_k}_s{args.stages}"
        )
        compiled.export_to_c(str(args.export_dir), name, f"glmrt_{name}")
    compiled(*cute_args)
    torch.cuda.synchronize()
    reference = torch.mm(activations[0], weight.T)
    quality = metrics(outputs[0], reference)
    print(
        "quality "
        f"relative_l2={quality['relative_l2']:.9f} "
        f"cosine={quality['cosine']:.9f} "
        f"max_abs={quality['max_abs']:.6f}"
    )
    if args.warp_specialized_fragments and args.rows >= 64:
        for row_begin in (0, 32):
            row_end = row_begin + 32
            group_quality = metrics(
                outputs[0][row_begin:row_end],
                reference[row_begin:row_end],
            )
            print(
                "quality_group "
                f"rows={row_begin}-{row_end - 1} "
                f"relative_l2={group_quality['relative_l2']:.9f} "
                f"cosine={group_quality['cosine']:.9f} "
                f"max_abs={group_quality['max_abs']:.6f}"
            )
    if args.compare_simt:
        launch_simt_projection(simt_weight)
        torch.cuda.synchronize()
        packed_vs_simt = metrics(outputs[0], simt_output)
        simt_vs_bf16 = metrics(simt_output, reference)
        print(
            "quality_compare "
            f"packed_vs_simt_relative_l2={packed_vs_simt['relative_l2']:.9f} "
            f"packed_vs_simt_cosine={packed_vs_simt['cosine']:.9f} "
            f"simt_vs_bf16_relative_l2={simt_vs_bf16['relative_l2']:.9f} "
            f"simt_vs_bf16_cosine={simt_vs_bf16['cosine']:.9f}"
        )
    if args.compare_packed_m1:
        launch_m1_packed_projection(m1_packed_weights[0])
        torch.cuda.synchronize()
        m1_packed_vs_bf16 = metrics(m1_packed_output, reference)
        print(
            "quality_shared_packed_m1 "
            f"relative_l2={m1_packed_vs_bf16['relative_l2']:.9f} "
            f"cosine={m1_packed_vs_bf16['cosine']:.9f} "
            f"max_abs={m1_packed_vs_bf16['max_abs']:.6f}"
        )
    if args.compare_w8a8_wmma:
        quantize_activations()
        launch_w8a8_projection(simt_weights[0])
        torch.cuda.synchronize()
        w8a8_vs_bf16 = metrics(w8a8_output, reference)
        print(
            "quality_w8a8_wmma "
            f"relative_l2={w8a8_vs_bf16['relative_l2']:.9f} "
            f"cosine={w8a8_vs_bf16['cosine']:.9f} "
            f"max_abs={w8a8_vs_bf16['max_abs']:.6f}"
        )
    timing_args = [
        make_cute_args(0, torch.cuda.current_stream(), packed_weight)
        for packed_weight in packed_weights
    ]
    timing = bench(
        lambda index: compiled(*timing_args[index % len(timing_args)]),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    bf16_timing = bench(
        lambda index: torch.mm(
            activations[0],
            bf16_weights[index % len(bf16_weights)].T,
            out=outputs[0],
        ),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(
        "timing "
        f"kernel=cute-packed-register-dequant median_ms={timing.median_ms:.6f} "
        f"range_ms={timing.minimum_ms:.6f}-{timing.maximum_ms:.6f}"
    )
    print(
        "timing "
        f"kernel=bf16-cublas median_ms={bf16_timing.median_ms:.6f} "
        f"range_ms={bf16_timing.minimum_ms:.6f}-{bf16_timing.maximum_ms:.6f}"
    )
    if args.compare_simt:
        simt_timing = bench(
            lambda index: launch_simt_projection(
                simt_weights[index % len(simt_weights)]
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            "timing "
            f"kernel=w8-simt median_ms={simt_timing.median_ms:.6f} "
            f"range_ms={simt_timing.minimum_ms:.6f}-{simt_timing.maximum_ms:.6f}"
        )
    if args.compare_packed_m1:
        m1_packed_timing = bench(
            lambda index: launch_m1_packed_projection(
                m1_packed_weights[index % len(m1_packed_weights)]
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            "timing "
            f"kernel=w8-shared-packed-m1 median_ms={m1_packed_timing.median_ms:.6f} "
            f"range_ms={m1_packed_timing.minimum_ms:.6f}-"
            f"{m1_packed_timing.maximum_ms:.6f}"
        )
    if args.compare_w8a8_wmma:
        quantize_activations()
        w8a8_timing = bench(
            lambda index: launch_w8a8_projection(
                simt_weights[index % len(simt_weights)]
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )

        def launch_w8a8_total(index: int) -> None:
            quantize_activations()
            launch_w8a8_projection(simt_weights[index % len(simt_weights)])

        w8a8_total_timing = bench(
            launch_w8a8_total,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            "timing "
            f"kernel=w8a8-wmma-projection median_ms={w8a8_timing.median_ms:.6f} "
            f"range_ms={w8a8_timing.minimum_ms:.6f}-"
            f"{w8a8_timing.maximum_ms:.6f}"
        )
        print(
            "timing "
            f"kernel=w8a8-wmma-quant-plus-projection "
            f"median_ms={w8a8_total_timing.median_ms:.6f} "
            f"range_ms={w8a8_total_timing.minimum_ms:.6f}-"
            f"{w8a8_total_timing.maximum_ms:.6f}"
        )

    if args.lanes > 1:
        lane_args = [
            make_cute_args(
                lane,
                lane_streams[lane],
                packed_weights[lane % len(packed_weights)],
            )
            for lane in range(args.lanes)
        ]
        lane_done = [torch.cuda.Event() for _ in range(args.lanes)]

        def launch_w8_lanes(_: int) -> None:
            for lane, lane_stream in enumerate(lane_streams):
                with torch.cuda.stream(lane_stream):
                    compiled(*lane_args[lane])
                    lane_done[lane].record(lane_stream)
            current = torch.cuda.current_stream()
            for event in lane_done:
                current.wait_event(event)

        def launch_bf16_lanes(_: int) -> None:
            for lane, lane_stream in enumerate(lane_streams):
                with torch.cuda.stream(lane_stream):
                    torch.mm(
                        activations[lane],
                        weight.T,
                        out=outputs[lane],
                    )
                    lane_done[lane].record(lane_stream)
            current = torch.cuda.current_stream()
            for event in lane_done:
                current.wait_event(event)

        w8_lanes = bench(
            launch_w8_lanes,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        bf16_lanes = bench(
            launch_bf16_lanes,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            "timing "
            f"kernel=cute-packed-register-dequant-c{args.lanes} "
            f"median_ms={w8_lanes.median_ms:.6f} "
            f"range_ms={w8_lanes.minimum_ms:.6f}-{w8_lanes.maximum_ms:.6f}"
        )
        print(
            "timing "
            f"kernel=bf16-cublas-c{args.lanes} median_ms={bf16_lanes.median_ms:.6f} "
            f"range_ms={bf16_lanes.minimum_ms:.6f}-{bf16_lanes.maximum_ms:.6f}"
        )


if __name__ == "__main__":
    main()
