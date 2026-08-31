#include "common.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if GLMRT_NATIVE_ENABLE_W8A16_AOT
#include "w8a16_row_major_aot.h"
#endif

namespace {

__global__ void linear_f32_kernel(const float* input, const float* weight, const float* bias,
                                  float* output, size_t rows, size_t input_dim,
                                  size_t output_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * output_dim;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / output_dim;
  const size_t out_col = idx % output_dim;
  const float* input_row = input + row * input_dim;
  const float* weight_row = weight + out_col * input_dim;
  float acc = bias == nullptr ? 0.0f : bias[out_col];
  for (size_t col = 0; col < input_dim; ++col) {
    acc += input_row[col] * weight_row[col];
  }
  output[idx] = acc;
}

__global__ void linear_bf16_kernel(const uint16_t* input, const uint16_t* weight,
                                   const uint16_t* bias, uint16_t* output, size_t rows,
                                   size_t input_dim, size_t output_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * output_dim;
  if (idx >= total) {
    return;
  }
  const size_t row = idx / output_dim;
  const size_t out_col = idx % output_dim;
  const uint16_t* input_row = input + row * input_dim;
  const uint16_t* weight_row = weight + out_col * input_dim;
  float acc = bias == nullptr ? 0.0f : bf16_to_f32(bias[out_col]);
  for (size_t col = 0; col < input_dim; ++col) {
    acc += bf16_to_f32(input_row[col]) * bf16_to_f32(weight_row[col]);
  }
  output[idx] = f32_to_bf16(acc);
}

__global__ void linear_bf16_add_bias_kernel(uint16_t* output, const uint16_t* bias, size_t rows,
                                            size_t output_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * output_dim;
  if (idx >= total) {
    return;
  }
  const size_t out_col = idx % output_dim;
  const float value = bf16_to_f32(output[idx]) + bf16_to_f32(bias[out_col]);
  output[idx] = f32_to_bf16(value);
}

__device__ __forceinline__ float warp_sum_f32(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_max_f32(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

template <bool NonCoherent>
__device__ __forceinline__ uint64_t load_w8_u64(const int8_t* pointer) {
  uint32_t low;
  uint32_t high;
  if constexpr (NonCoherent) {
    asm volatile("ld.global.nc.v2.u32 {%0, %1}, [%2];"
                 : "=r"(low), "=r"(high)
                 : "l"(pointer));
  } else {
    const uint2 value = *reinterpret_cast<const uint2*>(pointer);
    low = value.x;
    high = value.y;
  }
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

__device__ __forceinline__ int32_t signed_w8(uint64_t packed, int byte_index) {
  return static_cast<int32_t>(
      static_cast<int8_t>(static_cast<uint8_t>(packed >> (byte_index * 8))));
}

template <int RowsPerWarp, int WarpsPerBlock, bool NonCoherent>
__global__ __launch_bounds__(WarpsPerBlock * 32) void linear_w8a16_group256_m1_simt_kernel(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim) {
  constexpr int kGroupSize = 256;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const size_t row_base =
      (static_cast<size_t>(blockIdx.x) * WarpsPerBlock + warp) * RowsPerWarp;
  if (row_base >= output_dim) {
    return;
  }
  const size_t groups = input_dim / kGroupSize;
  float accumulators[RowsPerWarp] = {};

  for (size_t group = 0; group < groups; ++group) {
    const size_t input_offset = group * kGroupSize + static_cast<size_t>(lane) * 8;
    const uint4 packed_input =
        *reinterpret_cast<const uint4*>(input + input_offset);
    const uint32_t input_words[4] = {
        packed_input.x, packed_input.y, packed_input.z, packed_input.w};

#pragma unroll
    for (int local_row = 0; local_row < RowsPerWarp; ++local_row) {
      const size_t row = row_base + local_row;
      if (row >= output_dim) {
        continue;
      }
      const int8_t* weight_pointer =
          weight + row * input_dim + input_offset;
      const uint64_t packed_weight =
          load_w8_u64<NonCoherent>(weight_pointer);
      float dot = 0.0f;
#pragma unroll
      for (int value = 0; value < 8; ++value) {
        const uint32_t word = input_words[value / 2];
        const uint16_t input_bits = static_cast<uint16_t>(
            word >> ((value & 1) * 16));
        dot = fmaf(bf16_to_f32(input_bits),
                   static_cast<float>(signed_w8(packed_weight, value)), dot);
      }
      accumulators[local_row] =
          fmaf(dot, scales[row * groups + group], accumulators[local_row]);
    }
  }

#pragma unroll
  for (int local_row = 0; local_row < RowsPerWarp; ++local_row) {
    const size_t row = row_base + local_row;
    const float sum = warp_sum_f32(accumulators[local_row]);
    if (lane == 0 && row < output_dim) {
      *reinterpret_cast<__nv_bfloat16*>(output + row) =
          __float2bfloat16_rn(sum);
    }
  }
}

template <int WarpsPerBlock, bool NonCoherent>
__global__ __launch_bounds__(WarpsPerBlock * 32) void
linear_w8a16_group256_m1_simt_shared_input_kernel(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim) {
  constexpr int kGroupSize = 256;
  extern __shared__ __align__(16) uint16_t shared_input[];
  for (size_t offset = static_cast<size_t>(threadIdx.x) * 8;
       offset < input_dim;
       offset += static_cast<size_t>(WarpsPerBlock) * 32 * 8) {
    *reinterpret_cast<uint4*>(shared_input + offset) =
        *reinterpret_cast<const uint4*>(input + offset);
  }
  __syncthreads();

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const size_t row = static_cast<size_t>(blockIdx.x) * WarpsPerBlock + warp;
  if (row >= output_dim) {
    return;
  }
  const size_t groups = input_dim / kGroupSize;
  float accumulator = 0.0f;
  for (size_t group = 0; group < groups; ++group) {
    const size_t input_offset = group * kGroupSize + static_cast<size_t>(lane) * 8;
    const uint4 packed_input =
        *reinterpret_cast<const uint4*>(shared_input + input_offset);
    const uint32_t input_words[4] = {
        packed_input.x, packed_input.y, packed_input.z, packed_input.w};
    const uint64_t packed_weight = load_w8_u64<NonCoherent>(
        weight + row * input_dim + input_offset);
    float dot = 0.0f;
#pragma unroll
    for (int value = 0; value < 8; ++value) {
      const uint32_t word = input_words[value / 2];
      const uint16_t input_bits = static_cast<uint16_t>(
          word >> ((value & 1) * 16));
      dot = fmaf(bf16_to_f32(input_bits),
                 static_cast<float>(signed_w8(packed_weight, value)), dot);
    }
    accumulator = fmaf(dot, scales[row * groups + group], accumulator);
  }
  const float sum = warp_sum_f32(accumulator);
  if (lane == 0) {
    *reinterpret_cast<__nv_bfloat16*>(output + row) =
        __float2bfloat16_rn(sum);
  }
}

// M=2..8 projection with the exact per-row arithmetic used by the recurrent
// M=1 SIMT kernel. A warp owns one output channel and retains one accumulator
// per input row, so each W8 weight vector and group scale is read once for the
// whole speculative target batch. This preserves accepted-row numerical state
// without paying one full weight traversal per row.
template <int BatchRows, int WarpsPerBlock, bool NonCoherent>
__global__ __launch_bounds__(WarpsPerBlock * 32) void
linear_w8a16_group256_m1_parity_batched_kernel(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim) {
  constexpr int kGroupSize = 256;
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const size_t output_row =
      static_cast<size_t>(blockIdx.x) * WarpsPerBlock + warp;
  if (output_row >= output_dim) {
    return;
  }
  const size_t groups = input_dim / kGroupSize;
  float accumulators[BatchRows] = {};

  for (size_t group = 0; group < groups; ++group) {
    const size_t input_offset = group * kGroupSize + static_cast<size_t>(lane) * 8;
    const uint64_t packed_weight = load_w8_u64<NonCoherent>(
        weight + output_row * input_dim + input_offset);
    const float scale = scales[output_row * groups + group];
#pragma unroll
    for (int batch_row = 0; batch_row < BatchRows; ++batch_row) {
      const uint4 packed_input = *reinterpret_cast<const uint4*>(
          input + static_cast<size_t>(batch_row) * input_dim + input_offset);
      const uint32_t input_words[4] = {
          packed_input.x, packed_input.y, packed_input.z, packed_input.w};
      float dot = 0.0f;
#pragma unroll
      for (int value = 0; value < 8; ++value) {
        const uint32_t word = input_words[value / 2];
        const uint16_t input_bits =
            static_cast<uint16_t>(word >> ((value & 1) * 16));
        dot = fmaf(bf16_to_f32(input_bits),
                   static_cast<float>(signed_w8(packed_weight, value)), dot);
      }
      accumulators[batch_row] =
          fmaf(dot, scale, accumulators[batch_row]);
    }
  }

#pragma unroll
  for (int batch_row = 0; batch_row < BatchRows; ++batch_row) {
    const float sum = warp_sum_f32(accumulators[batch_row]);
    if (lane == 0) {
      *reinterpret_cast<__nv_bfloat16*>(
          output + static_cast<size_t>(batch_row) * output_dim + output_row) =
          __float2bfloat16_rn(sum);
    }
  }
}

// M=1 projection over the exact lane-major K16/N64 fragments consumed by the
// packed multirow MMA path. Each lane loads the four adjacent N16 fragments
// for its tensor-core lane in one 32-byte span and accumulates eight output
// channels. Independent K-split warps expose enough parallelism for recurrent
// decode without creating a second W8 permutation.
template <int KSplits, bool NonCoherent>
__global__ __launch_bounds__(KSplits * 32) void
linear_w8a16_group256_m1_warp_packed_kernel(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim) {
  constexpr int kGroupSize = 256;
  constexpr int kKTile = 16;
  constexpr int kNTile = 64;
  __shared__ float split_partials[KSplits][kNTile];

  const int lane = threadIdx.x & 31;
  const int k_split = threadIdx.x / 32;
  const int lane_group = lane & 3;
  const int tc_column = lane / 4;
  const size_t output_n64 = blockIdx.x;
  const size_t n_tiles = output_dim / kNTile;
  const size_t groups = input_dim / kGroupSize;
  const size_t groups_per_split = groups / KSplits;
  const size_t group_begin = static_cast<size_t>(k_split) * groups_per_split;
  const size_t group_end = group_begin + groups_per_split;
  float accumulators0[4] = {};
  float accumulators1[4] = {};

  for (size_t group = group_begin; group < group_end; ++group) {
    float group_dots0[4] = {};
    float group_dots1[4] = {};
#pragma unroll
    for (int tile_in_group = 0; tile_in_group < kGroupSize / kKTile;
         ++tile_in_group) {
      const size_t k_tile = group * (kGroupSize / kKTile) + tile_in_group;
      const size_t packed_position =
          (((k_tile * n_tiles + output_n64) * 128 + lane * 4) * 8);
      uint64_t packed_weights[4];
#pragma unroll
      for (int output_warp = 0; output_warp < 4; ++output_warp) {
        packed_weights[output_warp] = load_w8_u64<NonCoherent>(
            weight + packed_position + output_warp * 8);
      }

      const size_t input_base = k_tile * kKTile + lane_group * 2;
      const uint32_t input_pair0 =
          *reinterpret_cast<const uint32_t*>(input + input_base);
      const uint32_t input_pair8 =
          *reinterpret_cast<const uint32_t*>(input + input_base + 8);
      const uint16_t input_bits0 = static_cast<uint16_t>(input_pair0);
      const uint16_t input_bits1 = static_cast<uint16_t>(input_pair0 >> 16);
      const uint16_t input_bits8 = static_cast<uint16_t>(input_pair8);
      const uint16_t input_bits9 = static_cast<uint16_t>(input_pair8 >> 16);

#pragma unroll
      for (int output_warp = 0; output_warp < 4; ++output_warp) {
        const uint64_t packed_weight = packed_weights[output_warp];
        group_dots0[output_warp] = fmaf(
            bf16_to_f32(input_bits0),
            static_cast<float>(signed_w8(packed_weight, 0)),
            group_dots0[output_warp]);
        group_dots0[output_warp] = fmaf(
            bf16_to_f32(input_bits8),
            static_cast<float>(signed_w8(packed_weight, 1)),
            group_dots0[output_warp]);
        group_dots1[output_warp] = fmaf(
            bf16_to_f32(input_bits0),
            static_cast<float>(signed_w8(packed_weight, 2)),
            group_dots1[output_warp]);
        group_dots1[output_warp] = fmaf(
            bf16_to_f32(input_bits8),
            static_cast<float>(signed_w8(packed_weight, 3)),
            group_dots1[output_warp]);
        group_dots0[output_warp] = fmaf(
            bf16_to_f32(input_bits1),
            static_cast<float>(signed_w8(packed_weight, 4)),
            group_dots0[output_warp]);
        group_dots0[output_warp] = fmaf(
            bf16_to_f32(input_bits9),
            static_cast<float>(signed_w8(packed_weight, 5)),
            group_dots0[output_warp]);
        group_dots1[output_warp] = fmaf(
            bf16_to_f32(input_bits1),
            static_cast<float>(signed_w8(packed_weight, 6)),
            group_dots1[output_warp]);
        group_dots1[output_warp] = fmaf(
            bf16_to_f32(input_bits9),
            static_cast<float>(signed_w8(packed_weight, 7)),
            group_dots1[output_warp]);
      }
    }
#pragma unroll
    for (int output_warp = 0; output_warp < 4; ++output_warp) {
      const size_t output_base =
          output_n64 * kNTile + output_warp * 16;
      float scale0 = lane_group == 0
          ? scales[group * output_dim + output_base + tc_column]
          : 0.0f;
      float scale1 = lane_group == 0
          ? scales[group * output_dim + output_base + tc_column + 8]
          : 0.0f;
      scale0 = __shfl_sync(0xffffffffu, scale0, 0, 4);
      scale1 = __shfl_sync(0xffffffffu, scale1, 0, 4);
      accumulators0[output_warp] =
          fmaf(group_dots0[output_warp], scale0, accumulators0[output_warp]);
      accumulators1[output_warp] =
          fmaf(group_dots1[output_warp], scale1, accumulators1[output_warp]);
    }
  }

#pragma unroll
  for (int output_warp = 0; output_warp < 4; ++output_warp) {
    for (int offset = 2; offset > 0; offset /= 2) {
      accumulators0[output_warp] += __shfl_down_sync(
          0xffffffffu, accumulators0[output_warp], offset, 4);
      accumulators1[output_warp] += __shfl_down_sync(
          0xffffffffu, accumulators1[output_warp], offset, 4);
    }
  }
  if (lane_group == 0) {
#pragma unroll
    for (int output_warp = 0; output_warp < 4; ++output_warp) {
      const int local_output = output_warp * 16 + tc_column;
      split_partials[k_split][local_output] = accumulators0[output_warp];
      split_partials[k_split][local_output + 8] = accumulators1[output_warp];
    }
  }
  __syncthreads();

  if (k_split == 0 && lane_group == 0) {
#pragma unroll
    for (int output_warp = 0; output_warp < 4; ++output_warp) {
      const int local_output = output_warp * 16 + tc_column;
      float total0 = split_partials[0][local_output];
      float total1 = split_partials[0][local_output + 8];
#pragma unroll
      for (int split = 1; split < KSplits; ++split) {
        total0 += split_partials[split][local_output];
        total1 += split_partials[split][local_output + 8];
      }
      *reinterpret_cast<__nv_bfloat16*>(
          output + output_n64 * kNTile + local_output) =
          __float2bfloat16_rn(total0);
      *reinterpret_cast<__nv_bfloat16*>(
          output + output_n64 * kNTile + local_output + 8) =
          __float2bfloat16_rn(total1);
    }
  }
}

// M=2..8 projection over the packed K16/N64 resident. One CTA owns one N16
// fragment and reuses every packed weight word across the target rows. Each
// row retains the exact group-FMA, lane reduction, and K-split fold order of
// an independent packed M=1 launch, so accepted speculative rows remain
// bitwise recurrent-equivalent.
template <int BatchRows, int KSplits, int ActiveWarps, bool NonCoherent>
__global__ __launch_bounds__(ActiveWarps * 32) void
linear_w8a16_group256_m1_warp_packed_parity_batched_kernel(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim) {
  constexpr int kGroupSize = 256;
  constexpr int kKTile = 16;
  constexpr int kNTile = 64;
  constexpr int kOutputFragment = 16;
  static_assert(KSplits % ActiveWarps == 0);
  __shared__ float split_partials[BatchRows][KSplits][kOutputFragment];

  const int lane = threadIdx.x & 31;
  const int worker_warp = threadIdx.x / 32;
  const int lane_group = lane & 3;
  const int tc_column = lane / 4;
  const size_t output_n16 = blockIdx.x;
  const size_t output_n64 = output_n16 / 4;
  const int output_warp = static_cast<int>(output_n16 & 3);
  const size_t output_base = output_n16 * kOutputFragment;
  const size_t n_tiles = output_dim / kNTile;
  const size_t groups = input_dim / kGroupSize;
  const size_t groups_per_split = groups / KSplits;
  for (int k_split = worker_warp; k_split < KSplits; k_split += ActiveWarps) {
    const size_t group_begin = static_cast<size_t>(k_split) * groups_per_split;
    const size_t group_end = group_begin + groups_per_split;
    float accumulators0[BatchRows] = {};
    float accumulators1[BatchRows] = {};

    for (size_t group = group_begin; group < group_end; ++group) {
      float group_dots0[BatchRows] = {};
      float group_dots1[BatchRows] = {};
#pragma unroll
      for (int tile_in_group = 0; tile_in_group < kGroupSize / kKTile;
           ++tile_in_group) {
        const size_t k_tile = group * (kGroupSize / kKTile) + tile_in_group;
        const size_t packed_position =
            (((k_tile * n_tiles + output_n64) * 128 + lane * 4 + output_warp) *
             8);
        const uint64_t packed_weight =
            load_w8_u64<NonCoherent>(weight + packed_position);
        const size_t input_base = k_tile * kKTile + lane_group * 2;
#pragma unroll
        for (int batch_row = 0; batch_row < BatchRows; ++batch_row) {
          const uint16_t* input_row =
              input + static_cast<size_t>(batch_row) * input_dim;
          const uint32_t input_pair0 =
              *reinterpret_cast<const uint32_t*>(input_row + input_base);
          const uint32_t input_pair8 =
              *reinterpret_cast<const uint32_t*>(input_row + input_base + 8);
          const uint16_t input_bits0 = static_cast<uint16_t>(input_pair0);
          const uint16_t input_bits1 = static_cast<uint16_t>(input_pair0 >> 16);
          const uint16_t input_bits8 = static_cast<uint16_t>(input_pair8);
          const uint16_t input_bits9 = static_cast<uint16_t>(input_pair8 >> 16);
          group_dots0[batch_row] =
              fmaf(bf16_to_f32(input_bits0),
                   static_cast<float>(signed_w8(packed_weight, 0)),
                   group_dots0[batch_row]);
          group_dots0[batch_row] =
              fmaf(bf16_to_f32(input_bits8),
                   static_cast<float>(signed_w8(packed_weight, 1)),
                   group_dots0[batch_row]);
          group_dots1[batch_row] =
              fmaf(bf16_to_f32(input_bits0),
                   static_cast<float>(signed_w8(packed_weight, 2)),
                   group_dots1[batch_row]);
          group_dots1[batch_row] =
              fmaf(bf16_to_f32(input_bits8),
                   static_cast<float>(signed_w8(packed_weight, 3)),
                   group_dots1[batch_row]);
          group_dots0[batch_row] =
              fmaf(bf16_to_f32(input_bits1),
                   static_cast<float>(signed_w8(packed_weight, 4)),
                   group_dots0[batch_row]);
          group_dots0[batch_row] =
              fmaf(bf16_to_f32(input_bits9),
                   static_cast<float>(signed_w8(packed_weight, 5)),
                   group_dots0[batch_row]);
          group_dots1[batch_row] =
              fmaf(bf16_to_f32(input_bits1),
                   static_cast<float>(signed_w8(packed_weight, 6)),
                   group_dots1[batch_row]);
          group_dots1[batch_row] =
              fmaf(bf16_to_f32(input_bits9),
                   static_cast<float>(signed_w8(packed_weight, 7)),
                   group_dots1[batch_row]);
        }
      }
      float scale0 = lane_group == 0
                         ? scales[group * output_dim + output_base + tc_column]
                         : 0.0f;
      float scale1 =
          lane_group == 0
              ? scales[group * output_dim + output_base + tc_column + 8]
              : 0.0f;
      scale0 = __shfl_sync(0xffffffffu, scale0, 0, 4);
      scale1 = __shfl_sync(0xffffffffu, scale1, 0, 4);
#pragma unroll
      for (int batch_row = 0; batch_row < BatchRows; ++batch_row) {
        accumulators0[batch_row] =
            fmaf(group_dots0[batch_row], scale0, accumulators0[batch_row]);
        accumulators1[batch_row] =
            fmaf(group_dots1[batch_row], scale1, accumulators1[batch_row]);
      }
    }

#pragma unroll
    for (int batch_row = 0; batch_row < BatchRows; ++batch_row) {
      for (int offset = 2; offset > 0; offset /= 2) {
        accumulators0[batch_row] +=
            __shfl_down_sync(0xffffffffu, accumulators0[batch_row], offset, 4);
        accumulators1[batch_row] +=
            __shfl_down_sync(0xffffffffu, accumulators1[batch_row], offset, 4);
      }
      if (lane_group == 0) {
        split_partials[batch_row][k_split][tc_column] =
            accumulators0[batch_row];
        split_partials[batch_row][k_split][tc_column + 8] =
            accumulators1[batch_row];
      }
    }
  }
  __syncthreads();

  if (worker_warp == 0 && lane_group == 0) {
#pragma unroll
    for (int batch_row = 0; batch_row < BatchRows; ++batch_row) {
      float total0 = split_partials[batch_row][0][tc_column];
      float total1 = split_partials[batch_row][0][tc_column + 8];
#pragma unroll
      for (int split = 1; split < KSplits; ++split) {
        total0 += split_partials[batch_row][split][tc_column];
        total1 += split_partials[batch_row][split][tc_column + 8];
      }
      *reinterpret_cast<__nv_bfloat16*>(
          output + static_cast<size_t>(batch_row) * output_dim + output_base +
          tc_column) = __float2bfloat16_rn(total0);
      *reinterpret_cast<__nv_bfloat16*>(
          output + static_cast<size_t>(batch_row) * output_dim + output_base +
          tc_column + 8) = __float2bfloat16_rn(total1);
    }
  }
}

// Multirow projection with BF16 I/O, dynamically quantized signed-int8
// activations, and the same row-major signed-int8/group-256 weight resident
// used by the M=1 SIMT kernel.  INT8 tensor-core accumulation is reset at each
// quantization group so the two FP32 scales can be applied before accumulating
// into the final BF16 tile.
__global__ __launch_bounds__(512) void linear_w8a8_group256_wmma_kernel(
    const int8_t* input, const float* input_scales, const int8_t* weight,
    const float* weight_scales, uint16_t* output, size_t rows,
    size_t input_dim, size_t output_dim) {
  using namespace nvcuda;
  constexpr int kGroupSize = 256;
  constexpr int kTileM = 64;
  constexpr int kTileN = 64;
  constexpr int kWarpTile = 16;
  constexpr int kWarpsN = kTileN / kWarpTile;
  constexpr int kThreads = 512;
  constexpr int kTileElements = kTileM * kTileN;
  extern __shared__ __align__(16) int8_t shared_bytes[];
  int8_t* shared_a = shared_bytes;
  int8_t* shared_b = shared_a + kTileM * kGroupSize;
  int32_t* shared_c = reinterpret_cast<int32_t*>(
      shared_b + kTileN * kGroupSize);

  const int warp = threadIdx.x / 32;
  const int warp_m = warp / kWarpsN;
  const int warp_n = warp % kWarpsN;
  const size_t row_base = static_cast<size_t>(blockIdx.y) * kTileM;
  const size_t output_base = static_cast<size_t>(blockIdx.x) * kTileN;
  const size_t groups = input_dim / kGroupSize;
  float accumulators[kTileElements / kThreads] = {};

  for (size_t group = 0; group < groups; ++group) {
    for (int index = threadIdx.x; index < kTileM * kGroupSize;
         index += kThreads) {
      const int local_row = index / kGroupSize;
      const int local_k = index - local_row * kGroupSize;
      const size_t global_row = row_base + local_row;
      shared_a[index] = global_row < rows
          ? input[global_row * input_dim + group * kGroupSize + local_k]
          : 0;
    }
    for (int index = threadIdx.x; index < kTileN * kGroupSize;
         index += kThreads) {
      const int local_column = index / kGroupSize;
      const int local_k = index - local_column * kGroupSize;
      const size_t global_column = output_base + local_column;
      shared_b[index] = weight[
          global_column * input_dim + group * kGroupSize + local_k];
    }
    __syncthreads();

    wmma::fragment<wmma::accumulator, kWarpTile, kWarpTile, kWarpTile,
                   int32_t> integer_accumulator;
    wmma::fill_fragment(integer_accumulator, 0);
#pragma unroll
    for (int k = 0; k < kGroupSize; k += kWarpTile) {
      wmma::fragment<wmma::matrix_a, kWarpTile, kWarpTile, kWarpTile,
                     signed char, wmma::row_major> a_fragment;
      wmma::fragment<wmma::matrix_b, kWarpTile, kWarpTile, kWarpTile,
                     signed char, wmma::col_major> b_fragment;
      wmma::load_matrix_sync(
          a_fragment,
          reinterpret_cast<const signed char*>(
              shared_a + warp_m * kWarpTile * kGroupSize + k),
          kGroupSize);
      wmma::load_matrix_sync(
          b_fragment,
          reinterpret_cast<const signed char*>(
              shared_b + warp_n * kWarpTile * kGroupSize + k),
          kGroupSize);
      wmma::mma_sync(
          integer_accumulator, a_fragment, b_fragment, integer_accumulator);
    }
    wmma::store_matrix_sync(
        shared_c + warp_m * kWarpTile * kTileN + warp_n * kWarpTile,
        integer_accumulator, kTileN, wmma::mem_row_major);
    __syncthreads();

#pragma unroll
    for (int item = 0; item < kTileElements / kThreads; ++item) {
      const int index = threadIdx.x + item * kThreads;
      const int local_row = index / kTileN;
      const int local_column = index - local_row * kTileN;
      const size_t global_row = row_base + local_row;
      const size_t global_column = output_base + local_column;
      if (global_row < rows) {
        const float combined_scale =
            input_scales[global_row * groups + group] *
            weight_scales[global_column * groups + group];
        accumulators[item] = fmaf(
            static_cast<float>(shared_c[index]), combined_scale,
            accumulators[item]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int item = 0; item < kTileElements / kThreads; ++item) {
    const int index = threadIdx.x + item * kThreads;
    const int local_row = index / kTileN;
    const int local_column = index - local_row * kTileN;
    const size_t global_row = row_base + local_row;
    const size_t global_column = output_base + local_column;
    if (global_row < rows) {
      *reinterpret_cast<__nv_bfloat16*>(
          output + global_row * output_dim + global_column) =
          __float2bfloat16_rn(accumulators[item]);
    }
  }
}

template <int Layout>
__global__ __launch_bounds__(256) void quantize_bf16_w8a16_group256_kernel(
    const uint16_t* source, int8_t* weight, float* scales, size_t input_dim,
    size_t output_dim) {
  constexpr int kGroupSize = 256;
  constexpr int kWarps = 8;
  __shared__ float warp_maxima[kWarps];
  __shared__ float block_scale;

  const size_t groups = input_dim / kGroupSize;
  const size_t flat_group = blockIdx.x;
  const size_t row = flat_group / groups;
  const size_t group = flat_group - row * groups;
  const size_t k = group * kGroupSize + threadIdx.x;
  const float value = bf16_to_f32(source[row * input_dim + k]);
  float maximum = warp_max_f32(fabsf(value));
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  if (lane == 0) {
    warp_maxima[warp] = maximum;
  }
  __syncthreads();
  if (warp == 0) {
    maximum = lane < kWarps ? warp_maxima[lane] : 0.0f;
    maximum = warp_max_f32(maximum);
    if (lane == 0) {
      block_scale = maximum == 0.0f ? 1.0f : maximum / 127.0f;
      if constexpr (Layout == 1 || Layout == 2) {
        scales[group * output_dim + row] = block_scale;
      } else {
        scales[row * groups + group] = block_scale;
      }
    }
  }
  __syncthreads();
  int quantized = __float2int_rn(value / block_scale);
  quantized = max(-127, min(127, quantized));
  if constexpr (Layout == 1) {
    weight[k * output_dim + row] = static_cast<int8_t>(quantized);
  } else if constexpr (Layout == 2) {
    constexpr size_t kKTile = 16;
    constexpr size_t kNTile = 64;
    const size_t k_tile = k / kKTile;
    const size_t k_in_tile = k % kKTile;
    const size_t n_tile = row / kNTile;
    const size_t n_in_tile = row % kNTile;
    const size_t output_warp = n_in_tile / 16;
    const size_t tc_column = n_in_tile % 8;
    const size_t lane_group = (k_in_tile % 8) / 2;
    const size_t lane = tc_column * 4 + lane_group;
    const size_t byte = (k_in_tile & 1) * 4 +
                        ((n_in_tile % 16) >= 8 ? 2 : 0) +
                        (k_in_tile >= 8 ? 1 : 0);
    const size_t n_tiles = output_dim / kNTile;
    const size_t packed_offset =
        (((k_tile * n_tiles + n_tile) * 128 + lane * 4 + output_warp) * 8) +
        byte;
    weight[packed_offset] = static_cast<int8_t>(quantized);
  } else {
    weight[row * input_dim + k] = static_cast<int8_t>(quantized);
  }
}

__global__ __launch_bounds__(256) void dequantize_block_fp8_e4m3_bf16_kernel(
    const uint8_t* source, const float* scales, uint16_t* output,
    size_t input_dim, size_t output_dim) {
  constexpr size_t kBlockRows = 128;
  constexpr size_t kBlockColumns = 128;
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t values = input_dim * output_dim;
  if (index >= values) {
    return;
  }
  const size_t row = index / input_dim;
  const size_t column = index - row * input_dim;
  const size_t scale_columns = (input_dim + kBlockColumns - 1) / kBlockColumns;
  const float scale =
      scales[(row / kBlockRows) * scale_columns + column / kBlockColumns];
  output[index] = f32_to_bf16(f8e4m3_to_f32(source[index]) * scale);
}

// Transpose while dequantizing so both the K-major W8 input reads and the
// row-major BF16 output writes are coalesced.  This is the bounded-scratch
// fallback for projection row counts where direct W8A16 loses to cuBLAS.
__global__ __launch_bounds__(256) void dequantize_w8a16_group256_bf16_kernel(
    const int8_t* weight_k_major, const float* scales_group_major,
    uint16_t* weight_bf16, size_t input_dim, size_t output_dim) {
  constexpr int kTile = 32;
  constexpr int kBlockRows = 8;
  __shared__ __nv_bfloat16 tile[kTile][kTile + 1];

  const int lane = threadIdx.x;
  const int row_in_block = threadIdx.y;
  const size_t output_base = static_cast<size_t>(blockIdx.x) * kTile;
  const size_t input_base = static_cast<size_t>(blockIdx.y) * kTile;
  const size_t output_column = output_base + lane;

#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    const size_t input_column = input_base + row_in_block + offset;
    if (input_column < input_dim && output_column < output_dim) {
      const int8_t quantized = weight_k_major[
          input_column * output_dim + output_column];
      const float scale = scales_group_major[
          (input_column / 256) * output_dim + output_column];
      tile[row_in_block + offset][lane] = __float2bfloat16_rn(
          static_cast<float>(quantized) * scale);
    }
  }
  __syncthreads();

  const size_t transposed_output = output_base + row_in_block;
  const size_t transposed_input = input_base + lane;
#pragma unroll
  for (int offset = 0; offset < kTile; offset += kBlockRows) {
    const size_t output_row = transposed_output + offset;
    if (output_row < output_dim && transposed_input < input_dim) {
      *reinterpret_cast<__nv_bfloat16*>(
          weight_bf16 + output_row * input_dim + transposed_input) =
          tile[lane][row_in_block + offset];
    }
  }
}

__device__ __forceinline__ uint32_t lossless_bf16_exponent(
    uint32_t code, uint32_t position, uint32_t header, const uint32_t* escape_entries) {
  if (code != 15u) {
    return (header & 0xffu) + code;
  }
  const uint32_t escape_count = (header >> 8) & 0xffu;
  for (uint32_t slot = 0; slot < escape_count; ++slot) {
    const uint32_t entry = escape_entries[slot];
    if ((entry & 0xffffu) == position) {
      return entry >> 16;
    }
  }
  return 0u;
}

__global__ __launch_bounds__(128) void linear_lossless_bf16_m1_kernel(
    const uint16_t* input, const uint8_t* low, const uint8_t* codes,
    const uint32_t* metadata, uint16_t* output, size_t input_dim,
    size_t metadata_stride_words) {
  constexpr size_t kTileValues = 1024;
  constexpr size_t kPairsPerTile = kTileValues / 2;
  constexpr int kThreads = 128;
  constexpr int kWarps = kThreads / 32;
  __shared__ float warp_sums[kWarps];

  const size_t row = blockIdx.x;
  const size_t row_base = row * input_dim;
  const size_t code_row_base = row * (input_dim / 2);
  const size_t tiles_per_row = input_dim / kTileValues;
  float acc = 0.0f;

  for (size_t tile = 0; tile < tiles_per_row; ++tile) {
    const size_t tile_value_base = tile * kTileValues;
    const size_t tile_pair_base = tile * kPairsPerTile;
    const uint32_t* tile_metadata =
        metadata + (row * tiles_per_row + tile) * metadata_stride_words;
    const uint32_t header = tile_metadata[0];
    const uint32_t* escape_entries = tile_metadata + 1;

#pragma unroll
    for (size_t pair_offset = threadIdx.x; pair_offset < kPairsPerTile;
         pair_offset += kThreads) {
      const size_t position = pair_offset * 2;
      const size_t input_index = tile_value_base + position;
      const size_t weight_index = row_base + input_index;
      const uint16_t low_pair =
          *reinterpret_cast<const uint16_t*>(low + weight_index);
      const uint8_t packed_code = codes[code_row_base + tile_pair_base + pair_offset];
      const uint32_t low0 = low_pair & 0xffu;
      const uint32_t low1 = low_pair >> 8;
      const uint32_t exponent0 = lossless_bf16_exponent(
          packed_code & 0x0fu, static_cast<uint32_t>(position), header, escape_entries);
      const uint32_t exponent1 = lossless_bf16_exponent(
          packed_code >> 4, static_cast<uint32_t>(position + 1), header, escape_entries);
      const uint16_t bits0 = static_cast<uint16_t>(
          (low0 & 0x7fu) | ((low0 & 0x80u) << 8) | (exponent0 << 7));
      const uint16_t bits1 = static_cast<uint16_t>(
          (low1 & 0x7fu) | ((low1 & 0x80u) << 8) | (exponent1 << 7));
      acc = fmaf(bf16_to_f32(input[input_index]), bf16_to_f32(bits0), acc);
      acc = fmaf(bf16_to_f32(input[input_index + 1]), bf16_to_f32(bits1), acc);
    }
  }

  acc = warp_sum_f32(acc);
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x / 32;
  if (lane == 0) {
    warp_sums[warp] = acc;
  }
  __syncthreads();
  if (warp == 0) {
    acc = lane < kWarps ? warp_sums[lane] : 0.0f;
    acc = warp_sum_f32(acc);
    if (lane == 0) {
      *reinterpret_cast<__nv_bfloat16*>(output + row) = __float2bfloat16_rn(acc);
    }
  }
}

glmrt_status_t status_from_cublas(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? GLMRT_STATUS_OK : GLMRT_STATUS_INTERNAL_ERROR;
}

struct TritonDriverKernel {
  CUmodule module = nullptr;
  CUfunction function = nullptr;
};

std::mutex& triton_driver_kernel_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<std::string, TritonDriverKernel>&
triton_driver_kernel_cache() {
  static std::unordered_map<std::string, TritonDriverKernel> kernels;
  return kernels;
}

glmrt_status_t ensure_cuda_driver_context() {
  if (cuInit(0) != CUDA_SUCCESS) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  CUcontext context = nullptr;
  if (cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
    if (cudaFree(nullptr) != cudaSuccess ||
        cuCtxGetCurrent(&context) != CUDA_SUCCESS || context == nullptr) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t triton_driver_kernel(const char* cubin_path,
                                    const char* kernel_name,
                                    CUfunction* out) {
  if (cubin_path == nullptr || kernel_name == nullptr || out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const std::string key = std::string(cubin_path) + "\n" + kernel_name;
  std::lock_guard<std::mutex> lock(triton_driver_kernel_mutex());
  auto& kernels = triton_driver_kernel_cache();
  const auto found = kernels.find(key);
  if (found != kernels.end()) {
    *out = found->second.function;
    return GLMRT_STATUS_OK;
  }
  if (ensure_cuda_driver_context() != GLMRT_STATUS_OK) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  TritonDriverKernel loaded;
  if (cuModuleLoad(&loaded.module, cubin_path) != CUDA_SUCCESS ||
      cuModuleGetFunction(&loaded.function, loaded.module, kernel_name) !=
          CUDA_SUCCESS) {
    if (loaded.module != nullptr) {
      cuModuleUnload(loaded.module);
    }
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  *out = loaded.function;
  kernels.emplace(key, loaded);
  return GLMRT_STATUS_OK;
}

glmrt_status_t triton_driver_kernel_data(const char* cache_key,
                                         const unsigned char* cubin,
                                         const char* kernel_name,
                                         CUfunction* out) {
  if (cache_key == nullptr || cubin == nullptr || kernel_name == nullptr ||
      out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  std::lock_guard<std::mutex> lock(triton_driver_kernel_mutex());
  auto& kernels = triton_driver_kernel_cache();
  const auto found = kernels.find(cache_key);
  if (found != kernels.end()) {
    *out = found->second.function;
    return GLMRT_STATUS_OK;
  }
  if (ensure_cuda_driver_context() != GLMRT_STATUS_OK) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  TritonDriverKernel loaded;
  if (cuModuleLoadData(&loaded.module, cubin) != CUDA_SUCCESS ||
      cuModuleGetFunction(&loaded.function, loaded.module, kernel_name) !=
          CUDA_SUCCESS) {
    if (loaded.module != nullptr) {
      cuModuleUnload(loaded.module);
    }
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  *out = loaded.function;
  kernels.emplace(cache_key, loaded);
  return GLMRT_STATUS_OK;
}

glmrt_status_t cublas_handle(cublasHandle_t* out) {
  if (out == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  static thread_local cublasHandle_t handle = nullptr;
  if (handle == nullptr) {
    const cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return status_from_cublas(status);
    }
  }
  *out = handle;
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_linear_args(const float* input, const float* weight, const float* output,
                                    size_t rows, size_t input_dim, size_t output_dim) {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || input_dim == 0 || output_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, input_dim, &ignored) ||
      !checked_mul(output_dim, input_dim, &ignored) ||
      !checked_mul(rows, output_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_linear_bf16_args(const uint16_t* input, const uint16_t* weight,
                                         const uint16_t* output, size_t rows, size_t input_dim,
                                         size_t output_dim) {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (rows == 0 || input_dim == 0 || output_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t ignored = 0;
  if (!checked_mul(rows, input_dim, &ignored) ||
      !checked_mul(output_dim, input_dim, &ignored) ||
      !checked_mul(rows, output_dim, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_linear_bf16_strided_batched_args(
    const uint16_t* input, const uint16_t* weight, const uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t weight_batch_stride,
    size_t output_batch_stride) {
  if (input == nullptr || weight == nullptr || output == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (batch_count == 0 || rows == 0 || input_dim == 0 || output_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t matrix_values = 0;
  size_t batch_offset = 0;
  size_t ignored = 0;
  if (!checked_mul(rows, input_dim, &matrix_values) ||
      !checked_mul(batch_count - 1, input_batch_stride, &batch_offset) ||
      !checked_add(batch_offset, matrix_values, &ignored) ||
      !checked_mul(output_dim, input_dim, &matrix_values) ||
      !checked_mul(batch_count - 1, weight_batch_stride, &batch_offset) ||
      !checked_add(batch_offset, matrix_values, &ignored) ||
      !checked_mul(rows, output_dim, &matrix_values) ||
      !checked_mul(batch_count - 1, output_batch_stride, &batch_offset) ||
      !checked_add(batch_offset, matrix_values, &ignored)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t validate_bf16_graph_linear_buffers(glmrt_device_buffer_t input,
                                                  glmrt_device_buffer_t weight,
                                                  const glmrt_device_buffer_t* bias,
                                                  glmrt_device_buffer_t output, size_t rows,
                                                  size_t input_dim, size_t output_dim) {
  const glmrt_status_t valid = validate_linear_bf16_args(
      static_cast<const uint16_t*>(input.ptr), static_cast<const uint16_t*>(weight.ptr),
      static_cast<const uint16_t*>(output.ptr), rows, input_dim, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  size_t input_values = 0;
  size_t weight_values = 0;
  size_t output_values = 0;
  if (!checked_mul(rows, input_dim, &input_values) ||
      !checked_mul(output_dim, input_dim, &weight_values) ||
      !checked_mul(rows, output_dim, &output_values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t input_bytes = 0;
  size_t weight_bytes = 0;
  size_t output_bytes = 0;
  size_t bias_bytes = 0;
  if (!checked_mul(input_values, sizeof(uint16_t), &input_bytes) ||
      !checked_mul(weight_values, sizeof(uint16_t), &weight_bytes) ||
      !checked_mul(output_values, sizeof(uint16_t), &output_bytes) ||
      !checked_mul(output_dim, sizeof(uint16_t), &bias_bytes)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  if (input.bytes < input_bytes || weight.bytes < weight_bytes || output.bytes < output_bytes) {
    return GLMRT_STATUS_BUFFER_TOO_SMALL;
  }
  if (bias != nullptr) {
    if (bias->ptr == nullptr) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    if (bias->bytes < bias_bytes) {
      return GLMRT_STATUS_BUFFER_TOO_SMALL;
    }
  }
  return GLMRT_STATUS_OK;
}

glmrt_status_t launch_linear_bf16_cublas(const uint16_t* input, const uint16_t* weight,
                                         const uint16_t* bias, uint16_t* output, size_t rows,
                                         size_t input_dim, size_t output_dim,
                                         cudaStream_t stream) {
  const glmrt_status_t valid =
      validate_linear_bf16_args(input, weight, output, rows, input_dim, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      input_dim > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cublasHandle_t handle = nullptr;
  glmrt_status_t handle_status = cublas_handle(&handle);
  if (handle_status != GLMRT_STATUS_OK) {
    return handle_status;
  }
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status_from_cublas(status);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  status = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(output_dim),
                        static_cast<int>(rows), static_cast<int>(input_dim), &alpha, weight,
                        CUDA_R_16BF, static_cast<int>(input_dim), input, CUDA_R_16BF,
                        static_cast<int>(input_dim), &beta, output, CUDA_R_16BF,
                        static_cast<int>(output_dim), CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status_from_cublas(status);
  }

  if (bias != nullptr) {
    const int threads = 256;
    const size_t total = rows * output_dim;
    const size_t block_count = (total - 1) / threads + 1;
    if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
      return GLMRT_STATUS_INVALID_ARGUMENT;
    }
    linear_bf16_add_bias_kernel<<<static_cast<int>(block_count), threads, 0, stream>>>(
        output, bias, rows, output_dim);
    return status_from_cuda(cudaGetLastError());
  }
  return GLMRT_STATUS_OK;
}

struct CublasLtLinearShape {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t input = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
};

void destroy_cublaslt_linear_shape(CublasLtLinearShape* shape) {
  if (shape == nullptr) {
    return;
  }
  if (shape->output != nullptr) {
    cublasLtMatrixLayoutDestroy(shape->output);
  }
  if (shape->input != nullptr) {
    cublasLtMatrixLayoutDestroy(shape->input);
  }
  if (shape->weight != nullptr) {
    cublasLtMatrixLayoutDestroy(shape->weight);
  }
  if (shape->operation != nullptr) {
    cublasLtMatmulDescDestroy(shape->operation);
  }
  *shape = {};
}

bool create_cublaslt_linear_shape(CublasLtLinearShape* shape, size_t rows,
                                  size_t input_dim, size_t output_dim) {
  if (shape == nullptr || rows == 0 || input_dim == 0 || output_dim == 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      input_dim > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  if (cublasLtMatmulDescCreate(&shape->operation, CUBLAS_COMPUTE_32F,
                               CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) {
    destroy_cublaslt_linear_shape(shape);
    return false;
  }
  const cublasOperation_t transa = CUBLAS_OP_T;
  const cublasOperation_t transb = CUBLAS_OP_N;
  if (cublasLtMatmulDescSetAttribute(shape->operation,
                                    CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                    sizeof(transa)) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatmulDescSetAttribute(shape->operation,
                                    CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                    sizeof(transb)) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&shape->weight, CUDA_R_16BF, input_dim,
                                 output_dim, input_dim) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&shape->input, CUDA_R_16BF, input_dim, rows,
                                 input_dim) != CUBLAS_STATUS_SUCCESS ||
      cublasLtMatrixLayoutCreate(&shape->output, CUDA_R_16BF, output_dim, rows,
                                 output_dim) != CUBLAS_STATUS_SUCCESS) {
    destroy_cublaslt_linear_shape(shape);
    return false;
  }
  return true;
}

struct CublasLtM1ParityBatchedPlan {
  CublasLtLinearShape shape;
  cublasLtMatmulAlgo_t algorithm{};
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  std::mutex qualification_mutex;
  bool qualified = false;
  bool use_cublaslt = false;
};

cublasLtHandle_t cublaslt_handle() {
  static thread_local cublasLtHandle_t handle = nullptr;
  if (handle == nullptr && cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
    return nullptr;
  }
  return handle;
}

std::unordered_map<std::string, CublasLtM1ParityBatchedPlan*>&
cublaslt_m1_parity_plan_cache() {
  static std::unordered_map<std::string, CublasLtM1ParityBatchedPlan*> plans;
  return plans;
}

std::mutex& cublaslt_m1_parity_plan_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

CublasLtM1ParityBatchedPlan* create_cublaslt_m1_parity_plan(
    size_t rows, size_t input_dim, size_t output_dim) {
  constexpr size_t kHeuristicWorkspaceLimit = 8ull * 1024ull * 1024ull;
  constexpr size_t kMinimumWorkspaceBytes = 1024ull * 1024ull;
  constexpr size_t kMaximumWorkspaceBytes = 64ull * 1024ull * 1024ull;
  cublasLtHandle_t handle = cublaslt_handle();
  if (handle == nullptr) {
    return nullptr;
  }
  auto* plan = new CublasLtM1ParityBatchedPlan();
  if (!create_cublaslt_linear_shape(&plan->shape, rows, input_dim, output_dim)) {
    delete plan;
    return nullptr;
  }
  CublasLtLinearShape recurrent_shape;
  if (!create_cublaslt_linear_shape(&recurrent_shape, 1, input_dim, output_dim)) {
    destroy_cublaslt_linear_shape(&plan->shape);
    delete plan;
    return nullptr;
  }
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t result{};
  int result_count = 0;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &kHeuristicWorkspaceLimit, sizeof(kHeuristicWorkspaceLimit));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, recurrent_shape.operation, recurrent_shape.weight,
        recurrent_shape.input, recurrent_shape.output, recurrent_shape.output,
        preference, 1, &result, &result_count);
  }
  if (preference != nullptr) {
    cublasLtMatmulPreferenceDestroy(preference);
  }
  destroy_cublaslt_linear_shape(&recurrent_shape);
  if (status != CUBLAS_STATUS_SUCCESS || result_count != 1 ||
      result.state != CUBLAS_STATUS_SUCCESS) {
    destroy_cublaslt_linear_shape(&plan->shape);
    delete plan;
    return nullptr;
  }
  size_t scaled_workspace = 0;
  if (!checked_mul(std::max(result.workspaceSize, size_t{1}), rows,
                   &scaled_workspace)) {
    destroy_cublaslt_linear_shape(&plan->shape);
    delete plan;
    return nullptr;
  }
  plan->workspace_bytes = std::max(kMinimumWorkspaceBytes, scaled_workspace);
  if (plan->workspace_bytes > kMaximumWorkspaceBytes ||
      cudaMalloc(&plan->workspace, plan->workspace_bytes) != cudaSuccess) {
    destroy_cublaslt_linear_shape(&plan->shape);
    delete plan;
    return nullptr;
  }
  plan->algorithm = result.algo;
  return plan;
}

CublasLtM1ParityBatchedPlan* cublaslt_m1_parity_plan(
    size_t rows, size_t input_dim, size_t output_dim) {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return nullptr;
  }
  const std::string key = std::to_string(device) + "/" +
                          std::to_string(rows) + "/" +
                          std::to_string(input_dim) + "/" +
                          std::to_string(output_dim);
  std::lock_guard<std::mutex> lock(cublaslt_m1_parity_plan_cache_mutex());
  auto& plans = cublaslt_m1_parity_plan_cache();
  const auto found = plans.find(key);
  if (found != plans.end()) {
    return found->second;
  }
  CublasLtM1ParityBatchedPlan* plan =
      create_cublaslt_m1_parity_plan(rows, input_dim, output_dim);
  if (plan != nullptr) {
    plans.emplace(key, plan);
  }
  return plan;
}

glmrt_status_t launch_linear_bf16_recurrent_m1_rows(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim, cudaStream_t stream) {
  for (size_t row = 0; row < rows; ++row) {
    const glmrt_status_t status = launch_linear_bf16_cublas(
        input + row * input_dim, weight, nullptr, output + row * output_dim, 1,
        input_dim, output_dim, stream);
    if (status != GLMRT_STATUS_OK) {
      return status;
    }
  }
  return GLMRT_STATUS_OK;
}

cublasStatus_t launch_cublaslt_m1_parity_plan(
    cublasLtHandle_t handle, CublasLtM1ParityBatchedPlan* plan,
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    cudaStream_t stream) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  return cublasLtMatmul(
      handle, plan->shape.operation, &alpha, weight, plan->shape.weight, input,
      plan->shape.input, &beta, output, plan->shape.output, output,
      plan->shape.output, &plan->algorithm, plan->workspace,
      plan->workspace_bytes, stream);
}

glmrt_status_t launch_linear_bf16_m1_parity_batched_cublaslt(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim, cudaStream_t stream) {
  const glmrt_status_t valid =
      validate_linear_bf16_args(input, weight, output, rows, input_dim,
                                output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  if (rows < 2 || rows > 16) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  CublasLtM1ParityBatchedPlan* plan =
      cublaslt_m1_parity_plan(rows, input_dim, output_dim);
  cublasLtHandle_t handle = cublaslt_handle();
  if (plan == nullptr || handle == nullptr) {
    return launch_linear_bf16_recurrent_m1_rows(
        input, weight, output, rows, input_dim, output_dim, stream);
  }

  std::unique_lock<std::mutex> qualification_lock(plan->qualification_mutex);
  if (!plan->qualified) {
    size_t reference_values = 0;
    size_t reference_bytes = 0;
    uint16_t* reference = nullptr;
    bool exact = false;
    if (checked_mul(rows, output_dim, &reference_values) &&
        checked_mul(reference_values, sizeof(uint16_t), &reference_bytes) &&
        cudaMalloc(&reference, reference_bytes) == cudaSuccess &&
        launch_cublaslt_m1_parity_plan(handle, plan, input, weight, output,
                                       stream) == CUBLAS_STATUS_SUCCESS &&
        launch_linear_bf16_recurrent_m1_rows(
            input, weight, reference, rows, input_dim, output_dim, stream) ==
            GLMRT_STATUS_OK &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
      std::vector<uint16_t> candidate_host(reference_values);
      std::vector<uint16_t> reference_host(reference_values);
      if (cudaMemcpy(candidate_host.data(), output, reference_bytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess &&
          cudaMemcpy(reference_host.data(), reference, reference_bytes,
                     cudaMemcpyDeviceToHost) == cudaSuccess) {
        exact = candidate_host == reference_host;
      }
    }
    if (reference != nullptr) {
      cudaFree(reference);
    }
    plan->qualified = true;
    plan->use_cublaslt = exact;
    if (exact) {
      if (rows == 8) {
        // Production always warms the full physical target width before graph
        // capture. Use that allocation-safe window to qualify/JIT every
        // smaller adaptive width as well; otherwise the first short tail can
        // encounter cudaMalloc while its graph is already being captured and
        // become pinned to the exact but multi-launch fallback.
        qualification_lock.unlock();
        for (size_t preload_rows = 2; preload_rows < rows; ++preload_rows) {
          const glmrt_status_t preload_status =
              launch_linear_bf16_m1_parity_batched_cublaslt(
                  input, weight, output, preload_rows, input_dim, output_dim,
                  stream);
          if (preload_status != GLMRT_STATUS_OK) {
            return preload_status;
          }
        }
        if (launch_cublaslt_m1_parity_plan(handle, plan, input, weight, output,
                                           stream) == CUBLAS_STATUS_SUCCESS) {
          return GLMRT_STATUS_OK;
        }
        {
          std::lock_guard<std::mutex> lock(plan->qualification_mutex);
          plan->use_cublaslt = false;
        }
        return launch_linear_bf16_recurrent_m1_rows(
            input, weight, output, rows, input_dim, output_dim, stream);
      }
      return GLMRT_STATUS_OK;
    }
    return launch_linear_bf16_recurrent_m1_rows(
        input, weight, output, rows, input_dim, output_dim, stream);
  }

  const bool use_cublaslt = plan->use_cublaslt;
  qualification_lock.unlock();
  if (use_cublaslt &&
      launch_cublaslt_m1_parity_plan(handle, plan, input, weight, output,
                                     stream) == CUBLAS_STATUS_SUCCESS) {
    return GLMRT_STATUS_OK;
  }
  {
    std::lock_guard<std::mutex> lock(plan->qualification_mutex);
    plan->use_cublaslt = false;
  }
  return launch_linear_bf16_recurrent_m1_rows(
      input, weight, output, rows, input_dim, output_dim, stream);
}

glmrt_status_t launch_linear_bf16_strided_batched_cublas(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t weight_batch_stride,
    size_t output_batch_stride, cudaStream_t stream) {
  const glmrt_status_t valid = validate_linear_bf16_strided_batched_args(
      input, weight, output, batch_count, rows, input_dim, output_dim,
      input_batch_stride, weight_batch_stride, output_batch_stride);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  constexpr size_t kMaxInt = static_cast<size_t>(std::numeric_limits<int>::max());
  constexpr size_t kMaxStride = static_cast<size_t>(std::numeric_limits<long long>::max());
  if (batch_count > kMaxInt || rows > kMaxInt || input_dim > kMaxInt ||
      output_dim > kMaxInt || input_batch_stride > kMaxStride ||
      weight_batch_stride > kMaxStride || output_batch_stride > kMaxStride) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cublasHandle_t handle = nullptr;
  glmrt_status_t handle_status = cublas_handle(&handle);
  if (handle_status != GLMRT_STATUS_OK) {
    return handle_status;
  }
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status_from_cublas(status);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  // cuBLAS reports its own API status but does not consume CUDA's per-thread
  // launch-error slot. Isolate this launch from graph-capture fallbacks that
  // may have intentionally recovered from cudaErrorNotSupported.
  (void)cudaGetLastError();
  status = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(output_dim),
      static_cast<int>(rows), static_cast<int>(input_dim), &alpha, weight,
      CUDA_R_16BF, static_cast<int>(input_dim),
      static_cast<long long>(weight_batch_stride), input, CUDA_R_16BF,
      static_cast<int>(input_dim), static_cast<long long>(input_batch_stride),
      &beta, output, CUDA_R_16BF, static_cast<int>(output_dim),
      static_cast<long long>(output_batch_stride), static_cast<int>(batch_count),
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status_from_cublas(status);
  }
  return status_from_cuda(cudaGetLastError());
}

glmrt_status_t launch_matmul_bf16_strided_batched_cublas(
    const uint16_t* input, const uint16_t* right, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t right_batch_stride,
    size_t output_batch_stride, cudaStream_t stream) {
  const glmrt_status_t valid = validate_linear_bf16_strided_batched_args(
      input, right, output, batch_count, rows, input_dim, output_dim,
      input_batch_stride, right_batch_stride, output_batch_stride);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  constexpr size_t kMaxInt = static_cast<size_t>(std::numeric_limits<int>::max());
  constexpr size_t kMaxStride = static_cast<size_t>(std::numeric_limits<long long>::max());
  if (batch_count > kMaxInt || rows > kMaxInt || input_dim > kMaxInt ||
      output_dim > kMaxInt || input_batch_stride > kMaxStride ||
      right_batch_stride > kMaxStride || output_batch_stride > kMaxStride) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cublasHandle_t handle = nullptr;
  glmrt_status_t handle_status = cublas_handle(&handle);
  if (handle_status != GLMRT_STATUS_OK) {
    return handle_status;
  }
  cublasStatus_t status = cublasSetStream(handle, stream);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status_from_cublas(status);
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;
  (void)cudaGetLastError();
  // Row-major output = input[rows,input_dim] * right[input_dim,output_dim].
  // cuBLAS sees right as the column-major transpose [output_dim,input_dim].
  status = cublasGemmStridedBatchedEx(
      handle, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(output_dim),
      static_cast<int>(rows), static_cast<int>(input_dim), &alpha, right,
      CUDA_R_16BF, static_cast<int>(output_dim),
      static_cast<long long>(right_batch_stride), input, CUDA_R_16BF,
      static_cast<int>(input_dim), static_cast<long long>(input_batch_stride),
      &beta, output, CUDA_R_16BF, static_cast<int>(output_dim),
      static_cast<long long>(output_batch_stride), static_cast<int>(batch_count),
      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return status_from_cublas(status);
  }
  return status_from_cuda(cudaGetLastError());
}

}  // namespace

extern "C" glmrt_status_t glmrt_cuda_graph_update_linear_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t input,
    glmrt_device_buffer_t weight, const glmrt_device_buffer_t* bias,
    glmrt_device_buffer_t output, size_t rows, size_t input_dim, size_t output_dim) {
  if (cuda_graph == nullptr || cuda_graph_exec == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_status_t valid =
      validate_bf16_graph_linear_buffers(input, weight, bias, output, rows, input_dim, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }

  cudaGraphNode_t node = nullptr;
  const glmrt_status_t node_status = find_kernel_node_by_index(cuda_graph, kernel_node_index, &node);
  if (node_status != GLMRT_STATUS_OK) {
    return node_status;
  }

  cudaKernelNodeParams existing = {};
  cudaError_t err = cudaGraphKernelNodeGetParams(node, &existing);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  if (existing.func != reinterpret_cast<void*>(linear_bf16_kernel)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  const uint16_t* input_ptr = static_cast<const uint16_t*>(input.ptr);
  const uint16_t* weight_ptr = static_cast<const uint16_t*>(weight.ptr);
  const uint16_t* bias_ptr =
      bias == nullptr ? nullptr : static_cast<const uint16_t*>(bias->ptr);
  uint16_t* output_ptr = static_cast<uint16_t*>(output.ptr);
  void* args[] = {
      &input_ptr,
      &weight_ptr,
      &bias_ptr,
      &output_ptr,
      &rows,
      &input_dim,
      &output_dim,
  };
  const int threads = 256;
  const size_t total = rows * output_dim;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }

  cudaKernelNodeParams params = {};
  params.func = reinterpret_cast<void*>(linear_bf16_kernel);
  params.gridDim = dim3(static_cast<unsigned int>(block_count), 1, 1);
  params.blockDim = dim3(threads, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  err = cudaGraphKernelNodeSetParams(node, &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  err = cudaGraphExecKernelNodeSetParams(reinterpret_cast<cudaGraphExec_t>(cuda_graph_exec), node,
                                         &params);
  if (err != cudaSuccess) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  return GLMRT_STATUS_OK;
}

extern "C" glmrt_status_t glmrt_cuda_linear_f32_async(const float* input, const float* weight,
                                                      const float* bias, float* output,
                                                      size_t rows, size_t input_dim,
                                                      size_t output_dim, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_linear_args(input, weight, output, rows, input_dim, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * output_dim;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  linear_f32_kernel<<<blocks, threads, 0, stream>>>(input, weight, bias, output, rows, input_dim,
                                                    output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_linear_f32(const float* input, const float* weight,
                                                const float* bias, float* output, size_t rows,
                                                size_t input_dim, size_t output_dim) {
  const glmrt_status_t status =
      glmrt_cuda_linear_f32_async(input, weight, bias, output, rows, input_dim, output_dim,
                                  nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_linear_bf16_async(const uint16_t* input,
                                                       const uint16_t* weight,
                                                       const uint16_t* bias, uint16_t* output,
                                                       size_t rows, size_t input_dim,
                                                       size_t output_dim, void* cuda_stream) {
  const glmrt_status_t valid =
      validate_linear_bf16_args(input, weight, output, rows, input_dim, output_dim);
  if (valid != GLMRT_STATUS_OK) {
    return valid;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int threads = 256;
  const size_t total = rows * output_dim;
  const size_t block_count = (total - 1) / threads + 1;
  if (block_count > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const int blocks = static_cast<int>(block_count);
  linear_bf16_kernel<<<blocks, threads, 0, stream>>>(input, weight, bias, output, rows, input_dim,
                                                     output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_linear_bf16(const uint16_t* input, const uint16_t* weight,
                                                 const uint16_t* bias, uint16_t* output,
                                                 size_t rows, size_t input_dim,
                                                 size_t output_dim) {
  const glmrt_status_t status =
      glmrt_cuda_linear_bf16_async(input, weight, bias, output, rows, input_dim, output_dim,
                                   nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_linear_bf16_cublas_async(
    const uint16_t* input, const uint16_t* weight, const uint16_t* bias, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim, void* cuda_stream) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  return launch_linear_bf16_cublas(input, weight, bias, output, rows, input_dim, output_dim,
                                   stream);
}

extern "C" glmrt_status_t
glmrt_cuda_linear_bf16_m1_parity_batched_cublaslt_async(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim, void* cuda_stream) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  return launch_linear_bf16_m1_parity_batched_cublaslt(
      input, weight, output, rows, input_dim, output_dim, stream);
}

extern "C" glmrt_status_t glmrt_cuda_linear_lossless_bf16_m1_async(
    const uint16_t* input, const uint8_t* low, const uint8_t* codes,
    const uint32_t* metadata, uint16_t* output, size_t input_dim,
    size_t output_dim, size_t metadata_stride_words, void* cuda_stream) {
  constexpr size_t kTileValues = 1024;
  if (input == nullptr || low == nullptr || codes == nullptr || metadata == nullptr ||
      output == nullptr || input_dim == 0 || output_dim == 0 ||
      input_dim % kTileValues != 0 || metadata_stride_words < 2 ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int blocks = static_cast<int>(output_dim);
  linear_lossless_bf16_m1_kernel<<<blocks, 128, 0, stream>>>(
      input, low, codes, metadata, output, input_dim, metadata_stride_words);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_quantize_bf16_w8a16_group256_async(
    const uint16_t* source, int8_t* weight, float* scales, size_t input_dim,
    size_t output_dim, int k_major, void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  if (source == nullptr || weight == nullptr || scales == nullptr ||
      input_dim == 0 || output_dim == 0 || input_dim % kGroupSize != 0 ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t blocks = 0;
  if (!checked_mul(output_dim, input_dim / kGroupSize, &blocks) ||
      blocks > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      (k_major != 0 && k_major != 1)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (k_major != 0) {
    quantize_bf16_w8a16_group256_kernel<1>
        <<<static_cast<int>(blocks), 256, 0, stream>>>(
            source, weight, scales, input_dim, output_dim);
  } else {
    quantize_bf16_w8a16_group256_kernel<0>
        <<<static_cast<int>(blocks), 256, 0, stream>>>(
            source, weight, scales, input_dim, output_dim);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_quantize_bf16_w8a16_group256_packed_async(
    const uint16_t* source, int8_t* weight, float* scales, size_t input_dim,
    size_t output_dim, void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  constexpr size_t kNTile = 64;
  if (source == nullptr || weight == nullptr || scales == nullptr ||
      input_dim == 0 || output_dim == 0 || input_dim % kGroupSize != 0 ||
      output_dim % kNTile != 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t groups = input_dim / kGroupSize;
  const size_t flat_groups = output_dim * groups;
  if (flat_groups >
      static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  quantize_bf16_w8a16_group256_kernel<2>
      <<<static_cast<unsigned int>(flat_groups), 256, 0, stream>>>(
          source, weight, scales, input_dim, output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_dequantize_block_fp8_e4m3_bf16_async(
    const uint8_t* source, const float* scales, uint16_t* output,
    size_t input_dim, size_t output_dim, void* cuda_stream) {
  if (source == nullptr || scales == nullptr || output == nullptr || input_dim == 0 ||
      output_dim == 0) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  size_t values = 0;
  if (!checked_mul(input_dim, output_dim, &values)) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  constexpr size_t kThreads = 256;
  const size_t blocks = (values + kThreads - 1) / kThreads;
  if (blocks > static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  dequantize_block_fp8_e4m3_bf16_kernel
      <<<static_cast<unsigned int>(blocks), kThreads, 0, stream>>>(
          source, scales, output, input_dim, output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_dequantize_w8a16_group256_bf16_async(
    const int8_t* weight_k_major, const float* scales_group_major,
    uint16_t* weight_bf16, size_t input_dim, size_t output_dim,
    void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  constexpr size_t kTile = 32;
  if (weight_k_major == nullptr || scales_group_major == nullptr ||
      weight_bf16 == nullptr || input_dim == 0 || output_dim == 0 ||
      input_dim % kGroupSize != 0 ||
      (input_dim + kTile - 1) / kTile >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      (output_dim + kTile - 1) / kTile >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const dim3 blocks(
      static_cast<unsigned int>((output_dim + kTile - 1) / kTile),
      static_cast<unsigned int>((input_dim + kTile - 1) / kTile));
  const dim3 threads(32, 8);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  dequantize_w8a16_group256_bf16_kernel<<<blocks, threads, 0, stream>>>(
      weight_k_major, scales_group_major, weight_bf16, input_dim, output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_linear_w8a16_group256_m1_simt_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim, int variant,
    void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  if (input == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || input_dim == 0 || output_dim == 0 ||
      input_dim % kGroupSize != 0 ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
#define GLMRT_LAUNCH_W8_SIMT(ROWS, WARPS, NC)                                  \
  do {                                                                         \
    constexpr size_t rows_per_block = (ROWS) * (WARPS);                        \
    const size_t blocks = (output_dim + rows_per_block - 1) / rows_per_block;  \
    linear_w8a16_group256_m1_simt_kernel<(ROWS), (WARPS), (NC)>                \
        <<<static_cast<int>(blocks), (WARPS) * 32, 0, stream>>>(               \
            input, weight, scales, output, input_dim, output_dim);             \
  } while (false)
  switch (variant) {
    case 0: GLMRT_LAUNCH_W8_SIMT(1, 4, false); break;
    case 1: GLMRT_LAUNCH_W8_SIMT(2, 4, false); break;
    case 2: GLMRT_LAUNCH_W8_SIMT(4, 4, false); break;
    case 3: GLMRT_LAUNCH_W8_SIMT(1, 4, true); break;
    case 4: GLMRT_LAUNCH_W8_SIMT(2, 4, true); break;
    case 5: GLMRT_LAUNCH_W8_SIMT(4, 4, true); break;
    case 6: GLMRT_LAUNCH_W8_SIMT(2, 8, false); break;
    case 7: GLMRT_LAUNCH_W8_SIMT(4, 8, false); break;
    case 8: GLMRT_LAUNCH_W8_SIMT(2, 8, true); break;
    case 9: GLMRT_LAUNCH_W8_SIMT(4, 8, true); break;
    case 10: GLMRT_LAUNCH_W8_SIMT(1, 8, false); break;
    case 11: GLMRT_LAUNCH_W8_SIMT(1, 8, true); break;
    case 12:
      linear_w8a16_group256_m1_simt_shared_input_kernel<4, false>
          <<<static_cast<int>((output_dim + 3) / 4), 128,
             input_dim * sizeof(uint16_t), stream>>>(
              input, weight, scales, output, input_dim, output_dim);
      break;
    case 13:
      linear_w8a16_group256_m1_simt_shared_input_kernel<4, true>
          <<<static_cast<int>((output_dim + 3) / 4), 128,
             input_dim * sizeof(uint16_t), stream>>>(
              input, weight, scales, output, input_dim, output_dim);
      break;
    case 14:
      linear_w8a16_group256_m1_simt_shared_input_kernel<8, false>
          <<<static_cast<int>((output_dim + 7) / 8), 256,
             input_dim * sizeof(uint16_t), stream>>>(
              input, weight, scales, output, input_dim, output_dim);
      break;
    case 15:
      linear_w8a16_group256_m1_simt_shared_input_kernel<8, true>
          <<<static_cast<int>((output_dim + 7) / 8), 256,
             input_dim * sizeof(uint16_t), stream>>>(
              input, weight, scales, output, input_dim, output_dim);
      break;
    default:
#undef GLMRT_LAUNCH_W8_SIMT
      return GLMRT_STATUS_INVALID_ARGUMENT;
  }
#undef GLMRT_LAUNCH_W8_SIMT
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_linear_w8a16_group256_m1_parity_batched_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  constexpr int kWarpsPerBlock = 4;
  if (input == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || rows < 2 || rows > 16 || input_dim == 0 ||
      output_dim == 0 || input_dim % kGroupSize != 0 ||
      output_dim > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const size_t blocks =
      (output_dim + static_cast<size_t>(kWarpsPerBlock) - 1) /
      static_cast<size_t>(kWarpsPerBlock);
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
#define GLMRT_LAUNCH_W8_PARITY_BATCHED(ROWS)                                  \
  linear_w8a16_group256_m1_parity_batched_kernel<(ROWS), kWarpsPerBlock, true> \
      <<<static_cast<int>(blocks), kWarpsPerBlock * 32, 0, stream>>>(          \
          input, weight, scales, output, input_dim, output_dim)
  switch (rows) {
    case 2: GLMRT_LAUNCH_W8_PARITY_BATCHED(2); break;
    case 3: GLMRT_LAUNCH_W8_PARITY_BATCHED(3); break;
    case 4: GLMRT_LAUNCH_W8_PARITY_BATCHED(4); break;
    case 5: GLMRT_LAUNCH_W8_PARITY_BATCHED(5); break;
    case 6: GLMRT_LAUNCH_W8_PARITY_BATCHED(6); break;
    case 7: GLMRT_LAUNCH_W8_PARITY_BATCHED(7); break;
    case 8: GLMRT_LAUNCH_W8_PARITY_BATCHED(8); break;
    case 9: GLMRT_LAUNCH_W8_PARITY_BATCHED(9); break;
    case 10: GLMRT_LAUNCH_W8_PARITY_BATCHED(10); break;
    case 11: GLMRT_LAUNCH_W8_PARITY_BATCHED(11); break;
    case 12: GLMRT_LAUNCH_W8_PARITY_BATCHED(12); break;
    case 13: GLMRT_LAUNCH_W8_PARITY_BATCHED(13); break;
    case 14: GLMRT_LAUNCH_W8_PARITY_BATCHED(14); break;
    case 15: GLMRT_LAUNCH_W8_PARITY_BATCHED(15); break;
    case 16: GLMRT_LAUNCH_W8_PARITY_BATCHED(16); break;
    default:
#undef GLMRT_LAUNCH_W8_PARITY_BATCHED
      return GLMRT_STATUS_INVALID_ARGUMENT;
  }
#undef GLMRT_LAUNCH_W8_PARITY_BATCHED
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim, void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  constexpr size_t kNTile = 64;
  if (input == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || input_dim == 0 || output_dim == 0 ||
      input_dim % (kGroupSize * 8) != 0 ||
      output_dim % kNTile != 0 ||
      output_dim / kNTile >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int blocks = static_cast<int>(output_dim / kNTile);
  if (input_dim % (kGroupSize * 32) == 0) {
    linear_w8a16_group256_m1_warp_packed_kernel<32, true>
        <<<blocks, 1024, 0, stream>>>(
            input, weight, scales, output, input_dim, output_dim);
  } else {
    linear_w8a16_group256_m1_warp_packed_kernel<8, true>
        <<<blocks, 256, 0, stream>>>(
            input, weight, scales, output, input_dim, output_dim);
  }
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t
glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  constexpr size_t kNTile = 64;
  if (input == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || rows < 2 || rows > 16 || input_dim == 0 ||
      output_dim == 0 || input_dim % (kGroupSize * 8) != 0 ||
      output_dim % kNTile != 0 ||
      output_dim / 16 >
          static_cast<size_t>(std::numeric_limits<int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const int blocks = static_cast<int>(output_dim / 16);

#define GLMRT_LAUNCH_W8_PACKED_PARITY(ROWS, SPLITS, THREADS)                  \
  linear_w8a16_group256_m1_warp_packed_parity_batched_kernel<                \
      (ROWS), (SPLITS), (THREADS) / 32, true>                                \
      <<<blocks, (THREADS), 0, stream>>>(                                    \
          input, weight, scales, output, input_dim, output_dim)
#define GLMRT_DISPATCH_W8_PACKED_PARITY(SPLITS, THREADS)                     \
  switch (rows) {                                                             \
    case 2: GLMRT_LAUNCH_W8_PACKED_PARITY(2, SPLITS, THREADS); break;         \
    case 3: GLMRT_LAUNCH_W8_PACKED_PARITY(3, SPLITS, THREADS); break;         \
    case 4: GLMRT_LAUNCH_W8_PACKED_PARITY(4, SPLITS, THREADS); break;         \
    case 5: GLMRT_LAUNCH_W8_PACKED_PARITY(5, SPLITS, THREADS); break;         \
    case 6: GLMRT_LAUNCH_W8_PACKED_PARITY(6, SPLITS, THREADS); break;         \
    case 7: GLMRT_LAUNCH_W8_PACKED_PARITY(7, SPLITS, THREADS); break;         \
    case 8: GLMRT_LAUNCH_W8_PACKED_PARITY(8, SPLITS, THREADS); break;         \
    case 9: GLMRT_LAUNCH_W8_PACKED_PARITY(9, SPLITS, THREADS); break;         \
    case 10: GLMRT_LAUNCH_W8_PACKED_PARITY(10, SPLITS, THREADS); break;       \
    case 11: GLMRT_LAUNCH_W8_PACKED_PARITY(11, SPLITS, THREADS); break;       \
    case 12: GLMRT_LAUNCH_W8_PACKED_PARITY(12, SPLITS, THREADS); break;       \
    case 13: GLMRT_LAUNCH_W8_PACKED_PARITY(13, SPLITS, THREADS); break;       \
    case 14: GLMRT_LAUNCH_W8_PACKED_PARITY(14, SPLITS, THREADS); break;       \
    case 15: GLMRT_LAUNCH_W8_PACKED_PARITY(15, SPLITS, THREADS); break;       \
    case 16: GLMRT_LAUNCH_W8_PACKED_PARITY(16, SPLITS, THREADS); break;       \
    default: break;                                                           \
  }
  if (input_dim % (kGroupSize * 32) == 0) {
    if (rows == 6) {
      GLMRT_LAUNCH_W8_PACKED_PARITY(6, 32, 1024);
    } else {
      GLMRT_DISPATCH_W8_PACKED_PARITY(32, 512);
    }
  } else {
    GLMRT_DISPATCH_W8_PACKED_PARITY(8, 256);
  }
#undef GLMRT_DISPATCH_W8_PACKED_PARITY
#undef GLMRT_LAUNCH_W8_PACKED_PARITY
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_linear_w8a8_group256_wmma_async(
    const int8_t* input, const float* input_scales, const int8_t* weight,
    const float* weight_scales, uint16_t* output, size_t rows,
    size_t input_dim, size_t output_dim, void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  constexpr size_t kTileM = 64;
  constexpr size_t kTileN = 64;
  constexpr size_t kSharedBytes =
      (kTileM + kTileN) * kGroupSize * sizeof(int8_t) +
      kTileM * kTileN * sizeof(int32_t);
  if (input == nullptr || input_scales == nullptr || weight == nullptr ||
      weight_scales == nullptr || output == nullptr || rows == 0 ||
      input_dim == 0 || output_dim == 0 || input_dim % kGroupSize != 0 ||
      output_dim % kTileN != 0 ||
      (rows + kTileM - 1) / kTileM >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      output_dim / kTileN >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const dim3 blocks(
      static_cast<unsigned int>(output_dim / kTileN),
      static_cast<unsigned int>((rows + kTileM - 1) / kTileM));
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  linear_w8a8_group256_wmma_kernel<<<blocks, 512, kSharedBytes, stream>>>(
      input, input_scales, weight, weight_scales, output, rows, input_dim,
      output_dim);
  return status_from_cuda(cudaGetLastError());
}

extern "C" glmrt_status_t glmrt_cuda_linear_w8a16_group256_triton_file_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    const char* cubin_path, const char* kernel_name, size_t block_m,
    size_t block_n, size_t threads, size_t shared_bytes, void* cuda_stream) {
  constexpr size_t kGroupSize = 256;
  if (input == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || rows == 0 || input_dim == 0 || output_dim == 0 ||
      input_dim % kGroupSize != 0 || block_m == 0 || block_n == 0 ||
      output_dim % block_n != 0 || threads == 0 || threads > 1024 ||
      rows > static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
      (rows + block_m - 1) / block_m >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max()) ||
      output_dim / block_n >
          static_cast<size_t>(std::numeric_limits<unsigned int>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  CUfunction function = nullptr;
  const glmrt_status_t loaded =
      triton_driver_kernel(cubin_path, kernel_name, &function);
  if (loaded != GLMRT_STATUS_OK) {
    return loaded;
  }
  if (shared_bytes > 48 * 1024 &&
      cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          static_cast<int>(shared_bytes)) != CUDA_SUCCESS) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  CUdeviceptr input_arg = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr weight_arg = reinterpret_cast<CUdeviceptr>(weight);
  CUdeviceptr scales_arg = reinterpret_cast<CUdeviceptr>(scales);
  CUdeviceptr output_arg = reinterpret_cast<CUdeviceptr>(output);
  int32_t rows_arg = static_cast<int32_t>(rows);
  CUdeviceptr global_scratch_arg = 0;
  CUdeviceptr profile_scratch_arg = 0;
  void* arguments[] = {
      &input_arg, &weight_arg, &scales_arg, &output_arg, &rows_arg,
      &global_scratch_arg, &profile_scratch_arg};
  const size_t grid =
      ((rows + block_m - 1) / block_m) * (output_dim / block_n);
  const CUresult launched = cuLaunchKernel(
      function, static_cast<unsigned int>(grid), 1, 1,
      static_cast<unsigned int>(threads), 1, 1,
      static_cast<unsigned int>(shared_bytes),
      reinterpret_cast<CUstream>(cuda_stream), arguments, nullptr);
  return launched == CUDA_SUCCESS ? GLMRT_STATUS_OK
                                  : GLMRT_STATUS_INTERNAL_ERROR;
}

extern "C" glmrt_status_t glmrt_cuda_preload_w8a16_group256_aot(
    size_t input_dim, size_t output_dim) {
#if GLMRT_NATIVE_ENABLE_W8A16_AOT
  bool matched = false;
  for (size_t index = 0; index < glmrt_w8a16_aot::kernel_count; ++index) {
    const auto& config = glmrt_w8a16_aot::kernels[index];
    if (config.input_dim != input_dim || config.output_dim != output_dim) {
      continue;
    }
    matched = true;
    const std::string key = "w8a16-aot-" + std::to_string(input_dim) + "-" +
                            std::to_string(output_dim) + "-" +
                            std::to_string(config.max_rows);
    CUfunction function = nullptr;
    const glmrt_status_t loaded = triton_driver_kernel_data(
        key.c_str(), config.cubin, config.symbol, &function);
    if (loaded != GLMRT_STATUS_OK) {
      return loaded;
    }
    if (config.shared_bytes > 48 * 1024 &&
        cuFuncSetAttribute(
            function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            static_cast<int>(config.shared_bytes)) != CUDA_SUCCESS) {
      return GLMRT_STATUS_INTERNAL_ERROR;
    }
  }
  return matched ? GLMRT_STATUS_OK : GLMRT_STATUS_INVALID_ARGUMENT;
#else
  (void)input_dim;
  (void)output_dim;
  return GLMRT_STATUS_CUDA_UNAVAILABLE;
#endif
}

extern "C" glmrt_status_t glmrt_cuda_linear_w8a16_group256_aot_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    void* cuda_stream) {
#if GLMRT_NATIVE_ENABLE_W8A16_AOT
  if (input == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || rows == 0 || rows > 2048 || input_dim == 0 ||
      output_dim == 0 || input_dim % 256 != 0 ||
      rows > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const glmrt_w8a16_aot::KernelConfig* selected = nullptr;
  for (size_t index = 0; index < glmrt_w8a16_aot::kernel_count; ++index) {
    const auto& config = glmrt_w8a16_aot::kernels[index];
    if (config.input_dim == input_dim && config.output_dim == output_dim &&
        rows <= config.max_rows) {
      selected = &config;
      break;
    }
  }
  if (selected == nullptr) {
    return GLMRT_STATUS_INVALID_ARGUMENT;
  }
  const std::string key = "w8a16-aot-" + std::to_string(input_dim) + "-" +
                          std::to_string(output_dim) + "-" +
                          std::to_string(selected->max_rows);
  CUfunction function = nullptr;
  const glmrt_status_t loaded = triton_driver_kernel_data(
      key.c_str(), selected->cubin, selected->symbol, &function);
  if (loaded != GLMRT_STATUS_OK) {
    return loaded;
  }
  if (selected->shared_bytes > 48 * 1024 &&
      cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          static_cast<int>(selected->shared_bytes)) != CUDA_SUCCESS) {
    return GLMRT_STATUS_INTERNAL_ERROR;
  }
  CUdeviceptr input_arg = reinterpret_cast<CUdeviceptr>(input);
  CUdeviceptr weight_arg = reinterpret_cast<CUdeviceptr>(weight);
  CUdeviceptr scales_arg = reinterpret_cast<CUdeviceptr>(scales);
  CUdeviceptr output_arg = reinterpret_cast<CUdeviceptr>(output);
  int32_t rows_arg = static_cast<int32_t>(rows);
  CUdeviceptr global_scratch_arg = 0;
  CUdeviceptr profile_scratch_arg = 0;
  void* arguments[] = {
      &input_arg, &weight_arg, &scales_arg, &output_arg, &rows_arg,
      &global_scratch_arg, &profile_scratch_arg};
  const size_t grid =
      ((rows + selected->block_m - 1) / selected->block_m) *
      (output_dim / selected->block_n);
  const CUresult launched = cuLaunchKernel(
      function, static_cast<unsigned int>(grid), 1, 1,
      static_cast<unsigned int>(selected->threads), 1, 1,
      static_cast<unsigned int>(selected->shared_bytes),
      reinterpret_cast<CUstream>(cuda_stream), arguments, nullptr);
  return launched == CUDA_SUCCESS ? GLMRT_STATUS_OK
                                  : GLMRT_STATUS_INTERNAL_ERROR;
#else
  (void)input;
  (void)weight;
  (void)scales;
  (void)output;
  (void)rows;
  (void)input_dim;
  (void)output_dim;
  (void)cuda_stream;
  return GLMRT_STATUS_CUDA_UNAVAILABLE;
#endif
}

extern "C" glmrt_status_t glmrt_cuda_linear_bf16_cublas(
    const uint16_t* input, const uint16_t* weight, const uint16_t* bias, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim) {
  const glmrt_status_t status =
      glmrt_cuda_linear_bf16_cublas_async(input, weight, bias, output, rows, input_dim,
                                          output_dim, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_linear_bf16_strided_batched_cublas_async(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t weight_batch_stride,
    size_t output_batch_stride, void* cuda_stream) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  return launch_linear_bf16_strided_batched_cublas(
      input, weight, output, batch_count, rows, input_dim, output_dim,
      input_batch_stride, weight_batch_stride, output_batch_stride, stream);
}

extern "C" glmrt_status_t glmrt_cuda_linear_bf16_strided_batched_cublas(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t weight_batch_stride,
    size_t output_batch_stride) {
  const glmrt_status_t status =
      glmrt_cuda_linear_bf16_strided_batched_cublas_async(
          input, weight, output, batch_count, rows, input_dim, output_dim,
          input_batch_stride, weight_batch_stride, output_batch_stride, nullptr);
  if (status != GLMRT_STATUS_OK) {
    return status;
  }
  return status_from_cuda(cudaStreamSynchronize(nullptr));
}

extern "C" glmrt_status_t glmrt_cuda_matmul_bf16_strided_batched_cublas_async(
    const uint16_t* input, const uint16_t* right, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t right_batch_stride,
    size_t output_batch_stride, void* cuda_stream) {
  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  return launch_matmul_bf16_strided_batched_cublas(
      input, right, output, batch_count, rows, input_dim, output_dim,
      input_batch_stride, right_batch_stride, output_batch_stride, stream);
}
