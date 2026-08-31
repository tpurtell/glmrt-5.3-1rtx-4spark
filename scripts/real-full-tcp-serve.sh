#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
startup_started_ns="$(date +%s%N)"
startup_phase_started_ns="$startup_started_ns"

report_shell_startup_phase() {
  local stage="$1" now_ns elapsed_ms total_ms
  now_ns="$(date +%s%N)"
  elapsed_ms=$(((now_ns - startup_phase_started_ns) / 1000000))
  total_ms=$(((now_ns - startup_started_ns) / 1000000))
  echo "coordinator_shell_startup_phase stage=$stage elapsed_ms=$elapsed_ms total_ms=$total_ms" >&2
  startup_phase_started_ns="$now_ns"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: scripts/real-full-tcp-serve.sh

Starts the real GLM full API coordinator for external OpenAI-compatible
clients. By default it also starts all-layer real NVFP4 Spark expert daemons
and leaves them running.

Environment:
  ADDR or GLMRT_REAL_FULL_SERVE_ADDR
      API listen address; default: 0.0.0.0:8000
  GLMRT_REAL_FULL_SERVE_TRANSPORT
      sparse expert transport: tcp or verbs-host; default: tcp
  GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES
      concurrent verbs-host RDMA/NCCL request lanes in 1..8; default: 4.
      This must match the Spark expert setting.
  GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS
      maximum full-top8 combined-batch rows that bypass the general Spark
      CPU row/group planner, in 8..2064; default: 2064.
  GLMRT_PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER
      set 1 to use one completion-queue polling thread per execution lane
      instead of one busy poller per Spark QP; default: 1.
  GLMRT_REAL_FULL_SERVE_KV_CACHE_DTYPE
      coordinator MLA KV cache dtype: bf16, fp8, or nvfp4; default: fp8
  GLMRT_REAL_FULL_SERVE_MAX_CONTEXT_TOKENS
      maximum logical tokens in one target sequence; default: 131072.
  GLMRT_REAL_FULL_KV_POOL_TOKENS
      physical shared target KV/DSA pool capacity, divisible by 64 and at
      least the per-sequence maximum; defaults to MAX_CONTEXT_TOKENS.
      Qualified transition geometry: 400000 logical over 600000 physical.
  GLMRT_REAL_FULL_MAX_EXECUTION_LANES
      resident logical target execution states on the one pinned worker;
      default: 1, current diagnostic maximum: 8. This is the active execution
      width, not the 16-request executing-plus-pending admission ceiling.
      Reservation-aware queued admission is implemented: requests that cannot
      reserve a lane or target pages remain pending without private arenas and
      rotate behind fitting work. C=4 is the qualified serving width; C=8
      remains diagnostic.
  GLMRT_REAL_FULL_MOE_RESPONSE_DTYPE
      Spark MoE response payload: bf16, fp8-e4m3-row-scaled, or
      nvfp4-e2m1-fp8-e4m3; default: bf16. Quantization is applied after
      Spark-local expert aggregation and dequantized into the coordinator FP32
      routed accumulator.
  GLMRT_REAL_FULL_MOE_OWNER_RESPONSE_DTYPE
      Optional small-row owner response dtype; defaults to the general MoE
      response dtype. Use bf16 with an FP8 general dtype to protect decode.
  GLMRT_EXPERT_INTERMEDIATE_SHARDS
      expert MLP intermediate shards: 1 or 4. Defaults to 4 for verbs-host
      serving and 1 for TCP.
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION
      coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or
      spark-rdma-hybrid. Defaults to spark-rdma for four-shard verbs-host
      serving and coordinator otherwise.
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE
      Spark reduction wire dtype: bf16, fp8, or nvfp4; default: fp8.
  GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE
      small-row Spark owner-reduction wire dtype; default: bf16.
  GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS
      minimum rows for distributed Spark reduction; default: 16.
  GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS
      maximum rows for owner reduction in a hybrid mode; default: 8.
  GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION
      partition reduced rows across Spark ranks. Defaults to 1 for the custom
      Spark RDMA modes and 0 otherwise.
  GLMRT_REAL_FULL_MTP_MOE_RESPONSE_DTYPE
      MTP-layer response dtype; default: bf16. MTP batches bypass Spark-side
      reduction to protect draft acceptance. Set inherit to use the general MoE
      response and reduction policy instead.
  GLMRT_REAL_FULL_SERVE_START_EXPERTS
      set 0 to use already-running expert daemons; default: 1
  GLMRT_REAL_FULL_SERVE_CHECK_EXPERTS
      set 0 to skip TCP reachability checks; default: 1
  GLMRT_REAL_FULL_SERVE_BUILD_DAEMON
      set 0 to skip local cargo build before coordinator start; default: 1
  GLMRT_REAL_FULL_SERVE_BUILD_NATIVE
      set 0 to skip the incremental native CUDA/RDMA build; default: 1
  GLMRT_REAL_FULL_SERVE_BUILD_PROFILE
      cargo profile for the coordinator and warmup binary: release or debug;
      default: release
  GLMRT_REAL_FULL_SERVE_REQUIRE_CUDA
      set 0 for diagnostic-only coordinator start without native CUDA; default: 1
  GLMRT_PYTHON
      Python interpreter used by PyO3, native CMake header discovery, and the
      embedded coordinator runtime. Defaults to .venv/bin/python when present,
      otherwise python3.
  GLMRT_REAL_FULL_SERVE_FAST_TOKEN
      set 1 to use the embedding+lm_head shortcut instead of the full
      scheduler path; default: 0
  GLMRT_REAL_FULL_SERVE_FAST_TOKEN_LM_HEAD_ROWS
      resident lm_head rows scored by the fast token path; default: 1024
  GLMRT_REAL_FULL_SERVE_WARMUP_EXPERTS
      set 0 to skip binary ProtocolV2 expert precompile warmup; default: 1
  GLMRT_REAL_FULL_SERVE_WARMUP_TIMEOUT_MS
      timeout for precompile warmup frames; default: 120000
  GLMRT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS
      sparse expert request timeout used by the coordinator; default: 120000
      in this launcher so cold SparkInfer graph capture does not fail the first
      real request.
  GLMRT_REAL_FULL_SERVE_PREWARM_REQUEST
      set 0 to skip one real scheduler decode prewarm before serving;
      default: 1 so graph capture and weight-touch finish before external
      clients connect
  GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE
      set 1 to benchmark extension of a cached prefix during startup.
  GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS
  GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS
      comma-separated cached-prefix and new-suffix row counts; defaults to
      1008 for each. Startup benchmarks their Cartesian product and releases
      each KV state before advancing to the next case.
  GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS
      repeats per cached-prefix/new-suffix case in 1..8; default: 1.
  GLMRT_SPARK_HOSTS
      comma-separated Spark expert owners; default: ostrich,dodo,emu,kiwi
  GLMRT_REAL_FULL_SERVE_EXPERT_LINK_SUFFIX
      optional suffix for coordinator-to-expert addresses; default: empty.
      Do not set this to .200gb unless the coordinator can reach that fabric.
  GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS
      set 0 to disable CUDA graph replay for routed NVFP4 projection kernels;
      default: 1 for real Spark expert startup.
  GLMRT_B12X
      compatibility name: set 0 to disable coordinator Python graph
      capture/SparkInfer paths; default: 1.
  GLMRT_REAL_FULL_REQUEST_THREAD_PINNED
      keep each sequence's graph capture and replay on one CUDA-owning host thread;
      defaults to 1 when GLMRT_B12X=1 and otherwise defaults to 0.
  GLMRT_REAL_FULL_REQUEST_THREAD_PINNED_WORKERS
      sequence-affine CUDA graph worker count; default: 1. Additional workers
      currently duplicate max-context KV/DSA arenas and graph state; raise this
      only for explicit memory-capacity experiments until continuous batching
      shares cache ownership across requests.
  GLMRT_REAL_FULL_SERVE_SHARED_CPU_LIST
      optional taskset CPU list inherited by ordinary coordinator threads.
      Pair it with the two settings below and exclude both selected cores and
      their SMT siblings from this list.
  GLMRT_REAL_FULL_REQUEST_WORKER_CPUS
      optional comma-separated physical CPU assignment, one entry per pinned
      request worker.
  GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU
      optional physical CPU assignment for the persistent sparse-dispatch
      scheduler thread.
  GLMRT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS
      maximum global prefill rows admitted per layer iteration; default: 2048.
      Values above 512 apply only to large suffixes. Spark expert execution
      has native CuTe buckets through 2048 rows, but a wider bucket can reduce
      cross-layer pipeline occupancy and is not automatically faster. Large
      suffixes are divided nearly evenly and retain at least four chunks when
      the configured maximum would otherwise underfill the four lanes.
  GLMRT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS
      minimum uncached suffix rows for a configured chunk width above 256;
      default: 4096. Smaller requests below a 32K cached prefix use at most
      256-row chunks.
  GLMRT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS
      physical chunk width for suffixes below the large-prefill threshold when
      the cached prefix is at least 32K; default: 512, maximum: the direct DSA
      1024-query limit.
  GLMRT_REAL_FULL_ATTENTION_READY_FRONTIER_MAX_TOKENS
      maximum normalized/rotated KV rows retained per layer for direct
      attention reuse; default: 16384. Larger values consume GPU memory.
  GLMRT_REAL_FULL_KV_SNAPSHOT_SAVE
      exact destination directory for one packed KV/DSA snapshot. The
      directory must not already exist. Snapshot writing happens after the
      response finishes and is not included in token timing.
  GLMRT_REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS
      optional committed-prefix cutoff for the saved snapshot. This permits
      one longer request to save a reusable internal token boundary.
  GLMRT_REAL_FULL_KV_SNAPSHOT_SAVE_POINTS
      comma-separated TOKENS=PATH entries. All exact cutoffs are saved after
      one long request, for example 32768=.../code-32k,65536=.../code-64k.
  GLMRT_REAL_FULL_KV_SNAPSHOT_LOAD
      exact packed KV/DSA snapshot directory to restore for each external
      request. Requests must match every saved token and contain at least one
      uncached suffix token. Native MTP additionally requires a snapshot whose
      metadata records a complete committed layer-78 frontier.
  GLMRT_REAL_FULL_CUDNN_MLA_SUFFIX_QUERY_CAPACITY
      power-of-two cuDNN MLA suffix query capacity in 512..2048; default: 2048.
      Set this at least as large as the request prefill chunk size.
  GLMRT_REAL_FULL_MTP
      set 1 to enable recurrent layer-78 MTP drafting and batched target
      verification; default: 0 while quality and acceptance are qualified.
  GLMRT_REAL_FULL_MTP_MIN_D
  GLMRT_REAL_FULL_MTP_MAX_D
      request-local adaptive proposal-depth bounds in 1..7; defaults: 2 and 7.
      Fresh requests start at D=6 and update from at most 16 acceptance cycles.
      Setting either bound enables adaptive mode; an omitted bound keeps its
      default. D=1/M=2 is a qualified ordinary adaptive state whenever the
      configured minimum or output-budget tail selects it.
  GLMRT_REAL_FULL_MTP_DRAFT_TOKENS
      legacy fixed speculative proposal depth in 1..7. Physical target M is
      D+1, so the supported range is M=2..8. This is used only when
      neither adaptive bound is set.
  GLMRT_REAL_FULL_MTP_FULL_MATCH_BONUS
      set 0 only to disable full-match consumption for diagnostics. By default,
      all D matching proposals plus the already-computed target-only sample are
      emitted, and the layer-78 bridge repairs recurrence continuity.
  GLMRT_REAL_FULL_PACKED_FP8_MLA_BATCHED_SUFFIX
      set 0 to disable the default packed-FP8 MLA suffix graph. M=2..8 now use
      one multi-query FlashInfer decode plus one merge launch while retaining
      the recurrent M=1 split and reduction order for every query row.
  GLMRT_COORDINATOR_W8A16_Q_A
  GLMRT_COORDINATOR_W8A16_Q_B
  GLMRT_COORDINATOR_W8A16_O_PROJ
      group-256 W8A16 Q-A/Q-B/O projections. M=1 uses the row-SIMT kernel;
      multirow work uses recurrent-parity or bucketed embedded AOT kernels,
      and the superseded BF16 projection is released after startup
      quantization. All three default to
      enabled in this launcher; set one to 0 for BF16 or W4 diagnostics.
      Do not enable the matching W4 and W8 projection.
  GLMRT_COORDINATOR_W8A16_PACKED_O
      Store O projection W8 directly in the lane-major K16/N64 fragment layout
      shared by its decode and CuTe multirow kernels. Defaults to 1. Q-B stays
      row-major; neither projection retains BF16 or a second W8 layout.
  GLMRT_COORDINATOR_W8A16_ASYNC_ATTENTION
      W8 M=1 CUDA-event handoff from the packed attention graph to residual
      add. Removes the intervening host stream synchronization while preserving
      cross-stream ordering. Requires W8 O projection; defaults to enabled in
      this launcher.
  GLMRT_CUTE_DSL_LIBDIR
      optional directory containing libcute_dsl_runtime.so. The launcher
      otherwise discovers the CUDA-major-matched runtime in the repo venv.
  GLMRT_REAL_FULL_RETAIN_MTP_QUERY_PROJECTION_GRAPHS
      retain the bounded set of layer-specific M=2..8 recurrent-parity Q
      projection graphs instead of updating one graph exec at every layer;
      defaults to 1 in the runtime. Set 0 only for lifecycle/performance A/B.
  GLMRT_FLASHINFER_WORKSPACE_BASE
      writable base directory for FlashInfer JIT artifacts; defaults to a
      per-UID directory so host and container processes cannot conflict.
  GLMRT_REAL_FULL_SERVE_EXPERT_HOSTS or GLMRT_REAL_FULL_TCP_EXPERT_HOSTS
      explicit owner=host:port list; overrides GLMRT_SPARK_HOSTS/suffix mapping
  GLMRT_REAL_FULL_SERVE_EXPERT_ADDITIONAL_RAILS
      optional verbs-host owner=host:port list for extra coordinator-to-Spark
      QPs; default: none while the current coordinator PCIe link is limiting
  GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS
      minimum logical request rows before striping across rails; default: 512
  GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION
      set 1 to preserve collective ordering when an explicitly multi-rail
      coordinator request is partitioned. Defaults to 1 for custom Spark RDMA.
      Spark-to-Spark reduction rail selection is configured on each Spark.
  GLMRT_SPARK_EXPERT_PORT
      expert TCP port; default: 9100
EOF
  exit 0
fi

addr="${ADDR:-${GLMRT_REAL_FULL_SERVE_ADDR:-0.0.0.0:8000}}"
model_id="${GLMRT_MODEL_ID:-lukealonso/GLM-5.2-NVFP4}"
hosts_csv="${GLMRT_SPARK_HOSTS:-ostrich,dodo,emu,kiwi}"
expert_port="${GLMRT_SPARK_EXPERT_PORT:-9100}"
expert_link_suffix="${GLMRT_REAL_FULL_SERVE_EXPERT_LINK_SUFFIX:-}"
expert_hosts="${GLMRT_REAL_FULL_SERVE_EXPERT_HOSTS:-${GLMRT_REAL_FULL_TCP_EXPERT_HOSTS:-}}"
expert_additional_rails="${GLMRT_REAL_FULL_SERVE_EXPERT_ADDITIONAL_RAILS:-}"
catalog="${CATALOG:-}"
loadplan="${LOADPLAN:-}"
coordinator_transport="${GLMRT_REAL_FULL_SERVE_TRANSPORT:-tcp}"
protocol_v2_verbs_host_execution_lanes="${GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES:-4}"
protocol_v2_verbs_host_shared_cq_harvester="${GLMRT_PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER:-1}"
protocol_v2_verbs_host_stripe_min_rows="${GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS:-512}"
if [ "${GLMRT_EXPERT_INTERMEDIATE_SHARDS+x}" ]; then
  intermediate_shards="$GLMRT_EXPERT_INTERMEDIATE_SHARDS"
elif [ "$coordinator_transport" = "verbs-host" ]; then
  intermediate_shards=4
else
  intermediate_shards=1
fi
if [ "${GLMRT_EXPERT_INTERMEDIATE_REDUCTION+x}" ]; then
  intermediate_reduction="$GLMRT_EXPERT_INTERMEDIATE_REDUCTION"
elif [ "$coordinator_transport" = "verbs-host" ] && [ "$intermediate_shards" = "4" ]; then
  intermediate_reduction=spark-rdma
else
  intermediate_reduction=coordinator
fi
intermediate_reduction_dtype="${GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE:-fp8}"
intermediate_owner_reduction_dtype="${GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE:-bf16}"
intermediate_reduction_min_rows="${GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS:-16}"
intermediate_owner_max_rows="${GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS:-8}"
if [ "${GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION+x}" ]; then
  intermediate_row_sharded_reduction="$GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION"
else
  case "$intermediate_reduction" in
    spark-rdma|spark-rdma-hybrid) intermediate_row_sharded_reduction=1 ;;
    *) intermediate_row_sharded_reduction=0 ;;
  esac
fi
if [ "${GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION+x}" ]; then
  protocol_v2_verbs_host_stripe_spark_reduction="$GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION"
else
  case "$intermediate_reduction" in
    spark-rdma|spark-rdma-hybrid) protocol_v2_verbs_host_stripe_spark_reduction=1 ;;
    *) protocol_v2_verbs_host_stripe_spark_reduction=0 ;;
  esac
fi
kv_cache_dtype="${GLMRT_REAL_FULL_SERVE_KV_CACHE_DTYPE:-fp8}"
max_context_tokens="${GLMRT_REAL_FULL_SERVE_MAX_CONTEXT_TOKENS:-131072}"
native_lib_explicit=0
if [ -n "${GLMRT_NATIVE_LIB:-}" ]; then
  native_lib="$GLMRT_NATIVE_LIB"
  native_lib_explicit=1
elif [ "$coordinator_transport" = "verbs-host" ]; then
  native_lib="$repo_root/native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"
else
  native_lib="$repo_root/native/build-cuda/libglmrt_native.so"
fi
build_profile="${GLMRT_REAL_FULL_SERVE_BUILD_PROFILE:-release}"
bin="${GLMRT_BIN:-$repo_root/rust/target/${build_profile}/glmrt}"
start_experts="${GLMRT_REAL_FULL_SERVE_START_EXPERTS:-1}"
check_experts="${GLMRT_REAL_FULL_SERVE_CHECK_EXPERTS:-1}"
build_daemon="${GLMRT_REAL_FULL_SERVE_BUILD_DAEMON:-1}"
build_native="${GLMRT_REAL_FULL_SERVE_BUILD_NATIVE:-1}"
require_cuda="${GLMRT_REAL_FULL_SERVE_REQUIRE_CUDA:-1}"
shared_cpu_list="${GLMRT_REAL_FULL_SERVE_SHARED_CPU_LIST:-}"
w8a16_q_a="${GLMRT_COORDINATOR_W8A16_Q_A:-1}"
w8a16_q_b="${GLMRT_COORDINATOR_W8A16_Q_B:-1}"
w8a16_o_proj="${GLMRT_COORDINATOR_W8A16_O_PROJ:-1}"
w8a16_packed_o="${GLMRT_COORDINATOR_W8A16_PACKED_O:-1}"
w8a16_async_attention="${GLMRT_COORDINATOR_W8A16_ASYNC_ATTENTION:-1}"
w8a16_aot="${GLMRT_W8A16_AOT:-OFF}"
for projection_flag in "$w8a16_q_a" "$w8a16_q_b" "$w8a16_o_proj"; do
  case "${projection_flag,,}" in
    1|true|yes|on|w8a16) w8a16_aot=ON ;;
  esac
done
fast_token="${GLMRT_REAL_FULL_SERVE_FAST_TOKEN:-0}"
fast_token_lm_head_rows="${GLMRT_REAL_FULL_SERVE_FAST_TOKEN_LM_HEAD_ROWS:-1024}"
warmup_experts="${GLMRT_REAL_FULL_SERVE_WARMUP_EXPERTS:-1}"
warmup_layer_id="${GLMRT_REAL_FULL_SERVE_WARMUP_LAYER_ID:-${GLMRT_PHASE0_TCP_LAYER_ID:-3}}"
warmup_iterations="${GLMRT_REAL_FULL_SERVE_WARMUP_ITERATIONS:-1}"
warmup_rows="${GLMRT_REAL_FULL_SERVE_WARMUP_ROWS:-1}"
warmup_timeout_ms="${GLMRT_REAL_FULL_SERVE_WARMUP_TIMEOUT_MS:-120000}"
warmup_measured_timeout_ms="${GLMRT_REAL_FULL_SERVE_WARMUP_MEASURED_TIMEOUT_MS:-5000}"
warmup_roundtrip_rows="${GLMRT_REAL_FULL_SERVE_WARMUP_ROUNDTRIP_ROWS:-1}"
warmup_mtp_chain_rows="${GLMRT_REAL_FULL_SERVE_WARMUP_MTP_CHAIN_ROWS:-2}"
warmup_prefill_roundtrip_rows="${GLMRT_REAL_FULL_SERVE_WARMUP_PREFILL_ROUNDTRIP_ROWS:-16,256,512}"
warmup_prefill_chain_rows="${GLMRT_REAL_FULL_SERVE_WARMUP_PREFILL_CHAIN_ROWS:-16,256,512}"
warmup_expected_executor="${GLMRT_REAL_FULL_SERVE_WARMUP_EXPECTED_EXECUTOR:-protocol-v2-real-nvfp4-checkpoint-executor}"
expert_ready_timeout_secs="${GLMRT_REAL_FULL_SERVE_EXPERT_READY_TIMEOUT_SECS:-900}"
expert_warmup_status_file="${GLMRT_REAL_FULL_SERVE_EXPERT_WARMUP_STATUS_FILE:-}"
prewarm_request="${GLMRT_REAL_FULL_SERVE_PREWARM_REQUEST:-1}"
protocol_v2_timeout_ms="${GLMRT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS:-120000}"
mtp="${GLMRT_REAL_FULL_MTP:-0}"
mtp_draft_tokens="${GLMRT_REAL_FULL_MTP_DRAFT_TOKENS:-}"
mtp_min_d="${GLMRT_REAL_FULL_MTP_MIN_D:-}"
mtp_max_d="${GLMRT_REAL_FULL_MTP_MAX_D:-}"

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command not found: $1" >&2
    exit 2
  fi
}

for affinity_list in \
  "$shared_cpu_list" \
  "${GLMRT_REAL_FULL_REQUEST_WORKER_CPUS:-}" \
  "${GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU:-}"; do
  if [ -n "$affinity_list" ]; then
    need taskset
    if ! taskset -c "$affinity_list" true; then
      echo "invalid or unavailable CPU affinity list: $affinity_list" >&2
      exit 2
    fi
  fi
done

configure_host_python() {
  local configured_python="${GLMRT_PYTHON:-}"
  if [ -z "$configured_python" ] && [ -x "$repo_root/.venv/bin/python" ]; then
    configured_python="$repo_root/.venv/bin/python"
  fi
  if [ -z "$configured_python" ]; then
    configured_python="$(command -v python3 || true)"
  fi
  if [ -z "$configured_python" ] || [ ! -x "$configured_python" ]; then
    echo "Python interpreter not found; set GLMRT_PYTHON" >&2
    exit 2
  fi
  configured_python="$(realpath -s "$configured_python")"
  export GLMRT_PYTHON="$configured_python"
  export PYO3_PYTHON="${PYO3_PYTHON:-$configured_python}"

  local python_config
  python_config="$("$configured_python" - <<'PY'
import os
import sys
import sysconfig
from pathlib import Path

module_paths = []
for path in sys.path:
    if path and "site-packages" in path and path not in module_paths:
        module_paths.append(path)
nccl_root = next(
    (
        Path(path) / "nvidia" / "nccl"
        for path in module_paths
        if (Path(path) / "nvidia" / "nccl" / "include" / "nccl.h").is_file()
    ),
    None,
)
print(sys.base_prefix)
print(sysconfig.get_config_var("LIBDIR") or "")
print(os.pathsep.join(module_paths))
print(nccl_root or "")
PY
)"
  local python_base
  local python_sysconfig_lib
  local python_module_path
  local python_nccl_root
  python_base="$(sed -n '1p' <<<"$python_config")"
  python_sysconfig_lib="$(sed -n '2p' <<<"$python_config")"
  python_module_path="$(sed -n '3p' <<<"$python_config")"
  python_nccl_root="$(sed -n '4p' <<<"$python_config")"
  host_python_home="$python_base"
  host_python_module_path="$python_module_path"
  host_nccl_include_dir="${GLMRT_NCCL_INCLUDE_DIR:-}"
  host_nccl_library="${GLMRT_NCCL_LIBRARY:-}"

  local python_libs=()
  if [ -n "${GLMRT_PYTHON_LIBDIR:-}" ]; then
    python_libs+=("$GLMRT_PYTHON_LIBDIR")
  fi
  if [ -n "$python_sysconfig_lib" ]; then
    python_libs+=("$python_sysconfig_lib")
  fi
  if [ -n "$python_nccl_root" ]; then
    python_libs+=("$python_nccl_root/lib")
    if [ -z "$host_nccl_include_dir" ]; then
      host_nccl_include_dir="$python_nccl_root/include"
    fi
    if [ -z "$host_nccl_library" ]; then
      local nccl_library
      for nccl_library in "$python_nccl_root"/lib/libnccl.so*; do
        if [ -f "$nccl_library" ]; then
          host_nccl_library="$nccl_library"
          break
        fi
      done
    fi
  fi
  if [ -n "${GLMRT_CUTE_DSL_LIBDIR:-}" ]; then
    python_libs+=("$GLMRT_CUTE_DSL_LIBDIR")
  elif command -v nvcc >/dev/null 2>&1; then
    local cuda_toolkit_major
    cuda_toolkit_major="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\).*/\1/p' | head -n1)"
    local cute_dsl_lib
    for cute_dsl_lib in \
      "$repo_root"/.venv/lib/python*/site-packages/nvidia_cutlass_dsl/"cu${cuda_toolkit_major}"/lib; do
      if [ -n "$cuda_toolkit_major" ] && [ -d "$cute_dsl_lib" ]; then
        python_libs+=("$cute_dsl_lib")
        break
      fi
    done
  fi
  if [ "${#python_libs[@]}" -gt 0 ]; then
    local python_path
    python_path="$(IFS=:; printf '%s' "${python_libs[*]}")"
    export LD_LIBRARY_PATH="$python_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
}

configure_pinned_sparkinfer() {
  local sparkinfer_source="$repo_root/third_party/sparkinfer"
  local sparkinfer_lock="$repo_root/third_party/sparkinfer.lock.json"
  local python_paths=("$sparkinfer_source")
  if [ -n "$host_python_module_path" ]; then
    python_paths+=("$host_python_module_path")
  fi
  if [ -n "${PYTHONPATH:-}" ]; then
    python_paths+=("$PYTHONPATH")
  fi
  export PYTHONPATH="$(IFS=:; printf '%s' "${python_paths[*]}")"
  "$GLMRT_PYTHON" "$repo_root/scripts/verify-sparkinfer-source.py" \
    --source "$sparkinfer_source" \
    --lock "$sparkinfer_lock" \
    --assert-import-source
}

require_bool_flag() {
  local name="$1"
  local value="$2"
  case "$value" in
    0|1) ;;
    *)
      echo "$name must be 0 or 1" >&2
      exit 2
      ;;
  esac
}

target_addr() {
  local entry="$1"
  if [[ "$entry" == *=* ]]; then
    echo "${entry#*=}"
  else
    echo "$entry"
  fi
}

target_host() {
  local addr_part="$1"
  echo "${addr_part%:*}"
}

target_owner() {
  local entry="$1"
  if [[ "$entry" == *=* ]]; then
    echo "${entry%%=*}"
    return
  fi
  local addr_part
  addr_part="$(target_addr "$entry")"
  local host_part
  host_part="$(target_host "$addr_part")"
  if [ -n "$expert_link_suffix" ] && [[ "$host_part" == *"$expert_link_suffix" ]]; then
    echo "${host_part%"$expert_link_suffix"}"
  else
    echo "$host_part"
  fi
}

target_port() {
  local addr_part="$1"
  if [[ "$addr_part" == *:* ]]; then
    echo "${addr_part##*:}"
  else
    echo "$expert_port"
  fi
}

spark_primary_rail_ip() {
  case "$1" in
    ostrich) echo "10.55.0.1" ;;
    dodo) echo "10.55.0.2" ;;
    emu) echo "10.55.0.3" ;;
    kiwi) echo "10.55.0.4" ;;
    *) return 1 ;;
  esac
}

wait_for_tcp() {
  local host="$1"
  local port="$2"
  local label="$3"
  local attempts="${4:-30}"
  for _ in $(seq 1 "$attempts"); do
    if timeout 2 bash -c ":</dev/tcp/${host}/${port}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "$label did not become reachable at ${host}:${port}" >&2
  return 1
}

warmup_expert_ids_for_owner() {
  local owner="$1"
  local count="$2"
  if [ -z "$loadplan" ]; then
    case "$owner" in
      spark-[0-3])
        local owner_index="${owner#spark-}"
        local -a expert_ids=()
        local expert_id="$owner_index"
        while [ "${#expert_ids[@]}" -lt "$count" ]; do
          if [ "$expert_id" -ge 256 ]; then
            echo "cannot infer ${count} warmup experts for runtime role ${owner}" >&2
            exit 2
          fi
          expert_ids+=("$expert_id")
          expert_id=$((expert_id + 4))
        done
        (IFS=,; echo "${expert_ids[*]}")
        return
        ;;
      *)
        echo "cannot infer a warmup expert for runtime role ${owner}" >&2
        exit 2
        ;;
    esac
  fi
  local expert_ids
  if ! expert_ids="$(
    jq -er \
      --arg owner "$owner" \
      --argjson layer "$warmup_layer_id" \
      --argjson count "$count" '
      [
        .assignments[]
        | select(
            .owner == $owner
            and .layer_id == $layer
            and .expert_id != null
            and (
              (.tensor_name | endswith(".gate_proj.weight"))
              or (.tensor_name | endswith(".gate_proj.trellis"))
            )
          )
        | .expert_id
      ]
      | unique
      | sort
      | select(length >= $count)
      | .[:$count]
      | map(tostring)
      | join(",")
    ' "$loadplan"
  )"; then
    echo "fewer than ${count} owned layer ${warmup_layer_id} gate projection experts found for ${owner} in ${loadplan}" >&2
    exit 2
  fi
  echo "$expert_ids"
}

warmup_protocol_v2_experts() {
  if [ "$warmup_experts" != "1" ]; then
    echo "== skipping binary ProtocolV2 expert precompile warmup ==" >&2
    return
  fi

  local wire_contract="bf16-in/bf16-out"
  local warmup_routes_per_row=1
  local -a wire_args=()
  case "$model_id" in
  wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1|wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1)
    # The native EXL3 route is a fused top-k=8 NVFP4-ingress kernel. A
    # legacy single-route BF16 probe falls through to the W4A16 projection
    # loader and asks the EXL3 catalog for a nonexistent `.weight` tensor.
    # Warm the exact low-precision production shape instead.
    wire_contract="nvfp4-in/fp8-out"
    warmup_routes_per_row=8
    wire_args=(--nvfp4-fp8-roundtrip)
    ;;
  esac

  echo "== warming Spark experts with binary ProtocolV2 precompile frames transport=${coordinator_transport} wire=${wire_contract} ==" >&2
  IFS=',' read -r -a entries <<< "$expert_hosts"
  for entry in "${entries[@]}"; do
    local owner
    local addr_part
    local expert_id
    local expert_ids
    owner="$(target_owner "$entry")"
    addr_part="$(target_addr "$entry")"
    if [[ "$addr_part" != *:* ]]; then
      addr_part="${addr_part}:${expert_port}"
    fi
    # Distinct owner-valid IDs exercise the production top-k metadata shape
    # while remaining valid for both role-filtered and TP-sharded experts.
    expert_ids="$(warmup_expert_ids_for_owner "$owner" "$warmup_routes_per_row")"
    expert_id="${expert_ids%%,*}"
    echo "protocol_v2_warmup owner=${owner} addr=${addr_part} layer_id=${warmup_layer_id} expert_id=${expert_id} expert_ids=${expert_ids} routes_per_row=${warmup_routes_per_row} iterations=${warmup_iterations} timeout_ms=${warmup_timeout_ms}" >&2
    "$bin" bench-protocol-v2-tcp \
      --addr "$addr_part" \
      --transport "$coordinator_transport" \
      --target "real-full-serve-warmup-${owner}" \
      --hops 1 \
      --iterations 1 \
      --large-iterations 1 \
      --warmup-iterations "$warmup_iterations" \
      --warmup-rows "$warmup_rows" \
      --warmup-timeout-ms "$warmup_timeout_ms" \
      --warmup-only \
      --roundtrip-rows "$warmup_roundtrip_rows" \
      --mtp-chain-rows "$warmup_mtp_chain_rows" \
      --prefill-roundtrip-rows "$warmup_prefill_roundtrip_rows" \
      --prefill-chain-rows "$warmup_prefill_chain_rows" \
      --layer-id "$warmup_layer_id" \
      --expert-id "$expert_id" \
      --expert-ids "$expert_ids" \
      --routes-per-row "$warmup_routes_per_row" \
      --expected-executor "$warmup_expected_executor" \
      --require-expected-executor \
      "${wire_args[@]}" \
      --timeout-ms "$warmup_measured_timeout_ms" >/dev/null
  done
}

wait_for_protocol_v2_experts() {
  echo "== waiting for Spark expert control planes ==" >&2
  IFS=',' read -r -a entries <<< "$expert_hosts"
  for entry in "${entries[@]}"; do
    local addr_part
    addr_part="$(target_addr "$entry")"
    wait_for_tcp \
      "$(target_host "$addr_part")" \
      "$(target_port "$addr_part")" \
      "expert target ${entry}" \
      "$expert_ready_timeout_secs"
  done
}

require_bool_flag GLMRT_REAL_FULL_SERVE_START_EXPERTS "$start_experts"
require_bool_flag GLMRT_REAL_FULL_SERVE_CHECK_EXPERTS "$check_experts"
require_bool_flag GLMRT_REAL_FULL_SERVE_BUILD_DAEMON "$build_daemon"
require_bool_flag GLMRT_REAL_FULL_SERVE_BUILD_NATIVE "$build_native"
require_bool_flag GLMRT_REAL_FULL_SERVE_REQUIRE_CUDA "$require_cuda"
require_bool_flag GLMRT_REAL_FULL_SERVE_FAST_TOKEN "$fast_token"
require_bool_flag GLMRT_REAL_FULL_SERVE_WARMUP_EXPERTS "$warmup_experts"
require_bool_flag GLMRT_REAL_FULL_SERVE_PREWARM_REQUEST "$prewarm_request"
require_bool_flag GLMRT_REAL_FULL_MTP "$mtp"
case "$coordinator_transport" in
  tcp|verbs-host) ;;
  *)
    echo "GLMRT_REAL_FULL_SERVE_TRANSPORT must be tcp or verbs-host" >&2
    exit 2
    ;;
esac
case "$intermediate_shards" in
  1|4) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_SHARDS must be 1 or 4" >&2
    exit 2
    ;;
esac
case "$intermediate_reduction" in
  coordinator|spark|spark-owner|spark-hybrid|spark-rdma|spark-rdma-hybrid) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION must be coordinator, spark, spark-owner, spark-hybrid, spark-rdma, or spark-rdma-hybrid" >&2
    exit 2
    ;;
esac
if [ "$intermediate_reduction" != "coordinator" ] && [ "$intermediate_shards" != "4" ]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires GLMRT_EXPERT_INTERMEDIATE_SHARDS=4" >&2
  exit 2
fi
case "$intermediate_reduction" in
  spark-owner|spark-hybrid|spark-rdma|spark-rdma-hybrid)
    if [ "$coordinator_transport" != "verbs-host" ]; then
      echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires verbs-host serving" >&2
      exit 2
    fi
    ;;
esac
case "$intermediate_reduction_dtype" in
  bf16|fp8|nvfp4) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE must be bf16, fp8, or nvfp4" >&2
    exit 2
    ;;
esac
case "$intermediate_owner_reduction_dtype" in
  bf16|fp8|nvfp4) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE must be bf16, fp8, or nvfp4" >&2
    exit 2
    ;;
esac
case "$intermediate_row_sharded_reduction" in
  0|1) ;;
  *)
    echo "GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION must be 0 or 1" >&2
    exit 2
    ;;
esac
case "$intermediate_reduction" in
  spark-rdma|spark-rdma-hybrid)
    if [ "$intermediate_row_sharded_reduction" != "1" ]; then
      echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION=$intermediate_reduction requires row-sharded reduction" >&2
      exit 2
    fi
    ;;
esac
if ! [[ "$intermediate_reduction_min_rows" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$intermediate_owner_max_rows" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS must be a positive integer" >&2
  exit 2
fi
export GLMRT_EXPERT_INTERMEDIATE_SHARDS="$intermediate_shards"
export GLMRT_EXPERT_INTERMEDIATE_REDUCTION="$intermediate_reduction"
export GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE="$intermediate_reduction_dtype"
export GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE="$intermediate_owner_reduction_dtype"
export GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS="$intermediate_reduction_min_rows"
export GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS="$intermediate_owner_max_rows"
export GLMRT_EXPERT_INTERMEDIATE_ROW_SHARDED_REDUCTION="$intermediate_row_sharded_reduction"
if ! [[ "$protocol_v2_verbs_host_execution_lanes" =~ ^[1-8]$ ]]; then
  echo "GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES must be an integer in 1..8" >&2
  exit 2
fi
export GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES="$protocol_v2_verbs_host_execution_lanes"
case "$protocol_v2_verbs_host_shared_cq_harvester" in
  0|1) ;;
  *)
    echo "GLMRT_PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER must be 0 or 1" >&2
    exit 2
    ;;
esac
export GLMRT_PROTOCOL_V2_VERBS_HOST_SHARED_CQ_HARVESTER="$protocol_v2_verbs_host_shared_cq_harvester"
if ! [[ "$protocol_v2_verbs_host_stripe_min_rows" =~ ^[1-9][0-9]*$ ]]; then
  echo "GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS must be a positive integer" >&2
  exit 2
fi
export GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_MIN_ROWS="$protocol_v2_verbs_host_stripe_min_rows"
case "$protocol_v2_verbs_host_stripe_spark_reduction" in
  0|1) ;;
  *)
    echo "GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION must be 0 or 1" >&2
    exit 2
    ;;
esac
export GLMRT_PROTOCOL_V2_VERBS_HOST_STRIPE_SPARK_REDUCTION="$protocol_v2_verbs_host_stripe_spark_reduction"
case "$build_profile" in
  debug|release) ;;
  *)
    echo "GLMRT_REAL_FULL_SERVE_BUILD_PROFILE must be debug or release" >&2
    exit 2
    ;;
esac
need xargs
need jq
if [ "$check_experts" = "1" ]; then
  need timeout
fi
if [ "$warmup_experts" = "1" ]; then
  need jq
  need timeout
fi

if ! [[ "$expert_port" =~ ^[0-9]+$ ]] || [ "$expert_port" -lt 1 ] || [ "$expert_port" -gt 65535 ]; then
  echo "GLMRT_SPARK_EXPERT_PORT must be an integer in 1..65535" >&2
  exit 2
fi
if ! [[ "$max_context_tokens" =~ ^[0-9]+$ ]] || [ "$max_context_tokens" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_MAX_CONTEXT_TOKENS must be a positive integer" >&2
  exit 2
fi
for draft_setting in \
  "GLMRT_REAL_FULL_MTP_DRAFT_TOKENS:$mtp_draft_tokens" \
  "GLMRT_REAL_FULL_MTP_MIN_D:$mtp_min_d" \
  "GLMRT_REAL_FULL_MTP_MAX_D:$mtp_max_d"; do
  draft_name="${draft_setting%%:*}"
  draft_value="${draft_setting#*:}"
  if [ -n "$draft_value" ] && { ! [[ "$draft_value" =~ ^[0-9]+$ ]] || [ "$draft_value" -lt 1 ] || [ "$draft_value" -gt 8 ]; }; then
    echo "$draft_name must be an integer in 1..8" >&2
    exit 2
  fi
done
if [ -n "$mtp_min_d" ] && [ -n "$mtp_max_d" ] && [ "$mtp_min_d" -gt "$mtp_max_d" ]; then
  echo "GLMRT_REAL_FULL_MTP_MIN_D must not exceed GLMRT_REAL_FULL_MTP_MAX_D" >&2
  exit 2
fi
if ! [[ "$warmup_layer_id" =~ ^[0-9]+$ ]]; then
  echo "GLMRT_REAL_FULL_SERVE_WARMUP_LAYER_ID must be a non-negative integer" >&2
  exit 2
fi
if ! [[ "$warmup_iterations" =~ ^[0-9]+$ ]] || [ "$warmup_iterations" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_WARMUP_ITERATIONS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_rows" =~ ^[0-9]+$ ]] || [ "$warmup_rows" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_WARMUP_ROWS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_timeout_ms" =~ ^[0-9]+$ ]] || [ "$warmup_timeout_ms" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_WARMUP_TIMEOUT_MS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$expert_ready_timeout_secs" =~ ^[0-9]+$ ]] || [ "$expert_ready_timeout_secs" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_EXPERT_READY_TIMEOUT_SECS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$warmup_measured_timeout_ms" =~ ^[0-9]+$ ]] || [ "$warmup_measured_timeout_ms" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_WARMUP_MEASURED_TIMEOUT_MS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$protocol_v2_timeout_ms" =~ ^[0-9]+$ ]] || [ "$protocol_v2_timeout_ms" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS must be a positive integer" >&2
  exit 2
fi
if ! [[ "$fast_token_lm_head_rows" =~ ^[0-9]+$ ]] || [ "$fast_token_lm_head_rows" -lt 1 ]; then
  echo "GLMRT_REAL_FULL_SERVE_FAST_TOKEN_LM_HEAD_ROWS must be a positive integer" >&2
  exit 2
fi

if [ -n "$catalog" ]; then
  test -f "$catalog" || {
    echo "catalog not found: $catalog" >&2
    exit 2
  }
  catalog_model_id="$(jq -er '.model_id' "$catalog")" || {
    echo "catalog has no valid model_id: $catalog" >&2
    exit 2
  }
  [ "$catalog_model_id" = "$model_id" ] || {
    echo "catalog model ${catalog_model_id} does not match requested ${model_id}" >&2
    exit 2
  }
fi
if [ -n "$loadplan" ]; then
  test -f "$loadplan" || {
    echo "loadplan not found: $loadplan" >&2
    exit 2
  }
  loadplan_model_id="$(jq -er '.model_id' "$loadplan")" || {
    echo "loadplan has no valid model_id: $loadplan" >&2
    exit 2
  }
  [ "$loadplan_model_id" = "$model_id" ] || {
    echo "loadplan model ${loadplan_model_id} does not match requested ${model_id}" >&2
    exit 2
  }
fi

report_shell_startup_phase configuration
configure_host_python
report_shell_startup_phase host-python
configure_pinned_sparkinfer
report_shell_startup_phase sparkinfer-source-verification

if [ "$require_cuda" = "1" ]; then
  if [ "$build_native" = "1" ] && [ "$native_lib_explicit" = "0" ]; then
    nccl_cmake_args=()
    if [ -n "$host_nccl_include_dir" ] && [ -n "$host_nccl_library" ]; then
      nccl_cmake_args+=(
        "-DGLMRT_NCCL_INCLUDE_DIR=$host_nccl_include_dir"
        "-DGLMRT_NCCL_LIBRARY=$host_nccl_library"
      )
    fi
    native_rdma=OFF
    if [ "$coordinator_transport" = "verbs-host" ]; then
      native_rdma=ON
    fi
    cmake -S native -B "$(dirname "$native_lib")" -G Ninja \
      -U GLMRT_ENABLE_B12X_AOT \
      -U GLMRT_ENABLE_B12X_COORDINATOR_AOT \
      -DGLMRT_ENABLE_CUDA=ON \
      -DGLMRT_ENABLE_RDMA="$native_rdma" \
      -DGLMRT_ENABLE_SPARKINFER_AOT=OFF \
      -DGLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT="${GLMRT_SPARKINFER_COORDINATOR_AOT:-${GLMRT_B12X_COORDINATOR_AOT:-ON}}" \
      -DGLMRT_SPARKINFER_SOURCE_DIR="$repo_root/third_party/sparkinfer" \
      -DGLMRT_SPARKINFER_LOCK_FILE="$repo_root/third_party/sparkinfer.lock.json" \
      -DGLMRT_ENABLE_W8A16_AOT="$w8a16_aot" \
      -DGLMRT_ENABLE_NCCL=ON \
      "${nccl_cmake_args[@]}" \
      -DPython3_EXECUTABLE="$GLMRT_PYTHON" \
      -DGLMRT_CUDA_ARCHITECTURES="${GLMRT_CUDA_ARCH:-120}"
    cmake --build "$(dirname "$native_lib")"
  fi
  test -f "$native_lib" || {
    cat >&2 <<EOF
CUDA native library not found: $native_lib
Build it with:
  cmake -S native -B native/build-cuda -G Ninja -DGLMRT_ENABLE_CUDA=ON -DGLMRT_ENABLE_RDMA=OFF -DGLMRT_ENABLE_NCCL=ON -DGLMRT_CUDA_ARCHITECTURES=120
  cmake --build native/build-cuda
For verbs-host, build the RDMA-enabled native library:
  cmake -S native -B native/build-cuda-rdma -G Ninja -DGLMRT_ENABLE_CUDA=ON -DGLMRT_ENABLE_RDMA=ON -DGLMRT_ENABLE_NCCL=ON -DGLMRT_CUDA_ARCHITECTURES=120
  cmake --build native/build-cuda-rdma
or set GLMRT_REAL_FULL_SERVE_REQUIRE_CUDA=0 for a diagnostic-only start.
EOF
    exit 2
  }
  export GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS="${GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS:-1}"
  export GLMRT_NATIVE_LIB="$native_lib"
fi
export GLMRT_REAL_FULL_SERVE_FAST_TOKEN="$fast_token"
export GLMRT_REAL_FULL_SERVE_FAST_TOKEN_LM_HEAD_ROWS="$fast_token_lm_head_rows"
export GLMRT_REAL_FULL_SERVE_PREWARM_REQUEST="$prewarm_request"
export GLMRT_REAL_FULL_MTP="$mtp"
if [ -n "$mtp_draft_tokens" ]; then
  export GLMRT_REAL_FULL_MTP_DRAFT_TOKENS="$mtp_draft_tokens"
fi
if [ -n "$mtp_min_d" ]; then
  export GLMRT_REAL_FULL_MTP_MIN_D="$mtp_min_d"
fi
if [ -n "$mtp_max_d" ]; then
  export GLMRT_REAL_FULL_MTP_MAX_D="$mtp_max_d"
fi
export GLMRT_REAL_FULL_PROTOCOL_V2_TIMEOUT_MS="$protocol_v2_timeout_ms"
export GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS="${GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_GRAPHS:-1}"
export GLMRT_B12X="${GLMRT_B12X:-1}"
export GLMRT_COORDINATOR_W8A16_Q_A="$w8a16_q_a"
export GLMRT_COORDINATOR_W8A16_Q_B="$w8a16_q_b"
export GLMRT_COORDINATOR_W8A16_O_PROJ="$w8a16_o_proj"
export GLMRT_COORDINATOR_W8A16_PACKED_O="$w8a16_packed_o"
export GLMRT_COORDINATOR_W8A16_ASYNC_ATTENTION="$w8a16_async_attention"
export GLMRT_REAL_FULL_REQUEST_THREAD_PINNED="${GLMRT_REAL_FULL_REQUEST_THREAD_PINNED:-$GLMRT_B12X}"
export GLMRT_REAL_FULL_REQUEST_THREAD_PINNED_WORKERS="${GLMRT_REAL_FULL_REQUEST_THREAD_PINNED_WORKERS:-1}"
export GLMRT_ENGINE_COMMIT="${GLMRT_ENGINE_COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
kernel_cache_base="${GLMRT_KERNEL_CACHE_BASE:-${GLMRT_FLASHINFER_WORKSPACE_BASE:-${FLASHINFER_WORKSPACE_BASE:-$HOME/.cache/glmrt/kernel-cache-${UID}}}}"
kernel_cache_environment_id="${GLMRT_KERNEL_CACHE_ENVIRONMENT_ID:-}"
if [ -z "$kernel_cache_environment_id" ]; then
  kernel_cache_environment_id="host-$(uname -srm)-$(sha256sum "$native_lib" | awk '{print $1}')"
fi
mkdir -p "$kernel_cache_base"
kernel_cache_identity="$("$GLMRT_PYTHON" "$repo_root/scripts/kernel-cache-identity.py" \
  --cache-root "$kernel_cache_base" \
  --environment-id "$kernel_cache_environment_id")"
kernel_cache_identity_root="$kernel_cache_base/$kernel_cache_identity"
export FLASHINFER_WORKSPACE_BASE="$kernel_cache_identity_root/flashinfer"
export B12X_COMPILE_CACHE_DIR="${GLMRT_SPARKINFER_COMPILE_CACHE_DIR:-$kernel_cache_identity_root/sparkinfer/compile}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE" "$B12X_COMPILE_CACHE_DIR"
echo "kernel_cache_identity identity=$kernel_cache_identity base=$kernel_cache_base flashinfer=$FLASHINFER_WORKSPACE_BASE sparkinfer=$B12X_COMPILE_CACHE_DIR"
report_shell_startup_phase kernel-cache-identity

if [ -z "$expert_hosts" ]; then
  IFS=',' read -r -a hosts <<< "$hosts_csv"
  targets=()
  for host_index in "${!hosts[@]}"; do
    host="${hosts[$host_index]}"
    host="$(echo "$host" | xargs)"
    [ -n "$host" ] || continue
    if [ "$coordinator_transport" = "verbs-host" ] && primary_ip="$(spark_primary_rail_ip "$host")"; then
      targets+=("spark-${host_index}=${primary_ip}:${expert_port}")
    else
      targets+=("spark-${host_index}=${host}${expert_link_suffix}:${expert_port}")
    fi
  done
  expert_hosts="$(IFS=,; echo "${targets[*]}")"
fi

if [ -n "$expert_additional_rails" ]; then
  export GLMRT_PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS="$expert_additional_rails"
  coordinator_request_rails=explicit-multi
  echo "== verbs-host additional rails: $expert_additional_rails ==" >&2
else
  coordinator_request_rails=primary-only
  unset GLMRT_PROTOCOL_V2_VERBS_HOST_ADDITIONAL_RAILS || true
fi

if [ -z "$expert_hosts" ]; then
  echo "no expert targets configured; set GLMRT_SPARK_HOSTS or GLMRT_REAL_FULL_TCP_EXPERT_HOSTS" >&2
  exit 2
fi

if [ "$start_experts" = "1" ]; then
  echo "== starting all-layer real NVFP4 Spark expert daemons transport=${coordinator_transport} ==" >&2
  GLMRT_SPARK_HOSTS="$hosts_csv" \
    GLMRT_SPARK_KEEP_EXPERTS=1 \
    GLMRT_SPARK_EXPERT_REAL_LAYER="${GLMRT_SPARK_EXPERT_REAL_LAYER:-all}" \
    GLMRT_SPARK_EXPERT_TRANSPORT="$coordinator_transport" \
    GLMRT_PHASE0_SPARK_SKIP_BENCH=1 \
    GLMRT_PHASE0_SPARK_EXPERT_MODE=real \
    scripts/phase0-spark-tcp-bench.sh
fi
report_shell_startup_phase expert-launch

if [ "$check_experts" = "1" ] && [ "$coordinator_transport" != "verbs-host" ]; then
  echo "== checking expert TCP targets ==" >&2
  IFS=',' read -r -a entries <<< "$expert_hosts"
  for entry in "${entries[@]}"; do
    addr_part="$(target_addr "$entry")"
    wait_for_tcp "$(target_host "$addr_part")" "$(target_port "$addr_part")" "expert target ${entry}" 10
  done
elif [ "$check_experts" = "1" ]; then
  echo "== skipping raw TCP probes for verbs-host expert targets ==" >&2
fi

if [ "$build_daemon" = "1" ]; then
  echo "== building glmrt daemon (${build_profile}) ==" >&2
  need cargo
  cargo_profile_args=()
  if [ "$build_profile" = "release" ]; then
    cargo_profile_args+=(--release)
  fi
  cargo build --manifest-path rust/Cargo.toml -p glmrt-daemon "${cargo_profile_args[@]}"
fi
report_shell_startup_phase daemon-build

if [ ! -x "$bin" ]; then
  bin="$repo_root/scripts/glmrt"
fi

export PYTHONHOME="${PYTHONHOME:-$host_python_home}"

if [ -n "$expert_warmup_status_file" ]; then
  status_tmp="${expert_warmup_status_file}.tmp"
  rm -f "$expert_warmup_status_file" "$status_tmp"
  (
    if wait_for_protocol_v2_experts && warmup_protocol_v2_experts; then
      printf 'ready\n' >"$status_tmp"
    else
      status=$?
      printf 'failed exit_status=%s\n' "$status" >"$status_tmp"
    fi
    mv -f "$status_tmp" "$expert_warmup_status_file"
  ) &
else
  warmup_protocol_v2_experts
fi
report_shell_startup_phase expert-warmup-dispatch

echo "== starting real-full ${coordinator_transport} API coordinator ==" >&2
echo "listen=${addr}" >&2
echo "expert_hosts=${expert_hosts}" >&2
echo "intermediate_shards=${intermediate_shards} intermediate_reduction=${intermediate_reduction} reduction_dtype=${intermediate_reduction_dtype} owner_reduction_dtype=${intermediate_owner_reduction_dtype} reduction_min_rows=${intermediate_reduction_min_rows} owner_max_rows=${intermediate_owner_max_rows} row_sharded=${intermediate_row_sharded_reduction} coordinator_request_rails=${coordinator_request_rails} request_striped_reduction=${protocol_v2_verbs_host_stripe_spark_reduction} stripe_min_rows=${protocol_v2_verbs_host_stripe_min_rows}" >&2
coordinator_command=(
  "$bin" coordinator
  --backend real-glm-full
  --transport "$coordinator_transport"
  --kv-cache-dtype "$kv_cache_dtype"
  --max-context-tokens "$max_context_tokens"
  --listen "$addr"
  --model-id "$model_id"
  --expert-hosts "$expert_hosts"
)
[ -z "$catalog" ] || coordinator_command+=(--catalog "$catalog")
[ -z "$loadplan" ] || coordinator_command+=(--loadplan "$loadplan")
if [ -n "$shared_cpu_list" ] \
  || [ -n "${GLMRT_REAL_FULL_REQUEST_WORKER_CPUS:-}" ] \
  || [ -n "${GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU:-}" ]; then
  echo "shared_cpu_list=${shared_cpu_list:-none} request_worker_cpus=${GLMRT_REAL_FULL_REQUEST_WORKER_CPUS:-none} scheduler_worker_cpu=${GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU:-none}" >&2
fi
if [ -n "$shared_cpu_list" ]; then
  report_shell_startup_phase coordinator-exec
  exec taskset -c "$shared_cpu_list" "${coordinator_command[@]}"
fi
report_shell_startup_phase coordinator-exec
exec "${coordinator_command[@]}"
