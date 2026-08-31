use crate::commands::real_full::dflash::{
    dflash2_serving_requested, GLM53_DFLASH2_TARGET_CAPTURE_TAPS,
};
use crate::commands::real_full::dspark::dspark_target_hidden_tap_layer_ids;
use anyhow::{Context, Result};
use glmrt_core::{
    CompletionRoutePlanEntry, DType, ExpertBatch, ExpertBatchRoute, ExpertHostBatchSetAccumulation,
    ExpertOwnerLookup, GraphBucket, LayerId, LayerWave, ModelFacts,
    RollingExpertRowPackAccumulator, RollingExpertRowPackConfig, RollingExpertRowPackEmission,
    RowSource, RowSourceKind, TensorCatalog, TensorInfo, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS, GLM52_ROUTED_EXPERTS, GLM52_TOP_K,
    GLM52_TOTAL_LAYERS_WITH_MTP,
};
use glmrt_ffi::GlmrtDeviceBuffer;
use glmrt_loader::read_tensor_bytes_into;
use glmrt_transport::{
    expert_protocol_v2_compact_id, ExpertProtocolV2Request, ExpertProtocolV2RouteEntry,
    ExpertProtocolV2Status, ExpertV2Dtype, TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    TcpProtocolV2HostBatchSetDispatch, TcpProtocolV2HostBatchSetDispatchStats,
    TcpProtocolV2HostBatchSetPersistentClient, TcpProtocolV2HostBatchTarget,
    VerbsHostProtocolV2HostBatchSetBf16PayloadChunk,
    VerbsHostProtocolV2HostBatchSetPersistentClient, VerbsHostProtocolV2PersistentClient,
    VerbsHostProtocolV2ReducedIdentityPayloadPending,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    mpsc, Arc, Mutex, OnceLock,
};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crate::commands::real_full::coordinator_kernels::{
    begin_quantize_device_bf16_to_nvfp4_row_payload, concat_device_bf16_row_batches,
    concat_device_bf16_row_slices_async, coordinator_cuda_reference_kernels_enabled,
    cuda_native_library, cuda_stream_sparse_b_scatter_shared_residual_add_bf16_device_outputs,
    device_bf16_output_from_bf16_bytes, device_bf16_output_from_bf16_bytes_with_device_row_prefix,
    device_bf16_output_from_device_template_buffer,
    device_bf16_output_from_device_template_with_device_row_prefix,
    device_bf16_output_from_f32_values, device_bf16_output_from_owned_device_buffer,
    device_buffer_byte_view, finish_quantize_device_bf16_to_nvfp4_row_payload,
    nvfp4_e2m1_fp8_e4m3_row_bytes, preload_resident_weight_from_host_staging,
    resident_weight_is_preloaded, residual_add_bf16_device_input_delta_view_device_output,
    residual_add_bf16_device_inputs_device_output, residual_add_prefix_bf16_bytes_into,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output,
    rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_async,
    silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only,
    sparse_b_scatter_shared_residual_add_bf16_async_owned_device_output,
    sparse_b_scatter_shared_residual_add_bf16_device_output,
    sparse_b_scatter_shared_residual_add_low_precision_device_output, CoordinatorCudaEvent,
    CudaStreamedSparseBAccumulator, DeviceBf16Output, StreamedSparseBAccumulatorChunk,
    StreamedSparseBResidualSegment,
    CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
};
use crate::commands::real_full::dense::math::bf16_bytes_from_f32;
use crate::commands::real_full::dense::REAL_FULL_DENSE_RMSNORM_EPS;
use crate::commands::real_full::embedding::{
    real_full_embedding_device_hidden_for_tokens, real_full_embedding_hidden_for_token,
};
use crate::commands::real_full::expert_probe::REAL_NVFP4_PROTOCOL_V2_EXECUTOR;
use crate::commands::real_full::intermediate_sharding::spark_expert_reduction_dispatch_for_rows;
use crate::commands::real_full::sparse_mlp::route::{
    cuda_route_validation_enabled,
    execute_nvfp4_route_rows_bf16_accumulated_cached_device_input_device_output, RouteTensorCache,
};
use crate::commands::real_full::sparse_mlp::router::{
    score_real_router_routes_bf16_cached_device_input, RouterTensorCache, ScoredRoute,
};
use crate::commands::real_full::types::RealFullSchedulerNumericProgressionSelfTest;

use super::super::protocol_v2::{
    real_full_moe_response_dtype_for_batch, real_full_protocol_v2_transport_config,
    real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent,
    real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent_structural_stats as real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent_structural,
    real_full_scheduler_host_batch_set_tcp_dispatch_with_payload_persistent,
    real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent,
    real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_streaming,
    real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_structural_stats,
    real_full_scheduler_host_batch_set_verbs_host_dispatch_with_payload_persistent,
    real_full_scheduler_verbs_host_direct_owner_payload_dispatch_with_payload_persistent_streaming,
    real_full_scheduler_verbs_host_try_start_direct_owner_payload_dispatch,
    scheduler_dispatched_route_count, scored_routes_for_scheduler_batch,
    DirectOwnerPayloadDispatchStart, RealFullSchedulerSparseTcpDispatchProbe,
};
use super::RealFullSchedulerDeviceAttentionDelta;

const NUMERIC_PROGRESS_HIDDEN_DIM: usize = GLM52_HIDDEN_SIZE;
const NUMERIC_PROGRESS_RESIDUAL_DTYPE: &str = "bf16";
const SCHEDULER_MLP_DELTA_BACKEND: &str = "cuda-scheduler-hidden-dependent-bf16-mlp-delta";
const SCHEDULER_REAL_DENSE_MLP_DELTA_STATUS: &str = "cuda-real-dense-checkpoint-mlp-delta";
const SCHEDULER_REAL_SPARSE_ROUTED_MLP_DELTA_STATUS: &str =
    "cuda-real-sparse-routed-nvfp4-checkpoint-mlp-delta";
const SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_DELTA_BACKEND: &str =
    "protocol-v2-real-sparse-routed-nvfp4-tcp-device-output";
const SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_ROUTE_BACKEND: &str =
    "protocol-v2-real-sparse-routed-nvfp4-tcp";
const SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_DELTA_BACKEND: &str =
    "protocol-v2-real-sparse-routed-nvfp4-verbs-host-device-output";
const SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_ROUTE_BACKEND: &str =
    "protocol-v2-real-sparse-routed-nvfp4-verbs-host";
const DEFAULT_SCHEDULER_TCP_MAX_GLOBAL_ROWS: usize = 2048;
const STREAMED_SPARSE_B_RESPONSE_BATCH_ROWS: usize = 256;
// Decode plus the maximum native MTP verify window is at most nine rows.  At
// that size, incrementally applying Spark responses only changes floating-point
// accumulation order according to network arrival; it cannot make a row ready
// before the complete response.  Collect those responses so the existing
// host-sorted combine path is both deterministic and launch-efficient.  Larger
// prefill batches retain incremental network/GPU overlap.
const MIN_INCREMENTAL_SPARSE_B_ROWS: usize = 16;
const SPARSE_TCP_STAGE_TIMING_ENV: &str = "GLMRT_REAL_FULL_SPARSE_TCP_STAGE_TIMING";
const MOE_PAYLOAD_HASH_DIAGNOSTIC_ENV: &str = "GLMRT_REAL_FULL_DIAGNOSTIC_MOE_PAYLOAD_HASH";
const LAYER_BOUNDARY_DUMP_DIR_ENV: &str = "GLMRT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_DIR";
const LAYER_BOUNDARY_DUMP_LAYER_ENV: &str = "GLMRT_REAL_FULL_DIAGNOSTIC_LAYER_DUMP_LAYER";
const PHASE0_SPARK_EXPERT_MODE_ENV: &str = "GLMRT_PHASE0_SPARK_EXPERT_MODE";
const B12X_PACKED_HIDDEN_EXCHANGE_ENV: &str = "GLMRT_REAL_FULL_B12X_PACKED_HIDDEN_EXCHANGE";
const B12X_COORDINATOR_ENV: &str = "GLMRT_B12X";
const ROLLING_SPARSE_PACKS_ENV: &str = "GLMRT_REAL_FULL_ROLLING_SPARSE_PACKS";
const ROLLING_SPARSE_SOURCE_ADMISSION_ROWS: usize = 512;
const ROLLING_SPARSE_PACK_ROWS: usize = 256;
const ROLLING_SPARSE_LOOKAHEAD_ROWS: usize = 4096;
pub(super) const ROLLING_SPARSE_ACCUMULATOR_PAGE_ROWS: usize = 2048;
const ROLLING_SPARSE_REQUIRED_MIN_ROWS: usize = 8_193;
// Keep the qualified <=8K layer-major path, then require bounded hidden
// segments and reclaim-page sparse accumulation through the configured 400K
// logical ceiling. Four merely medium prefills otherwise retain concurrent
// BF16 [context, hidden] planes plus FP32 sparse accumulators and can consume
// the production 600K pool's entire 8-GiB allocator safety margin.
const ROLLING_SPARSE_MAX_ROWS: usize = 400_000;
const ROLLING_SPARSE_OLDEST_ROWS: usize = 64;
const ROLLING_SPARSE_SELECTION_ROWS: usize = 32;
const ROLLING_SPARSE_EXPERT_TILE_ROWS: usize = 32;
pub(super) struct RealFullSchedulerNumericProgression {
    shape: RealFullSchedulerNumericProgressionShape,
    live_request: bool,
    event_owned_raw_tp4_reduction: bool,
    retain_final_target_device_hidden: bool,
    retain_full_target_device_hidden: bool,
    target_device_hidden_tap_rows: usize,
    target_device_hidden_tap_layer_ids: Vec<usize>,
    target_device_hidden_taps: Vec<Option<DeviceBf16Output>>,
    residual_bf16: Vec<u8>,
    initial_prefill_embedding_rows: usize,
    initial_prefill_embedding_bytes_read: u64,
    initial_decode_embedding_rows: usize,
    initial_decode_embedding_bytes_read: u64,
    selected_prefill_rows: usize,
    selected_decode_rows: usize,
    selected_mtp_rows: usize,
    attention_value_updates: usize,
    mlp_value_updates: usize,
    source_segments: usize,
    attention_residual_adds: usize,
    mlp_residual_adds: usize,
    attention_residual_add_backend: Option<&'static str>,
    mlp_residual_add_backend: Option<&'static str>,
    attention_device_output_delta_rows: usize,
    attention_device_output_delta_values: usize,
    attention_device_output_delta_checksum: f64,
    attention_device_output_delta_backend: Option<&'static str>,
    attention_device_output_delta_device_prefix_rows: usize,
    attention_device_output_delta_device_prefix_values: usize,
    attention_device_output_delta_device_prefix_backend: Option<&'static str>,
    device_delta_template_uploads: usize,
    device_delta_template_uses: usize,
    device_delta_template_resident_values: usize,
    device_delta_templates: BTreeMap<DeviceDeltaTemplateKey, DeviceBf16Output>,
    device_delta_template_upload_bf16_scratch: Vec<u8>,
    device_mlp_delta_rows: usize,
    device_mlp_delta_values: usize,
    device_mlp_delta_checksum: f64,
    device_mlp_delta_backend: Option<&'static str>,
    device_mlp_weight_uploads: usize,
    device_mlp_weight_resident_values: usize,
    device_mlp_weights: Option<SchedulerMlpResidentWeights>,
    device_mlp_weight_upload_bf16_scratch: Vec<u8>,
    device_real_dense_mlp_delta_rows: usize,
    device_real_dense_mlp_delta_values: usize,
    device_real_dense_mlp_delta_checksum: f64,
    device_real_dense_mlp_delta_backend: Option<&'static str>,
    device_real_dense_mlp_norm_backend: Option<&'static str>,
    device_real_dense_mlp_weight_tensors: usize,
    device_real_dense_mlp_weight_bytes: u64,
    device_real_dense_mlp_source_segments: usize,
    device_real_dense_mlp_layers: BTreeSet<usize>,
    device_real_dense_mlp_resident_weight_names: BTreeSet<String>,
    device_real_dense_mlp_resident_weights_by_layer:
        BTreeMap<usize, SchedulerDenseMlpResidentWeights>,
    device_real_sparse_shared_mlp_delta_rows: usize,
    device_real_sparse_shared_mlp_delta_values: usize,
    device_real_sparse_shared_mlp_delta_checksum: f64,
    device_real_sparse_shared_mlp_delta_backend: Option<&'static str>,
    device_real_sparse_shared_mlp_norm_backend: Option<&'static str>,
    device_real_sparse_shared_mlp_weight_tensors: usize,
    device_real_sparse_shared_mlp_weight_bytes: u64,
    device_real_sparse_shared_mlp_source_segments: usize,
    device_real_sparse_shared_mlp_layers: BTreeSet<usize>,
    device_real_sparse_shared_mlp_resident_weight_names: BTreeSet<String>,
    device_real_sparse_shared_mlp_resident_weights_by_layer:
        BTreeMap<usize, SchedulerSparseSharedMlpResidentWeights>,
    device_real_sparse_routed_mlp_delta_rows: usize,
    device_real_sparse_routed_mlp_delta_values: usize,
    device_real_sparse_routed_mlp_delta_checksum: f64,
    device_real_sparse_routed_mlp_delta_backend: Option<&'static str>,
    device_real_sparse_routed_mlp_route_backend: Option<&'static str>,
    device_real_sparse_routed_mlp_router_backend: Option<&'static str>,
    device_real_sparse_routed_mlp_routes: usize,
    device_real_sparse_routed_mlp_router_weight_bytes: u64,
    device_real_sparse_routed_mlp_router_bias_bytes: u64,
    device_real_sparse_routed_mlp_route_cache_cuda_entries: usize,
    device_real_sparse_routed_mlp_route_cache_cuda_uploads: usize,
    device_real_sparse_routed_mlp_route_cache_cuda_hits: usize,
    device_real_sparse_routed_mlp_router_cache_entries: usize,
    device_real_sparse_routed_mlp_router_cache_hits: usize,
    device_real_sparse_routed_mlp_source_segments: usize,
    device_real_sparse_routed_mlp_layers: BTreeSet<usize>,
    device_real_sparse_routed_mlp_router_cache: RouterTensorCache,
    device_real_sparse_routed_mlp_route_cache: RouteTensorCache,
    device_real_sparse_routed_mlp_intermediate_rows: BTreeMap<(usize, usize), usize>,
    sparse_tcp_routed_mlp: Option<RealFullSchedulerSparseTcpRoutedMlpContext>,
    device_hidden_segment_residual_adds: usize,
    device_hidden_segment_value_updates: usize,
    device_hidden_segment_residual_add_backend: Option<&'static str>,
    device_hidden_segments: BTreeMap<DeviceHiddenSegmentKey, DeviceBf16Output>,
    delta_bf16_scratch: Vec<u8>,
    output_bf16_scratch: Vec<u8>,
    device_sparse_routed_normalized_readback_bf16_scratch: Vec<u8>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RealFullSchedulerNumericProgressionShape {
    pub(super) prefix_tokens: usize,
    pub(super) prefill_rows: usize,
    pub(super) prefill_chunk_tokens: usize,
    pub(super) decode_rows: usize,
    pub(super) mtp_rows: usize,
    pub(super) mtp_accepted_rows: usize,
    pub(super) source_segments_per_layer: usize,
    pub(super) sparse_source_segments_per_layer: usize,
}

impl RealFullSchedulerNumericProgressionShape {
    pub(super) fn from_execution_shape(shape: &super::RealFullSchedulerExecutionShape) -> Self {
        let prefill_chunks = shape.prefill_tokens.div_ceil(shape.prefill_chunk_tokens);
        let source_segments_per_layer =
            prefill_chunks + shape.decode_rows + usize::from(shape.mtp_rows > 0);
        Self {
            prefix_tokens: shape.prefix_tokens,
            prefill_rows: shape.prefill_tokens,
            prefill_chunk_tokens: shape.prefill_chunk_tokens,
            decode_rows: shape.decode_rows,
            mtp_rows: shape.mtp_rows,
            mtp_accepted_rows: shape.mtp_accepted_rows,
            source_segments_per_layer,
            sparse_source_segments_per_layer: source_segments_per_layer,
        }
    }

    fn unique_rows(self) -> usize {
        self.prefill_rows + self.decode_rows + self.mtp_rows
    }
}

#[derive(Clone, Copy)]
pub(super) struct RealFullSchedulerDeviceHiddenSource {
    pub(super) buffer: GlmrtDeviceBuffer,
    pub(super) rows: usize,
    pub(super) values_per_row: usize,
}

pub(super) struct RealFullSchedulerNumericProgressionFinish {
    pub(super) self_test: RealFullSchedulerNumericProgressionSelfTest,
    pub(super) final_decode_device_hidden: Option<DeviceBf16Output>,
    pub(super) final_target_device_hidden: Option<DeviceBf16Output>,
    pub(super) target_device_hidden_taps: Option<RealFullSchedulerTargetHiddenTaps>,
    pub(super) sparse_tcp_dispatch_probe: Option<RealFullSchedulerSparseTcpDispatchProbe>,
}

pub(in crate::commands::real_full) struct RealFullSchedulerTargetHiddenTaps {
    pub(in crate::commands::real_full) layer_ids: Vec<usize>,
    pub(in crate::commands::real_full) row_start: usize,
    pub(in crate::commands::real_full) rows: usize,
    pub(in crate::commands::real_full) values: Vec<DeviceBf16Output>,
}

pub(in crate::commands::real_full) struct RealFullSchedulerSparseTcpDispatchWorker {
    target_count: usize,
    transport: RealFullSchedulerSparseDispatchTransport,
    streaming_responses: bool,
    direct_owner_client: Option<VerbsHostProtocolV2HostBatchSetPersistentClient>,
    tx: tokio::sync::mpsc::UnboundedSender<SchedulerSparseTcpDispatchMessage>,
    join: Mutex<Option<JoinHandle<()>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::commands::real_full) enum RealFullSchedulerSparseDispatchTransport {
    Tcp,
    VerbsHost,
}

impl RealFullSchedulerSparseDispatchTransport {
    pub(in crate::commands::real_full) fn from_label(label: &str) -> Option<Self> {
        match label {
            "tcp" => Some(Self::Tcp),
            "verbs-host" => Some(Self::VerbsHost),
            _ => None,
        }
    }

    pub(in crate::commands::real_full) fn label(self) -> &'static str {
        match self {
            Self::Tcp => "tcp",
            Self::VerbsHost => "verbs-host",
        }
    }

    fn thread_name(self) -> &'static str {
        match self {
            Self::Tcp => "glmrt-scheduler-sparse-tcp-persistent",
            Self::VerbsHost => "glmrt-scheduler-sparse-verbs-host",
        }
    }

    fn dispatch_scope(self) -> &'static str {
        match self {
            Self::Tcp => "dispatch real sparse routed scheduler MLP deltas through ProtocolV2 TCP and feed the accumulated output into the live residual path",
            Self::VerbsHost => "dispatch real sparse routed scheduler MLP deltas through ProtocolV2 verbs-host RDMA and feed the accumulated output into the live residual path",
        }
    }

    fn passed_status(self) -> &'static str {
        match self {
            Self::Tcp => "request-shaped-sparse-tcp-residual-dispatch-passed",
            Self::VerbsHost => "request-shaped-sparse-verbs-host-residual-dispatch-passed",
        }
    }

    fn blocked_status(self) -> &'static str {
        match self {
            Self::Tcp => "request-shaped-sparse-tcp-residual-dispatch-blocked",
            Self::VerbsHost => "request-shaped-sparse-verbs-host-residual-dispatch-blocked",
        }
    }

    fn sparse_delta_backend(self) -> &'static str {
        match self {
            Self::Tcp => SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_DELTA_BACKEND,
            Self::VerbsHost => SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_DELTA_BACKEND,
        }
    }

    fn sparse_route_backend(self) -> &'static str {
        match self {
            Self::Tcp => SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_ROUTE_BACKEND,
            Self::VerbsHost => SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_ROUTE_BACKEND,
        }
    }
}

struct SchedulerSparseTcpPayloadDispatchHandle {
    batch: SchedulerSparseTcpPayloadDispatchBatchShape,
    batch_index: usize,
    started: Option<Instant>,
    row_sharded_completion_chunks: bool,
    chunk_rx: Option<mpsc::Receiver<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
    response_rx: Option<mpsc::Receiver<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>>,
    direct_owner_pending: Option<VerbsHostProtocolV2ReducedIdentityPayloadPending>,
    sliced_dispatches: Vec<SchedulerSparseTcpPayloadSliceDispatch>,
    sliced_poll_cursor: usize,
    deferred_streaming_completion: Option<TcpProtocolV2HostBatchSetBf16PayloadDispatch>,
}

impl SchedulerSparseTcpPayloadDispatchHandle {
    fn response_batch_target_rows(&self) -> usize {
        if self.row_sharded_completion_chunks {
            self.batch.rows
        } else {
            STREAMED_SPARSE_B_RESPONSE_BATCH_ROWS
        }
    }

    fn has_response_chunks(&self) -> bool {
        self.chunk_rx.is_some()
            || self.direct_owner_pending.is_some()
            || !self.sliced_dispatches.is_empty()
    }

    fn has_streaming_response_chunks(&self) -> bool {
        self.chunk_rx.is_some() || !self.sliced_dispatches.is_empty()
    }

    fn poll_streaming_response(
        &mut self,
        block: bool,
    ) -> Result<SchedulerSparseTcpPayloadStreamPoll> {
        if let Some(dispatch) = self.deferred_streaming_completion.take() {
            return Ok(SchedulerSparseTcpPayloadStreamPoll::Complete(dispatch));
        }
        anyhow::ensure!(
            self.direct_owner_pending.is_none(),
            "direct Spark-owner dispatch does not expose coordinator response chunks"
        );
        if self.sliced_dispatches.is_empty() {
            return self.poll_single_streaming_response(block);
        }
        anyhow::ensure!(
            self.chunk_rx.is_none() && self.response_rx.is_none(),
            "sliced sparse dispatch also carried a logical response channel"
        );
        self.poll_sliced_streaming_response(block)
    }

    fn poll_single_streaming_response(
        &mut self,
        block: bool,
    ) -> Result<SchedulerSparseTcpPayloadStreamPoll> {
        let chunk_rx = self
            .chunk_rx
            .as_ref()
            .context("streamed sparse dispatch is missing its response chunk channel")?;
        let chunk = if block {
            match chunk_rx.recv() {
                Ok(chunk) => Some(chunk),
                Err(_) => None,
            }
        } else {
            match chunk_rx.try_recv() {
                Ok(chunk) => Some(chunk),
                Err(mpsc::TryRecvError::Empty) => {
                    return Ok(SchedulerSparseTcpPayloadStreamPoll::Pending);
                }
                Err(mpsc::TryRecvError::Disconnected) => None,
            }
        };
        if let Some(chunk) = chunk {
            return Ok(SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk));
        }
        let response_rx = self
            .response_rx
            .as_ref()
            .context("streamed sparse dispatch is missing its worker response channel")?;
        let response = if block {
            Some(
                response_rx
                    .recv()
                    .context("receiving completed streamed sparse dispatch response")?,
            )
        } else {
            match response_rx.try_recv() {
                Ok(response) => Some(response),
                Err(mpsc::TryRecvError::Empty) => None,
                Err(mpsc::TryRecvError::Disconnected) => {
                    anyhow::bail!("streamed sparse dispatch response channel disconnected")
                }
            }
        };
        let Some(response) = response else {
            return Ok(SchedulerSparseTcpPayloadStreamPoll::Pending);
        };
        let _ = self.response_rx.take();
        Ok(SchedulerSparseTcpPayloadStreamPoll::Complete(response?))
    }

    fn poll_sliced_streaming_response(
        &mut self,
        block: bool,
    ) -> Result<SchedulerSparseTcpPayloadStreamPoll> {
        loop {
            let slice_count = self.sliced_dispatches.len();
            for offset in 0..slice_count {
                let slice_index = (self.sliced_poll_cursor + offset) % slice_count;
                let dispatch = &mut self.sliced_dispatches[slice_index];
                let Some(chunk_rx) = dispatch.chunk_rx.as_ref() else {
                    continue;
                };
                match chunk_rx.try_recv() {
                    Ok(mut chunk) => {
                        offset_sparse_payload_chunk_rows(&mut chunk, dispatch.row_start)?;
                        self.sliced_poll_cursor = (slice_index + 1) % slice_count;
                        return Ok(SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk));
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                    Err(mpsc::TryRecvError::Disconnected) => {
                        dispatch.chunk_rx = None;
                    }
                }
            }

            let Some(slice_index) = self
                .sliced_dispatches
                .iter()
                .position(|dispatch| dispatch.chunk_rx.is_some())
            else {
                break;
            };
            if !block {
                return Ok(SchedulerSparseTcpPayloadStreamPoll::Pending);
            }
            let dispatch = &mut self.sliced_dispatches[slice_index];
            let chunk_rx = dispatch
                .chunk_rx
                .as_ref()
                .expect("selected sliced dispatch has a response channel");
            match chunk_rx.recv() {
                Ok(mut chunk) => {
                    offset_sparse_payload_chunk_rows(&mut chunk, dispatch.row_start)?;
                    self.sliced_poll_cursor = (slice_index + 1) % slice_count;
                    return Ok(SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk));
                }
                Err(_) => dispatch.chunk_rx = None,
            }
        }

        let mut pending_response = false;
        for dispatch in &mut self.sliced_dispatches {
            if dispatch.response.is_some() {
                continue;
            }
            let response_rx = dispatch
                .response_rx
                .as_ref()
                .context("sliced sparse dispatch is missing its worker response channel")?;
            let response = if block {
                Some(
                    response_rx
                        .recv()
                        .context("receiving completed sliced sparse dispatch response")?,
                )
            } else {
                match response_rx.try_recv() {
                    Ok(response) => Some(response),
                    Err(mpsc::TryRecvError::Empty) => {
                        pending_response = true;
                        None
                    }
                    Err(mpsc::TryRecvError::Disconnected) => {
                        anyhow::bail!("sliced sparse dispatch response channel disconnected")
                    }
                }
            };
            if let Some(response) = response {
                dispatch.response = Some(response?);
                let _ = dispatch.response_rx.take();
            }
        }
        if pending_response {
            return Ok(SchedulerSparseTcpPayloadStreamPoll::Pending);
        }
        let dispatches = self
            .sliced_dispatches
            .iter_mut()
            .map(|dispatch| {
                Ok((
                    dispatch.row_start,
                    dispatch.row_count,
                    dispatch
                        .response
                        .take()
                        .context("completed sliced sparse dispatch lost its response")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(SchedulerSparseTcpPayloadStreamPoll::Complete(
            merge_scheduler_sparse_payload_slice_dispatches(self.batch, dispatches)?,
        ))
    }
}

struct SchedulerSparseTcpPayloadSliceDispatch {
    row_start: usize,
    row_count: usize,
    chunk_rx: Option<mpsc::Receiver<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
    response_rx: Option<mpsc::Receiver<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>>,
    response: Option<TcpProtocolV2HostBatchSetBf16PayloadDispatch>,
}

enum SchedulerSparseTcpPayloadStreamPoll {
    Pending,
    Chunk(VerbsHostProtocolV2HostBatchSetBf16PayloadChunk),
    Complete(TcpProtocolV2HostBatchSetBf16PayloadDispatch),
}

#[derive(Clone, Copy)]
struct SchedulerSparseTcpPayloadDispatchBatchShape {
    layer_id: LayerId,
    rows: usize,
    routes: usize,
    unique_experts: usize,
    max_expert_load: usize,
    expert_load_square_sum: usize,
    route_profiled: bool,
    hidden_dim: usize,
}

impl SchedulerSparseTcpPayloadDispatchBatchShape {
    fn from_batch_and_routes(batch: &ExpertBatch, routes: &[ExpertBatchRoute]) -> Self {
        let route_profiled = dspark_route_profile_enabled();
        let compute_route_shape = route_profiled || sparse_tcp_stage_timing_enabled();
        let mut expert_loads = [0_usize; GLM52_ROUTED_EXPERTS];
        if compute_route_shape {
            for route in routes {
                if let Some(load) = expert_loads.get_mut(route.expert_id) {
                    *load += 1;
                }
            }
        }
        let unique_experts = expert_loads.iter().filter(|load| **load > 0).count();
        let max_expert_load = expert_loads.iter().copied().max().unwrap_or(0);
        let expert_load_square_sum = expert_loads
            .iter()
            .map(|load| load.saturating_mul(*load))
            .sum();
        Self {
            layer_id: batch.layer_id,
            rows: batch.num_rows(),
            routes: batch.route_count(),
            unique_experts,
            max_expert_load,
            expert_load_square_sum,
            route_profiled,
            hidden_dim: batch.hidden_dim,
        }
    }
}

enum SchedulerSparseTcpDispatchMessage {
    Dispatch {
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        global_hidden_payload: Vec<u8>,
        request_id_base: u64,
        response_tx: mpsc::Sender<Result<TcpProtocolV2HostBatchSetDispatch>>,
    },
    DispatchPayload {
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        global_hidden_payload: SchedulerSparseTcpDispatchPayload,
        request_id_base: u64,
        include_contribution_counts: bool,
        chunk_tx: Option<mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
        response_tx: mpsc::Sender<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>,
    },
    Shutdown,
}

enum SchedulerSparseTcpDispatchPayload {
    Owned(Vec<u8>),
    SharedSlice {
        payload: Arc<Vec<u8>>,
        byte_start: usize,
        byte_end: usize,
    },
}

impl SchedulerSparseTcpDispatchPayload {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Owned(payload) => payload,
            Self::SharedSlice {
                payload,
                byte_start,
                byte_end,
            } => &payload[*byte_start..*byte_end],
        }
    }

    fn into_owned(self) -> Vec<u8> {
        match self {
            Self::Owned(payload) => payload,
            Self::SharedSlice {
                payload,
                byte_start,
                byte_end,
            } => payload[byte_start..byte_end].to_vec(),
        }
    }
}

enum SchedulerSparseProtocolV2PersistentClient {
    Tcp(TcpProtocolV2HostBatchSetPersistentClient),
    VerbsHost(VerbsHostProtocolV2HostBatchSetPersistentClient),
}

async fn run_scheduler_sparse_verbs_host_dispatch_worker(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<SchedulerSparseTcpDispatchMessage>,
    client: VerbsHostProtocolV2HostBatchSetPersistentClient,
    owner_lookup: Option<ExpertOwnerLookup>,
) {
    const MAX_OUTSTANDING_DISPATCHES: usize = 8;

    let owner_lookup = Arc::new(owner_lookup);
    let mut tasks = tokio::task::JoinSet::new();
    while let Some(message) = rx.recv().await {
        if matches!(message, SchedulerSparseTcpDispatchMessage::Shutdown) {
            break;
        }
        if tasks.len() == MAX_OUTSTANDING_DISPATCHES {
            if let Some(Err(error)) = tasks.join_next().await {
                tracing::error!(%error, "scheduler sparse verbs-host dispatch task failed");
            }
        }
        let client = client.clone();
        let owner_lookup = Arc::clone(&owner_lookup);
        tasks.spawn(async move {
            match message {
                SchedulerSparseTcpDispatchMessage::Dispatch {
                    batch,
                    routes,
                    global_hidden_payload,
                    request_id_base,
                    response_tx,
                } => {
                    let result = real_full_scheduler_host_batch_set_verbs_host_dispatch_with_payload_persistent(
                        &client,
                        &batch,
                        &routes,
                        &global_hidden_payload,
                        owner_lookup.as_ref().as_ref(),
                        request_id_base,
                    )
                    .await;
                    let _ = response_tx.send(result);
                }
                SchedulerSparseTcpDispatchMessage::DispatchPayload {
                    batch,
                    routes,
                    global_hidden_payload,
                    request_id_base,
                    include_contribution_counts,
                    chunk_tx,
                    response_tx,
                } => {
                    let global_hidden_payload = global_hidden_payload.as_slice();
                    let result = if include_contribution_counts {
                        real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent(
                            &client,
                            &batch,
                            &routes,
                            global_hidden_payload,
                            owner_lookup.as_ref().as_ref(),
                            request_id_base,
                        )
                        .await
                    } else if let Some(chunk_tx) = chunk_tx {
                        match real_full_scheduler_verbs_host_direct_owner_payload_dispatch_with_payload_persistent_streaming(
                            &client,
                            &batch,
                            &routes,
                            global_hidden_payload,
                            request_id_base,
                            &chunk_tx,
                        )
                        .await
                        {
                            Ok(Some(stats)) => {
                                Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
                                    partial_outputs_bf16_by_host: Vec::new(),
                                    global_row_indices_by_host: Vec::new(),
                                    completed_global_row_slices: Vec::new(),
                                    stats,
                                })
                            }
                            Ok(None) => real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_streaming(
                                &client,
                                &batch,
                                &routes,
                                global_hidden_payload,
                                owner_lookup.as_ref().as_ref(),
                                request_id_base,
                                chunk_tx,
                            )
                            .await
                            .map(|stats| TcpProtocolV2HostBatchSetBf16PayloadDispatch {
                                partial_outputs_bf16_by_host: Vec::new(),
                                global_row_indices_by_host: Vec::new(),
                                completed_global_row_slices: Vec::new(),
                                stats,
                            }),
                            Err(error) => Err(error),
                        }
                    } else {
                        real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_structural_stats(
                            &client,
                            &batch,
                            &routes,
                            global_hidden_payload,
                            owner_lookup.as_ref().as_ref(),
                            request_id_base,
                        )
                        .await
                    };
                    let _ = response_tx.send(result);
                }
                SchedulerSparseTcpDispatchMessage::Shutdown => unreachable!(
                    "scheduler sparse verbs-host shutdown is handled before task creation"
                ),
            }
        });
    }
    while let Some(result) = tasks.join_next().await {
        if let Err(error) = result {
            tracing::error!(%error, "scheduler sparse verbs-host dispatch task failed");
        }
    }
}

fn run_scheduler_sparse_direct_owner_dispatch_worker(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<SchedulerSparseTcpDispatchMessage>,
    client: VerbsHostProtocolV2PersistentClient,
) {
    while let Some(message) = rx.blocking_recv() {
        match message {
            SchedulerSparseTcpDispatchMessage::Dispatch { response_tx, .. } => {
                let _ = response_tx.send(Err(anyhow::anyhow!(
                    "direct Spark owner dispatch only supports BF16 payload responses"
                )));
            }
            SchedulerSparseTcpDispatchMessage::DispatchPayload {
                batch,
                routes,
                global_hidden_payload,
                request_id_base,
                include_contribution_counts,
                chunk_tx,
                response_tx,
            } => {
                let result = if chunk_tx.is_some() {
                    Err(anyhow::anyhow!(
                        "direct Spark owner dispatch does not emit streamed response chunks"
                    ))
                } else {
                    direct_owner_bf16_payload_dispatch(
                        &client,
                        &batch,
                        &routes,
                        global_hidden_payload.into_owned(),
                        request_id_base,
                        include_contribution_counts,
                    )
                };
                let _ = response_tx.send(result);
            }
            SchedulerSparseTcpDispatchMessage::Shutdown => break,
        }
    }
}

fn run_scheduler_sparse_zero_output_dispatch_worker(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<SchedulerSparseTcpDispatchMessage>,
) {
    while let Some(message) = rx.blocking_recv() {
        match message {
            SchedulerSparseTcpDispatchMessage::Dispatch { response_tx, .. } => {
                let _ = response_tx.send(Err(anyhow::anyhow!(
                    "Spark layer-block startup sizing requires BF16 payload dispatch"
                )));
            }
            SchedulerSparseTcpDispatchMessage::DispatchPayload {
                batch,
                include_contribution_counts,
                chunk_tx,
                response_tx,
                ..
            } => {
                let result = if chunk_tx.is_some() {
                    Err(anyhow::anyhow!(
                        "Spark layer-block startup sizing does not emit streamed chunks"
                    ))
                } else {
                    zero_bf16_payload_dispatch(&batch, include_contribution_counts)
                };
                let _ = response_tx.send(result);
            }
            SchedulerSparseTcpDispatchMessage::Shutdown => break,
        }
    }
}

fn zero_bf16_payload_dispatch(
    batch: &ExpertBatch,
    include_contribution_counts: bool,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let rows = batch.num_rows();
    let payload_bytes = rows
        .checked_mul(batch.hidden_dim)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("Spark layer-block startup zero payload byte count overflow")?;
    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host: vec![vec![0_u8; payload_bytes]],
        global_row_indices_by_host: vec![(0..rows).collect()],
        completed_global_row_slices: Vec::new(),
        stats: TcpProtocolV2HostBatchSetDispatchStats {
            hosts: 1,
            global_rows: rows,
            host_rows: rows,
            routes: scheduler_dispatched_route_count(batch.route_count())?,
            output_dim: batch.hidden_dim,
            output_values: rows * batch.hidden_dim,
            request_wire_bytes: 0,
            response_wire_bytes: 0,
            response_executor_ids: vec![expert_protocol_v2_compact_id(
                REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
            )],
            contribution_counts: if include_contribution_counts {
                vec![1; rows]
            } else {
                Vec::new()
            },
            output_checksum: 0.0,
            graph_pool_leases: 0,
            graph_pool_fixed_buffer_bytes: 0,
            graph_pool_active_rows: rows,
            graph_pool_active_routes: batch.route_count(),
            graph_pool_active_expert_tiles: batch.route_count(),
            graph_pool_bucket_rows: vec![batch.graph_bucket.row_capacity],
        },
    })
}

fn direct_owner_bf16_payload_dispatch(
    client: &VerbsHostProtocolV2PersistentClient,
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    global_hidden_payload: Vec<u8>,
    request_id: u64,
    include_contribution_counts: bool,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        batch.num_rows() == 1,
        "direct Spark owner dispatch currently supports single-row decode only"
    );
    let wire_routes = routes
        .iter()
        .map(|route| {
            Ok(ExpertProtocolV2RouteEntry {
                row_index: u32::try_from(route.row_index)
                    .context("direct owner route row index exceeds u32")?,
                expert_id: u32::try_from(route.expert_id)
                    .context("direct owner expert id exceeds u32")?,
                gate_weight: route.gate_weight,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let request = ExpertProtocolV2Request::from_expert_batch(
        request_id,
        batch,
        wire_routes,
        global_hidden_payload,
    )?
    .with_spark_reduction();
    let request_wire_bytes = request.wire_stats().wire_bytes;
    let response = client
        .enqueue_roundtrip(request.clone())?
        .wait()
        .context("waiting for direct Spark owner reduced response")?;
    anyhow::ensure!(
        response.header.status == ExpertProtocolV2Status::Ok,
        "direct Spark owner response status {:?} is not ok",
        response.header.status
    );
    anyhow::ensure!(
        response.header.request_id == request.header.request_id
            && response.header.placement_version == request.header.placement_version
            && response.header.layer_id == request.header.layer_id,
        "direct Spark owner response identity did not match request"
    );
    anyhow::ensure!(
        response.header.row_count as usize == batch.num_rows()
            && response.header.output_dim as usize == batch.hidden_dim
            && response.header.output_dtype == ExpertV2Dtype::Bf16,
        "direct Spark owner response shape/dtype mismatch: rows={} dim={} dtype={:?}",
        response.header.row_count,
        response.header.output_dim,
        response.header.output_dtype
    );
    let logical_row_bytes = ExpertV2Dtype::Bf16.row_bytes(batch.hidden_dim)?;
    let stride = response.header.output_row_stride_bytes as usize;
    anyhow::ensure!(
        stride >= logical_row_bytes
            && response.partial_output_payload.len() == batch.num_rows() * stride,
        "direct Spark owner response payload/stride mismatch"
    );
    let partial_output = if stride == logical_row_bytes {
        response.partial_output_payload.clone()
    } else {
        let mut compact = Vec::with_capacity(batch.num_rows() * logical_row_bytes);
        for row in response.partial_output_payload.chunks_exact(stride) {
            compact.extend_from_slice(&row[..logical_row_bytes]);
        }
        compact
    };
    let response_wire_bytes = response.wire_stats().wire_bytes;
    let rows = batch.num_rows();
    let replicated_routes = scheduler_dispatched_route_count(batch.route_count())?;
    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host: vec![partial_output],
        global_row_indices_by_host: vec![(0..rows).collect()],
        completed_global_row_slices: Vec::new(),
        stats: TcpProtocolV2HostBatchSetDispatchStats {
            hosts: 1,
            global_rows: rows,
            host_rows: rows,
            routes: replicated_routes,
            output_dim: batch.hidden_dim,
            output_values: rows * batch.hidden_dim,
            request_wire_bytes,
            response_wire_bytes,
            response_executor_ids: vec![response.header.executor_id],
            contribution_counts: if include_contribution_counts {
                vec![1; rows]
            } else {
                Vec::new()
            },
            output_checksum: 0.0,
            graph_pool_leases: 0,
            graph_pool_fixed_buffer_bytes: 0,
            graph_pool_active_rows: rows,
            graph_pool_active_routes: batch.route_count(),
            graph_pool_active_expert_tiles: batch.route_count(),
            graph_pool_bucket_rows: vec![batch.graph_bucket.row_capacity],
        },
    })
}

impl SchedulerSparseProtocolV2PersistentClient {
    fn new(
        transport: RealFullSchedulerSparseDispatchTransport,
        targets: Vec<TcpProtocolV2HostBatchTarget>,
    ) -> Result<Self> {
        let config = real_full_protocol_v2_transport_config()?;
        match transport {
            RealFullSchedulerSparseDispatchTransport::Tcp => Ok(Self::Tcp(
                TcpProtocolV2HostBatchSetPersistentClient::new(targets, config),
            )),
            RealFullSchedulerSparseDispatchTransport::VerbsHost => Ok(Self::VerbsHost(
                VerbsHostProtocolV2HostBatchSetPersistentClient::new(targets, config)?,
            )),
        }
    }
}

impl RealFullSchedulerSparseTcpDispatchWorker {
    pub(in crate::commands::real_full) fn new(
        targets: Vec<TcpProtocolV2HostBatchTarget>,
        owner_lookup: Option<ExpertOwnerLookup>,
    ) -> Result<Self> {
        Self::new_with_transport(
            RealFullSchedulerSparseDispatchTransport::Tcp,
            targets,
            owner_lookup,
        )
    }

    pub(in crate::commands::real_full) fn new_with_transport(
        transport: RealFullSchedulerSparseDispatchTransport,
        targets: Vec<TcpProtocolV2HostBatchTarget>,
        owner_lookup: Option<ExpertOwnerLookup>,
    ) -> Result<Self> {
        Self::new_with_transport_and_cpu_affinity(transport, targets, owner_lookup, None)
    }

    pub(in crate::commands::real_full) fn new_with_transport_and_cpu_affinity(
        transport: RealFullSchedulerSparseDispatchTransport,
        targets: Vec<TcpProtocolV2HostBatchTarget>,
        owner_lookup: Option<ExpertOwnerLookup>,
        worker_cpu: Option<usize>,
    ) -> Result<Self> {
        let target_count = targets.len();
        anyhow::ensure!(
            target_count > 0,
            "scheduler sparse {} dispatch worker requires at least one target",
            transport.label()
        );
        let (tx, mut rx) =
            tokio::sync::mpsc::unbounded_channel::<SchedulerSparseTcpDispatchMessage>();
        let (startup_tx, startup_rx) = std::sync::mpsc::sync_channel::<std::io::Result<()>>(1);
        let verbs_host_client = if transport == RealFullSchedulerSparseDispatchTransport::VerbsHost
        {
            Some(VerbsHostProtocolV2HostBatchSetPersistentClient::new(
                targets.clone(),
                real_full_protocol_v2_transport_config()?,
            )?)
        } else {
            None
        };
        let direct_owner_client = verbs_host_client.clone();
        let streaming_responses = transport == RealFullSchedulerSparseDispatchTransport::VerbsHost
            && spark_expert_reduction_dispatch_for_rows(
                scheduler_tcp_max_global_rows_per_dispatch(),
            )?
            .is_some();
        let worker_label = transport.label();
        let join = thread::Builder::new()
            .name(transport.thread_name().to_owned())
            .spawn(move || {
                let startup_result = worker_cpu
                    .map(glmrt_core::pin_current_thread_to_cpu)
                    .unwrap_or(Ok(()));
                let startup_succeeded = startup_result.is_ok();
                let _ = startup_tx.send(startup_result);
                if !startup_succeeded {
                    return;
                }
                let runtime = match tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .with_context(|| {
                        format!("building scheduler sparse {worker_label} runtime")
                    })
                {
                    Ok(runtime) => runtime,
                    Err(error) => {
                        while let Some(message) = rx.blocking_recv() {
                            match message {
                                SchedulerSparseTcpDispatchMessage::Dispatch {
                                    response_tx, ..
                                    } => {
                                        let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "scheduler sparse {worker_label} worker failed to start: {error:#}"
                                    )));
                                    }
                                SchedulerSparseTcpDispatchMessage::DispatchPayload {
                                    response_tx, ..
                                    } => {
                                        let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "scheduler sparse {worker_label} worker failed to start: {error:#}"
                                    )));
                                    }
                                SchedulerSparseTcpDispatchMessage::Shutdown => break,
                            }
                        }
                        return;
                    }
                };
                let client = match verbs_host_client {
                    Some(client) => Ok(SchedulerSparseProtocolV2PersistentClient::VerbsHost(client)),
                    None => SchedulerSparseProtocolV2PersistentClient::new(
                        transport,
                        targets.clone(),
                    ),
                };
                let client = match client {
                    Ok(client) => client,
                    Err(error) => {
                        while let Some(message) = rx.blocking_recv() {
                            match message {
                                SchedulerSparseTcpDispatchMessage::Dispatch {
                                    response_tx, ..
                                } => {
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "scheduler sparse {worker_label} persistent client failed to start: {error:#}"
                                    )));
                                }
                                SchedulerSparseTcpDispatchMessage::DispatchPayload {
                                    response_tx, ..
                                } => {
                                    let _ = response_tx.send(Err(anyhow::anyhow!(
                                        "scheduler sparse {worker_label} persistent client failed to start: {error:#}"
                                    )));
                                }
                                SchedulerSparseTcpDispatchMessage::Shutdown => break,
                            }
                        }
                        return;
                    }
                };
                let mut client = match client {
                    SchedulerSparseProtocolV2PersistentClient::VerbsHost(client) => {
                        runtime.block_on(run_scheduler_sparse_verbs_host_dispatch_worker(
                            rx,
                            client,
                            owner_lookup,
                        ));
                        return;
                    }
                    SchedulerSparseProtocolV2PersistentClient::Tcp(client) => {
                        SchedulerSparseProtocolV2PersistentClient::Tcp(client)
                    }
                };
                while let Some(message) = rx.blocking_recv() {
                    match message {
                        SchedulerSparseTcpDispatchMessage::Dispatch {
                            batch,
                            routes,
                            global_hidden_payload,
                            request_id_base,
                            response_tx,
                        } => {
                            let result = match transport {
                                RealFullSchedulerSparseDispatchTransport::Tcp => {
                                    let SchedulerSparseProtocolV2PersistentClient::Tcp(client) =
                                        &mut client
                                    else {
                                        unreachable!("TCP sparse dispatch has TCP client")
                                    };
                                    runtime.block_on(
                                        real_full_scheduler_host_batch_set_tcp_dispatch_with_payload_persistent(
                                            client,
                                            &batch,
                                            &routes,
                                            &global_hidden_payload,
                                            owner_lookup.as_ref(),
                                            request_id_base,
                                        ),
                                    )
                                }
                                RealFullSchedulerSparseDispatchTransport::VerbsHost => {
                                    let SchedulerSparseProtocolV2PersistentClient::VerbsHost(
                                        client,
                                    ) = &mut client
                                    else {
                                        unreachable!(
                                            "verbs-host sparse dispatch has verbs-host client"
                                        )
                                    };
                                    runtime.block_on(
                                        real_full_scheduler_host_batch_set_verbs_host_dispatch_with_payload_persistent(
                                            client,
                                            &batch,
                                            &routes,
                                            &global_hidden_payload,
                                            owner_lookup.as_ref(),
                                            request_id_base,
                                        ),
                                    )
                                }
                            };
                            let _ = response_tx.send(result);
                        }
                        SchedulerSparseTcpDispatchMessage::DispatchPayload {
                            batch,
                            routes,
                            global_hidden_payload,
                            request_id_base,
                            include_contribution_counts,
                            chunk_tx,
                            response_tx,
                        } => {
                            let global_hidden_payload = global_hidden_payload.as_slice();
                            let result = match (transport, include_contribution_counts) {
                                (RealFullSchedulerSparseDispatchTransport::Tcp, true) => {
                                    let SchedulerSparseProtocolV2PersistentClient::Tcp(client) =
                                        &mut client
                                    else {
                                        unreachable!("TCP sparse dispatch has TCP client")
                                    };
                                    runtime.block_on(
                                        real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent(
                                            client,
                                            &batch,
                                            &routes,
                                            global_hidden_payload,
                                            owner_lookup.as_ref(),
                                            request_id_base,
                                        ),
                                    )
                                }
                                (RealFullSchedulerSparseDispatchTransport::Tcp, false) => {
                                    let SchedulerSparseProtocolV2PersistentClient::Tcp(client) =
                                        &mut client
                                    else {
                                        unreachable!("TCP sparse dispatch has TCP client")
                                    };
                                    runtime.block_on(
                                        real_full_scheduler_host_batch_set_tcp_bf16_payload_dispatch_with_payload_persistent_structural(
                                            client,
                                            &batch,
                                            &routes,
                                            global_hidden_payload,
                                            owner_lookup.as_ref(),
                                            request_id_base,
                                        ),
                                    )
                                }
                                (RealFullSchedulerSparseDispatchTransport::VerbsHost, true) => {
                                    let SchedulerSparseProtocolV2PersistentClient::VerbsHost(
                                        client,
                                    ) = &mut client
                                    else {
                                        unreachable!(
                                            "verbs-host sparse dispatch has verbs-host client"
                                        )
                                    };
                                    runtime.block_on(
                                        real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent(
                                            client,
                                            &batch,
                                            &routes,
                                            global_hidden_payload,
                                            owner_lookup.as_ref(),
                                            request_id_base,
                                        ),
                                    )
                                }
                                (RealFullSchedulerSparseDispatchTransport::VerbsHost, false) => {
                                    let SchedulerSparseProtocolV2PersistentClient::VerbsHost(
                                        client,
                                    ) = &mut client
                                    else {
                                        unreachable!(
                                            "verbs-host sparse dispatch has verbs-host client"
                                        )
                                    };
                                    if let Some(chunk_tx) = chunk_tx {
                                        runtime.block_on(async {
                                            let stats = real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_streaming(
                                                client,
                                                &batch,
                                                &routes,
                                                global_hidden_payload,
                                                owner_lookup.as_ref(),
                                                request_id_base,
                                                chunk_tx,
                                            )
                                            .await?;
                                            Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
                                                partial_outputs_bf16_by_host: Vec::new(),
                                                global_row_indices_by_host: Vec::new(),
                                                completed_global_row_slices: Vec::new(),
                                                stats,
                                            })
                                        })
                                    } else {
                                        runtime.block_on(
                                            real_full_scheduler_host_batch_set_verbs_host_bf16_payload_dispatch_with_payload_persistent_structural_stats(
                                            client,
                                            &batch,
                                            &routes,
                                            global_hidden_payload,
                                            owner_lookup.as_ref(),
                                            request_id_base,
                                        ),
                                        )
                                    }
                                }
                            };
                            let _ = response_tx.send(result);
                        }
                        SchedulerSparseTcpDispatchMessage::Shutdown => break,
                    }
                }
            })
            .with_context(|| format!("spawning scheduler sparse {} worker", transport.label()))?;
        startup_rx
            .recv()
            .with_context(|| {
                format!(
                    "scheduler sparse {} worker stopped during startup",
                    transport.label()
                )
            })?
            .with_context(|| {
                format!(
                    "pinning scheduler sparse {} worker to configured CPU",
                    transport.label()
                )
            })?;
        Ok(Self {
            target_count,
            transport,
            streaming_responses,
            direct_owner_client,
            tx,
            join: Mutex::new(Some(join)),
        })
    }

    pub(in crate::commands::real_full) fn new_direct_owner(addr: SocketAddr) -> Result<Self> {
        let config = real_full_protocol_v2_transport_config()?;
        let client = VerbsHostProtocolV2PersistentClient::new(addr, config)?;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SchedulerSparseTcpDispatchMessage>();
        let join = thread::Builder::new()
            .name("scheduler-sparse-direct-owner".to_owned())
            .spawn(move || run_scheduler_sparse_direct_owner_dispatch_worker(rx, client))
            .context("spawning scheduler direct Spark owner worker")?;
        Ok(Self {
            target_count: 1,
            transport: RealFullSchedulerSparseDispatchTransport::VerbsHost,
            streaming_responses: false,
            direct_owner_client: None,
            tx,
            join: Mutex::new(Some(join)),
        })
    }

    pub(in crate::commands::real_full) fn new_zero_output() -> Result<Self> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<SchedulerSparseTcpDispatchMessage>();
        let join = thread::Builder::new()
            .name("scheduler-sparse-startup-zero".to_owned())
            .spawn(move || run_scheduler_sparse_zero_output_dispatch_worker(rx))
            .context("spawning scheduler sparse startup zero-output worker")?;
        Ok(Self {
            target_count: 1,
            transport: RealFullSchedulerSparseDispatchTransport::VerbsHost,
            streaming_responses: false,
            direct_owner_client: None,
            tx,
            join: Mutex::new(Some(join)),
        })
    }

    pub(in crate::commands::real_full) fn target_count(&self) -> usize {
        self.target_count
    }

    pub(in crate::commands::real_full) fn transport(
        &self,
    ) -> RealFullSchedulerSparseDispatchTransport {
        self.transport
    }

    fn supports_streaming_responses(&self) -> bool {
        self.streaming_responses
    }

    fn dispatch(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetDispatch> {
        let (response_tx, response_rx) =
            mpsc::channel::<Result<TcpProtocolV2HostBatchSetDispatch>>();
        self.tx
            .send(SchedulerSparseTcpDispatchMessage::Dispatch {
                batch: batch.clone(),
                routes: routes.to_vec(),
                global_hidden_payload: global_hidden_payload.to_vec(),
                request_id_base,
                response_tx,
            })
            .context("sending scheduler persistent sparse TCP dispatch request")?;
        response_rx
            .recv()
            .context("receiving scheduler persistent sparse TCP dispatch response")?
    }

    fn dispatch_payload(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        let response_rx = self.start_dispatch_payload_owned(
            batch.clone(),
            routes.to_vec(),
            global_hidden_payload.to_vec(),
            request_id_base,
            true,
            None,
        )?;
        response_rx
            .recv()
            .context("receiving scheduler persistent sparse TCP BF16 payload dispatch response")?
    }

    fn try_start_direct_owner_payload(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: Vec<u8>,
        request_id_base: u64,
    ) -> Result<DirectOwnerPayloadDispatchStart> {
        let Some(client) = self.direct_owner_client.as_ref() else {
            return Ok(DirectOwnerPayloadDispatchStart::Unavailable(
                global_hidden_payload,
            ));
        };
        real_full_scheduler_verbs_host_try_start_direct_owner_payload_dispatch(
            client,
            batch,
            routes,
            global_hidden_payload,
            request_id_base,
        )
    }

    fn start_dispatch_payload_owned(
        &self,
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        global_hidden_payload: Vec<u8>,
        request_id_base: u64,
        include_contribution_counts: bool,
        chunk_tx: Option<mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
    ) -> Result<mpsc::Receiver<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>> {
        self.start_dispatch_payload(
            batch,
            routes,
            SchedulerSparseTcpDispatchPayload::Owned(global_hidden_payload),
            request_id_base,
            include_contribution_counts,
            chunk_tx,
        )
    }

    fn start_dispatch_payload_shared_slice(
        &self,
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        global_hidden_payload: Arc<Vec<u8>>,
        byte_start: usize,
        byte_end: usize,
        request_id_base: u64,
        chunk_tx: mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>,
    ) -> Result<mpsc::Receiver<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>> {
        self.start_dispatch_payload(
            batch,
            routes,
            SchedulerSparseTcpDispatchPayload::SharedSlice {
                payload: global_hidden_payload,
                byte_start,
                byte_end,
            },
            request_id_base,
            false,
            Some(chunk_tx),
        )
    }

    fn start_dispatch_payload(
        &self,
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        global_hidden_payload: SchedulerSparseTcpDispatchPayload,
        request_id_base: u64,
        include_contribution_counts: bool,
        chunk_tx: Option<mpsc::Sender<VerbsHostProtocolV2HostBatchSetBf16PayloadChunk>>,
    ) -> Result<mpsc::Receiver<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>> {
        let (response_tx, response_rx) =
            mpsc::channel::<Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch>>();
        self.tx
            .send(SchedulerSparseTcpDispatchMessage::DispatchPayload {
                batch,
                routes,
                global_hidden_payload,
                request_id_base,
                include_contribution_counts,
                chunk_tx,
                response_tx,
            })
            .context("sending scheduler persistent sparse TCP BF16 payload dispatch request")?;
        Ok(response_rx)
    }
}

impl Drop for RealFullSchedulerSparseTcpDispatchWorker {
    fn drop(&mut self) {
        let _ = self.tx.send(SchedulerSparseTcpDispatchMessage::Shutdown);
        let Some(join) = self.join.get_mut().ok().and_then(Option::take) else {
            return;
        };
        if thread::current().id() != join.thread().id() {
            let _ = join.join();
        }
    }
}

pub(super) struct RealFullSchedulerSparseTcpRoutedMlpContext {
    request_id_base: u64,
    next_dispatch_index: usize,
    max_global_rows_per_dispatch: usize,
    transport: RealFullSchedulerSparseDispatchTransport,
    dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
    probe: RealFullSchedulerSparseTcpDispatchProbe,
}

impl RealFullSchedulerSparseTcpRoutedMlpContext {
    pub(super) fn new(
        scheduler_iterations_per_sparse_layer: usize,
        targets: Vec<TcpProtocolV2HostBatchTarget>,
        owner_lookup: Option<ExpertOwnerLookup>,
        request_id_base: u64,
    ) -> Result<Self> {
        let dispatch_worker = Arc::new(RealFullSchedulerSparseTcpDispatchWorker::new(
            targets,
            owner_lookup,
        )?);
        Self::with_dispatch_worker(
            scheduler_iterations_per_sparse_layer,
            dispatch_worker,
            request_id_base,
        )
    }

    pub(in crate::commands::real_full) fn with_dispatch_worker(
        scheduler_iterations_per_sparse_layer: usize,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        request_id_base: u64,
    ) -> Result<Self> {
        anyhow::ensure!(
            dispatch_worker.target_count() > 0,
            "scheduler sparse {} routed MLP context requires a worker with at least one target",
            dispatch_worker.transport().label()
        );
        let expected_real_executor_id =
            expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR);
        let max_global_rows_per_dispatch = scheduler_tcp_max_global_rows_per_dispatch();
        let transport = dispatch_worker.transport();
        Ok(Self {
            request_id_base,
            next_dispatch_index: 0,
            max_global_rows_per_dispatch,
            transport,
            dispatch_worker,
            probe: RealFullSchedulerSparseTcpDispatchProbe {
                status: "not-run",
                scope: transport.dispatch_scope(),
                sparse_layers: GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE,
                scheduler_iterations_per_sparse_layer,
                sparse_batches: 0,
                host_batches: 0,
                global_rows: 0,
                host_rows: 0,
                routes: 0,
                route_profiled_wire_batches: 0,
                route_profiled_assignments: 0,
                route_profiled_unique_experts: 0,
                route_profiled_reused_assignments: 0,
                route_profiled_max_expert_load: 0,
                route_profiled_load_square_sum: 0,
                request_wire_bytes: 0,
                response_wire_bytes: 0,
                output_values: 0,
                output_finite_values: 0,
                output_nonzero_values: 0,
                output_checksum: 0.0,
                expected_real_executor_id,
                response_executor_ids_observed: 0,
                real_executor_responses: 0,
                non_real_executor_responses: 0,
                all_responses_real_nvfp4: false,
                passed: false,
            },
        })
    }

    pub(super) fn supports_chunk_wavefront(&self) -> bool {
        self.dispatch_worker.supports_streaming_responses()
    }

    fn dispatch_routed_delta(
        &mut self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
    ) -> Result<TcpProtocolV2HostBatchSetDispatch> {
        let (batch_index, request_id_base) = self.reserve_dispatch_request_ids()?;
        let timing_enabled = sparse_tcp_stage_timing_enabled();
        let dispatch_start = timing_enabled.then(Instant::now);
        let dispatch = if batch.num_rows() > self.max_global_rows_per_dispatch {
            self.dispatch_routed_delta_sliced(batch, routes, global_hidden_payload, request_id_base)
                .with_context(|| {
                    format!(
                        "dispatching scheduler sparse TCP batch in {}-row slices",
                        self.max_global_rows_per_dispatch
                    )
                })?
        } else {
            self.dispatch_routed_delta_once(batch, routes, global_hidden_payload, request_id_base)?
        };
        if timing_enabled {
            let elapsed = elapsed_ms_optional(dispatch_start);
            eprintln!(
                "real_full_sparse_{}_dispatch_timing batch={} layer_id={} rows={} routes={} host_batches={} host_rows={} request_wire_bytes={} response_wire_bytes={} elapsed_ms={:.3}",
                self.transport.label(),
                batch_index,
                batch.layer_id.0,
                batch.num_rows(),
                batch.route_count(),
                dispatch.stats.hosts,
                dispatch.stats.host_rows,
                dispatch.stats.request_wire_bytes,
                dispatch.stats.response_wire_bytes,
                elapsed
            );
        }
        self.record_dispatch(batch, &dispatch)?;
        Ok(dispatch)
    }

    fn dispatch_routed_delta_payload(
        &mut self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        if batch.num_rows() > self.max_global_rows_per_dispatch {
            let (batch_index, request_id_base) = self.reserve_dispatch_request_ids()?;
            let timing_enabled = sparse_tcp_stage_timing_enabled();
            let dispatch_start = timing_enabled.then(Instant::now);
            let dispatch = self
                .dispatch_routed_delta_payload_sliced(
                    batch,
                    routes,
                    global_hidden_payload,
                    request_id_base,
                )
                .with_context(|| {
                    format!(
                        "dispatching scheduler sparse TCP BF16 payload batch in {}-row slices",
                        self.max_global_rows_per_dispatch
                    )
                })?;
            if timing_enabled {
                let elapsed = elapsed_ms_optional(dispatch_start);
                eprintln!(
                    "real_full_sparse_{}_payload_dispatch_timing batch={} layer_id={} rows={} routes={} host_batches={} host_rows={} request_wire_bytes={} response_wire_bytes={} elapsed_ms={:.3}",
                    self.transport.label(),
                    batch_index,
                    batch.layer_id.0,
                    batch.num_rows(),
                    batch.route_count(),
                    dispatch.stats.hosts,
                    dispatch.stats.host_rows,
                    dispatch.stats.request_wire_bytes,
                    dispatch.stats.response_wire_bytes,
                    elapsed
                );
            }
            let batch_shape =
                SchedulerSparseTcpPayloadDispatchBatchShape::from_batch_and_routes(batch, routes);
            self.record_payload_dispatch(batch_shape, &dispatch)?;
            return Ok(dispatch);
        }
        let handle =
            self.start_dispatch_routed_delta_payload(batch, routes, global_hidden_payload)?;
        self.finish_dispatch_routed_delta_payload(handle)
    }

    fn can_start_dispatch_routed_delta_payload(&self, batch: &ExpertBatch) -> bool {
        batch.num_rows() <= self.max_global_rows_per_dispatch
            || (self.max_global_rows_per_dispatch > 0
                && self.dispatch_worker.supports_streaming_responses())
    }

    fn start_dispatch_routed_delta_payload(
        &mut self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
    ) -> Result<SchedulerSparseTcpPayloadDispatchHandle> {
        self.start_dispatch_routed_delta_payload_owned(
            batch.clone(),
            routes.to_vec(),
            global_hidden_payload.to_vec(),
        )
    }

    fn start_dispatch_routed_delta_payload_owned(
        &mut self,
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        mut global_hidden_payload: Vec<u8>,
    ) -> Result<SchedulerSparseTcpPayloadDispatchHandle> {
        anyhow::ensure!(
            self.can_start_dispatch_routed_delta_payload(&batch),
            "scheduler sparse {} BF16 payload async dispatch cannot start rows={} max={}",
            self.transport.label(),
            batch.num_rows(),
            self.max_global_rows_per_dispatch
        );
        if batch.num_rows() > self.max_global_rows_per_dispatch {
            return self.start_dispatch_routed_delta_payload_sliced_owned(
                batch,
                routes,
                global_hidden_payload,
            );
        }
        let (batch_index, request_id_base) = self.reserve_dispatch_request_ids()?;
        let batch_shape =
            SchedulerSparseTcpPayloadDispatchBatchShape::from_batch_and_routes(&batch, &routes);
        let row_sharded_completion_chunks =
            spark_expert_reduction_dispatch_for_rows(batch.num_rows())?
                .is_some_and(|reduction| reduction.row_sharded);
        if self.dispatch_worker.supports_streaming_responses() {
            match self.dispatch_worker.try_start_direct_owner_payload(
                &batch,
                &routes,
                global_hidden_payload,
                request_id_base,
            )? {
                DirectOwnerPayloadDispatchStart::Started(direct_owner_pending) => {
                    return Ok(SchedulerSparseTcpPayloadDispatchHandle {
                        batch: batch_shape,
                        batch_index,
                        started: sparse_tcp_stage_timing_enabled().then(Instant::now),
                        row_sharded_completion_chunks,
                        chunk_rx: None,
                        response_rx: None,
                        direct_owner_pending: Some(direct_owner_pending),
                        sliced_dispatches: Vec::new(),
                        sliced_poll_cursor: 0,
                        deferred_streaming_completion: None,
                    });
                }
                DirectOwnerPayloadDispatchStart::Unavailable(payload) => {
                    global_hidden_payload = payload;
                }
            }
        }
        let (chunk_tx, chunk_rx) = if self.dispatch_worker.supports_streaming_responses() {
            let (chunk_tx, chunk_rx) = mpsc::channel();
            (Some(chunk_tx), Some(chunk_rx))
        } else {
            (None, None)
        };
        let response_rx = self.dispatch_worker.start_dispatch_payload_owned(
            batch,
            routes,
            global_hidden_payload,
            request_id_base,
            false,
            chunk_tx,
        )?;
        Ok(SchedulerSparseTcpPayloadDispatchHandle {
            batch: batch_shape,
            batch_index,
            started: sparse_tcp_stage_timing_enabled().then(Instant::now),
            row_sharded_completion_chunks,
            chunk_rx,
            response_rx: Some(response_rx),
            direct_owner_pending: None,
            sliced_dispatches: Vec::new(),
            sliced_poll_cursor: 0,
            deferred_streaming_completion: None,
        })
    }

    fn start_dispatch_routed_delta_payload_sliced_owned(
        &mut self,
        batch: ExpertBatch,
        routes: Vec<ExpertBatchRoute>,
        global_hidden_payload: Vec<u8>,
    ) -> Result<SchedulerSparseTcpPayloadDispatchHandle> {
        anyhow::ensure!(
            self.dispatch_worker.supports_streaming_responses(),
            "scheduler sparse {} sliced async dispatch requires streamed responses",
            self.transport.label()
        );
        anyhow::ensure!(
            self.max_global_rows_per_dispatch > 0,
            "scheduler sparse sliced async dispatch requires nonzero max rows"
        );
        let expected_hidden_bytes = batch
            .num_rows()
            .checked_mul(batch.hidden_bytes_per_row)
            .context("scheduler sparse sliced async hidden byte count overflow")?;
        anyhow::ensure!(
            global_hidden_payload.len() == expected_hidden_bytes,
            "scheduler sparse sliced async hidden bytes {} did not match expected {expected_hidden_bytes}",
            global_hidden_payload.len()
        );
        let (batch_index, request_id_base) = self.reserve_dispatch_request_ids()?;
        let batch_shape =
            SchedulerSparseTcpPayloadDispatchBatchShape::from_batch_and_routes(&batch, &routes);
        let row_sharded_completion_chunks =
            spark_expert_reduction_dispatch_for_rows(batch.num_rows())?
                .is_some_and(|reduction| reduction.row_sharded);
        let started = sparse_tcp_stage_timing_enabled().then(Instant::now);
        let slice_count = batch.num_rows().div_ceil(self.max_global_rows_per_dispatch);
        let global_hidden_payload = Arc::new(global_hidden_payload);
        let mut sliced_dispatches = Vec::with_capacity(slice_count);
        for (slice_index, row_start) in (0..batch.num_rows())
            .step_by(self.max_global_rows_per_dispatch)
            .enumerate()
        {
            let row_end = (row_start + self.max_global_rows_per_dispatch).min(batch.num_rows());
            let (slice_batch, slice_routes) =
                scheduler_sparse_tcp_batch_slice(&batch, &routes, row_start, row_end)
                    .with_context(|| {
                        format!("building scheduler sparse async row slice {row_start}..{row_end}")
                    })?;
            let byte_start = row_start
                .checked_mul(batch.hidden_bytes_per_row)
                .context("scheduler sparse sliced async byte start overflow")?;
            let byte_end = row_end
                .checked_mul(batch.hidden_bytes_per_row)
                .context("scheduler sparse sliced async byte end overflow")?;
            let request_id = request_id_base
                .checked_add(slice_index as u64 * EXPERT_HOSTS_REQUEST_SLICE_STRIDE)
                .context("scheduler sparse sliced async request ID overflow")?;
            let (chunk_tx, chunk_rx) = mpsc::channel();
            let response_rx = self.dispatch_worker.start_dispatch_payload_shared_slice(
                slice_batch,
                slice_routes,
                Arc::clone(&global_hidden_payload),
                byte_start,
                byte_end,
                request_id,
                chunk_tx,
            )?;
            sliced_dispatches.push(SchedulerSparseTcpPayloadSliceDispatch {
                row_start,
                row_count: row_end - row_start,
                chunk_rx: Some(chunk_rx),
                response_rx: Some(response_rx),
                response: None,
            });
        }
        anyhow::ensure!(
            sliced_dispatches.len() > 1,
            "scheduler sparse sliced async dispatch produced fewer than two slices"
        );
        Ok(SchedulerSparseTcpPayloadDispatchHandle {
            batch: batch_shape,
            batch_index,
            started,
            row_sharded_completion_chunks,
            chunk_rx: None,
            response_rx: None,
            direct_owner_pending: None,
            sliced_dispatches,
            sliced_poll_cursor: 0,
            deferred_streaming_completion: None,
        })
    }

    fn reserve_dispatch_request_ids(&mut self) -> Result<(usize, u64)> {
        let dispatch_index = self.next_dispatch_index;
        self.next_dispatch_index = self
            .next_dispatch_index
            .checked_add(1)
            .context("scheduler sparse dispatch index overflow")?;
        let request_offset = (dispatch_index as u64)
            .checked_mul(EXPERT_HOSTS_REQUEST_STRIDE)
            .context("scheduler sparse request ID offset overflow")?;
        let request_id_base = self
            .request_id_base
            .checked_add(request_offset)
            .context("scheduler sparse request ID base overflow")?;
        Ok((dispatch_index + 1, request_id_base))
    }

    fn finish_dispatch_routed_delta_payload(
        &mut self,
        mut handle: SchedulerSparseTcpPayloadDispatchHandle,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        let layer_id = handle.batch.layer_id.0 as usize;
        if let Some(dispatch) = handle.deferred_streaming_completion.take() {
            return self.finish_payload_dispatch_accounting(handle, dispatch);
        }
        if !handle.sliced_dispatches.is_empty() {
            let mut chunks = Vec::new();
            let mut dispatch = loop {
                match handle.poll_streaming_response(true)? {
                    SchedulerSparseTcpPayloadStreamPoll::Pending => continue,
                    SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk) => chunks.push(chunk),
                    SchedulerSparseTcpPayloadStreamPoll::Complete(dispatch) => break dispatch,
                }
            };
            chunks.sort_by_key(|chunk| chunk.host_index);
            anyhow::ensure!(
                dispatch.partial_outputs_bf16_by_host.is_empty()
                    && dispatch.global_row_indices_by_host.is_empty()
                    && dispatch.completed_global_row_slices.is_empty(),
                "streaming sliced scheduler dispatch unexpectedly returned collected payloads"
            );
            for chunk in chunks {
                anyhow::ensure!(
                    chunk.output_dtype == ExpertV2Dtype::Bf16,
                    "sliced scheduler BF16 dispatch returned {:?}",
                    chunk.output_dtype
                );
                dispatch
                    .partial_outputs_bf16_by_host
                    .push(chunk.partial_output.into_vec());
                dispatch
                    .global_row_indices_by_host
                    .push(chunk.global_row_indices);
                if !chunk.completed_global_row_indices.is_empty() {
                    dispatch
                        .completed_global_row_slices
                        .push(chunk.completed_global_row_indices);
                }
            }
            return self.finish_payload_dispatch_accounting(handle, dispatch);
        }
        let (mut dispatch, direct_owner_chunks) =
            if let Some(pending) = handle.direct_owner_pending.take() {
                let (stats, chunks) = pending
                    .finish()
                    .context("finishing direct Spark-owner BF16 response")?;
                (
                    TcpProtocolV2HostBatchSetBf16PayloadDispatch {
                        partial_outputs_bf16_by_host: Vec::new(),
                        global_row_indices_by_host: Vec::new(),
                        completed_global_row_slices: Vec::new(),
                        stats,
                    },
                    Some(chunks),
                )
            } else {
                let dispatch = handle
                    .response_rx
                    .as_ref()
                    .context("scheduler sparse dispatch is missing its worker response channel")?
                    .recv()
                    .context(
                        "receiving scheduler persistent sparse TCP BF16 payload dispatch response",
                    )??;
                (dispatch, None)
            };
        if let Some(mut chunks) = direct_owner_chunks {
            chunks.sort_by_key(|chunk| chunk.host_index);
            if handle.batch.rows <= 16 && moe_payload_hash_diagnostic_enabled_for_layer(layer_id) {
                log_moe_payload_hashes(layer_id, &chunks);
            }
            for chunk in chunks {
                anyhow::ensure!(
                    chunk.output_dtype == ExpertV2Dtype::Bf16,
                    "direct Spark-owner BF16 dispatch returned {:?}",
                    chunk.output_dtype
                );
                dispatch
                    .partial_outputs_bf16_by_host
                    .push(chunk.partial_output.into_vec());
                dispatch
                    .global_row_indices_by_host
                    .push(chunk.global_row_indices);
                if !chunk.completed_global_row_indices.is_empty() {
                    dispatch
                        .completed_global_row_slices
                        .push(chunk.completed_global_row_indices);
                }
            }
        }
        if let Some(chunk_rx) = handle.chunk_rx.take() {
            anyhow::ensure!(
                dispatch.partial_outputs_bf16_by_host.is_empty()
                    && dispatch.global_row_indices_by_host.is_empty()
                    && dispatch.completed_global_row_slices.is_empty(),
                "streaming scheduler verbs-host dispatch unexpectedly returned collected payloads"
            );
            let mut chunks = chunk_rx.into_iter().collect::<Vec<_>>();
            chunks.sort_by_key(|chunk| chunk.host_index);
            if handle.batch.rows <= 16 && moe_payload_hash_diagnostic_enabled_for_layer(layer_id) {
                log_moe_payload_hashes(layer_id, &chunks);
            }
            for chunk in chunks {
                anyhow::ensure!(
                    chunk.output_dtype == ExpertV2Dtype::Bf16,
                    "collected scheduler verbs-host dispatch requires BF16 chunks, got {:?}",
                    chunk.output_dtype
                );
                dispatch
                    .partial_outputs_bf16_by_host
                    .push(chunk.partial_output.into_vec());
                dispatch
                    .global_row_indices_by_host
                    .push(chunk.global_row_indices);
                if !chunk.completed_global_row_indices.is_empty() {
                    dispatch
                        .completed_global_row_slices
                        .push(chunk.completed_global_row_indices);
                }
            }
        }
        self.finish_payload_dispatch_accounting(handle, dispatch)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_dispatch_routed_delta_payload_collected_low_precision_device_output(
        &mut self,
        layer_id: usize,
        mut handle: SchedulerSparseTcpPayloadDispatchHandle,
        segment: &StreamedSparseBResidualSegment<'_>,
        dst_rows: usize,
        row_width: usize,
        expected_output_dtype: ExpertV2Dtype,
    ) -> Result<(
        TcpProtocolV2HostBatchSetBf16PayloadDispatch,
        DeviceBf16Output,
    )> {
        log_device_bf16_hash(layer_id, "residual", 0, segment.residual)?;
        log_device_bf16_hash(layer_id, "shared_delta", 0, segment.shared_delta)?;
        anyhow::ensure!(
            segment.row_start == 0 && segment.row_count == dst_rows,
            "collected low-precision Sparse-B segment {}+{} did not cover {dst_rows} rows",
            segment.row_start,
            segment.row_count
        );
        let (mut dispatch, mut chunks) = if let Some(pending) = handle.direct_owner_pending.take() {
            let (stats, chunks) = pending
                .finish()
                .context("finishing direct Spark-owner reduced response")?;
            (
                TcpProtocolV2HostBatchSetBf16PayloadDispatch {
                    partial_outputs_bf16_by_host: Vec::new(),
                    global_row_indices_by_host: Vec::new(),
                    completed_global_row_slices: Vec::new(),
                    stats,
                },
                chunks,
            )
        } else {
            let chunk_rx = handle.chunk_rx.take().context(
                "collected low-precision scheduler sparse dispatch is missing its response chunk channel",
            )?;
            let mut chunks = Vec::new();
            let mut completed_rows = vec![false; dst_rows];
            while completed_rows.iter().any(|completed| !*completed) {
                let chunk = chunk_rx.recv().context(
                    "collected low-precision sparse dispatch ended before every row completed",
                )?;
                for row in &chunk.completed_global_row_indices {
                    anyhow::ensure!(
                        *row < dst_rows,
                        "collected low-precision sparse dispatch completed out-of-range row {row}"
                    );
                    completed_rows[*row] = true;
                }
                chunks.push(chunk);
            }
            let dispatch = handle
                .response_rx
                .as_ref()
                .context("collected sparse dispatch is missing its worker response channel")?
                .recv()
                .context("receiving collected low-precision sparse dispatch response")??;
            (dispatch, chunks)
        };
        if dst_rows <= 16 && moe_payload_hash_diagnostic_enabled_for_layer(layer_id) {
            log_moe_payload_hashes(layer_id, &chunks);
        }
        chunks.sort_by_key(|chunk| chunk.host_index);
        let mut completed_global_row_slices = Vec::new();
        for chunk in &chunks {
            if !chunk.completed_global_row_indices.is_empty() {
                completed_global_row_slices.push(chunk.completed_global_row_indices.clone());
            }
        }
        anyhow::ensure!(
            dispatch.partial_outputs_bf16_by_host.is_empty()
                && dispatch.global_row_indices_by_host.is_empty()
                && dispatch.completed_global_row_slices.is_empty(),
            "collected low-precision sparse dispatch unexpectedly returned collected BF16 payloads"
        );
        anyhow::ensure!(
            !chunks.is_empty(),
            "collected low-precision sparse dispatch produced no response chunks"
        );
        let output_row_stride_bytes = chunks[0].output_row_stride_bytes;
        for chunk in &chunks {
            anyhow::ensure!(
                chunk.output_dtype == expected_output_dtype,
                "collected sparse response dtype {:?} did not match requested {:?}",
                chunk.output_dtype,
                expected_output_dtype
            );
            anyhow::ensure!(
                chunk.output_row_stride_bytes == output_row_stride_bytes,
                "collected sparse response row stride {} did not match {}",
                chunk.output_row_stride_bytes,
                output_row_stride_bytes
            );
        }
        let partial_outputs = chunks
            .iter()
            .map(|chunk| chunk.partial_output.as_ref())
            .collect::<Vec<_>>();
        let global_row_indices = chunks
            .iter()
            .map(|chunk| chunk.global_row_indices.as_slice())
            .collect::<Vec<_>>();
        let mut seen_global_rows = BTreeSet::new();
        let has_overlapping_rows = global_row_indices
            .iter()
            .flat_map(|rows| rows.iter().copied())
            .any(|row| !seen_global_rows.insert(row));
        let output = sparse_b_scatter_shared_residual_add_low_precision_device_output(
            segment.residual,
            segment.shared_delta,
            &partial_outputs,
            &global_row_indices,
            expected_output_dtype,
            output_row_stride_bytes,
            has_overlapping_rows,
            dst_rows,
            row_width,
        )
        .context("accumulating collected low-precision Sparse-B response chunks")?;
        log_device_bf16_hash(layer_id, "fused_output", 0, &output)?;
        dispatch.completed_global_row_slices = completed_global_row_slices;
        let dispatch = self.finish_payload_dispatch_accounting(handle, dispatch)?;
        Ok((dispatch, output))
    }

    fn finish_direct_owner_low_precision_device_outputs(
        &mut self,
        layer_id: usize,
        mut handle: SchedulerSparseTcpPayloadDispatchHandle,
        segments: &[StreamedSparseBResidualSegment<'_>],
        dst_rows: usize,
        row_width: usize,
        expected_output_dtype: ExpertV2Dtype,
    ) -> Result<(
        TcpProtocolV2HostBatchSetBf16PayloadDispatch,
        Vec<DeviceBf16Output>,
    )> {
        for (segment_index, segment) in segments.iter().enumerate() {
            log_device_bf16_hash(layer_id, "residual", segment_index, segment.residual)?;
            log_device_bf16_hash(
                layer_id,
                "shared_delta",
                segment_index,
                segment.shared_delta,
            )?;
        }
        let pending = handle
            .direct_owner_pending
            .take()
            .context("direct Spark-owner dispatch is missing its pending response")?;
        let (stats, mut chunks) = pending
            .finish()
            .context("finishing multi-segment direct Spark-owner reduced response")?;
        anyhow::ensure!(
            !chunks.is_empty(),
            "multi-segment direct Spark-owner dispatch produced no response chunks"
        );
        for chunk in &chunks {
            anyhow::ensure!(
                chunk.output_dtype == expected_output_dtype,
                "direct Spark-owner response dtype {:?} did not match requested {:?}",
                chunk.output_dtype,
                expected_output_dtype
            );
        }
        if dst_rows <= 16 && moe_payload_hash_diagnostic_enabled_for_layer(layer_id) {
            log_moe_payload_hashes(layer_id, &chunks);
        }
        chunks.sort_by_key(|chunk| chunk.host_index);
        let completed_global_row_slices = chunks
            .iter()
            .filter_map(|chunk| {
                (!chunk.completed_global_row_indices.is_empty())
                    .then(|| chunk.completed_global_row_indices.clone())
            })
            .collect::<Vec<_>>();
        let mut chunks = chunks.into_iter();
        let outputs = cuda_stream_sparse_b_scatter_shared_residual_add_bf16_device_outputs(
            segments,
            dst_rows,
            row_width,
            || {
                Ok(chunks.next().map(|chunk| {
                    (
                        chunk.partial_output,
                        chunk.global_row_indices,
                        chunk.output_dtype,
                        chunk.output_row_stride_bytes,
                    )
                }))
            },
        )
        .context("accumulating multi-segment direct Spark-owner response")?;
        for (segment_index, output) in outputs.iter().enumerate() {
            log_device_bf16_hash(layer_id, "fused_output", segment_index, output)?;
        }
        let dispatch = TcpProtocolV2HostBatchSetBf16PayloadDispatch {
            partial_outputs_bf16_by_host: Vec::new(),
            global_row_indices_by_host: Vec::new(),
            completed_global_row_slices,
            stats,
        };
        let dispatch = self.finish_payload_dispatch_accounting(handle, dispatch)?;
        Ok((dispatch, outputs))
    }

    fn finish_dispatch_routed_delta_payload_streamed_device_output(
        &mut self,
        layer_id: usize,
        mut handle: SchedulerSparseTcpPayloadDispatchHandle,
        segments: &[StreamedSparseBResidualSegment<'_>],
        dst_rows: usize,
        row_width: usize,
    ) -> Result<(
        TcpProtocolV2HostBatchSetBf16PayloadDispatch,
        Vec<DeviceBf16Output>,
    )> {
        for (segment_index, segment) in segments.iter().enumerate() {
            log_device_bf16_hash(layer_id, "residual", segment_index, segment.residual)?;
            log_device_bf16_hash(
                layer_id,
                "shared_delta",
                segment_index,
                segment.shared_delta,
            )?;
        }
        let chunk_rx = handle.chunk_rx.take().context(
            "streamed scheduler sparse verbs-host dispatch is missing its response chunk channel",
        )?;
        let stream_started = sparse_tcp_stage_timing_enabled().then(Instant::now);
        let mut chunks = Vec::new();
        let mut completed_rows = vec![false; dst_rows];
        while completed_rows.iter().any(|completed| !*completed) {
            let chunk = chunk_rx
                .recv()
                .context("streamed scheduler sparse dispatch ended before every row completed")?;
            for row in &chunk.completed_global_row_indices {
                anyhow::ensure!(
                    *row < dst_rows,
                    "streamed scheduler sparse dispatch completed out-of-range row {row}"
                );
                completed_rows[*row] = true;
            }
            chunks.push(chunk);
        }
        if dst_rows <= 16 && moe_payload_hash_diagnostic_enabled_for_layer(layer_id) {
            log_moe_payload_hashes(layer_id, &chunks);
        }
        chunks.sort_by_key(|chunk| chunk.host_index);
        let completed_global_row_slices = chunks
            .iter()
            .filter_map(|chunk| {
                (!chunk.completed_global_row_indices.is_empty())
                    .then(|| chunk.completed_global_row_indices.clone())
            })
            .collect::<Vec<_>>();
        let mut chunks = chunks.into_iter();
        let outputs_result = cuda_stream_sparse_b_scatter_shared_residual_add_bf16_device_outputs(
            segments,
            dst_rows,
            row_width,
            || {
                Ok(chunks.next().map(|chunk| {
                    (
                        chunk.partial_output,
                        chunk.global_row_indices,
                        chunk.output_dtype,
                        chunk.output_row_stride_bytes,
                    )
                }))
            },
        );
        let outputs = match outputs_result {
            Ok(outputs) => outputs,
            Err(stream_error) => match handle
                .response_rx
                .as_ref()
                .context("streamed sparse dispatch is missing its worker response channel")?
                .recv()
            {
                Ok(Err(dispatch_error)) => {
                    return Err(dispatch_error).context(
                        "streamed scheduler sparse verbs-host dispatch failed before producing response chunks",
                    );
                }
                Ok(Ok(_)) => {
                    return Err(stream_error).context(
                        "accumulating streamed scheduler sparse verbs-host response chunks",
                    );
                }
                Err(response_error) => {
                    return Err(stream_error).context(format!(
                        "accumulating streamed scheduler sparse verbs-host response chunks; dispatch response channel failed: {response_error}"
                    ));
                }
            },
        };
        for (segment_index, output) in outputs.iter().enumerate() {
            log_device_bf16_hash(layer_id, "fused_output", segment_index, output)?;
        }
        if let Some(started) = stream_started {
            eprintln!(
                "real_full_sparse_verbs_host_streamed_b_timing layer_id={} rows={} elapsed_ms={:.3}",
                handle.batch.layer_id.0,
                handle.batch.rows,
                elapsed_ms(started)
            );
        }
        let mut dispatch = handle
            .response_rx
            .as_ref()
            .context("streamed sparse dispatch is missing its worker response channel")?
            .recv()
            .context("receiving streamed scheduler sparse verbs-host dispatch response")??;
        anyhow::ensure!(
            dispatch.partial_outputs_bf16_by_host.is_empty()
                && dispatch.global_row_indices_by_host.is_empty()
                && dispatch.completed_global_row_slices.is_empty(),
            "streamed scheduler verbs-host dispatch unexpectedly returned collected payloads"
        );
        dispatch.completed_global_row_slices = completed_global_row_slices;
        let dispatch = self.finish_payload_dispatch_accounting(handle, dispatch)?;
        Ok((dispatch, outputs))
    }

    fn finish_payload_dispatch_accounting(
        &mut self,
        handle: SchedulerSparseTcpPayloadDispatchHandle,
        dispatch: TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        if let Some(started) = handle.started {
            let elapsed = elapsed_ms(started);
            eprintln!(
                "real_full_sparse_{}_payload_dispatch_timing batch={} layer_id={} rows={} routes={} unique_experts={} host_batches={} host_rows={} request_wire_bytes={} response_wire_bytes={} elapsed_ms={:.3}",
                self.transport.label(),
                handle.batch_index,
                handle.batch.layer_id.0,
                handle.batch.rows,
                handle.batch.routes,
                handle.batch.unique_experts,
                dispatch.stats.hosts,
                dispatch.stats.host_rows,
                dispatch.stats.request_wire_bytes,
                dispatch.stats.response_wire_bytes,
                elapsed
            );
        }
        self.record_payload_dispatch(handle.batch, &dispatch)?;
        Ok(dispatch)
    }

    fn dispatch_routed_delta_once(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetDispatch> {
        self.dispatch_worker
            .dispatch(batch, routes, global_hidden_payload, request_id_base)
    }

    fn dispatch_routed_delta_payload_once(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        self.dispatch_worker
            .dispatch_payload(batch, routes, global_hidden_payload, request_id_base)
    }

    fn dispatch_routed_delta_payload_sliced(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
        anyhow::ensure!(
            self.max_global_rows_per_dispatch > 0,
            "scheduler sparse TCP BF16 payload split dispatch requires nonzero max rows"
        );
        let expected_hidden_bytes = batch
            .num_rows()
            .checked_mul(batch.hidden_bytes_per_row)
            .context("scheduler sparse TCP BF16 payload split hidden byte count overflow")?;
        anyhow::ensure!(
            global_hidden_payload.len() == expected_hidden_bytes,
            "scheduler sparse TCP BF16 payload split bytes {} did not match expected {expected_hidden_bytes}",
            global_hidden_payload.len()
        );

        let mut partial_outputs_bf16_by_host = Vec::new();
        let mut global_row_indices_by_host = Vec::new();
        let mut completed_global_row_slices = Vec::new();
        let mut contribution_counts = vec![0_usize; batch.num_rows()];
        let mut host_batches = 0_usize;
        let mut host_rows = 0_usize;
        let mut route_count = 0_usize;
        let mut request_wire_bytes = 0_usize;
        let mut response_wire_bytes = 0_usize;
        let mut response_executor_ids = Vec::new();
        let mut graph_pool_leases = 0_usize;
        let mut graph_pool_fixed_buffer_bytes = 0_usize;
        let mut graph_pool_active_rows = 0_usize;
        let mut graph_pool_active_routes = 0_usize;
        let mut graph_pool_active_expert_tiles = 0_usize;
        let mut graph_pool_bucket_rows = Vec::new();

        for (slice_index, row_start) in (0..batch.num_rows())
            .step_by(self.max_global_rows_per_dispatch)
            .enumerate()
        {
            let row_end = (row_start + self.max_global_rows_per_dispatch).min(batch.num_rows());
            let (slice_batch, slice_routes) = scheduler_sparse_tcp_batch_slice(
                batch, routes, row_start, row_end,
            )
            .with_context(|| {
                format!(
                    "building scheduler sparse TCP BF16 payload row slice {row_start}..{row_end}"
                )
            })?;
            let byte_start = row_start
                .checked_mul(batch.hidden_bytes_per_row)
                .context("scheduler sparse TCP BF16 payload split byte start overflow")?;
            let byte_end = row_end
                .checked_mul(batch.hidden_bytes_per_row)
                .context("scheduler sparse TCP BF16 payload split byte end overflow")?;
            let slice_request_id_base =
                request_id_base + slice_index as u64 * EXPERT_HOSTS_REQUEST_SLICE_STRIDE;
            let dispatch = self
                .dispatch_routed_delta_payload_once(
                    &slice_batch,
                    &slice_routes,
                    &global_hidden_payload[byte_start..byte_end],
                    slice_request_id_base,
                )
                .with_context(|| {
                    format!(
                        "dispatching scheduler sparse TCP BF16 payload row slice {row_start}..{row_end}"
                    )
                })?;
            anyhow::ensure!(
                dispatch.stats.contribution_counts.len() == slice_batch.num_rows(),
                "scheduler sparse TCP BF16 payload split contribution counts {} did not match slice rows {}",
                dispatch.stats.contribution_counts.len(),
                slice_batch.num_rows()
            );
            for (local_row, contribution_count) in
                dispatch.stats.contribution_counts.iter().enumerate()
            {
                contribution_counts[row_start + local_row] = *contribution_count;
            }
            host_batches += dispatch.stats.hosts;
            host_rows += dispatch.stats.host_rows;
            route_count += dispatch.stats.routes;
            request_wire_bytes += dispatch.stats.request_wire_bytes;
            response_wire_bytes += dispatch.stats.response_wire_bytes;
            response_executor_ids.extend(dispatch.stats.response_executor_ids);
            graph_pool_leases += dispatch.stats.graph_pool_leases;
            graph_pool_fixed_buffer_bytes += dispatch.stats.graph_pool_fixed_buffer_bytes;
            graph_pool_active_rows += dispatch.stats.graph_pool_active_rows;
            graph_pool_active_routes += dispatch.stats.graph_pool_active_routes;
            graph_pool_active_expert_tiles += dispatch.stats.graph_pool_active_expert_tiles;
            graph_pool_bucket_rows.extend(dispatch.stats.graph_pool_bucket_rows);
            partial_outputs_bf16_by_host.extend(dispatch.partial_outputs_bf16_by_host);
            global_row_indices_by_host.extend(dispatch.global_row_indices_by_host.into_iter().map(
                |row_indices| {
                    row_indices
                        .into_iter()
                        .map(|row_index| row_start + row_index)
                        .collect::<Vec<_>>()
                },
            ));
            completed_global_row_slices.extend(
                dispatch
                    .completed_global_row_slices
                    .into_iter()
                    .map(|row_indices| {
                        row_indices
                            .into_iter()
                            .map(|row_index| row_start + row_index)
                            .collect::<Vec<_>>()
                    }),
            );
        }

        let output_values = batch
            .num_rows()
            .checked_mul(batch.hidden_dim)
            .context("scheduler sparse TCP BF16 payload split output value count overflow")?;
        Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
            partial_outputs_bf16_by_host,
            global_row_indices_by_host,
            completed_global_row_slices,
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: host_batches,
                global_rows: batch.num_rows(),
                host_rows,
                routes: route_count,
                output_dim: batch.hidden_dim,
                output_values,
                request_wire_bytes,
                response_wire_bytes,
                response_executor_ids,
                contribution_counts,
                output_checksum: 0.0,
                graph_pool_leases,
                graph_pool_fixed_buffer_bytes,
                graph_pool_active_rows,
                graph_pool_active_routes,
                graph_pool_active_expert_tiles,
                graph_pool_bucket_rows,
            },
        })
    }

    fn dispatch_routed_delta_sliced(
        &self,
        batch: &ExpertBatch,
        routes: &[ExpertBatchRoute],
        global_hidden_payload: &[u8],
        request_id_base: u64,
    ) -> Result<TcpProtocolV2HostBatchSetDispatch> {
        anyhow::ensure!(
            self.max_global_rows_per_dispatch > 0,
            "scheduler sparse TCP split dispatch requires nonzero max rows"
        );
        let expected_hidden_bytes = batch
            .num_rows()
            .checked_mul(batch.hidden_bytes_per_row)
            .context("scheduler sparse TCP split hidden byte count overflow")?;
        anyhow::ensure!(
            global_hidden_payload.len() == expected_hidden_bytes,
            "scheduler sparse TCP split payload bytes {} did not match expected {expected_hidden_bytes}",
            global_hidden_payload.len()
        );
        let output_values = batch
            .num_rows()
            .checked_mul(batch.hidden_dim)
            .context("scheduler sparse TCP split output value count overflow")?;
        let mut values = vec![0.0_f32; output_values];
        let mut contribution_counts = vec![0_usize; batch.num_rows()];
        let mut partial_outputs_bf16_by_host = Vec::new();
        let mut host_batches = 0_usize;
        let mut host_rows = 0_usize;
        let mut route_count = 0_usize;
        let mut request_wire_bytes = 0_usize;
        let mut response_wire_bytes = 0_usize;
        let mut response_executor_ids = Vec::new();
        let mut graph_pool_leases = 0_usize;
        let mut graph_pool_fixed_buffer_bytes = 0_usize;
        let mut graph_pool_active_rows = 0_usize;
        let mut graph_pool_active_routes = 0_usize;
        let mut graph_pool_active_expert_tiles = 0_usize;
        let mut graph_pool_bucket_rows = Vec::new();

        for (slice_index, row_start) in (0..batch.num_rows())
            .step_by(self.max_global_rows_per_dispatch)
            .enumerate()
        {
            let row_end = (row_start + self.max_global_rows_per_dispatch).min(batch.num_rows());
            let (slice_batch, slice_routes) =
                scheduler_sparse_tcp_batch_slice(batch, routes, row_start, row_end).with_context(
                    || format!("building scheduler sparse TCP row slice {row_start}..{row_end}"),
                )?;
            let byte_start = row_start
                .checked_mul(batch.hidden_bytes_per_row)
                .context("scheduler sparse TCP split byte start overflow")?;
            let byte_end = row_end
                .checked_mul(batch.hidden_bytes_per_row)
                .context("scheduler sparse TCP split byte end overflow")?;
            let slice_request_id_base =
                request_id_base + slice_index as u64 * EXPERT_HOSTS_REQUEST_SLICE_STRIDE;
            let dispatch = self
                .dispatch_routed_delta_once(
                    &slice_batch,
                    &slice_routes,
                    &global_hidden_payload[byte_start..byte_end],
                    slice_request_id_base,
                )
                .with_context(|| {
                    format!("dispatching scheduler sparse TCP row slice {row_start}..{row_end}")
                })?;
            anyhow::ensure!(
                dispatch.accumulation.contribution_counts.len() == slice_batch.num_rows(),
                "scheduler sparse TCP split contribution counts {} did not match slice rows {}",
                dispatch.accumulation.contribution_counts.len(),
                slice_batch.num_rows()
            );
            anyhow::ensure!(
                dispatch.accumulation.values.len() == slice_batch.num_rows() * batch.hidden_dim,
                "scheduler sparse TCP split output values {} did not match slice rows {} * hidden {}",
                dispatch.accumulation.values.len(),
                slice_batch.num_rows(),
                batch.hidden_dim
            );
            for local_row in 0..slice_batch.num_rows() {
                let src_start = local_row * batch.hidden_dim;
                let src_end = src_start + batch.hidden_dim;
                let dst_row = row_start + local_row;
                let dst_start = dst_row * batch.hidden_dim;
                let dst_end = dst_start + batch.hidden_dim;
                values[dst_start..dst_end]
                    .copy_from_slice(&dispatch.accumulation.values[src_start..src_end]);
                contribution_counts[dst_row] = dispatch.accumulation.contribution_counts[local_row];
            }
            host_batches += dispatch.stats.hosts;
            host_rows += dispatch.stats.host_rows;
            route_count += dispatch.stats.routes;
            request_wire_bytes += dispatch.stats.request_wire_bytes;
            response_wire_bytes += dispatch.stats.response_wire_bytes;
            response_executor_ids.extend(dispatch.stats.response_executor_ids);
            graph_pool_leases += dispatch.stats.graph_pool_leases;
            graph_pool_fixed_buffer_bytes += dispatch.stats.graph_pool_fixed_buffer_bytes;
            graph_pool_active_rows += dispatch.stats.graph_pool_active_rows;
            graph_pool_active_routes += dispatch.stats.graph_pool_active_routes;
            graph_pool_active_expert_tiles += dispatch.stats.graph_pool_active_expert_tiles;
            graph_pool_bucket_rows.extend(dispatch.stats.graph_pool_bucket_rows);
            partial_outputs_bf16_by_host.extend(dispatch.partial_outputs_bf16_by_host);
        }

        let output_checksum = values.iter().map(|value| *value as f64).sum::<f64>();
        let stats_contribution_counts = contribution_counts.clone();
        Ok(TcpProtocolV2HostBatchSetDispatch {
            accumulation: ExpertHostBatchSetAccumulation {
                values,
                contribution_counts,
            },
            partial_outputs_bf16_by_host,
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: host_batches,
                global_rows: batch.num_rows(),
                host_rows,
                routes: route_count,
                output_dim: batch.hidden_dim,
                output_values,
                request_wire_bytes,
                response_wire_bytes,
                response_executor_ids,
                contribution_counts: stats_contribution_counts,
                output_checksum,
                graph_pool_leases,
                graph_pool_fixed_buffer_bytes,
                graph_pool_active_rows,
                graph_pool_active_routes,
                graph_pool_active_expert_tiles,
                graph_pool_bucket_rows,
            },
        })
    }

    fn record_dispatch(
        &mut self,
        batch: &ExpertBatch,
        dispatch: &TcpProtocolV2HostBatchSetDispatch,
    ) -> Result<()> {
        anyhow::ensure!(
            dispatch.stats.global_rows == batch.num_rows(),
            "scheduler sparse {} routed MLP dispatch global rows {} did not match batch rows {}",
            self.transport.label(),
            dispatch.stats.global_rows,
            batch.num_rows()
        );
        anyhow::ensure!(
            dispatch.stats.routes == scheduler_dispatched_route_count(batch.route_count())?,
            "scheduler sparse {} routed MLP dispatch routes {} did not match batch routes {}",
            self.transport.label(),
            dispatch.stats.routes,
            batch.route_count()
        );
        anyhow::ensure!(
            dispatch.stats.output_values == batch.num_rows() * batch.hidden_dim,
            "scheduler sparse {} routed MLP dispatch output values {} did not match {} rows * {} hidden",
            self.transport.label(),
            dispatch.stats.output_values,
            batch.num_rows(),
            batch.hidden_dim
        );

        self.probe.sparse_batches += 1;
        self.probe.host_batches += dispatch.stats.hosts;
        self.probe.global_rows += dispatch.stats.global_rows;
        self.probe.host_rows += dispatch.stats.host_rows;
        self.probe.routes += batch.route_count();
        self.probe.request_wire_bytes += dispatch.stats.request_wire_bytes;
        self.probe.response_wire_bytes += dispatch.stats.response_wire_bytes;
        self.probe.output_values += dispatch.stats.output_values;
        self.probe.response_executor_ids_observed += dispatch.stats.response_executor_ids.len();
        let real_responses = dispatch
            .stats
            .response_executor_ids
            .iter()
            .filter(|executor_id| **executor_id == self.probe.expected_real_executor_id)
            .count();
        self.probe.real_executor_responses += real_responses;
        self.probe.non_real_executor_responses += dispatch
            .stats
            .response_executor_ids
            .len()
            .saturating_sub(real_responses);
        Ok(())
    }

    fn record_payload_dispatch(
        &mut self,
        batch: SchedulerSparseTcpPayloadDispatchBatchShape,
        dispatch: &TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    ) -> Result<()> {
        anyhow::ensure!(
            dispatch.stats.global_rows == batch.rows,
            "scheduler sparse {} routed MLP payload dispatch global rows {} did not match batch rows {}",
            self.transport.label(),
            dispatch.stats.global_rows,
            batch.rows
        );
        anyhow::ensure!(
            dispatch.stats.routes == scheduler_dispatched_route_count(batch.routes)?,
            "scheduler sparse {} routed MLP payload dispatch routes {} did not match batch routes {}",
            self.transport.label(),
            dispatch.stats.routes,
            batch.routes
        );
        anyhow::ensure!(
            dispatch.stats.output_values == batch.rows * batch.hidden_dim,
            "scheduler sparse {} routed MLP payload dispatch output values {} did not match {} rows * {} hidden",
            self.transport.label(),
            dispatch.stats.output_values,
            batch.rows,
            batch.hidden_dim
        );
        if dispatch.partial_outputs_bf16_by_host.is_empty()
            && dispatch.global_row_indices_by_host.is_empty()
        {
            let mut completed_rows = vec![false; batch.rows];
            for completed_slice in &dispatch.completed_global_row_slices {
                for row_index in completed_slice {
                    let completed = completed_rows.get_mut(*row_index).with_context(|| {
                        format!(
                            "streamed scheduler sparse payload completion row {row_index} exceeds batch rows {}",
                            batch.rows
                        )
                    })?;
                    anyhow::ensure!(
                        !*completed,
                        "streamed scheduler sparse payload completed row {row_index} twice"
                    );
                    *completed = true;
                }
            }
            anyhow::ensure!(
                completed_rows.iter().all(|completed| *completed),
                "streamed scheduler sparse payload did not complete every batch row"
            );
        } else {
            let payload_rows = dispatch.global_row_indices_by_host.iter().try_fold(
                0_usize,
                |rows, row_indices| {
                    rows.checked_add(row_indices.len())
                        .context("scheduler sparse TCP payload row count overflow")
                },
            )?;
            validate_bf16_payload_byte_count(
                &dispatch.partial_outputs_bf16_by_host,
                payload_rows,
                batch.hidden_dim,
            )?;
        }

        self.probe.sparse_batches += 1;
        self.probe.host_batches += dispatch.stats.hosts;
        self.probe.global_rows += dispatch.stats.global_rows;
        self.probe.host_rows += dispatch.stats.host_rows;
        self.probe.routes += batch.routes;
        if batch.route_profiled {
            self.probe.route_profiled_wire_batches += 1;
            self.probe.route_profiled_assignments += batch.routes;
            self.probe.route_profiled_unique_experts += batch.unique_experts;
            self.probe.route_profiled_reused_assignments +=
                batch.routes.saturating_sub(batch.unique_experts);
            self.probe.route_profiled_max_expert_load = self
                .probe
                .route_profiled_max_expert_load
                .max(batch.max_expert_load);
            self.probe.route_profiled_load_square_sum += batch.expert_load_square_sum;
        }
        self.probe.request_wire_bytes += dispatch.stats.request_wire_bytes;
        self.probe.response_wire_bytes += dispatch.stats.response_wire_bytes;
        self.probe.output_values += dispatch.stats.output_values;
        self.probe.response_executor_ids_observed += dispatch.stats.response_executor_ids.len();
        let real_responses = dispatch
            .stats
            .response_executor_ids
            .iter()
            .filter(|executor_id| **executor_id == self.probe.expected_real_executor_id)
            .count();
        self.probe.real_executor_responses += real_responses;
        self.probe.non_real_executor_responses += dispatch
            .stats
            .response_executor_ids
            .len()
            .saturating_sub(real_responses);
        Ok(())
    }

    fn finish(mut self) -> RealFullSchedulerSparseTcpDispatchProbe {
        let expected_sparse_batches =
            self.probe.sparse_layers * self.probe.scheduler_iterations_per_sparse_layer;
        self.probe.passed = self.probe.sparse_batches == expected_sparse_batches
            && self.probe.host_batches > 0
            && self.probe.global_rows > 0
            && self.probe.host_rows > 0
            && self.probe.routes == self.probe.global_rows * GLM52_TOP_K
            && self.probe.request_wire_bytes > 0
            && self.probe.response_wire_bytes > 0
            && self.probe.output_values == self.probe.global_rows * GLM52_HIDDEN_SIZE
            && self.probe.response_executor_ids_observed == self.probe.host_batches;
        self.probe.all_responses_real_nvfp4 = self.probe.host_batches > 0
            && self.probe.response_executor_ids_observed == self.probe.host_batches
            && self.probe.real_executor_responses == self.probe.host_batches
            && self.probe.non_real_executor_responses == 0;
        self.probe.status = if self.probe.passed {
            self.transport.passed_status()
        } else {
            self.transport.blocked_status()
        };
        self.probe
    }
}

const EXPERT_HOSTS_REQUEST_STRIDE: u64 = 65536;
const EXPERT_HOSTS_REQUEST_SLICE_STRIDE: u64 = 16;

fn scheduler_tcp_max_global_rows_per_dispatch() -> usize {
    match env::var("GLMRT_REAL_FULL_SCHEDULER_TCP_MAX_GLOBAL_ROWS") {
        Ok(value) => match value.parse::<usize>() {
            Ok(0) => usize::MAX,
            Ok(rows) => rows,
            Err(_) => DEFAULT_SCHEDULER_TCP_MAX_GLOBAL_ROWS,
        },
        Err(_) => DEFAULT_SCHEDULER_TCP_MAX_GLOBAL_ROWS,
    }
}

fn sparse_tcp_stage_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(SPARSE_TCP_STAGE_TIMING_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn dspark_route_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        [
            "GLMRT_REAL_FULL_DSPARK_TRACE",
            "GLMRT_REAL_FULL_DSPARK_ROUTE_PROFILE",
        ]
        .into_iter()
        .any(|name| {
            env::var(name)
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
    })
}

fn moe_payload_hash_diagnostic_enabled_for_layer(layer_id: usize) -> bool {
    static LAYERS: OnceLock<Option<BTreeSet<usize>>> = OnceLock::new();
    let layers = LAYERS.get_or_init(|| {
        let value = env::var(MOE_PAYLOAD_HASH_DIAGNOSTIC_ENV).ok()?;
        if matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ) {
            return Some(BTreeSet::new());
        }
        let layers = value
            .split(',')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .collect::<BTreeSet<_>>();
        (!layers.is_empty()).then_some(layers)
    });
    layers
        .as_ref()
        .map(|layers| layers.is_empty() || layers.contains(&layer_id))
        .unwrap_or(false)
}

fn layer_boundary_dump_config() -> Option<(&'static Path, &'static BTreeSet<usize>)> {
    static CONFIG: OnceLock<Option<(PathBuf, BTreeSet<usize>)>> = OnceLock::new();
    CONFIG
        .get_or_init(|| {
            let directory = env::var_os(LAYER_BOUNDARY_DUMP_DIR_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)?;
            let mut layer_ids = env::var(LAYER_BOUNDARY_DUMP_LAYER_ENV)
                .ok()
                .into_iter()
                .flat_map(|value| {
                    value
                        .split(',')
                        .filter_map(|part| part.trim().parse::<usize>().ok())
                        .collect::<Vec<_>>()
                })
                .collect::<BTreeSet<_>>();
            if layer_ids.is_empty() {
                layer_ids.insert(0);
            }
            Some((directory, layer_ids))
        })
        .as_ref()
        .map(|(directory, layer_ids)| (directory.as_path(), layer_ids))
}

fn dump_layer_boundary_device_hidden(
    layer_id: usize,
    kind: RowSourceKind,
    token_start: u64,
    row_count: usize,
    stage: &str,
    segments: &BTreeMap<DeviceHiddenSegmentKey, DeviceBf16Output>,
    desired: DeviceHiddenSegmentKey,
) -> Result<()> {
    let Some((directory, target_layer_ids)) = layer_boundary_dump_config() else {
        return Ok(());
    };
    if !target_layer_ids.contains(&layer_id) || row_count > 16 {
        return Ok(());
    }
    let (segment_key, output) = segments
        .iter()
        .filter(|(key, _)| key.byte_start <= desired.byte_start && key.byte_end >= desired.byte_end)
        .min_by_key(|(key, _)| key.byte_end - key.byte_start)
        .with_context(|| {
            format!(
                "finding resident layer-boundary segment for bytes {}..{}",
                desired.byte_start, desired.byte_end
            )
        })?;
    fs::create_dir_all(directory).with_context(|| {
        format!(
            "creating layer-boundary diagnostic directory {}",
            directory.display()
        )
    })?;
    let segment_bytes = output
        .rows
        .checked_mul(output.values_per_row * std::mem::size_of::<u16>())
        .context("layer-boundary resident segment byte count overflow")?;
    anyhow::ensure!(
        segment_bytes == segment_key.byte_end - segment_key.byte_start,
        "layer-boundary resident segment bytes do not match its key"
    );
    let expected_bytes = row_count
        .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>())
        .context("layer-boundary dump byte count overflow")?;
    anyhow::ensure!(
        desired.byte_end - desired.byte_start == expected_bytes,
        "layer-boundary requested range does not match source rows"
    );
    let offset = desired.byte_start - segment_key.byte_start;
    let view = device_buffer_byte_view(
        output.buffer(),
        offset,
        expected_bytes,
        "layer-boundary diagnostic source",
    )?;
    let mut bytes = vec![0_u8; expected_bytes];
    cuda_native_library()?
        .copy_d2h(&mut bytes, view)
        .with_context(|| {
            format!(
                "copying layer {layer_id} {stage} diagnostic for {:?} rows {}..{}",
                kind,
                token_start,
                token_start + row_count as u64
            )
        })?;
    let kind_label = format!("{kind:?}").to_ascii_lowercase();
    let path = directory.join(format!(
        "layer_{layer_id:02}_{kind}_start_{}_rows_{}_{stage}.bf16",
        token_start,
        row_count,
        kind = kind_label,
    ));
    fs::write(&path, &bytes)
        .with_context(|| format!("writing layer-boundary diagnostic {}", path.display()))?;
    eprintln!(
        "real_full_layer_boundary_dump layer_id={} kind={:?} token_start={} rows={} stage={} backend={} bytes={} fnv1a64={:016x} path={}",
        layer_id,
        kind,
        token_start,
        row_count,
        stage,
        output.backend,
        bytes.len(),
        fnv1a64(&bytes),
        path.display(),
    );
    Ok(())
}

fn dump_layer_boundary_token_ids(
    kind: RowSourceKind,
    row_start: usize,
    token_ids: &[usize],
) -> Result<()> {
    let Some((directory, _)) = layer_boundary_dump_config() else {
        return Ok(());
    };
    fs::create_dir_all(directory).with_context(|| {
        format!(
            "creating layer-boundary diagnostic directory {}",
            directory.display()
        )
    })?;
    let kind = format!("{kind:?}").to_ascii_lowercase();
    let path = directory.join(format!(
        "tokens_{kind}_start_{row_start}_rows_{}.txt",
        token_ids.len()
    ));
    let text = token_ids
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    fs::write(&path, format!("{text}\n"))
        .with_context(|| format!("writing layer-boundary token IDs {}", path.display()))?;
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn log_moe_request_row_zero(
    layer_id: usize,
    batch: &ExpertBatch,
    hidden_payload: &[u8],
    routes: &[ExpertBatchRoute],
) {
    if !moe_payload_hash_diagnostic_enabled_for_layer(layer_id) || batch.num_rows() > 16 {
        return;
    }
    let row_stride = batch.hidden_bytes_per_row;
    let Some(row_zero) = hidden_payload.get(..row_stride) else {
        return;
    };
    let route_summary = routes
        .iter()
        .filter(|route| route.row_index == 0)
        .map(|route| format!("{}:{:08x}", route.expert_id, route.gate_weight.to_bits()))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        "real_full_moe_request_row_hash layer_id={} global_row=0 rows={} bytes={} dtype={:?} fnv1a64={:016x} routes={}",
        layer_id,
        batch.num_rows(),
        row_zero.len(),
        batch.hidden_dtype,
        fnv1a64(row_zero),
        route_summary,
    );
}

fn log_device_bf16_hash(
    layer_id: usize,
    stage: &str,
    segment_index: usize,
    output: &DeviceBf16Output,
) -> Result<()> {
    if !moe_payload_hash_diagnostic_enabled_for_layer(layer_id) || output.rows > 16 {
        return Ok(());
    }
    let bytes = output.copy_to_host_bytes().with_context(|| {
        format!(
            "copying coordinator BF16 diagnostic stage {stage} for layer {layer_id} segment {segment_index}"
        )
    })?;
    eprintln!(
        "real_full_coordinator_bf16_hash layer_id={} stage={} segment={} rows={} values_per_row={} bytes={} fnv1a64={:016x}",
        layer_id,
        stage,
        segment_index,
        output.rows,
        output.values_per_row,
        bytes.len(),
        fnv1a64(&bytes),
    );
    Ok(())
}

fn log_moe_payload_hashes(
    layer_id: usize,
    chunks: &[VerbsHostProtocolV2HostBatchSetBf16PayloadChunk],
) {
    for chunk in chunks {
        eprintln!(
            "real_full_moe_payload_hash layer_id={} host_index={} rows={} bytes={} dtype={:?} stride={} fnv1a64={:016x}",
            layer_id,
            chunk.host_index,
            chunk.global_row_indices.len(),
            chunk.partial_output.as_ref().len(),
            chunk.output_dtype,
            chunk.output_row_stride_bytes,
            fnv1a64(chunk.partial_output.as_ref()),
        );
        let payload = chunk.partial_output.as_ref();
        for (row_offset, global_row_index) in chunk.global_row_indices.iter().copied().enumerate() {
            let byte_start = row_offset * chunk.output_row_stride_bytes;
            let byte_end = byte_start + chunk.output_row_stride_bytes;
            if let Some(row) = payload.get(byte_start..byte_end) {
                eprintln!(
                    "real_full_moe_payload_row_hash layer_id={} host_index={} global_row={} bytes={} dtype={:?} fnv1a64={:016x}",
                    layer_id,
                    chunk.host_index,
                    global_row_index,
                    row.len(),
                    chunk.output_dtype,
                    fnv1a64(row),
                );
            }
        }
    }
}

fn sparse_prefill_frontier_timing_sample(layer_id: usize, rows: usize) -> bool {
    static SAMPLES: AtomicUsize = AtomicUsize::new(0);
    if layer_id != GLM52_FIRST_K_DENSE_REPLACE || rows != 256 {
        return false;
    }
    SAMPLES.fetch_add(1, Ordering::Relaxed) < 2
}

fn b12x_packed_hidden_exchange_enabled() -> bool {
    env::var(B12X_PACKED_HIDDEN_EXCHANGE_ENV)
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or_else(|_| {
            env::var(B12X_COORDINATOR_ENV)
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
        })
}

fn b12x_packed_hidden_exchange_enabled_for_layer(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> Result<bool> {
    if layer_id == glmrt_core::GLM52_MTP_LAYER_ID
        && crate::commands::real_full::mtp_expert::mtp_bf16_experts_enabled()?
    {
        let tensor_name = format!("model.layers.{layer_id}.mlp.experts.0.gate_proj.weight");
        let source = catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .with_context(|| format!("finding retained BF16 MTP tensor {tensor_name}"))?;
        if source.dtype == DType::Bf16 {
            return Ok(false);
        }
    }
    Ok(b12x_packed_hidden_exchange_enabled())
}

pub(super) fn rolling_sparse_packs_enabled() -> bool {
    env::var(ROLLING_SPARSE_PACKS_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub(super) fn rolling_sparse_packs_supported_for_rows(rows: usize) -> bool {
    (rolling_sparse_packs_enabled() || rows >= ROLLING_SPARSE_REQUIRED_MIN_ROWS)
        && (ROLLING_SPARSE_LOOKAHEAD_ROWS..=ROLLING_SPARSE_MAX_ROWS).contains(&rows)
}

pub(super) fn bounded_long_prefill_wavefront_required(rows: usize) -> bool {
    (ROLLING_SPARSE_REQUIRED_MIN_ROWS..=ROLLING_SPARSE_MAX_ROWS).contains(&rows)
}

fn rolling_sparse_physical_dispatches_per_layer(rows: usize) -> usize {
    rows.div_ceil(ROLLING_SPARSE_PACK_ROWS)
}

pub(super) fn rolling_sparse_dispatches_per_layer_for_rows(rows: usize) -> Option<usize> {
    rolling_sparse_packs_supported_for_rows(rows)
        .then(|| rolling_sparse_physical_dispatches_per_layer(rows))
}

fn synthetic_sparse_spark_expert_mode_for_layer(layer_id: usize) -> bool {
    env::var(PHASE0_SPARK_EXPERT_MODE_ENV)
        .map(|value| phase0_spark_expert_mode_is_synthetic_sparse_for_layer(&value, layer_id))
        .unwrap_or(false)
}

fn phase0_spark_expert_mode_is_synthetic_sparse_for_layer(value: &str, layer_id: usize) -> bool {
    layer_id >= GLM52_FIRST_K_DENSE_REPLACE && value.eq_ignore_ascii_case("synthetic")
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn elapsed_ms_optional(start: Option<Instant>) -> f64 {
    start.map(elapsed_ms).unwrap_or(0.0)
}

fn validate_bf16_payload_byte_count(
    payloads: &[Vec<u8>],
    expected_rows: usize,
    row_width: usize,
) -> Result<()> {
    let expected_bytes = expected_rows
        .checked_mul(row_width)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("scheduler sparse TCP BF16 payload expected byte count overflow")?;
    let actual_bytes = payloads.iter().map(Vec::len).sum::<usize>();
    anyhow::ensure!(
        actual_bytes == expected_bytes,
        "scheduler sparse TCP BF16 payload bytes {} did not match expected {expected_bytes}",
        actual_bytes
    );
    for payload in payloads {
        anyhow::ensure!(
            payload.len() % std::mem::size_of::<u16>() == 0,
            "scheduler sparse TCP BF16 payload had odd byte length {}",
            payload.len()
        );
    }
    Ok(())
}

fn scheduler_sparse_tcp_payload_partials_for_segment(
    dispatch: &TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    segment_row_start: usize,
    segment_row_count: usize,
    row_width: usize,
) -> Result<(Vec<Vec<u8>>, Vec<Vec<usize>>)> {
    let segment_row_end = segment_row_start
        .checked_add(segment_row_count)
        .context("scheduler sparse TCP segment row range overflows usize")?;
    let row_bytes = row_width
        .checked_mul(std::mem::size_of::<u16>())
        .context("scheduler sparse TCP segment row byte count overflows usize")?;
    anyhow::ensure!(
        dispatch.partial_outputs_bf16_by_host.len() == dispatch.global_row_indices_by_host.len(),
        "scheduler sparse TCP payload host count mismatch: payloads={} row_maps={}",
        dispatch.partial_outputs_bf16_by_host.len(),
        dispatch.global_row_indices_by_host.len()
    );
    let mut segment_payloads = Vec::new();
    let mut segment_row_indices_by_host = Vec::new();
    let mut selected_host_rows = 0_usize;

    for (host_index, (payload, row_indices)) in dispatch
        .partial_outputs_bf16_by_host
        .iter()
        .zip(dispatch.global_row_indices_by_host.iter())
        .enumerate()
    {
        let expected_bytes = row_indices
            .len()
            .checked_mul(row_bytes)
            .context("scheduler sparse TCP host payload byte count overflows usize")?;
        anyhow::ensure!(
            payload.len() == expected_bytes,
            "scheduler sparse TCP host payload {host_index} bytes {} did not match rows {} * row bytes {}",
            payload.len(),
            row_indices.len(),
            row_bytes
        );
        let mut host_payload = Vec::new();
        let mut host_row_indices = Vec::new();
        for (payload_row_index, global_row_index) in row_indices.iter().enumerate() {
            if *global_row_index < segment_row_start || *global_row_index >= segment_row_end {
                continue;
            }
            let payload_byte_start = payload_row_index
                .checked_mul(row_bytes)
                .context("scheduler sparse TCP host payload row byte start overflows usize")?;
            let payload_byte_end = payload_byte_start
                .checked_add(row_bytes)
                .context("scheduler sparse TCP host payload row byte end overflows usize")?;
            host_payload.extend_from_slice(&payload[payload_byte_start..payload_byte_end]);
            host_row_indices.push(*global_row_index - segment_row_start);
        }
        if !host_row_indices.is_empty() {
            selected_host_rows += host_row_indices.len();
            segment_payloads.push(host_payload);
            segment_row_indices_by_host.push(host_row_indices);
        }
    }

    anyhow::ensure!(
        selected_host_rows > 0,
        "scheduler sparse TCP payload segment rows {segment_row_start}..{segment_row_end} had no routed partials"
    );
    Ok((segment_payloads, segment_row_indices_by_host))
}

fn scheduler_sparse_tcp_payload_dispatch_for_segment(
    dispatch: &TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    segment_row_start: usize,
    segment_row_count: usize,
    row_width: usize,
    route_count: usize,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    let (partial_outputs_bf16_by_host, global_row_indices_by_host) =
        scheduler_sparse_tcp_payload_partials_for_segment(
            dispatch,
            segment_row_start,
            segment_row_count,
            row_width,
        )?;
    let segment_row_end = segment_row_start
        .checked_add(segment_row_count)
        .context("scheduler sparse cohort segment row range overflow")?;
    let completed_global_row_slices = dispatch
        .completed_global_row_slices
        .iter()
        .filter_map(|rows| {
            let local_rows = rows
                .iter()
                .filter(|row| (segment_row_start..segment_row_end).contains(row))
                .map(|row| row - segment_row_start)
                .collect::<Vec<_>>();
            (!local_rows.is_empty()).then_some(local_rows)
        })
        .collect::<Vec<_>>();
    let mut stats = dispatch.stats.clone();
    stats.global_rows = segment_row_count;
    stats.host_rows = global_row_indices_by_host.iter().map(Vec::len).sum();
    stats.routes = scheduler_dispatched_route_count(route_count)?;
    stats.output_values = stats
        .global_rows
        .checked_mul(stats.output_dim)
        .context("scheduler sparse cohort output value count overflow")?;
    if !stats.contribution_counts.is_empty() {
        stats.contribution_counts = stats
            .contribution_counts
            .get(segment_row_start..segment_row_end)
            .context("scheduler sparse cohort contribution count slice is out of range")?
            .to_vec();
    }
    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host,
        global_row_indices_by_host,
        completed_global_row_slices,
        stats,
    })
}

fn remap_sparse_payload_chunk_rows(
    chunk: &mut VerbsHostProtocolV2HostBatchSetBf16PayloadChunk,
    dispatch_to_batch_rows: &[usize],
) -> Result<()> {
    for row_index in chunk
        .global_row_indices
        .iter_mut()
        .chain(chunk.completed_global_row_indices.iter_mut())
    {
        *row_index = *dispatch_to_batch_rows.get(*row_index).with_context(|| {
            format!(
                "streamed sparse response row {row_index} exceeds {} dispatch rows",
                dispatch_to_batch_rows.len()
            )
        })?;
    }
    Ok(())
}

fn rebalance_scheduler_sparse_rolling_unsupported_tail(
    emissions: &mut Vec<RollingExpertRowPackEmission>,
) -> Result<()> {
    rebalance_scheduler_sparse_rolling_unsupported_tail_with(
        emissions,
        ROLLING_SPARSE_PACK_ROWS,
        |rows| Ok(rows > 1 && spark_expert_reduction_dispatch_for_rows(rows)?.is_some()),
    )
}

fn rebalance_scheduler_sparse_rolling_unsupported_tail_with(
    emissions: &mut Vec<RollingExpertRowPackEmission>,
    max_pack_rows: usize,
    mut is_supported: impl FnMut(usize) -> Result<bool>,
) -> Result<()> {
    while emissions.len() >= 2 {
        let tail_rows = emissions
            .last()
            .expect("rolling tail exists")
            .row_indices
            .len();
        if is_supported(tail_rows)? {
            break;
        }
        let mut minimum_supported_rows = None;
        for rows in 1..=max_pack_rows {
            if is_supported(rows)? {
                minimum_supported_rows = Some(rows);
                break;
            }
        }
        let minimum_supported_rows = minimum_supported_rows
            .context("rolling sparse reduction has no supported physical row count")?;
        let tail_index = emissions.len() - 1;
        let previous_index = tail_index - 1;
        let combined_rows = emissions[previous_index].row_indices.len() + tail_rows;
        if combined_rows <= max_pack_rows && is_supported(combined_rows)? {
            let tail = emissions.pop().expect("rolling unsupported tail exists");
            let previous = emissions
                .last_mut()
                .expect("rolling unsupported tail has a previous emission");
            previous.row_indices.extend(tail.row_indices);
            previous.admitted_rows = previous.admitted_rows.max(tail.admitted_rows);
            previous.deadline_row_exclusive = previous
                .deadline_row_exclusive
                .or(tail.deadline_row_exclusive);
            refresh_scheduler_sparse_rolling_emission_span(previous);
            continue;
        }

        let moved_rows = minimum_supported_rows.saturating_sub(tail_rows);
        anyhow::ensure!(
            moved_rows > 0 && emissions[previous_index].row_indices.len() > moved_rows,
            "rolling sparse tail with {tail_rows} rows cannot be rebalanced to the minimum supported {minimum_supported_rows} rows"
        );
        let moved = {
            let previous = &mut emissions[previous_index];
            previous
                .row_indices
                .split_off(previous.row_indices.len() - moved_rows)
        };
        let previous_deadline = emissions[previous_index].deadline_row_exclusive;
        let previous_admitted_rows = emissions[previous_index].admitted_rows;
        emissions[tail_index].row_indices.splice(0..0, moved);
        emissions[tail_index].admitted_rows = emissions[tail_index]
            .admitted_rows
            .max(previous_admitted_rows);
        emissions[tail_index].deadline_row_exclusive = emissions[tail_index]
            .deadline_row_exclusive
            .or(previous_deadline);
        refresh_scheduler_sparse_rolling_emission_span(&mut emissions[previous_index]);
        refresh_scheduler_sparse_rolling_emission_span(&mut emissions[tail_index]);
        anyhow::ensure!(
            emissions[previous_index].row_indices.len() <= max_pack_rows
                && is_supported(emissions[previous_index].row_indices.len())?
                && emissions[tail_index].row_indices.len() <= max_pack_rows
                && is_supported(emissions[tail_index].row_indices.len())?,
            "rolling sparse tail rebalance did not produce supported physical packs"
        );
    }
    Ok(())
}

fn refresh_scheduler_sparse_rolling_emission_span(emission: &mut RollingExpertRowPackEmission) {
    let oldest = emission.row_indices.iter().copied().min().unwrap_or(0);
    emission.oldest_pending_row = oldest;
    emission.max_selected_row_offset = emission
        .row_indices
        .iter()
        .map(|row| row.saturating_sub(oldest))
        .max()
        .unwrap_or(0);
}

fn scheduler_sparse_rolling_chunk_for_row(
    chunks: &[SchedulerSparseRollingChunk],
    global_row_index: usize,
) -> Result<(&SchedulerSparseRollingChunk, usize)> {
    let chunk_index = chunks.partition_point(|chunk| chunk.global_row_end() <= global_row_index);
    let chunk = chunks.get(chunk_index).with_context(|| {
        format!("rolling sparse row {global_row_index} has no admitted source chunk")
    })?;
    anyhow::ensure!(
        global_row_index >= chunk.global_row_start && global_row_index < chunk.global_row_end(),
        "rolling sparse row {global_row_index} is outside source chunk {}..{}",
        chunk.global_row_start,
        chunk.global_row_end()
    );
    Ok((chunk, global_row_index - chunk.global_row_start))
}

fn build_scheduler_sparse_rolling_emission(
    chunks: &[SchedulerSparseRollingChunk],
    emission: RollingExpertRowPackEmission,
) -> Result<SchedulerSparseRollingQueuedEmission> {
    anyhow::ensure!(
        !emission.row_indices.is_empty(),
        "rolling sparse emission is empty"
    );
    let (first_chunk, _) = scheduler_sparse_rolling_chunk_for_row(chunks, emission.row_indices[0])?;
    let mut batch = first_chunk.batch.clone();
    batch.rows.clear();
    let mut routes = Vec::with_capacity(emission.row_indices.len() * GLM52_TOP_K);
    let hidden_bytes = emission
        .row_indices
        .len()
        .checked_mul(batch.hidden_bytes_per_row)
        .context("rolling sparse emission hidden byte count overflow")?;
    let mut hidden_payload = Vec::with_capacity(hidden_bytes);
    let mut seen_rows = BTreeSet::new();

    for (dispatch_row, global_row_index) in emission.row_indices.iter().copied().enumerate() {
        anyhow::ensure!(
            seen_rows.insert(global_row_index),
            "rolling sparse emission repeats row {global_row_index}"
        );
        let (chunk, local_row_index) =
            scheduler_sparse_rolling_chunk_for_row(chunks, global_row_index)?;
        anyhow::ensure!(
            chunk.batch.layer_id == batch.layer_id
                && chunk.batch.placement_version == batch.placement_version
                && chunk.batch.hidden_dim == batch.hidden_dim
                && chunk.batch.hidden_bytes_per_row == batch.hidden_bytes_per_row
                && chunk.batch.hidden_dtype == batch.hidden_dtype
                && chunk.batch.quantization_recipe == batch.quantization_recipe,
            "rolling sparse emission mixed incompatible source batches"
        );
        let source_row = chunk
            .batch
            .rows
            .get(local_row_index)
            .context("rolling sparse source row is out of range")?;
        let route_end = source_row
            .route_offset
            .checked_add(source_row.route_count)
            .context("rolling sparse source route range overflow")?;
        let source_routes = chunk
            .routes
            .get(source_row.route_offset..route_end)
            .context("rolling sparse source route range is out of bounds")?;
        let mut row = source_row.clone();
        row.row_id = dispatch_row as u64;
        row.route_offset = routes.len();
        for route in source_routes {
            anyhow::ensure!(
                route.row_index == local_row_index,
                "rolling sparse source route row {} did not match local row {local_row_index}",
                route.row_index
            );
            routes.push(ExpertBatchRoute {
                row_index: dispatch_row,
                expert_id: route.expert_id,
                gate_weight: route.gate_weight,
            });
        }
        batch.rows.push(row);

        let byte_start = local_row_index
            .checked_mul(batch.hidden_bytes_per_row)
            .context("rolling sparse source hidden byte start overflow")?;
        let byte_end = byte_start
            .checked_add(batch.hidden_bytes_per_row)
            .context("rolling sparse source hidden byte end overflow")?;
        hidden_payload.extend_from_slice(
            chunk
                .hidden_payload
                .get(byte_start..byte_end)
                .context("rolling sparse source hidden row is out of bounds")?,
        );
    }
    anyhow::ensure!(
        routes.len() == batch.route_count() && hidden_payload.len() == hidden_bytes,
        "rolling sparse emission envelope is inconsistent"
    );
    if batch.num_rows() > batch.graph_bucket.row_capacity {
        batch.graph_bucket = GraphBucket::new(batch.num_rows());
    }
    Ok(SchedulerSparseRollingQueuedEmission {
        emission,
        batch,
        routes,
        hidden_payload,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SchedulerSparseRollingRouteStats {
    unique_experts: usize,
    expert_tiles: usize,
    min_expert_rows: usize,
    max_expert_rows: usize,
}

fn scheduler_sparse_rolling_route_stats(
    routes: &[ExpertBatchRoute],
    row_start: usize,
    row_end: usize,
) -> Result<SchedulerSparseRollingRouteStats> {
    anyhow::ensure!(
        row_start < row_end,
        "rolling sparse route statistics require a non-empty row range"
    );
    let mut rows_by_expert = BTreeMap::<usize, usize>::new();
    for route in routes
        .iter()
        .filter(|route| (row_start..row_end).contains(&route.row_index))
    {
        *rows_by_expert.entry(route.expert_id).or_default() += 1;
    }
    anyhow::ensure!(
        !rows_by_expert.is_empty(),
        "rolling sparse row range {row_start}..{row_end} has no routes"
    );
    let expert_tiles = rows_by_expert
        .values()
        .map(|rows| rows.div_ceil(ROLLING_SPARSE_EXPERT_TILE_ROWS))
        .sum();
    Ok(SchedulerSparseRollingRouteStats {
        unique_experts: rows_by_expert.len(),
        expert_tiles,
        min_expert_rows: rows_by_expert.values().copied().min().unwrap_or(0),
        max_expert_rows: rows_by_expert.values().copied().max().unwrap_or(0),
    })
}

fn scheduler_sparse_tcp_batch_slice(
    batch: &ExpertBatch,
    routes: &[ExpertBatchRoute],
    row_start: usize,
    row_end: usize,
) -> Result<(ExpertBatch, Vec<ExpertBatchRoute>)> {
    anyhow::ensure!(
        row_start < row_end && row_end <= batch.num_rows(),
        "invalid scheduler sparse TCP row slice {row_start}..{row_end} for {} rows",
        batch.num_rows()
    );
    anyhow::ensure!(
        routes.len() == batch.route_count(),
        "scheduler sparse TCP slice routes {} did not match batch route count {}",
        routes.len(),
        batch.route_count()
    );

    let mut slice_rows = Vec::with_capacity(row_end - row_start);
    let mut slice_routes = Vec::new();
    let mut next_route_offset = 0_usize;
    for (local_row_index, row) in batch.rows[row_start..row_end].iter().enumerate() {
        let global_row_index = row_start + local_row_index;
        let route_start = row.route_offset;
        let route_end = route_start
            .checked_add(row.route_count)
            .context("scheduler sparse TCP slice route range overflow")?;
        let row_routes = routes.get(route_start..route_end).with_context(|| {
            format!(
                "scheduler sparse TCP slice route range {route_start}..{route_end} exceeds {} routes",
                routes.len()
            )
        })?;
        slice_rows.push(glmrt_core::ExpertBatchRow {
            row_id: local_row_index as u64,
            source_kind: row.source_kind,
            request_id: row.request_id.clone(),
            sequence_id: row.sequence_id.clone(),
            token_position: row.token_position,
            route_offset: next_route_offset,
            route_count: row.route_count,
        });
        for route in row_routes {
            anyhow::ensure!(
                route.row_index == global_row_index,
                "scheduler sparse TCP slice route row mismatch: expected {global_row_index}, got {}",
                route.row_index
            );
            slice_routes.push(ExpertBatchRoute {
                row_index: local_row_index,
                expert_id: route.expert_id,
                gate_weight: route.gate_weight,
            });
        }
        next_route_offset += row.route_count;
    }

    Ok((
        ExpertBatch {
            layer_id: batch.layer_id,
            placement_version: batch.placement_version.clone(),
            hidden_dim: batch.hidden_dim,
            hidden_bytes_per_row: batch.hidden_bytes_per_row,
            hidden_dtype: batch.hidden_dtype.clone(),
            graph_bucket: batch.graph_bucket,
            quantization_recipe: batch.quantization_recipe.clone(),
            rows: slice_rows,
        },
        slice_routes,
    ))
}

fn offset_sparse_payload_chunk_rows(
    chunk: &mut VerbsHostProtocolV2HostBatchSetBf16PayloadChunk,
    row_offset: usize,
) -> Result<()> {
    for row_index in chunk
        .global_row_indices
        .iter_mut()
        .chain(chunk.completed_global_row_indices.iter_mut())
    {
        *row_index = row_index
            .checked_add(row_offset)
            .context("sliced sparse response row offset overflow")?;
    }
    Ok(())
}

fn merge_scheduler_sparse_payload_slice_dispatches(
    batch: SchedulerSparseTcpPayloadDispatchBatchShape,
    mut slices: Vec<(usize, usize, TcpProtocolV2HostBatchSetBf16PayloadDispatch)>,
) -> Result<TcpProtocolV2HostBatchSetBf16PayloadDispatch> {
    anyhow::ensure!(
        slices.len() > 1,
        "sliced sparse dispatch merge requires at least two slices"
    );
    slices.sort_by_key(|(row_start, _, _)| *row_start);
    let include_contribution_counts = slices
        .iter()
        .any(|(_, _, dispatch)| !dispatch.stats.contribution_counts.is_empty());
    let mut contribution_counts = include_contribution_counts.then(|| vec![0_usize; batch.rows]);
    let mut partial_outputs_bf16_by_host = Vec::new();
    let mut global_row_indices_by_host = Vec::new();
    let mut completed_global_row_slices = Vec::new();
    let mut hosts = 0_usize;
    let mut host_rows = 0_usize;
    let mut routes = 0_usize;
    let mut request_wire_bytes = 0_usize;
    let mut response_wire_bytes = 0_usize;
    let mut response_executor_ids = Vec::new();
    let mut output_checksum = 0.0_f64;
    let mut graph_pool_leases = 0_usize;
    let mut graph_pool_fixed_buffer_bytes = 0_usize;
    let mut graph_pool_active_rows = 0_usize;
    let mut graph_pool_active_routes = 0_usize;
    let mut graph_pool_active_expert_tiles = 0_usize;
    let mut graph_pool_bucket_rows = Vec::new();
    let mut expected_row_start = 0_usize;

    for (row_start, row_count, dispatch) in slices {
        anyhow::ensure!(
            row_start == expected_row_start,
            "sliced sparse dispatch row start {row_start} did not match expected {expected_row_start}"
        );
        expected_row_start = expected_row_start
            .checked_add(row_count)
            .context("sliced sparse dispatch row coverage overflow")?;
        anyhow::ensure!(
            dispatch.stats.global_rows == row_count,
            "sliced sparse dispatch reported {} rows for {row_count}-row slice",
            dispatch.stats.global_rows
        );
        anyhow::ensure!(
            dispatch.stats.output_dim == batch.hidden_dim,
            "sliced sparse dispatch output dim {} did not match {}",
            dispatch.stats.output_dim,
            batch.hidden_dim
        );
        anyhow::ensure!(
            dispatch.partial_outputs_bf16_by_host.len()
                == dispatch.global_row_indices_by_host.len(),
            "sliced sparse dispatch payload and row-index group counts differ"
        );
        if let Some(counts) = contribution_counts.as_mut() {
            anyhow::ensure!(
                dispatch.stats.contribution_counts.len() == row_count,
                "sliced sparse dispatch contribution counts {} did not match {row_count} rows",
                dispatch.stats.contribution_counts.len()
            );
            counts[row_start..expected_row_start]
                .copy_from_slice(&dispatch.stats.contribution_counts);
        } else {
            anyhow::ensure!(
                dispatch.stats.contribution_counts.is_empty(),
                "sliced sparse dispatch mixed omitted and included contribution counts"
            );
        }

        hosts = hosts
            .checked_add(dispatch.stats.hosts)
            .context("sliced sparse dispatch host count overflow")?;
        host_rows = host_rows
            .checked_add(dispatch.stats.host_rows)
            .context("sliced sparse dispatch host row count overflow")?;
        routes = routes
            .checked_add(dispatch.stats.routes)
            .context("sliced sparse dispatch route count overflow")?;
        request_wire_bytes = request_wire_bytes
            .checked_add(dispatch.stats.request_wire_bytes)
            .context("sliced sparse dispatch request byte count overflow")?;
        response_wire_bytes = response_wire_bytes
            .checked_add(dispatch.stats.response_wire_bytes)
            .context("sliced sparse dispatch response byte count overflow")?;
        graph_pool_leases = graph_pool_leases
            .checked_add(dispatch.stats.graph_pool_leases)
            .context("sliced sparse dispatch graph lease count overflow")?;
        graph_pool_fixed_buffer_bytes = graph_pool_fixed_buffer_bytes
            .checked_add(dispatch.stats.graph_pool_fixed_buffer_bytes)
            .context("sliced sparse dispatch graph buffer byte count overflow")?;
        graph_pool_active_rows = graph_pool_active_rows
            .checked_add(dispatch.stats.graph_pool_active_rows)
            .context("sliced sparse dispatch graph row count overflow")?;
        graph_pool_active_routes = graph_pool_active_routes
            .checked_add(dispatch.stats.graph_pool_active_routes)
            .context("sliced sparse dispatch graph route count overflow")?;
        graph_pool_active_expert_tiles = graph_pool_active_expert_tiles
            .checked_add(dispatch.stats.graph_pool_active_expert_tiles)
            .context("sliced sparse dispatch graph expert tile count overflow")?;
        output_checksum += dispatch.stats.output_checksum;
        response_executor_ids.extend(dispatch.stats.response_executor_ids);
        graph_pool_bucket_rows.extend(dispatch.stats.graph_pool_bucket_rows);
        partial_outputs_bf16_by_host.extend(dispatch.partial_outputs_bf16_by_host);
        global_row_indices_by_host.extend(dispatch.global_row_indices_by_host.into_iter().map(
            |mut row_indices| {
                for row_index in &mut row_indices {
                    *row_index += row_start;
                }
                row_indices
            },
        ));
        completed_global_row_slices.extend(dispatch.completed_global_row_slices.into_iter().map(
            |mut row_indices| {
                for row_index in &mut row_indices {
                    *row_index += row_start;
                }
                row_indices
            },
        ));
    }

    anyhow::ensure!(
        expected_row_start == batch.rows,
        "sliced sparse dispatch covered {expected_row_start} of {} rows",
        batch.rows
    );
    let expected_dispatched_routes = scheduler_dispatched_route_count(batch.routes)?;
    anyhow::ensure!(
        routes == expected_dispatched_routes,
        "sliced sparse dispatch reported {routes} routes instead of {expected_dispatched_routes} dispatched routes"
    );
    let output_values = batch
        .rows
        .checked_mul(batch.hidden_dim)
        .context("sliced sparse dispatch output value count overflow")?;
    Ok(TcpProtocolV2HostBatchSetBf16PayloadDispatch {
        partial_outputs_bf16_by_host,
        global_row_indices_by_host,
        completed_global_row_slices,
        stats: TcpProtocolV2HostBatchSetDispatchStats {
            hosts,
            global_rows: batch.rows,
            host_rows,
            routes,
            output_dim: batch.hidden_dim,
            output_values,
            request_wire_bytes,
            response_wire_bytes,
            response_executor_ids,
            contribution_counts: contribution_counts.unwrap_or_default(),
            output_checksum,
            graph_pool_leases,
            graph_pool_fixed_buffer_bytes,
            graph_pool_active_rows,
            graph_pool_active_routes,
            graph_pool_active_expert_tiles,
            graph_pool_bucket_rows,
        },
    })
}

impl RealFullSchedulerNumericProgression {
    pub(super) fn new(shape: RealFullSchedulerNumericProgressionShape) -> Self {
        Self {
            shape,
            live_request: false,
            event_owned_raw_tp4_reduction: false,
            retain_final_target_device_hidden: false,
            retain_full_target_device_hidden: false,
            target_device_hidden_tap_rows: 0,
            target_device_hidden_tap_layer_ids: Vec::new(),
            target_device_hidden_taps: Vec::new(),
            residual_bf16: vec![
                0;
                shape.unique_rows()
                    * NUMERIC_PROGRESS_HIDDEN_DIM
                    * std::mem::size_of::<u16>()
            ],
            initial_prefill_embedding_rows: 0,
            initial_prefill_embedding_bytes_read: 0,
            initial_decode_embedding_rows: 0,
            initial_decode_embedding_bytes_read: 0,
            selected_prefill_rows: 0,
            selected_decode_rows: 0,
            selected_mtp_rows: 0,
            attention_value_updates: 0,
            mlp_value_updates: 0,
            source_segments: 0,
            attention_residual_adds: 0,
            mlp_residual_adds: 0,
            attention_residual_add_backend: None,
            mlp_residual_add_backend: None,
            attention_device_output_delta_rows: 0,
            attention_device_output_delta_values: 0,
            attention_device_output_delta_checksum: 0.0,
            attention_device_output_delta_backend: None,
            attention_device_output_delta_device_prefix_rows: 0,
            attention_device_output_delta_device_prefix_values: 0,
            attention_device_output_delta_device_prefix_backend: None,
            device_delta_template_uploads: 0,
            device_delta_template_uses: 0,
            device_delta_template_resident_values: 0,
            device_delta_templates: BTreeMap::new(),
            device_delta_template_upload_bf16_scratch: Vec::new(),
            device_mlp_delta_rows: 0,
            device_mlp_delta_values: 0,
            device_mlp_delta_checksum: 0.0,
            device_mlp_delta_backend: None,
            device_mlp_weight_uploads: 0,
            device_mlp_weight_resident_values: 0,
            device_mlp_weights: None,
            device_mlp_weight_upload_bf16_scratch: Vec::new(),
            device_real_dense_mlp_delta_rows: 0,
            device_real_dense_mlp_delta_values: 0,
            device_real_dense_mlp_delta_checksum: 0.0,
            device_real_dense_mlp_delta_backend: None,
            device_real_dense_mlp_norm_backend: None,
            device_real_dense_mlp_weight_tensors: 0,
            device_real_dense_mlp_weight_bytes: 0,
            device_real_dense_mlp_source_segments: 0,
            device_real_dense_mlp_layers: BTreeSet::new(),
            device_real_dense_mlp_resident_weight_names: BTreeSet::new(),
            device_real_dense_mlp_resident_weights_by_layer: BTreeMap::new(),
            device_real_sparse_shared_mlp_delta_rows: 0,
            device_real_sparse_shared_mlp_delta_values: 0,
            device_real_sparse_shared_mlp_delta_checksum: 0.0,
            device_real_sparse_shared_mlp_delta_backend: None,
            device_real_sparse_shared_mlp_norm_backend: None,
            device_real_sparse_shared_mlp_weight_tensors: 0,
            device_real_sparse_shared_mlp_weight_bytes: 0,
            device_real_sparse_shared_mlp_source_segments: 0,
            device_real_sparse_shared_mlp_layers: BTreeSet::new(),
            device_real_sparse_shared_mlp_resident_weight_names: BTreeSet::new(),
            device_real_sparse_shared_mlp_resident_weights_by_layer: BTreeMap::new(),
            device_real_sparse_routed_mlp_delta_rows: 0,
            device_real_sparse_routed_mlp_delta_values: 0,
            device_real_sparse_routed_mlp_delta_checksum: 0.0,
            device_real_sparse_routed_mlp_delta_backend: None,
            device_real_sparse_routed_mlp_route_backend: None,
            device_real_sparse_routed_mlp_router_backend: None,
            device_real_sparse_routed_mlp_routes: 0,
            device_real_sparse_routed_mlp_router_weight_bytes: 0,
            device_real_sparse_routed_mlp_router_bias_bytes: 0,
            device_real_sparse_routed_mlp_route_cache_cuda_entries: 0,
            device_real_sparse_routed_mlp_route_cache_cuda_uploads: 0,
            device_real_sparse_routed_mlp_route_cache_cuda_hits: 0,
            device_real_sparse_routed_mlp_router_cache_entries: 0,
            device_real_sparse_routed_mlp_router_cache_hits: 0,
            device_real_sparse_routed_mlp_source_segments: 0,
            device_real_sparse_routed_mlp_layers: BTreeSet::new(),
            device_real_sparse_routed_mlp_router_cache: RouterTensorCache::default(),
            device_real_sparse_routed_mlp_route_cache: RouteTensorCache::default(),
            device_real_sparse_routed_mlp_intermediate_rows: BTreeMap::new(),
            sparse_tcp_routed_mlp: None,
            device_hidden_segment_residual_adds: 0,
            device_hidden_segment_value_updates: 0,
            device_hidden_segment_residual_add_backend: None,
            device_hidden_segments: BTreeMap::new(),
            delta_bf16_scratch: Vec::new(),
            output_bf16_scratch: Vec::new(),
            device_sparse_routed_normalized_readback_bf16_scratch: Vec::new(),
        }
    }

    pub(super) fn with_sparse_tcp_routed_mlp(
        mut self,
        context: RealFullSchedulerSparseTcpRoutedMlpContext,
    ) -> Self {
        self.sparse_tcp_routed_mlp = Some(context);
        self
    }

    pub(super) fn with_live_request(mut self) -> Self {
        self.live_request = true;
        self
    }

    pub(super) fn with_event_owned_raw_tp4_reduction(mut self) -> Self {
        self.event_owned_raw_tp4_reduction = true;
        self
    }

    pub(super) fn with_final_target_device_hidden(mut self) -> Self {
        self.retain_final_target_device_hidden = true;
        self
    }

    pub(super) fn with_full_target_device_hidden(mut self) -> Self {
        self.retain_final_target_device_hidden = true;
        self.retain_full_target_device_hidden = true;
        self
    }

    pub(super) fn with_target_device_hidden_taps(mut self, rows: usize) -> Self {
        let layer_ids = if dflash2_serving_requested() {
            GLM53_DFLASH2_TARGET_CAPTURE_TAPS.to_vec()
        } else {
            dspark_target_hidden_tap_layer_ids().to_vec()
        };
        self.with_target_device_hidden_taps_for(rows, &layer_ids)
    }

    pub(super) fn with_target_device_hidden_taps_for(
        mut self,
        rows: usize,
        layer_ids: &[usize],
    ) -> Self {
        self.target_device_hidden_tap_rows = rows;
        self.target_device_hidden_tap_layer_ids = layer_ids.to_vec();
        self.target_device_hidden_taps = (0..layer_ids.len()).map(|_| None).collect();
        self
    }

    pub(super) fn capture_target_device_hidden_tap(
        &mut self,
        checkpoint_layer_id: usize,
    ) -> Result<()> {
        if self.target_device_hidden_tap_rows == 0 {
            return Ok(());
        }
        let Some(tap_index) = self
            .target_device_hidden_tap_layer_ids
            .iter()
            .position(|layer_id| *layer_id == checkpoint_layer_id)
        else {
            return Ok(());
        };
        anyhow::ensure!(
            self.target_device_hidden_taps[tap_index].is_none(),
            "scheduler dSpark target hidden tap {checkpoint_layer_id} was captured twice"
        );
        let total_rows = self
            .shape
            .prefill_rows
            .checked_add(self.shape.decode_rows)
            .and_then(|rows| rows.checked_add(self.shape.mtp_rows))
            .context("scheduler dSpark target tap row count overflow")?;
        let retained_rows = self.target_device_hidden_tap_rows.min(total_rows);
        anyhow::ensure!(
            retained_rows > 0,
            "scheduler dSpark target hidden taps require at least one current target row"
        );
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler dSpark target tap row bytes overflow")?;
        let row_start = total_rows - retained_rows;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .context("scheduler dSpark target tap byte start overflow")?;
        let byte_end = retained_rows
            .checked_mul(row_bytes)
            .and_then(|bytes| byte_start.checked_add(bytes))
            .context("scheduler dSpark target tap byte end overflow")?;
        let mut cursor = byte_start;
        let mut slice_specs = Vec::new();
        while cursor < byte_end {
            let (key, batch) = self
                .device_hidden_segments
                .iter()
                .filter(|(key, _)| key.byte_start <= cursor && key.byte_end > cursor)
                .min_by_key(|(key, _)| key.byte_end - key.byte_start)
                .with_context(|| {
                    format!(
                        "scheduler dSpark target tap {checkpoint_layer_id} has no resident suffix segment starting at byte {cursor} of {byte_end}"
                    )
                })?;
            anyhow::ensure!(
                key.byte_end > cursor
                    && key.byte_end - key.byte_start == batch.rows.saturating_mul(row_bytes)
                    && batch.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler dSpark target tap {checkpoint_layer_id} segment {}..{} shape {}x{} is invalid",
                key.byte_start,
                key.byte_end,
                batch.rows,
                batch.values_per_row
            );
            let slice_end = key.byte_end.min(byte_end);
            let source_byte_offset = cursor - key.byte_start;
            let slice_bytes = slice_end - cursor;
            anyhow::ensure!(
                source_byte_offset % row_bytes == 0 && slice_bytes % row_bytes == 0,
                "scheduler dSpark target tap {checkpoint_layer_id} slice is not row aligned"
            );
            slice_specs.push((
                *key,
                source_byte_offset / row_bytes,
                slice_bytes / row_bytes,
            ));
            cursor = slice_end;
        }
        let slices = slice_specs
            .iter()
            .map(|(key, row_start, rows)| {
                (
                    self.device_hidden_segments
                        .get(key)
                        .expect("dSpark tap segment key was selected above"),
                    *row_start,
                    *rows,
                )
            })
            .collect::<Vec<_>>();
        let hidden = concat_device_bf16_row_slices_async(
            slices.as_slice(),
            "scheduler dSpark target hidden tap",
        )?;
        anyhow::ensure!(
            hidden.rows == retained_rows && hidden.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler dSpark target tap {checkpoint_layer_id} shape mismatch: expected {retained_rows}x{NUMERIC_PROGRESS_HIDDEN_DIM}, got {}x{}",
            hidden.rows,
            hidden.values_per_row
        );
        let copy_ready = hidden
            .ready_event()
            .context("asynchronous dSpark target tap copy has no ready event")?;
        drop(slices);
        // The source residual is consumed by the next target layer. Make that
        // consumer wait for the tiny tap copy instead of synchronizing the
        // host after every one of the five checkpoint boundaries.
        for (key, _, _) in slice_specs {
            self.device_hidden_segments
                .get_mut(&key)
                .expect("dSpark tap segment remained resident during capture")
                .set_ready_event(Arc::clone(&copy_ready));
        }
        self.target_device_hidden_taps[tap_index] = Some(hidden);
        Ok(())
    }

    pub(super) fn seed_prefill_token_embeddings(
        &mut self,
        catalog: &TensorCatalog,
        token_ids: &[usize],
        vision_embeddings: Option<&[glmrt_api::RealFullVisionEmbedding]>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.selected_prefill_rows == 0
                && self.selected_decode_rows == 0
                && self.selected_mtp_rows == 0,
            "scheduler prefill embeddings must be seeded before applying selected rows"
        );
        anyhow::ensure!(
            token_ids.len() == self.shape.prefill_rows,
            "scheduler prefill embedding seed token count {} does not match prefill rows {}",
            token_ids.len(),
            self.shape.prefill_rows
        );
        dump_layer_boundary_token_ids(
            RowSourceKind::PrefillChunk,
            self.shape.prefix_tokens,
            token_ids,
        )?;
        if self.live_request {
            let mut offset = 0_usize;
            let mut device_bytes_read = 0_u64;
            let chunk_tokens = self.shape.prefill_chunk_tokens.max(1);
            while offset < token_ids.len() {
                let token_count = (token_ids.len() - offset).min(chunk_tokens);
                match self.seed_device_token_embedding_segment(
                    catalog,
                    &token_ids[offset..offset + token_count],
                    offset,
                )? {
                    Some(bytes_read) => {
                        device_bytes_read += bytes_read;
                        offset += token_count;
                    }
                    None if offset == 0 => break,
                    None => {
                        anyhow::bail!(
                            "scheduler live prefill embedding device seeding became unavailable after {offset} rows"
                        );
                    }
                }
            }
            if offset == token_ids.len() {
                self.initial_prefill_embedding_rows += token_ids.len();
                self.initial_prefill_embedding_bytes_read += device_bytes_read;
                self.apply_prefill_vision_embeddings(
                    vision_embeddings.unwrap_or_default(),
                    0,
                    token_ids.len(),
                )?;
                return Ok(());
            }
        }
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        for (row_index, token_id) in token_ids.iter().copied().enumerate() {
            let embedding = real_full_embedding_hidden_for_token(catalog, token_id)
                .with_context(|| format!("loading embedding hidden for prompt token {token_id}"))?;
            let embedding_bf16 = bf16_bytes_from_f32(&embedding.hidden);
            anyhow::ensure!(
                embedding_bf16.len() == row_bytes,
                "scheduler prefill embedding row byte length mismatch for token {}: expected {} got {}",
                token_id,
                row_bytes,
                embedding_bf16.len()
            );
            let byte_start = row_index
                .checked_mul(row_bytes)
                .context("scheduler prefill embedding byte offset overflows usize")?;
            let byte_end = byte_start + row_bytes;
            self.residual_bf16[byte_start..byte_end].copy_from_slice(&embedding_bf16);
            self.initial_prefill_embedding_rows += 1;
            self.initial_prefill_embedding_bytes_read += embedding.bytes_read;
        }
        self.apply_prefill_vision_embeddings(
            vision_embeddings.unwrap_or_default(),
            0,
            token_ids.len(),
        )?;
        Ok(())
    }

    pub(super) fn seed_bounded_prefill_token_embedding_chunk(
        &mut self,
        catalog: &TensorCatalog,
        token_ids: &[usize],
        row_start: usize,
        vision_embeddings: Option<&[glmrt_api::RealFullVisionEmbedding]>,
    ) -> Result<()> {
        anyhow::ensure!(
            self.live_request && !token_ids.is_empty(),
            "bounded prefill embedding chunks require a non-empty live request"
        );
        let row_end = row_start
            .checked_add(token_ids.len())
            .context("bounded prefill embedding row range overflow")?;
        anyhow::ensure!(
            row_end <= self.shape.prefill_rows,
            "bounded prefill embedding rows {row_start}..{row_end} exceed {} prompt rows",
            self.shape.prefill_rows
        );
        dump_layer_boundary_token_ids(
            RowSourceKind::PrefillChunk,
            self.shape.prefix_tokens + row_start,
            token_ids,
        )?;
        let bytes_read = self
            .seed_device_token_embedding_segment(catalog, token_ids, row_start)?
            .context("bounded prefill requires device-resident token embeddings")?;
        self.initial_prefill_embedding_rows = self
            .initial_prefill_embedding_rows
            .checked_add(token_ids.len())
            .context("bounded prefill embedding row accounting overflow")?;
        self.initial_prefill_embedding_bytes_read = self
            .initial_prefill_embedding_bytes_read
            .checked_add(bytes_read)
            .context("bounded prefill embedding byte accounting overflow")?;
        self.apply_prefill_vision_embeddings(
            vision_embeddings.unwrap_or_default(),
            row_start,
            token_ids.len(),
        )?;
        Ok(())
    }

    fn apply_prefill_vision_embeddings(
        &mut self,
        vision_embeddings: &[glmrt_api::RealFullVisionEmbedding],
        row_start: usize,
        row_count: usize,
    ) -> Result<()> {
        if vision_embeddings.is_empty() || row_count == 0 {
            return Ok(());
        }
        let row_end = row_start
            .checked_add(row_count)
            .context("vision embedding destination row range overflows usize")?;
        let absolute_start = self
            .shape
            .prefix_tokens
            .checked_add(row_start)
            .context("vision embedding absolute row start overflows usize")?;
        let absolute_end = self
            .shape
            .prefix_tokens
            .checked_add(row_end)
            .context("vision embedding absolute row end overflows usize")?;
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("vision embedding row bytes overflow usize")?;

        for embedding in vision_embeddings {
            anyhow::ensure!(
                embedding.hidden_size == NUMERIC_PROGRESS_HIDDEN_DIM
                    && embedding.hidden_bf16.len()
                        == embedding
                            .rows
                            .checked_mul(row_bytes)
                            .context("vision embedding source byte count overflows usize")?,
                "vision embedding {} shape {}x{} has {} BF16 bytes",
                embedding.image_sha256,
                embedding.rows,
                embedding.hidden_size,
                embedding.hidden_bf16.len()
            );
            let embedding_end = embedding
                .token_start
                .checked_add(embedding.rows)
                .context("vision embedding prompt row range overflows usize")?;
            let overlap_start = absolute_start.max(embedding.token_start);
            let overlap_end = absolute_end.min(embedding_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let source_byte_start = overlap_start
                .checked_sub(embedding.token_start)
                .and_then(|rows| rows.checked_mul(row_bytes))
                .context("vision embedding source byte start overflows usize")?;
            let overlap_rows = overlap_end - overlap_start;
            let overlap_bytes = overlap_rows
                .checked_mul(row_bytes)
                .context("vision embedding overlap byte count overflows usize")?;
            let source_byte_end = source_byte_start
                .checked_add(overlap_bytes)
                .context("vision embedding source byte end overflows usize")?;
            let destination_row = overlap_start
                .checked_sub(self.shape.prefix_tokens)
                .context("vision embedding starts before scheduler suffix")?;
            let destination_byte_start = destination_row
                .checked_mul(row_bytes)
                .context("vision embedding destination byte start overflows usize")?;
            let destination_byte_end = destination_byte_start
                .checked_add(overlap_bytes)
                .context("vision embedding destination byte end overflows usize")?;
            anyhow::ensure!(
                destination_byte_end <= self.residual_bf16.len(),
                "vision embedding destination bytes {destination_byte_start}..{destination_byte_end} exceed {}",
                self.residual_bf16.len()
            );
            let source = &embedding.hidden_bf16[source_byte_start..source_byte_end];
            self.residual_bf16[destination_byte_start..destination_byte_end]
                .copy_from_slice(source);

            let mut destination_cursor = destination_byte_start;
            for (key, hidden) in &mut self.device_hidden_segments {
                if key.byte_end <= destination_cursor
                    || key.byte_start >= destination_byte_end
                    || destination_cursor >= destination_byte_end
                {
                    continue;
                }
                anyhow::ensure!(
                    key.byte_start <= destination_cursor,
                    "live vision prompt bytes have a gap at {destination_cursor} before resident segment {}..{}",
                    key.byte_start,
                    key.byte_end
                );
                let copy_end = key.byte_end.min(destination_byte_end);
                let source_start = destination_cursor - destination_byte_start;
                let source_end = source_start + (copy_end - destination_cursor);
                hidden.overwrite_from_host_bytes(
                    destination_cursor - key.byte_start,
                    &source[source_start..source_end],
                    "MoonViT projected prompt embeddings",
                )?;
                destination_cursor = copy_end;
            }
            anyhow::ensure!(
                !self.live_request || destination_cursor == destination_byte_end,
                "live vision prompt rows {overlap_start}..{overlap_end} have no resident embedding segment"
            );
        }
        Ok(())
    }

    pub(super) fn release_bounded_prefill_device_hidden_chunk(
        &mut self,
        row_start: usize,
        row_count: usize,
    ) -> Result<()> {
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("bounded prefill hidden row bytes overflow")?;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .context("bounded prefill hidden byte start overflow")?;
        let byte_end = row_count
            .checked_mul(row_bytes)
            .and_then(|bytes| byte_start.checked_add(bytes))
            .context("bounded prefill hidden byte end overflow")?;
        let hidden = self
            .device_hidden_segments
            .remove(&DeviceHiddenSegmentKey {
                byte_start,
                byte_end,
            })
            .with_context(|| {
                format!(
                    "bounded prefill final hidden rows {row_start}..{} are not resident",
                    row_start + row_count
                )
            })?;
        anyhow::ensure!(
            hidden.rows == row_count && hidden.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
            "bounded prefill final hidden chunk shape mismatch"
        );
        Ok(())
    }

    pub(super) fn seed_decode_token_embeddings(
        &mut self,
        catalog: &TensorCatalog,
        token_ids: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            self.selected_prefill_rows == 0
                && self.selected_decode_rows == 0
                && self.selected_mtp_rows == 0,
            "scheduler decode embeddings must be seeded before applying selected rows"
        );
        anyhow::ensure!(
            token_ids.len() == self.shape.decode_rows,
            "scheduler decode embedding seed token count {} does not match decode rows {}",
            token_ids.len(),
            self.shape.decode_rows
        );
        dump_layer_boundary_token_ids(
            RowSourceKind::DecodeStep,
            self.shape
                .prefix_tokens
                .checked_add(self.shape.prefill_rows)
                .context("scheduler decode diagnostic token start overflows usize")?,
            token_ids,
        )?;
        if self.live_request {
            let mut device_bytes_read = 0_u64;
            let mut device_rows = 0_usize;
            for (row_index, token_id) in token_ids.iter().copied().enumerate() {
                let row_start = self
                    .shape
                    .prefill_rows
                    .checked_add(row_index)
                    .context("scheduler live decode embedding row offset overflows usize")?;
                match self.seed_device_token_embedding_segment(catalog, &[token_id], row_start)? {
                    Some(bytes_read) => {
                        device_bytes_read += bytes_read;
                        device_rows += 1;
                    }
                    None if row_index == 0 => break,
                    None => {
                        anyhow::bail!(
                            "scheduler live decode embedding device seeding became unavailable after {row_index} rows"
                        );
                    }
                }
            }
            if device_rows == token_ids.len() {
                self.initial_decode_embedding_rows += token_ids.len();
                self.initial_decode_embedding_bytes_read += device_bytes_read;
                return Ok(());
            }
        }
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        for (row_index, token_id) in token_ids.iter().copied().enumerate() {
            let embedding = real_full_embedding_hidden_for_token(catalog, token_id)
                .with_context(|| format!("loading embedding hidden for decode token {token_id}"))?;
            let embedding_bf16 = bf16_bytes_from_f32(&embedding.hidden);
            anyhow::ensure!(
                embedding_bf16.len() == row_bytes,
                "scheduler decode embedding row byte length mismatch for token {}: expected {} got {}",
                token_id,
                row_bytes,
                embedding_bf16.len()
            );
            let row_offset = self
                .shape
                .prefill_rows
                .checked_add(row_index)
                .context("scheduler decode embedding row offset overflows usize")?;
            let byte_start = row_offset
                .checked_mul(row_bytes)
                .context("scheduler decode embedding byte offset overflows usize")?;
            let byte_end = byte_start + row_bytes;
            self.residual_bf16[byte_start..byte_end].copy_from_slice(&embedding_bf16);
            self.initial_decode_embedding_rows += 1;
            self.initial_decode_embedding_bytes_read += embedding.bytes_read;
        }
        Ok(())
    }

    pub(super) fn seed_mtp_token_embeddings(
        &mut self,
        catalog: &TensorCatalog,
        token_ids: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            self.selected_prefill_rows == 0
                && self.selected_decode_rows == 0
                && self.selected_mtp_rows == 0,
            "scheduler MTP embeddings must be seeded before applying selected rows"
        );
        anyhow::ensure!(
            token_ids.len() == self.shape.mtp_rows,
            "scheduler MTP embedding seed token count {} does not match MTP rows {}",
            token_ids.len(),
            self.shape.mtp_rows
        );
        if token_ids.is_empty() {
            return Ok(());
        }

        let row_start = self
            .shape
            .prefill_rows
            .checked_add(self.shape.decode_rows)
            .context("scheduler live MTP embedding row offset overflows usize")?;
        dump_layer_boundary_token_ids(
            RowSourceKind::MtpVerifyBlock,
            self.shape
                .prefix_tokens
                .checked_add(row_start)
                .context("scheduler MTP diagnostic token start overflows usize")?,
            token_ids,
        )?;
        if self.live_request
            && self
                .seed_device_token_embedding_segment(catalog, token_ids, row_start)?
                .is_some()
        {
            return Ok(());
        }

        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        for (row_index, token_id) in token_ids.iter().copied().enumerate() {
            let embedding = real_full_embedding_hidden_for_token(catalog, token_id)
                .with_context(|| format!("loading embedding hidden for MTP token {token_id}"))?;
            let embedding_bf16 = bf16_bytes_from_f32(&embedding.hidden);
            anyhow::ensure!(
                embedding_bf16.len() == row_bytes,
                "scheduler MTP embedding row byte length mismatch for token {}: expected {} got {}",
                token_id,
                row_bytes,
                embedding_bf16.len()
            );
            let byte_start = row_start
                .checked_add(row_index)
                .and_then(|row| row.checked_mul(row_bytes))
                .context("scheduler MTP embedding byte offset overflows usize")?;
            let byte_end = byte_start + row_bytes;
            self.residual_bf16[byte_start..byte_end].copy_from_slice(&embedding_bf16);
        }
        Ok(())
    }

    pub(super) fn seed_external_decode_hidden(
        &mut self,
        token_position: usize,
        hidden_bf16: &[u8],
    ) -> Result<()> {
        anyhow::ensure!(
            self.live_request && self.shape.decode_rows == 1,
            "external scheduler hidden currently requires one live decode row"
        );
        let expected_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("external scheduler hidden byte count overflow")?;
        anyhow::ensure!(
            hidden_bf16.len() == expected_bytes,
            "external scheduler hidden bytes {} did not match {expected_bytes}",
            hidden_bf16.len()
        );
        let row_index =
            self.numeric_progression_row_index(RowSourceKind::DecodeStep, token_position, 0)?;
        let byte_start = row_index
            .checked_mul(expected_bytes)
            .context("external scheduler hidden byte offset overflow")?;
        let byte_end = byte_start + expected_bytes;
        self.residual_bf16[byte_start..byte_end].copy_from_slice(hidden_bf16);
        let hidden = device_bf16_output_from_bf16_bytes(
            hidden_bf16,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "Spark layer-block ingress hidden",
        )?;
        self.seed_external_decode_device_hidden(token_position, hidden)
    }

    pub(super) fn seed_external_decode_device_hidden(
        &mut self,
        token_position: usize,
        hidden: DeviceBf16Output,
    ) -> Result<()> {
        anyhow::ensure!(
            self.live_request && self.shape.decode_rows == 1,
            "external scheduler device hidden currently requires one live decode row"
        );
        anyhow::ensure!(
            hidden.rows == 1 && hidden.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
            "external scheduler device hidden shape must be 1x{}, got {}x{}",
            NUMERIC_PROGRESS_HIDDEN_DIM,
            hidden.rows,
            hidden.values_per_row
        );
        let expected_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("external scheduler device hidden byte count overflow")?;
        anyhow::ensure!(
            hidden.buffer().bytes >= expected_bytes,
            "external scheduler device hidden buffer has {} bytes, expected at least {expected_bytes}",
            hidden.buffer().bytes
        );
        let row_index =
            self.numeric_progression_row_index(RowSourceKind::DecodeStep, token_position, 0)?;
        let byte_start = row_index
            .checked_mul(expected_bytes)
            .context("external scheduler device hidden byte offset overflow")?;
        let byte_end = byte_start + expected_bytes;
        self.device_hidden_segments.insert(
            DeviceHiddenSegmentKey {
                byte_start,
                byte_end,
            },
            hidden,
        );
        Ok(())
    }

    pub(super) fn seed_external_prefill_decode_device_hidden(
        &mut self,
        token_start: usize,
        hidden: DeviceBf16Output,
    ) -> Result<()> {
        anyhow::ensure!(
            self.live_request && self.shape.decode_rows == 1 && self.shape.mtp_rows == 0,
            "external scheduler prefill/decode hidden requires one live decode row and no MTP rows"
        );
        anyhow::ensure!(
            token_start == self.shape.prefix_tokens,
            "external scheduler prefill/decode token start {token_start} does not match prefix {}",
            self.shape.prefix_tokens
        );
        let total_rows = self
            .shape
            .prefill_rows
            .checked_add(self.shape.decode_rows)
            .context("external scheduler prefill/decode row count overflow")?;
        anyhow::ensure!(
            hidden.rows == total_rows && hidden.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
            "external scheduler prefill/decode hidden shape must be {}x{}, got {}x{}",
            total_rows,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            hidden.rows,
            hidden.values_per_row
        );
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("external scheduler prefill/decode row bytes overflow")?;
        let total_bytes = total_rows
            .checked_mul(row_bytes)
            .context("external scheduler prefill/decode bytes overflow")?;
        anyhow::ensure!(
            hidden.buffer().bytes >= total_bytes,
            "external scheduler prefill/decode buffer has {} bytes, expected at least {total_bytes}",
            hidden.buffer().bytes
        );

        if self.shape.prefill_rows > 0 {
            let prefill_bytes = self
                .shape
                .prefill_rows
                .checked_mul(row_bytes)
                .context("external scheduler prefill hidden bytes overflow")?;
            let prefill_view = device_buffer_byte_view(
                hidden.buffer(),
                0,
                prefill_bytes,
                "Spark layer-block prefill ingress",
            )?;
            let prefill = device_bf16_output_from_device_template_buffer(
                prefill_view,
                self.shape.prefill_rows,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                "Spark layer-block prefill ingress",
            )?;
            self.device_hidden_segments.insert(
                DeviceHiddenSegmentKey {
                    byte_start: 0,
                    byte_end: prefill_bytes,
                },
                prefill,
            );
        }

        let decode_row_index = self.shape.prefill_rows;
        let decode_byte_start = decode_row_index
            .checked_mul(row_bytes)
            .context("external scheduler decode hidden offset overflow")?;
        let decode_view = device_buffer_byte_view(
            hidden.buffer(),
            decode_byte_start,
            row_bytes,
            "Spark layer-block final ingress row",
        )?;
        let decode = device_bf16_output_from_device_template_buffer(
            decode_view,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "Spark layer-block final ingress row",
        )?;
        self.device_hidden_segments.insert(
            DeviceHiddenSegmentKey {
                byte_start: decode_byte_start,
                byte_end: total_bytes,
            },
            decode,
        );
        Ok(())
    }

    fn seed_device_token_embedding_segment(
        &mut self,
        catalog: &TensorCatalog,
        token_ids: &[usize],
        row_start: usize,
    ) -> Result<Option<u64>> {
        let Some(embedding) = real_full_embedding_device_hidden_for_tokens(catalog, token_ids)?
        else {
            return Ok(None);
        };
        if embedding.token_count != token_ids.len()
            || embedding.device_hidden.rows != token_ids.len()
            || embedding.device_hidden.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM
        {
            anyhow::bail!(
                "scheduler live embedding device output shape mismatch: tokens={} token_count={} output={}x{}",
                token_ids.len(),
                embedding.token_count,
                embedding.device_hidden.rows,
                embedding.device_hidden.values_per_row
            );
        }
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler live embedding row byte count overflows usize")?;
        let byte_start = row_start
            .checked_mul(row_bytes)
            .context("scheduler live embedding byte start overflows usize")?;
        let byte_count = token_ids
            .len()
            .checked_mul(row_bytes)
            .context("scheduler live embedding byte count overflows usize")?;
        let byte_end = byte_start
            .checked_add(byte_count)
            .context("scheduler live embedding byte end overflows usize")?;
        if byte_end > self.residual_bf16.len() {
            anyhow::bail!(
                "scheduler live embedding rows {row_start}..{} exceed residual rows {}",
                row_start + token_ids.len(),
                self.shape.unique_rows()
            );
        }
        self.device_hidden_segments.insert(
            DeviceHiddenSegmentKey {
                byte_start,
                byte_end,
            },
            embedding.device_hidden,
        );
        Ok(Some(embedding.bytes_read))
    }

    pub(super) fn apply_selected(
        &mut self,
        layer_id: usize,
        catalog: &TensorCatalog,
        selected: &[LayerWave],
        attention_deltas: &[RealFullSchedulerDeviceAttentionDelta],
        sparse_batch_graph_bucket: GraphBucket,
        quantization_recipe: &str,
    ) -> Result<()> {
        if self.should_apply_selected_sparse_tcp_batched(layer_id, selected) {
            return self
                .apply_selected_sparse_tcp_batched(
                    layer_id,
                    catalog,
                    selected,
                    attention_deltas,
                    sparse_batch_graph_bucket,
                    quantization_recipe,
                )
                .with_context(|| {
                    format!(
                        "applying scheduler sparse routed ProtocolV2 TCP batch for layer {layer_id}"
                    )
                });
        }

        for wave in selected {
            for source in &wave.row_sources {
                let attention_delta = attention_deltas.iter().find(|delta| {
                    delta.kind == source.kind
                        && delta.token_start == source.token_start.0 as usize
                        && delta.row_count == source.row_count
                });
                let start_row_index = self
                    .numeric_progression_row_index(source.kind, source.token_start.0 as usize, 0)
                    .with_context(|| {
                        format!(
                            "mapping numeric progression row kind={:?} start",
                            source.kind
                        )
                    })?;
                self.apply_source_delta(
                    layer_id,
                    catalog,
                    start_row_index,
                    source.row_count,
                    source.kind,
                    source,
                    wave.graph_bucket,
                    &wave.placement_version,
                    attention_delta,
                )
                .with_context(|| {
                    format!(
                        "applying numeric progression row kind={:?} rows={}",
                        source.kind, source.row_count
                    )
                })?;
                self.record_selected_rows(source.kind, source.row_count);
            }
        }
        Ok(())
    }

    fn should_apply_selected_sparse_tcp_batched(
        &self,
        layer_id: usize,
        selected: &[LayerWave],
    ) -> bool {
        coordinator_cuda_reference_kernels_enabled()
            && self.sparse_tcp_routed_mlp.is_some()
            && layer_id >= GLM52_FIRST_K_DENSE_REPLACE
            && selected
                .iter()
                .flat_map(|wave| wave.row_sources.iter())
                .any(|source| source.row_count > 0)
            && selected
                .iter()
                .flat_map(|wave| wave.row_sources.iter())
                .all(|source| source.kind != RowSourceKind::Benchmark)
    }

    pub(super) fn can_pipeline_selected_sparse_tcp_batched(
        &self,
        layer_id: usize,
        selected: &[LayerWave],
    ) -> bool {
        if !self.should_apply_selected_sparse_tcp_batched(layer_id, selected) {
            return false;
        }
        let rows = selected
            .iter()
            .flat_map(|wave| wave.row_sources.iter())
            .map(|source| source.row_count)
            .sum::<usize>();
        self.sparse_tcp_routed_mlp.as_ref().is_some_and(|context| {
            rows > 0
                && (rows <= context.max_global_rows_per_dispatch
                    || (context.max_global_rows_per_dispatch > 0
                        && context.dispatch_worker.supports_streaming_responses()))
        })
    }

    fn apply_selected_sparse_tcp_batched(
        &mut self,
        layer_id: usize,
        catalog: &TensorCatalog,
        selected: &[LayerWave],
        attention_deltas: &[RealFullSchedulerDeviceAttentionDelta],
        sparse_batch_graph_bucket: GraphBucket,
        quantization_recipe: &str,
    ) -> Result<()> {
        let Some(pending) = self.start_apply_selected_sparse_tcp_batched(
            layer_id,
            catalog,
            selected,
            attention_deltas,
            sparse_batch_graph_bucket,
            quantization_recipe,
        )?
        else {
            return Ok(());
        };
        self.finish_apply_selected_sparse_tcp_batched(pending)
    }

    pub(super) fn start_apply_selected_sparse_tcp_batched(
        &mut self,
        layer_id: usize,
        catalog: &TensorCatalog,
        selected: &[LayerWave],
        attention_deltas: &[RealFullSchedulerDeviceAttentionDelta],
        sparse_batch_graph_bucket: GraphBucket,
        quantization_recipe: &str,
    ) -> Result<Option<SchedulerSparseTcpPendingApply>> {
        self.prepare_apply_selected_sparse_tcp_batched(
            layer_id,
            catalog,
            selected,
            attention_deltas,
            sparse_batch_graph_bucket,
            quantization_recipe,
        )?
        .map(|prepared| self.start_prepared_sparse_tcp_dispatch(catalog, prepared, true))
        .transpose()
    }

    pub(super) fn prepare_apply_selected_sparse_tcp_batched(
        &mut self,
        layer_id: usize,
        catalog: &TensorCatalog,
        selected: &[LayerWave],
        attention_deltas: &[RealFullSchedulerDeviceAttentionDelta],
        sparse_batch_graph_bucket: GraphBucket,
        quantization_recipe: &str,
    ) -> Result<Option<SchedulerSparseTcpPreparedDispatch>> {
        anyhow::ensure!(
            self.should_apply_selected_sparse_tcp_batched(layer_id, selected),
            "scheduler sparse TCP split apply requires an eligible sparse batch"
        );
        let mut normalized_readback_scratch =
            std::mem::take(&mut self.device_sparse_routed_normalized_readback_bf16_scratch);
        let result = self.prepare_apply_selected_sparse_tcp_batched_with_scratch(
            layer_id,
            catalog,
            selected,
            attention_deltas,
            sparse_batch_graph_bucket,
            quantization_recipe,
            &mut normalized_readback_scratch,
        );
        self.device_sparse_routed_normalized_readback_bf16_scratch = normalized_readback_scratch;
        result
    }

    fn prepare_apply_selected_sparse_tcp_batched_with_scratch(
        &mut self,
        layer_id: usize,
        catalog: &TensorCatalog,
        selected: &[LayerWave],
        attention_deltas: &[RealFullSchedulerDeviceAttentionDelta],
        sparse_batch_graph_bucket: GraphBucket,
        quantization_recipe: &str,
        normalized_readback_scratch: &mut Vec<u8>,
    ) -> Result<Option<SchedulerSparseTcpPreparedDispatch>> {
        let Some(first_wave) = selected.first() else {
            return Ok(None);
        };
        let mut batch = ExpertBatch::from_wave_with_envelope(
            first_wave,
            DType::Bf16,
            quantization_recipe.to_owned(),
            sparse_batch_graph_bucket,
        )
        .with_context(|| {
            format!("building scheduler sparse TCP ExpertBatch for layer {layer_id}")
        })?;
        for wave in &selected[1..] {
            batch
                .try_append_wave(wave, DType::Bf16, quantization_recipe.to_owned())
                .with_context(|| {
                    format!("appending selected scheduler wave to sparse TCP batch for layer {layer_id}")
                })?;
        }
        if batch.num_rows() == 0 {
            return Ok(None);
        }
        let packed_hidden_exchange =
            b12x_packed_hidden_exchange_enabled_for_layer(catalog, layer_id)?;
        let stage_timing_enabled = sparse_tcp_stage_timing_enabled()
            || sparse_prefill_frontier_timing_sample(layer_id, batch.num_rows());
        let stage_total_start = stage_timing_enabled.then(Instant::now);
        let mut attention_delta_ms = 0.0_f64;
        let mut norm_ms = 0.0_f64;
        let mut normalized_readback_ms = 0.0_f64;
        let mut router_ms = 0.0_f64;

        let normalized_row_stride_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler sparse TCP hidden row stride overflow")?;
        anyhow::ensure!(
            batch.hidden_dim == NUMERIC_PROGRESS_HIDDEN_DIM
                && batch.hidden_bytes_per_row == normalized_row_stride_bytes
                && batch.hidden_dtype == DType::Bf16,
            "scheduler sparse TCP batch hidden envelope mismatch: dim={} row_bytes={} dtype={:?}",
            batch.hidden_dim,
            batch.hidden_bytes_per_row,
            batch.hidden_dtype
        );
        let exchange_row_stride_bytes = if packed_hidden_exchange {
            nvfp4_e2m1_fp8_e4m3_row_bytes(NUMERIC_PROGRESS_HIDDEN_DIM)?
        } else {
            normalized_row_stride_bytes
        };
        if packed_hidden_exchange {
            batch.hidden_bytes_per_row = exchange_row_stride_bytes;
            batch.hidden_dtype = DType::F4;
        }
        let total_hidden_bytes = batch
            .num_rows()
            .checked_mul(exchange_row_stride_bytes)
            .context("scheduler sparse TCP hidden payload byte count overflow")?;
        let mut global_hidden_payload = Vec::with_capacity(total_hidden_bytes);
        let mut scored_row_routes = Vec::<Vec<ScoredRoute>>::with_capacity(batch.num_rows());
        let mut prepared_segments =
            Vec::<SchedulerSparseTcpPreparedSegment>::with_capacity(selected.len());
        let mut prepared_batch_row_offset = 0_usize;
        let mut router_weight_bytes = 0_u64;
        let mut router_bias_bytes = 0_u64;

        for wave in selected {
            for source in &wave.row_sources {
                if source.row_count == 0 {
                    continue;
                }
                anyhow::ensure!(
                    source.kind != RowSourceKind::Benchmark,
                    "scheduler sparse TCP residual batching does not support benchmark sources"
                );
                let attention_delta = attention_deltas.iter().find(|delta| {
                    delta.kind == source.kind
                        && delta.token_start == source.token_start.0 as usize
                        && delta.row_count == source.row_count
                });
                let start_row_index = self
                    .numeric_progression_row_index(source.kind, source.token_start.0 as usize, 0)
                    .with_context(|| {
                        format!(
                            "mapping numeric progression row kind={:?} start",
                            source.kind
                        )
                    })?;
                let end_row_index = start_row_index
                    .checked_add(source.row_count)
                    .context("scheduler sparse TCP source row range overflows usize")?;
                let start = start_row_index
                    .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
                    .context("scheduler sparse TCP source value start overflows usize")?;
                let end = end_row_index
                    .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
                    .context("scheduler sparse TCP source value end overflows usize")?;
                let byte_start = start
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("scheduler sparse TCP source byte start overflows usize")?;
                let byte_end = end
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("scheduler sparse TCP source byte end overflows usize")?;
                if byte_end > self.residual_bf16.len() {
                    anyhow::bail!(
                        "numeric progression row range {start_row_index}..{end_row_index} exceeds residual rows {}",
                        self.shape.unique_rows()
                    );
                }
                let key = DeviceHiddenSegmentKey {
                    byte_start,
                    byte_end,
                };
                dump_layer_boundary_device_hidden(
                    layer_id,
                    source.kind,
                    source.token_start.0,
                    source.row_count,
                    "input",
                    &self.device_hidden_segments,
                    key,
                )?;

                let (deterministic_attention_delta, _) = numeric_progression_deltas(source.kind);
                self.source_segments += 1;
                let attention_delta_start = stage_timing_enabled.then(Instant::now);
                self.apply_attention_delta(
                    byte_start,
                    byte_end,
                    deterministic_attention_delta,
                    source.row_count,
                    source.kind,
                    attention_delta,
                )?;
                attention_delta_ms += elapsed_ms_optional(attention_delta_start);
                dump_layer_boundary_device_hidden(
                    layer_id,
                    source.kind,
                    source.token_start.0,
                    source.row_count,
                    "post_attention",
                    &self.device_hidden_segments,
                    key,
                )?;
                let (hidden_buffer, hidden_ready_event) = {
                    let hidden = self.device_hidden_segments.get(&key).context(
                        "scheduler resident hidden segment missing before sparse TCP MLP delta",
                    )?;
                    if hidden.rows != source.row_count
                        || hidden.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM
                    {
                        anyhow::bail!(
                            "scheduler resident hidden segment shape mismatch before sparse TCP MLP delta: expected {}x{} got {}x{}",
                            source.row_count,
                            NUMERIC_PROGRESS_HIDDEN_DIM,
                            hidden.rows,
                            hidden.values_per_row
                        );
                    }
                    (hidden.buffer(), hidden.ready_event())
                };

                let norm_start = stage_timing_enabled.then(Instant::now);
                let normalized = self
                    .device_real_sparse_post_attention_norm_from_hidden_async(
                        catalog,
                        layer_id,
                        hidden_buffer,
                        hidden_ready_event.as_deref(),
                        source.row_count,
                    )
                    .with_context(|| {
                        format!(
                            "building real sparse scheduler post-attention normalized hidden for layer {layer_id} kind={:?}",
                            source.kind
                        )
                    })?;
                norm_ms += elapsed_ms_optional(norm_start);
                let expected_bytes = source
                    .row_count
                    .checked_mul(normalized_row_stride_bytes)
                    .context("scheduler sparse TCP normalized hidden byte count overflow")?;
                validate_sparse_routed_normalized_device_hidden(
                    layer_id,
                    &normalized,
                    source.row_count,
                    expected_bytes,
                )?;
                let readback_start = stage_timing_enabled.then(Instant::now);
                let normalized_bf16_for_router =
                    Self::sparse_routed_normalized_host_bf16_for_validation_or_fallback(
                        layer_id,
                        &normalized,
                        expected_bytes,
                        normalized_readback_scratch,
                    )?;
                normalized_readback_ms += elapsed_ms_optional(readback_start);
                let pending_packed_hidden = packed_hidden_exchange
                    .then(|| begin_quantize_device_bf16_to_nvfp4_row_payload(&normalized))
                    .transpose()?;
                let router_start = stage_timing_enabled.then(Instant::now);
                let scoring = score_real_router_routes_bf16_cached_device_input(
                    catalog,
                    layer_id,
                    &normalized,
                    normalized_bf16_for_router,
                    NUMERIC_PROGRESS_HIDDEN_DIM,
                    GLM52_TOP_K,
                    &mut self.device_real_sparse_routed_mlp_router_cache,
                )
                .with_context(|| {
                    format!("scoring scheduler sparse TCP routed rows for layer {layer_id}")
                })?;
                router_ms += elapsed_ms_optional(router_start);
                if packed_hidden_exchange {
                    finish_quantize_device_bf16_to_nvfp4_row_payload(
                        pending_packed_hidden
                            .expect("packed hidden exchange started before router scoring"),
                        normalized_readback_scratch,
                    )
                    .context("finishing packed coordinator hidden exchange")?;
                    anyhow::ensure!(
                        normalized_readback_scratch.len()
                            == source.row_count * exchange_row_stride_bytes,
                        "scheduler sparse TCP packed hidden bytes {} did not match rows {} * stride {}",
                        normalized_readback_scratch.len(),
                        source.row_count,
                        exchange_row_stride_bytes
                    );
                    global_hidden_payload.extend_from_slice(normalized_readback_scratch);
                } else {
                    let normalized_bf16 = if let Some(normalized_bf16) = normalized_bf16_for_router
                    {
                        normalized_bf16
                    } else {
                        let readback_start = stage_timing_enabled.then(Instant::now);
                        let normalized_bf16 =
                            Self::read_sparse_routed_normalized_host_bf16_into_scratch(
                                layer_id,
                                &normalized,
                                expected_bytes,
                                normalized_readback_scratch,
                            )?;
                        normalized_readback_ms += elapsed_ms_optional(readback_start);
                        normalized_bf16
                    };
                    global_hidden_payload.extend_from_slice(normalized_bf16);
                }
                if scoring.row_routes.len() != source.row_count {
                    anyhow::bail!(
                        "scheduler sparse TCP router row count mismatch for layer {layer_id}: scored {} expected {}",
                        scoring.row_routes.len(),
                        source.row_count
                    );
                }
                record_backend(
                    &mut self.device_real_sparse_routed_mlp_router_backend,
                    scoring.router_backend,
                    "device-real-sparse-routed-router",
                )?;
                router_weight_bytes += scoring.router_weight_bytes_read;
                router_bias_bytes += scoring.router_bias_bytes_read;
                scored_row_routes.extend(scoring.row_routes);
                let batch_row_start = prepared_batch_row_offset;
                prepared_batch_row_offset = prepared_batch_row_offset
                    .checked_add(source.row_count)
                    .context("scheduler sparse TCP prepared row offset overflows usize")?;
                prepared_segments.push(SchedulerSparseTcpPreparedSegment {
                    byte_start,
                    byte_end,
                    batch_row_start,
                    row_count: source.row_count,
                    kind: source.kind,
                    token_start: source.token_start.0,
                    normalized,
                });
            }
        }

        anyhow::ensure!(
            global_hidden_payload.len() == total_hidden_bytes,
            "scheduler sparse TCP normalized hidden payload bytes {} did not match expected {total_hidden_bytes}",
            global_hidden_payload.len()
        );
        anyhow::ensure!(
            scored_row_routes.len() == batch.num_rows(),
            "scheduler sparse TCP scored row count {} did not match batch rows {}",
            scored_row_routes.len(),
            batch.num_rows()
        );
        anyhow::ensure!(
            prepared_batch_row_offset == batch.num_rows(),
            "scheduler sparse TCP prepared rows {} did not match batch rows {}",
            prepared_batch_row_offset,
            batch.num_rows()
        );
        let routes_start = stage_timing_enabled.then(Instant::now);
        let routes = scored_routes_for_scheduler_batch(&batch, &scored_row_routes)?;
        let route_count = routes.len();
        let routes_ms = elapsed_ms_optional(routes_start);
        log_moe_request_row_zero(layer_id, &batch, &global_hidden_payload, &routes);
        Ok(Some(SchedulerSparseTcpPreparedDispatch {
            layer_id,
            batch,
            prepared_segments,
            routes,
            hidden_payload: global_hidden_payload,
            route_count,
            router_weight_bytes,
            router_bias_bytes,
            stage_timing_enabled,
            stage_total_start,
            attention_delta_ms,
            norm_ms,
            normalized_readback_ms,
            router_ms,
            routes_ms,
        }))
    }

    fn start_prepared_sparse_tcp_dispatch(
        &mut self,
        catalog: &TensorCatalog,
        prepared: SchedulerSparseTcpPreparedDispatch,
        start_dispatch: bool,
    ) -> Result<SchedulerSparseTcpPendingApply> {
        let SchedulerSparseTcpPreparedDispatch {
            layer_id,
            batch,
            prepared_segments,
            routes,
            hidden_payload,
            route_count,
            router_weight_bytes,
            router_bias_bytes,
            stage_timing_enabled,
            stage_total_start,
            attention_delta_ms,
            norm_ms,
            normalized_readback_ms,
            router_ms,
            routes_ms,
        } = prepared;
        let dispatch_start = stage_timing_enabled.then(Instant::now);
        let mut pending_routes = Some(routes);
        let mut pending_hidden_payload = Some(hidden_payload);
        let pending_dispatch = if start_dispatch {
            let tcp_context = self.sparse_tcp_routed_mlp.as_mut().context(
                "scheduler sparse TCP routed MLP context missing during payload dispatch",
            )?;
            if tcp_context.can_start_dispatch_routed_delta_payload(&batch) {
                Some(
                    tcp_context
                        .start_dispatch_routed_delta_payload_owned(
                            batch.clone(),
                            pending_routes
                                .take()
                                .expect("scheduler sparse TCP routes are present"),
                            pending_hidden_payload
                                .take()
                                .expect("scheduler sparse TCP hidden payload is present"),
                        )
                        .with_context(|| {
                            format!(
                                "starting scheduler sparse routed ProtocolV2 TCP BF16 payload batch for layer {layer_id}"
                            )
                        })?,
                )
            } else {
                None
            }
        } else {
            None
        };
        let mut shared_mlp_ms = 0.0_f64;
        let mut ready_segments =
            Vec::<SchedulerSparseTcpReadySegment>::with_capacity(prepared_segments.len());
        for segment in prepared_segments {
            let shared_mlp_start = stage_timing_enabled.then(Instant::now);
            let shared_delta = self
                .device_real_sparse_shared_mlp_delta_from_normalized(
                    catalog,
                    layer_id,
                    segment.normalized.buffer(),
                    segment.row_count,
                )
                .with_context(|| {
                    format!(
                        "building real sparse shared checkpoint MLP delta for scheduler layer {layer_id} kind={:?}",
                        segment.kind
                        )
                    })?;
            shared_mlp_ms += elapsed_ms_optional(shared_mlp_start);
            ready_segments.push(SchedulerSparseTcpReadySegment {
                byte_start: segment.byte_start,
                byte_end: segment.byte_end,
                batch_row_start: segment.batch_row_start,
                row_count: segment.row_count,
                kind: segment.kind,
                token_start: segment.token_start,
                shared_delta,
            });
        }
        Ok(SchedulerSparseTcpPendingApply {
            layer_id,
            batch,
            ready_segments,
            pending_dispatch,
            pending_routes,
            pending_hidden_payload,
            completed_dispatch: None,
            route_count,
            router_weight_bytes,
            router_bias_bytes,
            stage_timing_enabled,
            stage_total_start,
            dispatch_start,
            attention_delta_ms,
            norm_ms,
            shared_mlp_ms,
            normalized_readback_ms,
            router_ms,
            routes_ms,
            incremental_stream: None,
            incremental_complete: false,
        })
    }

    pub(super) fn start_prepared_sparse_tcp_apply_without_dispatch(
        &mut self,
        catalog: &TensorCatalog,
        prepared: SchedulerSparseTcpPreparedDispatch,
    ) -> Result<SchedulerSparseTcpPendingApply> {
        self.start_prepared_sparse_tcp_dispatch(catalog, prepared, false)
    }

    pub(super) fn start_prepared_sparse_tcp_apply(
        &mut self,
        catalog: &TensorCatalog,
        prepared: SchedulerSparseTcpPreparedDispatch,
    ) -> Result<SchedulerSparseTcpPendingApply> {
        self.start_prepared_sparse_tcp_dispatch(catalog, prepared, true)
    }

    pub(super) fn try_start_sparse_tcp_cohort_dispatch(
        &mut self,
        prepared: &[&SchedulerSparseTcpPreparedDispatch],
    ) -> Result<Option<SchedulerSparseTcpCohortPendingDispatch>> {
        anyhow::ensure!(
            prepared.len() == 2,
            "scheduler sparse cohort dispatch requires exactly two members"
        );
        let mut merged_batch = prepared[0].batch.clone();
        let mut merged_routes = prepared[0].routes.clone();
        let mut merged_hidden_payload = prepared[0].hidden_payload.clone();
        let mut member_row_counts = vec![prepared[0].batch.num_rows()];
        let mut row_offset = prepared[0].batch.num_rows();
        for member in &prepared[1..] {
            anyhow::ensure!(
                member.layer_id == prepared[0].layer_id,
                "scheduler sparse cohort mixed layers {} and {}",
                prepared[0].layer_id,
                member.layer_id
            );
            let member_rows = member.batch.num_rows();
            let mut transport_batch = member.batch.clone();
            // API request placement labels are request-scoped even though every
            // member of this executor uses the same immutable expert owner map.
            // Normalize only the merged wire envelope; request-local batches
            // retain their original labels for accounting and state.
            transport_batch.placement_version = merged_batch.placement_version.clone();
            merged_batch = merged_batch
                .try_merge(&transport_batch)
                .context("merging scheduler sparse cohort expert batches")?;
            merged_routes.extend(member.routes.iter().cloned().map(|mut route| {
                route.row_index += row_offset;
                route
            }));
            merged_hidden_payload.extend_from_slice(&member.hidden_payload);
            member_row_counts.push(member_rows);
            row_offset = row_offset
                .checked_add(member_rows)
                .context("scheduler sparse cohort row count overflow")?;
        }
        anyhow::ensure!(
            row_offset == merged_batch.num_rows()
                && merged_routes.len() == merged_batch.route_count()
                && merged_hidden_payload.len()
                    == merged_batch.num_rows() * merged_batch.hidden_bytes_per_row,
            "scheduler sparse cohort transport envelope is inconsistent"
        );
        if real_full_moe_response_dtype_for_batch(&merged_batch)? != ExpertV2Dtype::Bf16 {
            return Ok(None);
        }
        let tcp_context = self
            .sparse_tcp_routed_mlp
            .as_mut()
            .context("scheduler sparse TCP routed MLP context missing during cohort dispatch")?;
        if !tcp_context.can_start_dispatch_routed_delta_payload(&merged_batch) {
            return Ok(None);
        }
        let handle = tcp_context
            .start_dispatch_routed_delta_payload_owned(
                merged_batch,
                merged_routes,
                merged_hidden_payload,
            )
            .context("starting combined scheduler sparse cohort dispatch")?;
        Ok(Some(SchedulerSparseTcpCohortPendingDispatch {
            handle,
            member_row_counts,
            total_rows: row_offset,
        }))
    }

    pub(super) fn finish_sparse_tcp_cohort_dispatch(
        &mut self,
        pending: SchedulerSparseTcpCohortPendingDispatch,
    ) -> Result<Vec<TcpProtocolV2HostBatchSetBf16PayloadDispatch>> {
        let SchedulerSparseTcpCohortPendingDispatch {
            handle,
            member_row_counts,
            total_rows,
        } = pending;
        let dispatch = self
            .sparse_tcp_routed_mlp
            .as_mut()
            .context("scheduler sparse TCP routed MLP context missing while finishing cohort")?
            .finish_dispatch_routed_delta_payload(handle)
            .context("finishing combined scheduler sparse cohort dispatch")?;
        let mut row_start = 0_usize;
        let mut members = Vec::with_capacity(member_row_counts.len());
        for row_count in member_row_counts {
            members.push(scheduler_sparse_tcp_payload_dispatch_for_segment(
                &dispatch,
                row_start,
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                row_count * GLM52_TOP_K,
            )?);
            row_start = row_start
                .checked_add(row_count)
                .context("scheduler sparse cohort result row count overflow")?;
        }
        anyhow::ensure!(
            row_start == total_rows,
            "scheduler sparse cohort split consumed {row_start} rows instead of {total_rows}"
        );
        Ok(members)
    }

    pub(super) fn record_sparse_tcp_cohort_member_dispatch(
        &mut self,
        pending: &SchedulerSparseTcpPendingApply,
        dispatch: &TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    ) -> Result<()> {
        let routes = pending
            .pending_routes
            .as_ref()
            .context("scheduler sparse cohort member routes are missing")?;
        self.sparse_tcp_routed_mlp
            .as_mut()
            .context("scheduler sparse TCP context missing while recording cohort member")?
            .record_payload_dispatch(
                SchedulerSparseTcpPayloadDispatchBatchShape::from_batch_and_routes(
                    &pending.batch,
                    routes,
                ),
                dispatch,
            )
    }

    pub(super) fn push_prepared_sparse_rolling_layer(
        &mut self,
        catalog: &TensorCatalog,
        rolling: &mut SchedulerSparseRollingLayerApply,
        task_index: usize,
        prepared: SchedulerSparseTcpPreparedDispatch,
        input_finished: bool,
        collect_timing: bool,
    ) -> Result<SchedulerSparseRollingPushTiming> {
        anyhow::ensure!(
            !rolling.input_finished,
            "rolling sparse layer {} received input after finish",
            rolling.layer_id
        );
        anyhow::ensure!(
            task_index == rolling.chunks.len(),
            "rolling sparse layer {} task {task_index} was not admitted after {}",
            rolling.layer_id,
            rolling.chunks.len().saturating_sub(1)
        );
        let SchedulerSparseTcpPreparedDispatch {
            layer_id,
            batch,
            prepared_segments,
            routes,
            hidden_payload,
            route_count,
            router_weight_bytes,
            router_bias_bytes,
            stage_timing_enabled,
            stage_total_start: _,
            attention_delta_ms,
            norm_ms,
            normalized_readback_ms,
            router_ms,
            routes_ms,
        } = prepared;
        anyhow::ensure!(
            layer_id == rolling.layer_id,
            "rolling sparse layer {} received prepared layer {layer_id}",
            rolling.layer_id
        );
        let global_row_start = rolling.admitted_rows;
        let row_count = batch.num_rows();
        let global_row_end = global_row_start
            .checked_add(row_count)
            .context("rolling sparse admitted row range overflow")?;
        anyhow::ensure!(
            row_count > 0 && global_row_end <= rolling.total_rows,
            "rolling sparse layer {} admitted rows {global_row_start}..{global_row_end} outside total {}",
            rolling.layer_id,
            rolling.total_rows
        );
        anyhow::ensure!(
            routes.len() == route_count && route_count == batch.route_count(),
            "rolling sparse prepared route count is inconsistent"
        );
        anyhow::ensure!(
            hidden_payload.len() == row_count * batch.hidden_bytes_per_row,
            "rolling sparse prepared hidden payload is inconsistent"
        );

        let mut ready_segments = Vec::with_capacity(prepared_segments.len());
        let mut shared_mlp_ms = 0.0_f64;
        let mut expected_local_row = 0_usize;
        for segment in prepared_segments {
            anyhow::ensure!(
                segment.batch_row_start == expected_local_row,
                "rolling sparse prepared segment starts at {} instead of {expected_local_row}",
                segment.batch_row_start
            );
            expected_local_row = expected_local_row
                .checked_add(segment.row_count)
                .context("rolling sparse prepared segment rows overflow")?;
            let shared_mlp_start = (stage_timing_enabled || collect_timing).then(Instant::now);
            let shared_delta = self
                .device_real_sparse_shared_mlp_delta_from_normalized(
                    catalog,
                    layer_id,
                    segment.normalized.buffer(),
                    segment.row_count,
                )
                .with_context(|| {
                    format!(
                        "building rolling sparse shared MLP delta for layer {layer_id} kind={:?}",
                        segment.kind
                    )
                })?;
            shared_mlp_ms += elapsed_ms_optional(shared_mlp_start);
            ready_segments.push(SchedulerSparseTcpReadySegment {
                byte_start: segment.byte_start,
                byte_end: segment.byte_end,
                batch_row_start: global_row_start + segment.batch_row_start,
                row_count: segment.row_count,
                kind: segment.kind,
                token_start: segment.token_start,
                shared_delta,
            });
        }
        anyhow::ensure!(
            expected_local_row == row_count,
            "rolling sparse prepared segments cover {expected_local_row} of {row_count} rows"
        );
        for segment in &ready_segments {
            rolling
                .accumulator
                .register_segment(segment.batch_row_start, segment.row_count)?;
        }

        let planner_start = Instant::now();
        let mut emissions = Vec::new();
        for local_row_start in (0..row_count).step_by(ROLLING_SPARSE_SOURCE_ADMISSION_ROWS) {
            let local_row_end =
                (local_row_start + ROLLING_SPARSE_SOURCE_ADMISSION_ROWS).min(row_count);
            let mut entries = Vec::with_capacity((local_row_end - local_row_start) * GLM52_TOP_K);
            for local_row_index in local_row_start..local_row_end {
                let row = batch
                    .rows
                    .get(local_row_index)
                    .context("rolling sparse prepared row is out of range")?;
                let route_end = row
                    .route_offset
                    .checked_add(row.route_count)
                    .context("rolling sparse prepared route range overflow")?;
                let row_routes = routes
                    .get(row.route_offset..route_end)
                    .context("rolling sparse prepared route range is out of bounds")?;
                for route in row_routes {
                    anyhow::ensure!(
                        route.row_index == local_row_index,
                        "rolling sparse route row {} did not match local row {local_row_index}",
                        route.row_index
                    );
                    entries.push(CompletionRoutePlanEntry {
                        row_index: global_row_start + local_row_index,
                        expert_id: route.expert_id,
                        intermediate_rows: 0,
                    });
                }
            }
            emissions.extend(
                rolling
                    .planner
                    .push_chunk(&entries, local_row_end - local_row_start)
                    .map_err(anyhow::Error::new)
                    .context("admitting a rolling sparse routed chunk")?,
            );
        }
        rolling.admitted_rows = global_row_end;
        if input_finished {
            anyhow::ensure!(
                rolling.admitted_rows == rolling.total_rows,
                "rolling sparse layer {} finished after {} of {} rows",
                rolling.layer_id,
                rolling.admitted_rows,
                rolling.total_rows
            );
            emissions.extend(
                rolling
                    .planner
                    .finish()
                    .map_err(anyhow::Error::new)
                    .context("draining rolling sparse row packs")?,
            );
            rebalance_scheduler_sparse_rolling_unsupported_tail(&mut emissions)?;
            rolling.input_finished = true;
        }

        rolling.emitted_packs = rolling
            .emitted_packs
            .checked_add(emissions.len())
            .context("rolling sparse emitted pack count overflow")?;
        rolling.chunks.push(SchedulerSparseRollingChunk {
            task_index,
            global_row_start,
            batch,
            routes,
            hidden_payload,
            finalized_segments: vec![false; ready_segments.len()],
            ready_segments,
            task_completed: false,
        });
        for emission in emissions {
            rolling.deadline_emissions += usize::from(emission.deadline_row_exclusive.is_some());
            rolling.max_selected_row_offset = rolling
                .max_selected_row_offset
                .max(emission.max_selected_row_offset);
            rolling.emitted_rows = rolling
                .emitted_rows
                .checked_add(emission.row_indices.len())
                .context("rolling sparse emitted row count overflow")?;
            let queued = build_scheduler_sparse_rolling_emission(&rolling.chunks, emission)
                .context("building a rolling sparse physical emission")?;
            if rolling.stage_timing_enabled || stage_timing_enabled {
                let pack_stats = scheduler_sparse_rolling_route_stats(
                    &queued.routes,
                    0,
                    queued.batch.num_rows(),
                )?;
                let slice_rows = scheduler_tcp_max_global_rows_per_dispatch()
                    .min(queued.batch.num_rows())
                    .max(1);
                let slice_stats = (0..queued.batch.num_rows())
                    .step_by(slice_rows)
                    .map(|row_start| {
                        scheduler_sparse_rolling_route_stats(
                            &queued.routes,
                            row_start,
                            (row_start + slice_rows).min(queued.batch.num_rows()),
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                eprintln!(
                    "real_full_rolling_sparse_pack_plan layer_id={} pack={} rows={} oldest_pending_row={} admitted_rows={} deadline_row_exclusive={:?} max_selected_row_offset={} unique_experts={} expert_tiles={} min_expert_rows={} max_expert_rows={} slice_unique_experts={:?} slice_expert_tiles={:?}",
                    rolling.layer_id,
                    queued.emission.emitted_pack_index,
                    queued.batch.num_rows(),
                    queued.emission.oldest_pending_row,
                    queued.emission.admitted_rows,
                    queued.emission.deadline_row_exclusive,
                    queued.emission.max_selected_row_offset,
                    pack_stats.unique_experts,
                    pack_stats.expert_tiles,
                    pack_stats.min_expert_rows,
                    pack_stats.max_expert_rows,
                    slice_stats
                        .iter()
                        .map(|stats| stats.unique_experts)
                        .collect::<Vec<_>>(),
                    slice_stats
                        .iter()
                        .map(|stats| stats.expert_tiles)
                        .collect::<Vec<_>>(),
                );
            }
            rolling.queued_emissions.push_back(queued);
        }
        rolling.route_count = rolling
            .route_count
            .checked_add(route_count)
            .context("rolling sparse route count overflow")?;
        rolling.router_weight_bytes = rolling
            .router_weight_bytes
            .checked_add(router_weight_bytes)
            .context("rolling sparse router weight byte count overflow")?;
        rolling.router_bias_bytes = rolling
            .router_bias_bytes
            .checked_add(router_bias_bytes)
            .context("rolling sparse router bias byte count overflow")?;
        rolling.stage_timing_enabled |= stage_timing_enabled;
        rolling.attention_delta_ms += attention_delta_ms;
        rolling.norm_ms += norm_ms;
        rolling.shared_mlp_ms += shared_mlp_ms;
        rolling.normalized_readback_ms += normalized_readback_ms;
        rolling.router_ms += router_ms;
        rolling.routes_ms += routes_ms;
        let planner_ms = elapsed_ms(planner_start);
        rolling.planner_ms += planner_ms;
        Ok(SchedulerSparseRollingPushTiming {
            shared_mlp_ms,
            planner_ms,
        })
    }

    pub(super) fn start_queued_sparse_rolling_dispatches(
        &mut self,
        rolling: &mut SchedulerSparseRollingLayerApply,
        budget: usize,
    ) -> Result<usize> {
        let mut started = 0_usize;
        while started < budget {
            let Some(queued) = rolling.queued_emissions.pop_front() else {
                break;
            };
            anyhow::ensure!(
                queued.batch.num_rows() > 1
                    && spark_expert_reduction_dispatch_for_rows(queued.batch.num_rows())?.is_some(),
                "rolling sparse emission rows {} do not support streamed Spark reduction",
                queued.batch.num_rows()
            );
            let handle = {
                let tcp_context = self
                    .sparse_tcp_routed_mlp
                    .as_mut()
                    .context("rolling sparse dispatch is missing its transport context")?;
                anyhow::ensure!(
                    tcp_context.dispatch_worker.supports_streaming_responses()
                        && tcp_context.can_start_dispatch_routed_delta_payload(&queued.batch),
                    "rolling sparse emission is not eligible for asynchronous streamed dispatch"
                );
                tcp_context.start_dispatch_routed_delta_payload_owned(
                    queued.batch,
                    queued.routes,
                    queued.hidden_payload,
                )?
            };
            anyhow::ensure!(
                handle.has_streaming_response_chunks(),
                "rolling sparse dispatch did not expose streamed response chunks"
            );
            rolling
                .pending_emissions
                .push_back(SchedulerSparseRollingPendingEmission {
                    emission: queued.emission,
                    handle,
                    completed_dispatch_row_slices: Vec::new(),
                });
            started += 1;
        }
        Ok(started)
    }

    fn push_sparse_rolling_response_chunks(
        rolling: &mut SchedulerSparseRollingLayerApply,
        response_chunks: &mut [VerbsHostProtocolV2HostBatchSetBf16PayloadChunk],
    ) -> Result<()> {
        let mut seen_global_rows = BTreeSet::new();
        let has_overlapping_rows = response_chunks.iter().any(|chunk| {
            chunk
                .global_row_indices
                .iter()
                .any(|row| !seen_global_rows.insert(*row))
        });
        if has_overlapping_rows {
            response_chunks.sort_by_key(|chunk| chunk.host_index);
        }
        let views = response_chunks
            .iter()
            .map(|chunk| StreamedSparseBAccumulatorChunk {
                partial_output: chunk.partial_output.as_ref(),
                global_row_indices: &chunk.global_row_indices,
                completed_global_rows: &chunk.completed_global_row_indices,
                output_dtype: chunk.output_dtype,
                output_row_stride_bytes: chunk.output_row_stride_bytes,
            })
            .collect::<Vec<_>>();
        rolling.accumulator.push_chunks(&views)
    }

    fn finalize_ready_sparse_rolling_segments(
        &mut self,
        rolling: &mut SchedulerSparseRollingLayerApply,
    ) -> Result<Vec<usize>> {
        let mut ready = Vec::new();
        for (chunk_index, chunk) in rolling.chunks.iter().enumerate() {
            for (segment_index, segment) in chunk.ready_segments.iter().enumerate() {
                if !chunk.finalized_segments[segment_index]
                    && rolling
                        .accumulator
                        .segment_ready(segment.batch_row_start, segment.row_count)?
                {
                    ready.push((chunk_index, segment_index));
                }
            }
        }

        let dispatch_transport = self
            .sparse_tcp_routed_mlp
            .as_ref()
            .map(|context| context.transport)
            .unwrap_or(RealFullSchedulerSparseDispatchTransport::Tcp);
        let mut completed_tasks = Vec::new();
        for (chunk_index, segment_index) in ready {
            let segment = rolling
                .chunks
                .get(chunk_index)
                .and_then(|chunk| chunk.ready_segments.get(segment_index))
                .context("rolling sparse ready segment is out of range")?;
            let key = DeviceHiddenSegmentKey {
                byte_start: segment.byte_start,
                byte_end: segment.byte_end,
            };
            let sparse_b_start = rolling.stage_timing_enabled.then(Instant::now);
            let output = {
                let residual = self.device_hidden_segments.get(&key).context(
                    "scheduler resident hidden segment missing before rolling Sparse-B residual",
                )?;
                rolling
                    .accumulator
                    .finalize_segment(&StreamedSparseBResidualSegment {
                        residual,
                        shared_delta: &segment.shared_delta,
                        row_start: segment.batch_row_start,
                        row_count: segment.row_count,
                    })?
            };
            rolling.sparse_b_ms += elapsed_ms_optional(sparse_b_start);
            self.record_real_sparse_shared_mlp_delta_accounting(
                rolling.layer_id,
                segment.row_count,
                segment.shared_delta.backend,
            )?;
            self.record_real_sparse_routed_mlp_delta_accounting(
                rolling.layer_id,
                segment.row_count,
                dispatch_transport.sparse_delta_backend(),
            )?;
            let apply_start = rolling.stage_timing_enabled.then(Instant::now);
            self.apply_device_residual_output_bytes(
                segment.byte_start,
                segment.byte_end,
                ResidualDeltaStage::Mlp,
                output,
            )?;
            dump_layer_boundary_device_hidden(
                rolling.layer_id,
                segment.kind,
                segment.token_start,
                segment.row_count,
                "post_mlp",
                &self.device_hidden_segments,
                key,
            )?;
            rolling.apply_ms += elapsed_ms_optional(apply_start);
            self.record_selected_rows(segment.kind, segment.row_count);
            rolling.finalized_rows = rolling
                .finalized_rows
                .checked_add(segment.row_count)
                .context("rolling sparse finalized row count overflow")?;
            let chunk = rolling
                .chunks
                .get_mut(chunk_index)
                .context("rolling sparse finalized chunk is out of range")?;
            chunk.finalized_segments[segment_index] = true;
            if !chunk.task_completed && chunk.finalized_segments.iter().all(|finalized| *finalized)
            {
                chunk.task_completed = true;
                completed_tasks.push(chunk.task_index);
            }
        }
        Ok(completed_tasks)
    }

    pub(super) fn try_progress_sparse_rolling_layer(
        &mut self,
        rolling: &mut SchedulerSparseRollingLayerApply,
    ) -> Result<SchedulerSparseRollingProgress> {
        if rolling.completion_validated {
            return Ok(SchedulerSparseRollingProgress {
                completed_task_indices: Vec::new(),
                made_progress: false,
                layer_complete: true,
            });
        }
        let mut made_progress = false;
        let mut completed_pending_index = None;
        let mut response_chunks = Vec::new();
        for pending_index in 0..rolling.pending_emissions.len() {
            let response = rolling
                .pending_emissions
                .get_mut(pending_index)
                .context("rolling sparse pending emission is out of range")?
                .handle
                .poll_streaming_response(false)?;
            match response {
                SchedulerSparseTcpPayloadStreamPoll::Pending => continue,
                SchedulerSparseTcpPayloadStreamPoll::Chunk(mut chunk) => {
                    let pending = rolling
                        .pending_emissions
                        .get_mut(pending_index)
                        .context("rolling sparse response emission is out of range")?;
                    if !chunk.completed_global_row_indices.is_empty() {
                        pending
                            .completed_dispatch_row_slices
                            .push(chunk.completed_global_row_indices.clone());
                    }
                    remap_sparse_payload_chunk_rows(&mut chunk, &pending.emission.row_indices)?;
                    response_chunks.push(chunk);
                    let mut response_rows = response_chunks[0].global_row_indices.len();
                    let response_batch_target_rows = pending.handle.response_batch_target_rows();
                    // Row-sharded Spark reduction completes disjoint rows on
                    // each rank. Collect every already-ready rank shard for
                    // this physical pack so its staged H2D/scatter path pays
                    // one synchronization instead of one per rank.
                    while (pending.handle.row_sharded_completion_chunks
                        || response_chunks
                            .iter()
                            .all(|chunk| chunk.completed_global_row_indices.is_empty()))
                        && response_rows < response_batch_target_rows
                    {
                        match pending.handle.poll_streaming_response(false)? {
                            SchedulerSparseTcpPayloadStreamPoll::Pending => break,
                            SchedulerSparseTcpPayloadStreamPoll::Chunk(mut chunk) => {
                                if !chunk.completed_global_row_indices.is_empty() {
                                    pending
                                        .completed_dispatch_row_slices
                                        .push(chunk.completed_global_row_indices.clone());
                                }
                                remap_sparse_payload_chunk_rows(
                                    &mut chunk,
                                    &pending.emission.row_indices,
                                )?;
                                response_rows = response_rows
                                    .checked_add(chunk.global_row_indices.len())
                                    .context("rolling sparse response batch rows overflow")?;
                                response_chunks.push(chunk);
                            }
                            SchedulerSparseTcpPayloadStreamPoll::Complete(dispatch) => {
                                pending.handle.deferred_streaming_completion = Some(dispatch);
                                break;
                            }
                        }
                    }
                    made_progress = true;
                    break;
                }
                SchedulerSparseTcpPayloadStreamPoll::Complete(dispatch) => {
                    completed_pending_index = Some((pending_index, dispatch));
                    made_progress = true;
                    break;
                }
            }
        }

        if !response_chunks.is_empty() {
            let sparse_b_start = rolling.stage_timing_enabled.then(Instant::now);
            Self::push_sparse_rolling_response_chunks(rolling, &mut response_chunks)?;
            rolling.sparse_b_ms += elapsed_ms_optional(sparse_b_start);
        }

        if let Some((pending_index, mut dispatch)) = completed_pending_index {
            let pending = rolling
                .pending_emissions
                .remove(pending_index)
                .context("completed rolling sparse emission is out of range")?;
            anyhow::ensure!(
                dispatch.partial_outputs_bf16_by_host.is_empty()
                    && dispatch.global_row_indices_by_host.is_empty()
                    && dispatch.completed_global_row_slices.is_empty(),
                "rolling sparse streamed dispatch unexpectedly returned collected payloads"
            );
            for global_row in &pending.emission.row_indices {
                anyhow::ensure!(
                    rolling.accumulator.segment_ready(*global_row, 1)?,
                    "rolling sparse dispatch completed before logical row {global_row}"
                );
            }
            dispatch.completed_global_row_slices = pending.completed_dispatch_row_slices;
            let dispatch_transport = self
                .sparse_tcp_routed_mlp
                .as_ref()
                .map(|context| context.transport)
                .unwrap_or(RealFullSchedulerSparseDispatchTransport::Tcp);
            let completed_routes = pending.handle.batch.routes;
            self.sparse_tcp_routed_mlp
                .as_mut()
                .context("rolling sparse completion is missing its transport context")?
                .finish_payload_dispatch_accounting(pending.handle, dispatch)?;
            record_backend(
                &mut self.device_real_sparse_routed_mlp_route_backend,
                dispatch_transport.sparse_route_backend(),
                "device-real-sparse-routed-nvfp4-route",
            )?;
            self.device_real_sparse_routed_mlp_routes += completed_routes;
        }

        let completed_task_indices = self.finalize_ready_sparse_rolling_segments(rolling)?;
        made_progress |= !completed_task_indices.is_empty();
        let layer_complete = rolling.input_finished
            && rolling.admitted_rows == rolling.total_rows
            && rolling.emitted_rows == rolling.total_rows
            && rolling.finalized_rows == rolling.total_rows
            && rolling.queued_emissions.is_empty()
            && rolling.pending_emissions.is_empty();
        if layer_complete {
            rolling.accumulator.validate_complete()?;
            self.device_real_sparse_routed_mlp_router_weight_bytes += rolling.router_weight_bytes;
            self.device_real_sparse_routed_mlp_router_bias_bytes += rolling.router_bias_bytes;
            let router_stats = self.device_real_sparse_routed_mlp_router_cache.stats();
            self.device_real_sparse_routed_mlp_router_cache_entries = router_stats.entries;
            self.device_real_sparse_routed_mlp_router_cache_hits = router_stats.cache_hits;
            if rolling.stage_timing_enabled {
                eprintln!(
                    "real_full_rolling_sparse_layer_timing layer_id={} rows={} routes={} packs={} deadline_packs={} max_selected_row_offset={} accumulator_peak_pages={} accumulator_peak_rows={} attention_delta_ms={:.3} norm_ms={:.3} shared_mlp_ms={:.3} normalized_readback_ms={:.3} router_ms={:.3} routes_ms={:.3} planner_ms={:.3} sparse_b_ms={:.3} apply_ms={:.3} total_ms={:.3}",
                    rolling.layer_id,
                    rolling.total_rows,
                    rolling.route_count,
                    rolling.emitted_packs,
                    rolling.deadline_emissions,
                    rolling.max_selected_row_offset,
                    rolling.accumulator.peak_active_pages(),
                    rolling.accumulator.peak_active_rows(),
                    rolling.attention_delta_ms,
                    rolling.norm_ms,
                    rolling.shared_mlp_ms,
                    rolling.normalized_readback_ms,
                    rolling.router_ms,
                    rolling.routes_ms,
                    rolling.planner_ms,
                    rolling.sparse_b_ms,
                    rolling.apply_ms,
                    elapsed_ms_optional(rolling.stage_started),
                );
            }
            rolling.completion_validated = true;
        }
        Ok(SchedulerSparseRollingProgress {
            completed_task_indices,
            made_progress,
            layer_complete,
        })
    }

    pub(super) fn progress_apply_selected_sparse_tcp_batched(
        &mut self,
        pending: &mut SchedulerSparseTcpPendingApply,
    ) -> Result<SchedulerSparseTcpApplyProgress> {
        self.progress_apply_selected_sparse_tcp_batched_inner(pending, true)?
            .context("blocking sparse dispatch progression made no progress")
    }

    pub(super) fn try_progress_apply_selected_sparse_tcp_batched(
        &mut self,
        pending: &mut SchedulerSparseTcpPendingApply,
    ) -> Result<Option<SchedulerSparseTcpApplyProgress>> {
        self.progress_apply_selected_sparse_tcp_batched_inner(pending, false)
    }

    fn progress_apply_selected_sparse_tcp_batched_inner(
        &mut self,
        pending: &mut SchedulerSparseTcpPendingApply,
        block: bool,
    ) -> Result<Option<SchedulerSparseTcpApplyProgress>> {
        if pending.incremental_complete {
            return Ok(Some(SchedulerSparseTcpApplyProgress {
                completed_segment_indices: Vec::new(),
                dispatch_complete: true,
            }));
        }
        anyhow::ensure!(
            pending.supports_incremental_stream(),
            "scheduler sparse apply is not eligible for incremental response progression"
        );
        if pending.incremental_stream.is_none() {
            let mut expected_row_start = 0_usize;
            for segment in &pending.ready_segments {
                anyhow::ensure!(
                    segment.batch_row_start == expected_row_start,
                    "incremental scheduler Sparse-B segment row start {} did not match expected {expected_row_start}",
                    segment.batch_row_start
                );
                expected_row_start = expected_row_start
                    .checked_add(segment.row_count)
                    .context("incremental scheduler Sparse-B segment rows overflow usize")?;
            }
            anyhow::ensure!(
                expected_row_start == pending.batch.num_rows(),
                "incremental scheduler Sparse-B segments cover {expected_row_start} rows instead of {}",
                pending.batch.num_rows()
            );
            pending.incremental_stream = Some(SchedulerSparseTcpIncrementalStream {
                accumulator: CudaStreamedSparseBAccumulator::new(
                    pending.batch.num_rows(),
                    NUMERIC_PROGRESS_HIDDEN_DIM,
                )?,
                finalized_segments: vec![false; pending.ready_segments.len()],
                completed_global_row_slices: Vec::new(),
                sparse_b_ms: 0.0,
                apply_ms: 0.0,
                stream_started: sparse_tcp_stage_timing_enabled().then(Instant::now),
            });
        }

        let response = {
            let handle = pending
                .pending_dispatch
                .as_mut()
                .context("incremental scheduler sparse apply is missing its dispatch handle")?;
            handle.poll_streaming_response(block)?
        };
        let chunk = match response {
            SchedulerSparseTcpPayloadStreamPoll::Pending => return Ok(None),
            SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk) => chunk,
            SchedulerSparseTcpPayloadStreamPoll::Complete(mut dispatch) => {
                let handle = pending.pending_dispatch.take().context(
                    "completed incremental scheduler sparse apply lost its dispatch handle",
                )?;
                anyhow::ensure!(
                dispatch.partial_outputs_bf16_by_host.is_empty()
                    && dispatch.global_row_indices_by_host.is_empty()
                    && dispatch.completed_global_row_slices.is_empty(),
                "incremental scheduler verbs-host dispatch unexpectedly returned collected payloads"
            );
                let stream = pending.incremental_stream.as_mut().context(
                    "completed incremental scheduler sparse apply lost its stream state",
                )?;
                stream.accumulator.validate_complete()?;
                anyhow::ensure!(
                    stream.finalized_segments.iter().all(|finalized| *finalized),
                    "incremental scheduler sparse apply left unfinalized segments"
                );
                dispatch.completed_global_row_slices =
                    std::mem::take(&mut stream.completed_global_row_slices);
                let dispatch_transport = self
                    .sparse_tcp_routed_mlp
                    .as_ref()
                    .map(|context| context.transport)
                    .unwrap_or(RealFullSchedulerSparseDispatchTransport::Tcp);
                {
                    let tcp_context = self.sparse_tcp_routed_mlp.as_mut().context(
                    "scheduler sparse TCP routed MLP context missing while completing incremental dispatch",
                )?;
                    tcp_context
                        .finish_payload_dispatch_accounting(handle, dispatch)
                        .with_context(|| {
                            format!(
                                "finishing incremental scheduler sparse routed batch for layer {}",
                                pending.layer_id
                            )
                        })?;
                }
                record_backend(
                    &mut self.device_real_sparse_routed_mlp_route_backend,
                    dispatch_transport.sparse_route_backend(),
                    "device-real-sparse-routed-nvfp4-route",
                )?;
                self.device_real_sparse_routed_mlp_routes += pending.route_count;
                self.device_real_sparse_routed_mlp_router_weight_bytes +=
                    pending.router_weight_bytes;
                self.device_real_sparse_routed_mlp_router_bias_bytes += pending.router_bias_bytes;
                let router_stats = self.device_real_sparse_routed_mlp_router_cache.stats();
                self.device_real_sparse_routed_mlp_router_cache_entries = router_stats.entries;
                self.device_real_sparse_routed_mlp_router_cache_hits = router_stats.cache_hits;
                if let Some(started) = stream.stream_started {
                    eprintln!(
                    "real_full_sparse_verbs_host_streamed_b_timing layer_id={} rows={} elapsed_ms={:.3}",
                    pending.layer_id,
                    pending.batch.num_rows(),
                    elapsed_ms(started)
                );
                }
                if pending.stage_timing_enabled {
                    eprintln!(
                    "real_full_sparse_{}_stage_timing layer_id={} rows={} routes={} attention_delta_ms={:.3} norm_ms={:.3} shared_mlp_ms={:.3} normalized_readback_ms={:.3} router_ms={:.3} routes_ms={:.3} dispatch_ms={:.3} sparse_b_ms={:.3} apply_ms={:.3} total_ms={:.3}",
                    dispatch_transport.label(),
                    pending.layer_id,
                    pending.batch.num_rows(),
                    pending.batch.route_count(),
                    pending.attention_delta_ms,
                    pending.norm_ms,
                    pending.shared_mlp_ms,
                    pending.normalized_readback_ms,
                    pending.router_ms,
                    pending.routes_ms,
                    elapsed_ms_optional(pending.dispatch_start),
                    stream.sparse_b_ms,
                    stream.apply_ms,
                    elapsed_ms_optional(pending.stage_total_start)
                );
                }
                pending.incremental_complete = true;
                return Ok(Some(SchedulerSparseTcpApplyProgress {
                    completed_segment_indices: Vec::new(),
                    dispatch_complete: true,
                }));
            }
        };
        let mut response_chunks = vec![chunk];
        let mut response_batch_rows = response_chunks[0].global_row_indices.len();
        let response_batch_target_rows = pending
            .pending_dispatch
            .as_ref()
            .context("incremental scheduler sparse apply lost its dispatch handle before coalescing responses")?
            .response_batch_target_rows();
        let row_sharded_completion_chunks = pending
            .pending_dispatch
            .as_ref()
            .expect("incremental sparse dispatch exists while coalescing responses")
            .row_sharded_completion_chunks;
        // Unlike overlapping partial reductions, each row-sharded completion
        // chunk owns complete, disjoint rows. Coalesce the logical batch before
        // the existing synchronized device staging operation.
        while (row_sharded_completion_chunks
            || response_chunks
                .iter()
                .all(|chunk| chunk.completed_global_row_indices.is_empty()))
            && response_batch_rows < response_batch_target_rows
        {
            let next = {
                let handle = pending.pending_dispatch.as_mut().context(
                    "incremental scheduler sparse apply lost its dispatch handle while coalescing responses",
                )?;
                match handle.poll_streaming_response(block)? {
                    SchedulerSparseTcpPayloadStreamPoll::Pending => None,
                    SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk) => Some(chunk),
                    SchedulerSparseTcpPayloadStreamPoll::Complete(dispatch) => {
                        handle.deferred_streaming_completion = Some(dispatch);
                        None
                    }
                }
            };
            let Some(next) = next else {
                break;
            };
            response_batch_rows = response_batch_rows
                .checked_add(next.global_row_indices.len())
                .context("incremental scheduler sparse response batch rows overflow usize")?;
            response_chunks.push(next);
        }
        {
            let stream = pending
                .incremental_stream
                .as_mut()
                .context("incremental scheduler sparse apply lost its stream state")?;
            if pending.batch.num_rows() <= 8
                && moe_payload_hash_diagnostic_enabled_for_layer(pending.layer_id)
            {
                for chunk in &response_chunks {
                    eprintln!(
                        "real_full_moe_payload_hash layer_id={} host_index={} rows={} bytes={} dtype={:?} stride={} fnv1a64={:016x}",
                        pending.layer_id,
                        chunk.host_index,
                        chunk.global_row_indices.len(),
                        chunk.partial_output.as_ref().len(),
                        chunk.output_dtype,
                        chunk.output_row_stride_bytes,
                        fnv1a64(chunk.partial_output.as_ref()),
                    );
                }
            }
            let mut seen_global_rows = BTreeSet::new();
            let has_overlapping_rows = response_chunks.iter().any(|chunk| {
                chunk
                    .global_row_indices
                    .iter()
                    .any(|row| !seen_global_rows.insert(*row))
            });
            if has_overlapping_rows {
                response_chunks.sort_by_key(|chunk| chunk.host_index);
                let views = response_chunks
                    .iter()
                    .map(|chunk| StreamedSparseBAccumulatorChunk {
                        partial_output: chunk.partial_output.as_ref(),
                        global_row_indices: &chunk.global_row_indices,
                        completed_global_rows: &chunk.completed_global_row_indices,
                        output_dtype: chunk.output_dtype,
                        output_row_stride_bytes: chunk.output_row_stride_bytes,
                    })
                    .collect::<Vec<_>>();
                stream.accumulator.push_host_ordered_chunks(&views)?;
            } else {
                let views = response_chunks
                    .iter()
                    .map(|chunk| StreamedSparseBAccumulatorChunk {
                        partial_output: chunk.partial_output.as_ref(),
                        global_row_indices: &chunk.global_row_indices,
                        completed_global_rows: &chunk.completed_global_row_indices,
                        output_dtype: chunk.output_dtype,
                        output_row_stride_bytes: chunk.output_row_stride_bytes,
                    })
                    .collect::<Vec<_>>();
                stream.accumulator.push_chunks(&views)?;
            }
            for chunk in response_chunks {
                if !chunk.completed_global_row_indices.is_empty() {
                    stream
                        .completed_global_row_slices
                        .push(chunk.completed_global_row_indices);
                }
            }
        }

        let ready_segment_indices = {
            let stream = pending
                .incremental_stream
                .as_ref()
                .context("incremental scheduler sparse apply lost its stream state")?;
            pending
                .ready_segments
                .iter()
                .enumerate()
                .filter_map(|(segment_index, segment)| {
                    (!stream.finalized_segments[segment_index]
                        && stream
                            .accumulator
                            .segment_ready(segment.batch_row_start, segment.row_count)
                            .ok()?)
                    .then_some(segment_index)
                })
                .collect::<Vec<_>>()
        };
        let dispatch_transport = self
            .sparse_tcp_routed_mlp
            .as_ref()
            .map(|context| context.transport)
            .unwrap_or(RealFullSchedulerSparseDispatchTransport::Tcp);
        for segment_index in &ready_segment_indices {
            let segment = pending
                .ready_segments
                .get(*segment_index)
                .context("incremental scheduler Sparse-B ready segment is out of range")?;
            let key = DeviceHiddenSegmentKey {
                byte_start: segment.byte_start,
                byte_end: segment.byte_end,
            };
            let sparse_b_start = pending.stage_timing_enabled.then(Instant::now);
            let output = {
                let residual = self.device_hidden_segments.get(&key).context(
                    "scheduler resident hidden segment missing before incremental Sparse-B residual",
                )?;
                let streamed_segment = StreamedSparseBResidualSegment {
                    residual,
                    shared_delta: &segment.shared_delta,
                    row_start: segment.batch_row_start,
                    row_count: segment.row_count,
                };
                pending
                    .incremental_stream
                    .as_mut()
                    .context("incremental scheduler sparse apply lost its stream state")?
                    .accumulator
                    .finalize_segment(&streamed_segment)?
            };
            pending
                .incremental_stream
                .as_mut()
                .context("incremental scheduler sparse apply lost its stream state")?
                .sparse_b_ms += elapsed_ms_optional(sparse_b_start);
            self.record_real_sparse_shared_mlp_delta_accounting(
                pending.layer_id,
                segment.row_count,
                segment.shared_delta.backend,
            )?;
            self.record_real_sparse_routed_mlp_delta_accounting(
                pending.layer_id,
                segment.row_count,
                dispatch_transport.sparse_delta_backend(),
            )?;
            let apply_start = pending.stage_timing_enabled.then(Instant::now);
            self.apply_device_residual_output_bytes(
                segment.byte_start,
                segment.byte_end,
                ResidualDeltaStage::Mlp,
                output,
            )?;
            dump_layer_boundary_device_hidden(
                pending.layer_id,
                segment.kind,
                segment.token_start,
                segment.row_count,
                "post_mlp",
                &self.device_hidden_segments,
                key,
            )?;
            pending
                .incremental_stream
                .as_mut()
                .context("incremental scheduler sparse apply lost its stream state")?
                .apply_ms += elapsed_ms_optional(apply_start);
            self.record_selected_rows(segment.kind, segment.row_count);
            pending
                .incremental_stream
                .as_mut()
                .context("incremental scheduler sparse apply lost its stream state")?
                .finalized_segments[*segment_index] = true;
        }

        Ok(Some(SchedulerSparseTcpApplyProgress {
            completed_segment_indices: ready_segment_indices,
            dispatch_complete: false,
        }))
    }

    pub(super) fn finish_apply_selected_sparse_tcp_batched(
        &mut self,
        mut pending: SchedulerSparseTcpPendingApply,
    ) -> Result<()> {
        if pending.supports_incremental_stream() {
            loop {
                let progress = self
                    .progress_apply_selected_sparse_tcp_batched(&mut pending)
                    .context("progressing incremental scheduler sparse apply")?;
                if progress.dispatch_complete {
                    break;
                }
            }
            return Ok(());
        }
        let SchedulerSparseTcpPendingApply {
            layer_id,
            batch,
            mut ready_segments,
            pending_dispatch,
            pending_routes,
            pending_hidden_payload,
            completed_dispatch,
            route_count,
            router_weight_bytes,
            router_bias_bytes,
            stage_timing_enabled,
            stage_total_start,
            dispatch_start,
            attention_delta_ms,
            norm_ms,
            shared_mlp_ms,
            normalized_readback_ms,
            router_ms,
            routes_ms,
            incremental_stream,
            incremental_complete,
        } = pending;
        anyhow::ensure!(
            incremental_stream.is_none() && !incremental_complete,
            "non-streamed scheduler sparse apply carried incremental state"
        );
        let mut sparse_b_ms = 0.0_f64;
        let mut apply_ms = 0.0_f64;
        let mut streamed_sparse_b_outputs = None;
        let dispatch = if let Some(dispatch) = completed_dispatch {
            anyhow::ensure!(
                pending_dispatch.is_none(),
                "completed scheduler sparse apply also carried a pending dispatch"
            );
            dispatch
        } else {
            let tcp_context = self.sparse_tcp_routed_mlp.as_mut().context(
                "scheduler sparse TCP routed MLP context missing during payload dispatch",
            )?;
            if let Some(pending_dispatch) = pending_dispatch {
                let response_dtype = real_full_moe_response_dtype_for_batch(&batch)?;
                let can_collect_low_precision_sparse_b = pending_dispatch.has_response_chunks()
                    && response_dtype != ExpertV2Dtype::Bf16
                    && ready_segments.len() == 1;
                let can_collect_direct_owner_sparse_b = batch.num_rows() > 1
                    && pending_dispatch.direct_owner_pending.is_some()
                    && response_dtype != ExpertV2Dtype::Bf16
                    && !ready_segments.is_empty();
                let can_stream_sparse_b = pending_dispatch.chunk_rx.is_some()
                    && batch.num_rows() > 1
                    && !ready_segments.is_empty();
                if can_collect_low_precision_sparse_b {
                    let segment = &ready_segments[0];
                    let key = DeviceHiddenSegmentKey {
                        byte_start: segment.byte_start,
                        byte_end: segment.byte_end,
                    };
                    let residual = self.device_hidden_segments.get(&key).context(
                        "scheduler resident hidden segment missing before collected low-precision sparse TCP residual",
                    )?;
                    let collected_segment = StreamedSparseBResidualSegment {
                        residual,
                        shared_delta: &segment.shared_delta,
                        row_start: segment.batch_row_start,
                        row_count: segment.row_count,
                    };
                    let (dispatch, output) = tcp_context
                        .finish_dispatch_routed_delta_payload_collected_low_precision_device_output(
                            layer_id,
                            pending_dispatch,
                            &collected_segment,
                            batch.num_rows(),
                            NUMERIC_PROGRESS_HIDDEN_DIM,
                            response_dtype,
                        )
                        .with_context(|| {
                            format!(
                                "finishing collected low-precision scheduler sparse routed ProtocolV2 verbs-host batch for layer {layer_id}"
                            )
                        })?;
                    streamed_sparse_b_outputs = Some(vec![output].into_iter());
                    dispatch
                } else if can_collect_direct_owner_sparse_b {
                    let stream_segments = ready_segments
                        .iter()
                        .map(|segment| {
                            let key = DeviceHiddenSegmentKey {
                                byte_start: segment.byte_start,
                                byte_end: segment.byte_end,
                            };
                            let residual = self.device_hidden_segments.get(&key).context(
                                "scheduler resident hidden segment missing before direct-owner sparse TCP residual",
                            )?;
                            Ok(StreamedSparseBResidualSegment {
                                residual,
                                shared_delta: &segment.shared_delta,
                                row_start: segment.batch_row_start,
                                row_count: segment.row_count,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (dispatch, outputs) = tcp_context
                        .finish_direct_owner_low_precision_device_outputs(
                            layer_id,
                            pending_dispatch,
                            &stream_segments,
                            batch.num_rows(),
                            NUMERIC_PROGRESS_HIDDEN_DIM,
                            response_dtype,
                        )
                        .with_context(|| {
                            format!(
                                "finishing multi-segment direct-owner sparse routed batch for layer {layer_id}"
                            )
                        })?;
                    streamed_sparse_b_outputs = Some(outputs.into_iter());
                    dispatch
                } else if can_stream_sparse_b {
                    let stream_segments = ready_segments
                        .iter()
                        .map(|segment| {
                            let key = DeviceHiddenSegmentKey {
                                byte_start: segment.byte_start,
                                byte_end: segment.byte_end,
                            };
                            let residual = self.device_hidden_segments.get(&key).context(
                                "scheduler resident hidden segment missing before streamed sparse TCP residual",
                            )?;
                            Ok(StreamedSparseBResidualSegment {
                                residual,
                                shared_delta: &segment.shared_delta,
                                row_start: segment.batch_row_start,
                                row_count: segment.row_count,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let (dispatch, outputs) = tcp_context
                        .finish_dispatch_routed_delta_payload_streamed_device_output(
                            layer_id,
                            pending_dispatch,
                            &stream_segments,
                            batch.num_rows(),
                            NUMERIC_PROGRESS_HIDDEN_DIM,
                        )
                        .with_context(|| {
                            format!(
                                "finishing streamed scheduler sparse routed ProtocolV2 verbs-host BF16 payload batch for layer {layer_id}"
                            )
                        })?;
                    streamed_sparse_b_outputs = Some(outputs.into_iter());
                    dispatch
                } else {
                    tcp_context
                        .finish_dispatch_routed_delta_payload(pending_dispatch)
                        .with_context(|| {
                            format!(
                                "finishing scheduler sparse routed ProtocolV2 TCP BF16 payload batch for layer {layer_id}"
                            )
                        })?
                }
            } else {
                let routes = pending_routes
                    .as_ref()
                    .context("scheduler sparse TCP routes missing for sliced dispatch")?;
                let global_hidden_payload = pending_hidden_payload
                    .as_ref()
                    .context("scheduler sparse TCP hidden payload missing for sliced dispatch")?;
                tcp_context
                    .dispatch_routed_delta_payload(
                        &batch,
                        routes.as_slice(),
                        global_hidden_payload.as_slice(),
                    )
                    .with_context(|| {
                        format!(
                            "dispatching scheduler sparse routed ProtocolV2 TCP BF16 payload batch for layer {layer_id}"
                        )
                    })?
            }
        };
        let dispatch_ms = elapsed_ms_optional(dispatch_start);
        let dispatch_transport = self
            .sparse_tcp_routed_mlp
            .as_ref()
            .map(|context| context.transport)
            .unwrap_or(RealFullSchedulerSparseDispatchTransport::Tcp);

        record_backend(
            &mut self.device_real_sparse_routed_mlp_route_backend,
            dispatch_transport.sparse_route_backend(),
            "device-real-sparse-routed-nvfp4-route",
        )?;
        self.device_real_sparse_routed_mlp_routes += route_count;
        self.device_real_sparse_routed_mlp_router_weight_bytes += router_weight_bytes;
        self.device_real_sparse_routed_mlp_router_bias_bytes += router_bias_bytes;
        let router_stats = self.device_real_sparse_routed_mlp_router_cache.stats();
        self.device_real_sparse_routed_mlp_router_cache_entries = router_stats.entries;
        self.device_real_sparse_routed_mlp_router_cache_hits = router_stats.cache_hits;

        // Below the Spark-RDMA reduction threshold, a one-row decode returns
        // four raw BF16 rank partials.  Hand the exact ordered FP32 reduction
        // and residual closure to the next native stream through an event,
        // instead of synchronizing the coordinator host at every sparse
        // layer.  The graph copies response bytes into its own pinned staging;
        // only the two device inputs need to remain owned until completion.
        if self.event_owned_raw_tp4_reduction
            && streamed_sparse_b_outputs.is_none()
            && real_full_moe_response_dtype_for_batch(&batch)? == ExpertV2Dtype::Bf16
            && spark_expert_reduction_dispatch_for_rows(batch.num_rows())?.is_none()
            && ready_segments.len() == 1
        {
            let segment = ready_segments
                .pop()
                .expect("single-segment asynchronous raw TP4 path has one segment");
            anyhow::ensure!(
                segment.batch_row_start == 0 && segment.row_count == batch.num_rows(),
                "single-segment asynchronous raw TP4 path does not cover the complete batch"
            );
            let key = DeviceHiddenSegmentKey {
                byte_start: segment.byte_start,
                byte_end: segment.byte_end,
            };
            let residual = self.device_hidden_segments.remove(&key).context(
                "scheduler resident hidden segment missing before asynchronous raw TP4 reduction",
            )?;
            log_device_bf16_hash(layer_id, "residual", 0, &residual)?;
            log_device_bf16_hash(layer_id, "shared_delta", 0, &segment.shared_delta)?;
            let shared_delta_backend = segment.shared_delta.backend;
            let sparse_b_start = stage_timing_enabled.then(Instant::now);
            let output = sparse_b_scatter_shared_residual_add_bf16_async_owned_device_output(
                residual,
                segment.shared_delta,
                dispatch.partial_outputs_bf16_by_host,
                dispatch.global_row_indices_by_host,
                segment.row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
            )
            .context("applying asynchronous fused scheduler raw TP4 residual update")?;
            log_device_bf16_hash(layer_id, "fused_output", 0, &output)?;
            sparse_b_ms += elapsed_ms_optional(sparse_b_start);
            self.record_real_sparse_shared_mlp_delta_accounting(
                layer_id,
                segment.row_count,
                shared_delta_backend,
            )?;
            self.record_real_sparse_routed_mlp_delta_accounting(
                layer_id,
                segment.row_count,
                dispatch_transport.sparse_delta_backend(),
            )?;
            let apply_start = stage_timing_enabled.then(Instant::now);
            self.apply_device_residual_output_bytes(
                segment.byte_start,
                segment.byte_end,
                ResidualDeltaStage::Mlp,
                output,
            )?;
            dump_layer_boundary_device_hidden(
                layer_id,
                segment.kind,
                segment.token_start,
                segment.row_count,
                "post_mlp",
                &self.device_hidden_segments,
                key,
            )?;
            apply_ms += elapsed_ms_optional(apply_start);
            self.record_selected_rows(segment.kind, segment.row_count);
            if stage_timing_enabled {
                eprintln!(
                    "real_full_sparse_{}_stage_timing layer_id={} rows={} routes={} attention_delta_ms={:.3} norm_ms={:.3} shared_mlp_ms={:.3} normalized_readback_ms={:.3} router_ms={:.3} routes_ms={:.3} dispatch_ms={:.3} sparse_b_ms={:.3} apply_ms={:.3} total_ms={:.3}",
                    dispatch_transport.label(),
                    layer_id,
                    batch.num_rows(),
                    batch.route_count(),
                    attention_delta_ms,
                    norm_ms,
                    shared_mlp_ms,
                    normalized_readback_ms,
                    router_ms,
                    routes_ms,
                    dispatch_ms,
                    sparse_b_ms,
                    apply_ms,
                    elapsed_ms_optional(stage_total_start)
                );
            }
            return Ok(());
        }

        let mut output_row_offset = 0_usize;
        for (segment_index, segment) in ready_segments.into_iter().enumerate() {
            anyhow::ensure!(
                segment.batch_row_start == output_row_offset,
                "scheduler sparse TCP segment row start {} did not match expected {output_row_offset}",
                segment.batch_row_start
            );
            self.record_real_sparse_shared_mlp_delta_accounting(
                layer_id,
                segment.row_count,
                segment.shared_delta.backend,
            )?;
            output_row_offset += segment.row_count;
            let key = DeviceHiddenSegmentKey {
                byte_start: segment.byte_start,
                byte_end: segment.byte_end,
            };
            let sparse_b_start = stage_timing_enabled.then(Instant::now);
            let output = {
                let residual = self.device_hidden_segments.get(&key).context(
                    "scheduler resident hidden segment missing before fused sparse TCP residual",
                )?;
                if streamed_sparse_b_outputs.is_none() {
                    log_device_bf16_hash(layer_id, "residual", segment_index, residual)?;
                    log_device_bf16_hash(
                        layer_id,
                        "shared_delta",
                        segment_index,
                        &segment.shared_delta,
                    )?;
                }
                if let Some(outputs) = streamed_sparse_b_outputs.as_mut() {
                    outputs
                        .next()
                        .context("streamed scheduler Sparse-B segment output is missing")?
                } else if segment.batch_row_start == 0 && segment.row_count == batch.num_rows() {
                    sparse_b_scatter_shared_residual_add_bf16_device_output(
                        residual,
                        &segment.shared_delta,
                        &dispatch.partial_outputs_bf16_by_host,
                        &dispatch.global_row_indices_by_host,
                        segment.row_count,
                        NUMERIC_PROGRESS_HIDDEN_DIM,
                    )
                    .context("applying fused scheduler Sparse-B TCP payload residual update")?
                } else {
                    let (segment_payloads, segment_row_indices) =
                        scheduler_sparse_tcp_payload_partials_for_segment(
                            &dispatch,
                            segment.batch_row_start,
                            segment.row_count,
                            NUMERIC_PROGRESS_HIDDEN_DIM,
                        )?;
                    sparse_b_scatter_shared_residual_add_bf16_device_output(
                        residual,
                        &segment.shared_delta,
                        &segment_payloads,
                        &segment_row_indices,
                        segment.row_count,
                        NUMERIC_PROGRESS_HIDDEN_DIM,
                    )
                    .context("applying fused scheduler Sparse-B TCP payload residual update")?
                }
            };
            if streamed_sparse_b_outputs.is_none() {
                log_device_bf16_hash(layer_id, "fused_output", segment_index, &output)?;
            }
            sparse_b_ms += elapsed_ms_optional(sparse_b_start);
            self.record_real_sparse_routed_mlp_delta_accounting(
                layer_id,
                segment.row_count,
                dispatch_transport.sparse_delta_backend(),
            )?;
            let apply_start = stage_timing_enabled.then(Instant::now);
            self.apply_device_residual_output_bytes(
                segment.byte_start,
                segment.byte_end,
                ResidualDeltaStage::Mlp,
                output,
            )?;
            dump_layer_boundary_device_hidden(
                layer_id,
                segment.kind,
                segment.token_start,
                segment.row_count,
                "post_mlp",
                &self.device_hidden_segments,
                key,
            )?;
            apply_ms += elapsed_ms_optional(apply_start);
            self.record_selected_rows(segment.kind, segment.row_count);
        }
        if let Some(mut outputs) = streamed_sparse_b_outputs {
            anyhow::ensure!(
                outputs.next().is_none(),
                "streamed scheduler Sparse-B produced extra segment outputs"
            );
        }
        anyhow::ensure!(
            output_row_offset == batch.num_rows(),
            "scheduler sparse TCP output row split consumed {output_row_offset} rows, expected {}",
            batch.num_rows()
        );
        if stage_timing_enabled {
            eprintln!(
                "real_full_sparse_{}_stage_timing layer_id={} rows={} routes={} attention_delta_ms={:.3} norm_ms={:.3} shared_mlp_ms={:.3} normalized_readback_ms={:.3} router_ms={:.3} routes_ms={:.3} dispatch_ms={:.3} sparse_b_ms={:.3} apply_ms={:.3} total_ms={:.3}",
                dispatch_transport.label(),
                layer_id,
                batch.num_rows(),
                batch.route_count(),
                attention_delta_ms,
                norm_ms,
                shared_mlp_ms,
                normalized_readback_ms,
                router_ms,
                routes_ms,
                dispatch_ms,
                sparse_b_ms,
                apply_ms,
                elapsed_ms_optional(stage_total_start)
            );
        }
        Ok(())
    }

    fn record_selected_rows(&mut self, kind: RowSourceKind, row_count: usize) {
        match kind {
            RowSourceKind::PrefillChunk => self.selected_prefill_rows += row_count,
            RowSourceKind::DecodeStep => self.selected_decode_rows += row_count,
            RowSourceKind::MtpVerifyBlock => self.selected_mtp_rows += row_count,
            RowSourceKind::Benchmark => {}
        }
    }

    pub(super) fn rechunk_prefill_device_hidden(&mut self, chunk_rows: usize) -> Result<()> {
        anyhow::ensure!(
            chunk_rows > 0,
            "scheduler hidden rechunk requires nonzero rows"
        );
        anyhow::ensure!(
            coordinator_cuda_reference_kernels_enabled(),
            "scheduler hidden rechunk requires coordinator CUDA kernels"
        );
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler hidden rechunk row bytes overflow usize")?;
        let prefill_bytes = self
            .shape
            .prefill_rows
            .checked_mul(row_bytes)
            .context("scheduler hidden rechunk prefill bytes overflow usize")?;
        let desired_keys = (0..self.shape.prefill_rows)
            .step_by(chunk_rows)
            .map(|row_start| {
                let row_count = (self.shape.prefill_rows - row_start).min(chunk_rows);
                let byte_start = row_start * row_bytes;
                DeviceHiddenSegmentKey {
                    byte_start,
                    byte_end: byte_start + row_count * row_bytes,
                }
            })
            .collect::<Vec<_>>();
        let mut new_segments = Vec::new();
        for desired in &desired_keys {
            if self.device_hidden_segments.contains_key(desired) {
                continue;
            }
            let parent_key = smallest_containing_device_hidden_segment_key(
                &self.device_hidden_segments,
                *desired,
            )
            .with_context(|| {
                format!(
                    "scheduler hidden rechunk found no resident parent for bytes {}..{}",
                    desired.byte_start, desired.byte_end
                )
            })?;
            let parent = self
                .device_hidden_segments
                .get(&parent_key)
                .context("scheduler hidden rechunk resident parent disappeared")?;
            let parent_bytes = parent_key.byte_end - parent_key.byte_start;
            anyhow::ensure!(
                parent.rows * row_bytes == parent_bytes
                    && parent.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler hidden rechunk parent shape {}x{} did not match {parent_bytes} bytes",
                parent.rows,
                parent.values_per_row
            );
            let view_bytes = desired.byte_end - desired.byte_start;
            let view = device_buffer_byte_view(
                parent.buffer(),
                desired.byte_start - parent_key.byte_start,
                view_bytes,
                "scheduler hidden rechunk source",
            )?;
            let rows = view_bytes / row_bytes;
            let child = device_bf16_output_from_device_template_buffer(
                view,
                rows,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler sparse frontier hidden segment",
            )?;
            new_segments.push((*desired, child));
        }
        for (key, segment) in new_segments {
            self.device_hidden_segments.insert(key, segment);
        }
        let desired_key_set = desired_keys.iter().copied().collect::<BTreeSet<_>>();
        self.device_hidden_segments.retain(|key, _| {
            key.byte_start >= prefill_bytes
                || key.byte_end > prefill_bytes
                || desired_key_set.contains(key)
        });
        anyhow::ensure!(
            desired_keys
                .iter()
                .all(|key| self.device_hidden_segments.contains_key(key)),
            "scheduler hidden rechunk left a missing prefill segment"
        );
        self.shape.sparse_source_segments_per_layer =
            desired_keys.len() + self.shape.decode_rows + usize::from(self.shape.mtp_rows > 0);
        Ok(())
    }

    pub(super) fn device_hidden_source(
        &mut self,
        kind: RowSourceKind,
        token_start: usize,
        row_count: usize,
    ) -> Result<Option<RealFullSchedulerDeviceHiddenSource>> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(None);
        }
        if row_count == 0 {
            return Ok(None);
        }
        let start_row_index = self.numeric_progression_row_index(kind, token_start, 0)?;
        let end_row_index = start_row_index
            .checked_add(row_count)
            .context("scheduler device hidden source row range overflows usize")?;
        let start_value = start_row_index
            .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
            .context("scheduler device hidden source value start overflows usize")?;
        let end_value = end_row_index
            .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
            .context("scheduler device hidden source value end overflows usize")?;
        let byte_start = start_value
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler device hidden source byte start overflows usize")?;
        let byte_end = end_value
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler device hidden source byte end overflows usize")?;
        if byte_end > self.residual_bf16.len() {
            anyhow::bail!(
                "scheduler device hidden source row range {start_row_index}..{end_row_index} exceeds residual rows {}",
                self.shape.unique_rows()
            );
        }
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        if !self.device_hidden_segments.contains_key(&key) {
            let initial = if let Some(parent_key) =
                smallest_containing_device_hidden_segment_key(&self.device_hidden_segments, key)
            {
                let parent = self
                    .device_hidden_segments
                    .get(&parent_key)
                    .context("scheduler resident hidden parent disappeared")?;
                let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
                    .checked_mul(std::mem::size_of::<u16>())
                    .context("scheduler resident hidden row bytes overflow usize")?;
                let parent_bytes = parent_key.byte_end - parent_key.byte_start;
                anyhow::ensure!(
                    parent.rows * row_bytes == parent_bytes
                        && parent.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
                    "scheduler resident hidden parent shape {}x{} did not match {parent_bytes} bytes",
                    parent.rows,
                    parent.values_per_row
                );
                let view = device_buffer_byte_view(
                    parent.buffer(),
                    key.byte_start - parent_key.byte_start,
                    key.byte_end - key.byte_start,
                    "scheduler resident hidden parent slice",
                )?;
                device_bf16_output_from_device_template_buffer(
                    view,
                    row_count,
                    NUMERIC_PROGRESS_HIDDEN_DIM,
                    "scheduler numeric resident hidden parent slice",
                )?
            } else {
                device_bf16_output_from_bf16_bytes(
                    &self.residual_bf16[byte_start..byte_end],
                    row_count,
                    NUMERIC_PROGRESS_HIDDEN_DIM,
                    "scheduler numeric resident hidden source",
                )?
            };
            self.device_hidden_segments.insert(key, initial);
        }
        let segment = self
            .device_hidden_segments
            .get(&key)
            .context("scheduler resident hidden source missing after initialization")?;
        Ok(Some(RealFullSchedulerDeviceHiddenSource {
            buffer: segment.buffer(),
            rows: segment.rows,
            values_per_row: segment.values_per_row,
        }))
    }

    pub(super) fn fuse_device_hidden_sources(
        &mut self,
        sources: &[&RowSource],
        layer_id: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            sources.len() >= 2,
            "scheduler device hidden fusion requires at least two sources"
        );
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler fused hidden row bytes overflow")?;
        let mut keys = Vec::with_capacity(sources.len());
        let mut expected_start_row = None;
        for source in sources {
            anyhow::ensure!(
                source.row_count > 0,
                "scheduler fused hidden source is empty"
            );
            let start_row =
                self.numeric_progression_row_index(source.kind, source.token_start.0 as usize, 0)?;
            if let Some(expected) = expected_start_row {
                anyhow::ensure!(
                    start_row == expected,
                    "scheduler fused hidden sources are not contiguous: expected row {expected}, got {start_row}"
                );
            }
            let end_row = start_row
                .checked_add(source.row_count)
                .context("scheduler fused hidden row end overflow")?;
            let byte_start = start_row
                .checked_mul(row_bytes)
                .context("scheduler fused hidden byte start overflow")?;
            let byte_end = end_row
                .checked_mul(row_bytes)
                .context("scheduler fused hidden byte end overflow")?;
            keys.push(DeviceHiddenSegmentKey {
                byte_start,
                byte_end,
            });
            expected_start_row = Some(end_row);
        }
        let fused = {
            let batches = keys
                .iter()
                .map(|key| {
                    self.device_hidden_segments.get(key).with_context(|| {
                        format!(
                            "scheduler fused hidden segment {}..{} is missing",
                            key.byte_start, key.byte_end
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            // The owned-buffer pool key also includes the request-local
            // execution-lane bank. Both members of a C=4 pair retain distinct
            // allocations here; their routed expert payloads are merged later
            // into one Spark transport job. Within one lane, parity preserves
            // the prior layer's input until the next layer has enqueued while
            // keeping each layer's graph input address deterministic across
            // recurrent cycles.
            let label = if layer_id % 2 == 0 {
                "scheduler fused decode/MTP hidden rows even layer"
            } else {
                "scheduler fused decode/MTP hidden rows odd layer"
            };
            concat_device_bf16_row_batches(&batches, label)?
        };
        let fused_key = DeviceHiddenSegmentKey {
            byte_start: keys[0].byte_start,
            byte_end: keys
                .last()
                .expect("fused hidden keys are non-empty")
                .byte_end,
        };
        self.device_hidden_segments.insert(fused_key, fused);
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<RealFullSchedulerNumericProgressionFinish> {
        let prefill_values = self.shape.prefill_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let decode_values = self.shape.decode_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let mtp_start = prefill_values + decode_values;
        let prefill_bytes = prefill_values * std::mem::size_of::<u16>();
        let mtp_start_bytes = mtp_start * std::mem::size_of::<u16>();

        let device_hidden_segment_summary = self
            .account_device_hidden_segments()
            .context("accounting scheduler device hidden segments")?;
        let final_visible_checksum = 0.0;
        let rejected_mtp_checksum = 0.0;
        let selected_rows =
            self.selected_prefill_rows + self.selected_decode_rows + self.selected_mtp_rows;
        let expected_selected_rows = self.shape.unique_rows() * GLM52_NUM_HIDDEN_LAYERS;
        let expected_value_updates = expected_selected_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let sparse_layers = GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE;
        let expected_source_segments = GLM52_FIRST_K_DENSE_REPLACE
            * self.shape.source_segments_per_layer
            + sparse_layers * self.shape.sparse_source_segments_per_layer;
        let expected_real_dense_mlp_source_segments =
            GLM52_FIRST_K_DENSE_REPLACE * self.shape.source_segments_per_layer;
        let expected_real_dense_mlp_rows = GLM52_FIRST_K_DENSE_REPLACE * self.shape.unique_rows();
        let expected_real_dense_mlp_values =
            expected_real_dense_mlp_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let expected_real_sparse_shared_mlp_source_segments =
            sparse_layers * self.shape.sparse_source_segments_per_layer;
        let expected_real_sparse_shared_mlp_rows = sparse_layers * self.shape.unique_rows();
        let expected_real_sparse_shared_mlp_values =
            expected_real_sparse_shared_mlp_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let expected_real_sparse_routed_mlp_source_segments =
            expected_real_sparse_shared_mlp_source_segments;
        let expected_real_sparse_routed_mlp_rows = expected_real_sparse_shared_mlp_rows;
        let expected_real_sparse_routed_mlp_values = expected_real_sparse_shared_mlp_values;
        let expected_real_sparse_routed_mlp_routes =
            expected_real_sparse_routed_mlp_rows * GLM52_TOP_K;
        let expected_device_hidden_segment_residual_adds = expected_source_segments * 2;
        let expected_device_hidden_segment_value_updates = expected_value_updates * 2;
        let expected_device_delta_template_uses = expected_source_segments;
        let uses_device_attention_output_delta = self.attention_device_output_delta_rows > 0;
        let uses_full_width_device_attention_delta = uses_device_attention_output_delta
            && self.attention_device_output_delta_rows == expected_selected_rows
            && self.attention_device_output_delta_values == expected_value_updates;
        let device_delta_template_status = match (
            coordinator_cuda_reference_kernels_enabled(),
            uses_full_width_device_attention_delta,
            self.device_delta_template_uses > 0,
        ) {
            (true, true, false) => "cuda-device-delta-template-not-needed",
            (true, _, true) => "cuda-device-delta-template-cache",
            (true, false, false) => "cuda-device-delta-template-cache-missing",
            (false, _, _) => "not-run",
        };
        let device_delta_template_available = if coordinator_cuda_reference_kernels_enabled() {
            if uses_full_width_device_attention_delta {
                self.device_delta_template_uploads == 0
                    && self.device_delta_template_uses == 0
                    && self.device_delta_template_resident_values == 0
            } else {
                self.device_delta_template_uses == expected_device_delta_template_uses
                    && self.device_delta_template_uploads > 0
                    && self.device_delta_template_uploads < self.device_delta_template_uses
                    && self.device_delta_template_resident_values > 0
            }
        } else {
            self.device_delta_template_uploads == 0
                && self.device_delta_template_uses == 0
                && self.device_delta_template_resident_values == 0
        };
        let uses_device_mlp_delta = self.device_mlp_delta_rows > 0;
        let device_mlp_delta_status = match (
            coordinator_cuda_reference_kernels_enabled(),
            uses_device_mlp_delta,
        ) {
            (true, true) => "cuda-device-hidden-dependent-mlp-delta",
            (true, false) => "cuda-device-hidden-dependent-mlp-delta-not-needed",
            (false, _) => "not-run",
        };
        let device_mlp_delta_available = if coordinator_cuda_reference_kernels_enabled() {
            !uses_device_mlp_delta
                && self.device_mlp_delta_rows == 0
                && self.device_mlp_delta_values == 0
                && self.device_mlp_delta_checksum == 0.0
                && self.device_mlp_delta_backend.is_none()
                && self.device_mlp_weight_uploads == 0
                && self.device_mlp_weight_resident_values == 0
        } else {
            !uses_device_mlp_delta
                && self.device_mlp_delta_rows == 0
                && self.device_mlp_delta_values == 0
                && self.device_mlp_delta_checksum == 0.0
                && self.device_mlp_delta_backend.is_none()
                && self.device_mlp_weight_uploads == 0
                && self.device_mlp_weight_resident_values == 0
        };
        let uses_device_real_dense_mlp_delta = self.device_real_dense_mlp_delta_rows > 0;
        let device_real_dense_mlp_delta_status = match (
            coordinator_cuda_reference_kernels_enabled(),
            uses_device_real_dense_mlp_delta,
        ) {
            (true, true) => SCHEDULER_REAL_DENSE_MLP_DELTA_STATUS,
            (true, false) => "cuda-real-dense-checkpoint-mlp-delta-missing",
            (false, _) => "not-run",
        };
        let device_real_dense_mlp_layers = self.device_real_dense_mlp_layers.len();
        let device_real_dense_mlp_delta_available = if coordinator_cuda_reference_kernels_enabled()
        {
            uses_device_real_dense_mlp_delta
                    && self.device_real_dense_mlp_delta_rows == expected_real_dense_mlp_rows
                    && self.device_real_dense_mlp_delta_values == expected_real_dense_mlp_values
                    && self.device_real_dense_mlp_delta_checksum.is_finite()
                    && self.device_real_dense_mlp_delta_backend
                        == Some(
                            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
                        )
                    && self.device_real_dense_mlp_norm_backend.is_some()
                    && self.device_real_dense_mlp_weight_tensors
                        == GLM52_FIRST_K_DENSE_REPLACE * 4
                    && self.device_real_dense_mlp_weight_bytes > 0
                    && self.device_real_dense_mlp_source_segments
                        == expected_real_dense_mlp_source_segments
                    && device_real_dense_mlp_layers == GLM52_FIRST_K_DENSE_REPLACE
        } else {
            !uses_device_real_dense_mlp_delta
                && self.device_real_dense_mlp_delta_rows == 0
                && self.device_real_dense_mlp_delta_values == 0
                && self.device_real_dense_mlp_delta_checksum == 0.0
                && self.device_real_dense_mlp_delta_backend.is_none()
                && self.device_real_dense_mlp_norm_backend.is_none()
                && self.device_real_dense_mlp_weight_tensors == 0
                && self.device_real_dense_mlp_weight_bytes == 0
                && self.device_real_dense_mlp_source_segments == 0
                && device_real_dense_mlp_layers == 0
        };
        let uses_device_real_sparse_shared_mlp_delta =
            self.device_real_sparse_shared_mlp_delta_rows > 0;
        let device_real_sparse_shared_mlp_delta_status = match (
            coordinator_cuda_reference_kernels_enabled(),
            uses_device_real_sparse_shared_mlp_delta,
        ) {
            (true, true) => "cuda-real-sparse-shared-checkpoint-mlp-delta",
            (true, false) => "cuda-real-sparse-shared-checkpoint-mlp-delta-missing",
            (false, _) => "not-run",
        };
        let device_real_sparse_shared_mlp_layers = self.device_real_sparse_shared_mlp_layers.len();
        let device_real_sparse_shared_mlp_delta_available =
            if coordinator_cuda_reference_kernels_enabled() {
                uses_device_real_sparse_shared_mlp_delta
                    && self.device_real_sparse_shared_mlp_delta_rows
                        == expected_real_sparse_shared_mlp_rows
                    && self.device_real_sparse_shared_mlp_delta_values
                        == expected_real_sparse_shared_mlp_values
                    && self.device_real_sparse_shared_mlp_delta_checksum.is_finite()
                    && self.device_real_sparse_shared_mlp_delta_backend
                        == Some(
                            CUDA_REFERENCE_SILU_GATED_MLP_BF16_PRELOADED_GATE_UP_DOWN_RESIDENT_WEIGHT_BACKEND,
                        )
                    && self.device_real_sparse_shared_mlp_norm_backend.is_some()
                    && self.device_real_sparse_shared_mlp_weight_tensors == sparse_layers * 4
                    && self.device_real_sparse_shared_mlp_weight_bytes > 0
                    && self.device_real_sparse_shared_mlp_source_segments
                        == expected_real_sparse_shared_mlp_source_segments
                    && device_real_sparse_shared_mlp_layers == sparse_layers
            } else {
                !uses_device_real_sparse_shared_mlp_delta
                    && self.device_real_sparse_shared_mlp_delta_rows == 0
                    && self.device_real_sparse_shared_mlp_delta_values == 0
                    && self.device_real_sparse_shared_mlp_delta_checksum == 0.0
                    && self.device_real_sparse_shared_mlp_delta_backend.is_none()
                    && self.device_real_sparse_shared_mlp_norm_backend.is_none()
                    && self.device_real_sparse_shared_mlp_weight_tensors == 0
                    && self.device_real_sparse_shared_mlp_weight_bytes == 0
                    && self.device_real_sparse_shared_mlp_source_segments == 0
                    && device_real_sparse_shared_mlp_layers == 0
            };
        let uses_device_real_sparse_routed_mlp_delta =
            self.device_real_sparse_routed_mlp_delta_rows > 0;
        let device_real_sparse_routed_mlp_delta_status = match (
            coordinator_cuda_reference_kernels_enabled(),
            uses_device_real_sparse_routed_mlp_delta,
        ) {
            (true, true) => SCHEDULER_REAL_SPARSE_ROUTED_MLP_DELTA_STATUS,
            (true, false) => "cuda-real-sparse-routed-nvfp4-checkpoint-mlp-delta-missing",
            (false, _) => "not-run",
        };
        let device_real_sparse_routed_mlp_layers = self.device_real_sparse_routed_mlp_layers.len();
        let device_real_sparse_routed_mlp_delta_available =
            if coordinator_cuda_reference_kernels_enabled() {
                uses_device_real_sparse_routed_mlp_delta
                    && self.device_real_sparse_routed_mlp_delta_rows
                        == expected_real_sparse_routed_mlp_rows
                    && self.device_real_sparse_routed_mlp_delta_values
                        == expected_real_sparse_routed_mlp_values
                    && self
                        .device_real_sparse_routed_mlp_delta_checksum
                        .is_finite()
                    && routed_delta_backend_available(
                        self.device_real_sparse_routed_mlp_delta_backend,
                    )
                    && routed_route_backend_available(
                        self.device_real_sparse_routed_mlp_route_backend,
                    )
                    && self
                        .device_real_sparse_routed_mlp_router_backend
                        .map(|backend| {
                            backend.contains("router-topk-bf16") && backend.contains("device-input")
                        })
                        .unwrap_or(false)
                    && self.device_real_sparse_routed_mlp_routes
                        == expected_real_sparse_routed_mlp_routes
                    && self.device_real_sparse_routed_mlp_router_weight_bytes > 0
                    && self.device_real_sparse_routed_mlp_router_bias_bytes > 0
                    && routed_route_cache_available(
                        self.device_real_sparse_routed_mlp_route_backend,
                        self.device_real_sparse_routed_mlp_route_cache_cuda_entries,
                        self.device_real_sparse_routed_mlp_route_cache_cuda_uploads,
                    )
                    && self.device_real_sparse_routed_mlp_router_cache_entries == sparse_layers
                    && self.device_real_sparse_routed_mlp_source_segments
                        == expected_real_sparse_routed_mlp_source_segments
                    && device_real_sparse_routed_mlp_layers == sparse_layers
            } else {
                !uses_device_real_sparse_routed_mlp_delta
                    && self.device_real_sparse_routed_mlp_delta_rows == 0
                    && self.device_real_sparse_routed_mlp_delta_values == 0
                    && self.device_real_sparse_routed_mlp_delta_checksum == 0.0
                    && self.device_real_sparse_routed_mlp_delta_backend.is_none()
                    && self.device_real_sparse_routed_mlp_route_backend.is_none()
                    && self.device_real_sparse_routed_mlp_router_backend.is_none()
                    && self.device_real_sparse_routed_mlp_routes == 0
                    && self.device_real_sparse_routed_mlp_router_weight_bytes == 0
                    && self.device_real_sparse_routed_mlp_router_bias_bytes == 0
                    && self.device_real_sparse_routed_mlp_route_cache_cuda_entries == 0
                    && self.device_real_sparse_routed_mlp_route_cache_cuda_uploads == 0
                    && self.device_real_sparse_routed_mlp_route_cache_cuda_hits == 0
                    && self.device_real_sparse_routed_mlp_router_cache_entries == 0
                    && self.device_real_sparse_routed_mlp_router_cache_hits == 0
                    && self.device_real_sparse_routed_mlp_source_segments == 0
                    && device_real_sparse_routed_mlp_layers == 0
            };
        let expected_visible_checksum = 0.0;
        let expected_rejected_mtp_checksum = 0.0;
        let device_attention_output_delta_status = match (
            uses_device_attention_output_delta,
            uses_full_width_device_attention_delta,
        ) {
            (true, true) => "cuda-device-attention-hidden-delta",
            (true, false) => "cuda-device-attention-output-prefix-delta",
            (false, _) => "not-run",
        };
        let uses_device_hidden_segment_residual_add = self.device_hidden_segment_residual_adds > 0;
        let device_hidden_segment_status = match (
            coordinator_cuda_reference_kernels_enabled(),
            uses_device_hidden_segment_residual_add,
        ) {
            (true, true) => "cuda-device-hidden-segment-residual-add",
            (true, false) => "cuda-device-hidden-segment-residual-add-missing",
            (false, _) => "not-run",
        };
        let expected_device_hidden_segment_resident_segments =
            self.shape.sparse_source_segments_per_layer;
        let expected_device_hidden_segment_resident_values =
            self.shape.unique_rows() * NUMERIC_PROGRESS_HIDDEN_DIM;
        let device_hidden_segment_progression_passed =
            if coordinator_cuda_reference_kernels_enabled() {
                uses_device_hidden_segment_residual_add
                    && self.device_hidden_segment_residual_adds
                        == expected_device_hidden_segment_residual_adds
                    && self.device_hidden_segment_value_updates
                        == expected_device_hidden_segment_value_updates
                    && self.device_hidden_segment_residual_add_backend.is_some()
                    && device_hidden_segment_summary.resident_segments
                        == expected_device_hidden_segment_resident_segments
                    && device_hidden_segment_summary.resident_values
                        == expected_device_hidden_segment_resident_values
            } else {
                self.device_hidden_segment_residual_adds == 0
                    && self.device_hidden_segment_value_updates == 0
                    && self.device_hidden_segment_residual_add_backend.is_none()
                    && device_hidden_segment_summary.resident_segments == 0
                    && device_hidden_segment_summary.resident_values == 0
            };
        let passed = self.selected_prefill_rows
            == self.shape.prefill_rows * GLM52_NUM_HIDDEN_LAYERS
            && self.selected_decode_rows == self.shape.decode_rows * GLM52_NUM_HIDDEN_LAYERS
            && self.selected_mtp_rows == self.shape.mtp_rows * GLM52_NUM_HIDDEN_LAYERS
            && selected_rows == expected_selected_rows
            && self.source_segments == expected_source_segments
            && self.attention_residual_adds == expected_source_segments
            && self.mlp_residual_adds == expected_source_segments
            && self.attention_residual_add_backend.is_some()
            && self.mlp_residual_add_backend.is_some()
            && self.attention_value_updates == expected_value_updates
            && self.mlp_value_updates == expected_value_updates
            && device_delta_template_available
            && device_mlp_delta_available
            && device_real_dense_mlp_delta_available
            && device_real_sparse_shared_mlp_delta_available
            && device_real_sparse_routed_mlp_delta_available
            && if uses_device_attention_output_delta {
                self.attention_device_output_delta_rows == expected_selected_rows
                    && self.attention_device_output_delta_values > 0
                    && self.attention_device_output_delta_checksum.is_finite()
                    && self.attention_device_output_delta_backend.is_some()
                    && if coordinator_cuda_reference_kernels_enabled()
                        && !uses_full_width_device_attention_delta
                    {
                        self.attention_device_output_delta_device_prefix_rows
                            == expected_selected_rows
                            && self.attention_device_output_delta_device_prefix_values
                                == self.attention_device_output_delta_values
                            && self
                                .attention_device_output_delta_device_prefix_backend
                                .is_some()
                    } else {
                        self.attention_device_output_delta_device_prefix_rows == 0
                            && self.attention_device_output_delta_device_prefix_values == 0
                            && self
                                .attention_device_output_delta_device_prefix_backend
                                .is_none()
                    }
            } else {
                self.attention_device_output_delta_rows == 0
                    && self.attention_device_output_delta_values == 0
                    && self.attention_device_output_delta_checksum == 0.0
                    && self.attention_device_output_delta_backend.is_none()
                    && self.attention_device_output_delta_device_prefix_rows == 0
                    && self.attention_device_output_delta_device_prefix_values == 0
                    && self
                        .attention_device_output_delta_device_prefix_backend
                        .is_none()
            }
            && device_hidden_segment_progression_passed;

        let self_test = RealFullSchedulerNumericProgressionSelfTest {
            status: "numeric-scheduler-progression-self-test",
            scope: "apply admitted scheduler attention residual deltas plus CUDA real dense checkpoint MLP deltas for all dense-layer scheduler rows, CUDA real shared-expert checkpoint MLP deltas plus real NVFP4 routed checkpoint deltas for all sparse scheduler rows while keeping rejected MTP rows out of the visible checksum",
            layers: GLM52_NUM_HIDDEN_LAYERS,
            source_modes: ["prefill", "decode", "mtp_verify"],
            unique_source_rows: self.shape.unique_rows(),
            hidden_dim: NUMERIC_PROGRESS_HIDDEN_DIM,
            residual_dtype: NUMERIC_PROGRESS_RESIDUAL_DTYPE,
            selected_prefill_rows: self.selected_prefill_rows,
            selected_decode_rows: self.selected_decode_rows,
            selected_mtp_rows: self.selected_mtp_rows,
            mtp_accepted_rows_per_layer: self.shape.mtp_accepted_rows,
            mtp_rejected_rows_per_layer: self.shape.mtp_rows - self.shape.mtp_accepted_rows,
            source_segments: self.source_segments,
            attention_residual_adds: self.attention_residual_adds,
            mlp_residual_adds: self.mlp_residual_adds,
            attention_residual_add_backend: self
                .attention_residual_add_backend
                .unwrap_or("not-run"),
            mlp_residual_add_backend: self.mlp_residual_add_backend.unwrap_or("not-run"),
            attention_value_updates: self.attention_value_updates,
            mlp_value_updates: self.mlp_value_updates,
            device_attention_output_delta_status,
            attention_device_output_delta_rows: self.attention_device_output_delta_rows,
            attention_device_output_delta_values: self.attention_device_output_delta_values,
            attention_device_output_delta_checksum: self.attention_device_output_delta_checksum,
            attention_device_output_delta_backend: self
                .attention_device_output_delta_backend
                .unwrap_or("not-run"),
            attention_device_output_delta_device_prefix_rows: self
                .attention_device_output_delta_device_prefix_rows,
            attention_device_output_delta_device_prefix_values: self
                .attention_device_output_delta_device_prefix_values,
            attention_device_output_delta_device_prefix_backend: self
                .attention_device_output_delta_device_prefix_backend
                .unwrap_or("not-run"),
            uses_device_attention_output_delta,
            device_delta_template_status,
            device_delta_template_uploads: self.device_delta_template_uploads,
            device_delta_template_uses: self.device_delta_template_uses,
            device_delta_template_resident_values: self.device_delta_template_resident_values,
            device_mlp_delta_status,
            device_mlp_delta_rows: self.device_mlp_delta_rows,
            device_mlp_delta_values: self.device_mlp_delta_values,
            device_mlp_delta_checksum: self.device_mlp_delta_checksum,
            device_mlp_delta_backend: self.device_mlp_delta_backend.unwrap_or("not-run"),
            device_mlp_weight_uploads: self.device_mlp_weight_uploads,
            device_mlp_weight_resident_values: self.device_mlp_weight_resident_values,
            uses_device_mlp_delta,
            device_real_dense_mlp_delta_status,
            device_real_dense_mlp_delta_rows: self.device_real_dense_mlp_delta_rows,
            device_real_dense_mlp_delta_values: self.device_real_dense_mlp_delta_values,
            device_real_dense_mlp_delta_checksum: self.device_real_dense_mlp_delta_checksum,
            device_real_dense_mlp_delta_backend: self
                .device_real_dense_mlp_delta_backend
                .unwrap_or("not-run"),
            device_real_dense_mlp_norm_backend: self
                .device_real_dense_mlp_norm_backend
                .unwrap_or("not-run"),
            device_real_dense_mlp_weight_tensors: self.device_real_dense_mlp_weight_tensors,
            device_real_dense_mlp_weight_bytes: self.device_real_dense_mlp_weight_bytes,
            device_real_dense_mlp_layers,
            device_real_dense_mlp_source_segments: self.device_real_dense_mlp_source_segments,
            uses_device_real_dense_mlp_delta,
            device_real_sparse_shared_mlp_delta_status,
            device_real_sparse_shared_mlp_delta_rows: self.device_real_sparse_shared_mlp_delta_rows,
            device_real_sparse_shared_mlp_delta_values: self
                .device_real_sparse_shared_mlp_delta_values,
            device_real_sparse_shared_mlp_delta_checksum: self
                .device_real_sparse_shared_mlp_delta_checksum,
            device_real_sparse_shared_mlp_delta_backend: self
                .device_real_sparse_shared_mlp_delta_backend
                .unwrap_or("not-run"),
            device_real_sparse_shared_mlp_norm_backend: self
                .device_real_sparse_shared_mlp_norm_backend
                .unwrap_or("not-run"),
            device_real_sparse_shared_mlp_weight_tensors: self
                .device_real_sparse_shared_mlp_weight_tensors,
            device_real_sparse_shared_mlp_weight_bytes: self
                .device_real_sparse_shared_mlp_weight_bytes,
            device_real_sparse_shared_mlp_layers,
            device_real_sparse_shared_mlp_source_segments: self
                .device_real_sparse_shared_mlp_source_segments,
            uses_device_real_sparse_shared_mlp_delta,
            device_real_sparse_routed_mlp_delta_status,
            device_real_sparse_routed_mlp_delta_rows: self
                .device_real_sparse_routed_mlp_delta_rows,
            device_real_sparse_routed_mlp_delta_values: self
                .device_real_sparse_routed_mlp_delta_values,
            device_real_sparse_routed_mlp_delta_checksum: self
                .device_real_sparse_routed_mlp_delta_checksum,
            device_real_sparse_routed_mlp_delta_backend: self
                .device_real_sparse_routed_mlp_delta_backend
                .unwrap_or("not-run"),
            device_real_sparse_routed_mlp_route_backend: self
                .device_real_sparse_routed_mlp_route_backend
                .unwrap_or("not-run"),
            device_real_sparse_routed_mlp_router_backend: self
                .device_real_sparse_routed_mlp_router_backend
                .unwrap_or("not-run"),
            device_real_sparse_routed_mlp_routes: self.device_real_sparse_routed_mlp_routes,
            device_real_sparse_routed_mlp_router_weight_bytes: self
                .device_real_sparse_routed_mlp_router_weight_bytes,
            device_real_sparse_routed_mlp_router_bias_bytes: self
                .device_real_sparse_routed_mlp_router_bias_bytes,
            device_real_sparse_routed_mlp_route_cache_cuda_entries: self
                .device_real_sparse_routed_mlp_route_cache_cuda_entries,
            device_real_sparse_routed_mlp_route_cache_cuda_uploads: self
                .device_real_sparse_routed_mlp_route_cache_cuda_uploads,
            device_real_sparse_routed_mlp_route_cache_cuda_hits: self
                .device_real_sparse_routed_mlp_route_cache_cuda_hits,
            device_real_sparse_routed_mlp_router_cache_entries: self
                .device_real_sparse_routed_mlp_router_cache_entries,
            device_real_sparse_routed_mlp_router_cache_hits: self
                .device_real_sparse_routed_mlp_router_cache_hits,
            device_real_sparse_routed_mlp_layers,
            device_real_sparse_routed_mlp_source_segments: self
                .device_real_sparse_routed_mlp_source_segments,
            uses_device_real_sparse_routed_mlp_delta,
            device_hidden_segment_status,
            device_hidden_segment_residual_adds: self.device_hidden_segment_residual_adds,
            device_hidden_segment_value_updates: self.device_hidden_segment_value_updates,
            device_hidden_segment_residual_add_backend: self
                .device_hidden_segment_residual_add_backend
                .unwrap_or("not-run"),
            device_hidden_segment_resident_segments: device_hidden_segment_summary
                .resident_segments,
            device_hidden_segment_resident_values: device_hidden_segment_summary.resident_values,
            device_hidden_segment_final_checksum: device_hidden_segment_summary.final_checksum,
            expected_device_hidden_segment_final_checksum: device_hidden_segment_summary
                .expected_final_checksum,
            uses_device_hidden_segment_residual_add,
            final_visible_checksum,
            expected_visible_checksum,
            rejected_mtp_checksum,
            expected_rejected_mtp_checksum,
            passed,
        };
        let target_hidden_byte_start = if self.retain_full_target_device_hidden {
            0
        } else {
            prefill_bytes
        };
        let final_target_device_hidden = self
            .copy_final_target_device_hidden(target_hidden_byte_start, self.residual_bf16.len())?;
        let target_device_hidden_taps = self.take_target_device_hidden_taps()?;
        let final_decode_device_hidden =
            self.take_final_decode_device_hidden(prefill_bytes, mtp_start_bytes)?;
        let sparse_tcp_dispatch_probe = self
            .sparse_tcp_routed_mlp
            .take()
            .map(RealFullSchedulerSparseTcpRoutedMlpContext::finish);
        Ok(RealFullSchedulerNumericProgressionFinish {
            self_test,
            final_decode_device_hidden,
            final_target_device_hidden,
            target_device_hidden_taps,
            sparse_tcp_dispatch_probe,
        })
    }

    pub(super) fn finish_live_request(
        mut self,
    ) -> Result<RealFullSchedulerNumericProgressionFinish> {
        let prefill_values = self.shape.prefill_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let decode_values = self.shape.decode_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let mtp_start = prefill_values + decode_values;
        let prefill_bytes = prefill_values * std::mem::size_of::<u16>();
        let mtp_start_bytes = mtp_start * std::mem::size_of::<u16>();
        let target_hidden_byte_start = if self.retain_full_target_device_hidden {
            0
        } else {
            prefill_bytes
        };
        let final_target_device_hidden = self
            .copy_final_target_device_hidden(target_hidden_byte_start, self.residual_bf16.len())?;
        let target_device_hidden_taps = self.take_target_device_hidden_taps()?;
        let final_decode_device_hidden =
            self.take_final_decode_device_hidden(prefill_bytes, mtp_start_bytes)?;
        let has_final_decode_device_hidden = final_decode_device_hidden.is_some();
        let self_test = self.live_request_self_test(has_final_decode_device_hidden);
        let sparse_tcp_dispatch_probe = self
            .sparse_tcp_routed_mlp
            .take()
            .map(RealFullSchedulerSparseTcpRoutedMlpContext::finish);
        Ok(RealFullSchedulerNumericProgressionFinish {
            self_test,
            final_decode_device_hidden,
            final_target_device_hidden,
            target_device_hidden_taps,
            sparse_tcp_dispatch_probe,
        })
    }

    fn take_target_device_hidden_taps(
        &mut self,
    ) -> Result<Option<RealFullSchedulerTargetHiddenTaps>> {
        if self.target_device_hidden_tap_rows == 0 {
            return Ok(None);
        }
        let missing = self
            .target_device_hidden_taps
            .iter()
            .enumerate()
            .filter_map(|(index, tap)| {
                tap.is_none()
                    .then_some(self.target_device_hidden_tap_layer_ids[index])
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            missing.is_empty(),
            "scheduler dSpark target hidden taps are missing layer boundaries {missing:?}"
        );
        let values = self
            .target_device_hidden_taps
            .iter_mut()
            .map(|tap| {
                tap.take()
                    .expect("scheduler dSpark taps were checked above")
            })
            .collect::<Vec<_>>();
        let rows = values[0].rows;
        let total_rows = self
            .shape
            .prefill_rows
            .checked_add(self.shape.decode_rows)
            .and_then(|rows| rows.checked_add(self.shape.mtp_rows))
            .context("scheduler dSpark final target tap row count overflow")?;
        let expected_rows = self.target_device_hidden_tap_rows.min(total_rows);
        anyhow::ensure!(
            rows == expected_rows
                && values.iter().all(|tap| {
                    tap.rows == rows && tap.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM
                }),
            "scheduler dSpark target hidden tap geometry changed"
        );
        Ok(Some(RealFullSchedulerTargetHiddenTaps {
            layer_ids: self.target_device_hidden_tap_layer_ids.clone(),
            row_start: total_rows - expected_rows,
            rows,
            values,
        }))
    }

    fn live_request_self_test(
        &self,
        has_final_decode_device_hidden: bool,
    ) -> RealFullSchedulerNumericProgressionSelfTest {
        let selected_rows =
            self.selected_prefill_rows + self.selected_decode_rows + self.selected_mtp_rows;
        let expected_selected_rows = self.shape.unique_rows() * GLM52_NUM_HIDDEN_LAYERS;
        let expected_value_updates = expected_selected_rows * NUMERIC_PROGRESS_HIDDEN_DIM;
        let expected_source_segments = GLM52_FIRST_K_DENSE_REPLACE
            * self.shape.source_segments_per_layer
            + (GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE)
                * self.shape.sparse_source_segments_per_layer;
        let hidden_segment_resident_values = self
            .device_hidden_segments
            .values()
            .map(|segment| segment.rows.saturating_mul(segment.values_per_row))
            .sum::<usize>();
        let uses_device_attention_output_delta = self.attention_device_output_delta_rows > 0;
        let uses_full_width_device_attention_delta = uses_device_attention_output_delta
            && self.attention_device_output_delta_values
                == self
                    .attention_device_output_delta_rows
                    .saturating_mul(NUMERIC_PROGRESS_HIDDEN_DIM);
        let device_attention_output_delta_status = match (
            uses_device_attention_output_delta,
            uses_full_width_device_attention_delta,
        ) {
            (true, true) => "cuda-device-attention-hidden-delta",
            (true, false) => "cuda-device-attention-output-prefix-delta",
            (false, _) => "not-run",
        };
        let uses_device_mlp_delta = self.device_mlp_delta_rows > 0;
        let uses_device_real_dense_mlp_delta = self.device_real_dense_mlp_delta_rows > 0;
        let uses_device_real_sparse_shared_mlp_delta =
            self.device_real_sparse_shared_mlp_delta_rows > 0;
        let uses_device_real_sparse_routed_mlp_delta =
            self.device_real_sparse_routed_mlp_delta_rows > 0;
        let uses_device_hidden_segment_residual_add = self.device_hidden_segment_residual_adds > 0;
        let passed = self.selected_prefill_rows
            == self.shape.prefill_rows * GLM52_NUM_HIDDEN_LAYERS
            && self.selected_decode_rows == self.shape.decode_rows * GLM52_NUM_HIDDEN_LAYERS
            && self.selected_mtp_rows == self.shape.mtp_rows * GLM52_NUM_HIDDEN_LAYERS
            && selected_rows == expected_selected_rows
            && self.source_segments == expected_source_segments
            && self.attention_residual_adds == expected_source_segments
            && self.mlp_residual_adds == expected_source_segments
            && self.attention_value_updates == expected_value_updates
            && self.mlp_value_updates == expected_value_updates
            && self.attention_residual_add_backend.is_some()
            && self.mlp_residual_add_backend.is_some()
            && (self.shape.decode_rows == 0 || has_final_decode_device_hidden)
            && uses_device_hidden_segment_residual_add;

        RealFullSchedulerNumericProgressionSelfTest {
            status: "numeric-scheduler-progression-live-summary",
            scope:
                "summarize live scheduler progression without running the preflight proof self-test",
            layers: GLM52_NUM_HIDDEN_LAYERS,
            source_modes: ["prefill", "decode", "mtp_verify"],
            unique_source_rows: self.shape.unique_rows(),
            hidden_dim: NUMERIC_PROGRESS_HIDDEN_DIM,
            residual_dtype: NUMERIC_PROGRESS_RESIDUAL_DTYPE,
            selected_prefill_rows: self.selected_prefill_rows,
            selected_decode_rows: self.selected_decode_rows,
            selected_mtp_rows: self.selected_mtp_rows,
            mtp_accepted_rows_per_layer: self.shape.mtp_accepted_rows,
            mtp_rejected_rows_per_layer: self.shape.mtp_rows - self.shape.mtp_accepted_rows,
            source_segments: self.source_segments,
            attention_residual_adds: self.attention_residual_adds,
            mlp_residual_adds: self.mlp_residual_adds,
            attention_residual_add_backend: self
                .attention_residual_add_backend
                .unwrap_or("not-run"),
            mlp_residual_add_backend: self.mlp_residual_add_backend.unwrap_or("not-run"),
            attention_value_updates: self.attention_value_updates,
            mlp_value_updates: self.mlp_value_updates,
            device_attention_output_delta_status,
            attention_device_output_delta_rows: self.attention_device_output_delta_rows,
            attention_device_output_delta_values: self.attention_device_output_delta_values,
            attention_device_output_delta_checksum: 0.0,
            attention_device_output_delta_backend: self
                .attention_device_output_delta_backend
                .unwrap_or("not-run"),
            attention_device_output_delta_device_prefix_rows: self
                .attention_device_output_delta_device_prefix_rows,
            attention_device_output_delta_device_prefix_values: self
                .attention_device_output_delta_device_prefix_values,
            attention_device_output_delta_device_prefix_backend: self
                .attention_device_output_delta_device_prefix_backend
                .unwrap_or("not-run"),
            uses_device_attention_output_delta,
            device_delta_template_status: "live-summary",
            device_delta_template_uploads: self.device_delta_template_uploads,
            device_delta_template_uses: self.device_delta_template_uses,
            device_delta_template_resident_values: self.device_delta_template_resident_values,
            device_mlp_delta_status: if uses_device_mlp_delta {
                SCHEDULER_MLP_DELTA_BACKEND
            } else {
                "not-run"
            },
            device_mlp_delta_rows: self.device_mlp_delta_rows,
            device_mlp_delta_values: self.device_mlp_delta_values,
            device_mlp_delta_checksum: 0.0,
            device_mlp_delta_backend: self.device_mlp_delta_backend.unwrap_or("not-run"),
            device_mlp_weight_uploads: self.device_mlp_weight_uploads,
            device_mlp_weight_resident_values: self.device_mlp_weight_resident_values,
            uses_device_mlp_delta,
            device_real_dense_mlp_delta_status: if uses_device_real_dense_mlp_delta {
                SCHEDULER_REAL_DENSE_MLP_DELTA_STATUS
            } else {
                "not-run"
            },
            device_real_dense_mlp_delta_rows: self.device_real_dense_mlp_delta_rows,
            device_real_dense_mlp_delta_values: self.device_real_dense_mlp_delta_values,
            device_real_dense_mlp_delta_checksum: 0.0,
            device_real_dense_mlp_delta_backend: self
                .device_real_dense_mlp_delta_backend
                .unwrap_or("not-run"),
            device_real_dense_mlp_norm_backend: self
                .device_real_dense_mlp_norm_backend
                .unwrap_or("not-run"),
            device_real_dense_mlp_weight_tensors: self.device_real_dense_mlp_weight_tensors,
            device_real_dense_mlp_weight_bytes: self.device_real_dense_mlp_weight_bytes,
            device_real_dense_mlp_layers: self.device_real_dense_mlp_layers.len(),
            device_real_dense_mlp_source_segments: self.device_real_dense_mlp_source_segments,
            uses_device_real_dense_mlp_delta,
            device_real_sparse_shared_mlp_delta_status: if uses_device_real_sparse_shared_mlp_delta
            {
                "cuda-real-sparse-shared-checkpoint-mlp-delta"
            } else {
                "not-run"
            },
            device_real_sparse_shared_mlp_delta_rows: self.device_real_sparse_shared_mlp_delta_rows,
            device_real_sparse_shared_mlp_delta_values: self
                .device_real_sparse_shared_mlp_delta_values,
            device_real_sparse_shared_mlp_delta_checksum: 0.0,
            device_real_sparse_shared_mlp_delta_backend: self
                .device_real_sparse_shared_mlp_delta_backend
                .unwrap_or("not-run"),
            device_real_sparse_shared_mlp_norm_backend: self
                .device_real_sparse_shared_mlp_norm_backend
                .unwrap_or("not-run"),
            device_real_sparse_shared_mlp_weight_tensors: self
                .device_real_sparse_shared_mlp_weight_tensors,
            device_real_sparse_shared_mlp_weight_bytes: self
                .device_real_sparse_shared_mlp_weight_bytes,
            device_real_sparse_shared_mlp_layers: self.device_real_sparse_shared_mlp_layers.len(),
            device_real_sparse_shared_mlp_source_segments: self
                .device_real_sparse_shared_mlp_source_segments,
            uses_device_real_sparse_shared_mlp_delta,
            device_real_sparse_routed_mlp_delta_status: if uses_device_real_sparse_routed_mlp_delta
            {
                SCHEDULER_REAL_SPARSE_ROUTED_MLP_DELTA_STATUS
            } else {
                "not-run"
            },
            device_real_sparse_routed_mlp_delta_rows: self.device_real_sparse_routed_mlp_delta_rows,
            device_real_sparse_routed_mlp_delta_values: self
                .device_real_sparse_routed_mlp_delta_values,
            device_real_sparse_routed_mlp_delta_checksum: 0.0,
            device_real_sparse_routed_mlp_delta_backend: self
                .device_real_sparse_routed_mlp_delta_backend
                .unwrap_or("not-run"),
            device_real_sparse_routed_mlp_route_backend: self
                .device_real_sparse_routed_mlp_route_backend
                .unwrap_or("not-run"),
            device_real_sparse_routed_mlp_router_backend: self
                .device_real_sparse_routed_mlp_router_backend
                .unwrap_or("not-run"),
            device_real_sparse_routed_mlp_routes: self.device_real_sparse_routed_mlp_routes,
            device_real_sparse_routed_mlp_router_weight_bytes: self
                .device_real_sparse_routed_mlp_router_weight_bytes,
            device_real_sparse_routed_mlp_router_bias_bytes: self
                .device_real_sparse_routed_mlp_router_bias_bytes,
            device_real_sparse_routed_mlp_route_cache_cuda_entries: self
                .device_real_sparse_routed_mlp_route_cache_cuda_entries,
            device_real_sparse_routed_mlp_route_cache_cuda_uploads: self
                .device_real_sparse_routed_mlp_route_cache_cuda_uploads,
            device_real_sparse_routed_mlp_route_cache_cuda_hits: self
                .device_real_sparse_routed_mlp_route_cache_cuda_hits,
            device_real_sparse_routed_mlp_router_cache_entries: self
                .device_real_sparse_routed_mlp_router_cache_entries,
            device_real_sparse_routed_mlp_router_cache_hits: self
                .device_real_sparse_routed_mlp_router_cache_hits,
            device_real_sparse_routed_mlp_layers: self.device_real_sparse_routed_mlp_layers.len(),
            device_real_sparse_routed_mlp_source_segments: self
                .device_real_sparse_routed_mlp_source_segments,
            uses_device_real_sparse_routed_mlp_delta,
            device_hidden_segment_status: if uses_device_hidden_segment_residual_add {
                "cuda-device-hidden-segment-residual-add"
            } else {
                "not-run"
            },
            device_hidden_segment_residual_adds: self.device_hidden_segment_residual_adds,
            device_hidden_segment_value_updates: self.device_hidden_segment_value_updates,
            device_hidden_segment_residual_add_backend: self
                .device_hidden_segment_residual_add_backend
                .unwrap_or("not-run"),
            device_hidden_segment_resident_segments: self.device_hidden_segments.len(),
            device_hidden_segment_resident_values: hidden_segment_resident_values,
            device_hidden_segment_final_checksum: 0.0,
            expected_device_hidden_segment_final_checksum: 0.0,
            uses_device_hidden_segment_residual_add,
            final_visible_checksum: 0.0,
            expected_visible_checksum: 0.0,
            rejected_mtp_checksum: 0.0,
            expected_rejected_mtp_checksum: 0.0,
            passed,
        }
    }

    fn copy_final_target_device_hidden(
        &self,
        byte_start: usize,
        byte_end: usize,
    ) -> Result<Option<DeviceBf16Output>> {
        if !self.retain_final_target_device_hidden || !coordinator_cuda_reference_kernels_enabled()
        {
            return Ok(None);
        }
        anyhow::ensure!(
            byte_end > byte_start,
            "scheduler final target device hidden requires a non-empty current suffix"
        );
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler final target row bytes overflow")?;
        anyhow::ensure!(
            byte_start % row_bytes == 0 && byte_end % row_bytes == 0,
            "scheduler final target byte range {byte_start}..{byte_end} is not row aligned to {row_bytes}"
        );

        let mut cursor = byte_start;
        let mut batches = Vec::new();
        while cursor < byte_end {
            let (key, batch) = self
                .device_hidden_segments
                .iter()
                .filter(|(key, _)| key.byte_start == cursor && key.byte_end <= byte_end)
                .min_by_key(|(key, _)| key.byte_end - key.byte_start)
                .with_context(|| {
                    format!(
                        "scheduler final target device hidden has no resident segment starting at byte {cursor} of {byte_end}"
                    )
                })?;
            let segment_bytes = key
                .byte_end
                .checked_sub(key.byte_start)
                .context("scheduler final target segment byte range underflow")?;
            anyhow::ensure!(
                segment_bytes == batch.rows.saturating_mul(row_bytes)
                    && batch.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler final target segment {}..{} shape {}x{} is invalid",
                key.byte_start,
                key.byte_end,
                batch.rows,
                batch.values_per_row
            );
            batches.push(batch);
            cursor = key.byte_end;
        }
        let target = concat_device_bf16_row_batches(
            batches.as_slice(),
            "scheduler final target suffix hidden rows",
        )?;
        anyhow::ensure!(
            target.rows == (byte_end - byte_start) / row_bytes
                && target.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler final target hidden shape mismatch: expected {}x{}, got {}x{}",
            (byte_end - byte_start) / row_bytes,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            target.rows,
            target.values_per_row
        );
        Ok(Some(target))
    }

    fn take_final_decode_device_hidden(
        &mut self,
        byte_start: usize,
        byte_end: usize,
    ) -> Result<Option<DeviceBf16Output>> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(None);
        }
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        let Some(hidden) = self.device_hidden_segments.remove(&key) else {
            if self.device_hidden_segment_residual_adds > 0 {
                anyhow::bail!(
                    "scheduler final decode device hidden segment is missing after resident progression"
                );
            }
            return Ok(None);
        };
        if hidden.rows != self.shape.decode_rows
            || hidden.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM
        {
            anyhow::bail!(
                "scheduler final decode device hidden shape mismatch: expected {}x{} got {}x{}",
                self.shape.decode_rows,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                hidden.rows,
                hidden.values_per_row
            );
        }
        Ok(Some(hidden))
    }

    fn numeric_progression_row_index(
        &self,
        kind: RowSourceKind,
        token_start: usize,
        row_offset: usize,
    ) -> Result<usize> {
        match kind {
            RowSourceKind::PrefillChunk => {
                let prefill_offset = token_start
                    .checked_sub(self.shape.prefix_tokens)
                    .with_context(|| {
                        format!(
                            "numeric prefill token_start {token_start} precedes prefix tokens {}",
                            self.shape.prefix_tokens
                        )
                    })?;
                let row_index = prefill_offset + row_offset;
                if row_index >= self.shape.prefill_rows {
                    anyhow::bail!(
                        "numeric prefill row index {row_index} exceeds {} rows",
                        self.shape.prefill_rows
                    );
                }
                Ok(row_index)
            }
            RowSourceKind::DecodeStep => {
                let decode_token_start = self
                    .shape
                    .prefix_tokens
                    .checked_add(self.shape.prefill_rows)
                    .context("numeric decode token start overflows usize")?;
                let decode_offset = token_start
                    .checked_sub(decode_token_start)
                    .with_context(|| {
                        format!(
                            "numeric decode token_start {token_start} precedes decode rows start {decode_token_start}"
                        )
                    })?;
                let row_index = decode_offset
                    .checked_add(row_offset)
                    .context("numeric decode row index overflows usize")?;
                if row_index >= self.shape.decode_rows {
                    anyhow::bail!(
                        "numeric decode row offset {row_index} exceeds {} rows",
                        self.shape.decode_rows
                    );
                }
                Ok(self.shape.prefill_rows + row_index)
            }
            RowSourceKind::MtpVerifyBlock => {
                let mtp_token_start = self
                    .shape
                    .prefix_tokens
                    .checked_add(self.shape.prefill_rows)
                    .and_then(|start| start.checked_add(self.shape.decode_rows))
                    .context("numeric MTP start row overflows usize")?;
                let mtp_offset = token_start.checked_sub(mtp_token_start).with_context(|| {
                    format!(
                        "numeric MTP token_start {token_start} precedes MTP rows start {mtp_token_start}"
                    )
                })?;
                let row_index = mtp_offset
                    .checked_add(row_offset)
                    .context("numeric MTP row index overflows usize")?;
                if row_index >= self.shape.mtp_rows {
                    anyhow::bail!(
                        "numeric MTP row offset {row_index} exceeds {} rows",
                        self.shape.mtp_rows
                    );
                }
                Ok(self.shape.prefill_rows + self.shape.decode_rows + row_index)
            }
            RowSourceKind::Benchmark => {
                anyhow::bail!("benchmark rows are not part of real-full progression")
            }
        }
    }

    fn apply_source_delta(
        &mut self,
        layer_id: usize,
        catalog: &TensorCatalog,
        start_row_index: usize,
        row_count: usize,
        kind: RowSourceKind,
        source: &RowSource,
        graph_bucket: GraphBucket,
        placement_version: &glmrt_core::PlacementVersion,
        attention_delta: Option<&RealFullSchedulerDeviceAttentionDelta>,
    ) -> Result<()> {
        if row_count == 0 {
            return Ok(());
        }
        let end_row_index = start_row_index + row_count;
        let start = start_row_index * NUMERIC_PROGRESS_HIDDEN_DIM;
        let end = end_row_index * NUMERIC_PROGRESS_HIDDEN_DIM;
        let byte_start = start * std::mem::size_of::<u16>();
        let byte_end = end * std::mem::size_of::<u16>();
        if byte_end > self.residual_bf16.len() {
            anyhow::bail!(
                "numeric progression row range {start_row_index}..{end_row_index} exceeds residual rows {}",
                self.shape.unique_rows()
            );
        }
        let (deterministic_attention_delta, mlp_delta) = numeric_progression_deltas(kind);
        self.source_segments += 1;
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        dump_layer_boundary_device_hidden(
            layer_id,
            source.kind,
            source.token_start.0,
            source.row_count,
            "input",
            &self.device_hidden_segments,
            key,
        )?;
        self.apply_attention_delta(
            byte_start,
            byte_end,
            deterministic_attention_delta,
            row_count,
            kind,
            attention_delta,
        )?;
        dump_layer_boundary_device_hidden(
            layer_id,
            source.kind,
            source.token_start.0,
            source.row_count,
            "post_attention",
            &self.device_hidden_segments,
            key,
        )?;
        self.apply_mlp_delta(
            byte_start,
            byte_end,
            row_count,
            mlp_delta,
            layer_id,
            kind,
            source,
            graph_bucket,
            placement_version,
            catalog,
        )?;
        dump_layer_boundary_device_hidden(
            layer_id,
            source.kind,
            source.token_start.0,
            source.row_count,
            "post_mlp",
            &self.device_hidden_segments,
            key,
        )?;
        Ok(())
    }

    fn apply_attention_delta(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        deterministic_delta: f32,
        row_count: usize,
        kind: RowSourceKind,
        attention_delta: Option<&RealFullSchedulerDeviceAttentionDelta>,
    ) -> Result<()> {
        if let Some(attention_delta) = attention_delta {
            validate_device_attention_output_delta(row_count, kind, attention_delta)?;
            record_backend(
                &mut self.attention_device_output_delta_backend,
                attention_delta.backend,
                "attention-device-output-delta",
            )?;
            self.attention_device_output_delta_rows += attention_delta.row_count;
            self.attention_device_output_delta_values += attention_delta
                .row_count
                .checked_mul(attention_delta.values_per_row)
                .context("scheduler attention output delta value count overflow")?;
            self.attention_device_output_delta_checksum += attention_delta.checksum;
            if attention_delta.values_per_row == NUMERIC_PROGRESS_HIDDEN_DIM {
                if attention_delta.output_device_row_offset == 0
                    && attention_delta.output_device.rows == row_count
                {
                    return self.apply_device_direct_delta_bytes(
                        byte_start,
                        byte_end,
                        ResidualDeltaStage::Attention,
                        &attention_delta.output_device,
                    );
                }
                return self.apply_device_direct_delta_view_bytes(
                    byte_start,
                    byte_end,
                    ResidualDeltaStage::Attention,
                    attention_delta,
                );
            }

            self.delta_bf16_scratch.resize(byte_end - byte_start, 0);
            fill_repeated_bf16_bytes(deterministic_delta, &mut self.delta_bf16_scratch);
            overlay_device_attention_output_delta(
                &mut self.delta_bf16_scratch,
                row_count,
                kind,
                attention_delta,
            )?;
            let device_delta_template =
                self.device_delta_template_for(row_count, deterministic_delta)?;
            return self.apply_delta_bytes(
                byte_start,
                byte_end,
                ResidualDeltaStage::Attention,
                Some(attention_delta),
                device_delta_template,
                None,
            );
        } else {
            self.delta_bf16_scratch.resize(byte_end - byte_start, 0);
            fill_repeated_bf16_bytes(deterministic_delta, &mut self.delta_bf16_scratch);
            let device_delta_template =
                self.device_delta_template_for(row_count, deterministic_delta)?;
            self.apply_delta_bytes(
                byte_start,
                byte_end,
                ResidualDeltaStage::Attention,
                None,
                device_delta_template,
                None,
            )
        }
    }

    fn apply_constant_delta(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        delta: f32,
        stage: ResidualDeltaStage,
    ) -> Result<()> {
        self.delta_bf16_scratch.resize(byte_end - byte_start, 0);
        fill_repeated_bf16_bytes(delta, &mut self.delta_bf16_scratch);
        let row_count = row_count_from_delta_byte_range(byte_start, byte_end)?;
        let device_delta_template = self.device_delta_template_for(row_count, delta)?;
        self.apply_delta_bytes(
            byte_start,
            byte_end,
            stage,
            None,
            device_delta_template,
            None,
        )
    }

    fn apply_mlp_delta(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        row_count: usize,
        fallback_delta: f32,
        layer_id: usize,
        kind: RowSourceKind,
        source: &RowSource,
        graph_bucket: GraphBucket,
        placement_version: &glmrt_core::PlacementVersion,
        catalog: &TensorCatalog,
    ) -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return self.apply_constant_delta(
                byte_start,
                byte_end,
                fallback_delta,
                ResidualDeltaStage::Mlp,
            );
        }
        if synthetic_sparse_spark_expert_mode_for_layer(layer_id) {
            return self.apply_constant_delta(
                byte_start,
                byte_end,
                fallback_delta,
                ResidualDeltaStage::Mlp,
            );
        }
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        let hidden = self
            .device_hidden_segments
            .get(&key)
            .context("scheduler resident hidden segment missing before MLP delta")?;
        if hidden.rows != row_count || hidden.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM {
            anyhow::bail!(
                "scheduler resident hidden segment shape mismatch before MLP delta: expected {}x{} got {}x{}",
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                hidden.rows,
                hidden.values_per_row
            );
        }
        let hidden_buffer = hidden.buffer();
        let use_real_dense_mlp = should_use_real_dense_scheduler_mlp(layer_id);
        let use_real_sparse_shared_mlp =
            should_use_real_sparse_shared_scheduler_mlp(layer_id, kind);
        let mlp_delta = if use_real_dense_mlp {
            let mlp_delta = self.device_real_dense_mlp_delta_from_hidden(
                catalog,
                layer_id,
                hidden_buffer,
                row_count,
            )
            .with_context(|| {
                format!(
                    "building real dense checkpoint MLP delta for scheduler layer {layer_id} kind={kind:?}"
                )
            })?;
            self.record_real_dense_mlp_delta(layer_id, row_count, &mlp_delta)?;
            mlp_delta
        } else if use_real_sparse_shared_mlp {
            let normalized = self
                .device_real_sparse_post_attention_norm_from_hidden(
                    catalog,
                    layer_id,
                    hidden_buffer,
                    row_count,
                )
                .with_context(|| {
                    format!(
                        "building real sparse scheduler post-attention normalized hidden for layer {layer_id} kind={kind:?}"
                    )
                })?;
            let shared_delta = self
                .device_real_sparse_shared_mlp_delta_from_normalized(
                    catalog,
                    layer_id,
                    normalized.buffer(),
                    row_count,
                )
                .with_context(|| {
                    format!(
                        "building real sparse shared checkpoint MLP delta for scheduler layer {layer_id} kind={kind:?}"
                    )
                })?;
            self.record_real_sparse_shared_mlp_delta(layer_id, row_count, &shared_delta)?;
            let routed_delta = self
                .device_real_sparse_routed_mlp_delta_from_normalized(
                    catalog,
                    layer_id,
                    source,
                    graph_bucket,
                    placement_version,
                    &normalized,
                    row_count,
                )
                .with_context(|| {
                    format!(
                        "building real sparse routed NVFP4 checkpoint MLP delta for scheduler layer {layer_id} kind={kind:?}"
                    )
                })?;
            self.record_real_sparse_routed_mlp_delta(layer_id, row_count, &routed_delta)?;
            residual_add_bf16_device_inputs_device_output(&shared_delta, &routed_delta)
                .context("combining real sparse shared and routed MLP deltas")?
        } else {
            let mlp_delta = self
                .device_mlp_delta_from_hidden(hidden_buffer, row_count)
                .context("building scheduler hidden-dependent MLP delta")?;
            self.record_synthetic_mlp_delta(row_count, &mlp_delta)?;
            mlp_delta
        };
        self.apply_device_direct_delta_bytes(
            byte_start,
            byte_end,
            ResidualDeltaStage::Mlp,
            &mlp_delta,
        )
    }

    fn apply_delta_bytes(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        stage: ResidualDeltaStage,
        device_delta_prefix: Option<&RealFullSchedulerDeviceAttentionDelta>,
        device_delta_template: Option<DeviceDeltaTemplateView>,
        device_delta_direct: Option<&DeviceBf16Output>,
    ) -> Result<()> {
        self.output_bf16_scratch.resize(byte_end - byte_start, 0);
        let backend = residual_add_prefix_bf16_bytes_into(
            &self.residual_bf16[byte_start..byte_end],
            &self.delta_bf16_scratch,
            &mut self.output_bf16_scratch,
        )?;
        let device_hidden_segment = apply_device_hidden_segment_residual_add(
            &mut self.device_hidden_segments,
            &self.residual_bf16,
            byte_start,
            byte_end,
            &self.delta_bf16_scratch,
            device_delta_prefix,
            device_delta_template,
            device_delta_direct,
        )?;
        let values_updated = self.output_bf16_scratch.len() / std::mem::size_of::<u16>();
        if let Some(device_hidden_segment) = device_hidden_segment {
            record_backend(
                &mut self.device_hidden_segment_residual_add_backend,
                device_hidden_segment.backend,
                "device-hidden-segment",
            )?;
            self.device_hidden_segment_residual_adds += 1;
            self.device_hidden_segment_value_updates += device_hidden_segment.values_updated;
            if let Some(device_prefix_values) = device_hidden_segment.device_prefix_values {
                self.attention_device_output_delta_device_prefix_rows +=
                    device_hidden_segment.device_prefix_rows;
                self.attention_device_output_delta_device_prefix_values += device_prefix_values;
                record_backend(
                    &mut self.attention_device_output_delta_device_prefix_backend,
                    device_hidden_segment.delta_backend,
                    "attention-device-output-delta-device-prefix",
                )?;
            }
        }
        self.residual_bf16[byte_start..byte_end].copy_from_slice(&self.output_bf16_scratch);
        match stage {
            ResidualDeltaStage::Attention => {
                record_backend(
                    &mut self.attention_residual_add_backend,
                    backend,
                    "attention",
                )?;
                self.attention_residual_adds += 1;
                self.attention_value_updates += values_updated;
            }
            ResidualDeltaStage::Mlp => {
                record_backend(&mut self.mlp_residual_add_backend, backend, "mlp")?;
                self.mlp_residual_adds += 1;
                self.mlp_value_updates += values_updated;
            }
        }
        Ok(())
    }

    fn apply_device_direct_delta_bytes(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        stage: ResidualDeltaStage,
        device_delta_direct: &DeviceBf16Output,
    ) -> Result<()> {
        let segment_bytes = byte_end
            .checked_sub(byte_start)
            .context("scheduler direct device delta byte range underflows usize")?;
        let rows = row_count_from_delta_byte_range(byte_start, byte_end)?;
        validate_direct_device_delta(device_delta_direct, rows)?;
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        if !self.device_hidden_segments.contains_key(&key) {
            let initial = device_bf16_output_from_bf16_bytes(
                &self.residual_bf16[byte_start..byte_end],
                rows,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler numeric resident hidden segment",
            )?;
            self.device_hidden_segments.insert(key, initial);
        }
        let output = {
            let residual = self
                .device_hidden_segments
                .get(&key)
                .context("scheduler resident hidden segment missing for direct device delta")?;
            residual_add_bf16_device_inputs_device_output(residual, device_delta_direct)
                .context("applying scheduler direct device delta to resident hidden segment")?
        };
        let backend = output.backend;
        self.device_hidden_segments.insert(key, output);

        let values_updated = segment_bytes / std::mem::size_of::<u16>();
        record_backend(
            &mut self.device_hidden_segment_residual_add_backend,
            backend,
            "device-hidden-segment",
        )?;
        self.device_hidden_segment_residual_adds += 1;
        self.device_hidden_segment_value_updates += values_updated;
        match stage {
            ResidualDeltaStage::Attention => {
                record_backend(
                    &mut self.attention_residual_add_backend,
                    backend,
                    "attention",
                )?;
                self.attention_residual_adds += 1;
                self.attention_value_updates += values_updated;
            }
            ResidualDeltaStage::Mlp => {
                record_backend(&mut self.mlp_residual_add_backend, backend, "mlp")?;
                self.mlp_residual_adds += 1;
                self.mlp_value_updates += values_updated;
            }
        }
        Ok(())
    }

    fn apply_device_direct_delta_view_bytes(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        stage: ResidualDeltaStage,
        delta: &RealFullSchedulerDeviceAttentionDelta,
    ) -> Result<()> {
        let segment_bytes = byte_end
            .checked_sub(byte_start)
            .context("scheduler direct device delta-view byte range underflows usize")?;
        let rows = row_count_from_delta_byte_range(byte_start, byte_end)?;
        validate_device_attention_output_delta(rows, delta.kind, delta)?;
        let delta_byte_offset = delta
            .output_device_row_offset
            .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("scheduler direct device delta-view byte offset overflow")?;
        let delta_view = device_buffer_byte_view(
            delta.output_device.buffer(),
            delta_byte_offset,
            segment_bytes,
            "scheduler attention hidden delta row view",
        )?;
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        if !self.device_hidden_segments.contains_key(&key) {
            let initial = device_bf16_output_from_bf16_bytes(
                &self.residual_bf16[byte_start..byte_end],
                rows,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler numeric resident hidden segment",
            )?;
            self.device_hidden_segments.insert(key, initial);
        }
        let output = {
            let residual = self
                .device_hidden_segments
                .get(&key)
                .context("scheduler resident hidden segment missing for direct delta view")?;
            residual_add_bf16_device_input_delta_view_device_output(
                residual,
                &delta.output_device,
                delta_view,
            )
            .context("applying scheduler direct device delta row view")?
        };
        let backend = output.backend;
        self.device_hidden_segments.insert(key, output);

        let values_updated = segment_bytes / std::mem::size_of::<u16>();
        record_backend(
            &mut self.device_hidden_segment_residual_add_backend,
            backend,
            "device-hidden-segment",
        )?;
        self.device_hidden_segment_residual_adds += 1;
        self.device_hidden_segment_value_updates += values_updated;
        match stage {
            ResidualDeltaStage::Attention => {
                record_backend(
                    &mut self.attention_residual_add_backend,
                    backend,
                    "attention",
                )?;
                self.attention_residual_adds += 1;
                self.attention_value_updates += values_updated;
            }
            ResidualDeltaStage::Mlp => {
                record_backend(&mut self.mlp_residual_add_backend, backend, "mlp")?;
                self.mlp_residual_adds += 1;
                self.mlp_value_updates += values_updated;
            }
        }
        Ok(())
    }

    fn apply_device_residual_output_bytes(
        &mut self,
        byte_start: usize,
        byte_end: usize,
        stage: ResidualDeltaStage,
        output: DeviceBf16Output,
    ) -> Result<()> {
        let segment_bytes = byte_end
            .checked_sub(byte_start)
            .context("scheduler direct device residual output byte range underflows usize")?;
        let rows = row_count_from_delta_byte_range(byte_start, byte_end)?;
        validate_direct_device_delta(&output, rows)?;
        let key = DeviceHiddenSegmentKey {
            byte_start,
            byte_end,
        };
        let backend = output.backend;
        self.device_hidden_segments.insert(key, output);

        let values_updated = segment_bytes / std::mem::size_of::<u16>();
        record_backend(
            &mut self.device_hidden_segment_residual_add_backend,
            backend,
            "device-hidden-segment",
        )?;
        self.device_hidden_segment_residual_adds += 1;
        self.device_hidden_segment_value_updates += values_updated;
        match stage {
            ResidualDeltaStage::Attention => {
                record_backend(
                    &mut self.attention_residual_add_backend,
                    backend,
                    "attention",
                )?;
                self.attention_residual_adds += 1;
                self.attention_value_updates += values_updated;
            }
            ResidualDeltaStage::Mlp => {
                record_backend(&mut self.mlp_residual_add_backend, backend, "mlp")?;
                self.mlp_residual_adds += 1;
                self.mlp_value_updates += values_updated;
            }
        }
        Ok(())
    }

    fn device_delta_template_for(
        &mut self,
        row_count: usize,
        delta: f32,
    ) -> Result<Option<DeviceDeltaTemplateView>> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(None);
        }
        if row_count == 0 {
            anyhow::bail!("scheduler device delta template requires nonzero rows");
        }
        let key = DeviceDeltaTemplateKey {
            rows: row_count,
            delta_bits: bf16_bits(delta),
        };
        if !self.device_delta_templates.contains_key(&key) {
            let byte_count = row_count
                .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context("scheduler device delta template byte count overflows usize")?;
            self.device_delta_template_upload_bf16_scratch
                .resize(byte_count, 0);
            fill_repeated_bf16_bytes(delta, &mut self.device_delta_template_upload_bf16_scratch);
            let template = device_bf16_output_from_bf16_bytes(
                &self.device_delta_template_upload_bf16_scratch,
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                "scheduler numeric resident delta template",
            )?;
            self.device_delta_template_uploads += 1;
            self.device_delta_template_resident_values += row_count * NUMERIC_PROGRESS_HIDDEN_DIM;
            self.device_delta_templates.insert(key, template);
        }
        let template = self
            .device_delta_templates
            .get(&key)
            .context("scheduler device delta template missing after insertion")?;
        self.device_delta_template_uses += 1;
        Ok(Some(DeviceDeltaTemplateView {
            buffer: template.buffer(),
            rows: template.rows,
            values_per_row: template.values_per_row,
        }))
    }

    fn device_real_dense_mlp_delta_from_hidden(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        hidden_buffer: GlmrtDeviceBuffer,
        row_count: usize,
    ) -> Result<DeviceBf16Output> {
        if row_count == 0 {
            anyhow::bail!("scheduler real dense MLP delta requires nonzero rows");
        }
        let weights = self.ensure_real_dense_mlp_resident_weights(catalog, layer_id)?;
        let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
            &weights.norm_name,
            hidden_buffer,
            row_count,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )
        .with_context(|| format!("running real dense scheduler RMSNorm for layer {layer_id}"))?;
        record_backend(
            &mut self.device_real_dense_mlp_norm_backend,
            normalized.backend,
            "device-real-dense-mlp-norm",
        )?;
        silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
                &weights.gate_name,
                &weights.up_name,
                &weights.down_name,
                normalized.buffer(),
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                weights.intermediate_dim,
                weights.intermediate_dim,
                NUMERIC_PROGRESS_HIDDEN_DIM,
            )
            .with_context(|| {
                format!("running real dense scheduler MLP for layer {layer_id}")
            })
    }

    fn record_synthetic_mlp_delta(
        &mut self,
        row_count: usize,
        delta: &DeviceBf16Output,
    ) -> Result<()> {
        let values = scheduler_mlp_delta_values("synthetic", row_count, delta)?;
        self.device_mlp_delta_rows += row_count;
        self.device_mlp_delta_values += values;
        record_backend(
            &mut self.device_mlp_delta_backend,
            delta.backend,
            "device-mlp-delta",
        )
    }

    fn record_real_dense_mlp_delta(
        &mut self,
        layer_id: usize,
        row_count: usize,
        delta: &DeviceBf16Output,
    ) -> Result<()> {
        let values = scheduler_mlp_delta_values("real dense", row_count, delta)?;
        self.device_real_dense_mlp_delta_rows += row_count;
        self.device_real_dense_mlp_delta_values += values;
        self.device_real_dense_mlp_source_segments += 1;
        if !self.live_request {
            self.device_real_dense_mlp_layers.insert(layer_id);
        }
        record_backend(
            &mut self.device_real_dense_mlp_delta_backend,
            delta.backend,
            "device-real-dense-mlp-delta",
        )
    }

    fn record_real_sparse_shared_mlp_delta(
        &mut self,
        layer_id: usize,
        row_count: usize,
        delta: &DeviceBf16Output,
    ) -> Result<()> {
        let values = scheduler_mlp_delta_values("real sparse shared", row_count, delta)?;
        self.device_real_sparse_shared_mlp_delta_rows += row_count;
        self.device_real_sparse_shared_mlp_delta_values += values;
        self.device_real_sparse_shared_mlp_source_segments += 1;
        if !self.live_request {
            self.device_real_sparse_shared_mlp_layers.insert(layer_id);
        }
        record_backend(
            &mut self.device_real_sparse_shared_mlp_delta_backend,
            delta.backend,
            "device-real-sparse-shared-mlp-delta",
        )
    }

    fn record_real_sparse_shared_mlp_delta_accounting(
        &mut self,
        layer_id: usize,
        row_count: usize,
        backend: &'static str,
    ) -> Result<()> {
        let values = row_count
            .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
            .context("scheduler real sparse shared MLP fused value count overflow")?;
        self.device_real_sparse_shared_mlp_delta_rows += row_count;
        self.device_real_sparse_shared_mlp_delta_values += values;
        self.device_real_sparse_shared_mlp_source_segments += 1;
        if !self.live_request {
            self.device_real_sparse_shared_mlp_layers.insert(layer_id);
        }
        record_backend(
            &mut self.device_real_sparse_shared_mlp_delta_backend,
            backend,
            "device-real-sparse-shared-mlp-delta",
        )
    }

    fn record_real_sparse_routed_mlp_delta(
        &mut self,
        layer_id: usize,
        row_count: usize,
        delta: &DeviceBf16Output,
    ) -> Result<()> {
        let values = scheduler_mlp_delta_values("real sparse routed", row_count, delta)?;
        self.device_real_sparse_routed_mlp_delta_rows += row_count;
        self.device_real_sparse_routed_mlp_delta_values += values;
        self.device_real_sparse_routed_mlp_source_segments += 1;
        if !self.live_request {
            self.device_real_sparse_routed_mlp_layers.insert(layer_id);
        }
        record_backend(
            &mut self.device_real_sparse_routed_mlp_delta_backend,
            delta.backend,
            "device-real-sparse-routed-mlp-delta",
        )
    }

    fn record_real_sparse_routed_mlp_delta_accounting(
        &mut self,
        layer_id: usize,
        row_count: usize,
        backend: &'static str,
    ) -> Result<()> {
        let values = row_count
            .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
            .context("scheduler real sparse routed MLP fused value count overflow")?;
        self.device_real_sparse_routed_mlp_delta_rows += row_count;
        self.device_real_sparse_routed_mlp_delta_values += values;
        self.device_real_sparse_routed_mlp_source_segments += 1;
        if !self.live_request {
            self.device_real_sparse_routed_mlp_layers.insert(layer_id);
        }
        record_backend(
            &mut self.device_real_sparse_routed_mlp_delta_backend,
            backend,
            "device-real-sparse-routed-mlp-delta",
        )
    }

    fn ensure_real_dense_mlp_resident_weights(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
    ) -> Result<SchedulerDenseMlpResidentWeights> {
        if layer_id >= GLM52_FIRST_K_DENSE_REPLACE {
            anyhow::bail!(
                "scheduler real dense MLP only supports layers 0..{}, got {layer_id}",
                GLM52_FIRST_K_DENSE_REPLACE
            );
        }
        if let Some(weights) = self
            .device_real_dense_mlp_resident_weights_by_layer
            .get(&layer_id)
        {
            return Ok(weights.clone());
        }
        let cache_key = scheduler_mlp_layer_cache_key(catalog, layer_id);
        if let Some(weights) = scheduler_dense_mlp_resident_weight_cache()
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler dense MLP resident weight cache poisoned"))?
            .get(&cache_key)
            .cloned()
        {
            self.record_real_dense_mlp_resident_weight_usage(&weights);
            self.device_real_dense_mlp_resident_weights_by_layer
                .insert(layer_id, weights.clone());
            return Ok(weights);
        }

        let norm_name = format!("model.layers.{layer_id}.post_attention_layernorm.weight");
        let gate_name = format!("model.layers.{layer_id}.mlp.gate_proj.weight");
        let up_name = format!("model.layers.{layer_id}.mlp.up_proj.weight");
        let down_name = format!("model.layers.{layer_id}.mlp.down_proj.weight");

        let norm_shape = [NUMERIC_PROGRESS_HIDDEN_DIM];
        let gate_shape = scheduler_dense_bf16_shape(catalog, &gate_name)?;
        let up_shape = scheduler_dense_bf16_shape(catalog, &up_name)?;
        if gate_shape.len() != 2 || up_shape.len() != 2 {
            anyhow::bail!(
                "scheduler real dense MLP layer {layer_id} gate/up tensors must be matrices"
            );
        }
        if gate_shape != up_shape {
            anyhow::bail!(
                "scheduler real dense MLP layer {layer_id} gate/up shape mismatch: gate={gate_shape:?} up={up_shape:?}"
            );
        }
        let intermediate_dim = gate_shape[0];
        let hidden_dim = gate_shape[1];
        if hidden_dim != NUMERIC_PROGRESS_HIDDEN_DIM {
            anyhow::bail!(
                "scheduler real dense MLP layer {layer_id} hidden width mismatch: expected {} got {}",
                NUMERIC_PROGRESS_HIDDEN_DIM,
                hidden_dim
            );
        }
        let down_shape = [NUMERIC_PROGRESS_HIDDEN_DIM, intermediate_dim];

        let norm_bytes = self.preload_real_dense_mlp_resident_weight(
            catalog,
            &norm_name,
            &norm_shape,
            "scheduler real dense post-attention norm pinned staging",
        )?;
        let gate_bytes = self.preload_real_dense_mlp_resident_weight(
            catalog,
            &gate_name,
            &gate_shape,
            "scheduler real dense gate pinned staging",
        )?;
        let up_bytes = self.preload_real_dense_mlp_resident_weight(
            catalog,
            &up_name,
            &up_shape,
            "scheduler real dense up pinned staging",
        )?;
        let down_bytes = self.preload_real_dense_mlp_resident_weight(
            catalog,
            &down_name,
            &down_shape,
            "scheduler real dense down pinned staging",
        )?;

        let weights = SchedulerDenseMlpResidentWeights {
            weight_tensors: vec![
                (norm_name.clone(), norm_bytes),
                (gate_name.clone(), gate_bytes),
                (up_name.clone(), up_bytes),
                (down_name.clone(), down_bytes),
            ],
            norm_name,
            gate_name,
            up_name,
            down_name,
            intermediate_dim,
        };
        scheduler_dense_mlp_resident_weight_cache()
            .lock()
            .map_err(|_| anyhow::anyhow!("scheduler dense MLP resident weight cache poisoned"))?
            .insert(cache_key, weights.clone());
        self.record_real_dense_mlp_resident_weight_usage(&weights);
        self.device_real_dense_mlp_resident_weights_by_layer
            .insert(layer_id, weights.clone());
        Ok(weights)
    }

    fn preload_real_dense_mlp_resident_weight(
        &mut self,
        catalog: &TensorCatalog,
        tensor_name: &str,
        expected_shape: &[usize],
        label: &'static str,
    ) -> Result<u64> {
        let info = scheduler_dense_bf16_tensor_info(catalog, tensor_name)?;
        validate_scheduler_dense_bf16_tensor(info, expected_shape)?;
        let byte_length = info.byte_length;
        let expected_bytes = scheduler_dense_shape_bytes(expected_shape)?;
        preload_resident_weight_from_host_staging(tensor_name, expected_bytes, label, |staging| {
            let summary =
                read_tensor_bytes_into(catalog, tensor_name, staging).with_context(|| {
                    format!("reading scheduler dense tensor {tensor_name} into pinned staging")
                })?;
            if summary.dtype != DType::Bf16 {
                anyhow::bail!(
                    "scheduler dense tensor {tensor_name} expects BF16, got {:?}",
                    summary.dtype
                );
            }
            if summary.shape != expected_shape {
                anyhow::bail!(
                    "scheduler dense tensor {tensor_name} shape mismatch: expected {:?} got {:?}",
                    expected_shape,
                    summary.shape
                );
            }
            if summary.bytes_read as usize != expected_bytes {
                anyhow::bail!(
                    "scheduler dense tensor {tensor_name} read {} bytes, expected {}",
                    summary.bytes_read,
                    expected_bytes
                );
            }
            Ok(())
        })
        .with_context(|| {
            format!("preloading scheduler dense tensor {tensor_name} into resident CUDA buffer")
        })?;
        Ok(byte_length)
    }

    fn record_real_dense_mlp_resident_weight_usage(
        &mut self,
        weights: &SchedulerDenseMlpResidentWeights,
    ) {
        if self.live_request {
            return;
        }
        for (tensor_name, byte_length) in &weights.weight_tensors {
            if self
                .device_real_dense_mlp_resident_weight_names
                .insert(tensor_name.clone())
            {
                self.device_real_dense_mlp_weight_tensors += 1;
                self.device_real_dense_mlp_weight_bytes += *byte_length;
            }
        }
    }

    fn device_real_sparse_routed_mlp_delta_from_normalized(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        source: &RowSource,
        graph_bucket: GraphBucket,
        placement_version: &glmrt_core::PlacementVersion,
        normalized_device: &DeviceBf16Output,
        row_count: usize,
    ) -> Result<DeviceBf16Output> {
        let mut normalized_readback_scratch =
            std::mem::take(&mut self.device_sparse_routed_normalized_readback_bf16_scratch);
        let result = self.device_real_sparse_routed_mlp_delta_from_normalized_with_scratch(
            catalog,
            layer_id,
            source,
            graph_bucket,
            placement_version,
            normalized_device,
            row_count,
            &mut normalized_readback_scratch,
        );
        self.device_sparse_routed_normalized_readback_bf16_scratch = normalized_readback_scratch;
        result
    }

    fn device_real_sparse_routed_mlp_delta_from_normalized_with_scratch(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        source: &RowSource,
        graph_bucket: GraphBucket,
        placement_version: &glmrt_core::PlacementVersion,
        normalized_device: &DeviceBf16Output,
        row_count: usize,
        normalized_readback_scratch: &mut Vec<u8>,
    ) -> Result<DeviceBf16Output> {
        if row_count == 0 {
            anyhow::bail!("scheduler real sparse routed MLP delta requires nonzero rows");
        }
        let row_stride_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
            .checked_mul(std::mem::size_of::<u16>())
            .context("scheduler sparse routed hidden row stride overflow")?;
        let expected_bytes = row_count
            .checked_mul(row_stride_bytes)
            .context("scheduler sparse routed hidden byte count overflow")?;
        if normalized_device.rows != row_count
            || normalized_device.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM
        {
            anyhow::bail!(
                "scheduler sparse routed normalized device hidden shape mismatch for layer {layer_id}: expected {}x{} got {}x{}",
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                normalized_device.rows,
                normalized_device.values_per_row
            );
        }
        let normalized_buffer = normalized_device.buffer();
        if normalized_buffer.ptr.is_null() || normalized_buffer.bytes < expected_bytes {
            anyhow::bail!(
                "scheduler sparse routed normalized device hidden buffer mismatch for layer {layer_id}: bytes={} expected at least {expected_bytes}",
                normalized_buffer.bytes
            );
        }
        let normalized_bf16_for_router =
            Self::sparse_routed_normalized_host_bf16_for_validation_or_fallback(
                layer_id,
                normalized_device,
                expected_bytes,
                normalized_readback_scratch,
            )?;

        let scoring = score_real_router_routes_bf16_cached_device_input(
            catalog,
            layer_id,
            normalized_device,
            normalized_bf16_for_router,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            GLM52_TOP_K,
            &mut self.device_real_sparse_routed_mlp_router_cache,
        )
        .with_context(|| {
            format!("scoring scheduler sparse routed row batch for layer {layer_id}")
        })?;
        if scoring.row_routes.len() != row_count {
            anyhow::bail!(
                "scheduler sparse routed router row count mismatch for layer {layer_id}: scored {} expected {row_count}",
                scoring.row_routes.len()
            );
        }
        record_backend(
            &mut self.device_real_sparse_routed_mlp_router_backend,
            scoring.router_backend,
            "device-real-sparse-routed-router",
        )?;
        let router_weight_bytes = scoring.router_weight_bytes_read;
        let router_bias_bytes = scoring.router_bias_bytes_read;
        let scored_row_routes = scoring.row_routes;
        let route_count = scored_row_routes.iter().map(Vec::len).sum::<usize>();
        if let Some(tcp_context) = self.sparse_tcp_routed_mlp.as_mut() {
            let dispatch_transport = tcp_context.transport;
            let normalized_bf16 = if let Some(normalized_bf16) = normalized_bf16_for_router {
                normalized_bf16
            } else {
                Self::read_sparse_routed_normalized_host_bf16_into_scratch(
                    layer_id,
                    normalized_device,
                    expected_bytes,
                    normalized_readback_scratch,
                )?
            };
            let batch = scheduler_sparse_source_batch(
                layer_id,
                source,
                graph_bucket,
                placement_version,
                row_count,
            )?;
            let routes = scored_routes_for_scheduler_batch(&batch, &scored_row_routes)?;
            let dispatch = tcp_context
                .dispatch_routed_delta(&batch, &routes, normalized_bf16)
                .with_context(|| {
                    format!(
                        "dispatching scheduler sparse routed ProtocolV2 {} MLP delta for layer {layer_id}",
                        dispatch_transport.label()
                    )
                })?;
            record_backend(
                &mut self.device_real_sparse_routed_mlp_route_backend,
                dispatch_transport.sparse_route_backend(),
                "device-real-sparse-routed-nvfp4-route",
            )?;
            self.device_real_sparse_routed_mlp_routes += route_count;
            self.device_real_sparse_routed_mlp_router_weight_bytes += router_weight_bytes;
            self.device_real_sparse_routed_mlp_router_bias_bytes += router_bias_bytes;
            let router_stats = self.device_real_sparse_routed_mlp_router_cache.stats();
            self.device_real_sparse_routed_mlp_router_cache_entries = router_stats.entries;
            self.device_real_sparse_routed_mlp_router_cache_hits = router_stats.cache_hits;
            let mut output = device_bf16_output_from_f32_values(
                &dispatch.accumulation.values,
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                dispatch_transport.sparse_delta_backend(),
            )
            .with_context(|| {
                format!(
                    "uploading ProtocolV2 {} routed MLP accumulation as scheduler device delta",
                    dispatch_transport.label()
                )
            })?;
            output.backend = dispatch_transport.sparse_delta_backend();
            return Ok(output);
        }
        let mut row_routes = Vec::with_capacity(row_count);
        for scoring_routes in scored_row_routes {
            let mut routes = Vec::with_capacity(scoring_routes.len());
            for route in scoring_routes {
                let intermediate_rows =
                    self.scheduler_routed_intermediate_rows(catalog, layer_id, route.expert_id)?;
                routes.push((route, intermediate_rows));
            }
            row_routes.push(routes);
        }
        let execution =
            execute_nvfp4_route_rows_bf16_accumulated_cached_device_input_device_output(
                catalog,
                layer_id,
                normalized_device,
                normalized_bf16_for_router,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                row_stride_bytes,
                &row_routes,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                &mut self.device_real_sparse_routed_mlp_route_cache,
            )
            .with_context(|| {
                format!(
                    "executing scheduler sparse routed NVFP4 accumulated rows for layer {layer_id}"
                )
            })?;
        record_backend(
            &mut self.device_real_sparse_routed_mlp_route_backend,
            execution.kernel_backend,
            "device-real-sparse-routed-nvfp4-route",
        )?;
        self.device_real_sparse_routed_mlp_routes += route_count;
        self.device_real_sparse_routed_mlp_router_weight_bytes += router_weight_bytes;
        self.device_real_sparse_routed_mlp_router_bias_bytes += router_bias_bytes;
        let route_stats = self.device_real_sparse_routed_mlp_route_cache.stats();
        self.device_real_sparse_routed_mlp_route_cache_cuda_entries =
            route_stats.cuda_projection_entries;
        self.device_real_sparse_routed_mlp_route_cache_cuda_uploads =
            route_stats.cuda_projection_uploads;
        self.device_real_sparse_routed_mlp_route_cache_cuda_hits = route_stats.cuda_cache_hits;
        let router_stats = self.device_real_sparse_routed_mlp_router_cache.stats();
        self.device_real_sparse_routed_mlp_router_cache_entries = router_stats.entries;
        self.device_real_sparse_routed_mlp_router_cache_hits = router_stats.cache_hits;

        Ok(execution.output_device)
    }

    fn sparse_routed_normalized_host_bf16_for_validation_or_fallback<'a>(
        layer_id: usize,
        normalized_device: &DeviceBf16Output,
        expected_bytes: usize,
        scratch: &'a mut Vec<u8>,
    ) -> Result<Option<&'a [u8]>> {
        if coordinator_cuda_reference_kernels_enabled() && !cuda_route_validation_enabled() {
            return Ok(None);
        }
        Ok(Some(
            Self::read_sparse_routed_normalized_host_bf16_into_scratch(
                layer_id,
                normalized_device,
                expected_bytes,
                scratch,
            )?,
        ))
    }

    fn read_sparse_routed_normalized_host_bf16_into_scratch<'a>(
        layer_id: usize,
        normalized_device: &DeviceBf16Output,
        expected_bytes: usize,
        scratch: &'a mut Vec<u8>,
    ) -> Result<&'a [u8]> {
        normalized_device.copy_to_host_bytes_into(scratch).context(
            "reading real sparse scheduler normalized hidden into validation/fallback scratch",
        )?;
        if scratch.len() != expected_bytes {
            anyhow::bail!(
                "scheduler sparse routed normalized host hidden byte mismatch for layer {layer_id}: got {} expected {expected_bytes}",
                scratch.len()
            );
        }
        Ok(scratch.as_slice())
    }

    fn scheduler_routed_intermediate_rows(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        expert_id: usize,
    ) -> Result<usize> {
        if !(GLM52_FIRST_K_DENSE_REPLACE..GLM52_TOTAL_LAYERS_WITH_MTP).contains(&layer_id) {
            anyhow::bail!(
                "scheduler routed MLP expects sparse layer {}..{}, got {layer_id}",
                GLM52_FIRST_K_DENSE_REPLACE,
                GLM52_TOTAL_LAYERS_WITH_MTP
            );
        }
        if expert_id >= GLM52_ROUTED_EXPERTS {
            anyhow::bail!(
                "scheduler routed MLP expert id {expert_id} exceeds routed expert count {GLM52_ROUTED_EXPERTS}"
            );
        }
        let key = (layer_id, expert_id);
        if let Some(intermediate_rows) = self
            .device_real_sparse_routed_mlp_intermediate_rows
            .get(&key)
        {
            return Ok(*intermediate_rows);
        }
        let gate_name = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight");
        let info = catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == gate_name)
            .with_context(|| {
                format!("scheduler routed MLP gate projection {gate_name} missing from catalog")
            })?;
        if info.dtype != DType::U8 {
            anyhow::bail!(
                "scheduler routed MLP gate projection {gate_name} expects packed U8 NVFP4, got {:?}",
                info.dtype
            );
        }
        if info.shape.len() != 2 {
            anyhow::bail!(
                "scheduler routed MLP gate projection {gate_name} expected rank-2 tensor, got {:?}",
                info.shape
            );
        }
        let intermediate_rows = info.shape[0];
        if intermediate_rows == 0 {
            anyhow::bail!(
                "scheduler routed MLP gate projection {gate_name} has zero intermediate rows"
            );
        }
        if info.shape[1] == 0 {
            anyhow::bail!(
                "scheduler routed MLP gate projection {gate_name} has zero packed row width"
            );
        }
        self.device_real_sparse_routed_mlp_intermediate_rows
            .insert(key, intermediate_rows);
        Ok(intermediate_rows)
    }

    fn device_real_sparse_post_attention_norm_from_hidden(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        hidden_buffer: GlmrtDeviceBuffer,
        row_count: usize,
    ) -> Result<DeviceBf16Output> {
        if row_count == 0 {
            anyhow::bail!("scheduler real sparse post-attention RMSNorm requires nonzero rows");
        }
        let weights = self.ensure_real_sparse_shared_mlp_resident_weights(catalog, layer_id)?;
        let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output(
            &weights.norm_name,
            hidden_buffer,
            row_count,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )
        .with_context(|| {
            format!("running real sparse shared scheduler RMSNorm for layer {layer_id}")
        })?;
        record_backend(
            &mut self.device_real_sparse_shared_mlp_norm_backend,
            normalized.backend,
            "device-real-sparse-shared-mlp-norm",
        )?;
        Ok(normalized)
    }

    fn device_real_sparse_post_attention_norm_from_hidden_async(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        hidden_buffer: GlmrtDeviceBuffer,
        hidden_ready_event: Option<&CoordinatorCudaEvent>,
        row_count: usize,
    ) -> Result<DeviceBf16Output> {
        if row_count == 0 {
            anyhow::bail!("scheduler real sparse post-attention RMSNorm requires nonzero rows");
        }
        let weights = self.ensure_real_sparse_shared_mlp_resident_weights(catalog, layer_id)?;
        let normalized = rmsnorm_hidden_bf16_preloaded_resident_weight_device_input_output_async(
            &weights.norm_name,
            hidden_buffer,
            hidden_ready_event,
            row_count,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            REAL_FULL_DENSE_RMSNORM_EPS,
        )
        .with_context(|| {
            format!(
                "running asynchronous real sparse shared scheduler RMSNorm for layer {layer_id}"
            )
        })?;
        record_backend(
            &mut self.device_real_sparse_shared_mlp_norm_backend,
            normalized.backend,
            "device-real-sparse-shared-mlp-norm",
        )?;
        Ok(normalized)
    }

    fn device_real_sparse_shared_mlp_delta_from_normalized(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
        normalized_buffer: GlmrtDeviceBuffer,
        row_count: usize,
    ) -> Result<DeviceBf16Output> {
        if row_count == 0 {
            anyhow::bail!("scheduler real sparse shared MLP delta requires nonzero rows");
        }
        let weights = self.ensure_real_sparse_shared_mlp_resident_weights(catalog, layer_id)?;
        silu_gated_mlp_rows_bf16_preloaded_gate_up_down_resident_weight_device_input_device_output_only(
                &weights.gate_name,
                &weights.up_name,
                &weights.down_name,
                normalized_buffer,
                row_count,
                NUMERIC_PROGRESS_HIDDEN_DIM,
                weights.intermediate_dim,
                weights.intermediate_dim,
                NUMERIC_PROGRESS_HIDDEN_DIM,
            )
            .with_context(|| {
                format!("running real sparse shared scheduler MLP for layer {layer_id}")
            })
    }

    fn ensure_real_sparse_shared_mlp_resident_weights(
        &mut self,
        catalog: &TensorCatalog,
        layer_id: usize,
    ) -> Result<SchedulerSparseSharedMlpResidentWeights> {
        if !(GLM52_FIRST_K_DENSE_REPLACE..GLM52_TOTAL_LAYERS_WITH_MTP).contains(&layer_id) {
            anyhow::bail!(
                "scheduler real sparse shared MLP only supports layers {}..{}, got {layer_id}",
                GLM52_FIRST_K_DENSE_REPLACE,
                GLM52_TOTAL_LAYERS_WITH_MTP
            );
        }
        if let Some(weights) = self
            .device_real_sparse_shared_mlp_resident_weights_by_layer
            .get(&layer_id)
        {
            return Ok(weights.clone());
        }
        let cache_key = scheduler_mlp_layer_cache_key(catalog, layer_id);
        if let Some(weights) = scheduler_sparse_shared_mlp_resident_weight_cache()
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("scheduler sparse shared MLP resident weight cache poisoned")
            })?
            .get(&cache_key)
            .cloned()
        {
            self.record_real_sparse_shared_mlp_resident_weight_usage(&weights);
            self.device_real_sparse_shared_mlp_resident_weights_by_layer
                .insert(layer_id, weights.clone());
            return Ok(weights);
        }

        let norm_name = format!("model.layers.{layer_id}.post_attention_layernorm.weight");
        let gate_name = format!("model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight");
        let up_name = format!("model.layers.{layer_id}.mlp.shared_experts.up_proj.weight");
        let down_name = format!("model.layers.{layer_id}.mlp.shared_experts.down_proj.weight");

        let norm_shape = [NUMERIC_PROGRESS_HIDDEN_DIM];
        let gate_shape = scheduler_dense_bf16_shape(catalog, &gate_name)?;
        let up_shape = scheduler_dense_bf16_shape(catalog, &up_name)?;
        if gate_shape.len() != 2 || up_shape.len() != 2 {
            anyhow::bail!(
                "scheduler real sparse shared MLP layer {layer_id} gate/up tensors must be matrices"
            );
        }
        if gate_shape != up_shape {
            anyhow::bail!(
                "scheduler real sparse shared MLP layer {layer_id} gate/up shape mismatch: gate={gate_shape:?} up={up_shape:?}"
            );
        }
        let intermediate_dim = gate_shape[0];
        let hidden_dim = gate_shape[1];
        if hidden_dim != NUMERIC_PROGRESS_HIDDEN_DIM {
            anyhow::bail!(
                "scheduler real sparse shared MLP layer {layer_id} hidden width mismatch: expected {} got {}",
                NUMERIC_PROGRESS_HIDDEN_DIM,
                hidden_dim
            );
        }
        let down_shape = [NUMERIC_PROGRESS_HIDDEN_DIM, intermediate_dim];

        let norm_bytes = self.preload_real_sparse_shared_mlp_resident_weight(
            catalog,
            &norm_name,
            &norm_shape,
            "scheduler real sparse shared post-attention norm pinned staging",
        )?;
        let gate_bytes = self.preload_real_sparse_shared_mlp_resident_weight(
            catalog,
            &gate_name,
            &gate_shape,
            "scheduler real sparse shared gate pinned staging",
        )?;
        let up_bytes = self.preload_real_sparse_shared_mlp_resident_weight(
            catalog,
            &up_name,
            &up_shape,
            "scheduler real sparse shared up pinned staging",
        )?;
        let down_bytes = self.preload_real_sparse_shared_mlp_resident_weight(
            catalog,
            &down_name,
            &down_shape,
            "scheduler real sparse shared down pinned staging",
        )?;

        let weights = SchedulerSparseSharedMlpResidentWeights {
            weight_tensors: vec![
                (norm_name.clone(), norm_bytes),
                (gate_name.clone(), gate_bytes),
                (up_name.clone(), up_bytes),
                (down_name.clone(), down_bytes),
            ],
            norm_name,
            gate_name,
            up_name,
            down_name,
            intermediate_dim,
        };
        scheduler_sparse_shared_mlp_resident_weight_cache()
            .lock()
            .map_err(|_| {
                anyhow::anyhow!("scheduler sparse shared MLP resident weight cache poisoned")
            })?
            .insert(cache_key, weights.clone());
        self.record_real_sparse_shared_mlp_resident_weight_usage(&weights);
        self.device_real_sparse_shared_mlp_resident_weights_by_layer
            .insert(layer_id, weights.clone());
        Ok(weights)
    }

    fn preload_real_sparse_shared_mlp_resident_weight(
        &mut self,
        catalog: &TensorCatalog,
        tensor_name: &str,
        expected_shape: &[usize],
        label: &'static str,
    ) -> Result<u64> {
        let info = scheduler_dense_bf16_tensor_info(catalog, tensor_name)?;
        validate_scheduler_dense_bf16_tensor(info, expected_shape)?;
        let byte_length = info.byte_length;
        let expected_bytes = scheduler_dense_shape_bytes(expected_shape)?;
        preload_resident_weight_from_host_staging(tensor_name, expected_bytes, label, |staging| {
            let summary =
                read_tensor_bytes_into(catalog, tensor_name, staging).with_context(|| {
                    format!(
                        "reading scheduler sparse shared tensor {tensor_name} into pinned staging"
                    )
                })?;
            if summary.dtype != DType::Bf16 {
                anyhow::bail!(
                    "scheduler sparse shared tensor {tensor_name} expects BF16, got {:?}",
                    summary.dtype
                );
            }
            if summary.shape != expected_shape {
                anyhow::bail!(
                    "scheduler sparse shared tensor {tensor_name} shape mismatch: expected {:?} got {:?}",
                    expected_shape,
                    summary.shape
                );
            }
            if summary.bytes_read as usize != expected_bytes {
                anyhow::bail!(
                    "scheduler sparse shared tensor {tensor_name} read {} bytes, expected {}",
                    summary.bytes_read,
                    expected_bytes
                );
            }
            Ok(())
        })
        .with_context(|| {
            format!(
                "preloading scheduler sparse shared tensor {tensor_name} into resident CUDA buffer"
            )
        })?;
        Ok(byte_length)
    }

    fn record_real_sparse_shared_mlp_resident_weight_usage(
        &mut self,
        weights: &SchedulerSparseSharedMlpResidentWeights,
    ) {
        if self.live_request {
            return;
        }
        for (tensor_name, byte_length) in &weights.weight_tensors {
            if self
                .device_real_sparse_shared_mlp_resident_weight_names
                .insert(tensor_name.clone())
            {
                self.device_real_sparse_shared_mlp_weight_tensors += 1;
                self.device_real_sparse_shared_mlp_weight_bytes += *byte_length;
            }
        }
    }

    fn device_mlp_delta_from_hidden(
        &mut self,
        hidden_buffer: GlmrtDeviceBuffer,
        row_count: usize,
    ) -> Result<DeviceBf16Output> {
        if row_count == 0 {
            anyhow::bail!("scheduler device MLP delta requires nonzero rows");
        }
        let (gate_weight, up_weight, down_weight) = {
            let weights = self.scheduler_mlp_resident_weights()?;
            (
                weights.gate_weight.buffer(),
                weights.up_weight.buffer(),
                weights.down_weight.buffer(),
            )
        };
        let bytes = row_count
            .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("scheduler device MLP delta byte count overflows usize")?;
        let library = cuda_native_library()?;
        let mut output = library
            .alloc_device_buffer(bytes)
            .context("allocating scheduler device MLP delta output")?;
        if let Err(error) = library.cuda_scheduler_mlp_delta_bf16(
            hidden_buffer,
            gate_weight,
            up_weight,
            down_weight,
            output,
            row_count,
            NUMERIC_PROGRESS_HIDDEN_DIM,
        ) {
            let _ = library.free_device_buffer(&mut output);
            return Err(error).context("executing scheduler hidden-dependent MLP delta kernel");
        }
        device_bf16_output_from_owned_device_buffer(
            library,
            output,
            row_count,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            SCHEDULER_MLP_DELTA_BACKEND,
            "scheduler hidden-dependent MLP delta",
        )
    }

    fn scheduler_mlp_resident_weights(&mut self) -> Result<&SchedulerMlpResidentWeights> {
        if self.device_mlp_weights.is_none() {
            let gate_weight = self
                .upload_scheduler_mlp_resident_weight("gate", "scheduler MLP resident gate vector")
                .context("uploading scheduler MLP resident gate vector")?;
            let up_weight = self
                .upload_scheduler_mlp_resident_weight("up", "scheduler MLP resident up vector")
                .context("uploading scheduler MLP resident up vector")?;
            let down_weight = self
                .upload_scheduler_mlp_resident_weight("down", "scheduler MLP resident down vector")
                .context("uploading scheduler MLP resident down vector")?;
            self.device_mlp_weight_uploads += 3;
            self.device_mlp_weight_resident_values += NUMERIC_PROGRESS_HIDDEN_DIM * 3;
            self.device_mlp_weights = Some(SchedulerMlpResidentWeights {
                gate_weight,
                up_weight,
                down_weight,
            });
        }
        self.device_mlp_weights
            .as_ref()
            .context("scheduler MLP resident weights missing after upload")
    }

    fn upload_scheduler_mlp_resident_weight(
        &mut self,
        kind: &str,
        label: &'static str,
    ) -> Result<DeviceBf16Output> {
        fill_scheduler_mlp_weight_bf16(kind, &mut self.device_mlp_weight_upload_bf16_scratch);
        device_bf16_output_from_bf16_bytes(
            &self.device_mlp_weight_upload_bf16_scratch,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            label,
        )
    }

    fn account_device_hidden_segments(&self) -> Result<DeviceHiddenSegmentSummary> {
        if !coordinator_cuda_reference_kernels_enabled() {
            return Ok(DeviceHiddenSegmentSummary::default());
        }
        let mut resident_values = 0_usize;
        for (key, segment) in &self.device_hidden_segments {
            let expected_bytes = key.byte_end.checked_sub(key.byte_start).context(
                "scheduler resident hidden segment accounting byte range underflows usize",
            )?;
            let actual_bytes = segment
                .rows
                .checked_mul(segment.values_per_row)
                .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
                .context(
                    "scheduler resident hidden segment accounting byte count overflows usize",
                )?;
            if actual_bytes != expected_bytes {
                anyhow::bail!(
                    "scheduler resident hidden segment byte length mismatch: expected {} got {}",
                    expected_bytes,
                    actual_bytes
                );
            }
            resident_values += segment
                .rows
                .checked_mul(segment.values_per_row)
                .context("scheduler resident hidden segment value count overflows usize")?;
        }
        Ok(DeviceHiddenSegmentSummary {
            resident_segments: self.device_hidden_segments.len(),
            resident_values,
            final_checksum: 0.0,
            expected_final_checksum: 0.0,
        })
    }
}

#[derive(Clone, Copy)]
enum ResidualDeltaStage {
    Attention,
    Mlp,
}

struct DeviceHiddenSegmentResidualAdd {
    backend: &'static str,
    delta_backend: &'static str,
    values_updated: usize,
    device_prefix_rows: usize,
    device_prefix_values: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DeviceDeltaTemplateKey {
    rows: usize,
    delta_bits: u16,
}

#[derive(Clone, Copy)]
struct DeviceDeltaTemplateView {
    buffer: GlmrtDeviceBuffer,
    rows: usize,
    values_per_row: usize,
}

struct SchedulerMlpResidentWeights {
    gate_weight: DeviceBf16Output,
    up_weight: DeviceBf16Output,
    down_weight: DeviceBf16Output,
}

#[derive(Clone)]
struct SchedulerDenseMlpResidentWeights {
    weight_tensors: Vec<(String, u64)>,
    norm_name: String,
    gate_name: String,
    up_name: String,
    down_name: String,
    intermediate_dim: usize,
}

#[derive(Clone)]
struct SchedulerSparseSharedMlpResidentWeights {
    weight_tensors: Vec<(String, u64)>,
    norm_name: String,
    gate_name: String,
    up_name: String,
    down_name: String,
    intermediate_dim: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
struct SchedulerMlpLayerCacheKey {
    model_id: String,
    snapshot_path: String,
    layer_id: usize,
}

static SCHEDULER_DENSE_MLP_RESIDENT_WEIGHT_CACHE: OnceLock<
    Mutex<BTreeMap<SchedulerMlpLayerCacheKey, SchedulerDenseMlpResidentWeights>>,
> = OnceLock::new();
static SCHEDULER_SPARSE_SHARED_MLP_RESIDENT_WEIGHT_CACHE: OnceLock<
    Mutex<BTreeMap<SchedulerMlpLayerCacheKey, SchedulerSparseSharedMlpResidentWeights>>,
> = OnceLock::new();

fn scheduler_mlp_layer_cache_key(
    catalog: &TensorCatalog,
    layer_id: usize,
) -> SchedulerMlpLayerCacheKey {
    SchedulerMlpLayerCacheKey {
        model_id: catalog.model_id.clone(),
        snapshot_path: catalog.snapshot_path.clone(),
        layer_id,
    }
}

fn scheduler_dense_mlp_resident_weight_cache(
) -> &'static Mutex<BTreeMap<SchedulerMlpLayerCacheKey, SchedulerDenseMlpResidentWeights>> {
    SCHEDULER_DENSE_MLP_RESIDENT_WEIGHT_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn scheduler_sparse_shared_mlp_resident_weight_cache(
) -> &'static Mutex<BTreeMap<SchedulerMlpLayerCacheKey, SchedulerSparseSharedMlpResidentWeights>> {
    SCHEDULER_SPARSE_SHARED_MLP_RESIDENT_WEIGHT_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()))
}

struct SchedulerSparseTcpPreparedSegment {
    byte_start: usize,
    byte_end: usize,
    batch_row_start: usize,
    row_count: usize,
    kind: RowSourceKind,
    token_start: u64,
    normalized: DeviceBf16Output,
}

pub(super) struct SchedulerSparseTcpPreparedDispatch {
    layer_id: usize,
    batch: ExpertBatch,
    prepared_segments: Vec<SchedulerSparseTcpPreparedSegment>,
    routes: Vec<ExpertBatchRoute>,
    hidden_payload: Vec<u8>,
    route_count: usize,
    router_weight_bytes: u64,
    router_bias_bytes: u64,
    stage_timing_enabled: bool,
    stage_total_start: Option<Instant>,
    attention_delta_ms: f64,
    norm_ms: f64,
    normalized_readback_ms: f64,
    router_ms: f64,
    routes_ms: f64,
}

pub(super) struct SchedulerSparseTcpCohortPendingDispatch {
    handle: SchedulerSparseTcpPayloadDispatchHandle,
    member_row_counts: Vec<usize>,
    total_rows: usize,
}

struct SchedulerSparseTcpReadySegment {
    byte_start: usize,
    byte_end: usize,
    batch_row_start: usize,
    row_count: usize,
    kind: RowSourceKind,
    token_start: u64,
    shared_delta: DeviceBf16Output,
}

struct SchedulerSparseRollingChunk {
    task_index: usize,
    global_row_start: usize,
    batch: ExpertBatch,
    routes: Vec<ExpertBatchRoute>,
    hidden_payload: Vec<u8>,
    ready_segments: Vec<SchedulerSparseTcpReadySegment>,
    finalized_segments: Vec<bool>,
    task_completed: bool,
}

impl SchedulerSparseRollingChunk {
    fn row_count(&self) -> usize {
        self.batch.num_rows()
    }

    fn global_row_end(&self) -> usize {
        self.global_row_start + self.row_count()
    }
}

struct SchedulerSparseRollingQueuedEmission {
    emission: RollingExpertRowPackEmission,
    batch: ExpertBatch,
    routes: Vec<ExpertBatchRoute>,
    hidden_payload: Vec<u8>,
}

struct SchedulerSparseRollingPendingEmission {
    emission: RollingExpertRowPackEmission,
    handle: SchedulerSparseTcpPayloadDispatchHandle,
    completed_dispatch_row_slices: Vec<Vec<usize>>,
}

enum SchedulerSparseRollingAccumulatorPageState {
    Pending,
    Active(CudaStreamedSparseBAccumulator),
    Finalized,
}

struct SchedulerSparseRollingAccumulatorPage {
    row_start: usize,
    row_count: usize,
    finalized_rows: usize,
    state: SchedulerSparseRollingAccumulatorPageState,
}

impl SchedulerSparseRollingAccumulatorPage {
    fn row_end(&self) -> usize {
        self.row_start + self.row_count
    }
}

struct SchedulerSparseRollingAccumulatorOwnedChunk {
    partial_output: Vec<u8>,
    local_row_indices: Vec<usize>,
    completed_local_rows: Vec<usize>,
    output_dtype: ExpertV2Dtype,
    output_row_stride_bytes: usize,
}

/// Keeps only streamed expert rows that have not yet been fused into the residual.
///
/// The original rolling path allocated `[total_context_rows, hidden_dim]` FP32 for every
/// concurrently active layer so global row IDs could be scattered directly. These logical pages
/// retain the same stable host ordering, but remap each response to its source segment and return
/// the device plane to the coordinator buffer pool immediately after a bounded 2K-row reclaim
/// page is finalized. Storage therefore scales with the live response/finalization window rather
/// than multiplying the full context plane by every concurrently active layer.
struct SchedulerSparseRollingAccumulator {
    total_rows: usize,
    row_width: usize,
    pages: Vec<SchedulerSparseRollingAccumulatorPage>,
    registered_rows: usize,
    finalized_rows: usize,
    active_pages: usize,
    active_rows: usize,
    peak_active_pages: usize,
    peak_active_rows: usize,
}

impl SchedulerSparseRollingAccumulator {
    fn new(total_rows: usize, row_width: usize) -> Result<Self> {
        anyhow::ensure!(
            total_rows > 0 && row_width > 0,
            "rolling Sparse-B paged accumulator requires a non-empty destination"
        );
        Ok(Self {
            total_rows,
            row_width,
            pages: Vec::new(),
            registered_rows: 0,
            finalized_rows: 0,
            active_pages: 0,
            active_rows: 0,
            peak_active_pages: 0,
            peak_active_rows: 0,
        })
    }

    fn register_segment(&mut self, row_start: usize, row_count: usize) -> Result<()> {
        anyhow::ensure!(
            row_count > 0,
            "rolling Sparse-B cannot register an empty accumulator segment"
        );
        let expected_row_start = self.registered_rows;
        anyhow::ensure!(
            row_start == expected_row_start,
            "rolling Sparse-B accumulator segment starts at {row_start}, expected {expected_row_start}"
        );
        let row_end = row_start
            .checked_add(row_count)
            .context("rolling Sparse-B accumulator segment range overflows usize")?;
        anyhow::ensure!(
            row_end <= self.total_rows,
            "rolling Sparse-B accumulator segment {row_start}..{row_end} exceeds total rows {}",
            self.total_rows
        );
        let page_start =
            row_start / ROLLING_SPARSE_ACCUMULATOR_PAGE_ROWS * ROLLING_SPARSE_ACCUMULATOR_PAGE_ROWS;
        let page_end = page_start
            .checked_add(ROLLING_SPARSE_ACCUMULATOR_PAGE_ROWS)
            .context("rolling Sparse-B accumulator page range overflows usize")?
            .min(self.total_rows);
        anyhow::ensure!(
            row_end <= page_end,
            "rolling Sparse-B segment {row_start}..{row_end} crosses reclaim page {page_start}..{page_end}"
        );
        if self.pages.last().map(|page| page.row_start) != Some(page_start) {
            anyhow::ensure!(
                row_start == page_start,
                "rolling Sparse-B reclaim page {page_start} was first registered at row {row_start}"
            );
            self.pages.push(SchedulerSparseRollingAccumulatorPage {
                row_start: page_start,
                row_count: page_end - page_start,
                finalized_rows: 0,
                state: SchedulerSparseRollingAccumulatorPageState::Pending,
            });
        }
        self.registered_rows = row_end;
        Ok(())
    }

    fn page_index_for_row(&self, global_row: usize) -> Result<usize> {
        anyhow::ensure!(
            global_row < self.registered_rows,
            "rolling Sparse-B response row {global_row} exceeds {} registered rows",
            self.registered_rows
        );
        let page_index = self
            .pages
            .partition_point(|page| page.row_start <= global_row)
            .checked_sub(1)
            .with_context(|| {
                format!(
                    "rolling Sparse-B response row {global_row} has no registered accumulator segment"
                )
            })?;
        let page = &self.pages[page_index];
        anyhow::ensure!(
            global_row < page.row_end(),
            "rolling Sparse-B response row {global_row} is outside registered segment {}..{}",
            page.row_start,
            page.row_end()
        );
        Ok(page_index)
    }

    fn page_index_for_segment(&self, row_start: usize, row_count: usize) -> Result<usize> {
        let page_index = self.page_index_for_row(row_start)?;
        let page = &self.pages[page_index];
        let row_end = row_start
            .checked_add(row_count)
            .context("rolling Sparse-B residual segment range overflows usize")?;
        anyhow::ensure!(
            row_count > 0 && row_end <= page.row_end() && row_end <= self.registered_rows,
            "rolling Sparse-B residual segment {row_start}..{row_end} is outside registered reclaim page {}..{}",
            page.row_start,
            page.row_end()
        );
        Ok(page_index)
    }

    fn activate_page(&mut self, page_index: usize) -> Result<()> {
        let page = self
            .pages
            .get(page_index)
            .context("rolling Sparse-B accumulator page index is out of range")?;
        match page.state {
            SchedulerSparseRollingAccumulatorPageState::Active(_) => return Ok(()),
            SchedulerSparseRollingAccumulatorPageState::Finalized => {
                anyhow::bail!(
                    "rolling Sparse-B received another contribution for finalized segment {}+{}",
                    page.row_start,
                    page.row_count
                );
            }
            SchedulerSparseRollingAccumulatorPageState::Pending => {}
        }
        let page_rows = page.row_count;
        let accumulator = CudaStreamedSparseBAccumulator::new(page_rows, self.row_width)
            .context("allocating rolling Sparse-B segment accumulator")?;
        self.pages[page_index].state =
            SchedulerSparseRollingAccumulatorPageState::Active(accumulator);
        self.active_pages += 1;
        self.active_rows = self
            .active_rows
            .checked_add(page_rows)
            .context("rolling Sparse-B active row count overflow")?;
        self.peak_active_pages = self.peak_active_pages.max(self.active_pages);
        self.peak_active_rows = self.peak_active_rows.max(self.active_rows);
        Ok(())
    }

    fn push_chunks(&mut self, chunks: &[StreamedSparseBAccumulatorChunk<'_>]) -> Result<()> {
        let first = chunks
            .first()
            .context("rolling Sparse-B response batch is empty")?;
        let mut chunks_by_page = std::iter::repeat_with(Vec::new)
            .take(self.pages.len())
            .collect::<Vec<Vec<SchedulerSparseRollingAccumulatorOwnedChunk>>>();
        for chunk in chunks {
            anyhow::ensure!(
                chunk.output_dtype == first.output_dtype
                    && chunk.output_row_stride_bytes == first.output_row_stride_bytes,
                "rolling Sparse-B response metadata changed within a coalesced batch"
            );
            anyhow::ensure!(
                chunk.output_row_stride_bytes > 0,
                "rolling Sparse-B response row stride must be non-zero"
            );
            let expected_bytes = chunk
                .global_row_indices
                .len()
                .checked_mul(chunk.output_row_stride_bytes)
                .context("rolling Sparse-B response byte count overflows usize")?;
            anyhow::ensure!(
                chunk.partial_output.len() == expected_bytes,
                "rolling Sparse-B response bytes {} did not match expected {expected_bytes}",
                chunk.partial_output.len()
            );
            let completed_rows = chunk
                .completed_global_rows
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            anyhow::ensure!(
                completed_rows.len() == chunk.completed_global_rows.len(),
                "rolling Sparse-B response repeats a completed global row"
            );
            anyhow::ensure!(
                completed_rows
                    .iter()
                    .all(|row| chunk.global_row_indices.contains(row)),
                "rolling Sparse-B completion row was not present in its response chunk"
            );

            let mut owned_by_page =
                BTreeMap::<usize, SchedulerSparseRollingAccumulatorOwnedChunk>::new();
            for (payload_row, global_row) in chunk.global_row_indices.iter().copied().enumerate() {
                let page_index = self.page_index_for_row(global_row)?;
                let page = &self.pages[page_index];
                anyhow::ensure!(
                    !matches!(
                        page.state,
                        SchedulerSparseRollingAccumulatorPageState::Finalized
                    ),
                    "rolling Sparse-B received another contribution for finalized row {global_row}"
                );
                let byte_start = payload_row * chunk.output_row_stride_bytes;
                let byte_end = byte_start + chunk.output_row_stride_bytes;
                let owned = owned_by_page.entry(page_index).or_insert_with(|| {
                    SchedulerSparseRollingAccumulatorOwnedChunk {
                        partial_output: Vec::new(),
                        local_row_indices: Vec::new(),
                        completed_local_rows: Vec::new(),
                        output_dtype: chunk.output_dtype,
                        output_row_stride_bytes: chunk.output_row_stride_bytes,
                    }
                });
                owned
                    .partial_output
                    .extend_from_slice(&chunk.partial_output[byte_start..byte_end]);
                let local_row = global_row - page.row_start;
                owned.local_row_indices.push(local_row);
                if completed_rows.contains(&global_row) {
                    owned.completed_local_rows.push(local_row);
                }
            }
            for (page_index, owned) in owned_by_page {
                chunks_by_page[page_index].push(owned);
            }
        }

        for (page_index, owned_chunks) in chunks_by_page.into_iter().enumerate() {
            if owned_chunks.is_empty() {
                continue;
            }
            self.activate_page(page_index)?;
            let mut seen_rows = BTreeSet::new();
            let has_overlapping_rows = owned_chunks.iter().any(|chunk| {
                chunk
                    .local_row_indices
                    .iter()
                    .any(|row| !seen_rows.insert(*row))
            });
            let views = owned_chunks
                .iter()
                .map(|chunk| StreamedSparseBAccumulatorChunk {
                    partial_output: &chunk.partial_output,
                    global_row_indices: &chunk.local_row_indices,
                    completed_global_rows: &chunk.completed_local_rows,
                    output_dtype: chunk.output_dtype,
                    output_row_stride_bytes: chunk.output_row_stride_bytes,
                })
                .collect::<Vec<_>>();
            let accumulator = match &mut self.pages[page_index].state {
                SchedulerSparseRollingAccumulatorPageState::Active(accumulator) => accumulator,
                _ => unreachable!("rolling Sparse-B page was activated before response push"),
            };
            if has_overlapping_rows {
                accumulator.push_host_ordered_chunks(&views)?;
            } else {
                accumulator.push_chunks(&views)?;
            }
        }
        Ok(())
    }

    fn segment_ready(&self, row_start: usize, row_count: usize) -> Result<bool> {
        anyhow::ensure!(
            row_count > 0,
            "rolling Sparse-B readiness query requires at least one row"
        );
        let page_index = self.page_index_for_row(row_start)?;
        let page = &self.pages[page_index];
        let row_end = row_start
            .checked_add(row_count)
            .context("rolling Sparse-B readiness row range overflows usize")?;
        anyhow::ensure!(
            row_end <= page.row_end(),
            "rolling Sparse-B readiness range {row_start}..{row_end} crosses registered segment {}..{}",
            page.row_start,
            page.row_end()
        );
        match &page.state {
            SchedulerSparseRollingAccumulatorPageState::Pending => Ok(false),
            SchedulerSparseRollingAccumulatorPageState::Active(accumulator) => {
                accumulator.segment_ready(row_start - page.row_start, row_count)
            }
            SchedulerSparseRollingAccumulatorPageState::Finalized => Ok(true),
        }
    }

    fn finalize_segment(
        &mut self,
        segment: &StreamedSparseBResidualSegment<'_>,
    ) -> Result<DeviceBf16Output> {
        let page_index = self.page_index_for_segment(segment.row_start, segment.row_count)?;
        let page_row_start = self.pages[page_index].row_start;
        let local_segment = StreamedSparseBResidualSegment {
            residual: segment.residual,
            shared_delta: segment.shared_delta,
            row_start: segment.row_start - page_row_start,
            row_count: segment.row_count,
        };
        let (output, release_page) = {
            let page = &mut self.pages[page_index];
            let output = match &mut page.state {
                SchedulerSparseRollingAccumulatorPageState::Active(accumulator) => {
                    accumulator.finalize_segment(&local_segment)?
                }
                SchedulerSparseRollingAccumulatorPageState::Pending => {
                    anyhow::bail!(
                        "rolling Sparse-B segment {}+{} has no accumulated responses",
                        segment.row_start,
                        segment.row_count
                    );
                }
                SchedulerSparseRollingAccumulatorPageState::Finalized => {
                    anyhow::bail!(
                        "rolling Sparse-B segment {}+{} was finalized more than once",
                        segment.row_start,
                        segment.row_count
                    );
                }
            };
            page.finalized_rows = page
                .finalized_rows
                .checked_add(segment.row_count)
                .context("rolling Sparse-B page finalized row count overflow")?;
            anyhow::ensure!(
                page.finalized_rows <= page.row_count,
                "rolling Sparse-B reclaim page finalized too many rows"
            );
            let release_page = page.finalized_rows == page.row_count;
            if release_page {
                match &page.state {
                    SchedulerSparseRollingAccumulatorPageState::Active(accumulator) => {
                        accumulator.validate_complete()?;
                    }
                    _ => unreachable!("rolling Sparse-B page was active during finalization"),
                }
            }
            (output, release_page)
        };
        if release_page {
            let page_rows = self.pages[page_index].row_count;
            self.pages[page_index].state = SchedulerSparseRollingAccumulatorPageState::Finalized;
            self.active_pages -= 1;
            self.active_rows -= page_rows;
        }
        self.finalized_rows = self
            .finalized_rows
            .checked_add(segment.row_count)
            .context("rolling Sparse-B finalized row count overflow")?;
        Ok(output)
    }

    fn validate_complete(&self) -> Result<()> {
        anyhow::ensure!(
            self.registered_rows == self.total_rows,
            "rolling Sparse-B registered {} of {} rows",
            self.registered_rows,
            self.total_rows
        );
        anyhow::ensure!(
            self.finalized_rows == self.total_rows
                && self.active_pages == 0
                && self.active_rows == 0
                && self.pages.iter().all(|page| matches!(
                    page.state,
                    SchedulerSparseRollingAccumulatorPageState::Finalized
                )),
            "rolling Sparse-B paged accumulator did not finalize every row"
        );
        Ok(())
    }

    fn peak_active_pages(&self) -> usize {
        self.peak_active_pages
    }

    fn peak_active_rows(&self) -> usize {
        self.peak_active_rows
    }

    fn active_pages(&self) -> usize {
        self.active_pages
    }

    fn active_rows(&self) -> usize {
        self.active_rows
    }
}

pub(super) struct SchedulerSparseRollingLayerApply {
    layer_id: usize,
    total_rows: usize,
    planner: RollingExpertRowPackAccumulator,
    accumulator: SchedulerSparseRollingAccumulator,
    chunks: Vec<SchedulerSparseRollingChunk>,
    queued_emissions: VecDeque<SchedulerSparseRollingQueuedEmission>,
    pending_emissions: VecDeque<SchedulerSparseRollingPendingEmission>,
    admitted_rows: usize,
    emitted_rows: usize,
    emitted_packs: usize,
    finalized_rows: usize,
    input_finished: bool,
    completion_validated: bool,
    route_count: usize,
    router_weight_bytes: u64,
    router_bias_bytes: u64,
    stage_timing_enabled: bool,
    stage_started: Option<Instant>,
    attention_delta_ms: f64,
    norm_ms: f64,
    shared_mlp_ms: f64,
    normalized_readback_ms: f64,
    router_ms: f64,
    routes_ms: f64,
    planner_ms: f64,
    sparse_b_ms: f64,
    apply_ms: f64,
    deadline_emissions: usize,
    max_selected_row_offset: usize,
}

impl SchedulerSparseRollingLayerApply {
    pub(super) fn new(layer_id: usize, total_rows: usize) -> Result<Self> {
        anyhow::ensure!(
            rolling_sparse_packs_supported_for_rows(total_rows),
            "rolling sparse layer rows {total_rows} are outside the enabled range {}..={}",
            ROLLING_SPARSE_LOOKAHEAD_ROWS,
            ROLLING_SPARSE_MAX_ROWS
        );
        Ok(Self {
            layer_id,
            total_rows,
            planner: RollingExpertRowPackAccumulator::new(RollingExpertRowPackConfig {
                logical_chunk_rows: ROLLING_SPARSE_OLDEST_ROWS,
                max_pack_rows: ROLLING_SPARSE_PACK_ROWS,
                lookahead_rows: ROLLING_SPARSE_LOOKAHEAD_ROWS,
                expert_tile_rows: ROLLING_SPARSE_EXPERT_TILE_ROWS,
                selection_quantum_rows: ROLLING_SPARSE_SELECTION_ROWS,
            })
            .map_err(anyhow::Error::new)
            .context("creating rolling sparse row-pack accumulator")?,
            accumulator: SchedulerSparseRollingAccumulator::new(
                total_rows,
                NUMERIC_PROGRESS_HIDDEN_DIM,
            )?,
            chunks: Vec::new(),
            queued_emissions: VecDeque::new(),
            pending_emissions: VecDeque::new(),
            admitted_rows: 0,
            emitted_rows: 0,
            emitted_packs: 0,
            finalized_rows: 0,
            input_finished: false,
            completion_validated: false,
            route_count: 0,
            router_weight_bytes: 0,
            router_bias_bytes: 0,
            stage_timing_enabled: false,
            stage_started: sparse_tcp_stage_timing_enabled().then(Instant::now),
            attention_delta_ms: 0.0,
            norm_ms: 0.0,
            shared_mlp_ms: 0.0,
            normalized_readback_ms: 0.0,
            router_ms: 0.0,
            routes_ms: 0.0,
            planner_ms: 0.0,
            sparse_b_ms: 0.0,
            apply_ms: 0.0,
            deadline_emissions: 0,
            max_selected_row_offset: 0,
        })
    }

    pub(super) fn active_dispatches(&self) -> usize {
        self.pending_emissions.len()
    }

    pub(super) fn queued_dispatches(&self) -> usize {
        self.queued_emissions.len()
    }

    pub(super) fn buffered_dispatches(&self) -> usize {
        self.active_dispatches() + self.queued_dispatches()
    }

    pub(super) fn is_complete(&self) -> bool {
        self.completion_validated
    }

    pub(super) fn accumulator_peak_pages(&self) -> usize {
        self.accumulator.peak_active_pages()
    }

    pub(super) fn accumulator_peak_rows(&self) -> usize {
        self.accumulator.peak_active_rows()
    }

    pub(super) fn accumulator_active_pages(&self) -> usize {
        self.accumulator.active_pages()
    }

    pub(super) fn accumulator_active_rows(&self) -> usize {
        self.accumulator.active_rows()
    }
}

pub(super) struct SchedulerSparseRollingProgress {
    pub(super) completed_task_indices: Vec<usize>,
    pub(super) made_progress: bool,
    pub(super) layer_complete: bool,
}

pub(super) struct SchedulerSparseRollingPushTiming {
    pub(super) shared_mlp_ms: f64,
    pub(super) planner_ms: f64,
}

pub(super) struct SchedulerSparseTcpPendingApply {
    layer_id: usize,
    batch: ExpertBatch,
    ready_segments: Vec<SchedulerSparseTcpReadySegment>,
    pending_dispatch: Option<SchedulerSparseTcpPayloadDispatchHandle>,
    pending_routes: Option<Vec<ExpertBatchRoute>>,
    pending_hidden_payload: Option<Vec<u8>>,
    completed_dispatch: Option<TcpProtocolV2HostBatchSetBf16PayloadDispatch>,
    route_count: usize,
    router_weight_bytes: u64,
    router_bias_bytes: u64,
    stage_timing_enabled: bool,
    stage_total_start: Option<Instant>,
    dispatch_start: Option<Instant>,
    attention_delta_ms: f64,
    norm_ms: f64,
    shared_mlp_ms: f64,
    normalized_readback_ms: f64,
    router_ms: f64,
    routes_ms: f64,
    incremental_stream: Option<SchedulerSparseTcpIncrementalStream>,
    incremental_complete: bool,
}

impl SchedulerSparseTcpPendingApply {
    pub(super) fn supports_incremental_stream(&self) -> bool {
        self.pending_dispatch
            .as_ref()
            .is_some_and(SchedulerSparseTcpPayloadDispatchHandle::has_streaming_response_chunks)
            && self.batch.num_rows() >= MIN_INCREMENTAL_SPARSE_B_ROWS
            && !self.ready_segments.is_empty()
            && spark_expert_reduction_dispatch_for_rows(self.batch.num_rows())
                .ok()
                .flatten()
                .is_some()
    }

    pub(super) fn attach_completed_cohort_dispatch(
        &mut self,
        dispatch: TcpProtocolV2HostBatchSetBf16PayloadDispatch,
    ) -> Result<()> {
        anyhow::ensure!(
            self.pending_dispatch.is_none() && self.completed_dispatch.is_none(),
            "scheduler sparse cohort member already carries a dispatch"
        );
        self.pending_routes = None;
        self.pending_hidden_payload = None;
        self.completed_dispatch = Some(dispatch);
        Ok(())
    }
}

struct SchedulerSparseTcpIncrementalStream {
    accumulator: CudaStreamedSparseBAccumulator,
    finalized_segments: Vec<bool>,
    completed_global_row_slices: Vec<Vec<usize>>,
    sparse_b_ms: f64,
    apply_ms: f64,
    stream_started: Option<Instant>,
}

pub(super) struct SchedulerSparseTcpApplyProgress {
    pub(super) completed_segment_indices: Vec<usize>,
    pub(super) dispatch_complete: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DeviceHiddenSegmentKey {
    byte_start: usize,
    byte_end: usize,
}

fn smallest_containing_device_hidden_segment_key<T>(
    segments: &BTreeMap<DeviceHiddenSegmentKey, T>,
    desired: DeviceHiddenSegmentKey,
) -> Option<DeviceHiddenSegmentKey> {
    segments
        .keys()
        .copied()
        .filter(|key| key.byte_start <= desired.byte_start && key.byte_end >= desired.byte_end)
        .min_by_key(|key| key.byte_end - key.byte_start)
}

#[derive(Default)]
struct DeviceHiddenSegmentSummary {
    resident_segments: usize,
    resident_values: usize,
    final_checksum: f32,
    expected_final_checksum: f32,
}

fn should_use_real_dense_scheduler_mlp(layer_id: usize) -> bool {
    layer_id < GLM52_FIRST_K_DENSE_REPLACE
}

fn should_use_real_sparse_shared_scheduler_mlp(layer_id: usize, kind: RowSourceKind) -> bool {
    layer_id >= GLM52_FIRST_K_DENSE_REPLACE
        && kind != RowSourceKind::Benchmark
        && !synthetic_sparse_spark_expert_mode_for_layer(layer_id)
}

fn validate_sparse_routed_normalized_device_hidden(
    layer_id: usize,
    normalized_device: &DeviceBf16Output,
    row_count: usize,
    expected_bytes: usize,
) -> Result<()> {
    if normalized_device.rows != row_count
        || normalized_device.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM
    {
        anyhow::bail!(
            "scheduler sparse routed normalized device hidden shape mismatch for layer {layer_id}: expected {}x{} got {}x{}",
            row_count,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            normalized_device.rows,
            normalized_device.values_per_row
        );
    }
    let normalized_buffer = normalized_device.buffer();
    if normalized_buffer.ptr.is_null() || normalized_buffer.bytes < expected_bytes {
        anyhow::bail!(
            "scheduler sparse routed normalized device hidden buffer mismatch for layer {layer_id}: bytes={} expected at least {expected_bytes}",
            normalized_buffer.bytes
        );
    }
    Ok(())
}

fn scheduler_sparse_source_batch(
    layer_id: usize,
    source: &RowSource,
    graph_bucket: GraphBucket,
    placement_version: &glmrt_core::PlacementVersion,
    row_count: usize,
) -> Result<ExpertBatch> {
    anyhow::ensure!(
        row_count == source.row_count,
        "scheduler sparse source batch row count {} did not match source rows {}",
        row_count,
        source.row_count
    );
    anyhow::ensure!(
        row_count <= graph_bucket.row_capacity,
        "scheduler sparse source batch rows {} exceed graph bucket {}",
        row_count,
        graph_bucket.row_capacity
    );
    let rows = (0..row_count)
        .map(|row_offset| glmrt_core::ExpertBatchRow {
            row_id: row_offset as u64,
            source_kind: source.kind,
            request_id: source.request_id.clone(),
            sequence_id: source.sequence_id.clone(),
            token_position: glmrt_core::PositionId(source.token_start.0 + row_offset as u64),
            route_offset: row_offset * GLM52_TOP_K,
            route_count: GLM52_TOP_K,
        })
        .collect::<Vec<_>>();
    Ok(ExpertBatch {
        layer_id: LayerId(layer_id as u32),
        placement_version: placement_version.clone(),
        hidden_dim: NUMERIC_PROGRESS_HIDDEN_DIM,
        hidden_bytes_per_row: NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>(),
        hidden_dtype: DType::Bf16,
        graph_bucket,
        quantization_recipe: ModelFacts::default().quantization_recipe,
        rows,
    })
}

fn routed_delta_backend_available(backend: Option<&'static str>) -> bool {
    backend
        .map(|backend| {
            backend.contains("nvfp4-route-bf16-accumulated-device-output")
                || backend == SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_DELTA_BACKEND
                || backend == SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_DELTA_BACKEND
        })
        .unwrap_or(false)
}

fn routed_route_backend_available(backend: Option<&'static str>) -> bool {
    backend
        .map(|backend| {
            backend.contains("nvfp4-route-bf16-accumulated-device-input")
                || backend == SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_ROUTE_BACKEND
                || backend == SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_ROUTE_BACKEND
        })
        .unwrap_or(false)
}

fn routed_route_cache_available(
    route_backend: Option<&'static str>,
    cuda_entries: usize,
    cuda_uploads: usize,
) -> bool {
    if matches!(
        route_backend,
        Some(SCHEDULER_REAL_SPARSE_ROUTED_MLP_TCP_ROUTE_BACKEND)
            | Some(SCHEDULER_REAL_SPARSE_ROUTED_MLP_VERBS_HOST_ROUTE_BACKEND)
    ) {
        cuda_entries == 0 && cuda_uploads == 0
    } else {
        cuda_entries > 0 && cuda_uploads > 0
    }
}

fn scheduler_dense_bf16_shape(catalog: &TensorCatalog, tensor_name: &str) -> Result<Vec<usize>> {
    Ok(scheduler_dense_bf16_tensor_info(catalog, tensor_name)?
        .shape
        .clone())
}

fn scheduler_dense_bf16_tensor_info<'a>(
    catalog: &'a TensorCatalog,
    tensor_name: &str,
) -> Result<&'a TensorInfo> {
    catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == tensor_name)
        .with_context(|| format!("scheduler dense tensor {tensor_name} missing from catalog"))
}

fn validate_scheduler_dense_bf16_tensor(
    tensor: &TensorInfo,
    expected_shape: &[usize],
) -> Result<()> {
    if !matches!(tensor.dtype, DType::Bf16 | DType::F8E4M3) {
        anyhow::bail!(
            "scheduler dense tensor {} expects BF16 or startup-expanded block-FP8, got {:?}",
            tensor.name,
            tensor.dtype
        );
    }
    if tensor.shape != expected_shape {
        anyhow::bail!(
            "scheduler dense tensor {} shape mismatch: expected {:?} got {:?}",
            tensor.name,
            expected_shape,
            tensor.shape
        );
    }
    let expected_bytes = scheduler_dense_shape_bytes(expected_shape)?;
    let expected_source_bytes = match tensor.dtype {
        DType::Bf16 => expected_bytes,
        DType::F8E4M3 => {
            anyhow::ensure!(
                resident_weight_is_preloaded(&tensor.name, expected_bytes),
                "scheduler dense block-FP8 tensor {} has no startup-expanded BF16 resident weight",
                tensor.name
            );
            expected_bytes / std::mem::size_of::<u16>()
        }
        _ => unreachable!(),
    };
    if tensor.byte_length as usize != expected_source_bytes {
        anyhow::bail!(
            "scheduler dense tensor {} source byte length mismatch: expected {} got {}",
            tensor.name,
            expected_source_bytes,
            tensor.byte_length
        );
    }
    Ok(())
}

fn scheduler_dense_shape_bytes(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1_usize, |acc, dim| {
            acc.checked_mul(*dim)
                .context("scheduler dense tensor shape product overflows usize")
        })?
        .checked_mul(std::mem::size_of::<u16>())
        .context("scheduler dense tensor byte length overflows usize")
}

fn apply_device_hidden_segment_residual_add(
    device_hidden_segments: &mut BTreeMap<DeviceHiddenSegmentKey, DeviceBf16Output>,
    residual_bf16: &[u8],
    byte_start: usize,
    byte_end: usize,
    delta_bf16: &[u8],
    device_delta_prefix: Option<&RealFullSchedulerDeviceAttentionDelta>,
    device_delta_template: Option<DeviceDeltaTemplateView>,
    device_delta_direct: Option<&DeviceBf16Output>,
) -> Result<Option<DeviceHiddenSegmentResidualAdd>> {
    if !coordinator_cuda_reference_kernels_enabled() {
        return Ok(None);
    }
    let segment_bytes = byte_end
        .checked_sub(byte_start)
        .context("scheduler device hidden segment byte range underflow")?;
    validate_device_hidden_segment_shape(segment_bytes, delta_bf16)?;
    let values_updated = segment_bytes / std::mem::size_of::<u16>();
    let rows = values_updated / NUMERIC_PROGRESS_HIDDEN_DIM;
    let key = DeviceHiddenSegmentKey {
        byte_start,
        byte_end,
    };
    if !device_hidden_segments.contains_key(&key) {
        let initial = device_bf16_output_from_bf16_bytes(
            &residual_bf16[byte_start..byte_end],
            rows,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler numeric resident hidden segment",
        )?;
        device_hidden_segments.insert(key, initial);
    }
    if device_delta_direct.is_some()
        && (device_delta_prefix.is_some() || device_delta_template.is_some())
    {
        anyhow::bail!(
            "scheduler direct device delta cannot be combined with prefix/template delta"
        );
    }
    let (output, delta_backend) = {
        let residual = device_hidden_segments
            .get(&key)
            .context("scheduler resident hidden segment missing after initialization")?;
        if let Some(device_delta_direct) = device_delta_direct {
            validate_direct_device_delta(device_delta_direct, rows)?;
            (
                residual_add_bf16_device_inputs_device_output(residual, device_delta_direct)?,
                device_delta_direct.backend,
            )
        } else {
            let delta = match (device_delta_prefix, device_delta_template) {
                (Some(device_delta_prefix), Some(device_delta_template)) => {
                    validate_device_delta_template_view(device_delta_template, rows)?;
                    device_bf16_output_from_device_template_with_device_row_prefix(
                        device_delta_template.buffer,
                        rows,
                        NUMERIC_PROGRESS_HIDDEN_DIM,
                        &device_delta_prefix.output_device,
                        device_delta_prefix.output_device_row_offset,
                        device_delta_prefix.values_per_row,
                        "scheduler numeric resident attention delta segment",
                    )?
                }
                (None, Some(device_delta_template)) => {
                    validate_device_delta_template_view(device_delta_template, rows)?;
                    device_bf16_output_from_device_template_buffer(
                        device_delta_template.buffer,
                        rows,
                        NUMERIC_PROGRESS_HIDDEN_DIM,
                        "scheduler numeric resident delta segment",
                    )?
                }
                (Some(device_delta_prefix), None) => {
                    device_bf16_output_from_bf16_bytes_with_device_row_prefix(
                        delta_bf16,
                        rows,
                        NUMERIC_PROGRESS_HIDDEN_DIM,
                        &device_delta_prefix.output_device,
                        device_delta_prefix.output_device_row_offset,
                        device_delta_prefix.values_per_row,
                        "scheduler numeric resident attention delta segment",
                    )?
                }
                (None, None) => device_bf16_output_from_bf16_bytes(
                    delta_bf16,
                    rows,
                    NUMERIC_PROGRESS_HIDDEN_DIM,
                    "scheduler numeric resident delta segment",
                )?,
            };
            let delta_backend = delta.backend;
            (
                residual_add_bf16_device_inputs_device_output(residual, &delta)?,
                delta_backend,
            )
        }
    };
    let backend = output.backend;
    device_hidden_segments.insert(key, output);
    Ok(Some(DeviceHiddenSegmentResidualAdd {
        backend,
        delta_backend,
        values_updated,
        device_prefix_rows: device_delta_prefix.map_or(0, |prefix| prefix.row_count),
        device_prefix_values: device_delta_prefix
            .map(|prefix| prefix.row_count * prefix.values_per_row),
    }))
}

fn validate_device_delta_template_view(
    template: DeviceDeltaTemplateView,
    rows: usize,
) -> Result<()> {
    if template.rows != rows || template.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM {
        anyhow::bail!(
            "scheduler device delta template shape mismatch: expected rows={} width={}, got rows={} width={}",
            rows,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            template.rows,
            template.values_per_row
        );
    }
    Ok(())
}

fn validate_direct_device_delta(delta: &DeviceBf16Output, rows: usize) -> Result<()> {
    if delta.rows != rows || delta.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM {
        anyhow::bail!(
            "scheduler direct device delta shape mismatch: expected rows={} width={}, got rows={} width={}",
            rows,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            delta.rows,
            delta.values_per_row
        );
    }
    Ok(())
}

fn scheduler_mlp_delta_values(
    label: &'static str,
    row_count: usize,
    delta: &DeviceBf16Output,
) -> Result<usize> {
    validate_direct_device_delta(delta, row_count)?;
    row_count
        .checked_mul(NUMERIC_PROGRESS_HIDDEN_DIM)
        .with_context(|| format!("scheduler {label} MLP delta value count overflow"))
}

fn row_count_from_delta_byte_range(byte_start: usize, byte_end: usize) -> Result<usize> {
    let bytes = byte_end
        .checked_sub(byte_start)
        .context("scheduler delta byte range underflows usize")?;
    let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM
        .checked_mul(std::mem::size_of::<u16>())
        .context("scheduler delta row byte count overflows usize")?;
    if bytes == 0 || bytes % row_bytes != 0 {
        anyhow::bail!(
            "scheduler delta byte range {byte_start}..{byte_end} has {bytes} bytes, not a nonzero multiple of row bytes {row_bytes}"
        );
    }
    Ok(bytes / row_bytes)
}

fn validate_device_hidden_segment_shape(residual_bytes: usize, delta_bf16: &[u8]) -> Result<()> {
    if residual_bytes == 0 {
        anyhow::bail!("scheduler device hidden segment residual-add requires non-empty input");
    }
    if residual_bytes != delta_bf16.len() {
        anyhow::bail!(
            "scheduler device hidden segment residual-add byte length mismatch: residual={} delta={}",
            residual_bytes,
            delta_bf16.len()
        );
    }
    if residual_bytes % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "scheduler device hidden segment residual-add byte length must be even, got {}",
            residual_bytes
        );
    }
    let values_updated = residual_bytes / std::mem::size_of::<u16>();
    if values_updated % NUMERIC_PROGRESS_HIDDEN_DIM != 0 {
        anyhow::bail!(
            "scheduler device hidden segment residual-add values {values_updated} are not a multiple of hidden dim {NUMERIC_PROGRESS_HIDDEN_DIM}"
        );
    }
    Ok(())
}

fn overlay_device_attention_output_delta(
    delta_bf16: &mut [u8],
    row_count: usize,
    kind: RowSourceKind,
    attention_delta: &RealFullSchedulerDeviceAttentionDelta,
) -> Result<()> {
    validate_device_attention_output_delta(row_count, kind, attention_delta)?;
    let hidden_row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
    let expected_delta_bytes = row_count
        .checked_mul(hidden_row_bytes)
        .context("scheduler attention hidden delta byte count overflow")?;
    if delta_bf16.len() != expected_delta_bytes {
        anyhow::bail!(
            "scheduler attention hidden delta byte count mismatch: expected {expected_delta_bytes} got {}",
            delta_bf16.len()
        );
    }
    let output_row_bytes = attention_delta
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("scheduler attention output delta row byte count overflow")?;
    let output_bf16 = attention_delta
        .output_bf16
        .as_ref()
        .context("scheduler compact attention delta overlay requires host output bytes")?;
    for row in 0..row_count {
        let dst_start = row
            .checked_mul(hidden_row_bytes)
            .context("scheduler attention hidden row offset overflow")?;
        let src_start = row
            .checked_mul(output_row_bytes)
            .context("scheduler attention output row offset overflow")?;
        delta_bf16[dst_start..dst_start + output_row_bytes]
            .copy_from_slice(&output_bf16[src_start..src_start + output_row_bytes]);
    }
    Ok(())
}

fn validate_device_attention_output_delta(
    row_count: usize,
    kind: RowSourceKind,
    attention_delta: &RealFullSchedulerDeviceAttentionDelta,
) -> Result<()> {
    if attention_delta.kind != kind || attention_delta.row_count != row_count {
        anyhow::bail!(
            "scheduler attention output delta source mismatch: expected kind={kind:?} rows={row_count}, got kind={:?} rows={}",
            attention_delta.kind,
            attention_delta.row_count
        );
    }
    if attention_delta.values_per_row == 0
        || attention_delta.values_per_row > NUMERIC_PROGRESS_HIDDEN_DIM
    {
        anyhow::bail!(
            "scheduler attention output delta width {} outside 1..={NUMERIC_PROGRESS_HIDDEN_DIM}",
            attention_delta.values_per_row
        );
    }
    let output_row_bytes = attention_delta
        .values_per_row
        .checked_mul(std::mem::size_of::<u16>())
        .context("scheduler attention output delta row byte count overflow")?;
    let expected_output_bytes = row_count
        .checked_mul(output_row_bytes)
        .context("scheduler attention output delta byte count overflow")?;
    if let Some(output_bf16) = attention_delta.output_bf16.as_ref() {
        if output_bf16.len() != expected_output_bytes {
            anyhow::bail!(
                "scheduler attention output delta byte count mismatch: expected {expected_output_bytes} got {}",
                output_bf16.len()
            );
        }
    } else if attention_delta.values_per_row != NUMERIC_PROGRESS_HIDDEN_DIM {
        anyhow::bail!(
            "scheduler compact attention output delta requires host bytes: values_per_row={} hidden_dim={NUMERIC_PROGRESS_HIDDEN_DIM}",
            attention_delta.values_per_row
        );
    }
    if attention_delta.output_device.rows < attention_delta.output_device_row_offset
        || attention_delta
            .output_device_row_offset
            .checked_add(row_count)
            .context("scheduler attention output device row range overflow")?
            > attention_delta.output_device.rows
        || attention_delta.output_device.values_per_row != attention_delta.values_per_row
    {
        anyhow::bail!(
            "scheduler attention output device shape mismatch: row_offset={} rows={} values_per_row={} output_device={}x{}",
            attention_delta.output_device_row_offset,
            row_count,
            attention_delta.values_per_row,
            attention_delta.output_device.rows,
            attention_delta.output_device.values_per_row
        );
    }
    Ok(())
}

fn numeric_progression_deltas(kind: RowSourceKind) -> (f32, f32) {
    match kind {
        RowSourceKind::PrefillChunk => (1.0, 0.5),
        RowSourceKind::DecodeStep => (0.5, 0.25),
        RowSourceKind::MtpVerifyBlock => (1.5, 0.75),
        RowSourceKind::Benchmark => (0.0, 0.0),
    }
}

fn fill_scheduler_mlp_weight_bf16(kind: &str, output: &mut Vec<u8>) {
    output.clear();
    output.reserve(NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>());
    for col in 0..NUMERIC_PROGRESS_HIDDEN_DIM {
        let value = match kind {
            "gate" => {
                if col % 2 == 0 {
                    0.0625
                } else {
                    -0.046875
                }
            }
            "up" => 0.109375 + ((col % 5) as f32 - 2.0) / 1024.0,
            "down" => 0.25 + ((col % 7) as f32 - 3.0) / 2048.0,
            _ => 0.0,
        };
        output.extend_from_slice(&bf16_bits(value).to_le_bytes());
    }
}

#[cfg(test)]
fn checksum_bf16(bytes: &[u8]) -> Result<f32> {
    if bytes.len() % std::mem::size_of::<u16>() != 0 {
        anyhow::bail!(
            "scheduler BF16 checksum requires even byte length, got {}",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(bf16_chunk_to_f32)
        .map(|value| value as f64)
        .sum::<f64>() as f32)
}

fn fill_repeated_bf16_bytes(value: f32, output: &mut [u8]) {
    let bytes = bf16_bits(value).to_le_bytes();
    for chunk in output.chunks_exact_mut(std::mem::size_of::<u16>()) {
        chunk.copy_from_slice(&bytes);
    }
}

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

#[cfg(test)]
fn bf16_chunk_to_f32(chunk: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
    f32::from_bits((bits as u32) << 16)
}

fn record_backend(
    slot: &mut Option<&'static str>,
    backend: &'static str,
    stage: &'static str,
) -> Result<()> {
    match slot {
        Some(existing) if *existing != backend => {
            anyhow::bail!(
                "numeric scheduler {stage} residual-add backend changed from {existing} to {backend}"
            );
        }
        Some(_) => {}
        None => *slot = Some(backend),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::real_full::coordinator_kernels::cuda_reference_kernels_test_override;
    use glmrt_core::{
        DType, ExpertHostBatchSetAccumulation, ModelFacts, PlacementVersion, PositionId, RequestId,
        RowSourceKind, TensorInfo, TensorRole,
    };
    use glmrt_transport::{
        ExpertProtocolV2FrameBuffer, ExpertProtocolV2Request, ExpertProtocolV2RequestView,
        ProtocolV2ExpertExecutor, SyntheticRouteExecutor,
        TcpProtocolV2HostBatchSetBf16PayloadDispatch, TcpProtocolV2HostBatchSetDispatch,
        TcpProtocolV2HostBatchSetDispatchStats, EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN,
    };
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    fn test_progression() -> RealFullSchedulerNumericProgression {
        RealFullSchedulerNumericProgression::new(RealFullSchedulerNumericProgressionShape {
            prefix_tokens: 0,
            prefill_rows: 1,
            prefill_chunk_tokens: 1,
            decode_rows: 1,
            mtp_rows: 1,
            mtp_accepted_rows: 1,
            source_segments_per_layer: 3,
            sparse_source_segments_per_layer: 3,
        })
    }

    #[test]
    fn default_sparse_tcp_row_cap_covers_default_prefill_chunk() {
        assert!(DEFAULT_SCHEDULER_TCP_MAX_GLOBAL_ROWS >= 512 + 1 + 4);
    }

    #[test]
    fn device_hidden_subrange_selects_smallest_resident_parent() {
        let outer = DeviceHiddenSegmentKey {
            byte_start: 0,
            byte_end: 4_096,
        };
        let inner = DeviceHiddenSegmentKey {
            byte_start: 1_024,
            byte_end: 3_072,
        };
        let desired = DeviceHiddenSegmentKey {
            byte_start: 1_536,
            byte_end: 2_048,
        };
        let mut segments = BTreeMap::new();
        segments.insert(outer, ());
        segments.insert(inner, ());

        assert_eq!(
            smallest_containing_device_hidden_segment_key(&segments, desired),
            Some(inner)
        );
        assert_eq!(
            smallest_containing_device_hidden_segment_key(
                &segments,
                DeviceHiddenSegmentKey {
                    byte_start: 4_096,
                    byte_end: 4_608,
                },
            ),
            None
        );
    }

    #[test]
    fn device_hidden_subrange_copies_resident_parent_instead_of_zero_host_fallback() -> Result<()> {
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        let mut parent_bytes = vec![0_u8; 2 * row_bytes];
        fill_repeated_bf16_bytes(0.25, &mut parent_bytes[..row_bytes]);
        fill_repeated_bf16_bytes(-0.5, &mut parent_bytes[row_bytes..]);
        let parent = match device_bf16_output_from_bf16_bytes(
            &parent_bytes,
            2,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler resident parent slice unit",
        ) {
            Ok(parent) => parent,
            Err(error) => {
                eprintln!("skipped: CUDA device upload unavailable: {error:#}");
                return Ok(());
            }
        };
        let mut progression =
            RealFullSchedulerNumericProgression::new(RealFullSchedulerNumericProgressionShape {
                prefix_tokens: 0,
                prefill_rows: 2,
                prefill_chunk_tokens: 2,
                decode_rows: 0,
                mtp_rows: 0,
                mtp_accepted_rows: 0,
                source_segments_per_layer: 1,
                sparse_source_segments_per_layer: 1,
            });
        progression.device_hidden_segments.insert(
            DeviceHiddenSegmentKey {
                byte_start: 0,
                byte_end: 2 * row_bytes,
            },
            parent,
        );

        let source = progression
            .device_hidden_source(RowSourceKind::PrefillChunk, 1, 1)?
            .context("resident parent slice source unavailable")?;
        assert_eq!(source.rows, 1);
        let child = progression
            .device_hidden_segments
            .get(&DeviceHiddenSegmentKey {
                byte_start: row_bytes,
                byte_end: 2 * row_bytes,
            })
            .context("resident parent slice child was not retained")?;
        assert_eq!(child.copy_to_host_bytes()?, parent_bytes[row_bytes..]);
        Ok(())
    }

    #[test]
    fn rolling_sparse_is_mandatory_through_long_context_ceiling() {
        assert!(!rolling_sparse_packs_supported_for_rows(
            ROLLING_SPARSE_REQUIRED_MIN_ROWS - 1
        ));
        assert!(rolling_sparse_packs_supported_for_rows(
            ROLLING_SPARSE_REQUIRED_MIN_ROWS
        ));
        assert!(rolling_sparse_packs_supported_for_rows(
            ROLLING_SPARSE_MAX_ROWS
        ));
        assert!(!rolling_sparse_packs_supported_for_rows(
            ROLLING_SPARSE_MAX_ROWS + 1
        ));
    }

    #[test]
    fn phase0_spark_synthetic_sparse_mode_matches_sparse_layers_only() {
        assert!(!phase0_spark_expert_mode_is_synthetic_sparse_for_layer(
            "synthetic",
            GLM52_FIRST_K_DENSE_REPLACE - 1
        ));
        assert!(phase0_spark_expert_mode_is_synthetic_sparse_for_layer(
            "SYNTHETIC",
            GLM52_FIRST_K_DENSE_REPLACE
        ));
        assert!(!phase0_spark_expert_mode_is_synthetic_sparse_for_layer(
            "real",
            GLM52_FIRST_K_DENSE_REPLACE
        ));
    }

    #[test]
    fn numeric_progression_row_index_offsets_absolute_tokens_by_prefix_context() -> Result<()> {
        let progression =
            RealFullSchedulerNumericProgression::new(RealFullSchedulerNumericProgressionShape {
                prefix_tokens: 128,
                prefill_rows: 3,
                prefill_chunk_tokens: 3,
                decode_rows: 2,
                mtp_rows: 1,
                mtp_accepted_rows: 1,
                source_segments_per_layer: 5,
                sparse_source_segments_per_layer: 5,
            });

        assert_eq!(
            progression.numeric_progression_row_index(RowSourceKind::PrefillChunk, 128, 0)?,
            0
        );
        assert_eq!(
            progression.numeric_progression_row_index(RowSourceKind::PrefillChunk, 130, 0)?,
            2
        );
        assert_eq!(
            progression.numeric_progression_row_index(RowSourceKind::DecodeStep, 131, 0)?,
            3
        );
        assert_eq!(
            progression.numeric_progression_row_index(RowSourceKind::DecodeStep, 132, 0)?,
            4
        );
        assert_eq!(
            progression.numeric_progression_row_index(RowSourceKind::MtpVerifyBlock, 133, 0)?,
            5
        );
        assert!(progression
            .numeric_progression_row_index(RowSourceKind::DecodeStep, 130, 0)
            .is_err());
        Ok(())
    }

    #[test]
    fn scheduler_sparse_tcp_context_accounts_residual_dispatches() -> Result<()> {
        let target = TcpProtocolV2HostBatchTarget {
            host: "spark-0".to_owned(),
            addr: "127.0.0.1:1".parse()?,
        };
        let mut context =
            RealFullSchedulerSparseTcpRoutedMlpContext::new(1, vec![target], None, 7_000)?;
        let source = RowSource {
            kind: RowSourceKind::DecodeStep,
            request_id: RequestId::from("scheduler-sparse-tcp-context"),
            sequence_id: "scheduler-sparse-tcp-context-sequence".to_owned(),
            token_start: PositionId(11),
            row_count: 1,
        };
        let batch = scheduler_sparse_source_batch(
            GLM52_FIRST_K_DENSE_REPLACE,
            &source,
            GraphBucket::new(1),
            &PlacementVersion::from("scheduler-sparse-tcp-context-placement"),
            1,
        )?;
        let dispatch = TcpProtocolV2HostBatchSetDispatch {
            accumulation: ExpertHostBatchSetAccumulation {
                values: vec![1.0_f32; GLM52_HIDDEN_SIZE],
                contribution_counts: vec![1],
            },
            partial_outputs_bf16_by_host: vec![vec![
                0_u8;
                GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>()
            ]],
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: 1,
                global_rows: 1,
                host_rows: 1,
                routes: GLM52_TOP_K,
                output_dim: GLM52_HIDDEN_SIZE,
                output_values: GLM52_HIDDEN_SIZE,
                request_wire_bytes: 128,
                response_wire_bytes: 96,
                response_executor_ids: vec![context.probe.expected_real_executor_id],
                contribution_counts: vec![1],
                output_checksum: GLM52_HIDDEN_SIZE as f64,
                graph_pool_leases: 0,
                graph_pool_fixed_buffer_bytes: 0,
                graph_pool_active_rows: 0,
                graph_pool_active_routes: 0,
                graph_pool_active_expert_tiles: 0,
                graph_pool_bucket_rows: Vec::new(),
            },
        };

        for _ in 0..(GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE) {
            context.record_dispatch(&batch, &dispatch)?;
        }
        let probe = context.finish();

        assert_eq!(
            probe.status,
            "request-shaped-sparse-tcp-residual-dispatch-passed"
        );
        assert!(probe.passed);
        assert!(probe.all_responses_real_nvfp4);
        assert_eq!(
            probe.sparse_batches,
            GLM52_NUM_HIDDEN_LAYERS - GLM52_FIRST_K_DENSE_REPLACE
        );
        assert_eq!(probe.global_rows, probe.sparse_batches);
        assert_eq!(probe.host_batches, probe.sparse_batches);
        assert_eq!(probe.routes, probe.global_rows * GLM52_TOP_K);
        assert_eq!(probe.output_values, probe.global_rows * GLM52_HIDDEN_SIZE);
        assert_eq!(probe.output_finite_values, 0);
        assert_eq!(probe.output_nonzero_values, 0);
        assert_eq!(probe.output_checksum, 0.0);
        assert_eq!(probe.real_executor_responses, probe.host_batches);
        assert_eq!(probe.non_real_executor_responses, 0);
        Ok(())
    }

    #[test]
    fn scheduler_sparse_dispatch_ids_are_reserved_before_completion() -> Result<()> {
        let target = TcpProtocolV2HostBatchTarget {
            host: "spark-0".to_owned(),
            addr: "127.0.0.1:1".parse()?,
        };
        let mut context =
            RealFullSchedulerSparseTcpRoutedMlpContext::new(2, vec![target], None, 7_000)?;

        let first = context.reserve_dispatch_request_ids()?;
        let second = context.reserve_dispatch_request_ids()?;

        assert_eq!(first, (1, 7_000));
        assert_eq!(second, (2, 7_000 + EXPERT_HOSTS_REQUEST_STRIDE));
        assert_eq!(context.probe.sparse_batches, 0);
        Ok(())
    }

    #[test]
    fn scheduler_sparse_payload_accounting_accepts_spark_reduced_root_payload() -> Result<()> {
        let target = TcpProtocolV2HostBatchTarget {
            host: "spark-0".to_owned(),
            addr: "127.0.0.1:1".parse()?,
        };
        let mut context =
            RealFullSchedulerSparseTcpRoutedMlpContext::new(1, vec![target], None, 7_000)?;
        let batch = SchedulerSparseTcpPayloadDispatchBatchShape {
            layer_id: LayerId(GLM52_FIRST_K_DENSE_REPLACE as u32),
            rows: 1,
            routes: GLM52_TOP_K,
            unique_experts: 0,
            max_expert_load: 0,
            expert_load_square_sum: 0,
            route_profiled: false,
            hidden_dim: GLM52_HIDDEN_SIZE,
        };
        let expected_executor_id = context.probe.expected_real_executor_id;
        let dispatch = TcpProtocolV2HostBatchSetBf16PayloadDispatch {
            partial_outputs_bf16_by_host: vec![vec![
                0_u8;
                GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>()
            ]],
            global_row_indices_by_host: vec![vec![0]],
            completed_global_row_slices: vec![vec![0]],
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: 4,
                global_rows: 1,
                host_rows: 4,
                routes: GLM52_TOP_K,
                output_dim: GLM52_HIDDEN_SIZE,
                output_values: GLM52_HIDDEN_SIZE,
                request_wire_bytes: 4 * GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>(),
                response_wire_bytes: GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>(),
                response_executor_ids: vec![expected_executor_id; 4],
                contribution_counts: vec![1],
                output_checksum: 0.0,
                graph_pool_leases: 0,
                graph_pool_fixed_buffer_bytes: 0,
                graph_pool_active_rows: 0,
                graph_pool_active_routes: 0,
                graph_pool_active_expert_tiles: 0,
                graph_pool_bucket_rows: Vec::new(),
            },
        };

        context.record_payload_dispatch(batch, &dispatch)?;

        assert_eq!(context.probe.sparse_batches, 1);
        assert_eq!(context.probe.host_batches, 4);
        assert_eq!(context.probe.host_rows, 4);
        Ok(())
    }

    #[test]
    fn scheduler_sparse_tcp_payload_partials_split_contiguous_segments() -> Result<()> {
        let row_width = 2_usize;
        let row_bytes = row_width * std::mem::size_of::<u16>();
        let mut host0 = Vec::new();
        host0.extend_from_slice(&vec![10_u8; row_bytes]);
        host0.extend_from_slice(&vec![12_u8; row_bytes]);
        let mut host1 = Vec::new();
        host1.extend_from_slice(&vec![21_u8; row_bytes]);
        host1.extend_from_slice(&vec![22_u8; row_bytes]);
        let dispatch = TcpProtocolV2HostBatchSetBf16PayloadDispatch {
            partial_outputs_bf16_by_host: vec![host0, host1],
            global_row_indices_by_host: vec![vec![0, 2], vec![1, 2]],
            completed_global_row_slices: vec![vec![0, 1], vec![2]],
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: 2,
                global_rows: 3,
                host_rows: 4,
                routes: 4,
                output_dim: row_width,
                output_values: 3 * row_width,
                request_wire_bytes: 128,
                response_wire_bytes: 96,
                response_executor_ids: vec![1, 1],
                contribution_counts: vec![1, 1, 2],
                output_checksum: 0.0,
                graph_pool_leases: 0,
                graph_pool_fixed_buffer_bytes: 0,
                graph_pool_active_rows: 0,
                graph_pool_active_routes: 0,
                graph_pool_active_expert_tiles: 0,
                graph_pool_bucket_rows: Vec::new(),
            },
        };

        let (payloads, row_indices) =
            scheduler_sparse_tcp_payload_partials_for_segment(&dispatch, 1, 2, row_width)?;

        assert_eq!(row_indices, vec![vec![1], vec![0, 1]]);
        assert_eq!(payloads[0], vec![12_u8; row_bytes]);
        let mut expected_host1 = vec![21_u8; row_bytes];
        expected_host1.extend_from_slice(&vec![22_u8; row_bytes]);
        assert_eq!(payloads[1], expected_host1);
        Ok(())
    }

    #[test]
    fn scheduler_sparse_tcp_payload_dispatch_splits_and_rebases_cohort_member() -> Result<()> {
        let row_width = 2_usize;
        let row_bytes = row_width * std::mem::size_of::<u16>();
        let mut host0 = vec![10_u8; row_bytes];
        host0.extend_from_slice(&vec![12_u8; row_bytes]);
        let mut host1 = vec![21_u8; row_bytes];
        host1.extend_from_slice(&vec![22_u8; row_bytes]);
        let dispatch = TcpProtocolV2HostBatchSetBf16PayloadDispatch {
            partial_outputs_bf16_by_host: vec![host0, host1],
            global_row_indices_by_host: vec![vec![0, 2], vec![1, 2]],
            completed_global_row_slices: vec![vec![0, 1], vec![2]],
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: 2,
                global_rows: 3,
                host_rows: 4,
                routes: 24,
                output_dim: row_width,
                output_values: 4 * row_width,
                request_wire_bytes: 128,
                response_wire_bytes: 96,
                response_executor_ids: vec![1, 1],
                contribution_counts: vec![1, 1, 2],
                output_checksum: 0.0,
                graph_pool_leases: 0,
                graph_pool_fixed_buffer_bytes: 0,
                graph_pool_active_rows: 0,
                graph_pool_active_routes: 0,
                graph_pool_active_expert_tiles: 0,
                graph_pool_bucket_rows: Vec::new(),
            },
        };

        let member =
            scheduler_sparse_tcp_payload_dispatch_for_segment(&dispatch, 1, 2, row_width, 16)?;

        assert_eq!(member.global_row_indices_by_host, vec![vec![1], vec![0, 1]]);
        assert_eq!(member.completed_global_row_slices, vec![vec![0], vec![1]]);
        assert_eq!(member.stats.global_rows, 2);
        assert_eq!(member.stats.host_rows, 3);
        assert_eq!(member.stats.routes, 16);
        assert_eq!(member.stats.output_values, 2 * row_width);
        assert_eq!(member.stats.contribution_counts, vec![1, 2]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_sparse_tcp_context_dispatch_inside_tokio_runtime_does_not_nested_block_on(
    ) -> Result<()> {
        let target = TcpProtocolV2HostBatchTarget {
            host: "spark-0".to_owned(),
            addr: "127.0.0.1:1".parse()?,
        };
        let mut context =
            RealFullSchedulerSparseTcpRoutedMlpContext::new(1, vec![target], None, 7_000)?;
        let source = RowSource {
            kind: RowSourceKind::DecodeStep,
            request_id: RequestId::from("scheduler-sparse-tcp-runtime"),
            sequence_id: "scheduler-sparse-tcp-runtime-sequence".to_owned(),
            token_start: PositionId(11),
            row_count: 1,
        };
        let batch = scheduler_sparse_source_batch(
            GLM52_FIRST_K_DENSE_REPLACE,
            &source,
            GraphBucket::new(1),
            &PlacementVersion::from("scheduler-sparse-tcp-runtime-placement"),
            1,
        )?;
        let routes = (0..GLM52_TOP_K)
            .map(|_| ExpertBatchRoute {
                row_index: 0,
                expert_id: 0,
                gate_weight: 1.0 / GLM52_TOP_K as f32,
            })
            .collect::<Vec<_>>();
        let hidden_payload = vec![0_u8; batch.num_rows() * batch.hidden_bytes_per_row];

        let error = context
            .dispatch_routed_delta(&batch, &routes, &hidden_payload)
            .expect_err("unused TCP port should fail after entering helper runtime")
            .to_string();

        assert!(
            !error.contains("Cannot start a runtime from within a runtime"),
            "{error}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scheduler_sparse_tcp_shared_worker_reuses_connections_across_contexts() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0").await?;
        let addr = listener.local_addr()?;
        let accept_count = Arc::new(AtomicUsize::new(0));
        let accept_count_for_server = Arc::clone(&accept_count);
        let server = tokio::spawn(async move {
            loop {
                let (mut stream, _) = listener.accept().await?;
                accept_count_for_server.fetch_add(1, Ordering::SeqCst);
                tokio::spawn(async move {
                    let executor = SyntheticRouteExecutor;
                    let mut response_buffer = ExpertProtocolV2FrameBuffer::new();
                    loop {
                        let mut frame = vec![0_u8; EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN];
                        if stream.read_exact(&mut frame).await.is_err() {
                            return Ok::<(), anyhow::Error>(());
                        }
                        let wire_bytes = ExpertProtocolV2Request::wire_bytes_from_header(&frame)?;
                        frame.resize(wire_bytes, 0);
                        stream
                            .read_exact(&mut frame[EXPERT_PROTOCOL_V2_REQUEST_HEADER_LEN..])
                            .await?;
                        let request = ExpertProtocolV2RequestView::parse(&frame)?;
                        let response = executor.execute_with_identity(&request)?;
                        let frame = response_buffer.encode_response(&response)?;
                        stream.write_all(frame).await?;
                        stream.flush().await?;
                    }
                });
            }
            #[allow(unreachable_code)]
            Ok::<(), anyhow::Error>(())
        });
        let target = TcpProtocolV2HostBatchTarget {
            host: "spark-0".to_owned(),
            addr,
        };
        let owner_lookup = ExpertOwnerLookup::from_pairs([(
            (GLM52_FIRST_K_DENSE_REPLACE, 0),
            "spark-0".to_owned(),
        )]);
        let worker = Arc::new(RealFullSchedulerSparseTcpDispatchWorker::new(
            vec![target],
            Some(owner_lookup),
        )?);
        let source = RowSource {
            kind: RowSourceKind::DecodeStep,
            request_id: RequestId::from("scheduler-sparse-tcp-shared-worker"),
            sequence_id: "scheduler-sparse-tcp-shared-worker-sequence".to_owned(),
            token_start: PositionId(11),
            row_count: 1,
        };
        let batch = scheduler_sparse_source_batch(
            GLM52_FIRST_K_DENSE_REPLACE,
            &source,
            GraphBucket::new(1),
            &PlacementVersion::from("scheduler-sparse-tcp-shared-worker-placement"),
            1,
        )?;
        let routes = (0..GLM52_TOP_K)
            .map(|_| ExpertBatchRoute {
                row_index: 0,
                expert_id: 0,
                gate_weight: 1.0 / GLM52_TOP_K as f32,
            })
            .collect::<Vec<_>>();
        let hidden_payload = vec![0_u8; batch.num_rows() * batch.hidden_bytes_per_row];

        for request_id_base in [10_000_u64, 20_000_u64] {
            let mut context = RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(
                1,
                Arc::clone(&worker),
                request_id_base,
            )?;
            let dispatch = context.dispatch_routed_delta(&batch, &routes, &hidden_payload)?;
            assert_eq!(dispatch.stats.hosts, 1);
            assert_eq!(dispatch.stats.global_rows, 1);
            assert_eq!(dispatch.stats.routes, GLM52_TOP_K);
        }

        let mut context = RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(
            1,
            Arc::clone(&worker),
            25_000,
        )?;
        let payload_dispatch =
            context.dispatch_routed_delta_payload(&batch, &routes, &hidden_payload)?;
        assert_eq!(payload_dispatch.stats.hosts, 1);
        assert_eq!(payload_dispatch.stats.global_rows, 1);
        assert_eq!(payload_dispatch.stats.routes, GLM52_TOP_K);
        assert_eq!(payload_dispatch.global_row_indices_by_host, vec![vec![0]]);
        assert!(payload_dispatch.stats.contribution_counts.is_empty());

        let source = RowSource {
            kind: RowSourceKind::PrefillChunk,
            request_id: RequestId::from("scheduler-sparse-tcp-sliced-payload"),
            sequence_id: "scheduler-sparse-tcp-sliced-payload-sequence".to_owned(),
            token_start: PositionId(11),
            row_count: 2,
        };
        let batch = scheduler_sparse_source_batch(
            GLM52_FIRST_K_DENSE_REPLACE,
            &source,
            GraphBucket::new(2),
            &PlacementVersion::from("scheduler-sparse-tcp-sliced-payload-placement"),
            2,
        )?;
        let routes = batch
            .rows
            .iter()
            .enumerate()
            .flat_map(|(row_index, row)| {
                (0..row.route_count).map(move |_| ExpertBatchRoute {
                    row_index,
                    expert_id: 0,
                    gate_weight: 1.0 / row.route_count as f32,
                })
            })
            .collect::<Vec<_>>();
        let hidden_payload = vec![0_u8; batch.num_rows() * batch.hidden_bytes_per_row];
        let mut context =
            RealFullSchedulerSparseTcpRoutedMlpContext::with_dispatch_worker(1, worker, 30_000)?;
        context.max_global_rows_per_dispatch = 1;
        let dispatch = context.dispatch_routed_delta_payload(&batch, &routes, &hidden_payload)?;
        assert_eq!(dispatch.stats.global_rows, 2);
        assert_eq!(dispatch.stats.hosts, 2);
        assert_eq!(dispatch.stats.host_rows, 2);
        assert_eq!(dispatch.stats.routes, 2 * GLM52_TOP_K);
        assert_eq!(dispatch.global_row_indices_by_host, vec![vec![0], vec![1]]);

        server.abort();
        assert_eq!(accept_count.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[test]
    fn scheduler_sparse_tcp_batch_slice_rewrites_rows_and_routes() -> Result<()> {
        let source = RowSource {
            kind: RowSourceKind::PrefillChunk,
            request_id: RequestId::from("scheduler-sparse-tcp-slice"),
            sequence_id: "scheduler-sparse-tcp-slice-sequence".to_owned(),
            token_start: PositionId(11),
            row_count: 4,
        };
        let batch = scheduler_sparse_source_batch(
            GLM52_FIRST_K_DENSE_REPLACE,
            &source,
            GraphBucket::new(4),
            &PlacementVersion::from("scheduler-sparse-tcp-slice-placement"),
            4,
        )?;
        let routes = batch
            .rows
            .iter()
            .enumerate()
            .flat_map(|(row_index, row)| {
                (0..row.route_count).map(move |route| ExpertBatchRoute {
                    row_index,
                    expert_id: route,
                    gate_weight: 1.0 / row.route_count as f32,
                })
            })
            .collect::<Vec<_>>();

        let (slice, slice_routes) = scheduler_sparse_tcp_batch_slice(&batch, &routes, 1, 3)?;

        assert_eq!(slice.num_rows(), 2);
        assert_eq!(slice.route_count(), 2 * GLM52_TOP_K);
        assert_eq!(slice.rows[0].row_id, 0);
        assert_eq!(slice.rows[0].token_position, PositionId(12));
        assert_eq!(slice.rows[0].route_offset, 0);
        assert_eq!(slice.rows[0].route_count, GLM52_TOP_K);
        assert_eq!(slice.rows[1].row_id, 1);
        assert_eq!(slice.rows[1].token_position, PositionId(13));
        assert_eq!(slice.rows[1].route_offset, GLM52_TOP_K);
        assert!(slice_routes[..GLM52_TOP_K]
            .iter()
            .all(|route| route.row_index == 0));
        assert!(slice_routes[GLM52_TOP_K..]
            .iter()
            .all(|route| route.row_index == 1));
        Ok(())
    }

    #[test]
    fn scheduler_sparse_rolling_emission_reorders_rows_across_source_chunks() -> Result<()> {
        let make_chunk = |global_row_start: usize| -> Result<SchedulerSparseRollingChunk> {
            let source = RowSource {
                kind: RowSourceKind::PrefillChunk,
                request_id: RequestId::from("scheduler-sparse-rolling-emission"),
                sequence_id: "scheduler-sparse-rolling-emission-sequence".to_owned(),
                token_start: PositionId((20 + global_row_start) as u64),
                row_count: 2,
            };
            let batch = scheduler_sparse_source_batch(
                GLM52_FIRST_K_DENSE_REPLACE,
                &source,
                GraphBucket::new(2),
                &PlacementVersion::from("scheduler-sparse-rolling-emission-placement"),
                2,
            )?;
            let routes = batch
                .rows
                .iter()
                .enumerate()
                .flat_map(|(local_row, row)| {
                    (0..row.route_count).map(move |route_offset| ExpertBatchRoute {
                        row_index: local_row,
                        expert_id: ((global_row_start + local_row) * 17 + route_offset) % 64,
                        gate_weight: 1.0 / row.route_count as f32,
                    })
                })
                .collect::<Vec<_>>();
            let mut hidden_payload = Vec::with_capacity(2 * batch.hidden_bytes_per_row);
            for local_row in 0..2 {
                hidden_payload.extend(std::iter::repeat_n(
                    (global_row_start + local_row + 1) as u8,
                    batch.hidden_bytes_per_row,
                ));
            }
            Ok(SchedulerSparseRollingChunk {
                task_index: global_row_start / 2,
                global_row_start,
                batch,
                routes,
                hidden_payload,
                ready_segments: Vec::new(),
                finalized_segments: Vec::new(),
                task_completed: false,
            })
        };
        let chunks = vec![make_chunk(0)?, make_chunk(2)?];
        let row_indices = vec![2, 0, 3, 1];
        let queued = build_scheduler_sparse_rolling_emission(
            &chunks,
            RollingExpertRowPackEmission {
                row_indices: row_indices.clone(),
                emitted_pack_index: 0,
                admitted_rows: 4,
                oldest_pending_row: 0,
                max_selected_row_offset: 3,
                deadline_row_exclusive: None,
            },
        )?;

        assert_eq!(queued.emission.row_indices, row_indices);
        assert_eq!(queued.batch.num_rows(), 4);
        assert_eq!(queued.batch.graph_bucket.row_capacity, 4);
        assert_eq!(
            queued
                .batch
                .rows
                .iter()
                .map(|row| row.token_position)
                .collect::<Vec<_>>(),
            vec![
                PositionId(22),
                PositionId(20),
                PositionId(23),
                PositionId(21)
            ]
        );
        for (dispatch_row, row) in queued.batch.rows.iter().enumerate() {
            assert_eq!(row.row_id, dispatch_row as u64);
            assert_eq!(row.route_offset, dispatch_row * GLM52_TOP_K);
            assert!(
                queued.routes[row.route_offset..row.route_offset + row.route_count]
                    .iter()
                    .all(|route| route.row_index == dispatch_row)
            );
            let byte_start = dispatch_row * queued.batch.hidden_bytes_per_row;
            let byte_end = byte_start + queued.batch.hidden_bytes_per_row;
            let expected = (queued.emission.row_indices[dispatch_row] + 1) as u8;
            assert!(queued.hidden_payload[byte_start..byte_end]
                .iter()
                .all(|byte| *byte == expected));
        }
        Ok(())
    }

    #[test]
    fn scheduler_sparse_rolling_tail_rebalances_without_exceeding_physical_cap() -> Result<()> {
        let emission =
            |row_start: usize, row_count: usize, pack: usize| RollingExpertRowPackEmission {
                row_indices: (row_start..row_start + row_count).collect(),
                emitted_pack_index: pack,
                admitted_rows: 257,
                oldest_pending_row: row_start,
                max_selected_row_offset: row_count.saturating_sub(1),
                deadline_row_exclusive: (pack == 0).then_some(1),
            };
        let mut emissions = vec![emission(0, 256, 0), emission(256, 1, 1)];

        rebalance_scheduler_sparse_rolling_unsupported_tail_with(&mut emissions, 256, |rows| {
            Ok((2..=256).contains(&rows))
        })?;

        assert_eq!(emissions.len(), 2);
        assert_eq!(emissions[0].row_indices.len(), 255);
        assert_eq!(emissions[1].row_indices.len(), 2);
        assert_eq!(emissions[1].row_indices, vec![255, 256]);
        assert_eq!(emissions[1].oldest_pending_row, 255);
        assert_eq!(emissions[1].max_selected_row_offset, 1);
        assert_eq!(emissions[1].deadline_row_exclusive, Some(1));
        let mut rows = emissions
            .iter()
            .flat_map(|emission| emission.row_indices.iter().copied())
            .collect::<Vec<_>>();
        rows.sort_unstable();
        assert_eq!(rows, (0..257).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn scheduler_sparse_rolling_probe_counts_physical_packs() {
        assert_eq!(rolling_sparse_physical_dispatches_per_layer(4_096), 16);
        assert_eq!(rolling_sparse_physical_dispatches_per_layer(8_192), 32);
        assert_eq!(rolling_sparse_physical_dispatches_per_layer(16_384), 64);
        assert_eq!(rolling_sparse_physical_dispatches_per_layer(4_097), 17);
    }

    #[test]
    fn scheduler_sparse_rolling_response_rows_remap_to_layer_coordinates() -> Result<()> {
        let mut chunk = VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
            host_index: 2,
            partial_output: glmrt_transport::VerbsHostProtocolV2ResponsePayload::from_owned(
                vec![0_u8; 12],
            ),
            output_dtype: ExpertV2Dtype::Bf16,
            output_row_stride_bytes: 4,
            global_row_indices: vec![0, 2, 1],
            completed_global_row_indices: vec![2, 0],
        };

        remap_sparse_payload_chunk_rows(&mut chunk, &[17, 5, 99])?;

        assert_eq!(chunk.global_row_indices, vec![17, 99, 5]);
        assert_eq!(chunk.completed_global_row_indices, vec![99, 17]);
        Ok(())
    }

    #[test]
    fn scheduler_sparse_sliced_stream_offsets_chunks_and_merges_responses() -> Result<()> {
        let (chunk0_tx, chunk0_rx) = mpsc::channel();
        let (chunk1_tx, chunk1_rx) = mpsc::channel();
        let (response0_tx, response0_rx) = mpsc::channel();
        let (response1_tx, response1_rx) = mpsc::channel();
        let batch = SchedulerSparseTcpPayloadDispatchBatchShape {
            layer_id: LayerId(GLM52_FIRST_K_DENSE_REPLACE as u32),
            rows: 4,
            routes: 4 * GLM52_TOP_K,
            unique_experts: 0,
            max_expert_load: 0,
            expert_load_square_sum: 0,
            route_profiled: false,
            hidden_dim: 2,
        };
        let mut handle = SchedulerSparseTcpPayloadDispatchHandle {
            batch,
            batch_index: 1,
            started: None,
            row_sharded_completion_chunks: false,
            chunk_rx: None,
            response_rx: None,
            direct_owner_pending: None,
            sliced_dispatches: vec![
                SchedulerSparseTcpPayloadSliceDispatch {
                    row_start: 0,
                    row_count: 2,
                    chunk_rx: Some(chunk0_rx),
                    response_rx: Some(response0_rx),
                    response: None,
                },
                SchedulerSparseTcpPayloadSliceDispatch {
                    row_start: 2,
                    row_count: 2,
                    chunk_rx: Some(chunk1_rx),
                    response_rx: Some(response1_rx),
                    response: None,
                },
            ],
            sliced_poll_cursor: 0,
            deferred_streaming_completion: None,
        };
        assert_eq!(handle.response_batch_target_rows(), 256);
        handle.row_sharded_completion_chunks = true;
        assert_eq!(handle.response_batch_target_rows(), 4);
        let chunk = |row_index| VerbsHostProtocolV2HostBatchSetBf16PayloadChunk {
            host_index: 0,
            partial_output: glmrt_transport::VerbsHostProtocolV2ResponsePayload::from_owned(
                vec![0_u8; 4],
            ),
            output_dtype: ExpertV2Dtype::Bf16,
            output_row_stride_bytes: 4,
            global_row_indices: vec![row_index],
            completed_global_row_indices: vec![row_index],
        };
        chunk1_tx.send(chunk(0))?;
        let SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk1) =
            handle.poll_streaming_response(false)?
        else {
            anyhow::bail!("second physical slice did not produce a streamed chunk");
        };
        assert_eq!(chunk1.global_row_indices, vec![2]);
        assert_eq!(chunk1.completed_global_row_indices, vec![2]);

        chunk0_tx.send(chunk(1))?;
        let SchedulerSparseTcpPayloadStreamPoll::Chunk(chunk0) =
            handle.poll_streaming_response(false)?
        else {
            anyhow::bail!("first physical slice did not produce a streamed chunk");
        };
        assert_eq!(chunk0.global_row_indices, vec![1]);
        assert_eq!(chunk0.completed_global_row_indices, vec![1]);

        let slice_response = || TcpProtocolV2HostBatchSetBf16PayloadDispatch {
            partial_outputs_bf16_by_host: Vec::new(),
            global_row_indices_by_host: Vec::new(),
            completed_global_row_slices: Vec::new(),
            stats: TcpProtocolV2HostBatchSetDispatchStats {
                hosts: 4,
                global_rows: 2,
                host_rows: 8,
                routes: 2 * GLM52_TOP_K,
                output_dim: 2,
                output_values: 4,
                request_wire_bytes: 100,
                response_wire_bytes: 50,
                response_executor_ids: vec![1; 4],
                contribution_counts: Vec::new(),
                output_checksum: 0.0,
                graph_pool_leases: 0,
                graph_pool_fixed_buffer_bytes: 0,
                graph_pool_active_rows: 0,
                graph_pool_active_routes: 0,
                graph_pool_active_expert_tiles: 0,
                graph_pool_bucket_rows: Vec::new(),
            },
        };
        response0_tx.send(Ok(slice_response()))?;
        response1_tx.send(Ok(slice_response()))?;
        drop(chunk0_tx);
        drop(chunk1_tx);

        let SchedulerSparseTcpPayloadStreamPoll::Complete(dispatch) =
            handle.poll_streaming_response(false)?
        else {
            anyhow::bail!("sliced streamed dispatch did not complete");
        };
        assert_eq!(dispatch.stats.global_rows, 4);
        assert_eq!(dispatch.stats.routes, 4 * GLM52_TOP_K);
        assert_eq!(dispatch.stats.hosts, 8);
        assert_eq!(dispatch.stats.request_wire_bytes, 200);
        assert_eq!(dispatch.stats.response_wire_bytes, 100);
        Ok(())
    }

    #[test]
    fn scheduler_embedding_seed_uses_prompt_and_decode_token_rows() -> Result<()> {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let snapshot = tempfile::tempdir().context("creating embedding seed fixture dir")?;
        let file_path = snapshot.path().join("model.safetensors");
        let mut row0 = vec![0.0_f32; NUMERIC_PROGRESS_HIDDEN_DIM];
        let mut row1 = vec![0.0_f32; NUMERIC_PROGRESS_HIDDEN_DIM];
        row0[0] = 1.0;
        row0[NUMERIC_PROGRESS_HIDDEN_DIM - 1] = -2.0;
        row1[0] = 3.0;
        row1[1] = -4.0;
        row1[NUMERIC_PROGRESS_HIDDEN_DIM - 1] = 5.0;
        let row0_bf16 = bf16_bytes_from_f32(&row0);
        let row1_bf16 = bf16_bytes_from_f32(&row1);
        let mut tensor_bytes = row0_bf16.clone();
        tensor_bytes.extend_from_slice(&row1_bf16);
        fs::write(&file_path, tensor_bytes).context("writing embedding seed fixture")?;

        let catalog = TensorCatalog {
            model_id: "test".to_owned(),
            snapshot_path: snapshot.path().display().to_string(),
            facts: ModelFacts::default(),
            tensors: vec![TensorInfo {
                name: "model.embed_tokens.weight".to_owned(),
                file: "model.safetensors".to_owned(),
                dtype: DType::Bf16,
                shape: vec![2, NUMERIC_PROGRESS_HIDDEN_DIM],
                byte_offset: 0,
                byte_length: (2 * NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>()) as u64,
                role: TensorRole::Embedding,
                layer_id: None,
                expert_id: None,
                is_quantization_metadata: false,
            }],
        };
        let mut progression =
            RealFullSchedulerNumericProgression::new(RealFullSchedulerNumericProgressionShape {
                prefix_tokens: 0,
                prefill_rows: 2,
                prefill_chunk_tokens: 2,
                decode_rows: 1,
                mtp_rows: 1,
                mtp_accepted_rows: 1,
                source_segments_per_layer: 3,
                sparse_source_segments_per_layer: 3,
            });

        progression.seed_prefill_token_embeddings(&catalog, &[1, 0], None)?;
        progression.seed_decode_token_embeddings(&catalog, &[1])?;

        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        assert_eq!(progression.initial_prefill_embedding_rows, 2);
        assert_eq!(progression.initial_decode_embedding_rows, 1);
        assert_eq!(
            progression.initial_prefill_embedding_bytes_read,
            (2 * row_bytes) as u64
        );
        assert_eq!(
            progression.initial_decode_embedding_bytes_read,
            row_bytes as u64
        );
        assert_eq!(
            &progression.residual_bf16[..row_bytes],
            row1_bf16.as_slice()
        );
        assert_eq!(
            &progression.residual_bf16[row_bytes..row_bytes * 2],
            row0_bf16.as_slice()
        );
        assert_eq!(
            &progression.residual_bf16[row_bytes * 2..row_bytes * 3],
            row1_bf16.as_slice()
        );
        Ok(())
    }

    #[test]
    fn scheduler_direct_device_mlp_delta_updates_residual_without_host_delta_scratch() -> Result<()>
    {
        if !coordinator_cuda_reference_kernels_enabled() {
            eprintln!("skipped: CUDA reference kernels are not enabled");
            return Ok(());
        }

        let mut progression = test_progression();
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        let byte_start = 0;
        let byte_end = row_bytes;
        fill_repeated_bf16_bytes(1.0, &mut progression.residual_bf16[byte_start..byte_end]);

        let mut delta_bf16 = vec![0_u8; row_bytes];
        fill_repeated_bf16_bytes(0.5, &mut delta_bf16);
        let delta = match device_bf16_output_from_bf16_bytes(
            &delta_bf16,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler direct device MLP delta unit",
        ) {
            Ok(delta) => delta,
            Err(error) => {
                eprintln!("skipped: CUDA device upload unavailable: {error:#}");
                return Ok(());
            }
        };

        progression.apply_device_direct_delta_bytes(
            byte_start,
            byte_end,
            ResidualDeltaStage::Mlp,
            &delta,
        )?;

        assert!(progression.delta_bf16_scratch.is_empty());
        assert_eq!(progression.mlp_residual_adds, 1);
        assert_eq!(progression.mlp_value_updates, NUMERIC_PROGRESS_HIDDEN_DIM);
        assert_eq!(progression.device_hidden_segment_residual_adds, 1);
        assert_eq!(
            progression.device_hidden_segment_value_updates,
            NUMERIC_PROGRESS_HIDDEN_DIM
        );
        assert!(progression
            .mlp_residual_add_backend
            .expect("MLP residual backend")
            .contains("residual-add-bf16"));
        assert!(progression
            .device_hidden_segment_residual_add_backend
            .expect("device hidden residual backend")
            .contains("residual-add-bf16"));
        assert!(progression.output_bf16_scratch.is_empty());
        assert!(progression.residual_bf16[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 1.0));
        let first_summary = progression.account_device_hidden_segments()?;
        assert_eq!(first_summary.resident_segments, 1);
        assert_eq!(first_summary.resident_values, NUMERIC_PROGRESS_HIDDEN_DIM);
        assert!(progression.residual_bf16[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 1.0));

        progression.apply_device_direct_delta_bytes(
            byte_start,
            byte_end,
            ResidualDeltaStage::Mlp,
            &delta,
        )?;

        assert!(progression.delta_bf16_scratch.is_empty());
        assert!(progression.output_bf16_scratch.is_empty());
        assert_eq!(progression.mlp_residual_adds, 2);
        assert_eq!(
            progression.mlp_value_updates,
            NUMERIC_PROGRESS_HIDDEN_DIM * 2
        );
        assert_eq!(progression.device_hidden_segment_residual_adds, 2);
        assert_eq!(
            progression.device_hidden_segment_value_updates,
            NUMERIC_PROGRESS_HIDDEN_DIM * 2
        );
        let device_segment = progression
            .device_hidden_segments
            .get(&DeviceHiddenSegmentKey {
                byte_start,
                byte_end,
            })
            .context("scheduler resident hidden segment missing after second direct MLP delta")?;
        let device_segment_bytes = device_segment.copy_to_host_bytes()?;
        assert!(device_segment_bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 2.0));
        assert!(progression.residual_bf16[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 1.0));
        let second_summary = progression.account_device_hidden_segments()?;
        assert_eq!(second_summary.resident_segments, 1);
        assert_eq!(second_summary.resident_values, NUMERIC_PROGRESS_HIDDEN_DIM);
        assert!(progression.residual_bf16[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 1.0));

        Ok(())
    }

    #[test]
    fn scheduler_device_delta_template_upload_reuses_bf16_scratch() -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            eprintln!("skipped: CUDA reference kernels are not enabled");
            return Ok(());
        }

        let mut progression = test_progression();
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();

        match progression.device_delta_template_for(1, 0.5) {
            Ok(Some(_)) => {}
            Ok(None) => {
                eprintln!("skipped: CUDA reference kernels are not enabled");
                return Ok(());
            }
            Err(error) => {
                eprintln!("skipped: CUDA template upload unavailable: {error:#}");
                return Ok(());
            }
        }

        assert_eq!(progression.device_delta_template_uploads, 1);
        assert_eq!(progression.device_delta_template_uses, 1);
        assert_eq!(
            progression.device_delta_template_upload_bf16_scratch.len(),
            row_bytes
        );
        let first_upload_scratch_ptr = progression
            .device_delta_template_upload_bf16_scratch
            .as_ptr();
        let first_upload_scratch_capacity = progression
            .device_delta_template_upload_bf16_scratch
            .capacity();
        assert!(first_upload_scratch_capacity >= row_bytes);

        progression.device_delta_template_for(1, 0.25)?;

        assert_eq!(progression.device_delta_template_uploads, 2);
        assert_eq!(progression.device_delta_template_uses, 2);
        assert_eq!(
            progression
                .device_delta_template_upload_bf16_scratch
                .as_ptr(),
            first_upload_scratch_ptr
        );
        assert_eq!(
            progression
                .device_delta_template_upload_bf16_scratch
                .capacity(),
            first_upload_scratch_capacity
        );
        assert_eq!(
            progression.device_delta_template_upload_bf16_scratch.len(),
            row_bytes
        );

        progression.device_delta_template_for(1, 0.5)?;

        assert_eq!(progression.device_delta_template_uploads, 2);
        assert_eq!(progression.device_delta_template_uses, 3);
        assert_eq!(
            progression
                .device_delta_template_upload_bf16_scratch
                .as_ptr(),
            first_upload_scratch_ptr
        );
        assert_eq!(
            progression
                .device_delta_template_upload_bf16_scratch
                .capacity(),
            first_upload_scratch_capacity
        );

        Ok(())
    }

    #[test]
    fn scheduler_mlp_resident_weight_upload_reuses_bf16_scratch() -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            eprintln!("skipped: CUDA reference kernels are not enabled");
            return Ok(());
        }

        let mut progression = test_progression();
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();

        {
            match progression.scheduler_mlp_resident_weights() {
                Ok(_) => {}
                Err(error) => {
                    eprintln!("skipped: CUDA MLP resident weight upload unavailable: {error:#}");
                    return Ok(());
                }
            }
        }

        assert_eq!(progression.device_mlp_weight_uploads, 3);
        assert_eq!(
            progression.device_mlp_weight_resident_values,
            NUMERIC_PROGRESS_HIDDEN_DIM * 3
        );
        assert_eq!(
            progression.device_mlp_weight_upload_bf16_scratch.len(),
            row_bytes
        );
        let first_upload_scratch_ptr = progression.device_mlp_weight_upload_bf16_scratch.as_ptr();
        let first_upload_scratch_capacity =
            progression.device_mlp_weight_upload_bf16_scratch.capacity();
        assert!(first_upload_scratch_capacity >= row_bytes);

        {
            progression.scheduler_mlp_resident_weights()?;
        }

        assert_eq!(progression.device_mlp_weight_uploads, 3);
        assert_eq!(
            progression.device_mlp_weight_resident_values,
            NUMERIC_PROGRESS_HIDDEN_DIM * 3
        );
        assert_eq!(
            progression.device_mlp_weight_upload_bf16_scratch.as_ptr(),
            first_upload_scratch_ptr
        );
        assert_eq!(
            progression.device_mlp_weight_upload_bf16_scratch.capacity(),
            first_upload_scratch_capacity
        );
        assert_eq!(
            progression.device_mlp_weight_upload_bf16_scratch.len(),
            row_bytes
        );

        Ok(())
    }

    #[test]
    fn scheduler_sparse_routed_validation_reuses_normalized_hidden_readback_scratch() -> Result<()>
    {
        if !coordinator_cuda_reference_kernels_enabled() {
            eprintln!("skipped: CUDA reference kernels are not enabled");
            return Ok(());
        }

        let mut progression = test_progression();
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        let mut normalized_bf16 = vec![0_u8; row_bytes];
        fill_repeated_bf16_bytes(0.25, &mut normalized_bf16);
        let normalized = match device_bf16_output_from_bf16_bytes(
            &normalized_bf16,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler sparse routed validation normalized unit",
        ) {
            Ok(normalized) => normalized,
            Err(error) => {
                eprintln!("skipped: CUDA device upload unavailable: {error:#}");
                return Ok(());
            }
        };

        let first_readback = {
            let scratch = &mut progression.device_sparse_routed_normalized_readback_bf16_scratch;
            RealFullSchedulerNumericProgression::read_sparse_routed_normalized_host_bf16_into_scratch(
                3,
                &normalized,
                row_bytes,
                scratch,
            )?
        };
        assert_eq!(first_readback, normalized_bf16.as_slice());
        assert_eq!(
            progression
                .device_sparse_routed_normalized_readback_bf16_scratch
                .len(),
            row_bytes
        );
        let first_readback_ptr = progression
            .device_sparse_routed_normalized_readback_bf16_scratch
            .as_ptr();
        let first_readback_capacity = progression
            .device_sparse_routed_normalized_readback_bf16_scratch
            .capacity();
        assert!(first_readback_capacity >= row_bytes);

        fill_repeated_bf16_bytes(0.5, &mut normalized_bf16);
        let normalized = device_bf16_output_from_bf16_bytes(
            &normalized_bf16,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler sparse routed validation normalized unit",
        )?;
        let second_readback = {
            let scratch = &mut progression.device_sparse_routed_normalized_readback_bf16_scratch;
            RealFullSchedulerNumericProgression::read_sparse_routed_normalized_host_bf16_into_scratch(
                3,
                &normalized,
                row_bytes,
                scratch,
            )?
        };
        assert_eq!(second_readback, normalized_bf16.as_slice());
        assert_eq!(
            progression
                .device_sparse_routed_normalized_readback_bf16_scratch
                .as_ptr(),
            first_readback_ptr
        );
        assert_eq!(
            progression
                .device_sparse_routed_normalized_readback_bf16_scratch
                .capacity(),
            first_readback_capacity
        );

        Ok(())
    }

    #[test]
    fn scheduler_mlp_delta_accounting_avoids_summary_readback() -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            eprintln!("skipped: CUDA reference kernels are not enabled");
            return Ok(());
        }

        let mut progression = test_progression();
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        let mut delta_bf16 = vec![0_u8; row_bytes];
        fill_repeated_bf16_bytes(0.5, &mut delta_bf16);
        let delta = match device_bf16_output_from_bf16_bytes(
            &delta_bf16,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler summary scratch delta unit",
        ) {
            Ok(delta) => delta,
            Err(error) => {
                eprintln!("skipped: CUDA device upload unavailable: {error:#}");
                return Ok(());
            }
        };

        progression.record_synthetic_mlp_delta(1, &delta)?;

        assert_eq!(progression.device_mlp_delta_rows, 1);
        assert_eq!(
            progression.device_mlp_delta_values,
            NUMERIC_PROGRESS_HIDDEN_DIM
        );

        progression.record_synthetic_mlp_delta(1, &delta)?;

        assert_eq!(progression.device_mlp_delta_rows, 2);
        assert_eq!(
            progression.device_mlp_delta_values,
            NUMERIC_PROGRESS_HIDDEN_DIM * 2
        );

        Ok(())
    }

    #[test]
    fn scheduler_direct_device_attention_delta_updates_residual_without_host_delta_scratch(
    ) -> Result<()> {
        if !coordinator_cuda_reference_kernels_enabled() {
            eprintln!("skipped: CUDA reference kernels are not enabled");
            return Ok(());
        }

        let mut progression = test_progression();
        let row_bytes = NUMERIC_PROGRESS_HIDDEN_DIM * std::mem::size_of::<u16>();
        let byte_start = 0;
        let byte_end = row_bytes;
        fill_repeated_bf16_bytes(1.0, &mut progression.residual_bf16[byte_start..byte_end]);

        let mut output_bf16 = vec![0_u8; row_bytes];
        fill_repeated_bf16_bytes(0.25, &mut output_bf16);
        let output_device = match device_bf16_output_from_bf16_bytes(
            &output_bf16,
            1,
            NUMERIC_PROGRESS_HIDDEN_DIM,
            "scheduler direct device attention delta unit",
        ) {
            Ok(delta) => delta,
            Err(error) => {
                eprintln!("skipped: CUDA device upload unavailable: {error:#}");
                return Ok(());
            }
        };
        let checksum = checksum_bf16(&output_bf16)? as f64;
        let attention_delta = RealFullSchedulerDeviceAttentionDelta {
            kind: RowSourceKind::PrefillChunk,
            token_start: 0,
            row_count: 1,
            values_per_row: NUMERIC_PROGRESS_HIDDEN_DIM,
            output_bf16: None,
            output_device: Arc::new(output_device),
            output_device_row_offset: 0,
            checksum,
            backend: "cuda-device-attention-hidden-delta-unit",
        };

        progression.apply_attention_delta(
            byte_start,
            byte_end,
            0.125,
            1,
            RowSourceKind::PrefillChunk,
            Some(&attention_delta),
        )?;

        assert!(progression.delta_bf16_scratch.is_empty());
        assert_eq!(progression.attention_residual_adds, 1);
        assert_eq!(
            progression.attention_value_updates,
            NUMERIC_PROGRESS_HIDDEN_DIM
        );
        assert_eq!(progression.device_hidden_segment_residual_adds, 1);
        assert_eq!(
            progression.device_hidden_segment_value_updates,
            NUMERIC_PROGRESS_HIDDEN_DIM
        );
        assert_eq!(progression.attention_device_output_delta_rows, 1);
        assert_eq!(
            progression.attention_device_output_delta_values,
            NUMERIC_PROGRESS_HIDDEN_DIM
        );
        assert_eq!(
            progression.attention_device_output_delta_device_prefix_rows,
            0
        );
        assert_eq!(
            progression.attention_device_output_delta_device_prefix_values,
            0
        );
        assert!(progression
            .attention_residual_add_backend
            .expect("attention residual backend")
            .contains("residual-add-bf16"));
        assert!(progression
            .device_hidden_segment_residual_add_backend
            .expect("device hidden residual backend")
            .contains("residual-add-bf16"));
        assert!(progression.output_bf16_scratch.is_empty());
        assert!(progression.residual_bf16[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 1.0));
        let summary = progression.account_device_hidden_segments()?;
        assert_eq!(summary.resident_segments, 1);
        assert_eq!(summary.resident_values, NUMERIC_PROGRESS_HIDDEN_DIM);
        assert!(progression.residual_bf16[byte_start..byte_end]
            .chunks_exact(std::mem::size_of::<u16>())
            .all(|chunk| bf16_chunk_to_f32(chunk) == 1.0));

        Ok(())
    }
}
