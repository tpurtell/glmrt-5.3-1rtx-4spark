use anyhow::{bail, Context, Result};
use glmrt_core::{
    owner_for_expert, plan_completion_first_routes, CompletionRoutePlanEntry, DType, KvCacheConfig,
    PlacementPolicy, TensorCatalog, TensorInfo, TensorRole, EXPERT_HOSTS, GLM52_HIDDEN_SIZE,
    GLM52_MOE_INTERMEDIATE_SIZE, GLM52_MTP_LAYER_ID,
};
use glmrt_ffi::GlmrtDeviceBuffer;
use glmrt_loader::is_glm_exl3_recipe;
use glmrt_transport::{
    protocol_v2_verbs_host_execution_lanes, ExpertProtocolV2DeviceResponseRef,
    ExpertProtocolV2FrameBuffer, ExpertProtocolV2Request, ExpertProtocolV2RequestView,
    ExpertProtocolV2Response, ExpertProtocolV2ResponseRef, ExpertProtocolV2ResponseView,
    ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertProtocolV2Status,
    ExpertProtocolV2StreamPlan, ExpertV2Dtype, ProtocolV2ExecutorResponseRef,
    ProtocolV2ExpertExecutor, ProtocolV2RequestDevicePayload, TcpTransportConfig,
    VerbsHostMappedRdmaRing, VerbsHostMappedRdmaRingConfig, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
    MAX_VERBS_HOST_EXECUTION_LANES,
};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet, HashMap},
    env,
    net::SocketAddr,
    sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::commands::model_artifacts::ExpertOwnerLookup;
use crate::commands::real_full::intermediate_sharding::{
    ExpertIntermediateReductionDtype, ExpertIntermediateShard, SparkExpertOwnerReductionConfig,
};
use crate::commands::real_full::scheduler::{
    real_full_scheduler_execute_decode_layer_block,
    real_full_scheduler_precapture_layer_block_attention, RealFullSchedulerExecutionState,
    RealFullSchedulerSparseTcpDispatchWorker,
};
use crate::commands::real_full::sparse_mlp::route::{
    begin_nvfp4_route_ingress_stream_cached, cuda_reference_kernels_enabled,
    cuda_route_validation_enabled, execute_nvfp4_route_cached,
    execute_nvfp4_route_ingress_stream_chunk_cached,
    execute_nvfp4_route_rows_bf16_accumulated_cached,
    execute_nvfp4_route_rows_bf16_accumulated_cached_device_output,
    execute_nvfp4_route_rows_bf16_accumulated_streaming_cached,
    execute_nvfp4_route_rows_nvfp4_accumulated_cached,
    execute_nvfp4_route_rows_nvfp4_accumulated_cached_device_output,
    execute_nvfp4_route_rows_nvfp4_accumulated_streaming_cached,
    preload_bf16_route_projection_group_cache, preload_routed_bf16_projection_cuda_cache,
    preload_routed_exl3_projection_cuda_cache, preload_routed_quant_projection_cuda_cache,
    preload_routed_quant_projection_host_cache,
    preload_routed_quant_projection_scalar_cache_parallel,
    preload_startup_quantized_mtp_projection_cuda_cache, reduce_mapped_route_shards_cached,
    reduce_mapped_route_shards_cached_host_output, try_begin_packed_w4a16_topk8_prefill_cached,
    PackedW4a16Topk8Route, RouteNvfp4IngressStream, RouteNvfp4IngressStreamChunk,
    RouteProjectionCachePreloadRequest, RouteStreamingOutputDtype, RouteTensorCache,
    SparkCollectiveLaunchOrder, SparkCollectiveLaunchTicket,
};
use crate::commands::real_full::sparse_mlp::router::ScoredRoute;
use crate::commands::real_full::{
    mtp_bf16_experts_enabled, preload_real_full_spark_layer_block_weights, SparkLayerBlock,
};

pub(crate) const REAL_NVFP4_PROTOCOL_V2_EXECUTOR: &str =
    "protocol-v2-real-nvfp4-checkpoint-executor";
pub(crate) const REAL_NVFP4_CUDA_REFERENCE_KERNELS_ENV: &str =
    "GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS";
const REAL_NVFP4_PROTOCOL_V2_EXECUTOR_TIMING_ENV: &str =
    "GLMRT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING";
const REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS_ENV: &str =
    "GLMRT_REAL_FULL_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS";
const MIN_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS: usize = 8;
const DEFAULT_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS: usize = 2064;
const MAX_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS: usize = 2064;
const STRIPED_SPARK_COLLECTIVE_QUORUM_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(5);
const REGULAR_SPARK_REDUCTION_MAX_GROUP_ROWS: usize = 256;
const SPARK_OWNER_RING_SLOT_BYTES: usize = 128 * 1024;
const SPARK_OWNER_RESPONSE_TIMEOUT: Duration = Duration::from_secs(60);
const SPARK_OWNER_RING_DEPTH: usize = 8;

pub(crate) fn real_nvfp4_cuda_reference_kernels_enabled() -> bool {
    cuda_reference_kernels_enabled()
}

fn parse_real_nvfp4_protocol_v2_packed_direct_max_rows(value: Option<&str>) -> Result<usize> {
    let max_rows = match value {
        Some(value) => value.parse::<usize>().with_context(|| {
            format!("parsing {REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS_ENV}={value:?}")
        })?,
        None => DEFAULT_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS,
    };
    anyhow::ensure!(
        (MIN_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS
            ..=MAX_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS)
            .contains(&max_rows),
        "{REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS_ENV} must be an integer in {}..={}, got {max_rows}",
        MIN_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS,
        MAX_REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS,
    );
    Ok(max_rows)
}

fn real_nvfp4_protocol_v2_packed_direct_max_rows() -> Result<usize> {
    static MAX_ROWS: OnceLock<std::result::Result<usize, String>> = OnceLock::new();
    match MAX_ROWS.get_or_init(|| {
        parse_real_nvfp4_protocol_v2_packed_direct_max_rows(
            env::var(REAL_NVFP4_PROTOCOL_V2_PACKED_DIRECT_MAX_ROWS_ENV)
                .ok()
                .as_deref(),
        )
        .map_err(|error| error.to_string())
    }) {
        Ok(max_rows) => Ok(*max_rows),
        Err(error) => bail!(error.clone()),
    }
}

pub(crate) struct RealNvfp4ProtocolV2Executor {
    catalog: TensorCatalog,
    real_layer: Option<usize>,
    role_hostname: Option<String>,
    owner_lookup: Option<ExpertOwnerLookup>,
    intermediate_shard: Option<ExpertIntermediateShard>,
    owner_reduction: Option<SparkOwnerReduction>,
    layer_block: Option<SparkLayerBlockRuntime>,
    route_caches: Vec<Mutex<RouteTensorCache>>,
    spark_collective_orders: Vec<Arc<SparkCollectiveLaunchOrder>>,
    projection_rows_cache: Mutex<ProjectionRowsCache>,
    streamed_ingress_by_lane: Vec<Mutex<Option<StreamedIngressState>>>,
}

struct SparkOwnerReduction {
    dtype: ExpertIntermediateReductionDtype,
    shard_count: usize,
    max_rows: usize,
    transport: TcpTransportConfig,
    ring_config: VerbsHostMappedRdmaRingConfig,
    peers: Vec<SparkOwnerRingPeer>,
    child_request_frames: Vec<Mutex<ExpertProtocolV2FrameBuffer>>,
}

struct SparkOwnerRingPeer {
    rank: usize,
    addr: SocketAddr,
    ring: Mutex<Option<VerbsHostMappedRdmaRing>>,
}

struct SparkLayerBlockRuntime {
    block: SparkLayerBlock,
    tx: mpsc::Sender<SparkLayerBlockMessage>,
    join: Mutex<Option<JoinHandle<()>>>,
}

enum SparkLayerBlockMessage {
    Execute {
        source_request_id: u64,
        token_position: usize,
        hidden_bf16: Vec<u8>,
        request_id_base: u64,
        response_tx: mpsc::Sender<Result<Vec<u8>>>,
    },
    Shutdown,
}

struct StreamedIngressState {
    request_id: u64,
    placement_version: u64,
    layer_id: u32,
    hidden_dim: u32,
    hidden_dtype: ExpertV2Dtype,
    hidden_row_stride_bytes: u32,
    response_dtype: ExpertV2Dtype,
    spark_reduction: bool,
    spark_row_sharded_reduction: bool,
    debug_checksum: bool,
    rows: Vec<ExpertProtocolV2RowDescriptor>,
    routes: Vec<ExpertProtocolV2RouteEntry>,
    plan: ExpertProtocolV2StreamPlan,
    hidden_payload: Vec<u8>,
    received_rows: usize,
    route_stream: Option<RouteNvfp4IngressStream>,
}

impl Drop for SparkLayerBlockRuntime {
    fn drop(&mut self) {
        let _ = self.tx.send(SparkLayerBlockMessage::Shutdown);
        let Some(join) = self.join.get_mut().ok().and_then(Option::take) else {
            return;
        };
        if thread::current().id() != join.thread().id() {
            let _ = join.join();
        }
    }
}

fn run_spark_layer_block_worker(
    catalog: TensorCatalog,
    block: SparkLayerBlock,
    owner_endpoint: SocketAddr,
    kv_config: KvCacheConfig,
    rx: mpsc::Receiver<SparkLayerBlockMessage>,
    ready_tx: mpsc::SyncSender<Result<()>>,
) {
    let dispatch_worker = match (|| -> Result<Arc<RealFullSchedulerSparseTcpDispatchWorker>> {
        real_full_scheduler_precapture_layer_block_attention(&catalog, block, kv_config.clone())?;
        Ok(Arc::new(
            RealFullSchedulerSparseTcpDispatchWorker::new_direct_owner(owner_endpoint)
                .context("creating direct Spark layer-block owner dispatch worker")?,
        ))
    })() {
        Ok(worker) => worker,
        Err(error) => {
            let _ = ready_tx.send(Err(error));
            return;
        }
    };
    if ready_tx.send(Ok(())).is_err() {
        return;
    }

    let mut states = HashMap::<u64, RealFullSchedulerExecutionState>::new();
    while let Ok(message) = rx.recv() {
        match message {
            SparkLayerBlockMessage::Execute {
                source_request_id,
                token_position,
                hidden_bf16,
                request_id_base,
                response_tx,
            } => {
                let result = (|| -> Result<Vec<u8>> {
                    if !states.contains_key(&source_request_id) {
                        states.insert(
                            source_request_id,
                            RealFullSchedulerExecutionState::new(
                                kv_config.clone(),
                                format!("spark-layer-block-sequence-{source_request_id}"),
                            )?,
                        );
                    }
                    let state = states
                        .get_mut(&source_request_id)
                        .context("Spark layer-block state disappeared after insertion")?;
                    let output = real_full_scheduler_execute_decode_layer_block(
                        &catalog,
                        block,
                        source_request_id,
                        token_position,
                        &hidden_bf16,
                        Arc::clone(&dispatch_worker),
                        request_id_base,
                        state,
                    )
                    .with_context(|| {
                        format!(
                            "executing Spark layer block {}:{} at token {token_position}",
                            block.start_layer, block.end_layer
                        )
                    })?;
                    output.copy_to_host_bytes()
                })();
                let _ = response_tx.send(result);
            }
            SparkLayerBlockMessage::Shutdown => break,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RealNvfp4ResidentPreloadStats {
    pub(crate) projection_groups: usize,
    pub(crate) layers: usize,
    pub(crate) experts: usize,
    pub(crate) weight_bytes: u64,
    pub(crate) quant_metadata_bytes: u64,
    pub(crate) route_cache_entries: usize,
    pub(crate) route_cache_loads: usize,
    pub(crate) route_cache_hits: usize,
    pub(crate) projection_row_entries: usize,
    pub(crate) projection_row_loads: usize,
    pub(crate) projection_row_hits: usize,
    pub(crate) cuda_reference_enabled: bool,
    pub(crate) cuda_projection_groups: usize,
    pub(crate) cuda_weight_bytes: u64,
    pub(crate) cuda_weight_scale_bytes: u64,
    pub(crate) cuda_projection_entries: usize,
    pub(crate) cuda_projection_uploads: usize,
    pub(crate) cuda_cache_hits: usize,
    pub(crate) cuda_managed_projection_entries: usize,
    pub(crate) cuda_managed_projection_allocations_enabled: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RealNvfp4ResidentPreloadPlan {
    pub(crate) startup_required: bool,
    pub(crate) projection_groups: usize,
    pub(crate) layers: usize,
    pub(crate) experts: usize,
    pub(crate) complete_expert_projection_sets: usize,
    pub(crate) incomplete_expert_projection_sets: usize,
    pub(crate) weight_bytes: u64,
    pub(crate) weight_scale_bytes: u64,
    pub(crate) scalar_metadata_bytes: u64,
    pub(crate) missing_metadata_tensors: usize,
}

impl StreamedIngressState {
    fn from_plan(request: &ExpertProtocolV2RequestView<'_>) -> Result<Self> {
        anyhow::ensure!(
            request.stream_plan_enabled() && !request.stream_data_enabled(),
            "streamed expert ingress state requires a plan frame"
        );
        let rows = (0..request.header.row_count as usize)
            .map(|row| request.row(row))
            .collect::<Result<Vec<_>>>()?;
        let routes = (0..request.header.route_count as usize)
            .map(|route| request.route(route))
            .collect::<Result<Vec<_>>>()?;
        let plan = ExpertProtocolV2StreamPlan::decode(request.hidden_payload())
            .context("decoding streamed expert ingress plan")?;
        plan.validate_against_request(&rows, &routes)
            .context("validating streamed expert ingress plan against request routes")?;
        let hidden_payload_bytes = rows
            .len()
            .checked_mul(request.header.hidden_row_stride_bytes as usize)
            .context("streamed expert ingress hidden payload byte count overflow")?;
        Ok(Self {
            request_id: request.header.request_id,
            placement_version: request.header.placement_version,
            layer_id: request.header.layer_id,
            hidden_dim: request.header.hidden_dim,
            hidden_dtype: request.header.hidden_dtype,
            hidden_row_stride_bytes: request.header.hidden_row_stride_bytes,
            response_dtype: requested_response_dtype(request),
            spark_reduction: request.spark_reduction_enabled(),
            spark_row_sharded_reduction: request.spark_row_sharded_reduction_enabled(),
            debug_checksum: request.debug_checksum_enabled(),
            rows,
            routes,
            plan,
            hidden_payload: vec![0_u8; hidden_payload_bytes],
            received_rows: 0,
            route_stream: None,
        })
    }

    fn accept_data(&mut self, request: &ExpertProtocolV2RequestView<'_>) -> Result<bool> {
        let (row_offset, chunk_rows, row_end, complete) = self.validate_data(request)?;
        let stride = self.hidden_row_stride_bytes as usize;
        for chunk_row in 0..chunk_rows {
            let logical_row = self.plan.activation_row_order[row_offset + chunk_row] as usize;
            let source_start = chunk_row
                .checked_mul(stride)
                .context("streamed expert ingress source row offset overflow")?;
            let destination_start = logical_row
                .checked_mul(stride)
                .context("streamed expert ingress destination row offset overflow")?;
            self.hidden_payload[destination_start..destination_start + stride]
                .copy_from_slice(&request.hidden_payload()[source_start..source_start + stride]);
        }
        self.received_rows = row_end;
        Ok(complete)
    }

    fn validate_data(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<(usize, usize, usize, bool)> {
        anyhow::ensure!(
            request.stream_data_enabled() && !request.stream_plan_enabled(),
            "streamed expert ingress state requires a data frame"
        );
        anyhow::ensure!(
            request.header.request_id == self.request_id
                && request.header.placement_version == self.placement_version
                && request.header.layer_id == self.layer_id,
            "streamed expert ingress data identity does not match active plan"
        );
        anyhow::ensure!(
            request.header.hidden_dim == self.hidden_dim
                && request.header.hidden_dtype == self.hidden_dtype
                && request.header.hidden_row_stride_bytes == self.hidden_row_stride_bytes,
            "streamed expert ingress data hidden shape does not match active plan"
        );
        anyhow::ensure!(
            requested_response_dtype(request) == self.response_dtype
                && request.spark_reduction_enabled() == self.spark_reduction
                && request.spark_row_sharded_reduction_enabled()
                    == self.spark_row_sharded_reduction
                && request.debug_checksum_enabled() == self.debug_checksum,
            "streamed expert ingress data response contract does not match active plan"
        );
        let row_offset = request
            .stream_data_row_offset()
            .context("streamed expert ingress data frame has no row offset")?;
        anyhow::ensure!(
            row_offset == self.received_rows,
            "streamed expert ingress data row offset {row_offset} did not match next row {}",
            self.received_rows
        );
        let chunk_rows = request.header.row_count as usize;
        let row_end = row_offset
            .checked_add(chunk_rows)
            .context("streamed expert ingress data row range overflow")?;
        anyhow::ensure!(
            row_end <= self.plan.activation_row_order.len(),
            "streamed expert ingress data rows {row_offset}..{row_end} exceed plan rows {}",
            self.plan.activation_row_order.len()
        );
        let complete = row_end == self.plan.activation_row_order.len();
        anyhow::ensure!(
            request.stream_final_enabled() == complete,
            "streamed expert ingress final marker={} but received rows are {row_end}/{}",
            request.stream_final_enabled(),
            self.plan.activation_row_order.len()
        );
        Ok((row_offset, chunk_rows, row_end, complete))
    }

    fn into_request(self) -> Result<ExpertProtocolV2Request> {
        anyhow::ensure!(
            self.received_rows == self.plan.activation_row_order.len(),
            "streamed expert ingress completed {} of {} rows",
            self.received_rows,
            self.plan.activation_row_order.len()
        );
        let mut request = ExpertProtocolV2Request::new_with_hidden_stride(
            self.request_id,
            self.placement_version,
            self.layer_id,
            self.hidden_dim,
            self.hidden_dtype,
            self.hidden_row_stride_bytes,
            self.rows,
            self.routes,
            self.hidden_payload,
        )?;
        request = match self.response_dtype {
            ExpertV2Dtype::Fp8E4m3RowScaled => request.with_fp8_e4m3_row_scaled_response(),
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => request.with_nvfp4_e2m1_fp8_e4m3_response(),
            _ => request,
        };
        if self.spark_row_sharded_reduction {
            request = request.with_spark_row_sharded_reduction();
        } else if self.spark_reduction {
            request = request.with_spark_reduction();
        }
        if self.debug_checksum {
            request = request.with_debug_checksum();
        }
        Ok(request)
    }
}

fn requested_response_dtype(request: &ExpertProtocolV2RequestView<'_>) -> ExpertV2Dtype {
    if request.nvfp4_e2m1_fp8_e4m3_response_enabled() {
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
    } else if request.fp8_e4m3_row_scaled_response_enabled() {
        ExpertV2Dtype::Fp8E4m3RowScaled
    } else {
        ExpertV2Dtype::Bf16
    }
}

fn route_streaming_output_dtype(
    response_dtype: ExpertV2Dtype,
) -> Result<RouteStreamingOutputDtype> {
    match response_dtype {
        ExpertV2Dtype::Bf16 => Ok(RouteStreamingOutputDtype::Bf16),
        ExpertV2Dtype::Fp8E4m3RowScaled => Ok(RouteStreamingOutputDtype::Fp8E4m3RowScaled),
        ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => Ok(RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3),
        other => bail!("unsupported streamed route response dtype {other:?}"),
    }
}

fn regular_response_device_target(
    request: &ExpertProtocolV2RequestView<'_>,
    request_device_payload: Option<ProtocolV2RequestDevicePayload>,
    output_row_bytes: usize,
) -> Result<Option<GlmrtDeviceBuffer>> {
    let Some(response_slot) = request_device_payload.and_then(|payload| payload.response_slot)
    else {
        return Ok(None);
    };
    if request.debug_checksum_enabled()
        || request.stream_plan_enabled()
        || request.stream_data_enabled()
    {
        return Ok(None);
    }
    let row_count = request.header.row_count as usize;
    let row_index_bytes = row_count
        .checked_mul(std::mem::size_of::<u32>())
        .context("device response row-index byte count overflow")?;
    let payload_offset = EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN
        .checked_add(row_index_bytes)
        .context("device response payload offset overflow")?;
    let payload_bytes = row_count
        .checked_mul(output_row_bytes)
        .context("device response payload byte count overflow")?;
    let payload_end = payload_offset
        .checked_add(payload_bytes)
        .context("device response payload end overflow")?;
    anyhow::ensure!(
        payload_end <= response_slot.bytes,
        "device response payload range [{payload_offset}, {payload_end}) exceeds mapped slot bytes {}",
        response_slot.bytes
    );
    Ok(Some(GlmrtDeviceBuffer {
        ptr: unsafe { response_slot.ptr.cast::<u8>().add(payload_offset).cast() },
        bytes: payload_bytes,
        device_id: response_slot.device_id,
        flags: response_slot.flags,
    }))
}

fn reduction_expert_dtype(dtype: ExpertIntermediateReductionDtype) -> ExpertV2Dtype {
    match dtype {
        ExpertIntermediateReductionDtype::Bf16 => ExpertV2Dtype::Bf16,
        ExpertIntermediateReductionDtype::Fp8 => ExpertV2Dtype::Fp8E4m3RowScaled,
        ExpertIntermediateReductionDtype::Nvfp4 => ExpertV2Dtype::Nvfp4E2m1Fp8E4m3,
    }
}

fn route_stream_plan(
    row_route_plans: &[Vec<(ScoredRoute, usize)>],
    max_group_rows: usize,
) -> Result<ExpertProtocolV2StreamPlan> {
    let completion_entries = row_route_plans
        .iter()
        .enumerate()
        .flat_map(|(row_index, routes)| {
            routes
                .iter()
                .map(move |(route, intermediate_rows)| CompletionRoutePlanEntry {
                    row_index,
                    expert_id: route.expert_id,
                    intermediate_rows: *intermediate_rows,
                })
        })
        .collect::<Vec<_>>();
    let completion =
        plan_completion_first_routes(&completion_entries, row_route_plans.len(), max_group_rows)?;
    ExpertProtocolV2StreamPlan::from_completion_first(
        row_route_plans.len(),
        completion_entries.len(),
        &completion,
    )
}

fn validate_mapped_owner_partial_response(
    request: &ExpertProtocolV2RequestView<'_>,
    response: &ExpertProtocolV2ResponseView<'_>,
    expected_dtype: ExpertV2Dtype,
    index: usize,
) -> Result<()> {
    anyhow::ensure!(
        response.header.status == ExpertProtocolV2Status::Ok,
        "Spark owner partial {index} returned status {:?}",
        response.header.status
    );
    anyhow::ensure!(
        response.header.request_id == request.header.request_id
            && response.header.placement_version == request.header.placement_version
            && response.header.layer_id == request.header.layer_id,
        "Spark owner partial {index} identity did not match its request"
    );
    anyhow::ensure!(
        response.header.row_count == request.header.row_count
            && response.header.output_dim == request.header.hidden_dim,
        "Spark owner partial {index} shape rows={} width={} did not match rows={} width={}",
        response.header.row_count,
        response.header.output_dim,
        request.header.row_count,
        request.header.hidden_dim
    );
    anyhow::ensure!(
        response.header.output_dtype == expected_dtype,
        "Spark owner partial {index} dtype {:?} did not match expected {expected_dtype:?}",
        response.header.output_dtype
    );
    let row_stride = response
        .header
        .output_dtype
        .row_bytes(request.header.hidden_dim as usize)?;
    anyhow::ensure!(
        response.header.output_row_stride_bytes as usize == row_stride,
        "Spark owner partial {index} row stride {} did not match {:?} stride {row_stride}",
        response.header.output_row_stride_bytes,
        response.header.output_dtype
    );
    let expected_bytes = request
        .header
        .row_count
        .checked_mul(response.header.output_row_stride_bytes)
        .context("Spark owner partial payload byte count overflow")?
        as usize;
    anyhow::ensure!(
        response.partial_output_payload().len() == expected_bytes,
        "Spark owner partial {index} payload bytes {} did not match expected {expected_bytes}",
        response.partial_output_payload().len()
    );
    anyhow::ensure!(
        !response.more_chunks(),
        "Spark owner partial {index} unexpectedly has more chunks"
    );
    if response.row_indexed() {
        for row_index in 0..response.header.row_count as usize {
            anyhow::ensure!(
                response.request_row_index(row_index)? == row_index as u32,
                "Spark owner partial {index} returned unexpected row index at {row_index}"
            );
        }
    }
    if response.debug_checksum_enabled() {
        response.verify_checksum()?;
    }
    Ok(())
}

fn mapped_device_buffer_slice(
    buffer: GlmrtDeviceBuffer,
    offset_bytes: usize,
    bytes: usize,
) -> Result<GlmrtDeviceBuffer> {
    let end = offset_bytes
        .checked_add(bytes)
        .context("mapped owner payload device slice end overflow")?;
    anyhow::ensure!(
        end <= buffer.bytes,
        "mapped owner payload device slice [{offset_bytes}, {end}) exceeds {} bytes",
        buffer.bytes
    );
    Ok(GlmrtDeviceBuffer {
        ptr: unsafe { buffer.ptr.cast::<u8>().add(offset_bytes).cast() },
        bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    })
}

impl RealNvfp4ProtocolV2Executor {
    pub(crate) fn new(
        catalog: TensorCatalog,
        real_layer: Option<usize>,
        role_hostname: Option<String>,
    ) -> Self {
        Self {
            catalog,
            real_layer,
            role_hostname,
            owner_lookup: None,
            intermediate_shard: None,
            owner_reduction: None,
            layer_block: None,
            route_caches: (0..MAX_VERBS_HOST_EXECUTION_LANES)
                .map(|lane| {
                    Mutex::new(RouteTensorCache::for_execution_lane(
                        u32::try_from(lane).expect("execution lane count fits u32"),
                    ))
                })
                .collect(),
            spark_collective_orders: (0..MAX_VERBS_HOST_EXECUTION_LANES)
                .map(|_| {
                    Arc::new(SparkCollectiveLaunchOrder::new(
                        std::time::Duration::from_millis(20),
                    ))
                })
                .collect(),
            projection_rows_cache: Mutex::new(ProjectionRowsCache::default()),
            streamed_ingress_by_lane: (0..MAX_VERBS_HOST_EXECUTION_LANES)
                .map(|_| Mutex::new(None))
                .collect(),
        }
    }

    fn route_cache_for_execution_lane(
        &self,
        execution_lane: usize,
    ) -> Result<MutexGuard<'_, RouteTensorCache>> {
        let active_lanes = protocol_v2_verbs_host_execution_lanes()?;
        anyhow::ensure!(
            execution_lane < active_lanes,
            "Spark execution lane {execution_lane} exceeds configured lane count {active_lanes}"
        );
        let cache = self
            .route_caches
            .get(execution_lane)
            .context("Spark execution lane exceeds executor capacity")?;
        cache.lock().map_err(|_| {
            anyhow::anyhow!("real NVFP4 route tensor cache lane {execution_lane} is poisoned")
        })
    }

    fn streamed_ingress_for_execution_lane(
        &self,
        execution_lane: usize,
    ) -> Result<MutexGuard<'_, Option<StreamedIngressState>>> {
        let active_lanes = protocol_v2_verbs_host_execution_lanes()?;
        anyhow::ensure!(
            execution_lane < active_lanes,
            "Spark execution lane {execution_lane} exceeds configured lane count {active_lanes}"
        );
        let state = self
            .streamed_ingress_by_lane
            .get(execution_lane)
            .context("Spark execution lane exceeds streamed-ingress capacity")?;
        state.lock().map_err(|_| {
            anyhow::anyhow!("streamed expert ingress lane {execution_lane} is poisoned")
        })
    }

    fn acquire_striped_spark_collective_turn(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        execution_lane: usize,
    ) -> Result<Option<SparkCollectiveLaunchTicket>> {
        let part_count = request.spark_collective_part_count();
        if part_count == 0 {
            return Ok(None);
        }
        anyhow::ensure!(
            request.spark_row_sharded_reduction_enabled(),
            "striped Spark collective request requires row-sharded reduction"
        );
        let order = self
            .spark_collective_orders
            .get(execution_lane)
            .context("Spark execution lane exceeds collective-order capacity")?;
        let mut ticket = order.register(request.header.request_id)?;
        ticket.wait_for_turn_with_quorum(part_count, STRIPED_SPARK_COLLECTIVE_QUORUM_TIMEOUT)?;
        Ok(Some(ticket))
    }

    pub(crate) fn with_owner_lookup(mut self, owner_lookup: ExpertOwnerLookup) -> Self {
        self.owner_lookup = Some(owner_lookup);
        self
    }

    pub(crate) fn with_intermediate_shard(
        mut self,
        intermediate_shard: ExpertIntermediateShard,
    ) -> Self {
        self.intermediate_shard = Some(intermediate_shard);
        self
    }

    pub(crate) fn with_owner_reduction(
        mut self,
        config: SparkExpertOwnerReductionConfig,
    ) -> Result<Self> {
        let transport = TcpTransportConfig::default();
        let ring_config = VerbsHostMappedRdmaRingConfig::new(
            SPARK_OWNER_RING_SLOT_BYTES,
            SPARK_OWNER_RING_DEPTH,
        )?;
        let peers = config
            .peers
            .into_iter()
            .map(|(rank, addr)| SparkOwnerRingPeer {
                rank,
                addr,
                ring: Mutex::new(None),
            })
            .collect::<Vec<_>>();
        anyhow::ensure!(
            peers.len() + 1 == config.shard.count,
            "Spark owner reduction rank {} configured {} peers for {} shards",
            config.shard.rank,
            peers.len(),
            config.shard.count
        );
        self.owner_reduction = Some(SparkOwnerReduction {
            dtype: config.dtype,
            shard_count: config.shard.count,
            max_rows: config.max_rows,
            transport,
            ring_config,
            peers,
            child_request_frames: (0..MAX_VERBS_HOST_EXECUTION_LANES)
                .map(|_| {
                    Mutex::new(ExpertProtocolV2FrameBuffer::with_capacity(
                        SPARK_OWNER_RING_SLOT_BYTES,
                    ))
                })
                .collect(),
        });
        Ok(self)
    }

    pub(crate) fn with_layer_block(
        mut self,
        block: SparkLayerBlock,
        owner_endpoint: SocketAddr,
        kv_config: KvCacheConfig,
    ) -> Result<Self> {
        anyhow::ensure!(
            self.owner_reduction.is_some(),
            "Spark layer-block execution requires mapped owner reduction"
        );
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let catalog = self.catalog.clone();
        let join = thread::Builder::new()
            .name(format!(
                "spark-layer-block-{}-{}",
                block.start_layer, block.end_layer
            ))
            .spawn(move || {
                run_spark_layer_block_worker(
                    catalog,
                    block,
                    owner_endpoint,
                    kv_config,
                    rx,
                    ready_tx,
                )
            })
            .context("spawning Spark layer-block execution worker")?;
        let readiness = ready_rx
            .recv()
            .context("Spark layer-block execution worker exited before startup capture")?;
        if let Err(error) = readiness {
            let _ = join.join();
            return Err(error.context("preparing Spark layer-block execution worker"));
        }
        self.layer_block = Some(SparkLayerBlockRuntime {
            block,
            tx,
            join: Mutex::new(Some(join)),
        });
        Ok(self)
    }

    pub(crate) fn preload_assigned_projections(&self) -> Result<RealNvfp4ResidentPreloadStats> {
        self.preload_assigned_projections_with_cuda(cuda_reference_kernels_enabled())
    }

    pub(crate) fn resident_preload_plan(&self) -> Result<RealNvfp4ResidentPreloadPlan> {
        let specs = routed_projection_preload_specs(&self.catalog, self.intermediate_shard)?;
        resident_preload_plan_for_specs(&self.catalog, &specs, self.intermediate_shard)
    }

    pub(crate) fn preload_layer_block_weights(
        &self,
        block: SparkLayerBlock,
    ) -> Result<(usize, usize, u64)> {
        let stats = preload_real_full_spark_layer_block_weights(&self.catalog, block)?;
        Ok((stats.layers, stats.tensors, stats.bytes))
    }

    fn preload_assigned_projections_with_cuda(
        &self,
        preload_cuda: bool,
    ) -> Result<RealNvfp4ResidentPreloadStats> {
        let preload_started = Instant::now();
        let specs = routed_projection_preload_specs(&self.catalog, self.intermediate_shard)?;
        eprintln!(
            "real_nvfp4_preload_stage stage=specs specs={} elapsed_ms={:.3}",
            specs.len(),
            preload_started.elapsed().as_secs_f64() * 1_000.0
        );
        let mut route_cache = self
            .route_caches
            .first()
            .context("real NVFP4 executor is missing execution lane zero")?
            .lock()
            .map_err(|_| anyhow::anyhow!("real NVFP4 route tensor cache is poisoned"))?;
        let mut projection_rows_cache = self
            .projection_rows_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("real NVFP4 projection row cache is poisoned"))?;
        let mut layers = BTreeSet::new();
        let mut experts = BTreeSet::new();
        let mut projection_rows_by_expert =
            BTreeMap::<(usize, usize), (Option<usize>, Option<usize>, Option<usize>)>::new();
        let mut stats = RealNvfp4ResidentPreloadStats {
            projection_groups: specs.len(),
            ..Default::default()
        };
        let preload_host_projection_rows = !preload_cuda || cuda_route_validation_enabled();
        if !preload_host_projection_rows {
            let mut scalar_requests = Vec::with_capacity(specs.len());
            for spec in &specs {
                if !exl3_base_projection_spec(&self.catalog, spec)
                    && !retained_bf16_projection_spec(&self.catalog, spec)?
                    && !startup_quantized_mtp_projection_spec(&self.catalog, spec)?
                {
                    scalar_requests.push(RouteProjectionCachePreloadRequest {
                        layer_id: spec.layer_id,
                        expert_id: spec.expert_id,
                        projection: spec.projection,
                        row_count: spec.row_count,
                    });
                }
            }
            if !scalar_requests.is_empty() {
                let preload = preload_routed_quant_projection_scalar_cache_parallel(
                    &self.catalog,
                    &scalar_requests,
                    &mut route_cache,
                )?;
                stats.quant_metadata_bytes += preload.quant_metadata_bytes;
            }
            eprintln!(
                "real_nvfp4_preload_stage stage=scalar-cache entries={} elapsed_ms={:.3}",
                scalar_requests.len(),
                preload_started.elapsed().as_secs_f64() * 1_000.0
            );
        }

        for spec in &specs {
            let exl3_base = exl3_base_projection_spec(&self.catalog, spec);
            let retained_bf16 = retained_bf16_projection_spec(&self.catalog, spec)?;
            let startup_quantized_mtp = startup_quantized_mtp_projection_spec(&self.catalog, spec)?;
            layers.insert(spec.layer_id);
            experts.insert((spec.layer_id, spec.expert_id));
            projection_rows_cache.prepare_layer(spec.layer_id);
            let row_key = ProjectionRowsKey {
                layer_id: spec.layer_id,
                expert_id: spec.expert_id,
                projection: spec.projection,
            };
            if projection_rows_cache.get(&row_key).is_none() {
                projection_rows_cache.insert(row_key, spec.row_count);
            }
            let rows = projection_rows_by_expert
                .entry((spec.layer_id, spec.expert_id))
                .or_default();
            match spec.projection {
                "gate_proj" => rows.0 = Some(spec.row_count),
                "up_proj" => rows.1 = Some(spec.row_count),
                "down_proj" => rows.2 = Some(spec.row_count),
                projection => bail!("unsupported routed projection {projection}"),
            }
            if !exl3_base
                && !retained_bf16
                && !startup_quantized_mtp
                && preload_host_projection_rows
            {
                let preload = preload_routed_quant_projection_host_cache(
                    &self.catalog,
                    spec.layer_id,
                    spec.expert_id,
                    spec.projection,
                    spec.row_count,
                    &mut route_cache,
                )?;
                stats.weight_bytes += preload.weight_bytes;
                stats.quant_metadata_bytes += preload.quant_metadata_bytes;
            }
        }
        eprintln!(
            "real_nvfp4_preload_stage stage=projection-index entries={} elapsed_ms={:.3}",
            specs.len(),
            preload_started.elapsed().as_secs_f64() * 1_000.0
        );

        let hidden_dim = self.catalog.facts.hidden_size;
        for ((layer_id, expert_id), (gate_rows, up_rows, down_rows)) in projection_rows_by_expert {
            let gate_rows = gate_rows.with_context(|| {
                format!("layer {layer_id} expert {expert_id} is missing gate_proj")
            })?;
            let up_rows = up_rows.with_context(|| {
                format!("layer {layer_id} expert {expert_id} is missing up_proj")
            })?;
            let down_rows = down_rows.with_context(|| {
                format!("layer {layer_id} expert {expert_id} is missing down_proj")
            })?;
            anyhow::ensure!(
                gate_rows == up_rows,
                "layer {layer_id} expert {expert_id} gate/up rows differ: {gate_rows} vs {up_rows}"
            );
            anyhow::ensure!(
                down_rows == hidden_dim,
                "layer {layer_id} expert {expert_id} down rows {down_rows} differ from hidden size {hidden_dim}"
            );
            let retained_bf16 = retained_bf16_projection_spec(
                &self.catalog,
                &RouteProjectionPreloadSpec {
                    layer_id,
                    expert_id,
                    projection: "gate_proj",
                    row_count: gate_rows,
                },
            )?;
            let startup_quantized_mtp = startup_quantized_mtp_projection_spec(
                &self.catalog,
                &RouteProjectionPreloadSpec {
                    layer_id,
                    expert_id,
                    projection: "gate_proj",
                    row_count: gate_rows,
                },
            )?;
            let exl3_base = exl3_base_projection_spec(
                &self.catalog,
                &RouteProjectionPreloadSpec {
                    layer_id,
                    expert_id,
                    projection: "gate_proj",
                    row_count: gate_rows,
                },
            );
            if !exl3_base && !retained_bf16 && !startup_quantized_mtp {
                preload_bf16_route_projection_group_cache(
                    &self.catalog,
                    layer_id,
                    expert_id,
                    gate_rows,
                    down_rows,
                    hidden_dim,
                    &mut route_cache,
                )?;
            }
        }
        eprintln!(
            "real_nvfp4_preload_stage stage=route-groups groups={} elapsed_ms={:.3}",
            experts.len(),
            preload_started.elapsed().as_secs_f64() * 1_000.0
        );

        let cuda_residency_started = Instant::now();
        if preload_cuda {
            let mut exl3_specs = Vec::new();
            let mut retained_specs = Vec::new();
            let mut startup_quantized_specs = Vec::new();
            let mut quant_specs = Vec::new();
            for spec in &specs {
                if exl3_base_projection_spec(&self.catalog, spec) {
                    exl3_specs.push(spec);
                } else if retained_bf16_projection_spec(&self.catalog, spec)? {
                    retained_specs.push(spec);
                } else if startup_quantized_mtp_projection_spec(&self.catalog, spec)? {
                    startup_quantized_specs.push(spec);
                } else {
                    quant_specs.push(spec);
                }
            }
            let requests_for = |specs: Vec<&RouteProjectionPreloadSpec>| {
                specs
                    .into_iter()
                    .map(|spec| RouteProjectionCachePreloadRequest {
                        layer_id: spec.layer_id,
                        expert_id: spec.expert_id,
                        projection: spec.projection,
                        row_count: spec.row_count,
                    })
                    .collect::<Vec<_>>()
            };
            let exl3_requests = requests_for(exl3_specs);
            let quant_requests = requests_for(quant_specs);
            let retained_requests = requests_for(retained_specs);
            let startup_quantized_requests = requests_for(startup_quantized_specs);
            let mut cuda_preload = Default::default();
            if !exl3_requests.is_empty() {
                cuda_preload = preload_routed_exl3_projection_cuda_cache(
                    &self.catalog,
                    &exl3_requests,
                    &mut route_cache,
                )?;
            }
            if !quant_requests.is_empty() {
                let quant_preload = preload_routed_quant_projection_cuda_cache(
                    &self.catalog,
                    &quant_requests,
                    &mut route_cache,
                )?;
                cuda_preload.projection_groups += quant_preload.projection_groups;
                cuda_preload.weight_bytes += quant_preload.weight_bytes;
                cuda_preload.weight_scale_bytes += quant_preload.weight_scale_bytes;
            }
            if !retained_requests.is_empty() {
                let retained_preload = preload_routed_bf16_projection_cuda_cache(
                    &self.catalog,
                    &retained_requests,
                    &mut route_cache,
                )?;
                cuda_preload.projection_groups += retained_preload.projection_groups;
                cuda_preload.weight_bytes += retained_preload.weight_bytes;
                cuda_preload.weight_scale_bytes += retained_preload.weight_scale_bytes;
            }
            if !startup_quantized_requests.is_empty() {
                let startup_preload = preload_startup_quantized_mtp_projection_cuda_cache(
                    &self.catalog,
                    &startup_quantized_requests,
                    &mut route_cache,
                )?;
                cuda_preload.projection_groups += startup_preload.projection_groups;
                cuda_preload.weight_bytes += startup_preload.weight_bytes;
                cuda_preload.weight_scale_bytes += startup_preload.weight_scale_bytes;
            }
            stats.cuda_reference_enabled = true;
            stats.cuda_projection_groups = cuda_preload.projection_groups;
            stats.cuda_weight_bytes = cuda_preload.weight_bytes;
            stats.cuda_weight_scale_bytes = cuda_preload.weight_scale_bytes;
        }

        let route_stats = route_cache.stats();
        let row_stats = projection_rows_cache.stats();
        stats.layers = layers.len();
        stats.experts = experts.len();
        stats.route_cache_entries = route_stats.entries;
        stats.route_cache_loads = route_stats.projection_loads;
        stats.route_cache_hits = route_stats.cache_hits;
        stats.projection_row_entries = row_stats.entries;
        stats.projection_row_loads = row_stats.loads;
        stats.projection_row_hits = row_stats.hits;
        stats.cuda_projection_entries = route_stats.cuda_projection_entries;
        stats.cuda_projection_uploads = route_stats.cuda_projection_uploads;
        stats.cuda_cache_hits = route_stats.cuda_cache_hits;
        stats.cuda_managed_projection_entries = route_stats.cuda_managed_projection_entries;
        stats.cuda_managed_projection_allocations_enabled =
            route_stats.cuda_managed_projection_allocations_enabled;
        eprintln!(
            "real_nvfp4_preload_stage stage=cuda-residency projections={} elapsed_ms={:.3} total_ms={:.3}",
            stats.cuda_projection_groups,
            cuda_residency_started.elapsed().as_secs_f64() * 1_000.0,
            preload_started.elapsed().as_secs_f64() * 1_000.0
        );
        let lane_fork_started = Instant::now();
        let active_lanes = protocol_v2_verbs_host_execution_lanes()?;
        for execution_lane in 1..active_lanes {
            let lane_started = Instant::now();
            let auxiliary_cache = route_cache.fork_execution_lane(
                u32::try_from(execution_lane).context("execution lane exceeds u32")?,
            )?;
            eprintln!(
                "real_nvfp4_execution_lane_connect execution_lane={} elapsed_ms={:.3}",
                execution_lane,
                lane_started.elapsed().as_secs_f64() * 1_000.0,
            );
            let mut destination = self
                .route_caches
                .get(execution_lane)
                .context("execution lane exceeds route-cache capacity")?
                .lock()
                .map_err(|_| {
                    anyhow::anyhow!(
                        "real NVFP4 route tensor cache lane {execution_lane} is poisoned"
                    )
                })?;
            *destination = auxiliary_cache;
        }
        eprintln!(
            "real_nvfp4_preload_stage stage=execution-lane-fork lanes={} elapsed_ms={:.3} total_ms={:.3}",
            active_lanes,
            lane_fork_started.elapsed().as_secs_f64() * 1_000.0,
            preload_started.elapsed().as_secs_f64() * 1_000.0,
        );
        Ok(stats)
    }
}

impl RealNvfp4ProtocolV2Executor {
    fn request_packed_w4a16_topk8_routes(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<Option<Vec<PackedW4a16Topk8Route>>> {
        let row_count = request.header.row_count as usize;
        let expected_routes = row_count
            .checked_mul(8)
            .context("packed W4A16 top-k=8 request route count overflow")?;
        if row_count == 0 || request.header.route_count as usize != expected_routes {
            return Ok(None);
        }
        let mut routes = Vec::with_capacity(expected_routes);
        for row_index in 0..row_count {
            let row = request.row(row_index)?;
            anyhow::ensure!(
                row.route_offset as usize == row_index * 8 && row.route_count == 8,
                "packed W4A16 top-k=8 row {row_index} has route range {}..{}",
                row.route_offset,
                row.route_offset + row.route_count
            );
            for route_index in row_index * 8..row_index * 8 + 8 {
                let route = request.route(route_index)?;
                anyhow::ensure!(
                    route.row_index as usize == row_index,
                    "packed W4A16 top-k=8 route row {} did not match {row_index}",
                    route.row_index
                );
                routes.push(PackedW4a16Topk8Route {
                    expert_id: route.expert_id,
                    weight: route.gate_weight,
                });
            }
        }
        Ok(Some(routes))
    }

    fn request_row_route_plans(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        layer_id: usize,
    ) -> Result<Vec<Vec<(ScoredRoute, usize)>>> {
        let row_count = request.header.row_count as usize;
        let mut row_route_plans = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let row = request.row(row_index)?;
            let route_start = row.route_offset as usize;
            let route_end = route_start
                .checked_add(row.route_count as usize)
                .context("real NVFP4 ProtocolV2 row route range overflow")?;
            let mut route_plans = Vec::with_capacity(row.route_count as usize);
            for route_index in route_start..route_end {
                let route = request.route(route_index)?;
                anyhow::ensure!(
                    route.row_index as usize == row_index,
                    "real NVFP4 ProtocolV2 route row_index {} did not match row {row_index}",
                    route.row_index
                );
                self.validate_route_owner(layer_id, route.expert_id as usize)?;
                let scored_route = ScoredRoute {
                    expert_id: route.expert_id as usize,
                    score: route.gate_weight,
                    corrected_score: route.gate_weight,
                    normalized_weight: route.gate_weight,
                };
                let intermediate_rows =
                    self.projection_rows_cached(layer_id, scored_route.expert_id, "gate_proj")?;
                route_plans.push((scored_route, intermediate_rows));
            }
            row_route_plans.push(route_plans);
        }
        Ok(row_route_plans)
    }

    fn streamed_ingress_row_route_plans(
        &self,
        state: &StreamedIngressState,
    ) -> Result<Vec<Vec<(ScoredRoute, usize)>>> {
        let layer_id = state.layer_id as usize;
        if let Some(expected_layer) = self.real_layer {
            anyhow::ensure!(
                layer_id == expected_layer,
                "real NVFP4 ProtocolV2 executor pinned to layer {expected_layer}, got layer {layer_id}"
            );
        }
        let mut row_route_plans = Vec::with_capacity(state.rows.len());
        for (row_index, row) in state.rows.iter().enumerate() {
            let route_start = row.route_offset as usize;
            let route_end = route_start
                .checked_add(row.route_count as usize)
                .context("streamed expert ingress row route range overflow")?;
            let mut route_plans = Vec::with_capacity(row.route_count as usize);
            for route in state.routes.get(route_start..route_end).with_context(|| {
                format!("streamed expert ingress row {row_index} route range is invalid")
            })? {
                anyhow::ensure!(
                    route.row_index as usize == row_index,
                    "streamed expert ingress route row {} did not match {row_index}",
                    route.row_index
                );
                self.validate_route_owner(layer_id, route.expert_id as usize)?;
                let scored_route = ScoredRoute {
                    expert_id: route.expert_id as usize,
                    score: route.gate_weight,
                    corrected_score: route.gate_weight,
                    normalized_weight: route.gate_weight,
                };
                let intermediate_rows =
                    self.projection_rows_cached(layer_id, scored_route.expert_id, "gate_proj")?;
                route_plans.push((scored_route, intermediate_rows));
            }
            row_route_plans.push(route_plans);
        }
        Ok(row_route_plans)
    }

    fn begin_streamed_ingress(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        execution_lane: usize,
        response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
    ) -> Result<Option<ExpertProtocolV2Response>> {
        let mut state = StreamedIngressState::from_plan(request)?;
        if state.hidden_dtype == ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
            && cuda_reference_kernels_enabled()
            && !cuda_route_validation_enabled()
        {
            let row_route_plans = self.streamed_ingress_row_route_plans(&state)?;
            let output_dtype = route_streaming_output_dtype(state.response_dtype)?;
            let mut route_cache = self.route_cache_for_execution_lane(execution_lane)?;
            let mut route_stream = begin_nvfp4_route_ingress_stream_cached(
                &self.catalog,
                state.layer_id as usize,
                state.hidden_dim as usize,
                state.hidden_row_stride_bytes as usize,
                &row_route_plans,
                state.hidden_dim as usize,
                output_dtype,
                state.spark_reduction,
                state.spark_row_sharded_reduction,
                &state.plan,
                &mut route_cache,
            )?;
            if state.spark_reduction {
                route_stream.register_collective_request(state.request_id, &route_cache)?;
            }
            state.route_stream = Some(route_stream);
            state.hidden_payload.clear();
            state.hidden_payload.shrink_to_fit();
        }
        let mut active = self.streamed_ingress_for_execution_lane(execution_lane)?;
        if active.replace(state).is_some() {
            tracing::warn!(
                request_id = request.header.request_id,
                layer_id = request.header.layer_id,
                "replaced incomplete streamed expert ingress plan"
            );
        }
        drop(active);
        emit_or_return_streamed_ingress_response(
            streamed_ingress_acknowledgement(request)?,
            response_emit,
        )
    }

    fn continue_streamed_ingress(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        request_device_payload: Option<ProtocolV2RequestDevicePayload>,
        execution_lane: usize,
        response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
    ) -> Result<Option<ExpertProtocolV2Response>> {
        let mut active = self.streamed_ingress_for_execution_lane(execution_lane)?;
        let uses_route_stream = active
            .as_ref()
            .context("streamed expert ingress data arrived without an active plan")?
            .route_stream
            .is_some();
        if uses_route_stream {
            let chunk = {
                let state = active
                    .as_mut()
                    .expect("streamed expert ingress state exists above");
                let (row_offset, _, row_end, complete) = state.validate_data(request)?;
                let mut route_cache = self.route_cache_for_execution_lane(execution_lane)?;
                let chunk = execute_nvfp4_route_ingress_stream_chunk_cached(
                    &self.catalog,
                    state
                        .route_stream
                        .as_mut()
                        .expect("GPU route stream exists above"),
                    request.hidden_payload(),
                    request_device_payload.map(|payload| payload.hidden_payload),
                    false,
                    None,
                    row_offset,
                    request.stream_final_enabled(),
                    &mut route_cache,
                )?;
                anyhow::ensure!(
                    chunk.complete == complete,
                    "streamed expert ingress GPU completion disagreed with frame completion"
                );
                state.received_rows = row_end;
                chunk
            };
            if chunk.complete {
                active.take();
            }
            drop(active);
            let response = if chunk.reduction_follower || chunk.completed_rows.is_empty() {
                streamed_ingress_acknowledgement(request)?
            } else {
                streamed_ingress_chunk_response(request, chunk)?
            };
            return emit_or_return_streamed_ingress_response(response, response_emit);
        }

        let completed = {
            let state = active
                .as_mut()
                .context("streamed expert ingress data arrived without an active plan")?;
            if state.accept_data(request)? {
                active.take()
            } else {
                None
            }
        };
        drop(active);
        let Some(completed) = completed else {
            return emit_or_return_streamed_ingress_response(
                streamed_ingress_acknowledgement(request)?,
                response_emit,
            );
        };

        let assembled = completed.into_request()?;
        let frame = assembled.encode()?;
        let assembled_view = ExpertProtocolV2RequestView::parse(&frame)?;
        if let Some(emit) = response_emit {
            let mut indexed_emit = |response: ProtocolV2ExecutorResponseRef<'_>| match response {
                ProtocolV2ExecutorResponseRef::Host(response) => {
                    if response.header.row_count == 0 || response.row_indexed() {
                        return emit(ProtocolV2ExecutorResponseRef::Host(response));
                    }
                    let row_indices = (0..response.header.row_count).collect::<Vec<_>>();
                    let more_chunks = response.more_chunks();
                    emit(ProtocolV2ExecutorResponseRef::Host(
                        response.with_row_indices(&row_indices, more_chunks)?,
                    ))
                }
                ProtocolV2ExecutorResponseRef::Device(response) => {
                    emit(ProtocolV2ExecutorResponseRef::Device(response))
                }
            };
            self.execute_protocol_v2(
                &assembled_view,
                None,
                execution_lane,
                Some(&mut indexed_emit),
            )
            .context("executing fully assembled streamed expert ingress request")
        } else {
            let response = self
                .execute_protocol_v2(&assembled_view, None, execution_lane, None)
                .context("executing fully assembled streamed expert ingress request")?;
            response
                .map(|response| {
                    if response.header.row_count == 0 || response.row_indexed() {
                        Ok(response)
                    } else {
                        let row_indices = (0..response.header.row_count).collect::<Vec<_>>();
                        response.with_row_indices(row_indices, false)
                    }
                })
                .transpose()
        }
    }

    fn execute_owner_reduction(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        request_device_payload: Option<ProtocolV2RequestDevicePayload>,
        execution_lane: usize,
        mut response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
    ) -> Result<Option<ExpertProtocolV2Response>> {
        let owner = self
            .owner_reduction
            .as_ref()
            .context("Spark owner reduction request has no owner configuration")?;
        anyhow::ensure!(
            !request.stream_plan_enabled() && !request.stream_data_enabled(),
            "Spark owner reduction currently supports regular requests only"
        );
        anyhow::ensure!(
            !request.precompile_warmup_enabled(),
            "Spark owner reduction does not accept precompile warmups"
        );
        anyhow::ensure!(
            request.header.row_count as usize <= owner.max_rows,
            "Spark owner reduction rows {} exceed configured maximum {}",
            request.header.row_count,
            owner.max_rows
        );
        let started = Instant::now();
        let peer_dtype = reduction_expert_dtype(owner.dtype);
        let mut peer_frame_buffer = owner
            .child_request_frames
            .get(execution_lane)
            .context("Spark owner execution lane exceeds child request frame capacity")?
            .lock()
            .map_err(|_| anyhow::anyhow!("Spark owner child request frame is poisoned"))?;
        let peer_frame = peer_frame_buffer.encode_regular_forwarded_request(request, peer_dtype)?;
        anyhow::ensure!(
            peer_frame.len() <= owner.ring_config.slot_capacity_bytes,
            "Spark owner request bytes {} exceed mapped ring capacity {}",
            peer_frame.len(),
            owner.ring_config.slot_capacity_bytes
        );
        let mut peer_rings = Vec::with_capacity(owner.peers.len());
        for peer in &owner.peers {
            let mut ring = peer
                .ring
                .lock()
                .map_err(|_| anyhow::anyhow!("Spark owner ring rank {} is poisoned", peer.rank))?;
            if ring.is_none() {
                *ring = Some(
                    VerbsHostMappedRdmaRing::connect(
                        &peer.addr.to_string(),
                        &owner.transport,
                        owner.ring_config,
                    )
                    .with_context(|| {
                        format!(
                            "connecting mapped Spark owner ring to rank {} {}",
                            peer.rank, peer.addr
                        )
                    })?,
                );
            }
            if let Err(first_error) = ring
                .as_mut()
                .expect("Spark owner ring connected above")
                .send_copy(&peer_frame)
            {
                *ring = None;
                *ring = Some(
                    VerbsHostMappedRdmaRing::connect(
                        &peer.addr.to_string(),
                        &owner.transport,
                        owner.ring_config,
                    )
                    .with_context(|| {
                        format!(
                            "reconnecting mapped Spark owner ring to rank {} {} after send failure: {first_error:#}",
                            peer.rank, peer.addr
                        )
                    })?,
                );
                ring.as_mut()
                    .expect("Spark owner ring reconnected above")
                    .send_copy(&peer_frame)
                    .with_context(|| {
                        format!(
                            "sending mapped Spark owner request to rank {} {} after reconnect",
                            peer.rank, peer.addr
                        )
                    })?;
            }
            peer_rings.push((peer.rank, ring));
        }
        drop(peer_frame_buffer);
        let fanout_ms = elapsed_ms(started);
        let output_dtype = requested_response_dtype(request);
        let output_route_dtype = route_streaming_output_dtype(output_dtype)?;
        let peer_route_dtype = route_streaming_output_dtype(peer_dtype)?;
        let output_row_stride_bytes = output_dtype.row_bytes(request.header.hidden_dim as usize)?;
        let emit_borrowed_output = response_emit.is_some() && !request.debug_checksum_enabled();
        let owner_work = (|| -> Result<(Option<Vec<u8>>, f64, f64, f64)> {
            anyhow::ensure!(
                cuda_reference_kernels_enabled() && !cuda_route_validation_enabled(),
                "mapped Spark owner reduction requires non-validating CUDA route execution"
            );
            let layer_id = request.header.layer_id as usize;
            let row_count = request.header.row_count as usize;
            let hidden_dim = request.header.hidden_dim as usize;
            let row_route_plans = self.request_row_route_plans(request, layer_id)?;
            anyhow::ensure!(
                row_route_plans.iter().all(|routes| !routes.is_empty()),
                "mapped Spark owner reduction requires routes for every row"
            );
            let mut route_cache = self.route_cache_for_execution_lane(execution_lane)?;
            let local_started = Instant::now();
            let mut owned_local = None;
            let local_buffer = if request.header.hidden_dtype == ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 {
                let plan =
                    route_stream_plan(&row_route_plans, REGULAR_SPARK_REDUCTION_MAX_GROUP_ROWS)?;
                let packed_routes = self.request_packed_w4a16_topk8_routes(request)?;
                let packed_stream = if let Some(packed_routes) = packed_routes.as_deref() {
                    try_begin_packed_w4a16_topk8_prefill_cached(
                        layer_id,
                        hidden_dim,
                        request.header.hidden_row_stride_bytes as usize,
                        row_count,
                        packed_routes,
                        hidden_dim,
                        RouteStreamingOutputDtype::Bf16,
                        false,
                        false,
                        &mut route_cache,
                    )?
                } else {
                    None
                };
                let mut stream = match packed_stream {
                    Some(stream) => stream,
                    None => begin_nvfp4_route_ingress_stream_cached(
                        &self.catalog,
                        layer_id,
                        hidden_dim,
                        request.header.hidden_row_stride_bytes as usize,
                        &row_route_plans,
                        hidden_dim,
                        RouteStreamingOutputDtype::Bf16,
                        false,
                        false,
                        &plan,
                        &mut route_cache,
                    )?,
                };
                let activation_payload = if stream.consumes_request_order() {
                    Cow::Borrowed(request.hidden_payload())
                } else {
                    activation_ordered_hidden_payload(request, &plan)?
                };
                let activation_device_payload = matches!(&activation_payload, Cow::Borrowed(_))
                    .then(|| request_device_payload.map(|payload| payload.hidden_payload))
                    .flatten();
                let chunk = execute_nvfp4_route_ingress_stream_chunk_cached(
                    &self.catalog,
                    &mut stream,
                    activation_payload.as_ref(),
                    activation_device_payload,
                    true,
                    None,
                    0,
                    true,
                    &mut route_cache,
                )?;
                anyhow::ensure!(
                    chunk.complete,
                    "mapped Spark owner local direct batch did not complete"
                );
                anyhow::ensure!(
                    !chunk.reduction_follower,
                    "mapped Spark owner local direct batch became a reduction follower"
                );
                anyhow::ensure!(
                    chunk.output.is_empty(),
                    "mapped Spark owner local direct batch unexpectedly returned a host payload"
                );
                anyhow::ensure!(
                    chunk.completed_rows == (0..row_count).collect::<Vec<_>>(),
                    "mapped Spark owner local direct batch returned rows {:?}, expected request order",
                    chunk.completed_rows
                );
                chunk
                    .device_output
                    .context("mapped Spark owner local direct batch did not retain device output")?
            } else {
                let local = match request.header.hidden_dtype {
                    ExpertV2Dtype::Bf16 => {
                        execute_nvfp4_route_rows_bf16_accumulated_cached_device_output(
                            &self.catalog,
                            layer_id,
                            request.hidden_payload(),
                            hidden_dim,
                            request.header.hidden_row_stride_bytes as usize,
                            &row_route_plans,
                            hidden_dim,
                            &mut route_cache,
                        )?
                    }
                    ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => {
                        execute_nvfp4_route_rows_nvfp4_accumulated_cached_device_output(
                            &self.catalog,
                            layer_id,
                            request.hidden_payload(),
                            hidden_dim,
                            request.header.hidden_row_stride_bytes as usize,
                            &row_route_plans,
                            hidden_dim,
                            &mut route_cache,
                        )?
                    }
                    other => bail!(
                        "mapped Spark owner reduction requires BF16 or NVFP4 hidden input, got {other:?}"
                    ),
                };
                owned_local = Some(local.output_device);
                owned_local
                    .as_ref()
                    .expect("mapped owner local output was retained above")
                    .buffer()
            };
            let local_ms = elapsed_ms(local_started);

            let wait_started = Instant::now();
            let mut received_slots = Vec::with_capacity(peer_rings.len());
            let mut peer_payloads = Vec::with_capacity(peer_rings.len());
            for (index, (rank, ring_guard)) in peer_rings.iter_mut().enumerate() {
                let ring = ring_guard
                    .as_mut()
                    .context("mapped Spark owner ring disappeared while awaiting response")?;
                let slot = ring
                    .wait_recv_slot_with_timeout(SPARK_OWNER_RESPONSE_TIMEOUT)
                    .with_context(|| {
                        format!("waiting for mapped Spark owner response from shard rank {rank}")
                    })?;
                let storage = unsafe {
                    std::slice::from_raw_parts(slot.host_ptr.cast_const(), slot.capacity_bytes)
                };
                let wire_bytes = ExpertProtocolV2Response::wire_bytes_from_header(
                    &storage[..EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN],
                )?;
                anyhow::ensure!(
                    wire_bytes <= slot.capacity_bytes,
                    "mapped Spark owner response bytes {wire_bytes} exceed slot capacity {}",
                    slot.capacity_bytes
                );
                let response = ExpertProtocolV2ResponseView::parse(&storage[..wire_bytes])?;
                validate_mapped_owner_partial_response(request, &response, peer_dtype, index)?;
                let payload = response.partial_output_payload();
                let payload_offset = (payload.as_ptr() as usize)
                    .checked_sub(storage.as_ptr() as usize)
                    .context("mapped Spark owner payload pointer precedes its slot")?;
                peer_payloads.push(mapped_device_buffer_slice(
                    slot.device_buffer,
                    payload_offset,
                    payload.len(),
                )?);
                received_slots.push(slot);
            }
            anyhow::ensure!(
                peer_payloads.len() + 1 == owner.shard_count,
                "mapped Spark owner reduction collected {} of {} shards",
                peer_payloads.len() + 1,
                owner.shard_count
            );
            let wait_ms = elapsed_ms(wait_started);

            let reduce_started = Instant::now();
            if emit_borrowed_output {
                let output = reduce_mapped_route_shards_cached_host_output(
                    local_buffer,
                    &peer_payloads,
                    peer_route_dtype,
                    row_count,
                    hidden_dim,
                    output_route_dtype,
                    &mut route_cache,
                )?;
                drop(owned_local);
                for ((rank, ring_guard), slot) in peer_rings.iter_mut().zip(received_slots) {
                    ring_guard
                        .as_mut()
                        .context("mapped Spark owner ring disappeared before slot release")?
                        .release_recv_slot(slot.sequence)
                        .with_context(|| {
                            format!("releasing mapped Spark owner response slot for rank {rank}")
                        })?;
                }
                let response = ExpertProtocolV2ResponseRef::new_with_output_stride(
                    request.header.request_id,
                    request.header.placement_version,
                    request.header.layer_id,
                    request.header.row_count,
                    request.header.hidden_dim,
                    output_dtype,
                    u32::try_from(output_row_stride_bytes)
                        .context("Spark owner output row stride exceeds u32")?,
                    ExpertProtocolV2Status::Ok,
                    output,
                )?;
                let emit = response_emit
                    .as_deref_mut()
                    .context("borrowed Spark owner response has no mapped emitter")?;
                emit(ProtocolV2ExecutorResponseRef::Host(response))?;
                return Ok((None, local_ms, wait_ms, elapsed_ms(reduce_started)));
            }
            let output = reduce_mapped_route_shards_cached(
                local_buffer,
                &peer_payloads,
                peer_route_dtype,
                row_count,
                hidden_dim,
                output_route_dtype,
                &mut route_cache,
            )?;
            drop(owned_local);
            for ((rank, ring_guard), slot) in peer_rings.iter_mut().zip(received_slots) {
                ring_guard
                    .as_mut()
                    .context("mapped Spark owner ring disappeared before slot release")?
                    .release_recv_slot(slot.sequence)
                    .with_context(|| {
                        format!("releasing mapped Spark owner response slot for rank {rank}")
                    })?;
            }
            Ok((Some(output), local_ms, wait_ms, elapsed_ms(reduce_started)))
        })();
        if owner_work.is_err() {
            for (_, ring) in &mut peer_rings {
                **ring = None;
            }
        }
        let (output, local_ms, wait_ms, reduce_ms) = owner_work?;
        if real_nvfp4_protocol_v2_executor_timing_enabled() {
            eprintln!(
                "spark_owner_reduction_timing request_id={} layer_id={} rows={} routes={} peers={} peer_dtype={:?} output_dtype={:?} fanout_ms={:.3} local_ms={:.3} wait_ms={:.3} reduce_ms={:.3} total_ms={:.3}",
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                owner.peers.len(),
                peer_dtype,
                output_dtype,
                fanout_ms,
                local_ms,
                wait_ms,
                reduce_ms,
                elapsed_ms(started)
            );
        }
        let Some(output) = output else {
            return Ok(None);
        };
        let response = ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            request.header.row_count,
            request.header.hidden_dim,
            output_dtype,
            u32::try_from(output_row_stride_bytes)
                .context("Spark owner output row stride exceeds u32")?,
            ExpertProtocolV2Status::Ok,
            output,
        )?;
        let response = if request.debug_checksum_enabled() {
            response.with_debug_checksum()
        } else {
            response
        };
        if let Some(emit) = response_emit.as_deref_mut() {
            emit(ProtocolV2ExecutorResponseRef::Host(response.as_borrowed()))?;
            Ok(None)
        } else {
            Ok(Some(response))
        }
    }

    fn execute_layer_block(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<ExpertProtocolV2Response> {
        let runtime = self
            .layer_block
            .as_ref()
            .context("layer-block request reached an executor without a layer block")?;
        anyhow::ensure!(
            request.header.layer_id as usize == runtime.block.start_layer,
            "layer-block request starts at layer {}, but this Spark owns {}:{}",
            request.header.layer_id,
            runtime.block.start_layer,
            runtime.block.end_layer
        );
        anyhow::ensure!(
            request.header.row_count == 1
                && request.header.hidden_dim as usize == GLM52_HIDDEN_SIZE
                && request.header.hidden_dtype == ExpertV2Dtype::Bf16,
            "layer-block prototype requires one BF16 decode row of width {GLM52_HIDDEN_SIZE}"
        );
        anyhow::ensure!(
            !request.fp8_e4m3_row_scaled_response_enabled()
                && !request.nvfp4_e2m1_fp8_e4m3_response_enabled(),
            "layer-block boundary output currently requires BF16"
        );
        let row = request.row(0)?;
        anyhow::ensure!(
            row.source_kind == glmrt_transport::ExpertV2SourceKind::Decode,
            "layer-block prototype currently supports decode rows only"
        );
        let token_position = usize::try_from(row.token_position)
            .context("layer-block token position exceeds usize")?;
        let logical_hidden_bytes = ExpertV2Dtype::Bf16.row_bytes(GLM52_HIDDEN_SIZE)?;
        let hidden_row = request.hidden_row_payload(0)?;
        anyhow::ensure!(
            hidden_row.len() >= logical_hidden_bytes,
            "layer-block hidden row is shorter than BF16 hidden width"
        );
        let (response_tx, response_rx) = mpsc::channel();
        runtime
            .tx
            .send(SparkLayerBlockMessage::Execute {
                source_request_id: row.source_request_id,
                token_position,
                hidden_bf16: hidden_row[..logical_hidden_bytes].to_vec(),
                request_id_base: request.header.request_id.saturating_mul(1_000_000),
                response_tx,
            })
            .context("sending request to Spark layer-block execution worker")?;
        let output_payload = response_rx
            .recv()
            .context("Spark layer-block execution worker exited before responding")??;
        let response = ExpertProtocolV2Response::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            request.header.row_count,
            request.header.hidden_dim,
            ExpertV2Dtype::Bf16,
            u32::try_from(logical_hidden_bytes).context("layer-block BF16 stride exceeds u32")?,
            ExpertProtocolV2Status::Ok,
            output_payload,
        )?;
        Ok(if request.debug_checksum_enabled() {
            response.with_debug_checksum()
        } else {
            response
        })
    }

    fn execute_protocol_v2(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        request_device_payload: Option<ProtocolV2RequestDevicePayload>,
        execution_lane: usize,
        response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
    ) -> Result<Option<ExpertProtocolV2Response>> {
        let mut collective_ticket =
            self.acquire_striped_spark_collective_turn(request, execution_lane)?;
        let result = self.execute_protocol_v2_unordered(
            request,
            request_device_payload,
            execution_lane,
            response_emit,
        );
        if result.is_ok() {
            if let Some(ticket) = collective_ticket.as_mut() {
                ticket
                    .finish()
                    .context("advancing striped Spark collective launch order")?;
            }
        }
        result
    }

    fn execute_protocol_v2_unordered(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        request_device_payload: Option<ProtocolV2RequestDevicePayload>,
        execution_lane: usize,
        mut response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
    ) -> Result<Option<ExpertProtocolV2Response>> {
        let timing_enabled = real_nvfp4_protocol_v2_executor_timing_enabled();
        let total_started = Instant::now();
        if request.debug_checksum_enabled() {
            request.verify_checksum()?;
        }
        if request.layer_block_enabled() {
            let response = self.execute_layer_block(request)?;
            if let Some(emit) = response_emit.as_deref_mut() {
                emit(ProtocolV2ExecutorResponseRef::Host(response.as_borrowed()))?;
                return Ok(None);
            }
            return Ok(Some(response));
        }
        let retained_bf16_mtp =
            retained_bf16_layer(&self.catalog, request.header.layer_id as usize)?;
        if retained_bf16_mtp {
            anyhow::ensure!(
                request.header.hidden_dtype == ExpertV2Dtype::Bf16,
                "retained BF16 layer-78 experts require BF16 ingress, got {:?}",
                request.header.hidden_dtype
            );
            anyhow::ensure!(
                requested_response_dtype(request) == ExpertV2Dtype::Bf16,
                "retained BF16 layer-78 experts require BF16 responses"
            );
            anyhow::ensure!(
                !request.spark_reduction_enabled()
                    && !request.spark_row_sharded_reduction_enabled(),
                "retained BF16 layer-78 experts must bypass FP8/NVFP4 Spark reduction"
            );
        }
        if request.spark_reduction_enabled()
            && self
                .owner_reduction
                .as_ref()
                .is_some_and(|owner| request.header.row_count as usize <= owner.max_rows)
            && !request.stream_plan_enabled()
            && !request.stream_data_enabled()
        {
            return self.execute_owner_reduction(
                request,
                request_device_payload,
                execution_lane,
                response_emit,
            );
        }
        if request.stream_plan_enabled() {
            return self.begin_streamed_ingress(request, execution_lane, response_emit);
        }
        if request.stream_data_enabled() {
            return self.continue_streamed_ingress(
                request,
                request_device_payload,
                execution_lane,
                response_emit,
            );
        }
        let prequantized_hidden = request.header.hidden_dtype == ExpertV2Dtype::Nvfp4E2m1Fp8E4m3;
        if request.header.hidden_dtype != ExpertV2Dtype::Bf16 && !prequantized_hidden {
            bail!(
                "real NVFP4 ProtocolV2 executor requires BF16 or NVFP4 hidden dtype, got {:?}",
                request.header.hidden_dtype
            );
        }
        let layer_id = request.header.layer_id as usize;
        if let Some(expected_layer) = self.real_layer {
            if layer_id != expected_layer {
                bail!(
                    "real NVFP4 ProtocolV2 executor pinned to layer {expected_layer}, got layer {layer_id}"
                );
            }
        }

        let row_count = request.header.row_count as usize;
        let hidden_dim = request.header.hidden_dim as usize;
        let response_dtype = if request.nvfp4_e2m1_fp8_e4m3_response_enabled() {
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
        } else if request.fp8_e4m3_row_scaled_response_enabled() {
            ExpertV2Dtype::Fp8E4m3RowScaled
        } else {
            ExpertV2Dtype::Bf16
        };
        let route_streaming_output_dtype = match response_dtype {
            ExpertV2Dtype::Bf16 => RouteStreamingOutputDtype::Bf16,
            ExpertV2Dtype::Fp8E4m3RowScaled => RouteStreamingOutputDtype::Fp8E4m3RowScaled,
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3 => RouteStreamingOutputDtype::Nvfp4E2m1Fp8E4m3,
            _ => unreachable!("executor response dtype is restricted above"),
        };
        let bf16_output_row_bytes = hidden_dim
            .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
            .context("real NVFP4 ProtocolV2 output row byte count overflow")?;
        let response_output_row_bytes = response_dtype.row_bytes(hidden_dim)?;
        let response_device_target = regular_response_device_target(
            request,
            request_device_payload,
            response_output_row_bytes,
        )?;
        let output_stride = if response_dtype != ExpertV2Dtype::Bf16 || prequantized_hidden {
            response_output_row_bytes
        } else {
            request.header.hidden_row_stride_bytes as usize
        };
        if prequantized_hidden && !cuda_reference_kernels_enabled() {
            bail!("prequantized NVFP4 hidden exchange requires CUDA route execution");
        }
        if timing_enabled {
            eprintln!(
                "real_nvfp4_protocol_v2_executor_enter request_id={} layer_id={} rows={} routes={} hidden_dim={} cuda_reference={}",
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                request.header.hidden_dim,
                cuda_reference_kernels_enabled()
            );
        }
        let route_cache_lock_started = Instant::now();
        let mut route_cache = self.route_cache_for_execution_lane(execution_lane)?;
        let route_cache_lock_ms = elapsed_ms(route_cache_lock_started);
        if timing_enabled {
            eprintln!(
                "real_nvfp4_protocol_v2_executor_route_cache_locked request_id={} layer_id={} rows={} routes={} route_cache_lock_ms={:.3}",
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                route_cache_lock_ms
            );
        }
        let spark_reduction = request.spark_reduction_enabled();
        let packed_direct_max_rows = real_nvfp4_protocol_v2_packed_direct_max_rows()?;
        if (spark_reduction || (1..=packed_direct_max_rows).contains(&row_count))
            && prequantized_hidden
            && cuda_reference_kernels_enabled()
            && !cuda_route_validation_enabled()
        {
            let plan_started = Instant::now();
            let packed_routes = self.request_packed_w4a16_topk8_routes(request)?;
            let plan_ms = elapsed_ms(plan_started);
            if let Some(packed_routes) = packed_routes {
                let spark_route_started = Instant::now();
                if let Some(mut stream) = try_begin_packed_w4a16_topk8_prefill_cached(
                    layer_id,
                    hidden_dim,
                    request.header.hidden_row_stride_bytes as usize,
                    row_count,
                    &packed_routes,
                    hidden_dim,
                    route_streaming_output_dtype,
                    spark_reduction,
                    spark_reduction && request.spark_row_sharded_reduction_enabled(),
                    &mut route_cache,
                )? {
                    if spark_reduction {
                        stream
                            .register_collective_request(request.header.request_id, &route_cache)?;
                    }
                    let chunk = execute_nvfp4_route_ingress_stream_chunk_cached(
                        &self.catalog,
                        &mut stream,
                        request.hidden_payload(),
                        request_device_payload.map(|payload| payload.hidden_payload),
                        request_device_payload.is_some() && !request.debug_checksum_enabled(),
                        response_device_target,
                        0,
                        true,
                        &mut route_cache,
                    )?;
                    anyhow::ensure!(
                        chunk.complete,
                        "packed Spark-reduced route did not complete its full payload"
                    );
                    if timing_enabled {
                        eprintln!(
                            "real_nvfp4_protocol_v2_spark_reduction_timing execution_lane={} request_id={} layer_id={} rows={} routes={} route_cache_lock_ms={:.3} row_plan_ms={:.3} spark_route_ms={:.3} total_ms={:.3} packed_fast_path=1 spark_reduction={}",
                            execution_lane,
                            request.header.request_id,
                            request.header.layer_id,
                            request.header.row_count,
                            request.header.route_count,
                            route_cache_lock_ms,
                            plan_ms,
                            elapsed_ms(spark_route_started),
                            elapsed_ms(total_started),
                            spark_reduction,
                        );
                    }
                    return emit_complete_route_stream_chunk(
                        request,
                        chunk,
                        response_dtype,
                        response_output_row_bytes,
                        response_emit,
                    );
                }
            }
        }
        let plan_started = Instant::now();
        let row_route_plans = self.request_row_route_plans(request, layer_id)?;
        let plan_ms = elapsed_ms(plan_started);
        if timing_enabled {
            eprintln!(
                "real_nvfp4_protocol_v2_executor_plan_ready request_id={} layer_id={} rows={} routes={} nonempty_rows={} plan_ms={:.3}",
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                row_route_plans.iter().filter(|routes| !routes.is_empty()).count(),
                plan_ms
            );
        }
        let direct_small_topk8 = !spark_reduction
            && prequantized_hidden
            && (2..=8).contains(&row_count)
            && row_route_plans.iter().all(|routes| {
                routes.len() == 8
                    && routes
                        .iter()
                        .all(|(_, intermediate_rows)| *intermediate_rows == 512)
            });
        if spark_reduction || direct_small_topk8 {
            let spark_route_started = Instant::now();
            anyhow::ensure!(
                prequantized_hidden
                    && cuda_reference_kernels_enabled()
                    && !cuda_route_validation_enabled(),
                "direct Spark route ingress requires non-validating CUDA execution with NVFP4 ingress"
            );
            anyhow::ensure!(
                row_route_plans.iter().all(|routes| !routes.is_empty()),
                "direct Spark route ingress requires routes for every row"
            );
            let plan = route_stream_plan(&row_route_plans, REGULAR_SPARK_REDUCTION_MAX_GROUP_ROWS)?;
            let mut stream = begin_nvfp4_route_ingress_stream_cached(
                &self.catalog,
                layer_id,
                hidden_dim,
                request.header.hidden_row_stride_bytes as usize,
                &row_route_plans,
                hidden_dim,
                route_streaming_output_dtype,
                spark_reduction,
                spark_reduction && request.spark_row_sharded_reduction_enabled(),
                &plan,
                &mut route_cache,
            )?;
            if spark_reduction {
                stream.register_collective_request(request.header.request_id, &route_cache)?;
            }
            let activation_payload = if stream.consumes_request_order() {
                Cow::Borrowed(request.hidden_payload())
            } else {
                activation_ordered_hidden_payload(request, &plan)?
            };
            let activation_device_payload = matches!(&activation_payload, Cow::Borrowed(_))
                .then(|| request_device_payload.map(|payload| payload.hidden_payload))
                .flatten();
            let chunk = execute_nvfp4_route_ingress_stream_chunk_cached(
                &self.catalog,
                &mut stream,
                activation_payload.as_ref(),
                activation_device_payload,
                request_device_payload.is_some() && !request.debug_checksum_enabled(),
                response_device_target,
                0,
                true,
                &mut route_cache,
            )?;
            anyhow::ensure!(
                chunk.complete,
                "regular Spark-reduced route did not complete its full payload"
            );
            if timing_enabled {
                eprintln!(
                    "real_nvfp4_protocol_v2_spark_reduction_timing execution_lane={} request_id={} layer_id={} rows={} routes={} route_cache_lock_ms={:.3} row_plan_ms={:.3} spark_route_ms={:.3} total_ms={:.3}",
                    execution_lane,
                    request.header.request_id,
                    request.header.layer_id,
                    request.header.row_count,
                    request.header.route_count,
                    route_cache_lock_ms,
                    plan_ms,
                    elapsed_ms(spark_route_started),
                    elapsed_ms(total_started),
                );
            }
            return emit_complete_route_stream_chunk(
                request,
                chunk,
                response_dtype,
                response_output_row_bytes,
                response_emit,
            );
        }

        let cuda_started = Instant::now();
        let mut streamed_response_wire_bytes = 0_usize;
        let mut streamed_response_build_ms = 0.0_f64;
        let stream_cuda_completions = response_emit.is_some()
            && cuda_reference_kernels_enabled()
            && !cuda_route_validation_enabled()
            && row_route_plans.iter().any(|routes| !routes.is_empty());
        anyhow::ensure!(
            response_dtype == ExpertV2Dtype::Bf16
                || stream_cuda_completions
                || row_route_plans.iter().all(Vec::is_empty),
            "low-precision responses require streaming CUDA route completion"
        );
        let (output_payload, completion_slices) = if cuda_reference_kernels_enabled() {
            if row_route_plans.iter().any(|routes| !routes.is_empty()) {
                if timing_enabled {
                    eprintln!(
                        "real_nvfp4_protocol_v2_executor_cuda_route_start request_id={} layer_id={} rows={} routes={} hidden_dim={} output_stride={}",
                        request.header.request_id,
                        request.header.layer_id,
                        request.header.row_count,
                        request.header.route_count,
                        hidden_dim,
                        output_stride
                    );
                }
                if stream_cuda_completions {
                    let emit = response_emit
                        .as_deref_mut()
                        .expect("streaming response callback is present");
                    let mut emitted_rows = 0_usize;
                    let mut emit_completion = |completed_rows: &[usize], compact_output: &[u8]| {
                        let response_started = Instant::now();
                        let expected_compact_bytes = completed_rows
                            .len()
                            .checked_mul(response_output_row_bytes)
                            .context("streamed route compact output byte count overflow")?;
                        anyhow::ensure!(
                                    compact_output.len() == expected_compact_bytes,
                                    "streamed route compact output bytes {} did not match expected {expected_compact_bytes}",
                                    compact_output.len()
                                );
                        let padded_output;
                        let output_payload = if output_stride == response_output_row_bytes {
                            compact_output
                        } else {
                            padded_output = {
                                let mut output_payload = vec![
                                    0_u8;
                                    checked_row_payload_bytes(
                                        completed_rows.len(),
                                        output_stride,
                                    )?
                                ];
                                for chunk_row in 0..completed_rows.len() {
                                    let compact_start = chunk_row * response_output_row_bytes;
                                    let output_start = chunk_row * output_stride;
                                    output_payload
                                        [output_start..output_start + response_output_row_bytes]
                                        .copy_from_slice(
                                            &compact_output[compact_start
                                                ..compact_start + response_output_row_bytes],
                                        );
                                }
                                output_payload
                            };
                            &padded_output
                        };
                        emitted_rows = emitted_rows
                            .checked_add(completed_rows.len())
                            .context("streamed route emitted row count overflow")?;
                        let more_chunks = emitted_rows < row_count;
                        let row_indices = completed_rows
                            .iter()
                            .map(|row_index| {
                                u32::try_from(*row_index).with_context(|| {
                                    format!("streamed response row {row_index} exceeds u32")
                                })
                            })
                            .collect::<Result<Vec<_>>>()?;
                        let response = ExpertProtocolV2ResponseRef::new_with_output_stride(
                            request.header.request_id,
                            request.header.placement_version,
                            request.header.layer_id,
                            completed_rows.len() as u32,
                            request.header.hidden_dim,
                            response_dtype,
                            output_stride as u32,
                            ExpertProtocolV2Status::Ok,
                            output_payload,
                        )?
                        .with_row_indices(&row_indices, more_chunks)?;
                        let response = if request.debug_checksum_enabled() {
                            response.with_debug_checksum()
                        } else {
                            response
                        };
                        streamed_response_wire_bytes = streamed_response_wire_bytes
                            .checked_add(response.wire_stats().wire_bytes)
                            .context("streamed response wire byte count overflow")?;
                        streamed_response_build_ms += elapsed_ms(response_started);
                        emit(ProtocolV2ExecutorResponseRef::Host(response))
                    };
                    let execution = if prequantized_hidden {
                        execute_nvfp4_route_rows_nvfp4_accumulated_streaming_cached(
                            &self.catalog,
                            layer_id,
                            request.hidden_payload(),
                            hidden_dim,
                            request.header.hidden_row_stride_bytes as usize,
                            &row_route_plans,
                            hidden_dim,
                            &mut route_cache,
                            route_streaming_output_dtype,
                            &mut emit_completion,
                        )?
                    } else {
                        execute_nvfp4_route_rows_bf16_accumulated_streaming_cached(
                            &self.catalog,
                            layer_id,
                            request.hidden_payload(),
                            hidden_dim,
                            request.header.hidden_row_stride_bytes as usize,
                            &row_route_plans,
                            hidden_dim,
                            &mut route_cache,
                            route_streaming_output_dtype,
                            &mut emit_completion,
                        )?
                    };
                    anyhow::ensure!(
                        emitted_rows == row_count,
                        "streamed route emitted {emitted_rows} rows, expected {row_count}"
                    );
                    (None, execution.completion_slices)
                } else {
                    let execution = if prequantized_hidden {
                        execute_nvfp4_route_rows_nvfp4_accumulated_cached(
                            &self.catalog,
                            layer_id,
                            request.hidden_payload(),
                            hidden_dim,
                            request.header.hidden_row_stride_bytes as usize,
                            &row_route_plans,
                            hidden_dim,
                            &mut route_cache,
                        )?
                    } else {
                        execute_nvfp4_route_rows_bf16_accumulated_cached(
                            &self.catalog,
                            layer_id,
                            request.hidden_payload(),
                            hidden_dim,
                            request.header.hidden_row_stride_bytes as usize,
                            &row_route_plans,
                            hidden_dim,
                            &mut route_cache,
                        )?
                    };
                    let output_payload = if output_stride == bf16_output_row_bytes {
                        execution.output_bf16
                    } else {
                        let mut output_payload =
                            vec![0_u8; checked_row_payload_bytes(row_count, output_stride)?];
                        for row_index in 0..row_count {
                            let output_start = row_index * output_stride;
                            let compact_start = row_index * bf16_output_row_bytes;
                            let compact_end = compact_start + bf16_output_row_bytes;
                            output_payload[output_start..output_start + bf16_output_row_bytes]
                                .copy_from_slice(
                                    &execution.output_bf16[compact_start..compact_end],
                                );
                        }
                        output_payload
                    };
                    (Some(output_payload), execution.completion_slices)
                }
            } else {
                let mut output_payload =
                    vec![0_u8; checked_row_payload_bytes(row_count, output_stride)?];
                if response_dtype == ExpertV2Dtype::Fp8E4m3RowScaled {
                    for row_index in 0..row_count {
                        let scale_offset = row_index
                            .checked_mul(output_stride)
                            .and_then(|offset| offset.checked_add(hidden_dim))
                            .context("empty FP8 response scale offset overflow")?;
                        output_payload[scale_offset..scale_offset + std::mem::size_of::<f32>()]
                            .copy_from_slice(&1.0_f32.to_le_bytes());
                    }
                }
                (
                    Some(output_payload),
                    completion_slices_for_all_rows(row_count),
                )
            }
        } else {
            anyhow::ensure!(
                response_dtype == ExpertV2Dtype::Bf16,
                "low-precision responses require CUDA route execution"
            );
            let mut output_payload =
                vec![0_u8; checked_row_payload_bytes(row_count, output_stride)?];
            for (row_index, route_plans) in row_route_plans.iter().enumerate() {
                let hidden_payload = request.hidden_row_payload(row_index)?;
                let hidden = bf16_row_to_f32(hidden_payload, hidden_dim)?;
                let mut reduced_output = vec![0.0_f32; hidden_dim];
                for (scored_route, intermediate_rows) in route_plans {
                    let route_outputs = execute_nvfp4_route_cached(
                        &self.catalog,
                        layer_id,
                        &hidden,
                        scored_route,
                        *intermediate_rows,
                        hidden_dim,
                        &mut route_cache,
                    )?
                    .outputs;
                    for (dst, delta) in reduced_output.iter_mut().zip(route_outputs.iter()) {
                        *dst += *delta;
                    }
                }

                let start = row_index * output_stride;
                f32_values_to_bf16_bytes(
                    &reduced_output,
                    &mut output_payload[start..start + bf16_output_row_bytes],
                );
            }
            (
                Some(output_payload),
                completion_slices_for_all_rows(row_count),
            )
        };
        validate_completion_slices(&completion_slices, row_count)?;
        let cuda_ms = elapsed_ms(cuda_started);

        let response_started = Instant::now();
        let mut response = if let Some(output_payload) = output_payload {
            let response = ExpertProtocolV2Response::new_with_output_stride(
                request.header.request_id,
                request.header.placement_version,
                request.header.layer_id,
                request.header.row_count,
                request.header.hidden_dim,
                response_dtype,
                output_stride as u32,
                ExpertProtocolV2Status::Ok,
                output_payload,
            )?;
            Some(if request.debug_checksum_enabled() {
                response.with_debug_checksum()
            } else {
                response
            })
        } else {
            None
        };
        let response_build_ms = elapsed_ms(response_started) + streamed_response_build_ms;
        if let Some(emit) = response_emit.as_deref_mut() {
            if let Some(monolithic_response) = response.take() {
                streamed_response_wire_bytes = monolithic_response.wire_stats().wire_bytes;
                emit(ProtocolV2ExecutorResponseRef::Host(
                    monolithic_response.as_borrowed(),
                ))?;
            }
        }
        let response_wire_bytes = response
            .as_ref()
            .map(|response| response.wire_stats().wire_bytes)
            .unwrap_or(streamed_response_wire_bytes);
        if timing_enabled {
            eprintln!(
                "real_nvfp4_protocol_v2_executor_timing execution_lane={} request_id={} layer_id={} rows={} routes={} hidden_dim={} completion_slices={} first_completion_rows={} request_wire_bytes={} response_wire_bytes={} route_cache_lock_ms={:.3} plan_ms={:.3} cuda_ms={:.3} response_build_ms={:.3} total_ms={:.3}",
                execution_lane,
                request.header.request_id,
                request.header.layer_id,
                request.header.row_count,
                request.header.route_count,
                request.header.hidden_dim,
                completion_slices.len(),
                completion_slices.first().map(Vec::len).unwrap_or(0),
                request.wire_stats().wire_bytes,
                response_wire_bytes,
                route_cache_lock_ms,
                plan_ms,
                cuda_ms,
                response_build_ms,
                elapsed_ms(total_started)
            );
            tracing::info!(
                execution_lane = execution_lane,
                request_id = request.header.request_id,
                layer_id = request.header.layer_id,
                rows = request.header.row_count,
                routes = request.header.route_count,
                hidden_dim = request.header.hidden_dim,
                completion_slices = completion_slices.len(),
                first_completion_rows = completion_slices.first().map(Vec::len).unwrap_or(0),
                request_wire_bytes = request.wire_stats().wire_bytes,
                response_wire_bytes = response_wire_bytes,
                route_cache_lock_ms = route_cache_lock_ms,
                plan_ms = plan_ms,
                cuda_ms = cuda_ms,
                response_build_ms = response_build_ms,
                total_ms = elapsed_ms(total_started),
                "real_nvfp4_protocol_v2_executor_timing"
            );
        }
        Ok(response)
    }
}

fn activation_ordered_hidden_payload<'a>(
    request: &ExpertProtocolV2RequestView<'a>,
    plan: &ExpertProtocolV2StreamPlan,
) -> Result<Cow<'a, [u8]>> {
    if plan
        .activation_row_order
        .iter()
        .enumerate()
        .all(|(position, row)| *row as usize == position)
    {
        return Ok(Cow::Borrowed(request.hidden_payload()));
    }
    let stride = request.header.hidden_row_stride_bytes as usize;
    let mut payload = Vec::with_capacity(
        plan.activation_row_order
            .len()
            .checked_mul(stride)
            .context("regular Spark reduction activation payload byte count overflow")?,
    );
    for row in &plan.activation_row_order {
        payload.extend_from_slice(request.hidden_row_payload(*row as usize)?);
    }
    Ok(Cow::Owned(payload))
}

impl ProtocolV2ExpertExecutor for RealNvfp4ProtocolV2Executor {
    fn name(&self) -> &'static str {
        REAL_NVFP4_PROTOCOL_V2_EXECUTOR
    }

    fn execute(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
    ) -> Result<ExpertProtocolV2Response> {
        self.execute_protocol_v2(request, None, 0, None)?
            .context("real NVFP4 ProtocolV2 executor did not return a response")
    }

    fn execute_streaming(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        emit: &mut dyn FnMut(ExpertProtocolV2ResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let mut host_emit = |response: ProtocolV2ExecutorResponseRef<'_>| match response {
            ProtocolV2ExecutorResponseRef::Host(response) => emit(response),
            ProtocolV2ExecutorResponseRef::Device(_) => {
                bail!("device-backed response reached a host-only executor callback")
            }
        };
        let response = self.execute_protocol_v2(request, None, 0, Some(&mut host_emit))?;
        anyhow::ensure!(
            response.is_none(),
            "real NVFP4 streaming executor retained a response"
        );
        Ok(())
    }

    fn execute_streaming_device_payload(
        &self,
        request: &ExpertProtocolV2RequestView<'_>,
        device_payload: ProtocolV2RequestDevicePayload,
        emit: &mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>,
    ) -> Result<()> {
        let execution_lane = device_payload.execution_lane as usize;
        let response =
            self.execute_protocol_v2(request, Some(device_payload), execution_lane, Some(emit))?;
        anyhow::ensure!(
            response.is_none(),
            "real NVFP4 streaming executor retained a response"
        );
        Ok(())
    }
}

fn streamed_ingress_acknowledgement(
    request: &ExpertProtocolV2RequestView<'_>,
) -> Result<ExpertProtocolV2Response> {
    let output_dtype = requested_response_dtype(request);
    let output_row_stride_bytes = output_dtype.row_bytes(request.header.hidden_dim as usize)?;
    let response = ExpertProtocolV2Response::new_with_output_stride(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        0,
        request.header.hidden_dim,
        output_dtype,
        u32::try_from(output_row_stride_bytes)
            .context("streamed expert ingress acknowledgement row stride exceeds u32")?,
        ExpertProtocolV2Status::Ok,
        Vec::new(),
    )?;
    Ok(if request.debug_checksum_enabled() {
        response.with_debug_checksum()
    } else {
        response
    })
}

fn streamed_ingress_chunk_response(
    request: &ExpertProtocolV2RequestView<'_>,
    chunk: RouteNvfp4IngressStreamChunk,
) -> Result<ExpertProtocolV2Response> {
    anyhow::ensure!(
        chunk.device_output.is_none(),
        "device-backed streamed ingress response reached the host response builder"
    );
    let output_dtype = requested_response_dtype(request);
    let output_row_stride_bytes = output_dtype.row_bytes(request.header.hidden_dim as usize)?;
    let expected_bytes = chunk
        .completed_rows
        .len()
        .checked_mul(output_row_stride_bytes)
        .context("streamed expert ingress response byte count overflow")?;
    anyhow::ensure!(
        chunk.output.len() == expected_bytes,
        "streamed expert ingress response bytes {} did not match {expected_bytes}",
        chunk.output.len()
    );
    let row_indices = chunk
        .completed_rows
        .iter()
        .map(|row| u32::try_from(*row).context("streamed expert ingress response row exceeds u32"))
        .collect::<Result<Vec<_>>>()?;
    let response = ExpertProtocolV2Response::new_with_output_stride(
        request.header.request_id,
        request.header.placement_version,
        request.header.layer_id,
        u32::try_from(row_indices.len())
            .context("streamed expert ingress response row count exceeds u32")?,
        request.header.hidden_dim,
        output_dtype,
        u32::try_from(output_row_stride_bytes)
            .context("streamed expert ingress response row stride exceeds u32")?,
        ExpertProtocolV2Status::Ok,
        chunk.output,
    )?
    .with_row_indices(row_indices, false)?;
    Ok(if request.debug_checksum_enabled() {
        response.with_debug_checksum()
    } else {
        response
    })
}

fn emit_complete_route_stream_chunk(
    request: &ExpertProtocolV2RequestView<'_>,
    chunk: RouteNvfp4IngressStreamChunk,
    response_dtype: ExpertV2Dtype,
    response_output_row_bytes: usize,
    mut response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
) -> Result<Option<ExpertProtocolV2Response>> {
    if let Some(device_output) = chunk.device_output {
        anyhow::ensure!(
            !chunk.reduction_follower && chunk.output.is_empty(),
            "device-backed Spark response retained an invalid host or follower payload"
        );
        let row_indices = chunk
            .completed_rows
            .iter()
            .map(|row| u32::try_from(*row).context("device-backed Spark response row exceeds u32"))
            .collect::<Result<Vec<_>>>()?;
        let response = ExpertProtocolV2DeviceResponseRef::new_with_output_stride(
            request.header.request_id,
            request.header.placement_version,
            request.header.layer_id,
            u32::try_from(row_indices.len())
                .context("device-backed Spark response row count exceeds u32")?,
            request.header.hidden_dim,
            response_dtype,
            u32::try_from(response_output_row_bytes)
                .context("device-backed Spark response row stride exceeds u32")?,
            ExpertProtocolV2Status::Ok,
            device_output,
        )?
        .with_row_indices(&row_indices, false)?;
        let emit = response_emit
            .as_deref_mut()
            .context("device-backed Spark response has no mapped transport emitter")?;
        emit(ProtocolV2ExecutorResponseRef::Device(response))?;
        return Ok(None);
    }
    let response = if chunk.reduction_follower {
        streamed_ingress_acknowledgement(request)?
    } else {
        streamed_ingress_chunk_response(request, chunk)?
    };
    emit_or_return_streamed_ingress_response(response, response_emit)
}

fn emit_or_return_streamed_ingress_response(
    response: ExpertProtocolV2Response,
    response_emit: Option<&mut dyn FnMut(ProtocolV2ExecutorResponseRef<'_>) -> Result<()>>,
) -> Result<Option<ExpertProtocolV2Response>> {
    if let Some(emit) = response_emit {
        emit(ProtocolV2ExecutorResponseRef::Host(response.as_borrowed()))?;
        Ok(None)
    } else {
        Ok(Some(response))
    }
}

fn real_nvfp4_protocol_v2_executor_timing_enabled() -> bool {
    env::var(REAL_NVFP4_PROTOCOL_V2_EXECUTOR_TIMING_ENV)
        .map(|value| {
            matches!(
                value.to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn checked_row_payload_bytes(row_count: usize, row_stride_bytes: usize) -> Result<usize> {
    row_count
        .checked_mul(row_stride_bytes)
        .context("real NVFP4 ProtocolV2 output payload byte count overflow")
}

fn completion_slices_for_all_rows(row_count: usize) -> Vec<Vec<usize>> {
    (row_count > 0)
        .then(|| (0..row_count).collect())
        .into_iter()
        .collect()
}

fn validate_completion_slices(slices: &[Vec<usize>], row_count: usize) -> Result<()> {
    let mut seen = vec![false; row_count];
    for row_index in slices.iter().flatten().copied() {
        anyhow::ensure!(
            row_index < row_count,
            "completion slice row {row_index} exceeds row count {row_count}"
        );
        anyhow::ensure!(
            !seen[row_index],
            "completion slice row {row_index} was emitted more than once"
        );
        seen[row_index] = true;
    }
    anyhow::ensure!(
        seen.iter().all(|emitted| *emitted),
        "completion slices did not emit every response row"
    );
    Ok(())
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ProjectionRowsKey {
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
}

#[derive(Clone, Debug)]
struct RouteProjectionPreloadSpec {
    layer_id: usize,
    expert_id: usize,
    projection: &'static str,
    row_count: usize,
}

#[derive(Default)]
struct ProjectionRowsCache {
    rows: HashMap<ProjectionRowsKey, usize>,
    loads: usize,
    hits: usize,
    evictions: usize,
    active_layer: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg_attr(not(test), allow(dead_code))]
struct ProjectionRowsCacheStats {
    entries: usize,
    loads: usize,
    hits: usize,
    evictions: usize,
    active_layer: Option<usize>,
}

impl ProjectionRowsCache {
    fn prepare_layer(&mut self, layer_id: usize) {
        self.active_layer = Some(layer_id);
    }

    fn get(&mut self, key: &ProjectionRowsKey) -> Option<usize> {
        let rows = self.rows.get(key).copied();
        if rows.is_some() {
            self.hits += 1;
        }
        rows
    }

    fn insert(&mut self, key: ProjectionRowsKey, rows: usize) {
        self.loads += 1;
        self.rows.insert(key, rows);
    }

    fn stats(&self) -> ProjectionRowsCacheStats {
        ProjectionRowsCacheStats {
            entries: self.rows.len(),
            loads: self.loads,
            hits: self.hits,
            evictions: self.evictions,
            active_layer: self.active_layer,
        }
    }
}

impl RealNvfp4ProtocolV2Executor {
    fn validate_route_owner(&self, layer_id: usize, expert_id: usize) -> Result<()> {
        if self.intermediate_shard.is_some() {
            // Strict TP shards all execute every selected expert. Placement
            // ownership is neither consulted nor materialized on this path.
            return Ok(());
        }
        let owner = if let Some(owner_lookup) = &self.owner_lookup {
            owner_lookup
                .owner_for(layer_id, expert_id)
                .with_context(|| {
                    format!(
                        "real NVFP4 ProtocolV2 executor loadplan has no owner for layer {layer_id} expert {expert_id}"
                    )
                })?
                .to_owned()
        } else {
            let hosts = EXPERT_HOSTS
                .iter()
                .map(|host| (*host).to_owned())
                .collect::<Vec<_>>();
            owner_for_expert(layer_id, expert_id, &hosts, PlacementPolicy::Modulo)
                .context("real NVFP4 ProtocolV2 executor has no expert hosts")?
        };
        if let Some(role_hostname) = &self.role_hostname {
            if !host_matches(&owner, role_hostname) {
                bail!(
                    "real NVFP4 ProtocolV2 executor role {role_hostname} cannot serve layer {layer_id} expert {expert_id}, owned by {owner}"
                );
            }
        }
        Ok(())
    }

    fn projection_rows_cached(
        &self,
        layer_id: usize,
        expert_id: usize,
        projection: &'static str,
    ) -> Result<usize> {
        let key = ProjectionRowsKey {
            layer_id,
            expert_id,
            projection,
        };
        let mut cache = self
            .projection_rows_cache
            .lock()
            .map_err(|_| anyhow::anyhow!("real NVFP4 projection row cache is poisoned"))?;
        cache.prepare_layer(layer_id);
        if let Some(rows) = cache.get(&key) {
            return Ok(rows);
        }

        let full_rows = projection_rows(&self.catalog, layer_id, expert_id, projection)?;
        let rows = if matches!(projection, "gate_proj" | "up_proj") {
            self.intermediate_shard
                .map(|shard| shard.local_rows(full_rows))
                .transpose()?
                .unwrap_or(full_rows)
        } else {
            full_rows
        };
        cache.insert(key, rows);
        Ok(rows)
    }
}

fn host_matches(assignment_owner: &str, requested_owner: &str) -> bool {
    assignment_owner == requested_owner
        || assignment_owner.split('.').next() == Some(requested_owner)
        || requested_owner.split('.').next() == Some(assignment_owner)
}

fn projection_rows(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
    projection: &str,
) -> Result<usize> {
    if is_glm_exl3_recipe(&catalog.facts.quantization_recipe)
        && layer_id < catalog.facts.num_hidden_layers
    {
        return match projection {
            "gate_proj" | "up_proj" => Ok(GLM52_MOE_INTERMEDIATE_SIZE),
            "down_proj" => Ok(catalog.facts.hidden_size),
            other => bail!("unsupported GLM-5 EXL3 projection {other}"),
        };
    }
    let tensor_name =
        format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}.weight");
    let tensor = catalog_tensor_opt(catalog, &tensor_name)
        .with_context(|| format!("missing real NVFP4 projection tensor {tensor_name}"))?;
    let rows = tensor.shape.first().copied().with_context(|| {
        format!("real NVFP4 projection tensor {tensor_name} is missing row dimension")
    })?;
    if rows == 0 {
        bail!("real NVFP4 projection tensor {tensor_name} has zero rows");
    }
    Ok(rows)
}

fn routed_projection_preload_specs(
    catalog: &TensorCatalog,
    intermediate_shard: Option<ExpertIntermediateShard>,
) -> Result<Vec<RouteProjectionPreloadSpec>> {
    let mut specs = Vec::new();
    let mut seen = BTreeSet::new();
    for tensor in &catalog.tensors {
        let Some(mut spec) = routed_projection_preload_spec(tensor)? else {
            continue;
        };
        if matches!(spec.projection, "gate_proj" | "up_proj") {
            if let Some(shard) = intermediate_shard {
                spec.row_count = shard.local_rows(spec.row_count)?;
            }
        }
        if seen.insert((spec.layer_id, spec.expert_id, spec.projection)) {
            specs.push(spec);
        }
    }
    if specs.is_empty() {
        bail!("real NVFP4 resident preload found no routed expert projection weights");
    }
    specs.sort_by_key(|spec| (spec.layer_id, spec.expert_id, spec.projection));
    Ok(specs)
}

fn retained_bf16_projection_spec(
    catalog: &TensorCatalog,
    spec: &RouteProjectionPreloadSpec,
) -> Result<bool> {
    if !retained_bf16_layer(catalog, spec.layer_id)? {
        return Ok(false);
    }
    let tensor_name = format!(
        "model.layers.{}.mlp.experts.{}.{}.weight",
        spec.layer_id, spec.expert_id, spec.projection
    );
    let tensor = catalog_tensor(catalog, &tensor_name)?;
    anyhow::ensure!(
        tensor.dtype == DType::Bf16,
        "retained BF16 layer {} has non-BF16 projection {tensor_name}: {:?}",
        spec.layer_id,
        tensor.dtype
    );
    Ok(true)
}

fn startup_quantized_mtp_projection_spec(
    catalog: &TensorCatalog,
    spec: &RouteProjectionPreloadSpec,
) -> Result<bool> {
    if spec.layer_id != GLM52_MTP_LAYER_ID || retained_bf16_layer(catalog, spec.layer_id)? {
        return Ok(false);
    }
    let tensor_name = format!(
        "model.layers.{}.mlp.experts.{}.{}.weight",
        spec.layer_id, spec.expert_id, spec.projection
    );
    let tensor = catalog_tensor(catalog, &tensor_name)?;
    match tensor.dtype {
        DType::Bf16 => Ok(true),
        DType::F8E4M3 => {
            anyhow::ensure!(
                tensor.shape.len() == 2,
                "block-FP8 MTP projection {tensor_name} must be rank 2, got {:?}",
                tensor.shape
            );
            let scale_name = format!(
                "model.layers.{}.mlp.experts.{}.{}.weight_scale_inv",
                spec.layer_id, spec.expert_id, spec.projection
            );
            let scale = catalog_tensor(catalog, &scale_name)?;
            let expected_scale_shape =
                vec![tensor.shape[0].div_ceil(128), tensor.shape[1].div_ceil(128)];
            anyhow::ensure!(
                scale.dtype == DType::F32 && scale.shape == expected_scale_shape,
                "block-FP8 MTP inverse scale {scale_name} must be F32 {:?}, got {:?} {:?}",
                expected_scale_shape,
                scale.dtype,
                scale.shape,
            );
            Ok(true)
        }
        _ => Ok(false),
    }
}

fn exl3_base_projection_spec(catalog: &TensorCatalog, spec: &RouteProjectionPreloadSpec) -> bool {
    is_glm_exl3_recipe(&catalog.facts.quantization_recipe)
        && spec.layer_id < catalog.facts.num_hidden_layers
}

fn retained_bf16_layer(catalog: &TensorCatalog, layer_id: usize) -> Result<bool> {
    if layer_id != GLM52_MTP_LAYER_ID || !mtp_bf16_experts_enabled()? {
        return Ok(false);
    }
    let gate_name = format!("model.layers.{layer_id}.mlp.experts.0.gate_proj.weight");
    let gate = catalog_tensor(catalog, &gate_name)?;
    match gate.dtype {
        DType::Bf16 => Ok(true),
        DType::U8 | DType::F4 | DType::F8E4M3 => Ok(false),
        ref dtype => bail!(
            "layer {layer_id} MTP expert serving only supports BF16, block-FP8, or packed NVFP4 sources, got {dtype:?} for {gate_name}"
        ),
    }
}

fn resident_preload_plan_for_specs(
    catalog: &TensorCatalog,
    specs: &[RouteProjectionPreloadSpec],
    intermediate_shard: Option<ExpertIntermediateShard>,
) -> Result<RealNvfp4ResidentPreloadPlan> {
    let tensor_by_name = catalog
        .tensors
        .iter()
        .map(|tensor| (tensor.name.as_str(), tensor))
        .collect::<HashMap<_, _>>();
    let find_tensor = |name: &str| tensor_by_name.get(name).copied();
    let require_tensor = |name: &str| {
        find_tensor(name).with_context(|| format!("tensor {name} not found in catalog"))
    };
    let mut layers = BTreeSet::new();
    let mut projection_sets = BTreeMap::<(usize, usize), BTreeSet<&'static str>>::new();
    let mut plan = RealNvfp4ResidentPreloadPlan {
        startup_required: true,
        projection_groups: specs.len(),
        ..Default::default()
    };
    for spec in specs {
        layers.insert(spec.layer_id);
        projection_sets
            .entry((spec.layer_id, spec.expert_id))
            .or_default()
            .insert(spec.projection);
        let base_name = format!(
            "model.layers.{}.mlp.experts.{}.{}",
            spec.layer_id, spec.expert_id, spec.projection
        );
        let shard_count = intermediate_shard.map_or(1_u64, |shard| shard.count as u64);
        if is_glm_exl3_recipe(&catalog.facts.quantization_recipe)
            && spec.layer_id < catalog.facts.num_hidden_layers
        {
            let trellis = require_tensor(&format!("{base_name}.trellis"))?;
            let suh = require_tensor(&format!("{base_name}.suh"))?;
            let svh = require_tensor(&format!("{base_name}.svh"))?;
            let mcg = require_tensor(&format!("{base_name}.mcg"))?;
            anyhow::ensure!(
                trellis.dtype == DType::I16
                    && suh.dtype == DType::F16
                    && svh.dtype == DType::F16
                    && mcg.dtype == DType::I32,
                "GLM-5 EXL3 projection {base_name} has an invalid native tensor dtype"
            );
            plan.weight_bytes += trellis.byte_length / shard_count;
            match spec.projection {
                "gate_proj" | "up_proj" => {
                    // H-side input rotations are replicated; I-side output
                    // rotations follow the TP4 intermediate shard.
                    plan.weight_scale_bytes += suh.byte_length + svh.byte_length / shard_count;
                }
                "down_proj" => {
                    // The down input rotation follows I while the H-side
                    // output rotation is replicated.
                    plan.weight_scale_bytes += suh.byte_length / shard_count + svh.byte_length;
                }
                other => bail!("unsupported GLM-5 EXL3 projection {other}"),
            }
            // MCG is a validated recipe marker. The runtime keeps one codebook
            // LUT, not 768 duplicate scalar tensors per layer.
            plan.scalar_metadata_bytes += mcg.byte_length;
            continue;
        }
        let weight_name = format!("{base_name}.weight");
        let weight_scale_name = format!("{base_name}.weight_scale");
        let input_scale_name = format!("{base_name}.input_scale");
        let weight_scale_2_name = format!("{base_name}.weight_scale_2");
        let weight = require_tensor(&weight_name)?;
        if weight.dtype == DType::Bf16 && spec.layer_id == GLM52_MTP_LAYER_ID {
            let retained_bf16 = retained_bf16_projection_spec(catalog, spec)?;
            if retained_bf16 {
                plan.weight_bytes += weight.byte_length / shard_count;
            } else {
                // BF16 has two bytes per source element; packed NVFP4 has one
                // byte per pair and one E4M3 scale byte per 16 elements.
                plan.weight_bytes += weight.byte_length / 4 / shard_count;
                plan.weight_scale_bytes += weight.byte_length / 32 / shard_count;
                plan.scalar_metadata_bytes += 2 * std::mem::size_of::<f32>() as u64;
            }
            continue;
        }
        if weight.dtype == DType::F8E4M3 && spec.layer_id == GLM52_MTP_LAYER_ID {
            anyhow::ensure!(
                startup_quantized_mtp_projection_spec(catalog, spec)?,
                "block-FP8 MTP projection {base_name} does not satisfy the startup NVFP4 conversion contract"
            );
            // The source stores one E4M3 byte per value. The one-resident-copy
            // serving representation stores two E2M1 values per byte plus one
            // E4M3 scale per 16 values; weight_scale_inv is startup-only.
            plan.weight_bytes += weight.byte_length / 2 / shard_count;
            plan.weight_scale_bytes += weight.byte_length / 16 / shard_count;
            plan.scalar_metadata_bytes += 2 * std::mem::size_of::<f32>() as u64;
            continue;
        }
        plan.weight_bytes += weight.byte_length / shard_count;
        if let Some(weight_scale) = find_tensor(&weight_scale_name) {
            plan.weight_scale_bytes += weight_scale.byte_length / shard_count;
        } else {
            plan.missing_metadata_tensors += 1;
        }
        if let Some(input_scale) = find_tensor(&input_scale_name) {
            plan.scalar_metadata_bytes += input_scale.byte_length;
        } else {
            plan.missing_metadata_tensors += 1;
        }
        if let Some(weight_scale_2) = find_tensor(&weight_scale_2_name) {
            plan.scalar_metadata_bytes += weight_scale_2.byte_length;
        } else {
            plan.missing_metadata_tensors += 1;
        }
    }
    plan.layers = layers.len();
    plan.experts = projection_sets.len();
    plan.complete_expert_projection_sets = projection_sets
        .values()
        .filter(|projections| {
            projections.contains("gate_proj")
                && projections.contains("up_proj")
                && projections.contains("down_proj")
        })
        .count();
    plan.incomplete_expert_projection_sets = plan
        .experts
        .saturating_sub(plan.complete_expert_projection_sets);
    Ok(plan)
}

fn routed_projection_preload_spec(
    tensor: &TensorInfo,
) -> Result<Option<RouteProjectionPreloadSpec>> {
    if tensor.role != TensorRole::RoutedExpert || tensor.is_quantization_metadata {
        return Ok(None);
    }
    let (Some(layer_id), Some(expert_id)) = (tensor.layer_id, tensor.expert_id) else {
        return Ok(None);
    };
    let projection = routed_projection_name_from_weight_tensor(tensor, layer_id, expert_id)
        .or_else(|| routed_projection_name_from_exl3_trellis(tensor, layer_id, expert_id));
    let Some(projection) = projection else {
        return Ok(None);
    };
    let row_count = if tensor.name.ends_with(".trellis") {
        match projection {
            "gate_proj" | "up_proj" => GLM52_MOE_INTERMEDIATE_SIZE,
            "down_proj" => GLM52_HIDDEN_SIZE,
            _ => unreachable!("projection parser returned a known projection"),
        }
    } else {
        tensor.shape.first().copied().with_context(|| {
            format!(
                "real NVFP4 resident preload tensor {} has no row dimension",
                tensor.name
            )
        })?
    };
    if row_count == 0 {
        bail!(
            "real NVFP4 resident preload tensor {} has zero projection rows",
            tensor.name
        );
    }
    Ok(Some(RouteProjectionPreloadSpec {
        layer_id: layer_id as usize,
        expert_id: expert_id as usize,
        projection,
        row_count,
    }))
}

fn routed_projection_name_from_exl3_trellis(
    tensor: &TensorInfo,
    layer_id: u32,
    expert_id: u32,
) -> Option<&'static str> {
    let prefix = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.");
    let suffix = tensor
        .name
        .strip_prefix(&prefix)?
        .strip_suffix(".trellis")?;
    match suffix {
        "gate_proj" => Some("gate_proj"),
        "up_proj" => Some("up_proj"),
        "down_proj" => Some("down_proj"),
        _ => None,
    }
}

fn routed_projection_name_from_weight_tensor(
    tensor: &TensorInfo,
    layer_id: u32,
    expert_id: u32,
) -> Option<&'static str> {
    let prefix = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.");
    let suffix = tensor.name.strip_prefix(&prefix)?.strip_suffix(".weight")?;
    match suffix {
        "gate_proj" => Some("gate_proj"),
        "up_proj" => Some("up_proj"),
        "down_proj" => Some("down_proj"),
        _ => None,
    }
}

fn catalog_tensor_opt<'a>(catalog: &'a TensorCatalog, tensor_name: &str) -> Option<&'a TensorInfo> {
    catalog
        .tensors
        .binary_search_by(|tensor| tensor.name.as_str().cmp(tensor_name))
        .ok()
        .and_then(|index| catalog.tensors.get(index))
        .or_else(|| {
            catalog
                .tensors
                .iter()
                .find(|tensor| tensor.name == tensor_name)
        })
}

fn catalog_tensor<'a>(catalog: &'a TensorCatalog, tensor_name: &str) -> Result<&'a TensorInfo> {
    catalog_tensor_opt(catalog, tensor_name)
        .with_context(|| format!("tensor {tensor_name} not found in catalog"))
}

fn bf16_row_to_f32(row_bytes: &[u8], hidden_dim: usize) -> Result<Vec<f32>> {
    let logical_len = hidden_dim
        .checked_mul(ExpertV2Dtype::Bf16.bytes_per_element())
        .context("real NVFP4 ProtocolV2 BF16 row byte count overflow")?;
    if row_bytes.len() < logical_len {
        bail!(
            "real NVFP4 ProtocolV2 BF16 row has {} bytes, needs {logical_len}",
            row_bytes.len()
        );
    }
    Ok(row_bytes[..logical_len]
        .chunks_exact(2)
        .map(|chunk| {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect())
}

fn f32_values_to_bf16_bytes(values: &[f32], out: &mut [u8]) {
    for (value, dst) in values.iter().zip(out.chunks_exact_mut(2)) {
        let bf16 = (value.to_bits() >> 16) as u16;
        dst.copy_from_slice(&bf16.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        completion_slices_for_all_rows, parse_real_nvfp4_protocol_v2_packed_direct_max_rows,
        real_nvfp4_cuda_reference_kernels_enabled, regular_response_device_target,
        resident_preload_plan_for_specs, routed_projection_preload_specs,
        startup_quantized_mtp_projection_spec, validate_completion_slices,
        RealNvfp4ProtocolV2Executor, StreamedIngressState, REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
    };
    use crate::cli::ExpertDaemonArgs;
    use crate::commands::expertd::run_expertd;
    use crate::commands::model_artifacts::{read_expert_owner_lookup, ExpertOwnerLookup};
    use crate::commands::real_full::intermediate_sharding::{
        spark_expert_intermediate_shard_from_env, ExpertIntermediateShard,
    };
    use crate::commands::real_full::sparse_mlp::route::cuda_reference_kernels_test_override;
    use crate::commands::real_full::sparse_mlp::router::ScoredRoute;
    use glmrt_core::{
        DType, ExpertBatch, ExpertBatchRoute, ExpertHostBatch, ExpertHostBatchSet, GraphBucket,
        LayerId, ModelFacts, PlacementVersion, PositionId, RequestId, RowSourceKind, TensorCatalog,
        TensorInfo, TensorRole, EXPERT_HOSTS, GLM52_HIDDEN_SIZE, GLM52_MOE_INTERMEDIATE_SIZE,
        GLM52_MTP_LAYER_ID,
    };
    use glmrt_ffi::GlmrtDeviceBuffer;
    use glmrt_loader::{
        build_catalog_for_snapshot, validate_glm52_exl3_expert_catalog, GLM52_EXL3_RECIPE_K3_V1,
    };
    use glmrt_transport::{
        expert_protocol_v2_compact_id, serve_protocol_v2_tcp_listener_with_executor,
        tcp_protocol_v2_host_batch_set_bf16_dispatch, tcp_protocol_v2_roundtrip,
        ExpertProtocolV2Request, ExpertProtocolV2Response, ExpertProtocolV2ResponseView,
        ExpertProtocolV2RouteEntry, ExpertProtocolV2RowDescriptor, ExpertProtocolV2StreamPlan,
        ExpertProtocolV2StreamRouteGroup, ExpertV2Dtype, ExpertV2SourceKind,
        ProtocolV2ExpertExecutor, ProtocolV2RequestDevicePayload, TcpProtocolV2HostBatchTarget,
        TcpTransportConfig, EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN,
    };
    use std::{
        fs::File,
        io::Write,
        net::{SocketAddr, TcpListener as StdTcpListener},
        path::{Path, PathBuf},
        sync::Arc,
    };
    use tokio::{
        net::{TcpListener, TcpStream},
        time::{sleep, Duration},
    };

    #[test]
    fn packed_direct_max_rows_defaults_to_full_prefill_and_accepts_diagnostic_range() {
        assert_eq!(
            parse_real_nvfp4_protocol_v2_packed_direct_max_rows(None).unwrap(),
            2064
        );
        assert_eq!(
            parse_real_nvfp4_protocol_v2_packed_direct_max_rows(Some("8")).unwrap(),
            8
        );
        assert_eq!(
            parse_real_nvfp4_protocol_v2_packed_direct_max_rows(Some("64")).unwrap(),
            64
        );
    }

    #[test]
    fn packed_direct_max_rows_rejects_values_outside_diagnostic_range() {
        for value in ["7", "2065", "invalid"] {
            assert!(parse_real_nvfp4_protocol_v2_packed_direct_max_rows(Some(value)).is_err());
        }
    }

    #[test]
    fn completion_slice_contract_emits_each_row_once() {
        assert!(validate_completion_slices(&[vec![2], vec![0, 3], vec![1]], 4).is_ok());
        assert!(validate_completion_slices(&[vec![0], vec![0]], 1).is_err());
        assert!(validate_completion_slices(&[vec![1]], 1).is_err());
        assert_eq!(completion_slices_for_all_rows(3), vec![vec![0, 1, 2]]);
        assert!(completion_slices_for_all_rows(0).is_empty());
    }

    #[test]
    fn regular_device_response_target_follows_indexed_prefix() {
        let frame = tiny_request().encode().unwrap();
        let request = glmrt_transport::ExpertProtocolV2RequestView::parse(&frame).unwrap();
        let response_base = 0x20_0000_usize;
        let payload = ProtocolV2RequestDevicePayload {
            hidden_payload: GlmrtDeviceBuffer {
                ptr: 0x10_0000_usize as *mut std::ffi::c_void,
                bytes: request.hidden_payload().len(),
                device_id: 2,
                flags: 7,
            },
            response_slot: Some(GlmrtDeviceBuffer {
                ptr: response_base as *mut std::ffi::c_void,
                bytes: 4096,
                device_id: 2,
                flags: 7,
            }),
            execution_lane: 0,
        };

        let target = regular_response_device_target(&request, Some(payload), 4)
            .unwrap()
            .unwrap();

        assert_eq!(
            target.ptr as usize,
            response_base + EXPERT_PROTOCOL_V2_RESPONSE_HEADER_LEN + 4
        );
        assert_eq!(target.bytes, 4);
        assert_eq!(target.device_id, 2);
        assert_eq!(target.flags, 7);
    }

    #[test]
    fn streamed_ingress_state_restores_logical_hidden_row_order() {
        let rows = vec![
            stream_row(0, 0, 2),
            stream_row(1, 2, 1),
            stream_row(2, 3, 1),
        ];
        let routes = vec![
            stream_route(0, 5),
            stream_route(0, 9),
            stream_route(1, 5),
            stream_route(2, 9),
        ];
        let plan = ExpertProtocolV2StreamPlan::new(
            3,
            4,
            vec![1, 0, 2],
            vec![
                ExpertProtocolV2StreamRouteGroup {
                    ready_after_rows: 2,
                    route_indices: vec![2, 0],
                    completed_rows: vec![1],
                },
                ExpertProtocolV2StreamRouteGroup {
                    ready_after_rows: 3,
                    route_indices: vec![1, 3],
                    completed_rows: vec![0, 2],
                },
            ],
        )
        .unwrap();
        let plan_request = ExpertProtocolV2Request::new_stream_plan(
            42,
            7,
            3,
            2,
            ExpertV2Dtype::Bf16,
            rows,
            routes,
            plan.encode().unwrap(),
        )
        .unwrap()
        .with_fp8_e4m3_row_scaled_response();
        let plan_frame = plan_request.encode().unwrap();
        let plan_view = glmrt_transport::ExpertProtocolV2RequestView::parse(&plan_frame).unwrap();
        let mut state = StreamedIngressState::from_plan(&plan_view).unwrap();
        let first = ExpertProtocolV2Request::new_stream_data(
            42,
            7,
            3,
            2,
            ExpertV2Dtype::Bf16,
            4,
            0,
            2,
            vec![5, 6, 7, 8, 1, 2, 3, 4],
            false,
        )
        .unwrap()
        .with_fp8_e4m3_row_scaled_response();
        let second = ExpertProtocolV2Request::new_stream_data(
            42,
            7,
            3,
            2,
            ExpertV2Dtype::Bf16,
            4,
            2,
            1,
            vec![9, 10, 11, 12],
            true,
        )
        .unwrap()
        .with_fp8_e4m3_row_scaled_response();
        let first_frame = first.encode().unwrap();
        let second_frame = second.encode().unwrap();

        assert!(!state
            .accept_data(
                &glmrt_transport::ExpertProtocolV2RequestView::parse(&first_frame).unwrap()
            )
            .unwrap());
        assert!(state
            .accept_data(
                &glmrt_transport::ExpertProtocolV2RequestView::parse(&second_frame).unwrap()
            )
            .unwrap());
        let assembled = state.into_request().unwrap();
        assert_eq!(
            assembled.hidden_payload,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        );
        assert!(assembled.fp8_e4m3_row_scaled_response_enabled());
        assert!(!assembled.stream_plan_enabled());
        assert!(!assembled.stream_data_enabled());
    }

    fn stream_row(row: u64, route_offset: u32, route_count: u32) -> ExpertProtocolV2RowDescriptor {
        ExpertProtocolV2RowDescriptor {
            row_id: row,
            source_kind: ExpertV2SourceKind::Prefill,
            source_request_id: 10,
            token_position: row,
            route_offset,
            route_count,
        }
    }

    fn stream_route(row_index: u32, expert_id: u32) -> ExpertProtocolV2RouteEntry {
        ExpertProtocolV2RouteEntry {
            row_index,
            expert_id,
            gate_weight: 1.0,
        }
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_runs_tiny_checkpoint_route() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()));
        let request = tiny_request().with_debug_checksum();

        let response = execute_request(&executor, &request).unwrap();

        assert_eq!(response.header.request_id, request.header.request_id);
        assert_eq!(response.header.output_dim, request.header.hidden_dim);
        assert_eq!(response.partial_output_payload.len(), 4);
        let output = bf16_values(&response.partial_output_payload);
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
        assert!(output[0] != 0.0);
        assert!(output[1] != 0.0);
        let rows_stats = executor.projection_rows_cache.lock().unwrap().stats();
        assert_eq!(rows_stats.entries, 1);
        assert_eq!(rows_stats.loads, 1);
        assert_eq!(rows_stats.active_layer, Some(3));
        let encoded = response.encode().unwrap();
        ExpertProtocolV2ResponseView::parse(&encoded)
            .unwrap()
            .verify_checksum()
            .unwrap();
    }

    #[test]
    fn real_nvfp4_protocol_v2_streaming_fallback_emits_one_complete_response() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()));
        let request = tiny_request();
        let frame = request.encode().unwrap();
        let view = glmrt_transport::ExpertProtocolV2RequestView::parse(&frame).unwrap();
        let expected = executor.execute(&view).unwrap();
        let mut responses = Vec::new();

        executor
            .execute_streaming(&view, &mut |response| {
                responses.push(response.to_owned().unwrap());
                Ok(())
            })
            .unwrap();

        assert_eq!(responses, vec![expected]);
        assert!(!responses[0].row_indexed());
        assert!(!responses[0].more_chunks());
    }

    #[test]
    fn real_nvfp4_protocol_v2_streamed_ingress_matches_regular_execution() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()));
        let regular = tiny_request();
        let expected = execute_request(&executor, &regular).unwrap();
        let plan = ExpertProtocolV2StreamPlan::new(
            1,
            1,
            vec![0],
            vec![ExpertProtocolV2StreamRouteGroup {
                ready_after_rows: 1,
                route_indices: vec![0],
                completed_rows: vec![0],
            }],
        )
        .unwrap();
        let plan_request = ExpertProtocolV2Request::new_stream_plan(
            regular.header.request_id,
            regular.header.placement_version,
            regular.header.layer_id,
            regular.header.hidden_dim,
            regular.header.hidden_dtype,
            regular.rows.clone(),
            regular.routes.clone(),
            plan.encode().unwrap(),
        )
        .unwrap();
        let data_request = ExpertProtocolV2Request::new_stream_data(
            regular.header.request_id,
            regular.header.placement_version,
            regular.header.layer_id,
            regular.header.hidden_dim,
            regular.header.hidden_dtype,
            regular.header.hidden_row_stride_bytes,
            0,
            1,
            regular.hidden_payload.clone(),
            true,
        )
        .unwrap();
        let plan_frame = plan_request.encode().unwrap();
        let data_frame = data_request.encode().unwrap();
        let plan_view = glmrt_transport::ExpertProtocolV2RequestView::parse(&plan_frame).unwrap();
        let data_view = glmrt_transport::ExpertProtocolV2RequestView::parse(&data_frame).unwrap();
        let mut plan_responses = Vec::new();
        let mut data_responses = Vec::new();

        executor
            .execute_streaming(&plan_view, &mut |response| {
                plan_responses.push(response.to_owned().unwrap());
                Ok(())
            })
            .unwrap();
        executor
            .execute_streaming(&data_view, &mut |response| {
                data_responses.push(response.to_owned().unwrap());
                Ok(())
            })
            .unwrap();

        assert_eq!(plan_responses.len(), 1);
        assert_eq!(plan_responses[0].header.row_count, 0);
        assert_eq!(data_responses.len(), 1);
        assert!(data_responses[0].row_indexed());
        assert_eq!(data_responses[0].row_indices, Some(vec![0]));
        assert_eq!(
            data_responses[0].partial_output_payload,
            expected.partial_output_payload
        );
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_keeps_projection_caches_across_request_layers() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog_for_layers(tempdir.path(), &[3, 4]);
        let executor = RealNvfp4ProtocolV2Executor::new(catalog, None, None);
        let cuda_reference_enabled = real_nvfp4_cuda_reference_kernels_enabled();

        let layer3 = tiny_request_for_layer(3);
        execute_request(&executor, &layer3).unwrap();
        let layer3_rows = executor.projection_rows_cache.lock().unwrap().stats();
        assert_eq!(layer3_rows.entries, 1);
        assert_eq!(layer3_rows.loads, 1);
        assert_eq!(layer3_rows.hits, 0);
        assert_eq!(layer3_rows.evictions, 0);
        assert_eq!(layer3_rows.active_layer, Some(3));
        let layer3_routes = executor.route_caches[0].lock().unwrap().stats();
        if cuda_reference_enabled {
            assert_eq!(layer3_routes.entries, 0);
            assert_eq!(layer3_routes.projection_loads, 0);
            assert_eq!(layer3_routes.cuda_projection_entries, 3);
            assert_eq!(layer3_routes.cuda_projection_uploads, 3);
            assert_eq!(layer3_routes.cuda_cache_hits, 3);
            assert_eq!(layer3_routes.cuda_active_layer, Some(3));
        } else {
            assert_eq!(layer3_routes.entries, 3);
            assert_eq!(layer3_routes.projection_loads, 3);
            assert_eq!(layer3_routes.active_layer, Some(3));
        }
        assert_eq!(layer3_routes.projection_evictions, 0);
        assert_eq!(layer3_routes.cuda_projection_evictions, 0);

        let layer4 = tiny_request_for_layer(4);
        execute_request(&executor, &layer4).unwrap();
        let layer4_rows = executor.projection_rows_cache.lock().unwrap().stats();
        assert_eq!(layer4_rows.entries, 2);
        assert_eq!(layer4_rows.loads, 2);
        assert_eq!(layer4_rows.hits, 0);
        assert_eq!(layer4_rows.evictions, 0);
        assert_eq!(layer4_rows.active_layer, Some(4));
        let layer4_routes = executor.route_caches[0].lock().unwrap().stats();
        if cuda_reference_enabled {
            assert_eq!(layer4_routes.entries, 0);
            assert_eq!(layer4_routes.projection_loads, 0);
            assert_eq!(layer4_routes.cache_hits, 0);
            assert_eq!(layer4_routes.cuda_projection_entries, 6);
            assert_eq!(layer4_routes.cuda_projection_uploads, 6);
            assert_eq!(layer4_routes.cuda_cache_hits, 6);
            assert_eq!(layer4_routes.cuda_active_layer, Some(4));
        } else {
            assert_eq!(layer4_routes.entries, 6);
            assert_eq!(layer4_routes.projection_loads, 6);
            assert_eq!(layer4_routes.cache_hits, 0);
            assert_eq!(layer4_routes.active_layer, Some(4));
        }
        assert_eq!(layer4_routes.projection_evictions, 0);
        assert_eq!(layer4_routes.cuda_projection_evictions, 0);

        execute_request(&executor, &layer4).unwrap();
        let reused_rows = executor.projection_rows_cache.lock().unwrap().stats();
        assert_eq!(reused_rows.entries, 2);
        assert_eq!(reused_rows.loads, 2);
        assert_eq!(reused_rows.hits, 1);
        assert_eq!(reused_rows.evictions, 0);
        let reused_routes = executor.route_caches[0].lock().unwrap().stats();
        if cuda_reference_enabled {
            assert_eq!(reused_routes.entries, 0);
            assert_eq!(reused_routes.projection_loads, 0);
            assert_eq!(reused_routes.cache_hits, 0);
            assert_eq!(reused_routes.cuda_projection_entries, 6);
            assert_eq!(reused_routes.cuda_projection_uploads, 6);
            assert_eq!(reused_routes.cuda_cache_hits, 12);
            assert_eq!(reused_routes.cuda_active_layer, Some(4));
        } else {
            assert_eq!(reused_routes.entries, 6);
            assert_eq!(reused_routes.projection_loads, 6);
            assert_eq!(reused_routes.cache_hits, 3);
        }
        assert_eq!(reused_routes.projection_evictions, 0);
        assert_eq!(reused_routes.cuda_projection_evictions, 0);

        execute_request(&executor, &layer3).unwrap();
        let layer3_again_rows = executor.projection_rows_cache.lock().unwrap().stats();
        assert_eq!(layer3_again_rows.entries, 2);
        assert_eq!(layer3_again_rows.loads, 2);
        assert_eq!(layer3_again_rows.hits, 2);
        assert_eq!(layer3_again_rows.evictions, 0);
        assert_eq!(layer3_again_rows.active_layer, Some(3));
        let layer3_again_routes = executor.route_caches[0].lock().unwrap().stats();
        if cuda_reference_enabled {
            assert_eq!(layer3_again_routes.entries, 0);
            assert_eq!(layer3_again_routes.projection_loads, 0);
            assert_eq!(layer3_again_routes.cache_hits, 0);
            assert_eq!(layer3_again_routes.cuda_projection_entries, 6);
            assert_eq!(layer3_again_routes.cuda_projection_uploads, 6);
            assert_eq!(layer3_again_routes.cuda_cache_hits, 18);
            assert_eq!(layer3_again_routes.cuda_active_layer, Some(3));
        } else {
            assert_eq!(layer3_again_routes.entries, 6);
            assert_eq!(layer3_again_routes.projection_loads, 6);
            assert_eq!(layer3_again_routes.cache_hits, 6);
            assert_eq!(layer3_again_routes.active_layer, Some(3));
        }
        assert_eq!(layer3_again_routes.projection_evictions, 0);
        assert_eq!(layer3_again_routes.cuda_projection_evictions, 0);
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_preloads_assigned_projection_cache() {
        let _cuda_reference_override = cuda_reference_kernels_test_override(false);
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = preload_expert_catalog_for_layers(tempdir.path(), &[3, 4]);
        let executor = RealNvfp4ProtocolV2Executor::new(catalog, None, None);

        let preload = executor
            .preload_assigned_projections_with_cuda(false)
            .unwrap();

        assert_eq!(preload.projection_groups, 6);
        assert_eq!(preload.layers, 2);
        assert_eq!(preload.experts, 2);
        assert_eq!(preload.route_cache_entries, 6);
        assert_eq!(preload.route_cache_loads, 6);
        assert_eq!(preload.route_cache_hits, 0);
        assert_eq!(preload.projection_row_entries, 6);
        assert_eq!(preload.projection_row_loads, 6);
        assert_eq!(preload.projection_row_hits, 0);
        assert_eq!(preload.weight_bytes, 768);
        assert_eq!(preload.quant_metadata_bytes, 144);
        if preload.cuda_reference_enabled {
            assert_eq!(preload.cuda_projection_groups, 6);
            assert_eq!(preload.cuda_weight_bytes, 768);
            assert_eq!(preload.cuda_weight_scale_bytes, 96);
            assert_eq!(preload.cuda_projection_entries, 6);
            assert_eq!(preload.cuda_projection_uploads, 6);
            assert_eq!(preload.cuda_cache_hits, 0);
        } else {
            assert_eq!(preload.cuda_projection_groups, 0);
            assert_eq!(preload.cuda_weight_bytes, 0);
            assert_eq!(preload.cuda_weight_scale_bytes, 0);
            assert_eq!(preload.cuda_projection_entries, 0);
            assert_eq!(preload.cuda_projection_uploads, 0);
            assert_eq!(preload.cuda_cache_hits, 0);
        }

        let layer3 = preload_request_for_layer(3);
        execute_request(&executor, &layer3).unwrap();
        let route_stats = executor.route_caches[0].lock().unwrap().stats();
        assert_eq!(route_stats.entries, 6);
        assert_eq!(route_stats.projection_loads, 6);
        if preload.cuda_reference_enabled {
            assert_eq!(route_stats.cache_hits, 0);
            assert_eq!(route_stats.cuda_projection_entries, 6);
            assert_eq!(route_stats.cuda_projection_uploads, 6);
            assert_eq!(route_stats.cuda_cache_hits, 3);
            assert_eq!(route_stats.cuda_active_layer, Some(3));
            assert_eq!(route_stats.cuda_graph_entries, 0);
            assert_eq!(route_stats.cuda_graph_captures, 0);
            assert_eq!(route_stats.cuda_graph_launches, 0);
        } else {
            assert_eq!(route_stats.cache_hits, 3);
        }
        assert_eq!(route_stats.projection_evictions, 0);
        let row_stats = executor.projection_rows_cache.lock().unwrap().stats();
        assert_eq!(row_stats.entries, 6);
        assert_eq!(row_stats.loads, 6);
        assert_eq!(row_stats.hits, 1);
        assert_eq!(row_stats.evictions, 0);
    }

    #[test]
    fn exl3_trellis_drives_logical_preload_without_nvfp4_weight_fallback() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = preload_exl3_catalog(tempdir.path());
        let shard = ExpertIntermediateShard::new(4, 0).unwrap();

        let specs = routed_projection_preload_specs(&catalog, Some(shard)).unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(
            specs
                .iter()
                .map(|spec| (spec.projection, spec.row_count))
                .collect::<Vec<_>>(),
            vec![
                ("down_proj", GLM52_HIDDEN_SIZE),
                ("gate_proj", GLM52_MOE_INTERMEDIATE_SIZE / 4),
                ("up_proj", GLM52_MOE_INTERMEDIATE_SIZE / 4),
            ]
        );

        // This catalog deliberately has no NVFP4 `.weight` tensors. A valid
        // plan therefore also proves that format selection stayed on EXL3.
        let plan = resident_preload_plan_for_specs(&catalog, &specs, Some(shard)).unwrap();
        assert_eq!(plan.projection_groups, 3);
        assert_eq!(plan.layers, 1);
        assert_eq!(plan.experts, 1);
        assert_eq!(plan.complete_expert_projection_sets, 1);
        assert_eq!(plan.incomplete_expert_projection_sets, 0);
        assert_eq!(plan.missing_metadata_tensors, 0);
        assert_eq!(plan.weight_bytes, 48);
        assert_eq!(plan.weight_scale_bytes, 96);
        assert_eq!(plan.scalar_metadata_bytes, 12);
    }

    #[test]
    fn glm53_block_fp8_mtp_plans_one_packed_nvfp4_resident_copy() {
        let mut tensors = Vec::new();
        for (projection, shape) in [
            (
                "gate_proj",
                vec![GLM52_MOE_INTERMEDIATE_SIZE, GLM52_HIDDEN_SIZE],
            ),
            (
                "up_proj",
                vec![GLM52_MOE_INTERMEDIATE_SIZE, GLM52_HIDDEN_SIZE],
            ),
            (
                "down_proj",
                vec![GLM52_HIDDEN_SIZE, GLM52_MOE_INTERMEDIATE_SIZE],
            ),
        ] {
            let values = shape.iter().product::<usize>();
            tensors.push(TensorInfo {
                name: format!(
                    "model.layers.{GLM52_MTP_LAYER_ID}.mlp.experts.0.{projection}.weight"
                ),
                file: "mtp.safetensors".to_owned(),
                dtype: DType::F8E4M3,
                shape: shape.clone(),
                byte_offset: 0,
                byte_length: values as u64,
                role: TensorRole::RoutedExpert,
                layer_id: Some(GLM52_MTP_LAYER_ID as u32),
                expert_id: Some(0),
                is_quantization_metadata: false,
            });
            let scale_shape = vec![shape[0].div_ceil(128), shape[1].div_ceil(128)];
            tensors.push(TensorInfo {
                name: format!(
                    "model.layers.{GLM52_MTP_LAYER_ID}.mlp.experts.0.{projection}.weight_scale_inv"
                ),
                file: "mtp.safetensors".to_owned(),
                dtype: DType::F32,
                shape: scale_shape.clone(),
                byte_offset: 0,
                byte_length: (scale_shape.iter().product::<usize>() * 4) as u64,
                role: TensorRole::RoutedExpert,
                layer_id: Some(GLM52_MTP_LAYER_ID as u32),
                expert_id: Some(0),
                is_quantization_metadata: true,
            });
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        let catalog = TensorCatalog {
            model_id: "zai-org/GLM-5.3".to_owned(),
            snapshot_path: "/tmp/glm53".to_owned(),
            facts: ModelFacts {
                hidden_size: GLM52_HIDDEN_SIZE,
                num_hidden_layers: GLM52_MTP_LAYER_ID,
                ..ModelFacts::default()
            },
            tensors,
        };
        let shard = ExpertIntermediateShard::new(4, 0).unwrap();
        let specs = routed_projection_preload_specs(&catalog, Some(shard)).unwrap();
        assert_eq!(specs.len(), 3);
        assert!(specs
            .iter()
            .all(|spec| startup_quantized_mtp_projection_spec(&catalog, spec).unwrap()));

        let plan = resident_preload_plan_for_specs(&catalog, &specs, Some(shard)).unwrap();
        let source_values = GLM52_MOE_INTERMEDIATE_SIZE * GLM52_HIDDEN_SIZE;
        assert_eq!(plan.weight_bytes, (3 * source_values / 2 / 4) as u64);
        assert_eq!(plan.weight_scale_bytes, (3 * source_values / 16 / 4) as u64);
        assert_eq!(plan.scalar_metadata_bytes, 3 * 2 * 4);
        assert_eq!(plan.missing_metadata_tensors, 0);
        assert_eq!(plan.complete_expert_projection_sets, 1);
    }

    #[test]
    #[ignore = "requires the pinned zai-org/GLM-5.3 source snapshot"]
    fn validates_installed_glm53_block_fp8_mtp_catalog() {
        let snapshot = std::env::var_os("GLMRT_TEST_GLM53_SOURCE_SNAPSHOT")
            .map(PathBuf::from)
            .expect("set GLMRT_TEST_GLM53_SOURCE_SNAPSHOT to the pinned snapshot");
        let catalog = build_catalog_for_snapshot("zai-org/GLM-5.3", &snapshot).unwrap();
        let shard = ExpertIntermediateShard::new(4, 0).unwrap();
        let mtp_specs = routed_projection_preload_specs(&catalog, Some(shard))
            .unwrap()
            .into_iter()
            .filter(|spec| spec.layer_id == GLM52_MTP_LAYER_ID)
            .collect::<Vec<_>>();
        assert_eq!(mtp_specs.len(), 256 * 3);
        assert!(mtp_specs
            .iter()
            .all(|spec| startup_quantized_mtp_projection_spec(&catalog, spec).unwrap()));
        let plan = resident_preload_plan_for_specs(&catalog, &mtp_specs, Some(shard)).unwrap();
        assert_eq!(plan.layers, 1);
        assert_eq!(plan.experts, 256);
        assert_eq!(plan.complete_expert_projection_sets, 256);
        assert_eq!(plan.missing_metadata_tensors, 0);
        assert_eq!(plan.weight_bytes, 1_207_959_552);
        assert_eq!(plan.weight_scale_bytes, 150_994_944);
    }

    #[test]
    fn calibrated_exl3_checkpoint_preloads_and_executes_packed_protocol_v2_when_requested() {
        const CATALOG_ENV: &str = "GLMRT_EXL3_RUNTIME_TEST_CATALOG";
        let Some(catalog_path) = std::env::var_os(CATALOG_ENV).map(PathBuf::from) else {
            eprintln!("skipped: set {CATALOG_ENV} to a one-layer EXL3 projection catalog");
            return;
        };
        let catalog: TensorCatalog = serde_json::from_reader(
            File::open(&catalog_path)
                .unwrap_or_else(|error| panic!("opening {}: {error}", catalog_path.display())),
        )
        .unwrap_or_else(|error| panic!("parsing {}: {error}", catalog_path.display()));
        let summary = validate_glm52_exl3_expert_catalog(&catalog)
            .expect("validating the calibrated one-layer EXL3 catalog");
        assert_eq!(summary.base_routed_layers, 1);
        assert_eq!(summary.experts_per_layer, 256);
        assert_eq!(summary.expert_tensors, 256 * 3 * 4);

        let layer_id = catalog.facts.first_k_dense_replace;
        let shard = spark_expert_intermediate_shard_from_env()
            .expect("parsing the calibrated EXL3 TP shard environment")
            .expect("calibrated EXL3 hardware test requires a TP shard environment");
        assert_eq!(shard.count, 4);
        let _cuda_reference_override = cuda_reference_kernels_test_override(true);
        let executor = RealNvfp4ProtocolV2Executor::new(catalog, Some(layer_id), None)
            .with_intermediate_shard(shard);
        let preload = executor
            .preload_assigned_projections_with_cuda(true)
            .expect("preloading the calibrated EXL3 layer into its TP4 resident slab");
        assert_eq!(preload.layers, 1);
        assert_eq!(preload.experts, 256);
        assert_eq!(preload.projection_groups, 256 * 3);
        assert_eq!(preload.cuda_projection_groups, 256 * 3);
        assert_eq!(preload.cuda_projection_entries, 256 * 3);

        let rows = vec![ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 0xE3,
            token_position: 0,
            route_offset: 0,
            route_count: 8,
        }];
        let routes = (0..8)
            .map(|expert_id| ExpertProtocolV2RouteEntry {
                row_index: 0,
                expert_id,
                gate_weight: 1.0 / 8.0,
            })
            .collect::<Vec<_>>();
        let hidden_row_bytes = ExpertV2Dtype::Nvfp4E2m1Fp8E4m3
            .row_bytes(GLM52_HIDDEN_SIZE)
            .unwrap();
        let packed_bytes = GLM52_HIDDEN_SIZE / 2;
        let scale_bytes = GLM52_HIDDEN_SIZE / 16;
        assert_eq!(hidden_row_bytes, packed_bytes + scale_bytes);
        let mut hidden_payload = vec![0_u8; hidden_row_bytes];
        for (packed_index, packed) in hidden_payload[..packed_bytes].iter_mut().enumerate() {
            let code = |lane: usize| {
                let pattern = packed_index.wrapping_mul(2).wrapping_add(lane);
                let magnitude = 1 + pattern % 7;
                (magnitude | (((pattern / 7) & 1) << 3)) as u8
            };
            *packed = code(0) | (code(1) << 4);
        }
        hidden_payload[packed_bytes..].fill(0x18);
        let request = ExpertProtocolV2Request::new(
            0xE3,
            0x51CE,
            layer_id as u32,
            GLM52_HIDDEN_SIZE as u32,
            ExpertV2Dtype::Nvfp4E2m1Fp8E4m3,
            rows,
            routes,
            hidden_payload,
        )
        .unwrap();
        let response = execute_request(&executor, &request)
            .expect("executing the packed EXL3 ProtocolV2 route");
        assert_eq!(response.header.request_id, request.header.request_id);
        assert_eq!(response.header.output_dim as usize, GLM52_HIDDEN_SIZE);
        assert_eq!(response.header.output_dtype, ExpertV2Dtype::Bf16);
        assert_eq!(
            response.partial_output_payload.len(),
            GLM52_HIDDEN_SIZE * std::mem::size_of::<u16>()
        );
        let output = bf16_values(&response.partial_output_payload);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
        let checksum = output
            .iter()
            .enumerate()
            .map(|(index, value)| *value as f64 * (1 + index % 251) as f64)
            .sum::<f64>();
        eprintln!(
            "calibrated_exl3_protocol_v2_rank_pass tp_rank={} output_nonzero={} weighted_checksum={checksum:.9e}",
            shard.rank,
            output.iter().filter(|value| **value != 0.0).count(),
        );
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_preloads_cuda_projection_cache_when_available() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = preload_expert_catalog_for_layers(tempdir.path(), &[3, 4]);
        let executor = RealNvfp4ProtocolV2Executor::new(catalog, None, None);

        let preload = match executor.preload_assigned_projections_with_cuda(true) {
            Ok(preload) => preload,
            Err(error) => {
                let message = format!("{error:#}");
                if message.contains("requires GLMRT_NATIVE_LIB")
                    || message.contains("loading native CUDA")
                    || message.contains("cuda unavailable")
                    || message.contains("returned status 3")
                {
                    eprintln!("skipping CUDA projection preload test: {message}");
                    return;
                }
                panic!("unexpected CUDA projection preload error: {message}");
            }
        };

        assert!(preload.cuda_reference_enabled);
        assert_eq!(preload.cuda_projection_groups, 6);
        assert_eq!(preload.cuda_weight_bytes, 768);
        assert_eq!(preload.cuda_weight_scale_bytes, 96);
        assert_eq!(preload.cuda_projection_entries, 6);
        assert_eq!(preload.cuda_projection_uploads, 6);
        assert_eq!(preload.cuda_cache_hits, 0);
        assert_eq!(preload.route_cache_entries, 0);
        assert_eq!(preload.route_cache_loads, 0);
        assert_eq!(preload.route_cache_hits, 0);
        assert_eq!(preload.weight_bytes, 0);
        assert_eq!(preload.quant_metadata_bytes, 48);

        let second = executor
            .preload_assigned_projections_with_cuda(true)
            .expect("second CUDA preload should reuse resident device projections");
        assert_eq!(second.route_cache_loads, 0);
        assert_eq!(second.route_cache_hits, 0);
        assert_eq!(second.cuda_projection_entries, 6);
        assert_eq!(second.cuda_projection_uploads, 6);
        assert_eq!(second.cuda_cache_hits, 6);
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_enforces_role_hostname() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-1".to_owned()));
        let request = tiny_request();

        let error = execute_request(&executor, &request)
            .unwrap_err()
            .to_string();

        assert!(error.contains("cannot serve layer 3 expert 0"));
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_uses_loadplan_owner_lookup() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let owner_lookup = ExpertOwnerLookup::from_pairs([((3, 0), "spark-1".to_owned())]);
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-1".to_owned()))
                .with_owner_lookup(owner_lookup);
        let request = tiny_request();

        let response = execute_request(&executor, &request).unwrap();

        assert_eq!(response.header.request_id, request.header.request_id);
        assert!(bf16_values(&response.partial_output_payload)
            .iter()
            .any(|value| *value != 0.0));
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_rejects_unassigned_loadplan_route() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let owner_lookup = ExpertOwnerLookup::from_pairs([((3, 1), "spark-0".to_owned())]);
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup);
        let request = tiny_request();

        let error = execute_request(&executor, &request)
            .unwrap_err()
            .to_string();

        assert!(error.contains("loadplan has no owner for layer 3 expert 0"));
    }

    #[test]
    fn real_nvfp4_protocol_v2_executor_accepts_loadplan_hostname_alias() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let owner_lookup =
            ExpertOwnerLookup::from_pairs([((3, 0), "spark-0.spark.local".to_owned())]);
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup);
        let request = tiny_request();

        let response = execute_request(&executor, &request).unwrap();

        assert_eq!(response.header.request_id, request.header.request_id);
    }

    #[tokio::test]
    async fn real_nvfp4_protocol_v2_executor_serves_protocol_v2_tcp() {
        let tempdir = tempfile::tempdir().unwrap();
        let catalog = tiny_expert_catalog(tempdir.path());
        let expected_executor =
            RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some("spark-0".to_owned()));
        let request = tiny_request().with_debug_checksum();
        let expected = execute_request(&expected_executor, &request).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()));
        let server = tokio::spawn(async move {
            let _ =
                serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor)).await;
        });

        let response = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default())
            .await
            .unwrap();

        assert_tcp_response_matches_executor_response(&response, &expected);
        server.abort();
    }

    #[tokio::test]
    async fn real_checkpoint_nvfp4_protocol_v2_tcp_roundtrip_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let request = real_checkpoint_request(16).with_debug_checksum();
        let expected_executor =
            RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some("spark-0".to_owned()));
        let expected = execute_request(&expected_executor, &request).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()));
        let server = tokio::spawn(async move {
            let _ =
                serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor)).await;
        });

        let response = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default())
            .await
            .unwrap();

        assert_tcp_response_matches_executor_response(&response, &expected);
        let output = bf16_values(&response.partial_output_payload);
        assert_eq!(output.len(), 16);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
        let output_checksum = output.iter().map(|value| *value as f64).sum::<f64>();
        eprintln!(
            "real_checkpoint_nvfp4_protocol_v2_tcp_roundtrip executor={} layer={} expert=0 hidden_dim={} output_values={} output_checksum={output_checksum}",
            expected_executor.name(),
            request.header.layer_id,
            request.header.hidden_dim,
            output.len()
        );
        server.abort();
    }

    #[tokio::test]
    async fn real_checkpoint_nvfp4_protocol_v2_tcp_roundtrip_uses_loadplan_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(owner_lookup) = load_real_owner_lookup_or_skip() else {
            return;
        };
        let request = real_checkpoint_request(16).with_debug_checksum();
        let expected_executor =
            RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup.clone());
        let expected = execute_request(&expected_executor, &request).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup);
        let server = tokio::spawn(async move {
            let _ =
                serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor)).await;
        });

        let response = tcp_protocol_v2_roundtrip(addr, &request, TcpTransportConfig::default())
            .await
            .unwrap();

        assert_tcp_response_matches_executor_response(&response, &expected);
        let output = bf16_values(&response.partial_output_payload);
        let output_checksum = output.iter().map(|value| *value as f64).sum::<f64>();
        assert_eq!(output.len(), 16);
        assert!(output.iter().all(|value| value.is_finite()));
        assert!(output.iter().any(|value| *value != 0.0));
        eprintln!(
            "real_checkpoint_nvfp4_protocol_v2_loadplan_roundtrip executor={} loadplan=loadplan.spark-0.json layer={} expert=0 hidden_dim={} output_values={} output_checksum={output_checksum}",
            expected_executor.name(),
            request.header.layer_id,
            request.header.hidden_dim,
            output.len()
        );
        server.abort();
    }

    #[tokio::test]
    async fn real_checkpoint_host_batch_tcp_roundtrip_scatters_global_rows_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(owner_lookup) = load_real_owner_lookup_or_skip() else {
            return;
        };
        let fixture = real_checkpoint_host_batch_request(16, &owner_lookup).with_debug_checksum();
        let expected_executor =
            RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup.clone());
        let expected = execute_request(&expected_executor, &fixture.request).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup);
        let server = tokio::spawn(async move {
            let _ =
                serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor)).await;
        });

        let response =
            tcp_protocol_v2_roundtrip(addr, &fixture.request, TcpTransportConfig::default())
                .await
                .unwrap();

        assert_tcp_response_matches_executor_response(&response, &expected);
        assert_eq!(fixture.batch.num_rows(), 3);
        assert_eq!(fixture.host_batch.num_rows(), 2);
        assert_eq!(fixture.request.hidden_payload, fixture.compact_hidden);
        assert_ne!(fixture.request.hidden_payload, fixture.global_hidden);
        let encoded_response = response.encode().unwrap();
        let response_view = ExpertProtocolV2ResponseView::parse(&encoded_response).unwrap();
        let partials = (0..fixture.host_batch.num_rows())
            .map(|host_row_index| {
                response_view
                    .partial_output_row_payload(host_row_index)
                    .map(|row| row.to_vec())
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .unwrap();
        let scattered = fixture
            .host_batch
            .scatter_partial_outputs(&partials, fixture.batch.num_rows())
            .unwrap();
        assert!(scattered[0].is_some());
        assert!(scattered[1].is_none());
        assert!(scattered[2].is_some());
        let outputs = scattered
            .iter()
            .filter_map(|row| row.as_deref())
            .flat_map(bf16_values)
            .collect::<Vec<_>>();
        assert_eq!(
            outputs.len(),
            fixture.host_batch.num_rows() * fixture.batch.hidden_dim
        );
        assert!(outputs.iter().all(|value| value.is_finite()));
        assert!(outputs.iter().any(|value| *value != 0.0));
        let output_checksum = outputs.iter().map(|value| *value as f64).sum::<f64>();
        eprintln!(
            "real_checkpoint_host_batch_tcp_roundtrip executor={} host={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} output_checksum={output_checksum}",
            expected_executor.name(),
            fixture.host_batch.host,
            fixture.batch.num_rows(),
            fixture.host_batch.num_rows(),
            fixture.host_batch.route_count(),
            fixture.batch.hidden_dim,
            outputs.len()
        );
        server.abort();
    }

    #[tokio::test]
    async fn real_checkpoint_host_batch_set_tcp_roundtrips_and_accumulates_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(owner_lookup) = load_full_owner_lookup_or_skip() else {
            return;
        };
        let fixture = real_checkpoint_host_batch_set_fixture(16, &owner_lookup);
        let touched_hosts = fixture
            .set
            .touched_hosts()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(touched_hosts, vec!["spark-0", "spark-1"]);
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in &touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let executor =
                RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some(host.clone()))
                    .with_owner_lookup(owner_lookup.clone());
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor))
                    .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &fixture.set,
            &fixture.global_hidden,
            &targets,
            484,
            TcpTransportConfig::default(),
        )
        .await
        .unwrap();
        let accumulation = &dispatch.accumulation;
        let stats = &dispatch.stats;
        assert_eq!(stats.hosts, 2);
        assert_eq!(stats.global_rows, 2);
        assert_eq!(stats.host_rows, 2);
        assert_eq!(stats.routes, 2);
        assert_eq!(stats.output_dim, fixture.batch.hidden_dim);
        assert_eq!(stats.output_values, 32);
        assert!(stats.request_wire_bytes > 0);
        assert!(stats.response_wire_bytes > 0);
        assert_eq!(stats.contribution_counts, vec![1, 1]);
        assert_eq!(
            stats.output_checksum,
            accumulation
                .values
                .iter()
                .map(|value| *value as f64)
                .sum::<f64>()
        );
        fixture
            .set
            .reconstruction_plan
            .validate_for_batches(&fixture.set.batches, fixture.set.global_row_count)
            .unwrap();
        assert_eq!(accumulation.contribution_counts, vec![1, 1]);
        assert_eq!(
            accumulation.values.len(),
            fixture.batch.num_rows() * fixture.batch.hidden_dim
        );
        assert!(accumulation.values.iter().all(|value| value.is_finite()));
        assert!(accumulation.values.iter().any(|value| *value != 0.0));
        let output_checksum = accumulation
            .values
            .iter()
            .map(|value| *value as f64)
            .sum::<f64>();
        eprintln!(
            "real_checkpoint_host_batch_set_tcp_roundtrip executor={} helper=tcp_protocol_v2_host_batch_set_bf16_dispatch hosts={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={output_checksum}",
            REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
            stats.hosts,
            stats.global_rows,
            stats.host_rows,
            stats.routes,
            stats.output_dim,
            stats.output_values,
            stats.contribution_counts,
            stats.request_wire_bytes,
            stats.response_wire_bytes
        );
        for server in servers {
            server.abort();
        }
    }

    #[tokio::test]
    async fn real_checkpoint_host_batch_set_matches_local_route_reduction_when_available() {
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(owner_lookup) = load_full_owner_lookup_or_skip() else {
            return;
        };
        let fixture = real_checkpoint_topk_host_batch_set_fixture(16, &owner_lookup);
        let touched_hosts = fixture
            .set
            .touched_hosts()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(
            touched_hosts,
            vec!["spark-0", "spark-1", "spark-2", "spark-3"]
        );
        let expected = local_host_partial_bf16_accumulation(
            &catalog,
            &fixture.set,
            &fixture.global_hidden,
            &owner_lookup,
        )
        .unwrap();
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in &touched_hosts {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let executor =
                RealNvfp4ProtocolV2Executor::new(catalog.clone(), Some(3), Some(host.clone()))
                    .with_owner_lookup(owner_lookup.clone());
            servers.push(tokio::spawn(async move {
                let _ = serve_protocol_v2_tcp_listener_with_executor(listener, Arc::new(executor))
                    .await;
            }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &fixture.set,
            &fixture.global_hidden,
            &targets,
            489,
            TcpTransportConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(dispatch.stats.hosts, 4);
        assert_eq!(dispatch.stats.global_rows, 1);
        assert_eq!(dispatch.stats.host_rows, 4);
        assert_eq!(dispatch.stats.routes, 8);
        assert_eq!(dispatch.stats.output_dim, fixture.batch.hidden_dim);
        assert_eq!(dispatch.stats.output_values, 16);
        assert_eq!(dispatch.stats.contribution_counts, vec![4]);
        assert_eq!(
            dispatch.accumulation.contribution_counts,
            expected.contribution_counts
        );
        assert_eq!(dispatch.accumulation.values, expected.values);
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        eprintln!(
            "real_checkpoint_host_batch_set_local_route_reduction_match executor={} hosts={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
            dispatch.stats.hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum
        );
        for server in servers {
            server.abort();
        }
    }

    #[tokio::test]
    async fn real_checkpoint_host_batch_dispatches_through_expertd_entrypoint_when_available() {
        if !real_nvfp4_cuda_reference_kernels_enabled() {
            eprintln!(
                "skipping real-checkpoint expertd entrypoint dispatch because CUDA reference kernels are disabled"
            );
            return;
        }
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return;
        };
        let owner_lookup =
            read_expert_owner_lookup(&loadplan_path).expect("parsing full GLM loadplan fixture");
        let catalog_path = real_catalog_path();
        let fixture = real_checkpoint_host_batch_request(16, &owner_lookup);
        assert_eq!(fixture.host_batch.host, "spark-0");
        let expected_executor =
            RealNvfp4ProtocolV2Executor::new(catalog, Some(3), Some("spark-0".to_owned()))
                .with_owner_lookup(owner_lookup);
        let expected = execute_request(&expected_executor, &fixture.request).unwrap();
        let addr = unused_loopback_addr();
        let args = ExpertDaemonArgs {
            synthetic_weights: false,
            preflight_only: false,
            transport: "tcp".to_owned(),
            listen: addr.to_string(),
            loadplan: Some(loadplan_path),
            catalog: Some(catalog_path),
            model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
            real_layer: Some(3),
            role_hostname: Some("spark-0".to_owned()),
        };
        let server = tokio::spawn(async move { run_expertd(args).await });
        wait_for_expertd_tcp_listener(addr).await;

        let response =
            tcp_protocol_v2_roundtrip(addr, &fixture.request, TcpTransportConfig::default())
                .await
                .unwrap();

        assert_tcp_response_matches_executor_response(&response, &expected);
        assert_eq!(fixture.batch.num_rows(), 3);
        assert_eq!(fixture.host_batch.num_rows(), 2);
        assert_eq!(fixture.request.hidden_payload, fixture.compact_hidden);
        assert_ne!(fixture.request.hidden_payload, fixture.global_hidden);
        let encoded_response = response.encode().unwrap();
        let response_view = ExpertProtocolV2ResponseView::parse(&encoded_response).unwrap();
        let partials = (0..fixture.host_batch.num_rows())
            .map(|host_row_index| {
                response_view
                    .partial_output_row_payload(host_row_index)
                    .map(|row| row.to_vec())
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .unwrap();
        let scattered = fixture
            .host_batch
            .scatter_partial_outputs(&partials, fixture.batch.num_rows())
            .unwrap();
        assert!(scattered[0].is_some());
        assert!(scattered[1].is_none());
        assert!(scattered[2].is_some());
        let outputs = scattered
            .iter()
            .filter_map(|row| row.as_deref())
            .flat_map(bf16_values)
            .collect::<Vec<_>>();
        assert_eq!(
            outputs.len(),
            fixture.host_batch.num_rows() * fixture.batch.hidden_dim
        );
        assert!(outputs.iter().all(|value| value.is_finite()));
        assert!(outputs.iter().any(|value| *value != 0.0));
        let output_checksum = outputs.iter().map(|value| *value as f64).sum::<f64>();
        eprintln!(
            "real_checkpoint_host_batch_expertd_entrypoint_dispatch daemon=run_expertd executor={} host={} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} output_checksum={output_checksum}",
            REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
            fixture.host_batch.host,
            fixture.batch.num_rows(),
            fixture.host_batch.num_rows(),
            fixture.host_batch.route_count(),
            fixture.batch.hidden_dim,
            outputs.len()
        );
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn real_checkpoint_host_batch_set_dispatches_through_expertd_entrypoints_when_available()
    {
        if !real_nvfp4_cuda_reference_kernels_enabled() {
            eprintln!(
                "skipping real-checkpoint expertd entrypoint host-batch-set dispatch because CUDA reference kernels are disabled"
            );
            return;
        }
        let Some(catalog) = load_real_catalog_or_skip() else {
            return;
        };
        let Some(loadplan_path) = load_full_loadplan_path_or_skip() else {
            return;
        };
        let owner_lookup =
            read_expert_owner_lookup(&loadplan_path).expect("parsing full GLM loadplan fixture");
        let catalog_path = real_catalog_path();
        let fixture = real_checkpoint_topk_host_batch_set_fixture(16, &owner_lookup);
        let touched_hosts = fixture
            .set
            .touched_hosts()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        assert_eq!(
            touched_hosts,
            vec!["spark-0", "spark-1", "spark-2", "spark-3"]
        );
        let expected = local_host_partial_bf16_accumulation(
            &catalog,
            &fixture.set,
            &fixture.global_hidden,
            &owner_lookup,
        )
        .unwrap();
        let mut targets = Vec::new();
        let mut servers = Vec::new();
        for host in &touched_hosts {
            let addr = unused_loopback_addr();
            let args = ExpertDaemonArgs {
                synthetic_weights: false,
                preflight_only: false,
                transport: "tcp".to_owned(),
                listen: addr.to_string(),
                loadplan: Some(loadplan_path.clone()),
                catalog: Some(catalog_path.clone()),
                model_id: glmrt_core::DEFAULT_MODEL_ID.to_owned(),
                real_layer: Some(3),
                role_hostname: Some(host.clone()),
            };
            servers.push(tokio::spawn(async move { run_expertd(args).await }));
            targets.push(TcpProtocolV2HostBatchTarget {
                host: host.clone(),
                addr,
            });
        }
        for target in &targets {
            wait_for_expertd_tcp_listener(target.addr).await;
        }

        let dispatch = tcp_protocol_v2_host_batch_set_bf16_dispatch(
            &fixture.set,
            &fixture.global_hidden,
            &targets,
            505,
            TcpTransportConfig::default(),
        )
        .await
        .unwrap();

        assert_eq!(dispatch.stats.hosts, 4);
        assert_eq!(dispatch.stats.global_rows, 1);
        assert_eq!(dispatch.stats.host_rows, 4);
        assert_eq!(dispatch.stats.routes, 8);
        assert_eq!(dispatch.stats.output_dim, fixture.batch.hidden_dim);
        assert_eq!(dispatch.stats.output_values, 16);
        assert_eq!(dispatch.stats.contribution_counts, vec![4]);
        assert_eq!(
            dispatch.accumulation.contribution_counts,
            expected.contribution_counts
        );
        assert_eq!(dispatch.accumulation.values, expected.values);
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .all(|value| value.is_finite()));
        assert!(dispatch
            .accumulation
            .values
            .iter()
            .any(|value| *value != 0.0));
        eprintln!(
            "real_checkpoint_host_batch_set_expertd_entrypoint_dispatch daemon=run_expertd executor={} hosts={} host_names={:?} global_rows={} host_rows={} routes={} hidden_dim={} output_values={} contribution_counts={:?} request_wire_bytes={} response_wire_bytes={} output_checksum={}",
            REAL_NVFP4_PROTOCOL_V2_EXECUTOR,
            dispatch.stats.hosts,
            touched_hosts,
            dispatch.stats.global_rows,
            dispatch.stats.host_rows,
            dispatch.stats.routes,
            dispatch.stats.output_dim,
            dispatch.stats.output_values,
            dispatch.stats.contribution_counts,
            dispatch.stats.request_wire_bytes,
            dispatch.stats.response_wire_bytes,
            dispatch.stats.output_checksum
        );
        for server in servers {
            server.abort();
        }
    }

    fn tiny_expert_catalog(root: &std::path::Path) -> TensorCatalog {
        tiny_expert_catalog_for_layers(root, &[3])
    }

    fn tiny_expert_catalog_for_layers(root: &std::path::Path, layers: &[usize]) -> TensorCatalog {
        expert_catalog_for_layers_with_geometry(root, layers, 2, 1)
    }

    fn preload_expert_catalog_for_layers(
        root: &std::path::Path,
        layers: &[usize],
    ) -> TensorCatalog {
        // Startup preloading exercises packed kernels whose input widths must be
        // aligned to the model's 16-value quantization groups.
        const MINIMUM_PACKED_DIM: usize = 16;
        expert_catalog_for_layers_with_geometry(
            root,
            layers,
            MINIMUM_PACKED_DIM,
            MINIMUM_PACKED_DIM,
        )
    }

    fn preload_exl3_catalog(root: &std::path::Path) -> TensorCatalog {
        let mut tensors = Vec::new();
        let mut byte_offset = 0_u64;
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            for (suffix, dtype, shape, byte_length, is_quantization_metadata) in [
                ("trellis", DType::I16, vec![32], 64_u64, false),
                ("suh", DType::F16, vec![16], 32_u64, true),
                ("svh", DType::F16, vec![8], 16_u64, true),
                ("mcg", DType::I32, vec![], 4_u64, true),
            ] {
                tensors.push(TensorInfo {
                    name: format!("model.layers.3.mlp.experts.0.{projection}.{suffix}"),
                    file: "expert.safetensors".to_owned(),
                    dtype,
                    shape,
                    byte_offset,
                    byte_length,
                    role: TensorRole::RoutedExpert,
                    layer_id: Some(3),
                    expert_id: Some(0),
                    is_quantization_metadata,
                });
                byte_offset += byte_length;
            }
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        TensorCatalog {
            model_id: "test/glm52-exl3".to_owned(),
            snapshot_path: root.display().to_string(),
            facts: ModelFacts {
                quantization_recipe: GLM52_EXL3_RECIPE_K3_V1.to_owned(),
                ..ModelFacts::default()
            },
            tensors,
        }
    }

    fn expert_catalog_for_layers_with_geometry(
        root: &std::path::Path,
        layers: &[usize],
        hidden_size: usize,
        intermediate_size: usize,
    ) -> TensorCatalog {
        let shard_path = root.join("expert.bin");
        let mut shard_bytes = Vec::new();
        let mut tensors = Vec::new();
        let packed_hidden_width = hidden_size.div_ceil(2);
        let packed_intermediate_width = intermediate_size.div_ceil(2);
        let hidden_scale_width = hidden_size.div_ceil(16);
        let intermediate_scale_width = intermediate_size.div_ceil(16);
        for layer_id in layers {
            for projection in ["gate_proj", "up_proj"] {
                push_tensor(
                    &mut shard_bytes,
                    &mut tensors,
                    *layer_id,
                    projection,
                    "weight",
                    DType::U8,
                    vec![intermediate_size, packed_hidden_width],
                    &vec![0xaa; intermediate_size * packed_hidden_width],
                );
                push_tensor(
                    &mut shard_bytes,
                    &mut tensors,
                    *layer_id,
                    projection,
                    "weight_scale",
                    DType::F8E4M3,
                    vec![intermediate_size, hidden_scale_width],
                    &vec![0x38; intermediate_size * hidden_scale_width],
                );
                push_tensor(
                    &mut shard_bytes,
                    &mut tensors,
                    *layer_id,
                    projection,
                    "input_scale",
                    DType::F32,
                    Vec::new(),
                    &1.0_f32.to_le_bytes(),
                );
                push_tensor(
                    &mut shard_bytes,
                    &mut tensors,
                    *layer_id,
                    projection,
                    "weight_scale_2",
                    DType::F32,
                    Vec::new(),
                    &1.0_f32.to_le_bytes(),
                );
            }
            push_tensor(
                &mut shard_bytes,
                &mut tensors,
                *layer_id,
                "down_proj",
                "weight",
                DType::U8,
                vec![hidden_size, packed_intermediate_width],
                &vec![0x0a; hidden_size * packed_intermediate_width],
            );
            push_tensor(
                &mut shard_bytes,
                &mut tensors,
                *layer_id,
                "down_proj",
                "weight_scale",
                DType::F8E4M3,
                vec![hidden_size, intermediate_scale_width],
                &vec![0x38; hidden_size * intermediate_scale_width],
            );
            push_tensor(
                &mut shard_bytes,
                &mut tensors,
                *layer_id,
                "down_proj",
                "input_scale",
                DType::F32,
                Vec::new(),
                &1.0_f32.to_le_bytes(),
            );
            push_tensor(
                &mut shard_bytes,
                &mut tensors,
                *layer_id,
                "down_proj",
                "weight_scale_2",
                DType::F32,
                Vec::new(),
                &1.0_f32.to_le_bytes(),
            );
        }
        File::create(&shard_path)
            .unwrap()
            .write_all(&shard_bytes)
            .unwrap();
        TensorCatalog {
            model_id: "test/model".to_owned(),
            snapshot_path: root.display().to_string(),
            facts: ModelFacts {
                hidden_size,
                ..ModelFacts::default()
            },
            tensors,
        }
    }

    fn tiny_request() -> ExpertProtocolV2Request {
        tiny_request_for_layer(3)
    }

    fn tiny_request_for_layer(layer_id: u32) -> ExpertProtocolV2Request {
        request_for_layer(layer_id, &[1.0, 2.0])
    }

    fn preload_request_for_layer(layer_id: u32) -> ExpertProtocolV2Request {
        request_for_layer(layer_id, &[1.0; 16])
    }

    fn request_for_layer(layer_id: u32, hidden_values: &[f32]) -> ExpertProtocolV2Request {
        let rows = vec![ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 1,
            token_position: 0,
            route_offset: 0,
            route_count: 1,
        }];
        let routes = vec![ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id: 0,
            gate_weight: 1.0,
        }];
        let mut hidden_payload = Vec::new();
        for value in hidden_values {
            hidden_payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        ExpertProtocolV2Request::new(
            7,
            0x51CE,
            layer_id,
            hidden_values.len() as u32,
            ExpertV2Dtype::Bf16,
            rows,
            routes,
            hidden_payload,
        )
        .unwrap()
    }

    fn real_checkpoint_request(hidden_dim: usize) -> ExpertProtocolV2Request {
        let rows = vec![ExpertProtocolV2RowDescriptor {
            row_id: 0,
            source_kind: ExpertV2SourceKind::Decode,
            source_request_id: 480,
            token_position: 0,
            route_offset: 0,
            route_count: 1,
        }];
        let routes = vec![ExpertProtocolV2RouteEntry {
            row_index: 0,
            expert_id: 0,
            gate_weight: 1.0,
        }];
        let mut hidden_payload = Vec::with_capacity(hidden_dim * 2);
        for idx in 0..hidden_dim {
            let value = ((idx % 17) as f32 - 8.0) / 16.0;
            hidden_payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
        }
        ExpertProtocolV2Request::new(
            480,
            0x51CE,
            3,
            hidden_dim as u32,
            ExpertV2Dtype::Bf16,
            rows,
            routes,
            hidden_payload,
        )
        .unwrap()
    }

    struct HostBatchRequestFixture {
        batch: ExpertBatch,
        host_batch: ExpertHostBatch,
        request: ExpertProtocolV2Request,
        global_hidden: Vec<u8>,
        compact_hidden: Vec<u8>,
    }

    struct HostBatchSetFixture {
        batch: ExpertBatch,
        set: ExpertHostBatchSet,
        global_hidden: Vec<u8>,
    }

    impl HostBatchRequestFixture {
        fn with_debug_checksum(mut self) -> Self {
            self.request = self.request.with_debug_checksum();
            self
        }
    }

    fn real_checkpoint_host_batch_request(
        hidden_dim: usize,
        owner_lookup: &ExpertOwnerLookup,
    ) -> HostBatchRequestFixture {
        let batch = ExpertBatch {
            layer_id: LayerId(3),
            placement_version: PlacementVersion("phase0-real-host-batch".to_owned()),
            hidden_dim,
            hidden_bytes_per_row: hidden_dim * 2,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(2),
            quantization_recipe: ModelFacts::default().quantization_recipe,
            rows: vec![
                glmrt_core::ExpertBatchRow {
                    row_id: 0,
                    source_kind: RowSourceKind::DecodeStep,
                    request_id: RequestId("phase0-host-batch-row-0".to_owned()),
                    sequence_id: "seq-a".to_owned(),
                    token_position: PositionId(0),
                    route_offset: 0,
                    route_count: 1,
                },
                glmrt_core::ExpertBatchRow {
                    row_id: 1,
                    source_kind: RowSourceKind::PrefillChunk,
                    request_id: RequestId("phase0-host-batch-row-1".to_owned()),
                    sequence_id: "seq-b".to_owned(),
                    token_position: PositionId(1),
                    route_offset: 1,
                    route_count: 1,
                },
                glmrt_core::ExpertBatchRow {
                    row_id: 2,
                    source_kind: RowSourceKind::MtpVerifyBlock,
                    request_id: RequestId("phase0-host-batch-row-2".to_owned()),
                    sequence_id: "seq-c".to_owned(),
                    token_position: PositionId(2),
                    route_offset: 2,
                    route_count: 1,
                },
            ],
        };
        let routes = vec![
            ExpertBatchRoute {
                row_index: 0,
                expert_id: 0,
                gate_weight: 0.75,
            },
            ExpertBatchRoute {
                row_index: 1,
                expert_id: 1,
                gate_weight: 1.0,
            },
            ExpertBatchRoute {
                row_index: 2,
                expert_id: 0,
                gate_weight: 0.25,
            },
        ];
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let host_batch = ExpertHostBatch::from_expert_batch_with_owner_lookup(
            &batch,
            "spark-0",
            &routes,
            &hosts,
            owner_lookup,
        )
        .unwrap();
        assert_eq!(host_batch.num_rows(), 2);
        assert_eq!(
            host_batch.global_row_indices().collect::<Vec<_>>(),
            vec![0, 2]
        );
        let global_hidden = real_checkpoint_host_batch_hidden(batch.num_rows(), hidden_dim);
        let compact_hidden = host_batch
            .compact_hidden_payload(&global_hidden, batch.num_rows())
            .unwrap();
        let request = ExpertProtocolV2Request::from_expert_host_batch(
            483,
            &host_batch,
            compact_hidden.clone(),
        )
        .unwrap();
        HostBatchRequestFixture {
            batch,
            host_batch,
            request,
            global_hidden,
            compact_hidden,
        }
    }

    fn real_checkpoint_host_batch_set_fixture(
        hidden_dim: usize,
        owner_lookup: &ExpertOwnerLookup,
    ) -> HostBatchSetFixture {
        let batch = ExpertBatch {
            layer_id: LayerId(3),
            placement_version: PlacementVersion("phase0-real-host-batch-set".to_owned()),
            hidden_dim,
            hidden_bytes_per_row: hidden_dim * 2,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(2),
            quantization_recipe: ModelFacts::default().quantization_recipe,
            rows: vec![
                glmrt_core::ExpertBatchRow {
                    row_id: 0,
                    source_kind: RowSourceKind::DecodeStep,
                    request_id: RequestId("phase0-host-batch-set-row-0".to_owned()),
                    sequence_id: "seq-set-a".to_owned(),
                    token_position: PositionId(0),
                    route_offset: 0,
                    route_count: 1,
                },
                glmrt_core::ExpertBatchRow {
                    row_id: 1,
                    source_kind: RowSourceKind::PrefillChunk,
                    request_id: RequestId("phase0-host-batch-set-row-1".to_owned()),
                    sequence_id: "seq-set-b".to_owned(),
                    token_position: PositionId(1),
                    route_offset: 1,
                    route_count: 1,
                },
            ],
        };
        let routes = vec![
            ExpertBatchRoute {
                row_index: 0,
                expert_id: 0,
                gate_weight: 1.0,
            },
            ExpertBatchRoute {
                row_index: 1,
                expert_id: 1,
                gate_weight: 1.0,
            },
        ];
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
            &batch,
            &routes,
            &hosts,
            owner_lookup,
        )
        .unwrap();
        assert_eq!(set.num_hosts(), 2);
        assert_eq!(set.route_count(), 2);
        let global_hidden = real_checkpoint_host_batch_hidden(batch.num_rows(), hidden_dim);
        HostBatchSetFixture {
            batch,
            set,
            global_hidden,
        }
    }

    fn real_checkpoint_topk_host_batch_set_fixture(
        hidden_dim: usize,
        owner_lookup: &ExpertOwnerLookup,
    ) -> HostBatchSetFixture {
        let batch = ExpertBatch {
            layer_id: LayerId(3),
            placement_version: PlacementVersion("phase0-real-topk-host-batch-set".to_owned()),
            hidden_dim,
            hidden_bytes_per_row: hidden_dim * 2,
            hidden_dtype: DType::Bf16,
            graph_bucket: GraphBucket::new(1),
            quantization_recipe: ModelFacts::default().quantization_recipe,
            rows: vec![glmrt_core::ExpertBatchRow {
                row_id: 0,
                source_kind: RowSourceKind::PrefillChunk,
                request_id: RequestId("phase0-host-batch-set-topk-row-0".to_owned()),
                sequence_id: "seq-set-topk".to_owned(),
                token_position: PositionId(0),
                route_offset: 0,
                route_count: 8,
            }],
        };
        let routes = (0..8)
            .map(|expert_id| ExpertBatchRoute {
                row_index: 0,
                expert_id,
                gate_weight: 1.0 / 8.0,
            })
            .collect::<Vec<_>>();
        let hosts = EXPERT_HOSTS
            .iter()
            .map(|host| (*host).to_owned())
            .collect::<Vec<_>>();
        let set = ExpertHostBatchSet::from_expert_batch_with_owner_lookup(
            &batch,
            &routes,
            &hosts,
            owner_lookup,
        )
        .unwrap();
        assert_eq!(set.num_hosts(), 4);
        assert_eq!(set.route_count(), 8);
        let global_hidden = real_checkpoint_host_batch_hidden(batch.num_rows(), hidden_dim);
        HostBatchSetFixture {
            batch,
            set,
            global_hidden,
        }
    }

    fn local_host_partial_bf16_accumulation(
        catalog: &TensorCatalog,
        set: &ExpertHostBatchSet,
        global_hidden: &[u8],
        owner_lookup: &ExpertOwnerLookup,
    ) -> anyhow::Result<glmrt_core::ExpertHostBatchSetAccumulation> {
        let output_dim = set.batches.first().unwrap().hidden_dim;
        let mut partials_by_host = Vec::with_capacity(set.batches.len());
        for host_batch in &set.batches {
            let mut host_partials = Vec::with_capacity(host_batch.num_rows());
            for host_row in &host_batch.rows {
                let start = host_row.global_row_index * host_batch.hidden_bytes_per_row;
                let hidden =
                    bf16_values(&global_hidden[start..start + host_batch.hidden_bytes_per_row]);
                let mut reduced = vec![0.0_f32; output_dim];
                for route in &host_batch.routes
                    [host_row.route_offset..host_row.route_offset + host_row.route_count]
                {
                    owner_lookup
                        .owner_for(host_batch.layer_id.0 as usize, route.expert_id)
                        .unwrap();
                    let scored = ScoredRoute {
                        expert_id: route.expert_id,
                        score: route.gate_weight,
                        corrected_score: route.gate_weight,
                        normalized_weight: route.gate_weight,
                    };
                    let intermediate_rows = super::projection_rows(
                        catalog,
                        host_batch.layer_id.0 as usize,
                        route.expert_id,
                        "gate_proj",
                    )?;
                    let execution =
                        crate::commands::real_full::sparse_mlp::route::execute_nvfp4_route(
                            catalog,
                            host_batch.layer_id.0 as usize,
                            &hidden,
                            &scored,
                            intermediate_rows,
                            output_dim,
                        )?;
                    for (dst, delta) in reduced.iter_mut().zip(execution.outputs.iter()) {
                        *dst += *delta;
                    }
                }
                host_partials.push(reduced.into_iter().map(bf16_truncate).collect::<Vec<_>>());
            }
            partials_by_host.push(host_partials);
        }
        Ok(set.accumulate_partial_outputs_f32(&partials_by_host, output_dim)?)
    }

    fn real_checkpoint_host_batch_hidden(rows: usize, hidden_dim: usize) -> Vec<u8> {
        let mut hidden_payload = Vec::with_capacity(rows * hidden_dim * 2);
        for row in 0..rows {
            for idx in 0..hidden_dim {
                let value = (((row * 11 + idx) % 23) as f32 - 11.0) / 32.0;
                hidden_payload.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
            }
        }
        hidden_payload
    }

    fn execute_request(
        executor: &RealNvfp4ProtocolV2Executor,
        request: &ExpertProtocolV2Request,
    ) -> anyhow::Result<glmrt_transport::ExpertProtocolV2Response> {
        let frame = request.encode().unwrap();
        let view = glmrt_transport::ExpertProtocolV2RequestView::parse(&frame).unwrap();
        executor.execute(&view)
    }

    fn assert_tcp_response_matches_executor_response(
        response: &ExpertProtocolV2Response,
        expected: &ExpertProtocolV2Response,
    ) {
        let mut expected_header = expected.header.clone();
        expected_header.executor_id =
            expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR);
        assert_eq!(response.header, expected_header);
        assert_eq!(
            response.partial_output_payload,
            expected.partial_output_payload
        );
    }

    fn bf16_values(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    }

    fn bf16_truncate(value: f32) -> f32 {
        f32::from_bits(value.to_bits() & 0xFFFF_0000)
    }

    fn push_tensor(
        shard_bytes: &mut Vec<u8>,
        tensors: &mut Vec<TensorInfo>,
        layer_id: usize,
        projection: &str,
        suffix: &str,
        dtype: DType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) {
        let byte_offset = shard_bytes.len() as u64;
        shard_bytes.extend_from_slice(bytes);
        tensors.push(TensorInfo {
            name: format!("model.layers.{layer_id}.mlp.experts.0.{projection}.{suffix}"),
            file: "expert.bin".to_owned(),
            dtype,
            shape,
            byte_offset,
            byte_length: bytes.len() as u64,
            role: TensorRole::RoutedExpert,
            layer_id: Some(layer_id as u32),
            expert_id: Some(0),
            is_quantization_metadata: suffix != "weight",
        });
    }

    fn load_real_catalog_or_skip() -> Option<TensorCatalog> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "ProtocolV2 real NVFP4 executor",
            );
            return None;
        }
        let catalog_path = real_catalog_path();
        let Ok(file) = File::open(&catalog_path) else {
            eprintln!("skipped: missing {}", catalog_path.display());
            return None;
        };
        let catalog: TensorCatalog =
            serde_json::from_reader(file).expect("parsing real GLM catalog fixture");
        if !Path::new(&catalog.snapshot_path).exists() {
            eprintln!("skipped: missing snapshot {}", catalog.snapshot_path);
            return None;
        }
        Some(catalog)
    }

    fn real_catalog_path() -> PathBuf {
        repo_root().join(".glmrt-cache/model-artifacts/diagnostic/model_catalog.json")
    }

    fn load_real_owner_lookup_or_skip() -> Option<ExpertOwnerLookup> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "ProtocolV2 real owner lookup",
            );
            return None;
        }
        let loadplan_path =
            repo_root().join(".glmrt-cache/model-artifacts/diagnostic/loadplan.spark-0.json");
        if !loadplan_path.exists() {
            eprintln!("skipped: missing {}", loadplan_path.display());
            return None;
        }
        Some(read_expert_owner_lookup(&loadplan_path).expect("parsing real GLM loadplan fixture"))
    }

    fn load_full_owner_lookup_or_skip() -> Option<ExpertOwnerLookup> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "ProtocolV2 full owner lookup",
            );
            return None;
        }
        let loadplan_path =
            repo_root().join(".glmrt-cache/model-artifacts/diagnostic/loadplan.json");
        if !loadplan_path.exists() {
            eprintln!("skipped: missing {}", loadplan_path.display());
            return None;
        }
        Some(read_expert_owner_lookup(&loadplan_path).expect("parsing full GLM loadplan fixture"))
    }

    fn load_full_loadplan_path_or_skip() -> Option<PathBuf> {
        if !crate::commands::real_full::tests::fixture::real_checkpoint_tests_enabled() {
            crate::commands::real_full::tests::fixture::real_checkpoint_tests_skip_message(
                "ProtocolV2 full loadplan",
            );
            return None;
        }
        let loadplan_path =
            repo_root().join(".glmrt-cache/model-artifacts/diagnostic/loadplan.json");
        if !loadplan_path.exists() {
            eprintln!("skipped: missing {}", loadplan_path.display());
            return None;
        }
        Some(loadplan_path)
    }

    fn unused_loopback_addr() -> SocketAddr {
        let listener = StdTcpListener::bind("127.0.0.1:0").unwrap();
        listener.local_addr().unwrap()
    }

    async fn wait_for_expertd_tcp_listener(addr: SocketAddr) {
        for _ in 0..24_000 {
            if TcpStream::connect(addr).await.is_ok() {
                return;
            }
            sleep(Duration::from_millis(10)).await;
        }
        panic!("timed out waiting for expertd TCP listener at {addr}");
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
    }
}
