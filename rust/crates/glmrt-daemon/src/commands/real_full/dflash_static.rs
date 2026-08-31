#![allow(dead_code)]

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtDeviceBuffer, NativeLibrary};
use serde::Serialize;

use super::coordinator_kernels::{cuda_native_library, device_buffer_byte_view, DeviceBf16Output};
use super::dflash::{
    Dflash2ResidentWeights, GLM53_DFLASH2_BLOCK_SIZE, GLM53_DFLASH2_DRAFT_LAYERS,
    GLM53_DFLASH2_HEAD_DIM, GLM53_DFLASH2_KV_HEADS, GLM53_DFLASH2_MASK_TOKEN_ID,
    GLM53_DFLASH2_MAX_DRAFTS,
};
use super::dflash_body::{
    dflash2_body_buffer_plan, launch_python_dflash2_body, Dflash2BodyBuffers, Dflash2BodyConfig,
};
use super::dflash_head::{
    dflash2_head_buffer_plan, launch_python_dflash2_head, Dflash2HeadBuffers, Dflash2HeadConfig,
    Dflash2HeadResidentWeights,
};
use super::dflash_update::{
    dflash2_update_buffer_plan, dflash2_update_resident_weights, launch_python_dflash2_update,
    Dflash2UpdateBuffers, Dflash2UpdateConfig, Dflash2UpdateResidentWeights,
};
use super::dspark_attention::{
    timing_summary, DsparkCudaEvent, DsparkCudaGraph, DsparkCudaStream, DsparkDeviceBuffer,
    DsparkPagedAttentionTiming,
};
use super::dspark_kv::{DsparkKvStorage, DsparkPagedKvMetadata, DsparkPagedKvMetadataBuffers};
use super::dspark_query::launch_embedding;
use super::dspark_update::bf16_difference;

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2StaticBenchConfig {
    pub(super) active_requests: usize,
    pub(super) accepted_rows_per_request: usize,
    pub(super) proposal_tokens_per_request: usize,
    pub(super) context_tokens: usize,
    pub(super) kv_capacity_tokens: usize,
    pub(super) allocate_full_kv_capacity: bool,
    pub(super) capture_page_buckets: bool,
    pub(super) page_size: usize,
    pub(super) kv_storage: DsparkKvStorage,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) repeats: usize,
    pub(super) seed: i64,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Dflash2StaticGraphReport {
    backend: &'static str,
    kv_storage: DsparkKvStorage,
    active_requests: usize,
    accepted_rows_per_request: usize,
    update_rows: usize,
    query_rows_per_request: usize,
    proposal_tokens_per_request: usize,
    context_tokens: usize,
    body_kv_tokens: usize,
    page_size: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    shared_kv_bytes: u64,
    rust_owned_mutable_bytes: u64,
    resident_weight_bytes: u64,
    update_graph_nodes: usize,
    update_graph_kernel_nodes: usize,
    base_update_graph_validation: Dflash2UpdateGraphValidation,
    packed_update_graph_rows: Vec<usize>,
    packed_update_graph_validation: Vec<Dflash2UpdateGraphValidation>,
    suffix_graph_nodes: usize,
    suffix_graph_kernel_nodes: usize,
    eager_replay_exact: bool,
    dynamic_anchor_changes_output: bool,
    dynamic_token_changed_bytes: usize,
    restored_replay_exact: bool,
    gpu_ms_per_update_replay: DsparkPagedAttentionTiming,
    gpu_ms_per_suffix_replay: DsparkPagedAttentionTiming,
    gpu_ms_per_full_cycle: DsparkPagedAttentionTiming,
    host_ms_per_full_cycle: DsparkPagedAttentionTiming,
    target_embedding_alias: bool,
    target_lm_head_alias: bool,
    body_output_is_head_source: bool,
    shared_update_body_kv: bool,
    candidate_topk_sorted: bool,
    candidate_score_accumulation: &'static str,
    sliding_window_tokens: usize,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
}

#[derive(Clone, Debug, Serialize)]
struct Dflash2UpdateGraphValidation {
    rows: usize,
    reference_fused_hidden_max_abs: f64,
    reference_fused_hidden_relative_l2: f64,
    reference_key_max_abs: f64,
    reference_key_bf16_steps_at_max_abs: u32,
    reference_key_relative_l2: f64,
    reference_value_bf16_steps_at_max_abs: u32,
    reference_value_relative_l2: f64,
    eager_replay_exact: bool,
    dynamic_positions_change_keys: bool,
    dynamic_key_changed_bytes: usize,
    restored_replay_exact: bool,
}

pub(super) struct Dflash2DraftStep {
    pub(super) context_tokens: usize,
    pub(super) committed_rows: usize,
    pub(super) anchor_token: usize,
    pub(super) proposal_token_ids: Vec<usize>,
    pub(super) update_ms: f64,
    pub(super) suffix_ms: f64,
    pub(super) readback_ms: f64,
    pub(super) total_ms: f64,
    pub(super) packed_update_rows: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2SharedKvPool {
    k_cache: GlmrtDeviceBuffer,
    v_cache: GlmrtDeviceBuffer,
    total_physical_pages: usize,
}

pub(super) struct Dflash2BatchedSuffixRequest<'a> {
    pub(super) page_table: &'a [i32],
    pub(super) absolute_context_after_update: usize,
    pub(super) cache_context_after_update: usize,
    pub(super) anchor_token: usize,
}

pub(super) struct Dflash2BatchedUpdateRequest<'a> {
    pub(super) target_hidden_taps: [&'a DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS],
    pub(super) target_row_start: usize,
    pub(super) committed_rows: usize,
    pub(super) absolute_context_start: usize,
    pub(super) cache_context_start: usize,
    pub(super) page_table: &'a [i32],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2BatchedUpdateStep {
    pub(super) absolute_context_after_update: usize,
    pub(super) cache_context_after_update: usize,
    pub(super) update_ms: f64,
}

fn dflash2_packed_update_rows(
    active_requests: usize,
    actual_rows: usize,
    maximum_rows: usize,
) -> Option<usize> {
    if actual_rows == 0 {
        return None;
    }
    if active_requests == 1 {
        return (actual_rows <= maximum_rows).then_some(actual_rows);
    }
    if !matches!(active_requests, 2 | 4) {
        return None;
    }
    actual_rows
        .max(active_requests)
        .checked_next_power_of_two()
        .filter(|rows| *rows <= maximum_rows)
}

pub(super) fn dflash2_update_graph_buckets(active_requests: usize) -> Result<&'static [usize]> {
    match active_requests {
        1 => Ok(&[2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1_024]),
        2 => Ok(&[2, 4, 8, 16]),
        4 => Ok(&[4, 8, 16, 32]),
        _ => anyhow::bail!(
            "DFlash2 update graph buckets require C1, C2, or C4; got C{active_requests}"
        ),
    }
}

pub(super) struct Dflash2BatchedSuffixStep {
    pub(super) proposal_token_ids: Vec<Vec<usize>>,
    pub(super) suffix_ms: f64,
    pub(super) readback_ms: f64,
    pub(super) total_ms: f64,
}

pub(super) struct Dflash2RequestUpdate {
    pub(super) context_tokens: usize,
    pub(super) committed_rows: usize,
    pub(super) context_after_update: usize,
    pub(super) cache_context_after_update: usize,
    pub(super) update_ms: f64,
}

pub(super) fn benchmark_dflash2_static_graph(
    weights: Dflash2ResidentWeights,
    config: Dflash2StaticBenchConfig,
    physical_kv_pages: Option<usize>,
) -> Result<Dflash2StaticGraphReport> {
    let update_weights = dflash2_update_resident_weights(weights)?;
    let mut executor =
        Dflash2StaticExecutor::capture_with_physical_pages(weights, config, physical_kv_pages)?;
    let base_positions = (0..config.active_requests)
        .flat_map(|_| {
            (0..config.accepted_rows_per_request).map(|row| {
                i32::try_from(config.context_tokens + row)
                    .context("DFlash2 base validation position does not fit i32")
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let base_update_graph_validation = executor.validate_update_graph(
        executor.update_buffers,
        &executor.update_graph,
        executor.update_rows,
        &base_positions,
    )?;

    // Validate the suffix while its KV history still matches the eager path
    // prepared during executor capture. Packed update capture and validation
    // intentionally write their test rows into this executor's shared KV pool.
    executor.replay_suffix()?;
    executor.stream.synchronize()?;
    let eager = executor.read_tokens(executor.head_buffers.eager_tokens)?;
    let replay = executor.read_tokens(executor.head_buffers.output_tokens)?;
    anyhow::ensure!(
        eager == replay,
        "DFlash2 static suffix replay differs from its eager candidate path"
    );

    executor.upload_anchors(&executor.dynamic_anchors.clone())?;
    executor.replay_suffix()?;
    executor.stream.synchronize()?;
    let dynamic = executor.read_tokens(executor.head_buffers.output_tokens)?;
    let dynamic_token_changed_bytes = byte_mismatch_count(&eager, &dynamic);
    anyhow::ensure!(
        dynamic_token_changed_bytes > 0,
        "DFlash2 static suffix ignored changed device-side anchor tokens"
    );
    executor.upload_anchors(&executor.initial_anchors.clone())?;
    executor.replay_suffix()?;
    executor.stream.synchronize()?;
    let restored = executor.read_tokens(executor.head_buffers.output_tokens)?;
    anyhow::ensure!(
        restored == eager,
        "DFlash2 static suffix did not restore exact output after anchor restoration"
    );

    let packed_update_graph_rows = dflash2_update_graph_buckets(config.active_requests)?.to_vec();
    executor
        .capture_batched_update_graphs(update_weights, &packed_update_graph_rows)
        .context("capturing DFlash2 packed serving update graphs")?;
    anyhow::ensure!(
        packed_update_graph_rows
            .iter()
            .all(|rows| executor.supports_batched_update_rows(*rows)),
        "DFlash2 packed update registry does not cover its advertised rows"
    );
    let packed_update_graph_validation = executor
        .validate_batched_update_graphs(update_weights, &packed_update_graph_rows)
        .context("validating DFlash2 packed serving update graphs")?;

    for _ in 0..config.warmup {
        executor.replay_cycle()?;
    }
    executor.stream.synchronize()?;
    let update_gpu = benchmark_graph_pair(&executor, ReplayKind::Update)?;
    let suffix_gpu = benchmark_graph_pair(&executor, ReplayKind::Suffix)?;
    let (full_gpu, full_host) = benchmark_full_cycle(&executor)?;

    Ok(Dflash2StaticGraphReport {
        backend: match config.kv_storage {
            DsparkKvStorage::Bf16 => "fixed-address-bf16-flashinfer-fa2-windowed-dflash2",
            DsparkKvStorage::Fp8 => "fixed-address-fp8-flashinfer-fa2-windowed-dflash2",
        },
        kv_storage: config.kv_storage,
        active_requests: config.active_requests,
        accepted_rows_per_request: config.accepted_rows_per_request,
        update_rows: executor.update_rows,
        query_rows_per_request: config.proposal_tokens_per_request + 1,
        proposal_tokens_per_request: config.proposal_tokens_per_request,
        context_tokens: config.context_tokens,
        body_kv_tokens: executor.body_kv_tokens,
        page_size: config.page_size,
        total_physical_pages: executor.total_physical_pages,
        max_pages_per_request: executor.max_pages_per_request,
        shared_kv_bytes: executor.shared_kv_bytes,
        rust_owned_mutable_bytes: executor.arena.bytes,
        resident_weight_bytes: weights.draft_resident_bytes,
        update_graph_nodes: executor.update_graph.node_count,
        update_graph_kernel_nodes: executor.update_graph.kernel_node_count,
        base_update_graph_validation,
        packed_update_graph_rows,
        packed_update_graph_validation,
        suffix_graph_nodes: executor.suffix_graph.node_count,
        suffix_graph_kernel_nodes: executor.suffix_graph.kernel_node_count,
        eager_replay_exact: true,
        dynamic_anchor_changes_output: true,
        dynamic_token_changed_bytes,
        restored_replay_exact: true,
        gpu_ms_per_update_replay: update_gpu,
        gpu_ms_per_suffix_replay: suffix_gpu,
        gpu_ms_per_full_cycle: full_gpu,
        host_ms_per_full_cycle: full_host,
        target_embedding_alias: true,
        target_lm_head_alias: true,
        body_output_is_head_source: true,
        shared_update_body_kv: true,
        candidate_topk_sorted: true,
        candidate_score_accumulation: "bf16-edge-plus-unary-bf16",
        sliding_window_tokens: super::dflash::GLM53_DFLASH2_SLIDING_WINDOW,
        cold_capture_python_calls: 6,
        hot_replay_python_calls: 0,
    })
}

pub(super) struct Dflash2StaticExecutor {
    library: &'static NativeLibrary,
    update_graph: DsparkCudaGraph,
    suffix_graph: DsparkCudaGraph,
    suffix_page_graphs: BTreeMap<usize, DsparkCudaGraph>,
    stream: DsparkCudaStream,
    arena: Dflash2StaticArena,
    query_token_ids: GlmrtDeviceBuffer,
    update_buffers: Dflash2UpdateBuffers,
    body_buffers: Dflash2BodyBuffers,
    head_buffers: Dflash2HeadBuffers,
    paged_kv_metadata: DsparkPagedKvMetadataBuffers,
    config: Dflash2StaticBenchConfig,
    update_rows: usize,
    body_kv_tokens: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    request_page_tables: Vec<Vec<i32>>,
    shared_kv_bytes: u64,
    initial_anchors: Vec<u32>,
    dynamic_anchors: Vec<u32>,
    batched_update_graphs: Option<Dflash2BatchedUpdateGraphs>,
}

struct Dflash2BatchedUpdateGraphs {
    buffers: Dflash2UpdateBuffers,
    graphs: BTreeMap<usize, DsparkCudaGraph>,
    max_rows: usize,
}

#[allow(clippy::too_many_arguments)]
fn capture_dflash2_suffix_graph(
    library: &'static NativeLibrary,
    stream: &DsparkCudaStream,
    weights: Dflash2ResidentWeights,
    query_token_ids: GlmrtDeviceBuffer,
    body_buffers: Dflash2BodyBuffers,
    body_config: Dflash2BodyConfig,
    head_buffers: Dflash2HeadBuffers,
    head_config: Dflash2HeadConfig,
    total_query_rows: usize,
    label: &str,
) -> Result<DsparkCudaGraph> {
    unsafe {
        library
            .cuda_graph_begin_capture(stream.raw)
            .with_context(|| format!("beginning {label} capture"))?;
    }
    let capture_result = (|| {
        launch_embedding(
            library,
            stream.raw,
            weights.target_embedding,
            query_token_ids,
            body_buffers.input,
            total_query_rows,
        )?;
        launch_python_dflash2_body(
            stream.raw,
            body_buffers,
            weights,
            body_config,
            "capture_dspark_cudnn_paged_body",
        )?;
        launch_python_dflash2_head(
            stream.raw,
            head_buffers,
            Dflash2HeadResidentWeights::from(weights),
            head_config,
            "capture_dflash2_head",
        )
    })();
    if let Err(error) = capture_result {
        unsafe {
            let _ = library.cuda_graph_end_capture_retained(stream.raw);
        }
        return Err(error).with_context(|| format!("capturing {label}"));
    }
    let capture = unsafe {
        library
            .cuda_graph_end_capture_retained(stream.raw)
            .with_context(|| format!("ending {label} capture"))?
    };
    let graph = DsparkCudaGraph::new(library, capture)?;
    graph.validate_min_kernel_nodes(GLM53_DFLASH2_DRAFT_LAYERS + 2)?;
    Ok(graph)
}

impl Dflash2StaticExecutor {
    pub(super) fn capture(
        weights: Dflash2ResidentWeights,
        config: Dflash2StaticBenchConfig,
    ) -> Result<Self> {
        Self::capture_with_physical_pages(weights, config, None)
    }

    pub(super) fn capture_with_physical_pages(
        weights: Dflash2ResidentWeights,
        config: Dflash2StaticBenchConfig,
        physical_kv_pages: Option<usize>,
    ) -> Result<Self> {
        Self::capture_with_kv_pool(weights, config, physical_kv_pages, None)
    }

    pub(super) fn capture_with_shared_kv_pool(
        weights: Dflash2ResidentWeights,
        config: Dflash2StaticBenchConfig,
        shared_kv: Dflash2SharedKvPool,
    ) -> Result<Self> {
        Self::capture_with_kv_pool(
            weights,
            config,
            Some(shared_kv.total_physical_pages),
            Some(shared_kv),
        )
    }

    fn capture_with_kv_pool(
        weights: Dflash2ResidentWeights,
        config: Dflash2StaticBenchConfig,
        physical_kv_pages: Option<usize>,
        shared_kv: Option<Dflash2SharedKvPool>,
    ) -> Result<Self> {
        validate_config(config)?;
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let update_rows = checked_mul(
            config.active_requests,
            config.accepted_rows_per_request,
            "DFlash2 static update rows",
        )?;
        let query_rows_per_request = config.proposal_tokens_per_request + 1;
        let total_query_rows = checked_mul(
            config.active_requests,
            query_rows_per_request,
            "DFlash2 static query rows",
        )?;
        let context_after_update = config
            .context_tokens
            .checked_add(config.accepted_rows_per_request)
            .context("DFlash2 static updated context overflow")?;
        let body_kv_tokens = context_after_update
            .checked_add(query_rows_per_request)
            .context("DFlash2 static body KV length overflow")?;
        let physical_pages_per_request = if config.allocate_full_kv_capacity {
            config.kv_capacity_tokens.div_ceil(config.page_size)
        } else {
            body_kv_tokens.div_ceil(config.page_size)
        };
        let minimum_physical_pages = checked_mul(
            config.active_requests,
            physical_pages_per_request,
            "DFlash2 static physical pages",
        )?;
        let total_physical_pages = physical_kv_pages.unwrap_or(minimum_physical_pages);
        anyhow::ensure!(
            total_physical_pages >= minimum_physical_pages,
            "DFlash2 physical KV pool has {total_physical_pages} pages but the executor requires at least {minimum_physical_pages}"
        );
        let max_pages_per_request = config.kv_capacity_tokens.div_ceil(config.page_size);

        let body_config = Dflash2BodyConfig {
            active_requests: config.active_requests,
            query_rows_per_request,
            total_pages: total_physical_pages,
            page_size: config.page_size,
            max_pages_per_request,
            planning_pages_per_request: max_pages_per_request,
            fixed_split_pages: 0,
            kv_storage: config.kv_storage,
            seed: config.seed,
            initialize_input: false,
            initialize_kv: false,
        };
        let update_config = Dflash2UpdateConfig {
            rows: update_rows,
            active_requests: config.active_requests,
            total_pages: total_physical_pages,
            page_size: config.page_size,
            max_pages_per_request,
            kv_storage: config.kv_storage,
            seed: config.seed,
            initialize_target_hidden: true,
            initialize_kv: shared_kv.is_none(),
        };

        let mut arena = Dflash2StaticArena::default();
        let body_plan = dflash2_body_buffer_plan(body_config)?;
        let update_plan = dflash2_update_buffer_plan(update_config)?;
        let head_plan =
            dflash2_head_buffer_plan(config.active_requests, config.proposal_tokens_per_request)?;
        let k_cache_bytes = plan_bytes(&body_plan, "k_cache")?;
        let v_cache_bytes = plan_bytes(&body_plan, "v_cache")?;
        anyhow::ensure!(
            k_cache_bytes == plan_bytes(&update_plan, "k_cache")?
                && v_cache_bytes == plan_bytes(&update_plan, "v_cache")?,
            "DFlash2 update/body KV plans disagree"
        );
        let (k_cache, v_cache) = if let Some(shared) = shared_kv {
            anyhow::ensure!(
                shared.k_cache.device_id == shared.v_cache.device_id
                    && shared.k_cache.bytes >= k_cache_bytes
                    && shared.v_cache.bytes >= v_cache_bytes,
                "DFlash2 shared KV pool does not match the suffix executor"
            );
            (shared.k_cache, shared.v_cache)
        } else {
            (
                arena.allocate(library, k_cache_bytes, "DFlash2 static paged K")?,
                arena.allocate(library, v_cache_bytes, "DFlash2 static paged V")?,
            )
        };
        let shared_kv_bytes = (k_cache_bytes as u64)
            .checked_add(v_cache_bytes as u64)
            .context("DFlash2 shared KV byte count overflow")?;
        let block_tables = arena.allocate(
            library,
            plan_bytes(&body_plan, "block_tables")?,
            "DFlash2 static block tables",
        )?;

        let mut body = allocate_plan(
            &mut arena,
            library,
            body_plan.iter().map(|item| (item.name, item.bytes)),
            &["k_cache", "v_cache", "block_tables"],
            "DFlash2 static body",
        )?;
        body.insert("k_cache", k_cache);
        body.insert("v_cache", v_cache);
        body.insert("block_tables", block_tables);
        let body_buffers = Dflash2BodyBuffers {
            input: get(&body, "input")?,
            output: get(&body, "output")?,
            reference_output: get(&body, "reference_output")?,
            hidden_attention: get(&body, "hidden_attention")?,
            hidden_mlp: get(&body, "hidden_mlp")?,
            normalized: get(&body, "normalized")?,
            qkv: get(&body, "qkv")?,
            q: get(&body, "q")?,
            attention: get(&body, "attention")?,
            delta: get(&body, "delta")?,
            gate_up: get(&body, "gate_up")?,
            activation: get(&body, "activation")?,
            conv_dynamic: get(&body, "conv_dynamic")?,
            conv_output: get(&body, "conv_output")?,
            k_cache,
            v_cache,
            workspace: get(&body, "workspace")?,
            query_lengths: get(&body, "query_lengths")?,
            kv_lengths: get(&body, "kv_lengths")?,
            query_positions: get(&body, "query_positions")?,
            block_tables,
            query_offsets: get(&body, "query_offsets")?,
            output_offsets: get(&body, "output_offsets")?,
            query_indptr: get(&body, "query_indptr")?,
            kv_indptr: get(&body, "kv_indptr")?,
            page_indices: get(&body, "page_indices")?,
            last_page_len: get(&body, "last_page_len")?,
        };

        let mut update = allocate_plan(
            &mut arena,
            library,
            update_plan.iter().map(|item| (item.name, item.bytes)),
            &["k_cache", "v_cache", "block_tables"],
            "DFlash2 static update",
        )?;
        update.insert("k_cache", k_cache);
        update.insert("v_cache", v_cache);
        update.insert("block_tables", block_tables);
        let update_buffers = Dflash2UpdateBuffers {
            target_hidden: get(&update, "target_hidden")?,
            fusion_output: get(&update, "fusion_output")?,
            fused_hidden: get(&update, "fused_hidden")?,
            projected_kv: get(&update, "projected_kv")?,
            key_output: get(&update, "key_output")?,
            value_output: get(&update, "value_output")?,
            reference_fused_hidden: get(&update, "reference_fused_hidden")?,
            reference_key_output: get(&update, "reference_key_output")?,
            reference_value_output: get(&update, "reference_value_output")?,
            eager_fused_hidden: get(&update, "eager_fused_hidden")?,
            eager_key_output: get(&update, "eager_key_output")?,
            eager_value_output: get(&update, "eager_value_output")?,
            k_cache,
            v_cache,
            row_request_ids: get(&update, "row_request_ids")?,
            row_positions: get(&update, "row_positions")?,
            row_cache_positions: get(&update, "row_cache_positions")?,
            block_tables,
        };

        let mut head = allocate_plan(
            &mut arena,
            library,
            head_plan.iter().map(|item| (item.name, item.bytes)),
            &["hidden"],
            "DFlash2 static head",
        )?;
        anyhow::ensure!(
            body_buffers.output.bytes >= plan_bytes(&head_plan, "hidden")?,
            "DFlash2 body output is too small for the aliased head input"
        );
        head.insert("hidden", body_buffers.output);
        let head_buffers = Dflash2HeadBuffers {
            hidden: body_buffers.output,
            hidden_position_major: get(&head, "hidden_position_major")?,
            logits: get(&head, "logits")?,
            unary: get(&head, "unary")?,
            candidates: get(&head, "candidates")?,
            radix_candidates: get(&head, "radix_candidates")?,
            radix_row_states: get(&head, "radix_row_states")?,
            projected_hidden: get(&head, "projected_hidden")?,
            token_steps: get(&head, "token_steps")?,
            anchor_tokens: get(&head, "anchor_tokens")?,
            output_tokens: get(&head, "output_tokens")?,
            reference_tokens: get(&head, "reference_tokens")?,
            eager_tokens: get(&head, "eager_tokens")?,
        };
        let query_token_ids = arena.allocate(
            library,
            checked_mul(total_query_rows, 4, "DFlash2 query token IDs")?,
            "DFlash2 static query token IDs",
        )?;

        let mut block_table = vec![0_i32; config.active_requests * max_pages_per_request];
        for request in 0..config.active_requests {
            for page in 0..physical_pages_per_request {
                block_table[request * max_pages_per_request + page] =
                    i32::try_from(request * physical_pages_per_request + page)
                        .context("DFlash2 physical page ID does not fit i32")?;
            }
        }
        library
            .copy_h2d(block_tables, as_bytes(&block_table))
            .context("uploading DFlash2 static block table")?;
        let request_page_tables = block_table
            .chunks_exact(max_pages_per_request)
            .map(<[i32]>::to_vec)
            .collect::<Vec<_>>();
        let body_lengths = vec![
            i32::try_from(body_kv_tokens)
                .context("DFlash2 body KV length does not fit i32")?;
            config.active_requests
        ];
        library
            .copy_h2d(body_buffers.kv_lengths, as_bytes(&body_lengths))
            .context("uploading DFlash2 static KV lengths")?;
        let paged_buffers = DsparkPagedKvMetadataBuffers {
            query_indptr: body_buffers.query_indptr,
            kv_indptr: body_buffers.kv_indptr,
            page_indices: body_buffers.page_indices,
            last_page_len: body_buffers.last_page_len,
        };
        DsparkPagedKvMetadata::for_lengths(
            &body_lengths,
            query_rows_per_request,
            config.page_size,
            physical_pages_per_request,
        )?
        .upload(library, paged_buffers)?;
        let query_positions = (0..config.active_requests)
            .flat_map(|_| {
                (0..query_rows_per_request).map(|row| {
                    i32::try_from(context_after_update + row)
                        .context("DFlash2 query position does not fit i32")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        library
            .copy_h2d(body_buffers.query_positions, as_bytes(&query_positions))
            .context("uploading DFlash2 query positions")?;
        let mut request_ids = Vec::with_capacity(update_rows);
        let mut update_positions = Vec::with_capacity(update_rows);
        for request in 0..config.active_requests {
            for row in 0..config.accepted_rows_per_request {
                request_ids.push(i32::try_from(request).context("DFlash2 request ID overflow")?);
                update_positions.push(
                    i32::try_from(config.context_tokens + row)
                        .context("DFlash2 update position does not fit i32")?,
                );
            }
        }
        library
            .copy_h2d(update_buffers.row_request_ids, as_bytes(&request_ids))
            .context("uploading DFlash2 update request IDs")?;
        library
            .copy_h2d(update_buffers.row_positions, as_bytes(&update_positions))
            .context("uploading DFlash2 update positions")?;
        library
            .copy_h2d(
                update_buffers.row_cache_positions,
                as_bytes(&update_positions),
            )
            .context("uploading DFlash2 update cache positions")?;

        let initial_anchors = (0..config.active_requests)
            .map(|request| normalized_anchor(config.seed + request as i64 * 104_729))
            .collect::<Vec<_>>();
        let dynamic_anchors = initial_anchors
            .iter()
            .map(|token| normalized_anchor(if *token == 1 { 0 } else { 1 }))
            .collect::<Vec<_>>();
        upload_anchors(
            library,
            query_token_ids,
            head_buffers.anchor_tokens,
            &initial_anchors,
            config.proposal_tokens_per_request,
        )?;

        let update_weights = dflash2_update_resident_weights(weights)?;
        launch_python_dflash2_update(
            stream.raw,
            update_buffers,
            update_weights,
            update_config,
            "prepare_dspark_context_update",
        )?;
        stream.synchronize()?;
        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning DFlash2 static update capture")?;
        }
        if let Err(error) = launch_python_dflash2_update(
            stream.raw,
            update_buffers,
            update_weights,
            update_config,
            "capture_dspark_context_update",
        ) {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing DFlash2 static target-context update");
        }
        let update_capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending DFlash2 static update capture")?
        };
        let update_graph = DsparkCudaGraph::new(library, update_capture)?;
        update_graph.validate()?;

        launch_embedding(
            library,
            stream.raw,
            weights.target_embedding,
            query_token_ids,
            body_buffers.input,
            total_query_rows,
        )?;
        launch_python_dflash2_body(
            stream.raw,
            body_buffers,
            weights,
            body_config,
            "prepare_dspark_cudnn_paged_body",
        )?;
        let head_config = Dflash2HeadConfig {
            active_requests: config.active_requests,
            proposal_tokens_per_request: config.proposal_tokens_per_request,
            seed: config.seed,
            initialize_hidden: false,
        };
        launch_python_dflash2_head(
            stream.raw,
            head_buffers,
            Dflash2HeadResidentWeights::from(weights),
            head_config,
            "prepare_dflash2_head",
        )?;
        stream.synchronize()?;
        let suffix_graph = capture_dflash2_suffix_graph(
            library,
            &stream,
            weights,
            query_token_ids,
            body_buffers,
            body_config,
            head_buffers,
            head_config,
            total_query_rows,
            "DFlash2 static suffix",
        )?;
        let mut suffix_page_graphs = BTreeMap::new();
        if config.capture_page_buckets {
            anyhow::ensure!(
                config.active_requests == 1 && config.allocate_full_kv_capacity,
                "DFlash2 page-count suffix buckets require a full-capacity C1 executor"
            );
            for planning_pages in 1..=max_pages_per_request {
                let planning_tokens = planning_pages
                    .checked_mul(config.page_size)
                    .context("DFlash2 page-bucket token count overflow")?;
                let planning_length = i32::try_from(planning_tokens)
                    .context("DFlash2 page-bucket token count does not fit i32")?;
                library
                    .copy_h2d(
                        body_buffers.kv_lengths,
                        as_bytes(std::slice::from_ref(&planning_length)),
                    )
                    .context("uploading DFlash2 page-bucket KV length")?;
                DsparkPagedKvMetadata::for_page_tables(
                    &[planning_length],
                    query_rows_per_request,
                    config.page_size,
                    &request_page_tables,
                    total_physical_pages,
                )?
                .upload(library, paged_buffers)?;
                let page_body_config = Dflash2BodyConfig {
                    planning_pages_per_request: planning_pages,
                    // Two pages keep the split count invariant while the final
                    // page fills and matched freshly replanned FA2 bit-for-bit
                    // across every last-page length in qualification.
                    fixed_split_pages: 2,
                    ..body_config
                };
                launch_python_dflash2_body(
                    stream.raw,
                    body_buffers,
                    weights,
                    page_body_config,
                    "prepare_dspark_cudnn_paged_body",
                )?;
                stream.synchronize()?;
                let graph = capture_dflash2_suffix_graph(
                    library,
                    &stream,
                    weights,
                    query_token_ids,
                    body_buffers,
                    page_body_config,
                    head_buffers,
                    head_config,
                    total_query_rows,
                    &format!("DFlash2 C1 {planning_pages}-page suffix"),
                )?;
                suffix_page_graphs.insert(planning_pages, graph);
            }

            library
                .copy_h2d(body_buffers.kv_lengths, as_bytes(&body_lengths))
                .context("restoring DFlash2 base KV lengths after page capture")?;
            DsparkPagedKvMetadata::for_page_tables(
                &body_lengths,
                query_rows_per_request,
                config.page_size,
                &request_page_tables,
                total_physical_pages,
            )?
            .upload(library, paged_buffers)?;
            launch_python_dflash2_body(
                stream.raw,
                body_buffers,
                weights,
                body_config,
                "prepare_dspark_cudnn_paged_body",
            )?;
            stream.synchronize()?;
        }

        Ok(Self {
            library,
            update_graph,
            suffix_graph,
            suffix_page_graphs,
            stream,
            arena,
            query_token_ids,
            update_buffers,
            body_buffers,
            head_buffers,
            paged_kv_metadata: paged_buffers,
            config,
            update_rows,
            body_kv_tokens,
            total_physical_pages,
            max_pages_per_request,
            request_page_tables,
            shared_kv_bytes,
            initial_anchors,
            dynamic_anchors,
            batched_update_graphs: None,
        })
    }

    pub(super) fn shared_kv_pool(&self) -> Dflash2SharedKvPool {
        Dflash2SharedKvPool {
            k_cache: self.body_buffers.k_cache,
            v_cache: self.body_buffers.v_cache,
            total_physical_pages: self.total_physical_pages,
        }
    }

    pub(super) fn suffix_page_graph_count(&self) -> usize {
        self.suffix_page_graphs.len()
    }

    pub(super) fn capture_batched_update_graphs(
        &mut self,
        weights: Dflash2UpdateResidentWeights,
        row_buckets: &[usize],
    ) -> Result<()> {
        anyhow::ensure!(
            matches!(self.config.active_requests, 1 | 2 | 4)
                && self.config.accepted_rows_per_request == 1,
            "batched DFlash2 serving updates require a C1/C2/C4 one-row base executor"
        );
        anyhow::ensure!(
            self.batched_update_graphs.is_none(),
            "batched DFlash2 update graphs were already captured"
        );
        anyhow::ensure!(
            !row_buckets.is_empty()
                && row_buckets.windows(2).all(|pair| pair[0] < pair[1])
                && row_buckets
                    .iter()
                    .all(|rows| {
                        *rows >= self.config.active_requests
                            && *rows <= 1_024
                            && (rows.is_power_of_two()
                                || (self.config.active_requests == 1 && *rows <= 8))
                    }),
            "DFlash2 update rows must be unique ascending C1 exact-small or C2/C4 power-of-two buckets through 1024"
        );
        let max_rows = *row_buckets
            .last()
            .expect("non-empty DFlash2 update buckets were checked above");
        let buffers = allocate_batched_update_buffers(
            &mut self.arena,
            self.library,
            max_rows,
            self.total_physical_pages,
            self.max_pages_per_request,
            self.config,
            self.config.active_requests,
            self.update_buffers.k_cache,
            self.update_buffers.v_cache,
            self.update_buffers.block_tables,
        )?;
        let mut graphs = BTreeMap::new();
        for &rows in row_buckets {
            let request_ids = (0..rows)
                .map(|row| {
                    i32::try_from(row % self.config.active_requests)
                        .context("DFlash2 update request ID does not fit i32")
                })
                .collect::<Result<Vec<_>>>()?;
            let positions = (0..rows)
                .map(|row| i32::try_from(row).context("DFlash2 update row does not fit i32"))
                .collect::<Result<Vec<_>>>()?;
            self.library
                .copy_h2d(buffers.row_request_ids, as_bytes(&request_ids))
                .context("uploading DFlash2 batched update request IDs")?;
            self.library
                .copy_h2d(buffers.row_positions, as_bytes(&positions))
                .context("uploading DFlash2 batched update positions")?;
            self.library
                .copy_h2d(buffers.row_cache_positions, as_bytes(&positions))
                .context("uploading DFlash2 batched cache positions")?;
            let update_config = Dflash2UpdateConfig {
                rows,
                active_requests: self.config.active_requests,
                total_pages: self.total_physical_pages,
                page_size: self.config.page_size,
                max_pages_per_request: self.max_pages_per_request,
                kv_storage: self.config.kv_storage,
                seed: self.config.seed + rows as i64,
                initialize_target_hidden: true,
                initialize_kv: false,
            };
            launch_python_dflash2_update(
                self.stream.raw,
                buffers,
                weights,
                update_config,
                "prepare_dspark_context_update",
            )
            .with_context(|| format!("preparing {rows}-row DFlash2 serving update"))?;
            self.stream.synchronize()?;
            unsafe {
                self.library
                    .cuda_graph_begin_capture(self.stream.raw)
                    .with_context(|| format!("beginning {rows}-row DFlash2 update capture"))?;
            }
            if let Err(error) = launch_python_dflash2_update(
                self.stream.raw,
                buffers,
                weights,
                update_config,
                "capture_dspark_context_update",
            ) {
                unsafe {
                    let _ = self
                        .library
                        .cuda_graph_end_capture_retained(self.stream.raw);
                }
                return Err(error)
                    .with_context(|| format!("capturing {rows}-row DFlash2 update graph"));
            }
            let capture = unsafe {
                self.library
                    .cuda_graph_end_capture_retained(self.stream.raw)
                    .with_context(|| format!("ending {rows}-row DFlash2 update capture"))?
            };
            let graph = DsparkCudaGraph::new(self.library, capture)?;
            graph.validate()?;
            graphs.insert(rows, graph);
        }
        self.batched_update_graphs = Some(Dflash2BatchedUpdateGraphs {
            buffers,
            graphs,
            max_rows,
        });
        Ok(())
    }

    fn validate_batched_update_graphs(
        &self,
        weights: Dflash2UpdateResidentWeights,
        row_buckets: &[usize],
    ) -> Result<Vec<Dflash2UpdateGraphValidation>> {
        let set = self
            .batched_update_graphs
            .as_ref()
            .context("DFlash2 packed update graphs were not captured")?;
        anyhow::ensure!(
            row_buckets.len() == set.graphs.len()
                && row_buckets.iter().all(|rows| set.graphs.contains_key(rows)),
            "DFlash2 packed update validation rows do not match the captured registry"
        );
        let mut validation = Vec::with_capacity(row_buckets.len());
        for &rows in row_buckets {
            let request_ids = (0..rows)
                .map(|row| {
                    i32::try_from(row % self.config.active_requests)
                        .context("DFlash2 validation request ID does not fit i32")
                })
                .collect::<Result<Vec<_>>>()?;
            let positions = (0..rows)
                .map(|row| i32::try_from(row).context("DFlash2 validation row does not fit i32"))
                .collect::<Result<Vec<_>>>()?;
            self.library
                .copy_h2d(set.buffers.row_request_ids, as_bytes(&request_ids))
                .context("uploading DFlash2 validation request IDs")?;
            self.library
                .copy_h2d(set.buffers.row_positions, as_bytes(&positions))
                .context("uploading DFlash2 validation positions")?;
            self.library
                .copy_h2d(set.buffers.row_cache_positions, as_bytes(&positions))
                .context("uploading DFlash2 validation cache positions")?;
            let update_config = Dflash2UpdateConfig {
                rows,
                active_requests: self.config.active_requests,
                total_pages: self.total_physical_pages,
                page_size: self.config.page_size,
                max_pages_per_request: self.max_pages_per_request,
                kv_storage: self.config.kv_storage,
                seed: self.config.seed + rows as i64,
                initialize_target_hidden: true,
                initialize_kv: false,
            };
            launch_python_dflash2_update(
                self.stream.raw,
                set.buffers,
                weights,
                update_config,
                "prepare_dspark_context_update",
            )
            .with_context(|| format!("preparing {rows}-row DFlash2 update validation"))?;
            self.stream.synchronize()?;
            let graph = set
                .graphs
                .get(&rows)
                .expect("DFlash2 validation checked every row above");
            validation.push(self.validate_update_graph(set.buffers, graph, rows, &positions)?);
        }
        Ok(validation)
    }

    fn validate_update_graph(
        &self,
        buffers: Dflash2UpdateBuffers,
        graph: &DsparkCudaGraph,
        rows: usize,
        initial_positions: &[i32],
    ) -> Result<Dflash2UpdateGraphValidation> {
        anyhow::ensure!(
            initial_positions.len() == rows,
            "{rows}-row DFlash2 update validation has {} positions",
            initial_positions.len()
        );
        let replay = || -> Result<()> {
            graph.validate()?;
            unsafe {
                self.library
                    .cuda_graph_launch(graph.exec_raw, self.stream.raw)
                    .with_context(|| {
                        format!("replaying {rows}-row DFlash2 update validation graph")
                    })?;
            }
            self.stream.synchronize()
        };
        replay()?;
        let mut validation = self.validate_update_outputs(buffers, rows)?;
        let eager_keys = self.read_update_output(
            buffers.eager_key_output,
            rows,
            "DFlash2 eager keys for dynamic-position validation",
        )?;
        let dynamic_positions = initial_positions
            .iter()
            .map(|position| {
                position
                    .checked_add(
                        i32::try_from(self.config.page_size)
                            .context("DFlash2 validation page size does not fit i32")?,
                    )
                    .context("DFlash2 dynamic validation position overflow")
            })
            .collect::<Result<Vec<_>>>()?;
        self.library
            .copy_h2d(buffers.row_positions, as_bytes(&dynamic_positions))
            .context("uploading dynamic DFlash2 validation positions")?;
        replay()?;
        let dynamic_keys = self.read_update_output(
            buffers.key_output,
            rows,
            "DFlash2 dynamic-position replay keys",
        )?;
        validation.dynamic_key_changed_bytes = byte_mismatch_count(&eager_keys, &dynamic_keys);
        validation.dynamic_positions_change_keys = validation.dynamic_key_changed_bytes > 0;
        anyhow::ensure!(
            validation.dynamic_positions_change_keys,
            "{rows}-row DFlash2 update graph ignored changed row positions"
        );

        self.library
            .copy_h2d(buffers.row_positions, as_bytes(initial_positions))
            .context("restoring DFlash2 validation positions")?;
        replay()?;
        let restored = self.validate_update_outputs(buffers, rows)?;
        validation.restored_replay_exact = restored.eager_replay_exact;
        anyhow::ensure!(
            validation.restored_replay_exact,
            "{rows}-row DFlash2 update graph did not restore exact output"
        );
        Ok(validation)
    }

    fn validate_update_outputs(
        &self,
        buffers: Dflash2UpdateBuffers,
        rows: usize,
    ) -> Result<Dflash2UpdateGraphValidation> {
        let hidden_bytes = checked_mul(
            checked_mul(
                rows,
                super::dflash::GLM53_DFLASH2_HIDDEN_SIZE,
                "DFlash2 update validation hidden elements",
            )?,
            std::mem::size_of::<u16>(),
            "DFlash2 update validation hidden bytes",
        )?;
        let output_bytes = checked_mul(
            checked_mul(
                checked_mul(
                    GLM53_DFLASH2_DRAFT_LAYERS,
                    rows,
                    "DFlash2 update validation output rows",
                )?,
                GLM53_DFLASH2_KV_HEADS * GLM53_DFLASH2_HEAD_DIM,
                "DFlash2 update validation output elements",
            )?,
            std::mem::size_of::<u16>(),
            "DFlash2 update validation output bytes",
        )?;
        let read = |buffer, bytes, label: &str| -> Result<Vec<u8>> {
            let mut host = vec![0_u8; bytes];
            self.library
                .copy_d2h(&mut host, buffer)
                .with_context(|| format!("reading {label}"))?;
            Ok(host)
        };
        let reference_fused = read(
            buffers.reference_fused_hidden,
            hidden_bytes,
            "DFlash2 reference fused hidden",
        )?;
        let reference_keys = read(
            buffers.reference_key_output,
            output_bytes,
            "DFlash2 reference keys",
        )?;
        let reference_values = read(
            buffers.reference_value_output,
            output_bytes,
            "DFlash2 reference values",
        )?;
        let eager_fused = read(
            buffers.eager_fused_hidden,
            hidden_bytes,
            "DFlash2 eager fused hidden",
        )?;
        let eager_keys = read(buffers.eager_key_output, output_bytes, "DFlash2 eager keys")?;
        let eager_values = read(
            buffers.eager_value_output,
            output_bytes,
            "DFlash2 eager values",
        )?;
        let replay_fused = read(
            buffers.fused_hidden,
            hidden_bytes,
            "DFlash2 replay fused hidden",
        )?;
        let replay_keys = read(buffers.key_output, output_bytes, "DFlash2 replay keys")?;
        let replay_values = read(buffers.value_output, output_bytes, "DFlash2 replay values")?;
        let fused_difference = bf16_difference(&reference_fused, &replay_fused)?;
        let key_difference = bf16_difference(&reference_keys, &replay_keys)?;
        let value_difference = bf16_difference(&reference_values, &replay_values)?;
        anyhow::ensure!(
            fused_difference.max_abs <= 0.125 && fused_difference.relative_l2 <= 0.01,
            "{rows}-row DFlash2 update fused hidden exceeds its numerical gate: {fused_difference:?}"
        );
        anyhow::ensure!(
            key_difference.bf16_steps_at_max_abs <= 4 && key_difference.relative_l2 <= 0.01,
            "{rows}-row DFlash2 update keys exceed its numerical gate: {key_difference:?}"
        );
        anyhow::ensure!(
            value_difference.bf16_steps_at_max_abs <= 1 && value_difference.relative_l2 <= 0.01,
            "{rows}-row DFlash2 update values exceed its numerical gate: {value_difference:?}"
        );
        let eager_replay_exact = eager_fused == replay_fused
            && eager_keys == replay_keys
            && eager_values == replay_values;
        anyhow::ensure!(
            eager_replay_exact,
            "{rows}-row DFlash2 update graph replay differs from eager output"
        );
        Ok(Dflash2UpdateGraphValidation {
            rows,
            reference_fused_hidden_max_abs: fused_difference.max_abs,
            reference_fused_hidden_relative_l2: fused_difference.relative_l2,
            reference_key_max_abs: key_difference.max_abs,
            reference_key_bf16_steps_at_max_abs: key_difference.bf16_steps_at_max_abs,
            reference_key_relative_l2: key_difference.relative_l2,
            reference_value_bf16_steps_at_max_abs: value_difference.bf16_steps_at_max_abs,
            reference_value_relative_l2: value_difference.relative_l2,
            eager_replay_exact,
            dynamic_positions_change_keys: false,
            dynamic_key_changed_bytes: 0,
            restored_replay_exact: false,
        })
    }

    fn read_update_output(
        &self,
        buffer: GlmrtDeviceBuffer,
        rows: usize,
        label: &str,
    ) -> Result<Vec<u8>> {
        let bytes = checked_mul(
            checked_mul(
                checked_mul(
                    GLM53_DFLASH2_DRAFT_LAYERS,
                    rows,
                    "DFlash2 update readback rows",
                )?,
                GLM53_DFLASH2_KV_HEADS * GLM53_DFLASH2_HEAD_DIM,
                "DFlash2 update readback elements",
            )?,
            std::mem::size_of::<u16>(),
            "DFlash2 update readback bytes",
        )?;
        let mut host = vec![0_u8; bytes];
        self.library
            .copy_d2h(&mut host, buffer)
            .with_context(|| format!("reading {label}"))?;
        Ok(host)
    }

    pub(super) fn set_request_page_table(&mut self, page_table: &[i32]) -> Result<()> {
        anyhow::ensure!(
            self.config.active_requests == 1,
            "single-request DFlash2 page-table update requires C=1"
        );
        anyhow::ensure!(
            page_table.len() == self.max_pages_per_request,
            "DFlash2 request page table has {} entries, expected {}",
            page_table.len(),
            self.max_pages_per_request
        );
        anyhow::ensure!(
            page_table.iter().all(|page| {
                *page >= 0
                    && usize::try_from(*page).is_ok_and(|page| page < self.total_physical_pages)
            }),
            "DFlash2 request page table contains an invalid physical page: {page_table:?}"
        );
        self.library
            .copy_h2d(self.body_buffers.block_tables, as_bytes(page_table))
            .context("uploading DFlash2 request block table")?;
        self.request_page_tables[0].copy_from_slice(page_table);
        Ok(())
    }

    pub(super) fn supports_batched_update_rows(&self, actual_rows: usize) -> bool {
        let Some(set) = self.batched_update_graphs.as_ref() else {
            return false;
        };
        let padded_rows =
            dflash2_packed_update_rows(self.config.active_requests, actual_rows, set.max_rows);
        padded_rows.is_some_and(|rows| rows <= set.max_rows && set.graphs.contains_key(&rows))
    }

    pub(super) fn update_batched_request_caches(
        &mut self,
        requests: &[Dflash2BatchedUpdateRequest<'_>],
    ) -> Result<Vec<Dflash2BatchedUpdateStep>> {
        let update_started = Instant::now();
        anyhow::ensure!(
            self.config.active_requests > 1 && requests.len() == self.config.active_requests,
            "DFlash2 batched update expected {} requests, got {}",
            self.config.active_requests,
            requests.len()
        );
        let actual_rows = requests.iter().try_fold(0_usize, |rows, request| {
            anyhow::ensure!(
                request.committed_rows > 0,
                "DFlash2 batched update has no rows"
            );
            rows.checked_add(request.committed_rows)
                .context("DFlash2 batched update row count overflow")
        })?;
        anyhow::ensure!(
            self.supports_batched_update_rows(actual_rows),
            "DFlash2 C={} packed update has no graph for {actual_rows} rows",
            requests.len()
        );
        let set = self
            .batched_update_graphs
            .as_ref()
            .expect("packed update support requires the captured graph set");
        let padded_rows =
            dflash2_packed_update_rows(self.config.active_requests, actual_rows, set.max_rows)
                .context("DFlash2 padded update row count overflow")?;
        let graph = set
            .graphs
            .get(&padded_rows)
            .expect("packed update support checked the padded graph row count");
        let buffers = set.buffers;
        let mut request_ids: Vec<i32> = Vec::with_capacity(padded_rows);
        let mut row_positions: Vec<i32> = Vec::with_capacity(padded_rows);
        let mut row_cache_positions: Vec<i32> = Vec::with_capacity(padded_rows);
        let mut flat_page_tables = Vec::with_capacity(requests.len() * self.max_pages_per_request);
        let feature_bytes = checked_mul(
            super::dflash::GLM53_DFLASH2_HIDDEN_SIZE,
            std::mem::size_of::<u16>(),
            "DFlash2 packed update feature bytes",
        )?;
        let destination_pitch = checked_mul(
            GLM53_DFLASH2_DRAFT_LAYERS,
            feature_bytes,
            "DFlash2 packed target-feature pitch",
        )?;
        let mut packed_row_start = 0_usize;
        for (request_index, request) in requests.iter().enumerate() {
            anyhow::ensure!(
                request.page_table.len() == self.max_pages_per_request
                    && request.page_table.iter().all(|page| {
                        *page >= 0
                            && usize::try_from(*page)
                                .is_ok_and(|page| page < self.total_physical_pages)
                    })
                    && request.target_hidden_taps.iter().all(|tap| {
                        request
                            .target_row_start
                            .checked_add(request.committed_rows)
                            .is_some_and(|row_end| row_end <= tap.rows)
                            && tap.values_per_row == super::dflash::GLM53_DFLASH2_HIDDEN_SIZE
                            && tap.buffer().device_id == buffers.target_hidden.device_id
                    }),
                "DFlash2 packed update request {request_index} has invalid pages or target rows"
            );
            flat_page_tables.extend_from_slice(request.page_table);
            for row in 0..request.committed_rows {
                request_ids.push(
                    i32::try_from(request_index)
                        .context("DFlash2 packed request ID does not fit i32")?,
                );
                row_positions.push(
                    request
                        .absolute_context_start
                        .checked_add(row)
                        .context("DFlash2 packed absolute position overflow")?
                        .try_into()
                        .context("DFlash2 packed absolute position does not fit i32")?,
                );
                row_cache_positions.push(
                    request
                        .cache_context_start
                        .checked_add(row)
                        .context("DFlash2 packed cache position overflow")?
                        .try_into()
                        .context("DFlash2 packed cache position does not fit i32")?,
                );
            }
            for (tap_index, tap) in request.target_hidden_taps.iter().enumerate() {
                tap.wait_ready_on_stream(self.stream.raw).with_context(|| {
                    format!("waiting for DFlash2 packed request {request_index} tap {tap_index}")
                })?;
                let source_bytes = checked_mul(
                    request.committed_rows,
                    feature_bytes,
                    "DFlash2 packed source tap bytes",
                )?;
                let source = device_buffer_byte_view(
                    tap.buffer(),
                    checked_mul(
                        request.target_row_start,
                        feature_bytes,
                        "DFlash2 packed tap source offset",
                    )?,
                    source_bytes,
                    "DFlash2 packed target hidden source rows",
                )?;
                let destination_offset = checked_mul(
                    packed_row_start,
                    destination_pitch,
                    "DFlash2 packed target row offset",
                )?
                .checked_add(checked_mul(
                    tap_index,
                    feature_bytes,
                    "DFlash2 packed target tap offset",
                )?)
                .context("DFlash2 packed target offset overflow")?;
                let destination_span = request
                    .committed_rows
                    .saturating_sub(1)
                    .checked_mul(destination_pitch)
                    .and_then(|bytes| bytes.checked_add(feature_bytes))
                    .context("DFlash2 packed target tap span overflow")?;
                let destination = device_buffer_byte_view(
                    buffers.target_hidden,
                    destination_offset,
                    destination_span,
                    "DFlash2 packed target hidden destination feature",
                )?;
                unsafe {
                    self.library
                        .copy_d2d_2d_async(
                            destination,
                            destination_pitch,
                            source,
                            feature_bytes,
                            feature_bytes,
                            request.committed_rows,
                            self.stream.raw,
                        )
                        .with_context(|| {
                            format!(
                                "packing DFlash2 request {request_index} target tap {tap_index}"
                            )
                        })?;
                }
            }
            packed_row_start = packed_row_start
                .checked_add(request.committed_rows)
                .context("DFlash2 packed target row offset overflow")?;
        }
        debug_assert_eq!(packed_row_start, actual_rows);
        let padding_rows = padded_rows - actual_rows;
        if padding_rows > 0 {
            request_ids.extend_from_within(..padding_rows);
            row_positions.extend_from_within(..padding_rows);
            row_cache_positions.extend_from_within(..padding_rows);
            let source = device_buffer_byte_view(
                buffers.target_hidden,
                0,
                checked_mul(
                    padding_rows,
                    destination_pitch,
                    "DFlash2 packed padding bytes",
                )?,
                "DFlash2 packed padding source rows",
            )?;
            let destination = device_buffer_byte_view(
                buffers.target_hidden,
                checked_mul(
                    actual_rows,
                    destination_pitch,
                    "DFlash2 packed padding destination offset",
                )?,
                source.bytes,
                "DFlash2 packed padding destination rows",
            )?;
            unsafe {
                self.library
                    .copy_d2d_async(destination, source, source.bytes, self.stream.raw)
                    .context("padding DFlash2 packed update rows")?;
            }
        }
        self.library
            .copy_h2d(buffers.row_request_ids, as_bytes(&request_ids))
            .context("uploading DFlash2 packed request IDs")?;
        self.library
            .copy_h2d(buffers.row_positions, as_bytes(&row_positions))
            .context("uploading DFlash2 packed absolute positions")?;
        self.library
            .copy_h2d(buffers.row_cache_positions, as_bytes(&row_cache_positions))
            .context("uploading DFlash2 packed cache positions")?;
        self.library
            .copy_h2d(buffers.block_tables, as_bytes(&flat_page_tables))
            .context("uploading DFlash2 packed request page tables")?;
        graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.exec_raw, self.stream.raw)
                .with_context(|| {
                    format!("launching {padded_rows}-row packed DFlash2 update graph")
                })?;
        }
        self.stream.synchronize()?;
        let update_ms = update_started.elapsed().as_secs_f64() * 1_000.0;
        requests
            .iter()
            .map(|request| {
                Ok(Dflash2BatchedUpdateStep {
                    absolute_context_after_update: request
                        .absolute_context_start
                        .checked_add(request.committed_rows)
                        .context("DFlash2 packed absolute context overflow")?,
                    cache_context_after_update: request
                        .cache_context_start
                        .checked_add(request.committed_rows)
                        .context("DFlash2 packed cache context overflow")?,
                    update_ms,
                })
            })
            .collect()
    }

    pub(super) fn read_request_cache_snapshot(
        &self,
        page_table: &[i32],
        cache_context_tokens: usize,
    ) -> Result<Vec<u8>> {
        let logical_pages = cache_context_tokens.div_ceil(self.config.page_size);
        let page_bytes = self.request_cache_page_bytes()?;
        let snapshot_bytes = checked_mul(
            checked_mul(
                2 * GLM53_DFLASH2_DRAFT_LAYERS,
                logical_pages,
                "DFlash2 request snapshot plane/layer pages",
            )?,
            page_bytes,
            "DFlash2 request snapshot bytes",
        )?;
        self.validate_request_cache_snapshot_layout(page_table, logical_pages)?;
        self.stream
            .synchronize()
            .context("synchronizing before DFlash2 request cache snapshot")?;
        let mut snapshot = vec![0_u8; snapshot_bytes];
        for (plane, cache) in [self.body_buffers.k_cache, self.body_buffers.v_cache]
            .into_iter()
            .enumerate()
        {
            for layer in 0..GLM53_DFLASH2_DRAFT_LAYERS {
                let host_layer_base = (plane * GLM53_DFLASH2_DRAFT_LAYERS + layer)
                    .checked_mul(logical_pages)
                    .and_then(|pages| pages.checked_mul(page_bytes))
                    .context("DFlash2 request snapshot host offset overflow")?;
                copy_cache_pages_to_host(
                    self.library,
                    cache,
                    self.total_physical_pages,
                    layer,
                    page_table,
                    logical_pages,
                    page_bytes,
                    &mut snapshot,
                    host_layer_base,
                )?;
            }
        }
        self.zero_uncommitted_snapshot_tail(&mut snapshot, cache_context_tokens, logical_pages)?;
        Ok(snapshot)
    }

    pub(super) fn restore_request_cache_snapshot(
        &self,
        page_table: &[i32],
        cache_context_tokens: usize,
        snapshot: &[u8],
    ) -> Result<()> {
        let logical_pages = cache_context_tokens.div_ceil(self.config.page_size);
        let page_bytes = self.request_cache_page_bytes()?;
        let expected_bytes = checked_mul(
            checked_mul(
                2 * GLM53_DFLASH2_DRAFT_LAYERS,
                logical_pages,
                "DFlash2 request restore plane/layer pages",
            )?,
            page_bytes,
            "DFlash2 request restore bytes",
        )?;
        anyhow::ensure!(
            snapshot.len() == expected_bytes,
            "DFlash2 request cache snapshot has {} bytes, expected {expected_bytes}",
            snapshot.len()
        );
        self.validate_request_cache_snapshot_layout(page_table, logical_pages)?;
        for (plane, cache) in [self.body_buffers.k_cache, self.body_buffers.v_cache]
            .into_iter()
            .enumerate()
        {
            for layer in 0..GLM53_DFLASH2_DRAFT_LAYERS {
                let host_layer_base = (plane * GLM53_DFLASH2_DRAFT_LAYERS + layer)
                    .checked_mul(logical_pages)
                    .and_then(|pages| pages.checked_mul(page_bytes))
                    .context("DFlash2 request restore host offset overflow")?;
                copy_cache_pages_from_host(
                    self.library,
                    cache,
                    self.total_physical_pages,
                    layer,
                    page_table,
                    logical_pages,
                    page_bytes,
                    snapshot,
                    host_layer_base,
                )?;
            }
        }
        Ok(())
    }

    fn request_cache_page_bytes(&self) -> Result<usize> {
        checked_mul(
            checked_mul(
                GLM53_DFLASH2_KV_HEADS,
                self.config.page_size,
                "DFlash2 request cache page heads/tokens",
            )?,
            checked_mul(
                GLM53_DFLASH2_HEAD_DIM,
                self.config.kv_storage.element_bytes(),
                "DFlash2 request cache head width",
            )?,
            "DFlash2 request cache page",
        )
    }

    fn validate_request_cache_snapshot_layout(
        &self,
        page_table: &[i32],
        logical_pages: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            logical_pages <= page_table.len(),
            "DFlash2 snapshot needs {logical_pages} pages but the request table has {}",
            page_table.len()
        );
        anyhow::ensure!(
            page_table.iter().take(logical_pages).all(|page| {
                *page >= 0
                    && usize::try_from(*page)
                        .is_ok_and(|physical| physical < self.total_physical_pages)
            }),
            "DFlash2 snapshot page table contains an invalid physical page"
        );
        Ok(())
    }

    fn zero_uncommitted_snapshot_tail(
        &self,
        snapshot: &mut [u8],
        cache_context_tokens: usize,
        logical_pages: usize,
    ) -> Result<()> {
        let valid_tokens = cache_context_tokens % self.config.page_size;
        if valid_tokens == 0 || logical_pages == 0 {
            return Ok(());
        }
        let head_token_bytes = checked_mul(
            GLM53_DFLASH2_HEAD_DIM,
            self.config.kv_storage.element_bytes(),
            "DFlash2 snapshot head-token bytes",
        )?;
        let page_bytes = self.request_cache_page_bytes()?;
        let layer_bytes = checked_mul(logical_pages, page_bytes, "DFlash2 snapshot layer bytes")?;
        let final_page_offset = checked_mul(
            logical_pages - 1,
            page_bytes,
            "DFlash2 snapshot final-page offset",
        )?;
        for plane_layer in 0..2 * GLM53_DFLASH2_DRAFT_LAYERS {
            let layer_base = plane_layer
                .checked_mul(layer_bytes)
                .and_then(|offset| offset.checked_add(final_page_offset))
                .context("DFlash2 snapshot plane/layer offset overflow")?;
            for head in 0..GLM53_DFLASH2_KV_HEADS {
                let invalid_start = layer_base
                    .checked_add(
                        head.checked_mul(self.config.page_size)
                            .and_then(|tokens| tokens.checked_add(valid_tokens))
                            .and_then(|tokens| tokens.checked_mul(head_token_bytes))
                            .context("DFlash2 snapshot invalid-tail offset overflow")?,
                    )
                    .context("DFlash2 snapshot invalid-tail start overflow")?;
                let invalid_end = layer_base
                    .checked_add(
                        (head + 1)
                            .checked_mul(self.config.page_size)
                            .and_then(|tokens| tokens.checked_mul(head_token_bytes))
                            .context("DFlash2 snapshot invalid-tail end overflow")?,
                    )
                    .context("DFlash2 snapshot invalid-tail end overflow")?;
                snapshot[invalid_start..invalid_end].fill(0);
            }
        }
        Ok(())
    }

    pub(super) fn update_request_cache_with_cache_context(
        &mut self,
        target_hidden_taps: [&DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS],
        target_row_start: usize,
        committed_rows: usize,
        context_tokens: usize,
        cache_context_tokens: usize,
    ) -> Result<Dflash2RequestUpdate> {
        anyhow::ensure!(
            self.config.active_requests == 1 && self.config.accepted_rows_per_request == 1,
            "DFlash2 request update requires the C=1 one-row update graph"
        );
        anyhow::ensure!(committed_rows > 0, "DFlash2 update needs a committed row");
        anyhow::ensure!(
            target_hidden_taps.iter().all(|tap| {
                target_row_start
                    .checked_add(committed_rows)
                    .is_some_and(|row_end| row_end <= tap.rows)
                    && tap.values_per_row == super::dflash::GLM53_DFLASH2_HIDDEN_SIZE
                    && tap.buffer().device_id == self.update_buffers.target_hidden.device_id
            }),
            "DFlash2 target taps do not contain rows {target_row_start}..{} with width {} on the executor device",
            target_row_start.saturating_add(committed_rows),
            super::dflash::GLM53_DFLASH2_HIDDEN_SIZE,
        );
        let context_after_update = context_tokens
            .checked_add(committed_rows)
            .context("DFlash2 absolute context update overflow")?;
        let cache_context_after_update = cache_context_tokens
            .checked_add(committed_rows)
            .context("DFlash2 cache-context update overflow")?;
        let body_kv_tokens = cache_context_after_update
            .checked_add(self.config.proposal_tokens_per_request + 1)
            .context("DFlash2 body KV length overflow")?;
        anyhow::ensure!(
            body_kv_tokens <= self.config.kv_capacity_tokens,
            "DFlash2 body KV length {body_kv_tokens} exceeds capacity {}",
            self.config.kv_capacity_tokens
        );

        let update_start = Instant::now();
        let feature_bytes = checked_mul(
            super::dflash::GLM53_DFLASH2_HIDDEN_SIZE,
            std::mem::size_of::<u16>(),
            "DFlash2 request feature bytes",
        )?;
        for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
            tap.wait_ready_on_stream(self.stream.raw)
                .with_context(|| format!("waiting for DFlash2 target tap {tap_index}"))?;
        }
        let mut row_offset = 0_usize;
        while row_offset < committed_rows {
            let remaining = committed_rows - row_offset;
            let chunk_rows = self
                .batched_update_graphs
                .as_ref()
                .and_then(|set| {
                    set.graphs
                        .range(..=remaining.min(set.max_rows))
                        .next_back()
                        .map(|(rows, _)| *rows)
                })
                .unwrap_or(1);
            let (buffers, graph) = if chunk_rows == 1 {
                (&self.update_buffers, &self.update_graph)
            } else {
                let set = self
                    .batched_update_graphs
                    .as_ref()
                    .expect("a non-unit DFlash2 bucket requires the batched registry");
                (
                    &set.buffers,
                    set.graphs
                        .get(&chunk_rows)
                        .expect("the DFlash2 update bucket came from this registry"),
                )
            };
            let row_positions = (0..chunk_rows)
                .map(|chunk_row| {
                    context_tokens
                        .checked_add(row_offset)
                        .and_then(|position| position.checked_add(chunk_row))
                        .context("DFlash2 absolute update position overflow")?
                        .try_into()
                        .context("DFlash2 absolute update position does not fit i32")
                })
                .collect::<Result<Vec<i32>>>()?;
            let row_cache_positions = (0..chunk_rows)
                .map(|chunk_row| {
                    cache_context_tokens
                        .checked_add(row_offset)
                        .and_then(|position| position.checked_add(chunk_row))
                        .context("DFlash2 cache update position overflow")?
                        .try_into()
                        .context("DFlash2 cache update position does not fit i32")
                })
                .collect::<Result<Vec<i32>>>()?;
            self.library
                .copy_h2d(buffers.row_positions, as_bytes(&row_positions))
                .context("uploading DFlash2 absolute update positions")?;
            self.library
                .copy_h2d(buffers.row_cache_positions, as_bytes(&row_cache_positions))
                .context("uploading DFlash2 cache update positions")?;
            unsafe {
                let target_row = target_row_start
                    .checked_add(row_offset)
                    .context("DFlash2 source target row overflow")?;
                let source_bytes =
                    checked_mul(chunk_rows, feature_bytes, "DFlash2 source tap bytes")?;
                let destination_pitch = checked_mul(
                    GLM53_DFLASH2_DRAFT_LAYERS,
                    feature_bytes,
                    "DFlash2 target-feature pitch",
                )?;
                for (tap_index, tap) in target_hidden_taps.iter().enumerate() {
                    let source = device_buffer_byte_view(
                        tap.buffer(),
                        checked_mul(target_row, feature_bytes, "DFlash2 tap source offset")?,
                        source_bytes,
                        "DFlash2 target hidden source rows",
                    )?;
                    let destination_span = chunk_rows
                        .saturating_sub(1)
                        .checked_mul(destination_pitch)
                        .and_then(|bytes| bytes.checked_add(feature_bytes))
                        .context("DFlash2 target tap span overflow")?;
                    let destination = device_buffer_byte_view(
                        buffers.target_hidden,
                        checked_mul(tap_index, feature_bytes, "DFlash2 tap destination offset")?,
                        destination_span,
                        "DFlash2 target hidden destination feature",
                    )?;
                    self.library
                        .copy_d2d_2d_async(
                            destination,
                            destination_pitch,
                            source,
                            feature_bytes,
                            feature_bytes,
                            chunk_rows,
                            self.stream.raw,
                        )
                        .with_context(|| format!("copying DFlash2 target tap {tap_index}"))?;
                }
            }
            graph.validate()?;
            unsafe {
                self.library
                    .cuda_graph_launch(graph.exec_raw, self.stream.raw)
                    .with_context(|| {
                        format!("launching {chunk_rows}-row DFlash2 request update graph")
                    })?;
            }
            self.stream.synchronize()?;
            row_offset += chunk_rows;
        }
        let update_ms = update_start.elapsed().as_secs_f64() * 1_000.0;

        Ok(Dflash2RequestUpdate {
            context_tokens,
            committed_rows,
            context_after_update,
            cache_context_after_update,
            update_ms,
        })
    }

    pub(super) fn replay_request_step_with_cache_context(
        &mut self,
        target_hidden_taps: [&DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS],
        target_row_start: usize,
        committed_rows: usize,
        context_tokens: usize,
        cache_context_tokens: usize,
        anchor_token: usize,
    ) -> Result<Dflash2DraftStep> {
        let total_start = Instant::now();
        anyhow::ensure!(
            anchor_token < super::dflash::GLM53_DFLASH2_VOCAB_SIZE
                && anchor_token != GLM53_DFLASH2_MASK_TOKEN_ID,
            "DFlash2 anchor token {anchor_token} is invalid"
        );
        let update = self.update_request_cache_with_cache_context(
            target_hidden_taps,
            target_row_start,
            committed_rows,
            context_tokens,
            cache_context_tokens,
        )?;
        let context_after_update = update.context_after_update;
        let cache_context_after_update = update.cache_context_after_update;
        let body_kv_tokens = cache_context_after_update
            .checked_add(self.config.proposal_tokens_per_request + 1)
            .context("DFlash2 body KV length overflow")?;
        let update_ms = update.update_ms;

        let body_kv_i32 =
            i32::try_from(body_kv_tokens).context("DFlash2 body KV length does not fit i32")?;
        self.library
            .copy_h2d(
                self.body_buffers.kv_lengths,
                as_bytes(std::slice::from_ref(&body_kv_i32)),
            )
            .context("uploading DFlash2 body KV length")?;
        let query_rows_per_request = self.config.proposal_tokens_per_request + 1;
        let query_positions = (0..query_rows_per_request)
            .map(|row| {
                context_after_update
                    .checked_add(row)
                    .context("DFlash2 absolute query position overflow")?
                    .try_into()
                    .context("DFlash2 absolute query position does not fit i32")
            })
            .collect::<Result<Vec<i32>>>()?;
        self.library
            .copy_h2d(
                self.body_buffers.query_positions,
                as_bytes(&query_positions),
            )
            .context("uploading DFlash2 absolute query positions")?;
        DsparkPagedKvMetadata::for_page_tables(
            &[body_kv_i32],
            query_rows_per_request,
            self.config.page_size,
            &self.request_page_tables,
            self.total_physical_pages,
        )?
        .upload(self.library, self.paged_kv_metadata)?;
        self.upload_anchors(&[anchor_token as u32])?;

        let suffix_start = Instant::now();
        self.replay_suffix_for_body_tokens(body_kv_tokens)?;
        self.stream.synchronize()?;
        let suffix_ms = suffix_start.elapsed().as_secs_f64() * 1_000.0;
        let readback_start = Instant::now();
        let token_bytes = self.read_tokens(self.head_buffers.output_tokens)?;
        let proposal_token_ids = token_bytes
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|bytes| {
                let token = i64::from_ne_bytes(
                    bytes
                        .try_into()
                        .expect("DFlash2 request token chunk has i64 width"),
                );
                usize::try_from(token)
                    .context("DFlash2 proposal token is negative or does not fit usize")
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            proposal_token_ids.len() == self.config.proposal_tokens_per_request
                && proposal_token_ids
                    .iter()
                    .all(|token| *token < super::dflash::GLM53_DFLASH2_VOCAB_SIZE),
            "DFlash2 request proposal geometry or token values are invalid"
        );
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1_000.0;
        self.body_kv_tokens = body_kv_tokens;
        Ok(Dflash2DraftStep {
            context_tokens,
            committed_rows,
            anchor_token,
            proposal_token_ids,
            update_ms,
            suffix_ms,
            readback_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1_000.0,
            packed_update_rows: 0,
        })
    }

    pub(super) fn replay_batched_suffix(
        &mut self,
        requests: &[Dflash2BatchedSuffixRequest<'_>],
    ) -> Result<Dflash2BatchedSuffixStep> {
        let total_start = Instant::now();
        anyhow::ensure!(
            self.config.active_requests > 1 && requests.len() == self.config.active_requests,
            "DFlash2 batched suffix expected {} requests, got {}",
            self.config.active_requests,
            requests.len()
        );
        let mut page_tables = Vec::with_capacity(requests.len());
        let mut flat_page_tables = Vec::with_capacity(requests.len() * self.max_pages_per_request);
        let mut body_lengths = Vec::with_capacity(requests.len());
        let query_rows_per_request = self.config.proposal_tokens_per_request + 1;
        let mut query_positions: Vec<i32> =
            Vec::with_capacity(requests.len() * query_rows_per_request);
        let mut anchors = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.iter().enumerate() {
            anyhow::ensure!(
                request.page_table.len() == self.max_pages_per_request
                    && request.page_table.iter().all(|page| {
                        *page >= 0
                            && usize::try_from(*page)
                                .is_ok_and(|page| page < self.total_physical_pages)
                    }),
                "DFlash2 batched request {request_index} has an invalid page table"
            );
            anyhow::ensure!(
                request.anchor_token < super::dflash::GLM53_DFLASH2_VOCAB_SIZE
                    && request.anchor_token != GLM53_DFLASH2_MASK_TOKEN_ID,
                "DFlash2 batched request {request_index} has invalid anchor {}",
                request.anchor_token
            );
            let body_tokens = request
                .cache_context_after_update
                .checked_add(query_rows_per_request)
                .context("DFlash2 batched body KV length overflow")?;
            anyhow::ensure!(
                body_tokens <= self.config.kv_capacity_tokens,
                "DFlash2 batched request {request_index} body KV length {body_tokens} exceeds capacity {}",
                self.config.kv_capacity_tokens
            );
            body_lengths.push(
                i32::try_from(body_tokens)
                    .context("DFlash2 batched body KV length does not fit i32")?,
            );
            for row in 0..query_rows_per_request {
                query_positions.push(
                    request
                        .absolute_context_after_update
                        .checked_add(row)
                        .context("DFlash2 batched query position overflow")?
                        .try_into()
                        .context("DFlash2 batched query position does not fit i32")?,
                );
            }
            anchors.push(request.anchor_token as u32);
            flat_page_tables.extend_from_slice(request.page_table);
            page_tables.push(request.page_table.to_vec());
        }
        self.library
            .copy_h2d(self.body_buffers.block_tables, as_bytes(&flat_page_tables))
            .context("uploading DFlash2 batched request page tables")?;
        self.library
            .copy_h2d(self.body_buffers.kv_lengths, as_bytes(&body_lengths))
            .context("uploading DFlash2 batched body KV lengths")?;
        self.library
            .copy_h2d(
                self.body_buffers.query_positions,
                as_bytes(&query_positions),
            )
            .context("uploading DFlash2 batched absolute query positions")?;
        DsparkPagedKvMetadata::for_page_tables(
            &body_lengths,
            query_rows_per_request,
            self.config.page_size,
            &page_tables,
            self.total_physical_pages,
        )?
        .upload(self.library, self.paged_kv_metadata)?;
        self.upload_anchors(&anchors)?;
        self.request_page_tables = page_tables;
        self.body_kv_tokens = body_lengths
            .iter()
            .copied()
            .max()
            .and_then(|value| usize::try_from(value).ok())
            .context("DFlash2 batched body KV lengths are empty or invalid")?;

        let suffix_start = Instant::now();
        self.replay_suffix()?;
        self.stream.synchronize()?;
        let suffix_ms = suffix_start.elapsed().as_secs_f64() * 1_000.0;
        let readback_start = Instant::now();
        let token_bytes = self.read_tokens(self.head_buffers.output_tokens)?;
        let flat_tokens = token_bytes
            .chunks_exact(std::mem::size_of::<i64>())
            .map(|bytes| {
                usize::try_from(i64::from_ne_bytes(
                    bytes
                        .try_into()
                        .expect("DFlash2 batched token chunk has i64 width"),
                ))
                .context("DFlash2 batched proposal token is negative or does not fit usize")
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(
            flat_tokens.len() == requests.len() * self.config.proposal_tokens_per_request
                && flat_tokens
                    .iter()
                    .all(|token| *token < super::dflash::GLM53_DFLASH2_VOCAB_SIZE),
            "DFlash2 batched proposal geometry or token values are invalid"
        );
        let proposal_token_ids = flat_tokens
            .chunks_exact(self.config.proposal_tokens_per_request)
            .map(<[usize]>::to_vec)
            .collect();
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1_000.0;
        Ok(Dflash2BatchedSuffixStep {
            proposal_token_ids,
            suffix_ms,
            readback_ms,
            total_ms: total_start.elapsed().as_secs_f64() * 1_000.0,
        })
    }

    fn replay_update(&self) -> Result<()> {
        self.update_graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(self.update_graph.exec_raw, self.stream.raw)
                .context("launching DFlash2 update graph")
        }
    }

    fn replay_suffix(&self) -> Result<()> {
        self.replay_suffix_for_body_tokens(self.body_kv_tokens)
    }

    fn replay_suffix_for_body_tokens(&self, body_kv_tokens: usize) -> Result<()> {
        let pages = body_kv_tokens.div_ceil(self.config.page_size).max(1);
        let graph = self
            .suffix_page_graphs
            .get(&pages)
            .unwrap_or(&self.suffix_graph);
        graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(graph.exec_raw, self.stream.raw)
                .context("launching DFlash2 suffix graph")
        }
    }

    pub(super) fn replay_cycle(&self) -> Result<()> {
        self.replay_update()?;
        self.replay_suffix()
    }

    fn upload_anchors(&mut self, anchors: &[u32]) -> Result<()> {
        upload_anchors(
            self.library,
            self.query_token_ids,
            self.head_buffers.anchor_tokens,
            anchors,
            self.config.proposal_tokens_per_request,
        )
    }

    fn read_tokens(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = checked_mul(
            config_token_rows(self.config)?,
            std::mem::size_of::<i64>(),
            "DFlash2 token readback",
        )?;
        let mut output = vec![0_u8; bytes];
        self.library
            .copy_d2h(&mut output, buffer)
            .context("reading DFlash2 candidate tokens")?;
        Ok(output)
    }
}

#[derive(Clone, Copy)]
enum ReplayKind {
    Update,
    Suffix,
}

fn benchmark_graph_pair(
    executor: &Dflash2StaticExecutor,
    kind: ReplayKind,
) -> Result<DsparkPagedAttentionTiming> {
    let mut samples = Vec::with_capacity(executor.config.repeats);
    for _ in 0..executor.config.repeats {
        let start = DsparkCudaEvent::create(executor.library)?;
        let end = DsparkCudaEvent::create(executor.library)?;
        unsafe {
            executor
                .library
                .cuda_event_record(start.raw, executor.stream.raw)?;
        }
        for _ in 0..executor.config.iterations {
            match kind {
                ReplayKind::Update => executor.replay_update()?,
                ReplayKind::Suffix => executor.replay_suffix()?,
            }
        }
        unsafe {
            executor
                .library
                .cuda_event_record(end.raw, executor.stream.raw)?;
            executor.library.cuda_event_synchronize(end.raw)?;
            samples.push(
                executor.library.cuda_event_elapsed_ms(start.raw, end.raw)? as f64
                    / executor.config.iterations as f64,
            );
        }
    }
    timing_summary(samples)
}

fn benchmark_full_cycle(
    executor: &Dflash2StaticExecutor,
) -> Result<(DsparkPagedAttentionTiming, DsparkPagedAttentionTiming)> {
    let mut gpu = Vec::with_capacity(executor.config.repeats);
    let mut host = Vec::with_capacity(executor.config.repeats);
    for _ in 0..executor.config.repeats {
        let start = DsparkCudaEvent::create(executor.library)?;
        let end = DsparkCudaEvent::create(executor.library)?;
        unsafe {
            executor
                .library
                .cuda_event_record(start.raw, executor.stream.raw)?;
        }
        let host_started = Instant::now();
        for _ in 0..executor.config.iterations {
            executor.replay_cycle()?;
        }
        unsafe {
            executor
                .library
                .cuda_event_record(end.raw, executor.stream.raw)?;
            executor.library.cuda_event_synchronize(end.raw)?;
            gpu.push(
                executor.library.cuda_event_elapsed_ms(start.raw, end.raw)? as f64
                    / executor.config.iterations as f64,
            );
        }
        host.push(
            host_started.elapsed().as_secs_f64() * 1_000.0 / executor.config.iterations as f64,
        );
    }
    Ok((timing_summary(gpu)?, timing_summary(host)?))
}

#[derive(Default)]
struct Dflash2StaticArena {
    buffers: Vec<DsparkDeviceBuffer>,
    bytes: u64,
}

impl Dflash2StaticArena {
    fn allocate(
        &mut self,
        library: &'static NativeLibrary,
        bytes: usize,
        label: &str,
    ) -> Result<GlmrtDeviceBuffer> {
        let buffer = DsparkDeviceBuffer::new(library, bytes, label)?;
        let raw = buffer.raw;
        self.bytes = self
            .bytes
            .checked_add(raw.bytes as u64)
            .context("DFlash2 mutable byte count overflow")?;
        self.buffers.push(buffer);
        Ok(raw)
    }
}

#[allow(clippy::too_many_arguments)]
fn allocate_batched_update_buffers(
    arena: &mut Dflash2StaticArena,
    library: &'static NativeLibrary,
    rows: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    static_config: Dflash2StaticBenchConfig,
    active_requests: usize,
    k_cache: GlmrtDeviceBuffer,
    v_cache: GlmrtDeviceBuffer,
    block_tables: GlmrtDeviceBuffer,
) -> Result<Dflash2UpdateBuffers> {
    let plan = dflash2_update_buffer_plan(Dflash2UpdateConfig {
        rows,
        active_requests,
        total_pages: total_physical_pages,
        page_size: static_config.page_size,
        max_pages_per_request,
        kv_storage: static_config.kv_storage,
        seed: static_config.seed,
        initialize_target_hidden: false,
        initialize_kv: false,
    })?;
    let mut update = allocate_plan(
        arena,
        library,
        plan.iter().map(|item| (item.name, item.bytes)),
        &["k_cache", "v_cache", "block_tables"],
        "DFlash2 batched update",
    )?;
    update.insert("k_cache", k_cache);
    update.insert("v_cache", v_cache);
    update.insert("block_tables", block_tables);
    Ok(Dflash2UpdateBuffers {
        target_hidden: get(&update, "target_hidden")?,
        fusion_output: get(&update, "fusion_output")?,
        fused_hidden: get(&update, "fused_hidden")?,
        projected_kv: get(&update, "projected_kv")?,
        key_output: get(&update, "key_output")?,
        value_output: get(&update, "value_output")?,
        reference_fused_hidden: get(&update, "reference_fused_hidden")?,
        reference_key_output: get(&update, "reference_key_output")?,
        reference_value_output: get(&update, "reference_value_output")?,
        eager_fused_hidden: get(&update, "eager_fused_hidden")?,
        eager_key_output: get(&update, "eager_key_output")?,
        eager_value_output: get(&update, "eager_value_output")?,
        k_cache,
        v_cache,
        row_request_ids: get(&update, "row_request_ids")?,
        row_positions: get(&update, "row_positions")?,
        row_cache_positions: get(&update, "row_cache_positions")?,
        block_tables,
    })
}

#[allow(clippy::too_many_arguments)]
fn copy_cache_pages_to_host(
    library: &'static NativeLibrary,
    cache: GlmrtDeviceBuffer,
    total_physical_pages: usize,
    layer: usize,
    page_table: &[i32],
    logical_pages: usize,
    page_bytes: usize,
    snapshot: &mut [u8],
    host_layer_base: usize,
) -> Result<()> {
    let mut logical_page = 0;
    while logical_page < logical_pages {
        let physical_page = usize::try_from(page_table[logical_page])
            .context("DFlash2 snapshot physical page is negative")?;
        let mut run_pages = 1;
        while logical_page + run_pages < logical_pages {
            let next = usize::try_from(page_table[logical_page + run_pages])
                .context("DFlash2 snapshot physical page is negative")?;
            if next != physical_page + run_pages {
                break;
            }
            run_pages += 1;
        }
        let run_bytes = checked_mul(run_pages, page_bytes, "DFlash2 snapshot run bytes")?;
        let device_offset = layer
            .checked_mul(total_physical_pages)
            .and_then(|pages| pages.checked_add(physical_page))
            .and_then(|pages| pages.checked_mul(page_bytes))
            .context("DFlash2 snapshot device offset overflow")?;
        let source = device_buffer_byte_view(
            cache,
            device_offset,
            run_bytes,
            "DFlash2 snapshot device run",
        )?;
        let host_offset = host_layer_base
            .checked_add(checked_mul(
                logical_page,
                page_bytes,
                "DFlash2 snapshot host page offset",
            )?)
            .context("DFlash2 snapshot host offset overflow")?;
        library
            .copy_d2h(&mut snapshot[host_offset..host_offset + run_bytes], source)
            .context("reading DFlash2 request cache run")?;
        logical_page += run_pages;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn copy_cache_pages_from_host(
    library: &'static NativeLibrary,
    cache: GlmrtDeviceBuffer,
    total_physical_pages: usize,
    layer: usize,
    page_table: &[i32],
    logical_pages: usize,
    page_bytes: usize,
    snapshot: &[u8],
    host_layer_base: usize,
) -> Result<()> {
    let mut logical_page = 0;
    while logical_page < logical_pages {
        let physical_page = usize::try_from(page_table[logical_page])
            .context("DFlash2 restore physical page is negative")?;
        let mut run_pages = 1;
        while logical_page + run_pages < logical_pages {
            let next = usize::try_from(page_table[logical_page + run_pages])
                .context("DFlash2 restore physical page is negative")?;
            if next != physical_page + run_pages {
                break;
            }
            run_pages += 1;
        }
        let run_bytes = checked_mul(run_pages, page_bytes, "DFlash2 restore run bytes")?;
        let device_offset = layer
            .checked_mul(total_physical_pages)
            .and_then(|pages| pages.checked_add(physical_page))
            .and_then(|pages| pages.checked_mul(page_bytes))
            .context("DFlash2 restore device offset overflow")?;
        let destination = device_buffer_byte_view(
            cache,
            device_offset,
            run_bytes,
            "DFlash2 restore device run",
        )?;
        let host_offset = host_layer_base
            .checked_add(checked_mul(
                logical_page,
                page_bytes,
                "DFlash2 restore host page offset",
            )?)
            .context("DFlash2 restore host offset overflow")?;
        library
            .copy_h2d(destination, &snapshot[host_offset..host_offset + run_bytes])
            .context("restoring DFlash2 request cache run")?;
        logical_page += run_pages;
    }
    Ok(())
}

fn allocate_plan<I>(
    arena: &mut Dflash2StaticArena,
    library: &'static NativeLibrary,
    plan: I,
    aliases: &[&str],
    label: &str,
) -> Result<BTreeMap<&'static str, GlmrtDeviceBuffer>>
where
    I: IntoIterator<Item = (&'static str, usize)>,
{
    plan.into_iter()
        .filter(|(name, _)| !aliases.contains(name))
        .map(|(name, bytes)| {
            Ok((
                name,
                arena.allocate(library, bytes, &format!("{label} {name}"))?,
            ))
        })
        .collect()
}

fn get(
    buffers: &BTreeMap<&'static str, GlmrtDeviceBuffer>,
    name: &'static str,
) -> Result<GlmrtDeviceBuffer> {
    buffers
        .get(name)
        .copied()
        .with_context(|| format!("DFlash2 static buffer plan omitted {name}"))
}

fn plan_bytes<T>(plan: &[T], name: &str) -> Result<usize>
where
    T: NamedPlan,
{
    plan.iter()
        .find(|item| item.name() == name)
        .map(NamedPlan::bytes)
        .with_context(|| format!("DFlash2 buffer plan omitted {name}"))
}

trait NamedPlan {
    fn name(&self) -> &'static str;
    fn bytes(&self) -> usize;
}

impl NamedPlan for super::dflash_body::Dflash2BodyBufferPlan {
    fn name(&self) -> &'static str {
        self.name
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
}

impl NamedPlan for super::dflash_update::Dflash2UpdateBufferPlan {
    fn name(&self) -> &'static str {
        self.name
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
}

impl NamedPlan for super::dflash_head::Dflash2HeadBufferPlan {
    fn name(&self) -> &'static str {
        self.name
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
}

fn upload_anchors(
    library: &'static NativeLibrary,
    query_token_ids: GlmrtDeviceBuffer,
    head_anchor_tokens: GlmrtDeviceBuffer,
    anchors: &[u32],
    proposal_tokens_per_request: usize,
) -> Result<()> {
    let query_tokens = anchors
        .iter()
        .flat_map(|anchor| {
            std::iter::once(*anchor).chain(std::iter::repeat_n(
                GLM53_DFLASH2_MASK_TOKEN_ID as u32,
                proposal_tokens_per_request,
            ))
        })
        .collect::<Vec<_>>();
    library
        .copy_h2d(query_token_ids, as_bytes(&query_tokens))
        .context("uploading DFlash2 query tokens")?;
    let head_anchors = anchors
        .iter()
        .map(|token| *token as i64)
        .collect::<Vec<_>>();
    library
        .copy_h2d(head_anchor_tokens, as_bytes(&head_anchors))
        .context("uploading DFlash2 head anchors")
}

fn normalized_anchor(candidate: i64) -> u32 {
    let mut token = candidate.rem_euclid(super::dflash::GLM53_DFLASH2_VOCAB_SIZE as i64) as usize;
    if token == GLM53_DFLASH2_MASK_TOKEN_ID {
        token = (token + 1) % super::dflash::GLM53_DFLASH2_VOCAB_SIZE;
    }
    token as u32
}

fn config_token_rows(config: Dflash2StaticBenchConfig) -> Result<usize> {
    checked_mul(
        config.active_requests,
        config.proposal_tokens_per_request,
        "DFlash2 proposal token rows",
    )
}

fn validate_config(config: Dflash2StaticBenchConfig) -> Result<()> {
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "DFlash2 static active requests must be 1, 2, or 4"
    );
    anyhow::ensure!(
        (1..=GLM53_DFLASH2_MAX_DRAFTS).contains(&config.proposal_tokens_per_request),
        "DFlash2 proposal tokens per request must be in 1..={GLM53_DFLASH2_MAX_DRAFTS}"
    );
    anyhow::ensure!(
        config.accepted_rows_per_request > 0
            && config.accepted_rows_per_request <= GLM53_DFLASH2_BLOCK_SIZE,
        "DFlash2 accepted rows per request must be in 1..={} ",
        GLM53_DFLASH2_BLOCK_SIZE
    );
    let update_rows = config
        .active_requests
        .checked_mul(config.accepted_rows_per_request)
        .context("DFlash2 update row count overflow")?;
    let exact_small_c1 =
        config.active_requests == 1 && config.accepted_rows_per_request <= GLM53_DFLASH2_BLOCK_SIZE;
    anyhow::ensure!(
        exact_small_c1 || update_rows.is_power_of_two(),
        "DFlash2 static update rows must be exact-small C1 or a power of two, got C={} rows={update_rows}",
        config.active_requests,
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "DFlash2 page size must be 16, 32, 64, or 128"
    );
    let body_kv = config
        .context_tokens
        .checked_add(config.accepted_rows_per_request)
        .and_then(|tokens| tokens.checked_add(config.proposal_tokens_per_request + 1))
        .context("DFlash2 body KV length overflow")?;
    anyhow::ensure!(
        body_kv <= config.kv_capacity_tokens,
        "DFlash2 body KV length exceeds capacity"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "DFlash2 benchmark iterations and repeats must be positive"
    );
    anyhow::ensure!(
        !config.capture_page_buckets
            || (config.active_requests == 1 && config.allocate_full_kv_capacity),
        "DFlash2 page-count suffix buckets require a full-capacity C1 executor"
    );
    Ok(())
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("{label} byte/count overflow"))
}

fn byte_mismatch_count(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
        + left.len().abs_diff(right.len())
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_only_graph_safe_dflash2_static_buckets() {
        let base = Dflash2StaticBenchConfig {
            active_requests: 1,
            accepted_rows_per_request: 4,
            proposal_tokens_per_request: GLM53_DFLASH2_MAX_DRAFTS,
            context_tokens: 1_024,
            kv_capacity_tokens: 128 * 1_024,
            allocate_full_kv_capacity: false,
            capture_page_buckets: false,
            page_size: 64,
            kv_storage: DsparkKvStorage::Bf16,
            warmup: 1,
            iterations: 1,
            repeats: 1,
            seed: 1,
        };
        for active_requests in [1, 2, 4] {
            for proposal_tokens_per_request in 1..=GLM53_DFLASH2_MAX_DRAFTS {
                validate_config(Dflash2StaticBenchConfig {
                    active_requests,
                    proposal_tokens_per_request,
                    ..base
                })
                .unwrap();
            }
        }
        assert!(validate_config(Dflash2StaticBenchConfig {
            active_requests: 3,
            ..base
        })
        .is_err());
        validate_config(Dflash2StaticBenchConfig {
            accepted_rows_per_request: 3,
            ..base
        })
        .unwrap();
        assert!(validate_config(Dflash2StaticBenchConfig {
            active_requests: 2,
            accepted_rows_per_request: 3,
            ..base
        })
        .is_err());
        for proposal_tokens_per_request in [0, GLM53_DFLASH2_MAX_DRAFTS + 1] {
            assert!(validate_config(Dflash2StaticBenchConfig {
                proposal_tokens_per_request,
                ..base
            })
            .is_err());
        }
        assert!(validate_config(Dflash2StaticBenchConfig {
            active_requests: 2,
            capture_page_buckets: true,
            allocate_full_kv_capacity: true,
            ..base
        })
        .is_err());
    }

    #[test]
    fn packs_decode_updates_to_the_smallest_safe_c2_or_c4_graph() {
        assert_eq!(dflash2_packed_update_rows(1, 3, 8), Some(3));
        assert_eq!(dflash2_packed_update_rows(1, 7, 8), Some(7));
        assert_eq!(dflash2_packed_update_rows(1, 9, 8), None);
        assert_eq!(dflash2_packed_update_rows(2, 2, 16), Some(2));
        assert_eq!(dflash2_packed_update_rows(2, 3, 16), Some(4));
        assert_eq!(dflash2_packed_update_rows(2, 15, 16), Some(16));
        assert_eq!(dflash2_packed_update_rows(4, 4, 32), Some(4));
        assert_eq!(dflash2_packed_update_rows(4, 17, 32), Some(32));
        assert_eq!(dflash2_packed_update_rows(4, 33, 32), None);
        assert_eq!(dflash2_packed_update_rows(1, 1, 8), Some(1));
        assert_eq!(dflash2_packed_update_rows(3, 3, 8), None);
        assert_eq!(dflash2_packed_update_rows(2, 0, 16), None);
    }

    #[test]
    fn c1_captures_exact_decode_update_lengths_without_changing_c2_c4_padding() {
        assert_eq!(
            dflash2_update_graph_buckets(1).unwrap(),
            &[2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1_024]
        );
        assert_eq!(dflash2_update_graph_buckets(2).unwrap(), &[2, 4, 8, 16]);
        assert_eq!(dflash2_update_graph_buckets(4).unwrap(), &[4, 8, 16, 32]);
        assert!(dflash2_update_graph_buckets(3).is_err());
    }
}
