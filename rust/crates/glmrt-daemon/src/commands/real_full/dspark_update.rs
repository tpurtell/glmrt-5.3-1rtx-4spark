use std::ffi::c_void;
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_ffi::{GlmrtDeviceBuffer, NativeLibrary};
use serde::Serialize;

use super::coordinator_kernels::cuda_native_library;
use super::dspark_attention::{
    timing_summary, DsparkCudaEvent, DsparkCudaGraph, DsparkCudaStream, DsparkDeviceBuffer,
    DsparkPagedAttentionTiming,
};
use super::dspark_kv::DsparkKvStorage;
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

pub(super) const DSPARK_UPDATE_LAYERS: usize = 5;
pub(super) const DSPARK_UPDATE_HIDDEN: usize = 6_144;
pub(super) const DSPARK_UPDATE_TARGET_FEATURES: usize = DSPARK_UPDATE_LAYERS * DSPARK_UPDATE_HIDDEN;
pub(super) const DSPARK_UPDATE_HEADS: usize = 64;
pub(super) const DSPARK_UPDATE_HEAD_DIM: usize = 64;
pub(super) const DSPARK_UPDATE_ATTENTION_WIDTH: usize =
    DSPARK_UPDATE_HEADS * DSPARK_UPDATE_HEAD_DIM;
const DSPARK_UPDATE_K_NORM_NAMES: [&str; DSPARK_UPDATE_LAYERS] = [
    "layer_0_k_norm",
    "layer_1_k_norm",
    "layer_2_k_norm",
    "layer_3_k_norm",
    "layer_4_k_norm",
];
const DSPARK_UPDATE_KV_NAMES: [&str; DSPARK_UPDATE_LAYERS] = [
    "layer_0_kv",
    "layer_1_kv",
    "layer_2_kv",
    "layer_3_kv",
    "layer_4_kv",
];

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkUpdateLayerResidentWeights {
    pub(super) k_norm: GlmrtDeviceBuffer,
    pub(super) kv: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkUpdateResidentWeights {
    pub(super) target_fusion: GlmrtDeviceBuffer,
    pub(super) hidden_norm: GlmrtDeviceBuffer,
    pub(super) layers: [DsparkUpdateLayerResidentWeights; DSPARK_UPDATE_LAYERS],
    pub(super) active_layers: usize,
    pub(super) referenced_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkUpdateBenchConfig {
    pub(super) layers: usize,
    pub(super) rows: usize,
    pub(super) active_requests: usize,
    pub(super) context_tokens: usize,
    pub(super) kv_capacity_tokens: usize,
    pub(super) page_size: usize,
    pub(super) kv_storage: DsparkKvStorage,
    pub(super) warmup: usize,
    pub(super) iterations: usize,
    pub(super) repeats: usize,
    pub(super) seed: i64,
    pub(super) initialize_target_hidden: bool,
    pub(super) initialize_kv: bool,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DsparkUpdateGraphReport {
    backend: &'static str,
    kv_storage: DsparkKvStorage,
    kv_element_bytes: usize,
    rows: usize,
    active_requests: usize,
    initial_request_ids: Vec<i32>,
    context_tokens: usize,
    initial_positions: Vec<i32>,
    dynamic_positions: Vec<i32>,
    page_size: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    physical_kv_bytes: u64,
    rust_owned_mutable_bytes: u64,
    referenced_weight_bytes: u64,
    duplicate_kv_weight_bytes: u64,
    graph_nodes: usize,
    graph_kernel_nodes: usize,
    graph_memcpy_nodes: usize,
    graph_memset_nodes: usize,
    reference_fused_hidden_max_abs: f64,
    reference_fused_hidden_bf16_steps_at_max_abs: u32,
    reference_fused_hidden_abs_at_max_error: f64,
    reference_fused_hidden_relative_l2: f64,
    reference_key_max_abs: f64,
    reference_key_bf16_steps_at_max_abs: u32,
    reference_key_abs_at_max_error: f64,
    reference_key_relative_l2: f64,
    reference_value_max_abs: f64,
    reference_value_bf16_steps_at_max_abs: u32,
    reference_value_abs_at_max_error: f64,
    reference_value_relative_l2: f64,
    eager_replay_exact: bool,
    dynamic_positions_change_keys: bool,
    dynamic_key_changed_bytes: usize,
    restored_replay_exact: bool,
    warmup: usize,
    iterations: usize,
    repeats: usize,
    gpu_ms_per_update_replay: DsparkPagedAttentionTiming,
    host_ms_per_update_replay: DsparkPagedAttentionTiming,
    target_hidden_fusion: bool,
    split_resident_kv_views: bool,
    paged_kv_scatter: bool,
    rust_owned_scratch: bool,
    cold_capture_python_calls: usize,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

pub(super) fn benchmark_dspark_update_graph(
    weights: DsparkUpdateResidentWeights,
    config: DsparkUpdateBenchConfig,
) -> Result<DsparkUpdateGraphReport> {
    let mut graph = DsparkUpdateGraph::capture(weights, config)?;
    graph.replay()?;
    graph.stream.synchronize()?;

    let reference_fused = graph.read_hidden(graph.buffers.reference_fused_hidden)?;
    let reference_keys = graph.read_outputs(graph.buffers.reference_key_output)?;
    let reference_values = graph.read_outputs(graph.buffers.reference_value_output)?;
    let eager_fused = graph.read_hidden(graph.buffers.eager_fused_hidden)?;
    let eager_keys = graph.read_outputs(graph.buffers.eager_key_output)?;
    let eager_values = graph.read_outputs(graph.buffers.eager_value_output)?;
    let replay_fused = graph.read_hidden(graph.buffers.fused_hidden)?;
    let replay_keys = graph.read_outputs(graph.buffers.key_output)?;
    let replay_values = graph.read_outputs(graph.buffers.value_output)?;
    let fused_difference = bf16_difference(&reference_fused, &replay_fused)?;
    let key_difference = bf16_difference(&reference_keys, &replay_keys)?;
    let value_difference = bf16_difference(&reference_values, &replay_values)?;
    anyhow::ensure!(
        fused_difference.max_abs <= 0.125 && fused_difference.relative_l2 <= 0.01,
        "dSpark update fused hidden exceeds its numerical gate: {fused_difference:?}"
    );
    anyhow::ensure!(
        key_difference.max_abs <= 0.125 && key_difference.relative_l2 <= 0.01,
        "dSpark update keys exceed their numerical gate: {key_difference:?}"
    );
    anyhow::ensure!(
        value_difference.bf16_steps_at_max_abs <= 1 && value_difference.relative_l2 <= 0.01,
        "dSpark update values exceed their numerical gate: {value_difference:?}"
    );
    let eager_replay_exact =
        eager_fused == replay_fused && eager_keys == replay_keys && eager_values == replay_values;
    anyhow::ensure!(
        eager_replay_exact,
        "dSpark update graph replay differs from its eager output"
    );

    graph.set_positions(&graph.dynamic_positions.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let dynamic_keys = graph.read_outputs(graph.buffers.key_output)?;
    let dynamic_key_changed_bytes = byte_mismatch_count(&eager_keys, &dynamic_keys);
    let dynamic_positions_change_keys = dynamic_key_changed_bytes > 0;
    anyhow::ensure!(
        dynamic_positions_change_keys,
        "dSpark update graph ignored changed row positions"
    );

    graph.set_positions(&graph.initial_positions.clone())?;
    graph.replay()?;
    graph.stream.synchronize()?;
    let restored_fused = graph.read_hidden(graph.buffers.fused_hidden)?;
    let restored_keys = graph.read_outputs(graph.buffers.key_output)?;
    let restored_values = graph.read_outputs(graph.buffers.value_output)?;
    let restored_replay_exact = restored_fused == eager_fused
        && restored_keys == eager_keys
        && restored_values == eager_values;
    anyhow::ensure!(
        restored_replay_exact,
        "dSpark update graph did not restore exact output"
    );

    for _ in 0..config.warmup {
        graph.replay()?;
    }
    graph.stream.synchronize()?;
    let mut gpu_samples = Vec::with_capacity(config.repeats);
    let mut host_samples = Vec::with_capacity(config.repeats);
    for _ in 0..config.repeats {
        let start_event = DsparkCudaEvent::create(graph.library)?;
        let end_event = DsparkCudaEvent::create(graph.library)?;
        unsafe {
            graph
                .library
                .cuda_event_record(start_event.raw, graph.stream.raw)
                .context("recording dSpark update benchmark start event")?;
        }
        let host_started = Instant::now();
        for _ in 0..config.iterations {
            graph.replay()?;
        }
        unsafe {
            graph
                .library
                .cuda_event_record(end_event.raw, graph.stream.raw)
                .context("recording dSpark update benchmark end event")?;
            graph
                .library
                .cuda_event_synchronize(end_event.raw)
                .context("waiting for dSpark update benchmark end event")?;
        }
        host_samples
            .push(host_started.elapsed().as_secs_f64() * 1_000.0 / config.iterations as f64);
        gpu_samples.push(
            unsafe {
                graph
                    .library
                    .cuda_event_elapsed_ms(start_event.raw, end_event.raw)
                    .context("measuring dSpark update CUDA graph replay")?
            } as f64
                / config.iterations as f64,
        );
    }

    Ok(DsparkUpdateGraphReport {
        backend: match config.kv_storage {
            DsparkKvStorage::Bf16 => "fixed-address-bf16-split-kv-paged-scatter",
            DsparkKvStorage::Fp8 => "fixed-address-bf16-split-kv-fp8-e4m3-paged-scatter",
        },
        kv_storage: config.kv_storage,
        kv_element_bytes: config.kv_storage.element_bytes(),
        rows: config.rows,
        active_requests: config.active_requests,
        initial_request_ids: graph.initial_request_ids.clone(),
        context_tokens: config.context_tokens,
        initial_positions: graph.initial_positions.clone(),
        dynamic_positions: graph.dynamic_positions.clone(),
        page_size: config.page_size,
        total_physical_pages: graph.total_physical_pages,
        max_pages_per_request: graph.max_pages_per_request,
        physical_kv_bytes: graph.physical_kv_bytes,
        rust_owned_mutable_bytes: graph.rust_owned_mutable_bytes,
        referenced_weight_bytes: weights.referenced_bytes,
        duplicate_kv_weight_bytes: 0,
        graph_nodes: graph.graph.node_count,
        graph_kernel_nodes: graph.graph.kernel_node_count,
        graph_memcpy_nodes: graph.graph.memcpy_node_count,
        graph_memset_nodes: graph.graph.memset_node_count,
        reference_fused_hidden_max_abs: fused_difference.max_abs,
        reference_fused_hidden_bf16_steps_at_max_abs: fused_difference.bf16_steps_at_max_abs,
        reference_fused_hidden_abs_at_max_error: fused_difference.reference_abs_at_max_error,
        reference_fused_hidden_relative_l2: fused_difference.relative_l2,
        reference_key_max_abs: key_difference.max_abs,
        reference_key_bf16_steps_at_max_abs: key_difference.bf16_steps_at_max_abs,
        reference_key_abs_at_max_error: key_difference.reference_abs_at_max_error,
        reference_key_relative_l2: key_difference.relative_l2,
        reference_value_max_abs: value_difference.max_abs,
        reference_value_bf16_steps_at_max_abs: value_difference.bf16_steps_at_max_abs,
        reference_value_abs_at_max_error: value_difference.reference_abs_at_max_error,
        reference_value_relative_l2: value_difference.relative_l2,
        eager_replay_exact,
        dynamic_positions_change_keys,
        dynamic_key_changed_bytes,
        restored_replay_exact,
        warmup: config.warmup,
        iterations: config.iterations,
        repeats: config.repeats,
        gpu_ms_per_update_replay: timing_summary(gpu_samples)?,
        host_ms_per_update_replay: timing_summary(host_samples)?,
        target_hidden_fusion: true,
        split_resident_kv_views: true,
        paged_kv_scatter: true,
        rust_owned_scratch: true,
        cold_capture_python_calls: 2,
        hot_replay_python_calls: 0,
        serving_dispatch_enabled: false,
    })
}

struct DsparkUpdateGraph {
    library: &'static NativeLibrary,
    graph: DsparkCudaGraph,
    stream: DsparkCudaStream,
    _owned_buffers: Vec<DsparkDeviceBuffer>,
    buffers: DsparkPythonUpdateBuffers,
    config: DsparkUpdateBenchConfig,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    physical_kv_bytes: u64,
    rust_owned_mutable_bytes: u64,
    initial_request_ids: Vec<i32>,
    initial_positions: Vec<i32>,
    dynamic_positions: Vec<i32>,
}

impl DsparkUpdateGraph {
    fn capture(
        weights: DsparkUpdateResidentWeights,
        config: DsparkUpdateBenchConfig,
    ) -> Result<Self> {
        validate_config(config)?;
        validate_weights(weights)?;
        anyhow::ensure!(
            weights.active_layers == config.layers,
            "dSpark update resident/config layer mismatch: {} versus {}",
            weights.active_layers,
            config.layers,
        );
        let library = cuda_native_library()?;
        let stream = DsparkCudaStream::create(library)?;
        let max_rows_per_request = config.rows.div_ceil(config.active_requests);
        let last_dynamic_position = config
            .context_tokens
            .checked_add(max_rows_per_request)
            .and_then(|position| position.checked_add(config.page_size))
            .context("dSpark update dynamic position overflow")?;
        anyhow::ensure!(
            last_dynamic_position <= config.kv_capacity_tokens,
            "dSpark update dynamic positions exceed KV capacity"
        );
        let physical_pages_per_request = last_dynamic_position.div_ceil(config.page_size);
        let total_physical_pages = checked_mul(
            config.active_requests,
            physical_pages_per_request,
            "update physical pages",
        )?;
        let max_pages_per_request = config.kv_capacity_tokens.div_ceil(config.page_size);
        let target_hidden_bytes = tensor_bytes(
            config.rows,
            DSPARK_UPDATE_TARGET_FEATURES,
            2,
            "update target hidden",
        )?;
        let hidden_bytes = tensor_bytes(config.rows, DSPARK_UPDATE_HIDDEN, 2, "update hidden")?;
        let projected_kv_bytes = tensor_bytes(
            config.rows,
            2 * DSPARK_UPDATE_ATTENTION_WIDTH,
            2,
            "update projected KV",
        )?;
        let output_bytes = tensor_bytes(
            checked_mul(config.layers, config.rows, "update output rows")?,
            DSPARK_UPDATE_ATTENTION_WIDTH,
            2,
            "update output",
        )?;
        let one_cache_bytes = checked_mul(
            checked_mul(
                checked_mul(
                    checked_mul(
                        config.layers,
                        total_physical_pages,
                        "update cache layers/pages",
                    )?,
                    DSPARK_UPDATE_HEADS,
                    "update cache heads",
                )?,
                config.page_size,
                "update cache page rows",
            )?,
            DSPARK_UPDATE_HEAD_DIM * config.kv_storage.element_bytes(),
            "update cache bytes",
        )?;
        let physical_kv_bytes = one_cache_bytes
            .checked_mul(2)
            .context("dSpark update K/V cache byte count overflow")?
            as u64;
        let row_metadata_bytes = checked_mul(config.rows, 4, "update row metadata")?;
        let block_table_entries = checked_mul(
            config.active_requests,
            max_pages_per_request,
            "update block table entries",
        )?;
        let block_table_bytes = checked_mul(
            block_table_entries,
            std::mem::size_of::<i32>(),
            "update block table",
        )?;

        let mut owned = Vec::new();
        let mut allocate = |bytes, label| -> Result<GlmrtDeviceBuffer> {
            let buffer = DsparkDeviceBuffer::new(library, bytes, label)?;
            let raw = buffer.raw;
            owned.push(buffer);
            Ok(raw)
        };
        let buffers = DsparkPythonUpdateBuffers {
            target_hidden: allocate(target_hidden_bytes, "dSpark update target hidden")?,
            fusion_output: allocate(hidden_bytes, "dSpark update fusion output")?,
            fused_hidden: allocate(hidden_bytes, "dSpark update fused hidden")?,
            projected_kv: allocate(projected_kv_bytes, "dSpark update projected KV")?,
            key_output: allocate(output_bytes, "dSpark update key output")?,
            value_output: allocate(output_bytes, "dSpark update value output")?,
            reference_fused_hidden: allocate(hidden_bytes, "dSpark update reference fused hidden")?,
            reference_key_output: allocate(output_bytes, "dSpark update reference keys")?,
            reference_value_output: allocate(output_bytes, "dSpark update reference values")?,
            eager_fused_hidden: allocate(hidden_bytes, "dSpark update eager fused hidden")?,
            eager_key_output: allocate(output_bytes, "dSpark update eager keys")?,
            eager_value_output: allocate(output_bytes, "dSpark update eager values")?,
            k_cache: allocate(one_cache_bytes, "dSpark update paged K cache")?,
            v_cache: allocate(one_cache_bytes, "dSpark update paged V cache")?,
            row_request_ids: allocate(row_metadata_bytes, "dSpark update row request IDs")?,
            row_positions: allocate(row_metadata_bytes, "dSpark update row positions")?,
            row_cache_positions: allocate(row_metadata_bytes, "dSpark update row cache positions")?,
            block_tables: allocate(block_table_bytes, "dSpark update block table")?,
        };
        let rust_owned_mutable_bytes = owned.iter().try_fold(0_u64, |bytes, buffer| {
            bytes
                .checked_add(buffer.raw.bytes as u64)
                .context("dSpark update mutable byte count overflow")
        })?;

        let (row_request_ids, initial_positions) = balanced_row_metadata(config)?;
        let dynamic_positions = initial_positions
            .iter()
            .map(|position| {
                position
                    .checked_add(config.page_size as i32)
                    .context("dSpark update dynamic position does not fit i32")
            })
            .collect::<Result<Vec<_>>>()?;
        let mut block_table = vec![0_i32; block_table_entries];
        for request in 0..config.active_requests {
            for page in 0..physical_pages_per_request {
                block_table[request * max_pages_per_request + page] =
                    (request * physical_pages_per_request + page)
                        .try_into()
                        .context("dSpark update physical page does not fit i32")?;
            }
        }
        library
            .copy_h2d(buffers.row_request_ids, as_bytes(&row_request_ids))
            .context("uploading dSpark update row request IDs")?;
        library
            .copy_h2d(buffers.row_positions, as_bytes(&initial_positions))
            .context("uploading dSpark update row positions")?;
        library
            .copy_h2d(buffers.row_cache_positions, as_bytes(&initial_positions))
            .context("uploading dSpark update row cache positions")?;
        library
            .copy_h2d(buffers.block_tables, as_bytes(&block_table))
            .context("uploading dSpark update block table")?;

        launch_python_update(
            stream.raw,
            &buffers,
            weights,
            config,
            total_physical_pages,
            max_pages_per_request,
            "prepare_dspark_context_update",
        )?;
        stream.synchronize()?;
        unsafe {
            library
                .cuda_graph_begin_capture(stream.raw)
                .context("beginning dSpark update CUDA graph capture")?;
        }
        if let Err(error) = launch_python_update(
            stream.raw,
            &buffers,
            weights,
            config,
            total_physical_pages,
            max_pages_per_request,
            "capture_dspark_context_update",
        ) {
            unsafe {
                let _ = library.cuda_graph_end_capture_retained(stream.raw);
            }
            return Err(error).context("capturing dSpark target-context update");
        }
        let capture = unsafe {
            library
                .cuda_graph_end_capture_retained(stream.raw)
                .context("ending dSpark update CUDA graph capture")?
        };
        let graph = DsparkCudaGraph::new(library, capture)?;
        graph.validate()?;

        Ok(Self {
            library,
            graph,
            stream,
            _owned_buffers: owned,
            buffers,
            config,
            total_physical_pages,
            max_pages_per_request,
            physical_kv_bytes,
            rust_owned_mutable_bytes,
            initial_request_ids: row_request_ids,
            initial_positions,
            dynamic_positions,
        })
    }

    fn set_positions(&mut self, positions: &[i32]) -> Result<()> {
        anyhow::ensure!(
            positions.len() == self.config.rows
                && positions.iter().all(|position| {
                    *position >= 0 && (*position as usize) < self.config.kv_capacity_tokens
                }),
            "dSpark update positions are invalid: {positions:?}"
        );
        self.library
            .copy_h2d(self.buffers.row_positions, as_bytes(positions))
            .context("uploading dynamic dSpark update positions")
    }

    fn replay(&self) -> Result<()> {
        self.graph.validate()?;
        unsafe {
            self.library
                .cuda_graph_launch(self.graph.exec_raw, self.stream.raw)
                .context("launching dSpark update CUDA graph")
        }
    }

    fn read_hidden(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = tensor_bytes(
            self.config.rows,
            DSPARK_UPDATE_HIDDEN,
            2,
            "update hidden readback",
        )?;
        self.read_buffer(buffer, bytes, "reading dSpark update hidden")
    }

    fn read_outputs(&self, buffer: GlmrtDeviceBuffer) -> Result<Vec<u8>> {
        let bytes = tensor_bytes(
            checked_mul(
                self.config.layers,
                self.config.rows,
                "update output readback rows",
            )?,
            DSPARK_UPDATE_ATTENTION_WIDTH,
            2,
            "update output readback",
        )?;
        self.read_buffer(buffer, bytes, "reading dSpark update output")
    }

    fn read_buffer(
        &self,
        buffer: GlmrtDeviceBuffer,
        bytes: usize,
        label: &'static str,
    ) -> Result<Vec<u8>> {
        let mut output = vec![0_u8; bytes];
        self.library.copy_d2h(&mut output, buffer).context(label)?;
        Ok(output)
    }
}

#[derive(Clone, Copy)]
pub(super) struct DsparkPythonUpdateBuffers {
    pub(super) target_hidden: GlmrtDeviceBuffer,
    pub(super) fusion_output: GlmrtDeviceBuffer,
    pub(super) fused_hidden: GlmrtDeviceBuffer,
    pub(super) projected_kv: GlmrtDeviceBuffer,
    pub(super) key_output: GlmrtDeviceBuffer,
    pub(super) value_output: GlmrtDeviceBuffer,
    pub(super) reference_fused_hidden: GlmrtDeviceBuffer,
    pub(super) reference_key_output: GlmrtDeviceBuffer,
    pub(super) reference_value_output: GlmrtDeviceBuffer,
    pub(super) eager_fused_hidden: GlmrtDeviceBuffer,
    pub(super) eager_key_output: GlmrtDeviceBuffer,
    pub(super) eager_value_output: GlmrtDeviceBuffer,
    pub(super) k_cache: GlmrtDeviceBuffer,
    pub(super) v_cache: GlmrtDeviceBuffer,
    pub(super) row_request_ids: GlmrtDeviceBuffer,
    pub(super) row_positions: GlmrtDeviceBuffer,
    pub(super) row_cache_positions: GlmrtDeviceBuffer,
    pub(super) block_tables: GlmrtDeviceBuffer,
}

pub(super) fn launch_python_update(
    cuda_stream: *mut c_void,
    buffers: &DsparkPythonUpdateBuffers,
    weights: DsparkUpdateResidentWeights,
    config: DsparkUpdateBenchConfig,
    total_pages: usize,
    max_pages_per_request: usize,
    function: &str,
) -> Result<()> {
    let mut device_buffers = vec![
        python_buffer("target_hidden", buffers.target_hidden),
        python_buffer("fusion_output", buffers.fusion_output),
        python_buffer("fused_hidden", buffers.fused_hidden),
        python_buffer("projected_kv", buffers.projected_kv),
        python_buffer("key_output", buffers.key_output),
        python_buffer("value_output", buffers.value_output),
        python_buffer("reference_fused_hidden", buffers.reference_fused_hidden),
        python_buffer("reference_key_output", buffers.reference_key_output),
        python_buffer("reference_value_output", buffers.reference_value_output),
        python_buffer("eager_fused_hidden", buffers.eager_fused_hidden),
        python_buffer("eager_key_output", buffers.eager_key_output),
        python_buffer("eager_value_output", buffers.eager_value_output),
        python_buffer("k_cache", buffers.k_cache),
        python_buffer("v_cache", buffers.v_cache),
        python_buffer("row_request_ids", buffers.row_request_ids),
        python_buffer("row_positions", buffers.row_positions),
        python_buffer("row_cache_positions", buffers.row_cache_positions),
        python_buffer("block_tables", buffers.block_tables),
        python_buffer("target_fusion", weights.target_fusion),
        python_buffer("hidden_norm", weights.hidden_norm),
    ];
    for ((k_norm_name, kv_name), weights) in DSPARK_UPDATE_K_NORM_NAMES
        .iter()
        .copied()
        .zip(DSPARK_UPDATE_KV_NAMES.iter().copied())
        .zip(weights.layers.iter())
        .take(config.layers)
    {
        device_buffers.push(python_buffer(k_norm_name, weights.k_norm));
        device_buffers.push(python_buffer(kv_name, weights.kv));
    }
    let kwargs = [
        ("rows", PythonKernelArg::Usize(config.rows)),
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        ("layers", PythonKernelArg::Usize(config.layers)),
        ("hidden_size", PythonKernelArg::Usize(DSPARK_UPDATE_HIDDEN)),
        (
            "target_features",
            PythonKernelArg::Usize(DSPARK_UPDATE_TARGET_FEATURES),
        ),
        ("heads", PythonKernelArg::Usize(DSPARK_UPDATE_HEADS)),
        ("head_dim", PythonKernelArg::Usize(DSPARK_UPDATE_HEAD_DIM)),
        ("total_pages", PythonKernelArg::Usize(total_pages)),
        ("page_size", PythonKernelArg::Usize(config.page_size)),
        (
            "max_pages_per_request",
            PythonKernelArg::Usize(max_pages_per_request),
        ),
        ("seed", PythonKernelArg::I64(config.seed)),
        (
            "initialize_target_hidden",
            PythonKernelArg::Bool(config.initialize_target_hidden),
        ),
        ("initialize_kv", PythonKernelArg::Bool(config.initialize_kv)),
        (
            "cache_dtype",
            PythonKernelArg::Str(config.kv_storage.label()),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "dspark_update_capture",
        function,
        cuda_stream,
        buffers: &device_buffers,
        kwargs: &kwargs,
    })
}

fn python_buffer(name: &str, buffer: GlmrtDeviceBuffer) -> PythonDeviceBufferArg<'_> {
    PythonDeviceBufferArg {
        name,
        ptr: buffer.ptr,
        bytes: buffer.bytes,
        device_id: buffer.device_id,
        flags: buffer.flags,
    }
}

fn validate_config(config: DsparkUpdateBenchConfig) -> Result<()> {
    anyhow::ensure!(
        (1..=DSPARK_UPDATE_LAYERS).contains(&config.layers),
        "dSpark update layer count must be between 1 and {DSPARK_UPDATE_LAYERS}"
    );
    anyhow::ensure!(
        config.rows.is_power_of_two() && config.rows <= 1_024,
        "dSpark update rows must be a power of two no larger than 1024"
    );
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "dSpark update active requests must be 1, 2, or 4"
    );
    anyhow::ensure!(
        config.context_tokens > 0,
        "dSpark update context must be positive"
    );
    let required = config
        .context_tokens
        .checked_add(config.rows.div_ceil(config.active_requests))
        .and_then(|tokens| tokens.checked_add(config.page_size))
        .context("dSpark update required KV capacity overflow")?;
    anyhow::ensure!(
        required <= config.kv_capacity_tokens,
        "dSpark update KV capacity does not include its dynamic probe"
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "dSpark update page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        config.iterations > 0 && config.repeats > 0,
        "dSpark update benchmark iterations and repeats must be positive"
    );
    Ok(())
}

fn balanced_row_metadata(config: DsparkUpdateBenchConfig) -> Result<(Vec<i32>, Vec<i32>)> {
    let rows_per_request = config.rows / config.active_requests;
    let extra_rows = config.rows % config.active_requests;
    let mut request_ids = Vec::with_capacity(config.rows);
    let mut positions = Vec::with_capacity(config.rows);
    for request in 0..config.active_requests {
        let request_rows = rows_per_request + usize::from(request < extra_rows);
        for local_row in 0..request_rows {
            request_ids.push(
                request
                    .try_into()
                    .context("dSpark update request ID does not fit i32")?,
            );
            positions.push(
                config
                    .context_tokens
                    .checked_add(local_row)
                    .context("dSpark update initial position overflow")?
                    .try_into()
                    .context("dSpark update initial position does not fit i32")?,
            );
        }
    }
    anyhow::ensure!(
        request_ids.len() == config.rows && positions.len() == config.rows,
        "dSpark update row metadata has the wrong length"
    );
    Ok((request_ids, positions))
}

fn validate_weights(weights: DsparkUpdateResidentWeights) -> Result<()> {
    anyhow::ensure!(
        (1..=DSPARK_UPDATE_LAYERS).contains(&weights.active_layers),
        "dSpark update resident layer count is invalid"
    );
    validate_buffer(
        "target fusion",
        weights.target_fusion,
        DSPARK_UPDATE_HIDDEN * DSPARK_UPDATE_TARGET_FEATURES * 2,
    )?;
    validate_buffer("hidden norm", weights.hidden_norm, DSPARK_UPDATE_HIDDEN * 2)?;
    for (layer, weights) in weights
        .layers
        .iter()
        .take(weights.active_layers)
        .enumerate()
    {
        validate_buffer(
            &format!("layer {layer} K norm"),
            weights.k_norm,
            DSPARK_UPDATE_HEAD_DIM * 2,
        )?;
        validate_buffer(
            &format!("layer {layer} K/V projection"),
            weights.kv,
            2 * DSPARK_UPDATE_ATTENTION_WIDTH * DSPARK_UPDATE_HIDDEN * 2,
        )?;
    }
    Ok(())
}

fn validate_buffer(label: &str, buffer: GlmrtDeviceBuffer, expected_bytes: usize) -> Result<()> {
    anyhow::ensure!(
        !buffer.ptr.is_null(),
        "dSpark update {label} buffer is null"
    );
    anyhow::ensure!(
        buffer.bytes == expected_bytes,
        "dSpark update {label} has {} bytes, expected {expected_bytes}",
        buffer.bytes
    );
    Ok(())
}

fn tensor_bytes(rows: usize, width: usize, element_bytes: usize, label: &str) -> Result<usize> {
    checked_mul(checked_mul(rows, width, label)?, element_bytes, label)
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
    left.checked_mul(right)
        .with_context(|| format!("dSpark {label} overflow"))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Bf16Difference {
    pub(super) max_abs: f64,
    pub(super) bf16_steps_at_max_abs: u32,
    pub(super) reference_abs_at_max_error: f64,
    pub(super) relative_l2: f64,
}

pub(super) fn bf16_difference(reference: &[u8], candidate: &[u8]) -> Result<Bf16Difference> {
    anyhow::ensure!(
        reference.len() == candidate.len() && reference.len() % 2 == 0,
        "dSpark update BF16 comparison byte lengths are invalid"
    );
    let mut max_abs = 0.0_f64;
    let mut bf16_steps_at_max_abs = 0_u32;
    let mut reference_abs_at_max_error = 0.0_f64;
    let mut squared_delta = 0.0_f64;
    let mut squared_reference = 0.0_f64;
    for (reference, candidate) in reference.chunks_exact(2).zip(candidate.chunks_exact(2)) {
        let reference_bits = u16::from_le_bytes([reference[0], reference[1]]);
        let candidate_bits = u16::from_le_bytes([candidate[0], candidate[1]]);
        let reference = f32::from_bits((reference_bits as u32) << 16) as f64;
        let candidate = f32::from_bits((candidate_bits as u32) << 16) as f64;
        anyhow::ensure!(
            reference.is_finite() && candidate.is_finite(),
            "dSpark update output contains a non-finite BF16 value"
        );
        let delta = candidate - reference;
        if delta.abs() > max_abs {
            max_abs = delta.abs();
            bf16_steps_at_max_abs = if reference == candidate {
                0
            } else {
                bf16_ordered(reference_bits).abs_diff(bf16_ordered(candidate_bits))
            };
            reference_abs_at_max_error = reference.abs();
        }
        squared_delta += delta * delta;
        squared_reference += reference * reference;
    }
    Ok(Bf16Difference {
        max_abs,
        bf16_steps_at_max_abs,
        reference_abs_at_max_error,
        relative_l2: (squared_delta / squared_reference.max(f64::MIN_POSITIVE)).sqrt(),
    })
}

fn bf16_ordered(bits: u16) -> u32 {
    if bits & 0x8000 == 0 {
        (bits as u32) | 0x8000
    } else {
        (!bits) as u32 & 0xffff
    }
}

fn byte_mismatch_count(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(left, right)| left != right)
        .count()
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

#[cfg(test)]
mod tests {
    use super::{balanced_row_metadata, bf16_difference, validate_config, DsparkUpdateBenchConfig};
    use crate::commands::real_full::dspark_kv::DsparkKvStorage;

    fn config() -> DsparkUpdateBenchConfig {
        DsparkUpdateBenchConfig {
            layers: 5,
            rows: 16,
            active_requests: 1,
            context_tokens: 1_024,
            kv_capacity_tokens: 256 * 1_024,
            page_size: 64,
            kv_storage: DsparkKvStorage::Bf16,
            warmup: 2,
            iterations: 10,
            repeats: 3,
            seed: 17,
            initialize_target_hidden: true,
            initialize_kv: true,
        }
    }

    #[test]
    fn accepts_production_update_buckets() {
        for active_requests in [1, 2, 4] {
            for rows in [1, 2, 4, 8, 16, 64, 128, 256, 512, 1_024] {
                validate_config(DsparkUpdateBenchConfig {
                    rows,
                    active_requests,
                    ..config()
                })
                .unwrap();
            }
        }
    }

    #[test]
    fn rejects_update_capacity_and_shape_mismatches() {
        let mut invalid = config();
        invalid.rows = 3;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.active_requests = 3;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.kv_capacity_tokens = invalid.context_tokens + invalid.rows;
        assert!(validate_config(invalid).is_err());
        invalid = config();
        invalid.page_size = 63;
        assert!(validate_config(invalid).is_err());
    }

    #[test]
    fn packs_rows_without_padding_across_requests() {
        let (request_ids, positions) = balanced_row_metadata(DsparkUpdateBenchConfig {
            active_requests: 4,
            ..config()
        })
        .unwrap();
        assert_eq!(
            request_ids,
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
        );
        assert_eq!(
            positions,
            [
                1_024, 1_025, 1_026, 1_027, 1_024, 1_025, 1_026, 1_027, 1_024, 1_025, 1_026, 1_027,
                1_024, 1_025, 1_026, 1_027,
            ]
        );
    }

    #[test]
    fn reports_bf16_spacing_at_the_largest_error() {
        let reference = [0x00_u8, 0x42];
        let candidate = [0x01_u8, 0x42];
        let difference = bf16_difference(&reference, &candidate).unwrap();
        assert_eq!(difference.max_abs, 0.25);
        assert_eq!(difference.bf16_steps_at_max_abs, 1);
        assert_eq!(difference.reference_abs_at_max_error, 32.0);
    }
}
