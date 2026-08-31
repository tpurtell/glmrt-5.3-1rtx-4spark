use super::*;
use anyhow::{Context, Result};
use glmrt_core::{
    CoordinatorGraphInstancePlan, CoordinatorGraphKey, CoordinatorGraphShape, LayerId,
    LayerWaveMode, COORDINATOR_GRAPH_INSTANCE_COUNT, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_NUM_HIDDEN_LAYERS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K,
};
use glmrt_ffi::{
    GlmrtCudaGraphCaptureInfo, GlmrtDeviceBuffer, GlmrtHostBuffer, NativeLibrary,
    GLMRT_CUDA_ROUTER_TOPK_MAX_K, GLMRT_CUDA_SAMPLE_TOPK_MAX_K,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::{Mutex, OnceLock};

const B12X_COORDINATOR_W4A16_DECODE_ENV: &str = "GLMRT_B12X_COORDINATOR_W4A16_DECODE";
const B12X_COORDINATOR_W4A16_Q_B_DECODE_ENV: &str = "GLMRT_B12X_COORDINATOR_W4A16_Q_B";
const B12X_COORDINATOR_W4A16_O_PROJ_DECODE_ENV: &str = "GLMRT_B12X_COORDINATOR_W4A16_O_PROJ";
const COORDINATOR_W8A16_Q_A_DECODE_ENV: &str = "GLMRT_COORDINATOR_W8A16_Q_A";
const COORDINATOR_W8A16_Q_B_DECODE_ENV: &str = "GLMRT_COORDINATOR_W8A16_Q_B";
const COORDINATOR_W8A16_O_PROJ_DECODE_ENV: &str = "GLMRT_COORDINATOR_W8A16_O_PROJ";
const COORDINATOR_W8A16_PACKED_O_ENV: &str = "GLMRT_COORDINATOR_W8A16_PACKED_O";
const B12X_COORDINATOR_W4A16_SCRATCH_ELEMENTS: usize = 2_097_152;
const B12X_COORDINATOR_W4A16_LOCK_ELEMENTS: usize = 1_024;
const B12X_COORDINATOR_W4A16_ROUTE_SLOTS: usize = 8;
const B12X_COORDINATOR_W4A16_METADATA_BYTES: usize =
    80 + B12X_COORDINATOR_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>();
const COORDINATOR_W8A16_PACKED_O_ROUTE_SLOTS: usize = 16;
const COORDINATOR_W8A16_PACKED_O_SCRATCH_ELEMENTS: usize = 4 * 256 * 16 * 256;
const COORDINATOR_W8A16_PACKED_O_METADATA_BYTES: usize =
    176 + B12X_COORDINATOR_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>();

thread_local! {
    static PRELOADED_RESIDENT_WEIGHT_CACHE: RefCell<HashMap<String, (usize, GlmrtDeviceBuffer)>> =
        RefCell::new(HashMap::new());
    static PRELOADED_COORDINATOR_W4A16_CACHE: RefCell<
        HashMap<String, (usize, usize, CoordinatorW4a16ProjectionBuffers)>,
    > = RefCell::new(HashMap::new());
    static PRELOADED_COORDINATOR_W8A16_CACHE: RefCell<
        HashMap<String, (usize, usize, CoordinatorW8a16ProjectionBuffers)>,
    > = RefCell::new(HashMap::new());
    static B12X_COORDINATOR_W4A16_INITIALIZED_METADATA: RefCell<HashSet<usize>> =
        RefCell::new(HashSet::new());
}

#[derive(Clone, Copy)]
pub(in crate::commands::real_full) struct CoordinatorW4a16ProjectionBuffers {
    pub(in crate::commands::real_full) weight: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) scale: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) global_scale: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy)]
pub(in crate::commands::real_full) struct CoordinatorW8a16ProjectionBuffers {
    pub(in crate::commands::real_full) weight: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) scales: GlmrtDeviceBuffer,
    pub(in crate::commands::real_full) packed_layout: bool,
}

fn parse_coordinator_projection_flag(value: Option<&str>, precision_name: &str) -> bool {
    value
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on") || value == precision_name
        })
        .unwrap_or(false)
}

fn parse_coordinator_w4a16_flag(value: Option<&str>) -> bool {
    parse_coordinator_projection_flag(value, "w4a16")
}

fn parse_coordinator_w8a16_flag(value: Option<&str>) -> bool {
    parse_coordinator_projection_flag(value, "w8a16")
}

fn coordinator_w4a16_projection_flag_enabled(
    legacy_value: Option<&str>,
    projection_value: Option<&str>,
) -> bool {
    parse_coordinator_w4a16_flag(legacy_value) || parse_coordinator_w4a16_flag(projection_value)
}

fn coordinator_w4a16_projection_env_enabled(projection_name: &str) -> bool {
    let legacy_value = std::env::var(B12X_COORDINATOR_W4A16_DECODE_ENV).ok();
    let projection_value = std::env::var(projection_name).ok();
    coordinator_w4a16_projection_flag_enabled(legacy_value.as_deref(), projection_value.as_deref())
}

pub(in crate::commands::real_full) fn coordinator_w4a16_q_b_decode_enabled() -> bool {
    coordinator_w4a16_projection_env_enabled(B12X_COORDINATOR_W4A16_Q_B_DECODE_ENV)
}

pub(in crate::commands::real_full) fn coordinator_w4a16_o_proj_decode_enabled() -> bool {
    coordinator_w4a16_projection_env_enabled(B12X_COORDINATOR_W4A16_O_PROJ_DECODE_ENV)
}

pub(in crate::commands::real_full) fn coordinator_w8a16_o_proj_decode_enabled() -> bool {
    std::env::var(COORDINATOR_W8A16_O_PROJ_DECODE_ENV)
        .ok()
        .as_deref()
        .is_some_and(|value| parse_coordinator_w8a16_flag(Some(value)))
}

pub(in crate::commands::real_full) fn coordinator_w8a16_q_a_decode_enabled() -> bool {
    std::env::var(COORDINATOR_W8A16_Q_A_DECODE_ENV)
        .ok()
        .as_deref()
        .is_some_and(|value| parse_coordinator_w8a16_flag(Some(value)))
}

pub(in crate::commands::real_full) fn coordinator_w8a16_q_b_decode_enabled() -> bool {
    std::env::var(COORDINATOR_W8A16_Q_B_DECODE_ENV)
        .ok()
        .as_deref()
        .is_some_and(|value| parse_coordinator_w8a16_flag(Some(value)))
}

pub(in crate::commands::real_full) fn coordinator_w8a16_packed_o_enabled() -> bool {
    std::env::var(COORDINATOR_W8A16_PACKED_O_ENV)
        .ok()
        .as_deref()
        .is_some_and(|value| parse_coordinator_w8a16_flag(Some(value)))
}

fn coordinator_w4a16_resident_names(weight_name: &str) -> (String, String, String) {
    (
        format!("{weight_name}#b12x-w4a16-weight"),
        format!("{weight_name}#b12x-w4a16-scale"),
        format!("{weight_name}#b12x-w4a16-global-scale"),
    )
}

fn coordinator_w8a16_resident_names(weight_name: &str, packed_layout: bool) -> (String, String) {
    let layout = if packed_layout { "packed" } else { "row-major" };
    (
        format!("{weight_name}#w8a16-group256-{layout}-weight"),
        format!("{weight_name}#w8a16-group256-{layout}-scales"),
    )
}

fn coordinator_w8a16_projection_uses_packed_layout(weight_name: &str) -> bool {
    coordinator_w8a16_packed_o_enabled() && weight_name.ends_with(".self_attn.o_proj.weight")
}

impl CoordinatorCudaResidentWeights {
    fn preload_coordinator_w4a16_projection(
        &mut self,
        library: &'static NativeLibrary,
        weight_name: &str,
        size_k: usize,
        size_n: usize,
    ) -> Result<CoordinatorW4a16ProjectionBuffers> {
        let source_bytes = size_n
            .checked_mul(size_k)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("coordinator W4A16 source weight bytes overflow")?;
        let packed_weight_bytes = size_n
            .checked_mul(size_k / 2)
            .context("coordinator W4A16 packed weight bytes overflow")?;
        let packed_scale_bytes = size_n
            .checked_mul(size_k / 16)
            .context("coordinator W4A16 packed scale bytes overflow")?;
        let payload_bytes = packed_weight_bytes
            .checked_add(packed_scale_bytes)
            .context("coordinator W4A16 quantization payload bytes overflow")?;
        let source = self.preloaded_resident_weight_buffer(weight_name, source_bytes)?;
        let (packed_name, scale_name, global_name) = coordinator_w4a16_resident_names(weight_name);

        let packed_weight = {
            let key = resident_weight_registry_key(&packed_name, packed_weight_bytes);
            self.resident_weights
                .entry(key)
                .or_default()
                .ensure_capacity(
                    library,
                    &packed_name,
                    packed_weight_bytes,
                    "coordinator B12X W4A16 packed weight",
                )?
        };
        let packed_scale = {
            let key = resident_weight_registry_key(&scale_name, packed_scale_bytes);
            self.resident_weights
                .entry(key)
                .or_default()
                .ensure_capacity(
                    library,
                    &scale_name,
                    packed_scale_bytes,
                    "coordinator B12X W4A16 packed scale",
                )?
        };
        let global_scale = {
            let key = resident_weight_registry_key(&global_name, std::mem::size_of::<f32>());
            self.resident_weights
                .entry(key)
                .or_default()
                .ensure_capacity(
                    library,
                    &global_name,
                    std::mem::size_of::<f32>(),
                    "coordinator B12X W4A16 global scale",
                )?
        };
        let already_quantized = [
            (&packed_name, packed_weight_bytes),
            (&scale_name, packed_scale_bytes),
            (&global_name, std::mem::size_of::<f32>()),
        ]
        .into_iter()
        .all(|(name, bytes)| self.resident_weight_is_preloaded(name, bytes));
        if already_quantized {
            return Ok(CoordinatorW4a16ProjectionBuffers {
                weight: packed_weight,
                scale: packed_scale,
                global_scale,
            });
        }

        self.w4a16_quant_scratch.ensure_capacity(
            library,
            payload_bytes,
            "coordinator B12X W4A16 quantization scratch",
        )?;
        unsafe {
            library
                .cuda_b12x_coordinator_w4a16_quantize_pack_weight_async(
                    source,
                    self.w4a16_quant_scratch.buffer,
                    packed_weight,
                    packed_scale,
                    global_scale,
                    size_k,
                    size_n,
                    std::ptr::null_mut(),
                )
                .with_context(|| format!("quantizing coordinator W4A16 weight {weight_name}"))?;
            library
                .cuda_stream_synchronize(std::ptr::null_mut())
                .with_context(|| format!("synchronizing coordinator W4A16 weight {weight_name}"))?;
        }
        for (name, bytes) in [
            (&packed_name, packed_weight_bytes),
            (&scale_name, packed_scale_bytes),
            (&global_name, std::mem::size_of::<f32>()),
        ] {
            let key = resident_weight_registry_key(name, bytes);
            let resident = self
                .resident_weights
                .get_mut(&key)
                .with_context(|| format!("coordinator W4A16 resident buffer {name} disappeared"))?;
            resident.uploaded = true;
            resident.upload_count += 1;
        }
        Ok(CoordinatorW4a16ProjectionBuffers {
            weight: packed_weight,
            scale: packed_scale,
            global_scale,
        })
    }

    fn preloaded_coordinator_w4a16_projection(
        &self,
        weight_name: &str,
        size_k: usize,
        size_n: usize,
    ) -> Result<CoordinatorW4a16ProjectionBuffers> {
        let packed_weight_bytes = size_n
            .checked_mul(size_k / 2)
            .context("coordinator W4A16 packed weight bytes overflow")?;
        let packed_scale_bytes = size_n
            .checked_mul(size_k / 16)
            .context("coordinator W4A16 packed scale bytes overflow")?;
        let (packed_name, scale_name, global_name) = coordinator_w4a16_resident_names(weight_name);
        Ok(CoordinatorW4a16ProjectionBuffers {
            weight: self.preloaded_resident_weight_buffer(&packed_name, packed_weight_bytes)?,
            scale: self.preloaded_resident_weight_buffer(&scale_name, packed_scale_bytes)?,
            global_scale: self
                .preloaded_resident_weight_buffer(&global_name, std::mem::size_of::<f32>())?,
        })
    }
}

impl CoordinatorCudaResidentWeights {
    fn preload_coordinator_w8a16_projection(
        &mut self,
        library: &'static NativeLibrary,
        weight_name: &str,
        size_k: usize,
        size_n: usize,
    ) -> Result<CoordinatorW8a16ProjectionBuffers> {
        anyhow::ensure!(
            size_k > 0 && size_k % 256 == 0 && size_n > 0,
            "coordinator W8A16 projection {weight_name} requires positive N and K divisible by 256"
        );
        let source_bytes = size_n
            .checked_mul(size_k)
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("coordinator W8A16 source weight bytes overflow")?;
        let weight_bytes = size_n
            .checked_mul(size_k)
            .context("coordinator W8A16 packed weight bytes overflow")?;
        let scale_bytes = size_n
            .checked_mul(size_k / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("coordinator W8A16 scale bytes overflow")?;
        let packed_layout = coordinator_w8a16_projection_uses_packed_layout(weight_name);
        let (packed_name, scale_name) =
            coordinator_w8a16_resident_names(weight_name, packed_layout);
        let weight = {
            let key = resident_weight_registry_key(&packed_name, weight_bytes);
            self.resident_weights
                .entry(key)
                .or_default()
                .ensure_capacity(
                    library,
                    &packed_name,
                    weight_bytes,
                    "coordinator W8A16 group-256 weight",
                )?
        };
        let scales = {
            let key = resident_weight_registry_key(&scale_name, scale_bytes);
            self.resident_weights
                .entry(key)
                .or_default()
                .ensure_capacity(
                    library,
                    &scale_name,
                    scale_bytes,
                    "coordinator W8A16 group-256 scales",
                )?
        };
        let already_quantized = [(&packed_name, weight_bytes), (&scale_name, scale_bytes)]
            .into_iter()
            .all(|(name, bytes)| self.resident_weight_is_preloaded(name, bytes));
        if !already_quantized {
            let source = self.preloaded_resident_weight_buffer(weight_name, source_bytes)?;
            unsafe {
                let quantize_result = if packed_layout {
                    library.cuda_quantize_bf16_w8a16_group256_packed_async(
                        source,
                        weight,
                        scales,
                        size_k,
                        size_n,
                        std::ptr::null_mut(),
                    )
                } else {
                    library.cuda_quantize_bf16_w8a16_group256_async(
                        source,
                        weight,
                        scales,
                        size_k,
                        size_n,
                        false,
                        std::ptr::null_mut(),
                    )
                };
                quantize_result.with_context(|| {
                    format!("quantizing coordinator W8A16 weight {weight_name}")
                })?;
                library
                    .cuda_stream_synchronize(std::ptr::null_mut())
                    .with_context(|| {
                        format!("synchronizing coordinator W8A16 weight {weight_name}")
                    })?;
            }
            for (name, bytes) in [(&packed_name, weight_bytes), (&scale_name, scale_bytes)] {
                let key = resident_weight_registry_key(name, bytes);
                let resident = self.resident_weights.get_mut(&key).with_context(|| {
                    format!("coordinator W8A16 resident buffer {name} disappeared")
                })?;
                resident.uploaded = true;
                resident.upload_count += 1;
            }
        }
        Ok(CoordinatorW8a16ProjectionBuffers {
            weight,
            scales,
            packed_layout,
        })
    }

    fn preloaded_coordinator_w8a16_projection(
        &self,
        weight_name: &str,
        size_k: usize,
        size_n: usize,
    ) -> Result<CoordinatorW8a16ProjectionBuffers> {
        let weight_bytes = size_n
            .checked_mul(size_k)
            .context("coordinator W8A16 packed weight bytes overflow")?;
        let scale_bytes = size_n
            .checked_mul(size_k / 256)
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .context("coordinator W8A16 scale bytes overflow")?;
        let packed_layout = coordinator_w8a16_projection_uses_packed_layout(weight_name);
        let (packed_name, scale_name) =
            coordinator_w8a16_resident_names(weight_name, packed_layout);
        Ok(CoordinatorW8a16ProjectionBuffers {
            weight: self.preloaded_resident_weight_buffer(&packed_name, weight_bytes)?,
            scales: self.preloaded_resident_weight_buffer(&scale_name, scale_bytes)?,
            packed_layout,
        })
    }
}

pub(in crate::commands::real_full) fn preload_coordinator_w4a16_projection(
    weight_name: &str,
    size_k: usize,
    size_n: usize,
) -> Result<()> {
    let library = cuda_native_library()?;
    anyhow::ensure!(
        library.cuda_b12x_coordinator_aot_available()?,
        "coordinator W4A16 projection requires coordinator B12X AOT kernels"
    );
    library.cuda_b12x_coordinator_aot_init()?;
    lock_coordinator_cuda_resident_weights()?
        .preload_coordinator_w4a16_projection(library, weight_name, size_k, size_n)
        .map(|_| ())
}

pub(in crate::commands::real_full) fn preloaded_coordinator_w4a16_projection(
    weight_name: &str,
    size_k: usize,
    size_n: usize,
) -> Result<CoordinatorW4a16ProjectionBuffers> {
    if let Some((cached_k, cached_n, buffers)) =
        PRELOADED_COORDINATOR_W4A16_CACHE.with(|cache| cache.borrow().get(weight_name).copied())
    {
        anyhow::ensure!(
            cached_k == size_k && cached_n == size_n,
            "cached coordinator W4A16 projection {weight_name} shape {cached_k}x{cached_n} does not match requested {size_k}x{size_n}"
        );
        return Ok(buffers);
    }
    let buffers = lock_coordinator_cuda_resident_weights()?
        .preloaded_coordinator_w4a16_projection(weight_name, size_k, size_n)?;
    PRELOADED_COORDINATOR_W4A16_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(weight_name.to_owned(), (size_k, size_n, buffers));
    });
    Ok(buffers)
}

pub(in crate::commands::real_full) fn preload_coordinator_w8a16_projection(
    weight_name: &str,
    size_k: usize,
    size_n: usize,
) -> Result<()> {
    let library = cuda_native_library()?;
    unsafe {
        if coordinator_w8a16_projection_uses_packed_layout(weight_name) {
            library
                .cuda_w8a16_packed_o_aot_init()
                .context("preloading coordinator packed W8A16 O kernels")?;
        } else {
            library
                .cuda_preload_w8a16_group256_aot(size_k, size_n)
                .with_context(|| {
                    format!(
                        "preloading coordinator W8A16 AOT kernels for {weight_name} ({size_n}x{size_k})"
                    )
                })?;
        }
    }
    lock_coordinator_cuda_resident_weights()?
        .preload_coordinator_w8a16_projection(library, weight_name, size_k, size_n)
        .map(|_| ())
}

pub(in crate::commands::real_full) fn preloaded_coordinator_w8a16_projection(
    weight_name: &str,
    size_k: usize,
    size_n: usize,
) -> Result<CoordinatorW8a16ProjectionBuffers> {
    if let Some((cached_k, cached_n, buffers)) =
        PRELOADED_COORDINATOR_W8A16_CACHE.with(|cache| cache.borrow().get(weight_name).copied())
    {
        anyhow::ensure!(
            cached_k == size_k && cached_n == size_n,
            "cached coordinator W8A16 projection {weight_name} shape {cached_k}x{cached_n} does not match requested {size_k}x{size_n}"
        );
        return Ok(buffers);
    }
    let buffers = lock_coordinator_cuda_resident_weights()?
        .preloaded_coordinator_w8a16_projection(weight_name, size_k, size_n)?;
    PRELOADED_COORDINATOR_W8A16_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(weight_name.to_owned(), (size_k, size_n, buffers));
    });
    Ok(buffers)
}

pub(in crate::commands::real_full) fn linear_rows_w8a16_preloaded_resident_weight_device_output(
    weight_name: &str,
    input: GlmrtDeviceBuffer,
    rows: usize,
    size_k: usize,
    size_n: usize,
) -> Result<DeviceBf16Output> {
    anyhow::ensure!(
        rows > 0,
        "coordinator W8A16 projection requires at least one row"
    );
    let output_bytes = rows
        .checked_mul(size_n)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("coordinator W8A16 multirow output bytes overflow")?;
    let library = cuda_native_library()?;
    let projection = preloaded_coordinator_w8a16_projection(weight_name, size_k, size_n)?;
    let output = OwnedCoordinatorDeviceBuffer::new(
        library,
        output_bytes,
        "W8A16 preloaded resident device-output linear output",
    )?;
    let input_row_bytes = size_k
        .checked_mul(std::mem::size_of::<u16>())
        .context("coordinator W8A16 input row bytes overflow")?;
    let output_row_bytes = size_n
        .checked_mul(std::mem::size_of::<u16>())
        .context("coordinator W8A16 output row bytes overflow")?;
    unsafe {
        let max_chunk_rows =
            if size_k == GLM52_HIDDEN_SIZE && size_n == 2_048 && !projection.packed_layout {
                2_048
            } else {
                256
            };
        let mut row_offset = 0_usize;
        while row_offset < rows {
            let chunk_rows = (rows - row_offset).min(max_chunk_rows);
            let chunk_input = device_buffer_byte_view(
                input,
                row_offset * input_row_bytes,
                chunk_rows * input_row_bytes,
                "coordinator W8A16 input rows",
            )?;
            let chunk_output = device_buffer_byte_view(
                output.buffer,
                row_offset * output_row_bytes,
                chunk_rows * output_row_bytes,
                "coordinator W8A16 output rows",
            )?;
            if chunk_rows == 1 {
                if projection.packed_layout {
                    library.cuda_linear_w8a16_group256_m1_warp_packed_async(
                        chunk_input,
                        projection.weight,
                        projection.scales,
                        chunk_output,
                        size_k,
                        size_n,
                        std::ptr::null_mut(),
                    )?;
                } else {
                    library.cuda_linear_w8a16_group256_m1_simt_async(
                        chunk_input,
                        projection.weight,
                        projection.scales,
                        chunk_output,
                        size_k,
                        size_n,
                        3,
                        std::ptr::null_mut(),
                    )?;
                }
            } else if projection.packed_layout && (3..=8).contains(&chunk_rows) {
                library.cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
                    chunk_input,
                    projection.weight,
                    projection.scales,
                    chunk_output,
                    chunk_rows,
                    size_k,
                    size_n,
                    std::ptr::null_mut(),
                )?;
            } else if projection.packed_layout {
                let block_m = if chunk_rows <= 16 {
                    16
                } else if chunk_rows <= 64 {
                    32
                } else {
                    48
                };
                let route_slots = chunk_rows.div_ceil(block_m) * block_m;
                let route_blocks = route_slots / block_m;
                let align_metadata = |offset: usize| (offset + 15) & !15;
                let global_scale_bytes = std::mem::size_of::<f32>();
                let routes_offset = align_metadata(global_scale_bytes);
                let routes_bytes = route_slots * std::mem::size_of::<i32>();
                let block_experts_offset = align_metadata(routes_offset + routes_bytes);
                let block_experts_bytes = route_blocks * std::mem::size_of::<i32>();
                let route_count_offset = align_metadata(block_experts_offset + block_experts_bytes);
                let topk_offset = align_metadata(route_count_offset + std::mem::size_of::<i32>());
                let topk_bytes = route_slots * std::mem::size_of::<f32>();
                let locks_offset = align_metadata(topk_offset + topk_bytes);
                let metadata_bytes = locks_offset
                    + B12X_COORDINATOR_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>();
                let metadata = OwnedCoordinatorDeviceBuffer::new(
                    library,
                    metadata_bytes,
                    "packed W8A16 O launch metadata",
                )?;
                let output_scratch_elements = size_n
                    .checked_mul(route_slots)
                    .context("packed W8A16 O output scratch elements overflow")?;
                let split_k_scratch_elements = 4_usize
                    .checked_mul(256)
                    .and_then(|elements| elements.checked_mul(block_m))
                    .and_then(|elements| elements.checked_mul(256))
                    .context("packed W8A16 O split-K scratch elements overflow")?;
                let scratch_elements = output_scratch_elements.max(split_k_scratch_elements);
                let scratch_bytes = scratch_elements
                    .checked_mul(std::mem::size_of::<f32>())
                    .context("packed W8A16 O reduction scratch bytes overflow")?;
                let scratch = OwnedCoordinatorDeviceBuffer::new(
                    library,
                    scratch_bytes,
                    "packed W8A16 O reduction scratch",
                )?;
                let buffers = GlmrtB12xCoordinatorW4a16Buffers {
                    input: chunk_input,
                    weight: projection.weight,
                    output: chunk_output,
                    scale: projection.scales,
                    global_scale: device_buffer_byte_view(
                        metadata.buffer,
                        0,
                        global_scale_bytes,
                        "packed W8A16 O global scale",
                    )?,
                    packed_route_indices: device_buffer_byte_view(
                        metadata.buffer,
                        routes_offset,
                        routes_bytes,
                        "packed W8A16 O routes",
                    )?,
                    block_expert_ids: device_buffer_byte_view(
                        metadata.buffer,
                        block_experts_offset,
                        block_experts_bytes,
                        "packed W8A16 O block experts",
                    )?,
                    packed_route_count: device_buffer_byte_view(
                        metadata.buffer,
                        route_count_offset,
                        std::mem::size_of::<i32>(),
                        "packed W8A16 O route count",
                    )?,
                    topk_weights: device_buffer_byte_view(
                        metadata.buffer,
                        topk_offset,
                        topk_bytes,
                        "packed W8A16 O top-k weights",
                    )?,
                    c_tmp: scratch.buffer,
                    locks: device_buffer_byte_view(
                        metadata.buffer,
                        locks_offset,
                        B12X_COORDINATOR_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>(),
                        "packed W8A16 O locks",
                    )?,
                };
                library.cuda_w8a16_packed_o_initialize_launch_buffers_async(
                    &buffers,
                    chunk_rows,
                    block_m,
                    std::ptr::null_mut(),
                )?;
                library.cuda_w8a16_packed_o_async(&buffers, chunk_rows, std::ptr::null_mut())?;
            } else {
                library.cuda_linear_w8a16_group256_aot_async(
                    chunk_input,
                    projection.weight,
                    projection.scales,
                    chunk_output,
                    chunk_rows,
                    size_k,
                    size_n,
                    std::ptr::null_mut(),
                )?;
            }
            row_offset += chunk_rows;
        }
        library
            .cuda_stream_synchronize(std::ptr::null_mut())
            .with_context(|| {
                format!(
                    "synchronizing coordinator W8A16 multirow projection {weight_name} rows={rows}"
                )
            })?;
    }
    Ok(DeviceBf16Output {
        buffer: output,
        bytes: output_bytes,
        rows,
        values_per_row: size_n,
        backend: "cuda-w8a16-group256-aot-preloaded-resident-weight",
    })
}

pub(in crate::commands::real_full) fn coordinator_w4a16_launch_buffers(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    projection: CoordinatorW4a16ProjectionBuffers,
    input: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
    metadata_slot: CoordinatorCudaScratchSlot,
) -> Result<GlmrtB12xCoordinatorW4a16Buffers> {
    let scratch = slot.buffer(
        library,
        CoordinatorCudaScratchSlot::Q,
        B12X_COORDINATOR_W4A16_SCRATCH_ELEMENTS * std::mem::size_of::<f32>(),
        "coordinator B12X W4A16 reduction scratch",
    )?;
    let metadata = slot.buffer(
        library,
        metadata_slot,
        B12X_COORDINATOR_W4A16_METADATA_BYTES,
        "coordinator B12X W4A16 launch metadata",
    )?;
    let routes_bytes = B12X_COORDINATOR_W4A16_ROUTE_SLOTS * std::mem::size_of::<i32>();
    let block_experts_offset = routes_bytes;
    let route_count_offset = block_experts_offset + std::mem::size_of::<i32>();
    let topk_weights_offset = route_count_offset + std::mem::size_of::<i32>();
    let topk_weights_bytes = B12X_COORDINATOR_W4A16_ROUTE_SLOTS * std::mem::size_of::<f32>();
    let locks_offset = 80;
    let buffers = GlmrtB12xCoordinatorW4a16Buffers {
        input,
        weight: projection.weight,
        output,
        scale: projection.scale,
        global_scale: projection.global_scale,
        packed_route_indices: device_buffer_byte_view(
            metadata,
            0,
            routes_bytes,
            "coordinator W4A16 packed routes",
        )?,
        block_expert_ids: device_buffer_byte_view(
            metadata,
            block_experts_offset,
            std::mem::size_of::<i32>(),
            "coordinator W4A16 block expert",
        )?,
        packed_route_count: device_buffer_byte_view(
            metadata,
            route_count_offset,
            std::mem::size_of::<i32>(),
            "coordinator W4A16 route count",
        )?,
        topk_weights: device_buffer_byte_view(
            metadata,
            topk_weights_offset,
            topk_weights_bytes,
            "coordinator W4A16 top-k weights",
        )?,
        c_tmp: scratch,
        locks: device_buffer_byte_view(
            metadata,
            locks_offset,
            B12X_COORDINATOR_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>(),
            "coordinator W4A16 locks",
        )?,
    };

    let metadata_ptr = metadata.ptr as usize;
    let initialized = B12X_COORDINATOR_W4A16_INITIALIZED_METADATA
        .with(|initialized| initialized.borrow().contains(&metadata_ptr));
    if !initialized {
        unsafe {
            library
                .cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
                    &buffers,
                    slot.stream_ptr(),
                )
                .context("initializing coordinator W4A16 launch metadata")?;
        }
        B12X_COORDINATOR_W4A16_INITIALIZED_METADATA.with(|initialized| {
            initialized.borrow_mut().insert(metadata_ptr);
        });
    }
    Ok(buffers)
}

pub(in crate::commands::real_full) fn coordinator_w8a16_packed_o_launch_buffers(
    library: &'static NativeLibrary,
    slot: &mut CoordinatorCudaGraphWorkspaceSlot,
    projection: CoordinatorW8a16ProjectionBuffers,
    input: GlmrtDeviceBuffer,
    output: GlmrtDeviceBuffer,
    metadata_slot: CoordinatorCudaScratchSlot,
) -> Result<GlmrtB12xCoordinatorW4a16Buffers> {
    anyhow::ensure!(
        projection.packed_layout,
        "coordinator packed W8A16 O launch requires packed weights"
    );
    let scratch = slot.buffer(
        library,
        CoordinatorCudaScratchSlot::T,
        COORDINATOR_W8A16_PACKED_O_SCRATCH_ELEMENTS * std::mem::size_of::<f32>(),
        "coordinator packed W8A16 O reduction scratch",
    )?;
    let metadata = slot.buffer(
        library,
        metadata_slot,
        COORDINATOR_W8A16_PACKED_O_METADATA_BYTES,
        "coordinator packed W8A16 O launch metadata",
    )?;
    let routes_offset = 16;
    let routes_bytes = COORDINATOR_W8A16_PACKED_O_ROUTE_SLOTS * std::mem::size_of::<i32>();
    let block_experts_offset = 80;
    let route_count_offset = 96;
    let topk_weights_offset = 112;
    let topk_weights_bytes = COORDINATOR_W8A16_PACKED_O_ROUTE_SLOTS * std::mem::size_of::<f32>();
    let locks_offset = 176;
    Ok(GlmrtB12xCoordinatorW4a16Buffers {
        input,
        weight: projection.weight,
        output,
        scale: projection.scales,
        global_scale: device_buffer_byte_view(
            metadata,
            0,
            std::mem::size_of::<f32>(),
            "coordinator packed W8A16 O global scale",
        )?,
        packed_route_indices: device_buffer_byte_view(
            metadata,
            routes_offset,
            routes_bytes,
            "coordinator packed W8A16 O routes",
        )?,
        block_expert_ids: device_buffer_byte_view(
            metadata,
            block_experts_offset,
            std::mem::size_of::<i32>(),
            "coordinator packed W8A16 O block expert",
        )?,
        packed_route_count: device_buffer_byte_view(
            metadata,
            route_count_offset,
            std::mem::size_of::<i32>(),
            "coordinator packed W8A16 O route count",
        )?,
        topk_weights: device_buffer_byte_view(
            metadata,
            topk_weights_offset,
            topk_weights_bytes,
            "coordinator packed W8A16 O top-k weights",
        )?,
        c_tmp: scratch,
        locks: device_buffer_byte_view(
            metadata,
            locks_offset,
            B12X_COORDINATOR_W4A16_LOCK_ELEMENTS * std::mem::size_of::<i32>(),
            "coordinator packed W8A16 O locks",
        )?,
    })
}

pub(in crate::commands::real_full) fn resident_weight_is_preloaded(
    weight_name: &str,
    expected_bytes: usize,
) -> bool {
    if validate_resident_weight_name(weight_name).is_err() || expected_bytes == 0 {
        return false;
    }
    COORDINATOR_CUDA_RESIDENT_WEIGHTS
        .get()
        .and_then(|resident_weights| resident_weights.lock().ok())
        .map(|resident_weights| {
            resident_weights.resident_weight_is_preloaded(weight_name, expected_bytes)
        })
        .unwrap_or(false)
}

pub(in crate::commands::real_full) fn validate_resident_weight_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        anyhow::bail!("real full resident CUDA weight name must not be empty");
    }
    if name.as_bytes().iter().any(|byte| *byte == 0) {
        anyhow::bail!("real full resident CUDA weight name must not contain NUL bytes");
    }
    Ok(())
}

#[cfg(not(test))]
pub(in crate::commands::real_full) fn resident_weight_registry_key(
    name: &str,
    _bytes: usize,
) -> String {
    name.to_owned()
}

pub(in crate::commands::real_full) fn lock_coordinator_cuda_resident_weights(
) -> Result<MutexGuard<'static, CoordinatorCudaResidentWeights>> {
    COORDINATOR_CUDA_RESIDENT_WEIGHTS
        .get_or_init(|| Mutex::new(CoordinatorCudaResidentWeights::default()))
        .lock()
        .map_err(|_| anyhow::anyhow!("coordinator CUDA resident weights mutex poisoned"))
}

pub(in crate::commands::real_full) fn resident_weight_buffer_from_registry(
    name: &str,
    src: &[u8],
    label: &'static str,
) -> Result<GlmrtDeviceBuffer> {
    let library = cuda_native_library()?;
    let mut resident_weights = lock_coordinator_cuda_resident_weights()?;
    resident_weights.resident_weight_buffer(library, name, src, label)
}

pub(in crate::commands::real_full) fn preloaded_resident_weight_device_buffer(
    name: &str,
    expected_bytes: usize,
) -> Result<GlmrtDeviceBuffer> {
    if let Some((cached_bytes, buffer)) =
        PRELOADED_RESIDENT_WEIGHT_CACHE.with(|cache| cache.borrow().get(name).copied())
    {
        anyhow::ensure!(
            cached_bytes == expected_bytes,
            "cached resident CUDA weight {name} byte length {cached_bytes} does not match requested {expected_bytes}"
        );
        return Ok(buffer);
    }
    validate_resident_weight_name(name)?;
    let resident_weights = lock_coordinator_cuda_resident_weights()?;
    let buffer = resident_weights.preloaded_resident_weight_buffer(name, expected_bytes)?;
    drop(resident_weights);
    PRELOADED_RESIDENT_WEIGHT_CACHE.with(|cache| {
        cache
            .borrow_mut()
            .insert(name.to_owned(), (expected_bytes, buffer));
    });
    Ok(buffer)
}

pub(in crate::commands::real_full) fn release_preloaded_resident_weight_device_buffer(
    name: &str,
    expected_bytes: usize,
) -> Result<()> {
    validate_resident_weight_name(name)?;
    PRELOADED_RESIDENT_WEIGHT_CACHE.with(|cache| {
        cache.borrow_mut().remove(name);
    });
    let library = cuda_native_library()?;
    lock_coordinator_cuda_resident_weights()?.release_resident_weight(library, name, expected_bytes)
}

pub(in crate::commands::real_full) fn preloaded_resident_weight_device_buffer_view(
    name: &str,
    expected_full_bytes: usize,
    offset_bytes: usize,
    view_bytes: usize,
) -> Result<GlmrtDeviceBuffer> {
    anyhow::ensure!(
        view_bytes > 0,
        "preloaded resident coordinator CUDA weight {name} view requires non-zero bytes"
    );
    let full = preloaded_resident_weight_device_buffer(name, expected_full_bytes)?;
    let end = offset_bytes
        .checked_add(view_bytes)
        .context("preloaded resident coordinator CUDA weight view byte range overflows usize")?;
    anyhow::ensure!(
        end <= full.bytes,
        "preloaded resident coordinator CUDA weight {name} view [{offset_bytes}, {end}) exceeds resident bytes {}",
        full.bytes
    );
    Ok(GlmrtDeviceBuffer {
        ptr: full.ptr.cast::<u8>().wrapping_add(offset_bytes).cast(),
        bytes: view_bytes,
        device_id: full.device_id,
        flags: full.flags,
    })
}

pub(in crate::commands::real_full) fn preload_resident_weight_from_host_staging(
    weight_name: &str,
    weight_bytes: usize,
    label: &'static str,
    fill_staging: impl FnOnce(&mut [u8]) -> Result<()>,
) -> Result<()> {
    preload_resident_weight_from_host_staging_profiled(
        weight_name,
        weight_bytes,
        label,
        fill_staging,
    )
    .map(|_| ())
}

pub(in crate::commands::real_full) fn preload_resident_weight_from_host_staging_profiled(
    weight_name: &str,
    weight_bytes: usize,
    label: &'static str,
    fill_staging: impl FnOnce(&mut [u8]) -> Result<()>,
) -> Result<ResidentWeightPreloadTimings> {
    validate_resident_weight_name(weight_name)?;
    let library_started = std::time::Instant::now();
    let library = cuda_native_library()?;
    let library_ms = library_started.elapsed().as_secs_f64() * 1_000.0;
    let lock_started = std::time::Instant::now();
    let mut resident_weights = lock_coordinator_cuda_resident_weights()?;
    let lock_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
    let (_, mut timings) = resident_weights.resident_weight_buffer_from_host_staging_profiled(
        library,
        weight_name,
        weight_bytes,
        label,
        fill_staging,
    )?;
    timings.library_ms = library_ms;
    timings.lock_ms = lock_ms;
    Ok(timings)
}

pub(in crate::commands::real_full) fn preload_resident_weight_from_pinned_host_profiled(
    weight_name: &str,
    weight_bytes: usize,
    label: &'static str,
    source: GlmrtHostBuffer,
) -> Result<ResidentWeightPreloadTimings> {
    validate_resident_weight_name(weight_name)?;
    let library_started = std::time::Instant::now();
    let library = cuda_native_library()?;
    let library_ms = library_started.elapsed().as_secs_f64() * 1_000.0;
    let lock_started = std::time::Instant::now();
    let mut resident_weights = lock_coordinator_cuda_resident_weights()?;
    let lock_ms = lock_started.elapsed().as_secs_f64() * 1_000.0;
    let (_, mut timings) = resident_weights.resident_weight_buffer_from_pinned_host_profiled(
        library,
        weight_name,
        weight_bytes,
        label,
        source,
    )?;
    timings.library_ms = library_ms;
    timings.lock_ms = lock_ms;
    Ok(timings)
}

pub(in crate::commands::real_full) fn replace_preloaded_block_fp8_weight_with_bf16(
    weight_name: &str,
    scale_bytes: &[u8],
    size_k: usize,
    size_n: usize,
) -> Result<()> {
    validate_resident_weight_name(weight_name)?;
    anyhow::ensure!(
        size_k > 0 && size_n > 0,
        "block-FP8 coordinator weight {weight_name} requires positive dimensions"
    );
    let source_bytes = size_n
        .checked_mul(size_k)
        .context("block-FP8 coordinator source bytes overflow")?;
    let output_bytes = source_bytes
        .checked_mul(std::mem::size_of::<u16>())
        .context("block-FP8 coordinator BF16 bytes overflow")?;
    let expected_scale_bytes = size_n
        .div_ceil(128)
        .checked_mul(size_k.div_ceil(128))
        .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
        .context("block-FP8 coordinator scale bytes overflow")?;
    anyhow::ensure!(
        scale_bytes.len() == expected_scale_bytes,
        "block-FP8 coordinator weight {weight_name} has {} scale bytes, expected {expected_scale_bytes}",
        scale_bytes.len()
    );

    let library = cuda_native_library()?;
    let mut scales = library
        .alloc_device_buffer(expected_scale_bytes)
        .with_context(|| format!("allocating block-FP8 scales for {weight_name}"))?;
    let mut output = library
        .alloc_device_buffer(output_bytes)
        .with_context(|| format!("allocating block-FP8 BF16 output for {weight_name}"))?;
    let conversion = (|| -> Result<()> {
        library
            .copy_h2d(scales, scale_bytes)
            .with_context(|| format!("uploading block-FP8 scales for {weight_name}"))?;
        let mut resident_weights = lock_coordinator_cuda_resident_weights()?;
        let source =
            resident_weights.preloaded_resident_weight_buffer(weight_name, source_bytes)?;
        unsafe {
            library
                .cuda_dequantize_block_fp8_e4m3_bf16_async(
                    source,
                    scales,
                    output,
                    size_k,
                    size_n,
                    std::ptr::null_mut(),
                )
                .with_context(|| {
                    format!("dequantizing block-FP8 coordinator weight {weight_name}")
                })?;
            library
                .cuda_stream_synchronize(std::ptr::null_mut())
                .with_context(|| {
                    format!("synchronizing block-FP8 coordinator weight {weight_name}")
                })?;
        }

        let source_key = resident_weight_registry_key(weight_name, source_bytes);
        let mut source_resident = resident_weights
            .resident_weights
            .remove(&source_key)
            .with_context(|| format!("block-FP8 coordinator weight {weight_name} disappeared"))?;
        anyhow::ensure!(
            source_resident.bytes == source_bytes,
            "block-FP8 coordinator weight {weight_name} has {} resident bytes, expected {source_bytes}",
            source_resident.bytes
        );
        library
            .free_device_buffer(&mut source_resident.buffer)
            .with_context(|| format!("freeing superseded block-FP8 weight {weight_name}"))?;
        let output_key = resident_weight_registry_key(weight_name, output_bytes);
        let previous = resident_weights.resident_weights.insert(
            output_key,
            ResidentDeviceBuffer {
                buffer: output,
                bytes: output_bytes,
                uploaded: true,
                upload_count: 1,
                label: "coordinator block-FP8 expanded BF16 weight",
            },
        );
        anyhow::ensure!(
            previous.is_none(),
            "block-FP8 coordinator weight {weight_name} replacement collided"
        );
        output = GlmrtDeviceBuffer::default();
        PRELOADED_RESIDENT_WEIGHT_CACHE.with(|cache| {
            cache.borrow_mut().remove(weight_name);
        });
        Ok(())
    })();
    let scale_cleanup = library.free_device_buffer(&mut scales);
    let output_cleanup = if output.ptr.is_null() {
        Ok(())
    } else {
        library.free_device_buffer(&mut output)
    };
    conversion?;
    scale_cleanup.with_context(|| format!("freeing block-FP8 scales for {weight_name}"))?;
    output_cleanup.with_context(|| format!("freeing block-FP8 output for {weight_name}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        coordinator_w4a16_projection_flag_enabled, parse_coordinator_w4a16_flag,
        parse_coordinator_w8a16_flag,
    };

    #[test]
    fn coordinator_w4a16_flag_accepts_only_explicit_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "on", "W4A16"] {
            assert!(parse_coordinator_w4a16_flag(Some(value)), "{value}");
        }
        for value in ["", "0", "false", "off", "w4"] {
            assert!(!parse_coordinator_w4a16_flag(Some(value)), "{value}");
        }
        assert!(!parse_coordinator_w4a16_flag(None));
    }

    #[test]
    fn legacy_w4a16_flag_enables_each_projection_independently() {
        assert!(coordinator_w4a16_projection_flag_enabled(Some("1"), None));
        assert!(coordinator_w4a16_projection_flag_enabled(None, Some("1")));
        assert!(!coordinator_w4a16_projection_flag_enabled(
            Some("0"),
            Some("0")
        ));
    }

    #[test]
    fn coordinator_w8a16_flag_does_not_accept_the_w4a16_name() {
        for value in ["1", "true", "TRUE", " yes ", "on", "W8A16"] {
            assert!(parse_coordinator_w8a16_flag(Some(value)), "{value}");
        }
        for value in ["", "0", "false", "off", "w8", "w4a16"] {
            assert!(!parse_coordinator_w8a16_flag(Some(value)), "{value}");
        }
        assert!(!parse_coordinator_w8a16_flag(None));
    }
}
