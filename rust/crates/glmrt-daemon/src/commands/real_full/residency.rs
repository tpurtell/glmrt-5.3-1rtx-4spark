use anyhow::{Context, Result};
use glmrt_core::{DType, TensorCatalog, TensorInfo, TensorRole};
use glmrt_ffi::{GlmrtHostBuffer, NativeLibrary};
use glmrt_loader::{read_tensor_bytes_into, LoadedTensorSummary};
use std::collections::BTreeMap;
use std::env;
use std::slice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

use super::coordinator_kernels::{
    coordinator_w4a16_o_proj_decode_enabled, coordinator_w4a16_q_b_decode_enabled,
    coordinator_w8a16_o_proj_decode_enabled, coordinator_w8a16_q_a_decode_enabled,
    coordinator_w8a16_q_b_decode_enabled, cuda_native_library,
    preload_coordinator_w4a16_projection, preload_coordinator_w8a16_projection,
    preload_resident_weight_from_host_staging, preload_resident_weight_from_pinned_host_profiled,
    release_preloaded_resident_weight_device_buffer, replace_preloaded_block_fp8_weight_with_bf16,
    ResidentWeightPreloadTimings,
};
use super::layer_blocks::{tensor_is_spark_layer_block_resident, SparkLayerBlock};
use super::sparse_mlp::cache_router_correction_bias_host_values;
use super::types::RealFullCoordinatorResidentPreloadPlan;

const COORDINATOR_RESIDENT_PRELOAD_SCOPE: &str =
    "select coordinator-owned immutable GLM-5.2 tensors for named startup GPU residency";
const COORDINATOR_RESIDENT_SAMPLE_LIMIT: usize = 12;
const COORDINATOR_RESIDENT_SOURCE_WORKERS: usize = 8;
const COORDINATOR_BLOCK_FP8_BLOCK_ROWS: usize = 128;
const COORDINATOR_BLOCK_FP8_BLOCK_COLUMNS: usize = 128;
const COORDINATOR_INCLUDE_MTP_LAYER_ENV: &str = "GLMRT_COORDINATOR_INCLUDE_MTP_LAYER";
const REAL_FULL_MTP_ENV: &str = "GLMRT_REAL_FULL_MTP";
const REAL_FULL_MTP_PROBE_ENV: &str = "GLMRT_REAL_FULL_MTP_PROBE";
const REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS: [&str; 7] = [
    "Embedding",
    "LmHead",
    "Attention",
    "Router",
    "Norm",
    "DenseMlp",
    "SharedExpert",
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SparkLayerBlockResidentPreloadStats {
    pub(crate) layers: usize,
    pub(crate) tensors: usize,
    pub(crate) bytes: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct CoordinatorResidentTensorPreloadProfile {
    bytes: u64,
    total_ms: f64,
    validation_cache_ms: f64,
    resident: ResidentWeightPreloadTimings,
    w4a16_pack_ms: f64,
    w8a16_pack_ms: f64,
    release_ms: f64,
    block_fp8_dequant_ms: f64,
    block_fp8_dequantized: bool,
    w4a16_packed: bool,
    w8a16_q_a_packed: bool,
    w8a16_q_b_packed: bool,
    w8a16_o_packed: bool,
    source_released: bool,
}

struct CoordinatorPinnedSourceBuffer {
    library: &'static NativeLibrary,
    buffer: GlmrtHostBuffer,
}

// Ownership of the pinned allocation moves between exactly one source worker
// and the coordinator preload thread. The worker does not touch it again until
// the synchronous H2D copy returns and ownership is sent back.
unsafe impl Send for CoordinatorPinnedSourceBuffer {}

impl CoordinatorPinnedSourceBuffer {
    fn new(library: &'static NativeLibrary) -> Self {
        Self {
            library,
            buffer: GlmrtHostBuffer::default(),
        }
    }

    fn ensure_capacity(&mut self, bytes: usize) -> Result<f64> {
        if !self.buffer.ptr.is_null() && self.buffer.bytes >= bytes {
            return Ok(0.0);
        }
        let started = Instant::now();
        if !self.buffer.ptr.is_null() {
            self.library
                .free_host_buffer(&mut self.buffer)
                .context("freeing undersized coordinator resident source pinned buffer")?;
            self.buffer = GlmrtHostBuffer::default();
        }
        self.buffer = self
            .library
            .alloc_host_buffer(bytes)
            .context("allocating coordinator resident source pinned buffer")?;
        anyhow::ensure!(
            !self.buffer.ptr.is_null() && self.buffer.bytes >= bytes,
            "coordinator resident source pinned allocation returned {} bytes for {bytes}",
            self.buffer.bytes
        );
        Ok(started.elapsed().as_secs_f64() * 1_000.0)
    }

    fn capacity(&self) -> usize {
        self.buffer.bytes
    }

    fn as_mut_slice(&mut self, bytes: usize) -> Result<&mut [u8]> {
        anyhow::ensure!(
            !self.buffer.ptr.is_null() && self.buffer.bytes >= bytes,
            "coordinator resident source pinned buffer has {} bytes, needs {bytes}",
            self.buffer.bytes
        );
        Ok(unsafe { slice::from_raw_parts_mut(self.buffer.ptr.cast::<u8>(), bytes) })
    }

    fn as_slice(&self, bytes: usize) -> Result<&[u8]> {
        anyhow::ensure!(
            !self.buffer.ptr.is_null() && self.buffer.bytes >= bytes,
            "coordinator resident source pinned buffer has {} bytes, needs {bytes}",
            self.buffer.bytes
        );
        Ok(unsafe { slice::from_raw_parts(self.buffer.ptr.cast::<u8>(), bytes) })
    }
}

impl Drop for CoordinatorPinnedSourceBuffer {
    fn drop(&mut self) {
        if !self.buffer.ptr.is_null() {
            let _ = self.library.free_host_buffer(&mut self.buffer);
            self.buffer = GlmrtHostBuffer::default();
        }
    }
}

struct CoordinatorResidentSourceMessage {
    worker_id: usize,
    index: usize,
    summary: Result<LoadedTensorSummary>,
    buffer: CoordinatorPinnedSourceBuffer,
    completed_ms: f64,
    allocation_ms: f64,
    read_ms: f64,
}

pub(crate) fn preload_real_full_spark_layer_block_weights(
    catalog: &TensorCatalog,
    block: SparkLayerBlock,
) -> Result<SparkLayerBlockResidentPreloadStats> {
    let tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor_is_spark_layer_block_resident(tensor, block))
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !tensors.is_empty(),
        "Spark layer block {}:{} selected no resident tensors",
        block.start_layer,
        block.end_layer
    );
    let mut loaded_bytes = 0_u64;
    for tensor in &tensors {
        let expected_bytes: usize = tensor.byte_length.try_into().with_context(|| {
            format!(
                "Spark layer-block tensor {} byte length {} does not fit in usize",
                tensor.name, tensor.byte_length
            )
        })?;
        preload_resident_weight_from_host_staging(
            &tensor.name,
            expected_bytes,
            "startup resident Spark layer-block weight pinned staging",
            |staging| {
                let summary = read_tensor_bytes_into(catalog, &tensor.name, staging)
                    .with_context(|| format!("reading Spark layer-block tensor {}", tensor.name))?;
                validate_coordinator_resident_tensor_summary(tensor, &summary)?;
                cache_router_correction_bias_host_values(
                    catalog,
                    tensor,
                    &staging[..expected_bytes],
                )?;
                Ok(())
            },
        )
        .with_context(|| format!("preloading Spark layer-block tensor {}", tensor.name))?;
        loaded_bytes = loaded_bytes
            .checked_add(tensor.byte_length)
            .context("Spark layer-block resident byte count overflow")?;
    }
    Ok(SparkLayerBlockResidentPreloadStats {
        layers: block.layer_count(),
        tensors: tensors.len(),
        bytes: loaded_bytes,
    })
}

pub(super) fn real_full_coordinator_resident_preload_plan(
    catalog: &TensorCatalog,
) -> RealFullCoordinatorResidentPreloadPlan {
    coordinator_resident_preload_plan_for_tensors(catalog, "planned", 0)
}

pub(super) fn preload_real_full_coordinator_resident_weights(
    catalog: &TensorCatalog,
) -> Result<RealFullCoordinatorResidentPreloadPlan> {
    let preload_started = Instant::now();
    let mut tensors = coordinator_resident_tensors(catalog);
    // Give every source worker its largest tensor first. Its pinned allocation
    // can then be reused for every later tensor without repeated cudaHostAlloc /
    // cudaFreeHost calls contending with H2D and projection packing.
    tensors.sort_by(|left, right| {
        right
            .byte_length
            .cmp(&left.byte_length)
            .then_with(|| left.name.cmp(&right.name))
    });
    let source_library_started = Instant::now();
    let source_library = cuda_native_library()?;
    let source_library_ms = source_library_started.elapsed().as_secs_f64() * 1_000.0;
    let source_started = Instant::now();
    let next_tensor = AtomicUsize::new(0);
    let source_workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(COORDINATOR_RESIDENT_SOURCE_WORKERS)
        .min(tensors.len().max(1));
    let (source_sender, source_receiver) = mpsc::channel();
    let mut source_bytes = 0_u64;
    let mut source_ms = 0.0_f64;
    let mut source_allocation_ms = 0.0_f64;
    let mut source_read_ms = 0.0_f64;
    let mut source_worker_peak_bytes = vec![0_usize; source_workers];
    let mut loaded_bytes = 0_u64;
    let mut upload_pack_ms = 0.0_f64;
    let mut validation_cache_ms = 0.0_f64;
    let mut resident_library_ms = 0.0_f64;
    let mut resident_lock_ms = 0.0_f64;
    let mut device_allocation_ms = 0.0_f64;
    let mut staging_allocation_ms = 0.0_f64;
    let mut staging_fill_ms = 0.0_f64;
    let mut h2d_ms = 0.0_f64;
    let mut resident_finalize_ms = 0.0_f64;
    let mut w4a16_pack_ms = 0.0_f64;
    let mut w8a16_pack_ms = 0.0_f64;
    let mut release_ms = 0.0_f64;
    let mut block_fp8_dequant_ms = 0.0_f64;
    let mut uploaded_tensors = 0_usize;
    let mut w4a16_packed_tensors = 0_usize;
    let mut w8a16_q_a_packed_tensors = 0_usize;
    let mut w8a16_q_b_packed_tensors = 0_usize;
    let mut w8a16_o_packed_tensors = 0_usize;
    let mut released_source_tensors = 0_usize;
    let mut block_fp8_dequantized_tensors = 0_usize;
    std::thread::scope(|scope| -> Result<()> {
        let mut source_return_senders = Vec::with_capacity(source_workers);
        for worker_id in 0..source_workers {
            let sender = source_sender.clone();
            let next_tensor = &next_tensor;
            let tensors = &tensors;
            let (return_sender, return_receiver) = mpsc::channel();
            source_return_senders.push(return_sender);
            scope.spawn(move || {
                let mut buffer = CoordinatorPinnedSourceBuffer::new(source_library);
                loop {
                    let index = next_tensor.fetch_add(1, Ordering::Relaxed);
                    let Some(tensor) = tensors.get(index) else {
                        break;
                    };
                    let expected_bytes: usize =
                        match tensor.byte_length.try_into().with_context(|| {
                            format!(
                                "coordinator resident source tensor {} byte length {} does not fit in usize",
                                tensor.name, tensor.byte_length
                            )
                        }) {
                            Ok(bytes) => bytes,
                            Err(error) => {
                                let _ = sender.send(Err(error));
                                break;
                            }
                        };
                    let allocation_ms = match buffer.ensure_capacity(expected_bytes) {
                        Ok(elapsed_ms) => elapsed_ms,
                        Err(error) => {
                            let _ = sender.send(Err(error.context(format!(
                                "allocating coordinator resident source tensor {}",
                                tensor.name
                            ))));
                            break;
                        }
                    };
                    let read_started = Instant::now();
                    let summary = buffer
                        .as_mut_slice(expected_bytes)
                        .and_then(|staging| read_tensor_bytes_into(catalog, &tensor.name, staging))
                        .with_context(|| {
                            format!("reading coordinator resident tensor {}", tensor.name)
                        });
                    let read_ms = read_started.elapsed().as_secs_f64() * 1_000.0;
                    let completed_ms = source_started.elapsed().as_secs_f64() * 1_000.0;
                    let message = CoordinatorResidentSourceMessage {
                        worker_id,
                        index,
                        summary,
                        buffer,
                        completed_ms,
                        allocation_ms,
                        read_ms,
                    };
                    if sender.send(Ok(message)).is_err() {
                        break;
                    }
                    buffer = match return_receiver.recv() {
                        Ok(buffer) => buffer,
                        Err(_) => break,
                    };
                }
            });
        }
        drop(source_sender);
        for _ in 0..tensors.len() {
            let message = source_receiver
                .recv()
                .context("coordinator resident source workers stopped before completion")??;
            let tensor = tensors
                .get(message.index)
                .context("coordinator resident source worker returned an invalid tensor index")?;
            let summary = message.summary?;
            source_ms = source_ms.max(message.completed_ms);
            source_allocation_ms += message.allocation_ms;
            source_read_ms += message.read_ms;
            source_worker_peak_bytes[message.worker_id] =
                source_worker_peak_bytes[message.worker_id].max(message.buffer.capacity());
            source_bytes = source_bytes
                .checked_add(summary.bytes_read)
                .context("coordinator resident source byte count overflow")?;
            let profile = preload_real_full_coordinator_resident_tensor(
                catalog,
                tensor,
                &summary,
                &message.buffer,
            )?;
            upload_pack_ms += profile.total_ms;
            validation_cache_ms += profile.validation_cache_ms;
            resident_library_ms += profile.resident.library_ms;
            resident_lock_ms += profile.resident.lock_ms;
            device_allocation_ms += profile.resident.device_allocation_ms;
            staging_allocation_ms += profile.resident.staging_allocation_ms;
            staging_fill_ms += profile.resident.staging_fill_ms;
            h2d_ms += profile.resident.h2d_ms;
            resident_finalize_ms += profile.resident.finalize_ms;
            w4a16_pack_ms += profile.w4a16_pack_ms;
            w8a16_pack_ms += profile.w8a16_pack_ms;
            release_ms += profile.release_ms;
            block_fp8_dequant_ms += profile.block_fp8_dequant_ms;
            uploaded_tensors += usize::from(profile.resident.uploaded);
            w4a16_packed_tensors += usize::from(profile.w4a16_packed);
            w8a16_q_a_packed_tensors += usize::from(profile.w8a16_q_a_packed);
            w8a16_q_b_packed_tensors += usize::from(profile.w8a16_q_b_packed);
            w8a16_o_packed_tensors += usize::from(profile.w8a16_o_packed);
            released_source_tensors += usize::from(profile.source_released);
            block_fp8_dequantized_tensors += usize::from(profile.block_fp8_dequantized);
            loaded_bytes = loaded_bytes
                .checked_add(profile.bytes)
                .context("coordinator resident loaded byte count overflow")?;
            source_return_senders
                .get(message.worker_id)
                .context("coordinator resident source worker return channel is missing")?
                .send(message.buffer)
                .map_err(|_| {
                    anyhow::anyhow!(
                        "coordinator resident source worker {} stopped before buffer return",
                        message.worker_id
                    )
                })?;
        }
        drop(source_return_senders);
        Ok(())
    })?;
    let source_peak_pinned_bytes = source_worker_peak_bytes.iter().copied().sum::<usize>();
    let source_gbps = source_bytes as f64 / (source_ms * 1.0e6).max(1.0);
    eprintln!(
        "real_full_coordinator_resident_source_load tensors={} workers={} bytes={} elapsed_ms={source_ms:.3} source_gbps={source_gbps:.3} library_ms={source_library_ms:.3} allocation_ms={source_allocation_ms:.3} read_sum_ms={source_read_ms:.3} peak_pinned_bytes={source_peak_pinned_bytes}",
        tensors.len(),
        source_workers,
        source_bytes,
    );
    let total_ms = preload_started.elapsed().as_secs_f64() * 1_000.0;
    let overlap_ms = (source_ms + upload_pack_ms - total_ms).max(0.0);
    let total_gbps = loaded_bytes as f64 / (total_ms * 1.0e6).max(1.0);
    eprintln!(
        "real_full_coordinator_resident_preload tensors={} bytes={} source_ms={source_ms:.3} upload_pack_ms={upload_pack_ms:.3} overlap_ms={overlap_ms:.3} total_ms={total_ms:.3} effective_gbps={total_gbps:.3}",
        tensors.len(),
        loaded_bytes,
    );
    let attributed_ms = validation_cache_ms
        + resident_library_ms
        + resident_lock_ms
        + device_allocation_ms
        + staging_allocation_ms
        + staging_fill_ms
        + h2d_ms
        + resident_finalize_ms
        + w4a16_pack_ms
        + w8a16_pack_ms
        + release_ms
        + block_fp8_dequant_ms;
    let unattributed_ms = (upload_pack_ms - attributed_ms).max(0.0);
    eprintln!(
        "real_full_coordinator_resident_preload_detail uploaded_tensors={uploaded_tensors} validation_cache_ms={validation_cache_ms:.3} library_ms={resident_library_ms:.3} lock_ms={resident_lock_ms:.3} device_allocation_ms={device_allocation_ms:.3} staging_allocation_ms={staging_allocation_ms:.3} staging_fill_ms={staging_fill_ms:.3} h2d_ms={h2d_ms:.3} finalize_ms={resident_finalize_ms:.3} block_fp8_dequantized_tensors={block_fp8_dequantized_tensors} block_fp8_dequant_ms={block_fp8_dequant_ms:.3} w4a16_pack_tensors={w4a16_packed_tensors} w4a16_pack_ms={w4a16_pack_ms:.3} w8a16_q_a_tensors={w8a16_q_a_packed_tensors} w8a16_q_b_tensors={w8a16_q_b_packed_tensors} w8a16_o_tensors={w8a16_o_packed_tensors} w8a16_pack_ms={w8a16_pack_ms:.3} released_source_tensors={released_source_tensors} release_ms={release_ms:.3} unattributed_ms={unattributed_ms:.3}"
    );
    Ok(coordinator_resident_preload_plan_for_tensors(
        catalog,
        "loaded",
        loaded_bytes,
    ))
}

fn preload_real_full_coordinator_resident_tensor(
    catalog: &TensorCatalog,
    tensor: &TensorInfo,
    summary: &LoadedTensorSummary,
    source: &CoordinatorPinnedSourceBuffer,
) -> Result<CoordinatorResidentTensorPreloadProfile> {
    let tensor_started = Instant::now();
    let expected_bytes: usize = tensor.byte_length.try_into().with_context(|| {
        format!(
            "coordinator resident tensor {} byte length {} does not fit in usize",
            tensor.name, tensor.byte_length
        )
    })?;
    let validation_cache_started = Instant::now();
    validate_coordinator_resident_tensor_summary(tensor, summary)?;
    cache_router_correction_bias_host_values(catalog, tensor, source.as_slice(expected_bytes)?)?;
    let validation_cache_ms = validation_cache_started.elapsed().as_secs_f64() * 1_000.0;
    let resident = preload_resident_weight_from_pinned_host_profiled(
        &tensor.name,
        expected_bytes,
        "startup resident coordinator weight direct pinned source",
        source.buffer,
    )
    .with_context(|| format!("preloading coordinator resident tensor {}", tensor.name))?;
    let mut resident_source_bytes = expected_bytes;
    let mut block_fp8_dequant_ms = 0.0_f64;
    let block_fp8_dequantized = tensor.dtype == DType::F8E4M3;
    if block_fp8_dequantized {
        anyhow::ensure!(
            tensor.shape.len() == 2,
            "block-FP8 coordinator tensor {} must be a matrix",
            tensor.name
        );
        let size_n = tensor.shape[0];
        let size_k = tensor.shape[1];
        let scale_bytes = load_coordinator_block_fp8_scale_bytes(catalog, tensor)?;
        let dequant_started = Instant::now();
        replace_preloaded_block_fp8_weight_with_bf16(&tensor.name, &scale_bytes, size_k, size_n)
            .with_context(|| format!("expanding block-FP8 coordinator tensor {}", tensor.name))?;
        block_fp8_dequant_ms = dequant_started.elapsed().as_secs_f64() * 1_000.0;
        resident_source_bytes = expected_bytes
            .checked_mul(std::mem::size_of::<u16>())
            .context("block-FP8 coordinator resident BF16 bytes overflow")?;
    }
    let pack_w4a16_q_b = tensor.name.ends_with(".self_attn.q_b_proj.weight")
        && coordinator_w4a16_q_b_decode_enabled();
    let pack_w4a16_o_proj = tensor.name.ends_with(".self_attn.o_proj.weight")
        && coordinator_w4a16_o_proj_decode_enabled();
    let pack_w8a16_o_proj = tensor.name.ends_with(".self_attn.o_proj.weight")
        && coordinator_w8a16_o_proj_decode_enabled();
    let pack_w8a16_q_a = tensor.name.ends_with(".self_attn.q_a_proj.weight")
        && coordinator_w8a16_q_a_decode_enabled();
    let pack_w8a16_q_b = tensor.name.ends_with(".self_attn.q_b_proj.weight")
        && coordinator_w8a16_q_b_decode_enabled();
    anyhow::ensure!(
        !(pack_w4a16_o_proj && pack_w8a16_o_proj),
        "coordinator O projection cannot enable W4A16 and W8A16 simultaneously"
    );
    anyhow::ensure!(
        !(pack_w4a16_q_b && pack_w8a16_q_b),
        "coordinator Q-B projection cannot enable W4A16 and W8A16 simultaneously"
    );
    let mut w4a16_pack_ms = 0.0_f64;
    let mut w8a16_pack_ms = 0.0_f64;
    let mut release_ms = 0.0_f64;
    if pack_w4a16_q_b || pack_w4a16_o_proj {
        anyhow::ensure!(
            (tensor.dtype == DType::Bf16 || block_fp8_dequantized) && tensor.shape.len() == 2,
            "coordinator W4A16 projection {} must be a BF16 matrix",
            tensor.name
        );
        let size_n: usize = tensor.shape[0].try_into().with_context(|| {
            format!("coordinator W4A16 projection {} rows overflow", tensor.name)
        })?;
        let size_k: usize = tensor.shape[1].try_into().with_context(|| {
            format!(
                "coordinator W4A16 projection {} columns overflow",
                tensor.name
            )
        })?;
        let pack_started = Instant::now();
        preload_coordinator_w4a16_projection(&tensor.name, size_k, size_n)
            .with_context(|| format!("packing coordinator W4A16 projection {}", tensor.name))?;
        w4a16_pack_ms = pack_started.elapsed().as_secs_f64() * 1_000.0;
    }
    if pack_w8a16_q_a || pack_w8a16_q_b || pack_w8a16_o_proj {
        anyhow::ensure!(
            (tensor.dtype == DType::Bf16 || block_fp8_dequantized) && tensor.shape.len() == 2,
            "coordinator W8A16 projection {} must be a BF16 matrix",
            tensor.name
        );
        let size_n: usize = tensor.shape[0].try_into().with_context(|| {
            format!("coordinator W8A16 projection {} rows overflow", tensor.name)
        })?;
        let size_k: usize = tensor.shape[1].try_into().with_context(|| {
            format!(
                "coordinator W8A16 projection {} columns overflow",
                tensor.name
            )
        })?;
        let pack_started = Instant::now();
        preload_coordinator_w8a16_projection(&tensor.name, size_k, size_n)
            .with_context(|| format!("packing coordinator W8A16 projection {}", tensor.name))?;
        w8a16_pack_ms = pack_started.elapsed().as_secs_f64() * 1_000.0;
        let release_started = Instant::now();
        release_preloaded_resident_weight_device_buffer(&tensor.name, resident_source_bytes)
            .with_context(|| {
                format!(
                    "releasing superseded BF16 coordinator projection {}",
                    tensor.name
                )
            })?;
        release_ms = release_started.elapsed().as_secs_f64() * 1_000.0;
    }
    Ok(CoordinatorResidentTensorPreloadProfile {
        bytes: summary.bytes_read,
        total_ms: tensor_started.elapsed().as_secs_f64() * 1_000.0,
        validation_cache_ms,
        resident,
        w4a16_pack_ms,
        w8a16_pack_ms,
        release_ms,
        block_fp8_dequant_ms,
        block_fp8_dequantized,
        w4a16_packed: pack_w4a16_q_b || pack_w4a16_o_proj,
        w8a16_q_a_packed: pack_w8a16_q_a,
        w8a16_q_b_packed: pack_w8a16_q_b,
        w8a16_o_packed: pack_w8a16_o_proj,
        source_released: pack_w8a16_q_a || pack_w8a16_q_b || pack_w8a16_o_proj,
    })
}

fn load_coordinator_block_fp8_scale_bytes(
    catalog: &TensorCatalog,
    weight: &TensorInfo,
) -> Result<Vec<u8>> {
    let base = weight.name.strip_suffix(".weight").with_context(|| {
        format!(
            "block-FP8 coordinator tensor {} is not a weight",
            weight.name
        )
    })?;
    let scale_name = format!("{base}.weight_scale_inv");
    let scale = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.name == scale_name)
        .with_context(|| {
            format!(
                "block-FP8 coordinator tensor {} is missing {scale_name}",
                weight.name
            )
        })?;
    let expected_shape = vec![
        weight.shape[0].div_ceil(COORDINATOR_BLOCK_FP8_BLOCK_ROWS),
        weight.shape[1].div_ceil(COORDINATOR_BLOCK_FP8_BLOCK_COLUMNS),
    ];
    anyhow::ensure!(
        scale.dtype == DType::F32 && scale.shape == expected_shape,
        "block-FP8 coordinator scale {scale_name} must be F32 {:?}, got {:?} {:?}",
        expected_shape,
        scale.dtype,
        scale.shape
    );
    let scale_bytes: usize = scale.byte_length.try_into().with_context(|| {
        format!("block-FP8 coordinator scale {scale_name} byte length overflows usize")
    })?;
    let mut bytes = vec![0_u8; scale_bytes];
    let summary = read_tensor_bytes_into(catalog, &scale_name, &mut bytes)
        .with_context(|| format!("reading block-FP8 coordinator scale {scale_name}"))?;
    validate_coordinator_resident_tensor_summary(scale, &summary)?;
    Ok(bytes)
}

fn coordinator_resident_preload_plan_for_tensors(
    catalog: &TensorCatalog,
    status: &'static str,
    loaded_bytes: u64,
) -> RealFullCoordinatorResidentPreloadPlan {
    let tensors = coordinator_resident_tensors(catalog);
    let mut role_counts = BTreeMap::new();
    let mut role_bytes = BTreeMap::new();
    let mut bf16_tensors = 0_usize;
    let mut non_bf16_tensors = 0_usize;
    let mut selected_bytes = 0_u64;
    let mut sample_resident_keys = Vec::new();
    for tensor in &tensors {
        let role = coordinator_resident_role_label(&tensor.role).to_owned();
        *role_counts.entry(role.clone()).or_insert(0) += 1;
        *role_bytes.entry(role).or_insert(0) += tensor.byte_length;
        selected_bytes += tensor.byte_length;
        if tensor.dtype == DType::Bf16 {
            bf16_tensors += 1;
        } else {
            non_bf16_tensors += 1;
        }
        if sample_resident_keys.len() < COORDINATOR_RESIDENT_SAMPLE_LIMIT {
            sample_resident_keys.push(tensor.name.clone());
        }
    }
    let selected_tensor_count_from_roles = role_counts.values().copied().sum::<usize>();
    let selected_tensor_bytes_from_roles = role_bytes.values().copied().sum::<u64>();
    let missing_required_roles = REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS
        .iter()
        .filter(|role| role_counts.get(**role).copied().unwrap_or_default() == 0)
        .map(|role| (*role).to_owned())
        .collect::<Vec<_>>();

    let skipped_routed_expert_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
        .count();
    let skipped_routed_expert_bytes = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::RoutedExpert)
        .map(|tensor| tensor.byte_length)
        .sum();
    let skipped_quantization_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Quantization || tensor.is_quantization_metadata)
        .count();
    let skipped_quantization_bytes = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Quantization || tensor.is_quantization_metadata)
        .map(|tensor| tensor.byte_length)
        .sum();
    let skipped_mtp_tensors = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Mtp && !coordinator_resident_tensor(tensor))
        .count();
    let skipped_mtp_bytes = catalog
        .tensors
        .iter()
        .filter(|tensor| tensor.role == TensorRole::Mtp && !coordinator_resident_tensor(tensor))
        .map(|tensor| tensor.byte_length)
        .sum();

    RealFullCoordinatorResidentPreloadPlan {
        status,
        scope: COORDINATOR_RESIDENT_PRELOAD_SCOPE,
        startup_required: true,
        selected_tensor_count: tensors.len(),
        selected_tensor_bytes: selected_bytes,
        loaded_tensor_bytes: loaded_bytes,
        bf16_tensor_count: bf16_tensors,
        non_bf16_tensor_count: non_bf16_tensors,
        role_counts,
        role_bytes,
        required_role_count: REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS.len(),
        required_roles_present: REQUIRED_COORDINATOR_RESIDENT_ROLE_LABELS.len()
            - missing_required_roles.len(),
        missing_required_roles,
        selected_tensor_count_matches_roles: selected_tensor_count_from_roles == tensors.len(),
        selected_tensor_bytes_matches_roles: selected_tensor_bytes_from_roles == selected_bytes,
        skipped_routed_expert_tensors,
        skipped_routed_expert_bytes,
        skipped_quantization_tensors,
        skipped_quantization_bytes,
        skipped_mtp_tensors,
        skipped_mtp_bytes,
        sample_resident_keys,
        uses_named_resident_buffers: true,
    }
}

fn validate_coordinator_resident_tensor_summary(
    tensor: &TensorInfo,
    summary: &LoadedTensorSummary,
) -> Result<()> {
    if summary.tensor_name != tensor.name {
        anyhow::bail!(
            "coordinator resident tensor {} staged the wrong tensor {}",
            tensor.name,
            summary.tensor_name
        );
    }
    if summary.dtype != tensor.dtype {
        anyhow::bail!(
            "coordinator resident tensor {} dtype mismatch while staging: read {:?}, catalog {:?}",
            tensor.name,
            summary.dtype,
            tensor.dtype
        );
    }
    if summary.shape != tensor.shape {
        anyhow::bail!(
            "coordinator resident tensor {} shape mismatch while staging: read {:?}, catalog {:?}",
            tensor.name,
            summary.shape,
            tensor.shape
        );
    }
    if summary.role != tensor.role {
        anyhow::bail!(
            "coordinator resident tensor {} role mismatch while staging: read {:?}, catalog {:?}",
            tensor.name,
            summary.role,
            tensor.role
        );
    }
    if summary.layer_id != tensor.layer_id || summary.expert_id != tensor.expert_id {
        anyhow::bail!(
            "coordinator resident tensor {} layer/expert mismatch while staging: read layer={:?} expert={:?}, catalog layer={:?} expert={:?}",
            tensor.name,
            summary.layer_id,
            summary.expert_id,
            tensor.layer_id,
            tensor.expert_id
        );
    }
    if summary.byte_offset != tensor.byte_offset {
        anyhow::bail!(
            "coordinator resident tensor {} byte offset mismatch while staging: read {}, catalog {}",
            tensor.name,
            summary.byte_offset,
            tensor.byte_offset
        );
    }
    if summary.bytes_requested != tensor.byte_length || summary.bytes_read != tensor.byte_length {
        anyhow::bail!(
            "coordinator resident tensor {} byte count mismatch while staging: requested {} read {}, catalog {}",
            tensor.name,
            summary.bytes_requested,
            summary.bytes_read,
            tensor.byte_length
        );
    }
    Ok(())
}

fn coordinator_resident_tensors(catalog: &TensorCatalog) -> Vec<&TensorInfo> {
    catalog
        .tensors
        .iter()
        .filter(|tensor| coordinator_resident_tensor(tensor))
        .collect()
}

fn coordinator_resident_tensor(tensor: &TensorInfo) -> bool {
    coordinator_resident_role(&tensor.role)
        && !tensor.is_quantization_metadata
        && (tensor.role != TensorRole::Mtp || coordinator_mtp_residency_enabled())
        && !(tensor.role == TensorRole::Mtp && tensor.name.contains(".mlp.experts."))
}

fn coordinator_mtp_residency_enabled() -> bool {
    if let Some(include) = env::var(COORDINATOR_INCLUDE_MTP_LAYER_ENV)
        .ok()
        .and_then(|value| parse_bool_env_value(&value))
    {
        return include;
    }
    let mtp = env::var(REAL_FULL_MTP_ENV)
        .ok()
        .and_then(|value| parse_bool_env_value(&value));
    let probe = env::var(REAL_FULL_MTP_PROBE_ENV)
        .ok()
        .and_then(|value| parse_bool_env_value(&value))
        .unwrap_or(false);
    mtp.unwrap_or(true) || probe
}

fn parse_bool_env_value(value: &str) -> Option<bool> {
    match value.trim() {
        "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
        _ => None,
    }
}

fn coordinator_resident_role(role: &TensorRole) -> bool {
    matches!(
        role,
        TensorRole::Embedding
            | TensorRole::LmHead
            | TensorRole::Attention
            | TensorRole::Router
            | TensorRole::Norm
            | TensorRole::DenseMlp
            | TensorRole::SharedExpert
            | TensorRole::Mtp
    )
}

fn coordinator_resident_role_label(role: &TensorRole) -> &'static str {
    match role {
        TensorRole::Embedding => "Embedding",
        TensorRole::LmHead => "LmHead",
        TensorRole::Attention => "Attention",
        TensorRole::Router => "Router",
        TensorRole::Norm => "Norm",
        TensorRole::DenseMlp => "DenseMlp",
        TensorRole::SharedExpert => "SharedExpert",
        TensorRole::Mtp => "Mtp",
        _ => "Other",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        preload_real_full_coordinator_resident_weights,
        real_full_coordinator_resident_preload_plan, validate_coordinator_resident_tensor_summary,
    };
    use crate::commands::real_full::coordinator_kernels::{
        coordinator_cuda_reference_kernels_enabled, resident_weight_is_preloaded,
    };
    use crate::commands::real_full::tests::fixture::full_catalog;
    use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, DEFAULT_MODEL_ID};
    use glmrt_loader::LoadedTensorSummary;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn coordinator_resident_preload_plan_selects_only_coordinator_weights() {
        let catalog = full_catalog();
        let plan = real_full_coordinator_resident_preload_plan(&catalog);

        assert_eq!(plan.status, "planned");
        assert!(plan.startup_required);
        assert!(plan.uses_named_resident_buffers);
        assert_eq!(plan.loaded_tensor_bytes, 0);
        assert!(plan.selected_tensor_count > 0);
        assert!(plan.bf16_tensor_count > 0);
        assert!(
            plan.non_bf16_tensor_count > 0,
            "F32 router correction bias vectors are coordinator-resident immutable tensors"
        );
        assert_eq!(
            plan.selected_tensor_count,
            plan.bf16_tensor_count + plan.non_bf16_tensor_count
        );
        assert_eq!(
            plan.role_counts.get("Embedding").copied(),
            Some(1),
            "embedding table is coordinator-resident"
        );
        assert_eq!(
            plan.role_counts.get("LmHead").copied(),
            Some(1),
            "LM head is coordinator-resident"
        );
        assert!(plan.role_counts.get("Attention").copied().unwrap_or(0) > 0);
        assert!(plan.role_counts.get("Router").copied().unwrap_or(0) > 0);
        assert!(plan.role_counts.get("Norm").copied().unwrap_or(0) > 0);
        assert!(plan.role_counts.get("DenseMlp").copied().unwrap_or(0) > 0);
        assert_eq!(
            plan.required_roles_present, plan.required_role_count,
            "all coordinator-resident roles should be present in the fixture"
        );
        assert!(plan.missing_required_roles.is_empty());
        assert!(plan.selected_tensor_count_matches_roles);
        assert!(plan.selected_tensor_bytes_matches_roles);
        assert_eq!(plan.role_counts.get("RoutedExpert"), None);
        assert!(plan.skipped_routed_expert_tensors > 0);
        assert!(plan.skipped_quantization_tensors > 0);
        assert!(plan
            .sample_resident_keys
            .iter()
            .any(|name| name == "model.embed_tokens.weight"));
    }

    #[test]
    fn coordinator_resident_preload_plan_reports_missing_required_roles() {
        let mut catalog = full_catalog();
        catalog
            .tensors
            .retain(|tensor| tensor.role != TensorRole::LmHead);
        let plan = real_full_coordinator_resident_preload_plan(&catalog);

        assert!(plan.required_roles_present < plan.required_role_count);
        assert_eq!(plan.missing_required_roles, vec!["LmHead".to_owned()]);
        assert!(plan.selected_tensor_count_matches_roles);
        assert!(plan.selected_tensor_bytes_matches_roles);
    }

    #[test]
    fn coordinator_residency_selects_mtp_envelope_but_not_experts_or_metadata() {
        let mtp_layer = glmrt_core::GLM52_MTP_LAYER_ID as u32;
        let catalog = TensorCatalog {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: "/tmp/glmrt-snapshot".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                TensorInfo {
                    name: "model.layers.78.eh_proj.weight".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::Bf16,
                    shape: vec![1],
                    byte_offset: 0,
                    byte_length: 2,
                    role: TensorRole::Mtp,
                    layer_id: Some(mtp_layer),
                    expert_id: None,
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: "model.layers.78.eh_proj.weight_scale".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::F32,
                    shape: vec![1],
                    byte_offset: 2,
                    byte_length: 4,
                    role: TensorRole::Mtp,
                    layer_id: Some(mtp_layer),
                    expert_id: None,
                    is_quantization_metadata: true,
                },
                TensorInfo {
                    name: "model.layers.78.mlp.experts.0.gate_proj.weight".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::U8,
                    shape: vec![1],
                    byte_offset: 6,
                    byte_length: 1,
                    role: TensorRole::RoutedExpert,
                    layer_id: Some(mtp_layer),
                    expert_id: Some(0),
                    is_quantization_metadata: false,
                },
                TensorInfo {
                    name: "model.layers.78.mlp.experts.1.gate_proj.weight".to_owned(),
                    file: "model.safetensors".to_owned(),
                    dtype: DType::U8,
                    shape: vec![1],
                    byte_offset: 7,
                    byte_length: 1,
                    role: TensorRole::Mtp,
                    layer_id: Some(mtp_layer),
                    expert_id: Some(1),
                    is_quantization_metadata: false,
                },
            ],
        };

        let plan = real_full_coordinator_resident_preload_plan(&catalog);

        assert_eq!(plan.selected_tensor_count, 1);
        assert_eq!(plan.selected_tensor_bytes, 2);
        assert_eq!(plan.role_counts.get("Mtp"), Some(&1));
        assert_eq!(plan.skipped_mtp_tensors, 2);
        assert_eq!(plan.skipped_mtp_bytes, 5);
        assert_eq!(plan.skipped_routed_expert_tensors, 1);
        assert_eq!(plan.skipped_routed_expert_bytes, 1);
    }

    #[test]
    fn coordinator_resident_summary_validation_accepts_matching_catalog_entry() {
        let tensor = sample_tensor_info();
        let summary = sample_tensor_summary();

        validate_coordinator_resident_tensor_summary(&tensor, &summary)
            .expect("matching resident preload summary should validate");
    }

    #[test]
    fn coordinator_resident_summary_validation_rejects_identity_mismatch() {
        let tensor = sample_tensor_info();
        let mut summary = sample_tensor_summary();
        summary.tensor_name = "model.layers.0.self_attn.q_a_proj.weight".to_owned();

        let err = validate_coordinator_resident_tensor_summary(&tensor, &summary)
            .expect_err("wrong tensor name should fail validation");
        assert!(err.to_string().contains("staged the wrong tensor"));
    }

    #[test]
    fn coordinator_resident_summary_validation_rejects_shape_dtype_and_byte_mismatch() {
        let tensor = sample_tensor_info();

        let mut dtype_summary = sample_tensor_summary();
        dtype_summary.dtype = DType::F32;
        let err = validate_coordinator_resident_tensor_summary(&tensor, &dtype_summary)
            .expect_err("wrong dtype should fail validation");
        assert!(err.to_string().contains("dtype mismatch"));

        let mut shape_summary = sample_tensor_summary();
        shape_summary.shape = vec![4, 2];
        let err = validate_coordinator_resident_tensor_summary(&tensor, &shape_summary)
            .expect_err("wrong shape should fail validation");
        assert!(err.to_string().contains("shape mismatch"));

        let mut byte_summary = sample_tensor_summary();
        byte_summary.bytes_read -= 2;
        let err = validate_coordinator_resident_tensor_summary(&tensor, &byte_summary)
            .expect_err("short read should fail validation");
        assert!(err.to_string().contains("byte count mismatch"));
    }

    #[test]
    fn coordinator_resident_startup_preload_uploads_named_cuda_buffers_when_available() {
        let tempdir = tempfile::tempdir().unwrap();
        let (catalog, tensors) = tiny_resident_catalog(tempdir.path());

        let result = preload_real_full_coordinator_resident_weights(&catalog);
        let plan = match result {
            Ok(plan) => plan,
            Err(error) if !coordinator_cuda_reference_kernels_enabled() => {
                eprintln!("skipped: CUDA resident preload unavailable: {error:#}");
                return;
            }
            Err(error) => panic!("CUDA-required resident preload failed: {error:#}"),
        };

        assert_eq!(plan.status, "loaded");
        assert!(plan.startup_required);
        assert!(plan.uses_named_resident_buffers);
        assert_eq!(plan.required_roles_present, plan.required_role_count);
        assert!(plan.missing_required_roles.is_empty());
        assert_eq!(plan.selected_tensor_count, tensors.len());
        assert_eq!(plan.loaded_tensor_bytes, plan.selected_tensor_bytes);
        for tensor in tensors {
            assert!(
                resident_weight_is_preloaded(&tensor.name, tensor.byte_length as usize),
                "resident CUDA buffer {} should be preloaded",
                tensor.name
            );
        }
    }

    fn sample_tensor_info() -> TensorInfo {
        TensorInfo {
            name: "model.layers.0.input_layernorm.weight".to_owned(),
            file: "model-00001-of-00001.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![4],
            byte_offset: 128,
            byte_length: 8,
            role: TensorRole::Norm,
            layer_id: Some(0),
            expert_id: None,
            is_quantization_metadata: false,
        }
    }

    fn sample_tensor_summary() -> LoadedTensorSummary {
        LoadedTensorSummary {
            tensor_name: "model.layers.0.input_layernorm.weight".to_owned(),
            source_path: "/tmp/model-00001-of-00001.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![4],
            role: TensorRole::Norm,
            layer_id: Some(0),
            expert_id: None,
            byte_offset: 128,
            bytes_requested: 8,
            bytes_read: 8,
            elapsed_micros: 10,
            read_gbps: 0.001,
            sha256: String::new(),
        }
    }

    fn tiny_resident_catalog(snapshot_path: &std::path::Path) -> (TensorCatalog, Vec<TensorInfo>) {
        let shard_name = "tiny-resident.safetensors";
        let shard_path = snapshot_path.join(shard_name);
        let mut shard = File::create(&shard_path).expect("create tiny resident shard");
        let specs = [
            (
                "test.resident.embed_tokens.weight",
                TensorRole::Embedding,
                None,
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.lm_head.weight",
                TensorRole::LmHead,
                None,
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.self_attn.q_a_proj.weight",
                TensorRole::Attention,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.mlp.gate.weight",
                TensorRole::Router,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.input_layernorm.weight",
                TensorRole::Norm,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.0.mlp.gate_proj.weight",
                TensorRole::DenseMlp,
                Some(0),
                DType::Bf16,
                8_u64,
            ),
            (
                "test.resident.layers.3.mlp.shared_experts.gate_proj.weight",
                TensorRole::SharedExpert,
                Some(3),
                DType::Bf16,
                8_u64,
            ),
        ];
        let mut offset = 0_u64;
        let mut tensors = Vec::new();
        for (index, (name, role, layer_id, dtype, byte_length)) in specs.iter().enumerate() {
            let bytes = (0..*byte_length)
                .map(|byte| (index as u8).wrapping_mul(17).wrapping_add(byte as u8))
                .collect::<Vec<_>>();
            shard
                .write_all(&bytes)
                .expect("write tiny resident tensor bytes");
            tensors.push(TensorInfo {
                name: (*name).to_owned(),
                file: shard_name.to_owned(),
                dtype: dtype.clone(),
                shape: vec![(*byte_length / 2) as usize],
                byte_offset: offset,
                byte_length: *byte_length,
                role: role.clone(),
                layer_id: *layer_id,
                expert_id: None,
                is_quantization_metadata: false,
            });
            offset += *byte_length;
        }
        let catalog = TensorCatalog {
            model_id: DEFAULT_MODEL_ID.to_owned(),
            snapshot_path: snapshot_path.display().to_string(),
            facts: ModelFacts::default(),
            tensors: tensors.clone(),
        };
        (catalog, tensors)
    }
}
