#![allow(dead_code)]

use std::ffi::c_void;

use anyhow::{Context, Result};
use glmrt_ffi::GlmrtDeviceBuffer;

use super::dflash::{
    Dflash2ResidentWeights, GLM53_DFLASH2_ATTENTION_HEADS, GLM53_DFLASH2_BLOCK_SIZE,
    GLM53_DFLASH2_CONV_GROUP_SIZE, GLM53_DFLASH2_DRAFT_LAYERS, GLM53_DFLASH2_HEAD_DIM,
    GLM53_DFLASH2_HIDDEN_SIZE, GLM53_DFLASH2_INTERMEDIATE_SIZE, GLM53_DFLASH2_KV_HEADS,
};
use super::dspark_kv::DsparkKvStorage;
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

// FlashInfer 0.6.14's cuDNN paged-prefill planner needs about 188 MiB at the
// production C=2 DFlash2 suffix shape and scales further at C=4. Keep one
// bounded 512 MiB workspace per captured executor so every qualified
// concurrency is planned once at startup and hot replay remains allocation-free.
pub(super) const DFLASH2_BODY_WORKSPACE_BYTES: usize = 512 * 1024 * 1024;
const DFLASH2_ROPE_THETA: f64 = 1_000_000.0;

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2BodyConfig {
    pub(super) active_requests: usize,
    pub(super) query_rows_per_request: usize,
    pub(super) total_pages: usize,
    pub(super) page_size: usize,
    pub(super) max_pages_per_request: usize,
    pub(super) planning_pages_per_request: usize,
    pub(super) fixed_split_pages: usize,
    pub(super) kv_storage: DsparkKvStorage,
    pub(super) seed: i64,
    pub(super) initialize_input: bool,
    pub(super) initialize_kv: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2BodyBuffers {
    pub(super) input: GlmrtDeviceBuffer,
    pub(super) output: GlmrtDeviceBuffer,
    pub(super) reference_output: GlmrtDeviceBuffer,
    pub(super) hidden_attention: GlmrtDeviceBuffer,
    pub(super) hidden_mlp: GlmrtDeviceBuffer,
    pub(super) normalized: GlmrtDeviceBuffer,
    pub(super) qkv: GlmrtDeviceBuffer,
    pub(super) q: GlmrtDeviceBuffer,
    pub(super) attention: GlmrtDeviceBuffer,
    pub(super) delta: GlmrtDeviceBuffer,
    pub(super) gate_up: GlmrtDeviceBuffer,
    pub(super) activation: GlmrtDeviceBuffer,
    pub(super) conv_dynamic: GlmrtDeviceBuffer,
    pub(super) conv_output: GlmrtDeviceBuffer,
    pub(super) k_cache: GlmrtDeviceBuffer,
    pub(super) v_cache: GlmrtDeviceBuffer,
    pub(super) workspace: GlmrtDeviceBuffer,
    pub(super) query_lengths: GlmrtDeviceBuffer,
    pub(super) kv_lengths: GlmrtDeviceBuffer,
    pub(super) query_positions: GlmrtDeviceBuffer,
    pub(super) block_tables: GlmrtDeviceBuffer,
    pub(super) query_offsets: GlmrtDeviceBuffer,
    pub(super) output_offsets: GlmrtDeviceBuffer,
    pub(super) query_indptr: GlmrtDeviceBuffer,
    pub(super) kv_indptr: GlmrtDeviceBuffer,
    pub(super) page_indices: GlmrtDeviceBuffer,
    pub(super) last_page_len: GlmrtDeviceBuffer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Dflash2BodyBufferPlan {
    pub(super) name: &'static str,
    pub(super) bytes: usize,
}

pub(super) fn dflash2_body_buffer_plan(
    config: Dflash2BodyConfig,
) -> Result<Vec<Dflash2BodyBufferPlan>> {
    validate_config(config)?;
    let rows = config
        .active_requests
        .checked_mul(config.query_rows_per_request)
        .context("DFlash2 body row count overflow")?;
    let bf16 = std::mem::size_of::<u16>();
    let i32_bytes = std::mem::size_of::<i32>();
    let i64_bytes = std::mem::size_of::<i64>();
    let bytes = |rows: usize, width: usize, element_bytes: usize| -> Result<usize> {
        rows.checked_mul(width)
            .and_then(|values| values.checked_mul(element_bytes))
            .context("DFlash2 body buffer byte count overflow")
    };
    let hidden = bytes(rows, GLM53_DFLASH2_HIDDEN_SIZE, bf16)?;
    let q_width = GLM53_DFLASH2_ATTENTION_HEADS * GLM53_DFLASH2_HEAD_DIM;
    let kv_width = GLM53_DFLASH2_KV_HEADS * GLM53_DFLASH2_HEAD_DIM;
    let cache_elements = GLM53_DFLASH2_DRAFT_LAYERS
        .checked_mul(config.total_pages)
        .and_then(|values| values.checked_mul(GLM53_DFLASH2_KV_HEADS))
        .and_then(|values| values.checked_mul(config.page_size))
        .and_then(|values| values.checked_mul(GLM53_DFLASH2_HEAD_DIM))
        .context("DFlash2 body KV cache element count overflow")?;
    let one_cache = cache_elements
        .checked_mul(config.kv_storage.element_bytes())
        .context("DFlash2 body KV cache byte count overflow")?;
    let metadata = [
        (
            "query_lengths",
            bytes(config.active_requests, 1, i32_bytes)?,
        ),
        ("kv_lengths", bytes(config.active_requests, 1, i32_bytes)?),
        ("query_positions", bytes(rows, 1, i32_bytes)?),
        (
            "block_tables",
            bytes(
                config.active_requests,
                config.max_pages_per_request,
                i32_bytes,
            )?,
        ),
        (
            "query_offsets",
            bytes(config.active_requests + 1, 1, i64_bytes)?,
        ),
        (
            "output_offsets",
            bytes(config.active_requests + 1, 1, i64_bytes)?,
        ),
        (
            "query_indptr",
            bytes(config.active_requests + 1, 1, i32_bytes)?,
        ),
        (
            "kv_indptr",
            bytes(config.active_requests + 1, 1, i32_bytes)?,
        ),
        ("page_indices", bytes(config.total_pages, 1, i32_bytes)?),
        (
            "last_page_len",
            bytes(config.active_requests, 1, i32_bytes)?,
        ),
    ];
    let mut plan = vec![
        Dflash2BodyBufferPlan {
            name: "input",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "output",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "reference_output",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "hidden_attention",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "hidden_mlp",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "normalized",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "qkv",
            bytes: bytes(rows, q_width + 2 * kv_width, bf16)?,
        },
        Dflash2BodyBufferPlan {
            name: "q",
            bytes: bytes(rows, q_width, bf16)?,
        },
        Dflash2BodyBufferPlan {
            name: "attention",
            bytes: bytes(rows, q_width, bf16)?,
        },
        Dflash2BodyBufferPlan {
            name: "delta",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "gate_up",
            bytes: bytes(rows, 2 * GLM53_DFLASH2_INTERMEDIATE_SIZE, bf16)?,
        },
        Dflash2BodyBufferPlan {
            name: "activation",
            bytes: bytes(rows, GLM53_DFLASH2_INTERMEDIATE_SIZE, bf16)?,
        },
        Dflash2BodyBufferPlan {
            name: "conv_dynamic",
            bytes: bytes(
                rows,
                4 * (GLM53_DFLASH2_HIDDEN_SIZE / GLM53_DFLASH2_CONV_GROUP_SIZE),
                bf16,
            )?,
        },
        Dflash2BodyBufferPlan {
            name: "conv_output",
            bytes: hidden,
        },
        Dflash2BodyBufferPlan {
            name: "k_cache",
            bytes: one_cache,
        },
        Dflash2BodyBufferPlan {
            name: "v_cache",
            bytes: one_cache,
        },
        Dflash2BodyBufferPlan {
            name: "workspace",
            bytes: DFLASH2_BODY_WORKSPACE_BYTES,
        },
    ];
    plan.extend(
        metadata
            .into_iter()
            .map(|(name, bytes)| Dflash2BodyBufferPlan { name, bytes }),
    );
    Ok(plan)
}

pub(super) fn launch_python_dflash2_body(
    cuda_stream: *mut c_void,
    buffers: Dflash2BodyBuffers,
    weights: Dflash2ResidentWeights,
    config: Dflash2BodyConfig,
    function: &str,
) -> Result<()> {
    dflash2_body_buffer_plan(config)?;
    let mut device_buffers = vec![
        python_buffer("input", buffers.input),
        python_buffer("output", buffers.output),
        python_buffer("reference_output", buffers.reference_output),
        python_buffer("hidden_attention", buffers.hidden_attention),
        python_buffer("hidden_mlp", buffers.hidden_mlp),
        python_buffer("normalized", buffers.normalized),
        python_buffer("qkv", buffers.qkv),
        python_buffer("q", buffers.q),
        python_buffer("attention", buffers.attention),
        python_buffer("delta", buffers.delta),
        python_buffer("gate_up", buffers.gate_up),
        python_buffer("activation", buffers.activation),
        python_buffer("conv_dynamic", buffers.conv_dynamic),
        python_buffer("conv_output", buffers.conv_output),
        python_buffer("k_cache", buffers.k_cache),
        python_buffer("v_cache", buffers.v_cache),
        python_buffer("workspace", buffers.workspace),
        python_buffer("query_lengths", buffers.query_lengths),
        python_buffer("kv_lengths", buffers.kv_lengths),
        python_buffer("query_positions", buffers.query_positions),
        python_buffer("block_tables", buffers.block_tables),
        python_buffer("query_offsets", buffers.query_offsets),
        python_buffer("output_offsets", buffers.output_offsets),
        python_buffer("query_indptr", buffers.query_indptr),
        python_buffer("kv_indptr", buffers.kv_indptr),
        python_buffer("page_indices", buffers.page_indices),
        python_buffer("last_page_len", buffers.last_page_len),
        python_buffer("final_norm", weights.final_norm),
    ];
    let mut named_weights = Vec::with_capacity(GLM53_DFLASH2_DRAFT_LAYERS * 12);
    for (layer_id, layer) in weights.layers.iter().enumerate() {
        for (suffix, buffer) in [
            ("input_norm", layer.input_norm),
            ("post_norm", layer.post_norm),
            ("q_norm", layer.q_norm),
            ("k_norm", layer.k_norm),
            ("qkv", layer.qkv),
            ("output", layer.output),
            ("gate_up", layer.gate_up),
            ("down", layer.down),
            ("attention_conv_base", layer.attention_conv_base),
            ("attention_conv_projection", layer.attention_conv_projection),
            ("mlp_conv_base", layer.mlp_conv_base),
            ("mlp_conv_projection", layer.mlp_conv_projection),
        ] {
            named_weights.push((format!("layer_{layer_id}_{suffix}"), buffer));
        }
    }
    device_buffers.extend(
        named_weights
            .iter()
            .map(|(name, buffer)| python_buffer(name, *buffer)),
    );
    let kwargs = [
        ("layers", PythonKernelArg::Usize(GLM53_DFLASH2_DRAFT_LAYERS)),
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        (
            "query_rows",
            PythonKernelArg::Usize(config.query_rows_per_request),
        ),
        ("total_pages", PythonKernelArg::Usize(config.total_pages)),
        ("page_size", PythonKernelArg::Usize(config.page_size)),
        (
            "max_pages_per_request",
            PythonKernelArg::Usize(config.max_pages_per_request),
        ),
        (
            "planning_pages_per_request",
            PythonKernelArg::Usize(config.planning_pages_per_request),
        ),
        (
            "fixed_split_pages",
            PythonKernelArg::Usize(config.fixed_split_pages),
        ),
        (
            "hidden_size",
            PythonKernelArg::Usize(GLM53_DFLASH2_HIDDEN_SIZE),
        ),
        (
            "intermediate_size",
            PythonKernelArg::Usize(GLM53_DFLASH2_INTERMEDIATE_SIZE),
        ),
        (
            "heads",
            PythonKernelArg::Usize(GLM53_DFLASH2_ATTENTION_HEADS),
        ),
        ("kv_heads", PythonKernelArg::Usize(GLM53_DFLASH2_KV_HEADS)),
        ("head_dim", PythonKernelArg::Usize(GLM53_DFLASH2_HEAD_DIM)),
        ("rope_theta", PythonKernelArg::F64(DFLASH2_ROPE_THETA)),
        (
            "conv_group_size",
            PythonKernelArg::Usize(GLM53_DFLASH2_CONV_GROUP_SIZE),
        ),
        (
            "sliding_window",
            PythonKernelArg::Usize(super::dflash::GLM53_DFLASH2_SLIDING_WINDOW),
        ),
        ("seed", PythonKernelArg::I64(config.seed)),
        (
            "initialize_input",
            PythonKernelArg::Bool(config.initialize_input),
        ),
        ("initialize_kv", PythonKernelArg::Bool(config.initialize_kv)),
        (
            "cache_dtype",
            PythonKernelArg::Str(config.kv_storage.label()),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "dspark_body_capture",
        function,
        cuda_stream,
        buffers: &device_buffers,
        kwargs: &kwargs,
    })
}

fn validate_config(config: Dflash2BodyConfig) -> Result<()> {
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4),
        "DFlash2 body active request bucket must be 1, 2, or 4"
    );
    anyhow::ensure!(
        (2..=GLM53_DFLASH2_BLOCK_SIZE).contains(&config.query_rows_per_request),
        "DFlash2 body query rows per request must be in 2..={GLM53_DFLASH2_BLOCK_SIZE}"
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "DFlash2 body page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        config.total_pages >= config.active_requests && config.max_pages_per_request > 0,
        "DFlash2 body page counts are invalid"
    );
    anyhow::ensure!(
        (1..=config.max_pages_per_request).contains(&config.planning_pages_per_request),
        "DFlash2 body planning pages must fit the page-table capacity"
    );
    anyhow::ensure!(
        config.fixed_split_pages == 0
            || (config.active_requests == 1 && config.fixed_split_pages == 2),
        "DFlash2 fixed split pages must be zero or the qualified two-page C1 split"
    );
    // A live C=2/C=4 suffix aliases the C=1 engine's larger shared physical
    // pool. Its block table only needs to address the pages used by the
    // active requests; unused physical pages do not enlarge that table.
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config(active_requests: usize, storage: DsparkKvStorage) -> Dflash2BodyConfig {
        Dflash2BodyConfig {
            active_requests,
            query_rows_per_request: GLM53_DFLASH2_BLOCK_SIZE,
            total_pages: active_requests * 33,
            page_size: 64,
            max_pages_per_request: 33,
            planning_pages_per_request: 33,
            fixed_split_pages: 0,
            kv_storage: storage,
            seed: 1,
            initialize_input: true,
            initialize_kv: true,
        }
    }

    #[test]
    fn plans_gqa_dynamic_conv_body_without_expanding_kv_heads() {
        let mut narrow_config = config(1, DsparkKvStorage::Bf16);
        narrow_config.query_rows_per_request = 4;
        let narrow = dflash2_body_buffer_plan(narrow_config).unwrap();
        let bf16 = dflash2_body_buffer_plan(config(1, DsparkKvStorage::Bf16)).unwrap();
        let fp8 = dflash2_body_buffer_plan(config(1, DsparkKvStorage::Fp8)).unwrap();
        let bytes = |plan: &[Dflash2BodyBufferPlan], name: &str| {
            plan.iter().find(|item| item.name == name).unwrap().bytes
        };
        assert_eq!(bytes(&bf16, "qkv"), 8 * 10_240 * 2);
        assert_eq!(bytes(&narrow, "qkv"), 4 * 10_240 * 2);
        assert_eq!(bytes(&bf16, "conv_dynamic"), 8 * 1_536 * 2);
        assert_eq!(bytes(&bf16, "k_cache"), 6 * 33 * 8 * 64 * 128 * 2);
        assert_eq!(bytes(&fp8, "k_cache") * 2, bytes(&bf16, "k_cache"));
        assert_ne!(
            bytes(&bf16, "k_cache"),
            6 * 33 * 64 * 64 * 128 * 2,
            "DFlash2 must keep eight GQA KV heads rather than expanding to Q heads"
        );
    }

    #[test]
    fn admits_a_larger_shared_physical_pool_than_the_active_block_tables() {
        let mut shared = config(2, DsparkKvStorage::Bf16);
        shared.total_pages = 4 * 33;
        let plan = dflash2_body_buffer_plan(shared).unwrap();
        let bytes = |name: &str| plan.iter().find(|item| item.name == name).unwrap().bytes;
        assert_eq!(bytes("k_cache"), 6 * 4 * 33 * 8 * 64 * 128 * 2);
        assert_eq!(bytes("block_tables"), 2 * 33 * 4);
    }

    #[test]
    fn admits_only_qualified_page_specific_split_plans() {
        let mut page_bucket = config(1, DsparkKvStorage::Bf16);
        page_bucket.planning_pages_per_request = 17;
        page_bucket.fixed_split_pages = 2;
        dflash2_body_buffer_plan(page_bucket).unwrap();

        let mut invalid_pages = page_bucket;
        invalid_pages.planning_pages_per_request = 34;
        assert!(dflash2_body_buffer_plan(invalid_pages).is_err());

        let mut invalid_concurrency = config(2, DsparkKvStorage::Bf16);
        invalid_concurrency.fixed_split_pages = 2;
        assert!(dflash2_body_buffer_plan(invalid_concurrency).is_err());
    }
}
