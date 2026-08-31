#![allow(dead_code)]

use std::ffi::c_void;

use anyhow::{Context, Result};
use glmrt_ffi::GlmrtDeviceBuffer;

use super::coordinator_kernels::device_buffer_byte_view;
use super::dflash::{
    Dflash2ResidentWeights, GLM53_DFLASH2_ATTENTION_HEADS, GLM53_DFLASH2_DRAFT_LAYERS,
    GLM53_DFLASH2_HEAD_DIM, GLM53_DFLASH2_HIDDEN_SIZE, GLM53_DFLASH2_KV_HEADS,
};
use super::dspark_kv::DsparkKvStorage;
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

const DFLASH2_TARGET_FEATURES: usize = GLM53_DFLASH2_DRAFT_LAYERS * GLM53_DFLASH2_HIDDEN_SIZE;
const DFLASH2_KV_WIDTH: usize = GLM53_DFLASH2_KV_HEADS * GLM53_DFLASH2_HEAD_DIM;
const DFLASH2_Q_WIDTH: usize = GLM53_DFLASH2_ATTENTION_HEADS * GLM53_DFLASH2_HEAD_DIM;
const DFLASH2_ROPE_THETA: f64 = 1_000_000.0;

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2UpdateLayerResidentWeights {
    pub(super) k_norm: GlmrtDeviceBuffer,
    pub(super) kv: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2UpdateResidentWeights {
    pub(super) target_fusion: GlmrtDeviceBuffer,
    pub(super) hidden_norm: GlmrtDeviceBuffer,
    pub(super) layers: [Dflash2UpdateLayerResidentWeights; GLM53_DFLASH2_DRAFT_LAYERS],
}

pub(super) fn dflash2_update_resident_weights(
    weights: Dflash2ResidentWeights,
) -> Result<Dflash2UpdateResidentWeights> {
    let q_bytes = DFLASH2_Q_WIDTH
        .checked_mul(GLM53_DFLASH2_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DFlash2 Q projection byte count overflow")?;
    let kv_bytes = (2 * DFLASH2_KV_WIDTH)
        .checked_mul(GLM53_DFLASH2_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("DFlash2 KV projection byte count overflow")?;
    let layers = weights
        .layers
        .iter()
        .map(|layer| {
            Ok(Dflash2UpdateLayerResidentWeights {
                k_norm: layer.k_norm,
                kv: device_buffer_byte_view(
                    layer.qkv,
                    q_bytes,
                    kv_bytes,
                    "DFlash2 fused QKV KV suffix",
                )
                .context("slicing DFlash2 fused QKV resident into its KV update view")?,
            })
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("DFlash2 update layer count changed"))?;
    Ok(Dflash2UpdateResidentWeights {
        target_fusion: weights.target_fusion,
        hidden_norm: weights.hidden_norm,
        layers,
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2UpdateConfig {
    pub(super) rows: usize,
    pub(super) active_requests: usize,
    pub(super) total_pages: usize,
    pub(super) page_size: usize,
    pub(super) max_pages_per_request: usize,
    pub(super) kv_storage: DsparkKvStorage,
    pub(super) seed: i64,
    pub(super) initialize_target_hidden: bool,
    pub(super) initialize_kv: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2UpdateBuffers {
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Dflash2UpdateBufferPlan {
    pub(super) name: &'static str,
    pub(super) bytes: usize,
}

pub(super) fn dflash2_update_buffer_plan(
    config: Dflash2UpdateConfig,
) -> Result<Vec<Dflash2UpdateBufferPlan>> {
    validate_config(config)?;
    let bf16 = std::mem::size_of::<u16>();
    let i32_bytes = std::mem::size_of::<i32>();
    let bytes = |rows: usize, width: usize, element_bytes: usize| -> Result<usize> {
        rows.checked_mul(width)
            .and_then(|values| values.checked_mul(element_bytes))
            .context("DFlash2 update buffer byte count overflow")
    };
    let hidden = bytes(config.rows, GLM53_DFLASH2_HIDDEN_SIZE, bf16)?;
    let output = bytes(
        GLM53_DFLASH2_DRAFT_LAYERS * config.rows,
        DFLASH2_KV_WIDTH,
        bf16,
    )?;
    let cache_elements = GLM53_DFLASH2_DRAFT_LAYERS
        .checked_mul(config.total_pages)
        .and_then(|values| values.checked_mul(GLM53_DFLASH2_KV_HEADS))
        .and_then(|values| values.checked_mul(config.page_size))
        .and_then(|values| values.checked_mul(GLM53_DFLASH2_HEAD_DIM))
        .context("DFlash2 update KV cache element count overflow")?;
    let one_cache = cache_elements
        .checked_mul(config.kv_storage.element_bytes())
        .context("DFlash2 update KV cache byte count overflow")?;
    Ok(vec![
        Dflash2UpdateBufferPlan {
            name: "target_hidden",
            bytes: bytes(config.rows, DFLASH2_TARGET_FEATURES, bf16)?,
        },
        Dflash2UpdateBufferPlan {
            name: "fusion_output",
            bytes: hidden,
        },
        Dflash2UpdateBufferPlan {
            name: "fused_hidden",
            bytes: hidden,
        },
        Dflash2UpdateBufferPlan {
            name: "projected_kv",
            bytes: bytes(config.rows, 2 * DFLASH2_KV_WIDTH, bf16)?,
        },
        Dflash2UpdateBufferPlan {
            name: "key_output",
            bytes: output,
        },
        Dflash2UpdateBufferPlan {
            name: "value_output",
            bytes: output,
        },
        Dflash2UpdateBufferPlan {
            name: "reference_fused_hidden",
            bytes: hidden,
        },
        Dflash2UpdateBufferPlan {
            name: "reference_key_output",
            bytes: output,
        },
        Dflash2UpdateBufferPlan {
            name: "reference_value_output",
            bytes: output,
        },
        Dflash2UpdateBufferPlan {
            name: "eager_fused_hidden",
            bytes: hidden,
        },
        Dflash2UpdateBufferPlan {
            name: "eager_key_output",
            bytes: output,
        },
        Dflash2UpdateBufferPlan {
            name: "eager_value_output",
            bytes: output,
        },
        Dflash2UpdateBufferPlan {
            name: "k_cache",
            bytes: one_cache,
        },
        Dflash2UpdateBufferPlan {
            name: "v_cache",
            bytes: one_cache,
        },
        Dflash2UpdateBufferPlan {
            name: "row_request_ids",
            bytes: bytes(config.rows, 1, i32_bytes)?,
        },
        Dflash2UpdateBufferPlan {
            name: "row_positions",
            bytes: bytes(config.rows, 1, i32_bytes)?,
        },
        Dflash2UpdateBufferPlan {
            name: "row_cache_positions",
            bytes: bytes(config.rows, 1, i32_bytes)?,
        },
        Dflash2UpdateBufferPlan {
            name: "block_tables",
            bytes: bytes(
                config.active_requests,
                config.max_pages_per_request,
                i32_bytes,
            )?,
        },
    ])
}

pub(super) fn launch_python_dflash2_update(
    cuda_stream: *mut c_void,
    buffers: Dflash2UpdateBuffers,
    weights: Dflash2UpdateResidentWeights,
    config: Dflash2UpdateConfig,
    function: &str,
) -> Result<()> {
    dflash2_update_buffer_plan(config)?;
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
    let mut named_weights = Vec::with_capacity(GLM53_DFLASH2_DRAFT_LAYERS * 2);
    for (layer_id, layer) in weights.layers.iter().enumerate() {
        named_weights.push((format!("layer_{layer_id}_k_norm"), layer.k_norm));
        named_weights.push((format!("layer_{layer_id}_kv"), layer.kv));
    }
    device_buffers.extend(
        named_weights
            .iter()
            .map(|(name, buffer)| python_buffer(name, *buffer)),
    );
    let kwargs = [
        ("rows", PythonKernelArg::Usize(config.rows)),
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        ("layers", PythonKernelArg::Usize(GLM53_DFLASH2_DRAFT_LAYERS)),
        (
            "hidden_size",
            PythonKernelArg::Usize(GLM53_DFLASH2_HIDDEN_SIZE),
        ),
        (
            "target_features",
            PythonKernelArg::Usize(DFLASH2_TARGET_FEATURES),
        ),
        ("heads", PythonKernelArg::Usize(GLM53_DFLASH2_KV_HEADS)),
        ("head_dim", PythonKernelArg::Usize(GLM53_DFLASH2_HEAD_DIM)),
        ("rope_theta", PythonKernelArg::F64(DFLASH2_ROPE_THETA)),
        ("total_pages", PythonKernelArg::Usize(config.total_pages)),
        ("page_size", PythonKernelArg::Usize(config.page_size)),
        (
            "max_pages_per_request",
            PythonKernelArg::Usize(config.max_pages_per_request),
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

fn validate_config(config: Dflash2UpdateConfig) -> Result<()> {
    anyhow::ensure!(
        config.rows <= 1_024
            && (config.rows.is_power_of_two() || (config.active_requests == 1 && config.rows <= 8)),
        "DFlash2 update rows must be a C1 exact-small or power-of-two bucket no larger than 1024"
    );
    anyhow::ensure!(
        matches!(config.active_requests, 1 | 2 | 4) && config.rows >= config.active_requests,
        "DFlash2 update active request/row geometry is invalid"
    );
    anyhow::ensure!(
        matches!(config.page_size, 16 | 32 | 64 | 128),
        "DFlash2 update page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        config.total_pages >= config.active_requests && config.max_pages_per_request > 0,
        "DFlash2 update page counts are invalid"
    );
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

    fn config(storage: DsparkKvStorage) -> Dflash2UpdateConfig {
        Dflash2UpdateConfig {
            rows: 8,
            active_requests: 1,
            total_pages: 33,
            page_size: 64,
            max_pages_per_request: 33,
            kv_storage: storage,
            seed: 1,
            initialize_target_hidden: true,
            initialize_kv: true,
        }
    }

    #[test]
    fn plans_six_tap_gqa_update_buffers() {
        let bf16 = dflash2_update_buffer_plan(config(DsparkKvStorage::Bf16)).unwrap();
        let fp8 = dflash2_update_buffer_plan(config(DsparkKvStorage::Fp8)).unwrap();
        let bytes = |plan: &[Dflash2UpdateBufferPlan], name: &str| {
            plan.iter().find(|item| item.name == name).unwrap().bytes
        };
        assert_eq!(bytes(&bf16, "target_hidden"), 8 * 6 * 6_144 * 2);
        assert_eq!(bytes(&bf16, "projected_kv"), 8 * 2 * 1_024 * 2);
        assert_eq!(bytes(&bf16, "key_output"), 6 * 8 * 1_024 * 2);
        assert_eq!(bytes(&fp8, "k_cache") * 2, bytes(&bf16, "k_cache"));
    }

    #[test]
    fn admits_a_larger_shared_physical_pool_than_the_active_update_tables() {
        let mut shared = config(DsparkKvStorage::Bf16);
        shared.active_requests = 2;
        shared.total_pages = 4 * 33;
        let plan = dflash2_update_buffer_plan(shared).unwrap();
        let bytes = |name: &str| plan.iter().find(|item| item.name == name).unwrap().bytes;
        assert_eq!(bytes("k_cache"), 6 * 4 * 33 * 8 * 64 * 128 * 2);
        assert_eq!(bytes("block_tables"), 2 * 33 * 4);
    }

    #[test]
    fn admits_exact_c1_decode_updates_but_keeps_c2_c4_power_of_two() {
        for rows in 1..=8 {
            let mut exact = config(DsparkKvStorage::Bf16);
            exact.rows = rows;
            validate_config(exact).unwrap();
        }
        let mut c2_non_power = config(DsparkKvStorage::Bf16);
        c2_non_power.active_requests = 2;
        c2_non_power.rows = 3;
        assert!(validate_config(c2_non_power).is_err());
    }
}
