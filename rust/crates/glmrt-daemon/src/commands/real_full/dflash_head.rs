#![allow(dead_code)]

use std::env;
use std::ffi::c_void;

use anyhow::{Context, Result};
use glmrt_ffi::GlmrtDeviceBuffer;

use super::dflash::{
    Dflash2ResidentWeights, GLM53_DFLASH2_HIDDEN_SIZE, GLM53_DFLASH2_MAX_DRAFTS,
    GLM53_DFLASH2_SELECTOR_RANK, GLM53_DFLASH2_SELECTOR_TOP_K, GLM53_DFLASH2_VOCAB_SIZE,
};
use crate::python_graph_capture::{
    launch_python_graph_capture, PythonDeviceBufferArg, PythonGraphCaptureLaunch, PythonKernelArg,
};

const DFLASH2_RADIX_TOPK_ROW_STATES_BYTES: usize = 1024 * 1024;
const DFLASH2_TOPK_BACKEND_ENV: &str = "GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND";

pub(super) fn dflash2_topk_backend() -> Result<String> {
    let backend = env::var(DFLASH2_TOPK_BACKEND_ENV)
        .unwrap_or_else(|_| "torch".to_owned())
        .trim()
        .to_ascii_lowercase();
    anyhow::ensure!(
        matches!(backend.as_str(), "torch" | "flashinfer" | "flashinfer-dsa"),
        "{DFLASH2_TOPK_BACKEND_ENV} must be torch, flashinfer, or flashinfer-dsa, got {backend:?}"
    );
    Ok(backend)
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2HeadResidentWeights {
    pub(super) lm_head: GlmrtDeviceBuffer,
    pub(super) hidden_projection: GlmrtDeviceBuffer,
    pub(super) predecessor_codebook: GlmrtDeviceBuffer,
    pub(super) successor_codebook: GlmrtDeviceBuffer,
}

impl From<Dflash2ResidentWeights> for Dflash2HeadResidentWeights {
    fn from(weights: Dflash2ResidentWeights) -> Self {
        Self {
            lm_head: weights.target_lm_head,
            hidden_projection: weights.selector_hidden_projection,
            predecessor_codebook: weights.selector_predecessor,
            successor_codebook: weights.selector_successor,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2HeadConfig {
    pub(super) active_requests: usize,
    pub(super) proposal_tokens_per_request: usize,
    pub(super) seed: i64,
    pub(super) initialize_hidden: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2HeadBuffers {
    pub(super) hidden: GlmrtDeviceBuffer,
    pub(super) hidden_position_major: GlmrtDeviceBuffer,
    pub(super) logits: GlmrtDeviceBuffer,
    pub(super) unary: GlmrtDeviceBuffer,
    pub(super) candidates: GlmrtDeviceBuffer,
    pub(super) radix_candidates: GlmrtDeviceBuffer,
    pub(super) radix_row_states: GlmrtDeviceBuffer,
    pub(super) projected_hidden: GlmrtDeviceBuffer,
    pub(super) token_steps: GlmrtDeviceBuffer,
    pub(super) anchor_tokens: GlmrtDeviceBuffer,
    pub(super) output_tokens: GlmrtDeviceBuffer,
    pub(super) reference_tokens: GlmrtDeviceBuffer,
    pub(super) eager_tokens: GlmrtDeviceBuffer,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Dflash2HeadBufferPlan {
    pub(super) name: &'static str,
    pub(super) bytes: usize,
}

pub(super) fn dflash2_head_buffer_plan(
    active_requests: usize,
    proposal_tokens_per_request: usize,
) -> Result<Vec<Dflash2HeadBufferPlan>> {
    anyhow::ensure!(
        matches!(active_requests, 1 | 2 | 4),
        "DFlash2 head active request bucket must be 1, 2, or 4"
    );
    anyhow::ensure!(
        (1..=GLM53_DFLASH2_MAX_DRAFTS).contains(&proposal_tokens_per_request),
        "DFlash2 head proposal tokens per request must be in 1..={GLM53_DFLASH2_MAX_DRAFTS}"
    );
    let query_rows_per_request = proposal_tokens_per_request + 1;
    let proposals = active_requests
        .checked_mul(proposal_tokens_per_request)
        .context("DFlash2 head proposal row count overflow")?;
    let bf16 = std::mem::size_of::<u16>();
    let i32_bytes = std::mem::size_of::<i32>();
    let i64_bytes = std::mem::size_of::<i64>();
    let bytes = |rows: usize, width: usize, element_bytes: usize| -> Result<usize> {
        rows.checked_mul(width)
            .and_then(|values| values.checked_mul(element_bytes))
            .context("DFlash2 head buffer byte count overflow")
    };
    Ok(vec![
        Dflash2HeadBufferPlan {
            name: "hidden",
            bytes: bytes(
                active_requests * query_rows_per_request,
                GLM53_DFLASH2_HIDDEN_SIZE,
                bf16,
            )?,
        },
        Dflash2HeadBufferPlan {
            name: "hidden_position_major",
            bytes: bytes(proposals, GLM53_DFLASH2_HIDDEN_SIZE, bf16)?,
        },
        Dflash2HeadBufferPlan {
            name: "logits",
            bytes: bytes(proposals, GLM53_DFLASH2_VOCAB_SIZE, bf16)?,
        },
        Dflash2HeadBufferPlan {
            name: "unary",
            bytes: bytes(proposals, GLM53_DFLASH2_SELECTOR_TOP_K, bf16)?,
        },
        Dflash2HeadBufferPlan {
            name: "candidates",
            bytes: bytes(proposals, GLM53_DFLASH2_SELECTOR_TOP_K, i64_bytes)?,
        },
        Dflash2HeadBufferPlan {
            name: "radix_candidates",
            bytes: bytes(proposals, GLM53_DFLASH2_SELECTOR_TOP_K, i32_bytes)?,
        },
        Dflash2HeadBufferPlan {
            name: "radix_row_states",
            bytes: DFLASH2_RADIX_TOPK_ROW_STATES_BYTES,
        },
        Dflash2HeadBufferPlan {
            name: "projected_hidden",
            bytes: bytes(proposals, GLM53_DFLASH2_SELECTOR_RANK, bf16)?,
        },
        Dflash2HeadBufferPlan {
            name: "token_steps",
            bytes: bytes(proposals, 1, i64_bytes)?,
        },
        Dflash2HeadBufferPlan {
            name: "anchor_tokens",
            bytes: bytes(active_requests, 1, i64_bytes)?,
        },
        Dflash2HeadBufferPlan {
            name: "output_tokens",
            bytes: bytes(proposals, 1, i64_bytes)?,
        },
        Dflash2HeadBufferPlan {
            name: "reference_tokens",
            bytes: bytes(proposals, 1, i64_bytes)?,
        },
        Dflash2HeadBufferPlan {
            name: "eager_tokens",
            bytes: bytes(proposals, 1, i64_bytes)?,
        },
    ])
}

pub(super) fn launch_python_dflash2_head(
    cuda_stream: *mut c_void,
    buffers: Dflash2HeadBuffers,
    weights: Dflash2HeadResidentWeights,
    config: Dflash2HeadConfig,
    function: &str,
) -> Result<()> {
    dflash2_head_buffer_plan(config.active_requests, config.proposal_tokens_per_request)?;
    let device_buffers = [
        python_buffer("hidden", buffers.hidden),
        python_buffer("hidden_position_major", buffers.hidden_position_major),
        python_buffer("logits", buffers.logits),
        python_buffer("unary", buffers.unary),
        python_buffer("candidates", buffers.candidates),
        python_buffer("radix_candidates", buffers.radix_candidates),
        python_buffer("radix_row_states", buffers.radix_row_states),
        python_buffer("projected_hidden", buffers.projected_hidden),
        python_buffer("token_steps", buffers.token_steps),
        python_buffer("anchor_tokens", buffers.anchor_tokens),
        python_buffer("output_tokens", buffers.output_tokens),
        python_buffer("reference_tokens", buffers.reference_tokens),
        python_buffer("eager_tokens", buffers.eager_tokens),
        python_buffer("lm_head", weights.lm_head),
        python_buffer("hidden_projection", weights.hidden_projection),
        python_buffer("predecessor_codebook", weights.predecessor_codebook),
        python_buffer("successor_codebook", weights.successor_codebook),
    ];
    let kwargs = [
        (
            "active_requests",
            PythonKernelArg::Usize(config.active_requests),
        ),
        (
            "hidden_rows_per_request",
            PythonKernelArg::Usize(config.proposal_tokens_per_request + 1),
        ),
        (
            "proposal_tokens",
            PythonKernelArg::Usize(config.proposal_tokens_per_request),
        ),
        (
            "hidden_size",
            PythonKernelArg::Usize(GLM53_DFLASH2_HIDDEN_SIZE),
        ),
        (
            "selector_rank",
            PythonKernelArg::Usize(GLM53_DFLASH2_SELECTOR_RANK),
        ),
        (
            "selector_top_k",
            PythonKernelArg::Usize(GLM53_DFLASH2_SELECTOR_TOP_K),
        ),
        (
            "vocab_size",
            PythonKernelArg::Usize(GLM53_DFLASH2_VOCAB_SIZE),
        ),
        ("seed", PythonKernelArg::I64(config.seed)),
        (
            "initialize_hidden",
            PythonKernelArg::Bool(config.initialize_hidden),
        ),
    ];
    launch_python_graph_capture(PythonGraphCaptureLaunch {
        module: "dflash_head_capture",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_exact_dflash2_head_buffers_for_all_concurrency_buckets() {
        for active_requests in [1, 2, 4] {
            for proposal_tokens in 1..=GLM53_DFLASH2_MAX_DRAFTS {
                let plan = dflash2_head_buffer_plan(active_requests, proposal_tokens).unwrap();
                assert_eq!(plan.len(), 13);
                let bytes = |name: &str| plan.iter().find(|item| item.name == name).unwrap().bytes;
                assert_eq!(
                    bytes("hidden"),
                    active_requests * (proposal_tokens + 1) * 6_144 * std::mem::size_of::<u16>()
                );
                assert_eq!(
                    bytes("logits"),
                    active_requests * proposal_tokens * 154_880 * std::mem::size_of::<u16>()
                );
                assert!(plan.iter().all(|item| item.name != "transition_scores"));
                assert_eq!(
                    bytes("radix_candidates"),
                    active_requests * proposal_tokens * 16 * 4
                );
                assert_eq!(
                    bytes("radix_row_states"),
                    DFLASH2_RADIX_TOPK_ROW_STATES_BYTES
                );
            }
        }
        assert!(dflash2_head_buffer_plan(3, 7).is_err());
        assert!(dflash2_head_buffer_plan(1, 0).is_err());
        assert!(dflash2_head_buffer_plan(1, 8).is_err());
    }
}
