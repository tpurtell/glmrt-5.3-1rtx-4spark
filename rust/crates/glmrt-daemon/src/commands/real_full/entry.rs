use anyhow::{bail, Context, Result};
use glmrt_core::{
    coordinator_graph_bucket_for_active_rows, KvCacheConfig, KvCacheDType,
    MlaKvCacheRepresentation, TensorCatalog, TensorRole, COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS,
    EXPERT_HOSTS, GLM52_NUM_HIDDEN_LAYERS,
};
use glmrt_loader::{decode_tokenizer_ids, LoadedTokenizer};
use glmrt_transport::{expert_protocol_v2_compact_id, TcpProtocolV2HostBatchTarget};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs::{self, File};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::cli::CoordinatorArgs;
use crate::commands::model_artifacts::{read_expert_owner_lookup, ExpertOwnerLookup};
use crate::python_graph_capture::coordinator_python_capture_enabled;

use super::constants::REAL_GLM_FULL_BLOCKER;
use super::constraint::{
    RealFullConstraintBranch, RealFullConstraintCompiler, RealFullConstraintState,
};
use super::coordinator_kernels::{
    audit_glm_dsa_nvfp4_short_k_prefill_graph_retention,
    clear_transient_coordinator_owned_device_buffers, coordinator_owned_device_buffer_bank_scope,
    prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker,
    reset_glm_dsa_sparse_mla_transient_state, seal_coordinator_owned_device_buffer_pool,
    with_coordinator_owned_device_buffer_bank, DeviceBf16Output, GLM_DSA_PREFILL_MAX_QUERY_ROWS,
};
use super::dflash::{
    Dflash2BatchedReplayRequest, Dflash2RequestCacheSnapshot, Dflash2RequestEngine,
    Dflash2RequestState, GLM53_DFLASH2_DRAFT_LAYERS, GLM53_DFLASH2_MAX_DRAFTS,
    GLM53_DFLASH2_TARGET_CAPTURE_TAPS,
};
use super::dspark::{
    dspark_active_max_verify_drafts, dspark_target_hidden_tap_layer_ids,
    install_qualified_dflash2_cost_profile, install_qualified_dspark_cost_profile,
    schedule_dspark_verification_with_minimums, DsparkConfidenceCalibrator,
    DsparkConfidenceResidual, DsparkDraftPlan, DsparkRequestCacheSnapshot, DsparkRequestEngine,
    DsparkRequestState, DsparkRuntimeCostModel, DsparkRuntimeCostObservation, DsparkScheduleSearch,
    DsparkSpsProfile, DsparkVerificationSchedule,
};
use super::dspark_static::DsparkDraftStep;
use super::embedding::real_full_embedding_hidden_for_token;
use super::execution_plan::real_full_execution_plan;
use super::expert_probe::REAL_NVFP4_PROTOCOL_V2_EXECUTOR;
use super::kv::device::RealFullDeviceKvStorageHandle;
use super::layer_blocks::SparkLayerBlock;
use super::mtp::{
    prewarm_real_full_paired_target_token_sample_rows, prewarm_real_full_target_sampler_capacity,
    real_full_device_hidden_row, real_full_device_hidden_rows, real_full_mtp_draft_token,
    real_full_mtp_draft_token_constrained, real_full_mtp_envelope_device_hidden,
    real_full_mtp_shifted_input_token_ids, real_full_target_hidden_for_mtp,
    real_full_target_token_samples, real_full_target_token_samples_constrained,
    real_full_target_token_samples_pair, real_full_target_token_samples_with_options,
    RealFullMtpDraftToken,
};
use super::prefix_cache::{
    TargetKvExactSubtreeEviction, TargetKvRadixManager, TargetKvRadixReservation,
};
use super::preflight::{
    real_full_kv_cache_config, real_full_sparse_transport_plan, real_glm_full_preflight_report,
};
use super::residency::preload_real_full_coordinator_resident_weights;
use super::sampling::{
    score_real_lm_head_chunk_for_hidden, RealFullLmHeadSamplingOptions,
    RealLmHeadBatchScoreForHidden,
};
use super::scheduler::{
    load_real_full_kv_snapshot,
    real_full_scheduler_execute_prefill_decode_layer_block_device_input,
    real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden,
    real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden,
    real_full_scheduler_execution_for_shape_with_sparse_tcp,
    real_full_scheduler_execution_for_shape_with_state, save_real_full_kv_snapshot,
    RealFullKvSnapshot, RealFullSchedulerBatchedInput, RealFullSchedulerDeviceExecution,
    RealFullSchedulerExecutionShape, RealFullSchedulerExecutionState,
    RealFullSchedulerSparseDispatchTransport, RealFullSchedulerSparseTcpDispatchProbe,
    RealFullSchedulerSparseTcpDispatchWorker,
};
use super::sparse_mlp::route::B12X_EXL3_TOPK8_CAPACITY_ROWS;
use super::types::{
    RealFullCoordinatorResidentPreloadPlan, RealFullSchedulerExecutionDryRun,
    RealFullSchedulerTerminalLmHeadSample, RealGlmFullPreflightReport,
};

const DEFAULT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS: usize = 2 * 1024;
const DEFAULT_REAL_FULL_REQUEST_FRESH_SMALL_PREFILL_CHUNK_TOKENS: usize = 512;
const DEFAULT_REAL_FULL_REQUEST_SMALL_PREFILL_CHUNK_TOKENS: usize = 256;
const DEFAULT_REAL_FULL_REQUEST_CACHED_WIDE_SUFFIX_MIN_TOKENS: usize = 1_024 + 1;
const DEFAULT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS: usize = 4 * 1024;
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_MIN_TOKENS: usize = 32 * 1024;
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS: usize = 512;

fn request_lm_head_sampling_options(
    request: &glmrt_api::RealFullRequest,
) -> RealFullLmHeadSamplingOptions {
    request_lm_head_sampling_options_at(request, request.decode_step_index)
}

fn request_lm_head_sampling_options_at(
    request: &glmrt_api::RealFullRequest,
    decode_step_index: usize,
) -> RealFullLmHeadSamplingOptions {
    if request.greedy_sampling {
        return RealFullLmHeadSamplingOptions::diagnostic();
    }
    RealFullLmHeadSamplingOptions {
        random_uniform: request.sampling.random_uniform(decode_step_index),
        temperature: request.sampling.temperature(),
        top_k: request.sampling.top_k(),
        top_p: request.sampling.top_p(),
    }
}

fn request_sampling_uniforms(
    request: &glmrt_api::RealFullRequest,
    start_decode_step: usize,
    rows: usize,
) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            request
                .sampling
                .random_uniform(start_decode_step.saturating_add(row))
        })
        .collect()
}

fn real_full_constraint_target_samples(
    catalog: &TensorCatalog,
    state: &BudgetedRealFullSchedulerExecutionState,
    request: &glmrt_api::RealFullRequest,
    target_hidden: &DeviceBf16Output,
    suffix_rows: usize,
    draft_token_ids: &[usize],
) -> Result<Option<RealLmHeadBatchScoreForHidden>> {
    let Some(constraint) = state.constraint.as_ref() else {
        return Ok(None);
    };
    anyhow::ensure!(
        suffix_rows == draft_token_ids.len() + 1,
        "constrained target suffix has {suffix_rows} rows for {} speculative drafts",
        draft_token_ids.len()
    );
    let masks = constraint
        .masks_for_draft(draft_token_ids)
        .context("building constrained target sampling masks")?;
    let options = request_lm_head_sampling_options_at(request, request.decode_step_index);
    let random_uniforms = if request.greedy_sampling {
        vec![options.random_uniform; suffix_rows]
    } else {
        request_sampling_uniforms(request, request.decode_step_index, suffix_rows)
    };
    real_full_target_token_samples_constrained(
        catalog,
        target_hidden,
        suffix_rows,
        options,
        &random_uniforms,
        &masks,
    )
    .map(Some)
}
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_NUMERATOR: usize = 7;
const DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_DENOMINATOR: usize = 4;
const DEFAULT_REAL_FULL_REQUEST_MIN_STREAMING_TAIL_PREFILL_TOKENS: usize = 15;
const REAL_FULL_PREFILL_PIPELINE_LANES: usize = 4;
const REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS_ENV: &str =
    "GLMRT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS";
const REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS_ENV: &str =
    "GLMRT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS";
const REAL_FULL_SEQUENCE_EXTENSION_HEADROOM_TOKENS: usize = 4 * 1024;
const REAL_FULL_SHARED_KV_PAGE_TOKENS: usize = 64;
const REAL_FULL_MAX_ACTIVE_REQUESTS: usize = 16;
const REAL_FULL_MAX_EXECUTION_LANES_ENV: &str = "GLMRT_REAL_FULL_MAX_EXECUTION_LANES";
const REAL_FULL_DIAGNOSTIC_MAX_EXECUTION_LANES: usize = 8;
const REAL_FULL_REQUEST_MAX_MTP_VERIFY_ROWS: usize = 4;
const REAL_FULL_REQUEST_MTP_ACCEPTED_ROWS: usize = 2;
const REAL_FULL_SERVE_PREWARM_REQUEST_ENV: &str = "GLMRT_REAL_FULL_SERVE_PREWARM_REQUEST";
const REAL_FULL_STARTUP_SEAL_OWNED_BUFFER_POOL_PREFIX: &str =
    "real-full-startup-seal-owned-buffer-pool-";
const REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX: &str =
    "real-full-startup-prewarm-paired-lm-head-";
const REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX: &str =
    "real-full-startup-prewarm-batched-dspark-";
const REAL_FULL_STARTUP_MAX_PREFILL_CHUNK_PREFIX: &str =
    "real-full-startup-capture-arena-max-prefill-chunk-";
const REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX: &str =
    "real-full-startup-capture-arena-canonical-prefill-chunk-";
const REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_TOKENS: usize = 1_024;
const REAL_FULL_STARTUP_AUDIT_NVFP4_SHORT_K_PREFIX: &str =
    "real-full-startup-audit-nvfp4-short-k-q";
const REAL_FULL_STARTUP_TARGET_RADIX_PUBLISH_PREFIX: &str =
    "real-full-startup-capture-arena-radix-publish-";
const REAL_FULL_STARTUP_TARGET_RADIX_EVICT_PREFIX: &str =
    "real-full-startup-evict-target-radix-prefix-";
const REAL_FULL_STARTUP_BATCHED_DSPARK_BANK_MARKER: &str = "-batched-bank-";
const REAL_FULL_STARTUP_SCALAR_DSPARK_COHORT_MARKER: &str = "-scalar-cohort-";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ENV: &str = "GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV: &str =
    "GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS_ENV: &str =
    "GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS";
const REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS_ENV: &str =
    "GLMRT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS";
const MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS: usize = 8;
const REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN: &str = "alpha ";
const REAL_FULL_SERVE_PREWARM_BOUNDARY_TOKEN: &str = "beta ";
const REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS: usize = 2_049;
const REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS: usize = 512;
// The 512-row boundary is the canonical graph identity for this bucket. A
// historical 1,008-row sizing request was balanced into two 504-row chunks;
// its 99 exact-row BF16 linear captures were immediately replaced here, while
// its retained packed-attention capture set is equally seeded by this request.
const REAL_FULL_SERVE_PREWARM_PREFILL_ROWS: &[usize] = &[2_048, 512, 144, 72, 36, 18, 9, 8];
const REAL_FULL_SERVE_DSA_SELECTOR_PREWARM_QUERY_ROWS: &[usize] = &[8, 16, 32, 64, 128, 256, 512];
// Native NVFP4 chooses K=128/512/1024/2048 from the live context only for
// query buckets through 64. The dSpark sweep covers q1/q2/q4/q8, while the
// long-context selector sweep covers q16/q32/q64 only with the selector
// enabled. These exact cached-prefix widths fill the remaining no-selector
// Cartesian product after shared attention scratch reaches its final size.
const REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS: &[usize] = &[16, 32, 64];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealFullNvfp4ShortKPrefillCaptureAnchor {
    prompt_tokens: usize,
    sparse_topk: usize,
}

const REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_CAPTURE_ANCHORS:
    &[RealFullNvfp4ShortKPrefillCaptureAnchor] = &[
    RealFullNvfp4ShortKPrefillCaptureAnchor {
        prompt_tokens: 9,
        sparse_topk: 128,
    },
    RealFullNvfp4ShortKPrefillCaptureAnchor {
        prompt_tokens: 145,
        sparse_topk: 512,
    },
    RealFullNvfp4ShortKPrefillCaptureAnchor {
        prompt_tokens: 513,
        sparse_topk: 1_024,
    },
    RealFullNvfp4ShortKPrefillCaptureAnchor {
        prompt_tokens: 1_025,
        sparse_topk: 2_048,
    },
];

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealFullNvfp4ShortKPrefillCaptureCase {
    anchor_prompt_tokens: usize,
    query_rows: usize,
    total_rows: usize,
    sparse_topk: usize,
}

#[cfg(test)]
fn real_full_nvfp4_short_k_prefill_capture_plan() -> Vec<RealFullNvfp4ShortKPrefillCaptureCase> {
    let mut cases = Vec::with_capacity(
        REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_CAPTURE_ANCHORS.len()
            * REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS.len(),
    );
    for anchor in REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_CAPTURE_ANCHORS {
        let mut prompt_tokens = anchor.prompt_tokens;
        for query_rows in REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS {
            prompt_tokens += query_rows + 1;
            cases.push(RealFullNvfp4ShortKPrefillCaptureCase {
                anchor_prompt_tokens: anchor.prompt_tokens,
                query_rows: *query_rows,
                total_rows: prompt_tokens - 1,
                sparse_topk: anchor.sparse_topk,
            });
        }
    }
    cases
}

fn real_full_nvfp4_short_k_prefill_decode_budget() -> Result<usize> {
    REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS
        .iter()
        .try_fold(1_usize, |budget, query_rows| {
            query_rows
                .checked_add(1)
                .and_then(|rows| budget.checked_add(rows))
                .context("NVFP4 short-K prefill capture decode budget overflow")
        })
}

// Native MTP currently replays the target prompt hidden rows through layer 78.
// The bounded >=8,193-row target prefill wavefront releases completed hidden
// chunks, so keep the MTP-specific DSA graph seed on the largest qualified
// unbounded prompt until layer-78 replay is streamed alongside that wavefront.
const REAL_FULL_MTP_STARTUP_MAX_REPLAY_ROWS: usize = 8_192;
const DEFAULT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ROWS: usize = 1_008;
const REAL_FULL_SERVE_PREWARM_DECODE_BUDGET: usize = 2;
const REAL_FULL_REQUEST_TIMING_ENV: &str = "GLMRT_REAL_FULL_REQUEST_TIMING";
const REAL_FULL_REQUEST_THREAD_PINNED_ENV: &str = "GLMRT_REAL_FULL_REQUEST_THREAD_PINNED";
const REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV: &str =
    "GLMRT_REAL_FULL_REQUEST_THREAD_PINNED_WORKERS";
const REAL_FULL_REQUEST_WORKER_CPUS_ENV: &str = "GLMRT_REAL_FULL_REQUEST_WORKER_CPUS";
const REAL_FULL_SCHEDULER_WORKER_CPU_ENV: &str = "GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU";
const REAL_FULL_REQUEST_MTP_VERIFY_ENV: &str = "GLMRT_REAL_FULL_REQUEST_MTP_VERIFY";
const REAL_FULL_DSPARK_ENV: &str = "GLMRT_REAL_FULL_DSPARK";
const REAL_FULL_DSPARK_SHADOW_ENV: &str = "GLMRT_REAL_FULL_DSPARK_SHADOW";
const REAL_FULL_DSPARK_SNAPSHOT_ENV: &str = "GLMRT_REAL_FULL_DSPARK_SNAPSHOT";
const REAL_FULL_DSPARK_CONTEXT_TOKENS_ENV: &str = "GLMRT_REAL_FULL_DSPARK_CONTEXT_TOKENS";
const REAL_FULL_DSPARK_CACHE_MODE_ENV: &str = "GLMRT_REAL_FULL_DSPARK_CACHE_MODE";
const REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV: &str = "GLMRT_REAL_FULL_DSPARK_TAIL_CACHE_BYTES";
const REAL_FULL_DSPARK_TRACE_ENV: &str = "GLMRT_REAL_FULL_DSPARK_TRACE";
const REAL_FULL_DSPARK_CONFIDENCE_POLICY_ENV: &str = "GLMRT_REAL_FULL_DSPARK_CONFIDENCE_POLICY";
const REAL_FULL_DSPARK_FIXED_DRAFTS_ENV: &str = "GLMRT_REAL_FULL_DSPARK_FIXED_DRAFTS";
const REAL_FULL_DSPARK_PROFILE_AT_STARTUP_ENV: &str = "GLMRT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP";
const REAL_FULL_DSPARK_PROFILE_SAMPLES_ENV: &str = "GLMRT_REAL_FULL_DSPARK_PROFILE_SAMPLES";
const REAL_FULL_DFLASH2_ENV: &str = "GLMRT_REAL_FULL_DFLASH2";
const REAL_FULL_DFLASH2_SNAPSHOT_ENV: &str = "GLMRT_REAL_FULL_DFLASH2_SNAPSHOT";
const REAL_FULL_DFLASH2_FIXED_DRAFTS_ENV: &str = "GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS";
const REAL_FULL_DFLASH2_TAIL_CACHE_BYTES_ENV: &str = "GLMRT_REAL_FULL_DFLASH2_TAIL_CACHE_BYTES";
const REAL_FULL_DSPARK_SPARKINFER_REVISION_ENV: &str = "GLMRT_SPARKINFER_COMMIT";
const REAL_FULL_DSPARK_COORDINATOR_POWER_LIMIT_WATTS_ENV: &str =
    "GLMRT_COORDINATOR_POWER_LIMIT_WATTS";
const REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV: &str =
    "GLMRT_REAL_FULL_SERVE_EXPERT_READY_TIMEOUT_SECS";
const REAL_FULL_EXPERT_WARMUP_STATUS_FILE_ENV: &str =
    "GLMRT_REAL_FULL_SERVE_EXPERT_WARMUP_STATUS_FILE";
const DEFAULT_REAL_FULL_EXPERT_READY_TIMEOUT_SECS: u64 = 900;
const REAL_FULL_KV_POOL_TOKENS_ENV: &str = "GLMRT_REAL_FULL_KV_POOL_TOKENS";
const DEFAULT_REAL_FULL_DSPARK_REQUEST_LOCAL_CONTEXT_TOKENS: usize = 4 * 1024;
const DEFAULT_REAL_FULL_DSPARK_PROMPT_SWA_CONTEXT_TOKENS: usize = 2 * 1024;
const REAL_FULL_DSPARK_PAGE_SIZE: usize = 64;
const REAL_FULL_DSPARK_QUERY_ROWS: usize = 16;
const REAL_FULL_DSPARK_BF16_KV_BYTES_PER_TOKEN: usize = 81_920;
const REAL_FULL_DFLASH2_BF16_KV_BYTES_PER_TOKEN: usize = 24_576;
// Siro emits fifteen proposals. The adaptive policy covers the complete
// physical M=1..16 range; measured per-width costs decide how many are worth
// verifying on each cycle.
const REAL_FULL_DSPARK_ADAPTIVE_MAX_VERIFY_DRAFTS: usize = 15;
const REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS: usize = 15;
const REAL_FULL_MTP_PROBE_ENV: &str = "GLMRT_REAL_FULL_MTP_PROBE";
const REAL_FULL_MTP_ENV: &str = "GLMRT_REAL_FULL_MTP";
const REAL_FULL_MTP_DRAFT_TOKENS_ENV: &str = "GLMRT_REAL_FULL_MTP_DRAFT_TOKENS";
const REAL_FULL_MTP_MIN_DRAFT_TOKENS_ENV: &str = "GLMRT_REAL_FULL_MTP_MIN_D";
const REAL_FULL_MTP_MAX_DRAFT_TOKENS_ENV: &str = "GLMRT_REAL_FULL_MTP_MAX_D";
const REAL_FULL_MTP_PREFILL_CHUNK_TOKENS_ENV: &str = "GLMRT_REAL_FULL_MTP_PREFILL_CHUNK_TOKENS";
const REAL_FULL_MTP_FULL_MATCH_BONUS_ENV: &str = "GLMRT_REAL_FULL_MTP_FULL_MATCH_BONUS";
const REAL_FULL_MTP_ALLOW_PHYSICAL_M2_ENV: &str = "GLMRT_REAL_FULL_MTP_ALLOW_PHYSICAL_M2";
const REAL_FULL_MTP_DIAGNOSTIC_PHYSICAL_M2_ENV: &str = "GLMRT_REAL_FULL_MTP_DIAGNOSTIC_PHYSICAL_M2";
const REAL_FULL_KV_SNAPSHOT_LOAD_ENV: &str = "GLMRT_REAL_FULL_KV_SNAPSHOT_LOAD";
const REAL_FULL_KV_SNAPSHOT_SAVE_ENV: &str = "GLMRT_REAL_FULL_KV_SNAPSHOT_SAVE";
const REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS_ENV: &str = "GLMRT_REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS";
const REAL_FULL_KV_SNAPSHOT_SAVE_POINTS_ENV: &str = "GLMRT_REAL_FULL_KV_SNAPSHOT_SAVE_POINTS";
const REAL_FULL_ENGINE_COMMIT_ENV: &str = "GLMRT_ENGINE_COMMIT";
const DEFAULT_REAL_FULL_MTP_MIN_DRAFT_TOKENS: usize = 1;
const DEFAULT_REAL_FULL_MTP_MAX_DRAFT_TOKENS: usize = 7;
const DEFAULT_REAL_FULL_MTP_PREFILL_CHUNK_TOKENS: usize = 1024;
const MAX_REAL_FULL_MTP_DRAFT_TOKENS: usize = 7;
const REAL_FULL_MTP_REQUEST_ID_OFFSET: u64 = 900_000;
const REAL_FULL_MTP_REQUEST_ID_STEP_STRIDE: u64 = 10_000;

pub(crate) struct LoadedRealFullServing {
    pub(crate) info: glmrt_api::RealFullInfo,
    pub(crate) executor: Arc<dyn glmrt_api::RealFullRequestExecutor>,
}

struct RealFullPrefixPrefillProbe {
    cases: Vec<RealFullPrefixPrefillProbeCase>,
    repeats: usize,
}

struct RealFullPrefixPrefillProbeCase {
    prefix_prompt: String,
    prefix_prompt_tokens: usize,
    new_prompt_rows: usize,
}

struct RealFullSchedulerRequestExecutor {
    base_info: glmrt_api::RealFullInfo,
    catalog: TensorCatalog,
    kv_config: KvCacheConfig,
    device_kv_pool_config: KvCacheConfig,
    sparse_tcp_targets: Option<Vec<TcpProtocolV2HostBatchTarget>>,
    sparse_owner_lookup: Option<ExpertOwnerLookup>,
    sparse_tcp_dispatch_worker: Option<Arc<RealFullSchedulerSparseTcpDispatchWorker>>,
    scheduler_states: Mutex<HashMap<String, BudgetedRealFullSchedulerExecutionState>>,
    recycled_scheduler_states: Mutex<Vec<BudgetedRealFullSchedulerExecutionState>>,
    max_execution_lanes: usize,
    device_kv_storage: Mutex<Option<RealFullDeviceKvStorageHandle>>,
    context_budget: Arc<RealFullContextTokenBudget>,
    target_kv_radix: Arc<TargetKvRadixManager>,
    sampled_token_text_cache: Mutex<HashMap<usize, String>>,
    tokenizer: Mutex<LoadedTokenizer>,
    constraint_compiler: RealFullConstraintCompiler,
    kv_snapshot_load: Option<Arc<RealFullKvSnapshot>>,
    kv_snapshot_saves: Vec<RealFullKvSnapshotSave>,
    kv_snapshot_saved: AtomicBool,
    dspark: Option<Mutex<RealFullDsparkRuntime>>,
    engine_commit: String,
}

enum RealFullDraftEngine {
    Dspark(DsparkRequestEngine),
    Dflash2(Dflash2RequestEngine),
}

enum RealFullDraftRequestState {
    Dspark(DsparkRequestState),
    Dflash2(Dflash2RequestState),
}

enum RealFullDraftCacheSnapshot {
    Dspark(DsparkRequestCacheSnapshot),
    Dflash2(Dflash2RequestCacheSnapshot),
}

impl RealFullDraftRequestState {
    fn context_tokens(&self) -> usize {
        match self {
            Self::Dspark(state) => state.context_tokens(),
            Self::Dflash2(state) => state.context_tokens(),
        }
    }
}

impl RealFullDraftCacheSnapshot {
    fn context_tokens(&self) -> usize {
        match self {
            Self::Dspark(snapshot) => snapshot.context_tokens,
            Self::Dflash2(snapshot) => snapshot.context_tokens,
        }
    }

    fn cache_context_tokens(&self) -> usize {
        match self {
            Self::Dspark(snapshot) => snapshot.cache_context_tokens,
            Self::Dflash2(snapshot) => snapshot.cache_context_tokens,
        }
    }

    fn resident_bytes(&self) -> usize {
        match self {
            Self::Dspark(snapshot) => snapshot.resident_bytes(),
            Self::Dflash2(snapshot) => snapshot.resident_bytes(),
        }
    }
}

impl RealFullDraftEngine {
    fn is_dflash2(&self) -> bool {
        matches!(self, Self::Dflash2(_))
    }

    fn checkpoint_revision(&self) -> &'static str {
        match self {
            Self::Dspark(engine) => engine.checkpoint_revision(),
            Self::Dflash2(engine) => engine.checkpoint_revision(),
        }
    }

    fn max_verify_drafts(&self) -> usize {
        match self {
            Self::Dspark(engine) => engine.max_verify_drafts(),
            Self::Dflash2(engine) => engine.max_verify_drafts(),
        }
    }

    fn target_layer_ids(&self) -> Vec<usize> {
        match self {
            Self::Dspark(_) => dspark_target_hidden_tap_layer_ids().to_vec(),
            Self::Dflash2(engine) => engine.target_layer_ids().to_vec(),
        }
    }

    fn allocate_request_state(&mut self) -> Result<RealFullDraftRequestState> {
        match self {
            Self::Dspark(engine) => engine
                .allocate_request_state()
                .map(RealFullDraftRequestState::Dspark),
            Self::Dflash2(engine) => engine
                .allocate_request_state()
                .map(RealFullDraftRequestState::Dflash2),
        }
    }

    fn reset_request_state(&mut self, state: &mut RealFullDraftRequestState) -> Result<()> {
        match (self, state) {
            (Self::Dspark(engine), RealFullDraftRequestState::Dspark(state)) => {
                engine.reset_request_state(state)
            }
            (Self::Dflash2(engine), RealFullDraftRequestState::Dflash2(state)) => {
                engine.reset_request_state(state)
            }
            _ => bail!("draft request state does not match its engine"),
        }
    }

    fn release_request_state(&mut self, state: RealFullDraftRequestState) {
        match (self, state) {
            (Self::Dspark(engine), RealFullDraftRequestState::Dspark(state)) => {
                engine.release_request_state(state)
            }
            (Self::Dflash2(engine), RealFullDraftRequestState::Dflash2(state)) => {
                engine.release_request_state(state)
            }
            _ => debug_assert!(false, "draft request state does not match its engine"),
        }
    }

    fn snapshot_request_state(
        &self,
        state: &RealFullDraftRequestState,
    ) -> Result<Option<RealFullDraftCacheSnapshot>> {
        match (self, state) {
            (Self::Dspark(engine), RealFullDraftRequestState::Dspark(state)) => engine
                .snapshot_request_state(state)
                .map(|snapshot| snapshot.map(RealFullDraftCacheSnapshot::Dspark)),
            (Self::Dflash2(engine), RealFullDraftRequestState::Dflash2(state)) => engine
                .snapshot_request_state(state)
                .map(|snapshot| snapshot.map(RealFullDraftCacheSnapshot::Dflash2)),
            _ => bail!("draft request state does not match its engine"),
        }
    }

    fn snapshot_request_state_at_prefix(
        &self,
        state: &RealFullDraftRequestState,
        prefix_tokens: usize,
    ) -> Result<Option<RealFullDraftCacheSnapshot>> {
        match (self, state) {
            (Self::Dspark(engine), RealFullDraftRequestState::Dspark(state)) => engine
                .snapshot_request_state_at_prefix(state, prefix_tokens)
                .map(|snapshot| snapshot.map(RealFullDraftCacheSnapshot::Dspark)),
            (Self::Dflash2(engine), RealFullDraftRequestState::Dflash2(state)) => engine
                .snapshot_request_state_at_prefix(state, prefix_tokens)
                .map(|snapshot| snapshot.map(RealFullDraftCacheSnapshot::Dflash2)),
            _ => bail!("draft request state does not match its engine"),
        }
    }

    fn restore_request_state(
        &mut self,
        state: &mut RealFullDraftRequestState,
        snapshot: &RealFullDraftCacheSnapshot,
    ) -> Result<()> {
        match (self, state, snapshot) {
            (
                Self::Dspark(engine),
                RealFullDraftRequestState::Dspark(state),
                RealFullDraftCacheSnapshot::Dspark(snapshot),
            ) => engine.restore_request_state(state, snapshot),
            (
                Self::Dflash2(engine),
                RealFullDraftRequestState::Dflash2(state),
                RealFullDraftCacheSnapshot::Dflash2(snapshot),
            ) => engine.restore_request_state(state, snapshot),
            _ => bail!("draft cache snapshot does not match its engine"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn replay_step(
        &mut self,
        state: &mut RealFullDraftRequestState,
        target_hidden_taps: &[DeviceBf16Output],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<DsparkDraftStep> {
        match (self, state) {
            (Self::Dspark(engine), RealFullDraftRequestState::Dspark(state)) => {
                let taps: [&DeviceBf16Output; 5] = target_hidden_taps
                    .iter()
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|taps: Vec<_>| {
                        anyhow::anyhow!("dSpark requires 5 target taps, got {}", taps.len())
                    })?;
                engine.replay_step(
                    state,
                    taps,
                    target_row_start,
                    committed_rows,
                    absolute_context_start,
                    anchor_token,
                )
            }
            (Self::Dflash2(engine), RealFullDraftRequestState::Dflash2(state)) => {
                let taps: [&DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS] = target_hidden_taps
                    .iter()
                    .collect::<Vec<_>>()
                    .try_into()
                    .map_err(|taps: Vec<_>| {
                        anyhow::anyhow!(
                            "DFlash2 requires {GLM53_DFLASH2_DRAFT_LAYERS} target taps, got {}",
                            taps.len()
                        )
                    })?;
                let step = engine.replay_step(
                    state,
                    taps,
                    target_row_start,
                    committed_rows,
                    absolute_context_start,
                    anchor_token,
                )?;
                Ok(dspark_step_from_dflash(step))
            }
            _ => bail!("draft request state does not match its engine"),
        }
    }
}

fn dspark_step_from_dflash(step: super::dflash_static::Dflash2DraftStep) -> DsparkDraftStep {
    DsparkDraftStep {
        context_tokens: step.context_tokens,
        committed_rows: step.committed_rows,
        anchor_token: step.anchor_token,
        proposal_token_ids: step.proposal_token_ids,
        conditional_confidence: Vec::new(),
        update_ms: step.update_ms,
        suffix_ms: step.suffix_ms,
        readback_ms: step.readback_ms,
        total_ms: step.total_ms,
    }
}

fn dflash_draft_plan(
    step: &DsparkDraftStep,
    target_context_tokens: usize,
    adaptive: &Dflash2AdaptiveDraftState,
    sps: Option<&DsparkSpsProfile>,
) -> Result<DsparkDraftPlan> {
    let max_drafts = step.proposal_token_ids.len().min(GLM53_DFLASH2_MAX_DRAFTS);
    anyhow::ensure!(max_drafts > 0, "DFlash2 returned no proposal tokens");
    if let Some(fixed_drafts) = real_full_dflash2_fixed_drafts()? {
        let fixed_drafts = fixed_drafts.min(max_drafts);
        return Ok(DsparkDraftPlan {
            proposal_token_ids: step.proposal_token_ids[..fixed_drafts].to_vec(),
            conditional_confidence: Vec::new(),
            candidate_proposal_token_ids: step.proposal_token_ids[..max_drafts].to_vec(),
            candidate_conditional_confidence: Vec::new(),
            candidate_adjusted_confidence: Vec::new(),
            selected_drafts: fixed_drafts,
            minimum_drafts: fixed_drafts,
            target_batch_rows: fixed_drafts + 1,
            expected_committed_tokens: 0.0,
            expected_tokens_per_second: 0.0,
            confidence_logit_bias: 0.0,
            confidence_context_tokens: target_context_tokens,
            calibration_eligible: false,
        });
    }

    let adjusted_confidence = adaptive.conditional_confidence(max_drafts);
    let candidate_confidence = adjusted_confidence
        .iter()
        .copied()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    // Seed code-oriented traffic at the already-qualified K5 reference. Four
    // observed cycles are enough to distinguish its high-survival regime from
    // the lower-acceptance general blend without overreacting to one framing
    // miss.
    let minimum_drafts = if adaptive.cold_start() {
        DFLASH2_ADAPTIVE_START_DRAFTS.min(max_drafts)
    } else {
        0
    };
    let (selected_drafts, target_batch_rows, expected_committed_tokens, expected_tokens_per_second) =
        if let Some(sps) = sps {
            let schedule = if adaptive.cold_start() {
                let selected = DFLASH2_ADAPTIVE_START_DRAFTS.min(max_drafts);
                let mut survival = 1.0;
                let expected = 1.0
                    + adjusted_confidence
                        .iter()
                        .take(selected)
                        .map(|confidence| {
                            survival *= confidence;
                            survival
                        })
                        .sum::<f64>();
                DsparkVerificationSchedule {
                    prefix_lengths: vec![selected],
                    target_batch_rows: selected + 1,
                    expected_committed_tokens: expected,
                    expected_tokens_per_second: expected * sps.get(selected + 1)?,
                }
            } else {
                let candidate = schedule_dspark_verification_with_minimums(
                    std::slice::from_ref(&adjusted_confidence),
                    &[0],
                    sps,
                    DsparkScheduleSearch::GlobalMaximum,
                )?;
                let reference_width = DFLASH2_ADAPTIVE_START_DRAFTS.min(max_drafts);
                let reference_confidence = adjusted_confidence[..reference_width].to_vec();
                let reference = schedule_dspark_verification_with_minimums(
                    std::slice::from_ref(&reference_confidence),
                    std::slice::from_ref(&reference_width),
                    sps,
                    DsparkScheduleSearch::GlobalMaximum,
                )?;
                if candidate.prefix_lengths[0] != reference_width
                    && candidate.expected_tokens_per_second
                        < reference.expected_tokens_per_second * DFLASH2_ADAPTIVE_REFERENCE_MARGIN
                {
                    reference
                } else {
                    candidate
                }
            };
            (
                schedule.prefix_lengths[0],
                schedule.target_batch_rows,
                schedule.expected_committed_tokens,
                schedule.expected_tokens_per_second,
            )
        } else {
            // Batched cycles are replanned jointly after all request-local
            // posterior vectors are available. This width is only the safe
            // placeholder carried between draft replay and joint scheduling.
            let selected = DFLASH2_ADAPTIVE_START_DRAFTS
                .max(minimum_drafts)
                .min(max_drafts);
            let mut survival = 1.0;
            let expected = 1.0
                + adjusted_confidence
                    .iter()
                    .take(selected)
                    .map(|confidence| {
                        survival *= confidence;
                        survival
                    })
                    .sum::<f64>();
            (selected, selected + 1, expected, 0.0)
        };
    Ok(DsparkDraftPlan {
        proposal_token_ids: step.proposal_token_ids[..selected_drafts].to_vec(),
        conditional_confidence: candidate_confidence[..selected_drafts].to_vec(),
        candidate_proposal_token_ids: step.proposal_token_ids[..max_drafts].to_vec(),
        candidate_conditional_confidence: candidate_confidence,
        candidate_adjusted_confidence: adjusted_confidence,
        selected_drafts,
        minimum_drafts,
        target_batch_rows,
        expected_committed_tokens,
        expected_tokens_per_second,
        confidence_logit_bias: 0.0,
        confidence_context_tokens: target_context_tokens,
        calibration_eligible: true,
    })
}

fn dflash_batch_group_size(remaining: usize) -> Option<usize> {
    if remaining >= 4 {
        Some(4)
    } else if remaining >= 2 {
        Some(2)
    } else {
        None
    }
}

fn real_full_draft_absolute_context_start(
    generated_tokens: usize,
    cache_mode: Option<RealFullDsparkCacheMode>,
    token_prefix_tokens: usize,
    target_row_start: usize,
) -> Option<usize> {
    (generated_tokens == 0 && cache_mode == Some(RealFullDsparkCacheMode::PromptSwa))
        .then(|| token_prefix_tokens + target_row_start)
}

struct RealFullDsparkRuntime {
    mode: RealFullDsparkServingMode,
    confidence_policy: RealFullDsparkConfidencePolicy,
    cache_mode: RealFullDsparkCacheMode,
    context_tokens: usize,
    engine: RealFullDraftEngine,
    requests: HashMap<String, RealFullDsparkRequestRuntime>,
    tail_cache: RealFullDsparkTailCache,
    cost_model: DsparkRuntimeCostModel,
}

struct RealFullDsparkRequestRuntime {
    cache: RealFullDraftRequestState,
    confidence_calibrator: DsparkConfidenceCalibrator,
    confidence_residual: DsparkConfidenceResidual,
    dflash_adaptive_draft: Dflash2AdaptiveDraftState,
    pending_verification: Option<DsparkDraftPlan>,
    pending_windows: VecDeque<RealFullDsparkShadowWindow>,
}

const DFLASH2_ADAPTIVE_HISTORY_LIMIT: usize = 16;
const DFLASH2_ADAPTIVE_COLD_START_CYCLES: usize = 4;
const DFLASH2_ADAPTIVE_START_DRAFTS: usize = 5;
const DFLASH2_ADAPTIVE_PRIOR_SUCCESSES: usize = 3;
const DFLASH2_ADAPTIVE_PRIOR_TRIALS: usize = 4;
const DFLASH2_ADAPTIVE_REFERENCE_MARGIN: f64 = 1.02;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Dflash2AdaptiveDraftObservation {
    proposed: usize,
    accepted: usize,
}

#[derive(Clone, Debug, Default)]
struct Dflash2AdaptiveDraftState {
    history: VecDeque<Dflash2AdaptiveDraftObservation>,
}

impl Dflash2AdaptiveDraftState {
    fn reset(&mut self) {
        self.history.clear();
    }

    fn cold_start(&self) -> bool {
        self.history.len() < DFLASH2_ADAPTIVE_COLD_START_CYCLES
    }

    fn observe(&mut self, proposed: usize, accepted: usize) {
        if proposed == 0 {
            return;
        }
        self.history.push_back(Dflash2AdaptiveDraftObservation {
            proposed,
            accepted: accepted.min(proposed),
        });
        while self.history.len() > DFLASH2_ADAPTIVE_HISTORY_LIMIT {
            self.history.pop_front();
        }
    }

    fn conditional_confidence(&self, max_drafts: usize) -> Vec<f64> {
        (1..=max_drafts)
            .map(|position| {
                // A position is observed only when every preceding proposal
                // matched. Later positions after the first mismatch are
                // censored, not failures.
                let mut trials = DFLASH2_ADAPTIVE_PRIOR_TRIALS;
                let mut successes = DFLASH2_ADAPTIVE_PRIOR_SUCCESSES;
                for observation in &self.history {
                    if observation.proposed >= position
                        && observation.accepted >= position.saturating_sub(1)
                    {
                        trials += 1;
                        successes += usize::from(observation.accepted >= position);
                    }
                }
                successes as f64 / trials as f64
            })
            .collect()
    }
}

struct RealFullBatchedDraftReplayInput<'a> {
    sequence_id: &'a str,
    target_hidden_taps: &'a [DeviceBf16Output],
    target_row_start: usize,
    committed_rows: usize,
    absolute_context_start: Option<usize>,
    anchor_token: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RealFullDsparkTailKey {
    prefix_tokens: usize,
    prefix_sha256: [u8; 32],
}

struct RealFullDsparkTailEntry {
    key: RealFullDsparkTailKey,
    snapshot: RealFullDraftCacheSnapshot,
    confidence_calibrator: DsparkConfidenceCalibrator,
    confidence_residual: DsparkConfidenceResidual,
}

struct RealFullDsparkTailCache {
    entries: VecDeque<RealFullDsparkTailEntry>,
    resident_bytes: usize,
    max_bytes: usize,
}

impl RealFullDsparkTailCache {
    fn new(max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            resident_bytes: 0,
            max_bytes,
        }
    }

    fn take_exact_prefix(
        &mut self,
        prompt_token_ids: &[usize],
        matched_target_tokens: usize,
    ) -> Option<RealFullDsparkTailEntry> {
        let matched_target_tokens = matched_target_tokens.min(prompt_token_ids.len());
        let fingerprint =
            real_full_dspark_prefix_fingerprint(&prompt_token_ids[..matched_target_tokens]);
        let index = self.entries.iter().position(|entry| {
            entry.key.prefix_tokens == matched_target_tokens
                && entry.key.prefix_sha256 == fingerprint
        })?;
        let entry = self
            .entries
            .remove(index)
            .expect("the selected dSpark tail entry came from this deque");
        self.resident_bytes = self
            .resident_bytes
            .saturating_sub(entry.snapshot.resident_bytes());
        Some(entry)
    }

    fn longest_exact_prefix_tokens(
        &self,
        prompt_token_ids: &[usize],
        maximum_prefix_tokens: usize,
    ) -> usize {
        let maximum_prefix_tokens = maximum_prefix_tokens.min(prompt_token_ids.len());
        let mut candidate_lengths = self
            .entries
            .iter()
            .map(|entry| entry.key.prefix_tokens)
            .filter(|prefix_tokens| *prefix_tokens != 0 && *prefix_tokens <= maximum_prefix_tokens)
            .collect::<Vec<_>>();
        candidate_lengths.sort_unstable();
        candidate_lengths.dedup();
        let mut candidate_lengths = candidate_lengths.into_iter().peekable();
        let mut hasher = Sha256::new();
        let mut longest_match = 0;
        for (token_index, token_id) in prompt_token_ids[..maximum_prefix_tokens].iter().enumerate()
        {
            hasher.update((*token_id as u64).to_le_bytes());
            let prefix_tokens = token_index + 1;
            if candidate_lengths.peek().copied() != Some(prefix_tokens) {
                continue;
            }
            let prefix_sha256: [u8; 32] = hasher.clone().finalize().into();
            if self.entries.iter().any(|entry| {
                entry.key.prefix_tokens == prefix_tokens && entry.key.prefix_sha256 == prefix_sha256
            }) {
                longest_match = prefix_tokens;
            }
            candidate_lengths.next();
        }
        longest_match
    }

    fn insert(&mut self, entry: RealFullDsparkTailEntry) -> bool {
        let entry_bytes = entry.snapshot.resident_bytes();
        if self.max_bytes == 0 || entry_bytes > self.max_bytes {
            return false;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|current| current.key == entry.key)
        {
            let replaced = self
                .entries
                .remove(index)
                .expect("the duplicate dSpark tail entry came from this deque");
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(replaced.snapshot.resident_bytes());
        }
        while self
            .resident_bytes
            .checked_add(entry_bytes)
            .is_none_or(|bytes| bytes > self.max_bytes)
        {
            let Some(evicted) = self.entries.pop_front() else {
                return false;
            };
            self.resident_bytes = self
                .resident_bytes
                .saturating_sub(evicted.snapshot.resident_bytes());
        }
        self.resident_bytes += entry_bytes;
        self.entries.push_back(entry);
        true
    }
}

fn real_full_dspark_prefix_fingerprint(token_ids: &[usize]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    for token_id in token_ids {
        hasher.update((*token_id as u64).to_le_bytes());
    }
    hasher.finalize().into()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkServingMode {
    Active,
    Shadow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkCacheMode {
    RequestLocal,
    PromptSwa,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkConfidencePolicy {
    Calibrated,
    Raw,
    Residual,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealFullDsparkStartupProfileMode {
    Disabled,
    Report,
    Install,
}

struct RealFullDsparkShadowWindow {
    origin_context: usize,
    proposal_token_ids: Vec<usize>,
    conditional_confidence: Vec<f32>,
    matched: usize,
}

impl RealFullDsparkRuntime {
    fn batched_dflash_enabled(&self) -> bool {
        self.mode == RealFullDsparkServingMode::Active && self.engine.is_dflash2()
    }

    fn is_dflash2(&self) -> bool {
        self.engine.is_dflash2()
    }

    fn target_layer_ids(&self) -> Vec<usize> {
        self.engine.target_layer_ids()
    }

    fn trace_shadow_window(
        sequence_id: &str,
        window: &RealFullDsparkShadowWindow,
        resolution: &'static str,
        target_token: Option<usize>,
    ) {
        if !real_full_dspark_trace_enabled() {
            return;
        }
        let observed_positions = match resolution {
            "full_match" => window.proposal_token_ids.len(),
            "mismatch" => window.matched.saturating_add(1),
            _ => window.matched,
        };
        eprintln!(
            "real_full_dspark_shadow_policy_trace {}",
            serde_json::json!({
                "schema": "glmrt-dspark-shadow-policy-trace-v1",
                "sequence_id": sequence_id,
                "origin_context": window.origin_context,
                "proposal_token_ids": &window.proposal_token_ids,
                "conditional_confidence": &window.conditional_confidence,
                "accepted_prefix": window.matched,
                "observed_positions": observed_positions,
                "resolution": resolution,
                "target_token": target_token,
            })
        );
    }

    fn release_internal_sequences(&mut self) -> usize {
        let stale_internal_sequences = self
            .requests
            .keys()
            .filter(|candidate| real_full_internal_sequence(candidate))
            .cloned()
            .collect::<Vec<_>>();
        let released = stale_internal_sequences.len();
        for stale_sequence in stale_internal_sequences {
            let stale_request = self
                .requests
                .remove(&stale_sequence)
                .expect("the stale dSpark startup sequence came from this map");
            self.engine.release_request_state(stale_request.cache);
        }
        released
    }

    fn prepare_cycle(
        &mut self,
        sequence_id: &str,
        initial_decode: bool,
        reusable_target_prefix: Option<(&[usize], usize)>,
        startup_draft_tokens: Option<usize>,
    ) -> Result<Option<DsparkDraftPlan>> {
        if !self.requests.contains_key(sequence_id) {
            if real_full_internal_sequence(sequence_id)
                && !real_full_batched_dspark_prewarm_sequence(sequence_id)
            {
                self.release_internal_sequences();
            }
            let cache = self.engine.allocate_request_state()?;
            self.requests.insert(
                sequence_id.to_owned(),
                RealFullDsparkRequestRuntime {
                    cache,
                    confidence_calibrator: DsparkConfidenceCalibrator::default(),
                    confidence_residual: DsparkConfidenceResidual::default(),
                    dflash_adaptive_draft: Dflash2AdaptiveDraftState::default(),
                    pending_verification: None,
                    pending_windows: VecDeque::new(),
                },
            );
        } else if initial_decode {
            let request = self
                .requests
                .get_mut(sequence_id)
                .expect("the dSpark request was checked above");
            self.engine.reset_request_state(&mut request.cache)?;
        }
        let reusable_tail = if initial_decode {
            reusable_target_prefix.and_then(|(prompt_token_ids, matched_target_tokens)| {
                let uncached_suffix_tokens =
                    prompt_token_ids.len().saturating_sub(matched_target_tokens);
                (uncached_suffix_tokens < self.context_tokens).then(|| {
                    self.tail_cache
                        .take_exact_prefix(prompt_token_ids, matched_target_tokens)
                })?
            })
        } else {
            None
        };
        let request = self
            .requests
            .get_mut(sequence_id)
            .expect("the dSpark request was inserted above");
        if initial_decode {
            request.pending_verification = None;
            request.pending_windows.clear();
            request.dflash_adaptive_draft.reset();
            if let Some(entry) = reusable_tail {
                let prefix_tokens = entry.key.prefix_tokens;
                let cache_tokens = entry.snapshot.cache_context_tokens();
                let snapshot_bytes = entry.snapshot.resident_bytes();
                if let Err(error) = self
                    .engine
                    .restore_request_state(&mut request.cache, &entry.snapshot)
                {
                    let _ = self.tail_cache.insert(entry);
                    return Err(error).context("restoring a reusable dSpark tail");
                }
                request.confidence_calibrator = entry.confidence_calibrator.clone();
                request.confidence_residual = entry.confidence_residual.clone();
                let retained = self.tail_cache.insert(entry);
                anyhow::ensure!(
                    retained,
                    "restored dSpark tail no longer fits its configured host cache"
                );
                eprintln!(
                    "real_full_dspark_tail_restore sequence_id={} prefix_tokens={} cache_tokens={} snapshot_bytes={} cached_entries={} cached_bytes={}",
                    sequence_id,
                    prefix_tokens,
                    cache_tokens,
                    snapshot_bytes,
                    self.tail_cache.entries.len(),
                    self.tail_cache.resident_bytes,
                );
            } else {
                request.confidence_calibrator.reset();
                request.confidence_residual.reset();
            }
        }
        if self.mode == RealFullDsparkServingMode::Active && !initial_decode {
            if let Some(draft_tokens) = startup_draft_tokens {
                let draft_tokens = draft_tokens.min(self.engine.max_verify_drafts());
                // Startup width capture must replace the ordinary adaptive
                // plan, including the common empty plan produced by the seed
                // step. Otherwise every nominal width sweep remains scalar.
                request.pending_verification = Some(DsparkDraftPlan {
                    proposal_token_ids: vec![0; draft_tokens],
                    conditional_confidence: Vec::new(),
                    candidate_proposal_token_ids: vec![0; draft_tokens],
                    candidate_conditional_confidence: Vec::new(),
                    candidate_adjusted_confidence: Vec::new(),
                    selected_drafts: draft_tokens,
                    minimum_drafts: draft_tokens,
                    target_batch_rows: draft_tokens + 1,
                    expected_committed_tokens: 1.0,
                    expected_tokens_per_second: 0.0,
                    confidence_logit_bias: request.confidence_calibrator.logit_bias(),
                    confidence_context_tokens: request.cache.context_tokens(),
                    calibration_eligible: false,
                });
            }
        }
        Ok((self.mode == RealFullDsparkServingMode::Active)
            .then(|| request.pending_verification.take())
            .flatten())
    }

    fn restore_verification(&mut self, sequence_id: &str, plan: DsparkDraftPlan) {
        if self.mode == RealFullDsparkServingMode::Active {
            if let Some(request) = self.requests.get_mut(sequence_id) {
                request.pending_verification = Some(plan);
            }
        }
    }

    fn replay_step(
        &mut self,
        sequence_id: &str,
        target_hidden_taps: &[DeviceBf16Output],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<(DsparkDraftStep, DsparkDraftPlan)> {
        let request = self
            .requests
            .get_mut(sequence_id)
            .with_context(|| format!("dSpark request state is missing for {sequence_id}"))?;
        if self.mode == RealFullDsparkServingMode::Shadow {
            let mut retained = VecDeque::with_capacity(request.pending_windows.len() + 1);
            while let Some(mut window) = request.pending_windows.pop_front() {
                let expected = window.proposal_token_ids[window.matched];
                let confidence = window.conditional_confidence[window.matched];
                if expected == anchor_token {
                    window.matched += 1;
                    if window.matched == window.proposal_token_ids.len() {
                        Self::trace_shadow_window(sequence_id, &window, "full_match", None);
                        eprintln!(
                            "real_full_dspark_shadow_acceptance sequence_id={} origin_context={} accepted={} full_match=true",
                            sequence_id, window.origin_context, window.matched
                        );
                    } else {
                        retained.push_back(window);
                    }
                } else {
                    Self::trace_shadow_window(sequence_id, &window, "mismatch", Some(anchor_token));
                    eprintln!(
                        "real_full_dspark_shadow_acceptance sequence_id={} origin_context={} accepted={} full_match=false mismatch_position={} expected_token={} target_token={} mismatch_confidence={:.6}",
                        sequence_id,
                        window.origin_context,
                        window.matched,
                        window.matched,
                        expected,
                        anchor_token,
                        confidence,
                    );
                }
            }
            request.pending_windows = retained;
        }
        let step = self.engine.replay_step(
            &mut request.cache,
            target_hidden_taps,
            target_row_start,
            committed_rows,
            absolute_context_start,
            anchor_token,
        )?;
        let target_context_tokens = request.cache.context_tokens();
        let mut plan = if self.engine.is_dflash2() {
            let sps = self.cost_model.profile(1, &[target_context_tokens])?;
            dflash_draft_plan(
                &step,
                target_context_tokens,
                &request.dflash_adaptive_draft,
                Some(&sps),
            )?
        } else {
            let sps = self.cost_model.profile(1, &[target_context_tokens])?;
            let (confidence_logit_bias, position_logit_bias, force_probe) =
                match self.confidence_policy {
                    RealFullDsparkConfidencePolicy::Calibrated => (
                        request.confidence_calibrator.logit_bias(),
                        &[][..],
                        request.confidence_calibrator.force_probe_due(),
                    ),
                    RealFullDsparkConfidencePolicy::Raw => (0.0, &[][..], false),
                    RealFullDsparkConfidencePolicy::Residual => (
                        request
                            .confidence_residual
                            .global_logit_bias(target_context_tokens),
                        request.confidence_residual.position_logit_bias(),
                        false,
                    ),
                };
            let RealFullDraftEngine::Dspark(engine) = &self.engine else {
                unreachable!("DFlash2 was handled above")
            };
            engine.plan_verification(
                &step,
                REAL_FULL_DSPARK_ADAPTIVE_MAX_VERIFY_DRAFTS,
                confidence_logit_bias,
                position_logit_bias,
                target_context_tokens,
                force_probe,
                &sps,
            )?
        };
        if !self.engine.is_dflash2() {
            if let Some(fixed_drafts) = real_full_dspark_fixed_drafts()? {
                plan.proposal_token_ids = step.proposal_token_ids[..fixed_drafts].to_vec();
                plan.conditional_confidence = step.conditional_confidence[..fixed_drafts].to_vec();
                plan.candidate_proposal_token_ids = plan.proposal_token_ids.clone();
                plan.candidate_conditional_confidence = plan.conditional_confidence.clone();
                plan.candidate_adjusted_confidence.truncate(fixed_drafts);
                plan.selected_drafts = fixed_drafts;
                plan.minimum_drafts = fixed_drafts;
                plan.target_batch_rows = fixed_drafts + 1;
                plan.expected_committed_tokens = 0.0;
                plan.expected_tokens_per_second = 0.0;
                plan.calibration_eligible = false;
            }
        }
        if self.mode == RealFullDsparkServingMode::Active {
            request.pending_verification = Some(plan.clone());
        } else {
            request
                .pending_windows
                .push_back(RealFullDsparkShadowWindow {
                    origin_context: step.context_tokens,
                    proposal_token_ids: step.proposal_token_ids.clone(),
                    conditional_confidence: step.conditional_confidence.clone(),
                    matched: 0,
                });
        }
        Ok((step, plan))
    }

    fn replay_batched_dflash_steps(
        &mut self,
        inputs: &[RealFullBatchedDraftReplayInput<'_>],
    ) -> Result<Vec<(DsparkDraftStep, DsparkDraftPlan)>> {
        anyhow::ensure!(
            self.mode == RealFullDsparkServingMode::Active && self.engine.is_dflash2(),
            "batched DFlash2 replay requires the active DFlash2 runtime"
        );
        anyhow::ensure!(
            matches!(inputs.len(), 2 | 4),
            "batched DFlash2 replay requires exactly 2 or 4 requests"
        );
        for (index, input) in inputs.iter().enumerate() {
            anyhow::ensure!(
                !inputs[..index]
                    .iter()
                    .any(|prior| prior.sequence_id == input.sequence_id),
                "batched DFlash2 replay contains duplicate sequence {}",
                input.sequence_id
            );
        }

        let mut removed = Vec::with_capacity(inputs.len());
        for input in inputs {
            let Some(runtime) = self.requests.remove(input.sequence_id) else {
                for (sequence_id, runtime) in removed.drain(..) {
                    self.requests.insert(sequence_id, runtime);
                }
                bail!("DFlash2 request state is missing for {}", input.sequence_id);
            };
            removed.push((input.sequence_id.to_owned(), runtime));
        }

        let result = (|| {
            let tap_arrays = inputs
                .iter()
                .map(|input| {
                    input
                        .target_hidden_taps
                        .iter()
                        .collect::<Vec<_>>()
                        .try_into()
                        .map_err(|taps: Vec<_>| {
                            anyhow::anyhow!(
                                "DFlash2 requires {GLM53_DFLASH2_DRAFT_LAYERS} target taps, got {}",
                                taps.len()
                            )
                        })
                })
                .collect::<Result<Vec<[&DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS]>>>()?;
            let RealFullDraftEngine::Dflash2(engine) = &mut self.engine else {
                unreachable!("the DFlash2 runtime was checked above")
            };
            let mut replay_requests = Vec::with_capacity(inputs.len());
            for (((_, runtime), input), taps) in removed.iter_mut().zip(inputs).zip(tap_arrays) {
                let RealFullDraftRequestState::Dflash2(state) = &mut runtime.cache else {
                    bail!("DFlash2 runtime contains a non-DFlash request cache")
                };
                replay_requests.push(Dflash2BatchedReplayRequest {
                    state,
                    target_hidden_taps: taps,
                    target_row_start: input.target_row_start,
                    committed_rows: input.committed_rows,
                    absolute_context_start: input.absolute_context_start,
                    anchor_token: input.anchor_token,
                });
            }
            let steps = engine.replay_batched_steps(&mut replay_requests)?;
            drop(replay_requests);
            anyhow::ensure!(
                steps.len() == removed.len(),
                "DFlash2 engine returned the wrong batched step count"
            );
            if real_full_dspark_trace_enabled() {
                eprintln!(
                    "real_full_dflash2_batch requests={} committed_rows={:?} packed_update_rows={} update_ms_sum={:.3} suffix_ms={:.3} readback_ms={:.3} batch_total_ms={:.3}",
                    steps.len(),
                    steps
                        .iter()
                        .map(|step| step.committed_rows)
                        .collect::<Vec<_>>(),
                    steps.first().map_or(0, |step| step.packed_update_rows),
                    steps.iter().map(|step| step.update_ms).sum::<f64>(),
                    steps.first().map_or(0.0, |step| step.suffix_ms),
                    steps.first().map_or(0.0, |step| step.readback_ms),
                    steps.first().map_or(0.0, |step| step.total_ms),
                );
            }
            removed
                .iter_mut()
                .zip(steps)
                .map(|((_, runtime), step)| {
                    let target_context_tokens = runtime.cache.context_tokens();
                    let step = dspark_step_from_dflash(step);
                    let plan = dflash_draft_plan(
                        &step,
                        target_context_tokens,
                        &runtime.dflash_adaptive_draft,
                        None,
                    )?;
                    runtime.pending_verification = Some(plan.clone());
                    Ok((step, plan))
                })
                .collect::<Result<Vec<_>>>()
        })();
        for (sequence_id, runtime) in removed {
            // This insertion is required in optimized builds too. Keeping the
            // mutation inside debug_assert! compiled it out in release mode,
            // dropping every request state after a successful batched replay.
            let replaced = self.requests.insert(sequence_id, runtime);
            debug_assert!(replaced.is_none());
        }
        result
    }

    fn record_issued_plan(&mut self, sequence_id: &str, plan: &DsparkDraftPlan) {
        if !plan.calibration_eligible {
            return;
        }
        if let Some(request) = self.requests.get_mut(sequence_id) {
            match self.confidence_policy {
                RealFullDsparkConfidencePolicy::Calibrated => {
                    request
                        .confidence_calibrator
                        .record_selected_drafts(plan.selected_drafts);
                }
                RealFullDsparkConfidencePolicy::Residual => request
                    .confidence_residual
                    .record_selected_drafts(plan.selected_drafts),
                RealFullDsparkConfidencePolicy::Raw => {}
            }
        }
    }

    fn joint_schedule(
        &self,
        plans: &[(&DsparkDraftPlan, usize)],
        context_tokens: &[usize],
    ) -> Result<DsparkVerificationSchedule> {
        anyhow::ensure!(
            plans.len() == context_tokens.len(),
            "joint dSpark scheduling has {} plans and {} contexts",
            plans.len(),
            context_tokens.len(),
        );
        let conditional_confidence = plans
            .iter()
            .map(|(plan, max_useful_drafts)| {
                plan.calibrated_candidate_confidence(*max_useful_drafts)
            })
            .collect::<Vec<_>>();
        let minimum_prefix_lengths = plans
            .iter()
            .map(|(plan, max_useful_drafts)| plan.minimum_drafts.min(*max_useful_drafts))
            .collect::<Vec<_>>();
        let sps = self.cost_model.profile(plans.len(), context_tokens)?;
        let candidate = schedule_dspark_verification_with_minimums(
            &conditional_confidence,
            &minimum_prefix_lengths,
            &sps,
            DsparkScheduleSearch::GlobalMaximum,
        )?;
        if !self.engine.is_dflash2() {
            return Ok(candidate);
        }
        let reference_widths = plans
            .iter()
            .map(|(_, max_useful_drafts)| DFLASH2_ADAPTIVE_START_DRAFTS.min(*max_useful_drafts))
            .collect::<Vec<_>>();
        let reference_confidence = conditional_confidence
            .iter()
            .zip(&reference_widths)
            .map(|(confidence, width)| confidence[..*width].to_vec())
            .collect::<Vec<_>>();
        let reference = schedule_dspark_verification_with_minimums(
            &reference_confidence,
            &reference_widths,
            &sps,
            DsparkScheduleSearch::GlobalMaximum,
        )?;
        let cold_start = plans.iter().any(|(plan, _)| plan.minimum_drafts > 0);
        if cold_start
            || (candidate.prefix_lengths != reference.prefix_lengths
                && candidate.expected_tokens_per_second
                    < reference.expected_tokens_per_second * DFLASH2_ADAPTIVE_REFERENCE_MARGIN)
        {
            Ok(reference)
        } else {
            Ok(candidate)
        }
    }

    fn observe_runtime_cost(
        &mut self,
        context_tokens: &[usize],
        target_rows: usize,
        observed_ms: f64,
        route_critical_unique_experts: Option<usize>,
    ) -> Result<DsparkRuntimeCostObservation> {
        self.cost_model.observe(
            context_tokens.len(),
            context_tokens,
            target_rows,
            observed_ms,
            route_critical_unique_experts,
        )
    }

    fn install_runtime_cost_profile(
        &mut self,
        request_count: usize,
        rows: &[(usize, f64)],
    ) -> Result<()> {
        self.cost_model.install_profile(request_count, rows)
    }

    fn observe_verification(
        &mut self,
        sequence_id: &str,
        plan: &DsparkDraftPlan,
        accepted_drafts: usize,
    ) -> (f64, f64, usize) {
        let Some(request) = self.requests.get_mut(sequence_id) else {
            return (0.0, 0.0, 0);
        };
        if self.engine.is_dflash2() {
            request
                .dflash_adaptive_draft
                .observe(plan.selected_drafts, accepted_drafts);
        }
        if plan.calibration_eligible && accepted_drafts <= plan.conditional_confidence.len() {
            match self.confidence_policy {
                RealFullDsparkConfidencePolicy::Calibrated => request
                    .confidence_calibrator
                    .observe(&plan.conditional_confidence, accepted_drafts),
                RealFullDsparkConfidencePolicy::Residual => request.confidence_residual.observe(
                    &plan.conditional_confidence,
                    accepted_drafts,
                    plan.confidence_context_tokens,
                ),
                RealFullDsparkConfidencePolicy::Raw => {}
            }
        }
        match self.confidence_policy {
            RealFullDsparkConfidencePolicy::Residual => (
                request
                    .confidence_residual
                    .global_logit_bias(plan.confidence_context_tokens),
                0.0,
                request.confidence_residual.observation_cycles(),
            ),
            RealFullDsparkConfidencePolicy::Calibrated | RealFullDsparkConfidencePolicy::Raw => (
                request.confidence_calibrator.logit_bias(),
                request.confidence_calibrator.posterior_variance(),
                request.confidence_calibrator.observation_cycles(),
            ),
        }
    }

    fn request_context_tokens(&self, sequence_id: &str) -> Option<usize> {
        self.requests
            .get(sequence_id)
            .map(|request| request.cache.context_tokens())
    }

    fn publish_reusable_prefix(
        &mut self,
        sequence_id: &str,
        prompt_token_ids: &[usize],
        prefix_tokens: usize,
    ) -> Result<bool> {
        anyhow::ensure!(
            prefix_tokens <= prompt_token_ids.len(),
            "dSpark reusable prefix {prefix_tokens} exceeds prompt length {}",
            prompt_token_ids.len(),
        );
        let request = self
            .requests
            .get(sequence_id)
            .with_context(|| format!("dSpark request state is missing for {sequence_id}"))?;
        let Some(snapshot) = self
            .engine
            .snapshot_request_state_at_prefix(&request.cache, prefix_tokens)?
        else {
            return Ok(false);
        };
        anyhow::ensure!(
            snapshot.context_tokens() == prefix_tokens,
            "dSpark reusable snapshot ends at token {}, expected {prefix_tokens}",
            snapshot.context_tokens(),
        );
        let key = RealFullDsparkTailKey {
            prefix_tokens,
            prefix_sha256: real_full_dspark_prefix_fingerprint(&prompt_token_ids[..prefix_tokens]),
        };
        let cache_tokens = snapshot.cache_context_tokens();
        let snapshot_bytes = snapshot.resident_bytes();
        let retained = self.tail_cache.insert(RealFullDsparkTailEntry {
            key,
            snapshot,
            confidence_calibrator: request.confidence_calibrator.clone(),
            confidence_residual: request.confidence_residual.clone(),
        });
        eprintln!(
            "real_full_dspark_prefix_publish sequence_id={} prefix_tokens={} cache_tokens={} snapshot_bytes={} retained={} cached_entries={} cached_bytes={} cache_limit_bytes={}",
            sequence_id,
            prefix_tokens,
            cache_tokens,
            snapshot_bytes,
            retained,
            self.tail_cache.entries.len(),
            self.tail_cache.resident_bytes,
            self.tail_cache.max_bytes,
        );
        Ok(retained)
    }

    fn finish_sequence(
        &mut self,
        sequence_id: &str,
        committed_token_ids: Option<&[usize]>,
    ) -> Result<Option<usize>> {
        let Some(request) = self.requests.remove(sequence_id) else {
            return Ok(None);
        };
        let RealFullDsparkRequestRuntime {
            cache,
            confidence_calibrator,
            confidence_residual,
            pending_windows,
            ..
        } = request;
        for window in &pending_windows {
            Self::trace_shadow_window(sequence_id, window, "request_end", None);
        }
        let snapshot_result = if let Some(committed_token_ids) = committed_token_ids {
            self.engine
                .snapshot_request_state(&cache)
                .and_then(|snapshot| {
                    let Some(snapshot) = snapshot else {
                        return Ok(None);
                    };
                    anyhow::ensure!(
                        snapshot.context_tokens() <= committed_token_ids.len(),
                        "dSpark tail ends at token {} but the target committed frontier has only {} tokens",
                        snapshot.context_tokens(),
                        committed_token_ids.len(),
                    );
                    let key = RealFullDsparkTailKey {
                        prefix_tokens: snapshot.context_tokens(),
                        prefix_sha256: real_full_dspark_prefix_fingerprint(
                            &committed_token_ids[..snapshot.context_tokens()],
                        ),
                    };
                    let snapshot_bytes = snapshot.resident_bytes();
                    let cache_tokens = snapshot.cache_context_tokens();
                    let retained = self.tail_cache.insert(RealFullDsparkTailEntry {
                        key,
                        snapshot,
                        confidence_calibrator,
                        confidence_residual,
                    });
                    eprintln!(
                        "real_full_dspark_tail_publish sequence_id={} prefix_tokens={} cache_tokens={} snapshot_bytes={} retained={} cached_entries={} cached_bytes={} cache_limit_bytes={}",
                        sequence_id,
                        key.prefix_tokens,
                        cache_tokens,
                        snapshot_bytes,
                        retained,
                        self.tail_cache.entries.len(),
                        self.tail_cache.resident_bytes,
                        self.tail_cache.max_bytes,
                    );
                    Ok(retained.then_some(key.prefix_tokens))
                })
        } else {
            Ok(None)
        };
        self.engine.release_request_state(cache);
        snapshot_result
    }
}

struct RealFullKvSnapshotSave {
    root: PathBuf,
    token_count: Option<usize>,
}

struct RealFullContextTokenBudget {
    max_tokens: usize,
    inner: Mutex<RealFullContextTokenBudgetInner>,
}

struct RealFullContextTokenBudgetInner {
    free_extents: Vec<RealFullContextTokenExtent>,
    used_tokens: usize,
    active_reservations: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RealFullContextTokenExtent {
    token_base: usize,
    tokens: usize,
}

impl RealFullContextTokenBudget {
    fn new(max_tokens: usize) -> Self {
        debug_assert!(max_tokens > 0);
        Self {
            max_tokens,
            inner: Mutex::new(RealFullContextTokenBudgetInner {
                free_extents: vec![RealFullContextTokenExtent {
                    token_base: 0,
                    tokens: max_tokens,
                }],
                used_tokens: 0,
                active_reservations: 0,
            }),
        }
    }

    fn reserve(self: &Arc<Self>, tokens: usize) -> Result<RealFullContextTokenReservation> {
        anyhow::ensure!(tokens > 0, "real-full context reservation is empty");
        let reserved_tokens = tokens
            .checked_add(REAL_FULL_SHARED_KV_PAGE_TOKENS - 1)
            .context("real-full context reservation page rounding overflow")?
            / REAL_FULL_SHARED_KV_PAGE_TOKENS
            * REAL_FULL_SHARED_KV_PAGE_TOKENS;
        let mut inner = self
            .inner
            .lock()
            .map_err(|error| anyhow::anyhow!("locking real-full context budget failed: {error}"))?;
        anyhow::ensure!(
            inner.active_reservations < REAL_FULL_MAX_ACTIVE_REQUESTS,
            "real-full active request limit exhausted: active={} max={REAL_FULL_MAX_ACTIVE_REQUESTS}",
            inner.active_reservations
        );
        let extent_index = inner
            .free_extents
            .iter()
            .position(|extent| extent.tokens >= reserved_tokens)
            .with_context(|| {
                format!(
                    "real-full global context budget exhausted: requested_sequence_capacity={tokens} reserved_tokens={reserved_tokens} used={} max={}",
                    inner.used_tokens, self.max_tokens
                )
            })?;
        let token_base = inner.free_extents[extent_index].token_base;
        if inner.free_extents[extent_index].tokens == reserved_tokens {
            inner.free_extents.remove(extent_index);
        } else {
            inner.free_extents[extent_index].token_base += reserved_tokens;
            inner.free_extents[extent_index].tokens -= reserved_tokens;
        }
        inner.used_tokens = inner
            .used_tokens
            .checked_add(reserved_tokens)
            .context("real-full context token accounting overflow")?;
        inner.active_reservations += 1;
        Ok(RealFullContextTokenReservation {
            budget: Arc::clone(self),
            token_base,
            reserved_tokens,
        })
    }

    fn release(&self, extent: RealFullContextTokenExtent) {
        let mut inner = self
            .inner
            .lock()
            .expect("real-full context budget lock poisoned during release");
        let insert_at = inner
            .free_extents
            .partition_point(|candidate| candidate.token_base < extent.token_base);
        inner.free_extents.insert(insert_at, extent);
        let mut index = insert_at.saturating_sub(1);
        while index + 1 < inner.free_extents.len() {
            let left = inner.free_extents[index];
            let right = inner.free_extents[index + 1];
            if left.token_base.checked_add(left.tokens) == Some(right.token_base) {
                inner.free_extents[index].tokens = left
                    .tokens
                    .checked_add(right.tokens)
                    .expect("coalesced real-full context extent overflows usize");
                inner.free_extents.remove(index + 1);
            } else {
                index += 1;
            }
        }
        inner.used_tokens = inner
            .used_tokens
            .checked_sub(extent.tokens)
            .expect("real-full context token accounting underflow");
        inner.active_reservations = inner
            .active_reservations
            .checked_sub(1)
            .expect("real-full active request accounting underflow");
    }
}

struct RealFullContextTokenReservation {
    budget: Arc<RealFullContextTokenBudget>,
    token_base: usize,
    reserved_tokens: usize,
}

impl RealFullContextTokenReservation {
    fn token_base(&self) -> usize {
        self.token_base
    }
}

impl Drop for RealFullContextTokenReservation {
    fn drop(&mut self) {
        self.budget.release(RealFullContextTokenExtent {
            token_base: self.token_base,
            tokens: self.reserved_tokens,
        });
    }
}

struct BudgetedRealFullSchedulerExecutionState {
    state: RealFullSchedulerExecutionState,
    execution_lane_id: usize,
    _context_reservation: Option<RealFullContextTokenReservation>,
    target_radix_reservation: Option<TargetKvRadixReservation>,
    bound_target_pages: usize,
    graph_bound_arena: bool,
    snapshot_save_ready: bool,
    snapshot_restore_ms: f64,
    constraint: Option<RealFullConstraintState>,
}

impl Deref for BudgetedRealFullSchedulerExecutionState {
    type Target = RealFullSchedulerExecutionState;

    fn deref(&self) -> &Self::Target {
        &self.state
    }
}

impl DerefMut for BudgetedRealFullSchedulerExecutionState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.state
    }
}

impl BudgetedRealFullSchedulerExecutionState {
    fn bind_target_radix_reservation(
        &mut self,
        mut reservation: TargetKvRadixReservation,
        prompt_token_ids: &[usize],
    ) -> Result<()> {
        let matched_prefix_tokens = reservation.matched_prefix_tokens();
        anyhow::ensure!(
            matched_prefix_tokens <= prompt_token_ids.len(),
            "target KV radix match {matched_prefix_tokens} exceeds prompt token count {}",
            prompt_token_ids.len()
        );
        let logical_capacity_tokens = reservation.logical_capacity_tokens();
        reservation
            .ensure_materialized_through(prompt_token_ids.len().max(1).min(logical_capacity_tokens))
            .context("materializing initial target KV radix page table")?;
        self.state
            .rebind_sequence_physical_pages(reservation.physical_pages(), logical_capacity_tokens)
            .context("binding scheduler state to target KV radix pages")?;
        if let Some(boundary) = reservation.take_boundary_copy() {
            self.state
                .copy_target_kv_boundary_page(
                    boundary.source_page,
                    boundary.destination_page,
                    boundary.valid_tokens,
                )
                .context("copying target KV radix branch boundary")?;
        }
        self.state
            .seed_processed_token_ids(&prompt_token_ids[..matched_prefix_tokens])
            .context("seeding target KV radix processed-token frontier")?;
        self.bound_target_pages = reservation.physical_pages().len();
        self.target_radix_reservation = Some(reservation);
        Ok(())
    }

    fn ensure_target_radix_materialized_through(&mut self, tokens: usize) -> Result<()> {
        let Some(reservation) = self.target_radix_reservation.as_mut() else {
            return Ok(());
        };
        let capacity_tokens = reservation.logical_capacity_tokens();
        let pages = reservation
            .ensure_materialized_through(tokens.min(capacity_tokens))
            .context("materializing target KV radix pages for decode cycle")?;
        if pages.len() != self.bound_target_pages {
            self.state
                .extend_sequence_physical_pages(pages, capacity_tokens)
                .context("publishing extended target KV radix page table to device")?;
            self.bound_target_pages = pages.len();
        }
        Ok(())
    }
}

fn retain_graph_bound_scheduler_arena(
    graph_bound_arena: bool,
    arena_capacity_tokens: usize,
    max_context_tokens: usize,
) -> bool {
    graph_bound_arena && arena_capacity_tokens == max_context_tokens
}

#[derive(Debug, PartialEq, Eq)]
struct RealFullRequestTokenRows {
    prefix_tokens: usize,
    prefill_tokens: usize,
    prefill_token_ids: Option<Vec<usize>>,
    decode_token_ids: Vec<usize>,
}

struct PreparedBatchedDsparkCycle {
    request: glmrt_api::RealFullRequest,
    request_start: Instant,
    request_timing: bool,
    sequence_id: String,
    generated_tokens: usize,
    request_id_base: u64,
    token_prefix_tokens: usize,
    token_prefill_tokens: usize,
    committed_input_token_ids: Vec<usize>,
    decode_rows: usize,
    scheduler_start: Instant,
    shape: RealFullSchedulerExecutionShape,
    state: BudgetedRealFullSchedulerExecutionState,
    buffer_bank: usize,
    pending_dspark_plan: Option<DsparkDraftPlan>,
    pending_dspark_draft_token_ids: Vec<usize>,
    dspark_target_hidden_tap_rows: usize,
    snapshot_restore_ms: f64,
}

struct RealFullSpeculativeTerminalSample {
    vocab_size: usize,
    top_token_id: usize,
    sampled_token_id: usize,
    sample_top_k: usize,
    sample_top_p: f32,
    argmax_backend: &'static str,
    sampler_backend: &'static str,
    accepted_draft_tokens: usize,
    report_mtp_acceptance: bool,
}

fn apply_speculative_terminal_sample_to_report(
    report: &mut RealFullSchedulerExecutionDryRun,
    sample: &RealFullSpeculativeTerminalSample,
) {
    report.terminal_lm_head_sample = RealFullSchedulerTerminalLmHeadSample {
        status: "sampled",
        scope: "sample the terminal row from the retained speculative target hidden batch",
        uses_final_decode_device_hidden: true,
        covers_full_vocabulary: true,
        hidden_dim: glmrt_core::GLM52_HIDDEN_SIZE,
        vocab_size: sample.vocab_size,
        logits_evaluated: sample.vocab_size,
        top_token_id: Some(sample.top_token_id),
        sampled_token_id: Some(sample.sampled_token_id),
        sample_top_k: Some(sample.sample_top_k),
        sample_top_p: Some(sample.sample_top_p),
        argmax_kernel_backend: Some(sample.argmax_backend),
        sampler_kernel_backend: Some(sample.sampler_backend),
        passed: true,
        blocker: None,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RealFullMtpDraftPolicy {
    min: usize,
    max: usize,
    start: usize,
    adaptive: bool,
}

#[derive(Debug, PartialEq, Eq)]
struct RealFullMtpAcceptance {
    accepted_draft_tokens: usize,
    terminal_target_index: usize,
    full_match_bonus: bool,
}

fn constrain_dspark_plan(
    plan: &mut DsparkDraftPlan,
    constraint: &RealFullConstraintState,
) -> Result<()> {
    let valid = constraint
        .valid_draft_prefix(&plan.proposal_token_ids)
        .context("validating dSpark proposal against the active output grammar")?;
    plan.proposal_token_ids.truncate(valid);
    plan.conditional_confidence.truncate(valid);
    plan.selected_drafts = valid;
    plan.minimum_drafts = plan.minimum_drafts.min(valid);
    plan.target_batch_rows = valid + 1;
    Ok(())
}

impl RealFullSchedulerRequestExecutor {
    fn take_scheduler_state(
        &self,
        request: &glmrt_api::RealFullRequest,
        target_radix: Option<(TargetKvRadixReservation, &[usize])>,
    ) -> std::result::Result<BudgetedRealFullSchedulerExecutionState, String> {
        let sequence_id = request.sequence_id.as_str();
        let mut states = self
            .scheduler_states
            .lock()
            .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
        if let Some(mut state) = states.remove(sequence_id) {
            if target_radix.is_some() {
                states.insert(sequence_id.to_owned(), state);
                return Err(format!(
                    "active real-full sequence {sequence_id} received a second target KV radix reservation"
                ));
            }
            match (state.constraint.as_ref(), request.constraint.as_ref()) {
                (None, None) => {}
                (Some(active), Some(requested)) if active.matches_spec(requested) => {}
                _ => {
                    states.insert(sequence_id.to_owned(), state);
                    return Err(format!(
                        "active real-full sequence {sequence_id} changed its constrained-decoding specification"
                    ));
                }
            }
            drop(states);
            let materialization_tokens = request
                .prompt_tokens
                .checked_add(request.generated_token_ids.len())
                .and_then(|tokens| tokens.checked_add(REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS))
                .ok_or_else(|| "target KV materialization frontier overflow".to_owned())?;
            state
                .ensure_target_radix_materialized_through(materialization_tokens)
                .map_err(format_error_chain)?;
            return Ok(state);
        }
        drop(states);
        let capacity_tokens = real_full_sequence_capacity_tokens(
            request.prompt_tokens,
            request.decode_budget,
            self.kv_config.max_tokens,
        )
        .map_err(format_error_chain)?;
        let context_reservation =
            if real_full_internal_sequence(sequence_id) || target_radix.is_some() {
                None
            } else {
                Some(
                    self.context_budget
                        .reserve(capacity_tokens)
                        .map_err(format_error_chain)?,
                )
            };
        let physical_token_base = context_reservation
            .as_ref()
            .map_or(0, RealFullContextTokenReservation::token_base);
        let canonical_capture_arena = real_full_capture_arena_sequence(sequence_id)
            || !real_full_internal_sequence(sequence_id);
        let startup_execution_lane_id = real_full_batched_dspark_prewarm_buffer_bank(sequence_id)
            .map(|buffer_bank| {
                buffer_bank
                    .checked_sub(1)
                    .expect("batched dSpark startup buffer banks are nonzero")
            });
        if let Some(execution_lane_id) = startup_execution_lane_id {
            if execution_lane_id >= self.max_execution_lanes {
                return Err(format!(
                    "batched dSpark startup execution lane {execution_lane_id} exceeds configured lane count {}",
                    self.max_execution_lanes
                ));
            }
        }
        let mut state = if canonical_capture_arena {
            let mut recycled = self.recycled_scheduler_states.lock().map_err(|err| {
                format!("locking recycled real-full scheduler states failed: {err}")
            })?;
            if !real_full_internal_sequence(sequence_id) {
                recycled.retain(|candidate| {
                    !candidate.owned_by_current_thread()
                        || candidate.arena_capacity_tokens() == self.kv_config.max_tokens
                });
            }
            let candidate = recycled
                .iter()
                .enumerate()
                .filter(|(_, candidate)| {
                    candidate.owned_by_current_thread()
                        && candidate.arena_capacity_tokens() == self.kv_config.max_tokens
                })
                .filter(|(_, candidate)| {
                    startup_execution_lane_id
                        .is_none_or(|lane_id| candidate.execution_lane_id == lane_id)
                })
                .min_by_key(|(_, candidate)| candidate.execution_lane_id)
                .map(|(index, _)| index);
            candidate.map(|index| recycled.swap_remove(index))
        } else {
            None
        };
        if state.is_none() && canonical_capture_arena && startup_execution_lane_id.is_none() {
            let mut states = self
                .scheduler_states
                .lock()
                .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
            let candidate_key = states
                .iter()
                .find(|(candidate_sequence_id, candidate)| {
                    real_full_capture_arena_sequence(candidate_sequence_id)
                        && candidate.owned_by_current_thread()
                        && candidate.arena_capacity_tokens() == self.kv_config.max_tokens
                })
                .map(|(candidate_sequence_id, _)| candidate_sequence_id.clone());
            state = candidate_key
                .and_then(|candidate_sequence_id| states.remove(&candidate_sequence_id));
        }
        let mut new_execution_lane_id = startup_execution_lane_id.unwrap_or(0);
        if state.is_none() && !real_full_internal_sequence(sequence_id) {
            let mut occupied_execution_lanes = vec![false; self.max_execution_lanes];
            self
                .scheduler_states
                .lock()
                .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?
                .values()
                .filter(|candidate| {
                    retain_graph_bound_scheduler_arena(
                        candidate.graph_bound_arena,
                        candidate.arena_capacity_tokens(),
                        self.kv_config.max_tokens,
                    )
                })
                .try_for_each(|candidate| {
                    let occupied = occupied_execution_lanes
                        .get_mut(candidate.execution_lane_id)
                        .ok_or_else(|| {
                            format!(
                                "resident real-full execution lane {} exceeds configured lane count {}",
                                candidate.execution_lane_id, self.max_execution_lanes
                            )
                        })?;
                    *occupied = true;
                    Ok::<(), String>(())
                })?;
            self
                .recycled_scheduler_states
                .lock()
                .map_err(|err| {
                    format!("locking recycled real-full scheduler states failed: {err}")
                })?
                .iter()
                .filter(|candidate| {
                    retain_graph_bound_scheduler_arena(
                        candidate.graph_bound_arena,
                        candidate.arena_capacity_tokens(),
                        self.kv_config.max_tokens,
                    )
                })
                .try_for_each(|candidate| {
                    let occupied = occupied_execution_lanes
                        .get_mut(candidate.execution_lane_id)
                        .ok_or_else(|| {
                            format!(
                                "recycled real-full execution lane {} exceeds configured lane count {}",
                                candidate.execution_lane_id, self.max_execution_lanes
                            )
                        })?;
                    *occupied = true;
                    Ok::<(), String>(())
                })?;
            new_execution_lane_id = occupied_execution_lanes
                .iter()
                .position(|occupied| !occupied)
                .ok_or_else(|| format!(
                    "all {} configured real-full execution lanes are resident; request must remain pending",
                    self.max_execution_lanes
                ))?;
        }
        let mut state = match state {
            Some(mut state) => {
                state
                    .rebind_sequence(sequence_id.to_owned(), capacity_tokens, physical_token_base)
                    .map_err(format_error_chain)?;
                state
            }
            None => {
                let arena_capacity_tokens = if canonical_capture_arena {
                    self.kv_config.max_tokens
                } else {
                    capacity_tokens
                };
                let shared_storage = self
                    .device_kv_storage
                    .lock()
                    .map_err(|err| format!("locking shared device KV storage failed: {err}"))?
                    .clone();
                BudgetedRealFullSchedulerExecutionState {
                    state: RealFullSchedulerExecutionState::new_with_arena_capacity_and_storage(
                        self.kv_config.clone(),
                        sequence_id.to_owned(),
                        capacity_tokens,
                        arena_capacity_tokens,
                        self.device_kv_pool_config.clone(),
                        shared_storage,
                        physical_token_base,
                    )
                    .map_err(format_error_chain)?,
                    execution_lane_id: new_execution_lane_id,
                    _context_reservation: None,
                    target_radix_reservation: None,
                    bound_target_pages: 0,
                    graph_bound_arena: canonical_capture_arena,
                    snapshot_save_ready: false,
                    snapshot_restore_ms: 0.0,
                    constraint: None,
                }
            }
        };
        state.constraint = request
            .constraint
            .as_ref()
            .map(|spec| self.constraint_compiler.matcher(Arc::clone(spec)))
            .transpose()
            .map_err(format_error_chain)?;
        state._context_reservation = context_reservation;
        state.target_radix_reservation = None;
        state.bound_target_pages = 0;
        state.graph_bound_arena = canonical_capture_arena;
        state.snapshot_save_ready = false;
        state.snapshot_restore_ms = 0.0;
        {
            let mut shared_storage = self
                .device_kv_storage
                .lock()
                .map_err(|err| format!("locking shared device KV storage failed: {err}"))?;
            if shared_storage.is_none() {
                *shared_storage = state.device_kv_storage_handle();
            }
        }
        if !real_full_internal_sequence(sequence_id)
            && request.cached_prompt_tokens > 0
            && target_radix.is_none()
        {
            let snapshot = self.kv_snapshot_load.as_ref().ok_or_else(|| {
                "an external cached-prefix request requires GLMRT_REAL_FULL_KV_SNAPSHOT_LOAD"
                    .to_owned()
            })?;
            if request.cached_prompt_tokens != snapshot.token_count() {
                return Err(format!(
                    "cached-prefix request declares {} tokens but the loaded KV snapshot has {}",
                    request.cached_prompt_tokens,
                    snapshot.token_count()
                ));
            }
            let restore_start = Instant::now();
            snapshot
                .restore(&mut state.state)
                .map_err(format_error_chain)?;
            state.snapshot_restore_ms = elapsed_ms(restore_start);
            eprintln!(
                "real_full_kv_snapshot_restore path={} tokens={} elapsed_ms={:.3}",
                snapshot.root().display(),
                snapshot.token_count(),
                state.snapshot_restore_ms,
            );
        }
        if let Some((reservation, prompt_token_ids)) = target_radix {
            state
                .bind_target_radix_reservation(reservation, prompt_token_ids)
                .map_err(format_error_chain)?;
        }
        let materialization_tokens = request
            .prompt_tokens
            .checked_add(request.generated_token_ids.len())
            .and_then(|tokens| tokens.checked_add(REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS))
            .ok_or_else(|| "target KV materialization frontier overflow".to_owned())?;
        state
            .ensure_target_radix_materialized_through(materialization_tokens)
            .map_err(format_error_chain)?;
        Ok(state)
    }

    fn store_scheduler_state(
        &self,
        sequence_id: &str,
        state: BudgetedRealFullSchedulerExecutionState,
    ) -> std::result::Result<(), String> {
        let mut states = self
            .scheduler_states
            .lock()
            .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
        if states.insert(sequence_id.to_owned(), state).is_some() {
            return Err(format!(
                "real-full scheduler state for sequence {sequence_id} is already active"
            ));
        }
        Ok(())
    }

    fn finish_scheduler_state(
        &self,
        sequence_id: &str,
        mut state: BudgetedRealFullSchedulerExecutionState,
        final_decode_step: bool,
    ) -> std::result::Result<(), String> {
        if final_decode_step {
            let startup_radix_publish = real_full_startup_target_radix_publish_tokens(sequence_id)
                .is_some()
                && state.target_radix_reservation.is_some();
            if ((!self.kv_snapshot_saves.is_empty() || state.target_radix_reservation.is_some())
                && !real_full_internal_sequence(sequence_id))
                || startup_radix_publish
            {
                state.snapshot_save_ready = true;
                self.store_scheduler_state(sequence_id, state)
            } else {
                self.recycle_scheduler_state(state)
            }
        } else {
            self.store_scheduler_state(sequence_id, state)
        }
    }

    fn recycle_scheduler_state(
        &self,
        mut state: BudgetedRealFullSchedulerExecutionState,
    ) -> std::result::Result<(), String> {
        reset_glm_dsa_sparse_mla_transient_state().map_err(format_error_chain)?;
        state._context_reservation = None;
        state.target_radix_reservation = None;
        state.bound_target_pages = 0;
        // Only the max-context arena owns graph-bound KV/DSA addresses worth
        // retaining. Startup also creates short diagnostic states; pooling
        // those would keep one extra device arena per probe until the first
        // external request happened to prune them.
        if retain_graph_bound_scheduler_arena(
            state.graph_bound_arena,
            state.arena_capacity_tokens(),
            self.kv_config.max_tokens,
        ) {
            let mut recycled = self.recycled_scheduler_states.lock().map_err(|err| {
                format!("locking recycled real-full scheduler states failed: {err}")
            })?;
            recycled.push(state);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_mtp_draft_step(
        &self,
        source_request_id: u64,
        position_start: usize,
        request_id_base: u64,
        mtp_input_token_ids: &[usize],
        target_hidden: super::coordinator_kernels::DeviceBf16Output,
        greedy_sampling: bool,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        constraint_branch: Option<&mut RealFullConstraintBranch>,
        state: &mut RealFullSchedulerExecutionState,
    ) -> Result<(
        RealFullMtpDraftToken,
        super::coordinator_kernels::DeviceBf16Output,
    )> {
        let start = Instant::now();
        let envelope = real_full_mtp_envelope_device_hidden(
            &self.catalog,
            mtp_input_token_ids,
            &target_hidden,
            position_start,
        )
        .context("building real-full MTP draft envelope")?;
        let envelope_ms = elapsed_ms(start);
        let layer_start = Instant::now();
        let layer_hidden = real_full_scheduler_execute_prefill_decode_layer_block_device_input(
            &self.catalog,
            SparkLayerBlock::mtp(),
            source_request_id,
            position_start,
            envelope,
            real_full_mtp_prefill_chunk_tokens(),
            dispatch_worker,
            request_id_base,
            state,
        )
        .context("executing real-full MTP draft layer 78")?;
        let layer_ms = elapsed_ms(layer_start);
        let score_start = Instant::now();
        let (draft, recycle_hidden) = if let Some(constraint_branch) = constraint_branch {
            let masks = constraint_branch
                .next_mask()
                .context("building native MTP draft grammar mask")?;
            let scored = real_full_mtp_draft_token_constrained(
                &self.catalog,
                &layer_hidden,
                greedy_sampling,
                &masks,
            )
            .context("scoring constrained real-full MTP draft token")?;
            constraint_branch
                .accept(std::slice::from_ref(&scored.0.token_id))
                .context("advancing native MTP draft grammar state")?;
            scored
        } else {
            real_full_mtp_draft_token(&self.catalog, &layer_hidden, greedy_sampling)
                .context("scoring real-full MTP draft token")?
        };
        let score_ms = elapsed_ms(score_start);
        if real_full_request_timing_enabled() || real_full_mtp_probe_enabled() {
            eprintln!(
                "real_full_mtp_step source_request_id={} position_start={} rows={} final_position={} draft_token_id={} envelope_ms={:.3} layer_ms={:.3} score_ms={:.3} total_ms={:.3} logits={} top_logit={:.6} argmax_backend={}",
                source_request_id,
                position_start,
                mtp_input_token_ids.len(),
                position_start + mtp_input_token_ids.len() - 1,
                draft.token_id,
                envelope_ms,
                layer_ms,
                score_ms,
                elapsed_ms(start),
                draft.logits_evaluated,
                draft.top_logit,
                draft.argmax_backend,
            );
        }
        // GLM-5 uses the DeepSeek-style NextN recurrence: the shared-head
        // normalized hidden is both scored and fed into the next step's hnorm.
        Ok((draft, recycle_hidden))
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_mtp_prompt_prefix_step(
        &self,
        source_request_id: u64,
        position_start: usize,
        request_id_base: u64,
        mtp_input_token_ids: &[usize],
        target_hidden: super::coordinator_kernels::DeviceBf16Output,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        state: &mut RealFullSchedulerExecutionState,
    ) -> Result<()> {
        let start = Instant::now();
        let envelope = real_full_mtp_envelope_device_hidden(
            &self.catalog,
            mtp_input_token_ids,
            &target_hidden,
            position_start,
        )
        .context("building real-full MTP prompt-prefix envelope")?;
        let envelope_ms = elapsed_ms(start);
        let layer_start = Instant::now();
        real_full_scheduler_execute_prefill_decode_layer_block_device_input(
            &self.catalog,
            SparkLayerBlock::mtp(),
            source_request_id,
            position_start,
            envelope,
            real_full_mtp_prefill_chunk_tokens(),
            dispatch_worker,
            request_id_base,
            state,
        )
        .context("executing real-full MTP prompt-prefix layer 78")?;
        if real_full_request_timing_enabled() || real_full_mtp_probe_enabled() {
            eprintln!(
                "real_full_mtp_prompt_prefix_step source_request_id={} position_start={} rows={} final_position={} envelope_ms={:.3} layer_ms={:.3} total_ms={:.3}",
                source_request_id,
                position_start,
                mtp_input_token_ids.len(),
                position_start + mtp_input_token_ids.len() - 1,
                envelope_ms,
                elapsed_ms(layer_start),
                elapsed_ms(start),
            );
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_mtp_chain(
        &self,
        source_request_id: u64,
        position_start: usize,
        request_id_base: u64,
        initial_mtp_input_token_ids: &[usize],
        target_hidden: super::coordinator_kernels::DeviceBf16Output,
        draft_tokens: usize,
        greedy_sampling: bool,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        mut constraint_branch: Option<&mut RealFullConstraintBranch>,
        state: &mut RealFullSchedulerExecutionState,
    ) -> Result<Vec<RealFullMtpDraftToken>> {
        anyhow::ensure!(
            draft_tokens > 0 && !initial_mtp_input_token_ids.is_empty(),
            "real-full MTP chain requires non-empty input and a positive draft count"
        );
        if constraint_branch
            .as_deref()
            .map(RealFullConstraintBranch::is_completed)
            .transpose()?
            .unwrap_or(false)
        {
            return Ok(Vec::new());
        }
        let start = Instant::now();
        let initial_rows = initial_mtp_input_token_ids.len();
        // The target model exposes its final-norm hidden state to MTP. Recurrent
        // MTP outputs remain unnormalized until the next step's hnorm.
        let target_hidden = real_full_target_hidden_for_mtp(&target_hidden)
            .context("normalizing target hidden for real-full MTP draft chain")?;
        let max_prompt_rows =
            COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS[COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS.len() - 1];
        let mut initial_offset = 0_usize;
        while initial_rows - initial_offset > max_prompt_rows {
            let prefix_hidden =
                real_full_device_hidden_rows(&target_hidden, initial_offset, max_prompt_rows)
                    .context("slicing real-full MTP prompt-prefix target hidden")?;
            self.execute_mtp_prompt_prefix_step(
                source_request_id,
                position_start + initial_offset,
                request_id_base
                    .saturating_add(REAL_FULL_MTP_REQUEST_ID_OFFSET)
                    .saturating_add(initial_offset as u64),
                &initial_mtp_input_token_ids[initial_offset..initial_offset + max_prompt_rows],
                prefix_hidden,
                Arc::clone(&dispatch_worker),
                state,
            )?;
            initial_offset += max_prompt_rows;
        }
        let final_initial_hidden = real_full_device_hidden_rows(
            &target_hidden,
            initial_offset,
            initial_rows - initial_offset,
        )
        .context("slicing final real-full MTP prompt target hidden")?;
        let (first, previous_hidden) = self.execute_mtp_draft_step(
            source_request_id,
            position_start + initial_offset,
            request_id_base
                .saturating_add(REAL_FULL_MTP_REQUEST_ID_OFFSET)
                .saturating_add(initial_offset as u64),
            &initial_mtp_input_token_ids[initial_offset..],
            final_initial_hidden,
            greedy_sampling,
            Arc::clone(&dispatch_worker),
            constraint_branch.as_deref_mut(),
            state,
        )?;
        let mut drafts = Vec::with_capacity(draft_tokens);
        let first_token_id = first.token_id;
        drafts.push(first);
        if constraint_branch
            .as_deref()
            .map(RealFullConstraintBranch::is_completed)
            .transpose()?
            .unwrap_or(false)
        {
            return Ok(drafts);
        }
        if draft_tokens > 1 {
            drafts.extend(self.execute_mtp_recurrent_chain(
                source_request_id,
                position_start + initial_rows,
                request_id_base,
                first_token_id,
                previous_hidden,
                draft_tokens - 1,
                1,
                greedy_sampling,
                Arc::clone(&dispatch_worker),
                constraint_branch.as_deref_mut(),
                state,
            )?);
        }
        if real_full_request_timing_enabled() || real_full_mtp_probe_enabled() {
            eprintln!(
                "real_full_mtp_chain source_request_id={} position_start={} initial_rows={} drafts={} elapsed_ms={:.3}",
                source_request_id,
                position_start,
                initial_rows,
                drafts.len(),
                elapsed_ms(start),
            );
        }
        Ok(drafts)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_mtp_bridge_chain(
        &self,
        source_request_id: u64,
        position_start: usize,
        request_id_base: u64,
        bridge_input_token_id: usize,
        accepted_bonus_token_id: usize,
        target_hidden: super::coordinator_kernels::DeviceBf16Output,
        draft_tokens: usize,
        greedy_sampling: bool,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        mut constraint_branch: Option<&mut RealFullConstraintBranch>,
        state: &mut RealFullSchedulerExecutionState,
    ) -> Result<Vec<RealFullMtpDraftToken>> {
        anyhow::ensure!(
            draft_tokens > 0 && target_hidden.rows == 1,
            "real-full MTP bridge requires one target-hidden row and a positive draft count"
        );
        if constraint_branch
            .as_deref()
            .map(RealFullConstraintBranch::is_completed)
            .transpose()?
            .unwrap_or(false)
        {
            return Ok(Vec::new());
        }
        let start = Instant::now();
        let target_hidden = real_full_target_hidden_for_mtp(&target_hidden)
            .context("normalizing target hidden for real-full MTP bridge")?;
        let (bridge_draft, bridge_hidden) = self.execute_mtp_draft_step(
            source_request_id,
            position_start,
            request_id_base.saturating_add(REAL_FULL_MTP_REQUEST_ID_OFFSET),
            &[bridge_input_token_id],
            target_hidden,
            greedy_sampling,
            Arc::clone(&dispatch_worker),
            None,
            state,
        )?;
        let drafts = self.execute_mtp_recurrent_chain(
            source_request_id,
            position_start + 1,
            request_id_base,
            accepted_bonus_token_id,
            bridge_hidden,
            draft_tokens,
            1,
            greedy_sampling,
            dispatch_worker,
            constraint_branch.as_deref_mut(),
            state,
        )?;
        if real_full_request_timing_enabled() || real_full_mtp_probe_enabled() {
            eprintln!(
                "real_full_mtp_bridge source_request_id={} position_start={} bridge_draft_token_id={} accepted_bonus_token_id={} drafts={} elapsed_ms={:.3}",
                source_request_id,
                position_start,
                bridge_draft.token_id,
                accepted_bonus_token_id,
                drafts.len(),
                elapsed_ms(start),
            );
        }
        Ok(drafts)
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_mtp_recurrent_chain(
        &self,
        source_request_id: u64,
        position_start: usize,
        request_id_base: u64,
        initial_token_id: usize,
        initial_hidden: super::coordinator_kernels::DeviceBf16Output,
        draft_tokens: usize,
        request_step_start: usize,
        greedy_sampling: bool,
        dispatch_worker: Arc<RealFullSchedulerSparseTcpDispatchWorker>,
        mut constraint_branch: Option<&mut RealFullConstraintBranch>,
        state: &mut RealFullSchedulerExecutionState,
    ) -> Result<Vec<RealFullMtpDraftToken>> {
        let mut drafts = Vec::with_capacity(draft_tokens);
        let mut previous_token_id = initial_token_id;
        let mut previous_hidden = initial_hidden;
        for draft_offset in 0..draft_tokens {
            let request_step = request_step_start.saturating_add(draft_offset);
            let (draft, layer_hidden) = self.execute_mtp_draft_step(
                source_request_id,
                position_start + draft_offset,
                request_id_base
                    .saturating_add(REAL_FULL_MTP_REQUEST_ID_OFFSET)
                    .saturating_add(
                        (request_step as u64).saturating_mul(REAL_FULL_MTP_REQUEST_ID_STEP_STRIDE),
                    ),
                &[previous_token_id],
                previous_hidden,
                greedy_sampling,
                Arc::clone(&dispatch_worker),
                constraint_branch.as_deref_mut(),
                state,
            )?;
            previous_token_id = draft.token_id;
            previous_hidden = layer_hidden;
            drafts.push(draft);
            if constraint_branch
                .as_deref()
                .map(RealFullConstraintBranch::is_completed)
                .transpose()?
                .unwrap_or(false)
            {
                break;
            }
        }
        Ok(drafts)
    }
}

impl RealFullSchedulerRequestExecutor {
    fn profile_batched_dspark_sps(
        &self,
        sequence_ids: &[String],
        prompts: &[String],
        prompt_tokens: &[usize],
        decode_budget: usize,
        generated_token_ids: &mut [Vec<usize>],
        max_draft_tokens: usize,
        mode: RealFullDsparkStartupProfileMode,
        measured_samples: usize,
    ) -> std::result::Result<(), String> {
        debug_assert_ne!(mode, RealFullDsparkStartupProfileMode::Disabled);
        debug_assert_eq!(sequence_ids.len(), prompts.len());
        debug_assert_eq!(sequence_ids.len(), prompt_tokens.len());
        debug_assert_eq!(sequence_ids.len(), generated_token_ids.len());
        let profile_start = Instant::now();
        eprintln!(
            "real_full_dspark_sps_profile_start lanes={} samples={} source=startup-opt-in mode={mode:?}",
            self.max_execution_lanes, measured_samples,
        );
        for request_count in 1..=self.max_execution_lanes {
            let mut rows = Vec::with_capacity(request_count * max_draft_tokens + 1);
            for target_rows in request_count..=request_count * (max_draft_tokens + 1) {
                let total_drafts = target_rows - request_count;
                let base_drafts = total_drafts / request_count;
                let extra_drafts = total_drafts % request_count;
                let widths = (0..request_count)
                    .map(|lane_index| base_drafts + usize::from(lane_index < extra_drafts))
                    .collect::<Vec<_>>();
                debug_assert!(widths.iter().all(|width| *width <= max_draft_tokens));
                let mut samples = Vec::with_capacity(measured_samples);
                let mut sample_attempt = 0usize;
                let maximum_attempts = measured_samples.saturating_add(64).saturating_add(1);
                while samples.len() < measured_samples {
                    sample_attempt = sample_attempt.saturating_add(1);
                    if sample_attempt > maximum_attempts {
                        return Err(format!(
                            "dSpark SPS profile C={request_count} rows={target_rows} collected only {} clean samples in {maximum_attempts} attempts",
                            samples.len(),
                        ));
                    }
                    let lane_indices = (0..request_count)
                        .map(|offset| (sample_attempt - 1 + offset) % sequence_ids.len())
                        .collect::<Vec<_>>();
                    let requests = lane_indices
                        .iter()
                        .zip(&widths)
                        .enumerate()
                        .map(|(request_index, (lane_index, draft_tokens))| {
                            glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                                REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE
                                    + (*draft_tokens as u64)
                                        * REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE
                                    + request_index as u64,
                                &sequence_ids[*lane_index],
                                &prompts[*lane_index],
                                prompt_tokens[*lane_index],
                                1,
                                generated_token_ids[*lane_index].clone(),
                                generated_token_ids[*lane_index].len(),
                                decode_budget,
                            )
                        })
                        .collect::<Vec<_>>();
                    let sample_start = Instant::now();
                    let cycles =
                        glmrt_api::RealFullRequestExecutor::execute_real_full_decode_cycle_batch(
                            self, requests,
                        )
                        .into_iter()
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let observed_ms = elapsed_ms(sample_start);
                    for (lane_index, (cycle, expected_drafts)) in
                        cycles.iter().zip(&widths).enumerate()
                    {
                        if cycle.info.status != "ready"
                            || cycle.info.request_mtp_verify_rows != *expected_drafts
                        {
                            return Err(format!(
                                "dSpark SPS profile C={request_count} rows={target_rows} lane={lane_index} failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
                                cycle.info.status,
                                cycle.info.request_mtp_verify_rows,
                                expected_drafts,
                                cycle.info.blocker,
                                cycle.info.failed_requirements,
                            ));
                        }
                    }
                    let graph_captures = cycles
                        .iter()
                        .map(|cycle| cycle.info.request_coordinator_graph_captures)
                        .max()
                        .unwrap_or(0);
                    for (lane_index, cycle) in lane_indices.into_iter().zip(cycles) {
                        generated_token_ids[lane_index].extend(
                            cycle
                                .generated_tokens
                                .into_iter()
                                .map(|token| token.token_id),
                        );
                    }
                    if sample_attempt == 1 {
                        continue;
                    }
                    if graph_captures != 0 {
                        eprintln!(
                            "real_full_dspark_sps_profile_rewarm requests={} target_rows={} graph_captures={} attempt={} source=startup-opt-in",
                            request_count, target_rows, graph_captures, sample_attempt,
                        );
                    } else {
                        samples.push(observed_ms);
                    }
                }
                samples.sort_by(f64::total_cmp);
                let median_ms = if measured_samples % 2 == 0 {
                    (samples[measured_samples / 2 - 1] + samples[measured_samples / 2]) * 0.5
                } else {
                    samples[measured_samples / 2]
                };
                rows.push((target_rows, median_ms));
                eprintln!(
                    "real_full_dspark_sps_profile requests={} target_rows={} latency_ms={:.3} samples={} source=startup-opt-in",
                    request_count, target_rows, median_ms, measured_samples,
                );
            }
            if mode == RealFullDsparkStartupProfileMode::Install {
                self.dspark
                    .as_ref()
                    .expect("dSpark SPS profiling requires a runtime")
                    .lock()
                    .map_err(|error| format!("locking dSpark SPS profile failed: {error}"))?
                    .install_runtime_cost_profile(request_count, &rows)
                    .map_err(format_error_chain)?;
            }
        }
        eprintln!(
            "real_full_dspark_sps_profile_done lanes={} samples={} elapsed_ms={:.3} source=startup-opt-in mode={mode:?}",
            self.max_execution_lanes,
            measured_samples,
            elapsed_ms(profile_start),
        );
        Ok(())
    }

    fn batched_dspark_cycles_eligible(&self, requests: &[glmrt_api::RealFullRequest]) -> bool {
        let batched_startup_prewarm = requests
            .iter()
            .all(|request| real_full_batched_dspark_prewarm_sequence(&request.sequence_id));
        if !(2..=8).contains(&requests.len())
            || self.sparse_tcp_dispatch_worker.is_none()
            || real_full_mtp_enabled()
            || self.kv_snapshot_load.is_some()
            || !self.kv_snapshot_saves.is_empty()
            || requests.iter().any(|request| {
                request.generated_token_ids.is_empty()
                    || !request.greedy_sampling
                    || request.disable_speculation
                    || request.constraint.is_some()
                    || (real_full_internal_sequence(&request.sequence_id)
                        && !batched_startup_prewarm)
            })
        {
            return false;
        }
        self.dspark.as_ref().is_some_and(|runtime| {
            runtime
                .lock()
                .map(|runtime| runtime.mode == RealFullDsparkServingMode::Active)
                .unwrap_or(false)
        })
    }

    fn prepare_batched_dspark_cycle(
        &self,
        request: glmrt_api::RealFullRequest,
    ) -> std::result::Result<PreparedBatchedDsparkCycle, String> {
        prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker().map_err(format_error_chain)?;
        let request_start = Instant::now();
        let request_timing = real_full_request_timing_enabled();
        let sequence_id = request.sequence_id.clone();
        let generated_tokens = request.generated_token_ids.len();
        let request_id_base = request.request_index.saturating_mul(1_000_000);
        let token_rows =
            real_full_request_token_rows(&request, None).map_err(format_error_chain)?;
        let decode_rows = token_rows.decode_token_ids.len();
        let prefill_chunk_tokens = real_full_prefill_chunk_tokens_for_direct_dsa(
            real_full_request_prefill_chunk_tokens_for_sequence(
                &sequence_id,
                token_rows.prefix_tokens,
                token_rows.prefill_tokens,
            ),
        );
        let dspark_mode = self
            .dspark
            .as_ref()
            .context("paired dSpark cycle requires a request runtime")
            .map_err(format_error_chain)?
            .lock()
            .map(|runtime| runtime.mode)
            .map_err(|error| format!("locking dSpark request executor failed: {error}"))?;
        if dspark_mode != RealFullDsparkServingMode::Active {
            return Err("paired dSpark cycle requires active serving mode".to_owned());
        }
        let mut pending_dspark_plan = self
            .dspark
            .as_ref()
            .expect("paired dSpark cycle resolved its runtime")
            .lock()
            .map_err(|error| format!("locking dSpark request executor failed: {error}"))?
            .prepare_cycle(
                &sequence_id,
                false,
                None,
                real_full_batched_dspark_prewarm_requested_draft_tokens(
                    &sequence_id,
                    request.request_index,
                )
                .or_else(|| real_full_dspark_startup_draft_tokens(&sequence_id)),
            )
            .map_err(format_error_chain)?;
        if let Some(plan) = pending_dspark_plan.as_mut() {
            let max_useful_drafts = request
                .decode_budget
                .saturating_sub(generated_tokens)
                .saturating_sub(1);
            plan.proposal_token_ids.truncate(max_useful_drafts);
            plan.selected_drafts = plan.proposal_token_ids.len();
            plan.target_batch_rows = plan.selected_drafts + 1;
        }
        let pending_dspark_draft_token_ids = pending_dspark_plan
            .as_ref()
            .map(|plan| plan.proposal_token_ids.clone())
            .unwrap_or_default();
        let mtp_rows = pending_dspark_draft_token_ids.len();
        real_full_validate_sparse_wave_capacity(prefill_chunk_tokens, decode_rows, mtp_rows)
            .map_err(format_error_chain)?;
        let dspark_target_hidden_tap_rows =
            if request.decode_budget.saturating_sub(generated_tokens) > 1 {
                decode_rows + mtp_rows
            } else {
                0
            };
        let mut scheduler_token_ids = token_rows.decode_token_ids.clone();
        let committed_input_token_ids = token_rows.decode_token_ids.clone();
        scheduler_token_ids.extend_from_slice(&pending_dspark_draft_token_ids);
        let shape = RealFullSchedulerExecutionShape {
            request_id: request.request_id.clone(),
            sequence_id: sequence_id.clone(),
            placement_version: format!("real-full-api-request-{}", request.request_index),
            prefix_tokens: token_rows.prefix_tokens,
            prefill_tokens: token_rows.prefill_tokens,
            prefill_chunk_tokens,
            decode_rows,
            mtp_rows,
            mtp_accepted_rows: 0,
            prefill_token_ids: token_rows.prefill_token_ids,
            prefill_vision_embeddings: None,
            decode_token_ids: Some(scheduler_token_ids),
            lm_head_sampling: request_lm_head_sampling_options(&request),
        };
        let state = match self.take_scheduler_state(&request, None) {
            Ok(state) => state,
            Err(error) => {
                if let Some(plan) = pending_dspark_plan.take() {
                    if let Some(dspark) = self.dspark.as_ref() {
                        if let Ok(mut dspark) = dspark.lock() {
                            dspark.restore_verification(&sequence_id, plan);
                        }
                    }
                }
                return Err(error);
            }
        };
        let snapshot_restore_ms = state.snapshot_restore_ms;
        let buffer_bank = state.execution_lane_id + 1;
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=paired_prepare total_ms={:.3} prefix_tokens={} decode_rows={} mtp_rows={} execution_lane={} buffer_bank={}",
                request.request_id,
                elapsed_ms(request_start),
                token_rows.prefix_tokens,
                decode_rows,
                mtp_rows,
                state.execution_lane_id,
                buffer_bank,
            );
        }
        Ok(PreparedBatchedDsparkCycle {
            request,
            request_start,
            request_timing,
            sequence_id,
            generated_tokens,
            request_id_base,
            token_prefix_tokens: token_rows.prefix_tokens,
            token_prefill_tokens: token_rows.prefill_tokens,
            committed_input_token_ids,
            decode_rows,
            scheduler_start: Instant::now(),
            shape,
            state,
            buffer_bank,
            pending_dspark_plan,
            pending_dspark_draft_token_ids,
            dspark_target_hidden_tap_rows,
            snapshot_restore_ms,
        })
    }

    fn restore_prepared_batched_dspark_cycle(&self, mut prepared: PreparedBatchedDsparkCycle) {
        if let Some(plan) = prepared.pending_dspark_plan.take() {
            if let Some(dspark) = self.dspark.as_ref() {
                if let Ok(mut dspark) = dspark.lock() {
                    dspark.restore_verification(&prepared.sequence_id, plan);
                }
            }
        }
        let _ = self.store_scheduler_state(&prepared.sequence_id, prepared.state);
    }

    fn refresh_prepared_batched_dspark_shape(prepared: &mut PreparedBatchedDsparkCycle) {
        prepared.pending_dspark_draft_token_ids = prepared
            .pending_dspark_plan
            .as_ref()
            .map(|plan| plan.proposal_token_ids.clone())
            .unwrap_or_default();
        let mtp_rows = prepared.pending_dspark_draft_token_ids.len();
        prepared.shape.mtp_rows = mtp_rows;
        let mut scheduler_token_ids = prepared.committed_input_token_ids.clone();
        scheduler_token_ids.extend_from_slice(&prepared.pending_dspark_draft_token_ids);
        prepared.shape.decode_token_ids = Some(scheduler_token_ids);
        prepared.dspark_target_hidden_tap_rows = if prepared
            .request
            .decode_budget
            .saturating_sub(prepared.generated_tokens)
            > 1
        {
            prepared.decode_rows + mtp_rows
        } else {
            0
        };
    }

    fn replan_prepared_batched_dspark_cycles(
        &self,
        prepared: &mut [PreparedBatchedDsparkCycle],
    ) -> std::result::Result<(), String> {
        let context_tokens = prepared
            .iter()
            .map(|cycle| cycle.token_prefix_tokens + cycle.token_prefill_tokens)
            .collect::<Vec<_>>();
        let max_useful_drafts = prepared
            .iter()
            .map(|cycle| {
                cycle
                    .request
                    .decode_budget
                    .saturating_sub(cycle.generated_tokens)
                    .saturating_sub(1)
            })
            .collect::<Vec<_>>();
        let jointly_adaptive = prepared.iter().all(|cycle| {
            cycle
                .pending_dspark_plan
                .as_ref()
                .is_some_and(|plan| plan.calibration_eligible)
        });

        if jointly_adaptive {
            let schedule = {
                let plans = prepared
                    .iter()
                    .zip(&max_useful_drafts)
                    .map(|(cycle, max_useful)| {
                        (
                            cycle
                                .pending_dspark_plan
                                .as_ref()
                                .expect("jointly adaptive dSpark cycle has a plan"),
                            *max_useful,
                        )
                    })
                    .collect::<Vec<_>>();
                self.dspark
                    .as_ref()
                    .expect("joint dSpark scheduling requires a runtime")
                    .lock()
                    .map_err(|error| format!("locking dSpark joint scheduler failed: {error}"))?
                    .joint_schedule(&plans, &context_tokens)
                    .map_err(format_error_chain)?
            };
            for ((cycle, max_useful), selected_drafts) in prepared
                .iter_mut()
                .zip(&max_useful_drafts)
                .zip(&schedule.prefix_lengths)
            {
                let plan = cycle
                    .pending_dspark_plan
                    .as_mut()
                    .expect("joint dSpark selection has a request plan");
                plan.minimum_drafts = plan.minimum_drafts.min(*max_useful);
                plan.apply_joint_selection(
                    *selected_drafts,
                    schedule.target_batch_rows,
                    schedule.expected_committed_tokens,
                    schedule.expected_tokens_per_second,
                )
                .map_err(format_error_chain)?;
                Self::refresh_prepared_batched_dspark_shape(cycle);
            }
            if real_full_dspark_trace_enabled() {
                eprintln!(
                    "real_full_dspark_joint_plan requests={} contexts={:?} selected_drafts={:?} target_rows={} expected_tokens={:.4} expected_tps={:.3}",
                    prepared.len(),
                    context_tokens,
                    schedule.prefix_lengths,
                    schedule.target_batch_rows,
                    schedule.expected_committed_tokens,
                    schedule.expected_tokens_per_second,
                );
            }
        } else {
            for cycle in prepared.iter_mut() {
                Self::refresh_prepared_batched_dspark_shape(cycle);
            }
        }

        let mut dspark = self
            .dspark
            .as_ref()
            .expect("issued dSpark plans require a runtime")
            .lock()
            .map_err(|error| format!("locking dSpark issued-plan tracker failed: {error}"))?;
        for cycle in prepared {
            if let Some(plan) = cycle.pending_dspark_plan.as_ref() {
                dspark.record_issued_plan(&cycle.sequence_id, plan);
            }
        }
        Ok(())
    }

    fn finish_batched_dspark_cycle(
        &self,
        mut prepared: PreparedBatchedDsparkCycle,
        execution: RealFullSchedulerDeviceExecution,
        paired_target_samples: Option<RealLmHeadBatchScoreForHidden>,
        precomputed_draft_replay: Option<(DsparkDraftStep, DsparkDraftPlan)>,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String> {
        let target_submit_ms = elapsed_ms(prepared.scheduler_start);
        let mut report = execution.report;
        let probe = execution.sparse_tcp_dispatch;
        let mut target_hidden = execution.final_target_device_hidden;
        let target_hidden_taps = execution.target_device_hidden_taps;
        let mut cycle_token_ids = Vec::new();
        let mut terminal_sample = None;
        let mut dspark_cache_update = None;
        if prepared.pending_dspark_draft_token_ids.is_empty() {
            let anchor_token = report
                .terminal_lm_head_sample
                .top_token_id
                .context("paired dSpark scalar step requires a target token")
                .map_err(format_error_chain)?;
            cycle_token_ids.push(anchor_token);
            dspark_cache_update = Some((1, anchor_token));
        } else {
            let target_hidden = match target_hidden.take() {
                Some(hidden) => hidden,
                None => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(
                        "paired dSpark verification has no retained target hidden batch".to_owned(),
                    );
                }
            };
            let suffix_rows = prepared.decode_rows + prepared.pending_dspark_draft_token_ids.len();
            let target_sampling_start = Instant::now();
            let target_samples = match paired_target_samples {
                Some(samples) => Ok(samples),
                None => real_full_target_token_samples(&self.catalog, &target_hidden, suffix_rows),
            };
            let target_samples = match target_samples {
                Ok(samples) => samples,
                Err(error) => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(format_error_chain(error));
                }
            };
            let target_token_ids = target_samples.top_token_ids.as_slice();
            let acceptance = match real_full_mtp_acceptance(
                prepared.pending_dspark_draft_token_ids.as_slice(),
                target_token_ids,
                true,
                prepared
                    .request
                    .decode_budget
                    .saturating_sub(prepared.generated_tokens),
            ) {
                Ok(acceptance) => acceptance,
                Err(error) => {
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(format_error_chain(error));
                }
            };
            let accepted_draft_tokens = acceptance.accepted_draft_tokens;
            let terminal_target_index = acceptance.terminal_target_index;
            cycle_token_ids.extend_from_slice(&target_token_ids[..=terminal_target_index]);
            prepared.committed_input_token_ids.extend_from_slice(
                &prepared.pending_dspark_draft_token_ids[..accepted_draft_tokens],
            );
            let tentative_token_start =
                prepared.token_prefix_tokens + prepared.token_prefill_tokens + prepared.decode_rows;
            if let Err(error) = prepared.state.resolve_mtp_tentative_writes(
                tentative_token_start,
                prepared.pending_dspark_draft_token_ids.len(),
                accepted_draft_tokens,
            ) {
                self.restore_prepared_batched_dspark_cycle(prepared);
                return Err(format_error_chain(error));
            }
            report.committed_mtp_writes = accepted_draft_tokens * GLM52_NUM_HIDDEN_LAYERS;
            report.discarded_mtp_writes = (prepared.pending_dspark_draft_token_ids.len()
                - accepted_draft_tokens)
                * GLM52_NUM_HIDDEN_LAYERS;
            report.request_mtp_accepted_rows = accepted_draft_tokens;
            terminal_sample = Some(RealFullSpeculativeTerminalSample {
                vocab_size: target_samples.vocab_size,
                top_token_id: target_samples.top_token_ids[terminal_target_index],
                sampled_token_id: target_token_ids[terminal_target_index],
                sample_top_k: 1,
                sample_top_p: 1.0,
                argmax_backend: target_samples.argmax_kernel_backend,
                sampler_backend: target_samples.argmax_kernel_backend,
                accepted_draft_tokens,
                report_mtp_acceptance: true,
            });
            dspark_cache_update = Some((
                accepted_draft_tokens + 1,
                target_token_ids[terminal_target_index],
            ));
            let plan = prepared
                .pending_dspark_plan
                .as_ref()
                .expect("paired dSpark drafts came from a pending plan");
            let (next_confidence_logit_bias, calibration_variance, calibration_cycles) = self
                .dspark
                .as_ref()
                .expect("paired dSpark cycle has a runtime")
                .lock()
                .map_err(|error| format!("locking dSpark confidence calibrator failed: {error}"))?
                .observe_verification(&prepared.sequence_id, plan, accepted_draft_tokens);
            if real_full_dspark_trace_enabled() {
                eprintln!(
                    "real_full_dspark_acceptance request_id={} sequence_id={} target_context={} drafts={} accepted={} emitted={} full_match={} target_rows={} expected_tokens={:.4} expected_tps={:.3} confidence_logit_bias={:.4} next_confidence_logit_bias={:.4} calibration_variance={:.6} calibration_cycles={} target_submit_ms={:.3} target_sampling_ms={:.3} paired=true",
                    prepared.request.request_id,
                    prepared.sequence_id,
                    prepared.token_prefix_tokens + prepared.token_prefill_tokens,
                    prepared.pending_dspark_draft_token_ids.len(),
                    accepted_draft_tokens,
                    cycle_token_ids.len(),
                    acceptance.full_match_bonus,
                    plan.target_batch_rows,
                    plan.expected_committed_tokens,
                    plan.expected_tokens_per_second,
                    plan.confidence_logit_bias,
                    next_confidence_logit_bias,
                    calibration_variance,
                    calibration_cycles,
                    target_submit_ms,
                    elapsed_ms(target_sampling_start),
                );
            }
        }
        report.request_mtp_verify_rows = prepared.pending_dspark_draft_token_ids.len();
        let final_decode_step = {
            let emitted_tokens = cycle_token_ids.len().max(1);
            prepared.generated_tokens + emitted_tokens >= prepared.request.decode_budget
        };
        if let Some(taps) = target_hidden_taps {
            if !final_decode_step {
                let (committed_rows, anchor_token) = dspark_cache_update
                    .context("paired dSpark target taps require a cache update")
                    .map_err(format_error_chain)?;
                if taps.rows != prepared.dspark_target_hidden_tap_rows
                    || committed_rows > taps.rows
                    || taps.layer_ids != real_full_active_draft_target_layer_ids()
                {
                    let error = format!(
                        "paired dSpark expected {} target rows with {} committed, got rows={} layers={:?}",
                        prepared.dspark_target_hidden_tap_rows,
                        committed_rows,
                        taps.rows,
                        taps.layer_ids
                    );
                    self.restore_prepared_batched_dspark_cycle(prepared);
                    return Err(error);
                }
                let replay = if let Some((step, plan)) = precomputed_draft_replay {
                    self.dspark
                        .as_ref()
                        .expect("paired DFlash2 taps have a request executor")
                        .lock()
                        .map_err(|error| {
                            format!("locking precomputed DFlash2 request state failed: {error}")
                        })
                        .and_then(|dspark| {
                            dspark
                                .request_context_tokens(&prepared.sequence_id)
                                .map(|context| (step, plan, dspark.mode, context))
                                .ok_or_else(|| {
                                    format!(
                                        "precomputed DFlash2 request state is missing for {}",
                                        prepared.sequence_id
                                    )
                                })
                        })
                } else {
                    self.dspark
                        .as_ref()
                        .expect("paired dSpark taps have a request executor")
                        .lock()
                        .map_err(|error| format!("locking dSpark request executor failed: {error}"))
                        .and_then(|mut dspark| {
                            let absolute_context_start = real_full_draft_absolute_context_start(
                                prepared.generated_tokens,
                                Some(dspark.cache_mode),
                                prepared.token_prefix_tokens,
                                taps.row_start,
                            );
                            dspark
                                .replay_step(
                                    &prepared.sequence_id,
                                    &taps.values,
                                    0,
                                    committed_rows,
                                    absolute_context_start,
                                    anchor_token,
                                )
                                .map_err(format_error_chain)
                                .and_then(|(step, plan)| {
                                    dspark
                                        .request_context_tokens(&prepared.sequence_id)
                                        .map(|context| (step, plan, dspark.mode, context))
                                        .ok_or_else(|| {
                                            format!(
                                                "replayed dSpark request state is missing for {}",
                                                prepared.sequence_id
                                            )
                                        })
                                })
                        })
                };
                let (step, plan, draft_mode, draft_context_after) = match replay {
                    Ok(replay) => replay,
                    Err(error) => {
                        self.restore_prepared_batched_dspark_cycle(prepared);
                        return Err(error);
                    }
                };
                if real_full_dspark_trace_enabled() {
                    eprintln!(
                        "real_full_dspark_step request_id={} sequence_id={} mode={:?} target_context={} draft_context_before={} committed_rows={} draft_context_after={} anchor_token={} selected_drafts={} target_batch_rows={} expected_tokens={:.4} expected_tps={:.3} update_ms={:.3} suffix_ms={:.3} readback_ms={:.3} dspark_total_ms={:.3} selected_proposals={:?} proposals={:?} confidence={:?} paired=true",
                        prepared.request.request_id,
                        prepared.sequence_id,
                        draft_mode,
                        prepared.token_prefix_tokens + prepared.token_prefill_tokens,
                        step.context_tokens,
                        step.committed_rows,
                        draft_context_after,
                        step.anchor_token,
                        plan.selected_drafts,
                        plan.target_batch_rows,
                        plan.expected_committed_tokens,
                        plan.expected_tokens_per_second,
                        step.update_ms,
                        step.suffix_ms,
                        step.readback_ms,
                        step.total_ms,
                        plan.proposal_token_ids,
                        step.proposal_token_ids,
                        step.conditional_confidence,
                    );
                }
            }
        }
        if let Err(error) = prepared.state.record_processed_token_ids(
            prepared.token_prefix_tokens,
            &prepared.committed_input_token_ids,
        ) {
            self.restore_prepared_batched_dspark_cycle(prepared);
            return Err(format_error_chain(error));
        }
        if let Err(error) =
            self.finish_scheduler_state(&prepared.sequence_id, prepared.state, final_decode_step)
        {
            return Err(error);
        }
        if prepared.request.greedy_sampling {
            report.terminal_lm_head_sample.sampled_token_id =
                report.terminal_lm_head_sample.top_token_id;
            report.terminal_lm_head_sample.sample_top_k = Some(1);
            report.terminal_lm_head_sample.sample_top_p = Some(1.0);
            report.terminal_lm_head_sample.sampler_kernel_backend =
                report.terminal_lm_head_sample.argmax_kernel_backend;
        }
        if let Some(sample) = terminal_sample.as_ref() {
            apply_speculative_terminal_sample_to_report(&mut report, sample);
        }
        let sampled_token_text =
            self.decode_sampled_token_text_cached(report.terminal_lm_head_sample.sampled_token_id);
        let mut info = real_full_info_from_request_execution(
            &self.base_info,
            &self.catalog.snapshot_path,
            &report,
            sampled_token_text,
        );
        info.request_kv_snapshot_restore_ms = prepared.snapshot_restore_ms;
        apply_sparse_tcp_dispatch_probe(
            &mut info,
            self.sparse_tcp_targets.as_ref().map_or(0, Vec::len),
            &probe,
        );
        if let Some(sample) = terminal_sample.as_ref() {
            if sample.report_mtp_acceptance {
                info.request_mtp_accepted_rows = sample.accepted_draft_tokens;
            }
            info.scheduler_terminal_lm_head_top_token_id = Some(sample.top_token_id);
            info.scheduler_terminal_lm_head_sampled_token_id = Some(sample.sampled_token_id);
            info.scheduler_terminal_lm_head_sampled_text =
                self.decode_sampled_token_text_cached(Some(sample.sampled_token_id));
            info.scheduler_terminal_lm_head_sample_top_k = Some(sample.sample_top_k);
            info.scheduler_terminal_lm_head_sample_top_p = Some(sample.sample_top_p);
            info.scheduler_terminal_lm_head_argmax_backend = Some(sample.argmax_backend.to_owned());
            info.scheduler_terminal_lm_head_sampler_backend =
                Some(sample.sampler_backend.to_owned());
        }
        if prepared.request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=paired_finish scheduler_ms={:.3} total_ms={:.3} status={} sampled_token_id={:?}",
                prepared.request.request_id,
                target_submit_ms,
                elapsed_ms(prepared.request_start),
                info.status,
                info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }
        let generated_tokens = cycle_token_ids
            .into_iter()
            .map(|token_id| glmrt_api::RealFullGeneratedToken {
                token_id,
                text: self.decode_sampled_token_text_cached(Some(token_id)),
            })
            .collect();
        Ok(glmrt_api::RealFullDecodeCycle {
            info,
            generated_tokens,
        })
    }

    fn execute_real_full_decode_cycle_inner(
        &self,
        mut request: glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String> {
        if request.constraint.is_some() && self.sparse_tcp_dispatch_worker.is_none() {
            return Err(
                "constrained decoding requires the shared sparse TCP dispatch worker".to_owned(),
            );
        }
        prewarm_flashinfer_cudnn_mla_suffix_graphs_for_worker().map_err(format_error_chain)?;
        let request_start = Instant::now();
        let request_id = request.request_id.clone();
        let request_index = request.request_index;
        let prompt_tokens_hint = request.prompt_tokens;
        let max_tokens = request.max_tokens;
        let generated_tokens = request.generated_token_ids.len();
        let request_timing = real_full_request_timing_enabled();
        let sequence_id = request.sequence_id.clone();
        let final_decode_step = request.decode_step_index + 1 >= request.decode_budget;
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} request_index={} stage=start prompt_tokens_hint={} generated_tokens={} max_tokens={}",
                request_id, request_index, prompt_tokens_hint, generated_tokens, max_tokens
            );
        }
        let fast_token_enabled = real_full_serve_fast_token_enabled();
        let should_tokenize_prompt = request.generated_token_ids.is_empty();
        let tokenize_start = Instant::now();
        let prompt_token_ids = if should_tokenize_prompt {
            let tokenizer = self
                .tokenizer
                .lock()
                .map_err(|err| format!("locking real-full tokenizer failed: {err}"))?;
            request_prompt_token_ids(&tokenizer, &request).map_err(format_error_chain)?
        } else {
            None
        };
        if should_tokenize_prompt && !real_full_internal_sequence(&sequence_id) {
            if let Some(snapshot) = self.kv_snapshot_load.as_ref() {
                let prompt_token_ids = prompt_token_ids
                    .as_ref()
                    .expect("an initial request was tokenized");
                let snapshot_tokens = snapshot.token_count();
                if prompt_token_ids.len() <= snapshot_tokens {
                    return Err(format!(
                        "loaded KV snapshot has {snapshot_tokens} tokens but request {} has only {}; at least one uncached suffix token is required",
                        request.request_id,
                        prompt_token_ids.len()
                    ));
                }
                if prompt_token_ids[..snapshot_tokens] != *snapshot.token_ids() {
                    let mismatch = prompt_token_ids[..snapshot_tokens]
                        .iter()
                        .zip(snapshot.token_ids())
                        .position(|(request_token, snapshot_token)| request_token != snapshot_token)
                        .expect("unequal equal-length token slices have a mismatch");
                    return Err(format!(
                        "request {} does not match loaded KV snapshot {} at token {mismatch}: request={} snapshot={}",
                        request.request_id,
                        snapshot.root().display(),
                        prompt_token_ids[mismatch],
                        snapshot.token_ids()[mismatch],
                    ));
                }
                request.cached_prompt_tokens = snapshot_tokens;
            }
        }
        let persistent_scheduler_state =
            self.sparse_tcp_dispatch_worker.is_some() || self.sparse_tcp_targets.is_none();
        let native_mtp_request = real_full_mtp_enabled()
            && request.greedy_sampling
            && !request.disable_speculation
            && self.sparse_tcp_dispatch_worker.is_some()
            && real_full_native_mtp_sequence_enabled(&sequence_id);
        let mut target_radix_reservation = None;
        if should_tokenize_prompt
            && request.cached_prompt_tokens == 0
            && request.vision_embeddings.is_none()
            // A target-only radix hit cannot reconstruct the target hidden
            // rows needed to seed layer 78, and its shared physical pages do
            // not prove that the MTP KV plane has the same frontier. Until
            // cache nodes carry an MTP-ready variant, native MTP must compute
            // the full prompt rather than silently bind an incomplete layer.
            && !native_mtp_request
            && (!real_full_internal_sequence(&sequence_id)
                || (real_full_capture_arena_sequence(&sequence_id)
                    && !real_full_batched_dspark_prewarm_sequence(&sequence_id)
                    // Workspace sizing requests must execute their full
                    // physical shape even if an earlier synthetic startup
                    // seed has populated the radix. Only the designated
                    // publisher may bind a reservation in this namespace.
                    && (!real_full_startup_workspace_sizing_sequence(&sequence_id)
                        || real_full_startup_target_radix_publish_tokens(&sequence_id).is_some())))
            && self.kv_snapshot_load.is_none()
            && self.kv_snapshot_saves.is_empty()
            && persistent_scheduler_state
        {
            let prompt_ids = prompt_token_ids
                .as_ref()
                .expect("an initial external request was tokenized");
            let capacity_tokens = real_full_sequence_capacity_tokens(
                request.prompt_tokens,
                request.decode_budget,
                self.kv_config.max_tokens,
            )
            .map_err(format_error_chain)?;
            let reusable_prompt_ids = &prompt_ids[..prompt_ids.len().saturating_sub(1)];
            let mut reservation = self
                .target_kv_radix
                .reserve(reusable_prompt_ids, capacity_tokens)
                .map_err(format_error_chain)?;
            if request.decode_budget > 1 && !request.disable_speculation {
                if let Some(dspark) = self.dspark.as_ref() {
                    let dspark = dspark
                        .lock()
                        .map_err(|error| format!("locking dSpark prefix cache failed: {error}"))?;
                    if dspark.mode == RealFullDsparkServingMode::Active
                        && dspark.cache_mode == RealFullDsparkCacheMode::PromptSwa
                        && (!dspark.is_dflash2() || request.greedy_sampling)
                    {
                        let target_match = reservation.matched_prefix_tokens();
                        let aligned_match = dspark
                            .tail_cache
                            .longest_exact_prefix_tokens(prompt_ids, target_match);
                        if aligned_match < target_match {
                            drop(dspark);
                            drop(reservation);
                            reservation = self
                                .target_kv_radix
                                .reserve(&prompt_ids[..aligned_match], capacity_tokens)
                                .map_err(format_error_chain)?;
                            if reservation.matched_prefix_tokens() != aligned_match {
                                return Err(format!(
                                    "target/dSpark aligned radix reservation matched {} tokens, expected {aligned_match}",
                                    reservation.matched_prefix_tokens(),
                                ));
                            }
                            eprintln!(
                                "real_full_dspark_radix_align sequence_id={} target_match={} aligned_match={} recompute_tokens={}",
                                sequence_id,
                                target_match,
                                aligned_match,
                                target_match - aligned_match,
                            );
                        }
                    }
                }
            }
            request.cached_prompt_tokens = reservation.matched_prefix_tokens();
            target_radix_reservation = Some(reservation);
        }
        let stateful_decode_step = generated_tokens > 0;
        let token_rows = real_full_request_token_rows(&request, prompt_token_ids.clone())
            .map_err(format_error_chain)?;
        let committed_prefix_tokens = token_rows.prefix_tokens;
        let mut committed_input_token_ids = token_rows
            .prefill_token_ids
            .iter()
            .flatten()
            .copied()
            .chain(token_rows.decode_token_ids.iter().copied())
            .collect::<Vec<_>>();
        let decode_rows = token_rows.decode_token_ids.len();
        let planned_prefill_chunk_tokens = if request.cached_prompt_tokens > 0
            && sequence_id.starts_with("real-full-startup-dsa-selector-seed-")
        {
            // The selector sweep is graph-capture infrastructure, not a
            // serving policy probe. Preserve its requested physical query
            // bucket instead of applying the ordinary cached-suffix 256-row
            // cap; otherwise the log says "512" while only two 256-row
            // identities are captured.
            token_rows.prefill_tokens.max(1)
        } else {
            real_full_request_prefill_chunk_tokens_for_sequence(
                &sequence_id,
                token_rows.prefix_tokens,
                token_rows.prefill_tokens,
            )
        };
        let prefill_chunk_tokens =
            real_full_prefill_chunk_tokens_for_direct_dsa(planned_prefill_chunk_tokens);
        let request_id_base = request.request_index.saturating_mul(1_000_000);
        if fast_token_enabled {
            let fast_start = Instant::now();
            let info = self
                .fast_embedding_lm_head_token_info(
                    &request_id,
                    request.prompt_tokens,
                    generated_tokens,
                    max_tokens,
                    token_rows.prefill_tokens,
                    prefill_chunk_tokens,
                    decode_rows,
                    token_rows.decode_token_ids.as_slice(),
                )
                .map_err(format_error_chain)?;
            if request_timing {
                eprintln!(
                    "real_full_request_timing request_id={} stage=serve_fast_token elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
                    request_id,
                    elapsed_ms(fast_start),
                    elapsed_ms(request_start),
                    info.scheduler_terminal_lm_head_sampled_token_id
                );
            }
            return Ok(glmrt_api::RealFullDecodeCycle::single_token(info));
        }

        let mtp_enabled = native_mtp_request;
        let dspark_configuration = self
            .dspark
            .as_ref()
            .map(|runtime| {
                runtime
                    .lock()
                    .map(|runtime| {
                        (
                            runtime.mode,
                            runtime.cache_mode,
                            runtime.context_tokens,
                            runtime.is_dflash2(),
                        )
                    })
                    .map_err(|error| format!("locking dSpark request executor failed: {error}"))
            })
            .transpose()?;
        let dspark_mode = dspark_configuration.map(|configuration| configuration.0);
        let dspark_cache_mode = dspark_configuration.map(|configuration| configuration.1);
        let dspark_context_tokens = dspark_configuration.map(|configuration| configuration.2);
        let dflash2_active = dspark_configuration.is_some_and(|configuration| configuration.3);
        let dspark_active = dspark_mode == Some(RealFullDsparkServingMode::Active)
            && (!dflash2_active || request.greedy_sampling)
            && !request.disable_speculation
            && request.decode_budget.saturating_sub(generated_tokens) > 1
            && self.sparse_tcp_dispatch_worker.is_some()
            && (!real_full_internal_sequence(&sequence_id)
                || real_full_dspark_startup_draft_tokens(&sequence_id).is_some());
        let dspark_shadow = dspark_mode == Some(RealFullDsparkServingMode::Shadow)
            && request.greedy_sampling
            && !request.disable_speculation
            && request.decode_budget.saturating_sub(generated_tokens) > 1
            && self.sparse_tcp_dispatch_worker.is_some()
            && !real_full_internal_sequence(&sequence_id);
        let dspark_participating = dspark_active || dspark_shadow;
        let mut persistent_state = if persistent_scheduler_state {
            let radix_binding = target_radix_reservation.take().map(|reservation| {
                (
                    reservation,
                    prompt_token_ids
                        .as_deref()
                        .expect("target KV radix request retained prompt token IDs"),
                )
            });
            Some(self.take_scheduler_state(&request, radix_binding)?)
        } else {
            None
        };
        let snapshot_restore_ms = persistent_state
            .as_ref()
            .map_or(0.0, |state| state.snapshot_restore_ms);
        // A persistent scheduler state is permanently assigned to one
        // execution lane. Use that lane's production buffer bank for scalar
        // as well as batched execution so C=1 startup does not capture a
        // duplicate bank-0 graph set that live lane 0 can never reuse.
        let execution_buffer_bank = persistent_state
            .as_ref()
            .map_or(0, |state| state.execution_lane_id + 1);
        let _execution_buffer_bank_scope =
            coordinator_owned_device_buffer_bank_scope(execution_buffer_bank);
        let mut pending_mtp_draft_token_ids = if mtp_enabled {
            persistent_state
                .as_mut()
                .expect("live sparse MTP has persistent scheduler state")
                .take_pending_mtp_draft_token_ids()
        } else {
            Vec::new()
        };
        if let Some(constraint) = persistent_state
            .as_ref()
            .and_then(|state| state.constraint.as_ref())
        {
            let valid = constraint
                .valid_draft_prefix(&pending_mtp_draft_token_ids)
                .map_err(format_error_chain)?;
            pending_mtp_draft_token_ids.truncate(valid);
        }
        let requested_startup_dspark_drafts =
            real_full_scalar_dspark_prewarm_requested_draft_tokens(
                &sequence_id,
                request.request_index,
            )
            .or_else(|| {
                real_full_batched_dspark_prewarm_requested_draft_tokens(
                    &sequence_id,
                    request.request_index,
                )
            })
            .or_else(|| real_full_dspark_startup_draft_tokens(&sequence_id));
        let mut pending_dspark_plan = if dspark_participating {
            self.dspark
                .as_ref()
                .expect("dSpark participation requires a request executor")
                .lock()
                .map_err(|error| format!("locking dSpark request executor failed: {error}"))?
                .prepare_cycle(
                    &sequence_id,
                    generated_tokens == 0,
                    (generated_tokens == 0)
                        .then(|| {
                            prompt_token_ids
                                .as_deref()
                                .map(|token_ids| (token_ids, request.cached_prompt_tokens))
                        })
                        .flatten(),
                    requested_startup_dspark_drafts,
                )
                .map_err(format_error_chain)?
        } else {
            None
        };
        if let Some(plan) = pending_dspark_plan.as_mut() {
            if requested_startup_dspark_drafts.is_none() {
                let max_useful_drafts = request
                    .decode_budget
                    .saturating_sub(generated_tokens)
                    .saturating_sub(1);
                plan.proposal_token_ids.truncate(max_useful_drafts);
            }
            if let Some(constraint) = persistent_state
                .as_ref()
                .and_then(|state| state.constraint.as_ref())
            {
                constrain_dspark_plan(plan, constraint).map_err(format_error_chain)?;
            }
            plan.selected_drafts = plan.proposal_token_ids.len();
            plan.target_batch_rows = plan.selected_drafts + 1;
        }
        if let Some(plan) = pending_dspark_plan.as_ref() {
            self.dspark
                .as_ref()
                .expect("issued dSpark plan requires a request runtime")
                .lock()
                .map_err(|error| format!("locking dSpark issued-plan tracker failed: {error}"))?
                .record_issued_plan(&sequence_id, plan);
        }
        let pending_dspark_draft_token_ids = pending_dspark_plan
            .as_ref()
            .map(|plan| plan.proposal_token_ids.clone())
            .unwrap_or_default();
        let synthetic_mtp_rows = if mtp_enabled || dspark_active {
            0
        } else {
            real_full_request_mtp_rows(&request, stateful_decode_step)
        };
        let mtp_physical_padding_rows = if mtp_enabled
            && persistent_state
                .as_ref()
                .is_none_or(|state| state.constraint.is_none())
        {
            real_full_mtp_physical_padding_rows(
                pending_mtp_draft_token_ids.len(),
                real_full_mtp_physical_m2_enabled(),
            )
        } else {
            0
        };
        let mtp_rows = if mtp_enabled {
            pending_mtp_draft_token_ids.len() + mtp_physical_padding_rows
        } else if dspark_active {
            pending_dspark_draft_token_ids.len()
        } else {
            synthetic_mtp_rows
        };
        real_full_validate_sparse_wave_capacity(prefill_chunk_tokens, decode_rows, mtp_rows)
            .map_err(format_error_chain)?;
        let run_mtp_probe = !mtp_enabled
            && ((real_full_mtp_probe_enabled()
                && !sequence_id.starts_with("real-full-startup-prewarm-"))
                || (real_full_mtp_enabled()
                    && sequence_id.starts_with("real-full-startup-dsa-selector-seed-")));
        let initial_mtp_draft_limit = if mtp_enabled && pending_mtp_draft_token_ids.is_empty() {
            real_full_mtp_requested_draft_tokens(
                &sequence_id,
                persistent_state
                    .as_mut()
                    .expect("live sparse MTP has persistent scheduler state"),
            )
        } else {
            0
        };
        let initial_mtp_drafts = if mtp_enabled && pending_mtp_draft_token_ids.is_empty() {
            real_full_mtp_draft_tokens_for_cycle_with_limit(
                request.decode_budget,
                generated_tokens,
                1,
                request.prompt_tokens,
                self.kv_config.max_tokens,
                initial_mtp_draft_limit,
            )
        } else {
            0
        };
        let retain_target_hidden = run_mtp_probe
            || (mtp_enabled && (!pending_mtp_draft_token_ids.is_empty() || initial_mtp_drafts > 0))
            // D=0 is a valid adaptive decision, but its scalar target row must
            // still seed the next proposal window. Otherwise one low-confidence
            // decision permanently disables drafting and the recovery probe
            // can never run.
            || dspark_active
            || persistent_state
                .as_ref()
                .is_some_and(|state| state.constraint.is_some());
        let dspark_target_hidden_tap_rows =
            if dspark_participating && request.decode_budget.saturating_sub(generated_tokens) > 1 {
                if generated_tokens == 0
                    && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                {
                    token_rows
                        .prefill_tokens
                        .checked_add(decode_rows)
                        .and_then(|rows| rows.checked_add(mtp_rows))
                        .ok_or_else(|| "dSpark prompt-tail tap row count overflow".to_owned())?
                        .min(
                            dspark_context_tokens
                                .expect("dSpark participation has a resolved context window"),
                        )
                } else if generated_tokens > 0 {
                    decode_rows + mtp_rows
                } else {
                    // Request-local mode intentionally excludes the final prompt
                    // token. Generated token zero is captured on the next pass.
                    0
                }
            } else {
                0
            };
        let mtp_target_input_token_ids = (run_mtp_probe || initial_mtp_drafts > 0).then(|| {
            token_rows
                .prefill_token_ids
                .iter()
                .flatten()
                .copied()
                .chain(token_rows.decode_token_ids.iter().copied())
                .collect::<Vec<_>>()
        });
        let mut scheduler_token_ids = token_rows.decode_token_ids.clone();
        scheduler_token_ids.extend_from_slice(&pending_mtp_draft_token_ids);
        scheduler_token_ids.extend_from_slice(&pending_dspark_draft_token_ids);
        if mtp_physical_padding_rows > 0 {
            // Deterministic opt-out: execute logical D=1 as M=3. The duplicate
            // sentinel is sampled physically but never accepted logically.
            scheduler_token_ids.push(
                *pending_mtp_draft_token_ids
                    .last()
                    .expect("logical D=1 padding requires one pending MTP draft"),
            );
        }
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=tokenize elapsed_ms={:.3} prefill_tokens={} prefill_chunk_tokens={} decode_rows={} mtp_rows={} mtp_enabled={}",
                request_id,
                elapsed_ms(tokenize_start),
                token_rows.prefill_tokens,
                prefill_chunk_tokens,
                decode_rows,
                mtp_rows,
                mtp_enabled,
            );
        }

        if let Some(vision_embeddings) = request.vision_embeddings.as_deref() {
            for embedding in vision_embeddings {
                if !embedding
                    .token_start
                    .checked_add(embedding.rows)
                    .is_some_and(|end| end <= token_rows.prefill_tokens)
                {
                    return Err(format!(
                        "vision embedding {} rows {}..{} are not wholly inside {} prefill rows",
                        embedding.image_sha256,
                        embedding.token_start,
                        embedding.token_start.saturating_add(embedding.rows),
                        token_rows.prefill_tokens
                    ));
                }
            }
        }
        let shape = RealFullSchedulerExecutionShape {
            request_id: request.request_id.clone(),
            sequence_id: sequence_id.clone(),
            placement_version: format!("real-full-api-request-{}", request.request_index),
            prefix_tokens: token_rows.prefix_tokens,
            prefill_tokens: token_rows.prefill_tokens,
            prefill_chunk_tokens,
            decode_rows,
            mtp_rows,
            mtp_accepted_rows: if mtp_enabled || dspark_active {
                0
            } else {
                mtp_rows.min(REAL_FULL_REQUEST_MTP_ACCEPTED_ROWS)
            },
            prefill_token_ids: token_rows.prefill_token_ids,
            prefill_vision_embeddings: request.vision_embeddings.clone(),
            decode_token_ids: Some(scheduler_token_ids),
            lm_head_sampling: request_lm_head_sampling_options(&request),
        };
        let scheduler_start = Instant::now();
        let (mut report, sparse_tcp_dispatch, cycle_token_ids, mtp_terminal_sample) = if let Some(
            dispatch_worker,
        ) =
            self.sparse_tcp_dispatch_worker.as_ref()
        {
            let mut state = persistent_state
                .take()
                .expect("shared sparse dispatch has persistent scheduler state");
            let constrained_sampling = state.constraint.is_some();
            let execution =
                    real_full_scheduler_execution_for_shape_with_shared_sparse_tcp_and_state_device_hidden(
                        self.kv_config.clone(),
                        &self.catalog,
                        shape,
                        Arc::clone(dispatch_worker),
                        request_id_base,
                        &mut state,
                        retain_target_hidden,
                        run_mtp_probe || initial_mtp_drafts > 0,
                        constrained_sampling,
                        dspark_target_hidden_tap_rows,
                    );
            let execution = match execution {
                Ok(result) => result,
                Err(error) => {
                    if mtp_enabled {
                        state.set_pending_mtp_draft_token_ids(pending_mtp_draft_token_ids.clone());
                    }
                    if let Some(plan) = pending_dspark_plan.take() {
                        if let Some(dspark) = self.dspark.as_ref() {
                            if let Ok(mut dspark) = dspark.lock() {
                                dspark.restore_verification(&sequence_id, plan);
                            }
                        }
                    }
                    let _ = self.store_scheduler_state(&sequence_id, state);
                    return Err(format_error_chain(error));
                }
            };
            let target_submit_ms = elapsed_ms(scheduler_start);
            let mut report = execution.report;
            let probe = execution.sparse_tcp_dispatch;
            let mut target_hidden = execution.final_target_device_hidden;
            let dspark_target_hidden_taps = execution.target_device_hidden_taps;
            let mut cycle_token_ids = Vec::new();
            let mut mtp_terminal_sample = None;
            let mut dspark_cache_update = None;
            if state.constraint.is_some()
                && pending_mtp_draft_token_ids.is_empty()
                && pending_dspark_draft_token_ids.is_empty()
            {
                let constrained_hidden = target_hidden
                    .as_ref()
                    .context("constrained scalar decode has no retained target hidden row")
                    .map_err(format_error_chain)?;
                let constrained_samples = real_full_constraint_target_samples(
                    &self.catalog,
                    &state,
                    &request,
                    constrained_hidden,
                    1,
                    &[],
                )
                .map_err(format_error_chain)?
                .expect("active constraint produces target samples");
                let sampled_token_id = if request.greedy_sampling {
                    constrained_samples.top_token_ids[0]
                } else {
                    constrained_samples.sampled_token_ids[0]
                };
                apply_speculative_terminal_sample_to_report(
                    &mut report,
                    &RealFullSpeculativeTerminalSample {
                        vocab_size: constrained_samples.vocab_size,
                        top_token_id: constrained_samples.top_token_ids[0],
                        sampled_token_id,
                        sample_top_k: if request.greedy_sampling {
                            1
                        } else {
                            constrained_samples.sample_top_k
                        },
                        sample_top_p: if request.greedy_sampling {
                            1.0
                        } else {
                            constrained_samples.sample_top_p
                        },
                        argmax_backend: constrained_samples.argmax_kernel_backend,
                        sampler_backend: if request.greedy_sampling {
                            constrained_samples.argmax_kernel_backend
                        } else {
                            constrained_samples.sampler_kernel_backend
                        },
                        accepted_draft_tokens: 0,
                        report_mtp_acceptance: false,
                    },
                );
            }
            if mtp_enabled {
                if pending_mtp_draft_token_ids.is_empty() {
                    let sampled_token_id = match if request.greedy_sampling {
                        report.terminal_lm_head_sample.top_token_id
                    } else {
                        report.terminal_lm_head_sample.sampled_token_id
                    } {
                        Some(token_id) => token_id,
                        None => {
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format!(
                                "real-full MTP request {} produced no target token",
                                request.request_id
                            ));
                        }
                    };
                    cycle_token_ids.push(sampled_token_id);
                    let draft_tokens = initial_mtp_drafts;
                    if draft_tokens > 0 {
                        let target_hidden = match target_hidden.take() {
                            Some(hidden) => hidden,
                            None => {
                                let _ = self.store_scheduler_state(&sequence_id, state);
                                return Err(format!(
                                    "real-full MTP request {} has no final target device hidden for drafting",
                                    request.request_id
                                ));
                            }
                        };
                        let shifted_token_ids = real_full_mtp_shifted_input_token_ids(
                            mtp_target_input_token_ids
                                .as_deref()
                                .expect("initial MTP target input tokens were retained"),
                            sampled_token_id,
                        )
                        .map_err(format_error_chain)?;
                        let mut draft_constraint = state
                            .constraint
                            .as_ref()
                            .map(|constraint| constraint.draft_branch(&cycle_token_ids))
                            .transpose()
                            .map_err(format_error_chain)?;
                        let drafts = self
                            .execute_mtp_chain(
                                request.request_index,
                                token_rows.prefix_tokens,
                                request_id_base,
                                shifted_token_ids.as_slice(),
                                target_hidden,
                                draft_tokens,
                                request.greedy_sampling,
                                Arc::clone(dispatch_worker),
                                draft_constraint.as_mut(),
                                &mut state.state,
                            )
                            .map_err(format_error_chain)?;
                        state.set_pending_mtp_draft_token_ids(
                            drafts.into_iter().map(|draft| draft.token_id).collect(),
                        );
                    }
                } else {
                    let target_hidden = match target_hidden.take() {
                        Some(hidden) => hidden,
                        None => {
                            state.set_pending_mtp_draft_token_ids(
                                pending_mtp_draft_token_ids.clone(),
                            );
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format!(
                                "real-full MTP request {} has no final target device hidden for verification",
                                request.request_id
                            ));
                        }
                    };
                    let suffix_rows = decode_rows + mtp_rows;
                    let target_sampling_start = Instant::now();
                    let target_samples = match real_full_constraint_target_samples(
                        &self.catalog,
                        &state,
                        &request,
                        &target_hidden,
                        suffix_rows,
                        &pending_mtp_draft_token_ids,
                    )
                    .map_err(format_error_chain)?
                    {
                        Some(samples) => samples,
                        None => real_full_target_token_samples(
                            &self.catalog,
                            &target_hidden,
                            suffix_rows,
                        )
                        .map_err(format_error_chain)?,
                    };
                    if request_timing {
                        eprintln!(
                            "real_full_mtp_target_sampling_timing request_id={} rows={} elapsed_ms={:.3}",
                            request_id,
                            suffix_rows,
                            elapsed_ms(target_sampling_start),
                        );
                    }
                    let physical_target_token_ids = if request.greedy_sampling {
                        target_samples.top_token_ids.as_slice()
                    } else {
                        target_samples.sampled_token_ids.as_slice()
                    };
                    let target_token_ids = &physical_target_token_ids
                        [..decode_rows + pending_mtp_draft_token_ids.len()];
                    let acceptance = real_full_mtp_acceptance(
                        pending_mtp_draft_token_ids.as_slice(),
                        target_token_ids,
                        real_full_mtp_full_match_bonus_enabled(),
                        request.decode_budget.saturating_sub(generated_tokens),
                    )
                    .map_err(format_error_chain)?;
                    if real_full_mtp_probe_enabled() {
                        eprintln!(
                            "real_full_mtp_acceptance source_request_id={} generated_tokens={} pending_drafts={:?} target_tokens={:?} accepted={} terminal_target_index={} full_match={}",
                            request.request_index,
                            generated_tokens,
                            pending_mtp_draft_token_ids,
                            target_token_ids,
                            acceptance.accepted_draft_tokens,
                            acceptance.terminal_target_index,
                            acceptance.full_match_bonus,
                        );
                    }
                    let accepted_draft_tokens = acceptance.accepted_draft_tokens;
                    let terminal_target_index = acceptance.terminal_target_index;
                    cycle_token_ids.extend_from_slice(&target_token_ids[..=terminal_target_index]);
                    // These proposal rows were target-model inputs in this
                    // verify pass and their target KV writes just became
                    // committed. Keep the radix/snapshot token frontier in
                    // lockstep with committed KV; rejected proposals and
                    // physical padding remain absent.
                    committed_input_token_ids
                        .extend_from_slice(&pending_mtp_draft_token_ids[..accepted_draft_tokens]);
                    let tentative_token_start =
                        token_rows.prefix_tokens + token_rows.prefill_tokens + decode_rows;
                    state
                        .resolve_mtp_tentative_writes(
                            tentative_token_start,
                            mtp_rows,
                            accepted_draft_tokens,
                        )
                        .map_err(format_error_chain)?;
                    let target_suffix_start = target_hidden.rows - suffix_rows;
                    let boundary_target_index = if acceptance.full_match_bonus {
                        terminal_target_index.saturating_sub(1)
                    } else {
                        terminal_target_index
                    };
                    let boundary_position = token_rows.prefix_tokens
                        + token_rows.prefill_tokens
                        + boundary_target_index;
                    state
                        .rewind_mtp_draft_layer(boundary_position)
                        .map_err(format_error_chain)?;
                    let draft_policy = real_full_mtp_draft_policy();
                    state.observe_mtp_draft_acceptance(
                        draft_policy.min,
                        draft_policy.max,
                        draft_policy.start,
                        pending_mtp_draft_token_ids.len(),
                        accepted_draft_tokens,
                        draft_policy.adaptive,
                        real_full_mtp_physical_m2_enabled(),
                    );
                    let requested_draft_tokens =
                        real_full_mtp_requested_draft_tokens(&sequence_id, &mut state);
                    let draft_tokens = real_full_mtp_draft_tokens_for_cycle_with_limit(
                        request.decode_budget,
                        generated_tokens,
                        cycle_token_ids.len(),
                        request.prompt_tokens,
                        self.kv_config.max_tokens,
                        requested_draft_tokens,
                    );
                    if draft_tokens > 0 {
                        let boundary_hidden = real_full_device_hidden_row(
                            &target_hidden,
                            target_suffix_start + boundary_target_index,
                        )
                        .map_err(format_error_chain)?;
                        let mut draft_constraint = state
                            .constraint
                            .as_ref()
                            .map(|constraint| constraint.draft_branch(&cycle_token_ids))
                            .transpose()
                            .map_err(format_error_chain)?;
                        let drafts = if acceptance.full_match_bonus {
                            self.execute_mtp_bridge_chain(
                                request.request_index,
                                boundary_position,
                                request_id_base,
                                target_token_ids[boundary_target_index],
                                target_token_ids[terminal_target_index],
                                boundary_hidden,
                                draft_tokens,
                                request.greedy_sampling,
                                Arc::clone(dispatch_worker),
                                draft_constraint.as_mut(),
                                &mut state.state,
                            )
                        } else {
                            self.execute_mtp_chain(
                                request.request_index,
                                boundary_position,
                                request_id_base,
                                &target_token_ids[boundary_target_index..=terminal_target_index],
                                boundary_hidden,
                                draft_tokens,
                                request.greedy_sampling,
                                Arc::clone(dispatch_worker),
                                draft_constraint.as_mut(),
                                &mut state.state,
                            )
                        }
                        .map_err(format_error_chain)?;
                        state.set_pending_mtp_draft_token_ids(
                            drafts.into_iter().map(|draft| draft.token_id).collect(),
                        );
                    }
                    report.request_mtp_accepted_rows = accepted_draft_tokens;
                    report.committed_mtp_writes = accepted_draft_tokens * GLM52_NUM_HIDDEN_LAYERS;
                    report.discarded_mtp_writes =
                        (mtp_rows - accepted_draft_tokens) * GLM52_NUM_HIDDEN_LAYERS;
                    mtp_terminal_sample = Some(RealFullSpeculativeTerminalSample {
                        vocab_size: target_samples.vocab_size,
                        top_token_id: target_samples.top_token_ids[terminal_target_index],
                        sampled_token_id: target_token_ids[terminal_target_index],
                        sample_top_k: if request.greedy_sampling {
                            1
                        } else {
                            target_samples.sample_top_k
                        },
                        sample_top_p: if request.greedy_sampling {
                            1.0
                        } else {
                            target_samples.sample_top_p
                        },
                        argmax_backend: target_samples.argmax_kernel_backend,
                        sampler_backend: if request.greedy_sampling {
                            target_samples.argmax_kernel_backend
                        } else {
                            target_samples.sampler_kernel_backend
                        },
                        accepted_draft_tokens,
                        report_mtp_acceptance: true,
                    });
                    if request_timing {
                        eprintln!(
                            "real_full_mtp_verify request_id={} drafts={} accepted={} emitted={} next_drafts={} target_sampled={:?}",
                            request_id,
                            pending_mtp_draft_token_ids.len(),
                            accepted_draft_tokens,
                            cycle_token_ids.len(),
                            draft_tokens,
                            &target_token_ids[..=accepted_draft_tokens],
                        );
                    }
                }
            } else if dspark_active {
                if pending_dspark_draft_token_ids.is_empty() {
                    let anchor_token = if request.greedy_sampling {
                        report.terminal_lm_head_sample.top_token_id
                    } else {
                        report.terminal_lm_head_sample.sampled_token_id
                    }
                    .context("real-full dSpark scalar step requires a target token")
                    .map_err(format_error_chain)?;
                    let committed_rows = if generated_tokens == 0
                        && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                    {
                        dspark_target_hidden_tap_rows
                    } else {
                        1
                    };
                    cycle_token_ids.push(anchor_token);
                    dspark_cache_update = Some((committed_rows, anchor_token));
                } else {
                    let target_hidden = match target_hidden.take() {
                        Some(hidden) => hidden,
                        None => {
                            if let Some(plan) = pending_dspark_plan.take() {
                                if let Some(dspark) = self.dspark.as_ref() {
                                    if let Ok(mut dspark) = dspark.lock() {
                                        dspark.restore_verification(&sequence_id, plan);
                                    }
                                }
                            }
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format!(
                                "real-full dSpark request {} has no target hidden batch for verification",
                                request.request_id
                            ));
                        }
                    };
                    let suffix_rows = decode_rows + pending_dspark_draft_token_ids.len();
                    let target_sampling_start = Instant::now();
                    let sampled_uniforms = (!request.greedy_sampling).then(|| {
                        request_sampling_uniforms(&request, request.decode_step_index, suffix_rows)
                    });
                    let target_samples = match real_full_constraint_target_samples(
                        &self.catalog,
                        &state,
                        &request,
                        &target_hidden,
                        suffix_rows,
                        &pending_dspark_draft_token_ids,
                    )
                    .map_err(format_error_chain)?
                    {
                        Some(samples) => samples,
                        None => if let Some(random_uniforms) = sampled_uniforms.as_deref() {
                            real_full_target_token_samples_with_options(
                                &self.catalog,
                                &target_hidden,
                                suffix_rows,
                                request_lm_head_sampling_options_at(
                                    &request,
                                    request.decode_step_index,
                                ),
                                random_uniforms,
                            )
                        } else {
                            real_full_target_token_samples(
                                &self.catalog,
                                &target_hidden,
                                suffix_rows,
                            )
                        }
                        .map_err(format_error_chain)?,
                    };
                    let target_token_ids = if request.greedy_sampling {
                        target_samples.top_token_ids.as_slice()
                    } else {
                        target_samples.sampled_token_ids.as_slice()
                    };
                    let acceptance = real_full_mtp_acceptance(
                        pending_dspark_draft_token_ids.as_slice(),
                        target_token_ids,
                        true,
                        request.decode_budget.saturating_sub(generated_tokens),
                    )
                    .map_err(format_error_chain)?;
                    let accepted_draft_tokens = acceptance.accepted_draft_tokens;
                    let terminal_target_index = acceptance.terminal_target_index;
                    cycle_token_ids.extend_from_slice(&target_token_ids[..=terminal_target_index]);
                    committed_input_token_ids.extend_from_slice(
                        &pending_dspark_draft_token_ids[..accepted_draft_tokens],
                    );
                    let tentative_token_start =
                        token_rows.prefix_tokens + token_rows.prefill_tokens + decode_rows;
                    state
                        .resolve_mtp_tentative_writes(
                            tentative_token_start,
                            pending_dspark_draft_token_ids.len(),
                            accepted_draft_tokens,
                        )
                        .map_err(format_error_chain)?;
                    report.committed_mtp_writes = accepted_draft_tokens * GLM52_NUM_HIDDEN_LAYERS;
                    report.discarded_mtp_writes = (pending_dspark_draft_token_ids.len()
                        - accepted_draft_tokens)
                        * GLM52_NUM_HIDDEN_LAYERS;
                    // The public counters retain their historical MTP names,
                    // but describe target-verified speculative rows for both
                    // native MTP and dSpark.
                    report.request_mtp_accepted_rows = accepted_draft_tokens;
                    mtp_terminal_sample = Some(RealFullSpeculativeTerminalSample {
                        vocab_size: target_samples.vocab_size,
                        top_token_id: target_samples.top_token_ids[terminal_target_index],
                        sampled_token_id: target_token_ids[terminal_target_index],
                        sample_top_k: if request.greedy_sampling {
                            1
                        } else {
                            target_samples.sample_top_k
                        },
                        sample_top_p: if request.greedy_sampling {
                            1.0
                        } else {
                            target_samples.sample_top_p
                        },
                        argmax_backend: target_samples.argmax_kernel_backend,
                        sampler_backend: if request.greedy_sampling {
                            target_samples.argmax_kernel_backend
                        } else {
                            target_samples.sampler_kernel_backend
                        },
                        accepted_draft_tokens,
                        report_mtp_acceptance: true,
                    });
                    dspark_cache_update = Some((
                        accepted_draft_tokens + 1,
                        target_token_ids[terminal_target_index],
                    ));
                    let plan = pending_dspark_plan
                        .as_ref()
                        .expect("dSpark drafts came from a pending verification plan");
                    let (next_confidence_logit_bias, calibration_variance, calibration_cycles) =
                        self.dspark
                            .as_ref()
                            .expect("active dSpark verification has a request runtime")
                            .lock()
                            .map_err(|error| {
                                format!("locking dSpark confidence calibrator failed: {error}")
                            })?
                            .observe_verification(&sequence_id, plan, accepted_draft_tokens);
                    if real_full_dspark_trace_enabled() {
                        eprintln!(
                            "real_full_dspark_acceptance request_id={} sequence_id={} target_context={} drafts={} accepted={} emitted={} full_match={} target_rows={} expected_tokens={:.4} expected_tps={:.3} confidence_logit_bias={:.4} next_confidence_logit_bias={:.4} calibration_variance={:.6} calibration_cycles={} target_submit_ms={:.3} target_sampling_ms={:.3}",
                            request_id,
                            sequence_id,
                            token_rows.prefix_tokens + token_rows.prefill_tokens,
                            pending_dspark_draft_token_ids.len(),
                            accepted_draft_tokens,
                            cycle_token_ids.len(),
                            acceptance.full_match_bonus,
                            plan.target_batch_rows,
                            plan.expected_committed_tokens,
                            plan.expected_tokens_per_second,
                            plan.confidence_logit_bias,
                            next_confidence_logit_bias,
                            calibration_variance,
                            calibration_cycles,
                            target_submit_ms,
                            elapsed_ms(target_sampling_start),
                        );
                    }
                }
            } else if run_mtp_probe {
                let target_hidden = target_hidden
                    .take()
                    .context("real-full MTP probe requires final target hidden")
                    .map_err(format_error_chain)?;
                let sampled_token_id = if request.greedy_sampling {
                    report.terminal_lm_head_sample.top_token_id
                } else {
                    report.terminal_lm_head_sample.sampled_token_id
                }
                .context("real-full MTP probe requires a target token")
                .map_err(format_error_chain)?;
                let shifted_token_ids = real_full_mtp_shifted_input_token_ids(
                    mtp_target_input_token_ids
                        .as_deref()
                        .expect("MTP probe target input tokens were retained"),
                    sampled_token_id,
                )
                .map_err(format_error_chain)?;
                self.execute_mtp_chain(
                    request.request_index,
                    token_rows.prefix_tokens,
                    request_id_base,
                    shifted_token_ids.as_slice(),
                    target_hidden,
                    1,
                    request.greedy_sampling,
                    Arc::clone(dispatch_worker),
                    None,
                    &mut state.state,
                )
                .map_err(format_error_chain)?;
            }
            if mtp_enabled || dspark_active {
                // Public acceptance metrics describe logical drafts. Lower-level
                // scheduler/expert counters retain the physical padded work.
                report.request_mtp_verify_rows = if mtp_enabled {
                    pending_mtp_draft_token_ids.len()
                } else {
                    pending_dspark_draft_token_ids.len()
                };
            }
            let final_decode_step = if mtp_enabled || dspark_active {
                let emitted_tokens = if cycle_token_ids.is_empty() {
                    1
                } else {
                    cycle_token_ids.len()
                };
                generated_tokens + emitted_tokens >= request.decode_budget
            } else {
                final_decode_step
            };
            if let Some(taps) = dspark_target_hidden_taps {
                if !final_decode_step {
                    let (committed_rows, anchor_token) = if let Some(update) = dspark_cache_update {
                        update
                    } else {
                        let anchor_token = report
                            .terminal_lm_head_sample
                            .top_token_id
                            .context("real-full dSpark shadow step requires a target token")
                            .map_err(format_error_chain)?;
                        (
                            if generated_tokens == 0
                                && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                            {
                                taps.rows
                            } else {
                                1
                            },
                            anchor_token,
                        )
                    };
                    if taps.rows != dspark_target_hidden_tap_rows
                        || committed_rows > taps.rows
                        || taps.layer_ids != real_full_active_draft_target_layer_ids()
                    {
                        let _ = self.store_scheduler_state(&sequence_id, state);
                        return Err(format!(
                            "real-full dSpark expected {} physical target rows with {} committed at layers {:?}, got rows={} layers={:?}",
                            dspark_target_hidden_tap_rows,
                            committed_rows,
                            real_full_active_draft_target_layer_ids(),
                            taps.rows,
                            taps.layer_ids
                        ));
                    }
                    let absolute_context_start = real_full_draft_absolute_context_start(
                        generated_tokens,
                        dspark_cache_mode,
                        token_rows.prefix_tokens,
                        taps.row_start,
                    );
                    let mut dspark = match self
                        .dspark
                        .as_ref()
                        .expect("dSpark taps were retained only with a request executor")
                        .lock()
                    {
                        Ok(dspark) => dspark,
                        Err(error) => {
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format!("locking dSpark request executor failed: {error}"));
                        }
                    };
                    let (step, plan) = match dspark.replay_step(
                        &sequence_id,
                        &taps.values,
                        0,
                        committed_rows,
                        absolute_context_start,
                        anchor_token,
                    ) {
                        Ok(step) => step,
                        Err(error) => {
                            drop(dspark);
                            let _ = self.store_scheduler_state(&sequence_id, state);
                            return Err(format_error_chain(error));
                        }
                    };
                    if generated_tokens == 0
                        && dspark_cache_mode == Some(RealFullDsparkCacheMode::PromptSwa)
                        && !real_full_internal_sequence(&sequence_id)
                    {
                        let prompt_ids = prompt_token_ids
                            .as_deref()
                            .expect("fresh external dSpark request retained prompt token IDs");
                        let reusable_prefix_tokens = prompt_ids.len().saturating_sub(1);
                        if token_rows.prefix_tokens < reusable_prefix_tokens {
                            if let Err(error) = dspark.publish_reusable_prefix(
                                &sequence_id,
                                prompt_ids,
                                reusable_prefix_tokens,
                            ) {
                                drop(dspark);
                                let _ = self.store_scheduler_state(&sequence_id, state);
                                return Err(format_error_chain(error));
                            }
                        }
                    }
                    if real_full_dspark_trace_enabled() {
                        eprintln!(
                            "real_full_dspark_step request_id={} sequence_id={} mode={:?} target_context={} draft_context_before={} committed_rows={} draft_context_after={} anchor_token={} selected_drafts={} target_batch_rows={} expected_tokens={:.4} expected_tps={:.3} update_ms={:.3} suffix_ms={:.3} readback_ms={:.3} dspark_total_ms={:.3} selected_proposals={:?} proposals={:?} confidence={:?}",
                            request_id,
                            sequence_id,
                            dspark.mode,
                            token_rows.prefix_tokens + token_rows.prefill_tokens,
                            step.context_tokens,
                            step.committed_rows,
                            dspark
                                .request_context_tokens(&sequence_id)
                                .expect("the replayed dSpark request has live state"),
                            step.anchor_token,
                            plan.selected_drafts,
                            plan.target_batch_rows,
                            plan.expected_committed_tokens,
                            plan.expected_tokens_per_second,
                            step.update_ms,
                            step.suffix_ms,
                            step.readback_ms,
                            step.total_ms,
                            plan.proposal_token_ids,
                            step.proposal_token_ids,
                            step.conditional_confidence,
                        );
                    }
                }
            }
            if let Some(constraint) = state.constraint.as_mut() {
                if cycle_token_ids.is_empty() {
                    let emitted_token = report
                        .terminal_lm_head_sample
                        .sampled_token_id
                        .context("constrained decode produced no emitted token")
                        .map_err(format_error_chain)?;
                    constraint
                        .commit(std::slice::from_ref(&emitted_token))
                        .map_err(format_error_chain)?;
                } else {
                    constraint
                        .commit(&cycle_token_ids)
                        .map_err(format_error_chain)?;
                }
            }
            state
                .record_processed_token_ids(committed_prefix_tokens, &committed_input_token_ids)
                .map_err(format_error_chain)?;
            self.finish_scheduler_state(&sequence_id, state, final_decode_step)?;
            (report, Some(probe), cycle_token_ids, mtp_terminal_sample)
        } else if let Some(targets) = self.sparse_tcp_targets.as_ref() {
            let (report, probe) = real_full_scheduler_execution_for_shape_with_sparse_tcp(
                self.kv_config.clone(),
                &self.catalog,
                shape,
                targets.clone(),
                self.sparse_owner_lookup.clone(),
                request_id_base,
            )
            .map_err(format_error_chain)?;
            (report, Some(probe), Vec::new(), None)
        } else {
            let mut state = persistent_state
                .take()
                .expect("local scheduler execution has persistent state");
            let execution = real_full_scheduler_execution_for_shape_with_state(
                self.kv_config.clone(),
                &self.catalog,
                shape,
                &mut state,
            );
            let report = match execution {
                Ok(report) => report,
                Err(error) => {
                    let _ = self.store_scheduler_state(&sequence_id, state);
                    return Err(format_error_chain(error));
                }
            };
            state
                .record_processed_token_ids(committed_prefix_tokens, &committed_input_token_ids)
                .map_err(format_error_chain)?;
            self.finish_scheduler_state(&sequence_id, state, final_decode_step)?;
            (report, None, Vec::new(), None)
        };
        if request.greedy_sampling {
            report.terminal_lm_head_sample.sampled_token_id =
                report.terminal_lm_head_sample.top_token_id;
            report.terminal_lm_head_sample.sample_top_k = Some(1);
            report.terminal_lm_head_sample.sample_top_p = Some(1.0);
            report.terminal_lm_head_sample.sampler_kernel_backend =
                report.terminal_lm_head_sample.argmax_kernel_backend;
        }
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=scheduler elapsed_ms={:.3} total_ms={:.3} status={} sparse_batches={} host_batches={} sample_status={}",
                request_id,
                elapsed_ms(scheduler_start),
                elapsed_ms(request_start),
                report.status,
                report.sparse_expert_batches,
                report.sparse_expert_host_batches,
                report.terminal_lm_head_sample.status
            );
        }
        if let Some(sample) = mtp_terminal_sample.as_ref() {
            apply_speculative_terminal_sample_to_report(&mut report, sample);
        }
        let info_start = Instant::now();
        let sampled_token_text =
            self.decode_sampled_token_text_cached(report.terminal_lm_head_sample.sampled_token_id);
        let mut info = real_full_info_from_request_execution(
            &self.base_info,
            &self.catalog.snapshot_path,
            &report,
            sampled_token_text,
        );
        info.request_kv_snapshot_restore_ms = snapshot_restore_ms;
        if let Some(probe) = sparse_tcp_dispatch.as_ref() {
            apply_sparse_tcp_dispatch_probe(
                &mut info,
                self.sparse_tcp_targets.as_ref().map_or(0, Vec::len),
                probe,
            );
        }
        if let Some(sample) = mtp_terminal_sample.as_ref() {
            if sample.report_mtp_acceptance {
                info.request_mtp_accepted_rows = sample.accepted_draft_tokens;
            }
            info.scheduler_terminal_lm_head_top_token_id = Some(sample.top_token_id);
            info.scheduler_terminal_lm_head_sampled_token_id = Some(sample.sampled_token_id);
            info.scheduler_terminal_lm_head_sampled_text =
                self.decode_sampled_token_text_cached(Some(sample.sampled_token_id));
            info.scheduler_terminal_lm_head_sample_top_k = Some(sample.sample_top_k);
            info.scheduler_terminal_lm_head_sample_top_p = Some(sample.sample_top_p);
            info.scheduler_terminal_lm_head_argmax_backend = Some(sample.argmax_backend.to_owned());
            info.scheduler_terminal_lm_head_sampler_backend =
                Some(sample.sampler_backend.to_owned());
        }
        if request_timing {
            eprintln!(
                "real_full_request_timing request_id={} stage=info elapsed_ms={:.3} total_ms={:.3} status={} sampled_token_id={:?}",
                request_id,
                elapsed_ms(info_start),
                elapsed_ms(request_start),
                info.status,
                info.scheduler_terminal_lm_head_sampled_token_id
            );
        }
        let generated_tokens = cycle_token_ids
            .into_iter()
            .map(|token_id| glmrt_api::RealFullGeneratedToken {
                token_id,
                text: self.decode_sampled_token_text_cached(Some(token_id)),
            })
            .collect();
        if dspark_active
            && pending_dspark_plan
                .as_ref()
                .is_some_and(|plan| plan.calibration_eligible)
            && info.request_coordinator_graph_captures == 0
        {
            let route_profile = sparse_tcp_dispatch
                .as_ref()
                .and_then(|probe| dspark_runtime_route_profile_from_probes(std::iter::once(probe)));
            let observation = self
                .dspark
                .as_ref()
                .expect("adaptive dSpark cost observation requires a runtime")
                .lock()
                .map_err(|error| format!("locking dSpark runtime cost model failed: {error}"))
                .and_then(|mut dspark| {
                    dspark
                        .observe_runtime_cost(
                            &[token_rows.prefix_tokens + token_rows.prefill_tokens],
                            decode_rows + pending_dspark_draft_token_ids.len(),
                            elapsed_ms(request_start),
                            route_profile
                                .as_ref()
                                .map(|route| route.critical_unique_experts),
                        )
                        .map_err(format_error_chain)
                });
            match observation {
                Ok(observation) if real_full_dspark_trace_enabled() => {
                    log_dspark_runtime_cost(&observation, route_profile.as_ref());
                }
                Ok(_) => {}
                Err(error) => {
                    eprintln!("real_full_dspark_runtime_cost_ignored error={error}");
                }
            }
        }
        Ok(glmrt_api::RealFullDecodeCycle {
            info,
            generated_tokens,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DsparkRuntimeRouteProfile {
    wire_batches: usize,
    route_assignments: usize,
    unique_experts: usize,
    critical_unique_experts: usize,
    reused_assignments: usize,
    max_expert_load: usize,
    load_square_sum: usize,
}

fn dspark_runtime_route_profile_from_probes<'a>(
    probes: impl IntoIterator<Item = &'a RealFullSchedulerSparseTcpDispatchProbe>,
) -> Option<DsparkRuntimeRouteProfile> {
    let mut profile = DsparkRuntimeRouteProfile::default();
    let mut observed = false;
    for probe in probes {
        if probe.route_profiled_wire_batches == 0 {
            return None;
        }
        observed = true;
        profile.wire_batches = profile
            .wire_batches
            .saturating_add(probe.route_profiled_wire_batches);
        profile.route_assignments = profile
            .route_assignments
            .saturating_add(probe.route_profiled_assignments);
        profile.unique_experts = profile
            .unique_experts
            .saturating_add(probe.route_profiled_unique_experts);
        profile.critical_unique_experts = profile
            .critical_unique_experts
            .max(probe.route_profiled_unique_experts);
        profile.reused_assignments = profile
            .reused_assignments
            .saturating_add(probe.route_profiled_reused_assignments);
        profile.max_expert_load = profile
            .max_expert_load
            .max(probe.route_profiled_max_expert_load);
        profile.load_square_sum = profile
            .load_square_sum
            .saturating_add(probe.route_profiled_load_square_sum);
    }
    observed.then_some(profile)
}

fn dspark_runtime_route_profile_for_batched_executions(
    executions: &[RealFullSchedulerDeviceExecution],
) -> Option<DsparkRuntimeRouteProfile> {
    // Four- and eight-lane recurrent execution physically merges adjacent
    // request pairs into one expert wire cohort.  The even member's probe owns
    // the combined dispatch accounting; including the odd member would count
    // its request-local view a second time.  C2/C3 use independent wire
    // cohorts, so every probe contributes.
    if matches!(executions.len(), 4 | 8) {
        dspark_runtime_route_profile_from_probes(
            executions
                .iter()
                .step_by(2)
                .map(|execution| &execution.sparse_tcp_dispatch),
        )
    } else {
        dspark_runtime_route_profile_from_probes(
            executions
                .iter()
                .map(|execution| &execution.sparse_tcp_dispatch),
        )
    }
}

fn log_dspark_runtime_cost(
    observation: &DsparkRuntimeCostObservation,
    route_profile: Option<&DsparkRuntimeRouteProfile>,
) {
    if let Some(route) = route_profile {
        eprintln!(
            "real_full_dspark_runtime_cost requests={} context_work_bucket={} max_context_bucket={} target_rows={} observed_ms={:.3} predicted_ms_before={:.3} exact_samples={} route_wire_batches={} route_assignments={} route_unique_experts={} route_critical_unique_experts={} route_reused_assignments={} route_max_expert_load={} route_load_square_sum={}",
            observation.request_count,
            observation.context_work_bucket,
            observation.max_context_bucket,
            observation.target_rows,
            observation.observed_ms,
            observation.predicted_ms_before,
            observation.exact_samples,
            route.wire_batches,
            route.route_assignments,
            route.unique_experts,
            route.critical_unique_experts,
            route.reused_assignments,
            route.max_expert_load,
            route.load_square_sum,
        );
    } else {
        eprintln!(
            "real_full_dspark_runtime_cost requests={} context_work_bucket={} max_context_bucket={} target_rows={} observed_ms={:.3} predicted_ms_before={:.3} exact_samples={} route_profile=unavailable",
            observation.request_count,
            observation.context_work_bucket,
            observation.max_context_bucket,
            observation.target_rows,
            observation.observed_ms,
            observation.predicted_ms_before,
            observation.exact_samples,
        );
    }
}

fn real_full_sequence_capacity_tokens(
    prompt_tokens: usize,
    decode_budget: usize,
    max_context_tokens: usize,
) -> Result<usize> {
    let required_tokens = prompt_tokens
        .checked_add(decode_budget.max(1))
        .context("real-full sequence token capacity overflow")?;
    anyhow::ensure!(
        required_tokens <= max_context_tokens,
        "real-full sequence requires {required_tokens} context tokens but the global maximum is {max_context_tokens}"
    );
    let extension_headroom = prompt_tokens.min(REAL_FULL_SEQUENCE_EXTENSION_HEADROOM_TOKENS);
    Ok(required_tokens
        .saturating_add(extension_headroom)
        .min(max_context_tokens))
}

fn real_full_internal_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with("real-full-startup-")
}

fn real_full_startup_target_radix_publish_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix(REAL_FULL_STARTUP_TARGET_RADIX_PUBLISH_PREFIX)
        .or_else(|| sequence_id.strip_prefix(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX))
        .and_then(|suffix| suffix.split_once("-sequence-").map(|(tokens, _)| tokens))
        .and_then(|tokens| tokens.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
}

fn real_full_startup_target_radix_evict_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix(REAL_FULL_STARTUP_TARGET_RADIX_EVICT_PREFIX)
        .and_then(|suffix| suffix.split_once("-worker-").map(|(tokens, _)| tokens))
        .and_then(|tokens| tokens.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
}

fn real_full_startup_workspace_sizing_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with("real-full-startup-capture-arena-")
}

fn real_full_seal_owned_buffer_pool_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with(REAL_FULL_STARTUP_SEAL_OWNED_BUFFER_POOL_PREFIX)
}

fn real_full_prewarm_paired_lm_head_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with(REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX)
}

fn real_full_prewarm_batched_dspark_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with(REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX)
}

fn real_full_nvfp4_short_k_graph_audit(sequence_id: &str) -> Option<(usize, usize)> {
    let suffix = sequence_id.strip_prefix(REAL_FULL_STARTUP_AUDIT_NVFP4_SHORT_K_PREFIX)?;
    let (query_rows, sparse_topk_suffix) = suffix.split_once("-k")?;
    let query_rows = query_rows.parse::<usize>().ok()?;
    let (sparse_topk, worker_index) = sparse_topk_suffix.split_once("-worker-")?;
    let sparse_topk = sparse_topk.parse::<usize>().ok()?;
    worker_index.parse::<usize>().ok()?;
    (REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS.contains(&query_rows)
        && REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_CAPTURE_ANCHORS
            .iter()
            .any(|anchor| anchor.sparse_topk == sparse_topk))
    .then_some((query_rows, sparse_topk))
}

fn real_full_paired_lm_head_prewarm_range(fixed_drafts: Option<usize>) -> Option<(usize, usize)> {
    let max_single_rows = match fixed_drafts {
        Some(0) => return None,
        Some(fixed_drafts) => fixed_drafts + 1,
        None => REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS + 1,
    };
    Some((max_single_rows + 1, 2 * max_single_rows))
}

fn real_full_dspark_startup_draft_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix("real-full-startup-dspark-width-")
        .and_then(|suffix| suffix.split('-').next())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|draft_tokens| *draft_tokens <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS)
}

fn real_full_batched_dspark_prewarm_buffer_bank(sequence_id: &str) -> Option<usize> {
    sequence_id
        .split_once(REAL_FULL_STARTUP_BATCHED_DSPARK_BANK_MARKER)
        .and_then(|(_, suffix)| suffix.split('-').next())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|buffer_bank| *buffer_bank > 0)
}

fn real_full_batched_dspark_prewarm_sequence(sequence_id: &str) -> bool {
    real_full_dspark_startup_draft_tokens(sequence_id).is_some_and(|draft_tokens| draft_tokens > 0)
        && real_full_batched_dspark_prewarm_buffer_bank(sequence_id).is_some()
}

const REAL_FULL_DFLASH2_DSA_PREWARM_PROMPT_REPEATS: usize = 2_048;

fn real_full_draft_width_prewarm_prompt_repeats(dflash2: bool) -> usize {
    if dflash2 {
        REAL_FULL_DFLASH2_DSA_PREWARM_PROMPT_REPEATS
    } else {
        8
    }
}

fn real_full_draft_width_prewarm_passes(dflash2: bool) -> usize {
    if dflash2 {
        2
    } else {
        1
    }
}

const REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE: u64 = 92_000;
const REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE: u64 = 100;
const REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE: u64 = 91_000;
const REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE: u64 = 100;
// Paired LM-head sampling has a different live-buffer lifetime than the
// per-lane recurrent scheduler. Keeping it in a disjoint pool namespace makes
// both sets of CUDA graph pointer identities stable without a final recapture
// pass. Production supports at most eight execution lanes, so this range does
// not overlap the ordinary lane banks (1..=8).
const REAL_FULL_PAIRED_LM_HEAD_BUFFER_BANK_BASE: usize = 16;

fn real_full_paired_lm_head_buffer_bank(execution_buffer_bank: usize) -> usize {
    REAL_FULL_PAIRED_LM_HEAD_BUFFER_BANK_BASE + execution_buffer_bank
}

fn real_full_scalar_dspark_prewarm_requested_draft_tokens(
    sequence_id: &str,
    request_index: u64,
) -> Option<usize> {
    sequence_id
        .contains(REAL_FULL_STARTUP_SCALAR_DSPARK_COHORT_MARKER)
        .then_some(())?;
    let encoded = request_index.checked_sub(REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE)?
        / REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE;
    usize::try_from(encoded)
        .ok()
        .filter(|draft_tokens| *draft_tokens <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS)
}

fn real_full_batched_dspark_prewarm_requested_draft_tokens(
    sequence_id: &str,
    request_index: u64,
) -> Option<usize> {
    if !real_full_batched_dspark_prewarm_sequence(sequence_id) {
        return None;
    }
    let encoded = request_index.checked_sub(REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE)?
        / REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE;
    usize::try_from(encoded)
        .ok()
        .filter(|draft_tokens| *draft_tokens <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS)
}

fn real_full_capture_arena_sequence(sequence_id: &str) -> bool {
    sequence_id.starts_with("real-full-startup-capture-arena-")
        || sequence_id.starts_with("real-full-startup-dsa-selector-seed-")
        || sequence_id.starts_with("real-full-startup-prefix-prefill-seed-")
        || sequence_id.starts_with("real-full-startup-mtp-production-")
        || sequence_id.starts_with("real-full-startup-dspark-width-")
}

impl glmrt_api::RealFullRequestExecutor for RealFullSchedulerRequestExecutor {
    fn execute_real_full_request(
        &self,
        request: glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullInfo, String> {
        self.execute_real_full_decode_cycle_inner(request)
            .map(|cycle| cycle.info)
    }

    fn execute_real_full_decode_cycle(
        &self,
        request: glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String> {
        self.execute_real_full_decode_cycle_inner(request)
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        request: &glmrt_api::RealFullRequest,
    ) -> Option<Duration> {
        let dspark_active = self.dspark.as_ref().is_some_and(|runtime| {
            runtime
                .lock()
                .map(|runtime| runtime.mode == RealFullDsparkServingMode::Active)
                .unwrap_or(false)
        });
        (dspark_active
            && self.sparse_tcp_dispatch_worker.is_some()
            && self.max_execution_lanes > 1
            && !real_full_mtp_enabled()
            && request.greedy_sampling
            && !request.disable_speculation
            && !real_full_internal_sequence(&request.sequence_id))
        .then(|| {
            // Near-simultaneous initial requests may enter one persistent prefill
            // wave. This is an idle-start admission quantum; recurrent work is
            // coordinator-owned and never depends on this arrival window.
            if request.generated_token_ids.is_empty() {
                Duration::from_millis(50)
            } else {
                Duration::from_micros(100)
            }
        })
    }

    fn real_full_decode_cycle_batch_max_size(
        &self,
        _request: &glmrt_api::RealFullRequest,
    ) -> usize {
        self.max_execution_lanes
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        REAL_FULL_MAX_ACTIVE_REQUESTS
    }

    fn real_full_retryable_admission_error(
        &self,
        _request: &glmrt_api::RealFullRequest,
        error: &str,
    ) -> bool {
        error.contains("target KV guaranteed capacity exhausted")
            || error.contains("target KV active request limit exhausted")
            || error.contains("configured real-full execution lanes are resident")
            || error.contains("real-full global context budget exhausted")
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<glmrt_api::RealFullRequest>,
    ) -> Vec<std::result::Result<glmrt_api::RealFullDecodeCycle, String>> {
        if !self.batched_dspark_cycles_eligible(&requests) {
            return requests
                .into_iter()
                .map(|request| self.execute_real_full_decode_cycle_inner(request))
                .collect();
        }

        let batch_start = Instant::now();
        let retry_requests = requests.clone();
        let mut prepared = Vec::with_capacity(requests.len());
        for (request_index, request) in requests.into_iter().enumerate() {
            match self.prepare_batched_dspark_cycle(request) {
                Ok(cycle) => prepared.push(cycle),
                Err(error) => {
                    for cycle in prepared.drain(..) {
                        self.restore_prepared_batched_dspark_cycle(cycle);
                    }
                    return retry_requests
                        .into_iter()
                        .enumerate()
                        .map(|(retry_index, request)| {
                            if retry_index == request_index {
                                Err(error.clone())
                            } else {
                                self.execute_real_full_decode_cycle_inner(request)
                            }
                        })
                        .collect();
                }
            }
        }
        if let Err(error) = self.replan_prepared_batched_dspark_cycles(&mut prepared) {
            let request_count = prepared.len();
            for cycle in prepared {
                self.restore_prepared_batched_dspark_cycle(cycle);
            }
            return (0..request_count).map(|_| Err(error.clone())).collect();
        }
        let runtime_cost_contexts = prepared
            .iter()
            .map(|cycle| cycle.token_prefix_tokens + cycle.token_prefill_tokens)
            .collect::<Vec<_>>();
        let runtime_cost_target_rows = prepared
            .iter()
            .map(|cycle| cycle.decode_rows + cycle.pending_dspark_draft_token_ids.len())
            .sum::<usize>();
        let runtime_cost_eligible = prepared.iter().all(|cycle| {
            cycle
                .pending_dspark_plan
                .as_ref()
                .is_some_and(|plan| plan.calibration_eligible)
        });

        let scheduler_inputs = prepared
            .iter_mut()
            .map(|cycle| RealFullSchedulerBatchedInput {
                shape: cycle.shape.clone(),
                request_id_base: cycle.request_id_base,
                state: &mut cycle.state,
                buffer_bank: cycle.buffer_bank,
                retain_final_target_device_hidden: cycle.pending_dspark_plan.is_some(),
                target_device_hidden_tap_rows: cycle.dspark_target_hidden_tap_rows,
            })
            .collect();
        let execution =
            real_full_scheduler_execution_for_batched_shapes_with_shared_sparse_tcp_and_state_device_hidden(
            self.kv_config.clone(),
            &self.catalog,
            scheduler_inputs,
            Arc::clone(
                self.sparse_tcp_dispatch_worker
                    .as_ref()
                    .expect("batched dSpark cycles require a sparse dispatch worker"),
            ),
        );
        let executions = match execution {
            Ok(executions) => executions,
            Err(error) => {
                let error = format_error_chain(error);
                let request_count = prepared.len();
                for cycle in prepared {
                    self.restore_prepared_batched_dspark_cycle(cycle);
                }
                return (0..request_count).map(|_| Err(error.clone())).collect();
            }
        };
        let runtime_route_profile =
            dspark_runtime_route_profile_for_batched_executions(&executions);

        let mut paired_target_samples = (0..prepared.len()).map(|_| None).collect::<Vec<_>>();
        for pair_start in (0..prepared.len()).step_by(2) {
            let pair_end = pair_start + 1;
            if pair_end >= prepared.len() {
                break;
            }
            let cycle_a = &prepared[pair_start];
            let cycle_b = &prepared[pair_end];
            let suffix_rows_a = cycle_a.decode_rows + cycle_a.pending_dspark_draft_token_ids.len();
            let suffix_rows_b = cycle_b.decode_rows + cycle_b.pending_dspark_draft_token_ids.len();
            if cycle_a.pending_dspark_draft_token_ids.is_empty()
                || cycle_b.pending_dspark_draft_token_ids.is_empty()
                || suffix_rows_a.saturating_add(suffix_rows_b) > 32
            {
                continue;
            }
            let samples = with_coordinator_owned_device_buffer_bank(
                real_full_paired_lm_head_buffer_bank(cycle_a.buffer_bank),
                || {
                    let hidden_a = executions[pair_start]
                        .final_target_device_hidden
                        .as_ref()
                        .with_context(|| {
                            format!(
                                "batched dSpark request {pair_start} has no retained target hidden rows"
                            )
                        })?;
                    let hidden_b = executions[pair_end]
                        .final_target_device_hidden
                        .as_ref()
                        .with_context(|| {
                            format!(
                            "batched dSpark request {pair_end} has no retained target hidden rows"
                        )
                        })?;
                    real_full_target_token_samples_pair(
                        &self.catalog,
                        hidden_a,
                        suffix_rows_a,
                        hidden_b,
                        suffix_rows_b,
                    )
                },
            );
            match samples {
                Ok((sample_a, sample_b)) => {
                    paired_target_samples[pair_start] = Some(sample_a);
                    paired_target_samples[pair_end] = Some(sample_b);
                }
                Err(error) => {
                    let error = format_error_chain(error);
                    let request_count = prepared.len();
                    for cycle in prepared {
                        self.restore_prepared_batched_dspark_cycle(cycle);
                    }
                    return (0..request_count).map(|_| Err(error.clone())).collect();
                }
            }
        }

        let batched_dflash_cache_mode = self
            .dspark
            .as_ref()
            .and_then(|runtime| runtime.lock().ok())
            .and_then(|runtime| {
                runtime
                    .batched_dflash_enabled()
                    .then_some(runtime.cache_mode)
            });
        let mut precomputed_draft_replays = (0..prepared.len()).map(|_| None).collect::<Vec<_>>();
        if let Some(dflash_cache_mode) = batched_dflash_cache_mode {
            // The target sampler is still request-shaped. Resolve any samples
            // not already handled by the paired LM-head path before deriving
            // the committed DFlash2 cache rows for the collective suffix.
            for index in 0..prepared.len() {
                if prepared[index].pending_dspark_draft_token_ids.is_empty()
                    || paired_target_samples[index].is_some()
                {
                    continue;
                }
                let suffix_rows = prepared[index].decode_rows
                    + prepared[index].pending_dspark_draft_token_ids.len();
                let sample = with_coordinator_owned_device_buffer_bank(
                    prepared[index].buffer_bank,
                    || {
                        let hidden = executions[index]
                            .final_target_device_hidden
                            .as_ref()
                            .with_context(|| {
                                format!(
                                    "batched DFlash2 request {index} has no retained target hidden rows"
                                )
                            })?;
                        real_full_target_token_samples(&self.catalog, hidden, suffix_rows)
                    },
                );
                match sample {
                    Ok(sample) => paired_target_samples[index] = Some(sample),
                    Err(error) => {
                        let error = format_error_chain(error);
                        let request_count = prepared.len();
                        for cycle in prepared {
                            self.restore_prepared_batched_dspark_cycle(cycle);
                        }
                        return (0..request_count).map(|_| Err(error.clone())).collect();
                    }
                }
            }

            let mut candidates = Vec::new();
            for index in 0..prepared.len() {
                let cycle = &prepared[index];
                let cache_update = if cycle.pending_dspark_draft_token_ids.is_empty() {
                    executions[index]
                        .report
                        .terminal_lm_head_sample
                        .top_token_id
                        .context("paired DFlash2 scalar step requires a target token")
                        .map(|anchor_token| (1, anchor_token))
                } else {
                    let target_samples = paired_target_samples[index]
                        .as_ref()
                        .expect("DFlash2 target sampling was completed above");
                    real_full_mtp_acceptance(
                        &cycle.pending_dspark_draft_token_ids,
                        &target_samples.top_token_ids,
                        true,
                        cycle
                            .request
                            .decode_budget
                            .saturating_sub(cycle.generated_tokens),
                    )
                    .map(|acceptance| {
                        (
                            acceptance.accepted_draft_tokens + 1,
                            target_samples.top_token_ids[acceptance.terminal_target_index],
                        )
                    })
                };
                let (committed_rows, anchor_token) = match cache_update {
                    Ok(update) => update,
                    Err(error) => {
                        let error = format_error_chain(error);
                        let request_count = prepared.len();
                        for cycle in prepared {
                            self.restore_prepared_batched_dspark_cycle(cycle);
                        }
                        return (0..request_count).map(|_| Err(error.clone())).collect();
                    }
                };
                let final_decode_step =
                    cycle.generated_tokens + committed_rows >= cycle.request.decode_budget;
                let taps_match = executions[index]
                    .target_device_hidden_taps
                    .as_ref()
                    .is_some_and(|taps| {
                        taps.rows == cycle.dspark_target_hidden_tap_rows
                            && committed_rows <= taps.rows
                            && taps.layer_ids == real_full_active_draft_target_layer_ids()
                    });
                if !final_decode_step && taps_match {
                    let taps = executions[index]
                        .target_device_hidden_taps
                        .as_ref()
                        .expect("a matched DFlash2 candidate has target taps");
                    let absolute_context_start = real_full_draft_absolute_context_start(
                        cycle.generated_tokens,
                        Some(dflash_cache_mode),
                        cycle.token_prefix_tokens,
                        taps.row_start,
                    );
                    candidates.push((index, committed_rows, anchor_token, absolute_context_start));
                }
            }

            let mut cursor = 0;
            while let Some(group_size) =
                dflash_batch_group_size(candidates.len().saturating_sub(cursor))
            {
                let remaining = candidates.len() - cursor;
                debug_assert!(group_size <= remaining);
                let group = &candidates[cursor..cursor + group_size];
                let inputs = group
                    .iter()
                    .map(
                        |(index, committed_rows, anchor_token, absolute_context_start)| {
                            let taps = executions[*index]
                                .target_device_hidden_taps
                                .as_ref()
                                .expect("a DFlash2 batch candidate has target taps");
                            RealFullBatchedDraftReplayInput {
                                sequence_id: &prepared[*index].sequence_id,
                                target_hidden_taps: &taps.values,
                                target_row_start: 0,
                                committed_rows: *committed_rows,
                                absolute_context_start: *absolute_context_start,
                                anchor_token: *anchor_token,
                            }
                        },
                    )
                    .collect::<Vec<_>>();
                let replay = self
                    .dspark
                    .as_ref()
                    .expect("DFlash2 batching requires a runtime")
                    .lock()
                    .map_err(|error| format!("locking batched DFlash2 runtime failed: {error}"))
                    .and_then(|mut runtime| {
                        runtime
                            .replay_batched_dflash_steps(&inputs)
                            .map_err(format_error_chain)
                    });
                let replay = match replay {
                    Ok(replay) => replay,
                    Err(error) => {
                        let request_count = prepared.len();
                        for cycle in prepared {
                            self.restore_prepared_batched_dspark_cycle(cycle);
                        }
                        return (0..request_count).map(|_| Err(error.clone())).collect();
                    }
                };
                for ((index, _, _, _), replay) in group.iter().zip(replay) {
                    precomputed_draft_replays[*index] = Some(replay);
                }
                cursor += group_size;
            }
        }

        let results = prepared
            .into_iter()
            .zip(executions)
            .zip(paired_target_samples)
            .zip(precomputed_draft_replays)
            .map(
                |(((cycle, execution), target_samples), precomputed_replay)| {
                    with_coordinator_owned_device_buffer_bank(cycle.buffer_bank, || {
                        self.finish_batched_dspark_cycle(
                            cycle,
                            execution,
                            target_samples,
                            precomputed_replay,
                        )
                    })
                },
            )
            .collect::<Vec<_>>();
        let runtime_cost_clean = results.iter().all(|result| {
            result
                .as_ref()
                .is_ok_and(|cycle| cycle.info.request_coordinator_graph_captures == 0)
        });
        if runtime_cost_eligible && runtime_cost_clean {
            if let Some(dspark) = self.dspark.as_ref() {
                match dspark
                    .lock()
                    .map_err(|error| format!("locking dSpark runtime cost model failed: {error}"))
                    .and_then(|mut dspark| {
                        dspark
                            .observe_runtime_cost(
                                &runtime_cost_contexts,
                                runtime_cost_target_rows,
                                elapsed_ms(batch_start),
                                runtime_route_profile
                                    .as_ref()
                                    .map(|route| route.critical_unique_experts),
                            )
                            .map_err(format_error_chain)
                    }) {
                    Ok(observation) if real_full_dspark_trace_enabled() => {
                        log_dspark_runtime_cost(&observation, runtime_route_profile.as_ref());
                    }
                    Ok(_) => {}
                    Err(error) => {
                        eprintln!("real_full_dspark_runtime_cost_ignored error={error}");
                    }
                }
            }
        }
        results
    }

    fn prewarm_dflash2_dsa_lane_graphs(
        &self,
        max_draft_tokens: usize,
    ) -> std::result::Result<(), String> {
        let prompts = vec![
            REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN
                .repeat(real_full_draft_width_prewarm_prompt_repeats(true));
            self.max_execution_lanes
        ];
        let prompt_tokens = {
            let tokenizer = self.tokenizer.lock().map_err(|error| {
                format!("locking DFlash2 DSA prewarm tokenizer failed: {error}")
            })?;
            prompts
                .iter()
                .map(|prompt| {
                    tokenizer
                        .encode_text(prompt, false)
                        .map_err(format_error_chain)
                        .map(|encoded| encoded.token_count)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        if prompt_tokens
            .iter()
            .any(|prompt_tokens| *prompt_tokens <= 2_048)
        {
            return Err(format!(
                "DFlash2 startup width prewarm must cross the 2,048-token DSA selector boundary, got prompt token counts {prompt_tokens:?}"
            ));
        }

        // Seed all production lanes together so each persistent scheduler
        // state owns the same buffer bank that it will use while serving.
        // Run the width passes one lane at a time afterward: a multi-request
        // suffix is intentionally classified as direct DSA prefill, whereas
        // speculative verification uses the scalar-QA/batched-QB decode path
        // whose per-layer pointer identities must be retained here.
        let sequence_ids = (1..=self.max_execution_lanes)
            .map(|buffer_bank| {
                format!(
                    "real-full-startup-dspark-width-{max_draft_tokens}-batched-bank-{buffer_bank}-dflash2-dsa-sequence"
                )
            })
            .collect::<Vec<_>>();
        let decode_budget = 1_024;
        let start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start stage=dflash2-dsa-lane-widths lanes={} prompt_tokens={} max_drafts={} passes={}",
            self.max_execution_lanes,
            prompt_tokens.iter().sum::<usize>(),
            max_draft_tokens,
            real_full_draft_width_prewarm_passes(true),
        );

        let prewarm_result = (|| {
            let seed_requests = sequence_ids
                .iter()
                .zip(&prompts)
                .zip(&prompt_tokens)
                .enumerate()
                .map(|(lane_index, ((sequence_id, prompt), prompt_tokens))| {
                    glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                        90_100 + lane_index as u64,
                        sequence_id,
                        prompt,
                        *prompt_tokens,
                        1,
                        Vec::new(),
                        0,
                        decode_budget,
                    )
                })
                .collect::<Vec<_>>();
            let seed_start = Instant::now();
            let initial_cycles =
                glmrt_api::RealFullRequestExecutor::execute_real_full_decode_cycle_batch(
                    self,
                    seed_requests,
                )
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut generated_token_ids = Vec::with_capacity(sequence_ids.len());
            for (lane_index, initial_cycle) in initial_cycles.into_iter().enumerate() {
                if initial_cycle.info.status != "ready" {
                    return Err(format!(
                        "DFlash2 DSA lane seed {lane_index} failed: status={} blocker={} failed={:?}",
                        initial_cycle.info.status,
                        initial_cycle.info.blocker,
                        initial_cycle.info.failed_requirements,
                    ));
                }
                let token_id = initial_cycle
                    .generated_tokens
                    .first()
                    .map(|token| token.token_id)
                    .or(initial_cycle
                        .info
                        .scheduler_terminal_lm_head_sampled_token_id)
                    .ok_or_else(|| {
                        format!("DFlash2 DSA lane seed {lane_index} produced no token")
                    })?;
                generated_token_ids.push(vec![token_id]);
            }
            eprintln!(
                "real_full_startup_prewarm_step_done stage=dflash2-dsa-lane-seed lanes={} elapsed_ms={:.3} total_ms={:.3}",
                self.max_execution_lanes,
                elapsed_ms(seed_start),
                elapsed_ms(start),
            );

            let width_passes = real_full_draft_width_prewarm_passes(true);
            for draft_tokens in (0..=max_draft_tokens).rev() {
                for pass_index in 0..width_passes {
                    for lane_index in 0..self.max_execution_lanes {
                        let width_start = Instant::now();
                        let request_index = REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE
                            + (draft_tokens as u64)
                                * REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE
                            + (pass_index * 10 + lane_index) as u64;
                        let decode_step_index = generated_token_ids[lane_index].len();
                        let cycle = self.execute_real_full_decode_cycle_inner(
                            glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                                request_index,
                                &sequence_ids[lane_index],
                                &prompts[lane_index],
                                prompt_tokens[lane_index],
                                1,
                                generated_token_ids[lane_index].clone(),
                                decode_step_index,
                                decode_budget,
                            ),
                        )?;
                        if cycle.info.status != "ready"
                            || cycle.info.request_mtp_verify_rows != draft_tokens
                        {
                            return Err(format!(
                                "DFlash2 DSA M={} pass {} lane {} failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
                                draft_tokens + 1,
                                pass_index + 1,
                                lane_index,
                                cycle.info.status,
                                cycle.info.request_mtp_verify_rows,
                                draft_tokens,
                                cycle.info.blocker,
                                cycle.info.failed_requirements,
                            ));
                        }
                        if cycle.generated_tokens.is_empty() {
                            return Err(format!(
                                "DFlash2 DSA M={} pass {} lane {} emitted no tokens",
                                draft_tokens + 1,
                                pass_index + 1,
                                lane_index,
                            ));
                        }
                        generated_token_ids[lane_index].extend(
                            cycle
                                .generated_tokens
                                .into_iter()
                                .map(|token| token.token_id),
                        );
                        eprintln!(
                            "real_full_startup_prewarm_step_done stage=dflash2-dsa-lane-width lane={} pass={} physical_m={} reported_capture_delta={} elapsed_ms={:.3} total_ms={:.3}",
                            lane_index + 1,
                            pass_index + 1,
                            draft_tokens + 1,
                            cycle.info.request_coordinator_graph_captures,
                            elapsed_ms(width_start),
                            elapsed_ms(start),
                        );
                    }
                }
            }
            Ok(())
        })();

        let cleanup_errors = sequence_ids
            .iter()
            .filter_map(|sequence_id| {
                self.finish_real_full_sequence(sequence_id)
                    .err()
                    .map(|error| format!("{sequence_id}: {error}"))
            })
            .collect::<Vec<_>>();
        if let Err(error) = prewarm_result {
            if cleanup_errors.is_empty() {
                return Err(error);
            }
            return Err(format!(
                "{error}; DFlash2 DSA startup cleanup also failed: {}",
                cleanup_errors.join("; ")
            ));
        }
        if !cleanup_errors.is_empty() {
            return Err(format!(
                "DFlash2 DSA startup cleanup failed: {}",
                cleanup_errors.join("; ")
            ));
        }
        eprintln!(
            "real_full_startup_prewarm_done stage=dflash2-dsa-lane-widths lanes={} max_physical_m={} elapsed_ms={:.3}",
            self.max_execution_lanes,
            max_draft_tokens + 1,
            elapsed_ms(start),
        );
        Ok(())
    }

    fn prewarm_batched_dspark_graphs(&self) -> std::result::Result<(), String> {
        let startup_profile_mode =
            real_full_dspark_startup_profile_mode().map_err(format_error_chain)?;
        let startup_profile_samples =
            if startup_profile_mode == RealFullDsparkStartupProfileMode::Disabled {
                0
            } else {
                real_full_dspark_startup_profile_samples().map_err(format_error_chain)?
            };
        let Some(dspark) = self.dspark.as_ref() else {
            return Ok(());
        };
        let (checkpoint_max_drafts, dflash2) = dspark
            .lock()
            .map(|runtime| (runtime.engine.max_verify_drafts(), runtime.is_dflash2()))
            .map_err(|error| format!("locking draft startup prewarm failed: {error}"))?;
        let max_draft_tokens = if dflash2 {
            real_full_dflash2_fixed_drafts()
                .map_err(format_error_chain)?
                .unwrap_or(checkpoint_max_drafts)
        } else {
            match real_full_dspark_fixed_drafts().map_err(format_error_chain)? {
                Some(draft_tokens) => {
                    if draft_tokens > checkpoint_max_drafts {
                        return Err(format!(
                            "fixed dSpark width {draft_tokens} exceeds the active checkpoint maximum {checkpoint_max_drafts}"
                        ));
                    }
                    draft_tokens
                }
                None => checkpoint_max_drafts,
            }
        };
        // Serial width capture explicitly finishes each sequence. Defensively
        // drain any other internal prewarm states before materializing the
        // production lanes; a retained target-radix reservation would
        // silently reduce the serving admission pool.
        let stale_internal_sequences = self
            .scheduler_states
            .lock()
            .map_err(|error| format!("locking startup scheduler states failed: {error}"))?
            .keys()
            .filter(|sequence_id| real_full_internal_sequence(sequence_id))
            .cloned()
            .collect::<Vec<_>>();
        for stale_sequence in &stale_internal_sequences {
            self.finish_real_full_sequence(stale_sequence)?;
        }
        if !stale_internal_sequences.is_empty() {
            eprintln!(
                "real_full_startup_internal_scheduler_release released_sequences={}",
                stale_internal_sequences.len()
            );
        }
        if self.max_execution_lanes <= 1
            && startup_profile_mode == RealFullDsparkStartupProfileMode::Disabled
        {
            return Ok(());
        }
        if self.max_execution_lanes > 8 {
            return Err(format!(
                "batched dSpark startup prewarm supports at most 8 execution lanes, got {}",
                self.max_execution_lanes
            ));
        }
        if startup_profile_mode != RealFullDsparkStartupProfileMode::Disabled
            && self.max_execution_lanes > 4
        {
            return Err(format!(
                "dSpark startup SPS profiling supports at most 4 execution lanes, got {}",
                self.max_execution_lanes
            ));
        }
        {
            let mut dspark = dspark
                .lock()
                .map_err(|error| format!("locking dSpark startup prewarm failed: {error}"))?;
            let released = dspark.release_internal_sequences();
            if released > 0 {
                eprintln!(
                    "real_full_startup_internal_dspark_release released_sequences={released}"
                );
            }
        }

        let prompts = if startup_profile_mode != RealFullDsparkStartupProfileMode::Disabled {
            [
                "Implement a Rust function that merges overlapping integer intervals and explain its complexity.",
                "A product costs $240, receives a 25% discount, then 8% sales tax. Calculate the final price carefully.",
                "請用四個簡短條列解釋寫入時複製，並舉一個 fork 後修改記憶體頁面的例子。",
                "Write a short fable about two parrots sharing a mango tree, with a clear moral.",
            ]
            .into_iter()
            .take(self.max_execution_lanes)
            .map(str::to_owned)
            .collect::<Vec<_>>()
        } else {
            vec![REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(8); self.max_execution_lanes]
        };
        let prompt_tokens = {
            let tokenizer = self
                .tokenizer
                .lock()
                .map_err(|error| format!("locking startup prewarm tokenizer failed: {error}"))?;
            prompts
                .iter()
                .map(|prompt| {
                    tokenizer
                        .encode_text(prompt, false)
                        .map_err(format_error_chain)
                        .map(|encoded| encoded.token_count)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?
        };
        let decode_budget = if startup_profile_mode == RealFullDsparkStartupProfileMode::Disabled {
            1_024
        } else {
            4_096.max(startup_profile_samples.saturating_mul(512))
        };
        let sequence_ids = (1..=self.max_execution_lanes)
            .map(|buffer_bank| {
                format!(
                    "real-full-startup-dspark-width-{max_draft_tokens}-batched-bank-{buffer_bank}-sequence"
                )
            })
            .collect::<Vec<_>>();
        let start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start stage=batched-dspark-widths lanes={} prompt_tokens={} max_drafts={} max_physical_m={} startup_profile_mode={startup_profile_mode:?}",
            self.max_execution_lanes,
            prompt_tokens.iter().sum::<usize>(),
            max_draft_tokens,
            max_draft_tokens + 1,
        );

        let prewarm_result = (|| {
            let seed_start = Instant::now();
            let seed_requests = sequence_ids
                .iter()
                .zip(&prompts)
                .zip(&prompt_tokens)
                .enumerate()
                .map(|(lane_index, ((sequence_id, prompt), prompt_tokens))| {
                    glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                        90_000 + lane_index as u64,
                        sequence_id,
                        prompt,
                        *prompt_tokens,
                        1,
                        Vec::new(),
                        0,
                        decode_budget,
                    )
                })
                .collect::<Vec<_>>();
            let initial_cycles = self
                .execute_real_full_decode_cycle_batch(seed_requests)
                .into_iter()
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let reported_capture_delta = initial_cycles
                .iter()
                .map(|cycle| cycle.info.request_coordinator_graph_captures)
                .max()
                .unwrap_or(0);
            let mut generated_token_ids = Vec::with_capacity(sequence_ids.len());
            for (lane_index, initial_cycle) in initial_cycles.into_iter().enumerate() {
                if initial_cycle.info.status != "ready" {
                    return Err(format!(
                        "batched dSpark startup seed for lane {lane_index} failed: status={} blocker={} failed={:?}",
                        initial_cycle.info.status,
                        initial_cycle.info.blocker,
                        initial_cycle.info.failed_requirements,
                    ));
                }
                let token_id = initial_cycle
                    .generated_tokens
                    .first()
                    .map(|token| token.token_id)
                    .or(initial_cycle
                        .info
                        .scheduler_terminal_lm_head_sampled_token_id)
                    .ok_or_else(|| {
                        format!(
                            "batched dSpark startup seed for lane {lane_index} produced no token"
                        )
                    })?;
                generated_token_ids.push(vec![token_id]);
            }
            eprintln!(
                "real_full_startup_prewarm_step_done stage=batched-dspark-seed lanes={} reported_capture_delta={} elapsed_ms={:.3} total_ms={:.3}",
                self.max_execution_lanes,
                reported_capture_delta,
                elapsed_ms(seed_start),
                elapsed_ms(start),
            );

            // Each width uses the same live request/lane state, avoiding 4*C
            // redundant seed prefills. Long-context DFlash2 DSA identities
            // are captured separately through scalar lane-pinned execution.
            let width_passes = real_full_draft_width_prewarm_passes(false);
            for (width_index, draft_tokens) in (0..=max_draft_tokens).rev().enumerate() {
                for pass_index in 0..width_passes {
                    let requests = sequence_ids
                        .iter()
                        .zip(&prompts)
                        .zip(&prompt_tokens)
                        .zip(generated_token_ids.iter())
                        .enumerate()
                        .map(
                            |(
                                lane_index,
                                (((sequence_id, prompt), prompt_tokens), generated_tokens),
                            )| {
                                glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                                    REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_BASE
                                        + (draft_tokens as u64)
                                            * REAL_FULL_BATCHED_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE
                                        + (pass_index * 10 + lane_index) as u64,
                                    sequence_id,
                                    prompt,
                                    *prompt_tokens,
                                    1,
                                    generated_tokens.clone(),
                                    1 + width_index * width_passes + pass_index,
                                    decode_budget,
                                )
                            },
                        )
                        .collect::<Vec<_>>();
                    let width_start = Instant::now();
                    let cycles = self
                        .execute_real_full_decode_cycle_batch(requests)
                        .into_iter()
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let reported_capture_delta = cycles
                        .iter()
                        .map(|cycle| cycle.info.request_coordinator_graph_captures)
                        .max()
                        .unwrap_or(0);
                    for (lane_index, cycle) in cycles.iter().enumerate() {
                        if cycle.info.status != "ready"
                            || cycle.info.request_mtp_verify_rows != draft_tokens
                        {
                            return Err(format!(
                                "batched dSpark M={} startup pass {} lane {} failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
                                draft_tokens + 1,
                                pass_index + 1,
                                lane_index,
                                cycle.info.status,
                                cycle.info.request_mtp_verify_rows,
                                draft_tokens,
                                cycle.info.blocker,
                                cycle.info.failed_requirements,
                            ));
                        }
                        if cycle.generated_tokens.is_empty() {
                            return Err(format!(
                                "batched dSpark M={} startup pass {} lane {} emitted no tokens",
                                draft_tokens + 1,
                                pass_index + 1,
                                lane_index,
                            ));
                        }
                    }
                    for (generated_tokens, cycle) in
                        generated_token_ids.iter_mut().zip(cycles.into_iter())
                    {
                        generated_tokens.extend(
                            cycle
                                .generated_tokens
                                .into_iter()
                                .map(|token| token.token_id),
                        );
                    }
                    eprintln!(
                        "real_full_startup_prewarm_step_done stage=batched-dspark-widths pass={} lanes={} physical_m={} reported_capture_delta={} elapsed_ms={:.3} total_ms={:.3}",
                        pass_index + 1,
                        self.max_execution_lanes,
                        draft_tokens + 1,
                        reported_capture_delta,
                        elapsed_ms(width_start),
                        elapsed_ms(start),
                    );
                }
            }
            if startup_profile_mode != RealFullDsparkStartupProfileMode::Disabled {
                self.profile_batched_dspark_sps(
                    &sequence_ids,
                    &prompts,
                    &prompt_tokens,
                    decode_budget,
                    &mut generated_token_ids,
                    max_draft_tokens,
                    startup_profile_mode,
                    startup_profile_samples,
                )?;
            }
            Ok(())
        })();

        let cleanup_errors = sequence_ids
            .iter()
            .filter_map(|sequence_id| {
                self.finish_real_full_sequence(sequence_id)
                    .err()
                    .map(|error| format!("{sequence_id}: {error}"))
            })
            .collect::<Vec<_>>();
        if let Err(error) = prewarm_result {
            if cleanup_errors.is_empty() {
                return Err(error);
            }
            return Err(format!(
                "{error}; startup cleanup also failed: {}",
                cleanup_errors.join("; ")
            ));
        }
        if !cleanup_errors.is_empty() {
            return Err(format!(
                "batched dSpark startup cleanup failed: {}",
                cleanup_errors.join("; ")
            ));
        }
        eprintln!(
            "real_full_startup_prewarm_done stage=batched-dspark-widths lanes={} max_physical_m={} elapsed_ms={:.3}",
            self.max_execution_lanes,
            max_draft_tokens + 1,
            elapsed_ms(start),
        );
        if dflash2 {
            self.prewarm_dflash2_dsa_lane_graphs(max_draft_tokens)?;
        }
        Ok(())
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> std::result::Result<(), String> {
        if let Some(prefix_tokens) = real_full_startup_target_radix_evict_tokens(sequence_id) {
            let start = Instant::now();
            let prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(prefix_tokens);
            let encoding = self
                .tokenizer
                .lock()
                .map_err(|error| format!("locking startup radix tokenizer failed: {error}"))?
                .encode_text(&prompt, false)
                .map_err(format_error_chain)?;
            if encoding.token_count != prefix_tokens + 1 {
                return Err(format!(
                    "startup radix eviction prompt produced {} tokens for {prefix_tokens} reusable tokens",
                    encoding.token_count
                ));
            }
            let prefix_token_ids = encoding.token_ids[..prefix_tokens]
                .iter()
                .map(|token_id| *token_id as usize)
                .collect::<Vec<_>>();
            let eviction = self
                .target_kv_radix
                .evict_exact_inactive_subtree_if_present(&prefix_token_ids)
                .map_err(format_error_chain)?;
            let (status, evicted_pages, matched_tokens) = match eviction {
                TargetKvExactSubtreeEviction::Evicted { pages } => {
                    ("evicted", pages, prefix_tokens)
                }
                TargetKvExactSubtreeEviction::AlreadyAbsent { matched_tokens } => {
                    ("already-absent", 0, matched_tokens)
                }
            };
            let stats = self.target_kv_radix.stats();
            eprintln!(
                "real_full_startup_target_kv_radix_evict prefix_tokens={} status={} matched_tokens={} evicted_pages={} cached_pages={} free_pages={} radix_nodes={} elapsed_ms={:.3}",
                prefix_tokens,
                status,
                matched_tokens,
                evicted_pages,
                stats.cached_pages,
                stats.free_pages,
                stats.radix_nodes,
                elapsed_ms(start),
            );
            return Ok(());
        }
        if let Some((query_rows, sparse_topk)) = real_full_nvfp4_short_k_graph_audit(sequence_id) {
            return audit_glm_dsa_nvfp4_short_k_prefill_graph_retention(query_rows, sparse_topk)
                .map_err(format_error_chain);
        }
        if real_full_prewarm_batched_dspark_sequence(sequence_id) {
            return self.prewarm_batched_dspark_graphs();
        }
        if real_full_prewarm_paired_lm_head_sequence(sequence_id) {
            if let Some(dspark) = self.dspark.as_ref() {
                let released = dspark
                    .lock()
                    .map_err(|err| format!("locking dSpark request executor failed: {err}"))?
                    .release_internal_sequences();
                if released > 0 {
                    eprintln!(
                        "real_full_startup_internal_dspark_release released_sequences={released}"
                    );
                }
            }
            let max_sampler_rows = 2 * (real_full_active_max_verify_drafts() + 1);
            for buffer_bank in 1..=self.max_execution_lanes {
                let start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start stage=lm-head-sampler-capacity buffer_bank={} max_rows={} top_k=64",
                    buffer_bank, max_sampler_rows,
                );
                with_coordinator_owned_device_buffer_bank(buffer_bank, || {
                    prewarm_real_full_target_sampler_capacity(&self.catalog, max_sampler_rows)
                })
                .map_err(format_error_chain)?;
                eprintln!(
                    "real_full_startup_prewarm_step_done stage=lm-head-sampler-capacity buffer_bank={} max_rows={} top_k=64 elapsed_ms={:.3}",
                    buffer_bank,
                    max_sampler_rows,
                    elapsed_ms(start),
                );
            }
            let Some((min_paired_rows, max_paired_rows)) = real_full_paired_lm_head_prewarm_range(
                real_full_active_fixed_drafts().map_err(format_error_chain)?,
            ) else {
                return Ok(());
            };
            for buffer_bank in (1..self.max_execution_lanes).step_by(2) {
                let start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start stage=paired-lm-head buffer_bank={} min_rows={} max_rows={}",
                    buffer_bank, min_paired_rows, max_paired_rows,
                );
                with_coordinator_owned_device_buffer_bank(
                    real_full_paired_lm_head_buffer_bank(buffer_bank),
                    || {
                        prewarm_real_full_paired_target_token_sample_rows(
                            &self.catalog,
                            min_paired_rows,
                            max_paired_rows,
                        )
                    },
                )
                .map_err(format_error_chain)?;
                eprintln!(
                    "real_full_startup_prewarm_step_done stage=paired-lm-head buffer_bank={} min_rows={} max_rows={} elapsed_ms={:.3}",
                    buffer_bank,
                    min_paired_rows,
                    max_paired_rows,
                    elapsed_ms(start),
                );
            }
            return Ok(());
        }
        if real_full_seal_owned_buffer_pool_sequence(sequence_id) {
            return seal_coordinator_owned_device_buffer_pool().map_err(format_error_chain);
        }
        let state = {
            let mut states = self
                .scheduler_states
                .lock()
                .map_err(|err| format!("locking real-full scheduler state map failed: {err}"))?;
            if let Some(state) = states.get(sequence_id) {
                if !state.owned_by_current_thread() {
                    return Err(format!(
                        "real-full sequence {sequence_id} must be finished on its graph-owner thread"
                    ));
                }
            }
            states.remove(sequence_id)
        };
        let finished_token_ids = state
            .as_ref()
            .filter(|_| !real_full_internal_sequence(sequence_id))
            .map(|state| state.processed_token_ids().to_vec());
        let dspark_finish_result = if let Some(dspark) = self.dspark.as_ref() {
            dspark
                .lock()
                .map_err(|err| format!("locking dSpark request executor failed: {err}"))?
                .finish_sequence(sequence_id, finished_token_ids.as_deref())
                .map_err(format_error_chain)
        } else {
            Ok(None)
        };
        let dspark_retained_prefix_tokens = dspark_finish_result.as_ref().ok().copied().flatten();
        let mut state_finish_result = Ok(());
        if let Some(mut state) = state {
            let committed_token_ids = state.processed_token_ids().to_vec();
            let snapshot_result = if state.snapshot_save_ready {
                if !self.kv_snapshot_saves.is_empty() {
                    if self
                        .kv_snapshot_saved
                        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                        .is_ok()
                    {
                        let committed_tokens = state.processed_token_ids().len();
                        let result = (|| {
                            for save in &self.kv_snapshot_saves {
                                let token_count = save.token_count.unwrap_or(committed_tokens);
                                anyhow::ensure!(
                                    token_count <= committed_tokens,
                                    "KV snapshot save point {token_count} exceeds committed sequence frontier {committed_tokens}"
                                );
                            }
                            for save in &self.kv_snapshot_saves {
                                let token_count = save.token_count.unwrap_or(committed_tokens);
                                let save_start = Instant::now();
                                save_real_full_kv_snapshot(
                                    &mut state.state,
                                    &save.root,
                                    &self.catalog,
                                    &self.engine_commit,
                                    token_count,
                                )?;
                                eprintln!(
                                    "real_full_kv_snapshot_save path={} tokens={} elapsed_ms={:.3}",
                                    save.root.display(),
                                    token_count,
                                    elapsed_ms(save_start),
                                );
                            }
                            Ok(())
                        })();
                        result.map_err(format_error_chain)
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            } else {
                Ok(())
            };
            let radix_publish_result = if let Some(reservation) =
                state.target_radix_reservation.take()
            {
                // A reusable draft tail can end a handful of tokens behind
                // the final target pass because no next speculation cycle is
                // needed. Publish only the common exact frontier so a later
                // radix hit can restore the tail and replay that tiny suffix.
                let committed_tokens = real_full_startup_target_radix_publish_tokens(sequence_id)
                    .or(dspark_retained_prefix_tokens)
                    .unwrap_or(committed_token_ids.len());
                reservation
                        .commit_prefix(&committed_token_ids, committed_tokens)
                        .map(|published| {
                            let stats = self.target_kv_radix.stats();
                            eprintln!(
                                "real_full_target_kv_radix_publish sequence_id={} committed_tokens={} matched_existing_tokens={} published_pages={} duplicate_pages_freed={} cached_pages={} free_pages={} radix_nodes={} evicted_nodes={} evicted_pages={}",
                                sequence_id,
                                committed_tokens,
                                published.matched_existing_tokens,
                                published.published_pages,
                                published.duplicate_pages_freed,
                                stats.cached_pages,
                                stats.free_pages,
                                stats.radix_nodes,
                                stats.evicted_nodes,
                                stats.evicted_pages,
                            );
                        })
                        .map_err(format_error_chain)
            } else {
                Ok(())
            };
            let recycle_result = self.recycle_scheduler_state(state);
            state_finish_result = snapshot_result
                .and(radix_publish_result)
                .and(recycle_result);
        }
        state_finish_result?;
        dspark_finish_result?;
        if !real_full_internal_sequence(sequence_id) {
            clear_transient_coordinator_owned_device_buffers().map_err(format_error_chain)?;
        }
        Ok(())
    }
}

impl RealFullSchedulerRequestExecutor {
    fn decode_sampled_token_text_cached(&self, token_id: Option<usize>) -> Option<String> {
        let token_id = token_id?;
        if let Ok(cache) = self.sampled_token_text_cache.lock() {
            if let Some(text) = cache.get(&token_id) {
                return Some(text.clone());
            }
        }

        let tokenizer = self.tokenizer.lock().ok()?;
        let text = decode_sampled_token_text_with_tokenizer(&tokenizer, Some(token_id))?;
        if let Ok(mut cache) = self.sampled_token_text_cache.lock() {
            cache.insert(token_id, text.clone());
        }
        Some(text)
    }

    #[allow(clippy::too_many_arguments)]
    fn fast_embedding_lm_head_token_info(
        &self,
        request_id: &str,
        prompt_tokens_hint: usize,
        generated_tokens: usize,
        max_tokens: usize,
        prefill_tokens: usize,
        prefill_chunk_tokens: usize,
        decode_rows: usize,
        decode_token_ids: &[usize],
    ) -> Result<glmrt_api::RealFullInfo> {
        let source_token_id = decode_token_ids.last().copied().unwrap_or(0);
        let embedding = real_full_embedding_hidden_for_token(&self.catalog, source_token_id)
            .with_context(|| {
                format!("building serve-fast embedding hidden for token {source_token_id}")
            })?;
        let lm_head_rows = real_full_serve_fast_token_lm_head_rows();
        let sample =
            score_real_lm_head_chunk_for_hidden(&self.catalog, &embedding.hidden, lm_head_rows)
                .with_context(|| {
                    format!(
                "sampling serve-fast token from embedding hidden over {lm_head_rows} lm_head rows"
            )
                })?;
        let mut info = self.base_info.clone();
        info.status = "ready".to_owned();
        info.startup_diagnostic_mode = "serve-fast-token-embedding-lm-head".to_owned();
        info.blocker.clear();
        info.failed_requirements.clear();
        info.request_prefill_tokens = prefill_tokens;
        info.request_prefill_chunks = prefill_tokens.div_ceil(prefill_chunk_tokens);
        info.request_decode_budget = decode_rows;
        info.request_mtp_verify_rows = 0;
        info.request_mtp_accepted_rows = 0;
        info.scheduler_iterations = 0;
        info.selected_layerwaves = 0;
        info.request_deferred_layerwaves = 0;
        info.sparse_expert_batches = 0;
        info.request_expert_batch_rows = 0;
        info.request_expert_batch_routes = 0;
        info.request_expert_prefill_rows = 0;
        info.request_expert_decode_rows = 0;
        info.request_expert_mtp_verify_rows = 0;
        info.request_expert_prefill_routes = 0;
        info.request_expert_decode_routes = 0;
        info.request_expert_mtp_verify_routes = 0;
        info.scheduler_full_context_device_attention_complete = false;
        info.scheduler_terminal_lm_head_sample_status = "sampled".to_owned();
        info.scheduler_terminal_lm_head_sample_passed = true;
        info.scheduler_terminal_lm_head_uses_final_decode_device_hidden = false;
        info.scheduler_terminal_lm_head_covers_full_vocabulary = sample.covers_full_vocabulary;
        info.scheduler_terminal_lm_head_logits_evaluated = sample.logits_evaluated;
        info.scheduler_terminal_lm_head_vocab_size = sample.vocab_size;
        info.scheduler_terminal_lm_head_top_token_id = Some(sample.top_token_id);
        info.scheduler_terminal_lm_head_sampled_token_id = Some(sample.sampled_token_id);
        info.scheduler_terminal_lm_head_sampled_text =
            self.decode_sampled_token_text_cached(Some(sample.sampled_token_id));
        info.scheduler_terminal_lm_head_sample_top_k = Some(sample.sample_top_k);
        info.scheduler_terminal_lm_head_sample_top_p = Some(sample.sample_top_p);
        info.scheduler_terminal_lm_head_argmax_backend =
            Some(sample.argmax_kernel_backend.to_owned());
        info.scheduler_terminal_lm_head_sampler_backend =
            Some(sample.sampler_kernel_backend.to_owned());
        info.scheduler_terminal_lm_head_blocker = None;
        eprintln!(
            "real_full_fast_token_evidence request_id={} embedding_token={} embedding_backend={} prompt_tokens_hint={} generated_tokens={} max_tokens={} embedding_checksum={:.6} lm_head_rows={} lm_head_argmax_backend={} lm_head_sampler_backend={}",
            request_id,
            source_token_id,
            embedding.kernel_backend,
            prompt_tokens_hint,
            generated_tokens,
            max_tokens,
            embedding.checksum,
            lm_head_rows,
            sample.argmax_kernel_backend,
            sample.sampler_kernel_backend
        );
        Ok(info)
    }
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn real_full_request_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_REQUEST_TIMING_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn real_full_mtp_probe_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_MTP_PROBE_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn real_full_mtp_physical_padding_rows(logical_draft_rows: usize, physical_m2: bool) -> usize {
    usize::from(logical_draft_rows == 1 && !physical_m2)
}

fn real_full_mtp_physical_m2_enabled() -> bool {
    let configured = |name| {
        env::var(name).ok().map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
    };
    configured(REAL_FULL_MTP_ALLOW_PHYSICAL_M2_ENV)
        .or_else(|| configured(REAL_FULL_MTP_DIAGNOSTIC_PHYSICAL_M2_ENV))
        .unwrap_or(true)
}

fn real_full_mtp_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_MTP_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn real_full_native_mtp_sequence_enabled(sequence_id: &str) -> bool {
    !sequence_id.starts_with("real-full-startup-")
        || sequence_id.starts_with("real-full-startup-mtp-production-")
}

fn real_full_mtp_draft_policy_from_values(
    legacy_fixed: Option<usize>,
    configured_min: Option<usize>,
    configured_max: Option<usize>,
    adaptive_requested: bool,
) -> RealFullMtpDraftPolicy {
    let valid = |value: Option<usize>| {
        value.filter(|tokens| (1..=MAX_REAL_FULL_MTP_DRAFT_TOKENS).contains(tokens))
    };
    if !adaptive_requested {
        if let Some(fixed) = valid(legacy_fixed) {
            return RealFullMtpDraftPolicy {
                min: fixed,
                max: fixed,
                start: fixed,
                adaptive: false,
            };
        }
    }

    let min = valid(configured_min).unwrap_or(DEFAULT_REAL_FULL_MTP_MIN_DRAFT_TOKENS);
    let max = valid(configured_max)
        .unwrap_or(DEFAULT_REAL_FULL_MTP_MAX_DRAFT_TOKENS)
        .max(min);
    let span = max - min;
    let start = min + span.saturating_mul(3).saturating_add(3) / 4;
    RealFullMtpDraftPolicy {
        min,
        max,
        start,
        adaptive: true,
    }
}

fn real_full_mtp_draft_policy() -> &'static RealFullMtpDraftPolicy {
    static POLICY: OnceLock<RealFullMtpDraftPolicy> = OnceLock::new();
    POLICY.get_or_init(|| {
        let min_value = env::var(REAL_FULL_MTP_MIN_DRAFT_TOKENS_ENV).ok();
        let max_value = env::var(REAL_FULL_MTP_MAX_DRAFT_TOKENS_ENV).ok();
        real_full_mtp_draft_policy_from_values(
            env::var(REAL_FULL_MTP_DRAFT_TOKENS_ENV)
                .ok()
                .and_then(|value| value.parse::<usize>().ok()),
            min_value
                .as_deref()
                .and_then(|value| value.parse::<usize>().ok()),
            max_value
                .as_deref()
                .and_then(|value| value.parse::<usize>().ok()),
            min_value.is_some() || max_value.is_some(),
        )
    })
}

fn real_full_mtp_draft_tokens() -> usize {
    real_full_mtp_draft_policy().max
}

fn real_full_mtp_startup_forced_draft_tokens(sequence_id: &str) -> Option<usize> {
    sequence_id
        .strip_prefix("real-full-startup-mtp-production-draft-")
        .and_then(|suffix| suffix.split('-').next())
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| (1..=MAX_REAL_FULL_MTP_DRAFT_TOKENS).contains(tokens))
}

fn real_full_mtp_requested_draft_tokens(
    sequence_id: &str,
    state: &mut RealFullSchedulerExecutionState,
) -> usize {
    if let Some(forced) = real_full_mtp_startup_forced_draft_tokens(sequence_id) {
        return forced;
    }
    let policy = real_full_mtp_draft_policy();
    state.mtp_draft_width(policy.min, policy.max, policy.start, policy.adaptive)
}

fn real_full_mtp_draft_tokens_for_cycle_with_limit(
    decode_budget: usize,
    generated_tokens_before: usize,
    emitted_tokens: usize,
    prompt_tokens: usize,
    max_context_tokens: usize,
    draft_tokens: usize,
) -> usize {
    real_full_mtp_draft_tokens_for_cycle_with_policy(
        decode_budget,
        generated_tokens_before,
        emitted_tokens,
        prompt_tokens,
        max_context_tokens,
        draft_tokens,
        real_full_mtp_full_match_bonus_enabled(),
    )
}

fn real_full_mtp_draft_tokens_for_cycle_with_policy(
    decode_budget: usize,
    generated_tokens_before: usize,
    emitted_tokens: usize,
    prompt_tokens: usize,
    max_context_tokens: usize,
    draft_tokens: usize,
    fixed_width_enabled: bool,
) -> usize {
    let budget_limited = real_full_mtp_draft_tokens_after_cycle_with_limit(
        decode_budget,
        generated_tokens_before,
        emitted_tokens,
        draft_tokens,
    );
    if budget_limited == 0 || !fixed_width_enabled {
        return budget_limited;
    }

    // Full-consumption bridges are qualified at fixed physical target shapes.
    // Keep speculative work at the selected shape even on the final, short
    // output cycle; acceptance below limits committed and emitted rows to the
    // API budget.
    // Near the physical context ceiling, fall back to ordinary M=1 decode
    // instead of placing temporary target rows beyond the KV/DSA arena.
    let speculative_end = prompt_tokens
        .checked_add(generated_tokens_before)
        .and_then(|tokens| tokens.checked_add(emitted_tokens))
        .and_then(|tokens| tokens.checked_add(draft_tokens));
    if speculative_end.is_some_and(|tokens| tokens <= max_context_tokens) {
        draft_tokens
    } else {
        0
    }
}

fn real_full_mtp_draft_tokens_after_cycle_with_limit(
    decode_budget: usize,
    generated_tokens_before: usize,
    emitted_tokens: usize,
    draft_limit: usize,
) -> usize {
    decode_budget
        .saturating_sub(generated_tokens_before.saturating_add(emitted_tokens))
        .saturating_sub(1)
        .min(draft_limit)
}

fn real_full_mtp_acceptance(
    draft_token_ids: &[usize],
    target_sampled_token_ids: &[usize],
    full_match_bonus_enabled: bool,
    max_emitted_tokens: usize,
) -> Result<RealFullMtpAcceptance> {
    anyhow::ensure!(
        target_sampled_token_ids.len() == draft_token_ids.len() + 1,
        "real-full MTP target sample count {} must equal draft count {} plus one fallback",
        target_sampled_token_ids.len(),
        draft_token_ids.len()
    );
    let matching_prefix = draft_token_ids
        .iter()
        .zip(target_sampled_token_ids)
        .take_while(|(draft, target)| draft == target)
        .count();
    anyhow::ensure!(
        max_emitted_tokens > 0,
        "real-full MTP acceptance requires a positive emission budget"
    );
    let full_match_bonus = full_match_bonus_enabled
        && matching_prefix == draft_token_ids.len()
        && draft_token_ids.len().saturating_add(1) <= max_emitted_tokens;
    if full_match_bonus {
        return Ok(RealFullMtpAcceptance {
            accepted_draft_tokens: matching_prefix,
            terminal_target_index: matching_prefix,
            full_match_bonus: true,
        });
    }
    // The opt-out path retains the final matching draft as the fallback so the
    // next MTP chain begins after a contiguous layer-78 cache prefix. Full
    // consumption is the default; this path remains only as a diagnostic.
    let accepted_draft_tokens = matching_prefix
        .min(draft_token_ids.len().saturating_sub(1))
        .min(max_emitted_tokens.saturating_sub(1));
    Ok(RealFullMtpAcceptance {
        accepted_draft_tokens,
        terminal_target_index: accepted_draft_tokens,
        full_match_bonus: false,
    })
}

fn real_full_mtp_full_match_bonus_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_MTP_FULL_MATCH_BONUS_ENV)
            .map(|value| {
                !matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "0" | "false" | "no" | "off"
                )
            })
            .unwrap_or(true)
    })
}

fn real_full_request_thread_pinned_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_REQUEST_THREAD_PINNED_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or_else(|_| crate::python_graph_capture::coordinator_python_capture_enabled())
    })
}

fn real_full_request_thread_pinned_workers() -> Result<usize> {
    let worker_count = match env::var(REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV) {
        Ok(value) => value.parse::<usize>().with_context(|| {
            format!("{REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV} must be a positive integer")
        })?,
        Err(_) => 1,
    };
    anyhow::ensure!(
        worker_count > 0,
        "{REAL_FULL_REQUEST_THREAD_PINNED_WORKERS_ENV} must be a positive integer"
    );
    Ok(worker_count)
}

fn optional_cpu_list_env(name: &str) -> Result<Vec<usize>> {
    let value = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {name}")),
    };
    anyhow::ensure!(!value.trim().is_empty(), "{name} must not be empty");
    value
        .split(',')
        .enumerate()
        .map(|(index, value)| {
            let value = value.trim();
            anyhow::ensure!(!value.is_empty(), "{name} entry {index} must not be empty");
            value
                .parse::<usize>()
                .with_context(|| format!("{name} entry {index} must be a non-negative CPU index"))
        })
        .collect()
}

fn real_full_request_worker_cpus(worker_count: usize) -> Result<Vec<usize>> {
    let cpus = optional_cpu_list_env(REAL_FULL_REQUEST_WORKER_CPUS_ENV)?;
    anyhow::ensure!(
        cpus.is_empty() || cpus.len() == worker_count,
        "{REAL_FULL_REQUEST_WORKER_CPUS_ENV} has {} CPU assignments for {worker_count} request workers",
        cpus.len()
    );
    Ok(cpus)
}

fn real_full_scheduler_worker_cpu() -> Result<Option<usize>> {
    let cpus = optional_cpu_list_env(REAL_FULL_SCHEDULER_WORKER_CPU_ENV)?;
    anyhow::ensure!(
        cpus.len() <= 1,
        "{REAL_FULL_SCHEDULER_WORKER_CPU_ENV} must contain exactly one CPU index"
    );
    Ok(cpus.into_iter().next())
}

fn real_full_serve_fast_token_enabled() -> bool {
    env::var("GLMRT_REAL_FULL_SERVE_FAST_TOKEN")
        .ok()
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
}

fn real_full_serve_fast_token_lm_head_rows() -> usize {
    env::var("GLMRT_REAL_FULL_SERVE_FAST_TOKEN_LM_HEAD_ROWS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|rows| *rows > 0)
        .unwrap_or(1024)
}

fn format_error_chain(error: anyhow::Error) -> String {
    format!("{error:#}")
}

fn optional_nonempty_env_path(name: &str) -> Result<Option<PathBuf>> {
    match env::var(name) {
        Ok(value) => {
            let value = value.trim();
            anyhow::ensure!(!value.is_empty(), "{name} must not be empty");
            Ok(Some(PathBuf::from(value)))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {name}")),
    }
}

fn optional_positive_env_usize(name: &str) -> Result<Option<usize>> {
    match env::var(name) {
        Ok(value) => {
            let parsed = value
                .parse::<usize>()
                .with_context(|| format!("{name} must be a positive integer"))?;
            anyhow::ensure!(parsed > 0, "{name} must be a positive integer");
            Ok(Some(parsed))
        }
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error).with_context(|| format!("reading {name}")),
    }
}

fn real_full_kv_pool_tokens(max_context_tokens: usize) -> Result<usize> {
    let pool_tokens =
        optional_positive_env_usize(REAL_FULL_KV_POOL_TOKENS_ENV)?.unwrap_or(max_context_tokens);
    anyhow::ensure!(
        pool_tokens >= max_context_tokens,
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={pool_tokens} is smaller than max context {max_context_tokens}"
    );
    anyhow::ensure!(
        pool_tokens % REAL_FULL_SHARED_KV_PAGE_TOKENS == 0,
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={pool_tokens} must be divisible by {REAL_FULL_SHARED_KV_PAGE_TOKENS}"
    );
    anyhow::ensure!(
        u32::try_from(pool_tokens).is_ok(),
        "{REAL_FULL_KV_POOL_TOKENS_ENV}={pool_tokens} exceeds the physical position format"
    );
    Ok(pool_tokens)
}

fn real_full_max_execution_lanes() -> Result<usize> {
    let lanes = optional_positive_env_usize(REAL_FULL_MAX_EXECUTION_LANES_ENV)?.unwrap_or(1);
    anyhow::ensure!(
        lanes <= REAL_FULL_DIAGNOSTIC_MAX_EXECUTION_LANES,
        "{REAL_FULL_MAX_EXECUTION_LANES_ENV}={lanes} exceeds the current diagnostic maximum {REAL_FULL_DIAGNOSTIC_MAX_EXECUTION_LANES}"
    );
    Ok(lanes)
}

fn real_full_kv_snapshot_save_points(name: &str) -> Result<Vec<RealFullKvSnapshotSave>> {
    let value = match env::var(name) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => return Ok(Vec::new()),
        Err(error) => return Err(error).with_context(|| format!("reading {name}")),
    };
    anyhow::ensure!(!value.trim().is_empty(), "{name} must not be empty");
    value
        .split(',')
        .enumerate()
        .map(|(index, entry)| {
            let entry = entry.trim();
            let (tokens, path) = entry
                .split_once('=')
                .with_context(|| format!("{name} entry {index} must use the form TOKENS=PATH"))?;
            let token_count = tokens.trim().parse::<usize>().with_context(|| {
                format!("{name} entry {index} token count must be a positive integer")
            })?;
            anyhow::ensure!(
                token_count > 0,
                "{name} entry {index} token count must be positive"
            );
            let path = path.trim();
            anyhow::ensure!(
                !path.is_empty(),
                "{name} entry {index} path must not be empty"
            );
            Ok(RealFullKvSnapshotSave {
                root: PathBuf::from(path),
                token_count: Some(token_count),
            })
        })
        .collect()
}

fn validate_real_full_kv_snapshot_saves(saves: &[RealFullKvSnapshotSave]) -> Result<()> {
    for (index, save) in saves.iter().enumerate() {
        for prior in &saves[..index] {
            anyhow::ensure!(
                save.root != prior.root,
                "duplicate KV snapshot save destination {}",
                save.root.display()
            );
            if let (Some(tokens), Some(prior_tokens)) = (save.token_count, prior.token_count) {
                anyhow::ensure!(
                    tokens != prior_tokens,
                    "duplicate KV snapshot save token cutoff {tokens}"
                );
            }
        }
    }
    Ok(())
}

fn real_full_request_prefill_chunk_tokens() -> usize {
    env::var("GLMRT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS)
}

fn real_full_request_large_prefill_min_tokens() -> usize {
    env::var(REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_REQUEST_LARGE_PREFILL_MIN_TOKENS)
}

fn real_full_request_long_prefix_small_prefill_chunk_tokens() -> usize {
    env::var(REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS)
}

fn balanced_prefill_chunk_tokens(prefill_tokens: usize, max_chunk_tokens: usize) -> usize {
    if prefill_tokens == 0 {
        return max_chunk_tokens;
    }
    let chunk_count = prefill_tokens.div_ceil(max_chunk_tokens).max(1);
    // A measured 530-row request benefits from replacing 512+18 with two
    // balanced waves. At three or more useful waves, preserving full-width
    // expert launches wins. The exception is a third prefill tail below 15
    // rows: even after the joined decode row it cannot use the incremental
    // Spark-reduction path and serializes another complete 75-layer wave.
    if chunk_count == 2 {
        prefill_tokens.div_ceil(chunk_count)
    } else if tiny_non_streaming_third_prefill_chunk(prefill_tokens, max_chunk_tokens) {
        prefill_tokens.div_ceil(2)
    } else {
        max_chunk_tokens
    }
}

fn tiny_non_streaming_third_prefill_chunk(prefill_tokens: usize, max_chunk_tokens: usize) -> bool {
    // This exception is qualified only for the production 512-row stride.
    // Applying it to a wider operator-selected cap would both exceed that cap
    // and substitute an unmeasured kernel geometry.
    if max_chunk_tokens != DEFAULT_REAL_FULL_REQUEST_FRESH_SMALL_PREFILL_CHUNK_TOKENS {
        return false;
    }
    let chunk_count = prefill_tokens.div_ceil(max_chunk_tokens);
    let tail_tokens = prefill_tokens % max_chunk_tokens;
    chunk_count == 3
        && tail_tokens > 0
        && tail_tokens < DEFAULT_REAL_FULL_REQUEST_MIN_STREAMING_TAIL_PREFILL_TOKENS
}

fn real_full_request_prefill_chunk_tokens_for_shape_with(
    configured_chunk_tokens: usize,
    large_prefill_min_tokens: usize,
    long_prefix_small_prefill_chunk_tokens: usize,
    prefix_tokens: usize,
    prefill_tokens: usize,
) -> usize {
    if prefill_tokens >= large_prefill_min_tokens {
        let minimum_chunks = prefill_tokens.div_ceil(configured_chunk_tokens).max(1);
        let balanced_chunks =
            minimum_chunks.max(prefill_tokens.min(REAL_FULL_PREFILL_PIPELINE_LANES).max(1));
        prefill_tokens.div_ceil(balanced_chunks)
    } else if prefix_tokens >= DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_MIN_TOKENS {
        let base_chunk_tokens = configured_chunk_tokens
            .min(long_prefix_small_prefill_chunk_tokens)
            .min(GLM_DSA_PREFILL_MAX_QUERY_ROWS);
        // A tiny second chunk costs another complete 75-layer sparse wave at
        // long context. The measured one-wave advantage is robust through
        // 7/4 of the 512-row production stride and disappears near 1K.
        let tail_merge_ceiling = base_chunk_tokens
            .saturating_mul(DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_NUMERATOR)
            / DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_DENOMINATOR;
        let tail_merge_ceiling = tail_merge_ceiling.min(
            DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_SMALL_PREFILL_CHUNK_TOKENS
                * DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_NUMERATOR
                / DEFAULT_REAL_FULL_REQUEST_LONG_PREFIX_TAIL_MERGE_DENOMINATOR,
        );
        if prefill_tokens > base_chunk_tokens && prefill_tokens <= tail_merge_ceiling {
            prefill_tokens
        } else if tiny_non_streaming_third_prefill_chunk(prefill_tokens, base_chunk_tokens) {
            prefill_tokens.div_ceil(2)
        } else {
            base_chunk_tokens
        }
    } else if prefix_tokens == 0
        || prefill_tokens >= DEFAULT_REAL_FULL_REQUEST_CACHED_WIDE_SUFFIX_MIN_TOKENS
    {
        balanced_prefill_chunk_tokens(
            prefill_tokens,
            configured_chunk_tokens.min(DEFAULT_REAL_FULL_REQUEST_FRESH_SMALL_PREFILL_CHUNK_TOKENS),
        )
    } else {
        balanced_prefill_chunk_tokens(
            prefill_tokens,
            configured_chunk_tokens.min(DEFAULT_REAL_FULL_REQUEST_SMALL_PREFILL_CHUNK_TOKENS),
        )
    }
}

fn real_full_request_prefill_chunk_tokens_for_shape(
    prefix_tokens: usize,
    prefill_tokens: usize,
) -> usize {
    real_full_request_prefill_chunk_tokens_for_shape_with(
        real_full_request_prefill_chunk_tokens(),
        real_full_request_large_prefill_min_tokens(),
        real_full_request_long_prefix_small_prefill_chunk_tokens(),
        prefix_tokens,
        prefill_tokens,
    )
}

fn real_full_request_prefill_chunk_tokens_for_sequence(
    sequence_id: &str,
    prefix_tokens: usize,
    prefill_tokens: usize,
) -> usize {
    if sequence_id.starts_with(REAL_FULL_STARTUP_MAX_PREFILL_CHUNK_PREFIX) {
        return prefill_tokens
            .min(real_full_request_prefill_chunk_tokens())
            .max(1);
    }
    if sequence_id.starts_with(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX) {
        return prefill_tokens
            .min(real_full_request_prefill_chunk_tokens())
            .min(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_TOKENS)
            .max(1);
    }
    real_full_request_prefill_chunk_tokens_for_shape(prefix_tokens, prefill_tokens)
}

fn real_full_prefill_chunk_tokens_for_direct_dsa(planned_chunk_tokens: usize) -> usize {
    planned_chunk_tokens.min(GLM_DSA_PREFILL_MAX_QUERY_ROWS)
}

fn real_full_validate_sparse_wave_capacity(
    prefill_rows: usize,
    decode_rows: usize,
    mtp_rows: usize,
) -> Result<()> {
    let physical_rows = prefill_rows
        .checked_add(decode_rows)
        .and_then(|rows| rows.checked_add(mtp_rows))
        .context("real-full sparse wave row count overflow")?;
    anyhow::ensure!(
        physical_rows <= B12X_EXL3_TOPK8_CAPACITY_ROWS,
        "real-full sparse wave rows prefill={prefill_rows} decode={decode_rows} mtp={mtp_rows} total={physical_rows} exceed the EXL3 AOT capacity {B12X_EXL3_TOPK8_CAPACITY_ROWS}"
    );
    Ok(())
}

fn real_full_mtp_prefill_chunk_tokens() -> usize {
    env::var(REAL_FULL_MTP_PREFILL_CHUNK_TOKENS_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|tokens| *tokens > 0)
        .unwrap_or(DEFAULT_REAL_FULL_MTP_PREFILL_CHUNK_TOKENS)
}

fn real_full_request_token_rows(
    request: &glmrt_api::RealFullRequest,
    prompt_token_ids: Option<Vec<usize>>,
) -> Result<RealFullRequestTokenRows> {
    if request.cached_prompt_tokens > 0 {
        anyhow::ensure!(
            request.generated_token_ids.is_empty(),
            "real-full cached-prefix request cannot also contain generated tokens"
        );
        let prompt_token_ids = prompt_token_ids
            .context("real-full cached-prefix request is missing prompt token ids")?;
        anyhow::ensure!(
            prompt_token_ids.len() == request.prompt_tokens,
            "real-full cached-prefix prompt token id count {} does not match prompt tokens {}",
            prompt_token_ids.len(),
            request.prompt_tokens
        );
        anyhow::ensure!(
            request.cached_prompt_tokens < prompt_token_ids.len(),
            "real-full cached-prefix token count {} leaves no uncached prompt tokens out of {}",
            request.cached_prompt_tokens,
            prompt_token_ids.len()
        );
        let uncached_token_ids = &prompt_token_ids[request.cached_prompt_tokens..];
        let decode_token_id = *uncached_token_ids
            .last()
            .context("real-full cached-prefix request has no uncached prompt tokens")?;
        let prefill_tokens = uncached_token_ids.len() - 1;
        return Ok(RealFullRequestTokenRows {
            prefix_tokens: request.cached_prompt_tokens,
            prefill_tokens,
            prefill_token_ids: (prefill_tokens > 0)
                .then(|| uncached_token_ids[..prefill_tokens].to_vec()),
            decode_token_ids: vec![decode_token_id],
        });
    }
    if let Some(decode_token_id) = request.generated_token_ids.last().copied() {
        let prefix_tokens = request
            .prompt_tokens
            .checked_add(request.generated_token_ids.len().saturating_sub(1))
            .context("real-full request recurrent decode prefix token count overflows usize")?;
        return Ok(RealFullRequestTokenRows {
            prefix_tokens,
            prefill_tokens: 0,
            prefill_token_ids: None,
            decode_token_ids: vec![decode_token_id],
        });
    }

    let prompt_token_ids =
        prompt_token_ids.context("real-full initial decode request is missing prompt token ids")?;
    anyhow::ensure!(
        !prompt_token_ids.is_empty(),
        "real-full initial decode request has no prompt tokens"
    );
    anyhow::ensure!(
        prompt_token_ids.len() == request.prompt_tokens,
        "real-full initial decode prompt token id count {} does not match prompt tokens {}",
        prompt_token_ids.len(),
        request.prompt_tokens
    );
    let decode_token_id = *prompt_token_ids
        .last()
        .context("real-full initial decode prompt token ids unexpectedly empty")?;
    let prefill_tokens = prompt_token_ids.len() - 1;
    let prefill_token_ids = if prefill_tokens == 0 {
        None
    } else {
        Some(prompt_token_ids[..prefill_tokens].to_vec())
    };

    Ok(RealFullRequestTokenRows {
        prefix_tokens: 0,
        prefill_tokens,
        prefill_token_ids,
        decode_token_ids: vec![decode_token_id],
    })
}

fn real_full_request_mtp_rows(
    request: &glmrt_api::RealFullRequest,
    stateful_decode_step: bool,
) -> usize {
    real_full_request_mtp_rows_for_policy(
        request,
        stateful_decode_step,
        real_full_request_mtp_verify_enabled(),
    )
}

fn real_full_request_mtp_rows_for_policy(
    request: &glmrt_api::RealFullRequest,
    stateful_decode_step: bool,
    mtp_verify_enabled: bool,
) -> usize {
    if stateful_decode_step
        || !mtp_verify_enabled
        || request.decode_budget <= 1
        || request.decode_budget > request.max_tokens
    {
        0
    } else {
        request
            .max_tokens
            .max(1)
            .min(REAL_FULL_REQUEST_MAX_MTP_VERIFY_ROWS)
    }
}

fn real_full_request_mtp_verify_enabled() -> bool {
    env::var(REAL_FULL_REQUEST_MTP_VERIFY_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_dspark_enabled() -> bool {
    env::var(REAL_FULL_DSPARK_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_dflash2_enabled() -> bool {
    env::var(REAL_FULL_DFLASH2_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_draft_runtime_enabled() -> bool {
    real_full_dspark_enabled() || real_full_dflash2_enabled()
}

fn real_full_active_draft_target_layer_ids() -> Vec<usize> {
    if real_full_dflash2_enabled() {
        GLM53_DFLASH2_TARGET_CAPTURE_TAPS.to_vec()
    } else {
        dspark_target_hidden_tap_layer_ids().to_vec()
    }
}

fn real_full_dspark_shadow_enabled() -> bool {
    env::var(REAL_FULL_DSPARK_SHADOW_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_dspark_trace_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_DSPARK_TRACE_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

fn real_full_dspark_startup_profile_mode() -> Result<RealFullDsparkStartupProfileMode> {
    match env::var(REAL_FULL_DSPARK_PROFILE_AT_STARTUP_ENV)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        None | Some("") | Some("0") | Some("false") | Some("off") => {
            Ok(RealFullDsparkStartupProfileMode::Disabled)
        }
        Some("report") | Some("diagnostic") => Ok(RealFullDsparkStartupProfileMode::Report),
        Some("1") | Some("true") | Some("on") | Some("install") => {
            Ok(RealFullDsparkStartupProfileMode::Install)
        }
        Some(value) => bail!(
            "{REAL_FULL_DSPARK_PROFILE_AT_STARTUP_ENV} must be off, report, or install, got {value}"
        ),
    }
}

fn real_full_dspark_startup_profile_samples() -> Result<usize> {
    let samples = env::var(REAL_FULL_DSPARK_PROFILE_SAMPLES_ENV)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("parsing {REAL_FULL_DSPARK_PROFILE_SAMPLES_ENV}={value}"))
        })
        .transpose()?
        .unwrap_or(4);
    anyhow::ensure!(
        (4..=64).contains(&samples),
        "{REAL_FULL_DSPARK_PROFILE_SAMPLES_ENV} must be in 4..=64, got {samples}"
    );
    Ok(samples)
}

fn parse_real_full_dspark_confidence_policy(
    value: Option<&str>,
) -> Result<RealFullDsparkConfidencePolicy> {
    match value.map(str::trim).filter(|value| !value.is_empty()) {
        None | Some("residual") => Ok(RealFullDsparkConfidencePolicy::Residual),
        Some("calibrated") => Ok(RealFullDsparkConfidencePolicy::Calibrated),
        Some("raw") => Ok(RealFullDsparkConfidencePolicy::Raw),
        Some(value) => {
            bail!(
                "{REAL_FULL_DSPARK_CONFIDENCE_POLICY_ENV} must be calibrated, raw, or residual, got {value}"
            )
        }
    }
}

fn real_full_dspark_confidence_policy() -> Result<RealFullDsparkConfidencePolicy> {
    parse_real_full_dspark_confidence_policy(
        env::var(REAL_FULL_DSPARK_CONFIDENCE_POLICY_ENV)
            .ok()
            .as_deref(),
    )
}

fn real_full_dspark_fixed_drafts() -> Result<Option<usize>> {
    let Some(value) =
        env::var_os(REAL_FULL_DSPARK_FIXED_DRAFTS_ENV).filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{REAL_FULL_DSPARK_FIXED_DRAFTS_ENV} is not valid UTF-8"))?;
    let drafts = value
        .parse::<usize>()
        .with_context(|| format!("parsing {REAL_FULL_DSPARK_FIXED_DRAFTS_ENV}={value}"))?;
    anyhow::ensure!(
        drafts <= REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS,
        "{REAL_FULL_DSPARK_FIXED_DRAFTS_ENV} must be in 0..={REAL_FULL_DSPARK_MAX_VERIFY_DRAFTS}, got {drafts}"
    );
    Ok(Some(drafts))
}

fn real_full_dflash2_fixed_drafts() -> Result<Option<usize>> {
    let Some(value) =
        env::var_os(REAL_FULL_DFLASH2_FIXED_DRAFTS_ENV).filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{REAL_FULL_DFLASH2_FIXED_DRAFTS_ENV} is not valid UTF-8"))?;
    if value.trim().eq_ignore_ascii_case("adaptive") {
        return Ok(None);
    }
    let drafts = value
        .parse::<usize>()
        .with_context(|| format!("parsing {REAL_FULL_DFLASH2_FIXED_DRAFTS_ENV}={value}"))?;
    anyhow::ensure!(
        (1..=GLM53_DFLASH2_MAX_DRAFTS).contains(&drafts),
        "{REAL_FULL_DFLASH2_FIXED_DRAFTS_ENV} must be adaptive or in 1..={GLM53_DFLASH2_MAX_DRAFTS}, got {drafts}; use speculation=plain for the target-only baseline"
    );
    Ok(Some(drafts))
}

fn real_full_active_fixed_drafts() -> Result<Option<usize>> {
    if real_full_dflash2_enabled() {
        real_full_dflash2_fixed_drafts()
    } else {
        real_full_dspark_fixed_drafts()
    }
}

fn real_full_active_max_verify_drafts() -> usize {
    if real_full_dflash2_enabled() {
        GLM53_DFLASH2_MAX_DRAFTS
    } else {
        dspark_active_max_verify_drafts()
    }
}

fn real_full_dflash2_snapshot() -> Result<Option<PathBuf>> {
    if !real_full_dflash2_enabled() {
        return Ok(None);
    }
    anyhow::ensure!(
        !real_full_dspark_enabled() && !real_full_dspark_shadow_enabled(),
        "{REAL_FULL_DFLASH2_ENV}=1 cannot be combined with dSpark serving"
    );
    anyhow::ensure!(
        !real_full_mtp_enabled(),
        "{REAL_FULL_DFLASH2_ENV}=1 and {REAL_FULL_MTP_ENV}=1 cannot both be enabled"
    );
    let snapshot = env::var_os(REAL_FULL_DFLASH2_SNAPSHOT_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .with_context(|| format!("DFlash2 serving requires {REAL_FULL_DFLASH2_SNAPSHOT_ENV}"))?;
    anyhow::ensure!(
        snapshot.is_dir(),
        "configured DFlash2 snapshot does not exist: {}",
        snapshot.display()
    );
    Ok(Some(snapshot))
}

fn real_full_dflash2_tail_cache_bytes(max_execution_lanes: usize) -> Result<usize> {
    let page_size = 64_usize;
    let retained_tokens = (super::dflash::GLM53_DFLASH2_SLIDING_WINDOW + page_size - 1)
        .div_ceil(page_size)
        .checked_mul(page_size)
        .context("DFlash2 tail snapshot page rounding overflow")?;
    let default_bytes = retained_tokens
        .checked_mul(REAL_FULL_DFLASH2_BF16_KV_BYTES_PER_TOKEN)
        .and_then(|bytes| bytes.checked_mul(max_execution_lanes))
        .context("DFlash2 tail-cache default byte count overflow")?;
    match env::var(REAL_FULL_DFLASH2_TAIL_CACHE_BYTES_ENV) {
        Ok(value) => value.parse::<usize>().with_context(|| {
            format!("{REAL_FULL_DFLASH2_TAIL_CACHE_BYTES_ENV}={value} must be a byte count")
        }),
        Err(env::VarError::NotPresent) => Ok(default_bytes),
        Err(error) => {
            Err(error).with_context(|| format!("reading {REAL_FULL_DFLASH2_TAIL_CACHE_BYTES_ENV}"))
        }
    }
}

fn real_full_dspark_mode_and_snapshot() -> Result<Option<(RealFullDsparkServingMode, PathBuf)>> {
    let active = real_full_dspark_enabled();
    let shadow = real_full_dspark_shadow_enabled();
    anyhow::ensure!(
        !(active && shadow),
        "{REAL_FULL_DSPARK_ENV} and {REAL_FULL_DSPARK_SHADOW_ENV} cannot both be enabled"
    );
    if !active && !shadow {
        return Ok(None);
    }
    anyhow::ensure!(
        !(active && real_full_mtp_enabled()),
        "{REAL_FULL_DSPARK_ENV}=1 and {REAL_FULL_MTP_ENV}=1 cannot both be enabled"
    );
    let snapshot = env::var_os(REAL_FULL_DSPARK_SNAPSHOT_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .with_context(|| {
            format!(
                "dSpark serving requires {REAL_FULL_DSPARK_SNAPSHOT_ENV} when either {REAL_FULL_DSPARK_ENV} or {REAL_FULL_DSPARK_SHADOW_ENV} is enabled"
            )
        })?;
    anyhow::ensure!(
        snapshot.is_dir(),
        "configured dSpark snapshot does not exist: {}",
        snapshot.display()
    );
    let mode = if active {
        RealFullDsparkServingMode::Active
    } else {
        RealFullDsparkServingMode::Shadow
    };
    Ok(Some((mode, snapshot)))
}

fn real_full_dspark_context_tokens(cache_mode: RealFullDsparkCacheMode) -> Result<usize> {
    let default_tokens = match cache_mode {
        RealFullDsparkCacheMode::RequestLocal => {
            DEFAULT_REAL_FULL_DSPARK_REQUEST_LOCAL_CONTEXT_TOKENS
        }
        RealFullDsparkCacheMode::PromptSwa => DEFAULT_REAL_FULL_DSPARK_PROMPT_SWA_CONTEXT_TOKENS,
    };
    let tokens = env::var(REAL_FULL_DSPARK_CONTEXT_TOKENS_ENV)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .with_context(|| format!("parsing {REAL_FULL_DSPARK_CONTEXT_TOKENS_ENV}={value}"))
        })
        .transpose()?
        .unwrap_or(default_tokens);
    anyhow::ensure!(
        tokens > 0 && tokens % REAL_FULL_DSPARK_PAGE_SIZE == 0,
        "dSpark context token limit must be a positive multiple of the 64-token page size"
    );
    Ok(tokens)
}

fn real_full_dspark_tail_cache_bytes(context_tokens: usize) -> Result<usize> {
    let maximum_retained_tokens = context_tokens
        .checked_add(REAL_FULL_DSPARK_PAGE_SIZE - 1)
        .context("dSpark tail retention ceiling overflow")?;
    let maximum_snapshot_tokens = maximum_retained_tokens
        .div_ceil(REAL_FULL_DSPARK_PAGE_SIZE)
        .checked_mul(REAL_FULL_DSPARK_PAGE_SIZE)
        .context("dSpark tail snapshot page rounding overflow")?;
    let maximum_snapshot_bytes = maximum_snapshot_tokens
        .checked_mul(REAL_FULL_DSPARK_BF16_KV_BYTES_PER_TOKEN)
        .context("dSpark tail snapshot byte count overflow")?;
    let default_bytes = maximum_snapshot_bytes
        .checked_mul(REAL_FULL_MAX_ACTIVE_REQUESTS)
        .context("dSpark tail-cache default byte count overflow")?;
    match env::var(REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV) {
        Ok(value) => value.parse::<usize>().with_context(|| {
            format!("{REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV}={value} must be a byte count")
        }),
        Err(env::VarError::NotPresent) => Ok(default_bytes),
        Err(error) => {
            Err(error).with_context(|| format!("reading {REAL_FULL_DSPARK_TAIL_CACHE_BYTES_ENV}"))
        }
    }
}

fn real_full_dspark_cache_mode() -> Result<RealFullDsparkCacheMode> {
    match env::var(REAL_FULL_DSPARK_CACHE_MODE_ENV)
        .unwrap_or_else(|_| "request-local".to_owned())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "request-local" | "request_local" | "mode1" | "1" => {
            Ok(RealFullDsparkCacheMode::RequestLocal)
        }
        "prompt-swa" | "prompt_swa" | "mode2" | "2" => Ok(RealFullDsparkCacheMode::PromptSwa),
        value => bail!(
            "{REAL_FULL_DSPARK_CACHE_MODE_ENV} must be request-local/mode1 or prompt-swa/mode2, got {value}"
        ),
    }
}

fn request_prompt_token_ids(
    tokenizer: &LoadedTokenizer,
    request: &glmrt_api::RealFullRequest,
) -> Result<Option<Vec<usize>>> {
    if let Some(token_ids) = request.prompt_token_ids.as_ref() {
        anyhow::ensure!(
            token_ids.len() == request.prompt_tokens,
            "real-full explicit prompt token count differs from request: explicit={} request={}",
            token_ids.len(),
            request.prompt_tokens,
        );
        return Ok(Some(token_ids.as_ref().clone()));
    }
    let encoding = tokenizer
        .encode_text(&request.prompt, false)
        .with_context(|| {
            format!(
                "tokenizing real-full request prompt for {}",
                request.request_id
            )
        })?;
    anyhow::ensure!(
        encoding.token_count == request.prompt_tokens,
        "real-full request tokenizer count changed between API and daemon: api={} daemon={}",
        request.prompt_tokens,
        encoding.token_count
    );
    let token_ids = encoding
        .token_ids
        .into_iter()
        .map(|token_id| token_id as usize)
        .collect::<Vec<_>>();
    Ok(Some(token_ids))
}

pub(crate) fn run_real_glm_full_preflight(args: &CoordinatorArgs) -> Result<()> {
    let (catalog_path, catalog) = load_real_glm_full_catalog(args)?;
    let report = real_glm_full_preflight_report(args, &catalog_path, &catalog)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    bail!("{}", REAL_GLM_FULL_BLOCKER)
}

fn real_full_expert_ready_timeout_secs() -> Result<u64> {
    let timeout_secs = env::var(REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value.parse::<u64>().with_context(|| {
                format!("parsing {REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV}={value}")
            })
        })
        .transpose()?
        .unwrap_or(DEFAULT_REAL_FULL_EXPERT_READY_TIMEOUT_SECS);
    anyhow::ensure!(
        timeout_secs > 0,
        "{REAL_FULL_EXPERT_READY_TIMEOUT_SECS_ENV} must be positive"
    );
    Ok(timeout_secs)
}

fn wait_for_real_full_sparse_targets(
    targets: Option<&[TcpProtocolV2HostBatchTarget]>,
) -> Result<()> {
    let Some(targets) = targets else {
        return Ok(());
    };
    let timeout_secs = real_full_expert_ready_timeout_secs()?;
    let started = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    let connect_timeout = Duration::from_millis(200);
    let mut pending = targets.iter().collect::<Vec<_>>();
    eprintln!(
        "real_full_expert_readiness_wait targets={} timeout_secs={timeout_secs}",
        pending.len(),
    );
    while !pending.is_empty() {
        pending.retain(|target| TcpStream::connect_timeout(&target.addr, connect_timeout).is_err());
        if pending.is_empty() {
            break;
        }
        if started.elapsed() >= timeout {
            let pending_targets = pending
                .iter()
                .map(|target| format!("{}={}", target.host, target.addr))
                .collect::<Vec<_>>()
                .join(",");
            anyhow::bail!(
                "expert daemons did not become ready within {timeout_secs}s: {pending_targets}"
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "real_full_expert_readiness_ready targets={} elapsed_ms={:.3}",
        targets.len(),
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn wait_for_real_full_expert_warmup() -> Result<()> {
    let Some(status_file) = env::var_os(REAL_FULL_EXPERT_WARMUP_STATUS_FILE_ENV)
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    else {
        return Ok(());
    };
    let timeout_secs = real_full_expert_ready_timeout_secs()?;
    let started = Instant::now();
    let timeout = Duration::from_secs(timeout_secs);
    eprintln!(
        "real_full_expert_warmup_wait status_file={} timeout_secs={timeout_secs}",
        status_file.display(),
    );
    loop {
        match fs::read_to_string(&status_file) {
            Ok(status) if status.trim() == "ready" => break,
            Ok(status) if status.trim().starts_with("failed") => {
                anyhow::bail!(
                    "expert precompile warmup failed: {}",
                    status.trim().replace('\n', " ")
                );
            }
            Ok(status) => {
                anyhow::bail!(
                    "invalid expert precompile warmup status in {}: {:?}",
                    status_file.display(),
                    status.trim()
                );
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "reading expert precompile warmup status {}",
                        status_file.display()
                    )
                });
            }
        }
        if started.elapsed() >= timeout {
            anyhow::bail!(
                "expert precompile warmup did not finish within {timeout_secs}s: {}",
                status_file.display()
            );
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    eprintln!(
        "real_full_expert_warmup_ready elapsed_ms={:.3}",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
    Ok(())
}

fn report_real_full_startup_phase(
    stage: &str,
    startup_started: Instant,
    phase_started: &mut Instant,
) {
    let now = Instant::now();
    eprintln!(
        "real_full_startup_phase stage={stage} elapsed_ms={:.3} total_ms={:.3}",
        now.duration_since(*phase_started).as_secs_f64() * 1_000.0,
        now.duration_since(startup_started).as_secs_f64() * 1_000.0,
    );
    *phase_started = now;
}

pub(crate) fn load_real_full_serving(
    args: &CoordinatorArgs,
    python_capture_barrier: impl FnOnce() -> Result<()>,
) -> Result<LoadedRealFullServing> {
    let startup_started = Instant::now();
    let mut phase_started = startup_started;
    if args.backend != "cuda-reference" {
        anyhow::ensure!(
            coordinator_python_capture_enabled(),
            "real-glm-full serving requires GLMRT_B12X=1 for startup-captured production kernels"
        );
        anyhow::ensure!(
            real_full_serve_prewarm_request_enabled(),
            "real-glm-full serving requires startup prewarm so all production CUDA graphs are captured before serving"
        );
    }
    report_real_full_startup_phase("validation", startup_started, &mut phase_started);
    let (_catalog_path, catalog) = load_real_glm_full_catalog(args)?;
    let mut kv_config = real_full_kv_cache_config(args)?;
    if real_full_mtp_enabled() || real_full_mtp_probe_enabled() {
        kv_config = kv_config.with_mtp_layer();
    }
    let kv_config = kv_config.with_mla_representation(MlaKvCacheRepresentation::NormalizedRotated);
    println!(
        "real-glm-full MLA KV representation={}",
        kv_config.mla_representation.label()
    );
    report_real_full_startup_phase("catalog-kv-config", startup_started, &mut phase_started);
    let sparse_dispatch_transport =
        RealFullSchedulerSparseDispatchTransport::from_label(args.transport.as_str());
    let sparse_tcp_targets = real_full_sparse_tcp_targets_from_args(args)?;
    let sparse_owner_lookup = if sparse_tcp_targets.is_some() {
        real_full_sparse_owner_lookup_from_args(args, &catalog)?
    } else {
        None
    };
    let tokenizer = LoadedTokenizer::from_snapshot(Path::new(&catalog.snapshot_path))
        .context("loading real-full serving tokenizer")?;
    let constraint_vocab_size = catalog
        .tensors
        .iter()
        .find(|tensor| tensor.role == TensorRole::LmHead)
        .and_then(|tensor| tensor.shape.first().copied())
        .context("real-full constrained decoding requires a 2D lm_head tensor")?;
    let constraint_compiler = RealFullConstraintCompiler::new(
        Path::new(&catalog.snapshot_path).join("tokenizer.json"),
        constraint_vocab_size,
    )
    .context("configuring lazy real-full constrained decoding")?;
    report_real_full_startup_phase("targets-tokenizer", startup_started, &mut phase_started);
    let kv_snapshot_load_path = optional_nonempty_env_path(REAL_FULL_KV_SNAPSHOT_LOAD_ENV)?;
    let kv_snapshot_save_path = optional_nonempty_env_path(REAL_FULL_KV_SNAPSHOT_SAVE_ENV)?;
    let kv_snapshot_save_tokens =
        optional_positive_env_usize(REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS_ENV)?;
    anyhow::ensure!(
        kv_snapshot_save_path.is_some() || kv_snapshot_save_tokens.is_none(),
        "{REAL_FULL_KV_SNAPSHOT_SAVE_TOKENS_ENV} requires {REAL_FULL_KV_SNAPSHOT_SAVE_ENV}"
    );
    let mut kv_snapshot_saves =
        real_full_kv_snapshot_save_points(REAL_FULL_KV_SNAPSHOT_SAVE_POINTS_ENV)?;
    if let Some(root) = kv_snapshot_save_path {
        kv_snapshot_saves.push(RealFullKvSnapshotSave {
            root,
            token_count: kv_snapshot_save_tokens,
        });
    }
    validate_real_full_kv_snapshot_saves(&kv_snapshot_saves)?;
    let snapshot_enabled = kv_snapshot_load_path.is_some() || !kv_snapshot_saves.is_empty();
    let kv_snapshot_load = kv_snapshot_load_path
        .as_deref()
        .map(|path| load_real_full_kv_snapshot(path, &catalog, &kv_config))
        .transpose()?
        .map(Arc::new);
    if let Some(snapshot) = kv_snapshot_load.as_ref() {
        anyhow::ensure!(
            (!real_full_mtp_enabled() && !real_full_mtp_probe_enabled())
                || snapshot.is_mtp_ready(),
            "packed KV/DSA snapshot {} is target-only: its MTP layer frontier is {}/{} tokens; native MTP requires a snapshot created while MTP or the MTP probe populated layer 78",
            snapshot.root().display(),
            snapshot.mtp_layer_token_count(),
            snapshot.token_count(),
        );
        println!(
            "real-full packed KV/DSA snapshot loaded path={} tokens={} mtp_layer_tokens={} (device restore is deferred until request admission)",
            snapshot.root().display(),
            snapshot.token_count(),
            snapshot.mtp_layer_token_count(),
        );
    }
    for save in &kv_snapshot_saves {
        anyhow::ensure!(
            !save.root.exists(),
            "KV snapshot save destination already exists: {}",
            save.root.display()
        );
    }
    report_real_full_startup_phase("kv-snapshot-config", startup_started, &mut phase_started);
    let engine_commit = env::var(REAL_FULL_ENGINE_COMMIT_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    let prefix_prefill_probe =
        real_full_serve_prefix_prefill_probe(&tokenizer, kv_config.max_tokens)?;
    let prewarm_prompts = if real_full_serve_prewarm_request_enabled()
        || prefix_prefill_probe.is_some()
    {
        let mut prewarm_prefill_rows = REAL_FULL_SERVE_PREWARM_PREFILL_ROWS.to_vec();
        if kv_config.dtype == KvCacheDType::Nvfp4 {
            // Native NVFP4 uses a live sparse-K=2048 graph without the DSA
            // selector from context 1025 through 2048. FP8 always uses K=2048
            // and does not need this extra canonical boundary.
            prewarm_prefill_rows.push(1_024);
            prewarm_prefill_rows.sort_unstable_by(|left, right| right.cmp(left));
            prewarm_prefill_rows.dedup();
        }
        let prompts = prewarm_prefill_rows
            .iter()
            .map(|prefill_rows| {
                let prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(*prefill_rows);
                let prompt_tokens = tokenizer
                    .encode_text(&prompt, false)
                    .context("tokenizing real-full serving prewarm prompt")?
                    .token_count;
                anyhow::ensure!(
                    prompt_tokens == prefill_rows + 1,
                    "real-full serving prewarm prompt produced {prompt_tokens} tokens for {prefill_rows} requested prefill rows"
                );
                Ok((prompt, prompt_tokens))
            })
            .collect::<Result<Vec<_>>>()?;
        let largest_prompt_tokens = prompts
            .first()
            .map(|(_, prompt_tokens)| *prompt_tokens)
            .context("real-full serving prewarm prompt set is empty")?;
        let prefill_tokens = largest_prompt_tokens.saturating_sub(1);
        let prefill_chunk_tokens = real_full_request_prefill_chunk_tokens();
        anyhow::ensure!(
            prefill_tokens >= prefill_chunk_tokens,
            "real-full serving largest prewarm prompt token count {largest_prompt_tokens} does not include an exact {}-row prefill chunk",
            prefill_chunk_tokens,
        );
        Some(prompts)
    } else {
        None
    };
    report_real_full_startup_phase("prewarm-prompts", startup_started, &mut phase_started);
    let preload = preload_real_full_coordinator_resident_weights(&catalog)?;
    println!(
        "real-glm-full coordinator resident preload status={} tensors={} bytes={}",
        preload.status, preload.selected_tensor_count, preload.loaded_tensor_bytes
    );
    report_real_full_startup_phase(
        "coordinator-resident-preload",
        startup_started,
        &mut phase_started,
    );
    let max_execution_lanes = real_full_max_execution_lanes()?;
    // Draft KV is an execution lease, not queued-request or target-radix
    // residency. Size this pool for the lanes that can actually execute;
    // submitted requests beyond that limit must wait without owning pages.
    let dspark_load_started = Instant::now();
    let dspark = if let Some(snapshot) = real_full_dflash2_snapshot()? {
        let tail_cache_bytes = real_full_dflash2_tail_cache_bytes(max_execution_lanes)?;
        let fixed_drafts = real_full_dflash2_fixed_drafts()?;
        let engine = Dflash2RequestEngine::load(
            &snapshot,
            &catalog,
            kv_config.max_tokens,
            max_execution_lanes,
            // DFlash2 emits all seven candidates in one local suffix replay.
            // Keep proposal generation at the checkpoint maximum so the
            // target verifier can select a cheaper prefix without recapturing
            // or switching the draft graph family.
            GLM53_DFLASH2_MAX_DRAFTS,
        )
        .with_context(|| {
            format!(
                "loading DFlash2 request executor from {}",
                snapshot.display()
            )
        })?;
        let max_verify_drafts = engine.max_verify_drafts();
        if let Some(fixed_drafts) = fixed_drafts {
            anyhow::ensure!(
                fixed_drafts <= max_verify_drafts,
                "fixed DFlash2 width {fixed_drafts} exceeds the captured internal width {max_verify_drafts}"
            );
        }
        let selector = if fixed_drafts.is_some() {
            "greedy-fixed-width"
        } else {
            "empirical-survival-adaptive"
        };
        let verify_width = fixed_drafts
            .map(|drafts| drafts.to_string())
            .unwrap_or_else(|| "adaptive".to_owned());
        let mut cost_model =
            DsparkRuntimeCostModel::new(REAL_FULL_MAX_ACTIVE_REQUESTS, max_verify_drafts)?;
        let sparkinfer_revision = env::var(REAL_FULL_DSPARK_SPARKINFER_REVISION_ENV).ok();
        let coordinator_power_limit_watts =
            env::var(REAL_FULL_DSPARK_COORDINATOR_POWER_LIMIT_WATTS_ENV)
                .ok()
                .map(|value| {
                    value.parse::<usize>().with_context(|| {
                        format!(
                            "parsing {REAL_FULL_DSPARK_COORDINATOR_POWER_LIMIT_WATTS_ENV}={value}"
                        )
                    })
                })
                .transpose()?;
        let cost_profile = install_qualified_dflash2_cost_profile(
            &mut cost_model,
            &catalog.model_id,
            Path::new(&catalog.snapshot_path),
            engine.checkpoint_model_id(),
            engine.checkpoint_revision(),
            sparkinfer_revision.as_deref(),
            coordinator_power_limit_watts,
            max_execution_lanes,
            max_verify_drafts,
        )?;
        println!(
            "real-full DFlash2 ready mode=Active selector={} snapshot={} window_tokens={} proposal_drafts={} verify_drafts={} internal_query_rows={} gpu_request_slots={} host_tail_cache_bytes={} non_greedy=fallback-target cost_profile={} cost_profile_source={} cost_profile_sparkinfer={} cost_profile_topology={} cost_profile_power_watts={}",
            selector,
            snapshot.display(),
            super::dflash::GLM53_DFLASH2_SLIDING_WINDOW,
            max_verify_drafts,
            verify_width,
            max_verify_drafts + 1,
            max_execution_lanes,
            tail_cache_bytes,
            cost_profile.map_or("unqualified-runtime-prior", |profile| profile.profile_id),
            cost_profile.map_or("none", |profile| profile.source_sha256),
            cost_profile.map_or("unqualified", |profile| profile.sparkinfer_revision),
            cost_profile.map_or("unqualified", |profile| profile.topology),
            cost_profile.map_or(0, |profile| profile.power_limit_watts),
        );
        Some(Mutex::new(RealFullDsparkRuntime {
            mode: RealFullDsparkServingMode::Active,
            confidence_policy: RealFullDsparkConfidencePolicy::Raw,
            cache_mode: RealFullDsparkCacheMode::PromptSwa,
            context_tokens: super::dflash::GLM53_DFLASH2_SLIDING_WINDOW,
            engine: RealFullDraftEngine::Dflash2(engine),
            requests: HashMap::new(),
            tail_cache: RealFullDsparkTailCache::new(tail_cache_bytes),
            cost_model,
        }))
    } else {
        real_full_dspark_mode_and_snapshot()?
        .map(|(mode, snapshot)| {
            let cache_mode = real_full_dspark_cache_mode()?;
            let confidence_policy = real_full_dspark_confidence_policy()?;
            let context_tokens = real_full_dspark_context_tokens(cache_mode)?;
            let tail_cache_bytes = real_full_dspark_tail_cache_bytes(context_tokens)?;
            let kv_capacity_tokens = context_tokens
                .checked_add(REAL_FULL_DSPARK_PAGE_SIZE)
                .context("dSpark request page-slop capacity overflow")?
                .checked_add(REAL_FULL_DSPARK_QUERY_ROWS)
                .context("dSpark request KV capacity overflow")?;
            let engine = DsparkRequestEngine::load(
                &snapshot,
                &catalog,
                kv_capacity_tokens,
                max_execution_lanes,
            )
                .with_context(|| {
                    format!("loading dSpark request executor from {}", snapshot.display())
                })?;
            let max_verify_drafts = engine.max_verify_drafts();
            if let Some(fixed_drafts) = real_full_dspark_fixed_drafts()? {
                anyhow::ensure!(
                    fixed_drafts <= max_verify_drafts,
                    "fixed dSpark width {fixed_drafts} exceeds the active checkpoint maximum {max_verify_drafts}"
                );
            }
            let mut cost_model = DsparkRuntimeCostModel::new(
                REAL_FULL_MAX_ACTIVE_REQUESTS,
                max_verify_drafts,
            )?;
            let sparkinfer_revision = env::var(REAL_FULL_DSPARK_SPARKINFER_REVISION_ENV).ok();
            let coordinator_power_limit_watts =
                env::var(REAL_FULL_DSPARK_COORDINATOR_POWER_LIMIT_WATTS_ENV)
                    .ok()
                    .map(|value| {
                        value.parse::<usize>().with_context(|| {
                            format!(
                                "parsing {REAL_FULL_DSPARK_COORDINATOR_POWER_LIMIT_WATTS_ENV}={value}"
                            )
                        })
                    })
                    .transpose()?;
            let cost_profile = install_qualified_dspark_cost_profile(
                &mut cost_model,
                &catalog.model_id,
                Path::new(&catalog.snapshot_path),
                engine.checkpoint_revision(),
                sparkinfer_revision.as_deref(),
                coordinator_power_limit_watts,
                max_execution_lanes,
                max_verify_drafts,
            )?;
            println!(
                "real-full dSpark ready mode={mode:?} confidence_policy={confidence_policy:?} cache_mode={cache_mode:?} snapshot={} context_tokens={} kv_capacity_tokens={} max_verify_drafts={} gpu_request_slots={} host_tail_cache_bytes={} runtime_cost_max_rows={} runtime_context_bucket_tokens={} cost_profile={} cost_profile_source={} cost_profile_sparkinfer={} cost_profile_topology={} cost_profile_power_watts={}",
                snapshot.display(),
                context_tokens,
                kv_capacity_tokens,
                max_verify_drafts,
                max_execution_lanes,
                tail_cache_bytes,
                REAL_FULL_MAX_ACTIVE_REQUESTS * (max_verify_drafts + 1),
                super::dspark::DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS,
                cost_profile.map_or("unqualified-runtime-prior", |profile| profile.profile_id),
                cost_profile.map_or("none", |profile| profile.source_sha256),
                cost_profile.map_or("unqualified", |profile| profile.sparkinfer_revision),
                cost_profile.map_or("unqualified", |profile| profile.topology),
                cost_profile.map_or(0, |profile| profile.power_limit_watts),
            );
            Ok::<_, anyhow::Error>(Mutex::new(RealFullDsparkRuntime {
                mode,
                confidence_policy,
                cache_mode,
                context_tokens,
                engine: RealFullDraftEngine::Dspark(engine),
                requests: HashMap::new(),
                tail_cache: RealFullDsparkTailCache::new(tail_cache_bytes),
                cost_model,
            }))
        })
        .transpose()?
    };
    eprintln!(
        "real_full_dspark_preload elapsed_ms={:.3} enabled={}",
        dspark_load_started.elapsed().as_secs_f64() * 1_000.0,
        dspark.is_some(),
    );
    report_real_full_startup_phase("dspark-preload", startup_started, &mut phase_started);
    wait_for_real_full_sparse_targets(sparse_tcp_targets.as_deref())?;
    report_real_full_startup_phase("sparse-target-connect", startup_started, &mut phase_started);
    wait_for_real_full_expert_warmup()?;
    report_real_full_startup_phase("expert-warmup", startup_started, &mut phase_started);
    let sparse_dispatch_worker_cpu = real_full_scheduler_worker_cpu()?;
    let sparse_tcp_dispatch_worker = sparse_tcp_targets
        .as_ref()
        .map(|targets| {
            RealFullSchedulerSparseTcpDispatchWorker::new_with_transport_and_cpu_affinity(
                sparse_dispatch_transport
                    .expect("configured sparse dispatch target has supported transport"),
                targets.clone(),
                sparse_owner_lookup.clone(),
                sparse_dispatch_worker_cpu,
            )
            .map(Arc::new)
        })
        .transpose()?;
    report_real_full_startup_phase("dispatch-worker", startup_started, &mut phase_started);
    anyhow::ensure!(
        !snapshot_enabled || sparse_tcp_dispatch_worker.is_some() || sparse_tcp_targets.is_none(),
        "packed KV/DSA snapshots require a persistent scheduler execution state"
    );
    let mut info = real_full_info_from_startup(args, &catalog, preload)?;
    info.kv_bytes_per_token = kv_config.bytes_per_token();
    initialize_sparse_tcp_dispatch_status(&mut info, sparse_tcp_targets.as_ref());
    let max_context_tokens = kv_config.max_tokens;
    let serving_kv_dtype = kv_config.dtype;
    let kv_pool_tokens = real_full_kv_pool_tokens(max_context_tokens)?;
    let mut device_kv_pool_config = kv_config.clone();
    device_kv_pool_config.max_tokens = kv_pool_tokens;
    eprintln!(
        "real_full_kv_pool logical_max_tokens={} physical_pool_tokens={} page_tokens={} primary_bytes_per_token={} primary_capacity_bytes={} execution_lanes={}",
        max_context_tokens,
        kv_pool_tokens,
        REAL_FULL_SHARED_KV_PAGE_TOKENS,
        device_kv_pool_config.bytes_per_token(),
        device_kv_pool_config.capacity_bytes(),
        max_execution_lanes,
    );
    let scheduler_executor = RealFullSchedulerRequestExecutor {
        base_info: info.clone(),
        catalog,
        kv_config,
        device_kv_pool_config,
        sparse_tcp_targets,
        sparse_owner_lookup,
        sparse_tcp_dispatch_worker,
        scheduler_states: Mutex::new(HashMap::new()),
        recycled_scheduler_states: Mutex::new(Vec::new()),
        max_execution_lanes,
        device_kv_storage: Mutex::new(None),
        context_budget: Arc::new(RealFullContextTokenBudget::new(kv_pool_tokens)),
        target_kv_radix: Arc::new(
            TargetKvRadixManager::new(kv_pool_tokens, max_execution_lanes)
                .context("creating shared target KV radix manager")?,
        ),
        sampled_token_text_cache: Mutex::new(HashMap::new()),
        tokenizer: Mutex::new(tokenizer),
        constraint_compiler,
        kv_snapshot_load,
        kv_snapshot_saves,
        kv_snapshot_saved: AtomicBool::new(false),
        dspark,
        engine_commit,
    };
    report_real_full_startup_phase("executor-assembly", startup_started, &mut phase_started);
    python_capture_barrier().context("waiting for coordinator Python capture initialization")?;
    report_real_full_startup_phase(
        "python-capture-barrier",
        startup_started,
        &mut phase_started,
    );
    let executor: Arc<dyn glmrt_api::RealFullRequestExecutor> =
        if real_full_request_thread_pinned_enabled() {
            let worker_count = real_full_request_thread_pinned_workers()?;
            let worker_cpus = real_full_request_worker_cpus(worker_count)?;
            let executor =
                glmrt_api::ThreadPinnedRealFullRequestExecutor::spawn_pool_with_cpu_affinity(
                    "glmrt-real-full-request-worker",
                    scheduler_executor,
                    worker_count,
                    &worker_cpus,
                )
                .context("spawning thread-pinned real-full request worker")?;
            report_real_full_startup_phase(
                "request-worker-spawn",
                startup_started,
                &mut phase_started,
            );
            if let Some(prompts) = prewarm_prompts.as_ref() {
                for worker_index in 0..executor.worker_count() {
                    if max_execution_lanes > 1 && real_full_draft_runtime_enabled() {
                        executor
                            .finish_real_full_sequence_on_worker(
                                worker_index,
                                format!(
                                    "{REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX}{worker_index}"
                                ),
                            )
                            .map_err(anyhow::Error::msg)
                            .with_context(|| {
                                format!(
                                    "prewarming paired LM-head graphs for worker {worker_index}"
                                )
                            })?;
                    }
                    report_real_full_startup_phase(
                        "prewarm-paired-lm-head-initial",
                        startup_started,
                        &mut phase_started,
                    );
                    prewarm_real_full_serving_requests(
                        |request| {
                            executor.execute_real_full_decode_cycle_on_worker(worker_index, request)
                        },
                        |sequence_id| {
                            executor.finish_real_full_sequence_on_worker(
                                worker_index,
                                sequence_id.to_owned(),
                            )
                        },
                        prompts,
                        prefix_prefill_probe.as_ref(),
                        worker_index,
                        max_context_tokens,
                        serving_kv_dtype,
                    )?;
                    report_real_full_startup_phase(
                        "prewarm-main",
                        startup_started,
                        &mut phase_started,
                    );
                    if max_execution_lanes > 1 && real_full_draft_runtime_enabled() {
                        executor
                            .finish_real_full_sequence_on_worker(
                                worker_index,
                                format!(
                                    "{REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX}{worker_index}"
                                ),
                            )
                            .map_err(anyhow::Error::msg)
                            .with_context(|| {
                                format!(
                                    "prewarming batched dSpark graphs for worker {worker_index}"
                                )
                            })?;
                        report_real_full_startup_phase(
                            "prewarm-batched-dspark",
                            startup_started,
                            &mut phase_started,
                        );
                    }
                    if serving_kv_dtype == KvCacheDType::Nvfp4 {
                        audit_real_full_nvfp4_short_k_prefill_graphs(
                            |sequence_id| {
                                executor.finish_real_full_sequence_on_worker(
                                    worker_index,
                                    sequence_id.to_owned(),
                                )
                            },
                            worker_index,
                        )?;
                    }
                    executor
                        .finish_real_full_sequence_on_worker(
                            worker_index,
                            format!(
                                "{REAL_FULL_STARTUP_SEAL_OWNED_BUFFER_POOL_PREFIX}{worker_index}"
                            ),
                        )
                        .map_err(anyhow::Error::msg)
                        .with_context(|| {
                            format!(
                                "sealing coordinator owned device-buffer pool for worker {worker_index}"
                            )
                        })?;
                }
            }
            report_real_full_startup_phase(
                "prewarm-audit-seal",
                startup_started,
                &mut phase_started,
            );
            Arc::new(executor)
        } else {
            report_real_full_startup_phase(
                "request-worker-inline",
                startup_started,
                &mut phase_started,
            );
            if let Some(prompts) = prewarm_prompts.as_ref() {
                if max_execution_lanes > 1 && real_full_draft_runtime_enabled() {
                    glmrt_api::RealFullRequestExecutor::finish_real_full_sequence(
                        &scheduler_executor,
                        REAL_FULL_STARTUP_PREWARM_PAIRED_LM_HEAD_PREFIX,
                    )
                    .map_err(anyhow::Error::msg)
                    .context("prewarming paired LM-head graphs")?;
                }
                report_real_full_startup_phase(
                    "prewarm-paired-lm-head-initial",
                    startup_started,
                    &mut phase_started,
                );
                prewarm_real_full_serving_requests(
                    |request| {
                        glmrt_api::RealFullRequestExecutor::execute_real_full_decode_cycle(
                            &scheduler_executor,
                            request,
                        )
                    },
                    |sequence_id| {
                        glmrt_api::RealFullRequestExecutor::finish_real_full_sequence(
                            &scheduler_executor,
                            sequence_id,
                        )
                    },
                    prompts,
                    prefix_prefill_probe.as_ref(),
                    0,
                    max_context_tokens,
                    serving_kv_dtype,
                )?;
                report_real_full_startup_phase("prewarm-main", startup_started, &mut phase_started);
                if max_execution_lanes > 1 && real_full_draft_runtime_enabled() {
                    glmrt_api::RealFullRequestExecutor::finish_real_full_sequence(
                        &scheduler_executor,
                        REAL_FULL_STARTUP_PREWARM_BATCHED_DSPARK_PREFIX,
                    )
                    .map_err(anyhow::Error::msg)
                    .context("prewarming batched dSpark graphs")?;
                    report_real_full_startup_phase(
                        "prewarm-batched-dspark",
                        startup_started,
                        &mut phase_started,
                    );
                }
                if serving_kv_dtype == KvCacheDType::Nvfp4 {
                    audit_real_full_nvfp4_short_k_prefill_graphs(
                        |sequence_id| {
                            glmrt_api::RealFullRequestExecutor::finish_real_full_sequence(
                                &scheduler_executor,
                                sequence_id,
                            )
                        },
                        0,
                    )?;
                }
                seal_coordinator_owned_device_buffer_pool()
                    .context("sealing coordinator owned device-buffer pool")?;
            }
            report_real_full_startup_phase(
                "prewarm-audit-seal",
                startup_started,
                &mut phase_started,
            );
            Arc::new(scheduler_executor)
        };
    report_real_full_startup_phase("complete", startup_started, &mut phase_started);
    Ok(LoadedRealFullServing { info, executor })
}

fn real_full_serve_prewarm_request_enabled() -> bool {
    env::var(REAL_FULL_SERVE_PREWARM_REQUEST_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or_else(|_| coordinator_python_capture_enabled())
}

fn real_full_serve_prefix_prefill_probe_enabled() -> bool {
    env::var(REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn real_full_serve_prefix_prefill_probe_repeats() -> Result<usize> {
    if !real_full_serve_prefix_prefill_probe_enabled() {
        return Ok(1);
    }
    let repeats = env::var(REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| {
            value.parse::<usize>().with_context(|| {
                format!("parsing {REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV} value {value}")
            })
        })
        .transpose()?
        .unwrap_or(1);
    anyhow::ensure!(
        (1..=MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS).contains(&repeats),
        "{REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS_ENV} must be in 1..={MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS}, got {repeats}"
    );
    Ok(repeats)
}

fn real_full_serve_prefix_prefill_probe_rows(name: &str) -> Result<Vec<usize>> {
    let values = env::var(name).ok().filter(|value| !value.trim().is_empty());
    let rows = if let Some(values) = values {
        values
            .split(',')
            .map(str::trim)
            .map(|value| {
                anyhow::ensure!(!value.is_empty(), "{name} contains an empty row count");
                let rows = value
                    .parse::<usize>()
                    .with_context(|| format!("parsing {name} value {value}"))?;
                anyhow::ensure!(rows > 0, "{name} row counts must be greater than zero");
                Ok(rows)
            })
            .collect::<Result<Vec<_>>>()?
    } else {
        vec![DEFAULT_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_ROWS]
    };
    anyhow::ensure!(
        !rows.is_empty(),
        "{name} must contain at least one row count"
    );
    Ok(rows)
}

fn real_full_serve_prefix_prefill_probe(
    tokenizer: &LoadedTokenizer,
    max_context_tokens: usize,
) -> Result<Option<RealFullPrefixPrefillProbe>> {
    if !real_full_serve_prefix_prefill_probe_enabled() {
        return Ok(None);
    }
    let prefix_rows = real_full_serve_prefix_prefill_probe_rows(
        REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_PREFIX_ROWS_ENV,
    )?;
    let new_prompt_rows = real_full_serve_prefix_prefill_probe_rows(
        REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS_ENV,
    )?;
    anyhow::ensure!(
        new_prompt_rows.iter().all(|rows| *rows > 1),
        "{REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_NEW_ROWS_ENV} must be greater than one"
    );
    let case_count = prefix_rows
        .len()
        .checked_mul(new_prompt_rows.len())
        .context("real-full serving prefix-prefill probe case count overflow")?;
    let mut cases = Vec::with_capacity(case_count);
    for prefix_rows in prefix_rows {
        let prefix_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(prefix_rows);
        let prefix_prompt_tokens = tokenizer
            .encode_text(&prefix_prompt, false)
            .context("tokenizing real-full serving prefix-prefill probe prefix")?
            .token_count;
        anyhow::ensure!(
            prefix_prompt_tokens == prefix_rows + 1,
            "real-full serving prefix-prefill probe prefix produced {prefix_prompt_tokens} tokens for {prefix_rows} requested rows"
        );
        for new_prompt_rows in new_prompt_rows.iter().copied() {
            let seed_decode_budget = new_prompt_rows
                .checked_add(1)
                .context("real-full serving prefix-prefill probe decode budget overflow")?;
            real_full_sequence_capacity_tokens(
                prefix_prompt_tokens,
                seed_decode_budget,
                max_context_tokens,
            )
            .context("reserving real-full serving prefix-prefill probe context")?;
            cases.push(RealFullPrefixPrefillProbeCase {
                prefix_prompt: prefix_prompt.clone(),
                prefix_prompt_tokens,
                new_prompt_rows,
            });
        }
    }
    Ok(Some(RealFullPrefixPrefillProbe {
        cases,
        repeats: real_full_serve_prefix_prefill_probe_repeats()?,
    }))
}

fn real_full_startup_workspace_is_final_capture_set(
    prefix_prefill_probe_enabled: bool,
    native_mtp_enabled: bool,
    native_mtp_probe_enabled: bool,
) -> bool {
    !prefix_prefill_probe_enabled && !native_mtp_enabled && !native_mtp_probe_enabled
}

fn prewarm_real_full_serving_requests(
    mut execute: impl FnMut(
        glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String>,
    mut finish_sequence: impl FnMut(&str) -> std::result::Result<(), String>,
    prompts: &[(String, usize)],
    prefix_prefill_probe: Option<&RealFullPrefixPrefillProbe>,
    worker_index: usize,
    max_context_tokens: usize,
    kv_dtype: KvCacheDType,
) -> Result<()> {
    let start = Instant::now();
    let canonical_workspace_complete = real_full_startup_workspace_is_final_capture_set(
        prefix_prefill_probe.is_some(),
        real_full_mtp_enabled(),
        real_full_mtp_probe_enabled(),
    );
    let dsa_selector_query_rows = REAL_FULL_SERVE_DSA_SELECTOR_PREWARM_QUERY_ROWS
        .iter()
        .copied()
        .filter(|query_rows| *query_rows < real_full_request_prefill_chunk_tokens())
        .collect::<Vec<_>>();
    let dsa_selector_decode_budget = dsa_selector_query_rows.len() + 1;
    let mut recurrent_seed = None;
    let configured_prefill_chunk_tokens = real_full_request_prefill_chunk_tokens();
    let max_prefill_rows = max_context_tokens.checked_sub(2).context(
        "real-full serving max-chunk prewarm needs a tokenizer boundary and output token",
    )?;
    // Three full-width chunks establish GLM's initial, continuation/non-final,
    // and continuation/final graph lifecycles plus the maximum permanent
    // buffers without replaying a fourth identical continuation wave.
    let max_chunk_sizing_rows = configured_prefill_chunk_tokens
        .checked_mul(3)
        .filter(|rows| *rows <= max_prefill_rows)
        .unwrap_or(configured_prefill_chunk_tokens);
    let max_chunk_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(max_chunk_sizing_rows);
    let max_chunk_prompt_tokens = max_chunk_sizing_rows
        .checked_add(1)
        .context("real-full serving max-chunk prewarm prompt token count overflow")?;
    let max_chunk_sequence_id = format!(
        "{REAL_FULL_STARTUP_MAX_PREFILL_CHUNK_PREFIX}{max_chunk_prompt_tokens}-sequence-{worker_index}"
    );
    let mut max_chunk_request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
        0,
        &max_chunk_sequence_id,
        &max_chunk_prompt,
        max_chunk_prompt_tokens,
        1,
        Vec::new(),
        0,
        1,
    );
    max_chunk_request.disable_speculation = true;
    let max_chunk_start = Instant::now();
    eprintln!(
        "real_full_startup_prewarm_start worker={} stage=max-prefill-chunk prompt_tokens={} chunk_rows={}",
        worker_index, max_chunk_prompt_tokens, configured_prefill_chunk_tokens,
    );
    let max_chunk_cycle = execute(max_chunk_request)
        .map_err(anyhow::Error::msg)
        .context("capturing the configured maximum prefill chunk")?;
    let max_chunk_info = max_chunk_cycle.info;
    let expected_max_chunk_waves = max_chunk_sizing_rows.div_ceil(configured_prefill_chunk_tokens);
    anyhow::ensure!(
        max_chunk_info.status == "ready"
            && max_chunk_info.request_prefill_tokens == max_chunk_sizing_rows
            && max_chunk_info.request_prefill_chunks == expected_max_chunk_waves,
        "maximum prefill-chunk startup sizing failed: status={} prefill_rows={} chunks={} expected_rows={} expected_chunks={} blocker={} failed={:?}",
        max_chunk_info.status,
        max_chunk_info.request_prefill_tokens,
        max_chunk_info.request_prefill_chunks,
        max_chunk_sizing_rows,
        expected_max_chunk_waves,
        max_chunk_info.blocker,
        max_chunk_info.failed_requirements,
    );
    eprintln!(
        "real_full_startup_prewarm_step_done worker={} stage=max-prefill-chunk prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} expert_batches={} expert_rows={} graph_captures={} captured_graphs={}",
        worker_index,
        max_chunk_prompt_tokens,
        elapsed_ms(max_chunk_start),
        elapsed_ms(start),
        max_chunk_info.sparse_expert_batches,
        max_chunk_info.request_expert_batch_rows,
        max_chunk_info.request_coordinator_graph_captures,
        max_chunk_info.request_coordinator_graph_captured_graphs,
    );
    finish_sequence(&max_chunk_sequence_id)
        .map_err(anyhow::Error::msg)
        .context("finishing the maximum prefill-chunk startup sequence")?;

    // Exercise the ordinary four-lane 1K geometry with the same three-wave
    // lifecycle coverage. The max-width pass above and this pass
    // replace the old four-wave 8K and 4K sizing requests.
    let canonical_chunk_tokens = configured_prefill_chunk_tokens
        .min(REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_TOKENS)
        .max(1);
    if canonical_chunk_tokens < configured_prefill_chunk_tokens {
        let canonical_sizing_rows = canonical_chunk_tokens
            .checked_mul(3)
            .filter(|rows| *rows <= max_prefill_rows)
            .unwrap_or(canonical_chunk_tokens);
        let canonical_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(canonical_sizing_rows);
        let canonical_prompt_tokens = canonical_sizing_rows
            .checked_add(1)
            .context("real-full serving canonical-chunk prewarm token count overflow")?;
        let canonical_sequence_id = format!(
            "{REAL_FULL_STARTUP_CANONICAL_PREFILL_CHUNK_PREFIX}{canonical_sizing_rows}-sequence-{worker_index}"
        );
        let mut canonical_request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            0,
            &canonical_sequence_id,
            &canonical_prompt,
            canonical_prompt_tokens,
            1,
            Vec::new(),
            0,
            1,
        );
        canonical_request.disable_speculation = true;
        let canonical_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=canonical-prefill-chunk prompt_tokens={} chunk_rows={}",
            worker_index, canonical_prompt_tokens, canonical_chunk_tokens,
        );
        let canonical_cycle = execute(canonical_request)
            .map_err(anyhow::Error::msg)
            .context("capturing the canonical prefill chunk")?;
        let canonical_info = canonical_cycle.info;
        let expected_canonical_waves = canonical_sizing_rows.div_ceil(canonical_chunk_tokens);
        anyhow::ensure!(
            canonical_info.status == "ready"
                && canonical_info.request_prefill_tokens == canonical_sizing_rows
                && canonical_info.request_prefill_chunks == expected_canonical_waves,
            "canonical prefill-chunk startup sizing failed: status={} prefill_rows={} chunks={} expected_rows={} expected_chunks={} blocker={} failed={:?}",
            canonical_info.status,
            canonical_info.request_prefill_tokens,
            canonical_info.request_prefill_chunks,
            canonical_sizing_rows,
            expected_canonical_waves,
            canonical_info.blocker,
            canonical_info.failed_requirements,
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=canonical-prefill-chunk prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} expert_batches={} expert_rows={} graph_captures={} captured_graphs={}",
            worker_index,
            canonical_prompt_tokens,
            elapsed_ms(canonical_start),
            elapsed_ms(start),
            canonical_info.sparse_expert_batches,
            canonical_info.request_expert_batch_rows,
            canonical_info.request_coordinator_graph_captures,
            canonical_info.request_coordinator_graph_captured_graphs,
        );
        finish_sequence(&canonical_sequence_id)
            .map_err(anyhow::Error::msg)
            .context("finishing the canonical prefill-chunk startup sequence")?;
    }
    for (prompt_index, (prompt, prompt_tokens)) in prompts.iter().enumerate() {
        // The original prewarm ran every bucket twice: the first request grew
        // workspaces and the second recaptured graphs against the stable
        // pointers. The canonical-arena sweep below now performs that final
        // capture after every other startup sizing operation, so retaining the
        // historical middle traversal only creates graphs that are replaced.
        // In the ordinary dSpark path, bind every sizing request to the same
        // max-context arena used by production. With no intervening prefix or
        // native-MTP probes, these exact packed-KV identities are already the
        // final serving set and the historical verification sweep is pure
        // replay. Optional probes keep the conservative two-stage layout.
        let stage = "workspace-sizing";
        let stage_start = Instant::now();
        let canonical_sizing_request = canonical_workspace_complete || prompt_index == 0;
        let startup_radix_publish_tokens = (canonical_workspace_complete
            && prompt_index == 0
            && !dsa_selector_query_rows.is_empty())
        .then_some(prompt_tokens.saturating_sub(1));
        let sequence_id = if let Some(publish_tokens) = startup_radix_publish_tokens {
            format!(
                "{REAL_FULL_STARTUP_TARGET_RADIX_PUBLISH_PREFIX}{publish_tokens}-sequence-{worker_index}"
            )
        } else if canonical_sizing_request {
            format!("real-full-startup-capture-arena-{prompt_tokens}-sequence-{worker_index}")
        } else {
            format!("real-full-startup-prewarm-{stage}-{prompt_tokens}-sequence-{worker_index}")
        };
        let recurrent_candidate = if canonical_workspace_complete {
            prompt_index + 1 == prompts.len()
        } else {
            prompt_index == 0
        };
        let decode_budget = if recurrent_candidate {
            REAL_FULL_SERVE_PREWARM_DECODE_BUDGET
        } else {
            1
        };
        if canonical_workspace_complete
            && !dsa_selector_query_rows.is_empty()
            && *prompt_tokens == REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS
        {
            let cached_prompt_tokens = prompt_tokens
                .checked_sub(REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS + 1)
                .context("no-selector DSA boundary cached prefix underflow")?;
            anyhow::ensure!(
                cached_prompt_tokens % REAL_FULL_SHARED_KV_PAGE_TOKENS == 0,
                "no-selector DSA boundary prefix {cached_prompt_tokens} is not page aligned"
            );
            // Branch from the long alpha radix seed at exactly the requested
            // cached frontier. Leaving the whole prompt as alpha would match
            // all 2,048 reusable tokens, while declaring a cached prefix on a
            // fresh sequence would bypass radix binding and leave its
            // processed-token frontier at zero.
            let boundary_prompt = format!(
                "{}{}",
                REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(cached_prompt_tokens),
                REAL_FULL_SERVE_PREWARM_BOUNDARY_TOKEN
                    .repeat(REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS),
            );
            let sequence_id = format!(
                "real-full-startup-dsa-selector-seed-no-selector-boundary-{prompt_tokens}-sequence-{worker_index}"
            );
            let mut request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                0,
                &sequence_id,
                &boundary_prompt,
                *prompt_tokens,
                1,
                Vec::new(),
                0,
                1,
            );
            request.disable_speculation = true;
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=workspace-sizing-cached-boundary prompt_tokens={} cached_prompt_tokens={} query_rows={}",
                worker_index,
                prompt_tokens,
                cached_prompt_tokens,
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS,
            );
            let cycle = execute(request)
                .map_err(anyhow::Error::msg)
                .context("capturing the cached no-selector DSA boundary")?;
            let info = cycle.info;
            anyhow::ensure!(
                info.status == "ready"
                    && info.request_prefill_tokens
                        == REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS
                    && info.request_prefill_chunks == 1,
                "cached no-selector DSA boundary failed: status={} prefill_rows={} chunks={} expected_rows={} blocker={} failed={:?}",
                info.status,
                info.request_prefill_tokens,
                info.request_prefill_chunks,
                REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS,
                info.blocker,
                info.failed_requirements,
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=workspace-sizing-cached-boundary prompt_tokens={} cached_prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} expert_batches={} expert_rows={} graph_captures={} captured_graphs={}",
                worker_index,
                prompt_tokens,
                cached_prompt_tokens,
                elapsed_ms(stage_start),
                elapsed_ms(start),
                info.sparse_expert_batches,
                info.request_expert_batch_rows,
                info.request_coordinator_graph_captures,
                info.request_coordinator_graph_captured_graphs,
            );
            finish_sequence(&sequence_id)
                .map_err(anyhow::Error::msg)
                .context("finishing the cached no-selector DSA boundary sequence")?;
            continue;
        }
        let request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            0,
            &sequence_id,
            prompt,
            *prompt_tokens,
            1,
            Vec::new(),
            0,
            decode_budget,
        );
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage={} prompt_tokens={} max_tokens=1 decode_budget={}",
            worker_index, stage, prompt_tokens, decode_budget
        );
        let cycle = execute(request)
            .map_err(|err| anyhow::anyhow!(err))
            .with_context(|| {
                format!("executing real-full serving startup {stage} prewarm request")
            })?;
        let info = cycle.info;
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage={} prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} status={} sample_status={} sampled_token_id={:?} full_context_device_attention_complete={} sparse_dispatch_status={} prefill_tokens={} prefill_chunks={} expert_batches={} expert_rows={} graph_captures={} graph_launches={} captured_graphs={}",
            worker_index, stage, prompt_tokens, elapsed_ms(stage_start), elapsed_ms(start),
            info.status,
            info.scheduler_terminal_lm_head_sample_status,
            info.scheduler_terminal_lm_head_sampled_token_id,
            info.scheduler_full_context_device_attention_complete,
            info.scheduler_sparse_tcp_dispatch_status,
            info.request_prefill_tokens,
            info.request_prefill_chunks,
            info.sparse_expert_batches,
            info.request_expert_batch_rows,
            info.request_coordinator_graph_captures,
            info.request_coordinator_graph_launches,
            info.request_coordinator_graph_captured_graphs,
        );
        if info.status != "ready" {
            anyhow::bail!(
                "real-full serving startup {stage} prewarm did not produce a ready scheduler sample: status={} sample_status={} blocker={} failed={:?}",
                info.status,
                info.scheduler_terminal_lm_head_sample_status,
                info.blocker,
                info.failed_requirements
            );
        }
        if startup_radix_publish_tokens.is_some() {
            finish_sequence(&sequence_id)
                .map_err(anyhow::Error::msg)
                .context("publishing long-context startup target KV radix seed")?;
        }
        if recurrent_candidate {
            anyhow::ensure!(
                startup_radix_publish_tokens.is_none(),
                "startup recurrent seed cannot also publish the long-context target radix seed"
            );
            recurrent_seed = Some((sequence_id, prompt, *prompt_tokens, info));
        }
    }
    let (sequence_id, prompt, prompt_tokens, info) = recurrent_seed
        .context("real-full serving startup recurrent seed prewarm was not executed")?;
    let sampled_token_id = info
        .scheduler_terminal_lm_head_sampled_token_id
        .context("real-full serving startup recurrent seed produced no sampled token")?;
    let recurrent_start = Instant::now();
    let recurrent = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
        u64::try_from(prompts.len()).context("recurrent prewarm prompt count exceeds u64")?,
        &sequence_id,
        prompt,
        prompt_tokens,
        1,
        vec![sampled_token_id],
        1,
        REAL_FULL_SERVE_PREWARM_DECODE_BUDGET,
    );
    eprintln!(
        "real_full_startup_prewarm_start worker={} stage=recurrent prompt_tokens={} generated_tokens=1 max_tokens=1 decode_budget={}",
        worker_index, prompt_tokens,
        REAL_FULL_SERVE_PREWARM_DECODE_BUDGET
    );
    let recurrent_info = execute(recurrent)
        .map_err(|err| anyhow::anyhow!(err))
        .context("executing real-full serving startup recurrent prewarm request")?
        .info;
    eprintln!(
        "real_full_startup_prewarm_step_done worker={} stage=recurrent elapsed_ms={:.3} total_ms={:.3} status={} sample_status={} sampled_token_id={:?} full_context_device_attention_complete={} sparse_dispatch_status={}",
        worker_index, elapsed_ms(recurrent_start),
        elapsed_ms(start),
        recurrent_info.status,
        recurrent_info.scheduler_terminal_lm_head_sample_status,
        recurrent_info.scheduler_terminal_lm_head_sampled_token_id,
        recurrent_info.scheduler_full_context_device_attention_complete,
        recurrent_info.scheduler_sparse_tcp_dispatch_status
    );
    if recurrent_info.status != "ready" {
        anyhow::bail!(
            "real-full serving startup recurrent prewarm did not produce a ready scheduler sample: status={} sample_status={} blocker={} failed={:?}",
            recurrent_info.status,
            recurrent_info.scheduler_terminal_lm_head_sample_status,
            recurrent_info.blocker,
            recurrent_info.failed_requirements
        );
    }
    if let Some(probe) = prefix_prefill_probe {
        let worker_request_stride = probe
            .cases
            .len()
            .checked_mul(MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS)
            .context("real-full serving prefix-prefill worker request stride overflow")?;
        let worker_request_base = worker_index
            .checked_mul(worker_request_stride)
            .context("real-full serving prefix-prefill worker request base overflow")?;
        for (case_index, case) in probe.cases.iter().enumerate() {
            let seed_decode_budget = case
                .new_prompt_rows
                .checked_add(1)
                .context("real-full serving prefix-prefill probe decode budget overflow")?;
            let suffix_prompt = REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(case.new_prompt_rows);
            let full_prompt = format!("{}{suffix_prompt}", case.prefix_prompt);
            let full_prompt_tokens = case
                .prefix_prompt_tokens
                .checked_add(case.new_prompt_rows)
                .context("real-full serving prefix-prefill probe token count overflow")?;
            let case_request_base = case_index
                .checked_mul(MAX_REAL_FULL_SERVE_PREFIX_PREFILL_PROBE_REPEATS)
                .and_then(|offset| worker_request_base.checked_add(offset))
                .context("real-full serving prefix-prefill case request base overflow")?;
            let mut probe_elapsed_samples = Vec::with_capacity(probe.repeats);
            for repeat_index in 0..probe.repeats {
                let request_offset = case_request_base
                    .checked_add(repeat_index)
                    .context("real-full serving prefix-prefill request offset overflow")?;
                let request_offset = u64::try_from(request_offset)
                    .context("real-full serving prefix-prefill request offset exceeds u64")?;
                let probe_sequence_id = format!(
                    "real-full-startup-prefix-prefill-seed-{}-{}-repeat-{repeat_index}-sequence-{worker_index}",
                    case.prefix_prompt_tokens, case.new_prompt_rows
                );
                let seed_start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start worker={} stage=prefix-prefill-seed repeat={} prefix_tokens={} new_prompt_tokens={} decode_budget={}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    case.new_prompt_rows,
                    seed_decode_budget
                );
                let seed_info = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                    20_000_u64
                        .checked_add(request_offset)
                        .context("real-full serving prefix-prefill seed request index overflow")?,
                    &probe_sequence_id,
                    &case.prefix_prompt,
                    case.prefix_prompt_tokens,
                    1,
                    Vec::new(),
                    0,
                    seed_decode_budget,
                ))
                .map_err(|err| anyhow::anyhow!(err))
                .with_context(|| {
                    format!("executing real-full serving prefix-prefill seed {repeat_index}")
                })?
                .info;
                anyhow::ensure!(
                    seed_info.status == "ready",
                    "real-full serving prefix-prefill seed {repeat_index} was not ready: status={} blocker={} failed={:?}",
                    seed_info.status,
                    seed_info.blocker,
                    seed_info.failed_requirements
                );
                eprintln!(
                    "real_full_startup_prewarm_step_done worker={} stage=prefix-prefill-seed repeat={} prefix_tokens={} elapsed_ms={:.3} total_ms={:.3}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    elapsed_ms(seed_start),
                    elapsed_ms(start)
                );
                let request_index = 10_000_u64
                    .checked_add(request_offset)
                    .context("real-full serving prefix-prefill probe request index overflow")?;
                let probe_start = Instant::now();
                eprintln!(
                    "real_full_startup_prewarm_start worker={} stage=prefix-prefill repeat={} prefix_tokens={} new_prompt_tokens={} full_prompt_tokens={}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    case.new_prompt_rows,
                    full_prompt_tokens
                );
                let probe_info = execute(
                    glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                        request_index,
                        &probe_sequence_id,
                        &full_prompt,
                        full_prompt_tokens,
                        1,
                        Vec::new(),
                        1,
                        REAL_FULL_SERVE_PREWARM_DECODE_BUDGET,
                    )
                    .with_cached_prompt_tokens(case.prefix_prompt_tokens),
                )
                .map_err(|err| anyhow::anyhow!(err))
                .with_context(|| {
                    format!("executing real-full serving prefix-prefill probe {repeat_index}")
                })?
                .info;
                let probe_elapsed_ms = elapsed_ms(probe_start);
                anyhow::ensure!(
                    probe_info.status == "ready",
                    "real-full serving prefix-prefill probe {repeat_index} was not ready: status={} sample_status={} blocker={} failed={:?}",
                    probe_info.status,
                    probe_info.scheduler_terminal_lm_head_sample_status,
                    probe_info.blocker,
                    probe_info.failed_requirements
                );
                anyhow::ensure!(
                    probe_info.request_prefill_tokens + 1 == case.new_prompt_rows,
                    "real-full serving prefix-prefill probe {repeat_index} reported {} prefill rows for {} new prompt tokens",
                    probe_info.request_prefill_tokens,
                    case.new_prompt_rows
                );
                eprintln!(
                    "real_full_startup_prewarm_step_done worker={} stage=prefix-prefill repeat={} prefix_tokens={} new_prompt_tokens={} prefill_rows={} elapsed_ms={:.3} tokens_per_sec={:.3} total_ms={:.3} status={} sample_status={} sampled_token_id={:?}",
                    worker_index,
                    repeat_index,
                    case.prefix_prompt_tokens,
                    case.new_prompt_rows,
                    probe_info.request_prefill_tokens,
                    probe_elapsed_ms,
                    case.new_prompt_rows as f64 * 1_000.0 / probe_elapsed_ms,
                    elapsed_ms(start),
                    probe_info.status,
                    probe_info.scheduler_terminal_lm_head_sample_status,
                    probe_info.scheduler_terminal_lm_head_sampled_token_id,
                );
                probe_elapsed_samples.push(probe_elapsed_ms);
            }
            probe_elapsed_samples.sort_by(f64::total_cmp);
            let sample_midpoint = probe_elapsed_samples.len() / 2;
            let median_elapsed_ms = if probe_elapsed_samples.len() % 2 == 0 {
                (probe_elapsed_samples[sample_midpoint - 1]
                    + probe_elapsed_samples[sample_midpoint])
                    / 2.0
            } else {
                probe_elapsed_samples[sample_midpoint]
            };
            eprintln!(
                "real_full_startup_prefix_prefill_summary worker={} repeats={} prefix_tokens={} new_prompt_tokens={} median_elapsed_ms={:.3} median_tokens_per_sec={:.3} min_elapsed_ms={:.3} max_elapsed_ms={:.3}",
                worker_index,
                probe_elapsed_samples.len(),
                case.prefix_prompt_tokens,
                case.new_prompt_rows,
                median_elapsed_ms,
                case.new_prompt_rows as f64 * 1_000.0 / median_elapsed_ms,
                probe_elapsed_samples[0],
                probe_elapsed_samples[probe_elapsed_samples.len() - 1]
            );
        }
    }
    if real_full_mtp_enabled() {
        let (mtp_prompt, mtp_prompt_tokens) = prompts.get(1).unwrap_or(&prompts[0]);
        let draft_tokens = real_full_mtp_draft_tokens();
        let mtp_sequence_id = format!(
            "real-full-startup-mtp-production-draft-{draft_tokens}-{mtp_prompt_tokens}-sequence-{worker_index}"
        );
        let mtp_start = Instant::now();
        // The first verify sizes the batched LM-head scratch after the MLP
        // graphs have run. Keep enough budget to regenerate a full draft set,
        // then verify once more so those graphs are captured against the final
        // scratch layout before startup capture closes.
        let capture_budget = draft_tokens.saturating_mul(2).saturating_add(3);
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=mtp-production-draft prompt_tokens={} draft_tokens={} decode_budget={}",
            worker_index, mtp_prompt_tokens, draft_tokens, capture_budget
        );
        let initial_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            2,
            &mtp_sequence_id,
            mtp_prompt,
            *mtp_prompt_tokens,
            1,
            Vec::new(),
            0,
            capture_budget,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .context("executing real-full serving startup MTP draft prewarm")?;
        anyhow::ensure!(
            initial_cycle.info.status == "ready",
            "real-full serving startup MTP draft prewarm was not ready: status={} sample_status={} blocker={} failed={:?}",
            initial_cycle.info.status,
            initial_cycle.info.scheduler_terminal_lm_head_sample_status,
            initial_cycle.info.blocker,
            initial_cycle.info.failed_requirements
        );
        let initial_token_id = initial_cycle
            .generated_tokens
            .first()
            .map(|token| token.token_id)
            .or(initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id)
            .context("real-full serving startup MTP draft prewarm produced no token")?;
        let verify_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            3,
            &mtp_sequence_id,
            mtp_prompt,
            *mtp_prompt_tokens,
            1,
            vec![initial_token_id],
            1,
            capture_budget,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .context("executing real-full serving startup MTP target verification prewarm")?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "real-full serving startup MTP verify prewarm failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            draft_tokens,
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=mtp-production-verify prompt_tokens={} drafts={} accepted={} emitted={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            mtp_prompt_tokens,
            draft_tokens,
            verify_cycle.info.request_mtp_accepted_rows,
            verify_cycle.generated_tokens.len(),
            elapsed_ms(mtp_start),
            elapsed_ms(start),
        );
        let mut settled_generated_token_ids = vec![initial_token_id];
        settled_generated_token_ids.extend(
            verify_cycle
                .generated_tokens
                .iter()
                .map(|token| token.token_id),
        );
        let settled_verify_cycle = execute(
            glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                4,
                &mtp_sequence_id,
                mtp_prompt,
                *mtp_prompt_tokens,
                1,
                settled_generated_token_ids.clone(),
                settled_generated_token_ids.len(),
                capture_budget,
            ),
        )
        .map_err(|err| anyhow::anyhow!(err))
        .context("executing settled real-full serving startup MTP target verification prewarm")?;
        anyhow::ensure!(
            settled_verify_cycle.info.status == "ready"
                && settled_verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "settled real-full serving startup MTP verify prewarm failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
            settled_verify_cycle.info.status,
            settled_verify_cycle.info.request_mtp_verify_rows,
            draft_tokens,
            settled_verify_cycle.info.blocker,
            settled_verify_cycle.info.failed_requirements
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=mtp-production-settled-verify prompt_tokens={} drafts={} accepted={} emitted={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            mtp_prompt_tokens,
            draft_tokens,
            settled_verify_cycle.info.request_mtp_accepted_rows,
            settled_verify_cycle.generated_tokens.len(),
            elapsed_ms(mtp_start),
            elapsed_ms(start),
        );
        prewarm_real_full_mtp_adaptive_draft_widths(
            &mut execute,
            mtp_prompt,
            *mtp_prompt_tokens,
            worker_index,
            start,
        )?;
        prewarm_real_full_mtp_production_buckets(&mut execute, prompts, worker_index, start)?;
    }
    if real_full_mtp_probe_enabled() {
        for (mtp_prompt, mtp_prompt_tokens) in prompts {
            let mtp_sequence_id = format!(
                "real-full-startup-mtp-capture-{mtp_prompt_tokens}-sequence-{worker_index}"
            );
            let mtp_start = Instant::now();
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=mtp-capture prompt_tokens={} max_tokens=1 decode_budget=1",
                worker_index, mtp_prompt_tokens
            );
            let mtp_info = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                0,
                &mtp_sequence_id,
                mtp_prompt,
                *mtp_prompt_tokens,
                1,
                Vec::new(),
                0,
                1,
            ))
            .map_err(|err| anyhow::anyhow!(err))
            .with_context(|| {
                format!(
                    "executing real-full serving startup MTP graph-capture request for {mtp_prompt_tokens} prompt tokens"
                )
            })?
            .info;
            anyhow::ensure!(
                mtp_info.status == "ready",
                "real-full serving startup MTP graph-capture request for {mtp_prompt_tokens} prompt tokens was not ready: status={} sample_status={} blocker={} failed={:?}",
                mtp_info.status,
                mtp_info.scheduler_terminal_lm_head_sample_status,
                mtp_info.blocker,
                mtp_info.failed_requirements
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=mtp-capture prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
                worker_index,
                mtp_prompt_tokens,
                elapsed_ms(mtp_start),
                elapsed_ms(start),
                mtp_info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }

        let (validation_prompt, validation_prompt_tokens) = prompts.get(1).unwrap_or(&prompts[0]);
        let validation_sequence_id = format!(
            "real-full-startup-prewarm-post-mtp-{validation_prompt_tokens}-sequence-{worker_index}"
        );
        let validation_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=post-mtp-base-validation prompt_tokens={} max_tokens=1 decode_budget=1",
            worker_index, validation_prompt_tokens
        );
        let validation_info = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            0,
            &validation_sequence_id,
            validation_prompt,
            *validation_prompt_tokens,
            1,
            Vec::new(),
            0,
            1,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .context("executing real-full serving post-MTP base graph validation request")?
        .info;
        anyhow::ensure!(
            validation_info.status == "ready",
            "real-full serving post-MTP base graph validation was not ready: status={} sample_status={} blocker={} failed={:?}",
            validation_info.status,
            validation_info.scheduler_terminal_lm_head_sample_status,
            validation_info.blocker,
            validation_info.failed_requirements
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=post-mtp-base-validation prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
            worker_index,
            validation_prompt_tokens,
            elapsed_ms(validation_start),
            elapsed_ms(start),
            validation_info.scheduler_terminal_lm_head_sampled_token_id,
        );
    }
    // Every direct packed-KV attention graph captures the physical cache and
    // query-arena addresses. Ordinary dSpark sizing already uses the canonical
    // max-context arena and no later optional probe can replace those C=1
    // identities, so a second sweep would only replay the same graphs. Prefix
    // and native-MTP probe modes can still perturb the capture set; retain the
    // conservative final sweep for them and run its largest prompt last. The
    // selector seed is created separately after dSpark width prewarm because
    // those graph-bound requests intentionally recycle the one max-context
    // arena; retaining a seed here would let width prewarm rebind its state
    // while the selector sweep still believed the old sequence was active.
    if canonical_workspace_complete {
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=canonical-capture-skip reason=canonical-workspace-complete elapsed_ms=0.000 total_ms={:.3}",
            worker_index,
            elapsed_ms(start),
        );
    } else {
        for (prompt_index, (prompt, prompt_tokens)) in prompts.iter().rev().enumerate() {
            let sequence_id = format!(
                "real-full-startup-capture-arena-final-{prompt_tokens}-sequence-{worker_index}"
            );
            let decode_budget = 1;
            let capture_start = Instant::now();
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=canonical-capture prompt_tokens={} max_tokens=1 decode_budget={}",
                worker_index, prompt_tokens, decode_budget
            );
            let capture_info = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                u64::try_from(prompt_index)
                    .context("canonical capture prompt index exceeds u64")?,
                &sequence_id,
                prompt,
                *prompt_tokens,
                1,
                Vec::new(),
                0,
                decode_budget,
            ))
            .map_err(|err| anyhow::anyhow!(err))
            .with_context(|| {
                format!("executing canonical max-context capture for {prompt_tokens} prompt tokens")
            })?
            .info;
            anyhow::ensure!(
                capture_info.status == "ready",
                "canonical max-context capture for {prompt_tokens} prompt tokens was not ready: status={} sample_status={} blocker={} failed={:?}",
                capture_info.status,
                capture_info.scheduler_terminal_lm_head_sample_status,
                capture_info.blocker,
                capture_info.failed_requirements
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=canonical-capture prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
                worker_index,
                prompt_tokens,
                elapsed_ms(capture_start),
                elapsed_ms(start),
                capture_info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }
    }
    // Grow the shared multirow workspaces before capturing long-context DSA
    // identities. The dSpark M=2..8 sweep can enlarge bucket-8 scratch; if it
    // runs afterward, the pointer change correctly clears the earlier DSA
    // graphs and leaves the first >2K serving request unable to recapture.
    prewarm_real_full_dspark_widths(
        &mut execute,
        &mut finish_sequence,
        prompts,
        worker_index,
        start,
        None,
    )?;
    let dsa_selector_seed = if dsa_selector_query_rows.is_empty() {
        None
    } else {
        let (prompt, prompt_tokens) = if real_full_mtp_enabled() {
            prompts
                .iter()
                .find(|(_, prompt_tokens)| *prompt_tokens <= REAL_FULL_MTP_STARTUP_MAX_REPLAY_ROWS)
                .context("real-full MTP DSA selector prewarm has no replayable prompt")?
        } else {
            prompts
                .first()
                .context("real-full serving DSA selector prewarm prompt set is empty")?
        };
        let sequence_id =
            format!("real-full-startup-dsa-selector-seed-{prompt_tokens}-sequence-{worker_index}");
        let seed_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=dsa-selector-seed prompt_tokens={} decode_budget={}",
            worker_index, prompt_tokens, dsa_selector_decode_budget,
        );
        let mut seed_request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            29_999,
            &sequence_id,
            prompt,
            *prompt_tokens,
            1,
            Vec::new(),
            0,
            dsa_selector_decode_budget,
        );
        // This pass captures target-attention selector buckets. It does not
        // need a draft proposal, and target-only admission lets it reuse the
        // long prefix produced by workspace sizing.
        seed_request.disable_speculation = true;
        let seed_info = execute(seed_request)
            .map_err(|err| anyhow::anyhow!(err))
            .context("executing long-context DSA selector seed")?
            .info;
        anyhow::ensure!(
            seed_info.status == "ready",
            "long-context DSA selector seed was not ready: status={} blocker={} failed={:?}",
            seed_info.status,
            seed_info.blocker,
            seed_info.failed_requirements,
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=dsa-selector-seed prompt_tokens={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
            worker_index,
            prompt_tokens,
            elapsed_ms(seed_start),
            elapsed_ms(start),
            seed_info.scheduler_terminal_lm_head_sampled_token_id,
        );
        Some((sequence_id, prompt.clone(), *prompt_tokens))
    };
    if let Some((sequence_id, mut prompt, mut prompt_tokens)) = dsa_selector_seed {
        for (step_index, query_rows) in dsa_selector_query_rows.iter().copied().enumerate() {
            let cached_prompt_tokens = prompt_tokens;
            let uncached_prompt_rows = query_rows + 1;
            prompt.push_str(&REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(uncached_prompt_rows));
            prompt_tokens = prompt_tokens
                .checked_add(uncached_prompt_rows)
                .context("DSA selector prewarm prompt token count overflow")?;
            let sweep_start = Instant::now();
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=dsa-selector-bucket query_rows={} prefix_tokens={} prompt_tokens={} decode_budget={}",
                worker_index,
                query_rows,
                cached_prompt_tokens,
                prompt_tokens,
                dsa_selector_decode_budget,
            );
            let mut sweep_request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                30_000_u64
                    .checked_add(
                        u64::try_from(step_index)
                            .context("DSA selector prewarm step index exceeds u64")?,
                    )
                    .context("DSA selector prewarm request index overflow")?,
                &sequence_id,
                &prompt,
                prompt_tokens,
                1,
                Vec::new(),
                step_index + 1,
                dsa_selector_decode_budget,
            )
            .with_cached_prompt_tokens(cached_prompt_tokens);
            sweep_request.disable_speculation = true;
            let sweep_info = execute(sweep_request)
                .map_err(|err| anyhow::anyhow!(err))
                .with_context(|| {
                    format!("capturing long-context DSA selector query bucket {query_rows}")
                })?
                .info;
            anyhow::ensure!(
                sweep_info.status == "ready"
                    && sweep_info.request_prefill_tokens == query_rows,
                "long-context DSA selector query bucket {query_rows} was not ready: status={} prefill_rows={} blocker={} failed={:?}",
                sweep_info.status,
                sweep_info.request_prefill_tokens,
                sweep_info.blocker,
                sweep_info.failed_requirements,
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=dsa-selector-bucket query_rows={} elapsed_ms={:.3} total_ms={:.3} sampled_token_id={:?}",
                worker_index,
                query_rows,
                elapsed_ms(sweep_start),
                elapsed_ms(start),
                sweep_info.scheduler_terminal_lm_head_sampled_token_id,
            );
        }
        if canonical_workspace_complete {
            let radix_prefix_tokens = prompts
                .first()
                .map(|(_, prompt_tokens)| prompt_tokens.saturating_sub(1))
                .context("real-full serving radix cleanup prompt set is empty")?;
            let boundary_prefix_tokens = REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_PROMPT_TOKENS
                .checked_sub(REAL_FULL_SERVE_NO_SELECTOR_DSA_BOUNDARY_QUERY_ROWS + 1)
                .context("no-selector DSA boundary cleanup prefix underflow")?;
            // The cached no-selector pass splits the long alpha tree at its
            // 1,536-token branch point. Evict the longest suffix first, then
            // remove the complete synthetic branch subtree. With a larger KV
            // pool the selector sweep remains below that branch instead of
            // being displaced by LRU pressure.
            for eviction_tokens in [radix_prefix_tokens, boundary_prefix_tokens] {
                let eviction_sequence_id = format!(
                    "{REAL_FULL_STARTUP_TARGET_RADIX_EVICT_PREFIX}{eviction_tokens}-worker-{worker_index}"
                );
                finish_sequence(&eviction_sequence_id)
                    .map_err(anyhow::Error::msg)
                    .with_context(|| {
                        format!(
                            "evicting {eviction_tokens}-token synthetic startup target KV radix subtree"
                        )
                    })?;
            }
        }
    }
    // The canonical arena and DSA-selector passes above can grow or rebind
    // shared attention scratch after the first native-MTP bucket sweep. Replay
    // the short-context layer-78 identities only after that scratch layout is
    // final; otherwise the first ordinary semantic prompt is unable to capture
    // its sealed FlashInfer suffix graph at runtime.
    if real_full_mtp_enabled() {
        prewarm_real_full_mtp_production_buckets(&mut execute, prompts, worker_index, start)?;
    }
    // The first width pass and the DSA selector sweep establish the union of
    // required scratch capacities. Native NVFP4 attention additionally keys
    // its graph signature by sparse-attention K bucket and retains exact
    // q2/q4 recurrent buckets. Exercise one width from each physical query
    // bucket across all four short-K buckets plus the >2K selector regime
    // after scratch has reached its final size. FP8 needs only its q8/q16
    // identities in the selector regime. The first width pass already
    // captures all other exact-width kernels.
    if real_full_draft_runtime_enabled() {
        let required_attention_prompt_tokens = if kv_dtype == KvCacheDType::Nvfp4 {
            &[9, 145, 513, 1_025, 2_049][..]
        } else {
            &[2_049][..]
        };
        let attention_prompts = required_attention_prompt_tokens
            .into_iter()
            .map(|required_prompt_tokens| {
                prompts
                    .iter()
                    .find(|(_, prompt_tokens)| prompt_tokens == required_prompt_tokens)
                    .cloned()
                    .with_context(|| {
                        format!(
                            "real-full serving dSpark prewarm has no canonical {required_prompt_tokens}-token prompt"
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let checkpoint_max_drafts = real_full_active_max_verify_drafts();
        let maximum_drafts = match real_full_active_fixed_drafts()? {
            Some(drafts) => {
                anyhow::ensure!(
                    drafts <= checkpoint_max_drafts,
                    "fixed dSpark width {drafts} exceeds the active checkpoint maximum {checkpoint_max_drafts}"
                );
                drafts
            }
            None => checkpoint_max_drafts,
        };
        let mut attention_query_bucket_drafts = Vec::new();
        let query_bucket_representatives = if kv_dtype == KvCacheDType::Nvfp4 {
            &[maximum_drafts, 7, 3, 1, 0][..]
        } else {
            &[maximum_drafts, 7, 0][..]
        };
        for representative_drafts in query_bucket_representatives.iter().copied() {
            if representative_drafts <= maximum_drafts
                && !attention_query_bucket_drafts.contains(&representative_drafts)
            {
                attention_query_bucket_drafts.push(representative_drafts);
            }
        }
        for attention_prompt in &attention_prompts {
            prewarm_real_full_dspark_widths(
                &mut execute,
                &mut finish_sequence,
                std::slice::from_ref(attention_prompt),
                worker_index,
                start,
                Some(&attention_query_bucket_drafts),
            )?;
        }
    }
    if kv_dtype == KvCacheDType::Nvfp4 {
        // Keep this last within the request sweep so every graph captured
        // here sees the maximum shared-workspace capacities. The outer
        // same-width replays must then preserve these identities; a
        // post-outer registry audit below enforces that invariant.
        prewarm_real_full_nvfp4_short_k_prefill_graphs(
            &mut execute,
            &mut finish_sequence,
            prompts,
            worker_index,
            start,
        )?;
    }
    eprintln!(
        "real_full_startup_prewarm_done worker={} elapsed_ms={:.3} initial_sampled_token_id={:?} recurrent_sampled_token_id={:?}",
        worker_index, elapsed_ms(start),
        info.scheduler_terminal_lm_head_sampled_token_id,
        recurrent_info.scheduler_terminal_lm_head_sampled_token_id
    );
    Ok(())
}

fn prewarm_real_full_nvfp4_short_k_prefill_graphs(
    execute: &mut impl FnMut(
        glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String>,
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    prompts: &[(String, usize)],
    worker_index: usize,
    prewarm_start: Instant,
) -> Result<()> {
    // The seed owns one persistent scheduler state while every bucket appends
    // its query rows plus the next sampled row to the prompt. Reserve the full
    // cumulative sweep here; the ordinary small-prompt extension headroom is
    // intentionally too small for the 9-token K=128 anchor.
    let decode_budget = real_full_nvfp4_short_k_prefill_decode_budget()?;
    for (anchor_index, anchor) in REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_CAPTURE_ANCHORS
        .iter()
        .copied()
        .enumerate()
    {
        let (seed_prompt, seed_prompt_tokens) = prompts
            .iter()
            .find(|(_, prompt_tokens)| *prompt_tokens == anchor.prompt_tokens)
            .with_context(|| {
                format!(
                    "NVFP4 short-K prefill capture has no canonical {}-token prompt",
                    anchor.prompt_tokens
                )
            })?;
        let sequence_id = format!(
            "real-full-startup-dsa-selector-seed-short-k-{}-{}-sequence-{worker_index}",
            anchor.sparse_topk, anchor.prompt_tokens,
        );
        let anchor_request_base = 50_000_u64
            .checked_add(
                u64::try_from(anchor_index)
                    .context("NVFP4 short-K capture anchor index exceeds u64")?
                    .saturating_mul(100),
            )
            .context("NVFP4 short-K capture request base overflow")?;
        let seed_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=nvfp4-short-k-seed sparse_topk={} prompt_tokens={} decode_budget={}",
            worker_index, anchor.sparse_topk, seed_prompt_tokens, decode_budget,
        );
        let seed_info = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            anchor_request_base,
            &sequence_id,
            seed_prompt,
            *seed_prompt_tokens,
            1,
            Vec::new(),
            0,
            decode_budget,
        ))
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "seeding NVFP4 sparse-K={} prefill graph capture",
                anchor.sparse_topk
            )
        })?
        .info;
        anyhow::ensure!(
            seed_info.status == "ready",
            "NVFP4 sparse-K={} prefill capture seed was not ready: status={} blocker={} failed={:?}",
            anchor.sparse_topk,
            seed_info.status,
            seed_info.blocker,
            seed_info.failed_requirements,
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=nvfp4-short-k-seed sparse_topk={} prompt_tokens={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            anchor.sparse_topk,
            seed_prompt_tokens,
            elapsed_ms(seed_start),
            elapsed_ms(prewarm_start),
        );

        let mut prompt = seed_prompt.clone();
        let mut prompt_tokens = *seed_prompt_tokens;
        for (step_index, query_rows) in REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS
            .iter()
            .copied()
            .enumerate()
        {
            let cached_prompt_tokens = prompt_tokens;
            let uncached_prompt_rows = query_rows + 1;
            prompt.push_str(&REAL_FULL_SERVE_PREWARM_PROMPT_TOKEN.repeat(uncached_prompt_rows));
            prompt_tokens = prompt_tokens
                .checked_add(uncached_prompt_rows)
                .context("NVFP4 short-K capture prompt token count overflow")?;
            let sweep_start = Instant::now();
            eprintln!(
                "real_full_startup_prewarm_start worker={} stage=nvfp4-short-k-bucket sparse_topk={} query_rows={} prefix_tokens={} prompt_tokens={} decode_budget={}",
                worker_index,
                anchor.sparse_topk,
                query_rows,
                cached_prompt_tokens,
                prompt_tokens,
                decode_budget,
            );
            let sweep_info = execute(
                glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                    anchor_request_base
                        .checked_add(
                            u64::try_from(step_index)
                                .context("NVFP4 short-K capture step index exceeds u64")?
                                + 1,
                        )
                        .context("NVFP4 short-K capture request index overflow")?,
                    &sequence_id,
                    &prompt,
                    prompt_tokens,
                    1,
                    Vec::new(),
                    step_index + 1,
                    decode_budget,
                )
                .with_cached_prompt_tokens(cached_prompt_tokens),
            )
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "capturing NVFP4 sparse-K={} query bucket {query_rows}",
                    anchor.sparse_topk
                )
            })?
            .info;
            anyhow::ensure!(
                sweep_info.status == "ready"
                    && sweep_info.request_prefill_tokens == query_rows,
                "NVFP4 sparse-K={} query bucket {query_rows} was not ready: status={} prefill_rows={} blocker={} failed={:?}",
                anchor.sparse_topk,
                sweep_info.status,
                sweep_info.request_prefill_tokens,
                sweep_info.blocker,
                sweep_info.failed_requirements,
            );
            eprintln!(
                "real_full_startup_prewarm_step_done worker={} stage=nvfp4-short-k-bucket sparse_topk={} query_rows={} elapsed_ms={:.3} total_ms={:.3}",
                worker_index,
                anchor.sparse_topk,
                query_rows,
                elapsed_ms(sweep_start),
                elapsed_ms(prewarm_start),
            );
        }
        finish_sequence(&sequence_id)
            .map_err(anyhow::Error::msg)
            .with_context(|| {
                format!(
                    "finishing NVFP4 sparse-K={} startup capture sequence",
                    anchor.sparse_topk
                )
            })?;
    }
    Ok(())
}

fn audit_real_full_nvfp4_short_k_prefill_graphs(
    mut finish_sequence: impl FnMut(&str) -> std::result::Result<(), String>,
    worker_index: usize,
) -> Result<()> {
    for anchor in REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_CAPTURE_ANCHORS {
        for query_rows in REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS {
            let sequence_id = format!(
                "{REAL_FULL_STARTUP_AUDIT_NVFP4_SHORT_K_PREFIX}{query_rows}-k{}-worker-{worker_index}",
                anchor.sparse_topk,
            );
            finish_sequence(&sequence_id)
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!(
                        "auditing retained NVFP4 sparse-K={} query bucket {query_rows} after outer startup recaptures",
                        anchor.sparse_topk,
                    )
                })?;
        }
    }
    Ok(())
}

fn finish_real_full_dspark_width_prewarm_sequence(
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    sequence_id: &str,
    physical_m: usize,
) -> Result<()> {
    finish_sequence(sequence_id)
        .map_err(anyhow::Error::msg)
        .with_context(|| format!("finishing dSpark M={physical_m} startup capture sequence"))
}

#[allow(clippy::too_many_arguments)]
fn prewarm_real_full_dspark_width_cohort(
    execute: &mut impl FnMut(
        glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String>,
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    dspark_prompt: &str,
    dspark_prompt_tokens: usize,
    draft_widths: &[usize],
    worker_index: usize,
    prewarm_start: Instant,
) -> Result<()> {
    let maximum_drafts = draft_widths
        .iter()
        .copied()
        .max()
        .context("dSpark scalar width cohort is empty")?;
    let sequence_id = format!(
        "real-full-startup-dspark-width-{maximum_drafts}{REAL_FULL_STARTUP_SCALAR_DSPARK_COHORT_MARKER}{dspark_prompt_tokens}-sequence-{worker_index}"
    );
    let capture_budget = draft_widths
        .iter()
        .try_fold(2_usize, |budget, draft_tokens| {
            budget
                .checked_add(draft_tokens.saturating_add(1))
                .context("dSpark scalar width cohort decode budget overflow")
        })?;
    let cohort_start = Instant::now();
    eprintln!(
        "real_full_startup_prewarm_start worker={} stage=dspark-width-cohort prompt_tokens={} widths={:?} max_physical_m={} decode_budget={}",
        worker_index,
        dspark_prompt_tokens,
        draft_widths,
        maximum_drafts + 1,
        capture_budget,
    );
    let initial_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
        REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE - 1,
        &sequence_id,
        dspark_prompt,
        dspark_prompt_tokens,
        1,
        Vec::new(),
        0,
        capture_budget,
    ))
    .map_err(anyhow::Error::msg)
    .context("seeding a reusable dSpark scalar width cohort")?;
    anyhow::ensure!(
        initial_cycle.info.status == "ready",
        "dSpark scalar width cohort seed failed: status={} blocker={} failed={:?}",
        initial_cycle.info.status,
        initial_cycle.info.blocker,
        initial_cycle.info.failed_requirements,
    );
    let mut generated_token_ids = initial_cycle
        .generated_tokens
        .iter()
        .map(|token| token.token_id)
        .collect::<Vec<_>>();
    if generated_token_ids.is_empty() {
        generated_token_ids.push(
            initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id
                .context("dSpark scalar width cohort seed produced no token")?,
        );
    }
    for (width_index, draft_tokens) in draft_widths.iter().copied().enumerate() {
        let width_start = Instant::now();
        let request_index = REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_BASE
            .checked_add(
                u64::try_from(draft_tokens)
                    .context("dSpark scalar cohort width exceeds u64")?
                    .saturating_mul(REAL_FULL_SCALAR_DSPARK_PREWARM_WIDTH_REQUEST_STRIDE),
            )
            .and_then(|index| index.checked_add(u64::try_from(width_index).ok()?))
            .context("dSpark scalar cohort request index overflow")?;
        let decode_step_index = generated_token_ids.len();
        let verify_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index,
            &sequence_id,
            dspark_prompt,
            dspark_prompt_tokens,
            1,
            generated_token_ids.clone(),
            decode_step_index,
            capture_budget,
        ))
        .map_err(anyhow::Error::msg)
        .with_context(|| {
            format!(
                "executing dSpark M={} scalar cohort capture",
                draft_tokens + 1
            )
        })?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "dSpark M={} scalar cohort capture failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
            draft_tokens + 1,
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            draft_tokens,
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements,
        );
        anyhow::ensure!(
            !verify_cycle.generated_tokens.is_empty(),
            "dSpark M={} scalar cohort capture emitted no continuation token",
            draft_tokens + 1,
        );
        generated_token_ids.extend(
            verify_cycle
                .generated_tokens
                .iter()
                .map(|token| token.token_id),
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=dspark-width-cohort prompt_tokens={} drafts={} physical_m={} verify_rows={} generated_tokens={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            dspark_prompt_tokens,
            draft_tokens,
            draft_tokens + 1,
            verify_cycle.info.request_mtp_verify_rows,
            generated_token_ids.len(),
            elapsed_ms(width_start),
            elapsed_ms(prewarm_start),
        );
    }
    finish_real_full_dspark_width_prewarm_sequence(
        finish_sequence,
        &sequence_id,
        maximum_drafts + 1,
    )?;
    eprintln!(
        "real_full_startup_prewarm_done worker={} stage=dspark-width-cohort prompt_tokens={} widths={:?} elapsed_ms={:.3} total_ms={:.3}",
        worker_index,
        dspark_prompt_tokens,
        draft_widths,
        elapsed_ms(cohort_start),
        elapsed_ms(prewarm_start),
    );
    Ok(())
}

fn prewarm_real_full_dspark_widths(
    execute: &mut impl FnMut(
        glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String>,
    finish_sequence: &mut impl FnMut(&str) -> std::result::Result<(), String>,
    prompts: &[(String, usize)],
    worker_index: usize,
    prewarm_start: Instant,
    draft_widths_override: Option<&[usize]>,
) -> Result<()> {
    if !real_full_draft_runtime_enabled() {
        return Ok(());
    }

    // Width identities are keyed by physical rows/layer, not context length.
    // Use the smallest canonical prompt after the base canonical sweep has
    // established the serving arena. Long-context DSA capture runs afterward.
    let (dspark_prompt, dspark_prompt_tokens) = prompts
        .last()
        .context("real-full serving dSpark prewarm prompt set is empty")?;
    // Fixed-width diagnostics can only reach their configured M, so avoid
    // spending every coordinator restart recapturing unused wider widths.
    // Adaptive serving still captures widest-first: several coordinator
    // scratch buffers grow with M, and growing them invalidates every graph
    // identity in the shared slot. Descending widths establish maximum
    // capacity first, then retain all narrower DSA and attention identities.
    let draft_widths = match draft_widths_override {
        Some(draft_widths) => draft_widths.to_vec(),
        None => match real_full_active_fixed_drafts()? {
            // The final cycle truncates to the remaining output budget, so a
            // fixed-width request can still reach every narrower M. Include
            // D=0/M=1 explicitly: the adaptive policy can choose target-only
            // decode, and that decode graph must exist before capture closes.
            Some(draft_tokens) => (0..=draft_tokens).rev().collect(),
            None => (0..=real_full_active_max_verify_drafts()).rev().collect(),
        },
    };
    if draft_widths_override.is_some() && draft_widths.len() > 1 {
        return prewarm_real_full_dspark_width_cohort(
            execute,
            finish_sequence,
            dspark_prompt,
            *dspark_prompt_tokens,
            &draft_widths,
            worker_index,
            prewarm_start,
        );
    }
    for draft_tokens in draft_widths {
        let request_index = 30_000 + draft_tokens as u64 * 3;
        let capture_budget = real_full_active_max_verify_drafts() + 4;
        let sequence_id = format!(
            "real-full-startup-dspark-width-{draft_tokens}-{dspark_prompt_tokens}-sequence-{worker_index}"
        );
        let width_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=dspark-width prompt_tokens={} drafts={} physical_m={}",
            worker_index,
            dspark_prompt_tokens,
            draft_tokens,
            draft_tokens + 1,
        );
        let initial_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index,
            &sequence_id,
            dspark_prompt,
            *dspark_prompt_tokens,
            1,
            Vec::new(),
            0,
            capture_budget,
        ))
        .map_err(|error| anyhow::anyhow!(error))
        .with_context(|| {
            format!(
                "executing dSpark M={} scalar capture seed",
                draft_tokens + 1
            )
        })?;
        anyhow::ensure!(
            initial_cycle.info.status == "ready",
            "dSpark M={} scalar capture seed failed: status={} blocker={} failed={:?}",
            draft_tokens + 1,
            initial_cycle.info.status,
            initial_cycle.info.blocker,
            initial_cycle.info.failed_requirements,
        );
        let initial_token_id = initial_cycle
            .generated_tokens
            .first()
            .map(|token| token.token_id)
            .or(initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id)
            .with_context(|| {
                format!(
                    "dSpark M={} scalar capture seed has no token",
                    draft_tokens + 1
                )
            })?;
        let verify_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index + 1,
            &sequence_id,
            dspark_prompt,
            *dspark_prompt_tokens,
            1,
            vec![initial_token_id],
            1,
            capture_budget,
        ))
        .map_err(|error| anyhow::anyhow!(error))
        .with_context(|| format!("executing dSpark M={} target capture", draft_tokens + 1))?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "dSpark M={} target capture failed: status={} verify_rows={} expected_rows={} blocker={} failed={:?}",
            draft_tokens + 1,
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            draft_tokens,
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements,
        );
        // Speculative acceptance recomputes final_decode_step from the tokens
        // actually emitted, so the synthetic decode_step_index above does not
        // guarantee that this state was recycled. End every width
        // explicitly before the next width reserves target KV: at C=1 there
        // is no spare reservation slot for arena rebind to drop the old one.
        finish_real_full_dspark_width_prewarm_sequence(
            finish_sequence,
            &sequence_id,
            draft_tokens + 1,
        )?;
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=dspark-width prompt_tokens={} drafts={} physical_m={} verify_rows={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            dspark_prompt_tokens,
            draft_tokens,
            draft_tokens + 1,
            verify_cycle.info.request_mtp_verify_rows,
            elapsed_ms(width_start),
            elapsed_ms(prewarm_start),
        );
    }
    Ok(())
}

fn prewarm_real_full_mtp_adaptive_draft_widths(
    execute: &mut impl FnMut(
        glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String>,
    mtp_prompt: &str,
    mtp_prompt_tokens: usize,
    worker_index: usize,
    prewarm_start: Instant,
) -> Result<()> {
    let policy = *real_full_mtp_draft_policy();
    if !policy.adaptive || policy.max <= 2 {
        return Ok(());
    }

    // Output-budget tails may be narrower than the configured minimum. D=1
    // is already exercised by production-bucket prewarm, so capture every
    // remaining reachable target width below the max-width scratch-sizing run.
    for draft_tokens in 2..policy.max {
        let sequence_id = format!(
            "real-full-startup-mtp-production-draft-{draft_tokens}-adaptive-{mtp_prompt_tokens}-sequence-{worker_index}"
        );
        let request_index = 10_000_u64
            .checked_add(
                u64::try_from(worker_index)
                    .context("adaptive MTP prewarm worker index exceeds u64")?
                    .saturating_mul(100),
            )
            .and_then(|index| {
                index.checked_add(u64::try_from(draft_tokens).ok()?.saturating_mul(2))
            })
            .context("adaptive MTP prewarm request index overflow")?;
        let decode_budget = draft_tokens.saturating_add(3);
        let width_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=mtp-adaptive-width prompt_tokens={} draft_tokens={} decode_budget={}",
            worker_index, mtp_prompt_tokens, draft_tokens, decode_budget,
        );
        let initial_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index,
            &sequence_id,
            mtp_prompt,
            mtp_prompt_tokens,
            1,
            Vec::new(),
            0,
            decode_budget,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| format!("preparing adaptive MTP draft width {draft_tokens}"))?;
        anyhow::ensure!(
            initial_cycle.info.status == "ready",
            "adaptive MTP width {draft_tokens} initial cycle was not ready: status={} blocker={} failed={:?}",
            initial_cycle.info.status,
            initial_cycle.info.blocker,
            initial_cycle.info.failed_requirements,
        );
        let initial_token_id = initial_cycle
            .generated_tokens
            .first()
            .map(|token| token.token_id)
            .or(initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id)
            .with_context(|| {
                format!("adaptive MTP width {draft_tokens} initial cycle produced no token")
            })?;
        let verify_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index + 1,
            &sequence_id,
            mtp_prompt,
            mtp_prompt_tokens,
            1,
            vec![initial_token_id],
            1,
            decode_budget,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| format!("capturing adaptive MTP target width {draft_tokens}"))?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && verify_cycle.info.request_mtp_verify_rows == draft_tokens,
            "adaptive MTP width {draft_tokens} verify failed: status={} verify_rows={} blocker={} failed={:?}",
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements,
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=mtp-adaptive-width prompt_tokens={} drafts={} accepted={} emitted={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            mtp_prompt_tokens,
            draft_tokens,
            verify_cycle.info.request_mtp_accepted_rows,
            verify_cycle.generated_tokens.len(),
            elapsed_ms(width_start),
            elapsed_ms(prewarm_start),
        );
    }
    Ok(())
}

fn prewarm_real_full_mtp_production_buckets(
    execute: &mut impl FnMut(
        glmrt_api::RealFullRequest,
    ) -> std::result::Result<glmrt_api::RealFullDecodeCycle, String>,
    prompts: &[(String, usize)],
    worker_index: usize,
    prewarm_start: Instant,
) -> Result<()> {
    let production_bucket_rows =
        coordinator_graph_bucket_for_active_rows(real_full_mtp_prefill_chunk_tokens())
            .context("selecting the production MTP prefill graph bucket")?
            .row_capacity;
    for (prompt_index, (mtp_prompt, mtp_prompt_tokens)) in prompts.iter().enumerate() {
        let bucket_rows = coordinator_graph_bucket_for_active_rows(*mtp_prompt_tokens)
            .with_context(|| {
                format!(
                    "selecting the MTP graph bucket for {mtp_prompt_tokens} startup prompt tokens"
                )
            })?
            .row_capacity;
        // Replay the production bucket once after the initial MTP verify has
        // finished sizing shared scratch, so its graphs survive capture close.
        if bucket_rows > production_bucket_rows {
            continue;
        }

        let sequence_id = format!(
            "real-full-startup-mtp-production-bucket-{bucket_rows}-sequence-{worker_index}"
        );
        let request_index = 20_000_u64
            .checked_add(
                u64::try_from(worker_index)
                    .context("MTP bucket prewarm worker index exceeds u64")?
                    .saturating_mul(1_000),
            )
            .and_then(|index| {
                index.checked_add(u64::try_from(prompt_index).ok()?.saturating_mul(10))
            })
            .context("MTP bucket prewarm request index overflow")?;
        let decode_budget = 3;
        let bucket_start = Instant::now();
        eprintln!(
            "real_full_startup_prewarm_start worker={} stage=mtp-production-bucket prompt_tokens={} bucket_rows={} decode_budget={}",
            worker_index, mtp_prompt_tokens, bucket_rows, decode_budget,
        );
        let initial_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index,
            &sequence_id,
            mtp_prompt,
            *mtp_prompt_tokens,
            1,
            Vec::new(),
            0,
            decode_budget,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| {
            format!("executing the {bucket_rows}-row production MTP graph-capture request")
        })?;
        anyhow::ensure!(
            initial_cycle.info.status == "ready",
            "production MTP graph-capture request for bucket {bucket_rows} was not ready: status={} sample_status={} blocker={} failed={:?}",
            initial_cycle.info.status,
            initial_cycle.info.scheduler_terminal_lm_head_sample_status,
            initial_cycle.info.blocker,
            initial_cycle.info.failed_requirements,
        );
        let initial_token_id = initial_cycle
            .generated_tokens
            .first()
            .map(|token| token.token_id)
            .or(initial_cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id)
            .with_context(|| {
                format!(
                    "production MTP graph-capture request for bucket {bucket_rows} produced no token"
                )
            })?;
        let mut generated_token_ids = vec![initial_token_id];
        let verify_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            request_index + 1,
            &sequence_id,
            mtp_prompt,
            *mtp_prompt_tokens,
            1,
            generated_token_ids.clone(),
            generated_token_ids.len(),
            decode_budget,
        ))
        .map_err(|err| anyhow::anyhow!(err))
        .with_context(|| {
            format!("verifying the {bucket_rows}-row production MTP graph-capture request")
        })?;
        anyhow::ensure!(
            verify_cycle.info.status == "ready"
                && (1..=real_full_mtp_draft_tokens())
                    .contains(&verify_cycle.info.request_mtp_verify_rows),
            "production MTP graph-capture verification for bucket {bucket_rows} failed: status={} verify_rows={} expected_range=1..={} blocker={} failed={:?}",
            verify_cycle.info.status,
            verify_cycle.info.request_mtp_verify_rows,
            real_full_mtp_draft_tokens(),
            verify_cycle.info.blocker,
            verify_cycle.info.failed_requirements,
        );
        generated_token_ids.extend(
            verify_cycle
                .generated_tokens
                .iter()
                .map(|token| token.token_id),
        );
        if generated_token_ids.len() < decode_budget {
            let final_cycle = execute(glmrt_api::RealFullRequest::new_decode_step_for_sequence(
                request_index + 2,
                &sequence_id,
                mtp_prompt,
                *mtp_prompt_tokens,
                1,
                generated_token_ids.clone(),
                generated_token_ids.len(),
                decode_budget,
            ))
            .map_err(|err| anyhow::anyhow!(err))
            .with_context(|| {
                format!("draining the {bucket_rows}-row production MTP graph-capture sequence")
            })?;
            anyhow::ensure!(
                final_cycle.info.status == "ready",
                "production MTP graph-capture drain for bucket {bucket_rows} failed: status={} blocker={} failed={:?}",
                final_cycle.info.status,
                final_cycle.info.blocker,
                final_cycle.info.failed_requirements,
            );
            generated_token_ids.extend(
                final_cycle
                    .generated_tokens
                    .iter()
                    .map(|token| token.token_id),
            );
        }
        anyhow::ensure!(
            generated_token_ids.len() >= decode_budget,
            "production MTP graph-capture sequence for bucket {bucket_rows} emitted only {} of {decode_budget} tokens",
            generated_token_ids.len(),
        );
        eprintln!(
            "real_full_startup_prewarm_step_done worker={} stage=mtp-production-bucket prompt_tokens={} bucket_rows={} accepted={} emitted={} elapsed_ms={:.3} total_ms={:.3}",
            worker_index,
            mtp_prompt_tokens,
            bucket_rows,
            verify_cycle.info.request_mtp_accepted_rows,
            generated_token_ids.len(),
            elapsed_ms(bucket_start),
            elapsed_ms(prewarm_start),
        );
    }
    Ok(())
}

fn real_full_sparse_owner_lookup_from_args(
    args: &CoordinatorArgs,
    catalog: &TensorCatalog,
) -> Result<Option<ExpertOwnerLookup>> {
    if let Some(path) = args.loadplan.as_deref() {
        return read_expert_owner_lookup(path)
            .map(Some)
            .context("loading real-glm-full sparse expert owner lookup");
    }
    crate::commands::model_artifacts::build_runtime_owner_lookup(catalog)
        .map(Some)
        .context("inferring real-glm-full sparse expert owner lookup")
}

fn real_full_sparse_tcp_targets_from_args(
    args: &CoordinatorArgs,
) -> Result<Option<Vec<TcpProtocolV2HostBatchTarget>>> {
    let Some(dispatch_transport) =
        RealFullSchedulerSparseDispatchTransport::from_label(args.transport.as_str())
    else {
        return Ok(None);
    };
    if dispatch_transport == RealFullSchedulerSparseDispatchTransport::VerbsHost {
        glmrt_transport::verbs_host_preflight()
            .context("real-glm-full verbs-host sparse dispatch RDMA preflight failed")?;
    }
    let entries = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .collect::<Vec<_>>();
    if entries.is_empty() {
        bail!(
            "real-glm-full {} transport requires --expert-hosts",
            dispatch_transport.label()
        );
    }

    if entries.len() == 1 && !entries[0].contains('=') {
        let addr = resolve_real_full_sparse_tcp_addr(entries[0], dispatch_transport.label())?;
        return Ok(Some(
            EXPERT_HOSTS
                .iter()
                .map(|host| TcpProtocolV2HostBatchTarget {
                    host: (*host).to_owned(),
                    addr,
                })
                .collect(),
        ));
    }

    let mut targets = Vec::with_capacity(entries.len());
    for entry in entries {
        let (host, raw_addr) = if let Some((host, raw_addr)) = entry.split_once('=') {
            (host.trim(), raw_addr.trim())
        } else {
            let host = entry.split_once(':').map_or(entry, |(host, _)| host).trim();
            (host, entry)
        };
        if host.is_empty() {
            bail!(
                "real-glm-full {} expert target {entry:?} has empty host",
                dispatch_transport.label()
            );
        }
        targets.push(TcpProtocolV2HostBatchTarget {
            host: host.to_owned(),
            addr: resolve_real_full_sparse_tcp_addr(raw_addr, dispatch_transport.label())?,
        });
    }

    let missing_hosts = EXPERT_HOSTS
        .iter()
        .filter(|host| !targets.iter().any(|target| target.host.as_str() == **host))
        .copied()
        .collect::<Vec<_>>();
    if !missing_hosts.is_empty() {
        bail!(
            "real-glm-full {} sparse dispatch is missing expert targets for [{}]; pass host=ip:port entries or a single target to mirror to all expert hosts",
            dispatch_transport.label(),
            missing_hosts.join(",")
        );
    }
    Ok(Some(targets))
}

fn resolve_real_full_sparse_tcp_addr(raw_target: &str, transport: &str) -> Result<SocketAddr> {
    let raw_target = raw_target.trim();
    if raw_target.is_empty() {
        bail!("real-glm-full {transport} expert target address is empty");
    }
    let with_port = if raw_target.contains(':') {
        raw_target.to_owned()
    } else {
        format!("{raw_target}:9100")
    };
    with_port
        .to_socket_addrs()
        .with_context(|| format!("resolving real-glm-full {transport} expert target {with_port}"))?
        .next()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "real-glm-full {transport} expert target {with_port} resolved to no addresses"
            )
        })
}

fn initialize_sparse_tcp_dispatch_status(
    info: &mut glmrt_api::RealFullInfo,
    targets: Option<&Vec<TcpProtocolV2HostBatchTarget>>,
) {
    if let Some(targets) = targets {
        info.scheduler_sparse_tcp_dispatch_status = "configured-not-run".to_owned();
        info.scheduler_sparse_tcp_dispatch_targets = targets.len();
    }
}

fn scheduler_sparse_tcp_expected_real_executor_id() -> u64 {
    expert_protocol_v2_compact_id(REAL_NVFP4_PROTOCOL_V2_EXECUTOR)
}

fn apply_sparse_tcp_dispatch_probe(
    info: &mut glmrt_api::RealFullInfo,
    target_count: usize,
    probe: &RealFullSchedulerSparseTcpDispatchProbe,
) {
    info.scheduler_sparse_tcp_dispatch_status = probe.status.to_owned();
    info.scheduler_sparse_tcp_dispatch_targets = target_count;
    info.scheduler_sparse_tcp_dispatch_sparse_layers = probe.sparse_layers;
    info.scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer =
        probe.scheduler_iterations_per_sparse_layer;
    info.scheduler_sparse_tcp_dispatch_batches = probe.sparse_batches;
    info.scheduler_sparse_tcp_dispatch_host_batches = probe.host_batches;
    info.scheduler_sparse_tcp_dispatch_global_rows = probe.global_rows;
    info.scheduler_sparse_tcp_dispatch_host_rows = probe.host_rows;
    info.scheduler_sparse_tcp_dispatch_routes = probe.routes;
    info.scheduler_sparse_tcp_dispatch_request_wire_bytes = probe.request_wire_bytes;
    info.scheduler_sparse_tcp_dispatch_response_wire_bytes = probe.response_wire_bytes;
    info.scheduler_sparse_tcp_dispatch_output_values = probe.output_values;
    info.scheduler_sparse_tcp_dispatch_output_finite_values = probe.output_finite_values;
    info.scheduler_sparse_tcp_dispatch_output_nonzero_values = probe.output_nonzero_values;
    info.scheduler_sparse_tcp_dispatch_output_checksum = probe.output_checksum;
    info.scheduler_sparse_tcp_dispatch_passed = probe.passed;
    info.scheduler_sparse_tcp_dispatch_expected_real_executor_id = probe.expected_real_executor_id;
    info.scheduler_sparse_tcp_dispatch_response_executor_ids_observed =
        probe.response_executor_ids_observed;
    info.scheduler_sparse_tcp_dispatch_real_executor_responses = probe.real_executor_responses;
    info.scheduler_sparse_tcp_dispatch_non_real_executor_responses =
        probe.non_real_executor_responses;
    info.scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4 = probe.all_responses_real_nvfp4;
    info.scheduler_sparse_tcp_dispatch_consumed_by_residual =
        sparse_tcp_dispatch_consumed_by_residual(info, probe);
    if !probe.passed {
        info.status = "blocked".to_owned();
        if info.blocker.trim().is_empty() {
            info.blocker = format!("real-full sparse TCP dispatch status={}", probe.status);
        }
        if !info
            .failed_requirements
            .iter()
            .any(|requirement| requirement == "scheduler_sparse_tcp_dispatch")
        {
            info.failed_requirements
                .push("scheduler_sparse_tcp_dispatch".to_owned());
        }
    }
}

fn sparse_tcp_dispatch_consumed_by_residual(
    info: &glmrt_api::RealFullInfo,
    probe: &RealFullSchedulerSparseTcpDispatchProbe,
) -> bool {
    probe.passed
        && probe.output_values > 0
        && info.scheduler_numeric_progression_passed
        && info.request_numeric_progression_mlp_value_updates > 0
}

fn load_real_glm_full_catalog(args: &CoordinatorArgs) -> Result<(String, TensorCatalog)> {
    if let Some(catalog_path) = args.catalog.as_deref() {
        let catalog = load_catalog(catalog_path)?;
        anyhow::ensure!(
            catalog.model_id == args.model_id,
            "catalog model_id {} does not match requested {}",
            catalog.model_id,
            args.model_id
        );
        return Ok((catalog_path.display().to_string(), catalog));
    }
    let catalog = crate::commands::model_artifacts::build_runtime_catalog(&args.model_id)?;
    Ok((format!("hf://{}", args.model_id), catalog))
}

pub(in crate::commands::real_full) fn real_full_info_from_startup(
    args: &CoordinatorArgs,
    catalog: &TensorCatalog,
    preload: RealFullCoordinatorResidentPreloadPlan,
) -> Result<glmrt_api::RealFullInfo> {
    let expert_hosts = args
        .expert_hosts
        .split(',')
        .map(str::trim)
        .filter(|target| !target.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let kv_config = real_full_kv_cache_config(args)?;
    let execution_plan = real_full_execution_plan(&expert_hosts, kv_config.bytes_per_token());

    Ok(glmrt_api::RealFullInfo {
        status: "blocked".to_owned(),
        model_id: args.model_id.clone(),
        snapshot_path: Some(catalog.snapshot_path.clone()),
        catalog_hash: catalog.content_hash(),
        tensor_count: catalog.tensors.len(),
        startup_diagnostic_mode: "serving-startup-residency-only".to_owned(),
        coordinator_resident_preload_status: preload.status.to_owned(),
        coordinator_resident_preload_selected_tensors: preload.selected_tensor_count,
        coordinator_resident_preload_selected_bytes: preload.selected_tensor_bytes,
        coordinator_resident_preload_loaded_bytes: preload.loaded_tensor_bytes,
        layer_count: execution_plan.layer_count,
        dense_layer_count: execution_plan.dense_layer_count,
        sparse_layer_count: execution_plan.sparse_layer_count,
        kv_layout: kv_config.layout_label().to_owned(),
        kv_bytes_per_token: kv_config.bytes_per_token(),
        request_prefill_tokens: 0,
        request_prefill_chunks: 0,
        request_kv_snapshot_restore_ms: 0.0,
        request_decode_budget: 0,
        request_mtp_verify_rows: 0,
        request_mtp_accepted_rows: 0,
        request_coordinator_graph_slots: 0,
        request_coordinator_graph_captured_graphs: 0,
        request_coordinator_graph_captures: 0,
        request_coordinator_graph_launches: 0,
        request_candidate_layerwaves: 0,
        request_deferred_layerwaves: 0,
        scheduler_iterations: 0,
        selected_layerwaves: 0,
        sparse_expert_batches: 0,
        request_expert_batch_rows: 0,
        request_expert_batch_routes: 0,
        request_expert_prefill_rows: 0,
        request_expert_decode_rows: 0,
        request_expert_mtp_verify_rows: 0,
        request_expert_prefill_routes: 0,
        request_expert_decode_routes: 0,
        request_expert_mtp_verify_routes: 0,
        kv_read_blocks: 0,
        committed_kv_writes: 0,
        tentative_kv_writes: 0,
        request_committed_mtp_writes: 0,
        request_discarded_mtp_writes: 0,
        request_backed_kv_writes: 0,
        request_backed_kv_bytes: 0,
        request_kv_reservation_bytes: 0,
        request_byte_backed_scheduler_trace: false,
        scheduler_numeric_progression_passed: false,
        scheduler_numeric_progression_source_rows: 0,
        scheduler_numeric_progression_hidden_dim: 0,
        scheduler_numeric_progression_visible_checksum: 0.0,
        scheduler_numeric_progression_rejected_mtp_checksum: 0.0,
        request_numeric_progression_selected_prefill_rows: 0,
        request_numeric_progression_selected_decode_rows: 0,
        request_numeric_progression_selected_mtp_rows: 0,
        request_numeric_progression_attention_value_updates: 0,
        request_numeric_progression_mlp_value_updates: 0,
        scheduler_full_context_device_attention_complete: false,
        scheduler_terminal_lm_head_sample_status: "not-run".to_owned(),
        scheduler_terminal_lm_head_sample_passed: false,
        scheduler_terminal_lm_head_uses_final_decode_device_hidden: false,
        scheduler_terminal_lm_head_covers_full_vocabulary: false,
        scheduler_terminal_lm_head_logits_evaluated: 0,
        scheduler_terminal_lm_head_vocab_size: 0,
        scheduler_terminal_lm_head_top_token_id: None,
        scheduler_terminal_lm_head_sampled_token_id: None,
        scheduler_terminal_lm_head_sampled_text: None,
        scheduler_terminal_lm_head_sample_top_k: None,
        scheduler_terminal_lm_head_sample_top_p: None,
        scheduler_terminal_lm_head_argmax_backend: None,
        scheduler_terminal_lm_head_sampler_backend: None,
        scheduler_terminal_lm_head_blocker: Some(
            "serving startup only preloads resident weights; scheduler terminal lm_head sampling is only populated by preflight execution"
                .to_owned(),
        ),
        protocol: execution_plan.protocol_payloads.protocol.to_owned(),
        decode_wire_request_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .decode_wire_request_bytes_per_touched_host,
        decode_wire_response_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .decode_wire_response_bytes_per_touched_host,
        prefill_wire_request_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .prefill_wire_request_bytes_per_touched_host,
        prefill_wire_response_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .prefill_wire_response_bytes_per_touched_host,
        mtp_wire_request_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .mtp_wire_request_bytes_per_touched_host,
        mtp_wire_response_bytes_per_touched_host: execution_plan
            .protocol_payloads
            .mtp_wire_response_bytes_per_touched_host,
        decode_full_sparse_roundtrip_wire_bytes: execution_plan
            .protocol_payloads
            .decode_full_sparse_roundtrip_wire_bytes,
        prefill_full_sparse_roundtrip_wire_bytes: execution_plan
            .protocol_payloads
            .prefill_full_sparse_roundtrip_wire_bytes,
        mtp_full_sparse_roundtrip_wire_bytes: execution_plan
            .protocol_payloads
            .mtp_full_sparse_roundtrip_wire_bytes,
        scheduler_sparse_tcp_dispatch_status: "not-configured".to_owned(),
        scheduler_sparse_tcp_dispatch_targets: 0,
        scheduler_sparse_tcp_dispatch_sparse_layers: 0,
        scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: 0,
        scheduler_sparse_tcp_dispatch_batches: 0,
        scheduler_sparse_tcp_dispatch_host_batches: 0,
        scheduler_sparse_tcp_dispatch_global_rows: 0,
        scheduler_sparse_tcp_dispatch_host_rows: 0,
        scheduler_sparse_tcp_dispatch_routes: 0,
        scheduler_sparse_tcp_dispatch_request_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_response_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_output_values: 0,
        scheduler_sparse_tcp_dispatch_output_finite_values: 0,
        scheduler_sparse_tcp_dispatch_output_nonzero_values: 0,
        scheduler_sparse_tcp_dispatch_output_checksum: 0.0,
        scheduler_sparse_tcp_dispatch_passed: false,
        scheduler_sparse_tcp_dispatch_expected_real_executor_id:
            scheduler_sparse_tcp_expected_real_executor_id(),
        scheduler_sparse_tcp_dispatch_response_executor_ids_observed: 0,
        scheduler_sparse_tcp_dispatch_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_non_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: false,
        scheduler_sparse_tcp_dispatch_consumed_by_residual: false,
        sampling_default_lm_head_chunk_passed: false,
        sampling_default_lm_head_chunk_rows_scored: 0,
        sampling_default_lm_head_chunk_lm_head_bytes_read: 0,
        sampling_default_lm_head_chunk_top_token_id: None,
        sampling_default_lm_head_chunk_top_logit: None,
        sampling_default_lm_head_chunk_uses_real_dense_prefix: false,
        sampling_default_lm_head_chunk_residual_source_dense_layers: 0,
        sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read: 0,
        sampling_default_lm_head_chunk_residual_after_checksum: None,
        blocker: REAL_GLM_FULL_BLOCKER.to_owned(),
        failed_requirements: vec![
            "full_residual_stream_execution".to_owned(),
            "full_vocab_sampling".to_owned(),
        ],
    })
}

#[allow(dead_code)]
pub(in crate::commands::real_full) fn real_full_info_from_report(
    report: &RealGlmFullPreflightReport,
) -> glmrt_api::RealFullInfo {
    let lm_head_chunk = &report.sampling_dry_run.real_lm_head_default_chunk_probe;
    let terminal_lm_head_sample = &report.scheduler_execution_dry_run.terminal_lm_head_sample;
    glmrt_api::RealFullInfo {
        status: report.status.to_owned(),
        model_id: report.model_id.clone(),
        snapshot_path: Some(report.snapshot_path.clone()),
        catalog_hash: report.catalog_hash.clone(),
        tensor_count: report.tensor_count,
        startup_diagnostic_mode: "preflight-report".to_owned(),
        coordinator_resident_preload_status: report.coordinator_resident_preload.status.to_owned(),
        coordinator_resident_preload_selected_tensors: report
            .coordinator_resident_preload
            .selected_tensor_count,
        coordinator_resident_preload_selected_bytes: report
            .coordinator_resident_preload
            .selected_tensor_bytes,
        coordinator_resident_preload_loaded_bytes: report
            .coordinator_resident_preload
            .loaded_tensor_bytes,
        layer_count: report.execution_plan.layer_count,
        dense_layer_count: report.execution_plan.dense_layer_count,
        sparse_layer_count: report.execution_plan.sparse_layer_count,
        kv_layout: report.kv_plan.layout.to_owned(),
        kv_bytes_per_token: report.kv_plan.bytes_per_token,
        request_prefill_tokens: report.scheduler_execution_dry_run.request_prefill_tokens,
        request_prefill_chunks: report.scheduler_execution_dry_run.request_prefill_chunks,
        request_kv_snapshot_restore_ms: 0.0,
        request_decode_budget: report.scheduler_execution_dry_run.request_decode_rows,
        request_mtp_verify_rows: report.scheduler_execution_dry_run.request_mtp_verify_rows,
        request_mtp_accepted_rows: report.scheduler_execution_dry_run.request_mtp_accepted_rows,
        request_coordinator_graph_slots: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_slots,
        request_coordinator_graph_captured_graphs: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_captured_graphs,
        request_coordinator_graph_captures: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_captures,
        request_coordinator_graph_launches: report
            .scheduler_execution_dry_run
            .request_coordinator_graph_launches,
        request_candidate_layerwaves: report.scheduler_execution_dry_run.candidate_layerwaves,
        request_deferred_layerwaves: report.scheduler_execution_dry_run.deferred_layerwaves,
        scheduler_iterations: report.scheduler_execution_dry_run.iterations,
        selected_layerwaves: report.scheduler_execution_dry_run.selected_layerwaves,
        sparse_expert_batches: report.scheduler_execution_dry_run.sparse_expert_batches,
        request_expert_batch_rows: report.scheduler_execution_dry_run.sparse_expert_batch_rows,
        request_expert_batch_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_batch_routes,
        request_expert_prefill_rows: report
            .scheduler_execution_dry_run
            .sparse_expert_prefill_rows,
        request_expert_decode_rows: report.scheduler_execution_dry_run.sparse_expert_decode_rows,
        request_expert_mtp_verify_rows: report
            .scheduler_execution_dry_run
            .sparse_expert_mtp_verify_rows,
        request_expert_prefill_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_prefill_routes,
        request_expert_decode_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_decode_routes,
        request_expert_mtp_verify_routes: report
            .scheduler_execution_dry_run
            .sparse_expert_mtp_verify_routes,
        kv_read_blocks: report.scheduler_execution_dry_run.kv_read_blocks,
        committed_kv_writes: report.scheduler_execution_dry_run.committed_kv_writes,
        tentative_kv_writes: report.scheduler_execution_dry_run.tentative_kv_writes,
        request_committed_mtp_writes: report.scheduler_execution_dry_run.committed_mtp_writes,
        request_discarded_mtp_writes: report.scheduler_execution_dry_run.discarded_mtp_writes,
        request_backed_kv_writes: report.scheduler_execution_dry_run.backed_kv_writes,
        request_backed_kv_bytes: report
            .scheduler_execution_dry_run
            .backed_bytes_after_discard,
        request_kv_reservation_bytes: report.scheduler_execution_dry_run.kv_reservation_bytes,
        request_byte_backed_scheduler_trace: report
            .scheduler_execution_dry_run
            .byte_backed_scheduler_trace,
        scheduler_numeric_progression_passed: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .passed,
        scheduler_numeric_progression_source_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .unique_source_rows,
        scheduler_numeric_progression_hidden_dim: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .hidden_dim,
        scheduler_numeric_progression_visible_checksum: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .final_visible_checksum,
        scheduler_numeric_progression_rejected_mtp_checksum: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .rejected_mtp_checksum,
        request_numeric_progression_selected_prefill_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .selected_prefill_rows,
        request_numeric_progression_selected_decode_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .selected_decode_rows,
        request_numeric_progression_selected_mtp_rows: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .selected_mtp_rows,
        request_numeric_progression_attention_value_updates: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .attention_value_updates,
        request_numeric_progression_mlp_value_updates: report
            .scheduler_execution_dry_run
            .numeric_progression_self_test
            .mlp_value_updates,
        scheduler_full_context_device_attention_complete: report
            .scheduler_execution_dry_run
            .full_context_device_attention_complete,
        scheduler_terminal_lm_head_sample_status: terminal_lm_head_sample.status.to_owned(),
        scheduler_terminal_lm_head_sample_passed: terminal_lm_head_sample.passed,
        scheduler_terminal_lm_head_uses_final_decode_device_hidden: terminal_lm_head_sample
            .uses_final_decode_device_hidden,
        scheduler_terminal_lm_head_covers_full_vocabulary: terminal_lm_head_sample
            .covers_full_vocabulary,
        scheduler_terminal_lm_head_logits_evaluated: terminal_lm_head_sample.logits_evaluated,
        scheduler_terminal_lm_head_vocab_size: terminal_lm_head_sample.vocab_size,
        scheduler_terminal_lm_head_top_token_id: terminal_lm_head_sample.top_token_id,
        scheduler_terminal_lm_head_sampled_token_id: terminal_lm_head_sample.sampled_token_id,
        scheduler_terminal_lm_head_sampled_text: decode_sampled_token_text(
            &report.snapshot_path,
            terminal_lm_head_sample.sampled_token_id,
        ),
        scheduler_terminal_lm_head_sample_top_k: terminal_lm_head_sample.sample_top_k,
        scheduler_terminal_lm_head_sample_top_p: terminal_lm_head_sample.sample_top_p,
        scheduler_terminal_lm_head_argmax_backend: terminal_lm_head_sample
            .argmax_kernel_backend
            .map(str::to_owned),
        scheduler_terminal_lm_head_sampler_backend: terminal_lm_head_sample
            .sampler_kernel_backend
            .map(str::to_owned),
        scheduler_terminal_lm_head_blocker: terminal_lm_head_sample.blocker.clone(),
        protocol: report.execution_plan.protocol_payloads.protocol.to_owned(),
        decode_wire_request_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .decode_wire_request_bytes_per_touched_host,
        decode_wire_response_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .decode_wire_response_bytes_per_touched_host,
        prefill_wire_request_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .prefill_wire_request_bytes_per_touched_host,
        prefill_wire_response_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .prefill_wire_response_bytes_per_touched_host,
        mtp_wire_request_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .mtp_wire_request_bytes_per_touched_host,
        mtp_wire_response_bytes_per_touched_host: report
            .execution_plan
            .protocol_payloads
            .mtp_wire_response_bytes_per_touched_host,
        decode_full_sparse_roundtrip_wire_bytes: report
            .execution_plan
            .protocol_payloads
            .decode_full_sparse_roundtrip_wire_bytes,
        prefill_full_sparse_roundtrip_wire_bytes: report
            .execution_plan
            .protocol_payloads
            .prefill_full_sparse_roundtrip_wire_bytes,
        mtp_full_sparse_roundtrip_wire_bytes: report
            .execution_plan
            .protocol_payloads
            .mtp_full_sparse_roundtrip_wire_bytes,
        scheduler_sparse_tcp_dispatch_status: "not-configured".to_owned(),
        scheduler_sparse_tcp_dispatch_targets: 0,
        scheduler_sparse_tcp_dispatch_sparse_layers: 0,
        scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: 0,
        scheduler_sparse_tcp_dispatch_batches: 0,
        scheduler_sparse_tcp_dispatch_host_batches: 0,
        scheduler_sparse_tcp_dispatch_global_rows: 0,
        scheduler_sparse_tcp_dispatch_host_rows: 0,
        scheduler_sparse_tcp_dispatch_routes: 0,
        scheduler_sparse_tcp_dispatch_request_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_response_wire_bytes: 0,
        scheduler_sparse_tcp_dispatch_output_values: 0,
        scheduler_sparse_tcp_dispatch_output_finite_values: 0,
        scheduler_sparse_tcp_dispatch_output_nonzero_values: 0,
        scheduler_sparse_tcp_dispatch_output_checksum: 0.0,
        scheduler_sparse_tcp_dispatch_passed: false,
        scheduler_sparse_tcp_dispatch_expected_real_executor_id:
            scheduler_sparse_tcp_expected_real_executor_id(),
        scheduler_sparse_tcp_dispatch_response_executor_ids_observed: 0,
        scheduler_sparse_tcp_dispatch_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_non_real_executor_responses: 0,
        scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: false,
        scheduler_sparse_tcp_dispatch_consumed_by_residual: false,
        sampling_default_lm_head_chunk_passed: lm_head_chunk.passed,
        sampling_default_lm_head_chunk_rows_scored: lm_head_chunk.rows_scored,
        sampling_default_lm_head_chunk_lm_head_bytes_read: lm_head_chunk.lm_head_bytes_read,
        sampling_default_lm_head_chunk_top_token_id: lm_head_chunk.top_token_id,
        sampling_default_lm_head_chunk_top_logit: lm_head_chunk.top_logit,
        sampling_default_lm_head_chunk_uses_real_dense_prefix: lm_head_chunk.uses_real_dense_prefix,
        sampling_default_lm_head_chunk_residual_source_dense_layers: lm_head_chunk
            .residual_source_dense_layers,
        sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read: lm_head_chunk
            .residual_source_dense_weight_bytes_read,
        sampling_default_lm_head_chunk_residual_after_checksum: lm_head_chunk
            .residual_after_checksum,
        blocker: report.blocker.to_owned(),
        failed_requirements: report
            .requirements
            .iter()
            .filter(|requirement| !requirement.passed)
            .map(|requirement| requirement.name.to_owned())
            .collect(),
    }
}

fn real_full_info_from_request_execution(
    base_info: &glmrt_api::RealFullInfo,
    snapshot_path: &str,
    report: &RealFullSchedulerExecutionDryRun,
    sampled_token_text: Option<String>,
) -> glmrt_api::RealFullInfo {
    let terminal_lm_head_sample = &report.terminal_lm_head_sample;
    let completed = report.full_context_device_attention_complete && terminal_lm_head_sample.passed;
    let mut info = base_info.clone();
    info.status = if completed { "ready" } else { "blocked" }.to_owned();
    info.startup_diagnostic_mode = "request-scheduler-execution".to_owned();
    info.request_prefill_tokens = report.request_prefill_tokens;
    info.request_prefill_chunks = report.request_prefill_chunks;
    info.request_decode_budget = report.request_decode_rows;
    info.request_mtp_verify_rows = report.request_mtp_verify_rows;
    info.request_mtp_accepted_rows = report.request_mtp_accepted_rows;
    info.request_coordinator_graph_slots = report.request_coordinator_graph_slots;
    info.request_coordinator_graph_captured_graphs =
        report.request_coordinator_graph_captured_graphs;
    info.request_coordinator_graph_captures = report.request_coordinator_graph_captures;
    info.request_coordinator_graph_launches = report.request_coordinator_graph_launches;
    info.request_candidate_layerwaves = report.candidate_layerwaves;
    info.request_deferred_layerwaves = report.deferred_layerwaves;
    info.scheduler_iterations = report.iterations;
    info.selected_layerwaves = report.selected_layerwaves;
    info.sparse_expert_batches = report.sparse_expert_batches;
    info.request_expert_batch_rows = report.sparse_expert_batch_rows;
    info.request_expert_batch_routes = report.sparse_expert_batch_routes;
    info.request_expert_prefill_rows = report.sparse_expert_prefill_rows;
    info.request_expert_decode_rows = report.sparse_expert_decode_rows;
    info.request_expert_mtp_verify_rows = report.sparse_expert_mtp_verify_rows;
    info.request_expert_prefill_routes = report.sparse_expert_prefill_routes;
    info.request_expert_decode_routes = report.sparse_expert_decode_routes;
    info.request_expert_mtp_verify_routes = report.sparse_expert_mtp_verify_routes;
    info.kv_read_blocks = report.kv_read_blocks;
    info.committed_kv_writes = report.committed_kv_writes;
    info.tentative_kv_writes = report.tentative_kv_writes;
    info.request_committed_mtp_writes = report.committed_mtp_writes;
    info.request_discarded_mtp_writes = report.discarded_mtp_writes;
    info.request_backed_kv_writes = report.backed_kv_writes;
    info.request_backed_kv_bytes = report.backed_bytes_after_discard;
    info.request_kv_reservation_bytes = report.kv_reservation_bytes;
    info.request_byte_backed_scheduler_trace = report.byte_backed_scheduler_trace;
    info.scheduler_numeric_progression_passed = report.numeric_progression_self_test.passed;
    info.scheduler_numeric_progression_source_rows =
        report.numeric_progression_self_test.unique_source_rows;
    info.scheduler_numeric_progression_hidden_dim = report.numeric_progression_self_test.hidden_dim;
    info.scheduler_numeric_progression_visible_checksum =
        report.numeric_progression_self_test.final_visible_checksum;
    info.scheduler_numeric_progression_rejected_mtp_checksum =
        report.numeric_progression_self_test.rejected_mtp_checksum;
    info.request_numeric_progression_selected_prefill_rows =
        report.numeric_progression_self_test.selected_prefill_rows;
    info.request_numeric_progression_selected_decode_rows =
        report.numeric_progression_self_test.selected_decode_rows;
    info.request_numeric_progression_selected_mtp_rows =
        report.numeric_progression_self_test.selected_mtp_rows;
    info.request_numeric_progression_attention_value_updates =
        report.numeric_progression_self_test.attention_value_updates;
    info.request_numeric_progression_mlp_value_updates =
        report.numeric_progression_self_test.mlp_value_updates;
    info.scheduler_full_context_device_attention_complete =
        report.full_context_device_attention_complete;
    info.scheduler_terminal_lm_head_sample_status = terminal_lm_head_sample.status.to_owned();
    info.scheduler_terminal_lm_head_sample_passed = terminal_lm_head_sample.passed;
    info.scheduler_terminal_lm_head_uses_final_decode_device_hidden =
        terminal_lm_head_sample.uses_final_decode_device_hidden;
    info.scheduler_terminal_lm_head_covers_full_vocabulary =
        terminal_lm_head_sample.covers_full_vocabulary;
    info.scheduler_terminal_lm_head_logits_evaluated = terminal_lm_head_sample.logits_evaluated;
    info.scheduler_terminal_lm_head_vocab_size = terminal_lm_head_sample.vocab_size;
    info.scheduler_terminal_lm_head_top_token_id = terminal_lm_head_sample.top_token_id;
    info.scheduler_terminal_lm_head_sampled_token_id = terminal_lm_head_sample.sampled_token_id;
    info.scheduler_terminal_lm_head_sampled_text = sampled_token_text.or_else(|| {
        decode_sampled_token_text(snapshot_path, terminal_lm_head_sample.sampled_token_id)
    });
    info.scheduler_terminal_lm_head_sample_top_k = terminal_lm_head_sample.sample_top_k;
    info.scheduler_terminal_lm_head_sample_top_p = terminal_lm_head_sample.sample_top_p;
    info.scheduler_terminal_lm_head_argmax_backend = terminal_lm_head_sample
        .argmax_kernel_backend
        .map(str::to_owned);
    info.scheduler_terminal_lm_head_sampler_backend = terminal_lm_head_sample
        .sampler_kernel_backend
        .map(str::to_owned);
    info.scheduler_terminal_lm_head_blocker = terminal_lm_head_sample.blocker.clone();
    if completed {
        info.blocker.clear();
        info.failed_requirements.clear();
    } else {
        info.blocker = terminal_lm_head_sample
            .blocker
            .clone()
            .unwrap_or_else(|| REAL_GLM_FULL_BLOCKER.to_owned());
        info.failed_requirements = real_full_request_failed_requirements(report);
    }
    info
}

fn real_full_request_failed_requirements(report: &RealFullSchedulerExecutionDryRun) -> Vec<String> {
    let mut failed = Vec::new();
    if !report.numeric_progression_self_test.passed {
        failed.push("scheduler_numeric_progression".to_owned());
    }
    if !report.full_context_device_attention_complete {
        failed.push("full_residual_stream_execution".to_owned());
    }
    if !report.terminal_lm_head_sample.passed {
        failed.push("full_vocab_sampling".to_owned());
    }
    failed
}

fn decode_sampled_token_text(snapshot_path: &str, token_id: Option<usize>) -> Option<String> {
    let token_id = token_id?;
    let token_id = u32::try_from(token_id).ok()?;
    decode_tokenizer_ids(Path::new(snapshot_path), &[token_id], false)
        .ok()
        .map(|summary| summary.text)
}

fn decode_sampled_token_text_with_tokenizer(
    tokenizer: &LoadedTokenizer,
    token_id: Option<usize>,
) -> Option<String> {
    let token_id = token_id?;
    let token_id = u32::try_from(token_id).ok()?;
    tokenizer
        .decode_ids(&[token_id], false)
        .ok()
        .map(|summary| summary.text)
}

fn load_catalog(path: &Path) -> Result<TensorCatalog> {
    serde_json::from_reader(
        File::open(path).with_context(|| format!("opening {}", path.display()))?,
    )
    .with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::{
        dflash_batch_group_size, finish_real_full_dspark_width_prewarm_sequence,
        parse_real_full_dspark_confidence_policy, real_full_batched_dspark_prewarm_buffer_bank,
        real_full_batched_dspark_prewarm_requested_draft_tokens,
        real_full_batched_dspark_prewarm_sequence, real_full_capture_arena_sequence,
        real_full_draft_absolute_context_start, real_full_draft_width_prewarm_passes,
        real_full_draft_width_prewarm_prompt_repeats, real_full_dspark_prefix_fingerprint,
        real_full_dspark_startup_draft_tokens, real_full_mtp_acceptance,
        real_full_mtp_draft_policy_from_values, real_full_mtp_draft_tokens_after_cycle_with_limit,
        real_full_mtp_draft_tokens_for_cycle_with_policy, real_full_mtp_physical_padding_rows,
        real_full_mtp_startup_forced_draft_tokens, real_full_native_mtp_sequence_enabled,
        real_full_nvfp4_short_k_graph_audit, real_full_nvfp4_short_k_prefill_capture_plan,
        real_full_nvfp4_short_k_prefill_decode_budget, real_full_paired_lm_head_buffer_bank,
        real_full_paired_lm_head_prewarm_range, real_full_prefill_chunk_tokens_for_direct_dsa,
        real_full_request_mtp_rows_for_policy, real_full_request_prefill_chunk_tokens_for_sequence,
        real_full_request_prefill_chunk_tokens_for_shape_with, real_full_request_token_rows,
        real_full_scalar_dspark_prewarm_requested_draft_tokens, real_full_sequence_capacity_tokens,
        real_full_sparse_tcp_targets_from_args, real_full_startup_target_radix_evict_tokens,
        real_full_startup_target_radix_publish_tokens,
        real_full_startup_workspace_is_final_capture_set,
        real_full_startup_workspace_sizing_sequence, real_full_validate_sparse_wave_capacity,
        request_prompt_token_ids, retain_graph_bound_scheduler_arena, Dflash2AdaptiveDraftState,
        DsparkConfidenceCalibrator, DsparkConfidenceResidual, DsparkRequestCacheSnapshot,
        RealFullContextTokenBudget, RealFullContextTokenExtent, RealFullDraftCacheSnapshot,
        RealFullDsparkCacheMode, RealFullDsparkConfidencePolicy, RealFullDsparkTailCache,
        RealFullDsparkTailEntry, RealFullDsparkTailKey, TargetKvRadixManager,
        REAL_FULL_MAX_ACTIVE_REQUESTS, REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS,
        REAL_FULL_SHARED_KV_PAGE_TOKENS,
    };
    use crate::cli::CoordinatorArgs;
    use crate::commands::real_full::coordinator_kernels::{
        glm_dsa_sparse_mla_attention_topk, glm_dsa_sparse_mla_query_bucket,
    };
    use crate::commands::real_full::preflight::real_full_sparse_transport_plan;
    use glmrt_core::{KvCacheDType, DEFAULT_MODEL_ID, EXPERT_HOSTS};
    use glmrt_loader::LoadedTokenizer;
    use std::fs;
    use std::sync::Arc;

    fn coordinator_args(transport: &str, expert_hosts: &str) -> CoordinatorArgs {
        CoordinatorArgs {
            backend: "real-glm-full".to_owned(),
            transport: transport.to_owned(),
            kv_cache_dtype: "bf16".to_owned(),
            max_context_tokens: crate::cli::DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS,
            listen: "127.0.0.1:8000".to_owned(),
            model_id: DEFAULT_MODEL_ID.to_owned(),
            expert_hosts: expert_hosts.to_owned(),
            catalog: None,
            loadplan: None,
            preflight_only: false,
        }
    }

    #[test]
    fn dflash_adaptive_confidence_treats_unreached_positions_as_censored() {
        let mut state = Dflash2AdaptiveDraftState::default();
        assert_eq!(state.conditional_confidence(5), vec![0.75; 5]);
        state.observe(5, 2);
        assert_eq!(
            state.conditional_confidence(5),
            vec![0.8, 0.8, 0.6, 0.75, 0.75]
        );
    }

    #[test]
    fn dflash_adaptive_starts_at_k5_then_uses_a_bounded_recent_history() {
        let mut state = Dflash2AdaptiveDraftState::default();
        for _ in 0..3 {
            state.observe(5, 5);
            assert!(state.cold_start());
        }
        state.observe(5, 5);
        assert!(!state.cold_start());
        for _ in 0..64 {
            state.observe(7, 7);
        }
        assert_eq!(state.history.len(), super::DFLASH2_ADAPTIVE_HISTORY_LIMIT);
        assert!(state
            .conditional_confidence(7)
            .into_iter()
            .all(|confidence| confidence > 0.94));
    }

    #[test]
    fn dflash_batches_four_then_two_and_leaves_only_a_scalar_tail() {
        assert_eq!(dflash_batch_group_size(0), None);
        assert_eq!(dflash_batch_group_size(1), None);
        assert_eq!(dflash_batch_group_size(2), Some(2));
        assert_eq!(dflash_batch_group_size(3), Some(2));
        assert_eq!(dflash_batch_group_size(4), Some(4));
        assert_eq!(dflash_batch_group_size(7), Some(4));
    }

    #[test]
    fn dflash_width_prewarm_covers_dsa_and_the_settled_output_rotation() {
        assert_eq!(real_full_draft_width_prewarm_prompt_repeats(false), 8);
        assert_eq!(real_full_draft_width_prewarm_passes(false), 1);
        assert_eq!(real_full_draft_width_prewarm_prompt_repeats(true), 2_048);
        assert_eq!(real_full_draft_width_prewarm_passes(true), 2);
    }

    #[test]
    fn prompt_swa_replay_uses_the_absolute_prompt_suffix_position_only_once() {
        assert_eq!(
            real_full_draft_absolute_context_start(
                0,
                Some(RealFullDsparkCacheMode::PromptSwa),
                32_768,
                7,
            ),
            Some(32_775)
        );
        assert_eq!(
            real_full_draft_absolute_context_start(
                1,
                Some(RealFullDsparkCacheMode::PromptSwa),
                32_768,
                7,
            ),
            None
        );
        assert_eq!(
            real_full_draft_absolute_context_start(
                0,
                Some(RealFullDsparkCacheMode::RequestLocal),
                32_768,
                7,
            ),
            None
        );
        assert_eq!(
            real_full_draft_absolute_context_start(0, None, 32_768, 7),
            None
        );
    }

    #[test]
    fn dspark_confidence_policy_supports_calibrated_raw_and_residual_modes() {
        assert_eq!(
            parse_real_full_dspark_confidence_policy(None).unwrap(),
            RealFullDsparkConfidencePolicy::Residual
        );
        assert_eq!(
            parse_real_full_dspark_confidence_policy(Some("raw")).unwrap(),
            RealFullDsparkConfidencePolicy::Raw
        );
        assert_eq!(
            parse_real_full_dspark_confidence_policy(Some("residual")).unwrap(),
            RealFullDsparkConfidencePolicy::Residual
        );
        assert!(parse_real_full_dspark_confidence_policy(Some("legacy")).is_err());
    }

    fn dspark_tail_entry(token_ids: &[usize], bytes: usize) -> RealFullDsparkTailEntry {
        RealFullDsparkTailEntry {
            key: RealFullDsparkTailKey {
                prefix_tokens: token_ids.len(),
                prefix_sha256: real_full_dspark_prefix_fingerprint(token_ids),
            },
            snapshot: RealFullDraftCacheSnapshot::Dspark(DsparkRequestCacheSnapshot {
                context_tokens: token_ids.len(),
                cache_context_tokens: token_ids.len(),
                kv_bytes: vec![0_u8; bytes],
            }),
            confidence_calibrator: DsparkConfidenceCalibrator::default(),
            confidence_residual: DsparkConfidenceResidual::default(),
        }
    }

    #[test]
    fn dspark_tail_cache_requires_the_exact_target_radix_frontier() {
        let token_ids = [11, 12, 13, 14];
        let mut cache = RealFullDsparkTailCache::new(16);
        assert!(cache.insert(dspark_tail_entry(&token_ids[..3], 4)));
        assert!(cache.take_exact_prefix(&token_ids, 4).is_none());
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(
            cache
                .take_exact_prefix(&token_ids, 3)
                .expect("the exact dSpark tail is reusable")
                .key
                .prefix_tokens,
            3
        );
        assert_eq!(cache.resident_bytes, 0);
    }

    #[test]
    fn dspark_tail_cache_evicts_lru_entries_by_bytes() {
        let mut cache = RealFullDsparkTailCache::new(8);
        assert!(cache.insert(dspark_tail_entry(&[1], 4)));
        assert!(cache.insert(dspark_tail_entry(&[1, 2], 4)));
        assert!(cache.insert(dspark_tail_entry(&[1, 2, 3], 4)));
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.entries[0].key.prefix_tokens, 2);
        assert_eq!(cache.entries[1].key.prefix_tokens, 3);
        assert_eq!(cache.resident_bytes, 8);
    }

    #[test]
    fn dspark_tail_cache_finds_the_longest_target_aligned_prefix() {
        let prompt = [1, 2, 3, 4];
        let mut cache = RealFullDsparkTailCache::new(16);
        assert!(cache.insert(dspark_tail_entry(&prompt[..2], 4)));
        assert!(cache.insert(dspark_tail_entry(&prompt[..3], 4)));
        assert_eq!(cache.longest_exact_prefix_tokens(&prompt, 4), 3);
        assert_eq!(cache.longest_exact_prefix_tokens(&prompt, 2), 2);
        assert_eq!(cache.longest_exact_prefix_tokens(&[9, 2, 3, 4], 4), 0);
    }

    #[test]
    fn cached_prefix_startup_probe_uses_graph_stable_capture_arena() {
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-capture-arena-4097-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-prefix-prefill-seed-1009-1008-repeat-0-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-mtp-production-2049-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-mtp-production-bucket-1024-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-mtp-production-draft-8-2049-sequence-0"
        ));
        assert!(real_full_capture_arena_sequence(
            "real-full-startup-dsa-selector-seed-8193-sequence-0"
        ));
        assert!(!real_full_capture_arena_sequence(
            "real-full-startup-prewarm-initial-1009-sequence-0"
        ));
    }

    #[test]
    fn startup_target_radix_control_sequences_are_strictly_parsed() {
        let publish = "real-full-startup-capture-arena-radix-publish-8192-sequence-3";
        assert_eq!(
            real_full_startup_target_radix_publish_tokens(publish),
            Some(8192)
        );
        assert!(real_full_startup_workspace_sizing_sequence(publish));
        assert!(real_full_capture_arena_sequence(publish));
        assert_eq!(
            real_full_startup_target_radix_publish_tokens(
                "real-full-startup-capture-arena-canonical-prefill-chunk-3072-sequence-3"
            ),
            Some(3072)
        );

        assert_eq!(
            real_full_startup_target_radix_evict_tokens(
                "real-full-startup-evict-target-radix-prefix-8192-worker-3"
            ),
            Some(8192)
        );
        assert_eq!(
            real_full_startup_target_radix_publish_tokens(
                "real-full-startup-capture-arena-radix-publish-0-sequence-3"
            ),
            None
        );
        assert_eq!(
            real_full_startup_target_radix_evict_tokens(
                "real-full-startup-evict-target-radix-prefix-nope-worker-3"
            ),
            None
        );
    }

    #[test]
    fn startup_targeted_prefill_sequences_force_both_production_widths() {
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_sequence(
                "real-full-startup-capture-arena-max-prefill-chunk-6145-sequence-0",
                0,
                6_144,
            ),
            2_048,
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_sequence(
                "real-full-startup-capture-arena-canonical-prefill-chunk-3072-sequence-0",
                0,
                3_072,
            ),
            1_024,
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_sequence(
                "real-full-startup-capture-arena-2049-sequence-0",
                0,
                2_048,
            ),
            512,
        );
    }

    #[test]
    fn paired_lm_head_prewarm_starts_after_the_single_request_widths() {
        assert_eq!(real_full_paired_lm_head_prewarm_range(Some(0)), None);
        assert_eq!(
            real_full_paired_lm_head_prewarm_range(Some(1)),
            Some((3, 4))
        );
        assert_eq!(
            real_full_paired_lm_head_prewarm_range(Some(7)),
            Some((9, 16))
        );
        assert_eq!(real_full_paired_lm_head_prewarm_range(None), Some((17, 32)));
    }

    #[test]
    fn paired_lm_head_uses_disjoint_owned_buffer_banks() {
        assert_eq!(real_full_paired_lm_head_buffer_bank(1), 17);
        assert_eq!(real_full_paired_lm_head_buffer_bank(8), 24);
        assert!((1..=8).all(|bank| real_full_paired_lm_head_buffer_bank(bank) > 8));
    }

    #[test]
    fn batched_dspark_prewarm_sequences_bind_the_requested_production_bank() {
        let sequence_id = "real-full-startup-dspark-width-15-batched-bank-4-sequence";
        assert_eq!(
            real_full_dspark_startup_draft_tokens("real-full-startup-dspark-width-0-9-sequence-0"),
            Some(0)
        );
        assert_eq!(
            real_full_batched_dspark_prewarm_buffer_bank(sequence_id),
            Some(4)
        );
        assert!(real_full_batched_dspark_prewarm_sequence(sequence_id));
        assert_eq!(
            real_full_batched_dspark_prewarm_requested_draft_tokens(sequence_id, 93_511),
            Some(15)
        );
        assert_eq!(
            real_full_batched_dspark_prewarm_requested_draft_tokens(sequence_id, 92_011),
            Some(0)
        );
        assert_eq!(
            real_full_batched_dspark_prewarm_requested_draft_tokens(sequence_id, 93_611),
            None
        );
        assert!(!real_full_batched_dspark_prewarm_sequence(
            "real-full-startup-dspark-width-15-9-sequence-0"
        ));
        assert!(!real_full_batched_dspark_prewarm_sequence(
            "real-full-startup-dspark-width-0-batched-bank-4-sequence"
        ));
    }

    #[test]
    fn scalar_dspark_width_cohort_decodes_each_request_width() {
        let sequence_id = "real-full-startup-dspark-width-7-scalar-cohort-2049-sequence-0";
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(sequence_id, 91_700),
            Some(7)
        );
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(sequence_id, 91_001),
            Some(0)
        );
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(sequence_id, 90_999),
            None
        );
        assert_eq!(
            real_full_scalar_dspark_prewarm_requested_draft_tokens(
                "real-full-startup-dspark-width-7-2049-sequence-0",
                91_700,
            ),
            None
        );
    }

    #[test]
    fn serial_dspark_width_finish_releases_single_target_kv_slot() {
        let manager =
            Arc::new(TargetKvRadixManager::new(4 * REAL_FULL_SHARED_KV_PAGE_TOKENS, 1).unwrap());
        let first = manager.reserve(&[1, 2], REAL_FULL_SHARED_KV_PAGE_TOKENS);
        let mut first = Some(first.unwrap());
        let exhausted = manager
            .reserve(&[3, 4], REAL_FULL_SHARED_KV_PAGE_TOKENS)
            .unwrap_err();
        assert!(format!("{exhausted:#}")
            .contains("target KV active request limit exhausted: active=1 max=1"));

        let mut finished = Vec::new();
        {
            let mut finish_sequence = |sequence_id: &str| {
                finished.push(sequence_id.to_owned());
                drop(first.take());
                Ok(())
            };
            finish_real_full_dspark_width_prewarm_sequence(
                &mut finish_sequence,
                "real-full-startup-dspark-width-7-9-sequence-0",
                8,
            )
            .unwrap();
        }

        assert_eq!(finished, ["real-full-startup-dspark-width-7-9-sequence-0"]);
        assert_eq!(manager.stats().active_reservations, 0);
        let next = manager
            .reserve(&[3, 4], REAL_FULL_SHARED_KV_PAGE_TOKENS)
            .unwrap();
        assert_eq!(manager.stats().active_reservations, 1);
        drop(next);
    }

    #[test]
    fn nvfp4_short_k_prefill_capture_plan_covers_every_missing_no_selector_bucket() {
        let plan = real_full_nvfp4_short_k_prefill_capture_plan();
        assert_eq!(plan.len(), 12);
        let decode_budget = real_full_nvfp4_short_k_prefill_decode_budget().unwrap();
        assert_eq!(decode_budget, 116);

        let mut coverage = Vec::with_capacity(plan.len());
        for case in plan {
            let sequence_capacity = real_full_sequence_capacity_tokens(
                case.anchor_prompt_tokens,
                decode_budget,
                crate::cli::DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS,
            )
            .unwrap();
            assert!(
                sequence_capacity > case.total_rows,
                "seed state must cover the cumulative short-K sweep: {case:?}"
            );
            assert_eq!(
                glm_dsa_sparse_mla_query_bucket(KvCacheDType::Nvfp4, case.query_rows),
                Some(case.query_rows),
            );
            assert_eq!(
                glm_dsa_sparse_mla_attention_topk(
                    KvCacheDType::Nvfp4,
                    case.query_rows,
                    case.total_rows,
                ),
                case.sparse_topk,
            );
            assert_eq!(
                glm_dsa_sparse_mla_attention_topk(
                    KvCacheDType::Nvfp4,
                    case.query_rows,
                    case.total_rows + REAL_FULL_SERVE_NVFP4_SHORT_K_PREFILL_QUERY_ROWS.len() + 1,
                ),
                case.sparse_topk,
                "capture anchor must tolerate one sampled row per sweep step: {case:?}",
            );
            assert!(
                case.total_rows <= 2_048,
                "short-K prefill capture must stay below the selector threshold: {case:?}",
            );
            coverage.push((case.anchor_prompt_tokens, case.query_rows, case.sparse_topk));
        }

        assert_eq!(
            coverage,
            [
                (9, 16, 128),
                (9, 32, 128),
                (9, 64, 128),
                (145, 16, 512),
                (145, 32, 512),
                (145, 64, 512),
                (513, 16, 1_024),
                (513, 32, 1_024),
                (513, 64, 1_024),
                (1_025, 16, 2_048),
                (1_025, 32, 2_048),
                (1_025, 64, 2_048),
            ],
        );
    }

    #[test]
    fn nvfp4_short_k_post_outer_audit_sequence_is_strictly_parsed() {
        assert_eq!(
            real_full_nvfp4_short_k_graph_audit(
                "real-full-startup-audit-nvfp4-short-k-q64-k1024-worker-3"
            ),
            Some((64, 1_024)),
        );
        assert_eq!(
            real_full_nvfp4_short_k_graph_audit(
                "real-full-startup-audit-nvfp4-short-k-q8-k1024-worker-3"
            ),
            None,
        );
        assert_eq!(
            real_full_nvfp4_short_k_graph_audit(
                "real-full-startup-audit-nvfp4-short-k-q64-k256-worker-3"
            ),
            None,
        );
    }

    #[test]
    fn only_explicit_graph_bound_max_context_arenas_are_recycled() {
        assert!(retain_graph_bound_scheduler_arena(true, 8_192, 8_192));
        assert!(!retain_graph_bound_scheduler_arena(false, 8_192, 8_192));
        assert!(!retain_graph_bound_scheduler_arena(true, 4_096, 8_192));
    }

    #[test]
    fn real_full_sequence_capacity_is_bounded_and_leaves_small_extension_headroom() {
        let max_context_tokens = crate::cli::DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS;
        assert_eq!(
            real_full_sequence_capacity_tokens(1_009, 2, max_context_tokens).unwrap(),
            2_020
        );
        assert_eq!(
            real_full_sequence_capacity_tokens(max_context_tokens - 1, 1, max_context_tokens)
                .unwrap(),
            max_context_tokens
        );
        assert!(
            real_full_sequence_capacity_tokens(max_context_tokens, 1, max_context_tokens).is_err()
        );
    }

    #[test]
    fn request_prefill_chunk_width_distinguishes_small_suffixes_and_balances_large_ones() {
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(512, 4_096, 1_024, 0, 1_008),
            504
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(512, 4_096, 1_024, 1_009, 1_008,),
            256
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 1_024, 1_994,
            ),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(1_024, 4_096, 1_024, 0, 4_095,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 529,),
            265
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_041,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_033,),
            517
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_038,),
            519
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 1_039,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(1_024, 4_096, 1_024, 0, 4_096,),
            1_024
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(512, 4_096, 1_024, 1_009, 8_192,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 4_096,),
            1_024
        );
        // The largest published prefill-grid suffix is still executed as
        // eight 2,048-row sparse waves, not one 16K expert launch.
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 16_384,),
            2_048
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_768, 16_384,
            ),
            2_048
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 1_024, 0, 5_157,),
            1_290
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_768, 13_584,
            ),
            1_941
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_767, 1_008,
            ),
            256
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 32_768, 1_008,
            ),
            1_024
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_008,
            ),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 512, 100_000, 530,),
            530
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 512, 100_000, 896,),
            896
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(2_048, 4_096, 512, 100_000, 897,),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_032,
            ),
            516
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_038,
            ),
            519
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 512, 100_000, 1_039,
            ),
            512
        );
        assert_eq!(
            real_full_request_prefill_chunk_tokens_for_shape_with(
                2_048, 4_096, 1_024, 100_000, 2_050,
            ),
            1_024
        );
    }

    #[test]
    fn production_prefill_stays_within_direct_dsa_query_capacity() {
        assert_eq!(real_full_prefill_chunk_tokens_for_direct_dsa(4_096), 2_048);
        assert_eq!(real_full_prefill_chunk_tokens_for_direct_dsa(2_048), 2_048);
        assert_eq!(real_full_prefill_chunk_tokens_for_direct_dsa(1_024), 1_024);
        assert_eq!(real_full_prefill_chunk_tokens_for_direct_dsa(512), 512);
    }

    #[test]
    fn combined_prefill_and_maximum_dspark_suffix_fit_the_exl3_tail_bucket() {
        assert!(real_full_validate_sparse_wave_capacity(2_048, 1, 15).is_ok());
        assert!(real_full_validate_sparse_wave_capacity(2_048, 1, 16).is_err());
        assert!(real_full_validate_sparse_wave_capacity(0, 1, 15).is_ok());
    }

    #[test]
    fn real_full_context_budget_releases_dropped_sequence_reservations() {
        let budget = Arc::new(RealFullContextTokenBudget::new(3 * 64));
        let first = budget.reserve(65).unwrap();
        assert_eq!(first.token_base(), 0);
        assert_eq!(first.reserved_tokens, 128);
        let second = budget.reserve(1).unwrap();
        assert_eq!(second.token_base(), 128);
        assert!(budget.reserve(1).is_err());
        drop(first);
        let replacement = budget.reserve(64).unwrap();
        assert_eq!(replacement.token_base(), 0);
        drop((second, replacement));
        let inner = budget.inner.lock().unwrap();
        assert_eq!(inner.used_tokens, 0);
        assert_eq!(inner.active_reservations, 0);
        assert_eq!(
            inner.free_extents,
            vec![RealFullContextTokenExtent {
                token_base: 0,
                tokens: 3 * 64
            }]
        );
    }

    #[test]
    fn real_full_context_budget_caps_the_active_resident_set() {
        let budget = Arc::new(RealFullContextTokenBudget::new(
            (REAL_FULL_MAX_ACTIVE_REQUESTS + 1) * 64,
        ));
        let reservations = (0..REAL_FULL_MAX_ACTIVE_REQUESTS)
            .map(|_| budget.reserve(1).unwrap())
            .collect::<Vec<_>>();
        assert!(budget.reserve(1).is_err());
        drop(reservations);
    }

    #[test]
    fn real_full_sparse_tcp_targets_are_disabled_for_inproc() {
        let targets =
            real_full_sparse_tcp_targets_from_args(&coordinator_args("inproc", "127.0.0.1:9100"))
                .unwrap();

        assert!(targets.is_none());
    }

    #[test]
    fn real_full_sparse_tcp_targets_expand_single_addr_to_all_expert_hosts() {
        let targets =
            real_full_sparse_tcp_targets_from_args(&coordinator_args("tcp", "127.0.0.1:9100"))
                .unwrap()
                .unwrap();

        assert_eq!(targets.len(), EXPERT_HOSTS.len());
        for host in EXPERT_HOSTS {
            let target = targets
                .iter()
                .find(|target| target.host == host)
                .expect("expanded target for expert host");
            assert_eq!(target.addr.port(), 9100);
        }
    }

    #[test]
    fn real_full_sparse_tcp_targets_accept_owner_mapped_entries() {
        let targets = real_full_sparse_tcp_targets_from_args(&coordinator_args(
            "tcp",
            "spark-0=127.0.0.1:9101,spark-1=127.0.0.1:9102,spark-2=127.0.0.1:9103,spark-3=127.0.0.1:9104",
        ))
        .unwrap()
        .unwrap();

        assert_eq!(targets.len(), 4);
        assert_eq!(targets[0].host, "spark-0");
        assert_eq!(targets[0].addr.port(), 9101);
        assert_eq!(targets[3].host, "spark-3");
        assert_eq!(targets[3].addr.port(), 9104);
    }

    #[test]
    fn real_full_sparse_tcp_targets_require_all_expert_hosts() {
        let err = real_full_sparse_tcp_targets_from_args(&coordinator_args(
            "tcp",
            "spark-0=127.0.0.1:9101,spark-1=127.0.0.1:9102",
        ))
        .unwrap_err();

        assert!(err.to_string().contains("missing expert targets"));
        assert!(err.to_string().contains("spark-2,spark-3"));
    }

    #[test]
    fn real_full_sparse_verbs_host_targets_and_plan_follow_rdma_preflight() {
        let args = coordinator_args(
            "verbs-host",
            "spark-0=127.0.0.1:9100,spark-1=127.0.0.1:9100,spark-2=127.0.0.1:9100,spark-3=127.0.0.1:9100",
        );
        let preflight_ok = glmrt_transport::verbs_host_preflight().is_ok();
        let targets = real_full_sparse_tcp_targets_from_args(&args);
        if preflight_ok {
            let targets = targets.unwrap().unwrap();
            assert_eq!(targets.len(), EXPERT_HOSTS.len());
            assert!(targets.iter().all(|target| target.addr.port() == 9100));
        } else {
            assert!(targets
                .unwrap_err()
                .to_string()
                .contains("RDMA preflight failed"));
        }

        let plan = real_full_sparse_transport_plan(&args);
        assert_eq!(plan.transport, "verbs-host");
        assert_eq!(plan.supports_rdma, true);
        assert_eq!(plan.supports_host_registered_buffers, true);
        assert_eq!(plan.app_transport_implemented, true);
        assert_eq!(
            plan.app_transport_status,
            glmrt_transport::VERBS_HOST_APP_TRANSPORT_STATUS
        );
        assert_eq!(plan.sparse_dispatch_available, preflight_ok);
        assert_eq!(
            plan.scheduler_dispatch_backend.as_deref(),
            preflight_ok.then_some("verbs-host-protocol-v2-rc-qp")
        );
        assert_eq!(
            plan.frame_protocol.as_deref(),
            Some(glmrt_transport::EXPERT_PROTOCOL_V2_FRAME_PROTOCOL)
        );
        assert_eq!(plan.blocker.is_none(), preflight_ok);
        assert_eq!(plan.preflight_ok, preflight_ok);
    }

    #[test]
    fn real_full_request_mtp_rows_default_to_disabled_for_live_serve() {
        let first_decode_loop_step =
            glmrt_api::RealFullRequest::new_decode_step(1, "user: hi", 3, 4, Vec::new(), 0, 4);

        assert_eq!(
            real_full_request_mtp_rows_for_policy(&first_decode_loop_step, false, false),
            0
        );
    }

    #[test]
    fn ordinary_dspark_workspace_is_the_final_startup_capture_set() {
        assert!(real_full_startup_workspace_is_final_capture_set(
            false, false, false
        ));
    }

    #[test]
    fn optional_startup_probes_require_the_conservative_final_capture_sweep() {
        for (prefix_prefill_probe, native_mtp, native_mtp_probe) in [
            (true, false, false),
            (false, true, false),
            (false, false, true),
            (true, true, true),
        ] {
            assert!(!real_full_startup_workspace_is_final_capture_set(
                prefix_prefill_probe,
                native_mtp,
                native_mtp_probe,
            ));
        }
    }

    #[test]
    fn real_full_request_mtp_rows_are_opt_in_and_disabled_for_recurrent_or_single_token_decode() {
        let first_decode_loop_step =
            glmrt_api::RealFullRequest::new_decode_step(1, "user: hi", 3, 4, Vec::new(), 0, 4);
        let later_decode_loop_step =
            glmrt_api::RealFullRequest::new_decode_step(2, "user: hi", 3, 4, vec![13], 1, 4);
        let standalone_single_step =
            glmrt_api::RealFullRequest::new_decode_step(3, "user: hi", 3, 1, Vec::new(), 0, 1);

        assert_eq!(
            real_full_request_mtp_rows_for_policy(&first_decode_loop_step, false, true),
            4
        );
        assert_eq!(
            real_full_request_mtp_rows_for_policy(&later_decode_loop_step, true, true),
            0
        );
        assert_eq!(
            real_full_request_mtp_rows_for_policy(&standalone_single_step, false, true),
            0
        );
    }

    #[test]
    fn real_full_mtp_acceptance_without_bridge_keeps_a_cache_backed_leading_run() {
        assert_eq!(
            real_full_mtp_acceptance(&[11, 12, 13], &[11, 12, 99, 100], false, 4).unwrap(),
            super::RealFullMtpAcceptance {
                accepted_draft_tokens: 2,
                terminal_target_index: 2,
                full_match_bonus: false,
            }
        );
        assert_eq!(
            real_full_mtp_acceptance(&[11, 12], &[99, 100, 101], false, 3)
                .unwrap()
                .accepted_draft_tokens,
            0,
        );
        assert_eq!(
            real_full_mtp_acceptance(&[11, 12], &[11, 12, 13], false, 3)
                .unwrap()
                .accepted_draft_tokens,
            1,
        );
        assert_eq!(
            real_full_mtp_acceptance(&[11], &[11, 12], false, 2)
                .unwrap()
                .accepted_draft_tokens,
            0,
        );
    }

    #[test]
    fn real_full_mtp_acceptance_with_bridge_emits_the_full_match_bonus() {
        assert_eq!(
            real_full_mtp_acceptance(&[11, 12, 13], &[11, 12, 13, 14], true, 4).unwrap(),
            super::RealFullMtpAcceptance {
                accepted_draft_tokens: 3,
                terminal_target_index: 3,
                full_match_bonus: true,
            }
        );
        assert!(
            !real_full_mtp_acceptance(&[11, 12, 99], &[11, 12, 13, 14], true, 4)
                .unwrap()
                .full_match_bonus
        );
    }

    #[test]
    fn real_full_mtp_requires_one_target_fallback_sample() {
        let error = real_full_mtp_acceptance(&[11, 12], &[11, 12], true, 3).unwrap_err();
        assert!(error.to_string().contains("plus one fallback"));
    }

    #[test]
    fn real_full_mtp_only_pads_logical_d1_away_from_physical_m2() {
        assert_eq!(real_full_mtp_physical_padding_rows(0, false), 0);
        assert_eq!(real_full_mtp_physical_padding_rows(1, false), 1);
        for logical_draft_rows in 2..=7 {
            assert_eq!(
                real_full_mtp_physical_padding_rows(logical_draft_rows, false),
                0
            );
        }
        assert_eq!(real_full_mtp_physical_padding_rows(1, true), 0);
    }

    #[test]
    fn real_full_mtp_acceptance_clamps_fixed_width_tail_to_output_budget() {
        assert_eq!(
            real_full_mtp_acceptance(
                &[11, 12, 13, 14, 15, 16, 17],
                &[11, 12, 13, 14, 15, 16, 17, 18],
                true,
                6,
            )
            .unwrap(),
            super::RealFullMtpAcceptance {
                accepted_draft_tokens: 5,
                terminal_target_index: 5,
                full_match_bonus: false,
            }
        );
    }

    #[test]
    fn real_full_mtp_draft_budget_leaves_room_for_the_fallback_token() {
        assert_eq!(
            real_full_mtp_draft_tokens_after_cycle_with_limit(100, 0, 1, 6),
            6
        );
        assert_eq!(
            real_full_mtp_draft_tokens_after_cycle_with_limit(8, 1, 3, 6),
            3
        );
        assert_eq!(
            real_full_mtp_draft_tokens_after_cycle_with_limit(5, 3, 1, 6),
            0
        );
        assert_eq!(
            real_full_mtp_draft_tokens_after_cycle_with_limit(4, 4, 1, 6),
            0
        );
    }

    #[test]
    fn real_full_mtp_draft_policy_defaults_to_supported_adaptive_range() {
        let policy = real_full_mtp_draft_policy_from_values(None, None, None, false);
        assert_eq!(
            policy,
            super::RealFullMtpDraftPolicy {
                min: 1,
                max: 7,
                start: 6,
                adaptive: true,
            }
        );
    }

    #[test]
    fn legacy_mtp_draft_tokens_remain_a_fixed_width_override() {
        let policy = real_full_mtp_draft_policy_from_values(Some(4), None, None, false);
        assert_eq!(policy.min, 4);
        assert_eq!(policy.max, 4);
        assert_eq!(policy.start, 4);
        assert!(!policy.adaptive);
    }

    #[test]
    fn explicit_mtp_draft_bounds_enable_adaptation_and_force_startup_widths() {
        let policy = real_full_mtp_draft_policy_from_values(Some(4), Some(3), Some(6), true);
        assert_eq!(policy.min, 3);
        assert_eq!(policy.max, 6);
        assert_eq!(policy.start, 6);
        assert!(policy.adaptive);
        assert_eq!(
            real_full_mtp_startup_forced_draft_tokens(
                "real-full-startup-mtp-production-draft-5-adaptive-2049-sequence-0"
            ),
            Some(5)
        );
        assert_eq!(
            real_full_mtp_startup_forced_draft_tokens("external-agent-sequence"),
            None
        );
    }

    #[test]
    fn real_full_mtp_fixed_width_tail_requires_physical_context_headroom() {
        assert_eq!(
            real_full_mtp_draft_tokens_for_cycle_with_policy(25, 17, 1, 794, 131_072, 7, true),
            7,
        );
        assert_eq!(
            real_full_mtp_draft_tokens_for_cycle_with_policy(25, 17, 1, 794, 131_072, 7, false),
            6,
        );
        assert_eq!(
            real_full_mtp_draft_tokens_for_cycle_with_policy(25, 17, 1, 1_000, 1_024, 7, true),
            0,
        );
    }

    #[test]
    fn native_mtp_skips_generic_startup_sequences_only() {
        assert!(real_full_native_mtp_sequence_enabled(
            "external-agent-sequence"
        ));
        assert!(!real_full_native_mtp_sequence_enabled(
            "real-full-startup-capture-arena-final-4097-sequence-0"
        ));
        assert!(real_full_native_mtp_sequence_enabled(
            "real-full-startup-mtp-production-2049-sequence-0"
        ));
        assert!(real_full_native_mtp_sequence_enabled(
            "real-full-startup-mtp-production-bucket-1024-sequence-0"
        ));
    }

    #[test]
    fn real_full_request_token_rows_seed_initial_decode_from_last_prompt_token() {
        let request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            11,
            "split-initial-sequence",
            "prompt",
            5,
            1,
            Vec::new(),
            0,
            4,
        );

        let rows = real_full_request_token_rows(&request, Some(vec![10, 20, 30, 40, 50]))
            .expect("splitting initial token rows");

        assert_eq!(rows.prefix_tokens, 0);
        assert_eq!(rows.prefill_tokens, 4);
        assert_eq!(rows.prefill_token_ids, Some(vec![10, 20, 30, 40]));
        assert_eq!(rows.decode_token_ids, vec![50]);
    }

    #[test]
    fn real_full_request_token_rows_split_uncached_prompt_suffix() {
        let request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            13,
            "split-cached-prefix-sequence",
            "prompt",
            8,
            1,
            Vec::new(),
            1,
            2,
        )
        .with_cached_prompt_tokens(4);

        let rows =
            real_full_request_token_rows(&request, Some(vec![10, 20, 30, 40, 50, 60, 70, 80]))
                .expect("splitting uncached prompt suffix");

        assert_eq!(rows.prefix_tokens, 4);
        assert_eq!(rows.prefill_tokens, 3);
        assert_eq!(rows.prefill_token_ids, Some(vec![50, 60, 70]));
        assert_eq!(rows.decode_token_ids, vec![80]);
    }

    #[test]
    fn real_full_request_token_rows_seed_recurrent_decode_from_latest_generated_token() {
        let request = glmrt_api::RealFullRequest::new_decode_step_for_sequence(
            12,
            "split-recurrent-sequence",
            "prompt",
            5,
            1,
            vec![101, 102],
            1,
            4,
        );

        let rows =
            real_full_request_token_rows(&request, None).expect("splitting recurrent token rows");

        assert_eq!(rows.prefix_tokens, 6);
        assert_eq!(rows.prefill_tokens, 0);
        assert_eq!(rows.prefill_token_ids, None);
        assert_eq!(rows.decode_token_ids, vec![102]);
    }

    #[test]
    fn request_prompt_token_ids_tokenizes_prompt_with_loaded_tokenizer() {
        let snapshot = tempfile::tempdir().expect("creating tokenizer snapshot");
        fs::write(
            snapshot.path().join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"user: Use real full.":7},"unk_token":"[UNK]"}}"#,
        )
        .expect("writing tokenizer fixture");
        let tokenizer =
            LoadedTokenizer::from_snapshot(snapshot.path()).expect("loading tokenizer fixture");
        let request = glmrt_api::RealFullRequest::new_decode_step(
            9,
            "user: Use real full.",
            1,
            1,
            vec![42, 43],
            2,
            4,
        );

        let token_ids = request_prompt_token_ids(&tokenizer, &request)
            .expect("tokenizing request")
            .expect("token ids");

        assert_eq!(token_ids, vec![7]);
    }
}
