use std::fs;

use anyhow::{Context, Result};
use glmrt_core::TensorCatalog;
use serde::Serialize;

use super::dflash::{
    preload_dflash2_resident_weights, preload_dflash2_target_aliases,
    preloaded_dflash2_resident_weights, Dflash2Checkpoint, Dflash2ResidentPreloadStats,
    Dflash2TargetAliasPreloadStats, GLM53_DFLASH2, GLM53_DFLASH2_BLOCK_SIZE,
    GLM53_DFLASH2_MAX_DRAFTS,
};
use super::dflash_body::{dflash2_body_buffer_plan, Dflash2BodyConfig};
use super::dflash_head::{dflash2_head_buffer_plan, dflash2_topk_backend};
use super::dflash_static::{
    benchmark_dflash2_static_graph, Dflash2StaticBenchConfig, Dflash2StaticGraphReport,
};
use super::dflash_update::{dflash2_update_buffer_plan, Dflash2UpdateConfig};
use super::dspark_kv::DsparkKvStorage;
use crate::cli::DflashPreflightArgs;

const DFLASH2_PREFLIGHT_SCHEMA: &str = "glmrt-dflash2-preflight-v1";

#[derive(Debug, Serialize)]
struct Dflash2BufferReport {
    name: &'static str,
    bytes: usize,
}

#[derive(Debug, Serialize)]
struct Dflash2ConcurrencyPlan {
    active_requests: usize,
    accepted_rows_per_request: usize,
    update_rows: usize,
    query_rows_per_request: usize,
    proposal_tokens_per_request: usize,
    context_tokens: usize,
    body_kv_tokens: usize,
    total_physical_pages: usize,
    max_pages_per_request: usize,
    body_buffers: Vec<Dflash2BufferReport>,
    update_buffers: Vec<Dflash2BufferReport>,
    head_buffers: Vec<Dflash2BufferReport>,
    mutable_bytes_including_shared_kv_once: u64,
}

#[derive(Debug, Serialize)]
struct Dflash2PreflightReport {
    schema: &'static str,
    status: &'static str,
    checkpoint_repo_id: &'static str,
    checkpoint_revision: &'static str,
    checkpoint_config_sha256: &'static str,
    checkpoint_weight_lfs_sha256: &'static str,
    target_repo_id: &'static str,
    snapshot_path: String,
    tensor_count: usize,
    payload_bytes: u64,
    kv_storage: DsparkKvStorage,
    kv_element_bytes: usize,
    kv_capacity_tokens: usize,
    page_size: usize,
    proposal_tokens_per_request: usize,
    query_rows_per_request: usize,
    topk_backend: String,
    resident_preload: Option<Dflash2ResidentPreloadStats>,
    target_alias_preload: Option<Dflash2TargetAliasPreloadStats>,
    concurrency_plans: Vec<Dflash2ConcurrencyPlan>,
    static_graphs: Option<Vec<Dflash2StaticGraphReport>>,
}

pub(crate) fn run_dflash_preflight(args: DflashPreflightArgs) -> Result<()> {
    anyhow::ensure!(
        !args.capture_static || args.preload,
        "DFlash2 static capture requires --preload so all resident pointers remain stable"
    );
    anyhow::ensure!(
        matches!(args.max_concurrency, 1 | 2 | 4),
        "DFlash2 max concurrency must be 1, 2, or 4"
    );
    anyhow::ensure!(
        matches!(args.kv_page_size, 16 | 32 | 64 | 128),
        "DFlash2 KV page size must be 16, 32, 64, or 128"
    );
    anyhow::ensure!(
        args.accepted_rows_per_request > 0
            && args.accepted_rows_per_request <= GLM53_DFLASH2_BLOCK_SIZE,
        "DFlash2 accepted rows per request must be in 1..={} ",
        GLM53_DFLASH2_BLOCK_SIZE
    );
    anyhow::ensure!(
        (1..=GLM53_DFLASH2_MAX_DRAFTS).contains(&args.proposal_tokens_per_request),
        "DFlash2 proposal tokens per request must be in 1..={GLM53_DFLASH2_MAX_DRAFTS}"
    );
    let query_rows_per_request = args.proposal_tokens_per_request + 1;
    let topk_backend = dflash2_topk_backend()?;
    let kv_storage = DsparkKvStorage::parse(&args.kv_storage).with_context(|| {
        format!(
            "unknown DFlash2 KV storage {}; expected bf16 or fp8",
            args.kv_storage
        )
    })?;
    let target_catalog: TensorCatalog = serde_json::from_reader(
        fs::File::open(&args.target_catalog)
            .with_context(|| format!("opening {}", args.target_catalog.display()))?,
    )
    .with_context(|| format!("parsing {}", args.target_catalog.display()))?;
    let checkpoint = Dflash2Checkpoint::from_snapshot(&args.snapshot)?;

    let resident_preload = args
        .preload
        .then(|| preload_dflash2_resident_weights(&checkpoint))
        .transpose()?;
    let target_alias_preload = args
        .preload
        .then(|| preload_dflash2_target_aliases(&target_catalog))
        .transpose()?;
    let max_pages_per_request = args.kv_capacity_tokens.div_ceil(args.kv_page_size);
    let production_physical_pages = args
        .max_concurrency
        .checked_mul(max_pages_per_request)
        .context("DFlash2 production shared KV page count overflow")?;

    let static_graphs = args
        .capture_static
        .then(|| {
            let weights = preloaded_dflash2_resident_weights(&checkpoint, &target_catalog)?;
            [1, 2, 4]
                .into_iter()
                .filter(|active_requests| *active_requests <= args.max_concurrency)
                .map(|active_requests| {
                    benchmark_dflash2_static_graph(
                        weights,
                        Dflash2StaticBenchConfig {
                            active_requests,
                            // Serving captures one-row base executors and dispatches every
                            // larger commit through the packed update registry. Keep the GPU
                            // preflight identical to that live geometry; the CLI value still
                            // controls the independent capacity plans below.
                            accepted_rows_per_request: 1,
                            proposal_tokens_per_request: args.proposal_tokens_per_request,
                            context_tokens: args.context_tokens,
                            kv_capacity_tokens: args.kv_capacity_tokens,
                            allocate_full_kv_capacity: true,
                            capture_page_buckets: false,
                            page_size: args.kv_page_size,
                            kv_storage,
                            warmup: args.static_warmup,
                            iterations: args.static_iterations,
                            repeats: args.static_repeats,
                            seed: 20_260_829 + active_requests as i64,
                        },
                        Some(production_physical_pages),
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;

    let mut concurrency_plans = Vec::new();
    for active_requests in [1, 2, 4]
        .into_iter()
        .filter(|active_requests| *active_requests <= args.max_concurrency)
    {
        let update_rows = active_requests
            .checked_mul(args.accepted_rows_per_request)
            .context("DFlash2 update row count overflow")?;
        let body_kv_tokens = args
            .context_tokens
            .checked_add(args.accepted_rows_per_request)
            .and_then(|tokens| tokens.checked_add(query_rows_per_request))
            .context("DFlash2 body KV length overflow")?;
        anyhow::ensure!(
            body_kv_tokens <= args.kv_capacity_tokens,
            "DFlash2 context/update/query length {body_kv_tokens} exceeds KV capacity {}",
            args.kv_capacity_tokens
        );
        let body = dflash2_body_buffer_plan(Dflash2BodyConfig {
            active_requests,
            query_rows_per_request,
            total_pages: production_physical_pages,
            page_size: args.kv_page_size,
            max_pages_per_request,
            planning_pages_per_request: max_pages_per_request,
            fixed_split_pages: 0,
            kv_storage,
            seed: 20_260_829 + active_requests as i64,
            initialize_input: false,
            initialize_kv: false,
        })?;
        let update = dflash2_update_buffer_plan(Dflash2UpdateConfig {
            rows: update_rows,
            active_requests,
            total_pages: production_physical_pages,
            page_size: args.kv_page_size,
            max_pages_per_request,
            kv_storage,
            seed: 20_260_829 + active_requests as i64,
            initialize_target_hidden: false,
            initialize_kv: false,
        })?;
        let head = dflash2_head_buffer_plan(active_requests, args.proposal_tokens_per_request)?;
        let body_buffers = body
            .iter()
            .map(|item| Dflash2BufferReport {
                name: item.name,
                bytes: item.bytes,
            })
            .collect::<Vec<_>>();
        let update_buffers = update
            .iter()
            .map(|item| Dflash2BufferReport {
                name: item.name,
                bytes: item.bytes,
            })
            .collect::<Vec<_>>();
        let head_buffers = head
            .iter()
            .map(|item| Dflash2BufferReport {
                name: item.name,
                bytes: item.bytes,
            })
            .collect::<Vec<_>>();
        let shared_kv_bytes = body
            .iter()
            .filter(|item| matches!(item.name, "k_cache" | "v_cache"))
            .try_fold(0_u64, |bytes, item| {
                bytes
                    .checked_add(item.bytes as u64)
                    .context("DFlash2 shared KV byte count overflow")
            })?;
        let non_kv_body_bytes = body
            .iter()
            .filter(|item| !matches!(item.name, "k_cache" | "v_cache"))
            .map(|item| item.bytes as u64)
            .sum::<u64>();
        let non_kv_update_bytes = update
            .iter()
            .filter(|item| !matches!(item.name, "k_cache" | "v_cache"))
            .map(|item| item.bytes as u64)
            .sum::<u64>();
        let head_bytes = head.iter().map(|item| item.bytes as u64).sum::<u64>();
        let mutable_bytes_including_shared_kv_once = shared_kv_bytes
            .checked_add(non_kv_body_bytes)
            .and_then(|bytes| bytes.checked_add(non_kv_update_bytes))
            .and_then(|bytes| bytes.checked_add(head_bytes))
            .context("DFlash2 mutable buffer total overflow")?;
        concurrency_plans.push(Dflash2ConcurrencyPlan {
            active_requests,
            accepted_rows_per_request: args.accepted_rows_per_request,
            update_rows,
            query_rows_per_request,
            proposal_tokens_per_request: args.proposal_tokens_per_request,
            context_tokens: args.context_tokens,
            body_kv_tokens,
            total_physical_pages: production_physical_pages,
            max_pages_per_request,
            body_buffers,
            update_buffers,
            head_buffers,
            mutable_bytes_including_shared_kv_once,
        });
    }

    let report = Dflash2PreflightReport {
        schema: DFLASH2_PREFLIGHT_SCHEMA,
        status: "accepted",
        checkpoint_repo_id: GLM53_DFLASH2.repo_id,
        checkpoint_revision: GLM53_DFLASH2.revision,
        checkpoint_config_sha256: GLM53_DFLASH2.config_sha256,
        checkpoint_weight_lfs_sha256: GLM53_DFLASH2.weight_lfs_sha256,
        target_repo_id: GLM53_DFLASH2.target_repo_id,
        snapshot_path: args.snapshot.display().to_string(),
        tensor_count: checkpoint.weights.residency.len(),
        payload_bytes: checkpoint.weights.payload_bytes,
        kv_storage,
        kv_element_bytes: kv_storage.element_bytes(),
        kv_capacity_tokens: args.kv_capacity_tokens,
        page_size: args.kv_page_size,
        proposal_tokens_per_request: args.proposal_tokens_per_request,
        query_rows_per_request,
        topk_backend,
        resident_preload,
        target_alias_preload,
        concurrency_plans,
        static_graphs,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
