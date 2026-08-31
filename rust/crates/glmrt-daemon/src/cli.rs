use clap::{Args, Parser, Subcommand};
use glmrt_core::DEFAULT_MODEL_ID;
use std::path::PathBuf;

pub(crate) const DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS: usize = 128 * 1024;
const DEFAULT_DFLASH2_KV_CAPACITY_TOKENS: usize = 2_176;

#[derive(Debug, Parser)]
#[command(name = "glmrt", about = "GLMRT phase0 runtime CLI")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    Doctor(DoctorArgs),
    InspectModel(InspectModelArgs),
    MakeLoadplan(MakeLoadPlanArgs),
    LoadTensors(LoadTensorsArgs),
    Tokenize(TokenizeArgs),
    DsparkPreflight(DsparkPreflightArgs),
    DflashPreflight(DflashPreflightArgs),
    Coordinator(CoordinatorArgs),
    Expertd(ExpertDaemonArgs),
    BenchRdma(BenchRdmaArgs),
    BenchRdmaRing(BenchRdmaRingArgs),
    BenchCudaKernels(BenchCudaKernelsArgs),
    BenchProtocolV2Tcp(BenchProtocolV2TcpArgs),
    BenchExpertReductionReplay(BenchExpertReductionReplayArgs),
    TransportCapabilities(TransportCapabilitiesArgs),
    SchedulerSmoke(SchedulerSmokeArgs),
    SchedulerRowAudit(SchedulerRowAuditArgs),
}

#[derive(Debug, Args)]
pub(crate) struct DflashPreflightArgs {
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    #[arg(long)]
    pub(crate) target_catalog: PathBuf,
    #[arg(long, default_value_t = DEFAULT_DFLASH2_KV_CAPACITY_TOKENS)]
    pub(crate) kv_capacity_tokens: usize,
    #[arg(long, default_value_t = 4)]
    pub(crate) max_concurrency: usize,
    #[arg(long, default_value = "bf16")]
    pub(crate) kv_storage: String,
    #[arg(long, default_value_t = 64)]
    pub(crate) kv_page_size: usize,
    #[arg(long, default_value_t = 1_024)]
    pub(crate) context_tokens: usize,
    #[arg(long, default_value_t = 4)]
    pub(crate) accepted_rows_per_request: usize,
    #[arg(long, default_value_t = 7)]
    pub(crate) proposal_tokens_per_request: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) preload: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_static: bool,
    #[arg(long, default_value_t = 2)]
    pub(crate) static_warmup: usize,
    #[arg(long, default_value_t = 10)]
    pub(crate) static_iterations: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) static_repeats: usize,
}

#[derive(Debug, Args)]
pub(crate) struct DsparkPreflightArgs {
    #[arg(long, default_value = "redhat")]
    pub(crate) fixture: String,
    #[arg(long)]
    pub(crate) snapshot: PathBuf,
    #[arg(long)]
    pub(crate) target_catalog: PathBuf,
    #[arg(long, default_value_t = DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS)]
    pub(crate) kv_capacity_tokens: usize,
    #[arg(long, default_value_t = 4)]
    pub(crate) max_concurrency: usize,
    #[arg(long, default_value = "bf16")]
    pub(crate) kv_storage: String,
    #[arg(long, default_value_t = 64)]
    pub(crate) kv_page_size: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) preload: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_attention: bool,
    #[arg(long, default_value_t = 1_024)]
    pub(crate) attention_context_tokens: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) attention_warmup: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) attention_iterations: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) attention_repeats: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_body: bool,
    #[arg(long, default_value_t = 1_024)]
    pub(crate) body_context_tokens: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) body_warmup: usize,
    #[arg(long, default_value_t = 50)]
    pub(crate) body_iterations: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) body_repeats: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_head: bool,
    #[arg(long, default_value_t = 5)]
    pub(crate) head_warmup: usize,
    #[arg(long, default_value_t = 50)]
    pub(crate) head_iterations: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) head_repeats: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_query: bool,
    #[arg(long, default_value_t = 5)]
    pub(crate) query_warmup: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) query_iterations: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) query_repeats: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_update: bool,
    #[arg(long, default_value_t = 1_024)]
    pub(crate) update_context_tokens: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) update_warmup: usize,
    #[arg(long, default_value_t = 50)]
    pub(crate) update_iterations: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) update_repeats: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) capture_static: bool,
    #[arg(long, default_value_t = 1_024)]
    pub(crate) static_context_tokens: usize,
    #[arg(long, default_value_t = 4)]
    pub(crate) static_accepted_rows_per_request: usize,
    #[arg(long, default_value_t = 2)]
    pub(crate) static_warmup: usize,
    #[arg(long, default_value_t = 10)]
    pub(crate) static_iterations: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) static_repeats: usize,
}

#[derive(Debug, Args)]
pub(crate) struct DoctorArgs {
    #[arg(long, default_value = "coordinator")]
    pub(crate) role: String,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) json: bool,
}

#[derive(Debug, Args)]
pub(crate) struct InspectModelArgs {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) out: PathBuf,
    #[arg(long)]
    pub(crate) summary: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct MakeLoadPlanArgs {
    #[arg(long)]
    pub(crate) catalog: PathBuf,
    #[arg(long, default_value = "modulo")]
    pub(crate) policy: String,
    #[arg(long, default_value = "spark-0,spark-1,spark-2,spark-3")]
    pub(crate) hosts: String,
    #[arg(long)]
    pub(crate) out: PathBuf,
}

#[derive(Debug, Args)]
pub(crate) struct LoadTensorsArgs {
    #[arg(long)]
    pub(crate) catalog: PathBuf,
    #[arg(long)]
    pub(crate) summary: PathBuf,
    #[arg(long = "tensor")]
    pub(crate) tensors: Vec<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) verify_hashes: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TokenizeArgs {
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) hf_home: Option<PathBuf>,
    #[arg(long)]
    pub(crate) text: String,
    #[arg(long, default_value_t = false)]
    pub(crate) add_special_tokens: bool,
}

#[derive(Debug, Args)]
pub(crate) struct CoordinatorArgs {
    #[arg(long, default_value = "tiny")]
    pub(crate) backend: String,
    #[arg(long, default_value = "inproc")]
    pub(crate) transport: String,
    #[arg(long, default_value = "bf16")]
    pub(crate) kv_cache_dtype: String,
    #[arg(long, default_value_t = DEFAULT_REAL_FULL_MAX_CONTEXT_TOKENS)]
    pub(crate) max_context_tokens: usize,
    #[arg(long, default_value = "127.0.0.1:8000")]
    pub(crate) listen: String,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long, default_value = "spark-0,spark-1,spark-2,spark-3")]
    pub(crate) expert_hosts: String,
    #[arg(long)]
    pub(crate) catalog: Option<PathBuf>,
    #[arg(long)]
    pub(crate) loadplan: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) preflight_only: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ExpertDaemonArgs {
    #[arg(long, default_value_t = false)]
    pub(crate) synthetic_weights: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) preflight_only: bool,
    #[arg(long, default_value = "tcp")]
    pub(crate) transport: String,
    #[arg(long, default_value = "0.0.0.0:9100")]
    pub(crate) listen: String,
    #[arg(long)]
    pub(crate) loadplan: Option<PathBuf>,
    #[arg(long)]
    pub(crate) catalog: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    pub(crate) model_id: String,
    #[arg(long)]
    pub(crate) real_layer: Option<u32>,
    #[arg(long = "role", visible_alias = "role-hostname")]
    pub(crate) role_hostname: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaArgs {
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long, default_value = "auto")]
    pub(crate) mode: String,
    #[arg(long, default_value_t = 18515)]
    pub(crate) port: u16,
    #[arg(long, default_value = "4096,8192,12288,16384,32768,65536")]
    pub(crate) payload_bytes: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) duration_secs: u64,
}

#[derive(Debug, Args)]
pub(crate) struct BenchRdmaRingArgs {
    #[arg(long, default_value = "server")]
    pub(crate) mode: String,
    #[arg(long, default_value = "0.0.0.0:18525")]
    pub(crate) listen: String,
    #[arg(long)]
    pub(crate) peer: Option<String>,
    #[arg(long)]
    pub(crate) peers: Option<String>,
    #[arg(long, default_value_t = 16 * 1024)]
    pub(crate) slot_bytes: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) depth: usize,
    #[arg(long, default_value_t = 100)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) window: usize,
    #[arg(long)]
    pub(crate) request_bytes: Option<usize>,
    #[arg(long)]
    pub(crate) response_bytes: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub(crate) compute_delay_us: u64,
    #[arg(long, default_value = "unspecified")]
    pub(crate) network_label: String,
    #[arg(long, default_value_t = false)]
    pub(crate) gpu_echo: bool,
    #[arg(long, default_value = "fp8")]
    pub(crate) wire_codec: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) rows: usize,
    #[arg(long, default_value_t = 6144)]
    pub(crate) row_width: usize,
    #[arg(long, default_value_t = 1000)]
    pub(crate) kernel_iterations: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) reduction_rank: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) reduction_world_size: usize,
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long, default_value_t = 30_000)]
    pub(crate) timeout_ms: u64,
}

#[derive(Debug, Args)]
pub(crate) struct BenchCudaKernelsArgs {
    #[arg(long)]
    pub(crate) native_lib: Option<PathBuf>,
    #[arg(long = "kernel", value_delimiter = ',')]
    pub(crate) kernels: Vec<String>,
    #[arg(long, default_value_t = 16)]
    pub(crate) rows: usize,
    #[arg(long, default_value_t = 1024)]
    pub(crate) hidden_dim: usize,
    #[arg(long, default_value_t = 2048)]
    pub(crate) intermediate_dim: usize,
    #[arg(long, default_value_t = 1024)]
    pub(crate) output_dim: usize,
    #[arg(long, default_value_t = 4096)]
    pub(crate) vocab: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) routes: usize,
    #[arg(long, default_value_t = 8)]
    pub(crate) top_k: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 10)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = false)]
    pub(crate) require_cuda: bool,
}

#[derive(Debug, Args, Clone)]
pub(crate) struct BenchProtocolV2TcpArgs {
    #[arg(long)]
    pub(crate) addr: String,
    #[arg(long, default_value = "tcp")]
    pub(crate) transport: String,
    #[arg(long, default_value = "spark-tcp-expert")]
    pub(crate) target: String,
    #[arg(long, default_value_t = 1)]
    pub(crate) request_id_start: u64,
    #[arg(long, default_value_t = 75)]
    pub(crate) hops: usize,
    #[arg(long, default_value_t = 5)]
    pub(crate) iterations: usize,
    #[arg(long, default_value_t = 3)]
    pub(crate) large_iterations: usize,
    #[arg(long, default_value_t = 0)]
    pub(crate) warmup_iterations: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) warmup_rows: usize,
    #[arg(long)]
    pub(crate) warmup_timeout_ms: Option<u64>,
    #[arg(long, default_value_t = false)]
    pub(crate) warmup_only: bool,
    /// Stop after the independent round-trip measurements.
    #[arg(long, default_value_t = false)]
    pub(crate) roundtrip_only: bool,
    #[arg(long, default_value = "1,2,4,8,16,64,256,512")]
    pub(crate) roundtrip_rows: String,
    #[arg(long, default_value = "1,2,3,4,5,6,8")]
    pub(crate) mtp_chain_rows: String,
    #[arg(long, default_value = "16,32,64,128,256,512")]
    pub(crate) prefill_roundtrip_rows: String,
    #[arg(long, default_value = "16,32,256,512")]
    pub(crate) prefill_chain_rows: String,
    #[arg(long, default_value_t = 3)]
    pub(crate) layer_id: u32,
    #[arg(long, default_value_t = 0)]
    pub(crate) expert_id: u32,
    /// Comma-separated expert ID pattern, cycled across all generated routes.
    #[arg(long)]
    pub(crate) expert_ids: Option<String>,
    #[arg(long, default_value_t = 1)]
    pub(crate) routes_per_row: usize,
    /// Use the production single-row Spark-owner TP4 request and response codecs.
    #[arg(long, default_value_t = false)]
    pub(crate) spark_owner_decode: bool,
    /// Use production NVFP4 ingress and row-scaled FP8 responses without Spark reduction.
    #[arg(long, default_value_t = false)]
    pub(crate) nvfp4_fp8_roundtrip: bool,
    #[arg(long, default_value_t = false)]
    pub(crate) layer_block: bool,
    #[arg(long, default_value_t = 1)]
    pub(crate) layer_block_sequence_id: u64,
    #[arg(long)]
    pub(crate) expected_executor: Option<String>,
    #[arg(long, default_value_t = false)]
    pub(crate) require_expected_executor: bool,
    #[arg(long, default_value_t = 5000)]
    pub(crate) timeout_ms: u64,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    pub(crate) max_frame_bytes: usize,
}

#[derive(Debug, Args)]
pub(crate) struct BenchExpertReductionReplayArgs {
    /// Plain JSONL replay plan produced by plan_expert_reduction_replay.py.
    #[arg(long)]
    pub(crate) plan: PathBuf,
    /// Destination JSONL. The benchmark refuses to overwrite an existing file.
    #[arg(long)]
    pub(crate) output: PathBuf,
    #[arg(
        long,
        default_value = "ostrich=10.55.0.1:9100,dodo=10.55.0.2:9100,emu=10.55.0.3:9100,kiwi=10.55.0.4:9100"
    )]
    pub(crate) expert_hosts: String,
    #[arg(long, default_value = "semantic")]
    pub(crate) cohort: String,
    #[arg(long, default_value_t = 2)]
    pub(crate) warmup_chains_per_m: usize,
    /// Measure only the production coordinator-reduction path.  This avoids
    /// requiring workers launched for the distinct row-sharded protocol.
    #[arg(long, default_value_t = false)]
    pub(crate) coordinator_only: bool,
    #[arg(long, default_value_t = 30_000)]
    pub(crate) timeout_ms: u64,
}

#[derive(Debug, Args)]
pub(crate) struct TransportCapabilitiesArgs {
    #[arg(long)]
    pub(crate) benchmark_jsonl: Option<PathBuf>,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Debug, Args)]
pub(crate) struct SchedulerSmokeArgs {
    #[arg(long, default_value_t = 512)]
    pub(crate) prefill_tokens: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) chunk_tokens: usize,
    #[arg(long, default_value_t = 32)]
    pub(crate) decode_arrivals: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) decode_period_iterations: usize,
    #[arg(long, default_value_t = 16)]
    pub(crate) max_prefill_tokens_per_iteration: usize,
    #[arg(long, default_value_t = 1)]
    pub(crate) max_active_prefill_chunks: usize,
}

#[derive(Debug, Args)]
pub(crate) struct SchedulerRowAuditArgs {
    #[arg(long = "input")]
    pub(crate) inputs: Vec<PathBuf>,
    #[arg(long = "input-list")]
    pub(crate) input_lists: Vec<PathBuf>,
    #[arg(long, default_value_t = 1)]
    pub(crate) next_window_count: usize,
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}
