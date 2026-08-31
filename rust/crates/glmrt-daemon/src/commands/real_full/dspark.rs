#![allow(dead_code)]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
use glmrt_loader::{
    read_safetensors_metadata, read_tensor_bytes_into, LoadedTensorSummary,
    SafetensorsTensorMetadata,
};
use serde::{Deserialize, Serialize};

#[path = "dflash2_cost_profile.rs"]
mod dflash2_cost_profile;
use dflash2_cost_profile::{
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_MODEL,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_REVISION, GLM53_EXL3_K4_DFLASH2_COST_PROFILE_ID,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_CONCURRENCY,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_DRAFTS, GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MS,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_POWER_LIMIT_WATTS,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SOURCE_SHA256,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SPARKINFER_REVISION,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_MODEL,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_REVISION,
    GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TOPOLOGY,
};
#[path = "dspark_cost_profile.rs"]
mod dspark_cost_profile;
use dspark_cost_profile::{
    GLM52_REDHAT_DSPARK_COST_PROFILE_DSPARK_REVISION, GLM52_REDHAT_DSPARK_COST_PROFILE_ID,
    GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_CONCURRENCY, GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_DRAFTS,
    GLM52_REDHAT_DSPARK_COST_PROFILE_MS, GLM52_REDHAT_DSPARK_COST_PROFILE_POWER_LIMIT_WATTS,
    GLM52_REDHAT_DSPARK_COST_PROFILE_SOURCE_SHA256,
    GLM52_REDHAT_DSPARK_COST_PROFILE_SPARKINFER_REVISION,
    GLM52_REDHAT_DSPARK_COST_PROFILE_TARGET_MODEL,
    GLM52_REDHAT_DSPARK_COST_PROFILE_TARGET_REVISION, GLM52_REDHAT_DSPARK_COST_PROFILE_TOPOLOGY,
};

// The GLMRT fork adds two otherwise-missing SM121 register-table entries used
// only by the EXL3 K3 forced tiles. It does not change any specialization
// reachable by the calibrated NVFP4 profile, so that profile remains valid on
// this exact reviewed descendant. Any later SparkInfer revision must be
// requalified explicitly rather than inheriting compatibility transitively.
const GLM52_REDHAT_DSPARK_COST_PROFILE_GLMRT_EXL3_COMPATIBLE_SPARKINFER_REVISION: &str =
    "28e083482fd18ca3ce0e2553cd533102be85552f";

use super::coordinator_kernels::{
    preload_resident_weight_from_host_staging, preload_resident_weight_from_host_staging_profiled,
    preloaded_resident_weight_device_buffer, preloaded_resident_weight_device_buffer_view,
    DeviceBf16Output,
};
use super::dspark_attention::{
    benchmark_dspark_paged_attention_graph, DsparkPagedAttentionBenchConfig,
    DsparkPagedAttentionGraphReport,
};
use super::dspark_body::{
    benchmark_dspark_body_graph, DsparkBodyBenchConfig, DsparkBodyGraphReport,
    DsparkBodyLayerResidentWeights, DsparkBodyResidentWeights,
};
use super::dspark_head::{
    benchmark_dspark_head_graph, DsparkHeadBenchConfig, DsparkHeadGraphReport,
    DsparkHeadResidentWeights,
};
use super::dspark_kv::DsparkKvStorage;
use super::dspark_query::{
    benchmark_dspark_query_graph, DsparkQueryBenchConfig, DsparkQueryGraphReport,
    DsparkQueryResidentWeights,
};
use super::dspark_static::{
    benchmark_dspark_static_graph, DsparkDraftStep, DsparkStaticBenchConfig, DsparkStaticExecutor,
    DsparkStaticGraphReport, DsparkStaticResidentWeights,
};
use super::dspark_update::{
    benchmark_dspark_update_graph, DsparkUpdateBenchConfig, DsparkUpdateGraphReport,
    DsparkUpdateLayerResidentWeights, DsparkUpdateResidentWeights,
};
use crate::cli::DsparkPreflightArgs;

const GLM52_DSPARK_HIDDEN_SIZE: usize = 6_144;
const GLM52_DSPARK_INTERMEDIATE_SIZE: usize = 12_288;
const GLM52_DSPARK_VOCAB_SIZE: usize = 154_880;
const GLM52_DSPARK_TARGET_TAPS: usize = 5;
const GLM52_DSPARK_MAX_DRAFT_LAYERS: usize = 5;
const GLM52_DSPARK_HEAD_DIM: usize = 64;
const GLM52_DSPARK_ATTENTION_HEADS: usize = 64;
const GLM52_DSPARK_MARKOV_RANK: usize = 256;
const GLM52_DSPARK_MASK_TOKEN_ID: usize = 154_856;
const GLM52_DSPARK_EMBEDDING_SHA256: &str =
    "ee39119948cef0a062268af11c228acc825734d3044daec6069fe8721b340bee";
const GLM52_DSPARK_LM_HEAD_SHA256: &str =
    "a012be05e7716292407d418b408222de256d4dbe2fe2143a44d27d8e3553bfba";
const GLM52_TARGET_EMBEDDING_WEIGHT: &str = "model.embed_tokens.weight";
const GLM52_TARGET_LM_HEAD_WEIGHT: &str = "lm_head.weight";
const DSPARK_FLASHINFER_WORKSPACE_BYTES: u64 = 128 * 1024 * 1024;
const SIRO_GLM52_DSPARK_AUX_HIDDEN_STATE_LAYER_IDS: [usize; GLM52_DSPARK_TARGET_TAPS] =
    [8, 23, 39, 55, 70];
const REDHAT_GLM52_DSPARK_AUX_HIDDEN_STATE_LAYER_IDS: [usize; GLM52_DSPARK_TARGET_TAPS] =
    [2, 20, 39, 58, 75];
static ACTIVE_DSPARK_TARGET_TAPS: OnceLock<[usize; GLM52_DSPARK_TARGET_TAPS]> = OnceLock::new();
static ACTIVE_DSPARK_MAX_VERIFY_DRAFTS: OnceLock<usize> = OnceLock::new();
static ACTIVE_DSPARK_CONFIDENCE_CONTEXT_PRIOR: OnceLock<DsparkConfidenceContextPrior> =
    OnceLock::new();

pub(super) fn dspark_target_hidden_tap_layer_ids() -> [usize; GLM52_DSPARK_TARGET_TAPS] {
    ACTIVE_DSPARK_TARGET_TAPS
        .get()
        .copied()
        .unwrap_or(SIRO_GLM52_DSPARK_AUX_HIDDEN_STATE_LAYER_IDS)
}

pub(super) fn dspark_active_max_verify_drafts() -> usize {
    ACTIVE_DSPARK_MAX_VERIFY_DRAFTS
        .get()
        .copied()
        .unwrap_or(SIRO_GLM52_DSPARK_PREVIEW.max_verify_drafts)
}

fn activate_dspark_contract(fixture: DsparkPinnedFixture) -> Result<()> {
    if let Some(active) = ACTIVE_DSPARK_TARGET_TAPS.get() {
        anyhow::ensure!(
            *active == fixture.aux_hidden_state_layer_ids,
            "dSpark target taps are already active as {:?}, cannot switch to {:?} in one process",
            active,
            fixture.aux_hidden_state_layer_ids,
        );
    } else {
        let _ = ACTIVE_DSPARK_TARGET_TAPS.set(fixture.aux_hidden_state_layer_ids);
    }
    if let Some(active) = ACTIVE_DSPARK_MAX_VERIFY_DRAFTS.get() {
        anyhow::ensure!(
            *active == fixture.max_verify_drafts,
            "dSpark maximum verify width is already active as {active}, cannot switch to {} in one process",
            fixture.max_verify_drafts,
        );
    } else {
        let _ = ACTIVE_DSPARK_MAX_VERIFY_DRAFTS.set(fixture.max_verify_drafts);
    }
    if let Some(active) = ACTIVE_DSPARK_CONFIDENCE_CONTEXT_PRIOR.get() {
        anyhow::ensure!(
            *active == fixture.confidence_context_prior,
            "dSpark confidence context prior is already active as {active:?}, cannot switch to {:?} in one process",
            fixture.confidence_context_prior,
        );
    } else {
        let _ = ACTIVE_DSPARK_CONFIDENCE_CONTEXT_PRIOR.set(fixture.confidence_context_prior);
    }
    eprintln!(
        "real_full_dspark_confidence_context_prior revision={} start_tokens={} ramp_tokens={} limit_logit={:.3}",
        fixture.revision,
        fixture.confidence_context_prior.start_tokens,
        fixture.confidence_context_prior.ramp_tokens,
        fixture.confidence_context_prior.limit(),
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum DsparkCheckpointConvention {
    /// Original DeepSpec: N query rows produce N proposals, including slot zero.
    DeepSpecAnchorFirst,
    /// `speculators`: one bonus anchor followed by N proposal-bearing mask rows.
    SpeculatorsBonusAnchor,
}

impl DsparkCheckpointConvention {
    fn query_rows(self, proposal_tokens: usize) -> Result<usize> {
        anyhow::ensure!(proposal_tokens > 0, "dSpark requires at least one proposal");
        match self {
            Self::DeepSpecAnchorFirst => Ok(proposal_tokens),
            Self::SpeculatorsBonusAnchor => proposal_tokens
                .checked_add(1)
                .context("dSpark query row count overflow"),
        }
    }

    fn proposal_query_slot(self, proposal_index: usize, proposal_tokens: usize) -> Result<usize> {
        anyhow::ensure!(
            proposal_index < proposal_tokens,
            "dSpark proposal index {proposal_index} is outside 0..{proposal_tokens}"
        );
        Ok(match self {
            Self::DeepSpecAnchorFirst => proposal_index,
            Self::SpeculatorsBonusAnchor => proposal_index + 1,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DsparkQueryTokenKind {
    Anchor,
    Mask,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DsparkQueryLayout {
    convention: DsparkCheckpointConvention,
    query_token_kinds: Vec<DsparkQueryTokenKind>,
    proposal_query_slots: Vec<usize>,
    proposal_position_offsets: Vec<usize>,
}

impl DsparkQueryLayout {
    fn new(convention: DsparkCheckpointConvention, proposal_tokens: usize) -> Result<Self> {
        let query_rows = convention.query_rows(proposal_tokens)?;
        let mut query_token_kinds = vec![DsparkQueryTokenKind::Mask; query_rows];
        query_token_kinds[0] = DsparkQueryTokenKind::Anchor;
        let proposal_query_slots = (0..proposal_tokens)
            .map(|index| convention.proposal_query_slot(index, proposal_tokens))
            .collect::<Result<Vec<_>>>()?;
        let proposal_position_offsets = (1..=proposal_tokens).collect();
        Ok(Self {
            convention,
            query_token_kinds,
            proposal_query_slots,
            proposal_position_offsets,
        })
    }

    fn query_rows(&self) -> usize {
        self.query_token_kinds.len()
    }

    fn proposal_tokens(&self) -> usize {
        self.proposal_query_slots.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct DsparkConfidenceContextPrior {
    start_tokens: usize,
    ramp_tokens: usize,
    limit_millilogits: i16,
}

impl DsparkConfidenceContextPrior {
    const FLAT: Self = Self {
        start_tokens: 0,
        ramp_tokens: 1,
        limit_millilogits: 0,
    };

    fn limit(self) -> f64 {
        f64::from(self.limit_millilogits) / 1_000.0
    }

    fn at_context(self, context_tokens: usize) -> f64 {
        let ramp_progress = context_tokens
            .saturating_sub(self.start_tokens)
            .min(self.ramp_tokens);
        self.limit() * ramp_progress as f64 / self.ramp_tokens as f64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct DsparkPinnedFixture {
    repo_id: &'static str,
    revision: &'static str,
    verifier_repo_id: &'static str,
    verifier_revision: Option<&'static str>,
    draft_layers: usize,
    aux_hidden_state_layer_ids: [usize; GLM52_DSPARK_TARGET_TAPS],
    proposal_tokens: usize,
    max_verify_drafts: usize,
    confidence_context_prior: DsparkConfidenceContextPrior,
    convention: DsparkCheckpointConvention,
    native_sliding_window: Option<usize>,
    weight_bytes: u64,
    weight_payload_bytes: u64,
    tensor_count: usize,
}

const REDHAT_GLM52_DSPARK: DsparkPinnedFixture = DsparkPinnedFixture {
    repo_id: "RedHatAI/GLM-5.2-speculator.dspark",
    revision: "8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4",
    verifier_repo_id: "RedHatAI/GLM-5.2-NVFP4-FP8",
    verifier_revision: None,
    draft_layers: 3,
    aux_hidden_state_layer_ids: REDHAT_GLM52_DSPARK_AUX_HIDDEN_STATE_LAYER_IDS,
    proposal_tokens: 8,
    max_verify_drafts: 7,
    confidence_context_prior: DsparkConfidenceContextPrior::FLAT,
    convention: DsparkCheckpointConvention::DeepSpecAnchorFirst,
    native_sliding_window: Some(2_048),
    weight_bytes: 6_305_465_978,
    weight_payload_bytes: 6_305_461_506,
    tensor_count: 42,
};

const SIRO_GLM52_DSPARK_PREVIEW: DsparkPinnedFixture = DsparkPinnedFixture {
    repo_id: "siro1/glm-5.2-dspark-preview",
    revision: "7ff03018b3a443bfb9fca166739bd5f37ee5908b",
    verifier_repo_id: "nvidia/GLM-5.2-NVFP4",
    verifier_revision: Some("aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"),
    draft_layers: 5,
    aux_hidden_state_layer_ids: SIRO_GLM52_DSPARK_AUX_HIDDEN_STATE_LAYER_IDS,
    proposal_tokens: 15,
    max_verify_drafts: 15,
    confidence_context_prior: DsparkConfidenceContextPrior {
        start_tokens: 16 * 1024,
        ramp_tokens: 32 * 1024,
        limit_millilogits: -800,
    },
    convention: DsparkCheckpointConvention::SpeculatorsBonusAnchor,
    native_sliding_window: None,
    weight_bytes: 7_614_140_882,
    weight_payload_bytes: 7_614_134_018,
    tensor_count: 64,
};

fn production_dspark_fixture_for_snapshot(snapshot: &Path) -> Result<DsparkPinnedFixture> {
    let revision = snapshot.file_name().and_then(|name| name.to_str());
    [REDHAT_GLM52_DSPARK, SIRO_GLM52_DSPARK_PREVIEW]
        .into_iter()
        .find(|fixture| revision == Some(fixture.revision))
        .with_context(|| {
            format!(
                "unsupported production dSpark snapshot revision {:?}; expected {} or {}",
                revision, REDHAT_GLM52_DSPARK.revision, SIRO_GLM52_DSPARK_PREVIEW.revision
            )
        })
}

#[derive(Debug, Deserialize)]
struct DsparkCheckpointConfig {
    architectures: Vec<String>,
    aux_hidden_state_layer_ids: Vec<usize>,
    block_size: usize,
    confidence_head_with_markov: bool,
    draft_vocab_size: usize,
    enable_confidence_head: bool,
    markov_head_type: String,
    markov_rank: usize,
    mask_token_id: usize,
    #[serde(default)]
    sample_from_anchor: bool,
    speculators_model_type: String,
    speculators_config: DsparkSpeculatorsConfig,
    transformer_layer_config: DsparkTransformerConfig,
}

#[derive(Debug, Deserialize)]
struct DsparkSpeculatorsConfig {
    proposal_methods: Vec<DsparkProposalConfig>,
    verifier: DsparkVerifierConfig,
}

#[derive(Debug, Deserialize)]
struct DsparkProposalConfig {
    speculative_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct DsparkVerifierConfig {
    name_or_path: String,
}

#[derive(Debug, Deserialize)]
struct DsparkTransformerConfig {
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    #[serde(default)]
    layer_types: Vec<String>,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    vocab_size: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ValidatedDsparkCheckpoint {
    fixture: DsparkPinnedFixture,
    query_layout: DsparkQueryLayout,
    aux_hidden_state_layer_ids: [usize; GLM52_DSPARK_TARGET_TAPS],
}

impl ValidatedDsparkCheckpoint {
    fn from_config_json(fixture: DsparkPinnedFixture, config_json: &str) -> Result<Self> {
        let config: DsparkCheckpointConfig =
            serde_json::from_str(config_json).context("parsing dSpark config.json")?;
        anyhow::ensure!(
            config.architectures == ["DSparkDraftModel"],
            "dSpark architecture mismatch: {:?}",
            config.architectures
        );
        anyhow::ensure!(
            config.speculators_model_type == "dspark",
            "dSpark model type mismatch: {}",
            config.speculators_model_type
        );
        anyhow::ensure!(
            config.aux_hidden_state_layer_ids == fixture.aux_hidden_state_layer_ids,
            "dSpark hidden taps mismatch: {:?}",
            config.aux_hidden_state_layer_ids
        );
        anyhow::ensure!(
            config.draft_vocab_size == GLM52_DSPARK_VOCAB_SIZE
                && config.transformer_layer_config.vocab_size == GLM52_DSPARK_VOCAB_SIZE,
            "dSpark vocabulary mismatch"
        );
        anyhow::ensure!(
            config.markov_head_type == "vanilla" && config.markov_rank == GLM52_DSPARK_MARKOV_RANK,
            "dSpark Markov head mismatch: type={} rank={}",
            config.markov_head_type,
            config.markov_rank
        );
        anyhow::ensure!(
            config.enable_confidence_head && config.confidence_head_with_markov,
            "dSpark checkpoint must include the Markov-conditioned confidence head"
        );
        anyhow::ensure!(
            config.mask_token_id == GLM52_DSPARK_MASK_TOKEN_ID,
            "dSpark mask token mismatch: {}",
            config.mask_token_id
        );
        let transformer = &config.transformer_layer_config;
        anyhow::ensure!(
            transformer.hidden_size == GLM52_DSPARK_HIDDEN_SIZE
                && transformer.intermediate_size == GLM52_DSPARK_INTERMEDIATE_SIZE
                && transformer.num_hidden_layers == fixture.draft_layers
                && transformer.head_dim == GLM52_DSPARK_HEAD_DIM
                && transformer.num_attention_heads == GLM52_DSPARK_ATTENTION_HEADS
                && transformer.num_key_value_heads == GLM52_DSPARK_ATTENTION_HEADS,
            "dSpark transformer geometry mismatch"
        );
        let expected_layer_type = if fixture.native_sliding_window.is_some() {
            "sliding_attention"
        } else {
            "full_attention"
        };
        anyhow::ensure!(
            transformer.layer_types == vec![expected_layer_type.to_owned(); fixture.draft_layers]
                && transformer.sliding_window == fixture.native_sliding_window,
            "dSpark attention contract mismatch for {}: layer_types={:?} sliding_window={:?}",
            fixture.repo_id,
            transformer.layer_types,
            transformer.sliding_window,
        );
        anyhow::ensure!(
            config.speculators_config.proposal_methods.len() == 1,
            "dSpark checkpoint must define exactly one proposal method"
        );
        let proposal_tokens = config.speculators_config.proposal_methods[0].speculative_tokens;
        anyhow::ensure!(
            proposal_tokens == fixture.proposal_tokens,
            "dSpark proposal count mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.proposal_tokens,
            proposal_tokens
        );
        anyhow::ensure!(
            dspark_verifier_reference_matches(
                fixture,
                &config.speculators_config.verifier.name_or_path
            ),
            "dSpark verifier mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.verifier_repo_id,
            config.speculators_config.verifier.name_or_path
        );
        let query_layout = DsparkQueryLayout::new(fixture.convention, proposal_tokens)?;
        anyhow::ensure!(
            config.sample_from_anchor
                == matches!(
                    fixture.convention,
                    DsparkCheckpointConvention::DeepSpecAnchorFirst
                ),
            "dSpark sample_from_anchor mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            matches!(
                fixture.convention,
                DsparkCheckpointConvention::DeepSpecAnchorFirst
            ),
            config.sample_from_anchor
        );
        anyhow::ensure!(
            config.block_size == query_layout.query_rows(),
            "dSpark block/query mismatch for {}: config block {}, layout rows {}",
            fixture.repo_id,
            config.block_size,
            query_layout.query_rows()
        );
        Ok(Self {
            fixture,
            query_layout,
            aux_hidden_state_layer_ids: fixture.aux_hidden_state_layer_ids,
        })
    }

    fn from_snapshot(fixture: DsparkPinnedFixture, snapshot: &Path) -> Result<Self> {
        anyhow::ensure!(
            snapshot.file_name().and_then(|name| name.to_str()) == Some(fixture.revision),
            "dSpark snapshot revision mismatch for {}: expected {} at {}",
            fixture.repo_id,
            fixture.revision,
            snapshot.display()
        );
        let config_json = fs::read_to_string(snapshot.join("config.json"))
            .with_context(|| format!("reading dSpark config from {}", snapshot.display()))?;
        let weight_bytes = fs::metadata(snapshot.join("model.safetensors"))
            .with_context(|| format!("reading dSpark weights from {}", snapshot.display()))?
            .len();
        anyhow::ensure!(
            weight_bytes == fixture.weight_bytes,
            "dSpark weight size mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.weight_bytes,
            weight_bytes
        );
        Self::from_config_json(fixture, &config_json)
    }
}

fn dspark_verifier_reference_matches(fixture: DsparkPinnedFixture, observed: &str) -> bool {
    if observed == fixture.verifier_repo_id {
        return true;
    }
    let Some(revision) = fixture.verifier_revision else {
        return false;
    };
    let cache_model = format!("models--{}", fixture.verifier_repo_id.replace('/', "--"));
    Path::new(observed).ends_with(Path::new(&cache_model).join("snapshots").join(revision))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DsparkResidentWeightKind {
    DraftOwned,
    TargetAlias,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DsparkResidentWeightPlan {
    source_name: String,
    resident_name: String,
    kind: DsparkResidentWeightKind,
    dtype: DType,
    shape: Vec<usize>,
    byte_length: u64,
    expected_sha256: Option<&'static str>,
}

#[derive(Clone, Debug)]
struct DsparkWeightManifest {
    catalog: TensorCatalog,
    residency: Vec<DsparkResidentWeightPlan>,
    payload_bytes: u64,
    aliased_bytes: u64,
    draft_owned_bytes: u64,
}

impl DsparkWeightManifest {
    fn from_snapshot(fixture: DsparkPinnedFixture, snapshot: &Path) -> Result<Self> {
        let weight_path = snapshot.join("model.safetensors");
        let metadata = read_safetensors_metadata(&weight_path).with_context(|| {
            format!(
                "reading dSpark safetensors metadata from {}",
                weight_path.display()
            )
        })?;
        Self::from_metadata(fixture, snapshot, metadata)
    }

    fn from_metadata(
        fixture: DsparkPinnedFixture,
        snapshot: &Path,
        metadata: Vec<SafetensorsTensorMetadata>,
    ) -> Result<Self> {
        let expected = expected_dspark_tensor_shapes(fixture.draft_layers);
        anyhow::ensure!(
            expected.len() == fixture.tensor_count,
            "internal dSpark manifest has {} tensors, expected {}",
            expected.len(),
            fixture.tensor_count
        );
        anyhow::ensure!(
            metadata.len() == fixture.tensor_count,
            "dSpark tensor count mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.tensor_count,
            metadata.len()
        );
        let actual = metadata
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor))
            .collect::<BTreeMap<_, _>>();
        let unexpected = actual
            .keys()
            .filter(|name| !expected.contains_key(**name))
            .copied()
            .collect::<Vec<_>>();
        anyhow::ensure!(
            unexpected.is_empty(),
            "unexpected dSpark tensors for {}: {unexpected:?}",
            fixture.repo_id
        );

        let mut tensors = Vec::with_capacity(metadata.len());
        let mut residency = Vec::with_capacity(metadata.len());
        let mut payload_bytes = 0_u64;
        let mut aliased_bytes = 0_u64;
        for (name, shape) in expected {
            let tensor = actual
                .get(name.as_str())
                .with_context(|| format!("missing dSpark tensor {name} for {}", fixture.repo_id))?;
            anyhow::ensure!(
                tensor.dtype == DType::Bf16,
                "dSpark tensor {name} for {} must be BF16, got {:?}",
                fixture.repo_id,
                tensor.dtype
            );
            anyhow::ensure!(
                tensor.shape == shape,
                "dSpark tensor {name} shape mismatch for {}: expected {:?}, got {:?}",
                fixture.repo_id,
                shape,
                tensor.shape
            );
            let expected_bytes = checked_bf16_tensor_bytes(&shape)
                .with_context(|| format!("computing dSpark tensor {name} byte length"))?;
            anyhow::ensure!(
                tensor.byte_length == expected_bytes,
                "dSpark tensor {name} byte mismatch for {}: expected {}, got {}",
                fixture.repo_id,
                expected_bytes,
                tensor.byte_length
            );
            payload_bytes = payload_bytes
                .checked_add(tensor.byte_length)
                .context("dSpark payload byte count overflow")?;
            let (kind, resident_name, expected_sha256) = dspark_resident_binding(fixture, &name);
            if kind == DsparkResidentWeightKind::TargetAlias {
                aliased_bytes = aliased_bytes
                    .checked_add(tensor.byte_length)
                    .context("dSpark aliased byte count overflow")?;
            }
            residency.push(DsparkResidentWeightPlan {
                source_name: name.clone(),
                resident_name,
                kind,
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                byte_length: tensor.byte_length,
                expected_sha256,
            });
            tensors.push(TensorInfo {
                name: name.clone(),
                file: "model.safetensors".to_owned(),
                dtype: tensor.dtype.clone(),
                shape: tensor.shape.clone(),
                byte_offset: tensor.byte_offset,
                byte_length: tensor.byte_length,
                role: dspark_tensor_role(&name),
                layer_id: dspark_layer_id(&name),
                expert_id: None,
                is_quantization_metadata: false,
            });
        }
        anyhow::ensure!(
            payload_bytes == fixture.weight_payload_bytes,
            "dSpark payload mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.weight_payload_bytes,
            payload_bytes
        );
        let draft_owned_bytes = payload_bytes
            .checked_sub(aliased_bytes)
            .context("dSpark aliased bytes exceed payload bytes")?;
        Ok(Self {
            catalog: TensorCatalog {
                model_id: fixture.repo_id.to_owned(),
                snapshot_path: snapshot.display().to_string(),
                facts: ModelFacts {
                    model_id: fixture.repo_id.to_owned(),
                    hidden_size: GLM52_DSPARK_HIDDEN_SIZE,
                    num_hidden_layers: fixture.draft_layers,
                    first_k_dense_replace: fixture.draft_layers,
                    routed_experts: 0,
                    top_k: 0,
                    quantization_recipe: "bf16".to_owned(),
                },
                tensors,
            },
            residency,
            payload_bytes,
            aliased_bytes,
            draft_owned_bytes,
        })
    }

    fn validate_target_aliases(&self, target: &TensorCatalog) -> Result<()> {
        for binding in self
            .residency
            .iter()
            .filter(|binding| binding.kind == DsparkResidentWeightKind::TargetAlias)
        {
            let target_tensor = target
                .tensors
                .iter()
                .find(|tensor| tensor.name == binding.resident_name)
                .with_context(|| {
                    format!(
                        "dSpark shared tensor {} cannot alias missing target tensor {}",
                        binding.source_name, binding.resident_name
                    )
                })?;
            anyhow::ensure!(
                target_tensor.dtype == binding.dtype
                    && target_tensor.shape == binding.shape
                    && target_tensor.byte_length == binding.byte_length,
                "dSpark shared tensor {} is incompatible with target tensor {}",
                binding.source_name,
                binding.resident_name
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct DsparkCheckpoint {
    validated: ValidatedDsparkCheckpoint,
    weights: DsparkWeightManifest,
}

impl DsparkCheckpoint {
    fn from_snapshot(fixture: DsparkPinnedFixture, snapshot: &Path) -> Result<Self> {
        let validated = ValidatedDsparkCheckpoint::from_snapshot(fixture, snapshot)?;
        let weights = DsparkWeightManifest::from_snapshot(fixture, snapshot)?;
        Ok(Self { validated, weights })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct DsparkStaticEngineConfig {
    /// Total draft KV slots shared by all active requests.
    kv_capacity_tokens: usize,
    kv_page_size: usize,
    max_concurrency: usize,
    kv_storage: DsparkKvStorage,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DsparkGraphBucketPlan {
    active_requests: usize,
    draft_query_rows: usize,
    proposal_rows: usize,
    target_verification_rows: usize,
    hidden_tap_bytes: u64,
    hidden_ping_pong_bytes: u64,
    attention_output_bytes: u64,
    qkv_scratch_bytes: u64,
    mlp_gate_up_scratch_bytes: u64,
    lm_logits_scratch_bytes: u64,
    token_confidence_bytes: u64,
    paged_attention_metadata_bytes: u64,
    reusable_scratch_bytes: u64,
    peak_dynamic_bytes: u64,
}

impl DsparkGraphBucketPlan {
    fn new(
        checkpoint: &DsparkCheckpoint,
        active_requests: usize,
        kv_capacity_pages: usize,
    ) -> Result<Self> {
        let query_rows = checkpoint.validated.query_layout.query_rows();
        let proposal_tokens = checkpoint.validated.query_layout.proposal_tokens();
        let draft_query_rows = active_requests
            .checked_mul(query_rows)
            .context("dSpark draft query row count overflow")?;
        let proposal_rows = active_requests
            .checked_mul(proposal_tokens)
            .context("dSpark proposal row count overflow")?;
        let target_verification_rows = draft_query_rows;
        let hidden_tap_bytes = checked_buffer_bytes(
            target_verification_rows,
            GLM52_DSPARK_TARGET_TAPS * GLM52_DSPARK_HIDDEN_SIZE,
            std::mem::size_of::<u16>(),
            "dSpark hidden tap buffer",
        )?;
        let one_hidden_buffer = checked_buffer_bytes(
            draft_query_rows,
            GLM52_DSPARK_HIDDEN_SIZE,
            std::mem::size_of::<u16>(),
            "dSpark hidden buffer",
        )?;
        let hidden_ping_pong_bytes = one_hidden_buffer
            .checked_mul(2)
            .context("dSpark hidden ping-pong byte count overflow")?;
        let attention_width = GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM;
        let attention_output_bytes = checked_buffer_bytes(
            draft_query_rows,
            attention_width,
            std::mem::size_of::<u16>(),
            "dSpark attention output",
        )?;
        let qkv_scratch_bytes = checked_buffer_bytes(
            draft_query_rows,
            3 * attention_width,
            std::mem::size_of::<u16>(),
            "dSpark QKV scratch",
        )?;
        let mlp_gate_up_scratch_bytes = checked_buffer_bytes(
            draft_query_rows,
            2 * GLM52_DSPARK_INTERMEDIATE_SIZE,
            std::mem::size_of::<u16>(),
            "dSpark gate/up scratch",
        )?;
        let lm_logits_scratch_bytes = checked_buffer_bytes(
            proposal_rows,
            GLM52_DSPARK_VOCAB_SIZE,
            std::mem::size_of::<u16>(),
            "dSpark LM logits scratch",
        )?;
        let token_confidence_bytes = checked_buffer_bytes(
            proposal_rows,
            1,
            std::mem::size_of::<u64>() + std::mem::size_of::<f32>(),
            "dSpark token/confidence output",
        )?;
        let paged_attention_metadata_bytes =
            dspark_paged_attention_metadata_bytes(active_requests, kv_capacity_pages)?;
        let reusable_scratch_bytes = qkv_scratch_bytes
            .max(mlp_gate_up_scratch_bytes)
            .max(lm_logits_scratch_bytes);
        let peak_dynamic_bytes = hidden_tap_bytes
            .checked_add(hidden_ping_pong_bytes)
            .and_then(|bytes| bytes.checked_add(attention_output_bytes))
            .and_then(|bytes| bytes.checked_add(reusable_scratch_bytes))
            .and_then(|bytes| bytes.checked_add(token_confidence_bytes))
            .and_then(|bytes| bytes.checked_add(paged_attention_metadata_bytes))
            .context("dSpark graph bucket dynamic byte count overflow")?;
        Ok(Self {
            active_requests,
            draft_query_rows,
            proposal_rows,
            target_verification_rows,
            hidden_tap_bytes,
            hidden_ping_pong_bytes,
            attention_output_bytes,
            qkv_scratch_bytes,
            mlp_gate_up_scratch_bytes,
            lm_logits_scratch_bytes,
            token_confidence_bytes,
            paged_attention_metadata_bytes,
            reusable_scratch_bytes,
            peak_dynamic_bytes,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DsparkStaticEnginePlan {
    fixture: DsparkPinnedFixture,
    config: DsparkStaticEngineConfig,
    graph_buckets: Vec<DsparkGraphBucketPlan>,
    draft_owned_weight_bytes: u64,
    target_aliased_weight_bytes: u64,
    draft_kv_pages: usize,
    draft_kv_padded_tokens: usize,
    draft_kv_bytes: u64,
    flashinfer_workspace_bytes: u64,
    max_dynamic_bytes: u64,
    peak_incremental_device_bytes: u64,
    cold_capture_requires_python: bool,
    hot_replay_python_calls: usize,
    serving_dispatch_enabled: bool,
}

impl DsparkStaticEnginePlan {
    fn new(
        checkpoint: &DsparkCheckpoint,
        target: &TensorCatalog,
        config: DsparkStaticEngineConfig,
    ) -> Result<Self> {
        anyhow::ensure!(
            config.kv_capacity_tokens > 0,
            "dSpark KV capacity must be positive"
        );
        anyhow::ensure!(
            config.max_concurrency > 0,
            "dSpark max concurrency must be positive"
        );
        anyhow::ensure!(
            matches!(config.kv_page_size, 16 | 32 | 64 | 128),
            "dSpark KV page size must be one of 16, 32, 64, or 128"
        );
        checkpoint.weights.validate_target_aliases(target)?;
        let minimum_query_slots = config
            .max_concurrency
            .checked_mul(checkpoint.validated.query_layout.query_rows())
            .context("dSpark minimum query capacity overflow")?;
        anyhow::ensure!(
            config.kv_capacity_tokens >= minimum_query_slots,
            "dSpark KV capacity {} cannot hold one {}-row query for {} requests",
            config.kv_capacity_tokens,
            checkpoint.validated.query_layout.query_rows(),
            config.max_concurrency
        );

        let draft_kv_pages = config.kv_capacity_tokens.div_ceil(config.kv_page_size);
        let draft_kv_padded_tokens = draft_kv_pages
            .checked_mul(config.kv_page_size)
            .context("dSpark padded KV capacity overflow")?;
        let graph_buckets = dspark_concurrency_buckets(config.max_concurrency)
            .into_iter()
            .map(|concurrency| DsparkGraphBucketPlan::new(checkpoint, concurrency, draft_kv_pages))
            .collect::<Result<Vec<_>>>()?;
        let max_dynamic_bytes = graph_buckets
            .iter()
            .map(|bucket| bucket.peak_dynamic_bytes)
            .max()
            .context("dSpark graph plan produced no buckets")?;
        let draft_kv_bytes: u64 = dspark_context_kv_bytes(
            draft_kv_padded_tokens,
            checkpoint.validated.fixture.draft_layers,
            config.kv_storage.element_bytes(),
        )?
        .try_into()
        .context("dSpark KV byte count does not fit in u64")?;
        let peak_incremental_device_bytes = checkpoint
            .weights
            .draft_owned_bytes
            .checked_add(draft_kv_bytes)
            .and_then(|bytes| bytes.checked_add(DSPARK_FLASHINFER_WORKSPACE_BYTES))
            .and_then(|bytes| bytes.checked_add(max_dynamic_bytes))
            .context("dSpark incremental device byte count overflow")?;
        Ok(Self {
            fixture: checkpoint.validated.fixture,
            config,
            graph_buckets,
            draft_owned_weight_bytes: checkpoint.weights.draft_owned_bytes,
            target_aliased_weight_bytes: checkpoint.weights.aliased_bytes,
            draft_kv_pages,
            draft_kv_padded_tokens,
            draft_kv_bytes,
            flashinfer_workspace_bytes: DSPARK_FLASHINFER_WORKSPACE_BYTES,
            max_dynamic_bytes,
            peak_incremental_device_bytes,
            cold_capture_requires_python: true,
            hot_replay_python_calls: 0,
            serving_dispatch_enabled: false,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
struct DsparkResidentPreloadStats {
    selected_source_tensors: usize,
    selected_resident_buffers: usize,
    selected_bytes: u64,
    loaded_source_tensors: usize,
    loaded_resident_buffers: usize,
    loaded_bytes: u64,
    source_read_micros: u128,
    total_elapsed_micros: u128,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
struct DsparkTargetAliasPreloadStats {
    selected_bytes: u64,
    loaded: bool,
    loaded_bytes: u64,
    source_read_micros: u128,
    total_elapsed_micros: u128,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DsparkDraftResidentGroup {
    resident_name: String,
    source_names: Vec<String>,
    shape: Vec<usize>,
    byte_length: u64,
}

fn dspark_draft_resident_groups(
    checkpoint: &DsparkCheckpoint,
) -> Result<Vec<DsparkDraftResidentGroup>> {
    let bindings = checkpoint
        .weights
        .residency
        .iter()
        .filter(|binding| binding.kind == DsparkResidentWeightKind::DraftOwned)
        .map(|binding| (binding.source_name.as_str(), binding))
        .collect::<BTreeMap<_, _>>();
    let mut consumed = BTreeSet::new();
    let mut groups = Vec::new();
    for layer in 0..checkpoint.validated.fixture.draft_layers {
        for (suffix, source_suffixes, shape) in [
            (
                "self_attn.qkv_proj.weight",
                vec![
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                vec![
                    3 * GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM,
                    GLM52_DSPARK_HIDDEN_SIZE,
                ],
            ),
            (
                "mlp.gate_up_proj.weight",
                vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                vec![2 * GLM52_DSPARK_INTERMEDIATE_SIZE, GLM52_DSPARK_HIDDEN_SIZE],
            ),
        ] {
            let source_names = source_suffixes
                .into_iter()
                .map(|source_suffix| format!("layers.{layer}.{source_suffix}"))
                .collect::<Vec<_>>();
            let byte_length = source_names.iter().try_fold(0_u64, |bytes, source_name| {
                let binding = bindings.get(source_name.as_str()).with_context(|| {
                    format!("missing dSpark fused resident source {source_name}")
                })?;
                consumed.insert(source_name.clone());
                bytes
                    .checked_add(binding.byte_length)
                    .context("dSpark fused resident byte count overflow")
            })?;
            anyhow::ensure!(
                byte_length == checked_bf16_tensor_bytes(&shape)?,
                "dSpark fused resident layers.{layer}.{suffix} byte count mismatch"
            );
            groups.push(DsparkDraftResidentGroup {
                resident_name: format!(
                    "dspark:{}:layers.{layer}.{suffix}",
                    checkpoint.validated.fixture.revision
                ),
                source_names,
                shape,
                byte_length,
            });
        }
    }
    for binding in bindings.values() {
        if consumed.contains(&binding.source_name) {
            continue;
        }
        groups.push(DsparkDraftResidentGroup {
            resident_name: binding.resident_name.clone(),
            source_names: vec![binding.source_name.clone()],
            shape: binding.shape.clone(),
            byte_length: binding.byte_length,
        });
    }
    groups.sort_by(|left, right| left.resident_name.cmp(&right.resident_name));
    let source_count: usize = groups.iter().map(|group| group.source_names.len()).sum();
    let resident_bytes: u64 = groups.iter().map(|group| group.byte_length).sum();
    anyhow::ensure!(
        source_count == checkpoint.validated.fixture.tensor_count - 2
            && resident_bytes == checkpoint.weights.draft_owned_bytes,
        "dSpark grouped resident plan mismatch: sources {source_count} bytes {resident_bytes}"
    );
    Ok(groups)
}

fn preload_dspark_draft_owned_weights(
    checkpoint: &DsparkCheckpoint,
) -> Result<DsparkResidentPreloadStats> {
    let started = Instant::now();
    let mut stats = DsparkResidentPreloadStats::default();
    let bindings = checkpoint
        .weights
        .residency
        .iter()
        .map(|binding| (binding.source_name.as_str(), binding))
        .collect::<BTreeMap<_, _>>();
    let groups = dspark_draft_resident_groups(checkpoint)?;
    let mut device_allocation_ms = 0.0_f64;
    let mut staging_allocation_ms = 0.0_f64;
    let mut staging_fill_ms = 0.0_f64;
    let mut h2d_ms = 0.0_f64;
    stats.selected_source_tensors = groups.iter().map(|group| group.source_names.len()).sum();
    stats.selected_resident_buffers = groups.len();
    stats.selected_bytes = groups.iter().map(|group| group.byte_length).sum();
    for group in groups {
        let expected_bytes: usize = group.byte_length.try_into().with_context(|| {
            format!(
                "dSpark resident {} byte length {} does not fit in usize",
                group.resident_name, group.byte_length
            )
        })?;
        let mut loaded = Vec::<LoadedTensorSummary>::new();
        let timing = preload_resident_weight_from_host_staging_profiled(
            &group.resident_name,
            expected_bytes,
            "startup resident dSpark draft weight pinned staging",
            |staging| {
                let mut offset = 0_usize;
                for source_name in &group.source_names {
                    let binding = bindings.get(source_name.as_str()).with_context(|| {
                        format!("missing dSpark resident source binding {source_name}")
                    })?;
                    let source_bytes: usize =
                        binding.byte_length.try_into().with_context(|| {
                            format!("dSpark source {source_name} byte count does not fit in usize")
                        })?;
                    let end = offset
                        .checked_add(source_bytes)
                        .context("dSpark fused staging offset overflow")?;
                    let summary = read_tensor_bytes_into(
                        &checkpoint.weights.catalog,
                        source_name,
                        &mut staging[offset..end],
                    )
                    .with_context(|| format!("reading dSpark tensor {source_name}"))?;
                    validate_dspark_loaded_tensor(binding, &summary)?;
                    loaded.push(summary);
                    offset = end;
                }
                anyhow::ensure!(
                    offset == expected_bytes,
                    "dSpark resident {} staged {} bytes, expected {}",
                    group.resident_name,
                    offset,
                    expected_bytes
                );
                Ok(())
            },
        )
        .with_context(|| format!("preloading dSpark resident {}", group.resident_name))?;
        device_allocation_ms += timing.device_allocation_ms;
        staging_allocation_ms += timing.staging_allocation_ms;
        staging_fill_ms += timing.staging_fill_ms;
        h2d_ms += timing.h2d_ms;
        preloaded_resident_weight_device_buffer(&group.resident_name, expected_bytes)
            .with_context(|| format!("verifying dSpark resident {}", group.resident_name))?;
        if !loaded.is_empty() {
            stats.loaded_resident_buffers += 1;
            stats.loaded_source_tensors += loaded.len();
        }
        for summary in loaded {
            stats.loaded_bytes = stats
                .loaded_bytes
                .checked_add(summary.bytes_read)
                .context("dSpark loaded resident byte count overflow")?;
            stats.source_read_micros = stats
                .source_read_micros
                .checked_add(summary.elapsed_micros)
                .context("dSpark source read time overflow")?;
        }
    }
    anyhow::ensure!(
        stats.selected_source_tensors == checkpoint.validated.fixture.tensor_count - 2
            && stats.selected_bytes == checkpoint.weights.draft_owned_bytes,
        "dSpark draft preload selection mismatch: sources {} bytes {}",
        stats.selected_source_tensors,
        stats.selected_bytes
    );
    stats.total_elapsed_micros = started.elapsed().as_micros();
    let total_ms = stats.total_elapsed_micros as f64 / 1_000.0;
    let source_read_ms = stats.source_read_micros as f64 / 1_000.0;
    let unattributed_ms =
        (total_ms - device_allocation_ms - staging_allocation_ms - staging_fill_ms - h2d_ms)
            .max(0.0);
    eprintln!(
        "real_full_dspark_draft_preload_detail groups={} sources={} bytes={} total_ms={total_ms:.3} source_read_ms={source_read_ms:.3} device_allocation_ms={device_allocation_ms:.3} staging_allocation_ms={staging_allocation_ms:.3} staging_fill_ms={staging_fill_ms:.3} h2d_ms={h2d_ms:.3} unattributed_ms={unattributed_ms:.3}",
        stats.selected_resident_buffers,
        stats.selected_source_tensors,
        stats.loaded_bytes,
    );
    Ok(stats)
}

fn preload_dspark_head_lm_alias(
    checkpoint: &DsparkCheckpoint,
) -> Result<DsparkTargetAliasPreloadStats> {
    preload_dspark_target_alias(
        checkpoint,
        "lm_head.weight",
        GLM52_TARGET_LM_HEAD_WEIGHT,
        "startup resident dSpark target LM-head alias pinned staging",
        "LM-head",
    )
}

fn preload_dspark_query_embedding_alias(
    checkpoint: &DsparkCheckpoint,
) -> Result<DsparkTargetAliasPreloadStats> {
    preload_dspark_target_alias(
        checkpoint,
        "embed_tokens.weight",
        GLM52_TARGET_EMBEDDING_WEIGHT,
        "startup resident dSpark target embedding alias pinned staging",
        "embedding",
    )
}

fn preload_dspark_target_alias(
    checkpoint: &DsparkCheckpoint,
    source_name: &'static str,
    resident_name: &'static str,
    staging_label: &'static str,
    description: &'static str,
) -> Result<DsparkTargetAliasPreloadStats> {
    let started = Instant::now();
    let binding = checkpoint
        .weights
        .residency
        .iter()
        .find(|binding| binding.source_name == source_name)
        .with_context(|| format!("missing dSpark {description} alias binding"))?;
    anyhow::ensure!(
        binding.kind == DsparkResidentWeightKind::TargetAlias
            && binding.resident_name == resident_name,
        "dSpark {description} alias binding changed"
    );
    let expected_bytes: usize = binding
        .byte_length
        .try_into()
        .with_context(|| format!("dSpark {description} byte count does not fit usize"))?;
    let mut loaded = None::<LoadedTensorSummary>;
    preload_resident_weight_from_host_staging(
        &binding.resident_name,
        expected_bytes,
        staging_label,
        |staging| {
            let summary =
                read_tensor_bytes_into(&checkpoint.weights.catalog, &binding.source_name, staging)
                    .with_context(|| format!("reading dSpark target {description} alias"))?;
            validate_dspark_loaded_tensor(binding, &summary)?;
            loaded = Some(summary);
            Ok(())
        },
    )
    .with_context(|| format!("preloading dSpark target {description} alias"))?;
    preloaded_resident_weight_device_buffer(&binding.resident_name, expected_bytes)
        .with_context(|| format!("binding dSpark target {description} alias"))?;
    let (was_loaded, loaded_bytes, source_read_micros) = loaded
        .map(|summary| (true, summary.bytes_read, summary.elapsed_micros))
        .unwrap_or((false, 0, 0));
    Ok(DsparkTargetAliasPreloadStats {
        selected_bytes: binding.byte_length,
        loaded: was_loaded,
        loaded_bytes,
        source_read_micros,
        total_elapsed_micros: started.elapsed().as_micros(),
    })
}

fn preloaded_dspark_query_weights() -> Result<DsparkQueryResidentWeights> {
    let embedding_bytes = GLM52_DSPARK_VOCAB_SIZE
        .checked_mul(GLM52_DSPARK_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(2))
        .context("dSpark embedding byte count overflow")?;
    let embedding =
        preloaded_resident_weight_device_buffer(GLM52_TARGET_EMBEDDING_WEIGHT, embedding_bytes)
            .context("binding preloaded target embedding alias for dSpark")?;
    Ok(DsparkQueryResidentWeights { embedding })
}

fn preloaded_dspark_body_weights(
    checkpoint: &DsparkCheckpoint,
) -> Result<DsparkBodyResidentWeights> {
    let groups = dspark_draft_resident_groups(checkpoint)?;
    let revision = checkpoint.validated.fixture.revision;
    let resident = |suffix: &str| -> Result<glmrt_ffi::GlmrtDeviceBuffer> {
        let name = format!("dspark:{revision}:{suffix}");
        let group = groups
            .iter()
            .find(|group| group.resident_name == name)
            .with_context(|| format!("missing dSpark body resident plan {name}"))?;
        let expected_bytes: usize = group
            .byte_length
            .try_into()
            .with_context(|| format!("dSpark body resident {name} is too large"))?;
        preloaded_resident_weight_device_buffer(&name, expected_bytes)
            .with_context(|| format!("binding preloaded dSpark body resident {name}"))
    };

    let final_norm = resident("norm.weight")?;
    let active_layers = checkpoint.validated.fixture.draft_layers;
    let mut layers = Vec::with_capacity(GLM52_DSPARK_MAX_DRAFT_LAYERS);
    for layer in 0..active_layers {
        layers.push(DsparkBodyLayerResidentWeights {
            input_norm: resident(&format!("layers.{layer}.input_layernorm.weight"))?,
            post_norm: resident(&format!("layers.{layer}.post_attention_layernorm.weight"))?,
            q_norm: resident(&format!("layers.{layer}.self_attn.q_norm.weight"))?,
            k_norm: resident(&format!("layers.{layer}.self_attn.k_norm.weight"))?,
            qkv: resident(&format!("layers.{layer}.self_attn.qkv_proj.weight"))?,
            output: resident(&format!("layers.{layer}.self_attn.o_proj.weight"))?,
            gate_up: resident(&format!("layers.{layer}.mlp.gate_up_proj.weight"))?,
            down: resident(&format!("layers.{layer}.mlp.down_proj.weight"))?,
        });
    }
    let filler = *layers
        .last()
        .context("dSpark body checkpoint has no transformer layers")?;
    layers.resize(GLM52_DSPARK_MAX_DRAFT_LAYERS, filler);
    let layers: [DsparkBodyLayerResidentWeights; GLM52_DSPARK_MAX_DRAFT_LAYERS] = layers
        .try_into()
        .map_err(|_| anyhow::anyhow!("dSpark body resident layer count changed"))?;
    let resident_bytes = final_norm
        .bytes
        .checked_add(
            layers
                .iter()
                .take(active_layers)
                .flat_map(|layer| {
                    [
                        layer.input_norm,
                        layer.post_norm,
                        layer.q_norm,
                        layer.k_norm,
                        layer.qkv,
                        layer.output,
                        layer.gate_up,
                        layer.down,
                    ]
                })
                .try_fold(0_usize, |bytes, buffer| bytes.checked_add(buffer.bytes))
                .context("dSpark body resident byte count overflow")?,
        )
        .context("dSpark body resident byte count overflow")?
        .try_into()
        .context("dSpark body resident byte count does not fit u64")?;
    Ok(DsparkBodyResidentWeights {
        final_norm,
        layers,
        active_layers,
        resident_bytes,
    })
}

fn preloaded_dspark_head_weights(
    checkpoint: &DsparkCheckpoint,
) -> Result<DsparkHeadResidentWeights> {
    let groups = dspark_draft_resident_groups(checkpoint)?;
    let revision = checkpoint.validated.fixture.revision;
    let resident = |suffix: &str| -> Result<glmrt_ffi::GlmrtDeviceBuffer> {
        let name = format!("dspark:{revision}:{suffix}");
        let group = groups
            .iter()
            .find(|group| group.resident_name == name)
            .with_context(|| format!("missing dSpark head resident plan {name}"))?;
        let expected_bytes: usize = group
            .byte_length
            .try_into()
            .with_context(|| format!("dSpark head resident {name} is too large"))?;
        preloaded_resident_weight_device_buffer(&name, expected_bytes)
            .with_context(|| format!("binding preloaded dSpark head resident {name}"))
    };
    let lm_head_bytes = GLM52_DSPARK_VOCAB_SIZE
        .checked_mul(GLM52_DSPARK_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(2))
        .context("dSpark LM-head byte count overflow")?;
    let lm_head =
        preloaded_resident_weight_device_buffer(GLM52_TARGET_LM_HEAD_WEIGHT, lm_head_bytes)
            .context("binding preloaded target LM-head alias for dSpark")?;
    let markov_w1 = resident("markov_head.markov_w1.weight")?;
    let markov_w2 = resident("markov_head.markov_w2.weight")?;
    let confidence_weight = resident("confidence_head.proj.weight")?;
    let confidence_bias = resident("confidence_head.proj.bias")?;
    let resident_bytes = [
        lm_head,
        markov_w1,
        markov_w2,
        confidence_weight,
        confidence_bias,
    ]
    .into_iter()
    .try_fold(0_u64, |bytes, buffer| {
        bytes
            .checked_add(buffer.bytes as u64)
            .context("dSpark head resident byte count overflow")
    })?;
    Ok(DsparkHeadResidentWeights {
        lm_head,
        markov_w1,
        markov_w2,
        confidence_weight,
        confidence_bias,
        resident_bytes,
    })
}

fn preloaded_dspark_update_weights(
    checkpoint: &DsparkCheckpoint,
) -> Result<DsparkUpdateResidentWeights> {
    let groups = dspark_draft_resident_groups(checkpoint)?;
    let revision = checkpoint.validated.fixture.revision;
    let resident = |suffix: &str| -> Result<glmrt_ffi::GlmrtDeviceBuffer> {
        let name = format!("dspark:{revision}:{suffix}");
        let group = groups
            .iter()
            .find(|group| group.resident_name == name)
            .with_context(|| format!("missing dSpark update resident plan {name}"))?;
        let expected_bytes: usize = group
            .byte_length
            .try_into()
            .with_context(|| format!("dSpark update resident {name} is too large"))?;
        preloaded_resident_weight_device_buffer(&name, expected_bytes)
            .with_context(|| format!("binding preloaded dSpark update resident {name}"))
    };
    let target_fusion = resident("fc.weight")?;
    let hidden_norm = resident("hidden_norm.weight")?;
    let qkv_full_bytes = 3_usize
        .checked_mul(GLM52_DSPARK_ATTENTION_HEADS)
        .and_then(|values| values.checked_mul(GLM52_DSPARK_HEAD_DIM))
        .and_then(|values| values.checked_mul(GLM52_DSPARK_HIDDEN_SIZE))
        .and_then(|values| values.checked_mul(2))
        .context("dSpark update QKV resident byte count overflow")?;
    let kv_offset_bytes = GLM52_DSPARK_ATTENTION_HEADS
        .checked_mul(GLM52_DSPARK_HEAD_DIM)
        .and_then(|values| values.checked_mul(GLM52_DSPARK_HIDDEN_SIZE))
        .and_then(|values| values.checked_mul(2))
        .context("dSpark update K/V view offset overflow")?;
    let kv_view_bytes = kv_offset_bytes
        .checked_mul(2)
        .context("dSpark update K/V view byte count overflow")?;
    let active_layers = checkpoint.validated.fixture.draft_layers;
    let mut layers = Vec::with_capacity(GLM52_DSPARK_MAX_DRAFT_LAYERS);
    for layer in 0..active_layers {
        let qkv_name = format!("dspark:{revision}:layers.{layer}.self_attn.qkv_proj.weight");
        let qkv_group = groups
            .iter()
            .find(|group| group.resident_name == qkv_name)
            .with_context(|| format!("missing dSpark update QKV resident {qkv_name}"))?;
        anyhow::ensure!(
            qkv_group.byte_length == qkv_full_bytes as u64,
            "dSpark update QKV resident byte count changed"
        );
        let kv = preloaded_resident_weight_device_buffer_view(
            &qkv_name,
            qkv_full_bytes,
            kv_offset_bytes,
            kv_view_bytes,
        )
        .with_context(|| format!("binding dSpark layer {layer} K/V resident view"))?;
        layers.push(DsparkUpdateLayerResidentWeights {
            k_norm: resident(&format!("layers.{layer}.self_attn.k_norm.weight"))?,
            kv,
        });
    }
    let filler = *layers
        .last()
        .context("dSpark update checkpoint has no transformer layers")?;
    layers.resize(GLM52_DSPARK_MAX_DRAFT_LAYERS, filler);
    let layers: [DsparkUpdateLayerResidentWeights; GLM52_DSPARK_MAX_DRAFT_LAYERS] = layers
        .try_into()
        .map_err(|_| anyhow::anyhow!("dSpark update resident layer count changed"))?;
    let referenced_bytes = [target_fusion, hidden_norm]
        .into_iter()
        .chain(
            layers
                .iter()
                .take(active_layers)
                .flat_map(|layer| [layer.k_norm, layer.kv]),
        )
        .try_fold(0_u64, |bytes, buffer| {
            bytes
                .checked_add(buffer.bytes as u64)
                .context("dSpark update referenced byte count overflow")
        })?;
    Ok(DsparkUpdateResidentWeights {
        target_fusion,
        hidden_norm,
        layers,
        active_layers,
        referenced_bytes,
    })
}

fn roll_dspark_swa_page_table(
    page_table: &mut [i32],
    cache_context_tokens: usize,
    incoming_tokens: usize,
    max_cache_context_tokens: usize,
    page_size: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(!page_table.is_empty(), "dSpark SWA page table is empty");
    anyhow::ensure!(page_size > 0, "dSpark SWA page size is zero");
    anyhow::ensure!(
        max_cache_context_tokens > 0 && max_cache_context_tokens % page_size == 0,
        "dSpark SWA window must be a positive page multiple"
    );
    anyhow::ensure!(
        page_table.len() * page_size >= max_cache_context_tokens + page_size,
        "dSpark SWA page table has no page-granular retention slop"
    );
    let retention_ceiling = max_cache_context_tokens
        .checked_add(page_size - 1)
        .context("dSpark SWA retention ceiling overflow")?;
    let projected = cache_context_tokens
        .checked_add(incoming_tokens)
        .context("dSpark SWA cache length overflow")?;
    if projected <= retention_ceiling {
        return Ok((cache_context_tokens, 0));
    }
    let pages_to_drop = projected
        .checked_sub(retention_ceiling)
        .expect("projected cache exceeds the retention ceiling")
        .div_ceil(page_size);
    let tokens_to_drop = pages_to_drop
        .checked_mul(page_size)
        .context("dSpark SWA dropped-token count overflow")?;
    anyhow::ensure!(
        tokens_to_drop <= cache_context_tokens && pages_to_drop < page_table.len(),
        "dSpark SWA cannot drop {tokens_to_drop} tokens from a {cache_context_tokens}-token cache"
    );
    page_table.rotate_left(pages_to_drop);
    Ok((cache_context_tokens - tokens_to_drop, pages_to_drop))
}

pub(super) struct DsparkRequestEngine {
    executor: DsparkStaticExecutor,
    checkpoint_revision: &'static str,
    max_verify_drafts: usize,
    max_cache_context_tokens: usize,
    page_size: usize,
    pages_per_request: usize,
    free_slots: Vec<usize>,
    active_slot: Option<usize>,
}

pub(super) struct DsparkRequestState {
    slot: usize,
    context_tokens: usize,
    cache_context_tokens: usize,
    page_table: Vec<i32>,
    page_table_dirty: bool,
}

impl DsparkRequestState {
    pub(super) fn context_tokens(&self) -> usize {
        self.context_tokens
    }
}

pub(super) struct DsparkRequestCacheSnapshot {
    pub(super) context_tokens: usize,
    pub(super) cache_context_tokens: usize,
    pub(super) kv_bytes: Vec<u8>,
}

impl DsparkRequestCacheSnapshot {
    pub(super) fn resident_bytes(&self) -> usize {
        self.kv_bytes.len()
    }
}

fn dspark_request_slot_page_table(slot: usize, pages_per_request: usize) -> Result<Vec<i32>> {
    let page_base = slot
        .checked_mul(pages_per_request)
        .context("dSpark request physical page base overflow")?;
    (0..pages_per_request)
        .map(|page| {
            page_base
                .checked_add(page)
                .context("dSpark request physical page ID overflow")?
                .try_into()
                .context("dSpark request physical page ID does not fit i32")
        })
        .collect()
}

#[derive(Clone, Debug)]
pub(super) struct DsparkDraftPlan {
    pub(super) proposal_token_ids: Vec<usize>,
    pub(super) conditional_confidence: Vec<f32>,
    pub(super) candidate_proposal_token_ids: Vec<usize>,
    pub(super) candidate_conditional_confidence: Vec<f32>,
    pub(super) candidate_adjusted_confidence: Vec<f64>,
    pub(super) selected_drafts: usize,
    pub(super) minimum_drafts: usize,
    pub(super) target_batch_rows: usize,
    pub(super) expected_committed_tokens: f64,
    pub(super) expected_tokens_per_second: f64,
    pub(super) confidence_logit_bias: f64,
    pub(super) confidence_context_tokens: usize,
    pub(super) calibration_eligible: bool,
}

const DSPARK_CONFIDENCE_CALIBRATION_WINDOW: usize = 16;
const DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT: f64 = 13.0;
const DSPARK_CONFIDENCE_PRIOR_PRECISION: f64 = 0.25;
const DSPARK_CONFIDENCE_MIN_RECENCY_DECAY: f64 = 0.70;
const DSPARK_CONFIDENCE_MAX_RECENCY_DECAY: f64 = 0.96;
const DSPARK_CONFIDENCE_MIN_PROBE_INTERVAL: usize = 4;
const DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL: usize = 16;
const DSPARK_CONFIDENCE_RESIDUAL_POSITIONS: usize = 15;
const DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_DECAY: f64 = 0.90;
const DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_RATE: f64 = 0.40;
const DSPARK_CONFIDENCE_RESIDUAL_POSITION_DECAY: f64 = 0.98;
const DSPARK_CONFIDENCE_RESIDUAL_POSITION_RATE: f64 = 0.01;
const DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT: f64 = 4.0;
pub(super) const DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS: usize = 32 * 1024;
const DSPARK_RUNTIME_COST_EXACT_PRIOR_WEIGHT: f64 = 2.0;
const DSPARK_RUNTIME_COST_ROW_PRIOR_WEIGHT: f64 = 4.0;
const DSPARK_RUNTIME_COST_CONCURRENCY_PRIOR_WEIGHT: f64 = 4.0;
const DSPARK_RUNTIME_CONTEXT_BUCKET_PRIOR_MS: f64 = 2.0;
// Current-host complete-cycle costs for physical M=1..16. M=1..8 use the
// five-repeat realistic adaptive corpus. Naturally rare M=9..16 use the
// combined realistic and diagnostic adaptive samples; unobserved M=15 is
// interpolated between its measured fixed-width M=14/15/16 position. M=16
// includes the Spark-RDMA reduction-path improvement.
const DSPARK_PHYSICAL_M_CYCLE_MS: [f64; 16] = [
    51.02, 79.05, 97.98, 108.90, 120.47, 131.21, 142.90, 149.43, 159.10, 164.54, 173.30, 187.73,
    203.16, 213.94, 218.58, 220.62,
];

#[derive(Clone, Debug)]
struct DsparkConfidenceObservation {
    conditional_confidence: Vec<f32>,
    accepted_drafts: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct DsparkConfidenceCalibrator {
    observations: VecDeque<DsparkConfidenceObservation>,
    logit_bias: f64,
    posterior_variance: f64,
    consecutive_zero_draft_plans: usize,
}

impl DsparkConfidenceCalibrator {
    pub(super) fn reset(&mut self) {
        self.observations.clear();
        self.logit_bias = 0.0;
        self.posterior_variance = 1.0 / DSPARK_CONFIDENCE_PRIOR_PRECISION;
        self.consecutive_zero_draft_plans = 0;
    }

    pub(super) fn logit_bias(&self) -> f64 {
        self.logit_bias
    }

    pub(super) fn observation_cycles(&self) -> usize {
        self.observations.len()
    }

    pub(super) fn posterior_variance(&self) -> f64 {
        self.posterior_variance
    }

    pub(super) fn force_probe_due(&self) -> bool {
        let uncertainty = (self.posterior_variance / 4.0).clamp(0.0, 1.0);
        let interval = ((DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL as f64)
            - uncertainty
                * (DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL - DSPARK_CONFIDENCE_MIN_PROBE_INTERVAL)
                    as f64)
            .round() as usize;
        self.consecutive_zero_draft_plans >= interval.max(1)
    }

    pub(super) fn record_selected_drafts(&mut self, selected_drafts: usize) {
        if selected_drafts == 0 {
            self.consecutive_zero_draft_plans = self.consecutive_zero_draft_plans.saturating_add(1);
        } else {
            self.consecutive_zero_draft_plans = 0;
        }
    }

    pub(super) fn observe(&mut self, conditional_confidence: &[f32], accepted_drafts: usize) {
        if conditional_confidence.is_empty() || accepted_drafts > conditional_confidence.len() {
            return;
        }
        self.observations.push_back(DsparkConfidenceObservation {
            conditional_confidence: conditional_confidence.to_vec(),
            accepted_drafts,
        });
        while self.observations.len() > DSPARK_CONFIDENCE_CALIBRATION_WINDOW {
            self.observations.pop_front();
        }
        let fit = fit_dspark_confidence_logit_bias(&self.observations, self.logit_bias);
        self.logit_bias = fit.logit_bias;
        self.posterior_variance = fit.posterior_variance;
    }
}

/// Cheap request-local correction for the raw dSpark confidence chain.
///
/// Shadow-trace replay found that stream/category drift dominates a single
/// global calibration fit, while later proposal positions have only a small
/// repeatable residual.  The controller therefore maintains one fast pooled
/// residual and fifteen much slower position residuals.  Checkpoint metadata
/// supplies a low-capacity continuous context prior only when trace evidence
/// shows a systematic context drift.
#[derive(Clone, Debug)]
pub(super) struct DsparkConfidenceResidual {
    dynamic_bias: f64,
    position_bias: [f64; DSPARK_CONFIDENCE_RESIDUAL_POSITIONS],
    observation_cycles: usize,
}

impl Default for DsparkConfidenceResidual {
    fn default() -> Self {
        Self {
            dynamic_bias: 0.0,
            position_bias: [0.0; DSPARK_CONFIDENCE_RESIDUAL_POSITIONS],
            observation_cycles: 0,
        }
    }
}

impl DsparkConfidenceResidual {
    pub(super) fn reset(&mut self) {
        *self = Self::default();
    }

    fn context_prior(context_tokens: usize) -> f64 {
        ACTIVE_DSPARK_CONFIDENCE_CONTEXT_PRIOR
            .get()
            .copied()
            .unwrap_or(SIRO_GLM52_DSPARK_PREVIEW.confidence_context_prior)
            .at_context(context_tokens)
    }

    pub(super) fn global_logit_bias(&self, context_tokens: usize) -> f64 {
        Self::context_prior(context_tokens) + self.dynamic_bias
    }

    pub(super) fn position_logit_bias(&self) -> &[f64] {
        &self.position_bias
    }

    pub(super) fn observation_cycles(&self) -> usize {
        self.observation_cycles
    }

    pub(super) fn record_selected_drafts(&mut self, selected_drafts: usize) {
        if selected_drafts == 0 {
            self.dynamic_bias *= DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_DECAY;
        }
    }

    pub(super) fn observe(
        &mut self,
        conditional_confidence: &[f32],
        accepted_drafts: usize,
        context_tokens: usize,
    ) {
        if conditional_confidence.is_empty()
            || conditional_confidence.len() > self.position_bias.len()
            || accepted_drafts > conditional_confidence.len()
        {
            return;
        }
        let observed_positions = if accepted_drafts < conditional_confidence.len() {
            accepted_drafts + 1
        } else {
            accepted_drafts
        };
        if observed_positions == 0 {
            return;
        }
        let global_bias = self.global_logit_bias(context_tokens);
        let mut error_sum = 0.0;
        for (position, raw_probability) in conditional_confidence
            .iter()
            .copied()
            .take(observed_positions)
            .enumerate()
        {
            let predicted = apply_dspark_confidence_logit_bias(
                f64::from(raw_probability),
                global_bias + self.position_bias[position],
            );
            let outcome = f64::from(position < accepted_drafts);
            let error = outcome - predicted;
            error_sum += error;
            self.position_bias[position] = (DSPARK_CONFIDENCE_RESIDUAL_POSITION_DECAY
                * self.position_bias[position]
                + DSPARK_CONFIDENCE_RESIDUAL_POSITION_RATE * error)
                .clamp(
                    -DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
                    DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
                );
        }
        let mean_error = error_sum / observed_positions as f64;
        self.dynamic_bias = (DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_DECAY * self.dynamic_bias
            + DSPARK_CONFIDENCE_RESIDUAL_GLOBAL_RATE * mean_error)
            .clamp(
                -DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
                DSPARK_CONFIDENCE_RESIDUAL_BIAS_LIMIT,
            );
        self.observation_cycles = self.observation_cycles.saturating_add(1);
    }
}

#[derive(Clone, Copy, Debug)]
struct DsparkConfidenceFit {
    logit_bias: f64,
    posterior_variance: f64,
}

fn clamp_dspark_probability(probability: f64) -> f64 {
    probability.clamp(1.0e-6, 1.0 - 1.0e-6)
}

fn apply_dspark_confidence_logit_bias(probability: f64, logit_bias: f64) -> f64 {
    let probability = clamp_dspark_probability(probability);
    let logit = (probability / (1.0 - probability)).ln() + logit_bias;
    if logit >= 0.0 {
        1.0 / (1.0 + (-logit).exp())
    } else {
        let exponential = logit.exp();
        exponential / (1.0 + exponential)
    }
}

fn fit_dspark_confidence_logit_bias(
    observations: &VecDeque<DsparkConfidenceObservation>,
    previous_bias: f64,
) -> DsparkConfidenceFit {
    let newest_surprise = observations.back().map_or(0.0, |observation| {
        let observed_positions =
            if observation.accepted_drafts < observation.conditional_confidence.len() {
                observation.accepted_drafts + 1
            } else {
                observation.accepted_drafts
            };
        let (error, count) = observation
            .conditional_confidence
            .iter()
            .copied()
            .take(observed_positions)
            .enumerate()
            .fold((0.0, 0_usize), |(error, count), (position, raw)| {
                let predicted = apply_dspark_confidence_logit_bias(f64::from(raw), previous_bias);
                let outcome = f64::from(position < observation.accepted_drafts);
                (error + (predicted - outcome).abs(), count + 1)
            });
        if count == 0 {
            0.0
        } else {
            error / count as f64
        }
    });
    let recency_decay = (DSPARK_CONFIDENCE_MAX_RECENCY_DECAY
        - newest_surprise
            * (DSPARK_CONFIDENCE_MAX_RECENCY_DECAY - DSPARK_CONFIDENCE_MIN_RECENCY_DECAY))
        .clamp(
            DSPARK_CONFIDENCE_MIN_RECENCY_DECAY,
            DSPARK_CONFIDENCE_MAX_RECENCY_DECAY,
        );
    let evaluate = |bias: f64| {
        let mut gradient = DSPARK_CONFIDENCE_PRIOR_PRECISION * bias;
        let mut curvature = DSPARK_CONFIDENCE_PRIOR_PRECISION;
        for (age, observation) in observations.iter().rev().enumerate() {
            let weight = recency_decay.powi(age as i32);
            let observed_positions =
                if observation.accepted_drafts < observation.conditional_confidence.len() {
                    observation.accepted_drafts + 1
                } else {
                    observation.accepted_drafts
                };
            for (position, raw_probability) in observation
                .conditional_confidence
                .iter()
                .copied()
                .take(observed_positions)
                .enumerate()
            {
                let calibrated =
                    apply_dspark_confidence_logit_bias(f64::from(raw_probability), bias);
                let outcome = if position < observation.accepted_drafts {
                    1.0
                } else {
                    0.0
                };
                gradient += weight * (calibrated - outcome);
                curvature += weight * calibrated * (1.0 - calibrated);
            }
        }
        (gradient, curvature)
    };
    // The one-dimensional logistic objective is convex. Bisection is both
    // cheaper and more robust than an undamped Newton step when a stream
    // temporarily has near-perfect matches or misses at p≈0/1.
    let mut lower = -DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT;
    let mut upper = DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT;
    let lower_gradient = evaluate(lower).0;
    let upper_gradient = evaluate(upper).0;
    let bias = if lower_gradient >= 0.0 {
        lower
    } else if upper_gradient <= 0.0 {
        upper
    } else {
        for _ in 0..40 {
            let midpoint = 0.5 * (lower + upper);
            if evaluate(midpoint).0 < 0.0 {
                lower = midpoint;
            } else {
                upper = midpoint;
            }
        }
        0.5 * (lower + upper)
    };
    let final_curvature = evaluate(bias).1;
    DsparkConfidenceFit {
        logit_bias: bias,
        posterior_variance: 1.0 / final_curvature.max(DSPARK_CONFIDENCE_PRIOR_PRECISION),
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct DsparkRuntimeCostCell {
    samples: u64,
    mean_value: f64,
    value_m2: f64,
}

impl DsparkRuntimeCostCell {
    fn observe(&mut self, value: f64) {
        self.samples = self.samples.saturating_add(1);
        let delta = value - self.mean_value;
        self.mean_value += delta / self.samples as f64;
        let delta_after = value - self.mean_value;
        self.value_m2 += delta * delta_after;
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DsparkRuntimeCostKey {
    request_count: usize,
    context_work_bucket: usize,
    max_context_bucket: usize,
    target_rows: usize,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DsparkRuntimeGlobalCostKey {
    request_count: usize,
    target_rows: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct DsparkRuntimeRouteCostCell {
    samples: u64,
    mean_ratio: f64,
    mean_critical_unique_experts: f64,
    critical_unique_experts_m2: f64,
    minimum_critical_unique_experts: usize,
    maximum_critical_unique_experts: usize,
}

impl DsparkRuntimeRouteCostCell {
    fn observe(&mut self, ratio: f64, critical_unique_experts: usize) {
        self.samples = self.samples.saturating_add(1);
        self.mean_ratio += (ratio - self.mean_ratio) / self.samples as f64;
        let unique = critical_unique_experts as f64;
        let delta = unique - self.mean_critical_unique_experts;
        self.mean_critical_unique_experts += delta / self.samples as f64;
        self.critical_unique_experts_m2 += delta * (unique - self.mean_critical_unique_experts);
        if self.samples == 1 {
            self.minimum_critical_unique_experts = critical_unique_experts;
            self.maximum_critical_unique_experts = critical_unique_experts;
        } else {
            self.minimum_critical_unique_experts = self
                .minimum_critical_unique_experts
                .min(critical_unique_experts);
            self.maximum_critical_unique_experts = self
                .maximum_critical_unique_experts
                .max(critical_unique_experts);
        }
    }

    fn representative(&self) -> bool {
        self.samples >= 8
            && self.maximum_critical_unique_experts > self.minimum_critical_unique_experts
            && self.critical_unique_experts_m2 > 0.0
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkRuntimeCostObservation {
    pub(super) request_count: usize,
    pub(super) context_work_bucket: usize,
    pub(super) max_context_bucket: usize,
    pub(super) target_rows: usize,
    pub(super) observed_ms: f64,
    pub(super) predicted_ms_before: f64,
    pub(super) exact_samples: u64,
}

#[derive(Clone, Debug)]
pub(super) struct DsparkRuntimeCostModel {
    max_requests: usize,
    max_drafts_per_request: usize,
    exact: BTreeMap<DsparkRuntimeCostKey, DsparkRuntimeCostCell>,
    row: BTreeMap<DsparkRuntimeGlobalCostKey, DsparkRuntimeCostCell>,
    concurrency: BTreeMap<usize, DsparkRuntimeCostCell>,
    profiled_ms: BTreeMap<DsparkRuntimeGlobalCostKey, f64>,
    learn_profiled_residuals: bool,
    route_conditioned: BTreeMap<DsparkRuntimeGlobalCostKey, DsparkRuntimeRouteCostCell>,
}

impl DsparkRuntimeCostModel {
    pub(super) fn new(max_requests: usize, max_drafts_per_request: usize) -> Result<Self> {
        anyhow::ensure!(max_requests > 0, "dSpark cost model requires requests");
        anyhow::ensure!(
            max_drafts_per_request < DSPARK_PHYSICAL_M_CYCLE_MS.len(),
            "dSpark balanced prior covers at most physical M={}, but max drafts is {max_drafts_per_request}",
            DSPARK_PHYSICAL_M_CYCLE_MS.len(),
        );
        Ok(Self {
            max_requests,
            max_drafts_per_request,
            exact: BTreeMap::new(),
            row: BTreeMap::new(),
            concurrency: BTreeMap::new(),
            profiled_ms: BTreeMap::new(),
            learn_profiled_residuals: false,
            route_conditioned: BTreeMap::new(),
        })
    }

    fn enable_profiled_residual_learning(&mut self) {
        self.learn_profiled_residuals = true;
    }

    pub(super) fn install_profile(
        &mut self,
        request_count: usize,
        rows: &[(usize, f64)],
    ) -> Result<()> {
        let max_rows = self.max_target_rows(request_count)?;
        anyhow::ensure!(
            rows.len() == max_rows - request_count + 1,
            "dSpark SPS profile for {request_count} requests has {} rows, expected {}",
            rows.len(),
            max_rows - request_count + 1,
        );
        for (expected_rows, (target_rows, observed_ms)) in
            (request_count..=max_rows).zip(rows.iter().copied())
        {
            anyhow::ensure!(
                target_rows == expected_rows,
                "dSpark SPS profile for {request_count} requests expected target row {expected_rows}, got {target_rows}",
            );
            anyhow::ensure!(
                observed_ms.is_finite() && observed_ms > 0.0,
                "invalid profiled dSpark latency {observed_ms} ms at request count {request_count}, target rows {target_rows}",
            );
            let replaced = self.profiled_ms.insert(
                DsparkRuntimeGlobalCostKey {
                    request_count,
                    target_rows,
                },
                observed_ms,
            );
            anyhow::ensure!(
                replaced.is_none(),
                "duplicate dSpark SPS profile cell at request count {request_count}, target rows {target_rows}",
            );
        }
        Ok(())
    }

    pub(super) fn context_buckets(context_tokens: &[usize]) -> Result<(usize, usize)> {
        let max_context = context_tokens
            .iter()
            .copied()
            .max()
            .context("dSpark cost model requires at least one context")?;
        let context_work = context_tokens
            .iter()
            .try_fold(0_usize, |total, tokens| total.checked_add(*tokens))
            .context("dSpark aggregate attention context overflow")?;
        Ok((
            context_work / DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS,
            max_context / DSPARK_RUNTIME_CONTEXT_BUCKET_TOKENS,
        ))
    }

    fn max_target_rows(&self, request_count: usize) -> Result<usize> {
        anyhow::ensure!(
            (1..=self.max_requests).contains(&request_count),
            "dSpark cost request count {request_count} is outside 1..={}",
            self.max_requests,
        );
        request_count
            .checked_mul(self.max_drafts_per_request + 1)
            .context("dSpark runtime cost row limit overflow")
    }

    fn balanced_prior_ms(target_rows: usize, context_bucket: usize) -> Result<f64> {
        anyhow::ensure!(
            target_rows > 0,
            "dSpark physical target rows must be positive"
        );
        let short_context_ms = if target_rows <= DSPARK_PHYSICAL_M_CYCLE_MS.len() {
            DSPARK_PHYSICAL_M_CYCLE_MS[target_rows - 1]
        } else {
            let last = DSPARK_PHYSICAL_M_CYCLE_MS[DSPARK_PHYSICAL_M_CYCLE_MS.len() - 1];
            let tail_start = DSPARK_PHYSICAL_M_CYCLE_MS.len() - 5;
            let tail_slope = (last - DSPARK_PHYSICAL_M_CYCLE_MS[tail_start])
                / (DSPARK_PHYSICAL_M_CYCLE_MS.len() - 1 - tail_start) as f64;
            last + tail_slope * (target_rows - DSPARK_PHYSICAL_M_CYCLE_MS.len()) as f64
        };
        Ok(short_context_ms + context_bucket as f64 * DSPARK_RUNTIME_CONTEXT_BUCKET_PRIOR_MS)
    }

    fn base_prior_ms(
        &self,
        request_count: usize,
        target_rows: usize,
        context_bucket: usize,
    ) -> Result<f64> {
        let row_key = DsparkRuntimeGlobalCostKey {
            request_count,
            target_rows,
        };
        if let Some(profiled_ms) = self.profiled_ms.get(&row_key).copied() {
            return Ok(profiled_ms + context_bucket as f64 * DSPARK_RUNTIME_CONTEXT_BUCKET_PRIOR_MS);
        }
        Self::balanced_prior_ms(target_rows, context_bucket)
    }

    fn predicted_ms_for_bucket(
        &self,
        request_count: usize,
        context_work_bucket: usize,
        max_context_bucket: usize,
        target_rows: usize,
    ) -> Result<f64> {
        let max_rows = self.max_target_rows(request_count)?;
        anyhow::ensure!(
            (request_count..=max_rows).contains(&target_rows),
            "dSpark target rows {target_rows} are outside {request_count}..={max_rows}",
        );
        let prior = self.base_prior_ms(request_count, target_rows, context_work_bucket)?;
        let row_key = DsparkRuntimeGlobalCostKey {
            request_count,
            target_rows,
        };
        if let Some(route) = self
            .route_conditioned
            .get(&row_key)
            .filter(|route| route.representative())
        {
            let weight = route.samples as f64 / (route.samples as f64 + 8.0);
            return Ok((prior * ((1.0 - weight) + weight * route.mean_ratio)).max(0.001));
        }
        let concurrency = self
            .concurrency
            .get(&request_count)
            .copied()
            .unwrap_or_default();
        let concurrency_weight = concurrency.samples as f64
            / (concurrency.samples as f64 + DSPARK_RUNTIME_COST_CONCURRENCY_PRIOR_WEIGHT);
        // Scheduling needs E[T], so residual aggregation stays in linear
        // latency-ratio space.  Averaging log ratios would estimate a
        // geometric mean and systematically underprice route/stall tails.
        let mut mean_ratio =
            (1.0 - concurrency_weight) + concurrency_weight * concurrency.mean_value;
        let row = self.row.get(&row_key).copied().unwrap_or_default();
        let row_weight =
            row.samples as f64 / (row.samples as f64 + DSPARK_RUNTIME_COST_ROW_PRIOR_WEIGHT);
        mean_ratio = (1.0 - row_weight) * mean_ratio + row_weight * row.mean_value;
        let exact_key = DsparkRuntimeCostKey {
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
        };
        let exact = self.exact.get(&exact_key).copied().unwrap_or_default();
        let exact_weight =
            exact.samples as f64 / (exact.samples as f64 + DSPARK_RUNTIME_COST_EXACT_PRIOR_WEIGHT);
        mean_ratio = (1.0 - exact_weight) * mean_ratio + exact_weight * exact.mean_value;
        Ok((prior * mean_ratio).max(0.001))
    }

    pub(super) fn profile(
        &self,
        request_count: usize,
        context_tokens: &[usize],
    ) -> Result<DsparkSpsProfile> {
        anyhow::ensure!(
            context_tokens.len() == request_count,
            "dSpark cost profile has {} contexts for {request_count} requests",
            context_tokens.len(),
        );
        let (context_work_bucket, max_context_bucket) = Self::context_buckets(context_tokens)?;
        let max_rows = self.max_target_rows(request_count)?;
        let mut steps_per_second = vec![0.0; max_rows + 1];
        let mut previous_ms = 0.0_f64;
        for (target_rows, value) in steps_per_second.iter_mut().enumerate().skip(request_count) {
            // A larger verification pack cannot complete before all work in
            // its smaller prefix. Runtime noise in sparsely sampled cells must
            // not manufacture a false latency dip that attracts the global
            // scheduler.
            let predicted_ms = self
                .predicted_ms_for_bucket(
                    request_count,
                    context_work_bucket,
                    max_context_bucket,
                    target_rows,
                )?
                .max(previous_ms);
            previous_ms = predicted_ms;
            *value = 1_000.0 / predicted_ms;
        }
        for target_rows in 1..request_count {
            steps_per_second[target_rows] = steps_per_second[request_count];
        }
        DsparkSpsProfile::new(steps_per_second)
    }

    pub(super) fn observe(
        &mut self,
        request_count: usize,
        context_tokens: &[usize],
        target_rows: usize,
        observed_ms: f64,
        route_critical_unique_experts: Option<usize>,
    ) -> Result<DsparkRuntimeCostObservation> {
        anyhow::ensure!(
            observed_ms.is_finite() && observed_ms > 0.0,
            "invalid dSpark runtime cost observation {observed_ms}",
        );
        let (context_work_bucket, max_context_bucket) = Self::context_buckets(context_tokens)?;
        let predicted_ms_before = self.predicted_ms_for_bucket(
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
        )?;
        if !self.learn_profiled_residuals
            && self.profiled_ms.contains_key(&DsparkRuntimeGlobalCostKey {
                request_count,
                target_rows,
            })
        {
            return Ok(DsparkRuntimeCostObservation {
                request_count,
                context_work_bucket,
                max_context_bucket,
                target_rows,
                observed_ms,
                predicted_ms_before,
                exact_samples: 0,
            });
        }
        // DFlash2 may explicitly treat its topology/model/power-qualified
        // profile as a cold-start prior. Learn bounded residual ratios around
        // it so fallback cells and complete-cycle costs converge without
        // discarding that calibrated surface. Qualified dSpark profiles keep
        // the historical immutable behavior above.
        // Preserve real regime changes while preventing one host stall from
        // permanently poisoning a rarely visited row/context cell.
        let prior = self.base_prior_ms(request_count, target_rows, context_work_bucket)?;
        let robust_observation =
            observed_ms.clamp(predicted_ms_before * 0.25, predicted_ms_before * 4.0);
        if let Some(critical_unique_experts) = route_critical_unique_experts {
            let route = self
                .route_conditioned
                .entry(DsparkRuntimeGlobalCostKey {
                    request_count,
                    target_rows,
                })
                .or_default();
            route.observe(robust_observation / prior, critical_unique_experts);
            return Ok(DsparkRuntimeCostObservation {
                request_count,
                context_work_bucket,
                max_context_bucket,
                target_rows,
                observed_ms,
                predicted_ms_before,
                exact_samples: route.samples,
            });
        }
        let latency_ratio = robust_observation / prior;
        let exact_key = DsparkRuntimeCostKey {
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
        };
        let exact = self.exact.entry(exact_key).or_default();
        exact.observe(latency_ratio);
        let exact_samples = exact.samples;
        self.row
            .entry(DsparkRuntimeGlobalCostKey {
                request_count,
                target_rows,
            })
            .or_default()
            .observe(latency_ratio);
        self.concurrency
            .entry(request_count)
            .or_default()
            .observe(latency_ratio);
        Ok(DsparkRuntimeCostObservation {
            request_count,
            context_work_bucket,
            max_context_bucket,
            target_rows,
            observed_ms,
            predicted_ms_before,
            exact_samples,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DsparkCostProfileActivation {
    pub(super) profile_id: &'static str,
    pub(super) source_sha256: &'static str,
    pub(super) sparkinfer_revision: &'static str,
    pub(super) topology: &'static str,
    pub(super) power_limit_watts: usize,
}

pub(super) fn install_qualified_dspark_cost_profile(
    model: &mut DsparkRuntimeCostModel,
    target_model: &str,
    target_snapshot: &Path,
    checkpoint_revision: &str,
    sparkinfer_revision: Option<&str>,
    coordinator_power_limit_watts: Option<usize>,
    max_execution_lanes: usize,
    max_verify_drafts: usize,
) -> Result<Option<DsparkCostProfileActivation>> {
    let target_revision = target_snapshot.file_name().and_then(|name| name.to_str());
    let sparkinfer_profile_compatible = sparkinfer_revision.is_some_and(|revision| {
        revision == GLM52_REDHAT_DSPARK_COST_PROFILE_SPARKINFER_REVISION
            || revision
                == GLM52_REDHAT_DSPARK_COST_PROFILE_GLMRT_EXL3_COMPATIBLE_SPARKINFER_REVISION
    });
    if target_model != GLM52_REDHAT_DSPARK_COST_PROFILE_TARGET_MODEL
        || target_revision != Some(GLM52_REDHAT_DSPARK_COST_PROFILE_TARGET_REVISION)
        || checkpoint_revision != GLM52_REDHAT_DSPARK_COST_PROFILE_DSPARK_REVISION
        || !sparkinfer_profile_compatible
        || coordinator_power_limit_watts != Some(GLM52_REDHAT_DSPARK_COST_PROFILE_POWER_LIMIT_WATTS)
        || max_execution_lanes > GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_CONCURRENCY
        || max_verify_drafts != GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_DRAFTS
    {
        return Ok(None);
    }
    for (request_index, rows) in GLM52_REDHAT_DSPARK_COST_PROFILE_MS
        .iter()
        .take(max_execution_lanes)
        .enumerate()
    {
        model.install_profile(request_index + 1, rows)?;
    }
    Ok(Some(DsparkCostProfileActivation {
        profile_id: GLM52_REDHAT_DSPARK_COST_PROFILE_ID,
        source_sha256: GLM52_REDHAT_DSPARK_COST_PROFILE_SOURCE_SHA256,
        sparkinfer_revision: GLM52_REDHAT_DSPARK_COST_PROFILE_SPARKINFER_REVISION,
        topology: GLM52_REDHAT_DSPARK_COST_PROFILE_TOPOLOGY,
        power_limit_watts: GLM52_REDHAT_DSPARK_COST_PROFILE_POWER_LIMIT_WATTS,
    }))
}

pub(super) fn install_qualified_dflash2_cost_profile(
    model: &mut DsparkRuntimeCostModel,
    target_model: &str,
    target_snapshot: &Path,
    checkpoint_model: &str,
    checkpoint_revision: &str,
    sparkinfer_revision: Option<&str>,
    coordinator_power_limit_watts: Option<usize>,
    max_execution_lanes: usize,
    max_verify_drafts: usize,
) -> Result<Option<DsparkCostProfileActivation>> {
    let target_revision = target_snapshot.file_name().and_then(|name| name.to_str());
    if target_model != GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_MODEL
        || target_revision != Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_REVISION)
        || checkpoint_model != GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_MODEL
        || checkpoint_revision != GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_REVISION
        || sparkinfer_revision != Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SPARKINFER_REVISION)
        || coordinator_power_limit_watts
            != Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_POWER_LIMIT_WATTS)
        || max_execution_lanes > GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_CONCURRENCY
        || max_verify_drafts != GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_DRAFTS
    {
        return Ok(None);
    }
    for (request_index, rows) in GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MS
        .iter()
        .take(max_execution_lanes)
        .enumerate()
    {
        model.install_profile(request_index + 1, rows)?;
    }
    model.enable_profiled_residual_learning();
    Ok(Some(DsparkCostProfileActivation {
        profile_id: GLM53_EXL3_K4_DFLASH2_COST_PROFILE_ID,
        source_sha256: GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SOURCE_SHA256,
        sparkinfer_revision: GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SPARKINFER_REVISION,
        topology: GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TOPOLOGY,
        power_limit_watts: GLM53_EXL3_K4_DFLASH2_COST_PROFILE_POWER_LIMIT_WATTS,
    }))
}

// The serving executor serializes this engine behind one mutex. CUDA streams,
// graphs, and device buffers use the device primary context and may be handed
// to the pinned request worker, but they are never replayed concurrently.
unsafe impl Send for DsparkRequestEngine {}

impl DsparkRequestEngine {
    pub(super) fn checkpoint_revision(&self) -> &'static str {
        self.checkpoint_revision
    }

    pub(super) fn max_verify_drafts(&self) -> usize {
        self.max_verify_drafts
    }

    pub(super) fn load(
        snapshot: &Path,
        target_catalog: &TensorCatalog,
        kv_capacity_tokens: usize,
        max_active_requests: usize,
    ) -> Result<Self> {
        let load_started = Instant::now();
        let fixture = production_dspark_fixture_for_snapshot(snapshot)?;
        activate_dspark_contract(fixture)?;
        let checkpoint = DsparkCheckpoint::from_snapshot(fixture, snapshot)?;
        let checkpoint_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
        let query_rows = checkpoint.validated.query_layout.query_rows();
        anyhow::ensure!(
            kv_capacity_tokens >= query_rows + 1,
            "dSpark request KV capacity {kv_capacity_tokens} is too small"
        );
        anyhow::ensure!(
            max_active_requests > 0,
            "dSpark request executor requires at least one physical cache slot"
        );
        let page_size = 64;
        let cache_tokens_before_alignment = kv_capacity_tokens
            .checked_sub(query_rows + page_size)
            .context(
                "dSpark request KV capacity is smaller than its page slop and proposal suffix",
            )?;
        let max_cache_context_tokens = cache_tokens_before_alignment / page_size * page_size;
        anyhow::ensure!(
            max_cache_context_tokens > 0,
            "dSpark request cache window is empty after {page_size}-token page alignment"
        );
        checkpoint.weights.validate_target_aliases(target_catalog)?;
        let draft_preload_started = Instant::now();
        let preload = preload_dspark_draft_owned_weights(&checkpoint)?;
        let draft_preload_ms = draft_preload_started.elapsed().as_secs_f64() * 1_000.0;
        let alias_preload_started = Instant::now();
        let query_alias = preload_dspark_query_embedding_alias(&checkpoint)?;
        let head_alias = preload_dspark_head_lm_alias(&checkpoint)?;
        let alias_preload_ms = alias_preload_started.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "real_full_dspark_preload revision={} draft_buffers={} draft_bytes={} loaded_bytes={} query_alias_loaded={} head_alias_loaded={}",
            fixture.revision,
            preload.selected_resident_buffers,
            preload.selected_bytes,
            preload.loaded_bytes,
            query_alias.loaded,
            head_alias.loaded,
        );
        let weights = DsparkStaticResidentWeights {
            query: preloaded_dspark_query_weights()?,
            update: preloaded_dspark_update_weights(&checkpoint)?,
            body: preloaded_dspark_body_weights(&checkpoint)?,
            head: preloaded_dspark_head_weights(&checkpoint)?,
        };
        let resident_binding_ms = load_started.elapsed().as_secs_f64() * 1_000.0
            - checkpoint_ms
            - draft_preload_ms
            - alias_preload_ms;
        let pages_per_request = kv_capacity_tokens.div_ceil(page_size);
        let physical_kv_pages = pages_per_request
            .checked_mul(max_active_requests)
            .context("dSpark shared request KV page count overflow")?;
        let kv_bytes_per_token = fixture
            .draft_layers
            .checked_mul(GLM52_DSPARK_ATTENTION_HEADS)
            .and_then(|value| value.checked_mul(GLM52_DSPARK_HEAD_DIM))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
            .context("dSpark shared request KV bytes/token overflow")?;
        let physical_kv_bytes = physical_kv_pages
            .checked_mul(page_size)
            .and_then(|tokens| tokens.checked_mul(kv_bytes_per_token))
            .context("dSpark shared request KV byte count overflow")?;
        eprintln!(
            "real_full_dspark_kv_pool request_slots={} pages_per_request={} physical_pages={} page_tokens={} bytes_per_token={} physical_bytes={}",
            max_active_requests,
            pages_per_request,
            physical_kv_pages,
            page_size,
            kv_bytes_per_token,
            physical_kv_bytes,
        );
        let executor_create_started = Instant::now();
        let mut executor = DsparkStaticExecutor::capture_with_physical_pages(
            weights,
            DsparkStaticBenchConfig {
                draft_layers: fixture.draft_layers,
                active_requests: 1,
                query_rows: checkpoint.validated.query_layout.query_rows(),
                proposal_tokens: checkpoint.validated.query_layout.proposal_tokens(),
                proposal_start_row: checkpoint.validated.query_layout.proposal_query_slots[0],
                accepted_rows_per_request: 1,
                context_tokens: 0,
                kv_capacity_tokens,
                allocate_full_kv_capacity: true,
                page_size,
                kv_storage: DsparkKvStorage::Bf16,
                mask_token_id: GLM52_DSPARK_MASK_TOKEN_ID,
                warmup: 0,
                iterations: 1,
                repeats: 1,
                seed: 20_260_724,
            },
            Some(physical_kv_pages),
        )
        .context("capturing the live C=1 dSpark request executor")?;
        let executor_create_ms = executor_create_started.elapsed().as_secs_f64() * 1_000.0;
        let batched_capture_started = Instant::now();
        executor
            .capture_batched_update_graphs(weights.update, &[2, 4, 8, 16, 32, 64, 128, 256, 512])
            .context("capturing batched dSpark prompt/decode update graphs")?;
        let batched_capture_ms = batched_capture_started.elapsed().as_secs_f64() * 1_000.0;
        let total_ms = load_started.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "real_full_dspark_engine_load_detail checkpoint_ms={checkpoint_ms:.3} draft_preload_ms={draft_preload_ms:.3} alias_preload_ms={alias_preload_ms:.3} resident_binding_ms={resident_binding_ms:.3} executor_create_ms={executor_create_ms:.3} batched_capture_ms={batched_capture_ms:.3} total_ms={total_ms:.3}"
        );
        Ok(Self {
            executor,
            checkpoint_revision: fixture.revision,
            max_verify_drafts: fixture.max_verify_drafts,
            max_cache_context_tokens,
            page_size,
            pages_per_request,
            free_slots: (0..max_active_requests).rev().collect(),
            active_slot: None,
        })
    }

    pub(super) fn allocate_request_state(&mut self) -> Result<DsparkRequestState> {
        let slot = self
            .free_slots
            .pop()
            .context("dSpark shared request KV slots are exhausted")?;
        let page_table = dspark_request_slot_page_table(slot, self.pages_per_request)?;
        Ok(DsparkRequestState {
            slot,
            context_tokens: 0,
            cache_context_tokens: 0,
            page_table,
            page_table_dirty: true,
        })
    }

    pub(super) fn reset_request_state(&mut self, state: &mut DsparkRequestState) -> Result<()> {
        state.context_tokens = 0;
        state.cache_context_tokens = 0;
        state.page_table = dspark_request_slot_page_table(state.slot, self.pages_per_request)?;
        state.page_table_dirty = true;
        Ok(())
    }

    pub(super) fn release_request_state(&mut self, state: DsparkRequestState) {
        debug_assert!(!self.free_slots.contains(&state.slot));
        self.free_slots.push(state.slot);
    }

    pub(super) fn snapshot_request_state(
        &self,
        state: &DsparkRequestState,
    ) -> Result<Option<DsparkRequestCacheSnapshot>> {
        if state.cache_context_tokens == 0 {
            return Ok(None);
        }
        let kv_bytes = self
            .executor
            .read_request_cache_snapshot(&state.page_table, state.cache_context_tokens)
            .context("saving the committed dSpark request-cache tail")?;
        Ok(Some(DsparkRequestCacheSnapshot {
            context_tokens: state.context_tokens,
            cache_context_tokens: state.cache_context_tokens,
            kv_bytes,
        }))
    }

    pub(super) fn snapshot_request_state_at_prefix(
        &self,
        state: &DsparkRequestState,
        prefix_tokens: usize,
    ) -> Result<Option<DsparkRequestCacheSnapshot>> {
        anyhow::ensure!(
            prefix_tokens <= state.context_tokens,
            "dSpark reusable prefix {prefix_tokens} exceeds request context {}",
            state.context_tokens,
        );
        let rollback_tokens = state.context_tokens - prefix_tokens;
        anyhow::ensure!(
            rollback_tokens <= state.cache_context_tokens,
            "dSpark reusable-prefix rollback {rollback_tokens} exceeds cache context {}",
            state.cache_context_tokens,
        );
        let cache_context_tokens = state.cache_context_tokens - rollback_tokens;
        if cache_context_tokens == 0 {
            return Ok(None);
        }
        let kv_bytes = self
            .executor
            .read_request_cache_snapshot(&state.page_table, cache_context_tokens)
            .context("saving the reusable dSpark request-cache prefix")?;
        Ok(Some(DsparkRequestCacheSnapshot {
            context_tokens: prefix_tokens,
            cache_context_tokens,
            kv_bytes,
        }))
    }

    pub(super) fn restore_request_state(
        &mut self,
        state: &mut DsparkRequestState,
        snapshot: &DsparkRequestCacheSnapshot,
    ) -> Result<()> {
        anyhow::ensure!(
            snapshot.cache_context_tokens <= self.max_cache_context_tokens + self.page_size - 1,
            "dSpark tail snapshot retains {} cache tokens beyond the {}+{} token window/slop limit",
            snapshot.cache_context_tokens,
            self.max_cache_context_tokens,
            self.page_size - 1,
        );
        self.executor
            .restore_request_cache_snapshot(
                &state.page_table,
                snapshot.cache_context_tokens,
                &snapshot.kv_bytes,
            )
            .context("restoring the committed dSpark request-cache tail")?;
        state.context_tokens = snapshot.context_tokens;
        state.cache_context_tokens = snapshot.cache_context_tokens;
        state.page_table_dirty = true;
        Ok(())
    }

    pub(super) fn replay_step(
        &mut self,
        state: &mut DsparkRequestState,
        target_hidden_taps: [&DeviceBf16Output; GLM52_DSPARK_TARGET_TAPS],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<DsparkDraftStep> {
        if let Some(absolute_context_start) = absolute_context_start {
            if state.cache_context_tokens == 0 {
                state.context_tokens = absolute_context_start;
            } else {
                anyhow::ensure!(
                    state.context_tokens == absolute_context_start,
                    "restored dSpark tail ends at absolute context {} but the uncached target suffix starts at {absolute_context_start}",
                    state.context_tokens,
                );
            }
        }
        let (cache_context_tokens, dropped_pages) = roll_dspark_swa_page_table(
            &mut state.page_table,
            state.cache_context_tokens,
            committed_rows,
            self.max_cache_context_tokens,
            self.page_size,
        )?;
        if dropped_pages > 0 || state.page_table_dirty || self.active_slot != Some(state.slot) {
            self.executor
                .set_request_page_table(&state.page_table)
                .context("uploading the dSpark request SWA page rotation")?;
            state.page_table_dirty = false;
            self.active_slot = Some(state.slot);
        }
        let step = self.executor.replay_request_step_with_cache_context(
            target_hidden_taps,
            target_row_start,
            committed_rows,
            state.context_tokens,
            cache_context_tokens,
            anchor_token,
        )?;
        state.context_tokens += committed_rows;
        state.cache_context_tokens = cache_context_tokens + committed_rows;
        Ok(step)
    }

    pub(super) fn plan_verification(
        &self,
        step: &DsparkDraftStep,
        max_drafts: usize,
        confidence_logit_bias: f64,
        position_logit_bias: &[f64],
        confidence_context_tokens: usize,
        force_probe: bool,
        sps: &DsparkSpsProfile,
    ) -> Result<DsparkDraftPlan> {
        let max_drafts = max_drafts
            .min(self.max_verify_drafts)
            .min(step.proposal_token_ids.len());
        anyhow::ensure!(
            position_logit_bias.is_empty() || position_logit_bias.len() >= max_drafts,
            "dSpark position confidence bias has {} entries for {max_drafts} proposals",
            position_logit_bias.len(),
        );
        let raw_confidence = step
            .conditional_confidence
            .iter()
            .take(max_drafts)
            .copied()
            .collect::<Vec<_>>();
        let confidence = raw_confidence
            .iter()
            .copied()
            .enumerate()
            .map(|(position, value)| {
                apply_dspark_confidence_logit_bias(
                    f64::from(value),
                    confidence_logit_bias
                        + position_logit_bias.get(position).copied().unwrap_or(0.0),
                )
            })
            .collect::<Vec<_>>();
        let schedule = schedule_dspark_verification(
            &[confidence.clone()],
            sps,
            DsparkScheduleSearch::GlobalMaximum,
        )?;
        let selected_drafts = if force_probe && schedule.prefix_lengths[0] == 0 && max_drafts > 0 {
            1
        } else {
            schedule.prefix_lengths[0]
        };
        let (target_batch_rows, expected_committed_tokens, expected_tokens_per_second) =
            if selected_drafts == schedule.prefix_lengths[0] {
                (
                    schedule.target_batch_rows,
                    schedule.expected_committed_tokens,
                    schedule.expected_tokens_per_second,
                )
            } else {
                let expected_tokens = 1.0 + confidence[0];
                (2, expected_tokens, expected_tokens * sps.get(2)?)
            };
        Ok(DsparkDraftPlan {
            proposal_token_ids: step.proposal_token_ids[..selected_drafts].to_vec(),
            conditional_confidence: raw_confidence[..selected_drafts].to_vec(),
            candidate_proposal_token_ids: step.proposal_token_ids[..max_drafts].to_vec(),
            candidate_conditional_confidence: raw_confidence,
            candidate_adjusted_confidence: confidence,
            selected_drafts,
            minimum_drafts: usize::from(
                force_probe && schedule.prefix_lengths[0] == 0 && max_drafts > 0,
            ),
            target_batch_rows,
            expected_committed_tokens,
            expected_tokens_per_second,
            confidence_logit_bias,
            confidence_context_tokens,
            calibration_eligible: true,
        })
    }
}

impl DsparkDraftPlan {
    pub(super) fn calibrated_candidate_confidence(&self, max_drafts: usize) -> Vec<f64> {
        self.candidate_adjusted_confidence
            .iter()
            .copied()
            .take(max_drafts)
            .collect()
    }

    pub(super) fn apply_joint_selection(
        &mut self,
        selected_drafts: usize,
        target_batch_rows: usize,
        expected_committed_tokens: f64,
        expected_tokens_per_second: f64,
    ) -> Result<()> {
        anyhow::ensure!(
            selected_drafts >= self.minimum_drafts
                && selected_drafts <= self.candidate_proposal_token_ids.len()
                && selected_drafts <= self.candidate_conditional_confidence.len(),
            "joint dSpark selection {selected_drafts} is outside minimum {} and candidate proposal/confidence lengths {}/{}",
            self.minimum_drafts,
            self.candidate_proposal_token_ids.len(),
            self.candidate_conditional_confidence.len(),
        );
        self.proposal_token_ids = self.candidate_proposal_token_ids[..selected_drafts].to_vec();
        self.conditional_confidence =
            self.candidate_conditional_confidence[..selected_drafts].to_vec();
        self.selected_drafts = selected_drafts;
        self.target_batch_rows = target_batch_rows;
        self.expected_committed_tokens = expected_committed_tokens;
        self.expected_tokens_per_second = expected_tokens_per_second;
        Ok(())
    }
}

#[derive(Debug, Serialize)]
struct DsparkSharedAliasReport {
    source_name: String,
    target_resident_name: String,
    byte_length: u64,
    expected_sha256: &'static str,
}

#[derive(Debug, Serialize)]
struct DsparkPreflightReport {
    checkpoint_repo_id: &'static str,
    checkpoint_revision: &'static str,
    snapshot_path: String,
    tensor_count: usize,
    payload_bytes: u64,
    draft_owned_bytes: u64,
    target_aliased_bytes: u64,
    shared_aliases: Vec<DsparkSharedAliasReport>,
    engine_plan: DsparkStaticEnginePlan,
    preload: Option<DsparkResidentPreloadStats>,
    query_alias_preload: Option<DsparkTargetAliasPreloadStats>,
    head_alias_preload: Option<DsparkTargetAliasPreloadStats>,
    attention_graphs: Option<Vec<DsparkPagedAttentionGraphReport>>,
    query_graphs: Option<Vec<DsparkQueryGraphReport>>,
    body_graphs: Option<Vec<DsparkBodyGraphReport>>,
    head_graphs: Option<Vec<DsparkHeadGraphReport>>,
    update_graphs: Option<Vec<DsparkUpdateGraphReport>>,
    static_graphs: Option<Vec<DsparkStaticGraphReport>>,
}

pub(crate) fn run_dspark_preflight(args: DsparkPreflightArgs) -> Result<()> {
    if args.capture_attention
        || args.capture_body
        || args.capture_head
        || args.capture_query
        || args.capture_update
        || args.capture_static
    {
        anyhow::ensure!(
            matches!(args.max_concurrency, 1 | 2 | 4),
            "dSpark graph capture supports max concurrency 1, 2, or 4"
        );
    }
    if args.capture_body || args.capture_head || args.capture_update || args.capture_static {
        anyhow::ensure!(
            args.preload,
            "dSpark body/head/update/static capture requires --preload so resident pointers remain stable"
        );
    }
    let fixture = match args.fixture.trim().to_ascii_lowercase().as_str() {
        "redhat" => REDHAT_GLM52_DSPARK,
        "siro" | "siro-preview" => SIRO_GLM52_DSPARK_PREVIEW,
        other => {
            anyhow::bail!("unknown dSpark fixture {other}; expected redhat or siro")
        }
    };
    let kv_storage = DsparkKvStorage::parse(&args.kv_storage).with_context(|| {
        format!(
            "unknown dSpark KV storage {}; expected bf16 or fp8",
            args.kv_storage
        )
    })?;
    let target_catalog: TensorCatalog = serde_json::from_reader(
        fs::File::open(&args.target_catalog)
            .with_context(|| format!("opening {}", args.target_catalog.display()))?,
    )
    .with_context(|| format!("parsing {}", args.target_catalog.display()))?;
    let checkpoint = DsparkCheckpoint::from_snapshot(fixture, &args.snapshot)?;
    let engine_plan = DsparkStaticEnginePlan::new(
        &checkpoint,
        &target_catalog,
        DsparkStaticEngineConfig {
            kv_capacity_tokens: args.kv_capacity_tokens,
            kv_page_size: args.kv_page_size,
            max_concurrency: args.max_concurrency,
            kv_storage,
        },
    )?;
    let preload = args
        .preload
        .then(|| preload_dspark_draft_owned_weights(&checkpoint))
        .transpose()?;
    let head_alias_preload = (args.capture_head || args.capture_static)
        .then(|| preload_dspark_head_lm_alias(&checkpoint))
        .transpose()?;
    let query_alias_preload = (args.capture_query || args.capture_static)
        .then(|| preload_dspark_query_embedding_alias(&checkpoint))
        .transpose()?;
    let attention_graphs = args
        .capture_attention
        .then(|| {
            dspark_concurrency_buckets(args.max_concurrency)
                .into_iter()
                .map(|active_requests| {
                    benchmark_dspark_paged_attention_graph(DsparkPagedAttentionBenchConfig {
                        layers: checkpoint.validated.fixture.draft_layers,
                        active_requests,
                        query_rows: checkpoint.validated.query_layout.query_rows(),
                        context_tokens: args.attention_context_tokens,
                        kv_capacity_tokens: args.kv_capacity_tokens,
                        page_size: args.kv_page_size,
                        kv_storage,
                        warmup: args.attention_warmup,
                        iterations: args.attention_iterations,
                        repeats: args.attention_repeats,
                    })
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let query_graphs = args
        .capture_query
        .then(|| {
            let weights = preloaded_dspark_query_weights()?;
            dspark_concurrency_buckets(args.max_concurrency)
                .into_iter()
                .map(|active_requests| {
                    benchmark_dspark_query_graph(
                        weights,
                        DsparkQueryBenchConfig {
                            active_requests,
                            query_rows: checkpoint.validated.query_layout.query_rows(),
                            mask_tokens: checkpoint.validated.query_layout.query_rows() - 1,
                            mask_token_id: GLM52_DSPARK_MASK_TOKEN_ID,
                            warmup: args.query_warmup,
                            iterations: args.query_iterations,
                            repeats: args.query_repeats,
                            seed: 20_260_717 + active_requests as i64,
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let body_graphs = args
        .capture_body
        .then(|| {
            let weights = preloaded_dspark_body_weights(&checkpoint)?;
            dspark_concurrency_buckets(args.max_concurrency)
                .into_iter()
                .map(|active_requests| {
                    benchmark_dspark_body_graph(
                        weights,
                        DsparkBodyBenchConfig {
                            layers: checkpoint.validated.fixture.draft_layers,
                            active_requests,
                            query_rows: checkpoint.validated.query_layout.query_rows(),
                            context_tokens: args.body_context_tokens,
                            kv_capacity_tokens: args.kv_capacity_tokens,
                            page_size: args.kv_page_size,
                            kv_storage,
                            warmup: args.body_warmup,
                            iterations: args.body_iterations,
                            repeats: args.body_repeats,
                            seed: 20_260_717 + active_requests as i64,
                            initialize_input: true,
                            initialize_kv: true,
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let head_graphs = args
        .capture_head
        .then(|| {
            let weights = preloaded_dspark_head_weights(&checkpoint)?;
            dspark_concurrency_buckets(args.max_concurrency)
                .into_iter()
                .map(|active_requests| {
                    benchmark_dspark_head_graph(
                        weights,
                        DsparkHeadBenchConfig {
                            active_requests,
                            proposal_tokens: checkpoint.validated.query_layout.proposal_tokens(),
                            hidden_rows_per_request: checkpoint.validated.query_layout.query_rows(),
                            hidden_start_row: checkpoint
                                .validated
                                .query_layout
                                .proposal_query_slots[0],
                            warmup: args.head_warmup,
                            iterations: args.head_iterations,
                            repeats: args.head_repeats,
                            seed: 20_260_717 + active_requests as i64,
                            initialize_hidden: true,
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let update_graphs = args
        .capture_update
        .then(|| {
            let weights = preloaded_dspark_update_weights(&checkpoint)?;
            [1, 2, 4, 8, 16, 64, 128, 256, 512]
                .into_iter()
                .map(|rows| {
                    benchmark_dspark_update_graph(
                        weights,
                        DsparkUpdateBenchConfig {
                            layers: checkpoint.validated.fixture.draft_layers,
                            rows,
                            active_requests: 1,
                            context_tokens: args.update_context_tokens,
                            kv_capacity_tokens: args.kv_capacity_tokens,
                            page_size: args.kv_page_size,
                            kv_storage,
                            warmup: args.update_warmup,
                            iterations: args.update_iterations,
                            repeats: args.update_repeats,
                            seed: 20_260_717 + rows as i64,
                            initialize_target_hidden: true,
                            initialize_kv: true,
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let static_graphs = args
        .capture_static
        .then(|| {
            let weights = DsparkStaticResidentWeights {
                query: preloaded_dspark_query_weights()?,
                update: preloaded_dspark_update_weights(&checkpoint)?,
                body: preloaded_dspark_body_weights(&checkpoint)?,
                head: preloaded_dspark_head_weights(&checkpoint)?,
            };
            dspark_concurrency_buckets(args.max_concurrency)
                .into_iter()
                .map(|active_requests| {
                    benchmark_dspark_static_graph(
                        weights,
                        DsparkStaticBenchConfig {
                            draft_layers: checkpoint.validated.fixture.draft_layers,
                            active_requests,
                            query_rows: checkpoint.validated.query_layout.query_rows(),
                            proposal_tokens: checkpoint.validated.query_layout.proposal_tokens(),
                            proposal_start_row: checkpoint
                                .validated
                                .query_layout
                                .proposal_query_slots[0],
                            accepted_rows_per_request: args.static_accepted_rows_per_request,
                            context_tokens: args.static_context_tokens,
                            kv_capacity_tokens: args.kv_capacity_tokens,
                            allocate_full_kv_capacity: false,
                            page_size: args.kv_page_size,
                            kv_storage,
                            mask_token_id: GLM52_DSPARK_MASK_TOKEN_ID,
                            warmup: args.static_warmup,
                            iterations: args.static_iterations,
                            repeats: args.static_repeats,
                            seed: 20_260_717 + active_requests as i64,
                        },
                    )
                })
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    let shared_aliases = checkpoint
        .weights
        .residency
        .iter()
        .filter(|binding| binding.kind == DsparkResidentWeightKind::TargetAlias)
        .map(|binding| DsparkSharedAliasReport {
            source_name: binding.source_name.clone(),
            target_resident_name: binding.resident_name.clone(),
            byte_length: binding.byte_length,
            expected_sha256: binding
                .expected_sha256
                .expect("validated dSpark target aliases carry hashes"),
        })
        .collect();
    let report = DsparkPreflightReport {
        checkpoint_repo_id: fixture.repo_id,
        checkpoint_revision: fixture.revision,
        snapshot_path: args.snapshot.display().to_string(),
        tensor_count: checkpoint.weights.catalog.tensors.len(),
        payload_bytes: checkpoint.weights.payload_bytes,
        draft_owned_bytes: checkpoint.weights.draft_owned_bytes,
        target_aliased_bytes: checkpoint.weights.aliased_bytes,
        shared_aliases,
        engine_plan,
        preload,
        query_alias_preload,
        head_alias_preload,
        attention_graphs,
        query_graphs,
        body_graphs,
        head_graphs,
        update_graphs,
        static_graphs,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn validate_dspark_loaded_tensor(
    binding: &DsparkResidentWeightPlan,
    summary: &LoadedTensorSummary,
) -> Result<()> {
    anyhow::ensure!(
        summary.tensor_name == binding.source_name,
        "dSpark resident {} read source tensor {}",
        binding.resident_name,
        summary.tensor_name
    );
    anyhow::ensure!(
        summary.dtype == binding.dtype && summary.shape == binding.shape,
        "dSpark resident {} source geometry changed",
        binding.resident_name
    );
    anyhow::ensure!(
        summary.bytes_requested == binding.byte_length && summary.bytes_read == binding.byte_length,
        "dSpark resident {} source byte count mismatch: requested {} read {} expected {}",
        binding.resident_name,
        summary.bytes_requested,
        summary.bytes_read,
        binding.byte_length
    );
    anyhow::ensure!(
        summary.sha256.is_empty(),
        "dSpark hot preload unexpectedly hashed tensor {}",
        binding.source_name
    );
    Ok(())
}

fn dspark_concurrency_buckets(max_concurrency: usize) -> Vec<usize> {
    let mut buckets = Vec::new();
    let mut bucket = 1_usize;
    while bucket < max_concurrency {
        buckets.push(bucket);
        let Some(next) = bucket.checked_mul(2) else {
            break;
        };
        bucket = next;
    }
    if buckets.last().copied() != Some(max_concurrency) {
        buckets.push(max_concurrency);
    }
    buckets
}

fn checked_buffer_bytes(
    rows: usize,
    values_per_row: usize,
    element_bytes: usize,
    label: &str,
) -> Result<u64> {
    let bytes = rows
        .checked_mul(values_per_row)
        .and_then(|values| values.checked_mul(element_bytes))
        .with_context(|| format!("{label} byte count overflow"))?;
    bytes
        .try_into()
        .with_context(|| format!("{label} byte count does not fit in u64"))
}

fn checked_bf16_tensor_bytes(shape: &[usize]) -> Result<u64> {
    shape
        .iter()
        .try_fold(1_u64, |elements, dimension| {
            let dimension: u64 = (*dimension)
                .try_into()
                .context("dSpark tensor dimension does not fit in u64")?;
            elements
                .checked_mul(dimension)
                .context("dSpark tensor element count overflow")
        })?
        .checked_mul(std::mem::size_of::<u16>() as u64)
        .context("dSpark BF16 tensor byte count overflow")
}

fn dspark_paged_attention_metadata_bytes(
    active_requests: usize,
    kv_capacity_pages: usize,
) -> Result<u64> {
    let block_table_bytes = checked_buffer_bytes(
        active_requests,
        kv_capacity_pages,
        std::mem::size_of::<i32>(),
        "dSpark paged KV block table",
    )?;
    let length_bytes = checked_buffer_bytes(
        active_requests,
        2,
        std::mem::size_of::<i32>(),
        "dSpark paged attention lengths",
    )?;
    let offset_rows = active_requests
        .checked_add(1)
        .context("dSpark paged attention offset row count overflow")?;
    let offset_bytes = checked_buffer_bytes(
        offset_rows,
        2,
        std::mem::size_of::<i64>(),
        "dSpark paged attention offsets",
    )?;
    block_table_bytes
        .checked_add(length_bytes)
        .and_then(|bytes| bytes.checked_add(offset_bytes))
        .context("dSpark paged attention metadata byte count overflow")
}

fn dspark_resident_binding(
    fixture: DsparkPinnedFixture,
    source_name: &str,
) -> (DsparkResidentWeightKind, String, Option<&'static str>) {
    match source_name {
        "embed_tokens.weight" => (
            DsparkResidentWeightKind::TargetAlias,
            GLM52_TARGET_EMBEDDING_WEIGHT.to_owned(),
            Some(GLM52_DSPARK_EMBEDDING_SHA256),
        ),
        "lm_head.weight" => (
            DsparkResidentWeightKind::TargetAlias,
            GLM52_TARGET_LM_HEAD_WEIGHT.to_owned(),
            Some(GLM52_DSPARK_LM_HEAD_SHA256),
        ),
        _ => (
            DsparkResidentWeightKind::DraftOwned,
            format!("dspark:{}:{source_name}", fixture.revision),
            None,
        ),
    }
}

fn dspark_layer_id(name: &str) -> Option<u32> {
    let suffix = name.strip_prefix("layers.")?;
    let digits = suffix
        .chars()
        .take_while(|character| character.is_ascii_digit())
        .collect::<String>();
    (!digits.is_empty()).then(|| digits.parse().ok()).flatten()
}

fn dspark_tensor_role(name: &str) -> TensorRole {
    if name == "embed_tokens.weight" {
        TensorRole::Embedding
    } else if name == "lm_head.weight" {
        TensorRole::LmHead
    } else if name.contains(".self_attn.") {
        TensorRole::Attention
    } else if name.contains(".mlp.") {
        TensorRole::DenseMlp
    } else if name.contains("norm") {
        TensorRole::Norm
    } else {
        TensorRole::Other
    }
}

fn expected_dspark_tensor_shapes(draft_layers: usize) -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::from([
        ("confidence_head.proj.bias".to_owned(), vec![1]),
        (
            "confidence_head.proj.weight".to_owned(),
            vec![1, GLM52_DSPARK_HIDDEN_SIZE + GLM52_DSPARK_MARKOV_RANK],
        ),
        (
            "embed_tokens.weight".to_owned(),
            vec![GLM52_DSPARK_VOCAB_SIZE, GLM52_DSPARK_HIDDEN_SIZE],
        ),
        (
            "fc.weight".to_owned(),
            vec![
                GLM52_DSPARK_HIDDEN_SIZE,
                GLM52_DSPARK_TARGET_TAPS * GLM52_DSPARK_HIDDEN_SIZE,
            ],
        ),
        (
            "hidden_norm.weight".to_owned(),
            vec![GLM52_DSPARK_HIDDEN_SIZE],
        ),
        (
            "lm_head.weight".to_owned(),
            vec![GLM52_DSPARK_VOCAB_SIZE, GLM52_DSPARK_HIDDEN_SIZE],
        ),
        (
            "markov_head.markov_w1.weight".to_owned(),
            vec![GLM52_DSPARK_VOCAB_SIZE, GLM52_DSPARK_MARKOV_RANK],
        ),
        (
            "markov_head.markov_w2.weight".to_owned(),
            vec![GLM52_DSPARK_VOCAB_SIZE, GLM52_DSPARK_MARKOV_RANK],
        ),
        ("norm.weight".to_owned(), vec![GLM52_DSPARK_HIDDEN_SIZE]),
    ]);
    for layer in 0..draft_layers {
        let prefix = format!("layers.{layer}");
        for (suffix, shape) in [
            ("input_layernorm.weight", vec![GLM52_DSPARK_HIDDEN_SIZE]),
            (
                "mlp.down_proj.weight",
                vec![GLM52_DSPARK_HIDDEN_SIZE, GLM52_DSPARK_INTERMEDIATE_SIZE],
            ),
            (
                "mlp.gate_proj.weight",
                vec![GLM52_DSPARK_INTERMEDIATE_SIZE, GLM52_DSPARK_HIDDEN_SIZE],
            ),
            (
                "mlp.up_proj.weight",
                vec![GLM52_DSPARK_INTERMEDIATE_SIZE, GLM52_DSPARK_HIDDEN_SIZE],
            ),
            (
                "post_attention_layernorm.weight",
                vec![GLM52_DSPARK_HIDDEN_SIZE],
            ),
            ("self_attn.k_norm.weight", vec![GLM52_DSPARK_HEAD_DIM]),
            (
                "self_attn.k_proj.weight",
                vec![
                    GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM,
                    GLM52_DSPARK_HIDDEN_SIZE,
                ],
            ),
            (
                "self_attn.o_proj.weight",
                vec![
                    GLM52_DSPARK_HIDDEN_SIZE,
                    GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM,
                ],
            ),
            ("self_attn.q_norm.weight", vec![GLM52_DSPARK_HEAD_DIM]),
            (
                "self_attn.q_proj.weight",
                vec![
                    GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM,
                    GLM52_DSPARK_HIDDEN_SIZE,
                ],
            ),
            (
                "self_attn.v_proj.weight",
                vec![
                    GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM,
                    GLM52_DSPARK_HIDDEN_SIZE,
                ],
            ),
        ] {
            tensors.insert(format!("{prefix}.{suffix}"), shape);
        }
    }
    tensors
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DsparkHiddenTapSet<T> {
    generation: u64,
    values: [T; GLM52_DSPARK_TARGET_TAPS],
}

#[derive(Debug)]
struct DsparkHiddenTapCollector<T> {
    generation: u64,
    values: [Option<T>; GLM52_DSPARK_TARGET_TAPS],
}

impl<T> DsparkHiddenTapCollector<T> {
    fn new(generation: u64) -> Self {
        Self {
            generation,
            values: std::array::from_fn(|_| None),
        }
    }

    fn record(&mut self, generation: u64, checkpoint_layer_id: usize, value: T) -> Result<()> {
        anyhow::ensure!(
            generation == self.generation,
            "stale dSpark hidden tap generation {generation}; current generation is {}",
            self.generation
        );
        let tap_layer_ids = dspark_target_hidden_tap_layer_ids();
        let tap_index = tap_layer_ids
            .iter()
            .position(|layer_id| *layer_id == checkpoint_layer_id)
            .with_context(|| {
                format!("target layer {checkpoint_layer_id} is not a configured dSpark hidden tap")
            })?;
        anyhow::ensure!(
            self.values[tap_index].is_none(),
            "duplicate dSpark hidden tap for target layer {checkpoint_layer_id}"
        );
        self.values[tap_index] = Some(value);
        Ok(())
    }

    fn finish(mut self, generation: u64) -> Result<DsparkHiddenTapSet<T>> {
        anyhow::ensure!(
            generation == self.generation,
            "stale dSpark hidden tap finish generation {generation}; current generation is {}",
            self.generation
        );
        let tap_layer_ids = dspark_target_hidden_tap_layer_ids();
        let missing = self
            .values
            .iter()
            .enumerate()
            .filter_map(|(index, value)| value.is_none().then_some(tap_layer_ids[index]))
            .collect::<Vec<_>>();
        anyhow::ensure!(
            missing.is_empty(),
            "missing dSpark hidden taps: {missing:?}"
        );
        let values = std::array::from_fn(|index| {
            self.values[index]
                .take()
                .expect("dSpark hidden taps were checked above")
        });
        Ok(DsparkHiddenTapSet {
            generation: self.generation,
            values,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct DsparkSpsProfile {
    /// Index is the total target verification row count. Index zero is unused.
    steps_per_second: Vec<f64>,
}

impl DsparkSpsProfile {
    fn new(steps_per_second: Vec<f64>) -> Result<Self> {
        anyhow::ensure!(
            steps_per_second.len() >= 2,
            "dSpark SPS profile must cover at least target batch size one"
        );
        anyhow::ensure!(
            steps_per_second[0] == 0.0,
            "dSpark SPS profile index zero must be an unused zero"
        );
        for (batch_rows, value) in steps_per_second.iter().copied().enumerate().skip(1) {
            anyhow::ensure!(
                value.is_finite() && value > 0.0,
                "invalid dSpark SPS value {value} at target batch size {batch_rows}"
            );
        }
        Ok(Self { steps_per_second })
    }

    pub(super) fn get(&self, batch_rows: usize) -> Result<f64> {
        self.steps_per_second
            .get(batch_rows)
            .copied()
            .with_context(|| format!("dSpark SPS profile does not cover batch size {batch_rows}"))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DsparkScheduleSearch {
    /// Algorithm 1: stop at the first non-improving admission.
    CausalEarlyStop,
    /// Search every prefix when measured kernel/reduction costs are jagged.
    GlobalMaximum,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DsparkVerificationSchedule {
    pub(super) prefix_lengths: Vec<usize>,
    pub(super) target_batch_rows: usize,
    pub(super) expected_committed_tokens: f64,
    pub(super) expected_tokens_per_second: f64,
}

#[derive(Clone, Copy, Debug)]
struct DsparkPrefixCandidate {
    request_index: usize,
    prefix_length: usize,
    survival_probability: f64,
}

pub(super) fn schedule_dspark_verification(
    conditional_confidence: &[Vec<f64>],
    sps: &DsparkSpsProfile,
    search: DsparkScheduleSearch,
) -> Result<DsparkVerificationSchedule> {
    schedule_dspark_verification_with_minimums(
        conditional_confidence,
        &vec![0; conditional_confidence.len()],
        sps,
        search,
    )
}

pub(super) fn schedule_dspark_verification_with_minimums(
    conditional_confidence: &[Vec<f64>],
    minimum_prefix_lengths: &[usize],
    sps: &DsparkSpsProfile,
    search: DsparkScheduleSearch,
) -> Result<DsparkVerificationSchedule> {
    anyhow::ensure!(
        !conditional_confidence.is_empty(),
        "dSpark scheduling requires an active request"
    );
    let request_count = conditional_confidence.len();
    anyhow::ensure!(
        minimum_prefix_lengths.len() == request_count,
        "dSpark scheduling has {} minimum widths for {request_count} requests",
        minimum_prefix_lengths.len(),
    );
    let max_target_rows =
        request_count + conditional_confidence.iter().map(Vec::len).sum::<usize>();
    sps.get(max_target_rows)?;

    let mut candidates = Vec::new();
    let mut mandatory_expected_tokens = request_count as f64;
    for (request_index, confidence) in conditional_confidence.iter().enumerate() {
        anyhow::ensure!(
            minimum_prefix_lengths[request_index] <= confidence.len(),
            "dSpark minimum prefix {} exceeds request {request_index} confidence length {}",
            minimum_prefix_lengths[request_index],
            confidence.len(),
        );
        let mut survival = 1.0;
        for (position, value) in confidence.iter().copied().enumerate() {
            anyhow::ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "invalid dSpark confidence {value} for request {request_index} position {position}"
            );
            survival *= value;
            if position < minimum_prefix_lengths[request_index] {
                mandatory_expected_tokens += survival;
            } else if survival > 0.0 {
                candidates.push(DsparkPrefixCandidate {
                    request_index,
                    prefix_length: position + 1,
                    survival_probability: survival,
                });
            }
        }
    }
    candidates.sort_by(|left, right| {
        right
            .survival_probability
            .partial_cmp(&left.survival_probability)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.prefix_length.cmp(&right.prefix_length))
            .then_with(|| left.request_index.cmp(&right.request_index))
    });

    let mut current_lengths = minimum_prefix_lengths.to_vec();
    let mut best_lengths = current_lengths.clone();
    let mut target_batch_rows = request_count + minimum_prefix_lengths.iter().sum::<usize>();
    let mut expected_committed_tokens = mandatory_expected_tokens;
    let mut best_throughput = expected_committed_tokens * sps.get(target_batch_rows)?;
    let mut best_expected_tokens = expected_committed_tokens;
    let mut best_target_rows = target_batch_rows;

    for candidate in candidates {
        anyhow::ensure!(
            candidate.prefix_length == current_lengths[candidate.request_index] + 1,
            "dSpark confidence ordering violated prefix dependency for request {}: next {}, candidate {}",
            candidate.request_index,
            current_lengths[candidate.request_index] + 1,
            candidate.prefix_length
        );
        current_lengths[candidate.request_index] = candidate.prefix_length;
        target_batch_rows += 1;
        expected_committed_tokens += candidate.survival_probability;
        let throughput = expected_committed_tokens * sps.get(target_batch_rows)?;
        if throughput > best_throughput {
            best_throughput = throughput;
            best_expected_tokens = expected_committed_tokens;
            best_target_rows = target_batch_rows;
            best_lengths.clone_from(&current_lengths);
        } else if search == DsparkScheduleSearch::CausalEarlyStop {
            break;
        }
    }

    Ok(DsparkVerificationSchedule {
        prefix_lengths: best_lengths,
        target_batch_rows: best_target_rows,
        expected_committed_tokens: best_expected_tokens,
        expected_tokens_per_second: best_throughput,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SpeculationMethod {
    Mtp,
    Dspark(DsparkCheckpointConvention),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SpeculativeDraft {
    request_id: u64,
    generation: u64,
    method: SpeculationMethod,
    proposal_token_ids: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct GreedySpeculativeCommit {
    accepted_draft_tokens: usize,
    committed_token_ids: Vec<usize>,
}

#[derive(Debug)]
struct SpeculativeRequestLifecycle {
    request_id: u64,
    next_generation: u64,
}

impl SpeculativeRequestLifecycle {
    fn new(request_id: u64) -> Self {
        Self {
            request_id,
            next_generation: 0,
        }
    }

    fn begin(
        &self,
        method: SpeculationMethod,
        proposal_token_ids: Vec<usize>,
    ) -> Result<SpeculativeDraft> {
        anyhow::ensure!(
            !proposal_token_ids.is_empty(),
            "speculative draft requires at least one proposal"
        );
        Ok(SpeculativeDraft {
            request_id: self.request_id,
            generation: self.next_generation,
            method,
            proposal_token_ids,
        })
    }

    fn commit_greedy(
        &mut self,
        draft: &SpeculativeDraft,
        target_token_ids: &[usize],
    ) -> Result<GreedySpeculativeCommit> {
        anyhow::ensure!(
            draft.request_id == self.request_id,
            "speculative draft belongs to request {}, not {}",
            draft.request_id,
            self.request_id
        );
        anyhow::ensure!(
            draft.generation == self.next_generation,
            "stale speculative generation {}; expected {}",
            draft.generation,
            self.next_generation
        );
        anyhow::ensure!(
            target_token_ids.len() == draft.proposal_token_ids.len() + 1,
            "target verification must return proposal count + 1 tokens: proposals {}, target {}",
            draft.proposal_token_ids.len(),
            target_token_ids.len()
        );
        let accepted_draft_tokens = draft
            .proposal_token_ids
            .iter()
            .zip(target_token_ids)
            .take_while(|(draft_token, target_token)| draft_token == target_token)
            .count();
        let mut committed_token_ids = draft.proposal_token_ids[..accepted_draft_tokens].to_vec();
        committed_token_ids.push(target_token_ids[accepted_draft_tokens]);
        self.next_generation = self
            .next_generation
            .checked_add(1)
            .context("speculative generation overflow")?;
        Ok(GreedySpeculativeCommit {
            accepted_draft_tokens,
            committed_token_ids,
        })
    }
}

fn dspark_context_kv_bytes(
    context_tokens: usize,
    draft_layers: usize,
    element_bytes: usize,
) -> Result<usize> {
    anyhow::ensure!(
        draft_layers > 0,
        "dSpark draft layer count must be positive"
    );
    anyhow::ensure!(element_bytes > 0, "dSpark KV element size must be positive");
    context_tokens
        .checked_mul(draft_layers)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(GLM52_DSPARK_ATTENTION_HEADS))
        .and_then(|value| value.checked_mul(GLM52_DSPARK_HEAD_DIM))
        .and_then(|value| value.checked_mul(element_bytes))
        .context("dSpark context KV byte count overflow")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_weight_metadata(fixture: DsparkPinnedFixture) -> Vec<SafetensorsTensorMetadata> {
        let mut offset = 4_096_u64;
        expected_dspark_tensor_shapes(fixture.draft_layers)
            .into_iter()
            .map(|(name, shape)| {
                let byte_length = checked_bf16_tensor_bytes(&shape).unwrap();
                let metadata = SafetensorsTensorMetadata {
                    name,
                    dtype: DType::Bf16,
                    shape,
                    byte_offset: offset,
                    byte_length,
                };
                offset += byte_length;
                metadata
            })
            .collect()
    }

    fn compatible_target_alias_catalog(manifest: &DsparkWeightManifest) -> TensorCatalog {
        let tensors = manifest
            .residency
            .iter()
            .filter(|binding| binding.kind == DsparkResidentWeightKind::TargetAlias)
            .map(|binding| TensorInfo {
                name: binding.resident_name.clone(),
                file: "target.safetensors".to_owned(),
                dtype: binding.dtype.clone(),
                shape: binding.shape.clone(),
                byte_offset: 0,
                byte_length: binding.byte_length,
                role: if binding.resident_name == GLM52_TARGET_EMBEDDING_WEIGHT {
                    TensorRole::Embedding
                } else {
                    TensorRole::LmHead
                },
                layer_id: None,
                expert_id: None,
                is_quantization_metadata: false,
            })
            .collect();
        TensorCatalog {
            model_id: "target".to_owned(),
            snapshot_path: "/tmp/target".to_owned(),
            facts: ModelFacts::default(),
            tensors,
        }
    }

    fn synthetic_checkpoint(fixture: DsparkPinnedFixture) -> DsparkCheckpoint {
        DsparkCheckpoint {
            validated: ValidatedDsparkCheckpoint::from_config_json(
                fixture,
                &fixture_config(fixture),
            )
            .unwrap(),
            weights: DsparkWeightManifest::from_metadata(
                fixture,
                Path::new("/tmp/dspark"),
                complete_weight_metadata(fixture),
            )
            .unwrap(),
        }
    }

    fn fixture_config(fixture: DsparkPinnedFixture) -> String {
        format!(
            r#"{{
                "architectures":["DSparkDraftModel"],
                "aux_hidden_state_layer_ids":{},
                "block_size":{},
                "confidence_head_with_markov":true,
                "draft_vocab_size":154880,
                "enable_confidence_head":true,
                "markov_head_type":"vanilla",
                "markov_rank":256,
                "mask_token_id":154856,
                "sample_from_anchor":{},
                "speculators_model_type":"dspark",
                "speculators_config":{{
                    "proposal_methods":[{{"speculative_tokens":{}}}],
                    "verifier":{{"name_or_path":"{}"}}
                }},
                "transformer_layer_config":{{
                    "head_dim":64,
                    "hidden_size":6144,
                    "intermediate_size":12288,
                    "layer_types":{},
                    "num_attention_heads":64,
                    "num_hidden_layers":{},
                    "num_key_value_heads":64,
                    "sliding_window":{},
                    "vocab_size":154880
                }}
            }}"#,
            serde_json::to_string(&fixture.aux_hidden_state_layer_ids).unwrap(),
            fixture
                .convention
                .query_rows(fixture.proposal_tokens)
                .unwrap(),
            matches!(
                fixture.convention,
                DsparkCheckpointConvention::DeepSpecAnchorFirst
            ),
            fixture.proposal_tokens,
            fixture.verifier_repo_id,
            serde_json::to_string(&vec![
                if fixture.native_sliding_window.is_some() {
                    "sliding_attention"
                } else {
                    "full_attention"
                };
                fixture.draft_layers
            ])
            .unwrap(),
            fixture.draft_layers,
            serde_json::to_string(&fixture.native_sliding_window).unwrap(),
        )
    }

    #[test]
    fn validates_pinned_speculators_layouts() {
        for fixture in [REDHAT_GLM52_DSPARK, SIRO_GLM52_DSPARK_PREVIEW] {
            let validated =
                ValidatedDsparkCheckpoint::from_config_json(fixture, &fixture_config(fixture))
                    .unwrap();
            assert_eq!(
                validated.query_layout.query_rows(),
                fixture
                    .convention
                    .query_rows(fixture.proposal_tokens)
                    .unwrap()
            );
            let expected_slots = match fixture.convention {
                DsparkCheckpointConvention::DeepSpecAnchorFirst => {
                    (0..fixture.proposal_tokens).collect::<Vec<_>>()
                }
                DsparkCheckpointConvention::SpeculatorsBonusAnchor => {
                    (1..=fixture.proposal_tokens).collect::<Vec<_>>()
                }
            };
            assert_eq!(validated.query_layout.proposal_query_slots, expected_slots);
            assert_eq!(
                validated.query_layout.proposal_position_offsets,
                (1..=fixture.proposal_tokens).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn validates_exact_weight_manifest_and_aliases_only_shared_tables() {
        let manifest = DsparkWeightManifest::from_metadata(
            SIRO_GLM52_DSPARK_PREVIEW,
            Path::new("/tmp/dspark"),
            complete_weight_metadata(SIRO_GLM52_DSPARK_PREVIEW),
        )
        .unwrap();
        assert_eq!(
            manifest.catalog.tensors.len(),
            SIRO_GLM52_DSPARK_PREVIEW.tensor_count
        );
        assert_eq!(
            manifest.payload_bytes,
            SIRO_GLM52_DSPARK_PREVIEW.weight_payload_bytes
        );
        assert_eq!(manifest.aliased_bytes, 3_806_330_880);
        assert_eq!(manifest.draft_owned_bytes, 3_807_803_138);
        assert_eq!(
            manifest
                .residency
                .iter()
                .filter(|binding| binding.kind == DsparkResidentWeightKind::TargetAlias)
                .count(),
            2
        );
        assert_eq!(
            manifest
                .residency
                .iter()
                .filter(|binding| binding.expected_sha256.is_some())
                .map(|binding| binding.expected_sha256.unwrap())
                .collect::<Vec<_>>(),
            [GLM52_DSPARK_EMBEDDING_SHA256, GLM52_DSPARK_LM_HEAD_SHA256]
        );
        manifest
            .validate_target_aliases(&compatible_target_alias_catalog(&manifest))
            .unwrap();
    }

    #[test]
    fn rejects_manifest_dtype_shape_and_target_alias_mismatches() {
        let mut metadata = complete_weight_metadata(REDHAT_GLM52_DSPARK);
        metadata[0].dtype = DType::F16;
        assert!(DsparkWeightManifest::from_metadata(
            REDHAT_GLM52_DSPARK,
            Path::new("/tmp/dspark"),
            metadata,
        )
        .unwrap_err()
        .to_string()
        .contains("must be BF16"));

        let mut metadata = complete_weight_metadata(SIRO_GLM52_DSPARK_PREVIEW);
        metadata
            .iter_mut()
            .find(|tensor| tensor.name == "layers.3.self_attn.q_proj.weight")
            .unwrap()
            .shape[0] -= 1;
        assert!(DsparkWeightManifest::from_metadata(
            SIRO_GLM52_DSPARK_PREVIEW,
            Path::new("/tmp/dspark"),
            metadata,
        )
        .unwrap_err()
        .to_string()
        .contains("shape mismatch"));

        let manifest = DsparkWeightManifest::from_metadata(
            SIRO_GLM52_DSPARK_PREVIEW,
            Path::new("/tmp/dspark"),
            complete_weight_metadata(SIRO_GLM52_DSPARK_PREVIEW),
        )
        .unwrap();
        let mut target = compatible_target_alias_catalog(&manifest);
        target
            .tensors
            .iter_mut()
            .find(|tensor| tensor.name == GLM52_TARGET_LM_HEAD_WEIGHT)
            .unwrap()
            .byte_length -= 2;
        assert!(manifest
            .validate_target_aliases(&target)
            .unwrap_err()
            .to_string()
            .contains("incompatible"));
    }

    #[test]
    fn validates_requested_real_pinned_snapshots() {
        let requested = [
            ("GLMRT_TEST_DSPARK_REDHAT_SNAPSHOT", REDHAT_GLM52_DSPARK),
            (
                "GLMRT_TEST_DSPARK_SIRO_PREVIEW_SNAPSHOT",
                SIRO_GLM52_DSPARK_PREVIEW,
            ),
        ];
        let mut validated = 0;
        for (variable, fixture) in requested {
            let Some(snapshot) = std::env::var_os(variable) else {
                continue;
            };
            let checkpoint = DsparkCheckpoint::from_snapshot(fixture, Path::new(&snapshot))
                .unwrap_or_else(|error| panic!("validating {variable}: {error:#}"));
            assert_eq!(
                checkpoint.validated.query_layout.proposal_tokens(),
                fixture.proposal_tokens
            );
            assert_eq!(
                checkpoint.weights.catalog.tensors.len(),
                fixture.tensor_count
            );
            validated += 1;
        }
        if std::env::var_os("GLMRT_REQUIRE_DSPARK_SNAPSHOT_TESTS").is_some() {
            assert_eq!(validated, requested.len());
        }
    }

    #[test]
    fn preloads_requested_real_draft_weights() {
        let Some(snapshot) = std::env::var_os("GLMRT_TEST_DSPARK_PRELOAD_SNAPSHOT") else {
            return;
        };
        let checkpoint =
            DsparkCheckpoint::from_snapshot(SIRO_GLM52_DSPARK_PREVIEW, Path::new(&snapshot))
                .unwrap();
        let stats = preload_dspark_draft_owned_weights(&checkpoint).unwrap();
        assert_eq!(
            stats.selected_source_tensors,
            SIRO_GLM52_DSPARK_PREVIEW.tensor_count - 2
        );
        assert_eq!(stats.selected_resident_buffers, 47);
        assert_eq!(stats.selected_bytes, checkpoint.weights.draft_owned_bytes);
        assert_eq!(stats.loaded_source_tensors, stats.selected_source_tensors);
        assert_eq!(
            stats.loaded_resident_buffers,
            stats.selected_resident_buffers
        );
        assert_eq!(stats.loaded_bytes, stats.selected_bytes);
        assert!(stats.source_read_micros > 0);
        assert!(stats.total_elapsed_micros >= stats.source_read_micros);
        eprintln!(
            "dSpark preload sources={} residents={} bytes={} source_read_ms={:.3} total_ms={:.3} source_gbps={:.3}",
            stats.loaded_source_tensors,
            stats.loaded_resident_buffers,
            stats.loaded_bytes,
            stats.source_read_micros as f64 / 1_000.0,
            stats.total_elapsed_micros as f64 / 1_000.0,
            stats.loaded_bytes as f64 / stats.source_read_micros as f64 / 1_000.0,
        );
    }

    #[test]
    fn groups_qkv_and_gate_up_sources_without_duplicate_residency() {
        let checkpoint = synthetic_checkpoint(SIRO_GLM52_DSPARK_PREVIEW);
        let groups = dspark_draft_resident_groups(&checkpoint).unwrap();
        assert_eq!(groups.len(), 47);
        assert_eq!(
            groups
                .iter()
                .map(|group| group.source_names.len())
                .sum::<usize>(),
            SIRO_GLM52_DSPARK_PREVIEW.tensor_count - 2
        );
        assert_eq!(
            groups.iter().map(|group| group.byte_length).sum::<u64>(),
            checkpoint.weights.draft_owned_bytes
        );
        let qkv = groups
            .iter()
            .find(|group| {
                group
                    .resident_name
                    .ends_with("layers.3.self_attn.qkv_proj.weight")
            })
            .unwrap();
        assert_eq!(
            qkv.source_names,
            [
                "layers.3.self_attn.q_proj.weight",
                "layers.3.self_attn.k_proj.weight",
                "layers.3.self_attn.v_proj.weight",
            ]
        );
        assert_eq!(
            qkv.shape,
            [
                3 * GLM52_DSPARK_ATTENTION_HEADS * GLM52_DSPARK_HEAD_DIM,
                GLM52_DSPARK_HIDDEN_SIZE,
            ]
        );
        let gate_up = groups
            .iter()
            .find(|group| {
                group
                    .resident_name
                    .ends_with("layers.2.mlp.gate_up_proj.weight")
            })
            .unwrap();
        assert_eq!(
            gate_up.source_names,
            [
                "layers.2.mlp.gate_proj.weight",
                "layers.2.mlp.up_proj.weight",
            ]
        );
    }

    #[test]
    fn plans_static_siro_graphs_and_a_shared_256k_kv_pool() {
        let checkpoint = synthetic_checkpoint(SIRO_GLM52_DSPARK_PREVIEW);
        let target = compatible_target_alias_catalog(&checkpoint.weights);
        let plan = DsparkStaticEnginePlan::new(
            &checkpoint,
            &target,
            DsparkStaticEngineConfig {
                kv_capacity_tokens: 256 * 1_024,
                kv_page_size: 64,
                max_concurrency: 4,
                kv_storage: DsparkKvStorage::Bf16,
            },
        )
        .unwrap();
        assert_eq!(
            plan.graph_buckets
                .iter()
                .map(|bucket| bucket.active_requests)
                .collect::<Vec<_>>(),
            [1, 2, 4]
        );
        assert_eq!(
            plan.graph_buckets
                .iter()
                .map(|bucket| bucket.draft_query_rows)
                .collect::<Vec<_>>(),
            [16, 32, 64]
        );
        assert_eq!(
            plan.graph_buckets
                .iter()
                .map(|bucket| bucket.proposal_rows)
                .collect::<Vec<_>>(),
            [15, 30, 60]
        );
        assert!(plan.graph_buckets.iter().all(|bucket| {
            bucket.target_verification_rows == bucket.draft_query_rows
                && bucket.reusable_scratch_bytes == bucket.lm_logits_scratch_bytes
        }));
        assert_eq!(plan.draft_kv_bytes, 21_474_836_480);
        assert_eq!(plan.draft_kv_pages, 4_096);
        assert_eq!(plan.draft_kv_padded_tokens, 256 * 1_024);
        assert_eq!(
            plan.graph_buckets
                .iter()
                .map(|bucket| bucket.paged_attention_metadata_bytes)
                .collect::<Vec<_>>(),
            [16_424, 32_832, 65_648]
        );
        assert_eq!(plan.draft_owned_weight_bytes, 3_807_803_138);
        assert_eq!(plan.target_aliased_weight_bytes, 3_806_330_880);
        assert_eq!(
            plan.peak_incremental_device_bytes,
            plan.draft_owned_weight_bytes
                + plan.draft_kv_bytes
                + plan.flashinfer_workspace_bytes
                + plan.max_dynamic_bytes
        );
        assert!(plan.cold_capture_requires_python);
        assert_eq!(plan.hot_replay_python_calls, 0);
        assert!(!plan.serving_dispatch_enabled);
    }

    #[test]
    fn fp8_kv_planning_halves_only_the_draft_cache() {
        let checkpoint = synthetic_checkpoint(REDHAT_GLM52_DSPARK);
        let target = compatible_target_alias_catalog(&checkpoint.weights);
        let config = |kv_storage| DsparkStaticEngineConfig {
            kv_capacity_tokens: 128 * 1_024,
            kv_page_size: 64,
            max_concurrency: 4,
            kv_storage,
        };
        let bf16 = DsparkStaticEnginePlan::new(&checkpoint, &target, config(DsparkKvStorage::Bf16))
            .unwrap();
        let fp8 = DsparkStaticEnginePlan::new(&checkpoint, &target, config(DsparkKvStorage::Fp8))
            .unwrap();
        assert_eq!(
            bf16.graph_buckets
                .iter()
                .map(|bucket| bucket.draft_query_rows)
                .collect::<Vec<_>>(),
            [8, 16, 32]
        );
        assert_eq!(bf16.draft_kv_bytes, fp8.draft_kv_bytes * 2);
        assert_eq!(bf16.max_dynamic_bytes, fp8.max_dynamic_bytes);
        assert_eq!(
            bf16.peak_incremental_device_bytes - fp8.peak_incremental_device_bytes,
            fp8.draft_kv_bytes
        );
    }

    #[test]
    fn static_engine_plan_rejects_invalid_capacity_and_missing_aliases() {
        let checkpoint = synthetic_checkpoint(SIRO_GLM52_DSPARK_PREVIEW);
        let target = compatible_target_alias_catalog(&checkpoint.weights);
        let error = DsparkStaticEnginePlan::new(
            &checkpoint,
            &target,
            DsparkStaticEngineConfig {
                kv_capacity_tokens: 1,
                kv_page_size: 64,
                max_concurrency: 4,
                kv_storage: DsparkKvStorage::Bf16,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("cannot hold one"));

        let mut missing_alias = target;
        missing_alias
            .tensors
            .retain(|tensor| tensor.name != GLM52_TARGET_LM_HEAD_WEIGHT);
        let error = DsparkStaticEnginePlan::new(
            &checkpoint,
            &missing_alias,
            DsparkStaticEngineConfig {
                kv_capacity_tokens: 1_024,
                kv_page_size: 64,
                max_concurrency: 1,
                kv_storage: DsparkKvStorage::Bf16,
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("missing target tensor"));
    }

    #[test]
    fn keeps_deepspec_and_speculators_slot_conventions_distinct() {
        let original =
            DsparkQueryLayout::new(DsparkCheckpointConvention::DeepSpecAnchorFirst, 7).unwrap();
        let speculators =
            DsparkQueryLayout::new(DsparkCheckpointConvention::SpeculatorsBonusAnchor, 7).unwrap();
        assert_eq!(original.query_rows(), 7);
        assert_eq!(original.proposal_query_slots, (0..7).collect::<Vec<_>>());
        assert_eq!(speculators.query_rows(), 8);
        assert_eq!(speculators.proposal_query_slots, (1..8).collect::<Vec<_>>());
        assert_eq!(
            original.proposal_position_offsets,
            speculators.proposal_position_offsets
        );
    }

    #[test]
    fn rejects_a_silent_bonus_anchor_off_by_one() {
        let mut config: serde_json::Value =
            serde_json::from_str(&fixture_config(REDHAT_GLM52_DSPARK)).unwrap();
        config["block_size"] = 7.into();
        let error = ValidatedDsparkCheckpoint::from_config_json(
            REDHAT_GLM52_DSPARK,
            &serde_json::to_string(&config).unwrap(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("block/query mismatch"));
    }

    #[test]
    fn hidden_taps_are_generation_owned_complete_and_ordered() {
        let mut taps = DsparkHiddenTapCollector::new(11);
        for (layer_id, value) in [(39, "c"), (8, "a"), (70, "e"), (23, "b"), (55, "d")] {
            taps.record(11, layer_id, value).unwrap();
        }
        assert_eq!(taps.finish(11).unwrap().values, ["a", "b", "c", "d", "e"]);

        let mut duplicate = DsparkHiddenTapCollector::new(12);
        duplicate.record(12, 8, 1).unwrap();
        assert!(duplicate
            .record(12, 8, 2)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));

        let mut stale = DsparkHiddenTapCollector::new(13);
        assert!(stale
            .record(12, 8, 1)
            .unwrap_err()
            .to_string()
            .contains("stale"));
        stale.record(13, 8, 1).unwrap();
        assert!(stale
            .finish(13)
            .unwrap_err()
            .to_string()
            .contains("missing"));
    }

    #[test]
    fn causal_confidence_scheduler_stops_at_first_throughput_drop() {
        let profile = DsparkSpsProfile::new(vec![0.0, 100.0, 80.0, 50.0, 45.0]).unwrap();
        let schedule = schedule_dspark_verification(
            &[vec![0.9, 0.8, 0.7]],
            &profile,
            DsparkScheduleSearch::CausalEarlyStop,
        )
        .unwrap();
        assert_eq!(schedule.prefix_lengths, [1]);
        assert_eq!(schedule.target_batch_rows, 2);
        assert!((schedule.expected_committed_tokens - 1.9).abs() < 1.0e-12);
        assert!((schedule.expected_tokens_per_second - 152.0).abs() < 1.0e-12);
    }

    #[test]
    fn scheduler_preserves_each_request_prefix_and_supports_jagged_global_search() {
        let profile = DsparkSpsProfile::new(vec![0.0, 100.0, 49.0, 70.0, 65.0]).unwrap();
        let confidence = [vec![1.0, 1.0, 1.0]];
        let causal = schedule_dspark_verification(
            &confidence,
            &profile,
            DsparkScheduleSearch::CausalEarlyStop,
        )
        .unwrap();
        let global = schedule_dspark_verification(
            &confidence,
            &profile,
            DsparkScheduleSearch::GlobalMaximum,
        )
        .unwrap();
        assert_eq!(causal.prefix_lengths, [0]);
        assert_eq!(global.prefix_lengths, [3]);
        assert_eq!(global.target_batch_rows, 4);

        let flat = DsparkSpsProfile::new(vec![0.0; 1]).unwrap_err().to_string();
        assert!(flat.contains("at least"));
    }

    #[test]
    fn measured_profile_admits_the_full_physical_m16_width() {
        let profile = DsparkSpsProfile::new(
            std::iter::once(0.0)
                .chain(
                    DSPARK_PHYSICAL_M_CYCLE_MS
                        .into_iter()
                        .map(|milliseconds| 1_000.0 / milliseconds),
                )
                .collect(),
        )
        .unwrap();
        let schedule = schedule_dspark_verification(
            &[vec![1.0; 15]],
            &profile,
            DsparkScheduleSearch::GlobalMaximum,
        )
        .unwrap();
        assert_eq!(schedule.prefix_lengths, [15]);
        assert_eq!(schedule.target_batch_rows, 16);
    }

    #[test]
    fn confidence_calibrator_uses_every_observed_prefix_outcome() {
        let mut calibrator = DsparkConfidenceCalibrator::default();
        calibrator.observe(&[0.6, 0.6, 0.6, 0.6], 4);
        assert!(calibrator.logit_bias() > 0.0);

        calibrator.reset();
        calibrator.observe(&[0.9, 0.9, 0.9, 0.9], 0);
        assert!(calibrator.logit_bias() < 0.0);

        calibrator.reset();
        calibrator.observe(&[0.8, 0.8, 0.8, 0.8], 2);
        assert!(calibrator.logit_bias().is_finite());
        assert_eq!(calibrator.observation_cycles(), 1);
    }

    #[test]
    fn confidence_calibrator_bounds_bias_and_history() {
        let mut optimistic = DsparkConfidenceCalibrator::default();
        let mut pessimistic = DsparkConfidenceCalibrator::default();
        for _ in 0..64 {
            optimistic.observe(&[0.01; 15], 15);
            pessimistic.observe(&[0.99; 15], 0);
        }
        assert_eq!(
            optimistic.observation_cycles(),
            DSPARK_CONFIDENCE_CALIBRATION_WINDOW
        );
        assert_eq!(
            pessimistic.observation_cycles(),
            DSPARK_CONFIDENCE_CALIBRATION_WINDOW
        );
        assert!(optimistic.logit_bias() > 2.0);
        assert!(optimistic.logit_bias() <= DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT);
        assert!(pessimistic.logit_bias() < -2.0);
        assert!(pessimistic.logit_bias() >= -DSPARK_CONFIDENCE_LOGIT_BIAS_LIMIT);
    }

    #[test]
    fn confidence_logit_bias_changes_economic_admission_without_direct_width_tweaks() {
        let profile = DsparkSpsProfile::new(vec![0.0, 100.0, 60.0, 45.0, 35.0]).unwrap();
        let raw = [0.45, 0.8, 0.8];
        let schedule = |bias| {
            let adjusted = raw
                .into_iter()
                .map(|value| apply_dspark_confidence_logit_bias(value, bias))
                .collect::<Vec<_>>();
            schedule_dspark_verification(
                &[adjusted],
                &profile,
                DsparkScheduleSearch::CausalEarlyStop,
            )
            .unwrap()
            .prefix_lengths[0]
        };
        assert!(schedule(1.0) > schedule(-1.0));
    }

    #[test]
    fn adaptive_calibration_reacts_quickly_to_a_streaming_regime_change() {
        let mut calibrator = DsparkConfidenceCalibrator::default();
        calibrator.reset();
        for _ in 0..8 {
            calibrator.observe(&[0.8; 4], 4);
        }
        let matched_bias = calibrator.logit_bias();
        assert!(matched_bias > 0.0);
        calibrator.observe(&[0.8; 4], 0);
        calibrator.observe(&[0.8; 4], 0);
        assert!(calibrator.logit_bias() < matched_bias - 1.0);
        assert!(calibrator.posterior_variance().is_finite());
    }

    #[test]
    fn zero_draft_policy_eventually_probes_and_a_real_draft_resets_the_timer() {
        let mut calibrator = DsparkConfidenceCalibrator::default();
        calibrator.reset();
        assert!(!calibrator.force_probe_due());
        for _ in 0..DSPARK_CONFIDENCE_MAX_PROBE_INTERVAL {
            calibrator.record_selected_drafts(0);
        }
        assert!(calibrator.force_probe_due());
        calibrator.record_selected_drafts(1);
        assert!(!calibrator.force_probe_due());
    }

    #[test]
    fn confidence_context_priors_and_residual_censoring_are_explicit() {
        let mut residual = DsparkConfidenceResidual::default();
        let siro_prior = SIRO_GLM52_DSPARK_PREVIEW.confidence_context_prior;
        assert_eq!(siro_prior.at_context(siro_prior.start_tokens), 0.0);
        assert_eq!(
            siro_prior.at_context(siro_prior.start_tokens + siro_prior.ramp_tokens / 2),
            -0.4
        );
        assert_eq!(
            siro_prior.at_context(siro_prior.start_tokens + siro_prior.ramp_tokens),
            -0.8
        );
        assert_eq!(
            REDHAT_GLM52_DSPARK
                .confidence_context_prior
                .at_context(300_000),
            0.0
        );

        // One accepted position followed by the first mismatch has equal and
        // opposite pooled error at p=0.5, while retaining the position signal.
        residual.observe(&[0.5, 0.5], 1, 1_024);
        assert!(residual.dynamic_bias.abs() < 1.0e-12);
        assert!((residual.position_bias[0] - 0.005).abs() < 1.0e-12);
        assert!((residual.position_bias[1] + 0.005).abs() < 1.0e-12);
        assert_eq!(residual.observation_cycles(), 1);
    }

    #[test]
    fn confidence_residual_learns_fast_but_reopens_zero_draft_streams() {
        let mut residual = DsparkConfidenceResidual::default();
        residual.observe(&[0.5, 0.5], 2, 1_024);
        assert!((residual.dynamic_bias - 0.2).abs() < 1.0e-12);
        assert!(residual
            .position_bias
            .iter()
            .take(2)
            .all(|bias| *bias > 0.0));

        residual.record_selected_drafts(0);
        assert!((residual.dynamic_bias - 0.18).abs() < 1.0e-12);
        assert!((residual.global_logit_bias(1_024) - 0.18).abs() < 1.0e-12);

        residual.reset();
        assert_eq!(residual.global_logit_bias(1_024), 0.0);
        assert_eq!(residual.observation_cycles(), 0);
        assert!(residual.position_bias.iter().all(|bias| *bias == 0.0));
    }

    #[test]
    fn joint_scheduler_allocates_rows_to_the_request_with_better_survival() {
        let profile =
            DsparkSpsProfile::new(vec![0.0, 100.0, 100.0, 95.0, 80.0, 60.0, 40.0]).unwrap();
        let schedule = schedule_dspark_verification(
            &[vec![0.9, 0.9], vec![0.2, 0.2]],
            &profile,
            DsparkScheduleSearch::GlobalMaximum,
        )
        .unwrap();
        assert_eq!(schedule.prefix_lengths, [2, 0]);
        assert_eq!(schedule.target_batch_rows, 4);
    }

    #[test]
    fn joint_scheduler_preserves_a_required_low_confidence_probe() {
        let profile =
            DsparkSpsProfile::new(vec![0.0, 100.0, 100.0, 40.0, 30.0, 20.0, 10.0]).unwrap();
        let schedule = schedule_dspark_verification_with_minimums(
            &[vec![0.01, 0.01], vec![0.01, 0.01]],
            &[1, 0],
            &profile,
            DsparkScheduleSearch::GlobalMaximum,
        )
        .unwrap();
        assert_eq!(schedule.prefix_lengths, [1, 0]);
        assert_eq!(schedule.target_batch_rows, 3);
    }

    #[test]
    fn embedded_runtime_profile_is_immutable_by_default() {
        let mut model = DsparkRuntimeCostModel::new(2, 1).unwrap();
        model
            .install_profile(2, &[(2, 20.0), (3, 30.0), (4, 40.0)])
            .unwrap();
        assert!(model.install_profile(2, &[(2, 20.0)]).is_err());
        let before = 1_000.0 / model.profile(2, &[0, 0]).unwrap().get(2).unwrap();
        assert!((before - 20.0).abs() < 1.0e-12);
        let observation = model.observe(2, &[0, 0], 2, 1_000.0, None).unwrap();
        let after = 1_000.0 / model.profile(2, &[0, 0]).unwrap().get(2).unwrap();
        assert_eq!(observation.exact_samples, 0);
        assert_eq!(after, before);
    }

    #[test]
    fn embedded_runtime_profile_can_learn_bounded_residuals_when_enabled() {
        let mut model = DsparkRuntimeCostModel::new(2, 1).unwrap();
        model
            .install_profile(2, &[(2, 20.0), (3, 30.0), (4, 40.0)])
            .unwrap();
        model.enable_profiled_residual_learning();
        let before = 1_000.0 / model.profile(2, &[0, 0]).unwrap().get(2).unwrap();
        let observation = model.observe(2, &[0, 0], 2, 1_000.0, None).unwrap();
        let after = 1_000.0 / model.profile(2, &[0, 0]).unwrap().get(2).unwrap();
        assert_eq!(observation.exact_samples, 1);
        assert!(after > before);
        assert!(after <= before * 4.0);
    }

    #[test]
    fn glmrt_exl3_fork_preserves_only_the_nvfp4_embedded_profile() {
        let target_snapshot =
            Path::new("/tmp").join(GLM52_REDHAT_DSPARK_COST_PROFILE_TARGET_REVISION);
        let mut nvfp4 = DsparkRuntimeCostModel::new(
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_CONCURRENCY,
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap();
        let activation = install_qualified_dspark_cost_profile(
            &mut nvfp4,
            GLM52_REDHAT_DSPARK_COST_PROFILE_TARGET_MODEL,
            &target_snapshot,
            GLM52_REDHAT_DSPARK_COST_PROFILE_DSPARK_REVISION,
            Some(GLM52_REDHAT_DSPARK_COST_PROFILE_GLMRT_EXL3_COMPATIBLE_SPARKINFER_REVISION),
            Some(GLM52_REDHAT_DSPARK_COST_PROFILE_POWER_LIMIT_WATTS),
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_CONCURRENCY,
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap();
        assert!(activation.is_some());

        let mut exl3 = DsparkRuntimeCostModel::new(
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_CONCURRENCY,
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap();
        let activation = install_qualified_dspark_cost_profile(
            &mut exl3,
            glmrt_core::EXL3_MODEL_ID,
            &target_snapshot,
            GLM52_REDHAT_DSPARK_COST_PROFILE_DSPARK_REVISION,
            Some(GLM52_REDHAT_DSPARK_COST_PROFILE_GLMRT_EXL3_COMPATIBLE_SPARKINFER_REVISION),
            Some(GLM52_REDHAT_DSPARK_COST_PROFILE_POWER_LIMIT_WATTS),
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_CONCURRENCY,
            GLM52_REDHAT_DSPARK_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap();
        assert!(activation.is_none());
    }

    #[test]
    fn glm53_exl3_k4_dflash2_profile_requires_the_exact_serving_identity() {
        let target_snapshot =
            Path::new("/tmp").join(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_REVISION);
        let mut model = DsparkRuntimeCostModel::new(
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_CONCURRENCY,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap();
        let activation = install_qualified_dflash2_cost_profile(
            &mut model,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_MODEL,
            &target_snapshot,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_MODEL,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_REVISION,
            Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SPARKINFER_REVISION),
            Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_POWER_LIMIT_WATTS),
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_CONCURRENCY,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap()
        .unwrap();
        assert_eq!(activation.profile_id, GLM53_EXL3_K4_DFLASH2_COST_PROFILE_ID);
        let installed_ms = 1_000.0 / model.profile(1, &[0]).unwrap().get(1).unwrap();
        assert!((installed_ms - GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MS[0][0].1).abs() < 1.0e-9);

        let mut wrong_checkpoint = DsparkRuntimeCostModel::new(
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_CONCURRENCY,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap();
        assert!(install_qualified_dflash2_cost_profile(
            &mut wrong_checkpoint,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_TARGET_MODEL,
            &target_snapshot,
            "another/draft-checkpoint",
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_DSPARK_REVISION,
            Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_SPARKINFER_REVISION),
            Some(GLM53_EXL3_K4_DFLASH2_COST_PROFILE_POWER_LIMIT_WATTS),
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_CONCURRENCY,
            GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MAX_DRAFTS,
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn route_conditioned_runtime_residual_waits_for_route_coverage() {
        let mut model = DsparkRuntimeCostModel::new(2, 1).unwrap();
        let contexts = [0, 0];
        let before = 1_000.0 / model.profile(2, &contexts).unwrap().get(2).unwrap();
        for sample in 0..7 {
            let observation = model
                .observe(2, &contexts, 2, before * 2.0, Some(2_000 + sample % 2))
                .unwrap();
            assert_eq!(observation.exact_samples, (sample + 1) as u64);
        }
        let before_coverage = 1_000.0 / model.profile(2, &contexts).unwrap().get(2).unwrap();
        assert!((before_coverage - before).abs() < 1.0e-12);
        model
            .observe(2, &contexts, 2, before * 2.0, Some(2_001))
            .unwrap();
        let after_coverage = 1_000.0 / model.profile(2, &contexts).unwrap().get(2).unwrap();
        assert!(after_coverage > before);
    }

    #[test]
    fn runtime_cost_model_covers_c16_m16_and_learns_context_cells() {
        let mut model = DsparkRuntimeCostModel::new(16, 15).unwrap();
        let redhat_model = DsparkRuntimeCostModel::new(16, 7).unwrap();
        assert!(redhat_model
            .profile(16, &vec![400_000; 16])
            .unwrap()
            .get(128)
            .unwrap()
            .is_finite());
        assert!(DsparkRuntimeCostModel::new(16, 16).is_err());
        let short_contexts = vec![1_024; 2];
        let long_contexts = vec![131_072; 2];
        assert_eq!(
            DsparkRuntimeCostModel::context_buckets(&[1_024, 100_000]).unwrap(),
            (3, 3)
        );
        assert_eq!(
            DsparkRuntimeCostModel::context_buckets(&[100_000, 100_000]).unwrap(),
            (6, 3)
        );
        let short_before = 1_000.0 / model.profile(2, &short_contexts).unwrap().get(8).unwrap();
        let long_before = 1_000.0 / model.profile(2, &long_contexts).unwrap().get(8).unwrap();
        let unseen_row_before =
            1_000.0 / model.profile(2, &long_contexts).unwrap().get(12).unwrap();
        assert!(long_before > short_before);
        for _ in 0..4 {
            model.observe(2, &long_contexts, 8, 300.0, None).unwrap();
        }
        let long_after = 1_000.0 / model.profile(2, &long_contexts).unwrap().get(8).unwrap();
        let unseen_row_after = 1_000.0 / model.profile(2, &long_contexts).unwrap().get(12).unwrap();
        assert!(long_after > long_before);
        assert!(unseen_row_after > unseen_row_before);
        let max_profile = model.profile(16, &vec![400_000; 16]).unwrap();
        assert!(max_profile.get(16 * 16).unwrap().is_finite());
    }

    #[test]
    fn greedy_commit_is_shared_by_mtp_and_dspark_and_rejects_stale_drafts() {
        for method in [
            SpeculationMethod::Mtp,
            SpeculationMethod::Dspark(DsparkCheckpointConvention::SpeculatorsBonusAnchor),
        ] {
            let mut lifecycle = SpeculativeRequestLifecycle::new(41);
            let draft = lifecycle.begin(method, vec![10, 20, 30]).unwrap();
            let commit = lifecycle.commit_greedy(&draft, &[10, 99, 88, 77]).unwrap();
            assert_eq!(commit.accepted_draft_tokens, 1);
            assert_eq!(commit.committed_token_ids, [10, 99]);
            assert!(lifecycle
                .commit_greedy(&draft, &[10, 99, 88, 77])
                .unwrap_err()
                .to_string()
                .contains("stale"));
        }

        let mut lifecycle = SpeculativeRequestLifecycle::new(7);
        let all = lifecycle
            .begin(SpeculationMethod::Mtp, vec![1, 2, 3])
            .unwrap();
        let commit = lifecycle.commit_greedy(&all, &[1, 2, 3, 4]).unwrap();
        assert_eq!(commit.accepted_draft_tokens, 3);
        assert_eq!(commit.committed_token_ids, [1, 2, 3, 4]);
    }

    #[test]
    fn draft_context_kv_capacity_is_explicit() {
        assert_eq!(dspark_context_kv_bytes(1, 5, 2).unwrap(), 81_920);
        assert_eq!(dspark_context_kv_bytes(1_024, 5, 2).unwrap(), 83_886_080);
        assert_eq!(
            dspark_context_kv_bytes(128 * 1_024, 5, 2).unwrap(),
            10_737_418_240
        );
        assert_eq!(
            dspark_context_kv_bytes(256 * 1_024, 5, 1).unwrap(),
            10_737_418_240
        );
        assert_eq!(dspark_context_kv_bytes(1, 3, 2).unwrap(), 49_152);
    }

    #[test]
    fn page_aligned_swa_rotation_preserves_a_scratch_page() {
        let mut page_table = (0..34).collect::<Vec<i32>>();
        assert_eq!(
            roll_dspark_swa_page_table(&mut page_table, 61, 7, 64, 64).unwrap(),
            (61, 0)
        );
        assert_eq!(
            roll_dspark_swa_page_table(&mut page_table, 2_040, 8, 2_048, 64).unwrap(),
            (2_040, 0)
        );
        assert_eq!(page_table, (0..34).collect::<Vec<i32>>());

        assert_eq!(
            roll_dspark_swa_page_table(&mut page_table, 2_111, 1, 2_048, 64).unwrap(),
            (2_047, 1)
        );
        assert_eq!(page_table[0], 1);
        assert_eq!(page_table[32], 33);
        assert_eq!(page_table[33], 0);

        let mut no_scratch = (0..32).collect::<Vec<i32>>();
        assert!(roll_dspark_swa_page_table(&mut no_scratch, 2_048, 1, 2_048, 64).is_err());
    }

    #[test]
    fn serving_draft_slots_have_disjoint_compact_page_tables() {
        let tables = (0..16)
            .map(|slot| dspark_request_slot_page_table(slot, 34).unwrap())
            .collect::<Vec<_>>();
        assert!(tables.iter().all(|table| table.len() == 34));
        assert_eq!(tables[0], (0..34).collect::<Vec<i32>>());
        assert_eq!(tables[1], (34..68).collect::<Vec<i32>>());
        assert_eq!(tables[15], (510..544).collect::<Vec<i32>>());
        assert_eq!(
            tables
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>()
                .len(),
            544
        );
    }
}
