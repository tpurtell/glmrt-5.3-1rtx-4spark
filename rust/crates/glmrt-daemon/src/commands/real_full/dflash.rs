#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use glmrt_core::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
use glmrt_ffi::GlmrtDeviceBuffer;
use glmrt_loader::{
    read_safetensors_metadata, read_tensor_bytes_into, LoadedTensorSummary,
    SafetensorsTensorMetadata,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::coordinator_kernels::{
    preload_resident_weight_from_host_staging_profiled, preloaded_resident_weight_device_buffer,
    DeviceBf16Output,
};
use super::dflash_static::{
    dflash2_update_graph_buckets, Dflash2BatchedSuffixRequest, Dflash2BatchedUpdateRequest,
    Dflash2DraftStep, Dflash2StaticBenchConfig, Dflash2StaticExecutor,
};
use super::dflash_update::dflash2_update_resident_weights;
use super::dspark_kv::DsparkKvStorage;

const GLM53_DFLASH2_REPO_ID: &str = "incoai/GLM-5.3-DFlash2";
const GLM53_DFLASH2_REVISION: &str = "425aa615ce320caac34400208b30808c8f14f76c";
const GLM53_DFLASH2_CONFIG_SHA256: &str =
    "f59e1da17d41d24a1aba588aecee1607788adb34a03805f2c883add8ca954e9b";
const GLM53_DFLASH2_WEIGHT_LFS_SHA256: &str =
    "3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161";
const GLM53_DFLASH2_TARGET_REPO_ID: &str = "zai-org/GLM-5.3";
const GLM53_DFLASH2_ARCHITECTURE: &str = "DFlash2DraftModel";
const GLM53_DFLASH2_WEIGHT_BYTES: u64 = 4_918_859_112;
const GLM53_DFLASH2_WEIGHT_PAYLOAD_BYTES: u64 = 4_918_848_512;
const GLM53_DFLASH2_TENSOR_COUNT: usize = 96;
const GLM53_DFLASH2_SERVING_ENV: &str = "GLMRT_REAL_FULL_DFLASH2";

pub(super) const GLM53_DFLASH2_HIDDEN_SIZE: usize = 6_144;
pub(super) const GLM53_DFLASH2_INTERMEDIATE_SIZE: usize = 12_288;
pub(super) const GLM53_DFLASH2_VOCAB_SIZE: usize = 154_880;
pub(super) const GLM53_DFLASH2_DRAFT_LAYERS: usize = 6;
pub(super) const GLM53_DFLASH2_TARGET_LAYERS: usize = 78;
// The checkpoint advertises its maximum block geometry. Serving chooses a
// true internal K in 1..=7 and captures a K+1 row body/head graph.
pub(super) const GLM53_DFLASH2_BLOCK_SIZE: usize = 8;
pub(super) const GLM53_DFLASH2_MAX_DRAFTS: usize = GLM53_DFLASH2_BLOCK_SIZE - 1;
pub(super) const GLM53_DFLASH2_ATTENTION_HEADS: usize = 64;
pub(super) const GLM53_DFLASH2_KV_HEADS: usize = 8;
pub(super) const GLM53_DFLASH2_HEAD_DIM: usize = 128;
pub(super) const GLM53_DFLASH2_SLIDING_WINDOW: usize = 2_048;
pub(super) const GLM53_DFLASH2_CONV_KERNEL_SIZE: usize = 2;
pub(super) const GLM53_DFLASH2_CONV_GROUP_SIZE: usize = 16;
pub(super) const GLM53_DFLASH2_SELECTOR_RANK: usize = 256;
pub(super) const GLM53_DFLASH2_SELECTOR_TOP_K: usize = 16;
pub(super) const GLM53_DFLASH2_MASK_TOKEN_ID: usize = 154_856;
pub(super) const GLM53_DFLASH2_TARGET_TAPS: [usize; GLM53_DFLASH2_DRAFT_LAYERS] =
    [5, 19, 33, 47, 61, 75];
// Transformers stores the embedding at hidden_states[0] and each post-layer
// state after it. GLMRT capture hooks are post-layer boundaries numbered
// 1..=78, so config layer IDs are shifted by one for live capture.
pub(super) const GLM53_DFLASH2_TARGET_CAPTURE_TAPS: [usize; GLM53_DFLASH2_DRAFT_LAYERS] =
    [6, 20, 34, 48, 62, 76];

pub(super) fn dflash2_serving_requested() -> bool {
    env::var(GLM53_DFLASH2_SERVING_ENV)
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

const GLM53_TARGET_EMBEDDING_WEIGHT: &str = "model.embed_tokens.weight";
const GLM53_TARGET_LM_HEAD_WEIGHT: &str = "lm_head.weight";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Dflash2PinnedFixture {
    pub(super) repo_id: &'static str,
    pub(super) revision: &'static str,
    pub(super) target_repo_id: &'static str,
    pub(super) config_sha256: &'static str,
    pub(super) weight_lfs_sha256: &'static str,
    pub(super) weight_bytes: u64,
    pub(super) weight_payload_bytes: u64,
    pub(super) tensor_count: usize,
}

pub(super) const GLM53_DFLASH2: Dflash2PinnedFixture = Dflash2PinnedFixture {
    repo_id: GLM53_DFLASH2_REPO_ID,
    revision: GLM53_DFLASH2_REVISION,
    target_repo_id: GLM53_DFLASH2_TARGET_REPO_ID,
    config_sha256: GLM53_DFLASH2_CONFIG_SHA256,
    weight_lfs_sha256: GLM53_DFLASH2_WEIGHT_LFS_SHA256,
    weight_bytes: GLM53_DFLASH2_WEIGHT_BYTES,
    weight_payload_bytes: GLM53_DFLASH2_WEIGHT_PAYLOAD_BYTES,
    tensor_count: GLM53_DFLASH2_TENSOR_COUNT,
};

#[derive(Debug, Deserialize)]
struct Dflash2Config {
    architectures: Vec<String>,
    #[serde(default)]
    attention_bias: bool,
    dflash_config: Dflash2AlgorithmConfig,
    dtype: String,
    head_dim: usize,
    hidden_act: String,
    hidden_size: usize,
    intermediate_size: usize,
    is_causal: bool,
    layer_types: Vec<String>,
    max_window_layers: usize,
    max_position_embeddings: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_parameters: Dflash2RopeConfig,
    sliding_window: usize,
    tie_word_embeddings: bool,
    use_cache: bool,
    use_sliding_window: bool,
    vocab_size: usize,
}

#[derive(Debug, Deserialize)]
struct Dflash2RopeConfig {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Debug, Deserialize)]
struct Dflash2AlgorithmConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: usize,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ValidatedDflash2Checkpoint {
    pub(super) fixture: Dflash2PinnedFixture,
    pub(super) target_layer_ids: [usize; GLM53_DFLASH2_DRAFT_LAYERS],
}

impl ValidatedDflash2Checkpoint {
    fn from_config_json(fixture: Dflash2PinnedFixture, config_json: &str) -> Result<Self> {
        let config: Dflash2Config =
            serde_json::from_str(config_json).context("parsing DFlash2 config.json")?;
        anyhow::ensure!(
            config.architectures == [GLM53_DFLASH2_ARCHITECTURE],
            "DFlash2 architecture mismatch: {:?}",
            config.architectures
        );
        anyhow::ensure!(
            config.dtype.eq_ignore_ascii_case("bfloat16")
                && !config.attention_bias
                && config.hidden_act == "silu"
                && !config.is_causal
                && !config.tie_word_embeddings
                && !config.use_cache,
            "GLM-5.3 DFlash2 base transformer contract changed"
        );
        anyhow::ensure!(
            config.hidden_size == GLM53_DFLASH2_HIDDEN_SIZE
                && config.intermediate_size == GLM53_DFLASH2_INTERMEDIATE_SIZE
                && config.num_hidden_layers == GLM53_DFLASH2_DRAFT_LAYERS
                && config.num_attention_heads == GLM53_DFLASH2_ATTENTION_HEADS
                && config.num_key_value_heads == GLM53_DFLASH2_KV_HEADS
                && config.head_dim == GLM53_DFLASH2_HEAD_DIM
                && config.vocab_size == GLM53_DFLASH2_VOCAB_SIZE,
            "GLM-5.3 DFlash2 transformer geometry changed"
        );
        anyhow::ensure!(
            config.layer_types == vec!["sliding_attention".to_owned(); GLM53_DFLASH2_DRAFT_LAYERS]
                && config.use_sliding_window
                && config.max_window_layers == GLM53_DFLASH2_DRAFT_LAYERS
                && config.sliding_window == GLM53_DFLASH2_SLIDING_WINDOW,
            "GLM-5.3 DFlash2 attention/window contract changed"
        );
        anyhow::ensure!(
            config.rms_norm_eps == 1.0e-5,
            "GLM-5.3 DFlash2 RMS norm epsilon changed: {}",
            config.rms_norm_eps
        );
        anyhow::ensure!(
            config.max_position_embeddings == 1_048_576
                && config.rope_parameters.rope_theta == 1_000_000.0
                && config.rope_parameters.rope_type == "default",
            "GLM-5.3 DFlash2 rotary-position contract changed"
        );
        let draft = config.dflash_config;
        anyhow::ensure!(
            draft.block_size == GLM53_DFLASH2_BLOCK_SIZE
                && draft.conv_kernel_size == GLM53_DFLASH2_CONV_KERNEL_SIZE
                && draft.conv_group_size == GLM53_DFLASH2_CONV_GROUP_SIZE
                && draft.mask_token_id == GLM53_DFLASH2_MASK_TOKEN_ID
                && draft.selector_rank == GLM53_DFLASH2_SELECTOR_RANK
                && draft.selector_top_k == GLM53_DFLASH2_SELECTOR_TOP_K,
            "GLM-5.3 DFlash2 block, convolution, or selector contract changed"
        );
        let target_layer_ids: [usize; GLM53_DFLASH2_DRAFT_LAYERS] = draft
            .target_layer_ids
            .try_into()
            .map_err(|values: Vec<usize>| {
                anyhow::anyhow!(
                    "GLM-5.3 DFlash2 requires {} target taps, got {:?}",
                    GLM53_DFLASH2_DRAFT_LAYERS,
                    values
                )
            })?;
        anyhow::ensure!(
            target_layer_ids == GLM53_DFLASH2_TARGET_TAPS
                && target_layer_ids
                    .iter()
                    .all(|layer_id| *layer_id < GLM53_DFLASH2_TARGET_LAYERS),
            "GLM-5.3 DFlash2 target taps changed: {:?}",
            target_layer_ids
        );
        Ok(Self {
            fixture,
            target_layer_ids,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Dflash2ResidentWeightPlan {
    pub(super) name: String,
    pub(super) dtype: DType,
    pub(super) shape: Vec<usize>,
    pub(super) byte_offset: u64,
    pub(super) byte_length: u64,
}

#[derive(Clone, Debug)]
pub(super) struct Dflash2WeightManifest {
    pub(super) weight_path: PathBuf,
    catalog: TensorCatalog,
    pub(super) residency: Vec<Dflash2ResidentWeightPlan>,
    pub(super) payload_bytes: u64,
}

impl Dflash2WeightManifest {
    fn from_snapshot(fixture: Dflash2PinnedFixture, snapshot: &Path) -> Result<Self> {
        let weight_path = snapshot.join("model.safetensors");
        let metadata = read_safetensors_metadata(&weight_path).with_context(|| {
            format!(
                "reading GLM-5.3 DFlash2 safetensors metadata from {}",
                weight_path.display()
            )
        })?;
        Self::from_metadata(fixture, weight_path, metadata)
    }

    fn from_metadata(
        fixture: Dflash2PinnedFixture,
        weight_path: PathBuf,
        metadata: Vec<SafetensorsTensorMetadata>,
    ) -> Result<Self> {
        let expected = expected_dflash2_tensor_shapes();
        anyhow::ensure!(
            expected.len() == fixture.tensor_count,
            "internal DFlash2 manifest has {} tensors, expected {}",
            expected.len(),
            fixture.tensor_count
        );
        anyhow::ensure!(
            metadata.len() == fixture.tensor_count,
            "DFlash2 tensor count mismatch for {}: expected {}, got {}",
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
            "unexpected DFlash2 tensors for {}: {unexpected:?}",
            fixture.repo_id
        );

        let mut residency = Vec::with_capacity(expected.len());
        let mut payload_bytes = 0_u64;
        for (name, shape) in expected {
            let tensor = actual
                .get(name.as_str())
                .with_context(|| format!("missing DFlash2 tensor {name}"))?;
            anyhow::ensure!(
                tensor.dtype == DType::Bf16 && tensor.shape == shape,
                "DFlash2 tensor {name} mismatch: dtype={:?} shape={:?}, expected BF16 {:?}",
                tensor.dtype,
                tensor.shape,
                shape
            );
            let expected_bytes = checked_bf16_tensor_bytes(&shape)
                .with_context(|| format!("computing DFlash2 tensor {name} bytes"))?;
            anyhow::ensure!(
                tensor.byte_length == expected_bytes,
                "DFlash2 tensor {name} byte mismatch: expected {expected_bytes}, got {}",
                tensor.byte_length
            );
            payload_bytes = payload_bytes
                .checked_add(tensor.byte_length)
                .context("DFlash2 payload byte count overflow")?;
            residency.push(Dflash2ResidentWeightPlan {
                name,
                dtype: tensor.dtype.clone(),
                shape,
                byte_offset: tensor.byte_offset,
                byte_length: tensor.byte_length,
            });
        }
        anyhow::ensure!(
            payload_bytes == fixture.weight_payload_bytes,
            "DFlash2 payload mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.weight_payload_bytes,
            payload_bytes
        );
        let snapshot_path = weight_path
            .parent()
            .context("DFlash2 model.safetensors has no parent directory")?;
        let file_name = weight_path
            .file_name()
            .and_then(|name| name.to_str())
            .context("DFlash2 model.safetensors has a non-UTF-8 filename")?
            .to_owned();
        let catalog = TensorCatalog {
            model_id: fixture.repo_id.to_owned(),
            snapshot_path: snapshot_path.display().to_string(),
            facts: ModelFacts::default(),
            tensors: residency
                .iter()
                .map(|tensor| TensorInfo {
                    name: tensor.name.clone(),
                    file: file_name.clone(),
                    dtype: tensor.dtype.clone(),
                    shape: tensor.shape.clone(),
                    byte_offset: tensor.byte_offset,
                    byte_length: tensor.byte_length,
                    role: TensorRole::Other,
                    layer_id: None,
                    expert_id: None,
                    is_quantization_metadata: false,
                })
                .collect(),
        };
        Ok(Self {
            weight_path,
            catalog,
            residency,
            payload_bytes,
        })
    }
}

#[derive(Clone, Debug)]
pub(super) struct Dflash2Checkpoint {
    pub(super) validated: ValidatedDflash2Checkpoint,
    pub(super) weights: Dflash2WeightManifest,
}

impl Dflash2Checkpoint {
    pub(super) fn from_snapshot(snapshot: &Path) -> Result<Self> {
        let fixture = GLM53_DFLASH2;
        anyhow::ensure!(
            snapshot.file_name().and_then(|name| name.to_str()) == Some(fixture.revision),
            "DFlash2 snapshot revision mismatch for {}: expected {} at {}",
            fixture.repo_id,
            fixture.revision,
            snapshot.display()
        );
        let config_path = snapshot.join("config.json");
        let config_bytes = fs::read(&config_path)
            .with_context(|| format!("reading DFlash2 config from {}", config_path.display()))?;
        let config_sha256 = format!("{:x}", Sha256::digest(&config_bytes));
        anyhow::ensure!(
            config_sha256 == fixture.config_sha256,
            "DFlash2 config identity mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.config_sha256,
            config_sha256
        );
        let config_json =
            std::str::from_utf8(&config_bytes).context("DFlash2 config.json is not UTF-8")?;
        let validated = ValidatedDflash2Checkpoint::from_config_json(fixture, config_json)?;
        let weight_path = snapshot.join("model.safetensors");
        validate_hf_lfs_blob_identity(&weight_path, fixture.weight_lfs_sha256)?;
        let weight_bytes = fs::metadata(&weight_path)
            .with_context(|| format!("reading DFlash2 weights from {}", weight_path.display()))?
            .len();
        anyhow::ensure!(
            weight_bytes == fixture.weight_bytes,
            "DFlash2 weight size mismatch for {}: expected {}, got {}",
            fixture.repo_id,
            fixture.weight_bytes,
            weight_bytes
        );
        let weights = Dflash2WeightManifest::from_snapshot(fixture, snapshot)?;
        Ok(Self { validated, weights })
    }
}

fn validate_hf_lfs_blob_identity(path: &Path, expected_sha256: &str) -> Result<()> {
    let metadata = fs::symlink_metadata(path)
        .with_context(|| format!("reading DFlash2 LFS link metadata from {}", path.display()))?;
    anyhow::ensure!(
        metadata.file_type().is_symlink(),
        "DFlash2 weights must use the standard Hugging Face snapshot symlink: {}",
        path.display()
    );
    let target = fs::read_link(path)
        .with_context(|| format!("reading DFlash2 LFS link from {}", path.display()))?;
    anyhow::ensure!(
        target.file_name().and_then(|name| name.to_str()) == Some(expected_sha256),
        "DFlash2 LFS identity mismatch: expected {expected_sha256}, got {}",
        target.display()
    );
    let resolved = fs::canonicalize(path)
        .with_context(|| format!("resolving DFlash2 LFS link {}", path.display()))?;
    anyhow::ensure!(
        resolved.file_name().and_then(|name| name.to_str()) == Some(expected_sha256),
        "DFlash2 resolved LFS identity mismatch: expected {expected_sha256}, got {}",
        resolved.display()
    );
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Dflash2ResidentGroup {
    resident_name: String,
    source_names: Vec<String>,
    shape: Vec<usize>,
    byte_length: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(super) struct Dflash2ResidentPreloadStats {
    pub(super) source_tensors: usize,
    pub(super) resident_buffers: usize,
    pub(super) selected_bytes: u64,
    pub(super) loaded_source_tensors: usize,
    pub(super) loaded_resident_buffers: usize,
    pub(super) loaded_bytes: u64,
    pub(super) source_read_micros: u128,
    pub(super) total_elapsed_micros: u128,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub(super) struct Dflash2TargetAliasPreloadStats {
    pub(super) selected_tensors: usize,
    pub(super) selected_bytes: u64,
    pub(super) loaded_tensors: usize,
    pub(super) loaded_bytes: u64,
    pub(super) source_read_micros: u128,
    pub(super) total_elapsed_micros: u128,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2LayerResidentWeights {
    pub(super) attention_conv_base: GlmrtDeviceBuffer,
    pub(super) attention_conv_projection: GlmrtDeviceBuffer,
    pub(super) mlp_conv_base: GlmrtDeviceBuffer,
    pub(super) mlp_conv_projection: GlmrtDeviceBuffer,
    pub(super) input_norm: GlmrtDeviceBuffer,
    pub(super) post_norm: GlmrtDeviceBuffer,
    pub(super) q_norm: GlmrtDeviceBuffer,
    pub(super) k_norm: GlmrtDeviceBuffer,
    pub(super) qkv: GlmrtDeviceBuffer,
    pub(super) output: GlmrtDeviceBuffer,
    pub(super) gate_up: GlmrtDeviceBuffer,
    pub(super) down: GlmrtDeviceBuffer,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Dflash2ResidentWeights {
    pub(super) target_embedding: GlmrtDeviceBuffer,
    pub(super) target_lm_head: GlmrtDeviceBuffer,
    pub(super) target_fusion: GlmrtDeviceBuffer,
    pub(super) hidden_norm: GlmrtDeviceBuffer,
    pub(super) final_norm: GlmrtDeviceBuffer,
    pub(super) selector_hidden_projection: GlmrtDeviceBuffer,
    pub(super) selector_predecessor: GlmrtDeviceBuffer,
    pub(super) selector_successor: GlmrtDeviceBuffer,
    pub(super) layers: [Dflash2LayerResidentWeights; GLM53_DFLASH2_DRAFT_LAYERS],
    pub(super) draft_resident_bytes: u64,
}

fn dflash2_resident_groups(checkpoint: &Dflash2Checkpoint) -> Result<Vec<Dflash2ResidentGroup>> {
    let bindings = checkpoint
        .weights
        .residency
        .iter()
        .map(|binding| (binding.name.as_str(), binding))
        .collect::<BTreeMap<_, _>>();
    let mut consumed = BTreeSet::new();
    let mut groups = Vec::new();
    let q_width = GLM53_DFLASH2_ATTENTION_HEADS * GLM53_DFLASH2_HEAD_DIM;
    let kv_width = GLM53_DFLASH2_KV_HEADS * GLM53_DFLASH2_HEAD_DIM;
    for layer in 0..GLM53_DFLASH2_DRAFT_LAYERS {
        for (suffix, source_suffixes, shape) in [
            (
                "self_attn.qkv_proj.weight",
                vec![
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                ],
                vec![q_width + 2 * kv_width, GLM53_DFLASH2_HIDDEN_SIZE],
            ),
            (
                "mlp.gate_up_proj.weight",
                vec!["mlp.gate_proj.weight", "mlp.up_proj.weight"],
                vec![
                    2 * GLM53_DFLASH2_INTERMEDIATE_SIZE,
                    GLM53_DFLASH2_HIDDEN_SIZE,
                ],
            ),
        ] {
            let source_names = source_suffixes
                .into_iter()
                .map(|source_suffix| format!("layers.{layer}.{source_suffix}"))
                .collect::<Vec<_>>();
            let byte_length = source_names.iter().try_fold(0_u64, |bytes, source_name| {
                let binding = bindings.get(source_name.as_str()).with_context(|| {
                    format!("missing DFlash2 fused resident source {source_name}")
                })?;
                consumed.insert(source_name.clone());
                bytes
                    .checked_add(binding.byte_length)
                    .context("DFlash2 fused resident byte count overflow")
            })?;
            anyhow::ensure!(
                byte_length == checked_bf16_tensor_bytes(&shape)?,
                "DFlash2 fused resident layers.{layer}.{suffix} byte count mismatch"
            );
            groups.push(Dflash2ResidentGroup {
                resident_name: format!(
                    "dflash2:{}:layers.{layer}.{suffix}",
                    checkpoint.validated.fixture.revision
                ),
                source_names,
                shape,
                byte_length,
            });
        }
    }
    for binding in bindings.values() {
        if consumed.contains(&binding.name) {
            continue;
        }
        groups.push(Dflash2ResidentGroup {
            resident_name: format!(
                "dflash2:{}:{}",
                checkpoint.validated.fixture.revision, binding.name
            ),
            source_names: vec![binding.name.clone()],
            shape: binding.shape.clone(),
            byte_length: binding.byte_length,
        });
    }
    groups.sort_by(|left, right| left.resident_name.cmp(&right.resident_name));
    let source_count: usize = groups.iter().map(|group| group.source_names.len()).sum();
    let resident_bytes: u64 = groups.iter().map(|group| group.byte_length).sum();
    anyhow::ensure!(
        source_count == checkpoint.validated.fixture.tensor_count
            && resident_bytes == checkpoint.weights.payload_bytes,
        "DFlash2 grouped resident plan mismatch: sources {source_count} bytes {resident_bytes}"
    );
    Ok(groups)
}

pub(super) fn preload_dflash2_resident_weights(
    checkpoint: &Dflash2Checkpoint,
) -> Result<Dflash2ResidentPreloadStats> {
    let started = Instant::now();
    let bindings = checkpoint
        .weights
        .residency
        .iter()
        .map(|binding| (binding.name.as_str(), binding))
        .collect::<BTreeMap<_, _>>();
    let groups = dflash2_resident_groups(checkpoint)?;
    let mut stats = Dflash2ResidentPreloadStats {
        source_tensors: groups.iter().map(|group| group.source_names.len()).sum(),
        resident_buffers: groups.len(),
        selected_bytes: groups.iter().map(|group| group.byte_length).sum(),
        ..Dflash2ResidentPreloadStats::default()
    };
    let mut device_allocation_ms = 0.0_f64;
    let mut staging_allocation_ms = 0.0_f64;
    let mut staging_fill_ms = 0.0_f64;
    let mut h2d_ms = 0.0_f64;
    for group in groups {
        let expected_bytes: usize = group.byte_length.try_into().with_context(|| {
            format!(
                "DFlash2 resident {} byte length {} does not fit usize",
                group.resident_name, group.byte_length
            )
        })?;
        let mut loaded = Vec::<LoadedTensorSummary>::new();
        let timing = preload_resident_weight_from_host_staging_profiled(
            &group.resident_name,
            expected_bytes,
            "startup resident GLM-5.3 DFlash2 weight pinned staging",
            |staging| {
                let mut offset = 0_usize;
                for source_name in &group.source_names {
                    let binding = bindings.get(source_name.as_str()).with_context(|| {
                        format!("missing DFlash2 resident source binding {source_name}")
                    })?;
                    let source_bytes: usize =
                        binding.byte_length.try_into().with_context(|| {
                            format!("DFlash2 source {source_name} byte count does not fit usize")
                        })?;
                    let end = offset
                        .checked_add(source_bytes)
                        .context("DFlash2 fused staging offset overflow")?;
                    let summary = read_tensor_bytes_into(
                        &checkpoint.weights.catalog,
                        source_name,
                        &mut staging[offset..end],
                    )
                    .with_context(|| format!("reading DFlash2 tensor {source_name}"))?;
                    validate_dflash2_loaded_tensor(binding, &summary)?;
                    loaded.push(summary);
                    offset = end;
                }
                anyhow::ensure!(
                    offset == expected_bytes,
                    "DFlash2 resident {} staged {} bytes, expected {}",
                    group.resident_name,
                    offset,
                    expected_bytes
                );
                Ok(())
            },
        )
        .with_context(|| format!("preloading DFlash2 resident {}", group.resident_name))?;
        device_allocation_ms += timing.device_allocation_ms;
        staging_allocation_ms += timing.staging_allocation_ms;
        staging_fill_ms += timing.staging_fill_ms;
        h2d_ms += timing.h2d_ms;
        preloaded_resident_weight_device_buffer(&group.resident_name, expected_bytes)
            .with_context(|| format!("verifying DFlash2 resident {}", group.resident_name))?;
        if timing.uploaded {
            stats.loaded_resident_buffers += 1;
            stats.loaded_source_tensors += loaded.len();
        }
        for summary in loaded {
            stats.loaded_bytes = stats
                .loaded_bytes
                .checked_add(summary.bytes_read)
                .context("DFlash2 loaded byte count overflow")?;
            stats.source_read_micros = stats
                .source_read_micros
                .checked_add(summary.elapsed_micros)
                .context("DFlash2 source read time overflow")?;
        }
    }
    stats.total_elapsed_micros = started.elapsed().as_micros();
    let total_ms = stats.total_elapsed_micros as f64 / 1_000.0;
    let source_read_ms = stats.source_read_micros as f64 / 1_000.0;
    let unattributed_ms =
        (total_ms - device_allocation_ms - staging_allocation_ms - staging_fill_ms - h2d_ms)
            .max(0.0);
    eprintln!(
        "real_full_dflash2_preload_detail groups={} sources={} bytes={} total_ms={total_ms:.3} source_read_ms={source_read_ms:.3} device_allocation_ms={device_allocation_ms:.3} staging_allocation_ms={staging_allocation_ms:.3} staging_fill_ms={staging_fill_ms:.3} h2d_ms={h2d_ms:.3} unattributed_ms={unattributed_ms:.3}",
        stats.resident_buffers, stats.source_tensors, stats.loaded_bytes,
    );
    Ok(stats)
}

pub(super) fn preload_dflash2_target_aliases(
    target_catalog: &TensorCatalog,
) -> Result<Dflash2TargetAliasPreloadStats> {
    validate_dflash2_target_catalog(target_catalog)?;
    let started = Instant::now();
    let names = [GLM53_TARGET_EMBEDDING_WEIGHT, GLM53_TARGET_LM_HEAD_WEIGHT];
    let mut stats = Dflash2TargetAliasPreloadStats {
        selected_tensors: names.len(),
        ..Dflash2TargetAliasPreloadStats::default()
    };
    for name in names {
        let tensor = target_catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("GLM-5.3 DFlash2 target tensor {name} is missing"))?;
        let expected_bytes = tensor
            .shape
            .iter()
            .try_fold(1_usize, |values, dimension| values.checked_mul(*dimension))
            .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
            .context("GLM-5.3 DFlash2 target alias byte count overflow")?;
        stats.selected_bytes = stats
            .selected_bytes
            .checked_add(expected_bytes as u64)
            .context("GLM-5.3 DFlash2 target alias selected-byte count overflow")?;
        let mut loaded = None::<LoadedTensorSummary>;
        preload_resident_weight_from_host_staging_profiled(
            name,
            expected_bytes,
            "startup resident GLM-5.3 DFlash2 target alias pinned staging",
            |staging| {
                let summary = read_tensor_bytes_into(target_catalog, name, staging)
                    .with_context(|| format!("reading GLM-5.3 DFlash2 target alias {name}"))?;
                anyhow::ensure!(
                    summary.tensor_name == name
                        && summary.dtype == DType::Bf16
                        && summary.shape == [GLM53_DFLASH2_VOCAB_SIZE, GLM53_DFLASH2_HIDDEN_SIZE]
                        && summary.bytes_read == expected_bytes as u64,
                    "GLM-5.3 DFlash2 target alias {name} disagrees with its validated catalog"
                );
                loaded = Some(summary);
                Ok(())
            },
        )
        .with_context(|| format!("preloading GLM-5.3 DFlash2 target alias {name}"))?;
        preloaded_resident_weight_device_buffer(name, expected_bytes)
            .with_context(|| format!("binding GLM-5.3 DFlash2 target alias {name}"))?;
        if let Some(summary) = loaded {
            stats.loaded_tensors += 1;
            stats.loaded_bytes = stats
                .loaded_bytes
                .checked_add(summary.bytes_read)
                .context("GLM-5.3 DFlash2 target alias loaded-byte count overflow")?;
            stats.source_read_micros = stats
                .source_read_micros
                .checked_add(summary.elapsed_micros)
                .context("GLM-5.3 DFlash2 target alias read-time overflow")?;
        }
    }
    stats.total_elapsed_micros = started.elapsed().as_micros();
    Ok(stats)
}

pub(super) fn preloaded_dflash2_resident_weights(
    checkpoint: &Dflash2Checkpoint,
    target_catalog: &TensorCatalog,
) -> Result<Dflash2ResidentWeights> {
    validate_dflash2_target_catalog(target_catalog)?;
    let groups = dflash2_resident_groups(checkpoint)?;
    let revision = checkpoint.validated.fixture.revision;
    let resident = |suffix: &str| -> Result<GlmrtDeviceBuffer> {
        let name = format!("dflash2:{revision}:{suffix}");
        let group = groups
            .iter()
            .find(|group| group.resident_name == name)
            .with_context(|| format!("missing DFlash2 resident plan {name}"))?;
        let expected_bytes: usize = group
            .byte_length
            .try_into()
            .with_context(|| format!("DFlash2 resident {name} is too large"))?;
        preloaded_resident_weight_device_buffer(&name, expected_bytes)
            .with_context(|| format!("binding preloaded DFlash2 resident {name}"))
    };
    let target_bytes = GLM53_DFLASH2_VOCAB_SIZE
        .checked_mul(GLM53_DFLASH2_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(std::mem::size_of::<u16>()))
        .context("GLM-5.3 target embedding/LM-head byte count overflow")?;
    let target_embedding =
        preloaded_resident_weight_device_buffer(GLM53_TARGET_EMBEDDING_WEIGHT, target_bytes)
            .context("binding the GLM-5.3 target embedding for DFlash2")?;
    let target_lm_head =
        preloaded_resident_weight_device_buffer(GLM53_TARGET_LM_HEAD_WEIGHT, target_bytes)
            .context("binding the GLM-5.3 target LM head for DFlash2")?;
    let layers = (0..GLM53_DFLASH2_DRAFT_LAYERS)
        .map(|layer| {
            Ok(Dflash2LayerResidentWeights {
                attention_conv_base: resident(&format!(
                    "layers.{layer}.attention_conv.base_kernel"
                ))?,
                attention_conv_projection: resident(&format!(
                    "layers.{layer}.attention_conv.kernel_projection.weight"
                ))?,
                mlp_conv_base: resident(&format!("layers.{layer}.mlp_conv.base_kernel"))?,
                mlp_conv_projection: resident(&format!(
                    "layers.{layer}.mlp_conv.kernel_projection.weight"
                ))?,
                input_norm: resident(&format!("layers.{layer}.input_layernorm.weight"))?,
                post_norm: resident(&format!("layers.{layer}.post_attention_layernorm.weight"))?,
                q_norm: resident(&format!("layers.{layer}.self_attn.q_norm.weight"))?,
                k_norm: resident(&format!("layers.{layer}.self_attn.k_norm.weight"))?,
                qkv: resident(&format!("layers.{layer}.self_attn.qkv_proj.weight"))?,
                output: resident(&format!("layers.{layer}.self_attn.o_proj.weight"))?,
                gate_up: resident(&format!("layers.{layer}.mlp.gate_up_proj.weight"))?,
                down: resident(&format!("layers.{layer}.mlp.down_proj.weight"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?
        .try_into()
        .map_err(|_| anyhow::anyhow!("DFlash2 resident layer count changed"))?;
    Ok(Dflash2ResidentWeights {
        target_embedding,
        target_lm_head,
        target_fusion: resident("fc.weight")?,
        hidden_norm: resident("hidden_norm.weight")?,
        final_norm: resident("norm.weight")?,
        selector_hidden_projection: resident("candidate_selector.hidden_projection.weight")?,
        selector_predecessor: resident("candidate_selector.predecessor_codebook")?,
        selector_successor: resident("candidate_selector.successor_codebook")?,
        layers,
        draft_resident_bytes: checkpoint.weights.payload_bytes,
    })
}

fn validate_dflash2_loaded_tensor(
    binding: &Dflash2ResidentWeightPlan,
    summary: &LoadedTensorSummary,
) -> Result<()> {
    anyhow::ensure!(
        summary.tensor_name == binding.name
            && summary.dtype == binding.dtype
            && summary.shape == binding.shape
            && summary.byte_offset == binding.byte_offset
            && summary.bytes_read == binding.byte_length,
        "DFlash2 loaded tensor {} disagrees with its validated manifest",
        binding.name
    );
    Ok(())
}

fn validate_dflash2_target_catalog(catalog: &TensorCatalog) -> Result<()> {
    for name in [GLM53_TARGET_EMBEDDING_WEIGHT, GLM53_TARGET_LM_HEAD_WEIGHT] {
        let tensor = catalog
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .with_context(|| format!("GLM-5.3 DFlash2 target tensor {name} is missing"))?;
        anyhow::ensure!(
            tensor.dtype == DType::Bf16
                && tensor.shape == [GLM53_DFLASH2_VOCAB_SIZE, GLM53_DFLASH2_HIDDEN_SIZE],
            "GLM-5.3 DFlash2 target tensor {name} mismatch: dtype={:?} shape={:?}",
            tensor.dtype,
            tensor.shape
        );
    }
    Ok(())
}

fn checked_bf16_tensor_bytes(shape: &[usize]) -> Result<u64> {
    shape
        .iter()
        .try_fold(1_u64, |values, dimension| {
            values.checked_mul(*dimension as u64)
        })
        .and_then(|values| values.checked_mul(2))
        .context("BF16 tensor size overflow")
}

fn expected_dflash2_tensor_shapes() -> BTreeMap<String, Vec<usize>> {
    let mut tensors = BTreeMap::new();
    tensors.insert(
        "candidate_selector.hidden_projection.weight".to_owned(),
        vec![GLM53_DFLASH2_SELECTOR_RANK, GLM53_DFLASH2_HIDDEN_SIZE],
    );
    tensors.insert(
        "candidate_selector.predecessor_codebook".to_owned(),
        vec![GLM53_DFLASH2_VOCAB_SIZE, GLM53_DFLASH2_SELECTOR_RANK],
    );
    tensors.insert(
        "candidate_selector.successor_codebook".to_owned(),
        vec![GLM53_DFLASH2_VOCAB_SIZE, GLM53_DFLASH2_SELECTOR_RANK],
    );
    tensors.insert(
        "fc.weight".to_owned(),
        vec![
            GLM53_DFLASH2_HIDDEN_SIZE,
            GLM53_DFLASH2_DRAFT_LAYERS * GLM53_DFLASH2_HIDDEN_SIZE,
        ],
    );
    tensors.insert(
        "hidden_norm.weight".to_owned(),
        vec![GLM53_DFLASH2_HIDDEN_SIZE],
    );
    tensors.insert("norm.weight".to_owned(), vec![GLM53_DFLASH2_HIDDEN_SIZE]);

    let q_width = GLM53_DFLASH2_ATTENTION_HEADS * GLM53_DFLASH2_HEAD_DIM;
    let kv_width = GLM53_DFLASH2_KV_HEADS * GLM53_DFLASH2_HEAD_DIM;
    let conv_groups = GLM53_DFLASH2_HIDDEN_SIZE / GLM53_DFLASH2_CONV_GROUP_SIZE;
    let conv_projection = 2 * GLM53_DFLASH2_CONV_KERNEL_SIZE * conv_groups;
    for layer_id in 0..GLM53_DFLASH2_DRAFT_LAYERS {
        let prefix = format!("layers.{layer_id}");
        for conv in ["attention_conv", "mlp_conv"] {
            tensors.insert(
                format!("{prefix}.{conv}.base_kernel"),
                vec![2, GLM53_DFLASH2_CONV_KERNEL_SIZE, GLM53_DFLASH2_HIDDEN_SIZE],
            );
            tensors.insert(
                format!("{prefix}.{conv}.kernel_projection.weight"),
                vec![conv_projection, GLM53_DFLASH2_HIDDEN_SIZE],
            );
        }
        tensors.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.mlp.gate_proj.weight"),
            vec![GLM53_DFLASH2_INTERMEDIATE_SIZE, GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.mlp.up_proj.weight"),
            vec![GLM53_DFLASH2_INTERMEDIATE_SIZE, GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.mlp.down_proj.weight"),
            vec![GLM53_DFLASH2_HIDDEN_SIZE, GLM53_DFLASH2_INTERMEDIATE_SIZE],
        );
        tensors.insert(
            format!("{prefix}.self_attn.q_proj.weight"),
            vec![q_width, GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_proj.weight"),
            vec![kv_width, GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.self_attn.v_proj.weight"),
            vec![kv_width, GLM53_DFLASH2_HIDDEN_SIZE],
        );
        tensors.insert(
            format!("{prefix}.self_attn.o_proj.weight"),
            vec![GLM53_DFLASH2_HIDDEN_SIZE, q_width],
        );
        tensors.insert(
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![GLM53_DFLASH2_HEAD_DIM],
        );
        tensors.insert(
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![GLM53_DFLASH2_HEAD_DIM],
        );
    }
    tensors
}

pub(super) struct Dflash2RequestEngine {
    // Drop the aliasing suffix graphs before the C=1 executor that owns the
    // shared KV allocation.
    batched_suffix_executors: BTreeMap<usize, Dflash2StaticExecutor>,
    executor: Dflash2StaticExecutor,
    checkpoint_revision: &'static str,
    max_cache_context_tokens: usize,
    page_size: usize,
    pages_per_request: usize,
    proposal_tokens_per_request: usize,
    free_slots: Vec<usize>,
    active_slot: Option<usize>,
}

pub(super) struct Dflash2RequestState {
    slot: usize,
    context_tokens: usize,
    cache_context_tokens: usize,
    page_table: Vec<i32>,
    page_table_dirty: bool,
}

pub(super) struct Dflash2BatchedReplayRequest<'a> {
    pub(super) state: &'a mut Dflash2RequestState,
    pub(super) target_hidden_taps: [&'a DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS],
    pub(super) target_row_start: usize,
    pub(super) committed_rows: usize,
    pub(super) absolute_context_start: Option<usize>,
    pub(super) anchor_token: usize,
}

impl Dflash2RequestState {
    pub(super) fn context_tokens(&self) -> usize {
        self.context_tokens
    }
}

pub(super) struct Dflash2RequestCacheSnapshot {
    pub(super) context_tokens: usize,
    pub(super) cache_context_tokens: usize,
    pub(super) kv_bytes: Vec<u8>,
}

impl Dflash2RequestCacheSnapshot {
    pub(super) fn resident_bytes(&self) -> usize {
        self.kv_bytes.len()
    }
}

fn dflash2_request_slot_page_table(slot: usize, pages_per_request: usize) -> Result<Vec<i32>> {
    let page_base = slot
        .checked_mul(pages_per_request)
        .context("DFlash2 request physical page base overflow")?;
    (0..pages_per_request)
        .map(|page| {
            page_base
                .checked_add(page)
                .context("DFlash2 request physical page ID overflow")?
                .try_into()
                .context("DFlash2 request physical page ID does not fit i32")
        })
        .collect()
}

fn roll_dflash2_swa_page_table(
    page_table: &mut [i32],
    cache_context_tokens: usize,
    incoming_tokens: usize,
    max_cache_context_tokens: usize,
    page_size: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(!page_table.is_empty(), "DFlash2 SWA page table is empty");
    anyhow::ensure!(page_size > 0, "DFlash2 SWA page size is zero");
    anyhow::ensure!(
        max_cache_context_tokens > 0 && max_cache_context_tokens % page_size == 0,
        "DFlash2 SWA window must be a positive page multiple"
    );
    anyhow::ensure!(
        page_table.len() * page_size >= max_cache_context_tokens + page_size,
        "DFlash2 SWA page table has no page-granular retention slop"
    );
    let retention_ceiling = max_cache_context_tokens
        .checked_add(page_size - 1)
        .context("DFlash2 SWA retention ceiling overflow")?;
    let projected = cache_context_tokens
        .checked_add(incoming_tokens)
        .context("DFlash2 SWA cache length overflow")?;
    if projected <= retention_ceiling {
        return Ok((cache_context_tokens, 0));
    }
    let pages_to_drop = projected
        .checked_sub(retention_ceiling)
        .expect("projected DFlash2 cache exceeds retention ceiling")
        .div_ceil(page_size);
    let tokens_to_drop = pages_to_drop
        .checked_mul(page_size)
        .context("DFlash2 SWA dropped-token count overflow")?;
    anyhow::ensure!(
        tokens_to_drop <= cache_context_tokens && pages_to_drop < page_table.len(),
        "DFlash2 SWA cannot drop {tokens_to_drop} tokens from a {cache_context_tokens}-token cache"
    );
    page_table.rotate_left(pages_to_drop);
    Ok((cache_context_tokens - tokens_to_drop, pages_to_drop))
}

// The live runtime serializes this engine behind one mutex. Its CUDA objects
// stay on the device primary context and are never replayed concurrently.
unsafe impl Send for Dflash2RequestEngine {}

impl Dflash2RequestEngine {
    pub(super) fn checkpoint_model_id(&self) -> &'static str {
        GLM53_DFLASH2_REPO_ID
    }

    pub(super) fn checkpoint_revision(&self) -> &'static str {
        self.checkpoint_revision
    }

    pub(super) fn max_verify_drafts(&self) -> usize {
        self.proposal_tokens_per_request
    }

    pub(super) fn target_layer_ids(&self) -> &'static [usize; GLM53_DFLASH2_DRAFT_LAYERS] {
        &GLM53_DFLASH2_TARGET_CAPTURE_TAPS
    }

    pub(super) fn load(
        snapshot: &Path,
        target_catalog: &TensorCatalog,
        target_kv_capacity_tokens: usize,
        max_active_requests: usize,
        proposal_tokens_per_request: usize,
    ) -> Result<Self> {
        let load_started = Instant::now();
        anyhow::ensure!(
            max_active_requests > 0,
            "DFlash2 request executor requires a physical cache slot"
        );
        anyhow::ensure!(
            (1..=GLM53_DFLASH2_MAX_DRAFTS).contains(&proposal_tokens_per_request),
            "DFlash2 internal proposal width must be in 1..={GLM53_DFLASH2_MAX_DRAFTS}"
        );
        let query_rows_per_request = proposal_tokens_per_request + 1;
        let page_size = 64;
        let max_cache_context_tokens = GLM53_DFLASH2_SLIDING_WINDOW;
        let draft_kv_capacity_tokens =
            (max_cache_context_tokens + page_size + query_rows_per_request).div_ceil(page_size)
                * page_size;
        anyhow::ensure!(
            target_kv_capacity_tokens >= draft_kv_capacity_tokens,
            "target KV capacity {target_kv_capacity_tokens} is smaller than the DFlash2 request capacity {draft_kv_capacity_tokens}"
        );
        let checkpoint_started = Instant::now();
        let checkpoint = Dflash2Checkpoint::from_snapshot(snapshot)?;
        let checkpoint_ms = checkpoint_started.elapsed().as_secs_f64() * 1_000.0;
        let draft_preload_started = Instant::now();
        let draft_preload = preload_dflash2_resident_weights(&checkpoint)?;
        let draft_preload_ms = draft_preload_started.elapsed().as_secs_f64() * 1_000.0;
        let alias_preload_started = Instant::now();
        let target_aliases = preload_dflash2_target_aliases(target_catalog)?;
        let alias_preload_ms = alias_preload_started.elapsed().as_secs_f64() * 1_000.0;
        let weights = preloaded_dflash2_resident_weights(&checkpoint, target_catalog)?;
        let update_weights = dflash2_update_resident_weights(weights)?;
        let pages_per_request = draft_kv_capacity_tokens.div_ceil(page_size);
        let physical_kv_pages = pages_per_request
            .checked_mul(max_active_requests)
            .context("DFlash2 shared request KV page count overflow")?;
        let kv_bytes_per_token = GLM53_DFLASH2_DRAFT_LAYERS
            .checked_mul(GLM53_DFLASH2_KV_HEADS)
            .and_then(|value| value.checked_mul(GLM53_DFLASH2_HEAD_DIM))
            .and_then(|value| value.checked_mul(2))
            .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
            .context("DFlash2 shared request KV bytes/token overflow")?;
        let physical_kv_bytes = physical_kv_pages
            .checked_mul(page_size)
            .and_then(|tokens| tokens.checked_mul(kv_bytes_per_token))
            .context("DFlash2 shared request KV byte count overflow")?;
        eprintln!(
            "real_full_dflash2_kv_pool request_slots={} pages_per_request={} physical_pages={} page_tokens={} bytes_per_token={} physical_bytes={}",
            max_active_requests,
            pages_per_request,
            physical_kv_pages,
            page_size,
            kv_bytes_per_token,
            physical_kv_bytes,
        );
        let executor_started = Instant::now();
        let mut executor = Dflash2StaticExecutor::capture_with_physical_pages(
            weights,
            Dflash2StaticBenchConfig {
                active_requests: 1,
                accepted_rows_per_request: 1,
                proposal_tokens_per_request,
                context_tokens: 0,
                kv_capacity_tokens: draft_kv_capacity_tokens,
                allocate_full_kv_capacity: true,
                capture_page_buckets: true,
                page_size,
                kv_storage: DsparkKvStorage::Bf16,
                warmup: 0,
                iterations: 1,
                repeats: 1,
                seed: 20_260_829,
            },
            Some(physical_kv_pages),
        )
        .context("capturing the live C=1 DFlash2 request executor")?;
        let executor_ms = executor_started.elapsed().as_secs_f64() * 1_000.0;
        let batched_started = Instant::now();
        executor
            .capture_batched_update_graphs(update_weights, dflash2_update_graph_buckets(1)?)
            .context("capturing batched DFlash2 target-context updates")?;
        let shared_kv = executor.shared_kv_pool();
        let mut batched_suffix_executors = BTreeMap::new();
        for active_requests in [2, 4]
            .into_iter()
            .filter(|active_requests| *active_requests <= max_active_requests)
        {
            let mut suffix_executor = Dflash2StaticExecutor::capture_with_shared_kv_pool(
                weights,
                Dflash2StaticBenchConfig {
                    active_requests,
                    accepted_rows_per_request: 1,
                    proposal_tokens_per_request,
                    context_tokens: 0,
                    kv_capacity_tokens: draft_kv_capacity_tokens,
                    allocate_full_kv_capacity: true,
                    capture_page_buckets: false,
                    page_size,
                    kv_storage: DsparkKvStorage::Bf16,
                    warmup: 0,
                    iterations: 1,
                    repeats: 1,
                    seed: 20_260_829 + active_requests as i64,
                },
                shared_kv,
            )
            .with_context(|| {
                format!("capturing the live C={active_requests} DFlash2 suffix executor")
            })?;
            let packed_update_rows = dflash2_update_graph_buckets(active_requests)?;
            suffix_executor
                .capture_batched_update_graphs(update_weights, packed_update_rows)
                .with_context(|| {
                    format!("capturing live C={active_requests} packed DFlash2 update graphs")
                })?;
            batched_suffix_executors.insert(active_requests, suffix_executor);
        }
        let batched_ms = batched_started.elapsed().as_secs_f64() * 1_000.0;
        eprintln!(
            "real_full_dflash2_engine_load revision={} proposal_tokens={} query_rows={} c1_page_graphs={} draft_buffers={} draft_bytes={} loaded_bytes={} target_aliases={} target_alias_bytes={} checkpoint_ms={checkpoint_ms:.3} draft_preload_ms={draft_preload_ms:.3} alias_preload_ms={alias_preload_ms:.3} executor_ms={executor_ms:.3} batched_capture_ms={batched_ms:.3} total_ms={:.3}",
            checkpoint.validated.fixture.revision,
            proposal_tokens_per_request,
            query_rows_per_request,
            executor.suffix_page_graph_count(),
            draft_preload.resident_buffers,
            draft_preload.selected_bytes,
            draft_preload.loaded_bytes,
            target_aliases.selected_tensors,
            target_aliases.selected_bytes,
            load_started.elapsed().as_secs_f64() * 1_000.0,
        );
        Ok(Self {
            executor,
            batched_suffix_executors,
            checkpoint_revision: checkpoint.validated.fixture.revision,
            max_cache_context_tokens,
            page_size,
            pages_per_request,
            proposal_tokens_per_request,
            free_slots: (0..max_active_requests).rev().collect(),
            active_slot: None,
        })
    }

    pub(super) fn allocate_request_state(&mut self) -> Result<Dflash2RequestState> {
        let slot = self
            .free_slots
            .pop()
            .context("DFlash2 shared request KV slots are exhausted")?;
        Ok(Dflash2RequestState {
            slot,
            context_tokens: 0,
            cache_context_tokens: 0,
            page_table: dflash2_request_slot_page_table(slot, self.pages_per_request)?,
            page_table_dirty: true,
        })
    }

    pub(super) fn reset_request_state(&mut self, state: &mut Dflash2RequestState) -> Result<()> {
        state.context_tokens = 0;
        state.cache_context_tokens = 0;
        state.page_table = dflash2_request_slot_page_table(state.slot, self.pages_per_request)?;
        state.page_table_dirty = true;
        Ok(())
    }

    pub(super) fn release_request_state(&mut self, state: Dflash2RequestState) {
        debug_assert!(!self.free_slots.contains(&state.slot));
        self.free_slots.push(state.slot);
    }

    pub(super) fn snapshot_request_state(
        &self,
        state: &Dflash2RequestState,
    ) -> Result<Option<Dflash2RequestCacheSnapshot>> {
        if state.cache_context_tokens == 0 {
            return Ok(None);
        }
        let kv_bytes = self
            .executor
            .read_request_cache_snapshot(&state.page_table, state.cache_context_tokens)
            .context("saving the committed DFlash2 request-cache tail")?;
        Ok(Some(Dflash2RequestCacheSnapshot {
            context_tokens: state.context_tokens,
            cache_context_tokens: state.cache_context_tokens,
            kv_bytes,
        }))
    }

    pub(super) fn snapshot_request_state_at_prefix(
        &self,
        state: &Dflash2RequestState,
        prefix_tokens: usize,
    ) -> Result<Option<Dflash2RequestCacheSnapshot>> {
        anyhow::ensure!(
            prefix_tokens <= state.context_tokens,
            "DFlash2 reusable prefix {prefix_tokens} exceeds context {}",
            state.context_tokens
        );
        let rollback_tokens = state.context_tokens - prefix_tokens;
        anyhow::ensure!(
            rollback_tokens <= state.cache_context_tokens,
            "DFlash2 prefix rollback {rollback_tokens} exceeds cache context {}",
            state.cache_context_tokens
        );
        let cache_context_tokens = state.cache_context_tokens - rollback_tokens;
        if cache_context_tokens == 0 {
            return Ok(None);
        }
        let kv_bytes = self
            .executor
            .read_request_cache_snapshot(&state.page_table, cache_context_tokens)
            .context("saving the reusable DFlash2 request-cache prefix")?;
        Ok(Some(Dflash2RequestCacheSnapshot {
            context_tokens: prefix_tokens,
            cache_context_tokens,
            kv_bytes,
        }))
    }

    pub(super) fn restore_request_state(
        &mut self,
        state: &mut Dflash2RequestState,
        snapshot: &Dflash2RequestCacheSnapshot,
    ) -> Result<()> {
        anyhow::ensure!(
            snapshot.cache_context_tokens <= self.max_cache_context_tokens + self.page_size - 1,
            "DFlash2 snapshot retains {} tokens beyond the {}+{} window/slop limit",
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
            .context("restoring the committed DFlash2 request-cache tail")?;
        state.context_tokens = snapshot.context_tokens;
        state.cache_context_tokens = snapshot.cache_context_tokens;
        state.page_table_dirty = true;
        Ok(())
    }

    pub(super) fn replay_step(
        &mut self,
        state: &mut Dflash2RequestState,
        target_hidden_taps: [&DeviceBf16Output; GLM53_DFLASH2_DRAFT_LAYERS],
        target_row_start: usize,
        committed_rows: usize,
        absolute_context_start: Option<usize>,
        anchor_token: usize,
    ) -> Result<Dflash2DraftStep> {
        if let Some(absolute_context_start) = absolute_context_start {
            if state.cache_context_tokens == 0 {
                state.context_tokens = absolute_context_start;
            } else {
                anyhow::ensure!(
                    state.context_tokens == absolute_context_start,
                    "restored DFlash2 tail ends at {} but the uncached target suffix starts at {absolute_context_start}",
                    state.context_tokens
                );
            }
        }
        let (cache_context_tokens, dropped_pages) = roll_dflash2_swa_page_table(
            &mut state.page_table,
            state.cache_context_tokens,
            committed_rows,
            self.max_cache_context_tokens,
            self.page_size,
        )?;
        if dropped_pages > 0 || state.page_table_dirty || self.active_slot != Some(state.slot) {
            self.executor
                .set_request_page_table(&state.page_table)
                .context("uploading the DFlash2 request SWA page rotation")?;
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
        state.context_tokens = state
            .context_tokens
            .checked_add(committed_rows)
            .context("DFlash2 absolute request context overflow")?;
        state.cache_context_tokens = cache_context_tokens
            .checked_add(committed_rows)
            .context("DFlash2 request cache context overflow")?;
        Ok(step)
    }

    pub(super) fn replay_batched_steps(
        &mut self,
        requests: &mut [Dflash2BatchedReplayRequest<'_>],
    ) -> Result<Vec<Dflash2DraftStep>> {
        let batch_started = Instant::now();
        anyhow::ensure!(
            matches!(requests.len(), 2 | 4),
            "DFlash2 batched replay requires exactly 2 or 4 requests"
        );
        let request_count = requests.len();
        anyhow::ensure!(
            self.batched_suffix_executors.contains_key(&requests.len()),
            "DFlash2 C={} suffix executor was not captured",
            requests.len()
        );
        for request in requests.iter_mut() {
            let state = &mut *request.state;
            if let Some(absolute_context_start) = request.absolute_context_start {
                if state.cache_context_tokens == 0 {
                    state.context_tokens = absolute_context_start;
                } else {
                    anyhow::ensure!(
                        state.context_tokens == absolute_context_start,
                        "restored DFlash2 tail ends at {} but the uncached target suffix starts at {absolute_context_start}",
                        state.context_tokens
                    );
                }
            }
        }
        let total_committed_rows = requests.iter().try_fold(0_usize, |rows, request| {
            rows.checked_add(request.committed_rows)
                .context("DFlash2 collective committed-row count overflow")
        })?;
        let packed_update = self
            .batched_suffix_executors
            .get(&requests.len())
            .is_some_and(|executor| executor.supports_batched_update_rows(total_committed_rows));
        let packed_update_rows = packed_update
            .then(|| total_committed_rows.max(request_count).next_power_of_two())
            .unwrap_or(0);
        let mut update_records = Vec::with_capacity(requests.len());
        if packed_update {
            let mut update_starts = Vec::with_capacity(requests.len());
            for request in requests.iter_mut() {
                let state = &mut *request.state;
                let context_tokens = state.context_tokens;
                let (cache_context_tokens, _dropped_pages) = roll_dflash2_swa_page_table(
                    &mut state.page_table,
                    state.cache_context_tokens,
                    request.committed_rows,
                    self.max_cache_context_tokens,
                    self.page_size,
                )?;
                update_starts.push((context_tokens, cache_context_tokens));
            }
            let packed_requests = requests
                .iter()
                .zip(&update_starts)
                .map(|(request, (context_tokens, cache_context_tokens))| {
                    Dflash2BatchedUpdateRequest {
                        target_hidden_taps: request.target_hidden_taps,
                        target_row_start: request.target_row_start,
                        committed_rows: request.committed_rows,
                        absolute_context_start: *context_tokens,
                        cache_context_start: *cache_context_tokens,
                        page_table: &request.state.page_table,
                    }
                })
                .collect::<Vec<_>>();
            let updates = self
                .batched_suffix_executors
                .get_mut(&requests.len())
                .expect("the DFlash2 packed update executor was checked above")
                .update_batched_request_caches(&packed_requests)?;
            drop(packed_requests);
            anyhow::ensure!(
                updates.len() == requests.len(),
                "DFlash2 packed update returned the wrong request count"
            );
            for ((request, (context_tokens, _)), update) in
                requests.iter_mut().zip(update_starts).zip(updates)
            {
                request.state.context_tokens = update.absolute_context_after_update;
                request.state.cache_context_tokens = update.cache_context_after_update;
                // The packed executor owns a different block-table buffer.
                // Force any future scalar C=1 replay for this request to upload
                // its current rotated table before touching the shared KV pool.
                request.state.page_table_dirty = true;
                update_records.push((
                    context_tokens,
                    request.committed_rows,
                    update.update_ms / request_count as f64,
                ));
            }
            self.active_slot = None;
        } else {
            for request in requests.iter_mut() {
                let state = &mut *request.state;
                let context_tokens = state.context_tokens;
                let (cache_context_tokens, dropped_pages) = roll_dflash2_swa_page_table(
                    &mut state.page_table,
                    state.cache_context_tokens,
                    request.committed_rows,
                    self.max_cache_context_tokens,
                    self.page_size,
                )?;
                if dropped_pages > 0
                    || state.page_table_dirty
                    || self.active_slot != Some(state.slot)
                {
                    self.executor
                        .set_request_page_table(&state.page_table)
                        .context("uploading a batched DFlash2 request SWA page rotation")?;
                    state.page_table_dirty = false;
                    self.active_slot = Some(state.slot);
                }
                let update = self.executor.update_request_cache_with_cache_context(
                    request.target_hidden_taps,
                    request.target_row_start,
                    request.committed_rows,
                    context_tokens,
                    cache_context_tokens,
                )?;
                state.context_tokens = update.context_after_update;
                state.cache_context_tokens = update.cache_context_after_update;
                update_records.push((context_tokens, request.committed_rows, update.update_ms));
            }
        }

        let suffix_requests = requests
            .iter()
            .map(|request| Dflash2BatchedSuffixRequest {
                page_table: &request.state.page_table,
                absolute_context_after_update: request.state.context_tokens,
                cache_context_after_update: request.state.cache_context_tokens,
                anchor_token: request.anchor_token,
            })
            .collect::<Vec<_>>();
        let suffix = self
            .batched_suffix_executors
            .get_mut(&requests.len())
            .expect("the DFlash2 batch executor was checked above")
            .replay_batched_suffix(&suffix_requests)?;
        anyhow::ensure!(
            suffix.proposal_token_ids.len() == requests.len(),
            "DFlash2 batched suffix returned the wrong request count"
        );
        let total_ms = batch_started.elapsed().as_secs_f64() * 1_000.0;
        Ok(requests
            .iter()
            .zip(update_records)
            .zip(suffix.proposal_token_ids)
            .map(
                |((request, (context_tokens, committed_rows, update_ms)), proposals)| {
                    Dflash2DraftStep {
                        context_tokens,
                        committed_rows,
                        anchor_token: request.anchor_token,
                        proposal_token_ids: proposals,
                        update_ms,
                        suffix_ms: suffix.suffix_ms,
                        readback_ms: suffix.readback_ms,
                        total_ms,
                        packed_update_rows,
                    }
                },
            )
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_config_layer_ids_to_post_layer_capture_boundaries() {
        assert_eq!(
            GLM53_DFLASH2_TARGET_CAPTURE_TAPS,
            GLM53_DFLASH2_TARGET_TAPS.map(|layer_id| layer_id + 1)
        );
    }

    fn fixture_config() -> String {
        serde_json::json!({
            "architectures": [GLM53_DFLASH2_ARCHITECTURE],
            "attention_bias": false,
            "dflash_config": {
                "block_size": GLM53_DFLASH2_BLOCK_SIZE,
                "conv_group_size": GLM53_DFLASH2_CONV_GROUP_SIZE,
                "conv_kernel_size": GLM53_DFLASH2_CONV_KERNEL_SIZE,
                "mask_token_id": GLM53_DFLASH2_MASK_TOKEN_ID,
                "selector_rank": GLM53_DFLASH2_SELECTOR_RANK,
                "selector_top_k": GLM53_DFLASH2_SELECTOR_TOP_K,
                "target_layer_ids": GLM53_DFLASH2_TARGET_TAPS,
            },
            "dtype": "bfloat16",
            "head_dim": GLM53_DFLASH2_HEAD_DIM,
            "hidden_act": "silu",
            "hidden_size": GLM53_DFLASH2_HIDDEN_SIZE,
            "intermediate_size": GLM53_DFLASH2_INTERMEDIATE_SIZE,
            "is_causal": false,
            "layer_types": vec!["sliding_attention"; GLM53_DFLASH2_DRAFT_LAYERS],
            "max_position_embeddings": 1_048_576,
            "max_window_layers": GLM53_DFLASH2_DRAFT_LAYERS,
            "num_attention_heads": GLM53_DFLASH2_ATTENTION_HEADS,
            "num_hidden_layers": GLM53_DFLASH2_DRAFT_LAYERS,
            "num_key_value_heads": GLM53_DFLASH2_KV_HEADS,
            "rms_norm_eps": 1.0e-5,
            "rope_parameters": {
                "rope_theta": 1_000_000.0,
                "rope_type": "default",
            },
            "sliding_window": GLM53_DFLASH2_SLIDING_WINDOW,
            "tie_word_embeddings": false,
            "use_cache": false,
            "use_sliding_window": true,
            "vocab_size": GLM53_DFLASH2_VOCAB_SIZE,
        })
        .to_string()
    }

    fn complete_weight_metadata() -> Vec<SafetensorsTensorMetadata> {
        let mut offset = 10_600_u64;
        expected_dflash2_tensor_shapes()
            .into_iter()
            .map(|(name, shape)| {
                let byte_length = checked_bf16_tensor_bytes(&shape).unwrap();
                let tensor = SafetensorsTensorMetadata {
                    name,
                    dtype: DType::Bf16,
                    shape,
                    byte_offset: offset,
                    byte_length,
                };
                offset += byte_length;
                tensor
            })
            .collect()
    }

    fn checkpoint() -> Dflash2Checkpoint {
        Dflash2Checkpoint {
            validated: ValidatedDflash2Checkpoint::from_config_json(
                GLM53_DFLASH2,
                &fixture_config(),
            )
            .unwrap(),
            weights: Dflash2WeightManifest::from_metadata(
                GLM53_DFLASH2,
                PathBuf::from("/tmp/glm53-dflash2/model.safetensors"),
                complete_weight_metadata(),
            )
            .unwrap(),
        }
    }

    fn target_catalog() -> TensorCatalog {
        let tensor = |name: &str, role: TensorRole| TensorInfo {
            name: name.to_owned(),
            file: "model-target.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![GLM53_DFLASH2_VOCAB_SIZE, GLM53_DFLASH2_HIDDEN_SIZE],
            byte_offset: 0,
            byte_length: 2,
            role,
            layer_id: None,
            expert_id: None,
            is_quantization_metadata: false,
        };
        TensorCatalog {
            model_id: "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1".to_owned(),
            snapshot_path: "/tmp/glm53-target".to_owned(),
            facts: ModelFacts::default(),
            tensors: vec![
                tensor(GLM53_TARGET_EMBEDDING_WEIGHT, TensorRole::Embedding),
                tensor(GLM53_TARGET_LM_HEAD_WEIGHT, TensorRole::LmHead),
            ],
        }
    }

    #[test]
    fn validates_the_pinned_glm53_dflash2_contract() {
        let validated =
            ValidatedDflash2Checkpoint::from_config_json(GLM53_DFLASH2, &fixture_config()).unwrap();
        assert_eq!(validated.target_layer_ids, GLM53_DFLASH2_TARGET_TAPS);
        assert_eq!(GLM53_DFLASH2_MAX_DRAFTS, 7);
    }

    #[test]
    fn requires_the_exact_standard_hf_lfs_blob_identity() {
        let temporary = tempfile::tempdir().unwrap();
        let blobs = temporary.path().join("blobs");
        let snapshot = temporary.path().join("snapshots/revision");
        std::fs::create_dir_all(&blobs).unwrap();
        std::fs::create_dir_all(&snapshot).unwrap();
        let blob = blobs.join(GLM53_DFLASH2_WEIGHT_LFS_SHA256);
        std::fs::write(&blob, b"fixture").unwrap();
        let weight = snapshot.join("model.safetensors");
        std::os::unix::fs::symlink(
            format!("../../blobs/{GLM53_DFLASH2_WEIGHT_LFS_SHA256}"),
            &weight,
        )
        .unwrap();
        validate_hf_lfs_blob_identity(&weight, GLM53_DFLASH2_WEIGHT_LFS_SHA256).unwrap();
        assert!(validate_hf_lfs_blob_identity(&weight, &"0".repeat(64)).is_err());

        std::fs::remove_file(&weight).unwrap();
        std::fs::write(&weight, b"fixture").unwrap();
        assert!(validate_hf_lfs_blob_identity(&weight, GLM53_DFLASH2_WEIGHT_LFS_SHA256).is_err());
    }

    #[test]
    fn validates_the_exact_dflash2_tensor_namespace_and_payload() {
        let manifest = Dflash2WeightManifest::from_metadata(
            GLM53_DFLASH2,
            PathBuf::from("/tmp/glm53-dflash2/model.safetensors"),
            complete_weight_metadata(),
        )
        .unwrap();
        assert_eq!(manifest.residency.len(), GLM53_DFLASH2_TENSOR_COUNT);
        assert_eq!(manifest.payload_bytes, GLM53_DFLASH2_WEIGHT_PAYLOAD_BYTES);
        assert_eq!(
            manifest
                .residency
                .iter()
                .find(|tensor| tensor.name == "fc.weight")
                .unwrap()
                .shape,
            [6_144, 36_864]
        );
        assert_eq!(
            manifest
                .residency
                .iter()
                .find(|tensor| tensor.name == "layers.5.self_attn.k_proj.weight")
                .unwrap()
                .shape,
            [1_024, 6_144]
        );
    }

    #[test]
    fn groups_qkv_and_gate_up_into_fixed_address_resident_buffers() {
        let checkpoint = checkpoint();
        let groups = dflash2_resident_groups(&checkpoint).unwrap();
        assert_eq!(groups.len(), 78);
        assert_eq!(
            groups
                .iter()
                .map(|group| group.source_names.len())
                .sum::<usize>(),
            GLM53_DFLASH2_TENSOR_COUNT
        );
        assert_eq!(
            groups.iter().map(|group| group.byte_length).sum::<u64>(),
            GLM53_DFLASH2_WEIGHT_PAYLOAD_BYTES
        );
        let qkv = groups
            .iter()
            .find(|group| {
                group
                    .resident_name
                    .ends_with("layers.3.self_attn.qkv_proj.weight")
            })
            .unwrap();
        assert_eq!(qkv.shape, [10_240, 6_144]);
        assert_eq!(
            qkv.source_names,
            [
                "layers.3.self_attn.q_proj.weight",
                "layers.3.self_attn.k_proj.weight",
                "layers.3.self_attn.v_proj.weight",
            ]
        );
        let gate_up = groups
            .iter()
            .find(|group| {
                group
                    .resident_name
                    .ends_with("layers.4.mlp.gate_up_proj.weight")
            })
            .unwrap();
        assert_eq!(gate_up.shape, [24_576, 6_144]);
        assert_eq!(gate_up.source_names.len(), 2);
    }

    #[test]
    fn validates_target_embedding_and_lm_head_without_duplicate_draft_copies() {
        let mut catalog = target_catalog();
        validate_dflash2_target_catalog(&catalog).unwrap();
        catalog.tensors[1].dtype = DType::F8E4M3;
        assert!(validate_dflash2_target_catalog(&catalog).is_err());
    }

    #[test]
    fn request_slots_own_disjoint_contiguous_page_ranges() {
        let pages_per_request = 34;
        assert_eq!(
            dflash2_request_slot_page_table(0, pages_per_request).unwrap(),
            (0..34).collect::<Vec<_>>()
        );
        assert_eq!(
            dflash2_request_slot_page_table(3, pages_per_request).unwrap(),
            (102..136).collect::<Vec<_>>()
        );
    }

    #[test]
    fn swa_rolls_one_page_at_the_first_retention_boundary() {
        let mut page_table = (0..34).collect::<Vec<_>>();
        let (retained_tokens, dropped_pages) = roll_dflash2_swa_page_table(
            &mut page_table,
            GLM53_DFLASH2_SLIDING_WINDOW + 63,
            1,
            GLM53_DFLASH2_SLIDING_WINDOW,
            64,
        )
        .unwrap();
        assert_eq!(retained_tokens, GLM53_DFLASH2_SLIDING_WINDOW - 1);
        assert_eq!(dropped_pages, 1);
        assert_eq!(retained_tokens + 1, GLM53_DFLASH2_SLIDING_WINDOW);
        assert_eq!(page_table[0], 1);
        assert_eq!(page_table[33], 0);
    }

    #[test]
    fn swa_retention_stays_bounded_across_long_prefill_and_decode_chunks() {
        let mut page_table = (0..34).collect::<Vec<_>>();
        let mut retained_tokens = 0;
        for incoming_tokens in std::iter::once(2_048)
            .chain(std::iter::repeat_n(1_024, 16))
            .chain(std::iter::repeat_n(GLM53_DFLASH2_BLOCK_SIZE, 1_024))
        {
            let (rolled_tokens, _) = roll_dflash2_swa_page_table(
                &mut page_table,
                retained_tokens,
                incoming_tokens,
                GLM53_DFLASH2_SLIDING_WINDOW,
                64,
            )
            .unwrap();
            retained_tokens = rolled_tokens + incoming_tokens;
            assert!(retained_tokens <= GLM53_DFLASH2_SLIDING_WINDOW + 63);
        }
    }

    #[test]
    fn rejects_dflash_and_flash_architecture_substitutions() {
        let mut config: serde_json::Value = serde_json::from_str(&fixture_config()).unwrap();
        config["architectures"] = serde_json::json!(["DFlashDraftModel"]);
        assert!(
            ValidatedDflash2Checkpoint::from_config_json(GLM53_DFLASH2, &config.to_string())
                .is_err()
        );

        config["architectures"] = serde_json::json!(["Glm5NextForCausalLM"]);
        assert!(
            ValidatedDflash2Checkpoint::from_config_json(GLM53_DFLASH2, &config.to_string())
                .is_err()
        );
    }

    #[test]
    #[ignore = "requires the pinned 4.9-GB incoai/GLM-5.3-DFlash2 snapshot"]
    fn validates_an_installed_pinned_dflash2_snapshot_byte_for_byte() {
        let snapshot = std::env::var_os("GLMRT_TEST_DFLASH2_SNAPSHOT")
            .map(PathBuf::from)
            .expect("set GLMRT_TEST_DFLASH2_SNAPSHOT to the pinned snapshot directory");
        let checkpoint = Dflash2Checkpoint::from_snapshot(&snapshot).unwrap();
        assert_eq!(checkpoint.weights.residency.len(), 96);
        assert_eq!(checkpoint.weights.payload_bytes, 4_918_848_512);
        assert_eq!(dflash2_resident_groups(&checkpoint).unwrap().len(), 78);
    }
}
