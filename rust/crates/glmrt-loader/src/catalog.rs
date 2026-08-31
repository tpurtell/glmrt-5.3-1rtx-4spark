use crate::exl3_format::{
    exl3_recipe_from_quantization_config, is_glm_exl3_recipe, validate_glm52_exl3_expert_catalog,
};
use crate::snapshot::resolve_snapshot;
use anyhow::{Context, Result};
use glmrt_core::{
    DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, GLM52_MOE_INTERMEDIATE_SIZE,
};
use serde::Deserialize;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

const EXTERNAL_QUANTIZATION_CONFIG_FILES: [&str; 2] =
    ["quantize_config.json", "quantization_config.json"];
const COMPACT_EXL3_DECLARATION_FIELDS: [&str; 4] =
    ["quant_method", "format", "checkpoint_format", "bits"];

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsTensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SafetensorsTensorMetadata {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Debug, Deserialize)]
struct ConfigFile {
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    first_k_dense_replace: Option<usize>,
    n_routed_experts: Option<usize>,
    num_experts_per_tok: Option<usize>,
    moe_intermediate_size: Option<usize>,
    quantization_config: Option<Value>,
}

pub fn build_catalog(model_id: &str, hf_home: Option<&Path>) -> Result<TensorCatalog> {
    let resolution = resolve_snapshot(model_id, hf_home)?;
    let snapshot_path = resolution
        .snapshot_path
        .as_ref()
        .with_context(|| format!("no local snapshot found for {model_id}"))?;
    build_catalog_for_snapshot(model_id, snapshot_path)
}

pub fn build_catalog_for_snapshot(model_id: &str, snapshot_path: &Path) -> Result<TensorCatalog> {
    let facts = read_model_facts(model_id, snapshot_path)?;
    let index_path = snapshot_path.join("model.safetensors.index.json");
    let index: SafetensorsIndex = serde_json::from_reader(
        File::open(&index_path).with_context(|| format!("opening {}", index_path.display()))?,
    )
    .with_context(|| format!("parsing {}", index_path.display()))?;

    let files = index
        .weight_map
        .values()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let header_slots = (0..files.len())
        .map(|_| Mutex::new(None))
        .collect::<Vec<OptionSlot<BTreeMap<String, SafetensorsTensorHeader>>>>();
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(32)
        .min(files.len().max(1));
    let next_file = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| loop {
                let index = next_file.fetch_add(1, Ordering::Relaxed);
                let Some(file_name) = files.get(index) else {
                    break;
                };
                let path = snapshot_path.join(file_name);
                let parsed = parse_safetensors_header(&path)
                    .with_context(|| format!("parsing safetensors header {}", path.display()));
                *header_slots[index]
                    .lock()
                    .expect("safetensors header result slot is poisoned") = Some(parsed);
            });
        }
    });
    let mut header_by_tensor = BTreeMap::new();
    for (file_name, slot) in files.into_iter().zip(header_slots) {
        let entries = slot
            .into_inner()
            .map_err(|_| anyhow::anyhow!("safetensors header result slot is poisoned"))?
            .with_context(|| format!("safetensors header worker did not visit {file_name}"))??;
        for (name, header) in entries {
            header_by_tensor.insert(name, (file_name.clone(), header));
        }
    }

    let mut tensors = Vec::with_capacity(index.weight_map.len());
    for (name, file_name) in index.weight_map {
        let (header_file, header) = header_by_tensor
            .get(&name)
            .with_context(|| format!("tensor {name} missing from safetensors header"))?;
        if header_file != &file_name {
            anyhow::bail!(
                "index/header file mismatch for {name}: index={file_name} header={header_file}"
            );
        }
        let layer_id = extract_number_after(&name, "model.layers.").map(|v| v as u32);
        let expert_id = extract_number_after(&name, ".mlp.experts.").map(|v| v as u32);
        let dtype = DType::from_safetensors(&header.dtype);
        let is_quantization_metadata = is_quantization_tensor(&name);
        let role = classify_tensor(&name, layer_id, expert_id, is_quantization_metadata, &facts);
        tensors.push(TensorInfo {
            name,
            file: file_name,
            dtype,
            shape: header.shape.clone(),
            byte_offset: header.data_offsets[0],
            byte_length: header.data_offsets[1] - header.data_offsets[0],
            role,
            layer_id,
            expert_id,
            is_quantization_metadata,
        });
    }
    tensors.sort_by(|a, b| a.name.cmp(&b.name));

    let catalog = TensorCatalog {
        model_id: model_id.to_owned(),
        snapshot_path: snapshot_path.display().to_string(),
        facts,
        tensors,
    };
    if is_glm_exl3_recipe(&catalog.facts.quantization_recipe) {
        validate_glm52_exl3_expert_catalog(&catalog)
            .context("validating calibrated GLM-5 EXL3 routed experts")?;
    }
    Ok(catalog)
}

type OptionSlot<T> = Mutex<Option<Result<T>>>;

pub fn read_safetensors_metadata(path: &Path) -> Result<Vec<SafetensorsTensorMetadata>> {
    let entries = parse_safetensors_header(path)?;
    Ok(entries
        .into_iter()
        .map(|(name, header)| SafetensorsTensorMetadata {
            name,
            dtype: DType::from_safetensors(&header.dtype),
            shape: header.shape,
            byte_offset: header.data_offsets[0],
            byte_length: header.data_offsets[1] - header.data_offsets[0],
        })
        .collect())
}

pub fn read_model_facts(model_id: &str, snapshot_path: &Path) -> Result<ModelFacts> {
    let config_path = snapshot_path.join("config.json");
    let config: ConfigFile = serde_json::from_reader(
        File::open(&config_path).with_context(|| format!("opening {}", config_path.display()))?,
    )
    .with_context(|| format!("parsing {}", config_path.display()))?;
    anyhow::ensure!(
        config
            .moe_intermediate_size
            .unwrap_or(GLM52_MOE_INTERMEDIATE_SIZE)
            == GLM52_MOE_INTERMEDIATE_SIZE,
        "unsupported GLM-5 MoE intermediate size for {model_id}"
    );
    let quantization_config =
        resolve_model_quantization_config(snapshot_path, config.quantization_config.as_ref())?;
    let recipe = if let Some(recipe) =
        exl3_recipe_from_quantization_config(quantization_config.as_deref())?
    {
        recipe.to_owned()
    } else {
        quantization_config
            .as_deref()
            .and_then(|value| value.get("quant_algo"))
            .and_then(Value::as_str)
            .map(|algo| {
                if algo.eq_ignore_ascii_case("NVFP4") {
                    "glm52_nvfp4_lukealonso_v1".to_owned()
                } else {
                    format!("unknown_{algo}")
                }
            })
            .unwrap_or_else(|| "glm52_nvfp4_lukealonso_v1".to_owned())
    };
    Ok(ModelFacts {
        model_id: model_id.to_owned(),
        hidden_size: config.hidden_size.unwrap_or(glmrt_core::GLM52_HIDDEN_SIZE),
        num_hidden_layers: config
            .num_hidden_layers
            .unwrap_or(glmrt_core::GLM52_NUM_HIDDEN_LAYERS),
        first_k_dense_replace: config
            .first_k_dense_replace
            .unwrap_or(glmrt_core::GLM52_FIRST_K_DENSE_REPLACE),
        routed_experts: config
            .n_routed_experts
            .unwrap_or(glmrt_core::GLM52_ROUTED_EXPERTS),
        top_k: config
            .num_experts_per_tok
            .unwrap_or(glmrt_core::GLM52_TOP_K),
        quantization_recipe: recipe,
    })
}

fn resolve_model_quantization_config<'a>(
    snapshot_path: &Path,
    embedded: Option<&'a Value>,
) -> Result<Option<Cow<'a, Value>>> {
    let Some(embedded) = embedded else {
        return Ok(None);
    };
    let quant_method = embedded
        .get("quant_method")
        .or_else(|| embedded.get("quant_algo"))
        .and_then(Value::as_str)
        .unwrap_or("unquantized");
    let compact_exl3 =
        quant_method.eq_ignore_ascii_case("exl3") && embedded.get("tensor_storage").is_none();
    if !compact_exl3 {
        return Ok(Some(Cow::Borrowed(embedded)));
    }
    let embedded_object = embedded
        .as_object()
        .context("config.json quantization_config must be a JSON object")?;
    let mut resolved: Option<(String, Value)> = None;
    for filename in EXTERNAL_QUANTIZATION_CONFIG_FILES {
        let path = snapshot_path.join(filename);
        if !path.is_file() {
            continue;
        }
        let full: Value = serde_json::from_reader(
            File::open(&path).with_context(|| format!("opening {}", path.display()))?,
        )
        .with_context(|| format!("parsing {}", path.display()))?;
        let full_object = full.as_object().with_context(|| {
            format!("external EXL3 config {} must be an object", path.display())
        })?;
        let legacy_compact_matches = full_object.len() == embedded_object.len() + 1
            && embedded_object
                .iter()
                .all(|(key, value)| full_object.get(key) == Some(value));
        let minimal_compact_matches = embedded_object.len()
            == COMPACT_EXL3_DECLARATION_FIELDS.len()
            && COMPACT_EXL3_DECLARATION_FIELDS.iter().all(|field| {
                embedded_object.get(*field).is_some()
                    && embedded_object.get(*field) == full_object.get(*field)
            });
        anyhow::ensure!(
            full_object.get("tensor_storage").is_some()
                && (legacy_compact_matches || minimal_compact_matches),
            "config.json quantization_config differs from compact {}",
            path.display()
        );
        if let Some((prior_filename, prior)) = resolved.as_ref() {
            anyhow::ensure!(
                prior == &full,
                "external EXL3 configs {prior_filename} and {filename} differ"
            );
        } else {
            resolved = Some((filename.to_owned(), full));
        }
    }
    let (_, full) = resolved.with_context(|| {
        format!(
            "compact GLM-5 EXL3 config requires {} or {} in {}",
            EXTERNAL_QUANTIZATION_CONFIG_FILES[0],
            EXTERNAL_QUANTIZATION_CONFIG_FILES[1],
            snapshot_path.display()
        )
    })?;
    Ok(Some(Cow::Owned(full)))
}

fn parse_safetensors_header(path: &Path) -> Result<BTreeMap<String, SafetensorsTensorHeader>> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("reading metadata for {}", path.display()))?
        .len();
    let mut len_bytes = [0_u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    let header_len_usize: usize = header_len
        .try_into()
        .context("safetensors header length does not fit in memory")?;
    let data_start = 8_u64
        .checked_add(header_len)
        .context("safetensors data offset overflow")?;
    anyhow::ensure!(
        data_start <= file_len,
        "safetensors header extends beyond {}: data starts at {data_start}, file length is {file_len}",
        path.display()
    );
    let mut header_bytes = vec![0_u8; header_len_usize];
    file.read_exact(&mut header_bytes)?;
    let raw: BTreeMap<String, Value> = serde_json::from_slice(&header_bytes)?;
    let mut entries = BTreeMap::new();
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let mut header: SafetensorsTensorHeader = serde_json::from_value(value)
            .with_context(|| format!("parsing header entry {name} in {}", path.display()))?;
        anyhow::ensure!(
            header.data_offsets[0] <= header.data_offsets[1],
            "invalid safetensors offsets for {name} in {}: {:?}",
            path.display(),
            header.data_offsets
        );
        header.data_offsets[0] = header.data_offsets[0]
            .checked_add(data_start)
            .context("safetensors tensor start offset overflow")?;
        header.data_offsets[1] = header.data_offsets[1]
            .checked_add(data_start)
            .context("safetensors tensor end offset overflow")?;
        anyhow::ensure!(
            header.data_offsets[1] <= file_len,
            "safetensors tensor {name} extends beyond {}: end {}, file length {file_len}",
            path.display(),
            header.data_offsets[1]
        );
        entries.insert(name, header);
    }
    Ok(entries)
}

fn extract_number_after(name: &str, marker: &str) -> Option<usize> {
    let start = name.find(marker)? + marker.len();
    let digits = name[start..]
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse().ok()
    }
}

pub(crate) fn is_quantization_tensor(name: &str) -> bool {
    name.ends_with(".input_scale")
        || name.ends_with(".weight_scale")
        || name.ends_with(".weight_scale_2")
        || name.contains(".input_scale.")
        || name.contains(".weight_scale.")
        || name.ends_with(".suh")
        || name.ends_with(".svh")
        || name.ends_with(".mcg")
}

pub(crate) fn classify_tensor(
    name: &str,
    layer_id: Option<u32>,
    expert_id: Option<u32>,
    _is_quantization_metadata: bool,
    facts: &ModelFacts,
) -> TensorRole {
    if expert_id.is_some() && name.contains(".mlp.experts.") {
        return TensorRole::RoutedExpert;
    }
    if layer_id
        .map(|layer| layer as usize >= facts.num_hidden_layers)
        .unwrap_or(false)
    {
        return TensorRole::Mtp;
    }
    if name.contains(".mlp.shared_experts.") {
        return TensorRole::SharedExpert;
    }
    if name.contains(".mlp.gate.") {
        return TensorRole::Router;
    }
    if name == "model.embed_tokens.weight" {
        return TensorRole::Embedding;
    }
    if name == "lm_head.weight" {
        return TensorRole::LmHead;
    }
    if name.contains(".self_attn.") {
        return TensorRole::Attention;
    }
    if name.contains("layernorm") || name.ends_with(".norm.weight") || name.contains(".norm.") {
        return TensorRole::Norm;
    }
    if name.contains(".mlp.") {
        return TensorRole::DenseMlp;
    }
    TensorRole::Other
}

pub fn classification_summary_markdown(catalog: &TensorCatalog) -> String {
    let mut by_role = BTreeMap::<String, usize>::new();
    let mut routed_weight = 0_usize;
    let mut routed_quant = 0_usize;
    let mut shared = 0_usize;
    let mut files = BTreeSet::new();
    for tensor in &catalog.tensors {
        *by_role.entry(format!("{:?}", tensor.role)).or_default() += 1;
        files.insert(tensor.file.clone());
        if tensor.role == TensorRole::RoutedExpert && tensor.is_quantization_metadata {
            routed_quant += 1;
        } else if tensor.role == TensorRole::RoutedExpert {
            routed_weight += 1;
        }
        if tensor.role == TensorRole::SharedExpert {
            shared += 1;
        }
    }
    let mut out = String::new();
    out.push_str("# Tensor Classification Summary\n\n");
    out.push_str(&format!("- Model: `{}`\n", catalog.model_id));
    out.push_str(&format!("- Snapshot: `{}`\n", catalog.snapshot_path));
    out.push_str(&format!("- Catalog hash: `{}`\n", catalog.content_hash()));
    out.push_str(&format!("- Tensor count: `{}`\n", catalog.tensors.len()));
    out.push_str(&format!("- Safetensors files: `{}`\n", files.len()));
    out.push_str(&format!("- Hidden size: `{}`\n", catalog.facts.hidden_size));
    out.push_str(&format!(
        "- Hidden layers: `{}`\n",
        catalog.facts.num_hidden_layers
    ));
    out.push_str(&format!(
        "- First dense layers: `{}`\n",
        catalog.facts.first_k_dense_replace
    ));
    out.push_str(&format!(
        "- Routed experts per MoE layer: `{}`\n",
        catalog.facts.routed_experts
    ));
    out.push_str(&format!(
        "- Top-k experts per token: `{}`\n",
        catalog.facts.top_k
    ));
    out.push_str(&format!(
        "- Quantization recipe: `{}`\n\n",
        catalog.facts.quantization_recipe
    ));
    out.push_str("## Role Counts\n\n");
    out.push_str("| Role | Tensors |\n| --- | ---: |\n");
    for (role, count) in by_role {
        out.push_str(&format!("| {role} | {count} |\n"));
    }
    out.push_str("\n## Routed Expert Detail\n\n");
    out.push_str(&format!(
        "- Routed expert non-scale tensors: `{routed_weight}`\n"
    ));
    out.push_str(&format!(
        "- Routed expert quantization tensors: `{routed_quant}`\n"
    ));
    out.push_str(&format!("- Shared expert tensors: `{shared}`\n"));
    out
}

#[cfg(test)]
mod quantization_config_tests {
    use super::{read_model_facts, resolve_model_quantization_config};
    use crate::exl3_format::GLM53_EXL3_RECIPE_K4_V1;

    #[test]
    fn compact_exl3_config_resolves_only_an_exact_external_storage_extension() {
        let temporary = tempfile::tempdir().unwrap();
        let embedded = serde_json::json!({
            "method": "exl3",
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 3.0,
            "codebook": "mcg",
            "out_scales": "auto",
            "group_size": -1,
            "desc_act": false,
            "module_include": [
                "^model\\.layers\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"
            ],
            "meta": {"quantizer": "pinned"}
        });
        let mut full = embedded.as_object().unwrap().clone();
        full.insert(
            "tensor_storage".to_owned(),
            serde_json::json!({"model.layers.3.mlp.experts.0.gate_proj": {}}),
        );
        std::fs::write(
            temporary.path().join("quantize_config.json"),
            serde_json::to_vec(&full).unwrap(),
        )
        .unwrap();

        let resolved = resolve_model_quantization_config(temporary.path(), Some(&embedded))
            .unwrap()
            .unwrap();
        assert_eq!(resolved.as_ref(), &serde_json::Value::Object(full.clone()));

        full.insert("bits".to_owned(), serde_json::json!(2.0));
        std::fs::write(
            temporary.path().join("quantize_config.json"),
            serde_json::to_vec(&full).unwrap(),
        )
        .unwrap();
        let error =
            resolve_model_quantization_config(temporary.path(), Some(&embedded)).unwrap_err();
        assert!(error.to_string().contains("differs from compact"));
    }

    #[test]
    fn model_facts_require_the_complete_external_k4_ledger() {
        let temporary = tempfile::tempdir().unwrap();
        let embedded = serde_json::json!({
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 4.0
        });
        let mut full = serde_json::json!({
            "method": "exl3",
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 4.0,
            "codebook": "mcg",
            "out_scales": "auto",
            "group_size": -1,
            "desc_act": false,
            "module_include": [
                "^model\\.layers\\.(?:[3-9]|[1-6][0-9]|7[0-7])\\.mlp\\.experts\\.\\d+\\.(?:gate_proj|up_proj|down_proj)$"
            ],
            "meta": {
                "ds4rt_error_ledger": {
                    "family_join": {"sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},
                    "run": {"kind": "production"}
                }
            }
        })
        .as_object()
        .unwrap()
        .clone();
        std::fs::write(
            temporary.path().join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "moe_intermediate_size": 2048,
                "quantization_config": embedded
            }))
            .unwrap(),
        )
        .unwrap();
        full.insert(
            "tensor_storage".to_owned(),
            serde_json::json!({"model.layers.3.mlp.experts.0.gate_proj": {}}),
        );
        let external = temporary.path().join("quantize_config.json");
        std::fs::write(&external, serde_json::to_vec(&full).unwrap()).unwrap();

        let facts = read_model_facts("test/glm53-exl3-k4", temporary.path()).unwrap();
        assert_eq!(facts.quantization_recipe, GLM53_EXL3_RECIPE_K4_V1);

        std::fs::remove_file(external).unwrap();
        let error = read_model_facts("test/glm53-exl3-k4", temporary.path()).unwrap_err();
        assert!(error
            .to_string()
            .contains("compact GLM-5 EXL3 config requires quantize_config.json"));
    }
}
