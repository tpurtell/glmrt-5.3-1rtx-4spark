use anyhow::{Context, Result};
use glmrt_core::{
    DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_MOE_INTERMEDIATE_SIZE,
};
use serde_json::Value;
use std::collections::BTreeSet;

pub const GLM52_EXL3_RECIPE_K3_V1: &str = "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1";
pub const GLM53_EXL3_RECIPE_K4_V1: &str = "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1";
pub const GLM52_EXL3_BITS: usize = 3;
pub const GLM53_EXL3_BITS: usize = 4;
pub const GLM52_EXL3_CODEBOOK: &str = "mcg";
pub const GLM52_EXL3_MCG_MULTIPLIER: u32 = 0xCBAC_1FED;
pub const GLM52_EXL3_TENSOR_FORMAT: &str = "exllamav3_trellis_mcg";
pub const GLM52_EXL3_EXPERT_TP_WORLD_SIZE: usize = 4;
pub const GLM52_EXL3_T12_LUT_BYTES: usize = 1 << 12;

const BASE_EXPERT_MODULE_PATTERN: &str = r"^model\.layers\.(?:[3-9]|[1-6][0-9]|7[0-7])\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)$";

pub fn is_glm52_exl3_recipe(recipe: &str) -> bool {
    recipe == GLM52_EXL3_RECIPE_K3_V1
}

pub fn exl3_bits_for_recipe(recipe: &str) -> Option<usize> {
    match recipe {
        GLM52_EXL3_RECIPE_K3_V1 => Some(GLM52_EXL3_BITS),
        GLM53_EXL3_RECIPE_K4_V1 => Some(GLM53_EXL3_BITS),
        _ => None,
    }
}

pub fn is_glm_exl3_recipe(recipe: &str) -> bool {
    exl3_bits_for_recipe(recipe).is_some()
}

pub(crate) fn exl3_recipe_from_quantization_config(
    quantization_config: Option<&Value>,
) -> Result<Option<&'static str>> {
    let Some(value) = quantization_config else {
        return Ok(None);
    };
    let method = value
        .get("quant_method")
        .or_else(|| value.get("quant_algo"))
        .and_then(Value::as_str)
        .unwrap_or("unquantized");
    if !method.eq_ignore_ascii_case("exl3") {
        return Ok(None);
    }
    for field in ["quant_method", "method", "format", "checkpoint_format"] {
        anyhow::ensure!(
            value
                .get(field)
                .and_then(Value::as_str)
                .is_some_and(|candidate| candidate.eq_ignore_ascii_case("exl3")),
            "GLM-5 EXL3 requires quantization_config.{field}=exl3"
        );
    }
    let bits = value
        .get("bits")
        .and_then(Value::as_f64)
        .context("GLM-5 EXL3 requires an integral K3 or K4 payload")?;
    let recipe = match bits {
        value if value == GLM52_EXL3_BITS as f64 => GLM52_EXL3_RECIPE_K3_V1,
        value if value == GLM53_EXL3_BITS as f64 => GLM53_EXL3_RECIPE_K4_V1,
        _ => anyhow::bail!("GLM-5 EXL3 requires an exact K3 or K4 payload"),
    };
    anyhow::ensure!(
        value.get("codebook").and_then(Value::as_str) == Some(GLM52_EXL3_CODEBOOK)
            && value.get("out_scales").and_then(Value::as_str) == Some("auto")
            && value.get("group_size").and_then(Value::as_i64) == Some(-1)
            && value.get("desc_act").and_then(Value::as_bool) == Some(false),
        "GLM-5 EXL3 requires the MCG/auto-scale storage contract"
    );
    let includes = value
        .get("module_include")
        .and_then(Value::as_array)
        .context("GLM-5 EXL3 requires module_include")?;
    anyhow::ensure!(
        includes.len() == 1 && includes[0].as_str() == Some(BASE_EXPERT_MODULE_PATTERN),
        "GLM-5 EXL3 must select exactly the routed base experts in layers 3 through 77"
    );
    anyhow::ensure!(
        value
            .get("tensor_storage")
            .and_then(Value::as_object)
            .is_some_and(|storage| !storage.is_empty()),
        "GLM-5 EXL3 tensor_storage cannot be empty"
    );
    Ok(Some(recipe))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52Exl3Tp4ResidentGeometry {
    pub trellis_bits: usize,
    pub hidden_size: usize,
    pub global_intermediate_size: usize,
    pub local_intermediate_size: usize,
    pub experts: usize,
    pub top_k: usize,
    pub base_routed_layers: usize,
    pub projection_trellis_bytes: u64,
    pub w13_trellis_bytes: u64,
    pub w2_trellis_bytes: u64,
    pub hidden_rotation_table_bytes: u64,
    pub intermediate_rotation_bytes: u64,
    pub scalar_metadata_bytes: u64,
}

impl Glm52Exl3Tp4ResidentGeometry {
    pub fn from_model_facts(facts: &ModelFacts) -> Result<Self> {
        anyhow::ensure!(
            is_glm_exl3_recipe(&facts.quantization_recipe),
            "GLM-5 EXL3 TP4 geometry requires a supported recipe, got {:?}",
            facts.quantization_recipe
        );
        let trellis_bits = exl3_bits_for_recipe(&facts.quantization_recipe)
            .context("GLM-5 EXL3 recipe has no trellis bitrate")?;
        anyhow::ensure!(
            facts.hidden_size > 0 && facts.hidden_size % 128 == 0,
            "EXL3 hidden size {} must be a positive H128 multiple",
            facts.hidden_size
        );
        anyhow::ensure!(
            GLM52_MOE_INTERMEDIATE_SIZE % (GLM52_EXL3_EXPERT_TP_WORLD_SIZE * 128) == 0,
            "GLM-5 EXL3 intermediate size is not TP4/H128 aligned"
        );
        anyhow::ensure!(
            facts.first_k_dense_replace == GLM52_FIRST_K_DENSE_REPLACE
                && facts.num_hidden_layers > facts.first_k_dense_replace,
            "invalid GLM-5 routed layer interval {}..{}",
            facts.first_k_dense_replace,
            facts.num_hidden_layers
        );
        anyhow::ensure!(
            facts.routed_experts > 0 && facts.top_k > 0 && facts.top_k <= facts.routed_experts,
            "invalid GLM-5 routed geometry: experts={} top_k={}",
            facts.routed_experts,
            facts.top_k
        );
        let hidden = facts.hidden_size as u64;
        let experts = facts.routed_experts as u64;
        let local_intermediate_size = GLM52_MOE_INTERMEDIATE_SIZE / GLM52_EXL3_EXPERT_TP_WORLD_SIZE;
        let local_intermediate = local_intermediate_size as u64;
        let projection_trellis_bytes = hidden
            .checked_mul(local_intermediate)
            .and_then(|values| values.checked_mul(trellis_bits as u64))
            .and_then(|bits| bits.checked_div(8))
            .context("GLM-5 EXL3 projection size overflow")?;
        let w13_trellis_bytes = projection_trellis_bytes
            .checked_mul(experts)
            .and_then(|bytes| bytes.checked_mul(2))
            .context("GLM-5 EXL3 W13 size overflow")?;
        let w2_trellis_bytes = projection_trellis_bytes
            .checked_mul(experts)
            .context("GLM-5 EXL3 W2 size overflow")?;
        let hidden_rotation_table_bytes = experts
            .checked_mul(hidden)
            .and_then(|values| values.checked_mul(2))
            .context("GLM-5 EXL3 hidden rotation size overflow")?;
        let intermediate_rotation_bytes = experts
            .checked_mul(3)
            .and_then(|values| values.checked_mul(local_intermediate))
            .and_then(|values| values.checked_mul(2))
            .context("GLM-5 EXL3 intermediate rotation size overflow")?;
        // The final layer slab owns only one unit global scale per expert.
        // Expert offsets, shape scalars, and the T12 lookup table are AOT
        // kernel constants or shared route workspace, not per-layer weights.
        let scalar_metadata_bytes = facts
            .routed_experts
            .checked_mul(std::mem::size_of::<f32>())
            .context("GLM-5 EXL3 unit global scale size overflow")?
            as u64;
        Ok(Self {
            trellis_bits,
            hidden_size: facts.hidden_size,
            global_intermediate_size: GLM52_MOE_INTERMEDIATE_SIZE,
            local_intermediate_size,
            experts: facts.routed_experts,
            top_k: facts.top_k,
            base_routed_layers: facts.num_hidden_layers - facts.first_k_dense_replace,
            projection_trellis_bytes,
            w13_trellis_bytes,
            w2_trellis_bytes,
            hidden_rotation_table_bytes,
            intermediate_rotation_bytes,
            scalar_metadata_bytes,
        })
    }

    pub fn resident_weight_bytes_per_layer(self) -> u64 {
        self.w13_trellis_bytes + self.w2_trellis_bytes
    }

    pub fn resident_rotation_bytes_per_layer(self) -> u64 {
        3 * self.hidden_rotation_table_bytes + self.intermediate_rotation_bytes
    }

    pub fn resident_total_bytes_per_layer(self) -> u64 {
        self.resident_weight_bytes_per_layer()
            + self.resident_rotation_bytes_per_layer()
            + self.scalar_metadata_bytes
    }

    pub fn resident_total_bytes_per_rank(self) -> u64 {
        self.resident_total_bytes_per_layer() * self.base_routed_layers as u64
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm52Exl3ProjectionKind {
    Gate,
    Up,
    Down,
}

impl Glm52Exl3ProjectionKind {
    fn stem(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    fn logical_shape(self, catalog: &TensorCatalog) -> (usize, usize) {
        match self {
            Self::Gate | Self::Up => (catalog.facts.hidden_size, GLM52_MOE_INTERMEDIATE_SIZE),
            Self::Down => (GLM52_MOE_INTERMEDIATE_SIZE, catalog.facts.hidden_size),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Glm52Exl3Projection<'a> {
    pub kind: Glm52Exl3ProjectionKind,
    pub trellis: &'a TensorInfo,
    pub suh: &'a TensorInfo,
    pub svh: &'a TensorInfo,
    pub mcg: &'a TensorInfo,
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Glm52Exl3Expert<'a> {
    pub layer_id: usize,
    pub expert_id: usize,
    pub gate: Glm52Exl3Projection<'a>,
    pub up: Glm52Exl3Projection<'a>,
    pub down: Glm52Exl3Projection<'a>,
}

impl<'a> Glm52Exl3Expert<'a> {
    pub fn sparkinfer_w13(self) -> [Glm52Exl3Projection<'a>; 2] {
        [self.gate, self.up]
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52Exl3CatalogSummary {
    pub base_routed_layers: usize,
    pub experts_per_layer: usize,
    pub expert_tensors: usize,
    pub trellis_bytes: u64,
    pub rotation_bytes: u64,
}

pub fn glm52_exl3_expert(
    catalog: &TensorCatalog,
    layer_id: usize,
    expert_id: usize,
) -> Result<Glm52Exl3Expert<'_>> {
    anyhow::ensure!(
        layer_id >= catalog.facts.first_k_dense_replace
            && layer_id < catalog.facts.num_hidden_layers,
        "GLM-5 EXL3 layer {layer_id} is outside the base routed interval"
    );
    anyhow::ensure!(
        expert_id < catalog.facts.routed_experts,
        "GLM-5 EXL3 expert {expert_id} exceeds {} routed experts",
        catalog.facts.routed_experts
    );
    let projection = |kind: Glm52Exl3ProjectionKind| -> Result<Glm52Exl3Projection<'_>> {
        let base = format!(
            "model.layers.{layer_id}.mlp.experts.{expert_id}.{}",
            kind.stem()
        );
        let find = |suffix: &str| -> Result<&TensorInfo> {
            let name = format!("{base}.{suffix}");
            catalog
                .tensors
                .binary_search_by(|tensor| tensor.name.as_str().cmp(name.as_str()))
                .ok()
                .map(|index| &catalog.tensors[index])
                .with_context(|| format!("missing GLM-5 EXL3 tensor {name}"))
        };
        let (input_features, output_features) = kind.logical_shape(catalog);
        let value = Glm52Exl3Projection {
            kind,
            trellis: find("trellis")?,
            suh: find("suh")?,
            svh: find("svh")?,
            mcg: find("mcg")?,
            input_features,
            output_features,
        };
        validate_projection(
            value,
            exl3_bits_for_recipe(&catalog.facts.quantization_recipe)
                .context("GLM-5 EXL3 catalog has no trellis bitrate")?,
        )?;
        Ok(value)
    };
    Ok(Glm52Exl3Expert {
        layer_id,
        expert_id,
        gate: projection(Glm52Exl3ProjectionKind::Gate)?,
        up: projection(Glm52Exl3ProjectionKind::Up)?,
        down: projection(Glm52Exl3ProjectionKind::Down)?,
    })
}

pub fn validate_glm52_exl3_expert_catalog(
    catalog: &TensorCatalog,
) -> Result<Glm52Exl3CatalogSummary> {
    anyhow::ensure!(
        is_glm_exl3_recipe(&catalog.facts.quantization_recipe),
        "GLM-5 EXL3 validation requires a supported recipe, got {}",
        catalog.facts.quantization_recipe
    );
    let geometry = Glm52Exl3Tp4ResidentGeometry::from_model_facts(&catalog.facts)?;
    let mut expected_names = BTreeSet::new();
    let mut trellis_bytes = 0_u64;
    let mut rotation_bytes = 0_u64;
    for layer_id in catalog.facts.first_k_dense_replace..catalog.facts.num_hidden_layers {
        for expert_id in 0..catalog.facts.routed_experts {
            let expert = glm52_exl3_expert(catalog, layer_id, expert_id).with_context(|| {
                format!("validating GLM-5 EXL3 layer {layer_id} expert {expert_id}")
            })?;
            for projection in [expert.gate, expert.up, expert.down] {
                for tensor in [
                    projection.trellis,
                    projection.suh,
                    projection.svh,
                    projection.mcg,
                ] {
                    expected_names.insert(tensor.name.as_str());
                }
                trellis_bytes = trellis_bytes
                    .checked_add(projection.trellis.byte_length)
                    .context("GLM-5 EXL3 trellis total overflow")?;
                rotation_bytes = rotation_bytes
                    .checked_add(projection.suh.byte_length)
                    .and_then(|bytes| bytes.checked_add(projection.svh.byte_length))
                    .context("GLM-5 EXL3 rotation total overflow")?;
            }
        }
    }
    let actual_names = catalog
        .tensors
        .iter()
        .filter(|tensor| {
            tensor.role == TensorRole::RoutedExpert
                && tensor.layer_id.is_some_and(|layer| {
                    let layer = layer as usize;
                    layer >= catalog.facts.first_k_dense_replace
                        && layer < catalog.facts.num_hidden_layers
                })
        })
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    anyhow::ensure!(
        actual_names == expected_names,
        "GLM-5 EXL3 routed tensor set mismatch: expected {}, found {}; missing={:?}; unexpected={:?}",
        expected_names.len(),
        actual_names.len(),
        expected_names.difference(&actual_names).take(8).collect::<Vec<_>>(),
        actual_names.difference(&expected_names).take(8).collect::<Vec<_>>()
    );
    anyhow::ensure!(
        geometry.base_routed_layers
            == catalog.facts.num_hidden_layers - GLM52_FIRST_K_DENSE_REPLACE,
        "GLM-5 EXL3 geometry changed while validating the catalog"
    );
    Ok(Glm52Exl3CatalogSummary {
        base_routed_layers: geometry.base_routed_layers,
        experts_per_layer: geometry.experts,
        expert_tensors: expected_names.len(),
        trellis_bytes,
        rotation_bytes,
    })
}

fn validate_projection(projection: Glm52Exl3Projection<'_>, trellis_bits: usize) -> Result<()> {
    let expected_trellis = vec![
        projection.input_features / 16,
        projection.output_features / 16,
        16 * trellis_bits,
    ];
    validate_tensor(
        projection.trellis,
        DType::I16,
        &expected_trellis,
        (expected_trellis.iter().product::<usize>() * 2) as u64,
        false,
    )?;
    validate_tensor(
        projection.suh,
        DType::F16,
        &[projection.input_features],
        (projection.input_features * 2) as u64,
        true,
    )?;
    validate_tensor(
        projection.svh,
        DType::F16,
        &[projection.output_features],
        (projection.output_features * 2) as u64,
        true,
    )?;
    validate_tensor(projection.mcg, DType::I32, &[], 4, true)
}

fn validate_tensor(
    tensor: &TensorInfo,
    expected_dtype: DType,
    expected_shape: &[usize],
    expected_bytes: u64,
    expected_quantization_metadata: bool,
) -> Result<()> {
    anyhow::ensure!(
        tensor.role == TensorRole::RoutedExpert
            && tensor.dtype == expected_dtype
            && tensor.shape == expected_shape
            && tensor.byte_length == expected_bytes
            && tensor.is_quantization_metadata == expected_quantization_metadata,
        "invalid GLM-5 EXL3 tensor {}: role={:?} dtype={:?} shape={:?} bytes={} metadata={}; expected dtype={:?} shape={:?} bytes={expected_bytes} metadata={expected_quantization_metadata}",
        tensor.name,
        tensor.role,
        tensor.dtype,
        tensor.shape,
        tensor.byte_length,
        tensor.is_quantization_metadata,
        expected_dtype,
        expected_shape
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_layer_exl3_catalog(recipe: &str, trellis_bits: usize) -> TensorCatalog {
        let mut facts = ModelFacts::default();
        facts.num_hidden_layers = facts.first_k_dense_replace + 1;
        facts.quantization_recipe = recipe.to_owned();
        let layer_id = facts.first_k_dense_replace;
        let mut tensors = Vec::with_capacity(facts.routed_experts * 3 * 4);
        for expert_id in 0..facts.routed_experts {
            for (projection, input_features, output_features) in [
                ("gate_proj", facts.hidden_size, GLM52_MOE_INTERMEDIATE_SIZE),
                ("up_proj", facts.hidden_size, GLM52_MOE_INTERMEDIATE_SIZE),
                ("down_proj", GLM52_MOE_INTERMEDIATE_SIZE, facts.hidden_size),
            ] {
                let base = format!("model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}");
                for (suffix, dtype, shape, metadata) in [
                    (
                        "trellis",
                        DType::I16,
                        vec![input_features / 16, output_features / 16, trellis_bits * 16],
                        false,
                    ),
                    ("suh", DType::F16, vec![input_features], true),
                    ("svh", DType::F16, vec![output_features], true),
                    ("mcg", DType::I32, vec![], true),
                ] {
                    let byte_length = match dtype {
                        DType::I16 | DType::F16 => shape.iter().product::<usize>() * 2,
                        DType::I32 => 4,
                        _ => unreachable!("test EXL3 tensor has a fixed dtype"),
                    } as u64;
                    tensors.push(TensorInfo {
                        name: format!("{base}.{suffix}"),
                        file: "model-00001-of-00001.safetensors".to_owned(),
                        dtype,
                        shape,
                        byte_offset: 0,
                        byte_length,
                        role: TensorRole::RoutedExpert,
                        layer_id: Some(layer_id as u32),
                        expert_id: Some(expert_id as u32),
                        is_quantization_metadata: metadata,
                    });
                }
            }
        }
        tensors.sort_by(|left, right| left.name.cmp(&right.name));
        TensorCatalog {
            model_id: format!("test/glm5-exl3-k{trellis_bits}"),
            snapshot_path: format!("/test/glm5-exl3-k{trellis_bits}"),
            facts,
            tensors,
        }
    }

    #[test]
    fn exact_k3_and_k4_configs_are_recognized_without_weakening_nvfp4() {
        let config = serde_json::json!({
            "method": "exl3",
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 3.0,
            "codebook": "mcg",
            "out_scales": "auto",
            "group_size": -1,
            "desc_act": false,
            "module_include": [BASE_EXPERT_MODULE_PATTERN],
            "tensor_storage": {"model.layers.3.mlp.experts.0.gate_proj": {}}
        });
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&config)).unwrap(),
            Some(GLM52_EXL3_RECIPE_K3_V1)
        );
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&serde_json::json!({
                "quant_algo": "NVFP4"
            })))
            .unwrap(),
            None
        );
        let mut k4 = config;
        k4["bits"] = serde_json::json!(4.0);
        assert_eq!(
            exl3_recipe_from_quantization_config(Some(&k4)).unwrap(),
            Some(GLM53_EXL3_RECIPE_K4_V1)
        );
    }

    #[test]
    fn rank_local_geometry_is_compact_without_dual_residency() {
        let mut facts = ModelFacts::default();
        facts.quantization_recipe = GLM52_EXL3_RECIPE_K3_V1.to_owned();
        let geometry = Glm52Exl3Tp4ResidentGeometry::from_model_facts(&facts).unwrap();
        assert_eq!(geometry.local_intermediate_size, 512);
        assert_eq!(geometry.base_routed_layers, 75);
        assert_eq!(geometry.projection_trellis_bytes, 1_179_648);
        assert_eq!(geometry.resident_weight_bytes_per_layer(), 905_969_664);
        assert_eq!(geometry.resident_rotation_bytes_per_layer(), 10_223_616);
        assert_eq!(geometry.scalar_metadata_bytes, 1_024);
        assert_eq!(geometry.resident_total_bytes_per_layer(), 916_194_304);
        assert_eq!(geometry.resident_total_bytes_per_rank(), 68_714_572_800);

        facts.quantization_recipe = GLM53_EXL3_RECIPE_K4_V1.to_owned();
        let k4 = Glm52Exl3Tp4ResidentGeometry::from_model_facts(&facts).unwrap();
        assert_eq!(k4.trellis_bits, 4);
        assert_eq!(k4.projection_trellis_bytes, 1_572_864);
        assert_eq!(k4.resident_weight_bytes_per_layer(), 1_207_959_552);
        assert_eq!(k4.resident_total_bytes_per_layer(), 1_218_184_192);
        assert_eq!(k4.resident_total_bytes_per_rank(), 91_363_814_400);
    }

    #[test]
    fn complete_exl3_layer_requires_exact_native_tensor_namespace() {
        let mut catalog = one_layer_exl3_catalog(GLM52_EXL3_RECIPE_K3_V1, GLM52_EXL3_BITS);
        let summary = validate_glm52_exl3_expert_catalog(&catalog).unwrap();
        assert_eq!(summary.base_routed_layers, 1);
        assert_eq!(summary.experts_per_layer, 256);
        assert_eq!(summary.expert_tensors, 256 * 3 * 4);
        assert_eq!(summary.trellis_bytes, 3_623_878_656);
        assert_eq!(summary.rotation_bytes, 12_582_912);

        let expert = glm52_exl3_expert(&catalog, 3, 255).unwrap();
        let w13 = expert.sparkinfer_w13();
        assert_eq!(w13[0].trellis.name, expert.gate.trellis.name);
        assert_eq!(w13[1].trellis.name, expert.up.trellis.name);
        assert_eq!(expert.down.trellis.shape, [128, 384, 48]);

        let stale = TensorInfo {
            name: "model.layers.3.mlp.experts.0.gate_proj.weight".to_owned(),
            file: "model-00001-of-00001.safetensors".to_owned(),
            dtype: DType::Bf16,
            shape: vec![GLM52_MOE_INTERMEDIATE_SIZE, catalog.facts.hidden_size],
            byte_offset: 0,
            byte_length: (GLM52_MOE_INTERMEDIATE_SIZE * catalog.facts.hidden_size * 2) as u64,
            role: TensorRole::RoutedExpert,
            layer_id: Some(3),
            expert_id: Some(0),
            is_quantization_metadata: false,
        };
        catalog.tensors.push(stale);
        catalog
            .tensors
            .sort_by(|left, right| left.name.cmp(&right.name));
        let error = validate_glm52_exl3_expert_catalog(&catalog).unwrap_err();
        assert!(error.to_string().contains("routed tensor set mismatch"));
    }

    #[test]
    fn complete_k4_layer_uses_the_exact_native_trellis_geometry() {
        let catalog = one_layer_exl3_catalog(GLM53_EXL3_RECIPE_K4_V1, GLM53_EXL3_BITS);
        let summary = validate_glm52_exl3_expert_catalog(&catalog).unwrap();
        assert_eq!(summary.base_routed_layers, 1);
        assert_eq!(summary.experts_per_layer, 256);
        assert_eq!(summary.expert_tensors, 256 * 3 * 4);
        assert_eq!(summary.trellis_bytes, 4_831_838_208);
        assert_eq!(summary.rotation_bytes, 12_582_912);

        let expert = glm52_exl3_expert(&catalog, 3, 255).unwrap();
        assert_eq!(expert.gate.trellis.shape, [384, 128, 64]);
        assert_eq!(expert.up.trellis.shape, [384, 128, 64]);
        assert_eq!(expert.down.trellis.shape, [128, 384, 64]);
    }
}
