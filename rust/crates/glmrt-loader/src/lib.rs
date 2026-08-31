mod catalog;
mod exl3_format;
mod placement;
mod snapshot;
mod tensors;
mod tokenizer;

pub use catalog::{
    build_catalog, build_catalog_for_snapshot, classification_summary_markdown, read_model_facts,
    read_safetensors_metadata, SafetensorsTensorMetadata,
};
pub use exl3_format::{
    exl3_bits_for_recipe, glm52_exl3_expert, is_glm52_exl3_recipe, is_glm_exl3_recipe,
    validate_glm52_exl3_expert_catalog, Glm52Exl3CatalogSummary, Glm52Exl3Expert,
    Glm52Exl3Projection, Glm52Exl3ProjectionKind, Glm52Exl3Tp4ResidentGeometry, GLM52_EXL3_BITS,
    GLM52_EXL3_CODEBOOK, GLM52_EXL3_EXPERT_TP_WORLD_SIZE, GLM52_EXL3_MCG_MULTIPLIER,
    GLM52_EXL3_RECIPE_K3_V1, GLM52_EXL3_T12_LUT_BYTES, GLM52_EXL3_TENSOR_FORMAT, GLM53_EXL3_BITS,
    GLM53_EXL3_RECIPE_K4_V1,
};
pub use placement::{assignments_by_owner, build_load_plan};
pub use snapshot::{
    default_hf_home, empty_catalog_for_snapshot, model_cache_dir, resolve_snapshot,
    SnapshotResolution,
};
pub use tensors::{
    dtype_byte_width, load_tensor_bytes, load_tensor_bytes_with_options, load_tensor_rows,
    load_tensor_rows_with_options, read_tensor_bytes_into, read_tensor_bytes_into_with_options,
    read_tensor_row_prefix_into, read_tensor_row_prefix_into_with_options,
    read_tensor_row_window_into, read_tensor_row_window_into_with_options, read_tensor_rows_into,
    read_tensor_rows_into_with_options, LoadedTensor, LoadedTensorRows, LoadedTensorRowsSummary,
    LoadedTensorSummary, TensorLoadOptions,
};
pub use tokenizer::{
    decode_tokenizer_ids, encode_tokenizer_text, streaming_token_decoder, LoadedTokenizer,
    StreamingTokenDecoder, TokenizerDecodeSummary, TokenizerEncodingSummary,
};

#[cfg(test)]
mod tests;
