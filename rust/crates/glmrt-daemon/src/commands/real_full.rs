mod attention;
mod constants;
mod constraint;
mod coordinator_kernels;
mod coverage;
mod dense;
mod dflash;
mod dflash_body;
mod dflash_head;
mod dflash_preflight;
mod dflash_static;
mod dflash_update;
mod dspark;
mod dspark_attention;
mod dspark_body;
mod dspark_head;
mod dspark_kv;
mod dspark_query;
mod dspark_static;
mod dspark_update;
mod embedding;
mod entry;
mod execution_plan;
mod expert_probe;
mod experts;
mod intermediate_sharding;
mod kv;
mod layer_blocks;
mod mtp;
mod mtp_expert;
mod prefix_cache;
mod preflight;
mod probe_env;
mod rdma_reduction;
mod residency;
mod residual;
mod sampling;
mod scheduler;
mod sparse_mlp;
mod tensor_parallel;
mod types;

pub(crate) use dflash_preflight::run_dflash_preflight;
pub(crate) use dspark::run_dspark_preflight;
pub(crate) use entry::{load_real_full_serving, run_real_glm_full_preflight};
#[cfg(test)]
pub(crate) use expert_probe::REAL_NVFP4_PROTOCOL_V2_EXECUTOR;
pub(crate) use expert_probe::{
    real_nvfp4_cuda_reference_kernels_enabled, RealNvfp4ProtocolV2Executor,
    RealNvfp4ResidentPreloadPlan, REAL_NVFP4_CUDA_REFERENCE_KERNELS_ENV,
};
pub(crate) use intermediate_sharding::{
    expert_intermediate_shard_count_from_env, spark_expert_intermediate_shard_from_env,
    spark_expert_owner_reduction_config_from_env, ExpertIntermediateShard,
    EXPERT_INTERMEDIATE_SHARDS_ENV,
};
pub(crate) use layer_blocks::{
    spark_layer_block_from_env, spark_layer_block_kv_config_from_env,
    spark_layer_block_owner_endpoint_from_env, tensor_is_spark_layer_block_resident,
    SparkLayerBlock, SPARK_LAYER_BLOCKS_ENV, SPARK_LAYER_BLOCK_KV_DTYPE_ENV,
    SPARK_LAYER_BLOCK_OWNER_ENDPOINT_ENV, SPARK_LAYER_BLOCK_RANGE_ENV,
};
pub(crate) use mtp_expert::mtp_bf16_experts_enabled;
pub(crate) use residency::preload_real_full_spark_layer_block_weights;
pub(crate) use tensor_parallel::{
    preload_real_full_spark_transformer_tp_weights, probe_spark_transformer_tp_collective_from_env,
    spark_transformer_tp_from_env, tensor_is_spark_transformer_tp_resident, SparkTransformerTp,
    SPARK_TRANSFORMER_TP_COLLECTIVE_PROBE_ITERS_ENV, SPARK_TRANSFORMER_TP_ENV,
    SPARK_TRANSFORMER_TP_PORT_ENV, SPARK_TRANSFORMER_TP_RANGE_ENV, SPARK_TRANSFORMER_TP_ROOT_ENV,
};

#[cfg(test)]
mod tests;
