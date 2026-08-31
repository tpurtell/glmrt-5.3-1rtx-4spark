mod constants;
mod coordinator_graphs;
mod cpu_affinity;
mod debug_expert;
mod errors;
mod expert_batch;
mod expert_host_batch;
mod expert_route_plan;
mod graph_buffers;
mod ids;
mod kv_cache;
mod layerwave;
mod model;
mod node;
mod placement;
mod tiny;
mod transport_metrics;

pub use constants::{
    COORDINATOR_HOST, DEFAULT_MODEL_ID, EXL3_MODEL_ID, EXPERT_HOSTS,
    GLM52_COMPRESSED_DSA_BF16_BYTES_PER_TOKEN, GLM52_COMPRESSED_KV_BF16_BYTES_PER_TOKEN,
    GLM52_COMPRESSED_MAIN_MLA_BF16_BYTES_PER_TOKEN, GLM52_DSA_INDEXER_LAYERS,
    GLM52_DSA_INDEXER_LAYER_IDS, GLM52_DSA_INDEXER_LAYER_IDS_WITH_MTP, GLM52_DSA_INDEX_HEAD_DIM,
    GLM52_EXPANDED_DEBUG_KV_BF16_BYTES_PER_TOKEN, GLM52_FIRST_K_DENSE_REPLACE,
    GLM52_HIDDEN_BF16_BYTES, GLM52_HIDDEN_SIZE, GLM52_MLA_FP8_DS_BYTES_PER_TOKEN,
    GLM52_MLA_FP8_DS_SCALE_BYTES_PER_TOKEN, GLM52_MLA_KV_LORA_RANK, GLM52_MLA_MXFP4_BLOCK_SIZE,
    GLM52_MLA_MXFP4_CODE_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_DS_BYTES_PER_TOKEN,
    GLM52_MLA_MXFP4_PADDING_BYTES_PER_TOKEN, GLM52_MLA_MXFP4_SCALE_BYTES_PER_TOKEN,
    GLM52_MLA_QK_ROPE_HEAD_DIM, GLM52_MLA_ROPE_THETA, GLM52_MOE_INTERMEDIATE_SIZE,
    GLM52_MTP_LAYER_ID, GLM52_NUM_HIDDEN_LAYERS, GLM52_NUM_MTP_LAYERS, GLM52_ROUTED_EXPERTS,
    GLM52_ROUTED_SCALING_FACTOR, GLM52_TOP_K, GLM52_TOTAL_LAYERS_WITH_MTP, GLM53_EXL3_MODEL_ID,
    NVIDIA_MODEL_ID, SUPPORTED_MODEL_IDS,
};
pub use coordinator_graphs::{
    coordinator_graph_bucket_for_active_rows, CoordinatorGraphInstancePlan, CoordinatorGraphKey,
    CoordinatorGraphNetworkBoundary, CoordinatorGraphShape, COORDINATOR_GRAPH_DECODE_BUCKET_ROWS,
    COORDINATOR_GRAPH_INSTANCE_COUNT, COORDINATOR_GRAPH_PREFILL_BUCKET_ROWS,
    COORDINATOR_GRAPH_SHAPES,
};
pub use cpu_affinity::pin_current_thread_to_cpu;
pub use debug_expert::{
    ExpertRequest, ExpertRequestHeader, ExpertResponse, ExpertResponseHeader, ExpertRow,
    ExpertWaveMetadata, RouteEntry,
};
pub use errors::GlmrtError;
pub use expert_batch::{ExpertBatch, ExpertBatchRow};
pub use expert_host_batch::{
    ExpertBatchRoute, ExpertHostBatch, ExpertHostBatchRow, ExpertHostBatchSet,
    ExpertHostBatchSetAccumulation, HostRowToGlobalRowMap, PartialReconstructionPlan,
};
pub use expert_route_plan::{
    plan_completion_first_routes, plan_rolling_expert_row_packs, CompletionFirstRouteGroup,
    CompletionFirstRoutePlan, CompletionRoutePlanEntry, RollingExpertRowPackAccumulator,
    RollingExpertRowPackConfig, RollingExpertRowPackEmission, RollingExpertRowPackPlan,
};
pub use graph_buffers::{
    ExpertGraphActiveCounts, ExpertGraphBufferContract, ExpertGraphExecutionEnvelope,
    ExpertGraphHostBatchLease, ExpertGraphHostBatchSetLease, ExpertGraphInstancePool,
    ExpertGraphKey, ExpertGraphPoolEntry, ExpertGraphPoolLease, ExpertGraphPoolStats,
    ExpertWorkspaceContract, HiddenRowsBufferContract, PartialOutputBufferContract,
    RouteMetadataBufferContract, EXPERT_GRAPH_ACTIVE_COUNTS_BYTES,
    EXPERT_GRAPH_HOST_ROW_GLOBAL_INDEX_BYTES, EXPERT_GRAPH_PROTOCOL_V2_LAYOUT,
    EXPERT_GRAPH_ROUTE_ENTRY_BYTES, EXPERT_GRAPH_ROW_ROUTE_COUNT_BYTES,
    EXPERT_GRAPH_TILE_METADATA_BYTES,
};
pub use ids::{LayerId, PlacementVersion, PositionId, Priority, RequestId};
pub use kv_cache::{
    KvBackedBlock, KvCacheAllocator, KvCacheBackingStore, KvCacheConfig, KvCacheDType,
    KvCacheSnapshot, KvLayout, KvReservation, KvReservationState, KvWriteRecord, KvWriteState,
    MlaKvCacheRepresentation,
};
pub use layerwave::{
    admit_layerwaves_for_iteration, plan_prefill_chunks, DecodeStep, GraphBucket, HiddenShape,
    KvBlockDescriptor, LayerWave, LayerWaveAdmission, LayerWaveMode, MtpVerifyBlock, PrefillChunk,
    PrefillChunkPolicy, RouteMetadataPlaceholder, RowSource, RowSourceKind,
};
pub use model::{DType, ModelFacts, TensorCatalog, TensorInfo, TensorRole};
pub use node::NodeRole;
pub use placement::{
    owner_for_expert, ExpertOwnerLookup, LoadPlan, PlacementPolicy, TensorAssignment,
};
pub use tiny::deterministic_tiny_completion;
pub use transport_metrics::{
    TransportCapabilities, TransportPrefillBandwidthMeasurement, TransportRttMeasurement,
};

#[cfg(test)]
mod tests;
