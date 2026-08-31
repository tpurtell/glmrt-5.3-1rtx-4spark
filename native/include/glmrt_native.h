#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum glmrt_status_t {
  GLMRT_STATUS_OK = 0,
  GLMRT_STATUS_INVALID_ARGUMENT = 1,
  GLMRT_STATUS_BUFFER_TOO_SMALL = 2,
  GLMRT_STATUS_CUDA_UNAVAILABLE = 3,
  GLMRT_STATUS_ALLOCATION_FAILED = 4,
  GLMRT_STATUS_COPY_FAILED = 5,
  GLMRT_STATUS_INTERNAL_ERROR = 6,
  GLMRT_STATUS_RDMA_UNAVAILABLE = 7,
  GLMRT_STATUS_NCCL_UNAVAILABLE = 8,
} glmrt_status_t;

typedef enum glmrt_xgrammar_kind_t {
  GLMRT_XGRAMMAR_JSON_OBJECT = 1,
  GLMRT_XGRAMMAR_JSON_SCHEMA = 2,
  GLMRT_XGRAMMAR_STRUCTURAL_TAG = 3,
} glmrt_xgrammar_kind_t;

glmrt_status_t glmrt_xgrammar_compiler_create(
    const char* tokenizer_json_path, size_t vocab_size, const int32_t* stop_token_ids,
    size_t stop_token_count, void** out_compiler, char* error, size_t error_bytes);
glmrt_status_t glmrt_xgrammar_compiler_destroy(void* compiler);
glmrt_status_t glmrt_xgrammar_compile(
    void* compiler, glmrt_xgrammar_kind_t kind, const char* grammar_json, int strict,
    void** out_grammar, char* error, size_t error_bytes);
glmrt_status_t glmrt_xgrammar_grammar_destroy(void* grammar);
glmrt_status_t glmrt_xgrammar_matcher_create(
    const void* grammar, void** out_matcher, char* error, size_t error_bytes);
glmrt_status_t glmrt_xgrammar_matcher_fork(
    const void* matcher, void** out_matcher, char* error, size_t error_bytes);
glmrt_status_t glmrt_xgrammar_matcher_destroy(void* matcher);
glmrt_status_t glmrt_xgrammar_matcher_fill_bitmask(
    void* matcher, uint32_t* bitmask, size_t bitmask_words, int* out_needs_mask,
    char* error, size_t error_bytes);
glmrt_status_t glmrt_xgrammar_matcher_accept_token(
    void* matcher, uint32_t token_id, int* out_accepted, char* error, size_t error_bytes);
glmrt_status_t glmrt_xgrammar_matcher_is_completed(
    const void* matcher, int* out_completed, char* error, size_t error_bytes);

typedef enum glmrt_device_buffer_flags_t {
  GLMRT_DEVICE_BUFFER_FLAG_NONE = 0,
  GLMRT_DEVICE_BUFFER_FLAG_HOST_FALLBACK = 1,
  GLMRT_DEVICE_BUFFER_FLAG_MANAGED = 2,
  GLMRT_DEVICE_BUFFER_FLAG_MAPPED_HOST = 4,
} glmrt_device_buffer_flags_t;

typedef enum glmrt_host_buffer_flags_t {
  GLMRT_HOST_BUFFER_FLAG_NONE = 0,
  GLMRT_HOST_BUFFER_FLAG_PINNED = 1,
  GLMRT_HOST_BUFFER_FLAG_HOST_FALLBACK = 2,
  GLMRT_HOST_BUFFER_FLAG_MAPPED = 4,
} glmrt_host_buffer_flags_t;

typedef struct glmrt_cuda_device_info_t {
  int device_id;
  int cuda_available;
  int compute_capability_major;
  int compute_capability_minor;
  int integrated;
  int can_map_host_memory;
  int unified_addressing;
  uint64_t total_memory_bytes;
  char name[128];
  char driver_version[64];
  char runtime_version[64];
} glmrt_cuda_device_info_t;

typedef struct glmrt_device_buffer_t {
  void* ptr;
  size_t bytes;
  int device_id;
  uint64_t flags;
} glmrt_device_buffer_t;

typedef struct glmrt_host_buffer_t {
  void* ptr;
  size_t bytes;
  uint64_t flags;
} glmrt_host_buffer_t;

typedef enum glmrt_route_shard_wire_dtype_t {
  GLMRT_ROUTE_SHARD_WIRE_BF16 = 1,
  GLMRT_ROUTE_SHARD_WIRE_FP8_E4M3_ROW_SCALED = 2,
  GLMRT_ROUTE_SHARD_WIRE_NVFP4_E2M1_FP8_E4M3 = 3,
} glmrt_route_shard_wire_dtype_t;

typedef enum glmrt_route_shard_local_dtype_t {
  GLMRT_ROUTE_SHARD_LOCAL_F32 = 1,
  GLMRT_ROUTE_SHARD_LOCAL_BF16 = 2,
} glmrt_route_shard_local_dtype_t;

typedef struct glmrt_route_shard_reduction_buffers_t {
  glmrt_device_buffer_t local;
  glmrt_device_buffer_t peers[3];
  glmrt_device_buffer_t output_f32;
} glmrt_route_shard_reduction_buffers_t;

typedef struct glmrt_route_shard_fp8_rail_reduction_buffers_t {
  glmrt_device_buffer_t local_bf16;
  glmrt_device_buffer_t peer_rail0[3];
  glmrt_device_buffer_t peer_rail1[3];
  glmrt_device_buffer_t output_fp8;
} glmrt_route_shard_fp8_rail_reduction_buffers_t;

typedef struct glmrt_nvfp4_route_batched_metadata_t {
  uintptr_t gate_weight;
  uintptr_t gate_scale;
  uintptr_t up_weight;
  uintptr_t up_scale;
  uintptr_t down_weight;
  uintptr_t down_scale;
  size_t intermediate;
  size_t down_weight_row_stride_bytes;
  size_t down_scale_row_stride_bytes;
  float gate_scale_2;
  float up_scale_2;
  float down_scale_2;
} glmrt_nvfp4_route_batched_metadata_t;

typedef struct glmrt_b12x_spark_w4a16_moe_buffers_t {
  glmrt_device_buffer_t input;
  glmrt_device_buffer_t w13_weight;
  glmrt_device_buffer_t w2_weight;
  glmrt_device_buffer_t fc1_output;
  glmrt_device_buffer_t activated;
  glmrt_device_buffer_t output;
  glmrt_device_buffer_t w13_scale;
  glmrt_device_buffer_t w2_scale;
  glmrt_device_buffer_t w13_global_scale;
  glmrt_device_buffer_t w2_global_scale;
  glmrt_device_buffer_t packed_route_indices;
  glmrt_device_buffer_t block_expert_ids;
  glmrt_device_buffer_t packed_route_count;
  glmrt_device_buffer_t topk_weights;
  glmrt_device_buffer_t fc1_scratch;
  glmrt_device_buffer_t fc2_scratch;
  glmrt_device_buffer_t locks;
} glmrt_b12x_spark_w4a16_moe_buffers_t;

typedef struct glmrt_b12x_spark_exl3_k3_moe_buffers_t {
  glmrt_device_buffer_t input_bf16;
  glmrt_device_buffer_t rotation_a_gate;
  glmrt_device_buffer_t rotation_a_up;
  glmrt_device_buffer_t w13_trellis;
  glmrt_device_buffer_t w2_trellis;
  glmrt_device_buffer_t unit_global_scale;
  glmrt_device_buffer_t fc1_output;
  glmrt_device_buffer_t activated;
  glmrt_device_buffer_t fc2_output;
  glmrt_device_buffer_t output_f32;
  glmrt_device_buffer_t packed_route_indices;
  glmrt_device_buffer_t block_expert_ids;
  glmrt_device_buffer_t packed_route_count;
  glmrt_device_buffer_t topk_ids;
  glmrt_device_buffer_t topk_weights;
  glmrt_device_buffer_t fc1_scratch;
  glmrt_device_buffer_t fc2_scratch;
  glmrt_device_buffer_t locks;
  glmrt_device_buffer_t intermediate_rotations;
  glmrt_device_buffer_t gate_suh;
  glmrt_device_buffer_t up_suh;
  glmrt_device_buffer_t down_svh;
} glmrt_b12x_spark_exl3_k3_moe_buffers_t;

/* K3 and K4 use the same workspace ABI; only the resident Trellis payload
 * width and selected AOT kernel differ. */
typedef glmrt_b12x_spark_exl3_k3_moe_buffers_t
    glmrt_b12x_spark_exl3_k4_moe_buffers_t;

typedef struct glmrt_b12x_coordinator_w4a16_buffers_t {
  glmrt_device_buffer_t input;
  glmrt_device_buffer_t weight;
  glmrt_device_buffer_t output;
  glmrt_device_buffer_t scale;
  glmrt_device_buffer_t global_scale;
  glmrt_device_buffer_t packed_route_indices;
  glmrt_device_buffer_t block_expert_ids;
  glmrt_device_buffer_t packed_route_count;
  glmrt_device_buffer_t topk_weights;
  glmrt_device_buffer_t c_tmp;
  glmrt_device_buffer_t locks;
} glmrt_b12x_coordinator_w4a16_buffers_t;

typedef struct glmrt_cuda_graph_capture_info_t {
  void* graph;
  void* graph_exec;
  size_t node_count;
  size_t kernel_node_count;
  size_t memcpy_node_count;
  size_t memset_node_count;
} glmrt_cuda_graph_capture_info_t;

typedef struct glmrt_bf16_summary_t {
  double checksum;
  uint64_t values;
  uint64_t finite_values;
  uint64_t nonzero_values;
} glmrt_bf16_summary_t;

typedef struct glmrt_rdma_device_info_t {
  int rdma_enabled;
  int device_count;
  int first_device_openable;
  uint64_t first_device_guid;
  char first_device_name[128];
  char first_device_transport[64];
  char status[128];
} glmrt_rdma_device_info_t;

typedef struct glmrt_rdma_host_buffer_plan_t {
  uintptr_t original_addr;
  size_t original_bytes;
  size_t alignment;
  uintptr_t registered_addr;
  size_t prefix_bytes;
  size_t registered_span_bytes;
  int span_aligned;
  int rdma_enabled;
} glmrt_rdma_host_buffer_plan_t;

typedef struct glmrt_rdma_register_probe_t {
  size_t bytes;
  int registered;
  uint32_t lkey;
  uint32_t rkey;
  char device_name[128];
} glmrt_rdma_register_probe_t;

typedef struct glmrt_rdma_rc_qp_probe_t {
  int rdma_enabled;
  int created;
  uint32_t port_num;
  uint32_t qp_num;
  uint32_t lid;
  uint32_t active_mtu;
  uint32_t requested_send_wr;
  uint32_t requested_recv_wr;
  uint32_t requested_max_sge;
  uint32_t actual_max_send_wr;
  uint32_t actual_max_recv_wr;
  uint32_t actual_max_send_sge;
  uint32_t actual_max_recv_sge;
  uint32_t actual_max_inline_data;
  char device_name[128];
  char status[128];
} glmrt_rdma_rc_qp_probe_t;

typedef struct glmrt_rdma_rc_send_recv_probe_t {
  int rdma_enabled;
  int completed;
  int payload_matches;
  uint32_t port_num;
  size_t bytes;
  uint32_t sender_qp_num;
  uint32_t receiver_qp_num;
  uint32_t send_completions;
  uint32_t recv_completions;
  uint32_t poll_iterations;
  char device_name[128];
  char status[128];
} glmrt_rdma_rc_send_recv_probe_t;

typedef struct glmrt_rdma_rc_protocol_v2_loopback_probe_t {
  int rdma_enabled;
  int completed;
  int request_payload_matches;
  int response_payload_matches;
  uint32_t port_num;
  size_t request_bytes;
  size_t response_bytes;
  uint32_t client_qp_num;
  uint32_t server_qp_num;
  uint32_t send_completions;
  uint32_t recv_completions;
  uint32_t poll_iterations;
  char device_name[128];
  char status[128];
} glmrt_rdma_rc_protocol_v2_loopback_probe_t;

typedef struct glmrt_rdma_rc_endpoint_info_t {
  int rdma_enabled;
  void* handle;
  uint32_t port_num;
  uint32_t qp_num;
  uint32_t psn;
  uint32_t lid;
  uint32_t active_mtu;
  size_t send_frame_bytes;
  size_t recv_frame_bytes;
  size_t send_registered_span_bytes;
  size_t recv_registered_span_bytes;
  uint32_t max_send_wr;
  uint32_t max_recv_wr;
  uint32_t max_sge;
  char gid_hex[33];
  char device_name[128];
  char status[128];
} glmrt_rdma_rc_endpoint_info_t;

typedef struct glmrt_rdma_rc_endpoint_buffer_view_t {
  void* host_ptr;
  void* device_ptr;
  size_t bytes;
  int device_id;
  uint64_t host_flags;
} glmrt_rdma_rc_endpoint_buffer_view_t;

typedef struct glmrt_rdma_rc_completion_stats_t {
  uint32_t expected_send_completions;
  uint32_t expected_recv_completions;
  uint32_t send_completions;
  uint32_t recv_completions;
  uint32_t poll_iterations;
  char status[128];
} glmrt_rdma_rc_completion_stats_t;

glmrt_status_t glmrt_native_version(char* out, size_t out_len);
glmrt_status_t glmrt_cuda_device_info(int device_id, glmrt_cuda_device_info_t* out);
glmrt_status_t glmrt_alloc_host_buffer(size_t bytes, glmrt_host_buffer_t* out);
glmrt_status_t glmrt_cuda_host_buffer_device_alias(glmrt_host_buffer_t host,
                                                    glmrt_device_buffer_t* out);
glmrt_status_t glmrt_free_host_buffer(glmrt_host_buffer_t* buf);
glmrt_status_t glmrt_alloc_device_buffer(size_t bytes, glmrt_device_buffer_t* out);
glmrt_status_t glmrt_alloc_managed_device_buffer(size_t bytes, glmrt_device_buffer_t* out);
glmrt_status_t glmrt_free_device_buffer(glmrt_device_buffer_t* buf);
glmrt_status_t glmrt_cuda_stream_create(void** out_cuda_stream);
glmrt_status_t glmrt_cuda_stream_create_high_priority(void** out_cuda_stream);
glmrt_status_t glmrt_cuda_stream_destroy(void* cuda_stream);
glmrt_status_t glmrt_cuda_stream_synchronize(void* cuda_stream);
glmrt_status_t glmrt_cuda_stream_wait_event(void* cuda_stream, void* cuda_event);
glmrt_status_t glmrt_cuda_event_create(void** out_cuda_event);
glmrt_status_t glmrt_cuda_event_destroy(void* cuda_event);
glmrt_status_t glmrt_cuda_event_record(void* cuda_event, void* cuda_stream);
glmrt_status_t glmrt_cuda_event_synchronize(void* cuda_event);
glmrt_status_t glmrt_cuda_event_elapsed_ms(void* start_event, void* end_event, float* out_ms);
glmrt_status_t glmrt_cuda_graph_begin_capture(void* cuda_stream);
glmrt_status_t glmrt_cuda_graph_end_capture(void* cuda_stream, void** out_cuda_graph_exec);
glmrt_status_t glmrt_cuda_graph_end_capture_retained(
    void* cuda_stream, glmrt_cuda_graph_capture_info_t* out);
glmrt_status_t glmrt_cuda_graph_launch(void* cuda_graph_exec, void* cuda_stream);
glmrt_status_t glmrt_cuda_graph_exec_update(void* cuda_graph_exec, void* cuda_graph);
glmrt_status_t glmrt_cuda_graph_destroy(void* cuda_graph);
glmrt_status_t glmrt_cuda_graph_exec_destroy(void* cuda_graph_exec);
glmrt_status_t glmrt_cuda_graph_update_rmsnorm_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t weight, glmrt_device_buffer_t out, int rows, int hidden, float eps);
glmrt_status_t glmrt_cuda_graph_update_layernorm_affine_f32_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t weight, glmrt_device_buffer_t bias, glmrt_device_buffer_t out,
    int rows, int hidden, float eps);
glmrt_status_t glmrt_cuda_graph_update_layernorm_affine_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t weight, glmrt_device_buffer_t bias, glmrt_device_buffer_t out,
    int rows, int hidden, float eps);
glmrt_status_t glmrt_cuda_graph_update_linear_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t input,
    glmrt_device_buffer_t weight, const glmrt_device_buffer_t* bias,
    glmrt_device_buffer_t output, size_t rows, size_t input_dim, size_t output_dim);
glmrt_status_t glmrt_cuda_graph_update_embedding_lookup_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t embedding, glmrt_device_buffer_t token_ids,
    glmrt_device_buffer_t out, size_t rows, size_t vocab, size_t hidden);
glmrt_status_t glmrt_cuda_graph_update_lm_head_argmax_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t hidden, glmrt_device_buffer_t lm_head,
    glmrt_device_buffer_t out_indices, glmrt_device_buffer_t out_scores, size_t rows,
    size_t hidden_dim, size_t vocab);
glmrt_status_t glmrt_cuda_graph_update_lm_head_sample_topk_topp_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t hidden, glmrt_device_buffer_t lm_head,
    glmrt_device_buffer_t random_uniforms, glmrt_device_buffer_t out_indices,
    glmrt_device_buffer_t out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p);
glmrt_status_t glmrt_cuda_graph_update_router_topk_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t hidden, glmrt_device_buffer_t router_weight,
    glmrt_device_buffer_t correction_bias, glmrt_device_buffer_t topk_indices,
    glmrt_device_buffer_t topk_scores, glmrt_device_buffer_t topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k);
glmrt_status_t glmrt_cuda_graph_update_silu_gated_mlp_rows_bf16_down_stride_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t x,
    glmrt_device_buffer_t gate_weight, glmrt_device_buffer_t up_weight,
    glmrt_device_buffer_t down_weight, glmrt_device_buffer_t out, size_t rows, size_t hidden,
    size_t intermediate, size_t down_stride);
glmrt_status_t glmrt_cuda_graph_update_residual_add_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t residual, glmrt_device_buffer_t delta, glmrt_device_buffer_t out,
    size_t count);
glmrt_status_t glmrt_cuda_graph_update_residual_add_f32_delta_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t residual, glmrt_device_buffer_t delta_f32, glmrt_device_buffer_t out,
    size_t count);
glmrt_status_t glmrt_cuda_graph_update_residual_add_shared_f32_delta_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t residual, glmrt_device_buffer_t shared_delta,
    glmrt_device_buffer_t routed_delta_f32, glmrt_device_buffer_t out, size_t count);
glmrt_status_t glmrt_cuda_graph_update_causal_attention_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index, glmrt_device_buffer_t q,
    glmrt_device_buffer_t k, glmrt_device_buffer_t v, glmrt_device_buffer_t out, size_t rows,
    size_t heads, size_t qk_dim, size_t v_dim, float scale);
glmrt_status_t glmrt_cuda_graph_update_rope_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t input, glmrt_device_buffer_t positions, glmrt_device_buffer_t out,
    size_t rows, size_t heads, size_t rotary_dim, float theta);
glmrt_status_t glmrt_cuda_graph_update_mla_rope_attention_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t q_nope, glmrt_device_buffer_t q_rope,
    glmrt_device_buffer_t k_nope, glmrt_device_buffer_t k_rope, glmrt_device_buffer_t v,
    glmrt_device_buffer_t out, size_t rows, size_t heads, size_t nope_dim, size_t rope_dim,
    size_t v_dim, float scale);
glmrt_status_t glmrt_cuda_graph_update_mla_rope_attention_bf16_suffix_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t q_nope, glmrt_device_buffer_t q_rope,
    glmrt_device_buffer_t k_nope, glmrt_device_buffer_t k_rope, glmrt_device_buffer_t v,
    glmrt_device_buffer_t out, size_t rows, size_t query_row_offset, size_t query_rows,
    size_t heads, size_t nope_dim, size_t rope_dim, size_t v_dim, float scale);
glmrt_status_t glmrt_cuda_graph_update_mla_kv_cache_unpack_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t payload, glmrt_device_buffer_t kv_latent,
    glmrt_device_buffer_t k_rope, glmrt_device_buffer_t dsa_key, size_t rows,
    size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim, size_t payload_stride_bytes);
glmrt_status_t glmrt_cuda_graph_update_mla_kv_projected_split_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t projected, glmrt_device_buffer_t k_nope, glmrt_device_buffer_t v,
    size_t rows, size_t heads, size_t nope_dim, size_t v_dim);
glmrt_status_t glmrt_cuda_graph_update_f32_to_bf16_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t src, glmrt_device_buffer_t dst, size_t count);
glmrt_status_t glmrt_cuda_graph_update_scatter_add_rows_bf16_to_f32_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t src, glmrt_device_buffer_t row_indices, glmrt_device_buffer_t dst,
    size_t dst_rows, size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_graph_update_kv_cache_write_bytes_node(
    void* cuda_graph, void* cuda_graph_exec, size_t kernel_node_index,
    glmrt_device_buffer_t src, glmrt_device_buffer_t cache, size_t cache_offset_bytes,
    size_t bytes);
glmrt_status_t glmrt_copy_h2d(glmrt_device_buffer_t dst, const void* src, size_t bytes);
glmrt_status_t glmrt_copy_d2h(void* dst, glmrt_device_buffer_t src, size_t bytes);
glmrt_status_t glmrt_copy_d2d(glmrt_device_buffer_t dst, glmrt_device_buffer_t src, size_t bytes);
glmrt_status_t glmrt_copy_h2d_async(glmrt_device_buffer_t dst, const void* src, size_t bytes,
                                    void* cuda_stream);
glmrt_status_t glmrt_copy_d2h_async(void* dst, glmrt_device_buffer_t src, size_t bytes,
                                    void* cuda_stream);
glmrt_status_t glmrt_copy_d2d_async(glmrt_device_buffer_t dst, glmrt_device_buffer_t src,
                                    size_t bytes, void* cuda_stream);
glmrt_status_t glmrt_copy_d2d_2d_async(glmrt_device_buffer_t dst, size_t dst_pitch_bytes,
                                       glmrt_device_buffer_t src, size_t src_pitch_bytes,
                                       size_t width_bytes, size_t rows, void* cuda_stream);
glmrt_status_t glmrt_last_error(char* out, size_t out_len);
glmrt_status_t glmrt_nccl_unique_id_bytes(size_t* out_bytes);
glmrt_status_t glmrt_nccl_get_unique_id(void* out, size_t out_bytes);
glmrt_status_t glmrt_nccl_comm_init_rank(const void* unique_id, size_t unique_id_bytes,
                                         int world_size, int rank, void** out_handle);
glmrt_status_t glmrt_nccl_gather_u8_async(void* handle, glmrt_device_buffer_t send,
                                           glmrt_device_buffer_t recv, size_t bytes, int root,
                                           void* cuda_stream);
glmrt_status_t glmrt_nccl_row_all_to_all_u8_async(
    void* handle, glmrt_device_buffer_t send, glmrt_device_buffer_t recv, size_t rows,
    size_t row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_nccl_all_reduce_bf16_async(void* handle, glmrt_device_buffer_t send,
                                                glmrt_device_buffer_t recv, size_t values,
                                                void* cuda_stream);
glmrt_status_t glmrt_nccl_reduce_bf16_async(void* handle, glmrt_device_buffer_t send,
                                            glmrt_device_buffer_t recv, size_t values, int root,
                                            void* cuda_stream);
glmrt_status_t glmrt_nccl_comm_destroy(void* handle);
glmrt_status_t glmrt_rdma_device_info(glmrt_rdma_device_info_t* out);
glmrt_status_t glmrt_rdma_plan_host_buffer_registration(
    const void* ptr, size_t bytes, size_t alignment, glmrt_rdma_host_buffer_plan_t* out);
glmrt_status_t glmrt_rdma_register_host_buffer_probe(void* ptr, size_t bytes,
                                                     glmrt_rdma_register_probe_t* out);
glmrt_status_t glmrt_rdma_create_rc_qp_probe(uint32_t port_num, uint32_t send_wr,
                                             uint32_t recv_wr, uint32_t max_sge,
                                             glmrt_rdma_rc_qp_probe_t* out);
glmrt_status_t glmrt_rdma_rc_send_recv_loopback_probe(uint32_t port_num, size_t bytes,
                                                      glmrt_rdma_rc_send_recv_probe_t* out);
glmrt_status_t glmrt_rdma_rc_protocol_v2_loopback_probe(
    uint32_t port_num, const void* request_frame, size_t request_bytes,
    const void* response_frame, size_t response_bytes,
    glmrt_rdma_rc_protocol_v2_loopback_probe_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_create(
    uint32_t port_num, uint32_t local_psn, size_t send_frame_bytes, size_t recv_frame_bytes,
    size_t send_registered_span_bytes, size_t recv_registered_span_bytes, uint32_t max_send_wr,
    uint32_t max_recv_wr, uint32_t max_sge, glmrt_rdma_rc_endpoint_info_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_create_with_buffer_flags(
    uint32_t port_num, uint32_t local_psn, size_t send_frame_bytes, size_t recv_frame_bytes,
    size_t send_registered_span_bytes, size_t recv_registered_span_bytes, uint32_t max_send_wr,
    uint32_t max_recv_wr, uint32_t max_sge, uint64_t host_buffer_flags,
    glmrt_rdma_rc_endpoint_info_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_create_on_device_with_buffer_flags(
    const char* device_name, uint32_t port_num, uint32_t local_psn,
    size_t send_frame_bytes, size_t recv_frame_bytes,
    size_t send_registered_span_bytes, size_t recv_registered_span_bytes,
    uint32_t max_send_wr, uint32_t max_recv_wr, uint32_t max_sge,
    uint64_t host_buffer_flags, glmrt_rdma_rc_endpoint_info_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_buffer_view(
    void* handle, int receive_buffer, glmrt_rdma_rc_endpoint_buffer_view_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_connect(void* handle, uint32_t remote_qp_num,
                                               uint32_t remote_psn, uint32_t remote_lid,
                                               const char* remote_gid_hex);
glmrt_status_t glmrt_rdma_rc_endpoint_post_recv(void* handle, size_t bytes, uint64_t wr_id);
glmrt_status_t glmrt_rdma_rc_endpoint_post_recv_at(void* handle, size_t offset_bytes,
                                                   size_t bytes, uint64_t wr_id);
glmrt_status_t glmrt_rdma_rc_endpoint_post_send_at(void* handle, size_t offset_bytes,
                                                   size_t bytes, uint64_t wr_id);
glmrt_status_t glmrt_rdma_rc_endpoint_send(void* handle, const void* frame, size_t bytes,
                                           uint64_t wr_id);
glmrt_status_t glmrt_rdma_rc_endpoint_send_at(void* handle, const void* frame,
                                              size_t offset_bytes, size_t bytes,
                                              uint64_t wr_id);
glmrt_status_t glmrt_rdma_rc_endpoint_send_parts_at(
    void* handle, const void* prefix, size_t prefix_bytes, const void* payload,
    size_t payload_bytes, size_t offset_bytes, uint64_t wr_id);
glmrt_status_t glmrt_rdma_rc_endpoint_poll(void* handle, uint32_t expected_send_completions,
                                           uint32_t expected_recv_completions,
                                           uint32_t max_poll_iterations,
                                           glmrt_rdma_rc_completion_stats_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_poll_with_timeout(
    void* handle, uint32_t expected_send_completions, uint32_t expected_recv_completions,
    uint32_t max_poll_iterations, uint32_t active_event_poll_timeout_ms,
    glmrt_rdma_rc_completion_stats_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_try_poll(
    void* handle, uint32_t max_send_completions, uint32_t max_recv_completions,
    glmrt_rdma_rc_completion_stats_t* out);
glmrt_status_t glmrt_rdma_rc_endpoint_copy_recv(void* handle, void* out, size_t out_bytes,
                                                 size_t bytes);
glmrt_status_t glmrt_rdma_rc_endpoint_copy_recv_at(void* handle, void* out, size_t out_bytes,
                                                   size_t offset_bytes, size_t bytes);
glmrt_status_t glmrt_rdma_rc_endpoint_destroy(void* handle);

glmrt_status_t glmrt_cuda_rmsnorm_f32(const float* x, const float* weight, float* out,
                                      int rows, int hidden, float eps);
glmrt_status_t glmrt_cuda_rmsnorm_f32_async(const float* x, const float* weight, float* out,
                                            int rows, int hidden, float eps, void* cuda_stream);
glmrt_status_t glmrt_cuda_rmsnorm_bf16(const uint16_t* x, const uint16_t* weight, uint16_t* out,
                                       int rows, int hidden, float eps);
glmrt_status_t glmrt_cuda_rmsnorm_bf16_async(const uint16_t* x, const uint16_t* weight,
                                             uint16_t* out, int rows, int hidden, float eps,
                                             void* cuda_stream);
/* Benchmark-only exact Q-A graph candidate; serving does not reference this symbol. */
glmrt_status_t glmrt_cuda_mla_scalar_qa_batched_norm_candidate_async(
    const uint16_t* hidden, const uint16_t* input_norm_weight,
    uint16_t* normalized_hidden, const uint16_t* q_a_weight,
    uint16_t* q_a_projected, const uint16_t* q_a_norm_weight,
    uint16_t* q_a_normalized, size_t rows, size_t hidden_dim,
    size_t q_lora_rank, float eps, void* cuda_stream);
glmrt_status_t glmrt_cuda_layernorm_affine_f32_bf16(const float* x, const uint16_t* weight,
                                                    const uint16_t* bias, float* out, int rows,
                                                    int hidden, float eps);
glmrt_status_t glmrt_cuda_layernorm_affine_f32_bf16_async(
    const float* x, const uint16_t* weight, const uint16_t* bias, float* out, int rows,
    int hidden, float eps, void* cuda_stream);
glmrt_status_t glmrt_cuda_layernorm_affine_bf16(const uint16_t* x, const uint16_t* weight,
                                                const uint16_t* bias, uint16_t* out, int rows,
                                                int hidden, float eps);
glmrt_status_t glmrt_cuda_layernorm_affine_bf16_async(
    const uint16_t* x, const uint16_t* weight, const uint16_t* bias, uint16_t* out, int rows,
    int hidden, float eps, void* cuda_stream);
glmrt_status_t glmrt_cuda_silu_gated_mlp_f32(const float* x, const float* gate_weight,
                                             const float* up_weight, const float* down_weight,
                                             float* out, int hidden, int intermediate);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_f32(
    const float* x, const float* gate_weight, const float* up_weight, const float* down_weight,
    float* out, size_t rows, size_t hidden, size_t intermediate);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_f32_async(
    const float* x, const float* gate_weight, const float* up_weight, const float* down_weight,
    float* out, size_t rows, size_t hidden, size_t intermediate, void* cuda_stream);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_async(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_silu_mul_bf16_async(const uint16_t* gate_up, uint16_t* out,
                                              size_t rows, size_t intermediate,
                                              void* cuda_stream);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    size_t down_stride);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_async(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden, size_t intermediate,
    size_t down_stride, void* cuda_stream);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, float* activation_workspace, uint16_t* out, size_t rows,
    size_t hidden, size_t intermediate, size_t down_stride);
glmrt_status_t glmrt_cuda_silu_gated_mlp_rows_bf16_down_stride_staged_async(
    const uint16_t* x, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, float* activation_workspace, uint16_t* out, size_t rows,
    size_t hidden, size_t intermediate, size_t down_stride, void* cuda_stream);
glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    float* activation_workspace, float* accumulator, size_t rows, size_t routes,
    size_t hidden_dim, size_t hidden_row_stride, size_t intermediate, size_t output_dim,
    size_t down_weight_row_stride_bytes, size_t down_scale_row_stride_bytes,
    float gate_scale_2, float up_scale_2, float down_scale_2);
glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_grouped_staged_accumulate_f32_async(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const uint8_t* gate_weight, const uint8_t* gate_scale, const uint8_t* up_weight,
    const uint8_t* up_scale, const uint8_t* down_weight, const uint8_t* down_scale,
    float* activation_workspace, float* accumulator, size_t rows, size_t routes,
    size_t hidden_dim, size_t hidden_row_stride, size_t intermediate, size_t output_dim,
    size_t down_weight_row_stride_bytes, size_t down_scale_row_stride_bytes,
    float gate_scale_2, float up_scale_2, float down_scale_2, void* cuda_stream);
glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    float* accumulator, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t max_intermediate, size_t output_dim);
glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_accumulate_f32_async(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    float* accumulator, size_t rows, size_t routes, size_t hidden_dim,
    size_t hidden_row_stride, size_t max_intermediate, size_t output_dim, void* cuda_stream);
glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    uint16_t* out, size_t rows, size_t routes, size_t hidden_dim, size_t hidden_row_stride,
    size_t max_intermediate, size_t output_dim);
glmrt_status_t glmrt_cuda_nvfp4_silu_gated_mlp_route_bf16_batched_staged_single_row_bf16_async(
    const uint16_t* hidden, const uint32_t* row_indices, const float* route_weights,
    const glmrt_nvfp4_route_batched_metadata_t* route_metadata, float* activation_workspace,
    uint16_t* out, size_t rows, size_t routes, size_t hidden_dim, size_t hidden_row_stride,
    size_t max_intermediate, size_t output_dim, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_aot_available(int* out_available);
glmrt_status_t glmrt_cuda_b12x_spark_aot_init(void);
glmrt_status_t glmrt_cuda_b12x_quantize_bf16_nvfp4_row_payload_async(
    glmrt_device_buffer_t input, glmrt_device_buffer_t payload, size_t rows, size_t hidden_dim,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_w4a16_pack_weight_async(
    glmrt_device_buffer_t source, glmrt_device_buffer_t destination, size_t size_k,
    size_t size_n, size_t row_rotation, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_w4a16_pack_weight_strided_async(
    glmrt_device_buffer_t source, glmrt_device_buffer_t destination, size_t size_k,
    size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_w4a16_pack_scale_async(
    glmrt_device_buffer_t source, glmrt_device_buffer_t destination, size_t size_k,
    size_t size_n, size_t row_rotation, float scale_factor, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_w4a16_pack_scale_strided_async(
    glmrt_device_buffer_t source, glmrt_device_buffer_t destination, size_t size_k,
    size_t source_size_k, size_t source_start_k, size_t size_n,
    size_t row_rotation, float scale_factor, void* cuda_stream);
glmrt_status_t glmrt_cuda_quantize_bf16_weight_nvfp4_async(
    glmrt_device_buffer_t input, glmrt_device_buffer_t packed,
    glmrt_device_buffer_t scales, size_t rows, size_t cols,
    float global_scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_gather_nvfp4_rows_bf16_async(
    glmrt_device_buffer_t payload, size_t source_rows, size_t source_row_stride_bytes,
    glmrt_device_buffer_t row_indices, glmrt_device_buffer_t output, size_t rows,
    size_t hidden_dim, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, void* cuda_stream);
/* Benchmark entry point for atomic top-k accumulation into one BF16 row. */
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_decode_m1_fused_sum_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_m1_parity_m2_8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, size_t rows, void* cuda_stream);
glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_m1_parity_grouped_m2_8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
/* Grouped fixed-order output with the selected wider FC2 tile. */
glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_m1_parity_grouped_wide_m2_8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
/* Benchmark-only grouped decode grid sweep; serving does not reference this symbol. */
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_decode_m1_nvfp4_grid_candidate_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    glmrt_device_buffer_t topk_ids, int grid_x, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_exl3_k3_topk8_nvfp4_async(
    const glmrt_b12x_spark_exl3_k3_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
/* Full-rotation sum accumulates in FP32 and stores contiguous BF16 rows. */
glmrt_status_t glmrt_cuda_b12x_spark_exl3_k3_topk8_nvfp4_bf16_async(
    const glmrt_b12x_spark_exl3_k3_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
glmrt_status_t
glmrt_cuda_b12x_spark_exl3_k3_topk8_nvfp4_capacity_candidate_async(
    const glmrt_b12x_spark_exl3_k3_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, size_t capacity_rows, void* cuda_stream);
/* Benchmark-only EXL3 capacity/grid sweep; serving does not reference this symbol. */
glmrt_status_t
glmrt_cuda_b12x_spark_exl3_k3_topk8_nvfp4_capacity_grid_candidate_async(
    const glmrt_b12x_spark_exl3_k3_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, size_t capacity_rows, int grid_x, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_exl3_k4_topk8_nvfp4_async(
    const glmrt_b12x_spark_exl3_k4_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_exl3_k4_topk8_nvfp4_bf16_async(
    const glmrt_b12x_spark_exl3_k4_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, void* cuda_stream);
glmrt_status_t
glmrt_cuda_b12x_spark_exl3_k4_topk8_nvfp4_capacity_candidate_async(
    const glmrt_b12x_spark_exl3_k4_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, size_t capacity_rows, void* cuda_stream);
/* Benchmark-only EXL3 K4 capacity/grid sweep; serving does not reference this symbol. */
glmrt_status_t
glmrt_cuda_b12x_spark_exl3_k4_topk8_nvfp4_capacity_grid_candidate_async(
    const glmrt_b12x_spark_exl3_k4_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, size_t capacity_rows, int grid_x, void* cuda_stream);
/* Benchmark-only packed-prefill grid sweep; serving does not reference this symbol. */
glmrt_status_t
glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_grid_candidate_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, int grid_x, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_prefill_topk8_nvfp4_fp8_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers,
    glmrt_device_buffer_t input_payload, size_t input_payload_stride_bytes,
    size_t rows, glmrt_device_buffer_t output_fp8,
    size_t output_fp8_row_stride_bytes, void* cuda_stream);
/* Benchmarkable response postprocessing used by the fused FP8 serving candidate. */
glmrt_status_t glmrt_cuda_b12x_spark_sum_topk8_bf16_async(
    glmrt_device_buffer_t routed_bf16, glmrt_device_buffer_t output_bf16,
    size_t rows, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_sum_topk8_bf16_to_fp8_async(
    glmrt_device_buffer_t routed_bf16, glmrt_device_buffer_t output_fp8,
    size_t rows, size_t output_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_top1_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t rows,
    size_t capacity_rows, uint32_t expert_id, void* cuda_stream);
/* Benchmark-only grid sweep; serving does not reference this symbol. */
glmrt_status_t glmrt_cuda_b12x_spark_w4a16_top1_grid_candidate_async(
    const glmrt_b12x_spark_w4a16_moe_buffers_t* buffers, size_t rows,
    size_t capacity_rows, uint32_t expert_id, int grid_x, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_coordinator_aot_available(int* out_available);
glmrt_status_t glmrt_cuda_b12x_coordinator_aot_init(void);
glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_quantize_pack_weight_async(
    glmrt_device_buffer_t input_bf16, glmrt_device_buffer_t payload_scratch,
    glmrt_device_buffer_t packed_weight, glmrt_device_buffer_t packed_scale,
    glmrt_device_buffer_t global_scale, size_t size_k, size_t size_n,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_initialize_launch_buffers_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_q_b_m8_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t active_rows,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_q_b_m1_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream);
glmrt_status_t glmrt_cuda_b12x_coordinator_w4a16_o_proj_m1_tn64_candidate_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, void* cuda_stream);
glmrt_status_t glmrt_cuda_residual_add_f32(const float* residual, const float* delta,
                                           float* out, size_t count);
glmrt_status_t glmrt_cuda_residual_add_f32_async(const float* residual, const float* delta,
                                                 float* out, size_t count, void* cuda_stream);
glmrt_status_t glmrt_cuda_residual_add_bf16(const uint16_t* residual, const uint16_t* delta,
                                            uint16_t* out, size_t count);
glmrt_status_t glmrt_cuda_residual_add_bf16_async(const uint16_t* residual, const uint16_t* delta,
                                                  uint16_t* out, size_t count,
                                                  void* cuda_stream);
glmrt_status_t glmrt_cuda_residual_add_f32_delta_bf16(
    const uint16_t* residual, const float* delta_f32, uint16_t* out, size_t count);
glmrt_status_t glmrt_cuda_residual_add_f32_delta_bf16_async(
    const uint16_t* residual, const float* delta_f32, uint16_t* out, size_t count,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_residual_add_shared_f32_delta_bf16(
    const uint16_t* residual, const uint16_t* shared_delta, const float* routed_delta_f32,
    uint16_t* out, size_t count);
glmrt_status_t glmrt_cuda_residual_add_shared_f32_delta_bf16_async(
    const uint16_t* residual, const uint16_t* shared_delta, const float* routed_delta_f32,
    uint16_t* out, size_t count, void* cuda_stream);
glmrt_status_t glmrt_cuda_residual_add_shared_fp8_e4m3_row_scaled_bf16_async(
    const uint16_t* residual, const uint16_t* shared_delta, const uint8_t* routed_delta_fp8,
    uint16_t* out, size_t count, void* cuda_stream);
glmrt_status_t glmrt_cuda_fp8_decode_combine_residual_async(
    const uint16_t* residual, const uint16_t* shared_delta, const uint8_t* partials,
    size_t partial_row_stride_bytes, uint16_t* output, size_t partial_rows,
    size_t row_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_scheduler_mlp_delta_bf16(
    const uint16_t* hidden, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden_dim);
glmrt_status_t glmrt_cuda_scheduler_mlp_delta_bf16_async(
    const uint16_t* hidden, const uint16_t* gate_weight, const uint16_t* up_weight,
    const uint16_t* down_weight, uint16_t* out, size_t rows, size_t hidden_dim,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_summarize_bf16(const uint16_t* input, size_t count,
                                         glmrt_bf16_summary_t* out);
glmrt_status_t glmrt_cuda_summarize_bf16_async(const uint16_t* input, size_t count,
                                               glmrt_bf16_summary_t* out_device,
                                               void* cuda_stream);
glmrt_status_t glmrt_cuda_zero_f32(float* dst, size_t count);
glmrt_status_t glmrt_cuda_zero_f32_async(float* dst, size_t count, void* cuda_stream);
glmrt_status_t glmrt_cuda_zero_bytes(void* dst, size_t bytes);
glmrt_status_t glmrt_cuda_zero_bytes_async(void* dst, size_t bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_f32_to_bf16(const float* src, uint16_t* dst, size_t count);
glmrt_status_t glmrt_cuda_f32_to_bf16_async(const float* src, uint16_t* dst, size_t count,
                                            void* cuda_stream);
glmrt_status_t glmrt_cuda_gather_rows_f32(const float* src, const uint32_t* row_indices,
                                          float* dst, size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_gather_rows_f32_async(const float* src, const uint32_t* row_indices,
                                                float* dst, size_t rows, size_t row_width,
                                                void* cuda_stream);
/* Benchmark-only fused BF16 response pack; serving does not reference it. */
glmrt_status_t glmrt_cuda_gather_rows_f32_to_bf16_candidate_async(
    const float* src, const uint32_t* row_indices, uint16_t* dst, size_t rows,
    size_t row_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes);
glmrt_status_t glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream);
/* Benchmark alias for the register-cached 6144-wide production pack. */
glmrt_status_t glmrt_cuda_gather_rows_f32_to_fp8_e4m3_row_scaled_register_candidate_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_bf16_rows_to_fp8_e4m3_row_scaled_async(
    const uint16_t* src, uint8_t* dst, size_t rows, size_t row_width,
    size_t dst_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_combine_fp8_e4m3_row_scaled_to_fp8_async(
    const float* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_combine_bf16_fp8_e4m3_row_scaled_to_fp8_async(
    const uint16_t* local, const uint8_t* peers, size_t peer_payload_stride_bytes,
    size_t peer_count, size_t peer_row_stride_bytes, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes);
glmrt_status_t glmrt_cuda_gather_rows_f32_to_nvfp4_e2m1_fp8_e4m3_async(
    const float* src, const uint32_t* row_indices, uint8_t* dst, size_t rows,
    size_t row_width, size_t dst_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_gather_rows_bf16(const uint16_t* src, const uint32_t* row_indices,
                                           uint16_t* dst, size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_gather_rows_bf16_async(const uint16_t* src,
                                                 const uint32_t* row_indices, uint16_t* dst,
                                                 size_t rows, size_t row_width,
                                                 void* cuda_stream);
glmrt_status_t glmrt_cuda_copy_row_prefix_bf16(
    const uint16_t* src, uint16_t* dst, size_t rows, size_t src_row_width,
    size_t dst_row_width, size_t prefix_width, size_t src_row_offset);
glmrt_status_t glmrt_cuda_copy_row_prefix_bf16_async(
    const uint16_t* src, uint16_t* dst, size_t rows, size_t src_row_width,
    size_t dst_row_width, size_t prefix_width, size_t src_row_offset, void* cuda_stream);
glmrt_status_t glmrt_cuda_scatter_add_rows_f32(const float* src, const uint32_t* row_indices,
                                               float* dst, size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_scatter_add_rows_f32_async(const float* src,
                                                     const uint32_t* row_indices, float* dst,
                                                     size_t rows, size_t row_width,
                                                     void* cuda_stream);
glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_to_f32(const uint16_t* src,
                                                       const uint32_t* row_indices, float* dst,
                                                       size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_to_f32_async(
    const uint16_t* src, const uint32_t* row_indices, float* dst, size_t rows, size_t row_width,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_scatter_add_rows_fp8_e4m3_row_scaled_to_f32_async(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_scatter_add_rows_nvfp4_e2m1_fp8_e4m3_to_f32_async(
    const uint8_t* src, size_t src_row_stride_bytes, const uint32_t* row_indices, float* dst,
    size_t rows, size_t row_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_reduce_route_shards_to_f32(
    const glmrt_route_shard_reduction_buffers_t* buffers, size_t rows, size_t row_width,
    size_t peer_row_stride_bytes, uint32_t local_dtype, uint32_t peer_dtype,
    uint32_t peer_count);
glmrt_status_t glmrt_cuda_reduce_route_shards_to_f32_async(
    const glmrt_route_shard_reduction_buffers_t* buffers, size_t rows, size_t row_width,
    size_t peer_row_stride_bytes, uint32_t local_dtype, uint32_t peer_dtype,
    uint32_t peer_count, void* cuda_stream);
glmrt_status_t glmrt_cuda_reduce_route_shards_bf16_fp8_to_fp8_rail_candidate_async(
    const glmrt_route_shard_fp8_rail_reduction_buffers_t* buffers, size_t rows,
    size_t rail0_rows, size_t row_width, size_t peer_row_stride_bytes,
    size_t output_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_weighted_to_f32(
    const uint16_t* src, const uint32_t* row_indices, const float* row_weights, float* dst,
    size_t rows, size_t row_width);
glmrt_status_t glmrt_cuda_scatter_add_rows_bf16_weighted_to_f32_async(
    const uint16_t* src, const uint32_t* row_indices, const float* row_weights, float* dst,
    size_t rows, size_t row_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_kv_cache_write_bytes(const uint8_t* src, uint8_t* cache,
                                               size_t cache_offset_bytes, size_t bytes);
glmrt_status_t glmrt_cuda_kv_cache_write_bytes_async(const uint8_t* src, uint8_t* cache,
                                                     size_t cache_offset_bytes, size_t bytes,
                                                     void* cuda_stream);
glmrt_status_t glmrt_cuda_kv_cache_read_bytes(const uint8_t* cache, uint8_t* dst,
                                              size_t cache_offset_bytes, size_t bytes);
glmrt_status_t glmrt_cuda_kv_cache_read_bytes_async(const uint8_t* cache, uint8_t* dst,
                                                    size_t cache_offset_bytes, size_t bytes,
                                                    void* cuda_stream);
glmrt_status_t glmrt_cuda_kv_cache_write_blocks(
    const uint8_t* src, uint8_t* cache, const uint64_t* src_offsets,
    const uint64_t* cache_offsets, const uint64_t* block_bytes, size_t block_count);
glmrt_status_t glmrt_cuda_kv_cache_write_blocks_async(
    const uint8_t* src, uint8_t* cache, const uint64_t* src_offsets,
    const uint64_t* cache_offsets, const uint64_t* block_bytes, size_t block_count,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_kv_cache_read_blocks(
    const uint8_t* cache, uint8_t* dst, const uint64_t* cache_offsets,
    const uint64_t* dst_offsets, const uint64_t* block_bytes, size_t block_count);
glmrt_status_t glmrt_cuda_kv_cache_read_blocks_async(
    const uint8_t* cache, uint8_t* dst, const uint64_t* cache_offsets,
    const uint64_t* dst_offsets, const uint64_t* block_bytes, size_t block_count,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_cache_unpack_bf16(
    const uint8_t* payload, uint16_t* kv_latent, uint16_t* k_rope, uint16_t* dsa_key,
    size_t rows, size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim,
    size_t payload_stride_bytes);
glmrt_status_t glmrt_cuda_mla_kv_cache_unpack_bf16_async(
    const uint8_t* payload, uint16_t* kv_latent, uint16_t* k_rope, uint16_t* dsa_key,
    size_t rows, size_t kv_lora_rank, size_t rope_dim, size_t dsa_dim,
    size_t payload_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_projected_split_bf16(
    const uint16_t* projected, uint16_t* k_nope, uint16_t* v, size_t rows, size_t heads,
    size_t nope_dim, size_t v_dim);
glmrt_status_t glmrt_cuda_mla_kv_projected_split_bf16_async(
    const uint16_t* projected, uint16_t* k_nope, uint16_t* v, size_t rows, size_t heads,
    size_t nope_dim, size_t v_dim, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_prepare_bf16(
    const uint16_t* projected, const uint32_t* positions, const uint16_t* norm_weight,
    uint16_t* prepared, size_t rows, size_t projected_stride_bytes,
    size_t prepared_stride_bytes, float eps, float theta);
glmrt_status_t glmrt_cuda_mla_kv_prepare_bf16_async(
    const uint16_t* projected, const uint32_t* positions, const uint16_t* norm_weight,
    uint16_t* prepared, size_t rows, size_t projected_stride_bytes,
    size_t prepared_stride_bytes, float eps, float theta, void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_index_k_pack_b12x(
    const uint16_t* normalized_k, const uint32_t* positions,
    const uint32_t* cache_slots, uint8_t* index_k_cache, size_t rows,
    size_t cache_tokens, size_t normalized_stride_bytes, float theta);
glmrt_status_t glmrt_cuda_glm_dsa_index_k_pack_b12x_async(
    const uint16_t* normalized_k, const uint32_t* positions,
    const uint32_t* cache_slots, uint8_t* index_k_cache, size_t rows,
    size_t cache_tokens, size_t normalized_stride_bytes, float theta,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_query_prepare_b12x(
    const uint16_t* query, const uint16_t* raw_weights,
    const uint32_t* positions, uint8_t* query_fp8, float* adjusted_weights,
    size_t rows, size_t query_stride_bytes, size_t raw_weights_stride_bytes,
    size_t query_fp8_stride_bytes, size_t adjusted_weights_stride_bytes,
    float theta, float score_scale);
glmrt_status_t glmrt_cuda_glm_dsa_query_prepare_b12x_async(
    const uint16_t* query, const uint16_t* raw_weights,
    const uint32_t* positions, uint8_t* query_fp8, float* adjusted_weights,
    size_t rows, size_t query_stride_bytes, size_t raw_weights_stride_bytes,
    size_t query_fp8_stride_bytes, size_t adjusted_weights_stride_bytes,
    float theta, float score_scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_transpose_rows_heads_bf16(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width);
glmrt_status_t glmrt_cuda_transpose_rows_heads_bf16_async(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width, void* cuda_stream);
glmrt_status_t glmrt_cuda_transpose_heads_rows_bf16(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width);
glmrt_status_t glmrt_cuda_transpose_heads_rows_bf16_async(
    const uint16_t* input, uint16_t* output, size_t rows, size_t heads,
    size_t width, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_compose_absorbed_query_bf16(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width);
glmrt_status_t glmrt_cuda_mla_compose_absorbed_query_bf16_async(
    const uint16_t* latent_heads_rows, const uint16_t* rope_rows_heads,
    uint16_t* output_rows_heads, size_t rows, size_t heads,
    size_t latent_width, size_t rope_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_page_table_init(
    int32_t* page_table, size_t query_rows, size_t page_table_width);
glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_async(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_base(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    size_t base_offset);
glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_base_async(
    int32_t* page_table, size_t query_rows, size_t page_table_width,
    size_t base_offset, void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_offsets(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width);
glmrt_status_t glmrt_cuda_glm_dsa_page_table_init_offsets_async(
    int32_t* page_table, const int32_t* row_offsets, size_t query_rows,
    size_t page_table_width, void* cuda_stream);
glmrt_status_t glmrt_cuda_target_kv_page_table_expand_indices(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens);
glmrt_status_t glmrt_cuda_target_kv_page_table_expand_indices_async(
    int32_t* output_indices, const uint32_t* physical_pages,
    size_t query_rows, size_t output_width, size_t active_tokens,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_prefill_metadata(
    int32_t* cache_seqlens, int32_t* topk_lengths, int32_t* active_width,
    size_t bucket_rows, size_t active_rows, size_t prefix_rows,
    size_t total_rows, size_t topk);
glmrt_status_t glmrt_cuda_glm_dsa_prefill_metadata_async(
    int32_t* cache_seqlens, int32_t* topk_lengths, int32_t* active_width,
    size_t bucket_rows, size_t active_rows, size_t prefix_rows,
    size_t total_rows, size_t topk, void* cuda_stream);
glmrt_status_t glmrt_cuda_glm_dsa_sort_selected_indices_async(
    int32_t* selected_indices, size_t rows, size_t width,
    void* cuda_stream);
/* Benchmark-only cross-layer RoPE reuse candidate; serving does not reference it. */
glmrt_status_t glmrt_cuda_mla_rope_factors_f32_candidate_async(
    const uint32_t* positions, float* factors, size_t rows, float theta,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_prepare_bf16_precomputed_rope_candidate_async(
    const uint16_t* projected, const float* rope_factors,
    const uint16_t* norm_weight, uint16_t* prepared, size_t rows,
    size_t projected_stride_bytes, size_t prepared_stride_bytes, float eps,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_pack_fp8_ds_mla(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes);
glmrt_status_t glmrt_cuda_mla_kv_pack_fp8_ds_mla_async(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_unpack_fp8_ds_mla(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes);
glmrt_status_t glmrt_cuda_mla_kv_unpack_fp8_ds_mla_async(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_pack_mxfp4_ds_mla(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes);
glmrt_status_t glmrt_cuda_mla_kv_pack_mxfp4_ds_mla_async(
    const uint16_t* projected, uint8_t* packed, size_t rows, size_t projected_stride_bytes,
    size_t packed_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes);
glmrt_status_t glmrt_cuda_mla_kv_unpack_mxfp4_ds_mla_async(
    const uint8_t* packed, uint16_t* projected, size_t rows, size_t packed_stride_bytes,
    size_t projected_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_router_topk_f32(const float* hidden, const float* router_weight,
                                          const float* correction_bias, uint32_t* topk_indices,
                                          float* topk_scores, float* topk_weights, size_t rows,
                                          size_t hidden_dim, size_t experts, size_t top_k);
glmrt_status_t glmrt_cuda_router_topk_f32_async(
    const float* hidden, const float* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream);
glmrt_status_t glmrt_cuda_router_topk_bf16(const uint16_t* hidden,
                                           const uint16_t* router_weight,
                                           const float* correction_bias, uint32_t* topk_indices,
                                           float* topk_scores, float* topk_weights, size_t rows,
                                           size_t hidden_dim, size_t experts, size_t top_k);
glmrt_status_t glmrt_cuda_router_topk_bf16_async(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    uint32_t* topk_indices, float* topk_scores, float* topk_weights, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream);
glmrt_status_t glmrt_cuda_router_topk_bf16_cub(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k);
glmrt_status_t glmrt_cuda_router_topk_bf16_cub_async(
    const uint16_t* hidden, const uint16_t* router_weight, const float* correction_bias,
    float* corrected_scores, float* sorted_corrected_scores, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* topk_indices, float* topk_scores,
    float* topk_weights, void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows,
    size_t hidden_dim, size_t experts, size_t top_k, void* cuda_stream);
glmrt_status_t glmrt_cuda_linear_f32(const float* input, const float* weight, const float* bias,
                                     float* output, size_t rows, size_t input_dim,
                                     size_t output_dim);
glmrt_status_t glmrt_cuda_linear_f32_async(const float* input, const float* weight,
                                           const float* bias, float* output, size_t rows,
                                           size_t input_dim, size_t output_dim,
                                           void* cuda_stream);
glmrt_status_t glmrt_cuda_linear_bf16(const uint16_t* input, const uint16_t* weight,
                                      const uint16_t* bias, uint16_t* output, size_t rows,
                                      size_t input_dim, size_t output_dim);
glmrt_status_t glmrt_cuda_linear_bf16_async(const uint16_t* input, const uint16_t* weight,
                                            const uint16_t* bias, uint16_t* output, size_t rows,
                                            size_t input_dim, size_t output_dim,
                                            void* cuda_stream);
glmrt_status_t glmrt_cuda_linear_bf16_cublas(const uint16_t* input, const uint16_t* weight,
                                             const uint16_t* bias, uint16_t* output, size_t rows,
                                             size_t input_dim, size_t output_dim);
glmrt_status_t glmrt_cuda_linear_bf16_cublas_async(
    const uint16_t* input, const uint16_t* weight, const uint16_t* bias, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim, void* cuda_stream);
// M=2..8 shared-weight BF16 projection selected from the recurrent M=1
// cuBLASLt plan. The first live launch self-qualifies bitwise parity against
// repeated M=1 cuBLAS calls and retains that exact fallback if the local
// driver/toolkit selects an incompatible plan.
glmrt_status_t glmrt_cuda_linear_bf16_m1_parity_batched_cublaslt_async(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t rows, size_t input_dim, size_t output_dim, void* cuda_stream);
// One-row GEMV over losslessly packed BF16 weights. Each 1,024-value tile uses
// one sign/mantissa byte per value, two four-bit exponent codes per byte, and a
// metadata row whose header is base | (escape_count << 8). Remaining metadata
// words encode escape_position | (exact_exponent << 16).
glmrt_status_t glmrt_cuda_linear_lossless_bf16_m1_async(
    const uint16_t* input, const uint8_t* low, const uint8_t* codes,
    const uint32_t* metadata, uint16_t* output, size_t input_dim,
    size_t output_dim, size_t metadata_stride_words, void* cuda_stream);
// One-time per-output-row/per-256-value symmetric W8 quantizer. k_major=0
// emits weight[N,K] and scale[N,K/256] for the M=1 SIMT kernel; k_major=1 emits
// weight[K,N] and scale[K/256,N] for multirow direct-dequant paths.
glmrt_status_t glmrt_cuda_quantize_bf16_w8a16_group256_async(
    const uint16_t* source, int8_t* weight, float* scales, size_t input_dim,
    size_t output_dim, int k_major, void* cuda_stream);
// Quantizes directly into the lane-major K16/N64 fragment order shared by the
// packed M=1 and tensor-core O-projection kernels. Scales remain [K/256,N].
glmrt_status_t glmrt_cuda_quantize_bf16_w8a16_group256_packed_async(
    const uint16_t* source, int8_t* weight, float* scales, size_t input_dim,
    size_t output_dim, void* cuda_stream);
// Expands a row-major E4M3 matrix with [ceil(N/128),ceil(K/128)] FP32
// inverse scales into row-major BF16. This is a startup-only conversion for
// official GLM-5.3 block-FP8 coordinator tensors.
glmrt_status_t glmrt_cuda_dequantize_block_fp8_e4m3_bf16_async(
    const uint8_t* source, const float* scales, uint16_t* output,
    size_t input_dim, size_t output_dim, void* cuda_stream);
// Expands K-major W8/group-major scales into a row-major BF16 matrix. The
// caller owns one projection-sized scratch allocation; no per-layer BF16
// duplicate is required.
glmrt_status_t glmrt_cuda_dequantize_w8a16_group256_bf16_async(
    const int8_t* weight_k_major, const float* scales_group_major,
    uint16_t* weight_bf16, size_t input_dim, size_t output_dim,
    void* cuda_stream);
// Row-major W8/row-major FP32-scale M=1 projection. Variants 0..15 cover one,
// two, or four output rows per warp, four/eight warps per CTA, normal or
// non-coherent weight loads, and whole-input shared-memory staging.
glmrt_status_t glmrt_cuda_linear_w8a16_group256_m1_simt_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim, int variant,
    void* cuda_stream);
// M=2..8 projection that preserves the recurrent M=1 SIMT accumulation order
// for every row while sharing each W8 weight traversal across the row batch.
glmrt_status_t glmrt_cuda_linear_w8a16_group256_m1_parity_batched_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    void* cuda_stream);
// One-row projection over lane-major K16/N64 fragments stored as
// [K tile, N tile, lane, N16 warp, two int32 words]. The same bytes feed the
// packed tensor-core multirow path.
glmrt_status_t glmrt_cuda_linear_w8a16_group256_m1_warp_packed_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t input_dim, size_t output_dim,
    void* cuda_stream);
// M=2..8 packed projection that shares each weight read across rows while
// preserving the packed M=1 arithmetic independently for every row.
glmrt_status_t
glmrt_cuda_linear_w8a16_group256_m1_warp_packed_parity_batched_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    void* cuda_stream);
// Multirow BF16-I/O projection using dynamically quantized signed-int8
// activations and the row-major W8/group-256 resident used by M=1 decode.
glmrt_status_t glmrt_cuda_linear_w8a8_group256_wmma_async(
    const int8_t* input, const float* input_scales, const int8_t* weight,
    const float* weight_scales, uint16_t* output, size_t rows,
    size_t input_dim, size_t output_dim, void* cuda_stream);
// Benchmark/AOT bring-up entry: launch a row-major W8A16 Triton cubin through
// the CUDA driver on the caller's stream.  Production uses embedded cubins;
// this file-backed form verifies signatures and launch metadata first.
glmrt_status_t glmrt_cuda_linear_w8a16_group256_triton_file_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    const char* cubin_path, const char* kernel_name, size_t block_m,
    size_t block_n, size_t threads, size_t shared_bytes, void* cuda_stream);
// Preload all embedded row buckets for a projection before CUDA graph capture.
glmrt_status_t glmrt_cuda_preload_w8a16_group256_aot(
    size_t input_dim, size_t output_dim);
// Bucketed M=2..256 row-major W8A16 projection from embedded Triton cubins.
glmrt_status_t glmrt_cuda_linear_w8a16_group256_aot_async(
    const uint16_t* input, const int8_t* weight, const float* scales,
    uint16_t* output, size_t rows, size_t input_dim, size_t output_dim,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_w8a16_packed_o_aot_init(void);
glmrt_status_t glmrt_cuda_w8a16_packed_o_initialize_launch_buffers_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t rows,
    size_t block_m, void* cuda_stream);
glmrt_status_t glmrt_cuda_w8a16_packed_o_async(
    const glmrt_b12x_coordinator_w4a16_buffers_t* buffers, size_t rows,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_linear_bf16_strided_batched_cublas(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t weight_batch_stride,
    size_t output_batch_stride);
glmrt_status_t glmrt_cuda_linear_bf16_strided_batched_cublas_async(
    const uint16_t* input, const uint16_t* weight, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t weight_batch_stride,
    size_t output_batch_stride, void* cuda_stream);
glmrt_status_t glmrt_cuda_matmul_bf16_strided_batched_cublas_async(
    const uint16_t* input, const uint16_t* right, uint16_t* output,
    size_t batch_count, size_t rows, size_t input_dim, size_t output_dim,
    size_t input_batch_stride, size_t right_batch_stride,
    size_t output_batch_stride, void* cuda_stream);
glmrt_status_t glmrt_cuda_causal_attention_f32(const float* q, const float* k, const float* v,
                                               float* out, size_t rows, size_t heads,
                                               size_t qk_dim, size_t v_dim, float scale);
glmrt_status_t glmrt_cuda_causal_attention_f32_async(
    const float* q, const float* k, const float* v, float* out, size_t rows, size_t heads,
    size_t qk_dim, size_t v_dim, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_causal_attention_bf16(const uint16_t* q, const uint16_t* k,
                                                const uint16_t* v, uint16_t* out, size_t rows,
                                                size_t heads, size_t qk_dim, size_t v_dim,
                                                float scale);
glmrt_status_t glmrt_cuda_causal_attention_bf16_async(
    const uint16_t* q, const uint16_t* k, const uint16_t* v, uint16_t* out, size_t rows,
    size_t heads, size_t qk_dim, size_t v_dim, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_rope_f32(const float* input, const uint32_t* positions, float* out,
                                   size_t rows, size_t heads, size_t rotary_dim, float theta);
glmrt_status_t glmrt_cuda_rope_f32_async(const float* input, const uint32_t* positions,
                                         float* out, size_t rows, size_t heads,
                                         size_t rotary_dim, float theta, void* cuda_stream);
glmrt_status_t glmrt_cuda_rope_bf16(const uint16_t* input, const uint32_t* positions,
                                    uint16_t* out, size_t rows, size_t heads,
                                    size_t rotary_dim, float theta);
glmrt_status_t glmrt_cuda_rope_bf16_async(const uint16_t* input, const uint32_t* positions,
                                          uint16_t* out, size_t rows, size_t heads,
                                          size_t rotary_dim, float theta, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_rope_attention_bf16(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows, size_t heads,
    size_t nope_dim, size_t rope_dim, size_t v_dim, float scale);
glmrt_status_t glmrt_cuda_mla_rope_attention_bf16_async(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows, size_t heads,
    size_t nope_dim, size_t rope_dim, size_t v_dim, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_rope_attention_bf16_suffix(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows,
    size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale);
glmrt_status_t glmrt_cuda_mla_rope_attention_bf16_suffix_async(
    const uint16_t* q_nope, const uint16_t* q_rope, const uint16_t* k_nope,
    const uint16_t* k_rope, const uint16_t* v, uint16_t* out, size_t rows,
    size_t query_row_offset, size_t query_rows, size_t heads, size_t nope_dim,
    size_t rope_dim, size_t v_dim, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_compressed_attention_bf16(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_latent, const uint16_t* k_rope, uint16_t* out_latent,
    size_t rows, size_t heads, size_t rope_dim, size_t kv_lora_rank, float scale);
glmrt_status_t glmrt_cuda_mla_compressed_attention_bf16_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_latent, const uint16_t* k_rope, uint16_t* out_latent,
    size_t rows, size_t heads, size_t rope_dim, size_t kv_lora_rank, float scale,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_bf16(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, size_t rope_offset_bytes, float scale);
glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_bf16_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint16_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, size_t rope_offset_bytes, float scale,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_fp8(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale);
glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_fp8_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_mxfp4(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale);
glmrt_status_t glmrt_cuda_mla_compressed_attention_interleaved_mxfp4_async(
    const uint16_t* q_absorbed, const uint16_t* q_rope,
    const uint8_t* kv_payload, uint16_t* out_latent, size_t rows,
    size_t heads, size_t rope_dim, size_t kv_lora_rank,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_sparse_mla_nvfp4_async(
    const uint16_t* query, const uint8_t* kv_payload,
    const int32_t* selected_indices, const int32_t* topk_lengths,
    uint16_t* partial, float* partial_lse, uint16_t* output,
    float* output_lse, size_t query_rows, size_t heads, size_t topk,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_sparse_mla_bf16_async(
    const uint16_t* query, const uint8_t* kv_payload,
    const int32_t* selected_indices, const int32_t* topk_lengths,
    uint16_t* partial, float* partial_lse, uint16_t* output,
    float* output_lse, size_t query_rows, size_t heads, size_t topk,
    size_t kv_row_stride_bytes, float scale, void* cuda_stream);
glmrt_status_t glmrt_cuda_sparse_mla_bf16_gather_kv_async(
    const uint8_t* kv_payload, const int32_t* selected_indices,
    const int32_t* topk_lengths, uint16_t* gathered_k,
    uint16_t* gathered_v, size_t query_rows, size_t topk,
    size_t kv_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_sparse_mla_bf16_softmax_async(
    uint16_t* scores, const int32_t* topk_lengths, float* output_lse,
    size_t query_rows, size_t heads, size_t topk, float scale,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_sparse_mla_nvfp4_gather_fp8_async(
    const uint8_t* nvfp4_kv, const int32_t* selected_indices,
    const int32_t* topk_lengths, uint8_t* fp8_kv, int32_t* fp8_indices,
    size_t query_rows, size_t selected_index_stride, size_t staged_topk,
    size_t nvfp4_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_nvfp4_expand_fp8_paged_async(
    const uint8_t* nvfp4_kv, const uint32_t* physical_pages,
    const int32_t* active_rows, uint8_t* fp8_kv, size_t max_tokens,
    size_t page_size, size_t nvfp4_row_stride_bytes, void* cuda_stream);
glmrt_status_t glmrt_cuda_mla_merge_state_bf16(
    uint16_t* accumulator, float* accumulator_lse, const uint16_t* partial,
    const float* partial_lse, size_t heads, size_t kv_lora_rank);
glmrt_status_t glmrt_cuda_mla_merge_state_bf16_async(
    uint16_t* accumulator, float* accumulator_lse, const uint16_t* partial,
    const float* partial_lse, size_t heads, size_t kv_lora_rank, void* cuda_stream);
glmrt_status_t glmrt_cuda_packed_fp8_mla_exact_grouped_async(
    const void* q, const void* kv_cache, const void* indices, void* mid_out,
    void* mid_lse, const void* topk_length, void* output, void* out_lse,
    size_t num_tokens, size_t num_heads, size_t topk,
    size_t chunks_per_block, float sm_scale, size_t stride_kv_block,
    void* cuda_stream);
glmrt_status_t glmrt_cuda_embedding_lookup_f32(const float* embedding, const uint32_t* token_ids,
                                               float* out, size_t rows, size_t vocab,
                                               size_t hidden);
glmrt_status_t glmrt_cuda_embedding_lookup_f32_async(const float* embedding,
                                                     const uint32_t* token_ids, float* out,
                                                     size_t rows, size_t vocab, size_t hidden,
                                                     void* cuda_stream);
glmrt_status_t glmrt_cuda_embedding_lookup_bf16(const uint16_t* embedding,
                                                const uint32_t* token_ids, uint16_t* out,
                                                size_t rows, size_t vocab, size_t hidden);
glmrt_status_t glmrt_cuda_embedding_lookup_bf16_async(
    const uint16_t* embedding, const uint32_t* token_ids, uint16_t* out, size_t rows,
    size_t vocab, size_t hidden, void* cuda_stream);
glmrt_status_t glmrt_cuda_lm_head_argmax_bf16(const uint16_t* hidden, const uint16_t* lm_head,
                                              uint32_t* out_indices, float* out_scores,
                                              size_t rows, size_t hidden_dim, size_t vocab);
glmrt_status_t glmrt_cuda_lm_head_argmax_bf16_async(
    const uint16_t* hidden, const uint16_t* lm_head, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t hidden_dim, size_t vocab, void* cuda_stream);
glmrt_status_t glmrt_cuda_lm_head_sample_topk_topp_bf16(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p);
glmrt_status_t glmrt_cuda_lm_head_sample_topk_topp_bf16_async(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_indices, float* out_scores, size_t rows, size_t hidden_dim, size_t vocab,
    float temperature, size_t top_k, float top_p, void* cuda_stream);
glmrt_status_t glmrt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_argmax_indices, float* out_argmax_scores, uint32_t* out_sample_indices,
    float* out_sample_scores, float* logits_workspace, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p);
glmrt_status_t glmrt_cuda_lm_head_argmax_sample_topk_topp_bf16_staged_async(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    uint32_t* out_argmax_indices, float* out_argmax_scores, uint32_t* out_sample_indices,
    float* out_sample_scores, float* logits_workspace, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p, void* cuda_stream);
glmrt_status_t glmrt_cuda_lm_head_sample_topk_topp_bf16_cub(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    float* logits_workspace, float* sorted_logits, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* out_indices, float* out_scores,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p);
glmrt_status_t glmrt_cuda_lm_head_sample_topk_topp_bf16_cub_async(
    const uint16_t* hidden, const uint16_t* lm_head, const float* random_uniforms,
    float* logits_workspace, float* sorted_logits, uint32_t* unsorted_indices,
    uint32_t* sorted_indices, int* segment_offsets, uint32_t* out_indices, float* out_scores,
    void* cub_temp_storage, size_t cub_temp_storage_bytes, size_t rows, size_t hidden_dim,
    size_t vocab, float temperature, size_t top_k, float top_p, void* cuda_stream);
glmrt_status_t glmrt_cuda_logits_argmax_f32(const float* logits, uint32_t* out_indices,
                                            float* out_scores, size_t rows, size_t vocab);
glmrt_status_t glmrt_cuda_logits_argmax_f32_async(const float* logits, uint32_t* out_indices,
                                                  float* out_scores, size_t rows, size_t vocab,
                                                  void* cuda_stream);
glmrt_status_t glmrt_cuda_logits_sample_topk_topp_f32(
    const float* logits, const float* random_uniforms, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t vocab, float temperature, size_t top_k, float top_p);
glmrt_status_t glmrt_cuda_logits_sample_topk_topp_f32_async(
    const float* logits, const float* random_uniforms, uint32_t* out_indices, float* out_scores,
    size_t rows, size_t vocab, float temperature, size_t top_k, float top_p, void* cuda_stream);
glmrt_status_t glmrt_cuda_logits_sample_topk_topp_f32_cub(
    const float* logits, const float* random_uniforms, float* sorted_logits,
    uint32_t* unsorted_indices, uint32_t* sorted_indices, int* segment_offsets,
    uint32_t* out_indices, float* out_scores, void* cub_temp_storage,
    size_t cub_temp_storage_bytes, size_t rows, size_t vocab, float temperature, size_t top_k,
    float top_p);
glmrt_status_t glmrt_cuda_logits_sample_topk_topp_f32_cub_async(
    const float* logits, const float* random_uniforms, float* sorted_logits,
    uint32_t* unsorted_indices, uint32_t* sorted_indices, int* segment_offsets,
    uint32_t* out_indices, float* out_scores, void* cub_temp_storage,
    size_t cub_temp_storage_bytes, size_t rows, size_t vocab, float temperature, size_t top_k,
    float top_p, void* cuda_stream);
glmrt_status_t glmrt_cuda_pack_nibbles(const uint8_t* codes, uint8_t* packed, size_t count);
glmrt_status_t glmrt_cuda_unpack_nibbles(const uint8_t* packed, uint8_t* codes, size_t count);

#ifdef __cplusplus
}
#endif
