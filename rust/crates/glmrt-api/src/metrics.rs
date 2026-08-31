use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CompletionMetrics {
    pub queue_ms: f64,
    pub cache_load_ms: f64,
    pub prefill_ms: f64,
    pub time_to_first_token_ms: f64,
    pub decode_ms: f64,
    pub output_tokens: usize,
    pub prompt_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub reasoning_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tokens_per_sec: Option<f64>,
    pub transport_backend: &'static str,
    pub backend_mode: &'static str,
    pub prefill_chunk_count: usize,
    pub layerwave_prefill_rows: usize,
    pub layerwave_decode_rows: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub real_full: Option<RealFullDiagnosticMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RealFullDiagnosticMetrics {
    pub status: String,
    pub startup_diagnostic_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocker: Option<String>,
    pub failed_requirements: Vec<String>,
    pub scheduler_numeric_progression_passed: bool,
    pub scheduler_full_context_device_attention_complete: bool,
    pub scheduler_terminal_lm_head_sample_status: String,
    pub scheduler_terminal_lm_head_sample_passed: bool,
    pub scheduler_terminal_lm_head_uses_final_decode_device_hidden: bool,
    pub scheduler_terminal_lm_head_covers_full_vocabulary: bool,
    pub scheduler_terminal_lm_head_logits_evaluated: usize,
    pub scheduler_terminal_lm_head_vocab_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_terminal_lm_head_top_token_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_terminal_lm_head_sampled_token_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_terminal_lm_head_sample_top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_terminal_lm_head_sample_top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scheduler_terminal_lm_head_blocker: Option<String>,
    pub scheduler_sparse_tcp_dispatch_status: String,
    pub scheduler_sparse_tcp_dispatch_targets: usize,
    pub scheduler_sparse_tcp_dispatch_sparse_layers: usize,
    pub scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: usize,
    pub scheduler_sparse_tcp_dispatch_batches: usize,
    pub scheduler_sparse_tcp_dispatch_host_batches: usize,
    pub scheduler_sparse_tcp_dispatch_global_rows: usize,
    pub scheduler_sparse_tcp_dispatch_host_rows: usize,
    pub scheduler_sparse_tcp_dispatch_routes: usize,
    pub scheduler_sparse_tcp_dispatch_request_wire_bytes: usize,
    pub scheduler_sparse_tcp_dispatch_response_wire_bytes: usize,
    pub scheduler_sparse_tcp_dispatch_output_values: usize,
    pub scheduler_sparse_tcp_dispatch_output_finite_values: usize,
    pub scheduler_sparse_tcp_dispatch_output_nonzero_values: usize,
    pub scheduler_sparse_tcp_dispatch_output_checksum: f64,
    pub scheduler_sparse_tcp_dispatch_passed: bool,
    pub scheduler_sparse_tcp_dispatch_expected_real_executor_id: u64,
    pub scheduler_sparse_tcp_dispatch_response_executor_ids_observed: usize,
    pub scheduler_sparse_tcp_dispatch_real_executor_responses: usize,
    pub scheduler_sparse_tcp_dispatch_non_real_executor_responses: usize,
    pub scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: bool,
    pub scheduler_sparse_tcp_dispatch_consumed_by_residual: bool,
    pub request_scheduler_summary_runtime_reported: bool,
    pub request_prefill_tokens: usize,
    pub request_prefill_chunks: usize,
    pub request_kv_snapshot_restore_ms: f64,
    pub request_decode_budget: usize,
    pub request_mtp_verify_rows: usize,
    pub request_mtp_accepted_rows: usize,
    pub mtp_verify_cycles: usize,
    pub mtp_draft_tokens: usize,
    pub mtp_accepted_draft_tokens: usize,
    pub mtp_emitted_tokens_from_verify: usize,
    pub mtp_full_match_cycles: usize,
    pub mtp_total_verify_cycle_ms: f64,
    pub mtp_draft_lengths: Vec<usize>,
    pub mtp_accepted_draft_lengths: Vec<usize>,
    pub mtp_verify_cycle_ms: Vec<f64>,
    pub target_cycle_physical_m: Vec<usize>,
    pub target_cycle_ms: Vec<f64>,
    pub request_coordinator_graph_slots: usize,
    pub request_coordinator_graph_captured_graphs: usize,
    pub request_coordinator_graph_captures: usize,
    pub request_coordinator_graph_launches: usize,
    pub request_candidate_layerwaves: usize,
    pub request_layerwaves: usize,
    pub request_deferred_layerwaves: usize,
    pub request_admitted_iterations: usize,
    pub request_sparse_batches: usize,
    pub request_expert_batch_rows: usize,
    pub request_expert_batch_routes: usize,
    pub request_expert_prefill_rows: usize,
    pub request_expert_decode_rows: usize,
    pub request_expert_mtp_verify_rows: usize,
    pub request_expert_prefill_routes: usize,
    pub request_expert_decode_routes: usize,
    pub request_expert_mtp_verify_routes: usize,
    pub request_expert_source_modes_covered: bool,
    pub request_expert_route_entries_match_source_rows: bool,
    pub request_kv_reads: usize,
    pub request_committed_kv_writes: usize,
    pub request_tentative_kv_writes: usize,
    pub request_committed_mtp_writes: usize,
    pub request_discarded_mtp_writes: usize,
    pub request_backed_kv_writes: usize,
    pub request_backed_kv_bytes: usize,
    pub request_kv_reservation_bytes: usize,
    pub request_byte_backed_scheduler_trace: bool,
    pub request_numeric_progression_passed: bool,
    pub request_numeric_progression_source_rows: usize,
    pub request_numeric_progression_hidden_dim: usize,
    pub request_numeric_progression_selected_prefill_rows: usize,
    pub request_numeric_progression_selected_decode_rows: usize,
    pub request_numeric_progression_selected_mtp_rows: usize,
    pub request_numeric_progression_attention_value_updates: usize,
    pub request_numeric_progression_mlp_value_updates: usize,
    pub request_numeric_progression_visible_checksum: f32,
    pub request_numeric_progression_rejected_mtp_checksum: f32,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct BackendMetrics {
    pub(crate) cache_load_ms: f64,
    pub(crate) prefill_ms: f64,
    pub(crate) time_to_first_token_ms: Option<f64>,
    pub(crate) decode_ms: f64,
    pub(crate) reasoning_tokens: usize,
    pub(crate) cached_prompt_tokens: usize,
    pub(crate) prefill_tokens: usize,
    pub(crate) prefill_chunk_count: usize,
    pub(crate) layerwave_prefill_rows: usize,
    pub(crate) layerwave_decode_rows: usize,
    pub(crate) real_full: Option<RealFullDiagnosticMetrics>,
}

impl CompletionMetrics {
    pub(crate) fn from_backend(
        prompt_tokens: usize,
        output_tokens: usize,
        backend_mode: &'static str,
        transport_backend: &'static str,
        backend: BackendMetrics,
    ) -> Self {
        let prefill_tokens_per_sec = if backend.prefill_tokens > 0 && backend.prefill_ms > 0.0 {
            Some(backend.prefill_tokens as f64 / (backend.prefill_ms / 1000.0))
        } else {
            None
        };
        Self {
            queue_ms: 0.0,
            cache_load_ms: backend.cache_load_ms,
            prefill_ms: backend.prefill_ms,
            time_to_first_token_ms: backend
                .time_to_first_token_ms
                .unwrap_or(backend.prefill_ms + backend.decode_ms),
            decode_ms: backend.decode_ms,
            output_tokens,
            prompt_tokens,
            cached_prompt_tokens: backend.cached_prompt_tokens,
            reasoning_tokens: backend.reasoning_tokens,
            prefill_tokens_per_sec,
            transport_backend,
            backend_mode,
            prefill_chunk_count: backend.prefill_chunk_count,
            layerwave_prefill_rows: backend.layerwave_prefill_rows,
            layerwave_decode_rows: backend.layerwave_decode_rows,
            real_full: backend.real_full,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BackendMetrics, CompletionMetrics};

    #[test]
    fn real_full_metrics_report_radix_restored_prompt_rows() {
        let metrics = CompletionMetrics::from_backend(
            191,
            24,
            "real-glm-full",
            "verbs-host",
            BackendMetrics {
                prefill_tokens: 6,
                cached_prompt_tokens: 184,
                reasoning_tokens: 16,
                ..BackendMetrics::default()
            },
        );
        assert_eq!(metrics.cached_prompt_tokens, 184);
        assert_eq!(metrics.reasoning_tokens, 16);
    }
}
