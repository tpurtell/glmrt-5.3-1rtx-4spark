use crate::completion::{
    backend_name, completion_token_count, prompt_token_count, prompt_token_ids, selected_backend,
    transport_name,
};
use crate::constrained::request_constraint;
use crate::metrics::{BackendMetrics, CompletionMetrics, RealFullDiagnosticMetrics};
use crate::request::{
    real_glm_full_request_prompt_text, request_image_sources, request_max_tokens,
    request_sampling_params, stop_strings, tool_calls_enabled, unix_timestamp, validate_request,
};
use crate::streaming::{
    chat_stream_content_event, chat_stream_done_event, chat_stream_error_event,
    chat_stream_finish_event, chat_stream_reasoning_event, chat_stream_role_event,
    chat_stream_tool_call_event, chat_stream_usage_event,
};
use crate::tooling::{
    render_glm_tool_call, GlmToolCallStreamParser, GlmToolStreamDelta, GLM_TOOL_CALL_END,
};
use crate::{
    invalid_request, runtime_error, ApiBackend, ApiError, ApiState, BackendCompletion,
    ChatCompletionRequest, ChatTool, RealFullConstraint, RealFullGeneratedToken, RealFullRequest,
    RealFullRequestExecutor, RealFullSamplingParams, RealFullSequenceRequest,
    RealFullVisionEmbedding,
};
use async_stream::stream;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use std::convert::Infallible;
use std::env;
use std::path::Path;
use std::sync::{atomic::Ordering, Arc, OnceLock};
use std::time::Instant;
use uuid::Uuid;

mod trace;

use trace::real_full_request_trace;

const MULTI_TOKEN_DECODE_LOOP_REQUIREMENT: &str = "multi_token_decode_loop";
const GLM_CHAT_STOP_TOKEN_IDS: &[usize] = &[
    154_820, // <|endoftext|>
    154_825, // <eop>
    154_826, // <|system|>
    154_827, // <|user|>
    154_828, // <|assistant|>
    154_829, // <|observation|>
    154_843, // <tool_call>
    154_844, // </tool_call>
    154_845, // <tool_response>
    154_846, // </tool_response>
];

struct BorrowedRealFullSequenceGuard<'a> {
    executor: &'a dyn RealFullRequestExecutor,
    sequence_id: String,
}

impl Drop for BorrowedRealFullSequenceGuard<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.executor.finish_real_full_sequence(&self.sequence_id) {
            eprintln!(
                "real_full_sequence_finish_error sequence_id={} error={error}",
                self.sequence_id
            );
        }
    }
}

struct OwnedRealFullSequenceGuard {
    executor: Arc<dyn RealFullRequestExecutor>,
    sequence_id: String,
}

impl Drop for OwnedRealFullSequenceGuard {
    fn drop(&mut self) {
        if let Err(error) = self.executor.finish_real_full_sequence(&self.sequence_id) {
            eprintln!(
                "real_full_sequence_finish_error sequence_id={} error={error}",
                self.sequence_id
            );
        }
    }
}

#[derive(Debug, Default)]
struct RealFullMtpAcceptanceMetrics {
    verify_cycles: usize,
    draft_tokens: usize,
    accepted_draft_tokens: usize,
    emitted_tokens_from_verify: usize,
    full_match_cycles: usize,
    total_verify_cycle_ms: f64,
    draft_lengths: Vec<usize>,
    accepted_draft_lengths: Vec<usize>,
    verify_cycle_ms: Vec<f64>,
    target_cycle_physical_m: Vec<usize>,
    target_cycle_ms: Vec<f64>,
}

impl RealFullMtpAcceptanceMetrics {
    fn record(
        &mut self,
        full: &crate::RealFullInfo,
        emitted_tokens: usize,
        cycle_ms: f64,
        post_ttft: bool,
    ) {
        // The first target cycle includes prompt ingestion and is reported as
        // TTFT, not decode_ms. Keep it out of the physical-M decode curve so
        // every sample measures only a post-prefill target-model cycle.
        if post_ttft {
            self.target_cycle_physical_m
                .push(full.request_mtp_verify_rows.saturating_add(1));
            self.target_cycle_ms.push(cycle_ms);
        }
        if full.request_mtp_verify_rows == 0 {
            return;
        }
        self.verify_cycles += 1;
        self.draft_tokens += full.request_mtp_verify_rows;
        self.accepted_draft_tokens += full.request_mtp_accepted_rows;
        self.emitted_tokens_from_verify += emitted_tokens;
        self.full_match_cycles +=
            usize::from(full.request_mtp_accepted_rows == full.request_mtp_verify_rows);
        self.total_verify_cycle_ms += cycle_ms;
        self.draft_lengths.push(full.request_mtp_verify_rows);
        self.accepted_draft_lengths
            .push(full.request_mtp_accepted_rows);
        self.verify_cycle_ms.push(cycle_ms);
    }

    fn apply(&self, metrics: &mut RealFullDiagnosticMetrics) {
        metrics.mtp_verify_cycles = self.verify_cycles;
        metrics.mtp_draft_tokens = self.draft_tokens;
        metrics.mtp_accepted_draft_tokens = self.accepted_draft_tokens;
        metrics.mtp_emitted_tokens_from_verify = self.emitted_tokens_from_verify;
        metrics.mtp_full_match_cycles = self.full_match_cycles;
        metrics.mtp_total_verify_cycle_ms = self.total_verify_cycle_ms;
        metrics.mtp_draft_lengths = self.draft_lengths.clone();
        metrics.mtp_accepted_draft_lengths = self.accepted_draft_lengths.clone();
        metrics.mtp_verify_cycle_ms = self.verify_cycle_ms.clone();
        metrics.target_cycle_physical_m = self.target_cycle_physical_m.clone();
        metrics.target_cycle_ms = self.target_cycle_ms.clone();
    }
}

const GLM_THINK_OPEN_TOKEN_ID: usize = 154_841;
const GLM_THINK_CLOSE_TOKEN_ID: usize = 154_842;
const GLM_TOOL_CALL_TOKEN_IDS: &[usize] = &[154_843, 154_844];
const GLM_THINK_OPEN_TEXT: &str = "<think>";
const GLM_THINK_CLOSE_TEXT: &str = "</think>";
const GLM_TOOL_CALL_TEXTS: &[&str] = &["<tool_call>", "</tool_call>"];
const REAL_FULL_REQUEST_TIMING_ENV: &str = "GLMRT_REAL_FULL_REQUEST_TIMING";
const REAL_FULL_MAX_CONTEXT_TOKENS_ENV: &str = "GLMRT_REAL_FULL_SERVE_MAX_CONTEXT_TOKENS";
const REAL_FULL_MAX_OUTPUT_TOKENS_ENV: &str = "GLMRT_REAL_FULL_SERVE_MAX_OUTPUT_TOKENS";
const GLM_CHAT_STOP_TEXTS: &[&str] = &[
    "<|endoftext|>",
    "<eop>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
];

fn optional_positive_token_limit(name: &str) -> Result<Option<usize>, ApiError> {
    let Some(raw) = env::var(name).ok().filter(|value| !value.trim().is_empty()) else {
        return Ok(None);
    };
    let limit = raw
        .parse::<usize>()
        .map_err(|error| runtime_error(format!("invalid {name}={raw}: {error}")))?;
    if limit == 0 {
        return Err(runtime_error(format!("{name} must be greater than zero")));
    }
    Ok(Some(limit))
}

fn validate_real_full_request_token_limits(
    prompt_tokens: usize,
    max_tokens: usize,
    max_context_tokens: Option<usize>,
    max_output_tokens: Option<usize>,
) -> Result<(), ApiError> {
    if max_context_tokens.is_some_and(|limit| prompt_tokens > limit) {
        return Err(invalid_request(
            format!(
                "rendered prompt has {prompt_tokens} tokens, exceeding the server context limit {}",
                max_context_tokens.expect("checked above")
            ),
            Some("messages"),
        ));
    }
    if max_output_tokens.is_some_and(|limit| max_tokens > limit) {
        return Err(invalid_request(
            format!(
                "requested output has {max_tokens} tokens, exceeding the server output limit {}",
                max_output_tokens.expect("checked above")
            ),
            Some("max_tokens"),
        ));
    }
    if let Some(limit) = max_context_tokens {
        let total_tokens = prompt_tokens.checked_add(max_tokens).ok_or_else(|| {
            invalid_request(
                "rendered prompt plus requested output overflows the token budget",
                Some("max_tokens"),
            )
        })?;
        if total_tokens > limit {
            return Err(invalid_request(
                format!(
                    "rendered prompt ({prompt_tokens}) plus requested output ({max_tokens}) \
                     requires {total_tokens} tokens, exceeding the server context limit {limit}"
                ),
                Some("max_tokens"),
            ));
        }
    }
    Ok(())
}

fn validate_real_full_request_profile_limits(
    prompt_tokens: usize,
    max_tokens: usize,
) -> Result<(), ApiError> {
    validate_real_full_request_token_limits(
        prompt_tokens,
        max_tokens,
        optional_positive_token_limit(REAL_FULL_MAX_CONTEXT_TOKENS_ENV)?,
        optional_positive_token_limit(REAL_FULL_MAX_OUTPUT_TOKENS_ENV)?,
    )
}

fn persistent_real_full_sequence_request(
    request: RealFullRequest,
    max_output_tokens: usize,
    min_output_tokens: usize,
    ignore_eos: bool,
    allow_tool_calls: bool,
) -> RealFullSequenceRequest {
    RealFullSequenceRequest {
        request,
        max_output_tokens,
        min_output_tokens,
        ignore_eos,
        stop_token_ids: GLM_CHAT_STOP_TOKEN_IDS
            .iter()
            .copied()
            .filter(|token_id| !allow_tool_calls || !GLM_TOOL_CALL_TOKEN_IDS.contains(token_id))
            .collect(),
        stop_texts: GLM_CHAT_STOP_TEXTS
            .iter()
            .copied()
            .filter(|text| !allow_tool_calls || !GLM_TOOL_CALL_TEXTS.contains(text))
            .map(str::to_owned)
            .collect(),
    }
}

fn resolve_real_full_prompt_token_ids(
    state: &ApiState,
    request: &ChatCompletionRequest,
    prompt: &str,
    backend: ApiBackend,
) -> Option<Arc<Vec<usize>>> {
    let canonical = prompt_token_ids(state, backend, prompt).map(Arc::new)?;
    let (continuation, mismatch) = state
        .tool_continuations
        .lock()
        .ok()
        .map(|cache| {
            (
                cache.matching_prefix(request, prompt),
                cache.known_call_without_text_match(request, prompt),
            )
        })
        .unwrap_or_default();
    let Some(continuation) = continuation else {
        if let Some((call_id, matching_bytes)) = mismatch {
            eprintln!(
                "real_full_tool_continuation_miss call_id={} reason=text-mismatch matching_bytes={}",
                call_id, matching_bytes,
            );
        }
        return Some(canonical);
    };

    let suffix = &prompt[continuation.prefix_text_len..];
    let Some(suffix_token_ids) = prompt_token_ids(state, backend, suffix) else {
        return Some(canonical);
    };
    let mut exact = Vec::with_capacity(continuation.token_ids.len() + suffix_token_ids.len());
    exact.extend(continuation.token_ids.iter().copied());
    exact.extend(suffix_token_ids);
    eprintln!(
        "real_full_tool_continuation_reuse call_id={} prefix_tokens={} suffix_tokens={} canonical_tokens={} exact_tokens={}",
        continuation.call_id,
        continuation.token_ids.len(),
        exact.len().saturating_sub(continuation.token_ids.len()),
        canonical.len(),
        exact.len(),
    );
    Some(Arc::new(exact))
}

pub(crate) fn try_real_glm_full_streaming_response(
    state: Arc<ApiState>,
    request: &ChatCompletionRequest,
) -> Result<Option<Response>, ApiError> {
    validate_request(request)?;
    let backend = selected_backend(&state, request);
    if backend != ApiBackend::RealGlmFull {
        return Ok(None);
    }

    if stop_strings(request.stop.as_ref())
        .iter()
        .any(|stop| !stop.is_empty())
    {
        return Ok(None);
    }
    state
        .config
        .real_full
        .as_ref()
        .ok_or_else(|| runtime_error("real-glm-full backend has no preflight status"))?;
    let Some(executor) = state.config.real_full_executor.clone() else {
        return Ok(None);
    };

    let prompt = real_glm_full_request_prompt_text(request);
    let max_tokens = request_max_tokens(request);
    let min_tokens = request.min_tokens.unwrap_or(0);
    let ignore_eos = request.ignore_eos.unwrap_or(false);
    let sampling = request_sampling_params(request);
    let constraint = request_constraint(request)?;
    let image_sources = request_image_sources(request)?;
    let (resolved_prompt_token_ids, vision_embeddings) = if image_sources.is_empty() {
        (
            resolve_real_full_prompt_token_ids(&state, request, &prompt, backend),
            None,
        )
    } else {
        let initial_prompt_token_ids = prompt_token_ids(&state, backend, &prompt)
            .ok_or_else(|| runtime_error("vision input requires the loaded GLM tokenizer"))?;
        let prepared =
            crate::vision::prepare_vision_prompt(initial_prompt_token_ids, &image_sources)?;
        (Some(prepared.prompt_token_ids), Some(prepared.embeddings))
    };
    let prompt_tokens = resolved_prompt_token_ids.as_ref().map_or_else(
        || prompt_token_count(&state, backend, &prompt),
        |ids| ids.len(),
    );
    validate_real_full_request_profile_limits(prompt_tokens, max_tokens)?;
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = unix_timestamp();
    let model = request.model.clone();
    let transport_backend = transport_name(state.config.transport);
    let tools = tool_calls_enabled(request).then(|| request.tools.clone().unwrap_or_default());
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|options| options.include_usage);
    Ok(Some(real_glm_full_decode_stream_response(
        state,
        executor,
        prompt,
        prompt_tokens,
        resolved_prompt_token_ids,
        vision_embeddings,
        max_tokens,
        min_tokens,
        ignore_eos,
        sampling,
        id,
        created,
        model,
        transport_backend,
        tools,
        constraint,
        include_usage,
    )))
}

#[allow(clippy::too_many_arguments)]
fn real_glm_full_decode_stream_response(
    state: Arc<ApiState>,
    executor: Arc<dyn RealFullRequestExecutor>,
    prompt: String,
    prompt_tokens: usize,
    prompt_token_ids: Option<Arc<Vec<usize>>>,
    vision_embeddings: Option<Arc<Vec<RealFullVisionEmbedding>>>,
    max_tokens: usize,
    min_tokens: usize,
    ignore_eos: bool,
    sampling: RealFullSamplingParams,
    id: String,
    created: u64,
    model: String,
    transport_backend: &'static str,
    tools: Option<Vec<ChatTool>>,
    constraint: Option<Arc<RealFullConstraint>>,
    include_usage: bool,
) -> Response {
    let stream = stream! {
        eprintln!(
            "real_full_stream_start model={} prompt_tokens={} max_tokens={} transport={}",
            model, prompt_tokens, max_tokens, transport_backend
        );

        let mut generated_token_ids = Vec::with_capacity(max_tokens);
        let mut stream_role_sent = false;
        let mut generated_completion_tokens = 0_usize;
        let mut generated_reasoning_tokens = 0_usize;
        let mut last_full = None;
        let mut diagnostic_completion = None;
        let mut first_step_ms = None;
        let mut cache_load_ms = 0.0_f64;
        let mut total_step_ms = 0.0;
        let mut executed_decode_steps = 0_usize;
        let mut prefill_tokens_reported = 0_usize;
        let mut prefill_chunks_reported = 0_usize;
        let mut mtp_acceptance_metrics = RealFullMtpAcceptanceMetrics::default();
        let mut stopped_by_model = false;
        let mut raw_generated_text = String::new();
        let mut continuation_generated_token_ids = Vec::with_capacity(max_tokens);
        let mut continuation_reasoning_content = String::new();
        let mut continuation_visible_content = String::new();
        let mut continuation_cacheable = true;
        let sequence_id = format!("{id}-sequence");
        let allow_tool_calls = tools.is_some();
        let reservation = u64::try_from(max_tokens.saturating_add(1)).unwrap_or(u64::MAX);
        let initial_request_index = state
            .next_request_id
            .fetch_add(reservation, Ordering::Relaxed);
        let mut initial_request = RealFullRequest::new_decode_step_for_sequence(
            initial_request_index,
            sequence_id.clone(),
            &prompt,
            prompt_tokens,
            1,
            Vec::new(),
            0,
            max_tokens,
        )
        .with_sampling(sampling)
        .with_constraint(constraint.clone());
        if let Some(prompt_token_ids) = prompt_token_ids.as_ref() {
            initial_request =
                initial_request.with_prompt_token_ids(Arc::clone(prompt_token_ids));
        }
        if let Some(vision_embeddings) = vision_embeddings.as_ref() {
            initial_request =
                initial_request.with_vision_embeddings(Arc::clone(vision_embeddings));
        }
        let mut persistent = executor
            .start_real_full_sequence(persistent_real_full_sequence_request(
                initial_request,
                max_tokens,
                min_tokens,
                ignore_eos,
                allow_tool_calls,
            ))
            .ok();
        let mut sequence_guard = persistent
            .is_none()
            .then(|| OwnedRealFullSequenceGuard {
                executor: Arc::clone(&executor),
                sequence_id: sequence_id.clone(),
            });
        let mut output_filter = GlmAssistantOutputFilter::new(
            allow_tool_calls,
            prompt.ends_with(GLM_THINK_OPEN_TEXT),
        );
        let mut token_decoder = real_full_streaming_token_decoder(&state);
        let mut tool_stream = tools.map(GlmToolCallStreamParser::new);

        while generated_token_ids.len() < max_tokens {
            let decode_step_index = generated_token_ids.len();
            let cycle_result = if let Some(receiver) = persistent.as_mut() {
                match receiver.recv().await {
                    Some(Ok(event)) => {
                        let step_ms = (event.sequence_elapsed_ms - total_step_ms).max(0.0);
                        total_step_ms = event.sequence_elapsed_ms;
                        Ok((event.cycle, step_ms))
                    }
                    Some(Err(error)) => Err(error),
                    None => Err("persistent real-full sequence ended without a cycle".to_owned()),
                }
            } else {
                let request_index = initial_request_index.saturating_add(
                    u64::try_from(decode_step_index).unwrap_or(u64::MAX),
                );
                let request = RealFullRequest::new_decode_step_for_sequence(
                    request_index,
                    sequence_id.clone(),
                    &prompt,
                    prompt_tokens,
                    1,
                    generated_token_ids.clone(),
                    decode_step_index,
                    max_tokens,
                )
                .with_sampling(sampling)
                .with_constraint(constraint.clone());
                eprintln!(
                    "real_full_stream_decode_step_start request_id={} step={}/{} prefill_tokens={} generated_tokens={}",
                    request.request_id,
                    decode_step_index + 1,
                    max_tokens,
                    request.prompt_tokens + request.generated_token_ids.len(),
                    request.generated_token_ids.len()
                );
                let step_start = Instant::now();
                let executor_for_step = Arc::clone(&executor);
                match tokio::task::spawn_blocking(move || {
                    executor_for_step.execute_real_full_decode_cycle(request)
                })
                .await
                {
                    Ok(Ok(cycle)) => {
                        let step_ms = crate::duration_ms(step_start.elapsed());
                        total_step_ms += step_ms;
                        Ok((cycle, step_ms))
                    }
                    Ok(Err(error)) => Err(error),
                    Err(error) => Err(format!("joining real-full decode worker failed: {error}")),
                }
            };
            let (cycle, step_ms) = match cycle_result {
                Ok(result) => result,
                Err(error) => {
                    eprintln!(
                        "real_full_stream_decode_step_error step={}/{} elapsed_ms={:.3} error={}",
                        decode_step_index + 1,
                        max_tokens,
                        total_step_ms,
                        error
                    );
                    // A post-header streaming failure cannot change the HTTP
                    // status, but it must never masquerade as generated model
                    // content. The fallback executor guard is synchronous;
                    // release its scheduler/KV lane before publishing the
                    // terminal error, matching the persistent worker contract.
                    drop(sequence_guard.take());
                    yield Ok::<Event, Infallible>(chat_stream_error_event(format!(
                        "real-full streaming executor error: {error}"
                    )));
                    yield Ok::<Event, Infallible>(chat_stream_done_event());
                    return;
                }
            };
            executed_decode_steps += 1;
            let full = cycle.info;
            cache_load_ms = cache_load_ms.max(full.request_kv_snapshot_restore_ms);
            let cycle_tokens = cycle.generated_tokens;
            mtp_acceptance_metrics.record(
                &full,
                cycle_tokens.len().max(1),
                step_ms,
                first_step_ms.is_some(),
            );
            if first_step_ms.is_none() {
                first_step_ms = Some(step_ms);
            }
            if full.request_prefill_tokens > 0 {
                prefill_tokens_reported = prefill_tokens_reported.max(full.request_prefill_tokens);
                prefill_chunks_reported = prefill_chunks_reported.max(full.request_prefill_chunks);
            }
            eprintln!(
                "real_full_stream_decode_step_done step={}/{} elapsed_ms={:.3} status={} sample_status={} top_token_id={:?} sampled_token_id={:?} cycle_tokens={}",
                decode_step_index + 1,
                max_tokens,
                step_ms,
                full.status,
                full.scheduler_terminal_lm_head_sample_status,
                full.scheduler_terminal_lm_head_top_token_id,
                full.scheduler_terminal_lm_head_sampled_token_id,
                cycle_tokens.len().max(1)
            );
            if !stream_role_sent {
                // Match the established OpenAI-compatible serving behavior:
                // headers may be available earlier, but the first SSE event is
                // not model progress until the first decode result is ready.
                // Keep the role as its own event and immediately follow it with
                // the model-derived delta below so multi-token MTP cycles remain
                // batched exactly as produced by the executor.
                yield Ok::<Event, Infallible>(chat_stream_role_event(&id, created, &model));
                stream_role_sent = true;
            }
            let Some(samples) = terminal_cycle_samples(
                &full,
                &cycle_tokens,
                allow_tool_calls,
                token_decoder.as_mut(),
            ) else {
                let step_trace = real_full_request_trace(prompt_tokens + generated_token_ids.len(), 1)
                    .unwrap_or_else(|_| real_full_request_trace(prompt_tokens, 1)
                        .expect("single-token real-full diagnostic trace should be valid"));
                let mut completion = diagnostic_real_full_completion(&full, &step_trace);
                completion.metrics.cache_load_ms = cache_load_ms;
                completion.metrics.prefill_ms =
                    (first_step_ms.unwrap_or(0.0) - cache_load_ms).max(0.0);
                completion.metrics.time_to_first_token_ms = first_step_ms;
                completion.metrics.decode_ms =
                    (total_step_ms - first_step_ms.unwrap_or(0.0)).max(0.0);
                if let Some(metrics) = completion.metrics.real_full.as_mut() {
                    mtp_acceptance_metrics.apply(metrics);
                }
                if !completion.content.is_empty() {
                    yield Ok::<Event, Infallible>(chat_stream_content_event(
                        &id,
                        created,
                        &model,
                        completion.content.clone(),
                    ));
                }
                diagnostic_completion = Some(completion);
                break;
            };

            let mut should_stop = false;
            for mut sample in samples
                .into_iter()
                .take(max_tokens - generated_token_ids.len())
            {
                let stop_allowed = !ignore_eos
                    && generated_token_ids.len().saturating_add(1) >= min_tokens;
                if !stop_allowed {
                    sample.stop = false;
                }
                if sample.stop {
                    if !sample.content.is_empty() {
                        continuation_cacheable = false;
                    }
                    raw_generated_text.push_str(&sample.content);
                } else {
                    continuation_generated_token_ids.push(sample.token_id);
                    raw_generated_text.push_str(&sample.content);
                }
                generated_token_ids.push(sample.token_id);
                let mut sample = output_filter.apply(sample);
                if !stop_allowed {
                    sample.stop = false;
                }
                if sample.clear_prior_content {
                    continuation_reasoning_content.clear();
                    continuation_visible_content.clear();
                    generated_reasoning_tokens = 0;
                }
                if !sample.stop || !sample.content.is_empty() {
                    generated_completion_tokens += 1;
                }
                if !sample.reasoning_content.is_empty() {
                    generated_reasoning_tokens += 1;
                    continuation_reasoning_content.push_str(&sample.reasoning_content);
                    yield Ok::<Event, Infallible>(chat_stream_reasoning_event(
                        &id,
                        created,
                        &model,
                        sample.reasoning_content,
                    ));
                }
                if !sample.content.is_empty() {
                    if let Some(parser) = tool_stream.as_mut() {
                        for delta in parser.push(&sample.content) {
                            match delta {
                                GlmToolStreamDelta::Content(content) => {
                                    if !content.is_empty() {
                                        continuation_visible_content.push_str(&content);
                                        yield Ok::<Event, Infallible>(chat_stream_content_event(
                                            &id,
                                            created,
                                            &model,
                                            content,
                                        ));
                                    }
                                }
                                GlmToolStreamDelta::ToolCall {
                                    index,
                                    id: call_id,
                                    name,
                                    arguments,
                                } => {
                                    yield Ok::<Event, Infallible>(chat_stream_tool_call_event(
                                        &id,
                                        created,
                                        &model,
                                        index,
                                        call_id,
                                        name,
                                        arguments,
                                    ));
                                }
                            }
                        }
                    } else {
                        continuation_visible_content.push_str(&sample.content);
                        yield Ok::<Event, Infallible>(chat_stream_content_event(
                            &id,
                            created,
                            &model,
                            sample.content,
                        ));
                    }
                }
                if sample.stop {
                    should_stop = true;
                    break;
                }
            }
            last_full = Some(full);
            if should_stop {
                stopped_by_model = true;
                break;
            }
        }

        if let Some(parser) = tool_stream.as_mut() {
            for delta in parser.finish() {
                match delta {
                    GlmToolStreamDelta::Content(content) => {
                        if !content.is_empty() {
                            continuation_visible_content.push_str(&content);
                            yield Ok::<Event, Infallible>(chat_stream_content_event(
                                &id,
                                created,
                                &model,
                                content,
                            ));
                        }
                    }
                    GlmToolStreamDelta::ToolCall {
                        index,
                        id: call_id,
                        name,
                        arguments,
                    } => {
                        yield Ok::<Event, Infallible>(chat_stream_tool_call_event(
                            &id,
                            created,
                            &model,
                            index,
                            call_id,
                            name,
                            arguments,
                        ));
                    }
                }
            }
        }
        let completed_tool_calls = tool_stream
            .as_ref()
            .map_or(0, GlmToolCallStreamParser::completed_tool_calls);
        continuation_cacheable &= raw_generated_text.ends_with(GLM_TOOL_CALL_END);
        continuation_cacheable &= prompt.ends_with(GLM_THINK_OPEN_TEXT)
            || continuation_reasoning_content.is_empty();
        if completed_tool_calls > 0 && continuation_cacheable {
            if let (Some(prompt_token_ids), Some(parser)) =
                (prompt_token_ids.as_ref(), tool_stream.as_ref())
            {
                let mut continuation_token_ids = Vec::with_capacity(
                    prompt_token_ids.len() + continuation_generated_token_ids.len(),
                );
                continuation_token_ids.extend(prompt_token_ids.iter().copied());
                continuation_token_ids.extend(continuation_generated_token_ids.iter().copied());
                let continuation_token_ids = Arc::new(continuation_token_ids);
                let mut prefix_text = String::with_capacity(
                    prompt.len()
                        + continuation_reasoning_content.len()
                        + continuation_visible_content.len()
                        + raw_generated_text.len(),
                );
                prefix_text.push_str(&prompt);
                if prompt.ends_with(GLM_THINK_OPEN_TEXT) {
                    prefix_text.push_str(&continuation_reasoning_content);
                    prefix_text.push_str(GLM_THINK_CLOSE_TEXT);
                }
                prefix_text.push_str(&continuation_visible_content);
                for tool_call in parser.completed_tool_call_values() {
                    prefix_text.push_str(&render_glm_tool_call(tool_call));
                }
                if let Ok(mut cache) = state.tool_continuations.lock() {
                    cache.insert(
                        parser.completed_tool_call_ids().to_vec(),
                        prefix_text,
                        Arc::clone(&continuation_token_ids),
                    );
                    eprintln!(
                        "real_full_tool_continuation_publish calls={} prefix_tokens={} generated_tokens={}",
                        parser.completed_tool_call_ids().len(),
                        continuation_token_ids.len(),
                        continuation_generated_token_ids.len(),
                    );
                }
            }
        }

        let (finish_reason, completion_tokens, backend_metrics) =
            if let Some(completion) = diagnostic_completion {
                let completion_tokens =
                    completion_token_count(&completion.content, completion.completion_tokens);
                let finish_reason = if completion_tokens >= max_tokens {
                    "length"
                } else {
                    "stop"
                };
                (finish_reason.to_owned(), completion_tokens, completion.metrics)
            } else if let Some(final_full) = last_full.as_ref() {
                let mut real_full_metrics =
                    if RequestDiagnosticEvidence::runtime_reported(final_full) {
                        real_full_runtime_diagnostic_metrics(final_full)
                    } else {
                        let final_trace = real_full_request_trace(
                            prompt_tokens + generated_token_ids.len().saturating_sub(1),
                            1,
                        )
                        .unwrap_or_else(|_| {
                            real_full_request_trace(prompt_tokens, 1)
                                .expect("single-token real-full diagnostic trace should be valid")
                        });
                        real_full_diagnostic_metrics(final_full, &final_trace)
                    };
                mtp_acceptance_metrics.apply(&mut real_full_metrics);
                (
                    "length".to_owned(),
                    generated_completion_tokens,
                    BackendMetrics {
                        cache_load_ms,
                        prefill_ms: (first_step_ms.unwrap_or(0.0) - cache_load_ms).max(0.0),
                        time_to_first_token_ms: first_step_ms,
                        decode_ms: (total_step_ms - first_step_ms.unwrap_or(0.0)).max(0.0),
                        reasoning_tokens: generated_reasoning_tokens,
                        cached_prompt_tokens: prompt_tokens
                            .saturating_sub(prefill_tokens_reported.saturating_add(1)),
                        prefill_tokens: prefill_tokens_reported,
                        prefill_chunk_count: prefill_chunks_reported,
                        layerwave_prefill_rows: prefill_tokens_reported,
                        layerwave_decode_rows: executed_decode_steps,
                        real_full: Some(real_full_metrics),
                    },
                )
            } else {
                (
                    "stop".to_owned(),
                    0,
                    BackendMetrics::default(),
                )
            };

        let metrics = CompletionMetrics::from_backend(
            prompt_tokens,
            completion_tokens,
            backend_name(ApiBackend::RealGlmFull),
            transport_backend,
            backend_metrics,
        );
        let finish_reason = if completed_tool_calls > 0 {
            "tool_calls".to_owned()
        } else if stopped_by_model {
            "stop".to_owned()
        } else {
            finish_reason
        };
        yield Ok::<Event, Infallible>(chat_stream_finish_event(
            id.clone(),
            created,
            model.clone(),
            finish_reason,
            metrics.clone(),
        ));
        if include_usage {
            yield Ok::<Event, Infallible>(chat_stream_usage_event(
                id,
                created,
                model,
                &metrics,
            ));
        }
        yield Ok::<Event, Infallible>(chat_stream_done_event());
    };
    Sse::new(stream).into_response()
}

pub(crate) async fn real_glm_full_completion(
    state: &ApiState,
    prompt: &str,
    prompt_tokens: usize,
    prompt_token_ids: Option<Arc<Vec<usize>>>,
    vision_embeddings: Option<Arc<Vec<RealFullVisionEmbedding>>>,
    max_tokens: usize,
    min_tokens: usize,
    ignore_eos: bool,
    sampling: RealFullSamplingParams,
    allow_tool_calls: bool,
    constraint: Option<Arc<RealFullConstraint>>,
) -> Result<BackendCompletion, ApiError> {
    validate_real_full_request_profile_limits(prompt_tokens, max_tokens)?;
    let full = state
        .config
        .real_full
        .as_ref()
        .ok_or_else(|| runtime_error("real-glm-full backend has no preflight status"))?;
    if let Some(executor) = state.config.real_full_executor.as_ref() {
        if max_tokens > 1 || executor.real_full_persistent_sequence_scheduling_enabled() {
            return execute_real_full_decode_loop(
                state,
                executor.as_ref(),
                prompt,
                prompt_tokens,
                prompt_token_ids,
                vision_embeddings,
                max_tokens,
                min_tokens,
                ignore_eos,
                sampling,
                allow_tool_calls,
                constraint,
            )
            .await;
        }
    }
    let trace = real_full_request_trace(prompt_tokens, max_tokens)?;
    let mut executor_elapsed_ms = 0.0;
    let request_execution = if let Some(executor) = state.config.real_full_executor.as_ref() {
        let request_index = state.next_request_id.fetch_add(1, Ordering::Relaxed);
        let mut request = RealFullRequest::new(request_index, prompt, prompt_tokens, max_tokens)
            .with_sampling(sampling)
            .with_constraint(constraint);
        if let Some(prompt_token_ids) = prompt_token_ids {
            request = request.with_prompt_token_ids(prompt_token_ids);
        }
        if let Some(vision_embeddings) = vision_embeddings {
            request = request.with_vision_embeddings(vision_embeddings);
        }
        let _sequence_guard = BorrowedRealFullSequenceGuard {
            executor: executor.as_ref(),
            sequence_id: request.sequence_id.clone(),
        };
        let request_timing = real_full_request_timing_enabled();
        if request_timing {
            eprintln!(
                "real_full_request_start request_id={} prompt_tokens={} max_tokens={}",
                request.request_id, request.prompt_tokens, request.max_tokens
            );
        }
        let start = Instant::now();
        let full = executor
            .execute_real_full_request(request)
            .map_err(runtime_error)?;
        executor_elapsed_ms = crate::duration_ms(start.elapsed());
        if request_timing {
            eprintln!(
                "real_full_request_done elapsed_ms={:.3} status={} sample_status={} sampled_token_id={:?}",
                executor_elapsed_ms,
                full.status,
                full.scheduler_terminal_lm_head_sample_status,
                full.scheduler_terminal_lm_head_sampled_token_id
            );
        }
        Some(full)
    } else {
        None
    };
    let has_live_request_execution = request_execution.is_some();
    let full = request_execution.as_ref().unwrap_or(full);
    let request = RequestDiagnosticEvidence::new(full, &trace);
    let terminal_sample = terminal_sample(full, allow_tool_calls);
    let blocked_multi_token = if terminal_sample.is_some() && request.decode_budget > 1 {
        Some(real_full_with_multi_token_decode_blocker(
            full,
            request.decode_budget,
        ))
    } else {
        None
    };
    let generated_sample = if blocked_multi_token.is_none() {
        terminal_sample.map(|sample| {
            GlmAssistantOutputFilter::new(allow_tool_calls, prompt.ends_with(GLM_THINK_OPEN_TEXT))
                .apply(sample)
        })
    } else {
        None
    };
    let diagnostic_full = blocked_multi_token.as_ref().unwrap_or(full);
    let generated_runtime_metrics = || {
        (has_live_request_execution && request.runtime_reported)
            .then(|| real_full_diagnostic_metrics(diagnostic_full, &trace))
    };
    let (content, completion_tokens, stream_chunks, real_full_metrics) = match generated_sample {
        Some(sample) if sample.content.is_empty() => (
            String::new(),
            Some(0),
            Some(Vec::new()),
            generated_runtime_metrics(),
        ),
        Some(sample) => {
            let content = sample.content;
            (
                content.clone(),
                Some(1),
                Some(vec![content]),
                generated_runtime_metrics(),
            )
        }
        None => (
            real_full_diagnostic_content(diagnostic_full, &trace),
            None,
            None,
            Some(real_full_diagnostic_metrics(diagnostic_full, &trace)),
        ),
    };
    Ok(BackendCompletion {
        content,
        reasoning_content: None,
        completion_tokens,
        stream_chunks,
        metrics: BackendMetrics {
            cache_load_ms: 0.0,
            prefill_ms: executor_elapsed_ms,
            time_to_first_token_ms: if executor_elapsed_ms > 0.0 {
                Some(executor_elapsed_ms)
            } else {
                None
            },
            decode_ms: 0.0,
            reasoning_tokens: 0,
            cached_prompt_tokens: prompt_tokens
                .saturating_sub(request.prefill_tokens.saturating_add(1)),
            prefill_tokens: request.prefill_tokens,
            prefill_chunk_count: request.prefill_chunks,
            layerwave_prefill_rows: request.prefill_tokens,
            layerwave_decode_rows: request.decode_budget,
            real_full: real_full_metrics,
        },
    })
}

fn real_full_request_timing_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var(REAL_FULL_REQUEST_TIMING_ENV)
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
    })
}

async fn execute_real_full_decode_loop(
    state: &ApiState,
    executor: &dyn RealFullRequestExecutor,
    prompt: &str,
    prompt_tokens: usize,
    prompt_token_ids: Option<Arc<Vec<usize>>>,
    vision_embeddings: Option<Arc<Vec<RealFullVisionEmbedding>>>,
    max_tokens: usize,
    min_tokens: usize,
    ignore_eos: bool,
    sampling: RealFullSamplingParams,
    allow_tool_calls: bool,
    constraint: Option<Arc<RealFullConstraint>>,
) -> Result<BackendCompletion, ApiError> {
    let mut generated_token_ids = Vec::with_capacity(max_tokens);
    let mut content = String::new();
    let mut reasoning_content = String::new();
    let mut generated_reasoning_tokens = 0_usize;
    let mut stream_chunks = Vec::with_capacity(max_tokens);
    let mut last_full = None;
    let mut first_step_ms = None;
    let mut cache_load_ms = 0.0_f64;
    let mut total_step_ms = 0.0;
    let mut executed_decode_steps = 0_usize;
    let mut prefill_tokens_reported = 0_usize;
    let mut prefill_chunks_reported = 0_usize;
    let mut mtp_acceptance_metrics = RealFullMtpAcceptanceMetrics::default();
    let mut stopped_by_model = false;
    let sequence_id = format!("real-glm-full-api-decode-loop-{}", Uuid::new_v4());
    let reservation = u64::try_from(max_tokens.saturating_add(1)).unwrap_or(u64::MAX);
    let initial_request_index = state
        .next_request_id
        .fetch_add(reservation, Ordering::Relaxed);
    let mut initial_request = RealFullRequest::new_decode_step_for_sequence(
        initial_request_index,
        sequence_id.clone(),
        prompt,
        prompt_tokens,
        1,
        Vec::new(),
        0,
        max_tokens,
    )
    .with_sampling(sampling)
    .with_constraint(constraint.clone());
    if let Some(prompt_token_ids) = prompt_token_ids {
        initial_request = initial_request.with_prompt_token_ids(prompt_token_ids);
    }
    if let Some(vision_embeddings) = vision_embeddings {
        initial_request = initial_request.with_vision_embeddings(vision_embeddings);
    }
    let mut persistent = executor
        .start_real_full_sequence(persistent_real_full_sequence_request(
            initial_request,
            max_tokens,
            min_tokens,
            ignore_eos,
            allow_tool_calls,
        ))
        .ok();
    let _sequence_guard = persistent.is_none().then(|| BorrowedRealFullSequenceGuard {
        executor,
        sequence_id: sequence_id.clone(),
    });
    let mut output_filter =
        GlmAssistantOutputFilter::new(allow_tool_calls, prompt.ends_with(GLM_THINK_OPEN_TEXT));
    let mut token_decoder = real_full_streaming_token_decoder(state);
    let request_timing = real_full_request_timing_enabled();

    while generated_token_ids.len() < max_tokens {
        let decode_step_index = generated_token_ids.len();
        let (cycle, step_ms) = if let Some(receiver) = persistent.as_mut() {
            let event = receiver
                .recv()
                .await
                .ok_or_else(|| {
                    runtime_error("persistent real-full sequence ended without a cycle")
                })?
                .map_err(runtime_error)?;
            let step_ms = (event.sequence_elapsed_ms - total_step_ms).max(0.0);
            total_step_ms = event.sequence_elapsed_ms;
            (event.cycle, step_ms)
        } else {
            let request_index = initial_request_index
                .saturating_add(u64::try_from(decode_step_index).unwrap_or(u64::MAX));
            let request = RealFullRequest::new_decode_step_for_sequence(
                request_index,
                sequence_id.clone(),
                prompt,
                prompt_tokens,
                1,
                generated_token_ids.clone(),
                decode_step_index,
                max_tokens,
            )
            .with_sampling(sampling)
            .with_constraint(constraint.clone());
            if request_timing {
                eprintln!(
                    "real_full_decode_loop_step_start request_id={} step={}/{} prefill_tokens={} generated_tokens={}",
                    request.request_id,
                    decode_step_index + 1,
                    max_tokens,
                    request.prompt_tokens + request.generated_token_ids.len(),
                    request.generated_token_ids.len()
                );
            }
            let step_start = Instant::now();
            let cycle = executor
                .execute_real_full_decode_cycle(request)
                .map_err(runtime_error)?;
            let step_ms = crate::duration_ms(step_start.elapsed());
            total_step_ms += step_ms;
            (cycle, step_ms)
        };
        executed_decode_steps += 1;
        let full = cycle.info;
        cache_load_ms = cache_load_ms.max(full.request_kv_snapshot_restore_ms);
        let cycle_tokens = cycle.generated_tokens;
        mtp_acceptance_metrics.record(
            &full,
            cycle_tokens.len().max(1),
            step_ms,
            first_step_ms.is_some(),
        );
        if first_step_ms.is_none() {
            first_step_ms = Some(step_ms);
        }
        if full.request_prefill_tokens > 0 {
            prefill_tokens_reported = prefill_tokens_reported.max(full.request_prefill_tokens);
            prefill_chunks_reported = prefill_chunks_reported.max(full.request_prefill_chunks);
        }
        if request_timing {
            eprintln!(
                "real_full_decode_loop_step_done step={}/{} elapsed_ms={:.3} status={} sample_status={} top_token_id={:?} sampled_token_id={:?} cycle_tokens={}",
                decode_step_index + 1,
                max_tokens,
                step_ms,
                full.status,
                full.scheduler_terminal_lm_head_sample_status,
                full.scheduler_terminal_lm_head_top_token_id,
                full.scheduler_terminal_lm_head_sampled_token_id,
                cycle_tokens.len().max(1)
            );
        }
        let Some(samples) = terminal_cycle_samples(
            &full,
            &cycle_tokens,
            allow_tool_calls,
            token_decoder.as_mut(),
        ) else {
            let step_trace = real_full_request_trace(prompt_tokens + generated_token_ids.len(), 1)?;
            let mut completion = diagnostic_real_full_completion(&full, &step_trace);
            completion.metrics.cache_load_ms = cache_load_ms;
            completion.metrics.prefill_ms = (first_step_ms.unwrap_or(0.0) - cache_load_ms).max(0.0);
            completion.metrics.time_to_first_token_ms = first_step_ms;
            completion.metrics.decode_ms = (total_step_ms - first_step_ms.unwrap_or(0.0)).max(0.0);
            if let Some(metrics) = completion.metrics.real_full.as_mut() {
                mtp_acceptance_metrics.apply(metrics);
            }
            return Ok(completion);
        };

        let mut should_stop = false;
        for mut sample in samples
            .into_iter()
            .take(max_tokens - generated_token_ids.len())
        {
            let stop_allowed =
                !ignore_eos && generated_token_ids.len().saturating_add(1) >= min_tokens;
            if !stop_allowed {
                sample.stop = false;
            }
            generated_token_ids.push(sample.token_id);
            let mut sample = output_filter.apply(sample);
            if !stop_allowed {
                sample.stop = false;
            }
            if sample.clear_prior_content {
                content.clear();
                reasoning_content.clear();
                generated_reasoning_tokens = 0;
                stream_chunks.clear();
            }
            if !sample.reasoning_content.is_empty() {
                generated_reasoning_tokens += 1;
            }
            reasoning_content.push_str(&sample.reasoning_content);
            if !sample.content.is_empty() {
                content.push_str(&sample.content);
                stream_chunks.push(sample.content);
            }
            if sample.stop {
                should_stop = true;
                break;
            }
        }
        last_full = Some(full);
        if should_stop {
            stopped_by_model = true;
            break;
        }
    }

    let final_full =
        last_full.ok_or_else(|| runtime_error("real-full decode loop produced no steps"))?;
    let mut real_full_metrics = RequestDiagnosticEvidence::runtime_reported(&final_full)
        .then(|| real_full_runtime_diagnostic_metrics(&final_full));
    if let Some(metrics) = real_full_metrics.as_mut() {
        mtp_acceptance_metrics.apply(metrics);
    }
    Ok(BackendCompletion {
        content,
        reasoning_content: (!reasoning_content.is_empty()).then_some(reasoning_content),
        completion_tokens: Some(
            generated_token_ids
                .len()
                .saturating_sub(usize::from(stopped_by_model)),
        ),
        stream_chunks: Some(stream_chunks),
        metrics: BackendMetrics {
            cache_load_ms,
            prefill_ms: (first_step_ms.unwrap_or(0.0) - cache_load_ms).max(0.0),
            time_to_first_token_ms: first_step_ms,
            decode_ms: (total_step_ms - first_step_ms.unwrap_or(0.0)).max(0.0),
            reasoning_tokens: generated_reasoning_tokens,
            cached_prompt_tokens: prompt_tokens
                .saturating_sub(prefill_tokens_reported.saturating_add(1)),
            prefill_tokens: prefill_tokens_reported,
            prefill_chunk_count: prefill_chunks_reported,
            layerwave_prefill_rows: prefill_tokens_reported,
            layerwave_decode_rows: executed_decode_steps,
            real_full: real_full_metrics,
        },
    })
}

fn diagnostic_real_full_completion(
    full: &crate::RealFullInfo,
    trace: &trace::RealFullRequestTrace,
) -> BackendCompletion {
    let request = RequestDiagnosticEvidence::new(full, trace);
    BackendCompletion {
        content: real_full_diagnostic_content(full, trace),
        reasoning_content: None,
        completion_tokens: None,
        stream_chunks: None,
        metrics: BackendMetrics {
            cache_load_ms: 0.0,
            prefill_ms: 0.0,
            time_to_first_token_ms: None,
            decode_ms: 0.0,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            prefill_tokens: request.prefill_tokens,
            prefill_chunk_count: request.prefill_chunks,
            layerwave_prefill_rows: request.prefill_tokens,
            layerwave_decode_rows: request.decode_budget,
            real_full: Some(real_full_diagnostic_metrics(full, trace)),
        },
    }
}

fn real_full_with_multi_token_decode_blocker(
    full: &crate::RealFullInfo,
    decode_budget: usize,
) -> crate::RealFullInfo {
    let mut blocked = full.clone();
    blocked.status = "blocked".to_owned();
    blocked.blocker = format!(
        "real-full multi-token decode requires a live request executor: request_decode_budget={decode_budget}; preflight terminal lm_head sample covers one final decode row"
    );
    if !blocked
        .failed_requirements
        .iter()
        .any(|requirement| requirement == MULTI_TOKEN_DECODE_LOOP_REQUIREMENT)
    {
        blocked
            .failed_requirements
            .push(MULTI_TOKEN_DECODE_LOOP_REQUIREMENT.to_owned());
    }
    blocked
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TerminalSample {
    token_id: usize,
    content: String,
    stop: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FilteredTerminalSample {
    token_id: usize,
    content: String,
    reasoning_content: String,
    stop: bool,
    clear_prior_content: bool,
}

#[derive(Debug, Default)]
struct GlmAssistantOutputFilter {
    inside_thinking: bool,
    allow_tool_calls: bool,
}

impl GlmAssistantOutputFilter {
    fn new(allow_tool_calls: bool, inside_thinking: bool) -> Self {
        Self {
            inside_thinking,
            allow_tool_calls,
        }
    }

    fn apply(&mut self, sample: TerminalSample) -> FilteredTerminalSample {
        if sample.token_id == GLM_THINK_OPEN_TOKEN_ID {
            self.inside_thinking = true;
            return FilteredTerminalSample {
                token_id: sample.token_id,
                content: String::new(),
                reasoning_content: String::new(),
                stop: false,
                clear_prior_content: false,
            };
        }
        if sample.token_id == GLM_THINK_CLOSE_TOKEN_ID {
            let clear_prior_content = !self.inside_thinking;
            self.inside_thinking = false;
            return FilteredTerminalSample {
                token_id: sample.token_id,
                content: String::new(),
                reasoning_content: String::new(),
                stop: false,
                clear_prior_content,
            };
        }

        let (content, reasoning_content, stop_by_text, clear_prior_content) =
            self.filter_text_markers(&sample.content);
        FilteredTerminalSample {
            token_id: sample.token_id,
            content,
            reasoning_content,
            stop: sample.stop || stop_by_text,
            clear_prior_content,
        }
    }

    fn filter_text_markers(&mut self, text: &str) -> (String, String, bool, bool) {
        let mut visible = String::new();
        let mut reasoning = String::new();
        let mut offset = 0_usize;
        let mut stop = false;
        let mut clear_prior_content = false;
        while offset < text.len() {
            let rest = &text[offset..];
            let Some(marker) = next_glm_output_marker(rest, self.allow_tool_calls) else {
                if self.inside_thinking {
                    reasoning.push_str(rest);
                } else {
                    visible.push_str(rest);
                }
                break;
            };
            if self.inside_thinking {
                reasoning.push_str(&rest[..marker.index]);
            } else {
                visible.push_str(&rest[..marker.index]);
            }
            match marker.kind {
                GlmOutputMarkerKind::Stop => {
                    stop = true;
                    break;
                }
                GlmOutputMarkerKind::ThinkOpen => {
                    self.inside_thinking = true;
                }
                GlmOutputMarkerKind::ThinkClose => {
                    if !self.inside_thinking && !visible.is_empty() {
                        visible.clear();
                        clear_prior_content = true;
                    }
                    self.inside_thinking = false;
                }
            }
            offset += marker.index + marker.text.len();
        }
        (visible, reasoning, stop, clear_prior_content)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GlmOutputMarkerKind {
    Stop,
    ThinkOpen,
    ThinkClose,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GlmOutputMarker {
    index: usize,
    text: &'static str,
    kind: GlmOutputMarkerKind,
}

fn next_glm_output_marker(text: &str, allow_tool_calls: bool) -> Option<GlmOutputMarker> {
    GLM_CHAT_STOP_TEXTS
        .iter()
        .filter(|marker| !allow_tool_calls || !GLM_TOOL_CALL_TEXTS.contains(marker))
        .map(|marker| (*marker, GlmOutputMarkerKind::Stop))
        .chain([
            (GLM_THINK_OPEN_TEXT, GlmOutputMarkerKind::ThinkOpen),
            (GLM_THINK_CLOSE_TEXT, GlmOutputMarkerKind::ThinkClose),
        ])
        .filter_map(|(marker, kind)| {
            text.find(marker).map(|index| GlmOutputMarker {
                index,
                text: marker,
                kind,
            })
        })
        .min_by_key(|marker| marker.index)
}

fn terminal_sample(full: &crate::RealFullInfo, allow_tool_calls: bool) -> Option<TerminalSample> {
    let serve_fast_token = full.startup_diagnostic_mode == "serve-fast-token-embedding-lm-head";
    let request_scheduler_execution = full.startup_diagnostic_mode == "request-scheduler-execution";
    let blocked_by_overall_diagnostic = full.status == "blocked"
        || !full.blocker.trim().is_empty()
        || !full.failed_requirements.is_empty();
    // Do not surface partial scheduler samples as assistant text. They are useful
    // diagnostics, but user-visible content must come from a completed path.
    if !matches!(
        full.scheduler_terminal_lm_head_sample_status.as_str(),
        "passed" | "sampled"
    ) || blocked_by_overall_diagnostic
        || (!serve_fast_token
            && request_scheduler_execution
            && !full.scheduler_full_context_device_attention_complete)
        || !full.scheduler_terminal_lm_head_sample_passed
        || (!full.scheduler_terminal_lm_head_uses_final_decode_device_hidden && !serve_fast_token)
        || (!serve_fast_token && !full.scheduler_terminal_lm_head_covers_full_vocabulary)
        || (!serve_fast_token
            && full.scheduler_terminal_lm_head_logits_evaluated
                < full.scheduler_terminal_lm_head_vocab_size)
        || full.scheduler_terminal_lm_head_blocker.is_some()
    {
        return None;
    }

    let token_id = terminal_output_token_id(full)?;
    let stop_by_token = glm_chat_stop_token_id(token_id, allow_tool_calls);
    let content = if let Some(decoded) = terminal_output_token_text(full, token_id) {
        decoded
    } else if stop_by_token {
        String::new()
    } else {
        format!("glmrt-token:{token_id}")
    };
    let (content, stop_by_text) = trim_glm_chat_stop_text(&content, allow_tool_calls);
    Some(TerminalSample {
        token_id,
        content,
        stop: stop_by_token || stop_by_text,
    })
}

fn terminal_cycle_samples(
    full: &crate::RealFullInfo,
    generated_tokens: &[RealFullGeneratedToken],
    allow_tool_calls: bool,
    mut token_decoder: Option<&mut glmrt_loader::StreamingTokenDecoder>,
) -> Option<Vec<TerminalSample>> {
    let mut default_sample = terminal_sample(full, allow_tool_calls)?;
    if generated_tokens.is_empty() {
        decode_terminal_sample_streaming(&mut default_sample, token_decoder.as_deref_mut());
        return Some(vec![default_sample]);
    }

    Some(
        generated_tokens
            .iter()
            .map(|generated| {
                let stop_by_token = glm_chat_stop_token_id(generated.token_id, allow_tool_calls);
                let fallback_content = generated
                    .text
                    .clone()
                    .or_else(|| terminal_output_token_text(full, generated.token_id))
                    .unwrap_or_else(|| {
                        if stop_by_token {
                            String::new()
                        } else {
                            format!("glmrt-token:{}", generated.token_id)
                        }
                    });
                let mut sample = TerminalSample {
                    token_id: generated.token_id,
                    content: fallback_content,
                    stop: stop_by_token,
                };
                decode_terminal_sample_streaming(&mut sample, token_decoder.as_deref_mut());
                let (content, stop_by_text) =
                    trim_glm_chat_stop_text(&sample.content, allow_tool_calls);
                sample.content = content;
                sample.stop |= stop_by_text;
                sample
            })
            .collect(),
    )
}

fn real_full_streaming_token_decoder(
    state: &ApiState,
) -> Option<glmrt_loader::StreamingTokenDecoder> {
    let snapshot_path = state.config.real_full.as_ref()?.snapshot_path.as_deref()?;
    match glmrt_loader::streaming_token_decoder(Path::new(snapshot_path), false) {
        Ok(decoder) => Some(decoder),
        Err(error) => {
            eprintln!(
                "real_full_streaming_token_decoder_error snapshot={} error={error}",
                snapshot_path
            );
            None
        }
    }
}

fn decode_terminal_sample_streaming(
    sample: &mut TerminalSample,
    token_decoder: Option<&mut glmrt_loader::StreamingTokenDecoder>,
) {
    let Some(token_decoder) = token_decoder else {
        return;
    };
    let Ok(token_id) = u32::try_from(sample.token_id) else {
        return;
    };
    match token_decoder.step(token_id) {
        Ok(content) => sample.content = content.unwrap_or_default(),
        Err(error) => {
            eprintln!(
                "real_full_streaming_token_decode_error token_id={} error={error}",
                sample.token_id
            );
        }
    }
}

fn terminal_output_token_id(full: &crate::RealFullInfo) -> Option<usize> {
    full.scheduler_terminal_lm_head_sampled_token_id
        .or(full.scheduler_terminal_lm_head_top_token_id)
}

fn terminal_output_token_text(full: &crate::RealFullInfo, token_id: usize) -> Option<String> {
    if full.scheduler_terminal_lm_head_sampled_token_id == Some(token_id) {
        if let Some(decoded) = full
            .scheduler_terminal_lm_head_sampled_text
            .as_deref()
            .filter(|text| !text.is_empty())
        {
            return Some(decoded.to_owned());
        }
    }

    let snapshot_path = full.snapshot_path.as_deref()?;
    let token_id = u32::try_from(token_id).ok()?;
    glmrt_loader::decode_tokenizer_ids(Path::new(snapshot_path), &[token_id], false)
        .ok()
        .map(|summary| summary.text)
        .filter(|text| !text.is_empty())
}

fn glm_chat_stop_token_id(token_id: usize, allow_tool_calls: bool) -> bool {
    GLM_CHAT_STOP_TOKEN_IDS.contains(&token_id)
        && (!allow_tool_calls || !GLM_TOOL_CALL_TOKEN_IDS.contains(&token_id))
}

fn trim_glm_chat_stop_text(content: &str, allow_tool_calls: bool) -> (String, bool) {
    let Some((stop_index, _)) = GLM_CHAT_STOP_TEXTS
        .iter()
        .filter(|marker| !allow_tool_calls || !GLM_TOOL_CALL_TEXTS.contains(marker))
        .filter_map(|marker| content.find(marker).map(|index| (index, *marker)))
        .min_by_key(|(index, _)| *index)
    else {
        return (content.to_owned(), false);
    };
    (content[..stop_index].to_owned(), true)
}

fn real_full_diagnostic_content(
    full: &crate::RealFullInfo,
    trace: &trace::RealFullRequestTrace,
) -> String {
    let request = RequestDiagnosticEvidence::new(full, trace);
    format!(
            "real glm full status={} startup_diagnostic_mode={} tensors={} coordinator_resident_preload_status={} coordinator_resident_preload_selected_tensors={} coordinator_resident_preload_selected_bytes={} coordinator_resident_preload_loaded_bytes={} layers={} dense_layers={} sparse_layers={} kv_layout={} kv_bytes_per_token={} scheduler_iterations={} selected_layerwaves={} sparse_batches={} kv_reads={} committed_kv_writes={} tentative_kv_writes={} scheduler_numeric_progression_passed={} scheduler_numeric_progression_source_rows={} scheduler_numeric_progression_hidden_dim={} scheduler_numeric_progression_visible_checksum={} scheduler_numeric_progression_rejected_mtp_checksum={} scheduler_full_context_device_attention_complete={} scheduler_terminal_lm_head_sample_status={} scheduler_terminal_lm_head_sample_passed={} scheduler_terminal_lm_head_uses_final_decode_device_hidden={} scheduler_terminal_lm_head_covers_full_vocabulary={} scheduler_terminal_lm_head_logits_evaluated={} scheduler_terminal_lm_head_vocab_size={} scheduler_terminal_lm_head_top_token_id={:?} scheduler_terminal_lm_head_sampled_token_id={:?} scheduler_terminal_lm_head_sample_top_k={:?} scheduler_terminal_lm_head_sample_top_p={:?} scheduler_terminal_lm_head_argmax_backend={:?} scheduler_terminal_lm_head_sampler_backend={:?} scheduler_terminal_lm_head_blocker={:?} protocol={} decode_wire_request_bytes_per_touched_host={} decode_wire_response_bytes_per_touched_host={} prefill_wire_request_bytes_per_touched_host={} prefill_wire_response_bytes_per_touched_host={} mtp_wire_request_bytes_per_touched_host={} mtp_wire_response_bytes_per_touched_host={} decode_full_sparse_roundtrip_wire_bytes={} prefill_full_sparse_roundtrip_wire_bytes={} mtp_full_sparse_roundtrip_wire_bytes={} scheduler_sparse_tcp_dispatch_status={} scheduler_sparse_tcp_dispatch_targets={} scheduler_sparse_tcp_dispatch_sparse_layers={} scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer={} scheduler_sparse_tcp_dispatch_batches={} scheduler_sparse_tcp_dispatch_host_batches={} scheduler_sparse_tcp_dispatch_global_rows={} scheduler_sparse_tcp_dispatch_host_rows={} scheduler_sparse_tcp_dispatch_routes={} scheduler_sparse_tcp_dispatch_request_wire_bytes={} scheduler_sparse_tcp_dispatch_response_wire_bytes={} scheduler_sparse_tcp_dispatch_output_values={} scheduler_sparse_tcp_dispatch_output_finite_values={} scheduler_sparse_tcp_dispatch_output_nonzero_values={} scheduler_sparse_tcp_dispatch_output_checksum={} scheduler_sparse_tcp_dispatch_passed={} scheduler_sparse_tcp_dispatch_expected_real_executor_id={} scheduler_sparse_tcp_dispatch_response_executor_ids_observed={} scheduler_sparse_tcp_dispatch_real_executor_responses={} scheduler_sparse_tcp_dispatch_non_real_executor_responses={} scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4={} scheduler_sparse_tcp_dispatch_consumed_by_residual={} sampling_default_lm_head_chunk_passed={} sampling_default_lm_head_chunk_rows_scored={} sampling_default_lm_head_chunk_lm_head_bytes_read={} sampling_default_lm_head_chunk_top_token_id={:?} sampling_default_lm_head_chunk_top_logit={:?} sampling_default_lm_head_chunk_uses_real_dense_prefix={} sampling_default_lm_head_chunk_residual_source_dense_layers={} sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read={} sampling_default_lm_head_chunk_residual_after_checksum={:?} request_scheduler_summary_source={} request_coordinator_graphs(slots,captured,captures,launches)={}/{}/{}/{} request_prefill_tokens={} request_prefill_chunks={} request_decode_budget={} request_mtp_verify_rows={} request_mtp_accepted_rows={} request_candidate_layerwaves={} request_layerwaves={} request_deferred_layerwaves={} request_admitted_iterations={} request_sparse_batches={} request_expert_batch_rows={} request_expert_batch_routes={} request_expert_source_modes=[prefill_chunk,decode_step,mtp_verify] request_expert_prefill_rows={} request_expert_decode_rows={} request_expert_mtp_verify_rows={} request_expert_prefill_routes={} request_expert_decode_routes={} request_expert_mtp_verify_routes={} request_expert_source_modes_covered={} request_expert_route_entries_match_source_rows={} request_kv_reads={} request_committed_kv_writes={} request_tentative_kv_writes={} request_committed_mtp_writes={} request_discarded_mtp_writes={} request_backed_kv_writes={} request_backed_kv_bytes={} request_kv_reservation_bytes={} request_byte_backed_scheduler_trace={} request_numeric_progression_passed={} request_numeric_progression_source_rows={} request_numeric_progression_hidden_dim={} request_numeric_progression_selected_prefill_rows={} request_numeric_progression_selected_decode_rows={} request_numeric_progression_selected_mtp_rows={} request_numeric_progression_attention_value_updates={} request_numeric_progression_mlp_value_updates={} request_numeric_progression_visible_checksum={} request_numeric_progression_rejected_mtp_checksum={} blocker={:?} failed=[{}]",
            full.status,
            full.startup_diagnostic_mode,
            full.tensor_count,
            full.coordinator_resident_preload_status,
            full.coordinator_resident_preload_selected_tensors,
            full.coordinator_resident_preload_selected_bytes,
            full.coordinator_resident_preload_loaded_bytes,
            full.layer_count,
            full.dense_layer_count,
            full.sparse_layer_count,
            full.kv_layout,
            full.kv_bytes_per_token,
            full.scheduler_iterations,
            full.selected_layerwaves,
            full.sparse_expert_batches,
            full.kv_read_blocks,
            full.committed_kv_writes,
            full.tentative_kv_writes,
            full.scheduler_numeric_progression_passed,
            full.scheduler_numeric_progression_source_rows,
            full.scheduler_numeric_progression_hidden_dim,
            full.scheduler_numeric_progression_visible_checksum,
            full.scheduler_numeric_progression_rejected_mtp_checksum,
            full.scheduler_full_context_device_attention_complete,
            full.scheduler_terminal_lm_head_sample_status,
            full.scheduler_terminal_lm_head_sample_passed,
            full.scheduler_terminal_lm_head_uses_final_decode_device_hidden,
            full.scheduler_terminal_lm_head_covers_full_vocabulary,
            full.scheduler_terminal_lm_head_logits_evaluated,
            full.scheduler_terminal_lm_head_vocab_size,
            full.scheduler_terminal_lm_head_top_token_id,
            full.scheduler_terminal_lm_head_sampled_token_id,
            full.scheduler_terminal_lm_head_sample_top_k,
            full.scheduler_terminal_lm_head_sample_top_p,
            full.scheduler_terminal_lm_head_argmax_backend,
            full.scheduler_terminal_lm_head_sampler_backend,
            full.scheduler_terminal_lm_head_blocker,
            full.protocol,
            full.decode_wire_request_bytes_per_touched_host,
            full.decode_wire_response_bytes_per_touched_host,
            full.prefill_wire_request_bytes_per_touched_host,
            full.prefill_wire_response_bytes_per_touched_host,
            full.mtp_wire_request_bytes_per_touched_host,
            full.mtp_wire_response_bytes_per_touched_host,
            full.decode_full_sparse_roundtrip_wire_bytes,
            full.prefill_full_sparse_roundtrip_wire_bytes,
            full.mtp_full_sparse_roundtrip_wire_bytes,
            full.scheduler_sparse_tcp_dispatch_status,
            full.scheduler_sparse_tcp_dispatch_targets,
            full.scheduler_sparse_tcp_dispatch_sparse_layers,
            full.scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer,
            full.scheduler_sparse_tcp_dispatch_batches,
            full.scheduler_sparse_tcp_dispatch_host_batches,
            full.scheduler_sparse_tcp_dispatch_global_rows,
            full.scheduler_sparse_tcp_dispatch_host_rows,
            full.scheduler_sparse_tcp_dispatch_routes,
            full.scheduler_sparse_tcp_dispatch_request_wire_bytes,
            full.scheduler_sparse_tcp_dispatch_response_wire_bytes,
            full.scheduler_sparse_tcp_dispatch_output_values,
            full.scheduler_sparse_tcp_dispatch_output_finite_values,
            full.scheduler_sparse_tcp_dispatch_output_nonzero_values,
            full.scheduler_sparse_tcp_dispatch_output_checksum,
            full.scheduler_sparse_tcp_dispatch_passed,
            full.scheduler_sparse_tcp_dispatch_expected_real_executor_id,
            full.scheduler_sparse_tcp_dispatch_response_executor_ids_observed,
            full.scheduler_sparse_tcp_dispatch_real_executor_responses,
            full.scheduler_sparse_tcp_dispatch_non_real_executor_responses,
            full.scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4,
            full.scheduler_sparse_tcp_dispatch_consumed_by_residual,
            full.sampling_default_lm_head_chunk_passed,
            full.sampling_default_lm_head_chunk_rows_scored,
            full.sampling_default_lm_head_chunk_lm_head_bytes_read,
            full.sampling_default_lm_head_chunk_top_token_id,
            full.sampling_default_lm_head_chunk_top_logit,
            full.sampling_default_lm_head_chunk_uses_real_dense_prefix,
            full.sampling_default_lm_head_chunk_residual_source_dense_layers,
            full.sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read,
            full.sampling_default_lm_head_chunk_residual_after_checksum,
            request.summary_source,
            full.request_coordinator_graph_slots,
            full.request_coordinator_graph_captured_graphs,
            full.request_coordinator_graph_captures,
            full.request_coordinator_graph_launches,
            request.prefill_tokens,
            request.prefill_chunks,
            request.decode_budget,
            request.mtp_verify_rows,
            request.mtp_accepted_rows,
            request.candidate_layerwaves,
            request.layerwaves,
            request.deferred_layerwaves,
            request.admitted_iterations,
            request.sparse_batches,
            request.expert_batch_rows,
            request.expert_batch_routes,
            request.expert_prefill_rows,
            request.expert_decode_rows,
            request.expert_mtp_verify_rows,
            request.expert_prefill_routes,
            request.expert_decode_routes,
            request.expert_mtp_verify_routes,
            request.expert_source_modes_covered,
            request.expert_route_entries_match_source_rows,
            request.kv_read_blocks,
            request.committed_kv_writes,
            request.tentative_kv_writes,
            request.committed_mtp_writes,
            request.discarded_mtp_writes,
            request.backed_kv_writes,
            request.backed_kv_bytes,
            request.kv_reservation_bytes,
            request.byte_backed_scheduler_trace,
            request.numeric_progression_passed,
            request.numeric_progression_source_rows,
            request.numeric_progression_hidden_dim,
            request.numeric_progression_selected_prefill_rows,
            request.numeric_progression_selected_decode_rows,
            request.numeric_progression_selected_mtp_rows,
            request.numeric_progression_attention_value_updates,
            request.numeric_progression_mlp_value_updates,
            request.numeric_progression_visible_checksum,
            request.numeric_progression_rejected_mtp_checksum,
            full.blocker,
            full.failed_requirements.join(",")
        )
}

struct RequestDiagnosticEvidence {
    summary_source: &'static str,
    runtime_reported: bool,
    prefill_tokens: usize,
    prefill_chunks: usize,
    decode_budget: usize,
    mtp_verify_rows: usize,
    mtp_accepted_rows: usize,
    candidate_layerwaves: usize,
    deferred_layerwaves: usize,
    layerwaves: usize,
    admitted_iterations: usize,
    sparse_batches: usize,
    expert_batch_rows: usize,
    expert_batch_routes: usize,
    expert_prefill_rows: usize,
    expert_decode_rows: usize,
    expert_mtp_verify_rows: usize,
    expert_prefill_routes: usize,
    expert_decode_routes: usize,
    expert_mtp_verify_routes: usize,
    expert_source_modes_covered: bool,
    expert_route_entries_match_source_rows: bool,
    kv_read_blocks: usize,
    committed_kv_writes: usize,
    tentative_kv_writes: usize,
    committed_mtp_writes: usize,
    discarded_mtp_writes: usize,
    backed_kv_writes: usize,
    backed_kv_bytes: usize,
    kv_reservation_bytes: usize,
    byte_backed_scheduler_trace: bool,
    numeric_progression_passed: bool,
    numeric_progression_source_rows: usize,
    numeric_progression_hidden_dim: usize,
    numeric_progression_selected_prefill_rows: usize,
    numeric_progression_selected_decode_rows: usize,
    numeric_progression_selected_mtp_rows: usize,
    numeric_progression_attention_value_updates: usize,
    numeric_progression_mlp_value_updates: usize,
    numeric_progression_visible_checksum: f32,
    numeric_progression_rejected_mtp_checksum: f32,
}

impl RequestDiagnosticEvidence {
    fn runtime_reported(full: &crate::RealFullInfo) -> bool {
        full.startup_diagnostic_mode == "request-scheduler-execution"
    }

    fn from_runtime(full: &crate::RealFullInfo) -> Self {
        assert!(
            Self::runtime_reported(full),
            "runtime request evidence requires a scheduler execution report"
        );
        Self::new(full, &trace::RealFullRequestTrace::default())
    }

    fn new(full: &crate::RealFullInfo, trace: &trace::RealFullRequestTrace) -> Self {
        let runtime_reported = Self::runtime_reported(full);
        Self {
            summary_source: if runtime_reported {
                "runtime-scheduler-report"
            } else {
                "api-trace"
            },
            runtime_reported,
            prefill_tokens: if runtime_reported {
                full.request_prefill_tokens
            } else {
                trace.prefill_tokens
            },
            prefill_chunks: if runtime_reported {
                full.request_prefill_chunks
            } else {
                trace.prefill_chunks
            },
            decode_budget: if runtime_reported {
                full.request_decode_budget
            } else {
                trace.decode_budget
            },
            mtp_verify_rows: if runtime_reported {
                full.request_mtp_verify_rows
            } else {
                trace.mtp_verify_rows
            },
            mtp_accepted_rows: if runtime_reported {
                full.request_mtp_accepted_rows
            } else {
                trace.mtp_accepted_rows
            },
            candidate_layerwaves: if runtime_reported {
                full.request_candidate_layerwaves
            } else {
                trace.candidate_layerwaves
            },
            deferred_layerwaves: if runtime_reported {
                full.request_deferred_layerwaves
            } else {
                trace.deferred_layerwaves
            },
            layerwaves: if runtime_reported {
                full.selected_layerwaves
            } else {
                trace.layerwaves
            },
            admitted_iterations: if runtime_reported {
                full.scheduler_iterations
            } else {
                trace.admitted_iterations
            },
            sparse_batches: if runtime_reported {
                full.sparse_expert_batches
            } else {
                trace.sparse_batches
            },
            expert_batch_rows: if runtime_reported {
                full.request_expert_batch_rows
            } else {
                trace.expert_batch_rows
            },
            expert_batch_routes: if runtime_reported {
                full.request_expert_batch_routes
            } else {
                trace.expert_batch_routes
            },
            expert_prefill_rows: if runtime_reported {
                full.request_expert_prefill_rows
            } else {
                trace.expert_prefill_rows
            },
            expert_decode_rows: if runtime_reported {
                full.request_expert_decode_rows
            } else {
                trace.expert_decode_rows
            },
            expert_mtp_verify_rows: if runtime_reported {
                full.request_expert_mtp_verify_rows
            } else {
                trace.expert_mtp_verify_rows
            },
            expert_prefill_routes: if runtime_reported {
                full.request_expert_prefill_routes
            } else {
                trace.expert_prefill_routes
            },
            expert_decode_routes: if runtime_reported {
                full.request_expert_decode_routes
            } else {
                trace.expert_decode_routes
            },
            expert_mtp_verify_routes: if runtime_reported {
                full.request_expert_mtp_verify_routes
            } else {
                trace.expert_mtp_verify_routes
            },
            expert_source_modes_covered: if runtime_reported {
                full.request_expert_prefill_rows > 0
                    && full.request_expert_decode_rows > 0
                    && full.request_expert_mtp_verify_rows > 0
            } else {
                trace.expert_source_modes_covered
            },
            expert_route_entries_match_source_rows: if runtime_reported {
                full.request_expert_prefill_rows
                    + full.request_expert_decode_rows
                    + full.request_expert_mtp_verify_rows
                    == full.request_expert_batch_rows
                    && full.request_expert_prefill_routes
                        + full.request_expert_decode_routes
                        + full.request_expert_mtp_verify_routes
                        == full.request_expert_batch_routes
            } else {
                trace.expert_route_entries_match_source_rows
            },
            kv_read_blocks: if runtime_reported {
                full.kv_read_blocks
            } else {
                trace.kv_read_blocks
            },
            committed_kv_writes: if runtime_reported {
                full.committed_kv_writes
            } else {
                trace.committed_kv_writes
            },
            tentative_kv_writes: if runtime_reported {
                full.tentative_kv_writes
            } else {
                trace.tentative_kv_writes
            },
            committed_mtp_writes: if runtime_reported {
                full.request_committed_mtp_writes
            } else {
                trace.committed_mtp_writes
            },
            discarded_mtp_writes: if runtime_reported {
                full.request_discarded_mtp_writes
            } else {
                trace.discarded_mtp_writes
            },
            backed_kv_writes: if runtime_reported {
                full.request_backed_kv_writes
            } else {
                trace.backed_kv_writes
            },
            backed_kv_bytes: if runtime_reported {
                full.request_backed_kv_bytes
            } else {
                trace.backed_kv_bytes_after_discard
            },
            kv_reservation_bytes: if runtime_reported {
                full.request_kv_reservation_bytes
            } else {
                trace.kv_reservation_bytes
            },
            byte_backed_scheduler_trace: if runtime_reported {
                full.request_byte_backed_scheduler_trace
            } else {
                trace.byte_backed_scheduler_trace
            },
            numeric_progression_passed: if runtime_reported {
                full.scheduler_numeric_progression_passed
            } else {
                trace.request_numeric_progression_passed
            },
            numeric_progression_source_rows: if runtime_reported {
                full.scheduler_numeric_progression_source_rows
            } else {
                trace.request_numeric_progression_source_rows
            },
            numeric_progression_hidden_dim: if runtime_reported {
                full.scheduler_numeric_progression_hidden_dim
            } else {
                trace.request_numeric_progression_hidden_dim
            },
            numeric_progression_selected_prefill_rows: if runtime_reported {
                full.request_numeric_progression_selected_prefill_rows
            } else {
                trace.request_numeric_progression_selected_prefill_rows
            },
            numeric_progression_selected_decode_rows: if runtime_reported {
                full.request_numeric_progression_selected_decode_rows
            } else {
                trace.request_numeric_progression_selected_decode_rows
            },
            numeric_progression_selected_mtp_rows: if runtime_reported {
                full.request_numeric_progression_selected_mtp_rows
            } else {
                trace.request_numeric_progression_selected_mtp_rows
            },
            numeric_progression_attention_value_updates: if runtime_reported {
                full.request_numeric_progression_attention_value_updates
            } else {
                trace.request_numeric_progression_attention_value_updates
            },
            numeric_progression_mlp_value_updates: if runtime_reported {
                full.request_numeric_progression_mlp_value_updates
            } else {
                trace.request_numeric_progression_mlp_value_updates
            },
            numeric_progression_visible_checksum: if runtime_reported {
                full.scheduler_numeric_progression_visible_checksum
            } else {
                trace.request_numeric_progression_visible_checksum
            },
            numeric_progression_rejected_mtp_checksum: if runtime_reported {
                full.scheduler_numeric_progression_rejected_mtp_checksum
            } else {
                trace.request_numeric_progression_rejected_mtp_checksum
            },
        }
    }
}

fn real_full_diagnostic_metrics(
    full: &crate::RealFullInfo,
    trace: &trace::RealFullRequestTrace,
) -> RealFullDiagnosticMetrics {
    let request = RequestDiagnosticEvidence::new(full, trace);
    real_full_diagnostic_metrics_from_request(full, request)
}

fn real_full_runtime_diagnostic_metrics(full: &crate::RealFullInfo) -> RealFullDiagnosticMetrics {
    let request = RequestDiagnosticEvidence::from_runtime(full);
    real_full_diagnostic_metrics_from_request(full, request)
}

fn real_full_diagnostic_metrics_from_request(
    full: &crate::RealFullInfo,
    request: RequestDiagnosticEvidence,
) -> RealFullDiagnosticMetrics {
    RealFullDiagnosticMetrics {
        status: full.status.clone(),
        startup_diagnostic_mode: full.startup_diagnostic_mode.clone(),
        blocker: (!full.blocker.trim().is_empty()).then(|| full.blocker.clone()),
        failed_requirements: full.failed_requirements.clone(),
        scheduler_numeric_progression_passed: full.scheduler_numeric_progression_passed,
        scheduler_full_context_device_attention_complete: full
            .scheduler_full_context_device_attention_complete,
        scheduler_terminal_lm_head_sample_status: full
            .scheduler_terminal_lm_head_sample_status
            .clone(),
        scheduler_terminal_lm_head_sample_passed: full.scheduler_terminal_lm_head_sample_passed,
        scheduler_terminal_lm_head_uses_final_decode_device_hidden: full
            .scheduler_terminal_lm_head_uses_final_decode_device_hidden,
        scheduler_terminal_lm_head_covers_full_vocabulary: full
            .scheduler_terminal_lm_head_covers_full_vocabulary,
        scheduler_terminal_lm_head_logits_evaluated: full
            .scheduler_terminal_lm_head_logits_evaluated,
        scheduler_terminal_lm_head_vocab_size: full.scheduler_terminal_lm_head_vocab_size,
        scheduler_terminal_lm_head_top_token_id: full.scheduler_terminal_lm_head_top_token_id,
        scheduler_terminal_lm_head_sampled_token_id: full
            .scheduler_terminal_lm_head_sampled_token_id,
        scheduler_terminal_lm_head_sample_top_k: full.scheduler_terminal_lm_head_sample_top_k,
        scheduler_terminal_lm_head_sample_top_p: full.scheduler_terminal_lm_head_sample_top_p,
        scheduler_terminal_lm_head_blocker: full.scheduler_terminal_lm_head_blocker.clone(),
        scheduler_sparse_tcp_dispatch_status: full.scheduler_sparse_tcp_dispatch_status.clone(),
        scheduler_sparse_tcp_dispatch_targets: full.scheduler_sparse_tcp_dispatch_targets,
        scheduler_sparse_tcp_dispatch_sparse_layers: full
            .scheduler_sparse_tcp_dispatch_sparse_layers,
        scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: full
            .scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer,
        scheduler_sparse_tcp_dispatch_batches: full.scheduler_sparse_tcp_dispatch_batches,
        scheduler_sparse_tcp_dispatch_host_batches: full.scheduler_sparse_tcp_dispatch_host_batches,
        scheduler_sparse_tcp_dispatch_global_rows: full.scheduler_sparse_tcp_dispatch_global_rows,
        scheduler_sparse_tcp_dispatch_host_rows: full.scheduler_sparse_tcp_dispatch_host_rows,
        scheduler_sparse_tcp_dispatch_routes: full.scheduler_sparse_tcp_dispatch_routes,
        scheduler_sparse_tcp_dispatch_request_wire_bytes: full
            .scheduler_sparse_tcp_dispatch_request_wire_bytes,
        scheduler_sparse_tcp_dispatch_response_wire_bytes: full
            .scheduler_sparse_tcp_dispatch_response_wire_bytes,
        scheduler_sparse_tcp_dispatch_output_values: full
            .scheduler_sparse_tcp_dispatch_output_values,
        scheduler_sparse_tcp_dispatch_output_finite_values: full
            .scheduler_sparse_tcp_dispatch_output_finite_values,
        scheduler_sparse_tcp_dispatch_output_nonzero_values: full
            .scheduler_sparse_tcp_dispatch_output_nonzero_values,
        scheduler_sparse_tcp_dispatch_output_checksum: full
            .scheduler_sparse_tcp_dispatch_output_checksum,
        scheduler_sparse_tcp_dispatch_passed: full.scheduler_sparse_tcp_dispatch_passed,
        scheduler_sparse_tcp_dispatch_expected_real_executor_id: full
            .scheduler_sparse_tcp_dispatch_expected_real_executor_id,
        scheduler_sparse_tcp_dispatch_response_executor_ids_observed: full
            .scheduler_sparse_tcp_dispatch_response_executor_ids_observed,
        scheduler_sparse_tcp_dispatch_real_executor_responses: full
            .scheduler_sparse_tcp_dispatch_real_executor_responses,
        scheduler_sparse_tcp_dispatch_non_real_executor_responses: full
            .scheduler_sparse_tcp_dispatch_non_real_executor_responses,
        scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: full
            .scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4,
        scheduler_sparse_tcp_dispatch_consumed_by_residual: full
            .scheduler_sparse_tcp_dispatch_consumed_by_residual,
        request_scheduler_summary_runtime_reported: request.runtime_reported,
        request_prefill_tokens: request.prefill_tokens,
        request_prefill_chunks: request.prefill_chunks,
        request_kv_snapshot_restore_ms: full.request_kv_snapshot_restore_ms,
        request_decode_budget: request.decode_budget,
        request_mtp_verify_rows: request.mtp_verify_rows,
        request_mtp_accepted_rows: request.mtp_accepted_rows,
        mtp_verify_cycles: 0,
        mtp_draft_tokens: 0,
        mtp_accepted_draft_tokens: 0,
        mtp_emitted_tokens_from_verify: 0,
        mtp_full_match_cycles: 0,
        mtp_total_verify_cycle_ms: 0.0,
        mtp_draft_lengths: Vec::new(),
        mtp_accepted_draft_lengths: Vec::new(),
        mtp_verify_cycle_ms: Vec::new(),
        target_cycle_physical_m: Vec::new(),
        target_cycle_ms: Vec::new(),
        request_coordinator_graph_slots: full.request_coordinator_graph_slots,
        request_coordinator_graph_captured_graphs: full.request_coordinator_graph_captured_graphs,
        request_coordinator_graph_captures: full.request_coordinator_graph_captures,
        request_coordinator_graph_launches: full.request_coordinator_graph_launches,
        request_candidate_layerwaves: request.candidate_layerwaves,
        request_layerwaves: request.layerwaves,
        request_deferred_layerwaves: request.deferred_layerwaves,
        request_admitted_iterations: request.admitted_iterations,
        request_sparse_batches: request.sparse_batches,
        request_expert_batch_rows: request.expert_batch_rows,
        request_expert_batch_routes: request.expert_batch_routes,
        request_expert_prefill_rows: request.expert_prefill_rows,
        request_expert_decode_rows: request.expert_decode_rows,
        request_expert_mtp_verify_rows: request.expert_mtp_verify_rows,
        request_expert_prefill_routes: request.expert_prefill_routes,
        request_expert_decode_routes: request.expert_decode_routes,
        request_expert_mtp_verify_routes: request.expert_mtp_verify_routes,
        request_expert_source_modes_covered: request.expert_source_modes_covered,
        request_expert_route_entries_match_source_rows: request
            .expert_route_entries_match_source_rows,
        request_kv_reads: request.kv_read_blocks,
        request_committed_kv_writes: request.committed_kv_writes,
        request_tentative_kv_writes: request.tentative_kv_writes,
        request_committed_mtp_writes: request.committed_mtp_writes,
        request_discarded_mtp_writes: request.discarded_mtp_writes,
        request_backed_kv_writes: request.backed_kv_writes,
        request_backed_kv_bytes: request.backed_kv_bytes,
        request_kv_reservation_bytes: request.kv_reservation_bytes,
        request_byte_backed_scheduler_trace: request.byte_backed_scheduler_trace,
        request_numeric_progression_passed: request.numeric_progression_passed,
        request_numeric_progression_source_rows: request.numeric_progression_source_rows,
        request_numeric_progression_hidden_dim: request.numeric_progression_hidden_dim,
        request_numeric_progression_selected_prefill_rows: request
            .numeric_progression_selected_prefill_rows,
        request_numeric_progression_selected_decode_rows: request
            .numeric_progression_selected_decode_rows,
        request_numeric_progression_selected_mtp_rows: request
            .numeric_progression_selected_mtp_rows,
        request_numeric_progression_attention_value_updates: request
            .numeric_progression_attention_value_updates,
        request_numeric_progression_mlp_value_updates: request
            .numeric_progression_mlp_value_updates,
        request_numeric_progression_visible_checksum: request.numeric_progression_visible_checksum,
        request_numeric_progression_rejected_mtp_checksum: request
            .numeric_progression_rejected_mtp_checksum,
    }
}

#[cfg(test)]
mod token_limit_tests {
    use super::validate_real_full_request_token_limits;

    #[test]
    fn profile_limits_allow_flexible_input_output_splits_within_context() {
        validate_real_full_request_token_limits(300_000, 100_000, Some(400_000), Some(100_000))
            .expect("boundary request should pass");
        validate_real_full_request_token_limits(350_000, 50_000, Some(400_000), Some(100_000))
            .expect("larger input with smaller output should pass");

        let input_error =
            validate_real_full_request_token_limits(400_001, 1, Some(400_000), Some(100_000))
                .expect_err("oversized input should fail");
        assert_eq!(input_error.param.as_deref(), Some("messages"));
        assert!(input_error.message.contains("400001"));

        let output_error =
            validate_real_full_request_token_limits(1, 100_001, Some(400_000), Some(100_000))
                .expect_err("oversized output should fail");
        assert_eq!(output_error.param.as_deref(), Some("max_tokens"));
        assert!(output_error.message.contains("100001"));

        let combined_error =
            validate_real_full_request_token_limits(350_000, 50_001, Some(400_000), Some(100_000))
                .expect_err("combined sequence above context should fail");
        assert_eq!(combined_error.param.as_deref(), Some("max_tokens"));
        assert!(combined_error.message.contains("400001"));
    }

    #[test]
    fn absent_profile_limits_preserve_legacy_request_behavior() {
        validate_real_full_request_token_limits(500_000, 200_000, None, None)
            .expect("unset limits should defer to sequence-capacity admission");
    }
}
