use serde::{Deserialize, Serialize};
use std::collections::{hash_map::DefaultHasher, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tokio::sync::mpsc as tokio_mpsc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RealFullSamplingParams {
    temperature_bits: u32,
    top_p_bits: u32,
    top_k: usize,
    seed: u64,
}

impl RealFullSamplingParams {
    pub fn greedy() -> Self {
        Self::new(0.0, 1.0, 1, 0)
    }

    pub fn diagnostic() -> Self {
        Self::new(0.7, 0.95, 8, 0)
    }

    pub fn new(temperature: f32, top_p: f32, top_k: usize, seed: u64) -> Self {
        Self {
            temperature_bits: temperature.to_bits(),
            top_p_bits: top_p.to_bits(),
            top_k,
            seed,
        }
    }

    pub fn temperature(self) -> f32 {
        f32::from_bits(self.temperature_bits)
    }

    pub fn top_p(self) -> f32 {
        f32::from_bits(self.top_p_bits)
    }

    pub fn top_k(self) -> usize {
        self.top_k
    }

    pub fn seed(self) -> u64 {
        self.seed
    }

    pub fn is_greedy(self) -> bool {
        self.temperature() == 0.0 || self.top_k == 1
    }

    pub fn random_uniform(self, decode_step_index: usize) -> f32 {
        let step = u64::try_from(decode_step_index).unwrap_or(u64::MAX);
        let mut mixed = self
            .seed
            .wrapping_add(step.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        mixed ^= mixed >> 31;
        let mantissa = (mixed >> 40) as u32;
        mantissa as f32 * (1.0 / 16_777_216.0)
    }
}

impl Default for RealFullSamplingParams {
    fn default() -> Self {
        Self::greedy()
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum RealFullConstraintGrammar {
    Json,
    JsonSchema { schema_json: String, strict: bool },
    StructuralTag { structural_tag_json: String },
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RealFullConstraint {
    pub grammar: RealFullConstraintGrammar,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealFullRequest {
    pub request_index: u64,
    pub request_id: String,
    pub sequence_id: String,
    pub prompt: String,
    pub prompt_tokens: usize,
    pub prompt_token_ids: Option<Arc<Vec<usize>>>,
    pub vision_embeddings: Option<Arc<Vec<RealFullVisionEmbedding>>>,
    pub disable_speculation: bool,
    pub cached_prompt_tokens: usize,
    pub max_tokens: usize,
    pub generated_token_ids: Vec<usize>,
    pub decode_step_index: usize,
    pub decode_budget: usize,
    pub greedy_sampling: bool,
    pub sampling: RealFullSamplingParams,
    pub constraint: Option<Arc<RealFullConstraint>>,
}

impl RealFullRequest {
    pub fn new(
        request_index: u64,
        prompt: impl Into<String>,
        prompt_tokens: usize,
        max_tokens: usize,
    ) -> Self {
        Self::new_decode_step(
            request_index,
            prompt,
            prompt_tokens,
            max_tokens,
            Vec::new(),
            0,
            max_tokens,
        )
    }

    pub fn new_decode_step(
        request_index: u64,
        prompt: impl Into<String>,
        prompt_tokens: usize,
        max_tokens: usize,
        generated_token_ids: Vec<usize>,
        decode_step_index: usize,
        decode_budget: usize,
    ) -> Self {
        let request_id = format!("real-glm-full-api-{request_index}");
        Self::new_decode_step_for_sequence(
            request_index,
            format!("{request_id}-sequence"),
            prompt,
            prompt_tokens,
            max_tokens,
            generated_token_ids,
            decode_step_index,
            decode_budget,
        )
    }

    pub fn new_decode_step_for_sequence(
        request_index: u64,
        sequence_id: impl Into<String>,
        prompt: impl Into<String>,
        prompt_tokens: usize,
        max_tokens: usize,
        generated_token_ids: Vec<usize>,
        decode_step_index: usize,
        decode_budget: usize,
    ) -> Self {
        let request_id = format!("real-glm-full-api-{request_index}");
        Self {
            request_index,
            sequence_id: sequence_id.into(),
            request_id,
            prompt: prompt.into(),
            prompt_tokens,
            prompt_token_ids: None,
            vision_embeddings: None,
            disable_speculation: false,
            cached_prompt_tokens: 0,
            max_tokens,
            generated_token_ids,
            decode_step_index,
            decode_budget,
            greedy_sampling: true,
            sampling: RealFullSamplingParams::greedy(),
            constraint: None,
        }
    }

    pub fn with_greedy_sampling(mut self, greedy_sampling: bool) -> Self {
        self.greedy_sampling = greedy_sampling;
        self.sampling = if greedy_sampling {
            RealFullSamplingParams::greedy()
        } else {
            RealFullSamplingParams::diagnostic()
        };
        self
    }

    pub fn with_sampling(mut self, sampling: RealFullSamplingParams) -> Self {
        self.greedy_sampling = sampling.is_greedy();
        self.sampling = sampling;
        self
    }

    pub fn with_constraint(mut self, constraint: Option<Arc<RealFullConstraint>>) -> Self {
        self.constraint = constraint;
        self
    }

    pub fn with_cached_prompt_tokens(mut self, cached_prompt_tokens: usize) -> Self {
        self.cached_prompt_tokens = cached_prompt_tokens;
        self
    }

    pub fn with_prompt_token_ids(mut self, prompt_token_ids: Arc<Vec<usize>>) -> Self {
        self.prompt_tokens = prompt_token_ids.len();
        self.prompt_token_ids = Some(prompt_token_ids);
        self
    }

    pub fn with_vision_embeddings(
        mut self,
        vision_embeddings: Arc<Vec<RealFullVisionEmbedding>>,
    ) -> Self {
        self.vision_embeddings = Some(vision_embeddings);
        // The target KV is image-conditioned, while neither native MTP nor
        // dSpark currently receives the projected image rows. Keep the whole
        // sequence target-only rather than enabling an invalid draft path
        // after the initial prefill has released its host embeddings.
        self.disable_speculation = true;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealFullVisionEmbedding {
    /// Absolute row in the expanded prompt token sequence.
    pub token_start: usize,
    pub rows: usize,
    pub hidden_size: usize,
    pub image_sha256: String,
    pub hidden_bf16: Arc<Vec<u8>>,
}

pub trait RealFullRequestExecutor: Send + Sync + 'static {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String>;

    fn execute_real_full_decode_cycle(
        &self,
        request: RealFullRequest,
    ) -> Result<RealFullDecodeCycle, String> {
        self.execute_real_full_request(request)
            .map(RealFullDecodeCycle::single_token)
    }

    fn execute_real_full_decode_cycle_batch(
        &self,
        requests: Vec<RealFullRequest>,
    ) -> Vec<Result<RealFullDecodeCycle, String>> {
        requests
            .into_iter()
            .map(|request| self.execute_real_full_decode_cycle(request))
            .collect()
    }

    fn real_full_decode_cycle_batch_coalesce_timeout(
        &self,
        _request: &RealFullRequest,
    ) -> Option<Duration> {
        None
    }

    fn real_full_decode_cycle_batch_max_size(&self, _request: &RealFullRequest) -> usize {
        2
    }

    fn real_full_max_concurrent_sequences(&self) -> usize {
        usize::MAX
    }

    fn real_full_persistent_sequence_scheduling_enabled(&self) -> bool {
        false
    }

    fn real_full_retryable_admission_error(
        &self,
        _request: &RealFullRequest,
        _error: &str,
    ) -> bool {
        false
    }

    fn start_real_full_sequence(
        &self,
        _request: RealFullSequenceRequest,
    ) -> Result<tokio_mpsc::UnboundedReceiver<Result<RealFullSequenceCycle, String>>, String> {
        Err("real-full executor does not support persistent sequence scheduling".to_owned())
    }

    fn finish_real_full_sequence(&self, _sequence_id: &str) -> Result<(), String> {
        Ok(())
    }

    #[doc(hidden)]
    fn prewarm_batched_dspark_graphs(&self) -> Result<(), String> {
        Ok(())
    }

    #[doc(hidden)]
    fn prewarm_dflash2_dsa_lane_graphs(&self, _max_draft_tokens: usize) -> Result<(), String> {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RealFullGeneratedToken {
    pub token_id: usize,
    pub text: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RealFullDecodeCycle {
    pub info: RealFullInfo,
    pub generated_tokens: Vec<RealFullGeneratedToken>,
}

impl RealFullDecodeCycle {
    pub fn single_token(info: RealFullInfo) -> Self {
        Self {
            info,
            generated_tokens: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RealFullSequenceRequest {
    pub request: RealFullRequest,
    pub max_output_tokens: usize,
    pub min_output_tokens: usize,
    pub ignore_eos: bool,
    pub stop_token_ids: Vec<usize>,
    pub stop_texts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RealFullSequenceCycle {
    pub cycle: RealFullDecodeCycle,
    pub cycle_ms: f64,
    pub sequence_elapsed_ms: f64,
}

pub struct ThreadPinnedRealFullRequestExecutor {
    senders: Vec<mpsc::Sender<ThreadPinnedRealFullJob>>,
}

enum ThreadPinnedRealFullJob {
    Execute {
        request: RealFullRequest,
        reply: mpsc::Sender<Result<RealFullDecodeCycle, String>>,
    },
    Finish {
        sequence_id: String,
        reply: mpsc::Sender<Result<(), String>>,
    },
    StartSequence {
        request: RealFullSequenceRequest,
        events: tokio_mpsc::UnboundedSender<Result<RealFullSequenceCycle, String>>,
        submitted_at: Instant,
    },
}

struct ActiveRealFullSequence {
    request: RealFullSequenceRequest,
    events: tokio_mpsc::UnboundedSender<Result<RealFullSequenceCycle, String>>,
    started: Instant,
    next_request_index: u64,
}

impl ActiveRealFullSequence {
    fn new(
        request: RealFullSequenceRequest,
        events: tokio_mpsc::UnboundedSender<Result<RealFullSequenceCycle, String>>,
        submitted_at: Instant,
    ) -> Self {
        let next_request_index = request.request.request_index.saturating_add(1);
        Self {
            request,
            events,
            started: submitted_at,
            next_request_index,
        }
    }

    fn sequence_id(&self) -> &str {
        &self.request.request.sequence_id
    }

    fn apply_cycle(&mut self, mut cycle: RealFullDecodeCycle, cycle_ms: f64) -> bool {
        let remaining = self
            .request
            .max_output_tokens
            .saturating_sub(self.request.request.generated_token_ids.len());
        let mut candidates = std::mem::take(&mut cycle.generated_tokens);
        if candidates.is_empty() {
            if let Some(token_id) = cycle
                .info
                .scheduler_terminal_lm_head_sampled_token_id
                .or(cycle.info.scheduler_terminal_lm_head_top_token_id)
            {
                candidates.push(RealFullGeneratedToken {
                    token_id,
                    text: cycle.info.scheduler_terminal_lm_head_sampled_text.clone(),
                });
            }
        }

        let mut stopped = false;
        for generated in candidates.into_iter().take(remaining) {
            self.request
                .request
                .generated_token_ids
                .push(generated.token_id);
            let stop_by_token = self.request.stop_token_ids.contains(&generated.token_id);
            let stop_by_text = generated.text.as_deref().is_some_and(|text| {
                self.request
                    .stop_texts
                    .iter()
                    .any(|stop| !stop.is_empty() && text.contains(stop))
            });
            cycle.generated_tokens.push(generated);
            let stop_allowed = !self.request.ignore_eos
                && self.request.request.generated_token_ids.len() >= self.request.min_output_tokens;
            if stop_allowed && (stop_by_token || stop_by_text) {
                stopped = true;
                break;
            }
        }

        let invalid = cycle.info.status == "blocked"
            || !cycle.info.blocker.trim().is_empty()
            || !cycle.info.failed_requirements.is_empty()
            || cycle.generated_tokens.is_empty();
        let exhausted =
            self.request.request.generated_token_ids.len() >= self.request.max_output_tokens;
        let event = RealFullSequenceCycle {
            cycle,
            cycle_ms,
            sequence_elapsed_ms: self.started.elapsed().as_secs_f64() * 1_000.0,
        };
        if self.events.send(Ok(event)).is_err() || invalid || stopped || exhausted {
            return false;
        }

        self.request.request.request_index = self.next_request_index;
        self.next_request_index = self.next_request_index.saturating_add(1);
        self.request.request.request_id =
            format!("real-glm-full-api-{}", self.request.request.request_index);
        self.request.request.decode_step_index = self.request.request.generated_token_ids.len();
        // The projected image rows are consumed only while the initial prompt
        // is materialized. Recurrent cycles retain their effect in target KV
        // and should not pin tens of MiB of host BF16 data.
        self.request.request.vision_embeddings = None;
        true
    }
}

fn finish_active_real_full_sequence<E: RealFullRequestExecutor>(
    inner: &E,
    sequence: ActiveRealFullSequence,
) {
    if let Err(error) = inner.finish_real_full_sequence(sequence.sequence_id()) {
        eprintln!(
            "real_full_persistent_sequence_finish_error sequence_id={} error={error}",
            sequence.sequence_id()
        );
    }
}

fn prune_closed_active_real_full_sequences<E: RealFullRequestExecutor>(
    inner: &E,
    active_sequences: &mut Vec<ActiveRealFullSequence>,
) {
    let mut retained = Vec::with_capacity(active_sequences.len());
    for sequence in std::mem::take(active_sequences) {
        if sequence.events.is_closed() {
            finish_active_real_full_sequence(inner, sequence);
        } else {
            retained.push(sequence);
        }
    }
    *active_sequences = retained;
}

#[cfg(test)]
mod active_sequence_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct FinishRecordingExecutor {
        finishes: Arc<Mutex<Vec<String>>>,
    }

    impl RealFullRequestExecutor for FinishRecordingExecutor {
        fn execute_real_full_request(
            &self,
            _request: RealFullRequest,
        ) -> Result<RealFullInfo, String> {
            Err("closed-active pruning test must not execute another cycle".to_owned())
        }

        fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
            self.finishes.lock().unwrap().push(sequence_id.to_owned());
            Ok(())
        }
    }

    #[test]
    fn closed_active_sequence_is_pruned_before_another_cycle() {
        let finishes = Arc::new(Mutex::new(Vec::new()));
        let executor = FinishRecordingExecutor {
            finishes: Arc::clone(&finishes),
        };
        let (events, receiver) = tokio_mpsc::unbounded_channel();
        let request = RealFullSequenceRequest {
            request: RealFullRequest::new_decode_step_for_sequence(
                100,
                "cancelled-active-sequence",
                "prompt",
                1,
                1,
                vec![1],
                1,
                4,
            ),
            max_output_tokens: 4,
            min_output_tokens: 0,
            ignore_eos: false,
            stop_token_ids: Vec::new(),
            stop_texts: Vec::new(),
        };
        let mut active_sequences =
            vec![ActiveRealFullSequence::new(request, events, Instant::now())];
        drop(receiver);

        prune_closed_active_real_full_sequences(&executor, &mut active_sequences);

        assert!(active_sequences.is_empty());
        assert_eq!(
            finishes.lock().unwrap().as_slice(),
            &["cancelled-active-sequence".to_owned()]
        );
    }
}

fn execute_active_real_full_sequences<E: RealFullRequestExecutor>(
    inner: &E,
    mut sequences: Vec<ActiveRealFullSequence>,
) -> (Vec<ActiveRealFullSequence>, Vec<ActiveRealFullSequence>) {
    sequences.sort_by(|left, right| left.sequence_id().cmp(right.sequence_id()));
    let requests = sequences
        .iter()
        .map(|sequence| sequence.request.request.clone())
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut results = inner.execute_real_full_decode_cycle_batch(requests);
    let cycle_ms = started.elapsed().as_secs_f64() * 1_000.0;
    if results.len() != sequences.len() {
        let result_count = results.len();
        results = (0..sequences.len())
            .map(|_| {
                Err(format!(
                    "real-full persistent batch executor returned {result_count} results for {} requests",
                    sequences.len()
                ))
            })
            .collect();
    }

    let mut runnable = Vec::with_capacity(sequences.len());
    let mut retry_admission = Vec::new();
    for (mut sequence, result) in sequences.into_iter().zip(results) {
        match result {
            Ok(cycle) => {
                if sequence.apply_cycle(cycle, cycle_ms) {
                    runnable.push(sequence);
                } else {
                    finish_active_real_full_sequence(inner, sequence);
                }
            }
            Err(error) => {
                let initial_cycle = sequence.request.request.generated_token_ids.is_empty();
                if initial_cycle
                    && inner.real_full_retryable_admission_error(&sequence.request.request, &error)
                {
                    retry_admission.push(sequence);
                } else {
                    let _ = sequence.events.send(Err(error));
                    finish_active_real_full_sequence(inner, sequence);
                }
            }
        }
    }
    (runnable, retry_admission)
}

fn queue_thread_pinned_real_full_job(
    job: ThreadPinnedRealFullJob,
    active_sequence_count: usize,
    max_concurrent_sequences: usize,
    pending_sequences: &mut VecDeque<ActiveRealFullSequence>,
    deferred_jobs: &mut VecDeque<ThreadPinnedRealFullJob>,
) {
    match job {
        ThreadPinnedRealFullJob::StartSequence {
            request,
            events,
            submitted_at,
        } => {
            if active_sequence_count.saturating_add(pending_sequences.len())
                >= max_concurrent_sequences
            {
                let _ = events.send(Err(format!(
                    "real-full concurrent request limit exhausted: active={} queued={} max={max_concurrent_sequences}",
                    active_sequence_count,
                    pending_sequences.len(),
                )));
            } else {
                pending_sequences.push_back(ActiveRealFullSequence::new(
                    request,
                    events,
                    submitted_at,
                ));
            }
        }
        job => deferred_jobs.push_back(job),
    }
}

fn run_thread_pinned_real_full_worker<E: RealFullRequestExecutor>(
    inner: &E,
    receiver: mpsc::Receiver<ThreadPinnedRealFullJob>,
) {
    let mut active_sequences = Vec::<ActiveRealFullSequence>::new();
    let mut pending_sequences = VecDeque::<ActiveRealFullSequence>::new();
    let mut deferred_jobs = VecDeque::<ThreadPinnedRealFullJob>::new();
    let max_concurrent_sequences = inner.real_full_max_concurrent_sequences().max(1);

    loop {
        pending_sequences.retain(|sequence| !sequence.events.is_closed());
        while let Ok(job) = receiver.try_recv() {
            queue_thread_pinned_real_full_job(
                job,
                active_sequences.len(),
                max_concurrent_sequences,
                &mut pending_sequences,
                &mut deferred_jobs,
            );
        }

        if let Some(job) = deferred_jobs.pop_front() {
            match job {
                ThreadPinnedRealFullJob::Finish { sequence_id, reply } => {
                    active_sequences.retain(|sequence| sequence.sequence_id() != sequence_id);
                    pending_sequences.retain(|sequence| sequence.sequence_id() != sequence_id);
                    let result = inner.finish_real_full_sequence(&sequence_id);
                    let _ = reply.send(result);
                }
                ThreadPinnedRealFullJob::Execute { request, reply }
                    if !active_sequences.is_empty() || !pending_sequences.is_empty() =>
                {
                    let result = inner.execute_real_full_decode_cycle(request);
                    let _ = reply.send(result);
                }
                ThreadPinnedRealFullJob::Execute { request, reply } => {
                    let mut requests = vec![request];
                    let mut replies = vec![reply];
                    let coalesce_timeout =
                        inner.real_full_decode_cycle_batch_coalesce_timeout(&requests[0]);
                    let max_batch_size = inner
                        .real_full_decode_cycle_batch_max_size(&requests[0])
                        .max(1);
                    let mut deadline =
                        coalesce_timeout.and_then(|timeout| Instant::now().checked_add(timeout));
                    while requests.len() < max_batch_size {
                        let next_job =
                            deferred_jobs
                                .pop_front()
                                .or_else(|| match receiver.try_recv() {
                                    Ok(job) => Some(job),
                                    Err(mpsc::TryRecvError::Empty) => {
                                        deadline.and_then(|deadline| {
                                            let remaining =
                                                deadline.saturating_duration_since(Instant::now());
                                            if remaining.is_zero() {
                                                None
                                            } else {
                                                match receiver.recv_timeout(remaining) {
                                                    Ok(job) => Some(job),
                                                    Err(mpsc::RecvTimeoutError::Timeout)
                                                    | Err(mpsc::RecvTimeoutError::Disconnected) => {
                                                        None
                                                    }
                                                }
                                            }
                                        })
                                    }
                                    Err(mpsc::TryRecvError::Disconnected) => None,
                                });
                        match next_job {
                            Some(ThreadPinnedRealFullJob::Execute { request, reply }) => {
                                requests.push(request);
                                replies.push(reply);
                            }
                            Some(ThreadPinnedRealFullJob::Finish { sequence_id, reply }) => {
                                let result = inner.finish_real_full_sequence(&sequence_id);
                                let _ = reply.send(result);
                                deadline = coalesce_timeout
                                    .and_then(|timeout| Instant::now().checked_add(timeout));
                            }
                            Some(job @ ThreadPinnedRealFullJob::StartSequence { .. }) => {
                                queue_thread_pinned_real_full_job(
                                    job,
                                    active_sequences.len(),
                                    max_concurrent_sequences,
                                    &mut pending_sequences,
                                    &mut deferred_jobs,
                                );
                            }
                            None => break,
                        }
                    }
                    let mut request_replies = requests.into_iter().zip(replies).collect::<Vec<_>>();
                    request_replies
                        .sort_by(|left, right| left.0.sequence_id.cmp(&right.0.sequence_id));
                    let (requests, replies): (Vec<_>, Vec<_>) = request_replies.into_iter().unzip();
                    let mut results = inner.execute_real_full_decode_cycle_batch(requests);
                    if results.len() != replies.len() {
                        let result_count = results.len();
                        let reply_count = replies.len();
                        results = (0..replies.len())
                            .map(|_| {
                                Err(format!(
                                    "real-full batch executor returned {result_count} results for {reply_count} requests"
                                ))
                            })
                            .collect();
                    }
                    for (reply, result) in replies.into_iter().zip(results) {
                        let _ = reply.send(result);
                    }
                }
                ThreadPinnedRealFullJob::StartSequence { .. } => {
                    unreachable!("persistent sequence jobs are separated from direct worker jobs")
                }
            }
            continue;
        }

        // Membership changes only between complete decode/verify cycles. A
        // disconnected client must leave before the next recurrent batch,
        // rather than consuming one more target/dSpark cycle merely so the
        // subsequent event send can discover the closed receiver.
        prune_closed_active_real_full_sequences(inner, &mut active_sequences);

        let max_batch_size = active_sequences
            .first()
            .map(|sequence| {
                inner
                    .real_full_decode_cycle_batch_max_size(&sequence.request.request)
                    .max(1)
            })
            .or_else(|| {
                pending_sequences.front().map(|sequence| {
                    inner
                        .real_full_decode_cycle_batch_max_size(&sequence.request.request)
                        .max(1)
                })
            })
            .unwrap_or(1);

        if active_sequences.len() < max_batch_size {
            let available = max_batch_size - active_sequences.len();
            if active_sequences.is_empty() && pending_sequences.len() < available {
                let admission_timeout = pending_sequences.front().and_then(|sequence| {
                    inner.real_full_decode_cycle_batch_coalesce_timeout(&sequence.request.request)
                });
                if let Some(deadline) =
                    admission_timeout.and_then(|timeout| Instant::now().checked_add(timeout))
                {
                    while pending_sequences.len() < available {
                        let remaining = deadline.saturating_duration_since(Instant::now());
                        if remaining.is_zero() {
                            break;
                        }
                        match receiver.recv_timeout(remaining) {
                            Ok(job) => queue_thread_pinned_real_full_job(
                                job,
                                active_sequences.len(),
                                max_concurrent_sequences,
                                &mut pending_sequences,
                                &mut deferred_jobs,
                            ),
                            Err(mpsc::RecvTimeoutError::Timeout)
                            | Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                }
            }
            let mut admitted = Vec::with_capacity(available);
            while admitted.len() < available {
                let Some(sequence) = pending_sequences.pop_front() else {
                    break;
                };
                admitted.push(sequence);
            }
            if !admitted.is_empty() {
                let (runnable, retry_admission) =
                    execute_active_real_full_sequences(inner, admitted);
                let admitted_runnable = !runnable.is_empty();
                active_sequences.extend(runnable);
                for sequence in retry_admission {
                    pending_sequences.push_back(sequence);
                }
                if admitted_runnable {
                    continue;
                }
            }
        }

        if !active_sequences.is_empty() {
            let (runnable, retry_admission) =
                execute_active_real_full_sequences(inner, std::mem::take(&mut active_sequences));
            active_sequences = runnable;
            for sequence in retry_admission {
                pending_sequences.push_back(sequence);
            }
            continue;
        }

        if !pending_sequences.is_empty() {
            match receiver.recv_timeout(Duration::from_millis(1)) {
                Ok(job) => queue_thread_pinned_real_full_job(
                    job,
                    active_sequences.len(),
                    max_concurrent_sequences,
                    &mut pending_sequences,
                    &mut deferred_jobs,
                ),
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
            continue;
        }

        let job = match receiver.recv() {
            Ok(job) => job,
            Err(_) => break,
        };
        queue_thread_pinned_real_full_job(
            job,
            active_sequences.len(),
            max_concurrent_sequences,
            &mut pending_sequences,
            &mut deferred_jobs,
        );
    }

    for sequence in active_sequences {
        finish_active_real_full_sequence(inner, sequence);
    }
    for sequence in pending_sequences {
        finish_active_real_full_sequence(inner, sequence);
    }
}

impl ThreadPinnedRealFullRequestExecutor {
    pub fn spawn<E>(thread_name: impl Into<String>, inner: E) -> std::io::Result<Self>
    where
        E: RealFullRequestExecutor,
    {
        Self::spawn_pool(thread_name, inner, 1)
    }

    pub fn spawn_pool<E>(
        thread_name: impl Into<String>,
        inner: E,
        worker_count: usize,
    ) -> std::io::Result<Self>
    where
        E: RealFullRequestExecutor,
    {
        Self::spawn_pool_with_cpu_affinity(thread_name, inner, worker_count, &[])
    }

    pub fn spawn_pool_with_cpu_affinity<E>(
        thread_name: impl Into<String>,
        inner: E,
        worker_count: usize,
        worker_cpus: &[usize],
    ) -> std::io::Result<Self>
    where
        E: RealFullRequestExecutor,
    {
        if worker_count == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "thread-pinned real-full executor requires at least one worker",
            ));
        }
        if !worker_cpus.is_empty() && worker_cpus.len() != worker_count {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "thread-pinned real-full executor received {} CPU assignments for {worker_count} workers",
                    worker_cpus.len()
                ),
            ));
        }
        let thread_name = thread_name.into();
        let inner = Arc::new(inner);
        let mut senders = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let (sender, receiver) = mpsc::channel::<ThreadPinnedRealFullJob>();
            let (startup_sender, startup_receiver) = mpsc::sync_channel::<std::io::Result<()>>(1);
            let inner = Arc::clone(&inner);
            let worker_cpu = worker_cpus.get(worker_index).copied();
            let worker_name = if worker_count == 1 {
                thread_name.clone()
            } else {
                format!("{thread_name}-{worker_index}")
            };
            thread::Builder::new().name(worker_name).spawn(move || {
                let startup_result = worker_cpu
                    .map(glmrt_core::pin_current_thread_to_cpu)
                    .unwrap_or(Ok(()));
                let startup_succeeded = startup_result.is_ok();
                let _ = startup_sender.send(startup_result);
                if !startup_succeeded {
                    return;
                }
                run_thread_pinned_real_full_worker(inner.as_ref(), receiver);
            })?;
            startup_receiver.recv().map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!(
                        "thread-pinned real-full worker {worker_index} stopped during startup: {error}"
                    ),
                )
            })??;
            senders.push(sender);
        }
        Ok(Self { senders })
    }

    pub fn worker_count(&self) -> usize {
        self.senders.len()
    }

    pub fn execute_real_full_request_on_worker(
        &self,
        worker_index: usize,
        request: RealFullRequest,
    ) -> Result<RealFullInfo, String> {
        self.execute_real_full_decode_cycle_on_worker(worker_index, request)
            .map(|cycle| cycle.info)
    }

    pub fn execute_real_full_decode_cycle_on_worker(
        &self,
        worker_index: usize,
        request: RealFullRequest,
    ) -> Result<RealFullDecodeCycle, String> {
        let sender = self.senders.get(worker_index).ok_or_else(|| {
            format!(
                "real-full request worker index {worker_index} exceeds worker count {}",
                self.senders.len()
            )
        })?;
        let (reply, receiver) = mpsc::channel();
        sender
            .send(ThreadPinnedRealFullJob::Execute { request, reply })
            .map_err(|err| format!("real-full request worker is stopped: {err}"))?;
        receiver
            .recv()
            .map_err(|err| format!("real-full request worker stopped before replying: {err}"))?
    }

    pub fn finish_real_full_sequence_on_worker(
        &self,
        worker_index: usize,
        sequence_id: impl Into<String>,
    ) -> Result<(), String> {
        let sender = self.senders.get(worker_index).ok_or_else(|| {
            format!(
                "real-full request worker index {worker_index} exceeds worker count {}",
                self.senders.len()
            )
        })?;
        let (reply, receiver) = mpsc::channel();
        sender
            .send(ThreadPinnedRealFullJob::Finish {
                sequence_id: sequence_id.into(),
                reply,
            })
            .map_err(|err| format!("real-full request worker is stopped: {err}"))?;
        receiver
            .recv()
            .map_err(|err| format!("real-full request worker stopped before replying: {err}"))?
    }

    pub fn start_real_full_sequence_on_worker(
        &self,
        worker_index: usize,
        request: RealFullSequenceRequest,
    ) -> Result<tokio_mpsc::UnboundedReceiver<Result<RealFullSequenceCycle, String>>, String> {
        let sender = self.senders.get(worker_index).ok_or_else(|| {
            format!(
                "real-full request worker index {worker_index} exceeds worker count {}",
                self.senders.len()
            )
        })?;
        let (events, receiver) = tokio_mpsc::unbounded_channel();
        sender
            .send(ThreadPinnedRealFullJob::StartSequence {
                request,
                events,
                submitted_at: Instant::now(),
            })
            .map_err(|err| format!("real-full request worker is stopped: {err}"))?;
        Ok(receiver)
    }
}

impl RealFullRequestExecutor for ThreadPinnedRealFullRequestExecutor {
    fn execute_real_full_request(&self, request: RealFullRequest) -> Result<RealFullInfo, String> {
        self.execute_real_full_decode_cycle(request)
            .map(|cycle| cycle.info)
    }

    fn execute_real_full_decode_cycle(
        &self,
        request: RealFullRequest,
    ) -> Result<RealFullDecodeCycle, String> {
        let mut hasher = DefaultHasher::new();
        request.sequence_id.hash(&mut hasher);
        let worker_index = (hasher.finish() as usize) % self.senders.len();
        self.execute_real_full_decode_cycle_on_worker(worker_index, request)
    }

    fn start_real_full_sequence(
        &self,
        request: RealFullSequenceRequest,
    ) -> Result<tokio_mpsc::UnboundedReceiver<Result<RealFullSequenceCycle, String>>, String> {
        let mut hasher = DefaultHasher::new();
        request.request.sequence_id.hash(&mut hasher);
        let worker_index = (hasher.finish() as usize) % self.senders.len();
        self.start_real_full_sequence_on_worker(worker_index, request)
    }

    fn real_full_persistent_sequence_scheduling_enabled(&self) -> bool {
        true
    }

    fn finish_real_full_sequence(&self, sequence_id: &str) -> Result<(), String> {
        let mut hasher = DefaultHasher::new();
        sequence_id.hash(&mut hasher);
        let worker_index = (hasher.finish() as usize) % self.senders.len();
        self.finish_real_full_sequence_on_worker(worker_index, sequence_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealFullInfo {
    pub status: String,
    pub model_id: String,
    #[serde(default)]
    pub snapshot_path: Option<String>,
    pub catalog_hash: String,
    pub tensor_count: usize,
    pub startup_diagnostic_mode: String,
    pub coordinator_resident_preload_status: String,
    pub coordinator_resident_preload_selected_tensors: usize,
    pub coordinator_resident_preload_selected_bytes: u64,
    pub coordinator_resident_preload_loaded_bytes: u64,
    pub layer_count: usize,
    pub dense_layer_count: usize,
    pub sparse_layer_count: usize,
    pub kv_layout: String,
    pub kv_bytes_per_token: usize,
    #[serde(default)]
    pub request_prefill_tokens: usize,
    #[serde(default)]
    pub request_prefill_chunks: usize,
    #[serde(default)]
    pub request_kv_snapshot_restore_ms: f64,
    #[serde(default)]
    pub request_decode_budget: usize,
    #[serde(default)]
    pub request_mtp_verify_rows: usize,
    #[serde(default)]
    pub request_mtp_accepted_rows: usize,
    #[serde(default)]
    pub request_coordinator_graph_slots: usize,
    #[serde(default)]
    pub request_coordinator_graph_captured_graphs: usize,
    #[serde(default)]
    pub request_coordinator_graph_captures: usize,
    #[serde(default)]
    pub request_coordinator_graph_launches: usize,
    #[serde(default)]
    pub request_candidate_layerwaves: usize,
    #[serde(default)]
    pub request_deferred_layerwaves: usize,
    pub scheduler_iterations: usize,
    pub selected_layerwaves: usize,
    pub sparse_expert_batches: usize,
    #[serde(default)]
    pub request_expert_batch_rows: usize,
    #[serde(default)]
    pub request_expert_batch_routes: usize,
    #[serde(default)]
    pub request_expert_prefill_rows: usize,
    #[serde(default)]
    pub request_expert_decode_rows: usize,
    #[serde(default)]
    pub request_expert_mtp_verify_rows: usize,
    #[serde(default)]
    pub request_expert_prefill_routes: usize,
    #[serde(default)]
    pub request_expert_decode_routes: usize,
    #[serde(default)]
    pub request_expert_mtp_verify_routes: usize,
    pub kv_read_blocks: usize,
    pub committed_kv_writes: usize,
    pub tentative_kv_writes: usize,
    #[serde(default)]
    pub request_committed_mtp_writes: usize,
    #[serde(default)]
    pub request_discarded_mtp_writes: usize,
    #[serde(default)]
    pub request_backed_kv_writes: usize,
    #[serde(default)]
    pub request_backed_kv_bytes: usize,
    #[serde(default)]
    pub request_kv_reservation_bytes: usize,
    #[serde(default)]
    pub request_byte_backed_scheduler_trace: bool,
    pub scheduler_numeric_progression_passed: bool,
    pub scheduler_numeric_progression_source_rows: usize,
    pub scheduler_numeric_progression_hidden_dim: usize,
    pub scheduler_numeric_progression_visible_checksum: f32,
    pub scheduler_numeric_progression_rejected_mtp_checksum: f32,
    #[serde(default)]
    pub request_numeric_progression_selected_prefill_rows: usize,
    #[serde(default)]
    pub request_numeric_progression_selected_decode_rows: usize,
    #[serde(default)]
    pub request_numeric_progression_selected_mtp_rows: usize,
    #[serde(default)]
    pub request_numeric_progression_attention_value_updates: usize,
    #[serde(default)]
    pub request_numeric_progression_mlp_value_updates: usize,
    pub scheduler_full_context_device_attention_complete: bool,
    pub scheduler_terminal_lm_head_sample_status: String,
    pub scheduler_terminal_lm_head_sample_passed: bool,
    pub scheduler_terminal_lm_head_uses_final_decode_device_hidden: bool,
    pub scheduler_terminal_lm_head_covers_full_vocabulary: bool,
    pub scheduler_terminal_lm_head_logits_evaluated: usize,
    pub scheduler_terminal_lm_head_vocab_size: usize,
    pub scheduler_terminal_lm_head_top_token_id: Option<usize>,
    pub scheduler_terminal_lm_head_sampled_token_id: Option<usize>,
    #[serde(default)]
    pub scheduler_terminal_lm_head_sampled_text: Option<String>,
    pub scheduler_terminal_lm_head_sample_top_k: Option<usize>,
    pub scheduler_terminal_lm_head_sample_top_p: Option<f32>,
    pub scheduler_terminal_lm_head_argmax_backend: Option<String>,
    pub scheduler_terminal_lm_head_sampler_backend: Option<String>,
    pub scheduler_terminal_lm_head_blocker: Option<String>,
    pub protocol: String,
    pub decode_wire_request_bytes_per_touched_host: usize,
    pub decode_wire_response_bytes_per_touched_host: usize,
    pub prefill_wire_request_bytes_per_touched_host: usize,
    pub prefill_wire_response_bytes_per_touched_host: usize,
    pub mtp_wire_request_bytes_per_touched_host: usize,
    pub mtp_wire_response_bytes_per_touched_host: usize,
    pub decode_full_sparse_roundtrip_wire_bytes: usize,
    pub prefill_full_sparse_roundtrip_wire_bytes: usize,
    pub mtp_full_sparse_roundtrip_wire_bytes: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_status: String,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_targets: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_sparse_layers: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_iterations_per_sparse_layer: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_batches: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_host_batches: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_global_rows: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_host_rows: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_routes: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_request_wire_bytes: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_response_wire_bytes: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_output_values: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_output_finite_values: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_output_nonzero_values: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_output_checksum: f64,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_passed: bool,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_expected_real_executor_id: u64,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_response_executor_ids_observed: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_real_executor_responses: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_non_real_executor_responses: usize,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_all_responses_real_nvfp4: bool,
    #[serde(default)]
    pub scheduler_sparse_tcp_dispatch_consumed_by_residual: bool,
    pub sampling_default_lm_head_chunk_passed: bool,
    pub sampling_default_lm_head_chunk_rows_scored: usize,
    pub sampling_default_lm_head_chunk_lm_head_bytes_read: u64,
    pub sampling_default_lm_head_chunk_top_token_id: Option<usize>,
    pub sampling_default_lm_head_chunk_top_logit: Option<f32>,
    pub sampling_default_lm_head_chunk_uses_real_dense_prefix: bool,
    pub sampling_default_lm_head_chunk_residual_source_dense_layers: usize,
    pub sampling_default_lm_head_chunk_residual_source_dense_weight_bytes_read: u64,
    pub sampling_default_lm_head_chunk_residual_after_checksum: Option<f64>,
    pub blocker: String,
    pub failed_requirements: Vec<String>,
}
