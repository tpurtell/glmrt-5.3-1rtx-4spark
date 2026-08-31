# GLM-5.3 calibrated EXL3 K4 quantization

This directory contains the reproducible, restartable quantization path for the
GLMRT GLM-5.3 EXL3 K4 model. It quantizes only the routed experts in decoder
layers 3 through 77 to native EXL3/MCG tensors. All other tensors remain in
their source format. The generated artifact is intended for GLMRT's TP4 Spark
EXL3 loader and does not keep a second resident expert representation.

The production topology is exactly two compute-capability 12.0 coordinator
GPUs. The image, Python packages, Rust toolchain, GPTQModel fork, input model,
corpus, hardware identities, and numerical recipe are all content-bound in the
preflight report and execution plan. A resume fails closed if any immutable
input changes.

## GLM-5.3 K4 production recipe

The commands below are the complete GLM-5.3 production runbook:

- source: the exact `zai-org/GLM-5.3` snapshot;
- bitrate: pass `--bits 4` explicitly;
- source format: block-128x128 FP8 E4M3, which the plan validates together
  with every routed projection's scale tensor;
- native Spark validation: always pass `--trellis-bits 4`;
- Spark tile tuning: pass `--trellis-bits 4` to
  `bench_b12x_spark_exl3_tiles.py`; it maps each observed live M to the K4
  production capacity rather than timing an unrelated exact-M specialization.
  Use `--rows all-aot` for every compiled capacity or `--rows required-native`
  for the complete 44-row K4 live qualification surface;
- checkpoint-only Spark qualification: pass
  `--model-id wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1` to
  `build_exl3_projection_catalog.py`; the builder validates K4 trellis shapes
  and emits the K4 recipe;
- output identity: publish only as
  `wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1`;
- artifact validation: this plan binds both tokenizer files directly, so omit
  the obsolete `--tokenizer-attestation` argument;
- model-card template: `quantization/GLM53_EXL3_MODEL_CARD.md`;
- serving qualification: run
  `python/tools/validate_glm53_exl3_serving_qualification.py`, which compares
  matched native-MTP and DFlash2 runs and selects the measured agentic default;
- model-card renderer: `python/tools/render_glm53_exl3_model_card.py`.

The validated v1 corpus contains 1,441 examples and 1,082,141 prompt tokens.
Keep a slow, read-only source snapshot on scratch if necessary, but put the
rolling run frontier, projection checkpoints, active-layer staging, offload
state, and final output on the fast local NVMe. A representative fresh command
has this shape (omit `--resume` until resuming an already-bound plan):

```bash
python -X faulthandler quantization/quantize_glm52_gptqmodel.py \
  --snapshot /path/to/zai-org--GLM-5.3/snapshots/REVISION \
  --calibration-jsonl /fast-nvme/calibration/calibration.jsonl \
  --calibration-manifest /fast-nvme/calibration/manifest.json \
  --preflight-report /fast-nvme/quant-state/coordinator-preflight.json \
  --output /fast-nvme/models/GLM-5.3-EXL3-K4-calibrated-v1 \
  --run-state-dir /fast-nvme/quant-state/run-state \
  --projection-checkpoint-dir /fast-nvme/quant-state/projection-checkpoints \
  --offload-dir /fast-nvme/quant-state/offload \
  --bits 4
```

Qualify every production Spark rank against the pinned SparkInfer reference
with K4 selected explicitly:

```bash
native_evidence_root=/path/to/native-evidence
model_snapshot=/path/to/finalized-exl3-model
trellis_bits=4
expert_slot_fingerprint=$(jq -er '.fingerprints.expert_slot | select(test("^[0-9a-f]{64}$"))' \
  "$qualification_root/mtp-deployment.json")
mkdir -p "$native_evidence_root"
for tp_rank in 0 1 2 3; do
  PYTHONPATH=third_party/sparkinfer \
  python3 python/tools/validate_b12x_exl3_native.py \
    --native-library /path/to/libglmrt_native.so \
    --expert-slot-fingerprint "$expert_slot_fingerprint" \
    --trellis-bits "$trellis_bits" \
    --model-snapshot "$model_snapshot" \
    --layer-id 3 --tp-rank "$tp_rank" \
    --rows 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,64,128,129,256,257,512,513,1024,1025,2048,2049,2064 \
    --output "$native_evidence_root/native-tp${tp_rank}.json"
done
```

The final publication gate requires the correct GLM-5.3 `config.json`, a full
standalone `quantize_config.json` containing all 57,600 `tensor_storage`
records and all 230,400 stored-tensor descriptions, and exact agreement
between its compact declaration and `config.json.quantization_config`. For
GLM-5.3 that embedded declaration contains exactly `quant_method`, `format`,
`checkpoint_format`, and `bits`; calibration provenance, the error ledger,
module selection, storage details, and every other quantization field remain
only in `quantize_config.json`. The publication gate also rejects a main
`config.json` larger than 128 KiB. The Hub verifier repeats that agreement
check against the exact publication bytes, proves those bytes are the resolved
public revision through Hub Git/LFS object identities, requires the Hub API to
parse the compact EXL3 declaration, and checks the rendered model/tree/config
pages for visible errors. Matching local intent alone is not sufficient.

Publication retains `meta.ds4rt_error_ledger` byte-for-structure and every
other calibration or quantization field in the standalone file. It removes
only redundant top-level GPTQModel execution controls (for example the active
offload and CUDA-placement controls), which the pinned serializer normally
omits before the artifact is written. Content-bound paths and execution facts
inside the immutable calibration ledger are not stripped.

Treat the quantizer's first completed directory as a raw export. After the
quantizer has exited successfully, use this complete GLM-5.3 acceptance block:

```bash
raw_output_root=/fast-nvme/models/GLM-5.3-EXL3-K4-calibrated-v1
output_root=/fast-nvme/models/GLM-5.3-EXL3-K4-v1
state_root=/fast-nvme/quant-state
quant_image=glmrt-quant-coordinator:exl3-k4
model_id=wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1
source_snapshot=/path/to/zai-org--GLM-5.3/snapshots/REVISION
run_state="$state_root/run-state"
projection_root="$state_root/projection-checkpoints"
quant_evidence_report="$state_root/glm53-exl3-k4-quant-evidence.json"
artifact_report="$state_root/glm53-exl3-k4-artifact-validation.json"

# Acceptance mode intentionally omits --allow-incomplete,
# --skip-tensor-hashes, and --live-journal-snapshot.
python3 python/tools/validate_glm52_exl3_quant_evidence.py \
  --plan "$run_state/glmrt-gptqmodel-plan.json" \
  --projection-checkpoint-dir "$projection_root" \
  --error-journal "$run_state/.glmrt-exl3-error-journal.jsonl" \
  --output "$quant_evidence_report"

# The quantization image runs as root. After it has exited successfully,
# normalize only the atomically published raw artifact so the host can create
# protected hard links without duplicating its payload. Do not run this while
# the quantizer is active and do not broaden the mounted or chowned path.
raw_parent=$(dirname "$raw_output_root")
raw_name=$(basename "$raw_output_root")
host_uid=$(id -u)
host_gid=$(id -g)
docker run --rm --network none --ipc=none \
  --entrypoint /usr/bin/chown \
  -v "$raw_parent:/glmrt-models" \
  "$quant_image" \
  -R "$host_uid:$host_gid" "/glmrt-models/$raw_name"

python3 python/tools/finalize_glm5_exl3_artifact.py \
  --artifact "$raw_output_root" \
  --output "$output_root" \
  --report "$state_root/glm53-exl3-k4-finalization.json"

# GLM-5.3 plans bind both tokenizer files directly. Deliberately do not pass
# a separate tokenizer-attestation option here.
python3 python/tools/validate_glm52_exl3_artifact.py \
  --artifact "$output_root" \
  --source-snapshot "$source_snapshot" \
  --projection-checkpoint-dir "$projection_root" \
  --verify-artifact-file-hashes \
  --output "$artifact_report"
```

This is not a second weight copy: every unchanged payload is a hard link. The
tool removes Transformers serialization drift (including the historical
GLM-5 `head_dim` rewrite), preserves every source config field value-for-value,
restores the exact tokenizer and generation metadata, substitutes only the
compact EXL3 declaration, retains the complete standalone
`quantize_config.json.tensor_storage`, and rebinds both artifact manifests.
It deliberately accepts both the metadata-rich compact declaration emitted by
the currently running quantization image and the older GPTQModel raw
serialization that duplicated the complete `tensor_storage` object into
`config.json`, but only when the embedded object agrees exactly with the
standalone file. The finalized GLM-5.3 artifact always reduces that declaration
to the four discovery fields above. It also proves that the standalone
`meta.ds4rt_error_ledger` is exactly the immutable plan provenance plus any
content-bound execution upgrade.
Only `output_root` is eligible for the downstream commands below.

Stage the accepted artifact under its production model ID, then synchronize
that exact content revision to every Spark:

```bash
python3 python/tools/stage_glm52_exl3_hf_snapshot.py \
  --artifact "$output_root" \
  --validation-report "$artifact_report" \
  --quant-evidence-report "$quant_evidence_report" \
  --model-id "$model_id" \
  --update-ref

python3 python/tools/sync_glm52_exl3_hf_snapshot.py \
  --model-id "$model_id" \
  --hosts ostrich,dodo,emu,kiwi \
  --output "$state_root/glm53-exl3-k4-spark-sync.json"
```

For GLM-5.3, preserve this order; the block above provides the exact artifact
acceptance commands and later sections provide the serving commands:

1. Wait for the quantizer to atomically publish `raw_output_root`.
2. Validate the complete 57,600-record journal and projection store with full
   tensor hashing; an `--allow-incomplete` report is never release evidence.
3. Run the hard-linked finalizer above, then validate `output_root` against the
   pinned source snapshot without a separate tokenizer attestation.
4. Stage the validated artifact under
   `wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1` and synchronize that exact content
   revision to all four Sparks.
5. Build one final WIP slot, validate all four K4 TP ranks, tune the real K4
   tiles, and rerun the DFlash2 top-k/selector/body gates until the serving
   warp profile exactly matches every winner.
6. Run all seven fresh-process DFlash2 width trials, then fresh native-MTP and
   selected-width DFlash2 final arms, followed by the matching DFlash2
   preflight and serving qualifier.
7. Render the model card and create the standard-only publication tree. Upload
   only that tree to the exact repository above, verify its website/API and
   remote object inventory, then materialize the resolved Hub commit into the
   local HF cache without downloading a second copy of the model. Revalidate
   both configuration files, all 57,600 storage records, all 230,400 stored
   tensor descriptions, the inventory, manifests, and hashes from that layout.

### GLM-5.3 final serving evidence

Collect native-MTP and DFlash2 evidence from the same final WIP slot, balanced
profile, model revision, tokenizer, prompt seeds, and 400 W coordinator power
limit. Preserve each deployment and startup report exactly as described in the
general paired runbook below. In addition to the seven-type blend, Orchid,
full cache-aware 0/32K/64K/128K/256K by +1K/+2K/+4K/+8K/+16K/+32K prefill
grid, and tool evaluation, each arm must collect a prompt-bound C1/C2/C4 code
curve and the complete long-context needle grid:

Install the exact draft checkpoint in the same Hugging Face cache mounted into
the coordinator WIP/release container. The draft runtime aliases the accepted
target embedding and LM head, so it needs only this pinned 4.9-GB config/weight
snapshot and does not create another copy of those target tensors:

```bash
hf download incoai/GLM-5.3-DFlash2 \
  --revision 425aa615ce320caac34400208b30808c8f14f76c
```

If that exact repository cache already exists under an alternate `HF_HOME`,
copy its complete `models--incoai--GLM-5.3-DFlash2` cache directory (blobs,
snapshot symlinks, trees, and refs) into the coordinator container's mounted
HF cache instead of downloading the 4.9-GB payload again. Verify that the
snapshot name is the literal revision above and that `model.safetensors`
resolves to blob
`3105f14043bef642baa49a7d533fdf0b8b2895737ec84b6305601da662656161`.
The accepted five-file snapshot totals 4,919,153,766 bytes;
`config.json` has SHA-256
`f59e1da17d41d24a1aba588aecee1607788adb34a03805f2c883add8ca954e9b`.
The runtime additionally validates all 96 BF16 tensor names, shapes, offsets,
and their 4,918,848,512-byte payload before preload. It also rejects a regular
file or any snapshot symlink that does not resolve to the exact LFS blob above,
so this identity check adds no second 4.9-GB startup read.

The resolver fails closed if that exact revision is absent. If `HF_HOME` is
changed, create or recreate the WIP containers with that value so the mounted
cache and resolved profile refer to the same files; do not inject a one-off
snapshot path into a publication run.

Before collecting the final DFlash2 arm, sweep `DFLASH2_FIXED_DRAFTS=1` through
`7` with the matched seven-type blend and tool evaluation. Use
`SPECULATION=plain` as the non-speculative diagnostic control. Choose by tool
points first, code decode throughput second, and seven-type weighted decode
throughput third; prefer the narrower width only on an exact tie. Then rerun
the complete final arm at that width. Select the final agentic default between
native MTP and DFlash2 by tool points, code decode throughput, the C1/C2/C4
code-throughput geometric mean, and seven-type weighted decode throughput, in
that order.

After quantization releases the coordinator GPU, first select the 154,880-way
top-k backend across every C1/C2/C4, K1-K7 row case. The signed report requires
exact candidates and values for both the initial and changed graph inputs and
only advances a non-Torch backend when its equal-case aggregate median wins by
at least 1%. Each timing graph contains repeated top-k nodes so a one-node graph
replay launch cannot hide the kernel difference. Set `DFLASH2_TOPK_BACKEND` in
the launch configuration to the report's `selected_backend` before any width
trial; the complete service trials remain the acceptance gate.

Then performance-gate the fused candidate selector against the retained split
transition/argmax reference on the pinned checkpoint's real codebooks. This
microbenchmark covers C1/C2/C4, K1-K7, and both Torch `int64` and FlashInfer
`int32` candidate indices; it interleaves the timing order and refuses any
token mismatch. Preserve its content-bound report. Performance-gate the body
convolution/residual/RMS fusion the same way against its exact three-kernel
reference and sweep four versus eight warps. After timing layer 0, the body
gate also requires exact residual and normalized output for both epilogues in
all six real draft layers over the full C1/C2/C4, K1-K7 surface. Both fusions
must win the production cases before the final binary is accepted; the
microbench gate requires at least a 1% median speedup, after which full-cycle
service measurements remain authoritative. The reports also reject unless every
measured winning warp count matches the helper used by the serving graph. If a
winner changes, update that runtime helper and rerun the sweep so the report is
bound to the code that will ship. Otherwise tune the fusion or restore the
split path before running the service-width sweep.

```bash
python3 python/tools/tune_dflash2_topk.py \
  --snapshot /path/to/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c \
  --concurrency 1,2,4 --widths 1,2,3,4,5,6,7 \
  --output "$qualification_root/dflash2-topk-tuning.json"

python3 python/tools/tune_dflash2_selector.py \
  --snapshot /path/to/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c \
  --concurrency 1,2,4 --widths 1,2,3,4,5,6,7 \
  --candidate-dtypes both --fused-warps 4,8 \
  --output "$qualification_root/dflash2-selector-tuning.json"

python3 python/tools/tune_dflash2_body_fusion.py \
  --snapshot /path/to/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c \
  --concurrency 1,2,4 --widths 1,2,3,4,5,6,7 \
  --fused-warps 4,8 \
  --output "$qualification_root/dflash2-body-fusion-tuning.json"
```

The BF16 DFlash2 body remains the required baseline. As a separate follow-on,
measure the single-residency W8A16 candidate on every live C1/C2/C4, K1-K7
row shape. This first tunes layer 0 and then validates the selected tile on all
six real draft layers. A projection class advances only if every row/layer
case wins by at least 1% and passes its numerical sanity gate. Advancing means
it is eligible for implementation and full held-out acceptance testing, not
that the tool may modify the serving format. Never retain both its BF16 and W8
weight after an implementation is selected.

```bash
python3 python/tools/tune_dflash2_w8a16_body.py \
  --native-library /path/to/libglmrt_native.so \
  --snapshot /path/to/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c \
  --layers 0,1,2,3,4,5 --rows dflash \
  --output "$qualification_root/dflash2-w8a16-body-tuning.json"
```

If no projection class advances, stop there. If one does, add only its measured
small-row AOT shapes and one-format loader/capture contract, then compare full
DFlash2 acceptance, tool score, verify-cycle cost, and decode throughput with
the BF16 arm before promotion.

`run.sh` and `scripts/run-wip.sh` bind the selected width and configured top-k
backend into the resolved profile, deployment fingerprint, signed deployment
evidence, qualification report, and rendered model card; do not inject engine
environment variables manually for publication measurements.

Preserve each preliminary trial as
`dflash2-width-$width-{deployment,blended,tool-eval}.{json,jsonl,json}`. The
qualifier revalidates every raw file, requires all seven trials to use the same
binary/model/power and exact prompt fixtures, calculates the winner, and
rejects the final DFlash2 arm unless it uses that width:

The configured width is the full proposal width, not an assertion that every
target verification remains `M=K+1`. The output-budget tail is deliberately
truncated and may measure any `M` from 1 through `K+1`. Qualification requires
the full `K+1` case to occur, rejects any larger row count, and retains every
smaller tail measurement in the published physical-M curve.

Start a fresh service process for every width trial, then start it again at the
selected width before collecting the complete final DFlash2 arm. Likewise,
start a fresh process before the native-MTP final arm. Do not collect final
evidence from the process used by a preliminary trial: DFlash2 maintains an
aligned target/draft radix cache while native MTP deliberately recomputes the
prompt, so process reuse can make otherwise matched prompt timing incomparable.
Copy each deployment record only after that arm's fresh process has completed
startup. Deployment evidence binds the launcher's nanosecond start identity;
qualification rejects a reused launch identity anywhere among the seven width
trials, the selected-width final arm, or the native-MTP final arm.

```bash
width_trial_args=()
for width in 1 2 3 4 5 6 7; do
  width_trial_args+=(
    --dflash2-width-trial "$width"
    "$qualification_root/dflash2-width-$width-deployment.json"
    "$qualification_root/dflash2-width-$width-blended.jsonl"
    "$qualification_root/dflash2-width-$width-tool-eval.json"
  )
done
```

```bash
model=wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1
tokenizer=/path/to/the/staged/GLM-5.3-EXL3-K4-v1/tokenizer.json
nonce_seed=2026082901
needle_session=glm53-k4-needle-v1
prefill_run_id=glm53-k4-prefill-v1
frozen_corpus=/path/to/unchanged/prefill-corpus

# Set arm=mtp for the native-MTP deployment and arm=dflash2 for DFlash2.
cp .glmrt-wip/run/deployment.json \
  "$qualification_root/$arm-deployment.json"

python3 python/tools/bench_real_full_mtp_acceptance.py \
  --model "$model" --tokenizer "$tokenizer" \
  --suite weighted --repeats 5 --nonce-seed "$nonce_seed" \
  --output "$qualification_root/$arm-blended.jsonl"

python3 python/tools/bench_real_full_repeat_decode.py \
  --model "$model" --tokenizer "$tokenizer" \
  --word orchid --count 100 --max-tokens 1500 \
  --warmups 1 --repeats 5 --nonce-seed "$nonce_seed" \
  --output "$qualification_root/$arm-orchid.jsonl"

for concurrency in 1 2 4; do
  python3 python/tools/bench_real_full_concurrency.py code \
    --model "$model" --tokenizer "$tokenizer" \
    --concurrency "$concurrency" --warmups 2 --repeats 5 \
    --cache-state token-zero-nonce --nonce-seed "$nonce_seed" \
    --output "$qualification_root/$arm-concurrency-c$concurrency.jsonl"
done

python3 python/tools/bench_release_prefill_matrix.py \
  --model "$model" --tokenizer "$tokenizer" \
  --corpus-root "$frozen_corpus" --profile balanced \
  --run-id "$prefill_run_id" --repeats 2 \
  --output "$qualification_root/$arm-prefill.jsonl"

python3 python/tools/bench_real_full_needle.py \
  --model "$model" --tokenizer "$tokenizer" \
  --session-id "$needle_session" \
  --max-context-tokens 400000 --maximum-request-seconds 600 \
  --output "$qualification_root/$arm-needle.jsonl"

tool-eval-bench \
  --base-url http://127.0.0.1:8000/v1/ --parallel 1 \
  --temperature 0 \
  --model "$model" --json-file "$qualification_root/$arm-tool-eval.json"

# Preserve the exact Spark residency used by this speculation arm. Native MTP
# includes the retained layer 78, converted once from source block-FP8 into its
# startup W4A16 TP4 slab; DFlash2 must omit that otherwise-unused layer.
expert_runtime_fingerprint="$(
  jq -r .fingerprints.expert_runtime \
    "$qualification_root/$arm-deployment.json"
)"
for host in ostrich dodo emu kiwi; do
  ssh "$host" docker exec glrmt-spark-expert-wip \
    cat /wip/run/expert-9100.log \
    > "$qualification_root/$arm-$host-expert.log"
done
startup_mtp_arg=()
if [[ "$arm" == mtp ]]; then
  startup_mtp_arg+=(--include-mtp)
fi
python3 python/tools/analyze_glm52_expert_startup.py \
  --model "$model" --weight-format exl3 --cache-state warm \
  --expert-runtime-fingerprint "$expert_runtime_fingerprint" \
  "${startup_mtp_arg[@]}" \
  --log ostrich="$qualification_root/$arm-ostrich-expert.log" \
  --log dodo="$qualification_root/$arm-dodo-expert.log" \
  --log emu="$qualification_root/$arm-emu-expert.log" \
  --log kiwi="$qualification_root/$arm-kiwi-expert.log" \
  --output "$qualification_root/$arm-startup.json"
```

The selected-default commands in this subsection are intentionally adjacent to
the per-arm startup collection, but they run only **after** the signed serving
qualifier below has selected `selected_default`; this subsection is not an
instruction to guess the winner early. For the final selected-default startup
graphic, keep “cold” and “warm” launch
state separate from the compiled-kernel cache label above. A cold launch starts
and reloads all four Spark expert slabs. A warm launch uses `--restart` with
the identical slot/configuration and must retain all four fingerprint-matched
resident experts. Capture the complete launcher stream with `set -o pipefail`
and snapshot `/wip/run/coordinator-${ADDR##*:}.log` plus `deployment.json`
immediately after each launch; the persistent process log is append-only and
must not be referenced in place after the next restart.

Analyze each immutable snapshot against the same accepted Spark startup report:

```bash
python3 python/tools/analyze_glm53_full_startup.py \
  --cache-state cold --speculation "$selected_default" \
  --deployment "$qualification_root/default-cold-deployment.json" \
  --launcher-log "$qualification_root/default-cold-launcher.log" \
  --coordinator-log "$qualification_root/default-cold-coordinator.log" \
  --expert-startup "$qualification_root/$selected_default-startup.json" \
  --output "$qualification_root/default-cold-full-startup.json"

python3 python/tools/analyze_glm53_full_startup.py \
  --cache-state warm --speculation "$selected_default" \
  --deployment "$qualification_root/default-warm-deployment.json" \
  --launcher-log "$qualification_root/default-warm-launcher.log" \
  --coordinator-log "$qualification_root/default-warm-coordinator.log" \
  --expert-startup "$qualification_root/$selected_default-startup.json" \
  --output "$qualification_root/default-warm-full-startup.json"
```

The analyzer reconciles every launcher/shell/daemon phase clock, aligns the
parallel Spark readiness and coordinator critical paths, verifies reload versus
retention from the launcher lifecycle, and binds the exact runtime fingerprint.
The renderer reopens and rehashes all underlying logs/reports before producing
the new GLM-5.3-only graphic and its signed provenance report:

```bash
python3 python/tools/render_glm53_startup_timeline.py \
  --serving "$qualification_root/glm53-exl3-k4-serving.json" \
  --cold "$qualification_root/default-cold-full-startup.json" \
  --warm "$qualification_root/default-warm-full-startup.json" \
  --output "$qualification_root/glm53-startup-timeline.svg" \
  --report "$qualification_root/glm53-startup-timeline.json"
```

Render the selected-default production micro-timeline directly from its
accepted balanced seven-type run. The first code replay supplies the complete
chronological post-TTFT sequence; the renderer reconciles physical M,
acceptance, committed tokens, cycle time, and `decode_ms`, then shows the full
response ribbon, a time-scaled first-cycle zoom, and the independently
recomputed physical-M curve. This avoids the synchronization overhead and
separate diagnostic deployment used by prior development graphics:

```bash
python3 python/tools/render_glm53_balanced_micro_timeline.py \
  --serving "$qualification_root/glm53-exl3-k4-serving.json" \
  --deployment "$qualification_root/default-deployment.json" \
  --blended "$qualification_root/balanced-$selected_default-blended.jsonl" \
  --output "$qualification_root/glm53-balanced-micro-timeline.svg" \
  --report "$qualification_root/glm53-balanced-micro-timeline.json"
```

Use the same explicit nonce seed and needle session for both modes. The
qualifier recomputes every concurrency aggregate from its lanes and requires
identical request digests across modes. The needle gate covers 8K, 32K, 128K,
256K, and 384K at 10%, 50%, and 90% depth; all 15 requests per mode must recall
the exact key, complete within ten minutes, execute complete attention and
numeric progression, and report zero runtime graph captures.

The final OSS benchmark refresh is a superset of the serving gate and is
collected from this same accepted binary/artifact rather than rerun in the
release checkout. Publish GLM-5.3 EXL3 K4 only. Preserve enough signed evidence
to replace every measured section of the current OSS README:

- balanced native-MTP and selected-width DFlash2: seven-type decode, Orchid,
  C1/C2/C4 code concurrency, the complete cache-aware prefill matrix, startup,
  physical-M verify-cycle timings, and the long-context needle grid;
- the selected agentic default: the 0/32K/64K/128K/256K context decode matrix
  for code, creative writing, and math, three serial 69-scenario tool-eval runs
  with distinct recorded seeds, and isolated Pi coding-agent runs with
  reasoning off and high;
- native-MTP and DFlash2 under balanced, long, and accuracy profiles: the same
  five-replay seven-type decode workload plus an exact cached-2K/+8K prefill
  cell, retaining verify throughput and draft acceptance; and
- a new selected-default balanced-path micro-timeline and cold/warm startup
  timelines. Do not reuse diagrams or startup numbers from another model.

The recipes below define those release arms but are executed only after the
serving qualifier later in this section has accepted both balanced modes and
selected the default. Their `--serving` input must be that finished report;
they cannot be run top-to-bottom before it exists.

For the six profile/mode arms, use one fixed nonce seed and run ID. Record the
fresh deployment report for every arm; the balanced deployment and blended
files are the already-accepted serving arms, while their 2K/+8K prefill cell is
an additional release measurement. For each deployment run:

```bash
python3 python/tools/bench_real_full_mtp_acceptance.py \
  --model "$model" --profile "$profile" --suite weighted --repeats 5 \
  --nonce-seed 53 --tokenizer "$tokenizer" --run-id glm53-k4-profile-v1 \
  --output "$qualification_root/$profile-$arm-blended.jsonl"

python3 python/tools/bench_release_prefill_matrix.py \
  --model "$model" --profile "$profile" --base 2048 --suffix 8192 \
  --repeats 2 --tokenizer "$tokenizer" \
  --corpus-root /path/to/frozen/release-corpus \
  --run-id glm53-k4-profile-prefill-v1 \
  --output "$qualification_root/$profile-$arm-prefill.jsonl"
```

After all six arms, bind them to the accepted serving runtime and recompute the
decode, acceptance, verifier-throughput, prefill, mode-ratio, and
profile-retention tables. Repeat `--arm` once for every balanced/long/accuracy
and `mtp`/`dflash2` combination:

```bash
python3 python/tools/validate_glm53_profile_release_evidence.py \
  --serving "$qualification_root/glm53-exl3-k4-serving.json" \
  --arm balanced mtp \
    "$qualification_root/balanced-mtp-deployment.json" \
    "$qualification_root/balanced-mtp-blended.jsonl" \
    "$qualification_root/balanced-mtp-prefill.jsonl" \
  --arm balanced dflash2 \
    "$qualification_root/balanced-dflash2-deployment.json" \
    "$qualification_root/balanced-dflash2-blended.jsonl" \
    "$qualification_root/balanced-dflash2-prefill.jsonl" \
  --arm long mtp \
    "$qualification_root/long-mtp-deployment.json" \
    "$qualification_root/long-mtp-blended.jsonl" \
    "$qualification_root/long-mtp-prefill.jsonl" \
  --arm long dflash2 \
    "$qualification_root/long-dflash2-deployment.json" \
    "$qualification_root/long-dflash2-blended.jsonl" \
    "$qualification_root/long-dflash2-prefill.jsonl" \
  --arm accuracy mtp \
    "$qualification_root/accuracy-mtp-deployment.json" \
    "$qualification_root/accuracy-mtp-blended.jsonl" \
    "$qualification_root/accuracy-mtp-prefill.jsonl" \
  --arm accuracy dflash2 \
    "$qualification_root/accuracy-dflash2-deployment.json" \
    "$qualification_root/accuracy-dflash2-blended.jsonl" \
    "$qualification_root/accuracy-dflash2-prefill.jsonl" \
  --output "$qualification_root/glm53-exl3-k4-profile-release.json"
```

After the agentic default is selected, collect its three publication tool
runs with explicit, distinct seeds rather than relying on the tool's implicit
default. Keep the service process and every other sampling/control field fixed:

```bash
for seed in 2026082901 2026082902 2026082903; do
  tool-eval-bench \
    --base-url http://127.0.0.1:8000/v1/ --parallel 1 \
    --temperature 0 --seed "$seed" --model "$model" \
    --json-file "$qualification_root/default-tool-eval-seed-$seed.json"
done
```

Collect the two isolated Pi coding-agent arms into new, nonexisting evidence
directories. The wrapper pins the exact prompt, local `glmrt` provider,
temperature zero, built-in tool surface, and Pi event mode; it then validates
the complete transcript, token accounting, wall time, single HTML artifact,
and every module script with Node. These reports replace manually counted Pi
rows:

```bash
for thinking in off high; do
  scripts/bench-pi-coding-agent.sh \
    --root "$qualification_root/default-pi-$thinking" \
    --model wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 \
    --thinking "$thinking"
done
```

Bind those five publication runs to the balanced deployment of the measured
default and the accepted serving qualification. Pass the tool files in the
same seed order shown above. This gate rejects a different model revision,
slot/runtime fingerprint, speculation width/backend, pre-deployment run,
nonserial tool run, changed 69-scenario fixture, or reused Pi session:

```bash
python3 python/tools/validate_glm53_agentic_release_evidence.py \
  --serving "$qualification_root/glm53-exl3-k4-serving.json" \
  --deployment "$qualification_root/default-deployment.json" \
  --tool-eval "$qualification_root/default-tool-eval-seed-2026082901.json" \
  --tool-eval "$qualification_root/default-tool-eval-seed-2026082902.json" \
  --tool-eval "$qualification_root/default-tool-eval-seed-2026082903.json" \
  --pi-off "$qualification_root/default-pi-off/evidence.json" \
  --pi-high "$qualification_root/default-pi-high/evidence.json" \
  --output "$qualification_root/glm53-exl3-k4-agentic-release.json"
```

Collect the selected-default context decode grid from that same balanced
deployment. The harness constructs one exact retained prefix per nonzero
context and branches all timed workloads from it, rather than rebuilding the
long prefix for every sample. It refuses to overwrite evidence and rejects an
incomplete Cartesian product, insufficient prefix reuse, inconsistent
cache/prefill row accounting, incomplete attention, failed numeric progression,
or runtime graph capture:

```bash
python3 python/tools/bench_release_decode_matrix.py \
  --model wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 \
  --tokenizer "$tokenizer" \
  --corpus-root /path/to/frozen/release-corpus \
  --profile balanced \
  --run-id 20260830T000000Z \
  --output "$qualification_root/default-context-decode.jsonl"
```

For each nonzero cached-context prefill sample, `cached_prompt_tokens` must be
at least the requested base and
`prompt_tokens - cached_prompt_tokens - 1 == prefill_rows == suffix_tokens`.
The benchmark primes each base once, then branches every timed suffix from the
retained radix-cache prefix; rebuilding the long prefix for every cell is not
valid release evidence.

Generate the GPU DFlash2 preflight from the same final WIP binary and staged
K4 target. The draft cache is sliding-window bounded; 2,176 tokens is the
runtime's page-rounded envelope for the 2,048-token window, one 64-token page,
and every selected K1–K7 query block. Keep the checkpoint revision literal so a moving
Hub ref cannot enter publication evidence:

```bash
glmrt_bin=/path/to/the/final/wip/glmrt
dflash_snapshot=/path/to/models--incoai--GLM-5.3-DFlash2/snapshots/425aa615ce320caac34400208b30808c8f14f76c
target_catalog="$qualification_root/glm53-k4-target-catalog.json"
dflash2_width=SELECTED_WIDTH_FROM_K1_THROUGH_K7_SWEEP
dflash2_topk_backend=SELECTED_BACKEND_FROM_DFLASH2_TOPK_TUNING

"$glmrt_bin" inspect-model \
  --model-id wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 \
  --out "$target_catalog" \
  --summary "$qualification_root/glm53-k4-target-catalog.md"

GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND="$dflash2_topk_backend" \
"$glmrt_bin" dflash-preflight \
  --snapshot "$dflash_snapshot" \
  --target-catalog "$target_catalog" \
  --kv-capacity-tokens 2176 \
  --max-concurrency 4 \
  --kv-storage bf16 \
  --kv-page-size 64 \
  --context-tokens 1024 \
  --accepted-rows-per-request 4 \
  --proposal-tokens-per-request "$dflash2_width" \
  --preload --capture-static \
  --static-warmup 2 --static-iterations 10 --static-repeats 3 \
  >"$qualification_root/dflash2-preflight.json"
```

This preflight must be generated with the measured winning width. That width
changes the actual DFlash2 body/head graph geometry to `K+1` query rows and K
candidate rows; it is not a post-hoc truncation of a K7 graph. The accepted C1
graph registry must contain exact update rows 2 through 8,
followed by the larger power-of-two prefill buckets through 1,024. This keeps
every possible one-request DFlash2 decode commit (1 through 8 target rows) to
one update-graph replay; C2 and C4 continue to use safely padded power-of-two
registries. The `--accepted-rows-per-request 4` value above exercises the
independent memory/capacity plans. Static GPU capture always mirrors serving:
one-row base executors plus the complete packed registry. Every C1/C2/C4 graph
is captured against the production four-slot shared KV allocation: 34 pages
per request and 136 physical 64-token pages in total. A smaller
per-concurrency cache changes layer strides and is not accepted as
production-equivalent evidence. The preflight also
numerically checks the base update and every advertised packed bucket against
its reference, checks eager/replay identity, proves that runtime position
updates affect the keys, and proves exact output after restoring the positions.
The head proof also binds deterministic sorted top-k and the upstream-matched
BF16-edge/FP32-final candidate-score accumulation contract.
The qualifier rejects C1 reports that omit rows 3, 5, 6, or 7 or do not carry
that per-bucket proof.

After generating the four native TP-rank reports and the GPU DFlash2 preflight,
produce the signed serving report with the complete GLM-5.3 invocation:

```bash
serving_qualification_report="$state_root/glm53-exl3-k4-serving-qualification.json"

python3 python/tools/validate_glm53_exl3_serving_qualification.py \
  --artifact "$output_root" \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --native-blended "$qualification_root/mtp-blended.jsonl" \
  --dflash2-blended "$qualification_root/dflash2-blended.jsonl" \
  --native-repeat "$qualification_root/mtp-orchid.jsonl" \
  --dflash2-repeat "$qualification_root/dflash2-orchid.jsonl" \
  --native-prefill "$qualification_root/mtp-prefill.jsonl" \
  --dflash2-prefill "$qualification_root/dflash2-prefill.jsonl" \
  --native-tool-eval "$qualification_root/mtp-tool-eval.json" \
  --dflash2-tool-eval "$qualification_root/dflash2-tool-eval.json" \
  --native-startup "$qualification_root/mtp-startup.json" \
  --dflash2-startup "$qualification_root/dflash2-startup.json" \
  --native-deployment "$qualification_root/mtp-deployment.json" \
  --dflash2-deployment "$qualification_root/dflash2-deployment.json" \
  --native-concurrency "$qualification_root/mtp-concurrency-c1.jsonl" \
  --native-concurrency "$qualification_root/mtp-concurrency-c2.jsonl" \
  --native-concurrency "$qualification_root/mtp-concurrency-c4.jsonl" \
  --dflash2-concurrency "$qualification_root/dflash2-concurrency-c1.jsonl" \
  --dflash2-concurrency "$qualification_root/dflash2-concurrency-c2.jsonl" \
  --dflash2-concurrency "$qualification_root/dflash2-concurrency-c4.jsonl" \
  --native-needle "$qualification_root/mtp-needle.jsonl" \
  --dflash2-needle "$qualification_root/dflash2-needle.jsonl" \
  --native-validation "$native_evidence_root/native-tp0.json" \
  --native-validation "$native_evidence_root/native-tp1.json" \
  --native-validation "$native_evidence_root/native-tp2.json" \
  --native-validation "$native_evidence_root/native-tp3.json" \
  --dflash2-preflight "$qualification_root/dflash2-preflight.json" \
  --dflash2-topk-tuning "$qualification_root/dflash2-topk-tuning.json" \
  --dflash2-selector-tuning "$qualification_root/dflash2-selector-tuning.json" \
  --dflash2-body-fusion-tuning "$qualification_root/dflash2-body-fusion-tuning.json" \
  "${width_trial_args[@]}" \
  --expected-default auto \
  --output "$serving_qualification_report"
```

Once the profile, agentic, retained-context, startup-timeline, and production
micro-timeline reports are complete, bind the entire OSS benchmark refresh to
one selected production runtime. This final gate recursively reopens and
rehashes every referenced input and rendered SVG; copied summary numbers or a
report from a different binary, model revision, deployment, speculation
setting, power limit, or pre-deployment run are rejected:

```bash
python3 python/tools/validate_glm53_oss_release_evidence.py \
  --serving "$serving_qualification_report" \
  --agentic "$qualification_root/glm53-exl3-k4-agentic-release.json" \
  --profiles "$qualification_root/glm53-exl3-k4-profile-release.json" \
  --context-decode "$qualification_root/default-context-decode.jsonl" \
  --startup-timeline "$qualification_root/glm53-startup-timeline.json" \
  --micro-timeline "$qualification_root/glm53-balanced-micro-timeline.json" \
  --output "$qualification_root/glm53-exl3-k4-oss-release.json"

python3 python/tools/render_glm53_oss_benchmark_charts.py \
  --evidence "$qualification_root/glm53-exl3-k4-oss-release.json" \
  --prefill-output benchmarks/glm53-exl3-k4-prefill.svg \
  --decode-output benchmarks/glm53-exl3-k4-decode.svg \
  --report "$qualification_root/glm53-exl3-k4-benchmark-charts.json"

python3 python/tools/render_glm53_oss_benchmark_markdown.py \
  --evidence "$qualification_root/glm53-exl3-k4-oss-release.json" \
  --charts "$qualification_root/glm53-exl3-k4-benchmark-charts.json" \
  --output benchmarks/glm53-exl3-k4-release.md
```

The renderer emits the high-level mode comparison, all seven decode types and
acceptance rates, both 5x6 prefill matrices, selected-default 5x3 retained-
context decode matrix, C1/C2/C4 scaling, three seeded tool runs, both Pi arms,
both long-context needle grids, all six profile/mode arms, and the cold/warm
and production-timeline links. The selected-default prefill and retained-
context decode SVGs are generated from those same cells, and their signed
chart report is rebound and rehashed before the Markdown links them. Neither
renderer carries forward benchmark rows from another target model. The
resulting repository-local Markdown is the numeric source for the OSS README
refresh; do not transcribe tables from console output.

Render `quantization/GLM53_EXL3_MODEL_CARD.md` only from that accepted report,
then build the standard-only publication tree. Do not point the upload command
at the quantizer output directory:

```bash
public_root="$output_root-publication"
publication_report="$state_root/glm53-exl3-k4-publication.json"
rendered_model_card="$state_root/GLM53-EXL3-K4-README.md"

python3 python/tools/render_glm53_exl3_model_card.py \
  --template quantization/GLM53_EXL3_MODEL_CARD.md \
  --artifact-validation "$artifact_report" \
  --quant-evidence "$quant_evidence_report" \
  --serving-qualification "$serving_qualification_report" \
  --output "$rendered_model_card"

python3 python/tools/prepare_glm52_exl3_hf_publication.py \
  --artifact "$output_root" \
  --source-snapshot "$source_snapshot" \
  --validation-report "$artifact_report" \
  --quant-evidence-report "$quant_evidence_report" \
  --serving-qualification-report "$serving_qualification_report" \
  --readme "$rendered_model_card" \
  --output "$public_root" \
  --report "$publication_report"
```

The generated section includes both speculation modes, the exact seven-type
table, C1/C2/C4 decode, the independently recomputed post-TTFT C1 target-cycle
curve by physical M, including directly measured scalar M1 cycles and every
DFlash2 K1-K7 trial. Each request's measured cycle sum must equal the same
`decode_ms` used for TPS. The section also includes prefill, the long-context
needle grid, structural hashes, and the measured agentic default.

Re-stage the standard-only publication tree and synchronize it once more for
the final generation smoke test:

```bash
: "${model_id:?set model_id to the exact accepted artifact repository ID}"
python3 python/tools/stage_glm52_exl3_hf_snapshot.py \
  --artifact "$public_root" \
  --publication-report "$publication_report" \
  --model-id "$model_id" \
  --update-ref

python3 python/tools/sync_glm52_exl3_hf_snapshot.py \
  --model-id "$model_id" \
  --hosts ostrich,dodo,emu,kiwi
```

Upload the accepted standard-only tree to exactly this repository, then run
the Hub verifier against the resolved public revision:

```bash
hf upload wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 \
  "$public_root" . --no-private \
  --commit-message "Publish calibrated GLM-5.3 EXL3 K4 v1"

python3 python/tools/verify_glm52_exl3_hub_publication.py \
  --publication-report "$publication_report" \
  --revision main \
  --hf-home "${HF_HOME:-$HOME/.cache/huggingface}" \
  --output "$state_root/glm53-exl3-k4-hub-verification.json"
```

Do not upload the rolling quantizer output directly. Only the publication tree
created after the full artifact, serving, compact-config, and standalone
`tensor_storage` gates is eligible for this command.
The verifier does not clean-download the model. It checks the actual rendered
Hub model, revision-tree, and `config.json` pages, requires the Hub model API to
expose the compact EXL3 declaration, verifies every remote Git blob or LFS OID,
and hardlinks the accepted publication bytes into
`snapshots/<resolved-Hub-commit>` with standard remote blob names and
`refs/main`. As a final gate, it resolves that exact commit through
`huggingface_hub.snapshot_download(..., local_files_only=True)`; no model data
is downloaded. The final load/smoke test must use that exact materialized path.
