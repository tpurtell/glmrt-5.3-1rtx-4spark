#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/scripts/release-common.sh"
launcher_started_ns="$(date +%s%N)"
launcher_phase_started_ns="$launcher_started_ns"

report_wip_startup_phase() {
  local stage="$1" now_ns elapsed_ms total_ms
  now_ns="$(date +%s%N)"
  elapsed_ms=$(((now_ns - launcher_phase_started_ns) / 1000000))
  total_ms=$(((now_ns - launcher_started_ns) / 1000000))
  echo "wip_launcher_startup_phase stage=$stage elapsed_ms=$elapsed_ms total_ms=$total_ms" >&2
  launcher_phase_started_ns="$now_ns"
}

usage() {
  cat <<'EOF'
Usage: ./run.sh --wip [--wip-slot NAME] [--profile FILE] [--restart] [--dry-run]

Runs a named slot inside the five persistent WIP development containers.
Without --profile, the configuration frozen into the coordinator slot is used.
No source synchronization, compilation, image creation, or container
recreation is performed here; use wip.sh for those operations.
With --restart, exact fingerprint-matched resident Spark experts are retained;
otherwise all affected processes are restarted.
EOF
}

config="$repo_root/glmrt.config"
config_explicit=0
slot=current
restart=0
dry_run=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --wip)
      shift
      ;;
    --wip-slot)
      slot="${2:?--wip-slot requires a name}"
      shift 2
      ;;
    --profile|--config)
      config="${2:?$1 requires a configuration file}"
      config_explicit=1
      shift 2
      ;;
    --restart)
      restart=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      release_die "unknown WIP run argument: $1"
      ;;
  esac
done

[[ "$slot" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] ||
  release_die "invalid WIP slot name: $slot"
((restart == 0 || dry_run == 0)) ||
  release_die "--restart and --dry-run are mutually exclusive"

release_need docker
release_need ssh
release_need jq
release_need curl
release_need ss
release_need nvidia-smi
release_need sha256sum
release_need python3
docker info >/dev/null 2>&1 || release_die "local Docker daemon is unavailable"

coordinator_container=glrmt-coordinator-wip
spark_container=glrmt-spark-expert-wip
docker container inspect "$coordinator_container" >/dev/null 2>&1 ||
  release_die "persistent coordinator WIP container is missing; run ./wip.sh --slot '$slot'"
[[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]] ||
  release_die "persistent coordinator WIP container is stopped; run ./wip.sh --slot '$slot'"

# Bootstrap topology from the operator configuration, then prefer the exact
# configuration frozen with the slot unless the caller selected one explicitly.
release_load_config "$config"
state_dir="$repo_root/.glmrt-wip/run"
mkdir -p "$state_dir"
if ((config_explicit == 0)); then
  slot_config="$state_dir/${slot}.config"
  docker exec "$coordinator_container" \
    cat "/wip/slots/$slot/coordinator/workspace/glmrt.config" >"$slot_config" ||
    release_die "coordinator WIP slot is missing its frozen glmrt.config: $slot"
  config="$slot_config"
  release_load_config "$config"
fi
report_wip_startup_phase bootstrap

hosts_csv="$(release_hosts_csv)"
lane_a_csv="$(release_lane_a_csv)"
lane_b_csv="$(release_lane_b_csv)"
expert_hosts_csv="$(release_expert_hosts_csv)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$hf_home"
coordinator_workspace="/wip/slots/$slot/coordinator/workspace"
expert_workspace="/wip/slots/$slot/spark-expert/workspace"
coordinator_process="coordinator-${ADDR##*:}"
expert_process="expert-$EXPERT_PORT"

validate_local_slot() {
  local image_id
  image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV" 2>/dev/null || true)"
  [[ -n "$image_id" ]] || release_die "missing development image $COORDINATOR_DOCKER_DEV; run ./build.sh"
  [[ "$(docker inspect -f '{{.Image}}' "$coordinator_container")" == "$image_id" ]] ||
    release_die "$coordinator_container uses an old development image; run ./wip.sh --recreate"
  docker exec -i "$coordinator_container" bash -s -- \
    "$slot" coordinator "$image_id" <<'CONTAINER'
set -euo pipefail
slot="$1"
role="$2"
image_id="$3"
root="/wip/slots/$slot/$role"
test -s "$root/META.json"
test -s "$root/FINGERPRINT"
test -x "$root/workspace/.glmrt-wip/glmrt"
test -s "$root/workspace/.glmrt-wip/libglmrt_native.so"
actual_fingerprint="$(sha256sum "$root/META.json" | awk '{print $1}')"
test "$actual_fingerprint" = "$(<"$root/FINGERPRINT")"
python3 "$root/workspace/scripts/verify-release-source-manifest.py" \
  --source "$root/workspace" --manifest "$root/SOURCE_SHA256SUMS" >&2
(
  cd "$root/workspace/.glmrt-wip"
  sha256sum -c ARTIFACT_SHA256SUMS >&2
)
python3 - "$root/META.json" "$image_id" "$root" <<'PY'
import hashlib
import json
import pathlib
import sys

meta_path, image_id, root = sys.argv[1:]
meta = json.loads(pathlib.Path(meta_path).read_text())
assert meta["schema"] == 1
assert meta["base_image_id"] == image_id, (meta["base_image_id"], image_id)
source_sum = hashlib.sha256(pathlib.Path(root, "SOURCE_SHA256SUMS").read_bytes()).hexdigest()
artifact_sum = hashlib.sha256(pathlib.Path(root, "workspace/.glmrt-wip/ARTIFACT_SHA256SUMS").read_bytes()).hexdigest()
assert meta["source_manifest_sha256"] == source_sum
assert meta["artifact_manifest_sha256"] == artifact_sum
PY
cat "$root/FINGERPRINT"
CONTAINER
}

validate_remote_slot() {
  local host="$1"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$spark_container" "$SPARK_EXPERT_DOCKER_DEV" "$slot" "$RELEASE_MODEL_ID" <<'REMOTE'
set -euo pipefail
container="$1"
image="$2"
slot="$3"
model_id="$4"
test "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" = true
image_id="$(docker image inspect -f '{{.Id}}' "$image")"
container_image_id="$(docker inspect -f '{{.Image}}' "$container")"
test "$container_image_id" = "$image_id"
slot_fingerprint="$(docker exec -i "$container" bash -s -- "$slot" "$image_id" <<'CONTAINER'
set -euo pipefail
slot="$1"
image_id="$2"
root="/wip/slots/$slot/spark-expert"
test -s "$root/META.json"
test -s "$root/FINGERPRINT"
test -x "$root/workspace/.glmrt-wip/glmrt"
test -s "$root/workspace/.glmrt-wip/libglmrt_native.so"
actual_fingerprint="$(sha256sum "$root/META.json" | awk '{print $1}')"
test "$actual_fingerprint" = "$(<"$root/FINGERPRINT")"
python3 "$root/workspace/scripts/verify-release-source-manifest.py" \
  --source "$root/workspace" --manifest "$root/SOURCE_SHA256SUMS" >&2
(
  cd "$root/workspace/.glmrt-wip"
  sha256sum -c ARTIFACT_SHA256SUMS >&2
)
python3 - "$root/META.json" "$image_id" "$root" <<'PY'
import hashlib
import json
import pathlib
import sys

meta_path, image_id, root = sys.argv[1:]
meta = json.loads(pathlib.Path(meta_path).read_text())
assert meta["schema"] == 1
assert meta["base_image_id"] == image_id, (meta["base_image_id"], image_id)
source_sum = hashlib.sha256(pathlib.Path(root, "SOURCE_SHA256SUMS").read_bytes()).hexdigest()
artifact_sum = hashlib.sha256(pathlib.Path(root, "workspace/.glmrt-wip/ARTIFACT_SHA256SUMS").read_bytes()).hexdigest()
assert meta["source_manifest_sha256"] == source_sum
assert meta["artifact_manifest_sha256"] == artifact_sum
PY
cat "$root/FINGERPRINT"
CONTAINER
)"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
model_root="$hf_home/hub/models--${model_id//\//--}"
test -s "$model_root/refs/main"
model_revision="$(<"$model_root/refs/main")"
[[ "$model_revision" =~ ^[0-9a-f]{40,64}$ ]]
model_snapshot="$model_root/snapshots/$model_revision"
test -d "$model_snapshot"
! find "$model_snapshot" -xtype l -print -quit | grep -q .
printf '%s %s\n' "$slot_fingerprint" "$model_revision"
REMOTE
}

echo "== validating persistent WIP containers and slot $slot =="
validation_dir="$(mktemp -d "$state_dir/slot-validation.XXXXXX")"
validation_labels=(coordinator)
validation_pids=()
validate_local_slot >"$validation_dir/0.out" 2>"$validation_dir/0.err" &
validation_pids+=("$!")
validation_index=0
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  validation_index=$((validation_index + 1))
  validation_labels+=("$host")
  validate_remote_slot "$host" \
    >"$validation_dir/$validation_index.out" \
    2>"$validation_dir/$validation_index.err" &
  validation_pids+=("$!")
done
validation_failed=0
for validation_index in "${!validation_pids[@]}"; do
  if ! wait "${validation_pids[$validation_index]}"; then
    echo "${validation_labels[$validation_index]} WIP slot validation failed: $slot" >&2
    validation_failed=1
  fi
  cat "$validation_dir/$validation_index.err" >&2
done
if ((validation_failed)); then
  rm -rf "$validation_dir"
  release_die "one or more WIP slot validations failed: $slot"
fi
coordinator_slot_fingerprint="$(<"$validation_dir/0.out")"
coordinator_sparkinfer_commit="$(
  docker exec "$coordinator_container" python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["sparkinfer_revision"])' \
    "/wip/slots/$slot/coordinator/META.json"
)"
expert_slot_fingerprint=
expert_model_revision=
validation_index=0
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  validation_index=$((validation_index + 1))
  read -r remote_fingerprint remote_model_revision extra \
    <"$validation_dir/$validation_index.out" ||
    release_die "$host returned no expert slot identity"
  [[ -z "${extra:-}" && "$remote_fingerprint" =~ ^[0-9a-f]{64}$ &&
    "$remote_model_revision" =~ ^[0-9a-f]{40,64}$ ]] ||
    release_die "$host returned an invalid expert slot identity"
  if [[ -z "$expert_slot_fingerprint" ]]; then
    expert_slot_fingerprint="$remote_fingerprint"
    expert_model_revision="$remote_model_revision"
  else
    [[ "$remote_fingerprint" == "$expert_slot_fingerprint" ]] ||
      release_die "$host has a different Spark WIP slot fingerprint"
    [[ "$remote_model_revision" == "$expert_model_revision" ]] ||
      release_die "$host has a different cached text-model revision"
  fi
  echo "  $host: persistent container and expert slot ready"
done
rm -rf "$validation_dir"
coordinator_engine_commit="wip-${slot}-${coordinator_slot_fingerprint:0:12}-${expert_slot_fingerprint:0:12}"
coordinator_image_id="$(docker image inspect -f '{{.Id}}' "$COORDINATOR_DOCKER_DEV")"
report_wip_startup_phase slot-validation

profile_args=(
  --repo-root "$coordinator_workspace"
  --profile "$PROFILE"
  --model "$MODEL"
  --speculation "$SPECULATION"
  --vision "$RELEASE_VISION"
  --mtp-bf16-experts "$MTP_BF16_EXPERTS"
  --headroom-gib "$COORDINATOR_GPU_HEADROOM_GIB"
  --concurrency "$CONCURRENCY"
  --dry-run
)
[[ -z "$KV_POOL_TOKENS" ]] || profile_args+=(--kv-pool-tokens "$KV_POOL_TOKENS")
[[ -z "$MAX_CONTEXT_TOKENS" ]] || profile_args+=(--max-context-tokens "$MAX_CONTEXT_TOKENS")
[[ -z "$MAX_OUTPUT_TOKENS" ]] || profile_args+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")
[[ -z "$DFLASH2_FIXED_DRAFTS" ]] || profile_args+=(--dflash2-fixed-drafts "$DFLASH2_FIXED_DRAFTS")
profile_args+=(--dflash2-topk-backend "$DFLASH2_TOPK_BACKEND")

resolved_json="$state_dir/resolved-profile.json"
docker exec \
  -e GLMRT_DSPARK_MODEL_ID="$RELEASE_DSPARK_MODEL_ID" \
  -e GLMRT_DSPARK_REVISION="$RELEASE_DSPARK_REVISION" \
  -e PYTHONPATH="$coordinator_workspace/third_party/sparkinfer:$coordinator_workspace/python/reference/glmrt_reference:$coordinator_workspace/python/reference:/opt/glmrt/third_party/sparkinfer" \
  -w "$coordinator_workspace" \
  "$coordinator_container" \
  python3 "$coordinator_workspace/python/tools/resolve_serve_profile.py" \
  "${profile_args[@]}" >"$resolved_json"
jq -e . "$resolved_json" >/dev/null || release_die "profile resolver returned invalid JSON"
blockers="$(jq -r '.blockers[]?' "$resolved_json")"
[[ -z "$blockers" ]] || release_die "profile blockers:\n$blockers"
resolved_dflash2_fixed_drafts="$(
  jq -r '.environment.GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS // empty' "$resolved_json"
)"

config_sha256="$(sha256sum "$config" | awk '{print $1}')"
expert_runtime_fingerprint="$(
  python3 "$repo_root/scripts/wip-expert-runtime-identity.py" \
    --resolved-profile "$resolved_json" \
    --expert-slot-fingerprint "$expert_slot_fingerprint" \
    --setting "model_id=$RELEASE_MODEL_ID" \
    --setting "model_revision=$expert_model_revision" \
    --setting "expert_port=$EXPERT_PORT" \
    --setting "expert_image=$SPARK_EXPERT_DOCKER_DEV" \
    --setting "runtime_cache=/wip/cache" \
    --setting "transport=verbs-host" \
    --setting "spark_0=$SPARK_0_HOST,$SPARK_0_LANE_A,$SPARK_0_LANE_B" \
    --setting "spark_1=$SPARK_1_HOST,$SPARK_1_LANE_A,$SPARK_1_LANE_B" \
    --setting "spark_2=$SPARK_2_HOST,$SPARK_2_LANE_A,$SPARK_2_LANE_B" \
    --setting "spark_3=$SPARK_3_HOST,$SPARK_3_LANE_A,$SPARK_3_LANE_B"
)"
deployment_fingerprint="$({
  printf 'glmrt-wip-deployment-v2\n'
  jq -S . "$resolved_json"
  printf '%s\n' \
    "$config_sha256" "$coordinator_slot_fingerprint" \
    "$expert_runtime_fingerprint" "$ADDR" "$coordinator_engine_commit"
} | sha256sum | awk '{print $1}')"
report_wip_startup_phase profile-resolution

process_status_local() {
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" status "$coordinator_process" 2>/dev/null || echo stopped
}

process_identity_local() {
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" identity "$coordinator_process" \
    2>/dev/null || true
}

inspect_remote_service_state() {
  local host="$1"
  local release_container="${RELEASE_SPARK_CONTAINER_PREFIX}-${host}-${EXPERT_PORT}"
  local legacy_container="glmrt-phase0-tcp-expertd-${host}-${EXPERT_PORT}"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$release_container" "$legacy_container" "$spark_container" \
    "$expert_workspace/scripts/wip-process.sh" "$expert_process" <<'REMOTE'
set -euo pipefail
release_container="$1"
legacy_container="$2"
wip_container="$3"
wip_process_script="$4"
expert_process="$5"
standard_active=0
for container in "$release_container" "$legacy_container"; do
  [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" != true ]] ||
    standard_active=1
done
wip_active=0
wip_identity=-
if [[ "$(docker inspect -f '{{.State.Running}}' "$wip_container" 2>/dev/null || true)" == true ]]; then
  wip_status="$(docker exec "$wip_container" "$wip_process_script" status "$expert_process")"
  case "$wip_status" in
    running\ *)
      wip_active=1
      wip_identity="$(docker exec "$wip_container" "$wip_process_script" identity "$expert_process" 2>/dev/null || true)"
      [[ -n "$wip_identity" ]] || wip_identity=-
      ;;
    stopped) ;;
    *) echo "invalid WIP process status for $expert_process: $wip_status" >&2; exit 2 ;;
  esac
fi
printf '%s %s %s\n' "$standard_active" "$wip_active" "$wip_identity"
REMOTE
}

standard_services=0
[[ "$(docker inspect -f '{{.State.Running}}' "$RELEASE_COORDINATOR_CONTAINER_NAME" 2>/dev/null || true)" != true ]] || standard_services=1
wip_coordinator_running=0
[[ "$(process_status_local)" == running\ * ]] && wip_coordinator_running=1
wip_experts_running=0
expert_fingerprints_match=1
service_state_dir="$(mktemp -d "$state_dir/service-state.XXXXXX")"
service_state_hosts=()
service_state_pids=()
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  inspect_remote_service_state "$host" \
    >"$service_state_dir/$host.out" 2>"$service_state_dir/$host.err" &
  service_state_hosts+=("$host")
  service_state_pids+=("$!")
done
service_state_failed=0
for service_state_index in "${!service_state_pids[@]}"; do
  host="${service_state_hosts[$service_state_index]}"
  if ! wait "${service_state_pids[$service_state_index]}"; then
    echo "$host service-state inspection failed" >&2
    service_state_failed=1
  elif ! read -r remote_standard_active remote_wip_active remote_wip_identity extra \
    <"$service_state_dir/$host.out"; then
    echo "$host returned no service-state data" >&2
    service_state_failed=1
  else
    if [[ -n "${extra:-}" || ! "$remote_standard_active" =~ ^[01]$ || ! "$remote_wip_active" =~ ^[01]$ ]] ||
      [[ "$remote_wip_identity" != - && ! "$remote_wip_identity" =~ ^[0-9a-f]{64}$ ]]; then
      echo "$host returned invalid service-state data" >&2
      service_state_failed=1
    else
      [[ "$remote_standard_active" == 0 ]] || standard_services=1
      [[ "$remote_wip_active" == 0 ]] || ((wip_experts_running += 1))
      [[ "$remote_wip_identity" == "$expert_runtime_fingerprint" ]] ||
        expert_fingerprints_match=0
    fi
  fi
  cat "$service_state_dir/$host.err" >&2
done
rm -rf "$service_state_dir"
((service_state_failed == 0)) || release_die "one or more Spark service-state inspections failed"
services_active=$((standard_services + wip_coordinator_running + wip_experts_running))
reuse_spark_experts=0

if ((services_active)); then
  if ((!restart)); then
    current_fingerprint="$(process_identity_local)"
    if ((standard_services == 0 && wip_coordinator_running == 1 && wip_experts_running == 4 && expert_fingerprints_match)) &&
      [[ "$current_fingerprint" == "$deployment_fingerprint" ]] &&
      curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >/dev/null; then
      echo "All five WIP services already match slot '$slot' and the selected configuration."
      exit 0
    fi
    release_die "partial, standard, or configuration-mismatched service state is active; use --restart"
  fi
  if ((standard_services == 0 && wip_experts_running == 4)); then
    reuse_spark_experts="$expert_fingerprints_match"
  fi
  if ((reuse_spark_experts)); then
    echo "== reusing four fingerprint-matched resident WIP Spark experts =="
    release_stop_wip_coordinator "$coordinator_process"
  else
    echo "== stopping existing WIP and release services =="
    release_stop_wip_services "$coordinator_process" "$expert_process"
    ((standard_services == 0)) ||
      release_stop_services "$RELEASE_COORDINATOR_CONTAINER_NAME" "$RELEASE_SPARK_CONTAINER_PREFIX"
  fi
fi
report_wip_startup_phase service-reconciliation

check_model_cache_local() {
  local model_id="$1" revision="${2:-}"
  local root="$hf_home/hub/models--${model_id//\//--}"
  [[ -n "$revision" || ! -f "$root/refs/main" ]] || revision="$(<"$root/refs/main")"
  [[ -n "$revision" && -d "$root/snapshots/$revision" ]] ||
    release_die "coordinator model snapshot is missing: $model_id${revision:+@$revision}"
  find "$root/snapshots/$revision" -xtype l -print -quit | grep -q . &&
    release_die "coordinator model snapshot has unresolved blobs: $model_id@$revision"
  return 0
}

echo "== checking model snapshots =="
check_model_cache_local "$RELEASE_MODEL_ID"
coordinator_model_root="$hf_home/hub/models--${RELEASE_MODEL_ID//\//--}"
coordinator_model_revision="$(<"$coordinator_model_root/refs/main")"
[[ "$coordinator_model_revision" == "$expert_model_revision" ]] ||
  release_die "coordinator selected $RELEASE_MODEL_ID@$coordinator_model_revision; Sparks selected @$expert_model_revision"
[[ "$SPECULATION" != dspark ]] ||
  check_model_cache_local "$RELEASE_DSPARK_MODEL_ID" "$RELEASE_DSPARK_REVISION"
echo "  selected text-model snapshot is identical on all five hosts: $coordinator_model_revision"
report_wip_startup_phase model-snapshots

echo "== checking launch headroom =="
available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
((available_kib >= 8 * 1024 * 1024)) || release_die "coordinator has less than 8 GiB available system memory"
total_mib=0
free_mib=0
for _ in $(seq 1 50); do
  # Query the coordinator GPU explicitly. With two visible GPUs, piping the
  # multi-line result through `head` intermittently SIGPIPEs nvidia-smi under
  # `set -o pipefail` and aborts a restart after the old service is stopped.
  gpu_line="$(nvidia-smi --id=0 --query-gpu=memory.total,memory.free --format=csv,noheader,nounits)"
  IFS=, read -r total_mib free_mib <<<"$gpu_line"
  total_mib="$(release_trim "$total_mib")"
  free_mib="$(release_trim "$free_mib")"
  ((free_mib >= 80 * 1024)) && break
  sleep 0.1
done
((free_mib >= 80 * 1024)) || release_die "coordinator GPU has only ${free_mib} MiB free; 80 GiB is required"
echo "  coordinator: RAM $((available_kib / 1024)) MiB available; GPU ${free_mib}/${total_mib} MiB free"

resource_check_dir="$(mktemp -d "$state_dir/resource-check.XXXXXX")"
resource_check_hosts=()
resource_check_pids=()
for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
  if ((reuse_spark_experts)); then
    echo "  $host: reusing resident fingerprint-matched expert; launch headroom check skipped"
    continue
  fi
  ssh -o BatchMode=yes "$host" bash -s -- "$spark_container" \
    >"$resource_check_dir/$host.out" 2>"$resource_check_dir/$host.err" <<'REMOTE' &
set -euo pipefail
container="$1"
min_kib=$((105 * 1024 * 1024))
# A stopped CUDA process can remain charged to unified memory briefly while
# the driver tears down its UVM mappings.  A configuration-changing --restart
# used to inspect that transient state once and falsely reject an otherwise
# healthy Spark.  Poll the same hard gate; do not weaken it.
available_kib=0
for _ in $(seq 1 100); do
  available_kib="$(awk '/MemAvailable:/{print $2}' /proc/meminfo)"
  ((available_kib >= min_kib)) && break
  sleep 0.1
done
((available_kib >= min_kib)) || { echo "only $((available_kib / 1024)) MiB unified memory is available" >&2; exit 2; }
gpu_line="$(docker exec "$container" nvidia-smi --id=0 --query-gpu=memory.total,memory.free --format=csv,noheader,nounits)"
echo "RAM $((available_kib / 1024)) MiB available; GPU $gpu_line"
REMOTE
  resource_check_hosts+=("$host")
  resource_check_pids+=("$!")
done
resource_check_failed=0
for resource_check_index in "${!resource_check_pids[@]}"; do
  host="${resource_check_hosts[$resource_check_index]}"
  if ! wait "${resource_check_pids[$resource_check_index]}"; then
    echo "$host failed the 105 GiB WIP launch headroom check" >&2
    resource_check_failed=1
  else
    echo "  $host: $(<"$resource_check_dir/$host.out")"
  fi
  cat "$resource_check_dir/$host.err" >&2
done
rm -rf "$resource_check_dir"
((resource_check_failed == 0)) || release_die "one or more Spark launch headroom checks failed"
report_wip_startup_phase launch-headroom
echo "  WIP slot: $slot"
echo "  SparkInfer H64 query projection: $SPARKINFER_GLM_H64_QUERY_PROJECTION"
echo "  fixed dSpark drafts: ${DSPARK_FIXED_DRAFTS:-adaptive}"
echo "  fixed DFlash2 drafts: ${resolved_dflash2_fixed_drafts:-inactive}"
if ((dry_run)); then
  echo "WIP dry-run checks passed."
  exit 0
fi

eval "$(jq -r '.environment | to_entries[] | "\(.key)=\(.value | @sh); export \(.key)"' "$resolved_json")"
export GLMRT_SPARK_HOSTS="$hosts_csv"
export GLMRT_REAL_FULL_SERVE_EXPERT_HOSTS="$expert_hosts_csv"
export GLMRT_SPARK_IMAGE="$SPARK_EXPERT_DOCKER_DEV"
export GLMRT_SPARK_EXISTING_CONTAINER="$spark_container"
export GLMRT_SPARK_RUNTIME_CACHE_DIR=/wip/cache
export GLMRT_SPARK_WORKDIR="$expert_workspace"
export GLMRT_SPARK_PREBUILT=1
export GLMRT_SPARK_PREBUILT_BIN="$expert_workspace/.glmrt-wip/glmrt"
export GLMRT_SPARK_PREBUILT_NATIVE_LIB="$expert_workspace/.glmrt-wip/libglmrt_native.so"
export GLMRT_SPARK_SKIP_STAGE=1
export GLMRT_RELEASE_CONFIG_SHA256="$expert_runtime_fingerprint"
export GLMRT_SPARK_EXPERT_PORT="$EXPERT_PORT"
export GLMRT_SPARK_EXPERT_TRANSPORT=verbs-host
export GLMRT_SPARK_KEEP_EXPERTS=1
export GLMRT_SPARK_EXPERT_REAL_LAYER=all
export GLMRT_PHASE0_SPARK_SKIP_BENCH=1
export GLMRT_EXPERT_INTERMEDIATE_RDMA_PEERS="$lane_a_csv"
if [[ -n "$lane_b_csv" ]]; then
  export GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS="$lane_b_csv"
else
  unset GLMRT_EXPERT_INTERMEDIATE_RDMA_ADDITIONAL_PEERS || true
fi
spark_start_pid=
if ((reuse_spark_experts)); then
  echo "== retaining WIP Spark expert processes =="
else
  echo "== starting WIP Spark expert processes =="
  "$repo_root/scripts/phase0-spark-tcp-bench.sh" &
  spark_start_pid=$!
fi
report_wip_startup_phase spark-dispatch

env_file="$state_dir/coordinator.env"
jq -r '.environment | to_entries[] | "\(.key)=\(.value)"' "$resolved_json" >"$env_file"
coordinator_power_limit_watts="$(
  nvidia-smi --id=0 --query-gpu=power.limit --format=csv,noheader,nounits |
    awk '{printf "%d\n", $1 + 0.5}'
)"
{
  echo "ADDR=$ADDR"
  echo "GLMRT_REAL_FULL_SERVE_EXPERT_HOSTS=$expert_hosts_csv"
  echo "GLMRT_SPARK_HOSTS=$hosts_csv"
  echo "GLMRT_SPARK_EXPERT_PORT=$EXPERT_PORT"
  echo "GLMRT_SPARKINFER_GLM_H64_BF16_QUERY_PROJECTION=$SPARKINFER_GLM_H64_QUERY_PROJECTION"
  echo "GLMRT_REAL_FULL_DSPARK_FIXED_DRAFTS=$DSPARK_FIXED_DRAFTS"
  echo "GLMRT_REAL_FULL_SERVE_START_EXPERTS=0"
  echo "GLMRT_REAL_FULL_SERVE_BUILD_DAEMON=0"
  echo "GLMRT_REAL_FULL_SERVE_BUILD_NATIVE=0"
  echo "GLMRT_REAL_FULL_SERVE_REQUIRE_CUDA=1"
  echo "GLMRT_REAL_FULL_SERVE_EXPERT_WARMUP_STATUS_FILE=/wip/run/expert-warmup.status"
  echo "GLMRT_BIN=$coordinator_workspace/.glmrt-wip/glmrt"
  echo "GLMRT_NATIVE_LIB=$coordinator_workspace/.glmrt-wip/libglmrt_native.so"
  echo "GLMRT_ENGINE_COMMIT=$coordinator_engine_commit"
  echo "GLMRT_SPARKINFER_COMMIT=$coordinator_sparkinfer_commit"
  echo "GLMRT_COORDINATOR_POWER_LIMIT_WATTS=$coordinator_power_limit_watts"
  echo "GLMRT_RELEASE_CONFIG_SHA256=$deployment_fingerprint"
  echo "GLMRT_KERNEL_CACHE_BASE=/wip/cache/kernels"
  echo "GLMRT_KERNEL_CACHE_ENVIRONMENT_ID=$coordinator_image_id"
  echo "GLMRT_RUNTIME_CATALOG_CACHE_DIR=/wip/cache/catalogs"
  echo "HF_HOME=$hf_home"
  echo "PYTHONPATH=$coordinator_workspace/third_party/sparkinfer:$coordinator_workspace/python/reference/glmrt_reference:$coordinator_workspace/python/reference:/opt/glmrt/third_party/sparkinfer"
} >>"$env_file"

# Keep graph-capture tracing opt-in and launcher-scoped. The persistent WIP
# container otherwise receives only the resolved production environment, which
# made an A/B graph-identity audit require manually editing its generated env
# file between restarts.
if [[ -n "${GLMRT_REAL_FULL_GRAPH_CAPTURE_TRACE:-}" ]]; then
  echo "GLMRT_REAL_FULL_GRAPH_CAPTURE_TRACE=$GLMRT_REAL_FULL_GRAPH_CAPTURE_TRACE" \
    >>"$env_file"
fi
if [[ -n "${GLMRT_REAL_FULL_DSPARK_TRACE:-}" ]]; then
  echo "GLMRT_REAL_FULL_DSPARK_TRACE=$GLMRT_REAL_FULL_DSPARK_TRACE" \
    >>"$env_file"
fi
if [[ -n "${GLMRT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP:-}" ]]; then
  echo "GLMRT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP=$GLMRT_REAL_FULL_DSPARK_PROFILE_AT_STARTUP" \
    >>"$env_file"
fi
if [[ -n "${GLMRT_REAL_FULL_DSPARK_PROFILE_SAMPLES:-}" ]]; then
  echo "GLMRT_REAL_FULL_DSPARK_PROFILE_SAMPLES=$GLMRT_REAL_FULL_DSPARK_PROFILE_SAMPLES" \
    >>"$env_file"
fi
# Timing switches are intentionally launcher-scoped: they are useful for a
# WIP A/B, but must not become part of the frozen production profile.  Spark
# switches are inherited by phase0-spark-tcp-bench.sh; mirror the coordinator
# switches into its generated environment so one launch profiles both ends of
# the same request.
for timing_env in \
  GLMRT_REAL_FULL_REQUEST_TIMING \
  GLMRT_REAL_FULL_SCHEDULER_TIMING \
  GLMRT_REAL_FULL_SCHEDULER_SUMMARY_TIMING \
  GLMRT_REAL_FULL_SPARSE_TCP_STAGE_TIMING \
  GLMRT_REAL_FULL_ATTENTION_CUDA_TIMING \
  GLMRT_PROTOCOL_V2_TCP_TIMING \
  GLMRT_REAL_FULL_PROTOCOL_V2_EXECUTOR_TIMING \
  GLMRT_REAL_FULL_NVFP4_ROUTE_TIMING \
  GLMRT_REAL_FULL_NVFP4_ROUTE_CUDA_EVENT_TIMING
do
  if [[ -n "${!timing_env:-}" ]]; then
    printf '%s=%s\n' "$timing_env" "${!timing_env}" >>"$env_file"
  fi
done
if [[ -n "${GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS:-}" ]]; then
  echo "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS=$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS" \
    >>"$env_file"
fi
if [[ -n "${GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES:-}" ]]; then
  echo "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES=$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES" \
    >>"$env_file"
fi
if [[ -n "${GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID:-}" ]]; then
  [[ "$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID" =~ ^[A-Za-z0-9_.-]{1,128}$ ]] ||
    release_die "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID must contain 1..128 alphanumeric, '.', '_', or '-' characters"
  echo "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID=$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID" \
    >>"$env_file"
fi
if [[ -n "${GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE:-}" ]]; then
  [[ "$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE" =~ ^/[A-Za-z0-9_./-]{1,256}$ ]] ||
    release_die "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE must be a simple absolute path"
  echo "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE=$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE" \
    >>"$env_file"
fi

echo "== starting coordinator process in $coordinator_container =="
if ! docker exec -d --env-file "$env_file" -w "$coordinator_workspace" \
  "$coordinator_container" \
  "$coordinator_workspace/scripts/wip-process.sh" run "$coordinator_process" \
  "$coordinator_workspace/scripts/real-full-tcp-serve.sh"; then
  kill "$spark_start_pid" >/dev/null 2>&1 || true
  wait "$spark_start_pid" >/dev/null 2>&1 || true
  release_die "failed to start WIP coordinator process"
fi
report_wip_startup_phase coordinator-dispatch

if [[ -n "$spark_start_pid" ]] && ! wait "$spark_start_pid"; then
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" log "$coordinator_process" 200 >&2 || true
  docker exec "$coordinator_container" \
    "$coordinator_workspace/scripts/wip-process.sh" stop "$coordinator_process" || true
  release_die "one or more WIP Spark experts failed during startup"
fi

docker exec "$coordinator_container" \
  "$coordinator_workspace/scripts/wip-process.sh" bind-identity \
  "$coordinator_process" "$deployment_fingerprint"
if ((!reuse_spark_experts)); then
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    ssh -o BatchMode=yes "$host" \
      "docker exec '$spark_container' '$expert_workspace/scripts/wip-process.sh' bind-identity '$expert_process' '$expert_runtime_fingerprint'"
  done
fi

deadline=$((SECONDS + 900))
until curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >/dev/null 2>&1; do
  coordinator_state="$(process_status_local)"
  if [[ "$coordinator_state" != running\ * ]]; then
    docker exec "$coordinator_container" \
      "$coordinator_workspace/scripts/wip-process.sh" log "$coordinator_process" 200 >&2 || true
    release_die "WIP coordinator process exited during startup"
  fi
  ((SECONDS < deadline)) || {
    docker exec "$coordinator_container" \
      "$coordinator_workspace/scripts/wip-process.sh" log "$coordinator_process" 200 >&2 || true
    release_die "WIP API did not become ready within 900 seconds"
  }
  sleep 0.25
done

curl -fsS "http://127.0.0.1:${ADDR##*:}/v1/models" >"$state_dir/models.json"
deployment_evidence="$state_dir/deployment.json"
python3 "$repo_root/scripts/write-wip-deployment-evidence.py" \
  --model-id "$RELEASE_MODEL_ID" \
  --model-revision "$coordinator_model_revision" \
  --slot "$slot" --profile "$PROFILE" --speculation "$SPECULATION" \
  --launch-started-ns "$launcher_started_ns" \
  --power-limit-w "$coordinator_power_limit_watts" \
  --coordinator-slot-fingerprint "$coordinator_slot_fingerprint" \
  --expert-slot-fingerprint "$expert_slot_fingerprint" \
  --expert-runtime-fingerprint "$expert_runtime_fingerprint" \
  --deployment-fingerprint "$deployment_fingerprint" \
  --engine-identity "$coordinator_engine_commit" \
  --sparkinfer-revision "$coordinator_sparkinfer_commit" \
  --resolved-profile "$resolved_json" --config "$config" \
  --output "$deployment_evidence" >/dev/null
report_wip_startup_phase api-ready
echo "GLMRT WIP server is ready at http://127.0.0.1:${ADDR##*:}/v1/"
echo "  slot:        $slot"
echo "  profile:     $PROFILE"
echo "  model:       $RELEASE_MODEL_ID"
echo "  speculation: $SPECULATION"
echo "  H64 query:   $SPARKINFER_GLM_H64_QUERY_PROJECTION"
if [[ "$SPECULATION" == dflash2 ]]; then
  echo "  fixed draft: $resolved_dflash2_fixed_drafts"
else
  echo "  fixed draft: ${DSPARK_FIXED_DRAFTS:-adaptive}"
fi
echo "  concurrency: $CONCURRENCY"
echo "  containers:  persistent $coordinator_container + four $spark_container"
echo "  Spark reuse: $([[ "$reuse_spark_experts" == 1 ]] && echo yes || echo no)"
echo "  evidence:    $deployment_evidence"
