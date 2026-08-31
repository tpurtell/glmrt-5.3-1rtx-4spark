#!/usr/bin/env bash

RELEASE_COORDINATOR_CONTAINER_NAME=glrmt-coordinator
RELEASE_SPARK_CONTAINER_PREFIX=glrmt-spark-expert

release_die() {
  echo "glmrt release: $*" >&2
  exit 2
}

release_need() {
  command -v "$1" >/dev/null 2>&1 || release_die "required command not found: $1"
}

release_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

release_known_key() {
  case "$1" in
    PROFILE|MODEL|COORDINATOR_GPU_HEADROOM_GIB|KV_POOL_TOKENS|MAX_CONTEXT_TOKENS|MAX_OUTPUT_TOKENS|CONCURRENCY|VISION_MODEL|SPECULATION|MTP_BF16_EXPERTS|DSPARK_MODEL|DSPARK_REVISION|DSPARK_FIXED_DRAFTS|DFLASH2_FIXED_DRAFTS|DFLASH2_TOPK_BACKEND|SPARKINFER_GLM_H64_QUERY_PROJECTION|ADDR|EXPERT_PORT|SPARK_[0-3]_HOST|SPARK_[0-3]_LANE_A|SPARK_[0-3]_LANE_B|COORDINATOR_DOCKER_DEV|COORDINATOR_DOCKER_INFERENCE|SPARK_EXPERT_DOCKER_DEV|SPARK_EXPERT_DOCKER_INFERENCE)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

release_load_config() {
  local config="$1"
  [[ -f "$config" ]] || release_die "configuration file not found: $config"

  PROFILE=balanced
  MODEL=glm53-exl3
  COORDINATOR_GPU_HEADROOM_GIB=8
  KV_POOL_TOKENS=
  MAX_CONTEXT_TOKENS=
  MAX_OUTPUT_TOKENS=
  CONCURRENCY=4
  VISION_MODEL=off
  SPECULATION=dflash2
  MTP_BF16_EXPERTS=auto
  DSPARK_MODEL=redhat
  DSPARK_REVISION=
  DSPARK_FIXED_DRAFTS=
  DFLASH2_FIXED_DRAFTS=
  DFLASH2_TOPK_BACKEND=torch
  SPARKINFER_GLM_H64_QUERY_PROJECTION=auto
  ADDR=0.0.0.0:8000
  EXPERT_PORT=9100
  COORDINATOR_DOCKER_DEV=glrmt-coordinator-dev
  COORDINATOR_DOCKER_INFERENCE=glrmt-coordinator
  SPARK_EXPERT_DOCKER_DEV=glrmt-spark-expert-dev
  SPARK_EXPERT_DOCKER_INFERENCE=glrmt-spark-expert
  for release_i in 0 1 2 3; do
    printf -v "SPARK_${release_i}_HOST" '%s' ""
    printf -v "SPARK_${release_i}_LANE_A" '%s' ""
    printf -v "SPARK_${release_i}_LANE_B" '%s' ""
  done

  local raw line key value
  while IFS= read -r raw || [[ -n "$raw" ]]; do
    line="$(release_trim "${raw%%#*}")"
    [[ -n "$line" ]] || continue
    [[ "$line" == *=* ]] || release_die "invalid configuration line: $raw"
    key="$(release_trim "${line%%=*}")"
    value="$(release_trim "${line#*=}")"
    release_known_key "$key" || release_die "unknown configuration key: $key"
    if [[ "$value" == \"*\" && "$value" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "$value" == *[[:space:]]* ]]; then
      release_die "unquoted whitespace is not allowed for $key"
    fi
    printf -v "$key" '%s' "$value"
  done <"$config"

  case "$PROFILE" in balanced|long|accuracy) ;; *) release_die "PROFILE must be balanced, long, or accuracy" ;; esac
  case "$SPECULATION" in plain|mtp|dspark|dflash2) ;; *) release_die "SPECULATION must be plain, mtp, dspark, or dflash2" ;; esac
  case "$MTP_BF16_EXPERTS" in auto|on|off) ;; *) release_die "MTP_BF16_EXPERTS must be auto, on, or off" ;; esac
  case "$SPARKINFER_GLM_H64_QUERY_PROJECTION" in
    auto|disable|force) ;;
    *) release_die "SPARKINFER_GLM_H64_QUERY_PROJECTION must be auto, disable, or force" ;;
  esac
  if [[ -n "$DSPARK_FIXED_DRAFTS" ]]; then
    [[ "$DSPARK_FIXED_DRAFTS" =~ ^[0-7]$ ]] ||
      release_die "DSPARK_FIXED_DRAFTS must be empty or in 0..7"
    [[ "$SPECULATION" == dspark ]] ||
      release_die "DSPARK_FIXED_DRAFTS requires SPECULATION=dspark"
  fi
  if [[ -n "$DFLASH2_FIXED_DRAFTS" ]]; then
    [[ "$DFLASH2_FIXED_DRAFTS" =~ ^[1-7]$ ]] ||
      release_die "DFLASH2_FIXED_DRAFTS must be empty or in 1..7; use SPECULATION=plain for target-only"
    [[ "$SPECULATION" == dflash2 ]] ||
      release_die "DFLASH2_FIXED_DRAFTS requires SPECULATION=dflash2"
  fi
  case "$DFLASH2_TOPK_BACKEND" in
    torch|flashinfer|flashinfer-dsa) ;;
    *) release_die "DFLASH2_TOPK_BACKEND must be torch, flashinfer, or flashinfer-dsa" ;;
  esac
  case "${VISION_MODEL,,}" in ""|off|none|baseten) ;; *) release_die "VISION_MODEL must be baseten, off, or unset" ;; esac
  [[ "$CONCURRENCY" =~ ^[1-8]$ ]] || release_die "CONCURRENCY must be in 1..8"
  [[ "$EXPERT_PORT" =~ ^[0-9]+$ ]] && ((EXPERT_PORT >= 1 && EXPERT_PORT <= 65535)) || release_die "EXPERT_PORT must be in 1..65535"
  [[ "$COORDINATOR_GPU_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] || release_die "COORDINATOR_GPU_HEADROOM_GIB must be non-negative"
  for release_integer_name in KV_POOL_TOKENS MAX_CONTEXT_TOKENS MAX_OUTPUT_TOKENS; do
    value="${!release_integer_name}"
    [[ -z "$value" || "$value" =~ ^[1-9][0-9]*$ ]] || release_die "$release_integer_name must be a positive integer"
  done
  if [[ -n "$KV_POOL_TOKENS" ]]; then
    ((KV_POOL_TOKENS % 64 == 0)) || release_die "KV_POOL_TOKENS must be a multiple of 64"
  fi
  [[ "$ADDR" == *:* ]] || release_die "ADDR must be HOST:PORT"
  [[ -n "$MODEL" ]] || release_die "MODEL must not be empty"

  local missing_b=0 present_b=0
  for release_i in 0 1 2 3; do
    local host_name="SPARK_${release_i}_HOST"
    local lane_a_name="SPARK_${release_i}_LANE_A"
    local lane_b_name="SPARK_${release_i}_LANE_B"
    [[ -n "${!host_name}" ]] || release_die "$host_name must not be empty"
    [[ "${!lane_a_name}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]] || release_die "$lane_a_name must be an IPv4 address"
    if [[ -n "${!lane_b_name}" ]]; then
      ((present_b += 1))
    else
      ((missing_b += 1))
    fi
  done
  ((present_b == 0 || missing_b == 0)) || release_die "secondary Spark rail must provide all four LANE_B values or none"

  for release_image_name in COORDINATOR_DOCKER_DEV COORDINATOR_DOCKER_INFERENCE SPARK_EXPERT_DOCKER_DEV SPARK_EXPERT_DOCKER_INFERENCE; do
    [[ -n "${!release_image_name}" && "${!release_image_name}" != *[[:space:]]* ]] || release_die "$release_image_name must be a Docker image reference"
  done

  RELEASE_CONFIG="$(realpath "$config")"
  case "${VISION_MODEL,,}" in baseten) RELEASE_VISION=on ;; *) RELEASE_VISION=off ;; esac
  case "${MODEL,,}" in
    luke) RELEASE_MODEL_ID=lukealonso/GLM-5.2-NVFP4 ;;
    nvidia) RELEASE_MODEL_ID=nvidia/GLM-5.2-NVFP4 ;;
    exl3) RELEASE_MODEL_ID=wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1 ;;
    glm53-exl3) RELEASE_MODEL_ID=wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1 ;;
    *) release_die "MODEL must be luke, nvidia, exl3, or glm53-exl3" ;;
  esac
  case "${DSPARK_MODEL,,}" in
    redhat)
      RELEASE_DSPARK_MODEL_ID=RedHatAI/GLM-5.2-speculator.dspark
      RELEASE_DSPARK_REVISION="${DSPARK_REVISION:-8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4}"
      ;;
    *) release_die "DSPARK_MODEL must be redhat" ;;
  esac
}

release_hosts_csv() {
  printf '%s,%s,%s,%s' "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"
}

release_lane_a_csv() {
  printf '%s,%s,%s,%s' "$SPARK_0_LANE_A" "$SPARK_1_LANE_A" "$SPARK_2_LANE_A" "$SPARK_3_LANE_A"
}

release_lane_b_csv() {
  if [[ -z "$SPARK_0_LANE_B" ]]; then
    return
  fi
  printf '%s,%s,%s,%s' "$SPARK_0_LANE_B" "$SPARK_1_LANE_B" "$SPARK_2_LANE_B" "$SPARK_3_LANE_B"
}

release_expert_hosts_csv() {
  printf '%s=%s:%s,%s=%s:%s,%s=%s:%s,%s=%s:%s' \
    "spark-0" "$SPARK_0_LANE_A" "$EXPERT_PORT" \
    "spark-1" "$SPARK_1_LANE_A" "$EXPERT_PORT" \
    "spark-2" "$SPARK_2_LANE_A" "$EXPERT_PORT" \
    "spark-3" "$SPARK_3_LANE_A" "$EXPERT_PORT"
}

release_stop_local_container() {
  local container="$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  coordinator: stopping $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  coordinator: removing stopped $container"
  fi
  docker rm -f "$container" >/dev/null
}

release_stop_host_api() {
  local addr="$1"
  local port="${addr##*:}"
  local pids
  pids="$(
    ss -ltnp "sport = :$port" 2>/dev/null |
      sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' |
      sort -u
  )"
  [[ -n "$pids" ]] || return 0
  local pid command
  for pid in $pids; do
    command="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$command" == *glmrt*coordinator* ]] ||
      release_die "port $port is owned by a non-GLMRT process: pid=$pid $command"
    echo "  coordinator: stopping host API pid=$pid"
    kill -TERM "$pid"
  done
  for _ in $(seq 1 300); do
    ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q . || return 0
    sleep 0.1
  done
  release_die "host API did not exit within 30 seconds"
}

release_stop_remote_containers() {
  local host="$1"
  local release_container="$2"
  local legacy_container="$3"
  ssh -o BatchMode=yes "$host" bash -s -- \
    "$host" "$release_container" "$legacy_container" <<'REMOTE'
set -euo pipefail
host="$1"
shift
for container in "$@"; do
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    continue
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  $host: stopping $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  $host: removing stopped $container"
  fi
  docker rm -f "$container" >/dev/null
done
REMOTE
}

release_stop_services() {
  local coordinator_container="$1"
  local spark_container_prefix="$2"
  release_stop_local_container "$coordinator_container"
  release_stop_host_api "$ADDR"

  local host release_container legacy_container
  local failed=0
  local -a stop_hosts=()
  local -a stop_pids=()
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    release_container="${spark_container_prefix}-${host}-${EXPERT_PORT}"
    legacy_container="glmrt-phase0-tcp-expertd-${host}-${EXPERT_PORT}"
    release_stop_remote_containers \
      "$host" "$release_container" "$legacy_container" &
    stop_hosts+=("$host")
    stop_pids+=("$!")
  done
  local index
  for index in "${!stop_pids[@]}"; do
    if ! wait "${stop_pids[$index]}"; then
      echo "  ${stop_hosts[$index]}: failed to stop one or more GLMRT containers" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

release_stop_persistent_local_container() {
  local container="$1"
  if ! docker container inspect "$container" >/dev/null 2>&1; then
    return
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
    echo "  coordinator: stopping persistent $container"
    docker stop -t 30 "$container" >/dev/null
  else
    echo "  coordinator: persistent $container is already stopped"
  fi
}

release_stop_persistent_remote_container() {
  local host="$1"
  local container="$2"
  ssh -o BatchMode=yes "$host" bash -s -- "$host" "$container" <<'REMOTE'
set -euo pipefail
host="$1"
container="$2"
if ! docker container inspect "$container" >/dev/null 2>&1; then
  exit 0
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" == true ]]; then
  echo "  $host: stopping persistent $container"
  docker stop -t 30 "$container" >/dev/null
else
  echo "  $host: persistent $container is already stopped"
fi
REMOTE
}

release_stop_wip_containers() {
  local coordinator_container="${1:-glrmt-coordinator-wip}"
  local spark_container="${2:-glrmt-spark-expert-wip}"
  local failed=0

  release_stop_persistent_local_container "$coordinator_container" || failed=1

  local host
  local -a hosts=() pids=()
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    release_stop_persistent_remote_container "$host" "$spark_container" &
    hosts+=("$host")
    pids+=("$!")
  done
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "  ${hosts[$index]}: failed to stop persistent $spark_container" >&2
      failed=1
    fi
  done
  ((failed == 0))
}

release_stop_wip_process_in_container() {
  local container="$1"
  local process_name="$2"
  docker exec -i "$container" bash -s -- "$process_name" <<'CONTAINER'
set -euo pipefail
name="$1"
pid_file="/wip/run/$name.pid"
identity_file="/wip/run/$name.identity"
[ -f "$pid_file" ] || exit 0
pid="$(<"$pid_file")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file" "$identity_file"
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
case "$command_line" in
  *wip-process.sh*run*"$name"*) ;;
  *) echo "refusing to stop stale WIP pid $pid for $name: $command_line" >&2; exit 2 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$pid_file" "$identity_file"; exit 0; }
  sleep 0.1
done
echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
exit 2
CONTAINER
}

release_stop_wip_coordinator() {
  local coordinator_process="${1:-coordinator-${ADDR##*:}}"
  local coordinator_container=glrmt-coordinator-wip

  if docker container inspect "$coordinator_container" >/dev/null 2>&1 &&
    [[ "$(docker inspect -f '{{.State.Running}}' "$coordinator_container")" == true ]]; then
    echo "  coordinator: stopping WIP process $coordinator_process"
    release_stop_wip_process_in_container \
      "$coordinator_container" "$coordinator_process"
  fi
}

release_stop_wip_services() {
  local coordinator_process="${1:-coordinator-${ADDR##*:}}"
  local expert_process="${2:-expert-$EXPERT_PORT}"
  local coordinator_container=glrmt-coordinator-wip
  local spark_container=glrmt-spark-expert-wip
  local failed=0

  release_stop_wip_coordinator "$coordinator_process" || failed=1

  local host
  local -a hosts=() pids=()
  for host in "$SPARK_0_HOST" "$SPARK_1_HOST" "$SPARK_2_HOST" "$SPARK_3_HOST"; do
    (
      if ! ssh -o BatchMode=yes "$host" \
        "test \"\$(docker inspect -f '{{.State.Running}}' '$spark_container' 2>/dev/null || true)\" = true"; then
        exit 0
      fi
      ssh -o BatchMode=yes "$host" docker exec -i "$spark_container" \
        bash -s -- "$expert_process" <<'CONTAINER'
set -euo pipefail
name="$1"
pid_file="/wip/run/$name.pid"
identity_file="/wip/run/$name.identity"
[ -f "$pid_file" ] || exit 0
pid="$(<"$pid_file")"
if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  rm -f "$pid_file" "$identity_file"
  exit 0
fi
command_line="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
case "$command_line" in
  *wip-process.sh*run*"$name"*) ;;
  *) echo "refusing to stop stale WIP pid $pid for $name: $command_line" >&2; exit 2 ;;
esac
kill -TERM "$pid"
for _ in $(seq 1 300); do
  kill -0 "$pid" 2>/dev/null || { rm -f "$pid_file" "$identity_file"; exit 0; }
  sleep 0.1
done
echo "WIP process did not stop within 30 seconds: $name pid=$pid" >&2
exit 2
CONTAINER
    ) &
    hosts+=("$host")
    pids+=("$!")
  done
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "  ${hosts[$index]}: failed to stop WIP process $expert_process" >&2
      failed=1
    fi
  done
  ((failed == 0))
}
