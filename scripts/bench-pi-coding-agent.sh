#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: scripts/bench-pi-coding-agent.sh --root DIR --model MODEL
       --thinking off|high [--pi BINARY] [--node BINARY]

Runs the fixed isolated Pi WebGL coding task once and writes a signed evidence
report under DIR. DIR must not already exist. The configured `glmrt` Pi
provider must target http://127.0.0.1:8000/v1 with temperature zero.
EOF
}

root=
model=
thinking=
pi_binary=pi
node_binary=node
while (($#)); do
  case "$1" in
    --root) root="${2:?--root requires a directory}"; shift 2 ;;
    --model) model="${2:?--model requires an ID}"; shift 2 ;;
    --thinking) thinking="${2:?--thinking requires off or high}"; shift 2 ;;
    --pi) pi_binary="${2:?--pi requires a binary}"; shift 2 ;;
    --node) node_binary="${2:?--node requires a binary}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$root" && -n "$model" ]] || { usage >&2; exit 2; }
[[ "$thinking" == off || "$thinking" == high ]] || {
  echo "--thinking must be off or high" >&2
  exit 2
}
command -v "$pi_binary" >/dev/null || { echo "Pi binary not found: $pi_binary" >&2; exit 2; }
command -v "$node_binary" >/dev/null || { echo "Node binary not found: $node_binary" >&2; exit 2; }
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
command -v /usr/bin/time >/dev/null || { echo "/usr/bin/time is required" >&2; exit 2; }

pi_config_dir="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
settings="$pi_config_dir/settings.json"
models="$pi_config_dir/models.json"
jq -e '.temperature == 0' "$settings" >/dev/null || {
  echo "Pi settings must set temperature to zero: $settings" >&2
  exit 2
}
jq -e '
  .providers.glmrt.baseUrl == "http://127.0.0.1:8000/v1" and
  .providers.glmrt.api == "openai-completions"
' "$models" >/dev/null || {
  echo "Pi glmrt provider is not the pinned local OpenAI-completions endpoint" >&2
  exit 2
}

expanded_root="$(realpath -m "$root")"
[[ ! -e "$expanded_root" && ! -L "$expanded_root" ]] || {
  echo "benchmark root already exists: $expanded_root" >&2
  exit 2
}
mkdir -p "$expanded_root/work"

prompt='make a webgl game of a parrot flying around to steal food from people'
pi_version="$($pi_binary --version)"
(
  cd "$expanded_root/work"
  /usr/bin/time \
    -f $'elapsed_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\nmax_rss_kb=%M\nexit_status=%x' \
    -o "$expanded_root/time.txt" \
    "$pi_binary" \
      --provider glmrt --model "$model" --thinking "$thinking" \
      --mode json --print --no-session --no-context-files \
      --no-extensions --no-skills --no-prompt-templates \
      --tools read,bash,edit,write --no-approve \
      "$prompt" \
      2>"$expanded_root/stderr.log" \
    | jq -c 'select(.type != "message_update")' \
      >"$expanded_root/events.jsonl"
)

python3 "$repo_root/python/tools/validate_pi_coding_agent_run.py" \
  --events "$expanded_root/events.jsonl" \
  --time "$expanded_root/time.txt" \
  --stderr "$expanded_root/stderr.log" \
  --work "$expanded_root/work" \
  --model "$model" --thinking "$thinking" --pi-version "$pi_version" \
  --node "$node_binary" --output "$expanded_root/evidence.json"
