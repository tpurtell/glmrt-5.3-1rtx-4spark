set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

model_id := env_var_or_default("GLMRT_MODEL_ID", "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1")
expert_roles := env_var_or_default("GLMRT_EXPERT_HOSTS", "spark-0,spark-1,spark-2,spark-3")
spark_hosts := env_var_or_default("GLMRT_SPARK_HOSTS", "ostrich,dodo,emu,kiwi")
base_image := env_var_or_default("GLMRT_CONTAINER_BASE", "nvcr.io/nvidia/pytorch:26.05-py3")
spark_catalog := env_var_or_default("GLMRT_PHASE0_SPARK_CATALOG", ".glmrt-cache/model-artifacts/diagnostic/model_catalog.json")
spark_loadplan_dir := env_var_or_default("GLMRT_PHASE0_SPARK_LOADPLAN_DIR", ".glmrt-cache/model-artifacts/diagnostic")
HOSTS := env_var_or_default("HOSTS", "")
MODE := env_var_or_default("MODE", "")

default:
  @just --list

doctor-host:
  scripts/doctor.sh --role coordinator --model-id "{{model_id}}"

doctor:
  scripts/glmrt doctor --role coordinator --model-id "{{model_id}}"

doctor-container-oliver:
  scripts/glmrt-dev.sh coordinator glmrt doctor --role coordinator --model-id "{{model_id}}"

doctor-hosts HOSTS=spark_hosts:
  scripts/run-on-hosts.sh "{{HOSTS}}" 'cd {{justfile_directory()}} && scripts/doctor.sh --role expert --model-id "{{model_id}}"'

build-rust:
  python="{{justfile_directory()}}/.venv/bin/python"; test -x "$python"; \
    python_home="$("$python" -c 'import sys; print(sys.base_prefix)')"; \
    PYTHONHOME="$python_home" \
    PYO3_PYTHON="$python" \
    cargo build --manifest-path rust/Cargo.toml --workspace

test-rust:
  python="{{justfile_directory()}}/.venv/bin/python"; test -x "$python"; \
    python_lib="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    python_home="$("$python" -c 'import sys; print(sys.base_prefix)')"; \
    LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONHOME="$python_home" \
    PYO3_PYTHON="$python" \
    cargo test --manifest-path rust/Cargo.toml --workspace

test-rust-fast:
  python="{{justfile_directory()}}/.venv/bin/python"; test -x "$python"; \
    python_lib="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    python_home="$("$python" -c 'import sys; print(sys.base_prefix)')"; \
    LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONHOME="$python_home" \
    PYO3_PYTHON="$python" \
    RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
    env -u GLMRT_NATIVE_LIB -u GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS -u GLMRT_B12X \
    GLMRT_DISABLE_NATIVE_AUTO_DISCOVERY=1 \
    cargo test --manifest-path rust/Cargo.toml --workspace --exclude glmrt-daemon
  python="{{justfile_directory()}}/.venv/bin/python"; test -x "$python"; \
    python_lib="$("$python" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    python_home="$("$python" -c 'import sys; print(sys.base_prefix)')"; \
    LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" \
    PYTHONHOME="$python_home" \
    PYO3_PYTHON="$python" \
    RUSTFLAGS="${RUSTFLAGS:--Awarnings}" \
    env -u GLMRT_NATIVE_LIB -u GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS -u GLMRT_B12X \
    GLMRT_DISABLE_NATIVE_AUTO_DISCOVERY=1 \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon -- \
      --skip real_checkpoint \
      --skip when_available \
      --skip when_cuda_available \
      --skip when_cuda_enabled \
      --skip native_available \
      --skip real_full_preflight \
      --skip real_full_runtime \
      --skip real_full_info_from_report \
      --skip uses_coord_dense_graph_slot \
      --skip uses_coord_sparse_a_graph_slot \
      --skip replays_same_bucket_when_rows_change \
      --skip cuda_graph \
      --skip b12x \
      --skip triton

build-native-coordinator-test:
  python="{{justfile_directory()}}/.venv/bin/python"; \
    test -x "$python"; \
    cmake -S native -B native/build-cuda-rdma-coordinator-aot -G Ninja \
      -U GLMRT_ENABLE_B12X_AOT \
      -U GLMRT_ENABLE_B12X_COORDINATOR_AOT \
      -DGLMRT_ENABLE_CUDA=ON \
      -DGLMRT_ENABLE_RDMA=ON \
      -DGLMRT_ENABLE_SPARKINFER_AOT=OFF \
      -DGLMRT_ENABLE_SPARKINFER_COORDINATOR_AOT=ON \
      -DGLMRT_ENABLE_W8A16_AOT=ON \
      -DGLMRT_SPARKINFER_SOURCE_DIR="{{justfile_directory()}}/third_party/sparkinfer" \
      -DGLMRT_SPARKINFER_LOCK_FILE="{{justfile_directory()}}/third_party/sparkinfer.lock.json" \
      -DGLMRT_ENABLE_NCCL=OFF \
      -DPython3_EXECUTABLE="$python" \
      -DGLMRT_CUDA_ARCHITECTURES=120
  cmake --build native/build-cuda-rdma-coordinator-aot -j 16
  @echo "coordinator CUDA test library: {{justfile_directory()}}/native/build-cuda-rdma-coordinator-aot/libglmrt_native.so"

test-rust-phase0a-focused NATIVE_LIB="native/build-cuda-rdma-coordinator-aot/libglmrt_native.so": build-native-coordinator-test
  native_lib="{{NATIVE_LIB}}"; \
    if [[ "$native_lib" != /* ]]; then native_lib="{{justfile_directory()}}/$native_lib"; fi; \
    test -f "$native_lib"; \
    printf 'GLMRT_NATIVE_LIB=%s\n' "$native_lib"; \
    python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    export LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}"; \
    export RUSTFLAGS="${RUSTFLAGS:--Awarnings}"; \
    export GLMRT_NATIVE_LIB="$native_lib"; \
    export GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-core graph -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-core kv_cache -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon bench_cuda_kernels::tests -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon real_full_nvfp4_kv_accounting_uses_targeted_dry_runs -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon device_kv_execution_mirror_writes_nvfp4_projected_mla_kv_a -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon uses_coord -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon replays_same_bucket_when_rows_change -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon uses_triton_graph_when_python_enabled -- --test-threads=1; \
    cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon coordinator_cuda_graph -- --test-threads=1

test-rust-cuda-graphs NATIVE_LIB="native/build-cuda-rdma-coordinator-aot/libglmrt_native.so": build-native-coordinator-test
  ctest --test-dir native/build-cuda-rdma-coordinator-aot --output-on-failure
  native_lib="{{NATIVE_LIB}}"; \
    if [[ "$native_lib" != /* ]]; then native_lib="{{justfile_directory()}}/$native_lib"; fi; \
    python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    test -f "$native_lib"; \
    printf 'GLMRT_NATIVE_LIB=%s\n' "$native_lib"; \
    LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" GLMRT_NATIVE_LIB="$native_lib" GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS=1 cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon coordinator_cuda_graph
  native_lib="{{NATIVE_LIB}}"; \
    if [[ "$native_lib" != /* ]]; then native_lib="{{justfile_directory()}}/$native_lib"; fi; \
    python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" GLMRT_NATIVE_LIB="$native_lib" GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS=1 cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon real_checkpoint_layer_ordered_execution_probe_when_available

test-rust-full-attention NATIVE_LIB="native/build-cuda-rdma-coordinator-aot/libglmrt_native.so": build-native-coordinator-test
  native_lib="{{NATIVE_LIB}}"; \
    if [[ "$native_lib" != /* ]]; then native_lib="{{justfile_directory()}}/$native_lib"; fi; \
    python_lib="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")')"; \
    test -f "$native_lib"; \
    printf 'GLMRT_NATIVE_LIB=%s\n' "$native_lib"; \
    LD_LIBRARY_PATH="$python_lib:${LD_LIBRARY_PATH:-}" GLMRT_NATIVE_LIB="$native_lib" GLMRT_REAL_FULL_CUDA_REFERENCE_KERNELS=1 cargo test --manifest-path rust/Cargo.toml -p glmrt-daemon real_checkpoint_layer_ordered_full_output_mla_rope_attention_mlp_probe_when_available -- --ignored

test-python:
  cd python && uv run pytest reference/tests ../scripts/tests

test-native:
  cmake -S native -B native/build -G Ninja -DGLMRT_ENABLE_CUDA=OFF -DGLMRT_ENABLE_RDMA=OFF
  cmake --build native/build
  ctest --test-dir native/build --output-on-failure
  GLMRT_NATIVE_LIB="{{justfile_directory()}}/native/build/libglmrt_native.so" cargo test --manifest-path rust/Cargo.toml -p glmrt-ffi

test-native-rdma OUT="reports/phase0_artifacts/native_rdma_enabled_build_status.json":
  python python/tools/check_native_rdma_build.py --clean --output "{{OUT}}"

test-smoke: doctor-host build-rust test-rust

docker-build-oliver:
  docker build \
    --platform linux/amd64 \
    --build-arg BASE_IMAGE="{{base_image}}" \
    --build-arg GLMRT_ROLE=coordinator \
    --build-arg CUDA_ARCH=120 \
    --build-arg TARGET_PLATFORM=linux/amd64 \
    -f docker/Dockerfile.dev \
    -t glmrt-dev:oliver .

docker-build-spark HOSTS=spark_hosts:
  GLMRT_SPARK_HOSTS="{{HOSTS}}" \
    GLMRT_PHASE0_SPARK_EXPERT_MODE=synthetic \
    GLMRT_SPARK_IMAGE_COPY_METHOD=none \
    GLMRT_SPARK_BUILD_IMAGE=1 \
    GLMRT_SPARK_FORCE_BUILD_IMAGE=1 \
    GLMRT_SPARK_IMAGE_ONLY=1 \
    scripts/phase0-spark-tcp-bench.sh

docker-gpu-check IMAGE="glmrt-dev:oliver":
  GLMRT_DOCKER_GPU_VERIFY_IMAGE="{{IMAGE}}" scripts/configure-docker-nvidia-runtime.sh --verify-only

docker-configure-nvidia-runtime:
  sudo scripts/configure-docker-nvidia-runtime.sh

docker-shell-oliver *ARGS:
  scripts/glmrt-dev.sh coordinator {{ARGS}}

docker-shell-spark *ARGS:
  scripts/glmrt-dev.sh expert {{ARGS}}

inspect-model:
  scripts/glmrt inspect-model --model-id "{{model_id}}" --out .glmrt-cache/model-artifacts/diagnostic/model_catalog.json --summary .glmrt-cache/model-artifacts/diagnostic/tensor_summary.md

make-loadplan POLICY="modulo":
  scripts/glmrt make-loadplan --catalog .glmrt-cache/model-artifacts/diagnostic/model_catalog.json --policy "{{POLICY}}" --hosts "{{expert_roles}}" --out .glmrt-cache/model-artifacts/diagnostic/loadplan.json

api-smoke MODEL="glmrt-tiny" URL="http://127.0.0.1:8000":
  scripts/api-smoke.sh "{{URL}}" "{{MODEL}}"

api-prefill-smoke MODEL="glmrt-synthetic-glm-layer" PROMPT_TOKENS="16" URL="http://127.0.0.1:8000":
  scripts/api-prefill-smoke.sh "{{URL}}" "{{MODEL}}" "{{PROMPT_TOKENS}}"

real-slice-tcp-smoke ADDR="127.0.0.1:8073" BASE_PORT="9181":
  ADDR="{{ADDR}}" BASE_PORT="{{BASE_PORT}}" scripts/real-slice-tcp-smoke.sh

real-full-tcp-smoke URL="http://127.0.0.1:8000" MODEL="{{model_id}}-full" MAX_TOKENS="1":
  scripts/real-full-tcp-smoke.sh "{{URL}}" "{{MODEL}}" "{{MAX_TOKENS}}"

real-full-tcp-smoke-multi-token URL="http://127.0.0.1:8000" MODEL="{{model_id}}-full" MAX_TOKENS="2":
  scripts/real-full-tcp-smoke.sh "{{URL}}" "{{MODEL}}" "{{MAX_TOKENS}}"

real-full-tcp-smoke-long-prefill URL="http://127.0.0.1:8000" MODEL="{{model_id}}-full" MAX_TOKENS="1":
  GLMRT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_TOKEN=chunk GLMRT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_COUNT=600 GLMRT_REAL_FULL_TCP_SMOKE_MIN_PREFILL_CHUNKS=2 GLMRT_REAL_FULL_TCP_SMOKE_REQUIRE_RUNTIME_SUMMARY=1 scripts/real-full-tcp-smoke.sh "{{URL}}" "{{MODEL}}" "{{MAX_TOKENS}}"

real-full-tcp-stream-smoke URL="http://127.0.0.1:8000" MODEL="{{model_id}}-full" MAX_TOKENS="1":
  scripts/real-full-tcp-stream-smoke.sh "{{URL}}" "{{MODEL}}" "{{MAX_TOKENS}}"

real-full-tcp-live-smoke:
  scripts/real-full-tcp-live-smoke.sh

real-full-tcp-live-smoke-synthetic:
  GLMRT_PHASE0_SPARK_EXPERT_MODE=synthetic LOG_PREFIX=real-full-tcp-live-smoke-synthetic scripts/real-full-tcp-live-smoke.sh

real-full-tcp-live-smoke-multi-token:
  LOG_PREFIX=real-full-tcp-live-smoke-multi-token MAX_TOKENS=2 scripts/real-full-tcp-live-smoke.sh

real-full-tcp-live-smoke-long-prefill:
  LOG_PREFIX=real-full-tcp-live-smoke-long-prefill GLMRT_REAL_FULL_REQUEST_PREFILL_CHUNK_TOKENS=64 GLMRT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_TOKEN=chunk GLMRT_REAL_FULL_TCP_SMOKE_PROMPT_REPEAT_COUNT=130 GLMRT_REAL_FULL_TCP_SMOKE_MIN_PREFILL_CHUNKS=2 GLMRT_REAL_FULL_TCP_SMOKE_WARMUP_PREFILL_ROUNDTRIP_ROWS=64,128 GLMRT_REAL_FULL_TCP_SMOKE_WARMUP_PREFILL_CHAIN_ROWS=64,128 scripts/real-full-tcp-live-smoke.sh

serve-real-full-tcp:
  scripts/real-full-tcp-serve.sh

transport-capabilities BENCHMARK_JSONL="reports/phase0_artifacts/benchmarks/phase0_results.jsonl" OUT="reports/phase0_artifacts/transport_capabilities.json":
  scripts/glmrt transport-capabilities --benchmark-jsonl "{{BENCHMARK_JSONL}}" --out "{{OUT}}"

scheduler-smoke:
  scripts/glmrt scheduler-smoke

start-coordinator MODE="tiny" TRANSPORT="inproc" ADDR="127.0.0.1:8000":
  scripts/glmrt coordinator --backend "{{MODE}}" --transport "{{TRANSPORT}}" --listen "{{ADDR}}" --model-id "{{model_id}}" --expert-hosts "{{expert_roles}}"

start-experts-tcp HOSTS=HOSTS MODE=MODE:
  hosts="${HOSTS:-}"; \
  mode="${MODE:-}"; \
  positional=0; \
  for arg in "{{HOSTS}}" "{{MODE}}"; do \
    [ -n "$arg" ] || continue; \
    case "$arg" in \
      HOSTS=*) hosts="${arg#HOSTS=}" ;; \
      MODE=*) mode="${arg#MODE=}" ;; \
      real|synthetic) mode="$arg" ;; \
      *) \
        if [ "$positional" -eq 0 ]; then \
          hosts="$arg"; \
          positional=1; \
        else \
          mode="$arg"; \
        fi \
        ;; \
    esac; \
  done; \
  hosts="${hosts:-{{spark_hosts}}}"; \
  mode="${mode:-synthetic}"; \
  case "$mode" in \
    synthetic) \
      scripts/start-spark-experts-tcp.sh \
        --hosts "$hosts" \
        --mode synthetic \
      ;; \
    real) \
      scripts/start-spark-experts-tcp.sh \
        --hosts "$hosts" \
        --mode real \
        --catalog "{{spark_catalog}}" \
        --loadplan-dir "{{spark_loadplan_dir}}" \
      ;; \
    *) \
      echo 'MODE must be synthetic or real' >&2; \
      exit 2 \
      ;; \
  esac

bench-rdma HOST_A HOST_B:
  scripts/bench-rdma-pair.sh "{{HOST_A}}" "{{HOST_B}}"

bench-verbs-app HOST_A HOST_B:
  scripts/bench-verbs-app-pair.sh "{{HOST_A}}" "{{HOST_B}}"

bench-verbs-app-coordinator HOSTS=spark_hosts:
  scripts/bench-verbs-app-coordinator-links.sh "{{HOSTS}}"

bench-phase0-spark-tcp HOSTS=spark_hosts MODE="real":
  GLMRT_SPARK_HOSTS="{{HOSTS}}" GLMRT_PHASE0_SPARK_EXPERT_MODE="{{MODE}}" scripts/phase0-spark-tcp-bench.sh
