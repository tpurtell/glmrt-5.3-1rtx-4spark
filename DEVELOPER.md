# GLMRT agent development index

> **Agent helper:** this file is a dense orientation and command index for
> coding agents customizing GLMRT. It is not a human onboarding tutorial.
> Assume commands run from the repository root, inspect the worktree first,
> preserve unrelated changes, and use focused tests before broad live gates.

## Stable system contract

- The release topology is one x86_64 `sm_120` coordinator plus four ARM64
  `sm_121` DGX Spark expert hosts; changing the host count is an architecture
  change, not a configuration edit.
- Runtime roles are always `coordinator` and `spark-0` through `spark-3`;
  `SPARK_N_HOST` values are SSH/deployment names only.
- The coordinator owns API, scheduling, residuals, attention, target KV/DSA,
  dense/shared MLPs, LM head, sampling, and speculation control.
- Sparks own routed-expert weights/execution and distributed expert reduction;
  they are not full transformer replicas.
- Release serving uses `verbs-host`, ProtocolV2, four qualified execution/QP
  lanes, GPU-resident weights, direct packed attention, and no timed graph
  captures.
- `balanced`, `long`, and `accuracy` are launch profiles over one engine;
  the public GLM-5.3 path qualifies `plain`, `mtp`, and `dflash2` speculation.
- `MODEL=glm53-exl3` selects the calibrated GLM-5.3 EXL3 K4 checkpoint;
  adaptive DFlash2 K1–K7 is the production default.
- The EXL3 artifact keeps attention, dense/shared MLP, router, and other
  non-routed-expert tensors in their source dtype; only routed experts in
  layers 3–77 use native K4/MCG tensors.
- DFlash2 consumes six target residual taps, keeps bounded request-local draft
  KV, and selects verification width from the checked-in route-cost profile.
- Production builds derive tensor catalogs and deterministic modulo expert
  placement directly from local safetensors metadata; generated catalog and
  load-plan JSON files are diagnostic only.
- Model/runtime caches and generated output belong under ignored
  `.glmrt-cache/`, `.glmrt-release/`, `.glmrt-release-image/`, `dist/`,
  `rust/target/`, or `native/build*`, not in source control.

## Session setup

| Command | One-line use |
| --- | --- |
| `git status --short` | Establish the user's existing changes before editing. |
| `just --list` | List the maintained command surface. |
| `source .venv/bin/activate` | Enter the optional host Python 3.12 environment when it exists. |
| `export PATH=/usr/local/cuda/bin:$PATH` | Make the coordinator CUDA toolkit visible to CMake/NVCC. |
| `export GLMRT_PYTHON="$PWD/.venv/bin/python"` | Select the embedded Python used by host builds. |
| `export PYO3_PYTHON="$GLMRT_PYTHON"` | Bind PyO3 compilation to that same interpreter. |
| `just doctor-host` | Inspect coordinator GPU, toolchain, RDMA, routes, and model cache. |
| `just doctor-hosts` | Run the expert doctor over the four default SSH hosts. |
| `./run.sh --dry-run` | Validate the complete release configuration and current deployment without mutation. |

For a direct binary or focused Cargo test, derive the Python library directory
instead of hard-coding a Python patch version:

```bash
GLMRT_PYTHON_LIBDIR="$(
  "${GLMRT_PYTHON:-python3}" -c \
    'import sysconfig; print(sysconfig.get_config_var("LIBDIR") or "")'
)"
export LD_LIBRARY_PATH="$GLMRT_PYTHON_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Do not export `PYTHONHOME`; it breaks embedded interpreter discovery.

## Build and static checks

| Command | One-line use |
| --- | --- |
| `cargo fmt --manifest-path rust/Cargo.toml --all` | Format the complete Rust workspace. |
| `cargo fmt --manifest-path rust/Cargo.toml --all --check` | Check Rust formatting without changing files. |
| `cargo check --manifest-path rust/Cargo.toml --workspace` | Type-check every Rust crate without linking release artifacts. |
| `just build-rust` | Build the full Rust workspace in the host environment. |
| `PYO3_PYTHON="$PWD/.venv/bin/python" cargo build --manifest-path rust/Cargo.toml -p glmrt-daemon --release` | Build the release `glmrt` binary directly. |
| `just build-native-coordinator-test` | Incrementally build the production-shaped coordinator CUDA/RDMA/AOT library. |
| `cmake --build native/build-cuda-rdma-coordinator-aot -j 16` | Rebuild an already-configured coordinator native tree. |
| `python3 -m compileall -q python/reference/glmrt_reference python/tools` | Catch Python syntax/import compilation failures. |
| `bash -n build.sh run.sh scripts/*.sh docker/*.sh` | Syntax-check shell entry points and wrappers. |
| `git diff --check` | Reject whitespace errors and conflict markers before handoff. |
| `./build.sh` | Build both release architectures, export artifacts, and distribute the expert image. |

## Test ladder

| Command | One-line use |
| --- | --- |
| `just test-rust-fast` | Run the normal fast Rust gate with stale native auto-discovery disabled. |
| `just test-rust` | Run the full Rust workspace test suite. |
| `cargo test --manifest-path rust/Cargo.toml -p CRATE TEST_FILTER -- --nocapture` | Run one focused Rust crate/test filter while iterating. |
| `just test-python` | Run all Python reference/oracle tests through `uv`. |
| `PYTHONPATH=python/reference python3 -m pytest -q PATH_OR_TEST` | Run a focused Python reference test. |
| `just test-native` | Build/test the CPU-only native library and its Rust FFI boundary. |
| `just test-rust-cuda-graphs` | Rebuild the coordinator AOT library, run native CTest, and exercise CUDA graph integration. |
| `just test-rust-phase0a-focused` | Run focused graph, KV, kernel, and production-path daemon tests with the AOT library. |
| `just test-rust-full-attention` | Run the ignored real-checkpoint full-attention probe; requires model/GPU resources. |
| `just test-native-rdma` | Perform a clean native RDMA build check and write a status artifact. |
| `just test-smoke` | Run host doctor, Rust build, and full Rust tests. |

Pass the current library explicitly when a focused CUDA-backed daemon test
needs native symbols:

```bash
GLMRT_NATIVE_LIB="$PWD/native/build-cuda-rdma-coordinator-aot/libglmrt_native.so" \
  cargo test --manifest-path rust/Cargo.toml \
  -p glmrt-daemon TEST_FILTER -- --test-threads=1
```

Do not qualify against an old ignored `native/build-cuda` library. A missing
FFI symbol usually means the test loaded a stale native artifact.

## Containers and deployment

| Command | One-line use |
| --- | --- |
| `just docker-build-oliver` | Build the historical-name coordinator development image (`amd64`, `sm_120`). |
| `just docker-build-spark` | Build/stage the Spark development image (`arm64`, `sm_121`). |
| `just docker-shell-oliver bash` | Open a GPU-enabled coordinator development container. |
| `just docker-shell-spark bash` | Open a GPU/RDMA-enabled Spark development container. |
| `just docker-gpu-check` | Verify Docker NVIDIA runtime access with the selected image. |
| `./run.sh --dry-run` | Check images, SSH, snapshots, capacity, and resolved profile without mutation. |
| `./run.sh` | Start a clean deployment or leave an identical healthy deployment untouched. |
| `./run.sh --restart` | Stop exact GLMRT services and recreate all five containers. |
| `./run.sh --profile FILE --restart` | Restart using a complete alternate config file. |
| `./push-containers.sh TAG` | Tag and push both release images as `TAG` and `latest` to GHCR. |
| `docker logs -f glrmt-coordinator` | Follow coordinator startup and request logs. |
| `ssh HOST docker logs -f glrmt-spark-expert-HOST-9100` | Follow one release expert log; substitute the configured host/port. |
| `sha256sum -c dist/SHA256SUMS` | Verify exported coordinator and expert binaries/libraries. |

`build.sh` compiles x86 artifacts locally and ARM artifacts natively on
`SPARK_0_HOST`; it intentionally does not use QEMU/buildx cross-emulation.
`run.sh` waits up to 15 minutes for startup prewarm and API readiness.

## API and live smoke

| Command | One-line use |
| --- | --- |
| `curl -fsS http://127.0.0.1:8000/health` | Check coordinator health after prewarm. |
| `curl -fsS http://127.0.0.1:8000/v1/models` | Read the exact served model ID. |
| `just api-smoke` | Exercise the lightweight synthetic OpenAI-compatible route. |
| `just real-full-tcp-smoke` | Check one-token real-model API execution against resident experts. |
| `just real-full-tcp-smoke-multi-token` | Check a short multi-token non-streaming response. |
| `just real-full-tcp-stream-smoke` | Check streaming chunks through `[DONE]`. |
| `just real-full-tcp-smoke-long-prefill` | Check multi-chunk real-model prefill. |
| `just real-full-tcp-live-smoke` | Launch the production-shaped expert/API smoke workflow. |
| `just real-full-tcp-live-smoke-long-prefill` | Launch the production-shaped long-prefill smoke workflow. |

After API, sampling, streaming, tool, or scheduler changes, test both
non-streaming and SSE paths. Preserve `reasoning_content`, tool-call argument
streaming, usage, finish reason, and continuation token identity.

## Bench and diagnostic commands

| Command | One-line use |
| --- | --- |
| `just bench-phase0-spark-tcp` | Run the real Spark ProtocolV2 phase-0 benchmark workflow. |
| `just bench-verbs-app HOST_A HOST_B` | Measure application-level ProtocolV2 verbs between two Sparks. |
| `just bench-verbs-app-coordinator` | Measure coordinator-to-Spark ProtocolV2 verbs links. |
| `just bench-rdma HOST_A HOST_B` | Run vendor/component RDMA checks; not a substitute for app-level evidence. |
| `python3 python/tools/bench_real_full_exact_decode.py --help` | Discover the exact scalar decode gate options. |
| `python3 python/tools/bench_real_full_concurrency.py --help` | Discover real-model C=1/C=2/C=4 concurrency options. |
| `python3 python/tools/bench_real_full_prefill_concurrency.py --help` | Discover cache-cold prefill concurrency options. |
| `python3 python/tools/bench_real_full_long_context_session.py --help` | Discover the growing semantic long-context gate. |
| `python3 python/tools/bench_real_full_mtp_acceptance.py --help` | Discover native-MTP/DFlash2 acceptance and throughput options. |
| `python3 python/tools/bench_release_prefill_matrix.py --help` | Run the exact cache-aware public prefill matrix or a selected cell. |
| `python3 python/tools/bench_release_decode_matrix.py --help` | Run the retained-context GLM-5.3 decode matrix. |
| `python3 python/tools/bench_real_full_needle.py --help` | Run exact long-context needle-recall cases. |
| `python3 python/tools/validate_glm53_exl3_serving_qualification.py --help` | Validate matched native-MTP and DFlash2 release evidence. |
| `tool-eval-bench --base-url http://127.0.0.1:8000/v1/ --parallel 1 --model MODEL` | Run the serial 69-scenario tool-call correctness gate. |
| `python3 scripts/bench-real-full-mixed-concurrency.py --help` | Discover mixed streaming/admission/cancellation scenarios. |
| `just inspect-model` | Build a diagnostic tensor catalog from the selected snapshot. |
| `just make-loadplan` | Build a diagnostic modulo expert load plan; release serving does not consume it. |
| `just transport-capabilities` | Summarize transport evidence from benchmark JSONL. |

Benchmark only after startup prewarm and first weight touches have settled.
Record commit, image revision, config/profile, model revision, workload shape,
sample count, warmups, mean/median/spread, runtime captures, and correctness.
Treat target-only decode, speculative wall throughput, prefill, long-context
retention, and aggregate concurrency as separate measurements.

## Repository map

| Path | Condensed responsibility |
| --- | --- |
| `build.sh` | Build both architecture-specific development/inference images, export `glmrt` + native libraries, and distribute the Spark image. |
| `run.sh` | Validate and manage the five-container release lifecycle. |
| `glmrt.config` | Operator-owned profile, model, speculation, capacity, host, rail, and image selection. |
| `justfile` | Maintained developer command index; historical `oliver` names mean the current coordinator role. |
| `rust/crates/glmrt-api/` | Axum/OpenAI boundary, request translation, streaming, tools, continuation identity, metrics, and vision input. |
| `rust/crates/glmrt-core/` | IDs, placement, layer-wave scheduling, graph buffers, target KV allocation/backing, route plans, and invariants. |
| `rust/crates/glmrt-daemon/` | `glmrt` CLI plus coordinator, expert daemon, real-full execution, preflight, model tools, and benchmark commands. |
| `rust/crates/glmrt-ffi/` | Dynamic native-library loading and Rust wrappers for the C ABI. |
| `rust/crates/glmrt-loader/` | Safetensors snapshot/catalog inspection, tokenizer loading, tensor metadata, and deterministic placement. |
| `rust/crates/glmrt-transport/` | ExpertProtocolV2 framing/batching plus TCP, verbs, synthetic, capabilities, and metrics paths. |
| `native/include/glmrt_native.h` | Stable C ABI between Rust and the native library. |
| `native/src/` | C++ native dispatch, validation, resource ownership, and ABI implementation. |
| `native/cuda/kernels/` | CUDA kernels for attention/KV, routing/MoE, projections, residuals, sampling, DFlash2, and AOT launchers. |
| `native/tests/` | CPU/native and CUDA self-tests. |
| `native/tools/` | Native benchmark utilities, including constrained-generation/XGrammar coverage. |
| `python/reference/glmrt_reference/` | Independent math/config oracles and PyO3 graph-capture adapters used by tests/runtime. |
| `python/reference/tests/` | Reference correctness and profile contract tests. |
| `python/tools/` | Kernel exporters, tuners, model/cache inspection, real-model benchmarks, and vision worker. |
| `scripts/` | Development launcher, doctors, remote staging, expert lifecycle, API smokes, release support, and network benchmarks. |
| `docker/` | Development and inference Dockerfiles plus runtime entrypoints. |
| `quantization/` | Reproducible calibrated GLM-5.3 routed-expert EXL3 K4 recipe, validation, and model-card tooling. |
| `third_party/gptqmodel/` | Pinned build-only GPTQModel fork used by the EXL3 quantization image. |
| `third_party/sparkinfer/` | Pinned runtime/build source for EXL3 K4 Spark expert kernels. |
| `dist/` | Ignored exported release binaries/libraries and `SHA256SUMS`. |

## Change routing

| Change | Start here | Minimum relevant gate |
| --- | --- | --- |
| OpenAI request/response, SSE, tools | `rust/crates/glmrt-api/` | Focused API tests + stream smoke |
| Scheduler, admission, cache ownership | `glmrt-core/layerwave`, `glmrt-core/kv_cache`, daemon `real_full` | Fast Rust + CUDA graphs + C=4 live isolation |
| Model mapping/placement/tokenizer | `glmrt-loader`, daemon `model_artifacts` | Loader/core tests + real preflight |
| Protocol/frame/TCP/verbs | `glmrt-transport`, native RDMA ABI | Transport tests + app-level verbs bench |
| Coordinator CUDA path | daemon `real_full/coordinator_kernels`, `native/cuda/kernels` | Native CTest + CUDA graph gate + targeted real probe |
| Spark expert kernel/layout | daemon `expertd`, `native/cuda/kernels`, Python AOT exporters | Spark-native rebuild + real expert smoke/bench |
| EXL3 loading/kernel/quantization | daemon EXL3 loader, `SparkInfer`, `quantization/` | Quantization tests + four-rank K4 native parity + live qualification |
| DFlash2 draft execution/policy | daemon `real_full/dflash*`, Python capture modules, checked-in cost profile | Capture tests + preflight + matched native-MTP/DFlash2 live qualification |
| Profile/capacity/defaults | `serve_profiles.py`, `resolve_serve_profile.py`, `release-common.sh`, config | Python profile tests + `run.sh --dry-run` + live memory gate |
| Image/runtime dependency | `docker/`, `build-release-artifacts.sh` | Clean role build + import gates + five-container smoke |

## Handoff checklist

1. Re-read `git status --short`; do not include caches, reports, logs, or
   unrelated user work.
2. Run formatting, `git diff --check`, focused tests, and the smallest broader
   gate justified by the risk.
3. If CUDA/native code changed, identify the exact rebuilt
   `libglmrt_native.so`; if Spark code changed, rebuild/restage ARM artifacts.
4. If deployment semantics changed, run `./run.sh --dry-run` before any live
   restart.
5. Report commands run, results, skipped hardware gates, remaining uncertainty,
   and files changed.
