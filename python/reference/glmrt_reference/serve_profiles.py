"""Resolved production launch profiles for the real GLM-5 serving stack.

The profile layer deliberately contains no CUDA or Hugging Face imports.  It
is used by the launch CLI, unit tests, and future container entrypoints, so
the memory calculation and the environment contract have one implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import subprocess
from typing import Mapping


MIB = 1 << 20
GIB = 1 << 30
PAGE_TOKENS = 64

KV_BASE_BYTES_PER_TOKEN = {
    "bf16": 95_232,
    "fp8": 56_544,
    "nvfp4": 39_072,
}

# Native MTP adds layer 78 and its DSA index. Plain and dSpark do not reserve
# this target-KV plane or preload the checkpoint MTP envelope.
KV_MTP_EXTRA_BYTES_PER_TOKEN = {
    "bf16": 1_408,
    "fp8": 912,
    "nvfp4": 688,
}

MODEL_ALIASES = {
    "luke": "lukealonso/GLM-5.2-NVFP4",
    "nvidia": "nvidia/GLM-5.2-NVFP4",
    "exl3": "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1",
    "glm53-exl3": "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
}

DEFAULT_DSPARK_MODEL_ID = "RedHatAI/GLM-5.2-speculator.dspark"
DEFAULT_DSPARK_REVISION = "8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4"
DEFAULT_DFLASH2_MODEL_ID = "incoai/GLM-5.3-DFlash2"
DEFAULT_DFLASH2_REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
DEFAULT_DFLASH2_DRAFT_POLICY = "adaptive"
DEFAULT_VISION_MODEL_ID = "baseten/GLM-5.2-Vision-NVFP4"
DEFAULT_VISION_REVISION = "f6eab6117386a0c69152fdf272dc65bfd0254f9f"


@dataclass(frozen=True)
class ProfileDefinition:
    name: str
    kv_dtype: str
    fixed_reserve_gib: float
    context_cap: int
    output_cap: int
    qualification: str
    note: str


PROFILES = {
    "balanced": ProfileDefinition(
        name="balanced",
        kv_dtype="fp8",
        fixed_reserve_gib=55.492,
        context_cap=400_000,
        output_cap=100_000,
        qualification="qualified",
        note="Current W8 projection, FP8 target-KV, BF16 control/owner path.",
    ),
    "long": ProfileDefinition(
        name="long",
        kv_dtype="nvfp4",
        fixed_reserve_gib=55.434,
        context_cap=1_048_576,
        output_cap=100_000,
        qualification="qualified",
        note=(
            "Native 432-byte NVFP4 target-KV consumed directly by sparse MLA; "
            "qualified at 90%+ balanced wall throughput and through 425K context."
        ),
    ),
    "accuracy": ProfileDefinition(
        name="accuracy",
        kv_dtype="bf16",
        fixed_reserve_gib=55.492,
        context_cap=200_000,
        output_cap=50_000,
        qualification="qualified",
        note=(
            "BF16 target-KV attention plus BF16 expert exchange and distributed "
            "reduction; qualified W8A16 Q-A/Q-B/O residency is retained. "
            "Spark-local routed accumulation remains FP32. Qualified at 80.53% "
            "of balanced weighted throughput and through 128K context."
        ),
    ),
}


@dataclass(frozen=True)
class ResolvedServeProfile:
    profile: str
    speculation: str
    vision: bool
    model_id: str
    gpu_total_mib: int
    headroom_gib: float
    fixed_reserve_gib: float
    vision_reserve_gib: int
    kv_dtype: str
    kv_bytes_per_token: int
    kv_pool_tokens: int
    kv_pool_bytes: int
    max_context_tokens: int
    max_output_tokens: int
    concurrency: int
    qualification: str
    note: str
    environment: dict[str, str]
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def query_gpu_total_mib() -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    first = result.stdout.strip().splitlines()[0].strip()
    return int(first)


def normalize_model_id(model: str) -> str:
    try:
        return MODEL_ALIASES[model.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown text checkpoint {model!r}; choose luke, nvidia, exl3, or glm53-exl3"
        ) from error


def find_hf_snapshot(
    model_id: str,
    cache_root: Path | None = None,
    revision: str | None = None,
) -> Path | None:
    root = cache_root or Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = root / ("models--" + model_id.replace("/", "--"))
    if revision:
        candidate = repo_dir / "snapshots" / revision
        return candidate if candidate.is_dir() else None
    refs_main = repo_dir / "refs" / "main"
    if refs_main.is_file():
        revision = refs_main.read_text().strip()
        candidate = repo_dir / "snapshots" / revision
        if candidate.is_dir():
            return candidate
    snapshots = repo_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    candidates = sorted(
        (path for path in snapshots.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    return candidates[0] if candidates else None


def hf_hub_cache_root(environment: Mapping[str, str]) -> Path:
    hub_cache = environment.get("HF_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser()
    hf_home = environment.get("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _round_pool_tokens(available_bytes: int, bytes_per_token: int) -> int:
    return (available_bytes // bytes_per_token // PAGE_TOKENS) * PAGE_TOKENS


def resolve_serve_profile(
    *,
    repo_root: Path,
    profile: str = "balanced",
    speculation: str = "dspark",
    vision: bool = False,
    model: str = "luke",
    headroom_gib: float = 8.0,
    gpu_total_mib: int | None = None,
    max_context_tokens: int | None = None,
    max_output_tokens: int | None = None,
    kv_pool_tokens: int | None = None,
    concurrency: int = 4,
    dflash2_fixed_drafts: int | None = None,
    dflash2_topk_backend: str = "torch",
    inherited_environment: Mapping[str, str] | None = None,
) -> ResolvedServeProfile:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}")
    if speculation not in {"plain", "mtp", "dspark", "dflash2"}:
        raise ValueError(f"unknown speculation mode {speculation!r}")
    if headroom_gib < 0:
        raise ValueError("headroom_gib must be non-negative")
    if concurrency < 1 or concurrency > 8:
        raise ValueError("concurrency must be in 1..8")
    if dflash2_fixed_drafts is not None and not 1 <= dflash2_fixed_drafts <= 7:
        raise ValueError(
            "dflash2_fixed_drafts must be in 1..7; use speculation=plain for target-only"
        )
    if dflash2_fixed_drafts is not None and speculation != "dflash2":
        raise ValueError("dflash2_fixed_drafts requires speculation=dflash2")
    if dflash2_topk_backend not in {"torch", "flashinfer", "flashinfer-dsa"}:
        raise ValueError(
            "dflash2_topk_backend must be torch, flashinfer, or flashinfer-dsa"
        )

    definition = PROFILES[profile]
    total_mib = gpu_total_mib if gpu_total_mib is not None else query_gpu_total_mib()
    if total_mib <= 0:
        raise ValueError("gpu_total_mib must be positive")

    vision_reserve_gib = 2 if vision else 0
    reserved_bytes = int(
        (definition.fixed_reserve_gib + vision_reserve_gib + headroom_gib) * GIB
    )
    total_bytes = total_mib * MIB
    if reserved_bytes >= total_bytes:
        raise ValueError(
            f"profile reserves {reserved_bytes / GIB:.2f} GiB on a "
            f"{total_bytes / GIB:.2f} GiB GPU"
        )

    bytes_per_token = KV_BASE_BYTES_PER_TOKEN[definition.kv_dtype]
    if speculation == "mtp":
        bytes_per_token += KV_MTP_EXTRA_BYTES_PER_TOKEN[definition.kv_dtype]
    calculated_pool = _round_pool_tokens(total_bytes - reserved_bytes, bytes_per_token)
    pool_tokens = calculated_pool if kv_pool_tokens is None else kv_pool_tokens
    if pool_tokens <= 0 or pool_tokens % PAGE_TOKENS:
        raise ValueError(f"kv_pool_tokens must be a positive multiple of {PAGE_TOKENS}")
    if pool_tokens > calculated_pool:
        raise ValueError(
            f"kv_pool_tokens={pool_tokens} exceeds the headroom-safe "
            f"calculated capacity {calculated_pool}"
        )

    logical_tokens = (
        min(pool_tokens, definition.context_cap)
        if max_context_tokens is None
        else max_context_tokens
    )
    if logical_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    if logical_tokens > pool_tokens:
        raise ValueError("max_context_tokens cannot exceed kv_pool_tokens")
    output_tokens = (
        min(definition.output_cap, logical_tokens - 1)
        if max_output_tokens is None
        else max_output_tokens
    )
    if output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    if output_tokens > definition.output_cap:
        raise ValueError(
            f"max_output_tokens={output_tokens} exceeds the {profile} profile "
            f"cap {definition.output_cap}"
        )
    if output_tokens >= logical_tokens:
        raise ValueError("max_output_tokens must be smaller than max_context_tokens")

    model_id = normalize_model_id(model)
    qualification = definition.qualification
    nvidia_live_qualified = (
        model_id == MODEL_ALIASES["nvidia"]
        and profile == "balanced"
        and speculation in {"plain", "dspark"}
    )
    if (
        model_id == MODEL_ALIASES["nvidia"]
        and qualification == "qualified"
        and not nvidia_live_qualified
    ):
        qualification = "candidate"
    exl3_candidate = model_id == MODEL_ALIASES["exl3"]
    if exl3_candidate:
        # The checkpoint is intentionally launchable so its paired serving
        # gates can be run, but it cannot inherit the NVFP4 profile's live
        # qualification merely because its coordinator/KV geometry matches.
        qualification = "candidate"
    glm53_exl3 = model_id == MODEL_ALIASES["glm53-exl3"]
    inherited = inherited_environment or {}
    hf_cache_root = hf_hub_cache_root(inherited)
    environment = {
        "GLMRT_SERVE_PROFILE": profile,
        "GLMRT_SPECULATION_MODE": speculation,
        "GLMRT_VISION": "1" if vision else "0",
        "GLMRT_VISION_ENABLED": "1" if vision else "0",
        "GLMRT_MODEL_ID": model_id,
        "GLMRT_REAL_FULL_ENABLE_THINKING": "1",
        "GLMRT_REAL_FULL_SERVE_TRANSPORT": "verbs-host",
        "GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES": str(concurrency),
        "GLMRT_REAL_FULL_MAX_EXECUTION_LANES": str(concurrency),
        "GLMRT_EXPERT_INTERMEDIATE_SHARDS": "4",
        "GLMRT_EXPERT_INTERMEDIATE_REDUCTION": "spark-rdma",
        "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_MIN_ROWS": "16",
        "GLMRT_EXPERT_INTERMEDIATE_OWNER_MAX_ROWS": "8",
        "GLMRT_REAL_FULL_SERVE_KV_CACHE_DTYPE": definition.kv_dtype,
        "GLMRT_REAL_FULL_KV_POOL_TOKENS": str(pool_tokens),
        "GLMRT_REAL_FULL_SERVE_MAX_CONTEXT_TOKENS": str(logical_tokens),
        "GLMRT_REAL_FULL_SERVE_MAX_OUTPUT_TOKENS": str(output_tokens),
        "GLMRT_REAL_FULL_MOE_RESPONSE_DTYPE": "bf16",
        "GLMRT_REAL_FULL_MOE_OWNER_RESPONSE_DTYPE": "bf16",
        "GLMRT_COORDINATOR_W8A16_PACKED_O": "1",
        "GLMRT_COORDINATOR_W8A16_ASYNC_ATTENTION": "1",
        "GLMRT_MTP_BF16_EXPERTS": "0",
    }

    if profile == "accuracy":
        environment.update(
            {
                "GLMRT_COORDINATOR_W8A16_Q_A": "1",
                "GLMRT_COORDINATOR_W8A16_Q_B": "1",
                "GLMRT_COORDINATOR_W8A16_O_PROJ": "1",
                "GLMRT_B12X_COORDINATOR_W4A16_Q_B": "0",
                "GLMRT_B12X_COORDINATOR_W4A16_O_PROJ": "0",
                "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE": "bf16",
                "GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE": "bf16",
            }
        )
    else:
        environment.update(
            {
                "GLMRT_COORDINATOR_W8A16_Q_A": "1",
                "GLMRT_COORDINATOR_W8A16_Q_B": "1",
                "GLMRT_COORDINATOR_W8A16_O_PROJ": "1",
                "GLMRT_B12X_COORDINATOR_W4A16_Q_B": "0",
                "GLMRT_B12X_COORDINATOR_W4A16_O_PROJ": "0",
                "GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE": "fp8",
                "GLMRT_EXPERT_INTERMEDIATE_OWNER_REDUCTION_DTYPE": "bf16",
            }
        )

    blockers: list[str] = []
    warnings: list[str] = []
    if model_id == MODEL_ALIASES["nvidia"] and not nvidia_live_qualified:
        warnings.append(
            "this NVIDIA profile/speculation combination has not passed its live gates"
        )
    if exl3_candidate:
        warnings.append(
            "the calibrated EXL3 checkpoint has not passed its paired live serving gates"
        )
    if speculation == "plain":
        environment.update(
            {
                "GLMRT_REAL_FULL_MTP": "0",
                "GLMRT_REAL_FULL_DSPARK": "0",
                "GLMRT_REAL_FULL_DFLASH2": "0",
                "GLMRT_SPARK_INCLUDE_MTP_LAYER": "0",
                "GLMRT_COORDINATOR_INCLUDE_MTP_LAYER": "0",
            }
        )
    elif speculation == "mtp":
        environment.update(
            {
                "GLMRT_REAL_FULL_MTP": "1",
                "GLMRT_REAL_FULL_DSPARK": "0",
                "GLMRT_REAL_FULL_DFLASH2": "0",
                "GLMRT_SPARK_INCLUDE_MTP_LAYER": "1",
                "GLMRT_COORDINATOR_INCLUDE_MTP_LAYER": "1",
            }
        )
        if model_id == MODEL_ALIASES["nvidia"]:
            # Accuracy retains the checkpoint's BF16 MTP experts. Balanced
            # uses startup NVFP4 quantization: the retained checkpoint improved
            # long-context draft acceptance but lost end-to-end throughput and
            # costs roughly 3.5 GiB on every Spark.
            environment["GLMRT_MTP_BF16_EXPERTS"] = "1" if profile == "accuracy" else "0"
    elif speculation == "dspark":
        environment.update(
            {
                "GLMRT_REAL_FULL_MTP": "0",
                "GLMRT_REAL_FULL_DSPARK": "1",
                "GLMRT_REAL_FULL_DFLASH2": "0",
                "GLMRT_REAL_FULL_DSPARK_CACHE_MODE": "prompt-swa",
                "GLMRT_SPARK_INCLUDE_MTP_LAYER": "0",
                "GLMRT_COORDINATOR_INCLUDE_MTP_LAYER": "0",
            }
        )
        if glm53_exl3:
            blockers.append(
                "the GLM-5.2 dSpark checkpoint cannot draft for the GLM-5.3 target"
            )
        dspark_model_id = (
            inherited.get("GLMRT_DSPARK_MODEL_ID", DEFAULT_DSPARK_MODEL_ID).strip()
            or DEFAULT_DSPARK_MODEL_ID
        )
        environment["GLMRT_DSPARK_MODEL_ID"] = dspark_model_id
        requested_revision = inherited.get("GLMRT_DSPARK_REVISION", "").strip()
        dspark_revision = (
            requested_revision
            or (
                DEFAULT_DSPARK_REVISION
                if dspark_model_id == DEFAULT_DSPARK_MODEL_ID
                else None
            )
        )
        if dspark_revision is not None:
            environment["GLMRT_DSPARK_REVISION"] = dspark_revision
        dspark_snapshot = find_hf_snapshot(
            dspark_model_id,
            cache_root=hf_cache_root,
            revision=dspark_revision,
        )
        if dspark_snapshot is None:
            blockers.append(f"dSpark snapshot is not installed: {dspark_model_id}")
        else:
            environment["GLMRT_REAL_FULL_DSPARK_SNAPSHOT"] = str(dspark_snapshot)
    else:
        environment.update(
            {
                "GLMRT_REAL_FULL_MTP": "0",
                "GLMRT_REAL_FULL_DSPARK": "0",
                "GLMRT_REAL_FULL_DFLASH2": "1",
                "GLMRT_DFLASH2_MODEL_ID": DEFAULT_DFLASH2_MODEL_ID,
                "GLMRT_DFLASH2_REVISION": DEFAULT_DFLASH2_REVISION,
                "GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS": str(
                    DEFAULT_DFLASH2_DRAFT_POLICY
                    if dflash2_fixed_drafts is None
                    else dflash2_fixed_drafts
                ),
                "GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND": dflash2_topk_backend,
                "GLMRT_SPARK_INCLUDE_MTP_LAYER": "0",
                "GLMRT_COORDINATOR_INCLUDE_MTP_LAYER": "0",
            }
        )
        if model_id != MODEL_ALIASES["glm53-exl3"]:
            blockers.append(
                "DFlash2 requires the GLM-5.3 EXL3 K4 target checkpoint"
            )
        dflash2_snapshot = find_hf_snapshot(
            DEFAULT_DFLASH2_MODEL_ID,
            cache_root=hf_cache_root,
            revision=DEFAULT_DFLASH2_REVISION,
        )
        if dflash2_snapshot is None:
            blockers.append(
                f"DFlash2 snapshot is not installed: {DEFAULT_DFLASH2_MODEL_ID}@{DEFAULT_DFLASH2_REVISION}"
            )
        else:
            environment["GLMRT_REAL_FULL_DFLASH2_SNAPSHOT"] = str(dflash2_snapshot)

    if vision:
        vision_snapshot = find_hf_snapshot(
            DEFAULT_VISION_MODEL_ID,
            cache_root=hf_cache_root,
            revision=DEFAULT_VISION_REVISION,
        )
        environment["GLMRT_VISION_MODEL_ID"] = DEFAULT_VISION_MODEL_ID
        environment["GLMRT_VISION_REVISION"] = DEFAULT_VISION_REVISION
        if vision_snapshot is None:
            blockers.append(
                "vision snapshot is not installed: "
                f"{DEFAULT_VISION_MODEL_ID}@{DEFAULT_VISION_REVISION}; "
                "run scripts/fetch-vision-assets.sh or install it in the "
                "standard Hugging Face cache"
            )
        else:
            environment["GLMRT_VISION_ASSET_DIR"] = str(vision_snapshot)
            environment["GLMRT_VISION_MODEL"] = str(vision_snapshot)
            required_control_files = (
                "config.json",
                "configuration_glm5v.py",
                "kimi_k25_processor.py",
                "kimi_k25_vision_processing.py",
                "media_utils.py",
                "preprocessor_config.json",
            )
            projector = vision_snapshot / "mm_projector.safetensors"
            tower = vision_snapshot / "vision_tower.safetensors"
            missing_control_files = [
                name
                for name in required_control_files
                if not (vision_snapshot / name).is_file()
            ]
            if missing_control_files:
                blockers.append(
                    "Baseten vision configuration/preprocessing assets are incomplete: "
                    + ", ".join(missing_control_files)
                )
            if not projector.is_file():
                blockers.append("Baseten vision projector is not installed")
            if not tower.is_file():
                blockers.append(
                    "MoonViT encoder is not installed; the 99 MB projector alone "
                    "cannot encode image pixels"
                )

    affinity_defaults = {
        "GLMRT_REAL_FULL_SERVE_SHARED_CPU_LIST": "0,3-32,35-63",
        "GLMRT_REAL_FULL_REQUEST_WORKER_CPUS": "1",
        "GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU": "2",
    }
    missing_affinity = [
        name for name in affinity_defaults if not inherited.get(name, "").strip()
    ]
    if socket.gethostname().split(".", 1)[0] == "raptor":
        environment.update(
            {name: affinity_defaults[name] for name in missing_affinity}
        )
    elif missing_affinity:
        warnings.append(
            "request/scheduler CPU affinity is host-specific and must be supplied "
            "for this coordinator"
        )

    return ResolvedServeProfile(
        profile=profile,
        speculation=speculation,
        vision=vision,
        model_id=model_id,
        gpu_total_mib=total_mib,
        headroom_gib=headroom_gib,
        fixed_reserve_gib=definition.fixed_reserve_gib,
        vision_reserve_gib=vision_reserve_gib,
        kv_dtype=definition.kv_dtype,
        kv_bytes_per_token=bytes_per_token,
        kv_pool_tokens=pool_tokens,
        kv_pool_bytes=pool_tokens * bytes_per_token,
        max_context_tokens=logical_tokens,
        max_output_tokens=output_tokens,
        concurrency=concurrency,
        qualification=qualification,
        note=definition.note,
        environment=environment,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )
