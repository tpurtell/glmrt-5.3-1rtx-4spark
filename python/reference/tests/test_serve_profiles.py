from pathlib import Path

import pytest

from glmrt_reference.serve_profiles import resolve_serve_profile


GPU_TOTAL_MIB = 97_887


def resolve(tmp_path: Path, **kwargs):
    inherited_environment = kwargs.pop(
        "inherited_environment",
        {"GLMRT_REAL_FULL_REQUEST_WORKER_CPUS": "1"},
    )
    return resolve_serve_profile(
        repo_root=tmp_path,
        gpu_total_mib=GPU_TOTAL_MIB,
        inherited_environment=inherited_environment,
        **kwargs,
    )


def install_hf_snapshot(
    tmp_path: Path,
    model_id: str,
    revision: str,
    filenames: tuple[str, ...],
) -> tuple[Path, dict[str, str]]:
    hf_home = tmp_path / "huggingface"
    repo_dir = hf_home / "hub" / ("models--" + model_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    refs = repo_dir / "refs"
    refs.mkdir()
    (refs / "main").write_text(revision)
    for name in filenames:
        (snapshot / name).touch()
    return snapshot, {
        "HF_HOME": str(hf_home),
        "GLMRT_REAL_FULL_REQUEST_WORKER_CPUS": "1",
    }


def test_balanced_reproduces_qualified_pool_geometry(tmp_path):
    profile = resolve(tmp_path)
    assert profile.kv_dtype == "fp8"
    assert profile.kv_bytes_per_token == 56_544
    assert profile.kv_pool_tokens == 609_536
    assert profile.max_context_tokens == 400_000
    assert profile.max_output_tokens == 100_000
    assert "GLMRT_REAL_FULL_SERVE_MAX_INPUT_TOKENS" not in profile.environment
    assert profile.environment["GLMRT_REAL_FULL_SERVE_MAX_OUTPUT_TOKENS"] == "100000"
    assert profile.environment["GLMRT_REAL_FULL_ENABLE_THINKING"] == "1"
    assert profile.environment["GLMRT_COORDINATOR_W8A16_Q_A"] == "1"
    assert profile.environment["GLMRT_REAL_FULL_DSPARK"] == "1"
    assert profile.environment["GLMRT_DSPARK_MODEL_ID"] == (
        "RedHatAI/GLM-5.2-speculator.dspark"
    )
    assert profile.environment["GLMRT_DSPARK_REVISION"] == (
        "8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4"
    )


def test_raptor_affinity_is_part_of_the_easy_launch_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "glmrt_reference.serve_profiles.socket.gethostname", lambda: "raptor"
    )
    profile = resolve_serve_profile(
        repo_root=tmp_path,
        gpu_total_mib=GPU_TOTAL_MIB,
        inherited_environment={},
        speculation="plain",
    )
    assert profile.environment["GLMRT_REAL_FULL_SERVE_SHARED_CPU_LIST"] == (
        "0,3-32,35-63"
    )
    assert profile.environment["GLMRT_REAL_FULL_REQUEST_WORKER_CPUS"] == "1"
    assert profile.environment["GLMRT_REAL_FULL_SCHEDULER_WORKER_CPU"] == "2"


def test_long_uses_qualified_native_nvfp4_capacity(tmp_path):
    profile = resolve(tmp_path, profile="long", speculation="plain")
    assert profile.kv_dtype == "nvfp4"
    assert profile.kv_bytes_per_token == 39_072
    assert profile.kv_pool_tokens == 883_712
    assert profile.max_context_tokens == 883_712
    assert "GLMRT_REAL_FULL_SERVE_MAX_INPUT_TOKENS" not in profile.environment
    assert profile.environment["GLMRT_REAL_FULL_SERVE_MAX_OUTPUT_TOKENS"] == "100000"
    assert profile.qualification == "qualified"
    assert not profile.blockers


def test_accuracy_selects_bf16_attention_and_reduction_with_qualified_w8(tmp_path):
    profile = resolve(tmp_path, profile="accuracy", speculation="plain")
    assert profile.kv_dtype == "bf16"
    assert profile.fixed_reserve_gib == 55.492
    assert profile.kv_pool_tokens == 361_920
    assert profile.max_context_tokens == 200_000
    assert "GLMRT_REAL_FULL_SERVE_MAX_INPUT_TOKENS" not in profile.environment
    assert profile.environment["GLMRT_REAL_FULL_SERVE_MAX_OUTPUT_TOKENS"] == "50000"
    assert profile.environment["GLMRT_COORDINATOR_W8A16_Q_A"] == "1"
    assert profile.environment["GLMRT_COORDINATOR_W8A16_Q_B"] == "1"
    assert profile.environment["GLMRT_COORDINATOR_W8A16_O_PROJ"] == "1"
    assert profile.environment["GLMRT_B12X_COORDINATOR_W4A16_Q_B"] == "0"
    assert profile.environment["GLMRT_EXPERT_INTERMEDIATE_REDUCTION_DTYPE"] == "bf16"
    assert profile.qualification == "qualified"


def test_nvidia_mtp_selects_bf16_layer_78_without_blocking_dspark(tmp_path, monkeypatch):
    snapshot = tmp_path / "dspark"
    snapshot.mkdir()
    monkeypatch.setattr(
        "glmrt_reference.serve_profiles.find_hf_snapshot",
        lambda model_id, **kwargs: snapshot,
    )
    mtp = resolve(tmp_path, model="nvidia", speculation="mtp")
    assert not any("layer-78" in blocker for blocker in mtp.blockers)
    assert mtp.environment["GLMRT_COORDINATOR_INCLUDE_MTP_LAYER"] == "1"
    assert mtp.environment["GLMRT_MTP_BF16_EXPERTS"] == "0"
    retained = resolve(
        tmp_path, model="nvidia", profile="accuracy", speculation="mtp"
    )
    assert retained.environment["GLMRT_MTP_BF16_EXPERTS"] == "1"
    startup_quantized = resolve(
        tmp_path, model="nvidia", profile="long", speculation="mtp"
    )
    assert startup_quantized.environment["GLMRT_MTP_BF16_EXPERTS"] == "0"
    dspark = resolve(tmp_path, model="nvidia", speculation="dspark")
    assert not any("layer-78" in blocker for blocker in dspark.blockers)
    assert dspark.environment["GLMRT_COORDINATOR_INCLUDE_MTP_LAYER"] == "0"
    assert dspark.qualification == "qualified"
    assert not any("not passed" in warning for warning in dspark.warnings)

    accuracy = resolve(
        tmp_path, model="nvidia", profile="accuracy", speculation="dspark"
    )
    assert accuracy.qualification == "candidate"
    assert any("not passed" in warning for warning in accuracy.warnings)


def test_exl3_selects_calibrated_candidate_without_changing_profile_geometry(
    tmp_path,
):
    baseline = resolve(tmp_path, model="luke", speculation="plain")
    candidate = resolve(tmp_path, model="exl3", speculation="plain")

    assert candidate.model_id == (
        "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1"
    )
    assert candidate.environment["GLMRT_MODEL_ID"] == candidate.model_id
    assert candidate.fixed_reserve_gib == baseline.fixed_reserve_gib
    assert candidate.kv_pool_tokens == baseline.kv_pool_tokens
    assert candidate.qualification == "candidate"
    assert any("paired live serving gates" in warning for warning in candidate.warnings)


def test_dspark_checkpoint_can_be_selected_for_controlled_evaluation(
    tmp_path, monkeypatch
):
    preview = tmp_path / "preview"
    preview.mkdir()
    requested = []

    def find_snapshot(model_id, **kwargs):
        requested.append((model_id, kwargs.get("revision")))
        return preview

    monkeypatch.setattr(
        "glmrt_reference.serve_profiles.find_hf_snapshot", find_snapshot
    )
    profile = resolve_serve_profile(
        repo_root=tmp_path,
        gpu_total_mib=GPU_TOTAL_MIB,
        inherited_environment={
            "GLMRT_DSPARK_MODEL_ID": "siro1/glm-5.2-dspark-preview",
            "GLMRT_DSPARK_REVISION": "7ff03018b3a443bfb9fca166739bd5f37ee5908b",
        },
    )
    assert requested == [
        (
            "siro1/glm-5.2-dspark-preview",
            "7ff03018b3a443bfb9fca166739bd5f37ee5908b",
        )
    ]
    assert profile.environment["GLMRT_DSPARK_MODEL_ID"] == (
        "siro1/glm-5.2-dspark-preview"
    )
    assert profile.environment["GLMRT_REAL_FULL_DSPARK_SNAPSHOT"] == str(preview)


def test_glm53_k4_selects_the_pinned_qualified_dflash2_default(tmp_path):
    snapshot, inherited = install_hf_snapshot(
        tmp_path,
        "incoai/GLM-5.3-DFlash2",
        "425aa615ce320caac34400208b30808c8f14f76c",
        ("config.json", "model.safetensors"),
    )
    profile = resolve(
        tmp_path,
        model="glm53-exl3",
        speculation="dflash2",
        inherited_environment=inherited,
    )
    assert profile.model_id == "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
    assert profile.qualification == "qualified"
    assert profile.environment["GLMRT_REAL_FULL_DFLASH2"] == "1"
    assert profile.environment["GLMRT_REAL_FULL_DSPARK"] == "0"
    assert profile.environment["GLMRT_REAL_FULL_MTP"] == "0"
    assert profile.environment["GLMRT_SPARK_INCLUDE_MTP_LAYER"] == "0"
    assert profile.environment["GLMRT_COORDINATOR_INCLUDE_MTP_LAYER"] == "0"
    assert profile.environment["GLMRT_REAL_FULL_DFLASH2_SNAPSHOT"] == str(snapshot)
    assert profile.environment["GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS"] == "adaptive"
    assert profile.environment["GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND"] == "torch"
    assert not profile.blockers

    tuned = resolve(
        tmp_path,
        model="glm53-exl3",
        speculation="dflash2",
        dflash2_fixed_drafts=3,
        dflash2_topk_backend="flashinfer-dsa",
        inherited_environment=inherited,
    )
    assert tuned.environment["GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS"] == "3"
    assert tuned.environment["GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND"] == "flashinfer-dsa"


def test_glm53_native_mtp_alone_loads_the_retained_layer_78(tmp_path) -> None:
    profile = resolve(tmp_path, model="glm53-exl3", speculation="mtp")

    assert profile.environment["GLMRT_REAL_FULL_MTP"] == "1"
    assert profile.environment["GLMRT_REAL_FULL_DFLASH2"] == "0"
    assert profile.environment["GLMRT_SPARK_INCLUDE_MTP_LAYER"] == "1"
    assert profile.environment["GLMRT_COORDINATOR_INCLUDE_MTP_LAYER"] == "1"
    assert profile.environment["GLMRT_MTP_BF16_EXPERTS"] == "0"


def test_dflash2_fixed_width_is_scoped_and_bounded(tmp_path):
    with pytest.raises(ValueError, match="requires speculation=dflash2"):
        resolve(tmp_path, speculation="plain", dflash2_fixed_drafts=3)
    with pytest.raises(ValueError, match="must be in 1..7"):
        resolve(tmp_path, speculation="dflash2", dflash2_fixed_drafts=8)
    with pytest.raises(ValueError, match="must be in 1..7"):
        resolve(tmp_path, speculation="dflash2", dflash2_fixed_drafts=0)
    with pytest.raises(ValueError, match="must be torch, flashinfer"):
        resolve(tmp_path, speculation="dflash2", dflash2_topk_backend="other")


def test_dflash2_rejects_a_glm52_target(tmp_path, monkeypatch):
    snapshot = tmp_path / "dflash2"
    snapshot.mkdir()
    monkeypatch.setattr(
        "glmrt_reference.serve_profiles.find_hf_snapshot",
        lambda model_id, **kwargs: snapshot,
    )
    profile = resolve(tmp_path, model="exl3", speculation="dflash2")
    assert any("GLM-5.3 EXL3 K4" in blocker for blocker in profile.blockers)


def test_glm52_dspark_rejects_a_glm53_target(tmp_path, monkeypatch):
    snapshot = tmp_path / "dspark"
    snapshot.mkdir()
    monkeypatch.setattr(
        "glmrt_reference.serve_profiles.find_hf_snapshot",
        lambda model_id, **kwargs: snapshot,
    )
    profile = resolve(tmp_path, model="glm53-exl3", speculation="dspark")
    assert any("cannot draft for the GLM-5.3 target" in blocker for blocker in profile.blockers)


def test_native_mtp_alone_reserves_layer_78_kv(tmp_path):
    plain = resolve(tmp_path, profile="long", speculation="plain")
    mtp = resolve(tmp_path, profile="long", speculation="mtp")
    assert plain.kv_bytes_per_token == 39_072
    assert mtp.kv_bytes_per_token == 39_760


def test_vision_never_claims_projector_only_is_runnable(tmp_path):
    _, inherited = install_hf_snapshot(
        tmp_path,
        "baseten/GLM-5.2-Vision-NVFP4",
        "f6eab6117386a0c69152fdf272dc65bfd0254f9f",
        ("mm_projector.safetensors",),
    )
    profile = resolve(
        tmp_path,
        vision=True,
        speculation="plain",
        inherited_environment=inherited,
    )
    assert profile.vision_reserve_gib == 2
    assert any("configuration/preprocessing assets" in blocker for blocker in profile.blockers)
    assert any("MoonViT encoder" in blocker for blocker in profile.blockers)


def test_vision_control_assets_are_checked_as_a_set(tmp_path):
    asset_dir, inherited = install_hf_snapshot(
        tmp_path,
        "baseten/GLM-5.2-Vision-NVFP4",
        "f6eab6117386a0c69152fdf272dc65bfd0254f9f",
        (
            "config.json",
            "configuration_glm5v.py",
            "kimi_k25_processor.py",
            "kimi_k25_vision_processing.py",
            "media_utils.py",
            "preprocessor_config.json",
            "mm_projector.safetensors",
            "vision_tower.safetensors",
        ),
    )
    profile = resolve(
        tmp_path,
        vision=True,
        speculation="plain",
        inherited_environment=inherited,
    )
    assert not any(
        "configuration/preprocessing assets" in blocker for blocker in profile.blockers
    )
    assert not any("projector is not installed" in blocker for blocker in profile.blockers)
    assert not any("MoonViT encoder" in blocker for blocker in profile.blockers)
    assert not profile.blockers
    assert profile.environment["GLMRT_VISION_ENABLED"] == "1"
    assert profile.environment["GLMRT_VISION_MODEL"] == str(asset_dir)


def test_vision_requires_the_pinned_hugging_face_snapshot(tmp_path):
    profile = resolve(
        tmp_path,
        vision=True,
        speculation="plain",
        inherited_environment={"HF_HOME": str(tmp_path / "empty-hf-cache")},
    )
    assert profile.blockers == (
        "vision snapshot is not installed: "
        "baseten/GLM-5.2-Vision-NVFP4@"
        "f6eab6117386a0c69152fdf272dc65bfd0254f9f; "
        "run scripts/fetch-vision-assets.sh or install it in the "
        "standard Hugging Face cache",
    )
    assert "GLMRT_VISION_ASSET_DIR" not in profile.environment


def test_manual_pool_must_preserve_headroom_and_page_alignment(tmp_path):
    with pytest.raises(ValueError, match="multiple of 64"):
        resolve(tmp_path, kv_pool_tokens=1)
    with pytest.raises(ValueError, match="headroom-safe"):
        resolve(tmp_path, kv_pool_tokens=609_600)


def test_context_cannot_exceed_pool(tmp_path):
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve(
            tmp_path,
            speculation="plain",
            kv_pool_tokens=100_032,
            max_context_tokens=100_033,
        )


def test_context_and_output_are_independent_flexible_limits(tmp_path):
    profile = resolve(
        tmp_path,
        speculation="plain",
        max_context_tokens=400_000,
        max_output_tokens=80_000,
        concurrency=2,
    )
    assert profile.max_context_tokens == 400_000
    assert profile.max_output_tokens == 80_000
    assert profile.concurrency == 2
    assert profile.environment["GLMRT_REAL_FULL_MAX_EXECUTION_LANES"] == "2"
    assert profile.environment["GLMRT_PROTOCOL_V2_VERBS_HOST_EXECUTION_LANES"] == "2"


def test_output_must_fit_context_and_profile_cap(tmp_path):
    with pytest.raises(ValueError, match="smaller than max_context"):
        resolve(
            tmp_path,
            speculation="plain",
            max_context_tokens=100_000,
            max_output_tokens=100_000,
        )
    with pytest.raises(ValueError, match="profile cap"):
        resolve(tmp_path, speculation="plain", max_output_tokens=100_001)
