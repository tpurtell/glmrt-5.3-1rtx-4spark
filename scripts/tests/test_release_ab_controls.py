from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_COMMON = ROOT / "scripts" / "release-common.sh"

BASE_CONFIG = """\
PROFILE=balanced
MODEL=luke
SPECULATION=dspark
SPARK_0_HOST=ostrich
SPARK_1_HOST=dodo
SPARK_2_HOST=emu
SPARK_3_HOST=kiwi
SPARK_0_LANE_A=10.55.0.1
SPARK_1_LANE_A=10.55.0.2
SPARK_2_LANE_A=10.55.0.3
SPARK_3_LANE_A=10.55.0.4
"""


def load_config(tmp_path: Path, extra: str = "") -> subprocess.CompletedProcess[str]:
    config = tmp_path / "glmrt.config"
    config.write_text(BASE_CONFIG + extra, encoding="utf-8")
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'source "$1"; release_load_config "$2"; '
                'printf "%s\\n%s\\n" '
                '"$SPARKINFER_GLM_H64_QUERY_PROJECTION" '
                '"${DSPARK_FIXED_DRAFTS-unset}"'
            ),
            "bash",
            str(RELEASE_COMMON),
            str(config),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def load_model_id(tmp_path: Path, model: str) -> subprocess.CompletedProcess[str]:
    config = tmp_path / f"glmrt-{model}.config"
    config.write_text(
        BASE_CONFIG.replace("MODEL=luke", f"MODEL={model}"), encoding="utf-8"
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; release_load_config "$2"; printf "%s\\n" "$RELEASE_MODEL_ID"',
            "bash",
            str(RELEASE_COMMON),
            str(config),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_release_ab_controls_have_safe_defaults(tmp_path: Path) -> None:
    result = load_config(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["auto", ""]


def test_release_model_selector_accepts_calibrated_exl3(tmp_path: Path) -> None:
    result = load_model_id(tmp_path, "exl3")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1"


def test_profile_resolver_accepts_the_release_exl3_selector() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "python" / "reference")
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "python" / "tools" / "resolve_serve_profile.py"),
            "--repo-root",
            str(ROOT),
            "--model",
            "exl3",
            "--speculation",
            "plain",
            "--gpu-total-mib",
            "97887",
            "--dry-run",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 0, result.stderr
    resolved = json.loads(result.stdout)
    assert resolved["model_id"] == (
        "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1"
    )
    assert resolved["environment"]["GLMRT_MODEL_ID"] == resolved["model_id"]


def test_standard_launcher_binds_and_compares_exact_model_revision() -> None:
    launcher = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert 'remote_model_revision="$(check_model_cache_remote' in launcher
    assert 'remote_model_revision" == "$coordinator_model_revision' in launcher
    assert '"$coordinator_model_revision" \\' in launcher


def test_release_ab_controls_accept_explicit_qualification_values(
    tmp_path: Path,
) -> None:
    result = load_config(
        tmp_path,
        "SPARKINFER_GLM_H64_QUERY_PROJECTION=disable\nDSPARK_FIXED_DRAFTS=7\n",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["disable", "7"]


def test_fixed_dspark_depth_rejects_non_dspark_profile(tmp_path: Path) -> None:
    result = load_config(
        tmp_path,
        "SPECULATION=plain\nDSPARK_FIXED_DRAFTS=2\n",
    )

    assert result.returncode == 2
    assert "DSPARK_FIXED_DRAFTS requires SPECULATION=dspark" in result.stderr


def test_fixed_dflash2_depth_is_bounded_and_scoped(tmp_path: Path) -> None:
    accepted = load_config(
        tmp_path,
        "SPECULATION=dflash2\nMODEL=glm53-exl3\nDFLASH2_FIXED_DRAFTS=4\n",
    )
    assert accepted.returncode == 0, accepted.stderr

    wrong_mode = load_config(tmp_path, "DFLASH2_FIXED_DRAFTS=4\n")
    assert wrong_mode.returncode == 2
    assert "DFLASH2_FIXED_DRAFTS requires SPECULATION=dflash2" in wrong_mode.stderr

    invalid = load_config(
        tmp_path,
        "SPECULATION=dflash2\nMODEL=glm53-exl3\nDFLASH2_FIXED_DRAFTS=8\n",
    )
    assert invalid.returncode == 2
    assert "DFLASH2_FIXED_DRAFTS must be empty or in 1..7" in invalid.stderr

    target_only = load_config(
        tmp_path,
        "SPECULATION=dflash2\nMODEL=glm53-exl3\nDFLASH2_FIXED_DRAFTS=0\n",
    )
    assert target_only.returncode == 2
    assert "use SPECULATION=plain for target-only" in target_only.stderr


def test_dflash2_topk_backend_is_explicitly_bounded(tmp_path: Path) -> None:
    accepted = load_config(
        tmp_path,
        "SPECULATION=dflash2\nMODEL=glm53-exl3\nDFLASH2_TOPK_BACKEND=flashinfer-dsa\n",
    )
    assert accepted.returncode == 0, accepted.stderr

    invalid = load_config(tmp_path, "DFLASH2_TOPK_BACKEND=radix\n")
    assert invalid.returncode == 2
    assert "DFLASH2_TOPK_BACKEND must be torch, flashinfer, or flashinfer-dsa" in invalid.stderr


def test_release_ab_controls_reject_invalid_values(tmp_path: Path) -> None:
    invalid_h64 = load_config(
        tmp_path,
        "SPARKINFER_GLM_H64_QUERY_PROJECTION=enabled\n",
    )
    assert invalid_h64.returncode == 2
    assert (
        "SPARKINFER_GLM_H64_QUERY_PROJECTION must be auto, disable, or force"
        in invalid_h64.stderr
    )

    invalid_depth = load_config(tmp_path, "DSPARK_FIXED_DRAFTS=8\n")
    assert invalid_depth.returncode == 2
    assert "DSPARK_FIXED_DRAFTS must be empty or in 0..7" in invalid_depth.stderr


def test_run_fingerprints_and_explicitly_sets_both_ab_controls() -> None:
    launcher = (ROOT / "run.sh").read_text(encoding="utf-8")
    fingerprint = launcher.split(
        'deployment_fingerprint="$(', maxsplit=1
    )[1].split("check_model_cache_local()", maxsplit=1)[0]
    env_file = launcher.split('env_file="$state_dir/coordinator.env"', maxsplit=1)[
        1
    ].split("mkdir -p", maxsplit=1)[0]

    assert '"$SPARKINFER_GLM_H64_QUERY_PROJECTION"' in fingerprint
    assert '"$DSPARK_FIXED_DRAFTS"' in fingerprint
    assert '"$DFLASH2_FIXED_DRAFTS"' in fingerprint
    assert (
        "GLMRT_SPARKINFER_GLM_H64_BF16_QUERY_PROJECTION="
        "$SPARKINFER_GLM_H64_QUERY_PROJECTION"
    ) in env_file
    assert (
        "GLMRT_REAL_FULL_DSPARK_FIXED_DRAFTS=$DSPARK_FIXED_DRAFTS"
        in env_file
    )
    wip_launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")
    assert '--dflash2-fixed-drafts "$DFLASH2_FIXED_DRAFTS"' in wip_launcher
    assert '--dflash2-topk-backend "$DFLASH2_TOPK_BACKEND"' in wip_launcher
    assert "GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS" in wip_launcher
    assert "GLMRT_SPARKINFER_COMMIT=$coordinator_sparkinfer_commit" in env_file
    assert (
        "GLMRT_COORDINATOR_POWER_LIMIT_WATTS=$coordinator_power_limit_watts"
        in env_file
    )


def test_route_profile_capture_identity_is_launcher_scoped() -> None:
    for relative_path in ("run.sh", "scripts/run-wip.sh"):
        launcher = (ROOT / relative_path).read_text(encoding="utf-8")
        assert (
            "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS="
            "$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_STATS"
        ) in launcher
        assert (
            "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES="
            "$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES"
        ) in launcher
        assert (
            "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID="
            "$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_CAPTURE_ID"
        ) in launcher
        assert (
            "GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE="
            "$GLMRT_PROTOCOL_V2_EXPERT_QUEUE_ROW_ROUTES_GATE_FILE"
        ) in launcher


def test_release_and_wip_containers_never_auto_start_on_boot() -> None:
    release_launcher = (ROOT / "run.sh").read_text(encoding="utf-8")
    wip_builder = (ROOT / "wip.sh").read_text(encoding="utf-8")

    assert "--restart unless-stopped" not in release_launcher
    assert "--restart unless-stopped" not in wip_builder
    assert release_launcher.count("--restart no") == 1
    assert wip_builder.count("--restart no") == 2


def test_parallel_wip_builds_fail_before_finalizing_stale_outputs() -> None:
    wip_builder = (ROOT / "wip.sh").read_text(encoding="utf-8")

    assert wip_builder.count("|| return $?") == 2
    assert (
        "/wip/source coordinator 120 /wip/build/coordinator "
        "/wip/output/coordinator \\\n    || return $?"
    ) in wip_builder
    assert (
        "/wip/source expert 121 /wip/build/expert /wip/output/expert \\\n    || return $?"
    ) in wip_builder


def test_coordinator_only_wip_clone_fans_out_unchanged_expert_slot() -> None:
    wip_builder = (ROOT / "wip.sh").read_text(encoding="utf-8")

    assert 'if [[ "$role" != coordinator || -n "$from_slot" ]]; then' in wip_builder
    assert "distribute_expert_slot" in wip_builder
