from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "scripts" / "write-wip-deployment-evidence.py"
SPEC = importlib.util.spec_from_file_location("_wip_deployment_evidence", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def inputs(tmp_path: Path) -> dict:
    profile = tmp_path / "resolved.json"
    profile.write_text(
        json.dumps(
            {
                "model_id": "candidate/model",
                "profile": "balanced",
                "speculation": "dspark",
                "blockers": [],
                "environment": {"GLMRT_MODEL_ID": "candidate/model"},
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "glmrt.config"
    config.write_text("MODEL=exl3\n", encoding="utf-8")
    coordinator = "a" * 64
    expert = "b" * 64
    return {
        "model_id": "candidate/model",
        "model_revision": "c" * 64,
        "slot": "exl3-candidate",
        "profile": "balanced",
        "speculation": "dspark",
        "launch_started_ns": 1_725_000_000_000_000_001,
        "power_limit_w": 400,
        "coordinator_slot_fingerprint": coordinator,
        "expert_slot_fingerprint": expert,
        "expert_runtime_fingerprint": "d" * 64,
        "deployment_fingerprint": "e" * 64,
        "engine_identity": f"wip-exl3-candidate-{coordinator[:12]}-{expert[:12]}",
        "sparkinfer_revision": "f" * 40,
        "resolved_profile_path": profile,
        "config_path": config,
    }


def test_binds_successful_wip_deployment_to_exact_artifacts(tmp_path: Path) -> None:
    report = TOOL.build_evidence(**inputs(tmp_path))

    assert report["status"] == "ready"
    assert report["fingerprints"]["coordinator_slot"] == "a" * 64
    assert report["model_revision"] == "c" * 64
    assert report["launch_started_ns"] == 1_725_000_000_000_000_001
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == TOOL.hashlib.sha256(
        TOOL.canonical_json(body)
    ).hexdigest()


def test_rejects_engine_identity_from_different_slot_artifacts(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    arguments["engine_identity"] = "wip-exl3-candidate-wrong"

    with pytest.raises(TOOL.EvidenceError, match="engine identity"):
        TOOL.build_evidence(**arguments)


def test_rejects_resolved_profile_for_another_model(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    arguments["model_id"] = "different/model"

    with pytest.raises(TOOL.EvidenceError, match="resolved profile"):
        TOOL.build_evidence(**arguments)


@pytest.mark.parametrize("launch_started_ns", [0, -1, True])
def test_rejects_invalid_launcher_start_identity(
    tmp_path: Path, launch_started_ns: int
) -> None:
    arguments = inputs(tmp_path)
    arguments["launch_started_ns"] = launch_started_ns

    with pytest.raises(TOOL.EvidenceError, match="launcher start"):
        TOOL.build_evidence(**arguments)


def test_accepts_dflash2_deployment_evidence(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    arguments["speculation"] = "dflash2"
    profile = arguments["resolved_profile_path"]
    resolved = json.loads(profile.read_text(encoding="utf-8"))
    resolved["speculation"] = "dflash2"
    resolved["environment"].update(
        {
            "GLMRT_DFLASH2_MODEL_ID": TOOL.DFLASH2_MODEL_ID,
            "GLMRT_DFLASH2_REVISION": TOOL.DFLASH2_REVISION,
            "GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS": "4",
            "GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND": "flashinfer-dsa",
        }
    )
    profile.write_text(json.dumps(resolved), encoding="utf-8")

    report = TOOL.build_evidence(**arguments)

    assert report["speculation"] == "dflash2"
    assert report["speculation_settings"] == {
        "checkpoint_model_id": TOOL.DFLASH2_MODEL_ID,
        "checkpoint_revision": TOOL.DFLASH2_REVISION,
        "draft_policy": "fixed",
        "proposal_drafts": 7,
        "fixed_drafts": 4,
        "topk_backend": "flashinfer-dsa",
    }
    assert report["status"] == "ready"


def test_accepts_adaptive_dflash2_deployment_evidence(tmp_path: Path) -> None:
    arguments = inputs(tmp_path)
    arguments["speculation"] = "dflash2"
    profile = arguments["resolved_profile_path"]
    resolved = json.loads(profile.read_text(encoding="utf-8"))
    resolved["speculation"] = "dflash2"
    resolved["environment"].update(
        {
            "GLMRT_DFLASH2_MODEL_ID": TOOL.DFLASH2_MODEL_ID,
            "GLMRT_DFLASH2_REVISION": TOOL.DFLASH2_REVISION,
            "GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS": "adaptive",
            "GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND": "torch",
        }
    )
    profile.write_text(json.dumps(resolved), encoding="utf-8")

    report = TOOL.build_evidence(**arguments)

    assert report["speculation_settings"]["draft_policy"] == "adaptive"
    assert report["speculation_settings"]["proposal_drafts"] == 7
    assert report["speculation_settings"]["fixed_drafts"] is None


@pytest.mark.parametrize("topk_backend", [None, "other"])
def test_rejects_missing_or_unknown_dflash2_topk_backend(
    tmp_path: Path, topk_backend: str | None
) -> None:
    arguments = inputs(tmp_path)
    arguments["speculation"] = "dflash2"
    profile = arguments["resolved_profile_path"]
    resolved = json.loads(profile.read_text(encoding="utf-8"))
    resolved["speculation"] = "dflash2"
    resolved["environment"].update(
        {
            "GLMRT_DFLASH2_MODEL_ID": TOOL.DFLASH2_MODEL_ID,
            "GLMRT_DFLASH2_REVISION": TOOL.DFLASH2_REVISION,
            "GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS": "4",
        }
    )
    if topk_backend is not None:
        resolved["environment"][
            "GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND"
        ] = topk_backend
    profile.write_text(json.dumps(resolved), encoding="utf-8")

    with pytest.raises(TOOL.EvidenceError, match="top-k backend"):
        TOOL.build_evidence(**arguments)
