from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "_validate_glm53_agentic_release_evidence",
    TOOLS / "validate_glm53_agentic_release_evidence.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def signed_serving(tmp_path: Path) -> Path:
    body = {
        "schema": TOOL.SERVING_SCHEMA,
        "status": "accepted",
        "model_id": TOOL.GLM53_MODEL_ID,
        "failed_gates": [],
        "gates": {"complete": True},
        "runtime": {
            "default_speculation": "dflash2",
            "profile": "balanced",
            "model_revision": "a" * 40,
            "power_limit_w": 400,
            "engine_identity": "engine",
            "sparkinfer_revision": "b" * 40,
            "coordinator_slot_fingerprint": "c" * 64,
            "expert_slot_fingerprint": "d" * 64,
            "expert_runtime_fingerprints": {"dflash2": "e" * 64},
            "speculation_settings": {
                "dflash2": {
                    "checkpoint_model_id": "incoai/GLM-5.3-DFlash2",
                    "checkpoint_revision": "425aa615ce320caac34400208b30808c8f14f76c",
                    "fixed_drafts": 5,
                    "topk_backend": "flashinfer-dsa",
                }
            },
        },
        "thresholds": {"tool_eval_version": "2.3.2"},
        "results": {"default_speculation": "dflash2"},
    }
    report = {
        **body,
        "report_sha256": hashlib.sha256(TOOL.canonical_json(body)).hexdigest(),
    }
    path = tmp_path / "serving.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    tmp_path.mkdir(parents=True, exist_ok=True)
    serving = signed_serving(tmp_path)
    deployment_path = tmp_path / "deployment.json"
    deployment_path.write_text("{}", encoding="utf-8")
    launch = int(
        datetime(2026, 8, 29, tzinfo=timezone.utc).timestamp() * 1_000_000_000
    )
    deployed = {
        "identity": {
            "schema": "glmrt-wip-deployment-evidence-v2",
            "path": str(deployment_path),
            "bytes": deployment_path.stat().st_size,
            "sha256": "f" * 64,
        },
        "model": TOOL.GLM53_MODEL_ID,
        "model_revision": "a" * 40,
        "slot": "release",
        "profile": "balanced",
        "speculation": "dflash2",
        "speculation_settings": {
            "checkpoint_model_id": "incoai/GLM-5.3-DFlash2",
            "checkpoint_revision": "425aa615ce320caac34400208b30808c8f14f76c",
            "fixed_drafts": 5,
            "topk_backend": "flashinfer-dsa",
        },
        "launch_started_ns": launch,
        "power_limit_w": 400,
        "engine_identity": "engine",
        "sparkinfer_revision": "b" * 40,
        "fingerprints": {
            "coordinator_slot": "c" * 64,
            "expert_slot": "d" * 64,
            "expert_runtime": "e" * 64,
            "deployment": "f" * 64,
        },
    }
    monkeypatch.setattr(TOOL, "deployment", lambda *_args, **_kwargs: deployed)

    tool_paths = []
    for index, seed in enumerate(TOOL.TOOL_SEEDS, 1):
        path = tmp_path / f"tool-{seed}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": f"2026-08-30T00-00-0{index}.000000Z_run{index}",
                    "seed": seed,
                }
            ),
            encoding="utf-8",
        )
        tool_paths.append(path)

    def fake_tool_eval(path: Path, **_kwargs) -> dict:
        raw = json.loads(path.read_text())
        seed = raw["seed"]
        return {
            "identity": {
                "schema": "tool-eval-bench-json-v1",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            },
            "config": {
                "model": TOOL.GLM53_MODEL_ID,
                "seed": seed,
                "config_fingerprint": str(seed),
                "concurrency": 1,
                "temperature": 0.0,
            },
            "metadata": {
                "model": TOOL.GLM53_MODEL_ID,
                "seed": seed,
                "parallel": 1,
                "trials": 1,
                "thinking_enabled": True,
            },
            "scenario_ids": [f"TC-{value:02d}" for value in range(1, 70)],
            "maximum_points": 138,
            "total_points": 120 + seed % 3,
            "final_score": 87,
        }

    monkeypatch.setattr(TOOL, "tool_eval", fake_tool_eval)

    pi_paths = {}
    pi_reports = {}
    for mode, second in (("off", 10), ("high", 20)):
        path = tmp_path / f"pi-{mode}.json"
        path.write_text(json.dumps({"mode": mode}), encoding="utf-8")
        pi_paths[mode] = path
        pi_reports[mode] = {
            "schema": TOOL.PI_SCHEMA,
            "status": "accepted",
            "model_id": TOOL.GLM53_MODEL_ID,
            "pi_version": "0.82.0",
            "thinking": mode,
            "session_id": f"session-{mode}",
            "session_timestamp": f"2026-08-30T00:00:{second:02d}Z",
            "wall_seconds": 100.0 + second,
            "turns": 2,
            "tool_calls": 1,
            "tool_errors": 0,
            "usage": {"total": 1000 + second},
            "artifact": {"path": "game.html", "bytes": 10_000 + second},
        }

    monkeypatch.setattr(
        TOOL,
        "revalidate_pi",
        lambda path, **_kwargs: pi_reports["off" if path == pi_paths["off"] else "high"],
    )
    return {
        "serving_path": serving,
        "deployment_path": deployment_path,
        "tool_paths": tool_paths,
        "pi_off_path": pi_paths["off"],
        "pi_high_path": pi_paths["high"],
        "node_binary": "/bin/true",
        "_deployed": deployed,
        "_pi_reports": pi_reports,
    }


def clean(arguments: dict) -> dict:
    return {key: value for key, value in arguments.items() if not key.startswith("_")}


def test_binds_seeded_tool_and_pi_runs_to_the_qualified_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = TOOL.validate(**clean(fixture(tmp_path, monkeypatch)))

    assert report["status"] == "accepted"
    assert report["default_speculation"] == "dflash2"
    assert report["tool_eval"]["seeds"] == list(TOOL.TOOL_SEEDS)
    assert len(report["tool_eval"]["runs"]) == 3
    assert set(report["pi"]) == {"off", "high"}
    assert len(report["report_sha256"]) == 64


def test_rejects_a_different_deployment_or_tool_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = fixture(tmp_path, monkeypatch)
    arguments["_deployed"]["engine_identity"] = "different"
    with pytest.raises(TOOL.AgenticReleaseError, match="differs"):
        TOOL.validate(**clean(arguments))

    arguments = fixture(tmp_path / "second", monkeypatch)
    raw = json.loads(arguments["tool_paths"][0].read_text())
    raw["seed"] = 1
    arguments["tool_paths"][0].write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(TOOL.AgenticReleaseError, match="wrong seed"):
        TOOL.validate(**clean(arguments))


def test_rejects_reused_or_predeployment_pi_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = fixture(tmp_path, monkeypatch)
    arguments["_pi_reports"]["high"]["session_id"] = "session-off"
    with pytest.raises(TOOL.AgenticReleaseError, match="reused"):
        TOOL.validate(**clean(arguments))

    arguments = fixture(tmp_path / "second", monkeypatch)
    arguments["_pi_reports"]["off"]["session_timestamp"] = "2026-08-28T00:00:00Z"
    with pytest.raises(TOOL.AgenticReleaseError, match="differs"):
        TOOL.validate(**clean(arguments))
