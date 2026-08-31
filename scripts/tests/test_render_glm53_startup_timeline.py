from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "render_glm53_startup_timeline.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_startup_timeline", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def phases(stages: tuple[str, ...], elapsed: float = 10.0) -> list[dict[str, object]]:
    return [
        {
            "stage": stage,
            "elapsed_ms": elapsed,
            "total_ms": elapsed * (index + 1),
            "line": index + 1,
        }
        for index, stage in enumerate(stages)
    ]


def startup_report(state: str) -> dict[str, object]:
    real_stages = (
        "validation",
        "catalog-kv-config",
        "targets-tokenizer",
        "kv-snapshot-config",
        "prewarm-prompts",
        "coordinator-resident-preload",
        "dspark-preload",
        "sparse-target-connect",
        "expert-warmup",
        "dispatch-worker",
        "executor-assembly",
        "python-capture-barrier",
        "request-worker-spawn",
        "prewarm-paired-lm-head-initial",
        "prewarm-main",
        "prewarm-batched-dspark",
        "prewarm-audit-seal",
        "complete",
    )
    body = {
        "schema": "glmrt-glm53-full-startup-v1",
        "status": "accepted",
        "model_id": "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
        "model_revision": "1" * 40,
        "launch_state": state,
        "expert_compiled_cache_state": "warm",
        "profile": "balanced",
        "speculation": "dflash2",
        "speculation_settings": {"fixed_drafts": 4},
        "power_limit_w": 400,
        "engine_identity": "engine",
        "sparkinfer_revision": "2" * 40,
        "expert_runtime_fingerprint": "3" * 64,
        "alignment": {
            "launcher_wall_ms": 1_000.0 if state == "cold" else 600.0,
            "spark_dispatch_offset_ms": 60.0,
            "coordinator_dispatch_offset_ms": 80.0,
            "coordinator_shell_ms": 80.0,
            "coordinator_daemon_ms": 200.0,
            "spark_ready_ms": 500.0 if state == "cold" else 0.0,
            "experts_resident_at_start": state == "warm",
        },
        "phases": {
            "launcher": phases(
                (
                    "bootstrap",
                    "slot-validation",
                    "profile-resolution",
                    "service-reconciliation",
                    "model-snapshots",
                    "launch-headroom",
                    "spark-dispatch",
                    "coordinator-dispatch",
                    "api-ready",
                ),
                100.0,
            ),
            "coordinator_shell": [],
            "coordinator_daemon": phases(("real-full-serving", "api-bind"), 100.0),
            "real_full": phases(real_stages),
            "spark_hosts": [],
        },
        "evidence": {},
    }
    return body | {
        "report_sha256": hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    }


def serving_report() -> dict[str, object]:
    return {
        "schema": "glmrt-glm5-exl3-serving-qualification-v1",
        "runtime": {
            "model_revision": "1" * 40,
            "power_limit_w": 400,
            "engine_identity": "engine",
            "sparkinfer_revision": "2" * 40,
            "speculation_settings": {"dflash2": {"fixed_drafts": 4}},
            "expert_runtime_fingerprints": {"dflash2": "3" * 64},
        },
        "results": {"default_speculation": "dflash2"},
    }


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def attach_sources(
    tmp_path: Path, prefix: str, report: dict[str, object]
) -> dict[str, object]:
    evidence = {}
    for name in ("deployment", "expert_startup", "launcher_log", "coordinator_log"):
        source = tmp_path / f"{prefix}-{name}"
        source.write_bytes(f"{prefix}-{name}\n".encode())
        evidence[name] = {
            "schema": name,
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    report["evidence"] = evidence
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = hashlib.sha256(
        TOOL.canonical_json(body)
    ).hexdigest()
    return report


def test_renderer_uses_only_glm53_measured_inputs() -> None:
    svg = TOOL.render_svg(startup_report("cold"), startup_report("warm"))
    assert "GLMRT GLM-5.3 STARTUP" in svg
    assert "COLD · FULL EXPERT RELOAD" in svg
    assert "WARM · RETAINED EXPERTS" in svg
    assert "resident at t=0" in svg
    assert "GLM-5.2" not in svg
    assert "dSpark" not in svg


def test_inputs_bind_both_launches_to_selected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving_file = tmp_path / "serving.json"
    cold_file = tmp_path / "cold.json"
    warm_file = tmp_path / "warm.json"
    write(serving_file, serving_report())
    write(cold_file, attach_sources(tmp_path, "cold", startup_report("cold")))
    write(warm_file, attach_sources(tmp_path, "warm", startup_report("warm")))
    monkeypatch.setattr(
        TOOL,
        "signed_serving",
        lambda _path: (serving_file, serving_report()),
    )
    _serving, cold, warm, sources = TOOL.validate_inputs(
        serving_file, cold_file, warm_file
    )
    assert cold["launch_state"] == "cold"
    assert warm["launch_state"] == "warm"
    assert sources["cold"]["sha256"] == hashlib.sha256(
        cold_file.read_bytes()
    ).hexdigest()

    broken = attach_sources(tmp_path, "broken", startup_report("warm"))
    broken["model_revision"] = "f" * 40
    body = {key: value for key, value in broken.items() if key != "report_sha256"}
    broken["report_sha256"] = hashlib.sha256(
        TOOL.canonical_json(body)
    ).hexdigest()
    write(tmp_path / "broken.json", broken)
    with pytest.raises(TOOL.TimelineError, match="selected runtime"):
        TOOL.validate_inputs(serving_file, cold_file, tmp_path / "broken.json")


def test_render_writes_signed_svg_evidence_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = serving_report()
    cold = startup_report("cold")
    warm = startup_report("warm")
    sources = {
        name: {"schema": name, "path": f"/{name}", "bytes": 1, "sha256": name[0] * 64}
        for name in ("serving", "cold", "warm")
    }
    monkeypatch.setattr(
        TOOL,
        "validate_inputs",
        lambda *_args: (serving, cold, warm, sources),
    )
    svg = tmp_path / "startup.svg"
    evidence = tmp_path / "startup.json"
    report = TOOL.render(
        serving_path=tmp_path / "serving",
        cold_path=tmp_path / "cold",
        warm_path=tmp_path / "warm",
        output_path=svg,
        report_path=evidence,
    )
    assert report["svg"]["sha256"] == hashlib.sha256(svg.read_bytes()).hexdigest()
    assert json.loads(evidence.read_text()) == report
    with pytest.raises(TOOL.TimelineError, match="overwrite"):
        TOOL.render(
            serving_path=tmp_path / "serving",
            cold_path=tmp_path / "cold",
            warm_path=tmp_path / "warm",
            output_path=svg,
            report_path=tmp_path / "again.json",
        )
