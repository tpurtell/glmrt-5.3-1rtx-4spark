from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "analyze_glm53_full_startup.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_full_startup", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def phase_lines(prefix: str, stages: tuple[str, ...], elapsed: float = 1.0) -> list[str]:
    return [
        f"{prefix} stage={stage} elapsed_ms={elapsed:.3f} "
        f"total_ms={(index + 1) * elapsed:.3f}"
        for index, stage in enumerate(stages)
    ]


def logs(tmp_path: Path, *, warm: bool = False) -> tuple[Path, Path]:
    launcher = tmp_path / "launcher.log"
    lifecycle = (
        "== retaining WIP Spark expert processes =="
        if warm
        else "== starting WIP Spark expert processes =="
    )
    launcher.write_text(
        "\n".join(
            [lifecycle, *phase_lines(TOOL.LAUNCHER_PREFIX, TOOL.LAUNCHER_STAGES)]
        )
        + "\n",
        encoding="utf-8",
    )
    real_stages = list(TOOL.REAL_FULL_REQUIRED_ORDER)
    real_stages.insert(real_stages.index("prewarm-main"), "request-worker-spawn")
    coordinator = tmp_path / "coordinator.log"
    coordinator.write_text(
        "\n".join(
            [
                *phase_lines(TOOL.SHELL_PREFIX, TOOL.SHELL_STAGES),
                *phase_lines(TOOL.REAL_FULL_PREFIX, tuple(real_stages)),
                *phase_lines(TOOL.COORDINATOR_PREFIX, TOOL.COORDINATOR_STAGES, 10.0),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return launcher, coordinator


def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        TOOL,
        "deployment",
        lambda *_args, **_kwargs: {
            "identity": {"schema": "deployment", "sha256": "a" * 64},
            "model_revision": "1" * 40,
            "profile": "balanced",
            "speculation_settings": {},
            "power_limit_w": 400,
            "engine_identity": "engine",
            "sparkinfer_revision": "2" * 40,
            "fingerprints": {"expert_runtime": "b" * 64},
        },
    )
    monkeypatch.setattr(
        TOOL,
        "startup",
        lambda *_args, **_kwargs: {
            "identity": {"schema": "startup", "sha256": "c" * 64},
            "expert_runtime_fingerprint": "b" * 64,
            "maximum_service_handoff_total_ms": 2.0,
        },
    )


@pytest.mark.parametrize(("state", "warm"), (("cold", False), ("warm", True)))
def test_aligned_startup_binds_lifecycle_and_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    warm: bool,
) -> None:
    fake_runtime(monkeypatch)
    launcher, coordinator = logs(tmp_path, warm=warm)
    expert = tmp_path / "expert.json"
    expert.write_text(
        json.dumps({"cache_state": "warm", "hosts": [{"host": "ostrich"}]}),
        encoding="utf-8",
    )
    deployment = tmp_path / "deployment.json"
    deployment.write_text("{}", encoding="utf-8")

    report = TOOL.analyze(
        cache_state=state,
        mode="mtp",
        deployment_path=deployment,
        launcher_log_path=launcher,
        coordinator_log_path=coordinator,
        expert_startup_path=expert,
    )
    assert report["status"] == "accepted"
    assert report["launch_state"] == state
    assert report["alignment"]["experts_resident_at_start"] is warm
    assert report["alignment"]["spark_ready_ms"] == (0.0 if warm else 9.0)
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == hashlib.sha256(
        TOOL.canonical_json(body)
    ).hexdigest()


def test_lifecycle_and_phase_corruption_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_runtime(monkeypatch)
    launcher, coordinator = logs(tmp_path, warm=False)
    expert = tmp_path / "expert.json"
    expert.write_text(
        json.dumps({"cache_state": "cold", "hosts": []}), encoding="utf-8"
    )
    deployment = tmp_path / "deployment.json"
    deployment.write_text("{}", encoding="utf-8")

    with pytest.raises(TOOL.FullStartupError, match="lifecycle"):
        TOOL.analyze(
            cache_state="warm",
            mode="mtp",
            deployment_path=deployment,
            launcher_log_path=launcher,
            coordinator_log_path=coordinator,
            expert_startup_path=expert,
        )

    broken = coordinator.read_text().replace(
        "total_ms=2.000", "total_ms=99.000", 1
    )
    coordinator.write_text(broken, encoding="utf-8")
    with pytest.raises(TOOL.FullStartupError, match="timing"):
        TOOL.analyze(
            cache_state="cold",
            mode="mtp",
            deployment_path=deployment,
            launcher_log_path=launcher,
            coordinator_log_path=coordinator,
            expert_startup_path=expert,
        )


def test_real_full_phase_parser_rejects_missing_or_duplicate_stage() -> None:
    stages = list(TOOL.REAL_FULL_REQUIRED_ORDER)
    stages.insert(stages.index("prewarm-main"), "request-worker-inline")
    valid = phase_lines(TOOL.REAL_FULL_PREFIX, tuple(stages))
    assert TOOL.parse_real_full_phases(valid)[-1]["stage"] == "complete"

    duplicate = list(valid)
    duplicate.insert(2, duplicate[1])
    with pytest.raises(TOOL.FullStartupError, match="order"):
        TOOL.parse_real_full_phases(duplicate)
