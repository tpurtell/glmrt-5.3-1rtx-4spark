from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = (
    ROOT / "python" / "tools" / "render_glm53_balanced_micro_timeline.py"
)
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_micro_timeline", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def record() -> dict[str, object]:
    return {
        "case": "code",
        "repeat": 1,
        "prompt_sha256": "a" * 64,
        "completion_tokens": 6,
        "decode_ms": 60.0,
        "decode_tps": 5 / 0.06,
        "target_cycle_physical_m": [1, 3, 2],
        "target_cycle_ms": [10.0, 30.0, 20.0],
        "draft_lengths": [2, 1],
        "accepted_draft_lengths": [1, 1],
        "emitted_tokens_from_verify": 4,
    }


def data() -> dict[str, object]:
    row = record()
    cycles = TOOL.selected_code_cycles(row)
    curve = {
        "1": {
            "samples": 5,
            "total_ms": 50.0,
            "mean_ms": 10.0,
            "median_ms": 10.0,
            "min_ms": 9.0,
            "max_ms": 11.0,
        },
        "2": {
            "samples": 5,
            "total_ms": 100.0,
            "mean_ms": 20.0,
            "median_ms": 20.0,
            "min_ms": 19.0,
            "max_ms": 21.0,
        },
        "3": {
            "samples": 5,
            "total_ms": 150.0,
            "mean_ms": 30.0,
            "median_ms": 30.0,
            "min_ms": 29.0,
            "max_ms": 31.0,
        },
    }
    return {
        "serving": {
            "runtime": {"model_revision": "1" * 40},
            "results": {
                "default_speculation": "dflash2",
                "modes": {
                    "dflash2": {
                        "weighted_decode_tps": 30.0,
                        "accepted_draft_rate": 0.75,
                    }
                },
            },
        },
        "deployment": {
            "model_revision": "1" * 40,
            "speculation": "dflash2",
        },
        "record": row,
        "cycles": cycles,
        "curve": curve,
        "run": {"run_id": "balanced-v1"},
        "evidence": {},
    }


def test_cycle_sequence_reconciles_width_acceptance_tokens_and_time() -> None:
    cycles = TOOL.selected_code_cycles(record())
    assert [cycle["physical_m"] for cycle in cycles] == [1, 3, 2]
    assert [cycle["committed_tokens"] for cycle in cycles] == [1, 2, 2]
    assert sum(cycle["elapsed_ms"] for cycle in cycles) == 60.0

    broken = record()
    broken["completion_tokens"] = 7
    with pytest.raises(TOOL.MicroTimelineError, match="reconcile"):
        TOOL.selected_code_cycles(broken)


def test_cycle_sequence_trims_terminal_stop_tokens_from_nominal_acceptance() -> None:
    stopped = record()
    stopped["completion_tokens"] = 5
    stopped["emitted_tokens_from_verify"] = 4
    cycles = TOOL.selected_code_cycles(stopped)
    assert [cycle["committed_tokens"] for cycle in cycles] == [1, 2, 1]
    assert sum(cycle["committed_tokens"] for cycle in cycles) == 4


def test_svg_is_glm53_production_timing_without_legacy_diagnostic() -> None:
    svg = TOOL.render_svg(data())
    assert "GLMRT GLM-5.3 · PRODUCTION MICRO-TIMELINE" in svg
    assert "MEASURED TARGET-CYCLE CURVE" in svg
    assert "ONE TARGET CYCLE" in svg
    assert "COMPLETE POST-TTFT RESPONSE" not in svg
    assert "TIME-SCALED ZOOM" not in svg
    assert "8-TYPE WEIGHTED" in svg
    assert "M3" in svg
    assert "no synchronized instrumentation" in svg
    assert "GLM-5.2" not in svg
    assert "dSpark" not in svg


def test_render_writes_signed_svg_evidence_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = data()
    monkeypatch.setattr(TOOL, "validate_inputs", lambda *_args: evidence)
    svg = tmp_path / "micro.svg"
    report_path = tmp_path / "micro.json"
    report = TOOL.render(
        serving_path=tmp_path / "serving",
        deployment_path=tmp_path / "deployment",
        blended_path=tmp_path / "blended",
        output_path=svg,
        report_path=report_path,
    )
    assert report["svg"]["sha256"] == hashlib.sha256(svg.read_bytes()).hexdigest()
    assert json.loads(report_path.read_text()) == report
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == hashlib.sha256(
        TOOL.canonical_json(body)
    ).hexdigest()
    with pytest.raises(TOOL.MicroTimelineError, match="overwrite"):
        TOOL.render(
            serving_path=tmp_path / "serving",
            deployment_path=tmp_path / "deployment",
            blended_path=tmp_path / "blended",
            output_path=svg,
            report_path=tmp_path / "again.json",
        )
