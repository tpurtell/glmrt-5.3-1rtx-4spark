from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "render_glm53_oss_benchmark_charts.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_oss_charts", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def evidence() -> dict[str, object]:
    return {
        "schema": "glmrt-glm53-oss-release-evidence-v1",
        "model_id": "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
        "model_revision": "1" * 40,
        "default_speculation": "dflash2",
        "runtime": {
            "speculation_settings": {
                "dflash2": {
                    "draft_policy": "adaptive",
                    "fixed_drafts": None,
                    "proposal_drafts": 7,
                }
            }
        },
        "results": {
            "serving": {
                "prefill": {
                    "cells": [
                        {
                            "base_context_tokens": base,
                            "suffix_tokens": suffix,
                            "native_mtp_tps": 1_000.0 + suffix / 100,
                            "dflash2_tps": 1_100.0 + suffix / 100,
                        }
                        for base in TOOL.DEFAULT_BASE_CONTEXTS
                        for suffix in TOOL.DEFAULT_SUFFIX_ROWS
                    ]
                }
            },
            "context_decode": {
                "cells": [
                    {
                        "context_bucket_tokens": context,
                        "workload": workload,
                        "decode_tps": 20.0 + index,
                    }
                    for context in TOOL.DEFAULT_CONTEXTS
                    for index, workload in enumerate(TOOL.WORKLOADS)
                ]
            },
        },
    }


def test_chart_inputs_render_selected_adaptive_dflash2_without_legacy_labels() -> None:
    prefill, decode = TOOL.chart_inputs(evidence())
    prefill_text = prefill.decode()
    decode_text = decode.decode()

    assert "EXL3 K4 · DFlash2 adaptive K1-K7 prefill" in prefill_text
    assert "256K cached" in prefill_text
    assert "32K" in prefill_text
    assert "EXL3 K4 · DFlash2 adaptive K1-K7 decode" in decode_text
    assert "Python code" in decode_text
    assert "Creative writing" in decode_text
    assert "GLM-5.2" not in prefill_text + decode_text
    assert "NVFP4" not in prefill_text + decode_text


def test_renderer_writes_signed_source_bound_svgs_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "oss.json"
    source.write_text("{}\n", encoding="utf-8")
    value = evidence()
    monkeypatch.setattr(TOOL, "validate_evidence", lambda _path: (source, value))
    prefill = tmp_path / "prefill.svg"
    decode = tmp_path / "decode.svg"
    report_path = tmp_path / "charts.json"

    report = TOOL.render(
        evidence_path=source,
        prefill_output=prefill,
        decode_output=decode,
        report_output=report_path,
    )

    assert report["status"] == "rendered"
    assert report["charts"]["prefill"]["sha256"] == hashlib.sha256(
        prefill.read_bytes()
    ).hexdigest()
    assert report["charts"]["decode"]["sha256"] == hashlib.sha256(
        decode.read_bytes()
    ).hexdigest()
    assert json.loads(report_path.read_text()) == report
    body = {key: item for key, item in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == hashlib.sha256(
        TOOL.canonical_json(body)
    ).hexdigest()
    with pytest.raises(TOOL.ChartRenderError, match="overwrite"):
        TOOL.render(
            evidence_path=source,
            prefill_output=prefill,
            decode_output=tmp_path / "new-decode.svg",
            report_output=tmp_path / "new-report.json",
        )


def test_chart_rejects_missing_cells_and_invalid_shape() -> None:
    value = evidence()
    value["results"]["context_decode"]["cells"].pop()
    with pytest.raises(TOOL.ChartRenderError, match="5x3"):
        TOOL.chart_inputs(value)
    with pytest.raises(TOOL.ChartRenderError, match="invalid shape"):
        TOOL.line_chart(
            title="broken",
            subtitle="broken",
            x_labels=["one"],
            series=[("only", [1.0])],
            x_axis="broken",
        )
