from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "render_glm53_oss_benchmark_markdown.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_oss_markdown", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def mode_result(mode: str) -> dict[str, object]:
    scale = 1.0 if mode == "mtp" else 1.2
    return {
        "weighted_decode_tps": 25.0 * scale,
        "agentic_code_decode_tps": 35.0 * scale,
        "repeat_decode_tps": 60.0 * scale,
        "accepted_draft_rate": 0.60 * scale,
        "tool_points": 120 if mode == "mtp" else 124,
        "tool_maximum_points": 138,
        "decode_concurrency": {
            str(concurrency): {
                "median_aggregate_decode_tps": 30.0 * concurrency * scale
            }
            for concurrency in TOOL.REQUIRED_CONCURRENCIES
        },
        "long_context_needle": {
            "measurements": [
                {
                    "context_tokens": context,
                    "depth": depth,
                    "wall_seconds": 1.0 + context / 100_000 + depth,
                }
                for context in TOOL.REQUIRED_NEEDLE_CONTEXTS
                for depth in (0.1, 0.5, 0.9)
            ]
        },
    }


def report(tmp_path: Path) -> dict[str, object]:
    startup_svg = tmp_path / "startup.svg"
    micro_svg = tmp_path / "micro.svg"
    startup_svg.write_text("<svg/>", encoding="utf-8")
    micro_svg.write_text("<svg/>", encoding="utf-8")
    prefill = [
        {
            "base_context_tokens": base,
            "suffix_tokens": suffix,
            "native_mtp_tps": 1_000.0 + suffix / 100,
            "dflash2_tps": 1_100.0 + suffix / 100,
        }
        for base in TOOL.DEFAULT_BASE_CONTEXTS
        for suffix in TOOL.DEFAULT_SUFFIX_ROWS
    ]
    semantic = [
        {
            "category": f"type-{index}",
            "case": f"case-{index}",
            "native_mtp_decode_tps": 20.0 + index,
            "native_mtp_accepted_draft_rate": 0.6,
            "dflash2_decode_tps": 25.0 + index,
            "dflash2_accepted_draft_rate": 0.75,
        }
        for index in range(8)
    ]
    contexts = [
        {
            "context_bucket_tokens": context,
            "workload": workload,
            "decode_tps": 30.0,
        }
        for context in TOOL.DEFAULT_CONTEXTS
        for workload in TOOL.WORKLOADS
    ]
    profiles = {
        profile: {
            mode: {
                "weighted_decode_tps": 30.0,
                "verify_tokens_per_second": 40.0,
                "accepted_draft_rate": 0.75,
                "cached_2k_plus_8k_prefill_tps": 1_500.0,
            }
            for mode in TOOL.MODE_LABELS
        }
        for profile in TOOL.PROFILES
    }
    return {
        "schema": TOOL.OSS_SCHEMA,
        "status": "accepted",
        "model_id": TOOL.GLM53_MODEL_ID,
        "model_revision": "1" * 40,
        "default_speculation": "dflash2",
        "runtime": {
            "profile": "balanced",
            "power_limit_w": 400,
            "engine_identity": "engine-v1",
            "sparkinfer_revision": "2" * 40,
            "speculation_settings": {
                "mtp": {},
                "dflash2": {
                    "checkpoint_model_id": "incoai/GLM-5.3-DFlash2",
                    "checkpoint_revision": "3" * 40,
                    "draft_policy": "adaptive",
                    "fixed_drafts": None,
                    "proposal_drafts": 7,
                    "topk_backend": "torch",
                },
            },
        },
        "results": {
            "serving": {
                "modes": {
                    "mtp": mode_result("mtp"),
                    "dflash2": mode_result("dflash2"),
                },
                "dflash2_adaptive": {
                    "concurrency_geomean_tps": 60.0,
                    "k5_concurrency_geomean_tps": 59.0,
                    "k5_response_performance_score": 30.0,
                    "reference_width": 5,
                    "response_performance_score": 31.0,
                    "weighted_decode_ratio_vs_k5": 1.05,
                },
                "prefill": {"cells": prefill},
                "semantic_decode": {"cells": semantic},
            },
            "context_decode": {"cells": contexts},
            "agentic": {
                "tool_eval": {
                    "seeds": [1, 2, 3],
                    "maximum_points": 138,
                    "median_points": 124,
                    "runs": [
                        {"points": points, "score": score}
                        for points, score in ((123, 89), (124, 90), (125, 91))
                    ],
                },
                "pi": {
                    thinking: {
                        "wall_seconds": 100.0,
                        "turns": 2,
                        "tool_calls": 1,
                        "tool_errors": 0,
                        "usage": {
                            "fresh_input": 1_000,
                            "cache_read": 2_000,
                            "output": 3_000,
                            "reasoning": 100 if thinking == "high" else 0,
                            "total": 6_000,
                        },
                        "artifact": {"bytes": 20_480},
                    }
                    for thinking in ("off", "high")
                },
            },
            "profiles": {"results": profiles},
            "startup": {
                "cold_wall_ms": 10_000.0,
                "warm_wall_ms": 2_000.0,
                "cold_to_warm_ratio": 5.0,
                "svg": {"path": str(startup_svg)},
            },
            "micro_timeline": {
                "selected_request": {
                    "case": "code",
                    "repeat": 1,
                    "decode_tps": 30.0,
                    "decode_ms": 1_000.0,
                    "target_cycles": 20,
                },
                "svg": {"path": str(micro_svg)},
            },
        },
        "evidence": {},
        "report_sha256": "f" * 64,
    }


def test_renders_every_release_table_without_legacy_model_rows(
    tmp_path: Path,
) -> None:
    chart_paths = {
        "prefill": tmp_path / "prefill.svg",
        "decode": tmp_path / "decode.svg",
    }
    for path in chart_paths.values():
        path.write_text("<svg/>\n", encoding="utf-8")
    markdown = TOOL.render_markdown(
        report(tmp_path), output_parent=tmp_path, charts=chart_paths
    )

    for heading in (
        "High-level performance",
        "Eight-type decode and acceptance",
        "Cache-aware prefill",
        "Decode across retained context",
        "Decode concurrency",
        "Agentic evaluation",
        "Long-context needle recall",
        "Performance by serving profile",
        "Startup and production timing",
    ):
        assert heading in markdown
    assert "DFlash2 adaptive K1-K7" in markdown
    assert "256K" in markdown
    assert "393216" not in markdown
    assert "384K" in markdown
    assert "NVFP4" not in markdown
    assert "EXL3 K3" not in markdown
    assert "(prefill.svg)" in markdown
    assert "(decode.svg)" in markdown


def test_rejects_incomplete_matrix_and_refuses_output_overwrite(
    tmp_path: Path,
) -> None:
    evidence = report(tmp_path)
    evidence["results"]["serving"]["prefill"]["cells"].pop()
    with pytest.raises(TOOL.BenchmarkRenderError, match="5x6"):
        TOOL.render_markdown(evidence, output_parent=tmp_path)

    output = tmp_path / "benchmarks.md"
    TOOL.atomic_text(output, "first\n")
    with pytest.raises(TOOL.BenchmarkRenderError, match="overwrite"):
        TOOL.atomic_text(output, "second\n")


def test_signed_input_validation_rechecks_report_identity(tmp_path: Path) -> None:
    value = report(tmp_path)
    value.pop("report_sha256")
    value["report_sha256"] = hashlib.sha256(TOOL.canonical_json(value)).hexdigest()
    path = tmp_path / "oss.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    resolved, accepted = TOOL.validate_evidence(path)
    assert resolved == path.resolve()
    assert accepted["default_speculation"] == "dflash2"

    value["runtime"]["profile"] = "accuracy"
    body = {key: item for key, item in value.items() if key != "report_sha256"}
    value["report_sha256"] = hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    (tmp_path / "wrong.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(TOOL.BenchmarkRenderError, match="balanced"):
        TOOL.validate_evidence(tmp_path / "wrong.json")


def test_chart_report_is_signed_source_bound_and_rehashed(tmp_path: Path) -> None:
    source = tmp_path / "oss.json"
    source.write_text("source\n", encoding="utf-8")
    accepted = report(tmp_path)
    chart_entries = {}
    for name in ("prefill", "decode"):
        path = tmp_path / f"{name}.svg"
        path.write_text(f"<svg>{name}</svg>\n", encoding="utf-8")
        chart_entries[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    body = {
        "schema": TOOL.CHART_SCHEMA,
        "status": "rendered",
        "model_id": accepted["model_id"],
        "model_revision": accepted["model_revision"],
        "default_speculation": accepted["default_speculation"],
        "source": TOOL.evidence_identity(source, accepted["schema"]),
        "charts": chart_entries,
    }
    chart_report = body | {
        "report_sha256": hashlib.sha256(TOOL.canonical_json(body)).hexdigest()
    }
    chart_path = tmp_path / "charts.json"
    chart_path.write_text(json.dumps(chart_report), encoding="utf-8")

    paths = TOOL.validate_charts(
        chart_path, evidence_file=source, evidence=accepted
    )
    assert paths == {
        name: Path(entry["path"]).resolve()
        for name, entry in chart_entries.items()
    }

    Path(chart_entries["decode"]["path"]).write_text(
        "<svg>changed</svg>\n", encoding="utf-8"
    )
    with pytest.raises(TOOL.BenchmarkRenderError, match="invalid"):
        TOOL.validate_charts(
            chart_path, evidence_file=source, evidence=accepted
        )
