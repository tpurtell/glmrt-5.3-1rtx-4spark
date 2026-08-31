from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))

import render_glm53_exl3_model_card as TOOL  # noqa: E402


def test_rendered_k4_section_names_both_modes_and_exact_default() -> None:
    artifact = {
        "retained_native_bytes_verified": True,
        "artifact_manifest_file_hashes_verified": True,
        "projection_checkpoint_bytes_verified": True,
        "quantized_modules": 57_600,
        "exl3_tensors": 230_400,
        "exl3_tensor_bytes": 100 * 1024**3,
        "tp4_resident_bytes_per_spark": 25 * 1024**3,
        "source_metadata": [
            {"name": "tokenizer.json", "bytes": 1, "sha256": "1" * 64},
            {
                "name": "tokenizer_config.json",
                "bytes": 1,
                "sha256": "2" * 64,
            },
            {
                "name": "generation_config.json",
                "bytes": 1,
                "sha256": "3" * 64,
            },
        ],
    }
    quant = {
        "coverage": {"projection_count": 57_600, "complete_expert_count": 75 * 256},
        "integrity": {"tensor_payload_hashes_verified": True},
        "metrics": {"global": {"aggregate_hessian_weighted_relative_error": 0.01}},
    }
    serving = {
        "report_sha256": "a" * 64,
        "runtime": {
            "default_speculation": "dflash2",
            "engine_identity": "wip-test",
            "sparkinfer_revision": "b" * 40,
            "coordinator_slot_fingerprint": "c" * 64,
            "expert_slot_fingerprint": "d" * 64,
            "speculation_settings": {
                "mtp": {},
                "dflash2": {
                    "checkpoint_model_id": "incoai/GLM-5.3-DFlash2",
                    "checkpoint_revision": "425aa615ce320caac34400208b30808c8f14f76c",
                    "draft_policy": "adaptive",
                    "fixed_drafts": None,
                    "proposal_drafts": 7,
                    "topk_backend": "flashinfer-dsa",
                },
            },
        },
        "results": {
            "default_speculation": "dflash2",
            "modes": {
                mode: {
                    "weighted_decode_tps": 30.0 + index,
                    "agentic_code_decode_tps": 30.0 + index,
                    "agentic_c1_c2_c4_geomean_tps": 30.0 + index,
                    "repeat_decode_tps": 60.0 + index,
                    "accepted_draft_rate": 0.7,
                    "verify_cycle_by_physical_m": {
                        str(physical_m): {
                            "samples": 2,
                            "total_ms": 20.0 + index * 2,
                            "mean_ms": 10.0 + index,
                            "median_ms": 10.0 + index,
                            "min_ms": 9.0 + index,
                            "max_ms": 11.0 + index,
                        }
                        for physical_m in ([2, 6] if mode == "mtp" else [1, 8])
                    },
                    "tool_points": 100,
                    "tool_maximum_points": 100,
                    "maximum_service_handoff_total_ms": 10_000.0,
                    "decode_concurrency": {
                        str(concurrency): {
                            "mean_aggregate_decode_tps": 30.0 * concurrency + index,
                            "median_aggregate_decode_tps": 30.0 * concurrency + index,
                            "mean_response_window_tps": 29.0 * concurrency + index,
                        }
                        for concurrency in TOOL.REQUIRED_CONCURRENCIES
                    },
                    "long_context_needle": {
                        "maximum_wall_seconds": 200.0 + index,
                        "median_wall_seconds": 80.0 + index,
                        "measurements": [
                            {
                                "context_tokens": context,
                                "depth": depth,
                                "wall_seconds": context / 2_000.0 + depth + index,
                                "prefill_ms": context / 2.0,
                                "time_to_first_token_ms": context / 2.0 + 10.0,
                            }
                            for context in TOOL.REQUIRED_NEEDLE_CONTEXTS
                            for depth in TOOL.REQUIRED_NEEDLE_DEPTHS
                        ],
                    },
                }
                for index, mode in enumerate(TOOL.MODES)
            },
            "comparisons": {
                "dflash2_to_native_weighted_decode_ratio": 1.03,
                "dflash2_to_native_repeat_ratio": 1.02,
                "dflash2_to_native_acceptance_ratio": 1.0,
                "minimum_dflash2_to_native_prefill_ratio": 0.99,
                "dflash2_to_native_decode_concurrency_ratio": {
                    str(concurrency): 1.01
                    for concurrency in TOOL.REQUIRED_CONCURRENCIES
                },
                "dflash2_to_native_maximum_needle_wall_ratio": 1.01,
            },
            "semantic_decode": {
                "case_ids": list(TOOL.REQUIRED_SEMANTIC_CASE_IDS),
                "repeats": TOOL.REQUIRED_SEMANTIC_REPEATS,
                "cells": [
                    {
                        "case": case_id,
                        "category": f"category-{case_id}",
                        "samples": TOOL.REQUIRED_SEMANTIC_REPEATS,
                        "native_mtp_decode_tps": 30.0 + index,
                        "dflash2_decode_tps": 31.0 + index,
                        "dflash2_to_native_decode_ratio": (31.0 + index)
                        / (30.0 + index),
                        "native_mtp_accepted_draft_rate": 0.70,
                        "dflash2_accepted_draft_rate": 0.72,
                    }
                    for index, case_id in enumerate(TOOL.REQUIRED_SEMANTIC_CASE_IDS)
                ],
            },
            "prefill": {
                "cells": [
                    {
                        "base_context_tokens": 2048,
                        "suffix_tokens": 256,
                        "native_mtp_tps": 1000.0,
                        "dflash2_tps": 990.0,
                        "dflash2_to_native_ratio": 0.99,
                    }
                ]
            },
            "native_kernel": {
                "expert_slot_fingerprint": "d" * 64,
                "trellis_bits": 4,
                "tp_ranks": [0, 1, 2, 3],
                "layer_id": 3,
                "required_rows": list(range(21)),
            },
            "dflash2_preflight": {
                "checkpoint_repo_id": "incoai/GLM-5.3-DFlash2",
                "checkpoint_revision": "425aa615ce320caac34400208b30808c8f14f76c",
                "checkpoint_config_sha256": TOOL.DFLASH2_CONFIG_SHA256,
                "checkpoint_weight_lfs_sha256": TOOL.DFLASH2_WEIGHT_LFS_SHA256,
                "kv_storage": "bf16",
                "kv_element_bytes": 2,
                "page_size": 64,
                "kv_capacity_tokens": 2_176,
                "proposal_tokens_per_request": 7,
                "topk_backend": "flashinfer-dsa",
            },
            "dflash2_topk_tuning": {
                "selected_backend": "flashinfer-dsa",
                "micro_selected_backend": "flashinfer-dsa",
                "fastest_valid_backend": "flashinfer-dsa",
                "fastest_valid_speedup_vs_torch": 1.2,
                "aggregate_median_ms": {
                    "torch": 2.4,
                    "flashinfer": 2.2,
                    "flashinfer-dsa": 2.0,
                },
                "valid_backends": ["torch", "flashinfer", "flashinfer-dsa"],
                "full_service_gate": {
                    "selection_policy": TOOL.DFLASH2_TOPK_SERVICE_SELECTION_POLICY,
                    "minimum_non_torch_speedup": TOOL.DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP,
                    "selected_backend": "flashinfer-dsa",
                    "candidate_backend": "flashinfer-dsa",
                    "candidate_quality_passed": True,
                    "candidate_quality_failures": [],
                    "candidate_speedup_vs_torch": {
                        "weighted_decode": 1.04,
                        "median_repeat_decode": 1.03,
                    },
                    "weighted_decode_tps": {
                        "torch": 30.0,
                        "flashinfer-dsa": 31.2,
                    },
                    "median_repeat_decode_tps": {
                        "torch": 30.1,
                        "flashinfer-dsa": 31.0,
                    },
                    "accepted_draft_rate": {
                        "torch": 0.70,
                        "flashinfer-dsa": 0.71,
                    },
                    "response_hash_mismatches": 3,
                    "requests": len(TOOL.REQUIRED_SEMANTIC_CASE_IDS)
                    * TOOL.REQUIRED_SEMANTIC_REPEATS,
                },
            },
            "dflash2_fusion_tuning": {
                "selector": {
                    "winning_fused_warps": {"int32-c1-k1": 8},
                    "captured_launches": 16,
                },
                "body": {
                    "winning_fused_warps": {"c1-k1": 8},
                    "captured_launches": 16,
                },
            },
            "dflash2_adaptive": {
                "cost_profile": {
                    "profile_id": "glm53-test",
                    "source_sha256": "e" * 64,
                    "route_qualified_cells": 9,
                    "corpus_samples": 2_479,
                },
                "reference_width": 5,
                "response_performance_score": 31.0,
                "k5_response_performance_score": 30.0,
                "concurrency_geomean_tps": 60.0,
                "k5_concurrency_geomean_tps": 59.0,
                "weighted_decode_ratio_vs_k5": 1.03,
            },
        },
    }

    rendered = TOOL.render_section(
        artifact=artifact,
        quant=quant,
        serving=serving,
        hub_revision=None,
    )

    assert "`dflash2`" in rendered
    assert "mtp" in rendered
    assert "dflash2 (default)" in rendered
    assert "adaptive K1-K7 proposals" in rendered
    assert "`flashinfer-dsa` candidate selector" in rendered
    assert "C1/C2/C4, K1-K7 top-k sweep" in rendered
    assert "1.200x aggregate versus Torch" in rendered
    assert "BF16 draft KV, 64-token pages" in rendered
    assert "2,176-token cache envelope" in rendered
    assert "### DFlash2 adaptive policy" in rendered
    assert "2,479 corpus samples" in rendered
    assert "| Fixed K5 reference |" in rendered
    assert (
        "complete standalone `quantize_config.json.tensor_storage`" in rendered.lower()
    )
    assert "### Decode concurrency" in rendered
    assert "### Verify-cycle cost by physical M" in rendered
    assert "C1 post-TTFT target-cycle timings" in rendered
    assert "same `decode_ms` denominator used for TPS" in rendered
    assert "| 1 | — | — | 11.000 ms | 2 | — |" in rendered
    assert "| 8 | — | — | 11.000 ms | 2 | — |" in rendered
    assert "### Seven-type decode mix" in rendered
    assert "### Long-context needle recall" in rendered
    assert "393,216" in rendered

    serving["results"]["dflash2_adaptive"]["response_performance_score"] = 29.0
    with pytest.raises(TOOL.ModelCardError, match="does not beat"):
        TOOL.render_section(
            artifact=artifact,
            quant=quant,
            serving=serving,
            hub_revision=None,
        )

    serving["results"]["dflash2_adaptive"]["response_performance_score"] = 31.0
    serving["results"]["dflash2_preflight"]["checkpoint_config_sha256"] = "0" * 64
    with pytest.raises(TOOL.ModelCardError, match="production DFlash2 KV geometry"):
        TOOL.render_section(
            artifact=artifact,
            quant=quant,
            serving=serving,
            hub_revision=None,
        )

    serving["results"]["dflash2_preflight"][
        "checkpoint_config_sha256"
    ] = TOOL.DFLASH2_CONFIG_SHA256
    serving["results"]["dflash2_preflight"]["page_size"] = 128
    with pytest.raises(TOOL.ModelCardError, match="production DFlash2 KV geometry"):
        TOOL.render_section(
            artifact=artifact,
            quant=quant,
            serving=serving,
            hub_revision=None,
        )


def test_render_reopens_native_and_adaptive_dflash2_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = tmp_path / "template.md"
    template.write_text(f"before\n{TOOL.MARKER}\nafter\n", encoding="utf-8")
    artifact_path = tmp_path / "artifact.json"
    quant_path = tmp_path / "quant.json"
    serving_path = tmp_path / "serving.json"
    for path in (artifact_path, quant_path, serving_path):
        path.write_text("{}\n", encoding="utf-8")

    manifest_sha = "1" * 64
    plan_sha = "2" * 64
    evidence_sha = "3" * 64
    artifact = {
        "model_id": TOOL.GLM53_MODEL_ID,
        "artifact_manifest_sha256": manifest_sha,
        "plan_sha256": plan_sha,
        "projection_checkpoint": {"root": str(tmp_path)},
    }
    quant = {"plan": {"plan_sha256": plan_sha}}
    serving = {
        "model_id": TOOL.GLM53_MODEL_ID,
        "artifact": str(tmp_path.resolve()),
        "artifact_validation": {"sha256": evidence_sha},
        "quant_evidence": {"sha256": evidence_sha},
        "artifact_manifest_sha256": manifest_sha,
        "plan_sha256": plan_sha,
        "runtime": {
            "profile": "balanced",
            "speculation": "dflash2",
            "default_speculation": "dflash2",
            "qualified_speculation": list(TOOL.MODES),
            "sparkinfer_revision": "4" * 40,
            "coordinator_slot_fingerprint": "5" * 64,
            "expert_slot_fingerprint": "6" * 64,
        },
        "results": {"native_kernel": {"weight_source_root": str(tmp_path.resolve())}},
        "gates": {name: True for name in TOOL.REQUIRED_GATES},
        "failed_gates": [],
    }
    reports = {
        artifact_path: artifact,
        quant_path: quant,
        serving_path: serving,
    }

    def fake_signed_report(path: Path, _schema: str):
        resolved = path.resolve()
        return resolved, reports[resolved]

    calls: list[str] = []
    monkeypatch.setattr(TOOL, "signed_report", fake_signed_report)
    monkeypatch.setattr(TOOL, "hash_file", lambda _path: evidence_sha)
    monkeypatch.setattr(
        TOOL,
        "revalidate_native_evidence",
        lambda *_args, **_kwargs: calls.append("native"),
    )
    monkeypatch.setattr(
        TOOL,
        "revalidate_dflash2_fusion_evidence",
        lambda *_args, **_kwargs: calls.append("dflash2-fusions"),
    )
    monkeypatch.setattr(
        TOOL,
        "revalidate_dflash2_topk_evidence",
        lambda *_args, **_kwargs: calls.append("dflash2-topk"),
    )
    monkeypatch.setattr(
        TOOL,
        "revalidate_dflash2_adaptive_evidence",
        lambda *_args, **_kwargs: calls.append("dflash2-adaptive"),
    )
    monkeypatch.setattr(TOOL, "render_section", lambda **_kwargs: "accepted")

    rendered = TOOL.render(
        template_path=template,
        artifact_validation_path=artifact_path,
        quant_evidence_path=quant_path,
        serving_qualification_path=serving_path,
        hub_revision=None,
    )

    assert rendered == "before\naccepted\nafter\n"
    assert calls == [
        "native",
        "dflash2-fusions",
        "dflash2-topk",
        "dflash2-adaptive",
    ]

    def reject_adaptive(*_args, **_kwargs):
        raise TOOL.QualificationError("adaptive evidence changed")

    monkeypatch.setattr(TOOL, "revalidate_dflash2_adaptive_evidence", reject_adaptive)
    with pytest.raises(TOOL.ModelCardError, match="adaptive DFlash2 evidence"):
        TOOL.render(
            template_path=template,
            artifact_validation_path=artifact_path,
            quant_evidence_path=quant_path,
            serving_qualification_path=serving_path,
            hub_revision=None,
        )
