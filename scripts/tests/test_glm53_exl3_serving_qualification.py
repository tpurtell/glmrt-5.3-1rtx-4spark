from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))

import validate_glm53_exl3_serving_qualification as TOOL  # noqa: E402


def mode(
    tool_points: int,
    decode_tps: float,
    *,
    code_tps: float | None = None,
    concurrency_tps: float | dict[int, float] | None = None,
) -> dict:
    code_tps = decode_tps if code_tps is None else code_tps
    concurrency_tps = decode_tps if concurrency_tps is None else concurrency_tps
    return {
        "tool_eval": {"total_points": tool_points},
        "blended": {
            "wall_decode_tps": decode_tps,
            "case_results": [{"case": "code", "decode_tps": code_tps}],
        },
        "concurrency": {
            "cells": {
                concurrency: {
                    "median_aggregate_decode_tps": (
                        concurrency_tps[concurrency]
                        if isinstance(concurrency_tps, dict)
                        else concurrency_tps
                    )
                }
                for concurrency in TOOL.REQUIRED_CONCURRENCIES
            }
        },
    }


def width_trials(*, winning_width: int, quality_winner: int | None = None) -> dict:
    return {
        width: mode(
            101 if width == quality_winner else 100,
            40.0 if width == winning_width else 30.0 + width,
        )
        for width in TOOL.DFLASH2_REQUIRED_WIDTHS
    }


def test_default_selection_uses_response_performance_not_tool_score() -> None:
    assert (
        TOOL.select_default_mode(
            {
                TOOL.MODE_NATIVE_MTP: mode(98, 100.0),
                TOOL.MODE_DFLASH2: mode(100, 90.0),
            }
        )
        == TOOL.MODE_NATIVE_MTP
    )
    assert (
        TOOL.select_default_mode(
            {
                TOOL.MODE_NATIVE_MTP: mode(100, 90.0),
                TOOL.MODE_DFLASH2: mode(100, 100.0),
            }
        )
        == TOOL.MODE_DFLASH2
    )


def test_default_selection_uses_native_mtp_when_it_is_measurably_better() -> None:
    assert (
        TOOL.select_default_mode(
            {
                TOOL.MODE_NATIVE_MTP: mode(100, 101.0),
                TOOL.MODE_DFLASH2: mode(100, 100.0),
            }
        )
        == TOOL.MODE_NATIVE_MTP
    )


def test_default_selection_balances_code_and_general_decode() -> None:
    assert (
        TOOL.select_default_mode(
            {
                TOOL.MODE_NATIVE_MTP: mode(100, 110.0, code_tps=40.0),
                TOOL.MODE_DFLASH2: mode(100, 90.0, code_tps=55.0),
            }
        )
        == TOOL.MODE_DFLASH2
    )


def test_default_selection_does_not_depend_on_unpaired_concurrency() -> None:
    assert (
        TOOL.select_default_mode(
            {
                TOOL.MODE_NATIVE_MTP: mode(
                    100,
                    100.0,
                    code_tps=40.0,
                    concurrency_tps={1: 40.0, 2: 60.0, 4: 80.0},
                ),
                TOOL.MODE_DFLASH2: mode(
                    100,
                    90.0,
                    code_tps=50.0,
                    concurrency_tps={1: 35.0, 2: 55.0, 4: 75.0},
                ),
            }
        )
        == TOOL.MODE_DFLASH2
    )


def test_default_selection_rejects_incomplete_or_invalid_measurements() -> None:
    with pytest.raises(TOOL.QualificationError, match="requires"):
        TOOL.select_default_mode({TOOL.MODE_NATIVE_MTP: mode(100, 1.0)})
    with pytest.raises(TOOL.QualificationError, match="invalid"):
        TOOL.select_default_mode(
            {
                TOOL.MODE_NATIVE_MTP: mode(100, 1.0),
                TOOL.MODE_DFLASH2: mode(100, 0.0),
            }
        )


def test_dflash_width_selection_uses_decode_not_tool_score() -> None:
    assert TOOL.select_dflash2_width(width_trials(winning_width=4)) == 4
    assert (
        TOOL.select_dflash2_width(width_trials(winning_width=4, quality_winner=2)) == 4
    )


def test_dflash_width_selection_prioritizes_code_over_general_decode() -> None:
    trials = width_trials(winning_width=4)
    trials[4] = mode(100, 50.0, code_tps=30.0)
    trials[6] = mode(100, 40.0, code_tps=45.0)
    assert TOOL.select_dflash2_width(trials) == 6


def test_dflash_width_selection_requires_all_seven_and_prefers_narrow_exact_tie() -> (
    None
):
    trials = width_trials(winning_width=4)
    trials[3] = mode(100, 40.0)
    assert TOOL.select_dflash2_width(trials) == 3
    trials.pop(7)
    with pytest.raises(TOOL.QualificationError, match="1 through 7"):
        TOOL.select_dflash2_width(trials)


def test_dflash_width_sweep_binds_all_trials_and_selects_the_measured_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def width(path: Path) -> int:
        return int(path.stem.rsplit("-", 1)[1])

    def fake_deployment(path: Path, **_kwargs) -> dict:
        fixed = width(path)
        return {
            "identity": {"path": str(path), "sha256": f"{fixed:064x}"},
            "slot": "a",
            "profile": "balanced",
            "launch_started_ns": 1_725_000_000_000_000_000 + fixed,
            "power_limit_w": 400,
            "engine_identity": "wip-a",
            "sparkinfer_revision": "b" * 40,
            "model_revision": "c" * 40,
            "fingerprints": {"coordinator_slot": "d" * 64, "expert_slot": "e" * 64},
            "speculation_settings": {
                "fixed_drafts": fixed,
                "topk_backend": "flashinfer-dsa",
            },
        }

    def fake_blended(path: Path, **_kwargs) -> dict:
        fixed = width(path)
        decode_tps = 50.0 if fixed == 4 else 30.0 + fixed
        return {
            "identity": {"path": str(path), "sha256": f"{fixed + 8:064x}"},
            "contract": "contract",
            "prompt_contract": {"suite": "weighted"},
            "prompts": [{"request": 1}],
            "wall_decode_tps": decode_tps,
            "case_results": [{"case": "code", "decode_tps": decode_tps}],
            "accepted_draft_rate": 0.5 + fixed / 100,
        }

    def fake_tool_eval(path: Path, **_kwargs) -> dict:
        fixed = width(path)
        return {
            "identity": {"path": str(path), "sha256": f"{fixed + 16:064x}"},
            "config": {"temperature": 0.0},
            "scenario_ids": ["TC-01"],
            "maximum_points": 2,
            "total_points": 2,
            "final_score": 100,
        }

    monkeypatch.setattr(TOOL, "deployment", fake_deployment)
    monkeypatch.setattr(TOOL, "blended", fake_blended)
    monkeypatch.setattr(TOOL, "tool_eval", fake_tool_eval)
    monkeypatch.setattr(
        TOOL,
        "verify_cycle_curve",
        lambda path, **_kwargs: {
            str(width(path) + 1): {
                "samples": 2,
                "total_ms": 20.0,
                "mean_ms": 10.0,
                "median_ms": 10.0,
                "min_ms": 9.0,
                "max_ms": 11.0,
            }
        },
    )
    monkeypatch.setattr(TOOL, "require_eight_type_blended", lambda evidence: None)
    trials = [
        (
            fixed,
            tmp_path / f"deployment-{fixed}",
            tmp_path / f"blended-{fixed}",
            tmp_path / f"tools-{fixed}",
        )
        for fixed in TOOL.DFLASH2_REQUIRED_WIDTHS
    ]

    sweep = TOOL.dflash2_width_sweep(
        trials,
        expected_tool_eval_version=TOOL.TOOL_EVAL_VERSION,
    )

    assert sweep["winner"] == 4
    assert [trial["width"] for trial in sweep["trials"]] == list(range(1, 8))
    assert sweep["trials"][3]["weighted_decode_tps"] == 50.0
    assert sweep["trials"][3]["code_decode_tps"] == 50.0
    assert set(sweep["trials"][3]["verify_cycle_by_physical_m"]) == {"5"}
    assert len({trial["launch_started_ns"] for trial in sweep["trials"]}) == 7


def test_verify_cycle_curve_recomputes_each_physical_m_and_rejects_drift(
    tmp_path: Path,
) -> None:
    path = tmp_path / "blended.jsonl"
    measurements = [
        {
            "verify_cycles": 2,
            "draft_lengths": [1, 1],
            "accepted_draft_lengths": [1, 1],
            "verify_cycle_ms": [10.0, 20.0],
            "target_cycle_physical_m": [1, 2],
            "target_cycle_ms": [8.0, 10.0],
            "decode_ms": 18.0,
        },
        {
            "verify_cycles": 2,
            "draft_lengths": [1, 1],
            "accepted_draft_lengths": [0, 1],
            "verify_cycle_ms": [14.0, 24.0],
            "target_cycle_physical_m": [1, 3],
            "target_cycle_ms": [12.0, 24.0],
            "decode_ms": 36.0,
        },
    ]
    aggregate = {
        "verify_cycles": 4,
        "physical_m_histogram": {"2": 4},
        "target_cycle_physical_m_histogram": {"1": 2, "2": 1, "3": 1},
        "target_cycle_ms_by_physical_m": {
            "1": {
                "samples": 2,
                "total_ms": 20.0,
                "mean_ms": 10.0,
                "median_ms": 10.0,
                "min_ms": 8.0,
                "max_ms": 12.0,
            },
            "2": {
                "samples": 1,
                "total_ms": 10.0,
                "mean_ms": 10.0,
                "median_ms": 10.0,
                "min_ms": 10.0,
                "max_ms": 10.0,
            },
            "3": {
                "samples": 1,
                "total_ms": 24.0,
                "mean_ms": 24.0,
                "median_ms": 24.0,
                "min_ms": 24.0,
                "max_ms": 24.0,
            },
        },
    }
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [*measurements, {"aggregate": aggregate}]
        ),
        encoding="utf-8",
    )

    curve = TOOL.verify_cycle_curve(path)
    assert curve["1"]["median_ms"] == 10.0
    assert curve["2"]["median_ms"] == 10.0
    assert curve["3"]["samples"] == 1

    with pytest.raises(TOOL.QualificationError, match="exceeds fixed physical M"):
        TOOL.verify_cycle_curve(path, expected_fixed_drafts=1)

    measurements[0]["decode_ms"] = 17.0
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [*measurements, {"aggregate": aggregate}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(TOOL.QualificationError, match="target-cycle/decode timing"):
        TOOL.verify_cycle_curve(path)

    measurements[0]["decode_ms"] = 18.0
    aggregate["target_cycle_ms_by_physical_m"]["3"]["median_ms"] = 21.0
    path.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [*measurements, {"aggregate": aggregate}]
        ),
        encoding="utf-8",
    )
    with pytest.raises(TOOL.QualificationError, match="median_ms"):
        TOOL.verify_cycle_curve(path)


def test_verify_cycle_curve_accepts_a_native_scalar_only_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "native-scalar.jsonl"
    measurement = {
        "verify_cycles": 0,
        "draft_lengths": [],
        "accepted_draft_lengths": [],
        "verify_cycle_ms": [],
        "target_cycle_physical_m": [1, 1],
        "target_cycle_ms": [7.0, 9.0],
        "decode_ms": 16.0,
    }
    aggregate = {
        "verify_cycles": 0,
        "target_cycle_physical_m_histogram": {"1": 2},
        "target_cycle_ms_by_physical_m": {
            "1": {
                "samples": 2,
                "total_ms": 16.0,
                "mean_ms": 8.0,
                "median_ms": 8.0,
                "min_ms": 7.0,
                "max_ms": 9.0,
            }
        },
    }
    path.write_text(
        json.dumps(measurement) + "\n" + json.dumps({"aggregate": aggregate}) + "\n",
        encoding="utf-8",
    )

    curve = TOOL.verify_cycle_curve(path)

    assert curve == aggregate["target_cycle_ms_by_physical_m"]
    with pytest.raises(TOOL.QualificationError, match="no verify cycles"):
        TOOL.verify_cycle_curve(path, expected_fixed_drafts=1)


def test_verify_cycle_curve_accepts_fixed_dflash_tail_widths(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dflash-tail.jsonl"
    measurement = {
        "verify_cycles": 2,
        "draft_lengths": [3, 1],
        "accepted_draft_lengths": [2, 0],
        "verify_cycle_ms": [12.0, 8.0],
        "target_cycle_physical_m": [4, 2, 1],
        "target_cycle_ms": [12.0, 8.0, 6.0],
        "decode_ms": 26.0,
    }
    aggregate = {
        "verify_cycles": 2,
        "target_cycle_physical_m_histogram": {"1": 1, "2": 1, "4": 1},
        "target_cycle_ms_by_physical_m": {
            str(physical_m): {
                "samples": 1,
                "total_ms": elapsed_ms,
                "mean_ms": elapsed_ms,
                "median_ms": elapsed_ms,
                "min_ms": elapsed_ms,
                "max_ms": elapsed_ms,
            }
            for physical_m, elapsed_ms in ((1, 6.0), (2, 8.0), (4, 12.0))
        },
    }
    path.write_text(
        json.dumps(measurement) + "\n" + json.dumps({"aggregate": aggregate}) + "\n",
        encoding="utf-8",
    )

    curve = TOOL.verify_cycle_curve(path, expected_fixed_drafts=3)

    assert set(curve) == {"1", "2", "4"}
    with pytest.raises(TOOL.QualificationError, match="exceeds fixed draft width"):
        TOOL.verify_cycle_curve(path, expected_fixed_drafts=2)


def test_rejects_reused_service_launches() -> None:
    with pytest.raises(TOOL.QualificationError, match="reused a service launch"):
        TOOL.require_distinct_launch_instances(
            "DFlash2 width sweep",
            [{"launch_started_ns": 123}, {"launch_started_ns": 123}],
        )


def test_tool_eval_pairing_allows_a_real_quality_difference() -> None:
    fixture = {
        "config": {"model": TOOL.GLM53_MODEL_ID, "temperature": 0.0},
        "scenario_ids": ["TC-01", "TC-02"],
        "maximum_points": 4,
    }
    native = {**fixture, "total_points": 3, "scenarios": [("TC-01", 1, "partial")]}
    dflash2 = {**fixture, "total_points": 4, "scenarios": [("TC-01", 2, "pass")]}

    TOOL._pair_tool_evaluations(native, dflash2)


def test_tool_eval_pairing_rejects_a_different_scenario_fixture() -> None:
    native = {
        "config": {"model": TOOL.GLM53_MODEL_ID, "temperature": 0.0},
        "scenario_ids": ["TC-01", "TC-02"],
        "maximum_points": 4,
    }
    dflash2 = {**native, "scenario_ids": ["TC-01", "TC-03"]}

    with pytest.raises(TOOL.QualificationError, match="scenario sequence"):
        TOOL._pair_tool_evaluations(native, dflash2)


def test_k4_native_rows_cover_every_aot_boundary_and_2064_suffix() -> None:
    assert {1, 2048, 2049, 2064}.issubset(TOOL.K4_REQUIRED_NATIVE_ROWS)
    assert set(range(1, 33)).issubset(TOOL.K4_REQUIRED_NATIVE_ROWS)
    assert len(TOOL.K4_REQUIRED_NATIVE_ROWS) == 44


@pytest.mark.parametrize("include_mtp", [False, True])
def test_glm53_startup_evidence_requires_the_k4_schema(
    tmp_path: Path, include_mtp: bool
) -> None:
    report = {
        "schema": TOOL.GLM53_STARTUP_SCHEMA,
        "status": "accepted",
        "model": TOOL.GLM53_MODEL_ID,
        "expert_runtime_fingerprint": "a" * 64,
        "weight_format": "exl3",
        "preload_mode": "direct-resident",
        "cache_state": "warm",
        "include_mtp": include_mtp,
        "hosts": [{"host": host} for host in ("ostrich", "dodo", "emu", "kiwi")],
        "summary": {
            "maximum_resident_preload_ms": 1.0,
            "maximum_service_handoff_total_ms": 2.0,
        },
    }
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(report)).hexdigest()
    path = tmp_path / "startup.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    accepted = TOOL.startup(
        path,
        candidate=True,
        expected_model=TOOL.GLM53_MODEL_ID,
        expected_weight_format="exl3",
        expected_preload_modes={"direct-resident"},
        expected_include_mtp=include_mtp,
        expected_schema=TOOL.GLM53_STARTUP_SCHEMA,
    )
    assert accepted["include_mtp"] is include_mtp

    report["schema"] = "glmrt-glm52-expert-startup-v2"
    report["report_sha256"] = hashlib.sha256(
        TOOL.canonical_json(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="startup evidence"):
        TOOL.startup(
            path,
            candidate=True,
            expected_model=TOOL.GLM53_MODEL_ID,
            expected_weight_format="exl3",
            expected_preload_modes={"direct-resident"},
            expected_include_mtp=include_mtp,
            expected_schema=TOOL.GLM53_STARTUP_SCHEMA,
        )


def test_glm53_blended_gate_requires_exact_eight_type_five_replay_corpus() -> None:
    nonce_seed = 53
    tokenizer_sha256 = "a" * 64
    case_results = [
        {
            "case": case_id,
            "category": TOOL.SEMANTIC_DECODE_CASES[case_id].category,
            "samples": TOOL.REQUIRED_SEMANTIC_REPEATS,
        }
        for case_id in TOOL.REQUIRED_SEMANTIC_CASE_IDS
    ]
    evidence = {
        "prompt_contract": {
            "suite": "weighted",
            "cases": [
                {
                    "id": case_id,
                    "category": TOOL.SEMANTIC_DECODE_CASES[case_id].category,
                    "prompt": TOOL.SEMANTIC_DECODE_CASES[case_id].prompt,
                    "max_tokens": TOOL.SEMANTIC_DECODE_CASES[case_id].max_tokens,
                    "weight": TOOL.SEMANTIC_DECODE_CASES[case_id].weight,
                    "response_format": (
                        TOOL.structured_edit_response_format()
                        if TOOL.SEMANTIC_DECODE_CASES[case_id].json_schema
                        else None
                    ),
                }
                for case_id in TOOL.REQUIRED_SEMANTIC_CASE_IDS
            ],
            "repeats": TOOL.REQUIRED_SEMANTIC_REPEATS,
            "nonce_seed": nonce_seed,
            "nonce_policy": "token-zero",
            "tokenizer_sha256": tokenizer_sha256,
            "temperature": 0,
            "enable_thinking": False,
            "quality_contract_version": TOOL.SEMANTIC_QUALITY_CONTRACT_VERSION,
            "request_binding_version": TOOL.SEMANTIC_REQUEST_BINDING_VERSION,
        },
        "cases": len(TOOL.REQUIRED_SEMANTIC_CASE_IDS) * TOOL.REQUIRED_SEMANTIC_REPEATS,
        "case_results": case_results,
        "prompts": [],
    }
    for request_index in range(
        len(TOOL.REQUIRED_SEMANTIC_CASE_IDS) * TOOL.REQUIRED_SEMANTIC_REPEATS
    ):
        case_index = request_index % len(TOOL.REQUIRED_SEMANTIC_CASE_IDS)
        repeat_index = request_index // len(TOOL.REQUIRED_SEMANTIC_CASE_IDS)
        case_id = TOOL.REQUIRED_SEMANTIC_CASE_IDS[case_index]
        marker = chr(0x4E00 + request_index)
        prompt = (
            f"{marker} request nonce {nonce_seed}-{request_index}.\n"
            "Treat the preceding request nonce as irrelevant.\n"
            f"{TOOL.SEMANTIC_DECODE_CASES[case_id].prompt}"
        )
        request = TOOL.semantic_completion_payload(
            TOOL.GLM53_MODEL_ID,
            TOOL.SEMANTIC_DECODE_CASES[case_id],
            prompt_prefix=(
                f"{marker} request nonce {nonce_seed}-{request_index}.\n"
                "Treat the preceding request nonce as irrelevant.\n"
            ),
        )
        evidence["prompts"].append(
            {
                "case": case_id,
                "repeat": repeat_index + 1,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request_sha256": hashlib.sha256(request).hexdigest(),
                "nonce": {
                    "marker": marker,
                    "first_content_token_id": 1_000 + request_index,
                },
            }
        )
    TOOL.require_eight_type_blended(evidence)

    evidence["prompt_contract"]["cases"].pop()
    with pytest.raises(TOOL.QualificationError, match="eight-type"):
        TOOL.require_eight_type_blended(evidence)


def test_dflash_preflight_requires_all_concurrency_graphs(
    tmp_path: Path,
) -> None:
    timing = {name: 1.0 for name in ("min", "median", "p90", "max")}
    graph = {
        "accepted_rows_per_request": 1,
        "query_rows_per_request": 8,
        "proposal_tokens_per_request": 7,
        "eager_replay_exact": True,
        "dynamic_anchor_changes_output": True,
        "restored_replay_exact": True,
        "target_embedding_alias": True,
        "target_lm_head_alias": True,
        "body_output_is_head_source": True,
        "shared_update_body_kv": True,
        "candidate_topk_sorted": True,
        "candidate_score_accumulation": "bf16-edge-plus-unary-bf16",
        "sliding_window_tokens": 2048,
        "hot_replay_python_calls": 0,
        "gpu_ms_per_update_replay": timing,
        "gpu_ms_per_suffix_replay": timing,
        "gpu_ms_per_full_cycle": timing,
        "host_ms_per_full_cycle": timing,
    }

    def update_validation(rows: list[int]) -> list[dict[str, object]]:
        return [
            {
                "rows": value,
                "reference_fused_hidden_max_abs": 0.0,
                "reference_fused_hidden_relative_l2": 0.0,
                "reference_key_max_abs": 0.0,
                "reference_key_bf16_steps_at_max_abs": 0,
                "reference_key_relative_l2": 0.0,
                "reference_value_bf16_steps_at_max_abs": 0,
                "reference_value_relative_l2": 0.0,
                "eager_replay_exact": True,
                "dynamic_positions_change_keys": True,
                "dynamic_key_changed_bytes": 1,
                "restored_replay_exact": True,
            }
            for value in rows
        ]

    report = {
        "schema": TOOL.DFLASH2_PREFLIGHT_SCHEMA,
        "status": "accepted",
        "checkpoint_repo_id": TOOL.DFLASH2_REPO_ID,
        "checkpoint_revision": TOOL.DFLASH2_REVISION,
        "checkpoint_config_sha256": TOOL.DFLASH2_CONFIG_SHA256,
        "checkpoint_weight_lfs_sha256": TOOL.DFLASH2_WEIGHT_LFS_SHA256,
        "target_repo_id": "zai-org/GLM-5.3",
        "tensor_count": TOOL.DFLASH2_TENSOR_COUNT,
        "payload_bytes": TOOL.DFLASH2_PAYLOAD_BYTES,
        "kv_storage": TOOL.DFLASH2_SERVING_KV_STORAGE,
        "kv_element_bytes": TOOL.DFLASH2_SERVING_KV_ELEMENT_BYTES,
        "page_size": TOOL.DFLASH2_SERVING_KV_PAGE_SIZE,
        "kv_capacity_tokens": TOOL.DFLASH2_SERVING_KV_CAPACITY_TOKENS,
        "proposal_tokens_per_request": 7,
        "query_rows_per_request": 8,
        "topk_backend": "flashinfer-dsa",
        "resident_preload": {
            "source_tensors": TOOL.DFLASH2_TENSOR_COUNT,
            "loaded_source_tensors": TOOL.DFLASH2_TENSOR_COUNT,
            "selected_bytes": TOOL.DFLASH2_PAYLOAD_BYTES,
            "loaded_bytes": TOOL.DFLASH2_PAYLOAD_BYTES,
        },
        "target_alias_preload": {
            "selected_tensors": 2,
            "loaded_tensors": 2,
            "selected_bytes": 100,
            "loaded_bytes": 100,
        },
        "concurrency_plans": [
            {
                "active_requests": value,
                "query_rows_per_request": 8,
                "proposal_tokens_per_request": 7,
                "total_physical_pages": TOOL.DFLASH2_SERVING_PHYSICAL_PAGES,
                "max_pages_per_request": TOOL.DFLASH2_SERVING_MAX_PAGES_PER_REQUEST,
            }
            for value in (1, 2, 4)
        ],
        "static_graphs": [
            graph
            | {
                "active_requests": value,
                "total_physical_pages": TOOL.DFLASH2_SERVING_PHYSICAL_PAGES,
                "max_pages_per_request": TOOL.DFLASH2_SERVING_MAX_PAGES_PER_REQUEST,
                "base_update_graph_validation": update_validation([value])[0],
                "packed_update_graph_rows": {
                    1: [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024],
                    2: [2, 4, 8, 16],
                    4: [4, 8, 16, 32],
                }[value],
                "packed_update_graph_validation": update_validation(
                    {
                        1: [2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024],
                        2: [2, 4, 8, 16],
                        4: [4, 8, 16, 32],
                    }[value]
                ),
            }
            for value in (1, 2, 4)
        ],
    }
    path = tmp_path / "preflight.json"
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")

    accepted = TOOL.dflash2_preflight(path)
    assert accepted["checkpoint_revision"] == TOOL.DFLASH2_REVISION
    assert accepted["checkpoint_config_sha256"] == TOOL.DFLASH2_CONFIG_SHA256
    assert accepted["checkpoint_weight_lfs_sha256"] == TOOL.DFLASH2_WEIGHT_LFS_SHA256
    assert accepted["kv_storage"] == "bf16"
    assert accepted["page_size"] == 64
    assert accepted["kv_capacity_tokens"] == 2_176
    assert accepted["topk_backend"] == "flashinfer-dsa"
    assert report["static_graphs"][0]["packed_update_graph_rows"][:7] == list(
        range(2, 9)
    )

    report["checkpoint_config_sha256"] = "0" * 64
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="pinned checkpoint"):
        TOOL.dflash2_preflight(path)
    report["checkpoint_config_sha256"] = TOOL.DFLASH2_CONFIG_SHA256

    report["checkpoint_weight_lfs_sha256"] = "0" * 64
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="pinned checkpoint"):
        TOOL.dflash2_preflight(path)
    report["checkpoint_weight_lfs_sha256"] = TOOL.DFLASH2_WEIGHT_LFS_SHA256

    report["static_graphs"][0]["total_physical_pages"] = 34
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="four-slot shared KV pool"):
        TOOL.dflash2_preflight(path)
    report["static_graphs"][0][
        "total_physical_pages"
    ] = TOOL.DFLASH2_SERVING_PHYSICAL_PAGES

    for plan in report["concurrency_plans"]:
        plan["proposal_tokens_per_request"] = 5
        plan["query_rows_per_request"] = 6
    for static_graph in report["static_graphs"]:
        static_graph["proposal_tokens_per_request"] = 5
        static_graph["query_rows_per_request"] = 6
    report["proposal_tokens_per_request"] = 5
    report["query_rows_per_request"] = 6
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    assert TOOL.dflash2_preflight(path)["proposal_tokens_per_request"] == 5

    report["static_graphs"][0]["packed_update_graph_rows"].remove(3)
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="static graph contract"):
        TOOL.dflash2_preflight(path)
    report["static_graphs"][0]["packed_update_graph_rows"].insert(1, 3)

    report["static_graphs"][0]["base_update_graph_validation"][
        "reference_key_relative_l2"
    ] = 0.02
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="eager/reference"):
        TOOL.dflash2_preflight(path)
    report["static_graphs"][0]["base_update_graph_validation"][
        "reference_key_relative_l2"
    ] = 0.0

    for field, replacement in (
        ("kv_storage", "fp8"),
        ("kv_element_bytes", 1),
        ("page_size", 128),
        ("kv_capacity_tokens", 2_304),
    ):
        original = report[field]
        report[field] = replacement
        path.write_text(TOOL.json.dumps(report), encoding="utf-8")
        with pytest.raises(TOOL.QualificationError, match="pinned checkpoint"):
            TOOL.dflash2_preflight(path)
        report[field] = original

    report["static_graphs"].pop()
    path.write_text(TOOL.json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="C1/C2/C4"):
        TOOL.dflash2_preflight(path)


def test_dflash_topk_tuning_recomputes_complete_backend_selection(
    tmp_path: Path,
) -> None:
    backends = list(TOOL.DFLASH2_TOPK_BACKENDS)
    medians = {"torch": 1.0, "flashinfer": 0.9, "flashinfer-dsa": 0.8}
    results = []
    for concurrency in TOOL.REQUIRED_CONCURRENCIES:
        for width in TOOL.DFLASH2_REQUIRED_WIDTHS:
            results.append(
                {
                    "active_requests": concurrency,
                    "proposal_tokens": width,
                    "rows": concurrency * width,
                    "initial_valid": {backend: True for backend in backends},
                    "changed_input_valid": {backend: True for backend in backends},
                    "initial_index_exact": {backend: True for backend in backends},
                    "changed_input_index_exact": {
                        backend: True for backend in backends
                    },
                    "tie_policy": "equal_topk_values_valid_unique_ids_boundary_ties_allowed",
                    "timing_ms": {
                        backend: {
                            "minimum": median * 0.9,
                            "median": median,
                            "p90": median * 1.1,
                            "maximum": median * 1.2,
                        }
                        for backend, median in medians.items()
                    },
                    "speedup_vs_torch": {
                        backend: medians["torch"] / median
                        for backend, median in medians.items()
                    },
                    "unsupported_backends": {},
                }
            )
    cases = len(TOOL.REQUIRED_CONCURRENCIES) * len(TOOL.DFLASH2_REQUIRED_WIDTHS)
    report = {
        "schema": TOOL.DFLASH2_TOPK_TUNING_SCHEMA,
        "status": "measured",
        "repo_id": TOOL.DFLASH2_REPO_ID,
        "revision": TOOL.DFLASH2_REVISION,
        "snapshot": "/models/dflash2",
        "config_sha256": "1" * 64,
        "weight_sha256": "2" * 64,
        "runtime_head_sha256": TOOL.source_sha256(
            TOOL.REFERENCE_ROOT / "dflash_head_capture.py"
        ),
        "script_sha256": TOOL.source_sha256(TOOL.TOOLS_ROOT / "tune_dflash2_topk.py"),
        "flashinfer_version": "0.6.16",
        "torch_version": "2.9.1",
        "device": "RTX PRO 6000 Blackwell",
        "compute_capability": [12, 0],
        "concurrency": list(TOOL.REQUIRED_CONCURRENCIES),
        "widths": list(TOOL.DFLASH2_REQUIRED_WIDTHS),
        "seed": 53,
        "warmup": 50,
        "iterations": 500,
        "rounds": 9,
        "captured_launches": 32,
        "minimum_non_torch_speedup": TOOL.DFLASH2_TOPK_MIN_NON_TORCH_SPEEDUP,
        "selection_policy": TOOL.DFLASH2_TOPK_SELECTION_POLICY,
        "full_service_acceptance_required": True,
        "valid_backends": backends,
        "aggregate_median_ms": {
            backend: median * cases for backend, median in medians.items()
        },
        "fastest_valid_backend": "flashinfer-dsa",
        "fastest_valid_speedup_vs_torch": 1.25,
        "selected_backend": "flashinfer-dsa",
        "unsupported_backends": {},
        "results": results,
    }
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(report)).hexdigest()
    path = tmp_path / "topk.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    accepted = TOOL.dflash2_topk_tuning(path)
    assert accepted["selected_backend"] == "flashinfer-dsa"
    assert accepted["fastest_valid_speedup_vs_torch"] == pytest.approx(1.25)

    report["selected_backend"] = "torch"
    report["report_sha256"] = hashlib.sha256(
        TOOL.canonical_json(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="backend selection"):
        TOOL.dflash2_topk_tuning(path)

    report["valid_backends"] = ["torch", "flashinfer"]
    report["aggregate_median_ms"].pop("flashinfer-dsa")
    report["fastest_valid_backend"] = "flashinfer"
    report["fastest_valid_speedup_vs_torch"] = medians["torch"] / medians["flashinfer"]
    report["selected_backend"] = "flashinfer"
    report["unsupported_backends"] = {"flashinfer-dsa": "unsupported on SM120"}
    for result in report["results"]:
        result["initial_valid"].pop("flashinfer-dsa")
        result["changed_input_valid"].pop("flashinfer-dsa")
        result["initial_index_exact"].pop("flashinfer-dsa")
        result["changed_input_index_exact"].pop("flashinfer-dsa")
        result["timing_ms"].pop("flashinfer-dsa")
        result["speedup_vs_torch"].pop("flashinfer-dsa")
        result["unsupported_backends"] = {"flashinfer-dsa": "unsupported on SM120"}
    report["report_sha256"] = hashlib.sha256(
        TOOL.canonical_json(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")

    accepted = TOOL.dflash2_topk_tuning(path)
    assert accepted["selected_backend"] == "flashinfer"


def test_dflash_topk_service_gate_overrides_a_micro_winner_on_quality(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch_deployment = tmp_path / "torch-deployment.json"
    candidate_deployment = tmp_path / "candidate-deployment.json"
    torch_blended = tmp_path / "torch-blended.jsonl"
    candidate_blended = tmp_path / "candidate-blended.jsonl"
    for path, first_digest in (
        (torch_blended, "1" * 64),
        (candidate_blended, "2" * 64),
    ):
        path.write_text(
            "".join(
                json.dumps({"content_sha256": first_digest if index == 0 else "3" * 64})
                + "\n"
                for index in range(40)
            ),
            encoding="utf-8",
        )

    candidate_quality = {"passed": False}

    def fake_deployment(path: Path, **_kwargs) -> dict:
        is_candidate = "candidate" in path.name
        return {
            "identity": {
                "path": str(path),
                "sha256": ("2" if is_candidate else "1") * 64,
            },
            "slot": "slot",
            "profile": "balanced",
            "power_limit_w": 400,
            "engine_identity": "wip-slot",
            "sparkinfer_revision": "a" * 40,
            "model_revision": "b" * 40,
            "launch_started_ns": 2 if is_candidate else 1,
            "fingerprints": {
                "coordinator_slot": "c" * 64,
                "expert_slot": "d" * 64,
            },
            "speculation_settings": {
                "checkpoint_model_id": TOOL.DFLASH2_REPO_ID,
                "checkpoint_revision": TOOL.DFLASH2_REVISION,
                "draft_policy": "fixed",
                "fixed_drafts": TOOL.DFLASH2_REFERENCE_WIDTH,
                "proposal_drafts": max(TOOL.DFLASH2_REQUIRED_WIDTHS),
                "topk_backend": "flashinfer" if is_candidate else "torch",
            },
        }

    def fake_blended(path: Path, **_kwargs) -> dict:
        is_candidate = "candidate" in path.name
        quality = candidate_quality["passed"] if is_candidate else True
        return {
            "identity": {
                "path": str(path),
                "sha256": ("4" if is_candidate else "3") * 64,
            },
            "contract": "contract",
            "prompt_contract": {"suite": "weighted"},
            "prompts": [{"request": index} for index in range(40)],
            "cases": 40,
            "wall_decode_tps": 31.0 if is_candidate else 30.0,
            "median_repeat_wall_decode_tps": 30.5 if is_candidate else 30.0,
            "accepted_draft_rate": 0.51 if is_candidate else 0.50,
            "all_quality_contracts_passed": quality,
            "quality_contract_failures": [] if quality else [{"case": "multilingual"}],
        }

    monkeypatch.setattr(TOOL, "deployment", fake_deployment)
    monkeypatch.setattr(TOOL, "blended", fake_blended)
    monkeypatch.setattr(TOOL, "require_eight_type_blended", lambda _evidence: None)
    kwargs = {
        "torch_deployment_path": torch_deployment,
        "torch_blended_path": torch_blended,
        "candidate_deployment_path": candidate_deployment,
        "candidate_blended_path": candidate_blended,
        "topk_tuning": {"selected_backend": "flashinfer"},
    }
    rejected = TOOL.dflash2_topk_service_gate(**kwargs)
    assert rejected["selected_backend"] == "torch"
    assert rejected["response_hash_mismatches"] == 1

    candidate_quality["passed"] = True
    accepted = TOOL.dflash2_topk_service_gate(**kwargs)
    assert accepted["selected_backend"] == "flashinfer"


@pytest.mark.parametrize("kind", ["selector", "body"])
def test_dflash_fusion_tuning_binds_measured_winners_to_runtime_profile(
    tmp_path: Path,
    kind: str,
) -> None:
    schema = (
        TOOL.DFLASH2_SELECTOR_TUNING_SCHEMA
        if kind == "selector"
        else TOOL.DFLASH2_BODY_FUSION_TUNING_SCHEMA
    )
    runtime_name = (
        "dflash_head_capture.py" if kind == "selector" else "dspark_body_capture.py"
    )
    runtime_hash_field = (
        "runtime_selector_sha256" if kind == "selector" else "runtime_body_sha256"
    )
    dtypes = ("int64", "int32") if kind == "selector" else (None,)
    results = []
    winners = {}
    for dtype in dtypes:
        for concurrency in TOOL.REQUIRED_CONCURRENCIES:
            for width in TOOL.DFLASH2_REQUIRED_WIDTHS:
                key = (
                    f"{dtype}-c{concurrency}-k{width}"
                    if dtype is not None
                    else f"c{concurrency}-k{width}"
                )
                selected_warps = 4 if kind == "selector" else 8
                winners[key] = selected_warps
                fused_timings = (
                    ((4, 0.8), (8, 0.9))
                    if selected_warps == 4
                    else ((4, 0.9), (8, 0.8))
                )
                for warps, fused_median in fused_timings:
                    result = {
                        "active_requests": concurrency,
                        "proposal_tokens": width,
                        "fused_warps": warps,
                        "split_gpu_ms": {
                            "minimum": 0.9,
                            "median": 1.0,
                            "p90": 1.1,
                            "maximum": 1.2,
                        },
                        "fused_gpu_ms": {
                            "minimum": fused_median * 0.9,
                            "median": fused_median,
                            "p90": fused_median * 1.1,
                            "maximum": fused_median * 1.2,
                        },
                        "fused_speedup": 1.0 / fused_median,
                        "winner": "fused",
                        "performance_gate_passed": True,
                    }
                    if dtype is not None:
                        result.update(
                            {
                                "candidate_dtype": dtype,
                                "reference_exact": True,
                            }
                        )
                    else:
                        result.update(
                            {
                                "residual_exact": True,
                                "normalized_exact": True,
                            }
                        )
                    results.append(result)
    report = {
        "schema": schema,
        "status": "accepted",
        "repo_id": TOOL.DFLASH2_REPO_ID,
        "revision": TOOL.DFLASH2_REVISION,
        "snapshot": "/models/dflash2",
        "config_sha256": "1" * 64,
        "weight_sha256": "2" * 64,
        "script_sha256": TOOL.source_sha256(
            TOOL.TOOLS_ROOT
            / (
                "tune_dflash2_selector.py"
                if kind == "selector"
                else "tune_dflash2_body_fusion.py"
            )
        ),
        runtime_hash_field: TOOL.source_sha256(TOOL.REFERENCE_ROOT / runtime_name),
        "runtime_profile_sha256": TOOL.source_sha256(
            TOOL.REFERENCE_ROOT / "dflash_tuning_profile.py"
        ),
        "device": "RTX PRO 6000 Blackwell",
        "compute_capability": [12, 0],
        "seed": 53,
        "warmup": 5,
        "iterations": 20,
        "rounds": 7,
        "captured_launches": 16,
        "minimum_fused_speedup": TOOL.DFLASH2_FUSION_MIN_SPEEDUP,
        "results": results,
        "winning_fused_warps": winners,
        "runtime_fused_warps": winners,
        "fused_wins_all_cases": True,
        "runtime_matches_winners": True,
    }
    if kind == "body":
        report["real_weight_validation"] = [
            {
                "weight_case": f"layer-{layer}-{side}",
                "active_requests": concurrency,
                "proposal_tokens": width,
                "query_rows_per_request": width + 1,
                "total_rows": concurrency * (width + 1),
                "fused_warps": 8,
                "residual_exact": True,
                "normalized_exact": True,
            }
            for layer in range(6)
            for side in ("attention", "mlp")
            for concurrency in TOOL.REQUIRED_CONCURRENCIES
            for width in TOOL.DFLASH2_REQUIRED_WIDTHS
        ]
    report["report_sha256"] = hashlib.sha256(TOOL.canonical_json(report)).hexdigest()
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    accepted = TOOL.dflash2_fusion_tuning(path, kind=kind)
    assert accepted["winning_fused_warps"] == winners

    first_key = next(iter(winners))
    report["runtime_fused_warps"] = {
        **winners,
        first_key: 8 if winners[first_key] == 4 else 4,
    }
    report["report_sha256"] = hashlib.sha256(
        TOOL.canonical_json(
            {key: value for key, value in report.items() if key != "report_sha256"}
        )
    ).hexdigest()
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(TOOL.QualificationError, match="serving warp choices"):
        TOOL.dflash2_fusion_tuning(path, kind=kind)
    if kind == "body":
        report["runtime_fused_warps"] = winners
        report["real_weight_validation"][0]["residual_exact"] = False
        report["report_sha256"] = hashlib.sha256(
            TOOL.canonical_json(
                {key: value for key, value in report.items() if key != "report_sha256"}
            )
        ).hexdigest()
        path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(TOOL.QualificationError, match="real-weight validation"):
            TOOL.dflash2_fusion_tuning(path, kind=kind)


def write_concurrency(path: Path, concurrency: int, seed: int) -> None:
    warmups = 2
    repeats = 3
    batches = []
    request_schedule = []
    for batch_index in range(warmups + repeats):
        lanes = []
        requests = []
        for lane in range(concurrency):
            request_sha256 = hashlib.sha256(
                f"{concurrency}:{batch_index}:{lane}".encode()
            ).hexdigest()
            requests.append(request_sha256)
            lanes.append(
                {
                    "lane": lane,
                    "completion_tokens": 11,
                    "first_token_ms": 10.0 + lane,
                    "decode_end_ms": 110.0 + lane,
                    "response_end_ms": 120.0 + lane,
                    "request_sha256": request_sha256,
                    "prompt_nonce": {
                        "marker": chr(0x4E00 + lane + batch_index * concurrency),
                        "first_content_token_id": 1_000
                        + lane
                        + batch_index * concurrency,
                    },
                    "correct": True,
                    "runtime_captures": 0,
                }
            )
        request_schedule.append(requests)
        timed_tokens = 10 * concurrency
        decode_window_ms = 100.0 + concurrency - 1
        response_window_ms = 110.0 + concurrency - 1
        batches.append(
            {
                ("warmup" if batch_index < warmups else "repeat"): (
                    batch_index + 1
                    if batch_index < warmups
                    else batch_index - warmups + 1
                ),
                "cache_state": "token-zero-nonce",
                "nonce_seed": seed,
                "fixture": "code",
                "concurrency": concurrency,
                "timed_tokens": timed_tokens,
                "decode_window_ms": decode_window_ms,
                "aggregate_decode_tps": timed_tokens * 1_000.0 / decode_window_ms,
                "response_window_ms": response_window_ms,
                "aggregate_response_window_tps": timed_tokens
                * 1_000.0
                / response_window_ms,
                "all_correct": True,
                "all_zero_runtime_captures": True,
                "lanes": lanes,
            }
        )
    contract = {
        "model": TOOL.GLM53_MODEL_ID,
        "fixture": "code",
        "prompt": "Write a Python function.",
        "max_tokens": 320,
        "enable_thinking": False,
        "concurrency": concurrency,
        "warmups": warmups,
        "repeats": repeats,
        "cache_state": "token-zero-nonce",
        "nonce_seed": seed,
        "tokenizer_sha256": "a" * 64,
        "request_sha256": request_schedule,
    }
    samples = [batch["aggregate_decode_tps"] for batch in batches[warmups:]]
    response_samples = [
        batch["aggregate_response_window_tps"] for batch in batches[warmups:]
    ]
    summary = {
        "schema": TOOL.CONCURRENCY_SCHEMA,
        "model": TOOL.GLM53_MODEL_ID,
        "fixture": "code",
        "concurrency": concurrency,
        "warmups": warmups,
        "repeats": repeats,
        "cache_state": "token-zero-nonce",
        "nonce_seed": seed,
        "tokenizer_sha256": "a" * 64,
        "request_contract": contract,
        "request_contract_sha256": hashlib.sha256(
            TOOL.canonical_json(contract)
        ).hexdigest(),
        "mean_aggregate_decode_tps": statistics.mean(samples),
        "median_aggregate_decode_tps": statistics.median(samples),
        "min_aggregate_decode_tps": min(samples),
        "max_aggregate_decode_tps": max(samples),
        "stdev_aggregate_decode_tps": statistics.stdev(samples),
        "mean_aggregate_response_window_tps": statistics.mean(response_samples),
        "median_aggregate_response_window_tps": statistics.median(response_samples),
        "min_aggregate_response_window_tps": min(response_samples),
        "max_aggregate_response_window_tps": max(response_samples),
        "stdev_aggregate_response_window_tps": statistics.stdev(response_samples),
        "all_correct": True,
        "all_zero_runtime_captures": True,
        "all_warmups_correct": True,
        "all_warmups_zero_runtime_captures": True,
    }
    path.write_text(
        "".join(
            json.dumps(record) + "\n" for record in [*batches, {"aggregate": summary}]
        ),
        encoding="utf-8",
    )


def test_decode_concurrency_requires_prompt_bound_c1_c2_c4_curves(
    tmp_path: Path,
) -> None:
    paths = []
    for concurrency in TOOL.REQUIRED_CONCURRENCIES:
        path = tmp_path / f"c{concurrency}.jsonl"
        write_concurrency(path, concurrency, 41 + concurrency)
        paths.append(path)

    evidence = TOOL.decode_concurrency(paths)
    assert tuple(evidence["cells"]) == TOOL.REQUIRED_CONCURRENCIES
    assert all(
        cell["mean_aggregate_decode_tps"] > 0.0 for cell in evidence["cells"].values()
    )

    records = [json.loads(line) for line in paths[-1].read_text().splitlines()]
    records[0]["lanes"][0]["request_sha256"] = "0" * 64
    paths[-1].write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(TOOL.QualificationError, match="unbound"):
        TOOL.decode_concurrency(paths)


def write_needle(path: Path) -> None:
    prompt_contracts = []
    measurements = []
    walls = []
    for context in TOOL.REQUIRED_NEEDLE_CONTEXTS:
        for depth in TOOL.REQUIRED_NEEDLE_DEPTHS:
            key = f"N53-{context:08X}-{round(depth * 100):08X}"
            prompt = {
                "target_context_tokens": context,
                "actual_context_tokens": context,
                "target_tolerance_tokens": 8,
                "needle_depth": depth,
                "needle_key": key,
                "filler_tokens_before_needle": round((context - 100) * depth),
                "filler_tokens_after_needle": round((context - 100) * (1.0 - depth)),
                "messages_sha256": hashlib.sha256(
                    f"messages:{context}:{depth}".encode()
                ).hexdigest(),
                "request_sha256": hashlib.sha256(
                    f"request:{context}:{depth}".encode()
                ).hexdigest(),
            }
            prompt_contracts.append(prompt)
            wall = context / 2_000.0 + depth
            walls.append(wall)
            measurements.append(
                {
                    "schema": TOOL.NEEDLE_MEASUREMENT_SCHEMA,
                    "target_context_tokens": context,
                    "prompt_tokens": context,
                    "needle_depth": depth,
                    "needle_key": key,
                    "request_sha256": prompt["request_sha256"],
                    "prompt_contract_sha256": hashlib.sha256(
                        TOOL.canonical_json(prompt)
                    ).hexdigest(),
                    "wall_seconds": wall,
                    "within_request_time_ceiling": True,
                    "prefill_ms": wall * 900.0,
                    "prefill_tps": 2_000.0,
                    "time_to_first_token_ms": wall * 1_000.0,
                    "output_tokens": 4,
                    "decode_ms": 50.0,
                    "finish_reason": "stop",
                    "exact_recall": True,
                    "runtime_captures": 0,
                    "numeric_progression_passed": True,
                    "attention_complete": True,
                    "content_sha256": hashlib.sha256(key.encode()).hexdigest(),
                    "content": key,
                }
            )
    contract = {
        "model": TOOL.GLM53_MODEL_ID,
        "session_id": "needle-unit",
        "tokenizer_sha256": "a" * 64,
        "filler_sha256": "b" * 64,
        "contexts": list(TOOL.REQUIRED_NEEDLE_CONTEXTS),
        "depths": list(TOOL.REQUIRED_NEEDLE_DEPTHS),
        "max_context_tokens": 400_000,
        "max_output_tokens": 32,
        "maximum_request_seconds": TOOL.NEEDLE_MAX_REQUEST_SECONDS,
        "prompts": prompt_contracts,
    }
    contract_sha256 = hashlib.sha256(TOOL.canonical_json(contract)).hexdigest()
    meta = {
        "schema": TOOL.NEEDLE_META_SCHEMA,
        "model": TOOL.GLM53_MODEL_ID,
        "tokenizer_sha256": contract["tokenizer_sha256"],
        "filler_sha256": contract["filler_sha256"],
        "request_contract_sha256": contract_sha256,
        "request_contract": contract,
    }
    summary = {
        "schema": TOOL.NEEDLE_SUMMARY_SCHEMA,
        "status": "accepted",
        "model": TOOL.GLM53_MODEL_ID,
        "request_contract_sha256": contract_sha256,
        "measurements": len(measurements),
        "contexts": list(TOOL.REQUIRED_NEEDLE_CONTEXTS),
        "depths": list(TOOL.REQUIRED_NEEDLE_DEPTHS),
        "maximum_request_seconds": TOOL.NEEDLE_MAX_REQUEST_SECONDS,
        "maximum_measured_wall_seconds": max(walls),
        "median_measured_wall_seconds": statistics.median(walls),
        "all_exact_recall": True,
        "all_within_request_time_ceiling": True,
        "all_numeric_progression_passed": True,
        "all_attention_complete": True,
        "all_zero_runtime_captures": True,
    }
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in [meta, *measurements, summary]),
        encoding="utf-8",
    )


def test_long_context_needle_requires_exact_recall_at_all_depths_to_384k(
    tmp_path: Path,
) -> None:
    path = tmp_path / "needle.jsonl"
    write_needle(path)

    evidence = TOOL.long_context_needle(path)
    assert len(evidence["measurements"]) == 15
    assert evidence["maximum_wall_seconds"] < TOOL.NEEDLE_MAX_REQUEST_SECONDS

    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["exact_recall"] = False
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(TOOL.QualificationError, match="needle row"):
        TOOL.long_context_needle(path)


def test_k4_qualification_requires_the_complete_cache_aware_prefill_matrix() -> None:
    cells = {
        (base, suffix): 1_000.0
        for base in TOOL.REQUIRED_PREFILL_BASE_CONTEXTS
        for suffix in TOOL.REQUIRED_PREFILL_SUFFIX_ROWS
    }
    prompts = [
        {
            "base_context_tokens": base,
            "suffix_tokens": suffix,
            "repeat": repeat,
        }
        for base in TOOL.REQUIRED_PREFILL_BASE_CONTEXTS
        for suffix in TOOL.REQUIRED_PREFILL_SUFFIX_ROWS
        for repeat in TOOL.REQUIRED_PREFILL_REPEATS
    ]
    TOOL.require_release_prefill_grid({"cells": cells, "prompts": prompts})

    del cells[(262_144, 32_768)]
    with pytest.raises(TOOL.QualificationError, match="required 0/32K/64K/128K/256K"):
        TOOL.require_release_prefill_grid({"cells": cells, "prompts": prompts})


def test_adaptive_evidence_reopens_cost_profile_and_fixed_k5_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def identity(name: str) -> dict[str, object]:
        return {
            "schema": name,
            "path": str(tmp_path / name),
            "bytes": 1,
            "sha256": hashlib.sha256(name.encode()).hexdigest(),
        }

    current = identity("adaptive-deployment")
    cost = identity("cost-profile")
    k5_deployment = identity("k5-deployment")
    k5_blended = identity("k5-blended")
    k5_concurrency = [identity(f"k5-c{value}") for value in (1, 2, 4)]
    monkeypatch.setattr(
        TOOL,
        "deployment",
        lambda *_args, **_kwargs: {"identity": current},
    )
    monkeypatch.setattr(
        TOOL,
        "dflash2_cost_profile",
        lambda *_args, **_kwargs: {
            "identity": cost,
            "profile_id": "route-v1",
            "source_sha256": "a" * 64,
            "route_qualified_cells": 9,
            "corpus_samples": 2_479,
        },
    )
    reference = {
        "deployment": {"identity": k5_deployment},
        "blended": {"identity": k5_blended, "wall_decode_tps": 25.0},
        "concurrency": {"identities": k5_concurrency},
    }
    monkeypatch.setattr(TOOL, "dflash2_k5_reference", lambda **_kwargs: reference)
    monkeypatch.setattr(
        TOOL,
        "_agentic_code_decode_tps",
        lambda _mode, _label: 30.0,
    )
    monkeypatch.setattr(
        TOOL,
        "_agentic_concurrency_geomean_tps",
        lambda _mode, _label: 60.0,
    )
    expected = {
        "cost_profile": {
            "profile_id": "route-v1",
            "source_sha256": "a" * 64,
            "route_qualified_cells": 9,
            "corpus_samples": 2_479,
        },
        "reference_width": 5,
        "response_performance_score": math.sqrt(40.0 * 36.0),
        "k5_response_performance_score": math.sqrt(30.0 * 25.0),
        "concurrency_geomean_tps": 70.0,
        "k5_concurrency_geomean_tps": 60.0,
        "weighted_decode_ratio_vs_k5": 36.0 / 25.0,
    }
    report = {
        "evidence": {
            "dflash2_deployment": current,
            "dflash2_cost_profile": cost,
            "dflash2_k5_deployment": k5_deployment,
            "dflash2_k5_blended": k5_blended,
            "dflash2_k5_concurrency": k5_concurrency,
        },
        "results": {
            "modes": {
                "dflash2": {
                    "agentic_code_decode_tps": 40.0,
                    "weighted_decode_tps": 36.0,
                    "agentic_c1_c2_c4_geomean_tps": 70.0,
                }
            },
            "dflash2_adaptive": expected,
        },
    }
    assert TOOL.revalidate_dflash2_adaptive_evidence(report) == expected

    report["evidence"]["dflash2_k5_blended"] = identity("different")
    with pytest.raises(TOOL.QualificationError, match="differs from its files"):
        TOOL.revalidate_dflash2_adaptive_evidence(report)
