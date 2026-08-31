from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "bench_real_full_mtp_acceptance.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_mtp_acceptance", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_prompt_contract_is_model_independent_and_content_complete() -> None:
    tokenizer_sha256 = "a" * 64
    contract = TOOL.prompt_contract(
        ["code", "multilingual"],
        suite="explicit",
        repeats=5,
        nonce_seed=2026082201,
        tokenizer_sha256=tokenizer_sha256,
        max_tokens=None,
    )
    digest = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()

    assert contract == {
        "suite": "explicit",
        "cases": [
            {
                "id": "code",
                "category": "code",
                "prompt": TOOL.CASES["code"].prompt,
                "max_tokens": TOOL.CASES["code"].max_tokens,
                "weight": 1.0,
                "response_format": None,
            },
            {
                "id": "multilingual",
                "category": "multilingual",
                "prompt": TOOL.CASES["multilingual"].prompt,
                "max_tokens": TOOL.CASES["multilingual"].max_tokens,
                "weight": 1.0,
                "response_format": None,
            },
        ],
        "repeats": 5,
        "nonce_seed": 2026082201,
        "nonce_policy": "token-zero",
        "tokenizer_sha256": tokenizer_sha256,
        "temperature": 0,
        "enable_thinking": False,
        "quality_contract_version": TOOL.QUALITY_CONTRACT_VERSION,
        "request_binding_version": TOOL.REQUEST_BINDING_VERSION,
    }
    assert len(digest) == 64


def test_completion_payload_changes_only_model_for_matched_arms() -> None:
    case = TOOL.CASES["code"]
    prefix = "Qualification nonce 1-0-code. Treat this identifier as irrelevant.\n"
    baseline = json.loads(
        TOOL.completion_payload("baseline/nvfp4", case, prompt_prefix=prefix)
    )
    candidate = json.loads(
        TOOL.completion_payload("candidate/exl3", case, prompt_prefix=prefix)
    )

    assert baseline | {"model": candidate["model"]} == candidate


def test_semantic_contracts_accept_valid_code_json_and_math() -> None:
    code = '''```python
def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping integer intervals."""
    return intervals

assert merge_intervals([]) == []
assert merge_intervals([(1, 2)]) == [(1, 2)]
assert merge_intervals([(1, 2), (2, 3)])
```'''
    structured = json.dumps(
        {
            "path": "src/cache.rs",
            "operation": "replace",
            "line_start": 41,
            "line_end": 47,
            "rationale": "Remove a redundant copy.",
        }
    )

    assert TOOL.validate_case_content("code", code)["quality_contract_passed"]
    assert TOOL.validate_case_content("structured-json", structured)[
        "quality_contract_passed"
    ]
    assert TOOL.validate_case_content(
        "math", "240 × 0.75 = 180; 180 × 1.08 = $194.40."
    )["quality_contract_passed"]


def test_natural_structured_contract_accepts_one_json_fence() -> None:
    result = TOOL.validate_case_content(
        "structured-json",
        """```json
{"path":"src/cache.rs","operation":"replace","line_start":41,"line_end":47,"rationale":"Remove a redundant copy."}
```""",
    )

    assert result["quality_contract_passed"] is True


def test_constrained_structured_contract_requires_bare_json() -> None:
    result = TOOL.validate_case_content(
        "structured-json-schema",
        """```json
{"path":"src/cache.rs","operation":"replace","line_start":41,"line_end":47,"rationale":"Remove a redundant copy."}
```""",
    )

    assert result["quality_contract_passed"] is False
    assert result["quality_contract_issues"]


def test_structured_schema_payload_is_strict_and_prompt_matched() -> None:
    natural = json.loads(
        TOOL.completion_payload("candidate/exl3", TOOL.CASES["structured-json"])
    )
    constrained = json.loads(
        TOOL.completion_payload(
            "candidate/exl3", TOOL.CASES["structured-json-schema"]
        )
    )

    assert natural["messages"] == constrained["messages"]
    assert "response_format" not in natural
    assert constrained["response_format"] == TOOL.structured_edit_response_format()
    assert constrained["response_format"]["json_schema"]["strict"] is True


def test_weighted_suite_splits_the_structured_output_weight() -> None:
    assert len(TOOL.WEIGHTED_CASE_IDS) == 8
    assert TOOL.CASES["structured-json"].weight == 0.5
    assert TOOL.CASES["structured-json-schema"].weight == 0.5
    assert sum(TOOL.CASES[case_id].weight for case_id in TOOL.WEIGHTED_CASE_IDS) == 7


def test_fable_contract_accepts_an_unlabelled_final_moral() -> None:
    story_words = " ".join(["feather"] * 140)
    result = TOOL.validate_case_content(
        "fable",
        f"{story_words}. Sharing credit lets every teammate shine.",
    )

    assert result["quality_contract_passed"] is True


def test_summary_preserves_direct_post_ttft_physical_m_measurements() -> None:
    result = {
        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        "metrics": {
            "decode_ms": 19.0,
            "real_full": {
                "mtp_draft_lengths": [1, 2],
                "mtp_accepted_draft_lengths": [1, 1],
                "mtp_verify_cycle_ms": [9.0, 11.0],
                "target_cycle_physical_m": [1, 3],
                "target_cycle_ms": [8.0, 11.0],
                "mtp_verify_cycles": 2,
                "mtp_draft_tokens": 3,
                "mtp_accepted_draft_tokens": 2,
                "mtp_emitted_tokens_from_verify": 4,
                "mtp_total_verify_cycle_ms": 20.0,
                "mtp_full_match_cycles": 1,
                "request_coordinator_graph_captures": 0,
            },
        },
        "choices": [
            {
                "finish_reason": "length",
                "message": {"content": "240 × 0.75 = 180; 180 × 1.08 = $194.40."},
            }
        ],
    }

    summary = TOOL.summarize_case("math", result)

    assert summary["target_cycle_physical_m"] == [1, 3]
    assert summary["target_cycle_ms"] == [8.0, 11.0]
