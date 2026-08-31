from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "bench_real_full_repeat_decode.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_repeat_decode", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_repeat_prompt_contract_is_model_independent_and_content_bound() -> None:
    contract = TOOL.prompt_contract(
        word="orchid",
        count=100,
        max_tokens=1500,
        warmups=1,
        repeats=5,
        nonce_seed=2026082301,
        tokenizer_sha256="a" * 64,
    )

    assert "model" not in contract
    assert contract == {
        "word": "orchid",
        "requested_repetitions": 100,
        "requested_max_tokens": 1500,
        "warmups": 1,
        "repeats": 5,
        "nonce_seed": 2026082301,
        "temperature": 0,
        "enable_thinking": False,
        "tokenizer_sha256": "a" * 64,
    }
    assert TOOL.canonical_sha256(contract) != TOOL.canonical_sha256(
        contract | {"requested_repetitions": 101}
    )


def test_repeat_measurement_binds_exact_prompt() -> None:
    result = {
        "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        "metrics": {
            "decode_ms": 10.0,
            "real_full": {
                "mtp_verify_cycles": 1,
                "mtp_draft_tokens": 1,
                "mtp_accepted_draft_tokens": 1,
                "request_coordinator_graph_captures": 0,
            },
        },
        "choices": [{"message": {"content": "orchid"}, "finish_reason": "stop"}],
    }

    record = TOOL.summarize(
        result=result,
        prompt="repeat orchid",
        word="orchid",
        count=1,
        max_tokens=10,
        sample=0,
        timed=True,
        nonce={"marker": "x", "first_content_token_id": 1},
    )

    assert record["prompt_sha256"] == TOOL.hashlib.sha256(b"repeat orchid").hexdigest()


def test_packaged_benchmark_uses_content_bound_engine_identity(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("GLMRT_ENGINE_COMMIT", "wip-slot-coordinator-expert")

    assert TOOL.git_commit(tmp_path) == "wip-slot-coordinator-expert"
