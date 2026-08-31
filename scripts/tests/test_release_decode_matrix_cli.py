from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "bench_release_decode_matrix.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_release_decode", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_release_curve_covers_context_to_256k_and_three_workloads() -> None:
    assert TOOL.DEFAULT_CONTEXTS == (0, 32_768, 65_536, 131_072, 262_144)
    assert tuple(TOOL.WORKLOADS) == ("code", "writing", "math")


def test_cli_accepts_exact_candidate_identity_and_frozen_corpus(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    corpus = tmp_path / "corpus"
    tokenizer.write_text("{}\n", encoding="utf-8")
    corpus.mkdir()
    output = tmp_path / "decode.jsonl"

    args = TOOL.parse_args(
        [
            "--endpoint",
            "http://candidate/v1/chat/completions",
            "--model",
            "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
            "--tokenizer",
            str(tokenizer),
            "--corpus-root",
            str(corpus),
            "--profile",
            "balanced",
            "--run-id",
            "20260830T000000Z",
            "--context",
            "32768",
            "--workload",
            "code",
            "--repeats",
            "3",
            "--output",
            str(output),
        ]
    )

    assert args.model == "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
    assert args.endpoint == "http://candidate/v1/chat/completions"
    assert args.tokenizer == tokenizer
    assert args.corpus_root == corpus
    assert args.context == [32_768]
    assert args.workload == ["code"]
    assert args.repeats == 3
    assert args.output == output


def test_request_uses_selected_model_endpoint_and_decode_contract(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            event = {
                "choices": [
                    {
                        "delta": {"content": "ok", "reasoning_content": "hidden"},
                        "finish_reason": "stop",
                    }
                ],
                "metrics": {"decode_ms": 1.0},
            }
            yield f"data: {json.dumps(event)}\n".encode()
            yield b"data: [DONE]\n"

    def urlopen(request, *, timeout):
        observed["url"] = request.full_url
        observed["payload"] = json.loads(request.data)
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(TOOL.urllib.request, "urlopen", urlopen)
    metrics, content = TOOL.semantic_request(
        [{"role": "user", "content": "hello"}],
        12.5,
        192,
        endpoint="http://candidate/v1/chat/completions",
        model="candidate/exl3",
    )

    assert observed == {
        "url": "http://candidate/v1/chat/completions",
        "payload": {
            "model": "candidate/exl3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": 192,
        },
        "timeout": 12.5,
    }
    assert metrics["decode_ms"] == 1.0
    assert metrics["_reasoning"] == "hidden"
    assert metrics["_finish_reason"] == "stop"
    assert content == "ok"


def record(context: int, workload: str, repeat: int) -> dict[str, object]:
    cached = context if context else 0
    return {
        "context_bucket_tokens": context,
        "workload": workload,
        "repeat": repeat,
        "prompt_tokens": cached + 10,
        "cached_prompt_tokens": cached,
        "prefill_rows": 9,
        "output_tokens": 101,
        "decode_ms": 2_000.0,
        "decode_tps": 50.0,
        "draft_tokens": 100,
        "accepted_draft_tokens": 75,
        "runtime_captures": 0,
        "numeric_progression_passed": True,
        "attention_complete": True,
    }


def test_summary_requires_complete_unique_cache_valid_matrix() -> None:
    records = [
        record(context, workload, repeat)
        for context in (0, 32_768)
        for workload in ("code", "math")
        for repeat in (1, 2)
    ]
    cells = TOOL.summarize_records(
        records,
        contexts=[0, 32_768],
        workloads=["code", "math"],
        repeats=2,
    )
    assert len(cells) == 4
    assert all(cell["samples"] == 2 for cell in cells)
    assert all(cell["decode_tps"] == 50.0 for cell in cells)
    assert all(cell["accepted_draft_rate"] == 0.75 for cell in cells)

    with pytest.raises(RuntimeError, match="incomplete"):
        TOOL.summarize_records(
            records[:-1],
            contexts=[0, 32_768],
            workloads=["code", "math"],
            repeats=2,
        )
    with pytest.raises(RuntimeError, match="duplicate"):
        TOOL.summarize_records(
            records + [records[-1]],
            contexts=[0, 32_768],
            workloads=["code", "math"],
            repeats=2,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("cached_prompt_tokens", 32_767, "retain"),
        ("prefill_rows", 8, "reconcile"),
        ("numeric_progression_passed", False, "numeric"),
        ("attention_complete", False, "attention"),
        ("runtime_captures", 1, "graph capture"),
    ),
)
def test_record_rejects_invalid_runtime_evidence(
    field: str, value: object, message: str
) -> None:
    sample = record(32_768, "code", 1)
    sample[field] = value
    if field == "cached_prompt_tokens":
        # Preserve internally consistent row accounting so this reaches the
        # stronger retained-base-context gate.
        sample["prompt_tokens"] = int(value) + 10
    with pytest.raises(RuntimeError, match=message):
        TOOL.validate_record(sample)
