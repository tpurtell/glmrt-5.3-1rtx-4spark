from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "bench_release_prefill_matrix.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_release_prefill", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def test_release_curve_covers_cached_context_to_256k_and_suffix_to_32k() -> None:
    assert TOOL.DEFAULT_BASE_CONTEXTS == (0, 32_768, 65_536, 131_072, 262_144)
    assert TOOL.DEFAULT_SUFFIX_ROWS == (
        1_024,
        2_048,
        4_096,
        8_192,
        16_384,
        32_768,
    )


def test_cli_accepts_explicit_candidate_identity_and_frozen_corpus(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    corpus = tmp_path / "corpus"
    tokenizer.write_text("{}\n", encoding="utf-8")
    corpus.mkdir()

    args = TOOL.parse_args(
        [
            "--endpoint",
            "http://candidate/v1/chat/completions",
            "--model",
            "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1",
            "--tokenizer",
            str(tokenizer),
            "--corpus-root",
            str(corpus),
            "--profile",
            "long",
            "--run-id",
            "20260822T105051Z",
            "--base",
            "32768",
            "--suffix",
            "4096",
            "--repeats",
            "3",
        ]
    )

    assert args.model.endswith("EXL3-K3-calibrated-v1")
    assert args.endpoint == "http://candidate/v1/chat/completions"
    assert args.tokenizer == tokenizer
    assert args.corpus_root == corpus
    assert args.profile == "long"
    assert args.run_id == "20260822T105051Z"
    assert args.base == [32768]
    assert args.suffix == [4096]
    assert args.repeats == 3


def test_request_uses_selected_model_and_endpoint(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            event = {
                "choices": [{"delta": {"content": "ok"}}],
                "metrics": {"prefill_ms": 1.0},
            }
            yield f"data: {json.dumps(event)}\n".encode()
            yield b"data: [DONE]\n"

    def urlopen(request, *, timeout):
        observed["url"] = request.full_url
        observed["payload"] = json.loads(request.data)
        observed["timeout"] = timeout
        return Response()

    monkeypatch.setattr(TOOL.urllib.request, "urlopen", urlopen)
    metrics, content = TOOL.request(
        [{"role": "user", "content": "hello"}],
        12.5,
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
            "max_tokens": 1,
        },
        "timeout": 12.5,
    }
    assert metrics["prefill_ms"] == 1.0
    assert metrics["client_wall_ms"] >= 0.0
    assert content == "ok"
