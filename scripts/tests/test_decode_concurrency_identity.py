from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))

import bench_real_full_concurrency as TOOL  # noqa: E402


def test_token_zero_nonce_uses_a_real_line_break(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTokenizer:
        @classmethod
        def from_file(cls, _path: str) -> "FakeTokenizer":
            return cls()

        def encode(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            assert add_special_tokens is False
            return SimpleNamespace(ids=[ord(text[0])])

    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=FakeTokenizer))
    nonces = TOOL.token_zero_nonces(
        count=2,
        seed=29,
        tokenizer_path=tmp_path / "tokenizer.json",
    )

    assert all(nonce["prefix"].endswith(".\n") for nonce in nonces)
    assert all("\\n" not in nonce["prefix"] for nonce in nonces)


def batch(*digests: str) -> dict:
    return {
        "lanes": [
            {"lane": lane, "request_sha256": digest}
            for lane, digest in enumerate(digests)
        ]
    }


def test_concurrency_contract_binds_every_request_in_schedule() -> None:
    contract = TOOL.concurrency_contract(
        model="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
        fixture_name="code",
        concurrency=2,
        warmups=1,
        repeats=2,
        cache_state="token-zero-nonce",
        nonce_seed=29,
        tokenizer_sha256="a" * 64,
        batches=[batch("1" * 64, "2" * 64), batch("3" * 64, "4" * 64), batch("5" * 64, "6" * 64)],
    )

    assert contract["request_sha256"] == [
        ["1" * 64, "2" * 64],
        ["3" * 64, "4" * 64],
        ["5" * 64, "6" * 64],
    ]
    assert contract["enable_thinking"] is False
    digest = TOOL.canonical_sha256(contract)
    contract["request_sha256"][2][1] = "0" * 64
    assert TOOL.canonical_sha256(contract) != digest


def test_concurrency_payload_explicitly_disables_thinking() -> None:
    request = json.loads(
        TOOL.payload(
            "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
            TOOL.FIXTURES["code"],
        )
    )

    assert request["enable_thinking"] is False
