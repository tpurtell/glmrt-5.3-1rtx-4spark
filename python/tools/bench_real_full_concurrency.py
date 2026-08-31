#!/usr/bin/env python3
"""Measure concurrent real-full decode makespan on deterministic code or exact text."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from real_full_matrix import MODEL_ID, default_tokenizer_path


@dataclass(frozen=True)
class Fixture:
    prompt: str
    max_tokens: int
    expected: str | None = None


FIXTURES = {
    "code": Fixture(
        "Write a Python function merge_intervals(intervals) that merges overlapping "
        "integer intervals. Include type hints, a short docstring, and three assert-based "
        "examples. Return only one Python code block.",
        320,
    ),
    "exact-50": Fixture(
        "Background words below are irrelevant. "
        + "beta " * 946
        + "\nCount from 1 to 50, one number per line. Do not add any other text.",
        99,
        "\n".join(str(value) for value in range(1, 51)),
    ),
}


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def concurrency_contract(
    *,
    model: str,
    fixture_name: str,
    concurrency: int,
    warmups: int,
    repeats: int,
    cache_state: str,
    nonce_seed: int,
    tokenizer_sha256: str | None,
    batches: list[dict[str, Any]],
) -> dict[str, Any]:
    fixture = FIXTURES[fixture_name]
    return {
        "model": model,
        "fixture": fixture_name,
        "prompt": fixture.prompt,
        "max_tokens": fixture.max_tokens,
        "enable_thinking": False,
        "concurrency": concurrency,
        "warmups": warmups,
        "repeats": repeats,
        "cache_state": cache_state,
        "nonce_seed": nonce_seed,
        "tokenizer_sha256": tokenizer_sha256,
        "request_sha256": [
            [lane["request_sha256"] for lane in batch["lanes"]]
            for batch in batches
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", choices=FIXTURES)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--warmups",
        type=int,
        default=2,
        help="untimed batches at the requested concurrency before sampling; default: 2",
    )
    parser.add_argument(
        "--cache-state",
        choices=("token-zero-nonce", "exact-repeat"),
        default="token-zero-nonce",
        help=(
            "use a distinct tokenizer-verified first content token for every "
            "request (default), or intentionally exercise exact-prefix reuse"
        ),
    )
    parser.add_argument(
        "--nonce-seed",
        type=int,
        default=time.time_ns(),
        help="reproducible token-zero nonce-bank rotation",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help=(
            "tokenizer.json used to verify nonce tokens; defaults to the local "
            "Hugging Face snapshot for --model"
        ),
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL evidence path; refuses to overwrite an existing file",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    return args


def payload(model: str, fixture: Fixture, prompt_prefix: str = "") -> bytes:
    return json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "user", "content": f"{prompt_prefix}{fixture.prompt}"}
            ],
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": fixture.max_tokens,
        }
    ).encode()


def token_zero_nonces(
    *,
    count: int,
    seed: int,
    tokenizer_path: Path,
) -> list[dict[str, Any]]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SystemExit(
            "token-zero nonce mode requires the `tokenizers` package"
        ) from error

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    first_codepoint = 0x4E00
    last_codepoint = 0x9FFF
    candidate_count = last_codepoint - first_codepoint + 1
    start = seed % candidate_count
    nonces = []
    seen_token_ids = set()
    for offset in range(candidate_count):
        codepoint = first_codepoint + ((start + offset) % candidate_count)
        marker = chr(codepoint)
        prefix = f"{marker} request nonce {seed}-{len(nonces)}.\n"
        encoded = tokenizer.encode(prefix, add_special_tokens=False).ids
        if not encoded or encoded[0] in seen_token_ids:
            continue
        marker_ids = tokenizer.encode(marker, add_special_tokens=False).ids
        if len(marker_ids) != 1 or encoded[0] != marker_ids[0]:
            continue
        seen_token_ids.add(encoded[0])
        nonces.append(
            {
                "prefix": prefix,
                "marker": marker,
                "first_content_token_id": encoded[0],
            }
        )
        if len(nonces) == count:
            return nonces
    raise SystemExit(
        f"tokenizer exposed only {len(nonces)} suitable unique nonce tokens; "
        f"{count} are required"
    )


def judge_code(content: str) -> tuple[bool, str]:
    match = re.fullmatch(r"\s*```python\s*\n(.*?)\n```\s*", content, re.DOTALL)
    if match is None:
        return False, "not-one-python-code-block"
    try:
        tree = ast.parse(match.group(1))
    except SyntaxError:
        return False, "python-syntax-error"
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "merge_intervals"
    ]
    if len(functions) != 1:
        return False, "missing-unique-merge-intervals"
    function = functions[0]
    if (
        function.returns is None
        or not function.args.args
        or function.args.args[0].annotation is None
    ):
        return False, "missing-type-hints"
    if ast.get_docstring(function) is None:
        return False, "missing-docstring"
    if sum(isinstance(node, ast.Assert) for node in tree.body) < 3:
        return False, "missing-three-top-level-asserts"
    return True, "static-code-contract-passed"


def request_one(
    *,
    lane: int,
    url: str,
    request_payload: bytes,
    timeout: float,
    barrier: threading.Barrier,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=request_payload, headers={"Content-Type": "application/json"}
    )
    barrier.wait()
    request_start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    response_end = time.perf_counter()
    return {
        "lane": lane,
        "request_start": request_start,
        "response_end": response_end,
        "result": result,
    }


def summarize_lane(
    raw: dict[str, Any], fixture_name: str, fixture: Fixture, batch_start: float
) -> dict[str, Any]:
    result = raw["result"]
    usage = result["usage"]
    metrics = result["metrics"]
    real_full = metrics["real_full"]
    content = result["choices"][0]["message"]["content"]
    if fixture.expected is not None:
        correct = content == fixture.expected
        verdict = "exact" if correct else "exact-mismatch"
    else:
        correct, verdict = judge_code(content)
    request_start = float(raw["request_start"])
    response_end = float(raw["response_end"])
    ttft_ms = float(metrics["time_to_first_token_ms"])
    decode_ms = float(metrics["decode_ms"])
    first_token = request_start + ttft_ms / 1_000.0
    decode_end = first_token + decode_ms / 1_000.0
    completion_tokens = int(usage["completion_tokens"])
    return {
        "fixture": fixture_name,
        "lane": raw["lane"],
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": result["choices"][0]["finish_reason"],
        "request_start_ms": (request_start - batch_start) * 1_000.0,
        "first_token_ms": (first_token - batch_start) * 1_000.0,
        "decode_end_ms": (decode_end - batch_start) * 1_000.0,
        "response_end_ms": (response_end - batch_start) * 1_000.0,
        "ttft_ms": ttft_ms,
        "prefill_ms": float(metrics["prefill_ms"]),
        "decode_ms": decode_ms,
        "decode_tps": (
            (completion_tokens - 1) / (decode_ms / 1_000.0)
            if completion_tokens > 1 and decode_ms > 0.0
            else 0.0
        ),
        "runtime_captures": int(real_full["request_coordinator_graph_captures"]),
        "verify_cycles": int(real_full["mtp_verify_cycles"]),
        "draft_tokens": int(real_full["mtp_draft_tokens"]),
        "accepted_draft_tokens": int(real_full["mtp_accepted_draft_tokens"]),
        "draft_lengths": [
            int(value) for value in real_full.get("mtp_draft_lengths") or []
        ],
        "accepted_draft_lengths": [
            int(value)
            for value in real_full.get("mtp_accepted_draft_lengths") or []
        ],
        "verify_cycle_ms": [
            float(value) for value in real_full.get("mtp_verify_cycle_ms") or []
        ],
        "correct": correct,
        "verdict": verdict,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_preview": content[:160].replace("\n", "\\n"),
    }


def execute_batch(
    *,
    concurrency: int,
    fixture_name: str,
    fixture: Fixture,
    url: str,
    request_payloads: list[bytes],
    prompt_nonces: list[dict[str, Any] | None],
    timeout: float,
) -> dict[str, Any]:
    barrier = threading.Barrier(concurrency)
    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                request_one,
                lane=lane,
                url=url,
                request_payload=request_payloads[lane],
                timeout=timeout,
                barrier=barrier,
            )
            for lane in range(concurrency)
        ]
        raw_lanes = [future.result() for future in futures]
    lanes = [
        summarize_lane(raw, fixture_name, fixture, batch_start) for raw in raw_lanes
    ]
    for lane in lanes:
        nonce = prompt_nonces[lane["lane"]]
        lane["request_sha256"] = hashlib.sha256(
            request_payloads[lane["lane"]]
        ).hexdigest()
        lane["prompt_nonce"] = (
            None
            if nonce is None
            else {
                "marker": nonce["marker"],
                "first_content_token_id": nonce["first_content_token_id"],
            }
        )
    decode_window_start_ms = min(lane["first_token_ms"] for lane in lanes)
    decode_window_end_ms = max(lane["decode_end_ms"] for lane in lanes)
    response_window_end_ms = max(lane["response_end_ms"] for lane in lanes)
    timed_tokens = sum(lane["completion_tokens"] - 1 for lane in lanes)
    decode_window_ms = decode_window_end_ms - decode_window_start_ms
    response_window_ms = response_window_end_ms - decode_window_start_ms
    return {
        "fixture": fixture_name,
        "concurrency": concurrency,
        "timed_tokens": timed_tokens,
        "decode_window_ms": decode_window_ms,
        "aggregate_decode_tps": (
            timed_tokens / (decode_window_ms / 1_000.0)
            if decode_window_ms > 0.0
            else 0.0
        ),
        "response_window_ms": response_window_ms,
        "aggregate_response_window_tps": (
            timed_tokens / (response_window_ms / 1_000.0)
            if response_window_ms > 0.0
            else 0.0
        ),
        "all_correct": all(lane["correct"] for lane in lanes),
        "all_zero_runtime_captures": all(
            lane["runtime_captures"] == 0 for lane in lanes
        ),
        "lanes": lanes,
    }


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    fixture = FIXTURES[args.fixture]
    request_count = (args.warmups + args.repeats) * args.concurrency
    if args.cache_state == "token-zero-nonce":
        tokenizer_path = args.tokenizer or default_tokenizer_path(args.model)
        nonce_bank: list[dict[str, Any] | None] = token_zero_nonces(
            count=request_count,
            seed=args.nonce_seed,
            tokenizer_path=tokenizer_path,
        )
    else:
        tokenizer_path = None
        nonce_bank = [None] * request_count
    nonce_offset = 0

    def next_batch_inputs() -> tuple[list[bytes], list[dict[str, Any] | None]]:
        nonlocal nonce_offset
        prompt_nonces = nonce_bank[nonce_offset : nonce_offset + args.concurrency]
        nonce_offset += args.concurrency
        return (
            [
                payload(
                    args.model,
                    fixture,
                    "" if nonce is None else str(nonce["prefix"]),
                )
                for nonce in prompt_nonces
            ],
            prompt_nonces,
        )

    destination = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        destination = args.output.open("x", encoding="utf-8")

    def emit(value: dict[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False)
        print(line, flush=True)
        if destination is not None:
            destination.write(line + "\n")
            destination.flush()

    warmups = []
    for warmup in range(1, args.warmups + 1):
        request_payloads, prompt_nonces = next_batch_inputs()
        batch = execute_batch(
            concurrency=args.concurrency,
            fixture_name=args.fixture,
            fixture=fixture,
            url=args.url,
            request_payloads=request_payloads,
            prompt_nonces=prompt_nonces,
            timeout=args.timeout,
        )
        batch = {
            "warmup": warmup,
            "cache_state": args.cache_state,
            "nonce_seed": args.nonce_seed,
            **batch,
        }
        warmups.append(batch)
        emit(batch)

    batches = []
    for repeat in range(1, args.repeats + 1):
        request_payloads, prompt_nonces = next_batch_inputs()
        batch = execute_batch(
            concurrency=args.concurrency,
            fixture_name=args.fixture,
            fixture=fixture,
            url=args.url,
            request_payloads=request_payloads,
            prompt_nonces=prompt_nonces,
            timeout=args.timeout,
        )
        batch = {
            "repeat": repeat,
            "cache_state": args.cache_state,
            "nonce_seed": args.nonce_seed,
            **batch,
        }
        batches.append(batch)
        emit(batch)

    samples = [batch["aggregate_decode_tps"] for batch in batches]
    response_window_samples = [
        batch["aggregate_response_window_tps"] for batch in batches
    ]
    tokenizer_sha256 = (
        None if tokenizer_path is None else hash_file(Path(tokenizer_path).resolve())
    )
    contract = concurrency_contract(
        model=args.model,
        fixture_name=args.fixture,
        concurrency=args.concurrency,
        warmups=args.warmups,
        repeats=args.repeats,
        cache_state=args.cache_state,
        nonce_seed=args.nonce_seed,
        tokenizer_sha256=tokenizer_sha256,
        batches=[*warmups, *batches],
    )
    summary = {
        "schema": "glmrt-decode-concurrency-summary-v1",
        "model": args.model,
        "fixture": args.fixture,
        "concurrency": args.concurrency,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "cache_state": args.cache_state,
        "nonce_seed": args.nonce_seed,
        "tokenizer": None if tokenizer_path is None else str(tokenizer_path),
        "tokenizer_sha256": tokenizer_sha256,
        "request_contract_sha256": canonical_sha256(contract),
        "request_contract": contract,
        "mean_aggregate_decode_tps": statistics.mean(samples),
        "median_aggregate_decode_tps": statistics.median(samples),
        "min_aggregate_decode_tps": min(samples),
        "max_aggregate_decode_tps": max(samples),
        "stdev_aggregate_decode_tps": (
            statistics.stdev(samples) if len(samples) > 1 else 0.0
        ),
        "mean_aggregate_response_window_tps": statistics.mean(
            response_window_samples
        ),
        "median_aggregate_response_window_tps": statistics.median(
            response_window_samples
        ),
        "min_aggregate_response_window_tps": min(response_window_samples),
        "max_aggregate_response_window_tps": max(response_window_samples),
        "stdev_aggregate_response_window_tps": (
            statistics.stdev(response_window_samples)
            if len(response_window_samples) > 1
            else 0.0
        ),
        "all_correct": all(batch["all_correct"] for batch in batches),
        "all_zero_runtime_captures": all(
            batch["all_zero_runtime_captures"] for batch in batches
        ),
        "all_warmups_correct": all(batch["all_correct"] for batch in warmups),
        "all_warmups_zero_runtime_captures": all(
            batch["all_zero_runtime_captures"] for batch in warmups
        ),
    }
    emit({"aggregate": summary})
    if destination is not None:
        destination.close()
    if not all(
        (
            summary["all_correct"],
            summary["all_zero_runtime_captures"],
            summary["all_warmups_correct"],
            summary["all_warmups_zero_runtime_captures"],
        )
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
