#!/usr/bin/env python3
"""Measure a cache-busted low-entropy word-repetition decode workload."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import time
import urllib.request
from typing import Any

from bench_real_full_concurrency import token_zero_nonces
from real_full_matrix import MODEL_ID, default_tokenizer_path, git_commit, repo_root


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def prompt_contract(
    *,
    word: str,
    count: int,
    max_tokens: int,
    warmups: int,
    repeats: int,
    nonce_seed: int,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    return {
        "word": word,
        "requested_repetitions": count,
        "requested_max_tokens": max_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "nonce_seed": nonce_seed,
        "temperature": 0,
        "enable_thinking": False,
        "tokenizer_sha256": tokenizer_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--word", default="orchid")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nonce-seed", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.count, args.max_tokens, args.repeats) < 1:
        parser.error("count, max tokens, and repeats must be positive")
    if args.warmups < 0 or args.timeout_seconds <= 0.0:
        parser.error("warmups must be non-negative and timeout must be positive")
    if not args.word.strip():
        parser.error("word must not be empty")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    return args


def request_completion(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.load(response)


def summarize(
    *,
    result: dict[str, Any],
    prompt: str,
    word: str,
    count: int,
    max_tokens: int,
    sample: int,
    timed: bool,
    nonce: dict[str, Any],
) -> dict[str, Any]:
    usage = result["usage"]
    metrics = result["metrics"]
    real_full = metrics["real_full"]
    content = result["choices"][0]["message"]["content"]
    completion_tokens = int(usage["completion_tokens"])
    decode_ms = float(metrics["decode_ms"])
    occurrences = len(re.findall(rf"\b{re.escape(word)}\b", content, re.IGNORECASE))
    return {
        "record": "measurement",
        "sample": sample,
        "timed": timed,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "word": word,
        "requested_repetitions": count,
        "observed_word_occurrences": occurrences,
        "exact_repetition_count": occurrences == count,
        "requested_max_tokens": max_tokens,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": result["choices"][0]["finish_reason"],
        "prefill_ms": float(metrics.get("prefill_ms") or 0.0),
        "decode_ms": decode_ms,
        "decode_tps": (
            (completion_tokens - 1) * 1_000.0 / decode_ms
            if completion_tokens > 1 and decode_ms > 0.0
            else 0.0
        ),
        "verify_cycles": int(real_full.get("mtp_verify_cycles") or 0),
        "draft_tokens": int(real_full.get("mtp_draft_tokens") or 0),
        "accepted_draft_tokens": int(
            real_full.get("mtp_accepted_draft_tokens") or 0
        ),
        "runtime_captures": int(
            real_full.get("request_coordinator_graph_captures") or 0
        ),
        "nonce": {
            "marker": nonce["marker"],
            "first_content_token_id": nonce["first_content_token_id"],
        },
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_preview": content[:160].replace("\n", "\\n"),
    }


def main() -> None:
    args = parse_args()
    tokenizer_path = (args.tokenizer or default_tokenizer_path(args.model)).resolve()
    tokenizer_sha256 = hash_file(tokenizer_path)
    nonces = token_zero_nonces(
        count=args.warmups + args.repeats,
        seed=args.nonce_seed,
        tokenizer_path=tokenizer_path,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []
    with args.output.open("x", encoding="utf-8") as destination:
        meta = {
            "record": "meta",
            "schema": "glmrt-repeat-decode-v2",
            "commit": git_commit(repo_root()),
            "model": args.model,
            "word": args.word,
            "requested_repetitions": args.count,
            "requested_max_tokens": args.max_tokens,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "nonce_seed": args.nonce_seed,
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_sha256,
        }
        meta["prompt_contract_sha256"] = canonical_sha256(
            prompt_contract(
                word=args.word,
                count=args.count,
                max_tokens=args.max_tokens,
                warmups=args.warmups,
                repeats=args.repeats,
                nonce_seed=args.nonce_seed,
                tokenizer_sha256=tokenizer_sha256,
            )
        )
        destination.write(json.dumps(meta, sort_keys=True) + "\n")
        destination.flush()
        for sample, nonce in enumerate(nonces):
            prompt = (
                f"{args.word} {nonce['prefix'].strip()}\n"
                f'Repeat only the single word "{args.word}" exactly {args.count} '
                "times, separated by spaces. Do not repeat the nonce or add "
                "any other text."
            )
            result = request_completion(
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
                args.timeout_seconds,
            )
            record = summarize(
                result=result,
                prompt=prompt,
                word=args.word,
                count=args.count,
                max_tokens=args.max_tokens,
                sample=sample,
                timed=sample >= args.warmups,
                nonce=nonce,
            )
            destination.write(json.dumps(record, sort_keys=True) + "\n")
            destination.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
            if record["timed"]:
                measurements.append(record)

        tps = [float(record["decode_tps"]) for record in measurements]
        total_tokens = sum(int(record["completion_tokens"]) - 1 for record in measurements)
        total_decode_ms = sum(float(record["decode_ms"]) for record in measurements)
        summary = {
            "record": "summary",
            "timed_samples": len(measurements),
            "aggregate_decode_tps": (
                total_tokens * 1_000.0 / total_decode_ms
                if total_tokens > 0 and total_decode_ms > 0.0
                else 0.0
            ),
            "mean_decode_tps": statistics.mean(tps),
            "median_decode_tps": statistics.median(tps),
            "stdev_decode_tps": statistics.stdev(tps) if len(tps) > 1 else 0.0,
            "min_decode_tps": min(tps),
            "max_decode_tps": max(tps),
            "requested_completion_tokens": args.max_tokens,
            "actual_completion_tokens": [
                int(record["completion_tokens"]) for record in measurements
            ],
            "observed_word_occurrences": [
                int(record["observed_word_occurrences"]) for record in measurements
            ],
            "all_exact_repetition_count": all(
                bool(record["exact_repetition_count"]) for record in measurements
            ),
            "all_zero_runtime_captures": all(
                int(record["runtime_captures"]) == 0 for record in measurements
            ),
        }
        destination.write(json.dumps(summary, sort_keys=True) + "\n")
        destination.flush()
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
