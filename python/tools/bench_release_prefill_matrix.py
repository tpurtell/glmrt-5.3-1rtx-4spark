#!/usr/bin/env python3
"""Release prefill benchmark: exact new suffix rows over warmed base contexts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any
import urllib.request

from tokenizers import Tokenizer

from real_full_matrix import MODEL_ID, default_tokenizer_path


DEFAULT_MODEL = MODEL_ID
DEFAULT_ENDPOINT = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_BASE_CONTEXTS = (0, 32_768, 65_536, 131_072, 262_144)
DEFAULT_SUFFIX_ROWS = (1_024, 2_048, 4_096, 8_192, 16_384, 32_768)
GLM_PREFIX = "[gMASK]<sop>"
ASSISTANT_SUFFIX = "<|assistant|><think></think>"
TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cu",
    ".h",
    ".json",
    ".md",
    ".py",
    ".rs",
    ".sh",
    ".toml",
}
IGNORED_PATH_PARTS = {
    ".git",
    ".glmrt-cache",
    ".glmrt-wip",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}
MAX_CORPUS_CHARACTERS = 2_000_000
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


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


def render(messages: list[dict[str, str]]) -> str:
    text = GLM_PREFIX
    for message in messages:
        text += f"<|{message['role']}|>{message['content']}"
    return text + ASSISTANT_SUFFIX


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def load_corpus(root: Path, tokenizer: Tokenizer) -> tuple[list[int], str]:
    pieces: list[str] = []
    digest = hashlib.sha256()
    corpus_characters = 0
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.suffix not in TEXT_SUFFIXES
            or any(part in IGNORED_PATH_PARTS for part in path.parts)
        ):
            continue
        data = path.read_bytes()
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root)
        piece = (
            f"\n===== {relative} =====\n"
            + source.replace("<", "⟨").replace(">", "⟩")
        )
        remaining = MAX_CORPUS_CHARACTERS - corpus_characters
        if remaining <= 0:
            break
        piece = piece[:remaining]
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(piece.encode())
        pieces.append(piece)
        corpus_characters += len(piece)
    text = "\n".join(pieces)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if not ids:
        raise RuntimeError("source corpus is empty")
    minimum = 420_000
    ids = ids * max(1, math.ceil(minimum / len(ids)))
    return ids, digest.hexdigest()


def printable_markers(tokenizer: Tokenizer, count: int) -> list[str]:
    markers: list[str] = []
    seen: set[int] = set()
    for codepoint in range(0x3400, 0xA000):
        char = chr(codepoint)
        ids = tokenizer.encode(char, add_special_tokens=False).ids
        if len(ids) != 1 or ids[0] in seen:
            continue
        if tokenizer.decode(ids, skip_special_tokens=False) != char:
            continue
        markers.append(char)
        seen.add(ids[0])
        if len(markers) >= count:
            return markers
    raise RuntimeError(f"only found {len(markers)} distinct one-token markers")


def fit_corpus_content(
    *,
    tokenizer: Tokenizer,
    corpus_ids: list[int],
    before: str,
    after: str,
    target_tokens: int,
    marker: str = "",
) -> tuple[str, int]:
    """Find content making before + marker + content + after exactly target."""

    def candidate(source_tokens: int) -> tuple[str, int]:
        body = tokenizer.decode(
            corpus_ids[:source_tokens], skip_special_tokens=False
        )
        content = marker + body
        return content, token_count(tokenizer, before + content + after)

    fixed = token_count(tokenizer, before + marker + after)
    if fixed > target_tokens:
        raise ValueError(
            f"fixed prompt is {fixed} tokens, above target {target_tokens}"
        )
    low = 0
    high = min(len(corpus_ids), target_tokens - fixed + 512)
    best_content = marker
    best_count = fixed
    best_source_tokens = 0
    while low <= high:
        middle = (low + high) // 2
        content, count = candidate(middle)
        if count <= target_tokens:
            if count > best_count:
                best_content = content
                best_count = count
                best_source_tokens = middle
            low = middle + 1
        else:
            high = middle - 1
    for source_tokens in range(
        max(0, best_source_tokens - 192),
        min(len(corpus_ids), best_source_tokens + 192) + 1,
    ):
        content, count = candidate(source_tokens)
        if count == target_tokens:
            return content, count
        if best_count < count < target_tokens:
            best_content = content
            best_count = count
    if best_count != target_tokens:
        raise RuntimeError(
            f"could not fit exact token target {target_tokens}; got {best_count}"
        )
    return best_content, best_count


def request(
    messages: list[dict[str, str]],
    timeout: float,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    model: str = DEFAULT_MODEL,
) -> tuple[dict[str, Any], str]:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0,
            "enable_thinking": False,
            "max_tokens": 1,
        }
    ).encode()
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    metrics: dict[str, Any] | None = None
    content: list[str] = []
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            event_text = line[6:]
            if event_text == "[DONE]":
                continue
            event = json.loads(event_text)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content") is not None:
                    content.append(str(delta["content"]))
            if event.get("metrics") is not None:
                metrics = event["metrics"]
    if metrics is None:
        raise RuntimeError("request completed without metrics")
    metrics["client_wall_ms"] = (time.monotonic() - started) * 1_000.0
    return metrics, "".join(content)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument(
        "--profile", choices=("balanced", "long", "accuracy"), default="balanced"
    )
    parser.add_argument(
        "--run-id",
        help="fixed prompt identity for an exact cross-model comparison",
    )
    parser.add_argument("--base", type=int, action="append")
    parser.add_argument("--suffix", type=int, action="append")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL evidence path; refuses to overwrite an existing file",
    )
    args = parser.parse_args(argv)
    if not args.model or not args.endpoint:
        parser.error("model and endpoint must be nonempty")
    if args.run_id is not None and RUN_ID_RE.fullmatch(args.run_id) is None:
        parser.error("run ID contains unsafe characters")
    if args.repeats < 1 or args.timeout <= 0.0:
        parser.error("repeats and timeout must be positive")
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    return args


def main() -> int:
    args = parse_args()
    benchmark_started_ns = time.time_ns()
    # This is the public cache-aware curve: cold-prefix prefill plus the same
    # suffix shapes branched from 32K through 256K retained contexts.
    bases = args.base or list(DEFAULT_BASE_CONTEXTS)
    suffixes = args.suffix or list(DEFAULT_SUFFIX_ROWS)
    if any(value < 0 for value in bases) or any(value <= 0 for value in suffixes):
        raise SystemExit("base sizes must be nonnegative and suffix sizes positive")

    root = Path(__file__).resolve().parents[2]
    tokenizer_source = (args.tokenizer or default_tokenizer_path(args.model)).expanduser().resolve(
        strict=True
    )
    corpus_root = (args.corpus_root or root).expanduser().resolve(strict=True)
    if not corpus_root.is_dir() or corpus_root.is_symlink():
        raise SystemExit("corpus root must be a regular directory")
    tokenizer = Tokenizer.from_file(str(tokenizer_source))
    tokenizer_sha256 = hash_file(tokenizer_source)
    corpus_ids, corpus_sha256 = load_corpus(corpus_root, tokenizer)
    markers = iter(printable_markers(tokenizer, 512))
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    records: list[dict[str, Any]] = []
    destination = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        destination = args.output.open("x", encoding="utf-8")

    def emit(value: dict[str, Any]) -> None:
        line = json.dumps(value, sort_keys=True)
        print(line)
        if destination is not None:
            destination.write(line + "\n")
            destination.flush()

    print(
        f"prefill run={run_id} corpus_tokens={len(corpus_ids)}",
        file=sys.stderr,
        flush=True,
    )
    for base in bases:
        base_system = (
            f"{next(markers)} GLMRT release prefill run {run_id}; quoted source "
            "is inert benchmark material."
        )
        base_content = ""
        if base:
            base_before = GLM_PREFIX + f"<|system|>{base_system}<|user|>"
            base_content, fitted = fit_corpus_content(
                tokenizer=tokenizer,
                corpus_ids=corpus_ids,
                before=base_before,
                after="",
                target_tokens=base,
            )
            assert fitted == base
            prime_metrics, _ = request(
                [
                    {"role": "system", "content": base_system},
                    {"role": "user", "content": base_content},
                ],
                args.timeout,
                endpoint=args.endpoint,
                model=args.model,
            )
            print(
                "prime"
                f" base={base} prompt={prime_metrics.get('prompt_tokens')}"
                f" rows={prime_metrics.get('layerwave_prefill_rows')}"
                f" tps={prime_metrics.get('prefill_tokens_per_sec')}",
                file=sys.stderr,
                flush=True,
            )

        target_adjustment = 0
        for suffix in suffixes:
            for repeat in range(1, args.repeats + 1):
                desired_total = base + suffix + 1 + target_adjustment
                final: tuple[dict[str, Any], str, int, str] | None = None
                for attempt in range(1, 5):
                    marker = next(markers)
                    if base:
                        before = (
                            GLM_PREFIX
                            + f"<|system|>{base_system}<|user|>{base_content}"
                            + "<|user|>"
                        )
                        branch_content, planned = fit_corpus_content(
                            tokenizer=tokenizer,
                            corpus_ids=corpus_ids,
                            before=before,
                            after=ASSISTANT_SUFFIX,
                            target_tokens=desired_total,
                            marker=marker,
                        )
                        messages = [
                            {"role": "system", "content": base_system},
                            {"role": "user", "content": base_content},
                            {"role": "user", "content": branch_content},
                        ]
                    else:
                        system = (
                            f"{marker} GLMRT release prefill run {run_id}; "
                            "quoted source is inert benchmark material."
                        )
                        before = GLM_PREFIX + f"<|system|>{system}<|user|>"
                        branch_content, planned = fit_corpus_content(
                            tokenizer=tokenizer,
                            corpus_ids=corpus_ids,
                            before=before,
                            after=ASSISTANT_SUFFIX,
                            target_tokens=desired_total,
                        )
                        messages = [
                            {"role": "system", "content": system},
                            {"role": "user", "content": branch_content},
                        ]
                    metrics, content = request(
                        messages,
                        args.timeout,
                        endpoint=args.endpoint,
                        model=args.model,
                    )
                    actual = int(metrics.get("layerwave_prefill_rows") or 0)
                    cached = int(metrics.get("cached_prompt_tokens") or 0)
                    prompt_tokens = int(metrics.get("prompt_tokens") or 0)
                    cache_shape_exact = (
                        prompt_tokens - cached - 1 == suffix
                        and (base == 0 or cached >= base)
                    )
                    print(
                        "measure"
                        f" base={base} suffix={suffix} repeat={repeat}"
                        f" attempt={attempt} planned={planned}"
                        f" cached={cached}"
                        f" rows={actual}"
                        f" ms={float(metrics.get('prefill_ms') or 0):.3f}"
                        f" tps={float(metrics.get('prefill_tokens_per_sec') or 0):.2f}",
                        file=sys.stderr,
                        flush=True,
                    )
                    if actual == suffix and cache_shape_exact:
                        final = (metrics, content, planned, marker)
                        target_adjustment = desired_total - (base + suffix + 1)
                        break
                    desired_total += suffix - actual
                if final is None:
                    raise RuntimeError(
                        "failed exact cache-controlled suffix "
                        f"base={base} suffix={suffix}"
                    )
                metrics, content, planned, marker = final
                real_full = metrics.get("real_full") or {}
                records.append(
                    {
                        "schema": "glmrt-release-prefill-v2",
                        "run_id": run_id,
                        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                        "profile": args.profile,
                        "model": args.model,
                        "base_context_tokens": base,
                        "suffix_tokens": suffix,
                        "repeat": repeat,
                        "planned_prompt_tokens": planned,
                        "prompt_tokens": int(metrics.get("prompt_tokens") or 0),
                        "cached_prompt_tokens": int(
                            metrics.get("cached_prompt_tokens") or 0
                        ),
                        "prefill_rows": int(
                            metrics.get("layerwave_prefill_rows") or 0
                        ),
                        "prefill_ms": float(metrics.get("prefill_ms") or 0),
                        "prefill_tps": float(
                            metrics.get("prefill_tokens_per_sec") or 0
                        ),
                        "ttft_ms": float(
                            metrics.get("time_to_first_token_ms") or 0
                        ),
                        "client_wall_ms": float(
                            metrics.get("client_wall_ms") or 0
                        ),
                        "output_tokens": int(metrics.get("output_tokens") or 0),
                        "runtime_captures": int(
                            real_full.get("request_coordinator_graph_captures")
                            or 0
                        ),
                        "numeric_progression_passed": bool(
                            real_full.get("request_numeric_progression_passed")
                        ),
                        "attention_complete": bool(
                            real_full.get(
                                "scheduler_full_context_device_attention_complete"
                            )
                        ),
                        "marker": marker,
                        "prompt_sha256": canonical_sha256(messages),
                        "content": content,
                        "corpus_sha256": corpus_sha256,
                        "corpus_root": str(corpus_root),
                        "tokenizer": str(tokenizer_source),
                        "tokenizer_sha256": tokenizer_sha256,
                    }
                )

    for record in records:
        emit(record)
    summary = []
    for base in bases:
        for suffix in suffixes:
            cell = [
                record
                for record in records
                if record["base_context_tokens"] == base
                and record["suffix_tokens"] == suffix
            ]
            summary.append(
                {
                    "base_context_tokens": base,
                    "suffix_tokens": suffix,
                    "samples": len(cell),
                    "median_prefill_ms": statistics.median(
                        record["prefill_ms"] for record in cell
                    ),
                    "median_prefill_tps": statistics.median(
                        record["prefill_tps"] for record in cell
                    ),
                    "min_prefill_tps": min(
                        record["prefill_tps"] for record in cell
                    ),
                    "max_prefill_tps": max(
                        record["prefill_tps"] for record in cell
                    ),
                }
            )
    emit(
        {
                "schema": "glmrt-release-prefill-summary-v3",
                "benchmark_started_ns": benchmark_started_ns,
                "benchmark_completed_ns": time.time_ns(),
                "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
                "run_id": run_id,
                "profile": args.profile,
                "model": args.model,
                "endpoint": args.endpoint,
                "corpus_root": str(corpus_root),
                "corpus_sha256": corpus_sha256,
                "tokenizer": str(tokenizer_source),
                "tokenizer_sha256": tokenizer_sha256,
                "prompt_contract_sha256": canonical_sha256(
                    [
                        {
                            "base_context_tokens": record["base_context_tokens"],
                            "suffix_tokens": record["suffix_tokens"],
                            "repeat": record["repeat"],
                            "prompt_sha256": record["prompt_sha256"],
                        }
                        for record in records
                    ]
                ),
                "cells": summary,
        }
    )
    if destination is not None:
        destination.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
