#!/usr/bin/env python3
"""Measure cache-controlled suffix-prefill curves at retained prefix depths."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import urllib.request
from typing import Any, Callable

from tokenizers import Tokenizer

from bench_real_full_concurrency import token_zero_nonces
from bench_real_full_long_context_session import load_corpus
from real_full_matrix import (
    MODEL_ID,
    default_tokenizer_path,
    git_commit,
    render_messages,
    repo_root,
)


DEFAULT_PREFIXES = (2_048, 32_768)
DEFAULT_SUFFIX_ROWS = (256, 512, 1_024, 2_048, 4_096, 8_192, 16_384)


def comma_separated_positive_ints(value: str) -> list[int]:
    try:
        values = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must be positive")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument(
        "--prefix-tokens",
        type=comma_separated_positive_ints,
        default=list(DEFAULT_PREFIXES),
    )
    parser.add_argument(
        "--suffix-rows",
        type=comma_separated_positive_ints,
        default=list(DEFAULT_SUFFIX_ROWS),
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--nonce-seed", type=int, required=True)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--endpoint", default="http://127.0.0.1:8000/v1/chat/completions"
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.timeout_seconds <= 0.0:
        parser.error("warmups must be non-negative; repeats and timeout must be positive")
    if args.output.exists() and not args.dry_run:
        parser.error(f"refusing to overwrite output: {args.output}")
    return args


def common_prefix_tokens(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def closest_slice(
    *,
    source_ids: list[int],
    build: Callable[[str], tuple[int, Any]],
    target: int,
    initial_high: int,
) -> tuple[str, int, Any]:
    """Find a decoded source slice whose monotonic token metric is target."""
    high = min(max(initial_high, 1), len(source_ids))
    while True:
        text = source_token_text(source_ids, high)
        metric, extra = build(text)
        if metric >= target or high == len(source_ids):
            break
        high = min(high * 2, len(source_ids))
    low = 0
    best: tuple[int, int, str, Any] | None = None
    while low <= high:
        middle = (low + high) // 2
        text = source_token_text(source_ids, middle)
        metric, extra = build(text)
        candidate = (abs(metric - target), metric, text, extra)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
        if metric < target:
            low = middle + 1
        elif metric > target:
            high = middle - 1
        else:
            return text, metric, extra
    assert best is not None
    return best[2], best[1], best[3]


def source_token_text(source_ids: list[int], count: int) -> str:
    # Assigned by main before planning; keeping this helper avoids repeatedly
    # threading the tokenizer through the binary-search callback signature.
    return _TOKENIZER.decode(source_ids[:count], skip_special_tokens=False)


_TOKENIZER: Tokenizer


def request_completion(
    endpoint: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float,
) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1,
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


def request_record(
    *,
    result: dict[str, Any],
    target_prefix_tokens: int,
    planned_prefix_tokens: int,
    suffix_rows: int,
    expected_prompt_tokens: int,
    repeat: int,
    timed: bool,
    nonce: dict[str, Any],
) -> dict[str, Any]:
    usage = result["usage"]
    metrics = result["metrics"]
    real_full = metrics["real_full"]
    prompt_tokens = int(usage["prompt_tokens"])
    prefill_rows = int(metrics.get("layerwave_prefill_rows") or 0)
    inferred_cached_prefix = max(prompt_tokens - 1 - prefill_rows, 0)
    if prompt_tokens != expected_prompt_tokens:
        raise RuntimeError(
            f"server reported {prompt_tokens} prompt tokens; planned {expected_prompt_tokens}"
        )
    return {
        "record": "measurement",
        "timed": timed,
        "repeat": repeat,
        "target_prefix_tokens": target_prefix_tokens,
        "planned_prefix_tokens": planned_prefix_tokens,
        "requested_suffix_rows": suffix_rows,
        "prompt_tokens": prompt_tokens,
        "prefill_rows": prefill_rows,
        "inferred_cached_prefix_tokens": inferred_cached_prefix,
        "prefill_ms": float(metrics.get("prefill_ms") or 0.0),
        "prefill_tps": float(metrics.get("prefill_tokens_per_sec") or 0.0),
        "time_to_first_token_ms": float(metrics.get("time_to_first_token_ms") or 0.0),
        "prefill_chunks": int(metrics.get("prefill_chunk_count") or 0),
        "runtime_captures": int(
            real_full.get("request_coordinator_graph_captures") or 0
        ),
        "attention_complete": bool(
            real_full.get("scheduler_full_context_device_attention_complete")
        ),
        "numeric_progression_passed": bool(
            real_full.get("request_numeric_progression_passed")
        ),
        "nonce": {
            "marker": nonce["marker"],
            "first_content_token_id": nonce["first_content_token_id"],
        },
    }


def main() -> None:
    global _TOKENIZER
    args = parse_args()
    root = repo_root()
    tokenizer_path = (args.tokenizer or default_tokenizer_path(args.model)).resolve()
    _TOKENIZER = Tokenizer.from_file(str(tokenizer_path))
    corpus, corpus_sha256, source_manifest = load_corpus(args.source)
    minimum_corpus_tokens = max(args.prefix_tokens) + max(args.suffix_rows) + 8_192
    repeated_corpus = corpus
    while len(_TOKENIZER.encode(repeated_corpus, add_special_tokens=False).ids) < minimum_corpus_tokens:
        repeated_corpus += corpus
    corpus_ids = _TOKENIZER.encode(repeated_corpus, add_special_tokens=False).ids
    request_count = len(args.prefix_tokens) * (
        1 + (args.warmups + args.repeats) * len(args.suffix_rows)
    )
    nonces = token_zero_nonces(
        count=request_count,
        seed=args.nonce_seed,
        tokenizer_path=tokenizer_path,
    )
    nonce_offset = 0
    plans: list[dict[str, Any]] = []
    for target_prefix in args.prefix_tokens:
        if target_prefix < 2:
            raise RuntimeError("prefix targets must be at least two tokens")
        # The first suffix publishes the shared second-user role token into the
        # radix tree. Seed one token below the reported context target and make
        # the uncounted suffix warmup populate that stable separator token.
        shared_prefix_target = target_prefix - 1
        prefix_nonce = nonces[nonce_offset]
        nonce_offset += 1
        prefix_header = (
            f"{prefix_nonce['prefix']}Quoted benchmark source follows.\n"
        )
        suffix_probe = nonces[nonce_offset]["prefix"]

        def prefix_metric(source_text: str) -> tuple[int, None]:
            content = prefix_header + source_text + "\n"
            seed_tokens = _TOKENIZER.encode(
                render_messages([{"role": "user", "content": content}]),
                add_special_tokens=False,
            ).ids
            probe_tokens = _TOKENIZER.encode(
                render_messages(
                    [
                        {"role": "user", "content": content},
                        {
                            "role": "user",
                            "content": suffix_probe,
                        }
                    ]
                ),
                add_special_tokens=False,
            ).ids
            return common_prefix_tokens(seed_tokens, probe_tokens), None

        prefix_source, prefix_metric_tokens, _ = closest_slice(
            source_ids=corpus_ids,
            build=prefix_metric,
            target=shared_prefix_target,
            initial_high=target_prefix,
        )
        if prefix_metric_tokens != shared_prefix_target:
            padding_atoms = ("x", " x", ".", " .", "0", " 0", "一", " 一", "\n", " \n")
            padded_candidates = [
                prefix_source + (atom * count)
                for count in range(1, 17)
                for atom in padding_atoms
            ]
            exact = next(
                (
                    candidate
                    for candidate in padded_candidates
                    if prefix_metric(candidate)[0] == shared_prefix_target
                ),
                None,
            )
            if exact is None:
                raise RuntimeError(
                    f"could not construct exact {shared_prefix_target}-token shared prefix; "
                    f"nearest was {prefix_metric_tokens}"
                )
            prefix_source = exact
        prefix_content = prefix_header + prefix_source + "\n"
        planned_probe_prefix_tokens, _ = prefix_metric(prefix_source)
        seed_ids = _TOKENIZER.encode(
            render_messages([{"role": "user", "content": prefix_content}]),
            add_special_tokens=False,
        ).ids
        samples = []
        orders: list[list[int]] = []
        for _ in range(args.warmups):
            orders.append(list(args.suffix_rows))
        for repeat in range(args.repeats):
            order = list(args.suffix_rows)
            if repeat % 3 == 1:
                order.reverse()
            elif repeat % 3 == 2:
                random.Random(args.nonce_seed + target_prefix + repeat).shuffle(order)
            orders.append(order)
        for pass_index, order in enumerate(orders):
            for suffix_rows in order:
                suffix_nonce = nonces[nonce_offset]
                nonce_offset += 1
                suffix_header = suffix_nonce["prefix"]

                def suffix_metric(source_text: str) -> tuple[int, dict[str, Any]]:
                    messages = [
                        {"role": "user", "content": prefix_content},
                        {"role": "user", "content": suffix_header + source_text},
                    ]
                    request_ids = _TOKENIZER.encode(
                        render_messages(messages),
                        add_special_tokens=False,
                    ).ids
                    matched = common_prefix_tokens(seed_ids, request_ids)
                    rows = len(request_ids) - matched - 1
                    return rows, {
                        "messages": messages,
                        "prompt_tokens": len(request_ids),
                        "matched_prefix_tokens": matched,
                    }

                _, planned_rows, extra = closest_slice(
                    source_ids=corpus_ids,
                    build=suffix_metric,
                    target=suffix_rows + 1,
                    initial_high=suffix_rows + 1,
                )
                if planned_rows != suffix_rows + 1:
                    raise RuntimeError(
                        f"could not construct exact {suffix_rows + 1}-row planned "
                        f"suffix; got {planned_rows}"
                    )
                samples.append(
                    {
                        "suffix_rows": suffix_rows,
                        "repeat": pass_index - args.warmups,
                        "timed": pass_index >= args.warmups,
                        "nonce": suffix_nonce,
                        **extra,
                    }
                )
        plans.append(
            {
                "target_prefix_tokens": target_prefix,
                "prefix_content": prefix_content,
                "seed_prompt_tokens": len(seed_ids),
                "planned_probe_prefix_tokens": planned_probe_prefix_tokens,
                "prefix_nonce": prefix_nonce,
                "samples": samples,
            }
        )

    meta = {
        "record": "meta",
        "schema": "glmrt-prefill-curve-v1",
        "commit": git_commit(root),
        "model": args.model,
        "target_prefix_tokens": args.prefix_tokens,
        "requested_suffix_rows": args.suffix_rows,
        "warmups_per_size": args.warmups,
        "repeats_per_size": args.repeats,
        "nonce_seed": args.nonce_seed,
        "tokenizer": str(tokenizer_path),
        "corpus_sha256": corpus_sha256,
        "source_manifest": source_manifest,
        "plans": [
            {
                "target_prefix_tokens": plan["target_prefix_tokens"],
                "seed_prompt_tokens": plan["seed_prompt_tokens"],
                "planned_probe_prefix_tokens": plan["planned_probe_prefix_tokens"],
                "samples": [
                    {
                        key: sample[key]
                        for key in (
                            "suffix_rows",
                            "repeat",
                            "timed",
                            "prompt_tokens",
                            "matched_prefix_tokens",
                        )
                    }
                    for sample in plan["samples"]
                ],
            }
            for plan in plans
        ],
    }
    if args.dry_run:
        print(json.dumps(meta, indent=2, sort_keys=True))
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    measurements: list[dict[str, Any]] = []
    with args.output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(meta, sort_keys=True) + "\n")
        destination.flush()
        for plan in plans:
            seed_result = request_completion(
                args.endpoint,
                args.model,
                [{"role": "user", "content": plan["prefix_content"]}],
                args.timeout_seconds,
            )
            seed_record = {
                "record": "prefix-seed",
                "target_prefix_tokens": plan["target_prefix_tokens"],
                "prompt_tokens": int(seed_result["usage"]["prompt_tokens"]),
                "prefill_rows": int(
                    seed_result["metrics"].get("layerwave_prefill_rows") or 0
                ),
                "prefill_ms": float(seed_result["metrics"].get("prefill_ms") or 0.0),
            }
            destination.write(json.dumps(seed_record, sort_keys=True) + "\n")
            destination.flush()
            print(json.dumps(seed_record, sort_keys=True), flush=True)
            for sample in plan["samples"]:
                result = request_completion(
                    args.endpoint,
                    args.model,
                    sample["messages"],
                    args.timeout_seconds,
                )
                record = request_record(
                    result=result,
                    target_prefix_tokens=plan["target_prefix_tokens"],
                    planned_prefix_tokens=sample["matched_prefix_tokens"],
                    suffix_rows=sample["suffix_rows"],
                    expected_prompt_tokens=sample["prompt_tokens"],
                    repeat=sample["repeat"],
                    timed=sample["timed"],
                    nonce=sample["nonce"],
                )
                destination.write(json.dumps(record, sort_keys=True) + "\n")
                destination.flush()
                print(json.dumps(record, sort_keys=True), flush=True)
                if record["timed"]:
                    if record["prefill_rows"] != record["requested_suffix_rows"]:
                        raise RuntimeError(
                            "cache-controlled suffix shape mismatch: "
                            f"requested {record['requested_suffix_rows']}, "
                            f"observed {record['prefill_rows']}"
                        )
                    measurements.append(record)

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for record in measurements:
            grouped[
                (record["target_prefix_tokens"], record["requested_suffix_rows"])
            ].append(record)
        rows = []
        for (prefix_tokens, suffix_rows), records in sorted(grouped.items()):
            samples = [float(record["prefill_tps"]) for record in records]
            rows.append(
                {
                    "target_prefix_tokens": prefix_tokens,
                    "requested_suffix_rows": suffix_rows,
                    "actual_prefix_tokens": [
                        record["inferred_cached_prefix_tokens"] for record in records
                    ],
                    "mean_prefill_tps": statistics.mean(samples),
                    "median_prefill_tps": statistics.median(samples),
                    "stdev_prefill_tps": (
                        statistics.stdev(samples) if len(samples) > 1 else 0.0
                    ),
                    "min_prefill_tps": min(samples),
                    "max_prefill_tps": max(samples),
                }
            )
        summary = {
            "record": "summary",
            "timed_samples": len(measurements),
            "all_zero_runtime_captures": all(
                record["runtime_captures"] == 0 for record in measurements
            ),
            "all_attention_complete": all(
                record["attention_complete"] for record in measurements
            ),
            "all_numeric_progression_passed": all(
                record["numeric_progression_passed"] for record in measurements
            ),
            "rows": rows,
        }
        destination.write(json.dumps(summary, sort_keys=True) + "\n")
        destination.flush()
        print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
