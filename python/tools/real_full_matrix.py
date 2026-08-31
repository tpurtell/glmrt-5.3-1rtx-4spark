#!/usr/bin/env python3
"""Run concise, reproducible base/input/output serving-envelope cases."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import TYPE_CHECKING
import urllib.request

if TYPE_CHECKING:
    from tokenizers import Tokenizer


MODEL_ID = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
GLM_USER_PREFIX = "[gMASK]<sop><|user|>"
GLM_ASSISTANT_SUFFIX = "<|assistant|><think></think>"
DEFAULT_MAX_CONTEXT_TOKENS = 128 * 1024


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_tokenizer_path(model_id: str = MODEL_ID) -> Path:
    if not model_id or "/" not in model_id:
        raise ValueError(f"model ID is not a Hugging Face repository ID: {model_id!r}")
    cache_root = Path(
        os.environ.get(
            "HF_HOME",
            Path.home() / ".cache" / "huggingface",
        )
    )
    cache_name = "models--" + model_id.replace("/", "--")
    model_cache = cache_root / "hub" / cache_name
    main_ref = model_cache / "refs" / "main"
    if main_ref.exists():
        revision = main_ref.read_text(encoding="utf-8").strip()
        if not revision:
            raise ValueError(f"empty Hugging Face main ref: {main_ref}")
        tokenizer = model_cache / "snapshots" / revision / "tokenizer.json"
        if not tokenizer.is_file():
            raise FileNotFoundError(
                f"{model_id} main ref {revision!r} has no tokenizer.json at {tokenizer}"
            )
        return tokenizer

    candidates = sorted(
        (model_cache / "snapshots").glob("*/tokenizer.json")
    )
    if not candidates:
        raise FileNotFoundError(
            f"no local {model_id} tokenizer.json below {cache_root / 'hub' / cache_name}; "
            "pass --tokenizer"
        )
    return candidates[-1]


def parse_case(value: str) -> tuple[int, int, str]:
    fields = value.split(":", 2)
    if len(fields) not in (2, 3):
        raise argparse.ArgumentTypeError(
            "case must be INPUT_TOKENS:MAX_OUTPUT_TOKENS[:LABEL]"
        )
    try:
        input_tokens = int(fields[0])
        output_tokens = int(fields[1])
    except ValueError as error:
        raise argparse.ArgumentTypeError("case token counts must be integers") from error
    if input_tokens < 1 or output_tokens < 1:
        raise argparse.ArgumentTypeError("case token counts must be positive")
    label = fields[2] if len(fields) == 3 else f"in{input_tokens}-out{output_tokens}"
    if not label:
        raise argparse.ArgumentTypeError("case label must not be empty")
    return input_tokens, output_tokens, label


def load_corpus(
    paths: list[Path],
    *,
    source_labels: list[str] | None = None,
) -> tuple[str, str]:
    if source_labels is not None:
        if len(source_labels) != len(paths):
            raise ValueError("source_labels length must match paths")
        if (
            any(not label or "\0" in label for label in source_labels)
            or len(set(source_labels)) != len(source_labels)
        ):
            raise ValueError("source_labels must be unique, nonempty strings")
    sections: list[str] = []
    digest = hashlib.sha256()
    for index, path in enumerate(paths):
        resolved = path.resolve()
        data = resolved.read_bytes()
        label = (
            str(resolved)
            if source_labels is None
            else source_labels[index]
        )
        digest.update(label.encode())
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
        sections.append(f"\n\n===== {label} =====\n\n{data.decode('utf-8')}")
    return "".join(sections), digest.hexdigest()


def load_snapshot(
    root: Path,
    tokenizer: Tokenizer,
) -> tuple[list[int], str, dict]:
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "glmrt-kv-dsa-v2":
        raise ValueError(
            f"{metadata_path} is not a glmrt-kv-dsa-v2 snapshot; "
            "regenerate legacy snapshots with the current runtime"
        )
    producer_profile = metadata.get("producer_profile")
    if not isinstance(producer_profile, dict) or not isinstance(
        producer_profile.get("cache_semantics_revision"), int
    ):
        raise ValueError(
            f"{metadata_path} has no execution-profile fingerprint; "
            "regenerate it with the current runtime"
        )
    token_count = metadata.get("token_count")
    token_file = root / str(metadata.get("token_ids_file"))
    payload = token_file.read_bytes()
    if (
        not isinstance(token_count, int)
        or token_count < 1
        or len(payload) != token_count * 4
    ):
        raise ValueError(
            f"{token_file} has {len(payload)} bytes for invalid token count "
            f"{token_count!r}"
        )
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != metadata.get("token_ids_sha256"):
        raise ValueError(
            f"{token_file} sha256 {payload_sha256} does not match snapshot metadata"
        )
    token_ids = list(struct.unpack(f"<{token_count}I", payload))
    rendered_prefix = tokenizer.decode(token_ids, skip_special_tokens=False)
    if not rendered_prefix.startswith(GLM_USER_PREFIX):
        raise ValueError(
            "snapshot does not begin with the supported no-thinking GLM user template"
        )
    if tokenizer.encode(rendered_prefix, add_special_tokens=False).ids != token_ids:
        raise ValueError("snapshot token IDs do not round-trip through the tokenizer")
    return token_ids, rendered_prefix, metadata


def render_messages(messages: list[dict[str, str]]) -> str:
    rendered = "[gMASK]<sop>"
    for message in messages:
        role = message["role"]
        if role not in ("system", "user"):
            raise ValueError(f"matrix renderer does not support role {role!r}")
        rendered += f"<|{role}|>{message['content']}"
    return rendered + GLM_ASSISTANT_SUFFIX


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    engine_identity = os.environ.get("GLMRT_ENGINE_COMMIT", "").strip()
    if engine_identity:
        return engine_identity
    raise RuntimeError(
        f"cannot resolve source revision below {root} and GLMRT_ENGINE_COMMIT is unset"
    )


def request_metrics(
    endpoint: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout_seconds: float,
) -> tuple[dict, str]:
    body = json.dumps(
        {
            "model": MODEL_ID,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    metrics = None
    output_parts: list[str] = []
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                continue
            event = json.loads(payload)
            for choice in event.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    output_parts.append(content)
            if "metrics" in event:
                metrics = event["metrics"]
    if metrics is None:
        raise RuntimeError("stream completed without a metrics event")
    return metrics, "".join(output_parts)


def concise_record(
    *,
    case: tuple[int, int, str],
    repeat: int,
    metrics: dict,
    output_text: str,
    output_token_ids: list[int],
    commit: str,
    corpus_sha256: str,
    token_prefix_sha256: str,
    tokenizer_path: Path,
    mode: str,
    cached_prefix_tokens: int | None,
    planned_prompt_tokens: int,
    max_context_tokens: int,
    snapshot_path: Path | None,
    snapshot_token_sha256: str | None,
    prefix_verified: bool,
) -> dict:
    target_input, max_output, label = case
    real_full = metrics.get("real_full") or {}
    output_tokens = int(metrics.get("output_tokens") or 0)
    decode_ms = float(metrics.get("decode_ms") or 0.0)
    mtp_draft_lengths = [
        int(value) for value in real_full.get("mtp_draft_lengths") or []
    ]
    mtp_accepted_draft_lengths = [
        int(value) for value in real_full.get("mtp_accepted_draft_lengths") or []
    ]
    mtp_verify_cycle_ms = [
        float(value) for value in real_full.get("mtp_verify_cycle_ms") or []
    ]
    timed_decode_tokens = max(output_tokens - 1, 0)
    decode_tps = (
        timed_decode_tokens * 1000.0 / decode_ms
        if timed_decode_tokens and decode_ms > 0.0
        else None
    )
    return {
        "schema": "glmrt-real-full-matrix-v2",
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "commit": commit,
        "mode": mode,
        "case": label,
        "repeat": repeat,
        "target_source_tokens": target_input,
        "requested_uncached_source_tokens": (
            target_input if mode == "snapshot" else None
        ),
        "max_output_tokens": max_output,
        "planned_prompt_tokens": planned_prompt_tokens,
        "planned_uncached_prompt_tokens": (
            planned_prompt_tokens - cached_prefix_tokens
            if cached_prefix_tokens is not None
            else planned_prompt_tokens
        ),
        "required_context_tokens": planned_prompt_tokens + max_output,
        "max_context_tokens": max_context_tokens,
        "context_headroom_tokens": (
            max_context_tokens - planned_prompt_tokens - max_output
        ),
        "prompt_tokens": metrics.get("prompt_tokens"),
        "cached_prefix_tokens": cached_prefix_tokens,
        "uncached_prompt_tokens": (
            int(metrics["prompt_tokens"]) - cached_prefix_tokens
            if cached_prefix_tokens is not None and metrics.get("prompt_tokens") is not None
            else None
        ),
        "output_tokens": output_tokens,
        "output_text_sha256": hashlib.sha256(output_text.encode()).hexdigest(),
        "output_text_utf8_bytes": len(output_text.encode()),
        "output_text_preview": output_text[:160],
        "output_token_count_reencoded": len(output_token_ids),
        "output_token_prefix": output_token_ids[:16],
        "output_token_sha256": hashlib.sha256(
            b"".join(token_id.to_bytes(4, "little") for token_id in output_token_ids)
        ).hexdigest(),
        "cache_load_ms": metrics.get("cache_load_ms"),
        "prefill_ms": metrics.get("prefill_ms"),
        "prefill_tps": metrics.get("prefill_tokens_per_sec"),
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        "time_to_first_token_ms": metrics.get("time_to_first_token_ms"),
        "prefill_chunks": metrics.get("prefill_chunk_count"),
        "prefill_rows": metrics.get("layerwave_prefill_rows"),
        "decode_rows": metrics.get("layerwave_decode_rows"),
        "runtime_captures": real_full.get("request_coordinator_graph_captures"),
        "mtp_verify_cycles": real_full.get("mtp_verify_cycles"),
        "mtp_draft_tokens": real_full.get("mtp_draft_tokens"),
        "mtp_accepted_draft_tokens": real_full.get(
            "mtp_accepted_draft_tokens"
        ),
        "mtp_draft_lengths": mtp_draft_lengths,
        "mtp_accepted_draft_lengths": mtp_accepted_draft_lengths,
        "mtp_verify_cycle_ms": mtp_verify_cycle_ms,
        "status": real_full.get("status"),
        "numeric_progression_passed": real_full.get(
            "request_numeric_progression_passed"
        ),
        "attention_complete": real_full.get(
            "scheduler_full_context_device_attention_complete"
        ),
        "sample_status": real_full.get("scheduler_terminal_lm_head_sample_status"),
        "sparse_batches": real_full.get("request_sparse_batches"),
        "snapshot_path": str(snapshot_path) if snapshot_path is not None else None,
        "snapshot_token_sha256": snapshot_token_sha256,
        "snapshot_prefix_verified": prefix_verified,
        "corpus_sha256": corpus_sha256,
        "token_prefix_sha256": token_prefix_sha256,
        "tokenizer": str(tokenizer_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build prompts from real repository text and record a concise "
            "base/input/output serving envelope as JSONL."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        help="UTF-8 source file; repeat to build a larger deterministic corpus",
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case,
        help=(
            "SOURCE_TOKENS:MAX_OUTPUT_TOKENS[:LABEL]; SOURCE_TOKENS is the "
            "whole source slice in fresh mode and appended source in snapshot mode"
        ),
    )
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=("fresh", "snapshot"),
        default="fresh",
        help="label records as fresh prefill or packed-KV snapshot extension",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        help=(
            "packed KV/DSA snapshot root; required in snapshot mode so the "
            "tool can construct and verify the exact cached token prefix"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer.json; defaults to the local Hugging Face snapshot for --model",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/chat/completions",
    )
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=DEFAULT_MAX_CONTEXT_TOKENS,
        help="daemon sequence-capacity limit used to reject impossible cases",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL destination; defaults under ignored .glmrt-cache/benchmarks",
    )
    parser.add_argument(
        "--instruction",
        default="\n\nAnalyze the material above and explain its main technical design.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.max_context_tokens < 1:
        parser.error("--max-context-tokens must be positive")
    if args.mode == "snapshot" and args.snapshot is None:
        parser.error("--mode snapshot requires --snapshot")
    if args.mode == "fresh" and args.snapshot is not None:
        parser.error("--snapshot is only valid with --mode snapshot")
    return args


def main() -> int:
    args = parse_args()
    from tokenizers import Tokenizer

    root = repo_root()
    tokenizer_path = (args.tokenizer or default_tokenizer_path(args.model)).resolve()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus, corpus_sha256 = load_corpus(args.source)
    corpus_ids = tokenizer.encode(corpus, add_special_tokens=False).ids
    largest_case = max(case[0] for case in args.case)
    if largest_case > len(corpus_ids):
        raise ValueError(
            f"largest case needs {largest_case} source tokens, corpus has {len(corpus_ids)}"
        )

    snapshot_path = args.snapshot.resolve() if args.snapshot is not None else None
    snapshot_ids: list[int] = []
    snapshot_rendered_prefix = ""
    snapshot_metadata: dict = {}
    if snapshot_path is not None:
        snapshot_ids, snapshot_rendered_prefix, snapshot_metadata = load_snapshot(
            snapshot_path,
            tokenizer,
        )
    cached_prefix_tokens = len(snapshot_ids) if snapshot_ids else None
    snapshot_token_sha256 = (
        str(snapshot_metadata["token_ids_sha256"]) if snapshot_metadata else None
    )

    output = args.output
    if output is None:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        output = root / ".glmrt-cache" / "benchmarks" / f"prefill-matrix-{stamp}.jsonl"
    output = output.resolve()

    plans = []
    for case in args.case:
        source_tokens, max_output_tokens, label = case
        source_ids = corpus_ids[:source_tokens]
        source_text = tokenizer.decode(source_ids, skip_special_tokens=False)
        if args.mode == "snapshot":
            cached_message = snapshot_rendered_prefix[len(GLM_USER_PREFIX) :]
            messages = [
                {"role": "user", "content": cached_message},
                {
                    "role": "user",
                    "content": source_text + args.instruction,
                },
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": source_text + args.instruction,
                }
            ]
        planned_prompt_ids = tokenizer.encode(
            render_messages(messages),
            add_special_tokens=False,
        ).ids
        prefix_verified = (
            not snapshot_ids
            or planned_prompt_ids[: len(snapshot_ids)] == snapshot_ids
        )
        if not prefix_verified:
            mismatch = next(
                index
                for index, (planned, cached) in enumerate(
                    zip(planned_prompt_ids, snapshot_ids, strict=False)
                )
                if planned != cached
            )
            raise ValueError(
                f"case {label!r} diverges from snapshot at token {mismatch}"
            )
        required_context_tokens = len(planned_prompt_ids) + max_output_tokens
        if required_context_tokens > args.max_context_tokens:
            raise ValueError(
                f"case {label!r} requires {required_context_tokens} context "
                f"tokens but --max-context-tokens is {args.max_context_tokens}"
            )
        plans.append(
            {
                "case": case,
                "label": label,
                "messages": messages,
                "planned_prompt_tokens": len(planned_prompt_ids),
                "prefix_verified": prefix_verified,
                "token_prefix_sha256": hashlib.sha256(
                    b"".join(token_id.to_bytes(4, "little") for token_id in source_ids)
                ).hexdigest(),
            }
        )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "corpus_tokens": len(corpus_ids),
                    "corpus_sha256": corpus_sha256,
                    "tokenizer": str(tokenizer_path),
                    "mode": args.mode,
                    "snapshot": str(snapshot_path) if snapshot_path is not None else None,
                    "cached_prefix_tokens": cached_prefix_tokens,
                    "max_context_tokens": args.max_context_tokens,
                    "output": str(output),
                    "cases": [
                        {
                            "label": plan["label"],
                            "requested_source_tokens": plan["case"][0],
                            "max_output_tokens": plan["case"][1],
                            "planned_prompt_tokens": plan["planned_prompt_tokens"],
                            "planned_uncached_prompt_tokens": (
                                plan["planned_prompt_tokens"]
                                - (cached_prefix_tokens or 0)
                            ),
                            "required_context_tokens": (
                                plan["planned_prompt_tokens"] + plan["case"][1]
                            ),
                            "snapshot_prefix_verified": plan["prefix_verified"],
                        }
                        for plan in plans
                    ],
                },
                indent=2,
            )
        )
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    commit = git_commit(root)
    with output.open("a", encoding="utf-8") as destination:
        for plan in plans:
            for repeat in range(args.repeat):
                metrics, output_text = request_metrics(
                    args.endpoint,
                    plan["messages"],
                    plan["case"][1],
                    args.timeout_seconds,
                )
                if metrics.get("prompt_tokens") != plan["planned_prompt_tokens"]:
                    raise RuntimeError(
                        f"case {plan['label']!r} planned "
                        f"{plan['planned_prompt_tokens']} prompt tokens but the "
                        f"server reported {metrics.get('prompt_tokens')}"
                    )
                output_token_ids = tokenizer.encode(
                    output_text, add_special_tokens=False
                ).ids
                record = concise_record(
                    case=plan["case"],
                    repeat=repeat,
                    metrics=metrics,
                    output_text=output_text,
                    output_token_ids=output_token_ids,
                    commit=commit,
                    corpus_sha256=corpus_sha256,
                    token_prefix_sha256=plan["token_prefix_sha256"],
                    tokenizer_path=tokenizer_path,
                    mode=args.mode,
                    cached_prefix_tokens=cached_prefix_tokens,
                    planned_prompt_tokens=plan["planned_prompt_tokens"],
                    max_context_tokens=args.max_context_tokens,
                    snapshot_path=snapshot_path,
                    snapshot_token_sha256=snapshot_token_sha256,
                    prefix_verified=plan["prefix_verified"],
                )
                line = json.dumps(record, sort_keys=True)
                destination.write(line + "\n")
                destination.flush()
                print(line)
    print(f"wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
