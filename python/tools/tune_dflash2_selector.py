#!/usr/bin/env python3
"""Compare split and fused DFlash2 candidate selectors on real codebooks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

from huggingface_hub import snapshot_download
from safetensors import safe_open
import torch
import triton
import triton.language as tl

TOOLS = Path(__file__).resolve().parent
REFERENCE = TOOLS.parent / "reference" / "glmrt_reference"
sys.path.insert(0, str(REFERENCE))
from dflash_head_capture import (  # noqa: E402
    _dflash2_select_candidate,
)
from dflash_tuning_profile import dflash2_selector_num_warps  # noqa: E402

REPO_ID = "incoai/GLM-5.3-DFlash2"
REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
WEIGHT_BYTES = 4_918_859_112
VOCAB = 154_880
RANK = 256
TOP_K = 16
MIN_FUSED_SPEEDUP = 1.01
DEFAULT_CAPTURED_LAUNCHES = 16


@triton.jit
def _split_transition_scores(
    predecessor_codebook,
    predecessor_tokens,
    hidden,
    successor_codebook,
    candidates,
    unary,
    output,
    RANK: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    rank_offsets = tl.arange(0, BLOCK)
    rank_mask = rank_offsets < RANK
    candidate_offsets = tl.arange(0, TOP_K)
    tokens = tl.load(candidates + row * TOP_K + candidate_offsets)
    predecessor_token = tl.load(predecessor_tokens + row)
    previous = tl.load(
        predecessor_codebook + predecessor_token * RANK + rank_offsets,
        mask=rank_mask,
        other=0.0,
    )
    current = tl.load(
        hidden + row * RANK + rank_offsets,
        mask=rank_mask,
        other=0.0,
    )
    successor = tl.load(
        successor_codebook + tokens[:, None] * RANK + rank_offsets[None, :],
        mask=rank_mask[None, :],
        other=0.0,
    )
    conditioned = (previous * current).to(tl.bfloat16)
    conditioned_matrix = tl.broadcast_to(conditioned[:, None], (BLOCK, TOP_K))
    transition_matrix = tl.dot(
        successor, conditioned_matrix, out_dtype=tl.float32
    )
    transition = tl.sum(
        tl.where(candidate_offsets[None, :] == 0, transition_matrix, 0.0), axis=1
    ).to(tl.bfloat16)
    unary_scores = tl.load(unary + row * TOP_K + candidate_offsets)
    scores = (unary_scores.to(tl.bfloat16) + transition).to(tl.bfloat16)
    tl.store(output + row * TOP_K + candidate_offsets, scores)


@triton.jit
def _split_candidate_argmax(
    scores,
    candidates,
    predecessor_output,
    final_output,
    TOP_K: tl.constexpr,
    POSITION: tl.constexpr,
    PROPOSAL_TOKENS: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < TOP_K
    values = tl.load(
        scores + row * TOP_K + offsets,
        mask=mask,
        other=-float("inf"),
    )
    best = tl.max(values, axis=0)
    best_index = tl.min(
        tl.where((values == best) & mask, offsets, TOP_K),
        axis=0,
    )
    token = tl.load(candidates + row * TOP_K + best_index)
    tl.store(predecessor_output + row, token)
    tl.store(final_output + row * PROPOSAL_TOKENS + POSITION, token)


def _parse_ints(value: str, *, label: str, allowed: set[int]) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if (
        not values
        or len(set(values)) != len(values)
        or any(item not in allowed for item in values)
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must be unique comma-separated values from {sorted(allowed)}"
        )
    return values


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "maximum": ordered[-1],
    }


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _capture(
    launch: Callable[[], None], *, captured_launches: int
) -> torch.cuda.CUDAGraph:
    current = torch.cuda.current_stream()
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(current)
    with torch.cuda.stream(warmup_stream):
        launch()
    current.wait_stream(warmup_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(captured_launches):
            launch()
    torch.cuda.synchronize()
    return graph


def _measure_pair(
    split_graph: torch.cuda.CUDAGraph,
    fused_graph: torch.cuda.CUDAGraph,
    *,
    warmup: int,
    iterations: int,
    rounds: int,
    captured_launches: int,
) -> tuple[dict[str, float], dict[str, float]]:
    for _ in range(warmup):
        split_graph.replay()
        fused_graph.replay()
    torch.cuda.synchronize()
    samples: dict[str, list[float]] = {"split": [], "fused": []}
    for round_index in range(rounds):
        order = (
            (("split", split_graph), ("fused", fused_graph))
            if round_index % 2 == 0
            else (("fused", fused_graph), ("split", split_graph))
        )
        for label, graph in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graph.replay()
            end.record()
            end.synchronize()
            samples[label].append(
                start.elapsed_time(end) / (iterations * captured_launches)
            )
    return _summary(samples["split"]), _summary(samples["fused"])


def _run_reference(
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    anchors: torch.Tensor,
    projected_hidden: torch.Tensor,
    candidates: torch.Tensor,
    unary: torch.Tensor,
) -> torch.Tensor:
    width, concurrency, _ = candidates.shape
    output = torch.empty(
        (concurrency, width),
        dtype=torch.int64,
        device=candidates.device,
    )
    predecessor = anchors
    for position in range(width):
        conditioned = (
            predecessor_codebook[predecessor] * projected_hidden[position]
        ).to(torch.bfloat16)
        successor = successor_codebook[candidates[position].long()]
        transition = torch.einsum("br,bkr->bk", conditioned, successor).to(
            torch.bfloat16
        )
        scores = unary[position] + transition
        index = torch.argmax(scores, dim=-1)
        predecessor = candidates[position].long().gather(1, index[:, None])[:, 0]
        output[:, position].copy_(predecessor)
    return output


def _case(
    *,
    predecessor_codebook: torch.Tensor,
    successor_codebook: torch.Tensor,
    concurrency: int,
    width: int,
    candidate_dtype: torch.dtype,
    fused_warps: int,
    seed: int,
    warmup: int,
    iterations: int,
    rounds: int,
    captured_launches: int,
) -> dict[str, Any]:
    device = predecessor_codebook.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 100 * concurrency + width)
    candidates = torch.randint(
        VOCAB,
        (width, concurrency, TOP_K),
        dtype=candidate_dtype,
        device=device,
        generator=generator,
    )
    unary = torch.randn(
        (width, concurrency, TOP_K),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    projected_hidden = torch.randn(
        (width, concurrency, RANK),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    anchors = torch.randint(
        VOCAB,
        (concurrency,),
        dtype=torch.int64,
        device=device,
        generator=generator,
    )
    split_scores = torch.empty((concurrency, TOP_K), dtype=torch.bfloat16, device=device)
    split_steps = torch.empty((width, concurrency), dtype=torch.int64, device=device)
    split_output = torch.empty((concurrency, width), dtype=torch.int64, device=device)
    fused_steps = torch.empty_like(split_steps)
    fused_output = torch.empty_like(split_output)

    def split() -> None:
        predecessor = anchors
        for position in range(width):
            _split_transition_scores[(concurrency,)](
                predecessor_codebook,
                predecessor,
                projected_hidden[position],
                successor_codebook,
                candidates[position],
                unary[position],
                split_scores,
                RANK=RANK,
                TOP_K=TOP_K,
                BLOCK=RANK,
                num_warps=4,
            )
            _split_candidate_argmax[(concurrency,)](
                split_scores,
                candidates[position],
                split_steps[position],
                split_output,
                TOP_K=TOP_K,
                POSITION=position,
                PROPOSAL_TOKENS=width,
                BLOCK=TOP_K,
            )
            predecessor = split_steps[position]

    def fused() -> None:
        predecessor = anchors
        for position in range(width):
            _dflash2_select_candidate[(concurrency,)](
                predecessor_codebook,
                predecessor,
                projected_hidden[position],
                successor_codebook,
                candidates[position],
                unary[position],
                fused_steps[position],
                fused_output,
                RANK=RANK,
                TOP_K=TOP_K,
                POSITION=position,
                PROPOSAL_TOKENS=width,
                RANK_BLOCK=RANK,
                num_warps=fused_warps,
            )
            predecessor = fused_steps[position]

    reference = _run_reference(
        predecessor_codebook,
        successor_codebook,
        anchors,
        projected_hidden,
        candidates,
        unary,
    )
    split_graph = _capture(split, captured_launches=captured_launches)
    fused_graph = _capture(fused, captured_launches=captured_launches)
    split_mismatch = int(torch.count_nonzero(split_output != reference).item())
    fused_mismatch = int(torch.count_nonzero(fused_output != reference).item())
    if split_mismatch or fused_mismatch:
        raise RuntimeError(
            f"selector disagrees with reference at C{concurrency} K{width}: "
            f"split_mismatch={split_mismatch} fused_mismatch={fused_mismatch}"
        )
    split_timing, fused_timing = _measure_pair(
        split_graph,
        fused_graph,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
        captured_launches=captured_launches,
    )
    split_median = split_timing["median"]
    fused_median = fused_timing["median"]
    fused_speedup = split_median / fused_median
    return {
        "active_requests": concurrency,
        "proposal_tokens": width,
        "candidate_dtype": str(candidate_dtype).removeprefix("torch."),
        "fused_warps": fused_warps,
        "split_graph_kernel_nodes": 2 * width,
        "fused_graph_kernel_nodes": width,
        "split_gpu_ms": split_timing,
        "fused_gpu_ms": fused_timing,
        "fused_speedup": fused_speedup,
        "winner": "fused" if fused_median < split_median else "split",
        "performance_gate_passed": fused_speedup >= MIN_FUSED_SPEEDUP,
        "reference_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--widths", default="1,2,3,4,5,6,7")
    parser.add_argument(
        "--candidate-dtypes",
        choices=("i64", "i32", "both"),
        default="both",
    )
    parser.add_argument("--fused-warps", default="4,8")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--captured-launches", type=int, default=DEFAULT_CAPTURED_LAUNCHES
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        concurrency = _parse_ints(
            args.concurrency, label="concurrency", allowed={1, 2, 4}
        )
        widths = _parse_ints(args.widths, label="widths", allowed=set(range(1, 8)))
        fused_warps = _parse_ints(args.fused_warps, label="fused-warps", allowed={4, 8})
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if min(args.warmup, args.iterations, args.rounds) < 1 or args.captured_launches < 8:
        parser.error(
            "warmup, iterations, and rounds must be positive and "
            "captured-launches must be at least 8"
        )
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")

    snapshot = (
        args.snapshot.expanduser().resolve(strict=True)
        if args.snapshot is not None
        else Path(snapshot_download(REPO_ID, revision=REVISION, local_files_only=True))
    )
    if snapshot.name != REVISION:
        raise RuntimeError(
            f"DFlash2 snapshot must resolve to pinned revision {REVISION}, got {snapshot}"
        )
    config_path = snapshot / "config.json"
    weight_path = snapshot / "model.safetensors"
    if weight_path.stat().st_size != WEIGHT_BYTES:
        raise RuntimeError(
            f"DFlash2 weight size changed: {weight_path.stat().st_size} != {WEIGHT_BYTES}"
        )
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    with safe_open(weight_path, framework="pt", device=str(device)) as weights:
        predecessor_codebook = weights.get_tensor(
            "candidate_selector.predecessor_codebook"
        )
        successor_codebook = weights.get_tensor("candidate_selector.successor_codebook")
    if predecessor_codebook.shape != (VOCAB, RANK) or successor_codebook.shape != (
        VOCAB,
        RANK,
    ):
        raise RuntimeError("DFlash2 candidate codebook geometry changed")
    if (
        predecessor_codebook.dtype != torch.bfloat16
        or successor_codebook.dtype != torch.bfloat16
    ):
        raise RuntimeError("DFlash2 candidate codebooks are not BF16")

    dtypes = {
        "i64": (torch.int64,),
        "i32": (torch.int32,),
        "both": (torch.int64, torch.int32),
    }[args.candidate_dtypes]
    results = [
        _case(
            predecessor_codebook=predecessor_codebook,
            successor_codebook=successor_codebook,
            concurrency=requests,
            width=width,
            candidate_dtype=dtype,
            fused_warps=warps,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            captured_launches=args.captured_launches,
        )
        for dtype in dtypes
        for requests in concurrency
        for width in widths
        for warps in fused_warps
    ]
    selected = {
        f"{str(dtype).removeprefix('torch.')}-c{requests}-k{width}": min(
            (
                result
                for result in results
                if result["candidate_dtype"] == str(dtype).removeprefix("torch.")
                and result["active_requests"] == requests
                and result["proposal_tokens"] == width
            ),
            key=lambda result: result["fused_gpu_ms"]["median"],
        )
        for dtype in dtypes
        for requests in concurrency
        for width in widths
    }
    winners = {key: result["fused_warps"] for key, result in selected.items()}
    runtime_warps = {
        key: dflash2_selector_num_warps(
            result["active_requests"],
            result["proposal_tokens"],
            result["candidate_dtype"],
        )
        for key, result in selected.items()
    }
    fused_wins_all_cases = all(
        result["performance_gate_passed"] is True for result in selected.values()
    )
    runtime_matches_winners = runtime_warps == winners
    body = {
        "schema": "glmrt-dflash2-selector-tuning-v1",
        "status": (
            "accepted"
            if fused_wins_all_cases and runtime_matches_winners
            else "rejected"
        ),
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot": str(snapshot),
        "config_sha256": _hash_file(config_path),
        "weight_sha256": _hash_file(weight_path),
        "script_sha256": _hash_file(Path(__file__).resolve()),
        "runtime_selector_sha256": _hash_file(REFERENCE / "dflash_head_capture.py"),
        "runtime_profile_sha256": _hash_file(REFERENCE / "dflash_tuning_profile.py"),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "captured_launches": args.captured_launches,
        "minimum_fused_speedup": MIN_FUSED_SPEEDUP,
        "results": results,
        "winning_fused_warps": winners,
        "runtime_fused_warps": runtime_warps,
        "fused_wins_all_cases": fused_wins_all_cases,
        "runtime_matches_winners": runtime_matches_winners,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    report = {**body, "report_sha256": hashlib.sha256(canonical).hexdigest()}
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, args.output)
    print(encoded, end="")


if __name__ == "__main__":
    main()
