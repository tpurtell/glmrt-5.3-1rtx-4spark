#!/usr/bin/env python3
"""Performance-gate the fused DFlash2 convolution/residual/RMS chain."""

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

TOOLS = Path(__file__).resolve().parent
REFERENCE = TOOLS.parent / "reference" / "glmrt_reference"
sys.path.insert(0, str(REFERENCE))
from dspark_body_capture import (  # noqa: E402
    _dflash2_finish_dynamic_conv_add_rms_norm,
    _dflash2_grouped_dynamic_conv,
    _dspark_add,
    _dspark_rms_norm,
)
from dflash_tuning_profile import dflash2_body_num_warps  # noqa: E402

REPO_ID = "incoai/GLM-5.3-DFlash2"
REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
WEIGHT_BYTES = 4_918_859_112
WIDTH = 6_144
GROUP_SIZE = 16
GROUPS = WIDTH // GROUP_SIZE
RMS_EPSILON = 1.0e-5
MIN_FUSED_SPEEDUP = 1.01
DEFAULT_CAPTURED_LAUNCHES = 16


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


def _case(
    *,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    concurrency: int,
    proposal_tokens: int,
    fused_warps: int,
    seed: int,
    warmup: int,
    iterations: int,
    rounds: int,
    captured_launches: int,
    measure: bool = True,
) -> dict[str, Any]:
    query_rows = proposal_tokens + 1
    rows = concurrency * query_rows
    device = base.device
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + concurrency * 100 + proposal_tokens)
    source = torch.randn(
        (rows, WIDTH),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    dynamic = (
        torch.randn(
            (rows, 4 * GROUPS),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.02
    ).to(torch.bfloat16)
    residual = torch.randn(
        (rows, WIDTH),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    split_conv = torch.empty_like(source)
    split_residual = torch.empty_like(source)
    split_normalized = torch.empty_like(source)
    fused_residual = torch.empty_like(source)
    fused_normalized = torch.empty_like(source)
    value_grid = (triton.cdiv(rows * WIDTH, 256),)

    def split() -> None:
        _dflash2_grouped_dynamic_conv[value_grid](
            source,
            dynamic,
            base,
            split_conv,
            QUERY_ROWS=query_rows,
            TOTAL_VALUES=rows * WIDTH,
            HIDDEN_SIZE=WIDTH,
            GROUP_SIZE=GROUP_SIZE,
            SIDE=1,
            BLOCK=256,
        )
        _dspark_add[value_grid](
            residual,
            split_conv,
            split_residual,
            TOTAL=rows * WIDTH,
            BLOCK=256,
        )
        _dspark_rms_norm[(rows,)](
            split_residual,
            norm_weight,
            split_normalized,
            WIDTH=WIDTH,
            BLOCK=triton.next_power_of_2(WIDTH),
            EPSILON=RMS_EPSILON,
            num_warps=8,
        )

    def fused() -> None:
        _dflash2_finish_dynamic_conv_add_rms_norm[(rows,)](
            source,
            dynamic,
            base,
            residual,
            norm_weight,
            fused_residual,
            fused_normalized,
            QUERY_ROWS=query_rows,
            WIDTH=WIDTH,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK=triton.next_power_of_2(WIDTH),
            EPSILON=RMS_EPSILON,
            num_warps=fused_warps,
        )

    split_graph = (
        _capture(split, captured_launches=captured_launches) if measure else None
    )
    fused_graph = (
        _capture(fused, captured_launches=captured_launches) if measure else None
    )
    if not measure:
        split()
        fused()
        torch.cuda.synchronize()
    if not torch.equal(fused_residual, split_residual):
        mismatch = int(torch.count_nonzero(fused_residual != split_residual).item())
        raise RuntimeError(
            f"fused residual differs at C{concurrency} K{proposal_tokens}: {mismatch} values"
        )
    if not torch.equal(fused_normalized, split_normalized):
        mismatch = int(torch.count_nonzero(fused_normalized != split_normalized).item())
        raise RuntimeError(
            f"fused norm differs at C{concurrency} K{proposal_tokens}: {mismatch} values"
        )
    if not measure:
        return {
            "active_requests": concurrency,
            "proposal_tokens": proposal_tokens,
            "query_rows_per_request": query_rows,
            "total_rows": rows,
            "fused_warps": fused_warps,
            "residual_exact": True,
            "normalized_exact": True,
        }
    assert split_graph is not None and fused_graph is not None
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
        "proposal_tokens": proposal_tokens,
        "query_rows_per_request": query_rows,
        "total_rows": rows,
        "fused_warps": fused_warps,
        "split_graph_kernel_nodes": 3,
        "fused_graph_kernel_nodes": 1,
        "split_gpu_ms": split_timing,
        "fused_gpu_ms": fused_timing,
        "fused_speedup": fused_speedup,
        "estimated_six_layer_cycle_saved_ms": 12.0 * (split_median - fused_median),
        "winner": "fused" if fused_median < split_median else "split",
        "performance_gate_passed": fused_speedup >= MIN_FUSED_SPEEDUP,
        "residual_exact": True,
        "normalized_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--widths", default="1,2,3,4,5,6,7")
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
    real_weight_pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    with safe_open(weight_path, framework="pt", device=str(device)) as weights:
        final_norm = weights.get_tensor("norm.weight")
        for layer in range(6):
            attention_base = weights.get_tensor(
                f"layers.{layer}.attention_conv.base_kernel"
            )
            attention_norm = weights.get_tensor(
                f"layers.{layer}.post_attention_layernorm.weight"
            )
            mlp_base = weights.get_tensor(f"layers.{layer}.mlp_conv.base_kernel")
            mlp_norm = (
                weights.get_tensor(f"layers.{layer + 1}.input_layernorm.weight")
                if layer + 1 < 6
                else final_norm
            )
            real_weight_pairs.extend(
                (
                    (f"layer-{layer}-attention", attention_base, attention_norm),
                    (f"layer-{layer}-mlp", mlp_base, mlp_norm),
                )
            )
    if any(
        base.shape != (2, 2, WIDTH)
        or norm_weight.shape != (WIDTH,)
        or base.dtype != torch.bfloat16
        or norm_weight.dtype != torch.bfloat16
        for _, base, norm_weight in real_weight_pairs
    ):
        raise RuntimeError("DFlash2 convolution or norm weights changed")
    _, base, norm_weight = real_weight_pairs[0]

    results = [
        _case(
            base=base,
            norm_weight=norm_weight,
            concurrency=requests,
            proposal_tokens=width,
            fused_warps=warps,
            seed=args.seed,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
            captured_launches=args.captured_launches,
        )
        for requests in concurrency
        for width in widths
        for warps in fused_warps
    ]
    selected = {
        f"c{requests}-k{width}": min(
            (
                result
                for result in results
                if result["active_requests"] == requests
                and result["proposal_tokens"] == width
            ),
            key=lambda result: result["fused_gpu_ms"]["median"],
        )
        for requests in concurrency
        for width in widths
    }
    winners = {key: result["fused_warps"] for key, result in selected.items()}
    runtime_warps = {
        key: dflash2_body_num_warps(
            result["active_requests"], result["proposal_tokens"]
        )
        for key, result in selected.items()
    }
    fused_wins_all_cases = all(
        result["performance_gate_passed"] is True for result in selected.values()
    )
    runtime_matches_winners = runtime_warps == winners
    real_weight_validation = []
    for pair_index, (weight_case, pair_base, pair_norm) in enumerate(real_weight_pairs):
        for requests in concurrency:
            for width in widths:
                validation = _case(
                    base=pair_base,
                    norm_weight=pair_norm,
                    concurrency=requests,
                    proposal_tokens=width,
                    fused_warps=winners[f"c{requests}-k{width}"],
                    seed=args.seed + (pair_index + 1) * 10_000,
                    warmup=1,
                    iterations=1,
                    rounds=1,
                    captured_launches=8,
                    measure=False,
                )
                real_weight_validation.append(
                    {"weight_case": weight_case, **validation}
                )
    body = {
        "schema": "glmrt-dflash2-body-fusion-tuning-v1",
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
        "runtime_body_sha256": _hash_file(REFERENCE / "dspark_body_capture.py"),
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
        "real_weight_validation": real_weight_validation,
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
