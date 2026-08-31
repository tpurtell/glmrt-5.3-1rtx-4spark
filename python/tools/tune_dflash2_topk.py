#!/usr/bin/env python3
"""Select the DFlash2 154,880-way top-k backend over the live row surface."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Callable

from huggingface_hub import snapshot_download
import torch

TOOLS = Path(__file__).resolve().parent
REFERENCE = TOOLS.parent / "reference" / "glmrt_reference"
sys.path.insert(0, os.fspath(REFERENCE))
from dflash_head_capture import _flashinfer_raw_topk_module  # noqa: E402

REPO_ID = "incoai/GLM-5.3-DFlash2"
REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
WEIGHT_BYTES = 4_918_859_112
VOCAB_SIZE = 154_880
TOP_K = 16
CONCURRENCY = (1, 2, 4)
WIDTHS = (1, 2, 3, 4, 5, 6, 7)
BACKENDS = ("torch", "flashinfer", "flashinfer-dsa")
MIN_NON_TORCH_SPEEDUP = 1.01
SELECTION_POLICY = "lowest_valid_case_aggregate_median_with_1pct_non_torch_gate"
DEFAULT_CAPTURED_LAUNCHES = 32


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _parse_ints(
    value: str,
    *,
    label: str,
    allowed: set[int],
) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{label} must be comma-separated integers"
        ) from error
    if (
        not values
        or len(values) != len(set(values))
        or any(item not in allowed for item in values)
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must contain unique values from {sorted(allowed)}"
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
        # Amortize replay overhead so this reflects a node inside the
        # production multi-node head graph rather than a one-node graph.
        for _ in range(captured_launches):
            launch()
    torch.cuda.synchronize()
    return graph


def _measure(
    graphs: dict[str, torch.cuda.CUDAGraph],
    *,
    warmup: int,
    iterations: int,
    rounds: int,
    captured_launches: int,
) -> dict[str, dict[str, float]]:
    backends = tuple(graphs)
    for _ in range(warmup):
        for backend in backends:
            graphs[backend].replay()
    torch.cuda.synchronize()
    samples = {backend: [] for backend in backends}
    for round_index in range(rounds):
        offset = round_index % len(backends)
        order = backends[offset:] + backends[:offset]
        for backend in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                graphs[backend].replay()
            end.record()
            end.synchronize()
            samples[backend].append(
                start.elapsed_time(end) / (iterations * captured_launches)
            )
    return {backend: _summary(samples[backend]) for backend in backends}


def _case(
    module: Any,
    *,
    active_requests: int,
    proposal_tokens: int,
    seed: int,
    warmup: int,
    iterations: int,
    rounds: int,
    captured_launches: int,
    disabled_backends: dict[str, str],
) -> dict[str, Any]:
    rows = active_requests * proposal_tokens
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + active_requests * 100 + proposal_tokens)
    logits = torch.randn(
        (rows, VOCAB_SIZE),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    values = {
        backend: torch.empty((rows, TOP_K), dtype=torch.bfloat16, device=device)
        for backend in BACKENDS
    }
    indices = {
        "torch": torch.empty((rows, TOP_K), dtype=torch.int64, device=device),
        "flashinfer": torch.empty((rows, TOP_K), dtype=torch.int32, device=device),
        "flashinfer-dsa": torch.empty((rows, TOP_K), dtype=torch.int32, device=device),
    }
    row_states = {
        backend: torch.zeros(1024 * 1024, dtype=torch.uint8, device=device)
        for backend in BACKENDS[1:]
    }

    def launch_torch() -> None:
        torch.topk(
            logits,
            TOP_K,
            dim=-1,
            largest=True,
            sorted=True,
            out=(values["torch"], indices["torch"]),
        )

    def launch_flashinfer(backend: str) -> None:
        module.radix_topk(
            logits,
            indices[backend],
            values[backend],
            row_states[backend],
            TOP_K,
            True,
            True,
            0,
            backend == "flashinfer-dsa",
        )

    launches: dict[str, Callable[[], None]] = {
        "torch": launch_torch,
        "flashinfer": lambda: launch_flashinfer("flashinfer"),
        "flashinfer-dsa": lambda: launch_flashinfer("flashinfer-dsa"),
    }
    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for backend in BACKENDS:
        if backend in disabled_backends:
            continue
        try:
            graphs[backend] = _capture(
                launches[backend], captured_launches=captured_launches
            )
        except RuntimeError as error:
            if backend == "torch":
                raise
            # FlashInfer exposes architecture-conditional variants through
            # one API.  An unsupported CUDA operation is a valid platform
            # result, not a reason to discard measurements for other valid
            # backends.  Disable it for the rest of this device sweep.
            disabled_backends[backend] = str(error)
    if "torch" not in graphs:
        raise RuntimeError("Torch top-k reference backend is unavailable")

    def parity() -> tuple[dict[str, bool], dict[str, bool]]:
        graphs["torch"].replay()
        for backend in tuple(graphs)[1:]:
            graphs[backend].replay()
        torch.cuda.synchronize()
        valid: dict[str, bool] = {}
        index_exact: dict[str, bool] = {}
        for backend in graphs:
            candidate_ids = indices[backend].to(torch.int64)
            sorted_ids = torch.sort(candidate_ids, dim=-1).values
            unique = bool(torch.all(sorted_ids[:, 1:] != sorted_ids[:, :-1]))
            valid[backend] = (
                unique
                and torch.equal(values[backend], values["torch"])
                and torch.equal(torch.gather(logits, -1, candidate_ids), values[backend])
            )
            index_exact[backend] = torch.equal(candidate_ids, indices["torch"])
        return valid, index_exact

    initial_valid, initial_index_exact = parity()
    logits.neg_()
    changed_valid, changed_index_exact = parity()
    if not initial_valid["torch"] or not changed_valid["torch"]:
        raise RuntimeError("internal Torch top-k reference changed unexpectedly")
    timings = _measure(
        graphs,
        warmup=warmup,
        iterations=iterations,
        rounds=rounds,
        captured_launches=captured_launches,
    )
    torch_median = timings["torch"]["median"]
    return {
        "active_requests": active_requests,
        "proposal_tokens": proposal_tokens,
        "rows": rows,
        "initial_valid": initial_valid,
        "changed_input_valid": changed_valid,
        "initial_index_exact": initial_index_exact,
        "changed_input_index_exact": changed_index_exact,
        "tie_policy": "equal_topk_values_valid_unique_ids_boundary_ties_allowed",
        "timing_ms": timings,
        "speedup_vs_torch": {
            backend: torch_median / timings[backend]["median"] for backend in graphs
        },
        "unsupported_backends": dict(disabled_backends),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--widths", default="1,2,3,4,5,6,7")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument(
        "--captured-launches", type=int, default=DEFAULT_CAPTURED_LAUNCHES
    )
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        concurrency = _parse_ints(
            args.concurrency, label="concurrency", allowed=set(CONCURRENCY)
        )
        widths = _parse_ints(args.widths, label="widths", allowed=set(WIDTHS))
    except argparse.ArgumentTypeError as error:
        parser.error(str(error))
    if (
        args.warmup < 1
        or args.iterations < 1
        or args.rounds < len(BACKENDS)
        or args.captured_launches < 8
    ):
        parser.error(
            "warmup/iterations must be positive, rounds must cover all backends, "
            "and captured-launches must be at least 8"
        )
    if args.output.exists() or args.output.is_symlink():
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
    module = _flashinfer_raw_topk_module()
    disabled_backends: dict[str, str] = {}
    results = []
    for requests in concurrency:
        for width in widths:
            results.append(
                _case(
                    module,
                    active_requests=requests,
                    proposal_tokens=width,
                    seed=args.seed,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    rounds=args.rounds,
                    captured_launches=args.captured_launches,
                    disabled_backends=disabled_backends,
                )
            )
    valid_backends = [
        backend
        for backend in BACKENDS
        if all(
            result["initial_valid"].get(backend, False)
            and result["changed_input_valid"].get(backend, False)
            for result in results
        )
    ]
    aggregate_ms = {
        backend: sum(result["timing_ms"][backend]["median"] for result in results)
        for backend in valid_backends
    }
    fastest = min(aggregate_ms, key=aggregate_ms.get)
    torch_total = aggregate_ms["torch"]
    aggregate_speedup = torch_total / aggregate_ms[fastest]
    selected_backend = (
        fastest
        if fastest == "torch" or aggregate_speedup >= MIN_NON_TORCH_SPEEDUP
        else "torch"
    )
    body = {
        "schema": "glmrt-dflash2-topk-tuning-v1",
        # This report selects the backend to carry into the production graph
        # and service trials.  It is not, by itself, full-service acceptance.
        "status": "measured",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot": os.fspath(snapshot),
        "config_sha256": _hash_file(config_path),
        "weight_sha256": _hash_file(weight_path),
        "runtime_head_sha256": _hash_file(REFERENCE / "dflash_head_capture.py"),
        "script_sha256": _hash_file(Path(__file__).resolve()),
        "flashinfer_version": package_version("flashinfer-python"),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "concurrency": list(concurrency),
        "widths": list(widths),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "rounds": args.rounds,
        "captured_launches": args.captured_launches,
        "minimum_non_torch_speedup": MIN_NON_TORCH_SPEEDUP,
        "selection_policy": SELECTION_POLICY,
        "full_service_acceptance_required": True,
        "valid_backends": valid_backends,
        "unsupported_backends": disabled_backends,
        "aggregate_median_ms": aggregate_ms,
        "fastest_valid_backend": fastest,
        "fastest_valid_speedup_vs_torch": aggregate_speedup,
        "selected_backend": selected_backend,
        "results": results,
    }
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    report = {**body, "report_sha256": hashlib.sha256(canonical).hexdigest()}
    _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
