from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import triton
import triton.language as tl

from b12x_mla_capture import _bf16_tensor, _i32_tensor
from dspark_capture import _fp8_tensor


_DSPARK_UPDATE_STATES: dict[tuple[Any, ...], "_DsparkUpdateState"] = {}


@dataclass(frozen=True)
class _DsparkUpdateLayerWeights:
    k_norm: Any
    kv_t: Any


@dataclass(frozen=True)
class _DsparkUpdateState:
    device_id: int
    cuda_stream: int
    rows: int
    active_requests: int
    layers: int
    hidden_size: int
    target_features: int
    heads: int
    head_dim: int
    rope_theta: float
    total_pages: int
    page_size: int
    max_pages_per_request: int
    cache_dtype: str
    target_hidden: Any
    fusion_output: Any
    fused_hidden: Any
    projected_kv: Any
    key_output: Any
    value_output: Any
    reference_fused_hidden: Any
    reference_key_output: Any
    reference_value_output: Any
    eager_fused_hidden: Any
    eager_key_output: Any
    eager_value_output: Any
    k_cache: Any
    v_cache: Any
    row_request_ids: Any
    row_positions: Any
    row_cache_positions: Any
    block_tables: Any
    target_fusion_t: Any
    hidden_norm: Any
    layer_weights: tuple[_DsparkUpdateLayerWeights, ...]


def prepare_dspark_context_update(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Bind, initialize, compile, and validate one target-context update bucket."""

    import torch

    state = _update_state(ctx, kwargs, create=True)
    _run_reference(state)
    _run_update(state)
    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        state.eager_fused_hidden.copy_(state.fused_hidden)
        state.eager_key_output.copy_(state.key_output)
        state.eager_value_output.copy_(state.value_output)


def capture_dspark_context_update(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch the allocation-free target-context update during external capture."""

    _run_update(_update_state(ctx, kwargs, create=False))


def _update_state(
    ctx: dict[str, Any], kwargs: dict[str, Any], *, create: bool
) -> _DsparkUpdateState:
    rows = int(kwargs["rows"])
    active_requests = int(kwargs["active_requests"])
    layers = int(kwargs["layers"])
    hidden_size = int(kwargs["hidden_size"])
    target_features = int(kwargs["target_features"])
    heads = int(kwargs["heads"])
    head_dim = int(kwargs["head_dim"])
    rope_theta = float(kwargs.get("rope_theta", 8_000_000.0))
    total_pages = int(kwargs["total_pages"])
    page_size = int(kwargs["page_size"])
    max_pages_per_request = int(kwargs["max_pages_per_request"])
    seed = int(kwargs["seed"])
    initialize_target_hidden = bool(kwargs.get("initialize_target_hidden", True))
    initialize_kv = bool(kwargs.get("initialize_kv", True))
    cache_dtype = str(kwargs.get("cache_dtype", "bf16")).lower()
    if rows < 1 or rows > 1024:
        raise ValueError(f"context update rows must be in 1..1024, got {rows}")
    if active_requests not in (1, 2, 4):
        raise ValueError(
            "dSpark context update active_requests must be 1/2/4, "
            f"got {active_requests}"
        )
    dspark_geometry = (
        layers in (3, 5)
        and hidden_size == 6144
        and target_features == 5 * hidden_size
        and heads == 64
        and head_dim == 64
        and rope_theta == 8_000_000.0
    )
    dflash2_geometry = (
        layers == 6
        and hidden_size == 6144
        and target_features == 6 * hidden_size
        and heads == 8
        and head_dim == 128
        and rope_theta == 1_000_000.0
    )
    if not (dspark_geometry or dflash2_geometry):
        raise ValueError(
            "target-context update geometry is neither GLM-5.2 dSpark nor "
            "GLM-5.3 DFlash2"
        )
    if rows & (rows - 1) and not (
        dflash2_geometry and active_requests == 1 and rows <= 8
    ):
        raise ValueError(
            "context update rows must be a power of two except for exact-small "
            f"DFlash2 C1 decode buckets, got rows={rows} C={active_requests}"
        )
    if page_size not in (16, 32, 64, 128):
        raise ValueError(f"unsupported dSpark update page size {page_size}")
    if total_pages < 1 or max_pages_per_request < 1:
        raise ValueError("dSpark update page counts are invalid")
    if cache_dtype not in ("bf16", "fp8"):
        raise ValueError(f"unsupported dSpark update cache dtype {cache_dtype}")

    buffers = ctx["buffers"]
    mutable_names = (
        "target_hidden",
        "fusion_output",
        "fused_hidden",
        "projected_kv",
        "key_output",
        "value_output",
        "reference_fused_hidden",
        "reference_key_output",
        "reference_value_output",
        "eager_fused_hidden",
        "eager_key_output",
        "eager_value_output",
        "k_cache",
        "v_cache",
        "row_request_ids",
        "row_positions",
        "row_cache_positions",
        "block_tables",
    )
    weight_names = ["target_fusion", "hidden_norm"]
    for layer in range(layers):
        weight_names.extend((f"layer_{layer}_k_norm", f"layer_{layer}_kv"))
    required_names = (*mutable_names, *weight_names)
    missing = [name for name in required_names if name not in buffers]
    if missing:
        raise ValueError(f"dSpark context update is missing buffers: {missing}")

    device_id = int(buffers["target_hidden"]["device_id"])
    for name in required_names:
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(f"dSpark update buffer {name} is on another CUDA device")

    attention_width = heads * head_dim
    output_shape = (layers, rows, heads, head_dim)
    cache_shape = (layers, total_pages, heads, page_size, head_dim)
    shapes = {
        "target_hidden": (rows, target_features),
        "fusion_output": (rows, hidden_size),
        "fused_hidden": (rows, hidden_size),
        "projected_kv": (rows, 2 * attention_width),
        "key_output": output_shape,
        "value_output": output_shape,
        "reference_fused_hidden": (rows, hidden_size),
        "reference_key_output": output_shape,
        "reference_value_output": output_shape,
        "eager_fused_hidden": (rows, hidden_size),
        "eager_key_output": output_shape,
        "eager_value_output": output_shape,
        "k_cache": cache_shape,
        "v_cache": cache_shape,
        "row_request_ids": (rows,),
        "row_positions": (rows,),
        "row_cache_positions": (rows,),
        "block_tables": (active_requests, max_pages_per_request),
        "target_fusion": (hidden_size, target_features),
        "hidden_norm": (hidden_size,),
    }
    for layer in range(layers):
        shapes[f"layer_{layer}_k_norm"] = (head_dim,)
        shapes[f"layer_{layer}_kv"] = (2 * attention_width, hidden_size)
    for name, shape in shapes.items():
        element_bytes = (
            4
            if name
            in (
                "row_request_ids",
                "row_positions",
                "row_cache_positions",
                "block_tables",
            )
            else 2
        )
        if name in ("k_cache", "v_cache") and cache_dtype == "fp8":
            element_bytes = 1
        required_bytes = _tensor_bytes(shape, element_bytes)
        if int(buffers[name]["bytes"]) < required_bytes:
            raise ValueError(
                f"dSpark update buffer {name} has {buffers[name]['bytes']} bytes, "
                f"requires {required_bytes}"
            )

    key = (
        int(ctx["cuda_stream"]),
        rows,
        active_requests,
        layers,
        hidden_size,
        target_features,
        heads,
        head_dim,
        rope_theta,
        total_pages,
        page_size,
        max_pages_per_request,
        cache_dtype,
        initialize_target_hidden,
        initialize_kv,
        *((name, int(buffers[name]["ptr"])) for name in required_names),
    )
    state = _DSPARK_UPDATE_STATES.get(key)
    if state is not None:
        return state
    if not create:
        raise RuntimeError("dSpark update capture requires a matching startup prepare call")

    import torch

    cuda_stream = int(ctx["cuda_stream"])
    stream = torch.cuda.ExternalStream(cuda_stream, device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream), torch.no_grad():
        tensors: dict[str, Any] = {}
        for name, shape in shapes.items():
            if name in (
                "row_request_ids",
                "row_positions",
                "row_cache_positions",
                "block_tables",
            ):
                tensors[name] = _i32_tensor(buffers[name], shape)
            elif name in ("k_cache", "v_cache") and cache_dtype == "fp8":
                tensors[name] = _fp8_tensor(buffers[name], shape)
            else:
                tensors[name] = _bf16_tensor(buffers[name], shape)
        if initialize_target_hidden:
            generator = torch.Generator(device=device_id)
            generator.manual_seed(seed)
            tensors["target_hidden"].normal_(generator=generator)
        if initialize_kv:
            if cache_dtype == "bf16":
                tensors["k_cache"].zero_()
                tensors["v_cache"].zero_()
            else:
                tensors["k_cache"].view(torch.uint8).zero_()
                tensors["v_cache"].view(torch.uint8).zero_()

    layer_weights = tuple(
        _DsparkUpdateLayerWeights(
            k_norm=tensors[f"layer_{layer}_k_norm"],
            kv_t=tensors[f"layer_{layer}_kv"].t(),
        )
        for layer in range(layers)
    )
    state = _DsparkUpdateState(
        device_id=device_id,
        cuda_stream=cuda_stream,
        rows=rows,
        active_requests=active_requests,
        layers=layers,
        hidden_size=hidden_size,
        target_features=target_features,
        heads=heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
        total_pages=total_pages,
        page_size=page_size,
        max_pages_per_request=max_pages_per_request,
        cache_dtype=cache_dtype,
        target_hidden=tensors["target_hidden"],
        fusion_output=tensors["fusion_output"],
        fused_hidden=tensors["fused_hidden"],
        projected_kv=tensors["projected_kv"],
        key_output=tensors["key_output"],
        value_output=tensors["value_output"],
        reference_fused_hidden=tensors["reference_fused_hidden"],
        reference_key_output=tensors["reference_key_output"],
        reference_value_output=tensors["reference_value_output"],
        eager_fused_hidden=tensors["eager_fused_hidden"],
        eager_key_output=tensors["eager_key_output"],
        eager_value_output=tensors["eager_value_output"],
        k_cache=tensors["k_cache"],
        v_cache=tensors["v_cache"],
        row_request_ids=tensors["row_request_ids"],
        row_positions=tensors["row_positions"],
        row_cache_positions=tensors["row_cache_positions"],
        block_tables=tensors["block_tables"],
        target_fusion_t=tensors["target_fusion"].t(),
        hidden_norm=tensors["hidden_norm"],
        layer_weights=layer_weights,
    )
    _DSPARK_UPDATE_STATES[key] = state
    return state


def _rms_norm(source: Any, weight: Any) -> Any:
    import torch

    variance = source.float().pow(2).mean(dim=-1, keepdim=True)
    return source * torch.rsqrt(variance + 1.0e-5).to(source.dtype) * weight


def _apply_rope(
    source: Any, positions: Any, head_dim: int, rope_theta: float
) -> Any:
    import torch

    exponent = torch.arange(0, head_dim, 2, device=source.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (rope_theta ** (exponent / head_dim))
    angles = positions.float().unsqueeze(-1) * inverse_frequency
    cosine = angles.cos().to(source.dtype).view(-1, 1, head_dim // 2)
    sine = angles.sin().to(source.dtype).view(-1, 1, head_dim // 2)
    first, second = source.chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)


def _run_reference(state: _DsparkUpdateState) -> None:
    import torch
    import torch.nn.functional as functional

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        fused = _rms_norm(
            functional.linear(state.target_hidden, state.target_fusion_t.t()),
            state.hidden_norm,
        )
        state.reference_fused_hidden.copy_(fused)
        for layer, weights in enumerate(state.layer_weights):
            projected = functional.linear(fused, weights.kv_t.t())
            keys, values = projected.chunk(2, dim=-1)
            keys = keys.view(state.rows, state.heads, state.head_dim)
            values = values.view(state.rows, state.heads, state.head_dim)
            keys = _apply_rope(
                _rms_norm(keys, weights.k_norm),
                state.row_positions,
                state.head_dim,
                state.rope_theta,
            )
            state.reference_key_output[layer].copy_(keys)
            state.reference_value_output[layer].copy_(values)


def _run_update(state: _DsparkUpdateState) -> None:
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    attention_width = state.heads * state.head_dim
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        torch.mm(
            state.target_hidden,
            state.target_fusion_t,
            out=state.fusion_output,
        )
        _dspark_rms_norm[(state.rows,)](
            state.fusion_output,
            state.hidden_norm,
            state.fused_hidden,
            WIDTH=state.hidden_size,
            BLOCK=triton.next_power_of_2(state.hidden_size),
            EPSILON=1.0e-5,
            num_warps=8,
        )
        for layer, weights in enumerate(state.layer_weights):
            torch.mm(
                state.fused_hidden,
                weights.kv_t,
                out=state.projected_kv,
            )
            _dspark_kv_rope_scatter[(state.rows * state.heads,)](
                state.projected_kv,
                weights.k_norm,
                state.row_request_ids,
                state.row_positions,
                state.row_cache_positions,
                state.block_tables,
                state.key_output[layer],
                state.value_output[layer],
                state.k_cache[layer],
                state.v_cache[layer],
                ROWS=state.rows,
                HEADS=state.heads,
                HEAD_DIM=state.head_dim,
                PAGE_SIZE=state.page_size,
                MAX_PAGES=state.max_pages_per_request,
                KV_WIDTH=2 * attention_width,
                THETA=state.rope_theta,
                EPSILON=1.0e-5,
                BLOCK=state.head_dim,
                num_warps=4,
            )


@triton.jit
def _dspark_rms_norm(
    source,
    weight,
    output,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < WIDTH
    values = tl.load(source + row * WIDTH + offsets, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / WIDTH
    inverse_rms = tl.rsqrt(variance + EPSILON)
    scales = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    normalized = (values * inverse_rms).to(tl.bfloat16)
    scaled = (normalized.to(tl.float32) * scales).to(tl.bfloat16)
    tl.store(output + row * WIDTH + offsets, scaled, mask=mask)


@triton.jit
def _dspark_kv_rope_scatter(
    projected_kv,
    k_norm,
    row_request_ids,
    row_positions,
    row_cache_positions,
    block_tables,
    key_output,
    value_output,
    k_cache,
    v_cache,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    THETA: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    row = program // HEADS
    head = program - row * HEADS
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM
    head_base = head * HEAD_DIM
    row_base = row * KV_WIDTH
    key_values = tl.load(
        projected_kv + row_base + head_base + offsets, mask=mask, other=0.0
    ).to(tl.float32)
    variance = tl.sum(key_values * key_values, axis=0) / HEAD_DIM
    scales = tl.load(k_norm + offsets, mask=mask, other=0.0).to(tl.float32)
    key_values = (key_values * tl.rsqrt(variance + EPSILON)).to(tl.bfloat16)
    key_values = (key_values.to(tl.float32) * scales).to(tl.bfloat16)

    half = HEAD_DIM // 2
    pair = offsets % half
    paired_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    paired_values = tl.load(
        projected_kv + row_base + head_base + paired_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    paired_scales = tl.load(k_norm + paired_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    paired_values = (paired_values * tl.rsqrt(variance + EPSILON)).to(tl.bfloat16)
    paired_values = (
        paired_values.to(tl.float32) * paired_scales
    ).to(tl.bfloat16)
    position = tl.load(row_positions + row)
    frequency = tl.exp((-math.log(THETA) * (2.0 * pair)) / HEAD_DIM)
    angle = position.to(tl.float32) * frequency
    cosine = tl.cos(angle).to(tl.bfloat16)
    sine = tl.sin(angle).to(tl.bfloat16)
    cosine_term = (
        key_values.to(tl.float32) * cosine.to(tl.float32)
    ).to(tl.bfloat16)
    sine_term = (
        paired_values.to(tl.float32) * sine.to(tl.float32)
    ).to(tl.bfloat16)
    rotated = tl.where(
        offsets < half,
        (cosine_term.to(tl.float32) - sine_term.to(tl.float32)).to(tl.bfloat16),
        (cosine_term.to(tl.float32) + sine_term.to(tl.float32)).to(tl.bfloat16),
    ).to(tl.bfloat16)
    output_base = (row * HEADS + head) * HEAD_DIM
    tl.store(key_output + output_base + offsets, rotated, mask=mask)

    values = tl.load(
        projected_kv + row_base + HEADS * HEAD_DIM + head_base + offsets,
        mask=mask,
        other=0.0,
    )
    tl.store(value_output + output_base + offsets, values, mask=mask)

    request = tl.load(row_request_ids + row)
    cache_position = tl.load(row_cache_positions + row)
    logical_page = cache_position // PAGE_SIZE
    page_offset = cache_position - logical_page * PAGE_SIZE
    physical_page = tl.load(block_tables + request * MAX_PAGES + logical_page)
    cache_base = ((physical_page * HEADS + head) * PAGE_SIZE + page_offset) * HEAD_DIM
    tl.store(k_cache + cache_base + offsets, rotated, mask=mask)
    tl.store(v_cache + cache_base + offsets, values, mask=mask)


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    values = 1
    for dim in shape:
        values *= int(dim)
    return values * element_bytes
