from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import triton
import triton.language as tl

from b12x_mla_capture import _bf16_tensor, _i32_tensor, _u8_tensor
from dflash_tuning_profile import dflash2_body_num_warps
from dspark_capture import _fp8_tensor, _i64_tensor

_DSPARK_BODY_STATES: dict[tuple[Any, ...], "_DsparkBodyState"] = {}


@dataclass(frozen=True)
class _DsparkBodyLayerWeights:
    input_norm: Any
    post_norm: Any
    q_norm: Any
    k_norm: Any
    qkv_t: Any
    output_t: Any
    gate_up_t: Any
    down_t: Any
    attention_conv_base: Any | None
    attention_conv_projection_t: Any | None
    mlp_conv_base: Any | None
    mlp_conv_projection_t: Any | None


@dataclass(frozen=True)
class _DsparkBodyState:
    device_id: int
    cuda_stream: int
    layers: int
    active_requests: int
    query_rows: int
    total_rows: int
    total_pages: int
    page_size: int
    max_pages_per_request: int
    hidden_size: int
    intermediate_size: int
    heads: int
    kv_heads: int
    head_dim: int
    rope_theta: float
    conv_group_size: int
    sliding_window: int
    cache_dtype: str
    input: Any
    output: Any
    reference_output: Any
    hidden_attention: Any
    hidden_mlp: Any
    normalized: Any
    qkv: Any
    q: Any
    attention: Any
    attention_flat: Any
    delta: Any
    gate_up: Any
    activation: Any
    conv_dynamic: Any | None
    conv_output: Any | None
    k_cache: Any
    v_cache: Any
    workspace: Any
    query_lengths: Any
    kv_lengths: Any
    query_positions: Any
    block_tables: Any
    query_offsets: Any
    output_offsets: Any
    query_indptr: Any
    kv_indptr: Any
    page_indices: Any
    last_page_len: Any
    flashinfer_wrapper: Any | None
    final_norm: Any
    layer_weights: tuple[_DsparkBodyLayerWeights, ...]


def prepare_dspark_cudnn_paged_body(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Bind, initialize, compile, and warm one fixed-address dSpark body."""

    state = _body_state(ctx, kwargs, create=True)
    _run_body(state)
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        state.reference_output.copy_(state.output)


def capture_dspark_cudnn_paged_body(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch the allocation-free active draft-layer body during capture."""

    _run_body(_body_state(ctx, kwargs, create=False))


def _body_state(
    ctx: dict[str, Any], kwargs: dict[str, Any], *, create: bool
) -> _DsparkBodyState:
    layers = int(kwargs["layers"])
    active_requests = int(kwargs["active_requests"])
    query_rows = int(kwargs["query_rows"])
    total_pages = int(kwargs["total_pages"])
    page_size = int(kwargs["page_size"])
    max_pages_per_request = int(kwargs["max_pages_per_request"])
    hidden_size = int(kwargs["hidden_size"])
    intermediate_size = int(kwargs["intermediate_size"])
    heads = int(kwargs["heads"])
    kv_heads = int(kwargs.get("kv_heads", heads))
    head_dim = int(kwargs["head_dim"])
    rope_theta = float(kwargs.get("rope_theta", 8_000_000.0))
    conv_group_size = int(kwargs.get("conv_group_size", 0))
    sliding_window = int(kwargs.get("sliding_window", -1))
    seed = int(kwargs["seed"])
    initialize_input = bool(kwargs.get("initialize_input", True))
    initialize_kv = bool(kwargs.get("initialize_kv", True))
    cache_dtype = str(kwargs.get("cache_dtype", "bf16")).lower()
    planning_pages_per_request = int(
        kwargs.get("planning_pages_per_request", max_pages_per_request)
    )
    fixed_split_pages = int(kwargs.get("fixed_split_pages", 0))

    dspark_geometry = (
        layers in (3, 5)
        and heads == 64
        and kv_heads == 64
        and head_dim == 64
        and rope_theta == 8_000_000.0
        and conv_group_size == 0
        and sliding_window == -1
    )
    dflash2_geometry = (
        layers == 6
        and heads == 64
        and kv_heads == 8
        and head_dim == 128
        and rope_theta == 1_000_000.0
        and conv_group_size == 16
        and sliding_window == 2_048
    )
    if not (dspark_geometry or dflash2_geometry):
        raise ValueError(
            "draft body geometry is neither GLM-5.2 dSpark nor GLM-5.3 DFlash2"
        )
    if active_requests not in (1, 2, 4):
        raise ValueError(
            "dSpark body active_requests must be one of 1, 2, or 4, "
            f"got {active_requests}"
        )
    if query_rows not in ((8, 16) if dspark_geometry else tuple(range(2, 9))):
        raise ValueError(f"unsupported draft body query_rows {query_rows}")
    if page_size not in (16, 32, 64, 128):
        raise ValueError(f"unsupported dSpark body page size {page_size}")
    if hidden_size != 6144 or intermediate_size != 12288:
        raise ValueError(
            "GLM-5.2 dSpark body requires hidden/intermediate 6144/12288, "
            f"got {hidden_size}/{intermediate_size}"
        )
    if total_pages < active_requests or max_pages_per_request < 1:
        raise ValueError("dSpark body page counts are invalid")
    if not 1 <= planning_pages_per_request <= max_pages_per_request:
        raise ValueError(
            "draft body planning pages per request must fit the page-table capacity"
        )
    if fixed_split_pages < 0:
        raise ValueError("draft body fixed split pages cannot be negative")
    if fixed_split_pages and not dflash2_geometry:
        raise ValueError("fixed split pages are supported only for DFlash2")
    if cache_dtype not in ("bf16", "fp8"):
        raise ValueError(f"unsupported dSpark body cache dtype {cache_dtype}")

    buffers = ctx["buffers"]
    mutable_names = [
        "input",
        "output",
        "reference_output",
        "hidden_attention",
        "hidden_mlp",
        "normalized",
        "qkv",
        "q",
        "attention",
        "delta",
        "gate_up",
        "activation",
        "k_cache",
        "v_cache",
        "workspace",
        "query_lengths",
        "kv_lengths",
        "query_positions",
        "block_tables",
        "query_offsets",
        "output_offsets",
        "query_indptr",
        "kv_indptr",
        "page_indices",
        "last_page_len",
    ]
    if dflash2_geometry:
        mutable_names.extend(("conv_dynamic", "conv_output"))
    weight_names = ["final_norm"]
    for layer in range(layers):
        weight_names.extend(
            (
                f"layer_{layer}_input_norm",
                f"layer_{layer}_post_norm",
                f"layer_{layer}_q_norm",
                f"layer_{layer}_k_norm",
                f"layer_{layer}_qkv",
                f"layer_{layer}_output",
                f"layer_{layer}_gate_up",
                f"layer_{layer}_down",
            )
        )
        if dflash2_geometry:
            weight_names.extend(
                (
                    f"layer_{layer}_attention_conv_base",
                    f"layer_{layer}_attention_conv_projection",
                    f"layer_{layer}_mlp_conv_base",
                    f"layer_{layer}_mlp_conv_projection",
                )
            )
    required_names = (*mutable_names, *weight_names)
    missing = [name for name in required_names if name not in buffers]
    if missing:
        raise ValueError(f"dSpark body is missing buffers: {missing}")

    device_id = int(buffers["input"]["device_id"])
    for name in required_names:
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(f"dSpark body buffer {name} is on another CUDA device")

    total_rows = active_requests * query_rows
    attention_width = heads * head_dim
    kv_width = kv_heads * head_dim
    qkv_width = attention_width + 2 * kv_width
    conv_groups = hidden_size // conv_group_size if conv_group_size else 0
    conv_projection_width = 4 * conv_groups
    shapes = {
        "input": (total_rows, hidden_size),
        "output": (total_rows, hidden_size),
        "reference_output": (total_rows, hidden_size),
        "hidden_attention": (total_rows, hidden_size),
        "hidden_mlp": (total_rows, hidden_size),
        "normalized": (total_rows, hidden_size),
        "qkv": (total_rows, qkv_width),
        "q": (total_rows, heads, head_dim),
        "attention": (total_rows, heads, head_dim),
        "delta": (total_rows, hidden_size),
        "gate_up": (total_rows, 2 * intermediate_size),
        "activation": (total_rows, intermediate_size),
        "k_cache": (layers, total_pages, kv_heads, page_size, head_dim),
        "v_cache": (layers, total_pages, kv_heads, page_size, head_dim),
        "workspace": (int(buffers["workspace"]["bytes"]),),
        "query_lengths": (active_requests,),
        "kv_lengths": (active_requests,),
        "query_positions": (total_rows,),
        "block_tables": (active_requests, max_pages_per_request),
        "query_offsets": (active_requests + 1,),
        "output_offsets": (active_requests + 1,),
        "query_indptr": (active_requests + 1,),
        "kv_indptr": (active_requests + 1,),
        "page_indices": (total_pages,),
        "last_page_len": (active_requests,),
        "final_norm": (hidden_size,),
    }
    if dflash2_geometry:
        shapes["conv_dynamic"] = (total_rows, conv_projection_width)
        shapes["conv_output"] = (total_rows, hidden_size)
    for name, shape in shapes.items():
        element_bytes = 1 if name == "workspace" else 2
        if name in (
            "query_lengths",
            "kv_lengths",
            "query_positions",
            "block_tables",
            "query_indptr",
            "kv_indptr",
            "page_indices",
            "last_page_len",
        ):
            element_bytes = 4
        elif name in ("query_offsets", "output_offsets"):
            element_bytes = 8
        elif name in ("k_cache", "v_cache") and cache_dtype == "fp8":
            element_bytes = 1
        required_bytes = _tensor_bytes(shape, element_bytes)
        if int(buffers[name]["bytes"]) < required_bytes:
            raise ValueError(
                f"dSpark body buffer {name} has {buffers[name]['bytes']} bytes, "
                f"requires {required_bytes}"
            )

    key = (
        int(ctx["cuda_stream"]),
        layers,
        active_requests,
        query_rows,
        total_pages,
        page_size,
        max_pages_per_request,
        hidden_size,
        intermediate_size,
        heads,
        kv_heads,
        head_dim,
        rope_theta,
        conv_group_size,
        sliding_window,
        cache_dtype,
        initialize_input,
        initialize_kv,
        planning_pages_per_request,
        fixed_split_pages,
        *((name, int(buffers[name]["ptr"])) for name in required_names),
    )
    state = _DSPARK_BODY_STATES.get(key)
    if state is not None:
        return state
    if not create:
        raise RuntimeError(
            "dSpark body capture requires a matching startup prepare call"
        )

    import torch

    cuda_stream = int(ctx["cuda_stream"])
    stream = torch.cuda.ExternalStream(cuda_stream, device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream), torch.no_grad():
        tensors: dict[str, Any] = {}
        for name in (
            "input",
            "output",
            "reference_output",
            "hidden_attention",
            "hidden_mlp",
            "normalized",
            "qkv",
            "q",
            "attention",
            "delta",
            "gate_up",
            "activation",
            "final_norm",
        ):
            tensors[name] = _bf16_tensor(buffers[name], shapes[name])
        if dflash2_geometry:
            tensors["conv_dynamic"] = _bf16_tensor(
                buffers["conv_dynamic"], shapes["conv_dynamic"]
            )
            tensors["conv_output"] = _bf16_tensor(
                buffers["conv_output"], shapes["conv_output"]
            )
        cache_tensor = _bf16_tensor if cache_dtype == "bf16" else _fp8_tensor
        tensors["k_cache"] = cache_tensor(buffers["k_cache"], shapes["k_cache"])
        tensors["v_cache"] = cache_tensor(buffers["v_cache"], shapes["v_cache"])
        workspace = _u8_tensor(buffers["workspace"], shapes["workspace"])
        query_lengths = _i32_tensor(buffers["query_lengths"], shapes["query_lengths"])
        kv_lengths = _i32_tensor(buffers["kv_lengths"], shapes["kv_lengths"])
        query_positions = _i32_tensor(
            buffers["query_positions"], shapes["query_positions"]
        )
        block_tables = _i32_tensor(buffers["block_tables"], shapes["block_tables"])
        query_offsets = _i64_tensor(buffers["query_offsets"], shapes["query_offsets"])
        output_offsets = _i64_tensor(
            buffers["output_offsets"], shapes["output_offsets"]
        )
        query_indptr = _i32_tensor(buffers["query_indptr"], shapes["query_indptr"])
        kv_indptr = _i32_tensor(buffers["kv_indptr"], shapes["kv_indptr"])
        page_indices = _i32_tensor(buffers["page_indices"], shapes["page_indices"])
        last_page_len = _i32_tensor(buffers["last_page_len"], shapes["last_page_len"])

        layer_weights = []
        for layer in range(layers):
            prefix = f"layer_{layer}"
            input_norm = _bf16_tensor(buffers[f"{prefix}_input_norm"], (hidden_size,))
            post_norm = _bf16_tensor(buffers[f"{prefix}_post_norm"], (hidden_size,))
            q_norm = _bf16_tensor(buffers[f"{prefix}_q_norm"], (head_dim,))
            k_norm = _bf16_tensor(buffers[f"{prefix}_k_norm"], (head_dim,))
            qkv = _bf16_tensor(buffers[f"{prefix}_qkv"], (qkv_width, hidden_size))
            output = _bf16_tensor(
                buffers[f"{prefix}_output"], (hidden_size, attention_width)
            )
            gate_up = _bf16_tensor(
                buffers[f"{prefix}_gate_up"],
                (2 * intermediate_size, hidden_size),
            )
            down = _bf16_tensor(
                buffers[f"{prefix}_down"], (hidden_size, intermediate_size)
            )
            attention_conv_base = None
            attention_conv_projection_t = None
            mlp_conv_base = None
            mlp_conv_projection_t = None
            if dflash2_geometry:
                attention_conv_base = _bf16_tensor(
                    buffers[f"{prefix}_attention_conv_base"],
                    (2, 2, hidden_size),
                )
                attention_conv_projection_t = _bf16_tensor(
                    buffers[f"{prefix}_attention_conv_projection"],
                    (conv_projection_width, hidden_size),
                ).t()
                mlp_conv_base = _bf16_tensor(
                    buffers[f"{prefix}_mlp_conv_base"],
                    (2, 2, hidden_size),
                )
                mlp_conv_projection_t = _bf16_tensor(
                    buffers[f"{prefix}_mlp_conv_projection"],
                    (conv_projection_width, hidden_size),
                ).t()
            layer_weights.append(
                _DsparkBodyLayerWeights(
                    input_norm=input_norm,
                    post_norm=post_norm,
                    q_norm=q_norm,
                    k_norm=k_norm,
                    qkv_t=qkv.t(),
                    output_t=output.t(),
                    gate_up_t=gate_up.t(),
                    down_t=down.t(),
                    attention_conv_base=attention_conv_base,
                    attention_conv_projection_t=attention_conv_projection_t,
                    mlp_conv_base=mlp_conv_base,
                    mlp_conv_projection_t=mlp_conv_projection_t,
                )
            )

        query_lengths.fill_(query_rows)
        element_stride = query_rows * heads * head_dim
        offsets = (
            torch.arange(active_requests + 1, dtype=torch.int64, device=device_id)
            * element_stride
        )
        query_offsets.copy_(offsets)
        output_offsets.copy_(offsets)

        if initialize_input or initialize_kv:
            generator = torch.Generator(device=device_id)
            generator.manual_seed(seed)
            if initialize_input:
                tensors["input"].normal_(generator=generator)
            if initialize_kv:
                if cache_dtype == "bf16":
                    tensors["k_cache"].normal_(generator=generator)
                    tensors["v_cache"].normal_(generator=generator)
                else:
                    for cache in (tensors["k_cache"], tensors["v_cache"]):
                        source = torch.empty(
                            cache.shape,
                            dtype=torch.bfloat16,
                            device=cache.device,
                        )
                        source.normal_(generator=generator)
                        cache.copy_(source.to(torch.float8_e4m3fn))

        flashinfer_wrapper = None
        if cache_dtype == "fp8" or sliding_window >= 0:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper

            runtime_query_indptr = query_indptr.clone()
            runtime_kv_indptr = kv_indptr.clone()
            runtime_page_indices = page_indices.clone()
            runtime_last_page_len = last_page_len.clone()
            # The underlying KV allocation can be a larger pool shared by
            # separately captured C1/C2/C4 executors.  Plan each executor for
            # an explicit logical page count, never for an equal division of
            # every physical page in the shared pool.  The latter made a C1
            # executor plan 136 pages for a request that could address only
            # 34, so FlashInfer's captured schedule read unrelated slots.
            planning_pages = active_requests * planning_pages_per_request
            if planning_pages > total_pages:
                raise ValueError(
                    "dSpark body planning pages exceed the physical KV pool"
                )
            planning_query_indptr = (
                torch.arange(active_requests + 1, dtype=torch.int32, device=device_id)
                * query_rows
            )
            planning_kv_indptr = (
                torch.arange(active_requests + 1, dtype=torch.int32, device=device_id)
                * planning_pages_per_request
            )
            planning_page_indices = torch.arange(
                planning_pages, dtype=torch.int32, device=device_id
            )
            planning_last_page_len = torch.full(
                (active_requests,), page_size, dtype=torch.int32, device=device_id
            )
            flashinfer_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                workspace,
                kv_layout="HND",
                use_cuda_graph=True,
                qo_indptr_buf=query_indptr,
                paged_kv_indptr_buf=kv_indptr,
                paged_kv_indices_buf=page_indices,
                paged_kv_last_page_len_buf=last_page_len,
                # cuDNN paged prefill does not expose the symmetric-window
                # contract required by DFlash2. Force a FlashAttention
                # backend for that geometry; the right side can contain only
                # the seven current noise rows, all within the 2,048 window.
                # FlashInfer's separate SM120 TRT-LLM FMHA-v2 path is not an
                # equivalent shortcut: it supports full noncausal padding or
                # causal sliding-window masking, but not this noncausal block
                # with a bounded left window.
                backend="fa2" if sliding_window >= 0 else "auto",
            )
            kv_data_type = (
                torch.bfloat16 if cache_dtype == "bf16" else torch.float8_e4m3fn
            )
            flashinfer_wrapper.plan(
                planning_query_indptr,
                planning_kv_indptr,
                planning_page_indices,
                planning_last_page_len,
                heads,
                kv_heads,
                head_dim,
                page_size,
                causal=False,
                window_left=sliding_window - 1 if sliding_window >= 0 else -1,
                sm_scale=1.0 / math.sqrt(head_dim),
                q_data_type=torch.bfloat16,
                kv_data_type=kv_data_type,
                o_data_type=torch.bfloat16,
                # A fixed two-page split is used only by an exact C1 page-count
                # graph bucket.  Its number of chunks is invariant while the
                # final page fills.  The general graph is non-split because
                # FlashInfer otherwise bakes a length-dependent schedule into
                # plan_info that cannot be changed by updating graph metadata.
                fixed_split_size=fixed_split_pages or None,
                disable_split_kv=dflash2_geometry and fixed_split_pages == 0,
            )
            query_indptr.copy_(runtime_query_indptr)
            kv_indptr.copy_(runtime_kv_indptr)
            page_indices.copy_(runtime_page_indices)
            last_page_len.copy_(runtime_last_page_len)

    state = _DsparkBodyState(
        device_id=device_id,
        cuda_stream=cuda_stream,
        layers=layers,
        active_requests=active_requests,
        query_rows=query_rows,
        total_rows=total_rows,
        total_pages=total_pages,
        page_size=page_size,
        max_pages_per_request=max_pages_per_request,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        rope_theta=rope_theta,
        conv_group_size=conv_group_size,
        sliding_window=sliding_window,
        cache_dtype=cache_dtype,
        input=tensors["input"],
        output=tensors["output"],
        reference_output=tensors["reference_output"],
        hidden_attention=tensors["hidden_attention"],
        hidden_mlp=tensors["hidden_mlp"],
        normalized=tensors["normalized"],
        qkv=tensors["qkv"],
        q=tensors["q"],
        attention=tensors["attention"],
        attention_flat=tensors["attention"].view(total_rows, attention_width),
        delta=tensors["delta"],
        gate_up=tensors["gate_up"],
        activation=tensors["activation"],
        conv_dynamic=tensors.get("conv_dynamic"),
        conv_output=tensors.get("conv_output"),
        k_cache=tensors["k_cache"],
        v_cache=tensors["v_cache"],
        workspace=workspace,
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        query_positions=query_positions,
        block_tables=block_tables,
        query_offsets=query_offsets,
        output_offsets=output_offsets,
        query_indptr=query_indptr,
        kv_indptr=kv_indptr,
        page_indices=page_indices,
        last_page_len=last_page_len,
        flashinfer_wrapper=flashinfer_wrapper,
        final_norm=tensors["final_norm"],
        layer_weights=tuple(layer_weights),
    )
    _DSPARK_BODY_STATES[key] = state
    return state


def _run_body(
    state: _DsparkBodyState,
    trace: dict[str, Any] | None = None,
) -> None:
    import torch

    def record(name: str, value: Any) -> None:
        if trace is not None:
            trace[name] = value.detach().clone()

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    max_sequence_kv = state.max_pages_per_request * state.page_size
    scale = 1.0 / math.sqrt(state.head_dim)
    hidden = state.input
    fuse_dflash2_dynamic_residual_norm = state.conv_group_size != 0
    residual_add_grid = (triton.cdiv(state.total_rows * state.hidden_size, 256),)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        if fuse_dflash2_dynamic_residual_norm:
            _dspark_rms_norm[(state.total_rows,)](
                hidden,
                state.layer_weights[0].input_norm,
                state.normalized,
                WIDTH=state.hidden_size,
                BLOCK=triton.next_power_of_2(state.hidden_size),
                EPSILON=1.0e-5,
                num_warps=8,
            )
            record("layer_0.input_norm", state.normalized)
        for layer_index, weights in enumerate(state.layer_weights):
            if not fuse_dflash2_dynamic_residual_norm:
                _dspark_rms_norm[(state.total_rows,)](
                    hidden,
                    weights.input_norm,
                    state.normalized,
                    WIDTH=state.hidden_size,
                    BLOCK=triton.next_power_of_2(state.hidden_size),
                    EPSILON=1.0e-5,
                    num_warps=8,
                )
                record(f"layer_{layer_index}.input_norm", state.normalized)
            qkv_input = state.normalized
            if weights.attention_conv_projection_t is not None:
                _prepare_dynamic_conv(
                    state,
                    state.normalized,
                    weights.attention_conv_projection_t,
                    weights.attention_conv_base,
                )
                qkv_input = state.conv_output
            record(f"layer_{layer_index}.attention_input", qkv_input)
            torch.mm(qkv_input, weights.qkv_t, out=state.qkv)
            record(f"layer_{layer_index}.qkv_projection", state.qkv)
            _dspark_qkv_rope_append[(state.total_rows * state.heads,)](
                state.qkv,
                weights.q_norm,
                weights.k_norm,
                state.kv_lengths,
                state.query_positions,
                state.block_tables,
                state.q,
                state.k_cache[layer_index],
                state.v_cache[layer_index],
                QUERY_ROWS=state.query_rows,
                HEADS=state.heads,
                KV_HEADS=state.kv_heads,
                HEAD_DIM=state.head_dim,
                PAGE_SIZE=state.page_size,
                MAX_PAGES=state.max_pages_per_request,
                QKV_WIDTH=(state.heads + 2 * state.kv_heads) * state.head_dim,
                THETA=state.rope_theta,
                EPSILON=1.0e-5,
                BLOCK=state.head_dim,
                num_warps=4,
            )
            record(f"layer_{layer_index}.q_rope", state.q)
            if state.flashinfer_wrapper is not None:
                state.flashinfer_wrapper.run(
                    state.q,
                    (state.k_cache[layer_index], state.v_cache[layer_index]),
                    out=state.attention,
                )
            else:
                from flashinfer.cudnn.prefill import (
                    cudnn_batch_prefill_with_kv_cache,
                )

                cudnn_batch_prefill_with_kv_cache(
                    state.q,
                    state.k_cache[layer_index],
                    state.v_cache[layer_index],
                    scale,
                    state.workspace,
                    max_token_per_sequence=state.query_rows,
                    max_sequence_kv=max_sequence_kv,
                    actual_seq_lens_q=state.query_lengths,
                    actual_seq_lens_kv=state.kv_lengths,
                    block_tables=state.block_tables,
                    causal=False,
                    return_lse=False,
                    batch_offsets_q=state.query_offsets,
                    batch_offsets_o=state.output_offsets,
                    out=state.attention,
                    is_cuda_graph_compatible=True,
                )
            record(f"layer_{layer_index}.attention_values", state.attention)
            torch.mm(state.attention_flat, weights.output_t, out=state.delta)
            record(f"layer_{layer_index}.attention_output", state.delta)
            if fuse_dflash2_dynamic_residual_norm:
                _finish_dynamic_conv_add_rms_norm(
                    state,
                    state.delta,
                    weights.attention_conv_base,
                    hidden,
                    weights.post_norm,
                    state.hidden_attention,
                    state.normalized,
                )
            else:
                _dspark_add[residual_add_grid](
                    hidden,
                    state.delta,
                    state.hidden_attention,
                    TOTAL=state.total_rows * state.hidden_size,
                    BLOCK=256,
                )
                _dspark_rms_norm[(state.total_rows,)](
                    state.hidden_attention,
                    weights.post_norm,
                    state.normalized,
                    WIDTH=state.hidden_size,
                    BLOCK=triton.next_power_of_2(state.hidden_size),
                    EPSILON=1.0e-5,
                    num_warps=8,
                )
            record(f"layer_{layer_index}.attention_residual", state.hidden_attention)
            record(f"layer_{layer_index}.post_norm", state.normalized)
            mlp_input = state.normalized
            if weights.mlp_conv_projection_t is not None:
                _prepare_dynamic_conv(
                    state,
                    state.normalized,
                    weights.mlp_conv_projection_t,
                    weights.mlp_conv_base,
                )
                mlp_input = state.conv_output
            record(f"layer_{layer_index}.mlp_input", mlp_input)
            torch.mm(mlp_input, weights.gate_up_t, out=state.gate_up)
            record(f"layer_{layer_index}.gate_up_projection", state.gate_up)
            _dspark_silu_mul[
                (triton.cdiv(state.total_rows * state.intermediate_size, 256),)
            ](
                state.gate_up,
                state.activation,
                ROWS=state.total_rows,
                INTERMEDIATE=state.intermediate_size,
                BLOCK=256,
            )
            record(f"layer_{layer_index}.activation", state.activation)
            torch.mm(state.activation, weights.down_t, out=state.delta)
            record(f"layer_{layer_index}.mlp_output", state.delta)
            if fuse_dflash2_dynamic_residual_norm:
                final_layer = layer_index + 1 == state.layers
                next_norm = (
                    state.final_norm
                    if final_layer
                    else state.layer_weights[layer_index + 1].input_norm
                )
                normalized_output = state.output if final_layer else state.normalized
                _finish_dynamic_conv_add_rms_norm(
                    state,
                    state.delta,
                    weights.mlp_conv_base,
                    state.hidden_attention,
                    next_norm,
                    state.hidden_mlp,
                    normalized_output,
                )
                if not final_layer:
                    record(f"layer_{layer_index + 1}.input_norm", state.normalized)
            else:
                _dspark_add[residual_add_grid](
                    state.hidden_attention,
                    state.delta,
                    state.hidden_mlp,
                    TOTAL=state.total_rows * state.hidden_size,
                    BLOCK=256,
                )
            record(f"layer_{layer_index}.output", state.hidden_mlp)
            hidden = state.hidden_mlp

        if not fuse_dflash2_dynamic_residual_norm:
            _dspark_rms_norm[(state.total_rows,)](
                hidden,
                state.final_norm,
                state.output,
                WIDTH=state.hidden_size,
                BLOCK=triton.next_power_of_2(state.hidden_size),
                EPSILON=1.0e-5,
                num_warps=8,
            )


def _prepare_dynamic_conv(
    state: _DsparkBodyState,
    source: Any,
    projection_t: Any,
    base: Any,
) -> None:
    import torch

    if state.conv_dynamic is None or state.conv_output is None or base is None:
        raise RuntimeError("DFlash2 dynamic-convolution buffers are not initialized")
    torch.mm(source, projection_t, out=state.conv_dynamic)
    _dflash2_grouped_dynamic_conv[
        (triton.cdiv(state.total_rows * state.hidden_size, 256),)
    ](
        source,
        state.conv_dynamic,
        base,
        state.conv_output,
        QUERY_ROWS=state.query_rows,
        TOTAL_VALUES=state.total_rows * state.hidden_size,
        HIDDEN_SIZE=state.hidden_size,
        GROUP_SIZE=state.conv_group_size,
        SIDE=0,
        BLOCK=256,
    )


def _finish_dynamic_conv_add_rms_norm(
    state: _DsparkBodyState,
    source: Any,
    base: Any,
    residual: Any,
    norm_weight: Any,
    residual_output: Any,
    normalized_output: Any,
) -> None:
    if state.conv_dynamic is None or state.conv_output is None or base is None:
        raise RuntimeError("DFlash2 dynamic-convolution buffers are not initialized")
    _dflash2_finish_dynamic_conv_add_rms_norm[(state.total_rows,)](
        source,
        state.conv_dynamic,
        base,
        residual,
        norm_weight,
        residual_output,
        normalized_output,
        QUERY_ROWS=state.query_rows,
        WIDTH=state.hidden_size,
        GROUP_SIZE=state.conv_group_size,
        BLOCK=triton.next_power_of_2(state.hidden_size),
        EPSILON=1.0e-5,
        num_warps=dflash2_body_num_warps(
            state.active_requests,
            state.query_rows - 1,
        ),
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
    values = tl.load(source + row * WIDTH + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    variance = tl.sum(values * values, axis=0) / WIDTH
    inverse_rms = tl.rsqrt(variance + EPSILON)
    scales = tl.load(weight + offsets, mask=mask, other=0.0).to(tl.float32)
    # Qwen3RMSNorm casts the unit-RMS value back to the input dtype before
    # applying the learned weight. Keep that BF16 boundary: retaining both
    # multiplies in FP32 changes DFlash2's draft choices after six layers.
    normalized = (values * inverse_rms).to(tl.bfloat16)
    scaled = (normalized.to(tl.float32) * scales).to(tl.bfloat16)
    tl.store(output + row * WIDTH + offsets, scaled, mask=mask)


@triton.jit
def _dspark_add(
    left,
    right,
    output,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < TOTAL
    left_values = tl.load(left + offsets, mask=mask, other=0.0)
    right_values = tl.load(right + offsets, mask=mask, other=0.0)
    tl.store(output + offsets, left_values + right_values, mask=mask)


@triton.jit
def _dflash2_finish_dynamic_conv_add_rms_norm(
    source,
    dynamic,
    base,
    residual,
    norm_weight,
    residual_output,
    normalized_output,
    QUERY_ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPSILON: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK)
    mask = columns < WIDTH
    offsets = row * WIDTH + columns
    groups = WIDTH // GROUP_SIZE
    group = columns // GROUP_SIZE
    dynamic_base = row * (4 * groups) + 2 * groups + group

    current = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    base_current = tl.load(base + 2 * WIDTH + columns, mask=mask, other=0.0).to(
        tl.float32
    )
    dynamic_current = tl.load(dynamic + dynamic_base, mask=mask, other=0.0).to(
        tl.float32
    )
    # Preserve the four BF16 statements used by the upstream implementation:
    # each base-kernel accumulation and each dynamic addcmul rounds back to the
    # hidden dtype before the next causal tap is accumulated. Reassociating the
    # two coefficients in FP32 changes draft logits enough to destroy greedy
    # acceptance after the error compounds through the six-layer body.
    convolved = (current * base_current).to(tl.bfloat16)
    convolved = (
        convolved.to(tl.float32) + current * dynamic_current
    ).to(tl.bfloat16)

    previous_mask = mask & (row % QUERY_ROWS > 0)
    previous = tl.load(source + offsets - WIDTH, mask=previous_mask, other=0.0).to(
        tl.float32
    )
    base_previous = tl.load(
        base + 3 * WIDTH + columns, mask=previous_mask, other=0.0
    ).to(tl.float32)
    dynamic_previous = tl.load(
        dynamic + dynamic_base + groups,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    previous_base = (previous * base_previous).to(tl.bfloat16)
    convolved = (
        convolved.to(tl.float32) + previous_base.to(tl.float32)
    ).to(tl.bfloat16)
    convolved = (
        convolved.to(tl.float32) + previous * dynamic_previous
    ).to(tl.bfloat16)
    # Preserve the finish-convolution and residual-add BF16 boundaries from
    # the original three-kernel sequence before normalizing in registers. The
    # finish-convolution value has no consumer outside this fused operation,
    # so retaining its rounded value in registers avoids a dead materialized
    # write without changing arithmetic.
    residual_values = (
        tl.load(residual + offsets, mask=mask, other=0.0).to(tl.float32)
        + convolved.to(tl.float32)
    ).to(tl.bfloat16)
    tl.store(residual_output + offsets, residual_values, mask=mask)
    values = residual_values.to(tl.float32)
    variance = tl.sum(values * values, axis=0) / WIDTH
    inverse_rms = tl.rsqrt(variance + EPSILON)
    scales = tl.load(norm_weight + columns, mask=mask, other=0.0).to(tl.float32)
    normalized = (values * inverse_rms).to(tl.bfloat16)
    scaled = (normalized.to(tl.float32) * scales).to(tl.bfloat16)
    tl.store(
        normalized_output + offsets,
        scaled,
        mask=mask,
    )


@triton.jit
def _dspark_silu_mul(
    gate_up,
    activation,
    ROWS: tl.constexpr,
    INTERMEDIATE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = ROWS * INTERMEDIATE
    mask = offsets < total
    row = offsets // INTERMEDIATE
    column = offsets - row * INTERMEDIATE
    gate_index = row * (2 * INTERMEDIATE) + column
    gate = tl.load(gate_up + gate_index, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(gate_up + gate_index + INTERMEDIATE, mask=mask, other=0.0).to(
        tl.float32
    )
    silu = (gate * tl.sigmoid(gate)).to(tl.bfloat16)
    activated = (silu.to(tl.float32) * up).to(tl.bfloat16)
    tl.store(activation + offsets, activated, mask=mask)


@triton.jit
def _dflash2_grouped_dynamic_conv(
    source,
    dynamic,
    base,
    output,
    QUERY_ROWS: tl.constexpr,
    TOTAL_VALUES: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    SIDE: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = offsets // HIDDEN_SIZE
    column = offsets - row * HIDDEN_SIZE
    mask = offsets < TOTAL_VALUES
    groups = HIDDEN_SIZE // GROUP_SIZE
    group = column // GROUP_SIZE
    dynamic_base = row * (4 * groups) + SIDE * (2 * groups) + group

    current = tl.load(source + offsets, mask=mask, other=0.0).to(tl.float32)
    base_current = tl.load(
        base + (SIDE * 2) * HIDDEN_SIZE + column, mask=mask, other=0.0
    ).to(tl.float32)
    dynamic_current = tl.load(dynamic + dynamic_base, mask=mask, other=0.0).to(
        tl.float32
    )
    # Match upstream's two BF16 statements for each causal offset:
    #   output = output + kernel * values
    #   output = torch.addcmul(output, dynamic, values)
    # Reassociating this as values * (kernel + dynamic) in FP32 creates small
    # errors at every draft layer and materially changes DFlash2 proposals.
    values = (current * base_current).to(tl.bfloat16)
    values = (
        values.to(tl.float32) + current * dynamic_current
    ).to(tl.bfloat16)

    local_row = row % QUERY_ROWS
    previous_mask = mask & (local_row > 0)
    previous = tl.load(
        source + offsets - HIDDEN_SIZE, mask=previous_mask, other=0.0
    ).to(tl.float32)
    base_previous = tl.load(
        base + (SIDE * 2 + 1) * HIDDEN_SIZE + column,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    dynamic_previous = tl.load(
        dynamic + dynamic_base + groups,
        mask=previous_mask,
        other=0.0,
    ).to(tl.float32)
    previous_base = (previous * base_previous).to(tl.bfloat16)
    values = (
        values.to(tl.float32) + previous_base.to(tl.float32)
    ).to(tl.bfloat16)
    values = (
        values.to(tl.float32) + previous * dynamic_previous
    ).to(tl.bfloat16)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def _dspark_qkv_rope_append(
    qkv,
    q_norm,
    k_norm,
    kv_lengths,
    query_positions,
    block_tables,
    q_output,
    k_cache,
    v_cache,
    QUERY_ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    MAX_PAGES: tl.constexpr,
    QKV_WIDTH: tl.constexpr,
    THETA: tl.constexpr,
    EPSILON: tl.constexpr,
    BLOCK: tl.constexpr,
) -> None:
    program = tl.program_id(0)
    token_row = program // HEADS
    head = program - token_row * HEADS
    request = token_row // QUERY_ROWS
    query_row = token_row - request * QUERY_ROWS
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM
    kv_mask = mask & (head < KV_HEADS)

    row_base = token_row * QKV_WIDTH
    head_base = head * HEAD_DIM
    q_values = tl.load(qkv + row_base + head_base + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    k_values = tl.load(
        qkv + row_base + HEADS * HEAD_DIM + head_base + offsets,
        mask=kv_mask,
        other=0.0,
    ).to(tl.float32)
    q_variance = tl.sum(q_values * q_values, axis=0) / HEAD_DIM
    k_variance = tl.sum(k_values * k_values, axis=0) / HEAD_DIM
    q_scales = tl.load(q_norm + offsets, mask=mask, other=0.0).to(tl.float32)
    k_scales = tl.load(k_norm + offsets, mask=kv_mask, other=0.0).to(tl.float32)
    q_values = (q_values * tl.rsqrt(q_variance + EPSILON)).to(tl.bfloat16)
    q_values = (q_values.to(tl.float32) * q_scales).to(tl.bfloat16)
    k_values = (k_values * tl.rsqrt(k_variance + EPSILON)).to(tl.bfloat16)
    k_values = (k_values.to(tl.float32) * k_scales).to(tl.bfloat16)

    half = HEAD_DIM // 2
    pair = offsets % half
    paired_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    q_pair = tl.load(
        qkv + row_base + head_base + paired_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    k_pair = tl.load(
        qkv + row_base + HEADS * HEAD_DIM + head_base + paired_offsets,
        mask=kv_mask,
        other=0.0,
    ).to(tl.float32)
    q_pair_scales = tl.load(q_norm + paired_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    k_pair_scales = tl.load(k_norm + paired_offsets, mask=kv_mask, other=0.0).to(
        tl.float32
    )
    q_pair = (q_pair * tl.rsqrt(q_variance + EPSILON)).to(tl.bfloat16)
    q_pair = (q_pair.to(tl.float32) * q_pair_scales).to(tl.bfloat16)
    k_pair = (k_pair * tl.rsqrt(k_variance + EPSILON)).to(tl.bfloat16)
    k_pair = (k_pair.to(tl.float32) * k_pair_scales).to(tl.bfloat16)

    kv_length = tl.load(kv_lengths + request)
    position = tl.load(query_positions + token_row)
    frequency = tl.exp((-math.log(THETA) * (2.0 * pair)) / HEAD_DIM)
    angle = position.to(tl.float32) * frequency
    cosine = tl.cos(angle).to(tl.bfloat16)
    sine = tl.sin(angle).to(tl.bfloat16)
    q_cosine = (q_values.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
    q_sine = (q_pair.to(tl.float32) * sine.to(tl.float32)).to(tl.bfloat16)
    k_cosine = (k_values.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
    k_sine = (k_pair.to(tl.float32) * sine.to(tl.float32)).to(tl.bfloat16)
    q_rotated = tl.where(
        offsets < half,
        (q_cosine.to(tl.float32) - q_sine.to(tl.float32)).to(tl.bfloat16),
        (q_cosine.to(tl.float32) + q_sine.to(tl.float32)).to(tl.bfloat16),
    ).to(tl.bfloat16)
    k_rotated = tl.where(
        offsets < half,
        (k_cosine.to(tl.float32) - k_sine.to(tl.float32)).to(tl.bfloat16),
        (k_cosine.to(tl.float32) + k_sine.to(tl.float32)).to(tl.bfloat16),
    ).to(tl.bfloat16)
    tl.store(q_output + token_row * HEADS * HEAD_DIM + head_base + offsets, q_rotated)

    cache_position = kv_length - QUERY_ROWS + query_row
    logical_page = cache_position // PAGE_SIZE
    page_offset = cache_position - logical_page * PAGE_SIZE
    physical_page = tl.load(block_tables + request * MAX_PAGES + logical_page)
    cache_base = (
        (physical_page * KV_HEADS + head) * PAGE_SIZE + page_offset
    ) * HEAD_DIM
    tl.store(k_cache + cache_base + offsets, k_rotated, mask=kv_mask)
    v_values = tl.load(
        qkv + row_base + (HEADS + KV_HEADS) * HEAD_DIM + head_base + offsets,
        mask=kv_mask,
        other=0.0,
    )
    tl.store(v_cache + cache_base + offsets, v_values, mask=kv_mask)


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    values = 1
    for dim in shape:
        values *= int(dim)
    return values * element_bytes
