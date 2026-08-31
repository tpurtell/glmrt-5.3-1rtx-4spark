from __future__ import annotations

import ctypes
import os
from functools import cache
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_DLPACK_DEVICE_CUDA = 2
_DLPACK_CODE_INT = 0
_DLPACK_CODE_UINT = 1
_DLPACK_CODE_FLOAT = 2
_DLPACK_CODE_BFLOAT = 4
_GLM_NSA_NOPE_DIM = 512
_GLM_NSA_ROPE_DIM = 64
_GLM_NSA_V_DIM = 512
_GLM_NSA_HEADS = 8
_GLM_NSA_TOPK = 512
_GLM_ATTENTION_HEADS = (16, 64)
_GLM_DSA_INDEX_HEADS = 32
_GLM_DSA_INDEX_HEAD_DIM = 128
_GLM_DSA_TOPK = 2048
_GLM_DSA_PAGE_SIZE = 64
_GLM_DSA_PACKED_PAGE_BYTES = _GLM_DSA_PAGE_SIZE * (_GLM_DSA_INDEX_HEAD_DIM + 4)
_TARGET_ENV = "GLMRT_B12X_MLA_CAPTURE_TARGET"
_DLPACK_OWNERS: dict[int, Any] = {}
_DLPACK_BRIDGE: tuple[Any, Any] | None = None
_FLASHINFER_MODULES: dict[tuple[Any, ...], Any] = {}
_FLASHINFER_PREPARED_SHAPES: set[tuple[Any, ...]] = set()
_FLASHINFER_COMPRESSED_MLA_RUNNERS: dict[tuple[Any, ...], Any] = {}
_FLASHINFER_COMPRESSED_MLA_PREPARED: set[tuple[Any, ...]] = set()
_FLASHINFER_PACKED_FP8_MLA_PREPARED: set[tuple[Any, ...]] = set()
_FLASHINFER_PACKED_FP8_MLA_PREFILL_PREPARED: set[tuple[Any, ...]] = set()
_SPARKINFER_NVFP4_MLA_DECODE_PREPARED: set[tuple[Any, ...]] = set()
_SPARKINFER_NVFP4_MLA_PREFILL_PREPARED: set[tuple[Any, ...]] = set()
_SPARKINFER_GLM_H64_QUERY_PREPARED: set[tuple[int, int]] = set()
_SPARKINFER_GLM_H64_QUERY_LOGGED_PLANS: set[tuple[Any, ...]] = set()
_B12X_GLM_DSA_INDEXER_PLANS: dict[tuple[Any, ...], Any] = {}
_B12X_GLM_DSA_INDEXER_STATES: dict[tuple[Any, ...], Any] = {}
_B12X_GLM_DSA_INDEXER_PREPARED: set[tuple[Any, ...]] = set()
_SPARKINFER_GLM_H64_QUERY_POLICY_ENV = (
    "GLMRT_SPARKINFER_GLM_H64_BF16_QUERY_PROJECTION"
)


def plan_sparkinfer_glm_h64_bf16_query_projection(
    *,
    workload: str,
    query_rows: int,
    heads: int,
    nope_dim: int,
    latent_dim: int,
    device_id: int,
) -> bool:
    """Return SparkInfer's capture-static GLM H64 query-projection route."""

    import sys

    import torch
    from b12x.gemm.mla_query_projection import plan_glm_h64_bf16

    policy = os.environ.get(
        _SPARKINFER_GLM_H64_QUERY_POLICY_ENV, "auto"
    ).strip().lower()
    if policy not in {"auto", "force", "disable"}:
        raise ValueError(
            f"{_SPARKINFER_GLM_H64_QUERY_POLICY_ENV} must be auto, force, "
            f"or disable, got {policy!r}"
        )
    plan = plan_glm_h64_bf16(
        workload=workload,
        policy=policy,
        query_rows=int(query_rows),
        num_heads=int(heads),
        nope_dim=int(nope_dim),
        latent_dim=int(latent_dim),
        output_dtype=torch.bfloat16,
        device=torch.device("cuda", int(device_id)),
    )
    log_key = (
        int(device_id),
        workload,
        int(query_rows),
        policy,
        plan.backend,
        plan.reason,
    )
    if log_key not in _SPARKINFER_GLM_H64_QUERY_LOGGED_PLANS:
        print(
            "glmrt_sparkinfer_glm_h64_query_projection "
            f"device={device_id} workload={workload} rows={query_rows} "
            f"policy={policy} backend={plan.backend} reason={plan.reason}",
            file=sys.stderr,
        )
        _SPARKINFER_GLM_H64_QUERY_LOGGED_PLANS.add(log_key)
    return bool(plan.use_sparkinfer)


def prepare_sparkinfer_glm_h64_bf16_query_projection(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and first-launch SparkInfer's one-launch GLM H64 kernel."""

    _run_sparkinfer_glm_h64_bf16_query_projection(
        ctx, prepare_only=True, **kwargs
    )


def capture_sparkinfer_glm_h64_bf16_query_projection(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Project q-nope and append q-RoPE directly into the caller output."""

    _run_sparkinfer_glm_h64_bf16_query_projection(
        ctx, prepare_only=False, **kwargs
    )


def _run_sparkinfer_glm_h64_bf16_query_projection(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from b12x.gemm.mla_query_projection import (
        prewarm_glm_h64_bf16,
        run_glm_h64_bf16,
    )

    query_rows = int(kwargs["query_rows"])
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    latent_dim = int(kwargs["latent_dim"])
    weight_head_width = int(kwargs["weight_head_width"])
    if (
        heads != 64
        or nope_dim != 192
        or rope_dim != 64
        or latent_dim != 512
        or weight_head_width < nope_dim
        or not 1 <= query_rows <= 32
    ):
        raise ValueError(
            "SparkInfer GLM H64 BF16 query projection requires H=64, "
            "M=1..32, q-nope=192, q-RoPE=64, latent=512, and a resident "
            f"weight head width >=192; got H={heads}, M={query_rows}, "
            f"q-nope={nope_dim}, q-RoPE={rope_dim}, latent={latent_dim}, "
            f"weight_head_width={weight_head_width}"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q_nope"]["device_id"])
    for name in ("weight", "q_pe", "out"):
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(
                "SparkInfer GLM H64 BF16 query projection buffers must share "
                f"one device; q_nope is cuda:{device_id}, {name} is "
                f"cuda:{buffers[name]['device_id']}"
            )
    required_bytes = {
        "q_nope": query_rows * heads * nope_dim * 2,
        "weight": heads * weight_head_width * latent_dim * 2,
        "q_pe": query_rows * heads * rope_dim * 2,
        "out": query_rows * heads * (latent_dim + rope_dim) * 2,
    }
    for name, required in required_bytes.items():
        available = int(buffers[name]["bytes"])
        if available < required:
            raise ValueError(
                f"SparkInfer GLM H64 BF16 {name} needs {required} bytes, "
                f"got {available}"
            )

    block_m = 16 if query_rows <= 16 else 32
    prepared_key = (device_id, block_m)
    if prepare_only and prepared_key in _SPARKINFER_GLM_H64_QUERY_PREPARED:
        return

    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        resident_weight = _bf16_tensor(
            buffers["weight"], (heads, weight_head_width, latent_dim)
        )
        weight = resident_weight[:, :nope_dim, :]
        if prepare_only:
            prewarm_glm_h64_bf16(
                weight,
                (query_rows,),
                stream=stream,
                synchronize=False,
            )
            _SPARKINFER_GLM_H64_QUERY_PREPARED.add(prepared_key)
            return

        q_nope = _bf16_tensor(
            buffers["q_nope"], (query_rows, heads, nope_dim)
        ).permute(1, 0, 2)
        q_pe = _bf16_tensor(buffers["q_pe"], (query_rows, heads, rope_dim))
        out = _bf16_tensor(
            buffers["out"], (query_rows, heads, latent_dim + rope_dim)
        )
        run_glm_h64_bf16(q_nope, weight, q_pe, out, stream=stream)


@cache
def _glmrt_packed_fp8_mla_exact_grouped() -> Any:
    native_path = os.environ.get("GLMRT_NATIVE_LIB")
    if not native_path:
        raise RuntimeError(
            "exact grouped packed-FP8 MLA requires GLMRT_NATIVE_LIB"
        )
    library = ctypes.CDLL(str(Path(native_path).resolve()))
    function = library.glmrt_cuda_packed_fp8_mla_exact_grouped_async
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_float,
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


def _packed_fp8_mla_exact_grouped_chunks(
    query_rows: int, bucket_rows: int, heads: int
) -> int | None:
    if heads != 64:
        return None
    setting = os.environ.get(
        "GLMRT_REAL_FULL_PACKED_FP8_MLA_EXACT_GROUPED", "auto"
    ).strip().lower()
    if setting in {"0", "false", "no", "off"}:
        return None
    if setting == "auto":
        try:
            _glmrt_packed_fp8_mla_exact_grouped()
        except (AttributeError, OSError, RuntimeError):
            return None
    elif setting not in {"1", "true", "yes", "on"}:
        raise ValueError(
            "GLMRT_REAL_FULL_PACKED_FP8_MLA_EXACT_GROUPED must be auto or a "
            f"boolean value, got {setting!r}"
        )
    if bucket_rows == 1024:
        if 3 <= query_rows <= 5:
            return 2
        if 6 <= query_rows <= 8:
            return 3
    if bucket_rows == 2048:
        if 2 <= query_rows <= 3:
            return 2
        if query_rows == 4 or 7 <= query_rows <= 8:
            return 3
        if 5 <= query_rows <= 6:
            return 4
    return None


def prepare_b12x_glm_dsa_indexer_prefill(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm direct-paged GLM DSA scoring and top-k selection."""

    _run_b12x_glm_dsa_indexer_prefill(ctx, prepare_only=True, **kwargs)


def capture_b12x_glm_dsa_indexer_prefill(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Select GLM DSA top-k slots directly from the packed FP8 index cache."""

    _run_b12x_glm_dsa_indexer_prefill(ctx, prepare_only=False, **kwargs)


def _run_b12x_glm_dsa_indexer_prefill(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from b12x.attention.dsa_indexer.paged import index_topk_fp8
    from b12x.attention.dsa_indexer.scratch import (
        B12XIndexerPagedScratchCaps,
        plan_indexer_paged_scratch,
    )

    query_rows = int(kwargs["query_rows"])
    page_table_width = int(kwargs["page_table_width"])
    cache_pages = int(kwargs.get("cache_pages", page_table_width))
    topk = int(kwargs.get("topk", _GLM_DSA_TOPK))
    supertile_k = int(kwargs.get("supertile_k", 32768))
    shared_page_table = bool(kwargs.get("shared_page_table", False))
    if query_rows <= 0 or query_rows > 2048:
        raise ValueError(
            f"GLM DSA query_rows must be in [1, 2048], got {query_rows}"
        )
    if page_table_width <= 0 or cache_pages <= 0:
        raise ValueError(
            "GLM DSA prefill requires positive page_table_width and cache_pages, "
            f"got page_table_width={page_table_width} cache_pages={cache_pages}"
        )
    if topk != _GLM_DSA_TOPK:
        raise ValueError(f"GLM DSA prefill requires topk={_GLM_DSA_TOPK}, got {topk}")
    if supertile_k <= 0 or supertile_k % 512:
        raise ValueError(
            f"GLM DSA prefill supertile_k must be a positive multiple of 512, got {supertile_k}"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q_fp8"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    # Only a single query row is decode. Small prompt suffixes can be in
    # flight beside the terminal decode row and must use the multi-row tiled
    # selector; the batched decode route is not reliable for real q>1 inputs.
    selector_mode = "decode" if query_rows == 1 else "prefill"
    # A single decode row already consumes exactly one page-table row. SparkInfer's
    # fused q=1 route requires the ordinary (non-shared) contract. Multi-row
    # radix attention uses one physical page-ID row through a stride-zero
    # shared view instead of materializing query_rows copies.
    plan_shared_page_table = shared_page_table and query_rows > 1
    # Keep the fast fused scorer for decode, but invoke it explicitly below so
    # glmrt can disable its long-context cooperative merge.  That merge can
    # spin indefinitely at its inter-CTA barrier above 32K (100% reported GPU
    # activity at ~1% memory utilization).  Multi-row suffixes use the ordinary
    # tiled scorer/selector until glmrt owns the complete indexer kernel.
    selector_route = "auto" if query_rows == 1 else "paged_tiled"
    state_key = (
        device_id,
        int(buffers["scratch"]["ptr"]),
        int(buffers["page_table"]["ptr"]),
        int(buffers["cache_seqlens"]["ptr"]),
        int(buffers["active_width"]["ptr"]),
        query_rows,
        page_table_width,
        cache_pages,
        topk,
        supertile_k,
        selector_mode,
        selector_route,
        plan_shared_page_table,
    )
    plan_key = (
        device_id,
        query_rows,
        page_table_width,
        cache_pages,
        topk,
        supertile_k,
        selector_mode,
        selector_route,
        plan_shared_page_table,
    )

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q_fp8 = _u8_tensor(
            buffers["q_fp8"],
            (query_rows, _GLM_DSA_INDEX_HEADS, _GLM_DSA_INDEX_HEAD_DIM),
        ).view(torch.float8_e4m3fn)
        weights = _f32_tensor(
            buffers["weights"], (query_rows, _GLM_DSA_INDEX_HEADS)
        )
        index_k_cache = _u8_tensor(
            buffers["index_k_cache"],
            (cache_pages, _GLM_DSA_PACKED_PAGE_BYTES),
        )
        if shared_page_table:
            page_table_row = _i32_tensor(
                buffers["page_table"], (page_table_width,)
            )
            page_table = (
                page_table_row.view(1, page_table_width)
                if query_rows == 1
                else page_table_row.as_strided(
                    (query_rows, page_table_width), (0, 1)
                )
            )
        else:
            page_table = _i32_tensor(
                buffers["page_table"], (query_rows, page_table_width)
            )
        cache_seqlens = _i32_tensor(buffers["cache_seqlens"], (query_rows,))
        active_width = _i32_tensor(buffers["active_width"], (1,))
        output_indices = _i32_tensor(buffers["output_indices"], (query_rows, topk))
        scratch = _u8_tensor(
            buffers["scratch"], (int(buffers["scratch"]["bytes"]),)
        )

        state = _B12X_GLM_DSA_INDEXER_STATES.get(state_key)
        if state is None:
            caps = B12XIndexerPagedScratchCaps(
                device=q_fp8.device,
                num_q_heads=_GLM_DSA_INDEX_HEADS,
                max_q_rows=query_rows,
                max_page_table_width=page_table_width,
                topk=topk,
                page_size=_GLM_DSA_PAGE_SIZE,
                max_k_rows=page_table_width * _GLM_DSA_PAGE_SIZE,
                reserve_paged_logits=True,
                paged_logits_k_rows=page_table_width * _GLM_DSA_PAGE_SIZE,
                paged_tile_logits_k_rows=supertile_k,
                mode=selector_mode,
                shared_page_table=plan_shared_page_table,
                route=selector_route,
            )
            # Plans are immutable shape contracts. Bindings still remain
            # pointer-specific through state_key so every captured graph owns
            # the correct scratch and metadata addresses.
            plan = _B12X_GLM_DSA_INDEXER_PLANS.get(plan_key)
            if plan is None:
                plan = plan_indexer_paged_scratch(caps)
                _B12X_GLM_DSA_INDEXER_PLANS[plan_key] = plan
            required_scratch_bytes = int(plan.scratch_specs()[0].nbytes)
            if scratch.numel() < required_scratch_bytes:
                raise ValueError(
                    "GLM DSA prefill scratch is too small: "
                    f"have={scratch.numel()} need={required_scratch_bytes}"
                )
            binding = plan.bind(
                scratch=scratch[:required_scratch_bytes],
                real_page_table=page_table,
                cache_seqlens_int32=cache_seqlens,
                active_width=active_width,
                expected_num_q_heads=_GLM_DSA_INDEX_HEADS,
                shared_page_table=plan_shared_page_table,
                output_physical_slots=True,
            )
            state = (binding, required_scratch_bytes)
            _B12X_GLM_DSA_INDEXER_STATES[state_key] = state

        binding, _ = state
        # Once a shape has executed, later pointer bindings only need to be
        # constructed before their independent CUDA graph capture.
        if prepare_only and plan_key in _B12X_GLM_DSA_INDEXER_PREPARED:
            return
        if query_rows == 1:
            from b12x.attention.dsa_indexer.fused_indexer import (
                run_fused_paged_indexer,
            )
            from b12x.attention.dsa_indexer.kernel import (
                _split_index_k_cache_runtime_views,
            )

            if str(binding.route) != "paged_fused":
                raise RuntimeError(
                    "GLM DSA single-row selector expected the fused route, "
                    f"got {binding.route!r}"
                )
            k_quant, k_scales = _split_index_k_cache_runtime_views(index_k_cache)
            pack_values, pack_indices, merge_state = (
                binding.scratch.get_fused_indexer_scratch(topk=topk)
            )
            run_fused_paged_indexer(
                q_bytes=q_fp8.view(torch.uint8),
                weights=weights,
                k_quant_bytes=k_quant,
                k_scales=k_scales,
                real_page_table=page_table,
                seqlens=cache_seqlens,
                num_heads=_GLM_DSA_INDEX_HEADS,
                topk=topk,
                out_indices=output_indices,
                # Keep the cooperative grid below the 188-SM device width so
                # every participant can remain resident at the barrier.  The
                # upstream full-device policy can spin at long context.  Keep
                # the serial merge through 16K, where its lower barrier cost
                # still wins.
                ctas_per_group=128,
                merge_threshold=16384,
                pack_values=pack_values,
                pack_indices=pack_indices,
                merge_state=merge_state,
                # Keep graph replays independent when the shared selector
                # scratch is rebound across q=1 and multi-row captures.
                merge_state_preinitialized=False,
                output_physical_slots=True,
            )
        else:
            index_topk_fp8(
                q_fp8=q_fp8,
                weights=weights,
                index_k_cache=index_k_cache,
                binding=binding,
                page_size=_GLM_DSA_PAGE_SIZE,
                topk=topk,
                expected_num_q_heads=_GLM_DSA_INDEX_HEADS,
                out_indices=output_indices,
                supertile_k=supertile_k,
            )
        if prepare_only:
            _B12X_GLM_DSA_INDEXER_PREPARED.add(plan_key)


def prepare_flashinfer_packed_fp8_mla_prefill(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm direct packed-FP8 sparse MLA prefill."""

    _run_flashinfer_packed_fp8_mla_prefill(ctx, prepare_only=True, **kwargs)


def capture_flashinfer_packed_fp8_mla_prefill(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Run sparse multi-row MLA directly on caller-owned packed KV pages."""

    _run_flashinfer_packed_fp8_mla_prefill(ctx, prepare_only=False, **kwargs)


def _run_flashinfer_packed_fp8_mla_prefill(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from flashinfer.mla._sparse_mla_sm120 import (
        sparse_mla_sm120_decode_dsv3_2,
    )
    from flashinfer.mla._sparse_mla_sm120 import (
        _sparse_mla_sm120_paged_attention,
    )

    query_rows = int(kwargs["query_rows"])
    kv_pages = int(kwargs["kv_pages"])
    topk = int(kwargs.get("topk", _GLM_DSA_TOPK))
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    scale = float(kwargs["scale"])
    if query_rows <= 0 or query_rows > 2048:
        raise ValueError(
            f"packed FP8 MLA sparse query_rows must be in [1, 2048], got {query_rows}"
        )
    if kv_pages <= 0:
        raise ValueError(f"packed FP8 MLA prefill requires positive kv_pages, got {kv_pages}")
    if topk not in (128, 512, 1024, _GLM_DSA_TOPK):
        raise ValueError(
            "packed FP8 MLA prefill requires topk in "
            f"(128, 512, 1024, {_GLM_DSA_TOPK}), got {topk}"
        )
    if heads not in _GLM_ATTENTION_HEADS or nope_dim != 512 or rope_dim != 64:
        raise ValueError(
            "packed FP8 GLM-5.2 MLA prefill requires heads=16 or 64, "
            "nope_dim=512, and rope_dim=64"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        query_rows,
        kv_pages,
        topk,
        heads,
        nope_dim,
        rope_dim,
        scale,
    )
    if prepare_only and prepared_key in _FLASHINFER_PACKED_FP8_MLA_PREFILL_PREPARED:
        return
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q = _bf16_tensor(buffers["q"], (query_rows, heads, nope_dim + rope_dim))
        kv = _u8_tensor(buffers["kv"], (kv_pages, _GLM_DSA_PAGE_SIZE, 656))
        indices = _i32_tensor(buffers["indices"], (query_rows, topk))
        topk_length = _i32_tensor(buffers["topk_length"], (query_rows,))
        output = _bf16_tensor(buffers["output"], (query_rows, heads, nope_dim))
        out_lse = _f32_tensor(buffers["out_lse"], (query_rows, heads))
        mid_out = None
        mid_lse = None
        if query_rows <= 64:
            splits = (topk + _GLM_DSA_PAGE_SIZE - 1) // _GLM_DSA_PAGE_SIZE
            mid_out = _bf16_tensor(
                buffers["mid_out"], (query_rows, heads, splits, nope_dim)
            )
            mid_lse = _f32_tensor(
                buffers["mid_lse"], (query_rows, heads, splits)
            )

        if query_rows <= 8:
            exact_grouped_chunks = _packed_fp8_mla_exact_grouped_chunks(
                query_rows, topk, heads
            )
            if query_rows == 1:
                sparse_mla_sm120_decode_dsv3_2(
                    q,
                    kv,
                    indices,
                    mid_out,
                    mid_lse,
                    output,
                    out_lse,
                    scale,
                    topk_length=topk_length,
                    model_type=2,
                    chunks_per_block=1,
                )
            elif exact_grouped_chunks is not None:
                status = _glmrt_packed_fp8_mla_exact_grouped()(
                    ctypes.c_void_p(q.data_ptr()),
                    ctypes.c_void_p(kv.data_ptr()),
                    ctypes.c_void_p(indices.data_ptr()),
                    ctypes.c_void_p(mid_out.data_ptr()),
                    ctypes.c_void_p(mid_lse.data_ptr()),
                    ctypes.c_void_p(topk_length.data_ptr()),
                    ctypes.c_void_p(output.data_ptr()),
                    ctypes.c_void_p(out_lse.data_ptr()),
                    query_rows,
                    heads,
                    topk,
                    exact_grouped_chunks,
                    ctypes.c_float(scale),
                    kv.stride(0),
                    ctypes.c_void_p(int(ctx["cuda_stream"])),
                )
                if status != 0:
                    raise RuntimeError(
                        "exact grouped sparse DSA MLA launch failed with status "
                        f"{status}"
                    )
            else:
                # A deployment without the grouped object remains correct by
                # capturing the recurrent q=1 schedule independently per row.
                for row in range(query_rows):
                    sparse_mla_sm120_decode_dsv3_2(
                        q[row : row + 1],
                        kv,
                        indices[row : row + 1],
                        mid_out[row : row + 1],
                        mid_lse[row : row + 1],
                        output[row : row + 1],
                        out_lse[row : row + 1],
                        scale,
                        topk_length=topk_length[row : row + 1],
                        model_type=2,
                        chunks_per_block=1,
                    )
        else:
            _sparse_mla_sm120_paged_attention(
                q,
                kv,
                indices,
                output,
                out_lse,
                scale,
                d_v=nope_dim,
                kv_scale_format="arbitrary_fp32",
                topk_length=topk_length,
                mid_out=mid_out,
                mid_lse=mid_lse,
            )
        if prepare_only:
            _FLASHINFER_PACKED_FP8_MLA_PREFILL_PREPARED.add(prepared_key)


def prepare_flashinfer_packed_fp8_mla_decode(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm one allocation-free packed-FP8 GLM decode bucket."""

    _run_flashinfer_packed_fp8_mla_decode(ctx, prepare_only=True, **kwargs)


def capture_flashinfer_packed_fp8_mla_decode(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch packed-FP8 GLM decode for external CUDA graph capture."""

    _run_flashinfer_packed_fp8_mla_decode(ctx, prepare_only=False, **kwargs)


def _run_flashinfer_packed_fp8_mla_decode(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from flashinfer.mla._sparse_mla_sm120 import (
        sparse_mla_sm120_decode_dsv3_2,
    )

    bucket_rows = int(kwargs["bucket_rows"])
    kv_capacity_rows = int(kwargs.get("kv_capacity_rows", bucket_rows))
    query_rows = int(kwargs.get("query_rows", 1))
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    scale = float(kwargs["scale"])
    initialize_kv = bool(kwargs.get("initialize_kv", True))
    exact_grouped_chunks = _packed_fp8_mla_exact_grouped_chunks(
        query_rows, bucket_rows, heads
    )
    if bucket_rows not in (128, 512, 1024, 2048):
        raise ValueError(
            "packed FP8 MLA decode bucket must be one of 128, 512, 1024, or 2048, "
            f"got {bucket_rows}"
        )
    if kv_capacity_rows < bucket_rows or kv_capacity_rows % 64 != 0:
        raise ValueError(
            "packed FP8 MLA decode KV capacity must cover the bucket and be divisible "
            f"by 64, got capacity={kv_capacity_rows} bucket={bucket_rows}"
        )
    if query_rows <= 0 or query_rows > 16:
        raise ValueError(
            f"packed FP8 MLA decode query_rows must be in [1, 16], got {query_rows}"
        )
    if heads not in _GLM_ATTENTION_HEADS or nope_dim != 512 or rope_dim != 64:
        raise ValueError(
            "packed FP8 GLM-5.2 MLA decode requires heads=16 or 64, nope_dim=512, "
            "and rope_dim=64"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    splits = bucket_rows // 64
    prepared_key = (
        device_id,
        bucket_rows,
        kv_capacity_rows,
        query_rows,
        heads,
        nope_dim,
        rope_dim,
        scale,
        initialize_kv,
        exact_grouped_chunks,
    )
    if prepare_only and prepared_key in _FLASHINFER_PACKED_FP8_MLA_PREPARED:
        return
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q = _bf16_tensor(
            buffers["q"], (query_rows, heads, nope_dim + rope_dim)
        )
        kv = _u8_tensor(buffers["kv"], (kv_capacity_rows // 64, 64, 656))
        indices = _i32_tensor(buffers["indices"], (query_rows, bucket_rows))
        topk_length = _i32_tensor(buffers["topk_length"], (query_rows,))
        output = _bf16_tensor(
            buffers["output"], (query_rows, heads, nope_dim)
        )
        out_lse = _f32_tensor(buffers["out_lse"], (query_rows, heads))
        mid_out = _bf16_tensor(
            buffers["mid_out"], (query_rows, heads, splits, nope_dim)
        )
        mid_lse = _f32_tensor(
            buffers["mid_lse"], (query_rows, heads, splits)
        )

        if prepare_only and prepared_key not in _FLASHINFER_PACKED_FP8_MLA_PREPARED:
            q.zero_()
            if initialize_kv:
                kv.zero_()
                kv.view(kv_capacity_rows, 656)[:, 512:528].view(torch.float32).fill_(1.0)
            indices.copy_(
                torch.arange(bucket_rows, dtype=torch.int32, device=q.device)
                .view(1, -1)
                .expand(query_rows, -1)
            )
            if initialize_kv:
                topk_length.fill_(bucket_rows)

        # The production recurrent M=1 path uses one 64-row chunk per split.
        # Keeping that split topology in the multi-query launch makes every
        # row bitwise identical while still collapsing 2*query_rows kernels
        # into one attention and one merge launch. FlashInfer's M>1 automatic
        # chunks-per-block heuristic is slightly faster, but changes the
        # partial reduction boundaries and therefore the accepted state.
        if exact_grouped_chunks is None:
            sparse_mla_sm120_decode_dsv3_2(
                q,
                kv,
                indices,
                mid_out,
                mid_lse,
                output,
                out_lse,
                scale,
                topk_length=topk_length,
                model_type=2,
                chunks_per_block=1,
            )
        else:
            status = _glmrt_packed_fp8_mla_exact_grouped()(
                ctypes.c_void_p(q.data_ptr()),
                ctypes.c_void_p(kv.data_ptr()),
                ctypes.c_void_p(indices.data_ptr()),
                ctypes.c_void_p(mid_out.data_ptr()),
                ctypes.c_void_p(mid_lse.data_ptr()),
                ctypes.c_void_p(topk_length.data_ptr()),
                ctypes.c_void_p(output.data_ptr()),
                ctypes.c_void_p(out_lse.data_ptr()),
                query_rows,
                heads,
                bucket_rows,
                exact_grouped_chunks,
                ctypes.c_float(scale),
                kv.stride(0),
                ctypes.c_void_p(int(ctx["cuda_stream"])),
            )
            if status != 0:
                raise RuntimeError(
                    "exact grouped packed-FP8 MLA launch failed with status "
                    f"{status}"
                )
        if prepare_only:
            _FLASHINFER_PACKED_FP8_MLA_PREPARED.add(prepared_key)


def prepare_sparkinfer_nvfp4_mla_decode(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm native packed-NVFP4 sparse MLA split decode."""

    _run_sparkinfer_nvfp4_mla(ctx, prepare_only=True, decode=True, **kwargs)


def capture_sparkinfer_nvfp4_mla_decode(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch native packed-NVFP4 sparse MLA split decode for graph capture."""

    _run_sparkinfer_nvfp4_mla(ctx, prepare_only=False, decode=True, **kwargs)


def prepare_sparkinfer_nvfp4_mla_prefill(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm native packed-NVFP4 sparse MLA single-pass prefill."""

    _run_sparkinfer_nvfp4_mla(ctx, prepare_only=True, decode=False, **kwargs)


def capture_sparkinfer_nvfp4_mla_prefill(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch native packed-NVFP4 sparse MLA prefill for graph capture."""

    _run_sparkinfer_nvfp4_mla(ctx, prepare_only=False, decode=False, **kwargs)


def _run_sparkinfer_nvfp4_mla(
    ctx: dict[str, Any],
    *,
    prepare_only: bool,
    decode: bool,
    **kwargs: Any,
) -> None:
    """Consume the canonical 432-byte NVFP4 cache without an FP8 mirror."""

    import torch
    from b12x.attention._shared.mla.kernel import run_unified_decode
    from b12x.attention._shared.mla.prefill import run_unified_prefill
    from b12x.attention._shared.mla.traits import ScaleFormat

    query_rows = int(kwargs["query_rows"])
    kv_pages = int(kwargs["kv_pages"])
    topk = int(kwargs.get("topk", _GLM_DSA_TOPK))
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    scale = float(kwargs["scale"])
    if query_rows <= 0 or query_rows > 2048:
        raise ValueError(
            f"native NVFP4 MLA query_rows must be in [1, 2048], got {query_rows}"
        )
    if decode and query_rows > 16:
        raise ValueError(
            f"native NVFP4 MLA split decode supports at most 16 rows, got {query_rows}"
        )
    if kv_pages <= 0:
        raise ValueError(
            f"native NVFP4 MLA requires positive kv_pages, got {kv_pages}"
        )
    if topk not in (128, 512, 1024, _GLM_DSA_TOPK):
        raise ValueError(
            "native NVFP4 MLA requires topk in "
            f"(128, 512, 1024, {_GLM_DSA_TOPK}), got {topk}"
        )
    if heads not in _GLM_ATTENTION_HEADS or nope_dim != 512 or rope_dim != 64:
        raise ValueError(
            "native NVFP4 GLM-5.2 MLA requires heads=16 or 64, "
            "nope_dim=512, and rope_dim=64"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    prepared_key = (
        device_id,
        query_rows,
        kv_pages,
        topk,
        heads,
        nope_dim,
        rope_dim,
        scale,
        decode,
    )
    prepared = (
        _SPARKINFER_NVFP4_MLA_DECODE_PREPARED
        if decode
        else _SPARKINFER_NVFP4_MLA_PREFILL_PREPARED
    )
    # The low-level SparkInfer launch has no pointer-bound plan. Compile and
    # execute one warmup per shape; each pointer identity is still captured.
    if prepare_only and prepared_key in prepared:
        return
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q = _bf16_tensor(
            buffers["q"], (query_rows, heads, nope_dim + rope_dim)
        )
        kv = _u8_tensor(buffers["kv"], (kv_pages, _GLM_DSA_PAGE_SIZE, 432))
        indices = _i32_tensor(buffers["indices"], (query_rows, topk))
        topk_length = _i32_tensor(buffers["topk_length"], (query_rows,))
        output = _bf16_tensor(
            buffers["output"], (query_rows, heads, nope_dim)
        )
        out_lse = _f32_tensor(
            buffers["out_lse"], (query_rows, heads)
        )

        if decode:
            splits = topk // _GLM_DSA_PAGE_SIZE
            mid_out = _bf16_tensor(
                buffers["mid_out"], (query_rows, heads, splits, nope_dim)
            )
            mid_lse = _f32_tensor(
                buffers["mid_lse"], (query_rows, heads, splits)
            )
            # run_unified_decode only needs this compact duck-typed subset.
            # num_chunks is a compile-time merge argument; the pointer remains
            # part of the public contract but is not read on this static path.
            workspace = SimpleNamespace(
                max_chunks_per_row=splits,
                tmp_output=mid_out,
                tmp_lse=mid_lse,
                output_buffer=output,
                num_chunks_ptr=topk_length[:1],
            )
            run_unified_decode(
                q_all=q,
                swa_k_cache=kv,
                swa_indices=indices,
                swa_topk_lengths=topk_length,
                workspace=workspace,
                sm_scale=scale,
                latent_scale=1.0,
                swa_page_size=_GLM_DSA_PAGE_SIZE,
                out=output,
                scale_format_override=int(ScaleFormat.NVFP4_E4M3),
                fp8_rope_override=False,
            )
        else:
            run_unified_prefill(
                q=q,
                kv_cache=kv,
                topk_indices=indices,
                topk_length=topk_length,
                sm_scale=scale,
                latent_scale=1.0,
                page_block_size=_GLM_DSA_PAGE_SIZE,
                output=output,
                lse_out=out_lse,
                scale_format=int(ScaleFormat.NVFP4_E4M3),
                fp8_rope=False,
            )

        if prepare_only:
            prepared.add(prepared_key)


def prepare_flashinfer_compressed_mla_decode_chunk(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Plan and warm one allocation-free compressed-BF16 decode chunk."""

    _run_flashinfer_compressed_mla_decode_chunk(ctx, prepare_only=True, **kwargs)


def capture_flashinfer_compressed_mla_decode_chunk(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch one compressed-BF16 decode chunk for external graph capture."""

    _run_flashinfer_compressed_mla_decode_chunk(ctx, prepare_only=False, **kwargs)


def _run_flashinfer_compressed_mla_decode_chunk(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from flashinfer.mla import BatchMLAPagedAttentionWrapper

    rows = int(kwargs["rows"])
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    scale = float(kwargs["scale"])
    if rows <= 0 or rows > 2048:
        raise ValueError(f"compressed MLA decode chunk rows must be in [1, 2048], got {rows}")
    if heads not in _GLM_ATTENTION_HEADS or nope_dim != 512 or rope_dim != 64:
        raise ValueError(
            "compressed GLM-5.2 MLA decode requires heads=16 or 64, nope_dim=512, and rope_dim=64"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q_nope"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q_nope = _bf16_tensor(buffers["q_nope"], (1, heads, nope_dim))
        q_rope = _bf16_tensor(buffers["q_rope"], (1, heads, rope_dim))
        kv = _bf16_tensor(buffers["kv"], (rows, nope_dim + rope_dim))
        partial = _bf16_tensor(buffers["partial"], (1, heads, nope_dim))
        partial_lse = _f32_tensor(buffers["partial_lse"], (1, heads))
        workspace = _u8_tensor(buffers["workspace"], (int(buffers["workspace"]["bytes"]),))

        # Page size one permits exact 1-32 tail graphs without padding. The
        # cache views retain the interleaved 576-element row stride.
        ckv = kv.as_strided((rows, 1, nope_dim), (nope_dim + rope_dim, nope_dim, 1))
        kpe = kv[:, nope_dim:].as_strided(
            (rows, 1, rope_dim), (nope_dim + rope_dim, rope_dim, 1)
        )
        runner_key = (
            device_id,
            int(buffers["workspace"]["ptr"]),
            rows,
            heads,
            nope_dim,
            rope_dim,
            scale,
        )
        runner = _FLASHINFER_COMPRESSED_MLA_RUNNERS.get(runner_key)
        if runner is None:
            runner = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
            # FlashInfer defaults these workspaces to 8 MiB each. Batch-one
            # GLM decode needs less than 1 MiB even at 2K context.
            runner._int_workspace_buffer = torch.empty(
                1024 * 1024, dtype=torch.uint8, device=q_nope.device
            )
            runner._pin_memory_int_workspace_buffer = torch.empty(
                1024 * 1024, dtype=torch.uint8, pin_memory=True, device="cpu"
            )
            qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=q_nope.device)
            kv_indptr = torch.tensor([0, rows], dtype=torch.int32, device=q_nope.device)
            kv_indices = torch.arange(rows, dtype=torch.int32, device=q_nope.device)
            kv_len = torch.tensor([rows], dtype=torch.int32, device=q_nope.device)
            runner.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len,
                heads,
                nope_dim,
                rope_dim,
                1,
                False,
                scale,
                torch.bfloat16,
                torch.bfloat16,
            )
            _FLASHINFER_COMPRESSED_MLA_RUNNERS[runner_key] = runner

        if prepare_only and runner_key in _FLASHINFER_COMPRESSED_MLA_PREPARED:
            return
        runner.run(
            q_nope,
            q_rope,
            ckv,
            kpe,
            out=partial,
            lse=partial_lse,
            return_lse=True,
            return_lse_base_on_e=True,
        )
        if prepare_only:
            _FLASHINFER_COMPRESSED_MLA_PREPARED.add(runner_key)


def prepare_flashinfer_mla_rope_attention(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Compile and warm the allocation-free FlashInfer MLA launch."""

    _run_flashinfer_mla_rope_attention(ctx, prepare_only=True, **kwargs)


def capture_flashinfer_mla_rope_attention(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch allocation-free FlashInfer MLA operations for CUDA graph capture."""

    _run_flashinfer_mla_rope_attention(ctx, prepare_only=False, **kwargs)


def prepare_flashinfer_cudnn_mla_rope_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Compile and warm one dynamic-length cuDNN MLA suffix graph."""

    _run_flashinfer_cudnn_mla_rope_attention(ctx, **kwargs)


def capture_flashinfer_cudnn_mla_rope_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    """Launch dynamic-length cuDNN MLA suffix attention for graph capture."""

    _run_flashinfer_cudnn_mla_rope_attention(ctx, **kwargs)


def _run_flashinfer_cudnn_mla_rope_attention(
    ctx: dict[str, Any], **kwargs: Any
) -> None:
    import torch
    from flashinfer.cudnn.prefill import cudnn_batch_prefill_with_kv_cache

    row_capacity = int(kwargs["row_capacity"])
    query_capacity = int(kwargs["query_capacity"])
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    v_dim = int(kwargs["v_dim"])
    scale = float(kwargs["scale"])
    qk_dim = nope_dim + rope_dim
    if row_capacity <= 0 or query_capacity <= 0 or query_capacity > row_capacity:
        raise ValueError(
            "cuDNN MLA capture requires 0 < query_capacity <= row_capacity, got "
            f"query_capacity={query_capacity} row_capacity={row_capacity}"
        )
    if (
        heads not in _GLM_ATTENTION_HEADS
        or nope_dim != 192
        or rope_dim != 64
        or v_dim != 256
    ):
        raise ValueError(
            "cuDNN GLM-5.2 MLA capture requires heads=16 or 64, nope_dim=192, "
            "rope_dim=64, and v_dim=256"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q_nope"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q_nope = _bf16_tensor(
            buffers["q_nope"], (query_capacity, heads, nope_dim)
        )
        q_rope = _bf16_tensor(
            buffers["q_rope"], (query_capacity, heads, rope_dim)
        )
        k_nope = _bf16_tensor(
            buffers["k_nope"], (row_capacity, heads, nope_dim)
        )
        k_rope = _bf16_tensor(buffers["k_rope"], (row_capacity, rope_dim))
        values = _bf16_tensor(
            buffers["values"], (row_capacity, heads, v_dim)
        )
        q = _bf16_tensor(buffers["q"], (query_capacity, heads, qk_dim))
        k = _bf16_tensor(buffers["k"], (row_capacity, heads, qk_dim))
        output = _bf16_tensor(
            buffers["output"], (query_capacity, heads, v_dim)
        )
        workspace = _u8_tensor(
            buffers["workspace"], (int(buffers["workspace"]["bytes"]),)
        )
        query_lengths = _i32_tensor(buffers["query_lengths"], (1,))
        kv_lengths = _i32_tensor(buffers["kv_lengths"], (1,))

        torch.cat((q_nope, q_rope), dim=-1, out=q)
        torch.cat(
            (k_nope, k_rope[:, None, :].expand(-1, heads, -1)),
            dim=-1,
            out=k,
        )
        cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            values,
            scale,
            workspace,
            max_token_per_sequence=query_capacity,
            max_sequence_kv=row_capacity,
            actual_seq_lens_q=query_lengths,
            actual_seq_lens_kv=kv_lengths,
            causal=True,
            return_lse=False,
            out=output,
            is_cuda_graph_compatible=True,
        )


def _run_flashinfer_mla_rope_attention(
    ctx: dict[str, Any], *, prepare_only: bool, **kwargs: Any
) -> None:
    import torch
    from flashinfer.prefill import SINGLE_KERNEL_TMP_SIZE, get_single_prefill_module
    from flashinfer.utils import MaskMode, PosEncodingMode, TensorLayout

    rows = int(kwargs["rows"])
    query_row_offset = int(kwargs["query_row_offset"])
    query_rows = int(kwargs["query_rows"])
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    v_dim = int(kwargs["v_dim"])
    scale = float(kwargs["scale"])
    qk_dim = nope_dim + rope_dim
    if rows <= 0 or query_rows <= 0:
        raise ValueError(
            f"FlashInfer MLA capture requires positive rows, got rows={rows} query_rows={query_rows}"
        )
    if query_row_offset < 0 or query_row_offset + query_rows > rows:
        raise ValueError(
            "FlashInfer MLA query range exceeds KV rows: "
            f"offset={query_row_offset} query_rows={query_rows} rows={rows}"
        )
    if (
        heads not in _GLM_ATTENTION_HEADS
        or nope_dim != 192
        or rope_dim != 64
        or v_dim != 256
    ):
        raise ValueError(
            "FlashInfer GLM-5.2 MLA capture requires heads=16 or 64, nope_dim=192, "
            "rope_dim=64, and v_dim=256"
        )

    buffers = ctx["buffers"]
    device_id = int(buffers["q_nope"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q_nope = _bf16_tensor(buffers["q_nope"], (query_rows, heads, nope_dim))
        q_rope = _bf16_tensor(buffers["q_rope"], (query_rows, heads, rope_dim))
        k_nope = _bf16_tensor(buffers["k_nope"], (rows, heads, nope_dim))
        k_rope = _bf16_tensor(buffers["k_rope"], (rows, rope_dim))
        values = _bf16_tensor(buffers["values"], (rows, heads, v_dim))
        q = _bf16_tensor(buffers["q"], (query_rows, heads, qk_dim))
        k = _bf16_tensor(buffers["k"], (rows, heads, qk_dim))
        output = _bf16_tensor(buffers["output"], (query_rows, heads, v_dim))
        workspace = _u8_tensor(buffers["workspace"], (SINGLE_KERNEL_TMP_SIZE,))

        module_key = (device_id, q.dtype, k.dtype, output.dtype, qk_dim, v_dim)
        module = _FLASHINFER_MODULES.get(module_key)
        if module is None:
            module = get_single_prefill_module(
                "fa2",
                q.dtype,
                k.dtype,
                output.dtype,
                qk_dim,
                v_dim,
                PosEncodingMode.NONE.value,
                False,
                False,
                False,
            )
            _FLASHINFER_MODULES[module_key] = module
        prepared_shape = module_key + (rows, query_rows)
        if prepare_only and prepared_shape in _FLASHINFER_PREPARED_SHAPES:
            return

        torch.cat(
            (q_nope, q_rope),
            dim=-1,
            out=q,
        )
        torch.cat(
            (k_nope, k_rope[:, None, :].expand(-1, heads, -1)),
            dim=-1,
            out=k,
        )
        module.run(
            q,
            k,
            values,
            workspace,
            output,
            None,
            MaskMode.CAUSAL.value,
            TensorLayout.NHD.value,
            -1,
            None,
            None,
            0.0,
            scale,
            None,
            None,
            None,
            1.0,
            10_000.0,
            None,
            None,
        )
        if prepare_only:
            _FLASHINFER_PREPARED_SHAPES.add(prepared_shape)


def capture_mla_rope_attention(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Capture the GLM_NSA SparkInfer MLA kernel from GLMRT raw pointer metadata.

    This adapter intentionally supports only the SparkInfer GLM_NSA absorbed-MLA
    contract: 8 local heads, q_nope/kv_nope/v_dim=512 and q/k_rope=64. Smaller
    debug MLA shapes stay on the Rust CUDA reference path.
    """

    target = os.environ.get(_TARGET_ENV)
    if target:
        module_name, _, function_name = target.partition(":")
        if not module_name or not function_name:
            raise ValueError(f"{_TARGET_ENV} must be formatted as module:function")
        getattr(import_module(module_name), function_name)(ctx, **kwargs)
        return

    import torch
    from b12x.attention._shared.mla.prefill_mg import run_unified_prefill_mg
    from b12x.attention._shared.mla.reference import (
        pack_mla_kv_cache_reference,
    )
    from b12x.attention._shared.mla.traits import (
        ComputeMode,
        ModelType,
        ScaleFormat,
    )

    rows = int(kwargs["rows"])
    heads = int(kwargs["heads"])
    nope_dim = int(kwargs["nope_dim"])
    rope_dim = int(kwargs["rope_dim"])
    v_dim = int(kwargs["v_dim"])
    scale = float(kwargs["scale"])
    if (
        heads != _GLM_NSA_HEADS
        or nope_dim != _GLM_NSA_NOPE_DIM
        or rope_dim != _GLM_NSA_ROPE_DIM
        or v_dim != _GLM_NSA_V_DIM
    ):
        raise ValueError(
            "SparkInfer GLM_NSA MLA capture requires heads=8, nope_dim=512, "
            "rope_dim=64, and v_dim=512"
        )
    if rows <= 0 or rows > _GLM_NSA_TOPK:
        raise ValueError(f"SparkInfer GLM_NSA MLA capture requires rows in [1, 512], got {rows}")

    buffers = ctx["buffers"]
    device_id = int(buffers["q_nope"]["device_id"])
    stream = torch.cuda.ExternalStream(int(ctx["cuda_stream"]), device=device_id)

    with torch.cuda.device(device_id), torch.cuda.stream(stream):
        q_nope = _bf16_tensor(buffers["q_nope"], (rows, heads, nope_dim))
        q_rope = _bf16_tensor(buffers["q_rope"], (rows, heads, rope_dim))
        k_nope = _bf16_tensor(buffers["k_nope"], (rows, heads, nope_dim))
        k_rope = _bf16_tensor(buffers["k_rope"], (rows, rope_dim))
        output = _bf16_tensor(buffers["output"], (rows, heads, v_dim))

        q_all = torch.cat((q_nope, q_rope), dim=-1).contiguous()
        kv_cache = pack_mla_kv_cache_reference(k_nope[:, 0, :].contiguous(), k_rope.contiguous())
        topk_indices = torch.full(
            (rows, _GLM_NSA_TOPK),
            -1,
            dtype=torch.int32,
            device=q_all.device,
        )
        for row in range(rows):
            topk_indices[row, : row + 1] = torch.arange(
                row + 1,
                dtype=torch.int32,
                device=q_all.device,
            )
        topk_lengths = torch.arange(1, rows + 1, dtype=torch.int32, device=q_all.device)

        run_unified_prefill_mg(
            q=q_all,
            kv_cache=kv_cache,
            topk_indices=topk_indices,
            topk_length=topk_lengths,
            sm_scale=scale,
            page_block_size=1,
            output=output,
            compute_mode=ComputeMode.FP8,
            mg_n_hg=1,
            model_type=ModelType.GLM_NSA,
            scale_format=ScaleFormat.ARBITRARY_FP32,
            active_heads=heads,
        )


def _bf16_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    import torch

    dl_data_type, raw_dlpack_tensor = _dlpack_bridge()
    return torch.utils.dlpack.from_dlpack(
        raw_dlpack_tensor(
            ptr=int(buffer["ptr"]),
            shape=shape,
            dtype=dl_data_type(_DLPACK_CODE_BFLOAT, 16, 1),
            device_id=int(buffer["device_id"]),
        )
    )


def _u8_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    import torch

    dl_data_type, raw_dlpack_tensor = _dlpack_bridge()
    return torch.utils.dlpack.from_dlpack(
        raw_dlpack_tensor(
            ptr=int(buffer["ptr"]),
            shape=shape,
            dtype=dl_data_type(_DLPACK_CODE_UINT, 8, 1),
            device_id=int(buffer["device_id"]),
        )
    )


def _i32_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    import torch

    dl_data_type, raw_dlpack_tensor = _dlpack_bridge()
    return torch.utils.dlpack.from_dlpack(
        raw_dlpack_tensor(
            ptr=int(buffer["ptr"]),
            shape=shape,
            dtype=dl_data_type(_DLPACK_CODE_INT, 32, 1),
            device_id=int(buffer["device_id"]),
        )
    )


def _f32_tensor(buffer: dict[str, Any], shape: tuple[int, ...]):
    import torch

    dl_data_type, raw_dlpack_tensor = _dlpack_bridge()
    return torch.utils.dlpack.from_dlpack(
        raw_dlpack_tensor(
            ptr=int(buffer["ptr"]),
            shape=shape,
            dtype=dl_data_type(_DLPACK_CODE_FLOAT, 32, 1),
            device_id=int(buffer["device_id"]),
        )
    )


def _dlpack_bridge() -> tuple[Any, Any]:
    global _DLPACK_BRIDGE
    if _DLPACK_BRIDGE is not None:
        return _DLPACK_BRIDGE

    import ctypes

    class DLDevice(ctypes.Structure):
        _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]

    class DLDataType(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]

    class DLTensor(ctypes.Structure):
        _fields_ = [
            ("data", ctypes.c_void_p),
            ("device", DLDevice),
            ("ndim", ctypes.c_int),
            ("dtype", DLDataType),
            ("shape", ctypes.POINTER(ctypes.c_int64)),
            ("strides", ctypes.POINTER(ctypes.c_int64)),
            ("byte_offset", ctypes.c_uint64),
        ]

    class DLManagedTensor(ctypes.Structure):
        pass

    DLManagedTensorPtr = ctypes.POINTER(DLManagedTensor)
    DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, DLManagedTensorPtr)

    @DLManagedTensorDeleter
    def delete_dlpack_tensor(ptr: DLManagedTensorPtr) -> None:
        if bool(ptr):
            owners = globals().get("_DLPACK_OWNERS")
            if owners is not None:
                owners.pop(ctypes.addressof(ptr.contents), None)

    DLManagedTensor._fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", DLManagedTensorDeleter),
    ]

    py_capsule_new = ctypes.pythonapi.PyCapsule_New
    py_capsule_new.restype = ctypes.py_object
    py_capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    class RawDlpackTensor:
        def __init__(
            self,
            *,
            ptr: int,
            shape: tuple[int, ...],
            dtype: DLDataType,
            device_id: int,
        ) -> None:
            if ptr == 0:
                raise ValueError("raw DLPack tensor pointer is null")
            if any(dim < 0 for dim in shape):
                raise ValueError(f"raw DLPack tensor shape has a negative dimension: {shape}")
            self.ptr = int(ptr)
            self.shape = shape
            self.dtype = dtype
            self.device_id = int(device_id)
            self._shape = (ctypes.c_int64 * len(shape))(*shape)
            self._strides = _contiguous_strides(shape)
            self._strides_array = (ctypes.c_int64 * len(shape))(*self._strides)
            self._managed = DLManagedTensor()
            self._managed.dl_tensor = DLTensor(
                ctypes.c_void_p(self.ptr),
                DLDevice(_DLPACK_DEVICE_CUDA, self.device_id),
                len(shape),
                self.dtype,
                self._shape,
                self._strides_array,
                0,
            )
            self._managed.manager_ctx = None
            self._managed.deleter = delete_dlpack_tensor

        def __dlpack_device__(self) -> tuple[int, int]:
            return (_DLPACK_DEVICE_CUDA, self.device_id)

        def __dlpack__(self, stream: int | None = None) -> object:
            del stream
            address = ctypes.addressof(self._managed)
            _DLPACK_OWNERS[address] = self
            return py_capsule_new(ctypes.c_void_p(address), b"dltensor", None)

    _DLPACK_BRIDGE = (DLDataType, RawDlpackTensor)
    return _DLPACK_BRIDGE


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.append(stride)
        stride *= max(int(dim), 1)
    return tuple(reversed(strides))
