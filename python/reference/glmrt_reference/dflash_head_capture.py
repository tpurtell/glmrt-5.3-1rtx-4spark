from __future__ import annotations

from dataclasses import dataclass
from functools import cache
import os
import sys
from typing import Any

import triton
import triton.language as tl

from b12x_mla_capture import _bf16_tensor, _i32_tensor, _u8_tensor
from dflash_tuning_profile import dflash2_selector_num_warps
from dspark_capture import _i64_tensor

_DFLASH2_HEAD_STATES: dict[tuple[Any, ...], "_DFlash2HeadState"] = {}
_DFLASH2_TOPK_BACKEND_ENV = "GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND"
_DFLASH2_TOPK_BACKENDS = frozenset(("torch", "flashinfer", "flashinfer-dsa"))


@dataclass(frozen=True)
class _DFlash2HeadState:
    device_id: int
    cuda_stream: int
    active_requests: int
    hidden_rows_per_request: int
    proposal_tokens: int
    hidden_size: int
    selector_rank: int
    selector_top_k: int
    vocab_size: int
    topk_backend: str
    hidden: Any
    hidden_position_major: Any
    logits: Any
    unary: Any
    candidates: Any
    radix_candidates: Any
    radix_row_states: Any
    projected_hidden: Any
    token_steps: Any
    anchor_tokens: Any
    output_tokens: Any
    reference_tokens: Any
    eager_tokens: Any
    lm_head_t: Any
    hidden_projection_t: Any
    predecessor_codebook: Any
    successor_codebook: Any
    flashinfer_topk_module: Any | None


def prepare_dflash2_head(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Bind, initialize, compile, and validate one fixed-address DFlash2 head."""

    import torch

    state = _head_state(ctx, kwargs, create=True)
    _run_reference(state)
    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        reference_unary = state.unary.clone()
    _run_head(state)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        if state.topk_backend != "torch":
            # BF16 logits contain frequent ties, including at the 16th-place
            # cutoff. Torch and FlashInfer return the same sorted top-k values
            # but may choose different token IDs from that equal-valued cutoff
            # group. This is a valid top-k, though not bit-identical candidate
            # identity. Prove that every returned ID owns its reported value,
            # that values match Torch, and that IDs are unique. Startup also
            # verifies the selected output tokens on the full real-weight
            # fixture; release qualification then performance/quality-gates
            # the backend on real requests.
            radix_candidates_i64 = state.radix_candidates.to(dtype=torch.int64)
            candidate_values_equal = torch.equal(
                torch.gather(state.logits, -1, radix_candidates_i64),
                state.unary,
            )
            values_equal = torch.equal(reference_unary, state.unary)
            sorted_candidates = torch.sort(radix_candidates_i64, dim=-1).values
            candidates_unique = bool(
                torch.all(sorted_candidates[..., 1:] != sorted_candidates[..., :-1])
            )
            if not candidate_values_equal or not values_equal or not candidates_unique:
                candidate_value_mismatch = int(
                    torch.count_nonzero(
                        torch.gather(state.logits, -1, radix_candidates_i64)
                        != state.unary
                    ).item()
                )
                value_mismatch = int(
                    torch.count_nonzero(reference_unary != state.unary).item()
                )
                raise RuntimeError(
                    "DFlash2 FlashInfer radix top-k is invalid: "
                    f"candidate_value_mismatches={candidate_value_mismatch} "
                    f"value_mismatches={value_mismatch} "
                    f"candidates_unique={candidates_unique}"
                )
        if not torch.equal(state.reference_tokens, state.output_tokens):
            mismatch = int(
                torch.count_nonzero(
                    state.reference_tokens != state.output_tokens
                ).item()
            )
            if state.topk_backend == "torch":
                raise RuntimeError(
                    "DFlash2 candidate selector disagrees with its reference in "
                    f"{mismatch} tokens"
                )
            print(
                "glmrt_dflash2_topk_boundary_tie_output_delta "
                f"device={state.device_id} requests={state.active_requests} "
                f"drafts={state.proposal_tokens} differing_tokens={mismatch}",
                file=sys.stderr,
            )
        state.eager_tokens.copy_(state.output_tokens)


def capture_dflash2_head(ctx: dict[str, Any], **kwargs: Any) -> None:
    """Launch the allocation-free greedy DFlash2 candidate selector."""

    _run_head(_head_state(ctx, kwargs, create=False))


def _head_state(
    ctx: dict[str, Any], kwargs: dict[str, Any], *, create: bool
) -> _DFlash2HeadState:
    active_requests = int(kwargs["active_requests"])
    hidden_rows_per_request = int(kwargs.get("hidden_rows_per_request", 8))
    proposal_tokens = int(kwargs.get("proposal_tokens", 7))
    hidden_size = int(kwargs["hidden_size"])
    selector_rank = int(kwargs["selector_rank"])
    selector_top_k = int(kwargs["selector_top_k"])
    vocab_size = int(kwargs["vocab_size"])
    topk_backend = os.environ.get(_DFLASH2_TOPK_BACKEND_ENV, "torch").strip().lower()
    seed = int(kwargs["seed"])
    initialize_hidden = bool(kwargs.get("initialize_hidden", True))
    if active_requests not in (1, 2, 4):
        raise ValueError("DFlash2 head active_requests must be 1, 2, or 4")
    if (
        not 1 <= proposal_tokens <= 7
        or hidden_rows_per_request != proposal_tokens + 1
        or hidden_size != 6144
        or selector_rank != 256
        or selector_top_k != 16
        or vocab_size != 154880
    ):
        raise ValueError("head geometry does not match incoai/GLM-5.3-DFlash2")
    if topk_backend not in _DFLASH2_TOPK_BACKENDS:
        raise ValueError(
            f"{_DFLASH2_TOPK_BACKEND_ENV} must be torch, flashinfer, or "
            f"flashinfer-dsa, got {topk_backend!r}"
        )

    buffers = ctx["buffers"]
    mutable_names = (
        "hidden",
        "hidden_position_major",
        "logits",
        "unary",
        "candidates",
        "radix_candidates",
        "radix_row_states",
        "projected_hidden",
        "token_steps",
        "anchor_tokens",
        "output_tokens",
        "reference_tokens",
        "eager_tokens",
    )
    weight_names = (
        "lm_head",
        "hidden_projection",
        "predecessor_codebook",
        "successor_codebook",
    )
    required_names = (*mutable_names, *weight_names)
    missing = [name for name in required_names if name not in buffers]
    if missing:
        raise ValueError(f"DFlash2 head is missing buffers: {missing}")
    device_id = int(buffers["hidden"]["device_id"])
    for name in required_names:
        if int(buffers[name]["device_id"]) != device_id:
            raise ValueError(f"DFlash2 head buffer {name} is on another device")

    rows = active_requests * proposal_tokens
    shapes = {
        "hidden": (active_requests, hidden_rows_per_request, hidden_size),
        "hidden_position_major": (proposal_tokens, active_requests, hidden_size),
        "logits": (proposal_tokens, active_requests, vocab_size),
        "unary": (proposal_tokens, active_requests, selector_top_k),
        "candidates": (proposal_tokens, active_requests, selector_top_k),
        "radix_candidates": (proposal_tokens, active_requests, selector_top_k),
        "radix_row_states": (1024 * 1024,),
        "projected_hidden": (proposal_tokens, active_requests, selector_rank),
        "token_steps": (proposal_tokens, active_requests),
        "anchor_tokens": (active_requests,),
        "output_tokens": (active_requests, proposal_tokens),
        "reference_tokens": (active_requests, proposal_tokens),
        "eager_tokens": (active_requests, proposal_tokens),
        "lm_head": (vocab_size, hidden_size),
        "hidden_projection": (selector_rank, hidden_size),
        "predecessor_codebook": (vocab_size, selector_rank),
        "successor_codebook": (vocab_size, selector_rank),
    }
    i64_names = {
        "candidates",
        "token_steps",
        "anchor_tokens",
        "output_tokens",
        "reference_tokens",
        "eager_tokens",
    }
    i32_names = {"radix_candidates"}
    u8_names = {"radix_row_states"}
    for name, shape in shapes.items():
        element_bytes = (
            8
            if name in i64_names
            else 4 if name in i32_names else 1 if name in u8_names else 2
        )
        required_bytes = _tensor_bytes(shape, element_bytes)
        if int(buffers[name]["bytes"]) < required_bytes:
            raise ValueError(
                f"DFlash2 head buffer {name} has {buffers[name]['bytes']} bytes, "
                f"requires {required_bytes}"
            )

    key = (
        int(ctx["cuda_stream"]),
        active_requests,
        hidden_rows_per_request,
        proposal_tokens,
        topk_backend,
        initialize_hidden,
        *((name, int(buffers[name]["ptr"])) for name in required_names),
    )
    state = _DFLASH2_HEAD_STATES.get(key)
    if state is not None:
        return state
    if not create:
        raise RuntimeError("DFlash2 head capture requires a matching prepare call")

    import torch

    cuda_stream = int(ctx["cuda_stream"])
    stream = torch.cuda.ExternalStream(cuda_stream, device=device_id)
    with torch.cuda.device(device_id), torch.cuda.stream(stream), torch.no_grad():
        tensors: dict[str, Any] = {}
        for name, shape in shapes.items():
            if name in i64_names:
                tensors[name] = _i64_tensor(buffers[name], shape)
            elif name in i32_names:
                tensors[name] = _i32_tensor(buffers[name], shape)
            elif name in u8_names:
                tensors[name] = _u8_tensor(buffers[name], shape)
            else:
                tensors[name] = _bf16_tensor(buffers[name], shape)
        if initialize_hidden:
            generator = torch.Generator(device=device_id)
            generator.manual_seed(seed)
            tensors["hidden"].normal_(generator=generator)
        if topk_backend != "torch":
            # FlashInfer's public cache creates this scratch buffer zeroed.
            # Our caller-owned arena is intentionally uninitialized, so match
            # that one-time setup before any eager or captured radix launch.
            tensors["radix_row_states"].zero_()
        flashinfer_topk_module = (
            _flashinfer_raw_topk_module() if topk_backend != "torch" else None
        )

    state = _DFlash2HeadState(
        device_id=device_id,
        cuda_stream=cuda_stream,
        active_requests=active_requests,
        hidden_rows_per_request=hidden_rows_per_request,
        proposal_tokens=proposal_tokens,
        hidden_size=hidden_size,
        selector_rank=selector_rank,
        selector_top_k=selector_top_k,
        vocab_size=vocab_size,
        topk_backend=topk_backend,
        hidden=tensors["hidden"][:, 1:, :],
        hidden_position_major=tensors["hidden_position_major"],
        logits=tensors["logits"],
        unary=tensors["unary"],
        candidates=tensors["candidates"],
        radix_candidates=tensors["radix_candidates"],
        radix_row_states=tensors["radix_row_states"],
        projected_hidden=tensors["projected_hidden"],
        token_steps=tensors["token_steps"],
        anchor_tokens=tensors["anchor_tokens"],
        output_tokens=tensors["output_tokens"],
        reference_tokens=tensors["reference_tokens"],
        eager_tokens=tensors["eager_tokens"],
        lm_head_t=tensors["lm_head"].t(),
        hidden_projection_t=tensors["hidden_projection"].t(),
        predecessor_codebook=tensors["predecessor_codebook"],
        successor_codebook=tensors["successor_codebook"],
        flashinfer_topk_module=flashinfer_topk_module,
    )
    _DFLASH2_HEAD_STATES[key] = state
    print(
        "glmrt_dflash2_topk "
        f"device={device_id} requests={active_requests} drafts={proposal_tokens} "
        f"backend={topk_backend}",
        file=sys.stderr,
    )
    return state


@cache
def _flashinfer_raw_topk_module() -> Any:
    from flashinfer.jit.topk import gen_topk_module

    # Use the generated module directly: FlashInfer's public wrapper allocates
    # its int32 indices on every call, while the underlying kernel accepts the
    # fixed-address output and row-state buffers retained by our CUDA graph.
    return gen_topk_module().build_and_load()


def _base_logits(state: _DFlash2HeadState) -> Any:
    import torch

    if state.active_requests == 1:
        # The aliased body output is request-major and contains one anchor row
        # before the proposal rows.  With one request the proposal slice is
        # already one contiguous [K, H] view, so a position-major staging copy
        # would only add a graph node and rewrite the same bytes.
        flat_hidden = state.hidden.reshape(-1, state.hidden_size)
    else:
        state.hidden_position_major.copy_(state.hidden.permute(1, 0, 2))
        flat_hidden = state.hidden_position_major.view(-1, state.hidden_size)
    torch.mm(flat_hidden, state.lm_head_t, out=state.logits.view(-1, state.vocab_size))
    return flat_hidden


def _torch_topk(state: _DFlash2HeadState) -> None:
    import torch

    torch.topk(
        state.logits,
        state.selector_top_k,
        dim=-1,
        largest=True,
        sorted=True,
        out=(state.unary, state.candidates),
    )


def _flashinfer_topk(state: _DFlash2HeadState) -> None:
    if state.flashinfer_topk_module is None:
        raise RuntimeError("DFlash2 FlashInfer top-k module was not prepared")
    state.flashinfer_topk_module.radix_topk(
        state.logits.view(-1, state.vocab_size),
        state.radix_candidates.view(-1, state.selector_top_k),
        state.unary.view(-1, state.selector_top_k),
        state.radix_row_states,
        state.selector_top_k,
        True,
        True,
        0,
        state.topk_backend == "flashinfer-dsa",
    )


def _project_hidden(state: _DFlash2HeadState, flat_hidden: Any) -> None:
    import torch

    torch.mm(
        flat_hidden,
        state.hidden_projection_t,
        out=state.projected_hidden.view(-1, state.selector_rank),
    )


def _run_reference(state: _DFlash2HeadState) -> None:
    import torch
    import torch.nn.functional as functional

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        flat_hidden = _base_logits(state)
        _torch_topk(state)
        _project_hidden(state, flat_hidden)
        predecessor = state.anchor_tokens
        for position in range(state.proposal_tokens):
            predecessor_embedding = functional.embedding(
                predecessor, state.predecessor_codebook
            )
            successor_embedding = functional.embedding(
                state.candidates[position], state.successor_codebook
            )
            transition = torch.einsum(
                "br,bkr->bk",
                predecessor_embedding * state.projected_hidden[position],
                successor_embedding,
            )
            # The published DFlash2 selector adds its BF16 top-k unary and
            # BF16 einsum edge directly, so the greedy comparison observes a
            # BF16-rounded score.  Keep that boundary rather than silently
            # introducing an FP32 selector variant.
            scores = state.unary[position] + transition
            index = torch.argmax(scores, dim=-1)
            predecessor = state.candidates[position].gather(1, index[:, None])[:, 0]
            state.reference_tokens[:, position].copy_(predecessor)


def _run_head(state: _DFlash2HeadState) -> None:
    import torch

    stream = torch.cuda.ExternalStream(state.cuda_stream, device=state.device_id)
    with torch.cuda.device(state.device_id), torch.cuda.stream(stream), torch.no_grad():
        flat_hidden = _base_logits(state)
        if state.topk_backend == "torch":
            _torch_topk(state)
            candidate_tokens = state.candidates
        else:
            _flashinfer_topk(state)
            candidate_tokens = state.radix_candidates
        candidate_dtype = "int64" if state.topk_backend == "torch" else "int32"
        selector_num_warps = dflash2_selector_num_warps(
            state.active_requests,
            state.proposal_tokens,
            candidate_dtype,
        )
        _project_hidden(state, flat_hidden)
        predecessor = state.anchor_tokens
        for position in range(state.proposal_tokens):
            _dflash2_select_candidate[(state.active_requests,)](
                state.predecessor_codebook,
                predecessor,
                state.projected_hidden[position],
                state.successor_codebook,
                candidate_tokens[position],
                state.unary[position],
                state.token_steps[position],
                state.output_tokens,
                RANK=state.selector_rank,
                TOP_K=state.selector_top_k,
                POSITION=position,
                PROPOSAL_TOKENS=state.proposal_tokens,
                RANK_BLOCK=triton.next_power_of_2(state.selector_rank),
                num_warps=selector_num_warps,
            )
            predecessor = state.token_steps[position]


@triton.jit
def _dflash2_select_candidate(
    predecessor_codebook,
    predecessor_tokens,
    hidden,
    successor_codebook,
    candidates,
    unary,
    predecessor_output,
    final_output,
    RANK: tl.constexpr,
    TOP_K: tl.constexpr,
    POSITION: tl.constexpr,
    PROPOSAL_TOKENS: tl.constexpr,
    RANK_BLOCK: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    rank_offsets = tl.arange(0, RANK_BLOCK)
    rank_mask = rank_offsets < RANK
    predecessor_token = tl.load(predecessor_tokens + row)
    previous = tl.load(
        predecessor_codebook + predecessor_token * RANK + rank_offsets,
        mask=rank_mask,
        other=0.0,
    )
    current = tl.load(hidden + row * RANK + rank_offsets, mask=rank_mask, other=0.0)
    conditioned = (previous * current).to(tl.bfloat16)

    candidate_offsets = tl.arange(0, TOP_K)
    candidate_tokens = tl.load(candidates + row * TOP_K + candidate_offsets)
    successor = tl.load(
        successor_codebook + candidate_tokens[:, None] * RANK + rank_offsets[None, :],
        mask=rank_mask[None, :],
        other=0.0,
    )
    # Upstream evaluates this edge with a BF16 einsum. Use the tensor-core dot
    # reduction class here too: a scalar tl.sum can choose a different greedy
    # path when two BF16-rounded candidate scores are nearly tied.
    conditioned_matrix = tl.broadcast_to(
        conditioned[:, None], (RANK_BLOCK, TOP_K)
    )
    transition_matrix = tl.dot(
        successor, conditioned_matrix, out_dtype=tl.float32
    )
    transition = tl.sum(
        tl.where(candidate_offsets[None, :] == 0, transition_matrix, 0.0), axis=1
    ).to(tl.bfloat16)
    unary_scores = tl.load(unary + row * TOP_K + candidate_offsets)
    scores = (unary_scores.to(tl.bfloat16) + transition).to(tl.bfloat16)
    best = tl.max(scores, axis=0)
    best_index = tl.min(tl.where(scores == best, candidate_offsets, TOP_K), axis=0)
    token = tl.load(candidates + row * TOP_K + best_index)
    tl.store(predecessor_output + row, token)
    tl.store(final_output + row * PROPOSAL_TOKENS + POSITION, token)


def _tensor_bytes(shape: tuple[int, ...], element_bytes: int) -> int:
    values = 1
    for dimension in shape:
        values *= int(dimension)
    return values * element_bytes
