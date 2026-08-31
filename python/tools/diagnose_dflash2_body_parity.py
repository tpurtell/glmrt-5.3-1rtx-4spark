#!/usr/bin/env python3
"""Compare GLMRT's DFlash2 body with the pinned upstream PyTorch model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch

TOOLS = Path(__file__).resolve().parent
REFERENCE = TOOLS.parent / "reference" / "glmrt_reference"
sys.path.insert(0, str(REFERENCE))
from dspark_body_capture import _body_state, _run_body  # noqa: E402
from dspark_update_capture import _run_update, _update_state  # noqa: E402
from dflash_head_capture import _head_state, _run_head  # noqa: E402


def _load_model_class(source: Path) -> Any:
    spec = importlib.util.spec_from_file_location("glmrt_dflash_upstream", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import upstream DFlash source {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DFlash2DraftModel


def _buffer(tensor: torch.Tensor) -> dict[str, int]:
    assert tensor.is_cuda and tensor.is_contiguous()
    return {
        "ptr": tensor.data_ptr(),
        "bytes": tensor.numel() * tensor.element_size(),
        "device_id": tensor.device.index or 0,
    }


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    left = actual.float()
    right = expected.float()
    difference = left - right
    denominator = torch.linalg.vector_norm(right)
    return {
        "exact": bool(torch.equal(actual, expected)),
        "different_values": int(torch.count_nonzero(actual != expected).item()),
        "maximum_absolute": float(difference.abs().max().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(difference) / denominator).item()
        ),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                left.flatten(), right.flatten(), dim=0
            ).item()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--upstream-model-source", type=Path, required=True)
    parser.add_argument(
        "--target-lm-head-shard",
        type=Path,
        help="Optional target safetensors shard containing lm_head.weight",
    )
    parser.add_argument(
        "--live-trace-dir",
        type=Path,
        help=(
            "Optional live GLMRT layer-boundary trace. When supplied, replay the "
            "eight-row prefill plus row-8 decode taps instead of random inputs."
        ),
    )
    parser.add_argument(
        "--target-embedding-shard",
        type=Path,
        help="Target safetensors shard containing model.embed_tokens.weight.",
    )
    parser.add_argument("--live-anchor-id", type=int, default=9_703)
    parser.add_argument("--candidate-anchor-count", type=int, default=64)
    parser.add_argument("--context-rows", type=int, default=4)
    parser.add_argument("--query-rows", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.live_trace_dir is not None and args.target_embedding_shard is None:
        parser.error("--live-trace-dir requires --target-embedding-shard")
    if args.live_trace_dir is not None:
        args.context_rows = 9
        args.query_rows = 8
    if not 1 <= args.context_rows <= 8 or not 2 <= args.query_rows <= 8:
        if args.live_trace_dir is None:
            parser.error("context rows must be 1..8 and query rows must be 2..8")

    model_class = _load_model_class(args.upstream_model_source.resolve(strict=True))
    model = model_class.from_pretrained(
        args.snapshot.resolve(strict=True), dtype=torch.bfloat16
    ).cuda().eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    context_rows = args.context_rows
    query_rows = args.query_rows
    hidden_size = 6_144
    intermediate_size = 12_288
    layers = 6
    heads = 64
    kv_heads = 8
    head_dim = 128
    page_size = 64 if context_rows + query_rows > 16 else 16
    total_pages = 136 if args.live_trace_dir is not None else 1
    max_pages = 34 if args.live_trace_dir is not None else 1
    total_kv = context_rows + query_rows

    if args.live_trace_dir is None:
        target_features = torch.randn(
            (1, context_rows, layers * hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        noise = torch.randn(
            (1, query_rows, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
    else:
        from safetensors import safe_open

        trace_dir = args.live_trace_dir.resolve(strict=True)
        target_layers = (5, 19, 33, 47, 61, 75)
        tap_rows = []
        for target_layer in target_layers:
            prefill_path = trace_dir / (
                f"layer_{target_layer:02d}_prefillchunk_start_0_rows_8_post_mlp.bf16"
            )
            decode_path = trace_dir / (
                f"layer_{target_layer:02d}_decodestep_start_8_rows_1_post_mlp.bf16"
            )
            prefill = torch.from_file(
                str(prefill_path.resolve(strict=True)),
                shared=False,
                size=8 * hidden_size,
                dtype=torch.bfloat16,
            ).reshape(8, hidden_size)
            decode = torch.from_file(
                str(decode_path.resolve(strict=True)),
                shared=False,
                size=hidden_size,
                dtype=torch.bfloat16,
            ).reshape(1, hidden_size)
            tap_rows.append(torch.cat((prefill, decode), dim=0))
        target_features = torch.cat(tap_rows, dim=-1).unsqueeze(0).to(device)
        with safe_open(
            args.target_embedding_shard.resolve(strict=True),
            framework="pt",
            device="cpu",
        ) as handle:
            embeddings = handle.get_tensor("model.embed_tokens.weight").to(device)
        noise_ids = torch.tensor(
            [[args.live_anchor_id] + [int(model.mask_token_id)] * (query_rows - 1)],
            dtype=torch.int64,
            device=device,
        )
        noise = torch.nn.functional.embedding(noise_ids, embeddings)
    positions = torch.arange(total_kv, dtype=torch.int64, device=device)[None]
    upstream_trace: dict[str, torch.Tensor] = {}
    hooks = []
    for layer_index, layer in enumerate(model.layers):
        def save_output(name: str):
            def hook(_module, _args, output):
                value = output[0] if isinstance(output, tuple) else output
                upstream_trace[name] = value.detach().clone()
            return hook

        def save_input(name: str):
            def hook(_module, args, kwargs):
                value = args[0] if args else kwargs["hidden_states"]
                upstream_trace[name] = value.detach().clone()
            return hook

        hooks.extend(
            (
                layer.input_layernorm.register_forward_hook(
                    save_output(f"layer_{layer_index}.input_norm")
                ),
                layer.self_attn.register_forward_pre_hook(
                    save_input(f"layer_{layer_index}.attention_input"),
                    with_kwargs=True,
                ),
                layer.self_attn.register_forward_hook(
                    save_output(f"layer_{layer_index}.attention_output")
                ),
                layer.self_attn.q_norm.register_forward_hook(
                    save_output(f"layer_{layer_index}.q_norm")
                ),
                layer.self_attn.o_proj.register_forward_pre_hook(
                    save_input(f"layer_{layer_index}.attention_values"),
                    with_kwargs=True,
                ),
                layer.post_attention_layernorm.register_forward_pre_hook(
                    save_input(f"layer_{layer_index}.attention_residual"),
                    with_kwargs=True,
                ),
                layer.post_attention_layernorm.register_forward_hook(
                    save_output(f"layer_{layer_index}.post_norm")
                ),
                layer.mlp.register_forward_pre_hook(
                    save_input(f"layer_{layer_index}.mlp_input"),
                    with_kwargs=True,
                ),
                layer.mlp.register_forward_hook(
                    save_output(f"layer_{layer_index}.mlp_output")
                ),
                layer.mlp.down_proj.register_forward_pre_hook(
                    save_input(f"layer_{layer_index}.activation"),
                    with_kwargs=True,
                ),
                layer.register_forward_hook(
                    save_output(f"layer_{layer_index}.output")
                ),
            )
        )
    with torch.inference_mode():
        expected = model(
            target_hidden=target_features,
            noise_embedding=noise,
            position_ids=positions,
            use_cache=False,
        )
        fused_target = model.hidden_norm(model.fc(target_features))
    for hook in hooks:
        hook.remove()

    owned: dict[str, torch.Tensor] = {}

    def own(name: str, tensor: torch.Tensor) -> torch.Tensor:
        owned[name] = tensor.contiguous()
        return owned[name]

    def empty(name: str, shape: tuple[int, ...], dtype=torch.bfloat16) -> None:
        own(name, torch.empty(shape, dtype=dtype, device=device))

    empty("input", (query_rows, hidden_size))
    owned["input"].copy_(noise[0])
    for name in (
        "output",
        "reference_output",
        "hidden_attention",
        "hidden_mlp",
        "normalized",
    ):
        empty(name, (query_rows, hidden_size))
    empty("qkv", (query_rows, (heads + 2 * kv_heads) * head_dim))
    empty("q", (query_rows, heads, head_dim))
    empty("attention", (query_rows, heads, head_dim))
    empty("delta", (query_rows, hidden_size))
    empty("gate_up", (query_rows, 2 * intermediate_size))
    empty("activation", (query_rows, intermediate_size))
    empty("conv_dynamic", (query_rows, 4 * (hidden_size // 16)))
    empty("conv_output", (query_rows, hidden_size))
    empty("k_cache", (layers, total_pages, kv_heads, page_size, head_dim))
    empty("v_cache", (layers, total_pages, kv_heads, page_size, head_dim))
    own("workspace", torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device))
    own("query_lengths", torch.tensor([query_rows], dtype=torch.int32, device=device))
    own("kv_lengths", torch.tensor([total_kv], dtype=torch.int32, device=device))
    own(
        "query_positions",
        torch.arange(context_rows, total_kv, dtype=torch.int32, device=device),
    )
    own(
        "block_tables",
        torch.arange(max_pages, dtype=torch.int32, device=device).unsqueeze(0),
    )
    own("query_offsets", torch.tensor([0, query_rows * heads * head_dim], dtype=torch.int64, device=device))
    own("output_offsets", torch.tensor([0, query_rows * heads * head_dim], dtype=torch.int64, device=device))
    own("query_indptr", torch.tensor([0, query_rows], dtype=torch.int32, device=device))
    own("kv_indptr", torch.tensor([0, 1], dtype=torch.int32, device=device))
    own(
        "page_indices",
        torch.arange(total_pages, dtype=torch.int32, device=device),
    )
    own("last_page_len", torch.tensor([total_kv], dtype=torch.int32, device=device))
    own("final_norm", model.norm.weight)

    cos, sin = model.rotary_emb(noise, positions)
    for layer_index, layer in enumerate(model.layers):
        attention_input = upstream_trace[f"layer_{layer_index}.attention_input"]
        with torch.inference_mode():
            upstream_trace[f"layer_{layer_index}.qkv_projection"] = torch.cat(
                (
                    layer.self_attn.q_proj(attention_input),
                    layer.self_attn.k_proj(attention_input),
                    layer.self_attn.v_proj(attention_input),
                ),
                dim=-1,
            )
        q_normalized = upstream_trace.pop(f"layer_{layer_index}.q_norm").transpose(1, 2)
        q_cos = cos[:, None, -query_rows:, :]
        q_sin = sin[:, None, -query_rows:, :]
        q_first, q_second = q_normalized.chunk(2, dim=-1)
        q_rotated = torch.cat((-q_second, q_first), dim=-1)
        upstream_trace[f"layer_{layer_index}.q_rope"] = (
            q_normalized * q_cos + q_rotated * q_sin
        ).transpose(1, 2)
        prefix = f"layer_{layer_index}"
        own(f"{prefix}_input_norm", layer.input_layernorm.weight)
        own(f"{prefix}_post_norm", layer.post_attention_layernorm.weight)
        own(f"{prefix}_q_norm", layer.self_attn.q_norm.weight)
        own(f"{prefix}_k_norm", layer.self_attn.k_norm.weight)
        own(
            f"{prefix}_qkv",
            torch.cat(
                (
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                ),
                dim=0,
            ),
        )
        own(f"{prefix}_output", layer.self_attn.o_proj.weight)
        own(
            f"{prefix}_gate_up",
            torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight), dim=0),
        )
        own(f"{prefix}_down", layer.mlp.down_proj.weight)
        own(f"{prefix}_attention_conv_base", layer.attention_conv.base_kernel)
        own(
            f"{prefix}_attention_conv_projection",
            layer.attention_conv.kernel_projection.weight,
        )
        own(f"{prefix}_mlp_conv_base", layer.mlp_conv.base_kernel)
        own(
            f"{prefix}_mlp_conv_projection",
            layer.mlp_conv.kernel_projection.weight,
        )

        with torch.inference_mode():
            keys = layer.self_attn.k_norm(
                layer.self_attn.k_proj(fused_target).view(
                    1, context_rows, kv_heads, head_dim
                )
            ).transpose(1, 2)
            values = layer.self_attn.v_proj(fused_target).view(
                1, context_rows, kv_heads, head_dim
            ).transpose(1, 2)
            key_cos = cos[:, None, :context_rows]
            key_sin = sin[:, None, :context_rows]
            key_first, key_second = keys.chunk(2, dim=-1)
            key_rotated = torch.cat((-key_second, key_first), dim=-1)
            keys = keys * key_cos + key_rotated * key_sin
            owned["k_cache"][layer_index, 0, :, :context_rows].copy_(keys[0])
            owned["v_cache"][layer_index, 0, :, :context_rows].copy_(values[0])

    buffers = {name: _buffer(tensor) for name, tensor in owned.items()}
    ctx = {
        "cuda_stream": torch.cuda.current_stream(device).cuda_stream,
        "buffers": buffers,
    }
    kwargs = {
        "layers": layers,
        "active_requests": 1,
        "query_rows": query_rows,
        "total_pages": total_pages,
        "page_size": page_size,
        "max_pages_per_request": max_pages,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "rope_theta": 1_000_000.0,
        "conv_group_size": 16,
        "sliding_window": 2_048,
        "seed": args.seed,
        "initialize_input": False,
        "initialize_kv": False,
        "cache_dtype": "bf16",
    }
    state = _body_state(ctx, kwargs, create=True)
    glmrt_trace: dict[str, torch.Tensor] = {}
    _run_body(state, glmrt_trace)
    torch.cuda.synchronize()
    trace_metrics = {
        name: _metrics(glmrt_trace[name].view_as(value), value)
        for name, value in upstream_trace.items()
        if name in glmrt_trace
    }
    report = {
        "context_rows": context_rows,
        "query_rows": query_rows,
        "seed": args.seed,
        "live_trace_dir": (
            str(args.live_trace_dir.resolve())
            if args.live_trace_dir is not None
            else None
        ),
        "output": _metrics(owned["output"].view_as(expected), expected),
        "trace": trace_metrics,
    }
    projection_semantics = {}
    with torch.inference_mode():
        for layer_index, layer in enumerate(model.layers):
            attention_input = upstream_trace[f"layer_{layer_index}.attention_input"]
            q_expected = layer.self_attn.q_proj(attention_input)
            k_expected = layer.self_attn.k_proj(attention_input)
            v_expected = layer.self_attn.v_proj(attention_input)
            qkv_fused = torch.nn.functional.linear(
                attention_input,
                torch.cat(
                    (
                        layer.self_attn.q_proj.weight,
                        layer.self_attn.k_proj.weight,
                        layer.self_attn.v_proj.weight,
                    ),
                    dim=0,
                ),
            )
            q_actual, k_actual, v_actual = qkv_fused.split(
                (q_expected.shape[-1], k_expected.shape[-1], v_expected.shape[-1]),
                dim=-1,
            )
            mlp_input = upstream_trace[f"layer_{layer_index}.mlp_input"]
            gate_expected = layer.mlp.gate_proj(mlp_input)
            up_expected = layer.mlp.up_proj(mlp_input)
            gate_up_fused = torch.nn.functional.linear(
                mlp_input,
                torch.cat((layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight), dim=0),
            )
            gate_actual, up_actual = gate_up_fused.chunk(2, dim=-1)
            activation_expected = torch.nn.functional.silu(gate_expected) * up_expected
            activation_reassociated = (
                gate_expected.float()
                * torch.sigmoid(gate_expected.float())
                * up_expected.float()
            ).to(torch.bfloat16)
            glmrt_qkv = glmrt_trace[f"layer_{layer_index}.qkv_projection"].view(
                1, query_rows, -1
            )
            glmrt_q_source = glmrt_qkv[..., : heads * head_dim].view(
                1, query_rows, heads, head_dim
            )
            q_from_glmrt = layer.self_attn.q_norm(glmrt_q_source).transpose(1, 2)
            q_first, q_second = q_from_glmrt.chunk(2, dim=-1)
            q_rotated = torch.cat((-q_second, q_first), dim=-1)
            exact_q_rope_from_glmrt = (
                q_from_glmrt * cos[:, None, -query_rows:, :]
                + q_rotated * sin[:, None, -query_rows:, :]
            ).transpose(1, 2)
            glmrt_gate_up = glmrt_trace[
                f"layer_{layer_index}.gate_up_projection"
            ].view(1, query_rows, 2 * intermediate_size)
            glmrt_gate, glmrt_up = glmrt_gate_up.chunk(2, dim=-1)
            exact_activation_from_glmrt = (
                torch.nn.functional.silu(glmrt_gate) * glmrt_up
            )
            projection_semantics[f"layer_{layer_index}"] = {
                "fused_q": _metrics(q_actual, q_expected),
                "fused_k": _metrics(k_actual, k_expected),
                "fused_v": _metrics(v_actual, v_expected),
                "fused_gate": _metrics(gate_actual, gate_expected),
                "fused_up": _metrics(up_actual, up_expected),
                "reassociated_activation": _metrics(
                    activation_reassociated, activation_expected
                ),
                "q_rope_kernel": _metrics(
                    glmrt_trace[f"layer_{layer_index}.q_rope"].view_as(
                        exact_q_rope_from_glmrt
                    ),
                    exact_q_rope_from_glmrt,
                ),
                "activation_kernel": _metrics(
                    glmrt_trace[f"layer_{layer_index}.activation"].view_as(
                        exact_activation_from_glmrt
                    ),
                    exact_activation_from_glmrt,
                ),
            }
    report["projection_semantics"] = projection_semantics
    if args.target_lm_head_shard is not None:
        from safetensors import safe_open

        shard = args.target_lm_head_shard.resolve(strict=True)
        with safe_open(shard, framework="pt", device="cpu") as handle:
            lm_head = handle.get_tensor("lm_head.weight").to(device)
        actual = owned["output"].view_as(expected)
        expected_logits = torch.nn.functional.linear(expected, lm_head)
        actual_logits = torch.nn.functional.linear(actual, lm_head)
        top_k = int(model.candidate_selector.top_k)
        expected_top = torch.topk(expected_logits, top_k, dim=-1, sorted=False).indices
        actual_top = torch.topk(actual_logits, top_k, dim=-1, sorted=False).indices
        overlaps = []
        for position in range(query_rows):
            expected_set = set(expected_top[0, position].tolist())
            actual_set = set(actual_top[0, position].tolist())
            overlaps.append(len(expected_set & actual_set))

        anchor_generator = torch.Generator(device=device)
        anchor_generator.manual_seed(args.seed + 1)
        anchors = torch.randint(
            0,
            int(model.config.vocab_size),
            (args.candidate_anchor_count,),
            dtype=torch.int64,
            device=device,
            generator=anchor_generator,
        )
        expected_tokens, _, _ = model.candidate_selector.select(
            expected.expand(args.candidate_anchor_count, -1, -1),
            expected_logits.expand(args.candidate_anchor_count, -1, -1),
            anchors,
            0.0,
        )
        actual_tokens, _, _ = model.candidate_selector.select(
            actual.expand(args.candidate_anchor_count, -1, -1),
            actual_logits.expand(args.candidate_anchor_count, -1, -1),
            anchors,
            0.0,
        )

        def legacy_fp32_select(hidden: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
            unary, candidates = torch.topk(
                logits.expand(args.candidate_anchor_count, -1, -1),
                top_k,
                dim=-1,
                sorted=True,
            )
            projected = model.candidate_selector.hidden_projection(
                hidden.expand(args.candidate_anchor_count, -1, -1)
            )
            predecessor = anchors
            path = []
            for position in range(query_rows):
                conditioned = (
                    model.candidate_selector.predecessor_codebook(predecessor)
                    * projected[:, position]
                ).to(torch.bfloat16)
                successor = model.candidate_selector.successor_codebook(
                    candidates[:, position]
                )
                transition = torch.einsum(
                    "br,bkr->bk", conditioned, successor
                ).to(torch.bfloat16)
                scores = unary[:, position].float() + transition.float()
                index = torch.argmax(scores, dim=-1)
                predecessor = candidates[:, position].gather(
                    -1, index[:, None]
                )[:, 0]
                path.append(predecessor)
            return torch.stack(path, dim=1)

        legacy_expected_tokens = legacy_fp32_select(expected, expected_logits)
        legacy_actual_tokens = legacy_fp32_select(actual, actual_logits)
        matches = expected_tokens == actual_tokens
        legacy_reference_matches = expected_tokens == legacy_expected_tokens
        legacy_body_matches = expected_tokens == legacy_actual_tokens
        prefix_lengths = []
        for row in matches:
            mismatch = torch.nonzero(~row, as_tuple=False)
            prefix_lengths.append(
                query_rows if mismatch.numel() == 0 else int(mismatch[0, 0].item())
            )
        report["candidate_semantics"] = {
            "anchor_count": args.candidate_anchor_count,
            "logits": _metrics(actual_logits, expected_logits),
            "unary_argmax_matches": int(
                torch.count_nonzero(
                    actual_logits.argmax(dim=-1) == expected_logits.argmax(dim=-1)
                ).item()
            ),
            "unary_argmax_total": query_rows,
            "top16_overlap_by_position": overlaps,
            "selected_token_matches": int(torch.count_nonzero(matches).item()),
            "selected_token_total": int(matches.numel()),
            "full_path_matches": int(torch.count_nonzero(matches.all(dim=-1)).item()),
            "legacy_fp32_vs_official_selected_token_matches": int(
                torch.count_nonzero(legacy_reference_matches).item()
            ),
            "legacy_fp32_vs_official_full_path_matches": int(
                torch.count_nonzero(legacy_reference_matches.all(dim=-1)).item()
            ),
            "legacy_fp32_body_vs_official_selected_token_matches": int(
                torch.count_nonzero(legacy_body_matches).item()
            ),
            "legacy_fp32_body_vs_official_full_path_matches": int(
                torch.count_nonzero(legacy_body_matches.all(dim=-1)).item()
            ),
            "prefix_length_mean": sum(prefix_lengths) / len(prefix_lengths),
            "prefix_length_minimum": min(prefix_lengths),
            "prefix_length_histogram": {
                str(length): prefix_lengths.count(length)
                for length in sorted(set(prefix_lengths))
            },
        }
        if args.live_trace_dir is not None:
            official_hidden = expected[:, 1:, :]
            glmrt_hidden = actual[:, 1:, :]
            official_logits = torch.nn.functional.linear(official_hidden, lm_head)
            glmrt_logits = torch.nn.functional.linear(glmrt_hidden, lm_head)
            anchor = torch.tensor(
                [args.live_anchor_id], dtype=torch.int64, device=device
            )
            with torch.inference_mode():
                official_path, _, _ = model.candidate_selector.select(
                    official_hidden, official_logits, anchor, 0.0
                )
                glmrt_path, _, _ = model.candidate_selector.select(
                    glmrt_hidden, glmrt_logits, anchor, 0.0
                )

                def live_legacy_fp32_select(
                    hidden: torch.Tensor, logits: torch.Tensor
                ) -> torch.Tensor:
                    unary, candidates = torch.topk(
                        logits, top_k, dim=-1, sorted=True
                    )
                    projected = model.candidate_selector.hidden_projection(hidden)
                    predecessor = anchor
                    path = []
                    for position in range(hidden.shape[1]):
                        conditioned = (
                            model.candidate_selector.predecessor_codebook(predecessor)
                            * projected[:, position]
                        ).to(torch.bfloat16)
                        successor = model.candidate_selector.successor_codebook(
                            candidates[:, position]
                        )
                        transition = torch.einsum(
                            "br,bkr->bk", conditioned, successor
                        ).to(torch.bfloat16)
                        scores = unary[:, position].float() + transition.float()
                        index = torch.argmax(scores, dim=-1)
                        predecessor = candidates[:, position].gather(
                            -1, index[:, None]
                        )[:, 0]
                        path.append(predecessor)
                    return torch.stack(path, dim=1)

                official_fp32_path = live_legacy_fp32_select(
                    official_hidden, official_logits
                )
                glmrt_fp32_path = live_legacy_fp32_select(glmrt_hidden, glmrt_logits)
            report["live_candidate_semantics"] = {
                "anchor_id": args.live_anchor_id,
                "official_path": official_path[0].tolist(),
                "glmrt_body_with_official_context_path": glmrt_path[0].tolist(),
                "path_matches": (official_path[0] == glmrt_path[0]).tolist(),
                "official_body_legacy_fp32_selector_path": official_fp32_path[0].tolist(),
                "glmrt_body_legacy_fp32_selector_path": glmrt_fp32_path[0].tolist(),
                "observed_live_glmrt_path": [
                    4_080,
                    82,
                    1_090,
                    11,
                    4_400,
                    2_687,
                    4_223,
                ],
            }

            official_k_cache = owned["k_cache"].clone()
            official_v_cache = owned["v_cache"].clone()
            update_owned: dict[str, torch.Tensor] = {}

            def update_own(name: str, tensor: torch.Tensor) -> torch.Tensor:
                update_owned[name] = tensor.contiguous()
                return update_owned[name]

            def update_empty(
                name: str,
                shape: tuple[int, ...],
                dtype: torch.dtype = torch.bfloat16,
            ) -> None:
                update_own(name, torch.empty(shape, dtype=dtype, device=device))

            maximum_update_rows = 8
            update_empty(
                "target_hidden", (maximum_update_rows, layers * hidden_size)
            )
            update_owned["target_hidden"].copy_(target_features[0, :8])
            for name, shape in (
                ("fusion_output", (maximum_update_rows, hidden_size)),
                ("fused_hidden", (maximum_update_rows, hidden_size)),
                ("projected_kv", (maximum_update_rows, 2 * kv_heads * head_dim)),
                ("key_output", (layers, maximum_update_rows, kv_heads, head_dim)),
                ("value_output", (layers, maximum_update_rows, kv_heads, head_dim)),
                ("reference_fused_hidden", (maximum_update_rows, hidden_size)),
                ("reference_key_output", (layers, maximum_update_rows, kv_heads, head_dim)),
                ("reference_value_output", (layers, maximum_update_rows, kv_heads, head_dim)),
                ("eager_fused_hidden", (maximum_update_rows, hidden_size)),
                ("eager_key_output", (layers, maximum_update_rows, kv_heads, head_dim)),
                ("eager_value_output", (layers, maximum_update_rows, kv_heads, head_dim)),
            ):
                update_empty(name, shape)
            update_own("k_cache", owned["k_cache"])
            update_own("v_cache", owned["v_cache"])
            update_own(
                "row_request_ids",
                torch.zeros(maximum_update_rows, dtype=torch.int32, device=device),
            )
            update_own(
                "row_positions",
                torch.arange(maximum_update_rows, dtype=torch.int32, device=device),
            )
            update_own(
                "row_cache_positions",
                torch.arange(maximum_update_rows, dtype=torch.int32, device=device),
            )
            update_own(
                "block_tables",
                torch.arange(max_pages, dtype=torch.int32, device=device).unsqueeze(0),
            )
            update_own("target_fusion", model.fc.weight)
            update_own("hidden_norm", model.hidden_norm.weight)
            for layer_index, layer in enumerate(model.layers):
                update_own(f"layer_{layer_index}_k_norm", layer.self_attn.k_norm.weight)
                update_own(
                    f"layer_{layer_index}_kv",
                    torch.cat(
                        (
                            layer.self_attn.k_proj.weight,
                            layer.self_attn.v_proj.weight,
                        ),
                        dim=0,
                    ),
                )
            update_ctx = {
                "cuda_stream": torch.cuda.current_stream(device).cuda_stream,
                "buffers": {
                    name: _buffer(tensor) for name, tensor in update_owned.items()
                },
            }
            update_kwargs = {
                "rows": maximum_update_rows,
                "active_requests": 1,
                "layers": layers,
                "hidden_size": hidden_size,
                "target_features": layers * hidden_size,
                "heads": kv_heads,
                "head_dim": head_dim,
                "rope_theta": 1_000_000.0,
                "total_pages": total_pages,
                "page_size": page_size,
                "max_pages_per_request": max_pages,
                "seed": args.seed,
                "initialize_target_hidden": False,
                "initialize_kv": False,
                "cache_dtype": "bf16",
            }
            update_state = _update_state(update_ctx, update_kwargs, create=True)
            _run_update(update_state)
            torch.cuda.synchronize()
            first_fused_hidden = update_owned["fused_hidden"].clone()
            update_owned["target_hidden"][0].copy_(target_features[0, 8])
            update_owned["row_positions"][0] = 8
            update_owned["row_cache_positions"][0] = 8
            update_kwargs["rows"] = 1
            final_update_state = _update_state(update_ctx, update_kwargs, create=True)
            _run_update(final_update_state)
            torch.cuda.synchronize()
            native_fused_hidden = torch.cat(
                (first_fused_hidden, update_owned["fused_hidden"][:1].clone()), dim=0
            )
            native_update_k = owned["k_cache"].clone()
            native_update_v = owned["v_cache"].clone()
            _run_body(state)
            torch.cuda.synchronize()
            native_update_body_hidden = owned["output"].view_as(expected).clone()
            native_update_logits = torch.nn.functional.linear(
                native_update_body_hidden[:, 1:, :], lm_head
            )
            with torch.inference_mode():
                native_update_path, _, _ = model.candidate_selector.select(
                    native_update_body_hidden[:, 1:, :],
                    native_update_logits,
                    anchor,
                    0.0,
                )
            report["live_update_semantics"] = {
                "fused_hidden": _metrics(
                    native_fused_hidden, fused_target[0]
                ),
                "context_keys": _metrics(
                    native_update_k[:, 0, :, :context_rows, :],
                    official_k_cache[:, 0, :, :context_rows, :],
                ),
                "context_values": _metrics(
                    native_update_v[:, 0, :, :context_rows, :],
                    official_v_cache[:, 0, :, :context_rows, :],
                ),
                "body_output": _metrics(native_update_body_hidden, expected),
                "proposal_path": native_update_path[0].tolist(),
                "matches_official_path": (
                    native_update_path[0] == official_path[0]
                ).tolist(),
            }

            head_owned: dict[str, torch.Tensor] = {}

            def head_own(name: str, tensor: torch.Tensor) -> torch.Tensor:
                head_owned[name] = tensor.contiguous()
                return head_owned[name]

            def head_empty(
                name: str,
                shape: tuple[int, ...],
                dtype: torch.dtype = torch.bfloat16,
            ) -> None:
                head_own(name, torch.empty(shape, dtype=dtype, device=device))

            head_own("hidden", native_update_body_hidden)
            head_empty("hidden_position_major", (7, 1, hidden_size))
            head_empty("logits", (7, 1, int(model.config.vocab_size)))
            head_empty("unary", (7, 1, top_k))
            head_empty("candidates", (7, 1, top_k), torch.int64)
            head_empty("radix_candidates", (7, 1, top_k), torch.int32)
            head_empty("radix_row_states", (1024 * 1024,), torch.uint8)
            head_empty("projected_hidden", (7, 1, 256))
            head_empty("token_steps", (7, 1), torch.int64)
            head_own("anchor_tokens", anchor)
            head_empty("output_tokens", (1, 7), torch.int64)
            head_empty("reference_tokens", (1, 7), torch.int64)
            head_empty("eager_tokens", (1, 7), torch.int64)
            head_own("lm_head", lm_head)
            head_own(
                "hidden_projection",
                model.candidate_selector.hidden_projection.weight,
            )
            head_own(
                "predecessor_codebook",
                model.candidate_selector.predecessor_codebook.weight,
            )
            head_own(
                "successor_codebook",
                model.candidate_selector.successor_codebook.weight,
            )
            head_ctx = {
                "cuda_stream": torch.cuda.current_stream(device).cuda_stream,
                "buffers": {
                    name: _buffer(tensor) for name, tensor in head_owned.items()
                },
            }
            head_kwargs = {
                "active_requests": 1,
                "hidden_rows_per_request": 8,
                "proposal_tokens": 7,
                "hidden_size": hidden_size,
                "selector_rank": 256,
                "selector_top_k": top_k,
                "vocab_size": int(model.config.vocab_size),
                "seed": args.seed,
                "initialize_hidden": False,
            }
            head_state = _head_state(head_ctx, head_kwargs, create=True)
            _run_head(head_state)
            torch.cuda.synchronize()
            report["live_native_head_semantics"] = {
                "proposal_path": head_owned["output_tokens"][0].tolist(),
                "matches_official_path": (
                    head_owned["output_tokens"][0] == official_path[0]
                ).tolist(),
                "matches_observed_live_path": (
                    head_owned["output_tokens"][0]
                    == torch.tensor(
                        [4_080, 82, 1_090, 11, 4_400, 2_687, 4_223],
                        dtype=torch.int64,
                        device=device,
                    )
                ).tolist(),
            }

            metadata_paths = {}
            for metadata_name, kv_length, query_start in (
                ("correct", 17, 9),
                ("stale_query_positions", 17, 1),
                ("stale_kv_length", 9, 9),
                ("stale_both", 9, 1),
            ):
                owned["k_cache"].copy_(native_update_k)
                owned["v_cache"].copy_(native_update_v)
                owned["kv_lengths"].fill_(kv_length)
                owned["last_page_len"].fill_(kv_length)
                owned["query_positions"].copy_(
                    torch.arange(
                        query_start,
                        query_start + query_rows,
                        dtype=torch.int32,
                        device=device,
                    )
                )
                _run_body(state)
                torch.cuda.synchronize()
                metadata_hidden = owned["output"].view_as(expected).clone()[:, 1:, :]
                metadata_logits = torch.nn.functional.linear(metadata_hidden, lm_head)
                with torch.inference_mode():
                    metadata_path, _, _ = model.candidate_selector.select(
                        metadata_hidden, metadata_logits, anchor, 0.0
                    )
                metadata_paths[metadata_name] = metadata_path[0].tolist()
            report["live_metadata_sweep"] = metadata_paths
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
