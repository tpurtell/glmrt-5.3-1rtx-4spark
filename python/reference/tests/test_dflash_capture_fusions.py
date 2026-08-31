from __future__ import annotations

import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
BODY_CAPTURE = (
    ROOT / "python" / "reference" / "glmrt_reference" / "dspark_body_capture.py"
)
HEAD_CAPTURE = (
    ROOT / "python" / "reference" / "glmrt_reference" / "dflash_head_capture.py"
)
TOPK_TUNER = ROOT / "python" / "tools" / "tune_dflash2_topk.py"
SELECTOR_TUNER = ROOT / "python" / "tools" / "tune_dflash2_selector.py"
BODY_TUNER = ROOT / "python" / "tools" / "tune_dflash2_body_fusion.py"
W8_BODY_TUNER = ROOT / "python" / "tools" / "tune_dflash2_w8a16_body.py"
TUNING_PROFILE = (
    ROOT / "python" / "reference" / "glmrt_reference" / "dflash_tuning_profile.py"
)
RELEASE_DOCKERFILE = ROOT / "docker" / "Dockerfile.release"


def _annotated_fields(source: str, class_name: str) -> set[str]:
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return {
        item.target.id
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }


def _named_call_count(source: str, function_name: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    )


def _triton_call_count(source: str, function_name: str) -> int:
    return sum(
        1
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == function_name
    )


def test_dflash_head_fuses_transition_argmax_and_final_output_stores() -> None:
    source = HEAD_CAPTURE.read_text(encoding="utf-8")

    assert "state.output_tokens.copy_(state.token_steps.t())" not in source
    assert "transition_scores" not in source
    assert "def _dflash2_transition_scores(" not in source
    assert "def _dflash2_candidate_argmax(" not in source
    assert _triton_call_count(source, "_dflash2_select_candidate") == 1
    assert "tl.store(final_output + row * PROPOSAL_TOKENS + POSITION, token)" in source
    assert "predecessor_embedding" not in _annotated_fields(source, "_DFlash2HeadState")
    assert "state.active_requests == 1" in source
    assert "flat_hidden = state.hidden.reshape(-1, state.hidden_size)" in source
    assert "selector_num_warps = dflash2_selector_num_warps(" in source
    assert "num_warps=selector_num_warps" in source
    assert "scores = state.unary[position] + transition" in source
    assert "scores = (unary_scores.to(tl.bfloat16) + transition).to(tl.bfloat16)" in source


def test_dflash_body_uses_fused_dynamic_residual_norm_only_for_dflash() -> None:
    source = BODY_CAPTURE.read_text(encoding="utf-8")

    assert "fuse_dflash2_dynamic_residual_norm = state.conv_group_size != 0" in source
    assert _named_call_count(source, "_finish_dynamic_conv_add_rms_norm") == 2
    assert "def _finish_dynamic_conv(" not in source
    assert "tl.store(conv_output + offsets, convolved" not in source
    assert "not this noncausal block" in source
    assert 'backend="fa2" if sliding_window >= 0 else "auto"' in source
    assert "planning_pages = active_requests * planning_pages_per_request" in source
    assert "fixed_split_size=fixed_split_pages or None" in source
    assert "disable_split_kv=dflash2_geometry and fixed_split_pages == 0" in source
    assert "total_pages // active_requests" not in source
    # The retained GLM-5.2 dSpark branch still uses its established add path.
    assert source.count("_dspark_add[residual_add_grid](") == 2
    assert "num_warps=dflash2_body_num_warps(" in source


def test_fused_dflash_body_preserves_bf16_conv_and_residual_boundaries() -> None:
    width = 64
    group_size = 16
    groups = width // group_size
    columns = torch.arange(width)
    group_ids = columns // group_size
    skipped_boundary_changes_result = False

    for active_requests in (1, 2, 4):
        for query_rows in (2, 4, 8):
            rows = active_requests * query_rows
            generator = torch.Generator().manual_seed(rows * 101 + query_rows)
            source = torch.randn(rows, width, generator=generator).to(torch.bfloat16)
            dynamic = torch.randn(rows, 4 * groups, generator=generator).to(
                torch.bfloat16
            )
            base = torch.randn(2, 2, width, generator=generator).to(torch.bfloat16)
            residual = torch.randn(rows, width, generator=generator).to(torch.bfloat16)
            weight = torch.randn(width, generator=generator).to(torch.bfloat16)

            raw_convolved = []
            for row in range(rows):
                value = source[row].float() * (
                    base[1, 0].float() + dynamic[row, 2 * groups + group_ids].float()
                )
                if row % query_rows:
                    value += source[row - 1].float() * (
                        base[1, 1].float()
                        + dynamic[row, 3 * groups + group_ids].float()
                    )
                raw_convolved.append(value)
            raw_convolved_tensor = torch.stack(raw_convolved)

            separate_conv = raw_convolved_tensor.to(torch.bfloat16)
            separate_residual = (residual + separate_conv).to(torch.bfloat16)
            separate_norm = (
                separate_residual.float()
                * torch.rsqrt(
                    separate_residual.float().square().mean(-1, keepdim=True) + 1.0e-5
                )
                * weight.float()
            ).to(torch.bfloat16)

            fused_conv = raw_convolved_tensor.to(torch.bfloat16)
            fused_residual = (residual.float() + fused_conv.float()).to(torch.bfloat16)
            fused_norm = (
                fused_residual.float()
                * torch.rsqrt(
                    fused_residual.float().square().mean(-1, keepdim=True) + 1.0e-5
                )
                * weight.float()
            ).to(torch.bfloat16)

            assert torch.equal(fused_conv, separate_conv)
            assert torch.equal(fused_residual, separate_residual)
            assert torch.equal(fused_norm, separate_norm)
            skipped_boundary = (residual.float() + raw_convolved_tensor).to(
                torch.bfloat16
            )
            skipped_boundary_changes_result |= not torch.equal(
                skipped_boundary, separate_residual
            )

    assert skipped_boundary_changes_result


def test_dflash_fusions_have_complete_real_weight_performance_gates() -> None:
    selector = SELECTOR_TUNER.read_text(encoding="utf-8")
    body = BODY_TUNER.read_text(encoding="utf-8")

    for source in (selector, body):
        assert 'REPO_ID = "incoai/GLM-5.3-DFlash2"' in source
        assert 'default="1,2,4"' in source
        assert 'default="1,2,3,4,5,6,7"' in source
        assert "def _measure_pair(" in source
        assert "round_index % 2 == 0" in source
        assert "snapshot.name != REVISION" in source
        assert '"weight_sha256": _hash_file(weight_path)' in source
        assert '"fused_wins_all_cases": fused_wins_all_cases' in source
        assert '"runtime_matches_winners": runtime_matches_winners' in source
        assert "fused_wins_all_cases and runtime_matches_winners" in source
        assert "MIN_FUSED_SPEEDUP = 1.01" in source
        assert '"performance_gate_passed": fused_speedup >= MIN_FUSED_SPEEDUP' in source
        assert "torch.cuda.Stream()" in source
        assert "DEFAULT_CAPTURED_LAUNCHES = 16" in source
        assert "iterations * captured_launches" in source
    assert 'default="both"' in selector
    assert 'parser.add_argument("--fused-warps", default="4,8")' in selector
    assert "fused_output != reference" in selector
    assert 'default="4,8"' in body
    assert "fused_residual != split_residual" in body
    assert "fused_normalized != split_normalized" in body
    assert "dflash2_selector_num_warps(" in selector
    assert "dflash2_body_num_warps(" in body
    assert 'weights.get_tensor("norm.weight")' in body
    assert "for layer in range(6)" in body
    assert 'f"layer-{layer}-attention"' in body
    assert 'f"layer-{layer}-mlp"' in body
    assert '"real_weight_validation": real_weight_validation' in body
    for source in (selector, body):
        assert 'REFERENCE / "dflash_tuning_profile.py"' in source

    profile = TUNING_PROFILE.read_text(encoding="utf-8")
    assert "def dflash2_selector_num_warps(" in profile
    assert "def dflash2_body_num_warps(" in profile
    assert "for concurrency in _CONCURRENCY" in profile
    assert "for width in _WIDTHS" in profile


def test_dflash_topk_backend_has_complete_signed_selection_gate() -> None:
    source = TOPK_TUNER.read_text(encoding="utf-8")

    assert 'REPO_ID = "incoai/GLM-5.3-DFlash2"' in source
    assert 'REVISION = "425aa615ce320caac34400208b30808c8f14f76c"' in source
    assert 'BACKENDS = ("torch", "flashinfer", "flashinfer-dsa")' in source
    assert 'default="1,2,4"' in source
    assert 'default="1,2,3,4,5,6,7"' in source
    assert "round_index % len(backends)" in source
    assert "disabled_backends[backend] = str(error)" in source
    assert '"unsupported_backends": disabled_backends' in source
    assert '"status": "measured"' in source
    assert '"full_service_acceptance_required": True' in source
    assert "DEFAULT_CAPTURED_LAUNCHES = 32" in source
    assert "iterations * captured_launches" in source
    assert "aggregate_speedup >= MIN_NON_TORCH_SPEEDUP" in source
    assert "initial_valid, initial_index_exact = parity()" in source
    assert "changed_valid, changed_index_exact = parity()" in source
    assert '"tie_policy": "equal_topk_values_valid_unique_ids_boundary_ties_allowed"' in source
    assert "torch.gather(logits, -1, candidate_ids)" in source
    assert "sentinels" not in source
    assert '"report_sha256"' in source

    head = HEAD_CAPTURE.read_text(encoding="utf-8")
    assert "reference_unary = state.unary.clone()" in head
    assert "candidate_values_equal = torch.equal(" in head
    assert "values_equal = torch.equal(reference_unary, state.unary)" in head
    assert "glmrt_dflash2_topk_boundary_tie_output_delta" in head
    assert 'if state.topk_backend == "torch":' in head


def test_release_image_import_smokes_dflash_profile_and_head() -> None:
    source = RELEASE_DOCKERFILE.read_text(encoding="utf-8")

    assert "dflash_head_capture" in source
    assert "dflash_tuning_profile" in source


def test_dflash_w8_body_candidate_is_single_residency_and_full_surface_gated() -> None:
    source = W8_BODY_TUNER.read_text(encoding="utf-8")

    assert 'REPO_ID = "incoai/GLM-5.3-DFlash2"' in source
    assert 'REVISION = "425aa615ce320caac34400208b30808c8f14f76c"' in source
    assert "LIVE_ROWS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32)" in source
    for shape in ("10_240", "8_192", "24_576", "12_288", "1_536"):
        assert shape in source
    assert "BF16_PROJECTION_BYTES_PER_PASS != 4_303_355_904" in source
    assert "W8_PROJECTION_BYTES_PER_PASS != 2_185_297_920" in source
    assert 'parser.add_argument("--layers", default="0,1,2,3,4,5")' in source
    assert 'parser.add_argument("--rows", default="dflash")' in source
    assert "round_index % 2 == 0" in source
    assert "flush.add_(1)" in source
    assert '"single_residency_required": True' in source
    assert '"full_service_acceptance_required": True' in source
    assert '"status": "measured"' in source
    assert "speedup >= MIN_W8_SPEEDUP" in source
    assert '"promotable_to_full_service_gate"' in source
