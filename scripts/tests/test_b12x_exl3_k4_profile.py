from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
SPEC = importlib.util.spec_from_file_location(
    "_glmrt_exl3_k4_profile", TOOLS / "_b12x_exl3_k4_profile.py"
)
assert SPEC is not None and SPEC.loader is not None
PROFILE = importlib.util.module_from_spec(SPEC)
import sys

sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(PROFILE)
finally:
    sys.path.pop(0)


def test_exl3_k4_profile_covers_every_aot_bucket() -> None:
    assert PROFILE.EXL3_K4_AOT_REGIMES == (
        *range(1, 33),
        64,
        128,
        256,
        257,
        512,
        1024,
        2048,
        2064,
    )
    for rows in PROFILE.EXL3_K4_AOT_REGIMES:
        assert len(PROFILE.exl3_k4_tile_config(rows)) == 4
        assert PROFILE.exl3_k4_grid_x(rows) > 0
        assert PROFILE.exl3_k4_route_block_rows(rows) in (8, 16, 32, 48, 64)
    assert PROFILE.EXL3_K4_REQUIRED_LIVE_ROWS == (
        *range(1, 33),
        64,
        128,
        129,
        256,
        257,
        512,
        513,
        1024,
        1025,
        2048,
        2049,
        2064,
    )


def test_exl3_k4_capacity_selection_retains_exact_tail_buckets() -> None:
    assert PROFILE.exl3_k4_capacity_rows(1) == 1
    assert PROFILE.exl3_k4_capacity_rows(3) == 3
    assert PROFILE.exl3_k4_capacity_rows(7) == 7
    assert PROFILE.exl3_k4_capacity_rows(9) == 9
    assert PROFILE.exl3_k4_capacity_rows(10) == 10
    assert PROFILE.exl3_k4_capacity_rows(31) == 31
    assert PROFILE.exl3_k4_capacity_rows(33) == 64
    assert PROFILE.exl3_k4_capacity_rows(257) == 257
    assert PROFILE.exl3_k4_capacity_rows(258) == 512
    assert PROFILE.exl3_k4_capacity_rows(2049) == 2064


def test_exl3_k4_profile_is_an_explicit_aot_build_input() -> None:
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
    exporter = (ROOT / "python" / "tools" / "export_b12x_spark_moe_aot.py").read_text(
        encoding="utf-8"
    )

    assert (
        '"${CMAKE_CURRENT_SOURCE_DIR}/../python/tools/_b12x_exl3_k4_profile.py"'
        in cmake
    )
    assert "GLMRT_B12X_EXL3_K4_REGIMES" in cmake
    assert "from _b12x_exl3_k4_profile import" in exporter


def test_every_k4_aot_bucket_is_registered_end_to_end() -> None:
    native = (ROOT / "native" / "cuda" / "kernels" / "b12x_direct.cu").read_text(
        encoding="utf-8"
    )
    cmake = (ROOT / "native" / "CMakeLists.txt").read_text(encoding="utf-8")
    rust = (
        ROOT
        / "rust"
        / "crates"
        / "glmrt-daemon"
        / "src"
        / "commands"
        / "real_full"
        / "sparse_mlp"
        / "route.rs"
    ).read_text(encoding="utf-8")

    cmake_regimes = tuple(
        int(value)
        for value in re.findall(
            r"\d+",
            cmake.split("set(GLMRT_B12X_EXL3_K4_REGIMES", 1)[1].split(")", 1)[0],
        )
    )
    assert cmake_regimes == PROFILE.EXL3_K4_AOT_REGIMES
    for rows in PROFILE.EXL3_K4_AOT_REGIMES:
        assert f'#include "moe_tp4_exl3_k4_m{rows}_topk8.h"' in native
        assert f"GLMRT_DEFINE_EXL3_K4_MODULE({rows})" in native
        assert f"GLMRT_LOAD_EXL3_K4_MODULE({rows})" in native
        assert f"GLMRT_DEFINE_EXL3_K4_LAUNCH({rows})" in native
        assert (
            f"case {rows}:\n      return &launch_exl3_k4_m{rows}_topk8;" in native
        )
        assert f"GLMRT_EXL3_K4_GRID_CASE({rows})" in native

    assert native.count("rows <= 32 || rows == 257") == 2
    automatic_selectors = {
        symbol: native[
            native.index(symbol) : native.index('extern "C"', native.index(symbol))
        ]
        for symbol in (
            "glmrt_cuda_b12x_spark_exl3_k3_topk8_nvfp4_async",
            "glmrt_cuda_b12x_spark_exl3_k3_topk8_nvfp4_bf16_async",
            "glmrt_cuda_b12x_spark_exl3_k4_topk8_nvfp4_async",
            "glmrt_cuda_b12x_spark_exl3_k4_topk8_nvfp4_bf16_async",
        )
    }
    for symbol, body in automatic_selectors.items():
        if "_k4_" in symbol:
            assert "rows <= 32 || rows == 257" in body
        else:
            assert "rows == 9 || rows == 257" in body
            assert "rows <= 32" not in body
    assert "trellis_bits == 4 && rows <= 32" in rust
    assert "b12x_exl3_capacity_rows(rows, trellis_bits)?" in rust
    assert "exl3_trellis_bits: Option<usize>" in rust


def test_glm53_native_runbook_selects_k4_explicitly() -> None:
    runbook = (ROOT / "quantization" / "README.md").read_text(encoding="utf-8")
    qualifier = (
        ROOT / "python" / "tools" / "validate_glm53_exl3_serving_qualification.py"
    ).read_text(encoding="utf-8")

    assert "always pass `--trellis-bits 4`" in runbook
    assert '--trellis-bits "$trellis_bits"' in runbook
    assert '--expert-slot-fingerprint "$expert_slot_fingerprint"' in runbook
    required_rows = "--rows " + ",".join(
        str(row)
        for row in (
            *range(1, 33),
            64,
            128,
            129,
            256,
            257,
            512,
            513,
            1024,
            1025,
            2048,
            2049,
            2064,
        )
    )
    assert required_rows in runbook
    assert '"$glmrt_bin" dflash-preflight' in runbook
    assert "--kv-capacity-tokens 2176" in runbook
    assert "--max-concurrency 4" in runbook
    assert '--proposal-tokens-per-request "$dflash2_width"' in runbook
    assert "--preload --capture-static" in runbook
    assert "model_id=wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1" in runbook
    assert "quant_image=glmrt-quant-coordinator:exl3-k4" in runbook
    assert "--entrypoint /usr/bin/chown" in runbook
    assert '-R "$host_uid:$host_gid" "/glmrt-models/$raw_name"' in runbook
    acceptance = runbook.split(
        "raw_output_root=/fast-nvme/models/GLM-5.3-EXL3-K4-calibrated-v1", 1
    )[1].split("```", 1)[0]
    assert "glm53-exl3-k4-quant-evidence.json" in acceptance
    assert "glm53-exl3-k4-artifact-validation.json" in acceptance
    assert "validate_glm52_exl3_quant_evidence.py" in acceptance
    assert "validate_glm52_exl3_artifact.py" in acceptance
    assert "--verify-artifact-file-hashes" in acceptance
    assert "--tokenizer-attestation" not in acceptance
    assert runbook.count('--model-id "$model_id"') == 4
    assert ': "${model_id:?set model_id to the exact accepted artifact repository ID}"' in runbook
    assert qualifier.count("expected_trellis_bits=4") == 2
    assert qualifier.count("expected_expert_slot_fingerprint=") == 2


def test_tile_sweep_has_an_explicit_independent_k4_surface() -> None:
    sweep = (
        ROOT / "python" / "tools" / "bench_b12x_spark_exl3_tiles.py"
    ).read_text(encoding="utf-8")

    assert '"--trellis-bits"' in sweep
    assert "choices=(3, 4)" in sweep
    assert "trellis_bits=args.trellis_bits" in sweep
    assert '"trellis_bits": args.trellis_bits' in sweep
    assert "capacity_rows = _capacity_rows(rows, trellis_bits)" in sweep
    assert "max_tokens=capacity_rows" in sweep
    assert "launch_token_count = rows if capacity_rows <= 32 else capacity_rows" in sweep
    assert '"--projection-checkpoint-dir"' in sweep
    assert '"--model-snapshot"' in sweep
    assert 'row_spec == "all-aot"' in sweep
    assert 'row_spec == "required-native"' in sweep
    assert "EXL3_K4_REQUIRED_LIVE_ROWS" in sweep
    assert "_load_checkpoint_weight_tensors(" in sweep
    assert "_load_artifact_weight_tensors(" in sweep
    assert '"kind": "authenticated-calibrated-checkpoints"' in sweep
    assert 'trellis_codebook="mcg"' in sweep
    assert "BITS = 3" not in sweep


def test_native_validator_can_read_the_final_artifact_without_quant_checkpoints() -> None:
    validator = (TOOLS / "validate_b12x_exl3_native.py").read_text(
        encoding="utf-8"
    )

    assert '"--model-snapshot"' in validator
    assert "def _load_artifact_weight_tensors(" in validator
    assert '"kind": "finalized-exl3-artifact"' in validator
    assert 'trellis_codebook="mcg"' in validator
    assert 'ARTIFACT_MANIFEST_SCHEMA = "glmrt-glm5-gptqmodel-artifact-v2"' in validator


def test_checkpoint_k4_and_retained_k3_native_evidence_is_exact() -> None:
    reports = [
        ("glm53-exl3-k4-layer3-rank0-all-aot.json", 4, 15),
        ("glm52-exl3-k3-corrected-native-regression.json", 3, 4),
    ]
    for name, trellis_bits, expected_cases in reports:
        evidence = json.loads(
            (ROOT / "scripts" / "tests" / "fixtures" / name).read_text(
                encoding="utf-8"
            )
        )
        assert evidence["schema"] == "glmrt-b12x-exl3-native-validation-v1"
        assert evidence["status"] == "accepted"
        assert evidence["trellis_bits"] == trellis_bits
        assert len(evidence["cases"]) == expected_cases
        assert all(case["reference"] == "sparkinfer" for case in evidence["cases"])
        assert all(case["relative_l2"] == 0.0 for case in evidence["cases"])
        assert all(case["max_abs"] == 0.0 for case in evidence["cases"])
