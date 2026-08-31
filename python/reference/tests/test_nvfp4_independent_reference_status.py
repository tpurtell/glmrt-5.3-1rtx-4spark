from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from glmrt_reference.nvfp4_independent_status import build_independent_reference_status
from glmrt_reference.quant_ref import NVFP4_E2M1_VALUES


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


_NVFP4_FIXTURES_AVAILABLE = (
    _repo_root() / "tests/fixtures/nvfp4/real_tensor_decode.json"
).is_file() and (_repo_root() / "tests/fixtures/nvfp4/modelopt_reference.json").is_file()


@pytest.mark.skipif(
    not _NVFP4_FIXTURES_AVAILABLE,
    reason="private NVFP4 oracle fixtures are not part of the public release",
)
def test_nvfp4_independent_reference_status_keeps_recipe_provisional_until_oracle_runs():
    status = build_independent_reference_status(_repo_root())

    assert status["format_version"] == 1
    assert status["checked_fixture"]["source"] == "python-reference-raw-safetensors"
    assert status["checked_fixture"]["tensor"] == "model.layers.3.mlp.experts.0.gate_proj.weight"
    assert status["checked_fixture"]["projection"] == "gate_proj"
    assert status["checked_fixture"]["value_count"] == 64

    if status["independent_real_reference_comparison"]:
        assert status["status"] == "independent_modelopt_reference_comparison_passed"
        assert status["comparison_executed"] is True
        assert status["phase0_summary_claims_independent_comparison"] is True
        assert status["phase0_summary_consistent"] is True
        assert status["verification_artifact"]["independent_reference"].endswith("NVFP4QTensor")
    elif status["missing_oracle_modules"]:
        assert status["comparison_executed"] is False
        assert status["phase0_summary_claims_independent_comparison"] is False
        assert status["phase0_summary_consistent"] is True
        assert status["status"] == "blocked_missing_independent_oracle_dependency"
        assert "torch" in status["required_oracle_modules"]
        assert "modelopt" in status["required_oracle_modules"]
    else:
        assert status["comparison_executed"] is False
        assert status["phase0_summary_claims_independent_comparison"] is False
        assert status["phase0_summary_consistent"] is True
        assert status["status"] == "blocked_independent_oracle_adapter_not_implemented"


@pytest.mark.skipif(
    not _NVFP4_FIXTURES_AVAILABLE,
    reason="private NVFP4 oracle fixtures are not part of the public release",
)
def test_nvfp4_modelopt_artifact_matches_current_fixture_and_codebook():
    root = _repo_root()
    fixture_path = root / "tests/fixtures/nvfp4/real_tensor_decode.json"
    artifact_path = root / "tests/fixtures/nvfp4/modelopt_reference.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    assert artifact["schema"] == "glmrt.phase0.nvfp4_modelopt_reference.v1"
    assert artifact["comparison_executed"] is True
    assert artifact["comparison_passed"] is True
    assert artifact["fixture_path"] == "tests/fixtures/nvfp4/real_tensor_decode.json"
    assert artifact["fixture_sha256"] == hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    assert artifact["device"].startswith("cuda")
    assert artifact["cuda_available"] is True
    assert artifact["cuda_device_name"]
    assert artifact["cuda_device_capability"][0] >= 12
    assert artifact["modelopt_e2m1_codebook"] == list(NVFP4_E2M1_VALUES)
    assert artifact["window"]["max_abs_diff"] <= artifact["tolerance_abs"]
    assert artifact["full_row"]["checksum_abs_diff"] <= artifact["tolerance_abs"]
