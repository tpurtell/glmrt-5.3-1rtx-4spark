from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "build_exl3_projection_catalog.py"
SPEC = importlib.util.spec_from_file_location("_build_exl3_catalog", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


@pytest.mark.parametrize(
    ("projection", "expected_k3", "expected_k4"),
    [
        ("gate_proj", [384, 128, 48], [384, 128, 64]),
        ("up_proj", [384, 128, 48], [384, 128, 64]),
        ("down_proj", [128, 384, 48], [128, 384, 64]),
    ],
)
def test_projection_catalog_shapes_bind_the_physical_bitrate(
    projection: str, expected_k3: list[int], expected_k4: list[int]
) -> None:
    assert TOOL._expected_tensor_shapes(projection, 3)["trellis"] == expected_k3
    assert TOOL._expected_tensor_shapes(projection, 4)["trellis"] == expected_k4


def test_model_profiles_retain_k3_and_select_exact_glm53_k4_identity() -> None:
    assert TOOL.MODEL_PROFILES[TOOL.MODEL_ID] == {
        "recipe": TOOL.RECIPE,
        "trellis_bits": 3,
    }
    assert TOOL.MODEL_PROFILES[TOOL.GLM53_MODEL_ID] == {
        "recipe": TOOL.GLM53_RECIPE,
        "trellis_bits": 4,
    }


def test_projection_catalog_rejects_unknown_bitrate() -> None:
    with pytest.raises(ValueError, match="unsupported EXL3"):
        TOOL._expected_tensor_shapes("gate_proj", 5)
