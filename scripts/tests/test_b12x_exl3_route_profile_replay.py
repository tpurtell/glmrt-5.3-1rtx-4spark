from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "python" / "tools"
SPARKINFER = ROOT / "third_party" / "sparkinfer"
sys.path.insert(0, str(TOOLS))
sys.path.insert(0, str(SPARKINFER))
MODULE_PATH = TOOLS / "validate_b12x_exl3_native.py"
SPEC = importlib.util.spec_from_file_location("_b12x_exl3_route_replay", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


def _profile(path: Path) -> Path:
    body = {
        "schema": TOOL.ROUTE_PROFILE_SCHEMA,
        "status": "accepted",
        "capture_id": "qualification-v1",
        "samples": [
            {
                "rows": 9,
                "layer_id": 3,
                "expert_route_counts": [[expert_id, 9] for expert_id in range(8)],
            }
        ],
    }
    body["report_sha256"] = hashlib.sha256(TOOL._canonical_json(body)).hexdigest()
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_native_replay_authenticates_and_expands_sparse_route_counts(
    tmp_path: Path,
) -> None:
    path = _profile(tmp_path / "route-profile.json")
    counts, identity = TOOL._load_route_profile_sample(path, 0, 9)

    assert counts == [9] * 8 + [0] * 248
    assert TOOL._validate_route_counts(counts, 9) == counts
    assert identity["profile_report_sha256"] == json.loads(path.read_text())[
        "report_sha256"
    ]
    assert identity["sample_index"] == 0


def test_native_replay_rejects_tampered_route_profile(tmp_path: Path) -> None:
    path = _profile(tmp_path / "route-profile.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    report["samples"][0]["layer_id"] = 4
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="report_sha256"):
        TOOL._load_route_profile_sample(path, 0, 9)


def test_native_replay_rejects_a_route_profile_from_another_bitrate(
    tmp_path: Path,
) -> None:
    body = {
        "schema": TOOL.GLM5_ROUTE_PROFILE_SCHEMA,
        "status": "accepted",
        "capture_id": "glm53-k4-v1",
        "geometry": {"trellis_bits": 4},
        "samples": [
            {
                "rows": 9,
                "layer_id": 3,
                "expert_route_counts": [[expert_id, 9] for expert_id in range(8)],
            }
        ],
    }
    body["report_sha256"] = hashlib.sha256(TOOL._canonical_json(body)).hexdigest()
    path = tmp_path / "glm53-route-profile.json"
    path.write_text(json.dumps(body), encoding="utf-8")

    counts, _ = TOOL._load_route_profile_sample(path, 0, 9, 4)
    assert counts == [9] * 8 + [0] * 248
    with pytest.raises(ValueError, match="bitrate"):
        TOOL._load_route_profile_sample(path, 0, 9, 3)


@pytest.mark.parametrize("rows", [2048, 2064])
def test_checked_in_high_reuse_tail_fixture_is_a_legal_topk8_plan(rows: int) -> None:
    path = (
        ROOT
        / "scripts"
        / "tests"
        / "fixtures"
        / f"glm52-exl3-route-counts-m{rows}-high-reuse.json"
    )
    counts = json.loads(path.read_text(encoding="utf-8"))

    assert counts == [rows] * 8 + [0] * 248
    assert TOOL._validate_route_counts(counts, rows) == counts
