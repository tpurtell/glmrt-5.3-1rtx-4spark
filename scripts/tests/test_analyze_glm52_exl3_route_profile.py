from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "python" / "tools" / "analyze_glm52_exl3_route_profile.py"


def _trace(
    request_id: int,
    layer_id: int,
    host_index: int,
    *,
    rows: int = 9,
    route_counts: str = "0:9,1:9,2:9,3:9,4:9,5:9,6:9,7:9",
) -> str:
    return (
        "INFO protocol_v2_expert_queue_plan "
        "trace_schema=2 capture_id=qualification-v1 "
        f"transport=verbs-host request_id_base={request_id} "
        f"request_id={request_id + host_index} layer_id={layer_id} "
        f"host=spark{host_index} host_index={host_index} rows={rows} "
        f"routes={rows * 8} source_rows=DecodeStep:{rows} experts=8 "
        "single_expert_rows=0 multi_expert_rows=9 empty_rows=0 "
        "expert_only_rows=27 expert_only_extra_rows=18 expert_only_row_factor=3.0 "
        "expert_row_min=9 expert_row_p50=9 expert_row_max=9 "
        "expert_route_min=9 expert_route_p50=9 expert_route_max=9 "
        "least_hot_expert=0 least_hot_rows=9 hottest_expert=7 hottest_rows=9 "
        f"expert_route_counts={route_counts}\n"
    )


def _run(
    tmp_path: Path,
    log_text: str,
    *extra: str,
    model_id: str = "test/exl3",
) -> subprocess.CompletedProcess[str]:
    log = tmp_path / "coordinator.log"
    deployment = tmp_path / "deployment.json"
    output = tmp_path / "profile.json"
    log.write_text(log_text, encoding="utf-8")
    deployment.write_text(
        json.dumps({"schema": "test-deployment", "model_id": model_id}),
        encoding="utf-8",
    )
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--log",
            f"balanced={log}",
            "--deployment",
            str(deployment),
            "--capture-id",
            "qualification-v1",
            "--output",
            str(output),
            "--expected-layer-first",
            "3",
            "--expected-layer-last",
            "4",
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_route_profile_binds_complete_tp4_layer_corpus(tmp_path: Path) -> None:
    log_text = "noise\n" + "".join(
        _trace(1000 + layer_id * 10, layer_id, host_index)
        for layer_id in (3, 4)
        for host_index in range(4)
    )
    result = _run(tmp_path, log_text)
    assert result.returncode == 0, result.stderr

    report = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert report["status"] == "accepted"
    assert report["summary"]["trace_records"] == 8
    assert report["summary"]["tp4_samples"] == 2
    assert report["summary"]["layer_ids"] == [3, 4]
    assert report["summary"]["exact_m9_samples"] == 2
    assert len(report["samples"]) == 2
    assert report["summary"]["samples_by_active_experts"] == [[8, 2]]
    assert report["summary"]["samples_by_maximum_expert_reuse_rows"] == [[9, 2]]
    assert report["samples"][0]["route_shape"] == {
        "active_experts": 8,
        "minimum_expert_reuse_rows": 9,
        "maximum_expert_reuse_rows": 9,
        "reuse_rows_sum": 72,
        "padded_route_slots_by_block_rows": [
            [8, 128],
            [16, 128],
            [32, 256],
            [48, 384],
            [64, 512],
        ],
    }
    digest = report.pop("report_sha256")
    canonical = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    assert digest == hashlib.sha256(canonical).hexdigest()


def test_route_profile_rejects_tp_rank_route_mismatch(tmp_path: Path) -> None:
    log_text = "".join(
        _trace(
            1030,
            3,
            host_index,
            route_counts="0:8,1:9,2:9,3:9,4:9,5:9,6:9,7:9,8:1"
            if host_index == 2
            else "0:9,1:9,2:9,3:9,4:9,5:9,6:9,7:9",
        )
        for host_index in range(4)
    )
    result = _run(
        tmp_path,
        log_text,
        "--expected-layer-last",
        "3",
    )
    assert result.returncode != 0
    assert "TP route replication mismatch" in result.stderr


def test_route_profile_rejects_obsolete_trace_without_layer_id(tmp_path: Path) -> None:
    line = _trace(1030, 3, 0).replace("layer_id=3 ", "")
    result = _run(
        tmp_path,
        line,
        "--expected-hosts",
        "1",
        "--expected-layer-last",
        "3",
    )
    assert result.returncode != 0
    assert "missing layer_id" in result.stderr


def test_route_profile_rejects_duplicate_expert_within_a_row(tmp_path: Path) -> None:
    line = _trace(1030, 3, 0, route_counts="0:10,1:9,2:9,3:9,4:9,5:9,6:9,7:8")
    result = _run(
        tmp_path,
        line,
        "--expected-hosts",
        "1",
        "--expected-layer-last",
        "3",
    )
    assert result.returncode != 0
    assert "routed more than once per row" in result.stderr


def test_glm53_k4_route_profile_is_model_and_bitrate_bound(tmp_path: Path) -> None:
    log_text = "".join(
        _trace(1030, 3, host_index) for host_index in range(4)
    )
    result = _run(
        tmp_path,
        log_text,
        "--expected-layer-last",
        "3",
        "--trellis-bits",
        "4",
        model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((tmp_path / "profile.json").read_text(encoding="utf-8"))
    assert report["schema"] == "glmrt-glm5-exl3-route-profile-v1"
    assert report["geometry"]["trellis_bits"] == 4

    wrong = _run(
        tmp_path,
        log_text,
        "--expected-layer-last",
        "3",
        "--trellis-bits",
        "4",
    )
    assert wrong.returncode != 0
    assert "K4 route profile requires deployment model_id" in wrong.stderr
