from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "python/tools/calibrate_dspark_cost_profile.py"


def test_calibrator_replaces_only_well_sampled_short_context_cells(tmp_path: Path) -> None:
    startup = tmp_path / "startup.log"
    corpus = tmp_path / "corpus.log"
    output_json = tmp_path / "profile.json"
    output_rust = tmp_path / "profile.rs"
    route_reference = tmp_path / "route-reference.json"
    replay_plan = tmp_path / "replay-plan.jsonl"
    replay_result = tmp_path / "replay-result.jsonl"
    startup.write_text(
        "\n".join(
            f"real_full_dspark_sps_profile requests={requests} target_rows={rows} "
            f"latency_ms={10 * rows:.3f} samples=4 source=startup-opt-in"
            for requests, rows in [(1, 1), (1, 2), (2, 2), (2, 3), (2, 4)]
        )
        + "\nreal_full_dspark_sps_profile requests=1 target_rows=1 "
        "latency_ms=999.000 samples=32 source=startup-opt-in\n"
    )
    corpus.write_text(
        "\n".join(
            [
                "real_full_dspark_runtime_cost requests=1 context_work_bucket=0 "
                "max_context_bucket=0 target_rows=2 observed_ms=25.000 "
                f"predicted_ms_before=20.000 exact_samples={sample} "
                "route_wire_batches=75 route_assignments=1200 "
                "route_unique_experts=10 route_critical_unique_experts=10 "
                "route_reused_assignments=1190 route_max_expert_load=3 "
                "route_load_square_sum=2400"
                for sample in range(1, 6)
            ]
            + [
                "real_full_dspark_runtime_cost requests=2 context_work_bucket=0 "
                "max_context_bucket=0 target_rows=2 observed_ms=99.000 "
                f"predicted_ms_before=20.000 exact_samples={sample}"
                for sample in range(1, 5)
            ]
            + [
                "real_full_dspark_runtime_cost requests=1 context_work_bucket=1 "
                "max_context_bucket=1 target_rows=1 observed_ms=999.000 "
                "predicted_ms_before=10.000 exact_samples=1"
            ]
        )
        + "\n"
    )
    route_reference.write_text(
        json.dumps(
            {
                "schema": "glmrt-dspark-route-reference-v1",
                "source_sha256": "route-reference-hash",
                "max_concurrency": 2,
                "max_drafts": 1,
                "cells": {
                    f"{requests}:{rows}": {
                        "requests": requests,
                        "target_rows": rows,
                        "route_shape": {
                            "critical_unique_experts": {
                                "mean": 100.0
                                if (requests, rows) == (2, 3)
                                else 12.0
                            }
                        },
                    }
                    for requests, rows in [
                        (1, 1),
                        (1, 2),
                        (2, 2),
                        (2, 3),
                        (2, 4),
                    ]
                },
            }
        )
        + "\n"
    )
    replay_plan.write_text(
        "".join(
            json.dumps(
                {
                    "record": "chain",
                    "chain_id": f"chain-{index}",
                    "physical_m": 2,
                    "layers": [
                        {
                            "layer_id": 3,
                            "routes": [list(range(10 + index % 2))],
                        }
                    ],
                }
            )
            + "\n"
            for index in range(32)
        )
    )
    replay_result.write_text(
        "".join(
            json.dumps(
                {
                    "record": "measurement",
                    "chain_id": f"chain-{index}",
                    "physical_m": 2,
                    "path": "coordinator",
                    "dispatch_ms": 10.0 + 0.5 * (10 + index % 2),
                }
            )
            + "\n"
            for index in range(32)
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--startup-log",
            str(startup),
            "--corpus-log",
            f"1={corpus}",
            "--output-json",
            str(output_json),
            "--output-rust",
            str(output_rust),
            "--profile-id",
            "test-profile",
            "--target-model",
            "target/model",
            "--target-revision",
            "target-revision",
            "--dspark-model",
            "draft/model",
            "--dspark-revision",
            "draft-revision",
            "--sparkinfer-revision",
            "spark-revision",
            "--engine-commit",
            "engine-commit",
            "--topology",
            "test-topology",
            "--power-limit-watts",
            "400",
            "--max-concurrency",
            "2",
            "--max-drafts",
            "1",
            "--minimum-corpus-samples",
            "5",
            "--route-reference",
            str(route_reference),
            "--route-replay-plan",
            str(replay_plan),
            "--route-replay-result",
            str(replay_result),
            "--startup-samples",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    profile = json.loads(output_json.read_text())
    assert report["corpus_qualified_cells"] == 1
    assert report["ignored_nonzero_context_samples"] == 1
    assert profile["curves"]["1"][1]["latency_ms"] == 26.0
    assert (
        profile["curves"]["1"][1]["source"]
        == "route-standardized-corpus-mean"
    )
    assert profile["curves"]["2"][0]["latency_ms"] == 20.0
    assert profile["curves"]["2"][0]["source"] == "startup-sweep-fallback"
    assert profile["curves"]["2"][1]["latency_ms"] == 64.0
    assert profile["curves"]["2"][1]["route_work_increment_ms"] == 44.0
    assert profile["curves"]["1"][0]["startup"]["samples_per_sweep"] == [4]
    assert profile["qualification"]["corpus_logs"] == [
        {
            "path": str(corpus),
            "request_count": 1,
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        }
    ]
    assert "GLM52_REDHAT_DSPARK_COST_PROFILE_MS" in output_rust.read_text()


def test_calibrator_can_performance_gate_a_concurrency_to_baseline(
    tmp_path: Path,
) -> None:
    startup = tmp_path / "startup.log"
    corpus = tmp_path / "corpus.log"
    baseline = tmp_path / "baseline.json"
    output = tmp_path / "output.json"
    startup.write_text(
        "real_full_dspark_sps_profile requests=1 target_rows=1 "
        "latency_ms=10.000 samples=4 source=startup-opt-in\n"
        "real_full_dspark_sps_profile requests=1 target_rows=2 "
        "latency_ms=20.000 samples=4 source=startup-opt-in\n"
    )
    corpus.write_text("")
    baseline.write_text(
        json.dumps(
            {
                "profile_id": "qualified-control",
                "source_sha256": "control-hash",
                "identity": {
                    "target_model": "target/model",
                    "target_revision": "target-revision",
                    "dspark_model": "draft/model",
                    "dspark_revision": "draft-revision",
                    "sparkinfer_revision": "spark-revision",
                    "topology": "test-topology",
                    "power_limit_watts": 400,
                    "max_concurrency": 1,
                    "max_drafts": 1,
                },
                "curves": {
                    "1": [
                        {
                            "target_rows": 1,
                            "latency_ms": 11.0,
                            "source": "old-one",
                        },
                        {
                            "target_rows": 2,
                            "latency_ms": 22.0,
                            "source": "old-two",
                        },
                    ]
                },
            }
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--startup-log",
            str(startup),
            "--corpus-log",
            str(corpus),
            "--output-json",
            str(output),
            "--profile-id",
            "hybrid",
            "--target-model",
            "target/model",
            "--target-revision",
            "target-revision",
            "--dspark-model",
            "draft/model",
            "--dspark-revision",
            "draft-revision",
            "--sparkinfer-revision",
            "spark-revision",
            "--engine-commit",
            "engine-commit",
            "--topology",
            "test-topology",
            "--power-limit-watts",
            "400",
            "--max-concurrency",
            "1",
            "--max-drafts",
            "1",
            "--startup-samples",
            "4",
            "--baseline-profile",
            str(baseline),
            "--retain-baseline-concurrency",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    profile = json.loads(output.read_text())
    assert [cell["latency_ms"] for cell in profile["curves"]["1"]] == [
        11.0,
        22.0,
    ]
    assert profile["curves"]["1"][0]["candidate_latency_ms"] == 10.0
    assert profile["curves"]["1"][0]["source"] == "performance-gated-baseline"
    assert profile["qualification"]["retained_baseline_concurrencies"] == [1]


def test_calibrator_can_render_a_separate_dflash_profile_prefix(
    tmp_path: Path,
) -> None:
    startup = tmp_path / "startup.log"
    corpus = tmp_path / "corpus.log"
    output_json = tmp_path / "profile.json"
    output_rust = tmp_path / "profile.rs"
    startup.write_text(
        "real_full_dspark_sps_profile requests=1 target_rows=1 "
        "latency_ms=10.000 samples=4 source=startup-opt-in\n"
        "real_full_dspark_sps_profile requests=1 target_rows=2 "
        "latency_ms=20.000 samples=4 source=startup-opt-in\n"
    )
    corpus.write_text("")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--startup-log",
            str(startup),
            "--corpus-log",
            str(corpus),
            "--output-json",
            str(output_json),
            "--output-rust",
            str(output_rust),
            "--rust-constant-prefix",
            "GLM53_EXL3_K4_DFLASH2_COST_PROFILE",
            "--profile-id",
            "glm53-dflash-test",
            "--target-model",
            "target/model",
            "--target-revision",
            "target-revision",
            "--dspark-model",
            "draft/model",
            "--dspark-revision",
            "draft-revision",
            "--sparkinfer-revision",
            "spark-revision",
            "--engine-commit",
            "engine-commit",
            "--topology",
            "test-topology",
            "--power-limit-watts",
            "400",
            "--max-concurrency",
            "1",
            "--max-drafts",
            "1",
            "--startup-samples",
            "4",
        ],
        check=True,
    )
    rendered = output_rust.read_text()
    assert "GLM53_EXL3_K4_DFLASH2_COST_PROFILE_ID" in rendered
    assert "GLM53_EXL3_K4_DFLASH2_COST_PROFILE_MS" in rendered
    assert "GLM52_REDHAT_DSPARK_COST_PROFILE" not in rendered
