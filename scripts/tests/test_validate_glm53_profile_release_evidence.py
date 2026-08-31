from __future__ import annotations

import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = (
    ROOT / "python" / "tools" / "validate_glm53_profile_release_evidence.py"
)
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_profile_release", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_benchmark_metadata_requires_post_deployment_profile_bound_rows(
    tmp_path: Path,
) -> None:
    started = 2_000_000_000
    completed = 4_000_000_000
    stamp = dt.datetime.fromtimestamp(3, tz=dt.UTC).isoformat()
    path = tmp_path / "blended.jsonl"
    records = [
        {
            "run_id": "paired-v1",
            "profile": "long",
            "timestamp_utc": stamp,
        },
        {
            "aggregate": {
                "run_id": "paired-v1",
                "profile": "long",
                "benchmark_started_ns": started,
                "benchmark_completed_ns": completed,
                "timestamp_utc": stamp,
            }
        },
    ]
    write_jsonl(path, records)

    assert TOOL._benchmark_metadata(
        path,
        kind="blended",
        profile="long",
        launch_started_ns=1_000_000_000,
    )["run_id"] == "paired-v1"
    with pytest.raises(TOOL.ProfileReleaseError, match="post-deployment"):
        TOOL._benchmark_metadata(
            path,
            kind="blended",
            profile="long",
            launch_started_ns=started,
        )
    records[0]["profile"] = "accuracy"
    write_jsonl(tmp_path / "wrong.jsonl", records)
    with pytest.raises(TOOL.ProfileReleaseError, match="outside"):
        TOOL._benchmark_metadata(
            tmp_path / "wrong.jsonl",
            kind="blended",
            profile="long",
            launch_started_ns=1_000_000_000,
        )


def arm_specs(tmp_path: Path) -> list[tuple[str, str, Path, Path, Path]]:
    return [
        (
            profile,
            mode,
            tmp_path / f"{profile}-{mode}-deployment.json",
            tmp_path / f"{profile}-{mode}-blended.jsonl",
            tmp_path / f"{profile}-{mode}-prefill.jsonl",
        )
        for profile in TOOL.PROFILES
        for mode in TOOL.MODES
    ]


def fake_arm(profile: str, mode: str) -> dict[str, object]:
    dflash_settings = {
        "checkpoint_model_id": "incoai/GLM-5.3-DFlash2",
        "checkpoint_revision": "a" * 40,
        "fixed_drafts": 4,
        "topk_backend": "flashinfer",
    }
    identity = lambda kind: {
        "schema": kind,
        "path": f"/{profile}-{mode}-{kind}",
        "bytes": 1,
        "sha256": hashlib_value(profile + mode + kind),
    }
    return {
        "deployment": {
            "identity": identity("deployment"),
            "model_revision": "1" * 40,
            "slot": "final",
            "profile": profile,
            "speculation": mode,
            "speculation_settings": dflash_settings if mode == "dflash2" else {},
            "launch_started_ns": 10 if profile == "balanced" else 20,
            "power_limit_w": 400,
            "engine_identity": "engine",
            "sparkinfer_revision": "2" * 40,
            "fingerprints": {
                "coordinator_slot": "3" * 64,
                "expert_slot": "4" * 64,
                "expert_runtime": hashlib_value(profile + mode),
                "deployment": "5" * 64,
            },
        },
        "blended": {
            "identity": identity("blended"),
            "contract": "6" * 64,
            "prompt_contract": {"nonce_seed": 53},
            "prompts": [{"prompt_sha256": "7" * 64}],
            "wall_decode_tps": 30.0,
            "median_repeat_wall_decode_tps": 28.5,
            "accepted_draft_rate": 0.75,
        },
        "prefill": {
            "identity": identity("prefill"),
            "contract": "8" * 64,
            "corpus_sha256": "9" * 64,
            "tokenizer_sha256": "a" * 64,
            "prompts": [{"prompt_sha256": "b" * 64}],
            "cells": {TOOL.REQUIRED_PREFILL_CELL: 1_000.0},
        },
        "blended_run": {"run_id": "paired-v1"},
        "prefill_run": {"run_id": "paired-prefill-v1"},
        "verify_tps": 60.0,
        "verify_curve": {"1": {"samples": 1}},
    }


def hashlib_value(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


def fake_serving(arms: dict[tuple[str, str], dict[str, object]]) -> dict[str, object]:
    dflash = arms[("balanced", "dflash2")]
    return {
        "schema": "glmrt-glm5-exl3-serving-qualification-v1",
        "runtime": {
            "model_revision": "1" * 40,
            "power_limit_w": 400,
            "engine_identity": "engine",
            "sparkinfer_revision": "2" * 40,
            "coordinator_slot_fingerprint": "3" * 64,
            "expert_slot_fingerprint": "4" * 64,
            "speculation_settings": {
                "dflash2": dflash["deployment"]["speculation_settings"],
            },
            "launch_started_ns": {"dflash2": 10},
            "expert_runtime_fingerprints": {
                "dflash2": dflash["deployment"]["fingerprints"]["expert_runtime"],
            },
        },
        "evidence": {
            "dflash2_deployment": dflash["deployment"]["identity"],
            "dflash2_blended": dflash["blended"]["identity"],
        },
    }


def test_three_arm_report_binds_runtime_and_recomputes_retention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arms = {
        (profile, mode): fake_arm(profile, mode)
        for profile in TOOL.PROFILES
        for mode in TOOL.MODES
    }
    serving = fake_serving(arms)
    monkeypatch.setattr(TOOL, "signed_serving", lambda _path: (tmp_path / "serving", serving))
    monkeypatch.setattr(
        TOOL,
        "_validate_arm",
        lambda **kwargs: arms[(kwargs["profile"], kwargs["mode"])],
    )
    monkeypatch.setattr(
        TOOL,
        "evidence_identity",
        lambda _path, schema: {
            "schema": schema,
            "path": "/serving",
            "bytes": 1,
            "sha256": "c" * 64,
        },
    )

    report = TOOL.validate(
        serving_path=tmp_path / "serving.json",
        arm_specs=arm_specs(tmp_path),
    )
    assert report["status"] == "accepted"
    assert report["results"]["balanced"]["dflash2"][
        "weighted_decode_tps"
    ] == 30.0
    assert report["profile_retention"]["dflash2"]["long"][
        "weighted_decode_tps"
    ] == 1.0
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    assert report["report_sha256"] == hashlib_value(
        TOOL.canonical_json(body).decode()
    )

    arms[("long", "dflash2")]["blended"]["prompts"] = [
        {"prompt_sha256": "d" * 64}
    ]
    with pytest.raises(TOOL.ProfileReleaseError, match="same prompts"):
        TOOL.validate(
            serving_path=tmp_path / "serving.json",
            arm_specs=arm_specs(tmp_path),
        )


def test_three_arm_schedule_rejects_missing_or_reused_inputs(tmp_path: Path) -> None:
    specs = arm_specs(tmp_path)
    with pytest.raises(TOOL.ProfileReleaseError, match="exactly"):
        TOOL.validate(serving_path=tmp_path / "serving", arm_specs=specs[:-1])
    reused = list(specs)
    profile, mode, deployed, blended, _ = reused[-1]
    reused[-1] = (profile, mode, deployed, blended, specs[0][-1])
    with pytest.raises(TOOL.ProfileReleaseError, match="distinct"):
        TOOL.validate(serving_path=tmp_path / "serving", arm_specs=reused)
