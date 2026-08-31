from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "validate_glm53_oss_release_evidence.py"
sys.path.insert(0, str(TOOL_PATH.parent))
SPEC = importlib.util.spec_from_file_location("_glmrt_oss_release", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def context_evidence(path: Path) -> list[dict[str, object]]:
    stamp = dt.datetime.fromtimestamp(3, tz=dt.UTC).isoformat()
    records: list[dict[str, object]] = []
    prompts: list[dict[str, object]] = []
    for index, (context, workload, repeat) in enumerate(
        (
            (context, workload, repeat)
            for context in TOOL.DEFAULT_CONTEXTS
            for workload in TOOL.WORKLOADS
            for repeat in range(1, TOOL.REQUIRED_CONTEXT_REPEATS + 1)
        )
    ):
        content = f"output-{index}"
        prompt_sha256 = sha(f"prompt-{index}")
        records.append(
            {
                "schema": TOOL.CONTEXT_SCHEMA,
                "run_id": "glm53-context-v1",
                "timestamp_utc": stamp,
                "profile": "balanced",
                "model": TOOL.GLM53_MODEL_ID,
                "context_bucket_tokens": context,
                "workload": workload,
                "repeat": repeat,
                "prompt_tokens": context + 10,
                "cached_prompt_tokens": context,
                "prefill_rows": 9,
                "output_tokens": 101,
                "decode_ms": 2_000.0,
                "decode_tps": 50.0,
                "draft_tokens": 100,
                "accepted_draft_tokens": 75,
                "runtime_captures": 0,
                "numeric_progression_passed": True,
                "attention_complete": True,
                "marker": chr(0x400 + index),
                "prompt_sha256": prompt_sha256,
                "content": content,
                "content_sha256": sha(content),
                "reasoning_chars": 0,
                "finish_reason": "length",
                "corpus_root": str(path.parent),
                "corpus_sha256": "a" * 64,
                "tokenizer": str(path.parent / "tokenizer.json"),
                "tokenizer_sha256": "b" * 64,
            }
        )
        prompts.append(
            {
                "context_bucket_tokens": context,
                "workload": workload,
                "repeat": repeat,
                "prompt_sha256": prompt_sha256,
            }
        )
    summary = {
        "schema": TOOL.CONTEXT_SUMMARY_SCHEMA,
        "benchmark_started_ns": 2_000_000_000,
        "benchmark_completed_ns": 4_000_000_000,
        "timestamp_utc": stamp,
        "run_id": "glm53-context-v1",
        "profile": "balanced",
        "model": TOOL.GLM53_MODEL_ID,
        "contexts": list(TOOL.DEFAULT_CONTEXTS),
        "workloads": list(TOOL.WORKLOADS),
        "repeats": TOOL.REQUIRED_CONTEXT_REPEATS,
        "max_tokens": TOOL.REQUIRED_CONTEXT_MAX_TOKENS,
        "corpus_root": str(path.parent),
        "corpus_sha256": "a" * 64,
        "tokenizer": str(path.parent / "tokenizer.json"),
        "tokenizer_sha256": "b" * 64,
        "prompt_contract_sha256": TOOL.canonical_sha256(prompts),
        "cells": TOOL.summarize_records(
            records,
            contexts=list(TOOL.DEFAULT_CONTEXTS),
            workloads=list(TOOL.WORKLOADS),
            repeats=TOOL.REQUIRED_CONTEXT_REPEATS,
        ),
    }
    return records + [summary]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_context_decode_requires_complete_bound_post_deployment_matrix(
    tmp_path: Path,
) -> None:
    path = tmp_path / "context.jsonl"
    records = context_evidence(path)
    write_jsonl(path, records)

    result = TOOL.validate_context_decode(
        path, deployed={"launch_started_ns": 1_000_000_000}
    )
    assert len(result["cells"]) == 15
    assert result["run"]["run_id"] == "glm53-context-v1"

    records[0]["prompt_sha256"] = "not-a-hash"
    write_jsonl(tmp_path / "broken.jsonl", records)
    with pytest.raises(TOOL.OssReleaseError, match="unbound"):
        TOOL.validate_context_decode(
            tmp_path / "broken.jsonl",
            deployed={"launch_started_ns": 1_000_000_000},
        )


def identity(path: Path, schema: str) -> dict[str, object]:
    return {
        "schema": schema,
        "path": str(path),
        "bytes": 1,
        "sha256": sha(str(path)),
    }


def test_final_gate_binds_all_reports_to_selected_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = "dflash2"
    settings = {"fixed_drafts": 4, "topk_backend": "flashinfer"}
    runtime = {
        "model_revision": "1" * 40,
        "profile": "balanced",
        "power_limit_w": 400,
        "engine_identity": "engine-v1",
        "sparkinfer_revision": "2" * 40,
        "coordinator_slot_fingerprint": "3" * 64,
        "expert_slot_fingerprint": "4" * 64,
        "expert_runtime_fingerprints": {"mtp": "5" * 64, selected: "6" * 64},
        "speculation_settings": {"mtp": {}, selected: settings},
    }
    paths = {
        name: tmp_path / f"{name}.json"
        for name in ("serving", "agentic", "profiles", "startup", "micro")
    }
    serving_identity = identity(paths["serving"], "serving-v1")
    deployment_identity = identity(tmp_path / "deployment.json", "deployment-v1")
    serving = {
        "schema": "serving-v1",
        "runtime": runtime,
        "results": {"default_speculation": selected, "modes": {}},
        "evidence": {f"{selected}_deployment": deployment_identity},
    }
    agentic_runtime = {
        "model_revision": runtime["model_revision"],
        "profile": "balanced",
        "power_limit_w": 400,
        "engine_identity": "engine-v1",
        "sparkinfer_revision": "2" * 40,
        "coordinator_slot": "3" * 64,
        "expert_slot": "4" * 64,
        "expert_runtime": "6" * 64,
        "speculation_settings": settings,
        "launch_started_ns": 10,
        "slot": "final",
    }
    reports = {
        TOOL.AGENTIC_SCHEMA: {
            "model_id": TOOL.GLM53_MODEL_ID,
            "model_revision": runtime["model_revision"],
            "default_speculation": selected,
            "runtime": agentic_runtime,
            "tool_eval": {},
            "pi": {},
            "evidence": {
                "serving_qualification": serving_identity,
                "deployment": deployment_identity,
            },
        },
        TOOL.PROFILE_SCHEMA: {
            "model_id": TOOL.GLM53_MODEL_ID,
            "model_revision": runtime["model_revision"],
            "runtime": {
                "model_revision": runtime["model_revision"],
                "slot": "final",
                "power_limit_w": 400,
                "engine_identity": "engine-v1",
                "sparkinfer_revision": "2" * 40,
                "coordinator_slot": "3" * 64,
                "expert_slot": "4" * 64,
            },
                "speculation_settings": {selected: settings},
            "results": {},
            "profile_retention": {},
            "evidence": {"serving": serving_identity},
        },
        TOOL.STARTUP_TIMELINE_SCHEMA: {
            "model_id": TOOL.GLM53_MODEL_ID,
            "model_revision": runtime["model_revision"],
            "default_speculation": selected,
            "cold_wall_ms": 10.0,
            "warm_wall_ms": 5.0,
            "cold_to_warm_ratio": 2.0,
            "svg": {},
            "sources": {"serving": serving_identity},
        },
        TOOL.MICRO_SCHEMA: {
            "model_id": TOOL.GLM53_MODEL_ID,
            "model_revision": runtime["model_revision"],
            "profile": "balanced",
            "speculation": selected,
            "selected_request": {},
            "svg": {},
            "evidence": {"serving": serving_identity},
        },
    }
    monkeypatch.setattr(
        TOOL, "signed_serving", lambda _path: (paths["serving"], serving)
    )
    monkeypatch.setattr(
        TOOL,
        "signed_report",
        lambda path, *, schema, statuses: (path, reports[schema]),
    )
    monkeypatch.setattr(
        TOOL,
        "evidence_identity",
        lambda path, schema: identity(path, schema),
    )
    monkeypatch.setattr(
        TOOL,
        "deployment",
        lambda *_args, **_kwargs: {
            "identity": deployment_identity,
            "launch_started_ns": 10,
            "slot": "final",
        },
    )
    monkeypatch.setattr(
        TOOL,
        "validate_context_decode",
        lambda *_args, **_kwargs: {
            "identity": identity(tmp_path / "context.jsonl", "context-v1"),
            "prompt_contract_sha256": "7" * 64,
            "corpus_sha256": "8" * 64,
            "tokenizer_sha256": "9" * 64,
            "cells": [],
        },
    )
    monkeypatch.setattr(TOOL, "revalidate_identities", lambda *_args, **_kwargs: None)

    report = TOOL.validate(
        serving_path=paths["serving"],
        agentic_path=paths["agentic"],
        profiles_path=paths["profiles"],
        context_decode_path=tmp_path / "context.jsonl",
        startup_timeline_path=paths["startup"],
        micro_timeline_path=paths["micro"],
    )
    assert report["status"] == "accepted"
    assert report["default_speculation"] == selected

    reports[TOOL.PROFILE_SCHEMA]["runtime"]["engine_identity"] = "wrong"
    with pytest.raises(TOOL.OssReleaseError, match="selected model/runtime"):
        TOOL.validate(
            serving_path=paths["serving"],
            agentic_path=paths["agentic"],
            profiles_path=paths["profiles"],
            context_decode_path=tmp_path / "context.jsonl",
            startup_timeline_path=paths["startup"],
            micro_timeline_path=paths["micro"],
        )


def test_recursive_identity_revalidation_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text("first\n", encoding="utf-8")
    evidence = {
        "nested": [
            {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        ]
    }
    TOOL.revalidate_identities(evidence, checked={})
    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(TOOL.OssReleaseError, match="changed"):
        TOOL.revalidate_identities(evidence, checked={})


def test_recursive_identity_revalidation_ignores_report_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    evidence = {
        "artifact": {
            "path": "generated/index.html",
            "bytes": 12_345,
            "sha256": "a" * 64,
        }
    }
    TOOL.revalidate_identities(evidence, checked={})
