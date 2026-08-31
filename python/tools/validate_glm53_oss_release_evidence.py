#!/usr/bin/env python3
"""Create one complete, source-revalidated GLM-5.3 OSS release evidence report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from bench_release_decode_matrix import (
    DEFAULT_CONTEXTS,
    WORKLOADS,
    canonical_sha256,
    summarize_records,
)
from validate_glm52_exl3_serving_qualification import (
    deployment,
    evidence_identity,
    read_jsonl,
)
from validate_glm53_agentic_release_evidence import (
    SCHEMA as AGENTIC_SCHEMA,
    signed_serving,
)
from validate_glm53_exl3_serving_qualification import (
    GLM53_MODEL_ID,
    MODE_DFLASH2,
    MODES,
    QualificationError,
)
from validate_glm53_profile_release_evidence import (
    SCHEMA as PROFILE_SCHEMA,
    _benchmark_metadata,
)
from render_glm53_balanced_micro_timeline import SCHEMA as MICRO_SCHEMA
from render_glm53_startup_timeline import SCHEMA as STARTUP_TIMELINE_SCHEMA


SCHEMA = "glmrt-glm53-oss-release-evidence-v1"
CONTEXT_SCHEMA = "glmrt-release-decode-v2"
CONTEXT_SUMMARY_SCHEMA = "glmrt-release-decode-summary-v2"
REQUIRED_CONTEXT_REPEATS = 2
REQUIRED_CONTEXT_MAX_TOKENS = 192


class OssReleaseError(RuntimeError):
    """The final OSS release evidence is incomplete, stale, or mismatched."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise OssReleaseError(f"{label} is a symbolic link")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise OssReleaseError(f"{label} is not one regular file")
    return resolved


def signed_report(
    path: Path, *, schema: str, statuses: set[str]
) -> tuple[Path, dict[str, Any]]:
    resolved = regular_file(path, schema)
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OssReleaseError(f"{schema} report is not valid JSON") from error
    body = (
        {key: value for key, value in report.items() if key != "report_sha256"}
        if isinstance(report, dict)
        else None
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != schema
        or report.get("status") not in statuses
        or not isinstance(body, dict)
        or report.get("report_sha256")
        != hashlib.sha256(canonical_json(body)).hexdigest()
    ):
        raise OssReleaseError(f"{schema} report is not signed and accepted")
    return resolved, report


def revalidate_identities(value: Any, *, checked: dict[Path, tuple[int, str]]) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            path = Path(str(value["path"])).expanduser()
            # Signed child reports also carry paths relative to their own
            # artifact roots (for example, Pi's generated index.html and the
            # quantization qualification bundle).  They are record-local
            # identities, not filesystem references relative to this process.
            # Their containing signed report is revalidated through its
            # absolute evidence identity below; only absolute source paths can
            # be safely re-opened here without silently choosing the wrong
            # base directory.
            if not path.is_absolute():
                for child in value.values():
                    revalidate_identities(child, checked=checked)
                return
            if path.is_symlink():
                raise OssReleaseError(f"referenced evidence is a symbolic link: {path}")
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError as error:
                raise OssReleaseError(
                    f"referenced evidence is missing: {path}"
                ) from error
            if not resolved.is_file():
                raise OssReleaseError(f"referenced evidence is not a file: {resolved}")
            actual = checked.get(resolved)
            if actual is None:
                actual = (resolved.stat().st_size, hash_file(resolved))
                checked[resolved] = actual
            if value.get("bytes") != actual[0] or value.get("sha256") != actual[1]:
                raise OssReleaseError(f"referenced evidence changed: {resolved}")
        for child in value.values():
            revalidate_identities(child, checked=checked)
    elif isinstance(value, list):
        for child in value:
            revalidate_identities(child, checked=checked)


def validate_context_decode(
    path: Path, *, deployed: dict[str, Any]
) -> dict[str, Any]:
    resolved, records = read_jsonl(path)
    summaries = [
        record for record in records if record.get("schema") == CONTEXT_SUMMARY_SCHEMA
    ]
    measurements = [
        record for record in records if record.get("schema") == CONTEXT_SCHEMA
    ]
    if len(summaries) != 1 or len(measurements) + 1 != len(records):
        raise OssReleaseError("context decode has an invalid record set")
    summary = summaries[0]
    contexts = list(DEFAULT_CONTEXTS)
    workloads = list(WORKLOADS)
    if (
        summary.get("model") != GLM53_MODEL_ID
        or summary.get("profile") != "balanced"
        or summary.get("contexts") != contexts
        or summary.get("workloads") != workloads
        or summary.get("repeats") != REQUIRED_CONTEXT_REPEATS
        or summary.get("max_tokens") != REQUIRED_CONTEXT_MAX_TOKENS
    ):
        raise OssReleaseError("context decode does not cover the final 5x3 contract")
    try:
        run = _benchmark_metadata(
            resolved,
            kind="context-decode",
            profile="balanced",
            launch_started_ns=deployed["launch_started_ns"],
        )
        expected_cells = summarize_records(
            measurements,
            contexts=contexts,
            workloads=workloads,
            repeats=REQUIRED_CONTEXT_REPEATS,
        )
    except RuntimeError as error:
        raise OssReleaseError("context decode runtime evidence is invalid") from error
    common = {
        "run_id": summary.get("run_id"),
        "profile": "balanced",
        "model": GLM53_MODEL_ID,
        "corpus_root": summary.get("corpus_root"),
        "corpus_sha256": summary.get("corpus_sha256"),
        "tokenizer": summary.get("tokenizer"),
        "tokenizer_sha256": summary.get("tokenizer_sha256"),
    }
    markers = set()
    prompts = []
    for record in measurements:
        content = record.get("content")
        marker = record.get("marker")
        prompt_sha256 = record.get("prompt_sha256")
        if (
            any(record.get(key) != value for key, value in common.items())
            or not isinstance(content, str)
            or not content
            or record.get("content_sha256")
            != hashlib.sha256(content.encode()).hexdigest()
            or not isinstance(marker, str)
            or len(marker) != 1
            or marker in markers
            or not isinstance(prompt_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", prompt_sha256) is None
            or record.get("reasoning_chars") != 0
            or record.get("finish_reason") not in {"stop", "length"}
        ):
            raise OssReleaseError("context decode has unbound request/output evidence")
        markers.add(marker)
        prompts.append(
            {
                "context_bucket_tokens": record["context_bucket_tokens"],
                "workload": record["workload"],
                "repeat": record["repeat"],
                "prompt_sha256": prompt_sha256,
            }
        )
    if (
        summary.get("cells") != expected_cells
        or summary.get("prompt_contract_sha256") != canonical_sha256(prompts)
    ):
        raise OssReleaseError("context decode summary differs from measurements")
    return {
        "identity": evidence_identity(resolved, "glmrt-release-decode-jsonl-v2"),
        "run": run,
        "prompt_contract_sha256": summary["prompt_contract_sha256"],
        "corpus_sha256": summary["corpus_sha256"],
        "tokenizer_sha256": summary["tokenizer_sha256"],
        "cells": expected_cells,
    }


def _same_identity(
    actual: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    if actual != expected:
        raise OssReleaseError(f"{label} does not reference the same evidence")


def validate(
    *,
    serving_path: Path,
    agentic_path: Path,
    profiles_path: Path,
    context_decode_path: Path,
    startup_timeline_path: Path,
    micro_timeline_path: Path,
) -> dict[str, Any]:
    try:
        serving_file, serving = signed_serving(serving_path)
    except Exception as error:
        raise OssReleaseError("serving qualification is invalid") from error
    agentic_file, agentic = signed_report(
        agentic_path, schema=AGENTIC_SCHEMA, statuses={"accepted"}
    )
    profiles_file, profiles = signed_report(
        profiles_path, schema=PROFILE_SCHEMA, statuses={"accepted"}
    )
    startup_file, startup_timeline = signed_report(
        startup_timeline_path,
        schema=STARTUP_TIMELINE_SCHEMA,
        statuses={"rendered"},
    )
    micro_file, micro_timeline = signed_report(
        micro_timeline_path, schema=MICRO_SCHEMA, statuses={"rendered"}
    )
    selected = serving["results"]["default_speculation"]
    runtime = serving["runtime"]
    if selected not in MODES:
        raise OssReleaseError("serving qualification has no selected default")
    serving_identity = evidence_identity(serving_file, serving["schema"])
    _same_identity(
        agentic["evidence"]["serving_qualification"],
        serving_identity,
        "agentic report",
    )
    _same_identity(profiles["evidence"]["serving"], serving_identity, "profile report")
    _same_identity(
        startup_timeline["sources"]["serving"],
        serving_identity,
        "startup timeline",
    )
    _same_identity(
        micro_timeline["evidence"]["serving"],
        serving_identity,
        "micro timeline",
    )
    default_deployment_identity = agentic["evidence"]["deployment"]
    _same_identity(
        default_deployment_identity,
        serving["evidence"][f"{selected}_deployment"],
        "selected deployment",
    )
    try:
        deployed = deployment(
            Path(default_deployment_identity["path"]),
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=selected,
        )
    except (KeyError, QualificationError) as error:
        raise OssReleaseError("selected deployment evidence is invalid") from error
    _same_identity(
        deployed["identity"], default_deployment_identity, "selected deployment"
    )
    expected_agentic_runtime = {
        "model_revision": runtime["model_revision"],
        "profile": runtime["profile"],
        "power_limit_w": runtime["power_limit_w"],
        "engine_identity": runtime["engine_identity"],
        "sparkinfer_revision": runtime["sparkinfer_revision"],
        "coordinator_slot": runtime["coordinator_slot_fingerprint"],
        "expert_slot": runtime["expert_slot_fingerprint"],
        "expert_runtime": runtime["expert_runtime_fingerprints"][selected],
        "speculation_settings": runtime["speculation_settings"][selected],
        "launch_started_ns": deployed["launch_started_ns"],
        "slot": deployed["slot"],
    }
    expected_profile_runtime = {
        "model_revision": runtime["model_revision"],
        "slot": deployed["slot"],
        "power_limit_w": runtime["power_limit_w"],
        "engine_identity": runtime["engine_identity"],
        "sparkinfer_revision": runtime["sparkinfer_revision"],
        "coordinator_slot": runtime["coordinator_slot_fingerprint"],
        "expert_slot": runtime["expert_slot_fingerprint"],
    }
    if any(
        report.get("model_id") != GLM53_MODEL_ID
        or report.get("model_revision") != runtime["model_revision"]
        for report in (agentic, profiles, startup_timeline, micro_timeline)
    ) or (
        agentic.get("default_speculation") != selected
        or agentic.get("runtime") != expected_agentic_runtime
        or profiles.get("runtime") != expected_profile_runtime
        or profiles.get("speculation_settings")
        != {MODE_DFLASH2: runtime["speculation_settings"][MODE_DFLASH2]}
        or startup_timeline.get("default_speculation") != selected
        or micro_timeline.get("speculation") != selected
        or micro_timeline.get("profile") != "balanced"
    ):
        raise OssReleaseError("release reports differ from the selected model/runtime")
    context_decode = validate_context_decode(context_decode_path, deployed=deployed)

    checked: dict[Path, tuple[int, str]] = {}
    # Re-open the declared source-evidence trees. Other report sections can
    # contain identities captured inside containers or paths relative to an
    # artifact workspace; those are signed measurements, not host source
    # files. Walking entire reports would incorrectly reinterpret them in the
    # OSS binder's filesystem namespace.
    for evidence in (
        serving["evidence"],
        agentic["evidence"],
        profiles["evidence"],
        startup_timeline["sources"],
        micro_timeline["evidence"],
        context_decode["identity"],
    ):
        revalidate_identities(evidence, checked=checked)

    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model_id": GLM53_MODEL_ID,
        "model_revision": runtime["model_revision"],
        "default_speculation": selected,
        "runtime": {
            "engine_identity": runtime["engine_identity"],
            "sparkinfer_revision": runtime["sparkinfer_revision"],
            "power_limit_w": runtime["power_limit_w"],
            "profile": runtime["profile"],
            "speculation_settings": runtime["speculation_settings"],
        },
        "results": {
            "serving": serving["results"],
            "agentic": {
                "tool_eval": agentic["tool_eval"],
                "pi": agentic["pi"],
            },
            "profiles": {
                "results": profiles["results"],
                "retention": profiles["profile_retention"],
            },
            "context_decode": {
                "prompt_contract_sha256": context_decode[
                    "prompt_contract_sha256"
                ],
                "corpus_sha256": context_decode["corpus_sha256"],
                "tokenizer_sha256": context_decode["tokenizer_sha256"],
                "cells": context_decode["cells"],
            },
            "startup": {
                "cold_wall_ms": startup_timeline["cold_wall_ms"],
                "warm_wall_ms": startup_timeline["warm_wall_ms"],
                "cold_to_warm_ratio": startup_timeline["cold_to_warm_ratio"],
                "svg": startup_timeline["svg"],
            },
            "micro_timeline": {
                "selected_request": micro_timeline["selected_request"],
                "svg": micro_timeline["svg"],
            },
        },
        "evidence": {
            "serving": serving_identity,
            "agentic": evidence_identity(agentic_file, AGENTIC_SCHEMA),
            "profiles": evidence_identity(profiles_file, PROFILE_SCHEMA),
            "context_decode": context_decode["identity"],
            "startup_timeline": evidence_identity(
                startup_file, STARTUP_TIMELINE_SCHEMA
            ),
            "micro_timeline": evidence_identity(micro_file, MICRO_SCHEMA),
        },
    }
    return body | {"report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise OssReleaseError(f"refusing to overwrite output: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--agentic", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--context-decode", type=Path, required=True)
    parser.add_argument("--startup-timeline", type=Path, required=True)
    parser.add_argument("--micro-timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        serving_path=args.serving,
        agentic_path=args.agentic,
        profiles_path=args.profiles,
        context_decode_path=args.context_decode,
        startup_timeline_path=args.startup_timeline,
        micro_timeline_path=args.micro_timeline,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
