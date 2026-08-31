#!/usr/bin/env python3
"""Bind final GLM-5.3 agentic release runs to one qualified deployment."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any

from validate_glm52_exl3_serving_qualification import deployment, tool_eval
from validate_glm53_exl3_serving_qualification import (
    GLM53_MODEL_ID,
    MODES,
    SCHEMA as SERVING_SCHEMA,
    QualificationError,
)
from validate_pi_coding_agent_run import (
    SCHEMA as PI_SCHEMA,
    PiEvidenceError,
    revalidate as revalidate_pi,
)


SCHEMA = "glmrt-glm53-agentic-release-evidence-v1"
TOOL_SEEDS = (2_026_082_901, 2_026_082_902, 2_026_082_903)


class AgenticReleaseError(RuntimeError):
    """Final agentic evidence is incomplete or from a different runtime."""


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
        raise AgenticReleaseError(f"{label} is a symbolic link")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise AgenticReleaseError(f"{label} is not one regular file")
    return resolved


def identity(path: Path, schema: str) -> dict[str, Any]:
    resolved = regular_file(path, schema)
    return {
        "schema": schema,
        "path": os.fspath(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": hash_file(resolved),
    }


def signed_serving(path: Path) -> tuple[Path, dict[str, Any]]:
    resolved = regular_file(path, "serving qualification")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AgenticReleaseError("serving qualification is not valid JSON") from error
    body = (
        {key: value for key, value in report.items() if key != "report_sha256"}
        if isinstance(report, dict)
        else None
    )
    gates = report.get("gates") if isinstance(report, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("schema") != SERVING_SCHEMA
        or report.get("status") != "accepted"
        or report.get("model_id") != GLM53_MODEL_ID
        or report.get("failed_gates") != []
        or not isinstance(gates, dict)
        or not gates
        or not all(value is True for value in gates.values())
        or not isinstance(body, dict)
        or report.get("report_sha256")
        != hashlib.sha256(canonical_json(body)).hexdigest()
    ):
        raise AgenticReleaseError("serving qualification is not fully accepted")
    return resolved, report


def timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise AgenticReleaseError(f"{label} timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise AgenticReleaseError(f"{label} timestamp has no timezone")
    return parsed


def tool_run_timestamp(run_id: Any) -> datetime:
    if not isinstance(run_id, str) or "_" not in run_id:
        raise AgenticReleaseError("tool evaluation has no timestamped run ID")
    stamp, suffix = run_id.rsplit("_", 1)
    if not suffix or "T" not in stamp:
        raise AgenticReleaseError("tool evaluation run ID is invalid")
    date, clock = stamp.split("T", 1)
    return timestamp(f"{date}T{clock.replace('-', ':', 2)}", "tool evaluation")


def raw_json(path: Path) -> dict[str, Any]:
    resolved = regular_file(path, "tool evaluation")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AgenticReleaseError("tool evaluation is not valid JSON") from error
    if not isinstance(value, dict):
        raise AgenticReleaseError("tool evaluation is not a JSON object")
    return value


def without(value: dict[str, Any], *fields: str) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in fields}


def validate(
    *,
    serving_path: Path,
    deployment_path: Path,
    tool_paths: list[Path],
    pi_off_path: Path,
    pi_high_path: Path,
    node_binary: str = "node",
) -> dict[str, Any]:
    if len(tool_paths) != len(TOOL_SEEDS):
        raise AgenticReleaseError("agentic release requires exactly three tool runs")
    serving_file, serving = signed_serving(serving_path)
    runtime = serving.get("runtime")
    thresholds = serving.get("thresholds")
    selected = serving.get("results", {}).get("default_speculation")
    if (
        not isinstance(runtime, dict)
        or selected not in MODES
        or runtime.get("default_speculation") != selected
        or runtime.get("profile") != "balanced"
        or not isinstance(thresholds, dict)
        or not isinstance(thresholds.get("tool_eval_version"), str)
    ):
        raise AgenticReleaseError("serving qualification has no balanced measured default")
    try:
        deployed = deployment(
            deployment_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=selected,
        )
    except QualificationError as error:
        raise AgenticReleaseError("selected deployment evidence is invalid") from error
    expected_deployment = {
        "model_revision": runtime.get("model_revision"),
        "profile": runtime.get("profile"),
        "power_limit_w": runtime.get("power_limit_w"),
        "engine_identity": runtime.get("engine_identity"),
        "sparkinfer_revision": runtime.get("sparkinfer_revision"),
        "coordinator_slot": runtime.get("coordinator_slot_fingerprint"),
        "expert_slot": runtime.get("expert_slot_fingerprint"),
        "expert_runtime": runtime.get("expert_runtime_fingerprints", {}).get(selected),
        "speculation_settings": runtime.get("speculation_settings", {}).get(selected),
    }
    actual_deployment = {
        "model_revision": deployed["model_revision"],
        "profile": deployed["profile"],
        "power_limit_w": deployed["power_limit_w"],
        "engine_identity": deployed["engine_identity"],
        "sparkinfer_revision": deployed["sparkinfer_revision"],
        "coordinator_slot": deployed["fingerprints"]["coordinator_slot"],
        "expert_slot": deployed["fingerprints"]["expert_slot"],
        "expert_runtime": deployed["fingerprints"]["expert_runtime"],
        "speculation_settings": deployed["speculation_settings"],
    }
    if actual_deployment != expected_deployment:
        raise AgenticReleaseError(
            "agentic release deployment differs from the qualified default"
        )
    launch_started = datetime.fromtimestamp(
        deployed["launch_started_ns"] / 1_000_000_000,
        tz=timestamp("1970-01-01T00:00:00Z", "epoch").tzinfo,
    )

    parsed_tools = []
    run_ids = []
    for expected_seed, path in zip(TOOL_SEEDS, tool_paths, strict=True):
        try:
            parsed = tool_eval(
                path,
                candidate=True,
                expected_version=thresholds["tool_eval_version"],
                expected_model=GLM53_MODEL_ID,
            )
        except QualificationError as error:
            raise AgenticReleaseError("publication tool evaluation is invalid") from error
        raw = raw_json(path)
        config = parsed["config"]
        metadata = parsed["metadata"]
        run_id = raw.get("run_id")
        started = tool_run_timestamp(run_id)
        if (
            config.get("seed") != expected_seed
            or metadata.get("seed") != expected_seed
            or config.get("concurrency") != 1
            or metadata.get("parallel") != 1
            or metadata.get("trials") != 1
            or metadata.get("thinking_enabled") is not True
            or len(parsed["scenario_ids"]) != 69
            or parsed["maximum_points"] != 138
            or started < launch_started
        ):
            raise AgenticReleaseError(
                "publication tool evaluation has the wrong seed or serial contract"
            )
        run_ids.append(run_id)
        parsed_tools.append(parsed | {"run_id": run_id, "started": started.isoformat()})
    if len(set(run_ids)) != len(run_ids):
        raise AgenticReleaseError("publication tool evaluations reused a run ID")
    reference_config = without(
        parsed_tools[0]["config"], "seed", "config_fingerprint"
    )
    reference_metadata = without(parsed_tools[0]["metadata"], "seed")
    for parsed in parsed_tools[1:]:
        if (
            without(parsed["config"], "seed", "config_fingerprint")
            != reference_config
            or without(parsed["metadata"], "seed") != reference_metadata
            or parsed["scenario_ids"] != parsed_tools[0]["scenario_ids"]
            or parsed["maximum_points"] != parsed_tools[0]["maximum_points"]
        ):
            raise AgenticReleaseError(
                "publication tool evaluations do not share one test contract"
            )

    try:
        pi_reports = {
            "off": revalidate_pi(pi_off_path, node_binary=node_binary),
            "high": revalidate_pi(pi_high_path, node_binary=node_binary),
        }
    except PiEvidenceError as error:
        raise AgenticReleaseError("publication Pi evidence is invalid") from error
    if any(
        report.get("model_id") != GLM53_MODEL_ID
        or report.get("thinking") != mode
        or timestamp(report.get("session_timestamp"), f"Pi {mode}") < launch_started
        for mode, report in pi_reports.items()
    ):
        raise AgenticReleaseError(
            "publication Pi evidence differs from the selected deployment"
        )
    if pi_reports["off"]["session_id"] == pi_reports["high"]["session_id"]:
        raise AgenticReleaseError("publication Pi arms reused one session")

    points = [parsed["total_points"] for parsed in parsed_tools]
    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model_id": GLM53_MODEL_ID,
        "model_revision": deployed["model_revision"],
        "default_speculation": selected,
        "runtime": actual_deployment
        | {
            "launch_started_ns": deployed["launch_started_ns"],
            "slot": deployed["slot"],
        },
        "tool_eval": {
            "version": thresholds["tool_eval_version"],
            "seeds": list(TOOL_SEEDS),
            "points": points,
            "maximum_points": parsed_tools[0]["maximum_points"],
            "median_points": statistics.median(points),
            "runs": [
                {
                    "run_id": parsed["run_id"],
                    "started": parsed["started"],
                    "points": parsed["total_points"],
                    "score": parsed["final_score"],
                    "identity": parsed["identity"],
                }
                for parsed in parsed_tools
            ],
        },
        "pi": {
            mode: {
                key: report[key]
                for key in (
                    "pi_version",
                    "thinking",
                    "session_id",
                    "session_timestamp",
                    "wall_seconds",
                    "turns",
                    "tool_calls",
                    "tool_errors",
                    "usage",
                    "artifact",
                )
            }
            | {"identity": identity(path, PI_SCHEMA)}
            for mode, report, path in (
                ("off", pi_reports["off"], pi_off_path),
                ("high", pi_reports["high"], pi_high_path),
            )
        },
        "evidence": {
            "serving_qualification": identity(serving_file, SERVING_SCHEMA),
            "deployment": deployed["identity"],
        },
    }
    return {**body, "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--tool-eval", type=Path, action="append", required=True)
    parser.add_argument("--pi-off", type=Path, required=True)
    parser.add_argument("--pi-high", type=Path, required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        serving_path=args.serving,
        deployment_path=args.deployment,
        tool_paths=args.tool_eval,
        pi_off_path=args.pi_off,
        pi_high_path=args.pi_high,
        node_binary=args.node,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
