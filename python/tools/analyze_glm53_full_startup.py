#!/usr/bin/env python3
"""Align and bind one GLM-5.3 WIP launcher/coordinator/Spark startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from validate_glm52_exl3_artifact import _json_object
from validate_glm52_exl3_serving_qualification import (
    deployment,
    startup,
)
from validate_glm53_exl3_serving_qualification import (
    GLM53_MODEL_ID,
    GLM53_STARTUP_SCHEMA,
    MODE_DFLASH2,
    MODE_NATIVE_MTP,
    QualificationError,
)


SCHEMA = "glmrt-glm53-full-startup-v1"
PHASE_RE = re.compile(
    r"^(?P<prefix>[a-z_]+_startup_phase) stage=(?P<stage>[A-Za-z0-9-]+) "
    r"elapsed_ms=(?P<elapsed>[0-9]+(?:\.[0-9]+)?) "
    r"total_ms=(?P<total>[0-9]+(?:\.[0-9]+)?)$"
)
LAUNCHER_PREFIX = "wip_launcher_startup_phase"
SHELL_PREFIX = "coordinator_shell_startup_phase"
REAL_FULL_PREFIX = "real_full_startup_phase"
COORDINATOR_PREFIX = "coordinator_startup_phase"
LAUNCHER_STAGES = (
    "bootstrap",
    "slot-validation",
    "profile-resolution",
    "service-reconciliation",
    "model-snapshots",
    "launch-headroom",
    "spark-dispatch",
    "coordinator-dispatch",
    "api-ready",
)
SHELL_STAGES = (
    "configuration",
    "host-python",
    "sparkinfer-source-verification",
    "kernel-cache-identity",
    "expert-launch",
    "daemon-build",
    "expert-warmup-dispatch",
    "coordinator-exec",
)
REAL_FULL_REQUIRED_ORDER = (
    "validation",
    "catalog-kv-config",
    "targets-tokenizer",
    "kv-snapshot-config",
    "prewarm-prompts",
    "coordinator-resident-preload",
    "dspark-preload",
    "sparse-target-connect",
    "expert-warmup",
    "dispatch-worker",
    "executor-assembly",
    "python-capture-barrier",
    "prewarm-main",
    "prewarm-audit-seal",
    "complete",
)
REAL_FULL_ALLOWED = frozenset(
    {
        *REAL_FULL_REQUIRED_ORDER,
        "request-worker-spawn",
        "request-worker-inline",
        "prewarm-paired-lm-head-initial",
        "prewarm-batched-dspark",
    }
)
COORDINATOR_STAGES = ("real-full-serving", "api-bind")


class FullStartupError(RuntimeError):
    """Startup evidence is incomplete or does not identify one launch."""


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


def regular_log(path: Path, label: str) -> tuple[Path, list[str]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise FullStartupError(f"{label} is a symbolic link")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise FullStartupError(f"{label} is not one regular file")
    try:
        lines = resolved.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise FullStartupError(f"{label} is not UTF-8") from error
    return resolved, lines


def _phase_rows(lines: list[str], prefix: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(lines, 1):
        match = PHASE_RE.fullmatch(line)
        if match is None or match.group("prefix") != prefix:
            continue
        elapsed = float(match.group("elapsed"))
        total = float(match.group("total"))
        if not math.isfinite(elapsed) or not math.isfinite(total):
            raise FullStartupError(f"{prefix} contains a nonfinite phase")
        rows.append(
            {
                "stage": match.group("stage"),
                "elapsed_ms": elapsed,
                "total_ms": total,
                "line": line_number,
            }
        )
    return rows


def parse_exact_phases(
    lines: list[str], prefix: str, expected: tuple[str, ...]
) -> list[dict[str, Any]]:
    rows = _phase_rows(lines, prefix)
    starts = [index for index, row in enumerate(rows) if row["stage"] == expected[0]]
    if not starts:
        raise FullStartupError(f"{prefix} has no complete-process boundary")
    selected = rows[starts[-1] :]
    if tuple(row["stage"] for row in selected) != expected:
        raise FullStartupError(f"{prefix} phase order or coverage differs")
    validate_cumulative(selected, prefix)
    return selected


def parse_real_full_phases(lines: list[str]) -> list[dict[str, Any]]:
    rows = _phase_rows(lines, REAL_FULL_PREFIX)
    starts = [index for index, row in enumerate(rows) if row["stage"] == "validation"]
    if not starts:
        raise FullStartupError("coordinator log has no real-full process boundary")
    selected = rows[starts[-1] :]
    if selected[-1]["stage"] != "complete":
        raise FullStartupError("real-full startup never completed")
    stages = [row["stage"] for row in selected]
    if (
        len(stages) != len(set(stages))
        or not set(stages) <= REAL_FULL_ALLOWED
        or [stage for stage in stages if stage in REAL_FULL_REQUIRED_ORDER]
        != list(REAL_FULL_REQUIRED_ORDER)
        or ("request-worker-spawn" in stages) == ("request-worker-inline" in stages)
    ):
        raise FullStartupError("real-full startup phase order or coverage differs")
    validate_cumulative(selected, REAL_FULL_PREFIX)
    return selected


def validate_cumulative(rows: list[dict[str, Any]], label: str) -> None:
    previous = 0.0
    for row in rows:
        elapsed = row["elapsed_ms"]
        total = row["total_ms"]
        if elapsed < 0.0 or total < previous:
            raise FullStartupError(f"{label} has impossible cumulative timing")
        if not math.isclose(total - previous, elapsed, abs_tol=2.0, rel_tol=0.0):
            raise FullStartupError(f"{label} elapsed and total timing do not reconcile")
        previous = total


def analyze(
    *,
    cache_state: str,
    mode: str,
    deployment_path: Path,
    launcher_log_path: Path,
    coordinator_log_path: Path,
    expert_startup_path: Path,
) -> dict[str, Any]:
    if cache_state not in {"cold", "warm"} or mode not in {
        MODE_NATIVE_MTP,
        MODE_DFLASH2,
    }:
        raise FullStartupError("startup cache state or speculation mode is invalid")
    try:
        deployed = deployment(
            deployment_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=mode,
        )
        experts = startup(
            expert_startup_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_weight_format="exl3",
            expected_preload_modes={"direct-resident", "cooperative-coalesced"},
            expected_include_mtp=mode == MODE_NATIVE_MTP,
            expected_schema=GLM53_STARTUP_SCHEMA,
        )
    except QualificationError as error:
        raise FullStartupError("deployment or Spark startup evidence is invalid") from error
    if (
        experts["expert_runtime_fingerprint"]
        != deployed["fingerprints"]["expert_runtime"]
    ):
        raise FullStartupError("Spark startup/runtime fingerprint differs")
    expert_raw = _json_object(expert_startup_path.expanduser().resolve(strict=True))
    if expert_raw.get("cache_state") not in {"cold", "warm"}:
        raise FullStartupError("Spark startup has no compiled-cache state")

    launcher_path, launcher_lines = regular_log(launcher_log_path, "launcher log")
    coordinator_path, coordinator_lines = regular_log(
        coordinator_log_path, "coordinator log"
    )
    launcher = parse_exact_phases(
        launcher_lines, LAUNCHER_PREFIX, LAUNCHER_STAGES
    )
    shell = parse_exact_phases(coordinator_lines, SHELL_PREFIX, SHELL_STAGES)
    real_full = parse_real_full_phases(coordinator_lines)
    coordinator = parse_exact_phases(
        coordinator_lines, COORDINATOR_PREFIX, COORDINATOR_STAGES
    )
    starting = sum(
        line == "== starting WIP Spark expert processes ==" for line in launcher_lines
    )
    retaining = sum(
        line == "== retaining WIP Spark expert processes ==" for line in launcher_lines
    )
    if (
        cache_state == "cold"
        and (starting != 1 or retaining != 0)
    ) or (
        cache_state == "warm"
        and (retaining != 1 or starting != 0)
    ):
        raise FullStartupError(
            "launcher expert lifecycle does not match requested cache state"
        )
    launcher_total = launcher[-1]["total_ms"]
    shell_total = shell[-1]["total_ms"]
    coordinator_total = coordinator[-1]["total_ms"]
    real_full_total = real_full[-1]["total_ms"]
    if real_full_total > coordinator_total + 2.0:
        raise FullStartupError("real-full startup exceeds coordinator startup")
    dispatch_total = next(
        row["total_ms"] for row in launcher if row["stage"] == "coordinator-dispatch"
    )
    if dispatch_total + shell_total + coordinator_total > launcher_total + 5_000.0:
        raise FullStartupError("aligned coordinator timeline exceeds launcher wall time")
    spark_dispatch_total = next(
        row["total_ms"] for row in launcher if row["stage"] == "spark-dispatch"
    )
    if cache_state == "cold" and (
        spark_dispatch_total + experts["maximum_service_handoff_total_ms"]
        > launcher_total + 5_000.0
    ):
        raise FullStartupError("aligned Spark timeline exceeds launcher wall time")

    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model_id": GLM53_MODEL_ID,
        "model_revision": deployed["model_revision"],
        "launch_state": cache_state,
        "expert_compiled_cache_state": expert_raw["cache_state"],
        "profile": deployed["profile"],
        "speculation": mode,
        "speculation_settings": deployed["speculation_settings"],
        "power_limit_w": deployed["power_limit_w"],
        "engine_identity": deployed["engine_identity"],
        "sparkinfer_revision": deployed["sparkinfer_revision"],
        "expert_runtime_fingerprint": deployed["fingerprints"]["expert_runtime"],
        "alignment": {
            "launcher_wall_ms": launcher_total,
            "spark_dispatch_offset_ms": spark_dispatch_total,
            "coordinator_dispatch_offset_ms": dispatch_total,
            "coordinator_shell_ms": shell_total,
            "coordinator_daemon_ms": coordinator_total,
            "spark_ready_ms": (
                spark_dispatch_total + experts["maximum_service_handoff_total_ms"]
                if cache_state == "cold"
                else 0.0
            ),
            "experts_resident_at_start": cache_state == "warm",
        },
        "phases": {
            "launcher": launcher,
            "coordinator_shell": shell,
            "coordinator_daemon": coordinator,
            "real_full": real_full,
            "spark_hosts": expert_raw["hosts"],
        },
        "evidence": {
            "deployment": deployed["identity"],
            "expert_startup": experts["identity"],
            "launcher_log": {
                "schema": "glmrt-wip-launcher-log-v1",
                "path": os.fspath(launcher_path),
                "bytes": launcher_path.stat().st_size,
                "sha256": hash_file(launcher_path),
            },
            "coordinator_log": {
                "schema": "glmrt-wip-coordinator-log-v1",
                "path": os.fspath(coordinator_path),
                "bytes": coordinator_path.stat().st_size,
                "sha256": hash_file(coordinator_path),
            },
        },
    }
    return body | {"report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FullStartupError(f"refusing to overwrite output: {destination}")
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
    parser.add_argument("--cache-state", choices=("cold", "warm"), required=True)
    parser.add_argument("--speculation", choices=("mtp", "dflash2"), required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--launcher-log", type=Path, required=True)
    parser.add_argument("--coordinator-log", type=Path, required=True)
    parser.add_argument("--expert-startup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        cache_state=args.cache_state,
        mode=args.speculation,
        deployment_path=args.deployment,
        launcher_log_path=args.launcher_log,
        coordinator_log_path=args.coordinator_log,
        expert_startup_path=args.expert_startup,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
