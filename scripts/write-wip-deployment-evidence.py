#!/usr/bin/env python3
"""Write content-bound evidence for one successfully launched GLMRT WIP stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any


SCHEMA = "glmrt-wip-deployment-evidence-v2"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
SLOT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DFLASH2_MODEL_ID = "incoai/GLM-5.3-DFlash2"
DFLASH2_REVISION = "425aa615ce320caac34400208b30808c8f14f76c"
DFLASH2_TOPK_BACKENDS = frozenset(("torch", "flashinfer", "flashinfer-dsa"))


class EvidenceError(RuntimeError):
    """The launch identity is incomplete or internally inconsistent."""


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
        raise EvidenceError(f"{label} is a symbolic link")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise EvidenceError(f"{label} is not one regular file")
    return resolved


def build_evidence(
    *,
    model_id: str,
    model_revision: str,
    slot: str,
    profile: str,
    speculation: str,
    launch_started_ns: int,
    power_limit_w: int,
    coordinator_slot_fingerprint: str,
    expert_slot_fingerprint: str,
    expert_runtime_fingerprint: str,
    deployment_fingerprint: str,
    engine_identity: str,
    sparkinfer_revision: str,
    resolved_profile_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    fingerprints = {
        "coordinator_slot": coordinator_slot_fingerprint,
        "expert_slot": expert_slot_fingerprint,
        "expert_runtime": expert_runtime_fingerprint,
        "deployment": deployment_fingerprint,
    }
    if not model_id or REVISION_RE.fullmatch(model_revision) is None:
        raise EvidenceError("model ID or revision is invalid")
    if SLOT_RE.fullmatch(slot) is None:
        raise EvidenceError("WIP slot name is invalid")
    if profile not in {"balanced", "long", "accuracy"}:
        raise EvidenceError("serving profile is invalid")
    if speculation not in {"plain", "mtp", "dspark", "dflash2"}:
        raise EvidenceError("speculation mode is invalid")
    if isinstance(launch_started_ns, bool) or launch_started_ns <= 0:
        raise EvidenceError("launcher start time must be a positive integer")
    if isinstance(power_limit_w, bool) or power_limit_w <= 0:
        raise EvidenceError("power limit must be positive")
    if any(SHA256_RE.fullmatch(value) is None for value in fingerprints.values()):
        raise EvidenceError("WIP fingerprints must be lowercase SHA-256 values")
    expected_engine = (
        f"wip-{slot}-{coordinator_slot_fingerprint[:12]}-"
        f"{expert_slot_fingerprint[:12]}"
    )
    if engine_identity != expected_engine:
        raise EvidenceError("engine identity differs from the selected WIP slots")
    if REVISION_RE.fullmatch(sparkinfer_revision) is None:
        raise EvidenceError("SparkInfer revision is invalid")

    resolved_profile = regular_file(resolved_profile_path, "resolved profile")
    config = regular_file(config_path, "launch configuration")
    try:
        resolved = json.loads(resolved_profile.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceError("resolved profile is not valid JSON") from error
    if (
        not isinstance(resolved, dict)
        or resolved.get("model_id") != model_id
        or resolved.get("profile") != profile
        or resolved.get("speculation") != speculation
        or resolved.get("blockers") != []
    ):
        raise EvidenceError("resolved profile differs from the launched selection")
    environment = resolved.get("environment")
    if not isinstance(environment, dict):
        raise EvidenceError("resolved profile has no environment contract")
    speculation_settings: dict[str, Any] = {}
    if speculation == "dflash2":
        fixed_raw = environment.get("GLMRT_REAL_FULL_DFLASH2_FIXED_DRAFTS")
        topk_backend = environment.get("GLMRT_REAL_FULL_DFLASH2_TOPK_BACKEND")
        adaptive = fixed_raw == "adaptive"
        fixed_drafts: int | None = None
        if not adaptive:
            try:
                fixed_drafts = int(fixed_raw)
            except (TypeError, ValueError) as error:
                raise EvidenceError(
                    "resolved DFlash2 width is neither adaptive nor an integer"
                ) from error
        if (
            environment.get("GLMRT_DFLASH2_MODEL_ID") != DFLASH2_MODEL_ID
            or environment.get("GLMRT_DFLASH2_REVISION") != DFLASH2_REVISION
            or (
                not adaptive
                and (
                    fixed_drafts is None
                    or str(fixed_drafts) != fixed_raw
                    or not 1 <= fixed_drafts <= 7
                )
            )
            or topk_backend not in DFLASH2_TOPK_BACKENDS
        ):
            raise EvidenceError(
                "resolved DFlash2 checkpoint, width, or top-k backend is invalid"
            )
        speculation_settings = {
            "checkpoint_model_id": DFLASH2_MODEL_ID,
            "checkpoint_revision": DFLASH2_REVISION,
            "draft_policy": "adaptive" if adaptive else "fixed",
            "proposal_drafts": 7,
            "fixed_drafts": fixed_drafts,
            "topk_backend": topk_backend,
        }

    body = {
        "schema": SCHEMA,
        "status": "ready",
        "model_id": model_id,
        "model_revision": model_revision,
        "slot": slot,
        "profile": profile,
        "speculation": speculation,
        "speculation_settings": speculation_settings,
        "launch_started_ns": launch_started_ns,
        "power_limit_w": power_limit_w,
        "engine_identity": engine_identity,
        "sparkinfer_revision": sparkinfer_revision,
        "fingerprints": fingerprints,
        "inputs": {
            "resolved_profile": {
                "bytes": resolved_profile.stat().st_size,
                "sha256": hash_file(resolved_profile),
            },
            "configuration": {
                "bytes": config.stat().st_size,
                "sha256": hash_file(config),
            },
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
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--slot", required=True)
    parser.add_argument("--profile", choices=("balanced", "long", "accuracy"), required=True)
    parser.add_argument(
        "--speculation",
        choices=("plain", "mtp", "dspark", "dflash2"),
        required=True,
    )
    parser.add_argument("--launch-started-ns", type=int, required=True)
    parser.add_argument("--power-limit-w", type=int, required=True)
    parser.add_argument("--coordinator-slot-fingerprint", required=True)
    parser.add_argument("--expert-slot-fingerprint", required=True)
    parser.add_argument("--expert-runtime-fingerprint", required=True)
    parser.add_argument("--deployment-fingerprint", required=True)
    parser.add_argument("--engine-identity", required=True)
    parser.add_argument("--sparkinfer-revision", required=True)
    parser.add_argument("--resolved-profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_evidence(
        model_id=args.model_id,
        model_revision=args.model_revision,
        slot=args.slot,
        profile=args.profile,
        speculation=args.speculation,
        launch_started_ns=args.launch_started_ns,
        power_limit_w=args.power_limit_w,
        coordinator_slot_fingerprint=args.coordinator_slot_fingerprint,
        expert_slot_fingerprint=args.expert_slot_fingerprint,
        expert_runtime_fingerprint=args.expert_runtime_fingerprint,
        deployment_fingerprint=args.deployment_fingerprint,
        engine_identity=args.engine_identity,
        sparkinfer_revision=args.sparkinfer_revision,
        resolved_profile_path=args.resolved_profile,
        config_path=args.config,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
