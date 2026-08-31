#!/usr/bin/env python3
"""Bind the three DFlash2 GLM-5.3 profile arms to one qualified runtime."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from validate_glm52_exl3_serving_qualification import (
    blended,
    deployment,
    evidence_identity,
    finite_positive,
    integer,
    prefill,
    read_jsonl,
    require_close,
)
from validate_glm53_agentic_release_evidence import signed_serving
from validate_glm53_exl3_serving_qualification import (
    GLM53_MODEL_ID,
    MODE_DFLASH2,
    QualificationError,
    require_eight_type_blended,
    verify_cycle_curve,
)


SCHEMA = "glmrt-glm53-profile-release-evidence-v1"
PROFILES = ("balanced", "long", "accuracy")
MODES = (MODE_DFLASH2,)
REQUIRED_PREFILL_CELL = (2_048, 8_192)


class ProfileReleaseError(RuntimeError):
    """Profile release evidence is incomplete, stale, or mismatched."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def ratio(numerator: float, denominator: float, label: str) -> float:
    if not math.isfinite(numerator) or numerator < 0.0 or denominator <= 0.0:
        raise ProfileReleaseError(f"cannot compute {label} ratio")
    return numerator / denominator


def _timestamp_ns(value: Any, label: str) -> int:
    if not isinstance(value, str):
        raise ProfileReleaseError(f"{label} has no timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ProfileReleaseError(f"{label} has an invalid timestamp") from error
    if parsed.tzinfo is None:
        raise ProfileReleaseError(f"{label} timestamp has no timezone")
    return int(parsed.timestamp() * 1_000_000_000)


def _benchmark_metadata(
    path: Path,
    *,
    kind: str,
    profile: str,
    launch_started_ns: int,
) -> dict[str, Any]:
    resolved, records = read_jsonl(path)
    if kind == "blended":
        summaries = [record["aggregate"] for record in records if "aggregate" in record]
        measurements = [record for record in records if "aggregate" not in record]
    elif kind == "prefill":
        summaries = [
            record
            for record in records
            if record.get("schema") == "glmrt-release-prefill-summary-v3"
        ]
        measurements = [
            record
            for record in records
            if record.get("schema") == "glmrt-release-prefill-v2"
        ]
    elif kind == "context-decode":
        summaries = [
            record
            for record in records
            if record.get("schema") == "glmrt-release-decode-summary-v2"
        ]
        measurements = [
            record
            for record in records
            if record.get("schema") == "glmrt-release-decode-v2"
        ]
    else:
        raise AssertionError(f"unsupported benchmark kind: {kind}")
    if len(summaries) != 1 or not measurements:
        raise ProfileReleaseError(f"{resolved} has no unique {kind} summary")
    summary = summaries[0]
    run_id = summary.get("run_id")
    started_ns = summary.get("benchmark_started_ns")
    completed_ns = summary.get("benchmark_completed_ns")
    if (
        not isinstance(run_id, str)
        or not run_id
        or isinstance(started_ns, bool)
        or not isinstance(started_ns, int)
        or isinstance(completed_ns, bool)
        or not isinstance(completed_ns, int)
        or started_ns <= launch_started_ns
        or completed_ns < started_ns
        or summary.get("profile") != profile
    ):
        raise ProfileReleaseError(
            f"{resolved} is not a post-deployment {profile} {kind} run"
        )
    summary_timestamp_ns = _timestamp_ns(
        summary.get("timestamp_utc"), f"{resolved} {kind} summary"
    )
    if not started_ns <= summary_timestamp_ns <= completed_ns + 1_000_000:
        raise ProfileReleaseError(f"{resolved} has inconsistent summary time")
    for index, record in enumerate(measurements):
        measured_ns = _timestamp_ns(
            record.get("timestamp_utc"), f"{resolved} {kind} row {index}"
        )
        if (
            record.get("run_id") != run_id
            or record.get("profile") != profile
            or not started_ns <= measured_ns <= completed_ns + 1_000_000
        ):
            raise ProfileReleaseError(
                f"{resolved} has a row outside its {profile} {kind} run"
            )
    return {
        "run_id": run_id,
        "benchmark_started_ns": started_ns,
        "benchmark_completed_ns": completed_ns,
    }


def _verify_throughput(
    path: Path, expected_fixed_drafts: int | None
) -> tuple[float, dict[str, Any]]:
    curve = verify_cycle_curve(path, expected_fixed_drafts=expected_fixed_drafts)
    resolved, records = read_jsonl(path)
    aggregates = [record["aggregate"] for record in records if "aggregate" in record]
    measurements = [record for record in records if "aggregate" not in record]
    if len(aggregates) != 1:
        raise ProfileReleaseError(f"{resolved} has no unique blended aggregate")
    emitted = sum(
        integer(
            record.get("emitted_tokens_from_verify"),
            f"{resolved}: emitted verify tokens",
        )
        for record in measurements
    )
    verify_ms = sum(
        sum(
            finite_positive(value, f"{resolved}: verify cycle milliseconds")
            for value in record.get("verify_cycle_ms", [])
        )
        for record in measurements
    )
    if emitted <= 0 or verify_ms <= 0.0:
        raise ProfileReleaseError(f"{resolved} has no verify throughput")
    measured = emitted * 1_000.0 / verify_ms
    reported = require_close(
        aggregates[0].get("emitted_tokens_per_verify_cycle_second"),
        measured,
        f"{resolved}: verify throughput",
    )
    return reported, curve


def _validate_arm(
    *,
    profile: str,
    mode: str,
    deployment_path: Path,
    blended_path: Path,
    prefill_path: Path,
) -> dict[str, Any]:
    try:
        deployed = deployment(
            deployment_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
            expected_speculation=mode,
        )
        decoded = blended(
            blended_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
        )
        require_eight_type_blended(decoded)
        prefetched = prefill(
            prefill_path,
            candidate=True,
            expected_model=GLM53_MODEL_ID,
        )
    except QualificationError as error:
        raise ProfileReleaseError(
            f"{profile}/{mode} contains invalid qualification evidence"
        ) from error
    if deployed["profile"] != profile or prefetched["profile"] != profile:
        raise ProfileReleaseError(f"{profile}/{mode} profile identity differs")
    fixed = (
        deployed["speculation_settings"]["fixed_drafts"]
        if mode == MODE_DFLASH2
        else None
    )
    verify_tps, verify_curve = _verify_throughput(blended_path, fixed)
    blended_run = _benchmark_metadata(
        blended_path,
        kind="blended",
        profile=profile,
        launch_started_ns=deployed["launch_started_ns"],
    )
    prefill_run = _benchmark_metadata(
        prefill_path,
        kind="prefill",
        profile=profile,
        launch_started_ns=deployed["launch_started_ns"],
    )
    if set(prefetched["cells"]) != {REQUIRED_PREFILL_CELL}:
        raise ProfileReleaseError(
            f"{profile}/{mode} must contain exactly cached-2K/+8K prefill"
        )
    required_prompts = [
        prompt
        for prompt in prefetched["prompts"]
        if (
            prompt["base_context_tokens"],
            prompt["suffix_tokens"],
        )
        == REQUIRED_PREFILL_CELL
    ]
    if [prompt["repeat"] for prompt in required_prompts] != [1, 2]:
        raise ProfileReleaseError(f"{profile}/{mode} prefill repeats differ")
    return {
        "deployment": deployed,
        "blended": decoded,
        "prefill": prefetched,
        "blended_run": blended_run,
        "prefill_run": prefill_run,
        "verify_tps": verify_tps,
        "verify_curve": verify_curve,
    }


def validate(
    *,
    serving_path: Path,
    arm_specs: list[tuple[str, str, Path, Path, Path]],
) -> dict[str, Any]:
    expected = {(profile, mode) for profile in PROFILES for mode in MODES}
    keys = [(profile, mode) for profile, mode, *_ in arm_specs]
    if len(keys) != len(expected) or set(keys) != expected or len(set(keys)) != len(keys):
        raise ProfileReleaseError(
            "profile release requires exactly balanced/long/accuracy DFlash2 arms"
        )
    all_paths = [path.resolve() for _, _, *paths in arm_specs for path in paths]
    if len(set(all_paths)) != len(all_paths):
        raise ProfileReleaseError("profile release evidence paths must be distinct")
    try:
        serving_file, serving = signed_serving(serving_path)
    except Exception as error:
        raise ProfileReleaseError("serving qualification is invalid") from error
    runtime = serving["runtime"]
    arms = {
        (profile, mode): _validate_arm(
            profile=profile,
            mode=mode,
            deployment_path=deployment_path,
            blended_path=blended_path,
            prefill_path=prefill_path,
        )
        for profile, mode, deployment_path, blended_path, prefill_path in arm_specs
    }

    reference = arms[("balanced", MODE_DFLASH2)]
    shared_fields = {
        "model_revision": runtime["model_revision"],
        "slot": reference["deployment"]["slot"],
        "power_limit_w": runtime["power_limit_w"],
        "engine_identity": runtime["engine_identity"],
        "sparkinfer_revision": runtime["sparkinfer_revision"],
        "coordinator_slot": runtime["coordinator_slot_fingerprint"],
        "expert_slot": runtime["expert_slot_fingerprint"],
    }
    selected_dflash_settings = runtime["speculation_settings"][MODE_DFLASH2]
    for (profile, mode), arm in arms.items():
        deployed = arm["deployment"]
        actual = {
            "model_revision": deployed["model_revision"],
            "slot": deployed["slot"],
            "power_limit_w": deployed["power_limit_w"],
            "engine_identity": deployed["engine_identity"],
            "sparkinfer_revision": deployed["sparkinfer_revision"],
            "coordinator_slot": deployed["fingerprints"]["coordinator_slot"],
            "expert_slot": deployed["fingerprints"]["expert_slot"],
        }
        if actual != shared_fields:
            raise ProfileReleaseError(f"{profile}/{mode} runtime identity differs")
        if deployed["speculation_settings"] != selected_dflash_settings:
            raise ProfileReleaseError(f"{profile}/{mode} speculation settings differ")
        if profile == "balanced" and (
            deployed["launch_started_ns"]
            != runtime["launch_started_ns"][mode]
            or deployed["fingerprints"]["expert_runtime"]
            != runtime["expert_runtime_fingerprints"][mode]
            or deployed["identity"] != serving["evidence"][f"{mode}_deployment"]
            or arm["blended"]["identity"]
            != serving["evidence"][f"{mode}_blended"]
        ):
            raise ProfileReleaseError(
                f"balanced/{mode} is not the accepted serving arm"
            )

    first = arms[("balanced", MODE_DFLASH2)]
    for (profile, mode), arm in arms.items():
        if (
            arm["blended"]["prompt_contract"]
            != first["blended"]["prompt_contract"]
            or arm["blended"]["prompts"] != first["blended"]["prompts"]
            or arm["prefill"]["corpus_sha256"]
            != first["prefill"]["corpus_sha256"]
            or arm["prefill"]["tokenizer_sha256"]
            != first["prefill"]["tokenizer_sha256"]
            or arm["prefill"]["prompts"] != first["prefill"]["prompts"]
            or arm["blended_run"]["run_id"]
            != first["blended_run"]["run_id"]
            or arm["prefill_run"]["run_id"] != first["prefill_run"]["run_id"]
        ):
            raise ProfileReleaseError(
                f"{profile}/{mode} does not replay the same prompts"
            )

    results: dict[str, Any] = {}
    for profile in PROFILES:
        results[profile] = {}
        for mode in MODES:
            arm = arms[(profile, mode)]
            results[profile][mode] = {
                "weighted_decode_tps": arm["blended"]["wall_decode_tps"],
                "median_replay_decode_tps": arm["blended"][
                    "median_repeat_wall_decode_tps"
                ],
                "accepted_draft_rate": arm["blended"]["accepted_draft_rate"],
                "verify_tokens_per_second": arm["verify_tps"],
                "cached_2k_plus_8k_prefill_tps": arm["prefill"]["cells"][
                    REQUIRED_PREFILL_CELL
                ],
            }
    profile_retention = {
        mode: {
            profile: {
                metric: ratio(
                    results[profile][mode][metric],
                    results["balanced"][mode][metric],
                    f"{mode} {profile} {metric}",
                )
                for metric in (
                    "weighted_decode_tps",
                    "verify_tokens_per_second",
                    "cached_2k_plus_8k_prefill_tps",
                )
            }
            for profile in PROFILES
        }
        for mode in MODES
    }
    evidence = {
        "serving": evidence_identity(serving_file, serving["schema"]),
        "arms": {
            profile: {
                mode: {
                    "deployment": arms[(profile, mode)]["deployment"]["identity"],
                    "blended": arms[(profile, mode)]["blended"]["identity"],
                    "prefill": arms[(profile, mode)]["prefill"]["identity"],
                }
                for mode in MODES
            }
            for profile in PROFILES
        },
    }
    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model_id": GLM53_MODEL_ID,
        "model_revision": runtime["model_revision"],
        "runtime": shared_fields,
        "speculation_settings": {
            MODE_DFLASH2: runtime["speculation_settings"][MODE_DFLASH2]
        },
        "prompt_contracts": {
            "eight_type": first["blended"]["contract"],
            "prefill": first["prefill"]["contract"],
        },
        "results": results,
        "profile_retention": profile_retention,
        "evidence": evidence,
    }
    return body | {"report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ProfileReleaseError(f"refusing to overwrite output: {destination}")
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
    parser.add_argument(
        "--arm",
        nargs=5,
        action="append",
        metavar=("PROFILE", "MODE", "DEPLOYMENT", "BLENDED", "PREFILL"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arm_specs = [
        (profile, mode, Path(deployed), Path(decoded), Path(prefilled))
        for profile, mode, deployed, decoded, prefilled in args.arm
    ]
    report = validate(serving_path=args.serving, arm_specs=arm_specs)
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
