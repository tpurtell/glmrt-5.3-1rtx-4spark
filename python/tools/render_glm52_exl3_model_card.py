#!/usr/bin/env python3
"""Render the GLM-5.2 EXL3 model-card qualification from accepted evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage_glm52_exl3_hf_snapshot import MODEL_ID  # noqa: E402
from validate_glm52_exl3_serving_qualification import (  # noqa: E402
    QualificationError,
    REQUIRED_GATES,
    revalidate_native_evidence,
)


MARKER = "GLMRT_PUBLICATION_RESULTS_PENDING"
ARTIFACT_SCHEMA = "glmrt-glm52-exl3-artifact-validation-v5"
QUANT_SCHEMA = "glmrt-glm52-exl3-quant-evidence-validation-v2"
SERVING_SCHEMA = "glmrt-glm52-exl3-serving-qualification-v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ModelCardError(RuntimeError):
    """Qualification evidence cannot produce an accepted model card."""


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


def signed_report(path: Path, schema: str) -> tuple[Path, dict[str, Any]]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ModelCardError(f"evidence is a symbolic link: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ModelCardError(f"evidence is not one regular file: {resolved}")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ModelCardError(f"evidence is not valid JSON: {resolved}") from error
    if not isinstance(report, dict):
        raise ModelCardError(f"evidence root is not an object: {resolved}")
    digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    if (
        report.get("schema") != schema
        or report.get("status") != "accepted"
        or not isinstance(digest, str)
        or hashlib.sha256(canonical_json(body)).hexdigest() != digest
    ):
        raise ModelCardError(f"evidence is not an accepted signed {schema} report")
    return resolved, report


def integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelCardError(f"{label} is not a nonnegative integer")
    return value


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelCardError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ModelCardError(f"{label} is not finite and nonnegative")
    return result


def gib(value: int) -> str:
    return f"{value / (1024**3):,.2f} GiB"


def render_section(
    *,
    artifact: dict[str, Any],
    quant: dict[str, Any],
    serving: dict[str, Any],
    hub_revision: str | None,
) -> str:
    coverage = quant.get("coverage")
    integrity = quant.get("integrity")
    metrics = quant.get("metrics")
    global_metrics = metrics.get("global") if isinstance(metrics, dict) else None
    results = serving.get("results")
    runtime = serving.get("runtime")
    thresholds = serving.get("thresholds")
    if (
        not isinstance(coverage, dict)
        or coverage.get("projection_count") != 57_600
        or coverage.get("complete_expert_count") != 75 * 256
        or not isinstance(integrity, dict)
        or integrity.get("tensor_payload_hashes_verified") is not True
        or not isinstance(global_metrics, dict)
        or not isinstance(results, dict)
        or not isinstance(runtime, dict)
        or not isinstance(thresholds, dict)
    ):
        raise ModelCardError("qualification reports do not prove complete EXL3 coverage")
    if (
        artifact.get("retained_native_bytes_verified") is not True
        or artifact.get("artifact_manifest_file_hashes_verified") is not True
        or artifact.get("projection_checkpoint_bytes_verified") is not True
        or not isinstance(artifact.get("projection_checkpoint"), dict)
        or SHA256_RE.fullmatch(
            str(
                artifact.get("projection_checkpoint", {}).get(
                    "checkpoint_inventory_sha256", ""
                )
            )
        )
        is None
        or artifact["projection_checkpoint"]["checkpoint_inventory_sha256"]
        != integrity.get("checkpoint_inventory_sha256")
        or integer(artifact.get("quantized_modules"), "quantized modules") != 57_600
        or integer(artifact.get("exl3_tensors"), "EXL3 tensors") != 230_400
    ):
        raise ModelCardError("artifact report does not prove the final tensor replacement")
    blended = results.get("blended")
    repeat = results.get("repeat")
    prefill = results.get("prefill")
    tools = results.get("tool_eval")
    startup = results.get("expert_startup")
    native = results.get("native_kernel")
    if not all(
        isinstance(value, dict)
        for value in (blended, repeat, prefill, tools, startup, native)
    ):
        raise ModelCardError("serving report has incomplete result groups")
    if (
        native.get("trellis_bits") != 3
        or native.get("expert_slot_fingerprint")
        != serving.get("runtime", {}).get("expert_slot_fingerprint")
    ):
        raise ModelCardError("serving report native evidence is not EXL3 K3")
    if blended.get("candidate_all_quality_contracts_passed") is not True:
        raise ModelCardError("serving report does not prove candidate semantic contracts")
    semantic_cases = integer(blended.get("cases"), "semantic contract cases")
    acceptance_floor = number(
        thresholds.get("minimum_blended_acceptance_ratio"),
        "acceptance qualification floor",
    )
    prefill_floor = number(
        thresholds.get("minimum_per_cell_prefill_ratio"),
        "prefill qualification floor",
    )
    tradeoff_line = (
        "- Policy: explicit decode-optimized tradeoff; the ordinary 0.950x "
        "acceptance and per-cell prefill floors were not used"
        if acceptance_floor < 0.95 or prefill_floor < 0.95
        else "- Policy: ordinary release floors"
    )
    cells = prefill.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ModelCardError("serving report has no prefill cells")
    prefill_lines = [
        "| Context | Prefill rows | NVFP4 tok/s | EXL3 tok/s | Ratio |",
        "|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        if not isinstance(cell, dict):
            raise ModelCardError("serving report contains an invalid prefill cell")
        prefill_lines.append(
            "| {context:,} | {rows:,} | {baseline:,.1f} | {candidate:,.1f} | {ratio:.3f}x |".format(
                context=integer(cell.get("base_context_tokens"), "prefill context"),
                rows=integer(cell.get("suffix_tokens"), "prefill rows"),
                baseline=number(cell.get("baseline_tps"), "baseline prefill TPS"),
                candidate=number(cell.get("candidate_tps"), "candidate prefill TPS"),
                ratio=number(cell.get("ratio"), "prefill ratio"),
            )
        )
    hub_line = (
        f"- Verified Hub revision: `{hub_revision}`"
        if hub_revision is not None
        else "- Hub revision: assigned and verified immediately after the initial upload"
    )
    execution_upgrade_sha256 = artifact.get("execution_upgrade_sha256")
    quant_execution_upgrade = quant.get("execution_upgrade")
    quant_upgrade_sha256 = (
        quant_execution_upgrade.get("active_upgrade_sha256")
        if isinstance(quant_execution_upgrade, dict)
        else None
    )
    if execution_upgrade_sha256 != quant_upgrade_sha256 or (
        execution_upgrade_sha256 is not None
        and SHA256_RE.fullmatch(str(execution_upgrade_sha256)) is None
    ):
        raise ModelCardError(
            "artifact and quantizer reports bind different execution upgrades"
        )
    execution_upgrade_line = (
        f"- Quantizer execution upgrade SHA-256: `{execution_upgrade_sha256}`"
        if execution_upgrade_sha256 is not None
        else "- Quantizer execution upgrade: none"
    )
    return "\n".join(
        [
            "The exact artifact passed the content-bound GLMRT qualification gate.",
            "Both arms used identical coordinator and Spark binaries, the balanced",
            f"profile, dSpark speculation, and a {integer(runtime.get('power_limit_w'), 'power limit')} W coordinator power limit.",
            "",
            "### Structural and quantizer evidence",
            "",
            f"- Routed projections quantized: {integer(artifact.get('quantized_modules'), 'quantized modules'):,}",
            f"- EXL3 tensors: {integer(artifact.get('exl3_tensors'), 'EXL3 tensors'):,}",
            f"- Retained native tensors byte-compared: {integer(artifact.get('retained_native_tensors'), 'retained tensors'):,}",
            f"- EXL3 tensor payload: {gib(integer(artifact.get('exl3_tensor_bytes'), 'EXL3 bytes'))}",
            f"- TP4 resident payload per Spark: {gib(integer(artifact.get('tp4_resident_bytes_per_spark'), 'resident bytes'))}",
            "- Aggregate Hessian-weighted relative projection error: "
            f"{number(global_metrics.get('aggregate_hessian_weighted_relative_error'), 'aggregate quantization error'):.8g}",
            "- End-to-end serving decision: accepted",
            "",
            "### Decode and acceptance",
            "",
            "| Workload | NVFP4 | EXL3 | Ratio |",
            "|---|---:|---:|---:|",
            "| Weighted decode | {baseline:.3f} tok/s | {candidate:.3f} tok/s | {ratio:.3f}x |".format(
                baseline=number(blended.get("baseline_wall_decode_tps"), "baseline decode TPS"),
                candidate=number(blended.get("candidate_wall_decode_tps"), "candidate decode TPS"),
                ratio=number(blended.get("decode_ratio"), "decode ratio"),
            ),
            "| Orchid repeat | {baseline:.3f} tok/s | {candidate:.3f} tok/s | {ratio:.3f}x |".format(
                baseline=number(repeat.get("baseline_decode_tps"), "baseline repeat TPS"),
                candidate=number(repeat.get("candidate_decode_tps"), "candidate repeat TPS"),
                ratio=number(repeat.get("decode_ratio"), "repeat ratio"),
            ),
            "| dSpark accepted drafts | {baseline:.2%} | {candidate:.2%} | {ratio:.3f}x |".format(
                baseline=number(blended.get("baseline_accepted_draft_rate"), "baseline acceptance"),
                candidate=number(blended.get("candidate_accepted_draft_rate"), "candidate acceptance"),
                ratio=number(blended.get("acceptance_ratio"), "acceptance ratio"),
            ),
            "",
            tradeoff_line,
            f"- Selected acceptance floor: {acceptance_floor:.3f}x of NVFP4",
            f"- Candidate semantic contracts: {semantic_cases}/{semantic_cases} passed",
            "",
            "### Prefill",
            "",
            *prefill_lines,
            "",
            f"Minimum prefill cell ratio: {number(prefill.get('minimum_cell_ratio'), 'minimum prefill ratio'):.3f}x.",
            f"Selected per-cell prefill floor: {prefill_floor:.3f}x of NVFP4.",
            "",
            "### Tool use and startup",
            "",
            "- Tool-call score: {candidate}/{maximum} points (NVFP4: {baseline}/{maximum}; {ratio:.3f}x)".format(
                candidate=integer(tools.get("candidate_points"), "candidate tool points"),
                baseline=integer(tools.get("baseline_points"), "baseline tool points"),
                maximum=integer(tools.get("maximum_points"), "maximum tool points"),
                ratio=number(tools.get("points_ratio"), "tool points ratio"),
            ),
            "- Expert resident preload: {candidate:,.1f} ms (NVFP4: {baseline:,.1f} ms; {ratio:.3f}x)".format(
                candidate=number(startup.get("candidate_maximum_resident_preload_ms"), "candidate resident preload"),
                baseline=number(startup.get("baseline_maximum_resident_preload_ms"), "baseline resident preload"),
                ratio=number(startup.get("resident_preload_ratio"), "resident preload ratio"),
            ),
            "- Full expert service handoff: {candidate:,.1f} ms (NVFP4: {baseline:,.1f} ms; {ratio:.3f}x)".format(
                candidate=number(startup.get("candidate_maximum_service_handoff_total_ms"), "candidate startup"),
                baseline=number(startup.get("baseline_maximum_service_handoff_total_ms"), "baseline startup"),
                ratio=number(startup.get("startup_ratio"), "startup ratio"),
            ),
            "- Native EXL3 parity: TP ranks {ranks}; calibrated layer {layer}; rows {rows}".format(
                ranks=", ".join(
                    str(integer(rank, "native TP rank"))
                    for rank in native.get("tp_ranks", [])
                ),
                layer=integer(native.get("layer_id"), "native calibrated layer"),
                rows=", ".join(
                    str(integer(row, "native validation rows"))
                    for row in native.get("required_rows", [])
                ),
            ),
            "",
            "### Reproducibility identity",
            "",
            f"- GLMRT WIP engine: `{runtime.get('engine_identity')}`",
            f"- SparkInfer revision: `{runtime.get('sparkinfer_revision')}`",
            f"- Coordinator slot SHA-256: `{runtime.get('coordinator_slot_fingerprint')}`",
            f"- Spark slot SHA-256: `{runtime.get('expert_slot_fingerprint')}`",
            execution_upgrade_line,
            f"- Qualification evidence SHA-256: `{serving.get('report_sha256')}`",
            hub_line,
        ]
    )


def render(
    *,
    template_path: Path,
    artifact_validation_path: Path,
    quant_evidence_path: Path,
    serving_qualification_path: Path,
    hub_revision: str | None,
) -> str:
    template_file = template_path.expanduser()
    if template_file.is_symlink():
        raise ModelCardError("model-card template is a symbolic link")
    template_file = template_file.resolve(strict=True)
    template = template_file.read_text(encoding="utf-8")
    if template.count(MARKER) != 1:
        raise ModelCardError("model-card template must contain exactly one pending marker")
    artifact_path, artifact = signed_report(artifact_validation_path, ARTIFACT_SCHEMA)
    quant_path, quant = signed_report(quant_evidence_path, QUANT_SCHEMA)
    _serving_path, serving = signed_report(serving_qualification_path, SERVING_SCHEMA)
    gates = serving.get("gates")
    runtime = serving.get("runtime")
    if (
        artifact.get("model_id") != MODEL_ID
        or serving.get("model_id") != MODEL_ID
        or serving.get("artifact_validation", {}).get("sha256")
        != hash_file(artifact_path)
        or serving.get("quant_evidence", {}).get("sha256") != hash_file(quant_path)
        or serving.get("artifact_manifest_sha256")
        != artifact.get("artifact_manifest_sha256")
        or serving.get("plan_sha256") != artifact.get("plan_sha256")
        or quant.get("plan", {}).get("plan_sha256") != artifact.get("plan_sha256")
        or not isinstance(runtime, dict)
        or runtime.get("profile") != "balanced"
        or runtime.get("speculation") != "dspark"
        or REVISION_RE.fullmatch(str(runtime.get("sparkinfer_revision", ""))) is None
        or SHA256_RE.fullmatch(
            str(runtime.get("coordinator_slot_fingerprint", ""))
        )
        is None
        or SHA256_RE.fullmatch(str(runtime.get("expert_slot_fingerprint", "")))
        is None
        or not isinstance(gates, dict)
        or set(gates) != REQUIRED_GATES
        or serving.get("failed_gates") != []
        or any(value is not True for value in gates.values())
    ):
        raise ModelCardError("serving report is not bound to the structural evidence")
    try:
        revalidate_native_evidence(
            serving,
            expected_sparkinfer_revision=runtime["sparkinfer_revision"],
            expected_checkpoint_root=Path(
                artifact["projection_checkpoint"]["root"]
            ).expanduser().resolve(),
            expected_expert_slot_fingerprint=runtime[
                "expert_slot_fingerprint"
            ],
        )
    except (QualificationError, KeyError, TypeError) as error:
        raise ModelCardError(
            "serving report does not retain verifiable native EXL3 evidence"
        ) from error
    if hub_revision is not None and REVISION_RE.fullmatch(hub_revision) is None:
        raise ModelCardError("Hub revision must be a lowercase 40-64 digit hex value")
    return template.replace(
        MARKER,
        render_section(
            artifact=artifact,
            quant=quant,
            serving=serving,
            hub_revision=hub_revision,
        ),
    )


def atomic_text(path: Path, value: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            target.write(value)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--artifact-validation", type=Path, required=True)
    parser.add_argument("--quant-evidence", type=Path, required=True)
    parser.add_argument("--serving-qualification", type=Path, required=True)
    parser.add_argument("--hub-revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render(
        template_path=args.template,
        artifact_validation_path=args.artifact_validation,
        quant_evidence_path=args.quant_evidence,
        serving_qualification_path=args.serving_qualification,
        hub_revision=args.hub_revision,
    )
    atomic_text(args.output, rendered)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
