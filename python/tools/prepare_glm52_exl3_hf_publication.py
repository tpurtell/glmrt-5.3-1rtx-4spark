#!/usr/bin/env python3
"""Prepare a compact, standard-only GLM-5 EXL3 Hub publication tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from stage_glm52_exl3_hf_snapshot import (
    MODEL_ID,
    _quant_evidence,
    _validation_evidence,
)
from validate_glm52_exl3_artifact import (
    ARTIFACT_SCHEMA,
    ArtifactContract,
    SHA256_RE,
    _artifact_contract,
    _compact_exl3_declaration,
    _json_object,
    _module_contract,
    _validate_quantization_config,
)
from validate_glm52_exl3_serving_qualification import (
    QualificationError,
    REQUIRED_GATES as REQUIRED_SERVING_GATES,
    revalidate_native_evidence,
)
from validate_glm53_exl3_serving_qualification import (
    REQUIRED_GATES as GLM53_REQUIRED_SERVING_GATES,
    revalidate_dflash2_fusion_evidence,
    revalidate_dflash2_topk_evidence,
    revalidate_dflash2_width_evidence,
    revalidate_native_evidence as revalidate_glm53_native_evidence,
)

SCHEMA = "glmrt-hf-standard-publication-v3"
SERVING_QUALIFICATION_SCHEMA = "glmrt-glm52-exl3-serving-qualification-v1"
GLM53_SERVING_QUALIFICATION_SCHEMA = "glmrt-glm5-exl3-serving-qualification-v1"
SHARD_RE = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors\Z")
REVISION_RE = re.compile(r"[0-9a-f]{40,64}\Z")
PUBLIC_METADATA = (
    ".gitattributes",
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "quantize_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
PENDING_MARKERS = ("GLMRT_PUBLICATION_RESULTS_PENDING", "TODO_PUBLICATION")
HUB_LFS_ATTRIBUTE_PATHS = (
    "model.safetensors.index.json",
    "quantize_config.json",
    "tokenizer.json",
)
MAX_PUBLIC_CONFIG_BYTES = 128 * 1024
PRIVATE_EXECUTION_META = {
    "offload_to_disk",
    "offload_to_disk_path",
    "pack_impl",
    "gc_mode",
    "wait_for_submodule_finalizers",
    "auto_forward_data_parallel",
    "dense_vram_strategy",
    "dense_vram_strategy_devices",
    "moe_vram_strategy",
    "moe_vram_strategy_devices",
    "weight_only",
}


class PublicationError(RuntimeError):
    """The accepted local artifact cannot form a standard public model."""


def _serving_qualification(
    report_path: Path,
    *,
    artifact: Path,
    artifact_manifest_sha256: str,
    plan_sha256: str,
    validation_sha256: str,
    quant_evidence_sha256: str,
    projection_checkpoint_root: Path,
    contract: ArtifactContract | None = None,
) -> dict[str, Any]:
    expected_model_id = MODEL_ID if contract is None else contract.model_id
    expected_schema = (
        SERVING_QUALIFICATION_SCHEMA
        if contract is None or contract.exl3_bits == 3
        else GLM53_SERVING_QUALIFICATION_SCHEMA
    )
    is_glm53 = contract is not None and contract.exl3_bits == 4
    expected_gates = (
        GLM53_REQUIRED_SERVING_GATES if is_glm53 else REQUIRED_SERVING_GATES
    )
    resolved = report_path.expanduser().resolve(strict=True)
    if report_path.expanduser().is_symlink() or not resolved.is_file():
        raise PublicationError("serving-qualification report is not one regular file")
    report = _json_object(resolved)
    report_digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    runtime = report.get("runtime")
    gates = report.get("gates")
    valid_speculation = isinstance(runtime, dict) and (
        runtime.get("speculation") == "dspark"
        if not is_glm53
        else (
            runtime.get("speculation") in {"mtp", "dflash2"}
            and runtime.get("default_speculation") == runtime.get("speculation")
            and runtime.get("qualified_speculation") == ["mtp", "dflash2"]
        )
    )
    artifact_validation = report.get("artifact_validation")
    quant_evidence = report.get("quant_evidence")
    if (
        report.get("schema") != expected_schema
        or report.get("status") != "accepted"
        or report.get("model_id") != expected_model_id
        or Path(str(report.get("artifact", ""))).expanduser().resolve() != artifact
        or report.get("artifact_manifest_sha256") != artifact_manifest_sha256
        or report.get("plan_sha256") != plan_sha256
        or not isinstance(report_digest, str)
        or hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        != report_digest
        or not isinstance(artifact_validation, dict)
        or artifact_validation.get("sha256") != validation_sha256
        or not isinstance(quant_evidence, dict)
        or quant_evidence.get("sha256") != quant_evidence_sha256
        or not isinstance(runtime, dict)
        or not isinstance(runtime.get("engine_identity"), str)
        or not runtime["engine_identity"]
        or REVISION_RE.fullmatch(str(runtime.get("sparkinfer_revision", ""))) is None
        or SHA256_RE.fullmatch(str(runtime.get("coordinator_slot_fingerprint", "")))
        is None
        or SHA256_RE.fullmatch(str(runtime.get("expert_slot_fingerprint", ""))) is None
        or runtime.get("profile") not in {"balanced", "long", "accuracy"}
        or not valid_speculation
        or isinstance(runtime.get("power_limit_w"), bool)
        or not isinstance(runtime.get("power_limit_w"), int)
        or runtime["power_limit_w"] <= 0
        or not isinstance(gates, dict)
        or set(gates) != expected_gates
        or any(value is not True for value in gates.values())
        or report.get("failed_gates") != []
    ):
        raise PublicationError(
            "serving-qualification report does not accept this exact artifact"
        )
    try:
        revalidator = (
            revalidate_glm53_native_evidence if is_glm53 else revalidate_native_evidence
        )
        revalidator(
            report,
            expected_sparkinfer_revision=runtime["sparkinfer_revision"],
            expected_checkpoint_root=(
                artifact if is_glm53 else projection_checkpoint_root
            ),
            expected_expert_slot_fingerprint=runtime[
                "expert_slot_fingerprint"
            ],
        )
        if is_glm53:
            revalidate_dflash2_fusion_evidence(report)
            revalidate_dflash2_topk_evidence(report)
            revalidate_dflash2_width_evidence(report)
    except QualificationError as error:
        raise PublicationError(
            "serving-qualification report has unverifiable native EXL3 or DFlash2 fusion/top-k/width evidence"
        ) from error
    return {
        "path": os.fspath(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _hash_file(resolved),
        "schema": expected_schema,
        "report_sha256": report_digest,
    }


def _validated_source_snapshot(
    validation: dict[str, Any], requested_source: Path
) -> None:
    raw = validation.get("source_snapshot")
    if not isinstance(raw, str) or not raw:
        raise PublicationError("validation report has no source-snapshot identity")
    if Path(raw).expanduser().resolve() != requested_source:
        raise PublicationError(
            "publication source snapshot differs from artifact validation"
        )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes(path: Path, value: bytes) -> None:
    with path.open("xb") as target:
        target.write(value)
        target.flush()
        os.fsync(target.fileno())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _verify_artifact_metadata_hashes(
    artifact: Path,
    artifact_records: dict[str, Any],
) -> None:
    """Rebind mutable publication metadata to the accepted artifact manifest."""

    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = artifact / name
        record = artifact_records.get(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
            or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
            or path.stat().st_size != record["bytes"]
            or _hash_file(path) != record["sha256"]
        ):
            raise PublicationError(
                f"publication metadata differs from the accepted artifact manifest: {name}"
            )


def _referenced_shards(artifact: Path) -> tuple[str, ...]:
    index = _json_object(artifact / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise PublicationError("model index has no weight_map")
    shards: set[str] = set()
    for name, raw in weight_map.items():
        if not isinstance(name, str) or not name or not isinstance(raw, str):
            raise PublicationError("model index contains an invalid tensor mapping")
        path = PurePosixPath(raw)
        if path.name != raw or SHARD_RE.fullmatch(raw) is None:
            raise PublicationError(
                f"model index uses a nonstandard shard path: {raw!r}"
            )
        shards.add(raw)
    metadata = index.get("metadata")
    total_size = metadata.get("total_size") if isinstance(metadata, dict) else None
    if (
        isinstance(total_size, bool)
        or not isinstance(total_size, int)
        or total_size <= 0
    ):
        raise PublicationError("model index has no positive metadata.total_size")
    return tuple(sorted(shards))


def _public_configs(
    artifact: Path, contract: ArtifactContract | None = None
) -> tuple[bytes, bytes]:
    if contract is not None:
        _validate_quantization_config(artifact, _module_contract(), contract)
    config = _json_object(artifact / "config.json")
    embedded = config.get("quantization_config")
    external = _json_object(artifact / "quantize_config.json")
    if not isinstance(embedded, dict) or external.get("quant_method") != "exl3":
        raise PublicationError("artifact does not declare EXL3 quantization")
    storage = external.get("tensor_storage")
    if not isinstance(storage, dict) or not storage:
        raise PublicationError("external EXL3 config has no tensor_storage")
    external_declaration = dict(external)
    external_declaration.pop("tensor_storage")
    minimal_declaration = _compact_exl3_declaration(external)
    is_glm53 = (
        contract is not None and contract.exl3_bits == 4
    ) or external.get("bits") == 4.0
    accepted_embedded = (
        (minimal_declaration,)
        if is_glm53
        else (minimal_declaration, external_declaration)
    )
    if embedded not in accepted_embedded:
        raise PublicationError("embedded and external EXL3 configurations conflict")

    public_external = dict(external)
    meta = dict(public_external.get("meta", {}))
    for field in PRIVATE_EXECUTION_META:
        meta.pop(field, None)
    public_external["meta"] = meta
    public_declaration = dict(public_external)
    public_declaration.pop("tensor_storage")
    config["quantization_config"] = (
        _compact_exl3_declaration(public_external)
        if is_glm53 or embedded == minimal_declaration
        else public_declaration
    )
    rendered_config = (
        json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    rendered_external = (
        json.dumps(public_external, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    if public_external.get("tensor_storage") != storage:
        raise PublicationError(
            "public standalone EXL3 tensor_storage differs from the artifact"
        )
    if len(rendered_config) > MAX_PUBLIC_CONFIG_BYTES:
        raise PublicationError("compact public config.json exceeds 128 KiB")
    return rendered_config, rendered_external


def _public_gitattributes(source_snapshot: Path) -> bytes:
    source = source_snapshot / ".gitattributes"
    if not source.is_file():
        raise PublicationError("publication source is missing: .gitattributes")
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise PublicationError("source .gitattributes is not UTF-8") from error
    for path in HUB_LFS_ATTRIBUTE_PATHS:
        rule = f"{path} filter=lfs diff=lfs merge=lfs -text"
        if rule not in lines:
            lines.append(rule)
    return ("\n".join(lines) + "\n").encode()


def _source_files(
    artifact: Path,
    source_snapshot: Path,
    readme: Path,
    serving_report_sha256: str,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    shards = _referenced_shards(artifact)
    sources = {
        "LICENSE": source_snapshot / "LICENSE",
        "README.md": readme,
        "chat_template.jinja": source_snapshot / "chat_template.jinja",
        "generation_config.json": artifact / "generation_config.json",
        "model.safetensors.index.json": artifact / "model.safetensors.index.json",
        "tokenizer.json": artifact / "tokenizer.json",
        "tokenizer_config.json": artifact / "tokenizer_config.json",
    }
    sources.update({name: artifact / name for name in shards})
    for name, source in sources.items():
        if not source.is_file():
            raise PublicationError(f"publication source is missing: {name} ({source})")
    readme_text = readme.read_text(encoding="utf-8")
    if not readme_text.startswith("---\n") or any(
        marker in readme_text for marker in PENDING_MARKERS
    ):
        raise PublicationError("README is not a completed Hugging Face model card")
    if f"Qualification evidence SHA-256: `{serving_report_sha256}`" not in readme_text:
        raise PublicationError(
            "README is not bound to the serving qualification report"
        )
    return sources, shards


def prepare(
    artifact_path: Path,
    source_snapshot_path: Path,
    validation_report_path: Path,
    quant_evidence_report_path: Path,
    serving_qualification_report_path: Path,
    readme_path: Path,
    output_path: Path,
    *,
    link_mode: str,
) -> dict[str, Any]:
    artifact = artifact_path.expanduser().resolve(strict=True)
    source_snapshot = source_snapshot_path.expanduser().resolve(strict=True)
    readme = readme_path.expanduser().resolve(strict=True)
    output = output_path.expanduser().resolve()
    if output.exists() or output.is_symlink():
        raise PublicationError(f"publication output already exists: {output}")
    artifact_manifest = _json_object(artifact / "glmrt-gptqmodel-artifact.json")
    artifact_records = artifact_manifest.get("files")
    if not isinstance(artifact_records, dict):
        raise PublicationError("artifact manifest has no file inventory")
    contract = _artifact_contract(_json_object(artifact / "glmrt-gptqmodel-plan.json"))
    if artifact_manifest.get("schema") != contract.artifact_schema:
        raise PublicationError("source is not a completed GLMRT artifact")
    qualification, validation = _validation_evidence(
        validation_report_path,
        artifact=artifact,
        artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
        contract=contract,
    )
    _validated_source_snapshot(validation, source_snapshot)
    if SHA256_RE.fullmatch(str(validation.get("plan_sha256", ""))) is None:
        raise PublicationError(
            "validation report has no valid quantization plan identity"
        )
    quant_qualification, quant_report = _quant_evidence(
        quant_evidence_report_path,
        plan_sha256=validation["plan_sha256"],
        contract=contract,
    )
    if validation["projection_checkpoint"][
        "checkpoint_inventory_sha256"
    ] != quant_report["integrity"]["checkpoint_inventory_sha256"] or validation.get(
        "execution_upgrade_sha256"
    ) != (
        quant_report.get("execution_upgrade") or {}
    ).get(
        "active_upgrade_sha256"
    ):
        raise PublicationError(
            "artifact and quant evidence bind different projection or execution identities"
        )
    serving_qualification = _serving_qualification(
        serving_qualification_report_path,
        artifact=artifact,
        artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
        plan_sha256=validation["plan_sha256"],
        validation_sha256=qualification["sha256"],
        quant_evidence_sha256=quant_qualification["sha256"],
        projection_checkpoint_root=Path(validation["projection_checkpoint"]["root"])
        .expanduser()
        .resolve(),
        contract=contract,
    )
    _verify_artifact_metadata_hashes(artifact, artifact_records)
    sources, shards = _source_files(
        artifact,
        source_snapshot,
        readme,
        serving_qualification["report_sha256"],
    )
    compact_config, external_config = _public_configs(artifact, contract)
    gitattributes = _public_gitattributes(source_snapshot)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, source in sorted(sources.items()):
            destination = temporary / name
            if name.endswith(".safetensors") and link_mode == "hardlink":
                os.link(source.resolve(strict=True), destination, follow_symlinks=False)
            elif link_mode in {"hardlink", "copy"}:
                shutil.copy2(
                    source.resolve(strict=True), destination, follow_symlinks=False
                )
            else:
                raise PublicationError(f"unsupported link mode: {link_mode}")
            _fsync_file(destination)
        _write_bytes(temporary / "config.json", compact_config)
        _write_bytes(temporary / "quantize_config.json", external_config)
        _write_bytes(temporary / ".gitattributes", gitattributes)
        _fsync_directory(temporary)
        os.replace(temporary, output)
        _fsync_directory(output.parent)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    expected_names = set(PUBLIC_METADATA) | set(shards)
    actual_names = {path.name for path in output.iterdir()}
    if actual_names != expected_names:
        raise PublicationError(
            f"public file set differs: missing={sorted(expected_names - actual_names)} "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    entries: list[dict[str, Any]] = []
    for name in sorted(actual_names):
        path = output / name
        if path.is_symlink() or not path.is_file():
            raise PublicationError(f"public entry is not a regular file: {name}")
        if name in shards and link_mode == "hardlink":
            record = artifact_records.get(name)
            if (
                not isinstance(record, dict)
                or path.stat().st_size != record.get("bytes")
                or path.stat().st_ino != (artifact / name).stat().st_ino
            ):
                raise PublicationError(
                    f"public shard does not match the artifact: {name}"
                )
            digest = record["sha256"]
        else:
            digest = _hash_file(path)
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": digest})
    body = {
        "schema": SCHEMA,
        "model_id": contract.model_id,
        "source_artifact_manifest_sha256": artifact_manifest["manifest_sha256"],
        "source_validation_sha256": qualification["sha256"],
        "source_quant_evidence_sha256": quant_qualification["sha256"],
        "source_serving_qualification_sha256": serving_qualification["sha256"],
        "plan_sha256": validation["plan_sha256"],
        "files": entries,
    }
    return {
        **body,
        "publication_sha256": hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "status": "ready",
        "artifact": os.fspath(artifact),
        "source_snapshot": os.fspath(source_snapshot),
        "output": os.fspath(output),
        "link_mode": link_mode,
        "shards": len(shards),
        "file_bytes": sum(entry["bytes"] for entry in entries),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--quant-evidence-report", type=Path, required=True)
    parser.add_argument("--serving-qualification-report", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    args = parser.parse_args()
    report = prepare(
        args.artifact,
        args.source_snapshot,
        args.validation_report,
        args.quant_evidence_report,
        args.serving_qualification_report,
        args.readme,
        args.output,
        link_mode=args.link_mode,
    )
    _atomic_json(args.report.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
