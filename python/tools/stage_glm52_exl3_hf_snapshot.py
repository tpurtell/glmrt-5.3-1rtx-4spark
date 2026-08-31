#!/usr/bin/env python3
"""Stage an accepted GLM-5 EXL3 artifact in the Hugging Face cache.

Hardlink mode creates no second tensor payload on the coordinator.  Snapshot
entries use the normal Hugging Face blob/symlink layout, so GLMRT can qualify
the local artifact before it is uploaded to the Hub.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_glm52_exl3_artifact import (  # noqa: E402
    ArtifactContract,
    ARTIFACT_SCHEMA,
    GLM53_MODEL_ID,
    MODEL_ID,
    SCHEMA as VALIDATION_SCHEMA,
    SHA256_RE,
    _artifact_contract,
    _canonical_json,
    _json_object,
)
from validate_glm52_exl3_quant_evidence import (  # noqa: E402
    EXPECTED_PROJECTIONS,
    GLM53_SCHEMA as GLM53_QUANT_EVIDENCE_SCHEMA,
    SCHEMA as QUANT_EVIDENCE_SCHEMA,
)


SCHEMA = "glmrt-hf-staged-snapshot-v1"
PUBLICATION_SCHEMA = "glmrt-hf-standard-publication-v3"
MODEL_COMPONENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
SUPPORTED_MODEL_IDS = frozenset({MODEL_ID, GLM53_MODEL_ID})


class StagingError(RuntimeError):
    """The accepted artifact cannot be staged safely."""


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _model_cache_root(hf_home: Path, model_id: str) -> Path:
    components = model_id.split("/")
    if len(components) != 2 or any(
        MODEL_COMPONENT_RE.fullmatch(component) is None for component in components
    ):
        raise StagingError("--model-id must be a safe organization/repository pair")
    return hf_home / "hub" / f"models--{components[0]}--{components[1]}"


def _safe_relative(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise StagingError(f"invalid artifact path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise StagingError(f"unsafe artifact path: {value!r}")
    return path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _artifact_entries(
    artifact: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], ArtifactContract]:
    plan = _json_object(artifact / "glmrt-gptqmodel-plan.json")
    contract = _artifact_contract(plan)
    manifest = _json_object(artifact / "glmrt-gptqmodel-artifact.json")
    records = manifest.get("files")
    if manifest.get("schema") != contract.artifact_schema or not isinstance(records, dict):
        raise StagingError("artifact manifest is not a complete GLMRT artifact")
    entries: list[dict[str, Any]] = []
    for relative, record in records.items():
        path = _safe_relative(relative)
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise StagingError(f"invalid artifact manifest record: {relative}")
        source = artifact.joinpath(*path.parts)
        if source.is_symlink() or not source.is_file() or source.stat().st_size != record["bytes"]:
            raise StagingError(f"artifact file differs from its manifest: {relative}")
        entries.append(
            {"path": path.as_posix(), "bytes": record["bytes"], "sha256": record["sha256"]}
        )
    for relative in ("glmrt-gptqmodel-artifact.json", "glmrt-gptqmodel-run.json"):
        path = artifact / relative
        if path.is_symlink() or not path.is_file():
            raise StagingError(f"artifact lacks regular {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _hash_file(path)}
        )
    entries.sort(key=lambda entry: entry["path"])
    actual = {
        path.relative_to(artifact).as_posix()
        for path in artifact.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != {entry["path"] for entry in entries}:
        raise StagingError("artifact file set changed after validation")
    return entries, manifest, contract


def _validation_evidence(
    report_path: Path,
    *,
    artifact: Path,
    artifact_manifest_sha256: str,
    contract: ArtifactContract | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if contract is None:
        contract = _artifact_contract(
            _json_object(artifact / "glmrt-gptqmodel-plan.json")
        )
    report_path = report_path.expanduser().resolve(strict=True)
    if report_path.is_symlink() or not report_path.is_file():
        raise StagingError("validation report is not a regular file")
    report = _json_object(report_path)
    report_digest = report.get("report_sha256")
    report_body = {
        key: value for key, value in report.items() if key != "report_sha256"
    }
    reported_artifact = Path(str(report.get("artifact", ""))).expanduser().resolve()
    tokenizer_evidence = report.get("tokenizer_evidence")
    tokenizer_files = (
        tokenizer_evidence.get("tokenizer_files")
        if isinstance(tokenizer_evidence, dict)
        else None
    )
    tokenizer_mode = (
        tokenizer_evidence.get("mode")
        if isinstance(tokenizer_evidence, dict)
        else None
    )
    valid_tokenizer_files = (
        isinstance(tokenizer_files, list)
        and len(tokenizer_files) == 2
        and {record.get("name") for record in tokenizer_files if isinstance(record, dict)}
        == {"tokenizer.json", "tokenizer_config.json"}
        and all(
            isinstance(record, dict)
            and isinstance(record.get("bytes"), int)
            and record["bytes"] > 0
            and SHA256_RE.fullmatch(str(record.get("sha256", ""))) is not None
            for record in tokenizer_files
        )
    )
    valid_tokenizer_evidence = valid_tokenizer_files and (
        tokenizer_mode == "plan-bound"
        or (
            tokenizer_mode == "legacy-live-container-attestation"
            and SHA256_RE.fullmatch(
                str(tokenizer_evidence.get("file_sha256", ""))
            )
            is not None
            and SHA256_RE.fullmatch(
                str(tokenizer_evidence.get("attestation_sha256", ""))
            )
            is not None
            and SHA256_RE.fullmatch(
                str(tokenizer_evidence.get("prepared_token_stream_sha256", ""))
            )
            is not None
            and isinstance(tokenizer_evidence.get("total_tokens"), int)
            and tokenizer_evidence["total_tokens"] > 0
        )
    )
    source_metadata = report.get("source_metadata")
    source_metadata_by_name = (
        {
            record.get("name"): record
            for record in source_metadata
            if isinstance(record, dict)
        }
        if isinstance(source_metadata, list)
        else {}
    )
    valid_source_metadata = (
        isinstance(source_metadata, list)
        and len(source_metadata) == 3
        and set(source_metadata_by_name)
        == {"tokenizer.json", "tokenizer_config.json", "generation_config.json"}
        and all(
            set(record) == {"name", "bytes", "sha256"}
            and isinstance(record.get("bytes"), int)
            and not isinstance(record.get("bytes"), bool)
            and record["bytes"] > 0
            and SHA256_RE.fullmatch(str(record.get("sha256", ""))) is not None
            for record in source_metadata_by_name.values()
        )
        and all(
            source_metadata_by_name.get(record["name"])
            == {
                "name": record.get("name"),
                "bytes": record.get("bytes"),
                "sha256": record.get("sha256"),
            }
            for record in tokenizer_files or []
            if isinstance(record, dict)
        )
    )
    if contract.model_id != GLM53_MODEL_ID:
        valid_source_metadata = True
    quantization_config = report.get("quantization_config")
    quantize_config_path = artifact / "quantize_config.json"
    valid_quantization_config = (
        isinstance(quantization_config, dict)
        and quantization_config.get("tensor_storage_entries")
        == EXPECTED_PROJECTIONS
        and quantization_config.get("stored_tensor_descriptions")
        == EXPECTED_PROJECTIONS * 4
        and SHA256_RE.fullmatch(
            str(quantization_config.get("sha256", ""))
        )
        is not None
        and SHA256_RE.fullmatch(
            str(quantization_config.get("ledger_provenance_sha256", ""))
        )
        is not None
        and not quantize_config_path.is_symlink()
        and quantize_config_path.is_file()
        and _hash_file(quantize_config_path) == quantization_config["sha256"]
    )
    if contract.model_id != GLM53_MODEL_ID:
        valid_quantization_config = True
    checkpoint = report.get("projection_checkpoint")
    valid_checkpoint = (
        report.get("projection_checkpoint_bytes_verified") is True
        and isinstance(checkpoint, dict)
        and isinstance(checkpoint.get("root"), str)
        and bool(checkpoint["root"])
        and checkpoint.get("projection_count") == EXPECTED_PROJECTIONS
        and checkpoint.get("tensor_count") == EXPECTED_PROJECTIONS * 4
        and isinstance(checkpoint.get("tensor_bytes"), int)
        and checkpoint["tensor_bytes"] > 0
        and SHA256_RE.fullmatch(
            str(checkpoint.get("checkpoint_inventory_sha256", ""))
        )
        is not None
    )
    execution_upgrade_sha256 = report.get("execution_upgrade_sha256")
    valid_execution_upgrade = execution_upgrade_sha256 is None or (
        isinstance(execution_upgrade_sha256, str)
        and SHA256_RE.fullmatch(execution_upgrade_sha256) is not None
    )
    if (
        report.get("schema") != contract.validation_schema
        or report.get("status") != "accepted"
        or not isinstance(report_digest, str)
        or hashlib.sha256(_canonical_json(report_body)).hexdigest() != report_digest
        or report.get("model_id") != contract.model_id
        or reported_artifact != artifact
        or report.get("artifact_manifest_sha256") != artifact_manifest_sha256
        or SHA256_RE.fullmatch(str(report.get("plan_sha256", ""))) is None
        or report.get("retained_native_bytes_verified") is not True
        or report.get("artifact_manifest_file_hashes_verified") is not True
        or not valid_checkpoint
        or not valid_tokenizer_evidence
        or not valid_source_metadata
        or not valid_quantization_config
        or not valid_execution_upgrade
    ):
        raise StagingError("validation report does not accept this exact artifact")
    identity = {
        "path": "artifact-validation.json",
        "bytes": report_path.stat().st_size,
        "sha256": _hash_file(report_path),
        "schema": contract.validation_schema,
    }
    return identity, report


def _quant_evidence(
    report_path: Path,
    *,
    plan_sha256: str,
    contract: ArtifactContract | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_schema = (
        QUANT_EVIDENCE_SCHEMA
        if contract is None or contract.exl3_bits == 3
        else GLM53_QUANT_EVIDENCE_SCHEMA
    )
    report_path = report_path.expanduser().resolve(strict=True)
    if report_path.is_symlink() or not report_path.is_file():
        raise StagingError("quant-evidence report is not a regular file")
    report = _json_object(report_path)
    report_digest = report.get("report_sha256")
    body = {key: value for key, value in report.items() if key != "report_sha256"}
    plan = report.get("plan")
    coverage = report.get("coverage")
    integrity = report.get("integrity")
    metrics = report.get("metrics")
    global_metrics = metrics.get("global") if isinstance(metrics, dict) else None
    aggregate_error = (
        global_metrics.get("aggregate_hessian_weighted_relative_error")
        if isinstance(global_metrics, dict)
        else None
    )
    execution_upgrade = report.get("execution_upgrade")
    if execution_upgrade is None:
        execution_upgrade_sha256 = None
        valid_execution_upgrade = True
    else:
        active = (
            execution_upgrade.get("active_upgrade_sha256")
            if isinstance(execution_upgrade, dict)
            else None
        )
        chain = (
            execution_upgrade.get("chain")
            if isinstance(execution_upgrade, dict)
            else None
        )
        projection_records = (
            execution_upgrade.get("projection_records")
            if isinstance(execution_upgrade, dict)
            else None
        )
        execution_upgrade_sha256 = active
        valid_execution_upgrade = (
            isinstance(active, str)
            and SHA256_RE.fullmatch(active) is not None
            and isinstance(chain, list)
            and bool(chain)
            and chain[0] == active
            and all(
                isinstance(digest, str)
                and SHA256_RE.fullmatch(digest) is not None
                for digest in chain
            )
            and len(chain) == len(set(chain))
            and isinstance(projection_records, dict)
            and set(projection_records) <= {"parent-plan", *chain}
            and all(
                not isinstance(count, bool)
                and isinstance(count, int)
                and count >= 0
                for count in projection_records.values()
            )
            and sum(projection_records.values()) == EXPECTED_PROJECTIONS
        )
    expected_experts = 75 * 256
    if (
        report.get("schema") != expected_schema
        or report.get("status") != "accepted"
        or report.get("quality_scope")
        != "projection-quantizer-evidence-not-end-to-end-model-quality"
        or not isinstance(report_digest, str)
        or hashlib.sha256(_canonical_json(body)).hexdigest() != report_digest
        or not isinstance(plan, dict)
        or plan.get("plan_sha256") != plan_sha256
        or not isinstance(coverage, dict)
        or coverage.get("expected_projection_count") != EXPECTED_PROJECTIONS
        or coverage.get("projection_count") != EXPECTED_PROJECTIONS
        or coverage.get("expected_expert_count") != expected_experts
        or coverage.get("observed_expert_count") != expected_experts
        or coverage.get("complete_expert_count") != expected_experts
        or coverage.get("layers") != list(range(3, 78))
        or isinstance(coverage.get("recovered_expert_count"), bool)
        or not isinstance(coverage.get("recovered_expert_count"), int)
        or not 0 <= coverage["recovered_expert_count"] <= expected_experts
        or not isinstance(integrity, dict)
        or integrity.get("tensor_payload_hashes_verified") is not True
        or integrity.get("journal_record_count") != EXPECTED_PROJECTIONS
        or SHA256_RE.fullmatch(
            str(integrity.get("checkpoint_inventory_sha256", ""))
        )
        is None
        or not isinstance(aggregate_error, (int, float))
        or isinstance(aggregate_error, bool)
        or not math.isfinite(float(aggregate_error))
        or float(aggregate_error) < 0.0
        or not valid_execution_upgrade
    ):
        raise StagingError(
            "quant-evidence report does not accept the complete quantization"
        )
    identity = {
        "path": "quant-evidence.json",
        "bytes": report_path.stat().st_size,
        "sha256": _hash_file(report_path),
        "schema": expected_schema,
        "report_sha256": report_digest,
    }
    return identity, report


def _publication_evidence(
    report_path: Path,
    *,
    publication: Path,
    model_id: str = MODEL_ID,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    report_path = report_path.expanduser().resolve(strict=True)
    if report_path.is_symlink() or not report_path.is_file():
        raise StagingError("publication report is not a regular file")
    report = _json_object(report_path)
    body = {
        key: report.get(key)
        for key in (
            "schema",
            "model_id",
            "source_artifact_manifest_sha256",
            "source_validation_sha256",
            "source_quant_evidence_sha256",
            "source_serving_qualification_sha256",
            "plan_sha256",
            "files",
        )
    }
    publication_sha256 = report.get("publication_sha256")
    entries = report.get("files")
    if (
        report.get("schema") != PUBLICATION_SCHEMA
        or report.get("status") != "ready"
        or report.get("model_id") != model_id
        or Path(str(report.get("output", ""))).expanduser().resolve() != publication
        or SHA256_RE.fullmatch(str(report.get("plan_sha256", ""))) is None
        or SHA256_RE.fullmatch(
            str(report.get("source_quant_evidence_sha256", ""))
        )
        is None
        or SHA256_RE.fullmatch(
            str(report.get("source_serving_qualification_sha256", ""))
        )
        is None
        or not isinstance(entries, list)
        or not entries
        or not isinstance(publication_sha256, str)
        or hashlib.sha256(_canonical_json(body)).hexdigest() != publication_sha256
    ):
        raise StagingError("publication report does not bind this standard model tree")
    expected: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise StagingError("publication file identity is malformed")
        relative = _safe_relative(entry["path"])
        size = entry["bytes"]
        digest = entry["sha256"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or SHA256_RE.fullmatch(str(digest)) is None
            or relative.as_posix() in expected
        ):
            raise StagingError("publication file identity is invalid")
        path = publication.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
            raise StagingError(f"publication file differs: {relative}")
        expected.add(relative.as_posix())
        normalized.append(
            {"path": relative.as_posix(), "bytes": size, "sha256": digest}
        )
    actual = {
        path.relative_to(publication).as_posix()
        for path in publication.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected:
        raise StagingError("publication file set changed after its report was written")
    identity = {
        "path": "standard-publication.json",
        "bytes": report_path.stat().st_size,
        "sha256": _hash_file(report_path),
        "schema": PUBLICATION_SCHEMA,
    }
    return normalized, identity, report


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as target:
            target.write(value + "\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _install_file(source: Path, destination: Path, *, mode: str, expected: dict[str, Any]) -> None:
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or destination.stat().st_size != expected["bytes"]
            or _hash_file(destination) != expected["sha256"]
        ):
            raise StagingError(f"existing cache blob is inconsistent: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        if mode == "hardlink":
            try:
                os.link(source, temporary, follow_symlinks=False)
            except OSError as error:
                if error.errno == errno.EXDEV:
                    raise StagingError(
                        "artifact and HF cache are on different filesystems; "
                        "use --link-mode copy explicitly"
                    ) from error
                raise
        else:
            shutil.copy2(source, temporary, follow_symlinks=False)
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def stage(
    artifact_path: Path,
    validation_report_path: Path | None,
    *,
    quant_evidence_report_path: Path | None = None,
    publication_report_path: Path | None = None,
    model_id: str,
    hf_home: Path,
    link_mode: str,
    update_ref: bool,
) -> dict[str, Any]:
    if model_id not in SUPPORTED_MODEL_IDS:
        raise StagingError(
            "unsupported model ID; expected one of "
            + ", ".join(sorted(SUPPORTED_MODEL_IDS))
        )
    if (validation_report_path is None) == (publication_report_path is None):
        raise StagingError(
            "select exactly one of an artifact validation or standard publication report"
        )
    if (publication_report_path is None) != (quant_evidence_report_path is not None):
        raise StagingError(
            "artifact staging requires --quant-evidence-report; standard publication "
            "staging must inherit it from the publication report"
        )
    artifact = artifact_path.expanduser().resolve(strict=True)
    if artifact_path.expanduser().is_symlink() or not artifact.is_dir():
        raise StagingError("artifact is not a regular directory")
    if publication_report_path is None:
        assert validation_report_path is not None
        entries, artifact_manifest, contract = _artifact_entries(artifact)
        if model_id != contract.model_id:
            raise StagingError(
                f"this artifact must be staged as {contract.model_id}, got {model_id}"
            )
        qualification, validation = _validation_evidence(
            validation_report_path,
            artifact=artifact,
            artifact_manifest_sha256=artifact_manifest["manifest_sha256"],
            contract=contract,
        )
        artifact_manifest_sha256 = artifact_manifest["manifest_sha256"]
        plan_sha256 = validation["plan_sha256"]
        assert quant_evidence_report_path is not None
        quant_qualification, quant_report = _quant_evidence(
            quant_evidence_report_path,
            plan_sha256=plan_sha256,
            contract=contract,
        )
        if (
            validation["projection_checkpoint"]["checkpoint_inventory_sha256"]
            != quant_report["integrity"]["checkpoint_inventory_sha256"]
            or validation.get("execution_upgrade_sha256")
            != (
                quant_report.get("execution_upgrade") or {}
            ).get("active_upgrade_sha256")
        ):
            raise StagingError(
                "artifact and quant evidence bind different projection or execution identities"
            )
        quant_evidence_identity = quant_qualification
        quant_evidence_sha256 = quant_qualification["sha256"]
        evidence_sources = [
            (validation_report_path, qualification),
            (quant_evidence_report_path, quant_qualification),
        ]
    else:
        entries, qualification, validation = _publication_evidence(
            publication_report_path,
            publication=artifact,
            model_id=model_id,
        )
        artifact_manifest_sha256 = validation["source_artifact_manifest_sha256"]
        plan_sha256 = validation["plan_sha256"]
        quant_evidence_identity = None
        quant_evidence_sha256 = validation["source_quant_evidence_sha256"]
        evidence_sources = [(publication_report_path, qualification)]
    body = {
        "schema": SCHEMA,
        "model_id": model_id,
        "files": entries,
        "qualification": qualification,
        "quant_evidence": quant_evidence_identity,
        "quant_evidence_sha256": quant_evidence_sha256,
    }
    revision = hashlib.sha256(_canonical_json(body)).hexdigest()
    manifest = {
        **body,
        "revision": revision,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "plan_sha256": plan_sha256,
        "total_bytes": sum(entry["bytes"] for entry in entries),
    }

    cache = _model_cache_root(hf_home.expanduser().resolve(), model_id)
    ref = cache / "refs" / "main"
    if ref.exists():
        current = ref.read_text(encoding="utf-8").strip()
        if current != revision and not update_ref:
            raise StagingError(
                f"refs/main already selects {current}; pass --update-ref to select {revision}"
            )
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / revision
    staging = cache / "snapshots" / f".{revision}.{os.getpid()}.tmp"
    if staging.exists() or staging.is_symlink():
        raise StagingError(f"stale snapshot staging path exists: {staging}")
    blobs.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    try:
        for entry in entries:
            relative = _safe_relative(entry["path"])
            source = artifact.joinpath(*relative.parts)
            blob = blobs / entry["sha256"]
            _install_file(source, blob, mode=link_mode, expected=entry)
            link = staging.joinpath(*relative.parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(blob, link.parent))
        if snapshot.exists():
            expected = {entry["path"] for entry in entries}
            actual = {
                path.relative_to(snapshot).as_posix()
                for path in snapshot.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            if actual != expected:
                raise StagingError("existing staged snapshot file set differs")
            for entry in entries:
                relative = _safe_relative(entry["path"])
                link = snapshot.joinpath(*relative.parts)
                blob = blobs / entry["sha256"]
                if (
                    not link.is_symlink()
                    or link.resolve(strict=True) != blob.resolve(strict=True)
                ):
                    raise StagingError(
                        f"existing staged snapshot link differs: {relative}"
                    )
            shutil.rmtree(staging)
        else:
            for directory in sorted(
                (path for path in staging.rglob("*") if path.is_dir()),
                key=lambda path: len(path.parts),
                reverse=True,
            ):
                _fsync_directory(directory)
            _fsync_directory(staging)
            os.replace(staging, snapshot)
            _fsync_directory(snapshot.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    for evidence_source, evidence_identity in evidence_sources:
        evidence = (
            cache
            / "glmrt-qualifications"
            / revision
            / evidence_identity["path"]
        )
        _install_file(
            evidence_source.expanduser().resolve(strict=True),
            evidence,
            mode=link_mode,
            expected=evidence_identity,
        )
    _atomic_json(cache / "glmrt-manifests" / f"{revision}.json", manifest)
    _atomic_text(ref, revision)
    return {
        "schema": SCHEMA,
        "status": "staged",
        "model_id": model_id,
        "revision": revision,
        "cache_root": os.fspath(cache),
        "snapshot": os.fspath(snapshot),
        "files": len(entries),
        "bytes": manifest["total_bytes"],
        "link_mode": link_mode,
        "artifact_manifest_sha256": artifact_manifest_sha256,
        "evidence_report_sha256": qualification["sha256"],
        "quant_evidence_report_sha256": quant_evidence_sha256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--validation-report", type=Path)
    evidence.add_argument("--publication-report", type=Path)
    parser.add_argument("--quant-evidence-report", type=Path)
    parser.add_argument(
        "--model-id",
        required=True,
        help="exact accepted Hugging Face repository ID; never inferred from a legacy default",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
    )
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--update-ref", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = stage(
        args.artifact,
        args.validation_report,
        quant_evidence_report_path=args.quant_evidence_report,
        publication_report_path=args.publication_report,
        model_id=args.model_id,
        hf_home=args.hf_home,
        link_mode=args.link_mode,
        update_ref=args.update_ref,
    )
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
