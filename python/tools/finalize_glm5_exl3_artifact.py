#!/usr/bin/env python3
"""Finalize an already-exported GLM-5 EXL3 artifact without duplicating weights.

The quantizer's model library serializes a derived Transformers config.  That
serialization can add defaults and, for GLM-5, has historically changed
``head_dim``.  This tool creates a hard-linked sibling artifact whose
``config.json`` is reconstructed from the exact source snapshot plus the
validated compact EXL3 declaration, then rebinds the artifact manifests.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from validate_glm52_exl3_artifact import (
    EXACT_SOURCE_METADATA_FILES,
    ArtifactValidationError,
    _canonical_json,
    _compact_exl3_declaration,
    _json_object,
    _module_contract,
    _validate_manifest,
    _validate_quantization_config,
    _validate_quantization_provenance,
)


ARTIFACT_MANIFEST_FILENAME = "glmrt-gptqmodel-artifact.json"
RUN_FILENAME = "glmrt-gptqmodel-run.json"


class FinalizationError(RuntimeError):
    """The raw export cannot be normalized safely."""


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def bound(value: dict[str, Any], digest_field: str) -> dict[str, Any]:
    body = {key: item for key, item in value.items() if key != digest_field}
    return {
        **body,
        digest_field: hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


def fsync_staged_tree(root: Path, rewritten: set[Path]) -> None:
    for path in sorted(rewritten):
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
    for directory in sorted(
        directories,
        key=lambda path: len(path.relative_to(root).parts),
        reverse=True,
    ):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def normalized_model_config(artifact: Path, source_snapshot: Path) -> dict[str, Any]:
    source = _json_object(source_snapshot / "config.json")
    exported = _json_object(artifact / "config.json")
    external = _json_object(artifact / "quantize_config.json")
    embedded = exported.get("quantization_config")
    storage = external.get("tensor_storage")
    compact = dict(external)
    compact.pop("tensor_storage", None)
    minimal = _compact_exl3_declaration(external)
    # GPTQModel serializers have embedded either the metadata-rich declaration
    # or the complete standalone object, including tensor_storage. Both are
    # unambiguous only when they agree with quantize_config.json. Always emit
    # the four-field discovery declaration; all other metadata stays external.
    if not isinstance(embedded, dict) or embedded not in (
        minimal,
        compact,
        external,
    ):
        raise FinalizationError(
            "raw embedded and standalone EXL3 configurations do not agree"
        )
    if not isinstance(storage, dict) or not storage:
        raise FinalizationError("raw standalone EXL3 tensor_storage is empty")
    return {**source, "quantization_config": minimal}


def finalize(*, artifact_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = artifact_path.expanduser().resolve(strict=True)
    output = output_path.expanduser().resolve()
    if not artifact.is_dir() or artifact.is_symlink():
        raise FinalizationError("raw artifact must be a regular directory")
    if output.exists() or output.is_symlink():
        raise FinalizationError(f"finalized artifact already exists: {output}")
    try:
        plan, manifest, run, execution_upgrade, contract = _validate_manifest(
            artifact,
            verify_hashes=False,
        )
    except ArtifactValidationError as error:
        raise FinalizationError("raw artifact manifests are invalid") from error
    source_snapshot = Path(str(plan.get("source", {}).get("path", ""))).expanduser()
    if not source_snapshot.is_dir() or source_snapshot.is_symlink():
        raise FinalizationError("plan source snapshot is unavailable")
    source_snapshot = source_snapshot.resolve(strict=True)
    plan_source = plan.get("source")
    source_config_sha256 = hash_file(source_snapshot / "config.json")
    if (
        not isinstance(plan_source, dict)
        or plan_source.get("config_sha256") != source_config_sha256
    ):
        raise FinalizationError(
            "source config differs from the immutable quantization plan"
        )
    normalized = normalized_model_config(artifact, source_snapshot)
    rendered_config = (
        json.dumps(normalized, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        records: dict[str, dict[str, Any]] = {}
        rewritten: set[Path] = set()
        missing_source_metadata = set(EXACT_SOURCE_METADATA_FILES) - set(
            manifest["files"]
        )
        if missing_source_metadata:
            raise FinalizationError(
                "raw artifact is missing source metadata: "
                f"{sorted(missing_source_metadata)}"
            )
        for relative, identity in manifest["files"].items():
            source = artifact / relative
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if relative == "config.json":
                destination.write_bytes(rendered_config)
                rewritten.add(destination)
                records[relative] = {
                    "bytes": len(rendered_config),
                    "sha256": hashlib.sha256(rendered_config).hexdigest(),
                }
            elif relative in EXACT_SOURCE_METADATA_FILES:
                payload = (source_snapshot / relative).read_bytes()
                destination.write_bytes(payload)
                rewritten.add(destination)
                records[relative] = {
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            else:
                try:
                    os.link(source, destination, follow_symlinks=False)
                except OSError as error:
                    if error.errno in {errno.EACCES, errno.EPERM}:
                        raise FinalizationError(
                            "cannot hard-link the raw artifact payload; normalize "
                            "its ownership to the host user after the root-run "
                            "quantizer exits"
                        ) from error
                    raise FinalizationError(
                        "finalized artifact must share a filesystem with the raw export"
                    ) from error
                records[relative] = dict(identity)

        manifest_body = {
            key: value
            for key, value in manifest.items()
            if key != "manifest_sha256"
        }
        manifest_body.update(
            {
                "files": records,
                "file_count": len(records),
                "total_bytes": sum(record["bytes"] for record in records.values()),
            }
        )
        rebound_manifest = bound(manifest_body, "manifest_sha256")
        run_body = {key: value for key, value in run.items() if key != "run_sha256"}
        run_body["artifact_manifest_sha256"] = rebound_manifest["manifest_sha256"]
        rebound_run = bound(run_body, "run_sha256")
        (temporary / ARTIFACT_MANIFEST_FILENAME).write_text(
            json.dumps(rebound_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (temporary / RUN_FILENAME).write_text(
            json.dumps(rebound_run, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        rewritten.update(
            {
                temporary / ARTIFACT_MANIFEST_FILENAME,
                temporary / RUN_FILENAME,
            }
        )
        _validate_manifest(temporary, verify_hashes=False)
        finalized_quantization = _validate_quantization_config(
            temporary,
            _module_contract(),
            contract,
            source_config_path=source_snapshot / "config.json",
        )
        quantization_provenance = _validate_quantization_provenance(
            finalized_quantization,
            plan,
            execution_upgrade,
        )
        finalized_storage = finalized_quantization.get("tensor_storage")
        if not isinstance(finalized_storage, dict) or not finalized_storage:
            raise FinalizationError(
                "finalized standalone EXL3 tensor_storage is empty"
            )
        stored_tensor_descriptions = sum(
            len(entry.get("stored_tensors", {}))
            if isinstance(entry, dict)
            and isinstance(entry.get("stored_tensors"), dict)
            else 0
            for entry in finalized_storage.values()
        )
        fsync_staged_tree(temporary, rewritten)
        os.replace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    report = {
        "schema": "glmrt-glm5-exl3-artifact-finalization-v1",
        "status": "complete",
        "raw_artifact": os.fspath(artifact),
        "artifact": os.fspath(output),
        "plan_sha256": plan["plan_sha256"],
        "source_config_sha256": source_config_sha256,
        "raw_config_sha256": hash_file(artifact / "config.json"),
        "config_sha256": hash_file(output / "config.json"),
        "config_bytes": (output / "config.json").stat().st_size,
        "embedded_quantization_fields": sorted(
            normalized["quantization_config"]
        ),
        "quantize_config_sha256": hash_file(output / "quantize_config.json"),
        "quantize_config_bytes": (output / "quantize_config.json").stat().st_size,
        "tensor_storage_entries": len(finalized_storage),
        "stored_tensor_descriptions": stored_tensor_descriptions,
        "ledger_provenance_sha256": hashlib.sha256(
            _canonical_json(quantization_provenance)
        ).hexdigest(),
        "artifact_manifest_sha256": rebound_manifest["manifest_sha256"],
        "storage_mode": "hardlink",
    }
    return bound(report, "report_sha256")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = finalize(artifact_path=args.artifact, output_path=args.output)
        atomic_json(args.report, report)
    except (FinalizationError, ArtifactValidationError, OSError, ValueError) as error:
        parser.exit(2, f"finalize-glm5-exl3-artifact: {error}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
