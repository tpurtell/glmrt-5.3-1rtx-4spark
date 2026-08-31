#!/usr/bin/env python3
"""Validate a completed GLM-5 EXL3 artifact before publication.

The check is intentionally independent of GPTQModel.  It reads only JSON and
safetensors headers, proves the exact routed-expert replacement namespace and
EXL3 storage metadata, and byte-compares every retained native tensor with the
pinned source snapshot.  Tensor payloads are streamed; neither model is loaded
into host or device memory.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import math
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, os.fspath(_REPO_ROOT / "quantization"))
from glm52_execution_upgrade import (  # noqa: E402
    EXECUTION_UPGRADE_FILENAME,
    ExecutionUpgradeError,
    read_execution_upgrade,
)


SCHEMA = "glmrt-glm52-exl3-artifact-validation-v5"
PLAN_SCHEMAS = {
    "glmrt-glm52-gptqmodel-plan-v1",
    "glmrt-glm52-gptqmodel-plan-v2",
}
ARTIFACT_SCHEMA = "glmrt-glm52-gptqmodel-artifact-v1"
RUN_SCHEMA = "glmrt-glm52-gptqmodel-run-v1"
RECIPE = "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1"
MODULE_INCLUDE = (
    r"^model\.layers\.(?:[3-9]|[1-6][0-9]|7[0-7])\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)
MODEL_ID = "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1"
GLM53_MODEL_ID = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
GLM53_VALIDATION_SCHEMA = "glmrt-glm5-exl3-artifact-validation-v1"
GLM53_ARTIFACT_SCHEMA = "glmrt-glm5-gptqmodel-artifact-v2"
GLM53_RUN_SCHEMA = "glmrt-glm5-gptqmodel-run-v2"
HIDDEN_SIZE = 6144
MOE_INTERMEDIATE_SIZE = 2048
FIRST_ROUTED_LAYER = 3
BASE_LAYER_END = 78
MTP_LAYER = 78
ROUTED_EXPERTS = 256
TOP_K = 8
EXL3_BITS = 3
MCG_MULTIPLIER = 0xCBAC1FED
EXPECTED_MODULES = (BASE_LAYER_END - FIRST_ROUTED_LAYER) * ROUTED_EXPERTS * 3
EXPECTED_EXL3_TENSORS = EXPECTED_MODULES * 4
EXPECTED_TP4_RESIDENT_BYTES = 916_194_304 * (BASE_LAYER_END - FIRST_ROUTED_LAYER)
CHECKPOINT_SCHEMA = "ds4rt.exl3-projection-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
HF_BLOB_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
EXACT_SOURCE_METADATA_FILES = (*TOKENIZER_FILES, "generation_config.json")
COMPACT_EXL3_DECLARATION_FIELDS = (
    "quant_method",
    "format",
    "checkpoint_format",
    "bits",
)
TOKENIZER_ATTESTATION_SCHEMA = "glmrt-glm52-legacy-tokenizer-attestation-v1"
TOKENIZATION_CONTRACT = "gptqmodel-raw-text-add-special-tokens-return-pt-v1"
NATIVE_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}
TORCH_DTYPE = {
    "I16": "torch.int16",
    "F16": "torch.float16",
    "I32": "torch.int32",
}


@dataclass(frozen=True)
class ArtifactContract:
    validation_schema: str
    plan_schemas: frozenset[str]
    artifact_schema: str
    run_schema: str
    recipe: str
    model_id: str
    release: str
    source_format: str
    exl3_bits: int
    source_has_block_fp8_scales: bool

    @property
    def expected_tp4_resident_bytes(self) -> int:
        # Each Spark owns one quarter of every projection's intermediate
        # dimension. Rotations and the MCG marker follow that same TP4 view.
        trellis_per_projection = (
            (HIDDEN_SIZE // 16)
            * (MOE_INTERMEDIATE_SIZE // 16 // 4)
            * (16 * self.exl3_bits)
            * 2
        )
        rotations_per_projection = (
            HIDDEN_SIZE * 2 + (MOE_INTERMEDIATE_SIZE // 4) * 2
        )
        per_expert = 3 * (trellis_per_projection + rotations_per_projection) + 4
        per_layer = per_expert * ROUTED_EXPERTS
        return per_layer * (BASE_LAYER_END - FIRST_ROUTED_LAYER)


def _artifact_contract(plan: dict[str, Any]) -> ArtifactContract:
    recipe = plan.get("recipe")
    source = plan.get("source")
    if not isinstance(source, dict):
        raise ArtifactValidationError("artifact plan has no source contract")
    if recipe == "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1":
        contract = ArtifactContract(
            validation_schema=SCHEMA,
            plan_schemas=frozenset(PLAN_SCHEMAS),
            artifact_schema=ARTIFACT_SCHEMA,
            run_schema=RUN_SCHEMA,
            recipe=recipe,
            model_id=MODEL_ID,
            release="glm-5.2",
            source_format="bf16",
            exl3_bits=3,
            source_has_block_fp8_scales=False,
        )
    elif recipe == "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1":
        contract = ArtifactContract(
            validation_schema=GLM53_VALIDATION_SCHEMA,
            plan_schemas=frozenset({"glmrt-glm5-gptqmodel-plan-v3"}),
            artifact_schema=GLM53_ARTIFACT_SCHEMA,
            run_schema=GLM53_RUN_SCHEMA,
            recipe=recipe,
            model_id=GLM53_MODEL_ID,
            release="glm-5.3",
            source_format="fp8-e4m3-block128x128-dynamic",
            exl3_bits=4,
            source_has_block_fp8_scales=True,
        )
    else:
        raise ArtifactValidationError(f"unsupported GLM-5 EXL3 recipe: {recipe!r}")
    if (
        plan.get("schema") not in contract.plan_schemas
        or source.get("release", contract.release) != contract.release
        or source.get("format", contract.source_format) != contract.source_format
    ):
        raise ArtifactValidationError("artifact plan/source contract is inconsistent")
    return contract


class ArtifactValidationError(RuntimeError):
    """The candidate cannot be accepted as a calibrated GLM-5 artifact."""


def _compact_exl3_declaration(external: dict[str, Any]) -> dict[str, Any]:
    """Return only the Hub/runtime discovery fields embedded in config.json."""

    return {field: external.get(field) for field in COMPACT_EXL3_DECLARATION_FIELDS}


@dataclass(frozen=True)
class TensorRecord:
    file: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True)
class SnapshotInventory:
    root: Path
    tensors: dict[str, TensorRecord]
    files: tuple[Path, ...]
    tensor_bytes: int


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactValidationError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactValidationError(f"expected a JSON object in {path}")
    return value


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactValidationError(f"{label} is not a nonempty path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ArtifactValidationError(f"{label} is unsafe: {value!r}")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _hash_tensor_range(handle: Any, record: TensorRecord) -> str:
    digest = hashlib.sha256()
    offset = record.offset
    remaining = record.length
    while remaining:
        count = min(8 * 1024 * 1024, remaining)
        block = os.pread(handle.fileno(), count, offset)
        if len(block) != count:
            raise ArtifactValidationError(
                f"short tensor payload read from {record.file}"
            )
        digest.update(block)
        offset += count
        remaining -= count
    return digest.hexdigest()


def _source_tokenizer_identity(snapshot: Path, name: str) -> dict[str, Any]:
    if Path(name).name != name:
        raise ArtifactValidationError(f"unsafe tokenizer file name: {name}")
    path = snapshot / name
    resolved = path
    blob_id: str | None = None
    if path.is_symlink():
        link = Path(os.readlink(path))
        if (
            link.is_absolute()
            or len(link.parts) != 4
            or link.parts[:3] != ("..", "..", "blobs")
            or HF_BLOB_RE.fullmatch(link.parts[3]) is None
        ):
            raise ArtifactValidationError(
                f"source tokenizer is not a canonical Hugging Face blob: {path}"
            )
        blob_root = (snapshot.parent.parent / "blobs").resolve(strict=True)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise ArtifactValidationError(
                f"source tokenizer has a broken link: {path}"
            ) from exc
        expected_blob = blob_root / link.parts[3]
        if (
            resolved != expected_blob
            or expected_blob.is_symlink()
            or not expected_blob.is_file()
        ):
            raise ArtifactValidationError(
                f"source tokenizer escapes its Hugging Face blob store: {path}"
            )
        blob_id = link.parts[3]
    elif not path.is_file():
        raise ArtifactValidationError(f"source tokenizer file is missing: {path}")
    digest = _hash_file(resolved)
    if blob_id is not None and len(blob_id) == 64 and blob_id != digest:
        raise ArtifactValidationError(f"source tokenizer SHA-256 blob changed: {path}")
    result = {
        "name": name,
        "bytes": resolved.stat().st_size,
        "sha256": digest,
    }
    if blob_id is not None:
        result["hf_blob_id"] = blob_id
    return result


def _validate_tokenizer_evidence(
    *,
    plan: dict[str, Any],
    source: Path,
    attestation_path: Path | None,
) -> dict[str, Any]:
    plan_source = plan.get("source")
    assert isinstance(plan_source, dict)
    current = [
        _source_tokenizer_identity(source, name) for name in TOKENIZER_FILES
    ]
    planned = plan_source.get("tokenizer_files")
    if planned is not None:
        if attestation_path is not None:
            raise ArtifactValidationError(
                "a tokenizer attestation is invalid for a tokenizer-bound plan"
            )
        if planned != current:
            raise ArtifactValidationError(
                "source tokenizer files differ from the immutable plan"
            )
        return {"mode": "plan-bound", "tokenizer_files": current}

    if attestation_path is None:
        raise ArtifactValidationError(
            "legacy plan requires --tokenizer-attestation"
        )
    attestation_path = attestation_path.expanduser().resolve(strict=True)
    if attestation_path.is_symlink() or not attestation_path.is_file():
        raise ArtifactValidationError("tokenizer attestation is not a regular file")
    attestation = _json_object(attestation_path)
    claimed_digest = attestation.get("attestation_sha256")
    body = {
        key: value
        for key, value in attestation.items()
        if key != "attestation_sha256"
    }
    attested_source = attestation.get("source")
    attested_plan = attestation.get("plan")
    container = attestation.get("container")
    tokenization = attestation.get("tokenization")
    corpus = plan.get("corpus")
    preflight = plan.get("preflight")
    if (
        attestation.get("schema") != TOKENIZER_ATTESTATION_SCHEMA
        or attestation.get("status") != "accepted"
        or attestation.get("scope")
        != "legacy-plan-omitted-tokenizer-source-identity"
        or not isinstance(claimed_digest, str)
        or hashlib.sha256(_canonical_json(body)).hexdigest() != claimed_digest
        or not isinstance(attested_plan, dict)
        or attested_plan.get("plan_sha256") != plan.get("plan_sha256")
        or not isinstance(attested_source, dict)
        or Path(str(attested_source.get("path", ""))).expanduser().resolve()
        != source
        or attested_source.get("revision") != plan_source.get("revision")
        or not isinstance(container, dict)
        or not isinstance(preflight, dict)
        or container.get("image_digest")
        != preflight.get("image_digest")
        or container.get("restart_count") != 0
        or container.get("tokenizer_inputs_predate_start") is not True
        or not isinstance(corpus, dict)
        or attestation.get("corpus") != corpus
        or not isinstance(tokenization, dict)
        or tokenization.get("contract") != TOKENIZATION_CONTRACT
        or tokenization.get("add_special_tokens") is not True
        or tokenization.get("return_tensors") != "pt"
        or tokenization.get("records") != corpus.get("examples")
        or not isinstance(tokenization.get("total_tokens"), int)
        or tokenization["total_tokens"] <= 0
        or not isinstance(tokenization.get("minimum_tokens"), int)
        or tokenization["minimum_tokens"] <= 0
        or not isinstance(tokenization.get("maximum_tokens"), int)
        or tokenization["maximum_tokens"] < tokenization["minimum_tokens"]
        or SHA256_RE.fullmatch(
            str(tokenization.get("prepared_token_stream_sha256", ""))
        )
        is None
    ):
        raise ArtifactValidationError("legacy tokenizer attestation is invalid")
    attested_files = attested_source.get("tokenizer_files")
    if not isinstance(attested_files, list) or len(attested_files) != len(current):
        raise ArtifactValidationError("legacy tokenizer file evidence is incomplete")
    core_fields = ("name", "bytes", "sha256", "hf_blob_id")
    try:
        attested_core = [
            {key: record[key] for key in core_fields} for record in attested_files
        ]
    except (KeyError, TypeError) as exc:
        raise ArtifactValidationError(
            "legacy tokenizer file evidence is malformed"
        ) from exc
    if attested_core != current:
        raise ArtifactValidationError(
            "source tokenizer differs from the legacy attestation"
        )
    return {
        "mode": "legacy-live-container-attestation",
        "path": os.fspath(attestation_path),
        "file_sha256": _hash_file(attestation_path),
        "attestation_sha256": claimed_digest,
        "tokenizer_files": current,
        "prepared_token_stream_sha256": tokenization[
            "prepared_token_stream_sha256"
        ],
        "total_tokens": tokenization["total_tokens"],
    }


def _regular_root(path: Path, *, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ArtifactValidationError(f"{label} cannot be a symbolic link: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_dir():
        raise ArtifactValidationError(f"{label} is not a regular directory: {resolved}")
    return resolved


def _parse_safetensors(path: Path, *, reject_symlink: bool) -> dict[str, TensorRecord]:
    if reject_symlink and path.is_symlink():
        raise ArtifactValidationError(f"artifact contains a symbolic tensor shard: {path}")
    try:
        size = path.stat().st_size
        with path.open("rb") as source:
            prefix = source.read(8)
            if len(prefix) != 8:
                raise ArtifactValidationError(f"safetensors shard has no header length: {path}")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if header_bytes == 0 or header_bytes > min(size - 8, 1 << 30):
                raise ArtifactValidationError(
                    f"safetensors shard has an invalid header length: {path}"
                )
            payload_offset = 8 + header_bytes
            header = json.loads(source.read(header_bytes))
    except ArtifactValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as error:
        raise ArtifactValidationError(f"cannot inspect safetensors shard {path}: {error}") from error
    if not isinstance(header, dict):
        raise ArtifactValidationError(f"safetensors header is not an object: {path}")
    metadata = header.pop("__metadata__", None)
    if metadata is not None and not isinstance(metadata, dict):
        raise ArtifactValidationError(f"safetensors metadata is malformed: {path}")

    result: dict[str, TensorRecord] = {}
    ranges: list[tuple[int, int, str]] = []
    for name, entry in header.items():
        if not isinstance(name, str) or not name or not isinstance(entry, dict):
            raise ArtifactValidationError(f"safetensors tensor entry is malformed: {path}")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            dtype not in DTYPE_BYTES
            or not isinstance(shape, list)
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
        ):
            raise ArtifactValidationError(f"invalid safetensors metadata for {name} in {path}")
        length = offsets[1] - offsets[0]
        expected = math.prod(shape) * DTYPE_BYTES[dtype]
        if length != expected:
            raise ArtifactValidationError(
                f"safetensors byte count differs for {name}: {length} != {expected}"
            )
        result[name] = TensorRecord(
            file=path,
            dtype=dtype,
            shape=tuple(shape),
            offset=payload_offset + offsets[0],
            length=length,
        )
        ranges.append((offsets[0], offsets[1], name))

    cursor = 0
    for start, end, name in sorted(ranges):
        if start != cursor:
            raise ArtifactValidationError(
                f"safetensors data ranges are not contiguous before {name} in {path}"
            )
        cursor = end
    if payload_offset + cursor != size:
        raise ArtifactValidationError(
            f"safetensors payload length differs from the file size: {path}"
        )
    return result


def _snapshot_inventory(root: Path, *, reject_symlinks: bool) -> SnapshotInventory:
    root = _regular_root(root, label="snapshot")
    index = _json_object(root / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ArtifactValidationError("safetensors index has no weight_map")
    by_file: dict[str, set[str]] = {}
    for name, value in weight_map.items():
        if not isinstance(name, str) or not name:
            raise ArtifactValidationError("safetensors index contains an invalid tensor name")
        file_name = _safe_relative(value, label=f"index file for {name}")
        by_file.setdefault(file_name, set()).add(name)

    def parse(item: tuple[str, set[str]]) -> tuple[str, dict[str, TensorRecord]]:
        file_name, expected_names = item
        path = root / file_name
        actual = _parse_safetensors(path, reject_symlink=reject_symlinks)
        if set(actual) != expected_names:
            missing = sorted(expected_names - set(actual))[:8]
            unexpected = sorted(set(actual) - expected_names)[:8]
            raise ArtifactValidationError(
                f"index/header mismatch for {file_name}: missing={missing} unexpected={unexpected}"
            )
        return file_name, actual

    parsed: dict[str, dict[str, TensorRecord]] = {}
    with ThreadPoolExecutor(max_workers=min(32, len(by_file))) as workers:
        for file_name, tensors in workers.map(parse, sorted(by_file.items())):
            parsed[file_name] = tensors
    tensors = {
        name: parsed[file_name][name]
        for name, file_name in weight_map.items()
    }
    return SnapshotInventory(
        root=root,
        tensors=tensors,
        files=tuple(root / name for name in sorted(by_file)),
        tensor_bytes=sum(record.length for record in tensors.values()),
    )


def _module_contract() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    for layer in range(FIRST_ROUTED_LAYER, BASE_LAYER_END):
        for expert in range(ROUTED_EXPERTS):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            result[f"{prefix}.gate_proj"] = (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
            result[f"{prefix}.up_proj"] = (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)
            result[f"{prefix}.down_proj"] = (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)
    return result


def _tensor_contract(
    module: str,
    input_features: int,
    output_features: int,
    *,
    exl3_bits: int = EXL3_BITS,
) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        f"{module}.trellis": (
            "I16",
            (input_features // 16, output_features // 16, 16 * exl3_bits),
        ),
        f"{module}.suh": ("F16", (input_features,)),
        f"{module}.svh": ("F16", (output_features,)),
        f"{module}.mcg": ("I32", ()),
    }


def _checkpoint_pairs(root: Path) -> list[tuple[str, Path, Path]]:
    root = _regular_root(root, label="projection-checkpoint store")
    manifests: dict[str, Path] = {}
    tensors: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactValidationError(
                f"projection-checkpoint store contains a symbolic link: {path}"
            )
        if path.is_dir():
            relative = path.relative_to(root)
            if (
                len(relative.parts) not in {1, 2}
                or any(
                    len(part) != 2
                    or any(char not in "0123456789abcdef" for char in part)
                    for part in relative.parts
                )
            ):
                raise ArtifactValidationError(
                    f"projection-checkpoint directory is noncanonical: {relative}"
                )
            continue
        if not path.is_file() or path.suffix not in {".json", ".safetensors"}:
            raise ArtifactValidationError(
                f"projection-checkpoint store contains an unsafe entry: {path}"
            )
        relative = path.relative_to(root)
        digest = path.stem
        if (
            len(relative.parts) != 3
            or SHA256_RE.fullmatch(digest) is None
            or relative.parts[0] != digest[:2]
            or relative.parts[1] != digest[2:4]
            or path.name != f"{digest}{path.suffix}"
        ):
            raise ArtifactValidationError(
                f"projection-checkpoint path is noncanonical: {relative}"
            )
        target = manifests if path.suffix == ".json" else tensors
        if digest in target:
            raise ArtifactValidationError(
                f"projection-checkpoint store duplicates {digest}"
            )
        target[digest] = path
    if set(manifests) != set(tensors):
        raise ArtifactValidationError(
            "projection-checkpoint store contains an incomplete pair"
        )
    return [
        (digest, manifests[digest], tensors[digest])
        for digest in sorted(manifests)
    ]


def _validate_checkpoint_artifact_join(
    *,
    checkpoint_root: Path,
    plan: dict[str, Any],
    artifact_inventory: SnapshotInventory,
    modules: dict[str, tuple[int, int]],
    contract: ArtifactContract,
) -> dict[str, Any]:
    resolved = _regular_root(
        checkpoint_root,
        label="projection-checkpoint store",
    )
    checkpoint = plan.get("projection_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or not isinstance(checkpoint.get("root"), str)
        or Path(checkpoint["root"]).expanduser().resolve() != resolved
    ):
        raise ArtifactValidationError(
            "projection-checkpoint store differs from the immutable plan"
        )
    pairs = _checkpoint_pairs(resolved)
    if len(pairs) != EXPECTED_MODULES:
        raise ArtifactValidationError(
            "projection-checkpoint count differs: "
            f"{len(pairs)} != {EXPECTED_MODULES}"
        )

    seen_modules: set[str] = set()
    inventory = hashlib.sha256()
    tensor_bytes = 0
    with ExitStack() as stack:
        artifact_handles = {
            path: stack.enter_context(path.open("rb", buffering=0))
            for path in artifact_inventory.files
        }
        for request_digest, manifest_path, tensor_path in pairs:
            manifest = _json_object(manifest_path)
            manifest_digest = manifest.get("manifest_sha256")
            manifest_body = {
                key: value
                for key, value in manifest.items()
                if key != "manifest_sha256"
            }
            request = manifest.get("request")
            result = manifest.get("result")
            request_body = (
                {
                    key: value
                    for key, value in request.items()
                    if key != "request_sha256"
                }
                if isinstance(request, dict)
                else None
            )
            tensor_file_sha256 = manifest.get("tensor_sha256")
            ledger = result.get("ledger_record") if isinstance(result, dict) else None
            record_sha256 = (
                hashlib.sha256(_canonical_json(ledger)).hexdigest()
                if isinstance(ledger, dict)
                else None
            )
            if (
                manifest.get("schema") != CHECKPOINT_SCHEMA
                or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
                or manifest.get("request_sha256") != request_digest
                or not isinstance(request, dict)
                or request.get("request_sha256") != request_digest
                or hashlib.sha256(_canonical_json(request_body)).hexdigest()
                != request_digest
                or manifest.get("tensor_file") != tensor_path.name
                or SHA256_RE.fullmatch(str(tensor_file_sha256 or "")) is None
                or SHA256_RE.fullmatch(str(manifest_digest or "")) is None
                or hashlib.sha256(_canonical_json(manifest_body)).hexdigest()
                != manifest_digest
                or record_sha256 is None
            ):
                raise ArtifactValidationError(
                    f"projection-checkpoint manifest is invalid: {request_digest}"
                )
            module = request.get("module")
            if (
                not isinstance(module, str)
                or module not in modules
                or module in seen_modules
            ):
                raise ArtifactValidationError(
                    f"projection-checkpoint module identity is invalid: {module!r}"
                )
            seen_modules.add(module)
            if (
                ledger.get("module") != module
                or ledger.get("processor_layer_index")
                != request.get("processor_layer_index")
            ):
                raise ArtifactValidationError(
                    f"projection-checkpoint ledger differs for {module}"
                )

            expected = _tensor_contract(
                module,
                *modules[module],
                exl3_bits=contract.exl3_bits,
            )
            checkpoint_records = _parse_safetensors(
                tensor_path,
                reject_symlink=True,
            )
            expected_suffixes = {
                name.removeprefix(f"{module}."): (name, dtype, shape)
                for name, (dtype, shape) in expected.items()
            }
            specs = manifest.get("tensors")
            if (
                set(checkpoint_records) != set(expected_suffixes)
                or not isinstance(specs, dict)
                or set(specs) != set(expected_suffixes)
            ):
                raise ArtifactValidationError(
                    f"projection-checkpoint tensor set differs for {module}"
                )
            with tensor_path.open("rb", buffering=0) as checkpoint_handle:
                for suffix, (full_name, dtype, shape) in expected_suffixes.items():
                    source_record = checkpoint_records[suffix]
                    artifact_record = artifact_inventory.tensors[full_name]
                    spec = specs[suffix]
                    if (
                        source_record.dtype != dtype
                        or source_record.shape != shape
                        or not isinstance(spec, dict)
                        or spec.get("dtype") != TORCH_DTYPE[dtype]
                        or spec.get("shape") != list(shape)
                        or spec.get("numel") != math.prod(shape)
                        or spec.get("bytes") != source_record.length
                        or SHA256_RE.fullmatch(str(spec.get("sha256", ""))) is None
                    ):
                        raise ArtifactValidationError(
                            f"projection-checkpoint tensor metadata differs: "
                            f"{module}.{suffix}"
                        )
                    source_digest = _hash_tensor_range(
                        checkpoint_handle,
                        source_record,
                    )
                    artifact_digest = _hash_tensor_range(
                        artifact_handles[artifact_record.file],
                        artifact_record,
                    )
                    if source_digest != spec["sha256"] or artifact_digest != source_digest:
                        raise ArtifactValidationError(
                            f"artifact packed tensor differs from its calibrated "
                            f"checkpoint: {full_name}"
                        )
                    tensor_bytes += source_record.length

            stat_bytes = tensor_path.stat().st_size
            inventory.update(
                _canonical_json(
                    {
                        "request_sha256": request_digest,
                        "manifest_sha256": manifest_digest,
                        "tensor_sha256": tensor_file_sha256,
                        "tensor_file_bytes": stat_bytes,
                        "record_sha256": record_sha256,
                    }
                )
                + b"\n"
            )
    if seen_modules != set(modules):
        raise ArtifactValidationError(
            "projection-checkpoint module coverage differs from the artifact"
        )
    return {
        "root": os.fspath(resolved),
        "projection_count": len(pairs),
        "tensor_count": EXPECTED_EXL3_TENSORS,
        "tensor_bytes": tensor_bytes,
        "checkpoint_inventory_sha256": inventory.hexdigest(),
    }


def _validate_quantization_config(
    artifact: Path,
    modules: dict[str, tuple[int, int]],
    contract: ArtifactContract,
    *,
    source_config_path: Path | None = None,
) -> dict[str, Any]:
    config = _json_object(artifact / "config.json")
    if source_config_path is not None:
        source_config = _json_object(source_config_path)
        unexpected_config_fields = set(config) - (
            set(source_config) | {"quantization_config"}
        )
        if unexpected_config_fields:
            raise ArtifactValidationError(
                "artifact config added fields absent from the source: "
                f"{sorted(unexpected_config_fields)}"
            )
        for field, expected in source_config.items():
            if field == "quantization_config":
                continue
            if config.get(field) != expected:
                raise ArtifactValidationError(
                    f"artifact config changed source field {field}: "
                    f"{config.get(field)!r} != {expected!r}"
                )
    expected_geometry = {
        "model_type": "glm_moe_dsa",
        "hidden_size": HIDDEN_SIZE,
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "n_routed_experts": ROUTED_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "num_hidden_layers": BASE_LAYER_END,
        "first_k_dense_replace": FIRST_ROUTED_LAYER,
        "num_nextn_predict_layers": 1,
    }
    for field, expected in expected_geometry.items():
        if config.get(field) != expected:
            raise ArtifactValidationError(
                f"artifact config {field} differs: {config.get(field)!r} != {expected!r}"
            )
    external = _json_object(artifact / "quantize_config.json")
    embedded = config.get("quantization_config")
    if not isinstance(embedded, dict):
        raise ArtifactValidationError("config.json has no quantization_config object")
    required = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": float(contract.exl3_bits),
        "codebook": "mcg",
        "out_scales": "auto",
        "group_size": -1,
        "desc_act": False,
        "module_include": [MODULE_INCLUDE],
    }
    for field, expected in required.items():
        if external.get(field) != expected:
            raise ArtifactValidationError(
                f"external EXL3 config {field} differs from the required contract"
            )
    compact_external = dict(external)
    storage = compact_external.pop("tensor_storage", None)
    minimal_embedded = _compact_exl3_declaration(external)
    if contract.exl3_bits == 4 and embedded != minimal_embedded:
        raise ArtifactValidationError(
            "GLM-5.3 embedded EXL3 declaration is not the exact minimal contract"
        )
    if contract.exl3_bits == 3 and embedded not in (
        minimal_embedded,
        compact_external,
    ):
        raise ArtifactValidationError(
            "GLM-5.2 embedded EXL3 declaration differs from the external contract"
        )
    if not isinstance(storage, dict) or set(storage) != set(modules):
        actual = set(storage) if isinstance(storage, dict) else set()
        raise ArtifactValidationError(
            "EXL3 tensor_storage module set differs: "
            f"expected={len(modules)} actual={len(actual)} "
            f"missing={sorted(set(modules) - actual)[:8]} "
            f"unexpected={sorted(actual - set(modules))[:8]}"
        )
    for module, (input_features, output_features) in modules.items():
        entry = storage[module]
        expected_tensors = _tensor_contract(
            module,
            input_features,
            output_features,
            exl3_bits=contract.exl3_bits,
        )
        if (
            not isinstance(entry, dict)
            or set(entry) != {
                "stored_tensors",
                "quant_format",
                "bits_per_weight",
                "mcg_multiplier",
            }
            or entry.get("quant_format") != "exl3"
            or entry.get("bits_per_weight") != contract.exl3_bits
            or entry.get("mcg_multiplier") != MCG_MULTIPLIER
        ):
            raise ArtifactValidationError(f"invalid EXL3 tensor_storage entry for {module}")
        stored = entry.get("stored_tensors")
        if not isinstance(stored, dict) or set(stored) != set(expected_tensors):
            raise ArtifactValidationError(f"invalid stored tensor set for {module}")
        for name, (dtype, shape) in expected_tensors.items():
            metadata = stored[name]
            torch_dtype = {
                "I16": "int16",
                "F16": "float16",
                "I32": "int32",
            }[dtype]
            if metadata != {"shape": list(shape), "torch_dtype": torch_dtype}:
                raise ArtifactValidationError(f"invalid tensor_storage metadata for {name}")
    return external


def _validate_quantization_provenance(
    external: dict[str, Any],
    plan: dict[str, Any],
    execution_upgrade: dict[str, Any] | None,
) -> dict[str, Any]:
    """Bind the exported calibration ledger to the immutable execution plan."""

    planned = plan.get("ledger_provenance")
    if not isinstance(planned, dict):
        raise ArtifactValidationError(
            "artifact plan has no quantization ledger provenance"
        )
    expected = json.loads(json.dumps(planned))
    if execution_upgrade is not None:
        run = expected.get("run")
        if not isinstance(run, dict):
            raise ArtifactValidationError(
                "artifact plan quantization ledger has no run provenance"
            )
        run["execution_upgrade"] = {
            "schema": execution_upgrade["schema"],
            "upgrade_sha256": execution_upgrade["upgrade_sha256"],
            "parent_plan_sha256": execution_upgrade["parent_plan_sha256"],
            "upgraded_execution": execution_upgrade["upgraded_execution"],
            "resume_state": execution_upgrade.get("resume_state"),
        }
    meta = external.get("meta")
    actual = (
        meta.get("ds4rt_error_ledger")
        if isinstance(meta, dict)
        else None
    )
    if actual != expected:
        raise ArtifactValidationError(
            "standalone EXL3 quantization ledger differs from the immutable plan/execution"
        )
    return expected


def _validate_manifest(
    artifact: Path,
    *,
    verify_hashes: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    ArtifactContract,
]:
    plan = _json_object(artifact / "glmrt-gptqmodel-plan.json")
    contract = _artifact_contract(plan)
    manifest = _json_object(artifact / "glmrt-gptqmodel-artifact.json")
    run = _json_object(artifact / "glmrt-gptqmodel-run.json")

    def bound(value: dict[str, Any], field: str, label: str) -> None:
        digest = value.get(field)
        body = {key: item for key, item in value.items() if key != field}
        if (
            not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or hashlib.sha256(_canonical_json(body)).hexdigest() != digest
        ):
            raise ArtifactValidationError(f"{label} digest is invalid")

    bound(plan, "plan_sha256", "plan")
    bound(manifest, "manifest_sha256", "artifact manifest")
    bound(run, "run_sha256", "run manifest")
    execution_upgrade = None
    if (artifact / EXECUTION_UPGRADE_FILENAME).exists():
        try:
            execution_upgrade = read_execution_upgrade(
                artifact,
                parent_plan_sha256=str(plan.get("plan_sha256", "")),
            )
        except ExecutionUpgradeError as error:
            raise ArtifactValidationError(str(error)) from error
    expected_upgrade_sha256 = (
        execution_upgrade["upgrade_sha256"]
        if execution_upgrade is not None
        else None
    )
    records = manifest.get("files")
    if (
        plan.get("schema") not in contract.plan_schemas
        or plan.get("recipe") != contract.recipe
        or manifest.get("schema") != contract.artifact_schema
        or manifest.get("plan_sha256") != plan.get("plan_sha256")
        or run.get("schema") != contract.run_schema
        or run.get("status") != "complete"
        or run.get("plan_sha256") != plan.get("plan_sha256")
        or run.get("artifact_manifest_sha256") != manifest.get("manifest_sha256")
        or run.get("execution_upgrade_sha256") != expected_upgrade_sha256
        or run.get("quantized_base_layers") != list(range(FIRST_ROUTED_LAYER, BASE_LAYER_END))
        or run.get("preserved_mtp_layer") != MTP_LAYER
        or not isinstance(records, dict)
        or manifest.get("file_count") != len(records)
    ):
        raise ArtifactValidationError("artifact plan/manifests are inconsistent")
    expected_paths = set(records) | {
        "glmrt-gptqmodel-artifact.json",
        "glmrt-gptqmodel-run.json",
    }
    actual_paths: set[str] = set()
    for path in artifact.rglob("*"):
        if path.is_symlink():
            raise ArtifactValidationError(f"artifact contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ArtifactValidationError(f"artifact contains an unsupported entry: {path}")
        actual_paths.add(path.relative_to(artifact).as_posix())
    if actual_paths != expected_paths:
        raise ArtifactValidationError(
            f"artifact file set differs: missing={sorted(expected_paths - actual_paths)} "
            f"unexpected={sorted(actual_paths - expected_paths)}"
        )
    total_bytes = 0
    for relative, record in records.items():
        relative = _safe_relative(relative, label="artifact manifest path")
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
            or not isinstance(record.get("sha256"), str)
            or SHA256_RE.fullmatch(record["sha256"]) is None
        ):
            raise ArtifactValidationError(f"artifact file record is invalid: {relative}")
        path = artifact / relative
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ArtifactValidationError(f"artifact file size differs: {relative}")
        total_bytes += record["bytes"]
        if verify_hashes:
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while block := source.read(8 * 1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != record["sha256"]:
                raise ArtifactValidationError(f"artifact file hash differs: {relative}")
    if manifest.get("total_bytes") != total_bytes:
        raise ArtifactValidationError("artifact manifest total_bytes differs")
    return plan, manifest, run, execution_upgrade, contract


def _compare_retained_tensor(name: str, source: TensorRecord, artifact: TensorRecord) -> str:
    if (source.dtype, source.shape, source.length) != (
        artifact.dtype,
        artifact.shape,
        artifact.length,
    ):
        raise ArtifactValidationError(f"retained native tensor metadata differs: {name}")
    digest = hashlib.sha256()
    remaining = source.length
    source_offset = source.offset
    artifact_offset = artifact.offset
    with source.file.open("rb", buffering=0) as source_file, artifact.file.open(
        "rb", buffering=0
    ) as artifact_file:
        while remaining:
            count = min(8 * 1024 * 1024, remaining)
            source_block = os.pread(source_file.fileno(), count, source_offset)
            artifact_block = os.pread(artifact_file.fileno(), count, artifact_offset)
            if len(source_block) != count or source_block != artifact_block:
                raise ArtifactValidationError(f"retained native tensor payload differs: {name}")
            digest.update(source_block)
            remaining -= count
            source_offset += count
            artifact_offset += count
    return digest.hexdigest()


def _validate_exact_source_metadata(artifact: Path, source: Path) -> list[dict[str, Any]]:
    records = []
    for name in EXACT_SOURCE_METADATA_FILES:
        source_path = source / name
        artifact_path = artifact / name
        if (
            not source_path.is_file()
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
        ):
            raise ArtifactValidationError(f"artifact source metadata is missing: {name}")
        source_digest = _hash_file(source_path)
        artifact_digest = _hash_file(artifact_path)
        if (
            source_path.stat().st_size != artifact_path.stat().st_size
            or source_digest != artifact_digest
        ):
            raise ArtifactValidationError(f"artifact source metadata differs: {name}")
        records.append(
            {
                "name": name,
                "bytes": artifact_path.stat().st_size,
                "sha256": artifact_digest,
            }
        )
    return records


def validate(
    artifact_path: Path,
    source_path: Path,
    checkpoint_root: Path,
    *,
    skip_retained_native_bytes: bool,
    verify_artifact_file_hashes: bool,
    tokenizer_attestation_path: Path | None = None,
) -> dict[str, Any]:
    artifact = _regular_root(artifact_path, label="artifact")
    source = _regular_root(source_path, label="source snapshot")
    plan, manifest, run, execution_upgrade, contract = _validate_manifest(
        artifact,
        verify_hashes=verify_artifact_file_hashes,
    )
    source_inventory = _snapshot_inventory(source, reject_symlinks=False)
    artifact_inventory = _snapshot_inventory(artifact, reject_symlinks=True)
    modules = _module_contract()
    if len(modules) != EXPECTED_MODULES:
        raise AssertionError("internal GLM-5 module geometry is inconsistent")
    quantization_config = _validate_quantization_config(
        artifact,
        modules,
        contract,
        source_config_path=source / "config.json",
    )
    quantization_provenance = _validate_quantization_provenance(
        quantization_config,
        plan,
        execution_upgrade,
    )
    source_metadata = _validate_exact_source_metadata(artifact, source)

    source_native = {f"{module}.weight" for module in modules}
    source_scale_inv = (
        {f"{module}.weight_scale_inv" for module in modules}
        if contract.source_has_block_fp8_scales
        else set()
    )
    source_native |= source_scale_inv
    if not source_native.issubset(source_inventory.tensors):
        missing = sorted(source_native - set(source_inventory.tensors))[:8]
        raise ArtifactValidationError(f"source snapshot lacks routed weights: {missing}")
    for name in source_native - source_scale_inv:
        match = NATIVE_EXPERT_RE.fullmatch(name)
        assert match is not None
        module = name.removesuffix(".weight")
        input_features, output_features = modules[module]
        record = source_inventory.tensors[name]
        if record.shape != (output_features, input_features):
            raise ArtifactValidationError(f"source routed weight shape differs: {name}")

    expected_exl3: dict[str, tuple[str, tuple[int, ...]]] = {}
    for name in source_scale_inv:
        record = source_inventory.tensors[name]
        module = name.removesuffix(".weight_scale_inv")
        input_features, output_features = modules[module]
        expected_shape = (
            output_features // 128,
            input_features // 128,
        )
        if record.dtype != "F32" or record.shape != expected_shape:
            raise ArtifactValidationError(
                f"source block-FP8 inverse-scale metadata differs: {name}"
            )

    for module, dimensions in modules.items():
        expected_exl3.update(
            _tensor_contract(
                module,
                *dimensions,
                exl3_bits=contract.exl3_bits,
            )
        )
    retained = set(source_inventory.tensors) - source_native
    expected_artifact = retained | set(expected_exl3)
    actual_artifact = set(artifact_inventory.tensors)
    if actual_artifact != expected_artifact:
        raise ArtifactValidationError(
            "artifact tensor namespace differs: "
            f"expected={len(expected_artifact)} actual={len(actual_artifact)} "
            f"missing={sorted(expected_artifact - actual_artifact)[:8]} "
            f"unexpected={sorted(actual_artifact - expected_artifact)[:8]}"
        )
    for name, (dtype, shape) in expected_exl3.items():
        record = artifact_inventory.tensors[name]
        if (record.dtype, record.shape) != (dtype, shape):
            raise ArtifactValidationError(
                f"EXL3 tensor metadata differs for {name}: "
                f"{record.dtype}/{record.shape} != {dtype}/{shape}"
            )
    for name in retained:
        source_record = source_inventory.tensors[name]
        artifact_record = artifact_inventory.tensors[name]
        if (source_record.dtype, source_record.shape, source_record.length) != (
            artifact_record.dtype,
            artifact_record.shape,
            artifact_record.length,
        ):
            raise ArtifactValidationError(f"retained native tensor metadata differs: {name}")

    checkpoint_join = _validate_checkpoint_artifact_join(
        checkpoint_root=checkpoint_root,
        plan=plan,
        artifact_inventory=artifact_inventory,
        modules=modules,
        contract=contract,
    )
    if checkpoint_join["tensor_bytes"] != sum(
        artifact_inventory.tensors[name].length for name in expected_exl3
    ):
        raise ArtifactValidationError(
            "checkpoint/artifact EXL3 byte totals differ"
        )

    retained_digest: str | None = None
    if not skip_retained_native_bytes:
        aggregate = hashlib.sha256()
        for name in sorted(retained):
            digest = _compare_retained_tensor(
                name,
                source_inventory.tensors[name],
                artifact_inventory.tensors[name],
            )
            aggregate.update(name.encode())
            aggregate.update(b"\0")
            aggregate.update(bytes.fromhex(digest))
        retained_digest = aggregate.hexdigest()

    plan_source = plan.get("source")
    source_config_sha256 = hashlib.sha256((source / "config.json").read_bytes()).hexdigest()
    source_index_sha256 = hashlib.sha256(
        (source / "model.safetensors.index.json").read_bytes()
    ).hexdigest()
    plan_exl3 = plan.get("exl3")
    if (
        not isinstance(plan_source, dict)
        or not isinstance(plan_exl3, dict)
        or plan_source.get("config_sha256") != source_config_sha256
        or plan_source.get("index_sha256") != source_index_sha256
        or plan_exl3.get("bits") != contract.exl3_bits
        or plan_exl3.get("codebook") != "mcg"
        or plan_exl3.get("module_include") != [MODULE_INCLUDE]
    ):
        raise ArtifactValidationError(
            "artifact plan is not bound to this source/EXL3 recipe"
        )
    tokenizer_evidence = _validate_tokenizer_evidence(
        plan=plan,
        source=source,
        attestation_path=tokenizer_attestation_path,
    )

    exl3_tensor_bytes = sum(
        artifact_inventory.tensors[name].length for name in expected_exl3
    )
    retained_tensor_bytes = sum(
        artifact_inventory.tensors[name].length for name in retained
    )
    model_config_path = artifact / "config.json"
    model_config = _json_object(model_config_path)
    embedded_quantization = model_config.get("quantization_config")
    if not isinstance(embedded_quantization, dict):
        raise ArtifactValidationError(
            "artifact config lost its quantization declaration"
        )
    quantize_config_path = artifact / "quantize_config.json"
    report = {
        "schema": contract.validation_schema,
        "status": "accepted",
        "model_id": contract.model_id,
        "artifact": os.fspath(artifact),
        "source_snapshot": os.fspath(source),
        "plan_sha256": plan["plan_sha256"],
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "run_sha256": run["run_sha256"],
        "execution_upgrade_sha256": (
            execution_upgrade["upgrade_sha256"]
            if execution_upgrade is not None
            else None
        ),
        "source_tensors": len(source_inventory.tensors),
        "retained_native_tensors": len(retained),
        "quantized_modules": len(modules),
        "exl3_tensors": len(expected_exl3),
        "artifact_tensors": len(artifact_inventory.tensors),
        "artifact_shards": len(artifact_inventory.files),
        "retained_native_tensor_bytes": retained_tensor_bytes,
        "retained_native_bytes_verified": not skip_retained_native_bytes,
        "retained_native_content_sha256": retained_digest,
        "exl3_tensor_bytes": exl3_tensor_bytes,
        "quantization_config": {
            "model_config_bytes": model_config_path.stat().st_size,
            "embedded_fields": sorted(embedded_quantization),
            "standalone_bytes": quantize_config_path.stat().st_size,
            "sha256": _hash_file(quantize_config_path),
            "tensor_storage_entries": len(quantization_config["tensor_storage"]),
            "stored_tensor_descriptions": sum(
                len(entry["stored_tensors"])
                for entry in quantization_config["tensor_storage"].values()
            ),
            "ledger_provenance_sha256": hashlib.sha256(
                _canonical_json(quantization_provenance)
            ).hexdigest(),
        },
        "projection_checkpoint_bytes_verified": True,
        "projection_checkpoint": checkpoint_join,
        "tp4_resident_bytes_per_spark": contract.expected_tp4_resident_bytes,
        "artifact_manifest_file_hashes_verified": verify_artifact_file_hashes,
        "tokenizer_evidence": tokenizer_evidence,
        "source_metadata": source_metadata,
    }
    report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
    return report


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument(
        "--projection-checkpoint-dir",
        type=Path,
        required=True,
        help="complete plan-bound projection store whose bytes must match the artifact",
    )
    parser.add_argument(
        "--skip-retained-native-bytes",
        action="store_true",
        help="check retained tensor metadata but skip the acceptance-grade byte comparison",
    )
    parser.add_argument(
        "--verify-artifact-file-hashes",
        action="store_true",
        help="rehash every artifact file in addition to checking its bound manifest",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tokenizer-attestation",
        type=Path,
        help="required only for a legacy plan that omitted tokenizer source files",
    )
    args = parser.parse_args()
    report = validate(
        args.artifact,
        args.source_snapshot,
        args.projection_checkpoint_dir,
        skip_retained_native_bytes=args.skip_retained_native_bytes,
        verify_artifact_file_hashes=args.verify_artifact_file_hashes,
        tokenizer_attestation_path=args.tokenizer_attestation,
    )
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
