#!/usr/bin/env python3
"""Verify the exact public Hub revision of a calibrated GLM-5 EXL3 model."""

from __future__ import annotations

import argparse
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable
import urllib.request

from stage_glm52_exl3_hf_snapshot import (
    GLM53_MODEL_ID,
    MODEL_ID,
    SUPPORTED_MODEL_IDS,
    StagingError,
    _atomic_text,
    _canonical_json,
    _fsync_directory,
    _install_file,
    _json_object,
    _model_cache_root,
    _publication_evidence,
    _safe_relative,
)
from validate_glm52_exl3_artifact import _compact_exl3_declaration


SCHEMA = "glmrt-glm52-exl3-hub-verification-v1"
GLM53_SCHEMA = "glmrt-glm5-exl3-hub-verification-v1"
REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_TENSOR_STORAGE_ENTRIES = 57_600
EXPECTED_STORED_TENSOR_DESCRIPTIONS = 230_400
EXL3_MCG_MULTIPLIER = 0xCBAC1FED
MAX_COMPACT_CONFIG_BYTES = 128 * 1024
MAX_HUB_PAGE_BYTES = 16 * 1024 * 1024
HUB_PAGE_ERROR_MARKERS = (
    "config.json is too large",
    "config.json file is too large",
    "invalid config.json",
    "failed to parse config.json",
    "error loading config.json",
    "unable to load config.json",
    "this file is too large to display",
    "repository not found",
    "model not found",
)


class HubVerificationError(RuntimeError):
    """The remote revision does not exactly match the accepted publication."""


class _VisibleHtml(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hidden_depth = 0
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag.lower() in {"script", "style", "noscript"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.text.append(data)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def git_blob_oid(path: Path) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fetch_hub_page(url: str, token: bool | str | None) -> dict[str, Any]:
    headers = {"User-Agent": "glmrt-hub-publication-verifier/1"}
    if isinstance(token, str) and token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60.0) as response:
        payload = response.read(MAX_HUB_PAGE_BYTES + 1)
        return {
            "status": response.status,
            "url": response.geturl(),
            "content_type": response.headers.get_content_type(),
            "body": payload,
        }


def verify_hub_pages(
    *,
    model_id: str,
    revision: str,
    page_fetcher: Callable[[str, bool | str | None], dict[str, Any]],
    token: bool | str | None,
) -> list[dict[str, Any]]:
    root = f"https://huggingface.co/{model_id}"
    targets = (
        ("model", root),
        ("revision-tree", f"{root}/tree/{revision}"),
        ("compact-config", f"{root}/blob/{revision}/config.json"),
    )
    results = []
    for kind, url in targets:
        try:
            response = page_fetcher(url, token)
        except Exception as error:
            raise HubVerificationError(
                f"Hub website {kind} page could not be fetched"
            ) from error
        payload = response.get("body") if isinstance(response, dict) else None
        if (
            not isinstance(response, dict)
            or response.get("status") != 200
            or response.get("content_type") != "text/html"
            or not isinstance(payload, bytes)
            or not payload
            or len(payload) > MAX_HUB_PAGE_BYTES
        ):
            raise HubVerificationError(f"Hub website {kind} page is invalid")
        try:
            source = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise HubVerificationError(
                f"Hub website {kind} page is not UTF-8"
            ) from error
        parser = _VisibleHtml()
        parser.feed(source)
        visible = " ".join(parser.text).casefold()
        marker = next(
            (candidate for candidate in HUB_PAGE_ERROR_MARKERS if candidate in visible),
            None,
        )
        if marker is not None:
            raise HubVerificationError(
                f"Hub website {kind} page reports an error: {marker}"
            )
        results.append(
            {
                "kind": kind,
                "url": url,
                "resolved_url": response.get("url"),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "visible_error_markers": [],
            }
        )
    return results


def validate_hub_api_config(api_config: Any, local_config: dict[str, Any]) -> dict[str, Any]:
    local_quant = local_config.get("quantization_config")
    remote_quant = api_config.get("quantization_config") if isinstance(api_config, dict) else None
    if (
        not isinstance(api_config, dict)
        or not isinstance(local_quant, dict)
        or not isinstance(remote_quant, dict)
        or remote_quant.get("quant_method") != "exl3"
        or remote_quant.get("format") != "exl3"
        or remote_quant.get("bits") != local_quant.get("bits")
    ):
        raise HubVerificationError(
            "Hub API did not parse the compact EXL3 config.json declaration"
        )
    for key in ("architectures", "model_type", "num_experts_per_tok"):
        if key in local_config and api_config.get(key) != local_config[key]:
            raise HubVerificationError(f"Hub API config field differs: {key}")
    return {
        "architectures": api_config.get("architectures"),
        "model_type": api_config.get("model_type"),
        "num_experts_per_tok": api_config.get("num_experts_per_tok"),
        "quantization_config": {
            "quant_method": remote_quant["quant_method"],
            "format": remote_quant["format"],
            "bits": remote_quant["bits"],
        },
    }


def materialize_hub_cache(
    *,
    publication: Path,
    entries: dict[str, dict[str, Any]],
    remote: dict[str, Any],
    model_id: str,
    revision: str,
    hf_home: Path,
    link_mode: str,
) -> dict[str, Any]:
    cache = _model_cache_root(hf_home.expanduser().resolve(), model_id)
    blobs = cache / "blobs"
    snapshot = cache / "snapshots" / revision
    staging = cache / "snapshots" / f".{revision}.{os.getpid()}.tmp"
    if staging.exists() or staging.is_symlink():
        raise HubVerificationError(f"stale Hub-cache staging path exists: {staging}")
    blobs.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    blob_names: dict[str, str] = {}
    try:
        for path, expected in sorted(entries.items()):
            sibling = remote[path]
            lfs = field(sibling, "lfs")
            blob_name = (
                field(lfs, "sha256") if lfs is not None else field(sibling, "blob_id")
            )
            if not isinstance(blob_name, str) or REVISION_RE.fullmatch(blob_name) is None:
                raise HubVerificationError(f"Hub blob identity is invalid: {path}")
            blob_names[path] = blob_name
            source = publication.joinpath(*_safe_relative(path).parts)
            try:
                _install_file(source, blobs / blob_name, mode=link_mode, expected=expected)
            except (OSError, StagingError) as error:
                raise HubVerificationError(
                    f"could not materialize Hub cache blob: {path}"
                ) from error
            link = staging.joinpath(*_safe_relative(path).parts)
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(blobs / blob_name, link.parent))
        if snapshot.exists():
            actual = {
                path.relative_to(snapshot).as_posix()
                for path in snapshot.rglob("*")
                if path.is_file() or path.is_symlink()
            }
            if actual != set(entries):
                raise HubVerificationError(
                    "existing materialized Hub snapshot file set differs"
                )
            for path, blob_name in blob_names.items():
                link = snapshot.joinpath(*_safe_relative(path).parts)
                if (
                    not link.is_symlink()
                    or link.resolve(strict=True) != (blobs / blob_name).resolve(strict=True)
                ):
                    raise HubVerificationError(
                        f"existing materialized Hub snapshot link differs: {path}"
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
        _atomic_text(cache / "refs" / "main", revision)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        from huggingface_hub import snapshot_download

        locally_resolved = Path(
            snapshot_download(
                repo_id=model_id,
                revision=revision,
                cache_dir=hf_home / "hub",
                local_files_only=True,
            )
        ).resolve(strict=True)
    except Exception as error:
        raise HubVerificationError(
            "materialized Hub cache cannot resolve the exact revision locally"
        ) from error
    if locally_resolved != snapshot.resolve(strict=True):
        raise HubVerificationError(
            "materialized Hub cache resolved to an unexpected local snapshot"
        )
    return {
        "cache_root": os.fspath(cache),
        "snapshot": os.fspath(snapshot),
        "revision": revision,
        "ref": "main",
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries.values()),
        "link_mode": link_mode,
        "download_equivalent_layout": True,
        "local_only_snapshot_resolution": os.fspath(locally_resolved),
    }


def validate_downloaded_quantization_configs(
    *,
    config_payload: bytes,
    quantize_config_payload: bytes,
    model_id: str,
    expected_tensor_storage_entries: int = EXPECTED_TENSOR_STORAGE_ENTRIES,
    expected_stored_tensor_descriptions: int = EXPECTED_STORED_TENSOR_DESCRIPTIONS,
) -> dict[str, Any]:
    try:
        config = json.loads(config_payload)
        external = json.loads(quantize_config_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HubVerificationError("downloaded quantization configuration is invalid JSON") from error
    embedded = config.get("quantization_config") if isinstance(config, dict) else None
    storage = external.get("tensor_storage") if isinstance(external, dict) else None
    external_declaration = dict(external) if isinstance(external, dict) else None
    if isinstance(external_declaration, dict):
        external_declaration.pop("tensor_storage", None)
    minimal_declaration = (
        _compact_exl3_declaration(external)
        if isinstance(external, dict)
        else None
    )
    expected_bits = 3.0 if model_id == MODEL_ID else 4.0
    meta = external.get("meta") if isinstance(external, dict) else None
    ledger = (
        meta.get("ds4rt_error_ledger")
        if isinstance(meta, dict)
        else None
    )
    if (
        not isinstance(embedded, dict)
        or not isinstance(storage, dict)
        or not storage
        or (
            embedded != minimal_declaration
            if model_id == GLM53_MODEL_ID
            else embedded not in (minimal_declaration, external_declaration)
        )
        or len(config_payload) > MAX_COMPACT_CONFIG_BYTES
        or embedded.get("quant_method") != "exl3"
        or embedded.get("bits") != expected_bits
    ):
        raise HubVerificationError(
            "compact and standalone EXL3 configurations do not agree"
        )
    if model_id == GLM53_MODEL_ID and (
        not isinstance(ledger, dict)
        or not isinstance(ledger.get("family_join"), dict)
        or not isinstance(ledger.get("run"), dict)
    ):
        raise HubVerificationError(
            "standalone GLM-5.3 quantization ledger is missing"
        )
    if (
        isinstance(expected_tensor_storage_entries, bool)
        or not isinstance(expected_tensor_storage_entries, int)
        or expected_tensor_storage_entries <= 0
        or isinstance(expected_stored_tensor_descriptions, bool)
        or not isinstance(expected_stored_tensor_descriptions, int)
        or expected_stored_tensor_descriptions <= 0
    ):
        raise HubVerificationError("downloaded EXL3 storage expectations are invalid")
    if len(storage) != expected_tensor_storage_entries:
        raise HubVerificationError(
            "downloaded standalone EXL3 tensor_storage inventory is incomplete: "
            f"{len(storage)} != {expected_tensor_storage_entries}"
        )
    stored_tensor_descriptions = 0
    for module, entry in storage.items():
        if (
            not isinstance(module, str)
            or not module
            or not isinstance(entry, dict)
            or set(entry)
            != {
                "stored_tensors",
                "quant_format",
                "bits_per_weight",
                "mcg_multiplier",
            }
            or entry.get("quant_format") != "exl3"
            or entry.get("bits_per_weight") != int(expected_bits)
            or entry.get("mcg_multiplier") != EXL3_MCG_MULTIPLIER
        ):
            raise HubVerificationError(
                f"downloaded standalone EXL3 tensor_storage entry is invalid: {module!r}"
            )
        stored = entry.get("stored_tensors")
        if not isinstance(stored, dict) or len(stored) != 4:
            raise HubVerificationError(
                f"downloaded standalone EXL3 stored tensor set is incomplete: {module}"
            )
        for tensor, metadata in stored.items():
            shape = metadata.get("shape") if isinstance(metadata, dict) else None
            torch_dtype = (
                metadata.get("torch_dtype") if isinstance(metadata, dict) else None
            )
            if (
                not isinstance(tensor, str)
                or not tensor
                or not isinstance(metadata, dict)
                or set(metadata) != {"shape", "torch_dtype"}
                or not isinstance(shape, list)
                or any(
                    isinstance(dimension, bool)
                    or not isinstance(dimension, int)
                    or dimension < 0
                    for dimension in shape
                )
                or torch_dtype not in {"int16", "float16", "int32"}
            ):
                raise HubVerificationError(
                    f"downloaded standalone EXL3 tensor metadata is invalid: {tensor!r}"
                )
        stored_tensor_descriptions += len(stored)
    if stored_tensor_descriptions != expected_stored_tensor_descriptions:
        raise HubVerificationError(
            "downloaded standalone EXL3 tensor description inventory is incomplete: "
            f"{stored_tensor_descriptions} != {expected_stored_tensor_descriptions}"
        )
    return {
        "tensor_storage_entries": len(storage),
        "stored_tensor_descriptions": stored_tensor_descriptions,
        "bits": embedded["bits"],
        "config_sha256": hashlib.sha256(config_payload).hexdigest(),
        "quantize_config_sha256": hashlib.sha256(quantize_config_payload).hexdigest(),
        "ledger_provenance_sha256": (
            hashlib.sha256(_canonical_json(ledger)).hexdigest()
            if isinstance(ledger, dict)
            else None
        ),
    }


def verify(
    *,
    publication_report_path: Path,
    revision: str,
    api: Any,
    page_fetcher: Callable[[str, bool | str | None], dict[str, Any]],
    token: bool | str | None = None,
    hf_home: Path | None = None,
    link_mode: str = "hardlink",
    expected_tensor_storage_entries: int = EXPECTED_TENSOR_STORAGE_ENTRIES,
    expected_stored_tensor_descriptions: int = EXPECTED_STORED_TENSOR_DESCRIPTIONS,
) -> dict[str, Any]:
    if link_mode not in {"hardlink", "copy"}:
        raise HubVerificationError("Hub-cache link mode is invalid")
    report_path = publication_report_path.expanduser()
    if report_path.is_symlink():
        raise HubVerificationError("publication report is a symbolic link")
    report_path = report_path.resolve(strict=True)
    publication_report = _json_object(report_path)
    model_id = publication_report.get("model_id")
    if model_id not in SUPPORTED_MODEL_IDS:
        raise HubVerificationError("publication report has an unsupported model ID")
    publication = Path(str(publication_report.get("output", ""))).expanduser().resolve(
        strict=True
    )
    try:
        entries, publication_identity, publication_report = _publication_evidence(
            report_path,
            publication=publication,
            model_id=model_id,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise HubVerificationError("local publication evidence is invalid") from error
    expected = {entry["path"]: entry for entry in entries}

    info = api.model_info(
        model_id,
        revision=revision,
        files_metadata=True,
        token=token,
    )
    resolved_revision = field(info, "sha")
    siblings = field(info, "siblings")
    if (
        field(info, "id") != model_id
        or field(info, "private") is not False
        or field(info, "gated") not in {None, False}
        or not isinstance(resolved_revision, str)
        or REVISION_RE.fullmatch(resolved_revision) is None
        or not isinstance(siblings, list)
    ):
        raise HubVerificationError(
            "Hub model identity, visibility, or resolved revision is invalid"
        )

    remote: dict[str, Any] = {}
    for sibling in siblings:
        path = field(sibling, "path", field(sibling, "rfilename"))
        size = field(sibling, "size")
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or path in remote
        ):
            raise HubVerificationError("Hub file metadata is malformed")
        remote[path] = sibling
    if set(remote) != set(expected):
        raise HubVerificationError(
            "Hub file inventory differs: "
            f"missing={sorted(set(expected) - set(remote))} "
            f"unexpected={sorted(set(remote) - set(expected))}"
        )

    verified: list[dict[str, Any]] = []
    for path, expected_entry in sorted(expected.items()):
        sibling = remote[path]
        size = field(sibling, "size")
        blob_id = field(sibling, "blob_id")
        if size != expected_entry["bytes"]:
            raise HubVerificationError(f"Hub file size differs: {path}")
        if not isinstance(blob_id, str) or REVISION_RE.fullmatch(blob_id) is None:
            raise HubVerificationError(f"Hub Git blob identity is missing: {path}")
        lfs = field(sibling, "lfs")
        lfs_sha256 = field(lfs, "sha256") if lfs is not None else None
        if lfs is not None:
            if (
                not isinstance(lfs_sha256, str)
                or SHA256_RE.fullmatch(lfs_sha256) is None
                or lfs_sha256 != expected_entry["sha256"]
                or field(lfs, "size") != size
            ):
                raise HubVerificationError(f"Hub LFS identity differs: {path}")
            method = "lfs-sha256"
            remote_content_identity = lfs_sha256
        else:
            local_git_oid = git_blob_oid(publication / path)
            if blob_id != local_git_oid:
                raise HubVerificationError(f"Hub Git blob differs: {path}")
            method = "git-blob-sha1"
            remote_content_identity = blob_id
        verified.append(
            {
                "path": path,
                "bytes": size,
                "sha256": expected_entry["sha256"],
                "remote_content_identity": remote_content_identity,
                "method": method,
            }
        )

    config_payload = (publication / "config.json").read_bytes()
    quantize_config_payload = (publication / "quantize_config.json").read_bytes()
    quantization_config = validate_downloaded_quantization_configs(
        config_payload=config_payload,
        quantize_config_payload=quantize_config_payload,
        model_id=model_id,
        expected_tensor_storage_entries=expected_tensor_storage_entries,
        expected_stored_tensor_descriptions=expected_stored_tensor_descriptions,
    )
    local_config = json.loads(config_payload)
    hub_api_config = validate_hub_api_config(field(info, "config"), local_config)
    hub_pages = verify_hub_pages(
        model_id=model_id,
        revision=resolved_revision,
        page_fetcher=page_fetcher,
        token=token,
    )
    materialized_cache = (
        materialize_hub_cache(
            publication=publication,
            entries=expected,
            remote=remote,
            model_id=model_id,
            revision=resolved_revision,
            hf_home=hf_home,
            link_mode=link_mode,
        )
        if hf_home is not None
        else None
    )

    body = {
        "schema": GLM53_SCHEMA if model_id != MODEL_ID else SCHEMA,
        "status": "accepted",
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "visibility": "public",
        "gated": False,
        "publication": publication_identity,
        "publication_sha256": publication_report["publication_sha256"],
        "files": verified,
        "file_bytes": sum(entry["bytes"] for entry in verified),
        "remote_identity_methods": sorted({entry["method"] for entry in verified}),
        "hub_api_config": hub_api_config,
        "hub_pages": hub_pages,
        "quantization_config": quantization_config,
        "materialized_cache": materialized_cache,
        "full_model_redownloaded": False,
    }
    if not math.isfinite(float(body["file_bytes"])):
        raise HubVerificationError("remote byte total is invalid")
    return {
        **body,
        "report_sha256": hashlib.sha256(_canonical_json(body)).hexdigest(),
    }


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
    parser.add_argument("--publication-report", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
        help="materialize the resolved Hub commit into this standard HF cache",
    )
    parser.add_argument("--link-mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    report = verify(
        publication_report_path=args.publication_report,
        revision=args.revision,
        api=HfApi(),
        page_fetcher=fetch_hub_page,
        hf_home=args.hf_home,
        link_mode=args.link_mode,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
