#!/usr/bin/env python3
"""Concurrently distribute a staged GLM-5 EXL3 snapshot over RDMA."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from stage_glm52_exl3_hf_snapshot import SCHEMA, _model_cache_root


HOST_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
REVISION_RE = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_HOSTS = ("ostrich", "dodo", "emu", "kiwi")


REMOTE_VERIFY = r'''
import hashlib,json,os,re,sys
from pathlib import Path,PurePosixPath
root=Path(sys.argv[1]).resolve()
revision=sys.argv[2]
model_id=sys.argv[3]
verify_hashes=sys.argv[4]=="1"
manifest=json.loads((root/"glmrt-manifests"/(revision+".json")).read_text())
body={key:manifest[key] for key in ("schema","model_id","files","qualification","quant_evidence","quant_evidence_sha256")}
canonical=json.dumps(body,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
if (manifest.get("schema")!="glmrt-hf-staged-snapshot-v1" or
    manifest.get("model_id")!=model_id or manifest.get("revision")!=revision or
    re.fullmatch(r"[0-9a-f]{64}",str(manifest.get("quant_evidence_sha256"))) is None or
    hashlib.sha256(canonical).hexdigest()!=revision):
    raise SystemExit("remote staged manifest contract mismatch")
if (root/"refs"/"main").read_text().strip()!=revision:
    raise SystemExit("remote refs/main mismatch")
snapshot=root/"snapshots"/revision
if not snapshot.is_dir() or snapshot.is_symlink():
    raise SystemExit("remote snapshot is missing or unsafe")
expected=set()
verified=set()
total=0
for entry in manifest["files"]:
    raw=entry.get("path"); digest=entry.get("sha256"); size=entry.get("bytes")
    if not isinstance(raw,str) or "\\" in raw:
        raise SystemExit("remote staged path is invalid")
    relative=PurePosixPath(raw)
    if relative.is_absolute() or any(part in {"",".",".."} for part in relative.parts):
        raise SystemExit("remote staged path is unsafe")
    if re.fullmatch(r"[0-9a-f]{64}",str(digest)) is None or not isinstance(size,int) or isinstance(size,bool) or size<0:
        raise SystemExit("remote staged file identity is invalid")
    if raw in expected: raise SystemExit("remote staged path is duplicated")
    expected.add(raw); total+=size
    link=snapshot.joinpath(*relative.parts); blob=root/"blobs"/digest
    if not link.is_symlink() or link.resolve(strict=True)!=blob.resolve(strict=True):
        raise SystemExit("remote snapshot link mismatch: "+raw)
    if not blob.is_file() or blob.is_symlink() or blob.stat().st_size!=size:
        raise SystemExit("remote blob metadata mismatch: "+digest)
    if verify_hashes and digest not in verified:
        h=hashlib.sha256()
        with blob.open("rb") as source:
            while block:=source.read(8*1024*1024): h.update(block)
        if h.hexdigest()!=digest: raise SystemExit("remote blob hash mismatch: "+digest)
        verified.add(digest)
actual={p.relative_to(snapshot).as_posix() for p in snapshot.rglob("*") if p.is_file() or p.is_symlink()}
if actual!=expected: raise SystemExit("remote snapshot file set mismatch")
if manifest.get("total_bytes")!=total: raise SystemExit("remote staged byte total mismatch")
qualification=manifest["qualification"]
evidence_records=[qualification]
quant_evidence=manifest.get("quant_evidence")
if quant_evidence is not None: evidence_records.append(quant_evidence)
for evidence in evidence_records:
    raw=evidence.get("path") if isinstance(evidence,dict) else None
    relative=PurePosixPath(raw) if isinstance(raw,str) and "\\" not in raw else None
    size=evidence.get("bytes") if isinstance(evidence,dict) else None
    digest=evidence.get("sha256") if isinstance(evidence,dict) else None
    if (relative is None or relative.is_absolute() or
        any(part in {"",".",".."} for part in relative.parts) or
        not isinstance(size,int) or isinstance(size,bool) or size<0 or
        re.fullmatch(r"[0-9a-f]{64}",str(digest)) is None):
        raise SystemExit("remote qualification identity mismatch")
    qpath=(root/"glmrt-qualifications"/revision).joinpath(*relative.parts)
    if not qpath.is_file() or qpath.is_symlink() or qpath.stat().st_size!=size:
        raise SystemExit("remote qualification evidence mismatch")
    if verify_hashes:
        h=hashlib.sha256(qpath.read_bytes()).hexdigest()
        if h!=digest: raise SystemExit("remote qualification hash mismatch")
print(json.dumps({"revision":revision,"files":len(expected),"bytes":total,"verified_blobs":len(verified)},sort_keys=True))
'''


@dataclass(frozen=True)
class LocalContract:
    root: Path
    revision: str
    files: int
    bytes: int


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _hosts(raw: str | tuple[str, ...]) -> tuple[str, ...]:
    values = raw.split(",") if isinstance(raw, str) else list(raw)
    hosts = tuple(value.strip() for value in values if value.strip())
    if not hosts or len(hosts) != len(set(hosts)):
        raise ValueError("--hosts must contain unique host names")
    if any(HOST_RE.fullmatch(host) is None for host in hosts):
        raise ValueError("--hosts contains an unsafe host name")
    return hosts


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _local_contract(hf_home: Path, model_id: str) -> LocalContract:
    root = _model_cache_root(hf_home.expanduser().resolve(), model_id)
    revision = (root / "refs" / "main").read_text(encoding="utf-8").strip()
    if REVISION_RE.fullmatch(revision) is None:
        raise ValueError("local staged refs/main is not a content revision")
    manifest = json.loads(
        (root / "glmrt-manifests" / f"{revision}.json").read_text(encoding="utf-8")
    )
    body = {
        key: manifest.get(key)
        for key in (
            "schema",
            "model_id",
            "files",
            "qualification",
            "quant_evidence",
            "quant_evidence_sha256",
        )
    }
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("model_id") != model_id
        or manifest.get("revision") != revision
        or REVISION_RE.fullmatch(str(manifest.get("quant_evidence_sha256", "")))
        is None
        or hashlib.sha256(_canonical_json(body)).hexdigest() != revision
        or not isinstance(manifest.get("files"), list)
    ):
        raise ValueError("local staged manifest is invalid")
    snapshot = root / "snapshots" / revision
    expected: set[str] = set()
    total = 0
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise ValueError("local staged file identity is invalid")
        raw = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("bytes")
        if not isinstance(raw, str) or "\\" in raw:
            raise ValueError("local staged path is invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("local staged path is unsafe")
        if (
            REVISION_RE.fullmatch(str(digest)) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or raw in expected
        ):
            raise ValueError("local staged file identity is invalid")
        expected.add(raw)
        total += size
        link = snapshot.joinpath(*relative.parts)
        blob = root / "blobs" / digest
        if (
            not link.is_symlink()
            or link.resolve(strict=True) != blob.resolve(strict=True)
            or not blob.is_file()
            or blob.is_symlink()
            or blob.stat().st_size != size
        ):
            raise ValueError(f"local staged snapshot entry differs: {raw}")
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != expected or total != manifest.get("total_bytes"):
        raise ValueError("local staged snapshot inventory differs")
    qualification = manifest.get("qualification")
    quant_evidence = manifest.get("quant_evidence")
    if not isinstance(qualification, dict) or (
        quant_evidence is not None and not isinstance(quant_evidence, dict)
    ):
        raise ValueError("local staged snapshot has no qualification")
    for identity in (qualification, quant_evidence):
        if identity is None:
            continue
        raw = identity.get("path")
        size = identity.get("bytes")
        digest = identity.get("sha256")
        if not isinstance(raw, str) or "\\" in raw:
            raise ValueError("local staged qualification identity differs")
        relative = PurePosixPath(raw)
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or REVISION_RE.fullmatch(str(digest)) is None
        ):
            raise ValueError("local staged qualification identity differs")
        evidence = (root / "glmrt-qualifications" / revision).joinpath(
            *relative.parts
        )
        if (
            not evidence.is_file()
            or evidence.is_symlink()
            or evidence.stat().st_size != size
        ):
            raise ValueError("local staged qualification evidence differs")
    return LocalContract(root=root, revision=revision, files=len(expected), bytes=total)


def _remote_hf_home(host: str) -> Path:
    result = subprocess.run(
        _ssh_command(
            host,
            [
                "python3",
                "-c",
                "import os,pathlib; print(pathlib.Path(os.environ.get('HF_HOME', pathlib.Path.home()/'.cache'/'huggingface')).expanduser().resolve())",
            ],
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return Path(result.stdout.strip())


def _ssh_command(host: str, command: list[str]) -> list[str]:
    """Pass one shell-quoted remote command to OpenSSH.

    OpenSSH concatenates all arguments after the host before handing them to a
    remote shell.  Supplying ``python3``, ``-c``, and source as separate local
    argv entries therefore loses the source argument's quoting.  Build one
    explicit remote command string so Python snippets and paths survive that
    boundary exactly.
    """

    if HOST_RE.fullmatch(host) is None or not command or any(
        not isinstance(value, str) or not value for value in command
    ):
        raise ValueError("unsafe SSH command")
    return ["ssh", "-o", "BatchMode=yes", host, shlex.join(command)]


def _sync_host(
    host: str,
    contract: LocalContract,
    *,
    model_id: str,
    verify_hashes: bool,
) -> dict[str, Any]:
    remote_hf_home = _remote_hf_home(host)
    remote_root = _model_cache_root(remote_hf_home, model_id)
    subprocess.run(
        _ssh_command(host, ["mkdir", "-p", os.fspath(remote_root)]),
        check=True,
    )
    subprocess.run(
        [
            "rdmasync",
            "-aH",
            "--rdma=required",
            os.fspath(contract.root) + "/",
            f"{host}:{remote_root}/",
        ],
        check=True,
    )
    result = subprocess.run(
        _ssh_command(
            host,
            [
                "python3",
                "-c",
                REMOTE_VERIFY,
                os.fspath(remote_root),
                contract.revision,
                model_id,
                "1" if verify_hashes else "0",
            ],
        ),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    report = json.loads(result.stdout)
    return {"host": host, **report}


def sync(
    *,
    model_id: str,
    hf_home: Path,
    hosts: tuple[str, ...],
    verify_hashes: bool,
) -> dict[str, Any]:
    contract = _local_contract(hf_home, model_id)
    reports: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(hosts)) as workers:
        futures = {
            workers.submit(
                _sync_host,
                host,
                contract,
                model_id=model_id,
                verify_hashes=verify_hashes,
            ): host
            for host in hosts
        }
        for future in as_completed(futures):
            host = futures[future]
            try:
                reports.append(future.result())
            except BaseException as error:
                raise RuntimeError(f"EXL3 snapshot sync failed for {host}: {error}") from error
    reports.sort(key=lambda report: report["host"])
    return {
        "schema": "glmrt-hf-snapshot-sync-v1",
        "status": "complete",
        "model_id": model_id,
        "revision": contract.revision,
        "files": contract.files,
        "bytes": contract.bytes,
        "remote_payload_hashes_verified": verify_hashes,
        "hosts": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        required=True,
        help="exact staged Hugging Face repository ID",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")),
    )
    parser.add_argument("--hosts", default=",".join(DEFAULT_HOSTS))
    parser.add_argument(
        "--skip-remote-payload-hashes",
        action="store_true",
        help="verify remote file/link metadata without rereading every received blob",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = sync(
        model_id=args.model_id,
        hf_home=args.hf_home,
        hosts=_hosts(args.hosts),
        verify_hashes=not args.skip_remote_payload_hashes,
    )
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
