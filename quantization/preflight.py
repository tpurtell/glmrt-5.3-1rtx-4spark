#!/usr/bin/env python3
"""Fail-closed environment preflight for GLMRT quantization containers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import sysconfig
import tempfile
from typing import Any


IMAGE_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}")
PLATFORM_MACHINES = {
    "linux/amd64": {"amd64", "x86_64"},
    "linux/arm64": {"aarch64", "arm64"},
}
ROLE_CONTRACTS = {
    "coordinator": ("linux/amd64", "120"),
}
PACKAGE_VERSIONS = {
    "gptqmodel": "7.3.5",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
}
RUST_VERSION = "1.97.1"
UV_VERSION = "0.11.30"


class PreflightError(RuntimeError):
    """The process is not running in the qualified quantization environment."""


def report_identity_sha256(report: dict[str, Any]) -> str:
    """Hash immutable preflight facts without binding observation time.

    ``generated_at`` records when the same environment was observed, not which
    environment was observed.  Excluding only that field lets a resumed
    container prove the same image, packages, toolchain, and physical GPUs
    without making every fresh preflight report a different execution plan.
    """

    stable = {key: value for key, value in report.items() if key != "generated_at"}
    encoded = json.dumps(
        stable,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_capability(value: str) -> tuple[int, int]:
    if not re.fullmatch(r"[0-9]{2,3}", value):
        raise PreflightError(f"invalid GLMRT CUDA architecture: {value!r}")
    return int(value[:-1]), int(value[-1])


def _validate_platform(target_platform: str, machine: str) -> None:
    accepted = PLATFORM_MACHINES.get(target_platform)
    if accepted is None:
        raise PreflightError(f"unsupported target platform: {target_platform!r}")
    if machine.lower() not in accepted:
        raise PreflightError(
            f"target platform {target_platform} requires {sorted(accepted)}, "
            f"but the kernel reports {machine!r}"
        )


def _parse_nvidia_smi(output: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise PreflightError(f"unexpected nvidia-smi row: {line!r}")
        index, uuid, name, driver, power_limit, compute_capability = fields
        try:
            major_text, minor_text = compute_capability.split(".", 1)
            capability = [int(major_text), int(minor_text)]
            parsed_power = (
                None if power_limit in {"N/A", "[N/A]"} else float(power_limit)
            )
            parsed_index = int(index)
        except ValueError as exc:
            raise PreflightError(f"invalid nvidia-smi row: {line!r}") from exc
        rows.append(
            {
                "index": parsed_index,
                "uuid": uuid,
                "name": name,
                "driver_version": driver,
                "power_limit_watts": parsed_power,
                "compute_capability": capability,
            }
        )
    return sorted(rows, key=lambda row: row["index"])


def _nvidia_smi() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,power.limit,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreflightError("nvidia-smi is not available in the container") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"nvidia-smi failed: {detail}")
    return _parse_nvidia_smi(result.stdout)


def _tool_version(command: list[str], expected_prefix: str) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise PreflightError(f"required build tool is unavailable: {command[0]}") from exc
    value = (result.stdout or result.stderr).strip()
    if result.returncode != 0 or not value.startswith(expected_prefix):
        raise PreflightError(
            f"expected {' '.join(command)} output beginning {expected_prefix!r}, found {value!r}"
        )
    return value


def _verify_source(source: Path, lock_path: Path, verifier: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            sys.executable,
            os.fspath(verifier),
            "--source",
            os.fspath(source),
            "--lock",
            os.fspath(lock_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"GPTQModel source verification failed: {detail}")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read verified GPTQModel lock: {exc}") from exc
    spec = importlib.util.find_spec("gptqmodel")
    if spec is None or spec.origin is None:
        raise PreflightError("the verified GPTQModel package is not importable")
    module_path = Path(spec.origin).resolve()
    if not module_path.is_relative_to(source.resolve()):
        raise PreflightError(
            f"GPTQModel resolves outside verified source: {module_path} not under {source}"
        )
    return lock


def _verify_cusparselt_normalization(normalizer: Path) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, os.fspath(normalizer), "--verify"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreflightError(f"cuSPARSELt metadata verification failed: {detail}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise PreflightError(
            f"cuSPARSELt metadata verifier returned invalid JSON: {result.stdout!r}"
        ) from exc
    machine = platform.machine().lower()
    expected_status = (
        "normalized-and-verified"
        if machine in PLATFORM_MACHINES["linux/arm64"]
        else "not-applicable"
    )
    if report.get("status") != expected_status:
        raise PreflightError(
            "cuSPARSELt metadata verifier returned unexpected status: "
            f"expected {expected_status!r}, found {report.get('status')!r}"
        )
    return report


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as out:
            temporary = Path(out.name)
            out.write(encoded)
            out.flush()
            os.fsync(out.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    native_lock_name = (
        "requirements.arm64.lock"
        if platform.machine().lower() in PLATFORM_MACHINES["linux/arm64"]
        else "requirements.amd64.lock"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=os.environ.get("GLMRT_QUANT_ROLE"))
    parser.add_argument(
        "--target-platform", default=os.environ.get("GLMRT_QUANT_TARGET_PLATFORM")
    )
    parser.add_argument("--cuda-arch", default=os.environ.get("GLMRT_QUANT_CUDA_ARCH"))
    parser.add_argument(
        "--min-gpus", type=int, default=int(os.environ.get("GLMRT_QUANT_MIN_GPUS", "1"))
    )
    parser.add_argument(
        "--python-version",
        default=os.environ.get("GLMRT_QUANT_PYTHON_VERSION", "3.14.6"),
    )
    parser.add_argument(
        "--source", type=Path, default=root / "third_party" / "gptqmodel"
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=root / "third_party" / "gptqmodel.lock.json",
    )
    parser.add_argument(
        "--source-verifier",
        type=Path,
        default=root / "scripts" / "verify-gptqmodel-source.py",
    )
    parser.add_argument(
        "--requirements-lock",
        type=Path,
        default=Path(
            os.environ.get(
                "GLMRT_QUANT_REQUIREMENTS_LOCK",
                root / "quantization" / native_lock_name,
            )
        ),
    )
    parser.add_argument(
        "--requirements-sha256",
        default=os.environ.get("GLMRT_QUANT_REQUIREMENTS_SHA256"),
    )
    parser.add_argument(
        "--build-requirements-lock",
        type=Path,
        default=root / "quantization" / "build-requirements.lock",
    )
    parser.add_argument(
        "--build-requirements-sha256",
        default=os.environ.get("GLMRT_QUANT_BUILD_REQUIREMENTS_SHA256"),
    )
    parser.add_argument(
        "--cusparselt-normalizer",
        type=Path,
        default=root / "quantization" / "normalize_nvidia_cusparselt.py",
    )
    parser.add_argument(
        "--image-digest", default=os.environ.get("GLMRT_QUANT_IMAGE_DIGEST")
    )
    parser.add_argument(
        "--require-image-digest",
        action="store_true",
        default=os.environ.get("GLMRT_QUANT_REQUIRE_IMAGE_DIGEST") == "1",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.role not in ROLE_CONTRACTS:
        raise PreflightError(f"unsupported quantization role: {args.role!r}")
    expected_platform, expected_arch = ROLE_CONTRACTS[args.role]
    if args.target_platform != expected_platform or args.cuda_arch != expected_arch:
        raise PreflightError(
            f"role {args.role!r} requires target={expected_platform} cuda_arch={expected_arch}; "
            f"found target={args.target_platform!r} cuda_arch={args.cuda_arch!r}"
        )
    machine = platform.machine().lower()
    _validate_platform(args.target_platform, machine)
    cusparselt_normalization = _verify_cusparselt_normalization(
        args.cusparselt_normalizer.resolve()
    )

    if platform.python_version() != args.python_version:
        raise PreflightError(
            f"expected CPython {args.python_version}, found {platform.python_version()}"
        )
    if sys.implementation.name != "cpython":
        raise PreflightError(f"expected CPython, found {sys.implementation.name}")
    if sysconfig.get_config_var("Py_GIL_DISABLED") != 1:
        raise PreflightError("CPython was not built with free-threading enabled")
    if not hasattr(sys, "_is_gil_enabled") or sys._is_gil_enabled():
        raise PreflightError("the Python GIL is enabled")
    if os.environ.get("PYTHON_GIL") != "0":
        raise PreflightError("PYTHON_GIL must be explicitly set to 0")

    packages = _installed_packages()
    for name, expected in PACKAGE_VERSIONS.items():
        actual = packages.get(name)
        if actual != expected:
            raise PreflightError(f"expected {name}=={expected}, found {actual!r}")
    rustc_version = _tool_version(["rustc", "--version"], f"rustc {RUST_VERSION} ")
    cargo_version = _tool_version(["cargo", "--version"], "cargo ")
    uv_version = _tool_version(["uv", "--version"], f"uv {UV_VERSION} ")

    source_lock = _verify_source(
        args.source.resolve(), args.source_lock.resolve(), args.source_verifier.resolve()
    )
    requirements_digest = _sha256(args.requirements_lock.resolve())
    if not args.requirements_sha256:
        raise PreflightError("GLMRT_QUANT_REQUIREMENTS_SHA256 is not set")
    if requirements_digest != args.requirements_sha256:
        raise PreflightError(
            "quantization requirements lock digest mismatch: "
            f"expected {args.requirements_sha256}, found {requirements_digest}"
        )
    build_requirements_digest = _sha256(args.build_requirements_lock.resolve())
    if not args.build_requirements_sha256:
        raise PreflightError("GLMRT_QUANT_BUILD_REQUIREMENTS_SHA256 is not set")
    if build_requirements_digest != args.build_requirements_sha256:
        raise PreflightError(
            "quantization build requirements lock digest mismatch: "
            f"expected {args.build_requirements_sha256}, found {build_requirements_digest}"
        )

    image_digest = args.image_digest
    if image_digest and not IMAGE_DIGEST_RE.fullmatch(image_digest):
        raise PreflightError(f"invalid quantization image digest: {image_digest!r}")
    if args.require_image_digest and not image_digest:
        raise PreflightError("a content-addressed quantization image digest is required")

    import torch

    if torch.__version__ != PACKAGE_VERSIONS["torch"]:
        raise PreflightError(
            f"imported torch version differs from package metadata: {torch.__version__}"
        )
    if torch.version.cuda != "13.0":
        raise PreflightError(f"expected PyTorch CUDA 13.0, found {torch.version.cuda!r}")
    if not torch.cuda.is_available():
        raise PreflightError("CUDA is not available to PyTorch")
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < args.min_gpus:
        raise PreflightError(
            f"expected at least {args.min_gpus} visible GPUs, found {visible_gpu_count}"
        )

    expected_capability = _cuda_capability(args.cuda_arch)
    torch_gpus: list[dict[str, Any]] = []
    for index in range(visible_gpu_count):
        properties = torch.cuda.get_device_properties(index)
        capability = (properties.major, properties.minor)
        if capability != expected_capability:
            raise PreflightError(
                f"GPU {index} has compute capability {capability[0]}.{capability[1]}, "
                f"expected {expected_capability[0]}.{expected_capability[1]}"
            )
        torch_gpus.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": list(capability),
                "total_memory_bytes": properties.total_memory,
            }
        )

    smi_gpus = _nvidia_smi()
    if len(smi_gpus) != visible_gpu_count:
        raise PreflightError(
            f"nvidia-smi reports {len(smi_gpus)} GPUs but PyTorch sees {visible_gpu_count}"
        )
    for torch_gpu, smi_gpu in zip(torch_gpus, smi_gpus, strict=True):
        if smi_gpu["index"] != torch_gpu["index"]:
            raise PreflightError("nvidia-smi and PyTorch GPU indices do not agree")
        if tuple(smi_gpu["compute_capability"]) != expected_capability:
            raise PreflightError(
                f"nvidia-smi GPU {smi_gpu['index']} compute capability disagrees with the image"
            )

    return {
        "schema": 1,
        "status": "qualified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "target_platform": args.target_platform,
        "machine": machine,
        "base_image": os.environ.get("GLMRT_QUANT_BASE_IMAGE"),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "soabi": sysconfig.get_config_var("SOABI"),
            "gil_enabled": sys._is_gil_enabled(),
            "python_gil_environment": os.environ["PYTHON_GIL"],
        },
        "torch": {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "build_tools": {
            "rustc": rustc_version,
            "cargo": cargo_version,
            "uv": uv_version,
        },
        "cuda_arch": args.cuda_arch,
        "gpus": [
            {**torch_gpu, **smi_gpu}
            for torch_gpu, smi_gpu in zip(torch_gpus, smi_gpus, strict=True)
        ],
        "image_digest": image_digest,
        "requirements_lock": {
            "path": os.fspath(args.requirements_lock.resolve()),
            "sha256": requirements_digest,
        },
        "build_requirements_lock": {
            "path": os.fspath(args.build_requirements_lock.resolve()),
            "sha256": build_requirements_digest,
        },
        "cusparselt_normalization": cusparselt_normalization,
        "gptqmodel": {
            "source": os.fspath(args.source.resolve()),
            "revision": source_lock["revision"],
            "source_tree_sha256": source_lock["source_tree_sha256"],
        },
        "packages": packages,
    }


def main() -> int:
    args = parse_args()
    report = run(args)
    if args.output:
        _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as exc:
        print(f"quantization-preflight: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
