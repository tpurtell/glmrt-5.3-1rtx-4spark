from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).parents[2]
PYTHON_ROOTS = (ROOT / "benchmarks", ROOT / "python", ROOT / "scripts")
IGNORED_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
B12X_REQUIREMENT = re.compile(
    r"(?i)(?<![\w.-])b12x(?:\[[^\]\r\n]*\])?\s*"
    r"(?:===|==|~=|!=|<=|>=|<|>|@)"
)
CUTLASS_PACKAGES = (
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-core",
    "nvidia-cutlass-dsl-libs-cu12",
    "nvidia-cutlass-dsl-libs-cu13",
)
QUALIFIED_CUTLASS_VERSION = "4.6.2"
SPARKINFER_IMAGE_DOCKERFILES = (
    ROOT / "docker" / "Dockerfile.dev",
    ROOT / "docker" / "Dockerfile.release",
)
METADATA_FREE_PYTHON_CACHE_MARKERS = (
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "*.pyc",
    "*.pyo",
)


def active_python_sources() -> list[Path]:
    return sorted(
        path
        for root in PYTHON_ROOTS
        for path in root.rglob("*.py")
        if not IGNORED_PARTS.intersection(path.relative_to(ROOT).parts)
    )


def is_retired_sparkinfer_module(module: str | None) -> bool:
    return module == "sparkinfer" or bool(
        module and module.startswith("sparkinfer.")
    )


def test_active_python_does_not_import_retired_sparkinfer_package() -> None:
    violations: list[str] = []
    for path in active_python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module]
            else:
                continue
            for module in modules:
                if is_retired_sparkinfer_module(module):
                    relative = path.relative_to(ROOT)
                    violations.append(f"{relative}:{node.lineno}: {module}")

    assert not violations, (
        "active Python sources must import b12x, not the retired sparkinfer "
        "package:\n" + "\n".join(violations)
    )


def test_standalone_tools_bootstrap_pinned_source_before_b12x_imports() -> None:
    violations: list[str] = []
    tools_root = ROOT / "python" / "tools"
    for path in sorted(tools_root.glob("*.py")):
        if path.name == "_pinned_sparkinfer.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        external_imports = [
            node
            for node in ast.walk(tree)
            if (
                isinstance(node, ast.Import)
                and any(
                    alias.name == "b12x"
                    or alias.name.startswith("b12x.")
                    for alias in node.names
                )
            )
            or (
                isinstance(node, ast.ImportFrom)
                and (
                    node.module == "b12x"
                    or bool(node.module and node.module.startswith("b12x."))
                )
            )
        ]
        if not external_imports:
            continue
        bootstrap_imports = [
            node
            for node in tree.body
            if isinstance(node, ast.Import)
            and any(alias.name == "_pinned_sparkinfer" for alias in node.names)
        ]
        relative = path.relative_to(ROOT)
        if len(bootstrap_imports) != 1:
            violations.append(
                f"{relative}: expected one top-level _pinned_sparkinfer import"
            )
            continue
        bootstrap_line = bootstrap_imports[0].lineno
        first_external_line = min(node.lineno for node in external_imports)
        if bootstrap_line >= first_external_line:
            violations.append(
                f"{relative}:{bootstrap_line}: bootstrap follows SparkInfer "
                f"import at line {first_external_line}"
            )

    assert not violations, (
        "standalone tools must verify and prepend GLMRT's pinned b12x/SparkInfer "
        "tree before importing it:\n" + "\n".join(violations)
    )


def test_build_metadata_does_not_pin_retired_b12x_package() -> None:
    package_files = [
        ROOT / "python" / "pyproject.toml",
        ROOT / "python" / "uv.lock",
        *sorted((ROOT / "docker").glob("Dockerfile*")),
        *sorted(ROOT.glob("requirements*.txt")),
    ]
    violations: list[str] = []
    for path in package_files:
        if not path.is_file():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if B12X_REQUIREMENT.search(line):
                violations.append(
                    f"{path.relative_to(ROOT)}:{line_number}: {line.strip()}"
                )

    assert not violations, (
        "build metadata must not install or pin an external b12x package:\n"
        + "\n".join(violations)
    )


def test_metadata_free_release_copies_filter_and_reject_python_caches() -> None:
    for relative in ("build.sh", "scripts/build-release-artifacts.sh"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        exclude_lines = [
            line for line in text.splitlines() if "--exclude" in line
        ]
        for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
            assert any(marker in line for line in exclude_lines), (
                f"{relative} must exclude {marker} from metadata-free "
                "SparkInfer source copies"
            )
        assert text.count("--require-no-python-cache") == 1, (
            f"{relative} must guard its copied SparkInfer source exactly once"
        )

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
        assert marker in dockerignore, (
            f".dockerignore must exclude SparkInfer cache marker {marker}"
        )
    for relative in ("docker/Dockerfile.dev", "docker/Dockerfile.release"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "ENV PYTHONDONTWRITEBYTECODE=1" in text
        assert "--require-no-python-cache" in text, (
            f"{relative} must reject cached Python artifacts after COPY"
        )


def test_remote_dev_staging_reconciles_the_pinned_fork() -> None:
    for relative in (
        "scripts/phase0-spark-tcp-bench.sh",
        "scripts/bench-verbs-app-coordinator-links.sh",
        "scripts/bench-verbs-app-pair.sh",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "--delete-excluded" in text
        assert "--require-no-python-cache" in text
        for marker in METADATA_FREE_PYTHON_CACHE_MARKERS:
            assert marker in text, (
                f"{relative} must exclude SparkInfer cache marker {marker}"
            )


def test_fork_and_images_share_qualified_cutlass_pin() -> None:
    fork_metadata = ROOT / "third_party" / "sparkinfer" / "pyproject.toml"
    assert fork_metadata.is_file(), (
        "initialize the pinned third_party/sparkinfer source before testing "
        "dependency agreement"
    )
    dependencies = tomllib.loads(fork_metadata.read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    versions: dict[str, str] = {}
    for requirement in dependencies:
        for package in CUTLASS_PACKAGES:
            prefix = f"{package}=="
            if requirement.startswith(prefix):
                versions[package] = requirement.removeprefix(prefix)

    assert versions == {
        package: QUALIFIED_CUTLASS_VERSION for package in CUTLASS_PACKAGES
    }, (
        "the SparkInfer fork must pin every CUTLASS DSL package to the "
        f"qualified {QUALIFIED_CUTLASS_VERSION} set, found {versions}"
    )

    image_pin = re.compile(r'nvidia-cutlass-dsl\[cu13\]==([^"]+)"')
    for dockerfile in SPARKINFER_IMAGE_DOCKERFILES:
        match = image_pin.search(dockerfile.read_text(encoding="utf-8"))
        assert match is not None, f"{dockerfile.relative_to(ROOT)} has no CUTLASS DSL pin"
        assert match.group(1) == QUALIFIED_CUTLASS_VERSION, (
            f"{dockerfile.relative_to(ROOT)} pins CUTLASS DSL {match.group(1)}, "
            f"expected {QUALIFIED_CUTLASS_VERSION} to match the fork"
        )


def test_quantization_image_does_not_install_sparkinfer_runtime_dependencies() -> None:
    text = (ROOT / "docker" / "Dockerfile.quantization").read_text(encoding="utf-8")
    assert "third_party/sparkinfer" not in text
    assert "nvidia-cutlass-dsl" not in text


def test_fork_accepts_the_ngc_base_torch_prerelease() -> None:
    fork_metadata = ROOT / "third_party" / "sparkinfer" / "pyproject.toml"
    assert fork_metadata.is_file(), (
        "initialize the pinned third_party/sparkinfer source before testing "
        "the image dependency contract"
    )
    dependencies = tomllib.loads(fork_metadata.read_text(encoding="utf-8"))[
        "project"
    ]["dependencies"]
    torch_requirements = [
        Requirement(requirement)
        for requirement in dependencies
        if Requirement(requirement).name == "torch"
    ]
    assert len(torch_requirements) == 1
    assert Version("2.12.0a0") in torch_requirements[0].specifier, (
        "nvcr.io/nvidia/pytorch:26.05-py3 contains torch 2.12.0a0; "
        f"the fork requirement {torch_requirements[0]} rejects that base and "
        "makes the image's `uv pip check` fail"
    )


def test_standalone_bootstrap_imports_verified_submodule() -> None:
    env = os.environ.copy()
    tools_path = os.fspath(ROOT / "python" / "tools")
    env["PYTHONPATH"] = tools_path + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import _pinned_sparkinfer as pinned; "
                "print(pinned.IMPORTED_MODULE); print(pinned.REVISION); "
                "print(pinned.VERSION)"
            ),
        ],
        check=False,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    imported_module, revision, version = result.stdout.strip().splitlines()
    assert Path(imported_module).resolve().is_relative_to(
        (ROOT / "third_party" / "sparkinfer").resolve()
    )
    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert Version(version) == Version("1.3.0")


def test_live_launchers_override_stale_cmake_sparkinfer_cache_entries() -> None:
    coordinator = (ROOT / "scripts" / "real-full-tcp-serve.sh").read_text(
        encoding="utf-8"
    )
    spark = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")

    assert (
        '-DGLMRT_SPARKINFER_SOURCE_DIR="$repo_root/third_party/sparkinfer"'
        in coordinator
    )
    assert (
        '-DGLMRT_SPARKINFER_LOCK_FILE="$repo_root/third_party/sparkinfer.lock.json"'
        in coordinator
    )
    assert '"-DGLMRT_NCCL_INCLUDE_DIR=$host_nccl_include_dir"' in coordinator
    assert '"-DGLMRT_NCCL_LIBRARY=$host_nccl_library"' in coordinator
    assert "-DGLMRT_SPARKINFER_SOURCE_DIR=" in spark
    assert "-DGLMRT_SPARKINFER_LOCK_FILE=" in spark
    coordinator_recipe = justfile.split(
        "build-native-coordinator-test:", maxsplit=1
    )[1].split("\n\n", maxsplit=1)[0]
    assert "-U GLMRT_ENABLE_B12X_AOT" in coordinator_recipe
    assert "-U GLMRT_ENABLE_B12X_COORDINATOR_AOT" in coordinator_recipe
    assert "-DGLMRT_SPARKINFER_SOURCE_DIR=" in coordinator_recipe
    assert "-DGLMRT_SPARKINFER_LOCK_FILE=" in coordinator_recipe


def test_launchers_use_only_the_packed_spark_moe_layout() -> None:
    phase0 = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    release = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert 'docker_args+=(-e GLMRT_SERVE_PROFILE="$serve_profile")' in phase0
    assert "serve_profile=${serve_profile:-unset}" in phase0
    assert 'serve_profile="${76:-}"' in phase0
    assert "GLMRT_SPARK_MOE_MODE" not in phase0
    assert "GLMRT_SPARKINFER_SOURCE_W4A16" not in phase0
    assert "GLMRT_SPARKINFER_HYBRID_W4A4_W4A16" not in phase0
    assert "SPARK_MOE_MODE" not in release
    assert "export GLMRT_SPARK_PREBUILT=1" in release
    assert "export GLMRT_SPARK_SKIP_STAGE=1" in release

    env = os.environ.copy()
    env["GLMRT_SPARK_PREBUILT"] = "1"
    env["GLMRT_SPARK_MOE_MODE"] = "hybrid-w4a4-w4a16"
    env["GLMRT_SERVE_PROFILE"] = "balanced"
    result = subprocess.run(
        [ROOT / "scripts" / "start-spark-experts-tcp.sh", "--dry-run"],
        check=False,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "GLMRT_SPARK_MOE_MODE" not in result.stdout
    assert "GLMRT_SERVE_PROFILE=balanced" in result.stdout


def test_phase0_image_only_staging_does_not_remove_live_experts() -> None:
    phase0 = (ROOT / "scripts" / "phase0-spark-tcp-bench.sh").read_text(
        encoding="utf-8"
    )
    cleanup = phase0.split("cleanup() {", maxsplit=1)[1].split(
        "\n}\ntrap cleanup EXIT", maxsplit=1
    )[0]

    assert '[ "$image_only" = "1" ]' in cleanup
    assert "docker rm -f" in cleanup


def test_release_preflight_requires_matching_engine_revisions() -> None:
    release = (ROOT / "run.sh").read_text(encoding="utf-8")

    assert (
        """coordinator_engine_commit="$(
  docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}'"""
        in release
    )
    assert '""|"<no value>"|unknown|unknown-*)' in release
    assert (
        """remote_engine_commit="$(
    ssh -o BatchMode=yes "$host" \\
      "docker image inspect -f '{{index .Config.Labels \\"org.opencontainers.image.revision\\"}}'"""
        in release
    )
    assert '[[ "$remote_engine_commit" == "$coordinator_engine_commit" ]]' in release
    fingerprint = release.split('deployment_fingerprint="$(', maxsplit=1)[1].split(
        "check_model_cache_local()", maxsplit=1
    )[0]
    assert '"$coordinator_engine_commit"' in fingerprint


def test_packed_expert_warmups_use_model_appropriate_wire() -> None:
    for launcher_name in (
        "real-full-tcp-serve.sh",
        "real-full-tcp-live-smoke.sh",
    ):
        launcher = (ROOT / "scripts" / launcher_name).read_text(encoding="utf-8")
        warmup = launcher.split("warmup_protocol_v2_experts() {", 1)[1].split(
            "\n}\n", 1
        )[0]

        assert "GLMRT_SPARK_MOE_MODE" not in warmup
        assert 'if [ "$warmup_layer_id" != "78" ]; then' not in warmup
        assert 'local wire_contract="bf16-in/bf16-out"' in warmup
        assert "local warmup_routes_per_row=1" in warmup
        assert 'case "$model_id" in' in warmup
        assert "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1" in warmup
        assert "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1" in warmup
        assert 'wire_contract="nvfp4-in/fp8-out"' in warmup
        assert "warmup_routes_per_row=8" in warmup
        assert "wire_args=(--nvfp4-fp8-roundtrip)" in warmup
        assert (
            'expert_ids="$(warmup_expert_ids_for_owner "$owner" '
            '"$warmup_routes_per_row")"'
            in warmup
        )
        assert '--expert-ids "$expert_ids"' in warmup
        assert '--routes-per-row "$warmup_routes_per_row"' in warmup
        assert '"${wire_args[@]}"' in warmup
        for argument in (
            '--warmup-rows "$warmup_rows"',
            '--roundtrip-rows "$warmup_roundtrip_rows"',
            '--mtp-chain-rows "$warmup_mtp_chain_rows"',
            '--prefill-roundtrip-rows "$warmup_prefill_roundtrip_rows"',
            '--prefill-chain-rows "$warmup_prefill_chain_rows"',
        ):
            assert argument in warmup
        for variable, default in (
            ("warmup_rows", "1"),
            ("warmup_roundtrip_rows", "1"),
            ("warmup_mtp_chain_rows", "2"),
            ("warmup_prefill_roundtrip_rows", "16,256,512"),
            ("warmup_prefill_chain_rows", "16,256,512"),
        ):
            assert re.search(
                rf'^{variable}="\$\{{[^}}]+:-{re.escape(default)}\}}"$',
                launcher,
                re.MULTILINE,
            )

        expert_ids = launcher.split("warmup_expert_ids_for_owner() {", 1)[1].split(
            "\n}\n", 1
        )[0]
        assert ".owner == $owner" in expert_ids
        assert ".layer_id == $layer" in expert_ids
        assert 'endswith(".gate_proj.weight")' in expert_ids
        assert 'endswith(".gate_proj.trellis")' in expert_ids
        assert "select(length >= $count)" in expert_ids
        assert ".[:$count]" in expert_ids
        assert 'join(",")' in expert_ids
