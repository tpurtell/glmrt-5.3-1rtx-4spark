from __future__ import annotations

import hashlib
import os
import re
import signal
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WIP_PROCESS = ROOT / "scripts" / "wip-process.sh"
SOURCE_MANIFEST = ROOT / "scripts" / "verify-release-source-manifest.py"
FINGERPRINT = "a" * 64


def wait_for(path: Path, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {path}")


def invoke(runtime: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(WIP_PROCESS), *args],
        env={**os.environ, "GLMRT_WIP_RUNTIME_ROOT": str(runtime)},
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_identity_is_bound_to_live_pid_and_start_time(tmp_path: Path) -> None:
    runtime = tmp_path / "run"
    process = subprocess.Popen(
        [str(WIP_PROCESS), "run", "test-process", "sleep", "60"],
        env={**os.environ, "GLMRT_WIP_RUNTIME_ROOT": str(runtime)},
    )
    try:
        wait_for(runtime / "test-process.pid")
        bound = invoke(runtime, "bind-identity", "test-process", FINGERPRINT)
        assert bound.returncode == 0, bound.stderr

        identity = invoke(runtime, "identity", "test-process")
        assert identity.returncode == 0, identity.stderr
        assert identity.stdout.strip() == FINGERPRINT

        identity_file = runtime / "test-process.identity"
        contents = identity_file.read_text(encoding="utf-8")
        identity_file.write_text(
            contents.replace("start_ticks=", "start_ticks=0#"), encoding="utf-8"
        )
        stale = invoke(runtime, "identity", "test-process")
        assert stale.returncode != 0
        assert stale.stdout == ""
    finally:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=5)

    assert not (runtime / "test-process.pid").exists()
    assert not (runtime / "test-process.identity").exists()


def test_bind_identity_rejects_non_sha256_value(tmp_path: Path) -> None:
    result = invoke(tmp_path / "run", "bind-identity", "test-process", "not-a-hash")

    assert result.returncode == 2
    assert "invalid WIP process fingerprint" in result.stderr


def test_wip_launcher_has_separate_expert_and_deployment_identities() -> None:
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")

    assert "wip-expert-runtime-identity.py" in launcher
    assert "glmrt-wip-deployment-v2" in launcher
    assert 'GLMRT_RELEASE_CONFIG_SHA256="$expert_runtime_fingerprint"' in launcher
    assert "bind-identity" in launcher
    assert "'$expert_process' '$expert_runtime_fingerprint'" in launcher
    assert "reusing four fingerprint-matched resident WIP Spark experts" in launcher
    assert '["sparkinfer_revision"]' in launcher
    assert "GLMRT_SPARKINFER_COMMIT=$coordinator_sparkinfer_commit" in launcher
    assert (
        "GLMRT_COORDINATOR_POWER_LIMIT_WATTS=$coordinator_power_limit_watts"
        in launcher
    )
    assert 'coordinator_model_revision" == "$expert_model_revision' in launcher
    assert "write-wip-deployment-evidence.py" in launcher
    assert '--launch-started-ns "$launcher_started_ns"' in launcher
    assert 'deployment_evidence="$state_dir/deployment.json"' in launcher


def test_dflash_tuning_profile_invalidates_wip_source_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    profile = (
        source
        / "python"
        / "reference"
        / "glmrt_reference"
        / "dflash_tuning_profile.py"
    )
    profile.parent.mkdir(parents=True)
    profile.write_text("SELECTOR_WARPS = 8\n", encoding="utf-8")
    old_manifest = tmp_path / "old.SOURCE_SHA256SUMS"
    new_manifest = tmp_path / "new.SOURCE_SHA256SUMS"

    written = subprocess.run(
        [
            str(SOURCE_MANIFEST),
            "--source",
            str(source),
            "--write",
            str(old_manifest),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert written.returncode == 0, written.stderr
    relative_profile = (
        "./python/reference/glmrt_reference/dflash_tuning_profile.py"
    )
    assert relative_profile in old_manifest.read_text(encoding="utf-8")
    old_identity = hashlib.sha256(old_manifest.read_bytes()).hexdigest()

    profile.write_text("SELECTOR_WARPS = 4\n", encoding="utf-8")
    stale = subprocess.run(
        [
            str(SOURCE_MANIFEST),
            "--source",
            str(source),
            "--manifest",
            str(old_manifest),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert stale.returncode == 2
    assert "source content differs from the manifest" in stale.stderr

    rewritten = subprocess.run(
        [
            str(SOURCE_MANIFEST),
            "--source",
            str(source),
            "--write",
            str(new_manifest),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rewritten.returncode == 0, rewritten.stderr
    new_identity = hashlib.sha256(new_manifest.read_bytes()).hexdigest()
    assert new_identity != old_identity

    finalizer = (ROOT / "scripts" / "finalize-wip-slot.sh").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "run-wip.sh").read_text(encoding="utf-8")
    assert '"source_manifest_sha256": ${source_manifest_sha256@Q}' in finalizer
    assert 'sha256sum "$incoming/META.json"' in finalizer
    assert launcher.count("verify-release-source-manifest.py") == 2
    assert launcher.count(
        'test "$actual_fingerprint" = "$(<"$root/FINGERPRINT")"'
    ) == 2


def test_wip_builder_streams_every_local_heredoc_into_docker() -> None:
    builder = (ROOT / "wip.sh").read_text(encoding="utf-8")
    local_heredocs = re.findall(
        r'^\s*docker exec (?P<options>.*?)"\$coordinator_container" bash -s .*<<',
        builder,
        flags=re.MULTILINE,
    )

    assert len(local_heredocs) == 4
    assert all("-i" in options.split() for options in local_heredocs)
