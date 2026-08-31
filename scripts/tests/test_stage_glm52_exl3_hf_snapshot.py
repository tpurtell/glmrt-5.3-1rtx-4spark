from __future__ import annotations

import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
TOOL_PATH = TOOLS / "stage_glm52_exl3_hf_snapshot.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_snapshot_stager", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
SYNC_PATH = TOOLS / "sync_glm52_exl3_hf_snapshot.py"
SYNC_SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_snapshot_sync", SYNC_PATH)
assert SYNC_SPEC is not None and SYNC_SPEC.loader is not None
SYNC = importlib.util.module_from_spec(SYNC_SPEC)
sys.modules[SYNC_SPEC.name] = SYNC
SYNC_SPEC.loader.exec_module(SYNC)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(TOOL._canonical_json(value)).hexdigest()}


def make_quant_evidence(
    tmp_path: Path, plan_sha256: str, *, schema: str = TOOL.QUANT_EVIDENCE_SCHEMA
) -> Path:
    path = tmp_path / "quant-evidence.json"
    report = bound(
        {
            "schema": schema,
            "status": "accepted",
            "quality_scope": (
                "projection-quantizer-evidence-not-end-to-end-model-quality"
            ),
            "plan": {"plan_sha256": plan_sha256},
            "coverage": {
                "expected_projection_count": TOOL.EXPECTED_PROJECTIONS,
                "projection_count": TOOL.EXPECTED_PROJECTIONS,
                "expected_expert_count": 75 * 256,
                "observed_expert_count": 75 * 256,
                "complete_expert_count": 75 * 256,
                "recovered_expert_count": 0,
                "layers": list(range(3, 78)),
            },
            "integrity": {
                "tensor_payload_hashes_verified": True,
                "journal_record_count": TOOL.EXPECTED_PROJECTIONS,
                "checkpoint_inventory_sha256": "e" * 64,
            },
            "metrics": {
                "global": {"aggregate_hessian_weighted_relative_error": 0.003}
            },
        },
        "report_sha256",
    )
    path.write_bytes(TOOL._canonical_json(report) + b"\n")
    return path


def make_candidate(tmp_path: Path, *, bits: int = 3) -> tuple[Path, Path, Path]:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    tensor = artifact / "model.safetensors"
    tensor.write_bytes(b"accepted-exl3-tensors")
    plan = artifact / "glmrt-gptqmodel-plan.json"
    plan_contract = (
        {
            "schema": "glmrt-glm5-gptqmodel-plan-v3",
            "recipe": "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1",
            "source": {
                "release": "glm-5.3",
                "format": "fp8-e4m3-block128x128-dynamic",
            },
        }
        if bits == 4
        else {
            "schema": "glmrt-glm52-gptqmodel-plan-v2",
            "recipe": "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1",
            "source": {"release": "glm-5.2", "format": "bf16"},
        }
    )
    plan.write_text(json.dumps(plan_contract) + "\n", encoding="utf-8")
    if bits == 4:
        (artifact / "quantize_config.json").write_text(
            json.dumps({"fixture": "complete-k4-quantization-config"}) + "\n",
            encoding="utf-8",
        )
    records = {
        plan.name: {"bytes": plan.stat().st_size, "sha256": sha256(plan)},
        tensor.name: {"bytes": tensor.stat().st_size, "sha256": sha256(tensor)},
    }
    if bits == 4:
        qcfg = artifact / "quantize_config.json"
        records[qcfg.name] = {
            "bytes": qcfg.stat().st_size,
            "sha256": sha256(qcfg),
        }
    manifest = {
        "schema": (
            "glmrt-glm5-gptqmodel-artifact-v2"
            if bits == 4
            else TOOL.ARTIFACT_SCHEMA
        ),
        "manifest_sha256": "a" * 64,
        "files": records,
    }
    (artifact / "glmrt-gptqmodel-artifact.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (artifact / "glmrt-gptqmodel-run.json").write_text("{}\n", encoding="utf-8")
    report = tmp_path / "validation.json"
    report_body = {
        "schema": (
            "glmrt-glm5-exl3-artifact-validation-v1"
            if bits == 4
            else TOOL.VALIDATION_SCHEMA
        ),
        "status": "accepted",
        "model_id": TOOL.GLM53_MODEL_ID if bits == 4 else TOOL.MODEL_ID,
        "artifact": str(artifact.resolve()),
        "artifact_manifest_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "retained_native_bytes_verified": True,
        "artifact_manifest_file_hashes_verified": True,
        "projection_checkpoint_bytes_verified": True,
        "projection_checkpoint": {
            "root": str(tmp_path / "projection-checkpoints"),
            "projection_count": TOOL.EXPECTED_PROJECTIONS,
            "tensor_count": TOOL.EXPECTED_PROJECTIONS * 4,
            "tensor_bytes": 272_734_848_000,
            "checkpoint_inventory_sha256": "e" * 64,
        },
        "tokenizer_evidence": {
            "mode": "plan-bound",
            "tokenizer_files": [
                {
                    "name": "tokenizer.json",
                    "bytes": 1,
                    "sha256": "c" * 64,
                    **({"hf_blob_id": "1" * 64} if bits == 4 else {}),
                },
                {
                    "name": "tokenizer_config.json",
                    "bytes": 1,
                    "sha256": "d" * 64,
                    **({"hf_blob_id": "2" * 64} if bits == 4 else {}),
                },
            ],
        },
    }
    if bits == 4:
        report_body["quantization_config"] = {
            "sha256": sha256(artifact / "quantize_config.json"),
            "tensor_storage_entries": TOOL.EXPECTED_PROJECTIONS,
            "stored_tensor_descriptions": TOOL.EXPECTED_PROJECTIONS * 4,
            "ledger_provenance_sha256": "9" * 64,
        }
        report_body["source_metadata"] = [
            {"name": "tokenizer.json", "bytes": 1, "sha256": "c" * 64},
            {
                "name": "tokenizer_config.json",
                "bytes": 1,
                "sha256": "d" * 64,
            },
            {
                "name": "generation_config.json",
                "bytes": 1,
                "sha256": "f" * 64,
            },
        ]
    report.write_bytes(
        TOOL._canonical_json(bound(report_body, "report_sha256")) + b"\n"
    )
    return artifact, report, make_quant_evidence(
        tmp_path,
        "b" * 64,
        schema=(
            TOOL.GLM53_QUANT_EVIDENCE_SCHEMA
            if bits == 4
            else TOOL.QUANT_EVIDENCE_SCHEMA
        ),
    )


def test_hardlink_stage_uses_standard_blob_snapshot_and_plain_ref(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    hf_home = tmp_path / "hf"

    staged = TOOL.stage(
        artifact,
        report,
        quant_evidence_report_path=quant_evidence,
        model_id=TOOL.MODEL_ID,
        hf_home=hf_home,
        link_mode="hardlink",
        update_ref=False,
    )

    snapshot = Path(staged["snapshot"])
    tensor_link = snapshot / "model.safetensors"
    tensor_blob = tensor_link.resolve(strict=True)
    assert tensor_link.is_symlink()
    assert tensor_blob.stat().st_ino == (artifact / "model.safetensors").stat().st_ino
    ref = Path(staged["cache_root"]) / "refs" / "main"
    assert ref.read_text(encoding="utf-8") == staged["revision"] + "\n"
    contract = SYNC._local_contract(hf_home, TOOL.MODEL_ID)
    assert contract.revision == staged["revision"]
    assert contract.files == staged["files"]
    assert contract.bytes == staged["bytes"]


def test_glm53_k4_stages_only_under_exact_publication_id(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path, bits=4)

    staged = TOOL.stage(
        artifact,
        report,
        quant_evidence_report_path=quant_evidence,
        model_id=TOOL.GLM53_MODEL_ID,
        hf_home=tmp_path / "hf",
        link_mode="hardlink",
        update_ref=False,
    )

    assert staged["model_id"] == "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
    assert "models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1" in staged["snapshot"]

    with pytest.raises(TOOL.StagingError, match="must be staged as"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "wrong-hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_glm53_stage_rejects_missing_exact_source_metadata(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path, bits=4)
    value = json.loads(report.read_text(encoding="utf-8"))
    value.pop("source_metadata")
    body = {key: item for key, item in value.items() if key != "report_sha256"}
    report.write_bytes(
        TOOL._canonical_json(bound(body, "report_sha256")) + b"\n"
    )

    with pytest.raises(TOOL.StagingError, match="does not accept"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.GLM53_MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_glm53_stage_rejects_source_metadata_that_disagrees_with_tokenizer_evidence(
    tmp_path: Path,
) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path, bits=4)
    value = json.loads(report.read_text(encoding="utf-8"))
    value["source_metadata"][0]["sha256"] = "0" * 64
    body = {key: item for key, item in value.items() if key != "report_sha256"}
    report.write_bytes(
        TOOL._canonical_json(bound(body, "report_sha256")) + b"\n"
    )

    with pytest.raises(TOOL.StagingError, match="does not accept"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.GLM53_MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_glm53_stage_rejects_missing_quantization_config_proof(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path, bits=4)
    value = json.loads(report.read_text(encoding="utf-8"))
    value.pop("quantization_config")
    body = {key: item for key, item in value.items() if key != "report_sha256"}
    report.write_bytes(
        TOOL._canonical_json(bound(body, "report_sha256")) + b"\n"
    )

    with pytest.raises(TOOL.StagingError, match="does not accept"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.GLM53_MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_stage_ref_move_requires_explicit_permission(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    hf_home = tmp_path / "hf"
    cache = TOOL._model_cache_root(hf_home, TOOL.MODEL_ID)
    (cache / "refs").mkdir(parents=True)
    (cache / "refs" / "main").write_text("old-revision\n", encoding="utf-8")

    with pytest.raises(TOOL.StagingError, match="--update-ref"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=hf_home,
            link_mode="hardlink",
            update_ref=False,
        )


def test_stage_rejects_validation_without_tokenizer_evidence(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    value = json.loads(report.read_text(encoding="utf-8"))
    value.pop("tokenizer_evidence")
    report.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(TOOL.StagingError, match="does not accept"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_stage_rejects_tampered_quant_evidence(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    value = json.loads(quant_evidence.read_text(encoding="utf-8"))
    value["coverage"]["projection_count"] -= 1
    quant_evidence.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(TOOL.StagingError, match="quant-evidence"):
        TOOL.stage(
            artifact,
            report,
            quant_evidence_report_path=quant_evidence,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "hf",
            link_mode="hardlink",
            update_ref=False,
        )


def test_sync_host_list_is_unique_and_safe() -> None:
    assert SYNC._hosts("ostrich,dodo,emu,kiwi") == (
        "ostrich",
        "dodo",
        "emu",
        "kiwi",
    )
    with pytest.raises(ValueError, match="unique"):
        SYNC._hosts("ostrich,ostrich")
    with pytest.raises(ValueError, match="unsafe"):
        SYNC._hosts("ostrich,bad/host")


def test_remote_hf_home_preserves_python_source_as_one_ssh_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_run(command, **_kwargs):
        observed.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="/home/tj/.cache/huggingface\n",
        )

    monkeypatch.setattr(SYNC.subprocess, "run", fake_run)

    assert SYNC._remote_hf_home("ostrich") == Path(
        "/home/tj/.cache/huggingface"
    )
    assert observed == [
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "ostrich",
            shlex.join(
                [
                    "python3",
                    "-c",
                    "import os,pathlib; print(pathlib.Path(os.environ.get('HF_HOME', pathlib.Path.home()/'.cache'/'huggingface')).expanduser().resolve())",
                ]
            ),
        ]
    ]


def test_remote_verifier_hashes_the_exact_staged_payload(tmp_path: Path) -> None:
    artifact, report, quant_evidence = make_candidate(tmp_path)
    hf_home = tmp_path / "hf"
    staged = TOOL.stage(
        artifact,
        report,
        quant_evidence_report_path=quant_evidence,
        model_id=TOOL.MODEL_ID,
        hf_home=hf_home,
        link_mode="hardlink",
        update_ref=False,
    )
    command = [
        sys.executable,
        "-c",
        SYNC.REMOTE_VERIFY,
        staged["cache_root"],
        staged["revision"],
        TOOL.MODEL_ID,
        "1",
    ]

    accepted = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert accepted.returncode == 0, accepted.stderr
    verified = json.loads(accepted.stdout)
    assert verified["revision"] == staged["revision"]
    assert verified["bytes"] == staged["bytes"]

    tensor_blob = (Path(staged["snapshot"]) / "model.safetensors").resolve()
    original = tensor_blob.read_bytes()
    corrupted = original.replace(b"accepted", b"rejected", 1)
    assert len(corrupted) == len(original) and corrupted != original
    tensor_blob.write_bytes(corrupted)
    rejected = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode != 0
    assert "remote blob hash mismatch" in rejected.stderr


def test_sync_fans_out_all_hosts_concurrently_and_sorts_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hosts = ("ostrich", "dodo", "emu", "kiwi")
    contract = SYNC.LocalContract(
        root=tmp_path,
        revision="a" * 64,
        files=17,
        bytes=123_456,
    )
    barrier = threading.Barrier(len(hosts))
    observed: list[str] = []
    observed_lock = threading.Lock()

    monkeypatch.setattr(SYNC, "_local_contract", lambda *_args: contract)

    def fake_sync_host(host, actual_contract, *, model_id, verify_hashes):
        assert actual_contract == contract
        assert model_id == TOOL.MODEL_ID
        assert verify_hashes is True
        barrier.wait(timeout=2.0)
        with observed_lock:
            observed.append(host)
        return {
            "host": host,
            "revision": contract.revision,
            "files": contract.files,
            "bytes": contract.bytes,
            "verified_blobs": contract.files,
        }

    monkeypatch.setattr(SYNC, "_sync_host", fake_sync_host)
    result = SYNC.sync(
        model_id=TOOL.MODEL_ID,
        hf_home=tmp_path / "hf",
        hosts=hosts,
        verify_hashes=True,
    )

    assert set(observed) == set(hosts)
    assert [entry["host"] for entry in result["hosts"]] == sorted(hosts)
    assert result["remote_payload_hashes_verified"] is True


def test_glm53_sync_uses_exact_k4_cache_root_and_full_hash_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = SYNC.LocalContract(
        root=tmp_path
        / "hf"
        / "hub"
        / "models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1",
        revision="a" * 64,
        files=42,
        bytes=987_654,
    )
    commands: list[list[str]] = []

    monkeypatch.setattr(
        SYNC,
        "_remote_hf_home",
        lambda host: Path("/home/tj/.cache/huggingface"),
    )

    def fake_run(command, **kwargs):
        commands.append(command)
        if kwargs.get("stdout") == subprocess.PIPE:
            payload = {
                "revision": contract.revision,
                "files": contract.files,
                "bytes": contract.bytes,
                "verified_blobs": contract.files,
            }
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(payload),
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(SYNC.subprocess, "run", fake_run)

    report = SYNC._sync_host(
        "ostrich",
        contract,
        model_id=TOOL.GLM53_MODEL_ID,
        verify_hashes=True,
    )

    exact_root = (
        "/home/tj/.cache/huggingface/hub/"
        "models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1"
    )
    assert report["host"] == "ostrich"
    assert commands[1] == [
        "rdmasync",
        "-aH",
        "--rdma=required",
        f"{contract.root}/",
        f"ostrich:{exact_root}/",
    ]
    remote_verify = shlex.split(commands[2][-1])
    assert remote_verify[-4:] == [
        exact_root,
        contract.revision,
        TOOL.GLM53_MODEL_ID,
        "1",
    ]
