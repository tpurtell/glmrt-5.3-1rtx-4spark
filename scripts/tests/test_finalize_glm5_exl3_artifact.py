from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "_finalize_glm5_exl3_artifact",
    TOOLS / "finalize_glm5_exl3_artifact.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)


def bound(value: dict, field: str) -> dict:
    body = {key: item for key, item in value.items() if key != field}
    return {
        **body,
        field: hashlib.sha256(TOOL._canonical_json(body)).hexdigest(),
    }


def identity(path: Path) -> dict:
    payload = path.read_bytes()
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def raw_artifact(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    artifact = tmp_path / "raw"
    source.mkdir()
    artifact.mkdir()
    source_config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "head_dim": 192,
        "transformers_version": "5.15.0",
        "quantization_config": {"quant_method": "fp8"},
    }
    (source / "config.json").write_text(json.dumps(source_config))
    source_config_sha256 = hashlib.sha256(
        (source / "config.json").read_bytes()
    ).hexdigest()
    for name in TOOL.EXACT_SOURCE_METADATA_FILES:
        source_payload = json.dumps({"name": name, "source": True})
        (source / name).write_text(source_payload)
        (artifact / name).write_text(json.dumps({"name": name, "serialized": True}))
    ledger = {"family_join": {"sha256": "a" * 64}, "run": {"kind": "test"}}
    compact = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 4.0,
        "meta": {"ds4rt_error_ledger": ledger},
    }
    storage = {"module": {}}
    drifted = {
        **source_config,
        "head_dim": 64,
        "bos_token_id": 1,
        "transformers_version": "5.14.1",
        # This is the large legacy serialization shape emitted by the running
        # quantizer.  The finalizer must reduce it to the compact declaration.
        "quantization_config": {**compact, "tensor_storage": storage},
    }
    (artifact / "config.json").write_text(json.dumps(drifted))
    (artifact / "quantize_config.json").write_text(
        json.dumps({**compact, "tensor_storage": storage})
    )
    (artifact / "model.safetensors").write_bytes(b"fixture")
    plan = bound(
        {
            "schema": "glmrt-glm5-gptqmodel-plan-v3",
            "recipe": "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1",
            "source": {
                "release": "glm-5.3",
                "format": "fp8-e4m3-block128x128-dynamic",
                "path": str(source),
                "config_sha256": source_config_sha256,
            },
            "ledger_provenance": ledger,
        },
        "plan_sha256",
    )
    (artifact / "glmrt-gptqmodel-plan.json").write_text(json.dumps(plan))
    records = {
        path.name: identity(path) for path in artifact.iterdir() if path.is_file()
    }
    manifest = bound(
        {
            "schema": "glmrt-glm5-gptqmodel-artifact-v2",
            "plan_sha256": plan["plan_sha256"],
            "files": records,
            "file_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records.values()),
        },
        "manifest_sha256",
    )
    (artifact / TOOL.ARTIFACT_MANIFEST_FILENAME).write_text(json.dumps(manifest))
    run = bound(
        {
            "schema": "glmrt-glm5-gptqmodel-run-v2",
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "execution_upgrade_sha256": None,
            "quantized_base_layers": list(range(3, 78)),
            "preserved_mtp_layer": 78,
        },
        "run_sha256",
    )
    (artifact / TOOL.RUN_FILENAME).write_text(json.dumps(run))
    return artifact, source


def test_normalized_config_keeps_source_exact_and_exl3_separate(tmp_path: Path) -> None:
    artifact, source = raw_artifact(tmp_path)

    normalized = TOOL.normalized_model_config(artifact, source)

    source_config = json.loads((source / "config.json").read_text())
    assert normalized == source_config | {
        "quantization_config": {
            "quant_method": "exl3",
            "format": "exl3",
            "checkpoint_format": "exl3",
            "bits": 4.0,
        }
    }
    assert normalized["head_dim"] == 192
    assert "bos_token_id" not in normalized
    assert "meta" not in normalized["quantization_config"]
    assert json.loads((artifact / "quantize_config.json").read_text())["meta"][
        "ds4rt_error_ledger"
    ] == {
        "family_join": {"sha256": "a" * 64},
        "run": {"kind": "test"},
    }


def test_normalizer_accepts_active_images_metadata_rich_embedded_declaration(
    tmp_path: Path,
) -> None:
    artifact, source = raw_artifact(tmp_path)
    external = json.loads((artifact / "quantize_config.json").read_text())
    embedded = dict(external)
    embedded.pop("tensor_storage")
    exported = json.loads((artifact / "config.json").read_text())
    exported["quantization_config"] = embedded
    (artifact / "config.json").write_text(json.dumps(exported))

    normalized = TOOL.normalized_model_config(artifact, source)

    assert normalized["quantization_config"] == {
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 4.0,
    }


def test_finalizer_hardlinks_payload_and_rebinds_manifests(
    tmp_path: Path, monkeypatch
) -> None:
    artifact, source = raw_artifact(tmp_path)
    output = tmp_path / "finalized"
    monkeypatch.setattr(
        TOOL,
        "_validate_quantization_config",
        lambda root, *args, **kwargs: TOOL._json_object(root / "quantize_config.json"),
    )

    report = TOOL.finalize(artifact_path=artifact, output_path=output)

    normalized = json.loads((output / "config.json").read_text())
    source_config = json.loads((source / "config.json").read_text())
    assert normalized["head_dim"] == source_config["head_dim"] == 192
    assert normalized["transformers_version"] == "5.15.0"
    assert "bos_token_id" not in normalized
    for name in TOOL.EXACT_SOURCE_METADATA_FILES:
        assert (output / name).read_bytes() == (source / name).read_bytes()
    assert (artifact / "model.safetensors").stat().st_ino == (
        output / "model.safetensors"
    ).stat().st_ino
    assert (artifact / "quantize_config.json").stat().st_ino == (
        output / "quantize_config.json"
    ).stat().st_ino
    manifest = json.loads((output / TOOL.ARTIFACT_MANIFEST_FILENAME).read_text())
    run = json.loads((output / TOOL.RUN_FILENAME).read_text())
    assert run["artifact_manifest_sha256"] == manifest["manifest_sha256"]
    assert report["storage_mode"] == "hardlink"
    assert report["config_sha256"] != report["raw_config_sha256"]
    assert report["config_bytes"] == (output / "config.json").stat().st_size
    assert report["embedded_quantization_fields"] == [
        "bits",
        "checkpoint_format",
        "format",
        "quant_method",
    ]
    assert report["quantize_config_bytes"] == (
        output / "quantize_config.json"
    ).stat().st_size
    assert (
        report["quantize_config_sha256"]
        == identity(artifact / "quantize_config.json")["sha256"]
    )
    assert report["tensor_storage_entries"] == 1
    assert report["stored_tensor_descriptions"] == 0


def test_finalizer_rejects_source_config_changed_after_planning(
    tmp_path: Path,
) -> None:
    artifact, source = raw_artifact(tmp_path)
    source_config = json.loads((source / "config.json").read_text())
    source_config["head_dim"] = 64
    (source / "config.json").write_text(json.dumps(source_config))

    with pytest.raises(
        TOOL.FinalizationError,
        match="source config differs from the immutable quantization plan",
    ):
        TOOL.finalize(
            artifact_path=artifact,
            output_path=tmp_path / "must-not-exist",
        )


def test_finalizer_reports_root_owned_hardlink_failure(
    tmp_path: Path, monkeypatch
) -> None:
    artifact, _source = raw_artifact(tmp_path)

    def permission_denied(*_args, **_kwargs):
        raise OSError(TOOL.errno.EPERM, "operation not permitted")

    monkeypatch.setattr(TOOL.os, "link", permission_denied)
    output = tmp_path / "must-not-exist"
    with pytest.raises(
        TOOL.FinalizationError,
        match="normalize its ownership to the host user",
    ):
        TOOL.finalize(artifact_path=artifact, output_path=output)
    assert not output.exists()
