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
import stage_glm52_exl3_hf_snapshot as STAGE  # noqa: E402
from _b12x_exl3_k3_profile import (  # noqa: E402
    exl3_k3_capacity_rows,
    exl3_k3_route_block_rows,
)
from _b12x_exl3_k4_profile import (  # noqa: E402
    exl3_k4_capacity_rows,
    exl3_k4_route_block_rows,
)

TOOL_PATH = TOOLS / "prepare_glm52_exl3_hf_publication.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_exl3_publication", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
NATIVE_VALIDATOR_SHA256 = hashlib.sha256(
    (TOOLS / "validate_b12x_exl3_native.py").read_bytes()
).hexdigest()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound(value: dict, field: str) -> dict:
    return value | {field: hashlib.sha256(STAGE._canonical_json(value)).hexdigest()}


def test_public_configs_keep_complete_k4_tensor_storage_separate_and_compact(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    declaration = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 4.0,
        "codebook": "mcg",
        "calibration": {
            "schema": "glmrt-exl3-calibration-v1",
            "corpus_sha256": "9" * 64,
            "examples": 1441,
            "tokens": 1082141,
        },
        "meta": {
            "ds4rt_error_ledger": {
                "family_join": {
                    "corpus": {
                        "examples": 1441,
                        "tokens": 1082141,
                        "sha256": "8" * 64,
                    },
                    "quantizer_numerics": {
                        "hessian_capture": "raw-xtx-sum-fp32-v1",
                        "sigma_reg": 0.025,
                    },
                },
                "run": {
                    "execution_upgrade": {
                        "schema": "glmrt-glm5-execution-upgrade-v1",
                        "sha256": "7" * 64,
                    }
                },
            },
            "quantizer": "pinned-k4",
            "offload_to_disk": True,
            "offload_to_disk_path": "/machine-local/offload",
            "pack_impl": "threaded",
        },
    }
    tensor_storage = {
        "model.layers.3.mlp.experts.0.gate_proj": {
            "bits_per_weight": 4.0,
            "codebook": "mcg",
            "tensors": {
                "trellis": "model.layers.3.mlp.experts.0.gate_proj.trellis",
                "suh": "model.layers.3.mlp.experts.0.gate_proj.suh",
                "svh": "model.layers.3.mlp.experts.0.gate_proj.svh",
                "mcg": "model.layers.3.mlp.experts.0.gate_proj.mcg",
            },
        },
        "model.layers.77.mlp.experts.255.down_proj": {
            "bits_per_weight": 4.0,
            "codebook": "mcg",
            "tensors": {
                "trellis": "model.layers.77.mlp.experts.255.down_proj.trellis",
                "suh": "model.layers.77.mlp.experts.255.down_proj.suh",
                "svh": "model.layers.77.mlp.experts.255.down_proj.svh",
                "mcg": "model.layers.77.mlp.experts.255.down_proj.mcg",
            },
        },
    }
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    field: declaration[field]
                    for field in TOOL._compact_exl3_declaration(declaration)
                }
            }
        ),
        encoding="utf-8",
    )
    (artifact / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": tensor_storage}),
        encoding="utf-8",
    )

    compact_bytes, external_bytes = TOOL._public_configs(artifact)
    compact = json.loads(compact_bytes)
    external = json.loads(external_bytes)
    public_declaration = json.loads(json.dumps(declaration))
    for field in TOOL.PRIVATE_EXECUTION_META:
        public_declaration["meta"].pop(field, None)

    assert compact["quantization_config"] == TOOL._compact_exl3_declaration(
        public_declaration
    )
    assert "tensor_storage" not in compact["quantization_config"]
    assert "meta" not in compact["quantization_config"]
    assert "calibration" not in compact["quantization_config"]
    assert external == {**public_declaration, "tensor_storage": tensor_storage}
    assert (
        external["meta"]["ds4rt_error_ledger"]
        == declaration["meta"]["ds4rt_error_ledger"]
    )
    assert external["calibration"] == declaration["calibration"]
    assert not TOOL.PRIVATE_EXECUTION_META.intersection(external["meta"])

    oversized = dict(compact)
    oversized["unexpected_padding"] = "x" * TOOL.MAX_PUBLIC_CONFIG_BYTES
    (artifact / "config.json").write_text(json.dumps(oversized), encoding="utf-8")
    with pytest.raises(TOOL.PublicationError, match="exceeds 128 KiB"):
        TOOL._public_configs(artifact)

    compact["quantization_config"]["bits"] = 3.0
    (artifact / "config.json").write_text(json.dumps(compact), encoding="utf-8")
    with pytest.raises(TOOL.PublicationError, match="configurations conflict"):
        TOOL._public_configs(artifact)


def test_publication_metadata_must_still_match_accepted_manifest(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    records = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "quantize_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        path = artifact / name
        path.write_text(f"accepted-{name}\n", encoding="utf-8")
        records[name] = {"bytes": path.stat().st_size, "sha256": digest(path)}

    TOOL._verify_artifact_metadata_hashes(artifact, records)
    (artifact / "quantize_config.json").write_text(
        "tampered-quantization-config\n", encoding="utf-8"
    )
    with pytest.raises(TOOL.PublicationError, match="quantize_config.json"):
        TOOL._verify_artifact_metadata_hashes(artifact, records)


def quant_evidence(path: Path, plan_sha256: str) -> Path:
    report = bound(
        {
            "schema": STAGE.QUANT_EVIDENCE_SCHEMA,
            "status": "accepted",
            "quality_scope": (
                "projection-quantizer-evidence-not-end-to-end-model-quality"
            ),
            "plan": {"plan_sha256": plan_sha256},
            "coverage": {
                "expected_projection_count": STAGE.EXPECTED_PROJECTIONS,
                "projection_count": STAGE.EXPECTED_PROJECTIONS,
                "expected_expert_count": 75 * 256,
                "observed_expert_count": 75 * 256,
                "complete_expert_count": 75 * 256,
                "recovered_expert_count": 0,
                "layers": list(range(3, 78)),
            },
            "integrity": {
                "tensor_payload_hashes_verified": True,
                "journal_record_count": STAGE.EXPECTED_PROJECTIONS,
                "checkpoint_inventory_sha256": "e" * 64,
            },
            "metrics": {"global": {"aggregate_hessian_weighted_relative_error": 0.003}},
        },
        "report_sha256",
    )
    path.write_bytes(STAGE._canonical_json(report) + b"\n")
    return path


def serving_qualification(
    path: Path,
    *,
    artifact: Path,
    validation: Path,
    quant: Path,
    artifact_manifest_sha256: str,
    plan_sha256: str,
    model_id: str = TOOL.MODEL_ID,
    schema: str = TOOL.SERVING_QUALIFICATION_SCHEMA,
    speculation: str = "dspark",
    required_rows: list[int] | None = None,
) -> Path:
    library = path.parent / "libglmrt_native.so"
    library.write_bytes(b"test-native-library")
    library_identity = {
        "path": str(library.resolve()),
        "bytes": library.stat().st_size,
        "sha256": digest(library),
    }
    rows = required_rows or [1, 3, 9, 10, 129, 257, 513, 1025, 2049, 2064]
    capacity_for_rows = (
        exl3_k4_capacity_rows
        if schema == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA
        else exl3_k3_capacity_rows
    )
    trellis_bits = 4 if schema == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA else 3
    route_block_for_capacity = (
        exl3_k4_route_block_rows
        if schema == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA
        else exl3_k3_route_block_rows
    )

    def native_case(row: int) -> dict:
        capacity = capacity_for_rows(row)
        return {
            "rows": row,
            "capacity_rows": capacity,
            "route_block_rows": route_block_for_capacity(capacity),
            "packed_route_count": row * 8,
            "fc1_tile": [64, 256],
            "fc2_tile": [64, 256],
            "blocks_per_sm": 1,
            "registers_per_thread": 200,
            "local_memory_bytes": 0,
            "relative_l2": 0.0,
            "cosine": 1.0,
            "max_abs": 0.0,
        }

    native_paths: list[Path] = []
    glm53 = schema == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA
    native_inventory_sha256 = artifact_manifest_sha256 if glm53 else "f" * 64
    for tp_rank in range(4):
        weight_source = (
            {
                "kind": "finalized-exl3-artifact",
                "root": str(artifact.resolve()),
                "layer_id": 3,
                "tp_rank": tp_rank,
                "tp_world_size": 4,
                "projection_count": 768,
                "tensor_count": 3_072,
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "plan_sha256": plan_sha256,
                "authenticated_files": [
                    {"name": name, "bytes": 1, "sha256": "e" * 64}
                    for name in (
                        "model.safetensors.index.json",
                        "quantize_config.json",
                        "model-00001-of-00001.safetensors",
                    )
                ],
            }
            if glm53
            else {
                "kind": "calibrated-projection-checkpoints",
                "root": str((path.parent / "projection-checkpoints").resolve()),
                "layer_id": 3,
                "tp_rank": tp_rank,
                "tp_world_size": 4,
                "projection_count": 768,
                "tensor_bytes": 3_636_464_640,
                "inventory_sha256": "f" * 64,
            }
        )
        native = bound(
            {
                "schema": "glmrt-b12x-exl3-native-validation-v1",
                "status": "accepted",
                "script_sha256": NATIVE_VALIDATOR_SHA256,
                "expert_slot_fingerprint": "2" * 64,
                "trellis_bits": trellis_bits,
                "sparkinfer_revision": "3" * 40,
                "native_library": library_identity,
                "device": {
                    "name": "NVIDIA GB10",
                    "compute_capability": "12.1",
                },
                "weight_source": weight_source,
                "cases": [native_case(row) for row in rows],
            },
            "report_sha256",
        )
        native_path = path.parent / f"native-tp{tp_rank}.json"
        native_path.write_bytes(STAGE._canonical_json(native) + b"\n")
        native_paths.append(native_path)
    runtime = {
        "engine_identity": "wip-exl3-qualified-111111111111-222222222222",
        "coordinator_slot_fingerprint": "1" * 64,
        "expert_slot_fingerprint": "2" * 64,
        "sparkinfer_revision": "3" * 40,
        "profile": "balanced",
        "power_limit_w": 400,
        "speculation": speculation,
    }
    if schema == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA:
        runtime.update(
            {
                "default_speculation": speculation,
                "qualified_speculation": ["mtp", "dflash2"],
            }
        )
    report = bound(
        {
            "schema": schema,
            "status": "accepted",
            "model_id": model_id,
            "artifact": str(artifact.resolve()),
            "artifact_manifest_sha256": artifact_manifest_sha256,
            "plan_sha256": plan_sha256,
            "artifact_validation": {"sha256": digest(validation)},
            "quant_evidence": {"sha256": digest(quant)},
            "runtime": runtime,
            "gates": {
                name: True
                for name in sorted(
                    TOOL.GLM53_REQUIRED_SERVING_GATES
                    if schema == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA
                    else TOOL.REQUIRED_SERVING_GATES
                )
            },
            "failed_gates": [],
            "evidence": {
                "candidate_native_validations": [
                    {
                        "path": str(native_path.resolve()),
                        "bytes": native_path.stat().st_size,
                        "sha256": digest(native_path),
                        "schema": "glmrt-b12x-exl3-native-validation-v1",
                    }
                    for native_path in native_paths
                ]
            },
            "results": {
                "native_kernel": {
                    **(
                        {"weight_source_root": str(artifact.resolve())}
                        if glm53
                        else {}
                    ),
                    "expert_slot_fingerprint": "2" * 64,
                    "trellis_bits": trellis_bits,
                    "tp_ranks": [0, 1, 2, 3],
                    "layer_id": 3,
                    "checkpoint_inventory_sha256": native_inventory_sha256,
                    "native_library": library_identity,
                    "required_rows": rows,
                }
            },
        },
        "report_sha256",
    )
    path.write_bytes(STAGE._canonical_json(report) + b"\n")
    return path


def test_glm53_serving_gate_accepts_only_the_complete_k4_native_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact-k4"
    artifact.mkdir()
    validation = tmp_path / "validation-k4.json"
    quant = tmp_path / "quant-k4.json"
    validation.write_text("{}\n", encoding="utf-8")
    quant.write_text("{}\n", encoding="utf-8")
    contract = TOOL._artifact_contract(
        {
            "schema": "glmrt-glm5-gptqmodel-plan-v3",
            "recipe": "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1",
            "source": {
                "release": "glm-5.3",
                "format": "fp8-e4m3-block128x128-dynamic",
            },
        }
    )
    rows = sorted(
        TOOL.revalidate_glm53_native_evidence.__globals__["K4_REQUIRED_NATIVE_ROWS"]
    )
    serving = serving_qualification(
        tmp_path / "serving-k4.json",
        artifact=artifact,
        validation=validation,
        quant=quant,
        artifact_manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
        model_id=contract.model_id,
        schema=TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA,
        speculation="dflash2",
        required_rows=rows,
    )
    monkeypatch.setattr(TOOL, "revalidate_dflash2_fusion_evidence", lambda report: {})
    monkeypatch.setattr(TOOL, "revalidate_dflash2_topk_evidence", lambda report: {})
    monkeypatch.setattr(TOOL, "revalidate_dflash2_width_evidence", lambda report: {})

    accepted = TOOL._serving_qualification(
        serving,
        artifact=artifact.resolve(),
        artifact_manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
        validation_sha256=digest(validation),
        quant_evidence_sha256=digest(quant),
        projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        contract=contract,
    )

    assert accepted["schema"] == TOOL.GLM53_SERVING_QUALIFICATION_SCHEMA


def test_publication_is_standard_only_and_hardlinks_only_weight_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "artifact"
    source = tmp_path / "source"
    artifact.mkdir()
    source.mkdir()
    for name in (".gitattributes", "LICENSE", "chat_template.jinja"):
        (source / name).write_text(f"source {name}\n", encoding="utf-8")
    for name in ("generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        (artifact / name).write_text("{}\n", encoding="utf-8")
    shard = artifact / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"published-exl3-weights")
    (artifact / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard.stat().st_size},
                "weight_map": {"tensor": shard.name},
            }
        ),
        encoding="utf-8",
    )
    declaration = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 3.0,
        "meta": {
            "ds4rt_error_ledger": {"local": "/private/path"},
            "offload_to_disk": True,
            "offload_to_disk_path": "/private/offload",
            "moe_vram_strategy_devices": ["cuda:0", "cuda:1"],
            "quantizer": "pinned",
        },
    }
    (artifact / "config.json").write_text(
        json.dumps({"quantization_config": declaration}), encoding="utf-8"
    )
    (artifact / "quantize_config.json").write_text(
        json.dumps({**declaration, "tensor_storage": {"module": {}}}),
        encoding="utf-8",
    )
    plan = artifact / "glmrt-gptqmodel-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": "glmrt-glm52-gptqmodel-plan-v2",
                "recipe": "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1",
                "source": {"release": "glm-5.2", "format": "bf16"},
                "private": "local plan",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = {
        path.name: {"bytes": path.stat().st_size, "sha256": digest(path)}
        for path in artifact.iterdir()
        if path.name
        not in {
            "glmrt-gptqmodel-artifact.json",
            "glmrt-gptqmodel-run.json",
        }
    }
    manifest = {
        "schema": TOOL.ARTIFACT_SCHEMA,
        "manifest_sha256": "a" * 64,
        "files": records,
    }
    (artifact / "glmrt-gptqmodel-artifact.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (artifact / "glmrt-gptqmodel-run.json").write_text("{}\n", encoding="utf-8")
    validation = tmp_path / "validation.json"
    validation_body = {
        "schema": TOOL._validation_evidence.__globals__["VALIDATION_SCHEMA"],
        "status": "accepted",
        "model_id": TOOL.MODEL_ID,
        "artifact": str(artifact.resolve()),
        "source_snapshot": str(source.resolve()),
        "artifact_manifest_sha256": "a" * 64,
        "plan_sha256": "b" * 64,
        "retained_native_bytes_verified": True,
        "artifact_manifest_file_hashes_verified": True,
        "projection_checkpoint_bytes_verified": True,
        "projection_checkpoint": {
            "root": str(tmp_path / "projection-checkpoints"),
            "projection_count": STAGE.EXPECTED_PROJECTIONS,
            "tensor_count": STAGE.EXPECTED_PROJECTIONS * 4,
            "tensor_bytes": 272_734_848_000,
            "checkpoint_inventory_sha256": "e" * 64,
        },
        "tokenizer_evidence": {
            "mode": "plan-bound",
            "tokenizer_files": [
                {"name": "tokenizer.json", "bytes": 1, "sha256": "c" * 64},
                {
                    "name": "tokenizer_config.json",
                    "bytes": 1,
                    "sha256": "d" * 64,
                },
            ],
        },
    }
    validation.write_bytes(
        STAGE._canonical_json(bound(validation_body, "report_sha256")) + b"\n"
    )
    quant = quant_evidence(tmp_path / "quant-evidence.json", "b" * 64)
    serving = serving_qualification(
        tmp_path / "serving-qualification.json",
        artifact=artifact,
        validation=validation,
        quant=quant,
        artifact_manifest_sha256="a" * 64,
        plan_sha256="b" * 64,
    )
    readme = tmp_path / "README.md"
    serving_report_sha256 = json.loads(serving.read_text(encoding="utf-8"))[
        "report_sha256"
    ]
    readme.write_text(
        "---\nlicense: mit\n---\n\n# Calibrated K3\n\n"
        f"Qualification evidence SHA-256: `{serving_report_sha256}`\n",
        encoding="utf-8",
    )
    output = tmp_path / "public"
    monkeypatch.setattr(
        TOOL,
        "_validate_quantization_config",
        lambda _artifact, _modules, _contract: None,
    )

    report = TOOL.prepare(
        artifact,
        source,
        validation,
        quant,
        serving,
        readme,
        output,
        link_mode="hardlink",
    )

    assert report["status"] == "ready"
    assert {path.name for path in output.iterdir()} == set(TOOL.PUBLIC_METADATA) | {
        shard.name
    }
    attributes = (output / ".gitattributes").read_text(encoding="utf-8")
    for name in TOOL.HUB_LFS_ATTRIBUTE_PATHS:
        assert f"{name} filter=lfs diff=lfs merge=lfs -text" in attributes
    assert (output / shard.name).stat().st_ino == shard.stat().st_ino
    assert not (output / plan.name).exists()
    public_config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    public_external = json.loads(
        (output / "quantize_config.json").read_text(encoding="utf-8")
    )
    assert "tensor_storage" not in public_config["quantization_config"]
    assert public_external["tensor_storage"] == {"module": {}}
    assert public_external["meta"]["ds4rt_error_ledger"] == {"local": "/private/path"}
    assert not TOOL.PRIVATE_EXECUTION_META.intersection(public_external["meta"])
    assert public_external["meta"] == {
        "ds4rt_error_ledger": {"local": "/private/path"},
        "quantizer": "pinned",
    }
    assert report["plan_sha256"] == "b" * 64
    assert report["source_quant_evidence_sha256"] == digest(quant)
    assert report["source_serving_qualification_sha256"] == digest(serving)

    forged_native = json.loads(serving.read_text(encoding="utf-8"))
    forged_native.pop("report_sha256")
    forged_native["results"]["native_kernel"]["tp_ranks"] = [0, 1, 2, 2]
    forged_path = tmp_path / "serving-forged-native.json"
    forged_path.write_bytes(
        STAGE._canonical_json(bound(forged_native, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.PublicationError, match="unverifiable native EXL3"):
        TOOL._serving_qualification(
            forged_path,
            artifact=artifact.resolve(),
            artifact_manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            validation_sha256=digest(validation),
            quant_evidence_sha256=digest(quant),
            projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        )

    with pytest.raises(TOOL.PublicationError, match="source snapshot differs"):
        TOOL._validated_source_snapshot(
            json.loads(validation.read_text(encoding="utf-8")),
            tmp_path / "different-source",
        )

    publication_report = tmp_path / "publication.json"
    publication_report.write_text(json.dumps(report), encoding="utf-8")
    staged = STAGE.stage(
        output,
        None,
        publication_report_path=publication_report,
        model_id=TOOL.MODEL_ID,
        hf_home=tmp_path / "hf",
        link_mode="hardlink",
        update_ref=False,
    )
    assert (
        Path(staged["snapshot"]).joinpath(shard.name).resolve().stat().st_ino
        == shard.stat().st_ino
    )

    tampered = dict(report)
    tampered["plan_sha256"] = "c" * 64
    tampered_report = tmp_path / "publication-tampered.json"
    tampered_report.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(STAGE.StagingError, match="does not bind"):
        STAGE.stage(
            output,
            None,
            publication_report_path=tampered_report,
            model_id=TOOL.MODEL_ID,
            hf_home=tmp_path / "tampered-hf",
            link_mode="hardlink",
            update_ref=False,
        )

    rejected = json.loads(serving.read_text(encoding="utf-8"))
    rejected.pop("report_sha256")
    rejected["gates"]["tool_eval_points"] = False
    serving.write_bytes(STAGE._canonical_json(bound(rejected, "report_sha256")) + b"\n")
    with pytest.raises(TOOL.PublicationError, match="does not accept"):
        TOOL._serving_qualification(
            serving,
            artifact=artifact.resolve(),
            artifact_manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            validation_sha256=digest(validation),
            quant_evidence_sha256=digest(quant),
            projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        )

    incomplete = json.loads(serving.read_text(encoding="utf-8"))
    incomplete.pop("report_sha256")
    incomplete["gates"] = {"blended_decode": True}
    incomplete["failed_gates"] = []
    serving.write_bytes(
        STAGE._canonical_json(bound(incomplete, "report_sha256")) + b"\n"
    )
    with pytest.raises(TOOL.PublicationError, match="does not accept"):
        TOOL._serving_qualification(
            serving,
            artifact=artifact.resolve(),
            artifact_manifest_sha256="a" * 64,
            plan_sha256="b" * 64,
            validation_sha256=digest(validation),
            quant_evidence_sha256=digest(quant),
            projection_checkpoint_root=(tmp_path / "projection-checkpoints").resolve(),
        )
