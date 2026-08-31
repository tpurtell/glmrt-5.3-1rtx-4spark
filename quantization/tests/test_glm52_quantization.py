from __future__ import annotations

import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest
import torch

import quantize_glm52_gptqmodel as quant
from glm52_layer_boundary_store import (
    Glm52LayerBoundaryController,
    Glm52LayerBoundaryStore,
    LayerBoundaryError,
)


def _glm_config() -> dict:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "num_nextn_predict_layers": 1,
    }


def _glm_weight_map() -> dict[str, str]:
    names = {}
    projections = ("gate_proj", "up_proj", "down_proj")
    for layer in range(3, 79):
        for expert in range(256):
            for projection in projections:
                names[
                    f"model.layers.{layer}.mlp.experts.{expert}."
                    f"{projection}.weight"
                ] = "model.safetensors"
        for projection in projections:
            names[
                f"model.layers.{layer}.mlp.shared_experts.{projection}.weight"
            ] = "model.safetensors"
        names[f"model.layers.{layer}.mlp.gate.weight"] = "model.safetensors"
        names[
            f"model.layers.{layer}.mlp.gate.e_score_correction_bias"
        ] = "model.safetensors"
    return names


def _glm_fp8_weight_map() -> dict[str, str]:
    names = _glm_weight_map()
    for name, shard in tuple(names.items()):
        if ".mlp.experts." in name and name.endswith(".weight"):
            names[f"{name[:-len('.weight')]}.weight_scale_inv"] = shard
    return names


def test_glm_namespace_audit_separates_base_and_checkpoint_only_mtp():
    report = quant.glm52_namespace_audit(_glm_config(), _glm_weight_map())
    assert report["base_routed_layers"] == 75
    assert report["base_routed_projection_tensors"] == 57_600
    assert report["mtp_routed_projection_tensors"] == 768
    assert report["mtp_layer_index"] == 78
    assert report["quantized_scope"] == (
        "base-routed-expert-gate-up-down-only"
    )


def test_glm_namespace_audit_rejects_one_missing_projection():
    weights = _glm_weight_map()
    weights.pop("model.layers.78.mlp.experts.255.down_proj.weight")
    with pytest.raises(quant.LaunchError, match="inventory differs"):
        quant.glm52_namespace_audit(_glm_config(), weights)


def test_glm53_fp8_namespace_audit_binds_expert_scales():
    report = quant.glm52_namespace_audit(
        _glm_config(),
        _glm_fp8_weight_map(),
        source_format="fp8-e4m3-block128x128-dynamic",
    )

    assert report["base_routed_projection_scale_tensors"] == 57_600
    assert report["mtp_routed_projection_scale_tensors"] == 768


def test_source_variant_separates_full_glm52_and_glm53():
    assert quant.source_variant(_glm_config()) == {
        "release": "glm-5.2",
        "format": "bf16",
        "quantization_config_sha256": None,
    }
    config = _glm_config() | {
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        }
    }
    variant = quant.source_variant(config)
    assert variant["release"] == "glm-5.3"
    assert variant["format"] == "fp8-e4m3-block128x128-dynamic"
    assert len(variant["quantization_config_sha256"]) == 64


def _write_safetensors_fixture(
    root: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
) -> None:
    dtype_bytes = {"I16": 2, "F16": 2, "I32": 4, "BF16": 2}
    offset = 0
    header = {}
    for name, (dtype, shape) in tensors.items():
        length = 1
        for dimension in shape:
            length *= dimension
        length *= dtype_bytes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + length],
        }
        offset += length
    encoded = json.dumps(header, separators=(",", ":")).encode()
    shard = "model.safetensors"
    (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )


def _tiny_exl3_export_contract(tmp_path: Path):
    source = tmp_path / "source"
    export = tmp_path / "export"
    source.mkdir()
    export.mkdir()
    source_config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 4,
        "first_k_dense_replace": 3,
        "n_routed_experts": 1,
        "num_experts_per_tok": 1,
        "hidden_size": 16,
        "moe_intermediate_size": 16,
        "num_nextn_predict_layers": 1,
        "dtype": "bfloat16",
        "transformers_version": "source-version",
        "quantization_config": {"quant_method": "fp8"},
    }
    (source / "config.json").write_text(json.dumps(source_config))
    for name in quant.EXACT_SOURCE_METADATA_FILES:
        payload = json.dumps({"name": name, "source": True})
        (source / name).write_text(payload)
        (export / name).write_text(payload)
    modules = [
        "model.layers.3.mlp.experts.0.gate_proj",
        "model.layers.3.mlp.experts.0.up_proj",
        "model.layers.3.mlp.experts.0.down_proj",
    ]
    retained = "model.embed_tokens.weight"
    source_names = [retained]
    for module in modules:
        source_names.extend([f"{module}.weight", f"{module}.weight_scale_inv"])
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "source.safetensors" for name in source_names}})
    )

    storage = {}
    artifact_tensors = {retained: ("BF16", (16, 16))}
    for module in modules:
        stored = {
            f"{module}.trellis": {"shape": [1, 1, 64], "torch_dtype": "int16"},
            f"{module}.suh": {"shape": [16], "torch_dtype": "float16"},
            f"{module}.svh": {"shape": [16], "torch_dtype": "float16"},
            f"{module}.mcg": {"shape": [], "torch_dtype": "int32"},
        }
        storage[module] = {
            "stored_tensors": stored,
            "quant_format": "exl3",
            "bits_per_weight": 4,
            "mcg_multiplier": quant.EXL3_MCG_MULTIPLIER,
        }
        artifact_tensors.update(
            {
                f"{module}.trellis": ("I16", (1, 1, 64)),
                f"{module}.suh": ("F16", (16,)),
                f"{module}.svh": ("F16", (16,)),
                f"{module}.mcg": ("I32", ()),
            }
        )
    compact = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 4.0,
        "codebook": "mcg",
        "out_scales": "auto",
        "group_size": -1,
        "desc_act": False,
        "module_include": [quant.BASE_EXPERT_PATTERN],
        "lm_head": False,
        "pack_dtype": "int32",
        "meta": {"producer": "fixture"},
    }
    artifact_config = dict(source_config)
    artifact_config["quantization_config"] = quant.compact_exl3_declaration(compact)
    (export / "config.json").write_text(json.dumps(artifact_config))
    (export / "quantize_config.json").write_text(
        json.dumps(compact | {"tensor_storage": storage})
    )
    _write_safetensors_fixture(export, artifact_tensors)
    plan = {
        "source": {
            "path": os.fspath(source),
            "format": "fp8-e4m3-block128x128-dynamic",
            "geometry": {
                "first_target_layer": 3,
                "last_target_layer": 3,
                "hidden_size": 16,
                "moe_intermediate_size": 16,
                "n_routed_experts": 1,
            },
        },
        "exl3": {
            "bits": 4.0,
            "codebook": "mcg",
            "out_scales": "auto",
            "module_include": [quant.BASE_EXPERT_PATTERN],
        },
    }
    return export, plan


def test_export_quantization_contract_binds_full_storage_and_compact_config(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    quant.validate_export_quantization_contract(export, plan)

    external = json.loads((export / "quantize_config.json").read_text())
    external["tensor_storage"].pop(next(iter(external["tensor_storage"])))
    (export / "quantize_config.json").write_text(json.dumps(external))
    with pytest.raises(quant.LaunchError, match="tensor_storage module inventory"):
        quant.validate_export_quantization_contract(export, plan)


def test_export_config_normalization_restores_exact_source_fields(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    config = json.loads((export / "config.json").read_text())
    config["head_dim"] = 64
    config["transformers_version"] = "export-version"
    config["bos_token_id"] = 1
    (export / "config.json").write_text(json.dumps(config))

    normalized = quant.normalize_export_model_config(export, plan)
    source = json.loads(
        (Path(plan["source"]["path"]) / "config.json").read_text()
    )
    compact = quant.compact_exl3_declaration(
        json.loads((export / "quantize_config.json").read_text())
    )
    assert normalized == source | {"quantization_config": compact}
    assert json.loads((export / "config.json").read_text()) == normalized
    quant.validate_export_quantization_contract(export, plan)


def test_export_config_normalization_accepts_exact_full_embedded_storage(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    external = json.loads((export / "quantize_config.json").read_text())
    config = json.loads((export / "config.json").read_text())
    config["quantization_config"] = external
    (export / "config.json").write_text(json.dumps(config))

    normalized = quant.normalize_export_model_config(export, plan)

    compact = quant.compact_exl3_declaration(external)
    assert normalized["quantization_config"] == compact
    assert "meta" not in normalized["quantization_config"]
    assert json.loads((export / "quantize_config.json").read_text()) == external
    quant.validate_export_quantization_contract(export, plan)


def test_export_config_normalization_rejects_mismatched_full_embedded_storage(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    external = json.loads((export / "quantize_config.json").read_text())
    config = json.loads((export / "config.json").read_text())
    config["quantization_config"] = external
    module = next(iter(config["quantization_config"]["tensor_storage"]))
    config["quantization_config"]["tensor_storage"].pop(module)
    (export / "config.json").write_text(json.dumps(config))

    with pytest.raises(quant.LaunchError, match="embedded and standalone"):
        quant.normalize_export_model_config(export, plan)


def test_export_quantization_contract_rejects_corrupt_storage_metadata(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    external = json.loads((export / "quantize_config.json").read_text())
    module = next(iter(external["tensor_storage"]))
    tensor = next(iter(external["tensor_storage"][module]["stored_tensors"]))
    external["tensor_storage"][module]["stored_tensors"][tensor]["shape"] = [7]
    (export / "quantize_config.json").write_text(json.dumps(external))

    with pytest.raises(quant.LaunchError, match="tensor_storage metadata"):
        quant.validate_export_quantization_contract(export, plan)


def test_export_quantization_contract_rejects_noncompact_embedded_config(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    config = json.loads((export / "config.json").read_text())
    config["quantization_config"]["meta"] = {"producer": "different-export"}
    (export / "config.json").write_text(json.dumps(config))

    with pytest.raises(quant.LaunchError, match="exact minimal EXL3 declaration"):
        quant.validate_export_quantization_contract(export, plan)


def test_export_quantization_contract_rejects_source_config_drift(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    config = json.loads((export / "config.json").read_text())
    config["model_type"] = "wrong_architecture"
    (export / "config.json").write_text(json.dumps(config))

    with pytest.raises(quant.LaunchError, match="changed source field model_type"):
        quant.validate_export_quantization_contract(export, plan)


def test_export_quantization_contract_rejects_added_model_config_field(tmp_path):
    export, plan = _tiny_exl3_export_contract(tmp_path)
    config = json.loads((export / "config.json").read_text())
    config["unreviewed_runtime_override"] = True
    (export / "config.json").write_text(json.dumps(config))

    with pytest.raises(quant.LaunchError, match="added fields absent from the source"):
        quant.validate_export_quantization_contract(export, plan)


def test_source_metadata_identity_binds_hf_blob_and_content(tmp_path):
    model_root = tmp_path / "models--zai-org--GLM-5.2"
    snapshot = model_root / "snapshots" / ("a" * 40)
    blob_root = model_root / "blobs"
    snapshot.mkdir(parents=True)
    blob_root.mkdir()
    payload = b'{"tokenizer": "glm52"}\n'
    digest = quant.hashlib.sha256(payload).hexdigest()
    blob = blob_root / digest
    blob.write_bytes(payload)
    (snapshot / "tokenizer.json").symlink_to(Path("../../blobs") / digest)

    assert quant.source_metadata_identity(snapshot, "tokenizer.json") == {
        "name": "tokenizer.json",
        "bytes": len(payload),
        "sha256": digest,
        "hf_blob_id": digest,
    }

    blob.write_bytes(b"mutated\n")
    with pytest.raises(quant.LaunchError, match="SHA-256 blob has changed"):
        quant.source_metadata_identity(snapshot, "tokenizer.json")


def test_k3_storage_contract_accounts_for_compact_export_and_bounded_state():
    source = {
        "total_shard_bytes": 1_506_667_387_408,
        "geometry": {
            "first_target_layer": 3,
            "last_target_layer": 77,
            "n_routed_experts": 256,
            "hidden_size": 6144,
            "moe_intermediate_size": 2048,
        },
    }

    contract = quant.storage_contract(source)

    assert contract["native_replaced_payload_bytes"] == 1_449_551_462_400
    assert contract["exl3_projection_payload_bytes"] == 272_734_848_000
    assert contract["artifact_payload_estimate_bytes"] == 329_850_773_008
    assert contract["run_state_peak_bytes"] == 128 * 1024**3
    assert contract["retention"]["projection_checkpoints"] == (
        "all-completed-projections"
    )


def test_k4_storage_contract_replaces_fp8_weights_and_block_scales():
    source = {
        "total_shard_bytes": 755_632_050_320,
        "format": "fp8-e4m3-block128x128-dynamic",
        "geometry": {
            "first_target_layer": 3,
            "last_target_layer": 77,
            "n_routed_experts": 256,
            "hidden_size": 6144,
            "moe_intermediate_size": 2048,
        },
    }

    contract = quant.storage_contract(source, bits=4)

    assert contract["native_replaced_payload_bytes"] == 724_952_678_400
    assert contract["native_replaced_scale_payload_bytes"] == 176_947_200
    assert contract["exl3_projection_payload_bytes"] == 363_331_814_400
    assert contract["artifact_payload_estimate_bytes"] == 394_011_186_320
    assert contract["trellis_bits"] == 4


def test_storage_tree_excludes_externalized_projection_payload(tmp_path):
    run_state = tmp_path / "run-state"
    projection = run_state / quant.PROJECTION_CHECKPOINT_DIRNAME
    projection.mkdir(parents=True)
    (run_state / "boundary.bin").write_bytes(b"boundary")
    (projection / "projection.bin").write_bytes(b"projection")

    assert quant._tree_file_bytes(run_state) == len(b"boundaryprojection")
    assert quant._tree_file_bytes(run_state, exclude=(projection,)) == len(
        b"boundary"
    )


def test_projection_checkpoint_seed_is_content_bound_and_atomic(tmp_path):
    source = tmp_path / "source"
    family_join = {"recipe": "test"}
    request = {
        "schema": "test",
        "module": "model.layers.3.mlp.experts.0.gate_proj",
        "family_join": family_join,
    }
    request["request_sha256"] = quant.hashlib.sha256(
        quant.canonical_json(request)
    ).hexdigest()
    digest = request["request_sha256"]
    checkpoint = source / digest[:2] / digest[2:4]
    checkpoint.mkdir(parents=True)
    tensor_payload = b"tensor"
    ledger_record = {
        "record_kind": "projection",
        "logical_layer": 3,
        "expert": 0,
        "projection": "w1",
        "module": request["module"],
        "provenance": {"family_join": family_join},
    }
    manifest = {
        "schema": "test",
        "request": request,
        "request_sha256": digest,
        "tensor_file": f"{digest}.safetensors",
        "tensor_sha256": quant.hashlib.sha256(tensor_payload).hexdigest(),
        "result": {"ledger_record": ledger_record},
    }
    manifest["manifest_sha256"] = quant.hashlib.sha256(
        quant.canonical_json(manifest)
    ).hexdigest()
    (checkpoint / f"{digest}.json").write_bytes(
        quant.canonical_json(manifest) + b"\n"
    )
    (checkpoint / f"{digest}.safetensors").write_bytes(b"tensor")

    identity = quant.projection_checkpoint_seed_identity(source)
    destination = tmp_path / "destination"
    journal = tmp_path / "journal.jsonl"
    journal.write_bytes(b"")
    quant._seed_projection_checkpoints(identity, destination, journal)

    assert quant.projection_checkpoint_seed_identity(destination) | {
        "root": os.fspath(source.resolve())
    } == identity
    assert (destination / digest[:2] / digest[2:4] / f"{digest}.json").stat().st_ino == (
        checkpoint / f"{digest}.json"
    ).stat().st_ino
    bound_ledger = quant.read_json_object(journal)
    assert bound_ledger["module"] == request["module"]
    quant._validate_bound_record(
        bound_ledger,
        digest_field="record_sha256",
        label="seed ledger",
    )

    (checkpoint / f"{digest}.safetensors").write_bytes(b"changed\n")
    with pytest.raises(quant.LaunchError, match="content has changed"):
        quant._seed_projection_checkpoints(
            identity, tmp_path / "rejected", tmp_path / "rejected-journal"
        )


def test_projection_seed_family_allows_only_the_reviewed_resume_fix():
    seeded = {
        "recipe": "same",
        "gptqmodel": {
            "revision": "343290cddb72329a4bb3d1ee603ef579a3c488bf",
            "source_tree_sha256": "bedfe6dafbca9eb974736260c2cb1ee307f7ed84ff104a522f55cc662b7867b6",
        },
    }
    current = {
        "recipe": "same",
        "gptqmodel": {
            "revision": "fde23e4e3165843a9dfa74d1a8463e0375b43d42",
            "source_tree_sha256": "2fac9376f978a33e105c6f3382d0ad75535708a787e00ecf61029d47ba3bca46",
        },
    }

    current_numerics = quant.projection_seed_family_numerics(
        current, seeded_family_join=seeded
    )
    seeded_numerics = quant.projection_seed_family_numerics(seeded)
    seeded_numerics.pop("gptqmodel")
    assert current_numerics == seeded_numerics == {"recipe": "same"}

    current["gptqmodel"]["revision"] = "0" * 40
    assert "gptqmodel" in quant.projection_seed_family_numerics(
        current, seeded_family_join=seeded
    )


def test_execution_upgrade_legacy_source_allows_only_new_metadata_fields():
    legacy_source = {"revision": "a" * 40, "index_sha256": "b" * 64}
    legacy = {"schema": quant.LEGACY_PLAN_SCHEMA, "source": legacy_source}
    current = {
        **legacy_source,
        "tokenizer_files": [{"name": "tokenizer.json", "sha256": "c" * 64}],
    }
    assert quant.execution_upgrade_source_matches_parent(legacy, current)

    current["index_sha256"] = "d" * 64
    assert not quant.execution_upgrade_source_matches_parent(legacy, current)

    current["index_sha256"] = legacy_source["index_sha256"]
    current_plan = {"schema": quant.PLAN_SCHEMA, "source": legacy_source}
    assert not quant.execution_upgrade_source_matches_parent(current_plan, current)


def _store(tmp_path) -> Glm52LayerBoundaryStore:
    journal = tmp_path / "journal.jsonl"
    journal.write_bytes(b"")
    return Glm52LayerBoundaryStore(
        tmp_path / "boundaries",
        plan_sha256="a" * 64,
        family_join={"source": {"revision": "b" * 40}},
        projection_checkpoint_root=tmp_path / "projections",
        error_journal_path=journal,
        hidden_size=64,
        activation_rank=3,
        routed_experts=1,
        first_target_layer=3,
        last_target_layer=4,
    )


def _entries(layer: int) -> list[dict[str, str]]:
    return [
        {
            "module": f"model.layers.{layer}.mlp.experts.0.{projection}",
            "request_sha256": f"{layer + 1:x}" * 64,
            "record_sha256": f"{layer + 2:x}" * 64,
        }
        for projection in ("gate_proj", "up_proj", "down_proj")
    ]


def test_boundary_cumulative_index_starts_at_first_routed_layer(tmp_path):
    store = _store(tmp_path)
    first = store._validate_completed_projection_index(3, _entries(3))
    assert len(first) == 3
    second = store._validate_completed_projection_index(
        4,
        [*_entries(3), *_entries(4)],
    )
    assert len(second) == 6
    with pytest.raises(LayerBoundaryError, match="incomplete cumulative"):
        store._validate_completed_projection_index(4, _entries(4))


def test_boundary_controller_replays_dense_prefix_without_committing_it(tmp_path):
    controller = Glm52LayerBoundaryController(_store(tmp_path))
    processor = SimpleNamespace(
        completed_layer_checkpoint_entries=lambda _layer: []
    )
    assert controller.commit_layer(
        model=object(),
        processor=processor,
        layer_index=0,
        layer_name="model.layers.0",
    ) == {"status": "not-targeted", "layer_index": 0}


def test_boundary_stop_must_be_inside_routed_scope(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(ValueError, match="routed target layer"):
        Glm52LayerBoundaryController(store, stop_after_layer=2)


def test_boundary_round_trip_restores_evolving_prev_topk_state(
    tmp_path, monkeypatch
):
    store = _store(tmp_path)
    entries = tuple(_entries(3))
    monkeypatch.setattr(
        store, "_validate_projection_entries", lambda _layer, _entries: entries
    )
    monkeypatch.setattr(
        store,
        "_validate_completed_projection_index",
        lambda _layer, _entries: entries,
    )
    hidden = torch.arange(128, dtype=torch.bfloat16).reshape(1, 2, 64)
    prev_topk = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)
    committed_kwargs = [
        {
            "past_key_values": None,
            "position_embeddings": None,
            "prev_topk_indices": prev_topk,
            "use_cache": False,
        }
    ]
    store.commit(
        layer_index=3,
        layer_name="model.layers.3",
        layer_outputs=[[hidden]],
        layer_input_kwargs=committed_kwargs,
        position_ids=[None],
        attention_masks=[None],
        projection_entries=entries,
    )

    restored_kwargs = [
        {
            "past_key_values": None,
            "position_embeddings": None,
            "prev_topk_indices": None,
            "use_cache": False,
        }
    ]
    boundary = store.load_latest(
        layer_input_kwargs=restored_kwargs,
        position_ids=[None],
        attention_masks=[None],
    )

    assert boundary is not None
    assert torch.equal(restored_kwargs[0]["prev_topk_indices"], prev_topk)
    assert torch.equal(boundary.layer_inputs[0][0], hidden)


def test_forward_replay_scope_binds_safe_clone_policy(monkeypatch):
    monkeypatch.delenv(quant.FORWARD_REPLICA_ENV, raising=False)
    contract = {
        "policy": quant.FORWARD_REPLICA_POLICY,
        "torch_parallel_replicate": False,
    }
    with quant.forward_replica_scope(contract):
        assert os.environ[quant.FORWARD_REPLICA_ENV] == "0"
    assert quant.FORWARD_REPLICA_ENV not in os.environ


def test_forward_replay_scope_rejects_unbound_override(monkeypatch):
    monkeypatch.setenv(quant.FORWARD_REPLICA_ENV, "1")
    with pytest.raises(quant.LaunchError, match="immutable run state"):
        with quant.forward_replica_scope(
            {
                "policy": quant.FORWARD_REPLICA_POLICY,
                "torch_parallel_replicate": False,
            }
        ):
            pass


def test_exllamav3_jit_cache_scope_is_durable_and_restores_environment(
    tmp_path, monkeypatch
):
    variable = "GPTQMODEL_EXLLAMAV3_BUILD_ROOT"
    root = tmp_path / "run-state" / quant.EXLLAMAV3_JIT_DIRNAME
    monkeypatch.delenv(variable, raising=False)
    with quant.exllamav3_jit_cache_scope(root):
        assert not root.exists()
        assert os.environ[variable] == os.fspath(root)
    assert variable not in os.environ


def test_exllamav3_jit_cache_scope_rejects_an_unbound_root(tmp_path, monkeypatch):
    variable = "GPTQMODEL_EXLLAMAV3_BUILD_ROOT"
    monkeypatch.setenv(variable, os.fspath(tmp_path / "different-run"))
    with pytest.raises(quant.LaunchError, match="immutable run state"):
        with quant.exllamav3_jit_cache_scope(tmp_path / "current-run"):
            pass


def _execution_upgrade_fixture(tmp_path: Path, monkeypatch):
    run_state = tmp_path / "run-state"
    projection = run_state / quant.PROJECTION_CHECKPOINT_DIRNAME
    active_source = run_state / quant.ACTIVE_LAYER_SOURCE_DIRNAME
    offload = tmp_path / "offload"
    for path in (projection, active_source, offload):
        path.mkdir(parents=True)
    source = {"revision": "a" * 40, "path": os.fspath(tmp_path / "snapshot")}
    corpus = {"sha256": "b" * 64, "examples": 2}
    evidence = {"sha256": "c" * 64}
    parent_preflight = {
        "sha256": "d" * 64,
        "image_digest": "sha256:" + "1" * 64,
        "gptqmodel": {
            "revision": "2" * 40,
            "source_tree_sha256": "3" * 64,
        },
        "python": {"version": "3.14.6", "gil_enabled": False},
        "torch": {"version": "test"},
        "gpus": [
            {"index": 0, "uuid": "GPU-0"},
            {"index": 1, "uuid": "GPU-1"},
        ],
    }
    parent_toolchain = {"files": {"quantize": "4" * 64}}
    plan = {
        "plan_sha256": "5" * 64,
        "source": source,
        "corpus": corpus,
        "calibration_evidence": evidence,
        "quantization_toolchain": parent_toolchain,
        "preflight": parent_preflight,
        "output": os.fspath(tmp_path / "output"),
        "run_state_dir": os.fspath(run_state),
        "projection_checkpoint_dir": os.fspath(projection),
        "active_layer_source_dir": os.fspath(active_source),
        "offload_dir": os.fspath(offload),
        "projection_checkpoint_seed": None,
        "target_batch_size": 1,
        "remote_workers": None,
        "exl3": {"bits": 3},
        "ledger_provenance": {"family_join": {"recipe": "test"}, "run": {}},
    }
    quant.atomic_json(run_state / quant.PLAN_FILENAME, plan)
    journal_payload = b'{"record":"frontier"}\n'
    (run_state / quant.ERROR_JOURNAL_FILENAME).write_bytes(journal_payload)

    boundary_root = run_state / quant.LAYER_BOUNDARY_DIRNAME
    boundary_root.mkdir()
    boundary_body = {
        "schema": quant.BOUNDARY_SCHEMA,
        "schema_version": quant.BOUNDARY_SCHEMA_VERSION,
        "payload_hash_algorithm": quant.BOUNDARY_PAYLOAD_HASH_ALGORITHM,
        "plan_sha256": plan["plan_sha256"],
        "layer_index": 18,
        "layer_name": "model.layers.18",
        "activation_batches": 1426,
        "activation_bytes": 4096,
        "replay_state_bytes": 128,
        "completed_projection_entries": [{"module": "test"}],
    }
    boundary_digest = quant.hashlib.sha256(
        quant.canonical_json(boundary_body)
    ).hexdigest()
    boundary = boundary_root / f"layer-000018-{boundary_digest[:16]}"
    boundary.mkdir()
    quant.atomic_json(
        boundary / "manifest.json",
        {**boundary_body, "manifest_sha256": boundary_digest},
    )

    current_lock = {
        "schema": 1,
        "repository": "https://github.com/tpurtell/GPTQModel.git",
        "revision": "6" * 40,
        "source_tree_sha256": "7" * 64,
    }
    lock_path = tmp_path / "gptqmodel.lock.json"
    lock_path.write_text(json.dumps(current_lock), encoding="utf-8")
    current_preflight = json.loads(json.dumps(parent_preflight))
    current_preflight.update(
        {
            "sha256": "8" * 64,
            "image_digest": "sha256:" + "9" * 64,
            "gptqmodel": {
                "revision": current_lock["revision"],
                "source_tree_sha256": current_lock["source_tree_sha256"],
            },
        }
    )
    args = SimpleNamespace(
        output=Path(plan["output"]),
        run_state_dir=run_state,
        projection_checkpoint_dir=projection,
        projection_checkpoint_seed_dir=None,
        active_layer_source_dir=active_source,
        offload_dir=offload,
        snapshot=tmp_path / "snapshot",
        calibration_jsonl=tmp_path / "calibration.jsonl",
        calibration_manifest=tmp_path / "calibration-manifest.json",
        route_screen_report=None,
        gptqmodel_lock=lock_path,
        preflight_report=tmp_path / "preflight.json",
        bits=3,
        batch_size=1,
        coordinator_gpu_count=2,
        remote_worker=None,
    )
    monkeypatch.setattr(quant, "_validate_plan", lambda _plan: None)
    monkeypatch.setattr(quant, "snapshot_identity", lambda _path: source)
    monkeypatch.setattr(
        quant,
        "calibration_stream",
        lambda _path: (["one", "two"], corpus),
    )
    monkeypatch.setattr(
        quant,
        "calibration_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    monkeypatch.setattr(
        quant,
        "preflight_identity",
        lambda *_args, **_kwargs: current_preflight,
    )
    current_toolchain = {"files": {"quantize": "a" * 64}}
    monkeypatch.setattr(
        quant,
        "quantization_toolchain_identity",
        lambda: current_toolchain,
    )
    return (
        args,
        plan,
        current_preflight,
        current_toolchain,
        journal_payload,
        boundary_digest,
    )


def test_execution_upgrade_preserves_layer18_boundary_and_is_stable(
    tmp_path, monkeypatch
):
    args, parent, preflight, toolchain, journal_payload, boundary_digest = (
        _execution_upgrade_fixture(tmp_path, monkeypatch)
    )

    resumed, texts, first = quant.build_execution_upgrade(args)
    repeated, repeated_texts, second = quant.build_execution_upgrade(args)

    assert resumed == repeated == parent
    assert texts == repeated_texts == ["one", "two"]
    assert first == second
    assert first["parent_plan_sha256"] == parent["plan_sha256"]
    assert first["upgraded_execution"]["gptqmodel"] == preflight["gptqmodel"]
    assert first["upgraded_execution"]["quantization_toolchain"] == toolchain
    assert first["change_contract"]["capture_batch_payload"] == (
        quant.ROUTER_CANDIDATE_CAPTURE_PAYLOAD_CONTRACT
    )
    assert first["resume_state"] == {
        "contract": "latest-boundary-plus-journal-v1",
        "layer_boundary": {
            "directory": next(
                (Path(parent["run_state_dir"]) / quant.LAYER_BOUNDARY_DIRNAME).iterdir()
            ).name,
            "layer_index": 18,
            "layer_name": "model.layers.18",
            "manifest_sha256": boundary_digest,
            "activation_batches": 1426,
            "activation_bytes": 4096,
            "replay_state_bytes": 128,
            "completed_projection_entries": 1,
        },
        "error_journal": {
            "bytes": len(journal_payload),
            "records": 1,
            "sha256": quant.hashlib.sha256(journal_payload).hexdigest(),
        },
    }

    runtime = quant.runtime_ledger_provenance(parent)
    assert runtime["family_join"] == parent["ledger_provenance"]["family_join"]
    assert runtime["run"]["execution_upgrade"]["upgrade_sha256"] == first[
        "upgrade_sha256"
    ]


def test_execution_upgrade_repairs_interrupted_duplicate_archive(
    tmp_path, monkeypatch
):
    args, parent, _preflight, _toolchain, _journal, _boundary = (
        _execution_upgrade_fixture(tmp_path, monkeypatch)
    )
    _plan, _texts, first = quant.build_execution_upgrade(args)
    history = (
        Path(parent["run_state_dir"])
        / quant.EXECUTION_UPGRADE_HISTORY_DIRNAME
    )
    history.mkdir()
    duplicate = history / f"{first['upgrade_sha256']}.json"
    quant.atomic_json(duplicate, first)

    resumed, texts, repeated = quant.build_execution_upgrade(args)

    assert resumed == parent
    assert texts == ["one", "two"]
    assert repeated == first
    assert not duplicate.exists()


def test_execution_upgrade_rejects_gpu_identity_change(tmp_path, monkeypatch):
    args, _parent, current, _toolchain, _journal, _boundary = (
        _execution_upgrade_fixture(tmp_path, monkeypatch)
    )
    current["gpus"][1]["uuid"] = "GPU-DIFFERENT"

    with pytest.raises(quant.LaunchError, match="runtime or GPU identities"):
        quant.build_execution_upgrade(args)


def _staged_artifact(tmp_path: Path, monkeypatch):
    plan = {
        "schema": quant.PLAN_SCHEMA,
        "plan_sha256": "a" * 64,
        "source": {
            "geometry": {
                "first_target_layer": 3,
                "last_target_layer": 77,
                "mtp_layer_index": 78,
            }
        },
    }
    run_state = tmp_path / "run-state"
    stage = run_state / quant.EXPORT_STAGE_DIRNAME
    output = tmp_path / "model"
    stage.mkdir(parents=True)
    (stage / "model.safetensors").write_bytes(b"quantized")
    quant.atomic_json(stage / quant.PLAN_FILENAME, plan)
    monkeypatch.setattr(quant, "_validate_plan", lambda _plan: None)
    manifest = quant.write_artifact_manifest(stage, plan)
    run = quant._bound_record(
        {
            "schema": quant.RUN_SCHEMA,
            "status": "complete",
            "plan_sha256": plan["plan_sha256"],
            "artifact_manifest_sha256": manifest["manifest_sha256"],
            "quantized_base_layers": list(range(3, 78)),
            "preserved_mtp_layer": 78,
        },
        "run_sha256",
    )
    quant.atomic_json(stage / quant.RUN_FILENAME, run)
    return plan, stage, output


def test_resume_discards_only_uncommitted_export_stage(tmp_path):
    stage = tmp_path / "run-state" / quant.EXPORT_STAGE_DIRNAME
    stage.mkdir(parents=True)
    (stage / "partial.safetensors").write_bytes(b"partial")

    assert quant._recover_export_stage(stage, tmp_path / "model", {}) is False
    assert not stage.exists()


def test_resume_hashes_and_publishes_committed_export_stage(
    tmp_path, monkeypatch
):
    plan, stage, output = _staged_artifact(tmp_path, monkeypatch)

    assert quant._recover_export_stage(stage, output, plan) is True
    assert output.is_dir()
    assert not stage.exists()
    assert (output / "model.safetensors").read_bytes() == b"quantized"


def test_resume_preserves_and_rejects_corrupt_committed_export_stage(
    tmp_path, monkeypatch
):
    plan, stage, output = _staged_artifact(tmp_path, monkeypatch)
    (stage / "model.safetensors").write_bytes(b"corrupted")

    with pytest.raises(quant.LaunchError, match="failed hashing"):
        quant._recover_export_stage(stage, output, plan)
    assert stage.is_dir()
    assert not output.exists()


def test_resume_refuses_symlinked_export_stage(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_bytes(b"user")
    stage = tmp_path / "run-state" / quant.EXPORT_STAGE_DIRNAME
    stage.parent.mkdir()
    stage.symlink_to(outside, target_is_directory=True)

    with pytest.raises(quant.LaunchError, match="not a regular directory"):
        quant._recover_export_stage(stage, tmp_path / "model", {})
    assert (outside / "keep").read_bytes() == b"user"
