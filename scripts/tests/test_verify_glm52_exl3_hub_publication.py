from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "_verify_glm52_exl3_hub",
    TOOLS / "verify_glm52_exl3_hub_publication.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
TEST_STORAGE_COUNTS = {
    "expected_tensor_storage_entries": 1,
    "expected_stored_tensor_descriptions": 4,
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(data)}\0".encode() + data,
        usedforsecurity=False,
    ).hexdigest()


def fixture(tmp_path: Path, *, model_id: str = TOOL.MODEL_ID):
    publication = tmp_path / "publication"
    publication.mkdir()
    declaration = {
        "method": "exl3",
        "quant_method": "exl3",
        "format": "exl3",
        "checkpoint_format": "exl3",
        "bits": 3.0 if model_id == TOOL.MODEL_ID else 4.0,
    }
    if model_id == TOOL.GLM53_MODEL_ID:
        declaration["meta"] = {
            "ds4rt_error_ledger": {
                "family_join": {"sha256": "6" * 64},
                "run": {"kind": "production"},
            }
        }
    embedded = (
        TOOL._compact_exl3_declaration(declaration)
        if model_id == TOOL.GLM53_MODEL_ID
        else declaration
    )
    module = "model.layers.3.mlp.experts.0.up_proj"
    storage = {
        module: {
            "stored_tensors": {
                f"{module}.trellis": {
                    "shape": [1, 1, 64],
                    "torch_dtype": "int16",
                },
                f"{module}.suh": {"shape": [16], "torch_dtype": "float16"},
                f"{module}.svh": {"shape": [16], "torch_dtype": "float16"},
                f"{module}.mcg": {"shape": [], "torch_dtype": "int32"},
            },
            "quant_format": "exl3",
            "bits_per_weight": int(declaration["bits"]),
            "mcg_multiplier": TOOL.EXL3_MCG_MULTIPLIER,
        }
    }
    payloads = {
        "README.md": b"abc",
        "config.json": json.dumps({"quantization_config": embedded}).encode(),
        "quantize_config.json": json.dumps(
            {**declaration, "tensor_storage": storage}
        ).encode(),
        "model-00001-of-00001.safetensors": b"12345",
    }
    for name, data in payloads.items():
        (publication / name).write_bytes(data)
    body = {
        "schema": "glmrt-hf-standard-publication-v3",
        "model_id": model_id,
        "source_artifact_manifest_sha256": "1" * 64,
        "source_validation_sha256": "2" * 64,
        "source_quant_evidence_sha256": "3" * 64,
        "source_serving_qualification_sha256": "4" * 64,
        "plan_sha256": "5" * 64,
        "files": [
            {"path": name, "bytes": len(data), "sha256": digest(data)}
            for name, data in sorted(payloads.items())
        ],
    }
    report = {
        **body,
        "publication_sha256": hashlib.sha256(TOOL._canonical_json(body)).hexdigest(),
        "status": "ready",
        "output": str(publication),
    }
    report_path = tmp_path / "publication.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    revision = "a" * 40
    siblings = []
    for name, data in sorted(payloads.items()):
        if name.endswith(".safetensors"):
            siblings.append(
                SimpleNamespace(
                    rfilename=name,
                    size=len(data),
                    blob_id="f" * 40,
                    lfs=SimpleNamespace(size=len(data), sha256=digest(data)),
                )
            )
        else:
            siblings.append(
                SimpleNamespace(
                    path=name,
                    size=len(data),
                    blob_id=git_blob_oid(data),
                    lfs=None,
                )
            )
    info = SimpleNamespace(
        id=model_id,
        sha=revision,
        private=False,
        gated=False,
        siblings=siblings,
        config={"quantization_config": embedded},
    )
    return report_path, payloads, info


class FakeApi:
    def __init__(self, info):
        self.info = info

    def model_info(self, repo_id, **kwargs):
        assert repo_id == self.info.id
        assert kwargs["files_metadata"] is True
        return self.info


def page_fetcher(
    url: str, token: bool | str | None, *, body: bytes = b"<html><body>OK</body></html>"
) -> dict:
    assert token is None
    return {
        "status": 200,
        "url": url,
        "content_type": "text/html",
        "body": body,
    }


def test_accepts_exact_public_revision_and_materializes_download_equivalent_cache(
    tmp_path: Path,
) -> None:
    report_path, payloads, info = fixture(tmp_path)
    hf_home = tmp_path / "hf"
    report = TOOL.verify(
        publication_report_path=report_path,
        revision="main",
        api=FakeApi(info),
        page_fetcher=page_fetcher,
        hf_home=hf_home,
        **TEST_STORAGE_COUNTS,
    )

    assert report["status"] == "accepted"
    assert report["resolved_revision"] == "a" * 40
    assert report["full_model_redownloaded"] is False
    assert report["materialized_cache"]["download_equivalent_layout"] is True
    snapshot = Path(report["materialized_cache"]["snapshot"])
    assert Path(
        report["materialized_cache"]["local_only_snapshot_resolution"]
    ).resolve() == snapshot.resolve()
    assert snapshot.name == "a" * 40
    assert (snapshot / "config.json").read_bytes() == payloads["config.json"]
    assert (Path(report["materialized_cache"]["cache_root"]) / "refs" / "main").read_text().strip() == "a" * 40
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(
        repo_id=info.id,
        revision="a" * 40,
        cache_dir=hf_home / "hub",
        local_files_only=True,
    )
    assert Path(resolved).resolve() == snapshot.resolve()
    assert report["quantization_config"]["tensor_storage_entries"] == 1
    assert {entry["method"] for entry in report["files"]} == {
        "git-blob-sha1",
        "lfs-sha256",
    }
    assert [page["kind"] for page in report["hub_pages"]] == [
        "model",
        "revision-tree",
        "compact-config",
    ]


def test_glm53_k4_verifies_the_exact_destination_repository(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(
        tmp_path, model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
    )

    report = TOOL.verify(
        publication_report_path=report_path,
        revision="main",
        api=FakeApi(info),
        page_fetcher=page_fetcher,
        **TEST_STORAGE_COUNTS,
    )

    assert report["schema"] == TOOL.GLM53_SCHEMA
    assert report["model_id"] == "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"


def test_rejects_unexpected_remote_file(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)
    info.siblings.append(SimpleNamespace(path="junk.txt", size=1, lfs=None))

    with pytest.raises(TOOL.HubVerificationError, match="inventory differs"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            page_fetcher=page_fetcher,
            **TEST_STORAGE_COUNTS,
        )


def test_rejects_remote_git_blob_that_differs_from_publication(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)
    next(sibling for sibling in info.siblings if sibling.path == "config.json").blob_id = "0" * 40

    with pytest.raises(TOOL.HubVerificationError, match="Git blob differs"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            page_fetcher=page_fetcher,
            **TEST_STORAGE_COUNTS,
        )


def test_rejects_config_error_rendered_on_actual_hub_page(tmp_path: Path) -> None:
    report_path, _payloads, info = fixture(tmp_path)

    def error_page(url: str, token: bool | str | None) -> dict:
        return page_fetcher(
            url,
            token,
            body=b"<html><body>config.json is too large</body></html>",
        )

    with pytest.raises(TOOL.HubVerificationError, match="reports an error"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            page_fetcher=error_page,
            **TEST_STORAGE_COUNTS,
        )


def test_rejects_when_hub_api_cannot_parse_compact_config(tmp_path: Path) -> None:
    report_path, _payloads, info = fixture(tmp_path)
    info.config = None

    with pytest.raises(TOOL.HubVerificationError, match="did not parse"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            page_fetcher=page_fetcher,
            **TEST_STORAGE_COUNTS,
        )


def test_downloaded_quantization_configs_must_retain_storage_and_agree() -> None:
    declaration = {
        "quant_method": "exl3",
        "bits": 4.0,
        "meta": {
            "ds4rt_error_ledger": {
                "family_join": {"sha256": "6" * 64},
                "run": {"kind": "production"},
            }
        },
    }
    config = json.dumps(
        {"quantization_config": TOOL._compact_exl3_declaration(declaration)}
    ).encode()
    module = "model.layers.3.mlp.experts.0.up_proj"
    entry = {
        "stored_tensors": {
            f"{module}.trellis": {"shape": [1, 1, 64], "torch_dtype": "int16"},
            f"{module}.suh": {"shape": [16], "torch_dtype": "float16"},
            f"{module}.svh": {"shape": [16], "torch_dtype": "float16"},
            f"{module}.mcg": {"shape": [], "torch_dtype": "int32"},
        },
        "quant_format": "exl3",
        "bits_per_weight": 4,
        "mcg_multiplier": TOOL.EXL3_MCG_MULTIPLIER,
    }
    external = json.dumps(
        {**declaration, "tensor_storage": {module: entry}}
    ).encode()

    accepted = TOOL.validate_downloaded_quantization_configs(
        config_payload=config,
        quantize_config_payload=external,
        model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
        **TEST_STORAGE_COUNTS,
    )
    assert accepted["bits"] == 4.0
    assert accepted["tensor_storage_entries"] == 1
    assert TOOL.SHA256_RE.fullmatch(accepted["ledger_provenance_sha256"])

    with pytest.raises(TOOL.HubVerificationError, match="do not agree"):
        TOOL.validate_downloaded_quantization_configs(
            config_payload=json.dumps(
                {
                    "padding": "x" * TOOL.MAX_COMPACT_CONFIG_BYTES,
                    "quantization_config": TOOL._compact_exl3_declaration(
                        declaration
                    ),
                }
            ).encode(),
            quantize_config_payload=external,
            model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
            **TEST_STORAGE_COUNTS,
        )

    with pytest.raises(TOOL.HubVerificationError, match="do not agree"):
        TOOL.validate_downloaded_quantization_configs(
            config_payload=config,
            quantize_config_payload=json.dumps(
                {**declaration, "tensor_storage": {}}
            ).encode(),
            model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
            **TEST_STORAGE_COUNTS,
        )
    with pytest.raises(TOOL.HubVerificationError, match="do not agree"):
        TOOL.validate_downloaded_quantization_configs(
            config_payload=json.dumps(
                {
                    "quantization_config": TOOL._compact_exl3_declaration(
                        declaration | {"bits": 3.0}
                    )
                }
            ).encode(),
            quantize_config_payload=external,
            model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
            **TEST_STORAGE_COUNTS,
        )

    with pytest.raises(TOOL.HubVerificationError, match="inventory is incomplete"):
        TOOL.validate_downloaded_quantization_configs(
            config_payload=config,
            quantize_config_payload=external,
            model_id="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1",
            expected_tensor_storage_entries=2,
            expected_stored_tensor_descriptions=8,
        )

    no_ledger = dict(declaration)
    no_ledger.pop("meta")
    with pytest.raises(TOOL.HubVerificationError, match="ledger is missing"):
        TOOL.validate_downloaded_quantization_configs(
            config_payload=json.dumps(
                {"quantization_config": TOOL._compact_exl3_declaration(no_ledger)}
            ).encode(),
            quantize_config_payload=json.dumps(
                {**no_ledger, "tensor_storage": {module: entry}}
            ).encode(),
            model_id=TOOL.GLM53_MODEL_ID,
            **TEST_STORAGE_COUNTS,
        )


def test_rejects_private_remote_model(tmp_path: Path) -> None:
    report_path, payloads, info = fixture(tmp_path)
    info.private = True

    with pytest.raises(TOOL.HubVerificationError, match="visibility"):
        TOOL.verify(
            publication_report_path=report_path,
            revision="main",
            api=FakeApi(info),
            page_fetcher=page_fetcher,
            **TEST_STORAGE_COUNTS,
        )
