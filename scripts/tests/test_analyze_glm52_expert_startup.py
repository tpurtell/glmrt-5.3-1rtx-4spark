from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOL_PATH = ROOT / "python" / "tools" / "analyze_glm52_expert_startup.py"
SPEC = importlib.util.spec_from_file_location("_glmrt_expert_startup", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
RUNTIME_FINGERPRINT = "a" * 64


def log_text(
    *,
    exl3: bool,
    model: str | None = None,
    missing_layer: int | None = None,
    exit_status: int | None = None,
    cooperative: bool = False,
    include_mtp: bool = False,
    resident_bytes: int | None = None,
    role: str = "spark-0",
    runtime_fingerprint: str = RUNTIME_FINGERPRINT,
) -> str:
    model = model or (
        TOOL.EXL3_MODEL if exl3 else "lukealonso/GLM-5.2-NVFP4"
    )
    geometry = TOOL.EXL3_GEOMETRY_BY_MODEL.get(model)
    if exl3:
        assert geometry is not None
        if resident_bytes is None:
            resident_bytes = geometry.resident_bytes_per_rank_layer
    lines = [
        "== 2026-08-22T00:00:00+00:00 starting expert-9100: stale ==",
        "expertd_startup_phase stage=loadplan elapsed_ms=999.000 total_ms=999.000",
        "== 2026-08-23T00:00:00+00:00 starting expert-9100: current ==",
        "starting expertd synthetic_weights=false transport=verbs-host "
        f"listen=0.0.0.0:9100 model_id={model} "
        f"runtime_identity={runtime_fingerprint} loadplan=None "
        f'catalog_source=Some("hf://fixture") real_layer=None role=Some("{role}")',
    ]
    stages = [
        "loadplan",
        "python-capture",
        "catalog-owner-config",
        "catalog-filter-validation",
        "executor-configuration",
    ]
    for index, stage in enumerate(stages, 1):
        lines.append(
            f"expertd_startup_phase stage={stage} elapsed_ms={index}.000 "
            f"total_ms={index}.000"
        )
    if exl3:
        for layer in range(3, 78):
            if layer == missing_layer:
                continue
            if cooperative:
                lines.append(
                    "real_exl3_cuda_layer_preload "
                    f"layer_id={layer} experts=256 source_experts=64 cooperative=true "
                    "packed_exchange=true "
                    f"source_bytes={geometry.cooperative_source_bytes_per_rank_layer} "
                    "source_requests=768 "
                    "source_spans=1 direct_io=true source_gbps=12.000 load_ms=75.000 "
                    "pack_ms=65.000 allocation_ms=20.000 upload_ms=35.000 "
                    f"exchange_ms=40.000 resident_bytes={resident_bytes}"
                )
            else:
                lines.append(
                    "real_exl3_cuda_layer_preload "
                    f"layer_id={layer} experts=256 cooperative=false direct_resident=true "
                    f"source_bytes={geometry.direct_source_bytes_per_rank_layer} "
                    "source_gbps=2.900 allocation_ms=37.000 direct_ms=321.000 "
                    f"resident_bytes={resident_bytes}"
                )
    expected_layers = 75 + int(include_mtp)
    expected_projection_groups = expected_layers * 256 * 3
    cuda_weight_bytes = resident_bytes * 75 if exl3 else 68_714_572_800
    cuda_weight_scale_bytes = 0
    if include_mtp:
        assert exl3 and geometry is not None
        assert geometry.startup_mtp_weight_bytes_per_rank_layer is not None
        assert geometry.startup_mtp_scale_bytes_per_rank_layer is not None
        cuda_weight_bytes += geometry.startup_mtp_weight_bytes_per_rank_layer
        cuda_weight_scale_bytes = geometry.startup_mtp_scale_bytes_per_rank_layer
    lines.extend(
        [
            "expertd_startup_phase stage=resident-preload elapsed_ms=30000.000 total_ms=30005.000",
            "expertd_real_weight_resident_preload "
            f"projection_groups={expected_projection_groups} layers={expected_layers} "
            f"experts={expected_layers * 256} weight_bytes=0 "
            "quant_metadata_bytes=0 route_cache_entries=0 route_cache_loads=0 "
            f"route_cache_hits=0 projection_row_entries={expected_projection_groups} "
            "projection_row_loads=0 "
            "projection_row_hits=0 cuda_reference_enabled=true "
            f"cuda_projection_groups={expected_projection_groups} "
            f"cuda_weight_bytes={cuda_weight_bytes} "
            f"cuda_weight_scale_bytes={cuda_weight_scale_bytes} "
            f"cuda_projection_entries={expected_projection_groups} "
            f"cuda_projection_uploads={expected_projection_groups} cuda_cache_hits=0",
            "expertd_startup_phase stage=service-handoff elapsed_ms=1.000 total_ms=30006.000",
        ]
    )
    if exit_status is not None:
        lines.append(
            f"== 2026-08-23T01:00:00+00:00 expert-9100 exited status={exit_status} =="
        )
    return "\n".join(lines) + "\n"


def four_logs(
    tmp_path: Path,
    *,
    exl3: bool,
    model: str | None = None,
    missing_layer: int | None = None,
    exit_status: int | None = None,
    cooperative: bool = False,
    include_mtp: bool = False,
    resident_bytes: int | None = None,
    runtime_fingerprint: str = RUNTIME_FINGERPRINT,
):
    logs = []
    for index, host in enumerate(("ostrich", "dodo", "emu", "kiwi")):
        path = tmp_path / f"{host}.log"
        path.write_text(
            log_text(
                exl3=exl3,
                model=model,
                missing_layer=missing_layer if index == 0 else None,
                exit_status=exit_status,
                cooperative=cooperative,
                include_mtp=include_mtp,
                resident_bytes=resident_bytes,
                role=f"spark-{index}",
                runtime_fingerprint=runtime_fingerprint,
            ),
            encoding="utf-8",
        )
        logs.append((host, path))
    return logs


def test_accepts_complete_four_host_direct_exl3_startup(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model=TOOL.EXL3_MODEL,
        weight_format="exl3",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=True),
    )

    assert report["status"] == "accepted"
    assert report["summary"]["maximum_resident_preload_ms"] == 30000.0
    assert report["hosts"][0]["log"]["selected_start_line"] == 3
    assert report["hosts"][0]["process"]["model_id"] == TOOL.EXL3_MODEL
    assert report["hosts"][0]["exl3"]["layers"] == 75
    assert report["preload_mode"] == "direct-resident"
    assert report["hosts"][0]["exl3"]["preload_mode"] == "direct-resident"
    assert report["hosts"][0]["resident"]["projection_groups"] == 57600


def test_accepts_coalesced_cooperative_exl3_startup(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model=TOOL.EXL3_MODEL,
        weight_format="exl3",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=True, cooperative=True),
    )

    assert report["preload_mode"] == "cooperative-coalesced"
    assert report["hosts"][0]["exl3"]["source_requests"] == 75 * 768
    assert report["hosts"][0]["exl3"]["pack_ms"] == 75 * 65.0


@pytest.mark.parametrize("cooperative", [False, True])
def test_accepts_glm53_k4_exl3_startup(
    tmp_path: Path, cooperative: bool
) -> None:
    geometry = TOOL.EXL3_GEOMETRY_BY_MODEL[TOOL.GLM53_EXL3_MODEL]
    report = TOOL.analyze(
        model=TOOL.GLM53_EXL3_MODEL,
        weight_format="exl3",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(
            tmp_path,
            exl3=True,
            model=TOOL.GLM53_EXL3_MODEL,
            cooperative=cooperative,
        ),
    )

    expected_mode = (
        "cooperative-coalesced" if cooperative else "direct-resident"
    )
    expected_source_bytes = 75 * (
        geometry.cooperative_source_bytes_per_rank_layer
        if cooperative
        else geometry.direct_source_bytes_per_rank_layer
    )
    assert report["schema"] == TOOL.GLM53_SCHEMA
    assert report["model"] == TOOL.GLM53_EXL3_MODEL
    assert report["preload_mode"] == expected_mode
    assert report["hosts"][0]["exl3"]["trellis_bits"] == 4
    assert report["hosts"][0]["exl3"]["source_bytes"] == expected_source_bytes
    assert report["hosts"][0]["exl3"]["resident_bytes"] == (
        75 * geometry.resident_bytes_per_rank_layer
    )
    assert report["hosts"][0]["resident"]["cuda_weight_bytes"] == (
        75 * geometry.resident_bytes_per_rank_layer
    )


def test_accepts_glm53_k4_native_mtp_startup(tmp_path: Path) -> None:
    geometry = TOOL.EXL3_GEOMETRY_BY_MODEL[TOOL.GLM53_EXL3_MODEL]
    assert geometry.startup_mtp_weight_bytes_per_rank_layer == 1_207_959_552
    assert geometry.startup_mtp_scale_bytes_per_rank_layer == 150_994_944
    report = TOOL.analyze(
        model=TOOL.GLM53_EXL3_MODEL,
        weight_format="exl3",
        cache_state="warm",
        include_mtp=True,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(
            tmp_path,
            exl3=True,
            model=TOOL.GLM53_EXL3_MODEL,
            include_mtp=True,
        ),
    )

    host = report["hosts"][0]
    assert report["include_mtp"] is True
    assert host["resident"]["layers"] == 76
    assert host["resident"]["projection_groups"] == 58_368
    assert host["resident"]["cuda_weight_bytes"] == (
        75 * geometry.resident_bytes_per_rank_layer
        + geometry.startup_mtp_weight_bytes_per_rank_layer
    )
    assert host["resident"]["cuda_weight_scale_bytes"] == (
        geometry.startup_mtp_scale_bytes_per_rank_layer
    )
    assert host["exl3"]["startup_quantized_mtp"] == {
        "included": True,
        "weight_bytes": 1_207_959_552,
        "weight_scale_bytes": 150_994_944,
    }


def test_rejects_glm52_k3_native_mtp_startup(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="does not support"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="warm",
            include_mtp=True,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True),
        )


def test_rejects_wrong_direct_resident_geometry(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="resident geometry"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, resident_bytes=123),
        )


def test_rejects_incomplete_exl3_layer_coverage(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="layer coverage"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, missing_layer=37),
        )


def test_accepts_nvfp4_startup_without_exl3_layer_lines(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model="lukealonso/GLM-5.2-NVFP4",
        weight_format="nvfp4",
        cache_state="cold",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=False),
    )

    assert all(host["exl3"] is None for host in report["hosts"])


def test_accepts_orderly_container_stop_after_complete_startup(tmp_path: Path) -> None:
    report = TOOL.analyze(
        model=TOOL.EXL3_MODEL,
        weight_format="exl3",
        cache_state="warm",
        include_mtp=False,
        expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
        logs=four_logs(tmp_path, exl3=True, exit_status=143),
    )

    assert all(host["process"]["exit_status"] == 143 for host in report["hosts"])


def test_rejects_failed_process_after_startup(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="exited unsuccessfully"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="warm",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, exit_status=1),
        )


def test_rejects_model_label_that_differs_from_launched_process(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="launched model"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=False),
        )


def test_rejects_logs_from_another_expert_runtime(tmp_path: Path) -> None:
    with pytest.raises(TOOL.StartupError, match="runtime identity differs"):
        TOOL.analyze(
            model=TOOL.EXL3_MODEL,
            weight_format="exl3",
            cache_state="cold",
            include_mtp=False,
            expert_runtime_fingerprint=RUNTIME_FINGERPRINT,
            logs=four_logs(tmp_path, exl3=True, runtime_fingerprint="b" * 64),
        )
