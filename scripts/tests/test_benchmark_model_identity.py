from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
TOOLS = ROOT / "python" / "tools"
sys.path.insert(0, str(TOOLS))

import real_full_matrix as MATRIX  # noqa: E402
import bench_real_full_mtp_acceptance as BLENDED  # noqa: E402
import bench_real_full_concurrency as CONCURRENCY  # noqa: E402
import bench_real_full_needle as NEEDLE  # noqa: E402
import bench_real_full_prefill_curve as PREFILL  # noqa: E402
import bench_real_full_repeat_decode as REPEAT  # noqa: E402
import bench_release_prefill_matrix as PREFILL_MATRIX  # noqa: E402
import bench_release_decode_matrix as DECODE_MATRIX  # noqa: E402


GLM53_MODEL = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"


def test_release_benchmarks_default_to_the_exact_glm53_k4_model() -> None:
    assert MATRIX.MODEL_ID == GLM53_MODEL


def test_release_metadata_tools_can_import_matrix_without_optional_tokenizer_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(TOOLS)!r}); "
                "import real_full_matrix; "
                "assert real_full_matrix.MODEL_ID == "
                f"{GLM53_MODEL!r}"
            ),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr


def test_every_final_benchmark_cli_parses_the_exact_glm53_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cases = (
        (BLENDED, []),
        (CONCURRENCY, ["code"]),
        (
            NEEDLE,
            ["--output", str(tmp_path / "needle.jsonl")],
        ),
        (
            PREFILL,
            [
                "--source",
                str(tmp_path / "corpus.txt"),
                "--nonce-seed",
                "53",
                "--output",
                str(tmp_path / "prefill.jsonl"),
                "--dry-run",
            ],
        ),
        (
            REPEAT,
            [
                "--nonce-seed",
                "53",
                "--output",
                str(tmp_path / "repeat.jsonl"),
            ],
        ),
    )
    for module, arguments in cases:
        monkeypatch.setattr(sys, "argv", [module.__name__, *arguments])
        assert module.parse_args().model == GLM53_MODEL

    assert PREFILL_MATRIX.parse_args([]).model == GLM53_MODEL
    assert DECODE_MATRIX.parse_args([]).model == GLM53_MODEL


def test_eight_type_contract_binds_token_zero_nonce_policy_and_tokenizer() -> None:
    contract = BLENDED.prompt_contract(
        list(BLENDED.WEIGHTED_CASE_IDS),
        suite="weighted",
        repeats=5,
        nonce_seed=53,
        tokenizer_sha256="a" * 64,
        max_tokens=None,
    )
    assert len(contract["cases"]) == 8
    assert sum(case["weight"] for case in contract["cases"]) == 7
    constrained = next(
        case for case in contract["cases"] if case["id"] == "structured-json-schema"
    )
    assert constrained["response_format"]["type"] == "json_schema"
    assert contract["nonce_policy"] == "token-zero"
    assert contract["tokenizer_sha256"] == "a" * 64
    assert contract["request_binding_version"] == BLENDED.REQUEST_BINDING_VERSION


def test_default_tokenizer_is_resolved_from_the_selected_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = (
        tmp_path
        / "hub"
        / "models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1"
        / "snapshots"
        / ("a" * 40)
    )
    snapshot.mkdir(parents=True)
    tokenizer = snapshot / "tokenizer.json"
    tokenizer.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert MATRIX.default_tokenizer_path(GLM53_MODEL) == tokenizer


def test_default_tokenizer_honors_main_ref_over_lexical_snapshot_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_cache = (
        tmp_path
        / "hub"
        / "models--wrldsuksgo2mars--GLM-5.3-EXL3-K4-v1"
    )
    selected_revision = "1" * 40
    stale_revision = "f" * 40
    selected = model_cache / "snapshots" / selected_revision / "tokenizer.json"
    stale = model_cache / "snapshots" / stale_revision / "tokenizer.json"
    selected.parent.mkdir(parents=True)
    stale.parent.mkdir(parents=True)
    selected.write_text("{}\n", encoding="utf-8")
    stale.write_text("{}\n", encoding="utf-8")
    main_ref = model_cache / "refs" / "main"
    main_ref.parent.mkdir(parents=True)
    main_ref.write_text(f"{selected_revision}\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    assert MATRIX.default_tokenizer_path(GLM53_MODEL) == selected


def test_default_tokenizer_never_falls_back_to_a_different_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong_snapshot = (
        tmp_path
        / "hub"
        / "models--lukealonso--GLM-5.2-NVFP4"
        / "snapshots"
        / ("b" * 40)
    )
    wrong_snapshot.mkdir(parents=True)
    (wrong_snapshot / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(tmp_path))

    with pytest.raises(FileNotFoundError, match="GLM-5.3-EXL3-K4-v1"):
        MATRIX.default_tokenizer_path(GLM53_MODEL)
