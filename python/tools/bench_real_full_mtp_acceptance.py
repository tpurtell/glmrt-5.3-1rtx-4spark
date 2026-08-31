#!/usr/bin/env python3
"""Measure native-MTP acceptance on varied semantic generation tasks."""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import re
import statistics
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench_real_full_concurrency import token_zero_nonces
from real_full_matrix import default_tokenizer_path


@dataclass(frozen=True)
class PromptCase:
    category: str
    prompt: str
    max_tokens: int
    weight: float = 1.0
    json_schema: bool = False


STRUCTURED_EDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "operation": {"type": "string"},
        "line_start": {"type": "integer"},
        "line_end": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["path", "operation", "line_start", "line_end", "rationale"],
    "additionalProperties": False,
}


def structured_edit_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "file_edit",
            "strict": True,
            "schema": STRUCTURED_EDIT_SCHEMA,
        },
    }


CASES = {
    "count": PromptCase(
        "low-entropy",
        "Count from 1 to 64, one number per line. Do not add any other text.",
        160,
    ),
    "repeat": PromptCase(
        "repetition",
        'Repeat the exact text "red green blue" 24 times, one repetition per line. '
        "Do not number the lines.",
        160,
    ),
    "code": PromptCase(
        "code",
        "Write a Python function merge_intervals(intervals) that merges overlapping "
        "integer intervals. Include type hints, a short docstring, and three assert-based "
        "examples. Return only one Python code block.",
        320,
    ),
    "math": PromptCase(
        "reasoning",
        "A shop discounts a $240 jacket by 25%, then applies 8% sales tax to the "
        "discounted price. What is the final price? Show the calculation briefly.",
        128,
    ),
    "fable": PromptCase(
        "creative-prose",
        "Write a self-contained fable of exactly 150 words about two parrots who disagree "
        "about sharing credit. Output no title or preamble. Include the final one-sentence "
        "moral in the 150-word total. Before responding, silently revise the draft until the "
        "entire response is between 140 and 170 words.",
        256,
    ),
    "hello": PromptCase("short-response", "hi", 32),
    "topic": PromptCase(
        "exposition",
        "Explain virtual memory to a junior programmer in five concise bullet points, "
        "including paging, page faults, and the role of the TLB.",
        384,
    ),
    "structured-json": PromptCase(
        "structured-output-natural",
        "Return only a JSON object describing a file edit with keys path, operation, "
        "line_start, line_end, and rationale. Use path src/cache.rs, operation replace, "
        "lines 41 through 47, and a one-sentence rationale about removing a redundant copy.",
        128,
        weight=0.5,
    ),
    "structured-json-schema": PromptCase(
        "structured-output-constrained",
        "Return only a JSON object describing a file edit with keys path, operation, "
        "line_start, line_end, and rationale. Use path src/cache.rs, operation replace, "
        "lines 41 through 47, and a one-sentence rationale about removing a redundant copy.",
        128,
        weight=0.5,
        json_schema=True,
    ),
    "multilingual": PromptCase(
        "multilingual",
        "請用繁體中文，以四個簡短條列解釋什麼是寫入時複製（copy-on-write），"
        "並包含一個行程 fork 後修改記憶體頁面的例子。",
        384,
    ),
}

WEIGHTED_CASE_IDS = tuple(
    case_id for case_id in CASES if case_id not in {"count", "repeat"}
)

# Explicit rare-width diagnostics. They are selectable with --case but are
# excluded from the default weighted corpus because their repetitive syntax is
# deliberately favorable to long speculative windows.
CASES.update(
    {
        "syntax-rust": PromptCase(
            "diagnostic-syntax",
            "Return only a Rust code block declaring enum Op with exactly 128 "
            "variants named Op000 through Op127, one variant per line.",
            512,
        ),
        "syntax-python": PromptCase(
            "diagnostic-syntax",
            "Return only Python code defining POWERS_OF_TWO as a parenthesized "
            "tuple containing 2**0 through 2**127, one expression per line.",
            512,
        ),
    }
)
REACHABILITY_CASE_IDS = ("count", "repeat", "syntax-rust", "syntax-python")
QUALITY_CONTRACT_VERSION = "glmrt-semantic-decode-contract-v3"
REQUEST_BINDING_VERSION = "glmrt-semantic-decode-request-v2"
RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _python_block(content: str) -> tuple[str | None, list[str]]:
    match = re.fullmatch(
        r"\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if match is None:
        return None, ["response is not exactly one Python code block"]
    return match.group("code"), []


def _structured_json_content(content: str, *, allow_fence: bool) -> str:
    stripped = content.strip()
    if not allow_fence:
        return stripped
    match = re.fullmatch(
        r"```(?:json)?\s*\n(?P<json>.*)\n```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group("json").strip() if match is not None else stripped


def validate_case_content(case_id: str, content: str) -> dict[str, Any]:
    """Check prompt-visible contracts without executing generated content."""

    issues: list[str] = []
    stripped = content.strip()
    if not stripped:
        issues.append("response is empty")
    elif case_id == "count":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines != [str(value) for value in range(1, 65)]:
            issues.append("response is not exactly the integers 1 through 64")
    elif case_id == "repeat":
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines != ["red green blue"] * 24:
            issues.append("response is not exactly 24 requested repetition lines")
    elif case_id == "code":
        code, block_issues = _python_block(content)
        issues.extend(block_issues)
        if code is not None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                issues.append("Python code does not parse")
            else:
                functions = [
                    node
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "merge_intervals"
                ]
                if len(functions) != 1:
                    issues.append("merge_intervals function is missing or duplicated")
                else:
                    function = functions[0]
                    if (
                        not function.args.args
                        or function.args.args[0].annotation is None
                        or function.returns is None
                    ):
                        issues.append("merge_intervals lacks requested type hints")
                    if ast.get_docstring(function) is None:
                        issues.append("merge_intervals lacks a docstring")
                if sum(isinstance(node, ast.Assert) for node in ast.walk(tree)) < 3:
                    issues.append("fewer than three assert examples were provided")
    elif case_id == "math":
        normalized = stripped.replace(",", "")
        if re.search(r"(?<![0-9])(?:\$\s*)?194\.4(?:0)?(?![0-9])", normalized) is None:
            issues.append("response does not contain the correct final price 194.40")
        if (
            "240" not in normalized
            or not any(token in normalized for token in ("25", "75", "0.75", ".75"))
            or not any(token in normalized for token in ("8", "1.08"))
        ):
            issues.append("response does not show the requested calculation inputs")
    elif case_id == "fable":
        words = re.findall(r"\b[\w'-]+\b", stripped, flags=re.UNICODE)
        if not 140 <= len(words) <= 170:
            issues.append(f"fable has {len(words)} words, outside 140..170")
        sentence_matches = list(
            re.finditer(r"(?:^|(?<=[.!?]))\s*([^.!?]+[.!?])", stripped)
        )
        final_sentence = (
            sentence_matches[-1].group(1).strip() if sentence_matches else ""
        )
        moral_words = re.findall(r"\b[\w'-]+\b", final_sentence, flags=re.UNICODE)
        moral_terms = (
            "credit",
            "share",
            "sharing",
            "together",
            "cooperat",
            "team",
            "recognition",
            "praise",
            "glory",
            "harmony",
            "humility",
            "fair",
            "both",
        )
        if not 3 <= len(moral_words) <= 32 or not any(
            term in final_sentence.casefold() for term in moral_terms
        ):
            issues.append(
                "response does not end with a concise moral about sharing credit"
            )
    elif case_id == "hello":
        if len(stripped) > 512:
            issues.append("short greeting response is unexpectedly long")
    elif case_id == "topic":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-5][.)])\s+", line)
        ]
        if len(bullets) != 5:
            issues.append(f"response has {len(bullets)} bullets, expected five")
        lowered = stripped.casefold()
        for term in ("paging", "page fault", "tlb"):
            if term not in lowered:
                issues.append(f"response omits {term}")
    elif case_id in {"structured-json", "structured-json-schema"}:
        encoded = _structured_json_content(
            content,
            allow_fence=case_id == "structured-json",
        )
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            issues.append(
                "response is not valid bare-or-fenced JSON"
                if case_id == "structured-json"
                else "constrained response is not bare valid JSON"
            )
        else:
            expected_keys = {"path", "operation", "line_start", "line_end", "rationale"}
            if not isinstance(value, dict) or set(value) != expected_keys:
                issues.append("JSON object has the wrong key set")
            elif (
                value.get("path") != "src/cache.rs"
                or value.get("operation") != "replace"
                or value.get("line_start") != 41
                or value.get("line_end") != 47
                or not isinstance(value.get("rationale"), str)
                or not value["rationale"].strip()
            ):
                issues.append("JSON object does not preserve the requested edit")
    elif case_id == "multilingual":
        bullets = [
            line
            for line in stripped.splitlines()
            if re.match(r"^\s*(?:[-*•]|[1-4][.)、])\s*", line)
        ]
        if len(bullets) != 4:
            issues.append(f"response has {len(bullets)} bullets, expected four")
        lowered = stripped.casefold()
        if not ("寫入時複製" in stripped or "copy-on-write" in lowered):
            issues.append("response omits copy-on-write")
        if "fork" not in lowered or "頁" not in stripped:
            issues.append("response omits the requested fork/page example")
    elif case_id == "syntax-rust":
        variants = [
            int(match.group(1))
            for line in stripped.splitlines()
            if (match := re.match(r"^\s*Op([0-9]{3}),?\s*$", line))
        ]
        if variants != list(range(128)):
            issues.append("Rust enum does not contain exactly Op000 through Op127")
    elif case_id == "syntax-python":
        code, block_issues = _python_block(content)
        issues.extend(block_issues)
        if code is not None:
            try:
                tree = ast.parse(code)
            except SyntaxError:
                issues.append("Python code does not parse")
            else:
                assignments = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "POWERS_OF_TWO"
                        for target in node.targets
                    )
                ]
                if len(assignments) != 1 or not isinstance(
                    assignments[0].value, ast.Tuple
                ):
                    issues.append("POWERS_OF_TWO tuple assignment is missing")
                else:
                    exponents = []
                    for element in assignments[0].value.elts:
                        if (
                            not isinstance(element, ast.BinOp)
                            or not isinstance(element.op, ast.Pow)
                            or not isinstance(element.left, ast.Constant)
                            or element.left.value != 2
                            or not isinstance(element.right, ast.Constant)
                            or not isinstance(element.right.value, int)
                        ):
                            break
                        exponents.append(element.right.value)
                    if exponents != list(range(128)):
                        issues.append("tuple is not exactly 2**0 through 2**127")
    else:
        issues.append(f"no quality validator exists for {case_id}")
    return {
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "quality_contract_passed": not issues,
        "quality_contract_issues": issues,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--reference-url")
    parser.add_argument("--model", default="wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1")
    parser.add_argument(
        "--profile", choices=("balanced", "long", "accuracy"), default="balanced"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Override the selected corpus cases' completion-token budgets.",
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        choices=sorted(CASES),
        help="Run only this case; may be repeated. Overrides --suite.",
    )
    parser.add_argument(
        "--suite",
        choices=("weighted", "reachability", "all"),
        default="weighted",
        help=(
            "Corpus used when --case is omitted. The weighted suite excludes "
            "deliberately easy counting/repetition/syntax reachability probes."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Run the complete selected corpus this many times.",
    )
    parser.add_argument(
        "--nonce-seed",
        type=int,
        help=(
            "Prefix every prompt with a deterministic unique nonce, preventing "
            "prompt-cache reuse in paired performance qualification."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="tokenizer.json used to construct token-zero nonces",
    )
    parser.add_argument(
        "--run-id",
        help="immutable run identity used to bind release evidence to a deployment",
    )
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional JSONL evidence path; refuses to overwrite an existing file",
    )
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    if args.run_id is not None and RUN_ID_RE.fullmatch(args.run_id) is None:
        parser.error("run ID contains unsafe characters")
    return args


def completion_payload(
    model: str,
    case: PromptCase,
    max_tokens: int | None = None,
    prompt_prefix: str = "",
) -> bytes:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_prefix + case.prompt}],
        "temperature": 0,
        "enable_thinking": False,
        "max_tokens": case.max_tokens if max_tokens is None else max_tokens,
    }
    if case.json_schema:
        payload["response_format"] = structured_edit_response_format()
    return json.dumps(payload).encode()


def prompt_contract(
    selected: list[str],
    *,
    suite: str,
    repeats: int,
    nonce_seed: int | None,
    tokenizer_sha256: str | None,
    max_tokens: int | None,
) -> dict[str, Any]:
    """Return the model-independent identity for a matched decode replay."""

    return {
        "suite": suite,
        "cases": [
            {
                "id": case_id,
                "category": CASES[case_id].category,
                "prompt": CASES[case_id].prompt,
                "max_tokens": (
                    CASES[case_id].max_tokens if max_tokens is None else max_tokens
                ),
                "weight": CASES[case_id].weight,
                "response_format": (
                    structured_edit_response_format()
                    if CASES[case_id].json_schema
                    else None
                ),
            }
            for case_id in selected
        ],
        "repeats": repeats,
        "nonce_seed": nonce_seed,
        "nonce_policy": "token-zero" if nonce_seed is not None else "none",
        "tokenizer_sha256": tokenizer_sha256,
        "temperature": 0,
        "enable_thinking": False,
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "request_binding_version": REQUEST_BINDING_VERSION,
    }


def request_completion(url: str, payload: bytes, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def summarize_case(case_id: str, result: dict[str, Any]) -> dict[str, Any]:
    case = CASES[case_id]
    usage = result["usage"]
    metrics = result["metrics"]
    mtp = metrics["real_full"]
    draft_lengths = list(mtp.get("mtp_draft_lengths", []))
    accepted = list(mtp["mtp_accepted_draft_lengths"])
    cycle_ms = list(mtp["mtp_verify_cycle_ms"])
    target_cycle_physical_m = list(mtp["target_cycle_physical_m"])
    target_cycle_ms = list(mtp["target_cycle_ms"])
    verify_cycles = int(mtp["mtp_verify_cycles"])
    drafts = int(mtp["mtp_draft_tokens"])
    accepted_total = int(mtp["mtp_accepted_draft_tokens"])
    emitted = int(mtp["mtp_emitted_tokens_from_verify"])
    total_cycle_ms = float(mtp["mtp_total_verify_cycle_ms"])
    decode_ms = float(metrics["decode_ms"])
    completion_tokens = int(usage["completion_tokens"])
    content = result["choices"][0]["message"]["content"]
    if not (len(draft_lengths) == len(accepted) == len(cycle_ms) == verify_cycles):
        raise RuntimeError("server returned unaligned MTP cycle diagnostics")
    if len(target_cycle_physical_m) != len(target_cycle_ms):
        raise RuntimeError("server returned unaligned target-cycle diagnostics")
    if not math.isclose(
        sum(float(value) for value in target_cycle_ms),
        decode_ms,
        rel_tol=1.0e-9,
        abs_tol=1.0e-6,
    ):
        raise RuntimeError("post-TTFT target-cycle diagnostics do not sum to decode_ms")
    return {
        "case": case_id,
        "category": case.category,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": completion_tokens,
        "finish_reason": result["choices"][0]["finish_reason"],
        "decode_ms": decode_ms,
        "decode_tps": (
            (completion_tokens - 1) / (decode_ms / 1_000.0)
            if completion_tokens > 1 and decode_ms > 0.0
            else 0.0
        ),
        "verify_cycles": verify_cycles,
        "draft_tokens": drafts,
        "accepted_draft_tokens": accepted_total,
        "accepted_draft_rate": accepted_total / drafts if drafts else 0.0,
        "draft_lengths": draft_lengths,
        "mean_accepted_draft_length": statistics.mean(accepted) if accepted else 0.0,
        "accepted_draft_lengths": accepted,
        "verify_cycle_ms": cycle_ms,
        "target_cycle_physical_m": target_cycle_physical_m,
        "target_cycle_ms": target_cycle_ms,
        "full_match_cycles": int(mtp["mtp_full_match_cycles"]),
        "emitted_tokens_from_verify": emitted,
        "emitted_tokens_per_verify_cycle": (
            emitted / verify_cycles if verify_cycles else 0.0
        ),
        "emitted_tokens_per_verify_cycle_second": (
            emitted / (total_cycle_ms / 1_000.0) if total_cycle_ms > 0.0 else 0.0
        ),
        "runtime_captures": int(mtp["request_coordinator_graph_captures"]),
        "content_chars": len(content),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "content_preview": content[:160].replace("\n", "\\n"),
        **validate_case_content(case_id, content),
    }


def main() -> None:
    args = parse_args()
    benchmark_started_ns = time.time_ns()
    run_id = args.run_id or dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.max_tokens is not None and args.max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.cases:
        selected = args.cases
        selected_suite = "explicit"
    elif args.suite == "weighted":
        selected = list(WEIGHTED_CASE_IDS)
        selected_suite = "weighted"
    elif args.suite == "reachability":
        selected = list(REACHABILITY_CASE_IDS)
        selected_suite = "reachability"
    else:
        selected = list(CASES)
        selected_suite = "all"
    summaries = []
    repeat_summaries = []
    tokenizer_sha256 = None
    nonces: list[dict[str, Any]] = []
    if args.nonce_seed is not None:
        tokenizer_path = (
            (args.tokenizer or default_tokenizer_path(args.model))
            .expanduser()
            .resolve(strict=True)
        )
        tokenizer_sha256 = hash_file(tokenizer_path)
        nonces = token_zero_nonces(
            count=args.repeats * len(selected),
            seed=args.nonce_seed,
            tokenizer_path=tokenizer_path,
        )
    destination = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        destination = args.output.open("x", encoding="utf-8")

    def emit(value: dict[str, Any]) -> None:
        line = json.dumps(value, ensure_ascii=False)
        print(line, flush=True)
        if destination is not None:
            destination.write(line + "\n")
            destination.flush()

    for repeat_index in range(args.repeats):
        repeat_cases = []
        for case_id in selected:
            request_index = repeat_index * len(selected) + len(repeat_cases)
            nonce = nonces[request_index] if nonces else None
            prompt_prefix = (
                nonce["prefix"] + "Treat the preceding request nonce as irrelevant.\n"
                if nonce is not None
                else ""
            )
            payload = completion_payload(
                args.model,
                CASES[case_id],
                args.max_tokens,
                prompt_prefix,
            )
            result = request_completion(args.url, payload, args.timeout)
            summary = summarize_case(case_id, result)
            summary["run_id"] = run_id
            summary["timestamp_utc"] = dt.datetime.now(dt.UTC).isoformat()
            summary["profile"] = args.profile
            summary["repeat"] = repeat_index + 1
            summary["prompt"] = prompt_prefix + CASES[case_id].prompt
            summary["request_sha256"] = hashlib.sha256(payload).hexdigest()
            summary["prompt_sha256"] = hashlib.sha256(
                (prompt_prefix + CASES[case_id].prompt).encode()
            ).hexdigest()
            summary["nonce"] = (
                {
                    "marker": nonce["marker"],
                    "first_content_token_id": nonce["first_content_token_id"],
                }
                if nonce is not None
                else None
            )
            if args.reference_url:
                reference = request_completion(
                    args.reference_url, payload, args.timeout
                )
                summary["reference_content_match"] = (
                    reference["choices"][0]["message"]["content"]
                    == result["choices"][0]["message"]["content"]
                )
                summary["reference_finish_reason_match"] = (
                    reference["choices"][0]["finish_reason"] == summary["finish_reason"]
                )
                summary["reference_completion_tokens_match"] = (
                    int(reference["usage"]["completion_tokens"])
                    == summary["completion_tokens"]
                )
            repeat_cases.append(summary)
            summaries.append(summary)
            emit(summary)

        repeat_timed_tokens = sum(
            CASES[summary["case"]].weight * (summary["completion_tokens"] - 1)
            for summary in repeat_cases
        )
        repeat_decode_ms = sum(
            CASES[summary["case"]].weight * summary["decode_ms"]
            for summary in repeat_cases
        )
        repeat_emitted = sum(
            summary["emitted_tokens_from_verify"] for summary in repeat_cases
        )
        repeat_verify_ms = sum(
            sum(summary["verify_cycle_ms"]) for summary in repeat_cases
        )
        repeat_summaries.append(
            {
                "repeat": repeat_index + 1,
                "wall_decode_tps": (
                    repeat_timed_tokens / (repeat_decode_ms / 1_000.0)
                    if repeat_decode_ms > 0.0
                    else 0.0
                ),
                "emitted_tokens_per_verify_cycle_second": (
                    repeat_emitted / (repeat_verify_ms / 1_000.0)
                    if repeat_verify_ms > 0.0
                    else 0.0
                ),
            }
        )

    accepted_histogram = Counter(
        length for summary in summaries for length in summary["accepted_draft_lengths"]
    )
    draft_histogram = Counter(
        length for summary in summaries for length in summary["draft_lengths"]
    )
    physical_m_histogram = Counter(
        length + 1 for summary in summaries for length in summary["draft_lengths"]
    )
    emitted_length_histogram = Counter(
        length + 1
        for summary in summaries
        for length in summary["accepted_draft_lengths"]
    )
    accepted_by_physical_m: dict[int, Counter[int]] = {}
    full_matches_by_physical_m = Counter()
    target_cycle_ms_by_physical_m: dict[int, list[float]] = {}
    scalar_cycles = sum(
        max(
            0,
            summary["completion_tokens"] - summary["emitted_tokens_from_verify"],
        )
        for summary in summaries
    )
    physical_m_histogram[1] += scalar_cycles
    emitted_length_histogram[1] += scalar_cycles
    accepted_by_physical_m[1] = Counter({0: scalar_cycles})
    for summary in summaries:
        for drafts, accepted, cycle_ms in zip(
            summary["draft_lengths"],
            summary["accepted_draft_lengths"],
            summary["verify_cycle_ms"],
            strict=True,
        ):
            physical_m = drafts + 1
            accepted_by_physical_m.setdefault(physical_m, Counter())[accepted] += 1
            if accepted == drafts:
                full_matches_by_physical_m[physical_m] += 1
        for physical_m, cycle_ms in zip(
            summary["target_cycle_physical_m"],
            summary["target_cycle_ms"],
            strict=True,
        ):
            target_cycle_ms_by_physical_m.setdefault(physical_m, []).append(cycle_ms)
    total_drafts = sum(summary["draft_tokens"] for summary in summaries)
    total_accepted = sum(summary["accepted_draft_tokens"] for summary in summaries)
    total_emitted = sum(summary["emitted_tokens_from_verify"] for summary in summaries)
    total_cycles = sum(summary["verify_cycles"] for summary in summaries)
    total_cycle_ms = sum(sum(summary["verify_cycle_ms"]) for summary in summaries)
    total_timed_tokens = sum(
        CASES[summary["case"]].weight * (summary["completion_tokens"] - 1)
        for summary in summaries
    )
    total_decode_ms = sum(
        CASES[summary["case"]].weight * summary["decode_ms"]
        for summary in summaries
    )
    wall_samples = [summary["wall_decode_tps"] for summary in repeat_summaries]
    verifier_samples = [
        summary["emitted_tokens_per_verify_cycle_second"]
        for summary in repeat_summaries
    ]
    replay_contract = prompt_contract(
        selected,
        suite=selected_suite,
        repeats=args.repeats,
        nonce_seed=args.nonce_seed,
        tokenizer_sha256=tokenizer_sha256,
        max_tokens=args.max_tokens,
    )
    aggregate = {
        "schema": "glmrt-mtp-acceptance-aggregate-v4",
        "run_id": run_id,
        "benchmark_started_ns": benchmark_started_ns,
        "benchmark_completed_ns": time.time_ns(),
        "timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "profile": args.profile,
        "model": args.model,
        "endpoint": args.url,
        "nonce_seed": args.nonce_seed,
        "tokenizer_sha256": tokenizer_sha256,
        "prompt_contract": replay_contract,
        "prompt_contract_sha256": hashlib.sha256(
            json.dumps(
                replay_contract,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        ).hexdigest(),
        "suite": selected_suite,
        "selected_case_ids": selected,
        "case_weights": {case_id: CASES[case_id].weight for case_id in selected},
        "cases": len(summaries),
        "cases_per_repeat": len(selected),
        "corpus_repeats": args.repeats,
        "repeat_summaries": repeat_summaries,
        "wall_decode_tps": (
            total_timed_tokens / (total_decode_ms / 1_000.0)
            if total_decode_ms > 0.0
            else 0.0
        ),
        "median_repeat_wall_decode_tps": statistics.median(wall_samples),
        "min_repeat_wall_decode_tps": min(wall_samples),
        "max_repeat_wall_decode_tps": max(wall_samples),
        "stdev_repeat_wall_decode_tps": (
            statistics.stdev(wall_samples) if len(wall_samples) > 1 else 0.0
        ),
        "verify_cycles": total_cycles,
        "scalar_cycles": scalar_cycles,
        "target_cycles": scalar_cycles + total_cycles,
        "measured_post_ttft_target_cycles": sum(
            len(values) for values in target_cycle_ms_by_physical_m.values()
        ),
        "draft_tokens": total_drafts,
        "accepted_draft_tokens": total_accepted,
        "accepted_draft_rate": total_accepted / total_drafts if total_drafts else 0.0,
        "draft_length_histogram": dict(sorted(draft_histogram.items())),
        "physical_m_histogram": dict(sorted(physical_m_histogram.items())),
        "accepted_draft_length_histogram": dict(sorted(accepted_histogram.items())),
        "emitted_length_histogram": dict(sorted(emitted_length_histogram.items())),
        "accepted_drafts_by_physical_m": {
            physical_m: dict(sorted(histogram.items()))
            for physical_m, histogram in sorted(accepted_by_physical_m.items())
        },
        "full_matches_by_physical_m": dict(sorted(full_matches_by_physical_m.items())),
        "target_cycle_physical_m_histogram": {
            physical_m: len(values)
            for physical_m, values in sorted(target_cycle_ms_by_physical_m.items())
        },
        "target_cycle_ms_by_physical_m": {
            physical_m: {
                "samples": len(values),
                "total_ms": sum(values),
                "mean_ms": statistics.mean(values),
                "median_ms": statistics.median(values),
                "min_ms": min(values),
                "max_ms": max(values),
            }
            for physical_m, values in sorted(target_cycle_ms_by_physical_m.items())
        },
        "max_selected_physical_m": max(physical_m_histogram, default=1),
        "max_emitted_tokens_in_cycle": max(emitted_length_histogram, default=1),
        "emitted_tokens_from_verify": total_emitted,
        "emitted_tokens_per_verify_cycle": (
            total_emitted / total_cycles if total_cycles else 0.0
        ),
        "emitted_tokens_per_verify_cycle_second": (
            total_emitted / (total_cycle_ms / 1_000.0) if total_cycle_ms > 0.0 else 0.0
        ),
        "median_repeat_emitted_tokens_per_verify_cycle_second": statistics.median(
            verifier_samples
        ),
        "min_repeat_emitted_tokens_per_verify_cycle_second": min(verifier_samples),
        "max_repeat_emitted_tokens_per_verify_cycle_second": max(verifier_samples),
        "stdev_repeat_emitted_tokens_per_verify_cycle_second": (
            statistics.stdev(verifier_samples) if len(verifier_samples) > 1 else 0.0
        ),
        "all_zero_runtime_captures": all(
            summary["runtime_captures"] == 0 for summary in summaries
        ),
        "quality_contract_version": QUALITY_CONTRACT_VERSION,
        "all_quality_contracts_passed": all(
            summary["quality_contract_passed"] for summary in summaries
        ),
        "quality_contract_failures": [
            {
                "case": summary["case"],
                "repeat": summary["repeat"],
                "issues": summary["quality_contract_issues"],
            }
            for summary in summaries
            if not summary["quality_contract_passed"]
        ],
    }
    if args.reference_url:
        aggregate["all_reference_outputs_match"] = all(
            summary["reference_content_match"]
            and summary["reference_finish_reason_match"]
            and summary["reference_completion_tokens_match"]
            for summary in summaries
        )
    emit({"aggregate": aggregate})
    if destination is not None:
        destination.close()


if __name__ == "__main__":
    main()
