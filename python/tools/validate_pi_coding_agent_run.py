#!/usr/bin/env python3
"""Validate one isolated Pi coding-agent benchmark event stream and artifact."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any


SCHEMA = "glmrt-pi-coding-agent-evidence-v1"
PROMPT = "make a webgl game of a parrot flying around to steal food from people"
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()
THINKING_LEVELS = frozenset(("off", "high"))
TIME_FIELDS = frozenset(
    ("elapsed_seconds", "user_seconds", "system_seconds", "max_rss_kb", "exit_status")
)
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?$")


class PiEvidenceError(RuntimeError):
    """The Pi run is incomplete, mismatched, or not reproducible evidence."""


class InlineScripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.active: str | None = None
        self.current: list[str] = []
        self.scripts: list[tuple[str, str]] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = {key.lower(): value for key, value in attrs}
        script_type = (attributes.get("type") or "").lower()
        if (
            tag.lower() == "script"
            and "src" not in attributes
            and script_type
            in {"", "module", "text/javascript", "application/javascript"}
        ):
            if self.active is not None:
                raise PiEvidenceError("artifact contains nested inline scripts")
            self.active = "module" if script_type == "module" else "classic"
            self.current = []

    def handle_data(self, data: str) -> None:
        if self.active is not None:
            self.current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script" and self.active is not None:
            self.scripts.append((self.active, "".join(self.current)))
            self.active = None
            self.current = []


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise PiEvidenceError(f"{label} is a symbolic link")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise PiEvidenceError(f"{label} is not one regular file")
    return resolved


def jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise PiEvidenceError(
                        f"Pi event {line_number} is not a JSON object"
                    )
                records.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PiEvidenceError("Pi event stream is not valid UTF-8 JSONL") from error
    if not records:
        raise PiEvidenceError("Pi event stream is empty")
    return records


def nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PiEvidenceError(f"{label} is not a nonnegative integer")
    return value


def parse_time(path: Path) -> dict[str, int | float]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in fields:
            raise PiEvidenceError("Pi time report is malformed")
        fields[key] = value
    if set(fields) != TIME_FIELDS:
        raise PiEvidenceError("Pi time report has an unexpected field set")
    try:
        result: dict[str, int | float] = {
            "elapsed_seconds": float(fields["elapsed_seconds"]),
            "user_seconds": float(fields["user_seconds"]),
            "system_seconds": float(fields["system_seconds"]),
            "max_rss_kb": int(fields["max_rss_kb"]),
            "exit_status": int(fields["exit_status"]),
        }
    except ValueError as error:
        raise PiEvidenceError("Pi time report contains a nonnumeric value") from error
    if (
        any(
            not math.isfinite(float(result[field])) or float(result[field]) < 0.0
            for field in ("elapsed_seconds", "user_seconds", "system_seconds")
        )
        or result["elapsed_seconds"] <= 0.0
        or result["max_rss_kb"] <= 0
        or result["exit_status"] != 0
    ):
        raise PiEvidenceError("Pi process did not complete successfully")
    return result


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        raise PiEvidenceError("Pi message content is malformed")
    texts = [item.get("text") for item in content if isinstance(item, dict) and item.get("type") == "text"]
    if any(not isinstance(value, str) for value in texts):
        raise PiEvidenceError("Pi text message is malformed")
    return "".join(texts)


def validate_usage(message: dict[str, Any], model_id: str) -> dict[str, int]:
    if (
        message.get("role") != "assistant"
        or message.get("api") != "openai-completions"
        or message.get("provider") != "glmrt"
        or message.get("model") != model_id
    ):
        raise PiEvidenceError("Pi assistant message used another model or provider")
    usage = message.get("usage")
    if not isinstance(usage, dict):
        raise PiEvidenceError("Pi assistant message has no usage")
    values = {
        field: nonnegative_int(usage.get(field), f"Pi usage {field}")
        for field in (
            "input",
            "output",
            "cacheRead",
            "cacheWrite",
            "reasoning",
            "totalTokens",
        )
    }
    if values["totalTokens"] != sum(
        values[field] for field in ("input", "output", "cacheRead", "cacheWrite")
    ):
        raise PiEvidenceError("Pi usage total does not match its token fields")
    if values["reasoning"] > values["output"]:
        raise PiEvidenceError("Pi reasoning tokens exceed output tokens")
    return values


def artifact(
    work: Path, *, node_binary: str
) -> tuple[dict[str, Any], set[str]]:
    expanded = work.expanduser()
    if expanded.is_symlink():
        raise PiEvidenceError("Pi work directory is a symbolic link")
    root = expanded.resolve(strict=True)
    if not root.is_dir():
        raise PiEvidenceError("Pi work path is not a directory")
    entries = sorted(root.rglob("*"))
    if any(
        path.is_symlink() or not (path.is_file() or path.is_dir()) for path in entries
    ):
        raise PiEvidenceError("Pi work directory contains a link or special entry")
    files = [path for path in entries if path.is_file()]
    if len(files) != 1 or files[0].suffix.lower() != ".html":
        raise PiEvidenceError("Pi must produce exactly one HTML artifact")
    output = files[0]
    payload = output.read_bytes()
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PiEvidenceError("Pi HTML artifact is not UTF-8") from error
    parser = InlineScripts()
    parser.feed(source)
    if parser.active is not None or not parser.scripts:
        raise PiEvidenceError("Pi HTML artifact has no complete inline JavaScript")
    lowered = source.casefold()
    three_renderer = "three" in lowered and "three." in lowered and "webglrenderer" in lowered
    raw_renderer = (
        "getcontext('webgl" in lowered
        or 'getcontext("webgl' in lowered
        or "getcontext('webgl2" in lowered
        or 'getcontext("webgl2' in lowered
    )
    game_contract = {
        "webgl_renderer": three_renderer or raw_renderer,
        "parrot": "parrot" in lowered,
        "food": "food" in lowered,
        "animation_loop": "requestanimationframe" in lowered,
        "interactive_controls": "addeventlistener" in lowered,
    }
    for kind, script in parser.scripts:
        command = [node_binary, "--check"]
        if kind == "module":
            command.insert(1, "--input-type=module")
        checked = subprocess.run(
            command,
            input=script,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if checked.returncode != 0:
            raise PiEvidenceError("Pi HTML JavaScript failed Node syntax check")
    if not all(game_contract.values()):
        raise PiEvidenceError("Pi HTML artifact does not implement the WebGL game contract")
    relative = output.relative_to(root).as_posix()
    return (
        {
            "path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "classic_scripts": sum(kind == "classic" for kind, _ in parser.scripts),
            "module_scripts": sum(kind == "module" for kind, _ in parser.scripts),
            "javascript_syntax": "passed",
            "webgl_implementation": "three.js" if three_renderer else "raw-webgl",
            "game_contract": game_contract,
        },
        {relative},
    )


def validate(
    *,
    events_path: Path,
    time_path: Path,
    stderr_path: Path,
    work_path: Path,
    model_id: str,
    thinking: str,
    pi_version: str,
    node_binary: str = "node",
) -> dict[str, Any]:
    if not model_id or thinking not in THINKING_LEVELS:
        raise PiEvidenceError("Pi model or thinking level is invalid")
    if VERSION_RE.fullmatch(pi_version) is None:
        raise PiEvidenceError("Pi version is invalid")
    events_file = regular_file(events_path, "Pi event stream")
    time_file = regular_file(time_path, "Pi time report")
    stderr_file = regular_file(stderr_path, "Pi stderr log")
    records = jsonl(events_file)
    sessions = [record for record in records if record.get("type") == "session"]
    agent_end = [record for record in records if record.get("type") == "agent_end"]
    if (
        len(sessions) != 1
        or sessions[0].get("version") != 3
        or not isinstance(sessions[0].get("id"), str)
        or not isinstance(sessions[0].get("timestamp"), str)
        or len(agent_end) != 1
        or sum(record.get("type") == "agent_start" for record in records) != 1
        or sum(record.get("type") == "agent_settled" for record in records) != 1
    ):
        raise PiEvidenceError("Pi event stream has an incomplete session lifecycle")
    try:
        session_started = datetime.fromisoformat(
            sessions[0]["timestamp"].replace("Z", "+00:00")
        )
    except ValueError as error:
        raise PiEvidenceError("Pi session timestamp is invalid") from error
    if session_started.tzinfo is None:
        raise PiEvidenceError("Pi session timestamp has no timezone")
    work = work_path.expanduser().resolve(strict=True)
    try:
        event_cwd = Path(sessions[0]["cwd"]).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise PiEvidenceError("Pi session working directory is invalid") from error
    if event_cwd != work:
        raise PiEvidenceError("Pi session did not run inside the isolated work directory")

    message_ends = [
        record.get("message")
        for record in records
        if record.get("type") == "message_end"
    ]
    if any(not isinstance(message, dict) for message in message_ends):
        raise PiEvidenceError("Pi message-end event is malformed")
    final_messages = agent_end[0].get("messages")
    if not isinstance(final_messages, list) or final_messages != message_ends:
        raise PiEvidenceError("Pi final transcript differs from message-end events")
    user_messages = [message for message in message_ends if message.get("role") == "user"]
    assistant_messages = [
        message for message in message_ends if message.get("role") == "assistant"
    ]
    tool_results = [
        message for message in message_ends if message.get("role") == "toolResult"
    ]
    if (
        len(user_messages) != 1
        or message_text(user_messages[0]) != PROMPT
        or not assistant_messages
        or assistant_messages[-1].get("stopReason") != "stop"
        or any(
            message.get("stopReason") not in {"toolUse", "stop"}
            for message in assistant_messages
        )
    ):
        raise PiEvidenceError("Pi transcript does not match the benchmark contract")

    usage = [validate_usage(message, model_id) for message in assistant_messages]
    tool_calls = []
    for message in assistant_messages:
        content = message.get("content")
        if not isinstance(content, list):
            raise PiEvidenceError("Pi assistant content is malformed")
        tool_calls.extend(
            item
            for item in content
            if isinstance(item, dict) and item.get("type") == "toolCall"
        )
    if len(tool_results) != len(tool_calls):
        raise PiEvidenceError("Pi tool calls and results are incomplete")
    tool_ids = [call.get("id") for call in tool_calls]
    result_ids = [message.get("toolCallId") for message in tool_results]
    if (
        any(not isinstance(value, str) or not value for value in tool_ids)
        or tool_ids != result_ids
    ):
        raise PiEvidenceError("Pi tool result identity is invalid")
    if any(not isinstance(message.get("isError"), bool) for message in tool_results):
        raise PiEvidenceError("Pi tool result status is malformed")
    tool_errors = sum(message["isError"] for message in tool_results)

    artifact_report, artifact_paths = artifact(work, node_binary=node_binary)
    written_paths = set()
    for call in tool_calls:
        if call.get("name") not in {"write", "edit", "bash", "read"}:
            raise PiEvidenceError("Pi used an unexpected tool")
        arguments = call.get("arguments")
        if not isinstance(arguments, dict):
            raise PiEvidenceError("Pi tool arguments are malformed")
        if call.get("name") in {"write", "edit"}:
            path = arguments.get("path")
            if not isinstance(path, str):
                raise PiEvidenceError("Pi file tool has no path")
            relative = PurePosixPath(path)
            if relative.is_absolute() or ".." in relative.parts:
                raise PiEvidenceError("Pi file tool escaped the isolated work directory")
            written_paths.add(relative.as_posix())
    if not written_paths or not artifact_paths.issubset(written_paths):
        raise PiEvidenceError("Pi artifact was not produced by a recorded file tool")

    timing = parse_time(time_file)
    totals = {
        field: sum(item[field] for item in usage)
        for field in (
            "input",
            "output",
            "cacheRead",
            "cacheWrite",
            "reasoning",
            "totalTokens",
        )
    }
    publication_total = totals["input"] + totals["cacheRead"] + totals["output"]
    body = {
        "schema": SCHEMA,
        "status": "accepted",
        "model_id": model_id,
        "provider": "glmrt",
        "api": "openai-completions",
        "pi_version": pi_version,
        "thinking": thinking,
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "session_id": sessions[0]["id"],
        "session_timestamp": sessions[0].get("timestamp"),
        "work": os.fspath(work),
        "wall_seconds": timing["elapsed_seconds"],
        "turns": len(assistant_messages),
        "tool_calls": len(tool_calls),
        "tool_errors": tool_errors,
        "usage": {
            "fresh_input": totals["input"],
            "cache_read": totals["cacheRead"],
            "cache_write": totals["cacheWrite"],
            "output": totals["output"],
            "reasoning": totals["reasoning"],
            "total": publication_total,
            "api_reported_total": totals["totalTokens"],
        },
        "process": timing,
        "artifact": artifact_report,
        "inputs": {
            "events": {
                "path": os.fspath(events_file),
                "bytes": events_file.stat().st_size,
                "sha256": hash_file(events_file),
            },
            "time": {
                "path": os.fspath(time_file),
                "bytes": time_file.stat().st_size,
                "sha256": hash_file(time_file),
            },
            "stderr": {
                "path": os.fspath(stderr_file),
                "bytes": stderr_file.stat().st_size,
                "sha256": hash_file(stderr_file),
            },
        },
    }
    return {**body, "report_sha256": hashlib.sha256(canonical_json(body)).hexdigest()}


def revalidate(path: Path, *, node_binary: str = "node") -> dict[str, Any]:
    report_path = regular_file(path, "Pi evidence report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PiEvidenceError("Pi evidence report is not valid JSON") from error
    inputs = report.get("inputs") if isinstance(report, dict) else None
    body = (
        {key: value for key, value in report.items() if key != "report_sha256"}
        if isinstance(report, dict)
        else None
    )
    if (
        not isinstance(report, dict)
        or report.get("schema") != SCHEMA
        or report.get("status") != "accepted"
        or not isinstance(body, dict)
        or report.get("report_sha256")
        != hashlib.sha256(canonical_json(body)).hexdigest()
        or not isinstance(inputs, dict)
        or any(
            not isinstance(inputs.get(name), dict)
            or not isinstance(inputs[name].get("path"), str)
            for name in ("events", "time", "stderr")
        )
        or not isinstance(report.get("work"), str)
    ):
        raise PiEvidenceError("Pi evidence report is not accepted signed evidence")
    measured = validate(
        events_path=Path(inputs["events"]["path"]),
        time_path=Path(inputs["time"]["path"]),
        stderr_path=Path(inputs["stderr"]["path"]),
        work_path=Path(report["work"]),
        model_id=report["model_id"],
        thinking=report["thinking"],
        pi_version=report["pi_version"],
        node_binary=node_binary,
    )
    if measured != report:
        raise PiEvidenceError("Pi evidence report differs from its source files")
    return report


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as target:
            target.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--time", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", choices=sorted(THINKING_LEVELS), required=True)
    parser.add_argument("--pi-version", required=True)
    parser.add_argument("--node", default="node")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(
        events_path=args.events,
        time_path=args.time,
        stderr_path=args.stderr,
        work_path=args.work,
        model_id=args.model,
        thinking=args.thinking,
        pi_version=args.pi_version,
        node_binary=args.node,
    )
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
