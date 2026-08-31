from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "_validate_pi_coding_agent_run",
    ROOT / "python" / "tools" / "validate_pi_coding_agent_run.py",
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)
MODEL = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"


def fixture(tmp_path: Path) -> dict:
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (work / "game.html").write_text(
        '<!doctype html><script type="module">import * as THREE from "three"; '
        "const parrot = {}; const food = []; const renderer = new THREE.WebGLRenderer(); "
        "window.addEventListener('keydown', () => {}); requestAnimationFrame(() => {});"
        "</script>",
        encoding="utf-8",
    )
    user = {
        "role": "user",
        "content": [{"type": "text", "text": TOOL.PROMPT}],
        "timestamp": 1,
    }
    first = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "building"},
            {
                "type": "toolCall",
                "id": "call-1",
                "name": "write",
                "arguments": {"path": "game.html", "content": "fixture"},
            },
        ],
        "api": "openai-completions",
        "provider": "glmrt",
        "model": MODEL,
        "usage": {
            "input": 10,
            "output": 20,
            "cacheRead": 0,
            "cacheWrite": 0,
            "reasoning": 2,
            "totalTokens": 30,
        },
        "stopReason": "toolUse",
        "timestamp": 2,
    }
    tool_result = {
        "role": "toolResult",
        "toolCallId": "call-1",
        "toolName": "write",
        "content": [{"type": "text", "text": "ok"}],
        "isError": False,
        "timestamp": 3,
    }
    final = {
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
        "api": "openai-completions",
        "provider": "glmrt",
        "model": MODEL,
        "usage": {
            "input": 3,
            "output": 4,
            "cacheRead": 30,
            "cacheWrite": 0,
            "reasoning": 0,
            "totalTokens": 37,
        },
        "stopReason": "stop",
        "timestamp": 4,
    }
    messages = [user, first, tool_result, final]
    records = [
        {
            "type": "session",
            "version": 3,
            "id": "session-1",
            "timestamp": "2026-08-30T00:00:00Z",
            "cwd": str(work),
        },
        {"type": "agent_start"},
        *({"type": "message_end", "message": message} for message in messages),
        {"type": "agent_end", "messages": messages, "willRetry": False},
        {"type": "agent_settled"},
    ]
    events = tmp_path / "events.jsonl"
    events.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    timing = tmp_path / "time.txt"
    timing.write_text(
        "elapsed_seconds=12.5\n"
        "user_seconds=1.0\n"
        "system_seconds=0.5\n"
        "max_rss_kb=1000\n"
        "exit_status=0\n",
        encoding="utf-8",
    )
    stderr = tmp_path / "stderr.log"
    stderr.write_text("", encoding="utf-8")
    return {
        "events_path": events,
        "time_path": timing,
        "stderr_path": stderr,
        "work_path": work,
        "model_id": MODEL,
        "thinking": "high",
        "pi_version": "0.82.0",
        "node_binary": "/bin/true",
    }


def test_accepts_content_bound_isolated_pi_agent_run(tmp_path: Path) -> None:
    report = TOOL.validate(**fixture(tmp_path))

    assert report["status"] == "accepted"
    assert report["prompt_sha256"] == TOOL.PROMPT_SHA256
    assert report["wall_seconds"] == 12.5
    assert report["turns"] == 2
    assert report["tool_calls"] == 1
    assert report["tool_errors"] == 0
    assert report["usage"] == {
        "fresh_input": 13,
        "cache_read": 30,
        "cache_write": 0,
        "output": 24,
        "reasoning": 2,
        "total": 67,
        "api_reported_total": 67,
    }
    assert report["artifact"]["javascript_syntax"] == "passed"
    assert report["artifact"]["webgl_implementation"] == "three.js"
    assert all(report["artifact"]["game_contract"].values())
    assert len(report["report_sha256"]) == 64


def test_rejects_a_different_prompt_or_model(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    records = [json.loads(line) for line in arguments["events_path"].read_text().splitlines()]
    records[2]["message"]["content"][0]["text"] = "different"
    records[-2]["messages"][0]["content"][0]["text"] = "different"
    arguments["events_path"].write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(TOOL.PiEvidenceError, match="benchmark contract"):
        TOOL.validate(**arguments)

    arguments = fixture(tmp_path / "second")
    arguments["model_id"] = "different/model"
    with pytest.raises(TOOL.PiEvidenceError, match="another model"):
        TOOL.validate(**arguments)


def test_rejects_extra_artifacts_and_invalid_module_javascript(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    (arguments["work_path"] / "extra.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(TOOL.PiEvidenceError, match="exactly one HTML"):
        TOOL.validate(**arguments)

    arguments = fixture(tmp_path / "second")
    (arguments["work_path"] / "game.html").write_text(
        '<script type="module">const = ;</script>', encoding="utf-8"
    )
    arguments["node_binary"] = "/bin/false"
    with pytest.raises(TOOL.PiEvidenceError, match="syntax check"):
        TOOL.validate(**arguments)

    arguments = fixture(tmp_path / "third")
    (arguments["work_path"] / "game.html").write_text(
        '<script type="module">const valid = true;</script>', encoding="utf-8"
    )
    with pytest.raises(TOOL.PiEvidenceError, match="WebGL game contract"):
        TOOL.validate(**arguments)


def test_accepts_a_syntax_checked_raw_webgl_classic_script(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    (arguments["work_path"] / "game.html").write_text(
        """<!doctype html><canvas></canvas><script>
        const canvas = document.querySelector('canvas');
        const gl = canvas.getContext('webgl');
        const parrot = {}; const food = [];
        addEventListener('keydown', () => {});
        requestAnimationFrame(() => gl.clear(gl.COLOR_BUFFER_BIT));
        </script>""",
        encoding="utf-8",
    )
    arguments["node_binary"] = "node"

    report = TOOL.validate(**arguments)

    assert report["artifact"]["classic_scripts"] == 1
    assert report["artifact"]["module_scripts"] == 0
    assert report["artifact"]["webgl_implementation"] == "raw-webgl"


def test_accepts_one_html_artifact_in_a_nested_project_directory(
    tmp_path: Path,
) -> None:
    arguments = fixture(tmp_path)
    source = arguments["work_path"] / "game.html"
    project = arguments["work_path"] / "parrot-game"
    project.mkdir()
    source.rename(project / "index.html")
    records = [
        json.loads(line) for line in arguments["events_path"].read_text().splitlines()
    ]
    for record in records:
        messages = (
            [record["message"]]
            if record.get("type") == "message_end"
            else record.get("messages", [])
            if record.get("type") == "agent_end"
            else []
        )
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") == "toolCall" and item.get("name") == "write":
                    item["arguments"]["path"] = "parrot-game/index.html"
    arguments["events_path"].write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = TOOL.validate(**arguments)

    assert report["artifact"]["path"] == "parrot-game/index.html"


def test_accepts_and_records_a_recovered_tool_failure(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    records = [
        json.loads(line) for line in arguments["events_path"].read_text().splitlines()
    ]
    records[4]["message"]["isError"] = True
    records[-2]["messages"][2]["isError"] = True
    arguments["events_path"].write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    report = TOOL.validate(**arguments)

    assert report["status"] == "accepted"
    assert report["tool_calls"] == 1
    assert report["tool_errors"] == 1


def test_rejects_inconsistent_usage_and_failed_process(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    records = [json.loads(line) for line in arguments["events_path"].read_text().splitlines()]
    records[3]["message"]["usage"]["totalTokens"] = 31
    records[-2]["messages"][1]["usage"]["totalTokens"] = 31
    arguments["events_path"].write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(TOOL.PiEvidenceError, match="usage total"):
        TOOL.validate(**arguments)

    arguments = fixture(tmp_path / "second")
    arguments["time_path"].write_text(
        arguments["time_path"].read_text().replace("exit_status=0", "exit_status=1"),
        encoding="utf-8",
    )
    with pytest.raises(TOOL.PiEvidenceError, match="did not complete"):
        TOOL.validate(**arguments)

    arguments = fixture(tmp_path / "third")
    records = [
        json.loads(line) for line in arguments["events_path"].read_text().splitlines()
    ]
    records[4]["message"]["isError"] = "false"
    records[-2]["messages"][2]["isError"] = "false"
    arguments["events_path"].write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    with pytest.raises(TOOL.PiEvidenceError, match="status is malformed"):
        TOOL.validate(**arguments)


def test_runner_pins_an_isolated_zero_temperature_event_contract() -> None:
    runner = (ROOT / "scripts" / "bench-pi-coding-agent.sh").read_text(
        encoding="utf-8"
    )
    for fragment in (
        ".temperature == 0",
        'baseUrl == "http://127.0.0.1:8000/v1"',
        "--mode json --print --no-session --no-context-files",
        "--no-extensions --no-skills --no-prompt-templates",
        "--tools read,bash,edit,write --no-approve",
        "select(.type != \"message_update\")",
        "benchmark root already exists",
        "validate_pi_coding_agent_run.py",
    ):
        assert fragment in runner


def test_signed_report_reopens_every_pi_input(tmp_path: Path) -> None:
    arguments = fixture(tmp_path)
    report = TOOL.validate(**arguments)
    report_path = tmp_path / "evidence.json"
    TOOL.atomic_json(report_path, report)

    assert TOOL.revalidate(report_path, node_binary="/bin/true") == report
    (arguments["work_path"] / "game.html").write_text(
        (arguments["work_path"] / "game.html").read_text() + "\n<!-- changed -->\n",
        encoding="utf-8",
    )
    with pytest.raises(TOOL.PiEvidenceError, match="differs"):
        TOOL.revalidate(report_path, node_binary="/bin/true")
