#!/usr/bin/env python3
"""Build a content-bound GLM-5 K3/K4 TP4 route-replay profile from live logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path


LEGACY_SCHEMA = "glmrt-glm52-exl3-route-profile-v1"
SCHEMA = "glmrt-glm5-exl3-route-profile-v1"
GLM53_MODEL_ID = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
MARKER = "protocol_v2_expert_queue_plan "
SOURCE_KIND_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
REQUIRED_FIELDS = {
    "trace_schema",
    "capture_id",
    "transport",
    "request_id_base",
    "layer_id",
    "host",
    "host_index",
    "rows",
    "routes",
    "source_rows",
    "expert_route_counts",
}


class ProfileError(RuntimeError):
    pass


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ProfileError(f"symbolic links are not accepted: {expanded}")
    resolved = expanded.resolve(strict=True)
    if not resolved.is_file():
        raise ProfileError(f"not one regular file: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"invalid log label: {label}")
    return label, Path(raw_path)


def _unsigned(value: str, field: str, *, positive: bool = False) -> int:
    try:
        parsed = int(value, 10)
    except ValueError as error:
        raise ProfileError(f"invalid {field} integer: {value!r}") from error
    if parsed < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ProfileError(f"{field} must be {qualifier}, got {parsed}")
    return parsed


def _count_pairs(
    value: str,
    field: str,
    *,
    numeric_keys: bool,
    max_numeric_key: int | None = None,
) -> list[tuple[int | str, int]]:
    if not value:
        return []
    parsed: list[tuple[int | str, int]] = []
    seen: set[int | str] = set()
    for item in value.split(","):
        key_text, separator, count_text = item.partition(":")
        if not separator:
            raise ProfileError(f"invalid {field} entry: {item!r}")
        if numeric_keys:
            key: int | str = _unsigned(key_text, f"{field} key")
            if max_numeric_key is not None and key >= max_numeric_key:
                raise ProfileError(
                    f"{field} expert {key} is outside 0..{max_numeric_key - 1}"
                )
        else:
            if not SOURCE_KIND_RE.fullmatch(key_text):
                raise ProfileError(f"invalid {field} source kind: {key_text!r}")
            key = key_text
        if key in seen:
            raise ProfileError(f"duplicate {field} key: {key!r}")
        seen.add(key)
        parsed.append((key, _unsigned(count_text, f"{field} count", positive=True)))
    return sorted(parsed, key=lambda pair: pair[0])


def _parse_trace_line(
    line: str,
    *,
    label: str,
    line_number: int,
    experts: int,
    top_k: int,
    max_rows: int,
    capture_id: str,
) -> dict[str, object] | None:
    marker_offset = line.find(MARKER)
    if marker_offset < 0:
        return None
    payload = line[marker_offset + len(MARKER) :].strip()
    fields: dict[str, str] = {}
    for token in payload.split():
        key, separator, value = token.partition("=")
        if not separator or not key:
            raise ProfileError(f"{label}:{line_number}: malformed trace token {token!r}")
        if key in fields:
            raise ProfileError(f"{label}:{line_number}: duplicate trace field {key}")
        fields[key] = value
    missing = sorted(REQUIRED_FIELDS - fields.keys())
    # WIP logs are append-only and can contain the older unversioned trace.
    # Only the content-complete v2 contract is eligible for model tuning.
    if "trace_schema" not in fields:
        return None
    if missing:
        raise ProfileError(
            f"{label}:{line_number}: trace uses an obsolete/incomplete contract; "
            f"missing {','.join(missing)}"
        )
    if fields["trace_schema"] != "2":
        raise ProfileError(
            f"{label}:{line_number}: unsupported trace_schema={fields['trace_schema']}"
        )
    if fields["capture_id"] != capture_id:
        return None

    rows = _unsigned(fields["rows"], "rows", positive=True)
    if rows > max_rows:
        raise ProfileError(f"{label}:{line_number}: rows {rows} exceed {max_rows}")
    routes = _unsigned(fields["routes"], "routes", positive=True)
    if routes != rows * top_k:
        raise ProfileError(
            f"{label}:{line_number}: routes {routes} do not equal rows {rows} * top-k {top_k}"
        )
    source_rows = _count_pairs(fields["source_rows"], "source_rows", numeric_keys=False)
    if sum(count for _, count in source_rows) != rows:
        raise ProfileError(f"{label}:{line_number}: source_rows do not sum to rows={rows}")
    route_counts = _count_pairs(
        fields["expert_route_counts"],
        "expert_route_counts",
        numeric_keys=True,
        max_numeric_key=experts,
    )
    if sum(count for _, count in route_counts) != routes:
        raise ProfileError(
            f"{label}:{line_number}: expert_route_counts do not sum to routes={routes}"
        )
    if any(count > rows for _, count in route_counts):
        raise ProfileError(
            f"{label}:{line_number}: one expert is routed more than once per row"
        )

    return {
        "log": label,
        "line": line_number,
        "transport": fields["transport"],
        "request_id_base": _unsigned(fields["request_id_base"], "request_id_base"),
        "layer_id": _unsigned(fields["layer_id"], "layer_id"),
        "host": fields["host"],
        "host_index": _unsigned(fields["host_index"], "host_index"),
        "rows": rows,
        "routes": routes,
        "source_rows": [[key, count] for key, count in source_rows],
        "expert_route_counts": [[key, count] for key, count in route_counts],
    }


def _parse_log(
    label: str,
    path: Path,
    *,
    experts: int,
    top_k: int,
    max_rows: int,
    capture_id: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    identity = _file_identity(path)
    records: list[dict[str, object]] = []
    with Path(identity["path"]).open("r", encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            record = _parse_trace_line(
                line,
                label=label,
                line_number=line_number,
                experts=experts,
                top_k=top_k,
                max_rows=max_rows,
                capture_id=capture_id,
            )
            if record is not None:
                records.append(record)
    if not records:
        raise ProfileError(
            f"{label}: no {MARKER.strip()} trace_schema=2 capture_id={capture_id} records found"
        )
    identity["label"] = label
    identity["trace_records"] = len(records)
    return identity, records


def _deployment_binding(path: Path) -> dict[str, object]:
    identity = _file_identity(path)
    try:
        value = json.loads(Path(identity["path"]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"invalid deployment evidence: {error}") from error
    if not isinstance(value, dict):
        raise ProfileError("deployment evidence must be a JSON object")
    selected = {
        key: value[key]
        for key in (
            "schema",
            "model",
            "model_id",
            "model_revision",
            "profile",
            "fingerprints",
            "power_limits",
        )
        if key in value
    }
    return {"file": identity, "selected_fields": selected}


def _collapse_tp4_samples(
    records: list[dict[str, object]], expected_hosts: int
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        groups[
            (
                record["log"],
                record["transport"],
                record["request_id_base"],
                record["layer_id"],
            )
        ].append(record)

    expected_indices = list(range(expected_hosts))
    samples: list[dict[str, object]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda record: int(record["host_index"]))
        indices = [int(record["host_index"]) for record in group]
        if indices != expected_indices:
            raise ProfileError(
                "incomplete or duplicate TP host group "
                f"log={key[0]} request_id_base={key[2]} layer_id={key[3]}: "
                f"got {indices}, expected {expected_indices}"
            )
        reference = group[0]
        shared_fields = (
            "rows",
            "routes",
            "source_rows",
            "expert_route_counts",
        )
        for record in group[1:]:
            for field in shared_fields:
                if record[field] != reference[field]:
                    raise ProfileError(
                        "TP route replication mismatch "
                        f"log={key[0]} request_id_base={key[2]} layer_id={key[3]} "
                        f"host_index={record['host_index']} field={field}"
                    )
        counts = [int(pair[1]) for pair in reference["expert_route_counts"]]
        route_shape = {
            "active_experts": len(counts),
            "minimum_expert_reuse_rows": min(counts),
            "maximum_expert_reuse_rows": max(counts),
            "reuse_rows_sum": sum(counts),
            "padded_route_slots_by_block_rows": [
                [
                    block_rows,
                    sum(
                        ((count + block_rows - 1) // block_rows) * block_rows
                        for count in counts
                    ),
                ]
                for block_rows in (8, 16, 32, 48, 64)
            ],
        }
        samples.append(
            {
                "log": reference["log"],
                "transport": reference["transport"],
                "request_id_base": reference["request_id_base"],
                "layer_id": reference["layer_id"],
                "rows": reference["rows"],
                "routes": reference["routes"],
                "source_rows": reference["source_rows"],
                "expert_route_counts": reference["expert_route_counts"],
                "route_shape": route_shape,
                "hosts": [record["host"] for record in group],
                "host_lines": [record["line"] for record in group],
            }
        )
    return samples


def _summary(samples: list[dict[str, object]], trace_records: int) -> dict[str, object]:
    row_histogram = Counter(int(sample["rows"]) for sample in samples)
    layer_histogram = Counter(int(sample["layer_id"]) for sample in samples)
    active_expert_histogram = Counter(
        int(sample["route_shape"]["active_experts"]) for sample in samples
    )
    maximum_reuse_histogram = Counter(
        int(sample["route_shape"]["maximum_expert_reuse_rows"])
        for sample in samples
    )
    source_rows: Counter[str] = Counter()
    for sample in samples:
        for source_kind, rows in sample["source_rows"]:
            source_rows[str(source_kind)] += int(rows)
    return {
        "tp4_samples": len(samples),
        "trace_records": trace_records,
        "layer_ids": sorted(layer_histogram),
        "samples_by_layer": [[key, layer_histogram[key]] for key in sorted(layer_histogram)],
        "samples_by_rows": [[key, row_histogram[key]] for key in sorted(row_histogram)],
        "samples_by_active_experts": [
            [key, active_expert_histogram[key]] for key in sorted(active_expert_histogram)
        ],
        "samples_by_maximum_expert_reuse_rows": [
            [key, maximum_reuse_histogram[key]] for key in sorted(maximum_reuse_histogram)
        ],
        "source_rows": [[key, source_rows[key]] for key in sorted(source_rows)],
        "exact_m9_samples": row_histogram[9],
        "prefill_tail_samples": sum(
            count for rows, count in row_histogram.items() if 2048 < rows <= 2064
        ),
    }


def _write_atomic(path: Path, value: dict[str, object]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as output:
            output.write(json.dumps(value, indent=2, sort_keys=True).encode())
            output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", required=True, type=_labeled_path)
    parser.add_argument("--deployment", required=True, type=Path)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--expected-hosts", type=int, default=4)
    parser.add_argument("--expected-layer-first", type=int, default=3)
    parser.add_argument("--expected-layer-last", type=int, default=77)
    parser.add_argument("--max-rows", type=int, default=2064)
    parser.add_argument(
        "--trellis-bits",
        type=int,
        choices=(3, 4),
        default=3,
        help="model's checkpoint-native EXL3 bitrate (default: K3)",
    )
    args = parser.parse_args()

    if args.experts <= 0 or args.top_k <= 0 or args.expected_hosts <= 0:
        parser.error("experts, top-k, and expected-hosts must be positive")
    if args.expected_layer_first < 0 or args.expected_layer_last < args.expected_layer_first:
        parser.error("invalid expected layer range")
    if args.max_rows <= 0:
        parser.error("max-rows must be positive")
    labels = [label for label, _ in args.log]
    if len(labels) != len(set(labels)):
        parser.error("log labels must be unique")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.capture_id):
        parser.error("capture-id must contain 1..128 alphanumeric, '.', '_', or '-' characters")

    try:
        identities: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        for label, path in args.log:
            identity, parsed = _parse_log(
                label,
                path,
                experts=args.experts,
                top_k=args.top_k,
                max_rows=args.max_rows,
                capture_id=args.capture_id,
            )
            identities.append(identity)
            records.extend(parsed)
        samples = _collapse_tp4_samples(records, args.expected_hosts)
        observed_layers = {int(sample["layer_id"]) for sample in samples}
        expected_layers = set(range(args.expected_layer_first, args.expected_layer_last + 1))
        if observed_layers != expected_layers:
            missing = sorted(expected_layers - observed_layers)
            unexpected = sorted(observed_layers - expected_layers)
            raise ProfileError(
                f"route profile layer coverage mismatch: missing={missing} unexpected={unexpected}"
            )
        deployment = _deployment_binding(args.deployment)
        deployment_fields = deployment.get("selected_fields")
        if args.trellis_bits == 4 and (
            not isinstance(deployment_fields, dict)
            or deployment_fields.get("model_id") != GLM53_MODEL_ID
        ):
            raise ProfileError(
                f"K4 route profile requires deployment model_id={GLM53_MODEL_ID}"
            )
        report: dict[str, object] = {
            "schema": LEGACY_SCHEMA if args.trellis_bits == 3 else SCHEMA,
            "status": "accepted",
            "capture_id": args.capture_id,
            "geometry": {
                "experts": args.experts,
                "top_k": args.top_k,
                "tp_world_size": args.expected_hosts,
                "max_rows": args.max_rows,
                "layer_first": args.expected_layer_first,
                "layer_last": args.expected_layer_last,
                "trellis_bits": args.trellis_bits,
            },
            "deployment": deployment,
            "logs": identities,
            "summary": _summary(samples, len(records)),
            "samples": samples,
        }
        report["report_sha256"] = hashlib.sha256(_canonical_json(report)).hexdigest()
        _write_atomic(args.output, report)
    except (OSError, ProfileError) as error:
        parser.error(str(error))

    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
