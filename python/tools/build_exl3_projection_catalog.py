#!/usr/bin/env python3
"""Build a one-layer GLMRT TensorCatalog from EXL3 projection checkpoints.

The projection checkpoint store is the durable, resumable output written while
GLM-5 is being quantized.  This utility turns one complete decoder layer into
the same catalog entries that the final safetensors artifact will expose.  It is
intended for pre-publication loader/preload/dispatch qualification on a Spark;
it does not rewrite or duplicate any tensor payloads.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
from pathlib import Path
from typing import Any


MODEL_ID = "wrldsuksgo2mars/GLM-5.2-EXL3-K3-calibrated-v1"
RECIPE = "glm52_exl3_trellis_3bpw_calibrated_natural_route_v1"
GLM53_MODEL_ID = "wrldsuksgo2mars/GLM-5.3-EXL3-K4-v1"
GLM53_RECIPE = "glm53_exl3_trellis_4bpw_calibrated_natural_route_v1"
MODEL_PROFILES = {
    MODEL_ID: {"recipe": RECIPE, "trellis_bits": 3},
    GLM53_MODEL_ID: {"recipe": GLM53_RECIPE, "trellis_bits": 4},
}
EXPERTS = 256
HIDDEN = 6144
INTERMEDIATE = 2048
TOP_K = 8
FIRST_ROUTED_LAYER = 3
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
TENSOR_DTYPES = {
    "trellis": "i16",
    "suh": "f16",
    "svh": "f16",
    "mcg": "i32",
}
MODULE_RE = re.compile(
    r"^model\.layers\.(?P<layer>[0-9]+)\.mlp\.experts\."
    r"(?P<expert>[0-9]+)\.(?P<projection>gate_proj|up_proj|down_proj)$"
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _safetensors_header(path: Path) -> dict[str, dict[str, Any]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"truncated safetensors length in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        data_start = 8 + header_length
        if data_start > file_size:
            raise ValueError(f"safetensors header extends beyond {path}")
        raw_header = handle.read(header_length)
        if len(raw_header) != header_length:
            raise ValueError(f"truncated safetensors header in {path}")
    decoded = json.loads(raw_header)
    if not isinstance(decoded, dict):
        raise ValueError(f"invalid safetensors header in {path}")

    entries: dict[str, dict[str, Any]] = {}
    for name, raw in decoded.items():
        if name == "__metadata__":
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"invalid safetensors entry {name} in {path}")
        offsets = raw.get("data_offsets")
        shape = raw.get("shape")
        dtype = raw.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"invalid safetensors metadata for {name} in {path}")
        absolute_end = data_start + offsets[1]
        if absolute_end > file_size:
            raise ValueError(f"safetensors tensor {name} extends beyond {path}")
        entries[name] = {
            "dtype": dtype,
            "shape": shape,
            "byte_offset": data_start + offsets[0],
            "byte_length": offsets[1] - offsets[0],
        }
    return entries


def _projection_files(root: Path, layer_id: int) -> dict[tuple[int, str], Path]:
    projections: dict[tuple[int, str], Path] = {}
    for manifest_path in root.rglob("*.json"):
        manifest = _json_object(manifest_path)
        module = manifest.get("request", {}).get("module")
        match = MODULE_RE.fullmatch(module or "")
        if match is None or int(match.group("layer")) != layer_id:
            continue
        expert_id = int(match.group("expert"))
        projection = match.group("projection")
        if not 0 <= expert_id < EXPERTS:
            raise ValueError(
                f"layer {layer_id} projection checkpoint has expert {expert_id}, "
                f"expected 0..{EXPERTS - 1}"
            )
        tensor_file = manifest.get("tensor_file")
        if not isinstance(tensor_file, str) or Path(tensor_file).name != tensor_file:
            raise ValueError(f"invalid tensor_file in {manifest_path}")
        tensor_path = manifest_path.with_name(tensor_file)
        if not tensor_path.is_file():
            raise ValueError(f"missing projection tensor file {tensor_path}")
        key = (expert_id, projection)
        if key in projections:
            raise ValueError(
                f"duplicate layer {layer_id} expert {expert_id} {projection} checkpoint"
            )
        projections[key] = tensor_path

    expected = {
        (expert_id, projection)
        for expert_id in range(EXPERTS)
        for projection in PROJECTIONS
    }
    if projections.keys() != expected:
        missing = sorted(expected - projections.keys())[:8]
        unexpected = sorted(projections.keys() - expected)[:8]
        raise ValueError(
            f"layer {layer_id} projection checkpoint set is incomplete: "
            f"expected={len(expected)} found={len(projections)} "
            f"missing={missing} unexpected={unexpected}"
        )
    return projections


def _expected_tensor_shapes(
    projection: str, trellis_bits: int
) -> dict[str, list[int]]:
    if projection not in PROJECTIONS or trellis_bits not in (3, 4):
        raise ValueError(
            f"unsupported EXL3 projection/bitrate {projection!r}/K{trellis_bits}"
        )
    input_width, output_width = (
        (HIDDEN, INTERMEDIATE)
        if projection in {"gate_proj", "up_proj"}
        else (INTERMEDIATE, HIDDEN)
    )
    return {
        "trellis": [input_width // 16, output_width // 16, trellis_bits * 16],
        "suh": [input_width],
        "svh": [output_width],
        "mcg": [],
    }


def build_catalog(
    root: Path, layer_id: int, model_id: str = MODEL_ID
) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"projection checkpoint path is not a directory: {root}")
    if layer_id != FIRST_ROUTED_LAYER:
        raise ValueError(
            "the standalone runtime catalog must use the first routed layer "
            f"{FIRST_ROUTED_LAYER}, got {layer_id}"
        )
    try:
        profile = MODEL_PROFILES[model_id]
    except KeyError as error:
        raise ValueError(f"unsupported EXL3 catalog model ID {model_id!r}") from error

    projection_files = _projection_files(root, layer_id)
    tensors: list[dict[str, Any]] = []
    expected_header_dtypes = {
        "trellis": "I16",
        "suh": "F16",
        "svh": "F16",
        "mcg": "I32",
    }
    for expert_id in range(EXPERTS):
        for projection in PROJECTIONS:
            tensor_path = projection_files[(expert_id, projection)]
            header = _safetensors_header(tensor_path)
            expected_shapes = _expected_tensor_shapes(
                projection, profile["trellis_bits"]
            )
            if set(header) != set(TENSOR_DTYPES):
                raise ValueError(
                    f"{tensor_path} has tensors {sorted(header)}, expected "
                    f"{sorted(TENSOR_DTYPES)}"
                )
            relative_file = os.fspath(tensor_path.relative_to(root))
            base = f"model.layers.{layer_id}.mlp.experts.{expert_id}.{projection}"
            for suffix in ("trellis", "suh", "svh", "mcg"):
                metadata = header[suffix]
                if metadata["dtype"] != expected_header_dtypes[suffix]:
                    raise ValueError(
                        f"{tensor_path}:{suffix} has dtype {metadata['dtype']}, "
                        f"expected {expected_header_dtypes[suffix]}"
                    )
                if metadata["shape"] != expected_shapes[suffix]:
                    raise ValueError(
                        f"{tensor_path}:{suffix} has shape {metadata['shape']}, "
                        f"expected {expected_shapes[suffix]} for "
                        f"{model_id} K{profile['trellis_bits']}"
                    )
                tensors.append(
                    {
                        "name": f"{base}.{suffix}",
                        "file": relative_file,
                        "dtype": TENSOR_DTYPES[suffix],
                        "shape": metadata["shape"],
                        "byte_offset": metadata["byte_offset"],
                        "byte_length": metadata["byte_length"],
                        "role": "routed-expert",
                        "layer_id": layer_id,
                        "expert_id": expert_id,
                        "is_quantization_metadata": suffix != "trellis",
                    }
                )
    tensors.sort(key=lambda tensor: tensor["name"])
    return {
        "model_id": model_id,
        "snapshot_path": os.fspath(root),
        "facts": {
            "model_id": model_id,
            "hidden_size": HIDDEN,
            # Limit this diagnostic catalog to the selected complete layer.
            "num_hidden_layers": layer_id + 1,
            "first_k_dense_replace": FIRST_ROUTED_LAYER,
            "routed_experts": EXPERTS,
            "top_k": TOP_K,
            "quantization_recipe": profile["recipe"],
        },
        "tensors": tensors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, default=FIRST_ROUTED_LAYER)
    parser.add_argument("--model-id", choices=tuple(MODEL_PROFILES), default=MODEL_ID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog = build_catalog(
        args.projection_checkpoint_dir, args.layer_id, model_id=args.model_id
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": os.fspath(output),
                "layer_id": args.layer_id,
                "model_id": args.model_id,
                "trellis_bits": MODEL_PROFILES[args.model_id]["trellis_bits"],
                "experts": EXPERTS,
                "tensors": len(catalog["tensors"]),
                "snapshot_path": catalog["snapshot_path"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
