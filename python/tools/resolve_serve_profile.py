#!/usr/bin/env python3
"""Resolve or launch one GLMRT production profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from glmrt_reference.serve_profiles import (
    resolve_serve_profile,
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Resolve balanced/long/accuracy GLMRT launch settings."
    )
    parser.add_argument(
        "--profile", choices=("balanced", "long", "accuracy"), default="balanced"
    )
    parser.add_argument(
        "--speculation",
        choices=("plain", "mtp", "dspark", "dflash2"),
        default="dspark",
    )
    parser.add_argument("--vision", choices=("on", "off"), default="off")
    parser.add_argument(
        "--mtp-bf16-experts",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "override retained BF16 layer-78 experts; off startup-quantizes "
            "them to NVFP4"
        ),
    )
    parser.add_argument(
        "--model",
        choices=("luke", "nvidia", "exl3", "glm53-exl3"),
        default="luke",
        help="supported text checkpoint",
    )
    parser.add_argument("--headroom-gib", type=float, default=8.0)
    parser.add_argument("--gpu-total-mib", type=int)
    parser.add_argument("--max-context-tokens", type=int)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--kv-pool-tokens", type=int)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--dflash2-fixed-drafts",
        type=int,
        choices=range(1, 8),
        help="qualification-only fixed DFlash2 width in 1..7; omitted selects adaptive K1-K7",
    )
    parser.add_argument(
        "--dflash2-topk-backend",
        choices=("torch", "flashinfer", "flashinfer-dsa"),
        default="torch",
        help="DFlash2 top-16 backend; torch is qualified by the full-service A/B gate",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        help="artifact root; defaults to the source tree containing this tool",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-unqualified",
        action="store_true",
        help="permit a diagnostic launch despite explicit profile blockers",
    )
    parser.add_argument(
        "--launcher",
        help="launcher path; defaults to scripts/real-full-tcp-serve.sh",
    )
    return parser.parse_known_args()


def main() -> int:
    args, launcher_args = parse_args()
    repo_root = (
        args.repo_root.expanduser().resolve()
        if args.repo_root is not None
        else Path(__file__).resolve().parents[2]
    )
    resolved = resolve_serve_profile(
        repo_root=repo_root,
        profile=args.profile,
        speculation=args.speculation,
        vision=args.vision == "on",
        model=args.model,
        headroom_gib=args.headroom_gib,
        gpu_total_mib=args.gpu_total_mib,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        kv_pool_tokens=args.kv_pool_tokens,
        concurrency=args.concurrency,
        dflash2_fixed_drafts=args.dflash2_fixed_drafts,
        dflash2_topk_backend=args.dflash2_topk_backend,
        inherited_environment=os.environ,
    )
    if args.mtp_bf16_experts != "auto":
        resolved.environment["GLMRT_MTP_BF16_EXPERTS"] = (
            "1" if args.mtp_bf16_experts == "on" else "0"
        )

    if args.dry_run:
        print(resolved.to_json())
        return 0

    environment = os.environ.copy()
    environment.update(resolved.environment)
    if args.mtp_bf16_experts != "auto":
        environment["GLMRT_MTP_BF16_EXPERTS"] = (
            "1" if args.mtp_bf16_experts == "on" else "0"
        )
    if resolved.blockers and not args.allow_unqualified:
        print(
            json.dumps(
                {
                    "error": "profile has launch blockers",
                    "blockers": resolved.blockers,
                    "hint": "fix the blockers or use --allow-unqualified for diagnostics",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    launcher = (
        Path(args.launcher).expanduser()
        if args.launcher
        else repo_root / "scripts" / "real-full-tcp-serve.sh"
    )
    os.execve(str(launcher), [str(launcher), *launcher_args], environment)
    raise AssertionError("execve returned")


if __name__ == "__main__":
    raise SystemExit(main())
