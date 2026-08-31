"""Initial GLM-5.3 TP4 EXL3 K4 Spark kernel profile.

Large bucket boundaries and launch geometry inherit the already-qualified K3
profile as a conservative starting point.  K4 additionally compiles every
exact M through 32 because native-MTP and DFlash2 verification can produce any
of those row counts.  Each exact small-M shape gets an independent profile
surface so final model route replay can tune it without moving the retained
GLM-5.2 K3 path.
"""

from __future__ import annotations

from _b12x_exl3_k3_profile import (
    EXL3_K3_AOT_REGIMES,
    exl3_k3_grid_x,
    exl3_k3_tile_config,
)


EXL3_K4_AOT_REGIMES = (*range(1, 33), 64, 128, 256, 257, 512, 1024, 2048, 2064)
EXL3_K4_REQUIRED_LIVE_ROWS = (
    *range(1, 33),
    64,
    128,
    129,
    256,
    257,
    512,
    513,
    1024,
    1025,
    2048,
    2049,
    2064,
)


def _inherited_k3_capacity(capacity_rows: int) -> int:
    for inherited in EXL3_K3_AOT_REGIMES:
        if capacity_rows <= inherited:
            return inherited
    raise ValueError(f"unsupported EXL3 K4 AOT capacity {capacity_rows}")


def exl3_k4_tile_config(capacity_rows: int) -> tuple[int, int, int, int]:
    capacity_rows = int(capacity_rows)
    if capacity_rows not in EXL3_K4_AOT_REGIMES:
        raise ValueError(f"unsupported EXL3 K4 AOT capacity {capacity_rows}")
    return exl3_k3_tile_config(_inherited_k3_capacity(capacity_rows))


def exl3_k4_grid_x(capacity_rows: int) -> int:
    capacity_rows = int(capacity_rows)
    if capacity_rows not in EXL3_K4_AOT_REGIMES:
        raise ValueError(f"unsupported EXL3 K4 AOT capacity {capacity_rows}")
    return exl3_k3_grid_x(_inherited_k3_capacity(capacity_rows))


def exl3_k4_route_block_rows(capacity_rows: int) -> int:
    capacity_rows = int(capacity_rows)
    if capacity_rows not in EXL3_K4_AOT_REGIMES:
        raise ValueError(f"unsupported EXL3 K4 AOT capacity {capacity_rows}")
    route_count = capacity_rows * 8
    for block_rows in (8, 16, 32, 48, 64):
        if 10 * route_count < 9 * 256 * block_rows:
            return block_rows
    return 64


def exl3_k4_capacity_rows(live_rows: int) -> int:
    live_rows = int(live_rows)
    if live_rows <= 0:
        raise ValueError("EXL3 K4 live rows must be positive")
    for capacity_rows in EXL3_K4_AOT_REGIMES:
        if live_rows <= capacity_rows:
            return capacity_rows
    raise ValueError(
        f"EXL3 K4 live rows {live_rows} exceed maximum "
        f"{EXL3_K4_AOT_REGIMES[-1]}"
    )
