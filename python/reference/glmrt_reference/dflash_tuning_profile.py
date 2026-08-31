"""Measured launch choices for the production DFlash2 static graphs.

The real-weight tuners reject until these tables equal every measured winner.
Keeping the profile free of Torch/Triton imports lets release qualification
recompute the serving choice without initializing CUDA.
"""

from __future__ import annotations

_CONCURRENCY = (1, 2, 4)
_WIDTHS = tuple(range(1, 8))
_SELECTOR_DTYPES = ("int32", "int64")

# Complete real-weight C1/C2/C4, K1-K7 tuning on the fixed DFlash2 revision.
# The release gate recomputes all entries and rejects if a measured winner
# differs from this serving profile.
_SELECTOR_WARPS = {
    (dtype, concurrency, width): 4
    for dtype in _SELECTOR_DTYPES
    for concurrency in _CONCURRENCY
    for width in _WIDTHS
}
_BODY_WARPS = {
    (concurrency, width): 8 for concurrency in _CONCURRENCY for width in _WIDTHS
}


def dflash2_selector_num_warps(
    active_requests: int,
    proposal_tokens: int,
    candidate_dtype: str,
) -> int:
    try:
        return _SELECTOR_WARPS[(candidate_dtype, active_requests, proposal_tokens)]
    except KeyError as error:
        raise ValueError("invalid DFlash2 selector tuning geometry") from error


def dflash2_body_num_warps(active_requests: int, proposal_tokens: int) -> int:
    try:
        return _BODY_WARPS[(active_requests, proposal_tokens)]
    except KeyError as error:
        raise ValueError("invalid DFlash2 body tuning geometry") from error
