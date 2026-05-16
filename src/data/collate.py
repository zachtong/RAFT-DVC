"""Shared collation helpers for Phase-1 dataloaders.

PyTorch's ``default_collate`` recursively merges dict-typed fields by key,
which breaks when per-sample dicts have inhomogeneous keys -- e.g. our
``meta`` field carries different ``coefficients`` shapes depending on the
sample's ``deform_type`` (translation / affine / quadratic).

``safe_phase1_collate`` stacks tensor fields normally and keeps
non-tensor fields (``meta``, ``filename``) as plain lists.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


# Tensor fields we recognise and stack into a batch dim.  Any other key
# in the sample dict is collected as a list.
_TENSOR_KEYS = ("vol0", "vol1", "flow", "mask")


def safe_phase1_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of Phase-1 sample dicts into one batch dict.

    Tensor fields (``vol0``, ``vol1``, ``flow``, ``mask``) are stacked
    along a new batch dimension.  All other fields are returned as plain
    lists -- crucially preserving per-sample ``meta`` dicts whose keys
    may differ between samples.

    Args:
        batch: list of sample dicts, each containing at least the keys
            in ``_TENSOR_KEYS`` that are present in sample[0].

    Returns:
        Dict with stacked tensors and list-typed extras.
    """
    if not batch:
        return {}

    out: Dict[str, Any] = {}
    sample0 = batch[0]

    for key in _TENSOR_KEYS:
        if key in sample0:
            out[key] = torch.stack([item[key] for item in batch], dim=0)

    # All non-tensor fields collected as lists (preserves heterogeneity).
    for key in sample0:
        if key in _TENSOR_KEYS:
            continue
        out[key] = [item.get(key) for item in batch]

    return out


__all__ = ["safe_phase1_collate"]
