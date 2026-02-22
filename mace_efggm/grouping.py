"""Grouping utilities for E-FGGM masks."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

import torch


def module_wise_groups(named_parameters: Iterable) -> Dict[str, List[str]]:
    """Group parameters by top-level MACE module names."""
    groups: Dict[str, List[str]] = defaultdict(list)
    keys = {
        "embedding": "embedding",
        "interactions": "interaction_blocks",
        "radial": "radial_mlps",
        "readout": "readout",
    }
    for name, _ in named_parameters:
        low = name.lower()
        group = "other"
        for k, v in keys.items():
            if k in low:
                group = v
                break
        groups[group].append(name)
    return dict(groups)


def irreps_wise_groups(named_parameters: Iterable) -> Dict[str, List[str]]:
    """Best-effort grouping by irreps/tensor-block hints in parameter names.

    TODO: replace this with explicit e3nn Irreps parsing per block/channel in this fork.
    """
    groups: Dict[str, List[str]] = defaultdict(list)
    for name, _ in named_parameters:
        lname = name.lower()
        if "l0" in lname or "0e" in lname:
            groups["irreps_l0"].append(name)
        elif "l1" in lname or "1o" in lname or "1e" in lname:
            groups["irreps_l1"].append(name)
        elif "l2" in lname or "2e" in lname or "2o" in lname:
            groups["irreps_l2"].append(name)
        else:
            groups["irreps_misc"].append(name)
    return dict(groups)


def group_scores(
    fisher_tensors: Dict[str, torch.Tensor], groups: Dict[str, List[str]]
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for group_name, param_names in groups.items():
        if not param_names:
            out[group_name] = 0.0
            continue
        vals = [fisher_tensors[n].mean() for n in param_names if n in fisher_tensors]
        out[group_name] = float(torch.stack(vals).mean().item()) if vals else 0.0
    return out
