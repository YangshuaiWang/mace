"""Grouping utilities for E-FGGM masks."""

from __future__ import annotations

from collections import defaultdict
import logging
import re
from typing import Dict, Iterable, List

import torch


LOGGER = logging.getLogger(__name__)
_EMITTED_IRREPS_SUMMARY = False


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


def _extract_l_values_from_irreps_obj(irreps_obj) -> list[int]:
    l_values: list[int] = []
    if irreps_obj is None:
        return l_values
    if isinstance(irreps_obj, str):
        for match in re.findall(r"(\d+)[eo]?", irreps_obj):
            l_values.append(int(match))
        return l_values
    try:
        for item in irreps_obj:
            ir = getattr(item, "ir", None)
            if ir is not None and hasattr(ir, "l"):
                l_values.append(int(ir.l))
                continue
            if isinstance(item, tuple) and len(item) == 2:
                ir = item[1]
                if hasattr(ir, "l"):
                    l_values.append(int(ir.l))
    except TypeError:
        pass
    return l_values


def _extract_l_values_from_module(module: torch.nn.Module) -> list[int]:
    out: set[int] = set()
    for attr in ("irreps_in", "irreps_out", "irreps", "irreps_hidden"):
        out.update(_extract_l_values_from_irreps_obj(getattr(module, attr, None)))
    return sorted(out)


def _infer_l_from_name(name: str) -> int | None:
    lower = name.lower()
    m = re.search(r"(?:^|[_.])l(\d+)(?:$|[_.])", lower)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:^|[_.])(\d+)[eo](?:$|[_.])", lower)
    if m:
        return int(m.group(1))
    return None


def group_params_by_irreps(model: torch.nn.Module, irreps_grouping: str = "by_l") -> Dict[str, List[torch.nn.Parameter]]:
    """Best-effort parameter grouping by irreps angular momentum index l.

    Groups use names ``irrep_l{l}`` for ``by_l`` and ``irrep_l{l}::{module_path}``
    for ``by_l_and_block``. Parameters with no inferred l are assigned to
    ``irrep_unknown``.
    """

    module_l_map = {module_name: _extract_l_values_from_module(module) for module_name, module in model.named_modules()}
    groups: Dict[str, List[torch.nn.Parameter]] = defaultdict(list)

    total_params = 0
    unknown_params = 0
    counts: Dict[str, int] = defaultdict(int)

    for param_name, param in model.named_parameters():
        total_params += param.numel()
        module_path = param_name.rsplit(".", 1)[0] if "." in param_name else "root"
        l_values = module_l_map.get(module_path, [])
        l_value = l_values[0] if l_values else _infer_l_from_name(param_name)
        if l_value is None:
            group_name = "irrep_unknown"
            unknown_params += param.numel()
        else:
            group_name = f"irrep_l{l_value}"
            if irreps_grouping == "by_l_and_block":
                group_name = f"{group_name}::{module_path}"
        groups[group_name].append(param)
        counts[group_name] += param.numel()

    global _EMITTED_IRREPS_SUMMARY
    if not _EMITTED_IRREPS_SUMMARY and total_params > 0:
        unknown_frac = unknown_params / total_params
        summary = {k: counts[k] for k in sorted(counts)}
        LOGGER.warning(
            "Irreps grouping summary: %s (unknown_fraction=%.3f)",
            summary,
            unknown_frac,
        )
        _EMITTED_IRREPS_SUMMARY = True

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
