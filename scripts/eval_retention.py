#!/usr/bin/env python
from __future__ import annotations

"""Evaluate retention on held-out legacy benchmark batches.

This is evaluation-only: adaptation-time data is not required here. The script compares
base and adapted checkpoints on the same held-out old-domain batches.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from mace_efggm.grouping import module_wise_groups


TARGET_KEYS = {
    "energy": "target_energy",
    "forces": "target_forces",
    "stress": "target_stress",
}


def load_batches(path: str):
    data = torch.load(path)
    if isinstance(data, dict) and "batches" in data:
        return data["batches"]
    return data


def _init_metric_state() -> Dict[str, float]:
    return {"ae_sum": 0.0, "se_sum": 0.0, "n": 0.0}


def _update_state(state: Dict[str, float], pred: torch.Tensor, target: torch.Tensor) -> None:
    diff = pred - target
    state["ae_sum"] += float(diff.abs().sum().item())
    state["se_sum"] += float((diff**2).sum().item())
    state["n"] += float(diff.numel())


def _finalize(state: Dict[str, float]) -> Dict[str, float] | None:
    if state["n"] <= 0:
        return None
    return {
        "mae": state["ae_sum"] / state["n"],
        "rmse": (state["se_sum"] / state["n"]) ** 0.5,
    }


def eval_metrics(model, batches, device) -> Dict[str, Any]:
    model.eval()
    states = {k: _init_metric_state() for k in TARGET_KEYS}
    with torch.no_grad():
        for batch in batches:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            out = model(batch, training=False, compute_force=True, compute_stress=True, compute_virials=False)
            for metric_name, target_key in TARGET_KEYS.items():
                if target_key not in batch or metric_name not in out:
                    continue
                _update_state(states[metric_name], out[metric_name], batch[target_key])

    finalized = {k: _finalize(v) for k, v in states.items()}
    return {
        "energy": finalized["energy"],
        "force": finalized["forces"],
        "stress": finalized["stress"],
    }


def parameter_drift(base_model, adapted_model) -> Dict[str, Any]:
    base_params = dict(base_model.named_parameters())
    adapted_params = dict(adapted_model.named_parameters())
    groups = module_wise_groups(adapted_model.named_parameters())

    total_sq = 0.0
    per_group = {}
    for group_name, names in groups.items():
        sq = 0.0
        for name in names:
            if name not in base_params or name not in adapted_params:
                continue
            diff = adapted_params[name].detach() - base_params[name].detach()
            sq += float((diff**2).sum().item())
        per_group[group_name] = sq**0.5
        total_sq += sq
    return {
        "overall_l2": total_sq**0.5,
        "per_group_l2": per_group,
    }


def with_deltas(base: Dict[str, Any], adapted: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ["energy", "force", "stress"]:
        base_entry = base.get(name)
        adapted_entry = adapted.get(name)
        if base_entry is None or adapted_entry is None:
            out[name] = {"base": base_entry, "adapted": adapted_entry, "delta": None}
            continue
        out[name] = {
            "base": base_entry,
            "adapted": adapted_entry,
            "delta": {
                "mae": adapted_entry["mae"] - base_entry["mae"],
                "rmse": adapted_entry["rmse"] - base_entry["rmse"],
            },
        }
    return out


def print_compact_table(report: Dict[str, Any]) -> None:
    print("metric      base_mae    base_rmse   adapted_mae adapted_rmse delta_mae   delta_rmse")
    for name in ["energy", "force", "stress"]:
        row = report["metrics"][name]
        if row["delta"] is None:
            print(f"{name:<10} {'null':>10} {'null':>11} {'null':>11} {'null':>12} {'null':>10} {'null':>12}")
            continue
        print(
            f"{name:<10} {row['base']['mae']:10.6f} {row['base']['rmse']:11.6f} "
            f"{row['adapted']['mae']:11.6f} {row['adapted']['rmse']:12.6f} "
            f"{row['delta']['mae']:10.6f} {row['delta']['rmse']:12.6f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapted_model", required=True)
    p.add_argument("--retention_batches", required=True, help=".pt file with held-out benchmark batches")
    p.add_argument("--out", default="retention.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    batches = load_batches(args.retention_batches)
    base = torch.load(args.base_model, map_location=device)
    adapted = torch.load(args.adapted_model, map_location=device)

    base_metrics = eval_metrics(base, batches, device)
    adapted_metrics = eval_metrics(adapted, batches, device)
    report = {
        "schema_version": "1.0",
        "metrics": with_deltas(base_metrics, adapted_metrics),
        "parameter_drift": parameter_drift(base, adapted),
    }
    print_compact_table(report)
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
