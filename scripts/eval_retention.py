#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def eval_energy_mse(model, batches, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in batches:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            pred = model(batch, training=False, compute_force=False, compute_stress=False, compute_virials=False)["energy"]
            total += torch.nn.functional.mse_loss(pred, batch["target_energy"], reduction="sum").item()
            n += pred.numel()
    return total / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapted_model", required=True)
    p.add_argument("--retention_batches", required=True, help=".pt file with held-out benchmark batches")
    p.add_argument("--out", default="retention.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    batches = torch.load(args.retention_batches)
    base = torch.load(args.base_model, map_location=device)
    adapted = torch.load(args.adapted_model, map_location=device)

    base_mse = eval_energy_mse(base, batches, device)
    adapted_mse = eval_energy_mse(adapted, batches, device)
    report = {
        "base_mse": base_mse,
        "adapted_mse": adapted_mse,
        "retention_delta": adapted_mse - base_mse,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
