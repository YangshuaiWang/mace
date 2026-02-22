#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from mace_efggm import (
    AlphaSchedule,
    FisherEMA,
    GradientMasker,
    MaskedOptimizerWrapper,
    build_parameter_mask,
    group_scores,
    irreps_wise_groups,
    module_wise_groups,
    select_top_groups,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_batches(path: str):
    data = torch.load(path)
    if isinstance(data, dict) and "batches" in data:
        return data["batches"]
    return data


def train(cfg: dict) -> None:
    set_seed(int(cfg.get("seed", 123)))
    device = torch.device(cfg.get("device", "cpu"))
    model = torch.load(cfg["model"]["foundation_checkpoint"], map_location=device)
    model.train()
    model.to(device)

    batches = load_batches(cfg["data"]["train_batches_pt"])
    fisher_steps = int(cfg["efggm"]["fisher_steps"])
    total_steps = int(cfg["budget"]["steps"])

    fisher = FisherEMA(model.named_parameters())
    group_mode = cfg["efggm"].get("grouping", "module")
    groups = module_wise_groups(model.named_parameters()) if group_mode == "module" else irreps_wise_groups(model.named_parameters())

    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["budget"]["lr"]))
    masker = GradientMasker(model.named_parameters())
    optw = MaskedOptimizerWrapper(opt, masker)
    sched = AlphaSchedule(
        alpha_start=float(cfg["efggm"]["alpha_start"]),
        alpha_end=float(cfg["efggm"]["alpha_end"]),
        total_steps=total_steps,
    )

    logs = []
    step = 0
    while step < total_steps:
        for batch in batches:
            if step >= total_steps:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            optw.zero_grad(set_to_none=True)
            output = model(batch, training=True, compute_force=False, compute_stress=False, compute_virials=False)
            if "target_energy" not in batch:
                raise KeyError("Each batch must contain 'target_energy' tensor for training")
            loss = torch.nn.functional.mse_loss(output["energy"], batch["target_energy"])
            loss.backward()

            fisher.update_from_model(model.named_parameters())
            if step >= fisher_steps:
                scores = group_scores(fisher.tensors(), groups)
                alpha = sched.value(step)
                kept = select_top_groups(scores, alpha)
                masker.update_mask(build_parameter_mask(groups, kept))

            optw.step()
            logs.append({"step": step, "loss": float(loss.item())})
            step += 1

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model, out / "efggm_finetuned.model")
    (out / "metrics.json").write_text(json.dumps(logs, indent=2))
    (out / "config_dump.json").write_text(json.dumps(cfg, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = yaml.safe_load(Path(args.config).read_text())
    train(cfg)


if __name__ == "__main__":
    main()
