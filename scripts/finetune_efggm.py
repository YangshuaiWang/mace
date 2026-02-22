#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

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
from mace_efggm.repro import (
    compute_budget,
    dump_config,
    make_run_dir,
    normalize_budget_config,
    print_and_save_budget,
    set_seed,
    write_git_commit,
)


def load_batches(path: str):
    data = torch.load(path)
    if isinstance(data, dict) and "batches" in data:
        return data["batches"]
    return data


def infer_dataset_size(cfg: Dict[str, Any], batches: list[dict]) -> int:
    if "dataset_size" in cfg.get("data", {}):
        return int(cfg["data"]["dataset_size"])
    return len(batches)


def train(cfg: dict, run_dir: Path) -> None:
    set_seed(int(cfg.get("seed", 123)))
    device = torch.device(cfg.get("device", "cpu"))
    model = torch.load(cfg["model"]["foundation_checkpoint"], map_location=device)
    model.train()
    model.to(device)

    batches = load_batches(cfg["data"]["train_batches_pt"])
    budget_cfg = normalize_budget_config(cfg)
    total_steps = int(budget_cfg.max_steps)
    fisher_steps = int(cfg["efggm"].get("fisher_steps", 0))

    budget_report = compute_budget(
        budget_cfg,
        dataset_size=infer_dataset_size(cfg, batches),
        world_size=int(cfg.get("world_size", 1)),
    )
    print_and_save_budget(budget_report, run_dir)

    fisher = FisherEMA(model.named_parameters())
    group_mode = cfg["efggm"].get("grouping", "module")
    groups = module_wise_groups(model.named_parameters()) if group_mode == "module" else irreps_wise_groups(model.named_parameters())

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(budget_cfg.lr),
        weight_decay=float(budget_cfg.weight_decay),
    )
    masker = GradientMasker(model.named_parameters())
    optw = MaskedOptimizerWrapper(opt, masker)
    sched = AlphaSchedule(
        alpha_start=float(cfg["efggm"]["alpha_start"]),
        alpha_end=float(cfg["efggm"]["alpha_end"]),
        total_steps=total_steps,
    )

    mixed_precision = budget_cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    grad_accum = budget_cfg.grad_accum_steps

    metrics_path = run_dir / "metrics.jsonl"
    current_scores: Dict[str, float] = {}
    current_mask: Dict[str, float] = {name: 1.0 for name, _ in model.named_parameters()}

    step = 0
    optimizer_updates = 0
    while step < total_steps:
        for batch in batches:
            if step >= total_steps:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            if step % grad_accum == 0:
                optw.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                output = model(batch, training=True, compute_force=False, compute_stress=False, compute_virials=False)
                if "target_energy" not in batch:
                    raise KeyError("Each batch must contain 'target_energy' tensor for training")
                loss = torch.nn.functional.mse_loss(output["energy"], batch["target_energy"])

            scaled = loss / grad_accum
            scaler.scale(scaled).backward()

            fisher.update_from_model(model.named_parameters())
            if step >= fisher_steps:
                current_scores = group_scores(fisher.tensors(), groups)
                alpha = sched.value(step)
                kept = select_top_groups(current_scores, alpha)
                current_mask = build_parameter_mask(groups, kept)
                masker.update_mask(current_mask)

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optw.optimizer)
                masker.apply()
                scaler.step(optw.optimizer)
                scaler.update()
                optimizer_updates += 1

            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "loss": float(loss.item()), "optimizer_updates": optimizer_updates}) + "\n")
            step += 1

    torch.save(model, run_dir / "final_model.pt")
    (run_dir / "mask.json").write_text(json.dumps(current_mask, indent=2))
    (run_dir / "fisher_group_scores.json").write_text(json.dumps(current_scores, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir_base", default="runs/")
    parser.add_argument("--exp_name", default="efggm_finetune")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = yaml.safe_load(Path(args.config).read_text())
    run_dir = make_run_dir(args.run_dir_base, args.exp_name)
    dump_config(cfg, run_dir)
    write_git_commit(run_dir)
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
