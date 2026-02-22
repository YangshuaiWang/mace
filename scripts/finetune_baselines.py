#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from mace_efggm.repro import (
    compare_budget,
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


def make_optimizer(model: torch.nn.Module, budget_cfg):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if budget_cfg.optimizer.lower() == "adam":
        return torch.optim.Adam(trainable, lr=budget_cfg.lr, weight_decay=budget_cfg.weight_decay)
    raise ValueError(f"Unsupported optimizer in baseline step loop: {budget_cfg.optimizer}")


def make_scheduler(optimizer: torch.optim.Optimizer, budget_cfg):
    scheduler_name = budget_cfg.scheduler.lower()
    if scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler in baseline step loop: {budget_cfg.scheduler}")


def compute_task_loss(model: torch.nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    output = model(batch, training=True, compute_force=False, compute_stress=False, compute_virials=False)
    if "target_energy" not in batch:
        raise KeyError("Each batch must contain 'target_energy' tensor for training")
    return torch.nn.functional.mse_loss(output["energy"], batch["target_energy"])


def apply_variant_trainability(model: torch.nn.Module, variant: str) -> None:
    if variant in {"full_ft", "l2sp", "lora"}:
        for p in model.parameters():
            p.requires_grad = True
        if variant == "lora":
            logging.warning("LoRA baseline not implemented in step loop; falling back to full_ft semantics.")
        return

    if variant == "head_only":
        head_tokens = ("readout", "head")
        for name, p in model.named_parameters():
            p.requires_grad = any(token in name.lower() for token in head_tokens)
        num_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        if num_trainable == 0:
            raise ValueError("head_only variant found no trainable parameters; expected readout/head params.")
        return

    raise ValueError(f"Unknown baseline variant: {variant}")




def compute_l2sp_anchor_loss(model: torch.nn.Module, theta0: dict[str, torch.Tensor], l2sp_lambda: float) -> torch.Tensor:
    loss = torch.zeros((), device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if p.requires_grad:
            loss = loss + (p - theta0[name]).pow(2).sum()
    return l2sp_lambda * loss

def validation_loss(model: torch.nn.Module, valid_batches: list[dict], device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in valid_batches:
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            losses.append(float(compute_task_loss(model, batch).item()))
    model.train()
    if not losses:
        return float("nan")
    return float(sum(losses) / len(losses))


def run_variant(cfg: dict, variant: str, root_run_dir: Path, l2sp_lambda: float) -> None:
    variant_dir = root_run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.get("device", "cpu"))
    model = torch.load(cfg["model"]["foundation_checkpoint"], map_location=device)
    apply_variant_trainability(model, variant)
    model.train()
    model.to(device)

    theta0 = {
        name: p.detach().clone().to(device)
        for name, p in model.named_parameters()
        if p.requires_grad
    }

    train_batches = load_batches(cfg["data"]["train_batches_pt"])
    valid_batches = None
    if cfg.get("data", {}).get("valid_batches_pt"):
        valid_batches = load_batches(cfg["data"]["valid_batches_pt"])

    budget_cfg = normalize_budget_config(cfg)
    total_steps = int(budget_cfg.max_steps)
    grad_accum = budget_cfg.grad_accum_steps
    mixed_precision = budget_cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    optimizer = make_optimizer(model, budget_cfg)
    scheduler = make_scheduler(optimizer, budget_cfg)

    best_valid = float("inf")
    bad_steps = 0
    patience = int(budget_cfg.early_stopping_patience)

    metrics_path = variant_dir / "metrics.jsonl"
    step = 0
    optimizer_updates = 0
    while step < total_steps:
        for batch in train_batches:
            if step >= total_steps:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            if step % grad_accum == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                task_loss = compute_task_loss(model, batch)
                l2sp_loss = torch.zeros((), device=device)
                if variant == "l2sp":
                    l2sp_loss = compute_l2sp_anchor_loss(model, theta0, l2sp_lambda)
                total_loss = task_loss + l2sp_loss

            scaled = total_loss / grad_accum
            scaler.scale(scaled).backward()

            if (step + 1) % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer_updates += 1

            valid_loss = None
            if valid_batches is not None and ((step + 1) % grad_accum == 0):
                valid_loss = validation_loss(model, valid_batches, device)
                if valid_loss < best_valid:
                    best_valid = valid_loss
                    bad_steps = 0
                else:
                    bad_steps += 1
            elif valid_batches is None and patience == 0:
                valid_loss = None

            with metrics_path.open("a", encoding="utf-8") as f:
                payload = {
                    "step": step,
                    "optimizer_updates": optimizer_updates,
                    "task_loss": float(task_loss.item()),
                    "l2sp_loss": float(l2sp_loss.item()),
                    "total_loss": float(total_loss.item()),
                }
                if valid_loss is not None:
                    payload["valid_loss"] = float(valid_loss)
                f.write(json.dumps(payload) + "\n")

            step += 1
            if valid_batches is not None and patience > 0 and bad_steps > patience:
                logging.info("Early stopping %s at step=%s due to patience=%s", variant, step, patience)
                break

    torch.save(model, variant_dir / "final_model.pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir_base", default="runs/")
    parser.add_argument("--exp_name", default="baseline_finetune")
    parser.add_argument("--compare_budget_config", default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.compare_budget_config:
        other_cfg = yaml.safe_load(Path(args.compare_budget_config).read_text())
        mismatches = compare_budget(cfg, other_cfg)
        if mismatches:
            raise ValueError(f"Compute budget mismatch: {json.dumps(mismatches, indent=2)}")
    logging.basicConfig(level=logging.INFO)
    set_seed(int(cfg.get("seed", 123)))

    if "train_batches_pt" not in cfg.get("data", {}):
        raise KeyError("data.train_batches_pt is required for step-based baseline finetuning")
    if "foundation_checkpoint" not in cfg.get("model", {}):
        raise KeyError("model.foundation_checkpoint is required for step-based baseline finetuning")

    run_dir = make_run_dir(args.run_dir_base, args.exp_name)
    dump_config(cfg, run_dir)
    write_git_commit(run_dir)

    train_batches = load_batches(cfg["data"]["train_batches_pt"])
    budget_cfg = normalize_budget_config(cfg)
    budget_report = compute_budget(
        budget_cfg,
        dataset_size=infer_dataset_size(cfg, train_batches),
        world_size=int(cfg.get("world_size", 1)),
    )
    print_and_save_budget(budget_report, run_dir)

    variants = cfg.get("baselines", ["full_ft", "head_only", "l2sp"])
    for variant in variants:
        run_variant(cfg, variant, run_dir, float(cfg.get("l2sp_lambda", 1.0)))


if __name__ == "__main__":
    main()
