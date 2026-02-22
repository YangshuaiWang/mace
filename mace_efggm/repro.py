"""Reproducibility and compute-budget helpers shared by finetuning scripts."""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


@dataclass(frozen=True)
class BudgetConfig:
    optimizer: str
    lr: float
    weight_decay: float
    scheduler: str
    scheduler_params: Dict[str, Any]
    max_steps: int
    batch_size: int
    grad_accum_steps: int
    mixed_precision: bool
    early_stopping_patience: int
    early_stopping_metric: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_budget_config(cfg: Dict[str, Any]) -> BudgetConfig:
    budget = cfg.get("budget", {})
    max_steps = budget.get("max_steps")
    if max_steps is None:
        # Backward compatible path for older config files.
        max_steps = budget.get("steps") or budget.get("max_num_epochs")
    if max_steps is None:
        raise KeyError("budget.max_steps is required (or legacy steps/max_num_epochs)")

    early = budget.get("early_stopping", {})
    scheduler_params = budget.get("scheduler_params") or {}
    if not isinstance(scheduler_params, dict):
        raise TypeError("budget.scheduler_params must be a dict")

    return BudgetConfig(
        optimizer=str(budget.get("optimizer", "adam")),
        lr=float(budget.get("lr", 1e-3)),
        weight_decay=float(budget.get("weight_decay", 0.0)),
        scheduler=str(budget.get("scheduler", "none")),
        scheduler_params=scheduler_params,
        max_steps=int(max_steps),
        batch_size=int(budget.get("batch_size", 1)),
        grad_accum_steps=max(int(budget.get("grad_accum_steps", 1)), 1),
        mixed_precision=bool(budget.get("mixed_precision", False)),
        early_stopping_patience=int(early.get("patience", budget.get("patience", 0))),
        early_stopping_metric=str(early.get("metric", "valid_loss")),
    )


def make_run_dir(base_dir: str | Path, exp_name: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"{exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dump_config(config: Dict[str, Any], run_dir: Path) -> None:
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))


def write_git_commit(run_dir: Path) -> None:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        commit = "unknown"
    (run_dir / "git_commit.txt").write_text(f"{commit}\n")


def compute_budget(config: BudgetConfig, dataset_size: int, world_size: int = 1) -> Dict[str, Any]:
    effective_batch = config.batch_size * config.grad_accum_steps * max(world_size, 1)
    total_updates = int(config.max_steps)
    total_samples_seen = int(total_updates * effective_batch)
    return {
        "optimizer": config.optimizer,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "scheduler": config.scheduler,
        "scheduler_params": config.scheduler_params,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "mixed_precision": config.mixed_precision,
        "early_stopping": {
            "patience": config.early_stopping_patience,
            "metric": config.early_stopping_metric,
        },
        "dataset_size": int(dataset_size),
        "world_size": int(max(world_size, 1)),
        "effective_batch": int(effective_batch),
        "total_updates": total_updates,
        "total_samples_seen": total_samples_seen,
    }


def print_and_save_budget(budget_report: Dict[str, Any], run_dir: Path) -> None:
    print("=== Matched Compute Budget ===")
    print(json.dumps(budget_report, indent=2))
    (run_dir / "budget.json").write_text(json.dumps(budget_report, indent=2))


def budget_config_to_dict(cfg: BudgetConfig) -> Dict[str, Any]:
    return asdict(cfg)


def compare_budget(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Compare critical compute-budget knobs across two configs."""
    a = budget_config_to_dict(normalize_budget_config(cfg_a))
    b = budget_config_to_dict(normalize_budget_config(cfg_b))
    critical_keys = [
        "optimizer",
        "lr",
        "weight_decay",
        "scheduler",
        "scheduler_params",
        "max_steps",
        "batch_size",
        "grad_accum_steps",
        "mixed_precision",
        "early_stopping_patience",
        "early_stopping_metric",
    ]
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in critical_keys:
        if a.get(key) != b.get(key):
            mismatches[key] = {"a": a.get(key), "b": b.get(key)}
    return mismatches
