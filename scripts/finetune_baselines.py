#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Dict

import yaml

from mace_efggm.repro import (
    compute_budget,
    dump_config,
    make_run_dir,
    normalize_budget_config,
    print_and_save_budget,
    set_seed,
    write_git_commit,
)


def infer_dataset_size(cfg: Dict[str, Any]) -> int:
    if "dataset_size" in cfg.get("data", {}):
        return int(cfg["data"]["dataset_size"])
    # TODO: wire this to native MACE dataset loader for exact sample counts.
    return 0


def build_shared_args(cfg: dict, run_dir: Path) -> list[str]:
    budget_cfg = normalize_budget_config(cfg)
    data = cfg["data"]
    model = cfg["model"]
    args = [
        "--name", cfg["name"],
        "--seed", str(cfg.get("seed", 123)),
        "--train_file", data["train_file"],
        "--valid_file", data["valid_file"],
        "--foundation_model", model["foundation_model"],
        "--batch_size", str(budget_cfg.batch_size),
        "--valid_batch_size", str(cfg.get("budget", {}).get("valid_batch_size", budget_cfg.batch_size)),
        "--optimizer", budget_cfg.optimizer,
        "--lr", str(budget_cfg.lr),
        "--scheduler", budget_cfg.scheduler,
        "--weight_decay", str(budget_cfg.weight_decay),
        "--max_num_epochs", str(budget_cfg.max_steps),
        "--patience", str(budget_cfg.early_stopping_patience),
        "--error_table", "PerAtomRMSE",
        "--results_dir", str(run_dir),
    ]
    return args


def persist_placeholder_metrics(run_dir: Path, payload: Dict[str, Any]) -> None:
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def run_variant(base_cmd: list[str], variant: str, run_dir: Path, l2sp_lambda: float) -> None:
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["mace_run_train", *base_cmd, "--name", f"{variant}", "--results_dir", str(variant_dir)]
    if variant == "head_only":
        cmd += ["--foundation_model_readout"]
    elif variant == "l2sp":
        cmd += ["--lr", str(1e-4), "--weight_decay", "0.0"]
        logging.info(
            "Using L2-SP proxy with low LR; exact anchor penalty not native in this fork (lambda=%s)",
            l2sp_lambda,
        )
    elif variant == "lora":
        logging.warning(
            "LoRA baseline requested, but no native LoRA path in this fork; running full FT budget-matched."
        )

    persist_placeholder_metrics(variant_dir, {"event": "run_start", "variant": variant})
    logging.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    model_candidates = sorted(variant_dir.rglob("*.model")) + sorted(variant_dir.rglob("*.pt"))
    if model_candidates:
        (variant_dir / "final_model.pt").write_bytes(model_candidates[-1].read_bytes())
    persist_placeholder_metrics(variant_dir, {"event": "run_end", "variant": variant})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir_base", default="runs/")
    parser.add_argument("--exp_name", default="baseline_finetune")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    logging.basicConfig(level=logging.INFO)
    set_seed(int(cfg.get("seed", 123)))

    run_dir = make_run_dir(args.run_dir_base, args.exp_name)
    dump_config(cfg, run_dir)
    write_git_commit(run_dir)

    budget_cfg = normalize_budget_config(cfg)
    budget_report = compute_budget(
        budget_cfg,
        dataset_size=infer_dataset_size(cfg),
        world_size=int(cfg.get("world_size", 1)),
    )
    print_and_save_budget(budget_report, run_dir)

    shared = build_shared_args(cfg, run_dir)
    variants = cfg.get("baselines", ["full_ft", "head_only", "l2sp"])
    for variant in variants:
        run_variant(shared, variant, run_dir, float(cfg.get("l2sp_lambda", 1.0)))


if __name__ == "__main__":
    main()
