#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
from pathlib import Path

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_shared_args(cfg: dict) -> list[str]:
    budget = cfg["budget"]
    data = cfg["data"]
    model = cfg["model"]
    return [
        "--name", cfg["name"],
        "--seed", str(cfg.get("seed", 123)),
        "--train_file", data["train_file"],
        "--valid_file", data["valid_file"],
        "--foundation_model", model["foundation_model"],
        "--max_num_epochs", str(budget["max_num_epochs"]),
        "--batch_size", str(budget["batch_size"]),
        "--valid_batch_size", str(budget["valid_batch_size"]),
        "--optimizer", budget["optimizer"],
        "--lr", str(budget["lr"]),
        "--scheduler", budget["scheduler"],
        "--patience", str(budget["patience"]),
    ]


def run_variant(base_cmd: list[str], variant: str, out_dir: Path, l2sp_lambda: float) -> None:
    cmd = ["mace_run_train", *base_cmd, "--results_dir", str(out_dir / variant)]
    if variant == "head_only":
        cmd += ["--foundation_model_readout"]
    elif variant == "l2sp":
        # Minimal proxy for L2-SP: very small LR + zero decay to stay near foundation params.
        cmd += ["--lr", str(1e-4), "--weight_decay", "0.0"]
        logging.info("Using L2-SP proxy with low LR; exact anchor penalty not native in this fork (lambda=%s)", l2sp_lambda)
    elif variant == "lora":
        logging.warning("LoRA baseline requested, but no native LoRA path in this fork; running full FT budget-matched.")
    logging.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out_dir", default="runs/baselines")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)
    set_seed(int(cfg.get("seed", 123)))
    (out_dir / "config_dump.json").write_text(json.dumps(cfg, indent=2))

    shared = build_shared_args(cfg)
    for variant in cfg.get("baselines", ["full_ft", "head_only", "l2sp"]):
        run_variant(shared, variant, out_dir, float(cfg.get("l2sp_lambda", 1.0)))


if __name__ == "__main__":
    main()
