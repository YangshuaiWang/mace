#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


def parse_seeds(seed_arg: str) -> list[int]:
    return [int(x.strip()) for x in seed_arg.split(",") if x.strip()]


def runner_script(mode: str) -> str:
    if mode == "baseline":
        return "scripts/finetune_baselines.py"
    if mode == "efggm":
        return "scripts/finetune_efggm.py"
    raise ValueError(f"Unsupported mode: {mode}")


def run_one(config: Path, seed: int, run_dir_base: str, exp_name: str, mode: str, compare_budget_config: str | None) -> int:
    cfg: dict[str, Any] = yaml.safe_load(config.read_text())
    cfg["seed"] = seed
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(cfg, tmp)
        tmp_path = tmp.name

    cmd = [
        sys.executable,
        runner_script(mode),
        "--config",
        tmp_path,
        "--run_dir_base",
        run_dir_base,
        "--exp_name",
        exp_name,
    ]
    if compare_budget_config:
        cmd += ["--compare_budget_config", compare_budget_config]
    completed = subprocess.run(cmd, check=False)
    return completed.returncode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--run_dir_base", default="runs")
    p.add_argument("--exp_name_prefix", required=True)
    p.add_argument("--mode", choices=["baseline", "efggm"], required=True)
    p.add_argument("--compare_budget_config", default=None)
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    run_dirs = []
    for seed in seeds:
        exp_name = f"{args.exp_name_prefix}_seed{seed}"
        rc = run_one(
            config=Path(args.config),
            seed=seed,
            run_dir_base=args.run_dir_base,
            exp_name=exp_name,
            mode=args.mode,
            compare_budget_config=args.compare_budget_config,
        )
        if rc != 0:
            raise RuntimeError(f"Sweep failed for seed={seed} with exit code {rc}")
        run_dirs.append(str(Path(args.run_dir_base) / exp_name))

    sweep = {
        "config": args.config,
        "mode": args.mode,
        "seeds": seeds,
        "run_dir_base": args.run_dir_base,
        "exp_name_prefix": args.exp_name_prefix,
        "run_dirs": run_dirs,
    }
    out = Path(args.run_dir_base) / f"{args.exp_name_prefix}_sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sweep, indent=2))


if __name__ == "__main__":
    main()
