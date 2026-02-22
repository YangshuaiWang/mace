#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def read_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_final_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return {}
    return json.loads(lines[-1])


def get_metric(blob: Dict[str, Any] | None, keys: list[str], default=None):
    cur = blob
    for key in keys:
        if cur is None or not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def aggregate_run(run_dir: Path) -> Dict[str, Any] | None:
    cfg_path = run_dir / "config.yaml"
    budget_path = run_dir / "budget.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not (cfg_path.exists() and budget_path.exists() and metrics_path.exists()):
        return None

    cfg = yaml.safe_load(cfg_path.read_text())
    budget = read_json(budget_path) or {}
    metrics = read_final_metrics(metrics_path)
    retention = read_json(run_dir / "retention.json")
    md = read_json(run_dir / "md_stability.json")
    drift_spectrum = read_json(run_dir / "drift_spectrum.json")

    method = cfg.get("method") or cfg.get("name") or cfg.get("exp_name") or run_dir.name
    stress_new = get_metric(metrics, ["new", "stress", "rmse"])
    if stress_new is None:
        stress_new = get_metric(metrics, ["new_stress_rmse"])

    return {
        "exp_name": run_dir.name,
        "method": method,
        "seed": cfg.get("seed"),
        "max_steps": budget.get("max_steps") or get_metric(cfg, ["budget", "max_steps"]),
        "effective_batch": budget.get("effective_batch"),
        "total_updates": budget.get("total_updates"),
        "new_energy_rmse": get_metric(metrics, ["new", "energy", "rmse"], get_metric(metrics, ["new_energy_rmse"])),
        "new_force_rmse": get_metric(metrics, ["new", "force", "rmse"], get_metric(metrics, ["new_force_rmse"])),
        "stress": stress_new,
        "old_delta_energy_rmse": get_metric(retention, ["metrics", "energy", "delta", "rmse"]),
        "old_delta_force_rmse": get_metric(retention, ["metrics", "force", "delta", "rmse"]),
        "md_failure_rate": get_metric(md, ["aggregate", "failure_count", "mean"], get_metric(md, ["failure_rate"])),
        "md_energy_drift_max_abs": get_metric(md, ["aggregate", "energy_drift_max_abs", "mean"], get_metric(md, ["energy_drift", "max_abs"])),
        "param_drift_l2": get_metric(retention, ["parameter_drift", "overall_l2"], get_metric(drift_spectrum, ["overall_l2"])),
    }


def print_table(rows: list[Dict[str, Any]]) -> None:
    if not rows:
        print("No runs found.")
        return
    cols = ["exp_name", "method", "new_force_rmse", "old_delta_force_rmse", "md_energy_drift_max_abs", "param_drift_l2"]
    print(" ".join(c.ljust(28) for c in cols))
    for row in rows:
        print(" ".join(str(row.get(c, "")).ljust(28) for c in cols))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_glob", default="runs/*")
    p.add_argument("--out_csv", default="runs/summary.csv")
    p.add_argument("--out_json", default="runs/summary.json")
    args = p.parse_args()

    rows = []
    for run in sorted(glob.glob(args.runs_glob)):
        item = aggregate_run(Path(run))
        if item is not None:
            rows.append(item)

    print_table(rows)
    if rows:
        fieldnames = list(rows[0].keys())
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        out_json = Path(args.out_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
