#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import warnings
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict

import yaml

T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.16,
    14: 2.145,
    15: 2.131,
    16: 2.12,
    17: 2.11,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.08,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.06,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def read_json(path: Path) -> Dict[str, Any] | list[Any] | None:
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


def maybe_warn_missing(path: Path, kind: str, run_dir: Path) -> None:
    if not path.exists():
        warnings.warn(f"{kind} missing for run {run_dir.name}: {path.name}", RuntimeWarning, stacklevel=2)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def aggregate_run(run_dir: Path) -> Dict[str, Any] | None:
    cfg_path = run_dir / "config.yaml"
    budget_path = run_dir / "budget.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not (cfg_path.exists() and budget_path.exists() and metrics_path.exists()):
        return None

    cfg = yaml.safe_load(cfg_path.read_text())
    budget = read_json(budget_path) or {}
    metrics = read_final_metrics(metrics_path)

    retention_path = run_dir / "retention.json"
    md_path = run_dir / "md_stability.json"
    diag_path = run_dir / "grouping_diagnostics.json"
    maybe_warn_missing(retention_path, "retention", run_dir)
    maybe_warn_missing(md_path, "md_stability", run_dir)

    retention = read_json(retention_path)
    md = read_json(md_path)
    grouping_diag = read_json(diag_path)
    drift_spectrum = read_json(run_dir / "drift_spectrum.json")
    fisher_spectrum = read_json(run_dir / "fisher_spectrum.json")
    drift_spectrum_irreps = read_json(run_dir / "drift_spectrum_irreps.json")

    method = cfg.get("method") or cfg.get("name") or cfg.get("exp_name") or run_dir.name
    stress_new = get_metric(metrics, ["new", "stress", "rmse"])
    if stress_new is None:
        stress_new = get_metric(metrics, ["new_stress_rmse"])

    grouping = get_metric(cfg, ["efggm", "grouping", "mode"]) or get_metric(cfg, ["efggm", "grouping"])
    if isinstance(grouping, dict):
        grouping = grouping.get("mode")

    row = {
        "exp_name": run_dir.name,
        "run_dir": str(run_dir),
        "method": method,
        "seed": cfg.get("seed"),
        "max_steps": budget.get("max_steps") or get_metric(cfg, ["budget", "max_steps"]),
        "effective_batch": budget.get("effective_batch"),
        "total_updates": budget.get("total_updates"),
        "optimizer": get_metric(cfg, ["budget", "optimizer"]),
        "lr": get_metric(cfg, ["budget", "lr"]),
        "weight_decay": get_metric(cfg, ["budget", "weight_decay"]),
        "new_energy_rmse": get_metric(metrics, ["new", "energy", "rmse"], get_metric(metrics, ["new_energy_rmse"])),
        "new_force_rmse": get_metric(metrics, ["new", "force", "rmse"], get_metric(metrics, ["new_force_rmse"])),
        "stress": stress_new,
        "old_delta_energy_rmse": get_metric(retention, ["metrics", "energy", "delta", "rmse"]),
        "old_delta_force_rmse": get_metric(retention, ["metrics", "force", "delta", "rmse"]),
        "md_failure_rate": get_metric(md, ["aggregate", "failure_count", "mean"], get_metric(md, ["failure_rate"])),
        "md_energy_drift_max_abs": get_metric(md, ["aggregate", "energy_drift_max_abs", "mean"], get_metric(md, ["energy_drift", "max_abs"])),
        "param_drift_l2": get_metric(retention, ["parameter_drift", "overall_l2"], get_metric(drift_spectrum, ["overall_l2"])),
        "grouping_mode": grouping,
        "irreps_unknown_fraction": get_metric(grouping_diag, ["unknown_fraction"]),
        "fallback_used": get_metric(grouping_diag, ["fallback_used"]),
        "irreps_grouping": get_metric(grouping_diag, ["irreps_grouping"]),
    }
    if isinstance(fisher_spectrum, list):
        row["fisher_spectrum"] = json.dumps(fisher_spectrum)
    if isinstance(drift_spectrum_irreps, list):
        row["drift_spectrum_irreps"] = json.dumps(drift_spectrum_irreps)
    return row


def print_table(rows: list[Dict[str, Any]]) -> None:
    if not rows:
        print("No runs found.")
        return
    cols = ["exp_name", "method", "new_force_rmse", "old_delta_force_rmse", "md_energy_drift_max_abs", "param_drift_l2"]
    print(" ".join(c.ljust(28) for c in cols))
    for row in rows:
        print(" ".join(str(row.get(c, "")).ljust(28) for c in cols))


def _t_multiplier_95(n: int) -> float:
    if n <= 1:
        return 0.0
    dof = n - 1
    if dof in T_CRIT_95:
        return T_CRIT_95[dof]
    # normal approx for large n
    return NormalDist().inv_cdf(0.975)


def compute_ci_stats(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    mean = sum(values) / n
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95": 0.0}
    var = sum((x - mean) ** 2 for x in values) / (n - 1)
    std = math.sqrt(var)
    ci95 = _t_multiplier_95(n) * std / math.sqrt(n)
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def group_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("method"),
        row.get("grouping_mode"),
        row.get("max_steps"),
        row.get("effective_batch"),
        row.get("total_updates"),
        row.get("optimizer"),
        row.get("lr"),
        row.get("weight_decay"),
        row.get("irreps_grouping"),
    )


def aggregate_across_seeds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric_cols = [
        "new_energy_rmse",
        "new_force_rmse",
        "stress",
        "old_delta_energy_rmse",
        "old_delta_force_rmse",
        "md_failure_rate",
        "md_energy_drift_max_abs",
        "param_drift_l2",
        "irreps_unknown_fraction",
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(group_key(row), []).append(row)

    out = []
    for key, grows in grouped.items():
        item = {
            "method": key[0],
            "grouping_mode": key[1],
            "max_steps": key[2],
            "effective_batch": key[3],
            "total_updates": key[4],
            "optimizer": key[5],
            "lr": key[6],
            "weight_decay": key[7],
            "irreps_grouping": key[8],
            "n": len(grows),
            "ci_method": "t-based 95% CI (normal approx for dof>30)",
            "seed_values": json.dumps(sorted([r.get("seed") for r in grows if r.get("seed") is not None])),
            "fallback_used": any(bool(r.get("fallback_used")) for r in grows),
        }
        for metric in metric_cols:
            vals = [_to_float(r.get(metric)) for r in grows]
            vals = [v for v in vals if v is not None]
            stats = compute_ci_stats(vals)
            item[f"{metric}_mean"] = stats["mean"]
            item[f"{metric}_std"] = stats["std"]
            item[f"{metric}_ci95"] = stats["ci95"]
            item[f"{metric}_n"] = stats["n"]
        out.append(item)
    return sorted(out, key=lambda r: (str(r.get("method")), str(r.get("grouping_mode"))))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_glob", default="runs/*")
    p.add_argument("--out_csv", default=None, help="Back-compat: same as --out_csv_raw")
    p.add_argument("--out_json", default="runs/summary.json")
    p.add_argument("--out_csv_raw", default=None)
    p.add_argument("--out_csv_agg", default=None)
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    rows = []
    for run in sorted(glob.glob(args.runs_glob)):
        item = aggregate_run(Path(run))
        if item is not None:
            rows.append(item)

    print_table(rows)
    aggregated = aggregate_across_seeds(rows)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_csv_raw = out_dir / "summary_raw.csv"
        out_csv_agg = out_dir / "summary_agg.csv"
    else:
        out_csv_raw = Path(args.out_csv_raw or args.out_csv or "runs/summary_raw.csv")
        out_csv_agg = Path(args.out_csv_agg or "runs/summary_agg.csv")

    write_csv(out_csv_raw, rows)
    write_csv(out_csv_agg, aggregated)


if __name__ == "__main__":
    main()
