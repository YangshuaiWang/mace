#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def to_float(v):
    if v in (None, "", "None"):
        return None
    return float(v)


def to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def display_method(row: dict[str, Any], label_map: dict[str, str]) -> str:
    method = str(row.get("method", "method"))
    grouping = str(row.get("grouping_mode", "") or "")
    if "E-FGGM" in method and grouping:
        method = f"E-FGGM({grouping})"
    return label_map.get(method, method)


def style_for_grouping(grouping: str) -> tuple[str, str]:
    if grouping == "irreps":
        return "tab:orange", "^"
    if grouping == "module":
        return "tab:blue", "o"
    return "tab:gray", "s"


def _metric_and_error(row: dict[str, Any], base: str, aggregated: bool) -> tuple[float | None, float | None]:
    if aggregated:
        return to_float(row.get(f"{base}_mean")), to_float(row.get(f"{base}_ci95"))
    return to_float(row.get(base)), None


def fig_tradeoff(rows, out: Path, aggregated: bool, label_map: dict[str, str]):
    plt.figure(figsize=(6.4, 5.2))
    for row in rows:
        x, xerr = _metric_and_error(row, "new_force_rmse", aggregated)
        y, yerr = _metric_and_error(row, "old_delta_force_rmse", aggregated)
        if x is None or y is None:
            continue
        grouping = str(row.get("grouping_mode", "") or "")
        fallback = to_bool(row.get("fallback_used", False))
        color, marker = style_for_grouping(grouping)
        label = display_method(row, label_map)
        facecolors = "none" if fallback else color
        plt.scatter(x, y, edgecolors=color, facecolors=facecolors, marker=marker)
        if aggregated and (xerr is not None or yerr is not None):
            plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="none", ecolor=color, alpha=0.8, capsize=3)
        plt.text(x, y, label, fontsize=8)
    plt.xlabel("new_force_rmse")
    plt.ylabel("old_delta_force_rmse")
    plt.title("Adaptation-retention tradeoff")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def fig_md(rows, out: Path, aggregated: bool, label_map: dict[str, str]):
    points = []
    if aggregated:
        for row in rows:
            method = display_method(row, label_map)
            v = to_float(row.get("md_energy_drift_max_abs_mean"))
            err = to_float(row.get("md_energy_drift_max_abs_ci95"))
            if v is not None:
                points.append((method, v, err))
    else:
        grouped = defaultdict(list)
        for row in rows:
            method = display_method(row, label_map)
            v = to_float(row.get("md_energy_drift_max_abs"))
            if v is not None:
                grouped[method].append(v)
        for method in sorted(grouped):
            vals = grouped[method]
            points.append((method, sum(vals) / len(vals), None))

    methods = [p[0] for p in points]
    vals = [p[1] for p in points]
    errs = [p[2] for p in points]
    plt.figure(figsize=(7.2, 4.0))
    plt.bar(methods, vals, yerr=errs if aggregated else None, capsize=3 if aggregated else 0)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("md_energy_drift_max_abs")
    plt.title("MD energy drift by method")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def apply_paper_mode(enabled: bool) -> None:
    if not enabled:
        return
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Raw or aggregated CSV/JSON")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--paper_mode", action="store_true")
    p.add_argument("--label_map", default=None, help="Optional JSON file mapping display labels")
    p.add_argument("--with_drift_spectrum", action="store_true", help="Deprecated; kept for compatibility")
    p.add_argument("--with_irreps_spectrum", action="store_true", help="Deprecated; kept for compatibility")
    args = p.parse_args()

    rows = read_rows(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_mode(args.paper_mode)

    label_map = {}
    if args.label_map:
        label_map = json.loads(Path(args.label_map).read_text())

    aggregated = any("_mean" in k for k in (rows[0].keys() if rows else []))

    fig_tradeoff(rows, out_dir / "Fig_tradeoff.png", aggregated=aggregated, label_map=label_map)
    fig_md(rows, out_dir / "Fig_md.png", aggregated=aggregated, label_map=label_map)


if __name__ == "__main__":
    main()
