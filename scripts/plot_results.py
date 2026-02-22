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


def fig_tradeoff(rows, out: Path):
    plt.figure(figsize=(6, 5))
    for row in rows:
        x = to_float(row.get("new_force_rmse"))
        y = to_float(row.get("old_delta_force_rmse"))
        if x is None or y is None:
            continue
        label = str(row.get("method", "method"))
        plt.scatter(x, y)
        plt.text(x, y, label, fontsize=8)
    plt.xlabel("new_force_rmse")
    plt.ylabel("old_delta_force_rmse")
    plt.title("Adaptation-retention tradeoff")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_md(rows, out: Path):
    grouped = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "method"))
        v = to_float(row.get("md_energy_drift_max_abs"))
        if v is not None:
            grouped[method].append(v)
    methods = sorted(grouped)
    vals = [sum(grouped[m]) / len(grouped[m]) for m in methods]
    plt.figure(figsize=(7, 4))
    plt.bar(methods, vals)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("md_energy_drift_max_abs")
    plt.title("MD energy drift by method")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def fig_drift_spectrum(rows, out: Path):
    grouped = defaultdict(list)
    for row in rows:
        method = str(row.get("method", "method"))
        v = to_float(row.get("param_drift_l2"))
        if v is not None:
            grouped[method].append(v)
    if not grouped:
        return
    methods = sorted(grouped)
    vals = [sum(grouped[m]) / len(grouped[m]) for m in methods]
    plt.figure(figsize=(7, 4))
    plt.bar(methods, vals)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("param_drift_l2")
    plt.title("Parameter drift spectrum (method-level)")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Aggregated CSV or JSON")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--with_drift_spectrum", action="store_true")
    args = p.parse_args()

    rows = read_rows(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_tradeoff(rows, out_dir / "Fig_tradeoff.png")
    fig_md(rows, out_dir / "Fig_md.png")
    if args.with_drift_spectrum:
        fig_drift_spectrum(rows, out_dir / "Fig_drift_spectrum.png")


if __name__ == "__main__":
    main()
