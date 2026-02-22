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


def display_method(row: dict[str, Any]) -> str:
    method = str(row.get("method", "method"))
    grouping = str(row.get("grouping_mode", "") or "")
    if "E-FGGM" in method and grouping:
        return f"E-FGGM({grouping})"
    return method


def style_for_grouping(grouping: str) -> tuple[str, str]:
    if grouping == "irreps":
        return "tab:orange", "^"
    if grouping == "module":
        return "tab:blue", "o"
    return "tab:gray", "s"


def fig_tradeoff(rows, out: Path):
    plt.figure(figsize=(6, 5))
    for row in rows:
        x = to_float(row.get("new_force_rmse"))
        y = to_float(row.get("old_delta_force_rmse"))
        if x is None or y is None:
            continue
        grouping = str(row.get("grouping_mode", "") or "")
        color, marker = style_for_grouping(grouping)
        label = display_method(row)
        plt.scatter(x, y, c=color, marker=marker)
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
        method = display_method(row)
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
        method = display_method(row)
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


def _parse_spectrum(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return []


def fig_irreps_spectrum(rows, out_dir: Path):
    fisher_by_l: dict[int, float] = defaultdict(float)
    drift_by_l: dict[int, float] = defaultdict(float)
    for row in rows:
        for item in _parse_spectrum(row.get("fisher_spectrum")):
            fisher_by_l[int(item["l"])] += float(item.get("fisher_sum", 0.0))
        for item in _parse_spectrum(row.get("drift_spectrum_irreps")):
            drift_by_l[int(item["l"])] += float(item.get("drift_l2_sum", 0.0))

    if fisher_by_l:
        ls = sorted(fisher_by_l)
        plt.figure(figsize=(6, 4))
        plt.bar([str(l) for l in ls], [fisher_by_l[l] for l in ls])
        plt.xlabel("l")
        plt.ylabel("fisher_sum")
        plt.title("Irreps Fisher spectrum")
        plt.tight_layout()
        plt.savefig(out_dir / "Fig_irreps_spectrum_fisher.png", dpi=150)
        plt.close()

    if drift_by_l:
        ls = sorted(drift_by_l)
        plt.figure(figsize=(6, 4))
        plt.bar([str(l) for l in ls], [drift_by_l[l] for l in ls])
        plt.xlabel("l")
        plt.ylabel("drift_l2_sum")
        plt.title("Irreps drift spectrum")
        plt.tight_layout()
        plt.savefig(out_dir / "Fig_irreps_spectrum_drift.png", dpi=150)
        plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Aggregated CSV or JSON")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--with_drift_spectrum", action="store_true")
    p.add_argument("--with_irreps_spectrum", action="store_true")
    args = p.parse_args()

    rows = read_rows(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_tradeoff(rows, out_dir / "Fig_tradeoff.png")
    fig_md(rows, out_dir / "Fig_md.png")
    if args.with_drift_spectrum:
        fig_drift_spectrum(rows, out_dir / "Fig_drift_spectrum.png")
    if args.with_irreps_spectrum:
        fig_irreps_spectrum(rows, out_dir)


if __name__ == "__main__":
    main()
