#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(v: Any) -> float | None:
    if v in (None, "", "None"):
        return None
    return float(v)


def fmt_pm(mean: Any, spread: Any, digits: int = 4) -> str:
    m = to_float(mean)
    s = to_float(spread)
    if m is None:
        return "--"
    if s is None:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f} $\\pm$ {s:.{digits}f}"


def canonical_method(row: dict[str, Any]) -> str:
    method = str(row.get("method", ""))
    grouping = str(row.get("grouping_mode", "") or "")
    low = method.lower()
    if "full" in low:
        return "Full FT"
    if "head" in low:
        return "Head-only"
    if "l2" in low:
        return "L2-SP"
    if "lora" in low:
        return "LoRA"
    if "e-fggm" in method or "efggm" in low:
        if grouping:
            return f"E-FGGM({grouping})"
        return "E-FGGM"
    return method


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--spread", choices=["ci95", "std"], default="ci95")
    args = p.parse_args()

    rows = read_rows(Path(args.input))
    order = ["Full FT", "Head-only", "L2-SP", "LoRA", "E-FGGM(module)", "E-FGGM(irreps)"]

    indexed = {canonical_method(r): r for r in rows}

    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Method & New-domain Force RMSE & Old-domain $\\Delta$Force RMSE & MD drift max\\_abs & Failure rate \\\\",
        "\\midrule",
    ]
    suffix = args.spread
    for name in order:
        row = indexed.get(name)
        if row is None:
            continue
        lines.append(
            "{} & {} & {} & {} & {} ".format(
                name,
                fmt_pm(row.get("new_force_rmse_mean"), row.get(f"new_force_rmse_{suffix}")),
                fmt_pm(row.get("old_delta_force_rmse_mean"), row.get(f"old_delta_force_rmse_{suffix}")),
                fmt_pm(row.get("md_energy_drift_max_abs_mean"), row.get(f"md_energy_drift_max_abs_{suffix}")),
                fmt_pm(row.get("md_failure_rate_mean"), row.get(f"md_failure_rate_{suffix}")),
            ) + r"\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
