"""Evaluate IB-UQ gate_score as an OOD detector on ID/OOD xyz(extxyz) datasets.

This script reuses MACE's native xyz/extxyz ingestion path (`data.load_from_xyz` +
`AtomicData.from_config`) to stay compatible with training-time data semantics,
including the default REF_* keys.

Notes
-----
* `force_rmse`/`force_mae` and selective risk require force labels in the input xyz.
  If forces are missing, the script fails with a clear error.
* For IB-UQ stochastic inference, `--uq_samples > 1` reports per-structure
  `gate_score_mean` and `gate_score_std`; the mean is used for AUROC/AUPRC and
  risk-coverage metrics.
* `--deterministic` toggles model deterministic IB-UQ sampling
  (`ib_uq_deterministic=True`, `ib_uq_deterministic_zero_z0=True`) so repeated
  samples should match exactly (std ~ 0).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

# PyTorch >=2.6 defaults torch.load(..., weights_only=True), while e3nn constants
# may include trusted Python objects such as `slice`.
torch.serialization.add_safe_globals([slice])

try:
    from mace import data
    from mace.tools import torch_geometric, torch_tools, utils
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from mace import data
    from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Evaluate IB-UQ gate_score OOD detection on ID/OOD xyz datasets.",
    )
    parser.add_argument("--model_path", type=str, required=True, help="Trained MACE/IB-UQ-MACE checkpoint")
    parser.add_argument("--id_xyz", type=str, required=True, help="ID xyz/extxyz path")
    parser.add_argument("--ood_xyz", type=str, required=True, help="OOD xyz/extxyz path")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic IB-UQ inference (fixed/zero z0 via model flags)",
    )
    parser.add_argument("--uq_samples", type=int, default=1, help="Stochastic forward samples per structure")
    parser.add_argument("--out_dir", type=str, default="results/ood_eval")
    parser.add_argument(
        "--coverage_points",
        type=str,
        default="1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1",
        help="Comma-separated coverage values in (0,1]",
    )
    parser.add_argument("--smoke", action="store_true", help="Smoke mode: evaluate only a small number of samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on samples per split")
    return parser.parse_args()


def parse_coverage_points(raw: str) -> List[float]:
    vals = [float(v.strip()) for v in raw.split(",") if v.strip()]
    for v in vals:
        if v <= 0.0 or v > 1.0:
            raise ValueError(f"coverage point must be in (0,1], got {v}")
    return vals


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _group_from_ptr(values: np.ndarray, ptr: np.ndarray) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    for i in range(len(ptr) - 1):
        groups.append(values[ptr[i] : ptr[i + 1]])
    return groups


def _as_structure_gate(gate: torch.Tensor, batch_obj) -> np.ndarray:
    gate_np = _to_numpy(gate).reshape(-1)
    n_graphs = int(batch_obj.num_graphs)
    if gate_np.shape[0] == n_graphs:
        return gate_np

    # Fallback: per-atom gate -> unbiased per-structure mean via ptr segmentation.
    if gate_np.shape[0] == int(batch_obj.batch.shape[0]):
        ptr = _to_numpy(batch_obj.ptr)
        return np.array([gate_np[ptr[i] : ptr[i + 1]].mean() for i in range(n_graphs)], dtype=np.float64)

    raise ValueError(
        f"Unsupported gate_score shape {tuple(gate.shape)} for batch with {n_graphs} graphs and {int(batch_obj.batch.shape[0])} atoms."
    )


def _force_errors_per_structure(pred_forces: torch.Tensor, true_forces: torch.Tensor, ptr: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    pred = _to_numpy(pred_forces)
    true = _to_numpy(true_forces)
    ptr_np = _to_numpy(ptr)
    rmses: List[float] = []
    maes: List[float] = []
    for p, t in zip(_group_from_ptr(pred, ptr_np), _group_from_ptr(true, ptr_np)):
        diff = p - t
        rmses.append(float(np.sqrt(np.mean(diff**2))))
        maes.append(float(np.mean(np.abs(diff))))
    return np.array(rmses, dtype=np.float64), np.array(maes, dtype=np.float64)


def _manual_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    tpr = np.concatenate([[0.0], tps / pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / neg, [1.0]])
    return float(np.trapz(tpr, fpr))


def _manual_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    pos = int(y_true.sum())
    if pos == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y = y_true[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    recall = tps / pos
    precision = tps / np.maximum(tps + fps, 1)

    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def compute_ood_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score

        return {
            "auroc": float(roc_auc_score(y_true, y_score)),
            "auprc": float(average_precision_score(y_true, y_score)),
        }
    except Exception:
        return {
            "auroc": _manual_auroc(y_true, y_score),
            "auprc": _manual_auprc(y_true, y_score),
        }


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def selective_risk(
    gate_scores: np.ndarray,
    force_rmse: np.ndarray,
    force_mae: np.ndarray,
    coverage_points: Sequence[float],
) -> Tuple[List[float], List[float]]:
    # Higher gate_score => more uncertain => removed first.
    order_desc = np.argsort(-gate_scores)
    n = len(gate_scores)
    curve_rmse: List[float] = []
    curve_mae: List[float] = []

    for c in coverage_points:
        k_keep = max(1, int(math.floor(c * n)))
        n_reject = n - k_keep
        keep_idx = order_desc[n_reject:]
        curve_rmse.append(float(np.mean(force_rmse[keep_idx])))
        curve_mae.append(float(np.mean(force_mae[keep_idx])))

    return curve_rmse, curve_mae


def load_xyz_as_atomic_data(xyz_path: str, model: torch.nn.Module) -> List[data.AtomicData]:
    key_spec = data.KeySpecification.from_defaults()
    _, configs = data.load_from_xyz(
        file_path=xyz_path,
        key_specification=key_spec,
        extract_atomic_energies=False,
        keep_isolated_atoms=True,
    )

    heads = getattr(model, "heads", None)
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    cutoff = float(model.r_max)
    return [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff, heads=heads)
        for config in configs
    ]


def main() -> None:
    args = parse_args()
    if args.uq_samples < 1:
        raise ValueError("--uq_samples must be >= 1")

    coverage_points = parse_coverage_points(args.coverage_points)
    max_samples = args.max_samples
    if args.smoke and max_samples is None:
        max_samples = 20

    device = torch_tools.init_device(args.device)
    model = torch.load(f=args.model_path, map_location=args.device)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    if args.deterministic:
        setattr(model, "ib_uq_deterministic", True)
        setattr(model, "ib_uq_deterministic_zero_z0", True)

    id_dataset = load_xyz_as_atomic_data(args.id_xyz, model)
    ood_dataset = load_xyz_as_atomic_data(args.ood_xyz, model)

    if max_samples is not None:
        id_dataset = id_dataset[:max_samples]
        ood_dataset = ood_dataset[:max_samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    summary_path = out_dir / "summary.json"

    use_uq_stats = args.uq_samples > 1
    fieldnames = ["sample_id", "split", "n_atoms"]
    if use_uq_stats:
        fieldnames += ["gate_score_mean", "gate_score_std"]
    else:
        fieldnames += ["gate_score"]
    fieldnames += ["force_rmse", "force_mae"]

    all_gate: List[float] = []
    all_rmse: List[float] = []
    all_mae: List[float] = []
    all_split: List[int] = []

    split_datasets = [("id", 0, id_dataset), ("ood", 1, ood_dataset)]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        running_id = 0
        for split_name, split_label, dataset in split_datasets:
            loader = torch_geometric.dataloader.DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )

            for batch in loader:
                batch = batch.to(device)
                batch_dict = batch.to_dict()

                if "forces" not in batch_dict or batch_dict["forces"] is None:
                    raise ValueError(
                        "Input xyz must contain forces labels (default key: REF_forces) "
                        "to compute force_error and risk-coverage metrics."
                    )

                gate_samples: List[np.ndarray] = []
                pred_forces_ref: torch.Tensor | None = None
                for _ in range(args.uq_samples):
                    output = model(batch_dict, training=False)
                    if "ib_uq" not in output or "gate_score" not in output["ib_uq"]:
                        raise ValueError(
                            "Model output does not contain ib_uq.gate_score. "
                            "Please evaluate an IB-UQ-enabled checkpoint."
                        )
                    gate_samples.append(_as_structure_gate(output["ib_uq"]["gate_score"], batch))
                    pred_forces_ref = output["forces"]

                gate_arr = np.stack(gate_samples, axis=0)
                gate_mean = gate_arr.mean(axis=0)
                gate_std = gate_arr.std(axis=0)

                if pred_forces_ref is None:
                    raise RuntimeError("No force predictions found.")
                rmse, mae = _force_errors_per_structure(pred_forces_ref, batch_dict["forces"], batch.ptr)

                ptr_np = _to_numpy(batch.ptr)
                for i in range(int(batch.num_graphs)):
                    row = {
                        "sample_id": running_id,
                        "split": split_name,
                        "n_atoms": int(ptr_np[i + 1] - ptr_np[i]),
                        "force_rmse": float(rmse[i]),
                        "force_mae": float(mae[i]),
                    }
                    if use_uq_stats:
                        row["gate_score_mean"] = float(gate_mean[i])
                        row["gate_score_std"] = float(gate_std[i])
                    else:
                        row["gate_score"] = float(gate_mean[i])

                    writer.writerow(row)

                    all_gate.append(float(gate_mean[i]))
                    all_rmse.append(float(rmse[i]))
                    all_mae.append(float(mae[i]))
                    all_split.append(split_label)
                    running_id += 1

    y_true = np.array(all_split, dtype=np.int64)
    y_score = np.array(all_gate, dtype=np.float64)
    rmse_arr = np.array(all_rmse, dtype=np.float64)
    mae_arr = np.array(all_mae, dtype=np.float64)

    metrics = compute_ood_metrics(y_true, y_score)
    risk_curve_rmse, risk_curve_mae = selective_risk(
        gate_scores=y_score,
        force_rmse=rmse_arr,
        force_mae=mae_arr,
        coverage_points=coverage_points,
    )

    id_mask = y_true == 0
    ood_mask = y_true == 1
    summary = {
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "coverage_points": list(coverage_points),
        "risk_curve_rmse": risk_curve_rmse,
        "risk_curve_mae": risk_curve_mae,
        "gate_score_stats": {
            "id": _quantiles(y_score[id_mask]),
            "ood": _quantiles(y_score[ood_mask]),
        },
        "error_stats": {
            "id": {
                "force_rmse": _quantiles(rmse_arr[id_mask]),
                "force_mae": _quantiles(mae_arr[id_mask]),
            },
            "ood": {
                "force_rmse": _quantiles(rmse_arr[ood_mask]),
                "force_mae": _quantiles(mae_arr[ood_mask]),
            },
        },
        "n_samples": {
            "id": int(id_mask.sum()),
            "ood": int(ood_mask.sum()),
            "total": int(len(y_true)),
        },
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _risk_at(c: float, curve: Sequence[float]) -> float:
        idx = int(np.argmin(np.abs(np.array(coverage_points) - c)))
        return float(curve[idx])

    print("OOD gate evaluation summary")
    print(f"AUROC: {summary['auroc']:.6f}")
    print(f"AUPRC: {summary['auprc']:.6f}")
    print(
        "Risk (RMSE) @coverage {100%,50%,10%}: "
        f"{_risk_at(1.0, risk_curve_rmse):.6f}, "
        f"{_risk_at(0.5, risk_curve_rmse):.6f}, "
        f"{_risk_at(0.1, risk_curve_rmse):.6f}"
    )
    print(
        "Risk (MAE) @coverage {100%,50%,10%}: "
        f"{_risk_at(1.0, risk_curve_mae):.6f}, "
        f"{_risk_at(0.5, risk_curve_mae):.6f}, "
        f"{_risk_at(0.1, risk_curve_mae):.6f}"
    )
    print(f"Saved per-sample results: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
