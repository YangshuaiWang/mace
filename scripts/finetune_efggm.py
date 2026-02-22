#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from mace_efggm import (
    AlphaSchedule,
    FisherEMA,
    FisherEMAConfig,
    GradientMasker,
    MaskedOptimizerWrapper,
    build_parameter_mask,
    group_scores,
    group_params_by_irreps,
    irreps_wise_groups,
    module_wise_groups,
    select_top_groups,
)
from mace_efggm.repro import (
    compare_budget,
    compute_budget,
    dump_config,
    make_run_dir,
    normalize_budget_config,
    print_and_save_budget,
    set_seed,
    write_git_commit,
)


def load_batches(path: str):
    data = torch.load(path)
    if isinstance(data, dict) and "batches" in data:
        return data["batches"]
    return data


def infer_dataset_size(cfg: Dict[str, Any], batches: list[dict]) -> int:
    if "dataset_size" in cfg.get("data", {}):
        return int(cfg["data"]["dataset_size"])
    return len(batches)


def get_grouping(model, group_mode: str, irreps_grouping: str = "by_l"):
    if group_mode == "module":
        return module_wise_groups(model.named_parameters()), {"fallback_to_module": False, "unknown_fraction": 0.0, "requested_grouping": group_mode}
    if group_mode == "irreps":
        grouped_params = group_params_by_irreps(model, irreps_grouping=irreps_grouping)
        name_lookup = {id(p): n for n, p in model.named_parameters()}
        groups = {g: [name_lookup[id(p)] for p in params if id(p) in name_lookup] for g, params in grouped_params.items()}
        total = sum(p.numel() for p in model.parameters())
        unknown = sum(p.numel() for p in grouped_params.get("irrep_unknown", []))
        return groups, {"fallback_to_module": False, "unknown_fraction": (unknown / total if total else 0.0), "requested_grouping": group_mode}
    return irreps_wise_groups(model.named_parameters()), {"fallback_to_module": False, "unknown_fraction": 0.0, "requested_grouping": group_mode}


def compute_mask_coverage(model, groups: Dict[str, list[str]], kept: set[str], grouping: str) -> Dict[str, Any]:
    params = dict(model.named_parameters())
    total_params = sum(p.numel() for p in params.values())
    trainable_params = 0
    modules = {}
    for group, names in groups.items():
        group_params = int(sum(params[n].numel() for n in names if n in params))
        is_kept = group in kept
        if is_kept:
            trainable_params += group_params
        module_name = group.split("::", 1)[0]
        mod = modules.setdefault(module_name, {"total_params": 0, "trainable_params": 0, "groups_total": 0, "groups_kept": 0})
        mod["total_params"] += group_params
        mod["groups_total"] += 1
        if is_kept:
            mod["trainable_params"] += group_params
            mod["groups_kept"] += 1

    module_breakdown = {}
    for name, stats in modules.items():
        module_breakdown[name] = {
            **stats,
            "fraction_params_trainable": (stats["trainable_params"] / stats["total_params"] if stats["total_params"] > 0 else 0.0),
            "fraction_groups_kept": (stats["groups_kept"] / stats["groups_total"] if stats["groups_total"] > 0 else 0.0),
        }

    return {
        "schema_version": "1.0",
        "grouping": grouping,
        "experimental": grouping == "irreps",
        "num_groups_total": len(groups),
        "num_groups_kept": len(kept),
        "fraction_groups_kept": (len(kept) / len(groups) if groups else 0.0),
        "num_params_total": int(total_params),
        "num_params_trainable": int(trainable_params),
        "fraction_params_trainable": (trainable_params / total_params if total_params > 0 else 0.0),
        "per_module": module_breakdown,
    }


def compute_drift_spectrum(model, initial_state: Dict[str, torch.Tensor], groups: Dict[str, list[str]], grouping: str) -> Dict[str, Any]:
    params = dict(model.named_parameters())
    per_group = {}
    overall_sq = 0.0
    for group, names in groups.items():
        sq = 0.0
        for name in names:
            if name not in params or name not in initial_state:
                continue
            diff = params[name].detach() - initial_state[name]
            sq += float((diff**2).sum().item())
        per_group[group] = sq**0.5
        overall_sq += sq
    return {
        "schema_version": "1.0",
        "grouping": grouping,
        "experimental": grouping == "irreps",
        "overall_l2": overall_sq**0.5,
        "per_group_l2": per_group,
    }




def compute_irreps_spectrum_from_groups(values_by_group: Dict[str, float], model, groups: Dict[str, list[str]]) -> list[dict[str, Any]]:
    params = dict(model.named_parameters())
    per_l: Dict[str, Dict[str, float]] = {}
    for group, value in values_by_group.items():
        if not group.startswith("irrep_l"):
            continue
        l = group.split("::", 1)[0].replace("irrep_l", "")
        bucket = per_l.setdefault(l, {"sum": 0.0, "param_count": 0})
        bucket["sum"] += float(value)
        bucket["param_count"] += int(sum(params[n].numel() for n in groups.get(group, []) if n in params))
    return [
        {"l": int(l), "value": stats["sum"], "param_count": int(stats["param_count"])}
        for l, stats in sorted(per_l.items(), key=lambda kv: int(kv[0]))
    ]

def train(cfg: dict, run_dir: Path) -> None:
    set_seed(int(cfg.get("seed", 123)))
    device = torch.device(cfg.get("device", "cpu"))
    model = torch.load(cfg["model"]["foundation_checkpoint"], map_location=device)
    model.train()
    model.to(device)

    batches = load_batches(cfg["data"]["train_batches_pt"])
    budget_cfg = normalize_budget_config(cfg)
    total_steps = int(budget_cfg.max_steps)

    efggm = cfg.get("efggm", {})
    grouping_cfg = efggm.get("grouping", "module")
    if isinstance(grouping_cfg, dict):
        group_mode = grouping_cfg.get("mode", "module")
        irreps_grouping = grouping_cfg.get("irreps_grouping", "by_l")
    else:
        group_mode = grouping_cfg
        irreps_grouping = efggm.get("irreps_grouping", "by_l")
    unknown_fallback_threshold = float(efggm.get("unknown_fallback_threshold", 0.5))
    fisher_warmup_batches = int(efggm.get("fisher_warmup_batches", 50))
    fisher_ema_beta = float(efggm.get("fisher_ema_beta", 0.95))
    fisher_freeze = bool(efggm.get("fisher_freeze_after_warmup", True))
    mask_schedule = efggm.get("mask_schedule", {})
    schedule_type = mask_schedule.get("type", "linear")
    alpha_start = float(mask_schedule.get("alpha_start", efggm.get("alpha_start", 1.0)))
    alpha_end = float(mask_schedule.get("alpha_end", efggm.get("alpha_end", alpha_start)))
    schedule_steps = int(mask_schedule.get("schedule_steps", total_steps))
    schedule_interval = int(mask_schedule.get("schedule_interval", 1))

    budget_report = compute_budget(
        budget_cfg,
        dataset_size=infer_dataset_size(cfg, batches),
        world_size=int(cfg.get("world_size", 1)),
    )
    print_and_save_budget(budget_report, run_dir)

    fisher = FisherEMA(model.named_parameters(), config=FisherEMAConfig(beta=fisher_ema_beta))
    groups, grouping_meta = get_grouping(model, group_mode, irreps_grouping=irreps_grouping)
    if group_mode == "irreps" and grouping_meta["unknown_fraction"] > unknown_fallback_threshold:
        logging.warning("Unknown irreps fraction %.3f exceeds threshold %.3f; falling back to module grouping.", grouping_meta["unknown_fraction"], unknown_fallback_threshold)
        groups, _ = get_grouping(model, "module")
        grouping_meta["fallback_to_module"] = True
        grouping_meta["fallback_reason"] = "unknown_fraction_exceeded"
    grouping_meta["resolved_grouping"] = ("module" if grouping_meta.get("fallback_to_module") else group_mode)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(budget_cfg.lr),
        weight_decay=float(budget_cfg.weight_decay),
    )
    masker = GradientMasker(model.named_parameters())
    optw = MaskedOptimizerWrapper(opt, masker)
    sched = AlphaSchedule(
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        total_steps=max(schedule_steps, 1),
        schedule_type=schedule_type,
    )

    mixed_precision = budget_cfg.mixed_precision and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)
    grad_accum = budget_cfg.grad_accum_steps

    metrics_path = run_dir / "metrics.jsonl"
    current_scores: Dict[str, float] = {}
    kept_groups: set[str] = set(groups)
    current_mask: Dict[str, float] = {name: 1.0 for name, _ in model.named_parameters()}

    # warmup Fisher on new-domain data only (no optimizer steps)
    warmup_done = 0
    while warmup_done < fisher_warmup_batches:
        for batch in batches:
            if warmup_done >= fisher_warmup_batches:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            optw.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                output = model(batch, training=True, compute_force=False, compute_stress=False, compute_virials=False)
                if "target_energy" not in batch:
                    raise KeyError("Each batch must contain 'target_energy' tensor for training")
                loss = torch.nn.functional.mse_loss(output["energy"], batch["target_energy"])
            scaler.scale(loss).backward()
            fisher.update_from_model(model.named_parameters())
            warmup_done += 1

    current_scores = group_scores(fisher.tensors(), groups)
    alpha0 = sched.value(0)
    kept_groups = set(select_top_groups(current_scores, alpha0))
    current_mask = build_parameter_mask(groups, kept_groups)
    masker.update_mask(current_mask)

    initial_state = {name: p.detach().clone() for name, p in model.named_parameters()}

    step = 0
    optimizer_updates = 0
    while step < total_steps:
        for batch in batches:
            if step >= total_steps:
                break
            batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
            if step % grad_accum == 0:
                optw.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=mixed_precision):
                output = model(batch, training=True, compute_force=False, compute_stress=False, compute_virials=False)
                if "target_energy" not in batch:
                    raise KeyError("Each batch must contain 'target_energy' tensor for training")
                loss = torch.nn.functional.mse_loss(output["energy"], batch["target_energy"])

            scaled = loss / grad_accum
            scaler.scale(scaled).backward()

            if not fisher_freeze:
                fisher.update_from_model(model.named_parameters())

            if step % schedule_interval == 0:
                current_scores = group_scores(fisher.tensors(), groups)
                alpha = sched.value(step)
                kept_groups = set(select_top_groups(current_scores, alpha))
                current_mask = build_parameter_mask(groups, kept_groups)
                masker.update_mask(current_mask)

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optw.optimizer)
                masker.apply()
                scaler.step(optw.optimizer)
                scaler.update()
                optimizer_updates += 1

            coverage = compute_mask_coverage(model, groups, kept_groups, group_mode)
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "loss": float(loss.item()),
                            "optimizer_updates": optimizer_updates,
                            "alpha": float(sched.value(step)),
                            "num_groups_kept": int(coverage["num_groups_kept"]),
                            "num_params_trainable": int(coverage["num_params_trainable"]),
                        }
                    )
                    + "\n"
                )
            step += 1

    torch.save(model, run_dir / "final_model.pt")
    (run_dir / "mask.json").write_text(json.dumps(current_mask, indent=2))
    (run_dir / "fisher_group_scores.json").write_text(json.dumps(current_scores, indent=2))
    mask_coverage = compute_mask_coverage(model, groups, kept_groups, grouping_meta["resolved_grouping"])
    mask_coverage["grouping_meta"] = grouping_meta
    (run_dir / "mask_coverage.json").write_text(json.dumps(mask_coverage, indent=2))
    drift_spectrum = compute_drift_spectrum(model, initial_state, groups, grouping_meta["resolved_grouping"])
    drift_spectrum["grouping_meta"] = grouping_meta
    (run_dir / "drift_spectrum.json").write_text(json.dumps(drift_spectrum, indent=2))

    if group_mode == "irreps" and not grouping_meta.get("fallback_to_module"):
        fisher_spectrum = compute_irreps_spectrum_from_groups(current_scores, model, groups)
        drift_raw = {}
        params = dict(model.named_parameters())
        for group, names in groups.items():
            sq = 0.0
            for name in names:
                if name not in params or name not in initial_state:
                    continue
                diff = params[name].detach() - initial_state[name]
                sq += float((diff**2).sum().item())
            drift_raw[group] = sq**0.5
        drift_irreps = compute_irreps_spectrum_from_groups(drift_raw, model, groups)
        (run_dir / "fisher_spectrum.json").write_text(json.dumps([{"l": x["l"], "fisher_sum": x["value"], "param_count": x["param_count"]} for x in fisher_spectrum], indent=2))
        (run_dir / "drift_spectrum_irreps.json").write_text(json.dumps([{"l": x["l"], "drift_l2_sum": x["value"], "param_count": x["param_count"]} for x in drift_irreps], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir_base", default="runs/")
    parser.add_argument("--exp_name", default="efggm_finetune")
    parser.add_argument("--compare_budget_config", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.compare_budget_config:
        other_cfg = yaml.safe_load(Path(args.compare_budget_config).read_text())
        mismatches = compare_budget(cfg, other_cfg)
        if mismatches:
            raise ValueError(f"Compute budget mismatch: {json.dumps(mismatches, indent=2)}")
    run_dir = make_run_dir(args.run_dir_base, args.exp_name)
    dump_config(cfg, run_dir)
    write_git_commit(run_dir)
    train(cfg, run_dir)


if __name__ == "__main__":
    main()
