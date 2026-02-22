#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml
from ase import Atoms
from ase.io import read

from scripts.finetune_baselines import load_batches as load_baseline_batches
from scripts.make_batches_pt import detect_label_keys, validate_required_labels


def _parse_keys(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _sample_frames(path: Path, max_structures: int) -> list[Atoms]:
    frames = read(path, index=f":{max_structures}")
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise ValueError(f"No frames found in extxyz: {path}")
    return frames


def _detect_with_candidates(frames: list[Atoms], energy: list[str], forces: list[str], stress: list[str]) -> dict[str, str | None]:
    info_keys = set().union(*(a.info.keys() for a in frames))
    array_keys = set().union(*(a.arrays.keys() for a in frames))

    e = next((k for k in energy if k in info_keys), None)
    f = next((k for k in forces if k in array_keys), None)
    s = next((k for k in stress if k in info_keys), None)
    return {"energy": e, "forces": f, "stress": s}


def _check_finite(frames: list[Atoms], key: str | None, source: str) -> tuple[bool, int]:
    if key is None:
        return False, len(frames)
    bad = 0
    for at in frames:
        if source == "info":
            if key not in at.info or not np.isfinite(np.asarray(at.info[key], dtype=float)).all():
                bad += 1
        else:
            if key not in at.arrays or not np.isfinite(np.asarray(at.arrays[key], dtype=float)).all():
                bad += 1
    return bad == 0, bad


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _magnitude_warnings(frames: list[Atoms], keys: dict[str, str | None]) -> tuple[dict, list[str]]:
    energies = [abs(float(at.info[keys["energy"]])) / max(len(at), 1) for at in frames if keys["energy"] in at.info]
    forces = []
    for at in frames:
        if keys["forces"] in at.arrays:
            f = np.asarray(at.arrays[keys["forces"]], dtype=float)
            forces.extend(np.linalg.norm(f, axis=1).tolist())
    stresses = [
        float(np.linalg.norm(np.asarray(at.info[keys["stress"]], dtype=float).reshape(-1)))
        for at in frames
        if keys["stress"] is not None and keys["stress"] in at.info
    ]

    stats = {
        "energy_per_atom_median_abs": float(np.median(energies)) if energies else None,
        "force_magnitude_median": float(np.median(forces)) if forces else None,
        "stress_magnitude_median": float(np.median(stresses)) if stresses else None,
    }
    warnings: list[str] = []
    if stats["force_magnitude_median"] is not None:
        if stats["force_magnitude_median"] > 1e3:
            warnings.append("Force magnitude median is very high (>1e3); verify units.")
        if stats["force_magnitude_median"] < 1e-6:
            warnings.append("Force magnitude median is very low (<1e-6); verify units.")
    if stats["energy_per_atom_median_abs"] is not None:
        if stats["energy_per_atom_median_abs"] > 1e3:
            warnings.append("Energy/atom median is very high (>1e3 eV); verify units.")
        if stats["energy_per_atom_median_abs"] < 1e-6:
            warnings.append("Energy/atom median is very low (<1e-6 eV); verify units.")
    return stats, warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run preflight sanity checks before sweeps.")
    p.add_argument("--new_extxyz", type=Path, required=True)
    p.add_argument("--retention_extxyz", type=Path, required=True)
    p.add_argument("--foundation_checkpoint", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, default=Path("./data/preflight"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_structures", type=int, default=50)
    p.add_argument("--expect_keys_energy", default="energy")
    p.add_argument("--expect_keys_forces", default="forces")
    p.add_argument("--expect_keys_stress", default="stress")
    p.add_argument("--require_stress", action="store_true")
    p.add_argument("--md_steps", type=int, default=20)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--skip_smoke_train", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"warnings": []}

    if not args.foundation_checkpoint.exists():
        raise FileNotFoundError(f"Foundation checkpoint not found: {args.foundation_checkpoint}")

    energy_keys = _parse_keys(args.expect_keys_energy)
    forces_keys = _parse_keys(args.expect_keys_forces)
    stress_keys = _parse_keys(args.expect_keys_stress)

    new_frames = _sample_frames(args.new_extxyz, args.max_structures)
    retention_frames = _sample_frames(args.retention_extxyz, args.max_structures)

    chosen = _detect_with_candidates(new_frames, energy_keys, forces_keys, stress_keys)
    report["chosen_energy_key"] = chosen["energy"]
    report["chosen_forces_key"] = chosen["forces"]
    report["chosen_stress_key"] = chosen["stress"]

    if chosen["energy"] is None or chosen["forces"] is None:
        raise ValueError(
            "Missing required labels in new_extxyz. "
            f"Expected energy keys {energy_keys} and forces keys {forces_keys}. "
            f"Found info keys {sorted(set().union(*(a.info.keys() for a in new_frames)))} and "
            f"array keys {sorted(set().union(*(a.arrays.keys() for a in new_frames)))}."
        )

    # Reuse make_batches_pt helpers for alignment rules.
    helper_keys = detect_label_keys(new_frames, chosen["energy"], chosen["forces"], chosen["stress"])
    validate_required_labels(new_frames, helper_keys)

    finite_e, bad_e = _check_finite(new_frames, chosen["energy"], "info")
    finite_f, bad_f = _check_finite(new_frames, chosen["forces"], "arrays")
    stress_ok = True
    if args.require_stress:
        if chosen["stress"] is None:
            raise ValueError(f"--require_stress set, but none of stress keys found: {stress_keys}")
        stress_ok, bad_s = _check_finite(new_frames, chosen["stress"], "info")
        if not stress_ok:
            raise ValueError(f"Stress key '{chosen['stress']}' has NaN/Inf or missing values in {bad_s} frames")

    if not finite_e:
        raise ValueError(f"Energy key '{chosen['energy']}' has NaN/Inf or missing values in {bad_e} frames")
    if not finite_f:
        raise ValueError(f"Forces key '{chosen['forces']}' has NaN/Inf or missing values in {bad_f} frames")

    report["label_ok"] = True
    report["stress_ok"] = stress_ok and (chosen["stress"] is not None or not args.require_stress)
    report["extxyz_keys"] = {
        "new_info": sorted(set().union(*(a.info.keys() for a in new_frames))),
        "new_arrays": sorted(set().union(*(a.arrays.keys() for a in new_frames))),
        "retention_info": sorted(set().union(*(a.info.keys() for a in retention_frames))),
        "retention_arrays": sorted(set().union(*(a.arrays.keys() for a in retention_frames))),
    }

    stats, mag_warnings = _magnitude_warnings(new_frames, chosen)
    report["magnitude_stats"] = stats
    report["warnings"].extend(mag_warnings)

    unit_meta = sorted(
        {
            key
            for at in new_frames
            for key in at.info.keys()
            if "unit" in key.lower() or key.lower() in {"comment", "units"}
        }
    )
    if unit_meta:
        report["warnings"].append(f"Potential unit metadata fields in new_extxyz: {unit_meta}")

    out_train = args.out_dir / "preflight_train_batches.pt"
    out_valid = args.out_dir / "preflight_valid_batches.pt"
    _run(
        [
            sys.executable,
            "-m",
            "scripts.make_batches_pt",
            "--extxyz",
            str(args.new_extxyz),
            "--valid_fraction",
            "0.1",
            "--out_train_pt",
            str(out_train),
            "--out_valid_pt",
            str(out_valid),
            "--foundation_checkpoint",
            str(args.foundation_checkpoint),
            "--seed",
            str(args.seed),
            "--max_train_structures",
            str(args.max_structures),
            "--max_valid_structures",
            str(max(1, args.max_structures // 10)),
            "--energy_key",
            chosen["energy"],
            "--forces_key",
            chosen["forces"],
        ]
        + (["--stress_key", chosen["stress"]] if chosen["stress"] else [])
    )

    train_batches = load_baseline_batches(str(out_train))
    valid_batches = load_baseline_batches(str(out_valid))
    if not valid_batches:
        raise ValueError("Validation batches are empty after make_batches_pt split-mode preflight")
    required_keys = {"target_energy", "target_forces"}
    if not required_keys.issubset(valid_batches[0].keys()):
        raise ValueError(
            f"Batch schema mismatch. Expected keys {sorted(required_keys)} in first valid batch, "
            f"found {sorted(valid_batches[0].keys())}."
        )
    report["batch_schema_ok"] = True
    report["valid_ok"] = True

    smoke_baseline_ok = False
    smoke_efggm_ok = False
    if args.skip_smoke_train:
        report["warnings"].append("Smoke training skipped via --skip_smoke_train")
        smoke_baseline_ok = True
        smoke_efggm_ok = True
    else:
        baseline_cfg = {
            "seed": args.seed,
            "device": args.device,
            "model": {"foundation_checkpoint": str(args.foundation_checkpoint)},
            "data": {"train_batches_pt": str(out_train), "valid_batches_pt": str(out_valid)},
            "budget": {
                "optimizer": "adam",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "scheduler": "none",
                "scheduler_params": {},
                "max_steps": 2,
                "batch_size": 1,
                "valid_batch_size": 1,
                "grad_accum_steps": 1,
                "mixed_precision": False,
                "early_stopping": {"patience": 0, "metric": "valid_loss"},
            },
            "baselines": ["full_ft"],
        }
        efggm_cfg = {
            "seed": args.seed,
            "device": args.device,
            "model": {"foundation_checkpoint": str(args.foundation_checkpoint)},
            "data": {"train_batches_pt": str(out_train)},
            "budget": {
                "optimizer": "adam",
                "lr": 1e-3,
                "weight_decay": 0.0,
                "scheduler": "none",
                "scheduler_params": {},
                "max_steps": 2,
                "batch_size": 1,
                "grad_accum_steps": 1,
                "mixed_precision": False,
                "early_stopping": {"patience": 0, "metric": "valid_loss"},
            },
            "efggm": {
                "grouping": "module",
                "fisher_warmup_batches": min(5, max(1, len(train_batches))),
                "fisher_ema_beta": 0.95,
                "fisher_freeze_after_warmup": True,
                "mask_schedule": {
                    "type": "linear",
                    "alpha_start": 0.9,
                    "alpha_end": 0.3,
                    "schedule_steps": 2,
                    "schedule_interval": 1,
                },
            },
        }
        with TemporaryDirectory(prefix="preflight_cfg_") as td:
            td_path = Path(td)
            bcfg = td_path / "baseline_smoke.yaml"
            ecfg = td_path / "efggm_smoke.yaml"
            bcfg.write_text(yaml.safe_dump(baseline_cfg))
            ecfg.write_text(yaml.safe_dump(efggm_cfg))
            _run(
                [
                    sys.executable,
                    "-m",
                    "scripts.finetune_baselines",
                    "--config",
                    str(bcfg),
                    "--run_dir_base",
                    str(args.out_dir / "smoke_runs"),
                    "--exp_name",
                    "baseline_smoke",
                ]
            )
            smoke_baseline_ok = True
            _run(
                [
                    sys.executable,
                    "-m",
                    "scripts.finetune_efggm",
                    "--config",
                    str(ecfg),
                    "--run_dir_base",
                    str(args.out_dir / "smoke_runs"),
                    "--exp_name",
                    "efggm_smoke",
                ]
            )
            smoke_efggm_ok = True

    report["smoke_train_ok"] = {"baseline": smoke_baseline_ok, "efggm": smoke_efggm_ok}

    seed_out = args.out_dir / "seed_structure.extxyz"
    _run(
        [
            sys.executable,
            "-m",
            "scripts.select_seed_structure",
            "--source_extxyz",
            str(args.new_extxyz),
            "--out_xyz",
            str(seed_out),
            "--mode",
            "median_natoms",
            "--seed",
            str(args.seed),
        ]
    )
    seed_atoms = read(seed_out)
    natoms = np.array([len(a) for a in new_frames], dtype=float)
    p10, p90 = np.percentile(natoms, [10, 90])
    seed_n = len(seed_atoms)
    md_seed_ok = p10 <= seed_n <= p90
    if not md_seed_ok:
        report["warnings"].append(
            f"Selected seed natoms={seed_n} is outside sampled 10-90 percentile range [{p10:.1f}, {p90:.1f}]"
        )

    seed_energy = None
    if chosen["energy"] in seed_atoms.info:
        seed_energy = float(seed_atoms.info[chosen["energy"]])
    report["md_seed"] = {
        "natoms": seed_n,
        "elements": sorted(set(seed_atoms.get_chemical_symbols())),
        "pbc": bool(seed_atoms.pbc.any()),
        "cell_volume": float(seed_atoms.cell.volume),
        "energy": seed_energy,
    }
    report["md_seed_ok"] = True

    report_path = args.out_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("=== Preflight checklist ===")
    print(f"[OK] label/key alignment: {report['label_ok']}")
    print(f"[OK] batch schema + valid split: {report['batch_schema_ok']} / {report['valid_ok']}")
    print(f"[OK] smoke train baseline: {report['smoke_train_ok']['baseline']}")
    print(f"[OK] smoke train efggm: {report['smoke_train_ok']['efggm']}")
    print(
        f"[OK] seed structure: natoms={report['md_seed']['natoms']} elements={report['md_seed']['elements']} "
        f"pbc={report['md_seed']['pbc']} volume={report['md_seed']['cell_volume']:.3f}"
    )
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
