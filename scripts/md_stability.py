#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read
from ase.md.langevin import Langevin

from mace.calculators import MACECalculator
from mace_efggm.repro import set_seed


def run_single(args, seed: int):
    set_seed(seed)
    atoms = read(args.atoms, index=0)
    calc = MACECalculator(model_paths=[args.model], device=args.device)
    atoms.calc = calc

    dyn = Langevin(
        atoms,
        args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.02,
    )

    energies = []
    max_force = 0.0
    failure_count = 0
    steps_completed = 0

    for _ in range(args.steps):
        try:
            dyn.run(1)
            e = float(atoms.get_potential_energy())
            f = atoms.get_forces()
            if not np.isfinite(e) or not np.isfinite(f).all():
                failure_count += 1
                break
            max_force = max(max_force, float(np.abs(f).max()))
            energies.append(e)
            steps_completed += 1
        except Exception:
            failure_count += 1
            break

    e0 = energies[0] if energies else 0.0
    drift = np.array([e - e0 for e in energies], dtype=float) if energies else np.array([0.0])
    return {
        "seed": seed,
        "failure_count": int(failure_count),
        "max_force_over_traj": float(max_force),
        "energy_drift": {
            "mean": float(np.mean(drift)),
            "std": float(np.std(drift)),
            "max_abs": float(np.max(np.abs(drift))),
        },
        "steps_completed": int(steps_completed),
    }


def aggregate(runs: list[dict]):
    keys = ["failure_count", "max_force_over_traj", "steps_completed"]
    agg = {}
    for key in keys:
        vals = np.array([r[key] for r in runs], dtype=float)
        agg[key] = {"mean": float(vals.mean()), "std": float(vals.std())}

    for key in ["mean", "std", "max_abs"]:
        vals = np.array([r["energy_drift"][key] for r in runs], dtype=float)
        agg[f"energy_drift_{key}"] = {"mean": float(vals.mean()), "std": float(vals.std())}
    return agg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--atoms", required=True, help="Path to seed structure (.xyz or .extxyz; first frame is used)")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--timestep_fs", type=float, default=0.5)
    p.add_argument("--out", default="md_stability.json")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", default="0", help='Comma-separated seeds, e.g. "0,1,2"')
    args = p.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    per_seed = [run_single(args, seed) for seed in seeds]

    report = {
        "steps_requested": args.steps,
        "temperature_K": args.temperature,
        "timestep_fs": args.timestep_fs,
        "per_seed": per_seed,
        "aggregate": aggregate(per_seed) if len(per_seed) > 1 else None,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
