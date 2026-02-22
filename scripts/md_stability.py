#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import units
from ase.io import read
from ase.md.langevin import Langevin

from mace.calculators import MACECalculator


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--atoms", required=True, help="Path to structure readable by ASE")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--timestep_fs", type=float, default=0.5)
    p.add_argument("--out", default="md_stability.json")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    atoms = read(args.atoms)
    calc = MACECalculator(model_paths=[args.model], device=args.device)
    atoms.calc = calc

    dyn = Langevin(
        atoms,
        args.timestep_fs * units.fs,
        temperature_K=args.temperature,
        friction=0.02,
    )

    energies = []

    def collect():
        energies.append(float(atoms.get_potential_energy()))

    dyn.attach(collect, interval=1)
    dyn.run(args.steps)

    report = {
        "steps": args.steps,
        "temperature_K": args.temperature,
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "energy_drift": float(energies[-1] - energies[0]) if len(energies) > 1 else 0.0,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
