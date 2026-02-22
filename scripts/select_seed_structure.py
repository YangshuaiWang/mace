#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

from ase.io import read, write


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select a representative structure from extxyz for MD stability tests.")
    p.add_argument("--source_extxyz", type=Path, required=True)
    p.add_argument("--out_xyz", type=Path, required=True, help="Output path (.xyz or .extxyz)")
    p.add_argument("--mode", choices=["median_natoms", "random", "min_energy"], default="median_natoms")
    p.add_argument("--seed", type=int, default=123, help="Seed used by --mode random")
    p.add_argument("--energy_key", default=None, help="Override energy key for --mode min_energy")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.source_extxyz.exists():
        raise FileNotFoundError(f"source extxyz not found: {args.source_extxyz}")

    atoms_list = read(args.source_extxyz, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError("No structures found in source extxyz")

    if args.mode == "median_natoms":
        natoms = [len(at) for at in atoms_list]
        sorted_counts = sorted(natoms)
        median = sorted_counts[len(sorted_counts) // 2]
        idx = min(range(len(atoms_list)), key=lambda i: abs(len(atoms_list[i]) - median))
    elif args.mode == "random":
        idx = random.Random(args.seed).randrange(len(atoms_list))
    else:
        available_info = set().union(*(a.info.keys() for a in atoms_list))
        key = args.energy_key or ("REF_energy" if "REF_energy" in available_info else "energy")
        missing = [i for i, at in enumerate(atoms_list) if key not in at.info]
        if missing:
            raise ValueError(
                f"Cannot run --mode min_energy; missing energy key '{key}' for {len(missing)} structures. "
                f"First missing indices: {missing[:5]}"
            )
        idx = min(range(len(atoms_list)), key=lambda i: float(atoms_list[i].info[key]))

    chosen = atoms_list[idx]
    args.out_xyz.parent.mkdir(parents=True, exist_ok=True)
    write(args.out_xyz, chosen)

    elements = sorted(set(chosen.get_chemical_symbols()))
    print(
        f"Selected index={idx} natoms={len(chosen)} elements={elements} "
        f"pbc={bool(chosen.pbc.any())} output={args.out_xyz}"
    )


if __name__ == "__main__":
    main()
