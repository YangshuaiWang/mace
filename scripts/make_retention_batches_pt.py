#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

from scripts.make_batches_pt import (
    build_batches_from_atoms,
    dataset_recipe,
    detect_label_keys,
    load_extxyz,
    maybe_limit,
    save_batches,
    summarize_structures,
    validate_required_labels,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create retention benchmark .pt batches from old-domain extxyz.")
    p.add_argument("--retention_extxyz", type=Path, required=True)
    p.add_argument("--out_pt", type=Path, required=True)
    p.add_argument("--seed", type=int, default=123, help="Seed for deterministic subsampling order.")
    p.add_argument("--max_structures", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--foundation_checkpoint", type=Path, default=Path("checkpoints/foundation.model"))
    p.add_argument("--cutoff", type=float, default=None)
    p.add_argument("--energy_key", default=None)
    p.add_argument("--forces_key", default=None)
    p.add_argument("--stress_key", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    atoms = load_extxyz(args.retention_extxyz)
    rng = random.Random(args.seed)
    rng.shuffle(atoms)
    atoms = maybe_limit(atoms, args.max_structures)

    keys = detect_label_keys(atoms, args.energy_key, args.forces_key, args.stress_key)
    validate_required_labels(atoms, keys)
    z_table, cutoff = dataset_recipe(args.foundation_checkpoint, args.cutoff, atoms)

    summarize_structures("retention", atoms, keys)
    batches = build_batches_from_atoms(atoms, z_table=z_table, cutoff=cutoff, batch_size=args.batch_size, keys=keys)
    save_batches(args.out_pt, batches)
    print(f"Wrote retention batches: {args.out_pt} (num_batches={len(batches)})")


if __name__ == "__main__":
    main()
