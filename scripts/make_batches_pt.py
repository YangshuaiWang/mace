#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Iterable

import torch
from ase import Atoms
from ase.io import read

from mace import data as mace_data
from mace.data import AtomicData
from mace.tools import AtomicNumberTable, torch_geometric


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert extxyz structures into .pt batches consumable by finetuning scripts."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--train_extxyz", type=Path, help="Training extxyz path.")
    source.add_argument(
        "--extxyz",
        type=Path,
        help="Single extxyz path to split into train/valid with --valid_fraction.",
    )
    parser.add_argument("--valid_extxyz", type=Path, help="Optional separate validation extxyz path.")
    parser.add_argument(
        "--valid_fraction",
        type=float,
        default=None,
        help="Validation fraction in (0,1). Only used with --extxyz.",
    )
    parser.add_argument("--out_train_pt", type=Path, required=True, help="Output .pt file for train batches.")
    parser.add_argument("--out_valid_pt", type=Path, default=None, help="Output .pt file for valid batches.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for deterministic split/shuffle.")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of structures per packed batch.")
    parser.add_argument("--max_train_structures", type=int, default=None, help="Optional cap for train structures.")
    parser.add_argument("--max_valid_structures", type=int, default=None, help="Optional cap for valid structures.")
    parser.add_argument(
        "--foundation_checkpoint",
        type=Path,
        default=Path("checkpoints/foundation.model"),
        help="Checkpoint used to infer atomic number table and cutoff; defaults to ./checkpoints/foundation.model.",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Neighbor cutoff (Angstrom) used only if foundation checkpoint is unavailable.",
    )
    parser.add_argument("--energy_key", default=None, help="Override energy key in atoms.info.")
    parser.add_argument("--forces_key", default=None, help="Override forces key in atoms.arrays.")
    parser.add_argument("--stress_key", default=None, help="Override stress key in atoms.info.")
    return parser.parse_args()


def load_extxyz(path: Path) -> list[Atoms]:
    if not path.exists():
        raise FileNotFoundError(f"Input extxyz not found: {path}")
    frames = read(path, index=":")
    if not isinstance(frames, list):
        frames = [frames]
    if not frames:
        raise ValueError(f"No structures found in extxyz: {path}")
    return frames


def _pick_key(available: set[str], user_key: str | None, candidates: list[str], label: str) -> str:
    if user_key:
        if user_key not in available:
            raise KeyError(f"Requested --{label}_key '{user_key}' was not found. Available keys: {sorted(available)}")
        return user_key
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise KeyError(
        f"Could not find {label} labels in extxyz. Tried keys {candidates}. "
        f"Use --{label}_key to select the correct key."
    )


def detect_label_keys(atoms_list: list[Atoms], energy_key: str | None, forces_key: str | None, stress_key: str | None) -> dict[str, str | None]:
    info_keys = set().union(*(a.info.keys() for a in atoms_list))
    array_keys = set().union(*(a.arrays.keys() for a in atoms_list))
    selected_energy = _pick_key(info_keys, energy_key, ["REF_energy", "energy"], "energy")
    selected_forces = _pick_key(array_keys, forces_key, ["REF_forces", "forces"], "forces")
    selected_stress = None
    if stress_key is not None:
        if stress_key not in info_keys:
            raise KeyError(f"Requested --stress_key '{stress_key}' was not found in atoms.info keys: {sorted(info_keys)}")
        selected_stress = stress_key
    else:
        for candidate in ["REF_stress", "stress"]:
            if candidate in info_keys:
                selected_stress = candidate
                break
    return {"energy": selected_energy, "forces": selected_forces, "stress": selected_stress}


def validate_required_labels(atoms_list: list[Atoms], keys: dict[str, str | None]) -> None:
    missing_energy = [i for i, atoms in enumerate(atoms_list) if keys["energy"] not in atoms.info]
    missing_forces = [i for i, atoms in enumerate(atoms_list) if keys["forces"] not in atoms.arrays]
    if missing_energy:
        raise ValueError(
            f"Missing energy label '{keys['energy']}' in {len(missing_energy)} structures. "
            f"First missing indices: {missing_energy[:5]}"
        )
    if missing_forces:
        raise ValueError(
            f"Missing forces label '{keys['forces']}' in {len(missing_forces)} structures. "
            f"First missing indices: {missing_forces[:5]}"
        )


def dataset_recipe(foundation_checkpoint: Path, cutoff: float | None, atoms_list: list[Atoms]) -> tuple[AtomicNumberTable, float]:
    if foundation_checkpoint.exists():
        model = torch.load(foundation_checkpoint, map_location="cpu")
        z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
        return z_table, float(model.r_max.item())
    if cutoff is None:
        raise ValueError(
            "Foundation checkpoint not found and --cutoff not provided. "
            "Either place checkpoint at ./checkpoints/foundation.model, pass --foundation_checkpoint, or provide --cutoff."
        )
    atomic_numbers = sorted({int(z) for atoms in atoms_list for z in atoms.get_atomic_numbers()})
    return AtomicNumberTable(atomic_numbers), cutoff


def summarize_structures(name: str, atoms_list: list[Atoms], keys: dict[str, str | None]) -> None:
    natoms = [len(a) for a in atoms_list]
    elements = sorted({symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()})
    stress_key = keys["stress"]
    stress_count = sum(int(stress_key in atoms.info) for atoms in atoms_list) if stress_key else 0
    has_pbc = sum(int(bool(atoms.pbc.any())) for atoms in atoms_list)
    print(
        f"[{name}] structures={len(atoms_list)} avg_atoms={sum(natoms)/len(natoms):.2f} "
        f"elements={elements} energy={len(atoms_list)}/{len(atoms_list)} "
        f"forces={len(atoms_list)}/{len(atoms_list)} stress={stress_count}/{len(atoms_list)} pbc={has_pbc}/{len(atoms_list)}"
    )


def _canonicalize_atoms(atoms: Atoms, keys: dict[str, str | None]) -> Atoms:
    out = atoms.copy()
    out.info["__prep_energy__"] = float(out.info[keys["energy"]])
    out.arrays["__prep_forces__"] = out.arrays[keys["forces"]]
    if keys["stress"] is not None and keys["stress"] in out.info:
        out.info["__prep_stress__"] = out.info[keys["stress"]]
    return out


def build_batches_from_atoms(
    atoms_list: list[Atoms],
    *,
    z_table: AtomicNumberTable,
    cutoff: float,
    batch_size: int,
    keys: dict[str, str | None],
) -> list[dict[str, Any]]:
    key_spec = mace_data.KeySpecification(
        info_keys={"energy": "__prep_energy__", "stress": "__prep_stress__", "head": "head"},
        arrays_keys={"forces": "__prep_forces__"},
    )
    configs = [mace_data.config_from_atoms(_canonicalize_atoms(a, keys), key_specification=key_spec) for a in atoms_list]
    dataset = [AtomicData.from_config(config, z_table=z_table, cutoff=cutoff) for config in configs]
    loader = torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    batches: list[dict[str, Any]] = []
    for batch in loader:
        batch_dict = batch.to_dict()
        batch_dict["target_energy"] = batch_dict["energy"].clone()
        batch_dict["target_forces"] = batch_dict["forces"].clone()
        if "stress" in batch_dict:
            batch_dict["target_stress"] = batch_dict["stress"].clone()
        batches.append(batch_dict)
    return batches


def save_batches(path: Path, batches: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"batches": batches}, path)


def maybe_limit(items: list[Any], max_items: int | None) -> list[Any]:
    if max_items is None:
        return items
    return items[:max_items]


def deterministic_split(items: list[Any], valid_fraction: float, seed: int) -> tuple[list[Any], list[Any]]:
    if not (0.0 < valid_fraction < 1.0):
        raise ValueError("--valid_fraction must be in (0,1)")
    if len(items) < 2:
        raise ValueError("Need at least 2 structures when using --valid_fraction split")
    idx = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(idx)
    valid_size = max(1, int(round(len(items) * valid_fraction)))
    valid_idx = set(idx[:valid_size])
    train = [items[i] for i in range(len(items)) if i not in valid_idx]
    valid = [items[i] for i in range(len(items)) if i in valid_idx]
    return train, valid


def main() -> None:
    args = _parse_args()
    if args.train_extxyz and args.extxyz:
        raise ValueError("Use either --train_extxyz or --extxyz, not both.")
    if args.train_extxyz and args.valid_fraction is not None:
        raise ValueError("--valid_fraction only applies with --extxyz")
    if args.extxyz and args.valid_extxyz:
        raise ValueError("Cannot combine --extxyz split mode with --valid_extxyz")
    if args.valid_extxyz and args.out_valid_pt is None:
        raise ValueError("--out_valid_pt is required when --valid_extxyz is provided")

    if args.extxyz:
        all_atoms = load_extxyz(args.extxyz)
        if args.valid_fraction is None:
            raise ValueError("--valid_fraction is required when using --extxyz")
        train_atoms, valid_atoms = deterministic_split(all_atoms, args.valid_fraction, args.seed)
    else:
        train_atoms = load_extxyz(args.train_extxyz)
        valid_atoms = load_extxyz(args.valid_extxyz) if args.valid_extxyz else []

    train_atoms = maybe_limit(train_atoms, args.max_train_structures)
    valid_atoms = maybe_limit(valid_atoms, args.max_valid_structures)

    keys = detect_label_keys(train_atoms + valid_atoms, args.energy_key, args.forces_key, args.stress_key)
    validate_required_labels(train_atoms, keys)
    if valid_atoms:
        validate_required_labels(valid_atoms, keys)

    z_table, cutoff = dataset_recipe(args.foundation_checkpoint, args.cutoff, train_atoms + valid_atoms)
    print(f"Using z_table={list(z_table.zs)} cutoff={cutoff:.3f}")

    summarize_structures("train", train_atoms, keys)
    train_batches = build_batches_from_atoms(train_atoms, z_table=z_table, cutoff=cutoff, batch_size=args.batch_size, keys=keys)
    save_batches(args.out_train_pt, train_batches)
    print(f"Wrote train batches: {args.out_train_pt} (num_batches={len(train_batches)})")

    if valid_atoms:
        summarize_structures("valid", valid_atoms, keys)
        valid_batches = build_batches_from_atoms(valid_atoms, z_table=z_table, cutoff=cutoff, batch_size=args.batch_size, keys=keys)
        out_valid = args.out_valid_pt
        if out_valid is None:
            raise ValueError("Validation data exists but --out_valid_pt is missing")
        save_batches(out_valid, valid_batches)
        print(f"Wrote valid batches: {out_valid} (num_batches={len(valid_batches)})")


if __name__ == "__main__":
    main()
