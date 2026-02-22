from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.io import write

from scripts.finetune_baselines import load_batches
from scripts.make_batches_pt import (
    build_batches_from_atoms,
    dataset_recipe,
    detect_label_keys,
    load_extxyz,
    validate_required_labels,
)


def _atoms(shift: float) -> Atoms:
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 + shift]])
    atoms.info["REF_energy"] = -1.0 + shift
    atoms.info["REF_stress"] = np.zeros(6)
    atoms.arrays["REF_forces"] = np.zeros((2, 3))
    return atoms


def test_make_batches_pt_loadable(tmp_path: Path) -> None:
    extxyz = tmp_path / "tiny.extxyz"
    write(extxyz, [_atoms(0.0), _atoms(0.1)], format="extxyz")

    atoms = load_extxyz(extxyz)
    keys = detect_label_keys(atoms, energy_key=None, forces_key=None, stress_key=None)
    validate_required_labels(atoms, keys)
    z_table, cutoff = dataset_recipe(tmp_path / "missing.model", cutoff=4.5, atoms_list=atoms)
    batches = build_batches_from_atoms(atoms, z_table=z_table, cutoff=cutoff, batch_size=1, keys=keys)

    out_pt = tmp_path / "batches.pt"
    torch.save({"batches": batches}, out_pt)

    loaded = load_batches(str(out_pt))
    assert len(loaded) == 2
    assert "target_energy" in loaded[0]
    assert "positions" in loaded[0]
