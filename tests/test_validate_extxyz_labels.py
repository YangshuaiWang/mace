from __future__ import annotations

import subprocess
import sys

import numpy as np
from ase import Atoms
from ase.io import write


def test_validate_extxyz_labels_success(tmp_path):
    extxyz_path = tmp_path / "synthetic.extxyz"

    atoms1 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms1.info["energy"] = -1.0
    atoms1.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    atoms2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.76]])
    atoms2.info["energy"] = -0.8
    atoms2.arrays["forces"] = np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])

    write(extxyz_path, [atoms1, atoms2], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.validate_extxyz_labels",
            "--extxyz",
            str(extxyz_path),
            "--max_frames",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Validation status: OK" in result.stdout
