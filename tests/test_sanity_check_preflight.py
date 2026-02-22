from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from ase import Atoms
from ase.io import write


def _write_extxyz(path, nframes: int = 10):
    frames = []
    for i in range(nframes):
        at = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74 + 0.01 * i]])
        at.info["REF_energy"] = -1.0 + 0.01 * i
        at.arrays["REF_forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
        frames.append(at)
    write(path, frames, format="extxyz")


def test_preflight_label_and_batch_schema(tmp_path):
    new_path = tmp_path / "new.extxyz"
    retention_path = tmp_path / "retention.extxyz"
    _write_extxyz(new_path, nframes=12)
    _write_extxyz(retention_path, nframes=8)

    ckpt = tmp_path / "foundation.model"
    dummy = SimpleNamespace(atomic_numbers=torch.tensor([1]), r_max=torch.tensor(5.0))
    torch.save(dummy, ckpt)

    out_dir = tmp_path / "preflight"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.sanity_check_preflight",
            "--new_extxyz",
            str(new_path),
            "--retention_extxyz",
            str(retention_path),
            "--foundation_checkpoint",
            str(ckpt),
            "--out_dir",
            str(out_dir),
            "--max_structures",
            "10",
            "--expect_keys_energy",
            "REF_energy,energy",
            "--expect_keys_forces",
            "REF_forces,forces",
            "--skip_smoke_train",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    report_path = out_dir / "preflight_report.json"
    assert report_path.exists()

    report = __import__("json").loads(report_path.read_text())
    assert report["label_ok"] is True
    assert report["batch_schema_ok"] is True
    assert report["valid_ok"] is True
