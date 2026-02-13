"""Smoke-check IB-UQ training plumbing with and without wide gate-close regularization.

Run:
    python scripts/smoke_train_ib_uq.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# PyTorch >=2.6 defaults torch.load(..., weights_only=True), while e3nn constants
# contain trusted Python objects such as `slice`.
torch.serialization.add_safe_globals([slice])
from e3nn import o3

try:
    from mace import data, modules, tools
    from mace.tools import torch_geometric
    from mace.tools.train import take_step
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from mace import data, modules, tools
    from mace.tools import torch_geometric
    from mace.tools.train import take_step


def make_batch() -> torch_geometric.batch.Batch:
    rng = np.random.default_rng(0)
    z_table = tools.AtomicNumberTable([1, 8])
    configs = []
    for _ in range(2):
        n_atoms = 4
        configs.append(
            data.Configuration(
                atomic_numbers=rng.choice([1, 8], size=(n_atoms,)),
                positions=rng.normal(0.0, 1.0, size=(n_atoms, 3)),
                properties={
                    "energy": float(rng.normal()),
                    "forces": rng.normal(0.0, 0.1, size=(n_atoms, 3)),
                },
                property_weights={},
                weight=1.0,
            )
        )
    dataset = [data.AtomicData.from_config(cfg, z_table=z_table, cutoff=3.0) for cfg in configs]
    loader = torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=2, shuffle=False)
    return next(iter(loader))


def make_model() -> modules.MACE:
    table = tools.AtomicNumberTable([1, 8])
    return modules.ScaleShiftMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=len(table.zs),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(table.zs), dtype=float),
        avg_num_neighbors=4,
        atomic_numbers=table.zs,
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        ib_uq_enabled=True,
        ib_uq_latent_dim=4,
        ib_uq_deterministic=True,
        ib_uq_deterministic_zero_z0=True,
    ).to("cpu")


def main() -> None:
    torch.manual_seed(0)
    tools.set_default_dtype("float64")
    batch = make_batch()
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)

    output_args = {"forces": True, "virials": False, "stress": False}

    _, m0 = take_step(
        model=model,
        loss_fn=loss_fn,
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args=output_args,
        max_grad_norm=10.0,
        device=torch.device("cpu"),
        ib_uq_lambda=0.0,
        ib_uq_wide_aug="none",
        ib_uq_wide_frac=1.0,
    )
    assert "L_gate" in m0 and abs(float(m0["L_gate"])) == 0.0

    _, m1 = take_step(
        model=model,
        loss_fn=loss_fn,
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args=output_args,
        max_grad_norm=10.0,
        device=torch.device("cpu"),
        ib_uq_lambda=0.1,
        ib_uq_wide_aug="coord_noise",
        ib_uq_wide_frac=1.0,
    )
    assert "L_gate" in m1
    assert "train/gate_score_wide" in m1

    print("smoke_train_ib_uq: PASS", {k: m1[k] for k in ["L_gate", "train/gate_score_id", "train/gate_score_wide"] if k in m1})


if __name__ == "__main__":
    main()
