"""Quick IB-UQ sanity check.

Run with:
    python scripts/quick_check_ib_uq.py
"""

from __future__ import annotations

import numpy as np
import torch
import sys
from pathlib import Path

# PyTorch >=2.6 defaults torch.load(..., weights_only=True), while e3nn constants
# contain trusted Python objects such as `slice`.
torch.serialization.add_safe_globals([slice])
from e3nn import o3

try:
    from mace import data, modules, tools
    from mace.tools import torch_geometric
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    from mace import data, modules, tools
    from mace.tools import torch_geometric


def _make_random_config(rng: np.random.Generator, n_atoms: int = 4) -> data.Configuration:
    atomic_numbers = rng.choice([1, 8], size=(n_atoms,))
    positions = rng.normal(loc=0.0, scale=1.0, size=(n_atoms, 3))
    return data.Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions,
        properties={},
        property_weights={},
        weight=1.0,
    )


def _build_model(table: tools.AtomicNumberTable) -> modules.MACE:
    atomic_energies = np.zeros(len(table.zs), dtype=float)
    model = modules.MACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(table.zs),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=4,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
        ib_uq_enabled=True,
        ib_uq_latent_dim=4,
        ib_uq_deterministic=True,
        ib_uq_deterministic_zero_z0=True,
    )
    return model.to("cpu")


def main() -> None:
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    z_table = tools.AtomicNumberTable([1, 8])
    config_a = _make_random_config(rng)
    config_b = _make_random_config(rng)

    atomic_data_a = data.AtomicData.from_config(config_a, z_table=z_table, cutoff=3.0)
    atomic_data_b = data.AtomicData.from_config(config_b, z_table=z_table, cutoff=3.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data_a, atomic_data_b],
        batch_size=2,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cpu").to_dict()

    model = _build_model(z_table)
    model.eval()

    output_1 = model(batch, training=False)

    # 1) shape checks
    assert output_1["energy"].shape == (2,)
    assert output_1["forces"].shape == (len(config_a.atomic_numbers) + len(config_b.atomic_numbers), 3)

    # 2) gate_score existence + range checks
    assert "ib_uq" in output_1
    gate_score = output_1["ib_uq"]["gate_score"]
    assert gate_score.shape == (2,)
    assert torch.all((gate_score >= 0.0) & (gate_score <= 1.0))

    # 3) deterministic mode consistency
    output_2 = model(batch, training=False)

    assert torch.allclose(output_1["energy"], output_2["energy"], rtol=0.0, atol=0.0)
    assert torch.allclose(output_1["forces"], output_2["forces"], rtol=0.0, atol=0.0)
    assert torch.allclose(gate_score, output_2["ib_uq"]["gate_score"], rtol=0.0, atol=0.0)

    print("quick_check_ib_uq: PASS")


if __name__ == "__main__":
    main()
