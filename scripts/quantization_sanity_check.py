import numpy as np
import torch
from e3nn import o3

from mace.data.atomic_data import AtomicData
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE
from mace.modules.quantization import QuantizationConfig
from mace.tools.torch_geometric import Batch


def random_rotation(device: torch.device) -> torch.Tensor:
    matrix = torch.randn(3, 3, device=device)
    q, _ = torch.linalg.qr(matrix)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def build_batch(positions: torch.Tensor, num_elements: int) -> dict:
    num_atoms = positions.shape[0]
    node_attrs = torch.zeros(num_atoms, num_elements, device=positions.device)
    node_attrs[:, 0] = 1.0
    edge_index = torch.tensor(
        [[0, 1], [1, 0]], device=positions.device, dtype=torch.long
    )
    shifts = torch.zeros(edge_index.shape[1], 3, device=positions.device)
    unit_shifts = torch.zeros_like(shifts)
    cell = torch.zeros(3, 3, device=positions.device)
    data = AtomicData(
        edge_index=edge_index,
        node_attrs=node_attrs,
        positions=positions,
        shifts=shifts,
        unit_shifts=unit_shifts,
        cell=cell,
        weight=None,
        head=None,
        energy_weight=None,
        forces_weight=None,
        stress_weight=None,
        virials_weight=None,
        dipole_weight=None,
        charges_weight=None,
        polarizability_weight=None,
        forces=None,
        energy=None,
        stress=None,
        virials=None,
        dipole=None,
        charges=None,
        polarizability=None,
        elec_temp=None,
        total_charge=None,
        total_spin=None,
        pbc=None,
    )
    batch = Batch.from_data_list([data])
    return batch.to_dict()


def main() -> None:
    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)

    quant_config = QuantizationConfig(
        enable_ptq=True,
        enable_qat=False,
        scalar_int8=True,
        equiv_mddq=True,
        per_channel=True,
        fake_quant="minmax",
        norm_dtype="int8",
        eps=1e-8,
        norm_smooth_clip=5.0,
    )

    model = MACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticInteractionBlock,
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros((1, 1)),
        avg_num_neighbors=1.0,
        atomic_numbers=[1],
        correlation=2,
        gate=torch.nn.functional.silu,
        quant_config=quant_config,
    ).to(device)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], device=device, requires_grad=True
    )
    batch = build_batch(positions, num_elements=1)

    model.set_quantization(True)
    output = model(batch, compute_force=True)
    energy = output["energy"]
    forces = output["forces"]
    print(f"Quant energy: {energy.detach().cpu().numpy()}")
    print(f"Quant forces norm: {forces.norm(dim=1).detach().cpu().numpy()}")

    rotation = random_rotation(device)
    positions_rot = positions @ rotation.T
    batch_rot = build_batch(positions_rot, num_elements=1)
    output_rot = model(batch_rot, compute_force=True)
    energy_rot = output_rot["energy"]
    forces_rot = output_rot["forces"]

    energy_err = (energy - energy_rot).abs().max()
    forces_expected = forces @ rotation.T
    force_err = (forces_expected - forces_rot).norm(dim=1).max()
    print(f"Energy rotation error: {energy_err.item():.6f}")
    print(f"Force rotation error: {force_err.item():.6f}")

    model.set_quantization(False)
    output_fp = model(batch, compute_force=True)
    energy_fp = output_fp["energy"]
    print(
        f"FP32 energy delta vs quant: {(energy_fp - energy).abs().max().item():.6f}"
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
