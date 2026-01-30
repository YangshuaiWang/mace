import argparse
import sys

import numpy as np
import torch

if hasattr(torch, "serialization"):
    torch.serialization.add_safe_globals([slice])

from e3nn import o3

from mace.data.atomic_data import AtomicData
from mace.modules.blocks import (
    RealAgnosticInteractionBlock,
    RealAgnosticResidualInteractionBlock,
)
from mace.modules.models import MACE
from mace.modules.quantization import QuantizationConfig
from mace.tools.torch_geometric import Batch


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check that quantization changes MACE outputs."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-delta",
        type=float,
        default=1e-6,
        help="Minimum absolute delta to treat quantization as effective.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    output_quant = model(batch, compute_force=True)
    energy_quant = output_quant["energy"]
    forces_quant = output_quant["forces"]

    model.set_quantization(False)
    output_fp = model(batch, compute_force=True)
    energy_fp = output_fp["energy"]
    forces_fp = output_fp["forces"]

    energy_delta = (energy_quant - energy_fp).abs().max().item()
    force_delta = (forces_quant - forces_fp).abs().max().item()

    print(f"Energy delta (quant vs fp32): {energy_delta:.6f}")
    print(f"Force delta (quant vs fp32): {force_delta:.6f}")

    if energy_delta <= args.min_delta and force_delta <= args.min_delta:
        print(
            "Quantization did not introduce a measurable delta. "
            "Try reducing --min-delta or adjusting the model setup."
        )
        sys.exit(1)
    print("Quantization effect detected.")


if __name__ == "__main__":
    main()
