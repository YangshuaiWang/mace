import argparse
import json
import pathlib
from typing import Dict, List

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
from mace.tools.int8_quantization import (
    TORCHAO_AVAILABLE,
    build_int8_model,
    export_int8,
)
from mace.tools.torch_geometric import Batch


def random_rotation(device: torch.device) -> torch.Tensor:
    matrix = torch.randn(3, 3, device=device)
    q, _ = torch.linalg.qr(matrix)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def build_batch(positions: torch.Tensor, num_elements: int) -> Dict[str, torch.Tensor]:
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
    parser = argparse.ArgumentParser(description="Real INT8 quantization export for MACE")
    parser.add_argument("--backend", type=str, default="fbgemm")
    parser.add_argument("--use-torchao", action="store_true")
    parser.add_argument("--out-path", type=pathlib.Path, default=pathlib.Path("mace_int8.pt"))
    parser.add_argument(
        "--report-path", type=pathlib.Path, default=pathlib.Path("mace_int8_report.json")
    )
    parser.add_argument("--calibration-iters", type=int, default=8)
    return parser.parse_args()


def list_quantized_modules(model: torch.nn.Module) -> List[str]:
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Embedding)):
            names.append(f"{name}: {module.__class__.__name__}")
    return names


def model_has_int8_weights(model: torch.nn.Module) -> bool:
    for module in model.modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Embedding)):
            weight = module.weight()
            if weight.is_quantized:
                int_repr = weight.int_repr()
                if int_repr.dtype in (torch.int8, torch.uint8):
                    return True
    return False


def main() -> None:
    args = parse_args()
    if args.use_torchao and not TORCHAO_AVAILABLE:
        raise RuntimeError("torchao not available; install it or drop --use-torchao.")

    device = torch.device("cpu")
    torch.set_default_dtype(torch.float32)

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
        quant_config=None,
    ).to(device)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], device=device, requires_grad=True
    )
    calibration_batches = [
        build_batch(positions + 0.05 * torch.randn_like(positions), num_elements=1)
        for _ in range(args.calibration_iters)
    ]

    report = export_int8(
        model=model,
        out_path=str(args.out_path),
        report_path=str(args.report_path),
        calibration_batches=calibration_batches,
        backend=args.backend,
        use_torchao=args.use_torchao,
    )
    print(json.dumps(report, indent=2))

    quantized_model, _ = build_int8_model(
        model=model,
        calibration_batches=calibration_batches,
        backend=args.backend,
        use_torchao=args.use_torchao,
    )
    quantized_model.eval()

    print("Quantized modules:")
    for name in list_quantized_modules(quantized_model):
        print(f"  - {name}")
    has_int8 = model_has_int8_weights(quantized_model)
    print(f"State dict contains INT8 weights: {has_int8}")

    batch = build_batch(positions, num_elements=1)
    with torch.no_grad():
        output_quant = quantized_model(batch, compute_force=False)
    print(f"Quantized energy: {output_quant['energy'].detach().cpu().numpy()}")

    output_fp = model(batch, compute_force=True)
    energy_fp = output_fp["energy"]
    forces_fp = output_fp["forces"]
    rotation = random_rotation(device)
    positions_rot = positions @ rotation.T
    batch_rot = build_batch(positions_rot, num_elements=1)
    output_rot = model(batch_rot, compute_force=True)
    energy_rot = output_rot["energy"]
    forces_rot = output_rot["forces"]

    energy_err = (energy_fp - energy_rot).abs().max()
    forces_expected = forces_fp @ rotation.T
    force_err = (forces_expected - forces_rot).norm(dim=1).max()
    print(f"Energy rotation error (fp32): {energy_err.item():.6f}")
    print(f"Force rotation error (fp32): {force_err.item():.6f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    main()
