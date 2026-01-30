import argparse
import io
import time

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
    if num_atoms < 2:
        edge_index = torch.empty((2, 0), device=positions.device, dtype=torch.long)
    else:
        senders = torch.arange(num_atoms - 1, device=positions.device)
        receivers = senders + 1
        edge_index = torch.stack(
            [
                torch.cat([senders, receivers]),
                torch.cat([receivers, senders]),
            ],
            dim=0,
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


def model_bytes(state_dict: dict) -> int:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes


def timed_forward(
    model: torch.nn.Module,
    batch: dict,
    compute_force: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch, compute_force=compute_force)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(batch, compute_force=compute_force)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
    return (end - start) / iters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark quantized vs FP32 MACE inference."
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-atoms", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--compute-force", action="store_true")
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

    positions = torch.randn(args.num_atoms, 3, device=device, requires_grad=False)
    batch = build_batch(positions, num_elements=1)

    model.set_quantization(False)
    fp_bytes = model_bytes(model.state_dict())
    fp_time = timed_forward(
        model,
        batch,
        compute_force=args.compute_force,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    model.set_quantization(True)
    quant_bytes = model_bytes(model.state_dict())
    quant_time = timed_forward(
        model,
        batch,
        compute_force=args.compute_force,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
    )

    print(f"FP32 state dict size: {fp_bytes / 1e6:.2f} MB")
    print(f"Quant state dict size: {quant_bytes / 1e6:.2f} MB")
    print(f"FP32 avg latency: {fp_time * 1e3:.3f} ms")
    print(f"Quant avg latency: {quant_time * 1e3:.3f} ms")
    if fp_time > 0:
        speedup = fp_time / quant_time if quant_time > 0 else float("inf")
        print(f"Speedup (fp32/quant): {speedup:.2f}x")


if __name__ == "__main__":
    main()
