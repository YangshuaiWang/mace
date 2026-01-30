import argparse
import statistics
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from ase import build

from mace import data as mace_data
from mace.calculators.foundations_models import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MACE inference speed.")
    parser.add_argument("--model", type=str, default="medium")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--compile-mode", type=str, default="none")
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compute-force", action="store_true")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set torch intra-op threads (CPU only).",
    )
    parser.add_argument("--fullgraph", action="store_true")
    return parser.parse_args()


def load_model(
    model_name: str,
    dtype: str,
    device: torch.device,
    compile_mode: Optional[str],
    fullgraph: bool,
) -> torch.nn.Module:
    calc = mace_mp(
        model=model_name,
        default_dtype=dtype,
        device=str(device),
        compile_mode=compile_mode,
        fullgraph=fullgraph,
    )
    return calc.models[0].to(device)


def create_batch(
    size: int, model: torch.nn.Module, device: torch.device, compute_force: bool
) -> dict:
    cutoff = model.r_max.item()
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    config = mace_data.config_from_atoms(atoms)
    dataset = [mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch.to(device)
    if compute_force:
        batch.positions.requires_grad_(True)
    return batch.to_dict()


def timed_forward(
    model: torch.nn.Module,
    batch: dict,
    compute_force: bool,
    training: bool,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    with torch.set_grad_enabled(compute_force or training):
        for _ in range(warmup):
            _ = model(batch, training=training, compute_force=compute_force)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(batch, training=training, compute_force=compute_force)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        end = time.perf_counter()
    return (end - start) / iters


def run_benchmark(
    model: torch.nn.Module,
    batch: dict,
    compute_force: bool,
    training: bool,
    warmup: int,
    iters: int,
    repeats: int,
    device: torch.device,
) -> List[float]:
    latencies = []
    for _ in range(repeats):
        latencies.append(
            timed_forward(
                model,
                batch,
                compute_force=compute_force,
                training=training,
                warmup=warmup,
                iters=iters,
                device=device,
            )
        )
    return latencies


def summarize(latencies: List[float]) -> Tuple[float, float]:
    if len(latencies) == 1:
        return latencies[0], 0.0
    return statistics.mean(latencies), statistics.stdev(latencies)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")

    if args.threads is not None and device.type == "cpu":
        torch.set_num_threads(args.threads)

    compile_mode = None if args.compile_mode == "none" else args.compile_mode

    with torch_tools.default_dtype(args.dtype):
        model = load_model(
            args.model,
            args.dtype,
            device,
            compile_mode,
            fullgraph=args.fullgraph,
        )
        batch = create_batch(args.size, model, device, args.compute_force)
        latencies = run_benchmark(
            model,
            batch,
            compute_force=args.compute_force,
            training=compile_mode is not None,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            device=device,
        )

    mean_latency, std_latency = summarize(latencies)
    num_atoms = int(batch["positions"].shape[0])
    num_edges = int(batch["edge_index"].shape[1])
    steps_per_day = 86400 / mean_latency if mean_latency > 0 else float("inf")
    ns_per_day = 0.0864 / mean_latency if mean_latency > 0 else float("inf")

    print("== MACE Inference Benchmark ==")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Dtype: {args.dtype}")
    print(f"Compile mode: {compile_mode}")
    print(f"Fullgraph: {args.fullgraph}")
    if args.threads is not None and device.type == "cpu":
        print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Num atoms: {num_atoms}")
    print(f"Num edges: {num_edges}")
    print(f"Iters per repeat: {args.iters}")
    print(f"Repeats: {args.repeats}")
    print(f"Compute forces: {args.compute_force}")
    print(f"Avg latency: {mean_latency * 1e3:.3f} ms ± {std_latency * 1e3:.3f} ms")
    print(f"Steps per day: {steps_per_day:.0f}")
    print(f"ns/day (1 fs/step): {ns_per_day:.2f}")


if __name__ == "__main__":
    main()
