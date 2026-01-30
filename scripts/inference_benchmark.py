import argparse
import copy
import statistics
import time
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from ase import build

from mace import data as mace_data
from mace.calculators.foundations_models import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools
from mace.tools.int8_quantization import (
    calibrate_model,
    convert_static_int8,
    model_bytes,
    prepare_static_int8,
)


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
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Run FP32 vs INT8 static quantized comparison (INT8 on CPU only).",
    )
    parser.add_argument(
        "--calib-iters",
        type=int,
        default=50,
        help="Calibration iterations for static PTQ INT8.",
    )
    parser.add_argument(
        "--cpu-baseline",
        action="store_true",
        help="Also run FP32 on CPU for INT8 speedup comparison.",
    )
    parser.add_argument(
        "--quant-backend",
        type=str,
        default="fbgemm",
        choices=("fbgemm", "qnnpack"),
        help="Quantized backend to use when --int8 is set.",
    )
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


def batch_repeat(batch: dict, iters: int) -> Iterable[dict]:
    for _ in range(iters):
        yield batch


def print_benchmark_summary(
    title: str,
    args: argparse.Namespace,
    device: torch.device,
    batch: dict,
    mean_latency: float,
    std_latency: float,
    dtype_label: Optional[str] = None,
) -> None:
    num_atoms = int(batch["positions"].shape[0])
    num_edges = int(batch["edge_index"].shape[1])
    steps_per_day = 86400 / mean_latency if mean_latency > 0 else float("inf")
    ns_per_day = 0.0864 / mean_latency if mean_latency > 0 else float("inf")

    print(title)
    dtype_label = dtype_label or args.dtype
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype_label}")
    print(f"Compile mode: {None if args.compile_mode == 'none' else args.compile_mode}")
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


def print_int8_validation(float_model: torch.nn.Module, int8_model: torch.nn.Module):
    quant_modules = [
        module
        for module in int8_model.modules()
        if isinstance(
            module, (torch.nn.quantized.Linear, torch.nn.quantized.Embedding)
        )
    ]
    module_types = [type(module).__name__ for module in quant_modules[:5]]
    print(f"INT8 module types (sample): {module_types or 'None found'}")

    weight_samples = []
    for name, module in int8_model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor):
            weight_samples.append(
                (
                    name,
                    str(weight.dtype),
                    weight.is_quantized,
                )
            )
        if len(weight_samples) >= 5:
            break
    print(f"INT8 weight dtype samples: {weight_samples or 'None found'}")

    state_dict = int8_model.state_dict()
    quant_keys = [
        key
        for key, value in state_dict.items()
        if isinstance(value, torch.Tensor)
        and (value.is_quantized or value.dtype in (torch.qint8, torch.quint8))
    ]
    packed_keys = [
        key
        for key, value in state_dict.items()
        if "packed" in key or not isinstance(value, torch.Tensor)
    ]
    print(
        f"INT8 state_dict qint8/quint8 keys (sample): {quant_keys[:5] or 'None found'}"
    )
    print(
        f"INT8 state_dict packed keys (sample): {packed_keys[:5] or 'None found'}"
    )

    float_bytes = model_bytes(float_model.state_dict())
    int8_bytes = model_bytes(int8_model.state_dict())
    print(
        "Model size (state_dict): "
        f"FP32 {float_bytes / 1e6:.2f} MB -> INT8 {int8_bytes / 1e6:.2f} MB"
    )


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

    torch.backends.quantized.engine = args.quant_backend

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
    print_benchmark_summary(
        "== MACE Inference Benchmark (FP32) ==",
        args,
        device,
        batch,
        mean_latency,
        std_latency,
    )

    if args.int8:
        if args.dtype != "float32":
            print("WARNING: INT8 quantization expects float32 weights; using float32.")
        if args.compute_force:
            print("WARNING: INT8 path does not support force computation; disabling.")
        print("INT8 CPU only.")

        cpu_device = torch.device("cpu")
        if args.threads is not None:
            torch.set_num_threads(args.threads)
        torch.backends.quantized.engine = args.quant_backend

        with torch_tools.default_dtype("float32"):
            float_model_cpu = load_model(
                args.model,
                "float32",
                cpu_device,
                compile_mode=None,
                fullgraph=args.fullgraph,
            )
            batch_cpu = create_batch(
                args.size,
                float_model_cpu,
                cpu_device,
                compute_force=False,
            )

        if args.cpu_baseline:
            cpu_latencies = run_benchmark(
                float_model_cpu,
                batch_cpu,
                compute_force=False,
                training=False,
                warmup=args.warmup,
                iters=args.iters,
                repeats=args.repeats,
                device=cpu_device,
            )
            cpu_mean, cpu_std = summarize(cpu_latencies)
            print_benchmark_summary(
                "== MACE Inference Benchmark (FP32 CPU) ==",
                args,
                cpu_device,
                batch_cpu,
                cpu_mean,
                cpu_std,
                dtype_label="float32",
            )

        int8_model = prepare_static_int8(
            copy.deepcopy(float_model_cpu), backend=args.quant_backend
        )
        calibrate_model(int8_model, batch_repeat(batch_cpu, args.calib_iters))
        int8_model = convert_static_int8(int8_model)

        print_int8_validation(float_model_cpu, int8_model)
        print(f"Quantized backend: {torch.backends.quantized.engine}")

        int8_latencies = run_benchmark(
            int8_model,
            batch_cpu,
            compute_force=False,
            training=False,
            warmup=args.warmup,
            iters=args.iters,
            repeats=args.repeats,
            device=cpu_device,
        )
        int8_mean, int8_std = summarize(int8_latencies)
        print_benchmark_summary(
            "== MACE Inference Benchmark (INT8 CPU) ==",
            args,
            cpu_device,
            batch_cpu,
            int8_mean,
            int8_std,
            dtype_label="int8 (static)",
        )
        if args.cpu_baseline:
            speedup_cpu = cpu_mean / int8_mean if int8_mean > 0 else float("inf")
            print(f"Speedup CPU (FP32/INT8): {speedup_cpu:.2f}x")


if __name__ == "__main__":
    main()
