from __future__ import annotations

import copy
import dataclasses
import importlib.util
import io
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from mace.modules.embeddings import GenericJointEmbedding
from mace.modules.radial import RadialMLP

TORCHAO_AVAILABLE = importlib.util.find_spec("torchao") is not None


@dataclass
class QuantizationReportEntry:
    name: str
    module_type: str
    weight_dtype: str
    weight_int_repr_dtype: str
    qscheme: str
    scale: Optional[List[float]]
    zero_point: Optional[List[int]]
    numel: int
    float_bytes: int
    int8_bytes: int


def model_bytes(state_dict: Dict[str, torch.Tensor]) -> int:
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    return buffer.getbuffer().nbytes


def _disable_fake_quant(model: torch.nn.Module) -> None:
    if hasattr(model, "set_quantization"):
        model.set_quantization(False)


class QuantizedLinearBlock(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, qconfig: torch.ao.quantization.QConfig):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.linear = linear
        self.dequant = torch.ao.quantization.DeQuantStub()
        self.qconfig = qconfig
        self.linear.qconfig = qconfig

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.linear(x)
        x = self.dequant(x)
        return x


def _wrap_linear_layers(
    module: torch.nn.Module, qconfig: torch.ao.quantization.QConfig
) -> torch.nn.Module:
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, QuantizedLinearBlock(child, qconfig))
        else:
            _wrap_linear_layers(child, qconfig)
    return module


class QuantizedRadialMLP(torch.nn.Module):
    def __init__(self, float_module: RadialMLP, qconfig: torch.ao.quantization.QConfig):
        super().__init__()
        self.hs = list(float_module.hs)
        self.net = copy.deepcopy(float_module.net)
        _wrap_linear_layers(self.net, qconfig)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class QuantizedGenericJointEmbedding(torch.nn.Module):
    def __init__(
        self,
        float_module: GenericJointEmbedding,
        qconfig: torch.ao.quantization.QConfig,
        embedding_qconfig: torch.ao.quantization.QConfig,
    ):
        super().__init__()
        self.base_dim = float_module.base_dim
        self.out_dim = float_module.out_dim
        self.specs = dict(float_module.specs)
        self.embedders = torch.nn.ModuleDict()
        for name, embedder in float_module.embedders.items():
            if isinstance(embedder, torch.nn.Embedding):
                emb = copy.deepcopy(embedder)
                emb.qconfig = embedding_qconfig
                self.embedders[name] = emb
            else:
                embedder_copy = copy.deepcopy(embedder)
                _wrap_linear_layers(embedder_copy, qconfig)
                self.embedders[name] = embedder_copy
        self.project = copy.deepcopy(float_module.project)
        _wrap_linear_layers(self.project, qconfig)

    def forward(
        self, batch: torch.Tensor, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        embs = []
        for name, spec in self.specs.items():
            feat = features[name]
            if spec["per"] == "graph":
                feat = feat[batch].unsqueeze(-1)
            if spec["type"] == "categorical":
                feat = (feat + spec.get("offset", 0)).long().squeeze(-1)
            emb = self.embedders[name](feat)
            embs.append(emb)
        x = torch.cat(embs, dim=-1)
        return self.project(x)


def replace_quantizable_modules(
    model: torch.nn.Module,
    qconfig: torch.ao.quantization.QConfig,
    embedding_qconfig: torch.ao.quantization.QConfig,
) -> torch.nn.Module:
    for name, module in list(model.named_children()):
        if isinstance(module, RadialMLP):
            setattr(model, name, QuantizedRadialMLP(module, qconfig))
        elif isinstance(module, GenericJointEmbedding):
            setattr(
                model,
                name,
                QuantizedGenericJointEmbedding(module, qconfig, embedding_qconfig),
            )
        else:
            replace_quantizable_modules(module, qconfig, embedding_qconfig)
    return model


def _collect_quantized_modules(model: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    quantized = {}
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Embedding)):
            quantized[name] = module
    return quantized


def _apply_qconfig(
    model: torch.nn.Module,
    qconfig: torch.ao.quantization.QConfig,
    embedding_qconfig: torch.ao.quantization.QConfig,
) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            module.qconfig = embedding_qconfig
        elif isinstance(
            module,
            (
                torch.nn.Linear,
                torch.ao.quantization.QuantStub,
                torch.ao.quantization.DeQuantStub,
                QuantizedLinearBlock,
                QuantizedRadialMLP,
                QuantizedGenericJointEmbedding,
            ),
        ):
            module.qconfig = qconfig


def prepare_static_int8(
    model: torch.nn.Module, backend: str = "fbgemm"
) -> torch.nn.Module:
    torch.backends.quantized.engine = backend
    qconfig = torch.ao.quantization.get_default_qconfig(backend)
    embedding_qconfig = torch.ao.quantization.float_qparams_weight_only_qconfig
    model = replace_quantizable_modules(model, qconfig, embedding_qconfig)
    _apply_qconfig(model, qconfig, embedding_qconfig)
    _disable_fake_quant(model)
    model.eval()
    torch.ao.quantization.prepare(model, inplace=True)
    return model


def convert_static_int8(model: torch.nn.Module) -> torch.nn.Module:
    torch.ao.quantization.convert(model, inplace=True)
    return model


def quantize_with_torchao(model: torch.nn.Module) -> torch.nn.Module:
    if not TORCHAO_AVAILABLE:
        raise RuntimeError(
            "torchao is not available. Install torchao to use the int8 GEMM path."
        )
    from torchao.quantization import int8_weight_only, quantize_

    _disable_fake_quant(model)
    model.eval()
    quantize_(model, int8_weight_only())
    return model


def calibrate_model(
    model: torch.nn.Module,
    calibration_batches: Iterable[Dict[str, torch.Tensor]],
) -> None:
    with torch.no_grad():
        for batch in calibration_batches:
            _ = model(batch, compute_force=False)


def _weight_report(
    weight: torch.Tensor,
) -> Tuple[str, str, str, Optional[List[float]], Optional[List[int]], int]:
    if not weight.is_quantized:
        return (
            str(weight.dtype),
            str(weight.dtype),
            "fp",
            None,
            None,
            weight.numel(),
        )
    int_repr = weight.int_repr()
    if weight.qscheme() in {
        torch.per_channel_affine,
        torch.per_channel_symmetric,
    }:
        scale = weight.q_per_channel_scales().tolist()
        zero_point = weight.q_per_channel_zero_points().tolist()
    else:
        scale = [float(weight.q_scale())]
        zero_point = [int(weight.q_zero_point())]
    return (
        str(weight.dtype),
        str(int_repr.dtype),
        str(weight.qscheme()),
        scale,
        zero_point,
        weight.numel(),
    )


def build_quantization_report(
    float_model: torch.nn.Module,
    int8_model: torch.nn.Module,
) -> Dict[str, object]:
    quantized_modules = _collect_quantized_modules(int8_model)
    float_state = dict(float_model.named_modules())
    entries: List[QuantizationReportEntry] = []
    for name, module in quantized_modules.items():
        weight = module.weight()
        weight_dtype, int_dtype, qscheme, scale, zero_point, numel = _weight_report(
            weight
        )
        float_bytes = 0
        float_candidate = float_state.get(name)
        if float_candidate is None and name.endswith(".linear"):
            float_candidate = float_state.get(name[: -len(".linear")])
        if float_candidate is not None and hasattr(float_candidate, "weight"):
            float_weight = float_candidate.weight
            if isinstance(float_weight, torch.Tensor):
                float_bytes = float_weight.numel() * float_weight.element_size()
        int8_bytes = weight.int_repr().numel() * weight.int_repr().element_size()
        entries.append(
            QuantizationReportEntry(
                name=name,
                module_type=module.__class__.__name__,
                weight_dtype=weight_dtype,
                weight_int_repr_dtype=int_dtype,
                qscheme=qscheme,
                scale=scale,
                zero_point=zero_point,
                numel=numel,
                float_bytes=float_bytes,
                int8_bytes=int8_bytes,
            )
        )
    return {
        "modules": [dataclasses.asdict(entry) for entry in entries],
        "model_bytes": {
            "float": model_bytes(float_model.state_dict()),
            "int8": model_bytes(int8_model.state_dict()),
        },
    }


def build_int8_model(
    model: torch.nn.Module,
    calibration_batches: Iterable[Dict[str, torch.Tensor]],
    backend: str = "fbgemm",
    use_torchao: bool = False,
) -> Tuple[torch.nn.Module, Dict[str, object]]:
    float_model = copy.deepcopy(model).eval()
    _disable_fake_quant(float_model)
    if use_torchao:
        int8_model = quantize_with_torchao(copy.deepcopy(float_model))
    else:
        int8_model = prepare_static_int8(copy.deepcopy(float_model), backend=backend)
        calibrate_model(int8_model, calibration_batches)
        int8_model = convert_static_int8(int8_model)
    report = build_quantization_report(float_model, int8_model)
    return int8_model, report


def export_int8(
    model: torch.nn.Module,
    out_path: str,
    report_path: str,
    calibration_batches: Iterable[Dict[str, torch.Tensor]],
    backend: str = "fbgemm",
    use_torchao: bool = False,
) -> Dict[str, object]:
    int8_model, report = build_int8_model(
        model=model,
        calibration_batches=calibration_batches,
        backend=backend,
        use_torchao=use_torchao,
    )
    torch.save(int8_model.state_dict(), out_path)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return report
