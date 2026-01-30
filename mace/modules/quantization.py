from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import torch
from e3nn import o3


@dataclasses.dataclass
class QuantizationConfig:
    enable_ptq: bool = False
    enable_qat: bool = False
    scalar_int8: bool = True
    equiv_mddq: bool = True
    per_channel: bool = True
    fake_quant: str = "minmax"
    norm_dtype: str = "int8"  # "int8", "fp16", "fp32"
    eps: float = 1e-8
    norm_smooth_clip: Optional[float] = None
    active: bool = True

    def enabled(self) -> bool:
        return self.active and (self.enable_ptq or self.enable_qat)


class MinMaxObserver(torch.nn.Module):
    def __init__(
        self,
        num_channels: Optional[int],
        per_channel: bool,
        channel_axis: int = 1,
    ) -> None:
        super().__init__()
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        if per_channel:
            if num_channels is None:
                raise ValueError("num_channels is required for per-channel observers.")
            self.register_buffer("min_val", torch.zeros(num_channels))
            self.register_buffer("max_val", torch.zeros(num_channels))
        else:
            self.register_buffer("min_val", torch.tensor(0.0))
            self.register_buffer("max_val", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    def update(self, x: torch.Tensor) -> None:
        if self.per_channel:
            reduce_dims = [d for d in range(x.ndim) if d != self.channel_axis]
            x_min = x.amin(dim=reduce_dims)
            x_max = x.amax(dim=reduce_dims)
        else:
            x_min = x.min()
            x_max = x.max()
        if not self.initialized:
            self.min_val.copy_(x_min)
            self.max_val.copy_(x_max)
            self.initialized.fill_(True)
        else:
            self.min_val.copy_(torch.minimum(self.min_val, x_min))
            self.max_val.copy_(torch.maximum(self.max_val, x_max))

    def get_qparams(
        self, qmin: int, qmax: int, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_val = torch.maximum(self.max_val, self.min_val + eps)
        scale = (max_val - self.min_val) / float(qmax - qmin)
        scale = torch.clamp(scale, min=eps)
        zero_point = qmin - self.min_val / scale
        zero_point = zero_point.round().clamp(qmin, qmax).to(torch.int32)
        return scale, zero_point


def _fake_quantize(
    x: torch.Tensor,
    observer: MinMaxObserver,
    per_channel: bool,
    channel_axis: int,
    qmin: int,
    qmax: int,
    eps: float,
) -> torch.Tensor:
    if not observer.initialized:
        return x
    scale, zero_point = observer.get_qparams(qmin=qmin, qmax=qmax, eps=eps)
    scale = scale.detach()
    zero_point = zero_point.detach()
    if per_channel:
        return torch.fake_quantize_per_channel_affine(
            x, scale, zero_point, channel_axis, qmin, qmax
        )
    return torch.fake_quantize_per_tensor_affine(
        x, float(scale.item()), int(zero_point.item()), qmin, qmax
    )


def _smooth_clip(norm: torch.Tensor, alpha: Optional[float]) -> torch.Tensor:
    if alpha is None:
        return norm
    return alpha * torch.tanh(norm / alpha)


def _use_fake_quant(config: QuantizationConfig) -> bool:
    return config.fake_quant.lower() not in {"none", "disabled"}


class ScalarQuantizer(torch.nn.Module):
    def __init__(
        self,
        num_channels: int,
        config: Optional[QuantizationConfig],
        channel_axis: int = 1,
        qmin: int = -128,
        qmax: int = 127,
    ) -> None:
        super().__init__()
        self.config = config
        self.qmin = qmin
        self.qmax = qmax
        self.channel_axis = channel_axis
        self.observer = MinMaxObserver(
            num_channels=num_channels,
            per_channel=bool(config.per_channel) if config else False,
            channel_axis=channel_axis,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config is None or not self.config.enabled() or not self.config.scalar_int8:
            return x
        if not _use_fake_quant(self.config):
            return x
        if self.config.enable_ptq or (self.config.enable_qat and self.training):
            self.observer.update(x)
        return _fake_quantize(
            x,
            observer=self.observer,
            per_channel=self.observer.per_channel,
            channel_axis=self.channel_axis,
            qmin=self.qmin,
            qmax=self.qmax,
            eps=self.config.eps,
        )


class IrrepsQuantizer(torch.nn.Module):
    def __init__(
        self,
        irreps: o3.Irreps,
        config: Optional[QuantizationConfig],
    ) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.config = config
        self.slices: List[Tuple[int, int, int]] = []
        offset = 0
        for mul, ir in self.irreps:
            dim = ir.dim
            self.slices.append((offset, mul, dim))
            offset += mul * dim

        self.scalar_observers = torch.nn.ModuleList()
        self.norm_observers = torch.nn.ModuleList()
        for mul, ir in self.irreps:
            if ir.l == 0:
                self.scalar_observers.append(
                    MinMaxObserver(
                        num_channels=mul,
                        per_channel=bool(config.per_channel) if config else False,
                        channel_axis=1,
                    )
                )
                self.norm_observers.append(MinMaxObserver(None, False))
            else:
                self.scalar_observers.append(MinMaxObserver(None, False))
                self.norm_observers.append(
                    MinMaxObserver(
                        num_channels=mul,
                        per_channel=bool(config.per_channel) if config else False,
                        channel_axis=1,
                    )
                )

    def _quantize_scalar_block(
        self, block: torch.Tensor, observer: MinMaxObserver
    ) -> torch.Tensor:
        if self.config is None or not self.config.scalar_int8:
            return block
        if not _use_fake_quant(self.config):
            return block
        if self.config.enable_ptq or (self.config.enable_qat and self.training):
            observer.update(block)
        return _fake_quantize(
            block,
            observer=observer,
            per_channel=observer.per_channel,
            channel_axis=1,
            qmin=-128,
            qmax=127,
            eps=self.config.eps,
        )

    def _quantize_norm(
        self, norm: torch.Tensor, observer: MinMaxObserver
    ) -> torch.Tensor:
        if self.config is None:
            return norm
        if self.config.norm_dtype.lower() in {"fp16", "float16"}:
            return norm.to(torch.float16)
        if self.config.norm_dtype.lower() in {"fp32", "float32"}:
            return norm
        if not _use_fake_quant(self.config):
            return norm
        if self.config.enable_ptq or (self.config.enable_qat and self.training):
            observer.update(norm)
        return _fake_quantize(
            norm,
            observer=observer,
            per_channel=observer.per_channel,
            channel_axis=1,
            qmin=0,
            qmax=255,
            eps=self.config.eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config is None or not self.config.enabled():
            return x
        if x.ndim == 2:
            return self._forward_flat(x)
        if x.ndim == 3:
            return self._forward_mul_ir(x)
        return x

    def _forward_flat(self, x: torch.Tensor) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        for (offset, mul, dim), (mul_ir, ir), s_obs, n_obs in zip(
            self.slices, self.irreps, self.scalar_observers, self.norm_observers
        ):
            block = x[:, offset : offset + mul * dim].reshape(x.shape[0], mul, dim)
            if ir.l == 0:
                block = self._quantize_scalar_block(block.squeeze(-1), s_obs).unsqueeze(
                    -1
                )
            elif self.config.equiv_mddq:
                norm = torch.linalg.vector_norm(block, dim=-1)
                norm = _smooth_clip(norm, self.config.norm_smooth_clip)
                norm_q = self._quantize_norm(norm, n_obs)
                direction = block / (norm.unsqueeze(-1) + self.config.eps)
                block = norm_q.unsqueeze(-1) * direction
            outputs.append(block.reshape(x.shape[0], mul * dim))
        return torch.cat(outputs, dim=-1)

    def _forward_mul_ir(self, x: torch.Tensor) -> torch.Tensor:
        outputs: List[torch.Tensor] = []
        offset = 0
        for (mul, ir), s_obs, n_obs in zip(
            self.irreps, self.scalar_observers, self.norm_observers
        ):
            dim = ir.dim
            block = x[:, :, offset : offset + dim]
            offset += dim
            if ir.l == 0:
                block = self._quantize_scalar_block(block.squeeze(-1), s_obs).unsqueeze(
                    -1
                )
            elif self.config.equiv_mddq:
                norm = torch.linalg.vector_norm(block, dim=-1)
                norm = _smooth_clip(norm, self.config.norm_smooth_clip)
                norm_q = self._quantize_norm(norm, n_obs)
                direction = block / (norm.unsqueeze(-1) + self.config.eps)
                block = norm_q.unsqueeze(-1) * direction
            outputs.append(block)
        return torch.cat(outputs, dim=-1)


def set_quantization_active(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "config") and isinstance(module.config, QuantizationConfig):
            module.config.active = enabled
