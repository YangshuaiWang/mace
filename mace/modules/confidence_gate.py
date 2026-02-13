from __future__ import annotations

from typing import Callable, Union

import torch
from torch import nn


ActivationLike = Union[str, Callable[[torch.Tensor], torch.Tensor], nn.Module]


class _LambdaModule(nn.Module):
    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)


def _resolve_activation(activation: ActivationLike) -> nn.Module:
    if isinstance(activation, nn.Module):
        return activation
    if callable(activation) and not isinstance(activation, str):
        return _LambdaModule(activation)
    if isinstance(activation, str):
        key = activation.lower()
        if key == "silu":
            return nn.SiLU()
        if key == "relu":
            return nn.ReLU()
        if key == "gelu":
            return nn.GELU()
        if key == "tanh":
            return nn.Tanh()
        if key in {"identity", "none"}:
            return nn.Identity()
    raise ValueError(
        "Unsupported activation. Expected one of "
        "{'silu', 'relu', 'gelu', 'tanh', 'identity'} or an nn.Module."
    )


class ConfidenceGate(nn.Module):
    r"""IB-UQ confidence gate for invariant (`L=0`) atom-wise features.

    Given invariant features ``h0`` for each atom/environment, this module predicts:

    - ``m(h0) in [0, 1]^{d_z}``: a confidence gate,
    - ``z_bar(h0) in R^{d_z}``: a learned latent code.

    These are used in IB-UQ with

    .. math::
        z = \mathrm{diag}(m(h0))\, z_{\text{bar}}(h0)
            + \mathrm{diag}(1 - m(h0))\, z_0,
        \quad z_0 \sim \mathcal{N}(0, I).

    Interpretation:
    - ``m_j \approx 1`` means dimension ``j`` relies on learned in-distribution code
      ``z_bar_j``;
    - ``m_j \approx 0`` means dimension ``j`` falls back to noise prior ``z0_j``,
      indicating low confidence / potential OOD behavior.

    Notes:
    - Input/outputs are per-atom with shape ``[n_atoms, *]``.
    - The returned ``m`` is guaranteed in ``[0, 1]`` via sigmoid (or
      ``exp(logsigmoid(.))`` when ``use_logsigmoid=True``).
    """

    def __init__(
        self,
        d0: int,
        dz: int,
        hidden: int = 128,
        layernorm: bool = True,
        activation: ActivationLike = "silu",
        use_logsigmoid: bool = False,
    ) -> None:
        super().__init__()
        if d0 <= 0 or dz <= 0 or hidden <= 0:
            raise ValueError("d0, dz and hidden must all be positive integers.")

        norm = nn.LayerNorm(hidden) if layernorm else nn.Identity()

        self.encoder = nn.Sequential(
            nn.Linear(d0, hidden),
            norm,
            _resolve_activation(activation),
            nn.Linear(hidden, hidden),
            _resolve_activation(activation),
        )
        self.gate_head = nn.Linear(hidden, dz)
        self.code_head = nn.Linear(hidden, dz)
        self.use_logsigmoid = use_logsigmoid

    def forward(self, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h0.ndim != 2:
            raise ValueError(f"Expected h0 shape [n_atoms, d0], got {tuple(h0.shape)}")

        x = self.encoder(h0)
        logits = self.gate_head(x)
        if self.use_logsigmoid:
            m = torch.exp(torch.nn.functional.logsigmoid(logits))
        else:
            m = torch.sigmoid(logits)
        z_bar = self.code_head(x)
        return m, z_bar
