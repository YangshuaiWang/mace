"""Fisher-like importance estimation using new-domain gradients only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch


@dataclass
class FisherEMAConfig:
    beta: float = 0.95
    eps: float = 1e-12


class FisherEMA:
    """Tracks diagonal Fisher-like score with gradient-squared EMA.

    Fi <- beta * Fi + (1-beta) * (grad_i ** 2)
    """

    def __init__(self, named_parameters: Iterable, config: FisherEMAConfig | None = None):
        self.config = config or FisherEMAConfig()
        self.state: Dict[str, torch.Tensor] = {}
        for name, param in named_parameters:
            if param.requires_grad:
                self.state[name] = torch.zeros_like(param, memory_format=torch.preserve_format)

    @torch.no_grad()
    def update_from_model(self, named_parameters: Iterable):
        beta = self.config.beta
        for name, param in named_parameters:
            if not param.requires_grad or param.grad is None or name not in self.state:
                continue
            self.state[name].mul_(beta).addcmul_(param.grad, param.grad, value=1.0 - beta)

    def mean_scores(self) -> Dict[str, float]:
        return {name: float(value.mean().item()) for name, value in self.state.items()}

    def tensors(self) -> Dict[str, torch.Tensor]:
        return self.state
