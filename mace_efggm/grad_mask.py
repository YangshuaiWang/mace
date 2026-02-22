"""Gradient masking utilities."""

from __future__ import annotations

from typing import Dict, Iterable, List


class GradientMasker:
    """Applies per-parameter scalar mask in-place to gradients."""

    def __init__(self, named_parameters: Iterable):
        self._params = {name: p for name, p in named_parameters}
        self._mask: Dict[str, float] = {name: 1.0 for name in self._params}

    def update_mask(self, mask: Dict[str, float]) -> None:
        for name in self._params:
            self._mask[name] = float(mask.get(name, 1.0))

    def apply(self) -> None:
        for name, param in self._params.items():
            if param.grad is not None:
                param.grad.mul_(self._mask.get(name, 1.0))


class MaskedOptimizerWrapper:
    """Thin optimizer wrapper to apply masks pre-step with minimal invasion."""

    def __init__(self, optimizer, masker: GradientMasker):
        self.optimizer = optimizer
        self.masker = masker

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def step(self, closure=None):
        self.masker.apply()
        return self.optimizer.step(closure)

    @property
    def param_groups(self):
        return self.optimizer.param_groups
