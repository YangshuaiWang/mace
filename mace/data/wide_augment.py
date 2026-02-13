"""Wide-distribution data augmentation for IB-UQ gate regularization.

This module creates a perturbed copy of an in-memory training batch for the
"wide distribution gate-close" regularizer. The augmented batch is used only
for gate statistics (no labels required), keeping standard supervised loss
unchanged.

Supported transforms:
- coord_noise: add Gaussian noise to atomic positions.
- cell_scale: uniformly scale cell and positions by a random factor.
- mixed: randomly choose one of the above per step.

All tensors preserve input dtype/device. The augmentation only perturbs geometric
inputs and intentionally leaves backbone/message-passing code untouched.
"""

from __future__ import annotations

from typing import Dict

import torch


_COORD_NOISE_SIGMA = 0.03
_CELL_SCALE_MIN = 0.9
_CELL_SCALE_MAX = 1.1


def _resolve_aug_type(aug_type: str) -> str:
    if aug_type != "mixed":
        return aug_type
    idx = torch.randint(low=0, high=2, size=(1,)).item()
    return "coord_noise" if idx == 0 else "cell_scale"


def build_wide_batch(batch_dict: Dict[str, torch.Tensor], aug_type: str) -> Dict[str, torch.Tensor]:
    """Create a wide/OOD-like batch from an ID batch for IB-UQ gate-close loss.

    The resulting dictionary keeps exactly the same keys and batching/indexing as
    the input batch. For each atom/environment the model can compute gate values
    ``m(x)`` and latent codes while the regularizer pushes wide-distribution gates
    toward "closed" (smaller mean(m)).

    Args:
        batch_dict: Input batch dictionary from ``batch.to_dict()``.
        aug_type: One of ``none``, ``coord_noise``, ``cell_scale``, ``mixed``.

    Returns:
        A cloned dictionary with perturbed geometry tensors.
    """

    if aug_type == "none":
        return batch_dict

    wide = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch_dict.items()}
    resolved = _resolve_aug_type(aug_type)

    if resolved == "coord_noise":
        if "positions" in wide and torch.is_tensor(wide["positions"]):
            wide["positions"] = wide["positions"] + torch.randn_like(wide["positions"]) * _COORD_NOISE_SIGMA
        return wide

    if resolved == "cell_scale":
        if "positions" not in wide or not torch.is_tensor(wide["positions"]):
            return wide
        scale = torch.empty((), device=wide["positions"].device, dtype=wide["positions"].dtype).uniform_(
            _CELL_SCALE_MIN, _CELL_SCALE_MAX
        )
        wide["positions"] = wide["positions"] * scale
        if "cell" in wide and torch.is_tensor(wide["cell"]):
            wide["cell"] = wide["cell"] * scale
        return wide

    raise ValueError(f"Unsupported ib_uq_wide_aug '{aug_type}'")
