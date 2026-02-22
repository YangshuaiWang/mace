"""Builds binary masks from Fisher group scores."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List


@dataclass
class AlphaSchedule:
    alpha_start: float
    alpha_end: float
    total_steps: int
    schedule_type: str = "linear"

    def value(self, step: int) -> float:
        if self.total_steps <= 1:
            return self.alpha_end
        t = min(max(step, 0), self.total_steps - 1) / (self.total_steps - 1)
        if self.schedule_type == "linear":
            s = t
        elif self.schedule_type == "cosine":
            s = 0.5 * (1.0 - math.cos(math.pi * t))
        elif self.schedule_type == "step":
            s = 0.0 if step < (self.total_steps - 1) else 1.0
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        return self.alpha_start + s * (self.alpha_end - self.alpha_start)


def select_top_groups(group_scores: Dict[str, float], alpha: float) -> List[str]:
    alpha = min(max(alpha, 0.0), 1.0)
    if alpha == 0.0 or not group_scores:
        return []
    n_keep = max(1, int(round(alpha * len(group_scores))))
    ordered = sorted(group_scores.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in ordered[:n_keep]]


def build_parameter_mask(
    groups: Dict[str, List[str]], kept_groups: Iterable[str]
) -> Dict[str, float]:
    keep = set(kept_groups)
    mask: Dict[str, float] = {}
    for group_name, params in groups.items():
        v = 1.0 if group_name in keep else 0.0
        for p in params:
            mask[p] = v
    return mask
