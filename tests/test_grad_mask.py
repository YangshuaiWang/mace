import torch

from mace_efggm.grad_mask import GradientMasker


def test_gradient_mask_apply():
    p1 = torch.nn.Parameter(torch.tensor([1.0], requires_grad=True))
    p2 = torch.nn.Parameter(torch.tensor([2.0], requires_grad=True))
    named = [("p1", p1), ("p2", p2)]
    masker = GradientMasker(named)

    (p1 + p2).backward()
    masker.update_mask({"p1": 0.0, "p2": 1.0})
    masker.apply()

    assert p1.grad.item() == 0.0
    assert p2.grad.item() == 1.0
