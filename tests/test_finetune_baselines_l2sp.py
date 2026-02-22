import torch

from scripts.finetune_baselines import compute_l2sp_anchor_loss


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1.0]))
        self.b = torch.nn.Parameter(torch.tensor([0.0]))


def test_l2sp_anchor_loss_nonzero_when_params_drift():
    model = ToyModel()
    for p in model.parameters():
        p.requires_grad = True

    theta0 = {name: p.detach().clone() for name, p in model.named_parameters()}

    with torch.no_grad():
        model.w.add_(2.0)

    l2sp = compute_l2sp_anchor_loss(model, theta0, l2sp_lambda=0.5)
    assert float(l2sp.item()) > 0.0
