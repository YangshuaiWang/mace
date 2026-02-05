import pytest
import torch

from mace.tools import scripts_utils


if not hasattr(torch.optim, "Muon"):
    pytest.skip("Muon optimizer not available", allow_module_level=True)


def test_get_optimizer_muon():
    model = torch.nn.Linear(4, 2, bias=False)
    args = type("Args", (), {"optimizer": "muon", "beta": 0.9})()
    param_options = {
        "params": model.parameters(),
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "amsgrad": False,
        "betas": (0.9, 0.999),
    }
    optimizer = scripts_utils.get_optimizer(args, param_options)
    assert isinstance(optimizer, torch.optim.Muon)

    data = torch.randn(2, 4)
    loss = model(data).sum()
    loss.backward()
    optimizer.step()
