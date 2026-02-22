import torch

from mace_efggm.grouping import group_params_by_irreps


class DummyIrrepsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block0 = torch.nn.Linear(4, 4)
        self.block1 = torch.nn.Linear(4, 2)
        self.misc = torch.nn.Parameter(torch.randn(3))
        self.block0.irreps_out = "2x0e"
        self.block1.irreps_out = "1x1o"


def test_group_params_by_irreps_assigns_l_groups():
    model = DummyIrrepsModule()
    groups = group_params_by_irreps(model)
    assert "irrep_l0" in groups
    assert "irrep_l1" in groups


def test_group_params_by_irreps_assigns_unknown_and_never_raises():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU())
    groups = group_params_by_irreps(model)
    assert "irrep_unknown" in groups
    total = sum(len(v) for v in groups.values())
    assert total == len(list(model.parameters()))
