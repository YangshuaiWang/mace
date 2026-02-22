import torch

from mace_efggm.grouping import group_scores, irreps_wise_groups, module_wise_groups


def test_module_groups_exist():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 1))
    groups = module_wise_groups(model.named_parameters())
    assert "other" in groups
    assert sum(len(v) for v in groups.values()) == len(list(model.named_parameters()))


def test_irreps_groups_partition():
    params = [("interactions.l0.weight", torch.nn.Parameter(torch.randn(2, 2))), ("readout.weight", torch.nn.Parameter(torch.randn(2, 1)))]
    groups = irreps_wise_groups(params)
    assert any(k.startswith("irreps") for k in groups)
    assert sum(len(v) for v in groups.values()) == len(params)


def test_group_scores():
    fisher = {"a": torch.ones(2), "b": torch.full((2,), 2.0)}
    groups = {"g1": ["a", "b"]}
    scores = group_scores(fisher, groups)
    assert scores["g1"] == 1.5
