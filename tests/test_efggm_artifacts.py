import torch

from scripts.finetune_efggm import compute_mask_coverage


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(2, 2, bias=False)
        self.b = torch.nn.Linear(2, 1, bias=False)


def test_mask_coverage_toy_model():
    model = Toy()
    groups = {
        "a": ["a.weight"],
        "b": ["b.weight"],
    }
    coverage = compute_mask_coverage(model, groups, {"a"}, grouping="module")
    assert coverage["num_groups_total"] == 2
    assert coverage["num_groups_kept"] == 1
    assert coverage["fraction_groups_kept"] == 0.5
    assert coverage["num_params_total"] == 6
    assert coverage["num_params_trainable"] == 4
    assert coverage["fraction_params_trainable"] == 4 / 6
