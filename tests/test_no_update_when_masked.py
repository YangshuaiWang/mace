import torch

from mace_efggm.grad_mask import GradientMasker, MaskedOptimizerWrapper


class TinyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.group_a = torch.nn.Linear(4, 4, bias=False)
        self.group_b = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.group_a(x) + self.group_b(x)


def test_masked_parameters_do_not_update():
    torch.manual_seed(0)
    model = TinyModule()
    x = torch.randn(8, 4)
    target = torch.randn(8, 4)

    initial = {name: p.detach().clone() for name, p in model.named_parameters()}

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    masker = GradientMasker(model.named_parameters())
    masker.update_mask({
        "group_a.weight": 0.0,
        "group_b.weight": 1.0,
    })
    wrapped = MaskedOptimizerWrapper(optimizer, masker)

    wrapped.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    wrapped.step()

    assert torch.equal(model.group_a.weight, initial["group_a.weight"])
    assert not torch.equal(model.group_b.weight, initial["group_b.weight"])
