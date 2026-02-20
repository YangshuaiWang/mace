"""Regression tests for IB-UQ gate-close training plumbing.

The tests are fully self-contained and build a tiny toy batch in-memory.
"""

from __future__ import annotations

import numpy as np
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data.wide_augment import build_wide_batch
from mace.tools import torch_geometric
import importlib

train_tools = importlib.import_module("mace.tools.train")


def _make_batch() -> torch_geometric.batch.Batch:
    rng = np.random.default_rng(7)
    z_table = tools.AtomicNumberTable([1, 8])
    configs = []
    for _ in range(2):
        n_atoms = 4
        configs.append(
            data.Configuration(
                atomic_numbers=rng.choice([1, 8], size=(n_atoms,)),
                positions=rng.normal(0.0, 1.0, size=(n_atoms, 3)),
                properties={
                    "energy": float(rng.normal()),
                    "forces": rng.normal(0.0, 0.1, size=(n_atoms, 3)),
                },
                property_weights={},
                weight=1.0,
            )
        )
    dataset = [
        data.AtomicData.from_config(cfg, z_table=z_table, cutoff=3.0) for cfg in configs
    ]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )
    return next(iter(loader))


def _make_model() -> modules.ScaleShiftMACE:
    table = tools.AtomicNumberTable([1, 8])
    return modules.ScaleShiftMACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(table.zs),
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(table.zs), dtype=float),
        avg_num_neighbors=4,
        atomic_numbers=table.zs,
        correlation=2,
        gate=torch.nn.functional.silu,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        ib_uq_enabled=True,
        ib_uq_latent_dim=4,
        ib_uq_deterministic=True,
        ib_uq_deterministic_zero_z0=True,
    ).to("cpu")


def _expected_supervised_loss(model, loss_fn, batch):
    pred = model(
        batch.to_dict(),
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    return float(loss_fn(pred=pred, ref=batch).detach())


def test_take_step_no_wide_batch_when_disabled(monkeypatch):
    torch.manual_seed(0)
    tools.set_default_dtype("float64")
    batch = _make_batch()
    model = _make_model()
    loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)

    call_count = {"n": 0}
    original_builder = train_tools.build_wide_batch

    def _counting_builder(batch_dict, aug_type):
        call_count["n"] += 1
        return original_builder(batch_dict, aug_type)

    monkeypatch.setattr(train_tools, "build_wide_batch", _counting_builder)

    expected = _expected_supervised_loss(model, loss_fn, batch)
    _, metrics = train_tools.take_step(
        model=model,
        loss_fn=loss_fn,
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=10.0,
        device=torch.device("cpu"),
        ib_uq_lambda=0.0,
        ib_uq_wide_aug="none",
        ib_uq_wide_frac=1.0,
    )

    assert call_count["n"] == 0
    assert abs(float(metrics["L_gate"])) == 0.0
    assert "train/gate_score_wide" not in metrics
    assert np.isclose(float(metrics["loss"]), expected, rtol=0.0, atol=1e-10)


def test_take_step_wide_batch_when_enabled(monkeypatch):
    torch.manual_seed(1)
    tools.set_default_dtype("float64")
    batch = _make_batch()
    model = _make_model()
    loss_fn = modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)

    call_count = {"n": 0}
    original_builder = train_tools.build_wide_batch

    def _counting_builder(batch_dict, aug_type):
        call_count["n"] += 1
        return original_builder(batch_dict, aug_type)

    monkeypatch.setattr(train_tools, "build_wide_batch", _counting_builder)

    _, metrics = train_tools.take_step(
        model=model,
        loss_fn=loss_fn,
        batch=batch,
        optimizer=optimizer,
        ema=None,
        output_args={"forces": True, "virials": False, "stress": False},
        max_grad_norm=10.0,
        device=torch.device("cpu"),
        ib_uq_lambda=0.2,
        ib_uq_wide_aug="coord_noise",
        ib_uq_wide_frac=1.0,
    )

    assert call_count["n"] == 1
    assert float(metrics["L_gate"]) > 0.0
    assert "train/gate_score_id" in metrics
    assert "train/gate_score_wide" in metrics


def test_build_wide_batch_is_non_inplace_and_preserves_tensor_type():
    torch.manual_seed(2)
    tools.set_default_dtype("float64")
    batch = _make_batch().to_dict()

    positions_before = batch["positions"].clone()
    wide_batch = build_wide_batch(batch, aug_type="coord_noise")

    assert torch.equal(batch["positions"], positions_before)
    assert wide_batch["positions"].dtype == batch["positions"].dtype
    assert wide_batch["positions"].device == batch["positions"].device


def test_deterministic_gate_outputs_are_repeatable():
    torch.manual_seed(3)
    tools.set_default_dtype("float64")
    model = _make_model()
    batch_dict = _make_batch().to_dict()

    out1 = model(batch_dict, training=False)
    out2 = model(batch_dict, training=False)

    assert torch.allclose(out1["ib_uq"]["gate_score"], out2["ib_uq"]["gate_score"], rtol=0.0, atol=0.0)
    assert torch.allclose(out1["ib_uq"]["m_mean"], out2["ib_uq"]["m_mean"], rtol=0.0, atol=0.0)
