from scripts.aggregate_runs import aggregate_across_seeds, compute_ci_stats


def test_compute_ci_stats_known_values():
    stats = compute_ci_stats([1.0, 2.0, 3.0])
    assert stats["n"] == 3
    assert abs(stats["mean"] - 2.0) < 1e-12
    assert abs(stats["std"] - 1.0) < 1e-12
    # 95% t critical for dof=2 is 4.303 => ci ~ 2.484
    assert abs(stats["ci95"] - 2.4843382083) < 1e-6


def test_grouping_by_method_mode_and_hparams():
    rows = [
        {
            "method": "E-FGGM",
            "grouping_mode": "module",
            "seed": 0,
            "max_steps": 100,
            "effective_batch": 8,
            "total_updates": 100,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "irreps_grouping": "by_l",
            "new_force_rmse": 1.0,
            "old_delta_force_rmse": 2.0,
            "md_energy_drift_max_abs": 3.0,
        },
        {
            "method": "E-FGGM",
            "grouping_mode": "module",
            "seed": 1,
            "max_steps": 100,
            "effective_batch": 8,
            "total_updates": 100,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "irreps_grouping": "by_l",
            "new_force_rmse": 1.2,
            "old_delta_force_rmse": 2.2,
            "md_energy_drift_max_abs": 3.2,
        },
        {
            "method": "E-FGGM",
            "grouping_mode": "irreps",
            "seed": 0,
            "max_steps": 100,
            "effective_batch": 8,
            "total_updates": 100,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "irreps_grouping": "by_l",
            "new_force_rmse": 0.9,
            "old_delta_force_rmse": 2.4,
            "md_energy_drift_max_abs": 3.4,
        },
    ]

    grouped = aggregate_across_seeds(rows)
    assert len(grouped) == 2

    module_row = next(r for r in grouped if r["grouping_mode"] == "module")
    assert module_row["n"] == 2
    assert abs(module_row["new_force_rmse_mean"] - 1.1) < 1e-12

    irreps_row = next(r for r in grouped if r["grouping_mode"] == "irreps")
    assert irreps_row["n"] == 1
    assert irreps_row["new_force_rmse_ci95"] == 0.0
