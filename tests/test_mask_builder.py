from mace_efggm.mask_builder import AlphaSchedule, build_parameter_mask, select_top_groups


def test_select_top_groups():
    scores = {"a": 1.0, "b": 0.2, "c": 0.1}
    kept = select_top_groups(scores, alpha=0.34)
    assert kept == ["a"]


def test_parameter_mask():
    groups = {"g1": ["p1", "p2"], "g2": ["p3"]}
    mask = build_parameter_mask(groups, ["g2"])
    assert mask["p1"] == 0.0
    assert mask["p3"] == 1.0


def test_alpha_schedule():
    sched = AlphaSchedule(alpha_start=1.0, alpha_end=0.0, total_steps=5)
    assert sched.value(0) == 1.0
    assert sched.value(4) == 0.0
