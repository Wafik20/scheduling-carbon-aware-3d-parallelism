import pytest

from evaluate_carbon_policy import (
    _normalize_strategy_rows,
    static_policy_label,
    strategy_label as evaluation_label,
)
from carbon_aware_3d.carbon_aware_env import strategy_label as environment_label


@pytest.mark.parametrize("label", [evaluation_label, environment_label, static_policy_label])
def test_placement_and_micro_batch_size_distinguish_strategies(label):
    strategies = [
        dict(tp=2, pp=2, dp=2, micro_batch_size=mbs, placement_id=placement)
        for placement in ("tp-dp-pp", "pp-dp-tp")
        for mbs in (1, 2)
    ]
    labels = [label(strategy) for strategy in strategies]
    assert len(set(labels)) == len(strategies)
    for strategy, text in zip(strategies, labels):
        assert f"placement_id={strategy['placement_id']}" in text


@pytest.mark.parametrize("label,prefix", [
    (evaluation_label, ""), (environment_label, ""), (static_policy_label, "static"),
])
@pytest.mark.parametrize("extras,suffix", [
    ({}, ""), ({"dp": 2}, ",dp=2"),
    ({"dp": 2, "micro_batch_size": 4}, ",dp=2,mbs=4"),
])
def test_legacy_labels_unchanged(label, prefix, extras, suffix):
    assert label(dict(tp=2, pp=2, **extras)) == f"{prefix}(tp=2,pp=2{suffix})"


def test_plot_label_keeps_placement_when_dp_hidden():
    strategy = dict(tp=2, pp=2, dp=2, micro_batch_size=1, placement_id="tp-dp-pp")
    assert environment_label(strategy, include_dp=False) == (
        "(tp=2,pp=2,mbs=1,placement_id=tp-dp-pp)"
    )


def test_profile_normalization_preserves_distinct_placement_labels():
    rows = [dict(tp=2, pp=2, dp=2, micro_batch_size=1, placement_id=placement)
            for placement in ("tp-dp-pp", "pp-dp-tp")]
    strategies = _normalize_strategy_rows(rows, "fixture.csv")
    assert len({strategy["label"] for strategy in strategies}) == 2
    assert all(strategy["label"] == environment_label(strategy) for strategy in strategies)
