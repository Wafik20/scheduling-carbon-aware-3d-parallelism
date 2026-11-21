"""CPU placement checks, including actual Megatron definitions when available."""
import os
from collections import Counter
from pathlib import Path

import pytest

from experiments.placement_sweep.placements import (
    CONFIGS, ORDERS, enumerate_placements, groups_for_order, verify_megatron, unique_placements,
)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("order", ORDERS)
def test_groups_partition_world(config, order):
    tp, pp, dp = config
    groups = groups_for_order(tp, pp, dp, order)
    for axis, size in zip(("tp", "pp", "dp"), config):
        assert all(len(group) == size for group in groups[axis])
        assert sorted(rank for group in groups[axis] for rank in group) == list(range(tp * pp * dp))


def test_fastest_axis_and_pipeline_stage_order():
    groups = groups_for_order(2, 2, 2, "tp-dp-pp")
    assert groups == {
        "tp": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "pp": [[0, 4], [1, 5], [2, 6], [3, 7]],
        "dp": [[0, 2], [1, 3], [4, 6], [5, 7]],
    }


def test_dedup_preserves_every_requested_order():
    placements = enumerate_placements()
    assert len(placements) == 9
    assert len({p["placement_id"] for p in placements}) == len(placements)
    for config in CONFIGS:
        matching = [p for p in placements if (p["tp"], p["pp"], p["dp"]) == config]
        assert Counter(order for p in matching for order in p["equivalent_orders"]) == Counter(ORDERS)
        signatures = set()
        for placement in matching:
            signature = tuple(tuple(tuple(g) for g in placement["groups"][axis])
                              for axis in ("tp", "pp", "dp"))
            assert signature not in signatures
            signatures.add(signature)
            for order in placement["equivalent_orders"]:
                assert groups_for_order(*config, order) == placement["groups"]


def test_singleton_axes_collapse_and_all_active_axes_do_not():
    assert len(enumerate_placements([(8, 1, 1)])) == 1
    assert len(enumerate_placements([(2, 4, 1)])) == 2
    assert len(enumerate_placements([(2, 2, 2)])) == 6


@pytest.mark.parametrize("config,order", [
    ((0, 2, 4), "tp-pp-dp"), ((2, -1, 4), "tp-pp-dp"),
    ((True, 2, 4), "tp-pp-dp"), ((2, 2, 2), "tp-tp-dp"),
    ((2, 2, 2), "tp-pp"), ((2, 2, 2), "tp-pp-dp-cp"),
])
def test_invalid_placement_rejected(config, order):
    with pytest.raises(ValueError):
        groups_for_order(*config, order)


def test_actual_megatron_source_equivalence():
    checkout = Path(os.environ.get("MEGATRON_FORK", "/vast/cs2731_2026f/wat29/Megatron-LM"))
    source = checkout / "megatron/core/parallel_state.py"
    if not source.is_file():
        pytest.skip("Set MEGATRON_FORK to verify against an actual Megatron checkout")
    assert verify_megatron(source) == len(CONFIGS) * len(ORDERS)
