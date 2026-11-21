"""CPU-only rank placement enumeration, with Megatron's fastest-axis-first order.

CP and EP are fixed at one. Global ranks are unchanged: physical node/local-GPU
assignment is the launcher's responsibility, not a permutation applied here.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import List


CONFIGS = ((4, 1, 1), (1, 4, 1), (1, 1, 4), (2, 2, 1), (2, 1, 2), (1, 2, 2))
ORDERS = ("tp-dp-pp", "tp-pp-dp", "dp-tp-pp", "dp-pp-tp", "pp-tp-dp", "pp-dp-tp")

def configs_for_world_size(world_size: int) -> tuple:
    """All factor triples; retain the original four-GPU ordering for resume."""
    if world_size == 4:
        return CONFIGS
    if world_size != 8:
        raise ValueError("placement sweeps support world sizes 4 and 8")
    return tuple((tp, pp, world_size // (tp * pp))
                 for tp in (1, 2, 4, 8) for pp in (1, 2, 4, 8)
                 if world_size % (tp * pp) == 0)


def groups_for_order(tp: int, pp: int, dp: int, order: str) -> dict[str, list[list[int]]]:
    """Return TP, PP, DP groups in Megatron rank/group order."""
    sizes = {"tp": tp, "pp": pp, "dp": dp}
    if any(type(size) is not int or size < 1 for size in sizes.values()):
        raise ValueError("TP, PP, and DP must be positive integers")
    axes = order.split("-")
    if len(axes) != 3 or set(axes) != set(sizes):
        raise ValueError("order must contain tp, pp, dp exactly once")
    strides = {}
    stride = 1
    for axis in axes:
        strides[axis] = stride
        stride *= sizes[axis]
    groups = {}
    for axis in sizes:
        # A base rank has coordinate zero on the grouped axis. Increasing base
        # ranks enumerate the remaining coordinates in fastest-axis-first order.
        groups[axis] = [
            [base + i * strides[axis] for i in range(sizes[axis])]
            for base in range(stride)
            if (base // strides[axis]) % sizes[axis] == 0
        ]
    return groups


def group_signature(groups: dict[str, list[list[int]]]) -> tuple:
    """Canonical group membership; preserve pipeline-stage rank order."""
    return tuple((axis, tuple(sorted(tuple(group) for group in groups[axis])))
                 for axis in ("tp", "pp", "dp"))


def enumerate_placements(configs=None, orders=None) -> list[dict]:
    """Deduplicate equivalent orderings within each (TP, PP, DP) config."""
    configs = CONFIGS if configs is None else configs
    orders = ORDERS if orders is None else tuple(orders)
    placements = []
    for tp, pp, dp in configs:
        unique = {}
        for order in orders:
            groups = groups_for_order(tp, pp, dp, order)
            signature = group_signature(groups)
            if signature in unique:
                unique[signature]["equivalent_orders"].append(order)
                continue
            placement = {
                "placement_id": f"tp{tp}_pp{pp}_dp{dp}_{order}",
                "tp": tp, "pp": pp, "dp": dp, "order": order,
                "equivalent_orders": [order], "groups": groups,
            }
            unique[signature] = placement
            placements.append(placement)
    return placements


def load_megatron_rank_generator(parallel_state_path: str | Path):
    """Load the actual checkout's two pure definitions without CUDA imports.

    This is a CPU source-level reference check, not a distributed runtime test.
    Only use a trusted Megatron checkout: its Python definitions are executed.
    """
    path = Path(parallel_state_path)
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"generate_masked_orthogonal_rank_groups", "RankGenerator"}
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    if {node.name for node in definitions} != names:
        raise ValueError(f"Missing rank generator definitions in {path}")
    namespace = {"List": List}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["RankGenerator"]


def verify_megatron(parallel_state_path: str | Path, configs=None, orders=None) -> int:
    """Assert exact group and rank ordering against actual Megatron source."""
    generator = load_megatron_rank_generator(parallel_state_path)
    checked = 0
    for tp, pp, dp in CONFIGS if configs is None else configs:
        for order in ORDERS if orders is None else orders:
            reference = generator(tp=tp, pp=pp, dp=dp, ep=1, cp=1, order=order)
            groups = groups_for_order(tp, pp, dp, order)
            for axis in ("tp", "pp", "dp"):
                if groups[axis] != reference.get_ranks(axis):
                    raise AssertionError(f"Megatron mismatch: {(tp, pp, dp)}, {order}, {axis}")
            checked += 1
    return checked


def unique_placements(tp: int, pp: int, dp: int) -> list[dict]:
    """Return representative orders and their aliases for one configuration."""
    return enumerate_placements([(tp, pp, dp)])
