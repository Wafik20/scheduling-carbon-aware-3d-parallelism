#!/usr/bin/env python3
"""CPU checks using AST-isolated definitions from the actual Megatron fork.

This executes the fork's pure menu/shape/rank code, not a reimplementation.
It does not import CUDA dependencies or test distributed initialization.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def extract(path, names, namespace):
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Missing required fork definitions in {path}: {names}")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return hashlib.sha256(text.encode()).hexdigest()


def fork_logic(fork):
    # Register a real module so dataclass annotation handling works on Python 3.10+.
    module = types.ModuleType("_placement_fork_validation")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    exec("import json, sys\nfrom dataclasses import dataclass\nfrom typing import List, Optional", ns)
    hashes = {}
    paths = {
        "profile_strategies.py": ["ProfileCase", "_coerce_positive_int", "validate_profile_order", "load_profile_menu", "validate_micro_batch_cases_or_die"],
        "online_elastic_pretrain.py": ["validate_strategies_or_die"],
        "megatron/core/parallel_state.py": ["generate_masked_orthogonal_rank_groups", "RankGenerator"],
    }
    for relative, names in paths.items():
        hashes[relative] = extract(fork / relative, names, ns)
    return ns, hashes


def cli_options(path):
    tree = ast.parse(path.read_text())
    return sorted({arg.value for node in ast.walk(tree)
                   if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "add_argument"
                   for arg in node.args if isinstance(arg, ast.Constant)
                   and isinstance(arg.value, str) and arg.value.startswith("--")})


def validate_command_flags(command, supported):
    unsupported = {token.split("=", 1)[0] for token in command if token.startswith("--")} - set(supported)
    if unsupported:
        raise ValueError(f"Unsupported launcher arguments: {sorted(unsupported)}")


def model_configs():
    tree = ast.parse((ROOT / "train_online_elastic.py").read_text())
    assignments = [n for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and (t.id.endswith("_EXTRA_ARGS") or t.id == "MODEL_CONFIGS") for t in n.targets)]
    ns = {}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "model_configs", "exec"), ns)
    return ns["MODEL_CONFIGS"]


def validate_manifest(manifest_path, fork):
    manifest = json.loads(manifest_path.read_text())
    ns, hashes = fork_logic(fork)
    models = model_configs()
    world = manifest["world_size"]
    per_node = manifest["gpus_per_node"]
    if world <= 0 or per_node <= 0 or world % per_node:
        raise ValueError("Invalid world_size/gpus_per_node")
    if manifest.get("nodes", world // per_node) * per_node != world:
        raise ValueError("nodes * gpus_per_node must equal world_size")
    cases = manifest["cases"]
    if not cases:
        raise ValueError("Manifest has no cases")
    results = []
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from experiments.placement_sweep.sweep import profile_command
    launcher_options = cli_options(ROOT / "train_profile.py")
    ids = set()
    signatures = set()
    aliases_by_config = {}
    for case in cases:
        if case["id"] in ids:
            raise ValueError(f"Duplicate case id: {case['id']}")
        ids.add(case["id"])
        command = profile_command(manifest, case, ROOT / "runs" / "placement_validation" / case["id"])
        validate_command_flags(command, launcher_options)
        menu = Path(case["menu"])
        if not menu.is_absolute():
            menu = ROOT / menu
        loaded = ns["load_profile_menu"](str(menu))
        if len(loaded) != 1:
            raise ValueError(f"{menu}: isolated placement runs require exactly one menu case")
        row = loaded[0]
        mbs = row.effective_micro_batch_size(case["micro_batch_size"])
        for name in ("tp", "pp", "order"):
            if getattr(row, name) != case[name]:
                raise ValueError(f"{case['id']}: menu {name} differs from manifest")
        if mbs != case["micro_batch_size"]:
            raise ValueError(f"{case['id']}: inconsistent microbatch size")
        cfg = models[case["model"]]
        ns["validate_strategies_or_die"](loaded, world, cfg["num_layers"], cfg["num_attention_heads"], cfg["num_query_groups"], 0)
        ns["validate_micro_batch_cases_or_die"](loaded, world, manifest["global_batch_size"], case["micro_batch_size"], 0)
        dp = world // (row.tp * row.pp)
        if dp != case["dp"]:
            raise ValueError(f"{case['id']}: DP mismatch")
        gen = ns["RankGenerator"](tp=row.tp, pp=row.pp, dp=dp, ep=1, cp=1, order=row.order)
        groups = {axis: gen.get_ranks(axis) for axis in ("tp", "pp", "dp")}
        if groups != case["groups"]:
            raise ValueError(f"{case['id']}: predicted rank groups differ from actual fork RankGenerator")
        config_key = (case["model"], row.tp, row.pp, dp, mbs)
        signature = (config_key, json.dumps(groups, sort_keys=True))
        if signature in signatures:
            raise ValueError(f"{case['id']}: duplicate rank placement")
        signatures.add(signature)
        if "equivalent_orders" in case:
            aliases = case["equivalent_orders"]
            if row.order not in aliases:
                raise ValueError(f"{case['id']}: representative order missing from aliases")
            aliases_by_config.setdefault(config_key, []).extend(aliases)
            for order in aliases:
                alias_gen = ns["RankGenerator"](tp=row.tp, pp=row.pp, dp=dp, ep=1, cp=1, order=order)
                if {axis: alias_gen.get_ranks(axis) for axis in groups} != groups:
                    raise ValueError(f"{case['id']}: alias rank groups differ from representative")
        for axis, axis_groups in groups.items():
            if sorted(r for group in axis_groups for r in group) != list(range(world)):
                raise ValueError(f"{case['id']}: {axis} groups do not partition WORLD")
        results.append({"id": case["id"], "rank_groups": groups,
                        "cross_node_groups": {axis: sum(len({rank // per_node for rank in group}) > 1 for group in axis_groups)
                                              for axis, axis_groups in groups.items()}})
    from experiments.placement_sweep.placements import ORDERS
    for config, aliases in aliases_by_config.items():
        if sorted(aliases) != sorted(ORDERS):
            raise ValueError(f"{config}: aliases must cover all six orders exactly once")
    excluded = manifest.get("excluded_configs", [])
    for entry in excluded:
        cfg = models[entry["model"]]
        if not entry.get("reasons"):
            raise ValueError("Excluded configuration lacks reasons")
        if entry["tp"] * entry["pp"] * entry["dp"] != world:
            raise ValueError("Excluded configuration has incorrect world size")
        strategy = types.SimpleNamespace(tp=entry["tp"], pp=entry["pp"])
        try:
            ns["validate_strategies_or_die"]([strategy], world, cfg["num_layers"],
                                           cfg["num_attention_heads"], cfg["num_query_groups"], 1)
        except SystemExit as exc:
            if exc.code != 1:
                raise
        else:
            raise ValueError(f"Excluded configuration is supported by fork: {entry}")
    launcher_options = cli_options(ROOT / "train_profile.py")
    required = {"--model", "--profile-menu", "--output", "--num-gpus", "--micro-batch-size", "--global-batch-size", "--seq-length", "--slurm", "--in-container", "--work-dir", "--warmup-min-seconds", "--warmup-max-seconds", "--warmup-window", "--warmup-tol", "--measure-seconds", "--profile-no-checkpoints"}
    if required - set(launcher_options):
        raise ValueError(f"Launcher lacks required arguments: {sorted(required - set(launcher_options))}")
    profiler_options = cli_options(fork / "profile_strategies.py")
    required_profiler = {"--profile-menu", "--output", "--elastic-work-dir", "--warmup-min-seconds", "--warmup-max-seconds", "--warmup-window", "--warmup-tol", "--measure-seconds", "--profile-no-checkpoints"}
    if required_profiler - set(profiler_options):
        raise ValueError("Fork profiler lacks required CLI arguments")
    return {"status": "passed", "case_count": len(results), "validation_mode": "AST-isolated actual fork definitions (CPU only)",
            "limitations": "No CUDA/NCCL, model construction, runtime CLI validation, or measured rank/device validation. Cross-node predictions assume block ranks.",
            "fork": str(fork.resolve()), "source_sha256": hashes, "excluded_configs": excluded,
            "launcher_cli_options": launcher_options, "profiler_cli_options": profiler_options, "cases": results}


def validate_calibration(path, case):
    """Reject incomplete or mismatched GPU artifacts; return the matching row.

    Each isolated run must contain exactly one matching strategy.
    Host checks use the manifest topology and block-distributed ranks.
    """
    document = json.loads(Path(path).read_text())
    if len(document.get("strategies", [])) != 1:
        raise ValueError("Isolated calibration requires exactly one strategy")
    keys = ("tp", "pp", "dp", "order", "micro_batch_size")
    matches = [row for row in document.get("strategies", [])
               if all(row.get(key) == case[key] for key in keys)]
    if len(matches) != 1:
        raise ValueError("Calibration must contain exactly one matching strategy")
    row = matches[0]
    if not isinstance(row.get("warmup_converged"), bool):
        raise ValueError("Calibration requires boolean warmup_converged")
    for key in ("global_batch_size", "seq_length"):
        if row.get(key) != case[key] or document.get(key) != case[key]:
            raise ValueError(f"Calibration {key} mismatch")
    world = case.get("world_size", 4)
    per_node = case.get("gpus_per_node", 2)
    if document.get("n_gpus") != world:
        raise ValueError("Calibration n_gpus mismatch")
    for key in ("warmup_min_seconds", "warmup_max_seconds", "warmup_window", "warmup_tol", "measure_seconds"):
        if key in case and document.get(key) != case[key]:
            raise ValueError(f"Calibration {key} mismatch")
    for key in ("power_w", "step_time_s", "throughput_tok_s", "joules_per_token", "measure_iters", "measured_tokens"):
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Calibration requires positive finite {key}")
    tokens = case["global_batch_size"] * case["seq_length"]
    if row.get("tokens_per_iter") != tokens or row["measured_tokens"] != tokens * row["measure_iters"]:
        raise ValueError("Calibration token accounting mismatch")
    mapping = row.get("rank_mapping", [])
    if len(mapping) != world or sorted(entry["rank"] for entry in mapping) != list(range(world)):
        raise ValueError("Rank mapping must cover WORLD exactly once")
    if len({entry.get("gpu_uuid") for entry in mapping}) != world or any(not entry.get("gpu_uuid") for entry in mapping):
        raise ValueError("Rank mapping needs distinct nonempty GPU UUIDs")
    hosts = {}
    for entry in mapping:
        rank = entry["rank"]
        host = entry.get("hostname")
        if not host:
            raise ValueError("Rank mapping missing hostname")
        block = rank // per_node
        if block in hosts and hosts[block] != host:
            raise ValueError("Observed ranks are not block-distributed by host")
        hosts[block] = host
        if entry.get("local_rank") != rank % per_node:
            raise ValueError("Observed local rank differs from block mapping")
        for axis in ("tp", "pp", "dp"):
            expected = next(group for group in case["groups"][axis] if rank in group)
            if entry.get(axis + "_group") != expected:
                raise ValueError(f"Observed {axis} group mismatch at rank {rank}")
    if len(set(hosts.values())) != world // per_node:
        raise ValueError("Observed host count mismatch")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--megatron-fork", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON validation report")
    args = parser.parse_args()
    try:
        report = validate_manifest(args.manifest.resolve(), args.megatron_fork.resolve())
    except (ValueError, KeyError, OSError, AssertionError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated {report['case_count']} cases using actual fork menu loader and RankGenerator (AST-isolated, CPU only).")


if __name__ == "__main__":
    main()
