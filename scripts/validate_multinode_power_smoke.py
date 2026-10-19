"""Validate completion and raw power artifacts from the multinode smoke test."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def validate(run, duration=90):
    run = Path(run)
    layout = json.loads((run / "layout.json").read_text())
    world_size = layout["world_size"]
    expected_nodes = layout["nodes"]
    gpus_per_node = layout["gpus_per_node"]
    if (type(world_size) is not int or world_size <= 0
            or type(gpus_per_node) is not int or gpus_per_node <= 0
            or not isinstance(expected_nodes, list) or not expected_nodes
            or any(not isinstance(node, str) or not node for node in expected_nodes)
            or len(set(expected_nodes)) != len(expected_nodes)
            or world_size != len(expected_nodes) * gpus_per_node):
        raise ValueError("Invalid run layout")
    ranks = [json.loads(p.read_text()) for p in sorted((run / "ranks").glob("rank*.json"))]
    if len(ranks) != world_size or {r["rank"] for r in ranks} != set(range(world_size)):
        raise ValueError(f"Expected completion records for exactly {world_size} ranks")
    nodes = {}
    for rank in ranks:
        if rank["world_size"] != world_size or rank["steps"] <= 0 or rank["elapsed_s"] < duration:
            raise ValueError("Rank did not complete the requested training workload")
        nodes.setdefault(rank["hostname"], []).append(rank)
    if set(nodes) != set(expected_nodes) or any(len(group) != gpus_per_node for group in nodes.values()):
        raise ValueError("Completed ranks do not match the run layout")
    for group in nodes.values():
        if any(type(r["local_rank"]) is not int for r in group) or {
            r["local_rank"] for r in group
        } != set(range(gpus_per_node)):
            raise ValueError("Expected unique local ranks for each node")
        for rank in group:
            count, device = rank["visible_device_count"], rank["cuda_device"]
            if (type(count) is not int or count <= 0 or type(device) is not int
                    or not 0 <= device < count
                    or device != (0 if count == 1 else rank["local_rank"])):
                raise ValueError("Invalid rank CUDA device mapping")
    if len({r["steps"] for r in ranks}) != 1:
        raise ValueError("Ranks completed different step counts")
    files = list(run.glob("job_*/run_*/*.csv"))
    if len(files) != len(expected_nodes) or {p.stem for p in files} != set(nodes):
        raise ValueError("Expected one power CSV for each training node")
    for path in files:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) < 2:
            raise ValueError(f"{path}: insufficient samples")
        power_columns = {key for key in rows[0] if key.startswith("GPU_") and key.endswith("_W")}
        if power_columns != {f"GPU_{i}_W" for i in range(gpus_per_node)}:
            raise ValueError(f"{path}: GPU power columns do not match the run layout")
        identities = rows[0]["gpu_uuids"].split(";")
        if len(identities) != gpus_per_node or len(set(identities)) != gpus_per_node or not all(identities):
            raise ValueError(f"{path}: expected {gpus_per_node} distinct GPU UUIDs")
        elapsed = []
        for row in rows:
            stamp = datetime.fromisoformat(row["timestamp"])
            if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
                raise ValueError(f"{path}: timestamps must be UTC")
            if row["hostname"] != path.stem or int(row["n_gpus"]) != gpus_per_node or row["gpu_uuids"].split(";") != identities:
                raise ValueError(f"{path}: inconsistent node/GPU identity")
            powers = [float(row[f"GPU_{i}_W"]) for i in range(gpus_per_node)]
            total = float(row["total_power_W"])
            if not all(math.isfinite(p) and p > 0 for p in [*powers, total]):
                raise ValueError(f"{path}: missing or invalid power measurements")
            if not math.isclose(total, sum(powers), abs_tol=0.05):
                raise ValueError(f"{path}: total does not match GPU readings")
            elapsed.append(float(row["elapsed_s"]))
        if not all(math.isfinite(t) for t in elapsed) or any(b <= a for a, b in zip(elapsed, elapsed[1:])):
            raise ValueError(f"{path}: invalid elapsed times")
        if elapsed[-1] - elapsed[0] < duration - 1:
            raise ValueError(f"{path}: power series does not span training duration")
        print(f"{path}: {len(rows)} samples, {gpus_per_node} GPUs, valid totals")
    print(f"MULTINODE POWER SMOKE PASSED: {world_size} ranks, {len(nodes)} nodes, {world_size} GPUs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--duration", type=float, default=90)
    args = parser.parse_args()
    validate(args.run, args.duration)


if __name__ == "__main__":
    main()
