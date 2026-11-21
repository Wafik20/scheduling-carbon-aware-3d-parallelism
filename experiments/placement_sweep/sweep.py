#!/usr/bin/env python3
"""Generate and resume four/eight-GPU placement experiments; submission is opt-in."""
from __future__ import annotations
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train_profile import MODEL_CONFIGS
from experiments.placement_sweep.placements import enumerate_placements, configs_for_world_size

DEFAULT_ROOT = "runs/placement_sweep"
MODELS = ("gpt2", "gpt2_medium", "gpt2_large", "pythia_410m", "pythia_1_4b", "pythia_2_8b", "llama_1b", "llama_2b", "llama_3b")
SETTINGS = dict(world_size=4, nodes=2, gpus_per_node=2, global_batch_size=16,
                warmup_min_seconds=30.0, warmup_max_seconds=60.0,
                warmup_window=10, warmup_tol=0.02, measure_seconds=60.0)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def configuration_errors(cfg, tp, pp):
    """Static model divisibility requirements, checked again against the fork."""
    checks = (("num_layers", pp), ("num_attention_heads", tp), ("num_query_groups", tp))
    return [f"{key}={cfg[key]} is not divisible by {'PP' if key == 'num_layers' else 'TP'}={degree}"
            for key, degree in checks if cfg[key] is not None and cfg[key] % degree]


def generate(output=None, models=MODELS, micro_batches=(1, 2), **overrides):
    output = ROOT / (output or ("runs/placement_sweep_8gpu" if overrides.get("world_size", 4) == 8 else DEFAULT_ROOT))
    relative(output)  # Container commands require repository-local artifacts.
    settings = {**SETTINGS, **overrides}
    world = settings["world_size"]
    settings["gpus_per_node"] = overrides.get("gpus_per_node", world // 2)
    if settings["nodes"] != 2 or settings["gpus_per_node"] * 2 != world:
        raise ValueError("placement sweeps require two nodes with equal GPU counts")
    if settings["global_batch_size"] <= 0:
        raise ValueError("global batch size must be positive")
    for key in ("warmup_min_seconds", "warmup_max_seconds", "warmup_tol", "measure_seconds"):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if settings["warmup_min_seconds"] > settings["warmup_max_seconds"]:
        raise ValueError("warmup minimum exceeds maximum")
    cases, tasks, excluded = [], [], []
    placements = enumerate_placements(configs_for_world_size(world))
    for model in dict.fromkeys(models):
        cfg = MODEL_CONFIGS[model]
        for mbs in dict.fromkeys(micro_batches):
            if mbs <= 0 or settings["global_batch_size"] % (mbs * world):
                raise ValueError("global batch must be divisible by microbatch size times every DP degree")
            grouped = {}
            for placement in placements:
                tp, pp, dp = (placement[k] for k in ("tp", "pp", "dp"))
                errors = configuration_errors(cfg, tp, pp)
                if errors:
                    entry = dict(model=model, tp=tp, pp=pp, dp=dp, micro_batch_size=mbs, reasons=errors)
                    if entry not in excluded:
                        excluded.append(entry)
                    continue
                task_id = f"{model}_tp{tp}_pp{pp}_dp{dp}_mbs{mbs}"
                case_id = f"{task_id}_{placement['order']}"
                case = {**placement, "id": case_id, "task_id": task_id,
                        "model": model, "family": model.split("_")[0] if not model.startswith("gpt2") else "gpt2",
                        "model_size": {"gpt2": "124m", "gpt2_medium": "355m", "gpt2_large": "774m"}.get(model, model.split("_", 1)[-1]),
                        "micro_batch_size": mbs,
                        "seq_length": cfg["seq_length"],
                        "menu": relative(output / "cases" / case_id / "menu.json")}
                cases.append(case)
                grouped.setdefault(task_id, []).append(case_id)
            tasks.extend({"id": key, "case_ids": ids} for key, ids in grouped.items())
    manifest = {"schema_version": 1, **settings, "cases": cases, "tasks": tasks}
    if excluded:
        manifest["excluded_configs"] = excluded
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("existing manifest differs; use a new output directory to preserve results")
    for case in cases:
        write_json(ROOT / case["menu"], [{k: case[k] for k in ("tp", "pp", "micro_batch_size", "order")}])
        write_json(output / "cases" / case["id"] / "metadata.json", {**settings, **case})
    write_json(manifest_path, manifest)
    fields = ["id", "task_id", "model", "family", "model_size", "tp", "pp", "dp", "order", "equivalent_orders", "micro_batch_size", "seq_length", "groups", "menu"]
    with (output / "matrix.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            writer.writerow({key: json.dumps(value) if isinstance(value, (dict, list)) else value
                             for key, value in case.items()})
    return manifest_path


def load_manifest(path):
    return json.loads(Path(path).read_text())


def case_dir(manifest_path, case):
    return Path(manifest_path).resolve().parent / "cases" / case["id"]


def valid_result(manifest_path, manifest, case):
    from experiments.placement_sweep.validate import validate_calibration
    status_path = case_dir(manifest_path, case) / "status.json"
    if not status_path.exists():
        return None
    try:
        state = json.loads(status_path.read_text())
        if state.get("status") != "completed":
            return None
        validate_calibration(ROOT / state["calibration"], {**manifest, **case})
    except (ValueError, KeyError, OSError, TypeError):
        return None
    return state


def profile_command(manifest, case, attempt):
    launcher = ("batch_files/profile_multinode_4gpu.sbatch" if manifest["world_size"] == 4
                else "batch_files/profile_multinode.sbatch")
    command = ["bash", launcher, "--model", case["model"],
               "--profile-menu", case["menu"], "--output", relative(attempt / "calibration.json"),
               "--work-dir", relative(attempt / "work"), "--micro-batch-size", str(case["micro_batch_size"]),
               "--seq-length", str(case["seq_length"]), "--profile-no-checkpoints"]
    for key in ("global_batch_size", "warmup_min_seconds", "warmup_max_seconds", "warmup_window", "warmup_tol", "measure_seconds"):
        command += ["--" + key.replace("_", "-"), str(manifest[key])]
    return command


def cleanup_case_checkpoints(directory, inactive=False):
    """Remove only checkpoint payloads; retain metadata and every measurement.

    Caller holds the task lock. A running/unknown attempt may have orphan GPU
    writers after interruption, so require explicit confirmation it has ended.
    """
    recovered = 0
    for attempt in sorted((directory / "attempts").glob("[0-9]*")):
        state_path = attempt / "status.json"
        try:
            state = json.loads(state_path.read_text()).get("status")
        except (OSError, ValueError):
            state = None
        if state not in ("completed", "failed") and not inactive:
            continue
        checkpoint = attempt / "work" / "profile_ckpt"
        # Never follow a redirected work directory or checkpoint tree.
        if any(part.is_symlink() for part in (directory, directory / "attempts", attempt,
                                               attempt / "work", checkpoint)):
            continue
        for payload in checkpoint.rglob("*"):
            if (payload.is_file() and not payload.is_symlink()
                    and not any(parent.is_symlink() for parent in payload.parents)
                    and payload.suffix in (".distcp", ".pt")):
                recovered += payload.stat().st_blocks * 512
                payload.unlink()
    return recovered


def cleanup_checkpoints(manifest_path, inactive=False):
    """Clean idle tasks; --inactive also permits confirmed stopped attempts."""
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    locks = manifest_path.parent / "locks"
    locks.mkdir(exist_ok=True)
    by_id = {case["id"]: case for case in manifest["cases"]}
    recovered = 0
    for task in manifest["tasks"]:
        with (locks / (task["id"] + ".lock")).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"Skipping active task: {task['id']}")
                continue
            for case_id in task["case_ids"]:
                recovered += cleanup_case_checkpoints(
                    case_dir(manifest_path, by_id[case_id]), inactive=inactive)
    print(f"Removed checkpoint payloads: {recovered} allocated bytes")
    return recovered


def run_task(manifest_path, task_index):
    from experiments.placement_sweep.validate import validate_calibration
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    task = manifest["tasks"][task_index]
    locks = manifest_path.parent / "locks"
    locks.mkdir(exist_ok=True)
    with (locks / (task["id"] + ".lock")).open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("task is already running")
        failed = False
        by_id = {case["id"]: case for case in manifest["cases"]}
        for case_id in task["case_ids"]:
            case = by_id[case_id]
            cleanup_case_checkpoints(case_dir(manifest_path, case))
            if valid_result(manifest_path, manifest, case):
                print(f"Skipping validated completion: {case_id}", flush=True)
                continue
            directory = case_dir(manifest_path, case)
            attempts = directory / "attempts"
            attempts.mkdir(exist_ok=True)
            number = max([int(p.name) for p in attempts.iterdir() if p.is_dir() and p.name.isdigit()], default=0) + 1
            attempt = attempts / f"{number:04d}"
            attempt.mkdir()
            command = profile_command(manifest, case, attempt)
            state = dict(status="running", attempt=number, command=command,
                         started_at=datetime.now(timezone.utc).isoformat(),
                         calibration=relative(attempt / "calibration.json"))
            write_json(directory / "status.json", state)
            write_json(attempt / "status.json", state)
            write_json(attempt / "metadata.json", {**{k: v for k, v in manifest.items() if k not in ("cases", "tasks", "excluded_configs")}, **case})
            env = os.environ.copy()
            env["POWER_OUTPUT_DIR"] = str(attempt / "power")
            try:
                with (attempt / "launcher.log").open("w") as log:
                    result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                state["returncode"] = result.returncode
                if result.returncode:
                    tail = "\n".join((attempt / "launcher.log").read_text(errors="replace").splitlines()[-30:])
                    raise ValueError(f"profiler exited {result.returncode}; launcher.log tail:\n{tail}")
                validate_calibration(attempt / "calibration.json", {**manifest, **case})
                state["status"] = "completed"
            except (OSError, ValueError, KeyError, TypeError) as exc:
                state.update(status="failed", error=str(exc))
                failed = True
            state["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(attempt / "status.json", state)
            write_json(directory / "status.json", state)
            cleanup_case_checkpoints(directory)
        return int(failed)


def pending_tasks(manifest_path, manifest):
    cases = {case["id"]: case for case in manifest["cases"]}
    return [i for i, task in enumerate(manifest["tasks"])
            if any(not valid_result(manifest_path, manifest, cases[cid]) for cid in task["case_ids"])]


def submit(manifest_path, concurrency=1, execute=False):
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    manifest_path = Path(manifest_path).resolve()
    manifest = load_manifest(manifest_path)
    pending = pending_tasks(manifest_path, manifest)
    if not pending:
        print("All cases have validated completions; nothing to submit.")
        return
    script = ROOT / "experiments/placement_sweep/array.sbatch"
    logs = manifest_path.parent / "slurm"
    logs.mkdir(exist_ok=True)
    command = ["sbatch", "--parsable", "-M", "gpu", "--account=cs2731_2026f",
               "--partition=a100", "--constraint=amd", f"--nodes={manifest['nodes']}", f"--ntasks={manifest['world_size']}",
               f"--ntasks-per-node={manifest['gpus_per_node']}", f"--gres=gpu:{manifest['gpus_per_node']}", "--cpus-per-task=4", "--mem=64G", "--time=02:00:00",
               "--chdir=" + str(ROOT), "--output=" + str(logs / "%A_%a.out"),
               "--array=" + ",".join(map(str, pending)) + f"%{concurrency}", str(script), relative(manifest_path)]
    print("cd " + shlex.quote(str(ROOT)) + " && " + shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)
    else:
        print("Dry run only; pass --execute to submit.")


def status(manifest_path):
    manifest = load_manifest(manifest_path)
    counts = dict(completed=0, failed=0, running=0, incomplete=0)
    for case in manifest["cases"]:
        if valid_result(manifest_path, manifest, case):
            counts["completed"] += 1
        else:
            path = case_dir(manifest_path, case) / "status.json"
            state = json.loads(path.read_text()).get("status") if path.exists() else None
            counts[state if state in ("failed", "running") else "incomplete"] += 1
    print(json.dumps(counts, indent=2))
    return counts


def aggregate(manifest_path, output=None):
    manifest = load_manifest(manifest_path)
    rows = []
    for case in manifest["cases"]:
        state = valid_result(manifest_path, manifest, case)
        if not state:
            continue
        calibration = json.loads((ROOT / state["calibration"]).read_text())
        row = {**{k: v for k, v in manifest.items() if k not in ("cases", "tasks", "excluded_configs")}, **case,
               **{k: v for k, v in calibration.items() if k != "strategies"},
               **calibration["strategies"][0], "attempt": state["attempt"], "calibration": state["calibration"]}
        row["energy_per_step_j"] = row["power_w"] * row["step_time_s"]
        rows.append({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    output = Path(output) if output else Path(manifest_path).parent / "results.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ["id", "model", "tp", "pp", "dp", "order"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} validated cases to {output}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    gen = commands.add_parser("generate")
    gen.add_argument("--output-dir", default=None)
    gen.add_argument("--world-size", type=int, choices=(4, 8), default=4)
    gen.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    gen.add_argument("--micro-batch-sizes", nargs="+", type=int, default=[1, 2])
    for key in ("global_batch_size", "warmup_min_seconds", "warmup_max_seconds", "measure_seconds"):
        gen.add_argument("--" + key.replace("_", "-"), type=type(SETTINGS[key]), default=SETTINGS[key])
    for action in ("submit", "dry-run", "run-task", "status", "aggregate", "cleanup-checkpoints"):
        sub = commands.add_parser(action)
        sub.add_argument("--manifest", default=DEFAULT_ROOT + "/manifest.json")
        if action == "submit":
            sub.add_argument("--concurrency", type=int, default=1)
            sub.add_argument("--execute", action="store_true")
        if action == "run-task":
            sub.add_argument("--task-index", type=int, required=True)
        if action == "cleanup-checkpoints":
            sub.add_argument("--inactive", action="store_true",
                             help="Confirm old jobs/writers have ended, including running/unknown attempts")
        if action == "aggregate":
            sub.add_argument("--output")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    try:
        if args.action == "generate":
            output_dir = args.output_dir or (DEFAULT_ROOT if args.world_size == 4 else "runs/placement_sweep_8gpu")
            path = generate(output_dir, args.models, args.micro_batch_sizes, world_size=args.world_size,
                            **{key: getattr(args, key) for key in ("global_batch_size", "warmup_min_seconds", "warmup_max_seconds", "measure_seconds")})
            print(path)
        elif args.action == "dry-run":
            manifest = load_manifest(args.manifest)
            for case in manifest["cases"]:
                attempt = case_dir(args.manifest, case) / "attempts/DRY_RUN"
                print(shlex.join(profile_command(manifest, case, attempt)))
        elif args.action == "submit":
            submit(args.manifest, args.concurrency, args.execute)
        elif args.action == "run-task":
            if args.task_index < 0:
                raise ValueError("task index must be nonnegative")
            return run_task(args.manifest, args.task_index)
        elif args.action == "cleanup-checkpoints":
            cleanup_checkpoints(args.manifest, args.inactive)
        elif args.action == "status":
            status(args.manifest)
        else:
            aggregate(args.manifest, args.output)
    except (ValueError, OSError, KeyError, IndexError, RuntimeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
