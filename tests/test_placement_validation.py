"""CPU validation against the installed external fork; no distributed setup."""
import json
import os
from pathlib import Path

import pytest

from experiments.placement_sweep import validate
from experiments.placement_sweep.placements import groups_for_order

FORK = Path(os.environ.get("MEGATRON_FORK", "/vast/cs2731_2026f/wat29/Megatron-LM"))


@pytest.fixture
def manifest(tmp_path):
    menu = tmp_path / "menu.json"
    menu.write_text(json.dumps([dict(tp=2, pp=1, order="tp-dp-pp", micro_batch_size=1)]))
    case = dict(id="gpt2-check", model="gpt2", tp=2, pp=1, dp=2,
                order="tp-dp-pp", micro_batch_size=1, seq_length=1024,
                menu=str(menu), groups=groups_for_order(2, 1, 2, "tp-dp-pp"))
    data = dict(world_size=4, nodes=2, gpus_per_node=2, global_batch_size=16,
                warmup_min_seconds=30, warmup_max_seconds=60, warmup_window=10,
                warmup_tol=.02, measure_seconds=60, cases=[case])
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data))
    return path, data


@pytest.mark.skipif(not (FORK / "profile_strategies.py").exists(), reason="external Megatron fork unavailable")
def test_actual_fork_accepts_menu_and_records_sources(manifest):
    path, _ = manifest
    report = validate.validate_manifest(path, FORK)
    assert report["case_count"] == 1
    assert len(report["source_sha256"]) == 3
    assert all(len(value) == 64 for value in report["source_sha256"].values())


@pytest.mark.skipif(not (FORK / "profile_strategies.py").exists(), reason="external Megatron fork unavailable")
@pytest.mark.parametrize("corruption", ["menu", "groups"])
def test_rejects_corrupt_inputs(manifest, corruption):
    path, data = manifest
    if corruption == "menu":
        Path(data["cases"][0]["menu"]).write_text('[{"tp":2,"pp":1,"order":"invalid"}]')
    else:
        data["cases"][0]["groups"]["tp"] = [[0, 3], [1, 2]]
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        validate.validate_manifest(path, FORK)


def test_rejects_unsupported_generated_flag():
    with pytest.raises(ValueError, match="Unsupported launcher"):
        validate.validate_command_flags(["--model", "gpt2", "--made-up=1"], ["--model"])


def synthetic_calibration(case):
    """Schema fixture only, not measured GPU data."""
    settings = dict(global_batch_size=16, warmup_min_seconds=30, warmup_max_seconds=60,
                    warmup_window=10, warmup_tol=.02, measure_seconds=60)
    expected = dict(settings, **case)
    mapping = []
    for rank in range(4):
        entry = dict(rank=rank, hostname=f"host-{rank // 2}", local_rank=rank % 2,
                     cuda_device=0, gpu_uuid=f"fixture-gpu-{rank}")
        for axis in ("tp", "pp", "dp"):
            entry[axis + "_group"] = next(g for g in case["groups"][axis] if rank in g)
        mapping.append(entry)
    row = {key: expected[key] for key in ("tp", "pp", "dp", "order", "micro_batch_size", "seq_length", "global_batch_size")}
    tokens = row["seq_length"] * row["global_batch_size"]
    row.update(power_w=1000., step_time_s=1., throughput_tok_s=float(tokens),
               joules_per_token=1000 / tokens, measure_iters=60, measured_tokens=60 * tokens,
               tokens_per_iter=tokens, warmup_converged=True, rank_mapping=mapping)
    return dict(settings, n_gpus=4, seq_length=row["seq_length"], strategies=[row])


def test_valid_calibration_and_malformed_artifacts(manifest, tmp_path):
    _, data = manifest
    case = dict(data, **data["cases"][0])
    path = tmp_path / "calibration.json"
    artifact = synthetic_calibration(case)
    path.write_text(json.dumps(artifact))
    assert validate.validate_calibration(path, case)["warmup_converged"] is True
    artifact["strategies"].append(dict(artifact["strategies"][0]))
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="exactly one"):
        validate.validate_calibration(path, case)
    artifact["strategies"].pop()
    artifact["strategies"][0]["rank_mapping"][1]["gpu_uuid"] = "fixture-gpu-0"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="distinct"):
        validate.validate_calibration(path, case)
