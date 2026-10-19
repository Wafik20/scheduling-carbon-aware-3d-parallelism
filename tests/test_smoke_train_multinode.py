import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


spec = importlib.util.spec_from_file_location(
    "smoke_train_multinode", Path(__file__).parents[1] / "scripts/smoke_train_multinode.py"
)
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


@pytest.mark.parametrize("count,local_rank,device", [(1, 0, 0), (1, 3, 0), (4, 3, 3)])
def test_cuda_device_mapping_without_uuid(count, local_rank, device):
    cuda = SimpleNamespace(device_count=lambda: count, set_device=Mock())
    assert smoke.select_cuda_device(cuda, local_rank) == (device, count)
    cuda.set_device.assert_called_once_with(device)


@pytest.mark.parametrize("count,local_rank", [(0, 0), (2, 2), (4, -1)])
def test_invalid_cuda_device_mapping(count, local_rank):
    cuda = SimpleNamespace(device_count=lambda: count, set_device=Mock())
    with pytest.raises(RuntimeError):
        smoke.select_cuda_device(cuda, local_rank)
    cuda.set_device.assert_not_called()


def test_duration_default_and_required_result_dir(tmp_path):
    assert smoke.parse_args(["--result-dir", str(tmp_path)]).duration == 90
    with pytest.raises(SystemExit):
        smoke.parse_args([])


@pytest.mark.parametrize("duration", ["0", "-1", "nan", "inf"])
def test_invalid_duration(duration, tmp_path):
    with pytest.raises(SystemExit):
        smoke.parse_args(["--result-dir", str(tmp_path), "--duration", duration])


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_slurm_ranks(world_size):
    env = dict(SLURM_PROCID=str(world_size - 1), SLURM_NTASKS=str(world_size), MASTER_ADDR="node1", MASTER_PORT="29500")
    assert smoke.slurm_ranks(env) == (world_size - 1, world_size)
    with pytest.raises(ValueError):
        smoke.slurm_ranks(dict(env, SLURM_NTASKS="0"))
    with pytest.raises(ValueError):
        smoke.slurm_ranks(dict(env, MASTER_ADDR=""))
    with pytest.raises(ValueError):
        smoke.slurm_ranks(dict(env, SLURM_PROCID="8"))


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_gradient_averaging(world_size):
    grad = Mock()
    model = Mock()
    model.parameters.return_value = [SimpleNamespace(grad=grad), SimpleNamespace(grad=None)]
    dist = Mock()
    smoke.average_gradients(model, dist, "dp-group", world_size)
    dist.all_reduce.assert_called_once_with(grad, op=dist.ReduceOp.SUM, group="dp-group")
    grad.div_.assert_called_once_with(world_size)


@pytest.mark.parametrize("rank,elapsed,local_stop", [(0, 91, 1), (0, 89, 0), (7, 91, 0)])
def test_stop_is_broadcast_from_rank_zero(monkeypatch, rank, elapsed, local_stop):
    monkeypatch.setattr(smoke.time, "perf_counter", lambda: elapsed)
    torch = Mock()
    elapsed_tensor = Mock()
    elapsed_tensor.item.return_value = elapsed
    stop = Mock()
    stop.item.return_value = 1
    torch.tensor.side_effect = [elapsed_tensor, stop]
    dist = Mock()
    assert smoke.synchronized_stop(torch, dist, rank, 0, 90)
    torch.tensor.assert_any_call([local_stop], dtype=torch.int32, device="cuda")
    dist.all_reduce.assert_called_once_with(elapsed_tensor, op=dist.ReduceOp.MIN)
    dist.broadcast.assert_called_once_with(stop, src=0)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_per_rank_result(tmp_path, monkeypatch, world_size):
    monkeypatch.setattr(smoke.socket, "gethostname", lambda: "node2")
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    smoke.write_result(tmp_path / "results", world_size - 1, 5, 90.1, world_size, world_size // 2 - 1, 0, 1)
    result = json.loads((tmp_path / f"results/rank{world_size - 1}.json").read_text())
    assert result == dict(rank=world_size - 1, hostname="node2", steps=5, duration=90.1,
                          elapsed_s=90.1, world_size=world_size, local_rank=world_size // 2 - 1,
                          cuda_device=0, visible_device_count=1)
