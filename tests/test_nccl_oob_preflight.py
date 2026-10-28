import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "oob", Path(__file__).parents[1] / "scripts/nccl_oob_preflight.py")
oob = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oob)


def interface(name, ip=None, flags=("UP",)):
    return dict(ifname=name, flags=list(flags),
                addr_info=[dict(family="inet", local=ip)] if ip else [])


def test_selection_requires_real_up_nonloop_ipv4():
    entries = [interface("eth0", "10.0.0.1"), interface("eth5", "10.0.0.5", ()),
               interface("lo", "127.0.0.1", ("UP", "LOOPBACK")), interface("eth6")]
    assert oob.select_interface(entries, ["10.0.0.1"]) == ("eth0", "10.0.0.1")
    assert oob.select_interface(entries, [], "=eth0") == ("eth0", "10.0.0.1")
    for override in ("=eth5", "=lo", "=eth6", "=missing", "eth0", "^eth0", "=eth0,eth5"):
        with pytest.raises((ValueError, RuntimeError)):
            oob.select_interface(entries, ["10.0.0.1"], override)
    with pytest.raises(RuntimeError):
        oob.select_interface(entries, ["10.0.0.99"])


def test_real_bidirectional_tcp_handshake(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(oob.exchange, tmp_path, rank, "127.0.0.1", 3)
                   for rank in (0, 1)]
        peers = [future.result(timeout=5) for future in futures]
    assert [peer["rank"] for peer in peers] == [1, 0]
    for rank in (0, 1):
        assert json.loads((tmp_path / f"rank{rank}.oob-ready.json").read_text())["port"] > 0


def test_failure_retains_inventory_and_environment(monkeypatch, tmp_path):
    entries = [interface("eth5", "10.0.0.5", ()), interface("eth6")]
    monkeypatch.setattr(oob, "inventory", lambda: entries)
    monkeypatch.setattr(oob.socket, "getaddrinfo", lambda *a: [(0, 0, 0, "", ("10.0.0.5", 0))])
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "=eth5")
    with pytest.raises(RuntimeError):
        oob.configure_oob(tmp_path, 0)
    report = json.loads((tmp_path / "rank0.oob.json").read_text())
    assert report["interfaces"] == entries
    assert not report["verified"]
    assert oob.os.environ["NCCL_SOCKET_IFNAME"] == "=eth5"


@pytest.mark.parametrize("name, rank", [("eth5", 0), ("eno1", 1)])
def test_configures_verified_interface_without_changing_transport(monkeypatch, tmp_path, name, rank):
    monkeypatch.setattr(oob.os, "environ", dict(oob.os.environ))
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "node0")
    monkeypatch.setenv("NCCL_NET", "IB")
    monkeypatch.setenv("NCCL_IB_DISABLE", "0")
    monkeypatch.setenv("NCCL_NET_PLUGIN", "none")
    monkeypatch.setattr(oob, "inventory", lambda: [interface(name, "10.0.0.1")])
    monkeypatch.setattr(oob.socket, "getaddrinfo", lambda *a: [(0, 0, 0, "", ("10.0.0.1", 0))])
    def exchange(directory, actual_rank, ip):
        report = json.loads((directory / f"rank{rank}.oob.json").read_text())
        assert not report["verified"] and report["interface"] == name
        assert "NCCL_SOCKET_IFNAME" not in oob.os.environ
        assert actual_rank == rank and ip == "10.0.0.1"
        return dict(rank=1-rank, ip="10.0.0.2", port=12345)
    monkeypatch.setattr(oob, "exchange", exchange)
    oob.configure_oob(tmp_path, rank)
    assert oob.os.environ["NCCL_SOCKET_IFNAME"] == f"={name}"
    assert oob.os.environ["NCCL_SOCKET_FAMILY"] == "AF_INET"
    assert oob.os.environ["MASTER_ADDR"] == "node0"
    assert oob.os.environ["NCCL_NET"] == "IB"
    assert oob.os.environ["NCCL_IB_DISABLE"] == "0"
    assert oob.os.environ["NCCL_NET_PLUGIN"] == "none"
    assert json.loads((tmp_path / f"rank{rank}.oob.json").read_text())["verified"]


def test_ambiguous_hostname_mapping_rejected():
    with pytest.raises(RuntimeError):
        oob.select_interface([interface("eth0", "10.0.0.1"),
                              interface("ib0", "10.2.0.1")], ["10.0.0.1", "10.2.0.1"])


def test_unreachable_peer_stops_before_socket_selection(monkeypatch, tmp_path):
    monkeypatch.setattr(oob.os, "environ", dict(oob.os.environ))
    monkeypatch.delenv("NCCL_SOCKET_IFNAME", raising=False)
    monkeypatch.setattr(oob, "inventory", lambda: [interface("eno1", "10.0.0.1")])
    monkeypatch.setattr(oob.socket, "getaddrinfo", lambda *a: [(0, 0, 0, "", ("10.0.0.1", 0))])
    def fail(*args):
        raise TimeoutError("peer TCP connection timed out")
    monkeypatch.setattr(oob, "exchange", fail)
    with pytest.raises(TimeoutError):
        oob.configure_oob(tmp_path, 0)
    report = json.loads((tmp_path / "rank0.oob.json").read_text())
    assert not report["verified"] and "timed out" in report["error"]
    assert report["ipv4"] == "10.0.0.1"
    assert "NCCL_SOCKET_IFNAME" not in oob.os.environ


def test_tcp_listener_unreachable(tmp_path):
    # Reserve an unlistened local port so refusal is immediate and deterministic.
    with oob.socket.socket(oob.socket.AF_INET, oob.socket.SOCK_STREAM) as closed:
        closed.bind(("127.0.0.1", 0))
        oob.atomic_json(tmp_path / "rank1.oob-ready.json",
                        dict(rank=1, ip="127.0.0.1", port=closed.getsockname()[1]))
        with pytest.raises(ConnectionRefusedError):
            oob.exchange(tmp_path, 0, "127.0.0.1", timeout=1)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_real_cross_node_rank_pairs(tmp_path, world):
    half = world // 2
    with ThreadPoolExecutor(max_workers=world) as pool:
        futures = [pool.submit(oob.exchange, tmp_path, rank, "127.0.0.1", 3,
                               (rank + half) % world) for rank in range(world)]
        peers = [future.result(timeout=5) for future in futures]
    assert [peer["rank"] for peer in peers] == [(rank + half) % world for rank in range(world)]
