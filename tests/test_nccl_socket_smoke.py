import importlib.util
from pathlib import Path
import struct
import pytest

spec = importlib.util.spec_from_file_location("diag", Path(__file__).parents[1] / "scripts/nccl_socket_smoke.py")
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)

@pytest.mark.parametrize("up", [True, False])
def test_effective_config_and_interface(monkeypatch, capsys, up):
    for key, value in dict(NCCL_NET="Socket", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="1",
                           NCCL_COLLNET_ENABLE="0", NCCL_DMABUF_ENABLE="0", NCCL_DEBUG="INFO",
                           NCCL_SOCKET_IFNAME="=ib0").items():
        monkeypatch.setenv(key, value)
    def ioctl(fd, operation, request):
        assert request.startswith(b"ib0")
        if operation == 0x8913:
            return bytes(16) + struct.pack("H", int(up)) + bytes(238)
        return bytes(20) + bytes([10, 203, 2, 207]) + bytes(232)
    monkeypatch.setattr(diag.fcntl, "ioctl", ioctl)
    if not up:
        with pytest.raises(RuntimeError, match="must be up"):
            diag.check_configuration()
        return
    diag.check_configuration()
    assert "ulimit -l" in capsys.readouterr().out
    monkeypatch.setenv("NCCL_NET", "SPCX")
    with pytest.raises(RuntimeError, match="NCCL_NET"):
        diag.check_configuration()
