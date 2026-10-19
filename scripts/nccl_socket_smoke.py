"""Opt-in Socket diagnostic entrypoint; runs inside the training container."""
import fcntl
import os
from pathlib import Path
import resource
import runpy
import socket
import struct


def check_configuration():
    required = dict(NCCL_NET="Socket", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="1",
                    NCCL_COLLNET_ENABLE="0", NCCL_DMABUF_ENABLE="0", NCCL_DEBUG="INFO")
    for key, value in required.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"{key} must be {value!r} inside the container")
    interface = os.environ.get("NCCL_SOCKET_IFNAME", "")
    if not interface.startswith("=") or not interface[1:] or len(interface[1:].encode()) >= 16:
        raise RuntimeError("Expected one exact NCCL_SOCKET_IFNAME")
    interface = interface[1:]
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        request = struct.pack("256s", interface.encode())
        flags = struct.unpack_from("H", fcntl.ioctl(sock, 0x8913, request), 16)[0]
        if not flags & 1 or flags & 8:  # IFF_UP, IFF_LOOPBACK
            raise RuntimeError(f"{interface} must be up and non-loopback")
        address = socket.inet_ntoa(fcntl.ioctl(sock, 0x8915, request)[20:24])
    limits = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    print(f"[Socket diagnostic] host={socket.gethostname()} rank={os.environ.get('SLURM_PROCID')} "
          f"interface={interface} IPv4={address}", flush=True)
    print("[Socket diagnostic] effective NCCL: " + repr({
        key: value for key, value in sorted(os.environ.items()) if key.startswith("NCCL_")}), flush=True)
    print(f"[Socket diagnostic] ulimit -l soft/hard (bytes; -1=unlimited): {limits}", flush=True)


if __name__ == "__main__":
    check_configuration()
    runpy.run_path(str(Path(__file__).with_name("smoke_train_multinode.py")), run_name="__main__")
