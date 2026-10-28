"""Select a real IPv4 socket interface and test both TCP directions before CUDA."""
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct
import time


TIMEOUT_SECONDS = 60


def inventory():
    interfaces = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for _, name in socket.if_nameindex():
            entry = dict(ifname=name, flags=[], addr_info=[])
            interfaces.append(entry)
            request = struct.pack("256s", name.encode())
            try:
                flags = struct.unpack_from("H", fcntl.ioctl(sock, 0x8913, request), 16)[0]
                entry["flags"] = [label for bit, label in ((1, "UP"), (8, "LOOPBACK"))
                                  if flags & bit]
            except OSError as exc:
                entry["flags_error"] = str(exc)
            try:
                ip = socket.inet_ntoa(fcntl.ioctl(sock, 0x8915, request)[20:24])
                entry["addr_info"] = [dict(family="inet", local=ip)]
            except OSError as exc:
                entry["ipv4_error"] = str(exc)
    return interfaces

def select_interface(interfaces, hostname_ips, override=""):
    candidates = []
    for interface in interfaces:
        flags = interface.get("flags", [])
        if "UP" not in flags or "LOOPBACK" in flags:
            continue
        for address in interface.get("addr_info", []):
            if address.get("family") != "inet":
                continue
            ip = address["local"]
            if not ipaddress.ip_address(ip).is_loopback:
                candidates.append((interface["ifname"], ip))
    if override:
        if not override.startswith("=") or not override[1:] or "," in override:
            raise ValueError("NCCL_SOCKET_IFNAME must be one exact =interface")
        name = override[1:]
        candidates = [item for item in candidates if item[0] == name]
    else:
        candidates = [item for item in candidates if item[1] in hostname_ips]
    if len(candidates) != 1:
        raise RuntimeError(f"Need one UP non-loopback assigned IPv4 interface; "
                           f"override={override!r}, hostname_ipv4={hostname_ips}, "
                           f"matching_candidates={candidates}")
    return candidates[0]


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def exchange(result_dir, rank, ip, timeout=TIMEOUT_SECONDS, peer_rank=None):
    """Each peer binds, connects from its selected IP, and accepts once."""
    if peer_rank is None:
        peer_rank = 1 - rank
    deadline = time.monotonic() + timeout
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((ip, 0))
        listener.listen(1)
        atomic_json(result_dir / f"rank{rank}.oob-ready.json",
                    dict(rank=rank, ip=ip, port=listener.getsockname()[1]))
        peer_path = result_dir / f"rank{peer_rank}.oob-ready.json"
        while not peer_path.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("Peer did not publish OOB listener")
            time.sleep(0.1)
        peer = json.loads(peer_path.read_text())
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(max(0.001, deadline - time.monotonic()))
            client.bind((ip, 0))
            client.connect((peer["ip"], peer["port"]))
            client.sendall(bytes([rank]))
            listener.settimeout(max(0.001, deadline - time.monotonic()))
            connection, remote = listener.accept()
            with connection:
                connection.settimeout(max(0.001, deadline - time.monotonic()))
                if remote[0] != peer["ip"] or connection.recv(1) != bytes([peer_rank]):
                    raise RuntimeError("Unexpected OOB handshake peer")
                connection.sendall(b"Y")
            if client.recv(1) != b"Y":
                raise RuntimeError("Peer did not acknowledge OOB handshake")
    return peer


def configure_oob(result_dir, rank, peer_rank=None):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    report = dict(rank=rank, hostname=os.environ.get("SLURMD_NODENAME", socket.gethostname()), verified=False,
                  inherited_socket_ifname=os.environ.get("NCCL_SOCKET_IFNAME"),
                  inherited_socket_family=os.environ.get("NCCL_SOCKET_FAMILY"))
    try:
        report["interfaces"] = inventory()
        try:
            report["hostname_ipv4"] = sorted({entry[4][0] for entry in socket.getaddrinfo(
                report["hostname"], None, socket.AF_INET, socket.SOCK_STREAM)})
        except socket.gaierror as exc:
            report.update(hostname_ipv4=[], hostname_resolution_error=str(exc))
            if not os.environ.get("NCCL_SOCKET_IFNAME"):
                raise
        name, ip = select_interface(report["interfaces"], report["hostname_ipv4"],
                                    os.environ.get("NCCL_SOCKET_IFNAME", ""))
        report.update(interface=name, ipv4=ip, socket_ifname=f"={name}",
                      socket_family="AF_INET")
        atomic_json(result_dir / f"rank{rank}.oob.json", report)
        print(f"rank={rank} OOB selected {name}:{ip} AF_INET", flush=True)
        peer = (exchange(result_dir, rank, ip) if peer_rank is None else
                exchange(result_dir, rank, ip, peer_rank=peer_rank))
        os.environ.update(NCCL_SOCKET_IFNAME=f"={name}", NCCL_SOCKET_FAMILY="AF_INET")
        report.update(peer=peer, verified=True)
        print(f"rank={rank} OOB bidirectional TCP handshake passed", flush=True)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        atomic_json(result_dir / f"rank{rank}.oob.json", report)


def configure_profile_oob(work_dir):
    """Validate every task against its counterpart on the other block-placed node."""
    env = os.environ
    rank, world = int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"])
    node, local = int(env["SLURM_NODEID"]), int(env["SLURM_LOCALID"])
    per_node = world // 2
    if (int(env["SLURM_JOB_NUM_NODES"]) != 2 or world not in (2, 4, 8)
            or node not in (0, 1) or not 0 <= local < per_node
            or rank != node * per_node + local):
        raise ValueError("OOB preflight requires two block-placed nodes, 2/4/8 tasks")
    # Same invocation namespace as GPU UUID preflight; never read an old listener.
    directory = Path(work_dir).resolve() / "oob_preflight" / (
        f"{env['SLURM_JOB_ID']}.{env['SLURM_STEP_ID']}-{env['PROFILE_MAPPING_RUN']}")
    peer_rank = (1 - node) * per_node + local
    configure_oob(directory, rank, peer_rank=peer_rank)
