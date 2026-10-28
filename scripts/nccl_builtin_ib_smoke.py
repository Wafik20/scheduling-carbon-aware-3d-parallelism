"""Opt-in built-in IB experiment, configured after Apptainer initialization."""
import os
from pathlib import Path
import re
import runpy
import socket
import sys

if __package__:
    from .memlock_smoke import check_memlock
else:
    from memlock_smoke import check_memlock


SETTINGS = dict(NCCL_NET="IB", NCCL_NET_PLUGIN="none", NCCL_IB_DISABLE="0",
                NCCL_DEBUG="INFO",
                NCCL_DEBUG_SUBSYS="INIT,NET", NCCL_COLLNET_ENABLE="0")


def configure(log):
    # Applied inside the container before Torch/Megatron/NCCL is imported.
    os.environ.pop("NCCL_IB_HCA", None)
    os.environ.pop("APPTAINERENV_NCCL_IB_HCA", None)
    override = os.environ.get("SMOKE_NCCL_IB_HCA", "")
    if override:
        os.environ["NCCL_IB_HCA"] = override
    os.environ.update(SETTINGS)
    os.environ["NCCL_DEBUG_FILE"] = str(log)
    print(f"[built-in IB] host={socket.gethostname()} rank={os.environ['SLURM_PROCID']} "
          f"effective NCCL=" + repr({k: v for k, v in sorted(os.environ.items())
                                    if k.startswith("NCCL_")}), flush=True)


def verify_log(log):
    text = log.read_text()
    if not re.search(r"Using network IB(?:\s|$)", text):
        raise RuntimeError(f"{log}: NCCL did not report Using network IB")
    if re.search(r"(?:Loaded|Using).*?(?:SPCX|Spectrum.?X|spcx)", text, re.IGNORECASE):
        raise RuntimeError(f"{log}: external Spectrum-X plugin was loaded or used")
    adapters = [line for line in text.splitlines()
                if re.search(r"NET/IB\s*:.*Using.*\[", line)]
    if not adapters:
        raise RuntimeError(f"{log}: NCCL did not report selected IB adapters")
    for line in adapters:
        print(f"[built-in IB] selected adapters: {line}", flush=True)
    print(f"[built-in IB] verified Using network IB, no Spectrum-X load: {log}", flush=True)


def main():
    check_memlock()
    # Keep the training argument parser and all model arguments unchanged.
    result_dir = Path(sys.argv[sys.argv.index("--result-dir") + 1])
    result_dir.mkdir(parents=True, exist_ok=True)
    log = result_dir / f"nccl-rank{os.environ['SLURM_PROCID']}.log"
    configure(log)
    runpy.run_path(str(Path(__file__).with_name("smoke_train_multinode.py")), run_name="__main__")
    verify_log(log)


if __name__ == "__main__":
    main()
