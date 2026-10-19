"""Check inherited memlock inside Apptainer, then run unchanged training."""
import os
from pathlib import Path
import resource
import runpy
import socket


def check_memlock():
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    def display(limit):
        return "unlimited" if limit == resource.RLIM_INFINITY else f"{limit} bytes"
    print(f"[memlock] host={socket.gethostname()} rank={os.environ.get('SLURM_PROCID')} "
          f"soft={display(soft)} hard={display(hard)}", flush=True)
    if soft != resource.RLIM_INFINITY:
        raise RuntimeError("Apptainer training task did not inherit unlimited soft memlock")


if __name__ == "__main__":
    check_memlock()
    runpy.run_path(str(Path(__file__).with_name("smoke_train_multinode.py")), run_name="__main__")
