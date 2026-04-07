#!/usr/bin/env python3
"""
Launcher: runs benchmark_inner.py inside the NeMo container (one launch).
"""
import subprocess
import sys
from pathlib import Path

CONTAINER = "/u/data3/wat29/nemo.sif"
SCRIPT_DIR = Path(__file__).parent.resolve()
INNER_SCRIPT = SCRIPT_DIR / "benchmark_inner.py"

def main():
    if not INNER_SCRIPT.exists():
        print(f"ERROR: {INNER_SCRIPT} not found")
        sys.exit(1)

    cmd = [
        "apptainer", "exec", "--nv",
        "--bind", "/u/data3/wat29:/u/data3/wat29",
        "--bind", f"{SCRIPT_DIR}:{SCRIPT_DIR}",
        "--bind", "/afs:/afs",
        CONTAINER,
        "python", str(INNER_SCRIPT),
    ]

    print(f"Launching container (one-time cost)...")
    print(f"  Container: {CONTAINER}")
    print(f"  Inner script: {INNER_SCRIPT}")
    print()

    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()