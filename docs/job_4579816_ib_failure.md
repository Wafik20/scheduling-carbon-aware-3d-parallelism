# Job 4579816: cross-node TP failure

## Confirmed
- 2 nodes x 1 GPU, TP=2 PP=1 DP=1, order tp-dp-pp.
- Rank 0: gpu-n29, GPU-6518051f-109f-3c3c-772c-d2be92b23a85, PCI 07:00.0.
- Rank 1: gpu-n30, GPU-d17fca28-1551-fbad-3cc0-8a65d4062349, PCI 8B:00.0.
- Both filesystem preflight validations passed (Slurm log lines 1011, 1020).
- Observed TP group [0,1]; PP/DP singleton groups. No duplicate GPU assignment.
- First TP ALLREDUCE failed, 1,572,864 elements, during embedding forward/warmup.
- Slurm log lines 1067–1089: IBV_WC_REM_OP_ERR(11), IBV_WC_SEND(0),
  vendor_err=137, hca mlx5_0, peer 10.203.2.209 (gpu-n30).
- runs/profile_2gpu_tp_ib_check/nccl-rank0.log:1517–1545 shows
  NET/IB/0/GDRDMA channels, send QP 58, and failing completion at
  2026-10-28 15:31:18. Actual NCCL reports 2.30.7+cuda13.3.
- Both ranks reported unlimited memlock.

## Comparison and limitations
Job 4578989 failed with the same status/vendor/HCA signature on gpu-n28/gpu-n31.
Job 4578577 completed cross-node TP on gpu-n28/gpu-n32, also with NCCL 2.30.7
and GDRDMA. This is not evidence that the version alone, mixed adapter discovery,
or the port-speed query warning causes failure. Job 4579126 instead failed
UUID validation before NCCL: that separate allocation problem did not recur here.

Rank 1's NCCL file is empty at inspection. Both saved Slurm snapshots report
scontrol unavailable inside the container. The sender error alone cannot identify
a defective HCA, fabric fault, receiver failure, or a NCCL/driver/firmware bug.
No root-cause fix is claimed or applied.

## Next diagnostic gate (not executed)
Do not repeat the full profiler unchanged. Ask cluster support to inspect receiver
kernel/driver/HCA logs and fabric counters for the timestamp and QP above. Run a
standalone two-rank NCCL all_reduce_perf using the same container/library and
allocated GPU pair, with message sizes covering the failing payload and logs
from BOTH hosts. Pair it with host-memory and GPU-memory RDMA tests to separate
fabric/verbs transport from GPUDirect RDMA and NCCL. Any HCA selection or GDR
change should be a controlled one-variable diagnostic, not a production fix.
After a standalone pass, rerun the two-GPU profiler before larger profiling.

Artifacts: slurm-profile-multinode-4579816.out and all files under
runs/profile_2gpu_tp_ib_check/ (especially NCCL logs, rank mapping, preflight).
No jobs submitted and no runtime settings changed during this investigation.
