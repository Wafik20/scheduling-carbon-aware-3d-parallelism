"""Slurm-sized Megatron data-parallel smoke test."""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")
    return args


def slurm_ranks(environ):
    rank = int(environ["SLURM_PROCID"])
    world_size = int(environ["SLURM_NTASKS"])
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Expected positive Slurm world size and a valid rank (TP1 PP1)")
    if not environ.get("MASTER_ADDR") or not environ.get("MASTER_PORT"):
        raise ValueError("MASTER_ADDR and MASTER_PORT must be exported by the launcher")
    return rank, world_size


def select_cuda_device(cuda, local_rank):
    visible_device_count = cuda.device_count()
    cuda_device = 0 if visible_device_count == 1 else local_rank
    if local_rank < 0 or not 0 <= cuda_device < visible_device_count:
        raise RuntimeError("Slurm local rank cannot map to a visible CUDA device")
    cuda.set_device(cuda_device)
    return cuda_device, visible_device_count


def average_gradients(model, dist, group, world_size):
    """Synchronize gradients before AdamW updates replicated parameters."""
    for parameter in model.parameters():
        if parameter.grad is not None:
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=group)
            parameter.grad.div_(world_size)


def synchronized_stop(torch, dist, rank, start, duration):
    elapsed = torch.tensor([time.perf_counter() - start], dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MIN)
    stop = torch.tensor(
        [int(rank == 0 and elapsed.item() >= duration)],
        dtype=torch.int32, device="cuda",
    )
    dist.broadcast(stop, src=0)
    return bool(stop.item())


def write_result(result_dir, rank, steps, duration, world_size, local_rank, cuda_device, visible_device_count):
    result_dir.mkdir(parents=True, exist_ok=True)
    hostname = os.environ.get("SLURMD_NODENAME") or socket.gethostname().split(".")[0]
    result = dict(rank=rank, hostname=hostname, steps=steps, duration=duration,
                  elapsed_s=duration, world_size=world_size, local_rank=local_rank,
                  cuda_device=cuda_device, visible_device_count=visible_device_count)
    (result_dir / f"rank{rank}.json").write_text(json.dumps(result) + "\n")


def main(argv=None):
    args = parse_args(argv)
    rank, world_size = slurm_ranks(os.environ)

    # Import only when training, so launcher/unit tests need no GPU libraries.
    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    local_rank = int(os.environ["SLURM_LOCALID"])
    cuda_device, visible_device_count = select_cuda_device(torch.cuda, local_rank)
    print(f"Rank {rank} | local_rank={local_rank} | cuda_device={cuda_device} | "
          f"visible_device_count={visible_device_count} | "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    torch.manual_seed(42)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(42)
        group = parallel_state.get_data_parallel_group()
        if parallel_state.get_data_parallel_world_size() != world_size:
            raise RuntimeError(f"Expected DP{world_size} TP1 PP1")

        seq_len, vocab_size, batch_size = 128, 8192, 2
        config = TransformerConfig(
            num_layers=2, hidden_size=256, num_attention_heads=4,
            ffn_hidden_size=1024, hidden_dropout=0.0, attention_dropout=0.0,
            bf16=True, params_dtype=torch.bfloat16, use_cpu_initialization=False,
        )
        model = GPTModel(
            config=config, transformer_layer_spec=get_gpt_layer_local_spec(),
            vocab_size=vocab_size, max_sequence_length=seq_len,
            pre_process=True, post_process=True, parallel_output=False,
        ).cuda().to(dtype=torch.bfloat16)
        # Explicitly ensure identical initial replicas before rank-specific data.
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0, group=group)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.manual_seed(42 + rank)
        model.train()
        positions = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.triu(
            torch.ones(1, 1, seq_len, seq_len, device="cuda", dtype=torch.bool),
            diagonal=1,
        )
        torch.cuda.synchronize()
        dist.barrier()
        start = time.perf_counter()
        steps = 0
        # Always do at least one update, even with a very short requested duration.
        while True:
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            labels = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = model(input_ids=tokens, position_ids=positions,
                               attention_mask=attention_mask, labels=labels)
                loss = losses.float().mean()
            finite = torch.isfinite(loss).to(dtype=torch.int32)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
            if not finite.item():
                raise RuntimeError("Non-finite loss detected on a rank")
            loss.backward()
            average_gradients(model, dist, group, world_size)
            optimizer.step()
            torch.cuda.synchronize()
            steps += 1
            if rank == 0 and steps % 100 == 0:
                print(f"Step {steps} | Loss: {loss.item():.4f}", flush=True)
            if synchronized_stop(torch, dist, rank, start, args.duration):
                break
        dist.barrier()
        elapsed = time.perf_counter() - start
        write_result(args.result_dir, rank, steps, elapsed, world_size,
                     local_rank, cuda_device, visible_device_count)
        dist.barrier()
        print(f"Rank {rank} PASSED | steps={steps} | cuda_device={cuda_device}", flush=True)
        if rank == 0:
            print("MULTINODE MEGATRON TRAINING SMOKE TEST PASSED", flush=True)
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
