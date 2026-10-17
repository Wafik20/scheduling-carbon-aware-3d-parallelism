
import os
import time
from xml.parsers.expat import model

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig


def main():
    torch.manual_seed(42)
    torch.cuda.set_device(0)

    # Initialize single-GPU distributed training.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    dist.init_process_group(
        backend="nccl",
        rank=0,
        world_size=1,
    )

    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )

    # Initialize Megatron's model-parallel CUDA RNG states.
    model_parallel_cuda_manual_seed(42)

    # Tiny GPT configuration.
    seq_len = 128
    vocab_size = 8192
    batch_size = 2
    num_steps = 5

    config = TransformerConfig(
        num_layers=2,
        hidden_size=256,
        num_attention_heads=4,
        ffn_hidden_size=1024,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=False,
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=vocab_size,
        max_sequence_length=seq_len,
        pre_process=True,
        post_process=True,
        parallel_output=False,
    ).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
    )

    num_params = sum(p.numel() for p in model.parameters())

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Parameters: {num_params:,}", flush=True)
    print("Precision: BF16", flush=True)
    print(f"Training steps: {num_steps}", flush=True)

    model.train()

    for step in range(num_steps):
        start = time.perf_counter()

        tokens = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device="cuda",
        )

        positions = torch.arange(
            seq_len, device="cuda"
        ).unsqueeze(0).expand(batch_size, -1)

        labels = torch.randint(
            0, vocab_size,
            (batch_size, seq_len),
            device="cuda",
        )

        attention_mask = torch.triu(
            torch.ones(
                1, 1, seq_len, seq_len,
                device="cuda",
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with BF16 autocasting.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = model(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = losses.float().mean()

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss detected")

        # Backward pass.
        loss.backward()

        # Optimizer update.
        optimizer.step()

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        memory_gb = torch.cuda.max_memory_allocated() / 1024**3

        print(
            f"Step {step + 1}/{num_steps} | "
            f"Loss: {loss.item():.4f} | "
            f"Time: {elapsed:.3f}s | "
            f"GPU memory: {memory_gb:.2f} GB",
            flush=True,
        )

    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()

    print("\nMEGATRON TRAINING SMOKE TEST PASSED", flush=True)


if __name__ == "__main__":
    main()
