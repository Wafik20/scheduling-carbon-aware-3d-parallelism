#!/usr/bin/env python3
"""
profile_per_layer.py
====================
Per-Layer Compute Profiling for Pipeline Efficiency

GOAL:
Explain why different architectures (LLaMA, GPT-2, Pythia) respond
differently to PP vs TP by measuring per-layer compute distribution
and imbalance.

Key insight: Pipeline efficiency is governed by the distribution of
per-layer compute, not just total FLOPs.

USAGE:
    python profile_per_layer.py --model gpt2
    python profile_per_layer.py --model llama_3b --output-dir results/
    python profile_per_layer.py --all-models
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
except ModuleNotFoundError as exc:
    F = None
    DataLoader = None
    TensorDataset = None

    class _MissingCuda:
        @staticmethod
        def is_available():
            return False

        class Event:
            def __init__(self, *args, **kwargs):
                pass

        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def empty_cache():
            pass

    class _MissingTorch:
        Tensor = object
        bfloat16 = None
        cuda = _MissingCuda()

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def randint(*args, **kwargs):
            raise ImportError("PyTorch is required for per-layer profiling")

        @staticmethod
        def ones(*args, **kwargs):
            raise ImportError("PyTorch is required for per-layer profiling")

        class no_grad:
            def __enter__(self):
                return None

            def __exit__(self, *args):
                return False

    class _MissingNN:
        class Module:
            pass

    torch = _MissingTorch()
    nn = _MissingNN()
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None


# ────────────────────────────────────────────────────────────────
# MODEL LOADING
# ────────────────────────────────────────────────────────────────

def load_gpt2_model(model_name: str = "gpt2"):
    """Load GPT-2 from HuggingFace."""
    from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

    config_map = {
        "gpt2": "gpt2",
        "gpt2_medium": "gpt2-medium",
        "gpt2_large": "gpt2-large",
        "gpt2_xl": "gpt2-xl",
    }

    pretrained_name = config_map.get(model_name, model_name)
    print(f"Loading {pretrained_name}...")

    config = GPT2Config.from_pretrained(pretrained_name)
    model = GPT2LMHeadModel.from_pretrained(pretrained_name)
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_name)

    return model, config, tokenizer


def load_llama_model(model_name: str = "openlm-research/open_llama_3b_v2"):
    """Load LLaMA (or LLaMA-architecture) from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"Loading {model_name}...")

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, config, tokenizer


def load_pythia_model(model_name: str = "EleutherAI/pythia-410m"):
    """Load Pythia from HuggingFace."""
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

    print(f"Loading {model_name}...")

    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, config, tokenizer


# ────────────────────────────────────────────────────────────────
# INSTRUMENTATION (hook-based — robust to HF internal changes)
# ────────────────────────────────────────────────────────────────

class InstrumentedTransformer:
    """
    Wrapper that instruments a transformer model for per-layer timing.

    Uses forward pre/post hooks with cuda.Event timing, so we never
    have to manually walk model internals (which break across HF
    transformers releases — especially for GPT-2's attention mask
    handling and Pythia/GPT-NeoX's RoPE-at-model-level refactor).
    """

    def __init__(self, model: nn.Module, model_type: str = "gpt2"):
        self.model = model
        self.model_type = model_type
        self.device = next(model.parameters()).device
        self.layer_times = []  # List of lists: [iteration][layer] = time_ms
        self.current_iteration_times = None
        self._hooks = []
        self._start_events: Dict[str, list] = {}
        self._end_events: Dict[str, list] = {}
        self._module_order: List[str] = []
        self._setup_hooks()

    def _resolve_modules(self):
        """Locate the embedding, transformer-block list, final norm, and LM head."""
        m = self.model
        if self.model_type.startswith("gpt2"):
            blocks = m.transformer.h
            embed = m.transformer.wte
            norm = m.transformer.ln_f
            head = m.lm_head
            prefix = "gpt2"
        elif "llama" in self.model_type.lower():
            blocks = m.model.layers
            embed = m.model.embed_tokens
            norm = m.model.norm
            head = m.lm_head
            prefix = "llama"
        elif "pythia" in self.model_type.lower():
            blocks = m.gpt_neox.layers
            embed = m.gpt_neox.embed_in
            # HF renamed this attribute over time; accept either.
            norm = getattr(m.gpt_neox, "final_layer_norm",
                           getattr(m.gpt_neox, "final_ln", None))
            head = m.embed_out
            prefix = "pythia"
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.layer_names = [f"{prefix}_layer_{i}" for i in range(len(blocks))]
        return embed, blocks, norm, head

    def _setup_hooks(self):
        embed, blocks, norm, head = self._resolve_modules()
        self.num_layers = len(blocks)

        named = [("embeddings", embed)]
        named += [(f"layer_{i}", b) for i, b in enumerate(blocks)]
        if norm is not None:
            named.append(("final_ln", norm))
        named.append(("lm_head", head))

        self._module_order = [n for n, _ in named]

        for name, mod in named:
            self._start_events[name] = []
            self._end_events[name] = []

            def make_pre(n):
                def pre_hook(_m, _inp):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    self._start_events[n].append(e)
                return pre_hook

            def make_post(n):
                def post_hook(_m, _inp, _out):
                    e = torch.cuda.Event(enable_timing=True)
                    e.record()
                    self._end_events[n].append(e)
                return post_hook

            self._hooks.append(mod.register_forward_pre_hook(make_pre(name)))
            self._hooks.append(mod.register_forward_hook(make_post(name)))

        print(f"Instrumented {self.num_layers} layers")

    def forward(self, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        # Reset event buffers for this iteration
        for n in self._module_order:
            self._start_events[n].clear()
            self._end_events[n].clear()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Make sure all events have completed before reading elapsed times
        torch.cuda.synchronize()

        self.current_iteration_times = []
        for n in self._module_order:
            starts, ends = self._start_events[n], self._end_events[n]
            if starts and ends:
                # If a module fires more than once in a forward pass, sum.
                total = sum(s.elapsed_time(e) for s, e in zip(starts, ends))
            else:
                total = 0.0
            self.current_iteration_times.append(total)
        return outputs

    def record_iteration_times(self):
        """Store timing data for current iteration."""
        if self.current_iteration_times:
            self.layer_times.append(self.current_iteration_times.copy())

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ────────────────────────────────────────────────────────────────
# FLOPS ESTIMATION
# ────────────────────────────────────────────────────────────────

def _intermediate_size(config) -> int:
    """Best-effort fetch of MLP intermediate size across HF configs."""
    return (
        getattr(config, "intermediate_size", None)
        or getattr(config, "n_inner", None)
        or 4 * config.hidden_size
    )


def estimate_layer_flops(
    layer_idx: int,
    model_type: str,
    config,
    seq_len: int,
    batch_size: int,
) -> int:
    """Estimate FLOPs for a single transformer layer."""
    if model_type.startswith("gpt2"):
        return _estimate_gpt2_layer_flops(config, seq_len, batch_size)
    elif "llama" in model_type.lower():
        return _estimate_llama_layer_flops(config, seq_len, batch_size)
    elif "pythia" in model_type.lower():
        return _estimate_pythia_layer_flops(config, seq_len, batch_size)
    return 0


def _estimate_gpt2_layer_flops(config, seq_len: int, batch_size: int) -> int:
    """Estimate FLOPs for a single GPT-2 layer."""
    B, S = batch_size, seq_len
    H = config.hidden_size
    inter = _intermediate_size(config)

    # Q, K, V projections
    attn_flops = 3 * 2 * B * S * H * H
    # Attention compute (rough)
    attn_compute = B * S * S * H // max(1, config.num_attention_heads)
    # Attention output
    attn_output = 2 * B * S * S * (H // max(1, config.num_attention_heads)) * config.num_attention_heads
    # MLP
    mlp_flops = 2 * 2 * B * S * H * inter

    return attn_flops + attn_compute + attn_output + mlp_flops


def _estimate_llama_layer_flops(config, seq_len: int, batch_size: int) -> int:
    """Estimate FLOPs for a single LLaMA layer."""
    B, S = batch_size, seq_len
    H = config.hidden_size
    inter = _intermediate_size(config)

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", None) or num_heads

    qkv_flops = 2 * B * S * H * (H + 2 * H * num_kv_heads // num_heads)
    attn_compute = B * S * S * H // max(1, num_heads)
    attn_output = 2 * B * S * S * (H // num_heads) * num_heads
    mlp_flops = 2 * 2 * B * S * H * inter

    return qkv_flops + attn_compute + attn_output + mlp_flops


def _estimate_pythia_layer_flops(config, seq_len: int, batch_size: int) -> int:
    """Estimate FLOPs for a single Pythia layer."""
    B, S = batch_size, seq_len
    H = config.hidden_size
    inter = _intermediate_size(config)

    qkv_flops = 3 * 2 * B * S * H * H
    attn_compute = B * S * S * H // max(1, config.num_attention_heads)
    attn_output = 2 * B * S * S * (H // config.num_attention_heads) * config.num_attention_heads
    mlp_flops = 2 * 2 * B * S * H * inter

    return qkv_flops + attn_compute + attn_output + mlp_flops


# ────────────────────────────────────────────────────────────────
# DATA GENERATION
# ────────────────────────────────────────────────────────────────

def create_dummy_data(
    batch_size: int,
    seq_len: int,
    num_iters: int,
    vocab_size: int,
    device: torch.device = "cuda",
) -> DataLoader:
    """Create dummy data for profiling."""
    input_ids = torch.randint(0, vocab_size, (num_iters, batch_size, seq_len), device=device)
    attention_mask = torch.ones((num_iters, batch_size, seq_len), device=device)

    dataset = TensorDataset(input_ids, attention_mask)
    loader = DataLoader(dataset, batch_size=1)

    return loader


# ────────────────────────────────────────────────────────────────
# PROFILING
# ────────────────────────────────────────────────────────────────

def _safe_num_layers(config) -> int:
    """HF configs disagree on the layer-count attribute name."""
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        v = getattr(config, attr, None)
        if v is not None:
            return v
    return -1


def _segment_stats(
    layer_times_array: np.ndarray,
    module_order: List[str],
) -> Dict:
    """
    Split per-module stats into:
      - 'blocks': transformer blocks only (the thing that matters for PP balance)
      - 'embeddings', 'final_ln', 'lm_head': head/tail modules, each with a
         ratio_to_block_mean so you can see how big they are vs. one block.

    The "all modules" CV/Max-Min in stats['global'] is dominated by the
    lm_head matmul vs. the tiny embedding lookup. That's a real fact about
    pipeline stage balancing, but it's NOT the architectural block-imbalance
    story — for that, use stats['segments']['blocks']['cv'].
    """
    name_to_idx = {n: i for i, n in enumerate(module_order)}
    block_idxs = [i for i, n in enumerate(module_order) if n.startswith("layer_")]

    if not block_idxs:
        return {}

    block_times = layer_times_array[:, block_idxs]                    # (iters, num_blocks)
    block_mean_per_layer = np.mean(block_times, axis=0)               # (num_blocks,)

    block_global_mean = float(np.mean(block_mean_per_layer))
    block_global_std  = float(np.std(block_mean_per_layer))
    block_min         = float(np.min(block_mean_per_layer))
    block_max         = float(np.max(block_mean_per_layer))

    segments = {
        "blocks": {
            "num_blocks": len(block_idxs),
            "mean": block_global_mean,
            "std": block_global_std,
            "min": block_min,
            "max": block_max,
            "cv": (block_global_std / block_global_mean) if block_global_mean > 0 else float("inf"),
            "max_min_ratio": (block_max / block_min) if block_min > 0 else float("inf"),
            "variance": float(np.var(block_mean_per_layer)),
            "per_block_mean": block_mean_per_layer.tolist(),
        }
    }

    # Steady-state blocks: excludes block 0, which carries one-time setup
    # cost (causal-mask materialization, RoPE buffer init, etc.) attributed
    # to it by the hook. This is the cleanest architectural imbalance metric:
    # how much do transformer blocks vary once the model is in steady state?
    if len(block_idxs) >= 2:
        inner = block_mean_per_layer[1:]
        inner_mean = float(np.mean(inner))
        inner_std  = float(np.std(inner))
        inner_min  = float(np.min(inner))
        inner_max  = float(np.max(inner))
        segments["blocks_inner"] = {
            "num_blocks": len(inner),
            "excluded": "block_0",
            "mean": inner_mean,
            "std": inner_std,
            "min": inner_min,
            "max": inner_max,
            "cv": (inner_std / inner_mean) if inner_mean > 0 else float("inf"),
            "max_min_ratio": (inner_max / inner_min) if inner_min > 0 else float("inf"),
            "block0_overhead_ratio": (block_mean_per_layer[0] / inner_mean) if inner_mean > 0 else float("inf"),
        }

    # Head/tail modules: report mean and ratio relative to a single block.
    for name in ("embeddings", "final_ln", "lm_head"):
        if name in name_to_idx:
            idx = name_to_idx[name]
            module_times = layer_times_array[:, idx]
            mu = float(np.mean(module_times))
            segments[name] = {
                "mean": mu,
                "std": float(np.std(module_times)),
                "ratio_to_block_mean": (mu / block_global_mean) if block_global_mean > 0 else float("inf"),
            }

    return segments


def profile_model(
    model_name: str,
    model_type: str = "gpt2",
    batch_size: int = 1,
    seq_len: int = 512,
    total_steps: int = 200,
    warmup_steps: int = 50,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[List, Dict]:
    """Profile per-layer compute distribution."""
    print(f"\n{'='*70}")
    print(f"Profiling {model_name}")
    print(f"{'='*70}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")

    # Load model
    if model_type.startswith("gpt2"):
        model, config, tokenizer = load_gpt2_model(model_name)
    elif "llama" in model_type.lower():
        model, config, tokenizer = load_llama_model(model_name)
    elif "pythia" in model_type.lower():
        model, config, tokenizer = load_pythia_model(model_name)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)
    model.eval()

    # Instrument model
    instrumented = InstrumentedTransformer(model, model_type)

    try:
        # Create data loader
        vocab_size = config.vocab_size
        loader = create_dummy_data(batch_size, seq_len, total_steps, vocab_size, device)

        print(f"\nRunning forward passes...")

        # Forward passes
        with torch.no_grad():
            for step, (input_ids, attention_mask) in enumerate(loader):
                # Squeeze batch dimension from DataLoader
                input_ids = input_ids.squeeze(0)
                attention_mask = attention_mask.squeeze(0)

                _ = instrumented.forward(input_ids, attention_mask)
                instrumented.record_iteration_times()

                if (step + 1) % 50 == 0:
                    print(f"  Completed {step + 1}/{total_steps} steps")
    finally:
        instrumented.remove_hooks()

    # Discard warmup steps
    layer_times_no_warmup = instrumented.layer_times[warmup_steps:]

    print(f"Discarded {warmup_steps} warmup steps")
    print(f"Collecting data from {len(layer_times_no_warmup)} iterations")

    # Compute statistics
    layer_times_array = np.array(layer_times_no_warmup)  # shape: (iters, layers)

    stats = {
        "model_name": model_name,
        "model_type": model_type,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "num_layers": instrumented.num_layers,
        "config": {
            "hidden_size": config.hidden_size,
            "num_layers": _safe_num_layers(config),
            "num_attention_heads": config.num_attention_heads,
            "vocab_size": config.vocab_size,
        },
    }

    # Per-layer statistics
    mean_layer = np.mean(layer_times_array, axis=0)
    std_layer = np.std(layer_times_array, axis=0)
    min_layer = np.min(layer_times_array, axis=0)
    max_layer = np.max(layer_times_array, axis=0)

    global_mean = np.mean(mean_layer)
    global_std = np.std(mean_layer)
    global_min = np.min(mean_layer)
    global_max = np.max(mean_layer)

    stats["per_layer"] = {
        "mean": mean_layer.tolist(),
        "std": std_layer.tolist(),
        "min": min_layer.tolist(),
        "max": max_layer.tolist(),
    }

    stats["global"] = {
        "mean": float(global_mean),
        "std": float(global_std),
        "min": float(global_min),
        "max": float(global_max),
        "variance": float(np.var(mean_layer)),
        "max_min_ratio": float(global_max / global_min) if global_min > 0 else float("inf"),
        "cv": float(global_std / global_mean) if global_mean > 0 else float("inf"),
    }

    # Segmented stats: blocks-only, plus head/tail (embeddings, final_ln, lm_head).
    stats["module_order"] = list(instrumented._module_order)
    stats["segments"] = _segment_stats(layer_times_array, instrumented._module_order)

    print(f"\nGlobal Statistics (ALL modules — includes lm_head, dominates CV):")
    print(f"  Mean module time: {global_mean:.4f} ms")
    print(f"  Std dev:          {global_std:.4f} ms")
    print(f"  Min module time:  {global_min:.4f} ms")
    print(f"  Max module time:  {global_max:.4f} ms")
    print(f"  Max/Min ratio:    {stats['global']['max_min_ratio']:.2f}x")
    print(f"  Coeff. var:       {stats['global']['cv']:.4f}")

    if "blocks" in stats["segments"]:
        b = stats["segments"]["blocks"]
        print(f"\nBlock-only Statistics ({b['num_blocks']} transformer blocks, "
              f"excludes embeddings/final_ln/lm_head):")
        print(f"  Block mean:    {b['mean']:.4f} ms")
        print(f"  Block std:     {b['std']:.4f} ms")
        print(f"  Block min:     {b['min']:.4f} ms")
        print(f"  Block max:     {b['max']:.4f} ms")
        print(f"  Block CV:      {b['cv']:.4f}")
        print(f"  Block Max/Min: {b['max_min_ratio']:.2f}x")

        if "blocks_inner" in stats["segments"]:
            bi = stats["segments"]["blocks_inner"]
            print(f"\nSteady-State Block Statistics ({bi['num_blocks']} blocks, "
                  f"excludes {bi['excluded']} — cleanest architectural metric):")
            print(f"  Inner block CV:      {bi['cv']:.5f}")
            print(f"  Inner Max/Min:       {bi['max_min_ratio']:.4f}x")
            print(f"  Block-0 overhead:    {bi['block0_overhead_ratio']:.4f}x "
                  f"(one-time setup attributed to block 0)")

        print(f"\nHead/Tail Modules (vs. one transformer block):")
        for name in ("embeddings", "final_ln", "lm_head"):
            if name in stats["segments"]:
                s = stats["segments"][name]
                print(f"  {name:11s} {s['mean']:8.4f} ms   "
                      f"({s['ratio_to_block_mean']:6.2f}x block mean)")

    return layer_times_array, stats


# ────────────────────────────────────────────────────────────────
# OUTPUT GENERATION
# ────────────────────────────────────────────────────────────────

def save_summary_table(results: Dict[str, Dict], output_file: str):
    """
    Save summary statistics to CSV.

    Columns are split into three groups:
      - All-module CV / Max-Min  (legacy; dominated by lm_head — see segments)
      - Block-only CV / Max-Min  (the metric you actually want for PP balance)
      - Head/tail ratios         (how big embeddings, final_ln, lm_head are
                                  relative to one transformer block)
    """
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Model",
            "Num Blocks",
            # All-module (legacy)
            "All Mean (ms)",
            "All CV",
            "All Max/Min",
            # Block-only (the honest metric)
            "Block Mean (ms)",
            "Block Std (ms)",
            "Block Min (ms)",
            "Block Max (ms)",
            "Block CV",
            "Block Max/Min",
            # Steady-state blocks (excludes block 0)
            "Inner Block CV",
            "Inner Block Max/Min",
            "Block0 Overhead",
            # Head/tail ratios (× block mean)
            "Embed × block",
            "Final LN × block",
            "LM Head × block",
        ])

        for model_name, stats in results.items():
            g = stats["global"]
            seg = stats.get("segments", {})
            b = seg.get("blocks", {})
            bi = seg.get("blocks_inner", {})

            def _ratio(name):
                d = seg.get(name)
                if not d:
                    return ""
                r = d.get("ratio_to_block_mean")
                return f"{r:.4f}" if r is not None and np.isfinite(r) else ""

            writer.writerow([
                model_name,
                b.get("num_blocks", stats.get("num_layers", "")),
                # all-modules
                f"{g['mean']:.4f}",
                f"{g['cv']:.4f}",
                f"{g['max_min_ratio']:.2f}",
                # block-only
                f"{b.get('mean', float('nan')):.4f}" if b else "",
                f"{b.get('std',  float('nan')):.4f}" if b else "",
                f"{b.get('min',  float('nan')):.4f}" if b else "",
                f"{b.get('max',  float('nan')):.4f}" if b else "",
                f"{b.get('cv',   float('nan')):.4f}" if b else "",
                f"{b.get('max_min_ratio', float('nan')):.2f}" if b else "",
                # steady-state blocks
                f"{bi.get('cv', float('nan')):.5f}" if bi else "",
                f"{bi.get('max_min_ratio', float('nan')):.4f}" if bi else "",
                f"{bi.get('block0_overhead_ratio', float('nan')):.4f}" if bi else "",
                # head/tail ratios
                _ratio("embeddings"),
                _ratio("final_ln"),
                _ratio("lm_head"),
            ])

    print(f"Summary table saved to {output_file}")


def save_per_layer_times(
    layer_times_array: np.ndarray,
    stats: Dict,
    output_file: str,
):
    """Save per-layer timing data."""
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)

        num_layers = layer_times_array.shape[1] if layer_times_array.ndim == 2 else stats["num_layers"]
        header = ["iteration"] + [f"slot_{i}" for i in range(num_layers)]
        writer.writerow(header)

        for i, times in enumerate(layer_times_array):
            row = [i] + list(times)
            writer.writerow(row)

    print(f"Per-layer times saved to {output_file}")


def plot_per_layer_distribution(
    results: Dict[str, Tuple[np.ndarray, Dict]],
    output_file: str,
):
    """Generate per-layer distribution plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    num_models = len(results)
    fig, axes = plt.subplots(num_models, 1, figsize=(14, 5 * num_models))

    if num_models == 1:
        axes = [axes]

    for ax, (model_name, (layer_times_array, stats)) in zip(axes, results.items()):
        mean_layer = np.mean(layer_times_array, axis=0)
        std_layer = np.std(layer_times_array, axis=0)

        x = np.arange(len(mean_layer))

        ax.bar(x, mean_layer, yerr=std_layer, capsize=5, alpha=0.7, label="Mean ± Std")
        ax.set_xlabel("Module Index (embeddings, blocks..., final_ln, lm_head)")
        ax.set_ylabel("Forward Time (ms)")
        ax.set_title(
            f"{model_name}\n(CV={stats['global']['cv']:.3f}, "
            f"Max/Min={stats['global']['max_min_ratio']:.2f}x)"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_model_comparison(
    results: Dict[str, Dict],
    output_file: str,
):
    """
    Compare all-module CV and Max/Min across models.

    NOTE: these "all-module" numbers are dominated by the lm_head matmul vs.
    the embedding lookup. They are useful for end-to-end PP stage balance but
    do NOT reflect block-to-block compute imbalance. See plot_block_only_comparison
    for the cleaner architectural metric.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    models = list(results.keys())
    cvs = [results[m]["global"]["cv"] for m in models]
    max_mins = [results[m]["global"]["max_min_ratio"] for m in models]

    ax1.bar(models, cvs, alpha=0.7, color="steelblue")
    ax1.set_ylabel("Coefficient of Variation")
    ax1.set_title("All-Module Imbalance — CV (dominated by lm_head)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(models, max_mins, alpha=0.7, color="coral")
    ax2.set_ylabel("Max/Min Ratio")
    ax2.set_title("All-Module Range — Max/Min (dominated by lm_head)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved to {output_file}")
    plt.close()


def plot_block_only_comparison(
    results: Dict[str, Dict],
    output_file: str,
):
    """
    Block-only CV and Max/Min across models, with a second pair of bars for
    "inner blocks" (excludes block 0, which carries one-time setup cost).

    The gap between block CV and inner-block CV measures how much per-forward
    setup work (mask materialization, RoPE buffer init, etc.) gets attributed
    to block 0 by the hook timer.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    models = []
    cvs, max_mins = [], []
    inner_cvs, inner_max_mins = [], []
    for m, stats in results.items():
        b = stats.get("segments", {}).get("blocks")
        if not b:
            continue
        models.append(m)
        cvs.append(b["cv"])
        max_mins.append(b["max_min_ratio"])
        bi = stats.get("segments", {}).get("blocks_inner", {})
        inner_cvs.append(bi.get("cv", 0.0))
        inner_max_mins.append(bi.get("max_min_ratio", 1.0))

    if not models:
        print("No block-only stats available; skipping block-only plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    width = 0.4
    x = np.arange(len(models))

    ax1.bar(x - width/2, cvs, width=width, alpha=0.85, color="seagreen",
            label="All blocks")
    ax1.bar(x + width/2, inner_cvs, width=width, alpha=0.85, color="darkgreen",
            label="Inner blocks (excl. block 0)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel("Coefficient of Variation")
    ax1.set_title("Block Compute Imbalance (CV)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x - width/2, max_mins, width=width, alpha=0.85, color="darkorange",
            label="All blocks")
    ax2.bar(x + width/2, inner_max_mins, width=width, alpha=0.85, color="saddlebrown",
            label="Inner blocks (excl. block 0)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45)
    ax2.set_ylabel("Max/Min Ratio")
    ax2.set_title("Block Range (Max/Min)")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Block-only comparison plot saved to {output_file}")
    plt.close()


def plot_head_tail_breakdown(
    results: Dict[str, Dict],
    output_file: str,
):
    """
    Grouped bar chart: for each model, show how big embeddings / final_ln /
    lm_head are relative to one transformer block (ratio = module_mean / block_mean).

    A bar at 1.0 means "this module costs about as much as one block."
    >> 1.0 means "this module dominates" (typical for lm_head with large vocab).
    << 1.0 means "negligible" (typical for embeddings = lookup, final_ln = single norm).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plots")
        return

    modules = ["embeddings", "final_ln", "lm_head"]
    models = list(results.keys())

    # ratios[model][module]
    ratios = {m: [] for m in models}
    for m in models:
        seg = results[m].get("segments", {})
        for name in modules:
            r = seg.get(name, {}).get("ratio_to_block_mean", 0.0)
            ratios[m].append(r)

    n_models = len(models)
    n_mods = len(modules)
    width = 0.8 / max(1, n_models)
    x = np.arange(n_mods)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue", "seagreen", "darkorange", "purple", "crimson"]
    for i, m in enumerate(models):
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, ratios[m], width=width, label=m,
               color=colors[i % len(colors)], alpha=0.85)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6,
               label="= one block")
    ax.set_xticks(x)
    ax.set_xticklabels(modules)
    ax.set_ylabel("Ratio to block mean (× one transformer block)")
    ax.set_title("Head/Tail Module Cost Relative to One Transformer Block")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y", which="both")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Head/tail breakdown plot saved to {output_file}")
    plt.close()


# ────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-layer compute profiling for pipeline efficiency analysis"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "gpt2", "gpt2_xl", "pythia_2_8b", "llama_3b", "llama_7b",
        ],
        help="Model to profile",
    )
    parser.add_argument("--all-models", action="store_true", help="Profile all models")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--total-steps", type=int, default=200, help="Total forward passes")
    parser.add_argument("--warmup-steps", type=int, default=50, help="Warmup steps to discard")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/layer_profiles",
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch is not None and torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    args = parser.parse_args()
    if _TORCH_IMPORT_ERROR is not None:
        raise ImportError(
            "PyTorch is required for per-layer profiling. "
            "Install the profiling extras with `python -m pip install -e '.[profile]'` "
            "or install a cluster-specific CUDA/PyTorch build."
        ) from _TORCH_IMPORT_ERROR

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Architecture-comparison set, roughly size-matched (1.5B – 7B):
    #   gpt2-xl              — classic GPT-2 (serial attn→MLP, GELU, MHA, no RoPE)
    #   pythia-2.8b-deduped  — GPT-NeoX (parallel attn‖MLP, GELU, RoPE)
    #   open_llama_3b_v2     — LLaMA arch (SwiGLU, RMSNorm, RoPE, MHA at 3B)
    #   huggyllama/llama-7b  — LLaMA arch at 7B (same as above, wider+deeper).
    #                          Tests the "wide-and-shallow + small vocab is
    #                          friendly to PP/TP" hypothesis at larger scale.
    #
    # All ungated. Swap llama_7b to meta-llama/Llama-2-7b-hf if you have access
    # (architecturally identical at 7B — both are MHA, vocab=32000, H=4096).
    models_to_profile = {
        "gpt2":        ("gpt2",                             "gpt2"),
        "gpt2_xl":     ("gpt2-xl",                          "gpt2"),
        "pythia_2_8b": ("EleutherAI/pythia-2.8b-deduped",   "pythia"),
        "llama_3b":    ("openlm-research/open_llama_3b_v2", "llama"),
        "llama_7b":    ("huggyllama/llama-7b",              "llama"),
    }

    if args.all_models:
        selected_models = models_to_profile
    elif args.model:
        if args.model in models_to_profile:
            selected_models = {args.model: models_to_profile[args.model]}
        else:
            print(f"Unknown model: {args.model}")
            sys.exit(1)
    else:
        selected_models = {"gpt2": models_to_profile["gpt2"]}

    results = {}
    all_layer_times = {}

    for model_key, (model_name, model_type) in selected_models.items():
        try:
            layer_times_array, stats = profile_model(
                model_name=model_name,
                model_type=model_type,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                total_steps=args.total_steps,
                warmup_steps=args.warmup_steps,
                device=device,
            )

            results[model_key] = stats
            all_layer_times[model_key] = layer_times_array

            # Save full per-module times (incl. embeddings, final_ln, lm_head)
            per_layer_file = output_dir / f"{model_key}_layer_times.csv"
            save_per_layer_times(layer_times_array, stats, str(per_layer_file))

            # Also save a block-only CSV (for clean downstream analysis)
            module_order = stats.get("module_order", [])
            block_idxs = [i for i, n in enumerate(module_order) if n.startswith("layer_")]
            if block_idxs:
                block_times = layer_times_array[:, block_idxs]
                block_file = output_dir / f"{model_key}_block_times.csv"
                with open(block_file, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["iteration"] + [f"block_{i}" for i in range(len(block_idxs))])
                    for i, row in enumerate(block_times):
                        w.writerow([i] + list(row))
                print(f"Block-only times saved to {block_file}")

            # Free GPU memory between models
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error profiling {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if not results:
        print("No models successfully profiled")
        sys.exit(1)

    summary_file = output_dir / "summary.csv"
    save_summary_table(results, str(summary_file))

    json_results = {
        model: {
            "global": stats["global"],
            "segments": stats.get("segments", {}),
            "module_order": stats.get("module_order", []),
            "num_layers": stats["num_layers"],
            "config": stats["config"],
        }
        for model, stats in results.items()
    }

    json_file = output_dir / "results.json"
    with open(json_file, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to {json_file}")

    plot_per_layer_distribution(
        {k: (all_layer_times[k], results[k]) for k in results},
        str(output_dir / "per_layer_distribution.png"),
    )

    if len(results) > 1:
        plot_model_comparison(results, str(output_dir / "model_comparison.png"))
        plot_block_only_comparison(results, str(output_dir / "block_only_comparison.png"))
        plot_head_tail_breakdown(results, str(output_dir / "head_tail_breakdown.png"))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model_key in results:
        g = results[model_key]["global"]
        seg = results[model_key].get("segments", {})
        b = seg.get("blocks", {})
        bi = seg.get("blocks_inner", {})
        print(f"\n{model_key}:")
        print(f"  All-module CV:       {g['cv']:.4f}   "
              f"(Max/Min {g['max_min_ratio']:.2f}x — dominated by lm_head)")
        if b:
            print(f"  Block-only CV:       {b['cv']:.4f}   "
                  f"(Max/Min {b['max_min_ratio']:.2f}x — true architectural imbalance)")
        if bi:
            print(f"  Inner-block CV:      {bi['cv']:.5f}   "
                  f"(Max/Min {bi['max_min_ratio']:.4f}x — steady-state, excludes block 0)")
            print(f"  Block-0 overhead:    {bi['block0_overhead_ratio']:.4f}x block mean")
        if b:
            print(f"  Block mean:          {b['mean']:.4f} ms over {b['num_blocks']} blocks")
        for name in ("embeddings", "final_ln", "lm_head"):
            if name in seg:
                s = seg[name]
                print(f"  {name:11s} {s['mean']:8.4f} ms   "
                      f"({s['ratio_to_block_mean']:6.2f}x block mean)")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
