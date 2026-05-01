# Per-Layer Profiling for Carbon-Aware Parallelism Scheduling

## The question we wanted to answer

When you train a big language model on multiple GPUs, you have to decide *how* to split the work. There are three main ways:

- **Data Parallelism (DP):** every GPU has a full copy of the model and they each train on different batches of data. They sync up after each step.
- **Pipeline Parallelism (PP):** the model is sliced into chunks, and each GPU takes one chunk. Data flows through them like an assembly line — chunk 1 hands off to chunk 2, which hands off to chunk 3, and so on.
- **Tensor Parallelism (TP):** every layer of the model is split *across* GPUs. They each compute a piece of the same layer and combine results before moving on.

The carbon-aware scheduling project lives or dies on understanding *which split is most power-efficient under which conditions*. That's the practical question. But there's a deeper question underneath it: **why do different models seem to prefer different splits?** A configuration that works great for LLaMA might be wasteful for GPT-2, and vice versa. We wanted to figure out why.

## What we measured

We instrumented four transformer models and timed every single component during a forward pass — every transformer block, plus the embedding lookup at the start and the language-model head (the big classifier matrix) at the end. We ran 200 forward passes per model, threw away the first 50 to let the GPU warm up, and averaged the rest.

The four models we picked, all roughly in the 1.5B–7B parameter range so size wouldn't dominate the comparison:

| Model | Parameters | Layers | Hidden dim | Vocabulary |
|---|---|---|---|---|
| GPT-2 XL | 1.5B | 48 | 1,600 | 50,257 |
| Pythia 2.8B | 2.8B | 32 | 2,560 | 50,432 |
| OpenLLaMA 3B | 3.4B | 26 | 3,200 | 32,000 |
| LLaMA 7B | 6.7B | 32 | 4,096 | 32,000 |

Three different architectural families. GPT-2 is the classic "narrow and deep" design with a big vocabulary. The LLaMA family went "wide and shallow" with a smaller vocabulary. Pythia sits between them.

## The first surprise: the headline numbers were misleading

When we first plotted the data, the obvious metric — how uniform are the layer times? — looked like the answer. GPT-2 had a coefficient of variation of 0.24, Pythia 0.24, LLaMA 0.27. All roughly equal, all roughly bad. Story over: architectures don't matter much.

Except they do. The reason all three numbers looked similar is that **one component was dominating everything**: the language-model head at the very end. That final classifier matrix — which converts the model's internal representation into a probability over every word in the vocabulary — was so much bigger than everything else that it was drowning out any real signal from the transformer blocks themselves.

Once we separated "transformer blocks only" from "head and tail components," the picture got clearer.

## What we actually found

**Finding 1: Transformer blocks are essentially identical in compute cost.** Once we excluded the embedding, the final norm, and the LM head, every transformer block in every model took within about 1–2% of every other block's time. This was true for GPT-2, Pythia, and both LLaMA models. The little remaining variation was almost entirely a measurement artifact at the very first block (where some one-time setup work gets attributed to it). Steady-state, blocks are flat.

This matters: it means **pipeline parallelism doesn't have a balance problem inside the model body.** If you split a transformer model into N equal-sized chunks, the chunks will take essentially equal time. No assembly-line bubbles from the body.

**Finding 2: The interesting variation lives at the boundaries.** The LM head — that big final classifier — costs a wildly different amount across architectures relative to one block:

| Model | LM head cost (relative to one block) |
|---|---|
| GPT-2 XL | 2.04× — bigger than two blocks |
| Pythia 2.8B | 1.27× |
| LLaMA 3B | 0.62× — half a block |
| LLaMA 7B | 0.55× — half a block |

That's a 4× spread. It comes from a simple ratio: vocabulary size divided by hidden dimension. GPT-2 has a 50K vocabulary and a narrow 1,600-wide model, so the head matmul is huge relative to a block. LLaMA has a 32K vocabulary and a wide 4,096-dimension model, so the head is tiny relative to a block.

## What this means: the assembly line view

A GPU that's powered on but waiting around is still drawing 100+ watts. So the "lowest power" configuration is whichever one keeps GPUs *busy doing useful work* with the least idle waiting and the least communication overhead.

For pipeline parallelism, the power-waste source is *bubbles* — gaps in the assembly line where some GPUs finish their chunk early and idle while waiting for the next batch to reach them. Our finding 1 says the body of the assembly line is well-balanced for any modern transformer. Finding 2 says the *last station* is wildly different across architectures: it's the size of two normal stations for GPT-2, but only half a station for LLaMA. So under naive pipeline partitioning:

- **GPT-2-style models waste power** because the last GPU finishes way after the others, with everyone else idling.
- **LLaMA-style models barely waste any** because the last GPU's head matmul is small enough to fit comfortably in a normal stage.

For tensor parallelism, the power-waste source is different: every block has to do an all-reduce (a sync between GPUs) that's pure communication, no useful math. The math-to-talk ratio matters — bigger blocks mean more useful computation per sync. LLaMA's wider blocks (4,096 hidden dim) amortize the sync overhead better than GPT-2's narrower blocks (1,600 hidden dim).

So both effects point the same way: **wide-and-shallow models with small vocabularies are simultaneously friendlier to PP and TP**, for related but distinct reasons.

## Where we landed (and what we revised)

Our first take was a clean story: "wide-and-shallow with small vocab wins on both PP and TP, end of story." The data partly supports this, but as we added the 7B LLaMA model, the within-model block uniformity numbers got noisier — they varied by checkpoint quirks below the ~0.01 level, not just by architecture. So the cleanest part of the finding is narrower than we first claimed:

> **Inside the transformer stack, all four architectures are essentially balanced. Pipeline parallelism doesn't have to balance the body — it has to balance the head/tail boundary, and that boundary cost varies by ~4× across architectures, driven entirely by the vocabulary-to-hidden-dim ratio.**

That's the defensible, robust finding.

The other thing we revised is the conventional wisdom about "TP=4 always wins on a single node with NVLink." Our data doesn't directly measure TP overhead, but it strongly *suggests* that for narrow-and-deep models like GPT-2, the per-block all-reduce cost can dominate the per-block compute, making TP-heavy configs lose on power despite the fast intra-node links. The standard heuristic "use TP within a node, PP across nodes" is right *on average* but wrong for specific model shapes.

## What this means for the carbon-aware scheduler

For the broader project, two practical implications:

First, **TP=1 is a reasonable simplifying restriction**, especially if the scheduler needs to migrate work across regions or replicas. Tensor parallelism pins workloads to tightly-coupled nodes; pipeline parallelism is mobile across slow links. If the carbon-aware scheduler wants to redistribute work to follow clean energy, restricting to TP=1 makes that much easier — and the block-uniformity finding shows we don't pay much for the restriction.

Second, **the optimal `(DP, PP)` split within TP=1 isn't fixed.** It depends on the model's shape (which sets memory pressure and head/tail asymmetry) and on the link bandwidth profile. The scheduler should treat both as inputs, not constants.

## Caveats

A few things worth flagging:

The block-0 timing bump (~3–6% above steady state) is a measurement artifact — one-time setup like causal mask construction and RoPE buffer initialization gets charged to the first block by the timing hooks. It doesn't represent anything pipeline parallelism would actually have to balance.

We profiled at batch size 1, sequence length 512, in inference mode (no backward pass). Real training would amplify the per-block compute relative to fixed overheads, which would generally make all configurations look better but might shift the relative ranking somewhat.

We didn't directly measure power. The power-efficiency claims in this writeup are inferred from the timing data via the standard model that idle GPUs draw substantial power and communication itself draws power. Empirical power measurements (from `compute_power.py`) are needed to confirm the inferences.

Finally, modern frontier models (LLaMA 3+, DeepSeek V3, Qwen3) have larger 128K-class vocabularies and use Mixture-of-Experts architectures. Both choices change the head-asymmetry math: bigger vocabulary makes the LM head bigger again (working *against* the LLaMA-style advantage we found), and MoE introduces a third parallelism dimension entirely. Whether the findings here generalize to current-generation models is an open question and would need fresh measurements on something like LLaMA-3-8B or Qwen3-8B.