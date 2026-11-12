# Phase 2 — Neural LMs & Transformer Architecture

Track: **interleave why + build**. For each topic — the problem it solves and the theory first, then implement the modern variant into `gpt.py` and measure the effect. Starting point: the GPT-2-era model from Phase 1 (`gpt.py`, val loss 1.62), modernized toward a Llama-style architecture.

Planned topics (each a self-contained upgrade):
1. RMSNorm — replaces LayerNorm
2. RoPE (rotary position embeddings) — replaces learned position embeddings
3. SwiGLU — replaces the ReLU feed-forward
4. Grouped-query attention + KV cache — replaces plain MHA at inference

Baseline to beat: char-level tiny-shakespeare, val loss **1.62** (n_embd 192, 6 heads, 6 layers, block 128).

---

## Open Questions

_Things I hit but didn't fully resolve. Log here instead of pushing past them._

- _(none yet)_

## Resolved

_Open questions once answered, with a one-line summary of the resolution and where the discussion lives (commit / file / date)._

- _(none yet)_

## Key Insights

_Non-obvious takeaways worth remembering. "The thing I wish someone had told me."_

- **RMSNorm = LayerNorm minus the re-centering (and bias).** LN does two things: recenter to mean 0, then rescale to unit variance. RMSNorm keeps only the rescale: `x / sqrt(mean(x²)+eps) * gain`, no mean, no subtraction, no bias. The 2019 ablation showed re-scaling is what gives LN its training-stability benefit; re-centering barely matters — so dropping it costs no quality but saves a reduction, a subtraction, and the bias params at every norm. Confirmed empirically: swapping all 3 LayerNorms → RMSNorm left val loss unchanged (1.62 → 1.61, noise). Used in Llama/T5/Mistral/Qwen. **Theme:** a lot of post-2017 progress is *removing* parts that don't earn their cost (biases are another — Llama drops them almost everywhere), not adding complexity. `nn.Parameter(torch.ones(dim))` init = identity, so the norm starts as pure normalization and learns the gain from there.

## Experiments

_What I changed, what I expected, what I observed, what I learned. Link to the script / commit._

- _(none yet)_
