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
- **KV cache: cache past K/V so each decode step is O(1), not O(T).** Naive generate re-runs the whole context every step, recomputing K/V for tokens that never change. The cache stores K/V per layer; each step projects only the *new* token's k/v, appends along the time axis, and attends the new query against all cached keys. Two phases: **prefill** (process the whole prompt once, fill caches) then **decode** (feed one token at a time). Needs a RoPE position *offset* (= cache length) since the single new token's true position isn't 0, and a general causal mask built from absolute indices (`k_idx <= q_idx`) instead of a fixed `tril` buffer. Verified correct by comparing cached incremental logits to a full parallel forward: **max diff 3.8e-6** (float noise) — the real test, since coherent output alone wouldn't catch an off-by-one in positions. This is *the* reason LLM inference is tractable.
- **GQA: query heads share K/V heads to shrink the cache.** The KV cache's size (`batch·n_kv_head·seq·head_dim·2`) often dominates GPU memory at inference — it, not compute, caps context length and batch size. MHA: n Q heads, n KV heads. MQA: n Q, 1 KV (n× smaller cache, quality dips). GQA: n Q heads share g KV heads (`1<g<n`) — most of the saving, nearly no quality loss. Impl: project K/V to `n_kv_head·head_size`, then `repeat_interleave(n_head//n_kv_head, dim=1)` to expand before the matmul. Confirmed: 6 Q / 2 KV heads gave identical val loss to full MHA (1.52) while cutting K/V projections 3×. Llama-2-70B uses 64 Q / 8 KV.
- **Removing the length cap ≠ being able to use longer contexts (extrapolation finding).** RoPE has no hard position limit (unlike a learned table), and the KV cache lets generation run arbitrarily long. But a model trained at `block_size=128` generates coherently for ~128 tokens then **collapses into gibberish** past that — positions beyond the training length are out-of-distribution for RoPE and it doesn't generalize there. The old crop-to-block_size generate *hid* this (positions never exceeded trained range); the KV-cache generate exposed it. Lesson: context-length extension is an *open problem* that needs explicit tricks (NTK-aware / YaRN frequency scaling, or long-context fine-tuning), not just a positional scheme without a cap. Clean demo: generate ≤120 tokens → coherent throughout; generate 300 → coherent then garbage at ~128.
- **SwiGLU = a learned multiplicative gate replacing ReLU.** Old FFN: `ReLU(xW1)W2`, one hard gate. SwiGLU: `(Swish(xW_gate) ⊙ xW_up) W_down` — two parallel projections, one Swish-activated branch gates (element-wise multiply) the other, then project down. Swish/SiLU = `z·σ(z)`: smooth, non-monotonic, lets a little negative through (cleaner grads than ReLU's kink). Gating gives *multiplicative* interactions between two views of the input vs ReLU's additive-then-clamp. **Param subtlety:** 3 matrices not 2, so shrink hidden from `4d` to `8d/3` (≈2.67d; d=192→512) to keep params/FLOPs matched — the win must come at equal cost, not by spending more. Result: 1.54→1.52, modest-but-real (GLU variants are a consistent ~1-2%, not dramatic like RoPE). From Shazeer 2020 "GLU Variants Improve Transformer"; standard in Llama/PaLM/Mistral/Qwen. `bias=False` per Llama convention.
- **RoPE encodes *relative* position by rotating q and k.** Instead of adding an absolute position vector at the input, rotate each 2D coordinate-pair of q/k by angle `m·θ_i` (position × per-pair frequency `θ_i = 10000^(-2i/d)`). Key identity: `R(mθ)ᵀ R(nθ) = R((n−m)θ)`, so the attention dot product `(R(mθ)q)ᵀ(R(nθ)k) = qᵀ R((n−m)θ) k` depends only on the **relative offset n−m** — absolute positions cancel out. Applied to q and k only (not v: position affects *where* to attend, not the payload), inside every head. Wins vs learned absolute embeddings, all confirmed: lower loss (1.61→1.54), **faster convergence** (step 500: 1.87 vs ~2.27 baseline), **zero params** (deleted the 24,576-param table), and **no hard length cap** (basis for context-extension tricks like NTK/YaRN). Used in Llama/GPT-NeoX/PaLM/Mistral/Qwen. Impl gotchas: `x[...,0::2]`/`x[...,1::2]` split pairs, `stack(...,-1).flatten(-2)` interleaves back; precompute freqs as a buffer so they move to GPU.

## Experiments

_What I changed, what I expected, what I observed, what I learned. Link to the script / commit._

- **Modernization ladder (char tiny-shakespeare, val loss).** GPT-2-era baseline 1.62 → +RMSNorm 1.61 (free simplification) → +RoPE 1.54 (real win: lower loss, faster convergence, fewer params, no length cap) → +SwiGLU 1.52 (modest, matched cost) → batched-attention refactor 1.52 (equivalent) → +GQA 1.52 (same quality, 3× smaller KV) → +KV cache 1.52 (inference-only, training unchanged). Net: a Llama-style architecture, every piece hand-built and measured. Commits tagged `phase2:` in git history.
- **KV cache correctness test.** Compared full parallel forward vs incremental cached decode on a 50-token sequence → max logit diff 3.8e-6. Confirms the cache/offset/mask are right. See `check_cache()` in `gpt.py`.
- **RoPE extrapolation failure (observed).** Generated 300 tokens from a model trained at block_size=128 → coherent for ~128, then gibberish. Expected coherent throughout, observed a sharp break at the training length. Learned: RoPE lifts the *hard* cap but the model still can't *use* positions it never trained on. See the extrapolation Key Insight above.
