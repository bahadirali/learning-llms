# Phase 1 — Foundations

Math + PyTorch foundations via Karpathy's Zero to Hero. Working order:

1. micrograd — autograd as a DAG of local gradients ← **starting now**
2. makemore part 1 (bigrams)
3. makemore part 2 (MLP, Bengio 2003)
4. makemore part 3 (BatchNorm, activations, gradients, init)
5. makemore part 4 (manual backprop) — optional
6. makemore part 5 (WaveNet)
7. Build GPT from scratch — bridge into Phase 2

Math refreshers to weave in:
- Cross-entropy as −𝔼[log p(x)] and equivalence to NLL for one-hot targets
- Softmax Jacobian
- Broadcasting semantics in PyTorch tensor ops

---

## Open Questions

_Things I hit but didn't fully resolve. Log here instead of pushing past them._

- _(none yet)_

## Resolved

_Open questions once answered, with a one-line summary of the resolution and where the discussion lives (commit / file / date)._

- _(none yet)_

## Key Insights

_Non-obvious takeaways worth remembering. Aim for "the thing I wish someone had told me" rather than restating what was in the video._

- **When gradients look wrong, sanity-check the forward first.** Hit this debugging `tanh` in micrograd — silently used `math.tan` (trig) instead of `math.tanh` (hyperbolic). The local-derivative formula `1 − out.x²` was correct; the upstream `out.x` was garbage. Generalizable rule: before suspecting backward, print the forward output and check against a known value (e.g. `tanh(0.8) ≈ 0.6640`). Cheaper than reasoning about chain rule.
- **Count-based MLE and gradient descent are two roads to the same model.** The count bigram normalizes `N` row-wise to get `P`; a 27→27 neural net (`bigram_nn.py`) trained on NLL converges to the *same* ~2.45 loss. Not a coincidence: normalizing counts *is* the closed-form maximum-likelihood estimate (provable via a Lagrange multiplier on the per-row `Σθ=1` constraint), and the net is minimizing that same NLL by gradient descent — so it rediscovers the MLE. Takeaway: "count and normalize" and "define a loss and optimize" are the same objective wearing different clothes; the neural framing only pays off once the model needs more context than counting can capture (part 2). Loss floor of a pure bigram is ~2.45 nats (perplexity ≈ 11.6) — the bar for later models.
- **train/dev gap is a diagnostic, and zero gap is bad.** In the MLP (`mlp.py`), watch train vs dev loss together. train ≈ dev with *both high* = underfitting → scale up (bigger embedding dim / more hidden units / more steps). A *small* gap with dev *still falling across changes* = healthy, near-capacity fit — not something to "fix." train ≪ dev with dev *rising* = real overfitting → regularize / more data. The reliable signal is the trend of dev across runs, not any single gap (one dev number is noisy). Corollary: aiming for zero gap usually means you're leaving performance on the table. Test set stays sealed until all tuning is done — tuning against dev slowly contaminates it, and test is the one unbiased number you get to keep.
- **Context + embeddings is what beats counting.** Loss progression across the whole makemore arc: bigram 2.45 → small MLP (2-dim emb, 100 hidden) 2.35 → wider MLP (10-dim emb, 200 hidden) 2.27 → trained longer 2.19 dev. Samples tell the same story qualitatively — bigram produces garbage (`momasurailezitynn`), the MLP produces pronounceable near-names (`mariah`, `kayden`, `haniel`). The embedding table `C` is a *learned* map from discrete chars into a continuous space where predictively-similar chars cluster (vowels together); 2 dims is a teaching bottleneck (plottable), real capacity needs more. **Portfolio thread:** side-by-side bigram vs MLP samples + the loss ladder is a clean "why context and embeddings matter" story.

## Experiments

_Anything I ran: what I changed, what I expected, what I observed, what I learned. Link to the script / notebook / commit._

- _(none yet)_
