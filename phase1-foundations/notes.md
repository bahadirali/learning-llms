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

## Experiments

_Anything I ran: what I changed, what I expected, what I observed, what I learned. Link to the script / notebook / commit._

- _(none yet)_
