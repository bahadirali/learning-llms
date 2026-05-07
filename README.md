# learning-llms

From-scratch implementations and study notes for understanding LLMs end-to-end: tokenization, transformers, pretraining, fine-tuning, and inference.

This is a working repository tracking my progress through a structured curriculum to build, train, and tweak large language models in PyTorch. The goal is depth over breadth — every component is implemented and understood, not imported as a black box.

## Goals

- Implement the full LLM stack from autograd up to a trained, instruction-tuned model
- Understand the math and systems tradeoffs well enough to modify architectures, training procedures, and inference pipelines
- Produce reproducible code and notes alongside each phase

## Stack

- Python 3.11+, PyTorch
- Compute: [Colab / personal GPU / cloud — fill in]
- Notes in Markdown, code in standalone scripts and Jupyter notebooks

## Roadmap

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Math + PyTorch foundations (Karpathy Zero to Hero) | 🟡 In progress |
| 2 | Neural language models & transformer architecture | ⚪ Not started |
| 3 | Build a GPT from scratch (nanoGPT, BPE tokenizer) | ⚪ Not started |
| 4 | Pretraining at scale (data pipelines, distributed training) | ⚪ Not started |
| 5 | Post-training: SFT, RLHF, DPO, GRPO | ⚪ Not started |
| 6 | Inference optimization & modern architecture variants | ⚪ Not started |

## Repository structure

```
learning-llms/
├── phase1-foundations/
│   ├── notes.md           # concepts, open questions, key insights
│   ├── micrograd/         # reimplementation + experiments
│   ├── makemore/          # bigram → MLP → WaveNet
│   └── experiments/       # ad-hoc PyTorch exercises
├── phase2-transformers/
├── phase3-nanogpt/
├── ...
└── README.md
```

Each phase directory contains a `notes.md` documenting concepts, blockers, and resolved questions, plus the code I write while working through that phase.

## Curriculum sources

- [Karpathy — Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- [karpathy/nanochat](https://github.com/karpathy/nanochat)
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/)
- [mlabonne/llm-course](https://github.com/mlabonne/llm-course)
- [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- *Attention Is All You Need* (Vaswani et al., 2017) and follow-on papers (LLaMA, Mistral, DeepSeek)

## Progress log

I keep a running log of concepts learned, open questions, and blockers in each phase's `notes.md`. Implementations are committed incrementally as I work through them.

---

*This repo is a learning artifact in active development. Code quality and scope reflect the phase I'm currently working through.*
