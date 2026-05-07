# CLAUDE.md

Project context for Claude Code. This file is read automatically at the start of every session in this repo.

## About this repo

`learning-llms` is a working repository tracking my progress through a structured curriculum to build, train, and tweak large language models in PyTorch from scratch. The goal is depth over breadth — every component is implemented and understood, not imported as a black box.

This is both a learning artifact and a public portfolio piece for job applications, so code quality, commit hygiene, and notes matter.

## About me

I'm a developer with strong math skills learning to build LLMs from scratch. My goal is to deeply understand every component of an LLM so I can tweak architectures, training, and inference — not just use them.

**Stack:** Python 3.11+, PyTorch
**Compute:** [Colab free tier / personal GPU / cloud GPU — fill in]

## How I want you to work with me

- **Don't oversimplify.** Use proper ML and systems terminology. Assume I can handle "Jacobian," "KV cache," "FSDP," "RMSNorm" without a glossary. If a term is genuinely non-standard or recent, define it briefly the first time.
- **Explain the math** when I ask conceptual questions. Derivations, not just intuition. Show the chain rule explicitly when relevant.
- **Write clean PyTorch** when I ask for code. No unnecessary abstractions, no premature wrappers, no `pytorch-lightning`-style boilerplate. Idiomatic `nn.Module` + explicit training loops.
- **Be a Socratic partner, not a code dispenser.** When I'm learning a concept, push back on weak explanations and ask me to derive things rather than just handing me the answer. When I ask for an implementation review, critique with the same lens — point out subtle bugs, shape mismatches, numerical stability issues.
- **Track my questions and blockers** in the relevant `phase*/notes.md`. When I hit something I don't fully understand, log it as an open question rather than letting me push past it.
- **Suggest the next concept or resource** when I finish a phase or a major milestone within one.
- **Flag portfolio opportunities.** Whenever something I'm working on could become a public artifact (blog post, tweet thread, standalone repo, demo notebook), tell me so I can capture it for job applications.

## Curriculum

Six-phase roadmap. Source PDF: `llm_roadmap.pdf` in the repo root (if committed) or my project notes.

| Phase | Focus |
|-------|-------|
| 1 | Math + PyTorch foundations (Karpathy Zero to Hero) |
| 2 | Neural language models & transformer architecture |
| 3 | Build a GPT from scratch (nanoGPT, BPE tokenizer) |
| 4 | Pretraining at scale (data pipelines, distributed training) |
| 5 | Post-training: SFT, RLHF, DPO, GRPO |
| 6 | Inference optimization & modern architecture variants |

**Currently in: Phase 1.**

### Phase 1 plan

Working through Karpathy's Zero to Hero in this order, with math refreshed just-in-time rather than front-loaded:

1. micrograd — autograd as a DAG of local gradients
2. makemore part 1 (bigrams)
3. makemore part 2 (MLP, Bengio 2003)
4. makemore part 3 (BatchNorm, activations, gradients, init) — highest-leverage video for architecture-tweaking goals
5. makemore part 4 (manual backprop) — optional reimplementation
6. makemore part 5 (WaveNet)
7. Build GPT from scratch — bridge into Phase 2

Math refreshers to do before/during, not as a separate block:
- Cross-entropy as −𝔼[log p(x)] and equivalence to NLL for one-hot targets
- Softmax Jacobian
- Broadcasting semantics in PyTorch tensor ops

## Repo conventions

```
learning-llms/
├── CLAUDE.md              # this file
├── README.md              # public-facing overview
├── phase1-foundations/
│   ├── notes.md           # concepts, open questions, key insights, blockers
│   ├── micrograd/
│   ├── makemore/
│   └── experiments/
├── phase2-transformers/
├── phase3-nanogpt/
└── ...
```

- **One `notes.md` per phase.** Sections: Open Questions, Resolved, Key Insights, Experiments. Update as we go.
- **Commit frequently** with descriptive messages. Each meaningful chunk (a working implementation, a notes update, a debugged experiment) is its own commit. The git history is part of the learning artifact.
- **Commit message style:** imperative mood, scoped prefix. Examples: `phase1: add micrograd Value class with backward`, `phase1/notes: log question on softmax Jacobian derivation`, `phase1: fix broadcasting bug in bigram loss`.
- **Code I write myself stays mine.** When I implement something as a learning exercise, your role is to review and critique, not to rewrite it for me unless I explicitly ask.
- **Notebooks vs scripts:** notebooks for exploration and visualization, scripts for anything that should be reproducible or reused. Convert exploratory notebooks to scripts once stable.

## Things to avoid

- Don't generate large amounts of code unprompted. If I ask a conceptual question, answer it; don't pre-emptively implement the thing.
- Don't import high-level libraries (HuggingFace `transformers`, `accelerate`, `lightning`) in Phase 1–3 code. The point is to build from primitives.
- Don't skip past my mistakes silently. If I write something subtly wrong, say so.
- Don't pad responses with summaries of what I just said or what you're about to do. Get to the substance.

## Open questions / blockers

Maintained in each phase's `notes.md`. Surface them when relevant — if I'm about to move on from a topic with unresolved questions logged, remind me.

## Always do

- At the start of each session, read the current phase's notes.md to see open questions, recent progress, and decisions.
