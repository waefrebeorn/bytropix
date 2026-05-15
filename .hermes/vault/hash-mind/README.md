# Vault: hash-mind — WuBuMind JAX Versions (V1-V7.1)

## Core Files
- `WuBuMindV7.1.py` — The latest. Full pipeline: BPE tokenizer → Navigator (geometric embedding) → Oracle (text generation) → Funnel Cake (BallTree index) → Generate. Q-Controller LR, Rich TUI.
- `WuBuMindV7.py` — V7 architecture with GRU + hyperboloid projection + multi-head attention
- `WuBuMindV6-WuBuMindJAXv1337.py` — WuBuMindJAX "v1337" = V6 with massive layers
- `WuBuMindJAXv5.py` — Dual-agent Q-learning + PID hyperparameter control
- `WuBuMindJAXv3CORPUSPASTE.py` — Trained on CORPUS.py data, advanced
- `WuBuMindJAXv2(SHAKESPEARE).py` — Shakespeare-trained, message loop
- `WuBuMindJAX.py` — Original hyperbolic kNN attention
- `wubuMindv4JAX.py` — V4 with WebRadio support
- `wubuMindv4WEBRADIO.py` — WebRadio streaming version

## Training Modules (heavy lifters)
- `WuBuNest_Trainer.py` — 2900+ lines JAX training with DDP, Rich TUI, Q-Controller, checkpointing
- `WuBuNestmRnaTrainer.py` — PyTorch version (no Triton), RNA secondary structures
- `WuBuNest_Inference.py` — Separate inference module
- `wubumind_galactic_core_v1.py` / `v3_qlearn.py` — Galactic core attention with Q-learning

## SimpleHash Precursors (rolling hash attention)
- `SimpleHashV1.py`, `V2.py`, `V3.py` — Evolution of hash-based attention
- C port at `c/` directory

## Other
- `wubu_nesting_impl.py` — Core layer: HyperbolicUtils, Hamilton product, WuBuNestingLayer
- `wubu_nesting_example.py` — Usage example
- `wubu_nesting_visualization.py` — ASCII + matplotlib visualization
- `CORPUS.txt` — Training data (Shakespeare, fully commented)
- `GAAD-WuBu-ST1.md`, `GAAD-WuBu-ST2.md` — Academic paper drafts
- `ProjectAgentChimera.md` — Multi-agent training design spec

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) for navigation.*
