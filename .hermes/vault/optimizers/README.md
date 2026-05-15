# Vault: Optimizers — Meta-Learning LR/Momentum Control

## Location: `OPTIMIZERS/`

### q-controller/
- `qcontroller.py` — JAX Q-learning LR controller (10-state × 5-action Q-table, ε-greedy exploration)
- `qlearnerexample.py` — Standalone Q-learning example
- `pidexample.py` — PID controller example
- `toroidexample.py` — Toroidal gradient decomposition example

### pid-controller/
PID Lambda Controller for second-order loss balancing. Each clone (in Project Chimera) gets its own PID agent managing loss weights as control signals with P/I/D terms.

## Vault Audit (May 15)
- **Port priority:** P2 — low effort, high reuse value
- **Q-Controller:** 10-state × 5-action JAX prototype, tiny & clean (<100 lines). Port to C: store Q-table as float[10][5], ε-greedy action selection, Bellman update.
- **PID Lambda Controller:** Adapts LR from loss gradient P/I/D terms. Straightforward C math.
- **Relevance:** PGA LR tuning (current P1) needs adaptive LR — PID controller directly applicable to damp loss jumps 21.6→69.

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) for navigation.*
