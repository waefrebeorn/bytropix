# Vault: Optimizers — Meta-Learning LR/Momentum Control
#
## Location: `OPTIMIZERS/`

### q-controller/
- `qcontroller.py` — JAX Q-learning LR controller (10-state × 5-action Q-table, ε-greedy exploration)
- `qlearnerexample.py` — Standalone Q-learning example
- `pidexample.py` — PID controller example
- `toroidexample.py` — Toroidal gradient decomposition example

 pid-controller/
    12|PID Lambda Controller for second-order loss balancing. Each clone (in Project Chimera) gets its own PID agent managing loss weights as control signals with P/I/D terms.
    13|

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) and [Presentation Layer](../presentation/README.md) for navigation.*
