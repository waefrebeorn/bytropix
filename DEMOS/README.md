# Python Demos

This folder is a placeholder for future standalone, runnable demos extracted from the research scripts. Each file in `ENCODERS/`, `ATTENTION/`, and `DIFFUSION/` is a self-contained demo — see those directories for the actual code.

## Quick Start Demos

For the fastest "get something running" experience:

| Demo | File | What It Does |
|------|------|-------------|
| Hyperbolic kNN attention | `ENCODERS/hash-mind/WuBuMindJAX.py` | Text generation with Poincaré ball attention |
| Quaternion rotations | `ENCODERS/hash-mind/wubu_nesting_impl.py` | Hamilton product, exp/log maps, WuBu layer |
| Geodesic layer | `ENCODERS/hamilton-encoder-cpu/Wubu_Geodesic_Layer_Final.py` | Geodesic curvature with PID controllers |
| Sparse attention | `ATTENTION/wubu-sparse-attention/WuBuSparseAttention.py` | Working/associative memory attention |
| Q-learning optimizer | `OPTIMIZERS/q-controller/qcontroller.py` | Meta-learning LR scheduler |
| Topological sequence model | `ATTENTION/topological-sequence-model/topological_sequence_model1.py` | Linear-complexity attention |

## Requirements

Each script documents its own dependencies. Common requirements:
- `torch` or `jax` + `flax` + `optax`
- `numpy`, `Pillow`
- `tqdm` for training progress

See `DRAFT/requirements.txt` for the full list.
