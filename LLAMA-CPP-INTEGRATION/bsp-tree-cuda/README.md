# BSP Tree CUDA

Binary Space Partitioning tree using quaternion-split on GPU.

The BSP tree organizes KV cache entries by their quaternion encoding: each tree node represents a rotation in SO(4), splitting a spherical cap into two child caps based on angular distance.

**Status**: Implementation is in the `llama-cpp-rotorquant` fork (separate repo). See `LLAMA-CPP-INTEGRATION/README.md` for architecture documentation.

**Key details**:
- `BSPNode` struct with split quaternion + child indices + sphere cap boundary
- Persistent pool allocation (pre-allocated GPU memory, no runtime malloc)
- Iterative traversal (not recursive — GPU stack limitations)
- O(log N) query time vs O(N) for full attention
