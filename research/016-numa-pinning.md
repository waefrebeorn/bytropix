# 016 — NUMA-aware thread pinning (+19–21% throughput)

Source: NUMA-Aware Scheduling writeups (numactl --cpunodebind/--membind);
llama.cpp -t/-C thread-affinity docs; OpenVINO CPU plugin affinity.
Convergent with the game-console lesson (C01 arena/SoA): **the OS scheduler
will spread your worker threads across both sockets, and your data lands on the
remote socket's RAM — every weight read pays cross-socket latency.**

## Core idea
Our engine is OpenMP-parallel over M (output rows) in GEMV. On a multi-socket
box (or any NUMA topology) the default scheduler scatters threads + their
working set across nodes. Pinning threads to cores **and** their memory to the
*same* node recovers 19–21% throughput (measured 340→412 inf/s on one
report). On a single-socket WSL/CPU box the win is smaller but real:
`OMP_PROC_BIND=close OMP_PLACES=cores` keeps each GEMV row-chunk
on a stable core + its L1/L2, raising cache hit rate.

## Triple-DA
- P1 correctness: affinity changes *where* threads run, not *what* they
  compute. Bit-identical results. ✓
- P2 privacy: env vars / pthread attr — no external lib. ✓
- P3 robustness: on a single-socket machine the bind is a no-op (correct),
  never harmful. Must not over-pin and starve the OS.

## Implementation plan
- `wubu_thread.c/.h`: at engine init, read
  `WUBU_NUMA_NODE` (default = the node with the model's RAM, or 0).
  Set `OMP_PROC_BIND=close`, `OMP_PLACES=cores` via
  `omp_set_proc_bind` / pthread_setaffinity, and `numa_bind_memory`
  (or `set_mempolicy` / `mbind`) to that node for the weight + KV buffers.
- Wrap buffer allocs so the big weight tensor is interleaved or node-local.

## Test oracle
- Unit: after init, assert `sched_getaffinity` shows the pinned set and
  `numa_node_of_address(weight_ptr)` == configured node (on NUMA kernels).
- Perf: time a 1k-token decode of Qwen with vs without pinning;
  assert pinned is >= (no-regression) and >= +5% on multi-node test boxes.
