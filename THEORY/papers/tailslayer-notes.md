# Tailslayer: Hedged Reads & DRAM Refresh Tail Latency

**Origin:** `~/HASHMIND/tailslayer/` — clone of [LaurieWired/tailslayer](https://github.com/LaurieWired/tailslayer)
**Author:** LaurieWired
**License:** Apache 2.0

## What It Is

A C++ library that reduces tail latency in RAM reads caused by DRAM refresh stalls. It replicates data across multiple independent DRAM channels with uncorrelated refresh schedules, uses undocumented channel scrambling offsets (works on AMD, Intel, Graviton), and issues hedged reads (clflush+reload) — taking whichever replica responds first.

## Relevance to WuBuText / bytropix

### 1. Hedged Reads → Speculative Decoding (direct pattern match)

The core tailslayer pattern — issue N identical requests in parallel, take first valid response — maps **directly** to speculative decoding in LLM inference:

| Tailslayer Concept | Speculative Decoding Analog |
|---|---|
| N replicas on independent channels | N draft tokens speculated in parallel |
| Clflush+reload timing | Model forward pass timing |
| Hedged read (first response wins) | Accept longest valid prefix |
| Channel scrambling offset | Draft model distribution alignment |
| tREFI refresh stall as latency source | Model forward pass as latency source |

In speculative decoding:
- Draft model proposes N candidate tokens (analogous to N replicas)
- Target model verifies all N in a single forward pass (analogous to hedged reads)
- Accepted prefix = first valid response
- Rejected suffixes = dropped slow replicas

This is documented in: `DeepSeek-V3.2`, `Delta Attention`, speculative decoding papers in `THEORY/papers/`

### 2. Channel-Aware Memory Layout → CUDA Shared Memory

The tailslayer technique of computing which DRAM channel an address maps to (via physical address bit extraction) is analogous to:
- **CUDA shared memory bank conflicts**: Understanding which bank an address maps to enables conflict-free access patterns
- **cuBLAS workspace alignment**: Proper alignment reduces TLB misses

### 3. Precise Timing Measurement → CUDA Kernel Profiling

The `trefi_probe.c` technique (TSC calibration via nanosleep, jitter detection via clflush+reload timing) can be adapted for:
- Measuring CUDA kernel launch overhead
- Profiling memory bandwidth utilization
- Detecting PCIe transfer bottlenecks

### 4. N-Way Replication → MoE Expert Parallelism

The `N` replicas managed across independent channels maps to managing `E` experts across independent GPU SMs:
- Each expert is an independent computation path (like a DRAM channel)
- Gate/router selects which experts to activate (like selecting which replica to read)
- Load balancing across experts (like ensuring replicas land on different channels)

## Key Files Copied

| File | Original | Description |
|------|----------|-------------|
| `papers/tailslayer-README.md` | `README.md` | Full tailslayer documentation |
| `papers/tailslayer-hedged-reader.hpp` | `include/tailslayer/hedged_reader.hpp` | Core hedged reader implementation |
| `papers/tailslayer-trefi-probe.c` | `discovery/trefi_probe.c` | DRAM refresh timing probe |
| `papers/tailslayer-benchmark.md` | `discovery/benchmark/` files | Benchmark methodology |
