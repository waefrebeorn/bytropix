# Vault: Tailslayer — Hedged Reads for Speculative Decode

## Source
[LaurieWired/tailslayer](https://github.com/LaurieWired/tailslayer) — Apache 2.0
Cloned to: `~/HASHMIND/tailslayer/`
Docs at: `THEORY/papers/tailslayer-*.md`

## What It Is
C++ library that reduces DRAM refresh tail latency via hedged reads across independent memory channels. Replicates data N ways, issues simultaneous reads across channels with uncorrelated refresh schedules, first response wins.

## Files Examined (May 15)
- `hedged_reader.hpp` — 221-line C++ template, N-way hedged read across DRAM channels
- `trefi_probe.c` — 335-line DRAM refresh jitter detector (clflush+reload, TSC calibration, harmonic binning)
- `discovery/benchmark/main.cpp` — Multi-arm benchmark (single/hedged, quiet/stress)
- `discovery/benchmark/benchmark.cpp` — Measurement thread, sliding-window pair sampling, percentile stats
- `discovery/benchmark/hw_utils.hpp` — Core pinning, TSC calibration, virtual-to-physical address resolution

## Direct Pattern Match: WuBuText

| Tailslayer Pattern | WuBuText Analog | Priority |
|---|---|---|
| N replicas on independent DRAM channels | N draft tokens speculated in parallel | **P2** |
| clflush+reload timing | Forward pass timing for draft verification | P2 |
| Hedged read (first-response-wins) | Accept longest valid prefix, cancel remaining | **P2** |
| N replicas pinned → separate cores | E experts dispatched → S SMs | P3 |
| Physical addr → channel bit extraction | CUDA shared memory bank conflict analysis | P3 |
| tREFI probe (TSC calibration, harmonic binning) | CUDA kernel launch / PCIe timing | P3 |
| Sliding window pair sampling | Draft-target logit time alignment | P2 |
| N-way: any N ≤ available channels | MoE dispatch scaler | P3 |

## Port Plan
1. **Speculative Decode Kernel** (`spec_verify.cu`): Use hedged-read template pattern — launch N draft verification threads across GPU SMs, first valid prefix wins, cancel remaining
2. **tREFI probe for CUDA** (`pcie_probe.cu`): Port clflush+reload → CUDA event timing for PCIe transfer detection
3. **Sliding window pair sampling**: Align draft-target logits by timestamp, take minimum latency
4. **Bank conflict analysis** (`bank_analyzer.cu`): Port `compute_channel()` → `compute_bank()` for shared memory

---

*Part of the WuBuText AI project. See [Project Overview](../../README.md) for navigation.*
