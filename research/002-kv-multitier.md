# 002 — Predictive multi-tier KV cache (DRAM / NVMe / CXL / IB)

Source: Ganjihal, "Predictive Multi-Tier Memory Management for KV Cache in
Large-Scale GPU Inference", arXiv:2604.26968 (2026). Also MTDS (s40747-025-02200-4),
Hybe (3695053.3731051).

## Core idea
KV cache should not live only in the hottest tier. A 6-tier hierarchy (HBM→DRAM→
CXL→NVMe→RDMA→FS) extends effective capacity 40 GB → 38 TB. A Bayesian reuse
predictor (Beta priors over (block-type, transition) pairs) hits 70–84%, with
EMA head-granular eviction + RoPE-aware prefetch. On a single host this collapses
to: HOT = our existing arena (13 GB cap), WARM = mmap'd DRAM file, COLD = NVMe
file. The lever for US: we can run the 65 GB KAT-Coder and 56 GB Qwen on 13 GB
RAM by paging cold KV to disk, because decode only touches the last ~few K tokens
hot and the rest is sequential replay during prefill.

## Triple-DA
- P1 correctness: KV values are immutable once written for a position; tiering is a
  pure caching concern. Reads must return the same float regardless of tier. ✓
- P2 privacy: local mmap + file, no network needed on single host. OWN-C. ✓
- P3 robustness: tier miss = a fetch from colder tier (latency, not correctness).
  Must never evict a hot block we're about to read. Use EMA-scored LRU, not pure LRU.

## Implementation plan
- New `wubu_kv_tier.c/.h`: a KV arena with 3 local tiers (RAM / DRAM-mmap /
  NVMe-mmap). Each KV block carries a tier tag + last-access EMA.
- Hook into `kv_cache_read_head`/`write_head`: write to HOT; a background "colder"
  sweeper demotes low-EMA blocks to WARM/COLD under memory pressure (governed by
  a `WUBU_KV_TIER_LIMIT_MB` env, default = free RAM - headroom).
- Reuse predictor: simplest correct version = per-layer LRU with EMA; the Bayesian
  Beta model is an upgrade later.

## Test oracle
- Allocate a KV cache bigger than RAM budget; assert eviction demotes to file tier
  and a subsequent read returns the exact written values (bit-identical, since we
  store fp16 on cold tier).
- Assert decode of a long context (Qwen, 8k) completes without OOM on a capped
  RAM budget (proves tiering engaged).
