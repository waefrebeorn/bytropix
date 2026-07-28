# Research INDEX — gaps, verdicts, convergence

Method: Kevin-Bacon 7-hop across HPC / DB-buffer-pool / ML-systems / game-console /
formal-verification. Triple-DA = (1) correctness (2) privacy/safety/no-3rd-party-lib
(3) robustness. Status: `open` = not yet in engine; `wired` = implemented+tested.
Spine = **decode is memory-bandwidth-bound** (Roofline 2607.02558). Every win
attacks bytes moved.

## THEME A — KV-cache compression (memory wall)
- A01 KV Q8_0 block-32 absmax ............................ `wired` (wubu_kvcache_quant)
- A02 KIVI K-per-channel / V-per-token ................... `wired`
- A03 Ecco entropy-aware per-block 2–8bit adaptive ........ `open` → doc 001
- A04 Predictive multi-tier KV (DRAM/NVMe/CXL/IB, Bayesian reuse) `open` → doc 002
- A05 Multi-tier dynamic KV offload for edge (MTDS) ........ `open` (subsumed by 002)
- A06 LMCache/KVBM prefix+PD-disaggregation .............. `open` (prefix reuse → doc 010)
- A07 NVIDIA priority-based KV eviction (LRU+importance) .. `open` (ties to 002)
- A08 KV dtype alignment across tiers (fp8/bf16/fp16) .... `open` (ties to 002/003)
- A09 Attention-sink-free gated attention (kills massive activations) `open` → doc 011
- A10 RoPE-aware KV prefetch ............................. `open` (ties to 002)

## THEME B — Weight quantization (halve weight traffic)
- B01 int8 GEMV (row absmax, fp32 acc) ................. `wired` (wubu_gemv_tune)
- B02 int4 weight-only GEMV (Marlin-style pack) .......... `wired` → doc 003
- B03 BitNet 1.58 ternary {-1,0,+1} GEMV ............. `wired` → doc 004
- B04 SmoothQuant activation outlier migration ............. `wired` → doc 005
- B05 AWQ activation-aware 1% salient channel protect .... `open` (ties to 003/005)
- B06 GPTQ 2nd-order weight quant ...................... `open` (offline calib; safe)
- B07 FP8 E4M3/E5M2 mixed precision (HW-dependent) ...... `open` (CPU→emul, low prio)
- B08 NVFP4 dispatch (Blackwell) ....................... `open` (HW-gated, skip CPU)

## THEME C — Structure-of-Arrays / cache-aware layout (game-console lesson)
- C01 Arena allocator for per-request + KV buffers ....... `wired` → doc 006
- C02 SoA activation/state tensors (vs AoS malloc) ....... `open` (ties to 006)
- C03 Cache-line packing of KV pages (64B aligned) ...... `open` (ties to 006/002)
- C04 Fixed-timestep / deterministic decode step ......... `open` (scheduler → doc 007)
- C05 Hot/cold split (compute vs metadata) ............. `open` (ties to 006)
- C06 ECS-style component store for engine state ........ `open` (ties to 001/006)

## THEME D — Batching / scheduling / transport
- D01 Continuous (iteration-level) batching ............. `open` → doc 007
- D02 Prefix KV reuse across requests (hash map) ........ `open` → doc 010
- D03 Disaggregated prefill/decode (separate passes) ..... `open` (ties to 002/007)
- D04 Chunked prefill (overlap w/ decode) ............. `open` (ties to 007)
- D05 KV transfer layer (NIXL/UCX analog, localhost) .. `open` (ties to 002/003)

## THEME E — Architecture variants we must SUPPORT in loader/forward
- E01 GQA/MQA grouping factor G (already in engine) ..... `wired`
- E02 MLA (DeepSeek multi-head latent attention) ......... `open` (loader extension)
- E03 Gated-DeltaNet 3:1 hybrid linear attention ...... `open` → doc 008
- E04 Mixture-of-Depths dynamic layer skip ............. `open` (router in forward)
- E05 Fine-grained MoE expert choice routing ........... `open` (ties to wubu_moe)
- E06 Wide expert parallelism (≥8 GPU) ................ `open` (multi-host; skip single-host)

## THEME F — Formal verification of kernels
- F01 Bounded equivalence test: quant-GEMV == oracle .... `open` → doc 009
- F02 Alive2-style SMT check of GEMV rewrites ......... `open` (tooling; ties to 009)
- F03 Numerical-stability audit of dequant paths ........ `open` (ties to 009)

## THEME G — Speculative / draft acceleration
- G01 EAGLE self-draft tree verify ..................... `open` → doc 012
- G02 MEDUSA multiple guess heads ..................... `open` (ties to 012)
- G03 Lookahead / n-gram fallback ..................... `open` (cheap; ties to 012)

## THEME H — Prefill kernel / compute-bound phase
| H01 FlashAttention-style fused prefill (tile+softmax) . `open` (ties to 001/003)
- H02 Warp/thread specialization analog for CPU ......... `open` (ties to 007)
- H03 Incoherent FP8 processing (Hadamard) ............ `open` (HW-gated)
- H04 FlashDecoding parallel KV-load decode attn ..... `wired` → doc 015
- H05 QuaRot/SpinQuant Hadamard 4-bit W+A+KV ..... `wired` → doc 013
- H06 Sub-4-bit KV vector quant (CommVQ/TurboQuant) `wired*` → doc 014

## THEME I — Game-console hardware discipline (the "game-design our inference" ask)
- I01 Arena allocator for per-request + KV buffers .... `wired` → doc 006
- I02 SoA activation/state tensors (vs AoS malloc) .... `open` (ties to 006)
- I03 Cache-line packing of KV pages (64B aligned) ..... `open` (ties to 006)
- I04 Fixed-timestep / deterministic decode step ....... `open` (ties to 007)
- I05 NUMA/thread-affinity pinning (+19-21% thru) ... `wired` → doc 016
- I06 Hot/cold split (compute vs metadata) ............. `open` (ties to 006)

## THEME J — Adaptive compute (skip layers / early exit)
- J01 Mixture-of-Depths dynamic layer skip ............ `open` (ties to 008)
- J02 GateSkip/LayerSkip token-wise gate skip ........ `open` → doc 017
- J03 Early-exit + self-speculative verify ........... `open` (ties to 017/012)

## THEME K — Cascade speculative (small drafter + large verifier)
- K01 n-gram cascade drafter (no 3rd-party) ........ `wired*` → doc 018
- K02 Self-cascade (small local Colonel drafts) ...... `open` (ties to 018)
- K03 CAS-Spec adaptive deferral rule ............... `open` (ties to 018)

## Cross-cutting convergence statement
A/B/C all reduce *bytes per token*. D/E/F/G/H/I/J/K are about *amortizing*
those bytes across requests (D), matching the *model's own* structure (E/J),
*proving* the fast path is correct (F), *guessing* tokens to skip the
matmul (G/K), and *landing on real silicon* via console-game hardware
discipline (I: arena/SoA/NUMA/cache-line). The 013/014/015 wins are
the next halvings on top of shipped B01/B02/A01/A02.
