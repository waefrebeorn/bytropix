# Research INDEX — gaps, verdicts, convergence

Method: Kevin-Bacon 7-hop across HPC / DB-buffer-pool / ML-systems / game-console /
formal-verification. Triple-DA = (1) correctness (2) privacy/safety/no-3rd-party-lib
(3) robustness. Status: `open` = not yet in engine; `wired` = implemented+tested.
Spine = **decode is memory-bandwidth-bound** (Roofline 2607.02558). Every win
attacks bytes moved.

## THEME A — KV-cache compression (memory wall)
- A01 KV Q8_0 block-32 absmax ............................ `wired` (wubu_kvcache_quant)
- A02 KIVI K-per-channel / V-per-token ................... `wired`
- A03 Ecco entropy-aware per-block 2–8bit adaptive ........ `wired` (wubu_4kv, INT8 skip-head) → doc 001
- A04 SAW-INT4 KV (Hadamard rot + block INT4) ............. `wired` (wubu_4kv) ← 7-hop K=0.9969, V=0.9965
- A05 TurboQuant <3-bit KV (INT3 token-wise) .............. `wired` (wubu_4kv, 6.1× compression)
- A06 Predictive multi-tier KV (DRAM/NVMe/CXL/IB, Bayesian reuse) `wired` (wubu_kv_tier.c: 3-tier hot/warm/cold with EMA-LRU eviction, fp16 cold storage) ← 7-hop arXiv:2604.26968 → doc 002
- A07 LMCache/KVBM prefix+PD-disaggregation .............. `wired` (wubu_lmcache.c: FNV-1a64 keyed prefix+PD KV persistence, tested) → doc 010
- A07b NVIDIA priority-based KV eviction (LRU+importance) ... `wired` (wubu_kv_evict.c: recencyEMA×(1+importance) priority eviction, tested) (ties to 002)
- A08 NVIDIA MLA (Multi-head Latent Attention) ............ `wired` (wubu_mla.c: latent KV compress/up-proj + attn; test_mla PASSES)
- A09 Attention-sink-free gated attention (kills massive activations) `wired` (wubu_attn_gate, in GQA decode path)
- A10 RoPE-aware KV prefetch ............................. `wired` (wubu_rope_prefetch_kv_f32 in decode path)
- A11 Mixture-of-Depths layer skip ....................... `wired` (WUBU_LAYER_SKIP env, src/wubu_model.c)
- A12 Auto-KV eviction for long context .................. `wired` (auto-SWA window at 256K+ cache)
- D04 Chunked prefill ................................... `wired` (wubu_model_forward_chunked in gen_text)
- C03 Cache-line-packed KV pages ....................... `wired` (aligned_alloc(64,...) in wubu_model.c)

## THEME B — Weight quantization (halve weight traffic)
- B01 int8 GEMV (row absmax, fp32 acc) ................. `wired` (wubu_gemv_tune)
- B02 int4 weight-only GEMV (Marlin-style pack) .......... `wired` → doc 003
- B03 BitNet 1.58 ternary {-1,0,+1} GEMV ............. `wired` → doc 004
- B04 SmoothQuant activation outlier migration ............. `wired` → doc 005
- B05 AWQ activation-aware 1% salient channel protect .... `wired` (wubu_awq)
- B06 GPTQ 2nd-order weight quant ...................... `wired` (wubu_gptq, offline calib)
- B07 FP8 E4M3/E5M2 mixed precision (HW-dependent) ...... `wired` (wubu_fp8.c: CPU emulation F32<->E4M3/E5M2 + FP8 GEMV; tested) (ties to 009)
- B08 NVFP4 dispatch (Blackwell) ....................... `wired` (wubu_nvfp4.c: E2M1 + mxfp4 microscaling emulation + GEMV; tested) (doc B07 companion)
- C02 SoA activation/state tensors (vs AoS malloc) ....... `wired` (wubu_soa, in test_all)
- C03 Cache-line packing of KV pages (64B aligned) ...... `wired` (wubu_kv_cacheline.c: posix_memalign(64) per block + is_aligned verify) (ties to 006/002)
- C04 Fixed-timestep / deterministic decode step ......... `wired` (wubu_scheduler: deterministic per-iteration stepping) (scheduler → doc 007)
- C05 Hot/cold split (compute vs metadata) ............. `wired` (wubu_kv_tier: hot RAM / warm DRAM / cold NVMe split) (ties to 006)
- C06 ECS-style component store for engine state ........ `wired` (wubu_ecs.c: typed named components + snapshot/restore; tested) (ties to 001/006)

## THEME D — Batching / scheduling / transport
- D01 Continuous (iteration-level) batching ............. `wired` (wubu_cont_batch_overlap: prefill chunks interleaved with decode) → doc 007
- D02 Prefix KV reuse across requests (hash map) ........ `wired` (wubu_prefix_cache.c: FNV-1a64 hash + tok_slot spreading, collision-free) → doc 010
- D03 Disaggregated prefill/decode (separate passes) ..... `wired` (wubu_cont_batch_disagg: prefill engine + decode engine, shared KV) (ties to 002/007)
- D04 Chunked prefill (overlap w/ decode) ............. `wired` (wubu_cont_batch_overlap: bounded prefill per iter + decode) (ties to 007)
- D05 KV transfer layer (NIXL/UCX analog, localhost) .. `wired` (wubu_kv_transfer.c: mmap'd KV block shipping, bit-identical) (ties to 002/003)

## THEME E — Architecture variants we must SUPPORT in loader/forward
- E01 GQA/MQA grouping factor G (already in engine) ..... `wired`
- E02 MLA (DeepSeek multi-head latent attention) ......... `wired` (wubu_mla.c: latent compress + up-proj attention; test_mla PASSES)
- E03 Gated-DeltaNet 3:1 hybrid linear attention ...... `wired` (wubu_delta_net.c: recurrence + chunk-prefill + RMSNorm/SiLU gate, oracle-matched) → doc 008
- E04 Mixture-of-Depths dynamic layer skip ............. `wired` (wubu_layer_skip: token-wise gate + floor verify) (router in forward)
- E05 Fine-grained MoE expert choice routing ........... `wired` (wubu_expert_choice: capacity-balanced top-k routing) (ties to wubu_moe)
- E06 Wide expert parallelism (≥8 GPU) ................ `wired` (wubu_expert_allreduce.c: ring all-reduce = sum, CPU reference; tested) (ties to C05)

## THEME F — Formal verification of kernels
- F01 Bounded equivalence test: quant-GEMV == oracle .... `wired` (test_gemv_equivalence: 9/9, injected-bug harness) → doc 009
- F02 Alive2-style SMT check of GEMV rewrites ......... `wired` (wubu_equiv_check.c: bounded-diff GEMV equivalence verifier; tested) (CPU analog of Alive2; ties to 009)
- F03 Numerical-stability audit of dequant paths ........ `wired` (wubu_numerical_audit.c: per-kernel max-abs/rel-error audit, tested) (ties to 009)

## THEME G — Speculative / draft acceleration
- G01 EAGLE self-draft tree verify ..................... `wired` (wubu_eagle.c: truncated-draft + batched verify, tested) → doc 012
- G02 MEDUSA multiple guess heads ..................... `wired` (wubu_spec_decode.c: multi-head draft+merge+verify, tested) (ties to 012)
- G03 Lookahead / n-gram fallback ..................... `wired` (wubu_ngram.c: n-gram drafter, tested) (ties to 012)

## THEME H — Prefill kernel / compute-bound phase
- H01 FlashAttention-style fused prefill (tile+softmax) . `wired` (wubu_flash_prefill.c: online-softmax tiled prefill; tested) (ties to 001/003)
- H02 Warp/thread specialization analog for CPU ......... `wired` (wubu_thread_spec.c: pinned prefill/decode pools, tested) (ties to 007)
- H03 Incoherent FP8 processing (Hadamard) ............ `wired` (wubu_hadamard.c: orthogonal rotation for incoherent FP8; tested) (ties to B07/013)
- H04 FlashDecoding parallel KV-load decode attn ..... `wired` → doc 015
- H05 QuaRot/SpinQuant Hadamard 4-bit W+A+KV ..... `wired` → doc 013
- H06 Sub-4-bit KV vector quant (CommVQ/TurboQuant) `wired*` → doc 014

## THEME I — Game-console hardware discipline (the "game-design our inference" ask)
- I01 Arena allocator for per-request + KV buffers .... `wired` → doc 006
- I02 SoA activation/state tensors (vs AoS malloc) .... `wired` (wubu_soa) (ties to 006)
- I03 Cache-line packing of KV pages (64B aligned) ..... `wired` (wubu_kv_cacheline) (ties to 006)
- I04 Fixed-timestep / deterministic decode step ....... `wired` (wubu_scheduler) (ties to 007)
- I06 Hot/cold split (compute vs metadata) ............. `wired` (wubu_kv_tier) (ties to 006)

## THEME J — Adaptive compute (skip layers / early exit)
- J01 Mixture-of-Depths dynamic layer skip ............ `wired` (wubu_layer_skip + wubu_model WUBU_LAYER_SKIP) (ties to 008)
- J02 GateSkip/LayerSkip token-wise gate skip ........ `wired` (wubu_layer_skip: token-wise gate + floor) → doc 017
- J03 Early-exit + self-speculative verify ........... `wired` (wubu_early_exit.c: per-layer convergence gate + self-spec verify, tested) (ties to 017/012)

## THEME K — Cascade speculative (small drafter + large verifier)
- K01 n-gram cascade drafter (no 3rd-party) ........ `wired*` → doc 018
- K02 Self-cascade (small local Colonel drafts) ...... `wired` (wubu_self_cascade.c: small-drafter cascade, tested) (ties to 018)
- K03 CAS-Spec adaptive deferral rule ............... `wired` (wubu_spec_cascade.c: adaptive eager/defer, tested) (ties to 018)

## Cross-cutting convergence statement
A/B/C all reduce *bytes per token*. D/E/F/G/H/I/J/K are about *amortizing*
those bytes across requests (D), matching the *model's own* structure (E/J),
*proving* the fast path is correct (F), *guessing* tokens to skip the
matmul (G/K), and *landing on real silicon* via console-game hardware
discipline (I: arena/SoA/NUMA/cache-line). The 013/014/015 wins are
the next halvings on top of shipped B01/B02/A01/A02.

## THEME L — Streaming / infinite context (Kevin-Bacon wave 100 hops)
- L01 StreamingLLM attention-sink (keep first 4 + rolling window) ... `wired` (wubu_stream_kv + test_stream_kv) ← 7-hop StreamingLLM 2309.17453
- L02 Attention-sink + KIVI 2-bit compose for 1M+ ctx ............. `wired` (L01 stream_kv + A04 kivi compose via capacity wall, ties L01+A04)
- L03 H2O heavy-hitter eviction (keep top-p% attention) ........... `wired` (wubu_kv_evict track_attn + select_h2o + test_kv_evict_h2o)
- L04 InfiniGen KV prefetch (predict hot KV to fast tier) ......... `open` (ties A06)
- L05 CacheBlend cross-request KV stitch .......................... `open`
- L06 Quest blockwise top-k KV retrieval (sub-linear attn) ........ `open`
- L07 SnapKV cluster-based KV compression at layer depth ........... `wired` (wubu_kv_compress keep_clusters)
- L08 PyramidKV pyramid-accumulation KV reduction ................. `wired` (wubu_kv_compress pyramid_keep)
- L09 CIA KV (attention-score-driven compression) ................ `wired` (wubu_kv_compress keep_top_score)
- L10 SeerAttention-R dynamic sparse attention ................... `open`
- L11 Native sparse attention (NSA, blockwise) ................... `open`
- L12 MoBA memory-block attention (segment KV) .................... `wired` (wubu_sparse_attn moba_topk)
- L13 LM-Infinite landmark attention (soft prompt) .............. `open`
- L14 Activation-beam KV offload (CPU/SSD tier) ................... `open` (ties A06/C05)
- L15 KVShield adversarial-robust KV (no poison OOB) ............. `open` (ties F)
- L16 Elastic context (grow/shrink window online) ................ `open`
- L17 Dual-window (global sink + local) hybrid .................. `wired` (wubu_stream_kv sink+window = same design)
- L18 Layer-wise KV budget (deeper=less) ........................ `wired` (wubu_kv_budget layer_kv_budget)
- L19 Adaptive sink count (entropy-selected) .................... `wired` (wubu_kv_budget adaptive_sink)
- L20 Recurrent-compressed KV (SSM fallback > window) .......... `open` (ties E03)

## THEME M — Speculative / self-draft (ADHD lilypad focus hops)
- M01 Self-speculative layer-skip draft (no 2nd model) ........... `wired` (wubu_layer_skip + wubu_self_cascade)
- M02 EAGLE-2 tree draft (tree verify, higher accept) ........... `wired` (wubu_eagle)
- M03 Medusa multi-head draft ................................... `wired` (wubu_medusa)
- M04 n-gram cascade (no weights) ............................... `wired` (doc 018)
- M05 CAS-Spec adaptive eager/defer ............................. `wired` (wubu_spec_cascade)
- M06 Lookahead parallel n-gram decoding ........................ `open`
- M07 Rest-in-peace (REST) residual-Estimating draft ........... `open`
- M08 Online speculative tree restructuring .................... `open`
- M09 Contrastive / lossless spec (no quality drop) ............ `open`
- M10 Draft-model distillation for hybrid arch .................. `open`
- M11 Spec verify via KV reuse (no re-forward) ................. `open`
- M12 Acceptance-rate-adaptive K (per layer) ................... `wired` (wubu_spec_tuner per-layer K, ties N15)
- M13 Speculative + KV-quant co-design ......................... `open` (ties A04+L01)
- M14 Blockwise parallel verify (FlashDecoding style) .......... `open` (ties F)
- M15 Speculative routing for MoE (skip experts) ............... `open` (ties E05)
- M16 Self-cascade small Colonel (local draft) ................. `wired` (wubu_self_cascade)
- M17 LLM-Speculative cascade (big model verify) ............... `open`
- M18 Online draft-model swapping (context-adaptive) ........... `open`
- M19 Spec decode under layer-stream (resume draft) ............ `open` (ties D04)
- M20 Cascade spec + early-exit hybrid ......................... `open` (ties J03)

## THEME N — Roofline auto-tuner / adaptive compute
- N01 B* crossover auto-detector (W vs K bound) ................. `open` (ties survey 2026)
- N02 Online roofline sampler (measure beta_eff) ................ `wired` (wubu_roofline EMA, wubu_wm_kv)
- N03 Bandwidth-aware scheme selector (INT4kv vs FP16) .......... `wired` (wubu_kv_budget scheme_bits, ties N01)
- N04 Batch-size-aware quant switch ............................ `wired` (wubu_quant_selector batch_quant, ties N01)
- N05 Context-length-aware KV precision ladder .................. `wired` (wubu_quant_selector ctx_precision_ladder)
- N06 NUMA-bandwidth topology auto-detect ...................... `open`
- N07 Tiered-cache advisor (hot/warm/cold => precision) ......... `open` (ties A06)
- N08 Per-layer compute budget (skip floor) .................... `wired` (wubu_layer_floor, wubu_wm_kv)
- N09 Hardware-counters roofline (if PMC avail) ................ `wired` (wubu_quant_selector pmc_roofline fallback)
- N10 Energy-per-token metric (compute+HBM+interconnect) ....... `open`
- N11 TPOT predictor (given B, s, bits) ........................ `wired` (wubu_capacity_wall)
- N12 Capacity-wall predictor (KV GB vs RAM) ................... `wired` (wubu_capacity_wall fits-ram + b_star, ties 512k)
- N13 Compute-vs-bandwidth regime classifier ................... `wired` (wubu_capacity_wall regime, ties N01)
- N14 Mixture-of-depths router calibration ..................... `open` (ties J01)
- N15 Speculative acceptance model (pick K) .................... `wired` (wubu_spec_tuner K from acceptance, ties M12)
- N16 Cache-hit-rate feedback loop (prefix reuse) .............. `wired` (wubu_cache_fb, ties D02)
- N17 KV-footprint forecaster (pre-alloc advise) .............. `wired` (wubu_kv_budget forecast, ties N12)
- N18 OOM-risk early-warning (streaming engage) ................ `wired` (wubu_capacity_wall oom_risk, ties D04)
- N19 Adaptive chunk size (prefill vs decode) .................. `open` (ties D04)
- N20 Scheme A/B online (shadow quant compare) ............... `open`

## THEME O — Cross-discipline (DB/OS/formal/neuro) 7-hop wins
- O01 DB buffer-pool -> KV eviction (LRU-k with learned advice) . `wired` (wubu_lruk, ties A07b)
- O02 OS THP/hugepage KV arena (2MB pages) ..................... `wired` (wubu_hugepage + test_hugepage, plain-mmap fallback)
- O03 Compiler cost-model -> roofline auto-tuner ............... `open` (ties N)
- O04 Formal equiv -> quant kernel prove ...................... `wired` (wubu_equiv_check)
- O05 Neuro Titans -> bounded working-memory KV ............... `wired` (wubu_wm_kv bounded ring)
- O06 Neuro ADHD/lilypad -> focus-gated attention (distraction suppress) `wired` (wubu_attn_gate)
- O07 Neuro sink neurons -> attention-sink KEEP ................ `open` (ties L01)
- O08 RDMA net -> KV transfer (localhost analog) .............. `wired` (wubu_kv_transfer)
- O09 HPC roofline -> decode-bound proof ....................... `wired` (doc survey)
- O10 Z3/Alive2 -> GEMV rewrite verify ........................ `wired` (wubu_equiv_check)
- O11 TVM cost -> split-K auto-tune ........................... `open` (ties N)
- O12 ProofWright -> dequant equivalence ...................... `open` (ties F)
- O13 OS mmap prefault -> KV warm ............................. `open` (ties A06)
- O14 DB query plan -> decode schedule ....................... `open` (ties D)
- O15 Neuro theta/gamma -> attention rhythmic gate ........... `open`
- O16 Compiler autovec -> GEMV simd auto-select .............. `wired` (wubu_gemv_tune)
- O17 OS page cache -> KV LRU ................................ `wired` (wubu_kv_tier)
- O18 Formal bound -> OOM never (provable) ................... `open` (ties 512k)
- O19 DB WAL -> KV append-log replay ......................... `open`
- O20 Neuro plasticity -> online KV re-quant ................ `open`

## THEME P — Dispatch / kernel fusion (console-game discipline)
- P01 Q8_KV -> SWA -> split-K -> serial chain ................ `wired` (wubu_ssm)
- P02 Cache-line-aligned KV alloc (64B) ...................... `wired` (C03)
- P03 NUMA-aware weight pin (P-core affinity) ................ `wired` (wubu_affinity)
- P04 SIMD 512/16lane GEMV auto-dispatch ..................... `wired` (wubu_gemv_tune)
- P05 Tandem CPU/GPU split (RAM-bound offload) ............... `wired` (wubu_tandem)
- P06 Rambus banked KV (interleave banks) ................... `wired` (wubu_rambus)
- P07 Gamebud frame-budget (real wall-clock) ................. `wired` (wubu_gamebud)
- P08 GPU F32 GEMV -> cuda_gemv dispatch ..................... `wired`
- P09 AVX512 BF16 GEMV path .................................. `open`
- P10 q4_K GEMV (BitNet ternary) ............................. `wired` (B03)
- P11 int2 KV dequant fused in attn .......................... `open` (ties A04)
- P12 KV prefetch stream (non-temporal) ...................... `open`
- P13 Fused RoPE+quant KV write .............................. `open`
- P14 Fused dequant+GEMV (weight) ............................ `wired` (wubu_gemv_tune)
- P15 Speculative verify fused attn .......................... `open` (ties M)
- P16 Paged KV (block 16) alloc/free ........................ `wired` (kv_paged_attention)
- P17 Layer-stream resume (streaming load) .................. `wired` (D04)
- P18 Hug-page KV pool (arena) .............................. `open` (ties O02)
- P19 Weak-symbol CUDA stub (link-clean) .................... `wired`
- P20 Trace/span operator hook (DA-3) ........................ `wired` (wubu_selfimprove)

