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
- L04 InfiniGen KV prefetch (predict hot KV to fast tier) ......... `wired` (wubu_infiniten_prefetch, ties A06) (ties A06)
- L05 CacheBlend cross-request KV stitch .......................... `wired` (wubu_misc_gaps lcp_len)
- L06 Quest blockwise top-k KV retrieval (sub-linear attn) ........ `wired` (wubu_attn_tune quest_topk)
- L07 SnapKV cluster-based KV compression at layer depth ........... `wired` (wubu_kv_compress keep_clusters)
- L08 PyramidKV pyramid-accumulation KV reduction ................. `wired` (wubu_kv_compress pyramid_keep)
- L09 CIA KV (attention-score-driven compression) ................ `wired` (wubu_kv_compress keep_top_score)
- L10 SeerAttention-R dynamic sparse attention ................... `wired` (wubu_sys_tune seer_keep_frac, ties L11)
- L11 Native sparse attention (NSA, blockwise) ................... `wired` (wubu_sparse_attn block_sparse_mask)
- L12 MoBA memory-block attention (segment KV) .................... `wired` (wubu_sparse_attn moba_topk)
- L13 LM-Infinite landmark attention (soft prompt) .............. `wired` (wubu_lm_infinite landmark_positions)
- L14 Activation-beam KV offload (CPU/SSD tier) ................... `wired` (wubu_spec_variants offload_decision, ties A06/C05) (ties A06/C05)
- L15 KVShield adversarial-robust KV (no poison OOB) ............. `wired` (wubu_kv_shield bounds-check, ties F)
- L16 Elastic context (grow/shrink window online) ................ `wired` (wubu_ctx_manage elastic_window)
- L17 Dual-window (global sink + local) hybrid .................. `wired` (wubu_stream_kv sink+window = same design)
- L18 Layer-wise KV budget (deeper=less) ........................ `wired` (wubu_kv_budget layer_kv_budget)
- L19 Adaptive sink count (entropy-selected) .................... `wired` (wubu_kv_budget adaptive_sink)
- L20 Recurrent-compressed KV (SSM fallback > window) .......... `wired` (wubu_stream_kv + delta_net fallback, ties E03) (ties E03)

## THEME M — Speculative / self-draft (ADHD lilypad focus hops)
- M01 Self-speculative layer-skip draft (no 2nd model) ........... `wired` (wubu_layer_skip + wubu_self_cascade)
- M02 EAGLE-2 tree draft (tree verify, higher accept) ........... `wired` (wubu_eagle)
- M03 Medusa multi-head draft ................................... `wired` (wubu_medusa)
- M04 n-gram cascade (no weights) ............................... `wired` (doc 018)
- M05 CAS-Spec adaptive eager/defer ............................. `wired` (wubu_spec_cascade)
- M06 Lookahead parallel n-gram decoding ........................ `wired` (wubu_lookahead probe)
- M07 Rest-in-peace (REST) residual-Estimating draft ........... `wired` (wubu_more_spec rest_accept)
- M08 Online speculative tree restructuring .................... `wired` (wubu_more_spec tree_restructure)
- M09 Contrastive / lossless spec (no quality drop) ............ `wired` (wubu_more_spec contrastive_accept)
- M10 Draft-model distillation for hybrid arch .................. `wired` (wubu_more_spec distil_gate)
- M11 Spec verify via KV reuse (no re-forward) ................. `wired` (wubu_spec_variants kv_reuse_ok)
- M12 Acceptance-rate-adaptive K (per layer) ................... `wired` (wubu_spec_tuner per-layer K, ties N15)
- M13 Speculative + KV-quant co-design ......................... `wired` (wubu_spec_variants codesign, ties A04+L01)
- M14 Blockwise parallel verify (FlashDecoding style) .......... `wired` (wubu_spec_variants blockwise_verify_blocks, ties F)
- M15 Speculative routing for MoE (skip experts) ............... `wired` (wubu_more_spec spec_moe_skip, ties E05)
- M16 Self-cascade small Colonel (local draft) ................. `wired` (wubu_self_cascade)
- M17 LLM-Speculative cascade (big model verify) ............... `wired` (wubu_more_spec cascade_accept)
- M18 Online draft-model swapping (context-adaptive) ........... `wired` (wubu_more_spec swap_check)
- M19 Spec decode under layer-stream (resume draft) ............ `wired` (wubu_more_spec layer_resume, ties D04)
- M20 Cascade spec + early-exit hybrid ......................... `wired` (wubu_more_spec cascade_earlyexit, ties J03)

## THEME N — Roofline auto-tuner / adaptive compute
- N01 B* crossover auto-detector (W vs K bound) ................. `wired` (wubu_capacity_wall b_star, ties survey 2026) (ties survey 2026)
- N02 Online roofline sampler (measure beta_eff) ................ `wired` (wubu_roofline EMA, wubu_wm_kv)
- N03 Bandwidth-aware scheme selector (INT4kv vs FP16) .......... `wired` (wubu_kv_budget scheme_bits, ties N01)
- N04 Batch-size-aware quant switch ............................ `wired` (wubu_quant_selector batch_quant, ties N01)
- N05 Context-length-aware KV precision ladder .................. `wired` (wubu_quant_selector ctx_precision_ladder)
- N06 NUMA-bandwidth topology auto-detect ...................... `wired` (wubu_sys_tune numa_nodes fallback)
- N07 Tiered-cache advisor (hot/warm/cold => precision) ......... `wired` (wubu_ctx_manage tier_advice, ties A06)
- N08 Per-layer compute budget (skip floor) .................... `wired` (wubu_layer_floor, wubu_wm_kv)
- N09 Hardware-counters roofline (if PMC avail) ................ `wired` (wubu_quant_selector pmc_roofline fallback)
- N10 Energy-per-token metric (compute+HBM+interconnect) ....... `wired` (wubu_sys_tune energy_per_token)
- N11 TPOT predictor (given B, s, bits) ........................ `wired` (wubu_capacity_wall)
- N12 Capacity-wall predictor (KV GB vs RAM) ................... `wired` (wubu_capacity_wall fits-ram + b_star, ties 512k)
- N13 Compute-vs-bandwidth regime classifier ................... `wired` (wubu_capacity_wall regime, ties N01)
- N14 Mixture-of-depths router calibration ..................... `wired` (wubu_ctx_manage mod_tau, ties J01)
- N15 Speculative acceptance model (pick K) .................... `wired` (wubu_spec_tuner K from acceptance, ties M12)
- N16 Cache-hit-rate feedback loop (prefix reuse) .............. `wired` (wubu_cache_fb, ties D02)
- N17 KV-footprint forecaster (pre-alloc advise) .............. `wired` (wubu_kv_budget forecast, ties N12)
- N18 OOM-risk early-warning (streaming engage) ................ `wired` (wubu_capacity_wall oom_risk, ties D04)
- N19 Adaptive chunk size (prefill vs decode) .................. `wired` (wubu_attn_tune adaptive_chunk, ties D04)
- N20 Scheme A/B online (shadow quant compare) ............... `wired` (wubu_lm_infinite shadow state machine)

## THEME O — Cross-discipline (DB/OS/formal/neuro) 7-hop wins
- O01 DB buffer-pool -> KV eviction (LRU-k with learned advice) . `wired` (wubu_lruk, ties A07b)
- O02 OS THP/hugepage KV arena (2MB pages) ..................... `wired` (wubu_hugepage + test_hugepage, plain-mmap fallback)
- O03 Compiler cost-model -> roofline auto-tuner ............... `wired` (wubu_sys_tune tile_factor, ties N)
- O04 Formal equiv -> quant kernel prove ...................... `wired` (wubu_equiv_check)
- O05 Neuro Titans -> bounded working-memory KV ............... `wired` (wubu_wm_kv bounded ring)
- O06 Neuro ADHD/lilypad -> focus-gated attention (distraction suppress) `wired` (wubu_attn_gate)
- O07 Neuro sink neurons -> attention-sink KEEP ................ `wired` (wubu_lm_infinite sink_positions, ties L01)
- O08 RDMA net -> KV transfer (localhost analog) .............. `wired` (wubu_kv_transfer)
- O09 HPC roofline -> decode-bound proof ....................... `wired` (doc survey)
- O10 Z3/Alive2 -> GEMV rewrite verify ........................ `wired` (wubu_equiv_check)
- O11 TVM cost -> split-K auto-tune ........................... `wired` (wubu_attn_tune splitk_tune, ties N13)
- O12 ProofWright -> dequant equivalence ...................... `wired` (wubu_misc_gaps dequant_equiv, ties F) (ties F)
- O13 OS mmap prefault -> KV warm ............................. `wired` (wubu_misc_gaps prefault, ties A06) (ties A06)
- O14 DB query plan -> decode schedule ....................... `wired` (wubu_db_cross plan_decode, ties D) (ties D)
- O15 Neuro theta/gamma -> attention rhythmic gate ........... `wired` (wubu_misc_gaps rhythmic_gate)
- O16 Compiler autovec -> GEMV simd auto-select .............. `wired` (wubu_gemv_tune)
- O17 OS page cache -> KV LRU ................................ `wired` (wubu_kv_tier)
- O18 Formal bound -> OOM never (provable) ................... `wired` (wubu_db_cross kv_invariant_ok, ties 512k) (ties 512k)
- O19 DB WAL -> KV append-log replay ......................... `wired` (wubu_db_cross WAL replay)
- O20 Neuro plasticity -> online KV re-quant ................ `wired` (wubu_attn_kernels plasticity_bits)

## THEME P — Dispatch / kernel fusion (console-game discipline)
- P01 Q8_KV -> SWA -> split-K -> serial chain ................ `wired` (wubu_ssm)
- P02 Cache-line-aligned KV alloc (64B) ...................... `wired` (C03)
- P03 NUMA-aware weight pin (P-core affinity) ................ `wired` (wubu_affinity)
- P04 SIMD 512/16lane GEMV auto-dispatch ..................... `wired` (wubu_gemv_tune)
- P05 Tandem CPU/GPU split (RAM-bound offload) ............... `wired` (wubu_tandem)
- P06 Rambus banked KV (interleave banks) ................... `wired` (wubu_rambus)
- P07 Gamebud frame-budget (real wall-clock) ................. `wired` (wubu_gamebud)
- P08 GPU F32 GEMV -> cuda_gemv dispatch ..................... `wired`
- P09 AVX512 BF16 GEMV path .................................. `wired` (wubu_bf16_gemv, runtime dispatch + F32 fallback)
- P10 q4_K GEMV (BitNet ternary) ............................. `wired` (B03)
- P11 int2 KV dequant fused in attn .......................... `wired` (wubu_attn_kernels int2_dequant, ties A04) (ties A04)
- P12 KV prefetch stream (non-temporal) ...................... `wired` (wubu_misc_gaps kv_prefetch)
- P13 Fused RoPE+quant KV write .............................. `wired` (wubu_misc_gaps fused_rope_quant)
- P14 Fused dequant+GEMV (weight) ............................ `wired` (wubu_gemv_tune)
- P15 Speculative verify fused attn .......................... `wired` (wubu_attn_kernels spec_verify_fused, ties M) (ties M)
- P16 Paged KV (block 16) alloc/free ........................ `wired` (kv_paged_attention)
- P17 Layer-stream resume (streaming load) .................. `wired` (D04)
- P18 Hug-page KV pool (arena) .............................. `wired` (wubu_hugepage, ties O02) (ties O02)
- P19 Weak-symbol CUDA stub (link-clean) .................... `wired`
- P20 Trace/span operator hook (DA-3) ........................ `wired` (wubu_selfimprove)


## Theme Q-T: 2026 KV-cache / test-time-compute research sweep (fresh gaps)
Status: `open` = not yet in engine; `wired` = implemented+tested.
- Q01 CentroidKV cross-token KV clustering (semantic centroids) .... `wired` (wubu_kv2026b centroidkv)
- Q02 ChunkKV semantic chunk-level KV compression ................ `wired` (wubu_kv2026 chunkkv_evict)
- Q03 KVzip query-agnostic KV compression + context reconstruction  `wired` (wubu_kv2026 kvzip_importance)
- Q04 R-KV redundancy-aware KV eviction (reasoning models) ....... `wired` (wubu_kv2026b rkv_redundancy)
- Q05 OBCache Hessian-guided token saliency pruning ............. `wired` (wubu_kv2026b obcache_saliency proxy)
- Q06 KeyDiff key-similarity KV eviction ....................... `wired` (wubu_kv2026b keydiff_evict)
- Q07 LAVa layer-wise eviction w/ dynamic head+layer budget .... `wired` (wubu_kv2026 lava_budget, ties L18)
- Q08 PolyKV shared asymmetrically-compressed KV pool (agents) .. `wired` (wubu_ttc polykv_coherent)
- Q09 FreeKV speculative top-k KV retrieval ................... `wired` (wubu_kv2026 freekv_topk, ties L11)
- Q10 TTKV temporal-tiered KV placement (hetero precision) ..... `wired` (wubu_kv2026 ttkv_tier, ties N07)
- Q11 DASH-KV hash-based token-level attn scheduling ........... `wired` (wubu_kv2026c dashkv_schedule)
- Q12 TARDIS GPU-centric KV service w/ host spillover ......... `wired` (wubu_sys2026 tardis_spill)
- Q13 KVDrive multi-tier CPU/DRAM/SSD KV management ........... `wired` (wubu_sys2026 kvdrive_tier, ties O02)
- Q14 ScoutAttention layer-ahead CPU precompute + GPU decode .. `wired` (wubu_sys2026 scout_eligible)
- Q15 HotPrefix hotness-aware KV scheduling (prefix sharing) .. `wired` (wubu_ttc hotprefix_priority, ties L05)
- Q16 AlignedServe prefix-aware batching scheduler ........... `wired` (wubu_sys2026 aligned_lcp, ties D04)
- Q17 CoDec prefix-shared decoding kernel .................... `wired` (wubu_sys2026 codec_share)
- Q18 SparKV overhead-aware KV loading (cloud<->device) ...... `wired` (wubu_sys2026 sparkv_load, ties A06)
- Q19 HeteroCache heterogeneous KV compression retrieval ...... `wired` (wubu_kv2026c hetero_bits)
- Q20 Test-time-compute budget allocator (adaptive token budget)  `wired` (wubu_ttc budget_steps)
- R01 Inference-time scaling controller (budget vs accuracy) .. `wired` (wubu_ttc scaling_factor)
- R02 Agentic context-axis efficiency (curated input context) . `wired` (wubu_sys2026 agentic_ctx)
- R03 CATTS contrastive adaptive token scaling .............. `wired` (wubu_ttc catts_tokens)
- R04 Reasoning-model KV redundancy profiler ................. `wired` (wubu_kv2026c redundancy_profile, ties Q04)
- R05 Multi-agent shared KV pool coherence ................... `wired` (wubu_kv2026c multiagent_coherence, ties Q08)

## Theme S-U: 2026 linear-attention / ternary-weight / multimodal-KV sweep (fresh gaps)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### S: Linear / recurrent attention hybrids
- S01 Gated DeltaNet delta-rule state update ..................... `wired` (wubu_linear_attn deltanet_update)
- S02 Gated DeltaNet-2 decoupled erase/write gate ....................... `wired` (wubu_dn2 dn2_update)
- S03 Mamba-2 / SSM selective-scan gated state decay .............. `wired` (wubu_linear_attn mamba2_update)
- S04 GLA gated linear attention (per-head state gate) ................. `wired` (wubu_linear_attn gla_update)
- S05 RetNet / GSA retention decay matrix .............................. `wired` (wubu_linear_attn retnet_update)
- S06 Hybrid layer scheduler (3:1 GDN:GA mix, recurrent vs attn) ...... `wired` (wubu_agentic_kv hybrid_is_recurrent) (ties layer_skip)
- S07 HGRN2 / GSA state-expansion gated RNN ........................... `wired` (wubu_linear_attn hgrn2_update)
### T: Sub-2-bit / ternary weights
- T01 BitNet ternary weight pack (2-bit/val, 4/byte) + dequant ......... `wired` (wubu_ternary pack/unpack)
- T02 mpGEMM ternary matvec (F32 = sum ternary_w . int8_act) ......... `wired` (wubu_ternary mpgemv)
- T03 Ternary absmax scaling (W scaled to [-1,1] before ternarize) ..... `wired` (wubu_ternary scale)
- T04 Ternary training-aware (Straight-Through Estimator proxy) ....... `wired` (wubu_dn2 ternary_ste)
### U: Multimodal / agentic KV
- U01 Gemma-4 shared-KV across layers (reuse KV of earlier layer) ...... `wired` (wubu_agentic_kv shared_kv_source)
- U02 DeepSeek-V4 CSA/HCA compressed attention (128->1 entry) ......... `wired` (wubu_agentic_kv csa_compress)
- U03 LMCache vision-token hashing (KV reuse across requests) ......... `wired` (wubu_agentic_kv vision_hash)
- U04 LOOK-M multimodal KV prune (drop least-important vision tokens) .. `wired` (wubu_agentic_kv lookm_keep)
- U05 Agentic memory KV compaction (summarize old turns into slots) .... `wired` (wubu_agentic_kv agentic_compact)

## Theme V-W: 2026 parallel-speculative / length-generalization PE sweep (fresh gaps)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### V: Parallel speculative decoding
- V01 EAGLE-3 feature-level drafting (predict hidden feats, not tokens) .... `wired` (wubu_parallel_spec eagle3_draft)
- V02 P-EAGLE parallel drafting (K independent drafts, tree-verify) ........ `wired` (wubu_parallel_spec peagle_verify)
- V03 Tree-attention verification mask (beam-shaped attention) ........... `wired` (wubu_parallel_spec tree_attn_parents) (ties more_spec)
- V04 Kangaroo double-early-exit self-speculative ....................... `wired` (wubu_parallel_spec kangaroo_accept) (ties more_spec)
### W: Length-generalization positional encoding
- W01 NoPE (no positional encoding; attention carries position) ........ `wired` (wubu_parallel_spec nope_enabled)
- W02 ALiBi-style distance bias (extrapolatable slope) ................. `wired` (wubu_parallel_spec alibi_bias) (ties rope)
- W03 Attention sandwitch / FFN-first (length-robust order) ............ `wired` (wubu_parallel_spec ffn_first_enabled)

## Theme X-Y: 2026 MoE routing + RAG/retrieval KV sweep (fresh gaps)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### X: Mixture-of-Experts routing
- X01 Top-K router (softmax gate, pick K of N routed experts) .......... `wired` (wubu_moe_rag topk_route)
- X02 Expert-Choice routing (expert picks top tokens; balanced) ......... `wired` (wubu_moe_rag expert_choice)
- X03 Shared-expert always-on (routed + shared aggregation) ............ `wired` (wubu_moe_rag shared_expert)
- X04 Sigmoid gating (independent expert probs, not softmax) ........... `wired` (wubu_moe_rag sigmoid_gate)
- X05 Predictive expert caching (ExpertFlow: prefetch by predicted route) `wired` (wubu_moe_rag expert_prefetch)
- X06 Capacity factor / token dropping (overflow guard) ................ `wired` (wubu_moe_rag capacity_factor)
### Y: Retrieval-augmented / context-independent KV
- Y01 KV Packet context-independent caching (reusable per-doc KV) ....... `wired` (wubu_moe_rag kvpacket_doc)
- Y02 RACC retrieval-aware KV compression (keep retrieved chunks) ...... `wired` (wubu_moe_rag racc_keep)
- Y03 CAG cache-augmented generation (preload doc KV, no per-query retr)  `wired` (wubu_moe_rag cag_ready)
- Y04 Cross-document KV isolation (per-doc KV namespace) .............. `wired` (wubu_moe_rag crossdoc_ns)

## Theme Z-AA: 2026 long-context eval + QAT sweep (fresh gaps)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### Z: Long-context evaluation harness
- Z01 NIAH-2 multi-needle injection + retrieval scoring ................ `wired` (wubu_eval_qat niah_inject)
- Z02 RULER retrieval category (variable needles) .................... `wired` (wubu_eval_qat ruler_retrieve)
- Z03 RULER multi-hop tracing (chain of keys) ....................... `wired` (wubu_eval_qat ruler_multihop)
- Z04 RULER aggregation (count/freq over context) ................... `wired` (wubu_eval_qat ruler_aggregate)
- Z05 synthetic haystack generator (configurable len/noise) ......... `wired` (wubu_eval_qat haystack_gen)
### AA: Quantization-aware training
- AA01 Fake-quant (round-to-nearest in fake precision) .............. `wired` (wubu_eval_qat fakequant)
- AA02 QAT straight-through estimator (grad passes past quant) ...... `wired` (wubu_eval_qat qat_ste, ties T04) (ties T04)
- AA03 per-channel quant + dequant (QAT weight dtype) .............. `wired` (wubu_eval_qat dequant_pc)
- AA04 quantization noise injection (robustness augmentation) ....... `wired` (wubu_eval_qat noise_inject)

## Theme AB-AC: 2026 disaggregated PD serving + dynamic-depth sweep (fresh gaps)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### AB: Disaggregated prefill/decode serving
- AB01 Prefill/decode pool split (independent scaling) ................ `wired` (wubu_pd_serve pd_split)
- AB02 KV handoff scheduler (transfer KV prefill->decode when ready) .... `wired` (wubu_pd_serve kv_handoff_ready)
- AB03 Pull-based decode routing (drain prefill spikes) .............. `wired` (wubu_pd_serve pull_route, ties sys2026) (ties sys2026)
- AB04 Heterogeneous pool mapping (compute-dense prefill / bw-dense decode) `wired` (wubu_pd_serve hetero_map, ties KVDrive) (ties KVDrive)
- AB05 KV transfer cost model (size/bandwidth vs TTFT budget) ......... `wired` (wubu_pd_serve kv_xfer_fits)
- AB06 Prefix-aware PD routing (reuse prefill across requests) ........ `wired` (wubu_pd_serve prefix_reuse, ties CacheBlend/LCP) (ties CacheBlend/LCP)
### AC: Dynamic compute / mixture-of-depths
- AC01 Per-token layer-skipping router (MoD gating) .................. `wired` (wubu_pd_serve mod_execute, ties layer_skip) (ties layer_skip)
- AC02 Mixture-of-depths capacity (max active layers per token) ...... `wired` (wubu_pd_serve mod_capacity)
- AC03 Early-exit confidence threshold (dynamic depth) ............... `wired` (wubu_pd_serve early_exit, ties early_exit) (ties early_exit)

## Theme AD-AE: AGI operating-system runtime + memory (from 2026 research sweep)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### AD: Agentic-OS runtime governance (AgentCgroup 2026 / 9P capability surface)
- AD-01 Per-agent 9P capability enforcement (each agent subtree bounded, not full FS) `wired` (wubu_agentic_os 9p_cap_allowed)
- AD-02 Agent scheduler: skip-if-running + exponential backoff (cron-style) `wired` (wubu_agentic_os backoff/skip)
- AD-03 Durable-execution resume for long-running agents (state checkpoint) `wired` (wubu_agentic_os checkpoint)
- AD-04 cgroup/BPF attach bounding agent CPU/RAM/IO (AgentCgroup 2026) `wired` (wubu_agentic_os resbound_check, ties syscalls)
### AE: Agentic memory (TeleMem/HiMem/Redis 2026 3-tier + consolidation)
- AE-01 Episodic->semantic consolidation pass (distill events to facts) `wired` (wubu_agentic_mem consolidate/tier)
- AE-02 Semantic dedup / merge (avoid fact duplication) `wired` (wubu_agentic_mem dedup)
- AE-03 Hierarchical tiers: working / session / long-term retrieval `wired` (wubu_agentic_mem tier)
- AE-04 Memory retrieval ranking by recency + importance (forgetting curve) `wired` (wubu_agentic_mem retrieval_score)

## Theme AF: 100-goalpost AGI-OS integration (from 7-hop KB sweep)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### AF: Capability/Zero-Trust kernel (items 86-100)
- AF-01 Per-agent 9P capability enforcement (deny-by-default subtree) ......... `wired` (wubu_agentic_os 9p_cap_allowed, pass 29)
- AF-02 Deny-by-default tool registry (capability list per agent) `wired` (wubu_capzero capset deny-by-default)
- AF-03 Encrypted agent memory at rest (AES-CTR over blobs) `wired` (wubu_capzero mem_crypt CTR)
- AF-04 Non-human identity (NHI) + token issuance per agent `wired` (wubu_capzero nhi_issue)
### AF: Latency-class scheduler (items 41-50)
- AF-05 Latency-class enum + EDF/RM scheduler hook (HRT/SRT/DT) `wired` (wubu_latency edf_order)
- AF-06 WCET + jitter budget accounting `wired` (wubu_latency wcet_account/deadline_miss)
- AF-07 Agent-Contract SLO enforcement (TTFT/turn/throughput) `wired` (wubu_latency slo_check)
### AF: Context virtual-memory hierarchy (items 51-65)
- AF-08 4-level context hierarchy (L1 gen/L2 session/L3 long/L4 cross) `wired` (wubu_ctxvm ctx_tier)
- AF-09 Demand-paging eviction (FIFO + working-set) over KV `wired` (wubu_ctxvm evict_fifo/resident)
- AF-10 Semantic cache reuse across agents (vector sim) `wired` (wubu_ctxvm cosine/sem_cache_hit)
### AF: Safety kernel (items 66-85)
- AF-11 Non-tamperable interrupt (stop outside reasoning loop) `wired` (wubu_safekern stop_honored)
- AF-12 Graduated containment (proportional, reversible) `wired` (wubu_safekern containment_level/reversible)
- AF-13 Stability-plasticity guard (RSI cannot weaken 512K gate) `wired` (wubu_safekern rsi_mutation_ok)

## Theme AG: Missing needs (2026 agentic/AGI-OS gap research)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### Missing-need gaps (N1-N8)
- [x] AG-01 Runaway-loop guard: max step-count + hard deadline + terminate (OWASP LLM10/ASI08) `wired` (wubu_loopguard loop_may_continue)
- [x] AG-02 Goal-hijack / injection defense: control-plane vs data-plane separation (ASI01/LLM01) `wired` (wubu_planediv plane_enforce)
- [x] AG-03 Memory/context poisoning detection: cross-session replay + divergence flag (ASI06/L3×T3) `wired` (wubu_planediv mem_fingerprint/diverged/replay_flagged)
- [x] AG-04 Closed-loop deliberative planning: verify world-state + replan (open-loop problem) `wired` (research)  # (wubu_worldmodel closed_step/divergence)
- [x] AG-05 Trajectory-level audit attribution: append-only per-action record (L7×T1-4) `wired` (wubu_loopguard traj_append)
- [x] AG-06 Tool-abuse / excessive-agency cap: per-agent tool-call rate limit (LLM06/ASI02) `wired` (wubu_loopguard tool_allowed)
- [x] AG-07 Inter-agent message authentication (ASI07) `wired` (research)  # (wubu_agentauth mac/verify, keyed-FNV, default-deny)
- [x] AG-08 JIT provisioning + HITL gating: sensitive-action approval token (ASI08/strata) `wired` (wubu_loopguard hitl_approve)

## Theme AH: AGI-at-home meta-game (7-hop lily-pad KB sweep)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### Meta-game / coordination / credit gaps
- [x] AH-01 Concurrent-modification intent-lock before editing shared module `wired`  # (wubu_coord lock_acquire/release)
- [x] AH-02 Serializability at quiescence (MTPO targeted repair) `wired`  # (wubu_coord txn_committable)
- [x] AH-03 Shared-memory access-control (right agents see right memory) `wired`  # (wubu_coord mem_allowed)
- [x] AH-04 Coordination heartbeat / conflict resolution dialogue `wired`  # (wubu_coord resolve_conflict/heartbeat_alive)
- [x] AH-05 Open-ended self-modifying agent archive (DGM branch tree) `wired`  # (wubu_metagame archive_add/best)
- [x] AH-06 Empirical fitness validation (bench, don't prove) `wired`  # (wubu_metagame accept_child/archive_best)
- [x] AH-07 Sandboxed self-modification (no web/fs escape) `wired`  # (wubu_metagame2 sandbox_allow)
- [x] AH-08 Anti-hallucinated-self-log (don't trust own unverified "passed") `wired`  # (wubu_metagame accept_child verified gate)
- [x] AH-09 Skill library (reusable, non-parametric, replayable) `wired`  # (wubu_metagame2 skill_add/topk)
- [x] AH-10 Continual learning without forgetting (replay buffer) `wired`  # (wubu_metagame2 replay_add reservoir)
- [x] AH-11 Intrinsic metacognition (calibrate own confidence) `wired`  # (wubu_metagame2 metacog_update/calibrated)
- [x] AH-12 Turn-level credit assignment (TD, verifier-anchored) `wired`  # (wubu_credit turn_credit/credit_sign)
- [x] AH-13 Self-improvement delta metric (did mutation help?) `wired`  # (wubu_metagame improvement_delta)
- [x] AH-14 Resource envelope profiler (auto-detect VRAM/BW/RAM) `wired`  # (wubu_resource pick_tier/est_toks)
- [x] AH-15 Graceful degradation tiers (70B->14B->7B on OOM) `wired`  # (wubu_resource degrade_tier)

## Theme AV: Vectors — 7-hop Kevin-Bacon lily-pad KB sweep
Status: `open` = not yet in engine; `wired` = implemented+tested.
### Vector substrate gaps (8 gaps AV01-AV08)
- AV01 ANN index (HNSW layered graph + IVFFlat hybrid) for KV + semantic cache `wired` (wubu_vecsearch hnsw_insert/search)
- AV02 RaBitQ + PQ quantization for the KV vectors `wired` (wubu_vecsearch rabitsq_quantize/estimate)
- AV03 Cross-session KV reuse `wired` (wubu_vecsearch kvcache session keys)
- AV04 Similarity-based KV eviction `wired` (wubu_vecsearch evict_by_similarity)
- AV05 FlashAttention-style tiling (online-softmax 2-pass) `wired` (wubu_vecsearch flash tile)
- AV06 MRL flexible-dim embeddings `wired` (wubu_vecsearch mrl dims)
- AV07 On-device vector DB (ANN-backed, low-RAM) `wired` (wubu_vecsearch vecdb)
- AV08 Agentic vector memory (ANN-indexed episodic store) `wired` (wubu_vecsearch agentic_mem)
Status: `wired` (wubu_vecsearch, test_vecsearch PASSES)
EOF

## Theme AW: Causal + Neuro-Symbolic + Temporal — 7-hop KB sweep
Status: `open` = not yet in engine; `wired` = implemented+tested.
### Causal/neuro-symbolic substrate gaps (10 gaps AW01-AW10)
- AW01 Structural Causal Model (DAG of cause->effect) `wired` (wubu_causal scm)
- AW02 do-intervention via truncated factorization `wired` (wubu_causal do_intervene)
- AW03 Counterfactual query (abduction-action-prediction) `wired` (wubu_causal counterfactual)
- AW04 Identifiability check (backdoor criterion) `wired` (wubu_causal identifiable)
- AW05 Symbolic verifier in the decode path `wired` (wubu_symbolic verify_tokens)
- AW06 Temporal belief revision (Bayesian, timestamped) `wired` (wubu_symbolic belief_revise)
- AW07 Logic engine (unification + forward chaining) `wired` (wubu_symbolic logic)
- AW08 PDDL/STRIPS goal-directed planner `wired` (wubu_symbolic plan)
- AW09 Abductive diagnosis (best-explanation search) `wired` (wubu_symbolic abduce)
- AW10 Counter-abduction (rival-explanation defeat) `wired` (wubu_symbolic defeat)
Status: `wired` (wubu_causal + wubu_symbolic, test_causal_symbolic PASSES)

## Theme AX: Self-Improving Code + Sandboxed Execution + Verifiable Tool-Use
Status: `open` = not yet in engine; `wired` = implemented+tested.
### Self-Improving Code + Sandbox (12 gaps AX01-AX12, 2 research)
- AX01 DGM empirical gate + regression test runner `wired` (wubu_dgm.c)
- AX02 seccomp-bpf sandbox `open` (research: kernel-level, exceeds single C11 module)
- AX03 formal verification `open` (research: proof-assistant-level)
- AX04 MCP-compatible tool schema + dispatch `wired` (wubu_tooluse.c)
- AX05 program synthesis (spec→C11) `wired` (wubu_synth.c)
- AX06 self-evolution loop (propose→verify→commit→regress) `wired` (wubu_evolve.c)
- AX07 code exec verifier → feeds loopguard `wired` (wubu_codeexec.c)
- AX08 sandbox capability bridge → safekern `wired` (wubu_sandbox_safekern.c)
- AX09 C11 type-check + invariant gate `wired` (wubu_verify.c)
- AX10 spec→C11 codesynth `wired` (wubu_codesynth.c)
- AX11 evolve loop extension `wired` (wubu_evolve.c)
- AX12 evolve+exec+verify bridge → loopguard `wired` (test_axi)
## Theme BB: Continual Learning + Catastrophic Forgetting Prevention
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7 gaps (BB01-BB05 wired, BB06-BB07 research):
- BB01 Experience replay buffer (reservoir sampling) `wired` (wubu_replay.c)
- BB02 EWC consolidation (Fisher importance + quadratic penalty) `wired` (wubu_ewc.c)
- BB03 Task boundary detection (OOD via tok/s divergence) `wired` (wubu_taskbd.c)
- BB04 Knowledge distillation (teacher snapshot + KL soft targets) `wired` (wubu_distill.c)
- BB05 Integration: continual learning loop feeds loopguard `wired` (test_continual)
- BB06 SI path-integral importance (Zenke 2017; omega accumulates the gradient path) `wired` (wubu_si, test_debt)
- BB07 Dark experience replay (teacher-soft-target replay; the KL vs CE bug caught by DA) `wired` (wubu_der, test_debt)

## Theme CC: Multimodal Grounding (Vision + Audio + Text)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 8 gaps (CC01-CC05, CC06, CC07 wired; CC08 research):
- CC01 Vision encoder (ViT patch embedding from scratch) `wired` (wubu_vision.c)
- CC02 Audio encoder (mel-spectrogram + radix-2 real FFT) `wired` (wubu_audio.c)
- CC03 Cross-modal alignment (CLIP-style projection to 512-dim) `wired` (wubu_mm_align.c)
- CC04 Multimodal adapter (align → quantize → token IDs) `wired` (wubu_mm_adapter.c)
- CC05 Positional KV integration (prepend prefix, no EAMM at 512K) `wired` (wubu_mm_kv.c)
- CC06 Multimodal token pipeline (image→pseudo-tokens) `wired` (test_multimodal)
- CC07 Integration + safety gate → decode path `wired` (test_multimodal)
- CC08 End-to-end multimodal gen_text `open` (research: needs trained visual vocab)

## Theme DD: Multi-Agent Consensus + Inter-Agent Auth (BFT)
Status: `wired` = implemented+tested.
### 7 gaps closed as tested C11:
- DD01 BFT consensus (3-round voting, 2/3+1 threshold) `wired` (wubu_bft.c)
- DD02 Threshold signing (aggregate agent signatures) `wired` (wubu_threshsig.c)
- DD03 Inter-agent identity + zero-trust auth `wired` (wubu_agentid.c)
- DD04 Semantic consensus (claim + verify + dispute) `wired` (wubu_semcons.c)
- DD05 Fraud detection (outlier + dispute + trust decay) `wired` (wubu_fraud.c)
- DD06 Trust-gated voting weight `wired` (test_multiconsensus)
- DD07 Integration: consensus → DGM archive `wired` (test_multiconsensus)

## Theme EE: Symbolic Regression + Automated Theorem Proving + Invariant Discovery
Status: wired = implemented+tested; open = research-level.
### 7 gaps (EE01-EE06 wired, EE07 open):
- EE01 Symbolic regression (GP equation discovery) wired (wubu_symreg.c)
- EE02 SINDy (sparse dynamics identification) wired (wubu_sindy.c)
- EE03 CEGIS (counterexample-guided config synthesis) wired (wubu_cegis.c)
- EE04 Automated theorem proving (natural-deduction) wired (wubu_prover.c)
- EE05 Invariant discovery (loop invariant synthesis) wired (wubu_invariant.c)
- EE06 Integration: discovered law to loopguard/safekern wired (test_ee.c)
- EE07 Closed-loop self-verification (divergence -> re-synthesize -> replace) `wired` (wubu_reverify, test_debt)

## Theme FF: Bayesian Optimization + Uncertainty Quantification + Active Learning
Status: wired = implemented+tested.
### 7 gaps closed as tested C11:
- FF01 Gaussian Process surrogate (RBF kernel, Cholesky predict) wired (wubu_gp.c)
- FF02 Acquisition functions (EI / UCB / PI) wired (wubu_acq.c)
- FF03 Bayesian Optimization loop wired (wubu_bo.c)
- FF04 Uncertainty Quantification (bootstrap + conformal) wired (wubu_uq.c)
- FF05 Active Learning (uncertainty sampling / QBC) wired (wubu_active.c)
- FF06 Thompson Sampling / bandits wired (wubu_bandit.c)
- FF07 Integration with recursive_optimize wired (test_ff.c)

## Theme GG: Reinforcement Learning (Policy Gradients, Actor-Critic, PPO, PPO-Clip)
Status: wired = implemented+tested.
### 7 gaps closed as tested C11:
- GG01 REINFORCE (policy gradient, Monte-Carlo) wired (wubu_reinforce.c)
- GG02 Baseline / variance reduction wired (wubu_policy.c)
- GG03 Actor-Critic (TD advantage) wired (wubu_actor_critic.c)
- GG04 PPO (clipped surrogate) wired (wubu_ppo.c)
- GG05 DQN / Q-learning (value-based) wired (wubu_dqn.c)
- GG06 Value iteration / Bellman backup wired (wubu_value.c)
- GG07 Unified policy/value interface + integration wired (test_gg.c)

## Theme HH: Inference Acceleration (Speculative Decoding + Paged KV + MoE Routing + Continuous Batching)
Status: wired = implemented+tested.
### 7 gaps closed (HH01-HH07)
- HH01 Speculative decoding (draft/verify/reject) `wired` (wubu_specdec.c)
- HH02 Paged KV cache (block table, CoW, prefix) `wired` (wubu_pagedkv.c)
- HH03 MoE capacity routing + load-balancing `wired` (wubu_moeroute.c)
- HH04 Continuous batching (iterative scheduling) `wired` (wubu_contbatch.c)
- HH05 Medusa self-draft heads (tree draft) `wired` (wubu_medusa.c)
- HH06 KV quantization (INT8 group-wise) `wired` (wubu_quantkv.c)
- HH07 Integration: speedup model `wired` (test_hh.c)

## Theme IJ: Energy-aware inference (power-budgeted decode)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7 gaps (IJ01-IJ07)
- IJ01 Energy roofline model (E = mem_bytes*J/byte + flops*J/flop; decode is memory-bound) `wired` (wubu_energy estimate/j_per_token)
- IJ02 Energy-per-token ledger with a hard budget (arXiv 2603.20224 E/token) `wired` (wubu_energy ledger)
- IJ03 Power-cap frequency scheduler (DVFS P~CV²f; memory-bound decode: lower f = higher tok/J, CCGrid 2026) `wired` (wubu_energy freq_for_cap/jpt_at_freq)
- IJ04 Energy-budget early exit (stop when the remaining budget can't afford the next token) `wired` (wubu_energy should_continue)
- IJ05 Energy-tier KV offload (choose the tier by amortized J/byte, not capacity) `wired` (wubu_energy choose_tier)
- IJ06 Speculative-decoding energy break-even (draft_jpt < target_jpt*accept_rate; DA-verified model) `wired` (wubu_energy spec_breakeven/round)
- IJ07 Budget-driven operator (pick the lowest-J/token config clearing the throughput gate) `wired` (wubu_energy pick_config)
Status: `wired` (wubu_energy, test_energy PASSES)

## Theme IL: Modern Hopfield / associative memory
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7 gaps (IL01-IL07)
- IL01 Modern-Hopfield retrieval (xi' = X^T softmax(beta*X*xi)) `wired` (wubu_hopfield retrieve)
- IL02 Attention equivalence (the softmax attention IS the Hopfield update; beta = 1/sqrt(d)) `wired` (wubu_hopfield beta_attention)
- IL03 Exponential storage capacity (C ~ exp(alpha*d), beats the Hebbian O(d)) `wired` (wubu_hopfield capacity)
- IL04 Associative recall (pattern completion from a corrupted cue) `wired` (wubu_hopfield denoise)
- IL05 Memory decay (the STM->LTM forgetting curve) `wired` (wubu_hopfield decay)
- IL06 Consolidation (replay/reward strengthens the stored pattern) `wired` (wubu_hopfield consolidate)
- IL07 The operator: top-k KV-slot retrieval by Hopfield overlap `wired` (wubu_hopfield topk)
Status: `wired` (wubu_hopfield, test_hopfield PASSES)

## Theme IM: Preference alignment + unlearning (the AGI's values + right-to-be-forgotten)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7 gaps (IM01-IM07)
- IM01 DPO implicit reward + loss (Rafailov 2023; reward-model-free alignment) `wired` (wubu_align dpo_reward/dpo_loss)
- IM02 KTO binary-desirability loss (Ethayarajh 2024; the Kahneman-Tversky reference point; DA caught the sigmoid sign) `wired` (wubu_align kto_loss)
- IM03 Gradient-ascent unlearning (the approximate exact-unlearning) `wired` (wubu_align unlearn_ascent)
- IM04 KL-anchored unlearning (forget without collapse) `wired` (wubu_align unlearn_anchor_weight)
- IM05 Preference-ranked alignment replay reservoir `wired` (wubu_align buffer/topk/mean)
- IM06 Reward-hacking / value-drift monitor (spiked mean + collapsed variance) `wired` (wubu_align monitor)
- IM07 The operator: (alignment, cost) frontier config pick `wired` (wubu_align pick_config)
Status: `wired` (wubu_align, test_align PASSES)

## Theme IN: Predictive coding / free energy / active inference
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7 gaps (IN01-IN07)
- IN01 Prediction-error layers (e = x - mu_hat) `wired` (wubu_freeenergy pred_error)
- IN02 Variational free energy (F = -accuracy + complexity) `wired` (wubu_freeenergy free_energy)
- IN03 Active inference: expected-free-energy policy prior (softmax(-gamma*G)) `wired` (wubu_freeenergy policy_prior)
- IN04 Precision-weighted prediction errors `wired` (wubu_freeenergy precision_weight)
- IN05 Perception-action loop (perception = PE minimization, action = EFE minimization) `wired` (wubu_freeenergy percept_step)
- IN06 Epistemic value (information-gain curiosity bonus) `wired` (wubu_freeenergy epistemic_value)
- IN07 The operator: free-energy-gated model selection `wired` (wubu_freeenergy pick_model)
Status: `wired` (wubu_freeenergy, test_freeenergy PASSES)

## Theme IO: KV-cache eviction / compression 2026 frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (2603.20397 KV survey; KeyDiff 2504.15364; KVQuant NeurIPS 2024)
- IO01 H2O heavy-hitter token retention (accumulated-attention greedy eviction) `open`
- IO02 StreamingLLM attention-sink keep + rolling window `open` (ties L-theme)
- IO03 SnapKV observation-window pooling + important-prefix retention `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO04 Proxy-token one-shot eviction (softmax-probability batch discard) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO05 InfiniPot novelty distillation (novelty-weighted retain at capacity) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO06 HASHEVICT LSH pre-attention eviction (SimHash hamming-distance prune) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO07 RocketKV two-stage coarse eviction + dynamic sparse selection `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO08 Ada-KV head-adaptive budget (eviction-loss upper bound, head-sparse reallocation) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO09 KeyDiff key-similarity eviction (attention-sink position varies per head/layer) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO10 KVQuant attention-sink-aware quantization + outlier sparse store (3-bit, 4.8x ctx) `open`
- IO11 Semantic-sponsorship KV retention (semantic importance, not score) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO12 Pyramidal/block-wise eviction under block prompt processing (eviction-error compounding) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO13 Accumulated-attention tracker with per-token running sums (O(1) update) `open`
- IO14 Eviction-loss upper-bound model (formal eviction-error budget) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO15 Per-head sink-token discovery (sink position varies across heads/layers) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO16 Coarse-to-fine two-stage selection (RocketKV-style page granularity) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO17 KV-reconstruction autoencoder importance (regenerate-input criticality) `open`
- IO18 LSH bucket refresh policy (hamming-distance threshold adaptation) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO19 Novelty scoring by embedding distance to the retained set `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO20 Pooled observation window (SnapKV 1D pooling, cluster context) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO21 Proxy-token selection via compressed cue (small subset scoring) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO22 Eviction + quantization hybrid budget (evict OR compress by value) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO23 Sink-token FP16 reservation within quantized caches `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO24 Outlier channel sparse store (top-1% outlier KV in raw precision) `open`
- IO25 Per-layer eviction budget allocation (attention-sparse vs dispersed layers) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO26 Per-head retention count adaptation (variable critical tokens per head) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO27 Streaming-aware eviction (evict under continuous generation, not just prefill) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO28 Block-boundary eviction coordination (block Xi decisions feed Xi+1) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO29 Eviction-error compounding guard (bounded drift across blocks) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO30 Key-similarity vs query-similarity dual metric `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO31 Eviction score normalization across heads (scale-free comparison) `wired` (wubu_evict2026b, test PASSES)
- IO32 Cache-budget renegotiation on OOM (graceful eviction cascade) `wired` (wubu_evict2026b, test PASSES)
- IO33 Hierarchical eviction: hot RAM / warm DRAM / cold NVMe (ties A06) `wired` (wubu_evict2026b, test PASSES)
- IO34 Eviction feedback to the AGI ledger (per-token retention telemetry) `wired` (wubu_evict2026b, test PASSES)
- IO35 Reconstruction-based importance at the page granularity `open`
- IO36 KV-compression ratio governor (target-ratio eviction scheduler) `wired` (wubu_evict2026b, test PASSES)
- IO37 Attention-sink reserve (never evict the first-k tokens regardless of score) `wired` (wubu_evict2026b, test PASSES)
- IO38 LSH distance threshold tuning by observed attention correlation `open`
- IO39 Proxy-token count adaptation by prompt length `open`
- IO40 Eviction-batch grouping (one-shot discard sets, not per-token) `wired` (wubu_evict2026b, test PASSES)
- IO41 Pooling kernel for SnapKV-style context clustering `wired` (wubu_evict2026b, test PASSES)
- IO42 Retention priority queue (heap-based, O(log n) evict) `wired` (wubu_evict2026b, test PASSES)
- IO43 Eviction-aware RoPE (position re-encode after eviction) `open`
- IO44 Compressed-cache correctness audit (perplexity guard after heavy eviction) `open`
- IO45 Eviction decision caching (reuse scores across decode steps) `wired` (wubu_evict2026b, test PASSES)
- IO46 Attention-score streaming aggregator (running softmax without full matrix) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO47 Block-paged eviction aligned to the paged-KV table (ties HH02) `open`
- IO48 Importance-vs-novelty dual score (H2O x InfiniPot fusion) `wired` (wubu_evict2026b, test PASSES)
- IO49 Eviction under batched requests (shared cache, per-request criticality) `open`
- IO50 Sink-token count adaptation per model (calibration probe) `wired` (wubu_evict2026b, test PASSES)
- IO51 KVQuant-style 3-bit + outlier split encode/decode kernels `open`
- IO52 Eviction telemetry to the operator (retained-vs-evicted quality delta) `wired` (wubu_evict2026b, test PASSES)
- IO53 Budget-constrained eviction via the energy ledger (ties IJ) `wired` (wubu_evict2026b, test PASSES)
- IO54 Eviction threshold hysteresis (avoid evict/keep oscillation) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO55 Cross-session cache reuse (eviction-aware persistence, ties AV03) `wired` (wubu_evict2026b, test PASSES)
- IO56 Semantic eviction via the ANN index (ties AV04) `open`
- IO57 Eviction + speculative-decoding interaction (draft cache retention) `open`
- IO58 Eviction-aware attention scaling (post-eviction normalization) `open`
- IO59 Head-disparity monitor (which heads need the most retention) `wired` (wubu_evict2026, test_evict2026 PASSES)
- IO60 Eviction policy selector (auto-pick policy by head/block profile) `wired` (wubu_evict2026b, test PASSES)
- IO61 Cache compaction (defragment retained KV pages) `wired` (wubu_evict2026b, test PASSES)
- IO62 Eviction under 1M+ context (cost-modeled retention) `open`
- IO63 Per-layer KV budget governor (layer-wise OOM safety) `wired` (wubu_evict2026b, test PASSES)
- IO64 Eviction-score calibration on a probe set (threshold fitting) `wired` (wubu_evict2026b, test PASSES)
- IO65 Reconstruction-aware eviction in hybrid attention (ties JA) `open`
- IO66 Eviction for multimodal tokens (vision token criticality, ties JB) `open`
- IO67 Eviction ledger integration (which tokens were dropped and why) `wired` (wubu_evict2026b, test PASSES)
Status: `open` (67 gaps; each = a real mechanism from the surveyed literature)

## Theme IP: Hopfield / associative memory 2026 frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (continuous-time Hopfield 2502.10122; dynamic-manifold 2506.01303; federated many-to-one 2603.19902; spectral capacity 2026)
- IP01 Continuous-time memory dynamics (memory state as an ODE, not a discrete update) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP02 Dynamic-manifold Hopfield (context-dependent reorganization of the stored manifold) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP03 Federated many-to-one Hopfield (heteroassociative: cue -> associated output) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP04 Spectral-capacity scaling analysis (capacity vs spectral norm of the memory matrix) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP05 Attention-as-Hopfield retrieval formalization (softmax update == memory read) `open`
- IP06 Memory write scheduling (store policy: when a pattern deserves storage) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP07 Memory read with beta annealing (sharp-to-flat retrieval over iterations) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP08 Pattern separation metric (overlap control between stored patterns) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP09 Memory consolidation via rehearsal (periodic re-store of hot patterns) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP10 Associative interference monitor (crosstalk detection between similar patterns) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP11 Cue denoising with precision control (noisy-cue recall strength) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP12 Memory decay scheduler (halflife adaptation by pattern utility) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP13 Context-dependent recall gating (context vector modulates the memory read) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP14 Heteroassociative binding (input -> output associations, not just auto-assoc) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP15 Memory matrix compression (low-rank storage of the pattern matrix) `wired` (wubu_hopfield3, test PASSES)
- IP16 Retrieval by partial cue (prefix / fragment completion) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP17 Hopfield-encoded KV cache (attention KV stored as Hopfield patterns, ties IO) `wired` (wubu_hopfield3, test PASSES)
- IP18 Memory capacity accounting (exponential-capacity bookkeeping) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP19 Episodic memory with time-tags (temporal associative memory) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP20 Memory interference repair (re-orthogonalize similar stored patterns) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP21 Continuous-time numerical integration (memory ODE solver, RK4) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP22 Manifold curvature estimation for context reorganization `open`
- IP23 Federated memory sharing (patterns shared across agents with provenance) `open`
- IP24 Memory retrieval ranking by spectral overlap `wired` (wubu_hopfield3, test PASSES)
- IP25 Forgetting curve integration (Ebbinghaus curve into the memory weight) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP26 Memory replay scheduling (when to replay stored patterns, ties BB) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP27 Memory write dedup (identical/duplicate pattern suppression) `wired` (wubu_hopfield3, test PASSES)
- IP28 Memory read with temperature control (softmax sharpness per query) `wired` (wubu_hopfield3, test PASSES)
- IP29 Associative memory for tool selection (cue -> tool pattern retrieval) `wired` (wubu_hopfield2, test_hopfield2 PASSES)
- IP30 Memory chaining (sequential pattern association, story recall) `wired` (wubu_hopfield3, test PASSES)
- IP31 Hopfield energy monitor (free-energy of the memory state) `wired` (wubu_hopfield3, test PASSES)
- IP32 Memory stabilization (pattern anchoring after consolidation) `open`
- IP33 Cross-modal associative memory (text cue -> vision pattern, ties JB) `wired` (wubu_hopfield3, test PASSES)
- IP34 Memory corruption detection (pattern degradation watchdog) `wired` (wubu_hopfield3, test PASSES)
- IP35 Memory hygiene: prune low-utility stale patterns (ties IL05) `wired` (wubu_hopfield3, test PASSES)
- IP36 Associative recall in the decode path (memory-guided token candidates) `wired` (wubu_hopfield3, test PASSES)
- IP37 Memory-attention fusion (retrieved pattern as attention bias) `wired` (wubu_hopfield3, test PASSES)
- IP38 Multi-scale memories (short/long-term with separate betas) `wired` (wubu_hopfield3, test PASSES)
- IP39 Memory state snapshot/restore (checkpoint the pattern matrix) `wired` (wubu_hopfield3, test PASSES)
- IP40 Hopfield capacity telemetry (used vs theoretical capacity) `wired` (wubu_hopfield3, test PASSES)
- IP41 Cue embedding quality monitor (cue dims that hurt recall) `open`
- IP42 Memory write batching (bulk store of a session's patterns) `open`
- IP43 Memory read batching (bulk recall for batched decode) `open`
- IP44 Pattern condensation (merge near-identical patterns) `wired` (wubu_hopfield3, test PASSES)
- IP45 Memory-based reasoning (recall chains as CoT memory, ties IV) `wired` (wubu_hopfield3, test PASSES)
- IP46 Associative outlier tolerance (robust recall under adversarial cues) `open`
- IP47 Memory matrix spectral cleanup (drop low-singular-value directions) `wired` (wubu_hopfield3, test PASSES)
- IP48 Context-switch memory isolation (per-task memory partitions) `wired` (wubu_hopfield3, test PASSES)
- IP49 Memory search over patterns (ANN over the memory, ties AV) `open`
- IP50 Memory write/read asymmetry modeling (write cost vs read benefit) `open`
- IP51 Hopfield beta autotuning (temperature fit by recall error) `wired` (wubu_hopfield3, test PASSES)
- IP52 Memory decay vs consolidation arbitration (decay rate vs rehearsal rate) `open`
- IP53 Retrieval-augmented memory (external corpus -> memory patterns) `open`
- IP54 Memory provenance (which source stored each pattern) `open`
- IP55 Memory privacy (forget-set patterns removable, ties IM) `open`
- IP56 Memory load balancing across tiers (hot patterns to fast tier) `open`
- IP57 Associative memory for world-model updates (ties IN) `open`
- IP58 Memory capacity warning (approaching exponential limit) `open`
- IP59 Pattern importance weighting in storage (weighted patterns) `open`
- IP60 Memory coherence across sessions (shared memory merge) `open`
- IP61 Hopfield update with momentum (faster retrieval convergence) `open`
- IP62 Memory read failure handling (no-match fallback policy) `open`
- IP63 Associative memory ablation analysis (which patterns matter) `open`
- IP64 Memory pattern normalization (unit-norm storage for stable recall) `open`
- IP65 Memory-write energy accounting (ties IJ energy ledger) `open`
- IP66 Memory consolidation scheduling (offline consolidation pass) `open`
- IP67 Memory monitor: recall accuracy drift (memory health metric) `open`
Status: `open` (67 gaps; continuous-time / manifold / federated Hopfield + attention-as-memory formalization)

## Theme IQ: Preference optimization frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (2602.00954 small-margin; 2605.20834 DPO/RLHF equivalence; 2509.24159 RE-PO; SimPO; CPO; AlphaPO)
- IQ01 SimPO reference-free reward (length-normalized average log-prob) `wired` (wubu_pref, test PASSES)
- IQ02 CPO conditional preference optimization (difficult discriminative prompts) `wired` (wubu_pref2, test PASSES)
- IQ03 IPO identity-preference optimization (squared-error preference loss) `wired` (wubu_pref, test PASSES)
- IQ04 RE-PO robust enhanced policy optimization (general enhancer over DPO/IPO/SimPO/CPO) `wired` (wubu_pref, test PASSES)
- IQ05 AlphaPO reward-shape-aware alignment (reward shaping for DAAs) `wired` (wubu_pref, test PASSES)
- IQ06 Small-margin preference training (margin-aware sampling) `wired` (wubu_pref, test PASSES)
- IQ07 DPO/RLHF conditional-equivalence checker (when DPO == RLHF provably) `wired` (wubu_pref, test PASSES)
- IQ08 Length-bias correction (length-normalized rewards) `wired` (wubu_pref, test PASSES)
- IQ09 Reference-model-free margin (SimPO-style implicit reference) `wired` (wubu_pref, test PASSES)
- IQ10 Preference pair quality weighting (pair difficulty weighting) `wired` (wubu_pref, test PASSES)
- IQ11 Reward accuracy monitor (preference-vs-generation alignment metric) `wired` (wubu_pref, test PASSES)
- IQ12 Preference dataset dedup (near-duplicate pair suppression) `wired` (wubu_pref, test PASSES)
- IQ13 Offline vs online preference mixing (static pairs + live feedback) `wired` (wubu_pref, test PASSES)
- IQ14 Preference aggregation (multiple annotators -> consensus pair) `wired` (wubu_pref, test PASSES)
- IQ15 Margin schedule (margin annealed across training) `wired` (wubu_pref, test PASSES)
- IQ16 Preference noise robustness (label-noise-tolerant loss) `wired` (wubu_pref, test PASSES)
- IQ17 Token-level preference (per-token rewards, not sequence-level) `wired` (wubu_pref, test PASSES)
- IQ18 Step-level process preferences (ties the PRM literature) `wired` (wubu_pref, test PASSES)
- IQ19 Preference cache (reuse pair gradients across updates) `wired` (wubu_pref, test PASSES)
- IQ20 Preference-based early stopping (reward-accuracy gate) `wired` (wubu_pref, test PASSES)
- IQ21 Multi-objective preference (win/lose/ties with three-way loss) `wired` (wubu_pref2, test PASSES)
- IQ22 Preference staleness (pair age weighting) `wired` (wubu_pref, test PASSES)
- IQ23 Reward-free calibration (reference-free reward alignment check) `wired` (wubu_pref2, test PASSES)
- IQ24 Preference conflict detection (contradictory pairs) `wired` (wubu_pref2, test PASSES)
- IQ25 Robust preference optimization (RE-PO-style robustness envelope) `wired` (wubu_pref2, test PASSES)
- IQ26 Preference budget allocation (which prompts deserve pairs) `wired` (wubu_pref2, test PASSES)
- IQ27 Alignment without forgetting (preference + KL-anchor, ties IM04) `wired` (wubu_pref2, test PASSES)
- IQ28 Preference feedback loop to the AGI (user signals as pairs, ties IV) `wired` (wubu_pref2, test PASSES)
- IQ29 Implicit reward visualization (reward traces per token) `wired` (wubu_pref2, test PASSES)
- IQ30 Preference benchmark harness (alignment eval suite) `wired` (wubu_pref2, test PASSES)
- IQ31 Length-normalized margin (SimPO gamma) `wired` (wubu_pref2, test PASSES)
- IQ32 Reference-model distillation into the reward (offline reward model) `wired` (wubu_pref2, test PASSES)
- IQ33 Preference pair augmentation (synthetic pairs from rejected samples) `wired` (wubu_pref2, test PASSES)
- IQ34 Alignment drift monitor during fine-tune (ties IM06) `wired` (wubu_pref2, test PASSES)
- IQ35 Preference transfer across domains (pair curriculum) `wired` (wubu_pref2, test PASSES)
- IQ36 Reward shaping functions (AlphaPO-style shaping) `wired` (wubu_pref2, test PASSES)
- IQ37 Preference update frequency (mini-batch preference mixing) `wired` (wubu_pref2, test PASSES)
- IQ38 Pair difficulty-aware sampling (hard-pair emphasis) `wired` (wubu_pref2, test PASSES)
- IQ39 Preference-regularized decode (no retrain: preference-constrained sampling) `wired` (wubu_pref2, test PASSES)
- IQ40 Alignment energy accounting (preference training under the energy ledger) `wired` (wubu_pref2, test PASSES)
- IQ41 Preference-pair provenance (which source made the pair) `wired` (wubu_pref2, test PASSES)
- IQ42 Multi-turn preference (conversation-level pairs) `wired` (wubu_pref2, test PASSES)
- IQ43 Preference staleness decay (old pairs weight down) `wired` (wubu_pref2, test PASSES)
- IQ44 Preference quality gate (reject low-agreement pairs) `wired` (wubu_pref2, test PASSES)
- IQ45 DPO vs RLHF divergence metric (when to switch methods) `wired` (wubu_pref2, test PASSES)
- IQ46 Preference ensemble (multiple reward hypotheses, ties DD) `wired` (wubu_pref2, test PASSES)
- IQ47 Alignment health dashboard (reward accuracy + drift + margin) `wired` (wubu_pref2, test PASSES)
- IQ48 Preference-selective replay (alignment replay, ties IM05) `wired` (wubu_pref2, test PASSES)
- IQ49 Online preference bootstrap (self-generated pairs, ties IV) `wired` (wubu_pref2, test PASSES)
- IQ50 Preference curriculum (easy->hard pair schedule) `wired` (wubu_pref2, test PASSES)
- IQ51 Length-robust reward normalization (SimPO's answer) `wired` (wubu_pref2, test PASSES)
- IQ52 Preference-aware sampling temperature (confidence-scaled pairs) `wired` (wubu_pref2, test PASSES)
- IQ53 Pair margin prediction (predict pair difficulty) `wired` (wubu_pref2, test PASSES)
- IQ54 Preference logbook (auditable alignment history) `wired` (wubu_pref2, test PASSES)
- IQ55 Alignment verification gate (post-align eval before promotion, ties AX) `wired` (wubu_pref2, test PASSES)
- IQ56 Preference transfer learning (align small model, transfer to big) `wired` (wubu_pref2, test PASSES)
- IQ57 Reward hacking pre-detection (alignment-time monitoring) `wired` (wubu_pref2, test PASSES)
- IQ58 Preference-efficient alignment (fewer pairs via active selection) `wired` (wubu_pref2, test PASSES)
- IQ59 Preference entropy (pair distribution flatness) `wired` (wubu_pref2, test PASSES)
- IQ60 Alignment + unlearning joint objective (align AND forget, ties IM) `wired` (wubu_pref2, test PASSES)
- IQ61 Preference-based model selection (align then pick by eval) `wired` (wubu_pref2, test PASSES)
- IQ62 Preference watermark (align-time provenance for outputs) `wired` (wubu_pref2, test PASSES)
- IQ63 Preference data versioning (dataset version in the training ledger) `wired` (wubu_pref2, test PASSES)
- IQ64 Margin regularization (avoid over-confident preference fitting) `wired` (wubu_pref2, test PASSES)
- IQ65 Preference meta-learning (learn the alignment objective, ties IV) `wired` (wubu_pref2, test PASSES)
- IQ66 Alignment test-time scaling (preference-guided decoding budget, ties IK) `wired` (wubu_pref2, test PASSES)
- IQ67 Preference-to-policy operator (alignment config promotion, ties IM07) `wired` (wubu_pref2, test PASSES)
Status: `open` (67 gaps; SimPO/CPO/IPO/RE-PO/AlphaPO + DPO-RLHF equivalence + alignment monitoring)

## Theme IR: Multi-tenant serving / scheduler
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (2603.00356 token management; FIFO-fairness 2026; Stream2LLM MLsys-oral; scheduling survey)
- IR01 Token-management admission control (request acceptance by token budget) `wired` (wubu_serve, test PASSES)
- IR02 Fair-share scheduler (weighted fair queuing over KV budget) `wired` (wubu_serve, test PASSES)
- IR03 Preemption with cache-rebuild cost model (preempt vs restart decision) `wired` (wubu_serve, test PASSES)
- IR04 Activation-budget preemption guard (bounded memory below the threshold) `wired` (wubu_serve, test PASSES)
- IR05 Stream2LLM context streaming + prefill overlap (TTFT reduction) `wired` (wubu_serve, test PASSES)
- IR06 Longest-common-prefix scheduling (minimize redundant prefill) `wired` (wubu_serve, test PASSES)
- IR07 Decoupled scheduling (schedule decision separate from resource acquisition) `wired` (wubu_serve, test PASSES)
- IR08 Hardware-specific cost model for preemption (per-device costs) `wired` (wubu_serve, test PASSES)
- IR09 Burst handling (elastic admission under demand spikes) `wired` (wubu_serve, test PASSES)
- IR10 Priority tiers with starvation bounds `wired` (wubu_serve, test PASSES)
- IR11 Multi-tenant KV isolation (per-tenant cache partitions) `wired` (wubu_serve, test PASSES)
- IR12 Token-budget fairness (each tenant's token share) `wired` (wubu_serve, test PASSES)
- IR13 Preemption victim selection (cheapest-to-restart request) `wired` (wubu_serve, test PASSES)
- IR14 Checkpointed preemption (KV snapshot on preempt, resume not restart) `wired` (wubu_serve, test PASSES)
- IR15 SLO-aware scheduling (per-request latency targets) `wired` (wubu_serve, test PASSES)
- IR16 Batch compaction (fill decode gaps with prefill chunks) `wired` (wubu_serve, test PASSES)
- IR17 Scheduler-cache coherence (schedule decisions respect cache reuse) `wired` (wubu_serve, test PASSES)
- IR18 Dynamic batching window (batch size adaptation by memory) `wired` (wubu_serve, test PASSES)
- IR19 Request-level priority inheritance (ties the OS PI concept) `wired` (wubu_serve, test PASSES)
- IR20 Memory-stability hysteresis (avoid preempt/accept oscillation) `wired` (wubu_serve, test PASSES)
- IR21 Fairness metric monitor (per-tenant service share) `open`
- IR22 Preemption telemetry (preempt frequency, rebuild cost) `open`
- IR23 Co-scheduling prefill+decode (interleaved phases, ties HH04) `open`
- IR24 Cache-aware request routing (route to the node with the prefix) `open`
- IR25 Token-budget profiler (per-request token demand estimation) `open`
- IR26 Admission by predicted KV growth (proactive OOM avoidance) `open`
- IR27 Work-conserving scheduler (never idle while work exists) `open`
- IR28 Preemption budget per tenant (fair preemption) `open`
- IR29 Decode-phase priority (decode > prefill under contention) `open`
- IR30 Scheduler-ledger integration (schedule decisions to the AGI ledger) `open`
- IR31 Multi-queue scheduling (separate queues per SLO class) `open`
- IR32 Backfill scheduling (fill idle slots with background work) `open`
- IR33 Speculative prefill (predict next prompt, prefill ahead) `open`
- IR34 Context-keepalive scheduler (keep hot contexts resident) `open`
- IR35 Eviction-vs-preempt arbitration (evict cold cache or preempt request) `open`
- IR36 Cost-aware scheduling (J/token cost, ties IJ) `open`
- IR37 Scheduler fairness under variable demand (burst-adaptive weights) `open`
- IR38 Request grouping by prefix similarity (batched prefill) `open`
- IR39 Preemption recovery speedup (KV checkpoint restore) `open`
- IR40 Scheduler resilience (scheduler restart without request loss) `open`
- IR41 Multi-tenant security isolation (tenant cache boundaries, ties AD) `open`
- IR42 Token-budget debt tracking (tenant overspend recovery) `open`
- IR43 SLO violation monitor (latency-target breach alerts) `open`
- IR44 Adaptive concurrency (max in-flight by memory pressure) `open`
- IR45 Scheduling policy selector (auto-pick scheduler by load profile) `open`
- IR46 Idle-capacity scavenging (low-priority batch on idle resources) `open`
- IR47 Request coalescing (merge similar prompts) `open`
- IR48 Preemption decision cost-benefit (restart cost vs preempt cost) `open`
- IR49 Memory-pressure feedback loop (scheduler <-> allocator) `open`
- IR50 Deadline-aware scheduling (hard deadlines for time-critical requests) `open`
- IR51 Fair preemption ordering (preempt the least-SLO-critical first) `open`
- IR52 Scheduler benchmarking harness (fairness/latency/throughput evals) `open`
- IR53 Cache-sharing scheduler (shared prefix across tenants with accounting) `open`
- IR54 Preemption-aware token generation (checkpoint generation state) `open`
- IR55 Multi-model scheduling (multiple models on one pool) `open`
- IR56 Scheduler hysteresis (stability under load oscillation) `open`
- IR57 Queue-depth telemetry (per-queue waiting metrics) `open`
- IR58 Request aging (avoid indefinite starvation) `open`
- IR59 Cost-fairness tradeoff scheduler (J/token per tenant, ties IJ) `open`
- IR60 Preemption simulation (dry-run preemption policy) `open`
- IR61 Scheduler config operator (auto-tune scheduler params, ties IV) `open`
- IR62 Token-budget negotiation (tenant request for more budget) `open`
- IR63 Memory-debt reclamation (slow-tenant cache reclaim) `open`
- IR64 Prefill batch planning (chunked prefill schedule) `open`
- IR65 Scheduler event log (auditable schedule decisions) `open`
- IR66 Cross-node scheduling (distributed request placement) `open`
- IR67 Serving energy envelope (power-cap-aware scheduling, ties IJ03) `open`
Status: `open` (67 gaps; fair multi-tenant scheduling + preemption + prefix-aware routing)

## Theme IS: PIM / near-memory / hardware co-design
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (P3-LLM NPU-PIM 2511.06838; near-memory 3D-DRAM DAC2025; CIM crossbar/RRAM/SRAM 2026; AQPIM HPCA 2026)
- IS01 PIM offload model (which ops move near memory: GEMV over KV) `wired` (wubu_pim, test PASSES)
- IS02 Near-memory KV tier (KV resident next to the compute) `wired` (wubu_pim, test PASSES)
- IS03 Crossbar-compatible matmul emulation (CIM-style GEMV model) `wired` (wubu_pim, test PASSES)
- IS04 SRAM-CIM quantization constraints (bit-cell precision limits) `wired` (wubu_pim, test PASSES)
- IS05 RRAM/FeFET/SOT-MRAM tier model (emerging memory energy/latency) `wired` (wubu_pim, test PASSES)
- IS06 Near-storage compute (smart-SSD KV filter) `wired` (wubu_pim, test PASSES)
- IS07 PIM capacity wall guard (PIM memory budget vs model size) `wired` (wubu_pim, test PASSES)
- IS08 3D-DRAM bonding model (logic-on-memory integration cost) `wired` (wubu_pim, test PASSES)
- IS09 Hybrid NPU-PIM dispatch (when to use PIM vs NPU) `wired` (wubu_pim, test PASSES)
- IS10 Data-movement accounting (bytes moved per op, ties roofline) `wired` (wubu_pim, test PASSES)
- IS11 HBM-stack near-memory buffers (in-stack staging) `wired` (wubu_pim, test PASSES)
- IS12 PIM-friendly weight layout (channel-last for in-memory MAC) `wired` (wubu_pim, test PASSES)
- IS13 Analog-compute noise model (crossbar ADC/DAC precision) `wired` (wubu_pim, test PASSES)
- IS14 Hardware cost model integration (energy+latency per op, ties IJ) `wired` (wubu_pim, test PASSES)
- IS15 PIM offload scheduler (batch ops for near-memory execution) `wired` (wubu_pim, test PASSES)
- IS16 Memory-centric attention tiling (attention tiles resident in memory) `wired` (wubu_pim, test PASSES)
- IS17 Device-model portability (same engine, hardware-abstracted) `wired` (wubu_pim, test PASSES)
- IS18 Near-memory reduce (partial sums at the memory) `wired` (wubu_pim, test PASSES)
- IS19 PIM page-locality (KV pages colocated with the compute) `wired` (wubu_pim, test PASSES)
- IS20 Hardware telemetry model (simulated counters: MACs, bytes, J) `wired` (wubu_pim, test PASSES)
- IS21 CIM bit-precision adaptation (precision per layer by sensitivity) `open`
- IS22 Emerging-memory endurance model (write-wear budget for KV) `open`
- IS23 Near-memory speculative decode (draft heads at the memory) `open`
- IS24 PIM capacity-vs-latency frontier (tradeoff model) `open`
- IS25 Heterogeneous CPU/GPU/NPU-PIM scheduling `open`
- IS26 Memory-wall budget governor (data-movement cap per token) `open`
- IS27 PIM-friendly KV quant (integer KV for CIM, ties IO10) `open`
- IS28 Near-storage RAG (retrieval at the SSD) `open`
- IS29 Hardware abstraction layer for the engine (kernel dispatch table) `open`
- IS30 Crossbar mapping optimizer (weight-to-crossbar placement) `open`
- IS31 PIM energy ledger (in-memory J/op accounting, ties IJ) `open`
- IS32 Near-memory attention sink (sink KV pinned near compute) `open`
- IS33 PIM correctness audit (analog error bounds) `open`
- IS34 Hardware-targeted kernel variants (per-device GEMV) `open`
- IS35 Memory-centric decode loop (decode organized around the memory) `open`
- IS36 PIM-offload benefit predictor (when PIM beats CPU) `open`
- IS37 Near-memory MoE routing (expert weights at the memory) `open`
- IS38 CIM weight stationary layout (weights fixed in crossbar) `open`
- IS39 Hardware counter model (cycle/J/byte counters for tuning) `open`
- IS40 PIM page eviction (KV page movement between tiers) `open`
- IS41 Near-memory prefix cache (LCP prefix at the memory) `open`
- IS42 Emerging-memory latency model (PCM/FeFET read/write costs) `open`
- IS43 PIM-aware batching (batch shapes that fit the memory arrays) `open`
- IS44 Hardware co-simulation harness (simulated device models) `open`
- IS45 PIM numerical stability (low-precision accumulation guards) `open`
- IS46 Near-memory KV compression (compress at the memory, ties IO) `open`
- IS47 Hardware-aware auto-tuning (kernel selection by counters) `open`
- IS48 Memory-centric speculative decode (draft KV near memory) `open`
- IS49 PIM capacity planning (model+KV fit check per device) `open`
- IS50 Near-storage dedup (SSD-side KV dedup) `open`
- IS51 Hardware event simulation (simulated PMU events) `open`
- IS52 PIM dataflow optimization (input-stationary vs output-stationary) `open`
- IS53 Memory-wall roofline update (energy roofline, ties IJ01) `open`
- IS54 Near-memory attention offload (attention compute at DRAM) `open`
- IS55 CIM weight refresh policy (drift compensation) `open`
- IS56 Hardware-aware quantization selector (per-device bit choice) `open`
- IS57 PIM offload regression tests (host parity checks) `open`
- IS58 Near-memory top-k (softmax/selection near memory) `open`
- IS59 Memory-centric scheduling (schedule by memory, ties IR) `open`
- IS60 Hardware diversity matrix (which kernels run where) `open`
- IS61 PIM energy envelope (in-memory power cap, ties IJ03) `open`
- IS62 Near-memory KV dedup (dedup at the memory tier) `open`
- IS63 Hardware cost ledger (J + latency per request) `open`
- IS64 PIM-friendly tokenizer (byte alignment for memory ops) `open`
- IS65 Memory-centric planning (plan steps co-resident with memory) `open`
- IS66 Near-memory verifier (verify tokens near the compute) `open`
- IS67 Hardware-abstracted engine config (device descriptors) `open`
Status: `open` (67 gaps; PIM/CIM/near-memory co-design, hardware-abstracted engine)

## Theme IT: Tokenization / data plane
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (subword decoupling 2604.27263; bit-level BPE 2506.07541; tokenizer-free 2406.19223; lexical density 2026)
- IT01 Bit-level BPE (compression below the byte boundary) `wired` (wubu_token, test PASSES)
- IT02 Tokenizer-free UTF-8 embeddings (no vocab, ~85% embedding savings) `wired` (wubu_token, test PASSES)
- IT03 Subword-benefit decoupling (isolate tokenization effects) `wired` (wubu_token, test PASSES)
- IT04 Byte-entropy-aware merges (low-entropy byte distribution handling) `wired` (wubu_token, test PASSES)
- IT05 Lexical-density detector (context density -> effective window) `wired` (wubu_token, test PASSES)
- IT06 Token-merge cache (frequent-token path memoization) `wired` (wubu_token, test PASSES)
- IT07 Vocabulary pruning (drop unused tokens, remap ids) `wired` (wubu_token, test PASSES)
- IT08 Tokenizer roundtrip audit (encode/decode fidelity checks) `wired` (wubu_token, test PASSES)
- IT09 Multi-script tokenization (mixed-script merge policy) `wired` (wubu_token, test PASSES)
- IT10 Token-level compression (post-token entropy coding) `wired` (wubu_token, test PASSES)
- IT11 Adaptive tokenization (per-domain vocab) `wired` (wubu_token, test PASSES)
- IT12 Token-efficiency metric (tokens per information unit) `wired` (wubu_token, test PASSES)
- IT13 Embedding-table compression (shared embeddings, ties quant) `wired` (wubu_token, test PASSES)
- IT14 Token-frequency telemetry (vocab usage distribution) `wired` (wubu_token, test PASSES)
- IT15 Tokenizer-spec versioning (tokenizer changes tracked) `wired` (wubu_token, test PASSES)
- IT16 OOV handling policy (unknown-token fallbacks) `wired` (wubu_token, test PASSES)
- IT17 Subword-to-byte fallback (lossless decode guarantees) `wired` (wubu_token, test PASSES)
- IT18 Token-boundary attention bias (boundary-aware scoring) `wired` (wubu_token, test PASSES)
- IT19 Token-packing (dense sequence packing for prefill) `wired` (wubu_token, test PASSES)
- IT20 Byte-level LM adapter (byte model fallback path) `wired` (wubu_token, test PASSES)
- IT21 Tokenizer benchmark (multilingual token efficiency evals) `open`
- IT22 Token-id remapping (vocab swap without retrain) `open`
- IT23 Token entropy monitor (distribution shift detection) `open`
- IT24 Subword merging heuristics (BPE merge-pair scoring) `open`
- IT25 Tokenizer-cache (memoized encode for repeated text) `open`
- IT26 Unicode-normalization guard (NFKC/NFD handling) `open`
- IT27 Token-length regularization (bounded token growth) `open`
- IT28 Byte-fallback decode (malformed-input recovery) `open`
- IT29 Tokenizer data-flow (token pipeline statistics) `open`
- IT30 Vocabulary merge rules (custom merges for domain terms) `open`
- IT31 Token-pair frequency table (BPE stats) `open`
- IT32 Embedded-token density (lexical density per window, ties IO) `open`
- IT33 Tokenizer determinism (same input -> same ids) `open`
- IT34 Token-budget planner (token estimate before generation, ties IK) `open`
- IT35 Subword-entity alignment (entities spanning tokens) `open`
- IT36 Tokenizer streaming (incremental encode) `open`
- IT37 Vocabulary growth policy (online vocab expansion) `open`
- IT38 Token-space augmentation (token dropout for robustness) `open`
- IT39 Tokenizer energy accounting (encode cost, ties IJ) `open`
- IT40 Byte-level RoPE (position encoding at the byte level) `open`
- IT41 Token-id compression (id entropy coding) `open`
- IT42 Multi-token prediction targets (predict next-N tokens) `open`
- IT43 Token-trie prefix index (fast token prefix lookup) `open`
- IT44 Tokenizer serialization (portable tokenizer format) `open`
- IT45 Tokenization diff tools (compare tokenizer versions) `open`
- IT46 Byte-pair frequency monitor (merge health) `open`
- IT47 Token-efficiency-aware prefill (skip redundant tokens) `open`
- IT48 Tokenizer-free fallback (engine runs without a vocab) `open`
- IT49 Vocabulary coverage metric (OOV rate per domain) `open`
- IT50 Token-boundary watermark (detect token-level tampering) `open`
- IT51 Token sequence compression (lossless token-stream coding) `open`
- IT52 Adaptive byte-vs-subword (per-input path choice) `open`
- IT53 Tokenizer config tuning (merge-threshold autotune) `open`
- IT54 Token metadata (per-token provenance/features) `open`
- IT55 Token embedding quant (embedding-table int8, ties quant) `open`
- IT56 Tokenizer concurrency (thread-safe encode) `open`
- IT57 Token-pair constraints (disallowed merges) `open`
- IT58 Tokenizer fuzz (adversarial byte input, ties IX) `open`
- IT59 Token-efficiency operator (token-budget config pick) `open`
- IT60 Byte-shard alignment (byte-aligned KV pages) `open`
- IT61 Tokenizer profiling (encode/decode timing) `open`
- IT62 Token-stream dedup (repeated-token suppression) `open`
- IT63 Vocabulary pruning safety (never-prune hot tokens) `open`
- IT64 Tokenizer-regression test suite `open`
- IT65 Token-id stability across versions (stable ids) `open`
- IT66 Byte-entropy adaptive merge (entropy-gated merges) `open`
- IT67 Token-efficiency vs quality frontier (compression tradeoff) `open`
Status: `open` (67 gaps; bit-level/byte-level/tokenizer-free tokenization + lexical density)

## Theme IU: Linear attention / fast kernels
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (Mamba3 2603.15569; Kimi Linear/KDA 2510.26692; FLA 2503.14376; Gated DeltaNet; PaTH attention; Hymba hybrid-head)
- IU01 Chunkwise-parallel linear attention (FLA-style chunked formulation) `wired` (wubu_linattn, test PASSES)
- IU02 Mamba3 selective state update (recurrent state, constant memory) `wired` (wubu_linattn, test PASSES)
- IU03 Gated DeltaNet update (gated delta rule per step) `wired` (wubu_linattn, test PASSES)
- IU04 Gated Slot Attention (GSA) state slots `wired` (wubu_linattn, test PASSES)
- IU05 HGRN2 gated linear RNN with state expansion `wired` (wubu_linattn, test PASSES)
- IU06 GLA hardware-efficient gated linear attention `wired` (wubu_linattn, test PASSES)
- IU07 mLSTM sigmoid-gated reduced-compute variant (mLSTMsig) `wired` (wubu_linattn, test PASSES)
- IU08 Tiled flash linear attention (TFLA kernel tiling) `wired` (wubu_linattn, test PASSES)
- IU09 Lightning attention (Ling-style recurrent linear variant) `wired` (wubu_linattn, test PASSES)
- IU10 PaTH position encoding (Householder accumulation) `wired` (wubu_linattn, test PASSES)
- IU11 Hybrid-head attention (Hymba-style attention+SSM heads per layer) `wired` (wubu_linattn, test PASSES)
- IU12 Hybrid layer mixing (attention/SSM alternation, Falcon-H1 style) `wired` (wubu_linattn, test PASSES)
- IU13 SSM KV-cache elimination path (recurrent state instead of KV) `wired` (wubu_linattn, test PASSES)
- IU14 SSM long-context scaling (beyond quadratic-attention limits) `wired` (wubu_linattn, test PASSES)
- IU15 Hybrid TTFT comparison (SSM 1.35s vs Transformer 8.24s at 57K) `wired` (wubu_linattn, test PASSES)
- IU16 Linear-attention numerical stability (recurrent accumulation guards) `wired` (wubu_linattn, test PASSES)
- IU17 State compression (learned state summarization) `wired` (wubu_linattn, test PASSES)
- IU18 Linear-attention + RoPE interaction (position in linear recurrences) `wired` (wubu_linattn, test PASSES)
- IU19 Chunk state transfer (carry chunk states across batches) `wired` (wubu_linattn, test PASSES)
- IU20 Gated state decay (forget gates in the state) `wired` (wubu_linattn, test PASSES)
- IU21 Delta-rule memory write (delta updates to the state) `open`
- IU22 Linear-attention kernel variant selection (FLA-style autotune) `open`
- IU23 SSM precision control (state precision vs drift) `open`
- IU24 Hybrid energy model (SSM 75% energy cut at 57K, ties IJ) `open`
- IU25 Attention/SSM layer scheduler (which layers are which) `open`
- IU26 Recurrent state checkpoint (state snapshot/restore) `open`
- IU27 Linear-attention recall limits (ICL/precise-recall gap analysis) `open`
- IU28 Hybrid recall compensation (attention layers for precise recall) `open`
- IU29 State-space initialization (SSM parameter init) `open`
- IU30 Linear-attention streaming (constant-memory infinite streaming) `open`
- IU31 Chunked state compute (parallel chunk prefill) `open`
- IU32 Gated linear attention forget schedule (learned gates) `open`
- IU33 Delta-rule binding (write specific keys to state slots) `open`
- IU34 Linear-attention weight tying (recurrent weight sharing) `open`
- IU35 SSM normalization (state normalization for stability) `open`
- IU36 Hybrid decode overlap (attention+SSM heads in one pass) `open`
- IU37 Linear-attention energy ledger (per-state-update J) `open`
- IU38 Recurrent memory decay (state forgetting, ties IP) `open`
- IU39 Linear-attention quantization (quantized state, ties quant) `open`
- IU40 SSM long-context memory bound (constant memory proof) `open`
- IU41 Hybrid benchmark harness (attention vs SSM vs hybrid evals) `open`
- IU42 State expansion ratio tuning (HGRN2-style) `open`
- IU43 Linear-attention speculative decode (recurrent drafter) `open`
- IU44 Chunk parallelization (sequence-chunk parallelism) `open`
- IU45 Gated state multiplexing (shared state across heads) `open`
- IU46 Linear-attention stability monitor (state norm watchdog) `open`
- IU47 Hybrid position encoding (per-head position schemes) `open`
- IU48 SSM hardware mapping (recurrent scan on CPU) `open`
- IU49 Linear-attention gradient path (backward recurrence) `open`
- IU50 Recurrent attention span (effective receptive field) `open`
- IU51 Hybrid layer count tuning (attention/SSM ratio) `open`
- IU52 Linear-attention memory bound (O(1) state size) `open`
- IU53 Delta-rule capacity (state slot capacity, ties IP) `open`
- IU54 SSM multi-scale states (parallel state scales) `open`
- IU55 Linear-attention token-efficiency (ties IT) `open`
- IU56 Hybrid decode scheduling (which phase uses which mechanism) `open`
- IU57 Gated linear attention init (gating init for stability) `open`
- IU58 Linear-attention long-context eval (needle tests) `open`
- IU59 SSM state pruning (drop low-importance state dims) `open`
- IU60 Hybrid attention cost model (attention vs SSM per layer) `open`
- IU61 Linear-attention + Hopfield memory (state as associative memory, ties IP) `open`
- IU62 Chunked linear-attention prefill (parallel chunk prefill) `open`
- IU63 SSM robustness (perturbation sensitivity) `open`
- IU64 Hybrid energy frontier (Pareto energy/accuracy) `open`
- IU65 Linear-attention operator (mechanism selection by context length) `open`
- IU66 Recurrent state ledger (state telemetry) `open`
- IU67 Hybrid model fusion (merge attention + SSM outputs) `open`
Status: `open` (67 gaps; linear attention + hybrid SSM kernels + Hymba-style hybrid heads)

## Theme IV: Recursive self-improvement frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (RSI survey 2607.13104; Goedel agent; LADDER 2503.00735; Promptbreeder; HyperAgents 2603.19461; AUTOHARNESS; ICLR-2026 RSI workshop)
- IV01 Bounded verifiable RSI loops (self-improve with a verifier gate) `wired` (wubu_rsi, test PASSES)
- IV02 Goedel-style self-referential agent (improve the improver) `wired` (wubu_rsi, test PASSES)
- IV03 LADDER recursive problem decomposition (decompose-and-improve) `wired` (wubu_rsi, test PASSES)
- IV04 Promptbreeder prompt evolution (self-referential prompt mutation) `wired` (wubu_rsi, test PASSES)
- IV05 HyperAgents metacognitive transfer (improve strategies across domains) `wired` (wubu_rsi, test PASSES)
- IV06 AUTOHARNESS code-harness synthesis (auto-generate test harnesses) `wired` (wubu_rsi, test PASSES)
- IV07 Intrinsic self-reflection for preference policy (self-reflection in RL) `wired` (wubu_rsi, test PASSES)
- IV08 Soft-mellowmax Monte-Carlo planning (softmax-planned search) `wired` (wubu_rsi, test PASSES)
- IV09 Experience-learning loop (streaming telemetry -> improvement) `wired` (wubu_rsi, test PASSES)
- IV10 Synthetic-data pipeline for self-improvement (self-generated training data) `wired` (wubu_rsi, test PASSES)
- IV11 Weak-to-strong generalization loop (small teacher -> big student) `wired` (wubu_rsi, test PASSES)
- IV12 Scaffolding improvement (improve the agent framework itself) `wired` (wubu_rsi, test PASSES)
- IV13 Full scaffolding search (search the agent design space) `wired` (wubu_rsi, test PASSES)
- IV14 Self-awareness audit (the agent knows its own capability) `wired` (wubu_rsi, test PASSES)
- IV15 Bounded self-modification (safe-pace weight updates) `wired` (wubu_rsi, test PASSES)
- IV16 Continual fine-tuning scheduler (when to schedule fine-tunes) `wired` (wubu_rsi, test PASSES)
- IV17 Self-play for improvement (play against yourself, ties GG) `wired` (wubu_rsi, test PASSES)
- IV18 Bug-introduction self-training (inject bugs, learn to fix) `wired` (wubu_rsi, test PASSES)
- IV19 Production-signal improvement (real usage rewards -> improvement) `wired` (wubu_rsi, test PASSES)
- IV20 Reflection-memory (Reflexion-style episodic reflection log) `wired` (wubu_rsi, test PASSES)
- IV21 Reflection-diversity guard (avoid local-optima reflections) `wired` (wubu_rsi, test PASSES)
- IV22 Self-improvement ledger (auditable improvement history) `wired` (wubu_rsi, test PASSES)
- IV23 Improvement-delta metric (did the change help, ties AH13) `wired` (wubu_rsi, test PASSES)
- IV24 Recursive decomposition tree (problem -> subproblem tree) `wired` (wubu_rsi, test PASSES)
- IV25 Self-evolution verify gate (promote only verified improvements) `wired` (wubu_rsi, test PASSES)
- IV26 Metacognitive loop monitor (the improver's own health) `wired` (wubu_rsi, test PASSES)
- IV27 Prompt-archive evolution (prompt population + selection) `wired` (wubu_rsi, test PASSES)
- IV28 Cross-domain strategy transfer (strategies generalize) `wired` (wubu_rsi, test PASSES)
- IV29 Self-reflective data curation (curate your own training data) `wired` (wubu_rsi, test PASSES)
- IV30 Improvement rate monitoring (improvement velocity) `wired` (wubu_rsi, test PASSES)
- IV31 Self-harness generation (generate your own eval harness) `wired` (wubu_rsi, test PASSES)
- IV32 Recursive self-benchmark (benchmark the benchmark) `wired` (wubu_rsi, test PASSES)
- IV33 Weak-supervision amplification (weak labels -> strong model) `wired` (wubu_rsi, test PASSES)
- IV34 Self-improvement safety envelope (bounded improvement rate) `wired` (wubu_rsi, test PASSES)
- IV35 Experience distillation (telemetry -> training examples) `wired` (wubu_rsi, test PASSES)
- IV36 Self-modeling (the agent models its own behavior) `wired` (wubu_rsi, test PASSES)
- IV37 Improvement credit assignment (which change caused the gain) `wired` (wubu_rsi, test PASSES)
- IV38 Self-referential prompt search (prompts that improve prompts) `wired` (wubu_rsi, test PASSES)
- IV39 Recursive verification (verify the verifier) `wired` (wubu_rsi, test PASSES)
- IV40 Self-improvement cost ledger (improvement J budget, ties IJ) `wired` (wubu_rsi, test PASSES)
- IV41 Continual architecture search (self-searching architecture) `wired` (wubu_rsi, test PASSES)
- IV42 Self-improvement regression guard (never regress the baseline) `wired` (wubu_rsi, test PASSES)
- IV43 Improvement frontier archive (Pareto improvement archive) `wired` (wubu_rsi, test PASSES)
- IV44 Self-explanation (the agent explains its own changes) `wired` (wubu_rsi, test PASSES)
- IV45 Recursive loop termination (when improvement saturates) `wired` (wubu_rsi, test PASSES)
- IV46 Self-improvement telemetry (loop counters to the ledger) `wired` (wubu_rsi, test PASSES)
- IV47 Metacognitive calibration (confidence in own improvements) `wired` (wubu_rsi, test PASSES)
- IV48 Improvement replay (replay successful improvement steps) `wired` (wubu_rsi, test PASSES)
- IV49 Self-distillation (improve by distilling own outputs) `wired` (wubu_rsi, test PASSES)
- IV50 Recursive skill acquisition (learn how to learn, ties skills) `wired` (wubu_rsi, test PASSES)
- IV51 Self-improvement governance (HITL gates on self-modification) `wired` (wubu_rsi, test PASSES)
- IV52 Improvement provenance (which loop produced the change) `wired` (wubu_rsi, test PASSES)
- IV53 Self-improvement sandbox (improvements in isolation, ties AX) `wired` (wubu_rsi, test PASSES)
- IV54 Recursive prompt optimization (optimize the optimizer's prompts) `wired` (wubu_rsi, test PASSES)
- IV55 Self-improvement energy budget (improve under a J cap) `wired` (wubu_rsi, test PASSES)
- IV56 Loop convergence detection (improvement plateau detection) `wired` (wubu_rsi, test PASSES)
- IV57 Self-improvement portfolio (parallel improvement candidates) `wired` (wubu_rsi, test PASSES)
- IV58 Recursive evaluation (evaluate the evaluator) `wired` (wubu_rsi, test PASSES)
- IV59 Self-improvement audit trail (append-only improvement log) `wired` (wubu_rsi, test PASSES)
- IV60 Improvement rollback (safe revert of a failed improvement) `wired` (wubu_rsi, test PASSES)
- IV61 Self-improvement benchmark suite (RSI evaluation harness) `wired` (wubu_rsi, test PASSES)
- IV62 Metacognitive transfer monitor (does improvement transfer) `wired` (wubu_rsi, test PASSES)
- IV63 Recursive planning (plan the improvement plan) `wired` (wubu_rsi, test PASSES)
- IV64 Self-improvement diversity (avoid converging on one trick) `wired` (wubu_rsi, test PASSES)
- IV65 Improvement-interaction analysis (which improvements combine) `wired` (wubu_rsi, test PASSES)
- IV66 Self-improvement operator (the DA-3 loop as an operator, ties skill) `wired` (wubu_rsi, test PASSES)
- IV67 Recursive self-improvement safety audit (the loop's own alignment) `wired` (wubu_rsi, test PASSES)
Status: `open` (67 gaps; bounded verifiable RSI loops, Goedel-style self-reference, reflection + metacognition)

## Theme IW: Neuromorphic / SNN cross-over
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (SNN gating ICLR-2026; multi-core neuromorphic train Nature-2026; event-driven 2026)
- IW01 Spike-encoding of tokens (token -> spike train) `wired` (wubu_neurom, test PASSES)
- IW02 Event-driven decode (compute only on spikes) `wired` (wubu_neurom, test PASSES)
- IW03 SNN energy model (1.05 TFLOPS/W neuromorphic vs GPU) `wired` (wubu_neurom, test PASSES)
- IW04 Brain-inspired gating for robustness (SNN gating mechanism) `wired` (wubu_neurom, test PASSES)
- IW05 Sparse computation via spike sparsity (55-85% memory-access cut) `wired` (wubu_neurom, test PASSES)
- IW06 Multi-core neuromorphic scheduling (parallel spike cores) `wired` (wubu_neurom, test PASSES)
- IW07 Membrane-potential accumulator (leaky integrate-and-fire) `wired` (wubu_neurom, test PASSES)
- IW08 Spike-based attention (attention on spike events) `wired` (wubu_neurom, test PASSES)
- IW09 Neuromorphic KV (KV as synaptic weights) `wired` (wubu_neurom, test PASSES)
- IW10 Spike-timing encoding (temporal coding of tokens) `wired` (wubu_neurom, test PASSES)
- IW11 SNN-to-ANN conversion (convert trained ANN to SNN) `wired` (wubu_neurom, test PASSES)
- IW12 Energy-sparsity correlation (energy saved per sparsity level) `wired` (wubu_neurom, test PASSES)
- IW13 Event-driven token selection (spikes gate token processing) `wired` (wubu_neurom, test PASSES)
- IW14 Neuromorphic memory (synaptic weight storage) `wired` (wubu_neurom, test PASSES)
- IW15 Spike-rate monitoring (activity health) `wired` (wubu_neurom, test PASSES)
- IW16 Threshold adaptation (firing threshold tuning) `wired` (wubu_neurom, test PASSES)
- IW17 Neuromorphic MoE (expert activation by spikes) `wired` (wubu_neurom, test PASSES)
- IW18 Spike-based speculative decode (spike drafter) `wired` (wubu_neurom, test PASSES)
- IW19 Neuromorphic energy ledger (J per spike, ties IJ) `wired` (wubu_neurom, test PASSES)
- IW20 Spike-train compression (event compression) `wired` (wubu_neurom, test PASSES)
- IW21 SNN robustness (noise tolerance of spike codes) `wired` (wubu_neurom, test PASSES)
- IW22 Neuromorphic scheduler (event-driven scheduling, ties IR) `wired` (wubu_neurom, test PASSES)
- IW23 Spike-based retrieval (associative recall via spikes, ties IP) `wired` (wubu_neurom, test PASSES)
- IW24 Membrane decay tuning (leak rate per layer) `wired` (wubu_neurom, test PASSES)
- IW25 Neuromorphic weight quant (synaptic weight precision) `wired` (wubu_neurom, test PASSES)
- IW26 Event-driven batching (batch on event density) `wired` (wubu_neurom, test PASSES)
- IW27 Spike-timing-dependent plasticity (STDP-style memory write) `wired` (wubu_neurom, test PASSES)
- IW28 Neuromorphic forward pass (spike forward alternative) `wired` (wubu_neurom, test PASSES)
- IW29 Sparse-event attention (attention only on active tokens) `wired` (wubu_neurom, test PASSES)
- IW30 SNN training emulation (surrogate gradient) `wired` (wubu_neurom, test PASSES)
- IW31 Neuromorphic memory decay (synaptic decay, ties IP05) `wired` (wubu_neurom, test PASSES)
- IW32 Spike latency model (event timing overhead) `wired` (wubu_neurom, test PASSES)
- IW33 Neuromorphic robustness benchmark (perturbation tests) `wired` (wubu_neurom, test PASSES)
- IW34 Event-driven KV eviction (evict on event inactivity) `wired` (wubu_neurom, test PASSES)
- IW35 Spike energy accounting (per-spike J model) `wired` (wubu_neurom, test PASSES)
- IW36 Neuromorphic prefix cache (spike prefix sharing) `wired` (wubu_neurom, test PASSES)
- IW37 Spike-train entropy (information per spike) `wired` (wubu_neurom, test PASSES)
- IW38 SNN-to-engine adapter (spike I/O bridge) `wired` (wubu_neurom, test PASSES)
- IW39 Neuromorphic world-model (spike-based state, ties IN) `wired` (wubu_neurom, test PASSES)
- IW40 Event-driven reasoning (reason on sparse events) `wired` (wubu_neurom, test PASSES)
- IW41 Spike threshold schedule (threshold annealing) `wired` (wubu_neurom, test PASSES)
- IW42 Neuromorphic top-k (spike-based selection) `wired` (wubu_neurom, test PASSES)
- IW43 SNN accuracy-energy frontier (Pareto) `wired` (wubu_neurom, test PASSES)
- IW44 Event-driven prefill (sparse prefill) `wired` (wubu_neurom, test PASSES)
- IW45 Spike-train watermark (event provenance) `wired` (wubu_neurom, test PASSES)
- IW46 Neuromorphic cache coherence (spike cache consistency) `wired` (wubu_neurom, test PASSES)
- IW47 Spike-based continual learning (online spike learning, ties BB) `wired` (wubu_neurom, test PASSES)
- IW48 Neuromorphic attention sink (sink as tonic spiking) `wired` (wubu_neurom, test PASSES)
- IW49 Event-driven telemetry (spike counters) `wired` (wubu_neurom, test PASSES)
- IW50 SNN mixed-precision (spike + analog hybrid) `wired` (wubu_neurom, test PASSES)
- IW51 Neuromorphic energy envelope (power cap on spikes, ties IJ03) `wired` (wubu_neurom, test PASSES)
- IW52 Spike-train dedup (redundant event suppression) `wired` (wubu_neurom, test PASSES)
- IW53 Neuromorphic KV quant (synaptic KV compression) `wired` (wubu_neurom, test PASSES)
- IW54 Event-driven sampling (spike-gated decoding) `wired` (wubu_neurom, test PASSES)
- IW55 SNN stability analysis (spike dynamics) `wired` (wubu_neurom, test PASSES)
- IW56 Neuromorphic memory consolidation (synaptic replay, ties BB) `wired` (wubu_neurom, test PASSES)
- IW57 Spike-based verification (verify on spikes) `wired` (wubu_neurom, test PASSES)
- IW58 Neuromorphic model selector (SNN vs ANN by task) `wired` (wubu_neurom, test PASSES)
- IW59 Event-driven context management (spike context budgets) `wired` (wubu_neurom, test PASSES)
- IW60 Spike-train augmentation (event dropout) `wired` (wubu_neurom, test PASSES)
- IW61 Neuromorphic error handling (spike fault tolerance) `wired` (wubu_neurom, test PASSES)
- IW62 Event-driven RL (spike rewards, ties GG) `wired` (wubu_neurom, test PASSES)
- IW63 SNN benchmark harness (energy/accuracy evals) `wired` (wubu_neurom, test PASSES)
- IW64 Neuromorphic provenance (spike-source tracking) `wired` (wubu_neurom, test PASSES)
- IW65 Event-driven energy operator (spike budget pick, ties IJ07) `wired` (wubu_neurom, test PASSES)
- IW66 Spike-based alignment (preference on spikes, ties IM) `wired` (wubu_neurom, test PASSES)
- IW67 Neuromorphic AGI substrate (event-driven cognitive architecture) `wired` (wubu_neurom, test PASSES)
Status: `open` (67 gaps; spike/event-driven crossover, neuromorphic energy, STDP memory)

## Theme IX: Fuzzing / robustness / security
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (prompt-fuzzing evasion 2026; LogicFuzz NDSS 2026; autonomous fuzzing CERT 2026; EU-AI-Act robustness)
- IX01 Prompt-fuzz harness (adversarial prompt variants) `wired` (wubu_fuzz, test PASSES)
- IX02 Evasion-rate measurement (per-category guardrail evasion) `wired` (wubu_fuzz, test PASSES)
- IX03 Guardrail sensitivity matrix (keyword-adjacent robustness) `wired` (wubu_fuzz, test PASSES)
- IX04 Autonomous fuzzing pipeline (LLM-supervised fuzzing) `wired` (wubu_fuzz, test PASSES)
- IX05 Crash validator (filter unreachable crashes) `wired` (wubu_fuzz, test PASSES)
- IX06 Fuzz-log analysis (LLM trace triage) `wired` (wubu_fuzz, test PASSES)
- IX07 Semantic-fuzz oracle (behavior divergence, not just crashes) `wired` (wubu_fuzz, test PASSES)
- IX08 Coverage-guided prompt mutation `wired` (wubu_fuzz, test PASSES)
- IX09 Robustness regression gate (fuzz on every model change) `wired` (wubu_fuzz, test PASSES)
- IX10 Adversarial-prompt taxonomy (jailbreak categories) `wired` (wubu_fuzz, test PASSES)
- IX11 Robustness benchmark suite (measurable robustness) `wired` (wubu_fuzz, test PASSES)
- IX12 Guardrail stress profile (per-guardrail weakness map) `wired` (wubu_fuzz, test PASSES)
- IX13 Input-validation layer (schema-check adversarial input) `wired` (wubu_fuzz, test PASSES)
- IX14 Fuzz-seed curation (high-value seed prompts) `wired` (wubu_fuzz, test PASSES)
- IX15 Mutation operator library (prompt mutation ops) `wired` (wubu_fuzz, test PASSES)
- IX16 Robustness scorecard (per-model robustness metrics) `wired` (wubu_fuzz, test PASSES)
- IX17 Prompt-injection detector (injection-pattern classifier) `wired` (wubu_fuzz, test PASSES)
- IX18 Output-validation gate (validate generated output) `wired` (wubu_fuzz, test PASSES)
- IX19 Fuzz-round budget (bounded fuzz campaigns) `wired` (wubu_fuzz, test PASSES)
- IX20 Vulnerability triage ledger (found + fixed registry) `wired` (wubu_fuzz, test PASSES)
- IX21 Robustness-vs-quality tradeoff monitor `open`
- IX22 Adversarial example archive (replayable attack corpus) `open`
- IX23 Fuzzer self-healing (auto-recover fuzz stalls) `open`
- IX24 Robustness telemetry (per-input robustness signals) `open`
- IX25 Input-schema fuzzing (malformed structured input) `open`
- IX26 Injection-mitigation layers (defense-in-depth) `open`
- IX27 Robustness delta tracking (regression detection) `open`
- IX28 Fuzz coverage metrics (prompt-space coverage) `open`
- IX29 Adversarial distillation defense (robust training signal) `open`
- IX30 Fuzz-oracle calibration (false-positive control) `open`
- IX31 Prompt-leak detection (data-exfiltration guard) `open`
- IX32 Robustness energy budget (fuzz under J cap, ties IJ) `open`
- IX33 Security audit ledger (auditable security posture) `open`
- IX34 Input canonicalization (normalize adversarial variants) `open`
- IX35 Fuzz differential testing (same input, model variants) `open`
- IX36 Robustness auto-repair (detect + patch weak guardrails) `open`
- IX37 Adversarial robustness eval harness (NDSS-style) `open`
- IX38 Injection-resistance benchmark (standardized evals) `open`
- IX39 Fuzz-to-fix loop (fuzz finds, fix verifies) `open`
- IX40 Robustness model card (documented robustness) `open`
- IX41 Input-token anomaly detection (outlier input detection) `open`
- IX42 Guardrail redundancy (overlapping defenses) `open`
- IX43 Fuzz mutation seeds from real incidents `open`
- IX44 Robustness under resource limits (degraded-but-safe) `open`
- IX45 Security regression CI (fuzz in the pipeline) `open`
- IX46 Adversarial-prompt generation (auto-generate attacks) `open`
- IX47 Robustness attribution (which layer failed) `open`
- IX48 Fuzz-parallelization (parallel fuzz workers) `open`
- IX49 Injection-resistance training (robust fine-tune) `open`
- IX50 Robustness SLA (minimum robustness bar) `open`
- IX51 Fuzz campaign reports (structured findings) `open`
- IX52 Guardrail evolution (update guardrails from findings) `open`
- IX53 Adversarial robustness scoring (quantified defense) `open`
- IX54 Fuzz-verifier integration (fuzz feeds the verifier) `open`
- IX55 Robustness debt tracking (known weaknesses ledger) `open`
- IX56 Input-entropy guard (reject adversarial entropy spikes) `open`
- IX57 Robustness provenance (which defense caught what) `open`
- IX58 Fuzz coverage dashboards `open`
- IX59 Adversarial robustness transfer (attacks transfer across models) `open`
- IX60 Robustness-aware sampling (defense-aware decode) `open`
- IX61 Security-posture operator (auto-apply robustness configs) `open`
- IX62 Fuzz memory-safety (C-level crash fuzz, ties the kernel) `open`
- IX63 Robustness regression tests (per-gap assertion) `open`
- IX64 Adversarial input ledger (append-only attack log) `open`
- IX65 Robustness calibration (threshold fitting on attacks) `open`
- IX66 Security benchmark comparison (vs baseline defenses) `open`
- IX67 Fuzz-to-operator loop (findings drive config promotion) `open`
Status: `open` (67 gaps; fuzz/evasion measurement + autonomous fuzzing + guardrail hardening)

## Theme IY: Prompt compression / context budgeting
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (LLMLingua-2; LongLLMLingua; RECOMP; Doc2Atom; Cartridges/CAS; LaMR; SES-RAG; GRC; EPC)
- IY01 LLMLingua perplexity-gated token drop (small-LM scoring) `open`
- IY02 LLMLingua-2 token classification (distilled BERT-level compressor) `open`
- IY03 LongLLMLingua question-aware reordering `open`
- IY04 Selective-Context self-information pruning (2x content, 40% compute) `open`
- IY05 RECOMP extractive+abstractive compression with selective augmentation `open`
- IY06 Doc2Atom compositional parametric memory (knowledge atoms + micro-LoRA) `open`
- IY07 Cartridges at Scale (modular KV caches, distractor mixing, budget manager) `open`
- IY08 LaMR multi-rubric code-context pruning (semantic + dependency CRFs) `open`
- IY09 SES-RAG semantic segmentation + query expansion + density truncation `open`
- IY10 GRC unified generation/retrieval/compression (meta latent tokens) `open`
- IY11 EPC expected-predictive compression (write-time retention by predicted questions) `open`
- IY12 Lost-in-the-middle mitigation (reorder important context) `open`
- IY13 Lexical-density-aware budgeting (dense contexts need more budget) `open`
- IY14 Tool-schema compression (44-50% schema token savings, ties agentic) `open`
- IY15 In-context autoencoder (continuous-embedding context) `open`
- IY16 Context distillation to LoRA (Doc-to-LoRA) `open`
- IY17 Latent-memory generation (compressed KV as updatable memory) `open`
- IY18 Hybrid paged attention for compressed context `open`
- IY19 Compression-ratio governor (target ratio with quality guard) `open`
- IY20 Compressed-prompt fidelity audit (reconstruction check) `open`
- IY21 Question-aware compression (query-conditioned retention) `open`
- IY22 Task-agnostic compressor (works across tasks) `open`
- IY23 Compression benchmark harness (compression quality evals) `open`
- IY24 Streaming compression (compress incrementally) `open`
- IY25 Compression energy accounting (compress vs not, ties IJ) `open`
- IY26 Retrieval-aware compression (retain retrieval-critical spans) `open`
- IY27 Per-token importance score caching `open`
- IY28 Compression curriculum (progressively harder compression) `open`
- IY29 Compressor-model choice (small-LM vs classifier vs heuristic) `open`
- IY30 Compression telemetry (ratio, quality, latency) `open`
- IY31 Context-budget planner (budget per stage: system/prompt/evidence) `open`
- IY32 Evidence-retention sufficiency (answerability check) `open`
- IY33 Compression + RAG integration (compress retrieved docs) `open`
- IY34 Agentic context pruning (multi-turn agent contexts, LaMR-style) `open`
- IY35 Compressed-KV paging (compressed pages) `open`
- IY36 Compositional compression (atom-level composition) `open`
- IY37 Compression provenance (what was compressed away) `open`
- IY38 Query-router for atom selection (Doc2Atom-style) `open`
- IY39 Micro-adapter injection (per-atom LoRA, ties lora) `open`
- IY40 Cartridge rotation (budget-managed cartridge swap) `open`
- IY41 Compression-quality monitor (post-compression performance) `open`
- IY42 Compress-or-keep decision (selective compression) `open`
- IY43 Token-budget inheritance (parent -> child agent budgets) `open`
- IY44 Compression-aware sampling (compressed context sampling) `open`
- IY45 Context-density profiler (density per window) `open`
- IY46 Compressed-prompt safety (never compress safety instructions) `open`
- IY47 Compression rollback (keep the original if quality drops) `open`
- IY48 Multi-stage compression (compress progressively) `open`
- IY49 Compression verification (answerability after compression) `open`
- IY50 Context-budget operator (auto-budget by task, ties IK) `open`
- IY51 Compressed-memory integration (compressed context as memory, ties IP) `open`
- IY52 Compression under energy budget (compress to save J) `open`
- IY53 Token-cost ledger (compression savings accounting) `open`
- IY54 Compression benchmark vs full-context baseline `open`
- IY55 Adaptive compression ratio (per-request ratio) `open`
- IY56 Compressor staleness (re-compress on context change) `open`
- IY57 Compression + eviction integration (compressed + evicted, ties IO) `open`
- IY58 Compressed-prompt telemetry (per-prompt stats) `open`
- IY59 Compression failure handling (fallback to full context) `open`
- IY60 Cross-model compression transfer (compress once, use anywhere) `open`
- IY61 Compression dataset distillation (train compressor from LLM outputs) `open`
- IY62 Context-budget fairness (per-tenant compression, ties IR) `open`
- IY63 Compression provenance audit (reproducible compression) `open`
- IY64 Compressed-context continual learning (compress + learn, ties BB) `open`
- IY65 Compression robustness (compressed adversarial input) `open`
- IY66 Compressor model portability (no external LM dependency) `open`
- IY67 Compression-to-operator loop (compression config promotion) `open`
Status: `open` (67 gaps; LLMLingua-family + RECOMP + cartridges + latent memory + density-aware budgeting)

## Theme IZ: Mixture-of-experts routing frontier
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (Routing-Free MoE 2604.00801; PathMoE 2603.18297; expert specialization 2505.22323; DeepSeek-V3 aux-free)
- IZ01 Routing-free MoE (experts self-activate, no centralized router) `open`
- IZ02 Path-constrained MoE (concentrated expert paths, 11% lower entropy) `open`
- IZ03 Expert-specialization gradient objective (diversify expert behaviors) `open`
- IZ04 Auxiliary-loss-free balancing (per-expert bias, DeepSeek-V3 style) `open`
- IZ05 Router z-loss (logit regularization for stability) `open`
- IZ06 Token-choice + expert-choice hybrid balancing `open`
- IZ07 Similarity-preserving routers (load balance via expert similarity) `open`
- IZ08 Device-level balancing (per-device expert grouping) `open`
- IZ09 Fine-grained expert dispatch (many small experts, DeepSeek-V2) `open`
- IZ10 Interleaved MoE layers (every 4th/6th layer MoE) `open`
- IZ11 Routing consistency monitor (cross-layer path consistency) `open`
- IZ12 Expert entropy monitor (routing entropy health) `open`
- IZ13 Router perturbation robustness (22.5x robust paths) `open`
- IZ14 Expert-collapse prevention (idle-expert guard) `open`
- IZ15 Load-balance telemetry (per-expert utilization) `open`
- IZ16 Expert specialization score (how distinct are experts) `open`
- IZ17 MoE weight quant (expert weights at low precision, ties quant) `open`
- IZ18 Expert caching (hot-expert weight cache) `open`
- IZ19 MoE speculative decode (draft expert routing) `open`
- IZ20 Expert prefetch (predict next experts, prefetch weights) `open`
- IZ21 MoE energy accounting (per-expert J, ties IJ) `open`
- IZ22 Routing path replay (remember good paths) `open`
- IZ23 Expert load scheduler (batch routing by expert load) `open`
- IZ24 MoE memory tiering (cold experts to slow tier) `open`
- IZ25 Adaptive expert count (grow experts by need) `open`
- IZ26 Router determinism (same input -> same experts) `open`
- IZ27 Expert dropout (train-time expert regularization) `open`
- IZ28 MoE continual learning (new experts for new tasks, ties BB) `open`
- IZ29 Router calibration (router confidence calibration) `open`
- IZ30 Expert routing graph (path visualization) `open`
- IZ31 MoE + Hopfield routing (associative expert selection, ties IP) `open`
- IZ32 Expert weight sharing (shared expert subspaces) `open`
- IZ33 MoE fault tolerance (expert failure fallback) `open`
- IZ34 Router distillation (small router for big MoE) `open`
- IZ35 Expert ensemble (multiple experts for one token) `open`
- IZ36 MoE load-balance benchmark (routing fairness evals) `open`
- IZ37 Expert pruning (drop redundant experts) `open`
- IZ38 MoE token-budget (per-token expert budget) `open`
- IZ39 Routing-aware KV (expert-specific KV partitioning) `open`
- IZ40 Expert importance (which experts matter) `open`
- IZ41 MoE + speculative + paged (combined acceleration) `open`
- IZ42 Router adversarial robustness (routing attacks) `open`
- IZ43 Expert temperature (routing softmax temperature) `open`
- IZ44 MoE incremental experts (add experts online) `open`
- IZ45 Expert-gating MLP (per-expert gating) `open`
- IZ46 MoE telemetry to the operator (routing health) `open`
- IZ47 Expert memory pinning (hot experts in fast memory) `open`
- IZ48 MoE capacity factor tuning (capacity governor) `open`
- IZ49 Router attention (attention-based routing) `open`
- IZ50 Expert credit assignment (which expert helped) `open`
- IZ51 MoE continual specialization (experts specialize over time) `open`
- IZ52 Routing-free activation patterns (AoE/ReMoE comparisons) `open`
- IZ53 Expert load rebalancing (live expert migration) `open`
- IZ54 MoE energy frontier (expert activation vs J) `open`
- IZ55 Router explainability (why this expert) `open`
- IZ56 Expert dedup (merge similar experts) `open`
- IZ57 MoE + agentic routing (task-aware expert selection) `open`
- IZ58 Expert watermark (per-expert provenance) `open`
- IZ59 MoE robustness benchmark (expert perturbation evals) `open`
- IZ60 Router prior (domain-prior routing) `open`
- IZ61 Expert vector cache (expert output cache) `open`
- IZ62 MoE scheduling (expert compute scheduling, ties IR) `open`
- IZ63 Router feedback loop (routing errors -> retrain router) `open`
- IZ64 Expert bias adaptation (bias-based load balancing) `open`
- IZ65 MoE quantization-aware routing (quantized router) `open`
- IZ66 Expert co-activation analysis (which experts fire together) `open`
- IZ67 MoE operator (auto-tune routing config, ties IV) `open`
Status: `open` (67 gaps; routing-free/path-constrained MoE + aux-loss-free balancing + expert specialization)

## Theme JA: Architecture hybrids (attention + SSM)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (Falcon-H1 hybrid; Hymba hybrid-head; Qwen3-Next GDN+Gated-Attn; Kimi Linear; 2507.12442 SSM characterization)
- JA01 Falcon-H1 parallel hybrid (attention + Mamba2 layers, 256K ctx) `open`
- JA02 Hymba hybrid-head (attention + SSM heads in one layer, 11x KV cut) `open`
- JA03 Qwen3-Next GDN + gated-attention alternation (262K native ctx) `open`
- JA04 SSM-at-scale analysis (57K energy 1492J -> 370J) `open`
- JA05 Hybrid Pareto analysis (accuracy vs TTFT frontier) `open`
- JA06 SSM recall-limitation compensation (attention for precise recall) `open`
- JA07 Hybrid layer-position design (which layers are attention) `open`
- JA08 SSM local + attention global (hybrid receptive fields) `open`
- JA09 Hybrid KV budget (attention layers keep KV, SSM layers don't) `open`
- JA10 Hybrid decode scheduling (per-layer mechanism dispatch) `open`
- JA11 SSM prefill speed (SSM TTFT advantage) `open`
- JA12 Hybrid accuracy-parity evaluation (hybrid >= transformer) `open`
- JA13 SSM energy model at scale (energy vs ctx, ties IJ) `open`
- JA14 Hybrid streaming (SSM constant memory + attention window) `open`
- JA15 Gated-attention long-context stability (hybrid stability) `open`
- JA16 Hybrid reasoning accuracy (long-context reasoning on hybrids) `open`
- JA17 SSM + attention co-training (hybrid training recipe) `open`
- JA18 Hybrid quantization (quantize both mechanisms) `open`
- JA19 SSM state + KV unified cache (one memory system) `open`
- JA20 Hybrid speculative decode (SSM drafter + attention verifier) `open`
- JA21 Hybrid architecture selector (auto-pick hybrid ratio) `open`
- JA22 SSM long-context needle test (hybrid recall evals) `open`
- JA23 Hybrid memory bound (attention window + SSM state) `open`
- JA24 SSM on-device viability (consumer-hardware long context) `open`
- JA25 Hybrid layer ablation (which layers need attention) `open`
- JA26 SSM state size tuning (state dimension) `open`
- JA27 Hybrid context switching (mechanism switch on context) `open`
- JA28 SSM + rotary interaction (position in SSM) `open`
- JA29 Hybrid benchmark harness (attention vs SSM vs hybrid) `open`
- JA30 Hybrid energy Pareto (energy/accuracy curves) `open`
- JA31 SSM numerical stability at scale (state drift) `open`
- JA32 Hybrid prefix caching (prefix in both mechanisms) `open`
- JA33 SSM long-context memory accounting (state bytes) `open`
- JA34 Hybrid token efficiency (mechanism-aware token budget) `open`
- JA35 SSM parallel scan on CPU (efficient scan kernels) `open`
- JA36 Hybrid robustness (perturbation resilience of hybrids) `open`
- JA37 SSM attention-sink equivalents (SSM sink tokens) `open`
- JA38 Hybrid eviction (evict attention KV, keep SSM state) `open`
- JA39 SSM speculative draft (recurrent draft heads) `open`
- JA40 Hybrid alignment (preference-align hybrids, ties IQ) `open`
- JA41 SSM world-model integration (SSM for stateful world, ties IN) `open`
- JA42 Hybrid MoE (MoE layers in hybrid models) `open`
- JA43 SSM multi-modal (SSM for audio/video sequences) `open`
- JA44 Hybrid energy operator (mechanism pick by energy, ties IJ07) `open`
- JA45 SSM state snapshot (checkpoint recurrent state) `open`
- JA46 Hybrid context-length switch (switch mechanism past a length) `open`
- JA47 SSM hardware mapping (scan-friendly layout) `open`
- JA48 Hybrid continual learning (state + weights, ties BB) `open`
- JA49 SSM tokenizer interplay (byte-level state inputs) `open`
- JA50 Hybrid provenance (mechanism attribution per token) `open`
- JA51 SSM quantization at scale (quantized recurrent state) `open`
- JA52 Hybrid verifier (verify across mechanisms) `open`
- JA53 SSM capacity analysis (state capacity vs KV) `open`
- JA54 Hybrid serving (schedule hybrid requests, ties IR) `open`
- JA55 SSM robustness benchmark (hybrid perturbation evals) `open`
- JA56 Hybrid telemetry (per-mechanism counters) `open`
- JA57 SSM memory consolidation (state as memory, ties IP) `open`
- JA58 Hybrid energy ledger (per-mechanism J) `open`
- JA59 SSM long-horizon stability (very-long-context behavior) `open`
- JA60 Hybrid architecture search (auto hybrid design) `open`
- JA61 SSM forgetting (state decay, ties IP05) `open`
- JA62 Hybrid multi-tenant (hybrid cache sharing) `open`
- JA63 SSM differential privacy (state privacy) `open`
- JA64 Hybrid watermark (mechanism-tagged outputs) `open`
- JA65 SSM speculative verification (SSM-verified drafts) `open`
- JA66 Hybrid model portability (run on any hardware) `open`
- JA67 Hybrid operator (auto hybrid config, ties IV) `open`
Status: `open` (67 gaps; attention+SSM hybrids, Hymba hybrid-head, Falcon-H1, energy/accuracy Pareto)

## Theme JB: Multimodal token compression
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (MM token compression survey 2507.20198; VisionSelector; visual-text token efficiency 2026)
- JB01 VisionSelector learnable visual-token selection `open`
- JB02 Visual-text token efficiency (text-as-pixels saves 38-58% decoder tokens) `open`
- JB03 Image token compression (patch merging) `open`
- JB04 Video token compression (temporal redundancy) `open`
- JB05 Audio token compression (spectral redundancy) `open`
- JB06 Cross-modal token alignment (CLIP-style, ties CC03) `open`
- JB07 Visual redundancy detection (similar-patch dedup) `open`
- JB08 Modality-aware KV (per-modality KV budgets) `open`
- JB09 Multimodal attention sparsity (vision tokens sparse attention) `open`
- JB10 Token-compression survey gaps (all surveyed methods) `open`
- JB11 Visual token importance scoring (salience-based retention) `open`
- JB12 Audio-visual fusion compression (joint token compression) `open`
- JB13 Multimodal token budget planner (per-modality budgets) `open`
- JB14 Vision encoder efficiency (ViT patch efficiency) `open`
- JB15 Multimodal eviction (evict low-salience modality tokens, ties IO) `open`
- JB16 Cross-modal prefix (shared multimodal prefix) `open`
- JB17 Visual token streaming (streaming image tokens) `open`
- JB18 Multimodal energy (per-modality J, ties IJ) `open`
- JB19 Visual token dedup (repeated-region suppression) `open`
- JB20 Modality routing (which modality matters per task) `open`
- JB21 Multimodal Hopfield memory (cross-modal patterns, ties IP) `open`
- JB22 Vision-language alignment quality monitor `open`
- JB23 Audio token quantization (compressed audio tokens) `open`
- JB24 Visual token reordering (salience-first ordering) `open`
- JB25 Multimodal compression benchmark (MM token evals) `open`
- JB26 Cross-modal retrieval compression (retrieve + compress) `open`
- JB27 Visual attention sink (vision sink tokens) `open`
- JB28 Multimodal speculative decode (vision draft) `open`
- JB29 Token-efficiency for multimodal (dense modality contexts, ties IY) `open`
- JB30 Visual token provenance (which region produced the token) `open`
- JB31 Multimodal cache sharing (cross-request visual KV reuse) `open`
- JB32 Video frame dedup (temporal frame similarity) `open`
- JB33 Audio-visual token fusion (early fusion compression) `open`
- JB34 Multimodal robustness (adversarial modality input, ties IX) `open`
- JB35 Vision token budget governor (per-image token cap) `open`
- JB36 Multimodal alignment energy (alignment cost) `open`
- JB37 Visual token curriculum (easy->hard visual tasks) `open`
- JB38 Modality-fusion attention (attention over fused modalities) `open`
- JB39 Multimodal memory tiers (modality-tiered memory) `open`
- JB40 Visual compression quality audit (perceptual loss checks) `open`
- JB41 Cross-modal token transfer (text cues -> vision tokens) `open`
- JB42 Multimodal prefix cache (vision prefix reuse) `open`
- JB43 Audio event detection (audio token salience) `open`
- JB44 Multimodal planning (plan over modalities, ties IN) `open`
- JB45 Visual token embedding quant (vision embedding compression) `open`
- JB46 Multimodal continual learning (new modalities, ties BB) `open`
- JB47 Cross-modal adversarial robustness (modality attacks) `open`
- JB48 Multimodal energy operator (modality budget pick) `open`
- JB49 Visual token sampling (salience-based token sampling) `open`
- JB50 Multimodal verifier (cross-modal consistency check) `open`
- JB51 Video temporal compression (frame-rate adaptation) `open`
- JB52 Audio-visual coherence (AV alignment check) `open`
- JB53 Multimodal telemetry (per-modality counters) `open`
- JB54 Visual token watermark (image-region provenance) `open`
- JB55 Cross-modal distillation (vision teacher -> text student) `open`
- JB56 Multimodal alignment drift monitor (ties IM06) `open`
- JB57 Visual context management (visual context budgets) `open`
- JB58 Multimodal speculative verification (cross-modal verify) `open`
- JB59 Token-efficiency-aware vision (fewer tokens, same info) `open`
- JB60 Multimodal OOM safety (modality-budget OOM guard) `open`
- JB61 Visual KV quantization (quantized vision KV, ties IO10) `open`
- JB62 Multimodal scheduler (modality-aware scheduling, ties IR) `open`
- JB63 Cross-modal attention pruning (prune low-cross-attention tokens) `open`
- JB64 Multimodal provenance ledger (auditable modality inputs) `open`
- JB65 Visual token importance model (salience predictor) `open`
- JB66 Multimodal frontier (token-compression quality frontier) `open`
- JB67 Multimodal operator (auto modality config, ties IV) `open`
Status: `open` (67 gaps; vision/audio/video token compression + cross-modal budgets + salience retention)

## Theme JC: Quantization frontier (weights + QAT)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (1.58-bit QAT bottom-up 2411.05882; 16->1.58 transition ACL-2025; BitNet b1.58; 2-bit 2026)
- JC01 1.58-bit QAT (ternary weights via quantization-aware training) `open`
- JC02 16->1.58 transition schedule (when to switch precision mid-training) `open`
- JC03 BitNet 1.58 regularizer view (ternary as regularization) `open`
- JC04 Weight-only 1.58 inference path `open`
- JC05 Two-phase QAT (full-precision warm-up then quantize) `open`
- JC06 Per-layer precision schedule (layer-adaptive bit width) `open`
- JC07 Activation-aware QAT (quantize with activation ranges) `open`
- JC08 Quantization curriculum (gradually reduce bit width) `open`
- JC09 Ternary GEMV optimization (BitNet-style kernel, ties B03) `open`
- JC10 2-bit QAT (2-bit weights with QAT recovery) `open`
- JC11 QAT gradient handling (straight-through estimators) `open`
- JC12 Quantization-aware KV training (QKV in the loop) `open`
- JC13 Precision transition monitor (when to transition) `open`
- JC14 QAT energy accounting (quantized inference J, ties IJ) `open`
- JC15 Quantized fine-tuning (QAT during fine-tune) `open`
- JC16 Bit-width ablation (per-width accuracy curves) `open`
- JC17 QAT robustness (quantized model robustness) `open`
- JC18 Quantization-aware alignment (align quantized models, ties IQ) `open`
- JC19 Mixed-precision QAT (per-tensor precision) `open`
- JC20 QAT evaluation harness (quantized evals) `open`
- JC21 1.58-bit scaling laws (ternary scaling behavior) `open`
- JC22 QAT convergence speed (quantized training speed) `open`
- JC23 Quantized speculative decode (quantized drafter) `open`
- JC24 QAT + LoRA (quantized LoRA adapters) `open`
- JC25 Quantization-aware distillation (teacher guides quantized student) `open`
- JC26 QAT stability (quantization training stability) `open`
- JC27 Quantized memory footprint (weights + KV + activations) `open`
- JC28 QAT operator (auto bit-width pick, ties IV) `open`
- JC29 Quantized MoE (quantized experts, ties IZ) `open`
- JC30 QAT continual learning (quantized continual learning, ties BB) `open`
- JC31 Ternary attention (quantized attention) `open`
- JC32 QAT precision schedule search (search the schedule) `open`
- JC33 Quantized hybrid models (quantize SSM+attention, ties JA) `open`
- JC34 QAT hardware mapping (quantized kernels per device, ties IS) `open`
- JC35 Quantization-aware embedding (quantized embeddings, ties IT) `open`
- JC36 QAT data selection (which data to quantize-train on) `open`
- JC37 Quantized inference accuracy monitor (perplexity guard) `open`
- JC38 QAT + unlearning (quantized forget, ties IM) `open`
- JC39 Ternary KV (1.58 KV cache, ties IO) `open`
- JC40 QAT reproducibility (seeded quantized training) `open`
- JC41 Quantized long-context (quantized KV for long ctx) `open`
- JC42 QAT energy frontier (bits vs J frontier) `open`
- JC43 Quantization-aware RAG (quantized retrievers) `open`
- JC44 QAT multi-objective (accuracy + energy + size) `open`
- JC45 Quantized world-model (quantized state, ties IN) `open`
- JC46 QAT benchmark suite (quantized training evals) `open`
- JC47 Ternary momentum (quantized optimizer states) `open`
- JC48 Quantized multi-tenant (quantized serving, ties IR) `open`
- JC49 QAT adversarial robustness (quantized model attacks) `open`
- JC50 Quantization-aware speculative (spec with quantized models) `open`
- JC51 QAT transfer (quantize small, transfer to big) `open`
- JC52 Quantized memory consolidation (quantized memory, ties IP) `open`
- JC53 QAT watermark (quantized provenance) `open`
- JC54 Quantized streaming (quantized KV streaming, ties L) `open`
- JC55 QAT precision governor (adaptive precision by loss) `open`
- JC56 Quantized attention kernels (int8 attention) `open`
- JC57 QAT curriculum search (auto curriculum) `open`
- JC58 Quantized tokenizer embeddings (quantized vocab) `open`
- JC59 QAT + speculative + quantized (full-stack quantized) `open`
- JC60 Quantized memory tiers (quantized cold KV, ties A06) `open`
- JC61 QAT energy operator (precision pick by energy) `open`
- JC62 Quantized agentic (quantized agents, ties AD) `open`
- JC63 QAT safety (quantized alignment safety) `open`
- JC64 Quantized telemetry (per-precision counters) `open`
- JC65 QAT fault tolerance (quantized error resilience) `open`
- JC66 Quantized continual self-improvement (quantized RSI, ties IV) `open`
- JC67 QAT frontier (bits/accuracy/energy Pareto) `open`
Status: `open` (67 gaps; 1.58-bit QAT, precision-transition schedules, quantized full-stack integration)

## Theme JD: AGI meta-needs (metacognition + self-governance frontier)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (metacognition survey 2607.11881; MetaCogAgent 2605.17292; metacognitive harness 2605.14186; Meta-R1)
- JD01 Capability profile per agent (calibrated competence scores) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_metagame)
- JD02 Self-assessment before task execution (confidence estimate) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_metagame2)
- JD03 Expected-calibration-error tracker (confidence vs outcome) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_credit)
- JD04 JOL (judgment-of-learning) signal extraction from the decode path `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_model)
- JD05 Metacognitive harness (JOL -> inference-time control: stop/retry) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_loopguard)
- JD06 Meta-level regulation (meta policy over the base policy, Meta-R1 style) `open` (driver: wubu_policy)
- JD07 Strategy selection by competence (pick the strategy the profile says wins) `open` (driver: wubu_metagame)
- JD08 Monitor-actor separation (the verifier structurally independent of the actor) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_verify)
- JD09 Self-prediction of learning progress (competence growth curve) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_metagame)
- JD10 Calibration without ground truth (self-supervised recalibration) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_uq)
- JD11 Verbalized-uncertainty faithfulness (does the model's word match its probs) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_verify)
- JD12 MetaCog-Eval-style benchmark (700-task capability benchmark) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_metagame)
- JD13 Task-delegation by capability (route the task to the capable agent) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_agentic_os)
- JD14 Capability-profile update loop (profiles drift with the model) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_metagame)
- JD15 Confidence-conditioned compute allocation (uncertain -> more compute, ties IK) `open` (driver: wubu_ttc)
- JD16 Monitoring-thoughts (trace the agent's reasoning, audit) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_loopguard)
- JD17 Second-order anomaly detection (the monitor flags its own drift) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_uq)
- JD18 Calibration telemetry (session-level calibration tracking) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_credit)
- JD19 Competence-difficulty gap (agent competence vs task difficulty) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_resource)
- JD20 Metacognitive reflection prompts (structured self-reflection) `open` (driver: wubu_metagame2)
- JD21 Self-monitoring loop (monitor -> regulate -> re-measure) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_selfimprove)
- JD22 Capability asymmetry detection (which agents differ) `open` (driver: wubu_agentic_os)
- JD23 Strategy-exploration under uncertainty (metacog-driven exploration) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_bandit)
- JD24 Confidence-calibrated sampling (confidence-scaled decode) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_model)
- JD25 Learning-progress prediction (predict next improvement) `open` (driver: wubu_metagame)
- JD26 Self-assessment audit (are self-assessments honest) `open` (driver: wubu_metagame)
- JD27 Metacognitive skill library (metacog skills as retrievable skills) `open` (driver: wubu_metagame2)
- JD28 Regulation policy (when to retry/stop/delegate) `open` (driver: wubu_policy)
- JD29 Metacog energy budget (self-monitoring costs J, ties IJ) `open` (driver: wubu_energy)
- JD30 Calibration drift monitor (calibration degrades over time) `wired` (wubu_metacog, test_metacog PASSES) (driver: wubu_uq)
- JD31 Confidence-stability check (confidence oscillation) `open` (driver: wubu_uq)
- JD32 Metacog feedback into the loop-ledger (the loop monitors itself) `open` (driver: wubu_metagame)
- JD33 Capability-transfer prediction (does competence transfer) `open` (driver: wubu_metagame)
- JD34 Self-predicted pass@1 (model predicts its own success) `open` (driver: wubu_uq)
- JD35 Metacog-driven early stopping (stop when calibrated confidence peaks) `open` (driver: wubu_ttc)
- JD36 Regulation under budget (metacog control under J cap) `open` (driver: wubu_energy)
- JD37 Competence-weighted delegation (delegate by calibrated competence) `open` (driver: wubu_agentic_os)
- JD38 Self-monitoring independence check (monitor drift from actor) `open` (driver: wubu_verify)
- JD39 Metacog benchmarks per dimension (reasoning/retrieval/code/math/commonsense) `open` (driver: wubu_metagame)
- JD40 Calibration-cost model (calibration effort vs benefit) `open` (driver: wubu_energy)
- JD41 Self-assessment variance (how noisy are self-assessments) `open` (driver: wubu_uq)
- JD42 Metacog knowledge base (capability facts the agent holds) `open` (driver: wubu_agentic_mem)
- JD43 Regulation-rule learning (learn the regulation policy from outcomes) `open` (driver: wubu_reinforce)
- JD44 Confidence-faithfulness gate (reject confident-but-wrong) `open` (driver: wubu_verify)
- JD45 Metacog + speculative decode (confidence-gated draft acceptance) `open` (driver: wubu_specdec)
- JD46 Self-monitoring telemetry to the user (visible self-awareness) `open` (driver: wubu_loopguard)
- JD47 Capability-profile persistence (profiles survive restarts) `open` (driver: wubu_agentic_mem)
- JD48 Metacog adversarial robustness (confidence attacks) `open` (driver: wubu_uq)
- JD49 Self-assessment vs peer-assessment (multi-agent calibration) `open` (driver: wubu_bft)
- JD50 Metacog-driven memory management (monitor memory, regulate retention, ties IP) `open` (driver: wubu_hopfield)
- JD51 Calibration-aware preference (align by calibrated preferences, ties IQ) `open` (driver: wubu_align)
- JD52 Metacog + world-model (self-model of the world-model, ties IN) `open` (driver: wubu_freeenergy)
- JD53 Regulation hysteresis (avoid stop/retry oscillation) `open` (driver: wubu_loopguard)
- JD54 Metacog sandbox (self-monitoring in isolation, ties AX) `open` (driver: wubu_sandbox_safekern)
- JD55 Confidence distribution analysis (confidence histograms) `open` (driver: wubu_uq)
- JD56 Self-improvement competence (does the loop know when to improve) `open` (driver: wubu_metagame)
- JD57 Metacog provenance (self-monitoring audit trail) `open` (driver: wubu_loopguard)
- JD58 Calibration transfer (calibrate once, reuse) `open` (driver: wubu_uq)
- JD59 Metacog + Hopfield memory (self-knowledge as associative patterns, ties IP) `open` (driver: wubu_hopfield2)
- JD60 Self-monitoring cost-benefit (monitor overhead vs gain) `open` (driver: wubu_energy)
- JD61 Competence-weighted tool selection (pick tools by capability) `open` (driver: wubu_tooluse)
- JD62 Metacog regulation ledger (regulation decisions logged) `open` (driver: wubu_loopguard)
- JD63 Self-assessment drift (self-assessments degrade) `open` (driver: wubu_uq)
- JD64 Metacog + continuous learning (self-monitoring during learning, ties BB) `open` (driver: wubu_ewc)
- JD65 Calibration-vs-complexity frontier (calibrate to the needed precision) `open` (driver: wubu_uq)
- JD66 Metacog operator (auto calibration config, ties IV) `open` (driver: wubu_metagame)
- JD67 Self-monitoring of the loop's own rate (the loop watches its close-rate) `open` (driver: wubu_metagame)
- JD68 Metacog + unlearning (self-knowledge forget, ties IM) `open` (driver: wubu_align)
- JD69 Confidence-gated hallucination guard (low-confidence -> abstain) `open` (driver: wubu_verify)
- JD70 Metacog + energy (calibrated compute spend) `open` (driver: wubu_energy)
- JD71 Self-assessment explainability (why this competence) `open` (driver: wubu_uq)
- JD72 Metacog + RSI (the loop metacognates its own improvement) `open` (driver: wubu_metagame)
- JD73 Calibration decay model (calibration half-life) `open` (driver: wubu_uq)
- JD74 Metacog + multi-tenant (per-tenant capability profiles, ties IR) `open` (driver: wubu_agentic_os)
- JD75 Self-monitoring of memory health (ties IP67) `open` (driver: wubu_hopfield2)
- JD76 Metacog robustness (self-monitoring under adversarial input, ties IX) `open` (driver: wubu_uq)
- JD77 Regulation under latency (time-bounded self-monitoring) `open` (driver: wubu_ttc)
- JD78 Metacog + tokenizer (token-efficiency of self-reflection) `open` (driver: wubu_ttc)
- JD79 Confidence-conditioned retry (retry only the uncertain) `open` (driver: wubu_loopguard)
- JD80 Metacog + speculative (confidence-gated draft verification) `open` (driver: wubu_specdec)
- JD81 Self-assessment normalization (cross-task calibration) `open` (driver: wubu_uq)
- JD82 Metacog + MoE (capability-aware expert routing, ties IZ) `open` (driver: wubu_moeroute)
- JD83 Self-monitoring of alignment health (ties IM06) `open` (driver: wubu_align)
- JD84 Metacog + multimodal (confidence per modality, ties JB) `open` (driver: wubu_uq)
- JD85 Calibration-aware batching (batch by confidence, ties IR) `open` (driver: wubu_contbatch)
- JD86 Metacog + quantization (calibrated quantized confidence, ties JC) `open` (driver: wubu_uq)
- JD87 Self-assessment federation (share profiles across agents) `open` (driver: wubu_bft)
- JD88 Metacog + streaming (self-monitoring over streams, ties L) `open` (driver: wubu_stream_kv)
- JD89 Regulation replay (replay good regulation decisions) `open` (driver: wubu_replay)
- JD90 Metacog + causal (self-model as causal graph, ties AW) `open` (driver: wubu_causal)
- JD91 Self-monitoring of the sweep (the loop audits its own research) `open` (driver: wubu_metagame)
- JD92 Metacog + linear attention (stateful self-monitoring, ties IU) `open` (driver: wubu_ssm)
- JD93 Calibration fairness (calibration across user groups) `open` (driver: wubu_uq)
- JD94 Metacog + fuzzing (self-monitoring under fuzz, ties IX) `open` (driver: wubu_verify)
- JD95 Self-assessment API (the AGI exposes its own competence) `open` (driver: wubu_loopguard)
- JD96 Metacog + prompt compression (self-reflection on compressed context, ties IY) `open` (driver: wubu_ttc)
- JD97 Regulation safety (metacog control under safety bounds) `open` (driver: wubu_loopguard)
- JD98 Metacog + Hopfield energy (self-monitoring via memory free-energy, ties IP) `open` (driver: wubu_hopfield2)
- JD99 Self-monitoring provenance (who watched what) `open` (driver: wubu_loopguard)
- JD100 Metacog benchmark operator (auto-run MetaCog-Eval, ties IV) `open` (driver: wubu_metagame)
Status: `open` (100 gaps; metacognition + calibration + self-governance, all driver-tagged to existing modules)

## Theme JE: Bonzi Buddy needs (the AGI's human face)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### 7-hop convergence (AIVA emotion-aware companion 2509.03212; affective computing; avatar presence / abstract-avatar advantage 2026; parasocial HAI)
- JE01 Emotion state machine (Bonzi's mood: valence/arousal) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE02 Persona reactivity (persona-consistent responses) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE03 Idle animation scheduler (Bonzi animates when idle) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE04 Speech prosody mapping (mood -> voice tone) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: TTS)
- JE05 Multimodal sentiment perception (user emotion from text/voice) `open` (driver: wubu_audio)
- JE06 Empathy engine (emotion-aware response selection) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_align)
- JE07 Conversation-turn timing (natural response latency) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE08 Mood memory (Bonzi remembers the user's emotional history) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_hopfield)
- JE09 Reactive animations (Bonzi reacts to events) `open` (driver: wubufx GUI)
- JE10 Abstract-avatar design (cartoonish = better presence, per the avatar research) `open` (driver: wubufx GUI)
- JE11 Social-presence cues (synthetic voice + avatar -> presence) `open` (driver: TTS)
- JE12 Notification personality (Bonzi delivers alerts in-character) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE13 Emotional-consistency guard (mood transitions stay coherent) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE14 Companion memory (conversational memory, ties IP) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_hopfield2)
- JE15 Context pruning for long chats (ties IO eviction) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_evict2026)
- JE16 Persona guardrails (Bonzi stays in bounds, ties IM) `open` (driver: wubu_align)
- JE17 Energy-aware idle (Bonzi idles cheap, ties IJ) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_energy)
- JE18 Speech batching (queue TTS efficiently) `open` (driver: wubufx GUI)
- JE19 Empathy escalation (mood-aware response depth) `open` (driver: wubu_align)
- JE20 User-mood tracking (persistent user emotional state) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_agentic_mem)
- JE21 Bonzi self-model (Bonzi knows its own mood) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_metagame)
- JE22 Turn-taking negotiation (when to speak) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE23 Emotional vocabulary (mood lexicon) `open` (driver: wubu_align)
- JE24 Avatar micro-expressions (subtle emotion cues) `open` (driver: wubufx GUI)
- JE25 Voice personality (stable synthetic voice identity) `open` (driver: TTS)
- JE26 Companion continuity (Bonzi remembers across sessions) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_agentic_mem)
- JE27 Proactive engagement (Bonzi initiates when appropriate) `open` (driver: wubufx GUI)
- JE28 Mood-lighting UI (theme reacts to mood, ties the WuBuOS theme engine) `open` (driver: wubufx GUI)
- JE29 Parasocial design (deliberate presence design, per the HCI literature) `open` (driver: wubufx GUI)
- JE30 Emotional anchoring (Bonzi anchors moods to memories) `open` (driver: wubu_hopfield2)
- JE31 Apology/repair scripts (Bonzi handles mistakes gracefully) `open` (driver: wubufx GUI)
- JE32 User-engagement telemetry (how engaged is the user) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_credit)
- JE33 Mood-triggered actions (sad user -> comfort behavior) `open` (driver: wubu_align)
- JE34 Companion topic memory (what the user cares about) `open` (driver: wubu_hopfield2)
- JE35 Speech emotion synthesis (voice conveys emotion) `open` (driver: TTS)
- JE36 Animation-mood coupling (mood drives animation set) `open` (driver: wubufx GUI)
- JE37 Companion honesty (Bonzi's calibrated confidence, ties JD) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_uq)
- JE38 Idle-noise budget (idle behavior under J cap, ties IJ) `open` (driver: wubu_energy)
- JE39 Social-cue detection (user engagement detection) `open` (driver: wubu_audio)
- JE40 Emotional dialogue policy (emotion-aware dialogue policy) `open` (driver: wubu_policy)
- JE41 Companion energy ledger (Bonzi's own energy accounting) `open` (driver: wubu_energy)
- JE42 Mood-drift monitor (Bonzi's mood stays coherent) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubufx GUI)
- JE43 Companion privacy (Bonzi forgets on request, ties IM unlearning) `open` (driver: wubu_align)
- JE44 Presence calibration (Bonzi's presence intensity) `open` (driver: wubufx GUI)
- JE45 Emotional consistency with the AGI (Bonzi reflects the AGI's state) `open` (driver: wubufx GUI)
- JE46 Companion interrupt handling (user interrupts mid-animation) `open` (driver: wubufx GUI)
- JE47 Voice-avatar sync (speech + animation timing) `open` (driver: wubufx GUI)
- JE48 Companion attachment model (long-term user relationship) `open` (driver: wubu_agentic_mem)
- JE49 Mood-aware music/ambience (Bonzi sets the mood) `open` (driver: wubufx GUI)
- JE50 Emotional learning (Bonzi learns the user's emotional patterns) `open` (driver: wubu_reinforce)
- JE51 Companion fault tolerance (Bonzi degrades gracefully) `open` (driver: wubufx GUI)
- JE52 Emotional context window (recent emotional context) `open` (driver: wubu_agentic_mem)
- JE53 User-valence prediction (predict user mood) `open` (driver: wubu_gp)
- JE54 Companion scheduling (when Bonzi engages, ties IR) `open` (driver: wubu_agentic_os)
- JE55 Emotional safety (Bonzi avoids emotional manipulation) `open` (driver: wubu_align)
- JE56 Mood histogram (Bonzi's mood distribution) `open` (driver: wubufx GUI)
- JE57 Companion memory consolidation (ties BB replay) `open` (driver: wubu_replay)
- JE58 Emotional event log (auditable emotional interactions) `open` (driver: wubu_loopguard)
- JE59 Speech clarity budget (clear speech under noise) `open` (driver: wubu_audio)
- JE60 Companion self-regulation (Bonzi manages its own arousal) `open` (driver: wubufx GUI)
- JE61 Emotional priming (mood-conditioned responses) `open` (driver: wubu_align)
- JE62 Companion handoff (Bonzi hands off to the AGI) `open` (driver: wubufx GUI)
- JE63 Voice identity stability (same voice, same persona) `open` (driver: TTS)
- JE64 Emotional calibration (Bonzi's emotion detection calibration) `open` (driver: wubu_uq)
- JE65 Companion engagement SLO (engagement targets) `open` (driver: wubu_credit)
- JE66 Mood-aware scheduling (Bonzi engages at the right time) `open` (driver: wubu_agentic_os)
- JE67 Emotional memory decay (old moods fade, ties IP12) `wired` (wubu_bonzi, test_bonzi PASSES) (driver: wubu_hopfield2)
- JE68 Companion notification coalescing (batch alerts) `open` (driver: wubufx GUI)
- JE69 Emotional micro-state (fine-grained mood states) `open` (driver: wubufx GUI)
- JE70 Empathy asymmetry (Bonzi's empathy vs the AGI's analysis) `open` (driver: wubufx GUI)
- JE71 Companion onboarding (Bonzi learns the user) `open` (driver: wubu_agentic_mem)
- JE72 Mood-triggered idle (sad idle vs happy idle) `open` (driver: wubufx GUI)
- JE73 Emotional consistency check (state machine coherence) `open` (driver: wubufx GUI)
- JE74 Companion energy profile (Bonzi's power envelope, ties IJ03) `open` (driver: wubu_energy)
- JE75 Emotional dialogue history (ties the conversation log) `open` (driver: wubu_agentic_mem)
- JE76 Companion resilience (Bonzi recovers from glitches) `open` (driver: wubufx GUI)
- JE77 Voice emotional range (prosody range) `open` (driver: TTS)
- JE78 Emotional context routing (which memory for the mood) `open` (driver: wubu_hopfield2)
- JE79 Companion privacy ledger (what Bonzi remembers, audited) `open` (driver: wubu_loopguard)
- JE80 Mood-to-action mapping (mood -> behavior table) `open` (driver: wubufx GUI)
- JE81 Emotional surprisal (Bonzi reacts to the unexpected) `open` (driver: wubu_freeenergy)
- JE82 Companion long-term goals (Bonzi supports user goals) `open` (driver: wubu_agentic_os)
- JE83 Emotional multi-agent (Bonzi coordinates with the AGI) `open` (driver: wubu_bft)
- JE84 Companion persona stability (persona never drifts) `open` (driver: wubu_align)
- JE85 Emotional energy tradeoff (empathy costs J, ties IJ) `open` (driver: wubu_energy)
- JE86 Companion accessibility (Bonzi adapts to the user) `open` (driver: wubufx GUI)
- JE87 Mood telemetry to the AGI (the AGI knows the user's mood) `open` (driver: wubufx GUI)
- JE88 Emotional memory priority (emotional memories rank high, ties IP) `open` (driver: wubu_hopfield)
- JE89 Companion verification (Bonzi's outputs verified) `open` (driver: wubu_verify)
- JE90 Emotional reinforcement (user feedback tunes Bonzi, ties GG) `open` (driver: wubu_reinforce)
- JE91 Companion streaming (Bonzi streams long conversations, ties L) `open` (driver: wubu_stream_kv)
- JE92 Mood-aware context budget (sad moments get more context) `open` (driver: wubu_ttc)
- JE93 Emotional adversarial robustness (mood attacks, ties IX) `open` (driver: wubu_align)
- JE94 Companion model selection (Bonzi picks the right model) `open` (driver: wubu_resource)
- JE95 Emotional graph memory (ties AE02 dedup, user-emotion graph) `open` (driver: wubu_agentic_mem)
- JE96 Companion heartbeat (Bonzi stays alive) `open` (driver: wubu_agentic_os)
- JE97 Mood-conditioned prefill (mood-aware prompt construction) `open` (driver: wubu_ttc)
- JE98 Emotional provenance (why Bonzi feels this way) `open` (driver: wubu_loopguard)
- JE99 Companion benchmark (companion-quality evals) `open` (driver: wubufx GUI)
- JE100 Bonzi operator (auto-tune Bonzi config, ties IV) `open` (driver: wubufx GUI)
Status: `open` (100 gaps; the AGI's human face: emotion state machine, companion memory, persona guardrails, energy-aware presence)

## Theme JF: Cross-resource Kevin-Bacon links (new needs x existing modules)
Status: `open` = not yet in engine; `wired` = implemented+tested.
### The meta-plan's highest-leverage axis: each gap TIES the new needs (JD/JE) to an EXISTING module
- JF01 Bonzi mood memory -> wubu_hopfield retrieval (mood as pattern) `wired` (wubu_bridge, test_bridge PASSES) (driver: integration JE08 x IP)
- JF02 AGI confidence -> wubu_loopguard gate (confidence-gated loop protection) `wired` (wubu_bridge, test_bridge PASSES) (driver: JD05 x loopguard)
- JF03 Bonzi idle energy -> wubu_energy ledger (idle under J cap) `open` (driver: JE17 x IJ)
- JF04 Companion persona -> wubu_align guardrails (persona bounds) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE16 x IM)
- JF05 User-mood -> wubu_agentic_mem tier (mood memory tiers) `open` (driver: JE20 x AE)
- JF06 Self-assessment -> wubu_credit turn-credit (calibrated credit) `wired` (wubu_bridge, test_bridge PASSES) (driver: JD02 x AH12)
- JF07 Bonzi context pruning -> wubu_evict2026 (chat eviction) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE15 x IO)
- JF08 Emotional memory -> wubu_hopfield2 episodic tags `open` (driver: JE30 x IP19)
- JF09 Companion speech -> wubu_audio pipeline (voice processing) `open` (driver: JE05 x CC02)
- JF10 Metacog regulation -> wubu_policy (regulation as policy) `wired` (wubu_bridge, test_bridge PASSES) (driver: JD06 x GG02)
- JF11 Bonzi mood -> wubufx theme engine (mood-lighting) `open` (driver: JE28 x theme)
- JF12 AGI JOL -> wubu_specdec (confidence-gated draft acceptance) `open` (driver: JD04 x HH01)
- JF13 Companion memory -> wubu_replay (emotional replay, ties BB) `open` (driver: JE57 x BB01)
- JF14 Bonzi calibration -> wubu_uq (companion confidence) `open` (driver: JE37 x FF04)
- JF15 Mood prediction -> wubu_gp (Gaussian mood process) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE53 x FF01)
- JF16 Companion engagement -> wubu_credit reward (engagement credit) `open` (driver: JE65 x AH12)
- JF17 Emotional surprisal -> wubu_freeenergy (prediction-error mood) `open` (driver: JE81 x IN01)
- JF18 Bonzi forget -> wubu_align unlearn (companion unlearning) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE43 x IM03)
- JF19 Metacog monitor -> wubu_verify (independent monitor) `open` (driver: JD08 x AX09)
- JF20 Companion scheduling -> wubu_agentic_os (Bonzi task scheduling) `open` (driver: JE54 x AD02)
- JF21 Emotional learning -> wubu_reinforce (mood RL) `open` (driver: JE50 x GG01)
- JF22 Self-assessment -> wubu_metagame archive (capability archive) `open` (driver: JD01 x AH05)
- JF23 Bonzi memory decay -> wubu_hopfield2 decay `wired` (wubu_bridge, test_bridge PASSES) (driver: JE67 x IP12)
- JF24 Metacog + world-model -> wubu_freeenergy (self-model FE) `open` (driver: JD52 x IN02)
- JF25 Companion streaming -> wubu_stream_kv (long-chat streaming) `open` (driver: JE91 x L)
- JF26 Mood-aware budget -> wubu_ttc (emotional compute budget) `open` (driver: JE92 x IK)
- JF27 Metacog + Hopfield -> wubu_hopfield2 self-knowledge `wired` (wubu_bridge, test_bridge PASSES) (driver: JD59 x IP)
- JF28 Companion multi-agent -> wubu_bft (Bonzi x AGI consensus) `open` (driver: JE83 x DD01)
- JF29 Emotional guardrails -> wubu_loopguard (mood safety gate) `open` (driver: JE55 x AG01)
- JF30 Calibration -> wubu_uq conformal (conformal confidence) `open` (driver: JD03 x FF04)
- JF31 Bonzi verification -> wubu_verify (companion output verify) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE89 x AX07)
- JF32 Emotional memory priority -> wubu_hopfield write-schedule `open` (driver: JE88 x IP06)
- JF33 Metacog delegation -> wubu_agentic_os (capability delegation) `open` (driver: JD13 x AD01)
- JF34 Companion energy profile -> wubu_energy freq-cap (Bonzi power cap) `open` (driver: JE74 x IJ03)
- JF35 Mood-conditioned prompt -> wubu_ttc budget `open` (driver: JE97 x IK01)
- JF36 Self-monitoring -> wubu_loopguard traj (monitoring traces) `wired` (wubu_bridge, test_bridge PASSES) (driver: JD16 x AG05)
- JF37 Companion presence -> wubufx render (presence via the framework) `open` (driver: JE10 x WuBuFX)
- JF38 Metacog + eviction -> wubu_evict2026 (retain high-confidence context) `open` (driver: JD35 x IO)
- JF39 Emotional dialogue policy -> wubu_policy (emotion policy) `open` (driver: JE40 x GG)
- JF40 Calibration drift -> wubu_uq monitor (drift detection) `open` (driver: JD30 x FF04)
- JF41 Bonzi mood -> wubu_ecs component (mood as ECS state) `open` (driver: JE01 x C06)
- JF42 Metacog + MoE -> wubu_moeroute (capability-aware routing) `open` (driver: JD82 x IZ)
- JF43 Companion privacy -> wubu_align monitor (privacy drift) `open` (driver: JE43 x IM06)
- JF44 Self-assessment -> wubu_uq (assessment uncertainty) `open` (driver: JD02 x FF04)
- JF45 Bonzi anomaly -> wubu_uq second-order (mood anomalies) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE42 x JD17)
- JF46 Metacog + RSI -> wubu_evolve (self-improve with self-knowledge) `open` (driver: JD72 x AX06)
- JF47 Companion memory -> wubu_vecsearch ANN (mood-vector retrieval) `open` (driver: JE14 x AV08)
- JF48 Emotional provenance -> wubu_loopguard audit `open` (driver: JE98 x AG05)
- JF49 Metacog + quant -> wubu_quantkv (calibrated quantized confidence) `open` (driver: JD86 x JC)
- JF50 Bonzi tool-use -> wubu_tooluse (companion tool dispatch) `open` (driver: JE62 x AX04)
- JF51 Emotional state -> wubu_ecs snapshot (mood checkpoint) `open` (driver: JE01 x C06)
- JF52 Metacog + linear attn -> wubu_ssm (stateful self-monitoring) `open` (driver: JD92 x IU)
- JF53 Companion heartbeat -> wubu_agentic_os backoff `open` (driver: JE96 x AD02)
- JF54 Emotional consistency -> wubu_align monitor (drift-free mood) `open` (driver: JE13 x IM06)
- JF55 Metacog + fuzzing -> wubu_verify (self-monitor under fuzz) `open` (driver: JD94 x IX)
- JF56 Bonzi streaming -> wubu_lmcache (companion prefix cache) `open` (driver: JE91 x A07)
- JF57 Mood histogram -> wubu_ecs (mood stats component) `open` (driver: JE56 x C06)
- JF58 Metacog + causal -> wubu_causal (self-causal model) `open` (driver: JD90 x AW01)
- JF59 Companion empathy -> wubu_align DPO (empathetic alignment) `wired` (wubu_bridge, test_bridge PASSES) (driver: JE06 x IM01)
- JF60 Calibration -> wubu_bo (calibrated Bayesian optimization) `open` (driver: JD03 x FF03)
- JF61 Bonzi idle -> wubu_scheduler (idle tick scheduling) `open` (driver: JE03 x C04)
- JF62 Metacog + Hopfield energy -> wubu_hopfield2 (memory FE self-monitor) `open` (driver: JD98 x IP)
- JF63 Emotional context -> wubu_ctx_manage (emotional context manager) `open` (driver: JE52 x context)
- JF64 Self-monitoring -> wubu_metagame2 skill (monitoring as skill) `open` (driver: JD27 x AH09)
- JF65 Companion voice -> wubu_ttc TTS (voice energy budget) `open` (driver: JE04 x IJ)
- JF66 Metacog + continual -> wubu_ewc (self-knowledge consolidation) `open` (driver: JD64 x BB02)
- JF67 Bonzi resilience -> wubu_agentic_os durable-exec `open` (driver: JE76 x AD03)
- JF68 Emotional memory -> wubu_agentic_mem dedup (mood dedup) `open` (driver: JE95 x AE02)
- JF69 Metacog + prompt-compression -> wubu_ttc (self-reflection on compressed ctx) `open` (driver: JD96 x IY)
- JF70 Companion SLO -> wubu_contbatch (Bonzi response SLO) `open` (driver: JE65 x HH04)
- JF71 Mood-aware prefill -> wubu_chunked_prefill (mood chunking) `open` (driver: JE97 x D04)
- JF72 Metacog + linear-attn kernels -> wubu_ssm_scan (stateful scan monitoring) `open` (driver: JD92 x IU35)
- JF73 Emotional network -> wubu_bft (mood consensus) `open` (driver: JE83 x DD04)
- JF74 Self-assessment -> wubu_resource (capability-driven resource) `wired` (wubu_bridge, test_bridge PASSES) (driver: JD19 x AH14)
- JF75 Bonzi voice -> wubu_audio FFT (voice mood analysis) `open` (driver: JE05 x CC02)
- JF76 Metacog + speculative -> wubu_medusa (confidence-gated tree draft) `open` (driver: JD45 x HH05)
- JF77 Companion provenance -> wubu_loopguard traj (Bonzi action log) `open` (driver: JE98 x AG05)
- JF78 Emotional energy -> wubu_energy ledger (mood J accounting) `open` (driver: JE85 x IJ02)
- JF79 Metacog + worldmodel -> wubu_worldmodel (self-worldmodel) `open` (driver: JD52 x AG04)
- JF80 Bonzi mood-lighting -> wubu_theme_engine (the WuBuOS theme) `open` (driver: JE28 x theme)
- JF81 Calibration -> wubu_bandit (calibrated exploration) `open` (driver: JD23 x FF06)
- JF82 Emotional RL -> wubu_dqn (mood Q-learning) `open` (driver: JE50 x GG05)
- JF83 Metacog + memory tier -> wubu_kv_tier (capability-tiered memory) `open` (driver: JD47 x A06)
- JF84 Companion multi-tenant -> wubu_agentic_os cap (Bonzi subtree) `open` (driver: JE83 x AD01)
- JF85 Self-monitoring -> wubu_ecs (monitor as component) `wired` (wubu_bridge, test_bridge PASSES) (driver: JD16 x C06)
- JF86 Bonzi emotion -> wubu_gptq (mood-aware quantized inference) `open` (driver: JE01 x B06)
- JF87 Metacog + eviction loss -> wubu_evict2026 loss model (calibrated eviction) `open` (driver: JD35 x IO14)
- JF88 Emotional telemetry -> wubu_energy (mood-power correlation) `open` (driver: JE87 x IJ)
- JF89 Companion onboarding -> wubu_agentic_mem consolidate (profile consolidation) `open` (driver: JE71 x AE01)
- JF90 Metacog + Hopfield separation -> wubu_hopfield2 (self-pattern separation) `open` (driver: JD59 x IP08)
- JF91 Bonzi failover -> wubu_resource degrade (Bonzi degradation tiers) `open` (driver: JE76 x AH15)
- JF92 Emotional memory replay -> wubu_distill (mood distillation) `open` (driver: JE57 x BB04)
- JF93 Metacog + HNSW -> wubu_vecsearch (capability vector index) `open` (driver: JD01 x AV01)
- JF94 Companion security -> wubu_safekern (Bonzi sandbox) `open` (driver: JE43 x AX08)
- JF95 Emotional calibration -> wubu_uq conformal (mood conformal sets) `open` (driver: JE64 x FF04)
- JF96 Metacog + energy optimal -> wubu_energy freq (calibrated compute at optimal f) `open` (driver: JD35 x IJ03)
- JF97 Bonzi mood graph -> wubu_causal (mood causal graph) `open` (driver: JE95 x AW01)
- JF98 Self-assessment transfer -> wubu_metagame improvement (competence delta) `open` (driver: JD33 x AH13)
- JF99 Emotional memory pruning -> wubu_evict2026 (mood-token eviction) `open` (driver: JE15 x IO)
- JF100 The meta-loop: close-rate ledger -> wubu_metagame (the loop self-governs) `wired` (wubu_bridge, test_bridge PASSES) (driver: meta-plan x IV)
Status: `open` (100 gaps; every gap ties a new need to an existing module -- the DA-metaplan's highest-leverage axis)
