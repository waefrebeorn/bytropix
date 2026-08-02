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
- BB06 SI path-integral importance `open` (research: path tracking beyond sweep)
- BB07 Dark experience replay (distillation + replay hybrid) `open` (research: model-level)

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
- EE07 Closed-loop self-verification (re-discover on shift) open (research: needs world-model)

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
