# docs

<!-- repodoc:BEGIN -->
# Module Map (auto-generated 2026-08-04)

Full annotated table of `src/` modules. Regenerate with
`python3 tools/repodoc/repodoc.py . --modules`.

| File | Purpose |
|---|---|
| `src/bench.c` | GPU Output Projection — hidden @ output_weight^T via cuBLAS |
| `src/dequant_iq2_xxs.c` | IQ2_XXS block-level operations for on-the-fly dequant dot product. |
| `src/gaad_nesting_llm.c` | static int64_t golden_split_pos(int64_t length) { |
| `src/gguf_reader.c` | From ggml-common.h — lookup table for 1.5625 bpw dequantization |
| `src/kv_paged_attention.c` | - Prefix caching for shared prompts |
| `src/qlearner.c` | Reward = 1/(loss + eps): lower loss = higher reward. |
| `src/quantized_dot_generic.c` | Self-contained generic + SIMD implementations of quantized dot products. |
| `src/quantized_matmul.c` | For each output column, quantizes the F32 input to Q8_K then calls |
| `src/quantized_matmul_fixed.c` | col_stride_bytes: byte stride between columns (0 = packed) |
| `src/rsgd.c` | the Poincaré ball. The key steps: |
| `src/safetensors_reader.c` | F32 / F16 / BF16 / I8..I64 tensors to float32. |
| `src/safetensors_writer.c` | [ uint64 LE header_len ][ header JSON ][ padding to 8 ][ raw blob ] |
| `src/thread_pool.c` | ── Thread pool using OpenMP ─────────────────────────────────── |
| `src/tile_manager.c` | - Tiles = 64×64 "pixel" blocks (64 tokens each) |
| `src/wubu.c` | The WuBu-35M spine in C11 — original WaefreBeorn work (WaefreBeorn Umbrella License v3.0), the archived seed superseded by WuBu1 (docs/wubu1-base-model-design.md). Pure C11, no third-party deps. The forward |
| `src/wubu_4kv.c` | 1. KV-cache is memory-bandwidth-bound in decode (Roofline 2607.02558). |
| `src/wubu_acq.c` | - EI(x) = (μ-f*)Φ((μ-f*)/σ) + σ·φ((μ-f*)/σ)  [closed form] |
| `src/wubu_active.c` | - FF05: uncertainty sampling = query argmax σ(x); QBC = query argmax |
| `src/wubu_actor_critic.c` | - GG03: critic learns V(s) via TD: δ = r + γV(s') - V(s). Actor updates |
| `src/wubu_affinity.c` | C11, self-contained (Linux). No god headers. |
| `src/wubu_agentauth.c` | Agents in a multi-agent system exchange messages; without authentication a |
| `src/wubu_agentic_kv.c` | LMCache-vision-hash / LOOK-M / agentic-compaction 7-hop): |
| `src/wubu_agentic_mem.c` | - AE01 episodic->semantic consolidation: an episodic event is "distillable" |
| `src/wubu_agentic_os.c` | - AD01 9P capability enforcement: each agent gets a bounded subtree of the |
| `src/wubu_agentid.c` | - DD03: each CoAgent gets a verifiable identity (ID + name + capability |
| `src/wubu_agi.c` | 1. observe: push an observation into the hive |
| `src/wubu_align.c` | if (x > 0) return -logf(1.0f / (1.0f + expf(-x))); |
| `src/wubu_ambig.c` | static const wubu_us_slot_t *find_slot(const wubu_us_slot_t *state, |
| `src/wubu_amoeba.c` | can use the hive." The amoeba's cells ARE hive slots: |
| `src/wubu_arena.c` | Self-contained C11. See header. |
| `src/wubu_attn_gate.c` | *dynamically* to the attention output. Suppresses attention sinks and |
| `src/wubu_attn_kernels.c` | - P11 int2 KV dequant: KV stored as 2-bit (4 levels) per component with a |
| `src/wubu_attn_tune.c` | - L06 Quest: sub-linear attention by selecting, per query block, the top-k |
| `src/wubu_attnres.c` | C11, self-contained. AttnRes lets a layer READ representations written by |
| `src/wubu_audio.c` | hz→mel, power spectrogram, log-scale): |
| `src/wubu_awq.c` | Compression and Acceleration", MLSys 2024. |
| `src/wubu_backprop.c` | WuBu seed (12 layers, 7 Q heads / 1 KV head GQA, 448-dim, 16384 |
| `src/wubu_bandit.c` | - FF06: each "config family" (attention variant, quant scheme) is an arm. |
| `src/wubu_bf16_gemv.c` | dispatch + F32 fallback. C11. No third-party deps; uses <immintrin.h> only |
| `src/wubu_bft.c` | Two-Fold BFT, n=3f+1 threshold): |
| `src/wubu_bi.c` | norm change at layer l). Low BI = redundant layer (ShortGPT removes |
| `src/wubu_bo.c` | - FF03: maintains a candidate set, scores each with the acquisition function |
| `src/wubu_bonzi.c` | int wubu_bonzi_mood_step(wubu_bonzi_mood_t *m, float t_val, float t_ar, |
| `src/wubu_bonzi2.c` | int wubu_bonzi_sentiment(const float *text_feat, const float *voice_feat, |
| `src/wubu_bridge.c` | int wubu_br_mood_retrieve(const float *mood_patterns, int n_moods, |
| `src/wubu_bridge2.c` | Agnostic: a bridge-table (the JE emotion event → external driver), |
| `src/wubu_cache_advice.c` | C11, self-contained. Upgrades the ds4-ssd LRU slot-bank with a learned |
| `src/wubu_capacity_wall.c` | binding constraint oscillates between weight-I/O (W) and KV-I/O (K) and |
| `src/wubu_capzero.c` | - AF02 deny-by-default tool registry: an agent holds an explicit capability |
| `src/wubu_causal.c` | temporal/belief, logic engines, PDDL planning, abductive/counter-abductive, |
| `src/wubu_cegis.c` | - EE03: ∃f.∀x,y. φ(f,x,y). Loop: synthesize candidate f from grammar |
| `src/wubu_chunked_prefill.c` | Unveiled" (arXiv:2607.02558); disaggregated PD papers. |
| `src/wubu_cla.c` | C11, self-contained. CLA reduces KV cache by sharing K/V tensors across |
| `src/wubu_codeexec.c` | - AX07: verify generated code before it enters the decode loop. |
| `src/wubu_codesynth.c` | - AX10: the agent receives a textual spec (operation + func name), |
| `src/wubu_compress.c` | int wubu_comp_llmlingua(const float *perplexities, int n, float th, |
| `src/wubu_compress2.c` | int wubu_comp2_llmlingua(const float *perplexities, int n, float th, |
| `src/wubu_contbatch.c` | - HH04: schedule at iteration (token) granularity, not request granularity. |
| `src/wubu_continuous_batching.c` | C11, self-contained. Implements continuous batching (vLLM-style): |
| `src/wubu_coord.c` | access-control, intent-lock-before-edit, conflict-resolution 7-hop): |
| `src/wubu_credit.c` | - AH12: given a frozen reference model's answer-predictability before/after |
| `src/wubu_credit_sft.c` | (Orchard): a trajectory that never resolved still contains productive |
| `src/wubu_cross_attn.c` | (text) and K/V come from an encoder (vision/audio). Uses the same |
| `src/wubu_ctx_manage.c` | (L16 elastic context / N07 tiered-cache advisor / N14 MoD router). C11. |
| `src/wubu_ctxvm.c` | - AF08 4-level context hierarchy: L1 gen window, L2 session, L3 long-term, |
| `src/wubu_cuda_graph.c` | (Area E, items E.41/E.42/E.43/E.50). C11 planning logic is testable on CPU; |
| `src/wubu_db_cross.c` | gaps each import a database concept into the KV/decode engine as a small, |
| `src/wubu_dbstate.c` | static const char *find(const wubu_db_slot_t *state, int nslots, |
| `src/wubu_dedup.c` | A polynomial rolling hash over the window; the hash table maps the |
| `src/wubu_delta_net.c` | C11, self-contained. Implements the DeltaNet fast-weight update: |
| `src/wubu_deltanet.c` | int wubu_deltanet_state_init(wubu_deltanet_state_t *st, int n_heads, |
| `src/wubu_der.c` | int wubu_der_push(wubu_der_buffer_t *b, const float *teacher_logits, int ndim) |
| `src/wubu_dgm.c` | - AX01: DGM empirical gate -- verified=1 only when gen_text returns 0 |
| `src/wubu_dims.c` | See wubu_dims.h. The loader sets WUBU_DIMS from real tensor shapes; |
| `src/wubu_dims_gpu_stub.c` | CPU-only stub for the GPU dims sync symbol so CPU builds/tests link |
| `src/wubu_distill.c` | - BB04: teacher snapshot + KL divergence soft-target loss. |
| `src/wubu_dn2.c` | - S02 Gated DeltaNet-2: decouples erase and write. Two gates e (erase) and |
| `src/wubu_dqn.c` | - GG05: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]. Off-policy TD(0). |
| `src/wubu_dsa.c` | indexer (DSA indexer). Self-contained C11 (libc + libm only). |
| `src/wubu_eagle.c` | Draft model = truncated target model (fewer layers). |
| `src/wubu_early_exit.c` | See header. Self-contained C11. |
| `src/wubu_ecs.c` | wubu_ecs_t *wubu_ecs_create(int cap) { |
| `src/wubu_energy.c` | energy ledger can later be fed by real RAPL/CMU counters). |
| `src/wubu_epcap.c` | int wubu_epcap(const int *cost, int n, int budget, int *out) |
| `src/wubu_equiv_check.c` | int wubu_equiv_vectors(const float *a, const float *b, int n, |
| `src/wubu_eval.c` | int wubu_eval_run(const wubu_db_goal_t *goals, const wubu_eval_traj_t *trajs, |
| `src/wubu_eval_qat.c` | QAT-STE / per-channel / noise-injection 7-hop): |
| `src/wubu_evict2026.c` | int wubu_ev_pool_obs(const float *attn, int n, int w, float *out) |
| `src/wubu_evict2026b.c` | float wubu_ev_norm(float raw, float lo, float hi) |
| `src/wubu_evict2026c.c` | int wubu_evictc_h2o(const float *attention, int n, float th, int *keep) |
| `src/wubu_evolve.c` | - AX06: propose→verify→commit→regress loop. |
| `src/wubu_ewc.c` | - BB02: Elastic Weight Consolidation on the 15-dim sweep space. |
| `src/wubu_experibuf.c` | - BB01: reservoir-sampled ring buffer of past sweep configurations. |
| `src/wubu_expert_allreduce.c` | void wubu_allreduce_sum(const float *const *partials, int nranks, int len, float *out) { |
| `src/wubu_expert_choice.c` | Mixture-of-Experts", Google, 2024; Switch Transformer top-1 routing; |
| `src/wubu_fast_attn.c` | At 512K context, the per-query-position malloc(n_q_heads * attend_len * 4) |
| `src/wubu_flash_prefill.c` | Attention with IO-Awareness", NeurIPS 2022. |
| `src/wubu_flashdecode.c` | Self-contained C11. See header. Default chunk gives ~8 parallel KV chunks. |
| `src/wubu_fmt.c` | static int json_ok(const char *out) |
| `src/wubu_fp8.c` | uint8_t wubu_fp8_e4m3_from_f32(float x) { |
| `src/wubu_fraud.c` | evidence submission, trust decay, dispute arbitration): |
| `src/wubu_freeenergy.c` | inference (Theme IN). C11, deterministic. |
| `src/wubu_fuzz.c` | int wubu_fuzz_mutate(const char *in, char *out, int cap, uint32_t seed) |
| `src/wubu_fuzz2.c` | float wubu_fz2_tradeoff(float robustness, float quality, float w) |
| `src/wubu_gamebud.c` | clock_gettime(CLOCK_MONOTONIC, &ts); |
| `src/wubu_gemm.c` | - A panel packed into contiguous row-major (improves streaming + FMA |
| `src/wubu_gemma4_model.c` | Architecture: 48 layers, 40 sliding-window (HEAD_DIM=256) + 8 full-attention (HEAD_DIM=512 |
| `src/wubu_gemv_tune.c` | Pure C, routes through wubu_roofline for the B*-ridge decision. |
| `src/wubu_generate.c` | (doc 018 / K01). Self-contained C11. See header. |
| `src/wubu_gp.c` | - FF01: RBF kernel k(x,x') = σ²_f exp(-||x-x'||²/(2ℓ²)) + noise·δ. |
| `src/wubu_gptq.c` | Generative Pre-trained Transformers", ICLR 2023. |
| `src/wubu_grow.c` | the per-block weight byte size (all the block buffers) */ |
| `src/wubu_hadamard.c` | return n > 0 && (n & (n - 1)) == 0; |
| `src/wubu_hashrouter.c` | token. Slot k hashes (token_id, pos, salt_k, seed) with our own |
| `src/wubu_hive.c` | "live" when skip[s] == 0. Erase: skip[s] = 1, live--, and push the |
| `src/wubu_hopfield.c` | C11, deterministic, no third-party deps. |
| `src/wubu_hopfield2.c` | int wubu_hf_rk4_step(const float *state, const float *field, int dim, |
| `src/wubu_hopfield3.c` | static float dot(const float *a, const float *b, int d) |
| `src/wubu_hopfield4.c` | Implements the 26 remaining IP gaps (IP05-IP67 minus those already in |
| `src/wubu_hugepage.c` | bandwidth-bound and TLB-footprint-heavy; 2MB hugepages cut TLB misses and |
| `src/wubu_hwcaps.c` | See header. Self-contained C11. Raw CPUID, no third-party deps. |
| `src/wubu_hybrid.c` | int wubu_hyb_falcon(const float *attn_out, const float *ssm_out, |
| `src/wubu_hyper.c` | mobius_add_1d c x y = ((1 + 2cx·y + c·y²)·x + (1 - c·x²)·y) |
| `src/wubu_hyperbolic_output_proj.c` | exp_map(v): output[i] = tanh(||v||/R) * R/||v|| * v[i] |
| `src/wubu_imgenc.c` | static float lcg_randf(unsigned *seed) { |
| `src/wubu_integrate.c` | modules into the live decode path (option c: exploit discovered gaps). |
| `src/wubu_invariant.c` | - EE05: given a trace of loop states (var1, var2) at each iteration, discover |
| `src/wubu_kda.c` | C11, self-contained. KDA = DeltaNet with CHANNEL-WISE decay: each key channel |
| `src/wubu_kereq.c` | C11, self-contained. Genuine (if lightweight) SYMBOLIC prover: represents each |
| `src/wubu_kernel.c` | Adopted the kernel dispatch table pattern from waste_kernels[] |
| `src/wubu_kernel_backends.c` | register at runtime via wubu_kernel_register(). The engine never |
| `src/wubu_kv2026.c` | - Q02 ChunkKV: group consecutive KV tokens into semantic chunks, score each |
| `src/wubu_kv2026b.c` | - Q01 CentroidKV: cluster KV tokens by cosine similarity to a learned (here: |
| `src/wubu_kv2026c.c` | - Q11 DASH-KV: hash-based token-level attention scheduling. We compute a |
| `src/wubu_kv_adaptive.c` | LLMs via Entropy-Aware Cache Compression", ISCA 2025. |
| `src/wubu_kv_budget.c` | + footprint forecaster (L18 / L19 / N03 / N17). |
| `src/wubu_kv_cacheline.c` | starts on a cache-line boundary. This eliminates partial cache-line |
| `src/wubu_kv_compress.c` | slots carry little attention; retaining the *attention-mass-weighted* subset |
| `src/wubu_kv_evict.c` | See header for the policy. Self-contained C11. |
| `src/wubu_kv_runtime.c` | global g_kv_scheme instead of a compile-time #if, so the engine can pick the |
| `src/wubu_kv_select.c` | Pure C, routes through the tested wubu_roofline module. |
| `src/wubu_kv_shield.c` | cache accessed by untrusted indices (e.g. attacker-controlled attention spans, |
| `src/wubu_kv_styx.c` | KV-cache allocator (`wubu_kv_runtime.c`) and WuBuOS's 9P namespace. |
| `src/wubu_kv_tier.c` | HOT  = existing gqa_k_cache / gqa_v_cache (CPU RAM, current tokens) |
| `src/wubu_kv_transfer.c` | for a completed prefix to a transfer buffer (mmap'd temp file); a decode |
| `src/wubu_kvcache_quant.c` | KV-cache movement dominates bytes moved per token. |
| `src/wubu_kvquant.c` | C11, self-contained. FP8 (e4m3) and INT4-with-rotation KV storage. |
| `src/wubu_kvvq.c` | Self-contained C11. See header. |
| `src/wubu_latency.c` | - AF05 latency class (HRT/SRT/DT) + EDF/RM-ready scheduler hook: earliest- |
| `src/wubu_latentmoe.c` | C11, self-contained. 896 routed experts, top-k=16 active per token, PLUS a |
| `src/wubu_layer_skip.c` | y = x + gate * F(x)   where gate ∈ [0,1] |
| `src/wubu_linattn.c` | static float dot(const float *a, const float *b, int d) |
| `src/wubu_linattn2.c` | int wubu_la2_delta_write(float *state, int d, const float *k, const float *v, |
| `src/wubu_linear_attn.c` | These replace the O(n^2) attention with an O(n) recurrent state update. The |
| `src/wubu_lm_infinite.c` | - L13 LM-Infinite: landmark ("soft prompt") tokens are injected every `stride` |
| `src/wubu_lmcache.c` | latency via prefix offload + prefill/decode disaggregation. |
| `src/wubu_lookahead.c` | draft model, scan recent token history for a repeated n-gram and propose the |
| `src/wubu_loopguard.c` | LLM06/ASI02 tool-abuse cap, ASI08/strata JIT+HITL 7-hop): |
| `src/wubu_lora.c` | B^T @ A has shape [out_f, in_f] (matches W). Applied in place. |
| `src/wubu_lruk.c` | KV cache is a buffer pool; the right eviction policy is LRU-k (keep the k most |
| `src/wubu_masked_ce.c` | int wubu_masked_ce(const float *logits, const uint16_t *tokens, |
| `src/wubu_medusa.c` | - HH05: attach lightweight draft heads to the target's last layer → propose |
| `src/wubu_mega.c` | C11, self-contained. MEGA = single-head gated attention + multi-headed EMA |
| `src/wubu_mem_budget.c` | the safe KV cache size and forward buffer budget, never OOMs. |
| `src/wubu_metacog.c` | int wubu_mc_init(wubu_metacog_t *m, int n_agents) |
| `src/wubu_metagame.c` | fitness, faked-log lesson, self-improvement delta 7-hop): |
| `src/wubu_metagame2.c` | int wubu_meta_regulate(const float *policy_conf, int n, float th, int *action) |
| `src/wubu_mhc.c` | C11, self-contained. mHC widens the residual stream by factor `exp` and mixes |
| `src/wubu_mhc_mh.c` | manifold-constrained (row-softmax) mixing matrix, gated writes, and an |
| `src/wubu_misc_gaps.c` | P12/P13). C11, no third-party deps. |
| `src/wubu_mix.c` | long wubu_mix_build(const char **paths, const float *weights, int n, |
| `src/wubu_mla.c` | Mixture-of-Experts Language Model", arXiv:2405.04434. |
| `src/wubu_mm_adapter.c` | - CC04/CC06: projects vision/audio embeddings into text space (via |
| `src/wubu_mm_align.c` | - CC03: learned linear projection maps vision/audio features into the |
| `src/wubu_mm_kv.c` | - CC05: assembles the multimodal token prefix (vision + audio token IDs) |
| `src/wubu_mobius.c` | void wubu_mobius_add(const float *x, const float *y, int d, float R, float *z) { |
| `src/wubu_mobius_gyrate.c` | Optimized Möbius gyration using precomputed dot products. |
| `src/wubu_mobius_linear.c` | Helper: exp_map backward (matching interface from PGA backend) |
| `src/wubu_mobius_new.c` | τ = 1 + 2c⟨x,y⟩ + c²||x||²||y||² |
| `src/wubu_model.c` | Global tensor naming convention (set during model init) |
| `src/wubu_model_adapter.c` | self-contained, opaque). Hand-parses the JSON we care about: |
| `src/wubu_model_safetensors_bridge.c` | into wubuwizard's wubu_model_t and run them through the EXISTING |
| `src/wubu_moe.c` | GPU MoE expert forward (declared in wubu_model_gpu.cu, C linkage) |
| `src/wubu_moe2.c` | int wubu_moe2_route(const wubu_moe2_t *moe, const float *x, |
| `src/wubu_moe_backward.c` | Handles NULL expert weight pointers gracefully (skips that section). |
| `src/wubu_moe_grouped.c` | (Area D, items D.31/D.37/D.38). C11, self-contained. |
| `src/wubu_moe_hyperbolic.c` | Helper: map Euclidean vector to Poincaré ball via exp_map |
| `src/wubu_moe_hyperbolic_backward.c` | Poincaré router backward pass (single-level + nested 2-level). |
| `src/wubu_moe_rag.c` | KV-Packet / RACC / CAG / cross-doc-isolation 7-hop): |
| `src/wubu_moeroute.c` | - HH03: top-k routing with capacity factor C (each expert ≤ C tokens). |
| `src/wubu_moondream.c` | Self-contained C11 implementation of the MoonDream 3 bridge. |
| `src/wubu_more_spec.c` | (M07/M08/M09/M10/M15/M17/M18/M19/M20). C11. |
| `src/wubu_mxfp4.c` | C11, self-contained. MXFP4: 32-element blocks, each element E2M1 (1s/2e/1m), |
| `src/wubu_nest.c` | wubu_quat_t wubu_quat_mul(wubu_quat_t a, wubu_quat_t b) |
| `src/wubu_nested_ssm.c` | Nested SSM Forward Implementation |
| `src/wubu_nested_ssm_backward.c` | Nested SSM Forward-Save + Backward (BPTT through K Poincaré balls) |
| `src/wubu_neurom.c` | int wubu_neurom_encode(float value, float rate_max, float dt, int n_bins, |
| `src/wubu_nf4.c` | Quantization: normalize to [-1,1] via block absmax, then nearest-level |
| `src/wubu_ngram.c` | Pure C11, self-contained, zero external model weights. |
| `src/wubu_ngram_cascade.c` | Pure C11, self-contained. Uses prompt n-gram statistics to draft tokens. |
| `src/wubu_numerical_audit.c` | 1. No NaN / Inf in output (unless input has NaN/Inf) |
| `src/wubu_nvfp4.c` | uint8_t wubu_nvfp4_from_f32(float x) { |
| `src/wubu_paged_kv.c` | C11, self-contained. Implements vLLM-style paged attention bookkeeping: |
| `src/wubu_pagedkv.c` | - HH02: split KV into fixed-size blocks (16 tokens); logical block table → |
| `src/wubu_parallel_spec.c` | - V01 EAGLE-3 feature drafting: instead of drafting tokens, predict the next |
| `src/wubu_passk.c` | log C(n, k) via the log-gamma -- the counts are huge, the ratio is not */ |
| `src/wubu_pd_serve.c` | dynamic compute / mixture-of-depths (AC01-AC03). C11. |
| `src/wubu_pd_split.c` | C11, self-contained. Splits inference into a compute-bound prefill pool and a |
| `src/wubu_pim.c` | int wubu_pim_offload(int op_kind, long bytes, long compute_flops, |
| `src/wubu_pim2.c` | int wubu_pim2_bits(float sensitivity, float th_lo, float th_hi) |
| `src/wubu_planediv.c` | - AG02 control/data-plane separation: every input is tagged control-plane |
| `src/wubu_plateau.c` | float wubu_plateau_slope(const float *losses, int n, int window) |
| `src/wubu_poincare_gqa.c` | Dequant a [rows, cols] BF16/F16 matrix into F32 [cols, rows] (transposed), |
| `src/wubu_poincare_gqa_backward.c` | Helper: forward declarations for static helpers |
| `src/wubu_poincare_ssm_backward.c` | Poincaré SSM Backward (gyration chain rule) |
| `src/wubu_polar_pso.c` | serial bit reading for PolarQuant KV cache. |
| `src/wubu_polarquant.c` | paper (arXiv:2502.02617). Pairs of coordinates are transformed to |
| `src/wubu_policy.c` | - GG02: linear softmax policy π(a|s) = softmax(W·s + b). Baseline b(s) |
| `src/wubu_ppo.c` | - GG04: ratio r = π_θ(a|s)/π_θ_old(a|s). L = min(r·A, clip(r,1-ε,1+ε)·A). |
| `src/wubu_pref.c` | static float lg(float x) { return logf(1.0f + expf(-x)); } |
| `src/wubu_pref2.c` | static float lg(float x) { return logf(1.0f + expf(-x)); } |
| `src/wubu_prefix_cache.c` | Pure C11, self-contained. Uses FNV-1a 64-bit hash (no OpenSSL dep). |
| `src/wubu_priority.c` | the shame list (rolled-back events) prevents repeating failures, the |
| `src/wubu_prover.c` | - EE04: a lightweight propositional + arithmetic prover. Given premises and |
| `src/wubu_prover2.c` | checking in C11. The model proposes steps; the verifier accepts or |
| `src/wubu_q4k_m.c` | C11, self-contained. Matches GGUF Q4_K layout exactly. |
| `src/wubu_q8.c` | C11, self-contained. Q8_0 is effectively lossless (~0.5% vs FP16) at half |
| `src/wubu_quant_selector.c` | (N04 batch-size-aware, N05 context-length precision ladder, N09 PMC roofline |
| `src/wubu_quantkv.c` | - HH06: KV cache is memory-bound at 512K ctx. INT8 per-group (symmetric) |
| `src/wubu_rambus.c` | Interleaved banks + row-buffer banking + RDRAM-cycle cost model. C11. |
| `src/wubu_recency.c` | float wubu_recency_weight(long i, long n, float base, float power) |
| `src/wubu_reinforce.c` | - GG01: ∇J(θ) = E[Σ_t ∇log π(a_t|s_t) · (G_t - b)]. Monte-Carlo returns |
| `src/wubu_repetition.c` | as a ring buffer. repeat_penalty scans the recent window; DRY hashes |
| `src/wubu_resource.c` | degradation 70B->14B->7B 7-hop): |
| `src/wubu_reverify.c` | int wubu_reverify_init(wubu_reverify_t *r, double shift_thresh, |
| `src/wubu_ring_attn.c` | over 1M+ token contexts using the ring communication pattern. |
| `src/wubu_rollout.c` | int wubu_rollout_alloc(const float *succ, int n, int budget, |
| `src/wubu_roofline.c` | C11, self-contained. Implements the data-movement framework from the I/O |
| `src/wubu_rope_prefetch.c` | position encoding means K vectors at nearby positions have similar |
| `src/wubu_rotate.c` | Self-contained C11. See header for the invariance proof. |
| `src/wubu_rsi.c` | int wubu_rsi_gate(float verifier_score, float th, int *consecutive_fails) |
| `src/wubu_safekern.c` | - AF11 non-tamperable interrupt: a stop signal that lives OUTSIDE the agent's |
| `src/wubu_safetensors_model.c` | wubuwizard forward pass consumes. Dequantizes F16/BF16/F32 on the fly |
| `src/wubu_safetensors_shard.c` | See wubu_safetensors_shard.h. Self-contained; uses safetensors_reader. |
| `src/wubu_sandbox_safekern.c` | - AX08: bridge between sandbox isolation and safekern capabilities. |
| `src/wubu_save.c` | every trained checkpoint was a private .st dump no standard tooling |
| `src/wubu_scheduler.c` | batching + iteration-level KV-cache merge. Model-agnostic: operates |
| `src/wubu_seed.c` | static uint64_t splitmix64(uint64_t *x) |
| `src/wubu_self_cascade.c` | Pure C11. Calls a provided small-model forward function. |
| `src/wubu_semcons.c` | distributed semantic agreement, smart contract signalling): |
| `src/wubu_serve.c` | int wubu_serve_admit(long used_tokens, long budget, long req_tokens) |
| `src/wubu_serve2.c` | float wubu_serve2_fairness(long achieved, long entitled) |
| `src/wubu_si.c` | int wubu_si_init(wubu_si_t *s, const double *params, int ndim, double lambda) |
| `src/wubu_sindy.c` | - EE02: from trajectory (x_t, dx/dt) builds a candidate library (const, |
| `src/wubu_smoothquant.c` | Self-contained C11. See header. |
| `src/wubu_smt_check.c` | "Equivalence Checking of ML GPU Kernels". |
| `src/wubu_soa.c` | Arrays) for cache-friendly channel-wise access. In AoS, token i's hidden |
| `src/wubu_sparse_attn.c` | (L11 NSA / L12 MoBA). Self-contained C11. |
| `src/wubu_spawn.c` | C11, self-contained (no god headers). |
| `src/wubu_spec_cascade.c` | Pure C11, self-contained. Two cascade flavors: |
| `src/wubu_spec_decode.c` | proposal + target model verification via rejection sampling. |
| `src/wubu_spec_tuner.c` | should track the *measured* acceptance rate. If acceptance is high, raise K; |
| `src/wubu_spec_variants.c` | remaining M-family gaps are *combinations* of machinery already wired this |
| `src/wubu_specdec.c` | - HH01: draft model proposes K tokens; target verifies all in ONE forward |
| `src/wubu_ssd_moe.c` | See include/wubu_ssd_moe.h. Self-contained; C11; opaque ctx. |
| `src/wubu_ssm.c` | Global tensor naming convention (defined here for CORE_OBJ visibility) |
| `src/wubu_ssm_chunked.c` | written in matrix (outer-product / rank-1) form: |
| `src/wubu_ssm_scan.c` | C11, self-contained. Parallel (Blelloch) prefix scan over chunked SSM |
| `src/wubu_ssm_workspace.c` | static wubu_ssm_workspace_t g_pool[WUBU_SSM_WORKSPACE_MAX_LAYERS]; |
| `src/wubu_stream_kv.c` | 2026): at long context decode is KV-bandwidth/capacity bound. StreamingLLM |
| `src/wubu_symbolic.c` | - AW07: a Prolog-ish engine -- facts (predicate(args)) + rules |
| `src/wubu_symreg.c` | - EE01: discovers closed-form equations from (x, y) data. We implement a |
| `src/wubu_synth.c` | - AX05: spec→C11 code generation with compile-time verification. |
| `src/wubu_sys2026.c` | C11. Policy cores (hardware plumbing abstracted; the decision logic is real). |
| `src/wubu_sys_tune.c` | - L10 SeerAttention: per-head dynamic sparse attention -- predict each head's |
| `src/wubu_tandem.c` | Two stages (A=prefill/RSP, B=decode/RDP) run in tandem over a ring handoff. |
| `src/wubu_taskbd.c` | - BB03: detect task boundaries via performance divergence. When the |
| `src/wubu_tensor_store.c` | model file never loads weights -- it builds a name->(offset,dtype,shape) |
| `src/wubu_ternary.c` | int wubu_ternary_qat(const float *w, int n, float alpha, int8_t *out) |
| `src/wubu_thread_spec.c` | Two pinned thread pools (prefill / decode). See header. Self-contained C11. |
| `src/wubu_threshsig.c` | - DD02: simplified threshold signature scheme. Each agent produces a |
| `src/wubu_token.c` | int wubu_tok_bit_bpe_cost(int byte_len, int bits_per_symbol) |
| `src/wubu_token2.c` | float wubu_tok2_bench(long tokens, long chars) |
| `src/wubu_tokenizer.c` | Qwen3.6 exact byte-to-token mapping from original tokenizer.json |
| `src/wubu_tokenizer_hf.c` | Self-contained: embeds a tiny, correct recursive-descent JSON scanner |
| `src/wubu_tooluse.c` | - AX04: tool schema registry -- name+description+JSON Schema input, |
| `src/wubu_train.c` | grows here: the REAL backprop (wubu_backprop) + the REAL Muon |
| `src/wubu_traj_grpo.c` | recipe core). Group-relative advantage over the G trajectories: |
| `src/wubu_traj_sft.c` | The input is COPIED (never modified in place -- the in-place NUL |
| `src/wubu_tst.c` | TST: Token Superposition Training Implementation |
| `src/wubu_ttc.c` | - Q08 PolyKV: a shared, asymmetrically-compressed KV pool across agents. |
| `src/wubu_turboquant.c` | frame-based planning, and LRU eviction for the TurboQuant+/RotorQuant |
| `src/wubu_ubus.c` | Backends: CPU scalar (always), CPU OpenMP (12 threads), GPU cuBLAS |
| `src/wubu_uq.c` | - FF04: bootstrap ensemble over sweep replays → variance σ_uc² = 1/(B-1)Σ(f_b-μ)². |
| `src/wubu_user_sim.c` | static const char *find_slot(const wubu_us_user_t *u, |
| `src/wubu_uuid.c` | for 74 bits of randomness (only used once at startup — subsequent UUIDs |
| `src/wubu_value.c` | - GG06: Bellman optimality: V*(s) = max_a [R(s,a) + γ Σ_s' P(s'|s,a) V*(s')]. |
| `src/wubu_vecsearch.c` | PQ/RaBitQ/SQ quantization, FlashAttention, similarity metrics, |
| `src/wubu_verify.c` | - AX09: a lightweight formal gate — assertion-based invariant checking |
| `src/wubu_vision.c` | int wubu_vision_selector(const float *scores, int n, float th, int *keep) |
| `src/wubu_vision_moondream.c` | patch_embed → 27× ViT block → post_ln → proj_mlp → exp_map → Poincaré |
| `src/wubu_width.c` | old block in its top-left corner EXACTLY (no scaling) and zeroes the |
| `src/wubu_wm_kv.c` | (N02) + per-layer compute budget floor (N08). |
| `src/wubu_worldmodel.c` | 7-hop): pure LLM reasoning fails at agency because it is OPEN-LOOP -- it |
| `src/wubu_yarn.c` | C11, self-contained. Extends a model's trained context to longer lengths by |
<!-- repodoc:END -->
