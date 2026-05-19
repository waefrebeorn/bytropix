# Plan — May 18, 2026 — PHASE 4 DONE. NEXT: GPU DECODE.

## Phase 0: CORE INFERENCE FIXED ✓
### Achievements:
- GQA Q/gate interleave bug fixed ✓ (cos-sim -0.51 → 0.9968)
- MoE quantized path wired ✓ (IQ2_XXS/IQ3_XXS/IQ4_XS via blob)
- Shared expert quantized path wired ✓ (Q5_K/Q6_K)
- Per-layer dump infrastructure ✓ (modded llama.cpp + bytropix)
- Layer-by-layer comparison tooling ✓ (python3 script)

## Phase 1: DA v10 Real Priorities (May 18)

### Task 1.1: Build llama reference dumper tool [P0] ✓
Replaces llama-cli with direct libllama.so linkage for fast per-layer dumps.
- ref_dumper at /home/wubu/bytropix/ref_dumper ✓ (make ref_dumper)
- Links libllama.so directly, CUDA-backed
- Dumps 40 per-layer hidden states + logits in one call
- Verified: cos-sim 0.99696 against GPU reference

### Task 1.2: GQA RoPE implementation [P1] ✓
IMRoPE for Qwen3.6 with rope.dimension_sections=[11,11,10,0], theta=10M implemented.
- Applied to Q_norm and K_norm before attention in wubu_gqa_forward (line 1113)
- OpenMP on Q-head loop ✓
- Verified: T=1 cos-sim unchanged (0.99696), T=2 forward passes correctly

### Task 1.3: gen_text pipeline [P2] ✓
- gen_text tool built: ./gen_text [prompt] [max_tokens]
- Uses verified quantized model path (wubu_model_forward_from_embd)
- SSM state carries between decode steps
- Verified multi-token generation: "The capital of France is" → " the city of Paris..."
- Output: ~0.3 tok/s on CPU for 35B model (2.4 tok/s prefill)

### Task 1.4: Fix gen_text multi-token crash [P3] ✓
- Root cause: logits buffer was `vs` floats but forward writes `B*T*vs` for T>1
- Fixed: malloc(n_prompt * vs * sizeof(float))
- Tokenizer itself was fine

## Phase 2: Performance Optimization (May 18)

### Task 2.1: Profile and optimize decode speed [P0] ✓
- PROFILE env var added: per-layer wall-clock timing (PROFILE=1)
- MoE expert loop OpenMP: 3x speedup on MoE (44ms→15ms per layer)
- Embedding file: opened once at start, closed at end (removed fopen/fclose per decode step)
- Total decode speed: **0.3→0.6 tok/s** (2x)
- Breakdown per decode step (T=1, 16 threads):
  - MoE: 15ms avg (was 44ms) — 30% of time
  - SSM: 12ms avg — 25% of time
  - GQA: 15ms avg — 10% of time
  - Output proj: 8ms — 3% of time
  - Norms/overhead: ~33%
- Next targets: SSM quantized_matmul Q8_K pre-quantization, malloc reduction

### Task 2.2: KV cache for GQA decode [P1]
- Add K/V storage between decode steps to avoid full-attention recompute
- Currently each decode step recomputes attention for all positions
- Impact: GQA is only ~10% of time, so max 10% speedup from KV cache alone
- SKIP — low priority given current bottleneck distribution

## Phase 3: SIMD vec_dot — Close Cos-Sim Gap [DONE ✓]

### Task 3.1-3.3: SSE2/SSSE3/SSE4.1 for Q4_K, Q5_K, Q6_K ✓
- Q4_K/Q5_K: `_mm_maddubs_epi16` (SSSE3) unsigned×signed byte → int16 → int32
- Q6_K: `_mm_cvtepi8_epi16` (SSE4.1) signed×signed widen → `_mm_mullo_epi16` → `_mm_madd_epi16`
- Cos-sim improved: 0.9968 → 0.9970
- Auto-selected via #ifdef __SSSE3__/__SSE4_1__ guards

### Task 3.4: IQ2_XXS/IQ3_XXS/IQ4_XS vec_dot SIMD [P2 — SKIP for now]
Complex lookup tables (`iq2xxs_grid`, `ksigns_iq2xs`, `iq3xxs_grid`) make SSE difficult.
These only affect MoE experts (80+37+3 tensors), not the critical path. Low priority.

### Task 3.5: Profile and measure [P0] ✓
- Cos-sim: 0.9970 (up from 0.9968, quantization noise floor)
- Decode: 0.7 tok/s (no improvement — MoE bottleneck, not matmul)
- Conclusion: Q4_K/Q5_K/Q6_K matmuls were not the bottleneck

## Phase 4: KV Cache for GQA Decode [DONE ✓]
- K/V cache buffers per GQA layer [10][4096][512]
- Append-only: K_norm, V computed per-token, appended to cache
- Attention attends to ALL cached positions (correctness fix)
- Impact: decode now attends to all previous tokens
- cos-sim unchanged for T=1 (cache_len=0 path identical)

## Phase 5: Island Boy Batch Decode — Memory Bandwidth Optimization [NEXT — DETAILED]

### 5.0 Key Insight
35B MoE model = 10.7GB weights. DDR5 ~50GB/s → 214ms min per forward.
Current decode 1.5s = 7× memory bandwidth limit = streaming 1 token at a time wastes 85% of bandwidth (overhead + poor cache locality).

### 5.1 Batch-Aware Layer Forwarding

**Core idea**: Process B=4 tokens through each layer in one pass. Layer weights loaded ONCE from DRAM, shared across B tokens.

**Implementation** (`src/wubu_model.c`):
```
gen_text_batch(batch_size=4):
  1. Collect B pending tokens (warmup tokens or speculative draft)
  2. Embed all B tokens into [B, D_MODEL] matrix
  3. For each layer 0..39:
     - rms_norm([B, D_MODEL]) — vectorized, one norm for all B
     - if SSM layer: ssm_forward_batch(hidden, B)
     - if GQA layer: gqa_forward_batch(hidden, B, KV_cache)
     - moe_forward_batch(x, B) — router computed per-token
  4. output_proj_batch(hidden, B) → [B, 1, vs] logits
```

**Changes needed:**
- `wubu_ssm_forward` → `wubu_ssm_forward_batch(x, B, ...)` — inner dims scale by B
- `wubu_gqa_forward` → `wubu_gqa_forward_batch(x, B, ...)` — batch-GQA: each token attends to its own KVs
- `wubu_moe_forward` → `wubu_moe_forward_batch(x, B, ...)` — per-token expert routing, shared weight load
- `rms_norm_forward` → `rms_norm_batch(x, B)` — already vectorizable

**Memory layout for batch**: `[B, D_MODEL]` contiguous rows. Each row = one token's hidden state.

### 5.2 Warmup Phase (5-token startup lag)

**Why 5 tokens?** First 5 decode steps fill KV cache + SSM states. After position 5, only 1 new K/V per step needs compute.

```
Warmup:
  - Position 0..4: standard single-token decode (no batching yet)
  - KV cache appended each step, SSM state updated
  - After position 4: KV cache has 5 entries per GQA layer
  - SSM states fully initialized

Steady state (position 5+):
  - Batch size B=4 from waiting_queue
  - For server: accumulate requests into batch
  - For single-stream: speculative decode → batch verify
```

**Latency vs throughput tradeoff:** 5-token warmup = ~7.5s initial delay. Acceptable for conversational AI where sessions last 100+ tokens.

### 5.3 prefetch_weights — Software Prefetch for Weights

**Problem**: DRAM→L3 cache latency ~100ns. MoE weights are 10.7GB scattered across memory.

**Solution**: Insert `_mm_prefetch(weight_ptr, _MM_HINT_T0)` before each layer's weight access. Layered approach:

```c
// Level 1: Prefetch next layer's weights while computing current layer
static void prefetch_layer(int i) {
    wubu_layer_t *next = &wubu_model.layers[i+1];
    int n = sizeof(wubu_layer_t) / 64;  // cache lines
    for (int j = 0; j < n; j += 4) {     // stride to avoid cache pollution
        _mm_prefetch(&next->attn_q.weight[j*64], _MM_HINT_T0);
    }
}

// Level 2: Prefetch MoE expert weights within layer
// Router computed first → can prefetch only selected 8 experts
void* expert_ptr = get_expert_ptr(layer, expert_id);
for (int j = 0; j < N_CACHE_LINES_EXPERT; j += 4) {
    _mm_prefetch(expert_ptr + j*64, _MM_HINT_T1);
}
```

**Metrics**: Target 15-20% latency reduction from prefetch alone.

### 5.4 OpenMP Task-Based Weight Loading

**Current**: Single-threaded loop over layers. Other 15 threads idle while one loads weights.

**Optimization**:
```c
#pragma omp parallel
{
    #pragma omp single
    {
        for (int i = 0; i < n_layers; i++) {
            #pragma omp task depend(inout:layer[i])
            {
                compute_layer(i);  // B tokens × layer i
            }
            #pragma omp task depend(in:layer[i])
            {
                prefetch_next_layer(i+1);  // overlap load with compute
            }
        }
    }
}
```

**Impact**: Weight loading overlaps with computation. Layer i+1's weights arrive in L3 by the time layer i finishes.

### 5.5 Quantized MoE Batch Forward

**Current**: Each expert matmul in `moe_expert_forward` processes 1 token → 256 FMAs for gate/up, 512 for down.

**Batch approach** (`moe_expert_forward_batch`):
```c
// For 1 token:  matmul(h[1, D_MODEL], W_expert[D_MODEL, D_FF])
// For B tokens: matmul(H[B, D_MODEL], W_expert[D_MODEL, D_FF])
// Same weight read, B× the compute per cache line
// → B× throughput for ~1.1× memory traffic (output write only)

void moe_expert_forward_batch(float *out, float *in, int B,
                              const void *weight, int type) {
    // For each block of weight data:
    for (int b = 0; b < B; b++) {
        float *in_row = in + b * D_MODEL;
        float *out_row = out + b * D_FF;
        // Dequant + dot with in_row[b]
    }
    // Compiler auto-vectorizes inner loop over B
}
```

**Expert selection**: Per-token routing means different tokens may select different experts. Solution:
1. Router selects top-8 experts per token (independently)
2. Group tokens by expert → coinsurance (one expert handles all tokens that selected it)
3. Batch-process each expert group

```c
// Step 1: Route all B tokens
int expert_counts[256] = {0};
int token_experts[B][8];
for (int b = 0; b < B; b++) {
    route_token(in + b * D_MODEL, token_experts[b], expert_counts);
}

// Step 2: For each popular expert, batch its tokens
for (int e = 0; e < 256; e++) {
    if (expert_counts[e] == 0) continue;
    // Collect tokens that selected expert e
    int token_ids[MAX_BATCH];
    for (int b = 0; b < B; b++) {
        if (has_expert(token_experts[b], e))
            token_ids[n++] = b;
    }
    // Batch-process all n tokens through expert e's weights
    moe_expert_forward_batch(scratch, in, n, e_weights, e->type);
}
```

### 5.6 Verifiable Milestones

| Step | Metric | Target |
|------|--------|--------|
| 5.1 B=4 batch forward | tok/s vs single-stream | 2.0× speedup |
| 5.2 5-token warmup accepted | latency vs steady-state t/s | 7.5s → 0.6s/tok |
| 5.3 prefetch weights | per-layer time | -15% |
| 5.4 task-based overlapping | wall-clock per forward | -20% |
| 5.5 MoE expert group batch | expert time per token | -30% |
| **Final Phase 5** | **stable decode** | **1.2-1.5 tok/s** |

---

## Phase 6: MTP Head + Speculative Decode — 2-3× Improvement [AFTER PHASE 5]

### 6.0 Rationale
Phase 5 gets us to ~1.5 tok/s via batching. Phase 6 adds MTP speculative decoding to reach 2-3 tok/s without changing batch size.

### 6.1 MTP Model Loading

**Tensors to read from MTP model variant:**
```
blk.40.attn_norm.weight          [2048]      F32
blk.40.attn_q.weight             [2048,8192] Q5_K
blk.40.attn_k.weight             [2048,512]  Q5_K
blk.40.attn_v.weight             [2048,512]  Q5_K
blk.40.attn_output.weight        [4096,2048] Q5_K
blk.40.attn_q_norm.weight        [256]       F32  (QK)
blk.40.attn_k_norm.weight        [256]       F32  (QK)
blk.40.ffn_gate_inp.weight       [2048,256]  F32  (router)
blk.40.ffn_gate_inp_shexp.weight [2048]      F32
blk.40.ffn_gate_exps.weight      [2048,512,256] IQ2_XXS
blk.40.ffn_gate_shexp.weight     [2048,512]  Q5_K
blk.40.ffn_up_exps.weight        [2048,512,256] IQ2_XXS
blk.40.ffn_up_shexp.weight       [2048,512]  Q5_K
blk.40.ffn_down_exps.weight      [512,2048,256] IQ3_XXS
blk.40.ffn_down_shexp.weight     [512,2048]  Q6_K
blk.40.post_attention_norm.weight [2048]     F32

nextn.hnorm.weight               [2048]      F32
nextn.enorm.weight               [2048]      F32
nextn.eh_proj.weight             [2048,248320] Q4_K  (or Q5_K)
nextn.shared_head_norm.weight    [2048]      F32
```

**Implementation**: Extend `wubu_model_layer_t` with `nextn_hnorm`, `nextn_enorm`, `nextn_eh_proj`, `nextn_shared_head_norm`. Add `mtp_enabled` flag.

### 6.2 Draft Phase — Multi-Token Prediction

```c
// Given hidden state from layer 39 (after rms_norm):
// Forward through blk.40 → get next token
wubu_layer_forward(h_39_hat, &blk_40, &cache_40);
// Apply nextn.hnorm + eh_proj → logits over vocab
rms_norm(h_40, nextn_hnorm, D_MODEL);
quantized_matmul(logits, h_40, nextn_eh_proj, Q4_K);
// Argmax → token t+1
int token1 = argmax(logits, vocab_size);

// Embed token1, forward through blk.40 again
embedding[token1] → h_draft_1
rms_norm(h_draft_1, nextn_hnorm, D_MODEL);
quantized_matmul(logits_draft1, h_draft_1, nextn_eh_proj, Q4_K);
// Argmax → token t+2
int token2 = argmax(logits_draft1, vocab_size);

// Repeat for token3, token4
```

**Draft arm**: blk.40 is 1 GQA+MoE layer (~280MB weights). Each draft step = 1× read → ~15ms. Total: 4 drafts × 15ms = 60ms.

**Draft via full 40L**: Alternative — skip MTP head, run 40-layer autoregressive. 40×280MB = 11.2GB reads. Would take 214ms+. MTP head (blk.40) is 40× faster for drafting.

**Verdict**: Use blk.40 MTP head for drafting. 40L is too expensive for draft phase.

### 6.3 Verification Phase — Batch Accept/Reject

```c
// Pack all drafted tokens into batch
int draft_tokens[4] = {token1, token2, token3, token4};
// Embed all 4
float hidden_batch[4][D_MODEL];
for (int d = 0; d < 4; d++) {
    memcpy(hidden_batch[d], embedding[draft_tokens[d]], D_MODEL * sizeof(float));
}

// Forward through all 40 layers, B=4
for (int l = 0; l < 40; l++) {
    rms_norm_batch(hidden_batch, 4, layers[l].norm);
    if (is_ssm[l]) ssm_forward_batch(hidden_batch, 4, layers[l], &kv_cache[l]);
    else gqa_forward_batch(hidden_batch, 4, layers[l], &kv_cache[l]);
    moe_forward_batch(hidden_batch, 4, layers[l]);
}

// Output projections for each token
float logits[4][vocab_size];
output_proj_batch(logits, hidden_batch, 4);

// Acceptance: compare greedy from MTP head vs full model
// Accept longest prefix where argmax matches
int accept_count = 0;
for (int d = 1; d < 4; d++) {
    int target = argmax(logits[d]);  // full model's preferred continuation
    if (target == draft_tokens[d]) accept_count = d;
    else break;
}
```

**Key math**: If MTP head is 80% accurate at predicting next token → expected accepted tokens = 1 + 0.8 + 0.64 + 0.51 ≈ 2.95 tokens per verification. Verification takes 1× batch forward (~300ms). Total: 60ms (draft) + 300ms (verify) = 360ms for ~3 tokens → **~8 tok/s**.

**Realistic**: MTP head ~60% accurate → 1 + 0.6 + 0.36 + 0.216 ≈ 2.18 tokens per cycle. 360ms / 2.18 ≈ **6 tok/s**.

### 6.4 Acceptance Window & Rollback

**Edge case**: If 0 of 4 drafts accepted → waste 360ms. Probability = (1-0.6)^4 = 2.56%. Tolerable.

**Rollback**: If only 1 draft accepted, process that one token, re-draft from position t+1. KV cache unchanged for accepted prefix, roll back cache to position t+1 for unaccepted tail.

```c
// Rollback KV cache to accepted prefix length
int accept_len = accept_count + 1;  // +1 for the ground-truth token
for (int l = 0; l < 40; l++) {
    kv_cache[l].len = base_len + accept_len;
}
// SSM state may need rollback — more complex
// Strategy: save SSM state before draft, restore on partial accept
```

### 6.5 Verifiable Milestones (Phase 6)

| Step | Metric | Target |
|------|--------|--------|
| 6.1 MTP model loads | tensor count | 753 loaded correctly |
| 6.2 blk.40 forward | cos-sim vs ref | >0.996 |
| 6.3 Draft phase | tokens/second draft | 60ms for 4 drafts |
| 6.4 Verification batch | B=4 full forward | <300ms |
| 6.5 End-to-end spec-decode | tok/s vs vanilla | 2-3× speedup |
| **Final Phase 6** | **steam community conversation** | **2-4 tok/s** |

---

## Phase 7: Hardware Saturation — Maximum CPU Utilization [AFTER PHASE 6]

### 7.0 Goal
Push inference to theoretical memory bandwidth limit:
- DDR5: 2 channels × 5600 MT/s × 64 bits × 2 (dual-rank) ≈ 56.4 GB/s
- 35B model: 10.7GB / 0.0564 GB/ms = 190ms minimum per forward
- Target: **3-5 tok/s** (prefill), **5-8 tok/s** (decode with spec-decode + batching)

### 7.1 Batch-Aware Weight Layout

**Problem**: Current weight layout is tensor-contiguous. All of attn_q before attn_k before attn_v. This requires 3 separate DRAM scans per layer.

**Solution**: Interleave weights per layer for single-scan access.
```
Current layout:
  [attn_q][attn_k][attn_v][attn_output][ffn_gate_inp][ffn_gate_exps][ffn_up_exps][ffn_down_exps][shared_*][norm]

Better layout (layer-local):
  [interleave: attn_q_blk0 | attn_k_blk0 | attn_v_blk0 | attn_q_blk1 | ...]
  → stream entire layer in one pass, not 10 separate scans
```

**Implementation**: Custom GGUF re-writer that reorders layer weights. Or: just change `wubu_model_init` to load in access order. Less DRAM thrashing.

### 7.2 AVX-512 / AMX Path

**Current**: SSE2 (128-bit) fallback. Most modern CPUs have AVX2 (256-bit) or AVX-512 (512-bit).

**Detection**: CPUID feature flags.
```c
if (__builtin_cpu_supports("avx512f")) {
    vec_dot_fn = vec_dot_avx512;
} else if (__builtin_cpu_supports("avx2")) {
    vec_dot_fn = vec_dot_avx256;
} else {
    vec_dot_fn = vec_dot_sse;  // current
}
```

**AVX2 improvements over SSE2**:
- Q4_K: `_mm256_maddubs_epi16` (32 bytes/op vs 16) → 2× throughput
- Q5_K: `_mm256_maddubs_epi16` → 2× throughput
- Q6_K: `_mm256_cvtepi8_epi16` → 2× throughput
- F32 SGEMM: `_mm256_fmadd_ps` → 2× throughput

**AVX-512 improvements**:
- Q4_K: `_mm512_dpbusd_epi32` (64 bytes/op) → 4× vs SSE2
- Q5_K: packed multiply with broadcast → 3-4× vs SSE2
- Q6_K: `_mm512_cvtepi8_epi16` → 4× vs SSE2

**AMX (newer CPUs)**: 1024-bit tile operations. 8× vs SSE2. If CPU supports AMX-TILE/AMX-INT8, 20-30% over AVX-512.

### 7.3 NUMA-Aware Thread Scheduling

**Multi-socket systems**: CPU may have 2 NUMA domains. Cross-socket memory access is ~1.5× slower.

```c
// Pin threads to NUMA-local cores
#pragma omp parallel num_threads(numa_cores[0])
{
    // Each thread processes local memory partition
    int thread_id = omp_get_thread_num();
    // Thread affinity set to core thread_id on NUMA node 0
}

// For 16-core single socket: all threads on same L3 domain
// No NUMA penalty, but L3 cache (16MB) shared
// → L3 partition: each thread uses ~1MB of cache for weight blocks
```

**L3 cache strategy for our model**:
- Shared expert weights: 40 × (Q5_K: ~1.3M + Q6_K: ~0.8M) ≈ 84MB — won't fit in 16MB L3
- Routeable per-layer locality: keep current layer's weights in L3 (~2.8MB/layer)
- Next layer prefetch into L3 (not L1/L2) to avoid evicting current layer's data

```c
// Prefetch into L3 (HINT_T0 = L1+T0, HINT_T1 = L2, HINT_T2 = L3)
_mm_prefetch(next_layer_weights, _MM_HINT_T2);  // L3 only
_mm_prefetch(next_layer_experts, _MM_HINT_T2);   // L3 only
```

### 7.4 Quantized Scatter/Gather for MoE

**Current problem**: IQ2_XXS dequant is C-only, no SIMD. Complex lookup tables (8192-entry grid, 512-entry ksigns). Hard to vectorize.

**Solution**: Lookup-table gather with SIMD.
```c
// IQ2_XXS: 2 bit/weight → 4 weights per byte
// Grid: 8192 entries × 4 bytes = 32KB — fits in L1 (32KB)
static const float iq2xxs_grid[8192][4];  // 32KB table

// AVX2 gather version:
__m256i indices = _mm256_loadu_si256(...);  // 8 indices (32 bytes)
__m256 gathered = _mm256_i32gather_ps(&iq2xxs_grid[0][0], indices, 4);
// → 8 weights in 1 instruction instead of 8 scalar loads
```

**Gather support**: AVX2 (`_mm256_i32gather_ps`) requires Haswell+. All modern CPUs support it.

**For IQ3_XXS**: Grid = 8192 entries × 8 bytes = 64KB. Fits in L2, partially in L1.
```c
// Use _mm256_i32gather_ps with larger grid
__m256 gathered = _mm256_i32gather_ps(&iq3xxs_grid[indices_scsi][0], indices_pos, 8);
```

**For IQ4_XS**: Grid = 8192 entries × 16 bytes = 128KB. L2-cachable (256KB typical).
```c
// _mm256_loadu_si256 for grid entries → register-wide operations
```

### 7.5 Offload Strategy (Optional)

**If GPU available (dGPU)**: Move output projection to GPU.
- output.weight Q4_K: 2048×248320 = 1.9GB → fits in 6GB VRAM
- token_embd.weight Q5_K: 2048×248320 = 2.5GB → fits
- output_norm: trivial
- **No data copy cost**: output proj results directly from GPU memory

**If no GPU**: All on CPU. Phase 7 optimizations target full memory bandwidth utilization.

### 7.6 Thread Pool + Batch Pipeline

**Pipeline stages** (overlap via OpenMP tasks):
```
Thread 0: Load weights for layer i  (DRAM → L3)
Thread 1-14: Process B tokens × layer i (compute)
Thread 15: Prefetch layer i+1 weights (L3)
```

**Steady state pipeline**:
```
Time →          T0    T1    T2    T3
Layer 0:   [load][compute]     [prefetch next]
Layer 1:         [load][compute][prefetch next]
Layer 2:               [load][compute][prefetch next]
Layer 3:                     [load][compute][prefetch next]
```

**Pipeline bubbles**: First layer has no prefetch. Last layer has no load. ~2% overhead.

### 7.7 Final Performance Targets

| Phase | Tokens/sec | Improvement |
|-------|-----------|-------------|
| Current (Phase 4) | 0.7 tok/s | baseline |
| Phase 5 (batch B=4) | 1.2-1.5 tok/s | 1.7-2.1× |
| Phase 6 (MTP spec-decode) | 2-4 tok/s | 1.7-2.7× over P5 |
| Phase 7 (HW saturate) | 3-5 tok/s | 1.5× over P6 |
| **Final target** | **5 tok/s** | **7× improvement** |

### 7.8 User-Facing Outcomes

| Quality | Current | Target |
|---------|---------|--------|
| Chat response 128 tokens | 3 min | 25 sec |
| Reasoning chain 512 tokens | 12 min | 1.7 min |
| Interactive conversation | Unusable | Usable |
| Server deployment | N/A | Feasible (300ms/tok) |
