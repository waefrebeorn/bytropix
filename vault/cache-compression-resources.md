# KV Cache Compression Resources — bytropix Research References

**Three approaches + one legacy attention mechanism for fitting 256k+ context on laptop GPUs.**

---

## 1. Q4_0 — Implemented (Phase 22) ✅

**Status:** ✅ CPU path working, cos-sim 0.9994. 💤 GPU path pending.

**File:** `include/wubu_model.h` — `KV_CACHE_Q4_0` mode

**Block format:** `block_q4_0_cache {uint16_t d, uint8_t qs[16]}` — 32 elements per block.

**Compression:** 4:1 vs F16 (0.56 bytes/element). At 256k: **720 MB vs 2.56 GB**.

**Blind spot:** CPU path only. GPU uses separate FP16 growable cache (5.12 GB). Q4_0 on GPU would save ~3.7 GB VRAM.

**Example — quantize 32 floats to Q4_0 block:**
```c
typedef struct { uint16_t d; uint8_t qs[16]; } block_q4_0_cache;

void quantize_q4_0_block(const float *x, block_q4_0_cache *b) {
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) { float ax = fabsf(x[i]); if (ax > amax) amax = ax; }
    if (amax == 0.0f) { b->d = 0; memset(b->qs, 0, 16); return; }
    const float d = amax / 7.0f;
    const float id = 1.0f / d;
    b->d = fp32_to_fp16(d);
    for (int i = 0; i < 32; i++) {
        int q = (int)(x[i] * id + 8.0f);
        if (q < 0) q = 0; if (q > 15) q = 15;
        b->qs[i / 2] |= (uint8_t)(q << (4 * (i % 2)));
    }
}

void dequantize_q4_0_block(const block_q4_0_cache *b, float *x) {
    const float d = fp16_to_fp32(b->d);
    for (int i = 0; i < 32; i++) {
        int q = (b->qs[i / 2] >> (4 * (i % 2))) & 0xF;
        x[i] = ((float)q - 8.0f) * d;
    }
}
```

**K cache read (256-element head from Q4_0 cache):**
```c
void kv_cache_read_q4_0_head(const void *cache, int64_t offset, float *buf, int n) {
    const int block_n = 32;
    int start_block = (int)(offset / block_n);
    int start_elem = (int)(offset % block_n);
    const block_q4_0_cache *blocks = (const block_q4_0_cache *)cache;
    int done = 0;
    while (done < n) {
        float tmp[32];
        dequantize_q4_0_block(&blocks[start_block + (start_elem + done) / 32], tmp);
        int blk_off = (start_elem + done) % 32;
        int to_copy = n - done;
        if (to_copy > 32 - blk_off) to_copy = 32 - blk_off;
        for (int i = 0; i < to_copy; i++) buf[done + i] = tmp[blk_off + i];
        done += to_copy;
    }
}
```

---

## 2. TurboQuant+ — External Reference

**Repo:** `llama/turboquant_plus/` (shallow clone)
**Paper:** Google ICLR 2026 — PolarQuant + Walsh-Hadamard rotation

**Key ideas:**
- **PolarQuant:** Rotates K/V vectors into a polar coordinate system before quantization. The angle component carries more information than the magnitude, so it gets more bits.
- **Walsh-Hadamard Transform (WHT):** Applies a fast orthogonal transform to spread information across dimensions before quantization. Reduces outlier effects.
- **Asymmetric K/V:** Compress V harder than K (V has less impact on attention scores).
- **Sparse V:** Skip zero-valued V cache entries (common after activation functions).
- **Boundary V:** Layer-aware compression — deeper layers can tolerate more compression.

**Claimed compression:** 3.8-6.4× with near-q8_0 perplexity, zero speed penalty on GPU.

**Relevance to bytropix:** TurboQuant+'s WHT + PolarQuant approach could complement our Q4_0 cache. The asymmetric K/V would let us compress V to 2 bits while keeping K at 4 bits.

**Potential integration path:**
```c
// Hypothetical: TurboQuant-style K/V cache
// K cache at Q4_0 (same as Phase 22)
// V cache at Q2_0 (2-bit, higher compression)
// Apply Walsh-Hadamard transform before quantizing V
// WHT is O(n log n) — fast enough for per-token cache write
```

---

## 3. RotorQuant — External Reference

**Repo:** `llama/rotorquant/` (shallow clone)
**Paper:** Scrya Research — Block-diagonal Clifford rotors

**Key ideas:**
- **PlanarQuant (2D Givens):** Replaces WHT with 2×2 Givens rotation matrices applied as block-diagonal operators. Only 2 rotation parameters per pair instead of 64 per WHT row.
- **IsoQuant (4D Quaternion):** Extends to 4×4 quaternion rotations (Clifford rotors). 4 rotation parameters per block.
- **Block-diagonal structure:** Blocks of size 2 or 4, independent rotations per block. 44× fewer parameters than WHT.
- **Deferred quantization:** Keep F16 during prefill (no error compounding), quantize post-prefill.

**Claimed vs TurboQuant+:** -0.005 PPL, 28% faster decode, 5× faster prefill, 44× fewer rotation params.

**Relevance to bytropix:** RotorQuant's block-diagonal approach is simpler to implement (just 2×2 or 4×4 rotations) than full WHT. The deferred quantization pattern fits our architecture (prefill is separate from decode).

**Potential integration path:**
```c
// Hypothetical: RotorQuant-style rotation + Q4_0 quantization
// Before quantizing K/V to Q4_0, rotate with block-diagonal Givens
// Block size 2: cos/sin per adjacent pair
// After rotation, standard Q4_0 block quantize
// On read: dequant Q4_0, inverse rotate
```

---

## 4. Hamilton Encoder Attention — Legacy Concept (HASHMIND Project)

**Location:** `/mnt/c/projects/HASHMIND/llama-cpp-rotorquant/llama.cppCOPY/`
**Status:** Experimental. MLP encoder + BSP tree. Stale-range bug at 4096 recall window.

**Key idea:** Compress V cache into a **5-dimensional quaternion manifold** via a 3-layer MLP encoder. Partition the manifold with a Binary Space Partition (BSP) tree for O(log N) retrieval. At decode time, only tokens in the best-matching cell are attended — subset of the full KV cache.

```
Raw KV → MLP Encoder → [5, H, W] quaternion grid (711:1 compression)
         → BSP Tree → Subset Attention (O(N log N) instead of O(N²))
```

**Why it matters for bytropix:** This is an orthogonal approach to cache compression. Instead of quantizing individual values (Q4_0, TurboQuant+, RotorQuant), it compresses the ENTIRE cache into a learned representation. The BSP occlusion tree enables O(log N) attention at 256k+.

**The 4096 recall window bug:** The original implementation had a stale-range bug where tokens beyond 4096 positions were not correctly indexed in the BSP tree. The root cause was that grid cell indices were computed from token count modulo 4096 instead of absolute position. Fix: use actual token count for cell index calculation.

**Integration idea for bytropix:**
```c
// Hypothetical: Hamilton encoder as V cache compression layer
// 1. After each prefill chunk, run MLP encoder on V cache
// 2. Store quaternion grid in GPU persistent buffer
// 3. At decode time, project query into quaternion space
// 4. BSP tree returns subset of V tokens to attend
// 5. Only dequantize Q4_0 V cache entries for selected tokens
```

---

## Comparison

| Approach | Compression | Quality | Compute Overhead | Integration Complexity |
|----------|:-----------:|:-------:|:----------------:|:---------------------:|
| **Q4_0 (Phase 22)** | 4:1 | ✅ 0.9994 cos-sim | ~0 (inline dequant) | Low — header-only |
| **TurboQuant+** | 3.8-6.4:1 | Near q8_0 | WHT O(n log n) per write | Medium — needs WHT kernel |
| **RotorQuant** | 4-6:1 | Near q8_0 | 2×2 Givens per write | Low — 4 mults per pair |
| **Hamilton Encoder** | 711:1 (V only) | Unknown | MLP encode O(N) per chunk | High — MLP + BSP tree |

## Next Steps for bytropix

1. **Short-term:** Port Q4_0 to GPU path (same block format, CUDA dequant in attention kernel)
2. **Medium-term:** Implement RotorQuant's Givens rotation before Q4_0 quantize (block-diagonal, 2 mults per pair)
3. **Long-term:** Evaluate Hamilton encoder BSP attention for >512k context
