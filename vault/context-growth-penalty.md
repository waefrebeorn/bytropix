# Context Growth Penalty — Analysis

## The Problem
Decode speed drops 50% as context grows from <1K to ~2K tokens during multi-turn conversation.

### Measured Data (3-turn NES Q&A on i5-8365U)
| Turn | Context Size | Decode Speed | Notes |
|------|:------------:|:------------:|-------|
| 1 | ~50 tok (just prompt) | ~1.0 tok/s | Cold start, includes model load |
| 2 | ~500 tok (1 prior turn) | ~1.2 tok/s | Warm, short KV cache |
| 3 | ~2000 tok (2 prior turns) | ~0.6 tok/s | Growing KV cache, dense attn O(n²) |

## Root Cause Analysis

### Candidate: GQA Dense Attention O(n²)
The 10 GQA layers (layers 3,7,11,15,19,23,27,31,35,39) compute dense attention over all KV positions:
- At 50 tokens: 10 × 50 × 2048 = ~1M ops — negligible
- At 500 tokens: 10 × 500 × 2048 = ~10M ops — visible
- At 2000 tokens: 10 × 2000 × 2048 = ~40M ops — dominant
- At 4096 tokens: 10 × 4096 × 2048 = ~84M ops — sparse activates here

### Why Sparse Isn't Active Early
`SPARSE_MIN=4096` (default in wubu_ssm.h). Below this threshold, dense attention is used regardless of the `USE_SPARSE_ATTN=1` env var.

### Other Candidates
1. **SSM recurrence** — 30 SSM layers have O(T×D) recurrence, not O(n²). Should scale linearly.
2. **MoE expert computation** — Fixed 8 active experts per layer regardless of context. Not context-dependent.
3. **Output projection** — Fixed [2048 × 248320] matmul. Context-independent.

## Fix Options

### Option A: Lower SPARSE_MIN (Easiest)
Change `#define SPARSE_MIN 4096` to `#define SPARSE_MIN 512` or make it env-var controlled.
- **Impact**: Sparse attention at 512+ tokens. Sparse GQA uses local window (512) + global positions (128), reducing O(n) from 2000 to 640.
- **Risk**: Quality loss if important tokens are outside the sparse window. SSM already handles long-range, so GQA sparse window may be OK.
- **Effort**: 15 minutes

### Option B: Q4_0 KV Cache for Decode (Already Wired)
Cell 244 implemented Q4_0 KV cache format (4:1 compression vs F16). Not benchmarked vs F32 for decode speed.
- **Impact**: Less memory bandwidth per attention operation.
- **Risk**: Quantization loss from KV cache.
- **Effort**: 2-4 hours (benchmark, compare quality)

### Option C: OMP Thread Scaling
Fewer OMP threads at short context (less parallelism needed) → less thread contention.
- **Impact**: Marginal. OMP overhead vs benefit at small n.
- **Effort**: 1-2 hours

### Option D: Combine All Three
Lower SPARSE_MIN + benchmark Q4_0 KV + tune OMP. Expected: >1.0 tok/s at 2K context.
- **Effort**: 4-8 hours
- **Expected result**: No decay from turn 2→3. Maintain >0.8 tok/s across all context lengths.

## Verification
After fix: PROFILE at 5 context lengths (50, 256, 512, 1024, 2048 tok). Report tok/s curve. Run 3-turn conversation, measure per-turn tok/s.

## Related Files
- `src/wubu_ssm.c` — GQA attention forward (dense + sparse paths)
- `include/wubu_model.h` — SPARSE_MIN, SPARSE_W, SPARSE_G defines
- `src/wubu_model.c` — GQA cache management, env var parsing
- `tools/benchmark-context.sh` — TODO: create benchmark script
