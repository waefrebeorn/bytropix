# Walkway — Context Growth Penalty Fix Path

## THE PROBLEM (RE-DIAGNOSED May 27)
Decode speed drops 50% as context grows from short (<1K) to medium (~2K):
- Turn 2 (short KV): ~1.2 tok/s
- Turn 3 (growing KV): ~0.6 tok/s

**Original diagnosis was wrong.** Root cause is NOT GQA attention O(n²). Actual per-decode breakdown:
- Output projection [2048×248320 Q4_K]: **245ms (43.5%)** — context-independent
- MoE: ~144ms (25.6%) — context-independent
- SSM: ~130ms (23.1%) — context-independent  
- GQA attn: ~43ms (7.7%) — grows only 15% from 2→200 KV

The "50% decay" in multi-turn conversations is from **process-per-turn architecture** in serve_local.py — each turn spawns new gen_text_cpu and re-prefills full context. Not from per-token decode slowdown.

## THE PATH — Revised

### Step 1: Measure Exactly ✅ DONE
Profiled at 2, 50, 100, 200 KV positions. GQA grows 37.7→43.5ms (15%). Output proj dominates at 245ms fixed. See `vault/real-bottleneck-analysis.md`.

### Step 2: Fix Path
| Option | Effort | Impact | Risk | Status |
|--------|:------:|:------:|:----:|:------:|
| A: Lower SPARSE_MIN to 512 | 15 min | ~5% (GQA only 7.7%) | Low | ✅ DONE |
| B: Q4_0 KV benchmark | 2-4 hr | Marginal | Medium | ⬜ |
| C: OMP thread scaling | 1-2 hr | ~5% | Low | ⬜ |
| **D: Persistent KV process** | **8-16 hr** | **5-10× multi-turn** | **Medium** | **⬜ REAL FIX** |
| **E: Output proj chunked** | **4-8 hr** | **1.5-2× decode** | **Medium** | **⬜** |
| F: Logit cache (N-hop reuse) | 2-4 hr | 1.3-1.5× decode | Low | ⚠️ Verify 100% miss |

### Step 3: Implement
Per chosen option. Expected: sparse attention at 512+ tokens gives ~3x speedup.

### Step 4: Verify
```bash
# Short context decode speed
bash tools/test-hermes-integration.sh 8005
# Multi-turn conversation throughput
# Run 3-turn conversation, measure tok/s per turn
```

### Step 5: Document & Loop
Update state, battleship, plan. Push. Loop on remaining bottleneck.

## REFERENCE
- `vault/context-growth-penalty.md` — full analysis
- `state.md` — Test Harness section (baseline measurements)
- `plan.md` — Phase 5
- `battleship.md` — Row K (Context Growth Cells)
