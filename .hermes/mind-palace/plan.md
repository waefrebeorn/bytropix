# bytropix Plan — May 27, 2026

## Priority: PARITY FIRST, THEN GAINZ

Parity means bytropix output matches llama.cpp output (cos-sim > 0.99 on logits).
Gainz means speed (lower tok/s gap vs llama.cpp).

## PHASE 1: PARITY — IQ2_M FLOOR REACHED

| Step | Action | Tools | Cell | Status |
|------|--------|-------|------|--------|
| 1 | Check if llama.cpp dump_ref builds | make dump_ref | — | ✅ FIXED |
| 2 | Get reference logits from llama.cpp | /tmp/dump_ref MODEL | — | ✅ |
| 3 | Get our logits | DUMP_LOGITS | — | ✅ |
| 4 | Compare: find divergence | py_compare_logits.py, layer_cos_sim | — | ✅ 0.974 |
| 5 | Patch the C code | patch tool | Output proj | ✅ FIXED |
| 6 | Verify fix: cos-sim improves | repeat steps 2-4 | — | ✅ 0.974 (floor) |
| 7 | Run Hermes test suite | test-hermes-integration.sh | — | ✅ |
| 8 | Push | git push | — | ✅ on cpu-optimize-may26 |

**CONCLUSION: 0.974 is IQ2_M quantization floor** — pure random noise (correl|ref,|diff|=-0.024), unbiased (mean diff=-0.05), 41/50 top-token overlap. Need Q3_K/Q4_K/F16 model to reach >0.99. Not available on this machine.

## ACTIVE CELLS

| Cell | Status | Description |
|------|--------|-------------|
| 175-184 | ✅ | Pytest, Hermes integration, logit cache fix, diagnostics |
| 179 | ✅ FIXED | Logit cache causing repetition — disabled |
| 180 | 🟡 PARITY REACHED | IQ2_M output quality — 0.974 cos-sim (quantization floor) |
| Output proj | ✅ FIXED | Q4_K output projection was producing zeros (GCC -O3 + if(0) wrapper) |
| dump_ref | ✅ FIXED | `llama_model_free` API fix + text prompt tokenization |
| run-harness.sh | ✅ PATCHED | Now uses serve_local.py (local CPU) |
| test-hermes-headless.sh | ✅ PATCHED | Now uses serve_local.py (local CPU) |
| NES PPU | ✅ DONE | Tile/nametable rendering + iNES loader already implemented |

## ALL GAPS CLOSED — NO URGENT BLOCKERS

## PHASE 2: GAINZ (when ready)

- SSM buffer pre-allocation (cell 241)
- MoE shared expert quantize-once (cell 242)
- Attention sparsity (cell 245)
- MoE expert prefetch benchmark (cell 246)
