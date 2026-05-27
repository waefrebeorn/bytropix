# bytropix Plan — May 27, 2026

## Priority: PARITY FIRST, THEN GAINZ

Parity means bytropix output matches llama.cpp output (cos-sim > 0.99 on logits).
Gainz means speed (lower tok/s gap vs llama.cpp).

## PHASE 1: PARITY

| Step | Action | Tools | Cell |
|------|--------|-------|------|
| 1 | Check if llama.cpp dump_ref builds | make dump_ref | — |
| 2 | Get reference logits from llama.cpp | /tmp/dump_ref MODEL | — |
| 3 | Get our logits | DUMP_LOGITS=/tmp/our.bin gen_text_cpu | — |
| 4 | Compare: find where divergence starts | py_compare_logits.py, layer_cos_sim | — |
| 5 | Patch the C code that causes divergence | patch tool | TBD |
| 6 | Verify fix: cos-sim improves | repeat steps 2-4 | — |
| 7 | Run Hermes test suite | tools/test-hermes-integration.sh | — |
| 8 | Push | git push | — |
| 9 | Loop to step 2 | — | — |

## PHASE 2: GAINZ (after parity reached)

- SSM buffer pre-allocation (cell 241)
- MoE shared expert quantize-once (cell 242)
- Attention sparsity (cell 245)
- MoE expert prefetch benchmark (cell 246)

## ACTIVE CELLS

| Cell | Status | Description |
|------|--------|-------------|
| 175-184 | ✅ | Pytest, Hermes integration, logit cache fix, diagnostics |
| 179 | ✅ FIXED | Logit cache causing repetition — disabled |
| 180 | 🟡 | IQ2_M output quality — 0.974 cos-sim vs ref, needs investigation |
| Output proj | ✅ FIXED | Q4_K output projection was producing zeros (GCC -O3 + if(0) wrapper) |
| dump_ref | ✅ FIXED | `llama_model_free` API fix + text prompt tokenization |
| — | ✅ COMPLETE | dump_ref builds, reference logits acquired |
| — | ✅ COMPLETE | Our logits now non-zero, cos-sim=0.974 vs ref |

## NEXT: improve cos-sim from 0.974 to >0.99
- Check if IQ2_M quantization floor limits precision (diagnostics skill table says 2-bit @ 2048-dim produces flat output)
- Use T10/T50 model for debug cycles
- Layer-by-layer cos-sim to find exact divergence point

## BLOCKERS

- `llama_model_load_from_file` runtime error on new llama.cpp API — needs fixing in dump_ref.c
- llama-cli takes 3+ min to load 11GB model on this machine
