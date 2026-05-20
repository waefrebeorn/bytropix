# Goal Mantra — Phase 28 DA: Fix F32 Waste, Verify Correctness

**Target:** 1:1 inference parity with llama.cpp for Qwen3.6-35B-A3B-UD-IQ2_M.
**Current:** GPU_SUPPORT live. SSM GPU path compiles and runs — correctness UNVERIFIED.

## DA Findings Blocking Progress
1. 🔴 **F32 dequant SSM weights waste ~2.2 GB VRAM** — never used, never freed
2. 🔴 **Preill N>1 fallback uses broken column-major kernel** — produces garbage
3. 🔴 **wubu_model_gpu_free() leaks ~5.5 GB GPU memory** — d_attn_qkv_q etc. never freed
4. 🟡 **Phase 26 fused kernels: verified alone, never in full pipeline**
5. 🟡 **gen_text.c hardcoded 1-token prompt** — can't do proper comparison
6. 🟡 **No cos-sim comparison done yet** — GPU SSM path output vs CPU path

## Critical Fix Path
1. [ ] Remove F32 dequant SSM weight upload (save 2.2 GB)
2. [ ] Fix wubu_model_gpu_free() — free all quantized + F32 weights
3. [ ] Fix wubu_model_gpu_ssm_project() — use row_major kernel instead of broken column-major
4. [ ] Fix gen_text.c — accept prompt from stdin/argv for proper testing
5. [ ] Cos-sim: GPU SSM vs CPU SSM at single layer, then full 30 layers
6. [ ] Profile tok/s with GPU SSM vs CPU SSM baseline (~8-9 tok/s CPU)

## Verification Strategy (DA-corrected)
- Compare layer outputs: GPU SSM path vs CPU SSM path via DUMP_LAYER_DIR
- Use gen_text (CPU) as reference, gen_text_gpu with GPU=1 as test
- Fix tokenizer first: verify prompt tokenization matches llama.cpp
- Each layer's cos-sim must be >0.99, final output >0.95
- Profile AFTER correctness verified — meaningless to profile garbage output
