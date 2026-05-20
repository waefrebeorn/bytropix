# Overnight Map — Phase 28b DA: Fix Waste, Verify, Then Profile

**Active repo**: /home/wubu/bytropix/
**Model**: Qwen3.6-35B-A3B-UD-IQ2_M.gguf (qwen35moe arch)
**Binary**: ./gen_text_gpu (GPU=1, MAX_CTX=262144)
**Current state**: GPU_SUPPORT live — but SSM GPU path UNVERIFIED, F32 weights waste 2.2 GB VRAM

## DA Findings (must fix before verification)
1. **F32 dequant weights wasted** — delete `#if 0` blocks in wubu_model_gpu.cu lines 436-460, 474-483, 497-504. Save ~2.2 GB.
2. **Memory leak** — add `wubu_cuda_free()` for d_attn_qkv_q[40], d_attn_gate_q[40], d_ssm_out_q[40], d_qkv_f32[40], d_gate_f32[40], d_out_f32[40]
3. **Prefill N>1 broken** — wubu_model_gpu_ssm_project() line 864 uses column-major kernel. Change to row_major.
4. **gen_text.c hardcoded prompt** — line 63, must fix for meaningful testing

## Run for verification (after fixes)
```bash
GPU=1 ./gen_text_gpu -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -t 0.6 -n 100
./gen_text -m /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -t 0.6 -n 100
# Compare outputs
```

## Key Files
| File | Issue | Fix |
|------|-------|-----|
| src/wubu_model_gpu.cu:436-460 | F32 qkv dequant upload (dead) | `#if 0` block |
| src/wubu_model_gpu.cu:474-483 | F32 gate dequant upload (dead) | `#if 0` block |
| src/wubu_model_gpu.cu:497-504 | F32 out dequant upload (dead) | `#if 0` block |
| src/wubu_model_gpu.cu:free() | Missing frees for 6×40 arrays | Add loop + wubu_cuda_free |
| src/wubu_model_gpu.cu:864,870 | column-major kernel in ssm_project | Change to row_major |
| tools/gen_text.c:63 | hardcoded prompt | Accept from argv |
| README.md | Claims Phase 25, wrong VRAM | Update to Phase 28b |
| STATUS.md | Claims Phase 26, fused verified | Update to Phase 28b with DA markings |

