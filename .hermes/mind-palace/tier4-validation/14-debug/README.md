# Debugging — Known Issues and Workflows (May 15 PM)

## RESOLVED Issues (post-sprint)
1. GPU weight loading → Q4_K dequant bypass in unbuffered reads — fixed
2. Training pipeline CE=69 → CE=12.42 after fix
3. train_backprop hang → was just CPU slow (25s/step), not hanging
4. BBPE tokenizer → not needed as blocker (data pipeline works)
5. SSM implementation → read from llama.cpp qwen3next.cpp, verified
6. MoE quant → lazy dequant works, 9× speedup

## CURRENT Known Issues

### 🔴 P0: GPU Vision Pipeline Timed Out
`infer_vision_text_gpu` timed out at 120s. CPU vision is 74s for 256×256.
- Check: CUDA kernel launch in cuda_vision.cu, memory allocation, or model load
- Fix: test with `cuda-gdb` or add MARKers

### 🟡 P0: ~0.5% NaN in Model Logits
Persists across all input sources. Root cause unknown.
Masked by NaN→0 guards. Suspect MoE dequant or SSM recurrence.
- Can't be the old Q4_K dequant bug (that was fixed)
- Check: any_ssm_zero_state × exp(any_finite) at state[0,0]
- Check: MoE gate sigmoid overflow with large logits

### 🟡 P1: CPU RMSNorm Dim Mismatch
CPU GQA RMSNorm uses d=4096 with weight[256]. OOB read for head i>=256.
GPU uses per-head RMSNorm correctly (16 groups of 256).
- Fix: broadcast weight[256] across 16 heads instead of using d=4096
- Impact: CPU-only path affected (GPU not affected)

### 🟡 P1: All Math Extensions Standalone
RSGD, Poincaré GQA, Nested SSM, TST, Nested MoE, CUDA kernels, data pipeline — all pass tests but NOT wired to train_gpu or wubu_model_forward.

## Debugging Workflows
```bash
# Forward pass check
./test_model --layers 1 --tokens 8

# GPU weight match
./test_gpu

# NaN trace
CUDA_LAUNCH_BLOCKING=1 ./train_gpu --check-nan --max-steps 10

# CUDA OOM
./train_gpu --batch-size 1 --context 2048

# Vision pipeline
./infer_vision_text --image /path/to/test.png

# CUDA kernel tests
./test_cuda_kernels
```
