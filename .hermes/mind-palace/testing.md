# WuBuText AI — Testing Protocol (May 15 PM v6)

## Primary Test: train_integrated

```bash
# Forward-only (no backward)
./train_integrated /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/train_data.bin 3

# Flags
TST=1 ./train_integrated ...   # Token-superposition training
RSGD=1 ./train_integrated ...  # Riemannian SGD
PGA=1 ./train_integrated ...   # Poincaré GQA
NESTED_SSM=1 ./train_integrated ...  # Nested SSM K=4
NESTED_MOE=1 ./train_integrated ...  # Poincaré MoE router
POINCARE_R=0.956 ./train_integrated ...  # Hyperbolic SSM recurrence
```

Expected: loss ~21→19 after 2-3 steps, 0 NaN, 11-14s/step.

## All Binaries

| Binary | Command | Status |
|--------|---------|--------|
| `train_integrated` | `./train_integrated [model] [corpus] [steps]` | 🟢 11s/step, 0 NaN |
| `train_gpu` | `./train_gpu [model] [corpus] [steps]` | 🟢 CE~12.42 |
| `train_real` | `./train_real [model] [corpus]` | 🟢 CE~12.66 |
| `infer_moe_lazy` | `./infer_moe_lazy [model] [layer] [T]` | 🟢 37 tok/s |
| `infer_unified` | `./infer_unified [model] [T]` | 🟢 40-layer forward |
| `test_kv_cache` | `./test_kv_cache [model]` | 🟢 max_diff=0.00 |
| `infer_vision_gpu` | `./infer_vision_gpu [model] [image]` | 🟢 99ms |
| `infer_poincare` | `./infer_poincare` | 🟢 2835 tok/s |
| `test_moe` | `./test_moe` | 🟢 NaN=0 |
| `bench_e2e` | `PATH=... ./bench_e2e` | 🟢 GPU weights fixed |

## Vault Tests (to add)
- `test_sparse_attn` — Port sparse attention from vault, verify O(n·k) linear complexity
- `test_q_controller` — Port Q-Controller optimizer, verify convergence vs fixed LR
- `test_tailslayer_spec` — Port hedged-read CUDA kernel, verify N-draft verification

## Paper Discrepancy Verification

| Check | How | Status |
|-------|-----|--------|
| head_dim 256 vs 128 | Read `GQA_HEAD_DIM` and `SSM_D_STATE` in headers | ✅ Both correct |
| KV heads=2 | Read `GQA_KV_HEADS` | ✅ Correct |
| MRoPE missing | Code audit — check rope implementation | ⚠️ Verify |
| Conv dim 1536 vs 8192 | Code audit — check CONV_DIM | ⚠️ Investigate |
| RoPE theta=10M | Check constant in wubu_ssm.c | ⚠️ Verify |

## Known Issues

| Issue | Test | Severity |
|-------|------|----------|
| PGA loss jumps 21.6→69 | `PGA=1 ./train_integrated ...` | Lr_gqa too high |
| ~11s/step GPU compute | Baseline forward | RTX 5050 limit |
| CONV_DIM discrepancy | Code audit needed | Possible bug |
