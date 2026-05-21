# Vault — Reference & Debug Tools (May 21, 2026)

## Purpose
These tools generate reference inference data from llama.cpp's libllama.so for 1:1 parity comparison against bytropix.

## Reference Data Generation Flow

### 1. Per-Layer Hidden States
```bash
# Generate reference layer-by-layer dumps (1 BOS token)
DUMP_LAYER_DIR=/tmp/ref_layers ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
# Output: /tmp/ref_layers/ref_layer_0..39.bin (each 8192 bytes = 2048 floats)

# Generate our dumps
DUMP_LAYER_DIR=/tmp/our_layers ./gen_text "prompt" 0
# Output: /tmp/our_layers/ref_layer_0..39.bin

# Compare
./layer_cos_sim /tmp/ref_layers /tmp/our_layers 40
```

### 2. Per-Layer Q/K/V/Attention Intermediates
```bash
# Dumps ALL intermediate tensors: Qcur, Kcur, Vcur, beta, alpha, gate, attn scores, etc.
# 1997 files for 1 BOS token (~9MB total)
DUMP_INTERMEDIATE_DIR=/tmp/ref_interm ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
# Files: L0_Qcur_full.bin, L0_Qcur_normed.bin, L0_Kcur.bin, L0_attn_gated.bin, etc.
#         L0_ffn_moe_logits.bin, L0_ffn_moe_probs.bin, L0_ffn_moe_weights.bin, etc.
```

### 3. Final Logits
```bash
REF_LOGITS_PATH=/tmp/ref_logits.bin ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
REF_HIDDEN_PATH=/tmp/ref_hidden.bin ./ref_dumper /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
./compare_logits /tmp/ref_logits.bin /tmp/our_logits.bin
```

### 4. MTP Model Reference
```bash
DUMP_LAYER_DIR=/tmp/mtp_ref_layers ./ref_dumper_mtp /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf
```

### 5. Per-Expert MoE Comparison
```bash
GPU=1 ./compare_moe_expert    # Compare GPU vs CPU per-expert output
```

## Key Files

| File | Purpose | Lines |
|------|---------|-------|
| ref_dumper.cpp | Reference dumper (libllama.so) | 216 |
| ref_dumper_mtp.cpp | MTP reference dumper | ~300 |
| layer_cos_sim.c | Per-layer cos-sim comparison | ~100 |
| compare_moe_expert.c | GPU vs CPU per-expert cos-sim | ~200 |
| compare_logits.c | Final logit comparison | ~80 |
| dump_intermediates.c | Debug intermediate dumper | ~150 |
| dump_hidden.c | Hidden state dumper | ~100 |
| dump_layers.c | Layer dump tool | ~120 |
| test_moe_layer.c | Single MoE layer test | ~180 |
| test_vision_real.c | Vision pipeline test | ~200 |

## Env Variables Supported by llama.cpp Context

| Var | Effect |
|-----|--------|
| `DUMP_LAYER_DIR=path` | Dump ref_layer_<i>.bin for each layer |
| `DUMP_INTERMEDIATE_DIR=path` | Dump L<il>_<name>.bin for all intermediate tensors |
| `REF_LOGITS_PATH=path` | Save final logits to path |
| `REF_HIDDEN_PATH=path` | Save final hidden state to path |
| `LLAMA_N_THREADS=N` | Thread count (default: 16) |
