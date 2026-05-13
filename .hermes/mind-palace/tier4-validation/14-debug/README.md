# Debugging — Known Issues and Workflows

## RESOLVED Issues

These were identified in earlier planning but are now verified/corrected:

1. ~~GGUF IQ2_M quantization — embeddings are 2-bit~~ → **RESOLVED:** `token_embd.weight` is actually Q5_K (5-bit), not IQ2_M. Different tensors use different quantization types in the same GGUF.

2. ~~Poincaré boundary — ||x|| near 1~~ → **RESOLVED:** Our embedding norms are 0.30-0.34 after mapping (R=0.956). Maximum is 0.52, well below 1.0. Boundary instability won't trigger during normal forward passes.

3. ~~No BBPE tokenizer~~ → **NOT YET RESOLVED.** Still need to implement Qwen's BBPE tokenizer. This is the #1 blocker for Phase 3.

## CURRENT Known Issues

### 1. No BBPE Tokenizer (🔴 BLOCKER)
The C code has a 97-token ASCII tokenizer from the baseline. Qwen3.6 uses GPT-2 BPE with 248K tokens.
Can't use extracted embeddings without matching tokenizer.
- **Fix:** Implement GPT-2 BPE in C using the merge rules from GGUF.
- **Workaround:** Use Python subprocess for tokenization (llama.cpp's Python bindings or Hugging Face).

### 2. attn_qkv Split Unknown (🔴 BLOCKER for Phase 2)
The shape [2048, 8192] for `attn_qkv.weight` doesn't cleanly split into Q_full + K_full + V_full + Q_linear + V_linear.
The expected total (5120 + 2048 + 4096 = 11264) exceeds 8192, meaning some projections are separate tensors.
- **Fix:** Read `llama.cpp/ggml/src/ggml-ssm.h` and `llama.cpp/src/models/qwen35.cpp` to find the exact split.

### 3. SSM vs DeltaNet Implementation Unknown (🔴 BLOCKER for Phase 2)
The "Gated DeltaNet" in the model card may differ from the actual SSM implementation.
The `ssm_a`, `ssm_dt`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d` tensors suggest Mamba2-style selective scan.
- **Fix:** Study llama.cpp's SSM integration (`ggml-ssm.h`) for the exact recurrence formula.

### 4. VRAM Limit — 6.4GB shared with OS (🟡 WARNING)
3B active params requires ~6GB in f16. With KV cache (670MB) and activations (~2.7GB), we exceed 6.4GB.
- **Fix:** Load weights in Q5_K/Q8_K format (dequant to f16 on-the-fly).
  Offload optimizer states to CPU (AdamW × 2 = 24GB — must be CPU).
  Use gradient checkpointing.

### 5. MoE Weight Quantization (🟡 WARNING)
Expert weights are IQ2_XS (2-bit) and IQ1_S (1-bit). Poor quantization quality may affect
the hyperbolic router training (routing scores computed from highly quantized features).
- **Fix:** For active experts, keep f16 cache; dequant only when an expert is routed to.

### 6. 73 Zero-Norm Embeddings (🟢 LOW)
73 of 248320 embeddings are all-zeros (special/padded tokens). These map to origin in Poincaré ball.
During training, if these tokens are sampled, their gradient through exp_map is zero.
- **Fix:** In data pipeline, ensure zero-embedding tokens are either filtered or have
  a minimum norm (add small noise to prevent dead neurons).

## Debugging Workflows

### Embedding Extraction (Phase 1 — DONE)
```
./tools/extract_and_map               # Extracts + maps to Poincaré
# Verify: compare hash against Python extractor
sha256sum data/qwen36_embeddings_c.bin
# Check 73 zero-norm tokens are at correct indices
python3 -c "import numpy as np; e=np.memmap('data/qwen36_embeddings_c.bin',np.float32,'r',(248320,2048)); print(np.where(np.linalg.norm(e,axis=1)<1e-6))"
```

### Forward Pass Debug (Phase 2)
```
# Test DeltaNet forward against llama.cpp output
export LD_LIBRARY_PATH=/path/to/llama.cpp/build
gcc -O2 -I include -o test_deltanet test_deltanet_forward.c src/*.c -lm -L/path/to/llama -lggml
./test_deltanet --gguf /mnt/wslg/distro/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
                --input "test_vectors.bin" \
                --output "output_reference.bin"
# Compare output logits: mean relative diff should be < 0.01 (Q5_K quantization noise)
python3 -c "
import numpy as np
a = np.fromfile('output_wubu.bin',np.float32)
b = np.fromfile('output_reference.bin',np.float32)
print(f'Mean diff: {np.abs(a-b).mean():.6f}')
print(f'Max diff: {np.abs(a-b).max():.6f}')
"
```

### Training Crash
```
# Step 1: Check for NaN
./train --check-nan --max-steps 10
# If NaN at first backward: check exp_map derivatives (gradient explosion through tanh)
# If NaN after N steps: learning rate too high, or routing collapse
#
# Step 2: Reduce learning rate
./train --lr 1e-5
# Step 3: Disable hyperbolic (use pure Euclidean baseline)
./train --no-hyperbolic
# If Euclidean works but hyperbolic doesn't: bug in RSGD implementation
```

### CUDA Out of Memory
```
# Reduce batch size
./train --batch-size 1 --grad-accum 8
# Reduce context length
./train --context 2048
# Reduce model precision
./train --dtype float16
# Use gradient checkpointing (trades compute for memory)
./train --checkpoint-activations
```

## FAIL_LOG

| Date | Issue | Resolution |
|------|-------|------------|
| 2026-05-12 | GGUF tokenizer arrays OOM during metadata parsing | Added skip_large=True to read_value | 
| 2026-05-12 | GGUF KV parsing went out of sync | Rewrote with explicit byte counting per KV pair type |
| 2026-05-12 | Q5_K vs Q8_K type misidentification | dump_gguf.py had wrong GGML_TYPE dict (12→Q8_K but correct is 12→Q4_K) |
| 2026-05-12 | norm range 0.0-0.547 (Python) vs 0.0-0.547 (C) | Both match — C implementation verified correct |
