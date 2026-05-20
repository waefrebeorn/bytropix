# Vault — Verification & Legacy Tools

These files document critical verification results from the bytropix GPU acceleration project.

## GPU Kernel Verification Files

| File | What It Tests | Result | Date |
|------|--------------|--------|------|
| `test_ssm_rec_gpu.cu` | SSM selective scan recurrence on GPU | cos-sim=1.0 vs CPU, max err 1e-6 | May 19 |
| `test_gpu_vs_f32.cu` | GPU Q5_K matmul vs F32 dequant ref | cos-sim=1.0 | May 19 |
| `test_moe_gpu.cu` | GPU IQ2_XXS expert kernel | Non-zero output, no CUDA errors | May 19 |

## Important Verification Technique: Stale Binary Detection

Multiple bugs traced to stale object files. Always `touch` modified sources and
verify with a clean rebuild when behavior changes unexpectedly.

## Verification Protocol

For every GPU kernel: read → compile → run → verify → compare → fix → reverify.
"Compiles" ≠ "works." REAL verification is cos-sim vs a known-correct reference
(CPU implementation, llama.cpp output, or numpy reference).

## GPU Quantization Type Note

GGML type enums differ between codebases:
- Q4_K = 12 in bytropix `gguf_reader.h`
- Q4_K = 15 in llama.cpp `ggml.h`
- Always verify tensor type against local header, not documentation
