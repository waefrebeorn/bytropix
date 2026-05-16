# state — May 16 yolo session

## Done
- P0-a: Shared expert gate loaded and applied (was NULL)
- P1c: Single-pass O(EK) top-k (was O(EK²))
- P2b: Conv state build + update as device kernels (replaced host loops)
- P2c: Conv1d kernel with shared memory cache for weights
- TF32 math mode on all cuBLAS matmuls
- Block size 256→512 for element-wise CUDA kernels

## Performance
- CPU: prefill 4.2s, decode 1 tok/s (MOE=1, 40L)
- GPU: prefill 2.5s, decode 2.4 tok/s (MOE=1, 40L)
- GPU no-MoE: prefill 0.27s, decode 14 tok/s

## Next up
- P1a: Chunked DeltaNet (3× prefill) — algorithm understood, needs C impl
- P2a: Warp-level CUDA scan — template kernel from ssm-scan.cu
- P3a: On-the-fly IQ2_XXS dequant — keep weights quantized
- P0-b: Verify TF32 vs CPU cos-sim
