# Hamilton Encoder CUDA

CUDA kernel for encoding KV cache entries into quaternion representations.

The kernel performs:
1. 2×2 average pooling downsampling (stride 2)
2. RGB → HSL color space conversion
3. HSL → normalized quaternion (w, x, y, z) + amplitude (5-channel F32 output)

**Status**: Implementation in `llama-cpp-rotorquant` fork (`ggml/src/ggml-cuda/hamilton-encoder.cu`, ~137 lines). See `LLAMA-CPP-INTEGRATION/README.md` for full pipeline documentation.
