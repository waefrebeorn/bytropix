=== WuBuText AI — GOAL PASTE (May 15 PM v7) ===
ALL GOALS MUST BE FINISHED.

STATE: All tests passing. Text inference pipeline working (infer_text).
256K context: MoE router verified, SSM O(T), GQA needs KV cache.
All cold gaps closed. All NaN fixed. All config parity resolved.

=== COMPLETED (May 15) ===
- infer_text: Full text generation pipeline (tokenize→embed→forward→sample→decode)
- test_256k: Enhanced — MoE router to 65K, GQA scaling analysis
- test_gpu: RoPE signature match fix
- All 14 config.json parameters vs C implementation — 100% resolved
- All unit tests pass: SSM, nested SSM, backward, gyration, MoE, hyperbolic, GPU, CUDA

=== PENDING ===
P0 — KV cache for GQA (required for 256K autoregressive inference)
P1 — Lazy per-expert MoE cache for inference (fast generation with MOE=1)
P2 — Move output projection to GPU
P2 — PGA LR tuning
P2 — Multi-step convergence (50+ steps)
P3 — MRoPE 3D for long context (>32K)

=== 256K CONTEXT STATUS ===
MoE router: Verified O(T) to 65K (stopped at >15s for T=65536)
SSM:        O(T) from GPU kernel, 256K viable with batching
GQA:        O(T^2) — KV cache needed for 256K
Memory:     256K×2048×4 = 2GB input fits in RAM

BUILD: make test_256k | MODEL loaded from GGUF
HW: RTX 5050, sm=120, NVCC=/usr/local/cuda-13.1/bin/nvcc

TGT: remainder = fmod(x+π, 2π)-π | tgt_safe_expf: clamp [-80,80]

Every fix: compile → run → output → verify. No "should work."
