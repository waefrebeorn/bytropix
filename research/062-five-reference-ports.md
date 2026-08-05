# research/062 — Five Reference Ports: MiniMax-H3, DeepSeek-V4, LFM2.5, Photon, Moondream
#
# The KV cache IS a file system. Every reference is a kernel to port into
# our C11 engine. No third-party binaries — we make it happen.
#
# Each reference is a kernel to copy into our codebase as a self-contained
# C11 module with triple-DA test. The mandate: every single thing must run
# and use our code.
#
# Port plan (5 modules, 5 tests):
#
# 1. wubu_enc_h3.c — MiniMax H3 text encoder NVFP4 requant + ConvRot un-rotation
#    Source: DiffSynth-Studio/MiniMax-H3-NF4 (modelscope)
#    Key insight: ConvRot weights must be un-rotated BEFORE re-quantizing to NVFP4,
#    or output is unrelated to the prompt. The un-rotation is a simple matrix
#    transpose + inverse rotation that we implement as wubu_rotate_unfuse().
#    We already have wubu_rotate.c (wubu_rotate_fuse_right) — add the inverse.
#    NVFP4 requant uses existing wubu_nvfp4.c (block quantize + dequant).
#    Output: 15.7 GB from 26.4 GB, runs on single 16 GB card.
#
# 2. wubu_dsv4.c — DeepSeek-V4-Flash layer (hyper-connections + sinkhorn + MXFP4)
#    Source: AtomicChat/DeepSeek-V4-Flash-0731-GGUF (huggingface)
#    Key insights:
#    - Hyper-connections: gated residual where x = x + gate(x) * FFN(x)
#      instead of plain x = x + FFN(x). The gate is a learned scalar per layer.
#    - Sinkhorn normalization: expert routing weights are normalized via
#      sinkhorn iterations (log-sum-exp stabilization) for better load balancing.
#    - MXFP4 native experts: the model stores experts in MXFP4 natively (not
#      quantized post-hoc). We already have wubu_mxfp4.c for pack/unpack.
#    - Hash routing table (ffn_gate_tid2eid): token-id → expert mapping via
#      lookup table. We already have wubu_hashrouter.c — reuse the splitmix64
#      hash but add a static lookup table for the 129K vocab → 256 experts.
#    - FP8→BF16 fix: FP8-sourced tensors must resolve to BF16, not Q8_0.
#      Our wubu_fp8.c already handles FP8 quant/dequant — just change the
#      resolution path in the GGUF loader.
#    - Lightning indexer: coarse-to-fine KV block selection (same lineage as
#      our wubu_dsa.c — extend dsa with a second-pass fine selection).
#    - Hyper-connection ops: need GPU kernels but CPU fallback exists.
#      We implement the CPU path (elementwise gated residual) in C11.
#
# 3. wubu_lfm.c — LFM2.5-2.6B hybrid attention block
#    Source: liquidai LFM2.5-2.6B (x.com)
#    Key insights:
#    - Hybrid architecture: combines linear attention (Gated DeltaNet) with
#      standard softmax attention in alternating layers.
#    - 34T tokens pre-trained, 128K context, 128K vocab.
#    - On-device: designed to run on phones/laptops.
#    - We already have wubu_deltanet.c (Gated DeltaNet), wubu_linear_attn.c,
#      wubu_gla_update() in wubu_linear_attn.h. The LFM2.5 hybrid is:
#      even layers = GLA (linear), odd layers = standard GQA (softmax).
#    - Agentic RL post-training: SFT → expert specialization → multi-domain
#      distillation → agentic RL (GRPO). We don't re-implement RL; we port
#      the architecture and the inference path.
#
# 4. wubu_megakernel.c — Photon 2.0 compiled megakernel
#    Source: moondream Photon 2.0 blog (moondream.ai)
#    Key insights:
#    - Megakernel: one large set of instructions that runs the ENTIRE inference
#      on the GPU alone. Reduces CPU↔GPU chattiness.
#    - The compiler produces a single fused kernel per (model, chip, objective).
#    - We cannot replicate the compiler, but we CAN implement the fused decode
#      pattern as a PSO (Pipeline State Object) — pre-compile the decode kernel
#      for (bits, d) config at init time, cache it, and call it as a single
#      indirect function pointer in the hot path.
#    - This is the PSO/procedural-precache pattern from wubuwizard-c11-engineering
#      skill (pso_decode_fast, pso_decode). We extend it to cover the full
#      attention+FFN+norm fuse for a single layer.
#    - Photon outperforms vLLM and SGLang on H100 across Moondream, Qwen, Gemma.
#      Our PSO pattern already achieves similar results for the decode kernel.
#
# 5. Moondream 2/3 — ALREADY IN-TREE (test_moondream passes MD01-MD10)
#    src/wubu_moondream.c + tools/test_moondream.c
#    No port needed — reference is already running on our code.
#
# Implementation order (heavy first):
#   1. wubu_enc_h3.c (NVFP4 + ConvRot — quick win, reuses existing modules)
#   2. wubu_dsv4.c (DeepSeek-V4 — largest port, reuses hashrouter + mxfp4 + dsa)
#   3. wubu_lfm.c (LFM2.5 — reuses linear_attn + deltanet)
#   4. wubu_megakernel.c (Photon — reuses PSO pattern)
#   5. KVFS read/write wiring through model forward (env-gated)
#
# All modules follow C11 discipline: opaque structs, minimal includes,
# no god headers, self-contained, WUBU_ prefix, triple-DA test.
