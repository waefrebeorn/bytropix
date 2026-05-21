# Overnight Map — Phase 28j: DA v12 Complete, MoE/GQA Isolation Next

**Active repo:** /home/wubu/bytropix/
**Main model:** /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf (11.5GB, 40 layers)
**MTP model:** /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (11.9GB, +blk.40 head)
**Vision model:** /mnt/wslg/distro/models/qwen3.6-35b-mmproj-F16.gguf
**Current state:** GPU hidden cos-sim -0.0036 — ALL prior "coherent GPU output" claims DEBUNKED.

## Verifiable Facts (DO NOT RE-DERIVE)
- DA v12 written: C9 debunked (GPU output was garbage, not coherent)
- gen_text_cpu works correctly ✅
- gen_text_gpu produces garbage with ANY GPU acceleration (GQA or MoE)
- GPU MoE active for both prefill AND decode (no env guard)
- GPU GQA active for prefill only (N>1), CPU for decode
- 1-token test also garbage → MoE primarily suspected (GQA is CPU for 1-token decode)
- gen_text_mtp binary NOT compiled yet (make gen_text_mtp target exists)
- All 9 commits pushed to origin/master (8ef1ba3)
- 25+ DeepSeek papers in .hermes/vault/deepseek-collection/
- Vision encoder: 384 LoC, untested

## Workstreams (pick one)
**A [P0] Isolate hidden state corruption** — remove `layer->moe.gpu_ctx` in wubu_model.c:636-639 → rebuild → DUMP_HIDDEN test. If fixed, MoE kernel is buggy. If not, GQA also buggy.

**B [P1] MTP binary** — after GPU fix, build gen_text_mtp + test with MTP model. Acceptance rate 83% at 2 drafts.

**C [P2] Vision** — build test_vision_real, verify encoder, wire multi-modal.

## Data You Should Not Re-Derive
- Q5_K denormal fix: `d*sc*(v6-32)` not `d*sc*v6 - 32` (bf573b8)
- GPU output proj: [V][D] with CUBLAS_OP_T ld=D (08f5f23)
- SSM state sync: CPU→GPU after hybrid prefill, GPU→CPU after forward_full (08f5f23)
- MoE dequant dispatch: IQ2_XXS=66B/block, IQ3_XXS=98B, IQ4_XS=136B (9093c61)
- MTP model path: /models/Qwen3.6-35B-A3B-MTP-UD-IQ2_M.gguf (~300MB larger = MTP head)

## Vault Gaps (18 vaults checked, 13 have missing roadmap items)
See plan.md for full P3-P6 breakdown. Key gaps:
- hamilton/: quaternion attention + 10× KV cache compression (P2)
- attention/: WuBuSparseAttention, Topological Sequence Model, Entropix (P3)
- tailslayer/: N-way hedged speculative decode (P1)
- optimizers/: Q-Controller + PID Lambda for training (P4)
- lean-proofs/: formal verification not on roadmap (P6)
- phase3/+diffusion/+audio/: text-to-image, video, audio (P5 future)
- encoders/: geometric + topological autoencoders (P5)
