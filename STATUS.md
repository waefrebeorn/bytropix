═══ WUBUTEXT AI -- STATE (May 15 PM v6) ═══
Path: /home/wubu/bytropix | Branch: master
HW: RTX 5050 6.4GB, -arch=sm_120, NVCC: /usr/local/cuda-13.1/bin/nvcc
Build: make train_integrated
Model: /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf

=== COMPLETED ===
✅ P0: Per-block IQ2_XXS extraction
   - gguf_raw_size(IQ2_XXS) fixed: 72→66 bytes/block (empirically verified)
   - Full tensor dequant eliminated → per-expert dequant (3.9ms/expert)
   - Transpose: raw[ff][model] → [model][ff] for moe_expert_forward
   - train_integrated: 177s → 13s/step (10x)

✅ P1: Multi-flag verification (all 6 flags, all combos, 0 NaN)

✅ P2: MoE output magnitude resolved (hidden max=13, was 5e9)
✅ P2: Memory optimization (persistent buffers in lmoe_t)

=== REMAINING ===
- ~13s/step is GPU compute (40 layers), not dequant
- PGA loss jumps 21.6→69 (pre-existing LR issue)
- CPU output projection: O(N*V*D) for V=248320
