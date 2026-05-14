── WuBuText AI — GOAL PASTE ──
Path: /home/wubu/bytropix | Repo: waefrebeorn/bytropix
Model: deepseek-chat | HW: RTX 5050 6.4GB | English only | Pure C + CUDA
Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
Build: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== WHO WE ARE ===
Building Qwen3.6-35B-A3B from scratch in C + CUDA with WuBu nested hyperbolic geometry. 40-layer hybrid (30 SSM + 10 GQA), 2048 hidden, 248K vocab, 256 MoE experts. Embeddings mapped to Poincaré ball (R=0.956, exp_map).

=== WHERE WE ARE (DA Audit May 13) ===
✅ CPU forward + CE loss — train_real: 12.66 loss, 0.2 tok/s
✅ CJK tokenizer — round-trip verified
✅ MoE forward — 256 experts, clean output, 36.6 tok/s
✅ MMProj dump — 334 tensors verified
✅ Lean proofs — 4 verified (Möbius add, exp/log maps, gyration)
⛔ GPU weight loading broken — bench_e2e produces zeros (P0)
⛔ GPU training wrong loss — CE 69 vs 12.66 (same root cause)
⛔ Backprop hangs — train_backprop stalls at model init (P1)

=== MATH WEAPONS (from the vault) ===
Poincaré ball: exp_map(v)=tanh(||v||/R)·v/||v||, R=0.956
Möbius add: x⊕y = ((1+2⟨x,y⟩+||y||²)x + (1-||x||²)y) / (1+2⟨x,y⟩+||x||²||y||²)
Gyration: gyr[x,y]z = -(x⊕y)⊕(x⊕(y⊕z)) — rotation in tangent space
RSGD optimizer: step in tangent space, project back via exp_map
Full theory: THEORY/WuBu_Nesting.md

=== THE LOOP ===
pick → compile → run → verify (non-zero!) → document → next
ALL blocked? Fix docs or read theory from vault.
