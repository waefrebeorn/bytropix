── WuBuText AI — GOAL MANTRA (DA Audit May 13) ──
Path: /home/wubu/bytropix | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
CUDA: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== STATE (freshly verified) ===
✅ train_real — CE loss 12.66, logits non-zero, 0.2 tok/s CPU
✅ test_fused_vs_old — GPU diff 0.036, PASS
✅ test_tokenizer — CJK round-trip: 你好 → 109266 → 你好  
✅ test_moe — output [-0.028, 0.031], NaN=0  
✅ dump_mmproj — 334 tensors, PASS

⛔ bench_e2e — ALL ZEROS output (GPU and CPU)
⛔ train_gpu — CE loss 69 vs expected 12.4
⛔ train_backprop — HANGS

=== STALE OLD CLAIMS (DO NOT TRUST) ===
"CE loss commented out" — FALSE, train_real shows 12.66
"IQ2 dequant = garbage" — FALSE, Q4_K was root bug, fixed
"bench_e2e loops forever" — FALSE, runs but produces zeros

=== TRUE PRIORITY ===
P0 — Fix GPU weight loading (bench.c → zeros)
P1 — Fix train_backprop hang
P2 — Verify GPU forward gives same CE as CPU

=== THE LOOP ===
pick highest undone → compile → run → verify output → document → report
ALL blocked? Write plan.md. Still blocked? Do cleanup.
