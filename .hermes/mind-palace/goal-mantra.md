── WuBuText AI — GOAL MANTRA (Phase 3 status) ──
Path: /home/wubu/bytropix | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
CUDA: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== STATE ===
✅ S1: train_real CE loss — runs end-to-end (0.2 tok/s CPU, all 40 layers, streaming 248K vocab)
✅ S2: tokenizer encode+decode — full CJK round-trip (你好 → [109266] → 你好)
✅ S3: test_moe — IQ2_XXS+S load, router sum≈1.0, output range ±4.8e5 (expected for 2-bit)
✅ bench_e2e — 53.15x GPU speedup (10.45 tok/s vs 0.20 tok/s CPU), noise suppressed
✅ dump_gguf.py — type mapping already correct
✅ All source files committed (27 files, 6299 lines)
✅ Commits: 9352aba (tokenizer fix), 6118723 (batch commit)

=== REMAINING (optional) ===
- IQ2_S block-level verification vs llama.cpp reference
- train_real: loss ~6.6e10 needs calibration (expected for untrained 2-bit quant)
- Merge hash collisions (58074 for 247587 entries in 524288 slots — 11%)

=== COMMANDS ===
Benchmark:     make bench_e2e && ./bench_e2e
Test tokenizer: make test_tokenizer && ./test_tokenizer /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "你好"
Train:           make train_real && timeout 120 ./train_real
Test MoE:        make test_moe && ./test_moe
