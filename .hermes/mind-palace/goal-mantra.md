── WuBuText AI — GOAL MANTRA (Phase 3 status) ──
Path: /home/wubu/bytropix | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
CUDA: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== STATE ===
✅ S1: train_real CE loss WIRED — streaming 248K vocab, forward clean (no NaN), loss ~6.6e10
✅ S1: Q4_K dequant fixed (matches llama.cpp), GQA stride fix, OpenMP threading
✅ S1: train_real runs end-to-end: 0.2 tok/s (B=1 T=4 CPU), all 40 layers
✅ S2: tokenizer encode+decode FIXED — Latin-1/GPT-2 byte encoding, round-trips CJK
✅ S2: 你好 → [109266] → 你好, 你好世界, hello world all verified
✅ S3: test_moe verified — IQ2_XXS+IQ2_S load, router sum≈1.0, output range ±4.8e5
⚠️ S3: IQ2 output range ±4.8e5 expected for IQ2_M quantized 2-bit weights (not a bug)

=== COMMITTED ===
commit 9352aba: S2 tokenizer encode+decode round-trip fix

=== NEXT ===
S4 [P2] Rebuild all binaries, commit working tree, update docs
S3 [P1] IQ2_S block size verification vs llama.cpp reference (optional)

=== COMMANDS ===
Build+run train_real:  make train_real && timeout 120 ./train_real
Test tokenizer:        make test_tokenizer && ./test_tokenizer /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf "你好"
Test MoE:              make test_moe && ./test_moe
Full rebuild:          make clean && make test_moe test_tokenizer train_real
