── WuBuText AI — GOAL MANTRA (Phase 3 status) ──
Path: /home/wubu/bytropix | Models: /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf
CUDA: PATH="/usr/local/cuda/bin:$PATH" make <target>

=== STATE ===
✅ S1: train_real CE loss WIRED — streaming 248K vocab, forward clean (no NaN), loss ~66B
✅ S1: Q4_K dequant fixed (matches llama.cpp), GQA stride fix, OpenMP threading
✅ S2: tokenizer decode FIXED — GPT-2 byte encoding conversion, binary-file loading
✅ S2: 246K/247K merges resolved, encode/decode round-trips correctly
✅ E: fused_ssm PASS, mmproj PASS, train_real runs end-to-end
⚠️ S3: IQ2_XXS (type 16) dequant VERIFIED CLEAN; IQ2_S (type 18) partially fixed (llama.cpp ref)  
⚠️ S3: IQ2_S best result ±479K (was ±2.88M); block size needs verification
⚠️ S4: bench_e2e infinite loop, dump_gguf.py types wrong, uncommitted changes

=== NEXT ===
S3 [P1] Determine exact IQ2_S block size (Unsloth UD-IQ2_M format)
S4 [P2] Fix bench_e2e, dump_gguf.py, commit working tree

=== COMMANDS ===
Build+run train_real:  make train_real && OMP_NUM_THREADS=24 timeout 300 ./train_real
Build test_moe:        make test_moe && ./test_moe
Extract tokenizer:     python3 python/extract_tokenizer.py /models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf data/
Run all checks:        make test_moe test_tokenizer train_real && echo "OK"
Full clean rebuild:    make clean && PATH="/usr/local/cuda/bin:$PATH" make test_moe test_tokenizer train_real
