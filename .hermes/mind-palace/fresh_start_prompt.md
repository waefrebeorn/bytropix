═══ WUBUTEXT AI — FRESH START PROMPT (May 16 v7 — HONEST) ═══

HARD TRUTH: Inference is BROKEN. All binaries produce GARBAGE output.
llama.cpp reference: "Here's a thinking process:" — Us: "iscInset了下去idesiby客的我们都会论usher..."
Everything depends on fixing this first.

## Read in Order
1. `.hermes/mind-palace/plans/devils_advocate_v5.md` — Full meta audit
2. `.hermes/mind-palace/state.md` (v12) — HONEST state dashboard
3. `.hermes/mind-palace/goal-mantra.md` (v12) — HONEST goal paste
4. `.hermes/mind-palace/plan.md` — Priority queue (may still have old info)
5. `.hermes/mind-palace/entry.md` (v7) — Build + API server docs
6. `vault/bins/` — Archived old mind palace versions

## What Actually Works
- test_kv_cache: KV cache matches full recompute (max_diff=0.00) ✅
- test_256k: MoE router O(T) scaling to 65K ✅
- API server: tools/serve.py sandbox mode (fake responses) ✅
- llama.cpp reference: BUILT at ~/llama.cpp/build/bin/llama-cli ✅
- Individual component tests: SSM, GQA, MoE forward passes (numerical) ✅
- SGEMM ldC bug: FIXED (was all-zero logits)
- RoPE: Added to CPU GQA prefill path
- Sampling: temp/top-k/top-p added
- EOS detection: Fixed (eos=bos=248044)

## What's Broken (P0 — Fix First)
- infer_text_gpu v5: GPU accelerated but wrong output
- infer_text v2: CPU baseline, same wrong output
- MOE=1 also produces garbage (not just MOE=0 = no FFN)
- Root cause unknown: SSM impl? MoE dequant? Tokenizer?

## Key Architecture
- 40 layers: 30 SSM (Gated DeltaNet), 10 GQA
- No dense FFN — ALL FFN is MoE (256 experts, 8 active, D_FF=512)
- Qwen3.6 is instruction-tuned — needs chat template
- eos_token_id = bos_token_id = 248044
- Model file: ~11GB IQ2_M GGUF from Unsloth

## Reference Build
```bash
cd ~/llama.cpp/build/bin
./llama-cli -m /home/wubu/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf -p "prompt" -n 20 --temp 0.0
```

## API Server
```bash
cd /home/wubu/bytropix
python3 tools/serve.py --sandbox --port 8080    # sandbox (no GPU)
python3 tools/serve.py --port 8080              # production
bash tests/test_api.sh                           # 14 tests
```

## Vault Versioning
Old mind palace versions archived to `vault/bins/` before overwriting.
See `vault/bins/README.md` for index.

## Key Files
| File | Purpose |
|------|---------|
| tools/infer_text_gpu.c | GPU inference (broken) |
| tools/infer_text.c | CPU inference (broken) |
| tools/serve.py | API server (sandbox ✅, real backend broken) |
| src/gguf_reader.c | Dequant (Q5_K fix in source) |
| src/wubu_ssm.c | SSM/Gated DeltaNet (likely root cause) |
| src/wubu_moe.c | MoE router + expert compute |
| tests/test_api.sh | API sandbox test suite (14 tests ✅) |
| vault/unsloth-quantization-format.md | UD GGUF format docs |
| vault/api-server.md | API endpoint documentation |
