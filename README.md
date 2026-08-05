# wubuwizard — the BRAIN of the WuBu AGI

**From-scratch C11 inference + training engine.** No libggml, no Triton, no
third-party BLAS: quantization, matmul, tokenizers, model loaders, and the
training core are all in-tree. This is the backbone of an AGI — a colony of
specialized experts on nested spheres, watched by a diagnostic hive that grows
overworked cells, shrinks dead ones, and archives every validated mutation.

- **Code lives on GitHub** — [`waefrebeorn/wubuwizard`](https://github.com/waefrebeorn/wubuwizard)
- **Models + datasets live on HuggingFace** — the [`WaefreBeorn` org](https://huggingface.co/WaefreBeorn):
  the [**WuBu-35M**](https://huggingface.co/WaefreBeorn/WuBu-35M) seed (weights,
  tokenizer, config, checkpoints, AGI-disclosure card), plus the bytropix registry
  dataset. **Weights are never committed to git** — they ship on HF.

## The standing loop (the doctrine in one line)

```
corpus → train → diagnose → mutate → validate → archive → RLHF oracle → repeat
```

Every batch closes gaps ON this loop. WuBu is the amoeba (grows/shrinks by
measured cell fitness), the hive IS the diagnostic system (research/056), and
the Body (`wubuos`) hosts this engine on metal via the Live Colonel. The full
map: [docs/TOPOLOGY.md](docs/TOPOLOGY.md).

## Quick start

```bash
make gen_text            # CPU inference binary
make wubu_train          # the trainer (real backward + Muon)
make test_all            # the full test gate (299 targets)
make api_server          # OpenAI-compatible HTTP server

./gen_text "<prompt>"    # generate (needs weights — see Models)
```

C11 (GCC/Clang), `-std=c11`, opaque structs, minimal includes. CUDA paths need
`nvcc` (optional). Details: [docs/BUILDING.md](docs/BUILDING.md).

## What the engine does

| Layer | What | Modules |
|---|---|---|
| **Loaders** | safetensors, GGUF (incl. TurboQuant Q2_0/TQ3_1S/TQ4_1S + 3-part splits), `.st` dumps — the catalog doctrine: a format is a catalog over the same bytes | `gguf_reader.c`, `safetensors_reader.c`, `wubu_tensor_store.c` |
| **Quantization** | Q4_K…IQ4_XS + Q8_0 + TurboQuant; **mixed per-role ladders** (research/057-058): keep the sensitive (attn/embd Q8_0), crush the saturated (experts 2-bit) — 5.12× on the seed | `quantized_matmul.c`, `dequant_iq2_xxs.c`, … |
| **Attention** | GQA, **MLA** (latent KV), SSM-Gated-DeltaNet hybrid, ring/paged KV | `wubu_mla.c`, `wubu_ssm.c`, `wubu_kv_cache.c` |
| **MoE** | 256-expert routing + shared expert; **hash routing**; SSD-paged experts for small RAM | `wubu_moe.c`, `wubu_hashrouter.c`, `wubu_ssd_moe.c` |
| **Training** | real backward through every path, Muon + AdamW, rolling checkpoints | `wubu_train.c`, `wubu_backprop.c` |
| **The AGI organs** | amoeba + hive (diagnostics), mHC hyper-connections, DSA indexer, prover | `wubu_amoeba.c`, `wubu_hive.c`, `wubu_mhc_mh.c`, `wubu_dsa.c` |

Fresh counts (2026-08-04): **317 C modules, 316 headers, 77,473 LOC, 302
test targets, 45 research docs.** The LFM2.5 on-device engine is now live
(`src/lfm2_*.c`, 6 self-contained C11 modules; `make -f Makefile.lfm2`) — scale-gap
fixed per the HuggingFace `modeling_lfm2.py` spec (embedding_norm once after layers,
no final_norm). The full annotated module table: [docs/MODULES.md](docs/MODULES.md).

## Models

| Model | Form | Status |
|---|---|---|
| [**WuBu-35M**](https://huggingface.co/WaefreBeorn/WuBu-35M) (the seed) | safetensors / .st (HF) | training loop runs (SFT 8.04→7.32); mixed export 5.12× |
| Agents-A1-4B (dense hybrid) | safetensors (SD card) | real SSM forward verified ✅ |
| KAT-Coder-V2.5-Dev (MoE 256) | safetensors (SD card) | 13/13 shards; SSD slot-bank pages experts |
| Qwen3.6-27B (dense hybrid) | safetensors (SD card) | adapter derives dims; SSM forward finite ✅ |
| Qwen3.6-35B-A3B-UD-IQ2_M | GGUF (SSD) | 753 tensors dissected; 12 types NaN-free |
| **DeepSeek-V4-Flash-0731 Config-I** | GGUF 3-split (SSD) | **load gate PASSED**: 43L, 284.3B, 1328 tensors, 7 types, 0 mismatches — forward wiring next |

**Weights policy**: GGUF lives on the SSD (`/home/wubu/models/`); safetensors
model dirs are COLD STORAGE on the SD card (`/home/wubu/sdcard/models/` —
mount `D:` first: `sudo mount -t drvfs D: /home/wubu/sdcard`).

## Docs

- [docs/TOPOLOGY.md](docs/TOPOLOGY.md) — the master map of BOTH repos (Brain/Body split, boundaries, placement rules)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — the engine spine: data flow, layers, guarantees
- [docs/BUILDING.md](docs/BUILDING.md) — build, run, train
- [docs/MODULES.md](docs/MODULES.md) — full annotated module table (auto-generated)
- [STATUS.md](STATUS.md) — verified claims, per wave
- [research/INDEX.md](research/INDEX.md) — the gap ledger (AN01-AN11, 45 notes)
- [docs/wubu-model-card.md](docs/wubu-model-card.md) + [docs/wubu-model-blueprint.md](docs/wubu-model-blueprint.md) — the model
- [THEORY/](THEORY/) + [MATH/lean/](MATH/lean/) — the papers + Lean-verified proofs

## License

[Waefrebeorn Umbrella License v3.0](LICENSE) — source-available, not OSI/FSF
approved.

<!-- repodoc:BEGIN -->
## Module index (auto-generated 2026-08-04)

- **305 C modules** — full annotated table: [docs/MODULES.md](docs/MODULES.md)
- **348 test tools** (make targets `test_*`, e.g. `test_200, test_256k, test_256k_chunked, test_256k_context, test_256k_forward, test_300, test_400, test_4kv, test_512k_budget, test_adaptive_hotpath...`)
- **45 research docs** — full ledger: [research/INDEX.md](research/INDEX.md)

Regenerate with: `python3 tools/repodoc/repodoc.py . --readme`
<!-- repodoc:END -->
