# Status — implementation and verification state

> 2026-08-04 refresh. Every claim below was verified by running the command
> on this date (or the session it names). Weights policy: GGUF = SSD
> (`/home/wubu/models/`), safetensors = SD cold storage
> (`/home/wubu/sdcard/models/`, mount `D:` first).

## Verified this wave (2026-08-04)

| Subsystem | Evidence | Command |
|---|---|---|
| **TurboQuant loaders** (Q2_0/TQ3_1S/TQ4_1S) | synthetic GGUF round-trip: legacy type 42→47 remap, 2-bit {-1,0,+1,+2}×d exact, TQ3_1S inverse-RHT constant→delta exact | `make test_gguf_tq` |
| **DeepSeek-V4 Config-I load gate** | real 3-split headers: 43 layers, 284.3B params, 1328 tensors, 7 types, 0 unknown, 0 offset mismatches, NaN-free samples | `tools/test_gguf_load.c <part1>` |
| **HIVE f16 subnormal fix** | Q6_K blocks with subnormal scales (d=0x00cd) no longer NaN — all 12 types on the Qwen3.6 UD file decode clean | `tools/test_gguf_load.c <Qwen3.6 IQ2_M>` |
| **GGUF KV value-type table** | full spec (u8/i8/u16/i16/f64) — Config-I header parses without desync | load gate above |
| **Mixed export (5.12×)** | seed F32 140.3 MB → mixed GGUF 27.4 MB; Q8_0 cos 0.999986, Q4_0 0.9955+, IQ2_XXS 0.9955, norms maxdiff=0 | `make test_tensor_store` |
| **mHC multi-head (2512.24880 form)** | all oracles maxdiff=0 | `make test_wubu_mhc_mh` |
| **SFT run** | loss 8.04 → 7.32 @ step 2000 (seq 2048) | `tools/wubu_train_cli.c` run log |
| **Storage policy** | GGUF on SSD / safetensors on SD; KAT 72.7G + moondream 18.5G + Qwen3.6-27B 55.6G + Agents 8.5G + BTL 0.9G archived to card, verified byte-exact | `du -sb` src vs dst |

## Verified (earlier waves — require SD card mounted for real-weight claims)

| Subsystem | Evidence | Command |
|---|---|---|
| SSM forward (real weights) | Agents-A1-4B live BF16 forward, finite logits | `make test_real_load` (weights on SD) |
| ds4-ssd slot-bank | KAT 256 experts/layer paged from source shards, all finite | `./test_kat_decode_bank <KAT dir> 16` |
| HF BPE tokenizer | 248,044-vocab Qwen tokenizer round-trip | `./gen_text "<prompt>"` |
| Model config adapter | real KAT/Qwen3.6/Agents dims from config.json | `make test_model_config` |
| LoRA merge | BTL-3 base+adapter, delta applied, finite | `make test_btl3_lora` |
| Repetition (DRY + repeat-penalty) | wired to F16 params | `make test_repetition` |

## Build gates

- `make test_all` — the full gate (299 test targets; subset gates exist).
- Fresh counts 2026-08-04: 305 `src/*.c`, 21 `.cu`, 302 headers, 553 tools.
- Regenerate module tables: `python3 tools/repodoc/repodoc.py . --readme --modules`.
