# wubuwizard Building & Training

> 2026-08-04. Build, run, and train the Brain. Facts verified against the
> Makefile and repo on this date.

## Build

```bash
make gen_text            # CPU inference binary (tools/gen_text.c)
make wubu_train          # the trainer (tools/wubu_train_cli.c)
make wubu_train_gpu      # CUDA trainer (optional, needs nvcc)
make api_server          # OpenAI-compatible HTTP server
make test_all            # full test gate (299 test targets — subset gates exist)
make <any test_*>        # single module test
```

C11 (GCC/Clang), `-std=c11`, opaque structs, minimal includes. CUDA paths are
optional (`nvcc`); everything else is CPU. There is no third-party ML
dependency — quantization, matmul, tokenizers, and loaders are in-tree.

## Run inference

```bash
./gen_text "<prompt>"              # CPU generation (needs weights — see below)
./gen_text --model <path>          # explicit safetensors/GGUF path
```

Weights policy (2026-08-04): GGUF files live on the SSD (`/home/wubu/models/`),
safetensors model dirs are COLD STORAGE on the SD card (`/home/wubu/sdcard/
models/`, mount `D:` first: `sudo mount -t drvfs D: /home/wubu/sdcard`).
The WuBu-35M seed lives in-repo at `models/wubu/` (weights untracked in git —
canonical copy on HF `WaefreBeorn/WuBu-35M`; the tensor store rebuilds formats).

## Train (the standing loop)

```bash
corpus → train → diagnose → mutate → validate → archive → RLHF oracle → repeat
```

1. **Corpus** — `/home/wubu/models/corpus/` (SSD, active): `CORPUS.md` is the
   manifest; `.tok` uint16 streams from `tools/wubu_tokenc.c` (C11 BPE);
   checkpoints `seed.st-NNN.st` (rolling 3-per-line retention via
   `tools/wubu_ckpt_roll.py`).
2. **Train** — `tools/wubu_train_cli.c` + `src/wubu_train.c` +
   `src/wubu_backprop.c` (real backward + Muon). Reference SFT run: loss
   8.04 → 7.32 @ 2000 steps, seq 2048.
3. **Diagnose/mutate/validate** — the amoeba + hive (`wubu_amoeba`,
   `wubu_hive`); every mutation Triple-DA'd and archived (DGM pattern).
4. **RLHF oracle** — `tools/nvidia_nim.py`, `tools/openrouter_rlhf.py`
   (keys in `~/.hermes/profiles/mind-palace/secrets/hf.env`, never in repos).

## Model format doctrine

A format is a **catalog over the same bytes** (`wubu_tensor_store.c`):
open any checkpoint as (name, offset, dtype, shape); convert between .st /
safetensors / GGUF by streaming, never load-all-then-save. Mixed compression
is per-role (research/057-058): sensitive tensors keep Q8_0, saturated expert
weights drop to 2-bit (Q2_0/IQ2_XXS).

## Verify

- `make test_all` — the gate (299 targets).
- `tools/test_gguf_load.c <gguf>` — the load gate: walks every tensor, flags
  unknown types / offset mismatches / NaN dequants (passed on the real
  DeepSeek-V4 Config-I headers: 1328 tensors, 7 types, 0 mismatches).
- `tools/test_gguf_tq` — TurboQuant dequant unit test (Q2_0/TQ3_1S exact).

Regenerate docs: `python3 tools/repodoc/repodoc.py . --readme --modules`.
