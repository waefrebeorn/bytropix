# 047 — Knowledge substrate: the 7 research ponds (701 MB pure text)

> Status: `closed` (substrate built). Date: 2026-08-03.
> Maps to: THEME AM (amoeba self-improvement loop) + all research themes.

## What was built

The Kevin-Bacon doctrine says research must aggregate into DOWNLOADED
sources referenced over and over — nothing lives only in a conversation.
This is that, at scale: **7 ponds × 100 MB of pure text = 701 MB**,
archived, deduplicated, purity-gated, topically indexed.

Layout (SSD = active, SD card = cold archive):
```
/home/wubu/research-ponds-work/
├── PONDS.md            <- catalog + usage docs
├── ponds/              <- the 7 pond archives (working copies)
│   ├── pond-*/text/*.txt    <- pure text, one file per source
│   ├── pond-*/manifest.json <- byte ledger
│   ├── pond-*/index.json    <- topic -> file-number map
│   └── pond-*/sources.json  <- file-number -> arxiv:ID/github:repo:path
├── work/               <- live harvest scratch
└── tools/              <- harvest_pond.py, index_ponds.py, pond-*.json
/home/wubu/sdcard/archive/  <- COLD: one .tar.gz per completed pond
```

## The ponds (theme map)

| Pond | Theme | Dominant topics (indexed) | Files |
|---|---|---|---|
| pond-a-kv-memory | THEME A (KV/memory) | attention 2071, kv-cache 714, serving 586 | 2892 |
| pond-b-quantization | THEME B (quant) | quant 1592, jit 864, kernel 654 | 5219 |
| pond-c-moe | THEME WB/MoE | moe-expert 1465, attention 790 | 1778 |
| pond-d-tokenization | THEME AM02 (BLT) | tokenizer 1299, byte-patch 376 | 3302 |
| pond-e-training-rlhf | THEME RC (recipe) | data 780, rlhf 680, optimizer 267 | 8186 |
| pond-f-self-modifying | THEME AM01 (amoeba) | growth 289, continual 258 | 5061 |
| pond-g-systems-os | THEME C/D (systems) | jit 1226, kernel 764, self-improve 202 | 5411 |

## How the loop consumes it

The self-improvement loop (corpus → train → diagnose → mutate → validate
→ archive → RLHF oracle → repeat) pulls contextual sources per gap:
- grep the pond for the failing subject → file numbers → sources.json
  gives the paper/repo → read the full source, cite it, implement.

## Lessons (pitfalls that cost time)

1. **drvfs (the SD card) has NO chmod** — `git clone` fails instantly
   ("could not set core.filemode") AND `shutil.copy` fails (copymode).
   Fix: clone on the SSD (never the SD card); use `shutil.copyfile` not
   `copy` for SD writes. GitHub token works via `x-access-token:TOK@`
   URL form (the `http.extraheader` form fails on upload-pack).
2. **drvfs 256KB clusters** — thousands of small files waste ~230KB each
   (543 files → 137MB on disk for 34MB content). Cold storage must use
   ONE tar.gz per pond, not loose files.
3. **arXiv throttles ~0.44 files/s** regardless of worker count —
   ~1000 papers/pond ≈ 65 min/pond. Plan for it; GitHub repos are the
   volume lever (golang/go = 4747 files).
4. **Resume numbering bug**: `self.files += 1` per existing file breaks
   after a resume with index gaps — new files overwrite old ones and the
   byte ledger double-counts. Fix: `nfiles` = true count, `files` = next
   index (max+1). Verified: pond-g 92.2→100.0 after honest recount.
5. Stale `repos/` leftovers in pond archives must be cleaned (they're
   clone scratch, not content).
6. The manifest `sources[]` only tracks the current run after a resume —
   rebuild `sources.json` from the `# SOURCE:` headers (authoritative).

## Token

GitHub token stored at `~/.hermes/profiles/mind-palace/secrets/github.env`
(0600, never committed, passed via URL form only).
