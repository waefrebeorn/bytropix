# Deep Research Wave — 2026-10-18 (8 layers, 200 queries)

Kevin-Bacon chain executed at depth 25 per layer, then a second 25-deep pass
on the aggregated convergence.

## What is here

| file | what it holds |
|---|---|
| `layer1_labs.jsonl` | 200 hits — 25 frontier labs, what each released |
| `layer2_citations.jsonl` | 196 hits — citation chase from those reports + our open gaps |
| `layer3_arxiv.jsonl` | 230 real arXiv records from direct API sweeps (title/abstract/authors/date) |
| `layer4_github.jsonl` | 200 hits — implementations on GitHub |
| `layer5_code.jsonl` | 200 hits — how the algorithms are actually implemented |
| `layer6_blogs.jsonl` | 200 hits — engineering blogs / tutorials |
| `layer7_critique.jsonl` | 200 hits — criticisms, limitations, failed replications |
| `layer8_deep25.jsonl` | 236 hits — the second 25-deep dive on the top convergences |
| `candidate_ids.json` | 432 arXiv IDs extracted from all layers |
| `verified_arxiv.json` | 432/432 confirmed by the arXiv API (real metadata) |
| `unverified_ids.json` | 0 — no fabricated IDs survived verification |

Totals: **1,662 raw records → 1,389 unique URLs; 432 verified papers; 430 PDFs.**

## Where the PDFs live

`/home/wubu/research-archive/deep-2026-10-18/`

- `papers/<arxiv-id>.pdf` — 430 files, 0.96 GiB, each checked for a `%PDF` header
- `manifest.json` — one record per paper: id, title, published, authors,
  categories, theme, pdf path, size, abstract head
- `failures.json` — 2 IDs whose PDF endpoints returned HTTP errors
  (`2601.17473`, `2510.17898`); their metadata is still in the manifest

Archive is outside git on purpose (size). Metadata and synthesis are in-repo.

## Theme split of the 432 verified papers

| theme | count |
|---|---|
| os-kernel (kernel, RT, namespaces, caps, UEFI, NUMA, allocators) | 142 |
| inference (KV, spec, quant, MoE, attention, serving) | 119 |
| compiler (regalloc, JIT, LTO, dispatch, analysis) | 70 |
| math-proof (Lean, autoformalization, hyperbolic) | 38 |
| agents-rl | 26 |
| models (technical reports / system cards) | 20 |
| security | 5 |
| other | 12 |

By year: 2026 = 268, 2025 = 93, 2024 = 27, older = 44.
Newest: 2026-10-15. Oldest: 2010-10-11.

## Reproduce / extend

```bash
# re-verify every ID against arXiv
python3 - <<'PY'
import json, urllib.request, urllib.parse, re
ids=list(json.load(open('candidate_ids.json')))
# batch id_list in groups of 50 against export.arxiv.org/api/query
PY

# resume PDF download (idempotent, verifies %PDF + size)
python3 /home/wubu/opt/research/deep_dl.py \
        /home/wubu/wubuwizard/research/deep-2026-10-18
```

## Caveats (read before citing)

- Layer 1/2/4/5/6/7/8 rows are **search-engine hits**, not verified papers.
  Only `verified_arxiv.json` / `manifest.json` are API-confirmed.
- Two PDFs 404'd at the arXiv PDF endpoint; do not assume the full text exists
  locally for those two.
- A web hit's title can be stale or paraphrased. Always cite the arXiv ID and
  the manifest title, not the search-result title.
- Nothing in this wave is implemented. Each candidate still needs the
  Triple-DA audit (correctness / no-3rd-party / robustness) before it becomes a
  `research/NNN-*.md` design doc.
