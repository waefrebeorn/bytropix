# research/061 — THE KV CACHE IS A FILE SYSTEM (7-hop + implementation path)

> Kevin-Bacon 7-hop on the user directive **"THE KV CACHE IS A FILE
> SYSTEM"** (2026-08-05): trace the seed through the serving/OS lineage,
> aggregate the mechanisms that converge on one principle, map every
> mechanism to what is ALREADY in-tree, and name the one missing layer.
> Doctrine: `THEORY/05-kv-cache-filesystem.md`. Architecture:
> `docs/wubu-model-blueprint.md` §2.7.

## Seed

"All files are data. THE KV CACHE IS A FILE SYSTEM. A magic computer
where downloads are in the same space as synthesized data. The model is
ours, so it can be agnostic. The multimodal 12B single-encoder
non-standard Gemma is the closest base design. All inputs are encoded."

## The 7 hops

| Hop | System / paper | Core mechanism | What it proves for the doctrine |
|---|---|---|---|
| 1 | **TempleOS / ZealOS** (the body's own lineage) | The command line feeds the HolyC compiler; code is placed into memory it `MAlloc()`s; I/O space unified with memory space | The precedent: memory and files are ONE space. The AGI's body already lives this doctrine — the brain must too |
| 2 | **vLLM PagedAttention** (SOSP 2023) | KV cache = virtual memory: fixed KV blocks, block table maps logical→physical, global free pool, ref-counted copy-on-write prefix sharing; 60–80% waste → <4% | The KV cache IS paged memory. The block table is the address-translation layer of a filesystem |
| 3 | **SGLang RadixAttention** (arXiv 2312.07104) | Global radix tree over KV blocks; a prompt prefix IS a path in the tree; identical prefixes share blocks, 5–10× for agent swarms | The KV cache IS a hierarchical namespace. Paths are real: a radix tree is a filesystem directory structure over KV |
| 4 | **MemGPT: LLMs as Operating Systems** (arXiv 2310.08560) | Context window = main memory (RAM), external storage = disk; the MODEL pages its own memory via function calls, memory-pressure interrupts, self-editing memory blocks | The model must MANAGE its own memory lifecycle — the namespace is self-administered, not framework-administered |
| 5 | **Mooncake** (FAST '25, Kimi) + **llm-d tiered prefix cache** | Disaggregated KV pool over CPU/DRAM/SSD ("trading more storage for less computation"); production KV caches live as FILES under `/mnt/files-storage/kv-cache/` on Lustre/EFS/Storage Scale | The KV cache IS a storage service, and in production it literally lives on filesystems. Cross-node, cross-session |
| 6 | **LMCache** (arXiv 2510.09665) + NVMe offload tiers | Persistent tiered KV layer: hot HBM → warm DRAM → cold NVMe → remote S3/NFS; KV chunks stored as files (`file://local/disk/`, 256-token chunks), async offload, LRU eviction, survive restarts | The KV cache IS a tiered persistent filesystem. Hot/warm/cold = the memory hierarchy, exactly like an OS |
| 7 | **Infini-attention** (arXiv 2404.07143) + **Gemma 3 12B** (arXiv 2503.19786) | Infini: per-head compressive memory (linear-attention state) in EVERY block — the model writes its own compressed summaries into the memory and reads them forever. Gemma 3: ONE SigLIP encoder (400M), every modality → soft tokens in ONE sequence space (256 tokens/image), 5:1 local/global interleave cuts KV, 128K ctx | The two halves of "synthesized data + all inputs are encoded": compressive memory = the synthesized-write mechanism; single encoder = everything becomes data in the same space |

## Convergence

**Every mechanism of the KV-as-filesystem already ships in production or
in the literature. The missing layer is the namespace — the address
abstraction that unifies blocks, tiers, persistence, mounts, and
synthesized writes under one path-addressable view the model reads and
writes like a file system.**

The convergence principle, one line: **the KV cache is not a buffer with
filesystem features; it is a filesystem whose blocks happen to be KV
tensors — and the AGI's body (WuBuOS, 9P, single-level store) is the
proof that a whole organism can run on exactly that idea.**

## Triple-DA

1. **Correctness** (CPU / 13 GB RAM / no third-party): every mechanism
   below is a C11-routable data-structure job (radix tree, block table,
   chunk files via mmap, tier LRU) — no GPU kernel required for the
   address layer; the tensor layer (paged KV, MLA, quant) is already
   in-tree and verified.
2. **Privacy / no third-party**: all mechanisms are self-contained
   (Mooncake/LMCache are systems to LEARN from, not link); persistence
   is local files + mmap, consistent with the SSD-active/SD-cold storage
   doctrine.
3. **Robustness**: namespace layer degrades to a plain paged KV cache
   when unmounted (the "no namespace" fallback = today's decode path);
   every mount is a directory, not a hack; ring-bounded tiers can't
   bloat (the 103-checkpoint lesson).

## What is ALREADY in-tree (the doctrine is half-built)

| Mechanism | In-tree module | Status |
|---|---|---|
| Paged / ring KV (the block layer) | `wubu_kv_cache.c` (paged_kv, ring attention, kv_cacheline) | wired |
| Tiered hot/warm/cold (the memory hierarchy) | `wubu_kv_tier.c` (3-tier, EMA-LRU, fp16 cold) | wired (A06) |
| Prefix+PD persistence (the chunk-file layer) | `wubu_lmcache.c` (FNV-1a64 keyed chunks) | wired (A07) |
| Priority eviction (the filesystem policy) | `wubu_kv_evict.c` (recencyEMA × importance) | wired (A07b) |
| Latent KV (compressive memory seed) | `wubu_mla.c` (latent compress/up-proj) | wired (A08) |
| Coarse-to-fine indexer (the read path) | `wubu_dsa.c` (block-means indexer, top-k) | wired (AN03) |
| Path-addressed memory (the namespace seed) | `wubu_orbits.c` (polar recursion paths, hive backing) | wired (AN12) |
| Long context (1M+ tokens) | ring attention | wired |
| The OS body (the export target) | WuBuOS Styx/9P (`styxfs_vfs.c`), VSL, runtime spaces (`/n/...`) | wired |

## The gap (the missing layer — what "make it work" means)

**G1 — the namespace (`wubu_kvfs`)** — a path-addressable view over the
existing paged/tiered/persistent KV: paths resolve through a radix tree
(RadixAttention) whose leaves are KV block ranges (PagedAttention) whose
backing is tiered (wubu_kv_tier) and persistent (wubu_lmcache chunk
files). `open("/kv/in/doc-0042")` → block range → attention.
**`wired` (2026-08-05)** — see "G1 wired" below.

**G2 — the mount table** — `/kv/in` (external data: single-encoder
output lands here), `/kv/synth` (model writes), `/kv/mem` (persistent,
survives sessions), `/kv/meta` (diagnostics). Mount = bind a 9P path.

**G3 — the single-encoder modality head** — one encoder, every input
(image/audio/file bytes) → soft tokens in the ONE sequence space
(Gemma 3 pattern); nothing enters `/kv/` unencoded. This is the
modality-agnostic guarantee.

**G4 — synthesized write-back** — compressive memory heads (Infini-style
per-head linear state, seeded by the existing MLA latent path) so the
model's own summaries are WRITTEN into `/kv/synth` and persist — the
"thoughts are files" mechanism.

**G5 — the Styx export** — the namespace served to the OS at `/n/kv/`
(kevin-bacon step 5 was already planned: "KV-styx live registration 9P
export at /n/kv/") — the body can `ls` the mind; the mind can read the
body's files. One data plane.

**G6 — the self-administering loop** — MemGPT's lesson: the model pages
its own memory. Path writes/evicts/mounts become TOOL CALLS the model
makes (the agentic corpus already trains this shape); memory-pressure
interrupts fire at tier high-water marks.

## Sources (downloaded/verified 2026-08-05)

- Gemma 3 Technical Report — arXiv 2503.19786 (SigLIP 400M, 256 soft tokens/image, 5:1 local/global interleave, 128K)
- vLLM PagedAttention — SOSP 2023 (block tables, CoW sharing, <4% waste)
- vAttention — arXiv 2405.04437 (virtual contiguity, deferred physical commit)
- SGLang / RadixAttention — arXiv 2312.07104 (radix tree over KV blocks)
- MemGPT — arXiv 2310.08560 (virtual context management, paging via function calls)
- Mooncake — FAST '25 / arXiv 2407.00079 (disaggregated KV pool over CPU/DRAM/SSD)
- LMCache — arXiv 2510.09665 + docs.lmcache.ai (tiered persistent KV, chunk files on disk)
- Infini-attention — arXiv 2404.07143 (per-head compressive memory)
- MemOS — arXiv 2507.03724 (memory OS for AI systems; KV = implicit short-term memory)
- NVMe KV offload tiers (HBM ~3.35 TB/s / DRAM ~63 GB/s / NVMe ~7 GB/s)
- TempleOS docs (MAlloc-a-file, I/O space unified with memory space)

Status: G1 `wired` (2026-08-05) — see the G1-wired note below.
G2–G6 `open` — the mount table, single-encoder head, synthesized
write-back, Styx export at /n/kv/, and the self-administering loop are
the next implementation wave; the doctrine + architecture are wired
(THEORY/05 + blueprint §2.7).

---

## G1 WIRED — the kvfs address layer shipped (2026-08-05)

This session closed G1: `wubu_kvfs` is implemented, wired into
`wubu_model_t`, and verified with real numbers.

### What shipped

- **`include/wubu_kvfs.h` / `src/wubu_kvfs.c`** — the namespace module.
  Mount table is an **FNV-1a 64-bit open-addressing hash table** (linear
  probe, 50% load cap, tombstones, doubling) — NOT a linear scan.
  Longest-prefix lookup walks parent path segments (one hash probe per
  segment): O(path depth), not O(mount count).
- **Resolve-once handles** — `wubu_kvfs_open()` resolves a path once into
  an opaque handle carrying the precomputed absolute float offset +
  capacity; `wubu_kvfs_handle_read/write` are a bounds check + memcpy
  with **zero string ops**. The mount struct is cold (paths only touched
  at mount/snapshot time).
- **Model wiring (ADR-003)** — `wubu_model_t` gains `kvfs`,
  `kvfs_block_floats`, `kvfs_n_layers`, `kvfs_n_handles`,
  `kvfs_layer_handles[]` (per-layer resolve-once handles cached at init).
  Init mounts `/kv/layer_XX` per GQA layer and resolves the handles; free
  closes them then destroys the namespace. Accessors: `wubu_model_kvfs`,
  `wubu_model_kvfs_read/write` (path), `wubu_model_kvfs_layer_handle`,
  `wubu_model_kvfs_open_handle`, `wubu_model_kvfs_handle_read/write`,
  `wubu_model_kvfs_snapshot_json`.
- **Backend vtable routing** — `wubu_backend_t` gains `kvfs_read`,
  `kvfs_write`, `kvfs_snapshot` (+ `set_ssm_hybrid`, `sync_ssm_state_to_gpu`,
  `chunk_size` for the GPU paths). CPU stub falls through to the flat
  tensor; CUDA backend implements real methods. `void *stream` in the
  vtable, never `cudaStream_t` (no CUDA leak into the CPU header).

### Measured (test_kvfs on this box)

```
[BENCH] lookup /kv/layer_0511 (512 mounts): 16.6 ns/op (60 M ops/s)
[BENCH] handle write 64 floats: 3.1 ns/op (82.8 GB/s)
[BENCH] handle read 64 floats:  3.1 ns/op (82.3 GB/s)
```

The old linear scan over 512 mounts with strncmp would be ~microseconds
per lookup (~100x slower). Handle I/O is memcpy-bound.

### Pre-existing link errors fixed along the way (do NOT regress)

- `wubu_backend.o` was missing from the Makefile CORE_OBJ → added.
- `wubu_format_onnx_stub` undefined → created
  `src/wubu_model_format_onnx.c` (probe-only stub, real symbol) + added
  to CORE_OBJ.
- `wubu_ssm_backward_output_proj` / `wubu_ssm_backward_gated_norm` /
  `wubu_ssm_backward_gated_norm_weight` were "extracted to wubu_ops.c"
  but never actually moved — recovered from git history and implemented
  in `src/wubu_ops.c`. `test_ops` now links `wubu_dims.o` +
  `wubu_dims_gpu_stub.o` for `WUBU_DIMS`.
- `wubu_model_gpu_chunk_sz` / `wubu_gpu_sync_ssm_state_to_gpu` /
  `wubu_gpu_set_ssm_hybrid` → routed through the backend vtable instead
  of direct GPU calls (CPU-only builds link clean).

### Test gates (real, re-run 2026-08-05)

- `test_model_kvfs` — 19/19 PASS (self-contained mock model: init,
  accessor, mount count, path I/O round-trip, resolve-once handles,
  bounds checks, open_handle, snapshot JSON, teardown, NULL safety).
- `test_kvfs` — 13/13 PASS + benchmarks.
- `test_ops` 6/6, `test_fmt` ALL PASS, `test_enc_h3` 20/20, full
  `make -j4` 0 errors.

### Test-design lesson (the DA catch this session)

The first version of `test_model_kvfs` wrote `src[i] = i*0.25f` so
`src[0] = 0.0f` — a silent-write bug was MASKED because both src and dst
were zero at index 0 and the read-back "matched". Fix: use a non-zero
pattern `(i+1)*0.25f` AND write to the same layer you read via the
handle (layer_01 vs layer_02 mismatch was the second trap). Rule: a
round-trip test must write non-zero data and check the same address on
both sides.
