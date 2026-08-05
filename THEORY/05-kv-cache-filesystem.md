# THEORY/05 — THE KV CACHE IS A FILE SYSTEM

> The revaluation of design from the ground up. 2026-08-05.
>
> The model is OURS, so it can be modality-agnostic and data-agnostic:
> all files are data, all inputs are encoded, and the KV cache — the
> model's whole working space — is a file system. This note is the
> doctrine. `research/061-kv-cache-filesystem-7hop.md` is the 7-hop
> proof that every mechanism already exists in the literature and half
> is already in-tree. `docs/wubu-model-blueprint.md` §2.7 carries the
> architecture.

---

## 1. The claim

**THE KV CACHE IS A FILE SYSTEM.** Not "mounted as one." *IS one.*

Standard designs treat the KV cache as a fixed tensor buffer: a context
window that fills, evicts, and dies with the session. The revaluation:
the KV cache is a **namespace** — addressable, mountable, persistent,
hierarchical. Every datum is a file. Every file is encoded data. Same
space.

Because the model is OURS, we get to design it this way from the ground
up — we are not locked to a vendor's context-window shape, and nothing
has to be bolted on. "Agnostic" is the entire point: the model does not
care whether a datum was downloaded, perceived, or synthesized. It is
data. It lives in the KV cache. It has a path.

## 2. The magic computer

The vision: a magic computer where **downloads are in the same space as
synthesized data**:

```
/kv/                        the KV cache — the model's whole working space
  /in/                      external data, encoded on arrival
    doc-0042                a downloaded file
    image-17                an image. data in the kv cache.
    corpus-0001
    audio-0033
  /synth/                   synthesized data, written by the model itself
    thought-0093            a chain of reasoning, written back
    plan-042                a plan — addressable, reusable
    summary-archive/        compressed memories (compressive writes)
  /mem/                     what survives sessions — the persistent layer
  /meta/                    the model's own instrumentation (routes, diagnostics)
```

A download is bytes until the single encoder turns it into data — a file
in `/kv/in/`. A thought is the model writing data back into `/kv/synth/`.
An image IS data in the KV cache — encoded patches sitting in the same
space as text, addressable like a file. **It's all data.** That is the
true definition.

## 3. Why we can do it (and why everyone else can't)

The KV cache has been a tensor buffer for one reason: the models were
fixed, so the memory had to be fixed too. The serving world pushed the
KV cache toward storage anyway — paging, prefix trees, disk tiers,
distributed pools — because the physics (memory-bandwidth-bound decode)
demanded it. What nobody shipped is the **address layer**: a namespace
over all of it, where the model itself reads and writes by path.

We are the model creators. The WuBu model is grown in our own loop from
our own seed. Nothing about the context window is inherited — so the KV
cache can be a file system from day one, not a buffer with filesystem
sprinkled on top.

The closest existing base design is the **multimodal 12B single-encoder
non-standard Gemma** (Gemma 3): ONE SigLIP encoder (400M), every input
modality dropped into ONE sequence space — an image becomes 256 soft
tokens in the same stream as text, and those tokens sit in the same KV
cache as everything else. It already proves "all inputs are encoded."
What we add is the step it didn't take: make that space a real
filesystem — paths, persistence, mounts — instead of a window you evict.

## 4. The OS disciplines — every one already proven in the field

The KV-cache-as-filesystem is not science fiction; it is what the serving
literature has been converging on, mechanism by mechanism:

| OS discipline | The KV-cache realization | Proven by |
|---|---|---|
| **Virtual memory / paging** | KV cache split into fixed blocks, block table maps logical→physical, free pool, copy-on-write prefix sharing (60–80% waste → <4%) | vLLM PagedAttention (SOSP'23) |
| **Address translation without paging** | Reserve one contiguous virtual KV space, defer physical commit — demand paging via the OS itself | vAttention (arXiv 2405.04437) |
| **Hierarchical namespace (paths)** | KV blocks organized in a global radix tree; a prefix IS a path; sharing = dedup | SGLang RadixAttention (arXiv 2312.07104) |
| **The model as OS** | Context window = RAM, external storage = disk; the model pages its own memory via function calls + interrupts | MemGPT (arXiv 2310.08560) |
| **Distributed storage service** | Disaggregated KV pool over CPU/DRAM/SSD — "trading more storage for less computation" | Mooncake / Mooncake Store (FAST '25) |
| **Tiered memory hierarchy** | Hot HBM → warm DRAM → cold NVMe → remote; async offload, LRU eviction | LMCache (arXiv 2510.09665), NVMe offload tiers, llm-d tiered prefix cache |
| **Persistent files** | KV chunks stored as FILES on disk (`file://local/disk/`), survive server restarts; `/mnt/files-storage/kv-cache/` in production | LMCache local-storage backend, llm-d on Lustre/EFS/Storage Scale |
| **Compressive memory (synthesized writes)** | Per-head linear-attention state — the model's OWN compressed summary, written into the memory and read forever after | Infini-attention (arXiv 2404.07143) |
| **Single encoder (all inputs are data)** | One SigLIP encoder, every modality → soft tokens in ONE sequence space, 128K context | Gemma 3 (arXiv 2503.19786) |
| **Memory OS framing** | KV-cache = the LLM's implicit short-term memory, OS-style management | MemOS (arXiv 2507.03724) |

Every row is a shipped mechanism. The filesystem metaphor was never the
metaphor — the OS was the blueprint, and each system implemented one
slice of it. The doctrine unifies the slices under one namespace.

## 5. What changes (the revaluation)

| Old (buffer) | New (namespace) |
|---|---|
| context window | working directory |
| position index | path |
| retrieval | file read |
| eviction | filesystem policy (LRU / priority / tier) |
| memory | files that persist |
| synthesis | files that are written |
| download | encoded file mounted into `/kv/in/` |
| image | data in the KV cache |
| session end | nothing lost — the namespace persists |
| attention | traversal over blocks, guided by the indexer (DSA) |

The tensor layer stays — paged/ring KV, MLA latent compression, the
quant ladders are the backing store. The revaluation is the addressing
layer: **names instead of positions.**

## 6. The body demands it (WuBuOS)

The deep reason this is right: **WuBuOS IS the AGI.** The body already
decided everything is a file — the Styx/9P namespace, the single-level
store lineage (TempleOS: the command line feeds the compiler, code is
MAlloc'd into memory; I/O space unified with memory space), the
compilation spaces with 9P paths (`/n/java/`), the VSL dispatch. If the
brain speaks a different language than the body, the AGI is two systems
glued together. The KV cache being a file system is the body's
philosophy extending into the mind — **one organism, one data plane**:

- the model's memory IS a namespace the OS can serve (export at `/n/kv/`)
- the OS's files ARE data the model can attend to directly (encoded
  into `/kv/in/` via the single encoder)
- downloads, thoughts, diagnostics, plans — all data, all files, all
  in the same space

The AGI's whole world is a file system. The AGI's mind is a file system.
They are the same file system.

## 7. The true definition

All files are data.
All data is encoded.
All encoded data lives in one namespace.
The KV cache IS the file system.

## References

- `research/061-kv-cache-filesystem-7hop.md` — the 7-hop convergence + triple-DA + implementation path
- `docs/wubu-model-blueprint.md` §2.7 — the architecture
- In-tree already: `wubu_kv_tier.c` (3-tier hot/warm/cold), `wubu_lmcache.c` (FNV-1a64 prefix+PD persistence), `wubu_kv_evict.c` (priority eviction), `wubu_mla.c` (latent KV), `wubu_dsa.c` (coarse-to-fine KV indexer), `wubu_kv_cache.c` (paged/ring KV), `wubu_orbits.c` (path-addressed nested-sphere memory), ring attention (1M+ ctx)
- Serving lineage: PagedAttention SOSP'23 · RadixAttention arXiv 2312.07104 · MemGPT arXiv 2310.08560 · Mooncake FAST'25 · LMCache arXiv 2510.09665 · vAttention arXiv 2405.04437 · Infini-attention arXiv 2404.07143 · Gemma 3 arXiv 2503.19786 · MemOS arXiv 2507.03724
