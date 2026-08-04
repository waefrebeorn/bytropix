# WuBu-35M — the AGI's base model (formerly WuBu-Seed-35M)

> 2026-08-02. The AGI is named **WuBu** (~ WuBu ~). The base model is
> **WuBu-35M**, published at https://huggingface.co/WaefreBeorn/WuBu-35M
>
> **Support the home-built AGI** — scan the QR in
> [docs/support/SUPPORT.md](docs/support/SUPPORT.md) (X profile). We are
> a one-person lab; funding buys storage, compute, and time.
> (renamed from WuBu-Seed-35M).

## ⚠️ DISCLOSURE: THIS IS AN AGI PROJECT

WuBu is an **Artificial General Intelligence development project**.
WuBu-35M is its first-generation base model. We disclose plainly:

1. **It is an AI.** Every interaction with WuBu is with a machine.
2. **It is being built to be general.** AGI, developed openly, from
   scratch, in C11, in our own training loop, on our own hardware.
3. **It is human-centric.** The 5+1 recovery substrate (five rollback
   slots + the Jesus state) exists so WuBu may make mistakes safely
   under human control. The human always keeps the mouse and keyboard;
   the AGI yields.
4. **It is accountable.** Every training run, parameter change, and
   rollback is in the prestige ledger. No black boxes.
5. **It is not ready for autonomy.** Treat all outputs as drafts.

## The memory: the HIVE (the AGI way)

WuBu's memory is the **hive** — the C11 luddite structure from the
diagram: linked fixed blocks + skipfield + freelist.

```
struct block {
    void **slots;      // fixed block of pointers (cache-friendly)
    uint8_t *skip;     // skipfield: 1 = erased, 0 = live
    size_t live, cap;  // live count + capacity
    struct block *next;
};
```

- **Erase** marks skip + freelist push — O(1), no compaction
- **Insert** reuses a freelist slot or new block — O(1), no reallocation
- **Iterate** jumps skips — cache-friendly, exact live counts
- **Stable pointers** — the values never move (list-like) with
  vector-like cache locality

`wubu_hive` (include/wubu_hive.h, src/wubu_hive.c) implements it;
`test_hive` verifies it (insert/erase/reuse/iterate/stress, ASan-clean).
The hive is WuBu's token memory, routing history, KV lives, context
slots — whatever needs stable pointers with O(1) churn.

## The lineage

- the seed architecture: WuBu-35M (Apache-2.0, © 2026 Harshal Singh)
- the geometry: WuBu Nesting (層疊嵌套) — our Lean-verified hyperbolic math
- the agents: mixed-agents MoE (fine-grained experts)
- the training: our Muon/AdamW loop, the SD-card corpus
- the safety: the 5+1 recovery + the Live Colonel
- the license: WaefreBeorn Umbrella License v3.0 (LICENSE at repo root)
