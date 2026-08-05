# ADR-003: The KV Cache Is a File System (Styx/9P Namespace)

- **Status:** accepted
- **Date:** 2026-08-05

## Context

The user directive: "THE KV CACHE IS A FILE SYSTEM." The research
(066-H4, 061) proved every mechanism ships: PagedAttention (blocks/CoW,
SOSP'23), RadixAttention (radix tree = paths), MemGPT (LLM as OS),
Mooncake (disaggregated KV pool), LMCache (tiered persistent chunks),
Infini-attention (compressive writes). Half is already in-tree
(`wubu_kv_tier`, `wubu_lmcache`, `wubu_kv_evict`, `wubu_mla`, `wubu_dsa`,
`wubu_orbits`, paged/ring KV). The missing layer is the address layer.

## Decision

The KV cache lives at `/n/kv/` as a Styx/9P-exportable namespace: a
radix-path address layer (paths → block ranges → tiers), a mount table,
and a single-encoder modality head. All files are data; all inputs are
encoded; memory persists; diagnostics are files. The namespace is the
interface — the body can `ls` the mind.

## Consequences

- **Positive:** uniform inspection (`cat /n/kv/meta/...`), persistence
  across sessions, self-administering memory (pressure interrupts), one
  interface for file, tensor, and cache access.
- **Negative:** 9P protocol + path resolution overhead on the hot path —
  mitigated by paged blocks (copy-on-write, like PagedAttention) and
  direct block-range addressing.
- **Deferred:** G1 (wubu_kvfs namespace), G2 (mount table), G3 (modality
  head), G4 (compressive write-back), G5 (Styx export), G6
  (self-administering loop) — the next implementation wave.

## Verification

`make test_kvfs` green; namespace walkable with a 9P client.
