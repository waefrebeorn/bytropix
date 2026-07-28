# ds4-ssd slot-bank (SSD-paged MoE)

Technique: keep dense/shared/router tensors resident in RAM; page routed MoE
expert weights from an SSD sidecar into a fixed pool of resident slots per
layer, evicting least-recently-used on miss. Based on the Anemll `ds4-ssd`
approach ("LLM in a flash" applied to MoE). This lets a 256-expert model run
in a fraction of the RAM its full expert footprint would require.

## Why

A 256-expert MoE layer with D=2048, F=512 stores each expert's
`[gate, up, down]` as `3 · D · F` floats. Per layer that is
`3 · 2048 · 512 · 256 · 4 bytes ≈ 3.2 GB`. For 40 layers ≈ 128 GB — impossible
to keep resident. Only `K` (top-k) experts are active per token, and across a
decode step the active set is a small, cacheable subset. A slot-bank of `N`
resident experts per layer (e.g. N=8) needs `3 · 2048 · 512 · 8 · 4 ≈ 100 MB`
per layer, ~4 GB total — fits in RAM while the cold 248 experts live on disk.

## Sidecar layout

One file per layer:

```
<root>/experts.<L>.bin
```

Packed BF16, expert-major, each expert = `gate | up | down` contiguous:

```
[ expert 0: gate(D·F) | up(D·F) | down(F·D) ]   <- BF16 (2 bytes/elem)
[ expert 1: ... ]
...
[ expert E-1: ... ]
```

Expert `e` begins at byte offset `e · (3 · D · F · 2)`.

```
<root>/manifest.json
{ "n_layers", "n_experts", "d_model", "d_ff", "n_active", "slot_bank", "fmt":"bf16" }
```

The dense/shared/router tensors stay in the base checkpoint (loaded resident by
the engine) and are NOT duplicated into the sidecar.

## API (`include/wubu_ssd_moe.h`)

```c
wubu_ssd_moe_t *wubu_ssd_moe_open(const char *dir, int slot_bank);
int wubu_ssd_moe_get(m, layer, expert, float *out[3]);   // out = {gate, up, down}
void wubu_ssd_moe_stats(m, &pageins, &hits, &bytes_read);
void wubu_ssd_moe_close(m);

/* packer (offline) */
void wubu_ssd_moe_pack_layer(dir, layer, E, D, F, gate, up, down);
void wubu_ssd_moe_write_manifest(dir, n_layers, E, D, F, n_active, slot_bank);
```

`wubu_ssd_moe_get` returns `1` on slot hit, `0` on page-in (BF16→F32 dequant
from the checkpoint shards into the evicted slot), `-1` on error.

## No-sidecar design (bloat killed)

Experts are **paged directly from the source checkpoint shards** the model
already loads — NOT a redundant sidecar copy. The old design duplicated every
KAT expert weight into a `experts.<L>.bin` sidecar (~256 GB, 3.7× the 68 GB
checkpoint itself, 100% redundant). The checkpoint is already mmap'd / readable
via `wubu_shard_raw`, so `read_expert_mat()` reads each expert's BF16
gate/up/down tensors straight from disk. Zero extra disk footprint, zero weight
duplication. `wubu_ssd_moe_open(checkpoint_dir, slots)` opens the bank over the
checkpoint; no `pack_kat_sidecar` step, no `KAT_SIDECAR` env, no manifest.

## Forward integration

`wubu_moe_forward_ssd` (`src/wubu_moe.c`) mirrors `wubu_moe_forward` but, for
each selected expert, calls `wubu_ssd_moe_get(layer, e, out)` instead of
indexing an in-RAM blob. Router and shared expert remain in-RAM (from
`moe_weights_t`).

## Build / verify

```bash
make test_ssd_moe              # synthetic: 12 experts, 3-slot bank, exact matmul (no sidecar)
make test_kat_decode_bank     # REAL KAT-Coder: pages 256 experts/layer from the checkpoint
./test_kat_decode_bank /home/wubu/models/KAT-Coder-V2.5-Dev 16
```

`tools/test_kat_decode_bank.c` pages each real KAT expert from the checkpoint
shards and confirms the paged BF16 expert equals finite, correct weights — with
no sidecar file present on disk.

## Memory discipline

The reader must never `mmap`/scan the whole checkpoint. Tensor access is by
computed byte offset + `pread` of only the needed tensor. Peak RAM is
O(one shard header + one expert), independent of model size. This matters on
hosts with limited RAM (the build box has ~13 GB; the full KAT checkpoint is
~22 GB).
