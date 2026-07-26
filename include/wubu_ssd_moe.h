/*
 * wubu_ssd_moe.h -- SSD-paged Mixture-of-Experts (ds4-ssd slot-bank).
 *
 * Replicates Anemll ds4-ssd's signature technique (Apple "LLM in a flash"):
 *   - dense/shared tensors stay resident in RAM (loaded by the caller);
 *   - routed MoE expert weights live on disk (SSD) in a sidecar directory;
 *   - a fixed POOL of expert "slots" per layer is kept resident in RAM;
 *   - on a router miss, the selected expert is paged in from disk (pread)
 *     and the least-recently-used slot is evicted.
 * This lets a 256-expert model (e.g. KAT-Coder) run in a fraction of the
 * RAM its full expert footprint would require.
 *
 * Self-contained: only depends on wubu_safetensors_shard for raw reads and
 * the C stdlib. No god headers. C11, opaque ctx.
 *
 * Sidecar layout (one file per layer):
 *   <root>/experts.<LAYER>.bin
 *      packed: [expert 0: gate(up to d_ff*d_model) | up | down] ... [expert N-1]
 *      each expert's three matrices stored BF16 (2 bytes/elem) to keep the
 *      cold footprint small on disk; dequantized to F32 into the slot on page-in.
 *   <root>/manifest.json
 *      { "n_layers":L, "n_experts":E, "d_model":D, "d_ff":F,
 *        "n_active":K, "slot_bank":S, "expert_bytes":B, "fmt":"bf16" }
 */
#ifndef WUBU_SSD_MOE_H
#define WUBU_SSD_MOE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wubu_ssd_moe wubu_ssd_moe_t;

/* Open a sidecar directory. slot_bank = resident expert slots kept per layer
 * (ds4-ssd's --moe-slot-bank; e.g. 8..64). Returns NULL on failure. */
wubu_ssd_moe_t *wubu_ssd_moe_open(const char *sidecar_dir, int slot_bank);

/* Number of layers / experts this sidecar describes. */
int  wubu_ssd_moe_n_layers(const wubu_ssd_moe_t *m);
int  wubu_ssd_moe_n_experts(const wubu_ssd_moe_t *m);
int  wubu_ssd_moe_d_model(const wubu_ssd_moe_t *m);
int  wubu_ssd_moe_d_ff(const wubu_ssd_moe_t *m);

/* Page expert `e` of layer `layer` into a resident slot (evicting LRU on
 * miss) and return an F32 pointer to its three matrices in `out`:
 *   out[0] = gate [d_model, d_ff]
 *   out[1] = up   [d_model, d_ff]
 *   out[2] = down [d_ff, d_model]
 * The pointer is valid until the next call for THIS layer that may evict.
 * Returns 1 on hit (already resident), 0 on a page-in (disk read), -1 on err. */
int wubu_ssd_moe_get(wubu_ssd_moe_t *m, int layer, int expert, float *out[3]);

/* Stats (cumulative): page-ins, slot hits, bytes read from disk. */
void wubu_ssd_moe_stats(const wubu_ssd_moe_t *m, long *pageins, long *hits, long long *bytes_read);

/* Close + free all slots and handles. */
void wubu_ssd_moe_close(wubu_ssd_moe_t *m);

/* ---- Sidecar packer (offline tool helper) ----
 * Append one layer's expert tensors (already F32, layout [mat, expert]) to the
 * sidecar file for `layer`, writing BF16. `gate/up/down` are pointers to the
 * start of each matrix's expert-major data: expert e's gate is at
 * gate + e*(d_model*d_ff). Call once per layer after creating the file. */
void wubu_ssd_moe_pack_layer(const char *sidecar_dir, int layer,
                             int n_experts, int d_model, int d_ff,
                             const float *gate, const float *up, const float *down);

/* Write the manifest.json describing the sidecar. */
void wubu_ssd_moe_write_manifest(const char *sidecar_dir, int n_layers,
                                 int n_experts, int d_model, int d_ff,
                                 int n_active, int slot_bank);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_SSD_MOE_H */
