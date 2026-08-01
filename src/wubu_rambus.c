/*
 * wubu_rambus.c — RDRAM-style interleaved KV memory controller (doc "rambus").
 * Interleaved banks + row-buffer banking + RDRAM-cycle cost model. C11.
 */
#include "wubu_rambus.h"
#include <stdlib.h>
#include <string.h>

wubu_rambus_t *wubu_rambus_create(size_t total_bytes, int n_banks,
                                  int row_size, int clk_mhz) {
    if (n_banks < 1) n_banks = 1;
    if (row_size < 1) row_size = 64;
    if (clk_mhz < 1) clk_mhz = 800;

    /* round n_banks up to power of two */
    int p2 = 1;
    while (p2 < n_banks) p2 <<= 1;
    n_banks = p2;
    if (n_banks > WUBU_RB_MAX_BANKS) n_banks = WUBU_RB_MAX_BANKS;

    wubu_rambus_t *r = (wubu_rambus_t *)calloc(1, sizeof(*r));
    if (!r) return NULL;

    size_t per_bank = (total_bytes + n_banks - 1) / n_banks;
    /* align per-bank to row_size so row math is clean */
    per_bank = (per_bank + row_size - 1) & ~((size_t)row_size - 1);
    r->n_banks = n_banks;
    r->bank_bits = 0;
    while ((1 << r->bank_bits) < n_banks) r->bank_bits++;
    r->bytes_per_bank = per_bank;
    r->row_size = row_size;
    r->clk_mhz = clk_mhz;
    r->t_rp = 2; r->t_rcd = 2; r->t_burst = 1;  /* nominal RDRAM-ish cycles */

    r->base = (uint8_t *)malloc(per_bank * n_banks);
    if (!r->base) { free(r); return NULL; }
    memset(r->base, 0, per_bank * n_banks);
    for (int i = 0; i < WUBU_RB_MAX_BANKS; i++) r->open_row[i] = -1;
    return r;
}

void wubu_rambus_free(wubu_rambus_t *r) {
    if (!r) return;
    free(r->base);
    free(r);
}

/* Coordinate -> pointer. We interleave (token, head) across banks:
 *   bank = (token * n_heads + head) & (n_banks-1)
 * Within a bank, offset = (token * n_heads + head) / n_banks * elem_bytes * head_dim
 *                        + dim * elem_bytes
 * This spreads sequential token reads across banks (RDRAM-style streaming). */
uint8_t *wubu_rambus_kv_ptr(wubu_rambus_t *r, int token, int head,
                             int head_dim, int dim, size_t elem_bytes) {
    if (!r || token < 0 || head < 0 || dim < 0) return NULL;
    int n_heads = head_dim > 0 ? head_dim : 1;
    long long idx = (long long)token * n_heads + head;
    int bank = (int)(idx & (r->n_banks - 1));
    long long per_bank_elems = (idx / r->n_banks) * head_dim + dim;
    size_t off = (size_t)per_bank_elems * elem_bytes;
    if (off >= r->bytes_per_bank) return NULL;
    return r->base + (size_t)bank * r->bytes_per_bank + off;
}

void wubu_rambus_access(wubu_rambus_t *r, int token, int head,
                        int head_dim, size_t len) {
    if (!r) return;
    int n_heads = head_dim > 0 ? head_dim : 1;
    long long idx = (long long)token * n_heads + head;
    int bank = (int)(idx & (r->n_banks - 1));
    /* row = which row-buffer line this access falls into */
    long long per_bank_elems = idx / r->n_banks; /* approx; row is by token group */
    int row = (int)((per_bank_elems * head_dim) / (r->row_size / 4 + 1));

    if (r->open_row[bank] == row) {
        r->hits++;
        r->cycle_cost += (uint64_t)r->t_burst;
    } else {
        r->misses++;
        r->cycle_cost += (uint64_t)(r->t_rp + r->t_rcd + r->t_burst);
        r->open_row[bank] = row;
    }
}

double wubu_rambus_eff_bw(const wubu_rambus_t *r, uint64_t bytes_moved) {
    if (!r || r->cycle_cost == 0) return 0.0;
    /* cycles -> seconds via clk, then bytes / seconds */
    double secs = (double)r->cycle_cost / (r->clk_mhz * 1e6);
    return (double)bytes_moved / secs;  /* bytes/sec */
}

void wubu_rambus_stats(const wubu_rambus_t *r,
                       uint64_t *hits, uint64_t *misses, uint64_t *cycles) {
    if (hits)   *hits   = r ? r->hits : 0;
    if (misses) *misses = r ? r->misses : 0;
    if (cycles) *cycles = r ? r->cycle_cost : 0;
}

/* Effective sustained bandwidth in MB/s (convenience wrapper). */
double wubu_rambus_eff_bw_mbps(const wubu_rambus_t *r, uint64_t bytes_moved) {
    return wubu_rambus_eff_bw(r, bytes_moved) / (1024.0 * 1024.0);
}
