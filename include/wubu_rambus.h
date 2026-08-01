/*
 * wubu_rambus.h — RDRAM-style interleaved KV memory controller (doc "rambus").
 *
 * Real RDRAM (Nintendo 64) was a narrow (9-bit) very-high-clock channel: the
 * win was NOT width, it was INTERLEAVING many banks so that sequential access
 * streamed without dead cycles, plus row-buffer banking (open-row hits are ~0
 * latency). We apply that to the decode KV load, which is the bandwidth wall:
 *
 *   - The KV arena is striped across NBANKS interleaved banks (like RDRAM
 *     channels). A token's K/V vector is spread round-robin across banks so a
 *     full attention row-read streams bank-by-bank with no bank conflict.
 *   - Each bank keeps an "open row" (row-buffer). Accesses to the open row are
 *     hits (cheap); row misses pay a precharge+activate penalty (modeled).
 *   - The "clock" is high (we expose a cycles/byte cost model) so the engine
 *     can pick the interleave factor that maximizes sustained GB/s for its
 *     (seq_len, head_dim) access pattern.
 *
 * This is a *model + allocator*, not a hardware device: it lays KV out in
 * interleaved banks inside one contiguous buffer and reports the simulated
 * latency/cost so the scheduler can choose bank count and interleave. Pure C11.
 */
#ifndef WUBU_RAMBUS_H
#define WUBU_RAMBUS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WUBU_RB_MAX_BANKS 16

typedef struct {
    int    n_banks;
    int    bank_bits;     /* log2(n_banks) */
    size_t bytes_per_bank;/* contiguous bytes per bank slice */
    uint8_t *base;        /* whole arena (n_banks * bytes_per_bank) */
    /* row-buffer state per bank: currently-open row index (-1 = none) */
    int     open_row[WUBU_RB_MAX_BANKS];
    int     row_size;     /* bytes per "row" within a bank (row-buffer granularity) */
    /* stats */
    uint64_t hits, misses;
    uint64_t cycle_cost;  /* accumulated modeled cycles */
    int     t_rp, t_rcd, t_burst; /* precharge, activate, burst (in "RDRAM cycles") */
    int     clk_mhz;      /* simulated channel clock */
} wubu_rambus_t;

/* Create an RDRAM-style KV arena.
 * total_bytes : total KV bytes (rounded up to n_banks * row alignment)
 * n_banks     : interleave factor (2..16, power of two)
 * row_size    : row-buffer granularity in bytes (e.g. 256 = one cache line group)
 * clk_mhz     : simulated channel clock (e.g. 800 for RDRAM RIMM) */
wubu_rambus_t *wubu_rambus_create(size_t total_bytes, int n_banks,
                                  int row_size, int clk_mhz);

void wubu_rambus_free(wubu_rambus_t *r);

/* Map (token, head, dim) KV coordinate to a pointer in the interleaved layout.
 * Layout: token-major, but token+head spread across banks; within a bank the
 * per-bank slice is contiguous. Returns NULL on OOB. */
uint8_t *wubu_rambus_kv_ptr(wubu_rambus_t *r, int token, int head,
                             int head_dim, int dim, size_t elem_bytes);

/* Model a read of `len` bytes starting at the given (token,head,dim). Updates
 * hit/miss stats + cycle cost based on row-buffer state. Does NOT copy — call
 * wubu_rambus_kv_ptr to get the buffer, this just bills the access cost. */
void wubu_rambus_access(wubu_rambus_t *r, int token, int head,
                        int head_dim, size_t len);

/* Effective sustained bandwidth estimate (bytes/cycle * clk) in bytes/sec. */
double wubu_rambus_eff_bw(const wubu_rambus_t *r, uint64_t bytes_moved);

/* Stats: hits, misses, cycle cost. */
void wubu_rambus_stats(const wubu_rambus_t *r,
                       uint64_t *hits, uint64_t *misses, uint64_t *cycles);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_RAMBUS_H */
