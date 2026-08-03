/*
 * wubu_ubus.h -- the U-Bus: an AGI LLM substrate designed like an N64
 * RAM bus. One pool (cartridge weights / RDRAM activations / optimizer
 * state), a bus meter (measure, don't guess), and agnostic ops whose
 * backends (CPU scalar, CPU OpenMP, GPU cuBLAS, ...) are REGISTERED
 * capabilities. The op dispatch is a pure roofline function over the
 * measured table: t_b = max(flops/gfops_b, bytes/bw_b) + overhead_b.
 *
 * See research/AGI_LLM_AGNOSTIC_ARCH.md for the full design.
 */
#ifndef WUBU_UBUS_H
#define WUBU_UBUS_H

#include <stddef.h>

/* the pool regions (the unified address space) */
enum { UBUS_CART = 0,   /* the weights: the read-mostly cartridge */
       UBUS_RDRAM = 1,  /* KV + activations: the working memory */
       UBUS_OPT = 2 };  /* the optimizer state */

/* a registered backend's capability table (measured, not spec) */
typedef struct {
    const char *name;
    float gfops;          /* sustained fp32 GEMM, GFLOPS */
    float mem_bw_gbs;     /* the bandwidth the backend sees */
    float xfer_bw_gbs;    /* host<->backend transfer (0 = unified memory) */
    size_t resident_bytes;/* how much it can hold (0 = unlimited) */
} ubus_cap_t;

typedef struct ubus ubus_t;

/* the matmul backend signature (the single op shape, with flags):
 *   y[M,N] = a[M,K] @ b[K,N]
 * flags:
 *   UBUS_WT  b is stored [N,K] (the forward mm: use b^T)     -> b's storage [N,K]
 *   UBUS_AT  a is stored [K,M] (the backward tx: use a^T)     -> a's storage [K,M]
 * Neither flag = the straight GEMM (b stored [K,N], a stored [M,K]).
 */
enum { UBUS_WT = 1, UBUS_AT = 2 };
typedef int (*ubus_matmul_fn)(void *ctx, float *y, const float *a,
                              const float *b, int M, int N, int K, int flags);

/* ---- the bus ---- */
ubus_t *ubus_init(void);   /* registers the CPU backends + measures the bus */
void ubus_free(ubus_t *u);

/* register another backend (GPU, NPU, ...); returns the backend id */
int ubus_register(ubus_t *u, const ubus_cap_t *cap, void *ctx, ubus_matmul_fn fn);

/* the agnostic op: roofline-selects the backend */
int ubus_matmul(ubus_t *u, float *y, const float *a, const float *b,
                int M, int N, int K, int flags);
/* force a specific backend (tests / A/B) */
int ubus_matmul_backend(ubus_t *u, int backend_id, float *y, const float *a,
                        const float *b, int M, int N, int K, int flags);

/* the pool: bump-allocated flat regions */
float *ubus_alloc(ubus_t *u, size_t bytes, int region);

/* the meter + report */
int ubus_measure(ubus_t *u);     /* re-measure the bus bandwidths */
void ubus_report(const ubus_t *u); /* print the capability table */
int ubus_backend_count(const ubus_t *u);

#endif
