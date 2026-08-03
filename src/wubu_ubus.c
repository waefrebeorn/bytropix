/*
 * wubu_ubus.c -- the U-Bus substrate (see wubu_ubus.h + the design doc).
 * Backends: CPU scalar (always), CPU OpenMP (12 threads), GPU cuBLAS
 * (the existing weak-symbol dispatch = a registered backend, present
 * only when the CUDA objects are linked). The selector is the roofline:
 *   t_b = max(flops / gfops_b, bytes / bw_b) + overhead_b
 * and the GPU's xfer cost is waived for weights the weight-cache has
 * seen (the gpu_barun cache makes repeat uploads free -- the selector
 * mirrors that honesty by remembering recent weight pointers).
 */
#include "wubu_ubus.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#if defined(__GNUC__)
#define UBUS_WEAK __attribute__((weak))
#else
#define UBUS_WEAK
#endif
/* the GPU backend (weak: absent on CPU-only links) */
UBUS_WEAK int gpu_barun_init(void);
UBUS_WEAK int gpu_barun_ready(void);
UBUS_WEAK int gpu_barun_matmul(float *y, const float *w, const float *x,
                               int M, int N, int K);
UBUS_WEAK int gpu_barun_matmul_tx(float *y, const float *a, const float *b,
                                  int M, int N, int K);
UBUS_WEAK int gpu_barun_matmul_nt(float *y, const float *w, const float *x,
                                  int M, int N, int K);

#define MAX_BACKENDS 8
#define MAX_RECENT_W 64

typedef struct {
    ubus_cap_t cap;
    void *ctx;
    ubus_matmul_fn fn;
    int present;   /* registered + usable */
} backend_t;

struct ubus {
    backend_t b[MAX_BACKENDS];
    int nb;
    /* the pool: three bump regions */
    float *pool[3];
    size_t pool_cap[3], pool_used[3];
    /* the recent-weight set (mirrors the GPU weight cache) */
    const void *recent_w[MAX_RECENT_W];
    int n_recent;
    double overhead_s[MAX_BACKENDS];
};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---- CPU scalar backend ---- */
static int cpu_scalar_matmul(void *ctx, float *y, const float *a,
                             const float *b, int M, int N, int K, int flags)
{
    (void)ctx;
    int at = (flags & UBUS_AT) != 0;   /* a stored [K,M]: a^T[m,k] = a[k,m] */
    int wt = (flags & UBUS_WT) != 0;   /* b stored [N,K] */
    for (int m = 0; m < M; m++) {
        const float *ar = a + (size_t)m * (at ? M : K);
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *br = b + (size_t)n * (wt ? K : N);
            float acc = 0;
            if (at) {
                /* y[m,n] = sum_k a^T[m,k] b[k,n]: a col (stride M),
                 * b col (stride N) -- both stored row-major [K,*] */
                for (int k = 0; k < K; k++) acc += a[(size_t)k * M + m] * b[(size_t)k * N + n];
            } else if (wt) {
                for (int k = 0; k < K; k++) acc += ar[k] * br[k];
            } else {
                for (int k = 0; k < K; k++) acc += ar[k] * b[(size_t)k * N + n];
            }
            yr[n] = acc;
        }
    }
    return 1;
}

/* ---- CPU OpenMP backend (the seq loop is the parallel dim) ---- */
#ifdef _OPENMP
#include <omp.h>
static int cpu_omp_matmul(void *ctx, float *y, const float *a,
                          const float *b, int M, int N, int K, int flags)
{
    (void)ctx;
    int at = (flags & UBUS_AT) != 0;
    int wt = (flags & UBUS_WT) != 0;
    if (M < 4) return 0;   /* too small: let the scalar path have it */
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float *ar = a + (size_t)m * (at ? M : K);
        float *yr = y + (size_t)m * N;
        for (int n = 0; n < N; n++) {
            const float *br = b + (size_t)n * (wt ? K : N);
            float acc = 0;
            if (at) {
                /* y[m,n] = sum_k a^T[m,k] b[k,n]: a col (stride M),
                 * b col (stride N) -- both stored row-major [K,*] */
                for (int k = 0; k < K; k++) acc += a[(size_t)k * M + m] * b[(size_t)k * N + n];
            } else if (wt) {
                for (int k = 0; k < K; k++) acc += ar[k] * br[k];
            } else {
                for (int k = 0; k < K; k++) acc += ar[k] * b[(size_t)k * N + n];
            }
            yr[n] = acc;
        }
    }
    return 1;
}
#endif

/* ---- GPU backend (the gpu_barun dispatch) ---- */
static int gpu_matmul(void *ctx, float *y, const float *a,
                      const float *b, int M, int N, int K, int flags)
{
    (void)ctx;
    if (!gpu_barun_ready || !gpu_barun_ready()) return 0;
    int at = (flags & UBUS_AT) != 0;
    int wt = (flags & UBUS_WT) != 0;
    if (at) return gpu_barun_matmul_tx ? gpu_barun_matmul_tx(y, a, b, M, N, K) : 0;
    if (wt) return gpu_barun_matmul ? gpu_barun_matmul(y, b, a, M, N, K) : 0;
    return gpu_barun_matmul_nt ? gpu_barun_matmul_nt(y, b, a, M, N, K) : 0;
}

/* ---- the bus ---- */
ubus_t *ubus_init(void)
{
    ubus_t *u = calloc(1, sizeof(*u));
    if (!u) return NULL;
    /* the CPU scalar backend: always */
    {
        ubus_cap_t c = { "cpu-scalar", 3.0f, 30.0f, 0, 0 };
        u->b[u->nb].cap = c; u->b[u->nb].fn = cpu_scalar_matmul;
        u->b[u->nb].present = 1; u->nb++;
    }
#ifdef _OPENMP
    {
        ubus_cap_t c = { "cpu-omp", 60.0f, 40.0f, 0, 0 };
        u->b[u->nb].cap = c; u->b[u->nb].fn = cpu_omp_matmul;
        u->b[u->nb].present = 1; u->nb++;
    }
#endif
    if (gpu_barun_init && gpu_barun_init()) {
        ubus_cap_t c = { "gpu-cublas", 1800.0f, 250.0f, 12.0f, (size_t)6u << 30 };
        u->b[u->nb].cap = c; u->b[u->nb].fn = gpu_matmul;
        u->b[u->nb].present = 1; u->nb++;
    }
    ubus_measure(u);
    return u;
}

void ubus_free(ubus_t *u)
{
    if (!u) return;
    for (int r = 0; r < 3; r++) free(u->pool[r]);
    free(u);
}

int ubus_register(ubus_t *u, const ubus_cap_t *cap, void *ctx, ubus_matmul_fn fn)
{
    if (!u || u->nb >= MAX_BACKENDS) return -1;
    u->b[u->nb].cap = *cap; u->b[u->nb].ctx = ctx; u->b[u->nb].fn = fn;
    u->b[u->nb].present = 1;
    return u->nb++;
}

int ubus_backend_count(const ubus_t *u) { return u ? u->nb : 0; }

/* ---- the roofline selector ---- */
static int ubus_pick(ubus_t *u, long flops, long bytes, const void *w)
{
    /* the weight-cache honesty: recently seen weights transfer for free */
    int w_cached = 0;
    for (int i = 0; i < u->n_recent; i++)
        if (u->recent_w[i] == w) { w_cached = 1; break; }
    int best = -1;
    double best_t = 1e30;
    for (int i = 0; i < u->nb; i++) {
        if (!u->b[i].present) continue;
        double t = (double)flops / (u->b[i].cap.gfops * 1e9);
        double bw = u->b[i].cap.mem_bw_gbs;
        if (u->b[i].cap.xfer_bw_gbs > 0 && !w_cached) {
            /* the xfer path is the slowest link host<->device */
            double xfer = (double)bytes / (u->b[i].cap.xfer_bw_gbs * 1e9);
            t = fmax(t, xfer);
        } else {
            double mem = (double)bytes / (bw * 1e9);
            t = fmax(t, mem);
        }
        t += u->overhead_s[i];
        if (t < best_t) { best_t = t; best = i; }
    }
    if (best >= 0 && u->b[best].cap.xfer_bw_gbs > 0) {
        u->recent_w[u->n_recent % MAX_RECENT_W] = w;
        u->n_recent++;
    }
    return best;
}

int ubus_matmul(ubus_t *u, float *y, const float *a, const float *b,
                int M, int N, int K, int flags)
{
    if (!u || !y || !a || !b) return 0;
    long flops = 2L * M * N * K;
    long bytes = ((long)M * K + (long)K * N + (long)M * N) * 4L;
    int bid = ubus_pick(u, flops, bytes, flags & UBUS_AT ? a : b);
    if (bid < 0) return 0;
    return u->b[bid].fn(u->b[bid].ctx, y, a, b, M, N, K, flags);
}

int ubus_matmul_backend(ubus_t *u, int bid, float *y, const float *a,
                        const float *b, int M, int N, int K, int flags)
{
    if (!u || bid < 0 || bid >= u->nb || !u->b[bid].present) return 0;
    return u->b[bid].fn(u->b[bid].ctx, y, a, b, M, N, K, flags);
}

/* ---- the pool ---- */
float *ubus_alloc(ubus_t *u, size_t bytes, int region)
{
    if (!u || region < 0 || region > 2) return NULL;
    size_t need = (u->pool_used[region] + bytes + 15) & ~(size_t)15;
    if (need > u->pool_cap[region]) {
        size_t cap = u->pool_cap[region] ? u->pool_cap[region] * 2 : 1 << 20;
        while (cap < need) cap *= 2;
        float *np = realloc(u->pool[region], cap);
        if (!np) return NULL;
        u->pool[region] = np; u->pool_cap[region] = cap;
    }
    float *p = u->pool[region] + u->pool_used[region] / sizeof(float);
    u->pool_used[region] = need;
    return p;
}

/* ---- the bus meter: time a GEMM per backend, print the roofline ---- */
int ubus_measure(ubus_t *u)
{
    if (!u) return 0;
    enum { M = 512, N = 512, K = 512 };
    float *a = malloc((size_t)M * K * 4), *b = malloc((size_t)K * N * 4);
    float *y = malloc((size_t)M * N * 4);
    if (!a || !b || !y) { free(a); free(b); free(y); return 0; }
    for (int i = 0; i < M * K; i++) a[i] = (float)(i % 7) * 0.01f;
    for (int i = 0; i < K * N; i++) b[i] = (float)(i % 5) * 0.01f;
    for (int i = 0; i < u->nb; i++) {
        if (!u->b[i].present) continue;
        /* warm + time */
        u->b[i].fn(u->b[i].ctx, y, a, b, M, N, K, 0);
        int reps = 10;
        double t0 = now_s();
        for (int r = 0; r < reps; r++)
            if (!u->b[i].fn(u->b[i].ctx, y, a, b, M, N, K, 0)) break;
        double dt = (now_s() - t0) / reps;
        double gf = 2.0 * M * N * K * 1e-9;
        if (dt > 1e-6) u->b[i].cap.gfops = (float)(gf / dt);
        u->overhead_s[i] = dt * 0.5;   /* launch + ramp allowance */
    }
    free(a); free(b); free(y);
    return 1;
}

void ubus_report(const ubus_t *u)
{
    if (!u) return;
    printf("U-Bus backends:\n");
    for (int i = 0; i < u->nb; i++)
        if (u->b[i].present)
            printf("  [%d] %-12s gfops=%.0f  mem=%.0fGB/s  xfer=%.0fGB/s%s\n",
                   i, u->b[i].cap.name, u->b[i].cap.gfops, u->b[i].cap.mem_bw_gbs,
                   u->b[i].cap.xfer_bw_gbs,
                   u->b[i].cap.xfer_bw_gbs > 0 ? " (device)" : "");
}
