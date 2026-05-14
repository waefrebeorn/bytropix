#include "thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// ── Thread pool using OpenMP ───────────────────────────────────
// Non-atomic pattern: each thread writes to its OWN region of output.
// OpenMP #pragma omp parallel for splits loop iterations across threads
// with zero atomics — each iteration writes to non-overlapping output.
// This is the same pattern ggml-cpu uses (non-atomic per-thread accumulation).

struct thread_pool_t {
    int n_threads;
};

thread_pool_t *thread_pool_create(int n_threads) {
    if (n_threads <= 0) {
        n_threads = omp_get_num_procs();
    }
    thread_pool_t *pool = (thread_pool_t *)malloc(sizeof(thread_pool_t));
    if (pool) pool->n_threads = n_threads;
    return pool;
}

void thread_pool_destroy(thread_pool_t *pool) {
    free(pool);
}

int thread_pool_count(thread_pool_t *pool) {
    return pool ? pool->n_threads : 1;
}

void thread_pool_parallel_for(thread_pool_t *pool, int total,
                               void *userdata, range_fn_t fn) {
    if (!pool || total <= 0) return;
    int nt = pool->n_threads;
    if (nt <= 1) {
        fn(userdata, 0, total, 0);
        return;
    }

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        int chunk = total / nt;
        int rem = total % nt;
        int start = tid * chunk + (tid < rem ? tid : rem);
        int end = start + chunk + (tid < rem ? 1 : 0);
        if (start < total) {
            fn(userdata, start, end > total ? total : end, tid);
        }
    }
}

// ── Parallel matmul (non-atomic: each row written by exactly one thread) ──

typedef struct {
    int M, N, K;
    const float *A, *B;
    float *C;
} matmul_nt_data_t;

static void matmul_nt_chunk(void *userdata, int start, int end, int tid) {
    (void)tid;
    matmul_nt_data_t *d = (matmul_nt_data_t *)userdata;
    for (int m = start; m < end; m++) {
        for (int n = 0; n < d->N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < d->K; k++) {
                sum += d->A[m * d->K + k] * d->B[n * d->K + k];
            }
            d->C[m * d->N + n] = sum;
        }
    }
}

void thread_pool_matmul_nt(thread_pool_t *pool,
                            int M, int N, int K,
                            const float *A, const float *B,
                            float *C) {
    matmul_nt_data_t data = {M, N, K, A, B, C};
    thread_pool_parallel_for(pool, M, &data, matmul_nt_chunk);
}

// ── Global pool ───────────────────────────────────────────────

thread_pool_t *thread_pool_global(void) {
    static thread_pool_t *global = NULL;
    if (!global) {
        global = thread_pool_create(0);
    }
    return global;
}
