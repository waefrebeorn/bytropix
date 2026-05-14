#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Simple thread pool for parallelizing CPU ops.
// Non-atomic pattern: each thread writes to its own region of output.
// No locks/atomics in the hot path — just split work and go.

typedef struct thread_pool_t thread_pool_t;

// Create a thread pool with n_threads workers (0 = use all available)
thread_pool_t *thread_pool_create(int n_threads);

// Destroy the pool
void thread_pool_destroy(thread_pool_t *pool);

// Get thread count
int thread_pool_count(thread_pool_t *pool);

// A task function: process range [start, end) with given user data
typedef void (*range_fn_t)(void *userdata, int start, int end, int thread_id);

// Parallel for: split [0, total) into n_threads chunks, call fn on each
void thread_pool_parallel_for(thread_pool_t *pool, int total, void *userdata, range_fn_t fn);

// Parallel matmul: C[M,N] = A[M,K] @ B[K,N]  (B row-major [N,K])
// Splits M rows across threads, each thread writes its own rows -> NO atomics needed
void thread_pool_matmul_nt(thread_pool_t *pool,
                           int M, int N, int K,
                           const float *A, const float *B,
                           float *C);

// Default global pool (lazily created with all available threads)
thread_pool_t *thread_pool_global(void);

#ifdef __cplusplus
}
#endif

#endif // THREAD_POOL_H
