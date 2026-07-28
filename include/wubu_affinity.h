#ifndef WUBU_AFFINITY_H
#define WUBU_AFFINITY_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Pin current thread to given core list. Returns 0 on success, -1 if unavailable. */
int wubu_affinity_self_pin(const int *cores, int n_cores);

/* Number of online CPUs. */
int wubu_affinity_n_cpus(void);

/* Pin to first half of cores (typical P-core set on hybrid CPUs). Returns count. */
int wubu_affinity_pin_pcores(int *out_pinned, int max_out);

/* NUMA-aware allocation (falls back to aligned malloc). */
void *wubu_numa_alloc(size_t bytes, int node);
void wubu_numa_free(void *p);

/* Hugepage-aligned (2MB) allocation for KV arena. */
void *wubu_huge_alloc(size_t bytes);

#ifdef __cplusplus
}
#endif

#endif /* WUBU_AFFINITY_H */
