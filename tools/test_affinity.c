/* Test: wubu_affinity (Areas J/K — CPU/NUMA pinning + hugepages). */
#include "wubu_affinity.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

int main(void) {
    int n = wubu_affinity_n_cpus();
    printf("online CPUs = %d\n", n);
    assert(n >= 1);

    /* Pin to P-cores (first half). */
    int pinned[64];
    int k = wubu_affinity_pin_pcores(pinned, 64);
    printf("pinned %d P-cores (first core=%d)\n", k, k > 0 ? pinned[0] : -1);
    assert(k >= 0);

    /* NUMA alloc + write + read back. */
    size_t bytes = 1 << 20; /* 1 MB */
    float *buf = (float *)wubu_numa_alloc(bytes, 0);
    assert(buf != NULL);
    for (int i = 0; i < (int)(bytes / sizeof(float)); i += 1023) buf[i] = (float)i;
    assert(buf[0] == 0.0f);
    wubu_numa_free(buf);

    /* Hugepage alloc. */
    void *hp = wubu_huge_alloc(1 << 22);
    printf("hugepage alloc %s\n", hp ? "OK" : "FAILED");
    assert(hp != NULL);
    free(hp);

    printf("ALL AFFINITY TESTS PASSED\n");
    return 0;
}
