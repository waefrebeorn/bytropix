#include "thread_pool.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { int total; volatile int counter; } test_data_t;

static void test_fn(void *userdata, int start, int end, int tid) {
    test_data_t *d = (test_data_t *)userdata;
    for (int i = start; i < end; i++) {
        d->counter++;
    }
}

int main() {
    printf("Creating thread pool...\n");
    thread_pool_t *pool = thread_pool_create(4);
    if (!pool) { printf("FAIL: pool is NULL\n"); return 1; }
    printf("Threads: %d\n", thread_pool_count(pool));

    test_data_t data = {100, 0};
    printf("Running parallel_for...\n");
    thread_pool_parallel_for(pool, 100, &data, test_fn);
    printf("Counter: %d (expected 100)\n", data.counter);

    thread_pool_destroy(pool);
    printf("DONE\n");
    return 0;
}
