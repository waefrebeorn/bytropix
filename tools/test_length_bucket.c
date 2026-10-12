/* test_length_bucket.c — BRAIN #246: length bucket forward verified.
 *
 * Measured assertions on wubu_length_bucket + wubu_length_bucket_pad:
 *   1. exact boundary L == bound -> that bucket.
 *   2. just over a bound -> next bucket.
 *   3. overflow (L > all bounds) -> last bucket, pad == last bound.
 *   4. pad == the bucket's bound (>= L).
 *   5. bucket index monotonic non-decreasing in L.
 *   6. padding waste bounded: pad - L < bucket width.
 *   7. bad args rejected.
 */
#include "wubu_length_bucket.h"
#include <stdio.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_length_bucket (length bucketing, #246) ===\n");
    int bounds[4] = {16, 64, 256, 1024};
    const int k = 4;

    /* 1: exact boundary. */
    T(wubu_length_bucket(64, bounds, k) == 1, "L==64 -> bucket 1");
    T(wubu_length_bucket(16, bounds, k) == 0, "L==16 -> bucket 0");

    /* 2: just over -> next. */
    T(wubu_length_bucket(65, bounds, k) == 2, "L==65 -> bucket 2");
    T(wubu_length_bucket(17, bounds, k) == 1, "L==17 -> bucket 1");

    /* 3: overflow -> last, pad == last bound. */
    T(wubu_length_bucket(5000, bounds, k) == 3, "L>all -> last bucket");
    T(wubu_length_bucket_pad(5000, bounds, k) == 1024, "overflow pad == last bound");

    /* 4: pad == bucket bound, >= L. */
    int pad = wubu_length_bucket_pad(100, bounds, k);
    T(pad == 256 && pad >= 100, "L=100 -> bucket 2, pad 256");

    /* 5: monotonic. */
    int mono = 1, prev = -1;
    for (int L = 1; L <= 1100; L += 7) {
        int b = wubu_length_bucket(L, bounds, k);
        if (b < prev) mono = 0;
        prev = b;
    }
    T(mono, "bucket index monotonic in L");

    /* 6: padding waste < bucket width (pad - L < bound - prev_bound). */
    int waste_ok = 1;
    for (int L = 17; L <= 64; L++) {   /* bucket 1, width = 64-16 = 48 */
        if (wubu_length_bucket_pad(L, bounds, k) - L > 48 - 1) waste_ok = 0;
    }
    T(waste_ok, "padding waste < bucket width");

    /* 7: bad args. */
    T(wubu_length_bucket(10, NULL, k) == -1, "NULL bounds rejected");
    T(wubu_length_bucket(10, bounds, 0) == -1, "k<=0 rejected");
    T(wubu_length_bucket_pad(10, NULL, k) == -1, "pad NULL rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
