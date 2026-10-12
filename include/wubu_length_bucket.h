#ifndef WUBU_LENGTH_BUCKET_H
#define WUBU_LENGTH_BUCKET_H
/* wubu_length_bucket.h — length bucketing (BRAIN #246).
 *
 * Groups sequences into length buckets to minimize padding waste in batched
 * training. Given sorted ascending bucket upper-bounds `bounds` (k of them),
 * a sequence of length L falls in the first bucket whose bound >= L; sequences
 * longer than all bounds go in the last bucket. The bucket's padded length is
 * that bound. The forward (wubu_length_bucket) is covered by test_length_bucket
 * with measured assertions.
 */
#include <stddef.h>

/* Return the bucket index for length L given k ascending bounds. Returns -1 on
 * bad args (k<=0, NULL). L<=0 -> bucket 0. */
int wubu_length_bucket(int L, const int *bounds, int k);

/* The padded (bucket) length for L: the upper bound of its bucket. Returns -1
 * on bad args. */
int wubu_length_bucket_pad(int L, const int *bounds, int k);

#endif
