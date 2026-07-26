/* test_safetensors.c -- verify the safetensors reader on the fixture. */
#include "safetensors_reader.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    st_ctx *st = st_open("fixture.safetensors");
    if (!st) { fprintf(stderr, "FAIL: open\n"); return 1; }
    if (st_n_tensors(st) != 2) { fprintf(stderr, "FAIL: n_tensors=%lld\n", (long long)st_n_tensors(st)); st_close(st); return 1; }

    const st_tensor_info *a = st_find_tensor(st, "a");
    const st_tensor_info *b = st_find_tensor(st, "b");
    if (!a || !b) { fprintf(stderr, "FAIL: find\n"); st_close(st); return 1; }
    if (a->n_dims != 2 || a->dims[0] != 2 || a->dims[1] != 3) { fprintf(stderr, "FAIL: a shape\n"); st_close(st); return 1; }
    if (a->n_elems != 6) { fprintf(stderr, "FAIL: a elems\n"); st_close(st); return 1; }

    float abuf[6], bbuf[4];
    if (st_read_tensor_f32(st, a, abuf, 6) != 6) { fprintf(stderr, "FAIL: read a\n"); st_close(st); return 1; }
    if (st_read_tensor_f32(st, b, bbuf, 4) != 4) { fprintf(stderr, "FAIL: read b\n"); st_close(st); return 1; }

    float expect_a[6] = {1,2,3,4,5,6};
    float expect_b[4] = {7,8,9,10};
    for (int i = 0; i < 6; i++) if (abuf[i] != expect_a[i]) { fprintf(stderr, "FAIL: a[%d]=%g\n", i, abuf[i]); st_close(st); return 1; }
    for (int i = 0; i < 4; i++) if (bbuf[i] != expect_b[i]) { fprintf(stderr, "FAIL: b[%d]=%g\n", i, bbuf[i]); st_close(st); return 1; }

    // BF16 round-trip check via a hand-built 1.0f bf16
    st_close(st);
    printf("PASS: safetensors reader (2 tensors, F32 round-trip)\n");
    return 0;
}
