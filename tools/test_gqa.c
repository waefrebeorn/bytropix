/* test_gqa.c — BRAIN #234: grouped-query heads forward verified.
 *
 * Measured assertions on wubu_gqa_kv_head + wubu_gqa_group_size:
 *   1. MHA (n_kv==n_q): every q maps to itself (group size 1).
 *   2. GQA grouping: n_q=8, n_kv=2 -> groups {0-3}->kv0, {4-7}->kv1.
 *   3. every KV head is used (no orphan KV heads).
 *   4. group size == n_q/n_kv; non-divisible rejected.
 *   5. q out of range / n_kv<=0 rejected.
 */
#include "wubu_grouped_query_heads.h"
#include <stdio.h>

static int g_run = 0, g_pass = 0;
#define T(cond, msg) do { \
    g_run++; \
    if (cond) { g_pass++; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s (line %d)\n", msg, __LINE__); } \
} while (0)

int main(void) {
    printf("=== test_gqa (grouped-query heads, #234) ===\n");

    /* 1: MHA identity. */
    T(wubu_gqa_group_size(8, 8) == 1, "MHA group size == 1");
    int mha = 1;
    for (int q = 0; q < 8; q++) if (wubu_gqa_kv_head(q, 8, 8) != q) mha = 0;
    T(mha, "MHA: every q maps to itself");

    /* 2: GQA n_q=8, n_kv=2. */
    T(wubu_gqa_group_size(8, 2) == 4, "group size == 8/2 == 4");
    int grp = 1;
    for (int q = 0; q < 4; q++) if (wubu_gqa_kv_head(q, 8, 2) != 0) grp = 0;
    for (int q = 4; q < 8; q++) if (wubu_gqa_kv_head(q, 8, 2) != 1) grp = 0;
    T(grp, "GQA 8/2: q0-3->kv0, q4-7->kv1");

    /* 3: every KV head used (no orphan). n_q=12, n_kv=3. */
    int used[3] = {0,0,0};
    for (int q = 0; q < 12; q++) used[wubu_gqa_kv_head(q, 12, 3)] = 1;
    T(used[0] && used[1] && used[2], "every KV head used (12/3)");

    /* 4: non-divisible rejected. */
    T(wubu_gqa_group_size(8, 3) == -1, "non-divisible group size rejected");
    T(wubu_gqa_kv_head(0, 8, 3) == -1, "non-divisible kv_head rejected");

    /* 5: bad args. */
    T(wubu_gqa_kv_head(-1, 8, 2) == -1, "q<0 rejected");
    T(wubu_gqa_kv_head(8, 8, 2) == -1, "q>=n_q rejected");
    T(wubu_gqa_kv_head(0, 8, 0) == -1, "n_kv<=0 rejected");

    printf("=== Results: %d/%d passed ===\n", g_pass, g_run);
    return g_pass == g_run ? 0 : 1;
}
